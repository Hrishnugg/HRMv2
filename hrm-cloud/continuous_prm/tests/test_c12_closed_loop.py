import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import continuous_prm_dynamics as D
import continuous_prm_spacetime as ST
import continuous_prm_c12_closed_loop as CL
import continuous_prm_c12_latent_dynamics as C12


def _toy():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    adj = [[(2, 1.0)], [(2, 1.0)], [(0, 1.0), (1, 1.0)]]
    return points, adj


def test_normalized_perfect_prediction_matches_exact_future_table():
    episode, _ = C12.make_fixture_pair("direction_alias")
    t = episode.alias_time
    horizon = 8
    observation = episode.observe(t)
    future = episode.future(t, horizon)
    target_displacements = (
        future["centers"][1:] - future["centers"][0:1]
    ) / episode.world.side_len
    converted = CL.normalized_prediction_to_tabulated(
        current_centers=observation.centers,
        normalized_radii=observation.radii,
        predicted_displacements=target_displacements,
        predicted_gate_open=future["gate_open"][1:],
        current_gate_open=future["gate_open"][0],
        identity_mask=observation.identity_mask,
        gate_mask=np.ones(future["gate_open"].shape[1], dtype=np.uint8),
        gate_edges=episode.dynamics.gates.edge_ids,
        side_len=episode.world.side_len,
        dt=episode.dt,
    )
    exact = CL.exact_future_forecast(episode, t, horizon)
    assert np.allclose(converted.centers, exact.centers)
    assert np.allclose(converted.radii, exact.radii)
    assert np.array_equal(converted.gate_open, exact.gate_open)
    assert converted.gate_edges == exact.gate_edges


def test_deliberately_wrong_direction_prediction_changes_first_action():
    left, right = C12.make_fixture_pair("direction_alias")
    t = left.alias_time
    horizon = left.dynamics.cfg.forecast_horizon
    observation = left.observe(t)

    def converted(source):
        future = source.future(t, horizon)
        return CL.normalized_prediction_to_tabulated(
            observation.centers,
            observation.radii,
            (future["centers"][1:] - future["centers"][0:1]) / left.world.side_len,
            future["gate_open"][1:],
            future["gate_open"][0],
            observation.identity_mask,
            np.ones(future["gate_open"].shape[1], dtype=np.uint8),
            left.dynamics.gates.edge_ids,
            left.world.side_len,
            left.dt,
        )

    correct = CL.plan_episode_step(left, left.start_node, converted(left)).first_action
    wrong = CL.plan_episode_step(left, left.start_node, converted(right)).first_action
    assert correct != wrong


def test_tabulated_exact_c8_centers_match_analytic_feasibility():
    circle = D.MovingCircle(1.0, -1.0, 1.0, 1.0, period=4.0, radius=0.25)
    analytic = D.Dynamics([circle])
    table = CL.TabulatedDynamics.from_c8(analytic, start_time=0.0, horizon=8, dt=1.0)

    node = np.array([1.0, 0.0])
    a = np.array([0.0, 0.0])
    b = np.array([2.0, 0.0])
    for t0, t1 in ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)):
        assert table.node_free(node, t0, t1, samples=17) == analytic.node_free(
            node, t0, t1, samples=17
        )
        assert table.edge_free(a, b, t0, t1, samples=33) == analytic.edge_free(
            a, b, t0, t1, samples=33
        )


def test_predicted_gate_masks_only_intended_edge_and_step():
    centers = np.zeros((6, 0, 2), dtype=np.float64)
    gates = np.ones((6, 1), dtype=np.bool_)
    gates[2:4, 0] = False
    dyn = CL.TabulatedDynamics(
        centers=centers,
        radii=np.zeros(0),
        gate_open=gates,
        gate_edges=((0, 2),),
        dt=1.0,
    )
    assert dyn.gate_edge_valid((0, 2), 0, 1)
    assert not dyn.gate_edge_valid((2, 0), 1, 2)
    assert dyn.gate_edge_valid((1, 2), 1, 3)


def test_path_reconstructing_astar_matches_c8_arrival_and_expansions():
    points, adj = _toy()
    circle = D.MovingCircle(1.6, 0.0, 1.6, 4.0, period=2.0, radius=0.4)
    analytic = D.Dynamics([circle])
    table = CL.TabulatedDynamics.from_c8(analytic, 0.0, horizon=40, dt=1.0)
    h = np.zeros((3, 41), dtype=np.float64)

    expected = ST.space_time_astar_prm(
        adj, points, analytic, h, budget=5000, v_agent=1.0, dt=1.0, t_max=40
    )
    actual = CL.predicted_space_time_astar(
        adj, points, table, h, budget=5000, v_agent=1.0, dt=1.0, t_max=40
    )
    assert actual.found == expected["found"]
    assert actual.arrival == expected["arrival"]
    assert actual.expansions == expected["expansions"]
    assert actual.path_states[0] == (0, 0)
    assert actual.path_states[-1][0] == 1


def test_wait_is_first_action_when_corridor_will_clear():
    points, adj = _toy()
    circle = D.MovingCircle(1.6, 0.0, 1.6, 4.0, period=2.0, radius=0.4)
    table = CL.TabulatedDynamics.from_c8(D.Dynamics([circle]), 0.0, 40, 1.0)
    plan = CL.predicted_space_time_astar(
        adj,
        points,
        table,
        np.zeros((3, 41)),
        budget=5000,
        v_agent=1.0,
        dt=1.0,
        t_max=40,
    )
    assert plan.found
    assert any(action.kind == "wait" for action in plan.actions)


def test_returned_first_edge_is_forecast_valid():
    episode, _ = C12.make_fixture_pair("direction_alias")
    forecast = CL.exact_future_forecast(episode, episode.alias_time, 32)
    plan = CL.plan_episode_step(episode, episode.start_node, forecast)
    assert plan.found
    action = plan.first_action
    assert action is not None
    if action.kind == "edge":
        assert forecast.gate_edge_valid((action.source, action.target), 0, action.duration)
        assert forecast.edge_free(
            episode.roadmap.points[action.source],
            episode.roadmap.points[action.target],
            0,
            action.duration,
            samples=max(8, 4 * action.duration),
        )


class RecordingProvider:
    name = "recording_exact"

    def __init__(self):
        self.observation_times = []
        self.forecast_times = []

    def reset(self, episode):
        self.observation_times.clear()
        self.forecast_times.clear()

    def observe(self, episode, t, observation):
        self.observation_times.append(int(t))

    def forecast(self, episode, t, horizon):
        self.forecast_times.append(int(t))
        return CL.exact_future_forecast(episode, t, horizon)


def test_edge_traversal_updates_observations_but_replans_only_at_nodes():
    _, episode = C12.make_fixture_pair("direction_alias")
    provider = RecordingProvider()
    result = CL.run_closed_loop_episode(episode, provider)
    assert result.success
    assert result.observation_updates > result.replans
    assert provider.forecast_times == result.decision_times
    assert len(provider.observation_times) == result.observation_updates


class FrozenProvider(RecordingProvider):
    name = "frozen"

    def forecast(self, episode, t, horizon):
        self.forecast_times.append(int(t))
        current = episode.future(t, 0)
        centers = np.repeat(current["centers"], horizon + 1, axis=0)
        gates = np.repeat(current["gate_open"], horizon + 1, axis=0)
        return CL.TabulatedDynamics(
            centers,
            current["radii"],
            gates,
            tuple(map(tuple, current["gate_edges"])),
            dt=episode.dt,
        )


def test_true_scoring_detects_collision_missed_by_bad_forecast():
    episode, _ = C12.make_fixture_pair("direction_alias")
    result = CL.run_closed_loop_episode(episode, FrozenProvider())
    assert not result.success
    assert result.failure_reason == "collision"
    assert result.collisions == 1


def test_episode_termination_distinguishes_goal_horizon_and_no_plan():
    _, safe_episode = C12.make_fixture_pair("direction_alias")
    goal = CL.run_closed_loop_episode(safe_episode, RecordingProvider())
    assert goal.failure_reason == "goal" and goal.success

    horizon = CL.run_closed_loop_episode(
        safe_episode, RecordingProvider(), max_steps=safe_episode.alias_time + 1
    )
    assert horizon.failure_reason == "horizon" and not horizon.success

    class NoPlan(RecordingProvider):
        name = "no_plan"

        def forecast(self, episode, t, horizon):
            self.forecast_times.append(int(t))
            exact = CL.exact_future_forecast(episode, t, horizon)
            exact.gate_open[:] = False
            return exact

    no_plan = CL.run_closed_loop_episode(safe_episode, NoPlan())
    assert no_plan.failure_reason == "no_plan" and not no_plan.success
