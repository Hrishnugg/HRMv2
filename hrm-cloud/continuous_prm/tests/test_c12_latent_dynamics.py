import sys
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import continuous_prm_c12_latent_dynamics as C12


def _cfg() -> C12.C12DynamicsConfig:
    return C12.C12DynamicsConfig(raster_size=16)


def test_split_seed_namespaces_are_disjoint_and_deterministic():
    seen = set()
    for split in C12.SPLITS:
        values = {
            C12.component_seed(split, "direction_alias", i, component)
            for i in range(20)
            for component in ("map", "goal", "regime")
        }
        assert not (seen & values)
        seen |= values
        assert values == {
            C12.component_seed(split, "direction_alias", i, component)
            for i in range(20)
            for component in ("map", "goal", "regime")
        }


def test_evaluation_conditions_are_deterministic_and_training_stays_id():
    assert C12.evaluation_condition("TRAIN", 1, "C_dyn_rooms") == "development_id"
    assert C12.evaluation_condition("TEST", 0, "C_dyn_rooms") == "matched_id"
    assert C12.evaluation_condition("TEST", 1, "C_dyn_rooms") == "long_dwell_ood"
    assert (
        C12.evaluation_condition("TEST", 2, "C_dyn_rooms")
        == "heldout_phase_direction"
    )
    assert C12.evaluation_condition("TEST", 3, "C_dyn_rooms_large") == "scale_ood"


def test_long_dwell_and_heldout_phase_slices_change_only_eval_regime_contract():
    cfg = C12.C12DynamicsConfig()
    long_episode, _ = C12.make_fixture_pair(
        "slow_gate_phase", pair_index=1, cfg=cfg, split="SMOKE"
    )
    heldout_left, heldout_right = C12.make_fixture_pair(
        "slow_gate_phase", pair_index=2, cfg=cfg, split="SMOKE"
    )
    assert long_episode.schedule.slow_dwell >= int(1.5 * cfg.slow_dwell_max)
    assert long_episode.diagnostics["is_long_dwell_ood"] is True
    assert heldout_left.regime.gate_phase < heldout_right.regime.gate_phase
    assert heldout_left.diagnostics["is_heldout_combo"] is True


def test_slow_gate_alias_has_same_present_but_different_time_to_transition():
    left, right = C12.make_fixture_pair("slow_gate_phase")
    assert left.schedule.hazard_start != right.schedule.hazard_start
    left_now = C12.serialize_observation(left.observe(left.alias_time))
    right_now = C12.serialize_observation(right.observe(right.alias_time))
    assert np.allclose(left_now, right_now, atol=C12.ALIAS_ATOL, rtol=0.0)
    audit = C12.audit_alias_pair(left, right)
    assert audit["is_alias"] is True


@pytest.mark.parametrize("stratum", ["slow_gate_phase", "route_mode_junction"])
def test_persistent_challenges_hide_cue_outside_exact_transformer_window(stratum):
    left, right = C12.make_fixture_pair(stratum)
    assert not np.allclose(
        C12.serialize_observation(left.observe(0)),
        C12.serialize_observation(right.observe(0)),
        atol=C12.ALIAS_ATOL,
        rtol=0.0,
    )
    # With alias_time=16 and a 16-frame online window, the flat Transformer
    # retains exactly frames 1..16.  Those frames are deliberately aliased.
    for t in range(1, left.alias_time + 1):
        assert np.allclose(
            C12.serialize_observation(left.observe(t)),
            C12.serialize_observation(right.observe(t)),
            atol=C12.ALIAS_ATOL,
            rtol=0.0,
        )


@pytest.mark.parametrize(
    "stratum",
    ["direction_alias", "slow_gate_phase", "route_mode_junction"],
)
def test_constructed_alias_pairs_match_now_and_diverge_later(stratum):
    left, right = C12.make_fixture_pair(stratum, pair_index=3, cfg=_cfg())
    t = left.alias_time

    left_now = C12.serialize_observation(left.observe(t))
    right_now = C12.serialize_observation(right.observe(t))
    assert np.allclose(left_now, right_now, atol=1e-6, rtol=0.0)
    assert C12.present_observation_key(left.observe(t)) == C12.present_observation_key(
        right.observe(t)
    )

    left_future = left.future_occupancy(t, horizon=8)
    right_future = right.future_occupancy(t, horizon=8)
    assert not np.array_equal(left_future, right_future)
    assert left.oracle_first_action_hint != right.oracle_first_action_hint
    audit = C12.audit_alias_pair(left, right)
    assert audit["current_match"]
    assert audit["future_diverges"]
    assert audit["first_action_diverges"]
    assert audit["is_alias"]


def test_fixture_generation_is_bitwise_deterministic_and_periods_are_in_range():
    cfg = _cfg()
    a0, a1 = C12.make_fixture_pair("direction_alias", pair_index=17, cfg=cfg)
    b0, b1 = C12.make_fixture_pair("direction_alias", pair_index=17, cfg=cfg)

    for a, b in ((a0, b0), (a1, b1)):
        assert a.schedule == b.schedule
        assert np.array_equal(a.dynamics.centers, b.dynamics.centers)
        assert np.array_equal(a.dynamics.gate_open, b.dynamics.gate_open)
        assert cfg.fast_period_min <= a.schedule.fast_period <= cfg.fast_period_max
        assert cfg.slow_dwell_min <= a.schedule.slow_dwell <= cfg.slow_dwell_max


def test_present_sufficient_exposes_current_context_but_challenges_do_not():
    hidden, _ = C12.make_fixture_pair("direction_alias", pair_index=5, cfg=_cfg())
    visible, counterfactual = C12.make_fixture_pair("present_sufficient", pair_index=5, cfg=_cfg())

    hidden_payload = hidden.observe(hidden.alias_time).model_payload()
    visible_payload = visible.observe(visible.alias_time).model_payload()
    counter_payload = counterfactual.observe(counterfactual.alias_time).model_payload()

    assert "visible_regime_context" not in hidden_payload
    assert "visible_regime_context" in visible_payload
    assert not np.array_equal(
        visible_payload["visible_regime_context"],
        counter_payload["visible_regime_context"],
    )


def test_observation_schema_rejects_latent_leakage():
    episode, _ = C12.make_fixture_pair("route_mode_junction", pair_index=2, cfg=_cfg())
    payload = episode.observe(episode.alias_time).model_payload()
    C12.audit_model_payload(payload)

    forbidden = (
        "latent_regime",
        "phase_counter",
        "velocity",
        "future_waypoints",
        "future_occupancy",
    )
    for key in forbidden:
        bad = dict(payload)
        bad[key] = np.zeros(1, dtype=np.float32)
        with pytest.raises(ValueError, match="forbidden"):
            C12.audit_model_payload(bad)


def test_counterfactuals_share_map_and_goal_but_not_regime_seed():
    a, b = C12.make_fixture_pair("slow_gate_phase", pair_index=9, cfg=_cfg())
    assert a.map_seed == b.map_seed
    assert a.goal_seed == b.goal_seed
    assert a.regime_seed != b.regime_seed
    assert np.array_equal(a.roadmap.points, b.roadmap.points)
    assert np.array_equal(a.world.goal, b.world.goal)
    assert {a.regime.variant, b.regime.variant} == {0, 1}


def test_gate_schedule_masks_only_registered_edge_and_time():
    episode, _ = C12.make_fixture_pair("slow_gate_phase", pair_index=0, cfg=_cfg())
    edge = episode.schedule.hazard_edge
    other = episode.schedule.safe_edge
    t = episode.schedule.hazard_start

    assert not episode.dynamics.gate_edge_valid(edge, t, t + 1)
    assert episode.dynamics.gate_edge_valid(other, t, t + 1)
    assert episode.dynamics.gate_edge_valid(edge, 2, 3)
    assert episode.dynamics.gate_edge_valid(
        edge, episode.schedule.hazard_end + 1, episode.schedule.hazard_end + 2
    )


def test_reset_clears_mutable_cursor_and_cached_observation():
    episode, _ = C12.make_fixture_pair("direction_alias", pair_index=1, cfg=_cfg())
    sim = episode.dynamics
    sim.advance_to(episode.alias_time)
    _ = sim.observe(episode.alias_time, episode.start_node, episode.goal_node)
    assert sim.cursor == episode.alias_time
    assert sim.last_observation is not None

    state = sim.reset()
    assert state.t == 0
    assert sim.cursor == 0
    assert sim.last_observation is None
