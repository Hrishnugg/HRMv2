"""C12-A tabulated forecasts and path-reconstructing closed-loop planner."""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

import continuous_prm_spacetime as ST
from continuous_prm_c12_latent_dynamics import C12EpisodeSpec, C12Observation, EdgeId, canonical_edge


@dataclass
class TabulatedDynamics:
    """Predicted circle centers and gates on a fixed relative-time table.

    ``centers[k]`` and ``gate_open[k]`` describe relative forecast step
    ``k``. Geometry methods accept physical time; ``dt`` converts it back to
    table coordinates. Linear interpolation preserves C8 triangle-wave motion
    when its turning points land on sampled steps.
    """

    centers: np.ndarray
    radii: np.ndarray
    gate_open: np.ndarray
    gate_edges: Tuple[EdgeId, ...] = ()
    dt: float = 1.0

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=np.float64)
        self.radii = np.asarray(self.radii, dtype=np.float64)
        self.gate_open = np.asarray(self.gate_open, dtype=np.bool_)
        self.gate_edges = tuple(canonical_edge(e) for e in self.gate_edges)
        if self.centers.ndim != 3 or self.centers.shape[2] != 2:
            raise ValueError("centers must have shape [time, identity, 2]")
        if self.centers.shape[1] != self.radii.shape[0]:
            raise ValueError("center/radius identity mismatch")
        if self.gate_open.ndim != 2 or self.gate_open.shape[0] != self.centers.shape[0]:
            raise ValueError("gate table must have shape [time, gate]")
        if self.gate_open.shape[1] != len(self.gate_edges):
            raise ValueError("gate edge/table width mismatch")
        if self.dt <= 0:
            raise ValueError("dt must be positive")

    @classmethod
    def from_c8(
        cls,
        dynamics: Any,
        start_time: float,
        horizon: int,
        dt: float,
    ) -> "TabulatedDynamics":
        times = start_time + np.arange(int(horizon) + 1, dtype=np.float64) * float(dt)
        circles = list(dynamics.circles)
        if circles:
            centers = np.stack([c.centers_at(times) for c in circles], axis=1)
            radii = np.asarray([c.radius for c in circles], dtype=np.float64)
        else:
            centers = np.zeros((times.shape[0], 0, 2), dtype=np.float64)
            radii = np.zeros(0, dtype=np.float64)
        return cls(
            centers=centers,
            radii=radii,
            gate_open=np.zeros((times.shape[0], 0), dtype=np.bool_),
            gate_edges=(),
            dt=dt,
        )

    @property
    def horizon(self) -> int:
        return int(self.centers.shape[0] - 1)

    def centers_at(self, times: np.ndarray) -> np.ndarray:
        table_t = np.clip(np.asarray(times, dtype=np.float64) / self.dt, 0.0, self.horizon)
        lo = np.floor(table_t).astype(np.int64)
        hi = np.minimum(lo + 1, self.horizon)
        frac = table_t - lo
        return self.centers[lo] + (self.centers[hi] - self.centers[lo]) * frac[..., None, None]

    def point_free(self, point: np.ndarray, t: float) -> bool:
        if self.radii.size == 0:
            return True
        centers = self.centers_at(np.asarray([t]))[0]
        d = np.linalg.norm(centers - np.asarray(point, dtype=np.float64)[None, :], axis=1)
        return bool(np.all(d > self.radii))

    def node_free(self, point: np.ndarray, t0: float, t1: float, samples: int = 8) -> bool:
        if self.radii.size == 0:
            return True
        times = np.linspace(float(t0), float(t1), max(2, int(samples)))
        centers = self.centers_at(times)
        d = np.linalg.norm(
            centers - np.asarray(point, dtype=np.float64)[None, None, :], axis=2
        )
        return bool(np.all(d > self.radii[None, :]))

    def edge_free(
        self,
        a: np.ndarray,
        b: np.ndarray,
        t0: float,
        t1: float,
        samples: int = 16,
    ) -> bool:
        if self.radii.size == 0:
            return True
        times = np.linspace(float(t0), float(t1), max(2, int(samples)))
        fractions = (
            np.zeros_like(times)
            if t1 <= t0
            else (times - float(t0)) / (float(t1) - float(t0))
        )
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        agent = a[None, :] + (b - a)[None, :] * fractions[:, None]
        centers = self.centers_at(times)
        d = np.linalg.norm(centers - agent[:, None, :], axis=2)
        return bool(np.all(d > self.radii[None, :]))

    def gate_edge_valid(self, edge: EdgeId, t0: float, t1: float) -> bool:
        edge = canonical_edge(edge)
        matching = [i for i, candidate in enumerate(self.gate_edges) if candidate == edge]
        if not matching:
            return True
        lo = max(0, int(math.floor(min(t0, t1) / self.dt)))
        hi = min(self.horizon, int(math.ceil(max(t0, t1) / self.dt)))
        return bool(self.gate_open[lo : hi + 1, matching].all())


def normalized_prediction_to_tabulated(
    current_centers: np.ndarray,
    normalized_radii: np.ndarray,
    predicted_displacements: np.ndarray,
    predicted_gate_open: np.ndarray,
    current_gate_open: np.ndarray,
    identity_mask: np.ndarray,
    gate_mask: np.ndarray,
    gate_edges: Sequence[EdgeId],
    side_len: float,
    dt: float,
) -> TabulatedDynamics:
    """Convert the shared decoder contract into planner-facing dynamics."""
    current_centers = np.asarray(current_centers, dtype=np.float64)
    radii = np.asarray(normalized_radii, dtype=np.float64)
    displacements = np.asarray(predicted_displacements, dtype=np.float64)
    predicted_gates = np.asarray(predicted_gate_open, dtype=np.bool_)
    current_gates = np.asarray(current_gate_open, dtype=np.bool_)
    identity = np.asarray(identity_mask, dtype=np.bool_)
    gates = np.asarray(gate_mask, dtype=np.bool_)
    if displacements.ndim != 3 or displacements.shape[1:] != current_centers.shape:
        raise ValueError("predicted displacements must have shape [horizon,identity,2]")
    if predicted_gates.ndim != 2 or predicted_gates.shape[1] != current_gates.shape[0]:
        raise ValueError("predicted gates must have shape [horizon,gate]")
    if identity.shape != (current_centers.shape[0],) or radii.shape != identity.shape:
        raise ValueError("identity center/radius/mask shapes disagree")
    if gates.shape != current_gates.shape or gates.shape[0] != len(gate_edges):
        raise ValueError("gate edge/state/mask shapes disagree")
    centers = np.concatenate(
        (current_centers[None, :, :], current_centers[None, :, :] + displacements),
        axis=0,
    )
    gate_open = np.concatenate((current_gates[None, :], predicted_gates), axis=0)
    return TabulatedDynamics(
        centers=centers[:, identity, :] * float(side_len),
        radii=radii[identity] * float(side_len),
        gate_open=gate_open[:, gates],
        gate_edges=tuple(edge for edge, valid in zip(gate_edges, gates) if valid),
        dt=float(dt),
    )


@dataclass(frozen=True)
class PlanAction:
    kind: str
    source: int
    target: int
    duration: int

    def __post_init__(self) -> None:
        if self.kind not in ("wait", "edge"):
            raise ValueError(f"unknown plan action kind: {self.kind!r}")
        if self.duration < 1:
            raise ValueError("action duration must be positive")


@dataclass
class PlanResult:
    found: bool
    arrival: int
    expansions: int
    closed: int
    path_states: List[Tuple[int, int]] = field(default_factory=list)
    actions: List[PlanAction] = field(default_factory=list)

    @property
    def first_action(self) -> Optional[PlanAction]:
        return self.actions[0] if self.actions else None


def _reconstruct(
    parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], PlanAction]],
    state: Tuple[int, int],
) -> Tuple[List[Tuple[int, int]], List[PlanAction]]:
    states = [state]
    actions: List[PlanAction] = []
    while state in parent:
        previous, action = parent[state]
        actions.append(action)
        states.append(previous)
        state = previous
    states.reverse()
    actions.reverse()
    return states, actions


def predicted_space_time_astar(
    adj: Sequence[Sequence[Tuple[int, float]]],
    points: np.ndarray,
    dynamics: TabulatedDynamics,
    h_table: np.ndarray,
    budget: int,
    v_agent: float,
    dt: float,
    t_max: int,
    start: int = 0,
    goal: int = 1,
    planning_samples_per_step: int = 4,
) -> PlanResult:
    """C8-compatible space-time A* with parents and reconstructed actions."""
    h_table = np.asarray(h_table, dtype=np.float64)
    cap = int(h_table.shape[1]) - 1

    def hval(node: int, t: int) -> float:
        return float(h_table[node, min(int(t), cap)])

    g: Dict[Tuple[int, int], int] = {(int(start), 0): 0}
    parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], PlanAction]] = {}
    counter = 0
    queue: List[Tuple[float, int, int, int]] = [(hval(start, 0), counter, int(start), 0)]
    closed: set[Tuple[int, int]] = set()
    expansions = 0

    while queue and expansions < int(budget):
        _f, _counter, u, t = heapq.heappop(queue)
        state = (u, t)
        if state in closed:
            continue
        closed.add(state)
        expansions += 1
        if u == goal:
            states, actions = _reconstruct(parent, state)
            return PlanResult(True, int(t), expansions, len(closed), states, actions)

        if t + 1 <= t_max and dynamics.node_free(
            points[u], t * dt, (t + 1) * dt, samples=8
        ):
            next_state = (u, t + 1)
            if next_state not in closed and t + 1 < g.get(next_state, 1 << 30):
                g[next_state] = t + 1
                action = PlanAction("wait", u, u, 1)
                parent[next_state] = (state, action)
                counter += 1
                heapq.heappush(queue, (t + 1 + hval(u, t + 1), counter, u, t + 1))

        for v, length in adj[u]:
            steps = ST._edge_steps(float(length), float(v_agent), float(dt))
            nt = t + steps
            if nt > t_max:
                continue
            if not dynamics.gate_edge_valid((u, int(v)), t * dt, nt * dt):
                continue
            if not dynamics.edge_free(
                points[u],
                points[v],
                t * dt,
                nt * dt,
                samples=max(8, int(planning_samples_per_step) * steps),
            ):
                continue
            next_state = (int(v), nt)
            if next_state in closed or nt >= g.get(next_state, 1 << 30):
                continue
            g[next_state] = nt
            action = PlanAction("edge", u, int(v), steps)
            parent[next_state] = (state, action)
            counter += 1
            heapq.heappush(queue, (nt + hval(int(v), nt), counter, int(v), nt))

    return PlanResult(False, -1, expansions, len(closed))


def exact_future_forecast(
    episode: C12EpisodeSpec, t: int, horizon: int
) -> TabulatedDynamics:
    future = episode.future(t, horizon)
    edge_array = future["gate_edges"]
    gate_edges = tuple(tuple(int(x) for x in row) for row in edge_array.reshape(-1, 2))
    return TabulatedDynamics(
        centers=future["centers"],
        radii=future["radii"],
        gate_open=future["gate_open"],
        gate_edges=gate_edges,
        dt=episode.dt,
    )


def plan_episode_step(
    episode: C12EpisodeSpec,
    current_node: int,
    forecast: TabulatedDynamics,
    budget: Optional[int] = None,
) -> PlanResult:
    horizon = min(forecast.horizon, episode.dynamics.cfg.forecast_horizon)
    points = episode.roadmap.points
    euclid_steps = (
        np.linalg.norm(points - points[episode.goal_node][None, :], axis=1)
        / max(1e-12, episode.v_agent * episode.dt)
    )
    h_table = np.repeat(euclid_steps[:, None], horizon + 1, axis=1)
    return predicted_space_time_astar(
        episode.roadmap.adj,
        points,
        forecast,
        h_table,
        budget=episode.dynamics.cfg.planner_budget if budget is None else int(budget),
        v_agent=episode.v_agent,
        dt=episode.dt,
        t_max=horizon,
        start=int(current_node),
        goal=episode.goal_node,
        planning_samples_per_step=episode.dynamics.cfg.planning_samples_per_step,
    )


class ForecastProvider(Protocol):
    name: str

    def reset(self, episode: C12EpisodeSpec) -> None: ...

    def observe(self, episode: C12EpisodeSpec, t: int, observation: C12Observation) -> None: ...

    def forecast(
        self, episode: C12EpisodeSpec, t: int, horizon: int
    ) -> TabulatedDynamics: ...


@dataclass
class ClosedLoopResult:
    pair_id: str
    stratum: str
    provider: str
    success: bool
    failure_reason: str
    arrival_time: int
    elapsed_steps: int
    collisions: int
    first_collision_time: Optional[int]
    cumulative_expansions: int
    planning_ms: float
    replans: int
    failed_plans: int
    observation_updates: int
    decision_times: List[int]
    first_action: Optional[PlanAction]
    forecast_ms: float
    encoded_frames: int
    inference_calls: int


def run_closed_loop_episode(
    episode: C12EpisodeSpec,
    provider: ForecastProvider,
    max_steps: Optional[int] = None,
    forecast_horizon: Optional[int] = None,
) -> ClosedLoopResult:
    """Run one-action receding-horizon control and score only true dynamics."""
    cfg = episode.dynamics.cfg
    end_time = cfg.episode_steps if max_steps is None else int(max_steps)
    horizon = cfg.forecast_horizon if forecast_horizon is None else int(forecast_horizon)
    current_node = int(episode.start_node)
    t = int(episode.alias_time)
    cumulative_expansions = 0
    planning_ms = 0.0
    replans = 0
    failed_plans = 0
    observation_updates = 0
    decision_times: List[int] = []
    first_action: Optional[PlanAction] = None

    if hasattr(provider, "reset"):
        provider.reset(episode)
    burn_start = max(0, t - cfg.burn_in)
    for obs_t in range(burn_start, t + 1):
        obs = episode.observe(obs_t, current_node)
        provider.observe(episode, obs_t, obs)
        observation_updates += 1

    def finish(success: bool, reason: str, collisions: int = 0) -> ClosedLoopResult:
        return ClosedLoopResult(
            pair_id=episode.pair_id,
            stratum=episode.stratum,
            provider=str(getattr(provider, "name", provider.__class__.__name__)),
            success=bool(success),
            failure_reason=reason,
            arrival_time=int(t) if success else -1,
            elapsed_steps=int(t - episode.alias_time),
            collisions=int(collisions),
            first_collision_time=int(t) if collisions else None,
            cumulative_expansions=int(cumulative_expansions),
            planning_ms=float(planning_ms),
            replans=int(replans),
            failed_plans=int(failed_plans),
            observation_updates=int(observation_updates),
            decision_times=list(decision_times),
            first_action=first_action,
            forecast_ms=float(getattr(provider, "forecast_ms", 0.0)),
            encoded_frames=int(getattr(provider, "encoded_frames", 0)),
            inference_calls=int(getattr(provider, "inference_calls", 0)),
        )

    if current_node == episode.goal_node:
        return finish(True, "goal")

    while t < end_time:
        decision_times.append(int(t))
        forecast = provider.forecast(episode, t, horizon)
        tic = time.perf_counter()
        plan = plan_episode_step(episode, current_node, forecast)
        planning_ms += (time.perf_counter() - tic) * 1000.0
        cumulative_expansions += plan.expansions
        replans += 1
        action = plan.first_action
        if not plan.found or action is None:
            failed_plans += 1
            return finish(False, "no_plan")
        if first_action is None:
            first_action = action
        if t + action.duration > end_time:
            return finish(False, "horizon")

        source_t = t
        next_t = t + action.duration
        collided = False
        if action.kind == "wait":
            collided = not episode.dynamics.node_free(
                episode.roadmap.points[current_node],
                source_t * episode.dt,
                next_t * episode.dt,
                samples=max(8, cfg.scoring_samples_per_step * action.duration),
            )
        else:
            gate_ok = episode.dynamics.gate_edge_valid(
                (action.source, action.target), source_t * episode.dt, next_t * episode.dt
            )
            geometry_ok = episode.dynamics.edge_free(
                episode.roadmap.points[action.source],
                episode.roadmap.points[action.target],
                source_t * episode.dt,
                next_t * episode.dt,
                samples=max(16, cfg.scoring_samples_per_step * action.duration),
            )
            collided = not (gate_ok and geometry_ok)

        # Visible frames update at every simulator step.  During traversal the
        # agent is not a decision state; for the observation raster we retain
        # the source node until the arrival frame, then switch to the target.
        for step_t in range(source_t + 1, next_t + 1):
            obs_node = action.target if step_t == next_t else action.source
            obs = episode.observe(step_t, obs_node)
            provider.observe(episode, step_t, obs)
            observation_updates += 1
        t = next_t
        if collided:
            return finish(False, "collision", collisions=1)
        if action.kind == "edge":
            current_node = int(action.target)
        if current_node == episode.goal_node:
            return finish(True, "goal")

    return finish(False, "horizon")


__all__ = [
    "TabulatedDynamics",
    "normalized_prediction_to_tabulated",
    "PlanAction",
    "PlanResult",
    "ClosedLoopResult",
    "ForecastProvider",
    "predicted_space_time_astar",
    "exact_future_forecast",
    "plan_episode_step",
    "run_closed_loop_episode",
]
