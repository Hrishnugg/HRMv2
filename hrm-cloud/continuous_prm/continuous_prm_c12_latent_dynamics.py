"""C12-A deterministic hidden slow/fast dynamics and leakage-safe observations.

The module has two construction paths:

* :func:`build_challenge_pair` extracts a small route-choice subgraph from a
  real C8 roadmap.  This is the path used by the G0-A probe.
* :func:`make_fixture_pair` uses an analytic diamond roadmap.  It exists only
  to make geometry, aliasing, and planner unit tests exact and inexpensive.

The true future is deterministic conditional on the episode specification.
Latent fields live in :class:`LatentDynamicsState`; learned-model payloads are
created only by :class:`C12Observation.model_payload` and are schema audited.
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import continuous_prm_common as C


EdgeId = Tuple[int, int]
DYNAMICS_SCHEMA_VERSION = "c12a-dynamics-v6"

CHALLENGE_STRATA: Tuple[str, ...] = (
    "direction_alias",
    "slow_gate_phase",
    "route_mode_junction",
)
CONTROL_STRATUM = "present_sufficient"
STRATA: Tuple[str, ...] = CHALLENGE_STRATA + (CONTROL_STRATUM,)
SPLITS: Tuple[str, ...] = ("PROBE", "SMOKE", "TRAIN", "VALIDATION", "TEST", "PILOT")
EVALUATION_SPLITS = frozenset({"SMOKE", "PILOT", "TEST"})

_SPLIT_CODE = {name: i + 1 for i, name in enumerate(SPLITS)}
_STRATUM_CODE = {name: i + 1 for i, name in enumerate(STRATA)}
_COMPONENT_CODE = {"map": 1, "goal": 2, "roadmap": 3, "regime": 4, "bootstrap": 5}

FORBIDDEN_MODEL_FIELDS = frozenset(
    {
        "latent_regime",
        "regime",
        "direction",
        "phase",
        "phase_counter",
        "velocity",
        "future_waypoint",
        "future_waypoints",
        "future_occupancy",
        "future_centers",
        "route_mode",
        "hazard_edge",
    }
)
ALIAS_ATOL = 1.0e-6


def canonical_edge(edge: Sequence[int]) -> EdgeId:
    a, b = int(edge[0]), int(edge[1])
    return (a, b) if a <= b else (b, a)


def component_seed(
    split: str,
    stratum: str,
    episode_index: int,
    component: str,
    variant: int = 0,
) -> int:
    """Return a deterministic seed in a disjoint high-order namespace."""
    if split not in _SPLIT_CODE:
        raise KeyError(f"unknown C12 split: {split!r}")
    if stratum not in _STRATUM_CODE:
        raise KeyError(f"unknown C12 stratum: {stratum!r}")
    if component not in _COMPONENT_CODE:
        raise KeyError(f"unknown C12 seed component: {component!r}")
    if episode_index < 0 or variant < 0:
        raise ValueError("episode_index and variant must be non-negative")
    return int(
        _SPLIT_CODE[split] * 10**12
        + _STRATUM_CODE[stratum] * 10**9
        + int(episode_index) * 10**4
        + _COMPONENT_CODE[component] * 100
        + int(variant)
    )


def evaluation_condition(split: str, pair_index: int, map_family: str) -> str:
    """Deterministic, preregistered evaluation slice assignment."""
    if split not in EVALUATION_SPLITS:
        return "development_id"
    if map_family == "C_dyn_rooms_large":
        return "scale_ood"
    return ("matched_id", "long_dwell_ood", "heldout_phase_direction")[
        int(pair_index) % 3
    ]


@dataclass(frozen=True)
class C12DynamicsConfig:
    fast_period_min: int = 6
    fast_period_max: int = 12
    slow_dwell_min: int = 32
    slow_dwell_max: int = 64
    episode_steps: int = 128
    burn_in: int = 16
    forecast_horizon: int = 32
    alias_horizon: int = 8
    raster_size: int = 32
    roadmap_nodes: int = 96
    roadmap_k: int = 7
    roadmap_attempts_per_node: int = 80
    dt: float = 1.0
    default_v_agent: float = 0.20
    planner_budget: int = 5000
    planning_samples_per_step: int = 4
    scoring_samples_per_step: int = 8

    def __post_init__(self) -> None:
        if not (1 <= self.fast_period_min <= self.fast_period_max):
            raise ValueError("invalid fast-period range")
        if not (1 <= self.slow_dwell_min <= self.slow_dwell_max):
            raise ValueError("invalid slow-dwell range")
        if self.burn_in < 2 or self.forecast_horizon < self.alias_horizon:
            raise ValueError("burn-in/horizon do not support the alias contract")
        if self.episode_steps <= self.burn_in + self.alias_horizon:
            raise ValueError("episode is too short for burn-in plus alias horizon")
        if self.raster_size < 8:
            raise ValueError("raster_size must be at least 8")


@dataclass(frozen=True)
class LatentRegime:
    stratum: str
    variant: int
    direction: int
    gate_phase: int
    route_mode: int


@dataclass(frozen=True)
class RegimeSchedule:
    fast_period: int
    slow_dwell: int
    alias_time: int
    hazard_edge: EdgeId
    safe_edge: EdgeId
    hazard_start: int
    hazard_end: int


@dataclass
class GateSchedule:
    edge_ids: Tuple[EdgeId, ...]
    open_by_step: np.ndarray

    def __post_init__(self) -> None:
        self.edge_ids = tuple(canonical_edge(e) for e in self.edge_ids)
        self.open_by_step = np.asarray(self.open_by_step, dtype=np.bool_)
        if self.open_by_step.ndim != 2:
            raise ValueError("gate open table must have shape [time, gate]")
        if self.open_by_step.shape[1] != len(self.edge_ids):
            raise ValueError("gate edge/table width mismatch")


@dataclass(frozen=True)
class LatentDynamicsState:
    t: int
    centers: np.ndarray
    radii: np.ndarray
    gate_open: np.ndarray
    latent_regime: LatentRegime
    phase_counter: int
    future_waypoints: np.ndarray


@dataclass
class C12Observation:
    static_occupancy: np.ndarray
    dynamic_occupancy: np.ndarray
    gate_open_raster: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    identity_mask: np.ndarray
    agent_goal_raster: np.ndarray
    visible_regime_context: Optional[np.ndarray] = None

    def model_payload(self) -> Dict[str, np.ndarray]:
        payload: Dict[str, np.ndarray] = {
            "static_occupancy": np.asarray(self.static_occupancy, dtype=np.float32),
            "dynamic_occupancy": np.asarray(self.dynamic_occupancy, dtype=np.float32),
            "gate_open_raster": np.asarray(self.gate_open_raster, dtype=np.float32),
            "centers": np.asarray(self.centers, dtype=np.float32),
            "radii": np.asarray(self.radii, dtype=np.float32),
            "identity_mask": np.asarray(self.identity_mask, dtype=np.float32),
            "agent_goal_raster": np.asarray(self.agent_goal_raster, dtype=np.float32),
        }
        if self.visible_regime_context is not None:
            payload["visible_regime_context"] = np.asarray(
                self.visible_regime_context, dtype=np.float32
            )
        audit_model_payload(payload)
        return payload


def audit_model_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed if a privileged latent or future field reaches a model."""
    leaked = sorted(FORBIDDEN_MODEL_FIELDS.intersection(str(k) for k in payload))
    if leaked:
        raise ValueError(f"forbidden latent model fields: {', '.join(leaked)}")
    for key, value in payload.items():
        arr = np.asarray(value)
        if arr.dtype == object:
            raise ValueError(f"model field {key!r} has forbidden object dtype")
        if not np.isfinite(arr.astype(np.float64, copy=False)).all():
            raise ValueError(f"model field {key!r} contains non-finite values")


def serialize_observation(obs: C12Observation) -> np.ndarray:
    payload = obs.model_payload()
    ordered = [payload[k].reshape(-1) for k in sorted(payload)]
    return np.concatenate(ordered).astype(np.float32, copy=False)


def present_observation_key(obs: C12Observation) -> bytes:
    """Fixed-resolution key used only by the preregistered alias audit.

    The resolution is frozen to the design's ``atol=1e-6`` and is never used
    by a planner or learned model.  The audit additionally requires direct
    ``allclose(..., atol=1e-6, rtol=0)`` equality, so this key cannot create a
    favorable alias by coarse post-hoc binning.
    """
    values = serialize_observation(obs).astype(np.float64)
    quantized = np.rint(values / ALIAS_ATOL).astype(np.int64)
    return quantized.tobytes(order="C")


def _grid(side: float, size: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(size, dtype=np.float64) + 0.5) * side / size
    ys = (np.arange(size, dtype=np.float64) + 0.5) * side / size
    return np.meshgrid(xs, ys, indexing="xy")


def _render_static(world: C.World, size: int) -> np.ndarray:
    xx, yy = _grid(world.side_len, size)
    out = np.zeros((size, size), dtype=np.float32)
    for obstacle in world.obstacles:
        if obstacle.kind == "circle":
            mask = (xx - obstacle.cx) ** 2 + (yy - obstacle.cy) ** 2 <= obstacle.radius**2
        else:
            mask = (np.abs(xx - obstacle.cx) <= obstacle.hw) & (
                np.abs(yy - obstacle.cy) <= obstacle.hh
            )
        out[mask] = 1.0
    return out


def _render_circles(
    side: float, size: int, centers: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    xx, yy = _grid(side, size)
    out = np.zeros((size, size), dtype=np.float32)
    for center, radius in zip(centers, radii):
        mask = (xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2 <= float(radius) ** 2
        out[mask] = 1.0
    return out


def _mark_point(raster: np.ndarray, point: np.ndarray, side: float, value: float = 1.0) -> None:
    size = raster.shape[-1]
    x = int(np.clip(math.floor(float(point[0]) / side * size), 0, size - 1))
    y = int(np.clip(math.floor(float(point[1]) / side * size), 0, size - 1))
    raster[..., y, x] = value


class LatentDynamics:
    """Deterministic table-backed true simulator with C8-style geometry calls."""

    def __init__(
        self,
        world: C.World,
        roadmap: C.Roadmap,
        cfg: C12DynamicsConfig,
        regime: LatentRegime,
        schedule: RegimeSchedule,
        centers: np.ndarray,
        radii: np.ndarray,
        gates: GateSchedule,
        expose_regime: bool = False,
    ) -> None:
        self.world = world
        self.roadmap = roadmap
        self.cfg = cfg
        self.regime = regime
        self.schedule = schedule
        self.centers = np.asarray(centers, dtype=np.float64)
        self.radii = np.asarray(radii, dtype=np.float64)
        self.gates = gates
        self.expose_regime = bool(expose_regime)
        if self.centers.ndim != 3 or self.centers.shape[2] != 2:
            raise ValueError("centers must have shape [time, identity, 2]")
        if self.centers.shape[1] != self.radii.shape[0]:
            raise ValueError("center/radius identity mismatch")
        if self.centers.shape[0] != self.gates.open_by_step.shape[0]:
            raise ValueError("center/gate time-table mismatch")
        self.cursor = 0
        self.last_observation: Optional[C12Observation] = None
        self._static_raster = _render_static(world, cfg.raster_size)

    @property
    def gate_open(self) -> np.ndarray:
        return self.gates.open_by_step

    def _time_indices(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        clipped = np.clip(np.asarray(t, dtype=np.float64), 0.0, self.centers.shape[0] - 1)
        lo = np.floor(clipped).astype(np.int64)
        hi = np.minimum(lo + 1, self.centers.shape[0] - 1)
        frac = clipped - lo
        return lo, hi, frac

    def centers_at(self, times: np.ndarray) -> np.ndarray:
        lo, hi, frac = self._time_indices(times)
        a = self.centers[lo]
        b = self.centers[hi]
        return a + (b - a) * frac[..., None, None]

    def point_free(self, point: np.ndarray, t: float) -> bool:
        centers = self.centers_at(np.asarray([t]))[0]
        d = np.linalg.norm(centers - np.asarray(point, dtype=np.float64)[None, :], axis=1)
        return bool(np.all(d > self.radii))

    def node_free(self, point: np.ndarray, t0: float, t1: float, samples: int = 8) -> bool:
        ts = np.linspace(float(t0), float(t1), max(2, int(samples)))
        centers = self.centers_at(ts)
        d = np.linalg.norm(centers - np.asarray(point, dtype=np.float64)[None, None, :], axis=2)
        return bool(np.all(d > self.radii[None, :]))

    def edge_free(
        self,
        a: np.ndarray,
        b: np.ndarray,
        t0: float,
        t1: float,
        samples: int = 16,
    ) -> bool:
        ts = np.linspace(float(t0), float(t1), max(2, int(samples)))
        frac = np.zeros_like(ts) if t1 <= t0 else (ts - float(t0)) / (float(t1) - float(t0))
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        agent = a[None, :] + (b - a)[None, :] * frac[:, None]
        centers = self.centers_at(ts)
        d = np.linalg.norm(centers - agent[:, None, :], axis=2)
        return bool(np.all(d > self.radii[None, :]))

    def gate_edge_valid(self, edge: EdgeId, t0: float, t1: float) -> bool:
        edge = canonical_edge(edge)
        matching = [i for i, e in enumerate(self.gates.edge_ids) if e == edge]
        if not matching:
            return True
        lo = max(0, int(math.floor(min(t0, t1))))
        hi = min(self.gate_open.shape[0] - 1, int(math.ceil(max(t0, t1))))
        return bool(self.gate_open[lo : hi + 1, matching].all())

    def state_at(self, t: int) -> LatentDynamicsState:
        ti = int(np.clip(t, 0, self.centers.shape[0] - 1))
        future_stop = min(self.centers.shape[0], ti + self.cfg.forecast_horizon + 1)
        return LatentDynamicsState(
            t=ti,
            centers=self.centers[ti].copy(),
            radii=self.radii.copy(),
            gate_open=self.gate_open[ti].copy(),
            latent_regime=self.regime,
            phase_counter=(ti + self.regime.gate_phase) % self.schedule.slow_dwell,
            future_waypoints=self.centers[ti + 1 : future_stop].copy(),
        )

    def _gate_raster(self, t: int) -> np.ndarray:
        raster = np.zeros((self.cfg.raster_size, self.cfg.raster_size), dtype=np.float32)
        ti = int(np.clip(t, 0, self.gate_open.shape[0] - 1))
        for j, edge in enumerate(self.gates.edge_ids):
            mid = 0.5 * (self.roadmap.points[edge[0]] + self.roadmap.points[edge[1]])
            if self.gate_open[ti, j]:
                _mark_point(raster, mid, self.world.side_len, 1.0)
        return raster

    def observe(self, t: int, agent_node: int, goal_node: int) -> C12Observation:
        state = self.state_at(t)
        agent_goal = np.zeros((2, self.cfg.raster_size, self.cfg.raster_size), dtype=np.float32)
        _mark_point(agent_goal[0], self.roadmap.points[int(agent_node)], self.world.side_len)
        _mark_point(agent_goal[1], self.roadmap.points[int(goal_node)], self.world.side_len)
        visible = None
        if self.expose_regime:
            visible = np.asarray(
                [
                    float(self.regime.direction),
                    float(state.phase_counter) / float(max(1, self.schedule.slow_dwell)),
                    float(self.regime.route_mode),
                ],
                dtype=np.float32,
            )
        obs = C12Observation(
            static_occupancy=self._static_raster.copy(),
            dynamic_occupancy=_render_circles(
                self.world.side_len, self.cfg.raster_size, state.centers, state.radii
            ),
            gate_open_raster=self._gate_raster(state.t),
            centers=(state.centers / self.world.side_len).astype(np.float32),
            radii=(state.radii / self.world.side_len).astype(np.float32),
            identity_mask=np.ones(state.radii.shape[0], dtype=np.float32),
            agent_goal_raster=agent_goal,
            visible_regime_context=visible,
        )
        self.last_observation = obs
        return obs

    def future(self, t: int, horizon: int) -> Dict[str, np.ndarray]:
        start = int(np.clip(t, 0, self.centers.shape[0] - 1))
        idx = np.clip(np.arange(start, start + int(horizon) + 1), 0, self.centers.shape[0] - 1)
        return {
            "centers": self.centers[idx].copy(),
            "radii": self.radii.copy(),
            "gate_open": self.gate_open[idx].copy(),
            "gate_edges": np.asarray(self.gates.edge_ids, dtype=np.int64),
        }

    def future_occupancy(self, t: int, horizon: int) -> np.ndarray:
        future = self.future(t, horizon)
        frames: List[np.ndarray] = []
        for k in range(1, int(horizon) + 1):
            frame = _render_circles(
                self.world.side_len,
                self.cfg.raster_size,
                future["centers"][k],
                future["radii"],
            )
            for j, edge in enumerate(self.gates.edge_ids):
                if not bool(future["gate_open"][k, j]):
                    mid = 0.5 * (self.roadmap.points[edge[0]] + self.roadmap.points[edge[1]])
                    _mark_point(frame, mid, self.world.side_len, 1.0)
            frames.append(frame)
        return np.stack(frames, axis=0)

    def advance_to(self, t: int) -> LatentDynamicsState:
        self.cursor = int(np.clip(t, 0, self.centers.shape[0] - 1))
        return self.state_at(self.cursor)

    def reset(self) -> LatentDynamicsState:
        self.cursor = 0
        self.last_observation = None
        return self.state_at(0)


@dataclass
class C12EpisodeSpec:
    stratum: str
    split: str
    pair_index: int
    pair_id: str
    map_family: str
    map_seed: int
    goal_seed: int
    regime_seed: int
    world: C.World
    roadmap: C.Roadmap
    start_node: int
    goal_node: int
    alias_time: int
    v_agent: float
    dt: float
    regime: LatentRegime
    schedule: RegimeSchedule
    dynamics: LatentDynamics
    oracle_first_action_hint: EdgeId
    route_lengths: Tuple[float, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def observe(self, t: int, agent_node: Optional[int] = None) -> C12Observation:
        return self.dynamics.observe(
            t,
            self.start_node if agent_node is None else int(agent_node),
            self.goal_node,
        )

    def future(self, t: int, horizon: int) -> Dict[str, np.ndarray]:
        return self.dynamics.future(t, horizon)

    def future_occupancy(self, t: int, horizon: int) -> np.ndarray:
        return self.dynamics.future_occupancy(t, horizon)


@dataclass(frozen=True)
class C12EpisodeRecord:
    """Leakage-safe, fixed-shape representation of one simulator episode.

    ``model_arrays`` contains only visible inputs, supervised targets, and
    evaluation masks.  Hidden regime values are deliberately kept in the
    JSON-only ``privileged_diagnostics`` block so an ``.npz`` loader cannot
    accidentally expose them to a learned arm.
    """

    frame_rasters: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    identity_mask: np.ndarray
    visible_regime_context: np.ndarray
    visible_regime_mask: np.ndarray
    target_center_displacements: np.ndarray
    target_gate_open: np.ndarray
    gate_mask: np.ndarray
    route_critical_mask: np.ndarray
    route_edge_midpoints: np.ndarray
    metadata: Dict[str, Any]
    privileged_diagnostics: Dict[str, Any]

    def model_arrays(self) -> Dict[str, np.ndarray]:
        arrays = {
            "frame_rasters": np.asarray(self.frame_rasters),
            "centers": np.asarray(self.centers),
            "radii": np.asarray(self.radii),
            "identity_mask": np.asarray(self.identity_mask),
            "visible_regime_context": np.asarray(self.visible_regime_context),
            "visible_regime_mask": np.asarray(self.visible_regime_mask),
            "target_center_displacements": np.asarray(
                self.target_center_displacements
            ),
            "target_gate_open": np.asarray(self.target_gate_open),
            "gate_mask": np.asarray(self.gate_mask),
            "route_critical_mask": np.asarray(self.route_critical_mask),
            "route_edge_midpoints": np.asarray(self.route_edge_midpoints),
        }
        for name, array in arrays.items():
            if array.dtype == object:
                raise ValueError(f"episode array {name!r} has forbidden object dtype")
            if not np.isfinite(array.astype(np.float64, copy=False)).all():
                raise ValueError(f"episode array {name!r} contains non-finite values")
        leaked = sorted(FORBIDDEN_MODEL_FIELDS.intersection(arrays))
        if leaked:
            raise ValueError(f"forbidden latent episode arrays: {', '.join(leaked)}")
        return arrays


def _episode_static_map_id(
    episode: C12EpisodeSpec, static_occupancy: np.ndarray
) -> str:
    """Stable identifier shared by the two counterfactual regime variants."""
    digest = hashlib.sha256()
    digest.update(str(episode.map_family).encode("utf-8"))
    digest.update(np.asarray([episode.map_seed, episode.goal_seed], dtype=np.int64).tobytes())
    digest.update(np.asarray([episode.world.side_len], dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(static_occupancy, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(episode.roadmap.points, dtype=np.float64).tobytes())
    return digest.hexdigest()[:24]


def build_episode_record(episode: C12EpisodeSpec) -> C12EpisodeRecord:
    """Materialize one deterministic C12 episode for dataset sharding.

    The agent is held at the episode start in the offline observation stream.
    This makes the forecasting dataset independent of any planner policy while
    retaining the same current-agent/goal channels used at planning time.
    Targets are direct displacements from the current center at horizons
    ``1..H`` and therefore match the shared direct-horizon decoder contract.
    """
    cfg = episode.dynamics.cfg
    steps = int(cfg.episode_steps)
    horizon = int(cfg.forecast_horizon)
    side = float(episode.world.side_len)

    frame_rows: List[np.ndarray] = []
    center_rows: List[np.ndarray] = []
    radius_rows: List[np.ndarray] = []
    identity_rows: List[np.ndarray] = []
    visible_rows: List[np.ndarray] = []
    visible_mask_rows: List[np.ndarray] = []
    displacement_rows: List[np.ndarray] = []
    gate_rows: List[np.ndarray] = []
    gate_mask_rows: List[np.ndarray] = []
    critical_rows: List[np.ndarray] = []
    hazard_midpoint = 0.5 * (
        episode.roadmap.points[int(episode.schedule.hazard_edge[0])]
        + episode.roadmap.points[int(episode.schedule.hazard_edge[1])]
    )
    hazard_gate_indices = [
        index
        for index, edge in enumerate(episode.dynamics.gates.edge_ids)
        if canonical_edge(edge) == canonical_edge(episode.schedule.hazard_edge)
    ]

    for t in range(steps):
        obs = episode.observe(t, episode.start_node)
        payload = obs.model_payload()
        frame = np.stack(
            [
                payload["static_occupancy"],
                payload["dynamic_occupancy"],
                payload["gate_open_raster"],
                payload["agent_goal_raster"][0],
                payload["agent_goal_raster"][1],
            ],
            axis=0,
        )
        # Every current raster channel is binary, so uint8 preserves it exactly
        # while keeping full collection roughly eight times smaller than float.
        frame_rows.append(np.rint(np.clip(frame, 0.0, 1.0)).astype(np.uint8))
        center_rows.append(np.asarray(payload["centers"], dtype=np.float32))
        radius_rows.append(np.asarray(payload["radii"], dtype=np.float32))
        identity_rows.append(np.asarray(payload["identity_mask"], dtype=np.uint8))

        visible = payload.get("visible_regime_context")
        if visible is None:
            visible_rows.append(np.zeros(3, dtype=np.float32))
            visible_mask_rows.append(np.zeros(1, dtype=np.uint8))
        else:
            visible_rows.append(np.asarray(visible, dtype=np.float32).reshape(3))
            visible_mask_rows.append(np.ones(1, dtype=np.uint8))

        future = episode.future(t, horizon)
        future_centers = np.asarray(future["centers"], dtype=np.float64)
        displacement_rows.append(
            ((future_centers[1 : horizon + 1] - future_centers[0:1]) / side).astype(
                np.float32
            )
        )
        future_gates = np.asarray(future["gate_open"], dtype=np.uint8)
        gate_rows.append(future_gates[1 : horizon + 1])
        gate_mask_rows.append(np.ones(future_gates.shape[1], dtype=np.uint8))
        future_centers_physical = future_centers[1 : horizon + 1]
        center_is_critical = np.any(
            np.linalg.norm(
                future_centers_physical - hazard_midpoint[None, None, :], axis=-1
            )
            <= 1.0e-9,
            axis=1,
        )
        gate_is_critical = np.zeros(horizon, dtype=np.bool_)
        for gate_index in hazard_gate_indices:
            gate_is_critical |= ~future_gates[1 : horizon + 1, gate_index].astype(
                np.bool_
            )
        critical_rows.append((center_is_critical | gate_is_critical).astype(np.uint8))

    first_observation = episode.observe(0, episode.start_node)
    static_map_id = _episode_static_map_id(episode, first_observation.static_occupancy)
    edge_midpoints = []
    for edge in episode.dynamics.gates.edge_ids:
        midpoint = 0.5 * (
            episode.roadmap.points[int(edge[0])]
            + episode.roadmap.points[int(edge[1])]
        )
        edge_midpoints.append(midpoint / side)

    episode_id = f"{episode.pair_id}-v{int(episode.regime.variant)}"
    metadata: Dict[str, Any] = {
        "episode_id": episode_id,
        "pair_id": episode.pair_id,
        "pair_index": int(episode.pair_index),
        "variant": int(episode.regime.variant),
        "split": episode.split,
        "stratum": episode.stratum,
        "map_family": episode.map_family,
        "static_map_id": static_map_id,
        "map_seed": int(episode.map_seed),
        "goal_seed": int(episode.goal_seed),
        "regime_seed": int(episode.regime_seed),
        "config_hash": config_hash(cfg),
        "side_len": side,
        "episode_steps": steps,
        "forecast_horizon": horizon,
        "alias_time": int(episode.alias_time),
        "start_node": int(episode.start_node),
        "goal_node": int(episode.goal_node),
        "v_agent": float(episode.v_agent),
        "dt": float(episode.dt),
        "eval_condition": str(episode.diagnostics.get("eval_condition", "development_id")),
        "is_long_dwell_ood": bool(episode.diagnostics.get("is_long_dwell_ood", False)),
        "is_scale_ood": bool(episode.diagnostics.get("is_scale_ood", False)),
        "is_heldout_combo": bool(episode.diagnostics.get("is_heldout_combo", False)),
    }
    privileged: Dict[str, Any] = {
        "latent_regime": episode.regime.stratum,
        "direction": int(episode.regime.direction),
        "phase": int(episode.regime.gate_phase),
        "route_mode": int(episode.regime.route_mode),
        "hazard_edge": list(episode.schedule.hazard_edge),
        "safe_edge": list(episode.schedule.safe_edge),
        "hazard_start": int(episode.schedule.hazard_start),
        "hazard_end": int(episode.schedule.hazard_end),
        "fast_period": int(episode.schedule.fast_period),
        "slow_dwell": int(episode.schedule.slow_dwell),
        "oracle_first_action_hint": list(episode.oracle_first_action_hint),
        "route_lengths": [float(value) for value in episode.route_lengths],
        "episode_diagnostics": copy.deepcopy(episode.diagnostics),
    }
    record = C12EpisodeRecord(
        frame_rasters=np.stack(frame_rows, axis=0),
        centers=np.stack(center_rows, axis=0),
        radii=np.stack(radius_rows, axis=0),
        identity_mask=np.stack(identity_rows, axis=0),
        visible_regime_context=np.stack(visible_rows, axis=0),
        visible_regime_mask=np.stack(visible_mask_rows, axis=0),
        target_center_displacements=np.stack(displacement_rows, axis=0),
        target_gate_open=np.stack(gate_rows, axis=0),
        gate_mask=np.stack(gate_mask_rows, axis=0),
        route_critical_mask=np.stack(critical_rows, axis=0),
        route_edge_midpoints=np.asarray(edge_midpoints, dtype=np.float32),
        metadata=metadata,
        privileged_diagnostics=privileged,
    )
    record.model_arrays()  # fail closed before the record can reach disk
    return record


def audit_alias_pair(left: C12EpisodeSpec, right: C12EpisodeSpec) -> Dict[str, Any]:
    if left.pair_id != right.pair_id:
        raise ValueError("alias audit requires a declared counterfactual pair")
    left_obs = left.observe(left.alias_time)
    right_obs = right.observe(right.alias_time)
    left_now = serialize_observation(left_obs)
    right_now = serialize_observation(right_obs)
    current_match = bool(
        present_observation_key(left_obs) == present_observation_key(right_obs)
        and np.allclose(left_now, right_now, atol=ALIAS_ATOL, rtol=0.0)
    )
    future_diverges = not np.array_equal(
        left.future_occupancy(left.alias_time, left.dynamics.cfg.alias_horizon),
        right.future_occupancy(right.alias_time, right.dynamics.cfg.alias_horizon),
    )
    first_action_diverges = left.oracle_first_action_hint != right.oracle_first_action_hint
    return {
        "pair_id": left.pair_id,
        "current_match": current_match,
        "future_diverges": bool(future_diverges),
        "first_action_diverges": bool(first_action_diverges),
        "max_current_abs_diff": float(np.max(np.abs(left_now - right_now))),
        "is_alias": bool(current_match and future_diverges and first_action_diverges),
    }


def _roadmap(points: np.ndarray, edges: Sequence[Tuple[int, int]]) -> C.Roadmap:
    points = np.asarray(points, dtype=np.float64)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(points.shape[0])]
    for a, b in edges:
        w = float(np.linalg.norm(points[a] - points[b]))
        adj[a].append((b, w))
        adj[b].append((a, w))
    for row in adj:
        row.sort(key=lambda x: x[0])
    dist = C.dijkstra_to_goal(adj, goal_idx=1)
    return C.Roadmap(
        points=points,
        adj=adj,
        dist_to_goal=dist,
        connected_to_goal=np.isfinite(dist) & (dist < C.INF / 10.0),
    )


def _fixture_world_and_roadmap() -> Tuple[C.World, C.Roadmap, EdgeId, EdgeId, Tuple[float, float]]:
    points = np.asarray(
        [
            [0.10, 0.50],  # 0 decision start
            [0.90, 0.50],  # 1 goal
            [0.50, 0.50],  # 2 short route
            [0.10, 0.85],  # 3 long route a
            [0.90, 0.85],  # 4 long route b
        ],
        dtype=np.float64,
    )
    rm = _roadmap(points, [(0, 2), (2, 1), (0, 3), (3, 4), (4, 1)])
    world = C.World(
        spec_name="C12_fixture_diamond",
        side_len=1.0,
        obstacles=[],
        start=points[0].copy(),
        goal=points[1].copy(),
        descriptor=np.zeros(8, dtype=np.float32),
        meta={"c12_fixture": True},
    )
    short_edge = canonical_edge((0, 2))
    long_edge = canonical_edge((0, 3))
    short_len = float(np.linalg.norm(points[0] - points[2]) + np.linalg.norm(points[2] - points[1]))
    long_len = float(
        np.linalg.norm(points[0] - points[3])
        + np.linalg.norm(points[3] - points[4])
        + np.linalg.norm(points[4] - points[1])
    )
    return world, rm, short_edge, long_edge, (short_len, long_len)


def _path_length(rm: C.Roadmap, path: Sequence[int]) -> float:
    return float(
        sum(np.linalg.norm(rm.points[a] - rm.points[b]) for a, b in zip(path[:-1], path[1:]))
    )


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    point = np.asarray(point, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-18:
        return float(np.linalg.norm(point - a))
    frac = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + frac * ab)))


def _branch_hazards_are_separated(
    rm: C.Roadmap, short_edge: EdgeId, long_edge: EdgeId, side_len: float
) -> bool:
    """Ensure each constructed circle blocks only its intended first edge."""
    short_mid = 0.5 * (rm.points[short_edge[0]] + rm.points[short_edge[1]])
    long_mid = 0.5 * (rm.points[long_edge[0]] + rm.points[long_edge[1]])
    common = 0.5 * (short_mid + long_mid)
    radius = max(
        0.008 * side_len,
        min(0.025 * side_len, 0.15 * np.linalg.norm(short_mid - long_mid)),
    )
    start = rm.points[0]
    if min(np.linalg.norm(common - start), np.linalg.norm(short_mid - start), np.linalg.norm(long_mid - start)) <= 2.25 * radius:
        return False
    short_a, short_b = rm.points[short_edge[0]], rm.points[short_edge[1]]
    long_a, long_b = rm.points[long_edge[0]], rm.points[long_edge[1]]
    if _point_segment_distance(short_mid, long_a, long_b) <= 2.25 * radius:
        return False
    if _point_segment_distance(long_mid, short_a, short_b) <= 2.25 * radius:
        return False
    # Frozen-frame must see the current shared position as clear on both
    # branches; otherwise it fails because of present geometry, not aliasing.
    if _point_segment_distance(common, short_a, short_b) <= 1.25 * radius:
        return False
    if _point_segment_distance(common, long_a, long_b) <= 1.25 * radius:
        return False
    return True


def _weighted_shortest_path(
    adj: Sequence[Sequence[Tuple[int, float]]],
    start: int,
    goal: int,
    edge_multiplier: Optional[Mapping[EdgeId, float]] = None,
) -> Optional[List[int]]:
    dist = {int(start): 0.0}
    parent: Dict[int, int] = {}
    queue: List[Tuple[float, int]] = [(0.0, int(start))]
    while queue:
        d, u = heapq.heappop(queue)
        if d != dist.get(u):
            continue
        if u == goal:
            path = [u]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return path
        for v, w in adj[u]:
            mult = 1.0 if edge_multiplier is None else float(edge_multiplier.get(canonical_edge((u, v)), 1.0))
            nd = d + float(w) * mult
            if nd < dist.get(int(v), float("inf")):
                dist[int(v)] = nd
                parent[int(v)] = u
                heapq.heappush(queue, (nd, int(v)))
    return None


def _route_choice_subgraph(
    rm: C.Roadmap, seed: int, max_arrival_steps: int, v_agent: float, dt: float
) -> Optional[Tuple[C.Roadmap, EdgeId, EdgeId, Tuple[float, float]]]:
    """Extract two first-action-distinct routes, preserving real PRM edges.

    The decision junction need not be the original C8 start.  A first smoke
    calibration showed that some maze starts have only one alternative inside
    the fixed 32-step forecast even though a valid two-route junction exists
    later in the same roadmap.  We therefore search goal-reachable, degree>=2
    PRM nodes and remap the accepted junction to index 0.  The goal remains the
    original C8 goal (index 1).
    """
    rng = np.random.default_rng(seed)
    edge_ids = sorted(
        {canonical_edge((u, v)) for u, row in enumerate(rm.adj) for v, _ in row}
    )
    # Prefer junctions with 8--16 lower-bound steps remaining: enough route
    # structure to matter, but ample forecast slack.  Keep original start as a
    # candidate when it already satisfies the contract.
    candidate_starts = [
        node
        for node in range(len(rm.adj))
        if node != 1
        and len(rm.adj[node]) >= 2
        and np.isfinite(rm.dist_to_goal[node])
        and rm.dist_to_goal[node] / max(1e-12, v_agent * dt) <= max_arrival_steps
    ]
    candidate_starts.sort(
        key=lambda node: (
            0 if node == 0 else 1,
            abs(rm.dist_to_goal[node] / max(1e-12, v_agent * dt) - 12.0),
            node,
        )
    )
    # Deterministic diversity after the preferred prefix.
    if len(candidate_starts) > 24:
        prefix = candidate_starts[:8]
        tail = candidate_starts[8:]
        rng.shuffle(tail)
        candidate_starts = prefix + tail[:16]

    accepted: Optional[Tuple[int, float, List[int], float, List[int]]] = None
    accepted_ratio = -float("inf")
    for start_old in candidate_starts:
        candidates: Dict[Tuple[int, ...], List[int]] = {}
        base = _weighted_shortest_path(rm.adj, start_old, 1)
        if base is not None and len(base) >= 2:
            candidates[tuple(base)] = base
        for _ in range(40):
            multipliers = {
                edge: float(np.exp(rng.uniform(-1.8, 1.8))) for edge in edge_ids
            }
            path = _weighted_shortest_path(rm.adj, start_old, 1, multipliers)
            if path is not None and len(path) >= 2:
                candidates[tuple(path)] = path
        viable: List[Tuple[float, List[int]]] = []
        for path in candidates.values():
            length = _path_length(rm, path)
            arrival = sum(
                max(
                    1,
                    int(
                        math.ceil(
                            float(np.linalg.norm(rm.points[a] - rm.points[b]))
                            / (v_agent * dt)
                        )
                    ),
                )
                for a, b in zip(path[:-1], path[1:])
            )
            if arrival <= max_arrival_steps:
                viable.append((length, path))
        viable.sort(key=lambda item: (item[0], tuple(item[1])))
        if len(viable) < 2:
            continue
        short_len, short_path = viable[0]
        alternatives = [
            (length, path) for length, path in viable if path[1] != short_path[1]
        ]
        if not alternatives:
            continue
        long_len, long_path = max(
            alternatives, key=lambda item: (item[0], tuple(item[1]))
        )
        ratio = long_len / max(1e-12, short_len)
        if ratio > accepted_ratio:
            accepted_ratio = ratio
            accepted = (start_old, short_len, short_path, long_len, long_path)
        if ratio >= 1.35:
            break
    if accepted is None:
        return None
    start_old, short_len, short_path, long_len, long_path = accepted

    # Preserve only edges that belong to one of the two accepted PRM routes.
    used_old: List[int] = [start_old, 1]
    for node in list(short_path[1:-1]) + list(long_path[1:-1]):
        if node not in used_old:
            used_old.append(int(node))
    old_to_new = {old: new for new, old in enumerate(used_old)}
    points = rm.points[used_old].copy()
    new_edges: set[EdgeId] = set()
    for path in (short_path, long_path):
        for a, b in zip(path[:-1], path[1:]):
            new_edges.add(canonical_edge((old_to_new[a], old_to_new[b])))
    reduced = _roadmap(points, sorted(new_edges))
    short_edge = canonical_edge((0, old_to_new[short_path[1]]))
    long_edge = canonical_edge((0, old_to_new[long_path[1]]))
    if short_edge == long_edge or not reduced.connected_to_goal[0]:
        return None
    return reduced, short_edge, long_edge, (float(short_len), float(long_len))


def _resample_goal(world: C.World, goal_seed: int) -> Optional[C.World]:
    out = copy.deepcopy(world)
    rng = random.Random(goal_seed)
    for _ in range(160):
        goal = C.sample_free_point(rng, out.side_len, out.obstacles, max_attempts=200)
        if goal is None:
            continue
        if float(np.linalg.norm(goal - out.start)) < 0.50 * out.side_len:
            continue
        out.goal = goal.astype(np.float64)
        out.meta = dict(out.meta)
        out.meta["c12_goal_seed"] = int(goal_seed)
        return out
    return None


def _select_connected_goal(
    world: C.World, rm: C.Roadmap, goal_seed: int
) -> Optional[Tuple[C.World, C.Roadmap]]:
    """Choose a goal with an independent seed inside the start component.

    Sampling a free point before checking connectivity can put a goal in a
    sealed room, especially in held-out ``rooms_large``.  Instead, construct
    the ordinary C8 roadmap first, choose a distant connected roadmap node via
    the independent goal seed, and reindex that node to the canonical goal
    index 1.  Every retained edge is still an original C8 PRM edge.
    """
    start = rm.points[0]
    candidates = [
        i
        for i in range(2, rm.points.shape[0])
        if bool(rm.connected_to_goal[i])
        and float(np.linalg.norm(rm.points[i] - start)) >= 0.45 * world.side_len
    ]
    if not candidates:
        if bool(rm.connected_to_goal[1]) and float(np.linalg.norm(rm.points[1] - start)) >= 0.45 * world.side_len:
            candidates = [1]
        else:
            return None
    rng = np.random.default_rng(goal_seed)
    chosen = int(candidates[int(rng.integers(0, len(candidates)))])
    order = [0, chosen] + [i for i in range(rm.points.shape[0]) if i not in (0, chosen)]
    old_to_new = {old: new for new, old in enumerate(order)}
    points = rm.points[order].copy()
    adj: List[List[Tuple[int, float]]] = [[] for _ in order]
    for old_u in order:
        new_u = old_to_new[old_u]
        adj[new_u] = sorted(
            [(old_to_new[int(old_v)], float(w)) for old_v, w in rm.adj[old_u]],
            key=lambda item: item[0],
        )
    dist = C.dijkstra_to_goal(adj, goal_idx=1)
    connected = np.isfinite(dist) & (dist < C.INF / 10.0)
    if not connected[0]:
        return None
    out_world = copy.deepcopy(world)
    out_world.goal = points[1].copy()
    out_world.meta = dict(out_world.meta)
    out_world.meta.update(
        {
            "c12_goal_seed": int(goal_seed),
            "c12_goal_source_node": int(chosen),
        }
    )
    return out_world, C.Roadmap(
        points=points,
        adj=adj,
        dist_to_goal=dist,
        connected_to_goal=connected,
    )


def _build_tables(
    stratum: str,
    variant: int,
    world: C.World,
    rm: C.Roadmap,
    short_edge: EdgeId,
    long_edge: EdgeId,
    cfg: C12DynamicsConfig,
    regime_seed: int,
    condition: str,
) -> Tuple[LatentRegime, RegimeSchedule, np.ndarray, np.ndarray, GateSchedule, EdgeId]:
    rng = np.random.default_rng(regime_seed // 100 * 100)
    fast_period = int(rng.integers(cfg.fast_period_min, cfg.fast_period_max + 1))
    if condition == "long_dwell_ood":
        slow_dwell = int(
            rng.integers(
                math.ceil(1.5 * cfg.slow_dwell_max),
                math.floor(2.0 * cfg.slow_dwell_max) + 1,
            )
        )
    else:
        slow_dwell = int(rng.integers(cfg.slow_dwell_min, cfg.slow_dwell_max + 1))
    alias = int(cfg.burn_in)
    phase_variant = (
        1 - int(variant)
        if condition == "heldout_phase_direction"
        else int(variant)
    )
    hazard_edge = short_edge if int(variant) == 0 else long_edge
    safe_edge = long_edge if int(variant) == 0 else short_edge
    if stratum == "slow_gate_phase":
        # Both variants are visibly open at the alias point, but their seeded
        # phases imply different time-to-transition.  The held-out slice swaps
        # this phase/direction correlation without changing map or route mode.
        transition_offset = 1 if phase_variant == 0 else min(8, max(4, fast_period // 2 + 1))
        hazard_start = alias + transition_offset
        slow_pulse_width = max(2, min(4, fast_period // 3))
        hazard_end = min(
            cfg.episode_steps + cfg.forecast_horizon,
            hazard_start + slow_pulse_width - 1,
        )
    else:
        hazard_start = alias + 1
        hazard_end = min(
            cfg.episode_steps + cfg.forecast_horizon,
            hazard_start + max(2, fast_period // 2),
        )
    base_gate_phase = (
        (slow_dwell - 2) if phase_variant == 0 else max(1, slow_dwell // 3)
    )
    regime = LatentRegime(
        stratum=stratum,
        variant=int(variant),
        direction=1 if variant == 0 else -1,
        gate_phase=base_gate_phase,
        route_mode=1 if variant == 0 else -1,
    )
    schedule = RegimeSchedule(
        fast_period=fast_period,
        slow_dwell=slow_dwell,
        alias_time=alias,
        hazard_edge=canonical_edge(hazard_edge),
        safe_edge=canonical_edge(safe_edge),
        hazard_start=hazard_start,
        hazard_end=hazard_end,
    )

    total_steps = cfg.episode_steps + cfg.forecast_horizon + 1
    hazard_mid = 0.5 * (rm.points[hazard_edge[0]] + rm.points[hazard_edge[1]])
    other_edge = safe_edge
    other_mid = 0.5 * (rm.points[other_edge[0]] + rm.points[other_edge[1]])
    common = 0.5 * (hazard_mid + other_mid)
    radius = max(0.008 * world.side_len, min(0.025 * world.side_len, 0.15 * np.linalg.norm(hazard_mid - other_mid)))
    radii = np.asarray([radius], dtype=np.float64)
    centers = np.repeat(common[None, None, :], total_steps, axis=0)

    if stratum in ("direction_alias", CONTROL_STRATUM):
        # The variants occupy exactly the same point at the alias time but
        # arrive with opposite counterfactual velocities and continue toward
        # different branch edges.
        delta = hazard_mid - common
        for k in range(1, min(8, alias) + 1):
            centers[alias - k, 0] = common - delta
        centers[alias, 0] = common
        # The fast process recurs throughout the episode.  The first pulse
        # preserves G0's 8-step divergence while later pulses provide genuine
        # route-critical support in horizons 17--32.
        pulse_width = hazard_end - hazard_start + 1
        for future_t in range(hazard_start, total_steps):
            if (future_t - hazard_start) % fast_period < pulse_width:
                centers[future_t, 0] = hazard_mid
    elif stratum == "route_mode_junction":
        # A one-frame route cue occurs at t=0.  At the alias decision the
        # registered 16-frame Transformer has necessarily dropped it (the
        # online buffer contains t=1..16), while persistent cores can retain
        # it.  A short immediate pulse preserves the G0 eight-step divergence;
        # subsequent route-mode pulses follow the slow dwell schedule so the
        # second event is genuinely outside a short temporal window.
        centers[0, 0] = hazard_mid
        centers[1 : alias + 1, 0] = common
        pulse_width = max(2, min(4, fast_period // 3))
        first_pulse = hazard_start
        second_pulse = alias + max(17, slow_dwell // 2)
        pulse_starts = [first_pulse]
        next_pulse = second_pulse
        while next_pulse < total_steps:
            pulse_starts.append(next_pulse)
            next_pulse += slow_dwell
        for pulse_start in pulse_starts:
            centers[pulse_start : min(total_steps, pulse_start + pulse_width), 0] = hazard_mid
    else:  # slow_gate_phase
        centers[:, 0] = common

    gate_edges = (canonical_edge(short_edge), canonical_edge(long_edge))
    gate_open = np.ones((total_steps, len(gate_edges)), dtype=np.bool_)
    if stratum == "slow_gate_phase":
        j = gate_edges.index(canonical_edge(hazard_edge))
        second_offset = max(17, slow_dwell // 2)
        if condition != "long_dwell_ood":
            second_offset = min(29, second_offset)
        second_pulse = alias + second_offset + (3 if phase_variant == 1 else 0)
        pulse_starts = [hazard_start]
        next_pulse = second_pulse
        while next_pulse < total_steps:
            pulse_starts.append(next_pulse)
            next_pulse += slow_dwell
        for pulse_start in pulse_starts:
            gate_open[
                pulse_start : min(total_steps, pulse_start + slow_pulse_width), j
            ] = False
        # A visible earlier transition is the history cue.  Both variants are
        # open at the alias instant, but the location/timing of the earlier
        # closed interval identifies the future phase.
        # The phase cue is intentionally one frame at t=0.  It is available to
        # persistent state during burn-in but absent from the Transformer's
        # exact 16-frame window at the t=16 alias decision.
        gate_open[0, j] = False
    gates = GateSchedule(gate_edges, gate_open)
    return regime, schedule, centers, radii, gates, canonical_edge(safe_edge)


def _make_pair(
    stratum: str,
    split: str,
    pair_index: int,
    map_family: str,
    world: C.World,
    rm: C.Roadmap,
    short_edge: EdgeId,
    long_edge: EdgeId,
    route_lengths: Tuple[float, float],
    cfg: C12DynamicsConfig,
    map_seed: int,
    goal_seed: int,
    v_agent: float,
) -> Tuple[C12EpisodeSpec, C12EpisodeSpec]:
    if stratum not in STRATA:
        raise KeyError(f"unknown C12 stratum: {stratum!r}")
    pair_id = f"{split.lower()}-{stratum}-{pair_index:06d}"
    condition = evaluation_condition(split, pair_index, map_family)
    out: List[C12EpisodeSpec] = []
    for variant in (0, 1):
        regime_seed = component_seed(split, stratum, pair_index, "regime", variant)
        regime, schedule, centers, radii, gates, action_hint = _build_tables(
            stratum,
            variant,
            world,
            rm,
            short_edge,
            long_edge,
            cfg,
            regime_seed,
            condition,
        )
        dyn = LatentDynamics(
            world,
            rm,
            cfg,
            regime,
            schedule,
            centers,
            radii,
            gates,
            expose_regime=(stratum == CONTROL_STRATUM),
        )
        out.append(
            C12EpisodeSpec(
                stratum=stratum,
                split=split,
                pair_index=int(pair_index),
                pair_id=pair_id,
                map_family=map_family,
                map_seed=int(map_seed),
                goal_seed=int(goal_seed),
                regime_seed=int(regime_seed),
                world=world,
                roadmap=rm,
                start_node=0,
                goal_node=1,
                alias_time=cfg.burn_in,
                v_agent=float(v_agent),
                dt=float(cfg.dt),
                regime=regime,
                schedule=schedule,
                dynamics=dyn,
                oracle_first_action_hint=action_hint,
                route_lengths=route_lengths,
                diagnostics={
                    "short_edge": list(short_edge),
                    "long_edge": list(long_edge),
                    "route_ratio": float(route_lengths[1] / max(1e-12, route_lengths[0])),
                    "eval_condition": condition,
                    "is_long_dwell_ood": condition == "long_dwell_ood",
                    "is_scale_ood": condition == "scale_ood",
                    "is_heldout_combo": condition == "heldout_phase_direction",
                },
            )
        )
    return out[0], out[1]


def make_fixture_pair(
    stratum: str,
    pair_index: int = 0,
    cfg: Optional[C12DynamicsConfig] = None,
    split: str = "PROBE",
) -> Tuple[C12EpisodeSpec, C12EpisodeSpec]:
    cfg = cfg or C12DynamicsConfig()
    world, rm, short_edge, long_edge, lengths = _fixture_world_and_roadmap()
    map_seed = component_seed(split, stratum, pair_index, "map")
    goal_seed = component_seed(split, stratum, pair_index, "goal")
    return _make_pair(
        stratum,
        split,
        pair_index,
        "C12_fixture_diamond",
        world,
        rm,
        short_edge,
        long_edge,
        lengths,
        cfg,
        map_seed,
        goal_seed,
        cfg.default_v_agent,
    )


def build_challenge_pair(
    stratum: str,
    pair_index: int,
    map_family: str,
    cfg: Optional[C12DynamicsConfig] = None,
    split: str = "PROBE",
    max_retries: int = 200,
) -> Tuple[C12EpisodeSpec, C12EpisodeSpec]:
    """Construct a balanced counterfactual pair over a real C8 PRM family."""
    cfg = cfg or C12DynamicsConfig()
    if stratum not in STRATA:
        raise KeyError(f"unknown C12 stratum: {stratum!r}")
    import continuous_prm_c8_dynamic_maps as M8

    M8.install_c8_dynamic_maps()
    base_map_seed = component_seed(split, stratum, pair_index, "map")
    goal_seed = component_seed(split, stratum, pair_index, "goal")
    params = M8.dynamics_params(map_family)
    v_agent = float(params["v_agent"])
    for attempt in range(int(max_retries)):
        map_seed = base_map_seed + attempt * 7919
        built = M8.build_dynamic_world(map_family, map_seed)
        if built is None:
            continue
        world, _unused_c8_dynamics = built
        # The held-out side_len=2 rooms-large family is under-connected at
        # the 96-node in-distribution density for some seeds.  Preserve the
        # original 40-attempt prefix exactly; only after it is exhausted use
        # C8's established 192-node large-map density (see its solvability
        # tests) instead of rejecting a perfectly valid static world forever.
        large_fallback = map_family == "C_dyn_rooms_large" and attempt >= 40
        roadmap_cfg = C.RoadmapConfig(
            n_nodes=max(cfg.roadmap_nodes, 192) if large_fallback else cfg.roadmap_nodes,
            k_neighbors=max(cfg.roadmap_k, 8) if large_fallback else cfg.roadmap_k,
            sample_attempts_per_node=cfg.roadmap_attempts_per_node,
        )
        rm = C.build_prm(
            world,
            roadmap_cfg,
            seed=component_seed(split, stratum, pair_index, "roadmap") + attempt * 104729,
        )
        if rm is None or not rm.connected_to_goal[0]:
            continue
        selected = _select_connected_goal(world, rm, goal_seed)
        if selected is None:
            continue
        world, rm = selected
        extracted = _route_choice_subgraph(
            rm,
            seed=map_seed + 17,
            # G0-A uses a 32-step forecast.  Both route alternatives must be
            # reachable inside that single planning horizon; otherwise a
            # no-plan failure would measure horizon truncation, not memory.
            max_arrival_steps=min(
                cfg.episode_steps - cfg.burn_in - 8,
                # Leave ten steps of forecast slack for a transient hazard,
                # oracle waiting, and the conservative true-mode clearance
                # diagnostic.  The first smoke probe showed that accepting
                # 29--30 step static routes under a 32-step forecast produced
                # no-plan failures unrelated to partial observability.
                cfg.forecast_horizon - 10,
            ),
            v_agent=v_agent,
            dt=cfg.dt,
        )
        if extracted is None:
            continue
        reduced, short_edge, long_edge, lengths = extracted
        if not _branch_hazards_are_separated(
            reduced, short_edge, long_edge, world.side_len
        ):
            continue
        return _make_pair(
            stratum,
            split,
            pair_index,
            map_family,
            world,
            reduced,
            short_edge,
            long_edge,
            lengths,
            cfg,
            map_seed,
            goal_seed,
            v_agent,
        )
    raise RuntimeError(
        f"could not build C12 route-choice pair for {stratum}/{map_family}/"
        f"pair={pair_index} within {max_retries} retries"
    )


def config_hash(cfg: C12DynamicsConfig) -> str:
    payload = repr(
        (DYNAMICS_SCHEMA_VERSION, sorted(asdict(cfg).items()))
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


__all__ = [
    "CHALLENGE_STRATA",
    "DYNAMICS_SCHEMA_VERSION",
    "CONTROL_STRATUM",
    "STRATA",
    "SPLITS",
    "FORBIDDEN_MODEL_FIELDS",
    "ALIAS_ATOL",
    "C12DynamicsConfig",
    "LatentRegime",
    "RegimeSchedule",
    "GateSchedule",
    "LatentDynamicsState",
    "C12Observation",
    "C12EpisodeSpec",
    "C12EpisodeRecord",
    "LatentDynamics",
    "canonical_edge",
    "component_seed",
    "evaluation_condition",
    "audit_model_payload",
    "serialize_observation",
    "present_observation_key",
    "audit_alias_pair",
    "build_episode_record",
    "make_fixture_pair",
    "build_challenge_pair",
    "config_hash",
]
