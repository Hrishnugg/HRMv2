#!/usr/bin/env python3
"""C13 state-conditioned heuristic revalidation.

This module deliberately separates three objects that C6/C7 mixed together:

* a goal-proximity value (``constant - Euclidean``),
* an A* cost-to-go heuristic, and
* a learned, map-raster-conditioned residual field.

C13-A contains no shortest-path-derived training target.  Its non-trivial target
is one Bellman backup of Euclidean distance over the current node's one-hop PRM
actions.  Dijkstra remains available only for connectivity and evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import continuous_prm_common as C
import continuous_prm_c7_hard_maps as M7


TARGET_SOURCE = "one_step_local_euclidean_backup"
FORBIDDEN_INPUTS = (
    "occupancy_raster",
    "reachable_free_mask",
    "global_clearance_field",
    "start_channel",
    "world_descriptor",
    "global_obstacle_count",
    "global_free_fraction",
    "unbounded_goal_line_of_sight",
    "corridor_scan",
    "dist_to_goal",
    "shortest_path_result",
)


def parse_int_csv(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_float_csv(value: str) -> List[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return p


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    return p


@dataclass(frozen=True)
class LocalStateConfig:
    """The complete C13 runtime observation contract.

    Rays are physically bounded.  Successors are one-hop graph actions and are
    not radius-clipped: clipping the Bellman backup to a subset of outgoing
    edges can make it inadmissible relative to the full PRM.
    """

    sensor_radius_frac: float = 0.20
    num_rays: int = 16
    ray_steps: int = 24
    max_neighbors: int = 24
    token_dim: int = 16

    @property
    def seq_len(self) -> int:
        return 1 + int(self.num_rays) + int(self.max_neighbors)


@dataclass
class C13Config:
    mode: str = "audit"
    out_dir: str = "runs/c13_state"
    train_suites: str = "C_hard_maze,C_hard_rooms,C_hard_spiral"
    eval_suites: str = (
        "C_hard_maze,C_hard_maze_dense,C_hard_rooms,"
        "C_hard_spiral,C_hard_bugtrap,C_hard_rooms_large"
    )
    train_worlds: int = 96
    val_worlds: int = 24
    eval_worlds: int = 24
    train_nodes: int = 192
    density_nodes: str = "128,160,192,211,256"
    roadmap_k: int = 7
    budgets: str = "96,144,192"
    backup_depths: str = "0,1,2,4,8,16"
    seed: int = 1234
    max_world_retries: int = 200
    sensor_radius_frac: float = 0.20
    num_rays: int = 16
    ray_steps: int = 24
    max_neighbors: int = 24
    smoke_test: bool = False

    def state_cfg(self) -> LocalStateConfig:
        return LocalStateConfig(
            sensor_radius_frac=float(self.sensor_radius_frac),
            num_rays=int(self.num_rays),
            ray_steps=int(self.ray_steps),
            max_neighbors=int(self.max_neighbors),
        )


def apply_smoke_overrides(cfg: C13Config) -> C13Config:
    if not cfg.smoke_test:
        return cfg
    cfg.train_suites = "C_hard_maze"
    cfg.eval_suites = "C_hard_maze"
    cfg.train_worlds = 2
    cfg.val_worlds = 1
    cfg.eval_worlds = 2
    # Stay in C7's connected regime so smoke exercises target/search behavior.
    cfg.train_nodes = 192
    cfg.density_nodes = "192,211,256"
    cfg.roadmap_k = 7
    cfg.budgets = "96,144,192"
    cfg.num_rays = 8
    cfg.ray_steps = 8
    cfg.max_neighbors = 16
    return cfg


# ---------------------------------------------------------------------------
# Semantics controls
# ---------------------------------------------------------------------------


def euclidean_to_goal(points: np.ndarray, goal: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)
    return np.linalg.norm(pts - g[None, :], axis=1)


def proximity_constant(side_len: float) -> float:
    """A constant no smaller than any within-square Euclidean distance."""

    return math.sqrt(2.0) * float(side_len)


def goal_proximity_value(euclid: np.ndarray, side_len: float) -> np.ndarray:
    """Literal professor suggestion: larger values mean closer to the goal."""

    return proximity_constant(side_len) - np.asarray(euclid, dtype=np.float64)


def proximity_value_to_cost_rank(value: np.ndarray, side_len: float) -> np.ndarray:
    """Convert a maximized proximity value into the minimized rank convention."""

    return proximity_constant(side_len) - np.asarray(value, dtype=np.float64)


def literal_constant_residual_h(euclid: np.ndarray, side_len: float) -> np.ndarray:
    """What current C6 integration produces for residual = constant - E."""

    e = np.asarray(euclid, dtype=np.float64)
    residual = np.maximum(0.0, proximity_constant(side_len) - e)
    return e + residual


def semantics_audit(side_len: float = 1.0, n: int = 257) -> Dict[str, Any]:
    e = np.linspace(0.0, proximity_constant(side_len), int(n), dtype=np.float64)
    value = goal_proximity_value(e, side_len)
    rank = proximity_value_to_cost_rank(value, side_len)
    constant_h = literal_constant_residual_h(e, side_len)
    return {
        "side_len": float(side_len),
        "samples": int(n),
        "constant": proximity_constant(side_len),
        "max_rank_vs_euclid_error": float(np.max(np.abs(rank - e))),
        "constant_h_range": float(np.max(constant_h) - np.min(constant_h)),
        "interpretation": (
            "constant-E is a maximized proximity value; after orientation conversion it "
            "is Euclidean. Used as the current additive residual, it makes h constant."
        ),
    }


# ---------------------------------------------------------------------------
# One-step target (no shortest-path access)
# ---------------------------------------------------------------------------


def one_step_euclidean_backup(
    points: np.ndarray,
    adj: Sequence[Sequence[Tuple[int, float]]],
    goal_idx: int = 1,
) -> np.ndarray:
    """One Bellman backup of Euclidean over the current one-hop action set.

    This function intentionally accepts only points + adjacency.  It cannot read
    ``Roadmap.dist_to_goal`` and performs no recursive traversal.
    """

    pts = np.asarray(points, dtype=np.float64)
    if goal_idx < 0 or goal_idx >= len(pts):
        raise IndexError(f"goal_idx {goal_idx} outside roadmap of size {len(pts)}")
    e = euclidean_to_goal(pts, pts[goal_idx])
    out = e.copy()
    for u, nbrs in enumerate(adj):
        if u == goal_idx:
            out[u] = 0.0
            continue
        candidates = [float(w) + float(e[int(v)]) for v, w in nbrs]
        if candidates:
            # max(e, ...) removes only floating-point triangle-inequality noise.
            out[u] = max(float(e[u]), min(candidates))
    out[goal_idx] = 0.0
    return out

def bounded_euclidean_backup(
    points: np.ndarray,
    adj: Sequence[Sequence[Tuple[int, float]]],
    depth: int,
    goal_idx: int = 1,
) -> np.ndarray:
    """Apply a fixed number of Bellman backups starting from Euclidean.

    depth=0 is Euclidean and depth=1 is one_step_euclidean_backup. This is a
    diagnostic relaxation curve, not the C13-A training target: depths above
    one inspect successor-of-successor state and therefore test where useful
    guidance begins to require bounded graph traversal.
    """

    depth = int(depth)
    if depth < 0:
        raise ValueError(f"backup depth must be >= 0, got {depth}")
    pts = np.asarray(points, dtype=np.float64)
    if goal_idx < 0 or goal_idx >= len(pts):
        raise IndexError(f"goal_idx {goal_idx} outside roadmap of size {len(pts)}")
    euclid = euclidean_to_goal(pts, pts[goal_idx])
    h = euclid.copy()
    h[goal_idx] = 0.0
    for _ in range(depth):
        nxt = euclid.copy()
        for u, nbrs in enumerate(adj):
            if u == goal_idx:
                nxt[u] = 0.0
                continue
            candidates = [float(w) + float(h[int(v)]) for v, w in nbrs]
            if candidates:
                nxt[u] = max(float(euclid[u]), min(candidates))
        nxt[goal_idx] = 0.0
        h = nxt
    return h

def one_step_residual_target(
    points: np.ndarray,
    adj: Sequence[Sequence[Tuple[int, float]]],
    side_len: float,
    goal_idx: int = 1,
) -> np.ndarray:
    e = euclidean_to_goal(points, np.asarray(points, dtype=np.float64)[goal_idx])
    h1 = one_step_euclidean_backup(points, adj, goal_idx=goal_idx)
    return np.maximum(0.0, h1 - e) / max(C.EPS, float(side_len))


def one_step_property_audit(
    points: np.ndarray,
    adj: Sequence[Sequence[Tuple[int, float]]],
    oracle: Optional[np.ndarray] = None,
    goal_idx: int = 1,
) -> Dict[str, float]:
    e = euclidean_to_goal(points, np.asarray(points, dtype=np.float64)[goal_idx])
    h1 = one_step_euclidean_backup(points, adj, goal_idx=goal_idx)
    dominance_violation = float(np.max(np.maximum(0.0, e - h1)))
    consistency_violation = 0.0
    for u, nbrs in enumerate(adj):
        for v, w in nbrs:
            consistency_violation = max(
                consistency_violation,
                float(h1[u] - (float(w) + h1[int(v)])),
            )
    admissibility_violation = float("nan")
    if oracle is not None:
        d = np.asarray(oracle, dtype=np.float64)
        connected = np.isfinite(d) & (d < C.INF / 10.0)
        if connected.any():
            admissibility_violation = float(np.max(np.maximum(0.0, h1[connected] - d[connected])))
    return {
        "dominance_violation": max(0.0, dominance_violation),
        "consistency_violation": max(0.0, consistency_violation),
        "admissibility_violation": max(0.0, admissibility_violation)
        if math.isfinite(admissibility_violation)
        else float("nan"),
        "goal_abs": abs(float(h1[goal_idx])),
    }


# ---------------------------------------------------------------------------
# Bounded local-state features
# ---------------------------------------------------------------------------


def bounded_ray_distance(
    world: C.World,
    point: np.ndarray,
    angle: float,
    radius: float,
    steps: int,
) -> float:
    """Return free distance along a ray, never inspecting beyond ``radius``."""

    p = np.asarray(point, dtype=np.float64)
    unit = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
    radius = max(C.EPS, float(radius))
    steps = max(1, int(steps))
    last = 0.0
    for idx in range(1, steps + 1):
        dist = radius * idx / steps
        q = p + dist * unit
        if np.any(q < 0.0) or np.any(q > float(world.side_len)):
            return float(last)
        if not C.is_point_free(q, float(world.side_len), world.obstacles):
            return float(last)
        last = dist
    return float(radius)


def local_state_sequence(
    world: C.World,
    points: np.ndarray,
    adj: Sequence[Sequence[Tuple[int, float]]],
    node_idx: int,
    cfg: LocalStateConfig,
    goal_idx: int = 1,
) -> np.ndarray:
    """Encode only current geometry, bounded rays, and one-hop graph actions."""

    if int(cfg.token_dim) != 16:
        raise ValueError("C13 token schema currently requires token_dim=16")
    pts = np.asarray(points, dtype=np.float64)
    p = pts[int(node_idx)]
    goal = pts[int(goal_idx)]
    side = float(world.side_len)
    radius = max(C.EPS, float(cfg.sensor_radius_frac) * side)
    dx = float((goal[0] - p[0]) / side)
    dy = float((goal[1] - p[1]) / side)
    e_norm = float(np.linalg.norm(goal - p) / side)
    ray_distances = [
        bounded_ray_distance(
            world,
            p,
            2.0 * math.pi * ri / max(1, int(cfg.num_rays)),
            radius,
            int(cfg.ray_steps),
        )
        for ri in range(int(cfg.num_rays))
    ]
    nbrs = sorted(
        [(int(v), float(w)) for v, w in adj[int(node_idx)]],
        key=lambda rec: (rec[1], rec[0]),
    )
    improving = 0
    edge_lengths = []
    for v, w in nbrs:
        edge_lengths.append(w / side)
        if np.linalg.norm(pts[v] - goal) < np.linalg.norm(p - goal):
            improving += 1
    degree = len(nbrs)
    visible_goal = (
        1.0
        if np.linalg.norm(goal - p) <= radius
        and C.is_segment_free(p, goal, side, world.obstacles)
        else 0.0
    )
    clearance = min(radius, max(0.0, C.obstacle_clearance(p, side, world.obstacles))) / radius

    seq = np.zeros((cfg.seq_len, cfg.token_dim), dtype=np.float32)
    seq[0, :4] = [1.0, 0.0, 0.0, 0.0]
    seq[0, 4:16] = [
        dx,
        dy,
        e_norm,
        float(p[0] / side * 2.0 - 1.0),
        float(p[1] / side * 2.0 - 1.0),
        float(cfg.sensor_radius_frac),
        float(clearance),
        float(min(ray_distances) / radius if ray_distances else 1.0),
        float(min(1.0, degree / max(1, int(cfg.max_neighbors)))),
        float(np.mean(edge_lengths) if edge_lengths else 0.0),
        float(improving / max(1, degree)),
        float(visible_goal),
    ]

    for ri, distance in enumerate(ray_distances):
        angle = 2.0 * math.pi * ri / max(1, int(cfg.num_rays))
        row = 1 + ri
        seq[row, :4] = [0.0, 1.0, 0.0, 0.0]
        seq[row, 4:16] = [
            math.cos(angle),
            math.sin(angle),
            float(distance / radius),
            1.0 if distance < radius - 1.0e-9 else 0.0,
            dx,
            dy,
            e_norm,
            float(p[0] / side * 2.0 - 1.0),
            float(p[1] / side * 2.0 - 1.0),
            float(clearance),
            float(visible_goal),
            float(ri / max(1, int(cfg.num_rays) - 1)),
        ]

    base = 1 + int(cfg.num_rays)
    for rank, (v, w) in enumerate(nbrs[: int(cfg.max_neighbors)]):
        rel = pts[v] - p
        rel_norm = float(np.linalg.norm(rel))
        neigh_e = float(np.linalg.norm(pts[v] - goal))
        angle = math.atan2(float(rel[1]), float(rel[0]))
        row = base + rank
        seq[row, :4] = [0.0, 0.0, 1.0, 0.0]
        seq[row, 4:16] = [
            float(rel[0] / side),
            float(rel[1] / side),
            float(w / side),
            float(neigh_e / side),
            float((np.linalg.norm(p - goal) - neigh_e) / side),
            math.cos(angle),
            math.sin(angle),
            1.0 if v == goal_idx else 0.0,
            float(rank / max(1, int(cfg.max_neighbors) - 1)),
            float(min(1.0, degree / max(1, int(cfg.max_neighbors)))),
            float(rel_norm / side),
            float(max(0, degree - int(cfg.max_neighbors)) / max(1, degree)),
        ]
    return seq


def make_local_state_features(
    world: C.World,
    points: np.ndarray,
    adj: Sequence[Sequence[Tuple[int, float]]],
    cfg: LocalStateConfig,
    goal_idx: int = 1,
) -> np.ndarray:
    return np.stack(
        [local_state_sequence(world, points, adj, i, cfg, goal_idx=goal_idx) for i in range(len(points))],
        axis=0,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Deterministic world generation and leak-free dataset collection
# ---------------------------------------------------------------------------


def iter_worlds(
    spec: C.AnchorSpec,
    suite_idx: int,
    n_worlds: int,
    seed: int,
    min_start_goal_dist_frac: float = 0.45,
    retry: int = 200,
) -> Iterator[Tuple[int, C.World, int]]:
    valid = 0
    attempt = 0
    while valid < int(n_worlds) and attempt < int(n_worlds) * int(retry):
        attempt += 1
        world_seed = (
            int(seed)
            + 1_300_000
            + 1_000_003 * (int(suite_idx) + 1)
            + (valid + 1) * 7919
            + attempt
        )
        world = C.build_world(spec, world_seed, float(min_start_goal_dist_frac))
        if world is None:
            continue
        yield valid, world, world_seed
        valid += 1


def collect_state_dataset(
    cfg: C13Config,
    split: str,
    suites: Sequence[str],
    worlds_per_suite: int,
    seed_offset: int,
) -> Path:
    M7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    state_cfg = cfg.state_cfg()
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    e_norms: List[np.ndarray] = []
    world_ids: List[np.ndarray] = []
    records: List[Dict[str, Any]] = []
    global_world_id = 0
    for suite_idx, suite in enumerate(suites):
        if suite not in specs:
            raise KeyError(f"unknown suite {suite!r}; have {sorted(specs)}")
        spec = specs[suite]
        accepted = 0
        for world_idx, world, world_seed in iter_worlds(
            spec,
            suite_idx,
            int(worlds_per_suite) * int(cfg.max_world_retries),
            int(cfg.seed) + int(seed_offset),
            retry=cfg.max_world_retries,
        ):
            rm = C.build_prm(
                world,
                C.RoadmapConfig(n_nodes=int(cfg.train_nodes), k_neighbors=int(cfg.roadmap_k)),
                seed=world_seed + 17,
            )
            if rm is None:
                continue
            x = make_local_state_features(world, rm.points, rm.adj, state_cfg)
            y = one_step_residual_target(rm.points, rm.adj, world.side_len)
            e = euclidean_to_goal(rm.points, rm.points[1]) / float(world.side_len)
            xs.append(x)
            ys.append(y.astype(np.float32))
            e_norms.append(e.astype(np.float32))
            world_ids.append(np.full(len(rm.points), global_world_id, dtype=np.int32))
            records.append(
                {
                    "world_id": global_world_id,
                    "suite": suite,
                    "suite_world_index": accepted,
                    "world_seed": world_seed,
                    "roadmap_seed": world_seed + 17,
                    "nodes": len(rm.points),
                    "edges": sum(len(a) for a in rm.adj) // 2,
                    "positive_target_rate": float(np.mean(y > 1.0e-8)),
                    "target_mean": float(np.mean(y)),
                    "target_p95": float(np.quantile(y, 0.95)),
                }
            )
            global_world_id += 1
            accepted += 1
            if accepted >= int(worlds_per_suite):
                break
        if accepted < int(worlds_per_suite):
            raise RuntimeError(f"{split}/{suite} under-filled: {accepted}/{worlds_per_suite}")
    if not xs:
        raise RuntimeError(f"no C13 samples collected for split {split!r}")
    dataset_dir = ensure_dir(Path(cfg.out_dir) / "datasets")
    path = dataset_dir / f"c13_state_{split}.npz"
    np.savez_compressed(
        path,
        x=np.concatenate(xs, axis=0).astype(np.float32),
        y=np.concatenate(ys, axis=0).astype(np.float32),
        euclid_norm=np.concatenate(e_norms, axis=0).astype(np.float32),
        world_id=np.concatenate(world_ids, axis=0),
    )
    metadata = {
        "experiment": "C13-A",
        "split": split,
        "target_source": TARGET_SOURCE,
        "shortest_path_target": False,
        "recursive_graph_traversal_in_target": False,
        "dijkstra_role": "connectivity_and_evaluation_only",
        "runtime_scope": "current_geometry_bounded_rays_one_hop_actions",
        "forbidden_inputs": list(FORBIDDEN_INPUTS),
        "state_config": asdict(state_cfg),
        "roadmap_nodes": int(cfg.train_nodes),
        "roadmap_k": int(cfg.roadmap_k),
        "suites": list(suites),
        "worlds": records,
        "n_samples": int(sum(len(y) for y in ys)),
    }
    write_json(path.with_suffix(".metadata.json"), metadata)
    return path


# ---------------------------------------------------------------------------
# Density audit
# ---------------------------------------------------------------------------


def _finite_oracle(rm: C.Roadmap, side_len: float) -> np.ndarray:
    d = np.asarray(rm.dist_to_goal, dtype=np.float64).copy()
    connected = np.isfinite(d) & (d < C.INF / 10.0)
    fill = float(np.max(d[connected]) + side_len) if connected.any() else float(10.0 * side_len)
    d[~connected] = fill
    return d


def _search_record(
    common: Dict[str, Any],
    provider: str,
    heuristic: np.ndarray,
    rm: C.Roadmap,
    budget: int,
    optimal: float,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = C.astar_search(rm.adj, heuristic, int(budget))
    elapsed = time.perf_counter() - t0
    cost = float(result["cost"])
    return {
        **common,
        "provider": provider,
        "mode": "astar",
        "budget": int(budget),
        "found": bool(result["found"]),
        "expansions": int(result["expansions"]),
        "closed": int(result["closed"]),
        "cost": cost if math.isfinite(cost) else "",
        "cost_ratio": cost / optimal if math.isfinite(cost) and optimal > C.EPS else "",
        "search_seconds": elapsed,
    }


def run_density_audit(cfg: C13Config) -> Dict[str, Path]:
    M7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    nodes_grid = parse_int_csv(cfg.density_nodes)
    budgets = parse_int_csv(cfg.budgets)
    rows: List[Dict[str, Any]] = []
    property_rows: List[Dict[str, Any]] = []
    for suite_idx, suite in enumerate(parse_csv(cfg.eval_suites)):
        if suite not in specs:
            raise KeyError(f"unknown suite {suite!r}; have {sorted(specs)}")
        spec = specs[suite]
        for world_idx, world, world_seed in iter_worlds(
            spec,
            suite_idx,
            cfg.eval_worlds,
            cfg.seed + 90_000,
            retry=cfg.max_world_retries,
        ):
            for requested_nodes in nodes_grid:
                roadmap_cfg = C.RoadmapConfig(
                    n_nodes=int(requested_nodes),
                    k_neighbors=int(cfg.roadmap_k),
                )
                t0 = time.perf_counter()
                rm = C.build_prm(world, roadmap_cfg, seed=world_seed + 17)
                build_seconds = time.perf_counter() - t0
                base = {
                    "suite": suite,
                    "world_index": world_idx,
                    "world_seed": world_seed,
                    "roadmap_seed": world_seed + 17,
                    "requested_nodes": int(requested_nodes),
                    "roadmap_k": int(cfg.roadmap_k),
                    "side_len": float(world.side_len),
                    "requested_nodes_per_area": float(requested_nodes / (world.side_len**2)),
                    "build_seconds": build_seconds,
                }
                if rm is None:
                    for budget in budgets:
                        rows.append(
                            {
                                **base,
                                "actual_nodes": 0,
                                "edges": 0,
                                "connected": False,
                                "provider": "build_failure",
                                "mode": "astar",
                                "budget": int(budget),
                                "found": False,
                                "expansions": 0,
                                "closed": 0,
                                "cost": "",
                                "cost_ratio": "",
                                "search_seconds": 0.0,
                            }
                        )
                    continue
                e = euclidean_to_goal(rm.points, rm.points[1])
                h1 = one_step_euclidean_backup(rm.points, rm.adj)
                oracle = _finite_oracle(rm, world.side_len)
                constant_h = literal_constant_residual_h(e, world.side_len)
                optimal = float(rm.dist_to_goal[0])
                edges = sum(len(a) for a in rm.adj) // 2
                common = {
                    **base,
                    "actual_nodes": int(len(rm.points)),
                    "actual_nodes_per_area": float(len(rm.points) / (world.side_len**2)),
                    "edges": int(edges),
                    "mean_degree": float(2.0 * edges / max(1, len(rm.points))),
                    "connected": True,
                    "optimal_cost_eval_only": optimal,
                }
                props = one_step_property_audit(rm.points, rm.adj, oracle=rm.dist_to_goal)
                property_rows.append({**common, **props})
                for budget in budgets:
                    rows.append(_search_record(common, "euclid", e, rm, budget, optimal))
                    rows.append(_search_record(common, "one_step", h1, rm, budget, optimal))
                    rows.append(_search_record(common, "constant_cancel", constant_h, rm, budget, optimal))
                    rows.append(_search_record(common, "oracle_eval_only", oracle, rm, budget, optimal))
    results_dir = ensure_dir(Path(cfg.out_dir) / "results")
    raw = write_csv(results_dir / "c13_density_audit_raw.csv", rows)
    props = write_csv(results_dir / "c13_one_step_properties.csv", property_rows)
    semantic_path = write_json(results_dir / "c13_semantics_audit.json", semantics_audit())
    manifest = write_json(
        Path(cfg.out_dir) / "audit_manifest.json",
        {
            "experiment": "C13-A",
            "config": asdict(cfg),
            "target_source": TARGET_SOURCE,
            "shortest_path_target": False,
            "dijkstra_role": "connectivity_and_evaluation_only",
            "density_nodes": nodes_grid,
            "contains_plus_10_percent_of_192": 211 in nodes_grid,
            "raw": str(raw),
            "properties": str(props),
            "semantics": str(semantic_path),
        },
    )
    return {"raw": raw, "properties": props, "semantics": semantic_path, "manifest": manifest}


# ---------------------------------------------------------------------------
def run_backup_depth_audit(cfg: C13Config) -> Dict[str, Path]:
    """Measure how much search guidance appears at each fixed backup depth.

    Depths above one are diagnostics only. They deliberately expose how quickly
    a bounded relaxation approaches the evaluation oracle and therefore where a
    nominally local method starts to behave like graph traversal.
    """

    M7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    depths = sorted(set(parse_int_csv(cfg.backup_depths)))
    if not depths or depths[0] < 0:
        raise ValueError(f"invalid backup depths: {depths}")
    requested_budgets = parse_int_csv(cfg.budgets)
    rows: List[Dict[str, Any]] = []
    for suite_idx, suite in enumerate(parse_csv(cfg.eval_suites)):
        if suite not in specs:
            raise KeyError(f"unknown suite {suite!r}; have {sorted(specs)}")
        spec = specs[suite]
        accepted = 0
        candidate_worlds = int(cfg.eval_worlds) * int(cfg.max_world_retries)
        for _, world, world_seed in iter_worlds(
            spec,
            suite_idx,
            candidate_worlds,
            cfg.seed + 190_000,
            retry=cfg.max_world_retries,
        ):
            t_build = time.perf_counter()
            rm = C.build_prm(
                world,
                C.RoadmapConfig(n_nodes=int(cfg.train_nodes), k_neighbors=int(cfg.roadmap_k)),
                seed=world_seed + 17,
            )
            build_seconds = time.perf_counter() - t_build
            if rm is None:
                continue
            optimal = float(rm.dist_to_goal[0])
            if not math.isfinite(optimal) or optimal >= C.INF / 10.0:
                continue
            oracle_raw = np.asarray(rm.dist_to_goal, dtype=np.float64)
            connected = np.isfinite(oracle_raw) & (oracle_raw < C.INF / 10.0)
            euclid = euclidean_to_goal(rm.points, rm.points[1])
            budget = max([len(rm.points), *requested_budgets])
            edges = sum(len(a) for a in rm.adj) // 2
            for depth in depths:
                t_h = time.perf_counter()
                h = bounded_euclidean_backup(rm.points, rm.adj, depth=depth)
                heuristic_seconds = time.perf_counter() - t_h
                dominance = float(np.max(np.maximum(0.0, euclid - h)))
                admissibility = float(np.max(np.maximum(0.0, h[connected] - oracle_raw[connected])))
                consistency = 0.0
                for u, nbrs in enumerate(rm.adj):
                    for v, w in nbrs:
                        consistency = max(consistency, float(h[u] - (float(w) + h[int(v)])))
                t_search = time.perf_counter()
                result = C.astar_search(rm.adj, h, budget)
                search_seconds = time.perf_counter() - t_search
                cost = float(result["cost"])
                rows.append(
                    {
                        "suite": suite,
                        "world_index": accepted,
                        "world_seed": world_seed,
                        "roadmap_seed": world_seed + 17,
                        "nodes": len(rm.points),
                        "edges": edges,
                        "roadmap_k": int(cfg.roadmap_k),
                        "depth": int(depth),
                        "state_scope": (
                            "strict_geometry"
                            if depth == 0
                            else "one_hop_current_actions"
                            if depth == 1
                            else f"bounded_{depth}_hop_traversal"
                        ),
                        "allowed_as_c13a_target": depth == 1,
                        "build_seconds": build_seconds,
                        "heuristic_seconds": heuristic_seconds,
                        "budget": budget,
                        "found": bool(result["found"]),
                        "expansions": int(result["expansions"]),
                        "cost": cost if math.isfinite(cost) else "",
                        "cost_ratio": cost / optimal if math.isfinite(cost) else "",
                        "start_h_over_optimal": float(h[0] / optimal),
                        "mean_oracle_gap_connected": float(np.mean(oracle_raw[connected] - h[connected])),
                        "dominance_violation": max(0.0, dominance),
                        "admissibility_violation": max(0.0, admissibility),
                        "consistency_violation": max(0.0, consistency),
                        "search_seconds": search_seconds,
                    }
                )
            accepted += 1
            if accepted >= int(cfg.eval_worlds):
                break
        if accepted < int(cfg.eval_worlds):
            raise RuntimeError(f"relaxation/{suite} under-filled: {accepted}/{cfg.eval_worlds}")
    results_dir = ensure_dir(Path(cfg.out_dir) / "results")
    raw = write_csv(results_dir / "c13_backup_depth_audit.csv", rows)
    manifest = write_json(
        Path(cfg.out_dir) / "backup_depth_manifest.json",
        {
            "experiment": "C13 target-selection diagnostic",
            "config": asdict(cfg),
            "depths": depths,
            "shortest_path_training_target": False,
            "dijkstra_role": "evaluation_only",
            "interpretation": (
                "depth 0 is Euclidean; depth 1 is the proposed C13-A state target; "
                "depths >1 quantify gains that require successor-of-successor traversal"
            ),
            "raw": str(raw),
        },
    )
    return {"raw": raw, "manifest": manifest}


# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C13 state-conditioned heuristic and density audit")
    p.add_argument("--mode", choices=("semantic", "collect", "audit", "relaxation", "full"), default="audit")
    p.add_argument("--out-dir", default="runs/c13_state")
    p.add_argument("--train-suites", default=C13Config.train_suites)
    p.add_argument("--eval-suites", default=C13Config.eval_suites)
    p.add_argument("--train-worlds", type=int, default=96)
    p.add_argument("--val-worlds", type=int, default=24)
    p.add_argument("--eval-worlds", type=int, default=24)
    p.add_argument("--train-nodes", type=int, default=192)
    p.add_argument("--density-nodes", default="128,160,192,211,256")
    p.add_argument("--roadmap-k", type=int, default=7)
    p.add_argument("--budgets", default="96,144,192")
    p.add_argument("--backup-depths", default="0,1,2,4,8,16")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-world-retries", type=int, default=200)
    p.add_argument("--sensor-radius-frac", type=float, default=0.20)
    p.add_argument("--num-rays", type=int, default=16)
    p.add_argument("--ray-steps", type=int, default=24)
    p.add_argument("--max-neighbors", type=int, default=24)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> C13Config:
    return apply_smoke_overrides(
        C13Config(
            mode=str(args.mode),
            out_dir=str(args.out_dir),
            train_suites=str(args.train_suites),
            eval_suites=str(args.eval_suites),
            train_worlds=int(args.train_worlds),
            val_worlds=int(args.val_worlds),
            eval_worlds=int(args.eval_worlds),
            train_nodes=int(args.train_nodes),
            density_nodes=str(args.density_nodes),
            roadmap_k=int(args.roadmap_k),
            budgets=str(args.budgets),
            backup_depths=str(args.backup_depths),
            seed=int(args.seed),
            max_world_retries=int(args.max_world_retries),
            sensor_radius_frac=float(args.sensor_radius_frac),
            num_rays=int(args.num_rays),
            ray_steps=int(args.ray_steps),
            max_neighbors=int(args.max_neighbors),
            smoke_test=bool(args.smoke_test),
        )
    )


def main() -> None:
    cfg = config_from_args(parse_args())
    if cfg.mode == "semantic":
        path = write_json(Path(cfg.out_dir) / "results" / "c13_semantics_audit.json", semantics_audit())
        print(path)
        return
    if cfg.mode in {"collect", "full"}:
        train = collect_state_dataset(
            cfg,
            "train",
            parse_csv(cfg.train_suites),
            cfg.train_worlds,
            seed_offset=0,
        )
        val = collect_state_dataset(
            cfg,
            "val",
            parse_csv(cfg.train_suites),
            cfg.val_worlds,
            seed_offset=500_000,
        )
        print(f"train={train}\nval={val}")
    if cfg.mode in {"relaxation", "full"}:
        outputs = run_backup_depth_audit(cfg)
        for key, value in outputs.items():
            print(f"{key}={value}")

    if cfg.mode in {"audit", "full"}:
        outputs = run_density_audit(cfg)
        for key, value in outputs.items():
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
