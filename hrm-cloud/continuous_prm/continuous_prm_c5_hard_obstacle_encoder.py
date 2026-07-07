#!/usr/bin/env python3
"""
C5: hard-map continuous PRM experiment with a richer obstacle encoder.

This script intentionally keeps C1-C4 unchanged. It installs runtime-only
extensions over continuous_prm_common before delegating to the existing staged
runner:

- new hard maze/room suites with staggered wall gates and clutter;
- start/goal placement that forces long detours through alternating gates;
- a richer feature sequence that splits the ray-token budget into exact rays
  plus angular obstacle-sector summary tokens.

Default target: calibrate Euclidean A* into the 50-70% success band at low
budgets, then test whether HRM/ONLSTM learned residual heuristics recover.
"""

from __future__ import annotations

import math
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import continuous_prm_common as C


HARD_MODE = "hard_maze"
HARD_TRAIN_TASKS = "C_hard_maze,C_hard_rooms"
HARD_EVAL_SUITES = "C_hard_maze,C_hard_maze_dense,C_hard_rooms"
DEFAULT_SECTOR_TOKENS = 16

_ORIG_BUILD_ANCHOR_SPECS = C.build_anchor_specs
_ORIG_GENERATE_OBSTACLES = C.generate_obstacles
_ORIG_BUILD_WORLD = C.build_world
_ORIG_TASK_DESCRIPTOR = C.task_descriptor
_ORIG_APPLY_SMOKE_OVERRIDES = C.apply_smoke_overrides
_ORIG_MODEL_FORWARD = C.ContinuousHeuristicModel.forward

SECTOR_TOKENS = DEFAULT_SECTOR_TOKENS
SOFT_RESIDUAL_CAP = False


def _has_arg(flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])


def _ensure_default_arg(flag: str, value: str) -> None:
    if not _has_arg(flag):
        sys.argv.extend([flag, value])


def _pop_int_arg(flag: str, default: int) -> int:
    value = default
    out = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == flag:
            if i + 1 >= len(sys.argv):
                raise ValueError(f"{flag} requires an integer value")
            value = int(sys.argv[i + 1])
            i += 2
            continue
        if arg.startswith(flag + "="):
            value = int(arg.split("=", 1)[1])
            i += 1
            continue
        out.append(arg)
        i += 1
    sys.argv[:] = out
    return value


def _pop_bool_arg(flag: str, default: bool = False) -> bool:
    value = default
    out = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == flag:
            value = True
            i += 1
            continue
        if arg.startswith(flag + "="):
            text = arg.split("=", 1)[1].strip().lower()
            value = text in ("1", "true", "yes", "on")
            i += 1
            continue
        out.append(arg)
        i += 1
    sys.argv[:] = out
    return value


def configure_default_argv() -> None:
    global SECTOR_TOKENS, SOFT_RESIDUAL_CAP
    SECTOR_TOKENS = max(0, _pop_int_arg("--sector-tokens", DEFAULT_SECTOR_TOKENS))
    SOFT_RESIDUAL_CAP = _pop_bool_arg("--soft-residual-cap", False)
    _ensure_default_arg("--out-dir", "runs/continuous_prm_c5_hard_obstacle_encoder")
    _ensure_default_arg("--train-tasks", HARD_TRAIN_TASKS)
    _ensure_default_arg("--eval-suites", HARD_EVAL_SUITES)
    _ensure_default_arg("--budgets", "32,64,96,128,192")
    _ensure_default_arg("--roadmap-nodes", "192")
    _ensure_default_arg("--roadmap-k", "7")
    _ensure_default_arg("--nearest-obstacles", "12")
    _ensure_default_arg("--num-rays", "48")
    _ensure_default_arg("--ray-steps", "96")
    _ensure_default_arg("--train-worlds", "180")
    _ensure_default_arg("--nodes-per-world", "192")
    _ensure_default_arg("--eval-worlds", "60")
    _ensure_default_arg("--base-epochs", "12")
    _ensure_default_arg("--expert-epochs", "10")


def build_hard_anchor_specs() -> Dict[str, C.AnchorSpec]:
    specs = _ORIG_BUILD_ANCHOR_SPECS()
    specs.update({
        "C_hard_maze": C.AnchorSpec(
            name="C_hard_maze",
            side_len=1.0,
            mode=HARD_MODE,
            obstacle_count_range=(10, 20),
            radius_range=(0.018, 0.045),
            gap_width_frac=0.115,
            barrier_radius_frac=0.028,
            extra_clutter_range=(10, 20),
            rectangle_count_range=(3, 3),
        ),
        "C_hard_maze_dense": C.AnchorSpec(
            name="C_hard_maze_dense",
            side_len=1.0,
            mode=HARD_MODE,
            obstacle_count_range=(18, 32),
            radius_range=(0.018, 0.050),
            gap_width_frac=0.085,
            barrier_radius_frac=0.030,
            extra_clutter_range=(18, 32),
            rectangle_count_range=(4, 4),
            is_ood=True,
        ),
        "C_hard_rooms": C.AnchorSpec(
            name="C_hard_rooms",
            side_len=2.0,
            mode=HARD_MODE,
            obstacle_count_range=(24, 44),
            radius_range=(0.035, 0.080),
            gap_width_frac=0.095,
            barrier_radius_frac=0.022,
            extra_clutter_range=(24, 44),
            rectangle_count_range=(5, 5),
        ),
    })
    return specs


def _add_random_circles(spec: C.AnchorSpec, rng: random.Random, obstacles: List[C.Obstacle], n: int) -> None:
    side = spec.side_len
    target = len([o for o in obstacles if o.kind == "circle"]) + int(n)
    attempts = 0
    while len([o for o in obstacles if o.kind == "circle"]) < target and attempts < max(1, n) * 300:
        attempts += 1
        r = rng.uniform(spec.radius_range[0], spec.radius_range[1])
        cx = rng.uniform(r + 0.015 * side, side - r - 0.015 * side)
        cy = rng.uniform(r + 0.015 * side, side - r - 0.015 * side)
        ok = True
        for obs in obstacles:
            if np.linalg.norm(np.array([cx - obs.cx, cy - obs.cy])) < (r + obs.bounding_radius()) * 0.58:
                ok = False
                break
        if ok:
            obstacles.append(C.Obstacle("circle", cx, cy, radius=r))


def generate_hard_obstacles(spec: C.AnchorSpec, rng: random.Random) -> List[C.Obstacle]:
    if spec.mode != HARD_MODE:
        return _ORIG_GENERATE_OBSTACLES(spec, rng)
    side = spec.side_len
    obstacles: List[C.Obstacle] = []
    n_walls = max(2, int(spec.rectangle_count_range[0]))
    wall_thick = max(0.010 * side, spec.barrier_radius_frac * side)
    gap_width = max(0.040 * side, spec.gap_width_frac * side)
    xs = np.linspace(0.20 * side, 0.80 * side, n_walls)
    for wi, x in enumerate(xs):
        base_gap = rng.uniform(0.18 * side, 0.32 * side) if wi % 2 == 0 else rng.uniform(0.68 * side, 0.82 * side)
        gap_center = min(0.88 * side, max(0.12 * side, base_gap + rng.uniform(-0.035 * side, 0.035 * side)))
        gap_lo = max(0.0, gap_center - 0.5 * gap_width)
        gap_hi = min(side, gap_center + 0.5 * gap_width)
        xj = float(x + rng.uniform(-0.012 * side, 0.012 * side))
        if gap_lo > 0.055 * side:
            hh = 0.5 * gap_lo
            obstacles.append(C.Obstacle("rect", xj, hh, hw=0.5 * wall_thick, hh=hh))
        if side - gap_hi > 0.055 * side:
            h = side - gap_hi
            obstacles.append(C.Obstacle("rect", xj, gap_hi + 0.5 * h, hw=0.5 * wall_thick, hh=0.5 * h))
    clutter = rng.randint(*spec.extra_clutter_range)
    _add_random_circles(spec, rng, obstacles, clutter)
    return obstacles


def build_hard_world(spec: C.AnchorSpec, seed: int, min_start_goal_dist_frac: float) -> Optional[C.World]:
    if spec.mode != HARD_MODE:
        return _ORIG_BUILD_WORLD(spec, seed, min_start_goal_dist_frac)
    rng = random.Random(seed)
    side = spec.side_len
    obstacles = C.generate_obstacles(spec, rng)
    left = (0.03 * side, 0.15 * side, 0.40 * side, 0.60 * side)
    right = (0.85 * side, 0.97 * side, 0.40 * side, 0.60 * side)
    start_region, goal_region = (left, right) if rng.random() < 0.5 else (right, left)
    for _ in range(180):
        start = C.sample_free_point(rng, side, obstacles, region=start_region, max_attempts=3000)
        goal = C.sample_free_point(rng, side, obstacles, region=goal_region, max_attempts=3000)
        if start is None or goal is None:
            continue
        if float(np.linalg.norm(goal - start)) < min_start_goal_dist_frac * side:
            continue
        if not C.is_point_free(start, side, obstacles) or not C.is_point_free(goal, side, obstacles):
            continue
        desc = C.task_descriptor(spec, obstacles)
        return C.World(
            spec_name=spec.name,
            side_len=side,
            obstacles=obstacles,
            start=start,
            goal=goal,
            descriptor=desc,
            meta={
                "seed": seed,
                "mode": spec.mode,
                "obstacle_count": len(obstacles),
                "free_fraction_est": C.approximate_free_fraction(side, obstacles),
                "hard_wall_count": int(spec.rectangle_count_range[0]),
                "hard_gap_width_frac": float(spec.gap_width_frac),
            },
        )
    return None


def hard_task_descriptor(spec: C.AnchorSpec, obstacles: Sequence[C.Obstacle]) -> np.ndarray:
    if spec.mode != HARD_MODE:
        return _ORIG_TASK_DESCRIPTOR(spec, obstacles)
    side = spec.side_len
    n_obs = len(obstacles)
    mean_r = float(np.mean([o.bounding_radius() for o in obstacles])) if obstacles else 0.0
    density = sum(o.area() for o in obstacles) / max(C.EPS, side * side)
    free_frac = C.approximate_free_fraction(side, obstacles)
    rect_frac = float(sum(1 for o in obstacles if o.kind == "rect")) / max(1.0, float(n_obs))
    return np.array([
        math.log(max(C.EPS, side)),
        min(1.0, n_obs / 100.0),
        min(1.0, mean_r / max(C.EPS, side) / 0.20),
        min(1.0, density),
        free_frac,
        1.0,
        rect_frac,
        1.0 if spec.is_ood else 0.0,
    ], dtype=np.float32)


def obstacle_sector_records(p: np.ndarray, world: C.World, sectors: int) -> List[Dict[str, float]]:
    buckets: List[List[Tuple[C.Obstacle, float, float]]] = [[] for _ in range(max(0, int(sectors)))]
    side = world.side_len
    for obs in world.obstacles:
        relx = float(obs.cx - p[0])
        rely = float(obs.cy - p[1])
        angle = (math.atan2(rely, relx) + 2.0 * math.pi) % (2.0 * math.pi)
        idx = min(len(buckets) - 1, int(angle / (2.0 * math.pi) * len(buckets))) if buckets else 0
        if not buckets:
            break
        if obs.kind == "circle":
            clearance = math.sqrt(relx * relx + rely * rely) - obs.radius
        else:
            center = np.array([obs.cx, obs.cy], dtype=np.float64)
            clearance = float(np.linalg.norm(p - center)) - obs.bounding_radius()
        buckets[idx].append((obs, float(clearance / max(C.EPS, side)), float(obs.bounding_radius() / max(C.EPS, side))))
    out: List[Dict[str, float]] = []
    for bucket in buckets:
        if not bucket:
            out.append({"count": 0.0, "min_clearance": 1.5, "mean_clearance": 1.5, "mean_radius": 0.0, "rect_frac": 0.0})
            continue
        clearances = [b[1] for b in bucket]
        radii = [b[2] for b in bucket]
        out.append({
            "count": min(1.0, len(bucket) / 8.0),
            "min_clearance": float(max(-0.25, min(1.5, min(clearances)))),
            "mean_clearance": float(max(-0.25, min(1.5, float(np.mean(clearances))))),
            "mean_radius": float(min(1.0, float(np.mean(radii)) / 0.20)),
            "rect_frac": float(sum(1 for b in bucket if b[0].kind == "rect") / len(bucket)),
        })
    return out


def make_hard_feature_sequence(world: C.World, p: np.ndarray, feature_cfg: C.FeatureConfig) -> np.ndarray:
    side = world.side_len
    goal = world.goal
    token_dim = feature_cfg.token_dim
    sector_count = min(SECTOR_TOKENS, max(0, int(feature_cfg.num_rays) - 4))
    ray_count = max(4, int(feature_cfg.num_rays) - sector_count)
    seq = np.zeros((feature_cfg.seq_len, token_dim), dtype=np.float32)
    dx = (goal[0] - p[0]) / side
    dy = (goal[1] - p[1]) / side
    eu = float(np.linalg.norm(goal - p)) / side
    clearance = max(0.0, C.obstacle_clearance(p, side, world.obstacles)) / side
    los = 1.0 if C.is_segment_free(p, goal, side, world.obstacles) else 0.0
    corridor = C.corridor_blockage_score(p, goal, world)
    seq[0, 0:4] = [1.0, 0.0, 0.0, 0.0]
    seq[0, 4:16] = [
        float(dx), float(dy), float(eu),
        float(p[0] / side * 2.0 - 1.0), float(p[1] / side * 2.0 - 1.0),
        float(clearance), float(los), float(corridor),
        float(C.approximate_free_fraction(side, world.obstacles)),
        float(min(1.0, len(world.obstacles) / feature_cfg.max_obstacle_count_norm)),
        float(sum(1 for o in world.obstacles if o.kind == "rect") / max(1, len(world.obstacles))),
        1.0 if world.meta.get("mode") == HARD_MODE else 0.0,
    ]

    recs = C.nearest_obstacle_records(p, world, feature_cfg.nearest_obstacles)
    for oi in range(feature_cfg.nearest_obstacles):
        row = 1 + oi
        seq[row, 0:4] = [0.0, 1.0, 0.0, 0.0]
        if oi >= len(recs):
            continue
        obs, clearance_to_obs = recs[oi]
        relx = (obs.cx - p[0]) / side
        rely = (obs.cy - p[1]) / side
        br = obs.bounding_radius() / side
        dcenter = math.sqrt(relx * relx + rely * rely)
        angle = math.atan2(rely, relx)
        seq[row, 4:16] = [
            float(relx), float(rely), float(dcenter), float(br),
            float(clearance_to_obs / side),
            1.0 if obs.kind == "circle" else 0.0,
            1.0 if obs.kind == "rect" else 0.0,
            float(math.cos(angle)), float(math.sin(angle)),
            float(obs.hw / side if obs.kind == "rect" else 0.0),
            float(obs.hh / side if obs.kind == "rect" else 0.0),
            float(obs.area() / max(C.EPS, side * side)),
        ]

    ray_base = 1 + feature_cfg.nearest_obstacles
    for ri in range(ray_count):
        row = ray_base + ri
        angle = 2.0 * math.pi * ri / ray_count
        rd = C.raycast_distance(p, angle, world, feature_cfg.ray_steps) / side
        seq[row, 0:4] = [0.0, 0.0, 1.0, 0.0]
        seq[row, 4:16] = [
            float(math.cos(angle)), float(math.sin(angle)), float(min(1.5, rd)),
            float(ri / max(1, ray_count - 1)), float(dx), float(dy), float(eu),
            float(clearance), float(los), float(corridor), 0.0, 0.0,
        ]

    sector_base = ray_base + ray_count
    for si, rec in enumerate(obstacle_sector_records(p, world, sector_count)):
        row = sector_base + si
        angle = 2.0 * math.pi * (si + 0.5) / max(1, sector_count)
        seq[row, 0:4] = [0.0, 0.0, 1.0, 0.0]
        seq[row, 4:16] = [
            float(math.cos(angle)), float(math.sin(angle)),
            float(rec["count"]), float(rec["min_clearance"]), float(rec["mean_clearance"]),
            float(rec["mean_radius"]), float(rec["rect_frac"]),
            float(si / max(1, sector_count - 1)), float(dx), float(dy), float(eu), 1.0,
        ]

    task_row = feature_cfg.seq_len - 1
    seq[task_row, 0:4] = [0.0, 0.0, 0.0, 1.0]
    desc = world.descriptor.astype(np.float32)
    seq[task_row, 4 : 4 + len(desc)] = desc[: token_dim - 4]
    return seq


def make_hard_features_for_roadmap(world: C.World, roadmap: C.Roadmap, feature_cfg: C.FeatureConfig) -> np.ndarray:
    xs = np.zeros((roadmap.points.shape[0], feature_cfg.seq_len, feature_cfg.token_dim), dtype=np.float32)
    for i, p in enumerate(roadmap.points):
        xs[i] = C.make_feature_sequence(world, p, feature_cfg)
    return xs


def hard_smoke_overrides(args: Any) -> None:
    if not args.smoke_test:
        return
    _ORIG_APPLY_SMOKE_OVERRIDES(args)
    args.train_tasks = "C_hard_maze"
    args.eval_suites = "C_hard_maze"
    args.train_worlds = 1
    args.nodes_per_world = 24
    args.eval_worlds = 1
    args.roadmap_nodes = 64
    args.roadmap_k = 5
    args.nearest_obstacles = 6
    args.num_rays = 16
    args.ray_steps = 24
    args.budgets = "8,16,32"


def soft_bounded_model_forward(self: C.ContinuousHeuristicModel, x_seq: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    ctx = self.backbone.encode_sequence(x_seq)
    raw = self.head(ctx).squeeze(-1)
    out = F.softplus(raw)
    if clamp:
        cap = float(self.max_norm_residual)
        if cap > 0.0:
            out = cap * (1.0 - torch.exp(-out / cap))
        else:
            out = torch.clamp(out, min=0.0)
    return out


def install_runtime_extensions() -> None:
    C.build_anchor_specs = build_hard_anchor_specs
    C.generate_obstacles = generate_hard_obstacles
    C.build_world = build_hard_world
    C.task_descriptor = hard_task_descriptor
    C.make_feature_sequence = make_hard_feature_sequence
    C.make_features_for_roadmap = make_hard_features_for_roadmap
    C.apply_smoke_overrides = hard_smoke_overrides
    C.ContinuousHeuristicModel.forward = soft_bounded_model_forward if SOFT_RESIDUAL_CAP else _ORIG_MODEL_FORWARD


if __name__ == "__main__":
    configure_default_argv()
    install_runtime_extensions()
    from continuous_prm_stage_runner import main

    print(f"[c5] hard suites={HARD_EVAL_SUITES} sector_tokens={SECTOR_TOKENS} soft_residual_cap={SOFT_RESIDUAL_CAP}", flush=True)
    main(default_stage="full")
