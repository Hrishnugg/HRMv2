#!/usr/bin/env python3
"""
Continuous PRM residual-task LoRA v1.

This script implements a first continuous-coordinate analogue of the grid A*
heuristic experiments:

  continuous 2D world -> PRM roadmap -> Dijkstra labels -> learned A* heuristic

It preserves the recent experimental framing:

  1. Train one pooled avgbase model over all anchor task distributions.
  2. Freeze avgbase.
  3. Train one bounded residual LoRA expert per anchor distribution.
  4. Evaluate Euclidean, avgbase, matched/cross experts, and oracle expert.

The planner is continuous in state/world geometry, but A* is evaluated over a
sampled PRM graph. This is intentional: it gives us a controlled bridge from the
previous grid experiments to continuous geometric planning while retaining the
same core A* metrics: success, expansions, and path-cost ratio.

Example smoke run:

  python continuous_prm_residual_tasklora_v1.py --smoke-test --out-dir /tmp/cont_prm_smoke

Example fuller local run:

  python continuous_prm_residual_tasklora_v1.py \
      --out-dir runs/continuous_prm_v1 \
      --train-worlds 120 --eval-worlds 40 \
      --roadmap-nodes 256 --roadmap-k 14 \
      --base-epochs 12 --expert-epochs 8 \
      --backbones hrm,onlstm

The default runtime is deliberately moderate. Increase TRAIN_WORLDS,
ROADMAP_NODES, and epochs once the smoke run is clean.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch.utils.data import DataLoader, Dataset

try:
    if hasattr(torch.backends, "nnpack") and hasattr(torch.backends.nnpack, "set_flags"):
        torch.backends.nnpack.set_flags(False)
except Exception:
    pass

# -----------------------------------------------------------------------------
# Constants and small helpers
# -----------------------------------------------------------------------------

INF = 1.0e30
EPS = 1.0e-9
DEFAULT_TRAIN_TASKS = "C_open,C_clutter,C_narrow,C_large_clutter"
DEFAULT_EVAL_SUITES = (
    "C_open,C_clutter,C_narrow,C_large_clutter,"
    "C_extra_dense,C_tiny_passage,C_large_open,C_large_narrow,C_rectangles"
)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_csv(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fields: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


def sanitize_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-=" else "_" for ch in str(s)).strip("_")


def as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def finite_mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


# -----------------------------------------------------------------------------
# Continuous world specification
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Obstacle:
    kind: str  # "circle" or "rect"
    cx: float
    cy: float
    radius: float = 0.0
    hw: float = 0.0
    hh: float = 0.0

    def bounding_radius(self) -> float:
        if self.kind == "circle":
            return float(self.radius)
        return float(math.sqrt(self.hw * self.hw + self.hh * self.hh))

    def area(self) -> float:
        if self.kind == "circle":
            return math.pi * self.radius * self.radius
        return 4.0 * self.hw * self.hh


@dataclass(frozen=True)
class AnchorSpec:
    name: str
    side_len: float
    mode: str
    obstacle_count_range: Tuple[int, int]
    radius_range: Tuple[float, float]
    # Optional narrow-passage controls. Values are fractions of side_len.
    gap_width_frac: float = 0.18
    barrier_radius_frac: float = 0.035
    extra_clutter_range: Tuple[int, int] = (0, 0)
    rectangle_count_range: Tuple[int, int] = (0, 0)
    rect_hw_range: Tuple[float, float] = (0.03, 0.08)
    rect_hh_range: Tuple[float, float] = (0.03, 0.10)
    # Evaluation/training metadata only.
    is_ood: bool = False


@dataclass
class World:
    spec_name: str
    side_len: float
    obstacles: List[Obstacle]
    start: np.ndarray
    goal: np.ndarray
    descriptor: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoadmapConfig:
    n_nodes: int = 256
    k_neighbors: int = 14
    sample_attempts_per_node: int = 80
    min_start_goal_dist_frac: float = 0.45
    segment_resolution_frac: float = 0.010
    collision_margin: float = 0.0


@dataclass
class FeatureConfig:
    nearest_obstacles: int = 6
    num_rays: int = 16
    ray_steps: int = 80
    token_dim: int = 16
    max_obstacle_count_norm: float = 80.0
    max_radius_norm: float = 0.20
    max_side_for_log: float = 4.0

    @property
    def seq_len(self) -> int:
        return 1 + self.nearest_obstacles + self.num_rays + 1


@dataclass
class TrainingConfig:
    batch_size: int = 256
    base_epochs: int = 10
    expert_epochs: int = 8
    lr: float = 2.0e-4
    expert_lr: float = 1.5e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    max_norm_residual: float = 4.0
    residual_bound_quantile: float = 0.95
    residual_bound_floor: float = 0.08
    correction_l2: float = 1.0e-3
    num_workers: int = 0
    train_bias_with_lora: bool = False


@dataclass
class EvalConfig:
    budgets: Tuple[int, ...] = (100, 200, 500, 1000)
    eval_worlds: int = 40
    eval_experts_all_suites: bool = True
    eval_oracle_expert: bool = True
    max_eval_world_retries: int = 80


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    backbone_type: str
    hidden_dim: int
    num_layers: int
    chunk_size: int = 8
    k_step: int = 2
    num_heads: int = 4
    head_hidden: int = 256
    lora_rank: int = 8


def build_backbone_configs(args: argparse.Namespace) -> Dict[str, BackboneConfig]:
    # Defaults are lighter than the earlier grid scripts because PRM generation
    # and geometry feature extraction dominate in local runs. Hidden dimensions
    # can be increased from the command line.
    return {
        "hrm": BackboneConfig(
            name="hrm",
            backbone_type="hrm",
            hidden_dim=int(args.hrm_hidden),
            num_layers=int(args.hrm_layers),
            k_step=int(args.hrm_k_step),
            num_heads=int(args.hrm_heads),
            head_hidden=int(args.head_hidden),
            lora_rank=8,
        ),
        "onlstm": BackboneConfig(
            name="onlstm",
            backbone_type="onlstm",
            hidden_dim=int(args.onlstm_hidden),
            num_layers=int(args.onlstm_layers),
            chunk_size=int(args.onlstm_chunk_size),
            head_hidden=int(args.head_hidden),
            lora_rank=24,
        ),
    }


def build_anchor_specs() -> Dict[str, AnchorSpec]:
    # All lengths are in continuous world units. Most anchors use side_len=1.0;
    # large anchors use side_len=2.0 to test scale generalization.
    return {
        "C_open": AnchorSpec(
            name="C_open",
            side_len=1.0,
            mode="circles",
            obstacle_count_range=(3, 8),
            radius_range=(0.025, 0.060),
        ),
        "C_clutter": AnchorSpec(
            name="C_clutter",
            side_len=1.0,
            mode="circles",
            obstacle_count_range=(18, 30),
            radius_range=(0.025, 0.070),
        ),
        "C_narrow": AnchorSpec(
            name="C_narrow",
            side_len=1.0,
            mode="narrow",
            obstacle_count_range=(0, 0),
            radius_range=(0.025, 0.060),
            gap_width_frac=0.17,
            barrier_radius_frac=0.035,
            extra_clutter_range=(4, 10),
        ),
        "C_large_clutter": AnchorSpec(
            name="C_large_clutter",
            side_len=2.0,
            mode="circles",
            obstacle_count_range=(35, 55),
            radius_range=(0.045, 0.115),
        ),
        # OOD suites. These are not trained by default.
        "C_extra_dense": AnchorSpec(
            name="C_extra_dense",
            side_len=1.0,
            mode="circles",
            obstacle_count_range=(42, 68),
            radius_range=(0.022, 0.060),
            is_ood=True,
        ),
        "C_tiny_passage": AnchorSpec(
            name="C_tiny_passage",
            side_len=1.0,
            mode="narrow",
            obstacle_count_range=(0, 0),
            radius_range=(0.025, 0.060),
            gap_width_frac=0.10,
            barrier_radius_frac=0.038,
            extra_clutter_range=(8, 16),
            is_ood=True,
        ),
        "C_large_open": AnchorSpec(
            name="C_large_open",
            side_len=2.0,
            mode="circles",
            obstacle_count_range=(6, 14),
            radius_range=(0.050, 0.110),
            is_ood=True,
        ),
        "C_large_narrow": AnchorSpec(
            name="C_large_narrow",
            side_len=2.0,
            mode="narrow",
            obstacle_count_range=(0, 0),
            radius_range=(0.050, 0.115),
            gap_width_frac=0.13,
            barrier_radius_frac=0.032,
            extra_clutter_range=(14, 26),
            is_ood=True,
        ),
        "C_rectangles": AnchorSpec(
            name="C_rectangles",
            side_len=1.0,
            mode="rectangles",
            obstacle_count_range=(6, 12),
            radius_range=(0.0, 0.0),
            rectangle_count_range=(14, 24),
            rect_hw_range=(0.020, 0.070),
            rect_hh_range=(0.030, 0.120),
            is_ood=True,
        ),
    }


# -----------------------------------------------------------------------------
# Geometry and collision checking
# -----------------------------------------------------------------------------

def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= EPS:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def point_in_obstacle(p: np.ndarray, obs: Obstacle, margin: float = 0.0) -> bool:
    if obs.kind == "circle":
        return float(np.linalg.norm(p - np.array([obs.cx, obs.cy]))) <= obs.radius + margin
    if obs.kind == "rect":
        return abs(float(p[0]) - obs.cx) <= obs.hw + margin and abs(float(p[1]) - obs.cy) <= obs.hh + margin
    raise ValueError(f"unknown obstacle kind {obs.kind}")


def line_intersects_rect(a: np.ndarray, b: np.ndarray, obs: Obstacle, margin: float = 0.0) -> bool:
    # Liang-Barsky clipping against axis-aligned rectangle.
    xmin = obs.cx - obs.hw - margin
    xmax = obs.cx + obs.hw + margin
    ymin = obs.cy - obs.hh - margin
    ymax = obs.cy + obs.hh + margin
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    p = [-dx, dx, -dy, dy]
    q = [float(a[0] - xmin), float(xmax - a[0]), float(a[1] - ymin), float(ymax - a[1])]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < EPS:
            if qi < 0:
                return False
        else:
            r = qi / pi
            if pi < 0:
                if r > u2:
                    return False
                if r > u1:
                    u1 = r
            else:
                if r < u1:
                    return False
                if r < u2:
                    u2 = r
    return True


def segment_hits_obstacle(a: np.ndarray, b: np.ndarray, obs: Obstacle, margin: float = 0.0) -> bool:
    if obs.kind == "circle":
        center = np.array([obs.cx, obs.cy], dtype=np.float64)
        return point_segment_distance(center, a, b) <= obs.radius + margin
    if obs.kind == "rect":
        return line_intersects_rect(a, b, obs, margin=margin)
    raise ValueError(f"unknown obstacle kind {obs.kind}")


def is_point_free(p: np.ndarray, side_len: float, obstacles: Sequence[Obstacle], margin: float = 0.0) -> bool:
    if float(p[0]) < margin or float(p[0]) > side_len - margin:
        return False
    if float(p[1]) < margin or float(p[1]) > side_len - margin:
        return False
    for obs in obstacles:
        if point_in_obstacle(p, obs, margin=margin):
            return False
    return True


def is_segment_free(a: np.ndarray, b: np.ndarray, side_len: float, obstacles: Sequence[Obstacle], margin: float = 0.0) -> bool:
    # Endpoints must be free. Segment is otherwise inside the square domain.
    if not is_point_free(a, side_len, obstacles, margin=margin):
        return False
    if not is_point_free(b, side_len, obstacles, margin=margin):
        return False
    for obs in obstacles:
        if segment_hits_obstacle(a, b, obs, margin=margin):
            return False
    return True


def obstacle_clearance(p: np.ndarray, side_len: float, obstacles: Sequence[Obstacle]) -> float:
    # Boundary counts as an obstacle for local clearance.
    boundary = min(float(p[0]), float(p[1]), side_len - float(p[0]), side_len - float(p[1]))
    best = boundary
    for obs in obstacles:
        if obs.kind == "circle":
            d = float(np.linalg.norm(p - np.array([obs.cx, obs.cy]))) - obs.radius
        else:
            dx = abs(float(p[0]) - obs.cx) - obs.hw
            dy = abs(float(p[1]) - obs.cy) - obs.hh
            outside = math.sqrt(max(dx, 0.0) ** 2 + max(dy, 0.0) ** 2)
            inside = min(max(dx, dy), 0.0)
            d = outside + inside
        if d < best:
            best = d
    return float(best)


def segment_obstacle_proximity(a: np.ndarray, b: np.ndarray, obs: Obstacle) -> float:
    if obs.kind == "circle":
        center = np.array([obs.cx, obs.cy], dtype=np.float64)
        return point_segment_distance(center, a, b) - obs.radius
    # Approximate rectangle by bounding circle for corridor scoring.
    center = np.array([obs.cx, obs.cy], dtype=np.float64)
    return point_segment_distance(center, a, b) - obs.bounding_radius()


def sample_free_point(
    rng: random.Random,
    side_len: float,
    obstacles: Sequence[Obstacle],
    region: Optional[Tuple[float, float, float, float]] = None,
    margin: float = 1.0e-3,
    max_attempts: int = 2000,
) -> Optional[np.ndarray]:
    if region is None:
        xmin, xmax, ymin, ymax = margin, side_len - margin, margin, side_len - margin
    else:
        xmin, xmax, ymin, ymax = region
        xmin = max(margin, xmin)
        ymin = max(margin, ymin)
        xmax = min(side_len - margin, xmax)
        ymax = min(side_len - margin, ymax)
    for _ in range(max_attempts):
        p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=np.float64)
        if is_point_free(p, side_len, obstacles, margin=margin):
            return p
    return None


def approximate_free_fraction(side_len: float, obstacles: Sequence[Obstacle]) -> float:
    # Cheap area approximation. Obstacle overlap and clipping are ignored; the
    # value is a descriptor, not a geometric proof.
    occupied = sum(max(0.0, obs.area()) for obs in obstacles)
    return float(max(0.0, min(1.0, 1.0 - occupied / max(EPS, side_len * side_len))))


def generate_obstacles(spec: AnchorSpec, rng: random.Random) -> List[Obstacle]:
    side = spec.side_len
    obstacles: List[Obstacle] = []

    def add_random_circles(n: int) -> None:
        start_circle_count = len([o for o in obstacles if o.kind == "circle"])
        target_circle_count = start_circle_count + int(n)
        attempts = 0
        while len([o for o in obstacles if o.kind == "circle"]) < target_circle_count and attempts < max(1, n) * 200:
            attempts += 1
            r = rng.uniform(spec.radius_range[0], spec.radius_range[1])
            cx = rng.uniform(r + 0.015 * side, side - r - 0.015 * side)
            cy = rng.uniform(r + 0.015 * side, side - r - 0.015 * side)
            new = Obstacle("circle", cx, cy, radius=r)
            # Keep obstacles from collapsing into one huge blob.
            ok = True
            for obs in obstacles:
                if np.linalg.norm(np.array([cx - obs.cx, cy - obs.cy])) < (r + obs.bounding_radius()) * 0.65:
                    ok = False
                    break
            if ok:
                obstacles.append(new)

    if spec.mode == "circles":
        n = rng.randint(*spec.obstacle_count_range)
        add_random_circles(n)

    elif spec.mode == "narrow":
        # A vertical barrier of overlapping circles with a randomized gap. This
        # creates a continuous narrow-passage analogue of the grid barrier tasks.
        gap_center = rng.uniform(0.36 * side, 0.64 * side)
        gap_width = spec.gap_width_frac * side
        br = spec.barrier_radius_frac * side
        spacing = br * 1.55
        y = br
        while y <= side - br:
            if abs(y - gap_center) > 0.5 * gap_width:
                jitter = rng.uniform(-0.12 * br, 0.12 * br)
                obstacles.append(Obstacle("circle", 0.5 * side + jitter, y, radius=br))
            y += spacing
        extra = rng.randint(*spec.extra_clutter_range)
        add_random_circles(extra)

    elif spec.mode == "rectangles":
        n_rect = rng.randint(*spec.rectangle_count_range)
        attempts = 0
        while len([o for o in obstacles if o.kind == "rect"]) < n_rect and attempts < n_rect * 200:
            attempts += 1
            hw = rng.uniform(spec.rect_hw_range[0], spec.rect_hw_range[1]) * side
            hh = rng.uniform(spec.rect_hh_range[0], spec.rect_hh_range[1]) * side
            cx = rng.uniform(hw + 0.015 * side, side - hw - 0.015 * side)
            cy = rng.uniform(hh + 0.015 * side, side - hh - 0.015 * side)
            new = Obstacle("rect", cx, cy, hw=hw, hh=hh)
            ok = True
            for obs in obstacles:
                if np.linalg.norm(np.array([cx - obs.cx, cy - obs.cy])) < (new.bounding_radius() + obs.bounding_radius()) * 0.55:
                    ok = False
                    break
            if ok:
                obstacles.append(new)
        n_circle = rng.randint(*spec.obstacle_count_range)
        add_random_circles(n_circle)
    else:
        raise ValueError(f"unknown anchor mode {spec.mode}")

    return obstacles


def build_world(spec: AnchorSpec, seed: int, min_start_goal_dist_frac: float) -> Optional[World]:
    rng = random.Random(seed)
    side = spec.side_len
    obstacles = generate_obstacles(spec, rng)

    # Narrow-passage tasks force start and goal onto opposite sides of the wall.
    if spec.mode == "narrow":
        left = (0.03 * side, 0.40 * side, 0.05 * side, 0.95 * side)
        right = (0.60 * side, 0.97 * side, 0.05 * side, 0.95 * side)
        if rng.random() < 0.5:
            start_region, goal_region = left, right
        else:
            start_region, goal_region = right, left
    else:
        start_region = None
        goal_region = None

    for _ in range(120):
        start = sample_free_point(rng, side, obstacles, region=start_region)
        goal = sample_free_point(rng, side, obstacles, region=goal_region)
        if start is None or goal is None:
            continue
        if float(np.linalg.norm(goal - start)) < min_start_goal_dist_frac * side:
            continue
        if not is_point_free(start, side, obstacles) or not is_point_free(goal, side, obstacles):
            continue
        desc = task_descriptor(spec, obstacles)
        return World(
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
                "free_fraction_est": approximate_free_fraction(side, obstacles),
            },
        )
    return None


def task_descriptor(spec: AnchorSpec, obstacles: Sequence[Obstacle]) -> np.ndarray:
    side = spec.side_len
    n_obs = len(obstacles)
    mean_r = float(np.mean([o.bounding_radius() for o in obstacles])) if obstacles else 0.0
    density = sum(o.area() for o in obstacles) / max(EPS, side * side)
    free_frac = approximate_free_fraction(side, obstacles)
    narrow = 1.0 if spec.mode == "narrow" else 0.0
    rect_frac = float(sum(1 for o in obstacles if o.kind == "rect")) / max(1.0, float(n_obs))
    # 8-dimensional compact descriptor, intentionally no task one-hot.
    return np.array([
        math.log(max(EPS, side)),
        min(1.0, n_obs / 80.0),
        min(1.0, mean_r / max(EPS, side) / 0.20),
        min(1.0, density),
        free_frac,
        narrow,
        rect_frac,
        1.0 if spec.is_ood else 0.0,
    ], dtype=np.float32)


# -----------------------------------------------------------------------------
# PRM construction and graph search
# -----------------------------------------------------------------------------

@dataclass
class Roadmap:
    points: np.ndarray  # [N, 2], index 0=start, index 1=goal
    adj: List[List[Tuple[int, float]]]
    dist_to_goal: np.ndarray
    connected_to_goal: np.ndarray


def build_prm(world: World, cfg: RoadmapConfig, seed: int) -> Optional[Roadmap]:
    rng = random.Random(seed)
    side = world.side_len
    pts: List[np.ndarray] = [world.start.astype(np.float64), world.goal.astype(np.float64)]
    attempts = 0
    max_attempts = cfg.n_nodes * cfg.sample_attempts_per_node
    while len(pts) < cfg.n_nodes and attempts < max_attempts:
        attempts += 1
        p = sample_free_point(rng, side, world.obstacles, max_attempts=8)
        if p is not None:
            pts.append(p)
    if len(pts) < max(16, cfg.n_nodes // 3):
        return None

    points = np.stack(pts, axis=0).astype(np.float64)
    n = points.shape[0]
    diff = points[:, None, :] - points[None, :, :]
    dmat = np.sqrt(np.sum(diff * diff, axis=-1))
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    edge_set = set()

    for i in range(n):
        # Make start/goal slightly better connected. This reduces invalid worlds
        # caused by unlucky endpoint samples.
        k = cfg.k_neighbors * (2 if i in (0, 1) else 1)
        nn_idx = np.argsort(dmat[i])[1 : min(n, k + 1)]
        for j in nn_idx:
            a, b = int(i), int(j)
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in edge_set:
                continue
            if is_segment_free(points[a], points[b], side, world.obstacles, margin=cfg.collision_margin):
                w = float(dmat[a, b])
                adj[a].append((b, w))
                adj[b].append((a, w))
                edge_set.add(key)

    dist = dijkstra_to_goal(adj, goal_idx=1)
    connected = np.isfinite(dist) & (dist < INF / 10.0)
    if not connected[0]:
        return None
    return Roadmap(points=points, adj=adj, dist_to_goal=dist, connected_to_goal=connected)


def dijkstra_to_goal(adj: List[List[Tuple[int, float]]], goal_idx: int = 1) -> np.ndarray:
    n = len(adj)
    dist = np.full(n, INF, dtype=np.float64)
    dist[goal_idx] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, goal_idx)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def astar_search(adj: List[List[Tuple[int, float]]], heuristic: np.ndarray, budget: int, start_idx: int = 0, goal_idx: int = 1) -> Dict[str, Any]:
    n = len(adj)
    g = np.full(n, INF, dtype=np.float64)
    g[start_idx] = 0.0
    heap: List[Tuple[float, float, int]] = [(float(heuristic[start_idx]), 0.0, start_idx)]
    closed = np.zeros(n, dtype=np.bool_)
    expansions = 0
    while heap and expansions < budget:
        _, cur_g, u = heapq.heappop(heap)
        if closed[u]:
            continue
        if cur_g != g[u]:
            continue
        closed[u] = True
        expansions += 1
        if u == goal_idx:
            return {"found": True, "cost": float(g[u]), "expansions": expansions, "closed": int(closed.sum())}
        for v, w in adj[u]:
            if closed[v]:
                continue
            ng = g[u] + w
            if ng < g[v]:
                g[v] = ng
                f = ng + float(heuristic[v])
                heapq.heappush(heap, (f, ng, v))
    return {"found": False, "cost": float("nan"), "expansions": expansions, "closed": int(closed.sum())}


# -----------------------------------------------------------------------------
# Geometry feature extraction
# -----------------------------------------------------------------------------

def nearest_obstacle_records(p: np.ndarray, world: World, k: int) -> List[Tuple[Obstacle, float]]:
    recs: List[Tuple[Obstacle, float]] = []
    for obs in world.obstacles:
        if obs.kind == "circle":
            d = float(np.linalg.norm(p - np.array([obs.cx, obs.cy]))) - obs.radius
        else:
            center = np.array([obs.cx, obs.cy], dtype=np.float64)
            d = float(np.linalg.norm(p - center)) - obs.bounding_radius()
        recs.append((obs, d))
    recs.sort(key=lambda x: x[1])
    return recs[:k]


def raycast_distance(p: np.ndarray, angle: float, world: World, steps: int) -> float:
    side = world.side_len
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
    # Distance to square boundary along ray.
    max_t = side * math.sqrt(2.0)
    if abs(direction[0]) > EPS:
        tx1 = (0.0 - p[0]) / direction[0]
        tx2 = (side - p[0]) / direction[0]
        max_t = min(max_t, max(t for t in (tx1, tx2) if t > 0) if any(t > 0 for t in (tx1, tx2)) else max_t)
    if abs(direction[1]) > EPS:
        ty1 = (0.0 - p[1]) / direction[1]
        ty2 = (side - p[1]) / direction[1]
        max_t = min(max_t, max(t for t in (ty1, ty2) if t > 0) if any(t > 0 for t in (ty1, ty2)) else max_t)
    max_t = min(max_t, side * math.sqrt(2.0))
    if max_t <= 0:
        return 0.0
    # Conservative sampled ray cast. Exact intersections are unnecessary for a
    # learned descriptor; the feature only needs stable local geometry signal.
    for s in range(1, steps + 1):
        t = max_t * s / steps
        q = p + t * direction
        if not is_point_free(q, side, world.obstacles):
            return float(max_t * (s - 1) / steps)
    return float(max_t)


def corridor_blockage_score(p: np.ndarray, goal: np.ndarray, world: World) -> float:
    # Count obstacles close to the straight-line state-goal corridor.
    side = world.side_len
    corridor_width = 0.045 * side
    count = 0
    for obs in world.obstacles:
        if segment_obstacle_proximity(p, goal, obs) < corridor_width:
            count += 1
    return float(min(1.0, count / 8.0))


def make_feature_sequence(world: World, p: np.ndarray, feature_cfg: FeatureConfig) -> np.ndarray:
    side = world.side_len
    goal = world.goal
    token_dim = feature_cfg.token_dim
    seq = np.zeros((feature_cfg.seq_len, token_dim), dtype=np.float32)

    # State/goal token. First 4 dimensions are token-type flags.
    dx = (goal[0] - p[0]) / side
    dy = (goal[1] - p[1]) / side
    eu = float(np.linalg.norm(goal - p)) / side
    clearance = max(0.0, obstacle_clearance(p, side, world.obstacles)) / side
    los = 1.0 if is_segment_free(p, goal, side, world.obstacles) else 0.0
    corridor = corridor_blockage_score(p, goal, world)
    seq[0, 0:4] = [1.0, 0.0, 0.0, 0.0]
    seq[0, 4:16] = [
        float(dx),
        float(dy),
        float(eu),
        float(p[0] / side * 2.0 - 1.0),
        float(p[1] / side * 2.0 - 1.0),
        float(clearance),
        float(los),
        float(corridor),
        float(approximate_free_fraction(side, world.obstacles)),
        float(min(1.0, len(world.obstacles) / feature_cfg.max_obstacle_count_norm)),
        0.0,
        0.0,
    ]

    # Nearest obstacle tokens.
    recs = nearest_obstacle_records(p, world, feature_cfg.nearest_obstacles)
    base_idx = 1
    for oi in range(feature_cfg.nearest_obstacles):
        row = base_idx + oi
        seq[row, 0:4] = [0.0, 1.0, 0.0, 0.0]
        if oi >= len(recs):
            continue
        obs, clearance_to_obs = recs[oi]
        cx, cy = obs.cx, obs.cy
        relx = (cx - p[0]) / side
        rely = (cy - p[1]) / side
        br = obs.bounding_radius() / side
        dcenter = math.sqrt(relx * relx + rely * rely)
        angle = math.atan2(rely, relx)
        seq[row, 4:16] = [
            float(relx),
            float(rely),
            float(dcenter),
            float(br),
            float(clearance_to_obs / side),
            1.0 if obs.kind == "circle" else 0.0,
            1.0 if obs.kind == "rect" else 0.0,
            float(math.cos(angle)),
            float(math.sin(angle)),
            float(obs.hw / side if obs.kind == "rect" else 0.0),
            float(obs.hh / side if obs.kind == "rect" else 0.0),
            0.0,
        ]

    # Ray tokens.
    ray_base = 1 + feature_cfg.nearest_obstacles
    for ri in range(feature_cfg.num_rays):
        row = ray_base + ri
        angle = 2.0 * math.pi * ri / feature_cfg.num_rays
        rd = raycast_distance(p, angle, world, feature_cfg.ray_steps) / side
        seq[row, 0:4] = [0.0, 0.0, 1.0, 0.0]
        seq[row, 4:16] = [
            float(math.cos(angle)),
            float(math.sin(angle)),
            float(min(1.5, rd)),
            float(ri / max(1, feature_cfg.num_rays - 1)),
            float(dx),
            float(dy),
            float(eu),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    # Task descriptor token.
    task_row = feature_cfg.seq_len - 1
    seq[task_row, 0:4] = [0.0, 0.0, 0.0, 1.0]
    desc = world.descriptor.astype(np.float32)
    seq[task_row, 4 : 4 + len(desc)] = desc[: token_dim - 4]
    return seq


def make_features_for_roadmap(world: World, roadmap: Roadmap, feature_cfg: FeatureConfig) -> np.ndarray:
    xs = np.zeros((roadmap.points.shape[0], feature_cfg.seq_len, feature_cfg.token_dim), dtype=np.float32)
    for i, p in enumerate(roadmap.points):
        xs[i] = make_feature_sequence(world, p, feature_cfg)
    return xs


# -----------------------------------------------------------------------------
# Dataset collection
# -----------------------------------------------------------------------------

def collect_task_dataset(
    spec: AnchorSpec,
    out_dir: Path,
    split: str,
    n_worlds: int,
    nodes_per_world: int,
    roadmap_cfg: RoadmapConfig,
    feature_cfg: FeatureConfig,
    seed: int,
    max_world_retries: int = 100,
) -> Path:
    ensure_dir(out_dir)
    npz_path = out_dir / f"{spec.name}_{split}.npz"
    meta_path = out_dir / f"{spec.name}_{split}.json"
    if npz_path.exists() and meta_path.exists():
        return npz_path

    rng = random.Random(seed)
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_side: List[float] = []
    all_eu: List[float] = []
    world_meta: List[Dict[str, Any]] = []
    worlds_done = 0
    attempts = 0
    while worlds_done < n_worlds and attempts < n_worlds * max_world_retries:
        attempts += 1
        w_seed = rng.randint(0, 2**31 - 1)
        world = build_world(spec, w_seed, roadmap_cfg.min_start_goal_dist_frac)
        if world is None:
            continue
        roadmap = build_prm(world, roadmap_cfg, seed=w_seed + 17)
        if roadmap is None or not roadmap.connected_to_goal[0]:
            continue
        connected_idxs = np.where(roadmap.connected_to_goal)[0]
        connected_idxs = connected_idxs[connected_idxs != 1]  # no need to train on the goal label
        if len(connected_idxs) < max(12, nodes_per_world // 4):
            continue
        features = make_features_for_roadmap(world, roadmap, feature_cfg)
        # Bias toward nodes that matter for planning by always including start,
        # then sampling connected nodes uniformly.
        chosen: List[int] = [0]
        if nodes_per_world > 1:
            sample_size = min(nodes_per_world - 1, len(connected_idxs))
            chosen.extend(rng.sample([int(i) for i in connected_idxs], k=sample_size))
        chosen = chosen[:nodes_per_world]
        x = features[chosen]
        euclid = np.linalg.norm(roadmap.points[chosen] - world.goal[None, :], axis=1)
        residual = np.maximum(0.0, roadmap.dist_to_goal[chosen] - euclid) / world.side_len
        finite = np.isfinite(residual)
        if finite.sum() < max(8, len(chosen) // 3):
            continue
        x = x[finite].astype(np.float32)
        residual = residual[finite].astype(np.float32)
        euclid = euclid[finite].astype(np.float32)
        all_x.append(x)
        all_y.append(residual)
        all_eu.extend([float(v) for v in euclid])
        all_side.extend([float(world.side_len)] * len(residual))
        world_meta.append({
            **world.meta,
            "world_idx": worlds_done,
            "roadmap_nodes": int(roadmap.points.shape[0]),
            "roadmap_edges": int(sum(len(a) for a in roadmap.adj) // 2),
            "start_goal_oracle_cost": float(roadmap.dist_to_goal[0]),
            "samples": int(len(residual)),
        })
        worlds_done += 1
        print(f"[{now_str()}] collected {spec.name}/{split}: {worlds_done}/{n_worlds} worlds", flush=True)

    if not all_x:
        raise RuntimeError(f"failed to collect any data for {spec.name}/{split}")
    x_cat = np.concatenate(all_x, axis=0).astype(np.float32)
    y_cat = np.concatenate(all_y, axis=0).astype(np.float32)
    np.savez_compressed(npz_path, x=x_cat, y=y_cat, euclid=np.array(all_eu, dtype=np.float32), side=np.array(all_side, dtype=np.float32))
    meta = {
        "task": spec.name,
        "split": split,
        "n_worlds_requested": n_worlds,
        "n_worlds_collected": worlds_done,
        "n_samples": int(x_cat.shape[0]),
        "seq_len": int(feature_cfg.seq_len),
        "token_dim": int(feature_cfg.token_dim),
        "target_residual_norm_mean": float(np.mean(y_cat)),
        "target_residual_norm_p95": float(np.quantile(y_cat, 0.95)),
        "worlds": world_meta,
        "spec": asdict(spec),
    }
    write_json(meta_path, meta)
    return npz_path


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.float32)


class BalancedTaskDataset(Dataset):
    def __init__(self, task_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]], epoch_size: Optional[int] = None):
        self.tasks = list(task_arrays.keys())
        self.arrays = task_arrays
        self.max_len = max(len(task_arrays[t][1]) for t in self.tasks)
        self.epoch_size = int(epoch_size or self.max_len * len(self.tasks))

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        task = self.tasks[idx % len(self.tasks)]
        x, y = self.arrays[task]
        # Random local sample gives task-balanced training without duplicating arrays.
        j = random.randrange(len(y))
        return torch.from_numpy(x[j]), torch.tensor(y[j], dtype=torch.float32)


def load_npz_arrays(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["x"].astype(np.float32), data["y"].astype(np.float32)


# -----------------------------------------------------------------------------
# Models: HRM, ONLSTM, LoRA, continuous residual head
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        norm = x_f32.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return ((x_f32 / norm) * self.scale * self.g).to(dtype=x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GatedRecurrentBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, int(dim * 2.6))
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        h = (x + state) * 0.7071
        res = h
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm.unsqueeze(1), h_norm.unsqueeze(1), h_norm.unsqueeze(1), need_weights=False)
        h = res + attn_out.squeeze(1)
        candidate = h + self.ffn(self.norm2(h))
        z = torch.sigmoid(self.gate(torch.cat([candidate, state], dim=-1)))
        return z * candidate + (1.0 - z) * state


class ONLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, chunk_size: int = 8):
        super().__init__()
        if hidden_dim % chunk_size != 0:
            raise ValueError("hidden_dim must be divisible by chunk_size")
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.n_chunks = hidden_dim // chunk_size
        self.lin = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim + 2 * self.n_chunks)

    @staticmethod
    def cumax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return torch.cumsum(F.softmax(x, dim=dim), dim=dim)

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        gates = self.lin(torch.cat([x, h_prev], dim=-1))
        H = self.hidden_dim
        i, f, o, g = gates[:, : 4 * H].chunk(4, dim=-1)
        f_hat_lin, i_hat_lin = gates[:, 4 * H :].chunk(2, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        f_hat = self.cumax(f_hat_lin).repeat_interleave(self.chunk_size, dim=-1)
        i_hat = (1.0 - self.cumax(i_hat_lin)).repeat_interleave(self.chunk_size, dim=-1)
        omega = f_hat * i_hat
        f = f * omega + (f_hat - omega)
        i = i * omega + (i_hat - omega)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ONLSTMBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, chunk_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cells = nn.ModuleList([ONLSTMCell(hidden_dim, hidden_dim, chunk_size=chunk_size) for _ in range(num_layers)])

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        b, seq, _ = x.shape
        h = [torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype) for _ in self.cells]
        c = [torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype) for _ in self.cells]
        out = torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype)
        for t in range(seq):
            inp = self.input_proj(x[:, t])
            new_h: List[torch.Tensor] = []
            new_c: List[torch.Tensor] = []
            for li, cell in enumerate(self.cells):
                hi, ci = cell(inp, (h[li], c[li]))
                inp = hi
                new_h.append(hi)
                new_c.append(ci)
            h, c = new_h, new_c
            out = h[-1]
        return out


class DeepSapientHRMBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, k_step: int, num_heads: int, num_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_step = k_step
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.L_blocks = nn.ModuleList([GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.H_blocks = nn.ModuleList([GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        b, seq, _ = x.shape
        h_L = [torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype) for _ in self.L_blocks]
        h_H = [torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype) for _ in self.H_blocks]
        ctx = torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype)
        for t in range(seq):
            curr_in = self.embed(x[:, t])
            if t % max(1, self.k_step) == 0:
                h_in = h_L[-1].detach()
                new_hH: List[torch.Tensor] = []
                for i, blk in enumerate(self.H_blocks):
                    h = blk(h_in, h_H[i])
                    new_hH.append(h)
                    h_in = h
                h_H = new_hH
            l_in = curr_in + h_H[-1]
            new_hL: List[torch.Tensor] = []
            for i, blk in enumerate(self.L_blocks):
                h = blk(l_in, h_L[i])
                new_hL.append(h)
                l_in = h
            h_L = new_hL
            ctx = h_L[-1]
        return ctx


class ContinuousHeuristicModel(nn.Module):
    def __init__(self, cfg: BackboneConfig, token_dim: int, max_norm_residual: float = 4.0):
        super().__init__()
        self.cfg = cfg
        self.max_norm_residual = float(max_norm_residual)
        if cfg.backbone_type == "hrm":
            self.backbone = DeepSapientHRMBackbone(token_dim, cfg.hidden_dim, cfg.k_step, cfg.num_heads, cfg.num_layers)
        elif cfg.backbone_type == "onlstm":
            self.backbone = ONLSTMBackbone(token_dim, cfg.hidden_dim, cfg.num_layers, cfg.chunk_size)
        else:
            raise ValueError(f"unknown backbone type {cfg.backbone_type}")
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Linear(cfg.head_hidden, cfg.head_hidden),
            nn.GELU(),
            nn.Linear(cfg.head_hidden, 1),
        )
        # Most continuous PRM residuals are small in open maps. A negative final
        # bias prevents the initial heuristic from adding a large detour penalty
        # everywhere before learning.
        last = self.head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, -2.0)

    def forward(self, x_seq: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        ctx = self.backbone.encode_sequence(x_seq)
        raw = self.head(ctx).squeeze(-1)
        # The residual d*(x,g)-Euclidean(x,g) should be nonnegative in this
        # static geometric setting. Clamp avoids rare unbounded predictions.
        out = F.softplus(raw)
        if clamp:
            out = torch.clamp(out, min=0.0, max=self.max_norm_residual)
        return out


class SingleAdapterLoRA(nn.Module):
    def __init__(self, base_weight: torch.Tensor, rank: int, alpha: float, init_scale: float = 0.01):
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.out_dim = int(base_weight.shape[0])
        self.in_dim = int(base_weight.numel() // self.out_dim)
        self.weight_shape = tuple(base_weight.shape)
        scale = self.alpha / max(1, self.rank)
        self.register_buffer("adapter_scale", torch.tensor(float(scale), dtype=base_weight.dtype, device=base_weight.device))
        self.A = nn.Parameter(torch.randn(self.rank, self.in_dim, device=base_weight.device, dtype=base_weight.dtype) * init_scale)
        self.B = nn.Parameter(torch.zeros(self.out_dim, self.rank, device=base_weight.device, dtype=base_weight.dtype))

    def forward(self, base_w: torch.Tensor) -> torch.Tensor:
        flat = base_w.reshape(self.out_dim, self.in_dim)
        out = flat + self.adapter_scale * (self.B @ self.A)
        return out.reshape(self.weight_shape)


def iter_lora_targets(module: nn.Module, include_attn: bool = True):
    for name, sub in module.named_modules():
        if isinstance(sub, nn.Linear):
            yield name, sub, "weight"
        elif include_attn and isinstance(sub, nn.MultiheadAttention):
            yield name + ".in_proj", sub, "in_proj_weight"
            yield name + ".out_proj", sub.out_proj, "weight"


def apply_lora(module: nn.Module, rank: int, alpha: float, init_scale: float = 0.01) -> int:
    wrapped = 0
    seen = set()
    for _name, sub, attr in iter_lora_targets(module, include_attn=True):
        key = (id(sub), attr)
        if key in seen:
            continue
        seen.add(key)
        if parametrize.is_parametrized(sub, attr):
            continue
        w = getattr(sub, attr)
        if not isinstance(w, nn.Parameter):
            continue
        parametrize.register_parametrization(sub, attr, SingleAdapterLoRA(w.data, rank, alpha, init_scale=init_scale), unsafe=True)
        wrapped += 1
    return wrapped


def set_lora_trainable(module: nn.Module, train_bias: bool = False) -> None:
    for name, p in module.named_parameters():
        p.requires_grad = False
        if ".parametrizations." in name and (".A" in name or ".B" in name):
            p.requires_grad = True
        elif train_bias and name.endswith(".bias"):
            p.requires_grad = True


def trainable_parameters(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


# -----------------------------------------------------------------------------
# Training and prediction helpers
# -----------------------------------------------------------------------------

def model_checkpoint_path(out_dir: Path, backbone: str, kind: str, task: Optional[str] = None, alpha: Optional[float] = None) -> Path:
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    if kind == "avgbase":
        return ckpt_dir / f"avgbase__{backbone}.pt"
    if kind == "expert":
        alpha_part = f"__alpha={alpha:g}" if alpha is not None else ""
        return ckpt_dir / f"expert__{backbone}__{task}{alpha_part}.pt"
    raise ValueError(kind)


def build_model(cfg: BackboneConfig, feature_cfg: FeatureConfig, train_cfg: TrainingConfig, device: torch.device) -> ContinuousHeuristicModel:
    model = ContinuousHeuristicModel(cfg, token_dim=feature_cfg.token_dim, max_norm_residual=train_cfg.max_norm_residual)
    return model.to(device)


def safe_load_state(model: nn.Module, path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    return payload if isinstance(payload, dict) else {"model": state}


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)


def eval_regression(model: ContinuousHeuristicModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    maes: List[float] = []
    pred_vals: List[float] = []
    targ_vals: List[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb, reduction="mean")
            losses.append(float(loss.item()))
            maes.append(float(torch.mean(torch.abs(pred - yb)).item()))
            pred_vals.extend([float(v) for v in pred.detach().cpu().flatten()[:512]])
            targ_vals.extend([float(v) for v in yb.detach().cpu().flatten()[:512]])
    corr = float("nan")
    if len(pred_vals) > 3 and np.std(pred_vals) > 1.0e-8 and np.std(targ_vals) > 1.0e-8:
        corr = float(np.corrcoef(np.array(pred_vals), np.array(targ_vals))[0, 1])
    return {"loss": finite_mean(losses), "mae": finite_mean(maes), "corr_sample": corr}


def train_avgbase(
    backbone_cfg: BackboneConfig,
    task_npz: Dict[str, Path],
    out_dir: Path,
    feature_cfg: FeatureConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    seed: int,
) -> Path:
    ckpt_path = model_checkpoint_path(out_dir, backbone_cfg.name, "avgbase")
    if ckpt_path.exists():
        print(f"[{now_str()}] avgbase checkpoint exists: {ckpt_path}")
        return ckpt_path

    arrays = {task: load_npz_arrays(path) for task, path in task_npz.items()}
    ds = BalancedTaskDataset(arrays)
    loader = make_loader(ds, train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)
    model = build_model(backbone_cfg, feature_cfg, train_cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    history: List[Dict[str, Any]] = []
    set_global_seed(seed)
    for epoch in range(1, train_cfg.base_epochs + 1):
        model.train()
        losses: List[float] = []
        maes: List[float] = []
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            if not torch.isfinite(pred).all():
                raise RuntimeError(f"nonfinite avgbase predictions for {backbone_cfg.name}")
            loss = F.smooth_l1_loss(pred, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite avgbase loss for {backbone_cfg.name}")
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            losses.append(float(loss.item()))
            maes.append(float(torch.mean(torch.abs(pred.detach() - yb)).item()))
        row = {
            "epoch": epoch,
            "loss": finite_mean(losses),
            "mae": finite_mean(maes),
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(f"[{now_str()}] avgbase {backbone_cfg.name} epoch {epoch}/{train_cfg.base_epochs}: loss={row['loss']:.5f} mae={row['mae']:.5f}", flush=True)

    payload = {
        "model": model.state_dict(),
        "backbone_cfg": asdict(backbone_cfg),
        "feature_cfg": asdict(feature_cfg),
        "train_cfg": asdict(train_cfg),
        "tasks": list(task_npz.keys()),
        "history": history,
    }
    ensure_dir(ckpt_path.parent)
    torch.save(payload, ckpt_path)
    write_json(out_dir / "logs" / f"train_avgbase__{backbone_cfg.name}.json", {"history": history})
    return ckpt_path


def calibrate_residual_bound(
    base_model: ContinuousHeuristicModel,
    ds: ArrayDataset,
    train_cfg: TrainingConfig,
    device: torch.device,
) -> float:
    loader = make_loader(ds, train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)
    errors: List[np.ndarray] = []
    base_model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = base_model(xb)
            err = (yb - pred).detach().cpu().numpy().astype(np.float64)
            errors.append(err)
    if not errors:
        return train_cfg.residual_bound_floor
    abs_err = np.abs(np.concatenate(errors))
    q = float(np.quantile(abs_err, train_cfg.residual_bound_quantile))
    return float(max(train_cfg.residual_bound_floor, q))


def train_expert(
    backbone_cfg: BackboneConfig,
    task: str,
    task_npz_path: Path,
    base_ckpt: Path,
    out_dir: Path,
    feature_cfg: FeatureConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    seed: int,
    alpha: float,
) -> Path:
    ckpt_path = model_checkpoint_path(out_dir, backbone_cfg.name, "expert", task=task, alpha=alpha)
    if ckpt_path.exists():
        print(f"[{now_str()}] expert checkpoint exists: {ckpt_path}")
        return ckpt_path

    x, y = load_npz_arrays(task_npz_path)
    ds = ArrayDataset(x, y)
    loader = make_loader(ds, train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)

    base_model = build_model(backbone_cfg, feature_cfg, train_cfg, device)
    safe_load_state(base_model, base_ckpt)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    bound = calibrate_residual_bound(base_model, ds, train_cfg, device)

    expert = build_model(backbone_cfg, feature_cfg, train_cfg, device)
    safe_load_state(expert, base_ckpt)
    n_wrapped = apply_lora(expert, rank=backbone_cfg.lora_rank, alpha=alpha, init_scale=0.01)
    set_lora_trainable(expert, train_bias=train_cfg.train_bias_with_lora)
    params = trainable_parameters(expert)
    if not params:
        raise RuntimeError(f"no trainable LoRA parameters for {backbone_cfg.name}/{task}")
    opt = torch.optim.AdamW(params, lr=train_cfg.expert_lr, weight_decay=train_cfg.weight_decay)

    history: List[Dict[str, Any]] = []
    set_global_seed(seed)
    for epoch in range(1, train_cfg.expert_epochs + 1):
        expert.train()
        losses: List[float] = []
        maes: List[float] = []
        corr_abs: List[float] = []
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.no_grad():
                base_pred = base_model(xb)
            adapt_pred = expert(xb)
            correction = bound * torch.tanh((adapt_pred - base_pred) / max(bound, EPS))
            final_pred = torch.clamp(base_pred + correction, min=0.0, max=train_cfg.max_norm_residual)
            if not torch.isfinite(final_pred).all():
                raise RuntimeError(f"nonfinite expert prediction for {backbone_cfg.name}/{task}")
            data_loss = F.smooth_l1_loss(final_pred, yb)
            reg_loss = train_cfg.correction_l2 * torch.mean((correction / max(bound, EPS)) ** 2)
            loss = data_loss + reg_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite expert loss for {backbone_cfg.name}/{task}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, train_cfg.grad_clip)
            opt.step()
            losses.append(float(loss.item()))
            maes.append(float(torch.mean(torch.abs(final_pred.detach() - yb)).item()))
            corr_abs.append(float(torch.mean(torch.abs(correction.detach())).item()))
        row = {
            "epoch": epoch,
            "loss": finite_mean(losses),
            "mae": finite_mean(maes),
            "correction_abs_mean": finite_mean(corr_abs),
            "residual_bound": bound,
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"[{now_str()}] expert {backbone_cfg.name}/{task}/alpha={alpha:g} epoch {epoch}/{train_cfg.expert_epochs}: "
            f"loss={row['loss']:.5f} mae={row['mae']:.5f} |corr|={row['correction_abs_mean']:.5f} B={bound:.4f}",
            flush=True,
        )

    payload = {
        "model": expert.state_dict(),
        "backbone_cfg": asdict(backbone_cfg),
        "feature_cfg": asdict(feature_cfg),
        "train_cfg": asdict(train_cfg),
        "task": task,
        "alpha": alpha,
        "lora_rank": backbone_cfg.lora_rank,
        "lora_wrapped_modules": n_wrapped,
        "residual_bound": bound,
        "history": history,
    }
    ensure_dir(ckpt_path.parent)
    torch.save(payload, ckpt_path)
    write_json(out_dir / "logs" / f"train_expert__{backbone_cfg.name}__{task}__alpha={alpha:g}.json", payload | {"model": "<state_dict omitted>"})
    return ckpt_path


def predict_norm_residuals(
    model: ContinuousHeuristicModel,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device)
            pred = model(xb).detach().cpu().numpy().astype(np.float64)
            outs.append(pred)
    return np.concatenate(outs, axis=0) if outs else np.zeros((0,), dtype=np.float64)


def load_base_model(backbone_cfg: BackboneConfig, feature_cfg: FeatureConfig, train_cfg: TrainingConfig, base_ckpt: Path, device: torch.device) -> ContinuousHeuristicModel:
    model = build_model(backbone_cfg, feature_cfg, train_cfg, device)
    safe_load_state(model, base_ckpt)
    model.eval()
    return model


def load_expert_model(backbone_cfg: BackboneConfig, feature_cfg: FeatureConfig, train_cfg: TrainingConfig, base_ckpt: Path, expert_ckpt: Path, device: torch.device) -> Tuple[ContinuousHeuristicModel, float]:
    model = build_model(backbone_cfg, feature_cfg, train_cfg, device)
    safe_load_state(model, base_ckpt)
    payload = torch.load(expert_ckpt, map_location="cpu")
    apply_lora(model, rank=int(payload.get("lora_rank", backbone_cfg.lora_rank)), alpha=float(payload.get("alpha", 1.0)), init_scale=0.01)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.eval()
    return model, float(payload.get("residual_bound", train_cfg.residual_bound_floor))


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

@dataclass
class EvalWorldBundle:
    world: World
    roadmap: Roadmap
    features: np.ndarray
    euclidean_h: np.ndarray
    oracle_cost: float


def make_eval_world_bundle(spec: AnchorSpec, roadmap_cfg: RoadmapConfig, feature_cfg: FeatureConfig, seed: int, retries: int) -> Optional[EvalWorldBundle]:
    for r in range(retries):
        world = build_world(spec, seed + 9973 * r, roadmap_cfg.min_start_goal_dist_frac)
        if world is None:
            continue
        roadmap = build_prm(world, roadmap_cfg, seed=seed + 9973 * r + 33)
        if roadmap is None or not roadmap.connected_to_goal[0]:
            continue
        features = make_features_for_roadmap(world, roadmap, feature_cfg)
        euclid = np.linalg.norm(roadmap.points - world.goal[None, :], axis=1).astype(np.float64)
        return EvalWorldBundle(world=world, roadmap=roadmap, features=features, euclidean_h=euclid, oracle_cost=float(roadmap.dist_to_goal[0]))
    return None


def aggregate_eval_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    key_fields = ["suite", "method", "backbone", "expert_task", "alpha", "budget"]
    for row in records:
        key = tuple(row.get(k, "") for k in key_fields)
        grouped.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        successes = [int(r["found"]) for r in rows]
        expansions = [float(r["expansions"]) for r in rows]
        ratios = [float(r["cost_ratio"]) for r in rows if math.isfinite(float(r.get("cost_ratio", float("nan"))))]
        oracle_costs = [float(r["oracle_cost"]) for r in rows]
        out_row = {k: v for k, v in zip(key_fields, key)}
        out_row.update({
            "episodes": len(rows),
            "success_rate": float(np.mean(successes)) if successes else float("nan"),
            "expansions_mean": finite_mean(expansions),
            "cost_ratio_mean": finite_mean(ratios),
            "oracle_cost_mean": finite_mean(oracle_costs),
            "nonfinite_heuristic_rows": int(sum(int(r.get("nonfinite_heuristic", 0)) for r in rows)),
            "heuristic_max_mean": finite_mean([float(r.get("heuristic_max", float("nan"))) for r in rows]),
            "delta_mean_mean": finite_mean([float(r.get("delta_mean", float("nan"))) for r in rows]),
            "correction_abs_mean": finite_mean([float(r.get("correction_abs_mean", float("nan"))) for r in rows]),
        })
        out.append(out_row)
    out.sort(key=lambda r: (str(r["suite"]), str(r["method"]), str(r["backbone"]), str(r["expert_task"]), float(r.get("alpha") or 0), int(r["budget"])))
    return out


def evaluate_all(
    out_dir: Path,
    specs: Dict[str, AnchorSpec],
    train_tasks: List[str],
    eval_suites: List[str],
    backbone_cfgs: Dict[str, BackboneConfig],
    backbones: List[str],
    feature_cfg: FeatureConfig,
    roadmap_cfg: RoadmapConfig,
    train_cfg: TrainingConfig,
    eval_cfg: EvalConfig,
    device: torch.device,
    seed: int,
    lora_alphas: List[float],
) -> Path:
    records: List[Dict[str, Any]] = []
    pred_cache: Dict[Tuple[str, str, str, float], Tuple[ContinuousHeuristicModel, Optional[ContinuousHeuristicModel], float]] = {}

    base_models: Dict[str, ContinuousHeuristicModel] = {}
    for backbone in backbones:
        cfg = backbone_cfgs[backbone]
        base_ckpt = model_checkpoint_path(out_dir, backbone, "avgbase")
        base_models[backbone] = load_base_model(cfg, feature_cfg, train_cfg, base_ckpt, device)

    # Preload experts lazily to keep memory lower. Each cache entry is
    # (base_model, expert_model, residual_bound).
    def get_expert(backbone: str, task: str, alpha: float) -> Tuple[ContinuousHeuristicModel, ContinuousHeuristicModel, float]:
        key = ("expert", backbone, task, float(alpha))
        if key in pred_cache:
            bm, em, bnd = pred_cache[key]
            assert em is not None
            return bm, em, bnd
        cfg = backbone_cfgs[backbone]
        base_ckpt = model_checkpoint_path(out_dir, backbone, "avgbase")
        expert_ckpt = model_checkpoint_path(out_dir, backbone, "expert", task=task, alpha=alpha)
        expert_model, bound = load_expert_model(cfg, feature_cfg, train_cfg, base_ckpt, expert_ckpt, device)
        bm = base_models[backbone]
        pred_cache[key] = (bm, expert_model, bound)
        return bm, expert_model, bound

    for suite in eval_suites:
        spec = specs[suite]
        print(f"[{now_str()}] evaluating suite {suite}", flush=True)
        valid_worlds = 0
        attempts = 0
        while valid_worlds < eval_cfg.eval_worlds and attempts < eval_cfg.eval_worlds * eval_cfg.max_eval_world_retries:
            attempts += 1
            w_seed = seed + 1_000_003 * (eval_suites.index(suite) + 1) + attempts * 7919
            bundle = make_eval_world_bundle(spec, roadmap_cfg, feature_cfg, w_seed, retries=1)
            if bundle is None:
                continue
            valid_worlds += 1
            world = bundle.world
            roadmap = bundle.roadmap
            features = bundle.features
            euclid_h = bundle.euclidean_h
            oracle_cost = bundle.oracle_cost

            method_heuristics: List[Tuple[str, str, str, float, np.ndarray, Dict[str, float]]] = []
            # method, backbone, expert_task, alpha, heuristic, diagnostics
            method_heuristics.append(("euclidean", "", "", 0.0, euclid_h, {"delta_mean": 0.0, "correction_abs_mean": 0.0}))

            for backbone in backbones:
                base_model = base_models[backbone]
                base_delta_norm = predict_norm_residuals(base_model, features, device, batch_size=2048)
                base_delta_norm = np.clip(base_delta_norm, 0.0, train_cfg.max_norm_residual)
                base_h = euclid_h + base_delta_norm * world.side_len
                method_heuristics.append(("avgbase", backbone, "", 0.0, base_h, {"delta_mean": float(np.mean(base_delta_norm)), "correction_abs_mean": 0.0}))

                expert_tasks = train_tasks if eval_cfg.eval_experts_all_suites or suite in train_tasks else []
                for alpha in lora_alphas:
                    for task in expert_tasks:
                        bm, em, bound = get_expert(backbone, task, alpha)
                        # bm should equal base_model; recompute here for clarity and safety.
                        bpred = base_delta_norm
                        epred = predict_norm_residuals(em, features, device, batch_size=2048)
                        correction = bound * np.tanh((epred - bpred) / max(bound, EPS))
                        final_delta_norm = np.clip(bpred + correction, 0.0, train_cfg.max_norm_residual)
                        h = euclid_h + final_delta_norm * world.side_len
                        method_heuristics.append((
                            "tasklora",
                            backbone,
                            task,
                            float(alpha),
                            h,
                            {"delta_mean": float(np.mean(final_delta_norm)), "correction_abs_mean": float(np.mean(np.abs(correction)))},
                        ))

            # Evaluate every explicit method at all budgets.
            for method, backbone, expert_task, alpha, h, diag in method_heuristics:
                nonfinite = int(not np.isfinite(h).all())
                if nonfinite:
                    h = np.nan_to_num(h, nan=train_cfg.max_norm_residual * world.side_len, posinf=train_cfg.max_norm_residual * world.side_len, neginf=0.0)
                for budget in eval_cfg.budgets:
                    res = astar_search(roadmap.adj, h, budget=budget)
                    found = bool(res["found"])
                    cost = float(res["cost"])
                    ratio = float(cost / oracle_cost) if found and oracle_cost > EPS else float("nan")
                    records.append({
                        "suite": suite,
                        "world_index": valid_worlds - 1,
                        "method": method,
                        "backbone": backbone,
                        "expert_task": expert_task,
                        "alpha": alpha,
                        "budget": int(budget),
                        "found": int(found),
                        "cost": cost,
                        "cost_ratio": ratio,
                        "oracle_cost": oracle_cost,
                        "expansions": int(res["expansions"]),
                        "closed": int(res["closed"]),
                        "roadmap_nodes": int(roadmap.points.shape[0]),
                        "roadmap_edges": int(sum(len(a) for a in roadmap.adj) // 2),
                        "obstacle_count": int(len(world.obstacles)),
                        "side_len": float(world.side_len),
                        "heuristic_max": float(np.max(h)) if len(h) else float("nan"),
                        "heuristic_min": float(np.min(h)) if len(h) else float("nan"),
                        "nonfinite_heuristic": nonfinite,
                        **diag,
                    })

            # Oracle expert is a post-hoc diagnostic: for each backbone/alpha,
            # choose the successful expert with best expansions at this budget;
            # if no expert succeeds, it records failure. It is not deployable.
            if eval_cfg.eval_oracle_expert:
                for backbone in backbones:
                    for alpha in lora_alphas:
                        task_rows = [r for r in records if r["suite"] == suite and r["world_index"] == valid_worlds - 1 and r["method"] == "tasklora" and r["backbone"] == backbone and float(r["alpha"]) == float(alpha)]
                        for budget in eval_cfg.budgets:
                            candidates = [r for r in task_rows if int(r["budget"]) == int(budget)]
                            if not candidates:
                                continue
                            succ = [r for r in candidates if int(r["found"]) == 1]
                            if succ:
                                best = min(succ, key=lambda r: (float(r["expansions"]), float(r.get("cost_ratio", 999.0))))
                                records.append({
                                    **{k: best[k] for k in best.keys()},
                                    "method": "oracle_tasklora",
                                    "expert_task": str(best["expert_task"]),
                                })
                            else:
                                # Preserve one representative failure row.
                                rep = min(candidates, key=lambda r: float(r["expansions"]))
                                records.append({
                                    **{k: rep[k] for k in rep.keys()},
                                    "method": "oracle_tasklora",
                                    "expert_task": "<none>",
                                    "found": 0,
                                    "cost": float("nan"),
                                    "cost_ratio": float("nan"),
                                })

        print(f"[{now_str()}] suite {suite}: valid_worlds={valid_worlds}/{eval_cfg.eval_worlds}", flush=True)

    raw_path = out_dir / "results" / "continuous_prm_eval_raw.csv"
    summary_path = out_dir / "results" / "continuous_prm_eval_summary.csv"
    write_csv(raw_path, records)
    summary = aggregate_eval_records(records)
    write_csv(summary_path, summary)
    write_json(out_dir / "results" / "continuous_prm_eval_summary.json", {"rows": summary})
    maybe_make_figures(summary_path, out_dir)
    return summary_path


def maybe_make_figures(summary_csv: Path, out_dir: Path) -> None:
    # Figure generation is intentionally opt-in. Importing matplotlib can be
    # memory-heavy on small CPU smoke/CI runs; set CONTINUOUS_PRM_MAKE_FIGURES=1
    # or pass --make-figures through the stage runner for report-ready plots.
    if str(os.environ.get("CONTINUOUS_PRM_MAKE_FIGURES", "0")).lower() not in ("1", "true", "yes", "on"):
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    rows: List[Dict[str, str]] = []
    with open(summary_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return
    fig_dir = ensure_dir(out_dir / "figures")
    # Keep figures simple and robust. Detailed report figures can be generated
    # later from the CSV once the run finishes.
    for suite in sorted(set(r["suite"] for r in rows)):
        plot_rows = [r for r in rows if r["suite"] == suite and r["method"] in ("euclidean", "avgbase", "oracle_tasklora")]
        if not plot_rows:
            continue
        plt.figure(figsize=(8, 5))
        labels = sorted(set((r["method"], r["backbone"], r["alpha"]) for r in plot_rows))
        for method, backbone, alpha in labels:
            series = [r for r in plot_rows if r["method"] == method and r["backbone"] == backbone and r["alpha"] == alpha]
            series.sort(key=lambda r: int(float(r["budget"])))
            xs = [int(float(r["budget"])) for r in series]
            ys = [float(r["success_rate"]) for r in series]
            label = method if not backbone else f"{method}:{backbone}"
            if method == "oracle_tasklora":
                label += f":a={float(alpha):g}"
            plt.plot(xs, ys, marker="o", label=label)
        plt.xlabel("A* expansion budget")
        plt.ylabel("Success rate")
        plt.title(f"Continuous PRM success: {suite}")
        plt.ylim(-0.03, 1.03)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / f"success__{sanitize_name(suite)}.png", dpi=160)
        plt.close()


# -----------------------------------------------------------------------------
# CLI orchestration
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continuous PRM residual-task LoRA v1")
    p.add_argument("--out-dir", type=str, default="runs/continuous_prm_residual_tasklora_v1")
    p.add_argument("--mode", type=str, default="full", choices=["full", "collect", "train", "eval"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--train-tasks", type=str, default=DEFAULT_TRAIN_TASKS)
    p.add_argument("--eval-suites", type=str, default=DEFAULT_EVAL_SUITES)
    p.add_argument("--backbones", type=str, default="hrm,onlstm")
    p.add_argument("--lora-alphas", type=str, default="1.0")
    p.add_argument("--train-worlds", type=int, default=100)
    p.add_argument("--nodes-per-world", type=int, default=160)
    p.add_argument("--eval-worlds", type=int, default=40)
    p.add_argument("--roadmap-nodes", type=int, default=256)
    p.add_argument("--roadmap-k", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--base-epochs", type=int, default=10)
    p.add_argument("--expert-epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--expert-lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--budgets", type=str, default="100,200,500,1000")
    p.add_argument("--nearest-obstacles", type=int, default=6)
    p.add_argument("--num-rays", type=int, default=16)
    p.add_argument("--ray-steps", type=int, default=80)
    p.add_argument("--max-norm-residual", type=float, default=4.0)
    p.add_argument("--residual-bound-quantile", type=float, default=0.95)
    p.add_argument("--residual-bound-floor", type=float, default=0.08)
    p.add_argument("--correction-l2", type=float, default=1.0e-3)
    p.add_argument("--train-bias-with-lora", action="store_true")
    p.add_argument("--eval-experts-all-suites", type=int, default=1)
    p.add_argument("--no-oracle-expert", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=int(os.environ.get("TORCH_THREADS", "1")), help="CPU torch thread count; default 1 avoids severe oversubscription in PRM smoke/local runs")
    # Backbone size knobs.
    p.add_argument("--head-hidden", type=int, default=256)
    p.add_argument("--hrm-hidden", type=int, default=192)
    p.add_argument("--hrm-layers", type=int, default=2)
    p.add_argument("--hrm-k-step", type=int, default=2)
    p.add_argument("--hrm-heads", type=int, default=4)
    p.add_argument("--onlstm-hidden", type=int, default=256)
    p.add_argument("--onlstm-layers", type=int, default=2)
    p.add_argument("--onlstm-chunk-size", type=int, default=8)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    args.out_dir = args.out_dir if args.out_dir else "/tmp/continuous_prm_smoke"
    # Ultra-small smoke settings are intentionally conservative. They verify
    # CLI/data/model plumbing without stressing the limited local sandbox.
    args.train_tasks = "C_open,C_clutter"
    args.eval_suites = "C_open,C_clutter"
    args.backbones = "hrm"
    args.lora_alphas = "1.0"
    args.train_worlds = 1
    args.nodes_per_world = 12
    args.eval_worlds = 1
    args.roadmap_nodes = 32
    args.roadmap_k = 6
    args.batch_size = 16
    args.base_epochs = 1
    args.expert_epochs = 1
    args.nearest_obstacles = 2
    args.num_rays = 4
    args.ray_steps = 12
    args.budgets = "10,30"
    args.hrm_hidden = 32
    args.hrm_layers = 1
    args.hrm_heads = 4
    args.head_hidden = 48


def build_configs_from_args(args: argparse.Namespace) -> Tuple[RoadmapConfig, FeatureConfig, TrainingConfig, EvalConfig]:
    roadmap_cfg = RoadmapConfig(n_nodes=int(args.roadmap_nodes), k_neighbors=int(args.roadmap_k))
    feature_cfg = FeatureConfig(nearest_obstacles=int(args.nearest_obstacles), num_rays=int(args.num_rays), ray_steps=int(args.ray_steps))
    train_cfg = TrainingConfig(
        batch_size=int(args.batch_size),
        base_epochs=int(args.base_epochs),
        expert_epochs=int(args.expert_epochs),
        lr=float(args.lr),
        expert_lr=float(args.expert_lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        max_norm_residual=float(args.max_norm_residual),
        residual_bound_quantile=float(args.residual_bound_quantile),
        residual_bound_floor=float(args.residual_bound_floor),
        correction_l2=float(args.correction_l2),
        num_workers=int(args.num_workers),
        train_bias_with_lora=bool(args.train_bias_with_lora),
    )
    eval_cfg = EvalConfig(
        budgets=tuple(int(x) for x in parse_csv(args.budgets)),
        eval_worlds=int(args.eval_worlds),
        eval_experts_all_suites=bool(int(args.eval_experts_all_suites)),
        eval_oracle_expert=not bool(args.no_oracle_expert),
    )
    return roadmap_cfg, feature_cfg, train_cfg, eval_cfg


def main() -> None:
    args = parse_args()
    apply_smoke_overrides(args)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
        try:
            torch.set_num_interop_threads(max(1, min(4, int(args.torch_threads))))
        except RuntimeError:
            pass
    set_global_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    data_dir = ensure_dir(out_dir / "datasets")
    ensure_dir(out_dir / "logs")
    specs = build_anchor_specs()
    train_tasks = parse_csv(args.train_tasks)
    eval_suites = parse_csv(args.eval_suites)
    backbones = parse_csv(args.backbones)
    lora_alphas = [float(x) for x in parse_csv(args.lora_alphas)]
    for name in train_tasks + eval_suites:
        if name not in specs:
            raise ValueError(f"unknown task/suite {name!r}; available={sorted(specs)}")
    roadmap_cfg, feature_cfg, train_cfg, eval_cfg = build_configs_from_args(args)
    backbone_cfgs = build_backbone_configs(args)
    for b in backbones:
        if b not in backbone_cfgs:
            raise ValueError(f"unknown backbone {b!r}; available={sorted(backbone_cfgs)}")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[{now_str()}] out_dir={out_dir}")
    print(f"[{now_str()}] device={device} train_tasks={train_tasks} eval_suites={eval_suites} backbones={backbones}")
    write_json(out_dir / "config.json", {
        "args": vars(args),
        "roadmap_cfg": asdict(roadmap_cfg),
        "feature_cfg": asdict(feature_cfg),
        "train_cfg": asdict(train_cfg),
        "eval_cfg": asdict(eval_cfg),
        "train_tasks": train_tasks,
        "eval_suites": eval_suites,
        "backbones": backbones,
        "lora_alphas": lora_alphas,
    })

    task_npz: Dict[str, Path] = {}
    if args.mode in ("full", "collect", "train"):
        for i, task in enumerate(train_tasks):
            task_npz[task] = collect_task_dataset(
                specs[task],
                out_dir=data_dir,
                split="train",
                n_worlds=int(args.train_worlds),
                nodes_per_world=int(args.nodes_per_world),
                roadmap_cfg=roadmap_cfg,
                feature_cfg=feature_cfg,
                seed=int(args.seed) + 10_000 * (i + 1),
            )
    else:
        for task in train_tasks:
            p = data_dir / f"{task}_train.npz"
            if not p.exists():
                raise FileNotFoundError(f"missing dataset {p}; run --mode collect first")
            task_npz[task] = p

    if args.mode in ("full", "train"):
        for backbone in backbones:
            cfg = backbone_cfgs[backbone]
            base_ckpt = train_avgbase(cfg, task_npz, out_dir, feature_cfg, train_cfg, device, seed=int(args.seed) + 100)
            for alpha in lora_alphas:
                for j, task in enumerate(train_tasks):
                    train_expert(
                        cfg,
                        task,
                        task_npz[task],
                        base_ckpt,
                        out_dir,
                        feature_cfg,
                        train_cfg,
                        device,
                        seed=int(args.seed) + 1_000 + 37 * j,
                        alpha=alpha,
                    )

    if args.mode in ("full", "eval"):
        summary_path = evaluate_all(
            out_dir=out_dir,
            specs=specs,
            train_tasks=train_tasks,
            eval_suites=eval_suites,
            backbone_cfgs=backbone_cfgs,
            backbones=backbones,
            feature_cfg=feature_cfg,
            roadmap_cfg=roadmap_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            device=device,
            seed=int(args.seed) + 200_000,
            lora_alphas=lora_alphas,
        )
        print(f"[{now_str()}] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
