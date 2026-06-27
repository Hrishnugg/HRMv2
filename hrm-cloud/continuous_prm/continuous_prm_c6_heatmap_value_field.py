#!/usr/bin/env python3
"""
C6: continuous PRM heatmap/value-field heuristic learning.

This experiment is intentionally separate from C1-C5. It reuses the C5 hard
world generator, rasterizes each continuous world into a goal-conditioned value
field, trains field models, and evaluates their interpolated heuristics on the
same PRM A* budget protocol used by C5.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import continuous_prm_common as C
import continuous_prm_c5_hard_obstacle_encoder as C5


INF = 1.0e30
EPS = 1.0e-9
INPUT_CHANNELS = 8
MODEL_NAMES = ("oracle", "unet", "onlstm", "hrm")
DEFAULT_TRAIN_TASKS = "C_hard_maze,C_hard_rooms"
DEFAULT_EVAL_SUITES = "C_hard_maze,C_hard_maze_dense,C_hard_rooms"
DEFAULT_BUDGETS = "128,136,144,152,168"


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_csv(text: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in parse_csv(text))


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


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


def read_csv(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def finite_mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def sanitize_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-=" else "_" for ch in str(s)).strip("_")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def install_c5_hard_runtime(sector_tokens: int = 16) -> None:
    C5.SECTOR_TOKENS = int(sector_tokens)
    C5.SOFT_RESIDUAL_CAP = False
    C5.install_runtime_extensions()


@dataclass
class C6Config:
    grid_size: int = 64
    train_worlds: int = 160
    eval_worlds: int = 80
    roadmap_nodes: int = 192
    roadmap_k: int = 7
    epochs: int = 16
    batch_size: int = 16
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    loss_rank_weight: float = 0.2
    loss_path_weight: float = 0.05
    loss_consistency_weight: float = 0.02
    rank_pairs: int = 256
    train_tasks: str = DEFAULT_TRAIN_TASKS
    eval_suites: str = DEFAULT_EVAL_SUITES
    budgets: str = DEFAULT_BUDGETS
    models: str = "oracle,unet,onlstm,hrm"
    seed: int = 1234
    max_world_retries: int = 120
    sector_tokens: int = 16
    torch_threads: int = 1
    cpu: bool = False
    make_figures: bool = True


def config_from_args(args: argparse.Namespace) -> C6Config:
    return C6Config(
        grid_size=int(args.grid_size),
        train_worlds=int(args.train_worlds),
        eval_worlds=int(args.eval_worlds),
        roadmap_nodes=int(args.roadmap_nodes),
        roadmap_k=int(args.roadmap_k),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        loss_rank_weight=float(args.loss_rank_weight),
        loss_path_weight=float(args.loss_path_weight),
        loss_consistency_weight=float(args.loss_consistency_weight),
        rank_pairs=int(args.rank_pairs),
        train_tasks=str(args.train_tasks),
        eval_suites=str(args.eval_suites),
        budgets=str(args.budgets),
        models=str(args.models),
        seed=int(args.seed),
        max_world_retries=int(args.max_world_retries),
        sector_tokens=int(args.sector_tokens),
        torch_threads=int(args.torch_threads),
        cpu=bool(args.cpu),
        make_figures=not bool(args.no_figures),
    )


def config_from_params(params: Dict[str, Any]) -> C6Config:
    cfg = C6Config()
    for key in asdict(cfg):
        if key in params:
            setattr(cfg, key, params[key])
    cfg.grid_size = int(cfg.grid_size)
    cfg.train_worlds = int(cfg.train_worlds)
    cfg.eval_worlds = int(cfg.eval_worlds)
    cfg.roadmap_nodes = int(cfg.roadmap_nodes)
    cfg.roadmap_k = int(cfg.roadmap_k)
    cfg.epochs = int(cfg.epochs)
    cfg.batch_size = int(cfg.batch_size)
    cfg.rank_pairs = int(cfg.rank_pairs)
    cfg.seed = int(cfg.seed)
    cfg.max_world_retries = int(cfg.max_world_retries)
    cfg.sector_tokens = int(cfg.sector_tokens)
    cfg.torch_threads = int(cfg.torch_threads)
    cfg.cpu = bool(cfg.cpu)
    cfg.make_figures = bool(cfg.make_figures)
    return cfg


def roadmap_config(cfg: C6Config) -> C.RoadmapConfig:
    return C.RoadmapConfig(n_nodes=int(cfg.roadmap_nodes), k_neighbors=int(cfg.roadmap_k))


def grid_centers(side_len: float, grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(grid_size, dtype=np.float64) + 0.5) * float(side_len) / float(grid_size)
    ys = (np.arange(grid_size, dtype=np.float64) + 0.5) * float(side_len) / float(grid_size)
    return np.meshgrid(xs, ys, indexing="ij")


def point_to_cell(p: np.ndarray, side_len: float, grid_size: int) -> Tuple[int, int]:
    ix = int(np.clip(math.floor(float(p[0]) / float(side_len) * grid_size), 0, grid_size - 1))
    iy = int(np.clip(math.floor(float(p[1]) / float(side_len) * grid_size), 0, grid_size - 1))
    return ix, iy


def nearest_free_cell(free: np.ndarray, cell: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    if free[cell]:
        return cell
    coords = np.argwhere(free)
    if len(coords) == 0:
        return None
    d2 = (coords[:, 0] - cell[0]) ** 2 + (coords[:, 1] - cell[1]) ** 2
    row = coords[int(np.argmin(d2))]
    return int(row[0]), int(row[1])


def rasterize_world(world: C.World, grid_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    side = float(world.side_len)
    xx, yy = grid_centers(side, grid_size)
    free = np.zeros((grid_size, grid_size), dtype=np.bool_)
    clearance = np.zeros((grid_size, grid_size), dtype=np.float32)
    for ix in range(grid_size):
        for iy in range(grid_size):
            p = np.array([xx[ix, iy], yy[ix, iy]], dtype=np.float64)
            c = C.obstacle_clearance(p, side, world.obstacles)
            free[ix, iy] = C.is_point_free(p, side, world.obstacles)
            clearance[ix, iy] = float(np.clip(c / max(EPS, side), -0.25, 0.25) / 0.25)
    occupancy = (~free).astype(np.float32)
    return occupancy, free, clearance


def grid_dijkstra_to_goal(world: C.World, free: np.ndarray) -> np.ndarray:
    grid_size = int(free.shape[0])
    side = float(world.side_len)
    cell_cost = side / float(grid_size)
    goal_cell = nearest_free_cell(free, point_to_cell(world.goal, side, grid_size))
    dist = np.full((grid_size, grid_size), np.inf, dtype=np.float64)
    if goal_cell is None:
        return dist
    gx, gy = goal_cell
    dist[gx, gy] = 0.0
    heap: List[Tuple[float, int, int]] = [(0.0, gx, gy)]
    nbrs = [
        (1, 0, cell_cost),
        (-1, 0, cell_cost),
        (0, 1, cell_cost),
        (0, -1, cell_cost),
        (1, 1, cell_cost * math.sqrt(2.0)),
        (1, -1, cell_cost * math.sqrt(2.0)),
        (-1, 1, cell_cost * math.sqrt(2.0)),
        (-1, -1, cell_cost * math.sqrt(2.0)),
    ]
    while heap:
        d, ix, iy = heapq.heappop(heap)
        if d != dist[ix, iy]:
            continue
        for dx, dy, w in nbrs:
            nx, ny = ix + dx, iy + dy
            if nx < 0 or nx >= grid_size or ny < 0 or ny >= grid_size or not free[nx, ny]:
                continue
            if dx and dy and (not free[ix + dx, iy] or not free[ix, iy + dy]):
                continue
            nd = d + w
            if nd < dist[nx, ny]:
                dist[nx, ny] = nd
                heapq.heappush(heap, (nd, nx, ny))
    return dist


def gaussian_channel(world: C.World, center: np.ndarray, grid_size: int, sigma_frac: float = 0.035) -> np.ndarray:
    side = float(world.side_len)
    xx, yy = grid_centers(side, grid_size)
    sigma = max(EPS, sigma_frac * side)
    d2 = (xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2
    return np.exp(-0.5 * d2 / (sigma * sigma)).astype(np.float32)


def euclidean_grid(world: C.World, grid_size: int) -> np.ndarray:
    side = float(world.side_len)
    xx, yy = grid_centers(side, grid_size)
    return np.sqrt((xx - float(world.goal[0])) ** 2 + (yy - float(world.goal[1])) ** 2).astype(np.float64)


def trace_path_mask(world: C.World, free: np.ndarray, dist: np.ndarray) -> np.ndarray:
    grid_size = int(free.shape[0])
    side = float(world.side_len)
    start = nearest_free_cell(free, point_to_cell(world.start, side, grid_size))
    goal = nearest_free_cell(free, point_to_cell(world.goal, side, grid_size))
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    if start is None or goal is None:
        return mask
    cur = start
    if not math.isfinite(float(dist[cur])):
        return mask
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for _ in range(grid_size * grid_size):
        ix, iy = cur
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    mask[nx, ny] = 1.0
        if cur == goal or float(dist[cur]) <= 1.0e-9:
            break
        best = cur
        best_d = float(dist[cur])
        for dx, dy in nbrs:
            nx, ny = ix + dx, iy + dy
            if nx < 0 or nx >= grid_size or ny < 0 or ny >= grid_size or not free[nx, ny]:
                continue
            if dx and dy and (not free[ix + dx, iy] or not free[ix, iy + dy]):
                continue
            nd = float(dist[nx, ny])
            if nd < best_d:
                best = (nx, ny)
                best_d = nd
        if best == cur:
            break
        cur = best
    return mask


def make_heatmap_example(world: C.World, grid_size: int) -> Dict[str, np.ndarray]:
    occupancy, free, clearance = rasterize_world(world, grid_size)
    dist = grid_dijkstra_to_goal(world, free)
    euclid = euclidean_grid(world, grid_size)
    side = float(world.side_len)
    reachable_free = free & np.isfinite(dist)
    target_distance = np.zeros_like(dist, dtype=np.float32)
    target_distance[reachable_free] = (dist[reachable_free] / max(EPS, side)).astype(np.float32)
    target_residual = np.zeros_like(dist, dtype=np.float32)
    residual = np.maximum(0.0, dist - euclid) / max(EPS, side)
    target_residual[reachable_free] = residual[reachable_free].astype(np.float32)
    xx, yy = grid_centers(side, grid_size)
    x_norm = (xx / side * 2.0 - 1.0).astype(np.float32)
    y_norm = (yy / side * 2.0 - 1.0).astype(np.float32)
    eu_norm = (euclid / max(EPS, side)).astype(np.float32)
    x = np.stack(
        [
            occupancy.astype(np.float32),
            reachable_free.astype(np.float32),
            clearance.astype(np.float32),
            gaussian_channel(world, world.goal, grid_size),
            gaussian_channel(world, world.start, grid_size),
            x_norm,
            y_norm,
            eu_norm,
        ],
        axis=0,
    )
    return {
        "x": x.astype(np.float32),
        "target_residual": target_residual.astype(np.float32),
        "target_distance": target_distance.astype(np.float32),
        "free_mask": reachable_free.astype(np.bool_),
        "path_mask": trace_path_mask(world, reachable_free, dist).astype(np.float32),
        "grid_distance_world": dist.astype(np.float64),
    }


def interpolate_grid_values(grid: np.ndarray, world: C.World, points: np.ndarray) -> np.ndarray:
    n = int(grid.shape[0])
    side = float(world.side_len)
    out = np.zeros(points.shape[0], dtype=np.float64)
    for idx, p in enumerate(points):
        fx = float(p[0]) / side * n - 0.5
        fy = float(p[1]) / side * n - 0.5
        x0 = int(np.clip(math.floor(fx), 0, n - 1))
        y0 = int(np.clip(math.floor(fy), 0, n - 1))
        x1 = min(n - 1, x0 + 1)
        y1 = min(n - 1, y0 + 1)
        tx = float(np.clip(fx - x0, 0.0, 1.0))
        ty = float(np.clip(fy - y0, 0.0, 1.0))
        corners = np.array([grid[x0, y0], grid[x1, y0], grid[x0, y1], grid[x1, y1]], dtype=np.float64)
        if np.isfinite(corners).all():
            out[idx] = (
                (1.0 - tx) * (1.0 - ty) * corners[0]
                + tx * (1.0 - ty) * corners[1]
                + (1.0 - tx) * ty * corners[2]
                + tx * ty * corners[3]
            )
        else:
            finite = corners[np.isfinite(corners)]
            out[idx] = float(finite.min()) if len(finite) else float("inf")
    return out


def collect_dataset(
    spec: C.AnchorSpec,
    out_dir: Path,
    split: str,
    n_worlds: int,
    cfg: C6Config,
    seed: int,
) -> Path:
    ensure_dir(out_dir)
    npz_path = out_dir / f"{spec.name}_{split}.npz"
    meta_path = out_dir / f"{spec.name}_{split}.json"
    if npz_path.exists() and meta_path.exists():
        return npz_path
    rng = random.Random(seed)
    roadmap_cfg = roadmap_config(cfg)
    xs: List[np.ndarray] = []
    residuals: List[np.ndarray] = []
    distances: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    paths: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    worlds_done = 0
    attempts = 0
    while worlds_done < n_worlds and attempts < n_worlds * int(cfg.max_world_retries):
        attempts += 1
        w_seed = rng.randint(0, 2**31 - 1)
        world = C.build_world(spec, w_seed, roadmap_cfg.min_start_goal_dist_frac)
        if world is None:
            continue
        roadmap = C.build_prm(world, roadmap_cfg, seed=w_seed + 17)
        if roadmap is None or not roadmap.connected_to_goal[0]:
            continue
        ex = make_heatmap_example(world, int(cfg.grid_size))
        if int(ex["free_mask"].sum()) < max(16, int(cfg.grid_size)):
            continue
        xs.append(ex["x"])
        residuals.append(ex["target_residual"])
        distances.append(ex["target_distance"])
        masks.append(ex["free_mask"])
        paths.append(ex["path_mask"])
        metas.append(
            {
                **world.meta,
                "world_idx": worlds_done,
                "seed": w_seed,
                "task": spec.name,
                "side_len": float(world.side_len),
                "start": [float(v) for v in world.start],
                "goal": [float(v) for v in world.goal],
                "obstacle_count": int(len(world.obstacles)),
                "free_cells": int(ex["free_mask"].sum()),
                "roadmap_nodes": int(roadmap.points.shape[0]),
                "roadmap_edges": int(sum(len(a) for a in roadmap.adj) // 2),
                "oracle_cost": float(roadmap.dist_to_goal[0]),
            }
        )
        worlds_done += 1
        print(f"[{now_str()}] collected C6 {spec.name}/{split}: {worlds_done}/{n_worlds}", flush=True)
    if not xs:
        raise RuntimeError(f"failed to collect any C6 worlds for {spec.name}/{split}")
    np.savez_compressed(
        npz_path,
        x=np.stack(xs, axis=0).astype(np.float32),
        target_residual=np.stack(residuals, axis=0).astype(np.float32),
        target_distance=np.stack(distances, axis=0).astype(np.float32),
        free_mask=np.stack(masks, axis=0).astype(np.bool_),
        path_mask=np.stack(paths, axis=0).astype(np.float32),
    )
    write_json(
        meta_path,
        {
            "task": spec.name,
            "split": split,
            "n_worlds_requested": int(n_worlds),
            "n_worlds_collected": int(worlds_done),
            "grid_size": int(cfg.grid_size),
            "input_channels": INPUT_CHANNELS,
            "worlds": metas,
            "spec": asdict(spec),
        },
    )
    return npz_path


def merge_dataset_shards(shard_npzs: Sequence[Path], out_npz: Path, shard_metas: Sequence[Path], out_meta: Path, task: str) -> Path:
    arrays: Dict[str, List[np.ndarray]] = {
        "x": [],
        "target_residual": [],
        "target_distance": [],
        "free_mask": [],
        "path_mask": [],
    }
    worlds: List[Dict[str, Any]] = []
    for p in shard_npzs:
        data = np.load(p)
        for key in arrays:
            arrays[key].append(data[key])
    for p in shard_metas:
        meta = read_json(p)
        worlds.extend(meta.get("worlds", []))
    ensure_dir(out_npz.parent)
    np.savez_compressed(out_npz, **{k: np.concatenate(v, axis=0) for k, v in arrays.items()})
    write_json(out_meta, {"task": task, "n_worlds_collected": len(worlds), "worlds": worlds})
    return out_npz


class HeatmapDataset(Dataset):
    def __init__(self, paths: Sequence[str | Path]):
        xs: List[np.ndarray] = []
        residuals: List[np.ndarray] = []
        distances: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        paths_arr: List[np.ndarray] = []
        for path in paths:
            data = np.load(path)
            xs.append(data["x"].astype(np.float32))
            residuals.append(data["target_residual"].astype(np.float32))
            distances.append(data["target_distance"].astype(np.float32))
            masks.append(data["free_mask"].astype(np.bool_))
            paths_arr.append(data["path_mask"].astype(np.float32))
        self.x = np.concatenate(xs, axis=0)
        self.target_residual = np.concatenate(residuals, axis=0)
        self.target_distance = np.concatenate(distances, axis=0)
        self.free_mask = np.concatenate(masks, axis=0)
        self.path_mask = np.concatenate(paths_arr, axis=0)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[idx]),
            "target_residual": torch.from_numpy(self.target_residual[idx]),
            "target_distance": torch.from_numpy(self.target_distance[idx]),
            "free_mask": torch.from_numpy(self.free_mask[idx]),
            "path_mask": torch.from_numpy(self.path_mask[idx]),
        }


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(max(1, min(8, out_ch // 4)), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(max(1, min(8, out_ch // 4)), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetField(nn.Module):
    def __init__(self, in_channels: int = INPUT_CHANNELS, base: int = 32):
        super().__init__()
        self.e1 = DoubleConv(in_channels, base)
        self.e2 = DoubleConv(base, base * 2)
        self.e3 = DoubleConv(base * 2, base * 4)
        self.e4 = DoubleConv(base * 4, base * 8)
        self.b = DoubleConv(base * 8, base * 8)
        self.p = nn.MaxPool2d(2)
        self.u4 = nn.ConvTranspose2d(base * 8, base * 8, 2, stride=2)
        self.d4 = DoubleConv(base * 16, base * 4)
        self.u3 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.d3 = DoubleConv(base * 8, base * 2)
        self.u2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.d2 = DoubleConv(base * 4, base)
        self.u1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.d1 = DoubleConv(base * 2, base)
        self.residual_head = nn.Conv2d(base, 1, 1)
        self.path_head = nn.Conv2d(base, 1, 1)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.p(e1))
        e3 = self.e3(self.p(e2))
        e4 = self.e4(self.p(e3))
        b = self.b(self.p(e4))
        x = self.u4(b)
        x = self.d4(torch.cat([x, e4], dim=1))
        x = self.u3(x)
        x = self.d3(torch.cat([x, e3], dim=1))
        x = self.u2(x)
        x = self.d2(torch.cat([x, e2], dim=1))
        x = self.u1(x)
        return self.d1(torch.cat([x, e1], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_head(self._features(x))

    def forward_path(self, x: torch.Tensor) -> torch.Tensor:
        return self.path_head(self._features(x))

    def forward_both(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self._features(x)
        return self.residual_head(feat), self.path_head(feat)


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
        hdim = self.hidden_dim
        i, f, o, g = gates[:, : 4 * hdim].chunk(4, dim=-1)
        f_hat_lin, i_hat_lin = gates[:, 4 * hdim :].chunk(2, dim=-1)
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


class PatchDecoder(nn.Module):
    def __init__(self, hidden_dim: int, out_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, out_hidden, 4, stride=4),
            nn.GELU(),
            nn.Conv2d(out_hidden, out_hidden, 3, padding=1),
            nn.GELU(),
        )
        self.residual_head = nn.Conv2d(out_hidden, 1, 1)
        self.path_head = nn.Conv2d(out_hidden, 1, 1)

    def decode_features(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ONLSTMField(nn.Module):
    def __init__(self, in_channels: int = INPUT_CHANNELS, hidden_dim: int = 256, layers: int = 2, chunk_size: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stem = nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=4)
        self.cells = nn.ModuleList([ONLSTMCell(hidden_dim, hidden_dim, chunk_size=chunk_size) for _ in range(layers)])
        self.decoder = PatchDecoder(hidden_dim)

    def _patch_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        b, c, ph, pw = z.shape
        seq = z.flatten(2).transpose(1, 2)
        h = [torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype) for _ in self.cells]
        cell = [torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype) for _ in self.cells]
        outs: List[torch.Tensor] = []
        for t in range(seq.shape[1]):
            inp = seq[:, t]
            new_h: List[torch.Tensor] = []
            new_c: List[torch.Tensor] = []
            for i, oncell in enumerate(self.cells):
                hi, ci = oncell(inp, (h[i], cell[i]))
                inp = hi
                new_h.append(hi)
                new_c.append(ci)
            h, cell = new_h, new_c
            outs.append(h[-1])
        return torch.stack(outs, dim=1).transpose(1, 2).reshape(b, c, ph, pw)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder.decode_features(self._patch_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder.residual_head(self._features(x))

    def forward_path(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder.path_head(self._features(x))

    def forward_both(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self._features(x)
        return self.decoder.residual_head(feat), self.decoder.path_head(feat)


class SpatialRecurrentBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        h = (x + state) * 0.7071
        attn, _ = self.attn(self.norm1(h), self.norm1(h), self.norm1(h), need_weights=False)
        cand = h + attn
        cand = cand + self.ff(self.norm2(cand))
        z = torch.sigmoid(self.gate(torch.cat([cand, state], dim=-1)))
        return z * cand + (1.0 - z) * state


class HRMField(nn.Module):
    def __init__(self, in_channels: int = INPUT_CHANNELS, hidden_dim: int = 256, layers: int = 2, heads: int = 4, steps: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.steps = int(steps)
        self.stem = nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=4)
        self.low = nn.ModuleList([SpatialRecurrentBlock(hidden_dim, heads) for _ in range(layers)])
        self.high = nn.ModuleList([SpatialRecurrentBlock(hidden_dim, heads) for _ in range(layers)])
        self.decoder = PatchDecoder(hidden_dim)

    def _patch_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        b, c, ph, pw = z.shape
        tokens = z.flatten(2).transpose(1, 2)
        low_state = [torch.zeros_like(tokens) for _ in self.low]
        high_state = [torch.zeros_like(tokens) for _ in self.high]
        for step in range(max(1, self.steps)):
            high_in = low_state[-1].detach() if low_state else tokens
            new_high: List[torch.Tensor] = []
            for i, block in enumerate(self.high):
                h = block(high_in, high_state[i])
                new_high.append(h)
                high_in = h
            high_state = new_high
            low_in = tokens + high_state[-1]
            new_low: List[torch.Tensor] = []
            for i, block in enumerate(self.low):
                h = block(low_in, low_state[i])
                new_low.append(h)
                low_in = h
            low_state = new_low
        return low_state[-1].transpose(1, 2).reshape(b, c, ph, pw)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder.decode_features(self._patch_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder.residual_head(self._features(x))

    def forward_path(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder.path_head(self._features(x))

    def forward_both(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self._features(x)
        return self.decoder.residual_head(feat), self.decoder.path_head(feat)


def build_model(name: str) -> nn.Module:
    if name == "unet":
        return UNetField()
    if name == "onlstm":
        return ONLSTMField()
    if name == "hrm":
        return HRMField()
    raise ValueError(f"unknown learned C6 model {name!r}")


def checkpoint_path(out_dir: str | Path, model_name: str) -> Path:
    return ensure_dir(Path(out_dir) / "checkpoints") / f"c6_heatmap__{model_name}.pt"


def model_output_residual(raw: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw.squeeze(1))


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    if int(valid.sum().item()) == 0:
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[valid], target[valid])


def ranking_loss(pred: torch.Tensor, target_distance: torch.Tensor, mask: torch.Tensor, pairs: int) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    bsz = pred.shape[0]
    flat_pred = pred.reshape(bsz, -1)
    flat_target = target_distance.reshape(bsz, -1)
    flat_mask = mask.reshape(bsz, -1).bool()
    for b in range(bsz):
        idx = torch.nonzero(flat_mask[b], as_tuple=False).squeeze(1)
        if idx.numel() < 2:
            continue
        ia = idx[torch.randint(0, idx.numel(), (pairs,), device=idx.device)]
        ib = idx[torch.randint(0, idx.numel(), (pairs,), device=idx.device)]
        td = flat_target[b, ia] - flat_target[b, ib]
        keep = torch.abs(td) > 1.0e-4
        if int(keep.sum().item()) == 0:
            continue
        sign = torch.sign(td[keep])
        pd = flat_pred[b, ia[keep]] - flat_pred[b, ib[keep]]
        losses.append(F.softplus(-sign * pd / 0.25).mean())
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def path_bce_loss_from_logits(logits: torch.Tensor, path_mask: torch.Tensor, free_mask: torch.Tensor) -> torch.Tensor:
    logits = logits.squeeze(1)
    valid = free_mask.bool()
    if int(valid.sum().item()) == 0:
        return logits.sum() * 0.0
    target = (path_mask > 0.5).float()
    pos = float((target[valid] > 0.5).sum().item())
    neg = float(valid.sum().item()) - pos
    pos_weight = torch.tensor(min(20.0, max(1.0, neg / max(1.0, pos))), device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits[valid], target[valid], pos_weight=pos_weight)


def consistency_loss(pred_residual: torch.Tensor, x: torch.Tensor, free_mask: torch.Tensor) -> torch.Tensor:
    pred_dist = pred_residual + x[:, 7]
    mask = free_mask.bool()
    h = pred_dist.shape[-1]
    step = 1.0 / max(1.0, float(h))
    losses: List[torch.Tensor] = []
    right = mask[:, :, 1:] & mask[:, :, :-1]
    if int(right.sum().item()) > 0:
        diff = torch.abs(pred_dist[:, :, 1:] - pred_dist[:, :, :-1]) - step
        losses.append(torch.relu(diff[right]).pow(2).mean())
    down = mask[:, 1:, :] & mask[:, :-1, :]
    if int(down.sum().item()) > 0:
        diff = torch.abs(pred_dist[:, 1:, :] - pred_dist[:, :-1, :]) - step
        losses.append(torch.relu(diff[down]).pow(2).mean())
    return torch.stack(losses).mean() if losses else pred_residual.sum() * 0.0


def train_model(
    model_name: str,
    dataset_paths: Sequence[str | Path],
    out_dir: str | Path,
    cfg: C6Config,
    device: torch.device,
) -> Path:
    if model_name == "oracle":
        raise ValueError("oracle is not trainable")
    ckpt = checkpoint_path(out_dir, model_name)
    ds = HeatmapDataset(dataset_paths)
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True, num_workers=0)
    model = build_model(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    log_rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        sums: Dict[str, float] = {"loss": 0.0, "value": 0.0, "rank": 0.0, "path": 0.0, "consistency": 0.0}
        batches = 0
        for batch in loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            target = batch["target_residual"].to(device=device, dtype=torch.float32)
            target_dist = batch["target_distance"].to(device=device, dtype=torch.float32)
            mask = batch["free_mask"].to(device=device)
            path = batch["path_mask"].to(device=device, dtype=torch.float32)
            if hasattr(model, "forward_both"):
                raw_pred, path_logits = model.forward_both(x)
            else:
                raw_pred = model(x)
                path_logits = model.forward_path(x) if hasattr(model, "forward_path") else torch.zeros_like(raw_pred)
            pred = model_output_residual(raw_pred)
            value = masked_smooth_l1(pred, target, mask)
            rank = ranking_loss(pred, target_dist, mask, int(cfg.rank_pairs))
            path_loss = path_bce_loss_from_logits(path_logits, path, mask)
            cons = consistency_loss(pred, x, mask)
            loss = (
                value
                + float(cfg.loss_rank_weight) * rank
                + float(cfg.loss_path_weight) * path_loss
                + float(cfg.loss_consistency_weight) * cons
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(cfg.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            batches += 1
            sums["loss"] += float(loss.detach().cpu())
            sums["value"] += float(value.detach().cpu())
            sums["rank"] += float(rank.detach().cpu())
            sums["path"] += float(path_loss.detach().cpu())
            sums["consistency"] += float(cons.detach().cpu())
        row = {"epoch": epoch, "model": model_name, **{k: v / max(1, batches) for k, v in sums.items()}}
        log_rows.append(row)
        print(f"[{now_str()}] C6 train {model_name} epoch {epoch}/{cfg.epochs}: loss={row['loss']:.5f}", flush=True)
    torch.save({"model": model.state_dict(), "model_name": model_name, "cfg": asdict(cfg), "log": log_rows}, ckpt)
    write_json(Path(out_dir) / "logs" / f"train_c6_{model_name}.json", {"rows": log_rows})
    return ckpt


def load_model(model_name: str, ckpt: str | Path, device: torch.device) -> nn.Module:
    model = build_model(model_name).to(device)
    payload = torch.load(ckpt, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_residual_grid(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    xt = torch.from_numpy(x[None]).to(device=device, dtype=torch.float32)
    pred = model_output_residual(model(xt))[0].detach().cpu().numpy().astype(np.float64)
    return pred


def rankdata_np(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    ra = rankdata_np(a[mask])
    rb = rankdata_np(b[mask])
    sa = float(np.std(ra))
    sb = float(np.std(rb))
    if sa <= EPS or sb <= EPS:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def make_oracle_heuristic(world: C.World, roadmap: C.Roadmap, grid_distance: np.ndarray, euclid_h: np.ndarray) -> np.ndarray:
    vals = interpolate_grid_values(grid_distance, world, roadmap.points)
    finite = np.isfinite(vals)
    fill = float(np.nanmax(vals[finite]) + world.side_len) if finite.any() else float(10.0 * world.side_len)
    vals = np.nan_to_num(vals, nan=fill, posinf=fill, neginf=0.0)
    return np.maximum(euclid_h, vals)


def maybe_save_heatmap_figure(
    out_dir: Path,
    suite: str,
    world_idx: int,
    method: str,
    x: np.ndarray,
    heatmap: np.ndarray,
    enabled: bool,
) -> None:
    if not enabled or world_idx >= 2 or method not in {"grid_oracle", "unet", "onlstm", "hrm"}:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_dir = ensure_dir(out_dir / "figures" / "heatmaps")
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(x[0].T, origin="lower", cmap="gray_r")
    plt.title("occupancy")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap.T, origin="lower", cmap="viridis")
    plt.title(method)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{sanitize_name(suite)}__w{world_idx:03d}__{sanitize_name(method)}.png", dpi=140)
    plt.close()


def evaluate_shard(
    out_dir: str | Path,
    cfg: C6Config,
    suite: str,
    suite_idx: int,
    world_start: int,
    world_count: int,
    shard_idx: int,
    checkpoint_dir: Optional[str | Path] = None,
) -> Path:
    install_c5_hard_runtime(cfg.sector_tokens)
    specs = C.build_anchor_specs()
    spec = specs[suite]
    out_dir = Path(out_dir)
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else out_dir
    device = torch.device("cpu" if cfg.cpu or not torch.cuda.is_available() else "cuda")
    model_names = [m for m in parse_csv(cfg.models) if m != "oracle"]
    models: Dict[str, nn.Module] = {}
    for name in model_names:
        ckpt = checkpoint_path(ckpt_dir, name)
        if not ckpt.exists():
            raise FileNotFoundError(f"missing C6 checkpoint for {name}: {ckpt}")
        models[name] = load_model(name, ckpt, device)
    roadmap_cfg = roadmap_config(cfg)
    budgets = parse_int_csv(cfg.budgets)
    records: List[Dict[str, Any]] = []
    valid_worlds = 0
    attempts = 0
    while valid_worlds < world_count and attempts < world_count * int(cfg.max_world_retries):
        attempts += 1
        logical_world_idx = world_start + valid_worlds
        w_seed = int(cfg.seed) + 330_000 + 1_000_003 * (suite_idx + 1) + (logical_world_idx + 1) * 7919 + attempts
        world = C.build_world(spec, w_seed, roadmap_cfg.min_start_goal_dist_frac)
        if world is None:
            continue
        roadmap = C.build_prm(world, roadmap_cfg, seed=w_seed + 17)
        if roadmap is None or not roadmap.connected_to_goal[0]:
            continue
        ex = make_heatmap_example(world, int(cfg.grid_size))
        euclid_h = np.linalg.norm(roadmap.points - world.goal[None, :], axis=1).astype(np.float64)
        oracle_cost = float(roadmap.dist_to_goal[0])
        methods: List[Tuple[str, np.ndarray, np.ndarray]] = [("euclidean", euclid_h, euclid_h.reshape(1, -1))]
        if "oracle" in parse_csv(cfg.models):
            oracle_h = make_oracle_heuristic(world, roadmap, ex["grid_distance_world"], euclid_h)
            methods.append(("grid_oracle", oracle_h, ex["grid_distance_world"]))
            maybe_save_heatmap_figure(out_dir, suite, logical_world_idx, "grid_oracle", ex["x"], ex["grid_distance_world"], cfg.make_figures)
        for name, model in models.items():
            pred_resid_norm = predict_residual_grid(model, ex["x"], device)
            pred_resid_norm = np.nan_to_num(pred_resid_norm, nan=0.0, posinf=10.0, neginf=0.0)
            pred_resid_norm = np.maximum(0.0, pred_resid_norm)
            node_resid = interpolate_grid_values(pred_resid_norm, world, roadmap.points)
            node_resid = np.nan_to_num(node_resid, nan=0.0, posinf=10.0, neginf=0.0)
            h = euclid_h + np.maximum(0.0, node_resid) * float(world.side_len)
            methods.append((name, h, pred_resid_norm))
            maybe_save_heatmap_figure(out_dir, suite, logical_world_idx, name, ex["x"], pred_resid_norm, cfg.make_figures)
        valid_worlds += 1
        for method, h, heatmap in methods:
            nonfinite = int(not np.isfinite(h).all())
            if nonfinite:
                h = np.nan_to_num(h, nan=10.0 * world.side_len, posinf=10.0 * world.side_len, neginf=0.0)
            sp = spearman_corr(h, roadmap.dist_to_goal)
            hm_vals = heatmap[np.isfinite(heatmap)] if heatmap.ndim == 2 else h
            for budget in budgets:
                res = C.astar_search(roadmap.adj, h, budget=int(budget))
                found = bool(res["found"])
                cost = float(res["cost"])
                records.append(
                    {
                        "stage": "c6",
                        "suite": suite,
                        "world_index": logical_world_idx,
                        "method": method,
                        "backbone": method if method in {"unet", "onlstm", "hrm"} else "",
                        "expert_task": "",
                        "alpha": 0.0,
                        "budget": int(budget),
                        "found": int(found),
                        "cost": cost,
                        "cost_ratio": float(cost / oracle_cost) if found and oracle_cost > EPS else float("nan"),
                        "oracle_cost": oracle_cost,
                        "expansions": int(res["expansions"]),
                        "closed": int(res["closed"]),
                        "roadmap_nodes": int(roadmap.points.shape[0]),
                        "roadmap_edges": int(sum(len(a) for a in roadmap.adj) // 2),
                        "obstacle_count": int(len(world.obstacles)),
                        "side_len": float(world.side_len),
                        "heuristic_max": float(np.max(h)) if len(h) else float("nan"),
                        "heuristic_min": float(np.min(h)) if len(h) else float("nan"),
                        "heuristic_mean": float(np.mean(h)) if len(h) else float("nan"),
                        "heuristic_std": float(np.std(h)) if len(h) else float("nan"),
                        "spearman_to_prm_oracle": sp,
                        "heatmap_min": float(np.min(hm_vals)) if len(hm_vals) else float("nan"),
                        "heatmap_max": float(np.max(hm_vals)) if len(hm_vals) else float("nan"),
                        "heatmap_mean": float(np.mean(hm_vals)) if len(hm_vals) else float("nan"),
                        "heatmap_std": float(np.std(hm_vals)) if len(hm_vals) else float("nan"),
                        "nonfinite_heuristic": nonfinite,
                    }
                )
    out = ensure_dir(out_dir / "results" / "_shards" / "c6" / suite) / f"shard_{shard_idx:04d}.csv"
    write_csv(out, records)
    return out


def aggregate_eval_records(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["suite"]), str(row["method"]), int(row["budget"])), []).append(row)
    out: List[Dict[str, Any]] = []
    for (suite, method, budget), group in grouped.items():
        successes = [int(r["found"]) for r in group]
        ratios = [float(r["cost_ratio"]) for r in group if math.isfinite(float(r.get("cost_ratio", float("nan"))))]
        out.append(
            {
                "suite": suite,
                "method": method,
                "backbone": method if method in {"unet", "onlstm", "hrm"} else "",
                "expert_task": "",
                "alpha": 0.0,
                "budget": int(budget),
                "episodes": len(group),
                "success_rate": float(np.mean(successes)) if successes else float("nan"),
                "expansions_mean": finite_mean([float(r["expansions"]) for r in group]),
                "cost_ratio_mean": finite_mean(ratios),
                "oracle_cost_mean": finite_mean([float(r["oracle_cost"]) for r in group]),
                "heuristic_std_mean": finite_mean([float(r.get("heuristic_std", float("nan"))) for r in group]),
                "spearman_to_prm_oracle_mean": finite_mean([float(r.get("spearman_to_prm_oracle", float("nan"))) for r in group]),
                "heatmap_std_mean": finite_mean([float(r.get("heatmap_std", float("nan"))) for r in group]),
                "nonfinite_heuristic_rows": int(sum(int(r.get("nonfinite_heuristic", 0)) for r in group)),
            }
        )
    out.sort(key=lambda r: (str(r["suite"]), str(r["method"]), int(r["budget"])))
    return out


def mcnemar_exact_p(gain: int, loss: int) -> float:
    n = int(gain) + int(loss)
    if n == 0:
        return 1.0
    k = min(int(gain), int(loss))
    prob = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * prob))


def bh_q_values(pvals: Sequence[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [1.0] * m
    running = 1.0
    for rank_from_end, idx in enumerate(reversed(order), start=1):
        rank = m - rank_from_end + 1
        val = float(pvals[idx]) * m / max(1, rank)
        running = min(running, val)
        q[idx] = min(1.0, running)
    return q


def analyze_significance(raw_rows: Sequence[Dict[str, Any]], out_dir: str | Path) -> Tuple[Path, Path]:
    by_key: Dict[Tuple[str, int, int], Dict[str, Dict[str, Any]]] = {}
    for row in raw_rows:
        key = (str(row["suite"]), int(row["budget"]), int(row["world_index"]))
        by_key.setdefault(key, {})[str(row["method"])] = row
    comparisons: List[Dict[str, Any]] = []
    for suite, budget, _world_idx in sorted(by_key):
        methods = by_key[(suite, budget, _world_idx)]
        if "euclidean" not in methods:
            continue
    by_suite_budget_method: Dict[Tuple[str, int, str], List[Tuple[int, int, int, int]]] = {}
    for (suite, budget, _world_idx), methods in by_key.items():
        base = methods.get("euclidean")
        if base is None:
            continue
        for method, row in methods.items():
            if method == "euclidean":
                continue
            by_suite_budget_method.setdefault((suite, budget, method), []).append(
                (int(base["found"]), int(row["found"]), int(base["expansions"]), int(row["expansions"]))
            )
    pvals: List[float] = []
    for (suite, budget, method), pairs in sorted(by_suite_budget_method.items()):
        base_s = float(np.mean([p[0] for p in pairs])) if pairs else float("nan")
        meth_s = float(np.mean([p[1] for p in pairs])) if pairs else float("nan")
        gain = sum(1 for b, m, _, _ in pairs if m == 1 and b == 0)
        loss = sum(1 for b, m, _, _ in pairs if m == 0 and b == 1)
        p = mcnemar_exact_p(gain, loss)
        pvals.append(p)
        comparisons.append(
            {
                "suite": suite,
                "budget": int(budget),
                "method": method,
                "episodes": len(pairs),
                "euclidean_success": base_s,
                "method_success": meth_s,
                "delta": meth_s - base_s,
                "paired_gains": gain,
                "paired_losses": loss,
                "mcnemar_p": p,
                "expansion_delta": finite_mean([float(p[3] - p[2]) for p in pairs]),
            }
        )
    qvals = bh_q_values(pvals)
    for row, q in zip(comparisons, qvals):
        row["bh_q"] = q
        row["claim_candidate"] = int(
            0.50 <= float(row["euclidean_success"]) <= 0.70
            and float(row["delta"]) >= 0.10
            and float(row["bh_q"]) <= 0.05
            and row["method"] not in {"grid_oracle"}
        )
    sig_path = Path(out_dir) / "results" / "continuous_prm_c6_significance.csv"
    write_csv(sig_path, comparisons)
    md_path = Path(out_dir) / "results" / "continuous_prm_c6_significance.md"
    claim_rows = [r for r in comparisons if int(r.get("claim_candidate", 0)) == 1]
    target_rows = [r for r in comparisons if 0.50 <= float(r["euclidean_success"]) <= 0.70]
    lines = [
        "# C6 Heatmap Value-Field Significance Analysis",
        "",
        "## Claim Candidates",
        "",
        "|Suite|Budget|Euclid|Method|Delta|Gain|Loss|McNemar p|BH q|Exp Delta|",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in claim_rows:
        lines.append(
            f"|{r['suite']}|{r['budget']}|{float(r['euclidean_success']):.3f}|{r['method']}|"
            f"{float(r['delta']):.3f}|{r['paired_gains']}|{r['paired_losses']}|"
            f"{float(r['mcnemar_p']):.3g}|{float(r['bh_q']):.3g}|{float(r['expansion_delta']):.3f}|"
        )
    if not claim_rows:
        lines.append("|<none>|-|-|-|-|-|-|-|-|-|")
    lines.extend(
        [
            "",
            "## Target-Band Comparisons",
            "",
            "|Suite|Budget|Euclid|Method|Method Success|Delta|Gain|Loss|BH q|Exp Delta|",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in target_rows:
        lines.append(
            f"|{r['suite']}|{r['budget']}|{float(r['euclidean_success']):.3f}|{r['method']}|"
            f"{float(r['method_success']):.3f}|{float(r['delta']):.3f}|{r['paired_gains']}|"
            f"{r['paired_losses']}|{float(r['bh_q']):.3g}|{float(r['expansion_delta']):.3f}|"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `grid_oracle` is an upper-bound diagnostic and is excluded from claim candidates.",
            "- McNemar p-values are paired by suite, budget, and logical world index.",
            "- BH q-values are corrected across all method-vs-Euclidean C6 comparisons.",
        ]
    )
    ensure_dir(md_path.parent)
    md_path.write_text("\n".join(lines) + "\n")
    return sig_path, md_path


def merge_eval_shards(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    rows: List[Dict[str, Any]] = []
    for p in sorted((out_dir / "results" / "_shards" / "c6").glob("*/*.csv")):
        rows.extend(read_csv(p))
    raw_path = out_dir / "results" / "continuous_prm_c6_eval_raw.csv"
    summary_path = out_dir / "results" / "continuous_prm_c6_eval_summary.csv"
    write_csv(raw_path, rows)
    summary = aggregate_eval_records(rows)
    write_csv(summary_path, summary)
    write_json(out_dir / "results" / "continuous_prm_c6_eval_summary.json", {"rows": summary})
    analyze_significance(rows, out_dir)
    return summary_path


def collect_all(out_dir: Path, cfg: C6Config) -> Dict[str, Path]:
    install_c5_hard_runtime(cfg.sector_tokens)
    specs = C.build_anchor_specs()
    paths: Dict[str, Path] = {}
    for idx, task in enumerate(parse_csv(cfg.train_tasks)):
        paths[task] = collect_dataset(
            specs[task],
            ensure_dir(out_dir / "datasets"),
            "train",
            int(cfg.train_worlds),
            cfg,
            seed=int(cfg.seed) + 10_000 * (idx + 1),
        )
    return paths


def run_train(out_dir: Path, cfg: C6Config, dataset_paths: Dict[str, Path], device: torch.device) -> None:
    for model_name in parse_csv(cfg.models):
        if model_name == "oracle":
            continue
        if model_name not in MODEL_NAMES:
            raise ValueError(f"unknown C6 model {model_name!r}")
        train_model(model_name, list(dataset_paths.values()), out_dir, cfg, device)


def run_eval(out_dir: Path, cfg: C6Config) -> Path:
    suites = parse_csv(cfg.eval_suites)
    for suite_idx, suite in enumerate(suites):
        evaluate_shard(out_dir, cfg, suite, suite_idx, 0, int(cfg.eval_worlds), suite_idx)
    return merge_eval_shards(out_dir)


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    args.train_tasks = "C_hard_maze"
    args.eval_suites = "C_hard_maze"
    args.train_worlds = min(int(args.train_worlds), 8)
    args.eval_worlds = min(int(args.eval_worlds), 4)
    args.grid_size = min(int(args.grid_size), 48)
    args.roadmap_nodes = min(int(args.roadmap_nodes), 96)
    args.roadmap_k = min(int(args.roadmap_k), 6)
    args.epochs = min(int(args.epochs), 2)
    args.batch_size = min(int(args.batch_size), 4)
    args.budgets = args.budgets or "128,144"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C6 continuous PRM heatmap/value-field experiment")
    p.add_argument("--mode", choices=("collect", "train", "eval", "analyze", "full"), default="full")
    p.add_argument("--out-dir", type=str, default="runs/continuous_prm_c6_heatmap_value_field")
    p.add_argument("--models", type=str, default="oracle,unet,onlstm,hrm")
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--train-worlds", type=int, default=160)
    p.add_argument("--eval-worlds", type=int, default=80)
    p.add_argument("--roadmap-nodes", type=int, default=192)
    p.add_argument("--roadmap-k", type=int, default=7)
    p.add_argument("--budgets", type=str, default=DEFAULT_BUDGETS)
    p.add_argument("--train-tasks", type=str, default=DEFAULT_TRAIN_TASKS)
    p.add_argument("--eval-suites", type=str, default=DEFAULT_EVAL_SUITES)
    p.add_argument("--target-mode", choices=("residual",), default="residual")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--loss-rank-weight", type=float, default=0.2)
    p.add_argument("--loss-path-weight", type=float, default=0.05)
    p.add_argument("--loss-consistency-weight", type=float, default=0.02)
    p.add_argument("--rank-pairs", type=int, default=256)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-world-retries", type=int, default=120)
    p.add_argument("--sector-tokens", type=int, default=16)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--torch-threads", type=int, default=int(os.environ.get("TORCH_THREADS", "1")))
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_smoke_overrides(args)
    cfg = config_from_args(args)
    if int(cfg.torch_threads) > 0:
        torch.set_num_threads(int(cfg.torch_threads))
        try:
            torch.set_num_interop_threads(max(1, min(4, int(cfg.torch_threads))))
        except RuntimeError:
            pass
    set_global_seed(int(cfg.seed))
    out_dir = ensure_dir(args.out_dir)
    write_json(out_dir / "config__c6.json", {"args": vars(args), "cfg": asdict(cfg)})
    device = torch.device("cpu" if cfg.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[{now_str()}] C6 mode={args.mode} out_dir={out_dir} device={device} models={cfg.models}", flush=True)
    dataset_paths = {task: out_dir / "datasets" / f"{task}_train.npz" for task in parse_csv(cfg.train_tasks)}
    if args.mode in ("collect", "full"):
        dataset_paths = collect_all(out_dir, cfg)
    if args.mode in ("train", "full"):
        for task, path in dataset_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"missing C6 dataset for {task}: {path}")
        run_train(out_dir, cfg, dataset_paths, device)
    if args.mode in ("eval", "full"):
        run_eval(out_dir, cfg)
    if args.mode == "analyze":
        raw_path = out_dir / "results" / "continuous_prm_c6_eval_raw.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"missing raw eval CSV: {raw_path}")
        rows = read_csv(raw_path)
        summary = aggregate_eval_records(rows)
        write_csv(out_dir / "results" / "continuous_prm_c6_eval_summary.csv", summary)
        analyze_significance(rows, out_dir)


if __name__ == "__main__":
    main()
