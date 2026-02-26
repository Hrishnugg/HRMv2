"""
Transfer‑First A* Augmentation via Multilevel Adaptation (RL Heuristic Learning)

This experiment keeps Space‑Time A* central and trains recurrent models (HRM vs ON‑LSTM)
to provide a learned heuristic correction that improves search efficiency and transfers
to new map families and larger scales.

High-level flow:
1) Curriculum training on Family A:
   Stage 1: A, 32x32, D0 (static)
   Stage 2: A, 32x32, D1 (mild movers)
   Stage 3a: A, 64x64, D1 (scale transfer)
   Stage 3b: A, 64x64, D2 (gates + heavy dynamics)
2) Zero-shot transfer evaluation on held-out families B/C and scale
3) Few-shot adaptation on target domain B, 64x64, D2 with K={50,200,1000}

Usage:
  modal run hrm-cloud/transfer_astar_heuristic_rl.py
"""

from __future__ import annotations

import modal
import numpy as np
import torch

# Silence harmless NNPACK warnings on unsupported CPU hardware (common in container VMs).
# This does not affect correctness; it just prevents repeated log spam.
try:
    if hasattr(torch.backends, "nnpack") and hasattr(torch.backends.nnpack, "set_flags"):
        torch.backends.nnpack.set_flags(False)
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F
import heapq
import os
import json
import time
import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

# =============================================================================
# Modal setup
# =============================================================================

APP_NAME = "transfer-astar-heuristic-rl-v1"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch>=2.4.0", "numpy", "tqdm")
)

vol = modal.Volume.from_name("transfer-astar-heuristic-rl-vol", create_if_missing=True)

# =============================================================================
# Paths / constants
# =============================================================================

PATHS = {
    "root": "/data/transfer_astar_heuristic_rl_v1",
    "models": "/data/transfer_astar_heuristic_rl_v1/models",
    "checkpoints": "/data/transfer_astar_heuristic_rl_v1/checkpoints",
    "results": "/data/transfer_astar_heuristic_rl_v1/results",
}

# =============================================================================
# Config dataclasses
# =============================================================================

@dataclass
class ModelConfig:
    name: str
    model_type: str  # "onlstm" | "hrm"
    hidden_dim: int
    num_layers: int
    num_heads: int = 4         # HRM only
    chunk_size: int = 5        # ON-LSTM only
    patch_size: int = 15
    patch_embed_dim: int = 16
    obs_embed_dim: int = 128   # projection into recurrent core
    obstacle_embed_dim: int = 64
    max_obstacles: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    # compute placement
    gpu: str = "A10"
    # training batch sizes (episodes per update, SGD batch)
    episodes_per_update: int = 64
    minibatch_episodes: int = 16
    sgd_epochs: int = 2


@dataclass
class StageConfig:
    stage_id: str
    map_family: str  # "A"|"B"|"C"
    grid_size: int
    dynamics: str    # "D0"|"D1"|"D2"
    # dynamics counts
    n_gates: int
    n_patrollers: int
    n_drifters: int
    # episode config
    horizon: int
    max_steps: int
    # training loop
    updates: int
    gamma: float = 0.99
    expansions_lambda: float = 1e-4
    collision_penalty: float = 50.0
    timeout_penalty: float = 25.0
    alpha: float = 1.0  # heuristic scale in A* f = g + h_static + alpha*delta_h
    # exploration
    heuristic_noise_std: float = 0.05
    epsilon_action: float = 0.00  # optionally random action with prob epsilon_action
    # A* budget
    max_expansions: int = 8000


@dataclass
class EvalSuite:
    suite_id: str
    map_family: str
    grid_size: int
    dynamics: str
    n_gates: int
    n_patrollers: int
    n_drifters: int
    horizon: int
    max_steps: int
    episodes: int
    alpha: float = 1.0
    max_expansions: int = 8000


# =============================================================================
# Experiment configuration (edit these for sweeps)
# =============================================================================

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    # NOTE: This experiment is rollout/CPU-heavy (Space-Time A*). Using H100/B200 burns credits fast.
    # We keep the cost-saver budgets but run ON-LSTM on H100 and HRM on B200 for speed. You can change the decorator GPU or split functions later
    # if you want to selectively use H100/B200 for the largest tiers.

    # ON-LSTM tiers
    "onlstm_300k": ModelConfig("onlstm_300k", "onlstm", hidden_dim=155, num_layers=2, chunk_size=5,
                              lr=3e-4, gpu="H100", episodes_per_update=16, minibatch_episodes=8, sgd_epochs=4),
    "onlstm_1m": ModelConfig("onlstm_1m", "onlstm", hidden_dim=275, num_layers=2, chunk_size=5,
                             lr=3e-4, gpu="H100", episodes_per_update=16, minibatch_episodes=8, sgd_epochs=4),
    "onlstm_3m": ModelConfig("onlstm_3m", "onlstm", hidden_dim=475, num_layers=2, chunk_size=5,
                             lr=3e-4, gpu="H100", episodes_per_update=12, minibatch_episodes=6, sgd_epochs=4),
    "onlstm_10m": ModelConfig("onlstm_10m", "onlstm", hidden_dim=860, num_layers=3, chunk_size=5,
                              lr=2e-4, gpu="H100", episodes_per_update=8, minibatch_episodes=4, sgd_epochs=4),

    # HRM tiers
    "hrm_302k": ModelConfig("hrm_302k", "hrm", hidden_dim=128, num_layers=2, num_heads=4,
                            lr=3e-4, gpu="A10", episodes_per_update=16, minibatch_episodes=8, sgd_epochs=4),
    "hrm_3m": ModelConfig("hrm_3m", "hrm", hidden_dim=256, num_layers=2, num_heads=4,
                          lr=2e-4, gpu="A10", episodes_per_update=12, minibatch_episodes=6, sgd_epochs=4),
    "hrm_10m": ModelConfig("hrm_10m", "hrm", hidden_dim=384, num_layers=3, num_heads=6,
                           lr=2e-4, gpu="A10", episodes_per_update=8, minibatch_episodes=4, sgd_epochs=4),
}

# Curriculum stages (Family A only)
# Curriculum stages (Family A only)
# Cost controls:
# - Early stages are "warm start" and can be short once success is high.
# - Stage3a/3b are the expensive ones; keep updates modest and lean on few-shot for the hardest OOD.
STAGES: List[StageConfig] = [
    StageConfig(stage_id="stage1_A32_D0", map_family="A", grid_size=32, dynamics="D0",
                n_gates=0, n_patrollers=0, n_drifters=0,
                horizon=12, max_steps=80,
                updates=20,
                expansions_lambda=0.0,  # no need early
                alpha=1.0,
                heuristic_noise_std=0.08,
                max_expansions=4000),
    StageConfig(stage_id="stage2_A32_D1", map_family="A", grid_size=32, dynamics="D1",
                n_gates=0, n_patrollers=2, n_drifters=2,
                horizon=15, max_steps=90,
                updates=30,
                expansions_lambda=5e-5,
                alpha=1.0,
                heuristic_noise_std=0.06,
                max_expansions=5000),
    StageConfig(stage_id="stage3a_A64_D1", map_family="A", grid_size=64, dynamics="D1",
                n_gates=0, n_patrollers=4, n_drifters=4,
                horizon=20, max_steps=160,
                updates=30,
                expansions_lambda=8e-5,
                alpha=1.0,
                heuristic_noise_std=0.05,
                max_expansions=8000),
    StageConfig(stage_id="stage3b_A64_D2", map_family="A", grid_size=64, dynamics="D2",
                n_gates=2, n_patrollers=4, n_drifters=10,
                horizon=20, max_steps=180,
                updates=60,
                expansions_lambda=1e-4,
                alpha=1.0,
                heuristic_noise_std=0.04,
                max_expansions=10000),
]

# Evaluation suites (zero-shot transfer + ID)
# Evaluation suites (zero-shot transfer + ID)
# NOTE: evaluation is CPU-heavy (A*). Use 100 episodes per suite by default for cost control.
# You can increase to 200-500 for final reporting once budgets are validated.
EVAL_SUITES: List[EvalSuite] = [
    # In-distribution Family A
    EvalSuite("ID_A32_D1", "A", 32, "D1", n_gates=0, n_patrollers=2, n_drifters=2, horizon=20, max_steps=90, episodes=100),
    EvalSuite("ID_A64_D2", "A", 64, "D2", n_gates=2, n_patrollers=4, n_drifters=10, horizon=20, max_steps=180, episodes=100),

    # OOD topology B/C small
    EvalSuite("OOD_B32_D1", "B", 32, "D1", n_gates=0, n_patrollers=2, n_drifters=2, horizon=20, max_steps=90, episodes=100),
    EvalSuite("OOD_C32_D1", "C", 32, "D1", n_gates=0, n_patrollers=2, n_drifters=2, horizon=20, max_steps=90, episodes=100),

    # OOD topology + scale
    EvalSuite("OOD_B64_D2", "B", 64, "D2", n_gates=2, n_patrollers=4, n_drifters=10, horizon=20, max_steps=180, episodes=120),
    EvalSuite("OOD_C64_D2", "C", 64, "D2", n_gates=2, n_patrollers=4, n_drifters=10, horizon=20, max_steps=180, episodes=120),
]

# Few-shot target (Family B Large D2)
FEWSHOT_TARGET = EvalSuite("TARGET_B64_D2", "B", 64, "D2", n_gates=2, n_patrollers=4, n_drifters=10, horizon=20, max_steps=180, episodes=150)
# Default few-shot K values (cost-controlled). Increase / add 1000 once everything is stable.
FEWSHOT_K = [50, 200]
# =============================================================================
# Utilities
# =============================================================================

def ensure_dirs():
    os.makedirs(PATHS["root"], exist_ok=True)
    os.makedirs(PATHS["models"], exist_ok=True)
    os.makedirs(PATHS["checkpoints"], exist_ok=True)
    os.makedirs(PATHS["results"], exist_ok=True)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_determinism(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =============================================================================
# Map generation utilities
# =============================================================================

def _neighbors4(r: int, c: int):
    return ((r, c + 1), (r, c - 1), (r + 1, c), (r - 1, c))

def _bfs_path(static_map: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    H, W = static_map.shape
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
        return None
    if static_map[sr, sc] == 1 or static_map[gr, gc] == 1:
        return None
    from collections import deque
    q = deque()
    q.append((sr, sc))
    parent = {(sr, sc): None}
    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            path = []
            cur = (r, c)
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and static_map[nr, nc] == 0 and (nr, nc) not in parent:
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    return None

def _carve_L(static_map: np.ndarray, a: Tuple[int,int], b: Tuple[int,int], rng: np.random.Generator):
    r1, c1 = a
    r2, c2 = b
    if rng.random() < 0.5:
        for c in range(min(c1, c2), max(c1, c2)+1):
            static_map[r1, c] = 0
        for r in range(min(r1, r2), max(r1, r2)+1):
            static_map[r, c2] = 0
    else:
        for r in range(min(r1, r2), max(r1, r2)+1):
            static_map[r, c1] = 0
        for c in range(min(c1, c2), max(c1, c2)+1):
            static_map[r2, c] = 0

def compute_bfs_dist(static_map: np.ndarray, goal: Tuple[int,int]) -> np.ndarray:
    """Return dist grid; unreachable cells are +inf."""
    H, W = static_map.shape
    gr, gc = goal
    dist = np.full((H, W), np.inf, dtype=np.float32)
    if static_map[gr, gc] == 1:
        return dist
    from collections import deque
    q = deque()
    q.append((gr, gc))
    dist[gr, gc] = 0.0
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and static_map[nr, nc] == 0:
                if dist[nr, nc] == np.inf:
                    dist[nr, nc] = dist[r, c] + 1.0
                    q.append((nr, nc))
    return dist

def generate_rooms_corridors(size: int, rng: np.random.Generator,
                             n_rooms: int = 8, room_min: int = 4, room_max: int = 8, room_padding: int = 1) -> Tuple[np.ndarray, List[Tuple[int,int,int,int]]]:
    """Family A: rooms + corridors."""
    S = size
    static_map = np.ones((S, S), dtype=np.int8)
    rooms: List[Tuple[int,int,int,int]] = []
    attempts = 0
    max_attempts = n_rooms * 40

    def overlaps(x, y, w, h):
        for rx, ry, rw, rh in rooms:
            if (x < rx + rh + room_padding and x + h + room_padding > rx and
                y < ry + rw + room_padding and y + w + room_padding > ry):
                return True
        return False

    while len(rooms) < n_rooms and attempts < max_attempts:
        attempts += 1
        w = int(rng.integers(room_min, room_max + 1))
        h = int(rng.integers(room_min, room_max + 1))
        x = int(rng.integers(1, max(2, S - h - 1)))
        y = int(rng.integers(1, max(2, S - w - 1)))
        if overlaps(x, y, w, h):
            continue
        static_map[x:x+h, y:y+w] = 0
        rooms.append((x, y, w, h))

    if not rooms:
        static_map[:, :] = 0
        rooms = [(1, 1, S-2, S-2)]

    centers = [(x + h//2, y + w//2) for x,y,w,h in rooms]
    for i in range(1, len(centers)):
        _carve_L(static_map, centers[i-1], centers[i], rng)

    start = (0,0)
    goal = (S-1,S-1)

    def nearest_center(p):
        pr, pc = p
        best = centers[0]
        best_d = abs(best[0]-pr)+abs(best[1]-pc)
        for cc in centers[1:]:
            d = abs(cc[0]-pr)+abs(cc[1]-pc)
            if d < best_d:
                best_d = d
                best = cc
        return best

    _carve_L(static_map, start, nearest_center(start), rng)
    _carve_L(static_map, goal, nearest_center(goal), rng)

    static_map[0,0]=0
    static_map[S-1,S-1]=0
    return static_map, rooms

def generate_open_clutter(size: int, rng: np.random.Generator,
                          clutter_frac: float = 0.08) -> Tuple[np.ndarray, List[Tuple[int,int,int,int]]]:
    """Family C: mostly open with sparse clutter rectangles."""
    S = size
    static_map = np.zeros((S, S), dtype=np.int8)
    rooms: List[Tuple[int,int,int,int]] = []  # unused, for compatibility

    # add random rectangles
    n_rect = max(8, int(S * clutter_frac))
    for _ in range(n_rect):
        h = int(rng.integers(2, 6))
        w = int(rng.integers(2, 6))
        r = int(rng.integers(0, S-h))
        c = int(rng.integers(0, S-w))
        static_map[r:r+h, c:c+w] = 1

    static_map[0,0]=0
    static_map[S-1,S-1]=0

    # ensure connectivity; if not, carve L
    if _bfs_path(static_map, (0,0), (S-1,S-1)) is None:
        _carve_L(static_map, (0,0), (S-1,S-1), rng)
    return static_map, rooms

def generate_dfs_maze(size: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[Tuple[int,int,int,int]]]:
    """Family B: dense labyrinth (DFS perfect maze)."""
    S = size
    # initialize all walls
    grid = np.ones((S, S), dtype=np.int8)
    rooms: List[Tuple[int,int,int,int]] = []

    # we will carve passages on a reduced lattice
    # choose odd coordinates as cells if possible
    start = (0,0)
    grid[start]=0

    # Define carve with 2-step to keep walls; works best if S is odd.
    # For even S, we still carve but may leave last row/col dense; that's ok.
    stack = [start]
    visited = set([start])

    def neighbors(cell):
        r,c = cell
        cands=[]
        for dr,dc in [(2,0),(-2,0),(0,2),(0,-2)]:
            nr,nc = r+dr, c+dc
            if 0 <= nr < S and 0 <= nc < S:
                if (nr,nc) not in visited:
                    cands.append((nr,nc,(dr//2,dc//2)))
        return cands

    while stack:
        cell = stack[-1]
        cands = neighbors(cell)
        if not cands:
            stack.pop()
            continue
        nr,nc,(wdr,wdc) = cands[int(rng.integers(0,len(cands)))]
        # carve wall between
        wr,wc = cell[0]+wdr, cell[1]+wdc
        grid[wr,wc]=0
        grid[nr,nc]=0
        visited.add((nr,nc))
        stack.append((nr,nc))

    grid[0,0]=0
    grid[S-1,S-1]=0
    # ensure connectivity
    if _bfs_path(grid, (0,0), (S-1,S-1)) is None:
        _carve_L(grid, (0,0), (S-1,S-1), rng)
    return grid, rooms

def generate_static_map(family: str, size: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[Tuple[int,int,int,int]]]:
    if family == "A":
        # scale room sizes with map
        if size <= 32:
            return generate_rooms_corridors(size, rng, n_rooms=8, room_min=4, room_max=8, room_padding=1)
        else:
            return generate_rooms_corridors(size, rng, n_rooms=14, room_min=5, room_max=11, room_padding=1)
    if family == "B":
        return generate_dfs_maze(size, rng)
    if family == "C":
        return generate_open_clutter(size, rng, clutter_frac=0.06 if size<=32 else 0.05)
    raise ValueError(f"Unknown map family: {family}")

def extract_local_patch(static_map: np.ndarray, center_rc: Tuple[int,int], patch_size: int) -> np.ndarray:
    """Return (P,P) uint8 patch, 1=wall,0=free, centered at (r,c). Out-of-bounds treated as wall."""
    r, c = int(center_rc[0]), int(center_rc[1])
    rad = patch_size // 2
    P = patch_size
    patch = np.ones((P, P), dtype=np.uint8)
    H, W = static_map.shape
    for i in range(-rad, rad + 1):
        for j in range(-rad, rad + 1):
            rr = r + i
            cc = c + j
            if 0 <= rr < H and 0 <= cc < W:
                patch[i + rad, j + rad] = static_map[rr, cc].astype(np.uint8)
    return patch

# =============================================================================
# Environment: TransferDynamicMaze (supports families A/B/C and dynamics D0/D1/D2)
# =============================================================================

class TransferDynamicMazeEnv:
    """Grid environment with static walls, moving obstacles, and optional gates.

    Agent moves in 4-neighborhood + WAIT.
    Obstacles move deterministically given RNG for the episode (drifters stochastic but seeded).
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg)
        self.size = int(cfg["grid_size"])
        self.patch_size = int(cfg.get("patch_size", 15))
        # Safety default: some call-sites (especially evaluation/baselines) may
        # forget to include max_steps in cfg. Use the same scale as our suites
        # (≈2.8×grid_size) as a reasonable default.
        if "max_steps" not in self.cfg:
            self.cfg["max_steps"] = int(round(2.8 * self.size))
        self.rng: Optional[np.random.Generator] = None
        self.map_rng: Optional[np.random.Generator] = None
        self.dynamic_obs: List[Dict[str, Any]] = []
        self.step_count = 0

    def reset(self, seed: Optional[int]=None, map_seed: Optional[int]=None, dyn_seed: Optional[int]=None):
        if map_seed is None:
            map_seed = seed
        if dyn_seed is None:
            dyn_seed = seed

        self.map_rng = np.random.default_rng(map_seed)
        self.rng = np.random.default_rng(dyn_seed)

        # static map generation
        self.static_map, _rooms = generate_static_map(self.cfg["map_family"], self.size, self.map_rng)

        self.agent_pos = (0, 0)
        self.goal_pos = (self.size - 1, self.size - 1)
        self.static_map[0,0]=0
        self.static_map[self.size-1,self.size-1]=0

        # ensure connectivity
        path = _bfs_path(self.static_map, self.agent_pos, self.goal_pos)
        if path is None:
            _carve_L(self.static_map, self.agent_pos, self.goal_pos, self.map_rng)
            path = _bfs_path(self.static_map, self.agent_pos, self.goal_pos)
        self.main_path = path or [self.agent_pos, self.goal_pos]

        # precompute static distance
        self.dist_to_goal = compute_bfs_dist(self.static_map, self.goal_pos)

        # init dynamic objects
        self.dynamic_obs = []
        if self.cfg["dynamics"] in ("D1", "D2"):
            if int(self.cfg.get("n_gates", 0)) > 0:
                self._init_gates()
            self._init_patrollers(int(self.cfg.get("n_patrollers", 0)))
            self._init_drifters(int(self.cfg.get("n_drifters", 0)))
        self.step_count = 0
        return self._get_obs()

    def clone(self) -> "TransferDynamicMazeEnv":
        new = TransferDynamicMazeEnv.__new__(TransferDynamicMazeEnv)
        new.cfg = dict(self.cfg)
        new.size = self.size
        new.patch_size = self.patch_size
        # clone RNG state
        new.rng = np.random.default_rng()
        new.rng.bit_generator.state = self.rng.bit_generator.state  # type: ignore
        new.map_rng = None  # not needed for physics
        new.static_map = self.static_map.copy()
        new.agent_pos = tuple(self.agent_pos)
        new.goal_pos = tuple(self.goal_pos)
        new.main_path = list(self.main_path)
        new.dist_to_goal = self.dist_to_goal.copy()
        new.step_count = self.step_count
        new.dynamic_obs = []
        for o in self.dynamic_obs:
            typ = o["type"]
            if typ == "gate":
                new_o = {
                    "type": "gate",
                    "closed": o["closed"],
                    "open": o["open"],
                    "is_closed": o["is_closed"],
                    "timer": o["timer"],
                    "closed_len": o["closed_len"],
                    "open_len": o["open_len"],
                    "pos": np.array(o["pos"], dtype=np.float32),
                }
            elif typ == "patroller":
                new_o = {
                    "type": "patroller",
                    "route": list(o["route"]),
                    "idx": o["idx"],
                    "dwell": o["dwell"],
                    "waypoints": set(o["waypoints"]),
                    "pos": np.array(o["pos"], dtype=np.float32),
                }
            elif typ == "drifter":
                new_o = {
                    "type": "drifter",
                    "cell": o["cell"],
                    "heading": o["heading"],
                    "mode": o["mode"],
                    "mode_steps": o["mode_steps"],
                    "pos": np.array(o["pos"], dtype=np.float32),
                }
            else:
                continue
            new.dynamic_obs.append(new_o)
        return new

    # ---------- gates ----------
    def _init_gates(self):
        S = self.size
        n_gates = int(self.cfg.get("n_gates", 0))
        if n_gates <= 0:
            return
        path = self.main_path
        if len(path) < 10:
            gate_cells = [path[len(path)//2]]
        else:
            i1 = max(2, int(len(path)*0.33))
            i2 = min(len(path)-3, int(len(path)*0.66))
            gate_cells = [path[i1], path[i2]] if i1 != i2 else [path[i1]]
        while len(gate_cells) < n_gates and len(path) > 4:
            idx = int(self.map_rng.integers(2, len(path)-2))  # type: ignore
            cell = path[idx]
            if cell not in gate_cells:
                gate_cells.append(cell)
        gate_cells = gate_cells[:n_gates]

        for (gr, gc) in gate_cells:
            # carve alcove "open" position if possible
            candidates = []
            for nr,nc in _neighbors4(gr,gc):
                if 0 <= nr < S and 0 <= nc < S and self.static_map[nr,nc] == 1:
                    candidates.append((nr,nc))
            if candidates:
                open_pos = candidates[int(self.map_rng.integers(0,len(candidates)))]  # type: ignore
                self.static_map[open_pos[0], open_pos[1]] = 0
            else:
                # fallback to adjacent free cell (less interesting)
                neigh_free = []
                for nr,nc in _neighbors4(gr,gc):
                    if 0 <= nr < S and 0 <= nc < S and self.static_map[nr,nc] == 0 and (nr,nc) != (gr,gc):
                        neigh_free.append((nr,nc))
                open_pos = neigh_free[int(self.map_rng.integers(0,len(neigh_free)))] if neigh_free else (gr,gc)  # type: ignore

            closed_pos = (gr,gc)
            # gate schedule
            closed_len = int(self.rng.integers(3, 7))
            open_len = int(self.rng.integers(2, 6))
            is_closed = True
            timer = closed_len

            self.dynamic_obs.append({
                "type": "gate",
                "closed": closed_pos,
                "open": open_pos,
                "is_closed": is_closed,
                "timer": timer,
                "closed_len": closed_len,
                "open_len": open_len,
                "pos": np.array([float(closed_pos[0]), float(closed_pos[1])], dtype=np.float32)
            })

    # ---------- patrollers ----------
    def _init_patrollers(self, n_patrollers: int):
        if n_patrollers <= 0:
            return
        S = self.size
        free_cells = np.argwhere(self.static_map == 0)
        if len(free_cells) < 10:
            return

        for _ in range(n_patrollers):
            # pick two waypoints and BFS path between them; route = there and back
            a = tuple(free_cells[int(self.rng.integers(0, len(free_cells)))])
            b = tuple(free_cells[int(self.rng.integers(0, len(free_cells)))])
            path_ab = _bfs_path(self.static_map, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])))
            if path_ab is None or len(path_ab) < 4:
                continue
            route = path_ab + list(reversed(path_ab[1:-1]))
            idx = int(self.rng.integers(0, len(route)))
            r,c = route[idx]
            self.dynamic_obs.append({
                "type": "patroller",
                "route": route,
                "idx": idx,
                "dwell": 0,
                "waypoints": {route[0], route[len(path_ab)-1]},
                "pos": np.array([float(r), float(c)], dtype=np.float32)
            })

    # ---------- drifters ----------
    @staticmethod
    def _turn_left(h):
        dr, dc = h
        return (-dc, dr)

    @staticmethod
    def _turn_right(h):
        dr, dc = h
        return (dc, -dr)

    @staticmethod
    def _reverse(h):
        dr, dc = h
        return (-dr, -dc)

    def _init_drifters(self, n_drifters: int):
        if n_drifters <= 0:
            return
        S = self.size
        free_cells = np.argwhere(self.static_map == 0)
        if len(free_cells) < 10:
            return
        headings = [(0,1),(0,-1),(1,0),(-1,0)]
        regimes = ["left","right","random"]
        for _ in range(n_drifters):
            r,c = free_cells[int(self.rng.integers(0,len(free_cells)))]
            heading = headings[int(self.rng.integers(0,4))]
            mode = regimes[int(self.rng.integers(0,3))]
            mode_steps = int(self.rng.integers(4, 10))
            self.dynamic_obs.append({
                "type": "drifter",
                "cell": (int(r),int(c)),
                "heading": heading,
                "mode": mode,
                "mode_steps": mode_steps,
                "pos": np.array([float(r), float(c)], dtype=np.float32),
            })

    # ---------- physics ----------
    def step_physics(self):
        """Advance obstacles by one step."""
        S = self.size
        for o in self.dynamic_obs:
            typ = o["type"]
            if typ == "gate":
                o["timer"] -= 1
                if o["timer"] <= 0:
                    if o["is_closed"]:
                        o["is_closed"] = False
                        o["timer"] = o["open_len"]
                        r,c = o["open"]
                        o["pos"][:] = (float(r), float(c))
                    else:
                        o["is_closed"] = True
                        jitter = int(self.rng.integers(-1, 2))
                        o["timer"] = max(1, o["closed_len"] + jitter)
                        r,c = o["closed"]
                        o["pos"][:] = (float(r), float(c))
            elif typ == "patroller":
                if o["dwell"] > 0:
                    o["dwell"] -= 1
                else:
                    o["idx"] = (o["idx"] + 1) % len(o["route"])
                    r,c = o["route"][o["idx"]]
                    o["pos"][:] = (float(r), float(c))
                    if (r,c) in o["waypoints"] and self.rng.random() < 0.35:
                        o["dwell"] = int(self.rng.integers(1, 3))
            elif typ == "drifter":
                r,c = o["cell"]
                heading = o["heading"]
                mode = o["mode"]
                o["mode_steps"] -= 1
                if o["mode_steps"] <= 0:
                    o["mode"] = ["left","right","random"][int(self.rng.integers(0,3))]
                    o["mode_steps"] = int(self.rng.integers(4, 10))
                    mode = o["mode"]

                forward = heading
                left = self._turn_left(heading)
                right = self._turn_right(heading)
                rev = self._reverse(heading)

                def can_move(h):
                    dr,dc = h
                    nr,nc = r+dr, c+dc
                    return 0 <= nr < S and 0 <= nc < S and self.static_map[nr,nc] == 0

                chosen = None
                if can_move(forward) and self.rng.random() < 0.75:
                    chosen = forward
                else:
                    if mode == "left":
                        for cand in (left, forward, right, rev):
                            if can_move(cand):
                                chosen = cand
                                break
                    elif mode == "right":
                        for cand in (right, forward, left, rev):
                            if can_move(cand):
                                chosen = cand
                                break
                    else:
                        cands = [d for d in (forward,left,right,rev) if can_move(d)]
                        if cands:
                            chosen = cands[int(self.rng.integers(0,len(cands)))]

                if chosen is None:
                    chosen = (0,0)

                dr,dc = chosen
                nr,nc = r+dr, c+dc
                if 0 <= nr < S and 0 <= nc < S and self.static_map[nr,nc] == 0:
                    r,c = nr,nc
                o["cell"] = (int(r),int(c))
                o["heading"] = chosen if chosen != (0,0) else heading
                o["pos"][:] = (float(r), float(c))
        return self._get_obs()

    # ---------- agent step ----------
    def step(self, action: int):
        """Perform agent action then advance physics. Return (obs, reward, done, info)."""
        # action: 0=up,1=down,2=left,3=right,4=wait
        ar, ac = self.agent_pos
        drdc = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1),4:(0,0)}
        dr, dc = drdc.get(int(action), (0,0))
        nr, nc = ar + dr, ac + dc
        # bounds and wall check
        if not (0 <= nr < self.size and 0 <= nc < self.size) or self.static_map[nr,nc] == 1:
            nr, nc = ar, ac  # bump -> stay
        self.agent_pos = (int(nr), int(nc))

        # physics update
        self.step_physics()
        self.step_count += 1

        # check terminal
        collision = self._check_collision()
        success = (self.agent_pos == self.goal_pos)
        timeout = (self.step_count >= int(self.cfg["max_steps"]))

        done = collision or success or timeout
        info = {
            "collision": collision,
            "success": success,
            "timeout": timeout,
        }
        return self._get_obs(), done, info

    def _check_collision(self) -> bool:
        ar, ac = self.agent_pos
        # static collision handled in movement; dynamic collision
        for o in self.dynamic_obs:
            r, c = int(round(float(o["pos"][0]))), int(round(float(o["pos"][1])))
            if (r,c) == (ar,ac):
                return True
        return False

    def _get_obs(self):
        """Return raw obs for debugging; training uses feature extractor."""
        return {
            "agent": self.agent_pos,
            "goal": self.goal_pos,
            "obstacles": [(o["type"], float(o["pos"][0]), float(o["pos"][1]), int(o.get("is_closed", 0))) for o in self.dynamic_obs],
            "t": self.step_count,
        }


# =============================================================================
# Artifact helpers
# =============================================================================

@app.function(
    image=image,
    volumes={"/data": vol},
    cpu=2,
    timeout=60 * 5,
)
def available_base_models() -> Dict[str, Any]:
    """Return which models have a stage3b final checkpoint saved."""
    vol.reload()
    ensure_dirs()
    tag = "stage3b_A64_D2_final"
    avail = []
    missing = []
    for model_name in MODEL_CONFIGS.keys():
        p = os.path.join(PATHS["models"], f"{model_name}_{tag}.pt")
        if os.path.exists(p):
            avail.append(model_name)
        else:
            missing.append(model_name)
    return {"available": avail, "missing": missing}


# =============================================================================
# Models: ON-LSTM and HRM recurrent cores + heuristic head
# =============================================================================

class ONLSTMCell(nn.Module):
    """Chunked Ordered Neurons LSTM cell."""
    def __init__(self, input_dim: int, hidden_dim: int, chunk_size: int = 5):
        super().__init__()
        assert hidden_dim % chunk_size == 0, "hidden_dim must be divisible by chunk_size"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.n_chunks = hidden_dim // chunk_size
        out_dim = 4*hidden_dim + 2*self.n_chunks
        self.lin = nn.Linear(input_dim + hidden_dim, out_dim)

    @staticmethod
    def cumax(x: torch.Tensor, dim: int = -1):
        return torch.cumsum(F.softmax(x, dim=dim), dim=dim)

    def forward(self, x: torch.Tensor, state):
        h_prev, c_prev = state
        gates = self.lin(torch.cat([x, h_prev], dim=-1))
        H = self.hidden_dim
        i, f, o, g = gates[:, :4*H].chunk(4, dim=-1)
        f_hat_lin, i_hat_lin = gates[:, 4*H:].chunk(2, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        f_hat = self.cumax(f_hat_lin)       # (B, n_chunks)
        i_hat = 1.0 - self.cumax(i_hat_lin) # (B, n_chunks)
        f_hat = f_hat.repeat_interleave(self.chunk_size, dim=-1)
        i_hat = i_hat.repeat_interleave(self.chunk_size, dim=-1)
        omega = f_hat * i_hat
        f = f * omega + (f_hat - omega)
        i = i * omega + (i_hat - omega)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class ONLSTMCore(nn.Module):
    """Stateful ON-LSTM encoder returning hidden state each step."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, chunk_size: int = 5, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.cells.append(ONLSTMCell(in_dim, hidden_dim, chunk_size=chunk_size))

    def init_state(self, batch_size: int, device=None, dtype=None):
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        hs = [torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype) for _ in range(self.num_layers)]
        cs = [torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype) for _ in range(self.num_layers)]
        return (hs, cs)

    def step(self, x_t: torch.Tensor, state, t: int):
        hs, cs = state
        inp = x_t
        new_hs, new_cs = [], []
        for li, cell in enumerate(self.cells):
            h, c = cell(inp, (hs[li], cs[li]))
            new_hs.append(h)
            new_cs.append(c)
            inp = h
            if self.dropout > 0 and li < self.num_layers-1:
                inp = F.dropout(inp, p=self.dropout, training=self.training)
        return inp, (new_hs, new_cs)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # upcast for stability
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(norm + self.eps)
        return (x_norm * self.g).to(x.dtype)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim*2)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x2)

class GatedRecurrentBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, int(dim*2.6))
        self.gate = nn.Linear(dim*2, dim)

    def forward(self, x, state):
        h = (x + state) * 0.7071
        res = h
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm.unsqueeze(1), h_norm.unsqueeze(1), h_norm.unsqueeze(1))
        h = res + attn_out.squeeze(1)
        candidate = h + self.ffn(self.norm2(h))
        z = torch.sigmoid(self.gate(torch.cat([candidate, state], dim=-1)))
        return z*candidate + (1-z)*state

class HRMCore(nn.Module):
    """Simplified HRM with low/high recurrent stacks updated at k_step cadence."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int, k_step: int = 2):
        super().__init__()
        self.k_step = k_step
        self.hidden_dim = hidden_dim
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.L_blocks = nn.ModuleList([GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.H_blocks = nn.ModuleList([GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)])

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_state(self, batch_size: int, device=None, dtype=None):
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        h_L = [torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype) for _ in range(len(self.L_blocks))]
        h_H = [torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype) for _ in range(len(self.H_blocks))]
        return (h_L, h_H)

    def step(self, x_t: torch.Tensor, state, t: int):
        h_L, h_H = state
        curr_in = self.embed(x_t)
        if t % self.k_step == 0:
            h_in = h_L[-1].detach()
            for i, blk in enumerate(self.H_blocks):
                h_H[i] = blk(h_in, h_H[i])
                h_in = h_H[i]
        l_in = curr_in + h_H[-1]
        for i, blk in enumerate(self.L_blocks):
            h_L[i] = blk(l_in, h_L[i])
            l_in = h_L[i]
        return h_L[-1], (h_L, h_H)

class PatchEncoder(nn.Module):
    """Lightweight CNN encoding (P,P) occupancy patch into embedding."""
    def __init__(self, patch_size: int, embed_dim: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.proj = nn.Linear(16, embed_dim)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        # patch: (B,1,P,P)
        h = F.relu(self.conv1(patch))
        h = F.relu(self.conv2(h))
        h = h.mean(dim=(-1,-2))
        return self.proj(h)

class ObstacleEncoder(nn.Module):
    def __init__(self, feat_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(feat_dim, embed_dim)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # feats: (B, M, D), mask: (B, M) in {0,1}
        x = torch.tanh(self.proj(feats))
        x = x * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return x.sum(dim=1) / denom

class HeuristicModel(nn.Module):
    """Unified heuristic model: recurrent encoder produces context; head predicts delta-h for a node.

    Training supervises delta-h at the agent's actual node with delta_t=0.
    Planning queries delta-h for many nodes using the same context.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_encoder = PatchEncoder(cfg.patch_size, cfg.patch_embed_dim)

        # per obstacle features: [dx, dy, onehot(type=3), gate_closed]
        self.obs_feat_dim = 2 + 3 + 1
        self.obstacle_encoder = ObstacleEncoder(self.obs_feat_dim, cfg.obstacle_embed_dim)

        # embed into recurrent input
        # observation vector: patch_emb + goal(2) + obs_emb + time(1) + static_dist(1)
        obs_in_dim = cfg.patch_embed_dim + 2 + cfg.obstacle_embed_dim + 1 + 1
        self.obs_proj = nn.Linear(obs_in_dim, cfg.obs_embed_dim)

        if cfg.model_type == "onlstm":
            self.core = ONLSTMCore(cfg.obs_embed_dim, cfg.hidden_dim, cfg.num_layers, chunk_size=cfg.chunk_size, dropout=0.0)
            core_out_dim = cfg.hidden_dim
        elif cfg.model_type == "hrm":
            self.core = HRMCore(cfg.obs_embed_dim, cfg.hidden_dim, cfg.num_layers, num_heads=cfg.num_heads, k_step=2)
            core_out_dim = cfg.hidden_dim
        else:
            raise ValueError(f"Unknown model_type: {cfg.model_type}")

        # node features: node_patch_emb + node_goal(2) + delta_t(1) + h_static(1)
        node_feat_dim = cfg.patch_embed_dim + 2 + 1 + 1
        self.head = nn.Sequential(
            nn.Linear(core_out_dim + node_feat_dim, core_out_dim),
            nn.ReLU(),
            nn.Linear(core_out_dim, 1),
        )

    def init_state(self, batch_size: int, device=None, dtype=None):
        return self.core.init_state(batch_size, device=device, dtype=dtype)

    def encode_obstacles(self, env: TransferDynamicMazeEnv, agent_rc: Tuple[int,int]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (feats[M,D], mask[M]) numpy arrays with fixed M=max_obstacles."""
        M = self.cfg.max_obstacles
        feats = np.zeros((M, self.obs_feat_dim), dtype=np.float32)
        mask = np.zeros((M,), dtype=np.float32)

        ar, ac = agent_rc
        obs_list = env.dynamic_obs
        # stable order: as stored in env.dynamic_obs
        for i, o in enumerate(obs_list[:M]):
            typ = o["type"]
            rr = float(o["pos"][0]); cc = float(o["pos"][1])
            dx = (rr - ar) / env.size
            dy = (cc - ac) / env.size
            onehot = [0.0,0.0,0.0]  # gate, patroller, drifter
            if typ == "gate":
                onehot[0] = 1.0
                gate_closed = float(1.0 if o.get("is_closed", False) else 0.0)
            elif typ == "patroller":
                onehot[1] = 1.0
                gate_closed = 0.0
            else:
                onehot[2] = 1.0
                gate_closed = 0.0
            feats[i] = np.array([dx, dy, onehot[0], onehot[1], onehot[2], gate_closed], dtype=np.float32)
            mask[i] = 1.0
        return feats, mask

    @torch.no_grad()
    def precompute_patch_emb_grid(self, static_map: np.ndarray, device: torch.device) -> torch.Tensor:
        """Compute patch embedding for every cell. Returns tensor (S,S,E) on device."""
        S = static_map.shape[0]
        P = self.cfg.patch_size
        patches = []
        for r in range(S):
            for c in range(S):
                patches.append(extract_local_patch(static_map, (r,c), P))
        patches = np.stack(patches, axis=0).astype(np.float32)  # (S*S,P,P)
        # to tensor
        x = torch.from_numpy(patches).unsqueeze(1).to(device)  # (N,1,P,P)
        # batch to avoid huge memory
        out = []
        bs = 512 if S*S > 1024 else 256
        for i in range(0, x.size(0), bs):
            out.append(self.patch_encoder(x[i:i+bs]))
        emb = torch.cat(out, dim=0)  # (N,E)
        emb = emb.view(S, S, -1)
        return emb

    def step_context(self, obs_patch: torch.Tensor, goal_vec: torch.Tensor,
                     obs_feats: torch.Tensor, obs_mask: torch.Tensor,
                     time_feat: torch.Tensor, static_dist: torch.Tensor,
                     state, t: int):
        """Update recurrent context with one observation step."""
        # obs_patch: (B,1,P,P)
        patch_emb = self.patch_encoder(obs_patch)
        obs_emb = self.obstacle_encoder(obs_feats, obs_mask)
        x = torch.cat([patch_emb, goal_vec, obs_emb, time_feat, static_dist], dim=-1)
        x = self.obs_proj(x)
        h, new_state = self.core.step(x, state, t)
        return h, new_state

    def predict_delta_h(self, context_h: torch.Tensor, node_patch_emb: torch.Tensor,
                        node_goal_vec: torch.Tensor, delta_t: torch.Tensor,
                        h_static: torch.Tensor) -> torch.Tensor:
        """Predict delta-h for nodes. All tensors are (B, dim). Returns (B,1)."""
        node_feat = torch.cat([node_patch_emb, node_goal_vec, delta_t, h_static], dim=-1)
        x = torch.cat([context_h, node_feat], dim=-1)
        return torch.tanh(self.head(x))

# =============================================================================
# Space-Time A* planner (uses env forward model for dynamic occupancy, learned heuristic for guidance)
# =============================================================================

class SpaceTimeAStarPlanner:
    def __init__(self, env: TransferDynamicMazeEnv, model: Optional[HeuristicModel],
                 device: torch.device, stage: StageConfig, patch_emb_grid: Optional[torch.Tensor],
                 context_h: Optional[torch.Tensor], train_mode: bool = False):
        self.env = env
        self.model = model
        self.device = device
        self.stage = stage
        self.patch_emb_grid = patch_emb_grid  # (S,S,E) on device
        self.context_h = context_h  # (1,H) on device
        self.train_mode = train_mode

        # precompute future obstacle cells for horizon
        self.future_occ = self._simulate_future_occupancy(stage.horizon)

        # cache per-cell goal vectors and static distances to avoid per-node tensor allocations
        S = self.env.size
        gr, gc = self.env.goal_pos

        # raw static dist grid (steps); replace inf with Manhattan
        dist_raw = self.env.dist_to_goal.copy()
        for rr in range(S):
            for cc in range(S):
                if math.isinf(float(dist_raw[rr, cc])):
                    dist_raw[rr, cc] = abs(rr - gr) + abs(cc - gc)
        self.h_static_raw_grid = dist_raw  # numpy (S,S)

        # normalized for model input
        self.h_static_norm_grid = torch.from_numpy(dist_raw.astype(np.float32) / (2.0 * S)).unsqueeze(-1).to(self.device)

        gv = np.zeros((S, S, 2), dtype=np.float32)
        for rr in range(S):
            for cc in range(S):
                gv[rr, cc, 0] = (gr - rr) / S
                gv[rr, cc, 1] = (gc - cc) / S
        self.goal_vec_grid = torch.from_numpy(gv).to(self.device)

        # delta_t table
        dt = np.arange(0, int(self.stage.horizon) + 1, dtype=np.float32) / max(1, int(self.stage.horizon))
        self.delta_t_table = torch.from_numpy(dt).unsqueeze(-1).to(self.device)

    def _simulate_future_occupancy(self, horizon: int):
        """Return list length horizon+1, each is a set of occupied (r,c) by dynamic objects."""
        occ = []
        clone = self.env.clone()
        # t=0
        occ.append(self._occupied_cells_from_env(clone))
        for _ in range(horizon):
            clone.step_physics()
            occ.append(self._occupied_cells_from_env(clone))
        return occ

    @staticmethod
    def _occupied_cells_from_env(env: TransferDynamicMazeEnv):
        s = set()
        for o in env.dynamic_obs:
            r = int(round(float(o["pos"][0])))
            c = int(round(float(o["pos"][1])))
            s.add((r,c))
        return s

    def heuristic(self, r: int, c: int, t: int) -> float:
        # static distance baseline (steps)
        h_static = float(self.h_static_raw_grid[r, c])

        if self.model is None or self.context_h is None or self.patch_emb_grid is None:
            return h_static

        # learned delta-h (normalized in [-1,1]) -> rescale to steps
        with torch.no_grad():
            node_patch_emb = self.patch_emb_grid[r, c].unsqueeze(0)  # (1,E)
            goal_vec = self.goal_vec_grid[r, c].unsqueeze(0)         # (1,2)
            delta_t = self.delta_t_table[min(int(t), self.delta_t_table.shape[0]-1)].unsqueeze(0)  # (1,1)
            h_static_t = self.h_static_norm_grid[r, c].unsqueeze(0)  # (1,1)
            delta_norm = self.model.predict_delta_h(self.context_h, node_patch_emb, goal_vec, delta_t, h_static_t)
            delta_norm = float(delta_norm.squeeze(0).cpu().numpy())

        if self.train_mode and self.stage.heuristic_noise_std and self.stage.heuristic_noise_std > 0:
            delta_norm = float(np.clip(delta_norm + np.random.normal(0.0, self.stage.heuristic_noise_std), -1.0, 1.0))

        delta_steps = delta_norm * (2.0 * self.env.size)
        h = h_static + self.stage.alpha * delta_steps
        return max(0.0, float(h))

    def plan_next(self, start: Tuple[int,int]) -> Tuple[int, int, int]:
        """Return (action, expansions, planned_length). action is int 0..4."""
        S = self.env.size
        goal = self.env.goal_pos
        horizon = int(self.stage.horizon)
        max_exp = int(self.stage.max_expansions)

        start_node = (start[0], start[1], 0)
        # (f, g, node)
        pq = [(0.0, 0.0, start_node)]
        g_score = {start_node: 0.0}
        came_from: Dict[Tuple[int,int,int], Tuple[int,int,int]] = {}
        best_node = start_node
        best_h = float("inf")
        expansions = 0

        while pq:
            f, g, cur = heapq.heappop(pq)
            expansions += 1
            if expansions >= max_exp:
                break
            r, c, t = cur

            # goal reached
            if (r,c) == goal:
                return self._first_action(came_from, cur, start_node), expansions, int(g)

            # horizon reached -> track best node
            if t >= horizon:
                h_here = abs(r - goal[0]) + abs(c - goal[1])
                if h_here < best_h:
                    best_h = h_here
                    best_node = cur
                continue

            # expand neighbors
            for action, (dr,dc) in enumerate([(-1,0),(1,0),(0,-1),(0,1),(0,0)]):
                nr, nc = r + dr, c + dc
                nt = t + 1

                # bounds
                if not (0 <= nr < S and 0 <= nc < S):
                    continue
                # static wall
                if self.env.static_map[nr, nc] == 1:
                    continue
                # dynamic collision at nt
                if nt < len(self.future_occ) and (nr,nc) in self.future_occ[nt]:
                    continue

                new_g = g + 1.0
                neigh = (nr, nc, nt)
                if new_g < g_score.get(neigh, float("inf")):
                    g_score[neigh] = new_g
                    h = self.heuristic(nr, nc, nt)
                    heapq.heappush(pq, (new_g + h, new_g, neigh))
                    came_from[neigh] = cur

        # fallback to best node seen (or start)
        return self._first_action(came_from, best_node, start_node), expansions, int(g_score.get(best_node, 0.0))

    def _first_action(self, came_from: Dict[Tuple[int,int,int], Tuple[int,int,int]],
                      cur: Tuple[int,int,int], start: Tuple[int,int,int]) -> int:
        # trace back to start to get first move
        path = []
        node = cur
        while node in came_from:
            path.append(node)
            node = came_from[node]
        if not path:
            return 4  # wait
        first = path[-1]  # node at time 1
        sr, sc, _ = start
        fr, fc, _ = first
        dr, dc = fr - sr, fc - sc
        if (dr,dc) == (-1,0): return 0
        if (dr,dc) == (1,0): return 1
        if (dr,dc) == (0,-1): return 2
        if (dr,dc) == (0,1): return 3
        return 4

# =============================================================================
# Rollout + training (within one Modal GPU container)
# =============================================================================

def run_episode(env: TransferDynamicMazeEnv,
                model: Optional[HeuristicModel],
                device: torch.device,
                stage: StageConfig,
                train_mode: bool = True) -> Dict[str, Any]:
    """Run one episode using Space-Time A*. Returns trajectory dict with training data + metrics."""
    # episode reset
    seed = int(np.random.randint(0, 1_000_000_000))
    env.reset(seed=seed, map_seed=seed, dyn_seed=seed)
    # precompute patch embedding grid if model present
    patch_emb_grid = None
    if model is not None:
        patch_emb_grid = model.precompute_patch_emb_grid(env.static_map, device=device)

    # init recurrent state
    state = None
    context_h = None
    if model is not None:
        state = model.init_state(batch_size=1, device=device, dtype=torch.float32)

    # storage
    obs_patches = []
    obs_goal = []
    obs_feats = []
    obs_mask = []
    obs_time = []
    obs_static_dist = []
    targets_mask = []
    expansions_list = []

    # metrics
    total_expansions = 0
    steps = 0

    done = False
    info = {"collision": False, "success": False, "timeout": False}

    while not done and steps < stage.max_steps:
        ar, ac = env.agent_pos
        # build observation tensors (B=1)
        patch_np = extract_local_patch(env.static_map, (ar,ac), model.cfg.patch_size if model is not None else 15)
        goal_vec = np.array([(env.goal_pos[0]-ar)/env.size, (env.goal_pos[1]-ac)/env.size], dtype=np.float32)
        static_dist = env.dist_to_goal[ar, ac]
        if math.isinf(float(static_dist)):
            static_dist = abs(ar-env.goal_pos[0]) + abs(ac-env.goal_pos[1])

        if model is not None:
            ofeat_np, omask_np = model.encode_obstacles(env, (ar,ac))
        else:
            ofeat_np = np.zeros((1, 6), dtype=np.float32)
            omask_np = np.zeros((1,), dtype=np.float32)

        t_feat = np.array([steps / max(1, stage.max_steps)], dtype=np.float32)

        if model is not None:
            obs_patch_t = torch.from_numpy(patch_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            obs_goal_t = torch.from_numpy(goal_vec).unsqueeze(0).to(device)
            obs_feats_t = torch.from_numpy(ofeat_np).unsqueeze(0).to(device)
            obs_mask_t = torch.from_numpy(omask_np).unsqueeze(0).to(device)
            obs_time_t = torch.from_numpy(t_feat).unsqueeze(0).to(device)
            obs_static_t = torch.tensor([[float(static_dist) / (2*env.size)]], device=device, dtype=torch.float32)

            context_h, state = model.step_context(obs_patch_t, obs_goal_t, obs_feats_t, obs_mask_t,
                                                  obs_time_t, obs_static_t, state, steps)
            # store raw obs for training
            obs_patches.append(patch_np)
            obs_goal.append(goal_vec)
            obs_feats.append(ofeat_np)
            obs_mask.append(omask_np)
            obs_time.append(t_feat)
            obs_static_dist.append(float(static_dist))

        # action selection via A*
        planner = SpaceTimeAStarPlanner(env, model, device, stage, patch_emb_grid, context_h if context_h is not None else None, train_mode=train_mode)
        action, expansions, _plen = planner.plan_next(env.agent_pos)

        # exploration: random action
        if train_mode and stage.epsilon_action > 0 and np.random.random() < stage.epsilon_action:
            action = int(np.random.randint(0, 5))

        # step env
        _, done, info = env.step(action)
        expansions_list.append(expansions)
        total_expansions += expansions
        steps += 1

    # terminal type
    collision = bool(info.get("collision", False))
    success = bool(info.get("success", False))
    timeout = bool(info.get("timeout", False))
    term = "success" if success else ("collision" if collision else "timeout")

    # compute per-step costs and discounted cost-to-go targets
    costs = []
    for t, exp in enumerate(expansions_list):
        c = 1.0 + stage.expansions_lambda * float(exp)
        costs.append(c)
    if term == "collision":
        costs[-1] += stage.collision_penalty
    elif term == "timeout":
        costs[-1] += stage.timeout_penalty

    gamma = stage.gamma
    cost_to_go = [0.0]*len(costs)
    running = 0.0
    for t in reversed(range(len(costs))):
        running = costs[t] + gamma * running
        cost_to_go[t] = running

    # delta-h target: cost_to_go - h_static
    delta_targets = []
    if model is not None:
        for t in range(len(cost_to_go)):
            h_static = float(obs_static_dist[t])
            delta_targets.append(float(np.clip((cost_to_go[t] - h_static) / (2.0 * env.size), -1.0, 1.0)))

    return {
        "term": term,
        "steps": steps,
        "total_expansions": int(total_expansions),
        "obs_patches": np.array(obs_patches, dtype=np.uint8) if model is not None else None,  # (T,P,P)
        "obs_goal": np.array(obs_goal, dtype=np.float32) if model is not None else None,      # (T,2)
        "obs_feats": np.array(obs_feats, dtype=np.float32) if model is not None else None,    # (T,M,D)
        "obs_mask": np.array(obs_mask, dtype=np.float32) if model is not None else None,      # (T,M)
        "obs_time": np.array(obs_time, dtype=np.float32) if model is not None else None,      # (T,1)
        "obs_static": np.array(obs_static_dist, dtype=np.float32) if model is not None else None, # (T,)
        "delta_targets": np.array(delta_targets, dtype=np.float32) if model is not None else None, # (T,)
    }

def pad_batch(episodes: List[Dict[str,Any]], patch_size: int, max_obs: int, grid_size: int) -> Dict[str, torch.Tensor]:
    """Pad variable-length episode sequences into batch tensors."""
    B = len(episodes)
    T_max = max(ep["steps"] for ep in episodes)
    P = patch_size
    M = max_obs

    patches = np.zeros((B, T_max, 1, P, P), dtype=np.float32)
    goal = np.zeros((B, T_max, 2), dtype=np.float32)
    feats = np.zeros((B, T_max, M, 6), dtype=np.float32)
    mask = np.zeros((B, T_max, M), dtype=np.float32)
    time_feat = np.zeros((B, T_max, 1), dtype=np.float32)
    static_dist = np.zeros((B, T_max, 1), dtype=np.float32)
    targets = np.zeros((B, T_max, 1), dtype=np.float32)
    valid = np.zeros((B, T_max), dtype=np.float32)

    for i, ep in enumerate(episodes):
        T = ep["steps"]
        patches[i, :T, 0] = ep["obs_patches"].astype(np.float32)
        goal[i, :T] = ep["obs_goal"]
        feats[i, :T] = ep["obs_feats"]
        mask[i, :T] = ep["obs_mask"]
        time_feat[i, :T] = ep["obs_time"]
        static_dist[i, :T, 0] = ep["obs_static"] / (2.0 * float(grid_size))
        targets[i, :T, 0] = ep["delta_targets"]
        valid[i, :T] = 1.0

    return {
        "patches": torch.from_numpy(patches),
        "goal": torch.from_numpy(goal),
        "feats": torch.from_numpy(feats),
        "mask": torch.from_numpy(mask),
        "time": torch.from_numpy(time_feat),
        "static": torch.from_numpy(static_dist),
        "targets": torch.from_numpy(targets),
        "valid": torch.from_numpy(valid),
    }

def train_on_batch(model: HeuristicModel, batch: Dict[str, torch.Tensor], cfg: ModelConfig,
                   optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Train for one minibatch epoch over padded episodes."""
    model.train()
    patches = batch["patches"].to(device)
    goal = batch["goal"].to(device)
    feats = batch["feats"].to(device)
    mask = batch["mask"].to(device)
    time_feat = batch["time"].to(device)
    static = batch["static"].to(device)
    targets = batch["targets"].to(device)
    valid = batch["valid"].to(device)

    B, T = goal.shape[0], goal.shape[1]

    state = model.init_state(batch_size=B, device=device, dtype=torch.float32)
    losses = []
    # unroll
    for t in range(T):
        context_h, state = model.step_context(patches[:,t], goal[:,t], feats[:,t], mask[:,t],
                                              time_feat[:,t], static[:,t], state, t)
        # node features for agent node: use same patch emb as node patch
        node_patch_emb = model.patch_encoder(patches[:,t])
        node_goal_vec = goal[:,t]
        delta_t = torch.zeros((B,1), device=device, dtype=torch.float32)
        h_static = static[:,t]
        pred = model.predict_delta_h(context_h, node_patch_emb, node_goal_vec, delta_t, h_static)
        # loss masked
        m = valid[:,t].unsqueeze(-1)
        loss_t = F.smooth_l1_loss(pred, targets[:,t], reduction="none")
        loss_t = (loss_t * m).sum() / m.sum().clamp(min=1.0)
        losses.append(loss_t)

    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.grad_clip and cfg.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return float(loss.detach().cpu().item())

def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Ensure optimizer state tensors are on the correct device after loading a checkpoint."""
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)

def save_checkpoint(model: HeuristicModel, optimizer: torch.optim.Optimizer, model_name: str, stage_id: str, update: int):
    ckpt_path = os.path.join(PATHS["checkpoints"], f"{model_name}_{stage_id}.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "stage_id": stage_id,
        "update": update,
        "model_cfg": asdict(model.cfg),
    }, ckpt_path)
    vol.commit()

def load_checkpoint(
    model: HeuristicModel,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    stage_id: str,
    device: torch.device,
) -> int:
    """Load per-stage checkpoint if it exists. Returns next update index to run.

    Note: if the checkpoint is corrupted (e.g., interrupted write), we fall back to starting the stage from scratch.
    """
    ckpt_path = os.path.join(PATHS["checkpoints"], f"{model_name}_{stage_id}.pt")
    if not os.path.exists(ckpt_path):
        return 0

    try:
        # Load directly onto the training device to avoid optimizer device mismatch
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        _move_optimizer_state_to_device(optimizer, device)
        return int(ckpt.get("update", 0)) + 1
    except Exception as e:
        print(f"⚠️  Failed to load checkpoint {ckpt_path}: {e!r}. Restarting stage from scratch.")
        return 0



def save_final_model(model: HeuristicModel, model_name: str, tag: str):
    out_path = os.path.join(PATHS["models"], f"{model_name}_{tag}.pt")
    torch.save({
        "model": model.state_dict(),
        "model_cfg": asdict(model.cfg),
        "tag": tag,
    }, out_path)
    vol.commit()

def model_file_exists(model_name: str, tag: str) -> bool:
    return os.path.exists(os.path.join(PATHS["models"], f"{model_name}_{tag}.pt"))

# =============================================================================
# Training function (Modal)
# =============================================================================

def _train_model_full_impl(model_name: str) -> Dict[str, Any]:
    """Train one model through the full curriculum (Family A), producing a final Stage3b checkpoint."""
    vol.reload()
    ensure_dirs()

    cfg = MODEL_CONFIGS[model_name]
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollout_device = torch.device("cpu")

    # instantiate model (train on GPU) + a CPU copy for A* rollouts
    model = HeuristicModel(cfg).to(train_device)
    model_rollout = HeuristicModel(cfg).to(rollout_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # stage-by-stage training
    for stage in STAGES:
        # if stage already finished (final model exists), skip
        tag = stage.stage_id + "_final"
        if model_file_exists(model_name, tag):
            print(f"✓ {model_name}: {stage.stage_id} already complete, skipping")
            # load that model into memory for next stage init
            state = torch.load(os.path.join(PATHS["models"], f"{model_name}_{tag}.pt"), map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            model_rollout.load_state_dict(model.state_dict())
            continue

        # resume from stage checkpoint if available
        start_update = load_checkpoint(model, optimizer, model_name, stage.stage_id, train_device)
        model_rollout.load_state_dict(model.state_dict())
        if start_update > 0:
            print(f"↻ {model_name}: resuming {stage.stage_id} from update {start_update}")

        # stage env config
        env_cfg = {
            "map_family": stage.map_family,
            "grid_size": stage.grid_size,
            "dynamics": stage.dynamics,
            "n_gates": stage.n_gates,
            "n_patrollers": stage.n_patrollers,
            "n_drifters": stage.n_drifters,
            "max_steps": stage.max_steps,
            "patch_size": cfg.patch_size,
        }
        env = TransferDynamicMazeEnv(env_cfg)

        for upd in range(start_update, stage.updates):
            t0 = time.time()
            # collect episodes
            episodes: List[Dict[str,Any]] = []
            metrics = {"success": 0, "collision": 0, "timeout": 0, "steps": [], "exp": []}

            # sync weights to rollout model and collect on CPU
            model_rollout.load_state_dict(model.state_dict())
            model_rollout.eval()
            for _ in range(cfg.episodes_per_update):
                ep = run_episode(env, model_rollout, rollout_device, stage, train_mode=True)
                episodes.append(ep)
                metrics[ep["term"]] += 1
                metrics["steps"].append(ep["steps"])
                metrics["exp"].append(ep["total_expansions"])

            # train with SGD epochs over shuffled episodes
            losses = []
            idxs = np.arange(len(episodes))
            for _epoch in range(cfg.sgd_epochs):
                np.random.shuffle(idxs)
                for i in range(0, len(idxs), cfg.minibatch_episodes):
                    mb = [episodes[j] for j in idxs[i:i+cfg.minibatch_episodes]]
                    batch = pad_batch(mb, cfg.patch_size, cfg.max_obstacles, stage.grid_size)
                    loss = train_on_batch(model, batch, cfg, optimizer, train_device)
                    losses.append(loss)

            avg_loss = float(np.mean(losses)) if losses else 0.0
            dt = time.time() - t0
            succ = metrics["success"] / max(1, cfg.episodes_per_update)
            coll = metrics["collision"] / max(1, cfg.episodes_per_update)
            tout = metrics["timeout"] / max(1, cfg.episodes_per_update)
            mean_steps = float(np.mean(metrics["steps"])) if metrics["steps"] else 0.0
            mean_exp = float(np.mean(metrics["exp"])) if metrics["exp"] else 0.0

            print(f"[{model_name}][{stage.stage_id}] upd {upd+1}/{stage.updates} "
                  f"loss={avg_loss:.4f} succ={succ:.2f} coll={coll:.2f} tout={tout:.2f} "
                  f"steps={mean_steps:.1f} exp={mean_exp:.0f} ({dt:.1f}s)")

            # checkpoint every 5 updates
            if (upd + 1) % 5 == 0 or (upd + 1) == stage.updates:
                save_checkpoint(model, optimizer, model_name, stage.stage_id, upd)

        # stage complete -> save final model for that stage
        save_final_model(model, model_name, tag)

    # final summary
    params = count_parameters(model)
    return {"name": model_name, "params": params, "status": "trained"}

# =============================================================================
# Few-shot adaptation function (Modal)
# =============================================================================



@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="H100",
    cpu=8,
    timeout=60 * 60 * 24,  # 24h (Modal max)
)
def train_model_full(model_name: str) -> Dict[str, Any]:
    """Train one model through the full curriculum (Family A), producing a final Stage3b checkpoint.

    H100 variant (used for ON-LSTM by default).
    """
    return _train_model_full_impl(model_name)


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="B200",
    cpu=8,
    timeout=60 * 60 * 24,  # 24h (Modal max)
)
def train_model_full_b200(model_name: str) -> Dict[str, Any]:
    """Train one model through the full curriculum (Family A), producing a final Stage3b checkpoint.

    B200 variant (used for HRM by default).
    """
    return _train_model_full_impl(model_name)


def _fewshot_adapt_impl(model_name: str, K: int) -> Dict[str, Any]:
    """Fine-tune a trained model for K episodes on the FEWSHOT_TARGET domain and save adapted checkpoint."""
    vol.reload()
    ensure_dirs()
    # Skip if few-shot checkpoint already exists (unless FORCE_FEWSHOT=1)
    tag_existing = f"fewshotK{K}"
    if model_file_exists(model_name, tag_existing) and os.environ.get("FORCE_FEWSHOT", "0").strip() != "1":
        return {"model": model_name, "K": K, "skipped": True, "reason": f"Few-shot checkpoint already exists: {tag_existing}"}

    cfg = MODEL_CONFIGS[model_name]
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollout_device = torch.device("cpu")

    # load base (after stage3b)
    base_tag = "stage3b_A64_D2_final"
    base_path = os.path.join(PATHS["models"], f"{model_name}_{base_tag}.pt")
    if not os.path.exists(base_path):
        # Training may not have reached stage3b yet (e.g., timeouts). Skip gracefully.
        return {"model": model_name, "K": K, "skipped": True, "reason": f"Base model not found: {base_path}"}

    ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    model = HeuristicModel(cfg).to(train_device)
    model_rollout = HeuristicModel(cfg).to(rollout_device)
    model.load_state_dict(ckpt["model"])
    model_rollout.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.5, weight_decay=cfg.weight_decay)

    # target env
    stage = StageConfig(stage_id=f"fewshot_{K}", map_family=FEWSHOT_TARGET.map_family, grid_size=FEWSHOT_TARGET.grid_size,
                        dynamics=FEWSHOT_TARGET.dynamics, n_gates=FEWSHOT_TARGET.n_gates,
                        n_patrollers=FEWSHOT_TARGET.n_patrollers, n_drifters=FEWSHOT_TARGET.n_drifters,
                        horizon=FEWSHOT_TARGET.horizon, max_steps=FEWSHOT_TARGET.max_steps,
                        updates=1,
                        gamma=0.99, expansions_lambda=1e-4, collision_penalty=50.0, timeout_penalty=25.0,
                        alpha=1.0, heuristic_noise_std=0.02, max_expansions=FEWSHOT_TARGET.max_expansions)

    env_cfg = {
        "map_family": stage.map_family,
        "grid_size": stage.grid_size,
        "dynamics": stage.dynamics,
        "n_gates": stage.n_gates,
        "n_patrollers": stage.n_patrollers,
        "n_drifters": stage.n_drifters,
        "max_steps": stage.max_steps,
        "patch_size": cfg.patch_size,
    }
    env = TransferDynamicMazeEnv(env_cfg)

    # collect K episodes and train for a few SGD epochs total
    episodes: List[Dict[str,Any]] = []
    model.eval()
    for _ in range(K):
        episodes.append(run_episode(env, model_rollout, rollout_device, stage, train_mode=True))

    # fine-tune
    idxs = np.arange(len(episodes))
    losses = []
    # scale SGD epochs with K modestly
    sgd_epochs = max(1, int(math.log10(max(10, K))))
    for _ in range(sgd_epochs):
        np.random.shuffle(idxs)
        for i in range(0, len(idxs), cfg.minibatch_episodes):
            mb = [episodes[j] for j in idxs[i:i+cfg.minibatch_episodes]]
            batch = pad_batch(mb, cfg.patch_size, cfg.max_obstacles, stage.grid_size)
            loss = train_on_batch(model, batch, cfg, optimizer, train_device)
            losses.append(loss)

    tag = f"fewshotK{K}"
    save_final_model(model, model_name, tag)

    return {"name": model_name, "K": K, "loss": float(np.mean(losses)) if losses else 0.0, "status": "adapted"}

# =============================================================================
# Evaluation (Modal)
# =============================================================================



@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="H100",
    cpu=8,
    timeout=60 * 60 * 12,  # 12h
)
def fewshot_adapt(model_name: str, K: int) -> Dict[str, Any]:
    """Fine-tune one trained model on the few-shot target domain.

    H100 variant (used for ON-LSTM by default).
    """
    return _fewshot_adapt_impl(model_name, K)


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="B200",
    cpu=8,
    timeout=60 * 60 * 12,  # 12h
)
def fewshot_adapt_b200(model_name: str, K: int) -> Dict[str, Any]:
    """Fine-tune one trained model on the few-shot target domain.

    B200 variant (used for HRM by default).
    """
    return _fewshot_adapt_impl(model_name, K)




def evaluate_suite(suite: EvalSuite,
                   model_state: Optional[Dict[str, Any]],
                   model_cfg: Optional[ModelConfig]) -> Dict[str, Any]:
    """Evaluate a single suite for either:
    - baseline (model_state=None): static A* heuristic only
    - learned heuristic (model_state provided): HRM / ON-LSTM delta-h heuristic

    Returns summary metrics for the suite.
    """
    device = torch.device("cpu")

    # Build an env config from the suite
    env_cfg: Dict[str, Any] = {
        "map_family": suite.map_family,
        "grid_size": int(suite.grid_size),
        "max_steps": int(suite.max_steps),
        "dynamics": suite.dynamics,
        "n_gates": int(suite.n_gates),
        "n_patrollers": int(suite.n_patrollers),
        "n_drifters": int(suite.n_drifters),
        "patch_size": int(model_cfg.patch_size) if model_cfg is not None else 15,
    }
    env = TransferDynamicMazeEnv(env_cfg)

    # StageConfig is used by the planner + run_episode; for eval we set updates=0 and disable exploration/noise.
    stage = StageConfig(
        stage_id=f"eval_{suite.suite_id}",
        map_family=suite.map_family,
        grid_size=int(suite.grid_size),
        dynamics=suite.dynamics,
        n_gates=int(suite.n_gates),
        n_patrollers=int(suite.n_patrollers),
        n_drifters=int(suite.n_drifters),
        horizon=int(suite.horizon),
        max_steps=int(suite.max_steps),
        updates=0,
        gamma=0.99,
        expansions_lambda=1e-4,   # only affects training targets; harmless here
        collision_penalty=50.0,
        timeout_penalty=25.0,
        alpha=float(suite.alpha),
        heuristic_noise_std=0.0,
        epsilon_action=0.0,
        max_expansions=int(suite.max_expansions),
    )

    model: Optional[HeuristicModel] = None
    if model_state is not None:
        assert model_cfg is not None, "model_cfg must be provided when model_state is provided"
        model = HeuristicModel(model_cfg).to(device)
        model.load_state_dict(model_state)
        model.eval()

    total = int(suite.episodes)
    succ = 0
    coll = 0
    tout = 0
    steps_sum = 0
    exp_sum = 0
    steps_succ = 0
    steps_fail = 0
    exp_succ = 0
    exp_fail = 0

    # Progress logging (enabled by default; set EVAL_PROGRESS=0 to disable)
    label = model_cfg.name if model_cfg is not None else "baseline_static_astar"
    show_progress = os.environ.get("EVAL_PROGRESS", "1").strip() != "0"
    try:
        print_every = int(os.environ.get("EVAL_PRINT_EVERY", "0").strip() or 0)
    except Exception:
        print_every = 0
    if print_every <= 0:
        # Default: ~5% increments (at least every episode for very small evals)
        print_every = max(1, total // 20)

    t0 = time.time()
    if show_progress:
        print(
            f"   ▶️  [{label}] {suite.suite_id}: {total} eps | grid={suite.grid_size} | dyn={suite.dynamics} | "
            f"h={suite.horizon} | max_steps={suite.max_steps} | gates={suite.n_gates} pat={suite.n_patrollers} drift={suite.n_drifters}",
            flush=True,
        )

    for i in range(total):
        traj = run_episode(env, model, device, stage, train_mode=False)
        term = str(traj.get("term", "timeout"))
        steps = int(traj.get("steps", 0))
        exps = int(traj.get("total_expansions", 0))

        steps_sum += steps
        exp_sum += exps

        if term == "success":
            succ += 1
            steps_succ += steps
            exp_succ += exps
        elif term == "collision":
            coll += 1
            steps_fail += steps
            exp_fail += exps
        else:
            tout += 1
            steps_fail += steps
            exp_fail += exps

        if show_progress and (((i + 1) % print_every == 0) or (i + 1 == total)):
            elapsed = time.time() - t0
            done = i + 1
            eps_per_s = done / max(1e-9, elapsed)
            eta_s = (total - done) / max(1e-9, eps_per_s)
            avg_steps_so_far = steps_sum / done
            exp_per_step = exp_sum / max(1, steps_sum)
            succ_rate = succ / done
            print(
                f"      [{label}|{suite.suite_id}] ep {done}/{total} "
                f"succ={succ_rate:.2f} coll={coll} tout={tout} "
                f"avg_steps={avg_steps_so_far:.1f} exp/step={exp_per_step:.1f} "
                f"eps/s={eps_per_s:.2f} ETA={eta_s/60:.1f}m",
                flush=True,
            )

    denom = max(1, total)
    fail_n = max(1, coll + tout)

    if show_progress:
        elapsed = time.time() - t0
        exp_per_step = exp_sum / max(1, steps_sum)
        print(
            f"   ✓ [{label}] {suite.suite_id} complete in {elapsed/60:.1f}m: "
            f"success_rate={succ/denom:.2f} coll={coll} tout={tout} "
            f"avg_steps={steps_sum/denom:.1f} exp/step={exp_per_step:.1f}",
            flush=True,
        )

    return {
        "total": total,
        "successes": succ,
        "collisions": coll,
        "timeouts": tout,
        "success_rate": succ / denom,
        "avg_steps": steps_sum / denom,
        "avg_steps_success": steps_succ / max(1, succ),
        "avg_steps_fail": steps_fail / fail_n,
        "avg_expansions": exp_sum / denom,
        "expansions_per_step": exp_sum / max(1, steps_sum),
        "avg_exp_success": exp_succ / max(1, succ),
        "avg_exp_fail": exp_fail / fail_n,
    }


@app.function(
    image=image,
    volumes={"/data": vol},
    cpu=4,
    timeout=60 * 60 * 12,  # 12h
)
def evaluate_all() -> Dict[str, Any]:
    """Evaluate all trained models + static A* baseline on the evaluation suites."""
    vol.reload()
    ensure_dirs()

    results: Dict[str, Any] = {"timestamp": time.time(), "suites": [asdict(s) for s in EVAL_SUITES], "models": {}}

    # baseline (static A*)
    print(f"Evaluating static A* baseline ({len(EVAL_SUITES)} suites)...", flush=True)
    baseline = {}
    for si, suite in enumerate(EVAL_SUITES, 1):
        print(f"  🧪 [baseline] suite {si}/{len(EVAL_SUITES)}: {suite.suite_id} ({suite.episodes} eps)", flush=True)
        baseline[suite.suite_id] = evaluate_suite(suite, model_state=None, model_cfg=None)
    results["models"]["baseline_static_astar"] = baseline

    # learned models
    n_models_total = len(MODEL_CONFIGS)
    for mi, (model_name, cfg) in enumerate(MODEL_CONFIGS.items(), 1):
        tag = "stage3b_A64_D2_final"
        model_path = os.path.join(PATHS["models"], f"{model_name}_{tag}.pt")
        if not os.path.exists(model_path):
            print(f"⚠️  missing model {model_name}, skipping eval", flush=True)
            continue
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state = ckpt["model"]
        model_results = {}
        print(f"Evaluating {model_name} ({mi}/{n_models_total})...", flush=True)
        for si, suite in enumerate(EVAL_SUITES, 1):
            print(f"  🧪 [{model_name}] suite {si}/{len(EVAL_SUITES)}: {suite.suite_id} ({suite.episodes} eps)", flush=True)
            model_results[suite.suite_id] = evaluate_suite(suite, model_state=model_state, model_cfg=cfg)
        results["models"][model_name] = model_results

# save json
    out_path = os.path.join(PATHS["results"], "eval_zero_shot.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    return results

@app.function(
    image=image,
    volumes={"/data": vol},
    cpu=4,
    timeout=60 * 60 * 6,  # 6h
)
def evaluate_fewshot() -> Dict[str, Any]:
    """Evaluate few-shot adapted models on the hard target suite."""
    vol.reload()
    ensure_dirs()

    results: Dict[str, Any] = {"timestamp": time.time(), "target_suite": asdict(FEWSHOT_TARGET), "models": {}}

    print(f"Evaluating few-shot target suite {FEWSHOT_TARGET.suite_id}...", flush=True)

    # baseline
    print(f"  🧪 [baseline] {FEWSHOT_TARGET.suite_id} ({FEWSHOT_TARGET.episodes} eps)", flush=True)
    results["models"]["baseline_static_astar"] = {
        FEWSHOT_TARGET.suite_id: evaluate_suite(FEWSHOT_TARGET, model_state=None, model_cfg=None)
    }

    n_models_total = len(MODEL_CONFIGS)
    for mi, (model_name, cfg) in enumerate(MODEL_CONFIGS.items(), 1):
        res: Dict[str, Any] = {}

        print(f"Evaluating {model_name} ({mi}/{n_models_total}) on target...", flush=True)

        # zero-shot (stage3b final)
        base_tag = "stage3b_A64_D2_final"
        base_path = os.path.join(PATHS["models"], f"{model_name}_{base_tag}.pt")
        if os.path.exists(base_path):
            print(f"  🧪 [{model_name}] zero-shot ({FEWSHOT_TARGET.episodes} eps)", flush=True)
            ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
            res["zero_shot"] = evaluate_suite(FEWSHOT_TARGET, model_state=ckpt["model"], model_cfg=cfg)
        else:
            res["zero_shot"] = None

        # few-shot checkpoints
        for K in FEWSHOT_K:
            tag = f"fewshotK{K}"
            p = os.path.join(PATHS["models"], f"{model_name}_{tag}.pt")
            if os.path.exists(p):
                print(f"  🧪 [{model_name}] K={K} ({FEWSHOT_TARGET.episodes} eps)", flush=True)
                ckpt2 = torch.load(p, map_location="cpu", weights_only=False)
                res[f"K{K}"] = evaluate_suite(FEWSHOT_TARGET, model_state=ckpt2["model"], model_cfg=cfg)
            else:
                res[f"K{K}"] = None

        results["models"][model_name] = res

    out_path = os.path.join(PATHS["results"], "eval_fewshot_target.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    return results

# =============================================================================
# Main entrypoint
# =============================================================================

@app.local_entrypoint()
def main():
    print("="*78)
    print("TRANSFER-FIRST A* AUGMENTATION — RL HEURISTIC LEARNING")
    print("="*78)
    print("Models:", list(MODEL_CONFIGS.keys()))
    print("Stages:", [s.stage_id for s in STAGES])
    print("Eval suites:", [s.suite_id for s in EVAL_SUITES])
    print()

    # ----------------------------
    # Cost controls / run selectors
    # ----------------------------
    max_parallel = int(os.environ.get("MAX_PARALLEL_MODELS", "2"))
    max_parallel = max(1, max_parallel)

    only = os.environ.get("ONLY_MODELS", "").strip()
    if only:
        only_set = {s.strip() for s in only.split(",") if s.strip()}
        model_names = [n for n in MODEL_CONFIGS.keys() if n in only_set]
        print("ONLY_MODELS active ->", model_names)
    else:
        model_names = list(MODEL_CONFIGS.keys())

    if not model_names:
        print("No models selected; exiting.")
        return

    print(f"Max parallel training containers: {max_parallel}")
    print()

    # ----------------------------
    # Curriculum training
    # ----------------------------
    skip_train = os.environ.get("SKIP_TRAIN", "0").strip() == "1"
    if skip_train:
        print("⏭️  SKIP_TRAIN=1 set; skipping curriculum training (using existing checkpoints/models).")
        print()
    else:
        results: Dict[str, Any] = {}
        failures: Dict[str, str] = {}

        for i in range(0, len(model_names), max_parallel):
            batch = model_names[i:i+max_parallel]
            print(f"\n🚀 Launching training batch {i//max_parallel+1} with {len(batch)} model(s): {batch}")

            handles = {}
            for name in batch:
                cfg = MODEL_CONFIGS[name]
                print(f"   🚀 Launching: {name} ({cfg.model_type}, requested_gpu={cfg.gpu})")
                # NOTE: Modal doesn't allow per-call dynamic gpu selection in a single decorator.
                train_fn = train_model_full_b200 if cfg.model_type == "hrm" else train_model_full
                handles[name] = train_fn.spawn(name)

            for name, h in handles.items():
                print(f"   ⏳ Waiting for {name}...")
                try:
                    res = h.get()
                    results[name] = res
                    print(f"   ✓ {name} finished: {res}")
                except Exception as e:
                    failures[name] = repr(e)
                    print(f"   ✗ {name} failed: {e!r}")

        if failures:
            print("\n⚠️  Some training jobs failed (others may still have completed):")
            for name, err in failures.items():
                print(f"   - {name}: {err}")
            print("You can rerun to resume from checkpoints once the underlying issue is fixed.\n")

        # ----------------------------
    # Few-shot + evaluation (only when stage3b finals exist)
    # ----------------------------
    print("🔎 Checking which models have stage3b final checkpoints...")
    avail = available_base_models.remote()
    ready_models = avail.get("available", [])
    # restrict to models we actually ran this invocation
    ready_models = [m for m in ready_models if m in set(model_names)]

    if not ready_models:
        print("⚠️ No stage3b final models found yet; skipping few-shot + evaluation for now.")
        print("   Rerun the script to resume training from checkpoints.")
        return

    skip_fewshot_all = os.environ.get("SKIP_FEWSHOT", "0").strip() == "1"
    skip_fewshot_adapt = os.environ.get("SKIP_FEWSHOT_ADAPT", "0").strip() == "1"

    if skip_fewshot_all:
        print("⏭️  SKIP_FEWSHOT=1 set; skipping few-shot adaptation and few-shot evaluation.")
    elif skip_fewshot_adapt:
        print("⏭️  SKIP_FEWSHOT_ADAPT=1 set; skipping few-shot adaptation (will still evaluate if checkpoints exist).")
    else:
        adapth = []
        for name in ready_models:
            for K in FEWSHOT_K:
                print(f"🧪 Launching few-shot adaptation: {name}, K={K}")
                fs_fn = fewshot_adapt_b200 if MODEL_CONFIGS[name].model_type == "hrm" else fewshot_adapt
                adapth.append((name, K, fs_fn.spawn(name, K)))

        fs_failures: Dict[str, str] = {}
        for name, K, h in adapth:
            try:
                _ = h.get()
            except Exception as e:
                fs_failures[f"{name}_K{K}"] = repr(e)
                print(f"✗ Few-shot failed for {name} K={K}: {e!r}")

        if fs_failures:
            print("\n⚠️  Some few-shot jobs failed:")
            for key, err in fs_failures.items():
                print(f"   - {key}: {err}")
            print()

        print("✓ Few-shot adaptation complete")

    # Evaluate zero-shot suites
    print("📊 Running zero-shot evaluation suites...")
    zs = evaluate_all.remote()
    print("Zero-shot evaluation saved:", zs.keys())

    # Evaluate few-shot target
    if skip_fewshot_all:
        print("⏭️  SKIP_FEWSHOT=1 set; skipping few-shot target evaluation.")
    else:
        print("📊 Running few-shot target evaluation...")
        fs = evaluate_fewshot.remote()
        print("Few-shot evaluation saved:", fs.keys())

    print("✅ Done. Artifacts in", PATHS["root"])
