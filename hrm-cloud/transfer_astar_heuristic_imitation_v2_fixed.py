#!/usr/bin/env python3
"""
TRANSFER-FIRST A* AUGMENTATION — HEURISTIC IMITATION V2
======================================================

This experiment implements the "fixes" identified in the recent analysis:

1) Units/scale alignment:
   - Train the model to predict a residual heuristic in *steps*:
       Δh_target(x,y,t) = max(0, true_cost_to_go(x,y,t) - h_static(x,y))
   - A* priority uses: f = g + h_static + α * ReLU(Δh_pred)

2) Baseline ceiling removal:
   - Evaluate under strict per-replan expansion budgets (e.g., 200/500/2000)
     so heuristic quality matters for success, not only efficiency.

3) Survivorship-bias fix:
   - Supervised imitation is trained on the *search graveyard*:
     closed-list nodes expanded by an oracle Space-Time A* planner.

4) "RNN inside search tree" fix:
   - HRM / ON-LSTM runs ONCE per env step to produce a context vector.
   - A lightweight Node-MLP evaluates thousands of A* nodes using that context.

5) Node-centric features:
   - Goal vector and dynamic occupancy patches are expressed relative to the node.

6) Explicit obstacle projection:
   - Node patches include *projected* dynamic occupancy at node time t' (from simulator rollout).

Training is supervised (no RL): we generate an offline dataset by running an oracle planner on
Family-A maps across a curriculum of stages, then train HRM vs ON-LSTM models end-to-end
(encoder + node-MLP). We then evaluate zero-shot transfer on Families B/C and measure
success vs budget curves; and we optionally run few-shot adaptation (fine-tune) on a target OOD suite.

Run on Modal like:
    python -m modal run HRMv2/hrm-cloud/transfer_astar_heuristic_imitation_v2.py --detach

Useful environment variables:
    ONLY_MODELS="hrm_3m,onlstm_3m"       # restrict models
    MAX_PARALLEL_TRAIN=2                 # training concurrency
    MAX_PARALLEL_COLLECT=8               # data collection concurrency
    MAX_PARALLEL_EVAL=24                 # eval concurrency
    SKIP_COLLECT=1                       # skip dataset collection if cached
    SKIP_TRAIN=1                         # skip training (use existing checkpoints/models)
    SKIP_FEWSHOT=1                       # skip few-shot adapt
    EVAL_EPISODES=100                    # override eval episodes per suite
    EVAL_BUDGETS="200,500,2000"          # expansion budgets per replanning step
    ALPHA="1.0"                          # heuristic scale factor
"""

from __future__ import annotations

import os
import math
import time
import json
import random
import itertools
import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Silence harmless NNPACK warnings on many container CPUs
try:
    if hasattr(torch.backends, "nnpack") and hasattr(torch.backends.nnpack, "set_flags"):
        torch.backends.nnpack.set_flags(False)
except Exception:
    pass

import modal

# -----------------------------
# Modal setup
# -----------------------------

APP_NAME = "transfer-astar-heuristic-imitation-v2"
VOLUME_NAME = "transfer-astar-heuristic-imitation-v2-vol"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "torch>=2.4.0",
            "numpy",
            "tqdm",
        ]
    )
)

app = modal.App(APP_NAME)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# New experiment root (kept separate from earlier V2 runs to avoid mixing
# datasets/checkpoints produced before the critical A* + labeling fixes).
DATA_ROOT = "/data/transfer_astar_heuristic_imitation_v2_fixpack"
DATASETS_DIR = f"{DATA_ROOT}/datasets"
MODELS_DIR = f"{DATA_ROOT}/models"
CHECKPOINTS_DIR = f"{DATA_ROOT}/checkpoints"
RESULTS_DIR = f"{DATA_ROOT}/results"

# -----------------------------
# Config and helpers
# -----------------------------

def _env_flag(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return default if v is None else v

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return int(v)

def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return float(v)

def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out

def _parse_csv_strs(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]

def _ensure_dirs():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Curriculum + evaluation suites
# -----------------------------

@dataclass(frozen=True)
class Stage:
    stage_id: str
    family: str          
    size: int            
    dynamics: str        
    max_steps: int       
    plan_horizon: int    
    oracle_max_exp: int  
    collect_samples: int 
    nodes_per_sample: int
    collect_every: int   
    train_epochs: int    
    n_gates: int
    n_patrollers: int
    n_drifters: int

@dataclass(frozen=True)
class EvalSuite:
    suite_id: str
    family: str
    size: int
    dynamics: str
    max_steps: int
    plan_horizon: int
    n_gates: int
    n_patrollers: int
    n_drifters: int
    episodes: int

# Bumping epochs to allow proper convergence on offline supervised dataset
STAGES: List[Stage] = [
    Stage("stage1_A32_D0", "A", 32, "D0", 80, 18, 8000, 1200, 48, 2, 25, 0, 0, 0),
    Stage("stage2_A32_D1", "A", 32, "D1", 90, 18, 12000, 1800, 48, 2, 30, 1, 2, 2),
    Stage("stage3a_A64_D1", "A", 64, "D1", 160, 20, 15000, 2200, 48, 3, 30, 1, 4, 4),
    Stage("stage3b_A64_D2", "A", 64, "D2", 180, 22, 20000, 3200, 48, 3, 40, 2, 6, 6),
]

def build_eval_suites(default_episodes: int) -> List[EvalSuite]:
    return [
        EvalSuite("ID_A32_D1", "A", 32, "D1", 90, 18, 1, 2, 2, default_episodes),
        EvalSuite("ID_A64_D2", "A", 64, "D2", 180, 22, 2, 6, 6, default_episodes),
        EvalSuite("OOD_B32_D1", "B", 32, "D1", 90, 18, 1, 2, 2, default_episodes),
        EvalSuite("OOD_C32_D1", "C", 32, "D1", 90, 18, 1, 2, 2, default_episodes),
        EvalSuite("OOD_B64_D2", "B", 64, "D2", 180, 22, 2, 6, 6, default_episodes),
        EvalSuite("OOD_C64_D2", "C", 64, "D2", 180, 22, 2, 6, 6, default_episodes),
    ]

# -----------------------------
# Map generation
# -----------------------------

def _bfs_reachable(walls: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> bool:
    n = walls.shape[0]
    sx, sy = start
    gx, gy = goal
    if walls[sx, sy] or walls[gx, gy]:
        return False
    q = [(sx, sy)]
    seen = np.zeros((n, n), dtype=np.uint8)
    seen[sx, sy] = 1
    head = 0
    while head < len(q):
        x, y = q[head]
        head += 1
        if (x, y) == (gx, gy):
            return True
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < n and 0 <= ny < n and not walls[nx, ny] and not seen[nx, ny]:
                seen[nx, ny] = 1
                q.append((nx, ny))
    return False

def _sample_free_cell(rng: random.Random, walls: np.ndarray) -> Tuple[int,int]:
    n = walls.shape[0]
    while True:
        x = rng.randrange(1, n-1)
        y = rng.randrange(1, n-1)
        if not walls[x, y]:
            return (x, y)

def _manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def generate_map_family_A(rng: random.Random, n: int) -> np.ndarray:
    walls = np.ones((n, n), dtype=np.uint8)
    walls[0,:] = 1; walls[-1,:] = 1; walls[:,0] = 1; walls[:,-1] = 1
    num_rooms = 8 if n <= 32 else 18
    rooms: List[Tuple[int,int,int,int]] = []
    for _ in range(num_rooms):
        w = rng.randrange(4, 8 if n <= 32 else 10)
        h = rng.randrange(4, 8 if n <= 32 else 10)
        x = rng.randrange(1, n - w - 1)
        y = rng.randrange(1, n - h - 1)
        walls[x:x+w, y:y+h] = 0
        rooms.append((x, y, w, h))

    centers = [(x+w//2, y+h//2) for (x,y,w,h) in rooms]
    rng.shuffle(centers)
    for i in range(1, len(centers)):
        x1, y1 = centers[i-1]
        x2, y2 = centers[i]
        if rng.random() < 0.5:
            for y in range(min(y1,y2), max(y1,y2)+1): walls[x1, y] = 0
            for x in range(min(x1,x2), max(x1,x2)+1): walls[x, y2] = 0
        else:
            for x in range(min(x1,x2), max(x1,x2)+1): walls[x, y1] = 0
            for y in range(min(y1,y2), max(y1,y2)+1): walls[x2, y] = 0

    for _ in range((n*n)//40):
        x = rng.randrange(1, n-1)
        y = rng.randrange(1, n-1)
        walls[x, y] = 0
    return walls

def generate_map_family_B(rng: random.Random, n: int) -> np.ndarray:
    walls = np.ones((n, n), dtype=np.uint8)
    walls[0,:] = 1; walls[-1,:] = 1; walls[:,0] = 1; walls[:,-1] = 1
    for x in range(1, n-1, 2):
        for y in range(1, n-1, 2):
            walls[x, y] = 0

    stack = [(1,1)]
    seen = set(stack)
    def neighbors(cx, cy):
        out = []
        for dx, dy in ((2,0),(-2,0),(0,2),(0,-2)):
            nx, ny = cx+dx, cy+dy
            if 1 <= nx < n-1 and 1 <= ny < n-1 and (nx, ny) not in seen:
                out.append((nx, ny, dx//2, dy//2))
        rng.shuffle(out)
        return out

    while stack:
        cx, cy = stack[-1]
        nbrs = neighbors(cx, cy)
        if not nbrs:
            stack.pop()
            continue
        nx, ny, wx, wy = nbrs[0]
        walls[cx+wx, cy+wy] = 0
        stack.append((nx, ny))
        seen.add((nx, ny))

    for _ in range((n*n)//80):
        x = rng.randrange(1, n-1)
        y = rng.randrange(1, n-1)
        walls[x, y] = 0
    return walls

def generate_map_family_C(rng: random.Random, n: int) -> np.ndarray:
    walls = np.zeros((n, n), dtype=np.uint8)
    walls[0,:] = 1; walls[-1,:] = 1; walls[:,0] = 1; walls[:,-1] = 1
    p = 0.10 if n <= 32 else 0.08
    for x in range(1, n-1):
        for y in range(1, n-1):
            if rng.random() < p:
                walls[x, y] = 1

    for _ in range(6 if n <= 32 else 12):
        x = rng.randrange(1, n-2)
        for y in range(1, n-1):
            walls[x, y] = 0
    return walls

def generate_map(family: str, seed: int, n: int) -> Tuple[np.ndarray, Tuple[int,int], Tuple[int,int]]:
    rng = random.Random(seed)
    if family == "A": walls = generate_map_family_A(rng, n)
    elif family == "B": walls = generate_map_family_B(rng, n)
    elif family == "C": walls = generate_map_family_C(rng, n)
    else: raise ValueError(f"Unknown family: {family}")

    for _ in range(200):
        start = _sample_free_cell(rng, walls)
        goal = _sample_free_cell(rng, walls)
        if _manhattan(start, goal) < n//2: continue
        if _bfs_reachable(walls, start, goal):
            return walls, start, goal

    start = (1,1)
    goal = (n-2,n-2)
    walls[start] = 0
    walls[goal] = 0
    return walls, start, goal

@dataclass
class Drifter:
    x: int; y: int; dx: int; dy: int

@dataclass
class Patroller:
    x: int; y: int; dx: int; dy: int

@dataclass
class Gate:
    x: int; y: int; period: int; open_len: int; phase: int

def _dir_right(dx: int, dy: int) -> Tuple[int,int]: return (dy, -dx)
def _dir_left(dx: int, dy: int) -> Tuple[int,int]: return (-dy, dx)

def _move_entity(x: int, y: int, dx: int, dy: int, walls: np.ndarray) -> Tuple[int,int,int,int]:
    n = walls.shape[0]
    nx, ny = x + dx, y + dy
    if not (0 <= nx < n and 0 <= ny < n) or walls[nx, ny]:
        dx, dy = -dx, -dy
        nx, ny = x + dx, y + dy
        if not (0 <= nx < n and 0 <= ny < n) or walls[nx, ny]:
            return x, y, dx, dy
    return nx, ny, dx, dy

def step_dynamics(walls: np.ndarray, patrollers: List[Patroller], drifters: List[Drifter], gates: List[Gate]) -> None:
    for g in gates: g.phase = (g.phase + 1) % g.period
    for d in drifters: d.x, d.y, d.dx, d.dy = _move_entity(d.x, d.y, d.dx, d.dy, walls)
    for p in patrollers:
        cand = [(p.dx, p.dy), _dir_right(p.dx, p.dy), _dir_left(p.dx, p.dy), (-p.dx, -p.dy)]
        chosen = (p.dx, p.dy)
        for dx, dy in cand:
            nx, ny = p.x + dx, p.y + dy
            if 0 <= nx < walls.shape[0] and 0 <= ny < walls.shape[1] and not walls[nx, ny]:
                chosen = (dx, dy)
                break
        p.dx, p.dy = chosen
        p.x, p.y, p.dx, p.dy = _move_entity(p.x, p.y, p.dx, p.dy, walls)

def gate_closed(g: Gate) -> bool:
    return g.phase >= g.open_len

def init_dynamics(rng: random.Random, walls: np.ndarray, n_gates: int, n_patrollers: int, n_drifters: int) -> Tuple[List[Gate], List[Patroller], List[Drifter]]:
    n = walls.shape[0]
    gates, patrollers, drifters = [], [], []
    def free_cell(): return _sample_free_cell(rng, walls)

    for _ in range(n_gates):
        for _try in range(50):
            gx, gy = free_cell()
            if walls[gx-1, gy] + walls[gx+1, gy] + walls[gx, gy-1] + walls[gx, gy+1] <= 2:
                period = rng.choice([10, 12, 14, 16])
                open_len = rng.randint(period//3, (2*period)//3)
                phase = rng.randrange(period)
                gates.append(Gate(gx, gy, period, open_len, phase))
                break

    for _ in range(n_patrollers):
        px, py = free_cell()
        dx, dy = rng.choice([(1,0),(-1,0),(0,1),(0,-1)])
        patrollers.append(Patroller(px, py, dx, dy))

    for _ in range(n_drifters):
        x, y = free_cell()
        dx, dy = rng.choice([(1,0),(-1,0),(0,1),(0,-1)])
        drifters.append(Drifter(x, y, dx, dy))

    return gates, patrollers, drifters

def compute_static_distances(walls: np.ndarray, goal: Tuple[int,int]) -> np.ndarray:
    n = walls.shape[0]
    gx, gy = goal
    dist = np.full((n, n), 10_000, dtype=np.int16)
    if walls[gx, gy]: return dist
    q = [(gx, gy)]
    dist[gx, gy] = 0
    head = 0
    while head < len(q):
        x, y = q[head]; head += 1
        d = dist[x, y]
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < n and 0 <= ny < n and not walls[nx, ny] and dist[nx, ny] > d + 1:
                dist[nx, ny] = d + 1
                q.append((nx, ny))
    return dist

def simulate_occupancy(walls: np.ndarray, gates0: List[Gate], pat0: List[Patroller], drift0: List[Drifter], max_steps: int) -> Dict[str, np.ndarray]:
    n = walls.shape[0]
    T = max_steps
    gate_closed_seq = np.zeros((T+1, n, n), dtype=np.uint8)
    pat_seq = np.zeros((T+1, n, n), dtype=np.uint8)
    drift_seq = np.zeros((T+1, n, n), dtype=np.uint8)

    gates = [Gate(g.x, g.y, g.period, g.open_len, g.phase) for g in gates0]
    pats = [Patroller(p.x, p.y, p.dx, p.dy) for p in pat0]
    drifts = [Drifter(d.x, d.y, d.dx, d.dy) for d in drift0]

    def stamp(t: int):
        for g in gates:
            if gate_closed(g): gate_closed_seq[t, g.x, g.y] = 1
        for p in pats: pat_seq[t, p.x, p.y] = 1
        for d in drifts: drift_seq[t, d.x, d.y] = 1

    stamp(0)
    for t in range(1, T+1):
        step_dynamics(walls, pats, drifts, gates)
        stamp(t)

    walls_u8 = walls.astype(np.uint8)
    blocked = np.clip(walls_u8[None, :, :] + gate_closed_seq + pat_seq + drift_seq, 0, 1).astype(np.uint8)

    return {"walls": walls_u8, "gate": gate_closed_seq, "pat": pat_seq, "drift": drift_seq, "blocked": blocked}

INF16 = np.int16(30_000)

def compute_true_cost_to_goal(blocked_seq: np.ndarray, goal: Tuple[int,int], max_steps: int) -> np.ndarray:
    T = max_steps
    n = blocked_seq.shape[1]
    gx, gy = goal
    dist = np.full((T+1, n, n), INF16, dtype=np.int16)

    for t in range(T+1):
        if blocked_seq[t, gx, gy] == 0:
            dist[t, gx, gy] = 0

    for t in range(T-1, -1, -1):
        nxt = dist[t+1]
        cur = dist[t]
        for x in range(n):
            for y in range(n):
                if blocked_seq[t, x, y]: continue
                best = nxt[x, y]
                if x+1 < n: best = min(best, nxt[x+1, y])
                if x-1 >= 0: best = min(best, nxt[x-1, y])
                if y+1 < n: best = min(best, nxt[x, y+1])
                if y-1 >= 0: best = min(best, nxt[x, y-1])
                if best < INF16:
                    cand = np.int16(1 + int(best))
                    if cand < cur[x, y]:
                        cur[x, y] = cand
    return dist

MAX_GATES = 2
MAX_PATS = 8
MAX_DRIFTS = 8

def build_obs_vector(n: int, agent: Tuple[int,int], goal: Tuple[int,int], gates: List[Gate], pats: List[Patroller], drifts: List[Drifter]) -> np.ndarray:
    ax, ay = agent
    gx, gy = goal
    feat: List[float] = []
    feat.extend([ax / (n-1), ay / (n-1)])
    feat.extend([(gx-ax)/n, (gy-ay)/n])

    for i in range(MAX_GATES):
        if i < len(gates):
            g = gates[i]
            feat.extend([(g.x-ax)/n, (g.y-ay)/n]) 
            feat.append(0.0 if gate_closed(g) else 1.0)
            feat.append((g.phase % g.period) / g.period)
        else:
            feat.extend([0.0, 0.0, 0.0, 0.0]) 

    for i in range(MAX_PATS):
        if i < len(pats):
            p = pats[i]
            feat.extend([(p.x-ax)/n, (p.y-ay)/n, float(p.dx), float(p.dy)])
        else:
            feat.extend([0.0, 0.0, 0.0, 0.0])

    for i in range(MAX_DRIFTS):
        if i < len(drifts):
            d = drifts[i]
            feat.extend([(d.x-ax)/n, (d.y-ay)/n, float(d.dx), float(d.dy)])
        else:
            feat.extend([0.0, 0.0, 0.0, 0.0])

    return np.asarray(feat, dtype=np.float32)

def build_obs_sequence(history: Sequence[np.ndarray], H: int, obs_dim: int) -> np.ndarray:
    seq = np.zeros((H, obs_dim), dtype=np.float32)
    take = min(len(history), H)
    if take > 0:
        seq[-take:] = np.stack(history[-take:], axis=0)
    return seq

PATCH_R = 7  
PATCH_SIZE = PATCH_R * 2 + 1
PATCH_CH = 4  

META_DIM = 6  

def extract_patch(occ: Dict[str,np.ndarray], t_abs: int, x: int, y: int) -> np.ndarray:
    n = occ["walls"].shape[0]
    xs = max(0, x-PATCH_R); xe = min(n, x+PATCH_R+1)
    ys = max(0, y-PATCH_R); ye = min(n, y+PATCH_R+1)

    patch = np.zeros((PATCH_CH, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    patch[0, :, :] = 1  
    
    dx0 = xs - (x-PATCH_R)
    dy0 = ys - (y-PATCH_R)
    dx1 = dx0 + (xe-xs)
    dy1 = dy0 + (ye-ys)

    patch[0, dx0:dx1, dy0:dy1] = occ["walls"][xs:xe, ys:ye]
    patch[1, dx0:dx1, dy0:dy1] = occ["gate"][t_abs, xs:xe, ys:ye]
    patch[2, dx0:dx1, dy0:dy1] = occ["pat"][t_abs, xs:xe, ys:ye]
    patch[3, dx0:dx1, dy0:dy1] = occ["drift"][t_abs, xs:xe, ys:ye]
    return patch

def build_node_meta(n: int, node_xy: Tuple[int,int], goal_xy: Tuple[int,int], t_offset: int, plan_horizon: int, h_static: int) -> np.ndarray:
    x, y = node_xy
    gx, gy = goal_xy
    goal_dx = (gx - x) / n
    goal_dy = (gy - y) / n
    dt_norm = t_offset / max(1, plan_horizon)
    h_norm = float(h_static) / (2*n) 
    x_norm = x / (n-1)
    y_norm = y / (n-1)
    return np.asarray([goal_dx, goal_dy, dt_norm, h_norm, x_norm, y_norm], dtype=np.float32)

Action = int  
ACTIONS: List[Tuple[int,int]] = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]

@dataclass
class PlanResult:
    found: bool
    actions: List[Action]
    expansions: int
    closed: List[Tuple[int,int,int]]  

def space_time_astar(
    start_xy: Tuple[int,int],
    goal_xy: Tuple[int,int],
    t0_abs: int,
    plan_horizon: int,
    max_expansions: int,
    occ: Dict[str,np.ndarray],
    static_dist: np.ndarray,
    heuristic_batch_fn, 
    alpha: float,
) -> PlanResult:
    n = occ["walls"].shape[0]
    gx, gy = goal_xy
    sx, sy = start_xy

    max_t_abs = int(occ["blocked"].shape[0] - 1)
    t0_abs = min(max(0, int(t0_abs)), max_t_abs)

    if occ["blocked"][t0_abs, sx, sy]:
        return PlanResult(False, [4], 0, [])

    T = plan_horizon
    g_best = np.full((T+1, n, n), 10_000, dtype=np.int16)
    g_best[0, sx, sy] = 0

    parent: Dict[Tuple[int,int,int], Tuple[int,int,int,Action]] = {}

    def h_static(x: int, y: int) -> int:
        return int(static_dist[x, y])

    counter = itertools.count()
    heap: List[Tuple[float,int,int,int,int,int]] = []

    # Clamp and round initial node
    hd0 = max(0.0, float(list(heuristic_batch_fn([(sx, sy, 0)]))[0]))
    hs0 = h_static(sx, sy)
    f0 = round(0.0 + float(hs0) + alpha * hd0, 2)
    heapq.heappush(heap, (f0, 0, next(counter), sx, sy, 0))

    closed: List[Tuple[int,int,int]] = []
    expansions = 0
    best_goal_state: Optional[Tuple[int,int,int]] = None

    best_frontier_state: Tuple[int,int,int] = (sx, sy, 0)
    best_frontier_h = float("inf")

    while heap and expansions < max_expansions:
        f, neg_g, _tie, x, y, t = heapq.heappop(heap)
        g = -neg_g
        if g != int(g_best[t, x, y]):
            continue

        closed.append((x, y, t))
        expansions += 1

        if t > 0:
            h_est = float(f) - float(g)
            if h_est < best_frontier_h:
                best_frontier_h = h_est
                best_frontier_state = (x, y, t)

        t_abs = min(t0_abs + t, max_t_abs)
        if x == gx and y == gy and occ["blocked"][t_abs, x, y] == 0:
            best_goal_state = (x, y, t)
            break

        if t >= T:
            continue

        nt = t + 1
        nt_abs = min(t0_abs + nt, max_t_abs)

        children: List[Tuple[int,int,int,int,int,int]] = [] 
        for a, (dx, dy) in enumerate(ACTIONS):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < n and 0 <= ny < n):
                continue
            if occ["blocked"][nt_abs, nx, ny]:
                continue
            ng = g + 1
            if ng < int(g_best[nt, nx, ny]):
                hs = h_static(nx, ny)
                children.append((nx, ny, nt, ng, a, hs))

        if not children:
            continue

        states = [(c[0], c[1], c[2]) for c in children]
        hds = list(heuristic_batch_fn(states))

        for (nx, ny, nt, ng, a, hs), hd in zip(children, hds):
            if ng >= int(g_best[nt, nx, ny]):
                continue
            g_best[nt, nx, ny] = ng
            parent[(nx, ny, nt)] = (x, y, t, a)

            # CRITICAL FIX: Clamp negative network hallucination, and round to allow Depth-First Tie-Breaking
            hd_f = max(0.0, float(hd))
            nf = round(float(ng + hs) + alpha * hd_f, 2)
            heapq.heappush(heap, (nf, -ng, next(counter), nx, ny, nt))

            h_child = nf - float(ng)
            if h_child < best_frontier_h:
                best_frontier_h = h_child
                best_frontier_state = (nx, ny, nt)

    if best_goal_state is None:
        best_goal_state = best_frontier_state

    actions_rev: List[Action] = []
    cur = best_goal_state
    while cur != (sx, sy, 0):
        if cur not in parent:
            break
        px, py, pt, a = parent[cur]
        actions_rev.append(a)
        cur = (px, py, pt)
    actions_rev.reverse()
    if not actions_rev:
        actions_rev = [4]

    found = (best_goal_state[0] == gx and best_goal_state[1] == gy and occ["blocked"][min(t0_abs + best_goal_state[2], max_t_abs), gx, gy] == 0)
    return PlanResult(found, actions_rev, expansions, closed)

@dataclass
class Episode:
    walls: np.ndarray
    start: Tuple[int,int]
    goal: Tuple[int,int]
    gates: List[Gate]
    pats: List[Patroller]
    drifts: List[Drifter]
    max_steps: int

def step_episode(ep: Episode, agent_xy: Tuple[int,int], action: Action) -> Tuple[Tuple[int,int], bool, Dict[str,Any]]:
    n = ep.walls.shape[0]
    step_dynamics(ep.walls, ep.pats, ep.drifts, ep.gates)

    dx, dy = ACTIONS[action]
    ax, ay = agent_xy
    nx, ny = ax + dx, ay + dy
    if not (0 <= nx < n and 0 <= ny < n) or ep.walls[nx, ny]:
        nx, ny = ax, ay  

    for g in ep.gates:
        if g.x == nx and g.y == ny and gate_closed(g):
            nx, ny = ax, ay
            break

    collided = False
    for p in ep.pats:
        if p.x == nx and p.y == ny:
            collided = True
            break
    if not collided:
        for d in ep.drifts:
            if d.x == nx and d.y == ny:
                collided = True
                break

    reached = (nx, ny) == ep.goal and not collided
    info = {"collided": collided, "reached": reached}
    done = collided or reached
    return (nx, ny), done, info

class ONLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, chunk_size: int = 10):
        super().__init__()
        assert hidden_size % chunk_size == 0, "hidden_size must be divisible by chunk_size"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_chunks = hidden_size // chunk_size

        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.W_master = nn.Linear(input_size + hidden_size, 2 * self.n_chunks)

    def cumsoftmax(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(F.softmax(x, dim=-1), dim=-1)

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, c = state
        combined = torch.cat([x, h], dim=-1)
        gates = self.W(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        master_gates = self.W_master(combined)
        mf, mi = torch.chunk(master_gates, 2, dim=-1)
        mf = 1.0 - self.cumsoftmax(mf)  
        mi = self.cumsoftmax(mi)        

        mf_e = mf.repeat_interleave(self.chunk_size, dim=-1)
        mi_e = mi.repeat_interleave(self.chunk_size, dim=-1)

        # CRITICAL FIX: The Shen et al. implementation uses master forget as a direct mask. 
        # If mf_e=0 (inactive), we want perfect memory (f_hat=1). If mf_e=1 (active), normal forget gate (f_hat=f).
        f_hat = f * mf_e + (1.0 - mf_e)
        i_hat = i * mi_e

        c_new = f_hat * c + i_hat * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)

class ONLSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, chunk_size: int = 10):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            in_sz = input_size if l == 0 else hidden_size
            self.layers.append(ONLSTMCell(in_sz, hidden_size, chunk_size))
        self.hidden_size = hidden_size

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = seq.shape
        h = [seq.new_zeros((B, self.hidden_size)) for _ in self.layers]
        c = [seq.new_zeros((B, self.hidden_size)) for _ in self.layers]
        for t in range(T):
            x = seq[:, t]
            for l, cell in enumerate(self.layers):
                out, (hn, cn) = cell(x, (h[l], c[l]))
                h[l], c[l] = hn, cn
                x = out
        return h[-1]

class HRMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, slow_hidden: int, slow_update_every: int = 4):
        super().__init__()
        self.fast = nn.GRU(input_size, hidden_size, batch_first=True)
        self.slow_cell = nn.GRUCell(hidden_size, slow_hidden)
        self.slow_update_every = slow_update_every
        self.out_proj = nn.Linear(hidden_size + slow_hidden, hidden_size)

        self.hidden_size = hidden_size
        self.slow_hidden = slow_hidden

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = seq.shape
        fast_out, _ = self.fast(seq) 
        slow = seq.new_zeros((B, self.slow_hidden))
        for t in range(T):
            if (t % self.slow_update_every) == 0:
                slow = self.slow_cell(fast_out[:, t], slow)
        ctx = torch.cat([fast_out[:, -1], slow], dim=-1)
        return torch.tanh(self.out_proj(ctx))

class PatchEncoder(nn.Module):
    def __init__(self, patch_ch: int, emb_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(patch_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(32 * 7 * 7, emb_dim)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        x = self.conv(patch)
        x = F.relu(self.fc(x))
        return x

class NodeMLP(nn.Module):
    def __init__(self, patch_emb_dim: int, ctx_dim: int, meta_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(patch_emb_dim + ctx_dim + meta_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, patch_emb: torch.Tensor, ctx: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        x = torch.cat([patch_emb, ctx, meta], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        raw = self.out(x).squeeze(-1)
        # CRITICAL FIX: Removed ReLU to prevent dead gradients during supervised imitation.
        return raw  

class HeuristicModel(nn.Module):
    def __init__(self, encoder: nn.Module, ctx_dim: int, patch_emb_dim: int = 128, meta_dim: int = META_DIM, node_hidden: int = 256):
        super().__init__()
        self.encoder = encoder
        self.ctx_dim = ctx_dim
        self.patch_enc = PatchEncoder(PATCH_CH, patch_emb_dim)
        self.node_mlp = NodeMLP(patch_emb_dim, ctx_dim, meta_dim, hidden=node_hidden)

    def forward(self, obs_seq: torch.Tensor, patch: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        B, Nn, C, P, _ = patch.shape
        ctx = self.encoder(obs_seq) 
        patch_f = patch.float().reshape(B * Nn, C, P, P)
        meta_f = meta.view(B * Nn, -1).float()
        ctx_rep = ctx[:, None, :].expand(B, Nn, self.ctx_dim).contiguous().view(B * Nn, self.ctx_dim)
        patch_emb = self.patch_enc(patch_f)
        delta = self.node_mlp(patch_emb, ctx_rep, meta_f).view(B, Nn)
        return delta

    @torch.no_grad()
    def compute_context(self, obs_seq: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs_seq)

    @torch.no_grad()
    def eval_nodes(self, ctx: torch.Tensor, patch: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        Nn = patch.shape[0]
        patch_f = patch.float()  
        meta_f = meta.float()
        ctx_rep = ctx.expand(Nn, -1)
        patch_emb = self.patch_enc(patch_f)
        delta = self.node_mlp(patch_emb, ctx_rep, meta_f)
        return delta

@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_type: str  
    obs_dim: int
    ctx_dim: int
    onlstm_hidden: int = 256
    onlstm_layers: int = 1
    onlstm_chunk: int = 16
    hrm_fast: int = 256
    hrm_slow: int = 128
    hrm_update_every: int = 4
    patch_emb: int = 128
    node_hidden: int = 256

def build_model_configs(obs_dim: int) -> Dict[str, ModelConfig]:
    return {
        "onlstm_300k": ModelConfig("onlstm_300k", "onlstm", obs_dim, ctx_dim=128, onlstm_hidden=128, onlstm_layers=1, onlstm_chunk=16, patch_emb=96, node_hidden=160),
        "onlstm_1m":   ModelConfig("onlstm_1m",   "onlstm", obs_dim, ctx_dim=192, onlstm_hidden=192, onlstm_layers=1, onlstm_chunk=16, patch_emb=128, node_hidden=256),
        "onlstm_3m":   ModelConfig("onlstm_3m",   "onlstm", obs_dim, ctx_dim=256, onlstm_hidden=256, onlstm_layers=2, onlstm_chunk=16, patch_emb=160, node_hidden=320),
        "onlstm_10m":  ModelConfig("onlstm_10m",  "onlstm", obs_dim, ctx_dim=384, onlstm_hidden=384, onlstm_layers=2, onlstm_chunk=16, patch_emb=192, node_hidden=512),
        "hrm_302k":    ModelConfig("hrm_302k",    "hrm", obs_dim, ctx_dim=128, hrm_fast=128, hrm_slow=64, hrm_update_every=4, patch_emb=96, node_hidden=160),
        "hrm_3m":      ModelConfig("hrm_3m",      "hrm", obs_dim, ctx_dim=256, hrm_fast=256, hrm_slow=128, hrm_update_every=4, patch_emb=160, node_hidden=320),
        "hrm_10m":     ModelConfig("hrm_10m",     "hrm", obs_dim, ctx_dim=384, hrm_fast=384, hrm_slow=192, hrm_update_every=3, patch_emb=192, node_hidden=512),
    }

def build_model(cfg: ModelConfig) -> HeuristicModel:
    if cfg.model_type == "onlstm":
        enc = ONLSTMEncoder(cfg.obs_dim, cfg.onlstm_hidden, num_layers=cfg.onlstm_layers, chunk_size=cfg.onlstm_chunk)
        ctx_dim = cfg.onlstm_hidden
    elif cfg.model_type == "hrm":
        enc = HRMEncoder(cfg.obs_dim, hidden_size=cfg.hrm_fast, slow_hidden=cfg.hrm_slow, slow_update_every=cfg.hrm_update_every)
        ctx_dim = cfg.hrm_fast
    else:
        raise ValueError(cfg.model_type)
    model = HeuristicModel(enc, ctx_dim=ctx_dim, patch_emb_dim=cfg.patch_emb, meta_dim=META_DIM, node_hidden=cfg.node_hidden)
    return model

@dataclass
class StepSample:
    obs_seq: np.ndarray              
    node_patch: np.ndarray           
    node_meta: np.ndarray            
    target_delta: np.ndarray         
    mask: np.ndarray                 

def save_samples(path: str, samples: List[StepSample]) -> None:
    obs_seq = np.stack([s.obs_seq for s in samples], axis=0).astype(np.float16)
    node_patch = np.stack([s.node_patch for s in samples], axis=0).astype(np.uint8)
    node_meta = np.stack([s.node_meta for s in samples], axis=0).astype(np.float16)
    target_delta = np.stack([s.target_delta for s in samples], axis=0).astype(np.float16)
    mask = np.stack([s.mask for s in samples], axis=0).astype(np.uint8)
    payload = {
        "obs_seq": torch.from_numpy(obs_seq),
        "node_patch": torch.from_numpy(node_patch),
        "node_meta": torch.from_numpy(node_meta),
        "target_delta": torch.from_numpy(target_delta),
        "mask": torch.from_numpy(mask),
    }
    torch.save(payload, path)

def load_dataset(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")

class StepDataset(torch.utils.data.Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.obs_seq = data["obs_seq"]     
        self.node_patch = data["node_patch"]  
        self.node_meta = data["node_meta"]    
        self.target_delta = data["target_delta"]  
        self.mask = data["mask"]  
        self.S = self.obs_seq.shape[0]

    def __len__(self) -> int:
        return int(self.S)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.obs_seq[idx].float(),
            self.node_patch[idx],
            self.node_meta[idx].float(),
            self.target_delta[idx].float(),
            self.mask[idx].float(),
        )

def make_episode(seed: int, suite_family: str, n: int, max_steps: int, n_gates: int, n_pats: int, n_drifts: int) -> Episode:
    walls, start, goal = generate_map(suite_family, seed, n)
    rng = random.Random(seed + 999)
    gates, pats, drifts = init_dynamics(rng, walls, n_gates, n_pats, n_drifts)
    return Episode(walls, start, goal, gates, pats, drifts, max_steps=max_steps)

def oracle_collect_step_sample(
    ep: Episode,
    occ: Dict[str,np.ndarray],
    dist_abs: np.ndarray,
    static_dist: np.ndarray,
    t_abs: int,
    agent_xy: Tuple[int,int],
    obs_hist: List[np.ndarray],
    nodes_per_sample: int,
    plan_horizon: int,
    oracle_max_exp: int,
    alpha: float,
    rng: random.Random,
) -> Tuple[StepSample, Action, int]:
    
    n = ep.walls.shape[0]
    obs_dim = obs_hist[-1].shape[0]
    H = 20

    obs_seq = build_obs_sequence(obs_hist, H, obs_dim)

    def hfn_batch(states: List[Tuple[int,int,int]]) -> List[float]:
        return [0.0] * len(states)

    plan = space_time_astar(
        start_xy=agent_xy,
        goal_xy=ep.goal,
        t0_abs=t_abs,
        plan_horizon=plan_horizon,
        max_expansions=oracle_max_exp,
        occ=occ,
        static_dist=static_dist,
        heuristic_batch_fn=hfn_batch,
        alpha=alpha,
    )
    action0 = plan.actions[0] if plan.actions else 4

    closed = plan.closed
    if not closed:
        closed = [(agent_xy[0], agent_xy[1], 0)]

    path_like = closed[: min(8, len(closed))]
    rest = closed[len(path_like):]
    rng.shuffle(rest)

    selected = path_like + rest[: max(0, nodes_per_sample - len(path_like))]
    N = nodes_per_sample
    patch_arr = np.zeros((N, PATCH_CH, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    meta_arr = np.zeros((N, META_DIM), dtype=np.float32)
    target = np.zeros((N,), dtype=np.float32)
    mask = np.zeros((N,), dtype=np.uint8)

    for i, (x, y, t_off) in enumerate(selected[:N]):
        t_node_abs = t_abs + t_off
        if t_node_abs > ep.max_steps:
            continue
        patch_arr[i] = extract_patch(occ, t_node_abs, x, y)
        hs = int(static_dist[x, y])
        meta_arr[i] = build_node_meta(n, (x,y), ep.goal, t_off, plan_horizon, hs)

        true_ctg = int(dist_abs[t_node_abs, x, y])
        if true_ctg >= int(INF16):
            true_ctg = int(hs + plan_horizon * 4)

        delta = max(0.0, float(true_ctg - hs))
        delta = min(delta, float(plan_horizon * 4))
        target[i] = delta
        mask[i] = 1

    sample = StepSample(obs_seq=obs_seq, node_patch=patch_arr, node_meta=meta_arr, target_delta=target, mask=mask)
    return sample, action0, plan.expansions

@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 12,
    volumes={"/data": vol},
)
def collect_data_chunk(stage_id: str, chunk_id: int, num_samples: int, seed_base: int) -> str:
    _ensure_dirs()
    stage = next(s for s in STAGES if s.stage_id == stage_id)
    rng = random.Random(seed_base + 1000 * chunk_id)

    out_path = f"{DATASETS_DIR}/{stage_id}__chunk{chunk_id:03d}.pt"
    if os.path.exists(out_path):
        return out_path

    samples: List[StepSample] = []
    episodes_tried = 0

    while len(samples) < num_samples:
        ep_seed = seed_base + 10_000 * chunk_id + episodes_tried
        episodes_tried += 1
        ep = make_episode(ep_seed, stage.family, stage.size, stage.max_steps, stage.n_gates, stage.n_patrollers, stage.n_drifters)

        occ = simulate_occupancy(ep.walls, ep.gates, ep.pats, ep.drifts, ep.max_steps)
        dist_abs = compute_true_cost_to_goal(occ["blocked"], ep.goal, ep.max_steps)
        static_dist = compute_static_distances(ep.walls, ep.goal)

        agent = ep.start
        obs_hist: List[np.ndarray] = []
        for t_abs in range(ep.max_steps):
            obs = build_obs_vector(stage.size, agent, ep.goal, ep.gates, ep.pats, ep.drifts)
            obs_hist.append(obs)

            if (t_abs % stage.collect_every) == 0:
                sample, action0, _exp = oracle_collect_step_sample(
                    ep=ep, occ=occ, dist_abs=dist_abs, static_dist=static_dist,
                    t_abs=t_abs, agent_xy=agent, obs_hist=obs_hist,
                    nodes_per_sample=stage.nodes_per_sample,
                    plan_horizon=stage.plan_horizon,
                    oracle_max_exp=stage.oracle_max_exp,
                    alpha=1.0,
                    rng=rng,
                )
                samples.append(sample)
                a0 = action0
                
                if len(samples) >= num_samples:
                    break
            else:
                # CRITICAL FIX: Only run standard A* if the Oracle didn't just give us the answer!
                def hfn0_batch(states: List[Tuple[int,int,int]]) -> List[float]:
                    return [0.0] * len(states)

                plan = space_time_astar(
                    start_xy=agent,
                    goal_xy=ep.goal,
                    t0_abs=t_abs,
                    plan_horizon=stage.plan_horizon,
                    max_expansions=min(stage.oracle_max_exp, 8000),
                    occ=occ,
                    static_dist=static_dist,
                    heuristic_batch_fn=hfn0_batch,
                    alpha=1.0,
                )
                a0 = plan.actions[0] if plan.actions else 4

            agent, done, info = step_episode(ep, agent, a0)
            if done:
                break

    save_samples(out_path, samples[:num_samples])
    try:
        vol.commit()
    except Exception as e:
        print(f"[collect][{stage_id}][chunk{chunk_id:03d}] volume commit warning: {e}")
    return out_path

@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 2,
    volumes={"/data": vol},
)
def merge_chunks(stage_id: str, chunk_paths: List[str]) -> str:
    _ensure_dirs()
    out_path = f"{DATASETS_DIR}/{stage_id}__merged.pt"
    if os.path.exists(out_path):
        return out_path

    obs_list: List[torch.Tensor] = []
    patch_list: List[torch.Tensor] = []
    meta_list: List[torch.Tensor] = []
    tgt_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []

    for p in chunk_paths:
        if not os.path.exists(p):
            for _ in range(30):
                time.sleep(1)
                if os.path.exists(p):
                    break
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing dataset chunk: {p}. "
                f"This usually means the chunk job failed or its outputs were not committed to the Volume. "
                f"Check the collect_data_chunk logs for exceptions, and avoid disconnecting the client mid-collection."
            )

        d = torch.load(p, map_location="cpu")
        obs_list.append(d["obs_seq"])
        patch_list.append(d["node_patch"])
        meta_list.append(d["node_meta"])
        tgt_list.append(d["target_delta"])
        mask_list.append(d["mask"])
        
    merged = {
        "obs_seq": torch.cat(obs_list, dim=0),
        "node_patch": torch.cat(patch_list, dim=0),
        "node_meta": torch.cat(meta_list, dim=0),
        "target_delta": torch.cat(tgt_list, dim=0),
        "mask": torch.cat(mask_list, dim=0),
    }
    torch.save(merged, out_path)
    try:
        vol.commit()
    except Exception as e:
        print(f"[merge][{stage_id}] volume commit warning: {e}")
    return out_path

@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=60 * 10,
    volumes={"/data": vol},
)
def check_cached_dataset(stage_id: str) -> Dict[str, Any]:
    vol.reload()
    _ensure_dirs()
    merged = f"{DATASETS_DIR}/{stage_id}__merged.pt"
    chunks = []
    if os.path.exists(DATASETS_DIR):
        for fn in os.listdir(DATASETS_DIR):
            if fn.startswith(stage_id + "__chunk") and fn.endswith(".pt"):
                chunks.append(f"{DATASETS_DIR}/{fn}")
    return {
        "merged_exists": os.path.exists(merged),
        "merged_path": merged,
        "chunks": sorted(chunks),
    }

def _checkpoint_path(model_name: str, stage_id: str) -> str:
    return f"{CHECKPOINTS_DIR}/{model_name}__{stage_id}.pt"

def _final_model_path(model_name: str, stage_id: str) -> str:
    return f"{MODELS_DIR}/{model_name}__{stage_id}.pt"

def save_checkpoint(path: str, model: nn.Module, opt: torch.optim.Optimizer, epoch: int, best_loss: float) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_loss": best_loss,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
        },
        path,
    )
    vol.commit()

def load_checkpoint(path: str, model: nn.Module, opt: torch.optim.Optimizer) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    return int(ckpt["epoch"]) + 1, float(ckpt.get("best_loss", 1e9))

@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 12,
    volumes={"/data": vol},
)
def train_model_h100(model_name: str, stage_id: str, dataset_path: str, seed: int = 0) -> Dict[str, Any]:
    return _train_model_impl(model_name, stage_id, dataset_path, device="cuda", seed=seed)

@app.function(
    image=image,
    gpu="B200",
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 12,
    volumes={"/data": vol},
)
def train_model_b200(model_name: str, stage_id: str, dataset_path: str, seed: int = 0) -> Dict[str, Any]:
    return _train_model_impl(model_name, stage_id, dataset_path, device="cuda", seed=seed)

def _train_model_impl(model_name: str, stage_id: str, dataset_path: str, device: str, seed: int) -> Dict[str, Any]:
    vol.reload()
    _ensure_dirs()
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = load_dataset(dataset_path)
    ds = StepDataset(data)

    obs_dim = int(ds.obs_seq.shape[-1])
    cfgs = build_model_configs(obs_dim)
    cfg = cfgs[model_name]
    model = build_model(cfg).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"[{model_name}][{stage_id}] params={params:,} dataset={len(ds)}")

    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)

    ckpt_path = _checkpoint_path(model_name, stage_id)
    start_epoch = 0
    best_loss = 1e9
    if os.path.exists(ckpt_path):
        try:
            start_epoch, best_loss = load_checkpoint(ckpt_path, model, opt)
            print(f"[{model_name}][{stage_id}] resume epoch={start_epoch} best_loss={best_loss:.6f}")
        except Exception as e:
            print("Failed to load checkpoint:", e)

    if start_epoch == 0:
        prev_stage_id = None
        for i, s in enumerate(STAGES):
            if s.stage_id == stage_id:
                if i > 0:
                    prev_stage_id = STAGES[i-1].stage_id
                break
        if prev_stage_id is not None:
            prev_path = _final_model_path(model_name, prev_stage_id)
            if os.path.exists(prev_path):
                try:
                    prev = torch.load(prev_path, map_location="cpu")
                    model.load_state_dict(prev["state"], strict=False)
                    print(f"[{model_name}][{stage_id}] init from prev stage {prev_stage_id}")
                except Exception as e:
                    print(f"[{model_name}][{stage_id}] failed to init from prev stage:", e)

    stage_obj = next((s for s in STAGES if s.stage_id == stage_id), None)
    default_epochs = stage_obj.train_epochs if stage_obj is not None else 30

    override = _env_int(f"EPOCHS__{stage_id}", -1)
    if override > 0:
        epochs = override
    else:
        epochs = _env_int("EPOCHS_DEFAULT", default_epochs)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"[{model_name}][{stage_id}] training_epochs={epochs}")
    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        running = 0.0
        n_batches = 0
        for obs_seq, node_patch, node_meta, tgt, mask in dl:
            obs_seq = obs_seq.to(device)
            node_patch = node_patch.to(device)
            node_meta = node_meta.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)

            pred = model(obs_seq, node_patch, node_meta)
            diff = pred - tgt
            loss = F.smooth_l1_loss(diff * mask, torch.zeros_like(diff), reduction="sum") / (mask.sum() + 1e-6)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item())
            n_batches += 1

        epoch_loss = running / max(1, n_batches)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            
        scheduler.step() # <--- CRITICAL FIX: Schedule LR 
        save_checkpoint(ckpt_path, model, opt, epoch, best_loss)
        print(f"[{model_name}][{stage_id}] epoch {epoch+1}/{epochs} loss={epoch_loss:.6f} best={best_loss:.6f}")

    train_time = (time.time() - t0) / 60.0
    final_path = _final_model_path(model_name, stage_id)
    torch.save({"cfg": cfg.__dict__, "state": model.state_dict(), "params": params}, final_path)
    try:
        vol.commit()
    except Exception as e:
        print(f"[train][{model_name}][{stage_id}] volume commit warning: {e}")
    print(f"[{model_name}][{stage_id}] saved -> {final_path} ({train_time:.1f} min)")
    return {"name": model_name, "stage": stage_id, "params": params, "train_min": train_time, "status": "trained"}

# -----------------------------
# Few-shot adaptation (GPU)
# -----------------------------

@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 6,
    volumes={"/data": vol},
)
def fewshot_adapt_h100(model_name: str, base_stage: str, suite_id: str, k_episodes: int, seed: int = 0) -> Dict[str, Any]:
    return _fewshot_adapt_impl(model_name, base_stage, suite_id, k_episodes, device="cuda", seed=seed)

@app.function(
    image=image,
    gpu="B200",
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 6,
    volumes={"/data": vol},
)
def fewshot_adapt_b200(model_name: str, base_stage: str, suite_id: str, k_episodes: int, seed: int = 0) -> Dict[str, Any]:
    return _fewshot_adapt_impl(model_name, base_stage, suite_id, k_episodes, device="cuda", seed=seed)

def _fewshot_dataset_path(model_name: str, suite_id: str, k: int) -> str:
    return f"{DATASETS_DIR}/fewshot__{suite_id}__K{k}__{model_name}.pt"

def _fewshot_model_path(model_name: str, suite_id: str, k: int) -> str:
    return f"{MODELS_DIR}/{model_name}__fewshot_{suite_id}__K{k}.pt"

def _fewshot_adapt_impl(model_name: str, base_stage: str, suite_id: str, k_episodes: int, device: str, seed: int) -> Dict[str, Any]:
    vol.reload()
    _ensure_dirs()
    rng = random.Random(seed + 1234)

    eval_suites = build_eval_suites(default_episodes=100)
    suite = next(s for s in eval_suites if s.suite_id == suite_id)

    base_path = _final_model_path(model_name, base_stage)
    if not os.path.exists(base_path):
        raise FileNotFoundError(base_path)
    saved = torch.load(base_path, map_location="cpu")
    obs_dim = int(saved["cfg"]["obs_dim"])

    samples: List[StepSample] = []
    for epi in range(k_episodes):
        seed_ep = 10_000_000 + epi
        ep = make_episode(seed_ep, suite.family, suite.size, suite.max_steps, suite.n_gates, suite.n_patrollers, suite.n_drifters)
        occ = simulate_occupancy(ep.walls, ep.gates, ep.pats, ep.drifts, ep.max_steps)
        dist_abs = compute_true_cost_to_goal(occ["blocked"], ep.goal, ep.max_steps)
        static_dist = compute_static_distances(ep.walls, ep.goal)

        agent = ep.start
        obs_hist: List[np.ndarray] = []
        for t_abs in range(ep.max_steps):
            obs = build_obs_vector(suite.size, agent, ep.goal, ep.gates, ep.pats, ep.drifts)
            obs_hist.append(obs)
            if (t_abs % 3) == 0:
                sample, action0, _ = oracle_collect_step_sample(
                    ep, occ, dist_abs, static_dist, t_abs, agent, obs_hist,
                    nodes_per_sample=48,
                    plan_horizon=suite.plan_horizon,
                    oracle_max_exp=25_000,
                    alpha=1.0,
                    rng=rng,
                )
                samples.append(sample)
                a0 = action0
            else:
                def hfn0_batch(states: List[Tuple[int,int,int]]) -> List[float]:
                    return [0.0] * len(states)

                plan = space_time_astar(agent, ep.goal, t_abs, suite.plan_horizon, 8000, occ, static_dist, hfn0_batch, 1.0)
                a0 = plan.actions[0] if plan.actions else 4
                
            agent, done, _info = step_episode(ep, agent, a0)
            if done:
                break

    ds_path = _fewshot_dataset_path(model_name, suite_id, k_episodes)
    save_samples(ds_path, samples)
    try:
        vol.commit()
    except Exception as e:
        print(f"[fewshot][dataset] volume commit warning: {e}")
    print(f"[fewshot] dataset saved -> {ds_path} ({len(samples)} samples)")

    cfgs = build_model_configs(obs_dim)
    cfg = cfgs[model_name]
    model = build_model(cfg).to(device)
    model.load_state_dict(saved["state"])
    params = sum(p.numel() for p in model.parameters())
    print(f"[fewshot] loaded base model {model_name} params={params:,}")

    data = load_dataset(ds_path)
    ds = StepDataset(data)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    epochs = _env_int("FEWSHOT_EPOCHS", 25)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    model.train()
    
    for epc in range(epochs):
        running = 0.0
        nb = 0
        for obs_seq, node_patch, node_meta, tgt, mask in dl:
            obs_seq = obs_seq.to(device)
            node_patch = node_patch.to(device)
            node_meta = node_meta.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)

            pred = model(obs_seq, node_patch, node_meta)
            diff = pred - tgt
            loss = F.smooth_l1_loss(diff * mask, torch.zeros_like(diff), reduction="sum") / (mask.sum() + 1e-6)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item())
            nb += 1
            
        scheduler.step()
        print(f"[fewshot][{model_name}][{suite_id}][K={k_episodes}] epoch {epc+1}/{epochs} loss={running/max(1,nb):.6f}")

    out_path = _fewshot_model_path(model_name, suite_id, k_episodes)
    torch.save({"cfg": cfg.__dict__, "state": model.state_dict(), "params": params}, out_path)
    try:
        vol.commit()
    except Exception as e:
        print(f"[fewshot][model] volume commit warning: {e}")
    print(f"[fewshot] saved -> {out_path}")
    return {"name": model_name, "suite": suite_id, "k": k_episodes, "params": params, "status": "adapted"}

# -----------------------------
# Evaluation (CPU, parallelizable)
# -----------------------------

def run_policy_episode(
    suite: EvalSuite,
    seed: int,
    model_payload: Optional[Dict[str,Any]],
    alpha: float,
    max_expansions: int,
) -> Dict[str, Any]:
    """
    Run one episode with receding-horizon ST-A*.
    If model_payload is None, use baseline static heuristic (Δh=0).
    """
    rng = random.Random(seed)
    ep = make_episode(seed, suite.family, suite.size, suite.max_steps, suite.n_gates, suite.n_patrollers, suite.n_drifters)

    occ = simulate_occupancy(ep.walls, ep.gates, ep.pats, ep.drifts, ep.max_steps)
    static_dist = compute_static_distances(ep.walls, ep.goal)

    model: Optional[HeuristicModel] = None
    device = torch.device("cpu")
    if model_payload is not None:
        cfg_dict = model_payload["cfg"]
        cfg = ModelConfig(**cfg_dict)
        model = build_model(cfg).to(device)
        model.load_state_dict(model_payload["state"])
        model.eval()

    agent = ep.start
    obs_hist: List[np.ndarray] = []
    total_exp = 0
    steps = 0
    collided = False
    reached = False

    while steps < ep.max_steps:
        t_abs = steps
        obs = build_obs_vector(suite.size, agent, ep.goal, ep.gates, ep.pats, ep.drifts)
        obs_hist.append(obs)

        if model is None:
            def hfn_batch(states: List[Tuple[int,int,int]]) -> List[float]:
                return [0.0] * len(states)
        else:
            obs_seq = torch.from_numpy(build_obs_sequence(obs_hist, 20, obs.shape[0])).unsqueeze(0).float()
            with torch.no_grad():
                ctx = model.compute_context(obs_seq).squeeze(0)

            cache: Dict[Tuple[int,int,int], float] = {}

            def hfn_batch(states: List[Tuple[int,int,int]]) -> List[float]:
                results: List[float] = [0.0] * len(states)
                to_eval_idx: List[int] = []
                to_eval_states: List[Tuple[int,int,int]] = []

                for i, st in enumerate(states):
                    if st in cache:
                        results[i] = cache[st]
                        continue
                    x, y, t_off = st
                    t_node_abs = min(t_abs + t_off, ep.max_steps)
                    if t_node_abs > ep.max_steps:
                        cache[st] = 0.0
                        results[i] = 0.0
                        continue
                    to_eval_idx.append(i)
                    to_eval_states.append(st)

                if to_eval_states:
                    patches: List[np.ndarray] = []
                    metas: List[np.ndarray] = []
                    for (x, y, t_off) in to_eval_states:
                        t_node_abs = min(t_abs + t_off, ep.max_steps)
                        patches.append(extract_patch(occ, t_node_abs, x, y))
                        hs = int(static_dist[x, y])
                        metas.append(build_node_meta(suite.size, (x, y), ep.goal, t_off, suite.plan_horizon, hs))

                    patch_t = torch.from_numpy(np.stack(patches)).float()
                    meta_t = torch.from_numpy(np.stack(metas)).float()
                    with torch.no_grad():
                        deltas = model.eval_nodes(ctx.unsqueeze(0), patch_t, meta_t).cpu().numpy()

                    for idx, st, d in zip(to_eval_idx, to_eval_states, deltas):
                        d_float = float(d)
                        results[idx] = d_float
                        cache[st] = d_float

                return results

        plan = space_time_astar(
            start_xy=agent,
            goal_xy=ep.goal,
            t0_abs=t_abs,
            plan_horizon=suite.plan_horizon,
            max_expansions=max_expansions,
            occ=occ,
            static_dist=static_dist,
            heuristic_batch_fn=hfn_batch,
            alpha=alpha,
        )
        total_exp += plan.expansions
        a0 = plan.actions[0] if plan.actions else 4

        agent, done, info = step_episode(ep, agent, a0)
        steps += 1
        if info.get("collided"):
            collided = True
            break
        if info.get("reached"):
            reached = True
            break

    timeout = (not collided) and (not reached)
    return {
        "success": 1 if reached else 0,
        "collision": 1 if collided else 0,
        "timeout": 1 if timeout else 0,
        "steps": steps,
        "expansions": total_exp,
        "exp_per_step": total_exp / max(1, steps),
    }

@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=60 * 60 * 3,
    volumes={"/data": vol},
)
def evaluate_pair(
    suite: EvalSuite,
    model_name: str,
    model_path: Optional[str],
    alpha: float,
    max_expansions: int,
    seed_base: int,
) -> Dict[str, Any]:
    vol.reload()
    _ensure_dirs()
    payload = None
    if model_path is not None:
        payload = torch.load(model_path, map_location="cpu")

    metrics = {"success": 0, "collision": 0, "timeout": 0, "steps": 0.0, "expansions": 0.0}
    t0 = time.time()
    for i in range(suite.episodes):
        seed = seed_base + i
        r = run_policy_episode(suite, seed, payload, alpha=alpha, max_expansions=max_expansions)
        for k in ("success","collision","timeout"): metrics[k] += r[k]
        metrics["steps"] += r["steps"]
        metrics["expansions"] += r["expansions"]
        if (i+1) % max(1, suite.episodes // 5) == 0:
            elapsed = time.time() - t0
            eps_s = (i+1) / max(1e-6, elapsed)
            eta = (suite.episodes - (i+1)) / max(1e-6, eps_s)
            print(f"[eval][{model_name}][{suite.suite_id}][B={max_expansions}] {i+1}/{suite.episodes} eps_s={eps_s:.2f} ETA={eta/60:.1f}m")

    n = suite.episodes
    out = {
        "suite": suite.suite_id, "family": suite.family, "size": suite.size, "dynamics": suite.dynamics,
        "budget": max_expansions, "alpha": alpha, "model": model_name, "episodes": n,
        "success_rate": metrics["success"] / n, "collision_rate": metrics["collision"] / n, "timeout_rate": metrics["timeout"] / n,
        "avg_steps": metrics["steps"] / n, "avg_expansions": metrics["expansions"] / n,
        "expansions_per_step": metrics["expansions"] / max(1e-6, metrics["steps"]),
    }
    return out

def _choose_train_fn(model_name: str):
    if model_name.startswith("hrm"): return train_model_b200
    return train_model_h100

def _choose_fewshot_fn(model_name: str):
    if model_name.startswith("hrm"): return fewshot_adapt_b200
    return fewshot_adapt_h100

@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 24, # 24 hour timeout for the orchestrator
    volumes={"/data": vol},
)
def run_pipeline(only_models, max_parallel_train, max_parallel_collect, max_parallel_eval,
                 skip_collect, skip_train, skip_fewshot, eval_eps, budgets, alpha, seed_base):
    _ensure_dirs()
    
    eval_suites = build_eval_suites(eval_eps)

    dummy_obs = build_obs_vector(64, (0,0), (1,1), [], [], [])
    obs_dim = int(dummy_obs.shape[0])
    model_cfgs = build_model_configs(obs_dim)

    model_names = list(model_cfgs.keys())
    if only_models:
        model_names = [m for m in model_names if m in only_models]
    
    print("Models:", model_names)
    print("Stages:", [s.stage_id for s in STAGES])
    print("Eval suites:", [s.suite_id for s in eval_suites])
    print("Budgets:", budgets, "alpha:", alpha)
    print()

    stage_final_model_paths: Dict[str, str] = {}
    for stage in STAGES:
        print(f"\n📦 Stage {stage.stage_id}: dataset + training")

        vol.reload() 

        merged_path = f"{DATASETS_DIR}/{stage.stage_id}__merged.pt"
        merged_exists = os.path.exists(merged_path)

        if not merged_exists and not skip_collect:
            total = stage.collect_samples
            chunk_sz = max(200, total // max(1, max_parallel_collect))
            chunks = []
            chunk_id = 0
            print(f"  Collecting {total} samples in chunks of ~{chunk_sz} ...")
            while len(chunks) * chunk_sz < total:
                n_samp = min(chunk_sz, total - len(chunks)*chunk_sz)
                chunks.append(collect_data_chunk.spawn(stage.stage_id, chunk_id, n_samp, seed_base + 100_000))
                chunk_id += 1
            chunk_paths = [h.get() for h in chunks]
            
            merged_path = merge_chunks.remote(stage.stage_id, chunk_paths)
            vol.reload() 
            print(f"  ✓ dataset merged -> {merged_path}")
        else:
            if merged_exists: print(f"  ✓ using cached dataset: {merged_path}")
            else: print("  ⏭️  SKIP_COLLECT=1 set; assuming dataset exists (but merged file not found).")

        if skip_train:
            print("  ⏭️  SKIP_TRAIN=1 set; skipping training.")
        else:
            pending = []
            for m in model_names:
                train_fn = _choose_train_fn(m)
                print(f"  🚀 training {m} on {stage.stage_id} ...")
                pending.append((m, train_fn.spawn(m, stage.stage_id, merged_path, seed=0)))
                if len(pending) >= max_parallel_train:
                    for mm, hh in pending: print("   ✓", mm, hh.get()["status"])
                    pending = []
            for mm, hh in pending:
                print("   ✓", mm, hh.get()["status"])
            
        vol.reload() 
        
        for m in model_names:
            p = _final_model_path(m, stage.stage_id)
            if os.path.exists(p):
                stage_final_model_paths[m] = p

    # Few-shot adaptation
    base_stage = STAGES[-1].stage_id 
    fewshot_target = "OOD_B64_D2"
    fewshot_ks = [50, 200]
    fewshot_paths: Dict[Tuple[str,int], str] = {}

    if not skip_fewshot:
        print("\n🧪 Few-shot adaptation on", fewshot_target)
        handles = []
        for m in model_names:
            base_path = _final_model_path(m, base_stage)
            if not os.path.exists(base_path):
                print(f"  ! missing base model for {m} at {base_stage}, skipping few-shot")
                continue
            fn = _choose_fewshot_fn(m)
            for k in fewshot_ks:
                print(f"  🧪 adapt {m} K={k} ...")
                handles.append((m, k, fn.spawn(m, base_stage, fewshot_target, k, seed=0)))
        for m, k, h in handles:
            print("   ✓ few-shot", h.get()["status"])
            outp = _fewshot_model_path(m, fewshot_target, k)
            fewshot_paths[(m,k)] = outp
    else:
        print("\n⏭️  SKIP_FEWSHOT=1 set; skipping few-shot adaptation.")

    # Evaluation
    print("\n📊 Evaluation (parallel)")
    vol.reload() 

    eval_jobs = []
    for suite in eval_suites:
        for budget in budgets:
            eval_jobs.append(("baseline_static_astar", None, suite, budget))

    for m in model_names:
        p = _final_model_path(m, base_stage)
        if not os.path.exists(p):
            print(f"  ! missing model {m} at {base_stage}, skipping eval")
            continue
        for suite in eval_suites:
            for budget in budgets:
                eval_jobs.append((m, p, suite, budget))

    for (m,k), p in fewshot_paths.items():
        if not os.path.exists(p): continue
        suite = next(s for s in eval_suites if s.suite_id == fewshot_target)
        for budget in budgets:
            eval_jobs.append((f"{m}_fewshotK{k}", p, suite, budget))

    results: List[Dict[str,Any]] = []
    in_flight = []

    def flush_one():
        nonlocal in_flight, results
        mname, h = in_flight.pop(0)
        results.append(h.get())

    for (mname, path, suite, budget) in eval_jobs:
        h = evaluate_pair.spawn(suite, mname, path, alpha, budget, seed_base + 1_000_000)
        in_flight.append((mname, h))
        if len(in_flight) >= max_parallel_eval:
            flush_one()
    while in_flight:
        flush_one()

    ts = int(time.time())
    out_path = f"{RESULTS_DIR}/results__{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    vol.commit()
    print(f"\n✅ Saved results -> {out_path} ({len(results)} rows)")

    def key(r):
        return (r["model"], r["suite"], r["budget"])
    results_sorted = sorted(results, key=key)
    for r in results_sorted[: min(24, len(results_sorted))]:
        print(f"{r['model']:>18} | {r['suite']:>10} | B={r['budget']:>4} | succ={r['success_rate']:.3f} exp/step={r['expansions_per_step']:.1f}")


@app.local_entrypoint()
def main():
    print("=" * 78)
    print("TRANSFER-FIRST A* AUGMENTATION — HEURISTIC IMITATION V2")
    print("=" * 78)
    if _env_int("DETACH_HINT", 1) == 1:
        print("Tip: use `modal run --detach HRMv2/hrm-cloud/transfer_astar_heuristic_imitation_v2.py` to avoid disconnects.\n")
    
    only_models = _parse_csv_strs(_env_flag("ONLY_MODELS", ""))
    max_parallel_train = _env_int("MAX_PARALLEL_TRAIN", 2)
    max_parallel_collect = _env_int("MAX_PARALLEL_COLLECT", 8)
    max_parallel_eval = _env_int("MAX_PARALLEL_EVAL", 24)

    skip_collect = _env_int("SKIP_COLLECT", 0) == 1
    skip_train = _env_int("SKIP_TRAIN", 0) == 1
    skip_fewshot = _env_int("SKIP_FEWSHOT", 0) == 1

    eval_eps = _env_int("EVAL_EPISODES", 100)
    budgets = _parse_csv_ints(_env_flag("EVAL_BUDGETS", "200,500,2000"))
    alpha = _env_float("ALPHA", 1.0)
    seed_base = _env_int("EVAL_SEED_BASE", 0)

    run_pipeline.remote(
        only_models, max_parallel_train, max_parallel_collect, max_parallel_eval,
        skip_collect, skip_train, skip_fewshot, eval_eps, budgets, alpha, seed_base
    )