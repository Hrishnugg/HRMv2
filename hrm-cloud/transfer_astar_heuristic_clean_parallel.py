#!/usr/bin/env python3
"""
Clean transfer-learning experiment for learned A* residual heuristics.

This script rebuilds the older transfer_astar_heuristic experiments from the
 ground up with a cleaner target and observation design:

- Curriculum: map size scaling + subdued obstacle scaling.
- Base heuristic: Manhattan distance.
- Target: log1p(max(0, true_cost_to_go - Manhattan)).
- Step encoder: spatial frame CNN over walls / agent / goal / dynamics.
- Backbones: ON-LSTM vs DeepSapientHRM.
- Arms: full fine-tune vs stage-wise LoRA.
- Eval: sharded, cacheable, preemption-safe Modal orchestration.

The script is intentionally self-contained so `modal run` can mount only this
file and still work.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

try:
    if hasattr(torch.backends, "nnpack") and hasattr(torch.backends.nnpack, "set_flags"):
        torch.backends.nnpack.set_flags(False)
except Exception:
    pass

import modal

# -----------------------------------------------------------------------------
# Modal setup
# -----------------------------------------------------------------------------

APP_NAME = "transfer-astar-heuristic-clean-parallel-v1"
VOLUME_NAME = os.environ.get("VOLUME_NAME", "transfer-astar-heuristic-clean-parallel-v1-vol")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(["torch>=2.4.0", "numpy", "tqdm"])
)
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

DATA_ROOT = "/data/transfer_astar_heuristic_clean_parallel_v1"
DATASETS_DIR = f"{DATA_ROOT}/datasets"

# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


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


def _env_flag(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return default if v is None else v


def _parse_csv_strs(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_csv_ints(s: str) -> List[int]:
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def _parse_csv_floats(s: str) -> List[float]:
    return [float(p.strip()) for p in s.split(",") if p.strip()]


def _sanitize_file_component(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_", ".", "="):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "default"


def _summarize_exc(e: Exception, max_chars: int = 240) -> str:
    s = str(e).replace("\n", " | ")
    return s if len(s) <= max_chars else s[:max_chars] + "..."


def _parse_version_tuple(v: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for tok in str(v).split("."):
        digits = []
        for ch in tok:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if not digits:
            break
        parts.append(int("".join(digits)))
    return tuple(parts)


def _modal_supports_nonpreemptible() -> bool:
    try:
        return _parse_version_tuple(getattr(modal, "__version__", "0")) >= (1, 2, 3)
    except Exception:
        return False


def _read_json_safe(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _write_csv_atomic(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    tmp_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp_path, path)


def _eval_progress_count(payload: Optional[Dict[str, Any]]) -> int:
    if not payload:
        return 0
    if "completed_episodes" in payload:
        return int(payload.get("completed_episodes", 0))
    if payload.get("complete"):
        return int(payload.get("episodes", 0))
    if "metric_sums" in payload and "episodes" in payload and "ep_start" in payload:
        return int(payload.get("episodes", 0))
    return 0


def _is_complete_eval_shard(payload: Optional[Dict[str, Any]]) -> bool:
    if not payload:
        return False
    episodes = int(payload.get("episodes", 0))
    return episodes > 0 and _eval_progress_count(payload) >= episodes


def _write_eval_shard_progress(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    existing = _read_json_safe(path)
    new_done = _eval_progress_count(payload)
    if existing is not None:
        old_done = _eval_progress_count(existing)
        if _is_complete_eval_shard(existing) and not _is_complete_eval_shard(payload):
            return existing
        if old_done > new_done:
            return existing
    _write_json_atomic(path, payload)
    return payload


EVAL_FN_CPU = float(_env_float("EVAL_CPU", 2.0))
EVAL_FN_MEMORY_MB = _env_int("EVAL_MEMORY_MB", 8192)
EVAL_FN_TIMEOUT_SEC = _env_int("EVAL_TIMEOUT_SEC", 60 * 60 * 12)
EVAL_FN_NONPREEMPTIBLE = (_env_int("EVAL_NONPREEMPTIBLE", 1) == 1) and _modal_supports_nonpreemptible()

ORCH_FN_CPU = float(_env_float("ORCH_CPU", 1.0))
ORCH_FN_MEMORY_MB = _env_int("ORCH_MEMORY_MB", 4096)
ORCH_FN_TIMEOUT_SEC = _env_int("ORCH_TIMEOUT_SEC", 60 * 60 * 24)
ORCH_FN_NONPREEMPTIBLE = (_env_int("ORCH_NONPREEMPTIBLE", 1) == 1) and _modal_supports_nonpreemptible()


def _run_tag() -> str:
    raw = os.environ.get("RUN_TAG") or "clean_parallel_v1"
    return _sanitize_file_component(raw)


RUN_TAG = _run_tag()
MODEL_RUN_TAG = _sanitize_file_component(os.environ.get("MODEL_RUN_TAG") or RUN_TAG)
RUN_ROOT = f"{DATA_ROOT}/runs/{RUN_TAG}"
MODEL_RUN_ROOT = f"{DATA_ROOT}/runs/{MODEL_RUN_TAG}"
MODELS_DIR = f"{RUN_ROOT}/models"
CHECKPOINTS_DIR = f"{RUN_ROOT}/checkpoints"
RESULTS_DIR = f"{RUN_ROOT}/results"
ALPHAS_DIR = f"{RESULTS_DIR}/alphas"
SOURCE_MODELS_DIR = MODELS_DIR if RUN_ROOT == MODEL_RUN_ROOT else f"{MODEL_RUN_ROOT}/models"


def _ensure_dirs() -> None:
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ALPHAS_DIR, exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/eval_shards", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/eval_agg", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/diagnostics", exist_ok=True)


def _configure_eval_torch_threads() -> None:
    n = max(1, _env_int("EVAL_TORCH_THREADS", 1))
    try:
        torch.set_num_threads(n)
    except Exception:
        pass
    try:
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, _env_int("EVAL_TORCH_INTEROP_THREADS", 1)))
    except Exception:
        pass


def _alpha_tag(alpha: float) -> str:
    return _sanitize_file_component(f"{alpha:.4f}".rstrip("0").rstrip("."))


_MODEL_ALIASES = {
    "lstm": "onlstm",
    "on_lstm": "onlstm",
    "deephrm": "hrm",
    "deepsapienthrm": "hrm",
    "deepsapient_hrm": "hrm",
}


def _canonical_model_name(name: str) -> str:
    s = name.strip().lower().replace("-", "_").replace(" ", "")
    while "__" in s:
        s = s.replace("__", "_")
    return _MODEL_ALIASES.get(s, s)


def _parse_models_spec(spec: Optional[str]) -> Optional[List[str]]:
    if spec is None:
        return None
    s = spec.strip()
    if s == "" or s.lower() in ("all", "*"):
        return []
    return [_canonical_model_name(x) for x in _parse_csv_strs(s)]


# -----------------------------------------------------------------------------
# Curriculum
# -----------------------------------------------------------------------------


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


def build_curriculum_stages(include_stretch: bool) -> List[Stage]:
    stages = [
        Stage("A32_static", "A", 32, "static", 80, 18, 10_000, 1400, 64, 2, 18, 0, 0, 0),
        Stage("A64_static", "A", 64, "static", 160, 20, 15_000, 2200, 64, 3, 24, 0, 0, 0),
        Stage("A64_sparseDyn", "A", 64, "sparseDyn", 170, 20, 20_000, 2600, 64, 3, 30, 1, 1, 0),
    ]
    if include_stretch:
        stages.append(Stage("A64_fullDyn", "A", 64, "fullDyn", 180, 22, 25_000, 3200, 64, 3, 36, 2, 4, 4))
    return stages


def build_eval_suites(include_stretch: bool, eval_episodes: int) -> List[EvalSuite]:
    suites = [
        EvalSuite("ID_A32_static", "A", 32, "static", 80, 18, 0, 0, 0, eval_episodes),
        EvalSuite("ID_A64_static", "A", 64, "static", 160, 20, 0, 0, 0, eval_episodes),
        EvalSuite("ID_A64_sparseDyn", "A", 64, "sparseDyn", 170, 20, 1, 1, 0, eval_episodes),
        EvalSuite("OOD_B64_static", "B", 64, "static", 160, 20, 0, 0, 0, eval_episodes),
        EvalSuite("OOD_C64_static", "C", 64, "static", 160, 20, 0, 0, 0, eval_episodes),
        EvalSuite("OOD_B64_sparseDyn", "B", 64, "sparseDyn", 170, 20, 1, 1, 0, eval_episodes),
        EvalSuite("OOD_C64_sparseDyn", "C", 64, "sparseDyn", 170, 20, 1, 1, 0, eval_episodes),
    ]
    if include_stretch:
        suites.extend([
            EvalSuite("ID_A64_fullDyn", "A", 64, "fullDyn", 180, 22, 2, 4, 4, eval_episodes),
            EvalSuite("OOD_B64_fullDyn", "B", 64, "fullDyn", 180, 22, 2, 4, 4, eval_episodes),
            EvalSuite("OOD_C64_fullDyn", "C", 64, "fullDyn", 180, 22, 2, 4, 4, eval_episodes),
        ])
    return suites


# -----------------------------------------------------------------------------
# Maps and dynamics
# -----------------------------------------------------------------------------


Action = int
ACTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
WAIT_ACTION = 4
INF = 10 ** 9


@dataclass
class Gate:
    x: int
    y: int
    period: int
    open_steps: int
    phase: int
    t: int = 0


@dataclass
class Patroller:
    x: int
    y: int
    dir_idx: int


@dataclass
class Drifter:
    x: int
    y: int
    dir_idx: int


@dataclass
class Episode:
    walls: np.ndarray
    start: Tuple[int, int]
    goal: Tuple[int, int]
    gates: List[Gate]
    pats: List[Patroller]
    drifts: List[Drifter]
    max_steps: int


def manhattan(x: int, y: int, gx: int, gy: int) -> int:
    return abs(x - gx) + abs(y - gy)


def gate_closed(g: Gate) -> bool:
    return ((g.t + g.phase) % g.period) >= g.open_steps


def clone_gate(g: Gate) -> Gate:
    return Gate(g.x, g.y, g.period, g.open_steps, g.phase, g.t)


def clone_patroller(p: Patroller) -> Patroller:
    return Patroller(p.x, p.y, p.dir_idx)


def clone_drifter(d: Drifter) -> Drifter:
    return Drifter(d.x, d.y, d.dir_idx)


def generate_map_family_A(rng: random.Random, n: int) -> np.ndarray:
    walls = np.zeros((n, n), dtype=np.uint8)
    # border
    walls[0, :] = 1
    walls[-1, :] = 1
    walls[:, 0] = 1
    walls[:, -1] = 1
    # a few box rooms / bars
    num_rects = 4 if n <= 32 else 8
    for _ in range(num_rects):
        h = rng.randint(max(3, n // 10), max(4, n // 6))
        w = rng.randint(max(3, n // 10), max(4, n // 6))
        x0 = rng.randint(1, max(1, n - h - 2))
        y0 = rng.randint(1, max(1, n - w - 2))
        walls[x0 : x0 + h, y0] = 1
        walls[x0 : x0 + h, y0 + w - 1] = 1
        walls[x0, y0 : y0 + w] = 1
        walls[x0 + h - 1, y0 : y0 + w] = 1
        # doorways
        walls[x0 + rng.randint(1, h - 2), y0] = 0
        walls[x0 + rng.randint(1, h - 2), y0 + w - 1] = 0
    for _ in range(4 if n <= 32 else 8):
        x = rng.randint(2, n - 3)
        y1 = rng.randint(2, n // 2)
        y2 = rng.randint(n // 2, n - 3)
        if y1 < y2:
            walls[x, y1:y2] = 1
            walls[x, rng.randint(y1, y2 - 1)] = 0
    return walls


def generate_map_family_B(rng: random.Random, n: int) -> np.ndarray:
    walls = np.zeros((n, n), dtype=np.uint8)
    walls[0, :] = 1
    walls[-1, :] = 1
    walls[:, 0] = 1
    walls[:, -1] = 1
    step = 4 if n <= 32 else 6
    for x in range(2, n - 2, step):
        walls[x, 1 : n - 1] = 1
        for _ in range(2):
            walls[x, rng.randint(1, n - 2)] = 0
    for y in range(2, n - 2, step):
        walls[1 : n - 1, y] = 1
        for _ in range(2):
            walls[rng.randint(1, n - 2), y] = 0
    return walls


def generate_map_family_C(rng: random.Random, n: int) -> np.ndarray:
    walls = np.zeros((n, n), dtype=np.uint8)
    walls[0, :] = 1
    walls[-1, :] = 1
    walls[:, 0] = 1
    walls[:, -1] = 1
    # mostly open with sparse clutter
    clutter = int(0.06 * n * n)
    for _ in range(clutter):
        x = rng.randint(1, n - 2)
        y = rng.randint(1, n - 2)
        walls[x, y] = 1
    return walls


FAMILY_GENERATORS = {
    "A": generate_map_family_A,
    "B": generate_map_family_B,
    "C": generate_map_family_C,
}


def random_free_cell(rng: random.Random, walls: np.ndarray, banned: Optional[set] = None) -> Tuple[int, int]:
    n = walls.shape[0]
    banned = banned or set()
    while True:
        x = rng.randint(1, n - 2)
        y = rng.randint(1, n - 2)
        if walls[x, y] == 0 and (x, y) not in banned:
            return (x, y)


def spawn_obstacles(rng: random.Random, walls: np.ndarray, n_gates: int, n_pats: int, n_drifts: int,
                    reserved: Optional[set] = None) -> Tuple[List[Gate], List[Patroller], List[Drifter]]:
    reserved = set() if reserved is None else set(reserved)
    gates: List[Gate] = []
    pats: List[Patroller] = []
    drifts: List[Drifter] = []
    for _ in range(n_gates):
        x, y = random_free_cell(rng, walls, reserved)
        reserved.add((x, y))
        period = rng.choice([4, 5, 6])
        open_steps = rng.choice([1, 2, 3])
        open_steps = min(open_steps, period - 1)
        phase = rng.randint(0, period - 1)
        gates.append(Gate(x, y, period, open_steps, phase, 0))
    for _ in range(n_pats):
        x, y = random_free_cell(rng, walls, reserved)
        reserved.add((x, y))
        pats.append(Patroller(x, y, rng.randint(0, 3)))
    for _ in range(n_drifts):
        x, y = random_free_cell(rng, walls, reserved)
        reserved.add((x, y))
        drifts.append(Drifter(x, y, rng.randint(0, 3)))
    return gates, pats, drifts


def step_dynamics(walls: np.ndarray, pats: List[Patroller], drifts: List[Drifter], gates: List[Gate]) -> None:
    n = walls.shape[0]
    dirs = ACTIONS[:4]
    for g in gates:
        g.t += 1
    for p in pats:
        dx, dy = dirs[p.dir_idx]
        nx, ny = p.x + dx, p.y + dy
        if not (0 <= nx < n and 0 <= ny < n) or walls[nx, ny]:
            p.dir_idx = (p.dir_idx + 1) % 4
            dx, dy = dirs[p.dir_idx]
            nx, ny = p.x + dx, p.y + dy
            if not (0 <= nx < n and 0 <= ny < n) or walls[nx, ny]:
                nx, ny = p.x, p.y
        p.x, p.y = nx, ny
    for d in drifts:
        dx, dy = dirs[d.dir_idx]
        nx, ny = d.x + dx, d.y + dy
        if not (0 <= nx < n and 0 <= ny < n) or walls[nx, ny]:
            d.dir_idx = random.randint(0, 3)
            dx, dy = dirs[d.dir_idx]
            nx, ny = d.x + dx, d.y + dy
            if not (0 <= nx < n and 0 <= ny < n) or walls[nx, ny]:
                nx, ny = d.x, d.y
        d.x, d.y = nx, ny


def step_episode(ep: Episode, agent_xy: Tuple[int, int], action: Action) -> Tuple[Tuple[int, int], bool, Dict[str, Any]]:
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
    return (nx, ny), (collided or reached), {"collided": collided, "reached": reached}


def simulate_occupancy(walls: np.ndarray, gates0: List[Gate], pats0: List[Patroller], drifts0: List[Drifter],
                       max_steps: int) -> Dict[str, np.ndarray]:
    gates = [clone_gate(g) for g in gates0]
    pats = [clone_patroller(p) for p in pats0]
    drifts = [clone_drifter(d) for d in drifts0]
    n = walls.shape[0]
    gate_seq = np.zeros((max_steps + 1, n, n), dtype=np.uint8)
    pat_seq = np.zeros((max_steps + 1, n, n), dtype=np.uint8)
    drift_seq = np.zeros((max_steps + 1, n, n), dtype=np.uint8)
    blocked_seq = np.repeat(walls[None, :, :].astype(np.uint8), max_steps + 1, axis=0)
    for t in range(max_steps + 1):
        for g in gates:
            if gate_closed(g):
                gate_seq[t, g.x, g.y] = 1
                blocked_seq[t, g.x, g.y] = 1
        for p in pats:
            pat_seq[t, p.x, p.y] = 1
            blocked_seq[t, p.x, p.y] = 1
        for d in drifts:
            drift_seq[t, d.x, d.y] = 1
            blocked_seq[t, d.x, d.y] = 1
        if t < max_steps:
            step_dynamics(walls, pats, drifts, gates)
    return {"gate": gate_seq, "pat": pat_seq, "drift": drift_seq, "blocked": blocked_seq}


def compute_true_cost_to_goal(blocked_seq: np.ndarray, goal: Tuple[int, int], max_steps: int) -> np.ndarray:
    n = blocked_seq.shape[1]
    gx, gy = goal
    dist = np.full((max_steps + 1, n, n), INF, dtype=np.int32)
    if blocked_seq[max_steps, gx, gy] == 0:
        dist[max_steps, gx, gy] = 0
    for t in range(max_steps - 1, -1, -1):
        for x in range(n):
            for y in range(n):
                if blocked_seq[t, x, y] != 0:
                    continue
                best = dist[t + 1, x, y]
                for dx, dy in ACTIONS[:4]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n and blocked_seq[t + 1, nx, ny] == 0:
                        cand = dist[t + 1, nx, ny]
                        if cand < best:
                            best = cand
                if best < INF:
                    dist[t, x, y] = 1 + best
    return dist


@dataclass
class PlanResult:
    found: bool
    actions: List[Action]
    expansions: int
    closed: List[Tuple[int, int, int]]
    path_states: List[Tuple[int, int, int]]


def _reconstruct_path_states(parent: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]],
                             end_state: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    states: List[Tuple[int, int, int]] = []
    cur: Optional[Tuple[int, int, int]] = end_state
    while cur is not None:
        states.append(cur)
        cur = parent.get(cur)
    states.reverse()
    return states


def space_time_astar(
    start_xy: Tuple[int, int],
    goal_xy: Tuple[int, int],
    t0_abs: int,
    plan_horizon: int,
    max_expansions: int,
    occ: Dict[str, np.ndarray],
    heuristic_delta_batch_fn,
    alpha: float = 1.0,
) -> PlanResult:
    gx, gy = goal_xy
    max_t_abs = occ["blocked"].shape[0] - 1
    start_state = (start_xy[0], start_xy[1], 0)
    start_h = manhattan(start_xy[0], start_xy[1], gx, gy)
    start_delta = 0.0
    start_f = float(start_h) + alpha * start_delta
    pq: List[Tuple[float, int, Tuple[int, int, int]]] = []
    heapq.heappush(pq, (start_f, 0, start_state))
    g_cost = {start_state: 0}
    parent: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {start_state: None}
    closed: List[Tuple[int, int, int]] = []
    best_goal_state = start_state
    best_goal_score = start_h
    expansions = 0
    n = occ["blocked"].shape[1]
    while pq and expansions < max_expansions:
        _, g, s = heapq.heappop(pq)
        if g_cost.get(s, INF) != g:
            continue
        x, y, t_rel = s
        t_abs = min(t0_abs + t_rel, max_t_abs)
        closed.append(s)
        expansions += 1
        h_base = manhattan(x, y, gx, gy)
        if h_base < best_goal_score:
            best_goal_score = h_base
            best_goal_state = s
        if (x, y) == (gx, gy) and occ["blocked"][t_abs, x, y] == 0:
            best_goal_state = s
            break
        if t_rel >= plan_horizon:
            continue
        next_states: List[Tuple[int, int, int]] = []
        next_gs: List[int] = []
        for dx, dy in ACTIONS:
            nx, ny = x + dx, y + dy
            nt_rel = t_rel + 1
            nt_abs = min(t0_abs + nt_rel, max_t_abs)
            if not (0 <= nx < n and 0 <= ny < n):
                continue
            if occ["blocked"][nt_abs, nx, ny] != 0:
                continue
            ns = (nx, ny, nt_rel)
            ng = g + 1
            if ng < g_cost.get(ns, INF):
                next_states.append(ns)
                next_gs.append(ng)
        if not next_states:
            continue
        deltas = heuristic_delta_batch_fn(next_states)
        for ns, ng, delta in zip(next_states, next_gs, deltas):
            x2, y2, _ = ns
            h_base2 = manhattan(x2, y2, gx, gy)
            f = float(ng) + float(h_base2) + alpha * max(0.0, float(delta))
            if ng < g_cost.get(ns, INF):
                g_cost[ns] = ng
                parent[ns] = s
                heapq.heappush(pq, (f, ng, ns))
                if h_base2 < best_goal_score:
                    best_goal_score = h_base2
                    best_goal_state = ns
    path_states = _reconstruct_path_states(parent, best_goal_state)
    actions: List[Action] = []
    for a, b in zip(path_states[:-1], path_states[1:]):
        ax, ay, _ = a
        bx, by, _ = b
        dx, dy = bx - ax, by - ay
        try:
            actions.append(ACTIONS.index((dx, dy)))
        except ValueError:
            actions.append(WAIT_ACTION)
    if not actions:
        actions = [WAIT_ACTION]
    found = (best_goal_state[0], best_goal_state[1]) == (gx, gy) and occ["blocked"][min(t0_abs + best_goal_state[2], max_t_abs), gx, gy] == 0
    return PlanResult(found, actions, expansions, closed, path_states)


def make_episode(seed: int, family: str, n: int, max_steps: int, n_gates: int, n_pats: int, n_drifts: int) -> Episode:
    rng = random.Random(seed)
    walls = FAMILY_GENERATORS[family](rng, n)
    start = random_free_cell(rng, walls)
    goal = random_free_cell(rng, walls, banned={start})
    reserved = {start, goal}
    gates, pats, drifts = spawn_obstacles(rng, walls, n_gates, n_pats, n_drifts, reserved)
    return Episode(walls, start, goal, gates, pats, drifts, max_steps)

# -----------------------------------------------------------------------------
# Models: frame encoder, patch encoder, backbones, node head
# -----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x_f32 = x.float()
        norm = x_f32.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return ((x_f32 / norm) * self.scale * self.g).to(x_dtype)


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
        attn_out, _ = self.attn(h_norm.unsqueeze(1), h_norm.unsqueeze(1), h_norm.unsqueeze(1))
        h = res + attn_out.squeeze(1)
        candidate = h + self.ffn(self.norm2(h))
        z = torch.sigmoid(self.gate(torch.cat([candidate, state], dim=-1)))
        return z * candidate + (1.0 - z) * state


class ONLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, chunk_size: int = 5):
        super().__init__()
        assert hidden_dim % chunk_size == 0, "hidden_dim must be divisible by chunk_size"
        self.input_dim = input_dim
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
        f_hat = self.cumax(f_hat_lin)
        i_hat = 1.0 - self.cumax(i_hat_lin)
        f_hat = f_hat.repeat_interleave(self.chunk_size, dim=-1)
        i_hat = i_hat.repeat_interleave(self.chunk_size, dim=-1)
        omega = f_hat * i_hat
        f = f * omega + (f_hat - omega)
        i = i * omega + (i_hat - omega)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ONLSTMBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, chunk_size: int = 8, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cells = nn.ModuleList([
            ONLSTMCell(hidden_dim, hidden_dim, chunk_size=chunk_size) for _ in range(num_layers)
        ])

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        hs = [torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype) for _ in range(self.num_layers)]
        cs = [torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype) for _ in range(self.num_layers)]
        return hs, cs

    def step(self, x_t: torch.Tensor, state: Tuple[List[torch.Tensor], List[torch.Tensor]], t_idx: Optional[int] = None) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        hs, cs = state
        inp = self.input_proj(x_t)
        new_hs: List[torch.Tensor] = []
        new_cs: List[torch.Tensor] = []
        for li, cell in enumerate(self.cells):
            h, c = cell(inp, (hs[li], cs[li]))
            inp = h
            if self.dropout > 0 and li < self.num_layers - 1:
                inp = F.dropout(inp, p=self.dropout, training=self.training)
            new_hs.append(h)
            new_cs.append(c)
        return new_hs[-1], (new_hs, new_cs)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        b, seq, _ = x.shape
        state = self.init_state(b, x.device, x.dtype)
        ctx = torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype)
        for t in range(seq):
            ctx, state = self.step(x[:, t, :], state, t)
        return ctx


class DeepSapientHRMBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, k_step: int = 2, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_step = k_step
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.L_blocks = nn.ModuleList([GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.H_blocks = nn.ModuleList([GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        h_L = [torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype) for _ in range(len(self.L_blocks))]
        h_H = [torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype) for _ in range(len(self.H_blocks))]
        return h_L, h_H

    def step(self, x_t: torch.Tensor, state: Tuple[List[torch.Tensor], List[torch.Tensor]], t_idx: Optional[int] = None) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        h_L, h_H = state
        curr_in = self.embed(x_t)
        if t_idx is None or (t_idx % self.k_step == 0):
            h_in = h_L[-1].detach()
            new_h_H: List[torch.Tensor] = []
            for i, blk in enumerate(self.H_blocks):
                h = blk(h_in, h_H[i])
                new_h_H.append(h)
                h_in = h
            h_H = new_h_H
        l_in = curr_in + h_H[-1]
        new_h_L: List[torch.Tensor] = []
        for i, blk in enumerate(self.L_blocks):
            h = blk(l_in, h_L[i])
            new_h_L.append(h)
            l_in = h
        return new_h_L[-1], (new_h_L, h_H)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        b, seq, _ = x.shape
        state = self.init_state(b, x.device, x.dtype)
        ctx = torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype)
        for t in range(seq):
            ctx, state = self.step(x[:, t, :], state, t)
        return ctx


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 8):
        super().__init__()
        padding = kernel_size // 2
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialFrameEncoder(nn.Module):
    def __init__(self, in_ch: int, out_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            ConvGNAct(in_ch, 32),
            ConvGNAct(32, 32),
            ConvGNAct(32, 64, stride=2),
            ConvGNAct(64, 64),
            ConvGNAct(64, 128, stride=2),
            ConvGNAct(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)


class PatchEncoder(nn.Module):
    def __init__(self, in_ch: int, out_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            ConvGNAct(in_ch, 24),
            ConvGNAct(24, 48),
            ConvGNAct(48, 64),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)


class NodeHead(nn.Module):
    def __init__(self, ctx_dim: int, patch_dim: int, meta_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim + patch_dim + meta_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ctx: torch.Tensor, patch_emb: torch.Tensor, node_meta: torch.Tensor) -> torch.Tensor:
        h = torch.cat([ctx, patch_emb, node_meta], dim=-1)
        return self.net(h).squeeze(-1)


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    backbone_type: str
    frame_dim: int
    hidden_dim: int
    num_layers: int
    chunk_size: int
    k_step: int
    num_heads: int
    patch_dim: int
    node_hidden: int
    train_gpu: str


def build_model_configs() -> Dict[str, BackboneConfig]:
    return {
        "onlstm": BackboneConfig(
            name="onlstm",
            backbone_type="onlstm",
            frame_dim=256,
            hidden_dim=480,
            num_layers=2,
            chunk_size=8,
            k_step=0,
            num_heads=0,
            patch_dim=160,
            node_hidden=320,
            train_gpu="h100",
        ),
        "hrm": BackboneConfig(
            name="hrm",
            backbone_type="hrm",
            frame_dim=256,
            hidden_dim=256,
            num_layers=2,
            chunk_size=0,
            k_step=2,
            num_heads=4,
            patch_dim=160,
            node_hidden=320,
            train_gpu="b200",
        ),
    }


FRAME_CHANNELS = 8
PATCH_CHANNELS = 2
NODE_META_DIM = 6
PATCH_RADIUS = _env_int("PATCH_RADIUS", 7)
HISTORY_LEN = _env_int("HISTORY_LEN", 20)


class CleanHeuristicModel(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.frame_encoder = SpatialFrameEncoder(FRAME_CHANNELS, cfg.frame_dim)
        if cfg.backbone_type == "onlstm":
            self.backbone = ONLSTMBackbone(cfg.frame_dim, cfg.hidden_dim, cfg.num_layers, cfg.chunk_size)
        elif cfg.backbone_type == "hrm":
            self.backbone = DeepSapientHRMBackbone(cfg.frame_dim, cfg.hidden_dim, cfg.k_step, cfg.num_heads, cfg.num_layers)
        else:
            raise ValueError(f"unknown backbone_type={cfg.backbone_type}")
        self.patch_encoder = PatchEncoder(PATCH_CHANNELS, cfg.patch_dim)
        self.node_head = NodeHead(cfg.hidden_dim, cfg.patch_dim, NODE_META_DIM, cfg.node_hidden)

    def init_context_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Any:
        return self.backbone.init_state(batch_size, device, dtype)

    def step_context(self, frame_t: torch.Tensor, state: Any, t_idx: int) -> Tuple[torch.Tensor, Any]:
        frame_emb = self.frame_encoder(frame_t)
        return self.backbone.step(frame_emb, state, t_idx)

    def encode_obs_sequence(self, obs_seq: torch.Tensor) -> torch.Tensor:
        b, h, c, n, _ = obs_seq.shape
        flat = obs_seq.reshape(b * h, c, n, n)
        emb = self.frame_encoder(flat).reshape(b, h, -1)
        return self.backbone.encode_sequence(emb)

    def predict_log_delta_from_ctx(self, ctx: torch.Tensor, node_patch: torch.Tensor, node_meta: torch.Tensor) -> torch.Tensor:
        # ctx: (B,D), node_patch: (B,N,C,P,P), node_meta: (B,N,M)
        b, num_nodes = node_patch.shape[:2]
        flat_patch = node_patch.reshape(b * num_nodes, node_patch.shape[2], node_patch.shape[3], node_patch.shape[4])
        patch_emb = self.patch_encoder(flat_patch).reshape(b, num_nodes, -1)
        ctx_rep = ctx.unsqueeze(1).expand(-1, num_nodes, -1)
        flat_ctx = ctx_rep.reshape(b * num_nodes, -1)
        flat_patch_emb = patch_emb.reshape(b * num_nodes, -1)
        flat_meta = node_meta.reshape(b * num_nodes, -1)
        raw = self.node_head(flat_ctx, flat_patch_emb, flat_meta)
        # softplus keeps log-delta nonnegative while preserving gradients near 0
        log_delta = F.softplus(raw)
        return log_delta.reshape(b, num_nodes)

    def forward(self, obs_seq: torch.Tensor, node_patch: torch.Tensor, node_meta: torch.Tensor) -> torch.Tensor:
        ctx = self.encode_obs_sequence(obs_seq)
        return self.predict_log_delta_from_ctx(ctx, node_patch, node_meta)

    def predict_delta_from_ctx(self, ctx: torch.Tensor, node_patch: torch.Tensor, node_meta: torch.Tensor) -> torch.Tensor:
        log_delta = self.predict_log_delta_from_ctx(ctx, node_patch, node_meta)
        return torch.expm1(log_delta)

# -----------------------------------------------------------------------------
# LoRA helpers
# -----------------------------------------------------------------------------


class StackedWeightLoRA(nn.Module):
    def __init__(self, base_weight: torch.Tensor, rank: int, alpha: float, num_adapters: int, init_scale: float = 0.01):
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.num_adapters = int(num_adapters)
        self.out_dim = int(base_weight.shape[0])
        self.in_dim = int(base_weight.numel() // self.out_dim)
        self.weight_shape = tuple(base_weight.shape)
        scale = self.alpha / max(1, self.rank)
        self.adapter_scale = nn.Parameter(torch.full((self.num_adapters,), float(scale)), requires_grad=False)
        self.As = nn.ParameterList()
        self.Bs = nn.ParameterList()
        for _ in range(self.num_adapters):
            a = nn.Parameter(torch.randn(self.rank, self.in_dim) * init_scale)
            b = nn.Parameter(torch.zeros(self.out_dim, self.rank))
            self.As.append(a)
            self.Bs.append(b)

    def forward(self, base_w: torch.Tensor) -> torch.Tensor:
        out = base_w.reshape(self.out_dim, self.in_dim)
        for idx in range(self.num_adapters):
            out = out + self.adapter_scale[idx] * (self.Bs[idx] @ self.As[idx])
        return out.reshape(self.weight_shape)


def _iter_lora_targets(module: nn.Module, include_conv: bool, include_attn: bool):
    for name, sub in module.named_modules():
        if isinstance(sub, nn.Linear):
            yield name, sub, "weight"
        elif include_conv and isinstance(sub, nn.Conv2d):
            yield name, sub, "weight"
        elif include_attn and isinstance(sub, nn.MultiheadAttention):
            yield name, sub, "in_proj_weight"
            yield name, sub, "out_proj.weight"


def _get_nested_attr(obj: Any, dotted: str):
    cur = obj
    for tok in dotted.split("."):
        cur = getattr(cur, tok)
    return cur


def _apply_stacked_lora(module: nn.Module, rank: int, alpha: float, num_adapters: int,
                        include_conv: bool = True, include_attn: bool = True,
                        init_scale: float = 0.01) -> int:
    wrapped = 0
    for name, sub, attr in _iter_lora_targets(module, include_conv, include_attn):
        target_mod = sub
        target_attr = attr
        if attr == "out_proj.weight":
            target_mod = sub.out_proj
            target_attr = "weight"
        if parametrize.is_parametrized(target_mod, target_attr):
            continue
        base_w = getattr(target_mod, target_attr)
        if not isinstance(base_w, torch.nn.Parameter):
            continue
        parametrize.register_parametrization(
            target_mod,
            target_attr,
            StackedWeightLoRA(base_w.data, rank, alpha, num_adapters, init_scale=init_scale),
            unsafe=True,
        )
        wrapped += 1
    return wrapped


def _set_fullft_trainable(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def _set_lora_trainable(module: nn.Module, adapter_idx: int, train_bias: bool = True) -> None:
    for name, p in module.named_parameters():
        p.requires_grad = False
        if ".parametrizations." in name:
            if f"As.{adapter_idx}" in name or f"Bs.{adapter_idx}" in name:
                p.requires_grad = True
            elif "adapter_scale" in name:
                p.requires_grad = False
        elif train_bias and name.endswith(".bias"):
            p.requires_grad = True


def _trainable_params(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


# -----------------------------------------------------------------------------
# Dataset representation
# -----------------------------------------------------------------------------


COORD_CACHE: Dict[int, np.ndarray] = {}


def get_coord_channels(n: int) -> np.ndarray:
    cached = COORD_CACHE.get(n)
    if cached is not None:
        return cached
    xs = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    arr = np.stack([grid_x, grid_y], axis=0)
    COORD_CACHE[n] = arr
    return arr


def make_static_template(walls: np.ndarray, goal_xy: Tuple[int, int]) -> np.ndarray:
    n = walls.shape[0]
    gx, gy = goal_xy
    static = np.zeros((FRAME_CHANNELS, n, n), dtype=np.float32)
    static[0] = walls.astype(np.float32)
    static[2, gx, gy] = 1.0
    static[6:8] = get_coord_channels(n)
    return static


def build_step_frame(static_template: np.ndarray, agent_xy: Tuple[int, int],
                     gate_t: np.ndarray, pat_t: np.ndarray, drift_t: np.ndarray) -> np.ndarray:
    frame = static_template.copy()
    ax, ay = agent_xy
    frame[1, ax, ay] = 1.0
    frame[3] = gate_t.astype(np.float32)
    frame[4] = pat_t.astype(np.float32)
    frame[5] = drift_t.astype(np.float32)
    return frame


def extract_local_patch_2ch(walls: np.ndarray, dynamic_cur: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    p = 2 * radius + 1
    out = np.zeros((2, p, p), dtype=np.uint8)
    n = walls.shape[0]
    for i, dx in enumerate(range(-radius, radius + 1)):
        for j, dy in enumerate(range(-radius, radius + 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:
                out[0, i, j] = walls[nx, ny]
                out[1, i, j] = dynamic_cur[nx, ny]
            else:
                out[0, i, j] = 1
                out[1, i, j] = 0
    return out


def build_node_meta(x: int, y: int, gx: int, gy: int, dt_offset: int, n: int) -> np.ndarray:
    h_base = manhattan(x, y, gx, gy)
    return np.array([
        (gx - x) / max(1.0, float(n)),
        (gy - y) / max(1.0, float(n)),
        float(dt_offset) / max(1.0, float(n)),
        float(h_base) / max(1.0, float(2 * n)),
        (x / max(1.0, float(n - 1))) * 2.0 - 1.0,
        (y / max(1.0, float(n - 1))) * 2.0 - 1.0,
    ], dtype=np.float32)


def compute_target_delta_from_dist(dist_abs: np.ndarray, t_abs: int, x: int, y: int, gx: int, gy: int) -> Optional[float]:
    if t_abs < 0 or t_abs >= dist_abs.shape[0]:
        return None
    d = int(dist_abs[t_abs, x, y])
    if d >= INF:
        return None
    return float(max(0, d - manhattan(x, y, gx, gy)))


def _greedy_true_path_states(dist_abs: np.ndarray, blocked_seq: np.ndarray, start_xy: Tuple[int, int], goal_xy: Tuple[int, int],
                             t0_abs: int, horizon: int) -> List[Tuple[int, int, int]]:
    gx, gy = goal_xy
    x, y = start_xy
    n = blocked_seq.shape[1]
    max_t = blocked_seq.shape[0] - 1
    if dist_abs[t0_abs, x, y] >= INF:
        return []
    out = [(x, y, 0)]
    cur_t = t0_abs
    t_rel = 0
    while t_rel < horizon and (x, y) != (gx, gy):
        cur = dist_abs[cur_t, x, y]
        best_next: Optional[Tuple[int, int, int]] = None
        best_val = cur
        next_t = min(cur_t + 1, max_t)
        for dx, dy in ACTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and blocked_seq[next_t, nx, ny] == 0:
                cand = dist_abs[next_t, nx, ny]
                if cand < best_val:
                    best_val = cand
                    best_next = (nx, ny, t_rel + 1)
        if best_next is None:
            break
        x, y, t_rel = best_next
        cur_t = next_t
        out.append(best_next)
    return out


def _near_path_states(path_states: List[Tuple[int, int, int]], walls: np.ndarray, blocked_seq: np.ndarray, t0_abs: int) -> List[Tuple[int, int, int]]:
    n = walls.shape[0]
    max_t = blocked_seq.shape[0] - 1
    out: List[Tuple[int, int, int]] = []
    seen = set()
    for x, y, t_rel in path_states:
        t_abs = min(t0_abs + t_rel, max_t)
        if (x, y, t_rel) not in seen:
            seen.add((x, y, t_rel))
            out.append((x, y, t_rel))
        for dx, dy in ACTIONS[:4]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and blocked_seq[t_abs, nx, ny] == 0:
                s = (nx, ny, t_rel)
                if s not in seen:
                    seen.add(s)
                    out.append(s)
    return out


def _sample_uniform_valid_states(rng: random.Random, walls: np.ndarray, blocked_seq: np.ndarray, dist_abs: np.ndarray,
                                 goal_xy: Tuple[int, int], t0_abs: int, horizon: int, num_states: int) -> List[Tuple[int, int, int]]:
    n = walls.shape[0]
    max_t = blocked_seq.shape[0] - 1
    out: List[Tuple[int, int, int]] = []
    seen = set()
    gx, gy = goal_xy
    tries = 0
    while len(out) < num_states and tries < num_states * 50:
        tries += 1
        t_rel = rng.randint(0, max(0, horizon))
        t_abs = min(t0_abs + t_rel, max_t)
        x = rng.randint(0, n - 1)
        y = rng.randint(0, n - 1)
        if blocked_seq[t_abs, x, y] != 0:
            continue
        if dist_abs[t_abs, x, y] >= INF:
            continue
        s = (x, y, t_rel)
        if s in seen:
            continue
        seen.add(s)
        # lightly bias toward hard residuals if possible
        delta = compute_target_delta_from_dist(dist_abs, t_abs, x, y, gx, gy)
        if delta is None:
            continue
        if rng.random() < 0.65 or delta > 0:
            out.append(s)
    return out


def _residual_bucket(delta: float) -> int:
    if delta <= 1.0:
        return 0
    if delta <= 4.0:
        return 1
    return 2


def build_candidate_state_set(
    rng: random.Random,
    walls: np.ndarray,
    goal_xy: Tuple[int, int],
    occ: Dict[str, np.ndarray],
    dist_abs: np.ndarray,
    t_abs: int,
    current_xy: Tuple[int, int],
    plan_closed: List[Tuple[int, int, int]],
    plan_path: List[Tuple[int, int, int]],
    horizon: int,
    nodes_per_sample: int,
) -> List[Tuple[int, int, int]]:
    gx, gy = goal_xy
    blocked_seq = occ["blocked"]
    max_t = blocked_seq.shape[0] - 1
    candidates: Dict[Tuple[int, int, int], float] = {}
    closed_cap = max(nodes_per_sample, int(nodes_per_sample * 1.2))
    for s in plan_closed[:closed_cap]:
        x, y, t_rel = s
        abs_t = min(t_abs + t_rel, max_t)
        delta = compute_target_delta_from_dist(dist_abs, abs_t, x, y, gx, gy)
        if delta is not None:
            candidates[s] = delta
    path_states = _near_path_states(plan_path, walls, blocked_seq, t_abs)
    for s in path_states:
        x, y, t_rel = s
        abs_t = min(t_abs + t_rel, max_t)
        delta = compute_target_delta_from_dist(dist_abs, abs_t, x, y, gx, gy)
        if delta is not None:
            candidates[s] = delta
    for s in _sample_uniform_valid_states(rng, walls, blocked_seq, dist_abs, goal_xy, t_abs, horizon, nodes_per_sample * 2):
        x, y, t_rel = s
        abs_t = min(t_abs + t_rel, max_t)
        delta = compute_target_delta_from_dist(dist_abs, abs_t, x, y, gx, gy)
        if delta is not None:
            candidates[s] = delta
    bucketed = {0: [], 1: [], 2: []}
    for s, d in candidates.items():
        bucketed[_residual_bucket(d)].append((s, d))
    for vals in bucketed.values():
        rng.shuffle(vals)
    selected: List[Tuple[int, int, int]] = []
    quotas = [nodes_per_sample // 3, nodes_per_sample // 3, nodes_per_sample - 2 * (nodes_per_sample // 3)]
    for bi, q in enumerate(quotas):
        take = min(q, len(bucketed[bi]))
        selected.extend([s for s, _ in bucketed[bi][:take]])
        bucketed[bi] = bucketed[bi][take:]
    if len(selected) < nodes_per_sample:
        leftovers: List[Tuple[Tuple[int, int, int], float]] = bucketed[2] + bucketed[1] + bucketed[0]
        selected.extend([s for s, _ in leftovers[: nodes_per_sample - len(selected)]])
    if not selected:
        selected = [(current_xy[0], current_xy[1], 0)]
    return selected[:nodes_per_sample]


class StageEpisodeDataset(torch.utils.data.Dataset):
    def __init__(self, payload: Dict[str, Any], history_len: int = HISTORY_LEN, patch_radius: int = PATCH_RADIUS):
        self.history_len = history_len
        self.patch_radius = patch_radius
        self.episodes = payload["episodes"]
        self.samples = payload["samples"]
        self.ep_cache: List[Dict[str, Any]] = []
        for ep in self.episodes:
            walls = ep["walls"].numpy().astype(np.uint8)
            goal = tuple(int(v) for v in ep["goal"].tolist())
            gate = ep["gate"].numpy().astype(np.uint8)
            pat = ep["pat"].numpy().astype(np.uint8)
            drift = ep["drift"].numpy().astype(np.uint8)
            agent_traj = ep["agent_traj"].numpy().astype(np.int16)
            static_template = make_static_template(walls, goal)
            self.ep_cache.append({
                "walls": walls,
                "goal": goal,
                "gate": gate,
                "pat": pat,
                "drift": drift,
                "agent_traj": agent_traj,
                "static_template": static_template,
                "size": int(ep["size"]),
            })
        self.episode_index = payload["samples"]["episode_index"]
        self.t_abs = payload["samples"]["t_abs"]
        self.node_patch = payload["samples"]["node_patch"]
        self.node_meta = payload["samples"]["node_meta"]
        self.target_log_delta = payload["samples"]["target_log_delta"]
        self.target_delta = payload["samples"]["target_delta"]
        self.mask = payload["samples"]["mask"]

    def __len__(self) -> int:
        return int(self.episode_index.shape[0])

    def _build_obs_seq(self, ep_cache: Dict[str, Any], t_abs: int) -> np.ndarray:
        static_template = ep_cache["static_template"]
        gate = ep_cache["gate"]
        pat = ep_cache["pat"]
        drift = ep_cache["drift"]
        agent_traj = ep_cache["agent_traj"]
        frames: List[np.ndarray] = []
        start_idx = max(0, t_abs - self.history_len + 1)
        real_idxs = list(range(start_idx, t_abs + 1))
        if not real_idxs:
            real_idxs = [0]
        pad = self.history_len - len(real_idxs)
        full_idxs = [real_idxs[0]] * pad + real_idxs
        for idx in full_idxs:
            ax, ay = int(agent_traj[idx, 0]), int(agent_traj[idx, 1])
            frames.append(build_step_frame(static_template, (ax, ay), gate[idx], pat[idx], drift[idx]))
        return np.stack(frames, axis=0).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        epi_idx = int(self.episode_index[idx].item())
        t_abs = int(self.t_abs[idx].item())
        ep_cache = self.ep_cache[epi_idx]
        obs_seq = torch.from_numpy(self._build_obs_seq(ep_cache, t_abs))
        return {
            "obs_seq": obs_seq,
            "node_patch": self.node_patch[idx].float() / 1.0,
            "node_meta": self.node_meta[idx].float(),
            "target_log_delta": self.target_log_delta[idx].float(),
            "target_delta": self.target_delta[idx].float(),
            "mask": self.mask[idx].float(),
        }

# -----------------------------------------------------------------------------
# Selection / paths / stage helpers
# -----------------------------------------------------------------------------


EVAL_EPISODES = _env_int("EVAL_EPISODES", 100)
VALIDATION_EPISODES = _env_int("VALIDATION_EPISODES", 20)
INCLUDE_STRETCH_STAGE = (_env_int("INCLUDE_STRETCH_STAGE", 0) == 1)
STAGES: List[Stage] = build_curriculum_stages(INCLUDE_STRETCH_STAGE)
EVAL_SUITES: List[EvalSuite] = build_eval_suites(INCLUDE_STRETCH_STAGE, EVAL_EPISODES)
EVAL_SUITE_BY_ID: Dict[str, EvalSuite] = {s.suite_id: s for s in EVAL_SUITES}
STAGE_BY_ID: Dict[str, Stage] = {s.stage_id: s for s in STAGES}
STAGE_INDEX: Dict[str, int] = {s.stage_id: i for i, s in enumerate(STAGES)}


TRANSFER_MODES_ALL = ["fullft", "lora"]
MODEL_CONFIGS = build_model_configs()
MODEL_NAMES_ALL = list(MODEL_CONFIGS.keys())


TRAIN_MODELS_SPEC = _parse_models_spec(os.environ.get("TRAIN_MODELS") or os.environ.get("ONLY_MODELS"))
EVAL_MODELS_SPEC = _parse_models_spec(os.environ.get("EVAL_MODELS") or os.environ.get("ONLY_MODELS"))
TRAIN_TRANSFER_MODES_SPEC = _parse_models_spec(os.environ.get("TRAIN_TRANSFER_MODES") or "fullft,lora")
EVAL_TRANSFER_MODES_SPEC = _parse_models_spec(os.environ.get("EVAL_TRANSFER_MODES") or "fullft,lora")


if TRAIN_MODELS_SPEC is None:
    TRAIN_MODELS = ["onlstm", "hrm"]
elif TRAIN_MODELS_SPEC == []:
    TRAIN_MODELS = MODEL_NAMES_ALL
else:
    TRAIN_MODELS = TRAIN_MODELS_SPEC

if EVAL_MODELS_SPEC is None:
    EVAL_MODELS = ["onlstm", "hrm"]
elif EVAL_MODELS_SPEC == []:
    EVAL_MODELS = MODEL_NAMES_ALL
else:
    EVAL_MODELS = EVAL_MODELS_SPEC


def _normalize_modes(spec: Optional[List[str]]) -> List[str]:
    if spec is None or spec == []:
        return TRANSFER_MODES_ALL
    out = []
    for x in spec:
        x = x.strip().lower()
        if x in ("full", "fullft", "finetune", "fine_tune"):
            x = "fullft"
        if x in ("adapter", "adapters"):
            x = "lora"
        if x in TRANSFER_MODES_ALL and x not in out:
            out.append(x)
    return out or TRANSFER_MODES_ALL


TRAIN_TRANSFER_MODES = _normalize_modes(TRAIN_TRANSFER_MODES_SPEC)
EVAL_TRANSFER_MODES = _normalize_modes(EVAL_TRANSFER_MODES_SPEC)

START_STAGE_ID = os.environ.get("START_STAGE_ID", "").strip()
if START_STAGE_ID:
    if START_STAGE_ID not in STAGE_INDEX:
        raise ValueError(f"unknown START_STAGE_ID={START_STAGE_ID}; expected one of {list(STAGE_INDEX)}")
    START_STAGE_POS = STAGE_INDEX[START_STAGE_ID]
    STAGES_TO_RUN = STAGES[START_STAGE_POS:]
else:
    STAGES_TO_RUN = STAGES


EVAL_BUDGETS = _parse_csv_ints(os.environ.get("EVAL_BUDGETS", "200,500,2000"))
ALPHA_CANDIDATES = _parse_csv_floats(os.environ.get("ALPHA_CANDIDATES", "0.5,1.0,1.5,2.0"))
ALPHA_TUNE_BUDGET = _env_int("ALPHA_TUNE_BUDGET", 500)
EVAL_SHARD_SIZE = max(1, _env_int("EVAL_SHARD_SIZE", 10))
EVAL_CHECKPOINT_EVERY = max(1, _env_int("EVAL_CHECKPOINT_EVERY", 1))
EVAL_USE_GPU = (_env_int("EVAL_USE_GPU", 0) == 1)
MAX_PARALLEL_TRAIN = _env_int("MAX_PARALLEL_TRAIN", 4)
MAX_PARALLEL_COLLECT = _env_int("MAX_PARALLEL_COLLECT", 8)
MAX_PARALLEL_EVAL = _env_int("MAX_PARALLEL_EVAL", 48)
SEED_BASE = _env_int("SEED_BASE", 0)


LORA_R = _env_int("LORA_R", 8)
LORA_ALPHA = _env_float("LORA_ALPHA", 16.0)
LORA_INIT_SCALE = _env_float("LORA_INIT_SCALE", 0.01)
LORA_TRAIN_BIAS = (_env_int("LORA_TRAIN_BIAS", 1) == 1)
LORA_ON_CONV = (_env_int("LORA_ON_CONV", 1) == 1)
LORA_ON_ATTN = (_env_int("LORA_ON_ATTN", 1) == 1)


BATCH_SIZE = _env_int("BATCH_SIZE", 8)
NUM_WORKERS = _env_int("NUM_WORKERS", 0)
LR_FULL = _env_float("LR_FULL", 2e-4)
LR_LORA = _env_float("LR_LORA", 5e-4)
WEIGHT_DECAY_FULL = _env_float("WEIGHT_DECAY_FULL", 1e-3)
WEIGHT_DECAY_LORA = _env_float("WEIGHT_DECAY_LORA", 0.0)
GRAD_CLIP_NORM = _env_float("GRAD_CLIP_NORM", 1.0)
EPOCHS_DEFAULT = _env_int("EPOCHS_DEFAULT", 24)


SKIP_COLLECT = (_env_int("SKIP_COLLECT", 0) == 1)
SKIP_TRAIN = (_env_int("SKIP_TRAIN", 0) == 1)
SKIP_ALPHA_TUNE = (_env_int("SKIP_ALPHA_TUNE", 0) == 1)
SKIP_EVAL = (_env_int("SKIP_EVAL", 0) == 1)


def dataset_path(stage_id: str) -> str:
    return f"{DATASETS_DIR}/{stage_id}__merged.pt"


def dataset_chunk_path(stage_id: str, chunk_id: int) -> str:
    return f"{DATASETS_DIR}/{stage_id}__chunk_{chunk_id:04d}.pt"


def artifact_id(arm: str, model_name: str) -> str:
    return f"{arm}__{model_name}"


def model_path(arm: str, model_name: str, stage_id: str) -> str:
    return f"{MODELS_DIR}/{artifact_id(arm, model_name)}__{stage_id}.pt"


def source_model_path(arm: str, model_name: str, stage_id: str) -> str:
    p = f"{SOURCE_MODELS_DIR}/{artifact_id(arm, model_name)}__{stage_id}.pt"
    if arm == "lora" and not os.path.exists(p) and stage_id == STAGES[0].stage_id:
        # stage-1 base may only exist under fullft if the run predates the copy step
        p = f"{SOURCE_MODELS_DIR}/{artifact_id('fullft', model_name)}__{stage_id}.pt"
    return p


def checkpoint_path(arm: str, model_name: str, stage_id: str) -> str:
    return f"{CHECKPOINTS_DIR}/{artifact_id(arm, model_name)}__{stage_id}.pt"


def alpha_path(arm: str, model_name: str, stage_id: str) -> str:
    return f"{ALPHAS_DIR}/{artifact_id(arm, model_name)}__{stage_id}.json"


def eval_model_id(arm: str, model_name: str, stage_id: str) -> str:
    return _sanitize_file_component(f"{arm}__{model_name}__{stage_id}")


def eval_shard_path(model_eval_id: str, suite_id: str, budget: int, alpha: float, episodes: int, ep_start: int, ep_count: int) -> str:
    a = _alpha_tag(alpha)
    return f"{RESULTS_DIR}/eval_shards/{model_eval_id}__{suite_id}__B{budget}__a{a}__eps{episodes}__{ep_start:04d}_{ep_count:04d}.json"


def eval_agg_path(model_eval_id: str, suite_id: str, budget: int, alpha: float, episodes: int) -> str:
    a = _alpha_tag(alpha)
    return f"{RESULTS_DIR}/eval_agg/{model_eval_id}__{suite_id}__B{budget}__a{a}__eps{episodes}.json"


# -----------------------------------------------------------------------------
# Collection utilities
# -----------------------------------------------------------------------------


def _make_episode_payload(ep: Episode, occ: Dict[str, np.ndarray], goal_xy: Tuple[int, int], agent_traj: List[Tuple[int, int]]) -> Dict[str, Any]:
    return {
        "size": int(ep.walls.shape[0]),
        "max_steps": int(ep.max_steps),
        "walls": torch.from_numpy(ep.walls.astype(np.uint8)),
        "goal": torch.tensor(goal_xy, dtype=torch.int16),
        "gate": torch.from_numpy(occ["gate"].astype(np.uint8)),
        "pat": torch.from_numpy(occ["pat"].astype(np.uint8)),
        "drift": torch.from_numpy(occ["drift"].astype(np.uint8)),
        "agent_traj": torch.tensor(np.asarray(agent_traj, dtype=np.int16), dtype=torch.int16),
    }


def _build_sample_tensors_for_states(
    walls: np.ndarray,
    occ: Dict[str, np.ndarray],
    dist_abs: np.ndarray,
    goal_xy: Tuple[int, int],
    t_abs: int,
    states: List[Tuple[int, int, int]],
    nodes_per_sample: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = walls.shape[0]
    gx, gy = goal_xy
    p = 2 * PATCH_RADIUS + 1
    node_patch = np.zeros((nodes_per_sample, PATCH_CHANNELS, p, p), dtype=np.uint8)
    node_meta = np.zeros((nodes_per_sample, NODE_META_DIM), dtype=np.float16)
    target_log_delta = np.zeros((nodes_per_sample,), dtype=np.float16)
    target_delta = np.zeros((nodes_per_sample,), dtype=np.float16)
    mask = np.zeros((nodes_per_sample,), dtype=np.uint8)
    dynamic_cur = np.clip(occ["gate"][t_abs] + occ["pat"][t_abs] + occ["drift"][t_abs], 0, 1).astype(np.uint8)
    max_t = occ["blocked"].shape[0] - 1
    filled = 0
    for x, y, t_rel in states:
        if filled >= nodes_per_sample:
            break
        t_node_abs = min(t_abs + t_rel, max_t)
        delta = compute_target_delta_from_dist(dist_abs, t_node_abs, x, y, gx, gy)
        if delta is None:
            continue
        node_patch[filled] = extract_local_patch_2ch(walls, dynamic_cur, x, y, PATCH_RADIUS)
        node_meta[filled] = build_node_meta(x, y, gx, gy, t_rel, n).astype(np.float16)
        target_delta[filled] = np.float16(delta)
        target_log_delta[filled] = np.float16(np.log1p(delta))
        mask[filled] = 1
        filled += 1
    if filled == 0:
        x, y = int(goal_xy[0]), int(goal_xy[1])
        node_patch[0] = extract_local_patch_2ch(walls, dynamic_cur, x, y, PATCH_RADIUS)
        node_meta[0] = build_node_meta(x, y, gx, gy, 0, n).astype(np.float16)
        target_delta[0] = np.float16(0.0)
        target_log_delta[0] = np.float16(0.0)
        mask[0] = 1
    return node_patch, node_meta, target_log_delta, target_delta, mask

@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 6,
    volumes={"/data": vol},
)
def check_cached_dataset(stage_id: str) -> str:
    _ensure_dirs()
    p = dataset_path(stage_id)
    return p if os.path.exists(p) else ""


@app.function(
    image=image,
    cpu=6,
    memory=32768,
    timeout=60 * 60 * 8,
    volumes={"/data": vol},
)
def collect_data_chunk(stage_id: str, chunk_id: int, num_samples: int, seed_base: int) -> str:
    _ensure_dirs()
    out_path = dataset_chunk_path(stage_id, chunk_id)
    if os.path.exists(out_path):
        return out_path
    stage = STAGE_BY_ID[stage_id]
    rng = random.Random(seed_base + 10_000 * (chunk_id + 1))
    episodes: List[Optional[Dict[str, Any]]] = []
    sample_episode_index: List[int] = []
    sample_t_abs: List[int] = []
    sample_node_patch: List[np.ndarray] = []
    sample_node_meta: List[np.ndarray] = []
    sample_target_log_delta: List[np.ndarray] = []
    sample_target_delta: List[np.ndarray] = []
    sample_mask: List[np.ndarray] = []
    collected = 0
    ep_seed = seed_base + 100_000 * (chunk_id + 1)
    while collected < num_samples:
        ep = make_episode(ep_seed, stage.family, stage.size, stage.max_steps, stage.n_gates, stage.n_patrollers, stage.n_drifters)
        ep_seed += 1
        occ = simulate_occupancy(ep.walls, ep.gates, ep.pats, ep.drifts, ep.max_steps)
        dist_abs = compute_true_cost_to_goal(occ["blocked"], ep.goal, ep.max_steps)
        local_epi_idx = len(episodes)
        episodes.append(None)
        agent_xy = ep.start
        agent_traj: List[Tuple[int, int]] = []
        base_delta_fn = lambda states: [0.0 for _ in states]
        done = False
        for t_abs in range(ep.max_steps):
            agent_traj.append(agent_xy)
            if t_abs % stage.collect_every == 0 and collected < num_samples:
                plan = space_time_astar(agent_xy, ep.goal, t_abs, stage.plan_horizon, stage.oracle_max_exp, occ, base_delta_fn, alpha=1.0)
                true_path = _greedy_true_path_states(dist_abs, occ["blocked"], agent_xy, ep.goal, t_abs, stage.plan_horizon)
                if not true_path:
                    true_path = plan.path_states
                states = build_candidate_state_set(
                    rng, ep.walls, ep.goal, occ, dist_abs, t_abs, agent_xy,
                    plan.closed, true_path, stage.plan_horizon, stage.nodes_per_sample,
                )
                node_patch, node_meta, target_log_delta, target_delta, mask = _build_sample_tensors_for_states(
                    ep.walls, occ, dist_abs, ep.goal, t_abs, states, stage.nodes_per_sample
                )
                sample_episode_index.append(local_epi_idx)
                sample_t_abs.append(t_abs)
                sample_node_patch.append(node_patch)
                sample_node_meta.append(node_meta)
                sample_target_log_delta.append(target_log_delta)
                sample_target_delta.append(target_delta)
                sample_mask.append(mask)
                collected += 1
            if done:
                continue
            plan = space_time_astar(agent_xy, ep.goal, t_abs, stage.plan_horizon, min(stage.oracle_max_exp, 5000), occ, base_delta_fn, alpha=1.0)
            action = plan.actions[0] if plan.actions else WAIT_ACTION
            agent_xy, done, _info = step_episode(ep, agent_xy, action)
            if done:
                break
        if not agent_traj:
            agent_traj = [ep.start]
        episodes[local_epi_idx] = _make_episode_payload(ep, occ, ep.goal, agent_traj)
    payload = {
        "stage_id": stage_id,
        "episodes": episodes,
        "samples": {
            "episode_index": torch.tensor(sample_episode_index, dtype=torch.int16),
            "t_abs": torch.tensor(sample_t_abs, dtype=torch.int16),
            "node_patch": torch.tensor(np.stack(sample_node_patch, axis=0), dtype=torch.uint8),
            "node_meta": torch.tensor(np.stack(sample_node_meta, axis=0), dtype=torch.float16),
            "target_log_delta": torch.tensor(np.stack(sample_target_log_delta, axis=0), dtype=torch.float16),
            "target_delta": torch.tensor(np.stack(sample_target_delta, axis=0), dtype=torch.float16),
            "mask": torch.tensor(np.stack(sample_mask, axis=0), dtype=torch.uint8),
        },
    }
    torch.save(payload, out_path)
    vol.commit()
    return out_path


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=60 * 60,
    volumes={"/data": vol},
)
def merge_chunks(stage_id: str, chunk_paths: List[str]) -> str:
    _ensure_dirs()
    out_path = dataset_path(stage_id)
    if os.path.exists(out_path):
        return out_path
    episodes_all: List[Dict[str, Any]] = []
    epi_idx_all: List[torch.Tensor] = []
    t_abs_all: List[torch.Tensor] = []
    node_patch_all: List[torch.Tensor] = []
    node_meta_all: List[torch.Tensor] = []
    target_log_all: List[torch.Tensor] = []
    target_delta_all: List[torch.Tensor] = []
    mask_all: List[torch.Tensor] = []
    ep_offset = 0
    for p in chunk_paths:
        payload = torch.load(p, map_location="cpu")
        episodes = payload["episodes"]
        samples = payload["samples"]
        episodes_all.extend(episodes)
        epi_idx_all.append(samples["episode_index"].to(torch.int32) + ep_offset)
        t_abs_all.append(samples["t_abs"].to(torch.int16))
        node_patch_all.append(samples["node_patch"].to(torch.uint8))
        node_meta_all.append(samples["node_meta"].to(torch.float16))
        target_log_all.append(samples["target_log_delta"].to(torch.float16))
        target_delta_all.append(samples["target_delta"].to(torch.float16))
        mask_all.append(samples["mask"].to(torch.uint8))
        ep_offset += len(episodes)
    merged = {
        "stage_id": stage_id,
        "episodes": episodes_all,
        "samples": {
            "episode_index": torch.cat(epi_idx_all, dim=0).to(torch.int16),
            "t_abs": torch.cat(t_abs_all, dim=0).to(torch.int16),
            "node_patch": torch.cat(node_patch_all, dim=0).to(torch.uint8),
            "node_meta": torch.cat(node_meta_all, dim=0).to(torch.float16),
            "target_log_delta": torch.cat(target_log_all, dim=0).to(torch.float16),
            "target_delta": torch.cat(target_delta_all, dim=0).to(torch.float16),
            "mask": torch.cat(mask_all, dim=0).to(torch.uint8),
        },
    }
    torch.save(merged, out_path)
    vol.commit()
    return out_path

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def load_dataset(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


@dataclass
class SavedModelPayload:
    cfg: Dict[str, Any]
    arm: str
    model_name: str
    stage_id: str
    model_state: Dict[str, Any]
    metrics: Dict[str, Any]


def _save_model_artifact(path: str, cfg: BackboneConfig, arm: str, model_name: str, stage_id: str,
                         model: nn.Module, metrics: Dict[str, Any]) -> None:
    payload = {
        "cfg": asdict(cfg),
        "arm": arm,
        "model_name": model_name,
        "stage_id": stage_id,
        "model_state": model.state_dict(),
        "metrics": metrics,
    }
    torch.save(payload, path)


def _load_model_artifact(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def _copy_model_artifact(src_path: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)


def _resume_training_checkpoint(ckpt_path: str, model: nn.Module, opt: torch.optim.Optimizer) -> Tuple[int, float]:
    if not os.path.exists(ckpt_path):
        return 0, 1e9
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    opt.load_state_dict(ckpt["opt"])
    return int(ckpt["epoch"]) + 1, float(ckpt.get("best_loss", 1e9))


def _init_model_for_stage(model_name: str, arm: str, stage_id: str, device: str) -> Tuple[CleanHeuristicModel, BackboneConfig, int]:
    cfg = MODEL_CONFIGS[model_name]
    model = CleanHeuristicModel(cfg).to(device)
    stage_idx = STAGE_INDEX[stage_id]
    if arm == "lora" and stage_idx > 0:
        _apply_stacked_lora(model, LORA_R, LORA_ALPHA, stage_idx, include_conv=LORA_ON_CONV, include_attn=LORA_ON_ATTN, init_scale=LORA_INIT_SCALE)
    return model, cfg, stage_idx


def _load_previous_stage_weights(model: nn.Module, arm: str, model_name: str, stage_id: str) -> None:
    stage_idx = STAGE_INDEX[stage_id]
    if stage_idx == 0:
        return
    prev_stage = STAGES[stage_idx - 1].stage_id
    prev_path = source_model_path(arm, model_name, prev_stage)
    if not os.path.exists(prev_path) and arm == "lora":
        prev_path = source_model_path("fullft", model_name, prev_stage)
    if not os.path.exists(prev_path):
        raise FileNotFoundError(f"previous model missing: {prev_path}")
    payload = _load_model_artifact(prev_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    if missing or unexpected:
        print(f"[train][{arm}][{model_name}][{stage_id}] init strict=False; missing={len(missing)} unexpected={len(unexpected)}")


def _train_model_impl(model_name: str, arm: str, stage_id: str, dataset_pt: str, device: str, seed: int) -> Dict[str, Any]:
    vol.reload()
    _ensure_dirs()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data = load_dataset(dataset_pt)
    ds = StageEpisodeDataset(data)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    model, cfg, stage_idx = _init_model_for_stage(model_name, arm, stage_id, device)
    if stage_idx > 0:
        _load_previous_stage_weights(model, arm, model_name, stage_id)

    # stage-1 lora arm is just the shared base
    if arm == "lora" and stage_idx > 0:
        _set_lora_trainable(model, adapter_idx=stage_idx - 1, train_bias=LORA_TRAIN_BIAS)
        lr = LR_LORA
        wd = WEIGHT_DECAY_LORA
    else:
        _set_fullft_trainable(model)
        lr = LR_FULL
        wd = WEIGHT_DECAY_FULL

    params = _trainable_params(model)
    if not params:
        raise RuntimeError(f"no trainable parameters for {arm}/{model_name}/{stage_id}")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    ckpt_path = checkpoint_path(arm, model_name, stage_id)
    final_path = model_path(arm, model_name, stage_id)
    epochs = max(1, STAGE_BY_ID[stage_id].train_epochs or EPOCHS_DEFAULT)
    start_epoch, best_loss = _resume_training_checkpoint(ckpt_path, model, opt)
    model.train()
    scaler = None
    use_amp = (device == "cuda") and (_env_int("USE_AMP", 1) == 1)

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        steps = 0
        for batch in loader:
            obs_seq = batch["obs_seq"].to(device)
            node_patch = batch["node_patch"].to(device)
            node_meta = batch["node_meta"].to(device)
            target_log_delta = batch["target_log_delta"].to(device)
            target_delta = batch["target_delta"].to(device)
            mask = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_log = model(obs_seq, node_patch, node_meta)
                    base_loss = F.smooth_l1_loss(pred_log, target_log_delta, reduction="none")
                    weights = (1.0 + torch.clamp(target_delta / 6.0, max=3.0)) * mask
                    loss = (base_loss * weights).sum() / torch.clamp(mask.sum(), min=1.0)
            else:
                pred_log = model(obs_seq, node_patch, node_meta)
                base_loss = F.smooth_l1_loss(pred_log, target_log_delta, reduction="none")
                weights = (1.0 + torch.clamp(target_delta / 6.0, max=3.0)) * mask
                loss = (base_loss * weights).sum() / torch.clamp(mask.sum(), min=1.0)
            loss.backward()
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1
        epoch_loss /= max(1, steps)
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
            _save_model_artifact(final_path, cfg, arm, model_name, stage_id, model, {
                "best_loss": best_loss,
                "epoch": epoch,
                "arm": arm,
                "stage_id": stage_id,
                "model_name": model_name,
            })
            vol.commit()
        torch.save({
            "epoch": epoch,
            "best_loss": best_loss,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
        }, ckpt_path)
        vol.commit()
        print(f"[train][{arm}][{model_name}][{stage_id}] epoch {epoch+1}/{epochs} loss={epoch_loss:.6f} best={best_loss:.6f}")

    return {
        "ok": True,
        "model_name": model_name,
        "arm": arm,
        "stage_id": stage_id,
        "path": final_path,
        "best_loss": best_loss,
    }


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 12,
    volumes={"/data": vol},
)
def train_model_h100(model_name: str, arm: str, stage_id: str, dataset_pt: str, seed: int = 0) -> Dict[str, Any]:
    return _train_model_impl(model_name, arm, stage_id, dataset_pt, device="cuda", seed=seed)


@app.function(
    image=image,
    gpu="B200",
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 12,
    volumes={"/data": vol},
)
def train_model_b200(model_name: str, arm: str, stage_id: str, dataset_pt: str, seed: int = 0) -> Dict[str, Any]:
    return _train_model_impl(model_name, arm, stage_id, dataset_pt, device="cuda", seed=seed)

# -----------------------------------------------------------------------------
# Eval utilities and diagnostics
# -----------------------------------------------------------------------------


def _load_model_for_eval(model_path_str: str, device: str) -> CleanHeuristicModel:
    payload = _load_model_artifact(model_path_str, map_location="cpu")
    cfg = BackboneConfig(**payload["cfg"])
    model = CleanHeuristicModel(cfg)
    arm = payload.get("arm", "fullft")
    stage_id = payload.get("stage_id", STAGES[0].stage_id)
    stage_idx = STAGE_INDEX.get(stage_id, 0)
    if arm == "lora" and stage_idx > 0:
        _apply_stacked_lora(model, LORA_R, LORA_ALPHA, stage_idx, include_conv=LORA_ON_CONV, include_attn=LORA_ON_ATTN, init_scale=LORA_INIT_SCALE)
    model.load_state_dict(payload["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model


def _new_diag_accumulator() -> Dict[str, Any]:
    return {
        "pred_sum": 0.0,
        "pred_sq_sum": 0.0,
        "pred_max": 0.0,
        "pred_pos_count": 0,
        "pred_count": 0,
        "target_sum": 0.0,
        "target_sq_sum": 0.0,
        "target_count": 0,
        "corr_sum_xy": 0.0,
        "corr_sum_x": 0.0,
        "corr_sum_y": 0.0,
        "corr_sum_x2": 0.0,
        "corr_sum_y2": 0.0,
        "corr_count": 0,
        "ordering_sets": 0,
        "ordering_changed_sets": 0,
        "rank_disp_sum": 0.0,
        "bucket_counts": {"small": 0, "medium": 0, "large": 0},
        "bucket_pred_sum": {"small": 0.0, "medium": 0.0, "large": 0.0},
        "high_residual_seen": False,
    }


def _merge_diag(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k in [
        "pred_sum", "pred_sq_sum", "pred_max", "pred_pos_count", "pred_count",
        "target_sum", "target_sq_sum", "target_count", "corr_sum_xy", "corr_sum_x",
        "corr_sum_y", "corr_sum_x2", "corr_sum_y2", "corr_count",
        "ordering_sets", "ordering_changed_sets", "rank_disp_sum",
    ]:
        if k == "pred_max":
            dst[k] = max(float(dst[k]), float(src[k]))
        else:
            dst[k] = dst[k] + src[k]
    for b in ["small", "medium", "large"]:
        dst["bucket_counts"][b] += int(src["bucket_counts"][b])
        dst["bucket_pred_sum"][b] += float(src["bucket_pred_sum"][b])
    dst["high_residual_seen"] = bool(dst["high_residual_seen"] or src["high_residual_seen"])


def _finalize_diag(diag: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(diag)
    pred_count = max(1, int(diag["pred_count"]))
    target_count = max(1, int(diag["target_count"]))
    out["pred_mean"] = float(diag["pred_sum"]) / pred_count
    out["pred_std"] = max(0.0, float(diag["pred_sq_sum"]) / pred_count - out["pred_mean"] ** 2) ** 0.5
    out["pred_positive_frac"] = float(diag["pred_pos_count"]) / pred_count
    out["target_mean"] = float(diag["target_sum"]) / target_count
    out["target_std"] = max(0.0, float(diag["target_sq_sum"]) / target_count - out["target_mean"] ** 2) ** 0.5
    denom = math.sqrt(max(1e-12, float(diag["corr_count"]) * diag["corr_sum_x2"] - diag["corr_sum_x"] ** 2)) * math.sqrt(max(1e-12, float(diag["corr_count"]) * diag["corr_sum_y2"] - diag["corr_sum_y"] ** 2))
    if denom > 0 and diag["corr_count"] > 1:
        out["pred_target_corr"] = (float(diag["corr_count"]) * diag["corr_sum_xy"] - diag["corr_sum_x"] * diag["corr_sum_y"]) / denom
    else:
        out["pred_target_corr"] = 0.0
    out["ordering_change_frac"] = float(diag["ordering_changed_sets"]) / max(1, int(diag["ordering_sets"]))
    out["avg_rank_displacement"] = float(diag["rank_disp_sum"]) / max(1, int(diag["ordering_sets"]))
    out["bucket_pred_mean"] = {
        b: float(diag["bucket_pred_sum"][b]) / max(1, int(diag["bucket_counts"][b])) for b in ["small", "medium", "large"]
    }
    return out


def _diagnostics_update(diag: Dict[str, Any], target_deltas: List[float], pred_deltas: List[float], alpha: float, h_bases: List[int]) -> None:
    n = len(pred_deltas)
    if n == 0:
        return
    for t, p in zip(target_deltas, pred_deltas):
        diag["pred_sum"] += float(p)
        diag["pred_sq_sum"] += float(p) ** 2
        diag["pred_max"] = max(float(diag["pred_max"]), float(p))
        diag["pred_pos_count"] += int(p > 1e-6)
        diag["pred_count"] += 1
        diag["target_sum"] += float(t)
        diag["target_sq_sum"] += float(t) ** 2
        diag["target_count"] += 1
        diag["corr_sum_xy"] += float(p) * float(t)
        diag["corr_sum_x"] += float(p)
        diag["corr_sum_y"] += float(t)
        diag["corr_sum_x2"] += float(p) ** 2
        diag["corr_sum_y2"] += float(t) ** 2
        diag["corr_count"] += 1
        if t > 4.0:
            diag["high_residual_seen"] = True
        if t <= 1.0:
            b = "small"
        elif t <= 4.0:
            b = "medium"
        else:
            b = "large"
        diag["bucket_counts"][b] += 1
        diag["bucket_pred_sum"][b] += float(p)
    if n > 1:
        base_order = sorted(range(n), key=lambda i: h_bases[i])
        learned_order = sorted(range(n), key=lambda i: h_bases[i] + alpha * max(0.0, pred_deltas[i]))
        diag["ordering_sets"] += 1
        if base_order != learned_order:
            diag["ordering_changed_sets"] += 1
        rank_base = {idx: r for r, idx in enumerate(base_order)}
        rank_learned = {idx: r for r, idx in enumerate(learned_order)}
        disp = sum(abs(rank_base[i] - rank_learned[i]) for i in range(n)) / float(n)
        diag["rank_disp_sum"] += disp


def _validation_suite_ids(stage_id: str) -> List[str]:
    if stage_id == "A32_static":
        return ["ID_A32_static"]
    if stage_id == "A64_static":
        return ["ID_A64_static", "OOD_B64_static", "OOD_C64_static"]
    if stage_id == "A64_sparseDyn":
        return ["ID_A64_sparseDyn", "OOD_B64_sparseDyn", "OOD_C64_sparseDyn"]
    if stage_id == "A64_fullDyn":
        return ["ID_A64_fullDyn", "OOD_B64_fullDyn", "OOD_C64_fullDyn"]
    return []


def _load_saved_alpha(arm: str, model_name: str, stage_id: str) -> Optional[float]:
    p = alpha_path(arm, model_name, stage_id)
    data = _read_json_safe(p)
    if not data:
        return None
    try:
        return float(data["best_alpha"])
    except Exception:
        return None


def _save_alpha_choice(arm: str, model_name: str, stage_id: str, best_alpha: float, scored: Dict[str, float]) -> None:
    _write_json_atomic(alpha_path(arm, model_name, stage_id), {
        "arm": arm,
        "model_name": model_name,
        "stage_id": stage_id,
        "best_alpha": float(best_alpha),
        "scores": scored,
    })


def _model_display_name(arm: str, model_name: str) -> str:
    return f"{arm}__{model_name}"


def run_policy_episode(suite: EvalSuite, seed: int, model: Optional[CleanHeuristicModel], alpha: float,
                       max_expansions: int, device: str) -> Dict[str, Any]:
    ep = make_episode(seed, suite.family, suite.size, suite.max_steps, suite.n_gates, suite.n_patrollers, suite.n_drifters)
    occ = simulate_occupancy(ep.walls, ep.gates, ep.pats, ep.drifts, ep.max_steps)
    dist_abs = compute_true_cost_to_goal(occ["blocked"], ep.goal, ep.max_steps)
    static_template = make_static_template(ep.walls, ep.goal)
    agent_xy = ep.start
    done = False
    last_info = {"reached": False, "collided": False}
    total_expansions = 0
    diag_acc = _new_diag_accumulator()
    if model is not None:
        state = model.init_context_state(1, torch.device(device), torch.float32)
    else:
        state = None

    for t_abs in range(ep.max_steps):
        frame = build_step_frame(static_template, agent_xy, occ["gate"][t_abs], occ["pat"][t_abs], occ["drift"][t_abs])
        if model is not None:
            frame_t = torch.from_numpy(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                ctx, state = model.step_context(frame_t, state, t_abs)
        else:
            ctx = None

        dynamic_cur = np.clip(occ["gate"][t_abs] + occ["pat"][t_abs] + occ["drift"][t_abs], 0, 1).astype(np.uint8)
        gx, gy = ep.goal

        def heuristic_delta_batch_fn(states: List[Tuple[int, int, int]]) -> List[float]:
            h_bases = [manhattan(x, y, gx, gy) for x, y, _ in states]
            target_deltas: List[float] = []
            for x, y, t_rel in states:
                tgt = compute_target_delta_from_dist(dist_abs, min(t_abs + t_rel, ep.max_steps), x, y, gx, gy)
                target_deltas.append(0.0 if tgt is None else float(tgt))
            if model is None:
                pred = [0.0 for _ in states]
                _diagnostics_update(diag_acc, target_deltas, pred, alpha, h_bases)
                return pred
            p = 2 * PATCH_RADIUS + 1
            patches = np.zeros((1, len(states), PATCH_CHANNELS, p, p), dtype=np.float32)
            metas = np.zeros((1, len(states), NODE_META_DIM), dtype=np.float32)
            for i, (x, y, t_rel) in enumerate(states):
                patches[0, i] = extract_local_patch_2ch(ep.walls, dynamic_cur, x, y, PATCH_RADIUS).astype(np.float32)
                metas[0, i] = build_node_meta(x, y, gx, gy, t_rel, ep.walls.shape[0])
            patch_t = torch.from_numpy(patches).to(device)
            meta_t = torch.from_numpy(metas).to(device)
            with torch.no_grad():
                pred_t = model.predict_delta_from_ctx(ctx, patch_t, meta_t)[0].detach().float().cpu().numpy().tolist()
            pred = [float(v) for v in pred_t]
            _diagnostics_update(diag_acc, target_deltas, pred, alpha, h_bases)
            return pred

        plan = space_time_astar(agent_xy, ep.goal, t_abs, suite.plan_horizon, max_expansions, occ, heuristic_delta_batch_fn, alpha=alpha)
        total_expansions += int(plan.expansions)
        action = plan.actions[0] if plan.actions else WAIT_ACTION
        agent_xy, done, last_info = step_episode(ep, agent_xy, action)
        if done:
            steps = t_abs + 1
            break
    else:
        steps = ep.max_steps

    success = bool(last_info.get("reached", False))
    collided = bool(last_info.get("collided", False))
    timeout = (not success) and (not collided)
    out = {
        "success": success,
        "timeout": timeout,
        "collided": collided,
        "steps": int(steps),
        "expansions": int(total_expansions),
        "expansions_per_step": float(total_expansions) / max(1, int(steps)),
        "high_residual_seen": bool(diag_acc["high_residual_seen"]),
        "diag": diag_acc,
    }
    return out


def _blank_metric_sums() -> Dict[str, Any]:
    return {
        "successes": 0,
        "timeouts": 0,
        "collisions": 0,
        "steps": 0,
        "expansions": 0,
        "high_residual_eps": 0,
        "high_residual_successes": 0,
        "diag": _new_diag_accumulator(),
    }


def _merge_metric_sums(dst: Dict[str, Any], ep_res: Dict[str, Any]) -> None:
    dst["successes"] += int(ep_res["success"])
    dst["timeouts"] += int(ep_res["timeout"])
    dst["collisions"] += int(ep_res["collided"])
    dst["steps"] += int(ep_res["steps"])
    dst["expansions"] += int(ep_res["expansions"])
    if ep_res.get("high_residual_seen", False):
        dst["high_residual_eps"] += 1
        dst["high_residual_successes"] += int(ep_res["success"])
    _merge_diag(dst["diag"], ep_res["diag"])


def _evaluate_pair_chunk_impl(model_eval_id: str, display_name: str, model_path_str: str, suite_id: str,
                              alpha: float, budget: int, total_episodes: int, seed_base: int, ep_start: int, ep_count: int,
                              device: str) -> str:
    vol.reload()
    _ensure_dirs()
    _configure_eval_torch_threads()
    suite = EVAL_SUITE_BY_ID[suite_id]
    out_path = eval_shard_path(model_eval_id, suite_id, budget, alpha, total_episodes, ep_start, ep_count)
    existing = _read_json_safe(out_path)
    if _is_complete_eval_shard(existing):
        return out_path

    if model_path_str and model_path_str != "":
        model = _load_model_for_eval(model_path_str, device)
    else:
        model = None

    metric_sums = _blank_metric_sums()
    completed = 0
    if existing is not None:
        completed = int(existing.get("completed_episodes", 0))
        if "metric_sums" in existing:
            metric_sums = existing["metric_sums"]
            if "diag" not in metric_sums:
                metric_sums["diag"] = _new_diag_accumulator()
        print(f"[eval][{display_name}][{suite_id}][B={budget}] resuming shard eps {ep_start}-{ep_start + ep_count - 1} from {completed}/{ep_count} completed on device={device}")

    started_at = time.time()
    for local_idx in range(completed, ep_count):
        ep_idx = ep_start + local_idx
        seed = seed_base + ep_idx
        ep_res = run_policy_episode(suite, seed, model, alpha, budget, device)
        _merge_metric_sums(metric_sums, ep_res)
        completed = local_idx + 1
        if (completed % EVAL_CHECKPOINT_EVERY == 0) or (completed == ep_count):
            payload = {
                "model_eval_id": model_eval_id,
                "display_name": display_name,
                "suite_id": suite_id,
                "budget": budget,
                "alpha": alpha,
                "episodes": ep_count,
                "ep_start": ep_start,
                "completed_episodes": completed,
                "metric_sums": metric_sums,
                "complete": completed >= ep_count,
            }
            _write_eval_shard_progress(out_path, payload)
            vol.commit()
        if completed >= 2:
            elapsed = max(1e-6, time.time() - started_at)
            eps_s = completed / elapsed
            eta = (ep_count - completed) / max(1e-9, eps_s)
            print(f"[eval][{display_name}][{suite_id}][B={budget}] episodes {ep_start + completed}/{total_episodes} of {total_episodes} eps_s={eps_s:.2f} ETA={eta/60:.1f}m device={device}")
    return out_path


@app.function(
    image=image,
    cpu=EVAL_FN_CPU,
    memory=EVAL_FN_MEMORY_MB,
    timeout=EVAL_FN_TIMEOUT_SEC,
    nonpreemptible=EVAL_FN_NONPREEMPTIBLE,
    volumes={"/data": vol},
)
def evaluate_pair_chunk(model_eval_id: str, display_name: str, model_path_str: str, suite_id: str,
                        alpha: float, budget: int, total_episodes: int, seed_base: int, ep_start: int, ep_count: int) -> str:
    return _evaluate_pair_chunk_impl(model_eval_id, display_name, model_path_str, suite_id, alpha, budget, total_episodes, seed_base, ep_start, ep_count, device="cpu")


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=32768,
    timeout=EVAL_FN_TIMEOUT_SEC,
    volumes={"/data": vol},
)
def evaluate_pair_chunk_h100(model_eval_id: str, display_name: str, model_path_str: str, suite_id: str,
                             alpha: float, budget: int, total_episodes: int, seed_base: int, ep_start: int, ep_count: int) -> str:
    return _evaluate_pair_chunk_impl(model_eval_id, display_name, model_path_str, suite_id, alpha, budget, total_episodes, seed_base, ep_start, ep_count, device="cuda")


@app.function(
    image=image,
    gpu="B200",
    cpu=4,
    memory=32768,
    timeout=EVAL_FN_TIMEOUT_SEC,
    volumes={"/data": vol},
)
def evaluate_pair_chunk_b200(model_eval_id: str, display_name: str, model_path_str: str, suite_id: str,
                             alpha: float, budget: int, total_episodes: int, seed_base: int, ep_start: int, ep_count: int) -> str:
    return _evaluate_pair_chunk_impl(model_eval_id, display_name, model_path_str, suite_id, alpha, budget, total_episodes, seed_base, ep_start, ep_count, device="cuda")


def _merge_metric_sums_payload(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    dst["successes"] += int(src.get("successes", 0))
    dst["timeouts"] += int(src.get("timeouts", 0))
    dst["collisions"] += int(src.get("collisions", 0))
    dst["steps"] += int(src.get("steps", 0))
    dst["expansions"] += int(src.get("expansions", 0))
    dst["high_residual_eps"] += int(src.get("high_residual_eps", 0))
    dst["high_residual_successes"] += int(src.get("high_residual_successes", 0))
    _merge_diag(dst["diag"], src.get("diag", _new_diag_accumulator()))


def _aggregate_eval_job(job: Dict[str, Any], shard_paths: List[str]) -> Dict[str, Any]:
    metric_sums = _blank_metric_sums()
    completed = 0
    for p in shard_paths:
        payload = _read_json_safe(p)
        if payload is None:
            raise RuntimeError(f"missing shard payload: {p}")
        completed += int(payload.get("completed_episodes", 0))
        _merge_metric_sums_payload(metric_sums, payload.get("metric_sums", {}))
    episodes = int(job["episodes"])
    diag = _finalize_diag(metric_sums["diag"])
    row = {
        "model": job["display_name"],
        "arm": job.get("arm", "baseline"),
        "backbone": job.get("model_name", "baseline"),
        "stage": job.get("stage_id", "baseline"),
        "suite": job["suite_id"],
        "budget": int(job["budget"]),
        "alpha": float(job["alpha"]),
        "episodes": episodes,
        "completed": completed,
        "success_rate": float(metric_sums["successes"]) / max(1, episodes),
        "timeout_rate": float(metric_sums["timeouts"]) / max(1, episodes),
        "collision_rate": float(metric_sums["collisions"]) / max(1, episodes),
        "avg_steps": float(metric_sums["steps"]) / max(1, episodes),
        "avg_expansions": float(metric_sums["expansions"]) / max(1, episodes),
        "avg_expansions_per_step": float(metric_sums["expansions"]) / max(1, metric_sums["steps"]),
        "success_rate_high_residual": float(metric_sums["high_residual_successes"]) / max(1, metric_sums["high_residual_eps"]),
        "high_residual_episode_frac": float(metric_sums["high_residual_eps"]) / max(1, episodes),
        "pred_mean": diag["pred_mean"],
        "pred_std": diag["pred_std"],
        "pred_max": float(metric_sums["diag"]["pred_max"]),
        "pred_positive_frac": diag["pred_positive_frac"],
        "target_mean": diag["target_mean"],
        "pred_target_corr": diag["pred_target_corr"],
        "ordering_change_frac": diag["ordering_change_frac"],
        "avg_rank_displacement": diag["avg_rank_displacement"],
        "bucket_pred_mean_small": diag["bucket_pred_mean"]["small"],
        "bucket_pred_mean_medium": diag["bucket_pred_mean"]["medium"],
        "bucket_pred_mean_large": diag["bucket_pred_mean"]["large"],
        "bucket_count_small": int(metric_sums["diag"]["bucket_counts"]["small"]),
        "bucket_count_medium": int(metric_sums["diag"]["bucket_counts"]["medium"]),
        "bucket_count_large": int(metric_sums["diag"]["bucket_counts"]["large"]),
    }
    agg_payload = {"row": row, "metric_sums": metric_sums, "diag": diag, "complete": True}
    _write_json_atomic(eval_agg_path(job["model_eval_id"], job["suite_id"], job["budget"], job["alpha"], job["episodes"]), agg_payload)
    vol.commit()
    return row


def _choose_eval_fn(job: Dict[str, Any]):
    if not EVAL_USE_GPU or not job.get("model_path"):
        return evaluate_pair_chunk
    model_name = job.get("model_name", "")
    if model_name == "hrm":
        return evaluate_pair_chunk_b200
    return evaluate_pair_chunk_h100


def _run_eval_jobs(eval_jobs: List[Dict[str, Any]], max_parallel_eval: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pending_jobs: List[Dict[str, Any]] = []
    job_state: Dict[Tuple[str, str, int, float, int], Dict[str, Any]] = {}

    for job in eval_jobs:
        agg_path = eval_agg_path(job["model_eval_id"], job["suite_id"], job["budget"], job["alpha"], job["episodes"])
        agg = _read_json_safe(agg_path)
        if agg and agg.get("complete") and "row" in agg:
            rows.append(agg["row"])
            continue
        key = (job["model_eval_id"], job["suite_id"], int(job["budget"]), float(job["alpha"]), int(job["episodes"]))
        total_eps = int(job["episodes"])
        shard_paths = []
        to_spawn = []
        for ep_start in range(0, total_eps, EVAL_SHARD_SIZE):
            ep_count = min(EVAL_SHARD_SIZE, total_eps - ep_start)
            p = eval_shard_path(job["model_eval_id"], job["suite_id"], job["budget"], job["alpha"], total_eps, ep_start, ep_count)
            shard_paths.append(p)
            if not _is_complete_eval_shard(_read_json_safe(p)):
                to_spawn.append((ep_start, ep_count, p))
        job_state[key] = {"job": job, "shard_paths": shard_paths, "remaining": len(to_spawn)}
        for ep_start, ep_count, p in to_spawn:
            pending_jobs.append({
                "key": key,
                "job": job,
                "ep_start": ep_start,
                "ep_count": ep_count,
                "path": p,
            })
        if not to_spawn:
            rows.append(_aggregate_eval_job(job, shard_paths))

    inflight: List[Tuple[Any, Dict[str, Any]]] = []

    def spawn_one(item: Dict[str, Any]) -> None:
        fn = _choose_eval_fn(item["job"])
        h = fn.spawn(
            item["job"]["model_eval_id"],
            item["job"]["display_name"],
            item["job"].get("model_path") or "",
            item["job"]["suite_id"],
            float(item["job"]["alpha"]),
            int(item["job"]["budget"]),
            int(item["job"]["episodes"]),
            int(item["job"]["seed_base"]),
            int(item["ep_start"]),
            int(item["ep_count"]),
        )
        inflight.append((h, item))

    def finalize_key_if_done(key: Tuple[str, str, int, float, int]) -> Optional[Dict[str, Any]]:
        st = job_state[key]
        if st["remaining"] > 0:
            return None
        _refresh_volume(f"aggregate eval {key}")
        row = _aggregate_eval_job(st["job"], st["shard_paths"])
        return row

    def flush_one(block: bool = False) -> Optional[Dict[str, Any]]:
        if not inflight:
            return None
        while True:
            for idx, (h, item) in enumerate(list(inflight)):
                try:
                    _ = h.get(timeout=0)
                except modal.exception.TimeoutError:
                    continue
                except TimeoutError:
                    continue
                inflight.pop(idx)
                key = item["key"]
                job_state[key]["remaining"] -= 1
                row = finalize_key_if_done(key)
                return row
            if not block:
                return None
            time.sleep(0.5)

    while pending_jobs or inflight:
        while pending_jobs and len(inflight) < max_parallel_eval:
            spawn_one(pending_jobs.pop(0))
        row = flush_one(block=bool(inflight))
        if row is not None:
            rows.append(row)
    return rows


def resolve_model_path_for_stage(arm: str, model_name: str, stage_id: str) -> str:
    current = model_path(arm, model_name, stage_id)
    if os.path.exists(current):
        return current
    source = source_model_path(arm, model_name, stage_id)
    if os.path.exists(source):
        return source
    return ""


def _get_train_fn(model_name: str):
    cfg = MODEL_CONFIGS[model_name]
    return train_model_b200 if cfg.train_gpu.lower() == "b200" else train_model_h100


def _alpha_for_model_stage(arm: str, model_name: str, stage_id: str) -> float:
    saved = _load_saved_alpha(arm, model_name, stage_id)
    if saved is not None:
        return float(saved)
    return 1.0


@app.function(
    image=image,
    cpu=ORCH_FN_CPU,
    memory=ORCH_FN_MEMORY_MB,
    timeout=ORCH_FN_TIMEOUT_SEC,
    nonpreemptible=ORCH_FN_NONPREEMPTIBLE,
    volumes={"/data": vol},
)
def run_pipeline(train_models: List[str], eval_models: List[str], train_modes: List[str], eval_modes: List[str],
                 max_parallel_train: int, max_parallel_collect: int, max_parallel_eval: int, seed_base: int = 0) -> Dict[str, Any]:
    vol.reload()
    _ensure_dirs()
    print("=" * 78)
    print("CLEAN TRANSFER A* EXPERIMENT — MAP SCALE + SUBDUED DYNAMICS")
    print("=" * 78)
    print(f"VOLUME_NAME={VOLUME_NAME}")
    print(f"RUN_TAG={RUN_TAG}")
    if MODEL_RUN_TAG != RUN_TAG:
        print(f"MODEL_RUN_TAG={MODEL_RUN_TAG}  (reading existing models from this run tag)")
    print(f"Train backbones: {train_models}")
    print(f"Eval backbones:  {eval_models}")
    print(f"Train modes:     {train_modes}")
    print(f"Eval modes:      {eval_modes}")
    print(f"Stages:          {[s.stage_id for s in STAGES_TO_RUN]}")
    print(f"Eval suites:     {[s.suite_id for s in EVAL_SUITES]}")
    print(f"Budgets:         {EVAL_BUDGETS}")
    print(f"Alpha candidates:{ALPHA_CANDIDATES}")
    print("")

    # 1) dataset collection + merge
    for stage in STAGES_TO_RUN:
        print(f"\n📦 Stage {stage.stage_id}: dataset + training")
        merged_path = dataset_path(stage.stage_id)
        if os.path.exists(merged_path):
            print(f"  ✓ using cached dataset: {merged_path}")
        else:
            if SKIP_COLLECT:
                raise RuntimeError(f"SKIP_COLLECT=1 but dataset missing: {merged_path}")
            chunk_count = min(max_parallel_collect, max(1, math.ceil(stage.collect_samples / 400)))
            per_chunk = [stage.collect_samples // chunk_count] * chunk_count
            for i in range(stage.collect_samples % chunk_count):
                per_chunk[i] += 1
            handles = []
            for chunk_id, nsamp in enumerate(per_chunk):
                handles.append(collect_data_chunk.spawn(stage.stage_id, chunk_id, nsamp, seed_base + 1000 * STAGE_INDEX[stage.stage_id]))
            chunk_paths = [h.get() for h in handles]
            merged_path = merge_chunks.remote(stage.stage_id, chunk_paths)
            print(f"  ✓ built dataset: {merged_path}")

        # 2) training
        if SKIP_TRAIN:
            print("  ⏭️  SKIP_TRAIN=1 set; skipping training.")
        else:
            handles = []
            if STAGE_INDEX[stage.stage_id] == 0:
                if train_models and ("fullft" in train_modes or "lora" in train_modes):
                    for model_name in train_models:
                        dst = model_path("fullft", model_name, stage.stage_id)
                        if os.path.exists(dst):
                            print(f"  ✓ existing base model: {dst}")
                            continue
                        fn = _get_train_fn(model_name)
                        handles.append(fn.spawn(model_name, "fullft", stage.stage_id, merged_path, seed_base + 10 + STAGE_INDEX[stage.stage_id]))
                    for h in handles:
                        _ = h.get()
                if "lora" in train_modes:
                    for model_name in train_models:
                        src = model_path("fullft", model_name, stage.stage_id)
                        dst = model_path("lora", model_name, stage.stage_id)
                        if not os.path.exists(dst):
                            if not os.path.exists(src):
                                raise RuntimeError(f"missing stage-1 base to copy into lora arm: {src}")
                            _copy_model_artifact(src, dst)
                            vol.commit()
                            print(f"  ✓ copied shared stage-1 base -> {dst}")
            else:
                queue: List[Tuple[str, str]] = []
                for model_name in train_models:
                    for arm in train_modes:
                        dst = model_path(arm, model_name, stage.stage_id)
                        if os.path.exists(dst):
                            print(f"  ✓ existing model: {dst}")
                            continue
                        queue.append((model_name, arm))
                handles = []
                for model_name, arm in queue:
                    fn = _get_train_fn(model_name)
                    handles.append(fn.spawn(model_name, arm, stage.stage_id, merged_path, seed_base + 10 + STAGE_INDEX[stage.stage_id]))
                    if len(handles) >= max_parallel_train:
                        _ = handles.pop(0).get()
                for h in handles:
                    _ = h.get()

        # 3) stage-wise alpha tuning
        if SKIP_ALPHA_TUNE:
            print("  ⏭️  SKIP_ALPHA_TUNE=1 set; skipping alpha tuning.")
        else:
            val_suite_ids = _validation_suite_ids(stage.stage_id)
            tune_models = sorted(set(train_models) | set(eval_models))
            tune_modes = sorted(set(train_modes) | set(eval_modes))
            for model_name in tune_models:
                for arm in tune_modes:
                    mpath = resolve_model_path_for_stage(arm, model_name, stage.stage_id)
                    if not mpath:
                        continue
                    ap = alpha_path(arm, model_name, stage.stage_id)
                    if os.path.exists(ap):
                        print(f"  ✓ cached alpha: {ap}")
                        continue
                    jobs: List[Dict[str, Any]] = []
                    for alpha in ALPHA_CANDIDATES:
                        for suite_id in val_suite_ids:
                            jobs.append({
                                "model_eval_id": eval_model_id(arm, model_name, stage.stage_id),
                                "display_name": _model_display_name(arm, model_name),
                                "model_path": mpath,
                                "model_name": model_name,
                                "arm": arm,
                                "stage_id": stage.stage_id,
                                "suite_id": suite_id,
                                "budget": ALPHA_TUNE_BUDGET,
                                "alpha": float(alpha),
                                "episodes": VALIDATION_EPISODES,
                                "seed_base": seed_base + 500_000 + 100 * STAGE_INDEX[stage.stage_id],
                            })
                    rows = _run_eval_jobs(jobs, max_parallel_eval=max_parallel_eval)
                    grouped: Dict[float, List[Dict[str, Any]]] = {}
                    for row in rows:
                        grouped.setdefault(float(row["alpha"]), []).append(row)
                    best_alpha = ALPHA_CANDIDATES[0]
                    best_key: Optional[Tuple[float, float]] = None
                    score_payload: Dict[str, float] = {}
                    for alpha, vals in grouped.items():
                        succ = sum(float(v["success_rate"]) for v in vals) / max(1, len(vals))
                        expn = sum(float(v["avg_expansions"]) for v in vals) / max(1, len(vals))
                        score_payload[str(alpha)] = succ
                        key = (succ, -expn)
                        if best_key is None or key > best_key:
                            best_key = key
                            best_alpha = alpha
                    _save_alpha_choice(arm, model_name, stage.stage_id, best_alpha, score_payload)
                    vol.commit()
                    print(f"  ✓ tuned alpha [{arm}][{model_name}][{stage.stage_id}] = {best_alpha}")

    # 4) final evaluation on the last stage that exists in this run family
    if SKIP_EVAL:
        print("\n⏭️  SKIP_EVAL=1 set; skipping final evaluation.")
        return {"ok": True, "results": []}

    final_stage = STAGES_TO_RUN[-1].stage_id
    print(f"\n📊 Final evaluation on stage {final_stage} (parallel, sharded + cacheable)")
    eval_jobs: List[Dict[str, Any]] = []

    # baseline manhattan A*
    for suite in EVAL_SUITES:
        for budget in EVAL_BUDGETS:
            eval_jobs.append({
                "model_eval_id": eval_model_id("baseline", "manhattan_astar", final_stage),
                "display_name": "baseline_manhattan_astar",
                "model_path": "",
                "model_name": "baseline",
                "arm": "baseline",
                "stage_id": final_stage,
                "suite_id": suite.suite_id,
                "budget": int(budget),
                "alpha": 1.0,
                "episodes": suite.episodes,
                "seed_base": seed_base + 900_000,
            })

    for model_name in eval_models:
        for arm in eval_modes:
            mpath = resolve_model_path_for_stage(arm, model_name, final_stage)
            if not mpath:
                print(f"  ! missing model {arm}/{model_name} at {final_stage}, skipping eval")
                continue
            alpha = _alpha_for_model_stage(arm, model_name, final_stage)
            for suite in EVAL_SUITES:
                for budget in EVAL_BUDGETS:
                    eval_jobs.append({
                        "model_eval_id": eval_model_id(arm, model_name, final_stage),
                        "display_name": _model_display_name(arm, model_name),
                        "model_path": mpath,
                        "model_name": model_name,
                        "arm": arm,
                        "stage_id": final_stage,
                        "suite_id": suite.suite_id,
                        "budget": int(budget),
                        "alpha": float(alpha),
                        "episodes": suite.episodes,
                        "seed_base": seed_base + 900_000,
                    })

    rows = _run_eval_jobs(eval_jobs, max_parallel_eval=max_parallel_eval)
    rows = sorted(rows, key=lambda r: (str(r["model"]), str(r["suite"]), int(r["budget"])))
    results_json = f"{RESULTS_DIR}/final_results__{final_stage}.json"
    results_csv = f"{RESULTS_DIR}/final_results__{final_stage}.csv"
    _write_json_atomic(results_json, {"rows": rows, "final_stage": final_stage, "run_tag": RUN_TAG, "model_run_tag": MODEL_RUN_TAG})
    _write_csv_atomic(results_csv, rows)
    vol.commit()
    print(f"\n✅ Wrote results: {results_json}")
    print(f"✅ Wrote CSV:     {results_csv}")
    return {"ok": True, "results_json": results_json, "results_csv": results_csv, "rows": rows}


@app.local_entrypoint()
def main():
    run_pipeline.remote(
        TRAIN_MODELS,
        EVAL_MODELS,
        TRAIN_TRANSFER_MODES,
        EVAL_TRANSFER_MODES,
        MAX_PARALLEL_TRAIN,
        MAX_PARALLEL_COLLECT,
        MAX_PARALLEL_EVAL,
        SEED_BASE,
    )
