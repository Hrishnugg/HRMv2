#!/usr/bin/env python3
"""
Residual task LoRA v2 experiment for learned A* residual heuristics.

This revises the older stagewise transfer setup to follow the new LoRA framing:

- Collect the same per-task datasets as before.
- Train one pooled base model on the union of all training tasks, with task-balanced sampling.
- Freeze that pooled base and train one separate LoRA expert per training task.
- Evaluate the average base across the full suite and the task experts on a cross-task ID matrix
  by default (optionally all suites).

The script remains intentionally self-contained so `modal run` can mount only this
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
from pathlib import Path
from dataclasses import dataclass, asdict, replace
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

APP_NAME = "residual-tasklora-v2"
VOLUME_NAME = os.environ.get("VOLUME_NAME", "residual-tasklora-v2-vol")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(["torch>=2.4.0", "numpy", "tqdm"])
)
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Modal does NOT auto-forward local env vars to remote containers. The eval/orchestration
# code reads these as module globals at container-import time, so without forwarding they
# always take their in-container defaults. Forward an allowlist of eval-control vars via a
# Secret built from the LOCAL environment at deploy time (runtime injection, no image
# rebuild). Path/identity vars (RUN_TAG, MODEL_RUN_TAG, VOLUME_NAME) are intentionally
# EXCLUDED so data locations are never silently changed.
_EVAL_FORWARD_VARS = [
    "EVAL_DIAG",
    "PLANNER", "FOCAL_W",
    "EVAL_TORCH_THREADS", "EVAL_TORCH_INTEROP_THREADS",
    "EVAL_BUDGETS", "EVAL_SHARD_SIZE", "EVAL_CHECKPOINT_EVERY",
    "EVAL_ONLY_SUITES", "EVAL_SKIP_SUITES",
    "FORCE_REEVAL_SUITES", "FORCE_REEVAL", "FORCE_REEVAL_MODELED",
    "EVAL_EPISODES", "VALIDATION_EPISODES",
    "ALPHA_CANDIDATES", "ALPHA_TUNE_BUDGET",
    "SKIP_COLLECT", "SKIP_TRAIN", "SKIP_ALPHA_TUNE", "SKIP_EVAL",
    "EVAL_ARMS", "EVAL_MODELS",
]


def _eval_forward_env() -> Dict[str, str]:
    # EVAL_DIAG is always forwarded (defaulting to "1") so the speedup flag is never
    # silently lost; other vars are forwarded only when set locally so unset vars keep
    # their in-container defaults.
    out: Dict[str, str] = {"EVAL_DIAG": os.environ.get("EVAL_DIAG", "1")}
    for k in _EVAL_FORWARD_VARS:
        v = os.environ.get(k)
        if v is not None:
            out[k] = v
    return out


eval_env_secret = modal.Secret.from_dict(_eval_forward_env())

DATA_ROOT = "/data/residual_tasklora_v2"
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


def _parse_stage_epoch_overrides(raw: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    s = (raw or "").strip()
    if not s:
        return out
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"bad STAGE_EPOCHS entry: {item!r} (expected stage_id:epochs)")
        key, value = item.split(":", 1)
        out[key.strip()] = int(value.strip())
    return out


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
    raw = os.environ.get("RUN_TAG") or "residual_tasklora_v2"
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


def _refresh_volume(reason: str = "") -> None:
    try:
        vol.reload()
    except Exception as e:
        if reason:
            print(f"[volume] reload failed after {reason}: {_summarize_exc(e)}")
        else:
            print(f"[volume] reload failed: {_summarize_exc(e)}")


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
        Stage("A32_static", "A", 32, "static", 80, 18, 10_000, 1400, 64, 2, 40, 0, 0, 0),
        Stage("A64_static", "A", 64, "static", 160, 20, 15_000, 2200, 64, 3, 80, 0, 0, 0),
        Stage("A64_sparseDyn", "A", 64, "sparseDyn", 170, 20, 20_000, 2600, 64, 3, 120, 1, 1, 0),
        Stage("A64_moderateDyn", "A", 64, "moderateDyn", 175, 21, 22_000, 2800, 64, 3, 140, 1, 2, 1),
    ]
    if include_stretch:
        stages.append(Stage("A64_fullDyn", "A", 64, "fullDyn", 180, 22, 25_000, 3200, 64, 3, 160, 2, 3, 2))
    overrides = _parse_stage_epoch_overrides(os.environ.get("STAGE_EPOCHS", ""))
    if overrides:
        stages = [replace(stage, train_epochs=overrides.get(stage.stage_id, stage.train_epochs)) for stage in stages]
    return stages


def build_eval_suites(include_stretch: bool, eval_episodes: int) -> List[EvalSuite]:
    suites = [
        EvalSuite("ID_A32_static", "A", 32, "static", 80, 18, 0, 0, 0, eval_episodes),
        EvalSuite("ID_A64_static", "A", 64, "static", 160, 20, 0, 0, 0, eval_episodes),
        EvalSuite("ID_A64_sparseDyn", "A", 64, "sparseDyn", 170, 20, 1, 1, 0, eval_episodes),
        EvalSuite("ID_A64_moderateDyn", "A", 64, "moderateDyn", 175, 21, 1, 2, 1, eval_episodes),
        EvalSuite("OOD_B64_static", "B", 64, "static", 160, 20, 0, 0, 0, eval_episodes),
        EvalSuite("OOD_C64_static", "C", 64, "static", 160, 20, 0, 0, 0, eval_episodes),
        EvalSuite("OOD_B64_sparseDyn", "B", 64, "sparseDyn", 170, 20, 1, 1, 0, eval_episodes),
        EvalSuite("OOD_C64_sparseDyn", "C", 64, "sparseDyn", 170, 20, 1, 1, 0, eval_episodes),
        EvalSuite("OOD_B64_moderateDyn", "B", 64, "moderateDyn", 175, 21, 1, 2, 1, eval_episodes),
        EvalSuite("OOD_C64_moderateDyn", "C", 64, "moderateDyn", 175, 21, 1, 2, 1, eval_episodes),
        EvalSuite("OOD_A96_static", "A", 96, "static", 240, 26, 0, 0, 0, eval_episodes),
        EvalSuite("OOD_A96_sparseDyn", "A", 96, "sparseDyn", 250, 26, 1, 1, 0, eval_episodes),
        EvalSuite("OOD_A96_moderateDyn", "A", 96, "moderateDyn", 260, 28, 1, 2, 1, eval_episodes),
        EvalSuite("OOD_A128_static", "A", 128, "static", 320, 32, 0, 0, 0, eval_episodes),
        EvalSuite("OOD_A128_sparseDyn", "A", 128, "sparseDyn", 340, 32, 1, 1, 0, eval_episodes),
        EvalSuite("OOD_A128_moderateDyn", "A", 128, "moderateDyn", 350, 34, 1, 2, 1, eval_episodes),
        EvalSuite("OOD_A192_static", "A", 192, "static", 480, 40, 0, 0, 0, eval_episodes),
        EvalSuite("OOD_A192_sparseDyn", "A", 192, "sparseDyn", 500, 40, 1, 1, 0, eval_episodes),
        EvalSuite("OOD_A192_moderateDyn", "A", 192, "moderateDyn", 520, 42, 1, 2, 1, eval_episodes),
        EvalSuite("OOD_A256_static", "A", 256, "static", 640, 48, 0, 0, 0, eval_episodes),
        EvalSuite("OOD_A256_sparseDyn", "A", 256, "sparseDyn", 660, 48, 1, 1, 0, eval_episodes),
        EvalSuite("OOD_A256_moderateDyn", "A", 256, "moderateDyn", 680, 50, 1, 2, 1, eval_episodes),
    ]
    if include_stretch:
        suites.extend([
            EvalSuite("ID_A64_fullDyn", "A", 64, "fullDyn", 180, 22, 2, 3, 2, eval_episodes),
            EvalSuite("OOD_B64_fullDyn", "B", 64, "fullDyn", 180, 22, 2, 3, 2, eval_episodes),
            EvalSuite("OOD_C64_fullDyn", "C", 64, "fullDyn", 180, 22, 2, 3, 2, eval_episodes),
            EvalSuite("OOD_A96_fullDyn", "A", 96, "fullDyn", 270, 28, 2, 3, 2, eval_episodes),
            EvalSuite("OOD_A128_fullDyn", "A", 128, "fullDyn", 360, 34, 2, 3, 2, eval_episodes),
            EvalSuite("OOD_A192_fullDyn", "A", 192, "fullDyn", 540, 42, 2, 3, 2, eval_episodes),
            EvalSuite("OOD_A256_fullDyn", "A", 256, "fullDyn", 700, 50, 2, 3, 2, eval_episodes),
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


def space_time_focal_astar(
    start_xy: Tuple[int, int],
    goal_xy: Tuple[int, int],
    t0_abs: int,
    plan_horizon: int,
    max_expansions: int,
    occ: Dict[str, np.ndarray],
    heuristic_delta_batch_fn,
    w: float = 1.0,
) -> PlanResult:
    # Focal search (A*_eps): OPEN is ordered by the admissible f = g + manhattan, which
    # bounds suboptimality by w. Among OPEN nodes with f <= w * f_min (the focal band) we
    # expand the one minimizing the learned focal key hf = manhattan + delta. The learned
    # signal only orders within the bounded band -> it can never break admissibility or
    # misdirect the search the way the additive heuristic did; a bad signal degrades to
    # Manhattan ordering. Entry layout: (f, counter, g, state, hf).
    gx, gy = goal_xy
    max_t_abs = occ["blocked"].shape[0] - 1
    n = occ["blocked"].shape[1]
    w = max(1.0, float(w))
    start_state = (start_xy[0], start_xy[1], 0)
    start_h = manhattan(start_xy[0], start_xy[1], gx, gy)
    counter = 0
    open_heap: List[Tuple[float, int, int, Tuple[int, int, int], float]] = []
    heapq.heappush(open_heap, (float(start_h), counter, 0, start_state, float(start_h)))
    counter += 1
    g_cost = {start_state: 0}
    parent: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {start_state: None}
    closed: List[Tuple[int, int, int]] = []
    best_goal_state = start_state
    best_goal_score = start_h
    expansions = 0
    while open_heap and expansions < max_expansions:
        # drop stale entries (superseded by a cheaper path) from the OPEN top
        while open_heap and g_cost.get(open_heap[0][3], INF) != open_heap[0][2]:
            heapq.heappop(open_heap)
        if not open_heap:
            break
        f_min = open_heap[0][0]
        thresh = w * f_min
        # extract the focal band: all valid OPEN entries with f <= thresh
        band: List[Tuple[float, int, int, Tuple[int, int, int], float]] = []
        while open_heap and open_heap[0][0] <= thresh:
            e = heapq.heappop(open_heap)
            if g_cost.get(e[3], INF) == e[2]:
                band.append(e)
        if not band:
            break
        # expand the band node with the smallest learned focal key (tiebreak by f then counter)
        pick_idx = min(range(len(band)), key=lambda i: (band[i][4], band[i][0], band[i][1]))
        pick = band[pick_idx]
        for i, e in enumerate(band):
            if i != pick_idx:
                heapq.heappush(open_heap, e)
        _, _, g, s, _ = pick
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
            if ng < g_cost.get(ns, INF):
                g_cost[ns] = ng
                parent[ns] = s
                f = float(ng) + float(h_base2)                 # admissible bound (no delta)
                hf = float(h_base2) + max(0.0, float(delta))   # learned focal ordering key
                heapq.heappush(open_heap, (f, counter, ng, ns, hf))
                counter += 1
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
    lora_rank: int = 8


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
            lora_rank=24,
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
            lora_rank=8,
        ),
    }


FRAME_CHANNELS = 8
PATCH_CHANNELS = 2
NODE_META_DIM = 6
PATCH_RADIUS = _env_int("PATCH_RADIUS", 7)
HISTORY_LEN = _env_int("HISTORY_LEN", 20)
PRED_DELTA_MAX = _env_float("PRED_DELTA_MAX", 2048.0)


def _delta_from_log_delta(log_delta: torch.Tensor) -> torch.Tensor:
    max_delta = max(1.0, float(PRED_DELTA_MAX))
    max_log_delta = math.log1p(max_delta)
    stable_log_delta = torch.nan_to_num(
        log_delta,
        nan=0.0,
        posinf=max_log_delta,
        neginf=0.0,
    ).clamp(min=0.0, max=max_log_delta)
    return torch.expm1(stable_log_delta)


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
        return _delta_from_log_delta(log_delta)

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
        dev = base_weight.device
        scale = self.alpha / max(1, self.rank)
        self.adapter_scale = nn.Parameter(torch.full((self.num_adapters,), float(scale), device=dev), requires_grad=False)
        self.As = nn.ParameterList()
        self.Bs = nn.ParameterList()
        for _ in range(self.num_adapters):
            a = nn.Parameter(torch.randn(self.rank, self.in_dim, device=dev) * init_scale)
            b = nn.Parameter(torch.zeros(self.out_dim, self.rank, device=dev))
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


def _stack_frame_history(frames: Sequence[np.ndarray], history_len: int = HISTORY_LEN) -> np.ndarray:
    if not frames:
        raise ValueError("expected at least one frame in history")
    window = list(frames[-history_len:])
    if len(window) < history_len:
        window = [window[0]] * (history_len - len(window)) + window
    return np.stack(window, axis=0).astype(np.float32)


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
        for idx in range(start_idx, t_abs + 1):
            ax, ay = int(agent_traj[idx, 0]), int(agent_traj[idx, 1])
            frames.append(build_step_frame(static_template, (ax, ay), gate[idx], pat[idx], drift[idx]))
        return _stack_frame_history(frames, self.history_len)

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

MODEL_CONFIGS = build_model_configs()
MODEL_NAMES_ALL = list(MODEL_CONFIGS.keys())
EXPERIMENT_ARMS_ALL = ["avgbase", "tasklora"]

TRAIN_MODELS_SPEC = _parse_models_spec(os.environ.get("TRAIN_MODELS") or os.environ.get("ONLY_MODELS"))
EVAL_MODELS_SPEC = _parse_models_spec(os.environ.get("EVAL_MODELS") or os.environ.get("ONLY_MODELS"))


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


def _normalize_experiment_arms(spec_raw: Optional[str]) -> List[str]:
    if spec_raw is None:
        return EXPERIMENT_ARMS_ALL
    s = spec_raw.strip()
    if s == "" or s.lower() in ("all", "*"):
        return EXPERIMENT_ARMS_ALL
    aliases = {
        "base": "avgbase",
        "avg": "avgbase",
        "average": "avgbase",
        "average_base": "avgbase",
        "averagebase": "avgbase",
        "lora": "tasklora",
        "expert": "tasklora",
        "experts": "tasklora",
        "task_lora": "tasklora",
        "task-expert": "tasklora",
    }
    out: List[str] = []
    for x in _parse_csv_strs(s):
        key = x.strip().lower().replace("-", "_")
        key = aliases.get(key, key)
        if key in EXPERIMENT_ARMS_ALL and key not in out:
            out.append(key)
    return out or EXPERIMENT_ARMS_ALL


TRAIN_ARMS = _normalize_experiment_arms(os.environ.get("TRAIN_ARMS") or "avgbase,tasklora")
EVAL_ARMS = _normalize_experiment_arms(os.environ.get("EVAL_ARMS") or "avgbase,tasklora")

START_STAGE_ID = os.environ.get("START_STAGE_ID", "").strip()
if START_STAGE_ID:
    if START_STAGE_ID not in STAGE_INDEX:
        raise ValueError(f"unknown START_STAGE_ID={START_STAGE_ID}; expected one of {list(STAGE_INDEX)}")
    START_STAGE_POS = STAGE_INDEX[START_STAGE_ID]
    STAGES_TO_RUN = STAGES[START_STAGE_POS:]
else:
    STAGES_TO_RUN = STAGES

MULTITASK_BASE_STAGE_ID = "ALL_TASKS"

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
AVG_BASE_EPOCHS = _env_int("AVG_BASE_EPOCHS", EPOCHS_DEFAULT)
EVAL_TASK_EXPERTS_ALL_SUITES = (_env_int("EVAL_TASK_EXPERTS_ALL_SUITES", 0) == 1)

SKIP_COLLECT = (_env_int("SKIP_COLLECT", 0) == 1)
SKIP_TRAIN = (_env_int("SKIP_TRAIN", 0) == 1)
SKIP_ALPHA_TUNE = (_env_int("SKIP_ALPHA_TUNE", 0) == 1)
SKIP_EVAL = (_env_int("SKIP_EVAL", 0) == 1)


def dataset_path(stage_id: str) -> str:
    return f"{DATASETS_DIR}/{stage_id}__merged.pt"


def dataset_chunk_path(stage_id: str, chunk_id: int) -> str:
    return f"{DATASETS_DIR}/{stage_id}__chunk_{chunk_id:04d}.pt"


def pooled_manifest_path() -> str:
    return f"{DATASETS_DIR}/pooled_manifest__{MULTITASK_BASE_STAGE_ID}.json"


def artifact_id(arm: str, model_name: str) -> str:
    return f"{arm}__{model_name}"


def model_path(arm: str, model_name: str, stage_id: str) -> str:
    return f"{MODELS_DIR}/{artifact_id(arm, model_name)}__{stage_id}.pt"


def checkpoint_path(arm: str, model_name: str, stage_id: str) -> str:
    return f"{CHECKPOINTS_DIR}/{artifact_id(arm, model_name)}__{stage_id}.pt"


def alpha_path(arm: str, model_name: str, stage_id: str) -> str:
    return f"{ALPHAS_DIR}/{artifact_id(arm, model_name)}__{stage_id}.json"


def eval_model_id(arm: str, model_name: str, stage_id: str) -> str:
    return _sanitize_file_component(f"{arm}__{model_name}__{stage_id}")


def base_model_path(model_name: str) -> str:
    return model_path("avgbase", model_name, MULTITASK_BASE_STAGE_ID)


def task_expert_model_path(model_name: str, task_stage_id: str) -> str:
    return model_path("tasklora", model_name, task_stage_id)


def train_id_suite_id(stage_id: str) -> str:
    return f"ID_{stage_id}"


def training_id_suite_ids() -> List[str]:
    return [sid for sid in (train_id_suite_id(s.stage_id) for s in STAGES_TO_RUN) if sid in EVAL_SUITE_BY_ID]


def task_expert_eval_suite_ids() -> List[str]:
    if EVAL_TASK_EXPERTS_ALL_SUITES:
        return [suite.suite_id for suite in EVAL_SUITES]
    return training_id_suite_ids()


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
    _refresh_volume(f"merge_chunks {stage_id}")
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


class BalancedMultiStageDataset(torch.utils.data.Dataset):
    """
    Balanced round-robin view over several stage datasets.

    Each epoch spans `max_len * num_stages` examples so every task contributes
    equally, while smaller tasks are oversampled rather than dropped.
    """

    def __init__(self, payloads_by_stage: Dict[str, Dict[str, Any]],
                 history_len: int = HISTORY_LEN, patch_radius: int = PATCH_RADIUS):
        if not payloads_by_stage:
            raise ValueError("BalancedMultiStageDataset requires at least one stage payload")
        self.stage_ids = list(payloads_by_stage.keys())
        self.datasets: Dict[str, StageEpisodeDataset] = {
            stage_id: StageEpisodeDataset(payload, history_len=history_len, patch_radius=patch_radius)
            for stage_id, payload in payloads_by_stage.items()
        }
        self.stage_lengths: Dict[str, int] = {stage_id: len(ds) for stage_id, ds in self.datasets.items()}
        self.max_len = max(self.stage_lengths.values())
        self.total_len = self.max_len * len(self.stage_ids)
        self.max_grid_size = max(
            ep["size"] for ds in self.datasets.values() for ep in ds.ep_cache
        )

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stage_pos = int(idx % len(self.stage_ids))
        cycle_pos = int(idx // len(self.stage_ids))
        stage_id = self.stage_ids[stage_pos]
        ds = self.datasets[stage_id]
        sample_idx = cycle_pos % len(ds)
        item = dict(ds[sample_idx])
        obs = item["obs_seq"]
        h, w = obs.shape[-2], obs.shape[-1]
        if h < self.max_grid_size or w < self.max_grid_size:
            ph = self.max_grid_size - h
            pw = self.max_grid_size - w
            obs = F.pad(obs, (0, pw, 0, ph), value=0.0)
            obs[:, 0, h:, :] = 1.0
            obs[:, 0, :, w:] = 1.0
            item["obs_seq"] = obs
        item["task_id"] = torch.tensor(stage_pos, dtype=torch.int16)
        return item


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


def _resume_training_checkpoint(ckpt_path: str, model: nn.Module, opt: torch.optim.Optimizer) -> Tuple[int, float]:
    if not os.path.exists(ckpt_path):
        return 0, 1e9
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    opt.load_state_dict(ckpt["opt"])
    return int(ckpt["epoch"]) + 1, float(ckpt.get("best_loss", 1e9))


def _lora_plain_to_param_key_map(module: nn.Module) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    state_keys = set(module.state_dict().keys())
    for name, sub, attr in _iter_lora_targets(module, include_conv=LORA_ON_CONV, include_attn=LORA_ON_ATTN):
        if attr == "out_proj.weight":
            plain_key = f"{name}.out_proj.weight"
            orig_key = f"{name}.out_proj.parametrizations.weight.original"
        elif attr == "in_proj_weight":
            plain_key = f"{name}.in_proj_weight"
            orig_key = f"{name}.parametrizations.in_proj_weight.original"
        else:
            plain_key = f"{name}.{attr}"
            orig_key = f"{name}.parametrizations.{attr}.original"
        if orig_key in state_keys:
            mapping[plain_key] = orig_key
    return mapping


def _load_plain_state_into_parametrized_model(model: nn.Module, plain_state: Dict[str, Any],
                                              context: str = "") -> Tuple[List[str], List[str]]:
    model_sd = model.state_dict()
    key_map = _lora_plain_to_param_key_map(model)
    filtered_sd: Dict[str, Any] = {}
    remapped = 0
    skipped_shape = 0
    skipped_missing = 0
    for key, value in plain_state.items():
        target_key = key
        if target_key not in model_sd and key in key_map:
            target_key = key_map[key]
            remapped += 1
        if target_key not in model_sd:
            skipped_missing += 1
            continue
        if model_sd[target_key].shape != value.shape:
            skipped_shape += 1
            continue
        filtered_sd[target_key] = value
    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    if context:
        print(
            f"[load][{context}] remapped={remapped} skipped_missing={skipped_missing} "
            f"skipped_shape={skipped_shape} missing={len(missing)} unexpected={len(unexpected)}"
        )
    return list(missing), list(unexpected)


def _masked_log_delta_loss(model: CleanHeuristicModel, batch: Dict[str, torch.Tensor], device: str) -> torch.Tensor:
    obs_seq = batch["obs_seq"].to(device)
    node_patch = batch["node_patch"].to(device)
    node_meta = batch["node_meta"].to(device)
    target_log_delta = batch["target_log_delta"].to(device)
    target_delta = batch["target_delta"].to(device)
    mask = batch["mask"].to(device)
    pred_log = model(obs_seq, node_patch, node_meta)
    base_loss = F.smooth_l1_loss(pred_log, target_log_delta, reduction="none")
    weights = (1.0 + torch.clamp(target_delta / 6.0, max=3.0)) * mask
    return (base_loss * weights).sum() / torch.clamp(mask.sum(), min=1.0)


def _run_training_loop(model: CleanHeuristicModel, cfg: BackboneConfig, loader: torch.utils.data.DataLoader,
                       params: List[nn.Parameter], opt: torch.optim.Optimizer, ckpt_path: str, final_path: str,
                       arm: str, model_name: str, stage_id: str, epochs: int, device: str,
                       metrics_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    start_epoch, best_loss = _resume_training_checkpoint(ckpt_path, model, opt)
    model.train()
    use_amp = (device == "cuda") and (_env_int("USE_AMP", 1) == 1)

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        steps = 0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = _masked_log_delta_loss(model, batch, device)
            else:
                loss = _masked_log_delta_loss(model, batch, device)
            loss.backward()
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1
        epoch_loss /= max(1, steps)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            metrics = {
                "best_loss": best_loss,
                "epoch": epoch,
                "arm": arm,
                "stage_id": stage_id,
                "model_name": model_name,
            }
            if metrics_extra:
                metrics.update(metrics_extra)
            _save_model_artifact(final_path, cfg, arm, model_name, stage_id, model, metrics)
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


def _train_avg_base_impl(model_name: str, dataset_paths: List[str], device: str, seed: int) -> Dict[str, Any]:
    vol.reload()
    _ensure_dirs()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    payloads_by_stage = {
        Path(p).name.split("__merged.pt")[0]: load_dataset(p)
        for p in dataset_paths
    }
    ds = BalancedMultiStageDataset(payloads_by_stage)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    cfg = MODEL_CONFIGS[model_name]
    model = CleanHeuristicModel(cfg).to(device)
    _set_fullft_trainable(model)
    params = _trainable_params(model)
    opt = torch.optim.AdamW(params, lr=LR_FULL, weight_decay=WEIGHT_DECAY_FULL)

    ckpt_path = checkpoint_path("avgbase", model_name, MULTITASK_BASE_STAGE_ID)
    final_path = model_path("avgbase", model_name, MULTITASK_BASE_STAGE_ID)
    metrics_extra = {
        "train_stage_ids": ds.stage_ids,
        "stage_lengths": ds.stage_lengths,
        "balanced_epoch_len": len(ds),
    }
    return _run_training_loop(
        model=model,
        cfg=cfg,
        loader=loader,
        params=params,
        opt=opt,
        ckpt_path=ckpt_path,
        final_path=final_path,
        arm="avgbase",
        model_name=model_name,
        stage_id=MULTITASK_BASE_STAGE_ID,
        epochs=max(1, AVG_BASE_EPOCHS),
        device=device,
        metrics_extra=metrics_extra,
    )


def _train_task_lora_impl(model_name: str, task_stage_id: str, dataset_pt: str, base_model_pt: str,
                          device: str, seed: int) -> Dict[str, Any]:
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

    cfg = MODEL_CONFIGS[model_name]
    model = CleanHeuristicModel(cfg).to(device)
    _apply_stacked_lora(model, cfg.lora_rank, LORA_ALPHA, 1, include_conv=LORA_ON_CONV, include_attn=LORA_ON_ATTN, init_scale=LORA_INIT_SCALE)
    base_payload = _load_model_artifact(base_model_pt, map_location="cpu")
    _load_plain_state_into_parametrized_model(
        model,
        base_payload["model_state"],
        context=f"tasklora/{model_name}/{task_stage_id}/from_avgbase",
    )

    _set_lora_trainable(model, adapter_idx=0, train_bias=LORA_TRAIN_BIAS)
    params = _trainable_params(model)
    if not params:
        raise RuntimeError(f"no trainable parameters for tasklora/{model_name}/{task_stage_id}")
    opt = torch.optim.AdamW(params, lr=LR_LORA, weight_decay=WEIGHT_DECAY_LORA)

    ckpt_path = checkpoint_path("tasklora", model_name, task_stage_id)
    final_path = model_path("tasklora", model_name, task_stage_id)
    metrics_extra = {
        "base_model_path": base_model_pt,
        "train_stage_id": task_stage_id,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": LORA_ALPHA,
    }
    return _run_training_loop(
        model=model,
        cfg=cfg,
        loader=loader,
        params=params,
        opt=opt,
        ckpt_path=ckpt_path,
        final_path=final_path,
        arm="tasklora",
        model_name=model_name,
        stage_id=task_stage_id,
        epochs=max(1, STAGE_BY_ID[task_stage_id].train_epochs or EPOCHS_DEFAULT),
        device=device,
        metrics_extra=metrics_extra,
    )


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 12,
    volumes={"/data": vol},
)
def train_avg_base_model_h100(model_name: str, dataset_paths: List[str], seed: int = 0) -> Dict[str, Any]:
    return _train_avg_base_impl(model_name, dataset_paths, device="cuda", seed=seed)


@app.function(
    image=image,
    gpu="B200",
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 12,
    volumes={"/data": vol},
)
def train_avg_base_model_b200(model_name: str, dataset_paths: List[str], seed: int = 0) -> Dict[str, Any]:
    return _train_avg_base_impl(model_name, dataset_paths, device="cuda", seed=seed)


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 12,
    volumes={"/data": vol},
)
def train_task_lora_model_h100(model_name: str, task_stage_id: str, dataset_pt: str, base_model_pt: str,
                               seed: int = 0) -> Dict[str, Any]:
    return _train_task_lora_impl(model_name, task_stage_id, dataset_pt, base_model_pt, device="cuda", seed=seed)


@app.function(
    image=image,
    gpu="B200",
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 12,
    volumes={"/data": vol},
)
def train_task_lora_model_b200(model_name: str, task_stage_id: str, dataset_pt: str, base_model_pt: str,
                               seed: int = 0) -> Dict[str, Any]:
    return _train_task_lora_impl(model_name, task_stage_id, dataset_pt, base_model_pt, device="cuda", seed=seed)


# -----------------------------------------------------------------------------
# Eval utilities and diagnostics
# -----------------------------------------------------------------------------


def _load_model_for_eval(model_path_str: str, device: str) -> CleanHeuristicModel:
    payload = _load_model_artifact(model_path_str, map_location="cpu")
    cfg = BackboneConfig(**payload["cfg"])
    model = CleanHeuristicModel(cfg)
    arm = payload.get("arm", "avgbase")
    stage_id = payload.get("stage_id", MULTITASK_BASE_STAGE_ID)
    if arm == "tasklora":
        _apply_stacked_lora(model, cfg.lora_rank, LORA_ALPHA, 1, include_conv=LORA_ON_CONV, include_attn=LORA_ON_ATTN, init_scale=LORA_INIT_SCALE)
    elif arm == "lora":
        stage_idx = STAGE_INDEX.get(stage_id, 0)
        if stage_idx > 0:
            _apply_stacked_lora(model, cfg.lora_rank, LORA_ALPHA, stage_idx, include_conv=LORA_ON_CONV, include_attn=LORA_ON_ATTN, init_scale=LORA_INIT_SCALE)
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
    paths = [alpha_path(arm, model_name, stage_id)]
    if "source_alpha_path" in globals():
        src = source_alpha_path(arm, model_name, stage_id)
        if src not in paths:
            paths.append(src)
    for p in paths:
        data = _read_json_safe(p)
        if not data:
            continue
        try:
            return float(data["best_alpha"])
        except Exception:
            continue
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
    frame_history: List[np.ndarray] = []
    model_tag = "baseline" if model is None else getattr(model, "arm", "avgbase")

    for t_abs in range(ep.max_steps):
        frame = build_step_frame(static_template, agent_xy, occ["gate"][t_abs], occ["pat"][t_abs], occ["drift"][t_abs])
        eval_tag = f"eval/{model_tag}/{suite.suite_id}/seed={seed}/B={max_expansions}/t={t_abs}"
        if model is not None:
            frame_history.append(frame)
            if len(frame_history) > HISTORY_LEN:
                frame_history.pop(0)
            # Match training-time context encoding by re-encoding the latest fixed history window.
            obs_seq_t = torch.from_numpy(_stack_frame_history(frame_history, HISTORY_LEN)).unsqueeze(0).to(device)
            with torch.no_grad():
                ctx = model.encode_obs_sequence(obs_seq_t)
            _assert_finite_eval_value(f"{eval_tag}/ctx", ctx)
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
    force_reeval = FORCE_REEVAL or suite_id in FORCE_REEVAL_SUITE_IDS or (FORCE_REEVAL_MODELED and bool(model_path_str))
    existing = None if force_reeval else _read_json_safe(out_path)
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
    secrets=[eval_env_secret],
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
    secrets=[eval_env_secret],
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
    secrets=[eval_env_secret],
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
        force_reeval = _force_reeval_job(job)
        agg_path = eval_agg_path(job["model_eval_id"], job["suite_id"], job["budget"], job["alpha"], job["episodes"])
        agg = None if force_reeval else _read_json_safe(agg_path)
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
            if force_reeval or not _is_complete_eval_shard(_read_json_safe(p)):
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



def resolve_model_path(arm: str, model_name: str, stage_id: str) -> str:
    _refresh_volume(f"resolve model path {arm}/{model_name}/{stage_id}")
    current = model_path(arm, model_name, stage_id)
    if os.path.exists(current):
        return current
    source = source_model_path(arm, model_name, stage_id)
    return source if os.path.exists(source) else ""


def _get_avg_base_train_fn(model_name: str):
    cfg = MODEL_CONFIGS[model_name]
    return train_avg_base_model_b200 if cfg.train_gpu.lower() == "b200" else train_avg_base_model_h100


def _get_task_expert_train_fn(model_name: str):
    cfg = MODEL_CONFIGS[model_name]
    return train_task_lora_model_b200 if cfg.train_gpu.lower() == "b200" else train_task_lora_model_h100


def _alpha_for_model_stage(arm: str, model_name: str, stage_id: str) -> float:
    saved = _load_saved_alpha(arm, model_name, stage_id)
    if saved is not None:
        return float(saved)
    return 1.0


def _artifact_display_name(arm: str, model_name: str, stage_id: str) -> str:
    if arm == "avgbase":
        return f"avgbase__{model_name}"
    if arm == "tasklora":
        return f"tasklora__{model_name}__{stage_id}"
    return f"{arm}__{model_name}"


def _best_alpha_from_rows(rows: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
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
    return float(best_alpha), score_payload


@app.function(
    image=image,
    cpu=ORCH_FN_CPU,
    memory=ORCH_FN_MEMORY_MB,
    timeout=ORCH_FN_TIMEOUT_SEC,
    nonpreemptible=ORCH_FN_NONPREEMPTIBLE,
    volumes={"/data": vol},
    secrets=[eval_env_secret],
)
def run_pipeline(train_models: List[str], eval_models: List[str], train_arms: List[str], eval_arms: List[str],
                 max_parallel_train: int, max_parallel_collect: int, max_parallel_eval: int, seed_base: int = 0) -> Dict[str, Any]:
    vol.reload()
    _ensure_dirs()
    print("=" * 78)
    print("MULTITASK AVG-BASE + TASK-LORA A* EXPERIMENT")
    print("=" * 78)
    print(f"VOLUME_NAME={VOLUME_NAME}")
    print(f"RUN_TAG={RUN_TAG}")
    print(f"Train backbones: {train_models}")
    print(f"Eval backbones:  {eval_models}")
    print(f"Train arms:      {train_arms}")
    print(f"Eval arms:       {eval_arms}")
    print(f"Stages:          {[s.stage_id for s in STAGES_TO_RUN]}")
    print(f"Stage epochs:    { {s.stage_id: s.train_epochs for s in STAGES_TO_RUN} }")
    print(f"Avg-base epochs: {AVG_BASE_EPOCHS}")
    print(f"Eval suites:     {[s.suite_id for s in final_eval_suites()]}")
    print(f"Expert eval suites (default): {task_expert_eval_suite_ids()}")
    print(f"Budgets:         {EVAL_BUDGETS}")
    print(f"Alpha candidates:{ALPHA_CANDIDATES}")
    print("")

    dataset_paths: Dict[str, str] = {}

    # 1) per-task dataset collection
    for stage in STAGES_TO_RUN:
        print(f"\n📦 Dataset {stage.stage_id}")
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
            _refresh_volume(f"dataset merge {stage.stage_id}")
            print(f"  ✓ built dataset: {merged_path}")
        dataset_paths[stage.stage_id] = merged_path

    manifest = {
        "run_tag": RUN_TAG,
        "stages": [s.stage_id for s in STAGES_TO_RUN],
        "dataset_paths": dataset_paths,
        "expert_eval_suites": task_expert_eval_suite_ids(),
    }
    _write_json_atomic(pooled_manifest_path(), manifest)
    vol.commit()

    # 2) average base training
    if SKIP_TRAIN:
        print("\n⏭️  SKIP_TRAIN=1 set; skipping training.")
    else:
        if "avgbase" in train_arms:
            dataset_path_list = [dataset_paths[s.stage_id] for s in STAGES_TO_RUN]
            handles = []
            for model_name in train_models:
                dst = base_model_path(model_name)
                if os.path.exists(dst):
                    print(f"\n🎯 Avg-base {model_name}: cached at {dst}")
                    continue
                fn = _get_avg_base_train_fn(model_name)
                handles.append(fn.spawn(model_name, dataset_path_list, seed_base + 10))
                if len(handles) >= max_parallel_train:
                    _ = handles.pop(0).get()
            for h in handles:
                _ = h.get()
            _refresh_volume("avg-base training")

        # 3) task-specific LoRA experts
        if "tasklora" in train_arms:
            queue: List[Tuple[str, str, str, str]] = []
            for model_name in train_models:
                base_p = resolve_model_path("avgbase", model_name, MULTITASK_BASE_STAGE_ID)
                if not base_p:
                    raise RuntimeError(f"avg-base missing for {model_name}; expected {base_model_path(model_name)}")
                for stage in STAGES_TO_RUN:
                    dst = task_expert_model_path(model_name, stage.stage_id)
                    if os.path.exists(dst):
                        print(f"\n🎯 Task expert {model_name}/{stage.stage_id}: cached at {dst}")
                        continue
                    queue.append((model_name, stage.stage_id, dataset_paths[stage.stage_id], base_p))
            handles = []
            for model_name, stage_id, dpath, base_p in queue:
                fn = _get_task_expert_train_fn(model_name)
                handles.append(fn.spawn(model_name, stage_id, dpath, base_p, seed_base + 100 + STAGE_INDEX[stage_id]))
                if len(handles) >= max_parallel_train:
                    _ = handles.pop(0).get()
            for h in handles:
                _ = h.get()
            _refresh_volume("task-lora training")

    # 4) alpha tuning
    if SKIP_ALPHA_TUNE:
        print("\n⏭️  SKIP_ALPHA_TUNE=1 set; skipping alpha tuning.")
    else:
        tune_models = sorted(set(train_models) | set(eval_models))
        tune_arms = []
        for arm in EXPERIMENT_ARMS_ALL:
            if arm in set(train_arms) | set(eval_arms):
                tune_arms.append(arm)

        for model_name in tune_models:
            if "avgbase" in tune_arms:
                mpath = resolve_model_path("avgbase", model_name, MULTITASK_BASE_STAGE_ID)
                if mpath:
                    ap = alpha_path("avgbase", model_name, MULTITASK_BASE_STAGE_ID)
                    if os.path.exists(ap):
                        print(f"  ✓ cached alpha: {ap}")
                    else:
                        jobs: List[Dict[str, Any]] = []
                        for alpha in ALPHA_CANDIDATES:
                            for suite_id in training_id_suite_ids():
                                jobs.append({
                                    "model_eval_id": eval_model_id("avgbase", model_name, MULTITASK_BASE_STAGE_ID),
                                    "display_name": _artifact_display_name("avgbase", model_name, MULTITASK_BASE_STAGE_ID),
                                    "model_path": mpath,
                                    "model_name": model_name,
                                    "arm": "avgbase",
                                    "stage_id": MULTITASK_BASE_STAGE_ID,
                                    "suite_id": suite_id,
                                    "budget": ALPHA_TUNE_BUDGET,
                                    "alpha": float(alpha),
                                    "episodes": VALIDATION_EPISODES,
                                    "seed_base": seed_base + 500_000,
                                })
                        rows = _run_eval_jobs(jobs, max_parallel_eval=max_parallel_eval)
                        best_alpha, score_payload = _best_alpha_from_rows(rows)
                        _save_alpha_choice("avgbase", model_name, MULTITASK_BASE_STAGE_ID, best_alpha, score_payload)
                        vol.commit()
                        print(f"  ✓ tuned alpha [avgbase][{model_name}] = {best_alpha}")

            if "tasklora" in tune_arms:
                for stage in STAGES_TO_RUN:
                    mpath = resolve_model_path("tasklora", model_name, stage.stage_id)
                    if not mpath:
                        continue
                    ap = alpha_path("tasklora", model_name, stage.stage_id)
                    if os.path.exists(ap):
                        print(f"  ✓ cached alpha: {ap}")
                        continue
                    suite_id = train_id_suite_id(stage.stage_id)
                    if suite_id not in EVAL_SUITE_BY_ID:
                        continue
                    jobs = []
                    for alpha in ALPHA_CANDIDATES:
                        jobs.append({
                            "model_eval_id": eval_model_id("tasklora", model_name, stage.stage_id),
                            "display_name": _artifact_display_name("tasklora", model_name, stage.stage_id),
                            "model_path": mpath,
                            "model_name": model_name,
                            "arm": "tasklora",
                            "stage_id": stage.stage_id,
                            "suite_id": suite_id,
                            "budget": ALPHA_TUNE_BUDGET,
                            "alpha": float(alpha),
                            "episodes": VALIDATION_EPISODES,
                            "seed_base": seed_base + 600_000 + 100 * STAGE_INDEX[stage.stage_id],
                        })
                    rows = _run_eval_jobs(jobs, max_parallel_eval=max_parallel_eval)
                    best_alpha, score_payload = _best_alpha_from_rows(rows)
                    _save_alpha_choice("tasklora", model_name, stage.stage_id, best_alpha, score_payload)
                    vol.commit()
                    print(f"  ✓ tuned alpha [tasklora][{model_name}][{stage.stage_id}] = {best_alpha}")

    # 5) evaluation
    if SKIP_EVAL:
        print("\n⏭️  SKIP_EVAL=1 set; skipping final evaluation.")
        return {"ok": True, "results": []}

    _refresh_volume("pre-final-eval")
    print("\n📊 Final evaluation (parallel, sharded + cacheable)")
    eval_jobs: List[Dict[str, Any]] = []
    selected_eval_suites = final_eval_suites()

    # baseline Manhattan A*
    for suite in selected_eval_suites:
        for budget in EVAL_BUDGETS:
            eval_jobs.append({
                "model_eval_id": eval_model_id("baseline", "manhattan_astar", "baseline"),
                "display_name": "baseline_manhattan_astar",
                "model_path": "",
                "model_name": "baseline",
                "arm": "baseline",
                "stage_id": "baseline",
                "suite_id": suite.suite_id,
                "budget": int(budget),
                "alpha": 1.0,
                "episodes": suite.episodes,
                "seed_base": seed_base + 900_000,
            })

    for model_name in eval_models:
        if "avgbase" in eval_arms:
            mpath = resolve_model_path("avgbase", model_name, MULTITASK_BASE_STAGE_ID)
            if not mpath:
                print(f"  ! missing avg-base model for {model_name}, skipping")
            else:
                alpha = _alpha_for_model_stage("avgbase", model_name, MULTITASK_BASE_STAGE_ID)
                for suite in selected_eval_suites:
                    for budget in EVAL_BUDGETS:
                        eval_jobs.append({
                            "model_eval_id": eval_model_id("avgbase", model_name, MULTITASK_BASE_STAGE_ID),
                            "display_name": _artifact_display_name("avgbase", model_name, MULTITASK_BASE_STAGE_ID),
                            "model_path": mpath,
                            "model_name": model_name,
                            "arm": "avgbase",
                            "stage_id": MULTITASK_BASE_STAGE_ID,
                            "suite_id": suite.suite_id,
                            "budget": int(budget),
                            "alpha": float(alpha),
                            "episodes": suite.episodes,
                            "seed_base": seed_base + 900_000,
                        })

        if "tasklora" in eval_arms:
            expert_suite_ids = task_expert_eval_suite_ids()
            for stage in STAGES_TO_RUN:
                mpath = resolve_model_path("tasklora", model_name, stage.stage_id)
                if not mpath:
                    print(f"  ! missing task expert {model_name}/{stage.stage_id}, skipping")
                    continue
                alpha = _alpha_for_model_stage("tasklora", model_name, stage.stage_id)
                for suite_id in expert_suite_ids:
                    suite = EVAL_SUITE_BY_ID[suite_id]
                    for budget in EVAL_BUDGETS:
                        eval_jobs.append({
                            "model_eval_id": eval_model_id("tasklora", model_name, stage.stage_id),
                            "display_name": _artifact_display_name("tasklora", model_name, stage.stage_id),
                            "model_path": mpath,
                            "model_name": model_name,
                            "arm": "tasklora",
                            "stage_id": stage.stage_id,
                            "suite_id": suite.suite_id,
                            "budget": int(budget),
                            "alpha": float(alpha),
                            "episodes": suite.episodes,
                            "seed_base": seed_base + 900_000,
                        })

    rows = _run_eval_jobs(eval_jobs, max_parallel_eval=max_parallel_eval)
    rows = sorted(rows, key=lambda r: (str(r["model"]), str(r["stage"]), str(r["suite"]), int(r["budget"])))
    results_json = f"{RESULTS_DIR}/final_results__residual_tasklora_v2.json"
    results_csv = f"{RESULTS_DIR}/final_results__residual_tasklora_v2.csv"
    _write_json_atomic(results_json, {
        "rows": rows,
        "run_tag": RUN_TAG,
        "stages": [s.stage_id for s in STAGES_TO_RUN],
        "expert_eval_suites": task_expert_eval_suite_ids(),
    })
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
        TRAIN_ARMS,
        EVAL_ARMS,
        MAX_PARALLEL_TRAIN,
        MAX_PARALLEL_COLLECT,
        MAX_PARALLEL_EVAL,
        SEED_BASE,
    )


@app.local_entrypoint()
def resume_spawn():
    # Durable fire-and-forget launch. Unlike main()'s blocking `.remote()` (which streams
    # for the whole run and lets a local crash/disconnect tear down the orchestrator), this
    # submits run_pipeline server-side via `.spawn()` and returns immediately. Run it with
    # `modal run --detach ...::resume_spawn` so the app and the spawned run_pipeline persist
    # independently of this client. Combine with SKIP_COLLECT/SKIP_TRAIN/SKIP_ALPHA_TUNE +
    # EVAL_DIAG to resume just the eval matrix.
    fc = run_pipeline.spawn(
        TRAIN_MODELS,
        EVAL_MODELS,
        TRAIN_ARMS,
        EVAL_ARMS,
        MAX_PARALLEL_TRAIN,
        MAX_PARALLEL_COLLECT,
        MAX_PARALLEL_EVAL,
        SEED_BASE,
    )
    print(f"[resume_spawn] submitted run_pipeline server-side; function_call_id={fc.object_id}")

# =============================================================================
# Residual task LoRA v2 overrides
# =============================================================================

# The original v1 script above is kept as a base for data collection, models,
# Modal wiring, and orchestration. The definitions below intentionally override
# the training/evaluation behavior so the experiment follows the v2.1 spec:
# average-base training over the pooled task distribution + bounded residual
# task-specialization around that frozen average model.


def _storage_arm_name(arm: str) -> str:
    key = (arm or "").strip().lower().replace("-", "_")
    aliases = {
        "tasklora": "residtasklora",
        "task_lora": "residtasklora",
        "residtasklora": "residtasklora",
        "resid_task_lora": "residtasklora",
        "residual_task_lora": "residtasklora",
        "residualtasklora": "residtasklora",
        "expert": "residtasklora",
        "experts": "residtasklora",
        "avgbase": "avgbase",
        "avg": "avgbase",
        "average": "avgbase",
        "average_base": "avgbase",
        "averagebase": "avgbase",
        "base": "avgbase",
        "baseline": "baseline",
    }
    return aliases.get(key, key)


# -----------------------------------------------------------------------------
# v2 run / path defaults
# -----------------------------------------------------------------------------

RUN_TAG = _sanitize_file_component(os.environ.get("RUN_TAG") or "residual_tasklora_v2")
MODEL_RUN_TAG = _sanitize_file_component(os.environ.get("MODEL_RUN_TAG") or RUN_TAG)
RUN_ROOT = f"{DATA_ROOT}/runs/{RUN_TAG}"
MODEL_RUN_ROOT = f"{DATA_ROOT}/runs/{MODEL_RUN_TAG}"
MODELS_DIR = f"{RUN_ROOT}/models"
CHECKPOINTS_DIR = f"{RUN_ROOT}/checkpoints"
RESULTS_DIR = f"{RUN_ROOT}/results"
ALPHAS_DIR = f"{RESULTS_DIR}/alphas"
SOURCE_MODELS_DIR = MODELS_DIR if RUN_ROOT == MODEL_RUN_ROOT else f"{MODEL_RUN_ROOT}/models"


# -----------------------------------------------------------------------------
# v2 backbone / arm / stage configuration
# -----------------------------------------------------------------------------

PRIMARY_BACKBONE = _canonical_model_name(os.environ.get("PRIMARY_BACKBONE", "hrm"))
BACKBONES_DEFAULT = [_canonical_model_name(x) for x in _parse_csv_strs(os.environ.get("BACKBONES", "hrm,onlstm"))]
if not BACKBONES_DEFAULT:
    BACKBONES_DEFAULT = ["hrm", "onlstm"]

TRAIN_MODELS_SPEC = _parse_models_spec(os.environ.get("TRAIN_MODELS") or os.environ.get("ONLY_MODELS") or os.environ.get("BACKBONES"))
EVAL_MODELS_SPEC = _parse_models_spec(os.environ.get("EVAL_MODELS") or os.environ.get("ONLY_MODELS") or os.environ.get("BACKBONES"))

if TRAIN_MODELS_SPEC is None:
    TRAIN_MODELS = BACKBONES_DEFAULT
elif TRAIN_MODELS_SPEC == []:
    TRAIN_MODELS = MODEL_NAMES_ALL
else:
    TRAIN_MODELS = TRAIN_MODELS_SPEC

if EVAL_MODELS_SPEC is None:
    EVAL_MODELS = BACKBONES_DEFAULT
elif EVAL_MODELS_SPEC == []:
    EVAL_MODELS = MODEL_NAMES_ALL
else:
    EVAL_MODELS = EVAL_MODELS_SPEC

EXPERIMENT_ARMS_ALL = ["avgbase", "tasklora"]  # internal alias; storage/display use residtasklora


def _normalize_experiment_arms(spec_raw: Optional[str]) -> List[str]:
    if spec_raw is None:
        return EXPERIMENT_ARMS_ALL
    s = spec_raw.strip()
    if s == "" or s.lower() in ("all", "*"):
        return EXPERIMENT_ARMS_ALL
    aliases = {
        "base": "avgbase",
        "avg": "avgbase",
        "average": "avgbase",
        "average_base": "avgbase",
        "averagebase": "avgbase",
        "lora": "tasklora",
        "task_lora": "tasklora",
        "task-expert": "tasklora",
        "task_expert": "tasklora",
        "expert": "tasklora",
        "experts": "tasklora",
        "tasklora": "tasklora",
        "residtasklora": "tasklora",
        "resid_task_lora": "tasklora",
        "residual_task_lora": "tasklora",
        "residualtasklora": "tasklora",
        "residual": "tasklora",
    }
    out: List[str] = []
    for x in _parse_csv_strs(s):
        key = x.strip().lower().replace("-", "_")
        key = aliases.get(key, key)
        if key in EXPERIMENT_ARMS_ALL and key not in out:
            out.append(key)
    return out or EXPERIMENT_ARMS_ALL


TRAIN_ARMS = _normalize_experiment_arms(os.environ.get("TRAIN_ARMS") or "avgbase,residtasklora")
EVAL_ARMS = _normalize_experiment_arms(os.environ.get("EVAL_ARMS") or "avgbase,residtasklora")

ENABLE_A64_FULLDYN = (_env_int("ENABLE_A64_FULLDYN", _env_int("INCLUDE_STRETCH_STAGE", 0)) == 1)
STAGES = build_curriculum_stages(ENABLE_A64_FULLDYN)
EVAL_SUITES = build_eval_suites(ENABLE_A64_FULLDYN, EVAL_EPISODES)
EVAL_SUITE_BY_ID = {s.suite_id: s for s in EVAL_SUITES}
STAGE_BY_ID = {s.stage_id: s for s in STAGES}
STAGE_INDEX = {s.stage_id: i for i, s in enumerate(STAGES)}

_DEFAULT_TRAIN_STAGE_IDS = ["A32_static", "A64_static", "A64_sparseDyn", "A64_moderateDyn"]
if ENABLE_A64_FULLDYN:
    _DEFAULT_TRAIN_STAGE_IDS = _DEFAULT_TRAIN_STAGE_IDS + ["A64_fullDyn"]
TRAIN_STAGE_IDS = _parse_csv_strs(os.environ.get("TRAIN_TASKS", ",".join(_DEFAULT_TRAIN_STAGE_IDS)))
if not TRAIN_STAGE_IDS:
    TRAIN_STAGE_IDS = list(_DEFAULT_TRAIN_STAGE_IDS)
for _sid in TRAIN_STAGE_IDS:
    if _sid not in STAGE_BY_ID:
        raise ValueError(f"unknown TRAIN_TASKS stage_id={_sid}; expected one of {list(STAGE_BY_ID)}")
STAGES_TO_RUN = [STAGE_BY_ID[sid] for sid in TRAIN_STAGE_IDS]
MULTITASK_BASE_STAGE_ID = "ALL_TASKS"

EVAL_EXPERTS_FAMILY_A_ONLY = (_env_int("EVAL_EXPERTS_FAMILY_A_ONLY", 1) == 1)
if _env_int("EVAL_TASK_EXPERTS_ALL_SUITES", 0) == 1:
    EVAL_EXPERTS_FAMILY_A_ONLY = False

EVAL_BUDGETS = _parse_csv_ints(os.environ.get("EVAL_BUDGETS", "200,500,2000"))
EVAL_ONLY_SUITE_IDS = set(_parse_csv_strs(os.environ.get("EVAL_ONLY_SUITES", "")))
EVAL_SKIP_SUITE_IDS = set(_parse_csv_strs(os.environ.get("EVAL_SKIP_SUITES", "")))
FORCE_REEVAL = (_env_int("FORCE_REEVAL", 0) == 1)
FORCE_REEVAL_MODELED = (_env_int("FORCE_REEVAL_MODELED", 0) == 1)
FORCE_REEVAL_SUITE_IDS = set(_parse_csv_strs(os.environ.get("FORCE_REEVAL_SUITES", "")))
ALPHA_CANDIDATES = _parse_csv_floats(os.environ.get("ALPHA_CANDIDATES", "0.5,1.0,1.5,2.0"))
ALPHA_TUNE_BUDGET = _env_int("ALPHA_TUNE_BUDGET", 500)

RESIDUAL_BOUND_PCT = _env_float("RESIDUAL_BOUND_PCT", 99.0)
RESIDUAL_BOUND_MIN = _env_float("RESIDUAL_BOUND_MIN", 16.0)
RESIDUAL_BOUND_MAX = _env_float("RESIDUAL_BOUND_MAX", 128.0)
EXPERT_LOSS_RESID_W = _env_float("EXPERT_LOSS_RESID_W", 1.0)
EXPERT_LOSS_TOTAL_W = _env_float("EXPERT_LOSS_TOTAL_W", 0.25)
EXPERT_LOSS_MAG_W = _env_float("EXPERT_LOSS_MAG_W", 1e-3)
ABORT_ON_NONFINITE = (_env_int("ABORT_ON_NONFINITE", 1) == 1)
SANITIZE_NONFINITE_EVAL = (_env_int("SANITIZE_NONFINITE_EVAL", 1) == 1)
# Diagnostics (correction-saturation / ordering metrics) require an O(max_steps*n^2)
# pure-Python exact-cost DP per episode that does NOT affect A* decisions. Default ON
# (preserves headline metrics + diag exactly). Set EVAL_DIAG=0 for fast re-eval runs
# where only success/expansions are needed; that path also enables the per-replan
# heuristic cache.
EVAL_DIAG = (_env_int("EVAL_DIAG", 1) == 1)
# Planner selection. PLANNER="focal" uses bounded-suboptimal focal search, where the
# learned signal orders the focal band (robust to magnitude miscalibration) instead of
# inflating the heuristic. FOCAL_W is the suboptimality factor (w>=1; larger = more
# reliance on the learned ranking, fewer expansions, bounded-longer paths).
# Empirically (bench_focal on the large OOD maps) the win is in a narrow window
# w~1.0-1.05 (~17-22% fewer expansions at matched success via learned tie-breaking);
# wider bands tank success because the in-search ranking isn't reliable enough. So the
# default is the safe tie-breaking floor, NOT a wide band.
PLANNER = (os.environ.get("PLANNER", "astar").strip().lower() or "astar")
FOCAL_W = _env_float("FOCAL_W", 1.0)


def _refresh_eval_diag_from_env() -> bool:
    # Re-read EVAL_DIAG from the (possibly secret-injected) container environment at
    # runtime, so the flag is correct even if it changed after module import.
    global EVAL_DIAG
    EVAL_DIAG = (_env_int("EVAL_DIAG", 1) == 1)
    return EVAL_DIAG


EVAL_DELTA_SANITIZE_MAX = _env_float("EVAL_DELTA_SANITIZE_MAX", PRED_DELTA_MAX)
SANITIZE_NONFINITE_LOG_LIMIT = _env_int("SANITIZE_NONFINITE_LOG_LIMIT", 10)
_SANITIZE_NONFINITE_LOG_COUNT = 0
SAVE_TASK_DESCRIPTORS = (_env_int("SAVE_TASK_DESCRIPTORS", 1) == 1)
HRM_LR_MULT = _env_float("HRM_LR_MULT", 1.0)
ONLSTM_LR_MULT = _env_float("ONLSTM_LR_MULT", 1.0)
CORRECTION_SAT_THRESH_FRAC = _env_float("CORRECTION_SAT_THRESH_FRAC", 0.98)
DIAG_HIST_BINS = max(32, _env_int("DIAG_HIST_BINS", 256))
UNCORR_HIST_MAX = float(_env_float("UNCORR_HIST_MAX", max(256.0, RESIDUAL_BOUND_MAX * 2.0)))


# -----------------------------------------------------------------------------
# v2 helpers: descriptors / storage / suites
# -----------------------------------------------------------------------------


def _family_one_hot(family: str) -> Dict[str, int]:
    return {
        "family_A": int(family == "A"),
        "family_B": int(family == "B"),
        "family_C": int(family == "C"),
    }


_DYN_KEYS = ["static", "sparseDyn", "moderateDyn", "fullDyn"]


def _dynamics_one_hot(dynamics: str) -> Dict[str, int]:
    return {f"dyn_{k}": int(dynamics == k) for k in _DYN_KEYS}



def descriptor_for_stage(stage: Stage) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "stage_id": stage.stage_id,
        "family": stage.family,
        "size": int(stage.size),
        "dynamics": stage.dynamics,
        "max_steps": int(stage.max_steps),
        "plan_horizon": int(stage.plan_horizon),
        "n_gates": int(stage.n_gates),
        "n_patrollers": int(stage.n_patrollers),
        "n_drifters": int(stage.n_drifters),
        "collect_samples": int(stage.collect_samples),
        "nodes_per_sample": int(stage.nodes_per_sample),
    }
    out.update(_family_one_hot(stage.family))
    out.update(_dynamics_one_hot(stage.dynamics))
    return out



def descriptor_for_suite(suite: EvalSuite) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "suite_id": suite.suite_id,
        "family": suite.family,
        "size": int(suite.size),
        "dynamics": suite.dynamics,
        "max_steps": int(suite.max_steps),
        "plan_horizon": int(suite.plan_horizon),
        "n_gates": int(suite.n_gates),
        "n_patrollers": int(suite.n_patrollers),
        "n_drifters": int(suite.n_drifters),
    }
    out.update(_family_one_hot(suite.family))
    out.update(_dynamics_one_hot(suite.dynamics))
    return out



def _stage_descriptor_for_id(stage_id: str) -> Dict[str, Any]:
    if stage_id in STAGE_BY_ID:
        return descriptor_for_stage(STAGE_BY_ID[stage_id])
    if stage_id == MULTITASK_BASE_STAGE_ID:
        return {
            "stage_id": MULTITASK_BASE_STAGE_ID,
            "family": "mixed",
            "size": -1,
            "dynamics": "mixed",
            "max_steps": -1,
            "plan_horizon": -1,
            "n_gates": -1,
            "n_patrollers": -1,
            "n_drifters": -1,
            "family_A": 1,
            "family_B": 0,
            "family_C": 0,
            "dyn_static": 1,
            "dyn_sparseDyn": 1,
            "dyn_moderateDyn": 1,
            "dyn_fullDyn": int(ENABLE_A64_FULLDYN),
        }
    return {"stage_id": stage_id}



def artifact_id(arm: str, model_name: str) -> str:
    return f"{_storage_arm_name(arm)}__{model_name}"



def model_path(arm: str, model_name: str, stage_id: str) -> str:
    return f"{MODELS_DIR}/{artifact_id(arm, model_name)}__{stage_id}.pt"


def source_model_path(arm: str, model_name: str, stage_id: str) -> str:
    return f"{SOURCE_MODELS_DIR}/{artifact_id(arm, model_name)}__{stage_id}.pt"



def checkpoint_path(arm: str, model_name: str, stage_id: str) -> str:
    return f"{CHECKPOINTS_DIR}/{artifact_id(arm, model_name)}__{stage_id}.pt"



def alpha_path(arm: str, model_name: str, stage_id: str) -> str:
    return f"{ALPHAS_DIR}/{artifact_id(arm, model_name)}__{stage_id}.json"


def source_alpha_path(arm: str, model_name: str, stage_id: str) -> str:
    return f"{MODEL_RUN_ROOT}/results/alphas/{artifact_id(arm, model_name)}__{stage_id}.json"



def eval_model_id(arm: str, model_name: str, stage_id: str) -> str:
    return _sanitize_file_component(f"{_storage_arm_name(arm)}__{model_name}__{stage_id}")



def base_model_path(model_name: str) -> str:
    return model_path("avgbase", model_name, MULTITASK_BASE_STAGE_ID)



def task_expert_model_path(model_name: str, task_stage_id: str) -> str:
    return model_path("residtasklora", model_name, task_stage_id)



def train_id_suite_id(stage_id: str) -> str:
    return f"ID_{stage_id}"



def training_id_suite_ids() -> List[str]:
    return [sid for sid in (train_id_suite_id(s.stage_id) for s in STAGES_TO_RUN) if sid in EVAL_SUITE_BY_ID]



def family_a_id_suite_ids() -> List[str]:
    return [s.suite_id for s in EVAL_SUITES if s.family == "A" and s.suite_id.startswith("ID_")]



def family_a_size_ood_suite_ids() -> List[str]:
    return [s.suite_id for s in EVAL_SUITES if s.family == "A" and s.suite_id.startswith("OOD_A")]


def _filter_eval_suite_ids(suite_ids: Sequence[str]) -> List[str]:
    out = list(suite_ids)
    if EVAL_ONLY_SUITE_IDS:
        out = [sid for sid in out if sid in EVAL_ONLY_SUITE_IDS]
    if EVAL_SKIP_SUITE_IDS:
        out = [sid for sid in out if sid not in EVAL_SKIP_SUITE_IDS]
    return out


def final_eval_suites() -> List[EvalSuite]:
    return [EVAL_SUITE_BY_ID[sid] for sid in _filter_eval_suite_ids([suite.suite_id for suite in EVAL_SUITES])]



def task_expert_eval_suite_ids() -> List[str]:
    if EVAL_EXPERTS_FAMILY_A_ONLY:
        return _filter_eval_suite_ids(family_a_id_suite_ids() + family_a_size_ood_suite_ids())
    return _filter_eval_suite_ids([suite.suite_id for suite in EVAL_SUITES])


def _force_reeval_job(job: Dict[str, Any]) -> bool:
    if FORCE_REEVAL:
        return True
    if FORCE_REEVAL_MODELED and bool(job.get("model_path")):
        return True
    if job.get("suite_id") in FORCE_REEVAL_SUITE_IDS:
        return True
    return False



def _artifact_display_name(arm: str, model_name: str, stage_id: str) -> str:
    storage_arm = _storage_arm_name(arm)
    if storage_arm == "avgbase":
        return f"avgbase__{model_name}"
    if storage_arm == "residtasklora":
        return f"residtasklora__{model_name}__{stage_id}"
    if storage_arm == "baseline":
        return "baseline_manhattan_astar"
    return f"{storage_arm}__{model_name}__{stage_id}"


# -----------------------------------------------------------------------------
# v2 helpers: finite checks / dataset metadata / losses / calibration
# -----------------------------------------------------------------------------


def _lr_multiplier_for_model(model_name: str) -> float:
    if model_name == "hrm":
        return float(HRM_LR_MULT)
    if model_name == "onlstm":
        return float(ONLSTM_LR_MULT)
    return 1.0



def _count_nonfinite_tensor(t: torch.Tensor) -> int:
    if not isinstance(t, torch.Tensor):
        return 0
    return int((~torch.isfinite(t)).sum().item())


def _count_nonfinite_eval_value(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return _count_nonfinite_tensor(value)
    if isinstance(value, dict):
        return sum(_count_nonfinite_eval_value(subvalue) for subvalue in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_nonfinite_eval_value(subvalue) for subvalue in value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(not math.isfinite(float(value)))
    return 0


def _log_nonfinite_eval(message: str) -> None:
    global _SANITIZE_NONFINITE_LOG_COUNT
    limit = int(SANITIZE_NONFINITE_LOG_LIMIT)
    if limit < 0 or _SANITIZE_NONFINITE_LOG_COUNT < limit:
        print(message)
    elif _SANITIZE_NONFINITE_LOG_COUNT == limit:
        print(f"! further nonfinite eval sanitizer messages suppressed (SANITIZE_NONFINITE_LOG_LIMIT={limit})")
    _SANITIZE_NONFINITE_LOG_COUNT += 1



def _assert_finite_tensor(name: str, t: torch.Tensor) -> None:
    bad = _count_nonfinite_tensor(t)
    if bad > 0:
        msg = f"nonfinite tensor [{name}] count={bad}"
        if ABORT_ON_NONFINITE:
            raise FloatingPointError(msg)
        print(f"! {msg}")



def _assert_finite_gradients(params: Sequence[nn.Parameter], context: str) -> None:
    bad = 0
    for p in params:
        if p.grad is None:
            continue
        bad += _count_nonfinite_tensor(p.grad)
    if bad > 0:
        msg = f"nonfinite gradients [{context}] count={bad}"
        if ABORT_ON_NONFINITE:
            raise FloatingPointError(msg)
        print(f"! {msg}")



def _require_finite_scalar(name: str, value: float) -> float:
    if not math.isfinite(float(value)):
        msg = f"nonfinite scalar [{name}]={value}"
        if ABORT_ON_NONFINITE:
            raise FloatingPointError(msg)
        print(f"! {msg}")
        return 0.0
    return float(value)



def _assert_finite_eval_value(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        _assert_finite_tensor(name, value)
        return
    if isinstance(value, dict):
        for key, subvalue in value.items():
            _assert_finite_eval_value(f"{name}/{key}", subvalue)
        return
    if isinstance(value, (list, tuple)):
        for idx, subvalue in enumerate(value):
            _assert_finite_eval_value(f"{name}/{idx}", subvalue)
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        _require_finite_scalar(name, float(value))


def _clamp_tensor_optional(t: torch.Tensor, min_value: Optional[float] = None, max_value: Optional[float] = None) -> torch.Tensor:
    if min_value is not None and max_value is not None:
        return torch.clamp(t, min=float(min_value), max=float(max_value))
    if min_value is not None:
        return torch.clamp_min(t, float(min_value))
    if max_value is not None:
        return torch.clamp_max(t, float(max_value))
    return t


def _sanitize_eval_tensor(
    name: str,
    t: torch.Tensor,
    replacement: float = 0.0,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    bad = _count_nonfinite_tensor(t)
    if bad > 0:
        _log_nonfinite_eval(f"! nonfinite eval tensor [{name}] count={bad}; replacing with {replacement}")
        t = torch.nan_to_num(t, nan=float(replacement), posinf=float(replacement), neginf=float(replacement))
    return _clamp_tensor_optional(t, min_value, max_value), bad


def _sanitize_eval_scalar(
    name: str,
    value: float,
    replacement: float = 0.0,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Tuple[float, int]:
    bad = int(not math.isfinite(float(value)))
    out = float(replacement) if bad else float(value)
    if bad > 0:
        _log_nonfinite_eval(f"! nonfinite eval scalar [{name}]={value}; replacing with {replacement}")
    if min_value is not None:
        out = max(float(min_value), out)
    if max_value is not None:
        out = min(float(max_value), out)
    return out, bad


def _sanitize_residual_parts_for_eval(eval_tag: str, parts: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    out = dict(parts)
    original_bad = _count_nonfinite_eval_value(out)

    if "bound_B" in out:
        if isinstance(out["bound_B"], torch.Tensor):
            out["bound_B"], _ = _sanitize_eval_tensor(
                f"{eval_tag}/parts/bound_B", out["bound_B"], RESIDUAL_BOUND_MAX, RESIDUAL_BOUND_MIN, RESIDUAL_BOUND_MAX
            )
            bound_B = float(out["bound_B"].detach().float().mean().cpu().item())
        else:
            out["bound_B"], _ = _sanitize_eval_scalar(
                f"{eval_tag}/parts/bound_B", float(out["bound_B"]), RESIDUAL_BOUND_MAX, RESIDUAL_BOUND_MIN, RESIDUAL_BOUND_MAX
            )
            bound_B = float(out["bound_B"])
    else:
        bound_B = float(RESIDUAL_BOUND_MAX)

    for key in ["base_delta", "adapt_delta"]:
        if isinstance(out.get(key), torch.Tensor):
            out[key], _ = _sanitize_eval_tensor(
                f"{eval_tag}/parts/{key}", out[key], 0.0, 0.0, EVAL_DELTA_SANITIZE_MAX
            )

    if original_bad > 0 and isinstance(out.get("base_delta"), torch.Tensor) and isinstance(out.get("adapt_delta"), torch.Tensor):
        correction, final_delta, uncorr = _bounded_residual_from_deltas(out["base_delta"], out["adapt_delta"], bound_B)
        out["uncorrected_residual"] = uncorr
        out["correction"] = correction
        out["final_delta"] = final_delta

    for key, replacement, min_value, max_value in [
        ("uncorrected_residual", 0.0, -EVAL_DELTA_SANITIZE_MAX, EVAL_DELTA_SANITIZE_MAX),
        ("correction", 0.0, -RESIDUAL_BOUND_MAX, RESIDUAL_BOUND_MAX),
        ("final_delta", 0.0, 0.0, EVAL_DELTA_SANITIZE_MAX),
    ]:
        if isinstance(out.get(key), torch.Tensor):
            out[key], _ = _sanitize_eval_tensor(f"{eval_tag}/parts/{key}", out[key], replacement, min_value, max_value)

    if original_bad > 0:
        _log_nonfinite_eval(f"! sanitized {original_bad} nonfinite residual component values [{eval_tag}/parts]")
    return out, original_bad


def _sanitize_eval_delta_tensor(eval_tag: str, pred_delta: torch.Tensor) -> Tuple[torch.Tensor, int]:
    return _sanitize_eval_tensor(f"{eval_tag}/pred_delta", pred_delta, 0.0, 0.0, EVAL_DELTA_SANITIZE_MAX)


def _node_loss_weights(target_delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (1.0 + torch.clamp(target_delta / 6.0, max=3.0)) * mask



def _weighted_mean(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(weights.sum(), min=1.0)
    return (x * weights).sum() / denom



def _weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return _weighted_mean(F.smooth_l1_loss(pred, target, reduction="none"), weights)



def _bounded_residual_from_deltas(base_delta: torch.Tensor, adapt_delta: torch.Tensor, bound_B: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bound = max(1e-6, float(bound_B))
    uncorr = adapt_delta - base_delta
    correction = bound * torch.tanh(uncorr / bound)
    final_delta = torch.clamp(base_delta + correction, min=0.0, max=float(PRED_DELTA_MAX))
    return correction, final_delta, uncorr



def _dataset_stage_id_from_path(dataset_pt: str) -> str:
    name = Path(dataset_pt).name
    if name.endswith("__merged.pt"):
        return name[:-len("__merged.pt")]
    if "__chunk_" in name:
        return name.split("__chunk_", 1)[0]
    return name.replace(".pt", "")



def _ensure_dataset_descriptor(dataset_pt: str, stage_id: Optional[str] = None) -> None:
    if not SAVE_TASK_DESCRIPTORS:
        return
    sid = stage_id or _dataset_stage_id_from_path(dataset_pt)
    if sid not in STAGE_BY_ID:
        return
    payload = torch.load(dataset_pt, map_location="cpu")
    changed = False
    desc = descriptor_for_stage(STAGE_BY_ID[sid])
    if payload.get("stage_id") != sid:
        payload["stage_id"] = sid
        changed = True
    if payload.get("task_descriptor") != desc:
        payload["task_descriptor"] = desc
        changed = True
    if changed:
        torch.save(payload, dataset_pt)
        vol.commit()



def _calibrate_residual_bound(base_model: CleanHeuristicModel, dataset: StageEpisodeDataset, device: str) -> Tuple[float, Dict[str, Any]]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )
    use_amp = (device == "cuda") and (_env_int("USE_AMP", 1) == 1)
    vals: List[torch.Tensor] = []
    base_model.eval()
    with torch.no_grad():
        for batch in loader:
            obs_seq = batch["obs_seq"].to(device)
            node_patch = batch["node_patch"].to(device)
            node_meta = batch["node_meta"].to(device)
            target_delta = batch["target_delta"].to(device)
            mask = batch["mask"].to(device) > 0.5
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    base_log = base_model(obs_seq, node_patch, node_meta)
            else:
                base_log = base_model(obs_seq, node_patch, node_meta)
            base_delta = _delta_from_log_delta(base_log)
            _assert_finite_tensor("calibrate/base_delta", base_delta)
            resid = (target_delta - base_delta).abs()
            if mask.any():
                vals.append(resid[mask].detach().float().cpu())
    if vals:
        arr = torch.cat(vals, dim=0).numpy()
    else:
        arr = np.zeros((1,), dtype=np.float32)
    pct = float(np.percentile(arr, RESIDUAL_BOUND_PCT)) if arr.size > 0 else float(RESIDUAL_BOUND_MIN)
    bound_B = float(np.clip(pct, RESIDUAL_BOUND_MIN, RESIDUAL_BOUND_MAX))
    abs_arr = np.abs(arr)
    clip_frac = float((abs_arr > bound_B).mean()) if abs_arr.size > 0 else 0.0
    stats = {
        "abs_resid_mean": float(abs_arr.mean()) if abs_arr.size > 0 else 0.0,
        "abs_resid_std": float(abs_arr.std()) if abs_arr.size > 0 else 0.0,
        "abs_resid_p95": float(np.percentile(abs_arr, 95.0)) if abs_arr.size > 0 else 0.0,
        "abs_resid_p99": float(np.percentile(abs_arr, 99.0)) if abs_arr.size > 0 else 0.0,
        "abs_resid_max": float(abs_arr.max()) if abs_arr.size > 0 else 0.0,
        "clip_frac": clip_frac,
        "bound_B": bound_B,
        "bound_percentile": float(RESIDUAL_BOUND_PCT),
        "bound_min": float(RESIDUAL_BOUND_MIN),
        "bound_max": float(RESIDUAL_BOUND_MAX),
    }
    return bound_B, stats


# -----------------------------------------------------------------------------
# v2 avgbase training override
# -----------------------------------------------------------------------------


def _run_training_loop(model: CleanHeuristicModel, cfg: BackboneConfig, loader: torch.utils.data.DataLoader,
                       params: List[nn.Parameter], opt: torch.optim.Optimizer, ckpt_path: str, final_path: str,
                       arm: str, model_name: str, stage_id: str, epochs: int, device: str,
                       metrics_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    start_epoch, best_loss = _resume_training_checkpoint(ckpt_path, model, opt)
    model.train()
    use_amp = (device == "cuda") and (_env_int("USE_AMP", 1) == 1)
    nonfinite_loss_count = 0

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        steps = 0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = _masked_log_delta_loss(model, batch, device)
            else:
                loss = _masked_log_delta_loss(model, batch, device)
            if not torch.isfinite(loss).all():
                nonfinite_loss_count += 1
                _assert_finite_tensor(f"train/{arm}/{model_name}/{stage_id}/loss", loss)
            loss.backward()
            _assert_finite_gradients(params, f"train/{arm}/{model_name}/{stage_id}")
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1
        epoch_loss = _require_finite_scalar("epoch_loss", epoch_loss / max(1, steps))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            metrics = {
                "best_loss": best_loss,
                "epoch": epoch,
                "arm": _storage_arm_name(arm),
                "stage_id": stage_id,
                "model_name": model_name,
                "nonfinite_loss_count": int(nonfinite_loss_count),
            }
            if metrics_extra:
                metrics.update(metrics_extra)
            _save_model_artifact(final_path, cfg, _storage_arm_name(arm), model_name, stage_id, model, metrics)
            vol.commit()
        torch.save({
            "epoch": epoch,
            "best_loss": best_loss,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "nonfinite_loss_count": int(nonfinite_loss_count),
        }, ckpt_path)
        vol.commit()
        print(f"[train][{_storage_arm_name(arm)}][{model_name}][{stage_id}] epoch {epoch+1}/{epochs} loss={epoch_loss:.6f} best={best_loss:.6f}")

    return {
        "ok": True,
        "model_name": model_name,
        "arm": _storage_arm_name(arm),
        "stage_id": stage_id,
        "path": final_path,
        "best_loss": best_loss,
        "nonfinite_loss_count": int(nonfinite_loss_count),
    }



def _train_avg_base_impl(model_name: str, dataset_paths: List[str], device: str, seed: int) -> Dict[str, Any]:
    vol.reload()
    _ensure_dirs()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    payloads_by_stage: Dict[str, Dict[str, Any]] = {}
    for p in dataset_paths:
        sid = Path(p).name.split("__merged.pt")[0]
        _ensure_dataset_descriptor(p, sid)
        payloads_by_stage[sid] = load_dataset(p)

    ds = BalancedMultiStageDataset(payloads_by_stage)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    cfg = MODEL_CONFIGS[model_name]
    model = CleanHeuristicModel(cfg).to(device)
    _set_fullft_trainable(model)
    params = _trainable_params(model)
    opt = torch.optim.AdamW(params, lr=LR_FULL * _lr_multiplier_for_model(model_name), weight_decay=WEIGHT_DECAY_FULL)

    ckpt_path = checkpoint_path("avgbase", model_name, MULTITASK_BASE_STAGE_ID)
    final_path = model_path("avgbase", model_name, MULTITASK_BASE_STAGE_ID)
    metrics_extra = {
        "train_stage_ids": ds.stage_ids,
        "stage_lengths": ds.stage_lengths,
        "balanced_epoch_len": len(ds),
        "task_descriptors": [descriptor_for_stage(STAGE_BY_ID[s]) for s in ds.stage_ids if s in STAGE_BY_ID],
        "primary_backbone": PRIMARY_BACKBONE,
        "lr_multiplier": _lr_multiplier_for_model(model_name),
    }
    return _run_training_loop(
        model=model,
        cfg=cfg,
        loader=loader,
        params=params,
        opt=opt,
        ckpt_path=ckpt_path,
        final_path=final_path,
        arm="avgbase",
        model_name=model_name,
        stage_id=MULTITASK_BASE_STAGE_ID,
        epochs=max(1, AVG_BASE_EPOCHS),
        device=device,
        metrics_extra=metrics_extra,
    )


# -----------------------------------------------------------------------------
# v2 residual-task LoRA expert training override
# -----------------------------------------------------------------------------


def _train_task_lora_impl(model_name: str, task_stage_id: str, dataset_pt: str, base_model_pt: str,
                          device: str, seed: int) -> Dict[str, Any]:
    vol.reload()
    _ensure_dirs()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    _ensure_dataset_descriptor(dataset_pt, task_stage_id)
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

    cfg = MODEL_CONFIGS[model_name]

    base_payload = _load_model_artifact(base_model_pt, map_location="cpu")
    base_model = CleanHeuristicModel(BackboneConfig(**base_payload["cfg"]))
    base_model.load_state_dict(base_payload["model_state"], strict=False)
    base_model.to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    adapt_model = CleanHeuristicModel(cfg).to(device)
    _apply_stacked_lora(adapt_model, cfg.lora_rank, LORA_ALPHA, 1, include_conv=LORA_ON_CONV, include_attn=LORA_ON_ATTN, init_scale=LORA_INIT_SCALE)
    _load_plain_state_into_parametrized_model(
        adapt_model,
        base_payload["model_state"],
        context=f"residtasklora/{model_name}/{task_stage_id}/from_avgbase",
    )
    _set_lora_trainable(adapt_model, adapter_idx=0, train_bias=LORA_TRAIN_BIAS)
    params = _trainable_params(adapt_model)
    if not params:
        raise RuntimeError(f"no trainable parameters for residtasklora/{model_name}/{task_stage_id}")
    opt = torch.optim.AdamW(params, lr=LR_LORA * _lr_multiplier_for_model(model_name), weight_decay=WEIGHT_DECAY_LORA)

    ckpt_path = checkpoint_path("tasklora", model_name, task_stage_id)
    final_path = model_path("tasklora", model_name, task_stage_id)

    bound_B, calib_stats = _calibrate_residual_bound(base_model, ds, device)
    start_epoch = 0
    best_loss = float("inf")
    nonfinite_loss_count = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        adapt_model.load_state_dict(ckpt["model"], strict=False)
        opt.load_state_dict(ckpt["opt"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_loss = float(ckpt.get("best_loss", float("inf")))
        bound_B = float(ckpt.get("bound_B", bound_B))
        nonfinite_loss_count = int(ckpt.get("nonfinite_loss_count", 0))

    use_amp = (device == "cuda") and (_env_int("USE_AMP", 1) == 1)
    task_desc = descriptor_for_stage(STAGE_BY_ID[task_stage_id]) if task_stage_id in STAGE_BY_ID else {"stage_id": task_stage_id}
    train_epochs = max(1, STAGE_BY_ID[task_stage_id].train_epochs or EPOCHS_DEFAULT)

    for epoch in range(start_epoch, train_epochs):
        adapt_model.train()
        epoch_loss = 0.0
        epoch_resid = 0.0
        epoch_total = 0.0
        epoch_mag = 0.0
        steps = 0
        for batch in loader:
            obs_seq = batch["obs_seq"].to(device)
            node_patch = batch["node_patch"].to(device)
            node_meta = batch["node_meta"].to(device)
            target_delta = batch["target_delta"].to(device)
            mask = batch["mask"].to(device)
            weights = _node_loss_weights(target_delta, mask)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        base_log = base_model(obs_seq, node_patch, node_meta)
                        base_delta = _delta_from_log_delta(base_log)
                else:
                    base_log = base_model(obs_seq, node_patch, node_meta)
                    base_delta = _delta_from_log_delta(base_log)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    adapt_log = adapt_model(obs_seq, node_patch, node_meta)
                    adapt_delta = _delta_from_log_delta(adapt_log)
                    correction, final_delta, uncorr = _bounded_residual_from_deltas(base_delta.detach(), adapt_delta, bound_B)
                    residual_target_raw = target_delta - base_delta.detach()
                    residual_target = residual_target_raw.clamp(min=-bound_B, max=bound_B)
                    loss_resid = _weighted_smooth_l1(correction, residual_target, weights)
                    loss_total = _weighted_smooth_l1(final_delta, target_delta, weights)
                    loss_mag = _weighted_mean((correction / max(bound_B, 1e-6)) ** 2, weights)
                    loss = EXPERT_LOSS_RESID_W * loss_resid + EXPERT_LOSS_TOTAL_W * loss_total + EXPERT_LOSS_MAG_W * loss_mag
            else:
                adapt_log = adapt_model(obs_seq, node_patch, node_meta)
                adapt_delta = _delta_from_log_delta(adapt_log)
                correction, final_delta, uncorr = _bounded_residual_from_deltas(base_delta.detach(), adapt_delta, bound_B)
                residual_target_raw = target_delta - base_delta.detach()
                residual_target = residual_target_raw.clamp(min=-bound_B, max=bound_B)
                loss_resid = _weighted_smooth_l1(correction, residual_target, weights)
                loss_total = _weighted_smooth_l1(final_delta, target_delta, weights)
                loss_mag = _weighted_mean((correction / max(bound_B, 1e-6)) ** 2, weights)
                loss = EXPERT_LOSS_RESID_W * loss_resid + EXPERT_LOSS_TOTAL_W * loss_total + EXPERT_LOSS_MAG_W * loss_mag

            _assert_finite_tensor(f"train/residtasklora/{model_name}/{task_stage_id}/base_delta", base_delta)
            _assert_finite_tensor(f"train/residtasklora/{model_name}/{task_stage_id}/adapt_delta", adapt_delta)
            _assert_finite_tensor(f"train/residtasklora/{model_name}/{task_stage_id}/correction", correction)
            _assert_finite_tensor(f"train/residtasklora/{model_name}/{task_stage_id}/final_delta", final_delta)
            if not torch.isfinite(loss).all():
                nonfinite_loss_count += 1
                _assert_finite_tensor(f"train/residtasklora/{model_name}/{task_stage_id}/loss", loss)
            loss.backward()
            _assert_finite_gradients(params, f"train/residtasklora/{model_name}/{task_stage_id}")
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
            opt.step()
            epoch_loss += float(loss.item())
            epoch_resid += float(loss_resid.item())
            epoch_total += float(loss_total.item())
            epoch_mag += float(loss_mag.item())
            steps += 1

        epoch_loss = _require_finite_scalar("residtasklora_epoch_loss", epoch_loss / max(1, steps))
        epoch_resid = _require_finite_scalar("residtasklora_epoch_resid", epoch_resid / max(1, steps))
        epoch_total = _require_finite_scalar("residtasklora_epoch_total", epoch_total / max(1, steps))
        epoch_mag = _require_finite_scalar("residtasklora_epoch_mag", epoch_mag / max(1, steps))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            metrics = {
                "best_loss": best_loss,
                "epoch": epoch,
                "arm": "residtasklora",
                "stage_id": task_stage_id,
                "model_name": model_name,
                "base_model_path": base_model_pt,
                "train_stage_id": task_stage_id,
                "task_descriptor": task_desc,
                "bound_B": float(bound_B),
                "residual_bound_stats": calib_stats,
                "lora_rank": cfg.lora_rank,
                "lora_alpha": LORA_ALPHA,
                "lr_multiplier": _lr_multiplier_for_model(model_name),
                "epoch_loss_resid": epoch_resid,
                "epoch_loss_total": epoch_total,
                "epoch_loss_mag": epoch_mag,
                "nonfinite_loss_count": int(nonfinite_loss_count),
            }
            _save_model_artifact(final_path, cfg, "residtasklora", model_name, task_stage_id, adapt_model, metrics)
            vol.commit()

        torch.save({
            "epoch": epoch,
            "best_loss": best_loss,
            "model": adapt_model.state_dict(),
            "opt": opt.state_dict(),
            "bound_B": float(bound_B),
            "nonfinite_loss_count": int(nonfinite_loss_count),
        }, ckpt_path)
        vol.commit()
        print(
            f"[train][residtasklora][{model_name}][{task_stage_id}] epoch {epoch+1}/{train_epochs} "
            f"loss={epoch_loss:.6f} resid={epoch_resid:.6f} total={epoch_total:.6f} mag={epoch_mag:.6f} "
            f"B={bound_B:.3f} best={best_loss:.6f}"
        )

    return {
        "ok": True,
        "model_name": model_name,
        "arm": "residtasklora",
        "stage_id": task_stage_id,
        "path": final_path,
        "best_loss": best_loss,
        "bound_B": float(bound_B),
        "nonfinite_loss_count": int(nonfinite_loss_count),
    }


# -----------------------------------------------------------------------------
# v2 eval wrapper + diagnostics
# -----------------------------------------------------------------------------


class BoundedResidualExpertPolicy:
    def __init__(self, base_model: CleanHeuristicModel, adapt_model: CleanHeuristicModel, bound_B: float,
                 metadata: Optional[Dict[str, Any]] = None):
        self.base_model = base_model
        self.adapt_model = adapt_model
        self.bound_B = float(bound_B)
        self.metadata = metadata or {}
        self.arm = "residtasklora"

    def init_context_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Any:
        return {
            "base": self.base_model.init_context_state(batch_size, device, dtype),
            "adapt": self.adapt_model.init_context_state(batch_size, device, dtype),
        }

    def step_context(self, frame_t: torch.Tensor, state: Any, t_idx: int) -> Tuple[Dict[str, torch.Tensor], Any]:
        base_ctx, base_state = self.base_model.step_context(frame_t, state["base"], t_idx)
        adapt_ctx, adapt_state = self.adapt_model.step_context(frame_t, state["adapt"], t_idx)
        return {"base": base_ctx, "adapt": adapt_ctx}, {"base": base_state, "adapt": adapt_state}

    def encode_obs_sequence(self, obs_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "base": self.base_model.encode_obs_sequence(obs_seq),
            "adapt": self.adapt_model.encode_obs_sequence(obs_seq),
        }

    def predict_components_from_ctx(self, ctx: Any, node_patch: torch.Tensor, node_meta: torch.Tensor) -> Dict[str, torch.Tensor]:
        base_delta = self.base_model.predict_delta_from_ctx(ctx["base"], node_patch, node_meta)
        adapt_delta = self.adapt_model.predict_delta_from_ctx(ctx["adapt"], node_patch, node_meta)
        correction, final_delta, uncorr = _bounded_residual_from_deltas(base_delta, adapt_delta, self.bound_B)
        return {
            "base_delta": base_delta,
            "adapt_delta": adapt_delta,
            "uncorrected_residual": uncorr,
            "correction": correction,
            "final_delta": final_delta,
            "bound_B": torch.tensor(self.bound_B, dtype=final_delta.dtype, device=final_delta.device),
        }

    def predict_delta_from_ctx(self, ctx: Any, node_patch: torch.Tensor, node_meta: torch.Tensor) -> torch.Tensor:
        return self.predict_components_from_ctx(ctx, node_patch, node_meta)["final_delta"]



def _load_model_for_eval(model_path_str: str, device: str):
    payload = _load_model_artifact(model_path_str, map_location="cpu")
    cfg = BackboneConfig(**payload["cfg"])
    arm = _storage_arm_name(payload.get("arm", "avgbase"))
    metrics = payload.get("metrics", {})

    if arm == "residtasklora":
        base_model_path_str = metrics.get("base_model_path")
        if not base_model_path_str:
            raise RuntimeError(f"expert artifact missing base_model_path: {model_path_str}")
        base_payload = _load_model_artifact(base_model_path_str, map_location="cpu")
        base_cfg = BackboneConfig(**base_payload["cfg"])
        base_model = CleanHeuristicModel(base_cfg)
        base_model.load_state_dict(base_payload["model_state"], strict=False)
        base_model.to(device)
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad = False

        adapt_model = CleanHeuristicModel(cfg)
        _apply_stacked_lora(adapt_model, cfg.lora_rank, LORA_ALPHA, 1, include_conv=LORA_ON_CONV, include_attn=LORA_ON_ATTN, init_scale=LORA_INIT_SCALE)
        adapt_model.load_state_dict(payload["model_state"], strict=False)
        adapt_model.to(device)
        adapt_model.eval()
        for p in adapt_model.parameters():
            p.requires_grad = False

        return BoundedResidualExpertPolicy(
            base_model=base_model,
            adapt_model=adapt_model,
            bound_B=float(metrics.get("bound_B", RESIDUAL_BOUND_MAX)),
            metadata=metrics,
        )

    model = CleanHeuristicModel(cfg)
    if arm in ("tasklora",):
        _apply_stacked_lora(model, cfg.lora_rank, LORA_ALPHA, 1, include_conv=LORA_ON_CONV, include_attn=LORA_ON_ATTN, init_scale=LORA_INIT_SCALE)
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
        "base_sum": 0.0,
        "base_sq_sum": 0.0,
        "base_count": 0,
        "expert_correction_sum": 0.0,
        "expert_correction_sq_sum": 0.0,
        "expert_correction_count": 0,
        "expert_correction_abs_max": 0.0,
        "expert_correction_saturation_count": 0,
        "uncorr_abs_max": 0.0,
        "residual_target_sum": 0.0,
        "residual_target_sq_sum": 0.0,
        "residual_target_count": 0,
        "residual_target_clip_count": 0,
        "bound_B_sum": 0.0,
        "bound_B_count": 0,
        "nonfinite_pred_count": 0,
        "corr_abs_hist": [0 for _ in range(DIAG_HIST_BINS)],
        "uncorr_abs_hist": [0 for _ in range(DIAG_HIST_BINS)],
    }



def _upgrade_diag(diag: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = _new_diag_accumulator()
    if not diag:
        return base
    out = dict(base)
    for k, v in diag.items():
        out[k] = v
    if not isinstance(out.get("bucket_counts"), dict):
        out["bucket_counts"] = {"small": 0, "medium": 0, "large": 0}
    if not isinstance(out.get("bucket_pred_sum"), dict):
        out["bucket_pred_sum"] = {"small": 0.0, "medium": 0.0, "large": 0.0}
    if not isinstance(out.get("corr_abs_hist"), list):
        out["corr_abs_hist"] = [0 for _ in range(DIAG_HIST_BINS)]
    if not isinstance(out.get("uncorr_abs_hist"), list):
        out["uncorr_abs_hist"] = [0 for _ in range(DIAG_HIST_BINS)]
    if len(out["corr_abs_hist"]) != DIAG_HIST_BINS:
        out["corr_abs_hist"] = (list(out["corr_abs_hist"]) + [0] * DIAG_HIST_BINS)[:DIAG_HIST_BINS]
    if len(out["uncorr_abs_hist"]) != DIAG_HIST_BINS:
        out["uncorr_abs_hist"] = (list(out["uncorr_abs_hist"]) + [0] * DIAG_HIST_BINS)[:DIAG_HIST_BINS]
    return out



def _hist_update(counts: List[int], values: Sequence[float], max_value: float) -> None:
    if max_value <= 0:
        return
    scale = float(DIAG_HIST_BINS - 1) / float(max_value)
    for v in values:
        fv = abs(float(v))
        if not math.isfinite(fv):
            continue
        idx = int(min(DIAG_HIST_BINS - 1, max(0, math.floor(fv * scale))))
        counts[idx] += 1



def _hist_percentile(counts: Sequence[int], max_value: float, q: float) -> float:
    total = int(sum(int(c) for c in counts))
    if total <= 0:
        return 0.0
    threshold = max(1, int(math.ceil(float(q) * total)))
    running = 0
    for idx, c in enumerate(counts):
        running += int(c)
        if running >= threshold:
            return float(max_value) * float(idx) / max(1.0, float(DIAG_HIST_BINS - 1))
    return float(max_value)



def _merge_diag(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    src = _upgrade_diag(src)
    dst_u = _upgrade_diag(dst)
    dst.clear()
    dst.update(dst_u)
    numeric_sum_keys = [
        "pred_sum", "pred_sq_sum", "pred_pos_count", "pred_count",
        "target_sum", "target_sq_sum", "target_count",
        "corr_sum_xy", "corr_sum_x", "corr_sum_y", "corr_sum_x2", "corr_sum_y2", "corr_count",
        "ordering_sets", "ordering_changed_sets", "rank_disp_sum",
        "base_sum", "base_sq_sum", "base_count",
        "expert_correction_sum", "expert_correction_sq_sum", "expert_correction_count",
        "expert_correction_saturation_count",
        "residual_target_sum", "residual_target_sq_sum", "residual_target_count", "residual_target_clip_count",
        "bound_B_sum", "bound_B_count", "nonfinite_pred_count",
    ]
    for k in numeric_sum_keys:
        dst[k] = dst.get(k, 0) + src.get(k, 0)
    dst["pred_max"] = max(float(dst.get("pred_max", 0.0)), float(src.get("pred_max", 0.0)))
    dst["expert_correction_abs_max"] = max(float(dst.get("expert_correction_abs_max", 0.0)), float(src.get("expert_correction_abs_max", 0.0)))
    dst["uncorr_abs_max"] = max(float(dst.get("uncorr_abs_max", 0.0)), float(src.get("uncorr_abs_max", 0.0)))
    dst["high_residual_seen"] = bool(dst.get("high_residual_seen", False) or src.get("high_residual_seen", False))
    for b in ["small", "medium", "large"]:
        dst["bucket_counts"][b] = int(dst["bucket_counts"].get(b, 0)) + int(src.get("bucket_counts", {}).get(b, 0))
        dst["bucket_pred_sum"][b] = float(dst["bucket_pred_sum"].get(b, 0.0)) + float(src.get("bucket_pred_sum", {}).get(b, 0.0))
    dst["corr_abs_hist"] = [int(a) + int(b) for a, b in zip(dst.get("corr_abs_hist", [0] * DIAG_HIST_BINS), src.get("corr_abs_hist", [0] * DIAG_HIST_BINS))]
    dst["uncorr_abs_hist"] = [int(a) + int(b) for a, b in zip(dst.get("uncorr_abs_hist", [0] * DIAG_HIST_BINS), src.get("uncorr_abs_hist", [0] * DIAG_HIST_BINS))]



def _safe_mean(total: float, count: int) -> float:
    return float(total) / max(1, int(count))



def _safe_std(total: float, sq_total: float, count: int) -> float:
    c = max(1, int(count))
    mean = float(total) / c
    var = max(0.0, float(sq_total) / c - mean * mean)
    return math.sqrt(var)



def _safe_corrcoef(diag: Dict[str, Any]) -> float:
    n = int(diag.get("corr_count", 0))
    if n <= 1:
        return 0.0
    num = float(n) * float(diag["corr_sum_xy"]) - float(diag["corr_sum_x"]) * float(diag["corr_sum_y"])
    den_x = float(n) * float(diag["corr_sum_x2"]) - float(diag["corr_sum_x"]) ** 2
    den_y = float(n) * float(diag["corr_sum_y2"]) - float(diag["corr_sum_y"]) ** 2
    if den_x <= 1e-12 or den_y <= 1e-12:
        return 0.0
    out = num / math.sqrt(den_x * den_y)
    if not math.isfinite(out):
        return 0.0
    return float(max(-1.0, min(1.0, out)))



def _finalize_diag(diag: Dict[str, Any]) -> Dict[str, Any]:
    diag = _upgrade_diag(diag)
    out = dict(diag)
    out["pred_mean"] = _safe_mean(diag["pred_sum"], diag["pred_count"])
    out["pred_std"] = _safe_std(diag["pred_sum"], diag["pred_sq_sum"], diag["pred_count"])
    out["pred_positive_frac"] = float(diag["pred_pos_count"]) / max(1, int(diag["pred_count"]))
    out["target_mean"] = _safe_mean(diag["target_sum"], diag["target_count"])
    out["target_std"] = _safe_std(diag["target_sum"], diag["target_sq_sum"], diag["target_count"])
    out["pred_target_corr"] = _safe_corrcoef(diag)
    out["ordering_change_frac"] = float(diag["ordering_changed_sets"]) / max(1, int(diag["ordering_sets"]))
    out["avg_rank_displacement"] = float(diag["rank_disp_sum"]) / max(1, int(diag["ordering_sets"]))
    out["bucket_pred_mean"] = {
        b: float(diag["bucket_pred_sum"][b]) / max(1, int(diag["bucket_counts"][b]))
        for b in ["small", "medium", "large"]
    }
    out["base_delta_mean"] = _safe_mean(diag["base_sum"], diag["base_count"])
    out["base_delta_std"] = _safe_std(diag["base_sum"], diag["base_sq_sum"], diag["base_count"])
    out["correction_mean"] = _safe_mean(diag["expert_correction_sum"], diag["expert_correction_count"])
    out["correction_std"] = _safe_std(diag["expert_correction_sum"], diag["expert_correction_sq_sum"], diag["expert_correction_count"])
    out["correction_abs_p95"] = _hist_percentile(diag["corr_abs_hist"], RESIDUAL_BOUND_MAX, 0.95)
    out["correction_abs_max"] = float(diag["expert_correction_abs_max"])
    out["correction_saturation_frac"] = float(diag["expert_correction_saturation_count"]) / max(1, int(diag["expert_correction_count"]))
    out["uncorrected_residual_abs_p95"] = _hist_percentile(diag["uncorr_abs_hist"], UNCORR_HIST_MAX, 0.95)
    out["uncorrected_residual_abs_max"] = float(diag["uncorr_abs_max"])
    out["residual_target_mean"] = _safe_mean(diag["residual_target_sum"], diag["residual_target_count"])
    out["residual_target_std"] = _safe_std(diag["residual_target_sum"], diag["residual_target_sq_sum"], diag["residual_target_count"])
    out["residual_target_clip_frac"] = float(diag["residual_target_clip_count"]) / max(1, int(diag["residual_target_count"]))
    out["bound_B"] = _safe_mean(diag["bound_B_sum"], diag["bound_B_count"])
    out["nonfinite_pred_count"] = int(diag["nonfinite_pred_count"])
    for key in [
        "pred_mean", "pred_std", "pred_positive_frac", "target_mean", "target_std", "pred_target_corr",
        "ordering_change_frac", "avg_rank_displacement", "base_delta_mean", "base_delta_std",
        "correction_mean", "correction_std", "correction_abs_p95", "correction_abs_max",
        "correction_saturation_frac", "uncorrected_residual_abs_p95", "uncorrected_residual_abs_max",
        "residual_target_mean", "residual_target_std", "residual_target_clip_frac", "bound_B",
    ]:
        out[key] = _require_finite_scalar(key, out[key])
    return out



def _diagnostics_update(diag: Dict[str, Any], target_deltas: List[float], pred_deltas: List[float], alpha: float, h_bases: List[int],
                        base_deltas: Optional[List[float]] = None, corrections: Optional[List[float]] = None,
                        uncorr_residuals: Optional[List[float]] = None, bound_B: Optional[float] = None) -> None:
    n = len(pred_deltas)
    if n == 0:
        return
    diag = _upgrade_diag(diag)
    for i, (t, p) in enumerate(zip(target_deltas, pred_deltas)):
        if (not math.isfinite(float(t))) or (not math.isfinite(float(p))):
            diag["nonfinite_pred_count"] += 1
            if ABORT_ON_NONFINITE:
                raise FloatingPointError(f"nonfinite eval prediction/target t={t} p={p}")
            continue
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

        if base_deltas is not None and corrections is not None and uncorr_residuals is not None:
            base_v = float(base_deltas[i])
            corr_v = float(corrections[i])
            uncorr_v = float(uncorr_residuals[i])
            if not (math.isfinite(base_v) and math.isfinite(corr_v) and math.isfinite(uncorr_v)):
                diag["nonfinite_pred_count"] += 1
                if ABORT_ON_NONFINITE:
                    raise FloatingPointError(f"nonfinite expert eval component base={base_v} corr={corr_v} uncorr={uncorr_v}")
                continue
            diag["base_sum"] += base_v
            diag["base_sq_sum"] += base_v ** 2
            diag["base_count"] += 1
            diag["expert_correction_sum"] += corr_v
            diag["expert_correction_sq_sum"] += corr_v ** 2
            diag["expert_correction_count"] += 1
            abs_corr = abs(corr_v)
            diag["expert_correction_abs_max"] = max(float(diag["expert_correction_abs_max"]), abs_corr)
            diag["uncorr_abs_max"] = max(float(diag["uncorr_abs_max"]), abs(uncorr_v))
            if bound_B is not None:
                bb = float(bound_B)
                diag["bound_B_sum"] += bb
                diag["bound_B_count"] += 1
                if abs_corr >= CORRECTION_SAT_THRESH_FRAC * bb:
                    diag["expert_correction_saturation_count"] += 1
                residual_target_raw = float(t) - base_v
                residual_target = max(-bb, min(bb, residual_target_raw))
                diag["residual_target_sum"] += residual_target
                diag["residual_target_sq_sum"] += residual_target ** 2
                diag["residual_target_count"] += 1
                diag["residual_target_clip_count"] += int(abs(residual_target_raw) > bb + 1e-8)
            _hist_update(diag["corr_abs_hist"], [abs_corr], RESIDUAL_BOUND_MAX)
            _hist_update(diag["uncorr_abs_hist"], [abs(uncorr_v)], UNCORR_HIST_MAX)
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



def run_policy_episode(suite: EvalSuite, seed: int, model: Optional[Any], alpha: float,
                       max_expansions: int, device: str) -> Dict[str, Any]:
    ep = make_episode(seed, suite.family, suite.size, suite.max_steps, suite.n_gates, suite.n_patrollers, suite.n_drifters)
    occ = simulate_occupancy(ep.walls, ep.gates, ep.pats, ep.drifts, ep.max_steps)
    dist_abs = compute_true_cost_to_goal(occ["blocked"], ep.goal, ep.max_steps) if EVAL_DIAG else None
    static_template = make_static_template(ep.walls, ep.goal)
    agent_xy = ep.start
    done = False
    last_info = {"reached": False, "collided": False}
    total_expansions = 0
    diag_acc = _new_diag_accumulator()
    frame_history: List[np.ndarray] = []
    model_tag = "baseline" if model is None else getattr(model, "arm", "avgbase")

    for t_abs in range(ep.max_steps):
        frame = build_step_frame(static_template, agent_xy, occ["gate"][t_abs], occ["pat"][t_abs], occ["drift"][t_abs])
        eval_tag = f"eval/{model_tag}/{suite.suite_id}/seed={seed}/B={max_expansions}/t={t_abs}"
        if model is not None:
            frame_history.append(frame)
            if len(frame_history) > HISTORY_LEN:
                frame_history.pop(0)
            # Match training-time context encoding by re-encoding the latest fixed history window.
            obs_seq_t = torch.from_numpy(_stack_frame_history(frame_history, HISTORY_LEN)).unsqueeze(0).to(device)
            with torch.no_grad():
                ctx = model.encode_obs_sequence(obs_seq_t)
            _assert_finite_eval_value(f"{eval_tag}/ctx", ctx)
        else:
            ctx = None

        dynamic_cur = np.clip(occ["gate"][t_abs] + occ["pat"][t_abs] + occ["drift"][t_abs], 0, 1).astype(np.uint8)
        gx, gy = ep.goal
        delta_cache: Dict[Tuple[int, int, int], float] = {}

        def heuristic_delta_batch_fn(states: List[Tuple[int, int, int]]) -> List[float]:
            if model is None:
                pred = [0.0 for _ in states]
                if EVAL_DIAG:
                    h_bases = [manhattan(x, y, gx, gy) for x, y, _ in states]
                    target_deltas = []
                    for x, y, t_rel in states:
                        tgt = compute_target_delta_from_dist(dist_abs, min(t_abs + t_rel, ep.max_steps), x, y, gx, gy)
                        target_deltas.append(0.0 if tgt is None else float(tgt))
                    _diagnostics_update(diag_acc, target_deltas, pred, alpha, h_bases)
                return pred

            # ---- Fast path: cache NN delta per (x,y,t_rel) within this replan ----
            if not EVAL_DIAG:
                out: List[Optional[float]] = [None] * len(states)
                todo_idx: List[int] = []
                todo_states: List[Tuple[int, int, int]] = []
                for i, s in enumerate(states):
                    c = delta_cache.get(s)
                    if c is None:
                        todo_idx.append(i)
                        todo_states.append(s)
                    else:
                        out[i] = c
                if todo_states:
                    p = 2 * PATCH_RADIUS + 1
                    patches = np.zeros((1, len(todo_states), PATCH_CHANNELS, p, p), dtype=np.float32)
                    metas = np.zeros((1, len(todo_states), NODE_META_DIM), dtype=np.float32)
                    for j, (x, y, t_rel) in enumerate(todo_states):
                        patches[0, j] = extract_local_patch_2ch(ep.walls, dynamic_cur, x, y, PATCH_RADIUS).astype(np.float32)
                        metas[0, j] = build_node_meta(x, y, gx, gy, t_rel, ep.walls.shape[0])
                    patch_t = torch.from_numpy(patches).to(device)
                    meta_t = torch.from_numpy(metas).to(device)
                    _assert_finite_tensor(f"{eval_tag}/patch_t", patch_t)
                    _assert_finite_tensor(f"{eval_tag}/meta_t", meta_t)
                    with torch.no_grad():
                        if hasattr(model, "predict_components_from_ctx"):
                            parts = model.predict_components_from_ctx(ctx, patch_t, meta_t)
                            if SANITIZE_NONFINITE_EVAL:
                                parts, _ = _sanitize_residual_parts_for_eval(eval_tag, parts)
                            else:
                                _assert_finite_eval_value(f"{eval_tag}/parts", parts)
                            pred_t = parts["final_delta"]
                        else:
                            pred_t = model.predict_delta_from_ctx(ctx, patch_t, meta_t)
                            if SANITIZE_NONFINITE_EVAL:
                                pred_t, _ = _sanitize_eval_delta_tensor(eval_tag, pred_t)
                            else:
                                _assert_finite_tensor(f"{eval_tag}/pred_delta", pred_t)
                        vals = [float(v) for v in pred_t[0].detach().float().cpu().numpy().tolist()]
                    for j, s in enumerate(todo_states):
                        delta_cache[s] = vals[j]
                        out[todo_idx[j]] = vals[j]
                return [float(v) for v in out]

            # ---- Diagnostics-on path: unchanged behavior (no cache) ----
            h_bases = [manhattan(x, y, gx, gy) for x, y, _ in states]
            target_deltas = []
            for x, y, t_rel in states:
                tgt = compute_target_delta_from_dist(dist_abs, min(t_abs + t_rel, ep.max_steps), x, y, gx, gy)
                target_deltas.append(0.0 if tgt is None else float(tgt))
            p = 2 * PATCH_RADIUS + 1
            patches = np.zeros((1, len(states), PATCH_CHANNELS, p, p), dtype=np.float32)
            metas = np.zeros((1, len(states), NODE_META_DIM), dtype=np.float32)
            for i, (x, y, t_rel) in enumerate(states):
                patches[0, i] = extract_local_patch_2ch(ep.walls, dynamic_cur, x, y, PATCH_RADIUS).astype(np.float32)
                metas[0, i] = build_node_meta(x, y, gx, gy, t_rel, ep.walls.shape[0])
            patch_t = torch.from_numpy(patches).to(device)
            meta_t = torch.from_numpy(metas).to(device)
            _assert_finite_tensor(f"{eval_tag}/patch_t", patch_t)
            _assert_finite_tensor(f"{eval_tag}/meta_t", meta_t)
            with torch.no_grad():
                if hasattr(model, "predict_components_from_ctx"):
                    parts = model.predict_components_from_ctx(ctx, patch_t, meta_t)
                    if SANITIZE_NONFINITE_EVAL:
                        parts, nonfinite_component_count = _sanitize_residual_parts_for_eval(eval_tag, parts)
                        diag_acc["nonfinite_pred_count"] += int(nonfinite_component_count)
                    else:
                        _assert_finite_eval_value(f"{eval_tag}/parts", parts)
                    pred_t = parts["final_delta"][0].detach().float().cpu().numpy().tolist()
                    base_t = parts["base_delta"][0].detach().float().cpu().numpy().tolist()
                    corr_t = parts["correction"][0].detach().float().cpu().numpy().tolist()
                    uncorr_t = parts["uncorrected_residual"][0].detach().float().cpu().numpy().tolist()
                    bound_B = float(parts["bound_B"].detach().float().mean().cpu().item()) if isinstance(parts["bound_B"], torch.Tensor) else float(parts["bound_B"])
                    bound_B = _require_finite_scalar(f"{eval_tag}/bound_B", bound_B)
                    pred = [float(v) for v in pred_t]
                    base_vals = [float(v) for v in base_t]
                    corr_vals = [float(v) for v in corr_t]
                    uncorr_vals = [float(v) for v in uncorr_t]
                    _diagnostics_update(diag_acc, target_deltas, pred, alpha, h_bases, base_vals, corr_vals, uncorr_vals, bound_B)
                else:
                    pred_delta_t = model.predict_delta_from_ctx(ctx, patch_t, meta_t)
                    if SANITIZE_NONFINITE_EVAL:
                        pred_delta_t, nonfinite_pred_count = _sanitize_eval_delta_tensor(eval_tag, pred_delta_t)
                        diag_acc["nonfinite_pred_count"] += int(nonfinite_pred_count)
                    else:
                        _assert_finite_tensor(f"{eval_tag}/pred_delta", pred_delta_t)
                    pred_t = pred_delta_t[0].detach().float().cpu().numpy().tolist()
                    pred = [float(v) for v in pred_t]
                    _diagnostics_update(diag_acc, target_deltas, pred, alpha, h_bases)
            return pred

        if PLANNER == "focal":
            plan = space_time_focal_astar(agent_xy, ep.goal, t_abs, suite.plan_horizon, max_expansions, occ, heuristic_delta_batch_fn, w=FOCAL_W)
        else:
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
    _refresh_eval_diag_from_env()
    vol.reload()
    _ensure_dirs()
    _configure_eval_torch_threads()
    suite = EVAL_SUITE_BY_ID[suite_id]
    out_path = eval_shard_path(model_eval_id, suite_id, budget, alpha, total_episodes, ep_start, ep_count)
    force_reeval = FORCE_REEVAL or suite_id in FORCE_REEVAL_SUITE_IDS or (FORCE_REEVAL_MODELED and bool(model_path_str))
    existing = None if force_reeval else _read_json_safe(out_path)
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
            metric_sums["diag"] = _upgrade_diag(metric_sums.get("diag"))
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



def _merge_metric_sums_payload(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    dst["successes"] += int(src.get("successes", 0))
    dst["timeouts"] += int(src.get("timeouts", 0))
    dst["collisions"] += int(src.get("collisions", 0))
    dst["steps"] += int(src.get("steps", 0))
    dst["expansions"] += int(src.get("expansions", 0))
    dst["high_residual_eps"] += int(src.get("high_residual_eps", 0))
    dst["high_residual_successes"] += int(src.get("high_residual_successes", 0))
    _merge_diag(dst["diag"], src.get("diag", _new_diag_accumulator()))



def _artifact_metrics_from_job(job: Dict[str, Any]) -> Dict[str, Any]:
    if not job.get("model_path"):
        return {}
    try:
        return _load_model_artifact(job["model_path"], map_location="cpu").get("metrics", {})
    except Exception:
        return {}



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
    storage_arm = _storage_arm_name(job.get("arm", "baseline"))
    suite_desc = descriptor_for_suite(EVAL_SUITE_BY_ID[job["suite_id"]]) if job["suite_id"] in EVAL_SUITE_BY_ID else {}
    stage_desc = _stage_descriptor_for_id(job.get("stage_id", "baseline"))
    artifact_metrics = _artifact_metrics_from_job(job)
    row = {
        "model": job["display_name"],
        "arm": storage_arm,
        "backbone": job.get("model_name", "baseline"),
        "stage": job.get("stage_id", "baseline"),
        "suite": job["suite_id"],
        "suite_family": suite_desc.get("family", ""),
        "suite_size": suite_desc.get("size", -1),
        "suite_dynamics": suite_desc.get("dynamics", ""),
        "suite_max_steps": suite_desc.get("max_steps", -1),
        "suite_plan_horizon": suite_desc.get("plan_horizon", -1),
        "suite_n_gates": suite_desc.get("n_gates", -1),
        "suite_n_patrollers": suite_desc.get("n_patrollers", -1),
        "suite_n_drifters": suite_desc.get("n_drifters", -1),
        "train_family": stage_desc.get("family", ""),
        "train_size": stage_desc.get("size", -1),
        "train_dynamics": stage_desc.get("dynamics", ""),
        "train_max_steps": stage_desc.get("max_steps", -1),
        "train_plan_horizon": stage_desc.get("plan_horizon", -1),
        "train_n_gates": stage_desc.get("n_gates", -1),
        "train_n_patrollers": stage_desc.get("n_patrollers", -1),
        "train_n_drifters": stage_desc.get("n_drifters", -1),
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
        "pred_max": float(metric_sums["diag"].get("pred_max", 0.0)),
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
        "base_delta_mean": diag["base_delta_mean"],
        "base_delta_std": diag["base_delta_std"],
        "correction_mean": diag["correction_mean"],
        "correction_std": diag["correction_std"],
        "correction_abs_p95": diag["correction_abs_p95"],
        "correction_abs_max": diag["correction_abs_max"],
        "correction_saturation_frac": diag["correction_saturation_frac"],
        "uncorrected_residual_abs_p95": diag["uncorrected_residual_abs_p95"],
        "uncorrected_residual_abs_max": diag["uncorrected_residual_abs_max"],
        "residual_target_mean": diag["residual_target_mean"],
        "residual_target_std": diag["residual_target_std"],
        "residual_target_clip_frac": diag["residual_target_clip_frac"],
        "bound_B": diag["bound_B"] if storage_arm == "residtasklora" else float(artifact_metrics.get("bound_B", 0.0)),
        "nonfinite_pred_count": int(diag["nonfinite_pred_count"]),
        "nonfinite_loss_count": int(artifact_metrics.get("nonfinite_loss_count", 0)),
    }
    agg_payload = {"row": row, "metric_sums": metric_sums, "diag": diag, "complete": True}
    _write_json_atomic(eval_agg_path(job["model_eval_id"], job["suite_id"], job["budget"], job["alpha"], job["episodes"]), agg_payload)
    vol.commit()
    return row
