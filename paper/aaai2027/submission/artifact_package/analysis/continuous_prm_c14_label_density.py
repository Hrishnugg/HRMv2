#!/usr/bin/env python3
"""C14: label-count x world-diversity adaptation factorial.

Matches adaptation runs on SUPERVISED-STATE COUNT N (not map count) across the
static (C9 substrate, maze-dense) and dynamic (C9b substrate, maze-dense)
domains, at two diversity levels (concentrated = minimal world count w_min(N);
distributed = 8 x w_min(N) worlds), with rank-8 LoRA (unbounded) / full
fine-tuning / scratch at IDENTICAL total optimizer steps in every cell.

New-file-only; reuses C7/C8/C9/C9h/C9b machinery. Design (+ pre-execution
amendment v2 defining w_min-based diversity):
docs/experiments/continuous/c14/design/2026-07-23-c14-label-density-factorial.md
"""
from __future__ import annotations

import argparse
import csv as _csv
import dataclasses
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import continuous_prm_common as C
import continuous_prm_providers as P
import continuous_prm_c7_integration_compare as C7
import continuous_prm_c7_hard_maps as H7
import continuous_prm_c9_transfer as C9
import continuous_prm_c9h_transfer as C9H
import continuous_prm_c9b_dynamics_transfer as C9B
import continuous_prm_dynamic_providers as DP
import continuous_prm_c6_heatmap_value_field as C6


def now_str() -> str:
    return C.now_str()


def _torch_load(path) -> dict:
    """torch.load with weights_only=True when the payload permits it.

    These checkpoints are produced by this repo's own trainers (tensors +
    plain dicts), so the safe loader normally succeeds; older torch versions
    reject some legacy payload elements (e.g. numpy scalars) under
    weights_only, in which case we fall back to the default loader used by
    every incumbent loader in this codebase.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu")


def _parse_csv(s):
    return [t for t in str(s).split(",") if t != ""]


def _parse_ints(s):
    return [int(t) for t in _parse_csv(s)]


STATIC_TARGET = "C_hard_maze_dense"
DYNAMIC_TARGET = "C_dyn_maze_dense"
DIVERSITIES = ("conc", "dist")
METHODS = ("lora", "full_ft", "scratch")
DIST_FACTOR = 8  # distributed = DIST_FACTOR x w_min(N) worlds


@dataclass
class C14Config:
    static_source_dir: str = "runs/c7_local"
    dynamic_source_dir: str = "runs/c8_local_heavy"
    out_dir: str = "runs/c14_local"
    domains: str = "static,dynamic"
    n_grid: str = "256,1024,4096,16384,65536"
    diversities: str = "conc,dist"
    methods: str = "lora,full_ft,scratch"
    n_seeds: int = 3
    total_steps: int = 2560  # ceil(65536/256) * 10 epochs (static recipe anchor)
    rank: int = 8
    alpha: float = 1.0
    n_test_static: int = 30
    n_test_dynamic: int = 20
    grid_size: int = 64
    field_batch: int = 8
    field_lr: float = 2.0e-4
    field_weight_decay: float = 1.0e-4
    w_values: str = "1.0"
    seed: int = 1234
    mode: str = "full"
    cpu: bool = False


def _device(cfg: C14Config):
    return torch.device("cpu" if cfg.cpu or not torch.cuda.is_available() else "cuda")


def _out(cfg: C14Config) -> Path:
    return Path(cfg.out_dir)


def _manifest_path(cfg: C14Config) -> Path:
    return _out(cfg) / "c14_manifest.json"


def _load_manifest(cfg: C14Config) -> dict:
    p = _manifest_path(cfg)
    if p.exists():
        return json.loads(p.read_text())
    return {"cells": {}, "arms": [], "sources": {}, "config": {}}


def _save_manifest(cfg: C14Config, man: dict) -> None:
    man["config"] = {k: v for k, v in asdict(cfg).items()}
    C.ensure_dir(_out(cfg))
    C.write_json(_manifest_path(cfg), man)


def cell_key(domain: str, N: int, div: str) -> str:
    return f"{domain}__N{N}__{div}"


def arm_ckpt(cfg: C14Config, domain: str, N: int, div: str, method: str, s: int) -> Path:
    return _out(cfg) / "checkpoints" / f"c14__{domain}__N{N}__{div}__{method}__s{s}.pt"


# ---------------------------------------------------------------------------
# Static domain: pooled world stream -> per-cell (x, y) state samples
# ---------------------------------------------------------------------------
# Pool stream: deterministic world generator mirroring C.collect_task_dataset's
# acceptance rules (build_world -> build_prm(seed+17) -> connectivity checks),
# but keeping ALL connected labeled nodes per world (start + every connected
# node except the goal, filtered to finite residuals) instead of the 160-node
# subsample. Worlds colliding with the frozen 30-map TEST cohort (by
# C9.world_fingerprint) are skipped. w_min(N) = length of the shortest stream
# prefix whose cumulative state count reaches N; conc(N) uses that prefix,
# dist(N) uses the 8x prefix from the SAME stream.

def _static_pool_seed(cfg: C14Config) -> int:
    h = int(hashlib.md5(STATIC_TARGET.encode()).hexdigest()[:6], 16) % 1_000_000
    return 14_000_000 + h + int(cfg.seed)


def _static_test_fingerprints(cfg: C14Config, spec, rmcfg) -> set:
    c9cfg = C9.C9Config(seed=int(cfg.seed))
    fps = set()
    for _wi, world, _rm in C9.iter_test_worlds(spec, 0, c9cfg, rmcfg, cfg.n_test_static):
        fps.add(C9.world_fingerprint(world))
    return fps


def collect_static_pool(cfg: C14Config, n_states_needed: int,
                        n_worlds_needed: int = 0) -> dict:
    """Stream static worlds until cumulative labeled states >= n_states_needed
    AND at least n_worlds_needed worlds are collected.

    Returns {"x": (M, token_dim), "y": (M,), "world_of_row": (M,), "n_worlds",
    "per_world_counts", "fp_md5s"} for the full prefix collected. Cached to
    out_dir/datasets/static_pool.npz; the stream is deterministic (the cached
    attempt count fast-forwards the RNG), so a larger requirement extends the
    same prefix.
    """
    H7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    spec = specs[STATIC_TARGET]
    rmcfg = C.RoadmapConfig(n_nodes=192, k_neighbors=7)
    base = C9.load_source_base(Path(cfg.static_source_dir), "hrm", torch.device("cpu"))
    feature_cfg = base.feature_cfg

    pool_npz = _out(cfg) / "datasets" / "static_pool.npz"
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    world_of_row: List[np.ndarray] = []
    counts: List[int] = []
    fps: List[str] = []
    attempts_cached = 0
    if pool_npz.exists():
        z = np.load(pool_npz, allow_pickle=False)
        xs = [np.asarray(z["x"])]
        ys = [np.asarray(z["y"])]
        world_of_row = [np.asarray(z["world_of_row"])]
        counts = [int(c) for c in z["per_world_counts"]]
        fps = [str(f) for f in z["fp_md5s"]]
        attempts_cached = int(z["attempts"])
    total = int(sum(counts))
    if total >= n_states_needed and len(counts) >= n_worlds_needed:
        return dict(
            x=np.concatenate(xs, axis=0), y=np.concatenate(ys, axis=0),
            world_of_row=np.concatenate(world_of_row, axis=0),
            n_worlds=len(counts), per_world_counts=counts, fp_md5s=fps,
        )

    test_fps = _static_test_fingerprints(cfg, spec, rmcfg)
    rng = random.Random(_static_pool_seed(cfg))
    for _ in range(attempts_cached):
        rng.randint(0, 2**31 - 1)  # fast-forward past the cached prefix
    attempts = attempts_cached
    skipped = 0
    t0 = time.time()
    while total < n_states_needed or len(counts) < n_worlds_needed:
        w_seed = rng.randint(0, 2**31 - 1)
        attempts += 1
        world = C.build_world(spec, w_seed, rmcfg.min_start_goal_dist_frac)
        if world is None:
            continue
        roadmap = C.build_prm(world, rmcfg, seed=w_seed + 17)
        if roadmap is None or not roadmap.connected_to_goal[0]:
            continue
        connected_idxs = np.where(roadmap.connected_to_goal)[0]
        connected_idxs = connected_idxs[connected_idxs != 1]
        if len(connected_idxs) < 48:  # max(12, 192 // 4), mirrors collect_task_dataset
            continue
        fp = C9.world_fingerprint(world)
        if fp in test_fps:
            skipped += 1
            continue
        features = C.make_features_for_roadmap(world, roadmap, feature_cfg)
        chosen = [0] + [int(i) for i in connected_idxs if int(i) != 0]
        chosen = np.asarray(chosen, dtype=np.int64)
        euclid = np.linalg.norm(roadmap.points[chosen] - world.goal[None, :], axis=1)
        residual = np.maximum(0.0, roadmap.dist_to_goal[chosen] - euclid) / world.side_len
        finite = np.isfinite(residual)
        if finite.sum() < max(8, len(chosen) // 3):
            continue
        x = np.asarray(features[chosen][finite], dtype=np.float32)
        y = np.asarray(residual[finite], dtype=np.float32)
        widx = len(counts)
        xs.append(x)
        ys.append(y)
        world_of_row.append(np.full((x.shape[0],), widx, dtype=np.int64))
        counts.append(int(x.shape[0]))
        fps.append(hashlib.md5(str(fp).encode()).hexdigest())
        total += int(x.shape[0])
        if len(counts) % 200 == 0:
            print(f"[{now_str()}] c14 static pool: {len(counts)} worlds, {total} states "
                  f"({time.time()-t0:.0f}s)", flush=True)
    pool = dict(
        x=np.concatenate(xs, axis=0), y=np.concatenate(ys, axis=0),
        world_of_row=np.concatenate(world_of_row, axis=0),
        n_worlds=len(counts), per_world_counts=counts, fp_md5s=fps,
    )
    C.ensure_dir(pool_npz.parent)
    np.savez(pool_npz, x=pool["x"], y=pool["y"], world_of_row=pool["world_of_row"],
             per_world_counts=np.asarray(counts, dtype=np.int64),
             fp_md5s=np.asarray(fps, dtype="U64"),
             attempts=np.asarray(attempts, dtype=np.int64))
    print(f"[{now_str()}] c14 static pool: saved {len(counts)} worlds / {total} states "
          f"(skipped {skipped} test-colliding)", flush=True)
    return pool


def _w_min(per_world_counts: List[int], N: int) -> int:
    acc = 0
    for i, c in enumerate(per_world_counts):
        acc += int(c)
        if acc >= N:
            return i + 1
    raise RuntimeError(f"pool exhausted: {acc} states < N={N}")


def _sample_rows(pool: dict, N: int, div: str, cell_seed: int) -> Tuple[np.ndarray, dict]:
    """Sample N pool ROW indices for a cell. conc: uniform w/o replacement over
    the minimal prefix's rows. dist: evenly across the 8x prefix's worlds
    (uniform within world; deficits from small worlds spill to later worlds)."""
    counts = pool["per_world_counts"]
    w_min = _w_min(counts, N)
    rng = np.random.default_rng(cell_seed)
    world_of_row = pool["world_of_row"]
    if div == "conc":
        rows = np.where(world_of_row < w_min)[0]
        pick = rng.choice(rows, size=N, replace=False)
        info = dict(w_min=w_min, n_worlds=w_min)
        return np.sort(pick), info
    n_worlds = DIST_FACTOR * w_min
    if n_worlds > len(counts):
        raise RuntimeError(f"pool has {len(counts)} worlds < dist requirement {n_worlds}")
    base_quota = N // n_worlds
    rem = N % n_worlds
    picked: List[np.ndarray] = []
    deficit = 0
    for w in range(n_worlds):
        quota = base_quota + (1 if w < rem else 0) + deficit
        rows_w = np.where(world_of_row == w)[0]
        take = min(quota, len(rows_w))
        deficit = quota - take
        if take > 0:
            picked.append(rng.choice(rows_w, size=take, replace=False))
    if deficit > 0:
        # Final spill: draw the remainder uniformly from unpicked rows of the prefix.
        already = set(np.concatenate(picked).tolist()) if picked else set()
        rows_all = [r for r in np.where(world_of_row < n_worlds)[0] if int(r) not in already]
        if len(rows_all) < deficit:
            raise RuntimeError(f"cannot fill dist cell: short {deficit} rows")
        picked.append(rng.choice(np.asarray(rows_all), size=deficit, replace=False))
    pick = np.sort(np.concatenate(picked))
    if pick.shape[0] != N:
        raise RuntimeError(f"dist sample size {pick.shape[0]} != N={N}")
    info = dict(w_min=w_min, n_worlds=n_worlds)
    return pick, info


def _cell_seed(cfg: C14Config, domain: str, N: int, div: str) -> int:
    h = int(hashlib.md5(f"{domain}|{N}|{div}".encode()).hexdigest()[:7], 16)
    return 14_500_000 + (h % 10_000_000) + int(cfg.seed)


def collect_static_cells(cfg: C14Config, man: dict) -> None:
    Ns = _parse_ints(cfg.n_grid)
    divs = [d for d in _parse_csv(cfg.diversities)]
    # Pass 1: enough states for max(N) reveals w_min(max N); pass 2 (only if
    # dist cells are requested) extends the same stream to the 8x world prefix.
    pool = collect_static_pool(cfg, max(Ns))
    if "dist" in divs:
        need_worlds = DIST_FACTOR * max(_w_min(pool["per_world_counts"], N) for N in Ns)
        if pool["n_worlds"] < need_worlds:
            pool = collect_static_pool(cfg, max(Ns), n_worlds_needed=need_worlds)
    ds_dir = _out(cfg) / "datasets"
    C.ensure_dir(ds_dir)
    for N in Ns:
        for div in divs:
            key = cell_key("static", N, div)
            if key in man["cells"]:
                continue
            rows, info = _sample_rows(pool, N, div, _cell_seed(cfg, "static", N, div))
            npz = ds_dir / f"{key}.npz"
            np.savez(npz, x=pool["x"][rows], y=pool["y"][rows], rows=rows)
            man["cells"][key] = dict(
                domain="static", N=int(N), diversity=div, npz=str(npz),
                w_min=int(info["w_min"]), n_worlds=int(info["n_worlds"]),
                rows_md5=hashlib.md5(rows.tobytes()).hexdigest(),
                feasible=True,
            )
            print(f"[{now_str()}] c14 cell {key}: {N} states from {info['n_worlds']} worlds", flush=True)
    _save_manifest(cfg, man)


# ---------------------------------------------------------------------------
# Dynamic domain: pooled world stream -> per-cell keep-masks over field npz
# ---------------------------------------------------------------------------
# Pool stream: valid C9b world seeds for C_dyn_maze_dense excluding the frozen
# 20-map TEST seed set. Cumulative state count = sum of reachable (node, t)
# entries per world. The pool npz stores the CONCATENATED field-schema arrays
# for the collected world prefix (occ/cells/target/mask + world_of_sample);
# per-cell npz stores a `keep` mask (True on exactly the N sampled states).

def _c9b_cfg(cfg: C14Config) -> "C9B.C9bConfig":
    return C9B.C9bConfig(
        source_dir=str(cfg.dynamic_source_dir), grid_size=int(cfg.grid_size),
        n_test=int(cfg.n_test_dynamic), seed=int(cfg.seed),
    )


def _dynamic_pool_seed(cfg: C14Config) -> int:
    h = int(hashlib.md5(DYNAMIC_TARGET.encode()).hexdigest()[:6], 16) % 1_000_000
    return 14_250_000 + h + int(cfg.seed)


def _dynamic_source_ckpt(cfg: C14Config) -> Path:
    return Path(cfg.dynamic_source_dir) / "checkpoints" / "c8_field__unet_blind.pt"


def collect_dynamic_pool(cfg: C14Config, n_states_needed: int, n_worlds_needed: int) -> dict:
    """Stream valid dynamic world seeds (excluding TEST seeds) until BOTH the
    cumulative reachable-state count reaches n_states_needed AND at least
    n_worlds_needed worlds are collected. Returns the pooled field-schema
    arrays; caches to out_dir/datasets/dynamic_pool.npz (extended on demand)."""
    C9B.install()
    bcfg = _c9b_cfg(cfg)
    src = _torch_load(_dynamic_source_ckpt(cfg))
    W = int(src["window_w"])
    G = int(src["grid_size"])
    if W != 0:
        raise RuntimeError(f"dynamic source is not blind (window_w={W})")

    pool_npz = _out(cfg) / "datasets" / "dynamic_pool.npz"
    seeds: List[int] = []
    if pool_npz.exists():
        z = np.load(pool_npz, allow_pickle=False)
        seeds = [int(s) for s in z["seeds"]]

    test_set = set(C9B.test_world_seeds(DYNAMIC_TARGET, bcfg))
    rng = np.random.default_rng(_dynamic_pool_seed(cfg))

    def reachable_count(seed: int) -> int:
        lab = C9B._collect_world_labels_memo(DYNAMIC_TARGET, int(seed), G)
        return 0 if lab is None else int(np.asarray(lab["reachable"]).sum())

    counts = [reachable_count(s) for s in seeds]
    total = int(sum(counts))
    tries = 0
    while (total < n_states_needed or len(seeds) < n_worlds_needed) and tries < 100_000:
        s = int(rng.integers(0, 2**31 - 1))
        tries += 1
        if s in test_set or s in set(seeds):
            continue
        if not C9B._valid_world_seed(DYNAMIC_TARGET, s, bcfg):
            continue
        c = reachable_count(s)
        if c <= 0:
            continue
        seeds.append(s)
        counts.append(c)
        total += c
        print(f"[{now_str()}] c14 dynamic pool: world {len(seeds)} (seed {s}) "
              f"+{c} states -> {total}", flush=True)
    if total < n_states_needed or len(seeds) < n_worlds_needed:
        raise RuntimeError("dynamic pool collection failed to reach requirement")

    # Build the concatenated field-schema arrays for the collected prefix.
    tmp = _out(cfg) / "datasets" / "dynamic_pool_build.npz"
    C9B.collect_temporal_dataset(DYNAMIC_TARGET, seeds, "field_unet", W,
                                 k_patrollers=0, grid_size=G, out_npz=tmp)
    z = np.load(tmp, allow_pickle=False)
    occ = np.asarray(z["occ"])
    cells = np.asarray(z["cells"])
    target = np.asarray(z["target"])
    mask = np.asarray(z["mask"])
    window_w_z = np.asarray(z["window_w"])
    grid_size_z = np.asarray(z["grid_size"])
    in_channels_z = np.asarray(z["in_channels"])
    z.close()
    # world_of_sample: collect_temporal_dataset appends worlds in `seeds` order,
    # each contributing (t_max + 1) samples.
    world_of_sample = np.zeros((occ.shape[0],), dtype=np.int64)
    pos = 0
    per_world_samples: List[int] = []
    for wi, s in enumerate(seeds):
        lab = C9B._collect_world_labels_memo(DYNAMIC_TARGET, int(s), G)
        t_max = int(lab["params"]["t_max"])
        n = t_max + 1
        world_of_sample[pos:pos + n] = wi
        per_world_samples.append(n)
        pos += n
    if pos != occ.shape[0]:
        raise RuntimeError(f"sample bookkeeping mismatch: {pos} != {occ.shape[0]}")
    np.savez(pool_npz, occ=occ, cells=cells, target=target, mask=mask,
             seeds=np.asarray(seeds, dtype=np.int64),
             world_of_sample=world_of_sample,
             per_world_counts=np.asarray(counts, dtype=np.int64),
             window_w=window_w_z, grid_size=grid_size_z, in_channels=in_channels_z)
    tmp.unlink(missing_ok=True)
    return dict(occ=occ, cells=cells, target=target, mask=mask, seeds=seeds,
                world_of_sample=world_of_sample, per_world_counts=counts,
                n_worlds=len(seeds))


def collect_dynamic_cells(cfg: C14Config, man: dict) -> None:
    Ns = _parse_ints(cfg.n_grid)
    divs = [d for d in _parse_csv(cfg.diversities)]
    # First ensure enough states for max N, then enough worlds for every dist prefix.
    pool = collect_dynamic_pool(cfg, max(Ns), 1)
    for N in Ns:
        w = _w_min(pool["per_world_counts"], N)
        need = DIST_FACTOR * w
        if pool["n_worlds"] < need:
            pool = collect_dynamic_pool(cfg, int(sum(pool["per_world_counts"])), need)
    ds_dir = _out(cfg) / "datasets"
    C.ensure_dir(ds_dir)
    mask = pool["mask"]
    world_of_sample = pool["world_of_sample"]
    for N in Ns:
        for div in divs:
            key = cell_key("dynamic", N, div)
            if key in man["cells"]:
                continue
            counts = pool["per_world_counts"]
            w_min = _w_min(counts, N)
            n_worlds = w_min if div == "conc" else DIST_FACTOR * w_min
            rng = np.random.default_rng(_cell_seed(cfg, "dynamic", N, div))
            keep = np.zeros_like(mask, dtype=np.bool_)
            if div == "conc":
                cand = np.argwhere(mask & (world_of_sample[:, None] < w_min))
                pick = cand[rng.choice(len(cand), size=N, replace=False)]
                keep[pick[:, 0], pick[:, 1]] = True
            else:
                base_quota = N // n_worlds
                rem = N % n_worlds
                deficit = 0
                picked_total = 0
                for w in range(n_worlds):
                    quota = base_quota + (1 if w < rem else 0) + deficit
                    cand = np.argwhere(mask & (world_of_sample[:, None] == w))
                    take = min(quota, len(cand))
                    deficit = quota - take
                    if take > 0:
                        pick = cand[rng.choice(len(cand), size=take, replace=False)]
                        keep[pick[:, 0], pick[:, 1]] = True
                        picked_total += take
                if deficit > 0:
                    cand = np.argwhere(mask & (world_of_sample[:, None] < n_worlds) & ~keep)
                    if len(cand) < deficit:
                        raise RuntimeError(f"cannot fill {key}: short {deficit}")
                    pick = cand[rng.choice(len(cand), size=deficit, replace=False)]
                    keep[pick[:, 0], pick[:, 1]] = True
                    picked_total += deficit
                if picked_total != N:
                    raise RuntimeError(f"{key}: picked {picked_total} != N={N}")
            if int(keep.sum()) != N:
                raise RuntimeError(f"{key}: keep-mask has {int(keep.sum())} != N={N}")
            npz = ds_dir / f"{key}.npz"
            np.savez(npz, keep=keep)
            man["cells"][key] = dict(
                domain="dynamic", N=int(N), diversity=div, npz=str(npz),
                w_min=int(w_min), n_worlds=int(n_worlds),
                rows_md5=hashlib.md5(keep.tobytes()).hexdigest(),
                feasible=True,
            )
            print(f"[{now_str()}] c14 cell {key}: {N} states from {n_worlds} worlds", flush=True)
    _save_manifest(cfg, man)


# ---------------------------------------------------------------------------
# Step-matched trainers
# ---------------------------------------------------------------------------

class _Cycler:
    """Deterministic index cycler: reshuffles a permutation each pass."""

    def __init__(self, n: int, batch: int, seed: int):
        self.n = int(n)
        self.batch = int(batch)
        self.rng = np.random.default_rng(seed)
        self.perm = self.rng.permutation(self.n)
        self.pos = 0

    def next(self) -> np.ndarray:
        out: List[np.ndarray] = []
        need = self.batch
        while need > 0:
            take = min(need, self.n - self.pos)
            out.append(self.perm[self.pos:self.pos + take])
            self.pos += take
            need -= take
            if self.pos >= self.n:
                self.perm = self.rng.permutation(self.n)
                self.pos = 0
        return np.concatenate(out)


def train_static_arm(cfg: C14Config, npz_path: Path, method: str, s: int,
                     out_ckpt: Path, device) -> Path:
    """Step-matched static scalar trainer. Mirrors C9's/C9h's recipes:
    full_ft/scratch = plain smooth-L1 on model(x) (train_scalar_model);
    lora = unbounded rank-8 LoRA (train_scalar_lora bounded=False). All
    methods run EXACTLY cfg.total_steps optimizer steps at batch 256."""
    if out_ckpt.exists():
        return out_ckpt
    base = C9.load_source_base(Path(cfg.static_source_dir), "hrm", device)
    tcfg = base.train_cfg
    z = np.load(npz_path, allow_pickle=False)
    x = np.asarray(z["x"], dtype=np.float32)
    y = np.asarray(z["y"], dtype=np.float32)

    arm_seed = _cell_seed(cfg, "static", x.shape[0], method) + 101 * s
    C.set_global_seed(arm_seed)
    model = C.build_model(base.backbone_cfg, base.feature_cfg, tcfg, device)
    if method in ("full_ft", "lora"):
        C.safe_load_state(model, base.ckpt_path)
    max_resid = float(tcfg.max_norm_residual)
    if method == "lora":
        max_resid = float("inf")
        model.max_norm_residual = max_resid
        C.apply_lora(model, rank=int(cfg.rank), alpha=float(cfg.alpha))
        C.set_lora_trainable(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    cyc = _Cycler(x.shape[0], int(tcfg.batch_size), arm_seed + 1)
    model.train()
    losses: List[float] = []
    for step in range(int(cfg.total_steps)):
        idx = torch.from_numpy(cyc.next()).long()
        xb = xt[idx].to(device)
        yb = yt[idx].to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = F.smooth_l1_loss(pred, yb)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite c14 static loss ({method} s{s})")
        loss.backward()
        if tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, tcfg.grad_clip)
        opt.step()
        losses.append(float(loss.item()))
    payload = {
        "model": model.state_dict(),
        "backbone_cfg": asdict(base.backbone_cfg),
        "feature_cfg": asdict(base.feature_cfg),
        "train_cfg": asdict(tcfg),
        "method": method,
        "max_norm_residual": max_resid,
        "total_steps": int(cfg.total_steps),
        "final_loss": float(np.mean(losses[-50:])),
    }
    if method == "lora":
        payload["lora_rank"] = int(cfg.rank)
        payload["alpha"] = float(cfg.alpha)
    C.ensure_dir(out_ckpt.parent)
    torch.save(payload, out_ckpt)
    return out_ckpt


def train_dynamic_arm(cfg: C14Config, keep_npz: Path, method: str, s: int,
                      out_ckpt: Path, device) -> Path:
    """Step-matched dynamic field trainer. Mirrors C9b.train_field_temporal's
    forward/loss (grid -> gather-at-cells -> masked smooth-L1) with the loss
    mask restricted to the cell's sampled `keep` states; EXACTLY
    cfg.total_steps optimizer steps at batch cfg.field_batch, sampling only
    (world, t) samples that contain at least one kept state."""
    if out_ckpt.exists():
        return out_ckpt
    pool_npz = _out(cfg) / "datasets" / "dynamic_pool.npz"
    z = np.load(pool_npz, allow_pickle=False)
    occ = z["occ"]
    cells = z["cells"]
    target = z["target"]
    keep = np.load(keep_npz, allow_pickle=False)["keep"]
    if keep.shape != z["mask"].shape:
        raise RuntimeError("keep-mask shape mismatch with pool")
    eff_mask = z["mask"] & keep

    src_ckpt = _dynamic_source_ckpt(cfg)
    src = _torch_load(src_ckpt)
    in_channels = int(src["in_channels"])
    grid_size = int(src["grid_size"])
    backbone_name = src["backbone"]
    window_w = int(src["window_w"])

    arm_seed = _cell_seed(cfg, "dynamic", int(keep.sum()), method) + 101 * s
    C.set_global_seed(arm_seed)
    model = C6.build_model(backbone_name, in_channels=in_channels).to(device)
    if method != "scratch":
        C.safe_load_state(model, src_ckpt)
    if method == "lora":
        C9H.apply_conv_lora(model, rank=int(cfg.rank), alpha=float(cfg.alpha))
        C.set_lora_trainable(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.field_lr),
                            weight_decay=float(cfg.field_weight_decay))

    live = np.where(eff_mask.any(axis=1))[0]
    if len(live) == 0:
        raise RuntimeError("no (world, t) sample contains a kept state")
    cyc = _Cycler(len(live), int(cfg.field_batch), arm_seed + 1)
    model.train()
    losses: List[float] = []
    for step in range(int(cfg.total_steps)):
        bidx = live[cyc.next()]
        occ_b = torch.from_numpy(np.ascontiguousarray(occ[bidx])).to(device=device, dtype=torch.float32)
        cells_b = torch.from_numpy(np.ascontiguousarray(cells[bidx])).to(device)
        target_b = torch.from_numpy(np.ascontiguousarray(target[bidx])).to(device=device, dtype=torch.float32)
        mask_b = torch.from_numpy(np.ascontiguousarray(eff_mask[bidx])).to(device)
        pred_grid = C6.model_output_residual(model(occ_b))
        if not torch.isfinite(pred_grid).all():
            raise FloatingPointError(f"c14 dynamic {method}: non-finite grid")
        B, G, _ = pred_grid.shape
        ix = cells_b[..., 0].clamp(0, G - 1)
        iy = cells_b[..., 1].clamp(0, G - 1)
        pred_nodes = torch.gather(pred_grid.reshape(B, G * G), 1, ix * G + iy)
        m = mask_b.bool()
        loss = F.smooth_l1_loss(pred_nodes[m], target_b[m])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"c14 dynamic {method}: non-finite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.item()))
    payload = {
        "model": model.state_dict(),
        "in_channels": int(in_channels),
        "window_w": int(window_w),
        "grid_size": int(grid_size),
        "backbone": backbone_name,
        "method": method,
        "total_steps": int(cfg.total_steps),
        "final_loss": float(np.mean(losses[-50:])),
    }
    if method == "lora":
        payload["lora_rank"] = int(cfg.rank)
        payload["alpha"] = float(cfg.alpha)
    C.ensure_dir(out_ckpt.parent)
    torch.save(payload, out_ckpt)
    return out_ckpt


def run_adapt(cfg: C14Config, device) -> dict:
    man = _load_manifest(cfg)
    Ns = _parse_ints(cfg.n_grid)
    divs = _parse_csv(cfg.diversities)
    methods = _parse_csv(cfg.methods)
    domains = _parse_csv(cfg.domains)
    have = {(a["domain"], a["N"], a["diversity"], a["method"], a["seed"]) for a in man["arms"]}
    for domain in domains:
        for N in Ns:
            for div in divs:
                key = cell_key(domain, N, div)
                cell = man["cells"].get(key)
                if cell is None:
                    raise RuntimeError(f"cell {key} not collected; run --mode collect first")
                if not cell.get("feasible", True):
                    continue
                for method in methods:
                    for s in range(int(cfg.n_seeds)):
                        if (domain, N, div, method, s) in have:
                            continue
                        ck = arm_ckpt(cfg, domain, N, div, method, s)
                        t0 = time.time()
                        if domain == "static":
                            train_static_arm(cfg, Path(cell["npz"]), method, s, ck, device)
                        else:
                            train_dynamic_arm(cfg, Path(cell["npz"]), method, s, ck, device)
                        man["arms"].append(dict(domain=domain, N=int(N), diversity=div,
                                                method=method, seed=int(s), ckpt=str(ck)))
                        print(f"[{now_str()}] c14 adapt {key} {method} s{s}: "
                              f"{time.time()-t0:.0f}s", flush=True)
                        _save_manifest(cfg, man)
    return man


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

RAW_COLS = ["domain", "target", "N", "diversity", "method", "seed", "world_index",
            "provider", "mode", "w", "budget", "found", "expansions", "closed",
            "cost", "optimal", "suboptimality", "arrival", "optimal_arrival", "nonfinite"]


def _static_budgets(cfg: C14Config) -> List[int]:
    calib = Path(cfg.static_source_dir) / "calibration.json"
    b = (json.loads(calib.read_text()).get("budgets", {}) or {}).get(STATIC_TARGET)
    if not b:
        raise RuntimeError(f"no static budgets in {calib}")
    return [int(x) for x in b]


def eval_static(cfg: C14Config, man: dict, device,
                only_worlds: Optional[List[int]] = None) -> List[dict]:
    """only_worlds: optional test-world indices (positions in the frozen C7
    matched-world stream) to evaluate — used by the Modal harness to shard by
    world. The stream is still walked from the start so world identities are
    byte-identical to the unsharded run; non-selected worlds are skipped
    before any provider work."""
    H7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    spec = specs[STATIC_TARGET]
    rmcfg = C.RoadmapConfig(n_nodes=192, k_neighbors=7)
    budgets = _static_budgets(cfg)
    w_values = [float(x) for x in _parse_csv(cfg.w_values)]
    c9cfg = C9.C9Config(seed=int(cfg.seed))

    providers: Dict[str, object] = {"euclid": P.EuclidProvider(), "oracle": P.OracleProvider()}
    meta: Dict[str, dict] = {
        "euclid": dict(method="euclid", N=-1, diversity="", seed=-1),
        "oracle": dict(method="oracle", N=-1, diversity="", seed=-1),
    }
    base = C9.load_source_base(Path(cfg.static_source_dir), "hrm", device)
    zp = P.ScalarResidualProvider(base.model, base.feature_cfg, device, "hrm",
                                  base.train_cfg.max_norm_residual)
    zp.name = "zeroshot_hrm"
    providers[zp.name] = zp
    meta[zp.name] = dict(method="zero_shot", N=0, diversity="", seed=-1)
    for a in man["arms"]:
        if a["domain"] != "static":
            continue
        prov = C9H.load_scalar_provider_c9h(Path(a["ckpt"]), device)
        key = f'{a["method"]}_N{a["N"]}_{a["diversity"]}_s{a["seed"]}'
        prov.name = key
        providers[key] = prov
        meta[key] = dict(method=a["method"], N=a["N"], diversity=a["diversity"], seed=a["seed"])

    rows: List[dict] = []
    for world_index, world, rm in C9.iter_test_worlds(spec, 0, c9cfg, rmcfg, cfg.n_test_static):
        if only_worlds is not None and world_index not in only_worlds:
            continue
        recs = P.run_world_arms(world, rm, providers, budgets, w_values, goal_idx=1)
        for r in recs:
            m = meta.get(r["provider"], dict(method=r["provider"], N=-1, diversity="", seed=-1))
            r.update(dict(domain="static", target=STATIC_TARGET, world_index=world_index, **m))
            rows.append(r)
    print(f"[{now_str()}] c14 eval static: {len(rows)} rows "
          f"({len(providers)} providers x {cfg.n_test_static} worlds)", flush=True)
    return rows


def eval_dynamic(cfg: C14Config, man: dict, device,
                 only_worlds: Optional[List[int]] = None) -> List[dict]:
    """only_worlds: optional test-world POSITIONS (indices into the frozen test
    seed list) to evaluate — used by the Modal harness to shard by world."""
    C9B.install()
    bcfg = _c9b_cfg(cfg)
    budgets = C9B.budgets_for(DYNAMIC_TARGET, bcfg)
    w_values = [float(x) for x in _parse_csv(cfg.w_values)]

    seeds = C9B.test_world_seeds(DYNAMIC_TARGET, bcfg)
    worlds = []
    for wi, s in enumerate(seeds):
        if only_worlds is not None and wi not in only_worlds:
            continue
        lab = C9B._collect_world_labels_memo(DYNAMIC_TARGET, s, cfg.grid_size)
        if lab is not None:
            worlds.append((wi, lab))

    providers: Dict[str, object] = {"euclid": DP.EuclidTimeProvider(), "oracle": DP.OracleProvider()}
    meta: Dict[str, dict] = {
        "euclid": dict(method="euclid", N=-1, diversity="", seed=-1),
        "oracle": dict(method="oracle", N=-1, diversity="", seed=-1),
    }
    src = _dynamic_source_ckpt(cfg)
    zp = C9B.load_temporal_provider(src, "field_unet", device)
    zp.name = "zeroshot_field_unet_blind"
    providers[zp.name] = zp
    meta[zp.name] = dict(method="zero_shot", N=0, diversity="", seed=-1)
    for a in man["arms"]:
        if a["domain"] != "dynamic":
            continue
        bounded = a["method"] != "lora"  # design: LoRA is the unbounded arm
        prov = C9B.load_temporal_provider(Path(a["ckpt"]), "field_unet", device, bounded=bounded)
        key = f'{a["method"]}_N{a["N"]}_{a["diversity"]}_s{a["seed"]}'
        prov.name = key
        providers[key] = prov
        meta[key] = dict(method=a["method"], N=a["N"], diversity=a["diversity"], seed=a["seed"])

    rows: List[dict] = []
    for wi, lab in worlds:
        pp = lab["params"]
        recs = DP.run_world_arms_spacetime(
            lab["world"], lab["rm"], lab["dyn"], providers, budgets, w_values,
            pp["v_agent"], pp["dt"], int(pp["t_max"]),
        )
        for r in recs:
            m = meta.get(r["provider"], dict(method=r["provider"], N=-1, diversity="", seed=-1))
            r.update(dict(domain="dynamic", target=DYNAMIC_TARGET, world_index=wi, **m))
            rows.append(r)
    print(f"[{now_str()}] c14 eval dynamic: {len(rows)} rows "
          f"({len(providers)} providers x {len(worlds)} worlds)", flush=True)
    return rows


def run_eval(cfg: C14Config, device) -> Path:
    man = _load_manifest(cfg)
    domains = _parse_csv(cfg.domains)
    res_dir = _out(cfg) / "results"
    C.ensure_dir(res_dir)
    if "static" in domains:
        rows = eval_static(cfg, man, device)
        with open(res_dir / "c14_static_raw.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=RAW_COLS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in RAW_COLS})
    if "dynamic" in domains:
        rows = eval_dynamic(cfg, man, device)
        with open(res_dir / "c14_dynamic_raw.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=RAW_COLS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in RAW_COLS})
    # Rebuild the merged CSV from every per-domain file present on disk, so a
    # later single-domain eval never clobbers the other domain's rows.
    all_rows: List[dict] = []
    for name in ("c14_static_raw.csv", "c14_dynamic_raw.csv"):
        p = res_dir / name
        if p.exists():
            with open(p, newline="") as f:
                all_rows.extend(_csv.DictReader(f))
    raw = res_dir / "continuous_prm_c14_eval_raw.csv"
    with open(raw, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=RAW_COLS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in RAW_COLS})
    print(f"[{now_str()}] c14 eval: merged {len(all_rows)} rows -> {raw}", flush=True)
    return raw


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_collect(cfg: C14Config) -> dict:
    man = _load_manifest(cfg)
    domains = _parse_csv(cfg.domains)
    if "static" in domains:
        collect_static_cells(cfg, man)
    if "dynamic" in domains:
        collect_dynamic_cells(cfg, man)
    return man


def main(argv=None):
    p = argparse.ArgumentParser(description="C14 label-density factorial")
    p.add_argument("--mode", type=str, default="full",
                   choices=["collect", "adapt", "eval", "full"])
    p.add_argument("--domains", type=str, default="static,dynamic")
    p.add_argument("--out-dir", type=str, default="runs/c14_local")
    p.add_argument("--static-source-dir", type=str, default="runs/c7_local")
    p.add_argument("--dynamic-source-dir", type=str, default="runs/c8_local_heavy")
    p.add_argument("--n-grid", type=str, default="256,1024,4096,16384,65536")
    p.add_argument("--diversities", type=str, default="conc,dist")
    p.add_argument("--methods", type=str, default="lora,full_ft,scratch")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--total-steps", type=int, default=2560)
    p.add_argument("--n-test-static", type=int, default=30)
    p.add_argument("--n-test-dynamic", type=int, default=20)
    p.add_argument("--w-values", type=str, default="1.0")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)
    cfg = C14Config(
        mode=args.mode, domains=args.domains, out_dir=args.out_dir,
        static_source_dir=args.static_source_dir,
        dynamic_source_dir=args.dynamic_source_dir,
        n_grid=args.n_grid, diversities=args.diversities, methods=args.methods,
        n_seeds=args.n_seeds, total_steps=args.total_steps,
        n_test_static=args.n_test_static, n_test_dynamic=args.n_test_dynamic,
        w_values=args.w_values, seed=args.seed, cpu=args.cpu,
    )
    device = _device(cfg)
    print(f"[{now_str()}] C14 mode={cfg.mode} domains={cfg.domains} out={cfg.out_dir} "
          f"n_grid={cfg.n_grid} seeds={cfg.n_seeds} steps={cfg.total_steps} device={device}",
          flush=True)
    if cfg.mode in ("collect", "full"):
        run_collect(cfg)
    if cfg.mode in ("adapt", "full"):
        run_adapt(cfg, device)
    if cfg.mode in ("eval", "full"):
        run_eval(cfg, device)


if __name__ == "__main__":
    main()
