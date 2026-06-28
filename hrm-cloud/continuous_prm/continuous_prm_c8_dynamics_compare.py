#!/usr/bin/env python3
"""
C8 Dynamics Comparison — orchestrator skeleton.

Wires together the C8 space-time heuristic-provider pipeline (ScalarTemporal +
ValueFieldTemporal) against the three C8-dynamic train suites and the three new
C8-dynamic held-out suites, running matched-integrity space-time A* on PRM
graphs under multiple expansion budgets and focal weights.

Modes
-----
collect   — generate roadmap worlds + run reference space-time A* (Task 10)
train     — fit scalar-temporal and value-field-temporal models (Task 10)
eval      — sharded evaluation of all arms (Task 11)
calibrate — per-suite budget calibration (Task 12)
analyze   — aggregate stats + pre-registered comparisons (Task 13)
full      — collect → train → calibrate → eval → analyze (Tasks 10-13)
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import continuous_prm_common as C
import continuous_prm_dynamic_providers as P
import continuous_prm_spacetime as ST
import continuous_prm_dynamics as D
import continuous_prm_c8_dynamic_maps as M8

# Lazy imports for heavy modules (torch, C6 helpers) go inside functions to
# avoid pulling in GPU setup at argparse time.


# ---------------------------------------------------------------------------
# Helpers (mirrors C7 style; re-export from C6 so future tasks can import
# from this module directly without depending on C6's internal structure)
# ---------------------------------------------------------------------------

from continuous_prm_c6_heatmap_value_field import (  # noqa: F401
    ensure_dir,
    parse_csv,
    parse_int_csv,
    write_csv,
    write_json,
    read_csv,
    now_str,
    mcnemar_exact_p,
    bh_q_values,
    sanitize_name,
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class C8Config:
    # Grid / roadmap geometry
    grid_size: int = 64
    roadmap_nodes: int = 192
    roadmap_k: int = 7

    # Suite selection
    train_tasks: str = "C_dyn_maze,C_dyn_rooms,C_dyn_spiral"
    eval_suites: str = (
        "C_dyn_maze,C_dyn_rooms,C_dyn_spiral,"
        "C_dyn_maze_dense,C_dyn_crossing,C_dyn_rooms_large"
    )

    # Model families to benchmark
    scalar_backbones: str = "hrm,onlstm"
    field_backbones: str = "unet,onlstm,hrm"

    # Expansion budgets (fallback until calibration overrides per-suite).
    # Larger than the C7 default because space-time graphs are denser.
    budgets: str = "2000"

    # Focal weight grid (filled by scale preset when empty)
    w_values: str = ""

    # World / training counts (0 = filled by scale preset)
    eval_worlds: int = 0
    train_worlds: int = 0
    epochs: int = 0

    # Misc
    seed: int = 1234
    scale: str = "local"
    mode: str = "full"
    out_dir: str = "runs/c8_local"
    cpu: bool = False
    budget_grid_size: int = 0
    make_figures: bool = True

    # Dynamics-specific knobs
    # window_w: rollout window length for temporal heuristics (number of time
    # steps a scalar/field model looks back when building the feature vector).
    window_w: int = 8
    # k_patrollers: number of nearest patrollers used in scalar feature vectors.
    k_patrollers: int = 4

    # NOTE: per-suite v_agent / dt / t_max are NOT global config — they come
    # from M8.dynamics_params(suite) and are read per suite in eval/calibrate/
    # train modes. They vary by suite geometry and patroller density, so a
    # single global value would be incorrect.


# ---------------------------------------------------------------------------
# Scale presets
# (Local is smaller than C7 because each space-time eval is heavier —
# dynamics worlds include temporal graphs that are ~t_max times larger.)
# ---------------------------------------------------------------------------

def apply_scale_preset(cfg: C8Config) -> C8Config:
    if cfg.scale == "local":
        cfg.eval_worlds = cfg.eval_worlds or 16
        cfg.train_worlds = cfg.train_worlds or 64
        cfg.epochs = cfg.epochs or 12
        cfg.w_values = cfg.w_values or "1.0,1.1"
        cfg.budget_grid_size = cfg.budget_grid_size or 2
    else:  # cluster
        cfg.eval_worlds = cfg.eval_worlds or 80
        cfg.train_worlds = cfg.train_worlds or 120
        cfg.epochs = cfg.epochs or 20
        cfg.w_values = cfg.w_values or "1.0,1.05,1.1,1.25"
        cfg.budget_grid_size = cfg.budget_grid_size or 3
    return cfg


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _pick_device(cfg: C8Config):
    import torch
    return torch.device("cpu" if cfg.cpu or not torch.cuda.is_available() else "cuda")


# ---------------------------------------------------------------------------
# Task 10 — shared space-time label collection
# ---------------------------------------------------------------------------
#
# A "labelset" is everything one dynamic world contributes to supervised
# training: the world + roadmap + dynamics, the per-suite params, and the TRUE
# space-time time-to-go converted into the SAME normalized PRM-node residual
# target the providers invert at eval time
#   (h = euclid_steps + T_scale * clip(residual)).
# Both the scalar models (Task 10a, here) and the field models (Task 10b)
# supervise on `node_residual` masked by `reachable`. Keep _collect_world_labels
# general so Task 10b can reuse it verbatim for the field dataset.

# Distinct, well-separated seed per (suite, world) so two suites' world #k never
# share a seed (avoids accidental world reuse across suites).
_SUITE_SEED_STRIDE = 100_003
_WORLD_SEED_STRIDE = 9_973


def _world_seed(cfg: C8Config, suite_idx: int, world_idx: int) -> int:
    return int(cfg.seed) + _SUITE_SEED_STRIDE * (suite_idx + 1) + _WORLD_SEED_STRIDE * world_idx


def _collect_world_labels(suite: str, seed: int, cfg: C8Config) -> Optional[dict]:
    """Build one dynamic world's supervised space-time labelset, or None.

    Returns None (skips the world) when:
      - the static/dynamic world could not be built,
      - the PRM could not be built or start (node 0) is not connected to goal,
      - the world is space-time-unsolvable from the start (hstar[0,0] not finite).

    Otherwise returns a dict with keys:
      world, rm, dyn, params, ttg, node_residual, reachable
    where `node_residual` (N, t_max+1) is the clipped, T_scale-normalized
    regression target and `reachable` (N, t_max+1) is the supervision mask.
    Reused by Task 10b (field training) — keep general.
    """
    built = M8.build_dynamic_world(suite, seed)
    if built is None:
        return None
    world, dyn = built

    rm = C.build_prm(world, C.RoadmapConfig(cfg.roadmap_nodes, cfg.roadmap_k), seed)
    if rm is None or not bool(rm.connected_to_goal[0]):
        return None

    params = M8.dynamics_params(suite)
    v_agent = float(params["v_agent"])
    dt = float(params["dt"])
    t_max = int(params["t_max"])

    # TRUE space-time time-to-go at PRM nodes (goal = node 1).
    hstar = ST.backward_spacetime_dijkstra(rm.adj, rm.points, dyn, v_agent, dt, t_max, goal=1)
    if not np.isfinite(hstar[0, 0]):
        # space-time-unsolvable from the start at t=0
        return None

    ttg = ST.oracle_time_to_go(hstar, t_max)                       # (N, t_max+1)
    euclid_steps = P.euclid_time_row(rm, v_agent, goal_idx=1) / dt  # (N,)
    T_scale = float(world.side_len) / v_agent / dt                 # map-crossing time in steps
    max_norm_residual = float(C.TrainingConfig().max_norm_residual)
    # Normalized residual target: same inversion the providers apply
    #   h = euclid_steps + T_scale * clip(residual, 0, max_norm_residual).
    # Clip to [0, max_norm_residual] AFTER dividing by T_scale (normalized units).
    node_residual = np.clip(ttg - euclid_steps[:, None], 0.0, None) / T_scale
    node_residual = np.clip(node_residual, 0.0, max_norm_residual)
    reachable = np.isfinite(hstar) & (hstar < 1e29)               # (N, t_max+1) bool

    return {
        "world": world,
        "rm": rm,
        "dyn": dyn,
        "params": params,
        "ttg": ttg,
        "node_residual": node_residual,
        "reachable": reachable,
    }


def _collect_labelsets(cfg: C8Config) -> Tuple[List[dict], Dict[str, int]]:
    """Collect labelsets for cfg.train_tasks x cfg.train_worlds (seeded distinctly).

    Returns (labelsets, per_suite_counts). Skipped worlds (invalid / unsolvable)
    are simply omitted; the per-suite counts report how many of the requested
    worlds yielded a usable labelset.
    """
    tasks = parse_csv(cfg.train_tasks)
    labelsets: List[dict] = []
    counts: Dict[str, int] = {}
    for s_idx, suite in enumerate(tasks):
        kept = 0
        for w_idx in range(int(cfg.train_worlds)):
            seed = _world_seed(cfg, s_idx, w_idx)
            t0 = time.time()
            ls = _collect_world_labels(suite, seed, cfg)
            if ls is not None:
                ls["suite"] = suite
                ls["seed"] = seed
                labelsets.append(ls)
                kept += 1
            print(
                f"[{now_str()}] C8 collect: suite={suite} world={w_idx} "
                f"seed={seed} {'kept' if ls is not None else 'skip'} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
        counts[suite] = kept
        print(f"[{now_str()}] C8 collect: suite={suite} kept {kept}/{cfg.train_worlds}", flush=True)
    return labelsets, counts


# ---------------------------------------------------------------------------
# Task 10a — scalar temporal dataset + training
# ---------------------------------------------------------------------------

def _scalar_token_dim(cfg: C8Config) -> int:
    return 4 + 4 * int(cfg.k_patrollers)


def _scalar_backbone_cfg(name: str) -> "C.BackboneConfig":
    """Build a scalar BackboneConfig by name (mirrors ScalarTemporalProvider
    .untrained_for_test's minimal Namespace, at production-scale hidden dims)."""
    ns = argparse.Namespace(
        hrm_hidden=192,
        hrm_layers=2,
        hrm_k_step=2,
        hrm_heads=4,
        head_hidden=256,
        onlstm_hidden=192,
        onlstm_layers=2,
        onlstm_chunk_size=8,
    )
    configs = C.build_backbone_configs(ns)
    if name not in configs:
        raise ValueError(f"unknown scalar backbone {name!r}; have {sorted(configs)}")
    return configs[name]


def _build_scalar_dataset(labelsets: List[dict], cfg: C8Config, window_w: Optional[int] = None):
    """Build the pooled scalar-temporal dataset across collected worlds.

    Per world: features Xw (N, t_max+1, W+1, token_dim) from
    build_scalar_temporal_features, flattened to (N*(t_max+1), W+1, token_dim);
    target yw = node_residual.reshape(-1); mask mw = reachable.reshape(-1).
    Concatenated across worlds. token_dim = 4 + 4*k_patrollers.

    `window_w` overrides cfg.window_w (default None -> cfg.window_w). The W=0
    time-blind variant produces seq_len=1 (W+1=1) features with the SAME token_dim
    — only the sequence length differs, so the model is a normal
    ContinuousHeuristicModel trained on 1-frame sequences.

    Returns (X float32 (M, W+1, token_dim), y float32 (M,), mask bool (M,)).
    """
    W = int(cfg.window_w if window_w is None else window_w)
    token_dim = _scalar_token_dim(cfg)
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ms: List[np.ndarray] = []
    for ls in labelsets:
        world, rm, dyn = ls["world"], ls["rm"], ls["dyn"]
        params = ls["params"]
        dt = float(params["dt"])
        t_max = int(params["t_max"])
        Xw = P.build_scalar_temporal_features(
            world, rm, dyn, t_max, dt, W, int(cfg.k_patrollers), goal_idx=1
        )  # (N, t_max+1, W+1, token_dim)
        N = Xw.shape[0]
        Xs.append(Xw.reshape(N * (t_max + 1), W + 1, token_dim).astype(np.float32))
        ys.append(ls["node_residual"].reshape(-1).astype(np.float32))
        ms.append(ls["reachable"].reshape(-1).astype(np.bool_))
    if not Xs:
        return (
            np.zeros((0, W + 1, token_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.bool_),
        )
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    mask = np.concatenate(ms, axis=0)
    return X, y, mask


def _train_scalar(
    labelsets: List[dict],
    cfg: C8Config,
    device,
    window_w: Optional[int] = None,
    suffix: str = "",
) -> Dict[str, Path]:
    """Train each scalar backbone on the masked space-time residual target.

    `window_w` overrides cfg.window_w (default None -> cfg.window_w); the dataset
    is built internally with that W from the SHARED labelsets (no re-collection).
    `suffix` is appended to the checkpoint name: c8_scalar__{backbone}{suffix}.pt
    (e.g. suffix="_blind" for the W=0 time-blind ablation variant). Returns
    {backbone: ckpt_path}. Empty cfg.scalar_backbones -> trains nothing.
    """
    import torch
    import torch.nn.functional as F

    names = parse_csv(cfg.scalar_backbones)
    if not names:
        return {}

    out_dir = Path(cfg.out_dir)
    W = int(cfg.window_w if window_w is None else window_w)
    token_dim = _scalar_token_dim(cfg)
    train_cfg = C.TrainingConfig()
    max_norm_residual = float(train_cfg.max_norm_residual)
    lr = float(train_cfg.lr)
    epochs = int(cfg.epochs)
    batch_size = 256
    grad_clip = 1.0

    # Build the scalar dataset for THIS W from the shared labelsets (cheap;
    # only feature-building repeats, not the backward-Dijkstra collection).
    X, y, mask = _build_scalar_dataset(labelsets, cfg, window_w=W)

    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    logs_dir = ensure_dir(out_dir / "logs")

    # Masked tensors once: training only on reachable space-time states.
    mask_np = np.asarray(mask, dtype=np.bool_)
    n_total = int(mask_np.shape[0])
    n_keep = int(mask_np.sum())
    if n_keep == 0:
        raise RuntimeError("scalar train: no reachable (masked) examples to train on")
    Xk = torch.from_numpy(np.ascontiguousarray(X[mask_np])).to(device)
    yk = torch.from_numpy(np.ascontiguousarray(y[mask_np])).to(device)
    tag = suffix or ""
    print(
        f"[{now_str()}] C8 train: scalar{tag} dataset M={n_total} reachable={n_keep} "
        f"({100.0 * n_keep / max(1, n_total):.1f}%) token_dim={token_dim} W={W} seq_len={W + 1}",
        flush=True,
    )

    ckpts: Dict[str, Path] = {}
    for name in names:
        bb = _scalar_backbone_cfg(name)
        model = C.ContinuousHeuristicModel(
            bb, token_dim=token_dim, max_norm_residual=max_norm_residual
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(train_cfg.weight_decay))
        model.train()
        log_lines: List[str] = []
        for ep in range(epochs):
            perm = torch.randperm(n_keep, device=device)
            ep_loss = 0.0
            ep_count = 0
            for i in range(0, n_keep, batch_size):
                idx = perm[i : i + batch_size]
                xb = Xk[idx]
                yb = yk[idx]
                pred = model(xb)
                if not torch.isfinite(pred).all():
                    raise FloatingPointError(f"scalar train {name}: non-finite prediction")
                loss = F.smooth_l1_loss(pred, yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"scalar train {name}: non-finite loss")
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                ep_loss += float(loss.detach().cpu()) * int(idx.numel())
                ep_count += int(idx.numel())
            mean_loss = ep_loss / max(1, ep_count)
            line = f"epoch={ep} loss={mean_loss:.6f}"
            log_lines.append(line)
            print(f"[{now_str()}] C8 train: scalar{tag} {name} {line}", flush=True)

        ckpt_path = ckpt_dir / f"c8_scalar__{name}{suffix}.pt"
        payload = {
            "model": model.state_dict(),
            "backbone_cfg": asdict(bb),
            "window_w": int(W),
            "k_patrollers": int(cfg.k_patrollers),
            "token_dim": token_dim,
            "max_norm_residual": max_norm_residual,
            "backbone": name,
        }
        torch.save(payload, ckpt_path)
        ckpts[name] = ckpt_path
        log_path = logs_dir / f"c8_scalar__{name}{suffix}.log"
        log_path.write_text("\n".join(log_lines) + "\n")
        print(f"[{now_str()}] C8 train: scalar{tag} {name} -> {ckpt_path}", flush=True)

    return ckpts


# ---------------------------------------------------------------------------
# Task 10b — field temporal dataset + training
# ---------------------------------------------------------------------------
#
# The field models predict a per-cell residual GRID, but we supervise at the SAME
# PRM-node space-time residual target the scalar arm (T10a) uses
# (`node_residual[:, t]`, reachable-masked). We sample the predicted grid at the
# PRM node grid-cells and supervise there — this matches how
# `ValueFieldTemporalProvider.h_table` queries the field at eval time (it samples
# the predicted grid at PRM nodes via `C6.interpolate_grid_values`). Documented
# deviation from C6's dense full-grid supervision: the field is node-supervised
# here for (a) target consistency with the scalar arm and (b) tractability — it
# avoids an expensive grid-level space-time DP. (Minor sampling difference: train
# uses nearest-cell gather via `C6.point_to_cell`; eval uses bilinear interp via
# `C6.interpolate_grid_values`. Both reference the SAME cell convention — see the
# gather-convention note in the dataset below.)


def _build_field_node_cells(rm, side_len: float, grid_size: int) -> np.ndarray:
    """Integer PRM-node grid-cells, (N, 2) = [ix, iy] per node.

    Gather-convention (CRITICAL): `C6.point_to_cell(p, side, G)` returns
    ``(ix, iy)`` where ``ix = floor(p[0]/side*G)`` (x-index) and
    ``iy = floor(p[1]/side*G)`` (y-index). C6's predicted residual grid is
    indexed ``grid[ix, iy]`` — `C6.interpolate_grid_values` (the eval-time
    sampler) reads ``grid[x0, y0]`` with ``x0`` from ``p[0]`` and ``y0`` from
    ``p[1]``. So the training-time nearest-cell gather must be ``pred[b, ix, iy]``
    to reference the same cell the provider samples at eval time.
    """
    import continuous_prm_c6_heatmap_value_field as C6
    pts = np.asarray(rm.points, dtype=np.float64)
    cells = np.empty((pts.shape[0], 2), dtype=np.int64)
    for i in range(pts.shape[0]):
        ix, iy = C6.point_to_cell(pts[i], float(side_len), int(grid_size))
        cells[i, 0] = int(ix)
        cells[i, 1] = int(iy)
    return cells


class _FieldTemporalDataset:
    """Torch-style dataset of per-(world, t) field-training samples.

    Each sample yields:
      occ_stack  (8+W, G, G) float32 — `P.build_field_occupancy_stack` for (world, t)
      node_cells (N, 2)      int64   — [ix, iy] grid-cells per PRM node (const over t)
      target     (N,)        float32 — `node_residual[:, t]` (T_scale-normalized)
      mask       (N,)        bool    — `reachable[:, t]`

    Occupancy stacks are built ON DEMAND in `__getitem__` to bound memory (we do
    NOT materialize all (sum_w t_max+1) stacks up front). `P.build_field_occupancy_stack`
    owns the full per-(world, t) render (the W+1 patroller frames + the 7 static
    C6 channels via `make_heatmap_example`), so we keep that as the single source
    of truth to stay byte-identical with the eval-time provider path; we only cache
    the per-world node grid-cells (constant over t) up front.

    `window_w` overrides cfg.window_w (default None -> cfg.window_w). W=0 yields an
    8-channel occupancy stack (the time-blind ablation variant).
    """

    def __init__(self, labelsets: List[dict], cfg: C8Config, window_w: Optional[int] = None):
        self.cfg = cfg
        self.W = int(cfg.window_w if window_w is None else window_w)
        self.G = int(cfg.grid_size)
        self.labelsets = labelsets
        self._cells: List[np.ndarray] = []        # (N, 2) int64 per world (const over t)
        self._index: List[Tuple[int, int]] = []   # flat (world_idx, t)
        for w_idx, ls in enumerate(labelsets):
            world = ls["world"]
            self._cells.append(
                _build_field_node_cells(ls["rm"], float(world.side_len), self.G)
            )
            t_max = int(ls["params"]["t_max"])
            for t in range(t_max + 1):
                self._index.append((w_idx, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        w_idx, t = self._index[i]
        ls = self.labelsets[w_idx]
        world, dyn = ls["world"], ls["dyn"]
        dt = float(ls["params"]["dt"])
        # Build the occupancy stack ON DEMAND (bounds memory). This re-renders the
        # W+1 patroller frames for (world, t) — matches the eval-time provider path.
        occ = P.build_field_occupancy_stack(world, dyn, self.G, t, self.W, dt)
        occ = np.ascontiguousarray(occ, dtype=np.float32)  # (8+W, G, G)
        cells = self._cells[w_idx]                          # (N, 2) int64
        target = ls["node_residual"][:, t].astype(np.float32)  # (N,)
        mask = ls["reachable"][:, t].astype(np.bool_)          # (N,)
        return occ, cells, target, mask


def _field_collate(batch):
    """Collate variable-N field samples into a padded batch.

    Different worlds have different roadmap node counts N, so node_cells/target/
    mask are padded to the batch-max N and a row-validity is folded into `mask`
    (padded rows get mask=False so they never contribute to the loss).

    Returns torch tensors:
      occ    (B, 8+W, G, G) float32
      cells  (B, Nmax, 2)   long
      target (B, Nmax)      float32
      mask   (B, Nmax)      bool
    """
    import torch
    occs, cells_l, targets, masks = zip(*batch)
    B = len(occs)
    Nmax = max(int(c.shape[0]) for c in cells_l)
    occ = torch.from_numpy(np.stack(occs, axis=0))  # (B, 8+W, G, G)
    cells = torch.zeros((B, Nmax, 2), dtype=torch.long)
    target = torch.zeros((B, Nmax), dtype=torch.float32)
    mask = torch.zeros((B, Nmax), dtype=torch.bool)
    for b in range(B):
        n = int(cells_l[b].shape[0])
        cells[b, :n] = torch.from_numpy(np.ascontiguousarray(cells_l[b]))
        target[b, :n] = torch.from_numpy(targets[b])
        mask[b, :n] = torch.from_numpy(masks[b])
    return occ, cells, target, mask


def _train_field(
    labelsets: List[dict],
    cfg: C8Config,
    device,
    window_w: Optional[int] = None,
    suffix: str = "",
) -> Dict[str, Path]:
    """Train each field backbone on the masked PRM-node space-time residual target.

    For each (world, t): forward the occupancy stack through a C6 field model
    (in_channels = 8 + window_w) -> predicted residual grid (B, G, G) via
    `C6.model_output_residual(model(x))` (the same forward path
    `C6.predict_residual_grid` uses); GATHER the predicted residual at the PRM node
    grid-cells -> pred_nodes (B, N); masked smooth-L1 against node_residual[:, t].

    `window_w` overrides cfg.window_w (default None -> cfg.window_w); W=0 gives an
    8-channel occupancy stack -> C6.build_model(name, in_channels=8) (the time-blind
    ablation variant). `suffix` is appended to the checkpoint name:
    c8_field__{backbone}{suffix}.pt (e.g. "_blind"). Built from the SHARED labelsets
    (no re-collection; only occupancy-stack rendering + training repeats).

    Saves a payload that lets Task 11 rebuild a ValueFieldTemporalProvider. Returns
    {backbone: ckpt_path}. Empty cfg.field_backbones -> trains nothing (returns {}).
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import continuous_prm_c6_heatmap_value_field as C6

    names = parse_csv(cfg.field_backbones)
    if not names:
        return {}

    out_dir = Path(cfg.out_dir)
    W = int(cfg.window_w if window_w is None else window_w)
    in_channels = 8 + W
    train_cfg = C.TrainingConfig()
    lr = float(train_cfg.lr)
    weight_decay = float(train_cfg.weight_decay)
    epochs = int(cfg.epochs)
    batch_size = 8  # grids are heavy; small batch keeps CPU/GPU memory modest
    grad_clip = 1.0
    tag = suffix or ""

    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    logs_dir = ensure_dir(out_dir / "logs")

    dataset = _FieldTemporalDataset(labelsets, cfg, window_w=W)
    n_samples = len(dataset)
    if n_samples == 0:
        raise RuntimeError("field train: no (world, t) samples to train on")
    print(
        f"[{now_str()}] C8 train: field{tag} dataset samples={n_samples} "
        f"(worlds={len(labelsets)}) in_channels={in_channels} grid={cfg.grid_size} W={W}",
        flush=True,
    )

    ckpts: Dict[str, Path] = {}
    for name in names:
        model = C6.build_model(name, in_channels=in_channels).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=_field_collate,
        )
        log_lines: List[str] = []
        for ep in range(epochs):
            ep_loss = 0.0
            ep_count = 0
            for occ, cells, target, mask in loader:
                occ = occ.to(device=device, dtype=torch.float32)
                cells = cells.to(device)
                target = target.to(device=device, dtype=torch.float32)
                mask = mask.to(device)
                # Predicted residual grid (B, G, G) via C6's model->residual path.
                pred_grid = C6.model_output_residual(model(occ))
                if not torch.isfinite(pred_grid).all():
                    raise FloatingPointError(f"field train {name}: non-finite grid")
                # GATHER at PRM node grid-cells. Convention (see
                # _build_field_node_cells): cells[..., 0]=ix (x-index),
                # cells[..., 1]=iy (y-index); grid is indexed [ix, iy], matching the
                # eval-time sampler C6.interpolate_grid_values(grid[x0, y0]).
                B, G, _ = pred_grid.shape
                ix = cells[..., 0].clamp(0, G - 1)   # (B, N) x-index
                iy = cells[..., 1].clamp(0, G - 1)   # (B, N) y-index
                flat = pred_grid.reshape(B, G * G)
                lin = ix * G + iy                    # row-major [ix, iy] -> ix*G + iy
                pred_nodes = torch.gather(flat, 1, lin)  # (B, N)
                m = mask.bool()
                if not bool(m.any()):
                    continue
                loss = F.smooth_l1_loss(pred_nodes[m], target[m])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"field train {name}: non-finite loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                nm = int(m.sum().item())
                ep_loss += float(loss.detach().cpu()) * nm
                ep_count += nm
            mean_loss = ep_loss / max(1, ep_count)
            line = f"epoch={ep} loss={mean_loss:.6f}"
            log_lines.append(line)
            print(f"[{now_str()}] C8 train: field{tag} {name} {line}", flush=True)

        ckpt_path = ckpt_dir / f"c8_field__{name}{suffix}.pt"
        payload = {
            "model": model.state_dict(),
            "in_channels": in_channels,
            "window_w": int(W),
            "grid_size": int(cfg.grid_size),
            "backbone": name,
        }
        torch.save(payload, ckpt_path)
        ckpts[name] = ckpt_path
        log_path = logs_dir / f"c8_field__{name}{suffix}.log"
        log_path.write_text("\n".join(log_lines) + "\n")
        print(f"[{now_str()}] C8 train: field{tag} {name} -> {ckpt_path}", flush=True)

    return ckpts


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def run_collect(cfg: C8Config, out_dir: Path) -> Tuple[List[dict], Dict[str, int]]:
    """Collect space-time labelsets over train_tasks x train_worlds and report counts."""
    print(
        f"[{now_str()}] C8 collect: tasks={cfg.train_tasks} train_worlds={cfg.train_worlds}",
        flush=True,
    )
    labelsets, counts = _collect_labelsets(cfg)
    total = sum(counts.values())
    print(
        f"[{now_str()}] C8 collect: total usable worlds={total} per-suite={counts}",
        flush=True,
    )
    return labelsets, counts


def run_train(cfg: C8Config, out_dir: Path) -> Dict[str, object]:
    """Collect labels ONCE, then train the time-aware AND time-blind (W=0) variants
    of the scalar (Task 10a) and field (Task 10b) models.

    All four training passes supervise on the SAME shared `labelsets` (each carries
    world/dyn/node_residual/reachable from the EXPENSIVE backward space-time
    Dijkstra, run once in run_collect). The four passes only re-do the CHEAP work
    (feature-building / occupancy-stack rendering + model training):
      - scalar       W=cfg.window_w  suffix=""        -> c8_scalar__{bb}.pt
      - scalar_blind W=0             suffix="_blind"  -> c8_scalar__{bb}_blind.pt
      - field        W=cfg.window_w  suffix=""        -> c8_field__{bb}.pt
      - field_blind  W=0             suffix="_blind"  -> c8_field__{bb}_blind.pt
    The time-blind variants are the separately-trained W=0 models the
    time-aware-vs-time-blind ablation (the spotlight comparison) requires.

    A family is skipped (returns {}) when its backbone list is empty.
    train_manifest.json lists all four groups: scalar, scalar_blind, field,
    field_blind.
    """
    device = _pick_device(cfg)
    print(f"[{now_str()}] C8 train: device={device}", flush=True)

    # Collect ONCE. The backward space-time Dijkstra per world dominates cost; all
    # four training passes below reuse these labelsets (no re-collection).
    labelsets, counts = run_collect(cfg, out_dir)
    if not labelsets:
        raise RuntimeError(
            "C8 train: no usable training worlds collected "
            f"(tasks={cfg.train_tasks}, train_worlds={cfg.train_worlds})"
        )

    W = int(cfg.window_w)

    # ---- Scalar temporal models (Task 10a): time-aware + time-blind ----
    scalar_ckpts = _train_scalar(labelsets, cfg, device, window_w=W, suffix="")
    scalar_blind_ckpts = _train_scalar(labelsets, cfg, device, window_w=0, suffix="_blind")

    # ---- Field temporal models (Task 10b): time-aware + time-blind ----
    field_ckpts = _train_field(labelsets, cfg, device, window_w=W, suffix="")
    field_blind_ckpts = _train_field(labelsets, cfg, device, window_w=0, suffix="_blind")

    def _ckpt_map(ckpts: Dict[str, Path]) -> Dict[str, str]:
        return {name: str(p) for name, p in ckpts.items() if Path(p).exists()}

    manifest = {
        "stage": "c8_train",
        "timestamp": now_str(),
        "train_tasks": parse_csv(cfg.train_tasks),
        "train_worlds": int(cfg.train_worlds),
        "usable_worlds_per_suite": counts,
        "usable_worlds_total": int(sum(counts.values())),
        "window_w": W,
        "k_patrollers": int(cfg.k_patrollers),
        "token_dim": _scalar_token_dim(cfg),
        "scalar": {
            "backbones": parse_csv(cfg.scalar_backbones),
            "epochs": int(cfg.epochs),
            "window_w": W,
            "checkpoints": _ckpt_map(scalar_ckpts),
        },
        "scalar_blind": {
            "backbones": parse_csv(cfg.scalar_backbones),
            "epochs": int(cfg.epochs),
            "window_w": 0,
            "checkpoints": _ckpt_map(scalar_blind_ckpts),
        },
        "field": {
            "backbones": parse_csv(cfg.field_backbones),
            "epochs": int(cfg.epochs),
            "window_w": W,
            "in_channels": 8 + W,
            "grid_size": int(cfg.grid_size),
            "checkpoints": _ckpt_map(field_ckpts),
        },
        "field_blind": {
            "backbones": parse_csv(cfg.field_backbones),
            "epochs": int(cfg.epochs),
            "window_w": 0,
            "in_channels": 8,
            "grid_size": int(cfg.grid_size),
            "checkpoints": _ckpt_map(field_blind_ckpts),
        },
    }
    manifest_path = Path(out_dir) / "train_manifest.json"
    write_json(manifest_path, manifest)
    print(f"[{now_str()}] C8 train: wrote manifest -> {manifest_path}", flush=True)
    for group in ("scalar", "scalar_blind", "field", "field_blind"):
        print(
            f"[{now_str()}] C8 train: {group} checkpoints="
            f"{list(manifest[group]['checkpoints'].values())}",
            flush=True,
        )
    return manifest


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "C8 Dynamics Comparison: scalar-temporal vs value-field-temporal "
            "heuristics on dynamic moving-obstacle suites"
        )
    )
    p.add_argument(
        "--mode",
        choices=("collect", "train", "eval", "analyze", "full", "calibrate"),
        default="full",
    )
    p.add_argument(
        "--scale",
        choices=("local", "cluster"),
        default="local",
    )
    p.add_argument("--out-dir", type=str, default="runs/c8_local")
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--roadmap-nodes", type=int, default=192)
    p.add_argument("--roadmap-k", type=int, default=7)
    p.add_argument("--train-tasks", type=str, default="C_dyn_maze,C_dyn_rooms,C_dyn_spiral")
    p.add_argument(
        "--eval-suites",
        type=str,
        default=(
            "C_dyn_maze,C_dyn_rooms,C_dyn_spiral,"
            "C_dyn_maze_dense,C_dyn_crossing,C_dyn_rooms_large"
        ),
    )
    p.add_argument("--scalar-backbones", type=str, default="hrm,onlstm")
    p.add_argument("--field-backbones", type=str, default="unet,onlstm,hrm")
    p.add_argument("--budgets", type=str, default="2000")
    # Preset-filled fields default to 0/"" so the preset can fill them
    p.add_argument("--w-values", type=str, default="")
    p.add_argument("--eval-worlds", type=int, default=0)
    p.add_argument("--train-worlds", type=int, default=0)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--budget-grid-size", type=int, default=0)
    p.add_argument("--no-figures", action="store_true")
    # Dynamics-specific knobs
    p.add_argument(
        "--window-w",
        type=int,
        default=8,
        help="Rollout window length for temporal heuristics (time steps looked back).",
    )
    p.add_argument(
        "--k-patrollers",
        type=int,
        default=4,
        help="Number of nearest patrollers included in scalar feature vectors.",
    )
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> C8Config:
    return C8Config(
        grid_size=int(args.grid_size),
        roadmap_nodes=int(args.roadmap_nodes),
        roadmap_k=int(args.roadmap_k),
        train_tasks=str(args.train_tasks),
        eval_suites=str(args.eval_suites),
        scalar_backbones=str(args.scalar_backbones),
        field_backbones=str(args.field_backbones),
        budgets=str(args.budgets),
        w_values=str(args.w_values),
        eval_worlds=int(args.eval_worlds),
        train_worlds=int(args.train_worlds),
        epochs=int(args.epochs),
        seed=int(args.seed),
        scale=str(args.scale),
        mode=str(args.mode),
        out_dir=str(args.out_dir),
        cpu=bool(args.cpu),
        budget_grid_size=int(args.budget_grid_size),
        make_figures=not bool(args.no_figures),
        window_w=int(args.window_w),
        k_patrollers=int(args.k_patrollers),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    cfg = apply_scale_preset(cfg)

    # Install all dynamic suites (composes on C7 hard maps + C5 hard runtime)
    # into the common registry so C.build_anchor_specs() returns all six C8
    # dynamic suites alongside the static C7/C5 suites.
    M8.install_c8_dynamic_maps()

    out_dir = ensure_dir(cfg.out_dir)

    print(
        f"[{now_str()}] C8 mode={cfg.mode} scale={cfg.scale} "
        f"out_dir={out_dir} cpu={cfg.cpu} "
        f"eval_worlds={cfg.eval_worlds} train_worlds={cfg.train_worlds} "
        f"epochs={cfg.epochs} w_values={cfg.w_values} "
        f"window_w={cfg.window_w} k_patrollers={cfg.k_patrollers}",
        flush=True,
    )

    if cfg.mode == "collect":
        run_collect(cfg, out_dir)
    elif cfg.mode == "train":
        run_train(cfg, out_dir)
    elif cfg.mode == "eval":
        raise NotImplementedError("C8 eval: Task 11")
    elif cfg.mode == "calibrate":
        raise NotImplementedError("C8 calibrate: Task 12")
    elif cfg.mode == "analyze":
        raise NotImplementedError("C8 analyze: Task 13")
    elif cfg.mode == "full":
        raise NotImplementedError("C8 full: Tasks 10-13")
    else:
        raise ValueError(f"unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
