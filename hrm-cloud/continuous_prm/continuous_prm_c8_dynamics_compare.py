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

    # Scalar dataset cap: subsample to at most this many REACHABLE samples for
    # tractable local training. 0 = no cap (use all reachable samples).
    scalar_max_samples: int = 250000

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

    # Seeded subsample cap: restrict to at most scalar_max_samples REACHABLE
    # entries so that scalar training epochs are tractable on local machines.
    cap = int(cfg.scalar_max_samples)
    if cap > 0:
        idx_reachable = np.where(mask)[0]
        n_reachable = len(idx_reachable)
        if n_reachable > cap:
            rng = np.random.default_rng(int(cfg.seed))
            idx_keep = np.sort(rng.choice(idx_reachable, cap, replace=False))
            X = X[idx_keep]
            y = y[idx_keep]
            mask = np.ones(cap, dtype=np.bool_)
            print(f"[c8] scalar dataset capped: kept {cap}/{n_reachable}", flush=True)

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
        self._static_base: List[tuple] = []       # (static_occ, static7) per world (const over t)
        self._index: List[Tuple[int, int]] = []   # flat (world_idx, t)
        for w_idx, ls in enumerate(labelsets):
            world = ls["world"]
            self._cells.append(
                _build_field_node_cells(ls["rm"], float(world.side_len), self.G)
            )
            # Cache the per-world, t-independent field static base (grid Dijkstra +
            # static channels) ONCE — do NOT recompute it per sample in __getitem__.
            self._static_base.append(P.compute_field_static_base(world, self.G))
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
        # Reuse the cached per-world static base (grid Dijkstra computed once/world).
        occ = P.build_field_occupancy_stack(
            world, dyn, self.G, t, self.W, dt, static_base=self._static_base[w_idx]
        )
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
# Task 11 — eval mode (matched, sharded space-time arm evaluation)
# ---------------------------------------------------------------------------
#
# Mirrors C7's run_eval (continuous_prm_c7_integration_compare.run_eval): build
# every arm's provider ONCE, generate matched dynamic worlds per suite from a
# seeded retry loop, run all arms on each shared world via
# P.run_world_arms_spacetime, write per-suite shard CSVs, then merge bounded to
# exactly the suites this run evaluated.


def _load_eval_providers(cfg: C8Config, out_dir: Path, device) -> Dict[str, "P.SpaceTimeHeuristicProvider"]:
    """Build every arm's provider ONCE, keyed by provider.name.

    Always present: euclid, oracle. Plus, for each scalar/field backbone, the
    time-aware ("") and time-blind ("_blind") variants whose checkpoints exist
    under out_dir/'checkpoints' (saved by T10a/b/c). Missing checkpoints are
    skipped with a log line. The provider's .name already encodes "_blind", so
    keys are unique (scalar_<bb>, scalar_<bb>_blind, field_<bb>, field_<bb>_blind).
    """
    import torch
    import continuous_prm_c6_heatmap_value_field as C6

    providers: Dict[str, "P.SpaceTimeHeuristicProvider"] = {}
    eu = P.EuclidTimeProvider(); providers[eu.name] = eu     # "euclid"
    orc = P.OracleProvider(); providers[orc.name] = orc      # "oracle"

    ckdir = Path(out_dir) / "checkpoints"

    # Scalar temporal models (T10a + T10c blind).
    for bb in parse_csv(cfg.scalar_backbones):
        for suffix, blind in (("", False), ("_blind", True)):
            ck = ckdir / f"c8_scalar__{bb}{suffix}.pt"
            if not ck.exists():
                print(f"[c8] skip scalar {bb}{suffix}: {ck} missing", flush=True)
                continue
            pl = torch.load(ck, map_location="cpu")
            bbcfg = C.BackboneConfig(**pl["backbone_cfg"])
            model = C.ContinuousHeuristicModel(
                bbcfg, token_dim=pl["token_dim"], max_norm_residual=pl["max_norm_residual"]
            )
            model.load_state_dict(pl["model"]); model.to(device).eval()
            # NOTE: a "_blind" checkpoint stores window_w=0 AND we pass time_blind=True;
            # both are consistent (W=0). The provider name encodes "_blind".
            prov = P.ScalarTemporalProvider(
                model, device, bb, pl["window_w"], pl["k_patrollers"],
                pl["max_norm_residual"], time_blind=blind,
            )
            providers[prov.name] = prov                      # scalar_<bb> / scalar_<bb>_blind

    # Value-field temporal models (T10b + T10c blind).
    for bb in parse_csv(cfg.field_backbones):
        for suffix, blind in (("", False), ("_blind", True)):
            ck = ckdir / f"c8_field__{bb}{suffix}.pt"
            if not ck.exists():
                print(f"[c8] skip field {bb}{suffix}: {ck} missing", flush=True)
                continue
            pl = torch.load(ck, map_location="cpu")
            model = C6.build_model(bb, in_channels=pl["in_channels"])
            model.load_state_dict(pl["model"]); model.to(device).eval()
            prov = P.ValueFieldTemporalProvider(
                model, pl["grid_size"], device, bb, pl["window_w"], time_blind=blind,
            )
            providers[prov.name] = prov                      # field_<bb> / field_<bb>_blind

    return providers


def _load_calibration(out_dir: Path) -> dict:
    """Read out_dir/calibration.json (written by T12), or {} if absent/unreadable.

    Expected shape: {"budgets": {suite: [b1, b2, ...]}, ...}. run_eval falls back
    to parse_int_csv(cfg.budgets) for any suite absent from the calibration map.
    """
    import json
    calib_path = Path(out_dir) / "calibration.json"
    if not calib_path.exists():
        return {}
    try:
        return json.loads(calib_path.read_text())
    except (ValueError, OSError) as exc:
        print(f"[c8] eval: failed to read {calib_path} ({exc}); using fallback budgets", flush=True)
        return {}


def iter_dynamic_worlds(suite, suite_idx, cfg: C8Config, n_worlds, retry=30):
    """Yield (world_index, world, dyn, rm) for up to n_worlds connected dynamic
    worlds, using a deterministic eval seed formula. Shared by eval (and reused by
    T12 calibrate) so both observe identical worlds for a given (suite, world_idx).

    Skip rules: invalid dynamic world (build returns None), disconnected PRM (start
    node 0 not connected to goal). Space-time solvability is NOT pre-checked here:
    run_world_arms_spacetime reports suboptimality=nan for unsolvable worlds via its
    own backward Dijkstra, so we rely on that rather than an extra DP pass.
    """
    roadmap_cfg = C.RoadmapConfig(n_nodes=cfg.roadmap_nodes, k_neighbors=cfg.roadmap_k)
    valid = 0
    attempt = 0
    while valid < n_worlds and attempt < n_worlds * retry:
        seed = int(cfg.seed) + 880_000 + 1_000_003 * (suite_idx + 1) + (valid + 1) * 7919 + attempt
        attempt += 1
        res = M8.build_dynamic_world(suite, seed)
        if res is None:
            continue
        world, dyn = res
        rm = C.build_prm(world, roadmap_cfg, seed=seed)
        if rm is None or not bool(rm.connected_to_goal[0]):
            continue
        yield valid, world, dyn, rm
        valid += 1


def run_eval(cfg: C8Config, out_dir: Path, device, providers: Dict[str, "P.SpaceTimeHeuristicProvider"]) -> Path:
    """Matched, sharded multi-arm space-time eval across eval suites; write per-suite
    shard CSVs + a merged raw CSV. Returns the merged-CSV path.

    Worlds+PRMs+dynamics are generated per suite from a seeded retry loop and shared
    across ALL arms (matched). Per-suite budgets come from calibration.json (T12) if
    present, else parse_int_csv(cfg.budgets). Records carry provider/mode/w/budget/
    found/expansions/arrival/optimal_arrival/suboptimality/closed/nonfinite (from
    run_world_arms_spacetime) plus suite/world_index added here; world.meta is NOT
    spread in (segment by suite).
    """
    import csv

    out_dir = Path(out_dir)
    w_values = [float(x) for x in parse_csv(cfg.w_values)]
    calib = _load_calibration(out_dir)
    print(
        f"[{now_str()}] C8 eval: providers={sorted(providers)} "
        f"eval_worlds={cfg.eval_worlds} w_values={w_values} suites={cfg.eval_suites}",
        flush=True,
    )

    for suite_idx, suite in enumerate(parse_csv(cfg.eval_suites)):
        params = M8.dynamics_params(suite)
        v_agent = float(params["v_agent"])
        dt = float(params["dt"])
        t_max = int(params["t_max"])
        budgets = calib.get("budgets", {}).get(suite) or parse_int_csv(cfg.budgets)
        budgets = [int(b) for b in budgets]
        records = []
        valid = 0
        for wi, world, dyn, rm in iter_dynamic_worlds(suite, suite_idx, cfg, cfg.eval_worlds):
            recs = P.run_world_arms_spacetime(
                world, rm, dyn, providers, budgets, w_values,
                v_agent, dt, t_max, goal_idx=1, start_idx=0,
            )
            for r in recs:
                r["suite"] = suite
                r["world_index"] = wi
            records.extend(recs)
            valid = wi + 1
        if valid < cfg.eval_worlds:
            print(f"[c8] WARNING: {suite} under-filled: {valid}/{cfg.eval_worlds} worlds", flush=True)
        shard = ensure_dir(Path(out_dir) / "results" / "_shards" / "c8" / suite) / "shard_0000.csv"
        write_csv(shard, records)
        print(f"[c8] eval {suite}: {valid} worlds, {len(records)} arm-records -> {shard}", flush=True)

    # Merge into one raw CSV. Bound the merge to EXACTLY the suites this run
    # evaluated — a glob would silently pull in stale shards from prior runs with
    # different --eval-suites (or an interrupted run), corrupting the comparison.
    merged_rows: list = []
    for suite in parse_csv(cfg.eval_suites):
        shard_csv = Path(out_dir) / "results" / "_shards" / "c8" / suite / "shard_0000.csv"
        if shard_csv.exists():
            with open(shard_csv, newline="") as fh:
                merged_rows.extend(csv.DictReader(fh))
    merged_path = Path(out_dir) / "results" / "continuous_prm_c8_eval_raw.csv"
    write_csv(merged_path, merged_rows)
    print(f"[{now_str()}] C8 eval: merged {len(merged_rows)} rows -> {merged_path}", flush=True)
    return merged_path


# ---------------------------------------------------------------------------
# Task 12 — calibrate mode (Gate 1: per-suite binding-budget band selection)
# ---------------------------------------------------------------------------
#
# Mirrors C7's run_calibrate (continuous_prm_c7_integration_compare.run_calibrate)
# but uses the space-time search (ST.space_time_astar_prm) and space-time
# heuristic providers (P.EuclidTimeProvider / P.OracleProvider .h_table).
# Per-suite dynamics params (v_agent, dt, t_max) come from M8.dynamics_params.
# Worlds come from iter_dynamic_worlds (the same generator run_eval uses) so
# calibrate and eval observe IDENTICAL worlds for each (suite, world_idx).

# Space-time expansion budgets. Larger than C7 because the space-time graph has
# ~t_max times as many states; the [0.45, 0.70] euclid-success targets live in
# this range for the six C8 dynamic suites.
CALIB_GRID = [150, 250, 400, 600, 900, 1300, 1800, 2500, 3500]


def _select_band_budgets_c8(grid, euclid_success, k):
    """Pick k in-band budgets from `grid` by matching euclid success to targets.

    Targets are k points evenly spaced in [0.45, 0.70]. For each target, pick
    the grid budget whose euclid_success is closest (ties -> smaller budget,
    since `grid` is ascending). Dedupe preserving ascending budget order; if
    dedupe leaves < k, fill with the next-closest unused grid budgets (by
    distance to the nearest target). Returns a sorted ascending list of length
    min(k, len(grid)).
    """
    k = max(1, min(int(k), len(grid)))
    targets = np.linspace(0.45, 0.70, k)
    chosen: list = []
    for t in targets:
        best = int(np.argmin([abs(euclid_success[i] - t) for i in range(len(grid))]))
        if grid[best] not in chosen:
            chosen.append(grid[best])
    if len(chosen) < k:
        def nearest_target_dist(i):
            return min(abs(euclid_success[i] - t) for t in targets)
        remaining = sorted(
            (i for i in range(len(grid)) if grid[i] not in chosen),
            key=lambda i: (nearest_target_dist(i), grid[i]),
        )
        for i in remaining:
            if len(chosen) >= k:
                break
            chosen.append(grid[i])
    return sorted(chosen)


def run_calibrate(cfg: C8Config, out_dir: Path, device=None) -> Path:
    """Gate 1: sweep CALIB_GRID with ONLY Euclid-time + Oracle space-time A*
    per suite and pick the binding-budget band. Writes out_dir/calibration.json
    (the per-suite "budgets" map T11's run_eval consumes) plus diagnostics.

    No learned models are loaded — euclid-time and oracle are pure geometry /
    graph, so this runs fast on CPU. Worlds come from iter_dynamic_worlds, so
    the budgets are calibrated on the SAME worlds eval will later score.

    Per-suite dynamics params (v_agent, dt, t_max) come from M8.dynamics_params;
    h-tables are built once per world via EuclidTimeProvider.h_table /
    OracleProvider.h_table, then reused across all budgets for that world.
    """
    out_dir = Path(out_dir)
    M8.install_c8_dynamic_maps()

    euclid_provider = P.EuclidTimeProvider()
    oracle_provider = P.OracleProvider()

    print(
        f"[{now_str()}] C8 calibrate: grid={CALIB_GRID} eval_worlds={cfg.eval_worlds} "
        f"budget_grid_size={cfg.budget_grid_size} suites={cfg.eval_suites}",
        flush=True,
    )

    budgets_map: Dict[str, list] = {}
    measurements: Dict[str, list] = {}
    warnings: list = []

    for suite_idx, suite in enumerate(parse_csv(cfg.eval_suites)):
        params = M8.dynamics_params(suite)
        v_agent = float(params["v_agent"])
        dt = float(params["dt"])
        t_max = int(params["t_max"])

        # Per-budget tallies: found counts + sum of expansions over SOLVED instances
        # (mean expansions computed only where the method actually reached the goal).
        euclid_found = [0] * len(CALIB_GRID)
        oracle_found = [0] * len(CALIB_GRID)
        euclid_exp_sum = [0.0] * len(CALIB_GRID)
        oracle_exp_sum = [0.0] * len(CALIB_GRID)
        n_worlds = 0

        for _, world, dyn, rm in iter_dynamic_worlds(suite, suite_idx, cfg, cfg.eval_worlds):
            # Build h-tables once per world; reused across all budgets.
            euclid_ht = euclid_provider.h_table(world, rm, dyn, v_agent, dt, t_max, goal_idx=1)
            oracle_ht = oracle_provider.h_table(world, rm, dyn, v_agent, dt, t_max, goal_idx=1)

            for bi, b in enumerate(CALIB_GRID):
                re = ST.space_time_astar_prm(
                    rm.adj, rm.points, dyn, euclid_ht, int(b), v_agent, dt, t_max, 0, 1
                )
                if re["found"]:
                    euclid_found[bi] += 1
                    euclid_exp_sum[bi] += int(re["expansions"])

                ro = ST.space_time_astar_prm(
                    rm.adj, rm.points, dyn, oracle_ht, int(b), v_agent, dt, t_max, 0, 1
                )
                if ro["found"]:
                    oracle_found[bi] += 1
                    oracle_exp_sum[bi] += int(ro["expansions"])

            n_worlds += 1

        if n_worlds < cfg.eval_worlds:
            msg = f"{suite}: under-filled {n_worlds}/{cfg.eval_worlds} worlds"
            print(f"[c8] WARNING: calibrate {msg}", flush=True)
            warnings.append(msg)

        denom = max(1, n_worlds)
        euclid_success = [f / denom for f in euclid_found]
        oracle_success = [f / denom for f in oracle_found]
        euclid_exp = [
            (euclid_exp_sum[bi] / euclid_found[bi]) if euclid_found[bi] > 0 else None
            for bi in range(len(CALIB_GRID))
        ]
        oracle_exp = [
            (oracle_exp_sum[bi] / oracle_found[bi]) if oracle_found[bi] > 0 else None
            for bi in range(len(CALIB_GRID))
        ]

        chosen = _select_band_budgets_c8(CALIB_GRID, euclid_success, cfg.budget_grid_size)
        idx_of = {b: i for i, b in enumerate(CALIB_GRID)}
        chosen_eu = [round(euclid_success[idx_of[b]], 3) for b in chosen]
        chosen_eu_full = [euclid_success[idx_of[b]] for b in chosen]

        def _headroom_at(b):
            """1 - oracle_exp/euclid_exp on euclid-solved means; None if not computable."""
            i = idx_of[b]
            ee, oe = euclid_exp[i], oracle_exp[i]
            if ee is None or oe is None or ee <= 0:
                return None
            return 1.0 - (oe / ee)

        chosen_headroom = [_headroom_at(b) for b in chosen]
        chosen_eu_exp = [euclid_exp[idx_of[b]] for b in chosen]
        chosen_or_exp = [oracle_exp[idx_of[b]] for b in chosen]

        # Warn if no chosen budget has euclid success in [0.35, 0.95] (no usable
        # difficulty point) OR best chosen headroom < 0.15 (oracle barely cheaper
        # than euclid, leaving little room for learned models to help).
        if not any(0.35 <= es <= 0.95 for es in chosen_eu_full):
            msg = (
                f"{suite}: no usable difficulty point (no chosen budget has euclid_success "
                f"in [0.35, 0.95]; euclid@chosen={chosen_eu})"
            )
            print(f"[c8] WARNING: {msg}", flush=True)
            warnings.append(msg)

        valid_headroom = [h for h in chosen_headroom if h is not None]
        best_headroom = max(valid_headroom) if valid_headroom else None
        if best_headroom is not None and best_headroom < 0.15:
            msg = (
                f"{suite}: low expansion headroom (best chosen headroom={best_headroom:.3f} "
                f"< 0.15; oracle barely cheaper than euclid)"
            )
            print(f"[c8] WARNING: {msg}", flush=True)
            warnings.append(msg)

        budgets_map[suite] = chosen
        measurements[suite] = [
            {
                "budget": int(b),
                "euclid": round(euclid_success[bi], 4),
                "oracle": round(oracle_success[bi], 4),
                "euclid_exp": (round(euclid_exp[bi], 2) if euclid_exp[bi] is not None else None),
                "oracle_exp": (round(oracle_exp[bi], 2) if oracle_exp[bi] is not None else None),
                "worlds": int(n_worlds),
            }
            for bi, b in enumerate(CALIB_GRID)
        ]

        def _fmt(vals, nd=2):
            return [(round(v, nd) if v is not None else None) for v in vals]

        print(
            f"[c8] calibrate {suite}: chosen={chosen} euclid@chosen={chosen_eu} "
            f"euclid_exp@chosen={_fmt(chosen_eu_exp)} oracle_exp@chosen={_fmt(chosen_or_exp)} "
            f"headroom@chosen={_fmt(chosen_headroom, 3)}",
            flush=True,
        )

    calib = {
        "budgets": budgets_map,
        "grid": CALIB_GRID,
        "measurements": measurements,
        "warnings": warnings,
    }
    calib_path = out_dir / "calibration.json"
    write_json(calib_path, calib)
    print(
        f"[{now_str()}] C8 calibrate: wrote {calib_path} "
        f"({len(budgets_map)} suites, {len(warnings)} warnings)",
        flush=True,
    )
    return calib_path


# ---------------------------------------------------------------------------
# Task 13 — analyze mode (summary CSV + significance MD + six pre-registered
# comparisons MD + figures)
# ---------------------------------------------------------------------------
#
# Mirrors C7's analyze (continuous_prm_c7_integration_compare.run_analyze) but
# adapts to the C8 record schema (arrival/optimal_arrival instead of cost/optimal)
# and to the SIX C8-specific comparisons (spotlight: time-aware vs time-blind).
#
# DRY: the schema-agnostic stat helpers are IMPORTED from C7 (battle-tested) —
# wilcoxon_signed_rank_p, bootstrap_median_ci, _fmt_p, _fmt_num, _fmt_w. The
# C8-specific summary/significance/comparison BUILDERS are written here (the
# comparison logic is schema- and hypothesis-specific). mcnemar_exact_p /
# bh_q_values are already imported from C6 at the top of this module.

from continuous_prm_c7_integration_compare import (  # noqa: E402
    wilcoxon_signed_rank_p,
    bootstrap_median_ci,
    _fmt_p,
    _fmt_num,
    _fmt_w,
)

# An "arm" is (provider, mode, w). The euclid-time baseline arm is always astar.
EUCLID_BASELINE = ("euclid", "astar", None)
# Baseline/ceiling providers, NOT "learned arms": `euclid` is the reference
# baseline and `oracle` is the (exact space-time-Dijkstra) ceiling. Neither is a
# hypothesis under test, so both are EXCLUDED from the McNemar/BH/Wilcoxon
# significance family (oracle "beats" euclid trivially and would inflate the BH
# test count m, making learned-arm q-values overly conservative). Oracle still
# appears in the summary CSV and the gap-to-ceiling comparison.
NON_LEARNED = {"euclid", "oracle"}
# Below this n, a p-value is not trustworthy: print "n/a (n<6)" instead of a number
# (still report the counts / median / CI). Applies to McNemar discordant count and
# the expansion matched-set n.
MIN_N_FOR_P = 6
# In-distribution (trained) vs held-out (OOD) dynamic suites for comparison 6.
IN_DIST_SUITES = ("C_dyn_maze", "C_dyn_rooms", "C_dyn_spiral")
HELD_OUT_SUITES = ("C_dyn_maze_dense", "C_dyn_crossing", "C_dyn_rooms_large")


def _coerce_bool(v) -> bool:
    """Robustly coerce a CSV cell to bool (handles 'True'/'1'/'true'/True/1)."""
    return str(v).strip().lower() in ("1", "true", "yes")


def _coerce_float(v):
    """Coerce a CSV cell to float; '' / None / 'none' / 'nan' -> nan."""
    if v is None:
        return float("nan")
    s = str(v).strip()
    if s == "" or s.lower() in ("none", "nan"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _coerce_w(v):
    """Coerce a w cell to float, or None for astar rows (blank/None)."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_raw_rows(out_dir: Path):
    """Read continuous_prm_c8_eval_raw.csv and coerce CSV strings to typed fields.

    Returns a list of dicts with: suite, world_index(int), provider, mode,
    w(float|None), budget(int), found(bool), expansions(int), arrival(float),
    optimal_arrival(float), suboptimality(float), closed(int), nonfinite(bool).
    """
    raw_path = Path(out_dir) / "results" / "continuous_prm_c8_eval_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"analyze: missing raw eval CSV {raw_path}; run --mode eval first"
        )
    rows = read_csv(raw_path)
    out = []
    for r in rows:
        out.append(
            {
                "suite": str(r.get("suite", "")),
                "world_index": int(float(r.get("world_index", 0) or 0)),
                "provider": str(r.get("provider", "")),
                "mode": str(r.get("mode", "")),
                "w": _coerce_w(r.get("w")),
                "budget": int(float(r.get("budget", 0) or 0)),
                "found": _coerce_bool(r.get("found")),
                "expansions": int(float(r.get("expansions", 0) or 0)),
                "arrival": _coerce_float(r.get("arrival")),
                "optimal_arrival": _coerce_float(r.get("optimal_arrival")),
                "suboptimality": _coerce_float(r.get("suboptimality")),
                "closed": int(float(r.get("closed", 0) or 0)),
                "nonfinite": _coerce_bool(r.get("nonfinite")),
            }
        )
    return out


def _load_calib_budgets(out_dir: Path):
    """Return {suite: [budgets]} from out_dir/calibration.json, or {} if absent."""
    import json
    calib_path = Path(out_dir) / "calibration.json"
    if not calib_path.exists():
        return {}
    try:
        calib = json.loads(calib_path.read_text())
        return dict(calib.get("budgets", {}) or {})
    except (ValueError, OSError):
        return {}


def _load_calib_measurements(out_dir: Path):
    """Return {suite: [measurement dicts]} from out_dir/calibration.json "measurements",
    or {} if absent/unreadable."""
    import json
    calib_path = Path(out_dir) / "calibration.json"
    if not calib_path.exists():
        return {}
    try:
        calib = json.loads(calib_path.read_text())
        return dict(calib.get("measurements", {}) or {})
    except (ValueError, OSError):
        return {}


def _binding_budget(suite: str, suite_budgets, calib_budgets,
                    calib_measurements=None, euclid_floor: float = 0.05) -> int:
    """Binding budget for a suite: the lowest calibrated-band budget where euclid
    success >= euclid_floor (so a degenerate 0%-success edge is skipped); if no
    band budget qualifies, the highest band budget; else the first budget seen.

    When calib_measurements is None (not supplied), falls back to the old behavior
    of returning the minimum band budget unconditionally."""
    raw = calib_budgets.get(suite)
    if raw:
        try:
            band = sorted(int(b) for b in raw)
            if calib_measurements is not None and suite in calib_measurements:
                meas = calib_measurements[suite]
                eu = {int(m["budget"]): float(m.get("euclid") or 0.0) for m in meas}
                qualified = [b for b in band if eu.get(b, 0.0) >= euclid_floor]
                return int(min(qualified)) if qualified else int(max(band))
            # No measurements available: keep original behavior.
            return int(min(band))
        except (TypeError, ValueError):
            pass
    seen = sorted(suite_budgets.get(suite, []))
    return int(seen[0]) if seen else 0


# ---------------------------------------------------------------------------
# Indexing helpers over the typed raw rows
# ---------------------------------------------------------------------------

def _index_rows(rows):
    """Index typed raw rows: returns (arms_sorted, lookup, suite_budgets_sorted)
    where suite_budgets_sorted maps suite -> ascending budget list. (arms/lookup
    mirror C7 for parity; analyze uses suite_budgets + the per-arm row maps below.)
    """
    arms: Dict = {}
    lookup: Dict = {}
    suite_budgets: Dict = {}
    for r in rows:
        suite, budget = r["suite"], r["budget"]
        arm = (r["provider"], r["mode"], r["w"])
        arms.setdefault(suite, {}).setdefault(budget, set()).add(arm)
        lookup[(suite, budget, r["provider"], r["mode"], r["w"], r["world_index"])] = r
        suite_budgets.setdefault(suite, set()).add(budget)
    arms_sorted = {
        s: {b: sorted(v, key=lambda a: (a[0], a[1], (a[2] if a[2] is not None else -1.0)))
            for b, v in bd.items()}
        for s, bd in arms.items()
    }
    suite_budgets_sorted = {s: sorted(bs) for s, bs in suite_budgets.items()}
    return arms_sorted, lookup, suite_budgets_sorted


def _arm_rows_by_world(rows, suite, budget, provider, mode, w):
    """Map world_index -> row for one arm at (suite, budget)."""
    out = {}
    for r in rows:
        if (r["suite"] == suite and r["budget"] == budget and r["provider"] == provider
                and r["mode"] == mode and r["w"] == w):
            out[r["world_index"]] = r
    return out


def _present_arms(rows):
    """Set of providers present in the data."""
    return {r["provider"] for r in rows}


# ---------------------------------------------------------------------------
# 1. Summary CSV
# ---------------------------------------------------------------------------

def build_summary(rows, cfg: C8Config):
    """One row per (suite, provider, mode, w, budget): success_rate, n,
    expansions_mean/median (solved), suboptimality_mean/p95 (solved),
    exp_ratio_vs_euclid_median (matched-solved vs euclid/astar), matched_n."""
    import numpy as np

    groups: Dict = {}
    for r in rows:
        key = (r["suite"], r["provider"], r["mode"], r["w"], r["budget"])
        groups.setdefault(key, []).append(r)

    # euclid/astar expansions per (suite, budget, world) for the matched ratio.
    euclid_exp: Dict = {}
    for r in rows:
        if r["provider"] == "euclid" and r["mode"] == "astar" and r["found"]:
            euclid_exp[(r["suite"], r["budget"], r["world_index"])] = float(r["expansions"])

    out = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], (k[3] if k[3] is not None else -1.0), k[4])):
        suite, provider, mode, w, budget = key
        grp = groups[key]
        n = len({r["world_index"] for r in grp})
        solved = [r for r in grp if r["found"]]
        succ = (len(solved) / n) if n else float("nan")
        exps = [float(r["expansions"]) for r in solved]
        subs = [float(r["suboptimality"]) for r in solved if np.isfinite(r["suboptimality"])]
        ratios = []
        for r in solved:
            eu = euclid_exp.get((suite, budget, r["world_index"]))
            if eu is not None and eu > 0:
                ratios.append(float(r["expansions"]) / eu)
        out.append(
            {
                "suite": suite,
                "provider": provider,
                "mode": mode,
                "w": _fmt_w(w),
                "budget": int(budget),
                "n": int(n),
                "success_rate": round(succ, 4) if np.isfinite(succ) else "",
                "expansions_mean": round(float(np.mean(exps)), 3) if exps else "",
                "expansions_median": round(float(np.median(exps)), 3) if exps else "",
                "suboptimality_mean": round(float(np.mean(subs)), 5) if subs else "",
                "suboptimality_p95": round(float(np.percentile(subs, 95)), 5) if subs else "",
                "exp_ratio_vs_euclid_median": round(float(np.median(ratios)), 4) if ratios else "",
                "matched_n": int(len(ratios)),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 2. Significance MD (McNemar/BH success + Wilcoxon/bootstrap expansions)
# ---------------------------------------------------------------------------

def _success_significance(rows, suite_budgets):
    """McNemar paired (learned arm vs euclid/astar) on SUCCESS per (suite, budget),
    BH-corrected across the whole grid over LEARNED arms only (NON_LEARNED — euclid
    reference + oracle ceiling — excluded from the arm list AND the BH family)."""
    import numpy as np

    comparisons = []
    pvals = []
    for suite in sorted(suite_budgets):
        for budget in suite_budgets[suite]:
            eu_by_world = _arm_rows_by_world(rows, suite, budget, "euclid", "astar", None)
            if not eu_by_world:
                continue
            arm_keys = sorted(
                {(r["provider"], r["mode"], r["w"]) for r in rows
                 if r["suite"] == suite and r["budget"] == budget
                 and r["provider"] not in NON_LEARNED},
                key=lambda a: (a[0], a[1], (a[2] if a[2] is not None else -1.0)),
            )
            for (provider, mode, w) in arm_keys:
                arm_by_world = _arm_rows_by_world(rows, suite, budget, provider, mode, w)
                shared = sorted(set(eu_by_world) & set(arm_by_world))
                if not shared:
                    continue
                gain = sum(1 for wi in shared if arm_by_world[wi]["found"] and not eu_by_world[wi]["found"])
                loss = sum(1 for wi in shared if eu_by_world[wi]["found"] and not arm_by_world[wi]["found"])
                eu_succ = float(np.mean([1.0 if eu_by_world[wi]["found"] else 0.0 for wi in shared]))
                arm_succ = float(np.mean([1.0 if arm_by_world[wi]["found"] else 0.0 for wi in shared]))
                p = mcnemar_exact_p(gain, loss)
                pvals.append(p)
                comparisons.append(
                    {
                        "suite": suite, "budget": int(budget),
                        "provider": provider, "mode": mode, "w": w,
                        "n": len(shared),
                        "euclid_success": eu_succ, "arm_success": arm_succ,
                        "delta": arm_succ - eu_succ,
                        "gain": gain, "loss": loss,
                        "discordant": gain + loss, "mcnemar_p": p,
                    }
                )
    qvals = bh_q_values(pvals)
    for row, q in zip(comparisons, qvals):
        row["bh_q"] = q
    return comparisons


def _expansion_significance(rows, suite_budgets, cfg: C8Config):
    """For each LEARNED arm vs euclid/astar at (suite, budget): on the matched set
    (worlds euclid/astar solved AND the arm solved), report median exp_ratio, paired
    Wilcoxon signed-rank p IN RATIO-SPACE (ratio - 1.0), and bootstrap 95% CI on the
    median ratio. Exploratory + UNcorrected (disclosed in the MD)."""
    out = []
    for suite in sorted(suite_budgets):
        for budget in suite_budgets[suite]:
            eu_by_world = _arm_rows_by_world(rows, suite, budget, "euclid", "astar", None)
            if not eu_by_world:
                continue
            arm_keys = sorted(
                {(r["provider"], r["mode"], r["w"]) for r in rows
                 if r["suite"] == suite and r["budget"] == budget
                 and r["provider"] not in NON_LEARNED},
                key=lambda a: (a[0], a[1], (a[2] if a[2] is not None else -1.0)),
            )
            for (provider, mode, w) in arm_keys:
                arm_by_world = _arm_rows_by_world(rows, suite, budget, provider, mode, w)
                matched = sorted(
                    wi for wi in (set(eu_by_world) & set(arm_by_world))
                    if eu_by_world[wi]["found"] and arm_by_world[wi]["found"]
                    and float(eu_by_world[wi]["expansions"]) > 0
                )
                if not matched:
                    out.append(
                        {"suite": suite, "budget": int(budget), "provider": provider,
                         "mode": mode, "w": w, "n_matched": 0,
                         "median_ratio": None, "wilcoxon_p": float("nan"),
                         "ci_lo": None, "ci_hi": None}
                    )
                    continue
                ratios = [float(arm_by_world[wi]["expansions"]) / float(eu_by_world[wi]["expansions"])
                          for wi in matched]
                wp = wilcoxon_signed_rank_p([rt - 1.0 for rt in ratios])
                med, ci_lo, ci_hi = bootstrap_median_ci(ratios, seed=int(cfg.seed))
                out.append(
                    {"suite": suite, "budget": int(budget), "provider": provider,
                     "mode": mode, "w": w, "n_matched": len(matched),
                     "median_ratio": med, "wilcoxon_p": wp,
                     "ci_lo": ci_lo, "ci_hi": ci_hi}
                )
    return out


def write_significance_md(out_dir: Path, success_rows, expansion_rows):
    def arm_label(r):
        base = f"{r['provider']}/{r['mode']}"
        return base + (f"/w={_fmt_w(r['w'])}" if r["w"] is not None else "")

    lines = ["# C8 Dynamics Comparison — Significance", ""]
    lines += [
        "## Success: McNemar (learned arm vs euclid-time/astar), BH-corrected across the grid",
        "",
        "_Family: learned arms only (oracle ceiling and euclid-time reference excluded). "
        "BH correction is applied to THIS success/McNemar grid only._",
        "",
        "|Suite|Budget|Arm|n|Euclid succ|Arm succ|Delta|Gain|Loss|Discordant|McNemar p|BH q|",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not success_rows:
        lines.append("|<none>|-|-|-|-|-|-|-|-|-|-|-|")
    for r in sorted(success_rows, key=lambda x: (x["suite"], x["budget"], arm_label(x))):
        disc = int(r.get("discordant", r["gain"] + r["loss"]))
        lines.append(
            f"|{r['suite']}|{r['budget']}|{arm_label(r)}|{r['n']}|"
            f"{_fmt_num(r['euclid_success'])}|{_fmt_num(r['arm_success'])}|{_fmt_num(r['delta'])}|"
            f"{r['gain']}|{r['loss']}|{disc}|{_fmt_p(r['mcnemar_p'], n=disc)}|{_fmt_num(r['bh_q'])}|"
        )
    lines += [
        "",
        "## Expansions: matched-set median ratio (arm/euclid) + Wilcoxon p + bootstrap 95% CI",
        "",
        "_Exploratory and UNcorrected: the Wilcoxon p-values below are NOT BH-corrected. "
        "Treat the bootstrap 95% CI on the median ratio as the primary inference._",
        "",
        "|Suite|Budget|Arm|n matched|Median ratio|95% CI|Wilcoxon p|",
        "|---|---:|---|---:|---:|---|---:|",
    ]
    if not expansion_rows:
        lines.append("|<none>|-|-|-|-|-|-|")
    for r in sorted(expansion_rows, key=lambda x: (x["suite"], x["budget"], arm_label(x))):
        ci = (f"[{_fmt_num(r['ci_lo'])}, {_fmt_num(r['ci_hi'])}]"
              if r["ci_lo"] is not None else "n/a")
        lines.append(
            f"|{r['suite']}|{r['budget']}|{arm_label(r)}|{r['n_matched']}|"
            f"{_fmt_num(r['median_ratio'])}|{ci}|{_fmt_p(r['wilcoxon_p'], n=r['n_matched'])}|"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- McNemar pairs each LEARNED arm against `euclid/astar` on success over shared worlds;",
        "  gain = arm found & euclid not, loss = euclid found & arm not. `oracle` (space-time ceiling)",
        "  and `euclid` (time-aware reference) are NOT hypotheses under test and are excluded.",
        "- BH q-values correct ONLY across this success/McNemar grid. The expansion-Wilcoxon",
        "  p-values are UNcorrected; the bootstrap CIs are the primary expansion inference.",
        "- The expansion ratio uses the *matched set* (worlds euclid AND the arm both solved).",
        "  Median ratio < 1 means the arm expands fewer nodes than euclid-time. The Wilcoxon p tests",
        "  paired (ratio - 1) in ratio-space (matching the median ratio + CI estimand).",
        f"- A p-value is shown as `n/a (n<{MIN_N_FOR_P})` when the McNemar discordant count or the",
        f"  expansion matched-set n is below {MIN_N_FOR_P} (too few pairs for a trustworthy p).",
    ]
    md_path = Path(out_dir) / "results" / "continuous_prm_c8_significance.md"
    ensure_dir(md_path.parent)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# 3. Pre-registered comparisons MD (the SIX C8 comparisons)
# ---------------------------------------------------------------------------

def _matched_ratio_and_success(rows, suite, budget, provider, mode, w, cfg):
    """For one arm vs euclid/astar at (suite, budget): median exp_ratio (matched),
    Wilcoxon p (ratio-1), bootstrap CI, success delta, McNemar p, and n. Returns
    None if euclid is absent at this (suite, budget) or the arm has no rows."""
    import numpy as np
    eu_by_world = _arm_rows_by_world(rows, suite, budget, "euclid", "astar", None)
    arm_by_world = _arm_rows_by_world(rows, suite, budget, provider, mode, w)
    if not eu_by_world or not arm_by_world:
        return None
    shared = sorted(set(eu_by_world) & set(arm_by_world))
    if not shared:
        return None
    gain = sum(1 for wi in shared if arm_by_world[wi]["found"] and not eu_by_world[wi]["found"])
    loss = sum(1 for wi in shared if eu_by_world[wi]["found"] and not arm_by_world[wi]["found"])
    eu_succ = float(np.mean([1.0 if eu_by_world[wi]["found"] else 0.0 for wi in shared]))
    arm_succ = float(np.mean([1.0 if arm_by_world[wi]["found"] else 0.0 for wi in shared]))
    mp = mcnemar_exact_p(gain, loss)
    matched = sorted(
        wi for wi in shared
        if eu_by_world[wi]["found"] and arm_by_world[wi]["found"]
        and float(eu_by_world[wi]["expansions"]) > 0
    )
    if matched:
        ratios = [float(arm_by_world[wi]["expansions"]) / float(eu_by_world[wi]["expansions"]) for wi in matched]
        med, ci_lo, ci_hi = bootstrap_median_ci(ratios, seed=int(cfg.seed))
        wp = wilcoxon_signed_rank_p([rt - 1.0 for rt in ratios])
    else:
        med = ci_lo = ci_hi = None
        wp = float("nan")
    return {
        "n": len(shared), "n_matched": len(matched),
        "euclid_success": eu_succ, "arm_success": arm_succ, "success_delta": arm_succ - eu_succ,
        "gain": gain, "loss": loss, "discordant": gain + loss, "mcnemar_p": mp,
        "median_ratio": med, "ci_lo": ci_lo, "ci_hi": ci_hi, "wilcoxon_p": wp,
    }


def _aware_vs_blind(rows, suite, budget, aware, blind, cfg):
    """Spotlight stat: matched median ratio of aware/blind expansions + Wilcoxon p
    on (ratio-1) over worlds BOTH the aware and blind arms (astar) solved, plus the
    success delta (aware_succ - blind_succ) over shared worlds. Returns None if
    either arm is absent / has no shared worlds."""
    import numpy as np
    aw = _arm_rows_by_world(rows, suite, budget, aware, "astar", None)
    bl = _arm_rows_by_world(rows, suite, budget, blind, "astar", None)
    if not aw or not bl:
        return None
    shared = sorted(set(aw) & set(bl))
    if not shared:
        return None
    aw_succ = float(np.mean([1.0 if aw[wi]["found"] else 0.0 for wi in shared]))
    bl_succ = float(np.mean([1.0 if bl[wi]["found"] else 0.0 for wi in shared]))
    matched = sorted(
        wi for wi in shared
        if aw[wi]["found"] and bl[wi]["found"] and float(bl[wi]["expansions"]) > 0
    )
    if matched:
        ratios = [float(aw[wi]["expansions"]) / float(bl[wi]["expansions"]) for wi in matched]
        med, ci_lo, ci_hi = bootstrap_median_ci(ratios, seed=int(cfg.seed))
        wp = wilcoxon_signed_rank_p([rt - 1.0 for rt in ratios])
    else:
        med = ci_lo = ci_hi = None
        wp = float("nan")
    return {
        "n": len(shared), "n_matched": len(matched),
        "aware_success": aw_succ, "blind_success": bl_succ, "success_delta": aw_succ - bl_succ,
        "median_ratio": med, "ci_lo": ci_lo, "ci_hi": ci_hi, "wilcoxon_p": wp,
    }


def _gap_to_ceiling_fracs(eu_by_world, or_by_world, arm_by_world):
    """List of (arm_exp - oracle_exp)/(euclid_exp - oracle_exp) over worlds where
    euclid, oracle, and arm all solved and the euclid->oracle gap is positive
    (0 = matches oracle, 1 = no better than euclid)."""
    fracs = []
    shared = set(eu_by_world) & set(or_by_world) & set(arm_by_world)
    for wi in shared:
        e, o, a = eu_by_world[wi], or_by_world[wi], arm_by_world[wi]
        if not (e["found"] and o["found"] and a["found"]):
            continue
        eu_exp = float(e["expansions"]); or_exp = float(o["expansions"]); arm_exp = float(a["expansions"])
        denom = eu_exp - or_exp
        if denom <= 0:
            continue
        fracs.append((arm_exp - or_exp) / denom)
    return fracs


def _best_learned_arm(rows, binding, cfg, present):
    """Pick the 'best learned arm' (provider) for comparison 6: the TIME-AWARE
    learned provider (astar) with the lowest pooled-over-suites matched-median
    exp_ratio vs euclid at each suite's binding budget. Time-blind (`_blind`) and
    non-learned (euclid/oracle) arms are EXCLUDED — the generalization story should
    feature a time-aware arm (mirrors comparison 1's time_aware_learned filter).
    Returns provider name or None."""
    import numpy as np
    learned = sorted(
        p for p in present if p not in NON_LEARNED and not p.endswith("_blind")
    )
    best = None
    for prov in learned:
        ratios = []
        for s, b in binding.items():
            eu = _arm_rows_by_world(rows, s, b, "euclid", "astar", None)
            arm = _arm_rows_by_world(rows, s, b, prov, "astar", None)
            for wi in (set(eu) & set(arm)):
                if eu[wi]["found"] and arm[wi]["found"] and float(eu[wi]["expansions"]) > 0:
                    ratios.append(float(arm[wi]["expansions"]) / float(eu[wi]["expansions"]))
        if not ratios:
            continue
        med = float(np.median(ratios))
        if best is None or med < best[1]:
            best = (prov, med)
    return best[0] if best is not None else None


def _ratio_cell(st):
    if st is None or st.get("median_ratio") is None:
        return "n/a"
    ci = f"[{_fmt_num(st['ci_lo'])}, {_fmt_num(st['ci_hi'])}]"
    return f"{_fmt_num(st['median_ratio'])} {ci}"


def _aware_blind_pairs(present, cfg):
    """Build the list of (aware, blind) provider name pairs present in the data.
    For each scalar/field backbone, pair scalar_<bb> with scalar_<bb>_blind (and
    field likewise) when BOTH are present."""
    pairs = []
    for bb in parse_csv(cfg.scalar_backbones):
        aware, blind = f"scalar_{bb}", f"scalar_{bb}_blind"
        if aware in present and blind in present:
            pairs.append((aware, blind))
    for bb in parse_csv(cfg.field_backbones):
        aware, blind = f"field_{bb}", f"field_{bb}_blind"
        if aware in present and blind in present:
            pairs.append((aware, blind))
    return pairs


def write_preregistered_md(out_dir: Path, rows, cfg: C8Config, binding):
    """Write the SIX pre-registered C8 comparison sections. `binding` maps suite ->
    binding budget. Each section degrades gracefully (notes when an arm/suite is
    absent) instead of crashing."""
    import numpy as np
    present = _present_arms(rows)
    suites = sorted(binding)
    learned = sorted(p for p in present if p not in NON_LEARNED)
    # learned, time-aware (non-blind) arms for the time-aware-vs-euclid comparison.
    time_aware_learned = sorted(p for p in learned if not p.endswith("_blind"))

    lines = ["# C8 Dynamics Comparison — Pre-registered Comparisons", ""]
    lines.append(
        "Binding budget per suite (lowest calibrated-band budget where euclid success >= 0.05, "
        "so a degenerate 0%-success edge is skipped; if no band budget qualifies, the highest "
        "band budget; else the first budget seen): "
        + ", ".join(f"{s}={binding[s]}" for s in suites)
    )
    lines.append("")
    lines += [
        "_Multiplicity: BH correction is applied ONLY to the success/McNemar grid over learned "
        "arms (see `continuous_prm_c8_significance.md`). The p-values in THESE six pre-registered "
        "comparisons are UNcorrected; treat the bootstrap 95% CIs as the primary inference._",
        "",
        f"_Small-n: a p shown as `n/a (n<{MIN_N_FOR_P})` had too few discordant/matched pairs to "
        "trust (counts / median / CI are still reported)._",
        "",
    ]

    # --- Comparison 1: time-aware learned vs euclid-time ---------------------
    lines += ["## 1. Time-aware learned vs euclid-time (expansions + success), per suite", ""]
    lines += [
        "Each time-aware learned arm (field_<bb>/astar, scalar_<bb>/astar) vs `euclid/astar`.",
        "",
    ]
    if not time_aware_learned:
        lines += ["_no time-aware learned arms present — skipped._", ""]
    else:
        lines += [
            "|Suite|Budget|Arm|n|Euclid succ|Arm succ|Succ delta|McNemar p|n matched|Median ratio (95% CI)|Wilcoxon p|",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
        for s in suites:
            b = binding[s]
            for prov in time_aware_learned:
                st = _matched_ratio_and_success(rows, s, b, prov, "astar", None, cfg)
                if st is None:
                    lines.append(f"|{s}|{b}|{prov}/astar|_arm/euclid absent_|||||||")
                    continue
                lines.append(
                    f"|{s}|{b}|{prov}/astar|{st['n']}|{_fmt_num(st['euclid_success'])}|"
                    f"{_fmt_num(st['arm_success'])}|{_fmt_num(st['success_delta'])}|"
                    f"{_fmt_p(st['mcnemar_p'], n=st['discordant'])}|{st['n_matched']}|"
                    f"{_ratio_cell(st)}|{_fmt_p(st['wilcoxon_p'], n=st['n_matched'])}|"
                )
        lines.append("")

    # --- Comparison 2: time-aware vs time-blind (THE SPOTLIGHT) --------------
    lines += ["## 2. Time-aware vs time-blind — THE SPOTLIGHT: does the future window help?", ""]
    lines += [
        "Matched expansion ratio of the time-AWARE arm vs its time-BLIND (W=0) twin "
        "(e.g. `scalar_hrm` vs `scalar_hrm_blind`, `field_unet` vs `field_unet_blind`), both astar.",
        "Median ratio < 1 means the aware model expands fewer nodes than its blind twin "
        "(the future window helps). Success delta = aware - blind over shared worlds.",
        "",
    ]
    pairs = _aware_blind_pairs(present, cfg)
    if not pairs:
        lines += ["_no aware/blind pairs present (need both `<arm>` and `<arm>_blind`) — skipped._", ""]
    else:
        lines += [
            "|Suite|Budget|Aware|Blind|n|Aware succ|Blind succ|Succ delta|n matched|Median ratio aware/blind (95% CI)|Wilcoxon p|",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|",
        ]
        for s in suites:
            b = binding[s]
            for aware, blind in pairs:
                st = _aware_vs_blind(rows, s, b, aware, blind, cfg)
                if st is None:
                    lines.append(f"|{s}|{b}|{aware}|{blind}|_absent/no shared worlds_||||||")
                    continue
                lines.append(
                    f"|{s}|{b}|{aware}|{blind}|{st['n']}|{_fmt_num(st['aware_success'])}|"
                    f"{_fmt_num(st['blind_success'])}|{_fmt_num(st['success_delta'])}|"
                    f"{st['n_matched']}|{_ratio_cell(st)}|{_fmt_p(st['wilcoxon_p'], n=st['n_matched'])}|"
                )
        lines.append("")

    # --- Comparison 3: additive (astar) vs focal for learned arms -----------
    lines += ["## 3. Additive (astar) vs focal — does C7's additive-wins hold under dynamics?", ""]
    lines += [
        "For each learned arm: its astar (additive) ratio vs euclid, and its best-w focal ratio "
        "vs euclid, side by side. Best w = lowest matched-median exp_ratio (post-hoc; the focal p "
        "is optimistic — treat the CI as primary).",
        "",
    ]
    if not learned:
        lines += ["_no learned arms present — skipped._", ""]
    else:
        lines += [
            "|Suite|Budget|Arm|astar ratio (CI)|best w|focal ratio (CI)|focal Wilcoxon p|",
            "|---|---:|---|---|---:|---|---:|",
        ]
        for s in suites:
            b = binding[s]
            for prov in learned:
                st_a = _matched_ratio_and_success(rows, s, b, prov, "astar", None, cfg)
                ws = sorted({r["w"] for r in rows
                             if r["suite"] == s and r["budget"] == b and r["provider"] == prov
                             and r["mode"] == "focal" and r["w"] is not None})
                bw, st_f = None, None
                for w in ws:
                    cand = _matched_ratio_and_success(rows, s, b, prov, "focal", w, cfg)
                    if cand is None or cand["median_ratio"] is None:
                        continue
                    if st_f is None or cand["median_ratio"] < st_f["median_ratio"]:
                        bw, st_f = w, cand
                focal_p = _fmt_p(st_f["wilcoxon_p"], n=st_f["n_matched"]) if st_f else "n/a"
                lines.append(
                    f"|{s}|{b}|{prov}|{_ratio_cell(st_a)}|{_fmt_w(bw) if bw is not None else 'n/a'}|"
                    f"{_ratio_cell(st_f)}|{focal_p}|"
                )
        lines.append("")

    # --- Comparison 4: recurrent/hierarchical vs field U-Net ----------------
    lines += ["## 4. Recurrent/hierarchical vs field U-Net — do temporal models win when timing matters?", ""]
    lines += [
        "exp_ratio vs euclid (astar) side by side: the recurrent/hierarchical arms "
        "(scalar_hrm, scalar_onlstm, field_hrm, field_onlstm) vs the convolutional `field_unet`.",
        "",
    ]
    rec_provs = [p for p in ("scalar_hrm", "scalar_onlstm", "field_hrm", "field_onlstm", "field_unet")
                 if p in present]
    if not rec_provs:
        lines += ["_none of the recurrent/hierarchical/U-Net arms present — skipped._", ""]
    else:
        lines += [
            "|Suite|Budget|Arm|n matched|Median ratio vs euclid (95% CI)|Wilcoxon p|",
            "|---|---:|---|---:|---|---:|",
        ]
        for s in suites:
            b = binding[s]
            for prov in rec_provs:
                st = _matched_ratio_and_success(rows, s, b, prov, "astar", None, cfg)
                if st is None:
                    lines.append(f"|{s}|{b}|{prov}|_arm/euclid absent_|-|-|")
                    continue
                lines.append(
                    f"|{s}|{b}|{prov}|{st['n_matched']}|{_ratio_cell(st)}|"
                    f"{_fmt_p(st['wilcoxon_p'], n=st['n_matched'])}|"
                )
        lines.append("")

    # --- Comparison 5: learned vs oracle gap-to-ceiling ---------------------
    lines += ["## 5. Learned vs oracle — gap-to-ceiling", ""]
    if "oracle" not in present:
        lines += ["_oracle absent — skipped._", ""]
    else:
        lines += [
            "Median over the triple-matched set (euclid, oracle, arm all solved) of",
            "`(arm_exp - oracle_exp) / (euclid_exp - oracle_exp)` — the fraction of the",
            "euclid->oracle expansion gap left *uncaptured* (0 = matches oracle, 1 = no better than euclid).",
            "",
            "|Suite|Budget|Arm|n triple-matched|Median uncaptured-gap fraction|",
            "|---|---:|---|---:|---:|",
        ]
        any_row = False
        for s in suites:
            b = binding[s]
            eu_by_world = _arm_rows_by_world(rows, s, b, "euclid", "astar", None)
            or_by_world = _arm_rows_by_world(rows, s, b, "oracle", "astar", None)
            for prov in learned:
                arm_by_world = _arm_rows_by_world(rows, s, b, prov, "astar", None)
                fracs = _gap_to_ceiling_fracs(eu_by_world, or_by_world, arm_by_world)
                if not fracs:
                    lines.append(f"|{s}|{b}|{prov}/astar|0|n/a|")
                    continue
                lines.append(f"|{s}|{b}|{prov}/astar|{len(fracs)}|{_fmt_num(float(np.median(fracs)))}|")
                any_row = True
        if not any_row:
            lines.append("|<none>|-|-|-|-|")
        lines.append("")

    # --- Comparison 6: in-dist vs held-out for the best learned arm ---------
    lines += ["## 6. In-distribution vs held-out — best learned arm exp_ratio + success vs euclid", ""]
    best_arm = _best_learned_arm(rows, binding, cfg, present)
    if best_arm is None:
        lines += ["_no learned arm with a computable matched ratio — skipped._", ""]
    else:
        lines += [
            f"Best learned arm (lowest pooled matched-median exp_ratio vs euclid): **{best_arm}**/astar.  "
            f"In-distribution (trained): {', '.join(IN_DIST_SUITES)}.  "
            f"Held-out (OOD): {', '.join(HELD_OUT_SUITES)}.",
            "",
            "|Group|Suite|Budget|n matched|Median ratio (95% CI)|Succ delta vs euclid|",
            "|---|---|---:|---:|---|---:|",
        ]
        for group_name, group in (("in-dist", IN_DIST_SUITES), ("held-out", HELD_OUT_SUITES)):
            present_in_group = [s for s in group if s in binding]
            if not present_in_group:
                lines.append(f"|{group_name}|_no suites in this run_|-|-|-|-|")
                continue
            for s in present_in_group:
                b = binding[s]
                st = _matched_ratio_and_success(rows, s, b, best_arm, "astar", None, cfg)
                if st is None:
                    lines.append(f"|{group_name}|{s}|{b}|_arm/euclid absent_|-|-|")
                    continue
                lines.append(
                    f"|{group_name}|{s}|{b}|{st['n_matched']}|{_ratio_cell(st)}|"
                    f"{_fmt_num(st['success_delta'])}|"
                )
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Each comparison uses the per-suite binding budget. Arms or suites absent from this run",
        "  are skipped with a note rather than crashing.",
        "- Comparison 2 (the spotlight) pairs each time-aware arm with its W=0 time-blind twin;",
        "  median ratio < 1 means the future window reduces expansions.",
        "- Comparison 3 picks the focal `w` with the lowest matched-median exp_ratio vs euclid;",
        "  reporting that winner's p on the same data is optimistic — the CI is the primary inference.",
        "- These six p-values are UNcorrected (BH applies only to the success/McNemar grid).",
        "- The Wilcoxon p tests paired (ratio - 1) in ratio-space, matching the median ratio + CI.",
        "- Bootstrap CIs are seeded (`np.random.default_rng(cfg.seed)`), so this analysis is reproducible.",
    ]
    md_path = Path(out_dir) / "results" / "continuous_prm_c8_preregistered.md"
    ensure_dir(md_path.parent)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# 4. Figures (guarded, like C7's maybe_make_figures)
# ---------------------------------------------------------------------------

def maybe_make_figures(out_dir: Path, rows, cfg: C8Config, binding):
    """Write three figures if matplotlib is importable; skip silently otherwise.
    (a) exp-ratio-vs-euclid bars per suite (learned astar arms);
    (b) time-aware-vs-time-blind bars per suite (aware/blind median exp ratio);
    (c) suboptimality-vs-w for focal arms (pooled over suites at binding budgets)."""
    if not cfg.make_figures:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[c8] analyze: matplotlib unavailable; skipping figures", flush=True)
        return []
    import numpy as np

    fig_dir = ensure_dir(Path(out_dir) / "figures")
    written = []
    suites = sorted(binding)
    present = _present_arms(rows)
    learned = sorted(p for p in present if p not in NON_LEARNED)

    # (a) expansion-ratio bars per suite at the binding budget (learned astar arms).
    try:
        for s in suites:
            b = binding[s]
            labels, vals = [], []
            for prov in learned:
                st = _matched_ratio_and_success(rows, s, b, prov, "astar", None, cfg)
                if st is not None and st["median_ratio"] is not None:
                    labels.append(prov)
                    vals.append(st["median_ratio"])
            if not labels:
                continue
            plt.figure(figsize=(7, 3.5))
            plt.bar(range(len(labels)), vals, color="steelblue")
            plt.axhline(1.0, color="k", linestyle="--", linewidth=1)
            plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
            plt.ylabel("median exp ratio vs euclid")
            plt.title(f"{s} @ budget {b}")
            plt.tight_layout()
            p = fig_dir / f"exp_ratio__{sanitize_name(s)}.png"
            plt.savefig(p, dpi=130); plt.close()
            written.append(p)
    except Exception as exc:  # pragma: no cover - figure robustness
        print(f"[c8] analyze: figure (a) failed: {exc}", flush=True)

    # (b) time-aware vs time-blind median exp ratio (aware/blind) per suite.
    try:
        pairs = _aware_blind_pairs(present, cfg)
        if pairs:
            for s in suites:
                b = binding[s]
                labels, vals = [], []
                for aware, blind in pairs:
                    st = _aware_vs_blind(rows, s, b, aware, blind, cfg)
                    if st is not None and st["median_ratio"] is not None:
                        labels.append(aware)
                        vals.append(st["median_ratio"])
                if not labels:
                    continue
                plt.figure(figsize=(7, 3.5))
                colors = ["seagreen" if v < 1.0 else "indianred" for v in vals]
                plt.bar(range(len(labels)), vals, color=colors)
                plt.axhline(1.0, color="k", linestyle="--", linewidth=1)
                plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
                plt.ylabel("median exp ratio aware/blind")
                plt.title(f"time-aware vs time-blind: {s} @ budget {b}")
                plt.tight_layout()
                p = fig_dir / f"aware_vs_blind__{sanitize_name(s)}.png"
                plt.savefig(p, dpi=130); plt.close()
                written.append(p)
    except Exception as exc:  # pragma: no cover
        print(f"[c8] analyze: figure (b) failed: {exc}", flush=True)

    # (c) suboptimality-vs-w for focal arms (pooled over suites at binding budgets).
    try:
        focal_provs = sorted({r["provider"] for r in rows if r["mode"] == "focal"})
        plotted = False
        plt.figure(figsize=(7, 3.5))
        for prov in focal_provs:
            ws = sorted({r["w"] for r in rows if r["provider"] == prov and r["mode"] == "focal" and r["w"] is not None})
            xs, ys = [], []
            for w in ws:
                subs = []
                for s in suites:
                    b = binding[s]
                    for r in rows:
                        if (r["suite"] == s and r["budget"] == b and r["provider"] == prov
                                and r["mode"] == "focal" and r["w"] == w and r["found"]
                                and np.isfinite(r["suboptimality"])):
                            subs.append(float(r["suboptimality"]))
                if subs:
                    xs.append(w); ys.append(float(np.mean(subs)))
            if xs:
                plt.plot(xs, ys, marker="o", label=prov)
                plotted = True
        if plotted:
            plt.xlabel("focal w"); plt.ylabel("mean suboptimality (solved)")
            plt.title("suboptimality vs focal w (binding budgets)")
            plt.legend(fontsize=8)
            plt.tight_layout()
            p = fig_dir / "suboptimality_vs_w.png"
            plt.savefig(p, dpi=130); plt.close()
            written.append(p)
        else:
            plt.close()
    except Exception as exc:  # pragma: no cover
        print(f"[c8] analyze: figure (c) failed: {exc}", flush=True)

    if written:
        print(f"[c8] analyze: wrote {len(written)} figures -> {fig_dir}", flush=True)
    return written


# ---------------------------------------------------------------------------
# Analyze orchestrator
# ---------------------------------------------------------------------------

def run_analyze(cfg: C8Config, out_dir: Path) -> Dict[str, object]:
    """Read the raw eval CSV and write summary CSV, significance MD, pre-registered
    comparisons MD, and figures. Returns the paths written."""
    out_dir = Path(out_dir)
    rows = load_raw_rows(out_dir)
    print(f"[{now_str()}] C8 analyze: loaded {len(rows)} raw rows", flush=True)

    _, _, suite_budgets = _index_rows(rows)
    calib_budgets = _load_calib_budgets(out_dir)
    calib_meas = _load_calib_measurements(out_dir)
    binding = {s: _binding_budget(s, suite_budgets, calib_budgets, calib_meas) for s in suite_budgets}

    # 1. Summary CSV
    summary = build_summary(rows, cfg)
    summary_path = Path(out_dir) / "results" / "continuous_prm_c8_eval_summary.csv"
    write_csv(summary_path, summary)
    print(f"[{now_str()}] C8 analyze: wrote summary ({len(summary)} arm-rows) -> {summary_path}", flush=True)

    # 2. Significance MD
    success_rows = _success_significance(rows, suite_budgets)
    expansion_rows = _expansion_significance(rows, suite_budgets, cfg)
    sig_path = write_significance_md(out_dir, success_rows, expansion_rows)
    print(f"[{now_str()}] C8 analyze: wrote significance -> {sig_path}", flush=True)

    # 3. Pre-registered comparisons MD
    prereg_path = write_preregistered_md(out_dir, rows, cfg, binding)
    print(f"[{now_str()}] C8 analyze: wrote pre-registered comparisons -> {prereg_path}", flush=True)

    # 4. Figures (guarded)
    figs = maybe_make_figures(out_dir, rows, cfg, binding)

    return {
        "summary": summary_path,
        "significance": sig_path,
        "preregistered": prereg_path,
        "figures": figs,
    }


# ---------------------------------------------------------------------------
# Full pipeline orchestrator (Tasks 10-13)
# ---------------------------------------------------------------------------

def run_full(cfg: C8Config, out_dir: Path, device) -> None:
    """Run all C8 stages end-to-end in order.

    Stage order:
      1. train     — collect labels once + train scalar & field (time-aware AND
                     time-blind) models via run_train.
      2. calibrate — ONLY if out_dir/calibration.json does NOT already exist;
                     prints SKIPPED and reuses existing calibration otherwise.
      3. eval      — build providers via _load_eval_providers + run_eval.
      4. analyze   — run_analyze (summary CSV + significance MD + pre-registered
                     comparisons MD + figures).

    Prints a [c8] === STAGE: <name> === banner before each stage.
    """
    out_dir = Path(out_dir)

    # ---- Stage 1: train (includes collect) ----------------------------------
    print(f"\n[c8] === STAGE: train ===", flush=True)
    run_train(cfg, out_dir)

    # ---- Stage 2: calibrate (skip if calibration.json already exists) -------
    calib_path = out_dir / "calibration.json"
    if calib_path.exists():
        print(
            f"\n[c8] === STAGE: calibrate (SKIPPED — {calib_path} already exists) ===",
            flush=True,
        )
    else:
        print(f"\n[c8] === STAGE: calibrate ===", flush=True)
        run_calibrate(cfg, out_dir, device=None)

    # ---- Stage 3: eval -------------------------------------------------------
    print(f"\n[c8] === STAGE: eval ===", flush=True)
    providers = _load_eval_providers(cfg, out_dir, device)
    run_eval(cfg, out_dir, device, providers)

    # ---- Stage 4: analyze ----------------------------------------------------
    print(f"\n[c8] === STAGE: analyze ===", flush=True)
    run_analyze(cfg, out_dir)


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
    p.add_argument(
        "--scalar-max-samples",
        type=int,
        default=250000,
        help=(
            "Cap on REACHABLE scalar training samples (seeded subsample). "
            "0 = no cap (use all reachable samples)."
        ),
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
        scalar_max_samples=int(args.scalar_max_samples),
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
        device = _pick_device(cfg)
        print(f"[{now_str()}] C8 eval: device={device}", flush=True)
        providers = _load_eval_providers(cfg, out_dir, device)
        run_eval(cfg, out_dir, device, providers)
    elif cfg.mode == "calibrate":
        run_calibrate(cfg, out_dir, device=None)
    elif cfg.mode == "analyze":
        # Reads out_dir/results/continuous_prm_c8_eval_raw.csv (no models / device).
        print(f"[{now_str()}] C8 analyze: stats + pre-registered comparisons (CPU)", flush=True)
        run_analyze(cfg, out_dir)
    elif cfg.mode == "full":
        device = _pick_device(cfg)
        print(f"[{now_str()}] C8 full: device={device}", flush=True)
        run_full(cfg, out_dir, device)
    else:
        raise ValueError(f"unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
