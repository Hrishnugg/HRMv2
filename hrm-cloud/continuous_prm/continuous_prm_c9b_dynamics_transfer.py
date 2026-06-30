#!/usr/bin/env python3
"""C9b: few-shot transfer under dynamics. Adapt frozen C8 pooled space-time
heuristics (aware + blind) to held-out dynamic families via zero_shot/LoRA/full_ft/scratch.
New-file-only; reuses C8 + C9/C9h + common. See docs/superpowers/specs/2026-06-30-c9b-dynamics-transfer-design.md.
"""
from __future__ import annotations
import argparse, dataclasses, csv as _csv, json, hashlib, functools
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import continuous_prm_common as C
import continuous_prm_c8_dynamics_compare as M8
import continuous_prm_c8_dynamic_maps as M8MAPS
import continuous_prm_dynamic_providers as DP
import continuous_prm_spacetime as ST
import continuous_prm_dynamics as D
import continuous_prm_c9_transfer as C9
import continuous_prm_c9h_transfer as C9H

BACKBONES = ["scalar_hrm", "scalar_onlstm", "field_unet"]
AWARENESS = ["aware", "blind"]


def install():
    M8MAPS.install_c8_dynamic_maps()


def _parse_csv(s): return C9._parse_csv(s)
def _parse_ints(s): return C9._parse_ints(s)
def now_str() -> str: return C.now_str()


@dataclass
class C9bConfig:
    source_dir: str = "hrm-cloud/continuous_prm/runs/c8_local_heavy"
    out_dir: str = "hrm-cloud/continuous_prm/runs/c9b_local"
    backbones: str = "scalar_hrm,scalar_onlstm,field_unet"
    targets: str = "C_dyn_maze_dense,C_dyn_crossing,C_dyn_rooms_large"
    awareness: str = "aware,blind"
    methods: str = "lora,full_ft,scratch"
    k_grid: str = "1,4,16"
    n_adapt_seeds: int = 3
    n_test: int = 20
    rank: int = 8
    alpha: float = 1.0
    epochs: int = 10
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-4
    grid_size: int = 64
    budgets: str = ""
    w_values: str = "1.0,1.1"
    seed: int = 1234
    scale: str = "local"
    mode: str = "full"
    cpu: bool = False
    retrain_sources: bool = False

    def awareness_list(self): return _parse_csv(self.awareness)


def _src_ckpt(cfg: C9bConfig, backbone: str, awareness: str) -> Path:
    sub, bb = backbone.split("_", 1)          # "scalar","hrm" | "field","unet"
    suffix = "_blind" if awareness == "blind" else ""
    return Path(cfg.source_dir) / "checkpoints" / f"c8_{sub}__{bb}{suffix}.pt"


def resolve_sources(cfg: C9bConfig) -> Dict[Tuple[str, str], Path]:
    out = {}
    for b in _parse_csv(cfg.backbones):
        for a in cfg.awareness_list():
            out[(b, a)] = _src_ckpt(cfg, b, a)
    return out


def _is_field(backbone: str) -> bool: return backbone.startswith("field")


# -----------------------------------------------------------------------------
# ADAPT/TEST world split (C8 dynamic generator)
# -----------------------------------------------------------------------------
# A world counts as "valid" iff C8 can actually produce a usable supervised
# labelset for it. To keep selection consistent with Task 3's label collection
# (which calls the SAME function), `_valid_world_seed` delegates to C8's
# `_collect_world_labels` (continuous_prm_c8_dynamics_compare.py:174-227) rather
# than re-deriving acceptance. That function:
#   - builds the dynamic world (None -> reject),
#   - builds the PRM with cfg.roadmap_nodes/cfg.roadmap_k at the SAME `seed`
#     (no offset) and rejects when rm is None or start (node 0) is not
#     connected to goal,
#   - rejects when the world is space-time-unsolvable from (start, t=0)
#     (`hstar[0, 0]` not finite once moving patrollers are accounted for).
# Using the identical seed -> roadmap mapping (previously this used `seed + 17`)
# guarantees a seed we mark "valid" yields a non-None labelset when Task 3
# collects on the same seed.


def _c8cfg(cfg: C9bConfig) -> "M8.C8Config":
    """A C8Config for label collection that matches C9b's knobs (grid_size, etc.).
    Only override fields C9b cares about; leave C8 defaults (roadmap 192/k7,
    k_patrollers, window) intact. Reused by Task 3 for the actual collection."""
    return M8.C8Config(grid_size=int(cfg.grid_size))


def _build_world_only(suite: str, seed: int):
    res = M8MAPS.build_dynamic_world(suite, int(seed))
    return None if res is None else res[0]


@functools.lru_cache(maxsize=64)
def _collect_world_labels_memo(suite: str, seed: int, grid_size: int) -> Optional[dict]:
    """Bounded memo around M8._collect_world_labels keyed by (suite, seed, grid_size).

    `_collect_world_labels` runs a ~9s backward space-time Dijkstra per world;
    both `_valid_world_seed` (selection) and `collect_temporal_dataset` (Task 3)
    call it on the SAME seeds, so caching avoids doing that work twice per seed.
    grid_size is part of the key for forward-compat even though
    `_collect_world_labels` itself does not currently read it.
    """
    cfg = C9bConfig(grid_size=int(grid_size))
    try:
        return M8._collect_world_labels(suite, int(seed), _c8cfg(cfg))
    except Exception:
        return None


def _valid_world_seed(suite: str, seed: int, cfg: C9bConfig) -> bool:
    lab = _collect_world_labels_memo(suite, int(seed), int(cfg.grid_size))
    return lab is not None


def test_world_seeds(target: str, cfg: C9bConfig) -> List[int]:
    rng = np.random.default_rng(10_000_000 + (int(hashlib.md5(target.encode()).hexdigest()[:6], 16) % 1_000_000))
    out, tries = [], 0
    while len(out) < cfg.n_test and tries < cfg.n_test * 200:
        s = int(rng.integers(0, 2**31 - 1)); tries += 1
        if _valid_world_seed(target, s, cfg):
            out.append(s)
    return out


def adapt_world_seeds(target: str, K: int, seed_idx: int, cfg: C9bConfig) -> List[int]:
    base = C9.adapt_seed(target, K, seed_idx, cfg.seed)
    rng = np.random.default_rng(base)
    test_set = set(test_world_seeds(target, cfg))
    out, tries = [], 0
    while len(out) < K and tries < K * 400:
        s = int(rng.integers(0, 2**31 - 1)); tries += 1
        if s in test_set:
            continue
        if _valid_world_seed(target, s, cfg):
            out.append(s)
    return out


# -----------------------------------------------------------------------------
# Task 3 — temporal dataset collection (aware/blind, scalar/field)
# -----------------------------------------------------------------------------
#
# Build a K-shot dataset of TEMPORAL FEATURES + space-time-oracle RESIDUAL
# TARGETS for a given backbone (scalar_* or field_*) and window_w (W>0 = aware,
# W=0 = blind). The schema MUST mirror what C8's own trainers
# (`M8._train_scalar` / `M8._train_field`) consume, since the frozen C8 source
# checkpoints we adapt here were trained on exactly that representation.
#
# Scalar schema (mirrors M8._build_scalar_dataset, used by M8._train_scalar):
#   Per world: Xw = DP.build_scalar_temporal_features(world, rm, dyn, t_max, dt,
#     W, k_patrollers, goal_idx=1)              # (N, t_max+1, W+1, token_dim)
#   flattened to (N*(t_max+1), W+1, token_dim); yw = node_residual.reshape(-1);
#   mw = reachable.reshape(-1); concatenated across worlds; THEN masked down to
#   reachable rows only (M8._train_scalar does `Xk = X[mask_np]` before training)
#   -> x: (M, W+1, token_dim) float32, y: (M,) float32. We mask at collection
#   time (we are building the trainer's input directly, not the pre-mask pool).
#
# Field schema (mirrors M8._FieldTemporalDataset / M8._train_field's loader):
#   Per (world, t): occ = DP.build_field_occupancy_stack(world, dyn, grid_size,
#     t, W, dt, static_base=...)                # (8+W, G, G)
#   cells = per-PRM-node integer grid-cells (N, 2) [ix, iy] (constant over t,
#   via M8._build_field_node_cells using the SAME gather convention C6 uses);
#   target = node_residual[:, t] (N,) float32; mask = reachable[:, t] (N,) bool.
#   We store the FLAT (un-padded, un-batched) per-(world,t) samples; Task 5's
#   trainer is expected to do its own batching/padding (mirroring
#   M8._field_collate) since variable-N concatenation here would lose the
#   per-sample N needed to reconstruct (cells, target, mask) triples. Keys:
#     occ    (S, 8+W, G, G) float32 — S = sum over worlds of (t_max+1)
#     cells  (S, N, 2)      int64   — ragged N zero-padded to max N across ALL
#                                     samples (cells_valid marks real rows)
#     target (S, N)         float32
#     mask   (S, N)         bool    — reachable AND not-a-padding-row
#   (padding follows M8._field_collate's convention: mask=False on padded rows)


def collect_temporal_dataset(
    target: str,
    seeds: List[int],
    backbone: str,
    window_w: int,
    k_patrollers: int,
    grid_size: int,
    out_npz: "str | Path",
) -> Path:
    """Build and save a K-shot temporal dataset for `target` over `seeds`.

    `backbone` selects scalar vs field schema via `_is_field`. `window_w` is W
    (0 = time-blind, >0 = time-aware with that lookback). `k_patrollers` MUST
    match the frozen C8 source checkpoint's k_patrollers so token_dim lines up
    (scalar only; field has no k_patrollers dependence). Returns the Path the
    dataset was saved to (== Path(out_npz)).
    """
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    W = int(window_w)
    G = int(grid_size)
    is_field = _is_field(backbone)

    labelsets: List[dict] = []
    for s in seeds:
        lab = _collect_world_labels_memo(target, int(s), G)
        if lab is None:
            # Selected seeds are expected to always yield a labelset (Task 2's
            # selection already filtered on this); guard defensively anyway.
            continue
        labelsets.append(lab)

    if not is_field:
        # --- Scalar branch: mirrors M8._build_scalar_dataset exactly ---------
        token_dim = 4 + 4 * int(k_patrollers)
        Xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for lab in labelsets:
            world, rm, dyn = lab["world"], lab["rm"], lab["dyn"]
            params = lab["params"]
            dt = float(params["dt"])
            t_max = int(params["t_max"])
            Xw = DP.build_scalar_temporal_features(
                world, rm, dyn, t_max, dt, W, int(k_patrollers), goal_idx=1
            )  # (N, t_max+1, W+1, token_dim)
            N = Xw.shape[0]
            Xw = Xw.reshape(N * (t_max + 1), W + 1, token_dim).astype(np.float32)
            yw = lab["node_residual"].reshape(-1).astype(np.float32)
            mw = lab["reachable"].reshape(-1).astype(np.bool_)
            # Mask down to reachable rows now (mirrors _train_scalar's Xk = X[mask]).
            Xs.append(Xw[mw])
            ys.append(yw[mw])
        if Xs:
            x = np.concatenate(Xs, axis=0)
            y = np.concatenate(ys, axis=0)
        else:
            x = np.zeros((0, W + 1, token_dim), dtype=np.float32)
            y = np.zeros((0,), dtype=np.float32)
        np.savez(
            out_npz,
            x=x,
            y=y,
            seeds=np.asarray(seeds, dtype=np.int64),
            window_w=np.asarray(W, dtype=np.int64),
            k_patrollers=np.asarray(int(k_patrollers), dtype=np.int64),
            token_dim=np.asarray(token_dim, dtype=np.int64),
        )
        return out_npz

    # --- Field branch: mirrors M8._FieldTemporalDataset / _train_field's loader.
    occs: List[np.ndarray] = []
    cells_l: List[np.ndarray] = []
    targets_l: List[np.ndarray] = []
    masks_l: List[np.ndarray] = []
    for lab in labelsets:
        world, dyn, rm = lab["world"], lab["dyn"], lab["rm"]
        dt = float(lab["params"]["dt"])
        t_max = int(lab["params"]["t_max"])
        static_base = DP.compute_field_static_base(world, G)
        node_cells = M8._build_field_node_cells(rm, float(world.side_len), G)  # (N, 2)
        for t in range(t_max + 1):
            occ = DP.build_field_occupancy_stack(world, dyn, G, t, W, dt, static_base=static_base)
            occs.append(np.ascontiguousarray(occ, dtype=np.float32))           # (8+W, G, G)
            cells_l.append(node_cells)
            targets_l.append(lab["node_residual"][:, t].astype(np.float32))    # (N,)
            masks_l.append(lab["reachable"][:, t].astype(np.bool_))            # (N,)

    S = len(occs)
    in_channels = 8 + W
    if S == 0:
        occ_arr = np.zeros((0, in_channels, G, G), dtype=np.float32)
        cells_arr = np.zeros((0, 0, 2), dtype=np.int64)
        target_arr = np.zeros((0, 0), dtype=np.float32)
        mask_arr = np.zeros((0, 0), dtype=np.bool_)
    else:
        Nmax = max(int(c.shape[0]) for c in cells_l)
        occ_arr = np.stack(occs, axis=0)                                       # (S, 8+W, G, G)
        cells_arr = np.zeros((S, Nmax, 2), dtype=np.int64)
        target_arr = np.zeros((S, Nmax), dtype=np.float32)
        mask_arr = np.zeros((S, Nmax), dtype=np.bool_)
        for i in range(S):
            n = int(cells_l[i].shape[0])
            cells_arr[i, :n] = cells_l[i]
            target_arr[i, :n] = targets_l[i]
            mask_arr[i, :n] = masks_l[i]
            # rows >= n stay mask=False (padding), matching M8._field_collate.

    np.savez(
        out_npz,
        occ=occ_arr,
        cells=cells_arr,
        target=target_arr,
        mask=mask_arr,
        seeds=np.asarray(seeds, dtype=np.int64),
        window_w=np.asarray(W, dtype=np.int64),
        grid_size=np.asarray(G, dtype=np.int64),
        in_channels=np.asarray(in_channels, dtype=np.int64),
    )
    return out_npz


# -----------------------------------------------------------------------------
# Task 4 — scalar temporal trainer (scratch / full_ft / lora, matched recipe)
# -----------------------------------------------------------------------------
#
# `source_ckpt` is ALWAYS provided (even for method="scratch") because the
# MODEL ARCHITECTURE (backbone_cfg, window_w, k_patrollers, token_dim,
# max_norm_residual) lives in the frozen C8 scalar source payload — we never
# re-derive or default those from cfg. `method` only controls weight loading:
#   scratch  -> build same-architecture model, random init (no source weights)
#   full_ft  -> build + load source weights, train ALL params
#   lora     -> build + load source weights + C.apply_lora + freeze non-LoRA
# Mirrors M8._train_scalar's model build / smooth-L1 loss / AdamW / grad-clip
# / epoch loop (continuous_prm_c8_dynamics_compare.py:348-452), and
# C9H.train_scalar_lora's apply_lora + freeze-non-LoRA + bounded handling
# pattern (continuous_prm_c9h_transfer.py:82-119).

SCALAR_METHODS = ("scratch", "full_ft", "lora")


def train_scalar_temporal(
    dataset_npz: "str | Path",
    out_ckpt: "str | Path",
    source_ckpt: "str | Path",
    method: str,
    cfg: C9bConfig,
    device,
    seed: int,
) -> Path:
    """Train a scalar temporal adapter at matched compute from a frozen C8 source.

    Loads the C8 scalar source payload (cpu) to recover the model ARCHITECTURE
    (backbone_cfg, window_w, k_patrollers, token_dim, max_norm_residual) —
    these are never defaulted from cfg. `method` selects weight loading:
    scratch=random-init, full_ft=load+train-all, lora=load+apply_lora+freeze
    non-LoRA params. Asserts the dataset npz's window_w/token_dim/k_patrollers
    match the source's (schema guard). Returns the Path the checkpoint was
    saved to (== Path(out_ckpt)).
    """
    import torch.nn.functional as F

    if method not in SCALAR_METHODS:
        raise ValueError(f"unknown scalar temporal method {method!r}; have {SCALAR_METHODS}")

    out_ckpt = Path(out_ckpt)
    source_ckpt = Path(source_ckpt)

    source = torch.load(source_ckpt, map_location="cpu")
    backbone_cfg = C.BackboneConfig(**source["backbone_cfg"])
    window_w = int(source["window_w"])
    k_patrollers = int(source["k_patrollers"])
    token_dim = int(source["token_dim"])
    max_norm_residual = float(source["max_norm_residual"])
    backbone_name = source["backbone"]

    npz = np.load(dataset_npz)
    if int(npz["window_w"]) != window_w:
        raise ValueError(f"dataset window_w={int(npz['window_w'])} != source window_w={window_w}")
    if int(npz["token_dim"]) != token_dim:
        raise ValueError(f"dataset token_dim={int(npz['token_dim'])} != source token_dim={token_dim}")
    if int(npz["k_patrollers"]) != k_patrollers:
        raise ValueError(f"dataset k_patrollers={int(npz['k_patrollers'])} != source k_patrollers={k_patrollers}")
    x = np.ascontiguousarray(npz["x"])
    y = np.ascontiguousarray(npz["y"])
    if x.shape[0] == 0:
        raise RuntimeError(f"scalar temporal train [{method}]: empty/all-masked dataset {dataset_npz}")

    C.set_global_seed(int(seed))

    model = C.ContinuousHeuristicModel(
        backbone_cfg, token_dim=token_dim, max_norm_residual=max_norm_residual
    ).to(device)

    if method != "scratch":
        C.safe_load_state(model, source_ckpt)
    if method == "lora":
        C.apply_lora(model, rank=int(cfg.rank), alpha=float(cfg.alpha))
        C.set_lora_trainable(model)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    train_cfg = C.TrainingConfig()
    ds = C.ArrayDataset(x, y)
    loader = C.make_loader(ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)

    grad_clip = 1.0
    model.train()
    for _ep in range(int(cfg.epochs)):
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            if not torch.isfinite(pred).all():
                raise FloatingPointError(f"scalar temporal train {method}: non-finite prediction")
            loss = F.smooth_l1_loss(pred, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"scalar temporal train {method}: non-finite loss")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()

    payload = {
        "model": model.state_dict(),
        "backbone_cfg": asdict(backbone_cfg),
        "window_w": int(window_w),
        "k_patrollers": int(k_patrollers),
        "token_dim": int(token_dim),
        "max_norm_residual": float(max_norm_residual),
        "backbone": backbone_name,
        "method": method,
    }
    if method == "lora":
        payload["lora_rank"] = int(cfg.rank)
        payload["alpha"] = float(cfg.alpha)

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_ckpt)
    return out_ckpt


# -----------------------------------------------------------------------------
# Task 5 — field temporal trainer (scratch / full_ft / lora-via-conv-LoRA,
# matched recipe)
# -----------------------------------------------------------------------------
#
# `source_ckpt` is ALWAYS provided (even for method="scratch") because the
# MODEL ARCHITECTURE (in_channels, window_w, grid_size, backbone name) lives in
# the frozen C8 field source payload — we never re-derive or default those from
# cfg. `method` only controls weight loading:
#   scratch  -> build same-architecture U-Net, random init (no source weights)
#   full_ft  -> build + C.safe_load_state(model, source_ckpt), train ALL params
#   lora     -> build + load source + C9H.apply_conv_lora + freeze non-LoRA
#               params (mirrors C9H.train_field_model's freeze pattern)
#
# CRITICAL: the loss/forward/gather/mask mirrors M8._train_field EXACTLY
# (continuous_prm_c8_dynamics_compare.py:584-708) — NOT C9H.train_field_model's
# C6 heatmap loss (ranking/path/consistency), since the frozen C8 field source
# was trained with `_train_field`'s loss (plain masked smooth-L1 against the
# gathered node residual), not C6's multi-term heatmap loss:
#   pred_grid = C6.model_output_residual(model(occ))     # (B, G, G)
#   ix, iy = cells[..., 0].clamp(0, G-1), cells[..., 1].clamp(0, G-1)
#   pred_nodes = gather(pred_grid.reshape(B, G*G), 1, ix*G + iy)   # (B, N)
#   loss = F.smooth_l1_loss(pred_nodes[mask], target[mask])
#
# T3's field npz pre-pads (cells, target, mask) to a GLOBAL Nmax across all
# (world, t) samples (mask=False on padded rows), so we build a flat Dataset
# directly over the npz arrays (no re-batching/re-padding needed — it is
# already padded) rather than reusing M8._FieldTemporalDataset/_field_collate
# (which pad per-BATCH from ragged per-world N; T3's on-disk schema differs).

FIELD_METHODS = ("scratch", "full_ft", "lora")


class _FieldTemporalArrayDataset(torch.utils.data.Dataset):
    """Flat (S,) dataset over T3's pre-padded field npz arrays.

    Each item is one (world, t) sample: occ (8+W, G, G), cells (Nmax, 2),
    target (Nmax,), mask (Nmax,) — already padded by collect_temporal_dataset
    (mask=False on padded rows), so no further collate-time padding is needed;
    the default `torch.utils.data.DataLoader` collate (stacking on dim 0) is
    correct as-is since Nmax is constant across all samples in the npz.
    """

    def __init__(self, occ: np.ndarray, cells: np.ndarray, target: np.ndarray, mask: np.ndarray):
        self.occ = occ
        self.cells = cells
        self.target = target
        self.mask = mask

    def __len__(self) -> int:
        return int(self.occ.shape[0])

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(np.ascontiguousarray(self.occ[i])),
            torch.from_numpy(np.ascontiguousarray(self.cells[i])),
            torch.from_numpy(np.ascontiguousarray(self.target[i])),
            torch.from_numpy(np.ascontiguousarray(self.mask[i])),
        )


def train_field_temporal(
    dataset_npz: "str | Path",
    out_ckpt: "str | Path",
    source_ckpt: "str | Path",
    method: str,
    cfg: C9bConfig,
    device,
    seed: int,
) -> Path:
    """Train a field temporal adapter at matched compute from a frozen C8 source.

    Loads the C8 field source payload (cpu) to recover the model ARCHITECTURE
    (in_channels, window_w, grid_size, backbone name) — never defaulted from
    cfg. `method` selects weight loading: scratch=random-init,
    full_ft=load+train-all, lora=load+apply_conv_lora+freeze non-LoRA params.
    Asserts the dataset npz's window_w/in_channels/grid_size match the
    source's (schema guard). Mirrors M8._train_field's forward (U-Net ->
    model_output_residual) -> gather-at-cells -> mask -> smooth-L1 loss,
    AdamW + grad-clip, matched recipe (cfg.epochs/lr/weight_decay). Returns
    the Path the checkpoint was saved to (== Path(out_ckpt)).
    """
    import torch.nn.functional as F
    import continuous_prm_c6_heatmap_value_field as C6

    if method not in FIELD_METHODS:
        raise ValueError(f"unknown field temporal method {method!r}; have {FIELD_METHODS}")

    out_ckpt = Path(out_ckpt)
    source_ckpt = Path(source_ckpt)

    source = torch.load(source_ckpt, map_location="cpu")
    in_channels = int(source["in_channels"])
    window_w = int(source["window_w"])
    grid_size = int(source["grid_size"])
    backbone_name = source["backbone"]

    npz = np.load(dataset_npz)
    if int(npz["window_w"]) != window_w:
        raise ValueError(f"dataset window_w={int(npz['window_w'])} != source window_w={window_w}")
    if int(npz["in_channels"]) != in_channels:
        raise ValueError(f"dataset in_channels={int(npz['in_channels'])} != source in_channels={in_channels}")
    if int(npz["grid_size"]) != grid_size:
        raise ValueError(f"dataset grid_size={int(npz['grid_size'])} != source grid_size={grid_size}")

    occ = np.ascontiguousarray(npz["occ"])
    cells = np.ascontiguousarray(npz["cells"])
    target = np.ascontiguousarray(npz["target"])
    mask = np.ascontiguousarray(npz["mask"])
    if occ.shape[0] == 0:
        raise RuntimeError(f"field temporal train [{method}]: empty dataset {dataset_npz}")

    C.set_global_seed(int(seed))

    model = C6.build_model(backbone_name, in_channels=in_channels).to(device)

    if method != "scratch":
        C.safe_load_state(model, source_ckpt)
    if method == "lora":
        C9H.apply_conv_lora(model, rank=int(cfg.rank), alpha=float(cfg.alpha))
        C.set_lora_trainable(model)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    ds = _FieldTemporalArrayDataset(occ, cells, target, mask)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    grad_clip = 1.0
    model.train()
    for _ep in range(int(cfg.epochs)):
        for occ_b, cells_b, target_b, mask_b in loader:
            occ_b = occ_b.to(device=device, dtype=torch.float32)
            cells_b = cells_b.to(device)
            target_b = target_b.to(device=device, dtype=torch.float32)
            mask_b = mask_b.to(device)

            pred_grid = C6.model_output_residual(model(occ_b))
            if not torch.isfinite(pred_grid).all():
                raise FloatingPointError(f"field temporal train {method}: non-finite grid")

            B, G, _ = pred_grid.shape
            ix = cells_b[..., 0].clamp(0, G - 1)
            iy = cells_b[..., 1].clamp(0, G - 1)
            flat = pred_grid.reshape(B, G * G)
            lin = ix * G + iy
            pred_nodes = torch.gather(flat, 1, lin)

            m = mask_b.bool()
            if not bool(m.any()):
                continue
            loss = F.smooth_l1_loss(pred_nodes[m], target_b[m])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"field temporal train {method}: non-finite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()

    payload = {
        "model": model.state_dict(),
        "in_channels": int(in_channels),
        "window_w": int(window_w),
        "grid_size": int(grid_size),
        "backbone": backbone_name,
        "method": method,
    }
    if method == "lora":
        payload["lora_rank"] = int(cfg.rank)
        payload["alpha"] = float(cfg.alpha)

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_ckpt)
    return out_ckpt


# -----------------------------------------------------------------------------
# Task 6 — temporal provider loaders (LoRA-aware, aware/blind, scalar+field)
# -----------------------------------------------------------------------------
#
# Loads a checkpoint into the matching C8 temporal provider. Handles BOTH
# payload shapes:
#   - frozen C8 source (zero_shot arm): no "lora_rank", no "method" key.
#   - C9b adapter (Task 4 scalar / Task 5 field output): carries "method", and
#     "lora_rank"/"alpha" iff method == "lora".
# Branches scalar-vs-field via `_is_field(backbone)` and LoRA-vs-plain via
# `"lora_rank" in payload` (mirrors C9H.load_scalar_provider_c9h's pattern:
# apply_lora/apply_conv_lora BEFORE load_state_dict(strict=True), since the
# LoRA A/B parametrization params are part of the saved state_dict). `bounded`
# only affects the field provider's max_norm_residual (scalar always uses the
# source's own clamp, which is already baked into the payload).


def load_temporal_provider(ckpt: "str | Path", backbone: str, device, bounded: bool = True):
    """Load `ckpt` (frozen C8 source OR C9b adapter) into a C8 temporal provider.

    `backbone` selects scalar vs field schema via `_is_field` (must match the
    checkpoint's own architecture — it is only used for branching/the
    provider's display name, never to override payload-derived shapes).
    Awareness (time_blind) is derived from the payload's own `window_w`
    (`time_blind = (window_w == 0)`), NOT from `cfg.awareness`, since the
    checkpoint's window_w is authoritative for what it was trained/built with.
    Returns an eval-mode `DP.ScalarTemporalProvider` or
    `DP.ValueFieldTemporalProvider` wrapping the loaded model.
    """
    ckpt = Path(ckpt)
    payload = torch.load(ckpt, map_location="cpu")
    window_w = int(payload["window_w"])
    time_blind = window_w == 0
    is_lora = "lora_rank" in payload

    if not _is_field(backbone):
        backbone_cfg = C.BackboneConfig(**payload["backbone_cfg"])
        token_dim = int(payload["token_dim"])
        max_norm_residual = float(payload["max_norm_residual"])
        k_patrollers = int(payload["k_patrollers"])
        backbone_name = payload["backbone"]

        model = C.ContinuousHeuristicModel(
            backbone_cfg, token_dim=token_dim, max_norm_residual=max_norm_residual
        )
        if is_lora:
            C.apply_lora(model, rank=int(payload["lora_rank"]), alpha=float(payload["alpha"]))
        model.load_state_dict(payload["model"], strict=True)
        model = model.to(device).eval()

        return DP.ScalarTemporalProvider(
            model, device, backbone_name, window_w, k_patrollers, max_norm_residual,
            time_blind=time_blind,
        )

    # --- Field branch -------------------------------------------------------
    import continuous_prm_c6_heatmap_value_field as C6

    in_channels = int(payload["in_channels"])
    grid_size = int(payload["grid_size"])
    backbone_name = payload["backbone"]

    model = C6.build_model(backbone_name, in_channels=in_channels)
    if is_lora:
        C9H.apply_conv_lora(model, rank=int(payload["lora_rank"]), alpha=float(payload["alpha"]))
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device).eval()

    mnr = float(payload.get("max_norm_residual", 4.0)) if bounded else float("inf")
    return DP.ValueFieldTemporalProvider(
        model, grid_size, device, backbone_name, window_w,
        max_norm_residual=mnr, time_blind=time_blind,
    )


# -----------------------------------------------------------------------------
# Task 7 — adapt mode + manifest
# -----------------------------------------------------------------------------
#
# Orchestrates training EVERY adapter arm: for each target x backbone x
# awareness x method x K x seed, resolve the frozen C8 source for
# (backbone, awareness), read window_w/k_patrollers FROM THAT SOURCE PAYLOAD
# (never defaulted from cfg — aware sources carry window_w=8, blind carry
# window_w=0; scalar sources carry k_patrollers=4; field sources have none),
# collect (or reuse) a K-shot temporal dataset ONCE per
# (target, substrate, awareness, K, seed) cell — scalar_hrm and scalar_onlstm
# share the same "scalar" substrate dataset since collect_temporal_dataset's
# scalar schema does not depend on which scalar backbone will consume it —
# then dispatch the scalar/field trainer per method. zero_shot is NOT trained
# here: it is just the frozen source, recorded under manifest["sources"];
# Task 8 (eval) builds the zero_shot arm directly from those paths. Mirrors
# C9.run_adapt's resume/manifest pattern (continuous_prm_c9_transfer.py:222).


def run_adapt(cfg: C9bConfig, device, only_targets: Optional[List[str]] = None) -> dict:
    install()
    out_dir = Path(cfg.out_dir)
    ds_dir = out_dir / "datasets"
    ck_dir = out_dir / "checkpoints"
    C.ensure_dir(ds_dir)
    C.ensure_dir(ck_dir)

    sources = resolve_sources(cfg)
    targets = only_targets if only_targets is not None else _parse_csv(cfg.targets)
    backbones = _parse_csv(cfg.backbones)
    awareness_list = cfg.awareness_list()
    methods = _parse_csv(cfg.methods)
    Ks = [k for k in _parse_ints(cfg.k_grid) if k > 0]

    arms: List[dict] = []
    for target in targets:
        for backbone in backbones:
            substrate = "field" if _is_field(backbone) else "scalar"
            for awareness in awareness_list:
                source_ckpt = sources[(backbone, awareness)]
                source_payload = torch.load(source_ckpt, map_location="cpu")
                window_w = int(source_payload["window_w"])
                if _is_field(backbone):
                    k_patrollers = int(source_payload.get("k_patrollers", 0))
                else:
                    k_patrollers = int(source_payload["k_patrollers"])

                for K in Ks:
                    for seed_idx in range(int(cfg.n_adapt_seeds)):
                        seeds = adapt_world_seeds(target, K, seed_idx, cfg)
                        ds_npz = ds_dir / (
                            f"c9b__{target}__{substrate}__{awareness}__K{K}__s{seed_idx}.npz"
                        )
                        if not ds_npz.exists():
                            collect_temporal_dataset(
                                target, seeds, backbone, window_w, k_patrollers,
                                cfg.grid_size, ds_npz,
                            )

                        for method in methods:
                            ck = ck_dir / (
                                f"c9b__{target}__{backbone}__{awareness}__{method}__K{K}__s{seed_idx}.pt"
                            )
                            train_seed = C9.adapt_seed(target, K, seed_idx, cfg.seed)
                            if not ck.exists():
                                if _is_field(backbone):
                                    train_field_temporal(
                                        ds_npz, ck, source_ckpt=source_ckpt, method=method,
                                        cfg=cfg, device=device, seed=train_seed,
                                    )
                                else:
                                    train_scalar_temporal(
                                        ds_npz, ck, source_ckpt=source_ckpt, method=method,
                                        cfg=cfg, device=device, seed=train_seed,
                                    )
                            arms.append(dict(
                                target=target, backbone=backbone, awareness=awareness,
                                method=method, K=K, seed=seed_idx, window_w=window_w,
                                ckpt=str(ck),
                            ))
                        print(f"[{now_str()}] c9b adapt: {target} {backbone} {awareness} K={K} s={seed_idx} done", flush=True)

    manifest = {
        "arms": arms,
        "targets": targets,
        "backbones": backbones,
        "awareness": awareness_list,
        "methods": methods,
        "k_grid": Ks,
        "n_adapt_seeds": int(cfg.n_adapt_seeds),
        "seed": int(cfg.seed),
        "sources": {f"{b}__{a}": str(path) for (b, a), path in sources.items()},
    }
    C.write_json(out_dir / "adapt_manifest.json", manifest)
    return manifest


# -----------------------------------------------------------------------------
# Task 8 — eval mode (space-time arms over the budget grid -> raw CSV)
# -----------------------------------------------------------------------------
#
# Builds every provider for a target — euclid/oracle (no model), a frozen
# zero_shot provider per (backbone, awareness) straight from manifest["sources"],
# and one provider per adapted arm in manifest["arms"] — then runs C8's
# space-time A*/focal eval (`DP.run_world_arms_spacetime`) on TEST worlds
# (materialized via the SAME memoized label collector Task 3 uses, so the world/
# rm/dyn triple is byte-identical to what adaptation could have seen — modulo
# the adapt/test seed split enforced by Task 2). Mirrors C9.run_eval's
# providers/meta-dict + per-target shard + merged-CSV pattern
# (continuous_prm_c9_transfer.py:281-331), but reuses C8's space-time arm
# runner (continuous_prm_dynamic_providers.py:300-359) instead of C9's static
# P.run_world_arms, since C9b targets are dynamic (moving-patroller) worlds.
#
# `run_world_arms_spacetime` emits one record dict per (provider, mode[, w],
# budget) via `_arm_record_st` with EXACTLY these keys: provider, mode, w,
# budget, found, expansions, arrival, optimal_arrival, suboptimality, closed,
# nonfinite (confirmed by reading continuous_prm_dynamic_providers.py:362-371).
# That matches C9B_RAW_COLS's tail verbatim, so no key renaming is needed here
# — only target/backbone/awareness/method/K/seed/world_index are added via the
# meta dict (built per-provider, keyed by provider name) the way C9.run_eval
# does it.

C9B_RAW_COLS = ["target", "backbone", "awareness", "method", "K", "seed", "world_index",
                "provider", "mode", "w", "budget", "found", "expansions", "arrival",
                "optimal_arrival", "suboptimality", "closed", "nonfinite"]


def budgets_for(target: str, cfg: C9bConfig) -> List[int]:
    """Resolve the budget grid to eval `target` at.

    If cfg.budgets is set (explicit CLI/test override), use it verbatim for
    EVERY target (this is what the smoke test exercises). Otherwise read C8's
    calibration.json (written by C8's own T12 calibration pass) and use the
    single binding (lowest) budget of the calibrated band for `target` — the
    fair-fight point C9b's analysis is anchored on (see module docstring /
    task spec). Raises if neither is available, since silently picking an
    arbitrary budget would make the eval not comparable across arms.
    """
    if str(cfg.budgets).strip():
        return _parse_ints(cfg.budgets)
    calib = Path(cfg.source_dir) / "calibration.json"
    band = (json.loads(calib.read_text()).get("budgets", {}) or {}).get(target) if calib.exists() else None
    if not band:
        raise RuntimeError(f"no budget band for {target}: pass --budgets or provide {calib}")
    return [int(min(band))]


def run_eval(cfg: C9bConfig, device, only_targets: Optional[List[str]] = None) -> Path:
    """Eval every provider (euclid, oracle, frozen zero_shot per (backbone,
    awareness), and every adapted arm) on TEST worlds via C8's space-time A*,
    writing a per-target shard CSV plus a merged raw CSV (C9B_RAW_COLS).
    Returns the Path of the merged raw CSV.
    """
    install()
    out_dir = Path(cfg.out_dir)
    res_dir = out_dir / "results"
    shard_dir = res_dir / "_shards"
    C.ensure_dir(shard_dir)

    w_values = [float(x) for x in _parse_csv(cfg.w_values)]
    manifest = json.loads((out_dir / "adapt_manifest.json").read_text())
    targets = only_targets if only_targets is not None else _parse_csv(cfg.targets)

    all_rows: List[dict] = []
    for target in targets:
        budgets = budgets_for(target, cfg)

        # Materialize TEST worlds once (world, rm, dyn, params), reusing Task 3's
        # memoized label collector so this is the SAME world the labelset/
        # adaptation pipeline would see for this (target, seed).
        seeds = test_world_seeds(target, cfg)
        worlds = []
        for wi, s in enumerate(seeds):
            lab = _collect_world_labels_memo(target, s, cfg.grid_size)
            if lab is None:
                continue
            worlds.append((wi, lab))

        providers: Dict[str, object] = {"euclid": DP.EuclidTimeProvider(), "oracle": DP.OracleProvider()}
        meta: Dict[str, dict] = {
            "euclid": dict(method="euclid", backbone="", awareness="", K=-1, seed=-1),
            "oracle": dict(method="oracle", backbone="", awareness="", K=-1, seed=-1),
        }

        # Frozen zero_shot providers, one per (backbone, awareness) actually
        # adapted from in this run (manifest["sources"] is the authoritative
        # backbone/awareness -> frozen-C8-checkpoint map Task 7 recorded).
        for backbone in _parse_csv(cfg.backbones):
            for awareness in cfg.awareness_list():
                src = manifest["sources"][f"{backbone}__{awareness}"]
                p = load_temporal_provider(Path(src), backbone, device)
                key = f"zeroshot_{backbone}_{awareness}"
                p.name = key
                providers[key] = p
                meta[key] = dict(method="zero_shot", backbone=backbone, awareness=awareness, K=0, seed=-1)

        # Adapted arms for THIS target only.
        for a in manifest["arms"]:
            if a["target"] != target:
                continue
            p = load_temporal_provider(Path(a["ckpt"]), a["backbone"], device)
            key = f'{a["method"]}_{a["backbone"]}_{a["awareness"]}_K{a["K"]}_s{a["seed"]}'
            p.name = key
            providers[key] = p
            meta[key] = dict(method=a["method"], backbone=a["backbone"], awareness=a["awareness"],
                              K=a["K"], seed=a["seed"])

        rows: List[dict] = []
        for wi, lab in worlds:
            pp = lab["params"]
            recs = DP.run_world_arms_spacetime(
                lab["world"], lab["rm"], lab["dyn"], providers, budgets, w_values,
                pp["v_agent"], pp["dt"], int(pp["t_max"]),
            )
            for r in recs:
                m = meta.get(r["provider"], dict(method=r["provider"], backbone="", awareness="", K=-1, seed=-1))
                r.update(dict(target=target, world_index=wi, **m))
                rows.append(r)

        sp = shard_dir / f"{target}.csv"
        with open(sp, "w", newline="") as f:
            wri = _csv.DictWriter(f, fieldnames=C9B_RAW_COLS)
            wri.writeheader()
            for r in rows:
                wri.writerow({k: r.get(k, "") for k in C9B_RAW_COLS})
        all_rows.extend(rows)
        print(f"[{now_str()}] c9b eval {target}: {len(rows)} rows "
              f"({len(providers)} providers x {len(worlds)} worlds x {len(budgets)} budgets)", flush=True)

    raw = res_dir / "continuous_prm_c9b_eval_raw.csv"
    with open(raw, "w", newline="") as f:
        wri = _csv.DictWriter(f, fieldnames=C9B_RAW_COLS)
        wri.writeheader()
        for r in all_rows:
            wri.writerow({k: r.get(k, "") for k in C9B_RAW_COLS})
    print(f"[{now_str()}] c9b eval: merged {len(all_rows)} rows -> {raw}", flush=True)
    return raw
