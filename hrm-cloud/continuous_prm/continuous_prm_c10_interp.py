#!/usr/bin/env python3
"""C10: parameter-space LoRA interpolation for zero-shot transfer to interior held-out
families. New-file-only; reuses C9/C9h + C5/C7 generators + common.
See docs/superpowers/specs/2026-06-29-c10-interp-design.md.
"""
from __future__ import annotations
import argparse, dataclasses, csv as _csv, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import continuous_prm_common as C
import continuous_prm_providers as P
import continuous_prm_c5_hard_obstacle_encoder as C5
import continuous_prm_c7_hard_maps as H7
import continuous_prm_c9_transfer as C9
import continuous_prm_c9h_transfer as C9H

HARD_MODE = C5.HARD_MODE

SOURCE_FAMILIES = [
    "C10_maze_d0", "C10_maze_d1", "C10_maze_d2", "C10_maze_d3",
    "C10_rooms_s10", "C10_rooms_s20", "C10_rooms_s30", "C10_rooms_s40",
]
TARGET_FAMILIES = ["C10_maze_tgt", "C10_rooms_t25", "C10_rooms_t35"]


def _maze(base, gap, clutter):
    return dataclasses.replace(base, name="C_hard_maze", mode=HARD_MODE,
                               gap_width_frac=gap, extra_clutter_range=clutter,
                               obstacle_count_range=clutter, rectangle_count_range=(3, 3), is_ood=False)


def _rooms(base, side):
    return dataclasses.replace(base, name="C_hard_rooms", mode=HARD_MODE, side_len=float(side), is_ood=False)


def c10_family_specs() -> Dict[str, "C.AnchorSpec"]:
    H7.install_c7_hard_maps()
    base = C.build_anchor_specs()
    mz, rm = base["C_hard_maze"], base["C_hard_rooms"]
    return {
        "C10_maze_d0": _maze(mz, 0.18, (2, 4)),
        "C10_maze_d1": _maze(mz, 0.15, (4, 7)),
        "C10_maze_d2": _maze(mz, 0.13, (7, 11)),
        "C10_maze_d3": _maze(mz, 0.11, (10, 14)),
        "C10_maze_tgt": _maze(mz, 0.14, (6, 9)),
        "C10_rooms_s10": _rooms(rm, 1.0),
        "C10_rooms_s20": _rooms(rm, 2.0),
        "C10_rooms_s30": _rooms(rm, 3.0),
        "C10_rooms_s40": _rooms(rm, 4.0),
        "C10_rooms_t25": _rooms(rm, 2.5),
        "C10_rooms_t35": _rooms(rm, 3.5),
    }


def family_descriptor_centroid(spec, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    descs, tries = [], 0
    while len(descs) < n and tries < n * 50:
        tries += 1
        w = C.build_world(spec, int(rng.integers(0, 2**31 - 1)), 0.45)
        if w is not None:
            descs.append(np.asarray(w.descriptor, dtype=np.float64))
    if not descs:
        raise RuntimeError(f"no worlds for {spec.name}")
    return np.mean(np.stack(descs, 0), axis=0)


def bracketing_ok(z_t, source_centroids, rel_tol: float = 0.05) -> Tuple[bool, List[int]]:
    """Check whether target centroid ``z_t`` is bracketed by the source hull.

    Degenerate (inactive) dims -- where the sources barely vary
    (``spread <= 1e-6``) -- are SKIPPED: a near-constant dim cannot bracket and
    sampling noise there would otherwise produce spurious violations. On active
    dims a small tolerance proportional to the per-dim spread is allowed, so a
    violation is flagged only if ``z_t[d] < lo[d] - rel_tol*spread[d]`` or
    ``z_t[d] > hi[d] + rel_tol*spread[d]``. ``viol`` lists only active-dim
    violations.
    """
    Z = np.stack([np.asarray(c, dtype=np.float64) for c in source_centroids], 0)
    lo, hi = Z.min(0), Z.max(0)
    spread = hi - lo
    z = np.asarray(z_t, dtype=np.float64)
    viol = []
    for d in range(z.shape[0]):
        if spread[d] <= 1e-6:
            continue  # inactive dim: sources don't vary, cannot bracket
        tol = rel_tol * spread[d]
        if z[d] < lo[d] - tol or z[d] > hi[d] + tol:
            viol.append(int(d))
    return (len(viol) == 0, viol)


@dataclass
class C10Config:
    source_dir: str = "hrm-cloud/continuous_prm/runs/c7_local"
    out_dir: str = "hrm-cloud/continuous_prm/runs/c10_local"
    backbones: str = "hrm,onlstm"
    n_src_worlds: int = 48
    n_centroid_worlds: int = 24
    n_test: int = 30
    rank: int = 8
    alpha: float = 1.0
    rbf_sigma: float = 1.0
    epochs: int = 10
    lr: float = 2.0e-4
    roadmap_nodes: int = 192
    roadmap_k: int = 7
    budgets: str = "150,250,400,600,900,1300"   # eval grid; binding budget chosen in analyze
    w_values: str = "1.0,1.1"
    seed: int = 1234
    scale: str = "local"
    mode: str = "full"
    cpu: bool = False


def now_str() -> str:
    return C.now_str()


def rbf_weights(z_t, centroids, sigma: float) -> np.ndarray:
    Z = np.stack([np.asarray(c, dtype=np.float64) for c in centroids], 0)
    s = Z.std(0); s = np.where(s < 1e-6, 1.0, s)        # per-dim scale, epsilon-floored
    d2 = (((np.asarray(z_t, dtype=np.float64)[None, :] - Z) / s) ** 2).sum(1)
    logits = -d2 / (2.0 * max(1e-12, float(sigma) ** 2))
    logits -= logits.max()
    w = np.exp(logits); w /= w.sum()
    return w


def nearest_weights(z_t, centroids) -> np.ndarray:
    Z = np.stack([np.asarray(c, dtype=np.float64) for c in centroids], 0)
    s = Z.std(0); s = np.where(s < 1e-6, 1.0, s)
    d2 = (((np.asarray(z_t, dtype=np.float64)[None, :] - Z) / s) ** 2).sum(1)
    w = np.zeros(Z.shape[0]); w[int(np.argmin(d2))] = 1.0
    return w


def uniform_weights(n: int) -> np.ndarray:
    return np.full(int(n), 1.0 / int(n), dtype=np.float64)


# ---------------------------------------------------------------------------
# Task 3 — Source-expert training + descriptor centroids
# ---------------------------------------------------------------------------

import hashlib


def _stable_offset(s: str) -> int:
    """Deterministic per-string seed offset (Python's hash() is salted across processes)."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) & 0xFFFF


def _expert_ckpt(out_dir, family, backbone):
    return Path(out_dir) / "checkpoints" / f"c10_src__{family}__{backbone}.pt"


def train_source_experts(cfg: C10Config, device, only_families=None) -> dict:
    specs = c10_family_specs()
    fams = only_families or SOURCE_FAMILIES
    out_dir = Path(cfg.out_dir); ds_dir = out_dir / "datasets"
    C.ensure_dir(out_dir / "checkpoints"); C.ensure_dir(ds_dir)
    rmcfg = C.RoadmapConfig(n_nodes=cfg.roadmap_nodes, k_neighbors=cfg.roadmap_k)
    experts = []
    for backbone in C9._parse_csv(cfg.backbones):
        base = C9.load_source_base(Path(cfg.source_dir), backbone, device)
        tcfg = dataclasses.replace(base.train_cfg, base_epochs=int(cfg.epochs), lr=float(cfg.lr))
        for fam in fams:
            spec = specs[fam]
            seed = int(cfg.seed) + _stable_offset(fam)
            npz = C.collect_task_dataset(spec, ds_dir, f"src_{fam}", int(cfg.n_src_worlds),
                                         C9.SCALAR_NODES_PER_WORLD, rmcfg, base.feature_cfg, seed=seed)
            ck = _expert_ckpt(out_dir, fam, backbone)
            if not ck.exists():
                C9H.train_scalar_lora(base.backbone_cfg, npz, ck, base.feature_cfg, tcfg, device,
                                      seed=seed, init_ckpt=base.ckpt_path, rank=int(cfg.rank),
                                      alpha=float(cfg.alpha), bounded=True)
            cent = family_descriptor_centroid(spec, int(cfg.n_centroid_worlds), seed + 1)
            experts.append(dict(family=fam, backbone=backbone, ckpt=str(ck),
                                centroid=[float(v) for v in cent]))
            print(f"[{now_str()}] c10 source expert: {fam} {backbone} done", flush=True)
    man = {"experts": experts, "source_families": fams, "backbones": C9._parse_csv(cfg.backbones)}
    C.write_json(out_dir / "source_manifest.json", man)
    return man


# ---------------------------------------------------------------------------
# Task 4 — Weight-merge baker
# ---------------------------------------------------------------------------

def bake_weight_merge(base_ckpt, expert_ckpts, weights, out_ckpt, device) -> Path:
    """Bake W' = W_base + Σ_k w_k · scale_k · (B_k @ A_k) into a plain merged checkpoint.

    Loads the plain avgbase weights, adds the weighted LoRA deltas for every
    LoRA target, and saves a plain checkpoint (no LoRA params) that can be
    loaded by load_scalar_provider_c9h without apply_lora.

    MHA in_proj_weight special case: iter_lora_targets yields
    name="…attn.in_proj" and sub=MultiheadAttention, but PyTorch registers the
    parametrization on the MHA module itself, so the state-dict prefix is
    name[:-len(".in_proj")] (i.e., "…attn") not "…attn.in_proj".
    """
    import torch.nn as nn

    base_payload = torch.load(Path(base_ckpt), map_location="cpu")
    bb = C.BackboneConfig(**base_payload["backbone_cfg"])
    fc = C.FeatureConfig(**base_payload["feature_cfg"])
    tc = C.TrainingConfig(**base_payload["train_cfg"])
    model = C.build_model(bb, fc, tc, device)
    C.safe_load_state(model, Path(base_ckpt))   # plain base weights at {layer}.weight

    states = [torch.load(Path(p), map_location="cpu")["model"] for p in expert_ckpts]
    w = np.asarray(weights, dtype=np.float64)

    seen: set = set()
    for name, sub, attr in C.iter_lora_targets(model):
        key_id = (id(sub), attr)
        if key_id in seen:
            continue
        seen.add(key_id)

        # Derive the state-dict prefix that PyTorch actually used when
        # registering the parametrization.  For MHA in_proj_weight the
        # parametrization is registered on the MHA module (sub), whose
        # named_modules() name does NOT have the ".in_proj" suffix that
        # iter_lora_targets appends to name.
        if isinstance(sub, nn.MultiheadAttention) and attr == "in_proj_weight":
            sd_prefix = name[: -len(".in_proj")]
        else:
            sd_prefix = name

        W = getattr(sub, attr).data    # [out, in] or [3*embed, embed] for MHA

        delta = torch.zeros_like(W)
        for k, st in enumerate(states):
            if abs(float(w[k])) < 1e-12:
                continue
            A = st[f"{sd_prefix}.parametrizations.{attr}.0.A"]
            B = st[f"{sd_prefix}.parametrizations.{attr}.0.B"]
            scale = float(st[f"{sd_prefix}.parametrizations.{attr}.0.adapter_scale"])
            delta = delta + float(w[k]) * scale * (B.to(W.dtype) @ A.to(W.dtype)).reshape(W.shape)

        with torch.no_grad():
            getattr(sub, attr).add_(delta.to(W.device, W.dtype))

    payload = {
        "model": model.state_dict(),
        "backbone_cfg": asdict(bb),
        "feature_cfg": asdict(fc),
        "train_cfg": asdict(tc),
        "max_norm_residual": float(tc.max_norm_residual),
    }
    C.ensure_dir(Path(out_ckpt).parent)
    torch.save(payload, Path(out_ckpt))
    return Path(out_ckpt)


# ---------------------------------------------------------------------------
# Task 5 — Prediction-mix provider
# ---------------------------------------------------------------------------

class _PredMixProvider(P.HeuristicProvider):
    """RBF-weighted prediction-space mix: h = euclid + side*clip(Σ w_k ŷ_k, 0, B)."""
    def __init__(self, models, weights, feature_cfg, device, max_resid, name):
        self.models = models; self.w = np.asarray(weights, dtype=np.float64)
        self.feature_cfg = feature_cfg; self.device = device
        self.max_resid = float(max_resid); self.name = name

    def node_h(self, world, roadmap, goal_idx: int = 1) -> np.ndarray:
        feats = C5.make_hard_features_for_roadmap(world, roadmap, self.feature_cfg)
        acc = None
        for wk, m in zip(self.w, self.models):
            if abs(float(wk)) < 1e-12:
                continue
            yk = np.asarray(C.predict_norm_residuals(m, feats, self.device), dtype=np.float64)
            acc = (wk * yk) if acc is None else acc + wk * yk
        if acc is None:
            acc = np.zeros(roadmap.points.shape[0])
        yhat = np.clip(acc, 0.0, self.max_resid)
        euclid = P.euclid_to_goal(roadmap, goal_idx)
        h = euclid + world.side_len * yhat
        if not np.all(np.isfinite(h)):
            raise FloatingPointError("_PredMixProvider produced non-finite h")
        return np.maximum(h, 0.0)


def make_pred_mix_provider(expert_ckpts, weights, device, name) -> "_PredMixProvider":
    """Build a prediction-mix provider from a list of expert checkpoint paths and blend weights.

    Loads each expert via load_scalar_provider_c9h (applies LoRA, sets max_norm_residual),
    extracts the model, and wraps all experts in a _PredMixProvider.  feature_cfg and
    max_norm_residual are taken from the last loaded expert (all experts share the same
    backbone config so these values are identical across experts).
    """
    models, fc, mnr = [], None, 4.0
    for p in expert_ckpts:
        prov = C9H.load_scalar_provider_c9h(Path(p), device)   # builds + loads expert model (LoRA applied)
        models.append(prov.model); fc = prov.feature_cfg; mnr = prov.max_norm_residual
    return _PredMixProvider(models, weights, fc, device, mnr, name)
