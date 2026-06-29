#!/usr/bin/env python3
"""C9h: matched-compute transfer hardening (LoRA bounded/unbounded vs full-FT,
scalar + field conv-LoRA). New-file-only; reuses C9 (frozen) + C6/C7 field stack.
See docs/superpowers/specs/2026-06-29-c9-hardening-design.md.
"""
from __future__ import annotations
import argparse, dataclasses, csv as _csv, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

import continuous_prm_common as C
import continuous_prm_providers as P
import continuous_prm_c6_heatmap_value_field as C6
import continuous_prm_c7_hard_maps as H7
import continuous_prm_c9_transfer as C9


def _iter_conv2d(module: nn.Module):
    for sub in module.modules():
        if isinstance(sub, nn.Conv2d):
            yield sub


def apply_conv_lora(unet: nn.Module, rank: int, alpha: float, init_scale: float = 0.01) -> int:
    """Register a shape-agnostic SingleAdapterLoRA parametrization on every Conv2d.weight
    in the U-Net (reuses common.SingleAdapterLoRA; B init zero => identity at init).
    Returns the number of wrapped conv weights. set_lora_trainable then trains only A/B."""
    wrapped = 0
    for conv in _iter_conv2d(unet):
        if parametrize.is_parametrized(conv, "weight"):
            continue
        w = conv.weight
        parametrize.register_parametrization(
            conv, "weight", C.SingleAdapterLoRA(w.data, rank, alpha, init_scale=init_scale), unsafe=True)
        wrapped += 1
    return wrapped


@dataclass
class C9hConfig:
    source_dir: str = "hrm-cloud/continuous_prm/runs/c7_local"
    out_dir: str = "hrm-cloud/continuous_prm/runs/c9h_local"
    targets: str = "C_hard_maze_dense,C_hard_bugtrap,C_hard_rooms_large"
    backbones: str = "hrm,onlstm,unet"
    methods: str = "lora_bounded,lora_unbounded,full_ft,scratch"
    k_grid: str = "1,4,16"
    n_adapt_seeds: int = 3
    n_test: int = 30
    epochs: int = 10
    lr: float = 2.0e-4
    rank: int = 8
    alpha: float = 1.0
    grid_size: int = 64
    roadmap_nodes: int = 192
    roadmap_k: int = 7
    budgets: str = ""
    w_values: str = "1.0,1.1"
    seed: int = 1234
    scale: str = "local"
    mode: str = "full"
    cpu: bool = False


def _is_field(backbone: str) -> bool:
    return backbone == "unet"


def now_str() -> str:
    return C.now_str()


# ---------------------------------------------------------------------------
# Task 3 — Matched-compute scalar LoRA trainer (bounded / unbounded)
# ---------------------------------------------------------------------------

def train_scalar_lora(backbone_cfg, dataset_npz, out_ckpt, feature_cfg, train_cfg, device,
                      seed, init_ckpt, rank, alpha, bounded: bool):
    """Matched-compute scalar LoRA fine-tune. bounded=True keeps the model's finite
    max_norm_residual clamp; bounded=False sets it to inf (no clamp)."""
    import torch.nn.functional as F
    out_ckpt = Path(out_ckpt)
    x, y = C.load_npz_arrays(dataset_npz)
    ds = C.ArrayDataset(x, y)
    loader = C.make_loader(ds, train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)
    model = C.build_model(backbone_cfg, feature_cfg, train_cfg, device)
    if init_ckpt is not None:
        C.safe_load_state(model, Path(init_ckpt))
    max_resid = float(train_cfg.max_norm_residual) if bounded else float("inf")
    model.max_norm_residual = max_resid
    C.apply_lora(model, rank=int(rank), alpha=float(alpha))
    C.set_lora_trainable(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    C.set_global_seed(int(seed))
    for _ in range(train_cfg.base_epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite c9h scalar-lora loss")
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, train_cfg.grad_clip)
            opt.step()
    payload = {"model": model.state_dict(), "backbone_cfg": asdict(backbone_cfg),
               "feature_cfg": asdict(feature_cfg), "train_cfg": asdict(train_cfg),
               "lora_rank": int(rank), "alpha": float(alpha), "bounded": bool(bounded),
               "max_norm_residual": max_resid}
    C.ensure_dir(out_ckpt.parent); torch.save(payload, out_ckpt)
    return out_ckpt


def load_scalar_provider_c9h(ckpt, device):
    """Load a scalar checkpoint (avgbase/full-FT/scratch/LoRA, bounded/unbounded) into a provider."""
    payload = torch.load(Path(ckpt), map_location="cpu")
    bb = C.BackboneConfig(**payload["backbone_cfg"]); fc = C.FeatureConfig(**payload["feature_cfg"])
    tc = C.TrainingConfig(**payload["train_cfg"])
    model = C.build_model(bb, fc, tc, device)
    if "lora_rank" in payload:
        C.apply_lora(model, rank=int(payload["lora_rank"]), alpha=float(payload["alpha"]))
    model.load_state_dict(payload["model"], strict=True)
    mnr = float(payload.get("max_norm_residual", tc.max_norm_residual))
    model.max_norm_residual = mnr
    model.eval()
    return P.ScalarResidualProvider(model, fc, device, bb.name, mnr)
