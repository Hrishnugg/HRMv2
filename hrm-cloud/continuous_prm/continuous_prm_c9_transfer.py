#!/usr/bin/env python3
"""C9a: few-shot transfer learning for learned PRM heuristics.

Adapt the C7 pooled scalar base (avgbase) to held-out hard families from K worlds,
comparing zero-shot / LoRA / full fine-tune / from-scratch on HRM + ON-LSTM, and
report adaptation curves. New-file-only; reuses C7/C3 machinery. See
docs/superpowers/specs/2026-06-29-c9-transfer-design.md.
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import continuous_prm_common as C
import continuous_prm_providers as P
import continuous_prm_c7_integration_compare as C7
import continuous_prm_c7_hard_maps as H7


def now_str() -> str:
    return C.now_str()


@dataclass
class C9Config:
    source_dir: str = "hrm-cloud/continuous_prm/runs/c7_local"
    out_dir: str = "hrm-cloud/continuous_prm/runs/c9_local"
    targets: str = "C_hard_maze_dense,C_hard_bugtrap,C_hard_rooms_large"
    backbones: str = "hrm,onlstm"
    k_grid: str = "0,1,2,4,8,16,32"
    n_adapt_seeds: int = 5
    n_test: int = 30
    alpha: float = 1.0
    adapt_epochs: int = 0
    roadmap_nodes: int = 192
    roadmap_k: int = 7
    budgets: str = ""
    w_values: str = "1.0,1.1"
    seed: int = 1234
    scale: str = "local"
    mode: str = "full"
    cpu: bool = False


@dataclass
class SourceBase:
    model: object
    backbone_cfg: object
    feature_cfg: object
    train_cfg: object
    ckpt_path: Path
    backbone: str


def load_source_base(source_dir, backbone: str, device) -> SourceBase:
    """Load the C7 avgbase checkpoint + its configs for `backbone`."""
    import torch
    ckpt = Path(source_dir) / "checkpoints" / f"avgbase__{backbone}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"source base not found: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu")
    backbone_cfg = C.BackboneConfig(**payload["backbone_cfg"])
    feature_cfg = C.FeatureConfig(**payload["feature_cfg"])
    train_cfg = C.TrainingConfig(**payload["train_cfg"])
    model = C.load_base_model(backbone_cfg, feature_cfg, train_cfg, ckpt, device)
    return SourceBase(model, backbone_cfg, feature_cfg, train_cfg, ckpt, backbone)
