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
