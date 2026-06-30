#!/usr/bin/env python3
"""C9b: few-shot transfer under dynamics. Adapt frozen C8 pooled space-time
heuristics (aware + blind) to held-out dynamic families via zero_shot/LoRA/full_ft/scratch.
New-file-only; reuses C8 + C9/C9h + common. See docs/superpowers/specs/2026-06-30-c9b-dynamics-transfer-design.md.
"""
from __future__ import annotations
import argparse, dataclasses, csv as _csv, json, hashlib
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


def _valid_world_seed(suite: str, seed: int, cfg: C9bConfig) -> bool:
    try:
        lab = M8._collect_world_labels(suite, int(seed), _c8cfg(cfg))
    except Exception:
        return False
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
