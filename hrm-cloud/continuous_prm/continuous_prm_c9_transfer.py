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
import hashlib
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


# ---------------------------------------------------------------------------
# Task 2 — ADAPT/TEST world-split helpers
# ---------------------------------------------------------------------------

def world_fingerprint(world) -> tuple:
    """Stable identity of a world (start, goal, obstacle centers)."""
    start = tuple(np.round(np.asarray(world.start, dtype=np.float64), 6))
    goal = tuple(np.round(np.asarray(world.goal, dtype=np.float64), 6))
    obs = tuple(sorted((round(float(getattr(o, "cx", 0.0)), 6), round(float(getattr(o, "cy", 0.0)), 6)) for o in world.obstacles))
    return (round(float(world.side_len), 6), start, goal, obs)


def iter_test_worlds(spec, suite_idx: int, cfg: C9Config, roadmap_cfg, n_test: int):
    """TEST worlds for a target = the C7 deterministic eval worlds (yields (idx, world, rm))."""
    c7cfg = C7.C7Config(seed=int(cfg.seed), roadmap_nodes=cfg.roadmap_nodes, roadmap_k=cfg.roadmap_k)
    yield from C7.iter_matched_worlds(spec, suite_idx, c7cfg, roadmap_cfg, n_test)


def adapt_seed(target: str, K: int, adapt_seed_idx: int, base_seed: int) -> int:
    """Deterministic (cross-process stable), well-separated seed for an
    ADAPT(target, K, adapt_seed_idx) collection. Cross-process stability matters
    because collect_task_dataset caches by file path: a salted hash would let a
    resume reuse an npz built under a different seed."""
    h = int(hashlib.md5(target.encode()).hexdigest()[:4], 16)
    # 100_000*h % 7_000_000 is a coarse per-target offset; the actual separation
    # between collections comes from the 1_009*K + idx terms.
    return int(base_seed) + 5_000_000 + 100_000 * h % 7_000_000 + 1_009 * int(K) + int(adapt_seed_idx)


def adapt_world_fingerprints(spec, n_worlds, nodes_per_world, roadmap_cfg, feature_cfg, seed):
    """Replays collect_task_dataset's world-generation loop to expose ADAPT world fingerprints
    (for the disjointness test). Mirrors C.collect_task_dataset world acceptance rules."""
    import random as _random
    rng = _random.Random(int(seed))
    fps, done, attempts = [], 0, 0
    while done < n_worlds and attempts < n_worlds * 100:
        attempts += 1
        w_seed = rng.randint(0, 2**31 - 1)
        world = C.build_world(spec, w_seed, roadmap_cfg.min_start_goal_dist_frac)
        if world is None:
            continue
        rm = C.build_prm(world, roadmap_cfg, seed=w_seed + 17)
        if rm is None or not rm.connected_to_goal[0]:
            continue
        connected_idxs = np.where(rm.connected_to_goal)[0]
        connected_idxs = connected_idxs[connected_idxs != 1]
        if len(connected_idxs) < max(12, nodes_per_world // 4):
            continue
        # mirror collect_task_dataset's rng.sample so replayed worlds match the real ADAPT collection
        if nodes_per_world > 1:
            rng.sample([int(i) for i in connected_idxs], k=min(nodes_per_world - 1, len(connected_idxs)))
        # (collect_task_dataset's later finite-residual gate cannot drop connected
        # nodes, so it draws no rng and need not be replicated here.)
        fps.append(world_fingerprint(world))
        done += 1
    return fps
