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
