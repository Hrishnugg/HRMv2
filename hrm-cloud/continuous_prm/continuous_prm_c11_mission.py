#!/usr/bin/env python3
"""C11 compositional-mission PRM heuristics -- Task 1: module skeleton,
config, and TRAIN/TEST dataset builders.

Builds the phase's dataset layer on top of the FROZEN G0-H headroom probe
module `continuous_prm_c11_headroom.py`: the (config, K) cell grid, the
TRAIN/TEST world-seed streams, and per-world bundles carrying the exact
oracle field plus admissibility-clipped residual-over-legsum targets. This
module never modifies the probe -- it only imports `sample_mission`,
`product_oracle`, `h_legsum`, `mission_reachable`, and
`door_adj_valid_factory` from it, mirroring `eval_cell`'s own world-build/
skip loop exactly (see `continuous_prm_c11_headroom.eval_cell` for the
canonical version of this loop).

See docs/superpowers/specs/2026-07-07-c11-compositional-mission-design.md
and docs/superpowers/plans/2026-07-07-c11-mission.md (Task 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import continuous_prm_common as C
import continuous_prm_c7_hard_maps as H7
import continuous_prm_c11_headroom as C11P

H7.install_c7_hard_maps()

ROADMAP_CFG = C.RoadmapConfig(n_nodes=192, k_neighbors=7)


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------

@dataclass
class C11MissionConfig:
    """All C11-mission-phase constants pinned as fields (spec section 2/4;
    plan's "Shared core definitions"). Nothing here is tuned post hoc -- the
    binding budgets and recipe fields are the plan's pre-registered values."""

    k_values: Tuple[int, ...] = (0, 2, 4, 8)
    k_max: int = 8

    n_train_worlds: int = 40
    n_test_worlds: int = 25
    max_world_attempts: int = 300

    budgets_grid: Tuple[int, ...] = (100, 200, 400, 800, 1600, 3200)
    # Probe-calibrated binding budgets for K in {2, 4, 8}, keyed
    # (config_label, K). K=0 cells (A, B) are calibrated at collect time
    # against `budgets_grid` (Task 4+) -- not pre-registered here.
    binding_budgets: Dict[Tuple[str, int], int] = field(default_factory=lambda: {
        ("A", 2): 200, ("A", 4): 400, ("A", 8): 1600,
        ("B", 2): 100, ("B", 4): 200, ("B", 8): 800,
        ("C", 2): 200, ("C", 4): 400, ("C", 8): 1600,
    })

    residual_cap: float = 4.0

    # Matched training recipe (arms 1-4; deviations are bugs).
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 1024
    epochs: int = 40
    smooth_l1_beta: float = 1.0

    # I/O contract knobs (encoders, Task 2+).
    token_dim: int = 12
    seq_max: int = 10
    grid_channels: int = 5
    grid_size: int = 64
    gnn_hidden: int = 128
    gnn_rounds: int = 8
    mlp_width: int = 768
    film_dim: int = 32

    train_seeds: Tuple[int, ...] = (0, 1, 2)


# ---------------------------------------------------------------------------
# Cell grid.
# ---------------------------------------------------------------------------

# (config_label, spec_name, config_idx, doors) -- pre-registered, matches the
# probe's own config_idx assignment (config C's config_idx=2 is pinned by the
# probe's door tests; never renumber).
_CONFIG_DEFS: Tuple[Tuple[str, str, int, bool], ...] = (
    ("A", "C_hard_maze", 0, False),
    ("B", "C_hard_rooms_large", 1, False),
    ("C", "C_hard_maze", 2, True),
)


def build_cell_grid(cfg: Optional[C11MissionConfig] = None) -> List[dict]:
    """The 11-cell (config, K) grid: A and B at K in (0, 2, 4, 8), C at K in
    (2, 4, 8) only (K=0 with doors degenerates to config A's plain-mission
    distribution -- doors have nothing to gate with zero waypoints -- so C is
    dropped at K=0 per the spec/plan). Order: A K0,2,4,8; B K0,2,4,8; C
    K2,4,8."""
    cfg = cfg or C11MissionConfig()
    cells: List[dict] = []
    for config_label, spec_name, config_idx, doors in _CONFIG_DEFS:
        for K in cfg.k_values:
            if config_label == "C" and K == 0:
                continue
            cells.append({
                "config_label": config_label,
                "spec_name": spec_name,
                "config_idx": config_idx,
                "K": K,
                "doors": doors,
            })
    return cells


# ---------------------------------------------------------------------------
# Seed formulas.
# ---------------------------------------------------------------------------

def test_seed(w: int, config_idx: int, K: int) -> int:
    """The probe's exact TEST-world seed formula (`eval_cell`'s formula) --
    K in {2, 4, 8} TEST cells are the probe's exact 25 worlds/cell."""
    return 1234 + 7919 * w + 104729 * config_idx + 15485863 * K


def train_seed(w: int, config_idx: int, K: int) -> int:
    """The disjoint TRAIN-world seed stream (same shape, different additive
    constant -- 900001 instead of 1234)."""
    return 900001 + 7919 * w + 104729 * config_idx + 15485863 * K


# ---------------------------------------------------------------------------
# Per-world bundle.
# ---------------------------------------------------------------------------

@dataclass
class WorldBundle:
    """Everything collected for one valid (cell, seed) world: the built
    world/roadmap/mission, the exact oracle field, the door adjacency
    predicate (None for configs A/B), the legsum callable, and the
    admissibility-clipped residual-over-legsum training targets."""

    world: C.World
    rm: C.Roadmap
    wp: Tuple[int, ...]
    oracle: np.ndarray
    adj_valid: Optional[Callable[[int, int, int], bool]]
    hl: Callable[[int, int], float]
    targets: np.ndarray  # (n_finite, 3) float64 columns (i, s, y)
    preclip_min: float
    seed: int


def collect_world_bundle(cell: dict, seed: int, cfg: Optional[C11MissionConfig] = None) -> WorldBundle:
    """Build one world for `cell` at `seed` and compute its oracle field +
    residual targets, raising RuntimeError on any skip condition (single
    skip-signal convention, mirroring `continuous_prm_c11_headroom.eval_cell`'s
    own world-build loop exactly):

      - `C.build_world` / `C.build_prm` returning None.
      - `C11P.sample_mission` raising RuntimeError (K=0 never calls this --
        the K=0 mission is the empty tuple by construction, so there is
        nothing to sample and no RuntimeError path here).
      - the doors adjacency factory (`C11P.door_adj_valid_factory`) raising
        RuntimeError, or returning None.
      - `not C11P.mission_reachable(oracle)`.
    """
    cfg = cfg or C11MissionConfig()
    spec_name = cell["spec_name"]
    config_idx = cell["config_idx"]
    K = cell["K"]
    doors = cell["doors"]

    specs = C.build_anchor_specs()
    spec = specs[spec_name]

    world = C.build_world(spec, seed, 0.45)
    if world is None:
        raise RuntimeError(f"build_world returned None (cell={cell}, seed={seed})")
    rm = C.build_prm(world, ROADMAP_CFG, seed + 17)
    if rm is None:
        raise RuntimeError(f"build_prm returned None (cell={cell}, seed={seed})")

    if K == 0:
        wp: Tuple[int, ...] = ()
    else:
        wp = tuple(C11P.sample_mission(rm, world, K, seed))
        # sample_mission raises RuntimeError on its own failure path; that
        # propagates uncaught, which is exactly the desired skip signal.

    adj_valid = None
    if doors:
        adj_valid = C11P.door_adj_valid_factory(rm, world, wp, seed)
        if adj_valid is None:
            raise RuntimeError(f"door_adj_valid_factory returned None (cell={cell}, seed={seed})")

    oracle = C11P.product_oracle(rm, wp, adj_valid)
    if not C11P.mission_reachable(oracle):
        raise RuntimeError(f"mission not reachable (cell={cell}, seed={seed})")

    hl = C11P.h_legsum(rm, wp)

    side_len = float(world.side_len)
    N = rm.points.shape[0]
    Kk = len(wp)

    finite_i, finite_s, ys, preclip_min = _residual_targets(oracle, hl, side_len, N, Kk, cfg.residual_cap)

    targets = np.stack([finite_i, finite_s, ys], axis=1).astype(np.float64)

    return WorldBundle(
        world=world, rm=rm, wp=wp, oracle=oracle, adj_valid=adj_valid, hl=hl,
        targets=targets, preclip_min=preclip_min, seed=seed,
    )


def _residual_targets(oracle: np.ndarray, hl: Callable[[int, int], float], side_len: float,
                       N: int, K: int, cap: float):
    """Compute the admissibility-clipped residual-over-legsum targets for
    every finite-oracle (i, s) state. `y = clip((oracle[s,i] - hl(i,s)) /
    side_len, 0, cap)`; pre-clip values must be >= -1e-9 (admissibility:
    h_legsum <= h*), asserted here and also surfaced as `preclip_min` so
    callers/tests can check it directly rather than re-deriving it."""
    finite_i: List[int] = []
    finite_s: List[int] = []
    ys: List[float] = []
    preclip_min = float("inf")

    for s in range(K + 1):
        for i in range(N):
            h_star = oracle[s, i]
            if h_star >= C.INF / 10.0:
                continue
            preclip = (float(h_star) - hl(i, s)) / side_len
            if preclip < preclip_min:
                preclip_min = preclip
            assert preclip >= -1e-9, (
                f"admissibility violated: h_legsum > h* at (i={i}, s={s}): "
                f"h*={h_star}, preclip_residual={preclip}"
            )
            y = float(np.clip(preclip, 0.0, cap))
            finite_i.append(i)
            finite_s.append(s)
            ys.append(y)

    if preclip_min == float("inf"):
        # No finite states at all -- product_oracle's own mission_reachable
        # check should have already raised before this is ever reached (the
        # start state (0,0) being finite implies at least one finite entry),
        # but guard against a degenerate 0-node roadmap defensively.
        preclip_min = 0.0

    return (
        np.asarray(finite_i, dtype=np.float64),
        np.asarray(finite_s, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        preclip_min,
    )


# ---------------------------------------------------------------------------
# Per-cell dataset collection.
# ---------------------------------------------------------------------------

def collect_cell_dataset(cell: dict, split: str, n_worlds: int,
                          cfg: Optional[C11MissionConfig] = None) -> List[WorldBundle]:
    """Collect `n_worlds` valid `WorldBundle`s for `cell` from the given
    split's seed stream, looping attempt index w = 0, 1, 2, ... and catching
    per-world RuntimeError as a skip (same convention as
    `continuous_prm_c11_headroom.eval_cell`'s world loop). Capped at
    `cfg.max_world_attempts` attempts, after which a RuntimeError is raised."""
    cfg = cfg or C11MissionConfig()
    if split == "train":
        seed_fn = train_seed
    elif split == "test":
        seed_fn = test_seed
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    config_idx = cell["config_idx"]
    K = cell["K"]

    bundles: List[WorldBundle] = []
    w = 0
    attempts = 0
    while len(bundles) < n_worlds:
        if attempts >= cfg.max_world_attempts:
            raise RuntimeError(
                f"exhausted {cfg.max_world_attempts} attempts collecting {split} worlds for "
                f"cell={cell}: only found {len(bundles)}/{n_worlds}"
            )
        seed = seed_fn(w, config_idx, K)
        attempts += 1
        w += 1
        try:
            bundle = collect_world_bundle(cell, seed, cfg)
        except RuntimeError:
            continue
        bundles.append(bundle)

    return bundles
