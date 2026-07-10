#!/usr/bin/env python3
"""C11 compositional-mission PRM heuristics -- Task 1: module skeleton,
config, and TRAIN/TEST dataset builders. Task 2: structure-exposing
encoders (trace tokens, MLP flatten, field grids, product graph). Task 3:
the four native arm models (MLP control, FiLM U-Net, product-graph GNN,
HRM/ON-LSTM trace) + the matched trainer + per-arm field prediction.

Builds the phase's dataset layer on top of the FROZEN G0-H headroom probe
module `continuous_prm_c11_headroom.py`: the (config, K) cell grid, the
TRAIN/TEST world-seed streams, and per-world bundles carrying the exact
oracle field plus admissibility-clipped residual-over-legsum targets. This
module never modifies the probe -- it only imports `sample_mission`,
`product_oracle`, `h_legsum`, `mission_reachable`, and
`door_adj_valid_factory` from it, mirroring `eval_cell`'s own world-build/
skip loop exactly (see `continuous_prm_c11_headroom.eval_cell` for the
canonical version of this loop).

Task 2 adds the encoder layer: every arm consumes the SAME underlying
information (query-node geometry + the chained mission-leg trace), encoded
natively for its architecture class -- a padded token sequence for the
trace-sequence arms (`encode_trace` / `encode_trace_padded`), a flattened
vector for the MLP control (`encode_mlp`), rasterized grids for the field
U-Net (`encode_field_grids`, reusing `continuous_prm_c6_heatmap_value_field`'s
occupancy/gaussian rasterizer by import), and a product-graph tensor triple
for the GNN (`encode_product_graph`).

Task 3 adds `build_arm` (5 constructors: `mlp`, `unet_film`, `gnn`,
`hrm_trace`, `onlstm_trace` -- all within the [0.5M, 3.5M] param band),
`train_arm` (the matched recipe: smooth-L1, AdamW, grad-clip, identical
across arms 1-4 by construction -- deviations are bugs), and
`predict_field` (batch every product state through an arm's encoder+model,
deterministic, no_grad). Every arm's output goes through the SAME
`clamp(softplus(raw), 0, 4)` convention as `C.ContinuousHeuristicModel.forward`
(the trace arms reuse that class directly, so they get the clamp for free;
the other three apply it once, explicitly, in their `forward`).

Task 4 adds the provider registry (`PROVIDER_BUILDERS` /
`register_provider_builder` / `make_provider` -- the hook the eventual T7
HRM-v2 module registers into post-import) and the `train`/`eval` modes:
`run_train` (per (arm, cell, seed) checkpoint written ATOMICALLY
(tmp+`os.replace`) + a load-merge-write manifest, resumable via a
`Path.exists()` check -- the convention `continuous_prm_c9_transfer.
run_adapt` uses -- with a self-heal path for the crash window: a ckpt
whose manifest entry is missing gets its entry rebuilt from the ckpt's own
stored meta, and an unreadable ckpt is moved aside and retrained) and
`run_eval` (reference arms `h_next`/`h_legsum`/`h_oracle` PLUS every
manifest-registered learned arm, all run through the identical
`C11P.astar_product` path on the SAME probe-native TEST bundles, budgets
from `cfg.budgets_grid`, K=0 binding-budget calibration via
`C11P.calibrate_binding_budget` stored in `binding_k0.json`). The registry
carries the FULL per-arm contract -- `build(cfg)` construction AND
`forward_batch(...)` inference -- so an externally-registered arm works
end-to-end through `build_arm`/`make_provider`/`predict_field`/`run_eval`
(T4-review Important 1). A thin CLI stub (`--mode train|eval`) rounds out
the module; `write_json`/`write_csv` are `continuous_prm_common`'s
existing atomic (tmp+`os.replace`) helpers, reused rather than
reimplemented.

See docs/experiments/continuous/c11/design/2026-07-07-c11-compositional-mission-design.md
(sections 3/4/6/7 are authoritative on the I/O contract, matched recipe,
and eval/stats conventions) and docs/experiments/continuous/c11/plans/2026-07-07-c11-mission.md
(Tasks 1-4).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import continuous_prm_common as C
import continuous_prm_c6_heatmap_value_field as C6
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
    admissibility-clipped residual-over-legsum training targets.

    `node_rays` / `field_grids` are lazily-computed per-bundle caches
    (store-on-bundle pattern: the cache's lifetime is exactly the bundle's
    own, so it can never be handed to a different bundle -- unlike a
    module-global dict keyed by `id(bundle)`, whose entries outlive their
    bundle and get inherited by a NEW bundle when the allocator reuses the
    freed address). Filled by `_node_rays` ((N, 8) ray distances) and
    `_bundle_grids` ((K+1, grid_channels, grid_size, grid_size) field-grid
    stack); pure cost caches, value-transparent by construction and pinned
    by `test_ray_cache_consistency` / `test_bundle_grids_value_transparency`."""

    world: C.World
    rm: C.Roadmap
    wp: Tuple[int, ...]
    oracle: np.ndarray
    adj_valid: Optional[Callable[[int, int, int], bool]]
    hl: Callable[[int, int], float]
    targets: np.ndarray  # (n_finite, 3) float64 columns (i, s, y)
    preclip_min: float
    seed: int
    node_rays: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    field_grids: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


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
        # `door_adj_valid_factory` returns `placement.adj_valid` (a bound
        # method -- see its own docstring); `__self__` recovers the
        # `DoorPlacement` without threading a second return value through
        # this function's signature (the same convention already used by
        # `encode_field_grids`/tests to reach `placement.rects`). Data-
        # integrity guard, not a bare `assert` (must survive `python -O`):
        # config C is pinned to exactly 2 keys->doors everywhere else in
        # this module (`_DOOR_KEY_LEGS = {0, 1}`), so a placement with a
        # different door count would silently desync the trace-token
        # door-key/door-open flags from the actual blocked-edge set.
        placement = adj_valid.__self__
        if len(placement.blocked) != 2:
            raise AssertionError(
                f"expected exactly 2 doors (DoorPlacement.blocked), got "
                f"{len(placement.blocked)} (cell={cell}, seed={seed})"
            )

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
            if preclip < -1e-9:
                # Bare `assert` is stripped under `python -O`; this check is a
                # data-integrity guard that must survive optimized runs.
                raise AssertionError(
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


def stack_targets(bundles: Sequence[WorldBundle]) -> np.ndarray:
    """Vstack every bundle's `targets` (n_b, 3) columns (i, s, y) into one
    (sum(n_b), 4) array, appending a 4th column = the bundle's index in
    `bundles` (int, cast to float64 -- matches `targets`' own dtype so the
    stack is a plain concatenation, no per-column dtype promotion)."""
    parts = []
    for b_idx, bundle in enumerate(bundles):
        n = bundle.targets.shape[0]
        idx_col = np.full((n, 1), float(b_idx), dtype=np.float64)
        parts.append(np.concatenate([bundle.targets, idx_col], axis=1))
    if not parts:
        return np.zeros((0, 4), dtype=np.float64)
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Task 2: structure-exposing encoders.
# ---------------------------------------------------------------------------
#
# Every arm sees IDENTICAL underlying information -- query-node local
# geometry (position + 8 coarse ray distances) plus the chained mission-leg
# trace (the same leg-length decomposition `C11P.h_legsum` sums into its
# admissible heuristic) -- encoded natively per architecture class:
#   - trace-sequence arms (HRM-trace / ON-LSTM-trace / HRM-v2):
#     `encode_trace` / `encode_trace_padded` (a variable-length token
#     sequence, query token first, then one token per remaining leg).
#   - MLP control: `encode_mlp` (the padded sequence, flattened -- the
#     control arm gets no sequence/graph structure, only the flat bag).
#   - field U-Net: `encode_field_grids` (rasterized occupancy/gaussian
#     grids, reusing `continuous_prm_c6_heatmap_value_field`'s rasterizer).
#   - product-graph GNN: `encode_product_graph` (node/edge tensors over the
#     product graph (node, stage), reusing the same query-token geometry as
#     node features and the probe's own `transition_stage`/adjacency rules
#     for edges).
#
# TRACE_TOKEN_LAYOUT documents the 12 slots of every trace token (query AND
# leg tokens share one dim-12 layout; unused leg slots are zero-padded, and
# the query token's slots 10-11 have no leg-token analog -- see each
# function's docstring for the exact per-token-kind semantics).
TRACE_TOKEN_LAYOUT: Tuple[str, ...] = (
    "x_over_side_OR_dx_over_side",       # query: x/side.      leg: dx/side.
    "y_over_side_OR_dy_over_side",       # query: y/side.      leg: dy/side.
    "ray0_OR_dist_over_side",            # query: ray(dir 0).  leg: leg length/side.
    "ray2_OR_t_minus_s_over_kmax",       # query: ray(dir 2).  leg: (t-s)/k_max.
    "ray4_OR_is_door_key",               # query: ray(dir 4).  leg: is_door_key.
    "ray6_OR_door_open_at_s",            # query: ray(dir 6).  leg: door_open_at_s.
    "ray8_OR_remaining_frac",            # query: ray(dir 8).  leg: (K-t)/(K+1).
    "ray10_OR_is_goal_leg",              # query: ray(dir 10). leg: is_goal_leg.
    "ray12_OR_pad",                      # query: ray(dir 12). leg: 0.
    "ray14_OR_pad",                      # query: ray(dir 14). leg: 0.
    "s_over_kmax_OR_pad",                # query: s/k_max.     leg: 0.
    "k_remaining_over_kmax_OR_pad",      # query: (K-s)/k_max. leg: 0.
)

# Ray directions sampled: every OTHER compass direction (0, 2, ..., 14) of
# the existing 16-direction ray extractor (`C.FeatureConfig().num_rays`),
# at angle `2*pi*d/16`.
_RAY_DIRS: Tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14)

# Config C's 2 doors are keyed to waypoints 0 and 1 (`C11ProbeConfig.n_doors
# == 2`; door d's key is completing waypoint d -- see
# `continuous_prm_c11_headroom`'s door-placement docstring). Pinned here
# rather than re-derived from `DoorPlacement` so `encode_trace` needs only
# `bundle.adj_valid is not None` (config C's marker) plus this constant, no
# reach into the door-placement's internals.
_DOOR_KEY_LEGS: frozenset = frozenset({0, 1})

# T2-review micro-fix, amended per T3 review: per-node ray memo.
# `_query_token` is called once per query row in `encode_trace` and once per
# (node, stage) in `encode_product_graph` -- without memoizing, the same
# node's 8 rays get re-raycast on every call (encode_product_graph alone
# calls it N times per bundle; a training loop calls it thousands of times).
# The memo lives ON the bundle (`WorldBundle.node_rays`, store-on-bundle
# pattern) so its lifetime is exactly the bundle's own. The earlier
# module-global dict keyed by `id(bundle)` was unsound: `id()` values are
# reused after garbage collection, so a fresh bundle allocated at a dead
# bundle's address inherited the dead bundle's rays (silent corruption --
# demonstrated in the T3 review as 22/40 fresh bundles receiving stale rays
# under the training loop's per-cell bundle lifecycle). Purely a cost
# optimization -- every cached value is byte-identical to what
# `C.raycast_distance` returns directly (pinned by
# `test_ray_cache_consistency`).

def _node_rays(bundle: "WorldBundle") -> np.ndarray:
    """The bundle's `(N, 8)` per-node ray-distance array (already /side),
    computed once on first use and stored on the bundle itself
    (`bundle.node_rays`)."""
    if bundle.node_rays is not None:
        return bundle.node_rays
    world = bundle.world
    side = float(world.side_len)
    steps = C.FeatureConfig().ray_steps
    points = bundle.rm.points
    N = points.shape[0]
    rays = np.zeros((N, len(_RAY_DIRS)), dtype=np.float32)
    for i in range(N):
        p = points[i]
        for k, d in enumerate(_RAY_DIRS):
            angle = 2.0 * math.pi * d / 16.0
            rays[i, k] = C.raycast_distance(p, angle, world, steps=steps) / side
    bundle.node_rays = rays
    return rays


def _query_token(p: np.ndarray, world: C.World, s: int, K: int, cfg: C11MissionConfig,
                  rays: Optional[np.ndarray] = None) -> np.ndarray:
    """The dim-12 query-node token: `[x/side, y/side, 8 coarse ray
    distances, s/k_max, K_remaining/k_max]`. `rays`, if given, is the
    node's precomputed 8-ray row (from `_node_rays`) -- skips raycasting
    entirely. If omitted, rays are computed directly from `p` (used by call
    sites that don't have a roadmap node index / bundle, e.g. none currently,
    kept for API compatibility with earlier direct-point callers)."""
    side = float(world.side_len)
    tok = np.zeros(12, dtype=np.float32)
    tok[0] = p[0] / side
    tok[1] = p[1] / side
    if rays is not None:
        tok[2:10] = rays
    else:
        steps = C.FeatureConfig().ray_steps
        for k, d in enumerate(_RAY_DIRS):
            angle = 2.0 * math.pi * d / 16.0
            tok[2 + k] = C.raycast_distance(p, angle, world, steps=steps) / side
    tok[10] = s / float(cfg.k_max)
    tok[11] = (K - s) / float(cfg.k_max)
    return tok


def _leg_token(prev_p: np.ndarray, tgt_p: np.ndarray, t: int, s: int, K: int,
                is_doors_config: bool, side: float, cfg: C11MissionConfig) -> np.ndarray:
    """The dim-12 leg token for leg `t` (t in s..K), chained from `prev_p`
    (the previous node's/leg's position) to `tgt_p` (leg t's target
    position). Layout: `[dx/side, dy/side, dist/side, (t-s)/k_max,
    is_door_key, door_open_at_s, remaining_frac, is_goal_leg, 0, 0, 0, 0]`."""
    dx = (tgt_p[0] - prev_p[0]) / side
    dy = (tgt_p[1] - prev_p[1]) / side
    dist = math.hypot(dx, dy)
    is_door_key = 1.0 if (is_doors_config and t < K and t in _DOOR_KEY_LEGS) else 0.0
    door_open_at_s = 1.0 if (is_door_key and s > t) else 0.0
    remaining_frac = (K - t) / float(K + 1)
    is_goal_leg = 1.0 if t == K else 0.0

    tok = np.zeros(12, dtype=np.float32)
    tok[0] = dx
    tok[1] = dy
    tok[2] = dist
    tok[3] = (t - s) / float(cfg.k_max)
    tok[4] = is_door_key
    tok[5] = door_open_at_s
    tok[6] = remaining_frac
    tok[7] = is_goal_leg
    # slots 8-11 stay zero (leg tokens use only 8 of the 12 slots).
    return tok


def encode_trace(bundle: WorldBundle, i: int, s: int,
                  cfg: Optional[C11MissionConfig] = None) -> np.ndarray:
    """The real mission trace for product state (i, s): `[query token] +
    [one leg token per remaining leg t = s..K]` (length `1 + (K - s + 1)`).
    Leg t < K targets `wp[t]`; the LAST leg (t == K) targets the goal (node
    1). Legs CHAIN geometrically exactly like `C11P.h_legsum`'s own
    leg-length decomposition: the first leg's `prev` is the query node i's
    own position; every subsequent leg's `prev` is the PREVIOUS leg's
    target position (not node i's position again) -- e.g. at s=0 the trace
    is `i -> wp[0] -> wp[1] -> ... -> wp[K-1] -> goal`, matching the chain
    `h_legsum` sums leg lengths over. Pure, deterministic, float32."""
    cfg = cfg or C11MissionConfig()
    world = bundle.world
    side = float(world.side_len)
    wp = bundle.wp
    K = len(wp)
    is_doors_config = bundle.adj_valid is not None

    p_i = bundle.rm.points[i]
    node_rays = _node_rays(bundle)[i]
    tokens = [_query_token(p_i, world, s, K, cfg, rays=node_rays)]

    prev_p = p_i
    for t in range(s, K + 1):
        tgt_node = wp[t] if t < K else 1
        tgt_p = bundle.rm.points[tgt_node]
        tokens.append(_leg_token(prev_p, tgt_p, t, s, K, is_doors_config, side, cfg))
        prev_p = tgt_p

    return np.stack(tokens, axis=0).astype(np.float32)


def encode_trace_padded(bundle: WorldBundle, states: Iterable[Tuple[int, int]],
                         cfg: Optional[C11MissionConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Batch + pad `encode_trace` over `states` (an iterable of (i, s)
    pairs) to a fixed `(n, cfg.seq_max, cfg.token_dim)` tensor plus a
    `(n, cfg.seq_max)` bool mask (True on real tokens, False on padding).
    Padding rows are all-zero."""
    cfg = cfg or C11MissionConfig()
    states = list(states)
    n = len(states)
    tokens = np.zeros((n, cfg.seq_max, cfg.token_dim), dtype=np.float32)
    mask = np.zeros((n, cfg.seq_max), dtype=np.bool_)
    for row, (i, s) in enumerate(states):
        seq = encode_trace(bundle, i, s, cfg)
        L = seq.shape[0]
        tokens[row, :L] = seq
        mask[row, :L] = True
    return tokens, mask


def encode_mlp(bundle: WorldBundle, states: Iterable[Tuple[int, int]],
               cfg: Optional[C11MissionConfig] = None) -> np.ndarray:
    """The MLP control's input: the padded trace sequence, flattened to
    `(n, seq_max * token_dim)`. Implemented AS a call to
    `encode_trace_padded` followed by `.reshape` (byte-identical to it by
    construction -- the control arm gets no sequence structure, only the
    flat concatenation of the same tokens every other arm sees)."""
    cfg = cfg or C11MissionConfig()
    padded, _mask = encode_trace_padded(bundle, states, cfg)
    n = padded.shape[0]
    return padded.reshape(n, -1)


def encode_field_grids(bundle: WorldBundle, s: int,
                        cfg: Optional[C11MissionConfig] = None) -> np.ndarray:
    """The field U-Net's per-stage input: `(5, grid_size, grid_size)`
    float32 grids `[occupancy, start gaussian, goal gaussian, current-target
    gaussian at tgt(s), closed-door mask at stage s]`.

    Reuses `continuous_prm_c6_heatmap_value_field`'s rasterizer by IMPORT
    (module-level pure functions: `rasterize_world` for occupancy,
    `gaussian_channel` for the point-splat channels, `point_to_cell` for the
    door-rect rasterization) rather than replicating the math -- both
    functions are importable and side-effect-free.

    Grid coordinate convention (matches the C6 node gather EXACTLY, verified
    against `point_to_cell`/`interpolate_grid_values`): array axis 0 is the
    X cell index, axis 1 is the Y cell index -- i.e. `grid[ix, iy]`, NOT
    `grid[row=y, col=x]`. `point_to_cell(p, side, n)` returns `(ix, iy) =
    (floor(p.x/side*n), floor(p.y/side*n))`, and `gaussian_channel` builds
    its grid from `grid_centers` (`np.meshgrid(xs, ys, indexing="ij")`,
    i.e. `xx[ix, iy]` varies along axis 0), so every channel here is
    x-major/y-minor and a point's rasterized location is always
    `grid[point_to_cell(p, side, n)]`.
    """
    cfg = cfg or C11MissionConfig()
    world = bundle.world
    n = cfg.grid_size
    side = float(world.side_len)
    K = len(bundle.wp)

    occupancy, _free, _clearance = C6.rasterize_world(world, n)
    start_g = C6.gaussian_channel(world, world.start, n)
    goal_g = C6.gaussian_channel(world, world.goal, n)

    tgt_node = C11P.mission_target(s, bundle.wp)
    tgt_p = bundle.rm.points[tgt_node]
    target_g = C6.gaussian_channel(world, tgt_p, n)

    door_mask = np.zeros((n, n), dtype=np.float32)
    if bundle.adj_valid is not None:
        placement = bundle.adj_valid.__self__  # DoorPlacement (see WorldBundle
        # construction in `collect_world_bundle`: `door_adj_valid_factory`
        # always returns `placement.adj_valid`, a bound method, so
        # `__self__` recovers the DoorPlacement -- a public dataclass of the
        # frozen probe module, not a private/internal reach-through).
        for d, rect in enumerate(placement.rects):
            if s <= d:
                _rasterize_rect(door_mask, rect, side, n)

    grids = np.stack([occupancy, start_g, goal_g, target_g, door_mask], axis=0)
    return grids.astype(np.float32)


def _rasterize_rect(grid: np.ndarray, rect: Tuple[float, float, float, float],
                     side: float, n: int) -> None:
    """Set `grid[ix, iy] = 1.0` for every cell whose CENTER falls inside
    axis-aligned `rect = (xmin, ymin, xmax, ymax)`. In-place; same
    `point_to_cell` cell-center convention as the rest of this module's
    grids (`C6.grid_centers`: cell (ix, iy)'s center is at `(ix+0.5, iy+0.5)
    * side/n`)."""
    xmin, ymin, xmax, ymax = rect
    xx, yy = C6.grid_centers(side, n)
    inside = (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)
    grid[inside] = 1.0


def encode_product_graph(bundle: WorldBundle,
                          cfg: Optional[C11MissionConfig] = None) -> Dict[str, np.ndarray]:
    """The GNN's product-graph tensors: `node_feats` ((K+1)*N, 14),
    `edge_index` (2, E) int64, `edge_feats` (E, 3).

    Nodes: every product state (i, s), flat id `s * N + i` (N = 192 roadmap
    nodes) -- `node_feats[s*N + i]` = the query-node token for (i, s) (12
    dims, `_query_token`) concatenated with `[dist(p_i, p_tgt(s))/side,
    s/k_max]` (2 dims). Rays are computed ONCE per node i and reused across
    every stage s (ray geometry is stage-independent -- only the trailing 2
    dims and the query token's own s-dependent slots 10-11 vary by stage).

    Edges: for every roadmap edge (i, j, w) in `bundle.rm.adj` and every
    stage s where `bundle.adj_valid(i, j, s)` holds (always True for
    configs A/B, since `bundle.adj_valid is None` there), a directed product
    edge from (i, s) to (j, s2) with `s2 = C11P.transition_stage(s, j,
    bundle.wp)`. BOTH directions of each undirected roadmap edge are
    processed independently (`rm.adj[i]` already lists both directions), so
    each undirected roadmap edge yields up to 2 product edges per stage.
    `edge_feats = [w/side, is_arrival (1.0 iff s2 != s), 1.0]`.

    Pure and deterministic: two calls on the same bundle produce identical
    tensors (no RNG, no dict/set iteration order dependence -- edges are
    built by iterating `range(N)` and `rm.adj[i]` in their stored order)."""
    cfg = cfg or C11MissionConfig()
    world = bundle.world
    side = float(world.side_len)
    rm = bundle.rm
    wp = bundle.wp
    K = len(wp)
    N = rm.points.shape[0]

    # Rays computed once per node (via the bundle-level cache), reused
    # across every stage.
    node_rays = _node_rays(bundle)
    query_toks = np.stack(
        [_query_token(rm.points[i], world, 0, K, cfg, rays=node_rays[i]) for i in range(N)], axis=0
    )  # (N, 12); slots 10-11 (s/k_max, K_remaining/k_max) are stage-0 values
       # here and OVERWRITTEN per stage below (they are the only
       # stage-dependent slots in the query-token layout).

    node_feats = np.zeros(((K + 1) * N, 14), dtype=np.float32)
    for s in range(K + 1):
        tgt_node = C11P.mission_target(s, wp)
        tgt_p = rm.points[tgt_node]
        dist_tgt = np.linalg.norm(rm.points - tgt_p[None, :], axis=1) / side  # (N,)
        block = query_toks.copy()
        block[:, 10] = s / float(cfg.k_max)
        block[:, 11] = (K - s) / float(cfg.k_max)
        extra = np.stack([dist_tgt.astype(np.float32), np.full(N, s / float(cfg.k_max), dtype=np.float32)], axis=1)
        node_feats[s * N:(s + 1) * N] = np.concatenate([block, extra], axis=1)

    src_list: List[int] = []
    dst_list: List[int] = []
    w_list: List[float] = []
    arrival_list: List[float] = []

    for s in range(K + 1):
        for i in range(N):
            for j, w in rm.adj[i]:
                if bundle.adj_valid is not None and not bundle.adj_valid(i, j, s):
                    continue
                s2 = C11P.transition_stage(s, j, wp)
                src_list.append(s * N + i)
                dst_list.append(s2 * N + j)
                w_list.append(w / side)
                arrival_list.append(1.0 if s2 != s else 0.0)

    E = len(src_list)
    edge_index = np.zeros((2, E), dtype=np.int64)
    edge_index[0] = np.asarray(src_list, dtype=np.int64)
    edge_index[1] = np.asarray(dst_list, dtype=np.int64)

    edge_feats = np.zeros((E, 3), dtype=np.float32)
    edge_feats[:, 0] = np.asarray(w_list, dtype=np.float32)
    edge_feats[:, 1] = np.asarray(arrival_list, dtype=np.float32)
    edge_feats[:, 2] = 1.0

    return {"node_feats": node_feats, "edge_index": edge_index, "edge_feats": edge_feats}


# ---------------------------------------------------------------------------
# Task 3: arm models.
# ---------------------------------------------------------------------------
#
# Every arm's raw scalar output goes through the SAME
# `clamp(softplus(raw), 0, residual_cap)` convention as
# `C.ContinuousHeuristicModel.forward` (the two trace arms reuse that class
# directly and get the clamp for free -- see `continuous_prm_common.py:1215-
# 1223` -- so this helper is applied exactly once per non-trace arm's
# forward, never stacked on top of the trace arms' own internal clamp).

def _softplus_clamp(raw: torch.Tensor, cap: float = 4.0) -> torch.Tensor:
    return torch.clamp(F.softplus(raw), min=0.0, max=float(cap))


class MLPArm(nn.Module):
    """Arm 1: the explicit MLP control. `flatten(pad(sequence))` (120-dim)
    -> 3 hidden GELU layers of `cfg.mlp_width` -> scalar, softplus-clamped.
    No sequence/graph structure available -- this is the arm the whole
    hierarchy-vs-substrate audit demanded (see MEMORY `program-audit-c11`)."""

    def __init__(self, cfg: C11MissionConfig):
        super().__init__()
        in_dim = cfg.seq_max * cfg.token_dim
        w = cfg.mlp_width
        # Cap sourced from cfg (like the trace arms' max_norm_residual and
        # the unet path's cfg.residual_cap) so a future cap change cannot
        # silently diverge across arms (T3-review Minor).
        self.cap = float(cfg.residual_cap)
        self.net = nn.Sequential(
            nn.Linear(in_dim, w),
            nn.GELU(),
            nn.Linear(w, w),
            nn.GELU(),
            nn.Linear(w, w),
            nn.GELU(),
            nn.Linear(w, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x).squeeze(-1)
        return _softplus_clamp(raw, cap=self.cap)


class UNetFiLMField(nn.Module):
    """Arm 2: the field U-Net with FiLM stage conditioning. Encoder-decoder
    identical in shape to `continuous_prm_c6_heatmap_value_field.UNetField`
    (`DoubleConv` stages, imported -- not reimplemented), but `in_channels=5`
    (`encode_field_grids`'s 5 grid channels), `base=24`, PLUS a stage
    embedding (`K_max+1=9` stages) FiLM-modulating the bottleneck features
    `b = b * (1 + gamma) + beta` before decoding. Single residual head
    (`Conv2d(base, 1, 1)`) -- no path-mask head (C11 has no path-mask
    target). `forward(grid, stage_ids) -> (B, 1, 64, 64)`; the softplus-clamp
    is applied AFTER gathering node values from the grid (`predict_field`),
    not inside this class's `forward` -- the raw field is a continuous
    surface, and clamping the surface before bilinear gather vs. after
    differ only at the (immaterial) sub-cell interpolation boundary, but the
    per-arm contract elsewhere in this module (`MLPArm`, `ProductGraphGNN`)
    clamps the FINAL scalar per query state, so this class matches that by
    leaving the surface unclamped and letting `predict_field` clamp the
    gathered values."""

    def __init__(self, cfg: C11MissionConfig, base: int = 24):
        super().__init__()
        in_channels = cfg.grid_channels
        self.base = base
        self.e1 = C6.DoubleConv(in_channels, base)
        self.e2 = C6.DoubleConv(base, base * 2)
        self.e3 = C6.DoubleConv(base * 2, base * 4)
        self.e4 = C6.DoubleConv(base * 4, base * 8)
        self.b = C6.DoubleConv(base * 8, base * 8)
        self.p = nn.MaxPool2d(2)
        self.u4 = nn.ConvTranspose2d(base * 8, base * 8, 2, stride=2)
        self.d4 = C6.DoubleConv(base * 16, base * 4)
        self.u3 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.d3 = C6.DoubleConv(base * 8, base * 2)
        self.u2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.d2 = C6.DoubleConv(base * 4, base)
        self.u1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.d1 = C6.DoubleConv(base * 2, base)
        self.residual_head = nn.Conv2d(base, 1, 1)

        n_stages = cfg.k_max + 1
        self.stage_emb = nn.Embedding(n_stages, cfg.film_dim)
        self.film_gamma = nn.Linear(cfg.film_dim, base * 8)
        self.film_beta = nn.Linear(cfg.film_dim, base * 8)

    def forward(self, grid: torch.Tensor, stage_ids: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(grid)
        e2 = self.e2(self.p(e1))
        e3 = self.e3(self.p(e2))
        e4 = self.e4(self.p(e3))
        b = self.b(self.p(e4))

        emb = self.stage_emb(stage_ids)
        gamma = self.film_gamma(emb)
        beta = self.film_beta(emb)
        b = b * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

        x = self.u4(b)
        x = self.d4(torch.cat([x, e4], dim=1))
        x = self.u3(x)
        x = self.d3(torch.cat([x, e3], dim=1))
        x = self.u2(x)
        x = self.d2(torch.cat([x, e2], dim=1))
        x = self.u1(x)
        x = self.d1(torch.cat([x, e1], dim=1))
        return self.residual_head(x)


class ProductGraphGNN(nn.Module):
    """Arm 3: hand-rolled message passing over the product graph (no
    torch-geometric -- Blackwell/Windows wheel risk, per spec section 3).
    `cfg.gnn_rounds` (8) UNTIED rounds of hidden 128: per round, every edge
    computes a message from its source node's hidden state + its own edge
    features (`MLP_r([h_src, edge_feats])`, 131->128->128, GELU), messages
    are MEAN-aggregated per destination node (`index_add_` + degree
    division -- no scatter/torch_geometric dependency), and the destination
    node's hidden state is residually updated
    (`h = h + MLP_r'([h, agg])`, 256->128->128, GELU). Input projection
    `Linear(14, 128)`; head `Linear(128,128) -> GELU -> Linear(128,1)` +
    softplus-clamp. `forward(node_feats, edge_index, edge_feats) ->
    (n_nodes,)`."""

    def __init__(self, cfg: C11MissionConfig):
        super().__init__()
        H = cfg.gnn_hidden
        self.hidden = H
        self.rounds = cfg.gnn_rounds
        # Cap sourced from cfg -- see MLPArm.__init__ (T3-review Minor).
        self.cap = float(cfg.residual_cap)
        self.in_proj = nn.Linear(14, H)
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(H + 3, H), nn.GELU(), nn.Linear(H, H), nn.GELU())
            for _ in range(self.rounds)
        ])
        self.update_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Linear(H, H), nn.GELU())
            for _ in range(self.rounds)
        ])
        self.head = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, 1))

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor,
                edge_feats: torch.Tensor) -> torch.Tensor:
        n_nodes = node_feats.shape[0]
        h = self.in_proj(node_feats)

        if edge_index.shape[1] > 0:
            src = edge_index[0]
            dst = edge_index[1]
            ones = torch.ones(edge_index.shape[1], device=node_feats.device, dtype=h.dtype)
            degree = torch.zeros(n_nodes, device=node_feats.device, dtype=h.dtype)
            degree.index_add_(0, dst, ones)
            degree = degree.clamp(min=1.0)

        for r in range(self.rounds):
            if edge_index.shape[1] == 0:
                agg = torch.zeros_like(h)
            else:
                h_src = h[src]
                msg_in = torch.cat([h_src, edge_feats], dim=-1)
                msg = self.msg_mlps[r](msg_in)
                agg = torch.zeros(n_nodes, self.hidden, device=node_feats.device, dtype=h.dtype)
                agg.index_add_(0, dst, msg)
                agg = agg / degree[:, None]
            upd_in = torch.cat([h, agg], dim=-1)
            h = h + self.update_mlps[r](upd_in)

        raw = self.head(h).squeeze(-1)
        return _softplus_clamp(raw, cap=self.cap)


ARM_NAMES: Tuple[str, ...] = ("mlp", "unet_film", "gnn", "hrm_trace", "onlstm_trace")

# Trace-arm hidden dims chosen to land the [0.5M, 3.5M] param band (verified
# by `test_arm_constructors_and_param_counts`): HRM H=224 -> ~2.90M,
# ON-LSTM H=256 -> ~1.25M. Both land in-band on the first try -- no
# adjustment needed (num_layers=2, k_step=2, num_heads=4, chunk_size=8,
# head_hidden=256 per spec section 3 item 4).
_HRM_TRACE_HIDDEN = 224
_ONLSTM_TRACE_HIDDEN = 256

# ---------------------------------------------------------------------------
# Task 10: scaled-addendum big arms (`unet_film_big` / `hrm_trace_big`).
#
# Per the C11_RESULTS.md diagnosis addendum's "Provenance" paragraph: the
# scaled addendum (T10) re-runs the BEST (unet_film) and WORST (hrm_trace)
# matched-scale arms at ~10-20M params, on config A at K in {2, 8}, 3 seeds,
# to test whether scale (a) changes the shallow-K ranking or (b) escapes
# the hrm_trace training-collapse attractor observed at matched scale (A8/
# C8 collapsed to constant cap-predictions -- see the addendum's Finding 2).
#
# BOTH big arms are registered as FIRST-CLASS arm names -- own tuple
# `BIG_ARM_NAMES` (kept SEPARATE from `ARM_NAMES`, deliberately NOT
# appended to it: `ARM_NAMES` backs `_default_full_arms()`, which
# `--scale-preset local`/the CLI's default `--arms` both consume, and the
# `_big` arms must NEVER leak into the matched grid) -- dispatched inside
# `build_arm`/`_forward_arm_batch` by an explicit `arm_name ==
# "..._big"`/`in (..., "..._big")` branch alongside the 5 matched natives'
# own branches. Training rides `_resolve_trainer`'s EXISTING fallback path
# for free: `arm_name in ARM_NAMES` is False for a `_big` name and
# `PROVIDER_BUILDERS.get(arm_name)` is also `None` (big arms are not
# registry entries), so `_resolve_trainer` already returns `train_arm`
# unchanged via its final "unknown name -- let train_arm/build_arm raise"
# branch -- and `train_arm` itself only ever calls `build_arm(arm_name,
# cfg)` internally, which DOES know these two names. No edit to
# `_resolve_trainer`/`ARM_NAMES` itself was needed or made.
#
# This shape -- new arm NAMES, not a `C11MissionConfig.arm_scale` flag
# threaded through `build_arm` -- was chosen because `_ckpt_key`'s format
# string (`{arm}__{config_label}{K}__s{seed}`) already derives its
# checkpoint/manifest key purely from `arm_name`, so a `_big`-suffixed NAME
# gives the spec's required distinct, non-colliding key for free -- no
# separate suffix-injection parameter to keep in sync across
# `run_train`/`run_eval`/the manifest schema, and `build_arm(name)`'s
# signature stays unchanged. The ONLY two arms with a big variant are
# unet_film and hrm_trace (spec: "ONLY these two arms support 'big'"); any
# other `*_big` name (e.g. `mlp_big`) falls straight through to
# `build_arm`'s existing "unknown arm name" ValueError -- no special
# rejection branch is needed, since these names were simply never wired in.
#
# unet_film_big: `base=64` (same FiLM structure as the matched arm, just a
# wider `UNetFiLMField.base` -- spec's literal "base=96" example overshoots
# the [10e6, 20e6] band by ~67% (33.3M params measured directly); base=64
# lands mid-band at 14,824,161 params, verified by
# `test_big_arm_constructors_and_param_counts`).
#
# hrm_trace_big: `hidden_dim=512` (BackboneConfig), `head_hidden=512`
# (scaled up from the matched arm's 256 in step with hidden_dim -- an
# unscaled 256-wide head would be a bottleneck relative to a 512-wide
# backbone), `num_layers=2` unchanged per spec ("num_layers stays 2").
# Spec's literal "start H=640" also overshoots the band (23.2M measured);
# H=512 lands mid-band at 15,015,937 params -- both big arms land within
# ~200K params of each other, a deliberately close pairing so the T10
# scaled comparison isn't confounded by one arm getting a much bigger
# compute/capacity budget than the other.
BIG_ARM_NAMES: Tuple[str, ...] = ("unet_film_big", "hrm_trace_big")
_UNET_FILM_BIG_BASE = 64
_HRM_TRACE_BIG_HIDDEN = 512
_HRM_TRACE_BIG_HEAD_HIDDEN = 512

# ---------------------------------------------------------------------------
# Task 4: provider registry (defined here, ahead of `build_arm`, so
# `build_arm` itself can fall back to it -- see `build_arm`'s final branch).
# ---------------------------------------------------------------------------
#
# `PROVIDER_BUILDERS` maps arm name -> `{"build", "forward_batch"}`:
# EVERYTHING an externally-defined arm needs to participate end-to-end --
# construction (`build(cfg) -> nn.Module`, consumed by `build_arm` and hence
# `make_provider`) AND batch inference (`forward_batch(model, bundle,
# states, cfg, device) -> (len(states),) tensor of yhat in
# [0, cfg.residual_cap]`, consumed by `_forward_arm_batch`, hence
# `predict_field`, hence the providers `make_provider` returns and
# `run_eval` runs). Registering construction alone would be a trap:
# `make_provider` would succeed and the returned provider would then crash
# inside `predict_field` on its first bundle (the T4 review demonstrated
# exactly this), so registration requires BOTH pieces up front. This is the
# hook the T7 HRM-v2 module (`continuous_prm_c11_hrmv2_arm`) registers
# `hrmv2_act` / `hrmv2_act_k{1,2,4,8}` through, post-import, without this
# module knowing about it (no circular import, no edit-this-module-for-
# every-new-arm coupling). Registered arms need only `make_provider` +
# `predict_field`/eval to work (T7 brings its OWN trainer); `train_arm`
# additionally works for free if the registered `forward_batch` is
# differentiable, but that is a bonus, not part of the contract. NATIVE arm
# names are REJECTED (ValueError): the 5 native constructors/forwards are
# pinned by this module's own if/elif chains, and silently shadowing them
# from outside would desync `build_arm` (native-first) from the registry --
# one authority per name.

PROVIDER_BUILDERS: Dict[str, Dict[str, Callable]] = {}


def register_provider_builder(
    name: str,
    build: Callable[[C11MissionConfig], nn.Module],
    forward_batch: Callable[..., torch.Tensor],
    train: Optional[Callable[..., Tuple[Dict[str, torch.Tensor], dict]]] = None,
    ckpt_alias: Optional[str] = None,
) -> None:
    """Register an external arm under `name` in the module-level provider
    registry, carrying the FULL per-arm contract:

      - `build(cfg) -> nn.Module`: construct the arm's model (fresh weights;
        `make_provider` loads a state_dict into it afterwards).
      - `forward_batch(model, bundle, states, cfg, device) -> torch.Tensor`:
        batch inference over `states` (a sequence of (i, s) pairs) on
        `bundle`, returning a `(len(states),)` tensor of residual
        predictions ALREADY in `[0, cfg.residual_cap]` (the registered arm
        owns its own output-clamp convention, exactly like the 5 native
        arms own theirs) -- the same signature as this module's private
        per-arm forward helpers (`_forward_mlp_or_trace` etc.), so the
        registered arm plugs into `_forward_arm_batch`'s dispatch and from
        there into `predict_field`/`make_provider`/`run_eval` end-to-end.
      - `train` (OPTIONAL, T7): `train(cell, bundles, seed, cfg, epochs) ->
        (state_dict_cpu, meta)`, the SAME return contract as `train_arm`.
        When given, `run_train` calls THIS function for the arm instead of
        `train_arm` (see `run_train`'s dispatch). When omitted (`None`,
        the default), the arm is EVAL-ONLY: `run_train` skips it entirely
        rather than silently falling through to `train_arm` (which would
        crash or -- worse -- silently train the wrong model, since
        `train_arm` only knows the 5 native `build_arm` branches plus
        whatever `PROVIDER_BUILDERS[name]["build"]` returns, with no
        awareness of a bespoke training loop like HRM-v2's per-segment
        ACT trainer). This is how the forced-k HRM-v2 diagnostics
        (`hrmv2_act_k{1,2,4,8}`) stay eval-only: they register `build`
        (needed so `make_provider`/`predict_field` can reconstruct the
        model) and `forward_batch` (the forced-k inference path) but no
        `train`.
      - `ckpt_alias` (OPTIONAL, T7): the name of ANOTHER registered (or
        native) arm whose CHECKPOINT this arm should be evaluated against.
        Used by `run_eval` to let eval-only arms (the forced-k HRM-v2
        diagnostics) ride the SAME trained weights as `hrmv2_act` without
        ever training their own copy -- see `run_eval`'s alias-resolution
        block. `None` (the default) means "this arm's own checkpoints only"
        (every native arm and most registered arms).

    `build`/`forward_batch` are REQUIRED (a build-only registration would
    crash at provider call time, not registration time -- the failure the
    T4 review flagged). Registering a native arm name raises ValueError.

    Safe to call from a different module after this one has been imported
    (e.g. `continuous_prm_c11_hrmv2_arm` registering `hrmv2_act` and
    friends) -- `PROVIDER_BUILDERS` is a plain module-level dict, so any
    import of `continuous_prm_c11_mission` sees every registration made so
    far against it, regardless of which module performed the registration."""
    if name in ARM_NAMES:
        raise ValueError(
            f"cannot register a provider builder under native arm name {name!r}; "
            f"the native arms {ARM_NAMES} are constructed by build_arm's own chain "
            f"(silent shadowing is not allowed -- pick a distinct name)"
        )
    PROVIDER_BUILDERS[name] = {
        "build": build,
        "forward_batch": forward_batch,
        "train": train,
        "ckpt_alias": ckpt_alias,
    }


def build_arm(name: str, cfg: C11MissionConfig) -> nn.Module:
    """Construct one of the 5 native arm models by name, falling back to the
    provider registry (`PROVIDER_BUILDERS`) for any externally-registered
    name before raising. `hrm_trace`/`onlstm_trace` are
    `C.ContinuousHeuristicModel` instances (the existing trace backbones,
    `token_dim=12`, `max_norm_residual=cfg.residual_cap`) fed the real
    mission trace (`encode_trace_padded`) rather than the legacy 24-token
    feature bag."""
    if name == "mlp":
        return MLPArm(cfg)
    if name == "unet_film":
        return UNetFiLMField(cfg, base=24)
    if name == "gnn":
        return ProductGraphGNN(cfg)
    if name == "hrm_trace":
        backbone_cfg = C.BackboneConfig(
            name="hrm_trace", backbone_type="hrm", hidden_dim=_HRM_TRACE_HIDDEN,
            num_layers=2, k_step=2, num_heads=4, chunk_size=8, head_hidden=256,
        )
        return C.ContinuousHeuristicModel(backbone_cfg, token_dim=cfg.token_dim, max_norm_residual=cfg.residual_cap)
    if name == "onlstm_trace":
        backbone_cfg = C.BackboneConfig(
            name="onlstm_trace", backbone_type="onlstm", hidden_dim=_ONLSTM_TRACE_HIDDEN,
            num_layers=2, chunk_size=8, head_hidden=256,
        )
        return C.ContinuousHeuristicModel(backbone_cfg, token_dim=cfg.token_dim, max_norm_residual=cfg.residual_cap)
    if name == "unet_film_big":
        # Task 10 scaled addendum -- same FiLM structure as "unet_film",
        # just a wider base (see BIG_ARM_NAMES block above for the exact
        # param-count derivation).
        return UNetFiLMField(cfg, base=_UNET_FILM_BIG_BASE)
    if name == "hrm_trace_big":
        # Task 10 scaled addendum -- same hrm backbone_type/num_layers=2 as
        # "hrm_trace", raised hidden_dim (+ head_hidden scaled in step).
        backbone_cfg = C.BackboneConfig(
            name="hrm_trace_big", backbone_type="hrm", hidden_dim=_HRM_TRACE_BIG_HIDDEN,
            num_layers=2, k_step=2, num_heads=4, chunk_size=8, head_hidden=_HRM_TRACE_BIG_HEAD_HIDDEN,
        )
        return C.ContinuousHeuristicModel(backbone_cfg, token_dim=cfg.token_dim, max_norm_residual=cfg.residual_cap)
    entry = PROVIDER_BUILDERS.get(name)
    if entry is not None:
        return entry["build"](cfg)
    raise ValueError(
        f"unknown arm name {name!r}; expected one of {ARM_NAMES}, {BIG_ARM_NAMES}, "
        f"or a registered provider builder"
    )


# ---------------------------------------------------------------------------
# Task 3: differentiable grid gather (matches
# `continuous_prm_c6_heatmap_value_field.interpolate_grid_values` exactly).
# ---------------------------------------------------------------------------

def _gather_grid_values_torch(grid: torch.Tensor, side: float, points: torch.Tensor) -> torch.Tensor:
    """Bilinear-gather `grid` (H, W) at `points` (n, 2) world coordinates,
    byte-for-byte the same indexing math as
    `continuous_prm_c6_heatmap_value_field.interpolate_grid_values`
    (verified equal on random inputs) but differentiable end-to-end through
    `grid` (needed so training gradients reach the U-Net's weights) --
    `interpolate_grid_values` itself is numpy-only and used at eval time by
    `encode_field_grids`'s own tests, not reimplemented here, only mirrored.
    No non-finite fallback branch: the U-Net's raw output is always finite
    (no INF cells like the world's distance-field target), so that branch of
    the original numpy version has no analog here."""
    n = grid.shape[-1]
    fx = points[:, 0] / side * n - 0.5
    fy = points[:, 1] / side * n - 0.5
    x0 = torch.clamp(torch.floor(fx), 0, n - 1).long()
    y0 = torch.clamp(torch.floor(fy), 0, n - 1).long()
    x1 = torch.clamp(x0 + 1, max=n - 1)
    y1 = torch.clamp(y0 + 1, max=n - 1)
    tx = torch.clamp(fx - x0.to(fx.dtype), 0.0, 1.0)
    ty = torch.clamp(fy - y0.to(fy.dtype), 0.0, 1.0)
    c00 = grid[x0, y0]
    c10 = grid[x1, y0]
    c01 = grid[x0, y1]
    c11 = grid[x1, y1]
    return (1.0 - tx) * (1.0 - ty) * c00 + tx * (1.0 - ty) * c10 + (1.0 - tx) * ty * c01 + tx * ty * c11


# ---------------------------------------------------------------------------
# Task 3: per-arm batch encoding (shared by `train_arm` and `predict_field`).
# ---------------------------------------------------------------------------

def _points_tensor(bundle: WorldBundle, node_indices: Sequence[int], device: torch.device) -> torch.Tensor:
    pts = bundle.rm.points[np.asarray(node_indices, dtype=np.int64)]
    return torch.from_numpy(pts.astype(np.float32)).to(device)


def _forward_mlp_or_trace(arm_name: str, model: nn.Module, bundle: WorldBundle,
                           states: Sequence[Tuple[int, int]], cfg: C11MissionConfig,
                           device: torch.device) -> torch.Tensor:
    if arm_name == "mlp":
        flat = encode_mlp(bundle, states, cfg)
        x = torch.from_numpy(flat).to(device)
        return model(x)
    # hrm_trace / onlstm_trace: ContinuousHeuristicModel consumes the padded
    # token sequence directly (pad tokens are zero rows -- acceptable; the
    # sequence is consumed left-to-right, so real tokens always precede
    # padding and no attention mask is needed for these recurrent backbones).
    padded, _mask = encode_trace_padded(bundle, states, cfg)
    x = torch.from_numpy(padded).to(device)
    return model(x)


def _bundle_grids(bundle: WorldBundle, cfg: C11MissionConfig) -> np.ndarray:
    """The bundle's full `(K+1, grid_channels, grid_size, grid_size)`
    float32 field-grid stack (`encode_field_grids` for every stage s in
    0..K), computed once on first use and stored on the bundle
    (`bundle.field_grids`). Rasterizing a stage costs ~quarter-second;
    the training loop touches every (batch x bundle x distinct-stage)
    combination, so re-encoding per batch turns a minutes-scale U-Net run
    into hours at the full recipe (T3-review Important). Value-transparent
    by construction -- a plain per-stage stack of `encode_field_grids`
    outputs, pinned by `test_bundle_grids_value_transparency` and the
    call-count test `test_unet_grid_cache_call_count`."""
    if bundle.field_grids is not None:
        return bundle.field_grids
    K = len(bundle.wp)
    grids = np.stack([encode_field_grids(bundle, s, cfg) for s in range(K + 1)], axis=0)
    bundle.field_grids = grids
    return grids


def _forward_unet_film(model: "UNetFiLMField", bundle: WorldBundle,
                        states: Sequence[Tuple[int, int]], cfg: C11MissionConfig,
                        device: torch.device) -> torch.Tensor:
    """Forward each DISTINCT stage's grid ONCE per batch (grids themselves
    come from the bundle-level `_bundle_grids` stack -- rasterized once per
    bundle EVER, not per batch), then gather every batch row's node value
    from that stage's output surface (bilinear, matching the C6 node-gather
    convention -- see `_gather_grid_values_torch`)."""
    side = float(bundle.world.side_len)
    by_stage: Dict[int, List[int]] = {}
    for row, (i, s) in enumerate(states):
        by_stage.setdefault(s, []).append(row)

    stage_list = sorted(by_stage.keys())
    all_grids = _bundle_grids(bundle, cfg)  # (K+1, C, H, W)
    grids_np = all_grids[np.asarray(stage_list, dtype=np.int64)]
    grids_t = torch.from_numpy(grids_np).to(device)
    stage_ids_t = torch.tensor(stage_list, dtype=torch.long, device=device)

    surfaces = model(grids_t, stage_ids_t).squeeze(1)  # (n_stages, 64, 64)

    n_rows = len(states)
    out = torch.zeros(n_rows, device=device)
    for stage_pos, s in enumerate(stage_list):
        rows = by_stage[s]
        node_indices = [states[r][0] for r in rows]
        pts = _points_tensor(bundle, node_indices, device)
        vals = _gather_grid_values_torch(surfaces[stage_pos], side, pts)
        idx_t = torch.tensor(rows, dtype=torch.long, device=device)
        out.index_copy_(0, idx_t, vals)
    return out


def _forward_gnn(model: "ProductGraphGNN", bundle: WorldBundle,
                  states: Sequence[Tuple[int, int]], cfg: C11MissionConfig,
                  device: torch.device) -> torch.Tensor:
    """Forward the bundle's FULL product graph once, then index the batch's
    rows by flat id `s*N + i`."""
    graph = encode_product_graph(bundle, cfg)
    node_feats = torch.from_numpy(graph["node_feats"]).to(device)
    edge_index = torch.from_numpy(graph["edge_index"]).to(device)
    edge_feats = torch.from_numpy(graph["edge_feats"]).to(device)

    all_out = model(node_feats, edge_index, edge_feats)  # ((K+1)*N,)

    N = bundle.rm.points.shape[0]
    flat_ids = [s * N + i for (i, s) in states]
    idx_t = torch.tensor(flat_ids, dtype=torch.long, device=device)
    return all_out[idx_t]


def _forward_arm_batch(arm_name: str, model: nn.Module, bundle: WorldBundle,
                        states: Sequence[Tuple[int, int]], cfg: C11MissionConfig,
                        device: torch.device) -> torch.Tensor:
    """Dispatch to the per-arm batch-forward helper. Returns a `(len(states),)`
    tensor of predictions in `[0, cfg.residual_cap]` (every helper's model
    already applies the softplus-clamp -- `MLPArm`/`ProductGraphGNN`/
    `ContinuousHeuristicModel` internally; `UNetFiLMField` does NOT (its
    `forward` returns the raw unclamped surface), so `_forward_unet_film`'s
    gathered output is clamped here, the ONE place the clamp is applied for
    that arm -- never inside `UNetFiLMField.forward` itself, which would
    double-apply it if `predict_field` ever gathered from an already-clamped
    surface).

    Externally-registered arms dispatch to their registered `forward_batch`
    (same signature minus `arm_name`; the registered arm owns its own clamp
    convention, per `register_provider_builder`'s contract) -- this is the
    branch that makes the registry contract hold END-TO-END: without it,
    `make_provider`'s reconstruction succeeds but the returned provider
    crashes here on its first bundle (T4-review Important 1)."""
    if arm_name in ("mlp", "hrm_trace", "onlstm_trace", "hrm_trace_big"):
        # hrm_trace_big (Task 10) is, like hrm_trace/onlstm_trace, a
        # ContinuousHeuristicModel fed the padded trace sequence directly --
        # `_forward_mlp_or_trace`'s trace-token branch (its final 3 lines,
        # reached whenever arm_name != "mlp") is already architecture-
        # generic, so no new branch is needed inside that helper itself.
        return _forward_mlp_or_trace(arm_name, model, bundle, states, cfg, device)
    if arm_name in ("unet_film", "unet_film_big"):
        # unet_film_big (Task 10) reuses the SAME `_forward_unet_film` path
        # (its own signature is architecture-generic -- it only calls
        # `model(grids_t, stage_ids_t)`, which works identically for a
        # wider-`base` UNetFiLMField) and the SAME post-gather clamp
        # convention as the matched arm (see this function's own docstring
        # on why the clamp lives here, not inside UNetFiLMField.forward).
        raw = _forward_unet_film(model, bundle, states, cfg, device)
        return _softplus_clamp(raw, cap=cfg.residual_cap)
    if arm_name == "gnn":
        return _forward_gnn(model, bundle, states, cfg, device)
    entry = PROVIDER_BUILDERS.get(arm_name)
    if entry is not None:
        return entry["forward_batch"](model, bundle, states, cfg, device)
    raise ValueError(
        f"unknown arm name {arm_name!r}; expected one of {ARM_NAMES}, {BIG_ARM_NAMES}, "
        f"or a registered provider builder"
    )


# ---------------------------------------------------------------------------
# Task 3: predict_field.
# ---------------------------------------------------------------------------

def predict_field(arm_name: str, model: nn.Module, bundle: WorldBundle,
                   cfg: Optional[C11MissionConfig] = None) -> np.ndarray:
    """Batch ALL (i, s) product states through `arm_name`'s encoder+model
    under `torch.no_grad()`, returning `(K+1, N)` float64 values in
    `[0, cfg.residual_cap]`. Deterministic (no dropout/batchnorm in any arm;
    eval mode is the caller's responsibility for exact reproducibility
    across calls, but `no_grad` alone is already sufficient here since none
    of the 5 arms have train/eval-mode-dependent layers)."""
    cfg = cfg or C11MissionConfig()
    K = len(bundle.wp)
    N = bundle.rm.points.shape[0]
    device = next(model.parameters()).device

    states = [(i, s) for s in range(K + 1) for i in range(N)]

    with torch.no_grad():
        out = _forward_arm_batch(arm_name, model, bundle, states, cfg, device)

    field = out.detach().cpu().numpy().astype(np.float64).reshape(K + 1, N)
    return field


# ---------------------------------------------------------------------------
# Task 3: matched trainer.
# ---------------------------------------------------------------------------

def _training_seed(seed: int, cell: dict) -> int:
    """The pre-registered training-seed formula (identical for torch and the
    numpy shuffle RNG -- see plan section "Matched recipe"): `10007*(seed+1)
    + 101*config_idx + K`."""
    return 10007 * (seed + 1) + 101 * int(cell["config_idx"]) + int(cell["K"])


def _forward_batch_by_bundle(arm_name: str, model: nn.Module, bundles: Sequence[WorldBundle],
                              rows_i: np.ndarray, rows_s: np.ndarray, rows_bidx: np.ndarray,
                              cfg: C11MissionConfig, device: torch.device) -> torch.Tensor:
    """Forward one batch's rows through `model`, grouped by `bundle_idx`
    (each arm's per-bundle encode is computed once per DISTINCT bundle
    present in the batch, not once per row -- see `_forward_unet_film`'s own
    per-stage grouping for the analogous within-bundle savings). Returns
    predictions in the SAME row order as `rows_i`/`rows_s`/`rows_bidx`."""
    n = rows_i.shape[0]
    out = torch.zeros(n, device=device)
    unique_bidx = np.unique(rows_bidx)
    for b_idx in unique_bidx:
        b_idx = int(b_idx)
        row_mask = rows_bidx == b_idx
        row_positions = np.nonzero(row_mask)[0]
        states = list(zip(rows_i[row_mask].tolist(), rows_s[row_mask].tolist()))
        bundle = bundles[b_idx]
        preds = _forward_arm_batch(arm_name, model, bundle, states, cfg, device)
        idx_t = torch.from_numpy(row_positions.astype(np.int64)).to(device)
        out.index_copy_(0, idx_t, preds)
    return out


def train_arm(arm_name: str, cell: dict, bundles: Sequence[WorldBundle], seed: int,
              cfg: Optional[C11MissionConfig] = None, epochs: Optional[int] = None
              ) -> Tuple[Dict[str, torch.Tensor], dict]:
    """The matched recipe (spec section 4 / plan "Shared core definitions"):
    smooth-L1 (beta=`cfg.smooth_l1_beta`) on ŷ vs y, AdamW(lr=`cfg.lr`,
    weight_decay=`cfg.weight_decay`), grad-clip `cfg.grad_clip`, batch
    `cfg.batch_size`, `cfg.epochs` epochs (overridable via `epochs`),
    IDENTICAL across all 5 arms by construction (every arm goes through this
    one function -- no per-arm branch touches lr/epochs/optimizer/loss).

    Seeding: `torch.manual_seed` and the numpy shuffle RNG both seeded with
    `_training_seed(seed, cell)` -- two calls with identical inputs produce
    bit-identical `state_dict`s (verified by `test_trainer_determinism`).
    Bit-determinism is guaranteed on CPU ONLY: the CUDA kernels backing
    `index_add_` (GNN aggregation, batch scatter-copy) and cuDNN
    convolutions are not bitwise deterministic, so identical-seed GPU runs
    may differ in the last ulps. Arm fairness rests on the matched
    data/recipe and the 3 pre-registered training seeds per (arm, config,
    K), not on bit-reproducibility of any single run.

    Returns `(state_dict_cpu, meta)`; `meta` carries `param_count`,
    `first_epoch_loss` (mean batch loss of epoch 1, for the tiny-overfit
    smoke test), `final_loss` (mean batch loss of the last epoch), `epochs`,
    `seed`, `arm`, `config_label`, `K`, `wall_s`."""
    cfg = cfg or C11MissionConfig()
    n_epochs = int(epochs) if epochs is not None else int(cfg.epochs)

    train_seed_val = _training_seed(seed, cell)
    torch.manual_seed(train_seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed_val)
    shuffle_rng = np.random.default_rng(train_seed_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_arm(arm_name, cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    targets = stack_targets(bundles)  # (n, 4): (i, s, y, bundle_idx)
    n_rows = targets.shape[0]
    all_i = targets[:, 0].astype(np.int64)
    all_s = targets[:, 1].astype(np.int64)
    all_y = targets[:, 2].astype(np.float32)
    all_bidx = targets[:, 3].astype(np.int64)

    t0 = time.time()
    first_epoch_loss: Optional[float] = None
    final_loss: float = float("nan")

    model.train()
    for epoch in range(n_epochs):
        order = shuffle_rng.permutation(n_rows)
        epoch_losses: List[float] = []
        for start in range(0, n_rows, cfg.batch_size):
            batch_idx = order[start:start + cfg.batch_size]
            rows_i = all_i[batch_idx]
            rows_s = all_s[batch_idx]
            rows_bidx = all_bidx[batch_idx]
            y_batch = torch.from_numpy(all_y[batch_idx]).to(device)

            opt.zero_grad(set_to_none=True)
            preds = _forward_batch_by_bundle(arm_name, model, bundles, rows_i, rows_s, rows_bidx, cfg, device)
            if not torch.isfinite(preds).all():
                raise RuntimeError(f"nonfinite predictions for arm {arm_name!r}")
            loss = F.smooth_l1_loss(preds, y_batch, beta=cfg.smooth_l1_beta)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss for arm {arm_name!r}")
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            epoch_losses.append(float(loss.item()))

        epoch_mean = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        if epoch == 0:
            first_epoch_loss = epoch_mean
        final_loss = epoch_mean

    wall_s = time.time() - t0

    state_dict_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    meta = {
        "arm": arm_name,
        "config_label": cell.get("config_label"),
        "K": int(cell["K"]),
        "seed": int(seed),
        "epochs": n_epochs,
        "param_count": int(param_count),
        "first_epoch_loss": first_epoch_loss,
        "final_loss": final_loss,
        "wall_s": wall_s,
    }
    return state_dict_cpu, meta


# ---------------------------------------------------------------------------
# Task 4: make_provider (the provider registry itself -- PROVIDER_BUILDERS /
# register_provider_builder -- is defined earlier, just ahead of `build_arm`,
# so `build_arm` can fall back to it too; see that block's comment).
# ---------------------------------------------------------------------------

def make_provider(arm_name: str, state_dict: Dict[str, torch.Tensor],
                   cfg: Optional[C11MissionConfig] = None) -> Callable[[WorldBundle], np.ndarray]:
    """Build a provider callable for `arm_name`: reconstructs the model via
    `build_arm` (native chain, falling back to the `PROVIDER_BUILDERS`
    registry for externally-registered names), loads `state_dict` (CPU,
    `strict=True`), sets eval mode, and returns `provider(bundle) -> h_hat`
    where `h_hat[s, i] = bundle.hl(i, s) + bundle.world.side_len *
    yhat[s, i]` (`yhat = predict_field(arm_name, model, bundle, cfg)`, already
    in `[0, cfg.residual_cap]` by the arm's own softplus-clamp convention).
    By construction `h_hat >= hl` everywhere (`yhat >= 0`), so a learned
    provider's output is never LESS admissible-looking than the leg-sum
    floor it's built on top of (it need not be admissible itself -- these
    arms are explicitly inadmissible, per spec section 6 -- but it can never
    fall below hl).

    Handles every cell's bundles (any K including 0, any of configs A/B/C):
    `predict_field` batches over `range(K+1) x range(N)` directly from the
    bundle's own `wp`/`rm`, so no arm-specific K/door branching is needed
    here.

    The returned callable is deterministic: `make_provider` reconstructs a
    FRESH model from `state_dict` (no shared mutable state across providers
    built from the same state_dict), and `predict_field` runs under
    `torch.no_grad()` with no dropout/batchnorm in any of the 5 native arms,
    so two calls on the same bundle -- from the same OR a freshly-built
    provider -- produce byte-identical output. (Externally-registered arms
    flow through the same path; THEIR determinism is the registered
    `forward_batch`'s own responsibility, per `register_provider_builder`.)"""
    cfg = cfg or C11MissionConfig()
    model = build_arm(arm_name, cfg)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    def _provider(bundle: WorldBundle) -> np.ndarray:
        yhat = predict_field(arm_name, model, bundle, cfg)  # (K+1, N) float64, in [0, cap]
        K = len(bundle.wp)
        N = bundle.rm.points.shape[0]
        side = float(bundle.world.side_len)
        hl_arr = np.array([[bundle.hl(i, s) for i in range(N)] for s in range(K + 1)], dtype=np.float64)
        return hl_arr + side * yhat

    return _provider


# ---------------------------------------------------------------------------
# Task 4: checkpoint path + manifest helpers.
# ---------------------------------------------------------------------------

def _ckpt_key(arm_name: str, cell: dict, seed: int) -> str:
    """The manifest key / checkpoint stem for one (arm, cell, seed) run:
    `{arm}__{config_label}{K}__s{seed}`, e.g. `mlp__A2__s0`."""
    return f"{arm_name}__{cell['config_label']}{int(cell['K'])}__s{int(seed)}"


def _ckpt_path(out_dir: Path, arm_name: str, cell: dict, seed: int) -> Path:
    return out_dir / "ckpt" / f"{_ckpt_key(arm_name, cell, seed)}.pt"


def _load_manifest(out_dir: Path) -> Dict[str, dict]:
    path = out_dir / "manifest.json"
    if not path.exists():
        return {}
    return dict(C.read_json(path))


def _save_manifest_entry(out_dir: Path, key: str, entry: dict) -> None:
    """Load-merge-write: reads the CURRENT manifest.json (if any), sets
    `manifest[key] = entry`, and writes the whole thing back atomically via
    `C.write_json` (tmp file + `os.replace`). Not safe against concurrent
    writers, but `run_train` is a single in-process loop. Crash story: the
    ckpt is written (atomically) BEFORE this manifest update, so a crash in
    between leaves a ckpt with no manifest entry -- `run_train`'s resume
    path detects exactly that state and rebuilds the entry from the ckpt's
    own stored `meta` payload (see the self-heal branch there), so the
    window costs nothing on the next run."""
    path = out_dir / "manifest.json"
    manifest = _load_manifest(out_dir)
    manifest[key] = entry
    C.write_json(path, manifest)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """`torch.save` via tmp file + `os.replace` (the same atomicity
    convention as `C.write_json`/`C.write_csv`): a crash mid-save can leave
    only a stray `.tmp.{pid}` file, never a truncated file at the REAL ckpt
    path -- so `run_train`'s exists()-based resume check can trust that an
    existing ckpt file is a complete one (T4-review Important 2a). The
    unreadable-ckpt heal branch in `run_train` still guards against
    pre-atomicity partials and disk-level corruption."""
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Task 7: trainer-registry dispatch (native `train_arm` vs. a registered
# arm's own `train` function).
# ---------------------------------------------------------------------------

def _resolve_trainer(arm_name: str) -> Optional[Callable[..., Tuple[Dict[str, torch.Tensor], dict]]]:
    """Returns a callable `(arm_name, cell, bundles, seed, cfg, epochs) ->
    (state_dict, meta)` for `arm_name` -- i.e. `train_arm`'s OWN signature,
    regardless of whether `arm_name` is native or registered, so
    `run_train`'s single call site (`trainer_fn(arm_name, cell, bundles,
    seed, cfg=cfg, epochs=epochs)`) never needs to branch on that
    distinction itself.

      - Native arms (`arm_name in ARM_NAMES`): always `train_arm` (every
        native arm shares the one matched-recipe trainer by construction).
      - Registered arms WITH a `train` entry (e.g. `hrmv2_act` ->
        `train_hrmv2`): wrapped in a thin adapter that drops the leading
        `arm_name` argument, since a registered trainer already knows which
        arm it trains (`register_provider_builder`'s `train(cell, bundles,
        seed, cfg, epochs)` contract has no `arm_name` parameter -- unlike
        `train_arm`, which dispatches ON `arm_name` internally via
        `build_arm`).
      - Registered arms WITHOUT a `train` entry (the forced-k HRM-v2
        diagnostics): returns `None` -- `run_train` treats this as
        "eval-only, skip" rather than falling through to `train_arm` (which
        would either crash inside `build_arm`'s native-only dispatch, since
        these names aren't in `ARM_NAMES`, or -- if `build_arm`'s registry
        fallback kicked in -- silently train the registered `build()`
        model with the WRONG recipe, an even worse outcome than crashing).
      - Unknown arm names (neither native nor registered at all): returns
        `train_arm` unchanged, preserving the pre-T7 behavior of letting
        `train_arm`/`build_arm` raise their own `ValueError` at call time
        (not this function's job to pre-validate arm existence)."""
    if arm_name in ARM_NAMES:
        return train_arm

    entry = PROVIDER_BUILDERS.get(arm_name)
    if entry is None:
        return train_arm  # Unknown name -- let train_arm/build_arm raise.

    registered_train = entry.get("train")
    if registered_train is None:
        return None  # Eval-only registered arm -- run_train must skip it.

    def _adapter(_arm_name: str, cell: dict, bundles: Sequence[WorldBundle], seed: int,
                 cfg: Optional[C11MissionConfig] = None, epochs: Optional[int] = None
                 ) -> Tuple[Dict[str, torch.Tensor], dict]:
        return registered_train(cell, bundles, seed, cfg, epochs)

    return _adapter


# ---------------------------------------------------------------------------
# Task 4: run_train.
# ---------------------------------------------------------------------------

def run_train(out_dir: str, arms: Sequence[str], cells: Sequence[dict], seeds: Sequence[int],
              cfg: Optional[C11MissionConfig] = None, n_train_worlds: Optional[int] = None,
              epochs: Optional[int] = None) -> None:
    """Train every (arm, cell, seed) triple in `arms x cells x seeds`,
    skipping any whose checkpoint already exists (resume support for the
    full grid -- 132 native + 33 hrm-v2 + 12 scaled runs per the plan/spec's
    run count). Each cell's TRAIN bundles are collected ONCE per cell and
    cached across every (arm, seed) that touches that cell within this call
    (bundles are arm-agnostic -- the same `WorldBundle`s feed every arm's
    `train_arm`, per plan Task 4's "collect once, reuse" instruction), NOT
    re-collected per arm.

    Writes `{out_dir}/ckpt/{arm}__{config_label}{K}__s{seed}.pt`
    (`torch.save({"state_dict": ..., "meta": ...})`, CPU tensors, ATOMIC via
    `_atomic_torch_save`) and a load-merge-write `{out_dir}/manifest.json`
    entry per completed run, keyed `{arm}__{config_label}{K}__s{seed}` (see
    `_ckpt_key`), carrying `train_arm`'s own `meta` dict PLUS `ckpt` (the
    checkpoint's path RELATIVE to `out_dir`, so the manifest is portable if
    `out_dir` itself moves).

    Resume semantics (T4-review Important 2): a run is skipped iff its ckpt
    file exists AND is trusted --
      - ckpt exists + manifest entry exists: plain skip.
      - ckpt exists + manifest entry MISSING (the crash window between the
        ckpt write and the manifest update): the entry is rebuilt from the
        ckpt's own stored `meta` and the run is then skipped -- no retrain,
        and `run_eval` (which discovers learned arms through the manifest)
        sees the arm again.
      - ckpt exists but UNREADABLE (pre-atomicity partial write, disk
        corruption): moved aside to `{name}.pt.corrupt` (kept for
        postmortem, `os.replace` so a stale .corrupt is overwritten) and
        the run RETRAINS -- an unreadable ckpt must never be skipped
        forever.

    Prints one line per completed run (arm/cell/seed, final_loss, wall
    time) AND one line per skipped/healed key (auditable resumed grids,
    T4-review Minor 1)."""
    cfg = cfg or C11MissionConfig()
    n_worlds = int(n_train_worlds) if n_train_worlds is not None else int(cfg.n_train_worlds)
    out_path = Path(out_dir)
    C.ensure_dir(out_path / "ckpt")

    manifest = _load_manifest(out_path)
    bundle_cache: Dict[Tuple[str, int], List[WorldBundle]] = {}

    def _cell_key(cell: dict) -> Tuple[str, int]:
        return (cell["config_label"], int(cell["K"]))

    for cell in cells:
        for arm_name in arms:
            trainer_fn = _resolve_trainer(arm_name)
            if trainer_fn is None:
                # Registered (non-native) arm with no `train` entry --
                # EVAL-ONLY by construction (the forced-k HRM-v2
                # diagnostics: they need `build`/`forward_batch` to work
                # inside `run_eval`, but must NEVER be trained themselves,
                # let alone silently fall through to `train_arm`, which has
                # no idea how to train an arbitrary registered arm). Skip
                # every seed for this arm in one print, not once per seed.
                print(
                    f"[c11-mission train] skip arm {arm_name!r} for cell "
                    f"{cell['config_label']}{cell['K']} (no registered trainer -- "
                    f"eval-only arm)",
                    flush=True,
                )
                continue
            for seed in seeds:
                key = _ckpt_key(arm_name, cell, seed)
                ckpt_path = _ckpt_path(out_path, arm_name, cell, seed)
                if ckpt_path.exists():
                    if key in manifest:
                        print(f"[c11-mission train] skip {key} (checkpoint exists)", flush=True)
                        continue
                    # Crash window: ckpt landed but the manifest update never
                    # ran. Rebuild the entry from the ckpt's own stored meta;
                    # if the ckpt itself won't load (pre-atomicity partial /
                    # corruption -- torch.load can raise RuntimeError,
                    # EOFError, UnpicklingError, BadZipFile, KeyError on a
                    # foreign payload...), treat it as absent: move it aside
                    # and fall through to retraining.
                    try:
                        payload = torch.load(ckpt_path, map_location="cpu")
                        healed_meta = dict(payload["meta"])
                    except Exception as exc:  # noqa: BLE001 -- any load failure means "not a usable ckpt"
                        corrupt_path = ckpt_path.with_suffix(ckpt_path.suffix + ".corrupt")
                        os.replace(ckpt_path, corrupt_path)
                        print(
                            f"[c11-mission train] {key}: unreadable checkpoint "
                            f"({type(exc).__name__}) moved to {corrupt_path.name}; retraining",
                            flush=True,
                        )
                    else:
                        entry = healed_meta
                        entry["ckpt"] = str(ckpt_path.relative_to(out_path))
                        _save_manifest_entry(out_path, key, entry)
                        manifest[key] = entry
                        print(
                            f"[c11-mission train] skip {key} (checkpoint exists; "
                            f"manifest entry rebuilt from ckpt meta)",
                            flush=True,
                        )
                        continue

                ck = _cell_key(cell)
                if ck not in bundle_cache:
                    bundle_cache[ck] = collect_cell_dataset(cell, split="train", n_worlds=n_worlds, cfg=cfg)
                bundles = bundle_cache[ck]

                state_dict, meta = trainer_fn(arm_name, cell, bundles, seed, cfg=cfg, epochs=epochs)

                C.ensure_dir(ckpt_path.parent)
                _atomic_torch_save({"state_dict": state_dict, "meta": meta}, ckpt_path)

                entry = dict(meta)
                entry["ckpt"] = str(ckpt_path.relative_to(out_path))
                _save_manifest_entry(out_path, key, entry)
                manifest[key] = entry

                print(
                    f"[c11-mission train] {arm_name} {cell['config_label']}{cell['K']} s{seed}: "
                    f"final_loss={meta['final_loss']:.4f} wall={meta['wall_s']:.1f}s",
                    flush=True,
                )


# ---------------------------------------------------------------------------
# Task 4: run_eval.
# ---------------------------------------------------------------------------

REFERENCE_ARMS: Tuple[str, ...] = ("h_next", "h_legsum", "h_oracle")

EVAL_RAW_COLS: Tuple[str, ...] = (
    "config", "config_label", "config_idx", "K", "world_idx", "seed", "arm", "train_seed",
    "budget", "found", "cost", "expansions", "closed", "opt_cost",
)


def _reference_h_array(arm_name: str, bundle: WorldBundle) -> np.ndarray:
    """Materialize `h_next`/`h_legsum`/`h_oracle` as a `(K+1, N)` ndarray for
    `astar_product` (which indexes `h[s, i]`) -- `h_oracle` already IS such
    an array (`bundle.oracle`); `h_next`/`h_legsum` are CALLABLES
    (`C11P.h_next(rm, wp)` / `bundle.hl`) that must be densified first."""
    K = len(bundle.wp)
    N = bundle.rm.points.shape[0]
    if arm_name == "h_oracle":
        return bundle.oracle
    if arm_name == "h_legsum":
        fn = bundle.hl
    elif arm_name == "h_next":
        fn = C11P.h_next(bundle.rm, bundle.wp)
    else:
        raise ValueError(f"unknown reference arm {arm_name!r}")
    return np.array([[fn(i, s) for i in range(N)] for s in range(K + 1)], dtype=np.float64)


def _eval_bundle_arm_budget(bundle: WorldBundle, arm_name: str, h_arr: np.ndarray, budget: int,
                             is_reference: bool) -> dict:
    """Run `C11P.astar_product` for one (bundle, arm, budget) and assemble
    the raw partial record dict (missing only the world/cell identity
    columns, which the caller fills in). Integrity asserts: a reference
    (admissible) arm's found cost must equal `opt_cost` within 1e-6 (the
    SAME optimality guarantee `continuous_prm_c11_headroom.eval_cell`
    already asserts for these three arms); a learned (inadmissible) arm's
    found cost need only be >= `opt_cost` - 1e-6 (A* with an inadmissible
    heuristic can still return the optimum, and often does when the
    heuristic under-guides only slightly, but is never REQUIRED to)."""
    opt_cost = float(bundle.oracle[0, 0])
    res = C11P.astar_product(bundle.rm.adj, bundle.wp, h_arr, budget, adj_valid=bundle.adj_valid)

    # Explicit raises, not bare `assert`s: these are data-integrity guards
    # that must survive `python -O` (same convention as this module's
    # `collect_world_bundle`/`_residual_targets` guards).
    if res["found"]:
        if is_reference and abs(res["cost"] - opt_cost) > 1e-6:
            raise AssertionError(
                f"reference arm {arm_name!r} found cost {res['cost']} != opt_cost {opt_cost} "
                f"(admissible heuristics must recover the optimum)"
            )
        if not is_reference and res["cost"] < opt_cost - 1e-6:
            raise AssertionError(
                f"learned arm {arm_name!r} found cost {res['cost']} < opt_cost {opt_cost} - 1e-6 "
                f"(A* cost can never beat the true optimum, admissible heuristic or not)"
            )

    return {
        "arm": arm_name,
        "budget": int(budget),
        "found": bool(res["found"]),
        "cost": float(res["cost"]),
        "expansions": int(res["expansions"]),
        "closed": int(res["closed"]),
        "opt_cost": opt_cost,
    }


def run_eval(out_dir: str, cells: Sequence[dict], cfg: Optional[C11MissionConfig] = None,
             n_test_worlds: Optional[int] = None, budgets: Optional[Sequence[int]] = None,
             arms: Optional[Sequence[str]] = None) -> None:
    """Per cell: collect TEST bundles ONCE (probe-native seeds/skip
    semantics via `collect_cell_dataset(cell, split="test", ...)`). Reference
    arms (`h_next`, `h_legsum`, `h_oracle`) run on EVERY bundle x budget.
    Learned arms: every `{out_dir}/manifest.json` entry whose `config_label`/
    `K` match the cell (filtered to `arms` if given) is loaded via
    `make_provider` and run on every bundle x budget too. Every arm --
    reference or learned -- goes through the SAME `C11P.astar_product(
    bundle.rm.adj, bundle.wp, h_array, budget, adj_valid=bundle.adj_valid)`
    call (single astar path, per plan Task 4's self-review question).

    `budgets` defaults to `cfg.budgets_grid` (the full grid for every arm --
    A* is ms-fast, so this is cheap and gives dose-response curves for free,
    per plan Task 4 spec). `arms=None` means "every learned arm present in
    the manifest for this cell"; `arms=()` (the empty tuple, NOT None) means
    "no learned arms at all" -- used by the K=0 binding-calibration path,
    which only needs the reference arms' records.

    K=0 cells: after that cell's `h_legsum` records exist, calibrates the
    binding budget via `C11P.calibrate_binding_budget(legsum_records,
    cfg.budgets_grid)` and merges `{config_label}_K0 -> {"config_label",
    "K", "budget", "degenerate"}` into `{out_dir}/binding_k0.json` (load-
    merge-write, same convention as the training manifest). K in {2,4,8}
    cells are NOT touched here -- their budgets are the pre-registered
    `cfg.binding_budgets`, already fixed at config-time.

    Writes every record (one dict per bundle x arm x budget) to
    `{out_dir}/results/c11_eval_raw.csv` via `C.write_csv` (csv.DictWriter
    under the hood). Deterministic ordering: cells in the given `cells`
    order, worlds ascending (bundle collection order == world_idx order,
    per `collect_cell_dataset`), arms sorted (reference arms first in a
    fixed tuple order, then learned arms sorted by name), budgets
    ascending.

    Task 8 side-channel dumps (review-derived integration requirements 2/3
    -- see the plan/task prompt's "Halt-steps dumping"/"State-MAE dumping"
    sections): immediately after EVERY learned-provider call (`h_arr =
    provider(bundle)`, ONE call per (world, arm) regardless of how many
    budgets are evaluated against that h_arr), BEFORE the next provider
    runs on the SAME bundle:
      - if `hasattr(bundle, "hrmv2_halt_steps")` (arm-agnostic trigger --
        the attribute only ever appears from an ACT-live provider's
        `_stash_halt_steps`, so keying off its presence rather than the
        literal name "hrmv2_act" also picks up any other ACT-live-style
        provider without a hardcoded name check): buffers one row
        `(config_label, K, world_idx, seed, train_seed, arm,
        mean_halt_steps, n_queried)` -- mean over the non-NaN entries of
        the `(K+1, N)` array; `arm` attributes the row to the provider
        that produced it (T8-review minor 1), since the trigger itself is
        arm-agnostic -- then `del bundle.hrmv2_halt_steps` immediately (per the
        task prompt's explicit choice of "del" over an overwrite-tolerant
        read: the attribute is "last writer per cell" per its own
        docstring, so leaving it would let a LATER, unrelated provider's
        dump on the SAME bundle silently inherit stale cells from an
        earlier model -- deleting closes that window at its narrowest
        point, right after this provider's own read).
      - for EVERY learned-arm provider call (reference arms `h_next`/
        `h_legsum`/`h_oracle` are skipped -- they have no residual to
        score and are never "learned"): computes `mean(|h_arr -
        bundle.oracle|)` over the finite-oracle mask (`oracle < C.INF /
        10.0`, the same mask `_residual_targets` uses), divided by
        `bundle.world.side_len`, and buffers a row `(config_label, K,
        world_idx, seed, train_seed, arm, state_mae)`.
    Both buffers are written ONCE at the end (same convention as
    `all_records`/`C.write_csv`, which overwrites rather than appends) to
    `{out_dir}/results/c11_halt_steps.csv` / `{out_dir}/results/
    c11_state_mae.csv` respectively -- SKIPPED (no file write) if the
    corresponding buffer ended up empty (e.g. an eval run with no hrmv2/
    learned arms at all), so `run_analyze` can distinguish "ran with zero
    qualifying rows" from "never ran" only by file absence, which is
    exactly the signal it needs for its "missing halt/mae files -> G2
    marked not run" contract."""
    cfg = cfg or C11MissionConfig()
    n_worlds = int(n_test_worlds) if n_test_worlds is not None else int(cfg.n_test_worlds)
    eval_budgets = tuple(sorted(int(b) for b in (budgets if budgets is not None else cfg.budgets_grid)))
    out_path = Path(out_dir)

    manifest = _load_manifest(out_path)

    all_records: List[dict] = []
    binding_k0_updates: Dict[str, dict] = {}
    halt_step_rows: List[dict] = []
    state_mae_rows: List[dict] = []

    for cell in cells:
        config_label = cell["config_label"]
        config_idx = int(cell["config_idx"])
        K = int(cell["K"])
        spec_name = cell["spec_name"]

        bundles = collect_cell_dataset(cell, split="test", n_worlds=n_worlds, cfg=cfg)

        # Learned-arm providers for this cell: every manifest entry whose
        # config_label/K match, optionally filtered to `arms`.
        cell_entries = [
            (key, entry) for key, entry in manifest.items()
            if entry.get("config_label") == config_label and int(entry.get("K", -1)) == K
        ]
        if arms is not None:
            allowed = set(arms)
            cell_entries = [(key, entry) for key, entry in cell_entries if entry.get("arm") in allowed]
        cell_entries.sort(key=lambda ke: ke[0])

        providers: List[Tuple[str, Callable[[WorldBundle], np.ndarray], int]] = []
        for key, entry in cell_entries:
            ckpt_path = out_path / entry["ckpt"]
            payload = torch.load(ckpt_path, map_location="cpu")
            arm_name = entry["arm"]
            provider = make_provider(arm_name, payload["state_dict"], cfg)
            providers.append((arm_name, provider, int(entry["seed"])))

        # T7: ckpt_alias resolution -- eval-only registered arms (the
        # forced-k HRM-v2 diagnostics: `hrmv2_act_k{1,2,4,8}`) have NO
        # manifest entry of their own (they are never trained -- see
        # `_resolve_trainer`/`run_train`'s eval-only skip), so the loop
        # above never discovers them. For every `arms is None` request (or
        # a request that explicitly includes an aliased name), find every
        # PROVIDER_BUILDERS entry with a `ckpt_alias` set, and -- for each
        # of the ALIASED arm's manifest entries in THIS cell (one per
        # training seed) -- build a provider under the ALIASING arm's own
        # name (so its registered `forward_batch`, e.g. the forced-k path,
        # is what actually runs) but loaded from the ALIASED arm's
        # checkpoint (so `hrmv2_act_k4` evaluates the SAME trained weights
        # as `hrmv2_act`, per spec: "all five share the SAME checkpoint").
        # Records from these providers still carry `arm=hrmv2_act_kN` (the
        # aliasing arm's own name, from `arm_name` below), never the
        # alias's name -- `make_provider` is called with `arm_name`
        # (aliasing), not the entry's own `entry["arm"]` (aliased), so
        # `build_arm`/`_forward_arm_batch` dispatch to the ALIASING arm's
        # registered `build`/`forward_batch` while the loaded weights come
        # from the ALIASED arm's state_dict.
        alias_names = [
            name for name, entry in PROVIDER_BUILDERS.items()
            if entry.get("ckpt_alias") is not None
        ]
        if arms is not None:
            alias_names = [n for n in alias_names if n in set(arms)]
        alias_names.sort()
        for aliasing_arm in alias_names:
            aliased_arm = PROVIDER_BUILDERS[aliasing_arm]["ckpt_alias"]
            aliased_entries = [
                (key, entry) for key, entry in manifest.items()
                if entry.get("config_label") == config_label and int(entry.get("K", -1)) == K
                and entry.get("arm") == aliased_arm
            ]
            aliased_entries.sort(key=lambda ke: ke[0])
            for key, entry in aliased_entries:
                ckpt_path = out_path / entry["ckpt"]
                payload = torch.load(ckpt_path, map_location="cpu")
                provider = make_provider(aliasing_arm, payload["state_dict"], cfg)
                providers.append((aliasing_arm, provider, int(entry["seed"])))

        legsum_records_for_calib: List[dict] = []

        for world_idx, bundle in enumerate(bundles):
            arm_h_arrays: List[Tuple[str, np.ndarray, bool, str]] = []
            for arm_name in REFERENCE_ARMS:
                h_arr = _reference_h_array(arm_name, bundle)
                arm_h_arrays.append((arm_name, h_arr, True, ""))
            for arm_name, provider, train_seed_val in providers:
                h_arr = provider(bundle)
                arm_h_arrays.append((arm_name, h_arr, False, str(train_seed_val)))

                # T8 requirement 2: read-then-delete the halt-steps side
                # channel IMMEDIATELY after this provider call, before the
                # NEXT provider runs on this same bundle (arm-agnostic
                # trigger -- see this function's own docstring for the
                # "last writer per cell" staleness rationale).
                halt_arr = getattr(bundle, "hrmv2_halt_steps", None)
                if halt_arr is not None:
                    finite_halt = halt_arr[np.isfinite(halt_arr)]
                    mean_halt = float(np.mean(finite_halt)) if finite_halt.size > 0 else float("nan")
                    halt_step_rows.append({
                        "config_label": config_label,
                        "K": K,
                        "world_idx": world_idx,
                        "seed": bundle.seed,
                        "train_seed": str(train_seed_val),
                        # T8-review minor 1: attribute the row to the arm
                        # whose provider call produced it -- the dump
                        # TRIGGER is deliberately arm-agnostic (hasattr),
                        # so without this column a future SECOND ACT-live
                        # provider's rows would pool indistinguishably
                        # into G2b's hrmv2_act correlation.
                        "arm": arm_name,
                        "mean_halt_steps": mean_halt,
                        "n_queried": int(finite_halt.size),
                    })
                    del bundle.hrmv2_halt_steps

                # T8 requirement 3: per-call state-MAE, every learned arm
                # (reference arms are skipped -- they have no residual to
                # score, see this function's own docstring).
                finite_mask = bundle.oracle < C.INF / 10.0
                if np.any(finite_mask):
                    side = float(bundle.world.side_len)
                    mae = float(np.mean(np.abs(h_arr - bundle.oracle)[finite_mask])) / side
                else:
                    mae = float("nan")
                state_mae_rows.append({
                    "config_label": config_label,
                    "K": K,
                    "world_idx": world_idx,
                    "seed": bundle.seed,
                    "train_seed": str(train_seed_val),
                    "arm": arm_name,
                    "state_mae": mae,
                })

            for arm_name, h_arr, is_reference, train_seed_str in arm_h_arrays:
                for budget in eval_budgets:
                    partial = _eval_bundle_arm_budget(bundle, arm_name, h_arr, budget, is_reference)
                    record = {
                        "config": spec_name,
                        "config_label": config_label,
                        "config_idx": config_idx,
                        "K": K,
                        "world_idx": world_idx,
                        "seed": bundle.seed,
                        "train_seed": train_seed_str,
                        **partial,
                    }
                    all_records.append(record)
                    if arm_name == "h_legsum":
                        legsum_records_for_calib.append(record)

        if K == 0:
            budget_val, degenerate = C11P.calibrate_binding_budget(legsum_records_for_calib, cfg.budgets_grid)
            binding_k0_updates[f"{config_label}_K0"] = {
                "config_label": config_label,
                "K": 0,
                "budget": int(budget_val),
                "degenerate": bool(degenerate),
            }

    # Column order: EVAL_RAW_COLS first (deterministic schema), any surplus
    # keys (none expected today) appended by C.write_csv's own first-seen
    # fallback.
    ordered_records = [
        {col: r[col] for col in EVAL_RAW_COLS} for r in all_records
    ]
    C.write_csv(out_path / "results" / "c11_eval_raw.csv", ordered_records)

    # T8 requirements 2/3: side-dump CSVs, skipped entirely (no file) when
    # the buffer is empty -- see this function's docstring for why file
    # ABSENCE (not an empty-but-present file) is the signal `run_analyze`
    # relies on to mark G2 sections "not run".
    if halt_step_rows:
        C.write_csv(out_path / "results" / "c11_halt_steps.csv", halt_step_rows)
    if state_mae_rows:
        C.write_csv(out_path / "results" / "c11_state_mae.csv", state_mae_rows)

    if binding_k0_updates:
        binding_path = out_path / "binding_k0.json"
        existing = C.read_json(binding_path) if binding_path.exists() else {}
        existing.update(binding_k0_updates)
        C.write_json(binding_path, existing)


# ---------------------------------------------------------------------------
# Task 5: analyze mode -- pure G1 stats (arm-vs-MLP paired comparison, BH
# correction, K=0 continuity control, G1 dose-response verdict).
#
# Everything below is a PURE function over the raw eval-CSV records (a list
# of dicts, string-typed exactly as `csv.DictReader` hands them back from
# `run_eval`'s `c11_eval_raw.csv`) -- no I/O, no world/PRM building, no
# training. This mirrors the C-series "analyzer is a pure function over the
# raw CSV" convention (spec section 6) and is why this task is entirely
# synthetic-testable: every test below hand-builds records or hand-builds
# `analysis` dicts directly, never touching `collect_world_bundle`/
# `run_train`/`run_eval`.
#
# `_mcnemar_exact_p` / `_bh` are COPIED from `continuous_prm_c6_heatmap_
# value_field.mcnemar_exact_p` / `.bh_q_values` (verified byte-identical
# algorithm) rather than imported, per the plan's "phase modules must not
# import each other" convention -- C9's `continuous_prm_c9_transfer.py` and
# C9b's `continuous_prm_c9b_dynamics_transfer.py` both import those same two
# functions directly from C6 (a shared utility module, not a phase module),
# so importing from C6 here would be consistent with that precedent; the
# plan's instruction is nonetheless to copy with attribution, so that is
# what's done. `_bootstrap_ratio_ci` mirrors `continuous_prm_c7_integration_
# compare.bootstrap_median_ci`'s exact resampling convention (percentile CI
# on the median via `np.random.default_rng(seed)`) and IS imported directly
# from that (non-phase, shared-utility) module, matching C9/C9b's own usage
# of it.
# ---------------------------------------------------------------------------

from continuous_prm_c7_integration_compare import bootstrap_median_ci as _bootstrap_median_ci


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided binomial McNemar p-value for discordant counts
    (b, c). Copied from `continuous_prm_c6_heatmap_value_field.
    mcnemar_exact_p` (attribution per plan Task 5): n = b + c, k = min(b, c),
    prob = sum_{i=0}^{k} C(n, i) * 0.5^n, p = min(1.0, 2*prob). n == 0 (no
    discordant pairs at all) returns 1.0 -- there is no evidence of a
    difference when nothing disagrees."""
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    prob = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * prob))


def _bh(pvals: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg q-values, order-preserving (q[i] corresponds to
    pvals[i], NOT sorted). Copied from `continuous_prm_c6_heatmap_value_
    field.bh_q_values` (attribution per plan Task 5): q_(m) = p_(m); walking
    from the largest p-value down to the smallest, q_(i) = min(q_(i+1),
    p_(i) * m / i) -- i.e. q_i = min over j >= i of p_(j) * m / j, exactly
    the pre-registered definition."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [1.0] * m
    running = 1.0
    for rank_from_end, idx in enumerate(reversed(order), start=1):
        rank = m - rank_from_end + 1
        val = float(pvals[idx]) * m / max(1, rank)
        running = min(running, val)
        q[idx] = min(1.0, running)
    return q


def _coerce_record(row: dict) -> dict:
    """Coerce one raw eval record (string-typed, as read back from
    `c11_eval_raw.csv` via `csv.DictReader`, OR already-typed if the caller
    built it directly) into a canonically-typed dict. The ONE place this
    coercion happens (plan Task 5: "write ONE `_coerce_record` helper used
    everywhere"). Idempotent: re-coercing an already-coerced record is a
    no-op, since `bool`/`int`/`float` pass through their own constructors
    unchanged and the found-parsing branch only string-parses `str` inputs."""
    found_val = row["found"]
    if isinstance(found_val, str):
        found = found_val.strip().lower() in ("true", "1", "yes")
    else:
        found = bool(found_val)
    return {
        "config": str(row["config"]),
        "config_label": str(row["config_label"]),
        "config_idx": int(row["config_idx"]),
        "K": int(row["K"]),
        "world_idx": int(row["world_idx"]),
        "seed": str(row["seed"]),
        "arm": str(row["arm"]),
        "train_seed": str(row["train_seed"]),
        "budget": int(row["budget"]),
        "found": found,
        "cost": float(row["cost"]),
        "expansions": int(row["expansions"]),
        "closed": int(row["closed"]),
        "opt_cost": float(row["opt_cost"]),
    }


def analyze(records: Sequence[dict], binding: Dict[Tuple[str, int], int],
            cfg: Optional[C11MissionConfig] = None) -> dict:
    """Pure analysis over raw eval records -- no I/O, no world/PRM/A* calls.

    `records`: list of dicts as read back from `c11_eval_raw.csv` (or
    already-typed equivalents); coerced uniformly via `_coerce_record`.
    `binding`: `{(config_label, K): budget}` for every cell present --
    assembled by the caller from `cfg.binding_budgets` (K in {2,4,8}) plus
    `binding_k0.json` (K=0); this function does NOT assemble it (that is
    T8's job, per the plan).

    Restricts EVERY per-cell computation to that cell's binding-budget rows
    ONLY (a world found at a non-binding budget but not at the binding
    budget counts as not-found -- the binding budget is the sole source of
    truth for a cell's "found" status, matching the probe/C-series
    convention of reading dose-response curves but reporting headline
    numbers at one calibrated budget per cell).

    Per cell (config_label, K), returns:
        {"binding": budget,
         "success": {arm: rate},
         "vs_mlp": {arm: {ratio_median, ci_lo, ci_hi, n_pairs, mcnemar_p,
                           mcnemar_q, b, c}},           # arm != "mlp" only
         "vs_legsum": {arm: {ratio_median, n}},         # every arm incl. mlp
         "oracle_vs_legsum": {ratio_median, n}}
    keyed by `(config_label, K)`."""
    cfg = cfg or C11MissionConfig()
    coerced = [_coerce_record(r) for r in records]

    by_cell: Dict[Tuple[str, int], List[dict]] = {}
    for r in coerced:
        by_cell.setdefault((r["config_label"], r["K"]), []).append(r)

    result: Dict[Tuple[str, int], dict] = {}

    for cell_key, cell_binding in binding.items():
        cell_key = (str(cell_key[0]), int(cell_key[1]))
        cell_rows = by_cell.get(cell_key, [])
        binding_rows = [r for r in cell_rows if r["budget"] == int(cell_binding)]

        # -- success rates, per arm. Learned arms: mean over (world x
        # train_seed) of found. Reference arms (train_seed == ""): mean
        # over worlds (there is exactly one record per world for these).
        by_arm: Dict[str, List[dict]] = {}
        for r in binding_rows:
            by_arm.setdefault(r["arm"], []).append(r)
        success: Dict[str, float] = {
            arm: float(np.mean([1.0 if r["found"] else 0.0 for r in rows]))
            for arm, rows in by_arm.items()
        }

        # -- pairing indices. Learned-arm rows keyed by (world_idx,
        # train_seed) since that IS the pre-registered pairing unit.
        # Reference-arm rows (h_legsum, h_oracle) keyed by world_idx alone
        # (train_seed == "" for all of them; one row per world). Both
        # builders raise on a duplicate key: a dict comprehension would
        # silently last-win, and a duplicated (world, seed) record in the
        # CSV is data corruption that must be loud in the gate-rendering
        # layer, not absorbed.
        def _by_world_seed(arm_name: str) -> Dict[Tuple[int, str], dict]:
            out: Dict[Tuple[int, str], dict] = {}
            for r in by_arm.get(arm_name, []):
                key = (r["world_idx"], r["train_seed"])
                if key in out:
                    raise AssertionError(
                        f"duplicate record for arm {arm_name!r} at (world_idx, train_seed)="
                        f"{key} in cell {cell_key} at binding budget {int(cell_binding)} "
                        f"(a dict build would silently keep only the last one)"
                    )
                out[key] = r
            return out

        def _by_world(arm_name: str) -> Dict[int, dict]:
            out: Dict[int, dict] = {}
            for r in by_arm.get(arm_name, []):
                wi = r["world_idx"]
                if wi in out:
                    raise AssertionError(
                        f"duplicate record for reference arm {arm_name!r} at world_idx={wi} "
                        f"in cell {cell_key} at binding budget {int(cell_binding)} "
                        f"(a dict build would silently keep only the last one)"
                    )
                out[wi] = r
            return out

        mlp_by_ws = _by_world_seed("mlp")
        legsum_by_world = _by_world("h_legsum")
        oracle_by_world = _by_world("h_oracle")

        learned_arms = sorted(a for a in by_arm if a not in REFERENCE_ARMS and a != "mlp")

        # -- primary: arm-vs-MLP paired comparison, one per learned arm.
        vs_mlp_pvals: List[float] = []
        vs_mlp_stats: Dict[str, dict] = {}
        for arm_name in learned_arms:
            arm_by_ws = _by_world_seed(arm_name)

            # Key-set-parity guard: a complete `run_eval` CSV carries EXACTLY
            # one arm record AND one mlp record per (world_idx, train_seed)
            # at the binding budget (every provider runs on every bundle x
            # budget). A key present on one side but missing on the other is
            # a malformed/partial CSV (crashed run, truncated file, mixed
            # manifests) -- silently counting the missing side as "not
            # found" would corrupt b/c and the pair set, so it is a loud
            # error in this gate-rendering layer instead. (A record whose
            # `found` is False is NOT a parity violation -- the record
            # exists; only a MISSING record is.)
            missing_mlp = sorted(set(arm_by_ws) - set(mlp_by_ws))
            missing_arm = sorted(set(mlp_by_ws) - set(arm_by_ws))
            if missing_mlp or missing_arm:
                raise AssertionError(
                    f"arm-vs-mlp key-set parity violation in cell {cell_key} at binding "
                    f"budget {int(cell_binding)} for arm {arm_name!r}: (world_idx, "
                    f"train_seed) keys with no mlp record: {missing_mlp}; keys with no "
                    f"{arm_name!r} record: {missing_arm}"
                )

            ratios: List[float] = []
            b_count = 0  # arm found, mlp not (discordant).
            c_count = 0  # mlp found, arm not (discordant).
            for key in sorted(arm_by_ws):
                arm_row = arm_by_ws[key]
                mlp_row = mlp_by_ws[key]
                arm_found = bool(arm_row["found"])
                mlp_found = bool(mlp_row["found"])
                if arm_found and mlp_found:
                    ratios.append(float(arm_row["expansions"]) / float(mlp_row["expansions"]))
                elif arm_found and not mlp_found:
                    b_count += 1
                elif mlp_found and not arm_found:
                    c_count += 1

            med, ci_lo, ci_hi = _bootstrap_median_ci(ratios, seed=12345, n_boot=1000, lo=2.5, hi=97.5)
            p = _mcnemar_exact_p(b_count, c_count)
            vs_mlp_pvals.append(p)
            vs_mlp_stats[arm_name] = {
                "ratio_median": med,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_pairs": len(ratios),
                "mcnemar_p": p,
                "b": b_count,
                "c": c_count,
            }

        qvals = _bh(vs_mlp_pvals)
        for arm_name, q in zip(learned_arms, qvals):
            vs_mlp_stats[arm_name]["mcnemar_q"] = q

        # -- secondary: every arm (incl. mlp) vs h_legsum. Pairing: each
        # learned (world, train_seed) row against legsum's PER-WORLD row
        # (legsum has exactly one record per world -- no seed dimension).
        vs_legsum: Dict[str, dict] = {}
        for arm_name in sorted(a for a in by_arm if a not in REFERENCE_ARMS):
            arm_by_ws = _by_world_seed(arm_name)
            ratios = []
            for (world_idx, _train_seed), arm_row in arm_by_ws.items():
                ls_row = legsum_by_world.get(world_idx)
                if ls_row is None:
                    continue
                if bool(arm_row["found"]) and bool(ls_row["found"]):
                    ratios.append(float(arm_row["expansions"]) / float(ls_row["expansions"]))
            med, _lo, _hi = _bootstrap_median_ci(ratios, seed=12345, n_boot=1000, lo=2.5, hi=97.5)
            vs_legsum[arm_name] = {"ratio_median": med, "n": len(ratios)}

        # -- oracle ceiling row: h_oracle vs h_legsum, matched per-world
        # (both are reference arms, one record per world, no seed dim).
        oracle_ratios = []
        for world_idx, or_row in oracle_by_world.items():
            ls_row = legsum_by_world.get(world_idx)
            if ls_row is None:
                continue
            if bool(or_row["found"]) and bool(ls_row["found"]):
                oracle_ratios.append(float(or_row["expansions"]) / float(ls_row["expansions"]))
        o_med, _o_lo, _o_hi = _bootstrap_median_ci(oracle_ratios, seed=12345, n_boot=1000, lo=2.5, hi=97.5)
        oracle_vs_legsum = {"ratio_median": o_med, "n": len(oracle_ratios)}

        result[cell_key] = {
            "binding": int(cell_binding),
            "success": success,
            "vs_mlp": vs_mlp_stats,
            "vs_legsum": vs_legsum,
            "oracle_vs_legsum": oracle_vs_legsum,
        }

    return result


def k0_continuity(analysis: dict) -> dict:
    """G1's continuity control (spec section 1: "at K=0 the task reduces
    exactly to C7 ... and all arms must statistically tie -- if they don't,
    it is a formulation/implementation bug and the phase halts for
    diagnosis, not a result").

    Over K=0 cells ONLY: PASS iff NO learned arm's vs_mlp comparison shows
    separation from the MLP control, where separation = (mcnemar_q < 0.05)
    OR (the ratio bootstrap CI entirely excludes 1.0, i.e. ci_hi < 1.0 or
    ci_lo > 1.0). This is the pre-registered C7-continuity control: it gates
    whether G1 is readable at all, so it is checked FIRST and independently
    of the dose-response verdict itself.

    Returns {"pass": bool, "violations": [(config_label, arm, reason)]}."""
    violations: List[Tuple[str, str, str]] = []
    for (config_label, K), cell in analysis.items():
        if K != 0:
            continue
        for arm_name, stats in cell.get("vs_mlp", {}).items():
            reasons = []
            if stats["mcnemar_q"] < 0.05:
                reasons.append(f"mcnemar_q={stats['mcnemar_q']:.4g} < 0.05")
            # (n_pairs == 1 collapses the bootstrap CI to (med, med, med), so
            # a single separated pair CAN flag a violation -- deliberate; see
            # the matching degenerate-CI note in `g1_verdict`.)
            if stats["ci_hi"] < 1.0:
                reasons.append(f"ci_hi={stats['ci_hi']:.4g} < 1.0")
            elif stats["ci_lo"] > 1.0:
                reasons.append(f"ci_lo={stats['ci_lo']:.4g} > 1.0")
            if reasons:
                violations.append((config_label, arm_name, "; ".join(reasons)))
    return {"pass": len(violations) == 0, "violations": violations}


# The pre-registered G1 arm family: the structured arms compared against
# the MLP control for the dose-response verdict (spec section 1). NOTE:
# `g1_verdict` intersects THIS hardcoded tuple with the arms actually
# present in the analysis dict -- an arm absent from this tuple is
# invisible to G1 no matter what records exist for it. Task 7 MUST append
# "hrmv2_act" here when that arm lands (the ACT-live provider is a G1
# participant); the forced-k variants ("hrmv2_act_k1"/"_k2"/"_k4"/"_k8")
# are G2-only diagnostics and must NEVER enter this tuple.
G1_STRUCTURED_ARM_NAMES: Tuple[str, ...] = ("unet_film", "gnn", "hrm_trace", "onlstm_trace", "hrmv2_act")

G1_K_VALUES: Tuple[int, ...] = (2, 4, 8)


def g1_verdict(analysis: dict) -> dict:
    """G1 verdict logic (spec section 1, quoted verbatim):

    "G1 -- dose-response in structure. Primary read: per (config, K), each
    structured arm vs the MLP control on matched per-world expansion-ratio
    differences (both-solved worlds) + success McNemar. Continuity control:
    at K=0 the task reduces exactly to C7 (legsum == euclid, no mission
    tokens beyond the goal leg) and all arms must statistically tie -- if
    they don't, it is a formulation/implementation bug and the phase halts
    for diagnosis, not a result. Positive verdict: >=1 structured arm beats
    the MLP with BH q < 0.05 at >=2 of the 3 K in {2,4,8} values on >=1
    config, with the gap monotone non-decreasing in K. Anything less is a
    negative for that arm."

    (Note: "monotone non-decreasing in K" in the spec's prose describes the
    GAP growing with depth, i.e. the arm-vs-MLP expansion RATIO shrinking
    toward/below 1 as K increases -- the plan's pre-registered operational
    definition below makes this precise: "the ratio_median sequence over
    the K values where n_pairs >= 1 is monotone non-increasing".)

    Over K in {2,4,8} cells: for each structured arm (`unet_film`, `gnn`,
    `hrm_trace`, `onlstm_trace` -- restricted to arms actually present in
    `analysis`, so an incomplete run doesn't KeyError) and config: a "beat"
    at (config, K) = (mcnemar_q < 0.05 AND success_arm > success_mlp) OR
    (ratio ci_hi < 1.0). Arm is POSITIVE on a config iff beats at >= 2 of
    the 3 K values AND the ratio_median sequence over the K values where
    n_pairs >= 1 is monotone non-increasing (gap grows with depth). Cells
    with n_pairs == 0 contribute no beat and break monotonicity assessment
    only if they leave < 2 usable K values (then that (arm, config) is
    "insufficient", not positive).

    Returns {"positive": bool, "positive_arms": [(arm, config_label)],
             "table": {(arm, config_label): {"beats": {K: bool},
                                              "ratio_by_k": {K: ratio_median or None},
                                              "monotone": bool or None,
                                              "n_beats": int,
                                              "status": "positive"|"negative"|"insufficient"}}}."""
    configs = sorted({key[0] for key in analysis if key[1] in G1_K_VALUES})
    present_arms = set()
    for key, cell in analysis.items():
        if key[1] in G1_K_VALUES:
            present_arms.update(cell.get("vs_mlp", {}).keys())
    arm_names = [a for a in G1_STRUCTURED_ARM_NAMES if a in present_arms]

    table: Dict[Tuple[str, str], dict] = {}
    positive_arms: List[Tuple[str, str]] = []

    for arm_name in arm_names:
        for config_label in configs:
            beats: Dict[int, bool] = {}
            ratio_by_k: Dict[int, Optional[float]] = {}
            for K in G1_K_VALUES:
                cell = analysis.get((config_label, K))
                if cell is None:
                    ratio_by_k[K] = None
                    continue
                stats = cell.get("vs_mlp", {}).get(arm_name)
                if stats is None or stats["n_pairs"] == 0:
                    ratio_by_k[K] = None
                    continue
                ratio_by_k[K] = stats["ratio_median"]
                success = cell.get("success", {})
                # n_pairs == 1 degenerate-CI caveat: `bootstrap_median_ci`
                # collapses to (med, med, med) for a single value, so ONE
                # both-found pair with ratio < 1.0 constitutes a "beat" here
                # (and, symmetrically, a violation in `k0_continuity`) --
                # inherited from the C7 bootstrap convention, kept
                # deliberately rather than special-cased.
                beat = (
                    (stats["mcnemar_q"] < 0.05 and success.get(arm_name, 0.0) > success.get("mlp", 0.0))
                    or (stats["ci_hi"] < 1.0)
                )
                beats[K] = bool(beat)

            n_beats = sum(1 for v in beats.values() if v)
            usable_ks = [K for K in G1_K_VALUES if ratio_by_k.get(K) is not None]
            usable_ratios = [ratio_by_k[K] for K in usable_ks]
            monotone = None
            if len(usable_ks) >= 2:
                # 1e-12 is float-hygiene tolerance ONLY: "monotone
                # non-increasing" must not flip on ulp-level noise between
                # medians that are equal in exact arithmetic; it is far too
                # small to mask any real (statistically meaningful) increase.
                monotone = all(
                    usable_ratios[i] >= usable_ratios[i + 1] - 1e-12
                    for i in range(len(usable_ratios) - 1)
                )

            if len(usable_ks) < 2:
                status = "insufficient"
            elif n_beats >= 2 and monotone:
                status = "positive"
            else:
                status = "negative"

            table[(arm_name, config_label)] = {
                "beats": beats,
                "ratio_by_k": ratio_by_k,
                "monotone": monotone,
                "n_beats": n_beats,
                "status": status,
            }
            if status == "positive":
                positive_arms.append((arm_name, config_label))

    return {
        "positive": len(positive_arms) > 0,
        "positive_arms": positive_arms,
        "table": table,
    }


# ---------------------------------------------------------------------------
# Task 8: G2 analysis -- forced-segment curves (G2a) + halting-vs-K
# correlation (G2b). Both are PURE functions over already-collected rows
# (the raw eval records for G2a's expansion-ratio half, plus the
# `c11_state_mae.csv` / `c11_halt_steps.csv` side-dump rows `run_eval`
# writes per T8's requirements 2/3) -- no I/O, no world/PRM/A* calls,
# mirroring `analyze`'s own "pure function over the raw CSV" convention.
#
# Spec section 1, G2 (quoted verbatim):
# "G2 -- depth-of-compute (HRM-v2 mechanism). (a) Forced-segment curves:
# eval the trained ACT arm at forced k in {1,2,4,8} segments; positive iff
# quality (state-level MAE and/or expansion-ratio) improves monotonically
# with bootstrap-CI separation between k=1 and k=8 on >=1 config at K=8.
# (b) Learned halting: Spearman correlation of mean halt-steps vs K
# positive with q < 0.05 (\"thinks longer on deeper missions\")."
# ---------------------------------------------------------------------------

# The forced-k diagnostic arm family (G2a): `hrmv2_act_k1`/`_k2`/`_k4`/`_k8`
# (forced-segment inference, no q-halt) plus `hrmv2_act` itself as the
# ACT-live reference point (its OWN learned halting decides how many
# segments run, so it is not "forced" at any particular k -- included in
# the curve as a free-running comparison point, per the task spec: "+
# `hrmv2_act` as the ACT-live reference point").
G2A_FORCED_K_ARMS: Tuple[str, ...] = ("hrmv2_act_k1", "hrmv2_act_k2", "hrmv2_act_k4", "hrmv2_act_k8")
G2A_ARM_FAMILY: Tuple[str, ...] = G2A_FORCED_K_ARMS + ("hrmv2_act",)

# Forced-k arm name -> its k value (hrmv2_act has NO fixed k -- excluded
# from the ratio-monotonicity/mae-monotonicity checks below, which are
# defined only over the 4 FORCED values; it still gets its own row in the
# per-cell table for reporting).
_FORCED_K_VALUE: Dict[str, int] = {"hrmv2_act_k1": 1, "hrmv2_act_k2": 2, "hrmv2_act_k4": 4, "hrmv2_act_k8": 8}


def g2_analysis(records: Sequence[dict], mae_rows: Sequence[dict],
                 binding: Dict[Tuple[str, int], int], cfg: Optional[C11MissionConfig] = None) -> dict:
    """G2a: per (config_label, K) cell at that cell's binding budget, for
    every arm in `G2A_ARM_FAMILY` present in `records`: success rate,
    matched expansion-ratio vs `h_legsum` (paired per-(world_idx,
    train_seed), both-found rows only, median + bootstrap CI 1000 resamples
    seeded 12345 -- SAME convention as `analyze`'s `vs_legsum` block), and
    mean `state_mae` from `mae_rows` for that (config_label, K, arm) (over
    every mae_rows entry matching config_label/K/arm, regardless of K in
    mae_rows -- mae_rows are typically all logged at K=8 per T8 requirement
    3's "for EVERY learned-arm provider call", but this function does not
    assume that; it groups by whatever (config_label, K, arm) keys are
    actually present).

    `records`/`mae_rows`: raw-CSV-style rows (string-typed, as read back by
    `csv.DictReader`, OR already-typed) -- coerced via `_coerce_record` /
    a local mae-row coercion respectively. `binding`: `{(config_label, K):
    budget}`, same contract as `analyze`'s `binding` argument (the caller
    assembles it; this function does not).

    Returns `{"cells": {(config_label, K): {"binding": budget, "arms":
    {arm: {"success", "ratio_median", "ci_lo", "ci_hi", "n_pairs",
    "mean_state_mae", "n_mae"}}}}}` -- the caller (`g2a_verdict`) reduces
    this into the pre-registered positive/negative/insufficient verdict."""
    cfg = cfg or C11MissionConfig()
    coerced = [_coerce_record(r) for r in records]

    def _coerce_mae_row(row: dict) -> dict:
        return {
            "config_label": str(row["config_label"]),
            "K": int(row["K"]),
            "world_idx": int(row["world_idx"]),
            "seed": str(row["seed"]),
            "train_seed": str(row["train_seed"]),
            "arm": str(row["arm"]),
            "state_mae": float(row["state_mae"]),
        }

    coerced_mae = [_coerce_mae_row(r) for r in mae_rows]

    by_cell: Dict[Tuple[str, int], List[dict]] = {}
    for r in coerced:
        by_cell.setdefault((r["config_label"], r["K"]), []).append(r)

    mae_by_cell_arm: Dict[Tuple[str, int, str], List[float]] = {}
    for r in coerced_mae:
        mae_by_cell_arm.setdefault((r["config_label"], r["K"], r["arm"]), []).append(r["state_mae"])

    cells_out: Dict[Tuple[str, int], dict] = {}

    for cell_key, cell_binding in binding.items():
        cell_key = (str(cell_key[0]), int(cell_key[1]))
        config_label, K = cell_key
        cell_rows = by_cell.get(cell_key, [])
        binding_rows = [r for r in cell_rows if r["budget"] == int(cell_binding)]

        by_arm: Dict[str, List[dict]] = {}
        for r in binding_rows:
            by_arm.setdefault(r["arm"], []).append(r)

        present_arms = [a for a in G2A_ARM_FAMILY if a in by_arm]
        if not present_arms:
            continue

        def _by_world_seed(arm_name: str) -> Dict[Tuple[int, str], dict]:
            out: Dict[Tuple[int, str], dict] = {}
            for r in by_arm.get(arm_name, []):
                out[(r["world_idx"], r["train_seed"])] = r
            return out

        legsum_by_world: Dict[int, dict] = {}
        for r in cell_rows:
            if r["arm"] == "h_legsum" and r["budget"] == int(cell_binding):
                legsum_by_world[r["world_idx"]] = r

        arms_out: Dict[str, dict] = {}
        for arm_name in present_arms:
            arm_rows = by_arm[arm_name]
            success = float(np.mean([1.0 if r["found"] else 0.0 for r in arm_rows])) if arm_rows else float("nan")

            arm_by_ws = _by_world_seed(arm_name)
            ratios: List[float] = []
            for (world_idx, _train_seed), arm_row in arm_by_ws.items():
                ls_row = legsum_by_world.get(world_idx)
                if ls_row is None:
                    continue
                if bool(arm_row["found"]) and bool(ls_row["found"]):
                    ratios.append(float(arm_row["expansions"]) / float(ls_row["expansions"]))
            med, ci_lo, ci_hi = _bootstrap_median_ci(ratios, seed=12345, n_boot=1000, lo=2.5, hi=97.5)

            mae_vals = mae_by_cell_arm.get((config_label, K, arm_name), [])
            mean_mae = float(np.mean(mae_vals)) if mae_vals else None

            arms_out[arm_name] = {
                "success": success,
                "ratio_median": med,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_pairs": len(ratios),
                "mean_state_mae": mean_mae,
                "n_mae": len(mae_vals),
            }

        cells_out[cell_key] = {"binding": int(cell_binding), "arms": arms_out}

    return {"cells": cells_out}


def g2a_verdict(g2: dict) -> dict:
    """G2a verdict (spec section 1 / task prompt, quoted): positive iff on
    >=1 config AT K=8 BOTH (a) the ratio medians are monotone
    non-increasing in k over usable k values (n_pairs >= 1) AND (b) the k=1
    vs k=8 bootstrap CIs are disjoint (ci_hi(k8) < ci_lo(k1)); ALTERNATIVELY
    the same two conditions on mean state_mae (monotone decreasing + a
    clear k1-vs-k8 gap -- operationalized: mae(k8) < mae(k1) AND the mae
    sequence monotone non-increasing; no CI needed for mae, it is
    near-deterministic per seed). Reports WHICH criterion fired ("ratio",
    "mae", or "both" if both independently qualify). Cells with < 2 usable
    k values (ratio n_pairs >= 1 for the ratio criterion, or a present
    mean_state_mae for the mae criterion) -> insufficient for that
    criterion; a (config, K=8) cell is "insufficient" overall only if BOTH
    criteria are insufficient (< 2 usable k values each).

    Only K=8 cells are read (the spec pins the verdict criterion to K=8
    specifically -- the forced-k arms are trained/evaluated the same way at
    every K, but the pre-registered depth-of-compute read is explicitly at
    the deepest mission).

    Returns `{"positive": bool, "positive_configs": [config_label, ...],
    "table": {(config_label, K): {"status", "criterion", "ratio_by_k",
    "mae_by_k", "ci_by_k"}}}` (K is always 8 in table keys here, kept as a
    tuple key for symmetry with `g1_verdict`'s `(arm, config)`-keyed
    table)."""
    table: Dict[Tuple[str, int], dict] = {}
    positive_configs: List[str] = []

    for cell_key, cell in g2.get("cells", {}).items():
        config_label, K = cell_key
        if K != 8:
            continue
        arms = cell.get("arms", {})

        ratio_by_k: Dict[int, Optional[float]] = {}
        ci_by_k: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
        mae_by_k: Dict[int, Optional[float]] = {}
        for arm_name, k_val in _FORCED_K_VALUE.items():
            stats = arms.get(arm_name)
            if stats is None or stats["n_pairs"] == 0:
                ratio_by_k[k_val] = None
                ci_by_k[k_val] = (None, None)
            else:
                ratio_by_k[k_val] = stats["ratio_median"]
                ci_by_k[k_val] = (stats["ci_lo"], stats["ci_hi"])
            mae_by_k[k_val] = stats["mean_state_mae"] if stats is not None else None

        # -- ratio criterion.
        usable_ratio_ks = sorted(k for k in ratio_by_k if ratio_by_k[k] is not None)
        ratio_fires = False
        if len(usable_ratio_ks) >= 2:
            usable_ratios = [ratio_by_k[k] for k in usable_ratio_ks]
            monotone = all(
                usable_ratios[i] >= usable_ratios[i + 1] - 1e-12
                for i in range(len(usable_ratios) - 1)
            )
            k1_ci = ci_by_k.get(1)
            k8_ci = ci_by_k.get(8)
            disjoint = (
                k1_ci[0] is not None and k8_ci[1] is not None
                and k8_ci[1] < k1_ci[0]
            )
            ratio_fires = bool(monotone and disjoint)

        # -- mae criterion. Known edge (T8-review minor 4, accepted as-is):
        # a NaN mean_state_mae would count as "usable" here (it is not
        # None) and then fail every comparison below (NaN compares False),
        # rendering the cell "negative" rather than "insufficient" --
        # practically unreachable, since a NaN state_mae requires a bundle
        # with zero finite-oracle states, which `mission_reachable` already
        # excludes at collect time.
        usable_mae_ks = sorted(k for k in mae_by_k if mae_by_k[k] is not None)
        mae_fires = False
        if len(usable_mae_ks) >= 2:
            usable_maes = [mae_by_k[k] for k in usable_mae_ks]
            mae_monotone = all(
                usable_maes[i] >= usable_maes[i + 1] - 1e-12
                for i in range(len(usable_maes) - 1)
            )
            mae1 = mae_by_k.get(1)
            mae8 = mae_by_k.get(8)
            mae_gap = mae1 is not None and mae8 is not None and mae8 < mae1
            mae_fires = bool(mae_monotone and mae_gap)

        if len(usable_ratio_ks) < 2 and len(usable_mae_ks) < 2:
            status = "insufficient"
            criterion = None
        elif ratio_fires or mae_fires:
            status = "positive"
            if ratio_fires and mae_fires:
                criterion = "both"
            elif ratio_fires:
                criterion = "ratio"
            else:
                criterion = "mae"
        else:
            status = "negative"
            criterion = None

        table[(config_label, 8)] = {
            "status": status,
            "criterion": criterion,
            "ratio_by_k": ratio_by_k,
            "ci_by_k": ci_by_k,
            "mae_by_k": mae_by_k,
        }
        if status == "positive":
            positive_configs.append(config_label)

    return {
        "positive": len(positive_configs) > 0,
        "positive_configs": positive_configs,
        "table": table,
    }


def g2b_analysis(halt_rows: Sequence[dict]) -> dict:
    """G2b: Spearman correlation of `mean_halt_steps` vs `K`, pooled across
    every row with K in {2,4,8} (K=0 excluded -- hrmv2 DOES train at K=0
    (it is one of the 5 native-comparable arms in G1), but G2b's question
    is specifically about depth-SENSITIVITY across K variation; K=0 has no
    mission legs at all, so including it would conflate "does it think
    longer on deeper missions" with "does it behave differently on the
    K=0-vs-K>0 formulation boundary", a different question already covered
    by the K=0 continuity control -- documented per the task spec's
    instruction), plus a per-config_label breakdown of the same statistic.

    Arm scoping (T8-review minor 1): rows are additionally filtered to
    `arm == "hrmv2_act"` -- the halt-dump trigger in `run_eval` is
    arm-agnostic (any provider that sets the side channel produces rows),
    so a future second ACT-live provider's rows must not silently pool
    into THIS arm's depth-sensitivity correlation. Rows WITHOUT an `arm`
    column (legacy CSVs written before the column existed, or minimal
    synthetic fixtures) default to "hrmv2_act" -- every such row was
    necessarily produced by the one ACT-live provider that existed then,
    so the default is exact, not a guess.

    Permutation test: 2000 permutations of the K labels (values shuffled
    against the fixed halt-step values), seeded `np.random.default_rng(24680)`,
    two-sided p = (count of |rho_perm| >= |rho_obs|, add-one-smoothed) /
    (n_perm + 1).

    DEGENERATE handling: if every `mean_halt_steps` value pooled is
    identical (a constant channel -- e.g. the Q-head learned immediate-halt
    at every depth, `mean_halt_steps == 1.0` everywhere), Spearman rho is
    mathematically undefined (zero-variance input): returns `{"constant":
    True, "value": <the constant>, "positive": False}` for that scope
    instead of computing rho -- per the spec, this is treated as a
    SUBSTANTIVE negative result (\"no learned depth-sensitivity\"), not a
    crash or a silently-omitted stat.

    Verdict positive iff rho > 0 AND p < 0.05.

    Returns `{"pooled": {...}, "by_config": {config_label: {...}}}` where
    each `{...}` is either the constant-channel dict above or `{"rho":
    float, "p": float, "n": int, "constant": False, "positive": bool}`."""

    def _coerce_halt_row(row: dict) -> dict:
        return {
            "config_label": str(row["config_label"]),
            "K": int(row["K"]),
            "world_idx": int(row["world_idx"]),
            "seed": str(row["seed"]),
            "train_seed": str(row["train_seed"]),
            # Missing-column default is "hrmv2_act", NOT a sentinel -- see
            # the arm-scoping paragraph in this function's docstring.
            "arm": str(row.get("arm", "hrmv2_act")),
            "mean_halt_steps": float(row["mean_halt_steps"]),
            "n_queried": int(row["n_queried"]),
        }

    coerced_all = [_coerce_halt_row(r) for r in halt_rows]
    coerced = [r for r in coerced_all if r["K"] in (2, 4, 8) and r["arm"] == "hrmv2_act"]

    def _spearman_with_permutation(rows: Sequence[dict]) -> dict:
        if not rows:
            return {"rho": float("nan"), "p": float("nan"), "n": 0, "constant": False, "positive": False}

        vals = np.array([r["mean_halt_steps"] for r in rows], dtype=np.float64)
        ks = np.array([r["K"] for r in rows], dtype=np.float64)
        n = vals.shape[0]

        if np.allclose(vals, vals[0]):
            return {"constant": True, "value": float(vals[0]), "positive": False, "n": int(n)}

        def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
            # Rank-based Pearson correlation (average ranks for ties, the
            # standard Spearman definition) -- no scipy dependency (this
            # module's existing convention: `_mcnemar_exact_p`/`_bh` are
            # hand-rolled too, not scipy-imported).
            rx = _rankdata(x)
            ry = _rankdata(y)
            if np.std(rx) == 0 or np.std(ry) == 0:
                return float("nan")
            return float(np.corrcoef(rx, ry)[0, 1])

        rho_obs = _spearman_rho(ks, vals)
        if not np.isfinite(rho_obs):
            # K itself is constant (e.g. every row is K=8) -- correlation
            # undefined for a different reason than the value channel;
            # treated the same way as the value-constant case (nothing to
            # correlate against), but keeps `constant: False` since the
            # DEGENERATE channel here is K, not mean_halt_steps -- reported
            # via nan rho/p rather than the constant-channel branch (which
            # is specifically about `mean_halt_steps` being constant).
            return {"rho": float("nan"), "p": float("nan"), "n": int(n), "constant": False, "positive": False}

        rng = np.random.default_rng(24680)
        n_perm = 2000
        count_ge = 0
        for _ in range(n_perm):
            perm_ks = rng.permutation(ks)
            rho_perm = _spearman_rho(perm_ks, vals)
            if np.isfinite(rho_perm) and abs(rho_perm) >= abs(rho_obs):
                count_ge += 1
        p = (count_ge + 1) / (n_perm + 1)

        positive = bool(rho_obs > 0 and p < 0.05)
        return {"rho": rho_obs, "p": float(p), "n": int(n), "constant": False, "positive": positive}

    pooled = _spearman_with_permutation(coerced)

    by_config: Dict[str, dict] = {}
    config_labels = sorted({r["config_label"] for r in coerced})
    for config_label in config_labels:
        rows = [r for r in coerced if r["config_label"] == config_label]
        by_config[config_label] = _spearman_with_permutation(rows)

    return {"pooled": pooled, "by_config": by_config}


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks with ties (the standard Spearman-rank convention,
    matching `scipy.stats.rankdata`'s default `method="average"`): equal
    values receive the mean of the rank positions they would occupy if
    broken arbitrarily. 1-indexed ranks (the traditional convention;
    irrelevant to the correlation itself, which is invariant to a constant
    shift)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# Task 8: results-doc writer.
#
# Gate texts copied VERBATIM from `docs/experiments/continuous/c11/design/2026-07-07-c11-
# compositional-mission-design.md` section 1 (read there, not re-derived --
# any future edit to the spec's wording must be mirrored here by hand,
# per the task prompt's "spec/plan links, pre-registered gate texts
# VERBATIM from spec section 1" instruction).
# ---------------------------------------------------------------------------

G1_GATE_TEXT: str = (
    "G1 -- dose-response in structure. Primary read: per (config, K), each structured arm "
    "vs the MLP control on matched per-world expansion-ratio differences (both-solved worlds) "
    "+ success McNemar. Continuity control: at K=0 the task reduces exactly to C7 (legsum == "
    "euclid, no mission tokens beyond the goal leg) and all arms must statistically tie -- if "
    "they don't, it is a formulation/implementation bug and the phase halts for diagnosis, not "
    "a result. Positive verdict: >=1 structured arm beats the MLP with BH q < 0.05 at >=2 of "
    "the 3 K in {2,4,8} values on >=1 config, with the gap monotone non-decreasing in K. "
    "Anything less is a negative for that arm."
)

G2_GATE_TEXT: str = (
    "G2 -- depth-of-compute (HRM-v2 mechanism). (a) Forced-segment curves: eval the trained "
    "ACT arm at forced k in {1,2,4,8} segments; positive iff quality (state-level MAE and/or "
    "expansion-ratio) improves monotonically with bootstrap-CI separation between k=1 and k=8 "
    "on >=1 config at K=8. (b) Learned halting: Spearman correlation of mean halt-steps vs K "
    "positive with q < 0.05 (\"thinks longer on deeper missions\")."
)

G3_GATE_TEXT: str = (
    "G3 -- honest closure. If G0-H passed (it did), the I/O exposes real structure, and the "
    "MLP control still ties everything: \"learned planning heuristics are architecture-agnostic\" "
    "graduates to a strong publishable claim; the program pivots to the transfer+integration "
    "paper with the architecture chapter closed."
)


def write_results_doc(g1_analysis: dict, k0: dict, g1: dict, g2: dict, g2b: dict,
                       meta: dict, path: str, addendum_md: Optional[str] = None) -> None:
    """Write `C11_RESULTS.md` to the GIVEN `path` (NEVER a hardcoded repo
    path -- see the module-level `run_analyze`/CLI split below for the
    clobber-avoidance convention this mirrors exactly from the probe's own
    `run_probe`/`_write_results_doc`/`main` split, per review requirement
    4). All content is derived from the passed-in analysis dicts; no
    hardcoded result glosses -- the ONLY place prose branches on booleans
    is the G3 closure paragraph, and every branch there reads its wording
    from the SAME computed booleans it reports (`k0["pass"]`,
    `g1["positive"]`, `g2a_verdict(g2)["positive"]`,
    `g2b["pooled"].get("positive")`), never a value independent of them.

    G3 gating on the continuity control (post-run diagnosis fix): the
    three closure branches are consulted ONLY when `k0["pass"]` is True.
    When the K=0 continuity control FAILED, G3 renders a HALT section
    instead -- per the pre-registered protocol (spec section 1) a K=0
    failure makes the G1 read uninterpretable until the separation is
    diagnosed, so rendering ANY closure claim (all-tie, all-positive, or
    partial) beneath a failing K=0 section would be factually
    self-contradictory. (Exactly what an earlier version did on the first
    full run: it branched on the G1/G2a/G2b booleans alone and printed
    the all-tie closure text under a K=0 section listing eight
    CI-separated violations.)

    `addendum_md` (optional): a hand-authored markdown string, rendered
    VERBATIM as a final `## Diagnosis addendum` section -- the injection
    point for the post-diagnosis interpretation after a K=0 failure (or
    any other post-run annotation). `None` (the default) renders no such
    section. The CLI threads this from `--addendum <file>` via
    `run_analyze`/`_rerun_write_results_doc_at`.

    `g1_analysis`: the raw `analyze(records, binding, cfg)` dict (per-cell
    success/vs_mlp/vs_legsum/oracle_vs_legsum), keyed `(config_label, K)`.
    `k0`: `k0_continuity(g1_analysis)`. `g1`: `g1_verdict(g1_analysis)`.
    `g2`: `g2_analysis(records, mae_rows, binding, cfg)`. `g2b`:
    `g2b_analysis(halt_rows)`. `meta`: caller-supplied run metadata --
    `date`, `branch`, `spec_path`, `plan_path`, `param_counts` (a `{arm:
    int}` dict, used for the caveats section's "param-count spread within
    band" line; missing/absent entries are rendered as "n/a", never
    crash)."""
    g2a = g2a_verdict(g2)
    g2b_pooled = g2b.get("pooled", {})

    config_labels = sorted({key[0] for key in g1_analysis})
    k_values = sorted({key[1] for key in g1_analysis})

    lines: List[str] = []
    lines.append("# C11 -- Compositional-Mission PRM Heuristics: Results")
    lines.append("")
    lines.append(f"**Date:** {meta.get('date', 'n/a')}")
    lines.append(f"**Branch:** {meta.get('branch', 'n/a')}")
    lines.append(f"**Spec:** `{meta.get('spec_path', 'n/a')}`")
    lines.append(f"**Plan:** `{meta.get('plan_path', 'n/a')}`")
    lines.append("")
    lines.append("## Pre-registered gates (verbatim, spec section 1)")
    lines.append("")
    lines.append(f"**{G1_GATE_TEXT}**")
    lines.append("")
    lines.append(f"**{G2_GATE_TEXT}**")
    lines.append("")
    lines.append(f"**{G3_GATE_TEXT}**")
    lines.append("")

    # -- Per-cell G1 tables.
    lines.append("## G1 -- per-cell results")
    lines.append("")
    for config_label in config_labels:
        lines.append(f"### Config {config_label}")
        lines.append("")
        header = "| K | Binding budget | Arm | Success | vs-MLP ratio [CI] (n) | vs-legsum ratio (n) |"
        lines.append(header)
        lines.append("|---|---|---|---|---|---|")
        for K in k_values:
            cell = g1_analysis.get((config_label, K))
            if cell is None:
                continue
            binding_budget = cell["binding"]
            success = cell.get("success", {})
            vs_mlp = cell.get("vs_mlp", {})
            vs_legsum = cell.get("vs_legsum", {})
            for arm_name in sorted(success):
                succ = _fmt_g(success.get(arm_name), ".2f")
                if arm_name in vs_mlp:
                    stats = vs_mlp[arm_name]
                    ratio_str = (
                        f"{_fmt_g(stats['ratio_median'])} "
                        f"[{_fmt_g(stats['ci_lo'])}, {_fmt_g(stats['ci_hi'])}] (n={stats['n_pairs']})"
                    )
                elif arm_name == "mlp":
                    ratio_str = "-- (control)"
                else:
                    ratio_str = "n/a"
                if arm_name in vs_legsum:
                    ls = vs_legsum[arm_name]
                    legsum_str = f"{_fmt_g(ls['ratio_median'])} (n={ls['n']})"
                else:
                    legsum_str = "n/a"
                lines.append(f"| {K} | {binding_budget} | {arm_name} | {succ} | {ratio_str} | {legsum_str} |")
            oracle_row = cell.get("oracle_vs_legsum", {})
            if oracle_row:
                oracle_str = f"{_fmt_g(oracle_row.get('ratio_median'))} (n={oracle_row.get('n')})"
                lines.append(f"| {K} | {binding_budget} | h_oracle (ceiling) | -- | -- | {oracle_str} |")
        lines.append("")

    # -- K=0 continuity verdict.
    lines.append("## K=0 continuity control")
    lines.append("")
    k0_status = "PASS" if k0.get("pass") else "FAIL"
    lines.append(f"**Verdict: {k0_status}**")
    lines.append("")
    if not k0.get("pass"):
        lines.append("Violations (config_label, arm, reason):")
        lines.append("")
        for config_label, arm_name, reason in k0.get("violations", []):
            lines.append(f"- {config_label} / {arm_name}: {reason}")
        lines.append("")
    else:
        lines.append(
            "No arm showed statistically significant separation from the MLP control at K=0 -- "
            "the residual formulation reduces exactly to C7 at K=0 as required, so the G1 read "
            "at K in {2,4,8} below is trusted."
        )
        lines.append("")

    # -- G1 verdict.
    lines.append("## G1 verdict")
    lines.append("")
    g1_positive = bool(g1.get("positive"))
    if g1_positive:
        arm_config_pairs = ", ".join(f"{arm}/{cfg_label}" for arm, cfg_label in g1.get("positive_arms", []))
        lines.append(f"**Positive.** Beating arm/config pairs: {arm_config_pairs}.")
    else:
        lines.append("**Negative.** No structured arm beat the MLP control at >=2 of the 3 K values on any config.")
    lines.append("")
    lines.append("| Arm | Config | K=2 beat | K=4 beat | K=8 beat | Monotone | Status |")
    lines.append("|---|---|---|---|---|---|---|")
    for (arm_name, config_label), row in sorted(g1.get("table", {}).items()):
        beats = row.get("beats", {})
        beat_strs = ["yes" if beats.get(K) else ("n/a" if K not in beats else "no") for K in (2, 4, 8)]
        monotone_str = "yes" if row.get("monotone") else ("n/a" if row.get("monotone") is None else "no")
        lines.append(
            f"| {arm_name} | {config_label} | {beat_strs[0]} | {beat_strs[1]} | {beat_strs[2]} | "
            f"{monotone_str} | {row.get('status')} |"
        )
    lines.append("")

    # -- G2a table + verdict.
    lines.append("## G2a -- forced-segment curves (K=8)")
    lines.append("")
    if not g2.get("cells"):
        lines.append("**Not run** -- no forced-k / hrmv2_act records present at K=8.")
        lines.append("")
    else:
        # T8-review minor 2: the ACT-live (`hrmv2_act`) reference columns
        # make the free-running-vs-forced-k8 comparison visible in the
        # phase deliverable itself -- `g2_analysis` was already computing
        # these stats per cell; only the rendering was missing.
        header = ("| Config | k=1 ratio | k=2 ratio | k=4 ratio | k=8 ratio | act-live ratio [CI] | "
                  "k=1 mae | k=2 mae | k=4 mae | k=8 mae | act-live mae |")
        lines.append(header)
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for cell_key, cell in sorted(g2.get("cells", {}).items()):
            config_label, K = cell_key
            if K != 8:
                continue
            arms = cell.get("arms", {})
            ratio_cols = []
            mae_cols = []
            for k_val in (1, 2, 4, 8):
                arm_name = f"hrmv2_act_k{k_val}"
                stats = arms.get(arm_name)
                ratio_cols.append(_fmt_g(stats["ratio_median"]) if stats else "n/a")
                mae_cols.append(_fmt_g(stats["mean_state_mae"]) if stats else "n/a")
            act_stats = arms.get("hrmv2_act")
            if act_stats is not None and act_stats["n_pairs"] > 0:
                act_ratio = (
                    f"{_fmt_g(act_stats['ratio_median'])} "
                    f"[{_fmt_g(act_stats['ci_lo'])}, {_fmt_g(act_stats['ci_hi'])}]"
                )
            else:
                act_ratio = "n/a"
            act_mae = _fmt_g(act_stats["mean_state_mae"]) if act_stats is not None else "n/a"
            lines.append(
                f"| {config_label} | " + " | ".join(ratio_cols) + f" | {act_ratio} | "
                + " | ".join(mae_cols) + f" | {act_mae} |"
            )
        lines.append("")
    g2a_positive = bool(g2a.get("positive"))
    if g2a_positive:
        fired = ", ".join(
            f"{cfg_label} (via {row['criterion']})"
            for (cfg_label, K), row in g2a.get("table", {}).items()
            if row.get("status") == "positive"
        )
        lines.append(f"**G2a verdict: positive.** Fired on: {fired}.")
    else:
        lines.append("**G2a verdict: negative or insufficient data.**")
    lines.append("")

    # -- G2b.
    lines.append("## G2b -- halting vs mission depth")
    lines.append("")
    if g2b_pooled.get("constant"):
        lines.append(
            f"**Constant would-halt channel** (mean_halt_steps == {_fmt_g(g2b_pooled.get('value'))} "
            "for every queried row, pooled across K in {2,4,8}). Per the pre-registered handling: "
            "a constant would-halt channel (e.g. all 1.0 = the Q-head learned immediate-halt) is a "
            "SUBSTANTIVE 'no learned depth-sensitivity' negative, not a measurement artifact -- the "
            "channel itself was verified to move with training (it is not a dead wire)."
        )
        lines.append("")
        lines.append("**G2b verdict: negative (constant channel).**")
    elif g2b_pooled:
        lines.append(
            f"Pooled Spearman rho = {_fmt_g(g2b_pooled.get('rho'))}, permutation p = "
            f"{_fmt_g(g2b_pooled.get('p'))} (n={g2b_pooled.get('n', 0)})."
        )
        lines.append("")
        lines.append(f"**G2b verdict: {'positive' if g2b_pooled.get('positive') else 'negative'}.**")
    else:
        lines.append("**Not run** -- no halt-step records present.")
    lines.append("")

    # -- G3 closure paragraph, generated CONDITIONALLY from computed
    # booleans -- gated FIRST on the K=0 continuity control (post-run
    # diagnosis fix: the first full run's K=0 control FAILED, and this
    # block -- then branching on the G1/G2a/G2b booleans alone -- rendered
    # the all-tie/architecture-agnostic closure text directly beneath a
    # K=0 section listing eight CI-separated violations; factually
    # self-contradictory and procedurally invalid, since the
    # pre-registered protocol makes G1 unreadable until the K=0 separation
    # is diagnosed). When k0 passes: three branches (all-negative,
    # all-positive, mixed) -- no hardcoded glosses, per the probe's
    # conditional-prose lesson (an earlier version of the PROBE's own doc
    # hardcoded text that contradicted its own computed monotonicity
    # booleans; see `git log --oneline --grep=clobber`, commit fd4aa55).
    lines.append("## G3 -- closure")
    lines.append("")
    if not bool(k0.get("pass")):
        # HALT branch: NO closure claim may render. The wording below must
        # never contain any closure-branch-unique phrase (the writer test
        # asserts their absence from this section).
        lines.append(
            "**G3 closure WITHHELD pending diagnosis.** The K=0 continuity control FAILED "
            "(see the K=0 section above). Per the pre-registered protocol (spec section 1: at "
            "K=0 all arms must statistically tie -- \"if they don't, it is a formulation/"
            "implementation bug and the phase halts for diagnosis, not a result\"), the G1 "
            "dose-response verdict line above still renders -- it is computed mechanically "
            "from the records -- but it is NOT interpretable as an architecture result until "
            "the K=0 separation is diagnosed, and no pre-registered closure claim (all-tie, "
            "all-positive, or partial) is rendered for this run."
        )
        lines.append("")
        lines.append("K=0 violations (config, arm, reason):")
        lines.append("")
        for config_label, arm_name, reason in k0.get("violations", []):
            lines.append(f"- {config_label} / {arm_name}: {reason}")
        lines.append("")
        if addendum_md is not None:
            lines.append(
                "The 'Diagnosis addendum' section at the end of this document carries the "
                "post-diagnosis interpretation."
            )
        else:
            lines.append(
                "No diagnosis addendum is attached to this render; once the diagnosis is "
                "written, re-run analyze with `--addendum <file>` to inject it."
            )
    else:
        g2b_positive = bool(g2b_pooled.get("positive"))
        all_negative = (not g1_positive) and (not g2a_positive) and (not g2b_positive)
        if all_negative:
            lines.append(
                "All three gates (G1 dose-response, G2a forced-segment curves, G2b halting-vs-depth) "
                "came back negative or insufficient: every structured arm tied the MLP control, forced "
                "compute depth did not improve quality, and learned halting showed no depth-sensitivity "
                "(or a constant/degenerate channel). G0-H passed (the substrate has real, measured "
                "oracle headroom) and the I/O contract exposes genuine sequence/graph/product-graph "
                "structure to every arm identically -- so this is not an I/O-starvation or "
                "headroom-absence artifact. Per the pre-registered G3 text: 'learned planning heuristics "
                "are architecture-agnostic' graduates to a strong publishable claim on this substrate. "
                "The architecture chapter of this program is closed; the next thrust is the "
                "transfer+integration paper."
            )
        elif g1_positive and g2a_positive and g2b_positive:
            lines.append(
                "All three gates came back positive: >=1 structured arm beats the MLP control with a "
                "monotone, statistically-separated dose-response in mission depth (G1); forced compute "
                "depth improves quality monotonically with a clear k=1-vs-k=8 separation (G2a); and "
                "learned halting correlates positively with mission depth at q < 0.05 (G2b). The "
                "hierarchy/iterative-compute thesis lives on this substrate -- structure is not "
                "architecture-agnostic here, and depth-of-compute is a real, learned mechanism, not "
                "noise. Per the decision rule: invest in the high-DOF substrate (S1) next."
            )
        else:
            fired = []
            if g1_positive:
                fired.append("G1 (structure dose-response)")
            if g2a_positive:
                fired.append("G2a (forced-segment quality curve)")
            if g2b_positive:
                fired.append("G2b (halting-vs-depth correlation)")
            not_fired = [g for g in ("G1", "G2a", "G2b") if not any(g in f for f in fired)]
            lines.append(
                f"Mixed result: {', '.join(fired)} came back positive, while "
                f"{', '.join(not_fired) if not_fired else 'no other gate'} did not. This is not the "
                "clean all-tie the G3 architecture-agnostic text describes, nor the clean all-positive "
                "hierarchy-thesis-lives text -- the honest reading is partial signal: at least one "
                "mechanism (structure and/or iterative compute) shows a real, statistically-supported "
                "effect on this substrate, but not every pre-registered gate agrees. This warrants "
                "targeted follow-up on the SPECIFIC gate(s) that fired (e.g. a scaled-arm addendum "
                "focused on the positive arm/mechanism) before either closing the architecture chapter "
                "or fully committing to the high-DOF substrate."
            )
    lines.append("")

    # -- Caveats (standing list, requirement 4).
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- **HRM-v2 optimizer/loop confound (pre-registered accepted).** The HRM-v2 arm's "
        "optimizer (adam-atan2 + constant LR + warmup) and per-segment training loop are "
        "paper-faithful, not matched to arms 1-4's AdamW/smooth-L1 recipe -- mechanism-"
        "faithfulness IS the arm, and this confound was pre-registered as accepted (spec section 5)."
    )
    lines.append(
        "- **GNN m=8 rounds (pre-registered).** No post-hoc m tuning; a single pre-registered "
        "m-sensitivity appendix run at m in {4,16} on one cell is allowed for interpretation only "
        "(spec section 8)."
    )
    param_counts = meta.get("param_counts") or {}
    if param_counts:
        counts_str = ", ".join(f"{arm}={n:,}" for arm, n in sorted(param_counts.items()))
    else:
        counts_str = "n/a (no manifest param counts supplied to write_results_doc)"
    lines.append(
        f"- **Param-count spread within band.** Per-arm parameter counts (from the training "
        f"manifest, via `meta`): {counts_str}. Every native arm targets the [0.5M, 3.5M] band; "
        "the HRM-v2 arm targets [1M, 4M] (spec section 3, item 5) -- these bands overlap but are "
        "not identical, a known spread."
    )
    lines.append(
        "- **`door_open_at_s` trace slot (T2-review note).** This trace-token slot is "
        "structurally 0.0 for configs without doors, and even for config C it does not directly "
        "encode live door state -- door state is carried by leg presence/ordering in the mission "
        "trace itself, not by this slot. Documented, not a bug."
    )
    lines.append(
        "- **Eval-time fixed-compute ACT.** The would-halt channel (`hrmv2_halt_steps`) is a "
        "readout of the model's OWN halting signal (`q_halt_logits > q_continue_logits`), not an "
        "actual variable-compute eval loop -- ACT-live eval still runs to `carry.halted.all()` "
        "(or the safety cap), so G2b measures learned depth-preference, not realized eval-time "
        "compute savings."
    )
    lines.append(
        "- **Oracle headroom is an upper bound.** `h_oracle` is the exact product-graph shortest "
        "path; every learned arm is explicitly inadmissible (the residual formulation has no "
        "admissibility guarantee at eval time) -- learned-arm costs are recorded but never "
        "asserted optimal, and the oracle row in the G1 tables above is a CEILING, not a target "
        "any learned arm is expected to reach."
    )
    lines.append("")

    # -- Diagnosis addendum (optional, VERBATIM): the injection point for
    # the hand-authored post-diagnosis interpretation -- this writer never
    # generates its content, only hosts it (see the docstring).
    if addendum_md is not None:
        lines.append("## Diagnosis addendum")
        lines.append("")
        lines.append(addendum_md)
        lines.append("")

    out_path = Path(path)
    C.ensure_dir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    os.replace(tmp, out_path)


def _fmt_g(v, spec: str = ".3f") -> str:
    """Format a possibly-None/NaN numeric value for the results doc (mirrors
    the probe's own `_fmt` helper, kept local to avoid importing across
    phase modules for a one-line formatting utility)."""
    if v is None:
        return "n/a"
    if isinstance(v, float) and v != v:  # NaN
        return "nan"
    if isinstance(v, (int, float)):
        return format(v, spec)
    return str(v)


# ---------------------------------------------------------------------------
# Task 8: run_analyze.
# ---------------------------------------------------------------------------

def _read_csv_rows(path: Path) -> List[dict]:
    """`csv.DictReader` readback -- the standard C-series pattern (see e.g.
    `continuous_prm_c6_heatmap_value_field.read_csv`), reimplemented here
    (rather than imported) since `continuous_prm_common` itself has no
    generic CSV reader (only `write_csv`) and importing a reader from a
    peer PHASE module (C6) would violate this module's own "phase modules
    must not import each other" convention already noted above `_mcnemar_
    exact_p`'s docstring."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_g1_eligible_arm(arm_name: str) -> bool:
    """True iff `arm_name` is allowed to reach `analyze`/G1 (the
    pre-registered BH family). Excludes two DISTINCT arm families, both of
    which exist to ride the SAME eval CSV as the G1 arms without diluting
    G1's own statistics:

      - forced-k HRM-v2 diagnostics (`hrmv2_act_k1`/`_k2`/`_k4`/`_k8`,
        prefix `"hrmv2_act_k"`) -- G2-only, per `G1_STRUCTURED_ARM_NAMES`'s
        own docstring.
      - Task 10 scaled-addendum arms (`unet_film_big`/`hrm_trace_big`,
        suffix `"_big"`) -- an addendum comparing scale within EACH of
        those two arms, never new participants in the matched-scale G1
        dose-response family (`G1_STRUCTURED_ARM_NAMES` never contains
        either name -- see that tuple's own big-arm exclusion note).

    A single shared predicate (rather than two separate `not
    ...startswith(...)` / `not ...endswith(...)` clauses inlined at every
    call site) so a future third excluded family only needs ONE edit."""
    name = str(arm_name)
    if name.startswith("hrmv2_act_k"):
        return False
    if name.endswith("_big"):
        return False
    return True


def _assemble_analysis(out_dir: str, cfg: C11MissionConfig) -> dict:
    """Shared read+analyze core for `run_analyze` and the CLI's
    `--canonical-results` re-write path (`_rerun_write_results_doc_at`) --
    factored out so the two call sites can never drift on the binding-map
    assembly / BH-family-scoping / G2-gating logic (they read the SAME
    on-disk artifacts and must produce byte-identical analysis dicts; only
    the results-doc WRITE destination differs between the two callers).

    Reads `{out_dir}/results/c11_eval_raw.csv`, `{out_dir}/binding_k0.json`,
    `{out_dir}/results/c11_halt_steps.csv`, `{out_dir}/results/
    c11_state_mae.csv` (the latter two OPTIONAL -- their absence means "G2
    sections marked not run", never a crash, per the task spec's explicit
    contract).

    Binding map assembly: `cfg.binding_budgets` (the pre-registered K in
    {2,4,8} budgets) PLUS, for every K=0 entry present in `binding_k0.json`
    (keyed `f"{config_label}_K0"`, written by `run_eval`'s own K=0
    calibration path), `(config_label, 0) -> budget`. This is the ONE place
    the binding map is assembled from its two sources -- `analyze`/
    `g2_analysis` never do this themselves (per their own docstrings).

    BH-family scoping (review-derived integration requirement 1, MANDATORY):
    filters `records` to `g1_records = [r for r in records if
    _is_g1_eligible_arm(r["arm"])]` BEFORE calling `analyze` -- the
    forced-k diagnostic rows would otherwise dilute the pre-registered BH
    family from 5 structured arms to 9, and (Task 10) the scaled-addendum
    `*_big` rows would otherwise inject 2 NEW arms the pre-registered G1
    family never accounted for. The UNFILTERED `records` (forced-k AND
    `_big` rows included) go to `g2_analysis`/every other consumer --
    `_is_g1_eligible_arm` scopes ONLY the analyze/G1 call below, per
    `_is_g1_eligible_arm`'s own docstring.

    Returns `{"g1_analysis", "k0", "g1", "g2", "g2a", "g2b", "meta"}` --
    everything `write_results_doc` needs, plus the meta dict (date/branch/
    spec+plan paths/manifest param counts)."""
    out_path = Path(out_dir)

    records = _read_csv_rows(out_path / "results" / "c11_eval_raw.csv")

    binding_k0_path = out_path / "binding_k0.json"
    binding_k0 = C.read_json(binding_k0_path) if binding_k0_path.exists() else {}
    binding: Dict[Tuple[str, int], int] = dict(cfg.binding_budgets)
    for _key, entry in binding_k0.items():
        binding[(str(entry["config_label"]), int(entry["K"]))] = int(entry["budget"])

    halt_path = out_path / "results" / "c11_halt_steps.csv"
    halt_rows = _read_csv_rows(halt_path) if halt_path.exists() else []

    mae_path = out_path / "results" / "c11_state_mae.csv"
    mae_rows = _read_csv_rows(mae_path) if mae_path.exists() else []

    # -- Requirement 1 (+ Task 10 extension): forced-k rows AND `_big`
    # scaled-addendum rows NEVER reach analyze/G1 -- see
    # `_is_g1_eligible_arm`'s own docstring for both excluded families.
    g1_records = [r for r in records if _is_g1_eligible_arm(r["arm"])]

    g1_analysis = analyze(g1_records, binding, cfg)
    k0 = k0_continuity(g1_analysis)
    g1 = g1_verdict(g1_analysis)

    # -- G2 sees the FULL (unfiltered) records -- forced-k rows are its
    # only reason to exist. Gated on whether ANY G2A_ARM_FAMILY arm is
    # present in `records` at all (NOT on whether the mae/halt files
    # happened to exist -- `g2_analysis` itself already handles an empty
    # `mae_rows` gracefully, reporting `mean_state_mae: None` per arm, so
    # a present-but-empty mae file must not suppress a real forced-k
    # eval's ratio-only G2a read). File absence is what the "not run"
    # contract is about (an eval that never touched any hrmv2 arm at
    # all), not a merely-empty side-dump.
    has_g2_arms = any(str(r["arm"]) in G2A_ARM_FAMILY for r in records)
    g2 = g2_analysis(records, mae_rows, binding, cfg) if has_g2_arms else {"cells": {}}
    g2a = g2a_verdict(g2)
    g2b = g2b_analysis(halt_rows) if halt_rows else {"pooled": {}, "by_config": {}}

    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "branch": _current_git_branch(),
        "spec_path": "docs/experiments/continuous/c11/design/2026-07-07-c11-compositional-mission-design.md",
        "plan_path": "docs/experiments/continuous/c11/plans/2026-07-07-c11-mission.md",
        "param_counts": _manifest_param_counts(out_path),
    }

    return {"g1_analysis": g1_analysis, "k0": k0, "g1": g1, "g2": g2, "g2a": g2a, "g2b": g2b, "meta": meta}


def run_analyze(out_dir: str, cfg: Optional[C11MissionConfig] = None,
                addendum_path: Optional[str] = None) -> dict:
    """Assembles the full analysis (`_assemble_analysis`) and writes
    `write_results_doc(...)` to `{out_dir}/C11_RESULTS.md` (an
    out_dir-scoped path -- NEVER the canonical repo path; only the CLI's
    `--canonical-results` flag writes there, via a SEPARATE explicit call
    after this one, per requirement 4/the probe's clobber lesson). Prints
    the three gate verdict lines (K=0 continuity / G1 / G2a+G2b) and
    returns `{"k0": k0, "g1": g1, "g2a": g2a, "g2b": g2b}`.

    `addendum_path` (optional; the CLI's `--addendum`): path to a
    hand-authored markdown file whose CONTENT is injected verbatim as the
    doc's final `## Diagnosis addendum` section (see `write_results_doc`).
    A nonexistent path raises loudly (FileNotFoundError) -- silently
    skipping a requested addendum would be the clobber-lesson class of
    bug in reverse (a doc of record missing the interpretation the caller
    explicitly asked to attach)."""
    cfg = cfg or C11MissionConfig()
    out_path = Path(out_dir)

    addendum_md = Path(addendum_path).read_text(encoding="utf-8") if addendum_path else None

    a = _assemble_analysis(out_dir, cfg)
    k0, g1, g2, g2a, g2b = a["k0"], a["g1"], a["g2"], a["g2a"], a["g2b"]

    results_path = out_path / "C11_RESULTS.md"
    write_results_doc(a["g1_analysis"], k0, g1, g2, g2b, a["meta"], str(results_path),
                       addendum_md=addendum_md)

    k0_line = f"[c11-mission analyze] K=0 continuity: {'PASS' if k0['pass'] else 'FAIL'}"
    g1_line = f"[c11-mission analyze] G1 verdict: {'positive' if g1['positive'] else 'negative'}"
    g2_line = (
        f"[c11-mission analyze] G2 verdict: G2a="
        f"{'positive' if g2a['positive'] else 'negative'}, G2b="
        f"{'positive' if g2b.get('pooled', {}).get('positive') else 'negative'}"
    )
    print(k0_line, flush=True)
    print(g1_line, flush=True)
    print(g2_line, flush=True)

    return {"k0": k0, "g1": g1, "g2a": g2a, "g2b": g2b}


def _current_git_branch() -> str:
    """Best-effort current git branch name for the results doc's header --
    NEVER raises (a detached HEAD, missing git binary, or non-repo cwd all
    fall back to "n/a" rather than crashing analyze)."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(Path(__file__).resolve().parent),
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            if branch:
                return branch
    except Exception:  # noqa: BLE001 -- best-effort metadata only
        pass
    return "n/a"


def _manifest_param_counts(out_path: Path) -> Dict[str, int]:
    """Per-arm parameter counts from `{out_path}/manifest.json`, for the
    results doc's caveats section ("param-count spread within band").
    Takes the FIRST manifest entry seen per arm name (param counts do not
    vary across cell/seed for a given arm -- the matched recipe pins model
    size per arm, not per cell -- so any entry is representative). Returns
    `{}` if no manifest exists (never crashes `run_analyze`)."""
    manifest = _load_manifest(out_path)
    counts: Dict[str, int] = {}
    for _key, entry in sorted(manifest.items()):
        arm_name = entry.get("arm")
        if arm_name and arm_name not in counts and "param_count" in entry:
            counts[arm_name] = int(entry["param_count"])
    return counts


# ---------------------------------------------------------------------------
# Task 7: HRM-v2 registration hook.
#
# `continuous_prm_c11_hrmv2_arm.py` never imports this module at its OWN
# top level (T6/T7's isolation contract: `hrm` absent must not break this
# core module, or anything that imports it, or the hrmv2 module itself).
# The reverse direction is fine and is how the two modules connect: THIS
# module imports the hrmv2 module and asks it to register its providers,
# with the import wrapped in a try/except that catches ONLY ImportError
# (never a bare `except Exception`, which would silently swallow a real
# bug inside `register_hrmv2_providers` itself -- e.g. a typo in one of the
# five `register_provider_builder` calls should still crash loudly). When
# `hrm` (the HRM-v2 package) is not installed, `import
# continuous_prm_c11_hrmv2_arm` itself still succeeds (it has no top-level
# `hrm` import either -- see that module's own `_load_hrm` convention), but
# `register_hrmv2_providers()` EAGERLY calls `_load_hrm()` as its very
# first action (a deliberate design choice, not incidental -- an earlier
# version deferred all `hrm`-touching to provider-construction time, which
# meant registration itself never raised and silently registered
# `hrmv2_act` even with `hrm` fully absent), so THIS call is what actually
# raises ImportError the moment `hrm` is unavailable, and THAT is what this
# guard catches -- so the net effect is "this whole hook is a no-op when
# `hrm` is absent, and core still imports fine" (verified by
# `test_hrmv2_registration_import_error_only_guard` in
# `tests/test_c11_mission.py`, via a subprocess with `hrm` blocked at
# `sys.meta_path`).
try:
    import continuous_prm_c11_hrmv2_arm as _HA
    _HA.register_hrmv2_providers()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Task 8: run_full.
# ---------------------------------------------------------------------------

def run_full(out_dir: str, arms: Sequence[str], cells: Sequence[dict], seeds: Sequence[int],
             cfg: Optional[C11MissionConfig] = None, n_train_worlds: Optional[int] = None,
             n_test_worlds: Optional[int] = None, epochs: Optional[int] = None,
             budgets: Optional[Sequence[int]] = None,
             addendum_path: Optional[str] = None) -> None:
    """Collect (implicit) -> `run_train` -> `run_eval` -> `run_analyze`, all
    resume-friendly (every stage's own existing skip logic: `run_train`
    skips a (arm, cell, seed) whose checkpoint already exists; `run_eval`
    unconditionally re-runs A* -- cheap, ms-scale -- and OVERWRITES the raw/
    halt/mae CSVs and `binding_k0.json` merge-entries each call, so re-
    running `run_full` is safe but not incremental at the eval stage;
    `run_analyze` likewise always regenerates `C11_RESULTS.md`). Bundle
    collection itself has no separate "collect" step here -- `run_train`/
    `run_eval` each collect their OWN split's bundles internally (per their
    own docstrings), so "collect" is implicit in those two calls, per the
    task spec's own parenthetical.

    `arms` defaults (at the CLI layer, not here -- see `_resolve_scale_
    preset`) to the 5 natives + `hrmv2_act`; forced-k arms are NEVER passed
    to `run_train` (they have no trainer, see `_resolve_trainer`'s eval-
    only branch, which `run_train` already handles by skipping with a
    print line) -- but `run_eval` picks them up automatically whenever the
    ALIASED arm's (`hrmv2_act`'s) checkpoint exists in the manifest, via its
    own `ckpt_alias` resolution block, regardless of whether `arms` here
    lists them explicitly. So `run_full`'s own `arms` argument does not
    need to (and should not) include the forced-k names -- they ride along
    for free once `hrmv2_act` itself is trained and evaluated.

    Writes everything `run_train`/`run_eval`/`run_analyze` write, under
    `out_dir`; the results doc is `{out_dir}/C11_RESULTS.md` -- NEVER the
    canonical repo path (that is exclusively the CLI's `--canonical-results`
    flag's job, applied as a SEPARATE call after this function returns).
    `addendum_path` is threaded straight to `run_analyze` (same contract),
    so `--mode full --addendum <file>` attaches the addendum to the
    out_dir copy too, not only the flag-gated canonical one."""
    cfg = cfg or C11MissionConfig()

    run_train(out_dir, arms=arms, cells=cells, seeds=seeds, cfg=cfg,
               n_train_worlds=n_train_worlds, epochs=epochs)
    run_eval(out_dir, cells=cells, cfg=cfg, n_test_worlds=n_test_worlds,
              budgets=budgets, arms=None)
    run_analyze(out_dir, cfg, addendum_path=addendum_path)


# ---------------------------------------------------------------------------
# Task 4/8: CLI.
# ---------------------------------------------------------------------------
#
# `--mode train|eval|analyze|full`; `--canonical-results` (analyze/full
# only -- writes ADDITIONALLY to the canonical experiment-documentation
# `C11_RESULTS.md`, never in place of the out_dir copy); `--scale-preset
# smoke|local|big` (train/full only -- overrides n_train_worlds/
# n_test_worlds/epochs/budgets/cells/arms with the pre-registered small,
# full, or Task-10-scaled-addendum grid, each individually still
# overridable by an explicit flag of the same name, per
# `_resolve_scale_preset`'s own "explicit flag wins" contract).

def _parse_csv_arg(s: str) -> List[str]:
    return [t.strip() for t in str(s).split(",") if t.strip()]


def _select_cells(configs: Sequence[str], k_values: Sequence[int],
                   cfg: Optional[C11MissionConfig] = None) -> List[dict]:
    """`build_cell_grid()` filtered to `configs` (config_label) x
    `k_values`. Cells whose (config, K) isn't in the grid at all (e.g.
    config C at K=0, which is dropped by `build_cell_grid` itself) are
    simply absent from the result -- not an error, since the CLI's own
    defaults (`--configs A,B,C --k-values 0,2,4,8`) request that combination
    and rely on this silently yielding no C/K0 cell."""
    cfg = cfg or C11MissionConfig()
    configs_set = set(configs)
    k_set = set(int(k) for k in k_values)
    return [c for c in build_cell_grid(cfg) if c["config_label"] in configs_set and c["K"] in k_set]


# Native arms + hrmv2_act (if registered) -- the default `run_full`/`--mode
# full` arm set (spec: "Default arms = 5 natives + hrmv2_act (+ forced-k in
# EVAL only ...)"). Resolved lazily inside `_default_full_arms` (not a
# module-level constant) since `hrmv2_act`'s registration happens at
# IMPORT time via the try/except hook further down this file, which itself
# runs AFTER `ARM_NAMES` is defined but the hook's outcome (registered or
# not) is only knowable once the whole module has finished importing --
# a module-level tuple built at class-body time would freeze the answer
# too early if some future refactor moved the hook's position.
def _default_full_arms() -> Tuple[str, ...]:
    if "hrmv2_act" in PROVIDER_BUILDERS:
        return ARM_NAMES + ("hrmv2_act",)
    return ARM_NAMES


def _resolve_scale_preset(name: str, cfg: Optional[C11MissionConfig] = None) -> dict:
    """Resolve `--scale-preset smoke|local|big` into a dict of run-parameter
    overrides: `{"n_train_worlds", "n_test_worlds", "epochs", "budgets",
    "arms", "cells"}`.

    `smoke`: n_train_worlds=2, n_test_worlds=2, epochs=1, budgets=(400,
    3200), cells restricted to config A at K in {0,2} ONLY, arms
    mlp+gnn (no hrmv2 -- too slow for smoke, per the task spec's explicit
    instruction).

    `local`: the full pre-registered grid -- every field pulled straight
    from `cfg` (`n_train_worlds`, `n_test_worlds`, `epochs`,
    `budgets_grid`), `cells` = the full 11-cell `build_cell_grid()`, `arms`
    = `_default_full_arms()` (5 natives + `hrmv2_act` if registered).

    `big` (Task 10 scaled addendum): cells restricted to config A at K in
    {2, 8} ONLY (the two pre-registered addendum cells -- a shallow-K cell
    (K=2) and the collapse cell (K=8, where hrm_trace collapsed to constant
    cap-predictions at matched scale, per the diagnosis addendum's Finding
    2), arms = `BIG_ARM_NAMES` (`unet_film_big`, `hrm_trace_big`) ONLY --
    NEVER blended with the matched natives (the matched arms already have
    their own results; this preset exists to answer "does scale change the
    ranking/escape the collapse" for exactly these two arms, at exactly
    these two cells). `n_train_worlds`/`n_test_worlds`/`epochs`/`budgets`
    are the FULL pre-registered values from `cfg` (same recipe scale as
    `local` -- only the arm width/depth and the arm+cell selection differ,
    per the task spec: 'same recipe, same TEST cells'). Seeds are NOT part
    of this dict (as with `smoke`/`local`) -- the CLI's own `--seeds`
    default (`0,1,2`) already matches the addendum's pre-registered 3
    seeds, so no preset-level override is needed."""
    cfg = cfg or C11MissionConfig()
    if name == "smoke":
        smoke_cells = [c for c in build_cell_grid(cfg) if c["config_label"] == "A" and c["K"] in (0, 2)]
        return {
            "n_train_worlds": 2,
            "n_test_worlds": 2,
            "epochs": 1,
            "budgets": (400, 3200),
            "arms": ("mlp", "gnn"),
            "cells": smoke_cells,
        }
    if name == "local":
        return {
            "n_train_worlds": cfg.n_train_worlds,
            "n_test_worlds": cfg.n_test_worlds,
            "epochs": cfg.epochs,
            "budgets": tuple(cfg.budgets_grid),
            "arms": _default_full_arms(),
            "cells": build_cell_grid(cfg),
        }
    if name == "big":
        big_cells = [c for c in build_cell_grid(cfg) if c["config_label"] == "A" and c["K"] in (2, 8)]
        return {
            "n_train_worlds": cfg.n_train_worlds,
            "n_test_worlds": cfg.n_test_worlds,
            "epochs": cfg.epochs,
            "budgets": tuple(cfg.budgets_grid),
            "arms": BIG_ARM_NAMES,
            "cells": big_cells,
        }
    raise ValueError(f"unknown scale preset {name!r}; expected 'smoke', 'local', or 'big'")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="C11 compositional-mission PRM heuristics.")
    parser.add_argument("--mode", choices=["train", "eval", "analyze", "full"], required=True)
    parser.add_argument("--out-dir", default="runs/c11_local")
    parser.add_argument("--arms", default=None)
    parser.add_argument("--configs", default="A,B,C")
    parser.add_argument("--k-values", default="0,2,4,8")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-train-worlds", type=int, default=None)
    parser.add_argument("--n-test-worlds", type=int, default=None)
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--scale-preset", choices=["smoke", "local", "big"], default=None)
    parser.add_argument(
        "--canonical-results", action="store_true",
        help="Additionally write C11_RESULTS.md to the canonical documentation path "
             "(docs/experiments/continuous/c11/results/C11_RESULTS.md). ONLY this flag ever writes "
             "there -- analyze/full otherwise write exclusively under --out-dir.",
    )
    parser.add_argument(
        "--addendum", default=None,
        help="Path to a hand-authored markdown file injected VERBATIM as a final "
             "'## Diagnosis addendum' section of C11_RESULTS.md (analyze/full modes; "
             "applies to the out_dir copy AND, with --canonical-results, the canonical "
             "copy). The intended carrier for the post-diagnosis interpretation after "
             "a K=0 continuity failure.",
    )
    args = parser.parse_args(argv)

    cfg = C11MissionConfig()

    # --scale-preset supplies DEFAULTS for the run-shape flags; any
    # EXPLICITLY-passed flag of the same name overrides it (checked via
    # `argv`'s raw presence, since argparse itself can't distinguish "user
    # passed --epochs 1" from "user didn't pass --epochs" once both already
    # collapsed to the same default=None sentinel otherwise).
    preset = _resolve_scale_preset(args.scale_preset, cfg) if args.scale_preset else None

    raw_argv = list(argv) if argv is not None else sys.argv[1:]

    def _explicit(flag: str) -> bool:
        # Known edge (T8-review minor 3, accepted as-is): argparse's prefix
        # abbreviation (e.g. `--epoch` for `--epochs`) is exact-match-missed
        # here, so an ABBREVIATED explicit flag would lose to the preset.
        return any(a == flag or a.startswith(flag + "=") for a in raw_argv)

    if preset is not None:
        cells = preset["cells"]
        arms = tuple(_parse_csv_arg(args.arms)) if _explicit("--arms") else preset["arms"]
        n_train_worlds = args.n_train_worlds if _explicit("--n-train-worlds") else preset["n_train_worlds"]
        n_test_worlds = args.n_test_worlds if _explicit("--n-test-worlds") else preset["n_test_worlds"]
        epochs = args.epochs if _explicit("--epochs") else preset["epochs"]
        budgets = tuple(int(b) for b in _parse_csv_arg(args.budgets)) if _explicit("--budgets") else preset["budgets"]
    else:
        configs = _parse_csv_arg(args.configs)
        k_values = [int(k) for k in _parse_csv_arg(args.k_values)]
        cells = _select_cells(configs, k_values, cfg)
        arms = tuple(_parse_csv_arg(args.arms)) if args.arms else _default_full_arms()
        n_train_worlds = args.n_train_worlds
        n_test_worlds = args.n_test_worlds
        epochs = args.epochs
        budgets = tuple(int(b) for b in _parse_csv_arg(args.budgets)) if args.budgets else None

    seeds = tuple(int(s) for s in _parse_csv_arg(args.seeds))

    if args.mode == "train":
        run_train(args.out_dir, arms=arms, cells=cells, seeds=seeds, cfg=cfg,
                   n_train_worlds=n_train_worlds, epochs=epochs)
    elif args.mode == "eval":
        run_eval(args.out_dir, cells=cells, cfg=cfg, n_test_worlds=n_test_worlds,
                  budgets=budgets, arms=arms)
    elif args.mode == "analyze":
        run_analyze(args.out_dir, cfg, addendum_path=args.addendum)
        if args.canonical_results:
            canonical_path = (
                Path(__file__).resolve().parents[2]
                / "docs" / "experiments" / "continuous" / "c11" / "results" / "C11_RESULTS.md"
            )
            _rerun_write_results_doc_at(args.out_dir, cfg, str(canonical_path),
                                         addendum_path=args.addendum)
    elif args.mode == "full":
        run_full(args.out_dir, arms=arms, cells=cells, seeds=seeds, cfg=cfg,
                  n_train_worlds=n_train_worlds, n_test_worlds=n_test_worlds,
                  epochs=epochs, budgets=budgets, addendum_path=args.addendum)
        if args.canonical_results:
            canonical_path = (
                Path(__file__).resolve().parents[2]
                / "docs" / "experiments" / "continuous" / "c11" / "results" / "C11_RESULTS.md"
            )
            _rerun_write_results_doc_at(args.out_dir, cfg, str(canonical_path),
                                         addendum_path=args.addendum)


def _rerun_write_results_doc_at(out_dir: str, cfg: C11MissionConfig, path: str,
                                 addendum_path: Optional[str] = None) -> None:
    """Regenerate the results doc at an ADDITIONAL `path` from the SAME
    already-written `out_dir` artifacts `run_analyze` just produced, via
    the shared `_assemble_analysis` core (byte-identical analysis to what
    `run_analyze` itself just wrote under `out_dir` -- re-reading rather
    than threading a second `md_path` parameter through `run_analyze`
    keeps `run_analyze`'s OWN contract to always write to `{out_dir}/
    C11_RESULTS.md` unconditionally, exactly mirroring the probe's
    `run_probe(md_path=None)` -> default-under-out_dir convention while
    keeping the CANONICAL-path write behind ITS OWN explicit call site
    here, one level up, guarded by `--canonical-results` alone -- the same
    two-call-site split `main`'s own `--probe` branch uses for
    `run_probe`). `addendum_path`: same contract as `run_analyze`'s (the
    CLI passes the SAME `--addendum` to both call sites, so the canonical
    copy and the out_dir copy carry the same addendum)."""
    addendum_md = Path(addendum_path).read_text(encoding="utf-8") if addendum_path else None
    a = _assemble_analysis(out_dir, cfg)
    write_results_doc(a["g1_analysis"], a["k0"], a["g1"], a["g2"], a["g2b"], a["meta"], path,
                       addendum_md=addendum_md)
    print(f"[c11-mission analyze] wrote canonical results doc: {path}", flush=True)


if __name__ == "__main__":
    # Delegate to the CANONICAL module object rather than calling main() on
    # `__main__`. Script execution creates a SECOND module object
    # (`__main__`) with its own copy of every module-level global, including
    # PROVIDER_BUILDERS. The hrmv2 module registers its providers via
    # `import continuous_prm_c11_mission` (the canonical object), so under
    # script execution the registrations land in the canonical registry
    # while run_train/run_eval executing on `__main__` consult the empty
    # copy -- `--arms hrmv2_act` then dies with "unknown arm name" even
    # though every import-based caller (tests, the estimator) works. The T9
    # launch hit exactly this. Delegation makes all CLI state live in the
    # one canonical module object.
    import continuous_prm_c11_mission as _canonical

    _canonical.main()
