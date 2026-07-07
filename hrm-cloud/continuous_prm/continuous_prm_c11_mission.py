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

See docs/superpowers/specs/2026-07-07-c11-compositional-mission-design.md
(sections 3/4/6/7 are authoritative on the I/O contract, matched recipe,
and eval/stats conventions) and docs/superpowers/plans/2026-07-07-c11-mission.md
(Tasks 1-4).
"""
from __future__ import annotations

import argparse
import math
import os
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

    Both pieces are REQUIRED (a build-only registration would crash at
    provider call time, not registration time -- the failure the T4 review
    flagged). Registering a native arm name raises ValueError.

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
    PROVIDER_BUILDERS[name] = {"build": build, "forward_batch": forward_batch}


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
    entry = PROVIDER_BUILDERS.get(name)
    if entry is not None:
        return entry["build"](cfg)
    raise ValueError(f"unknown arm name {name!r}; expected one of {ARM_NAMES} or a registered provider builder")


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
    if arm_name in ("mlp", "hrm_trace", "onlstm_trace"):
        return _forward_mlp_or_trace(arm_name, model, bundle, states, cfg, device)
    if arm_name == "unet_film":
        raw = _forward_unet_film(model, bundle, states, cfg, device)
        return _softplus_clamp(raw, cap=cfg.residual_cap)
    if arm_name == "gnn":
        return _forward_gnn(model, bundle, states, cfg, device)
    entry = PROVIDER_BUILDERS.get(arm_name)
    if entry is not None:
        return entry["forward_batch"](model, bundle, states, cfg, device)
    raise ValueError(f"unknown arm name {arm_name!r}; expected one of {ARM_NAMES} or a registered provider builder")


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

                state_dict, meta = train_arm(arm_name, cell, bundles, seed, cfg=cfg, epochs=epochs)

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
    ascending."""
    cfg = cfg or C11MissionConfig()
    n_worlds = int(n_test_worlds) if n_test_worlds is not None else int(cfg.n_test_worlds)
    eval_budgets = tuple(sorted(int(b) for b in (budgets if budgets is not None else cfg.budgets_grid)))
    out_path = Path(out_dir)

    manifest = _load_manifest(out_path)

    all_records: List[dict] = []
    binding_k0_updates: Dict[str, dict] = {}

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

        legsum_records_for_calib: List[dict] = []

        for world_idx, bundle in enumerate(bundles):
            arm_h_arrays: List[Tuple[str, np.ndarray, bool, str]] = []
            for arm_name in REFERENCE_ARMS:
                h_arr = _reference_h_array(arm_name, bundle)
                arm_h_arrays.append((arm_name, h_arr, True, ""))
            for arm_name, provider, train_seed_val in providers:
                h_arr = provider(bundle)
                arm_h_arrays.append((arm_name, h_arr, False, str(train_seed_val)))

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

    if binding_k0_updates:
        binding_path = out_path / "binding_k0.json"
        existing = C.read_json(binding_path) if binding_path.exists() else {}
        existing.update(binding_k0_updates)
        C.write_json(binding_path, existing)


# ---------------------------------------------------------------------------
# Task 4: CLI stub.
# ---------------------------------------------------------------------------
#
# `--mode train|eval` only (T8 adds `analyze`/`full` -- this parser's
# `choices` and the cell-selection helper below are written so adding those
# modes later is a pure extension: new `choices` entries + new branches in
# `main`, no restructuring of what's here).

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


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="C11 compositional-mission PRM heuristics.")
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--out-dir", default="runs/c11_local")
    parser.add_argument("--arms", default=",".join(ARM_NAMES))
    parser.add_argument("--configs", default="A,B,C")
    parser.add_argument("--k-values", default="0,2,4,8")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-train-worlds", type=int, default=None)
    parser.add_argument("--n-test-worlds", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = C11MissionConfig()
    arms = tuple(_parse_csv_arg(args.arms))
    configs = _parse_csv_arg(args.configs)
    k_values = [int(k) for k in _parse_csv_arg(args.k_values)]
    seeds = tuple(int(s) for s in _parse_csv_arg(args.seeds))
    cells = _select_cells(configs, k_values, cfg)

    if args.mode == "train":
        run_train(args.out_dir, arms=arms, cells=cells, seeds=seeds, cfg=cfg,
                   n_train_worlds=args.n_train_worlds, epochs=args.epochs)
    elif args.mode == "eval":
        run_eval(args.out_dir, cells=cells, cfg=cfg, n_test_worlds=args.n_test_worlds,
                  arms=arms)


if __name__ == "__main__":
    main()
