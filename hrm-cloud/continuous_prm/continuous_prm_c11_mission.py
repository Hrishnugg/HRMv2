#!/usr/bin/env python3
"""C11 compositional-mission PRM heuristics -- Task 1: module skeleton,
config, and TRAIN/TEST dataset builders. Task 2: structure-exposing
encoders (trace tokens, MLP flatten, field grids, product graph).

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

See docs/superpowers/specs/2026-07-07-c11-compositional-mission-design.md
(section 3 is authoritative on the I/O contract) and
docs/superpowers/plans/2026-07-07-c11-mission.md (Tasks 1-2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

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


def _query_token(p: np.ndarray, world: C.World, s: int, K: int, cfg: C11MissionConfig) -> np.ndarray:
    """The dim-12 query-node token: `[x/side, y/side, 8 coarse ray
    distances, s/k_max, K_remaining/k_max]`."""
    side = float(world.side_len)
    steps = C.FeatureConfig().ray_steps
    tok = np.zeros(12, dtype=np.float32)
    tok[0] = p[0] / side
    tok[1] = p[1] / side
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
    tokens = [_query_token(p_i, world, s, K, cfg)]

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

    # Rays computed once per node, reused across every stage.
    query_toks = np.stack(
        [_query_token(rm.points[i], world, 0, K, cfg) for i in range(N)], axis=0
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
