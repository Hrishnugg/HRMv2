#!/usr/bin/env python3
"""C11 G0-H headroom probe -- Tasks 1-3: mission layer, product graph, exact
oracle, product A*, the matched three-arm eval, and the keys->doors
stage-dependent adjacency config.

Measures whether compositional missions (visit K ordered waypoints, then reach
the goal) on the existing hard-map roadmaps create real heuristic headroom for
A* search, ahead of any learning. Task 1 built the ground-truth machinery:
mission sampling, the product-graph transition rule, an exact backward-
Dijkstra oracle over the product graph, and the two admissible heuristics
(`h_next`, `h_legsum`). Task 2 adds `astar_product` (budget-limited A* on the
product graph, mirroring `C.astar_search`'s conventions), binding-budget
calibration, and `eval_cell` (the matched three-arm eval for one config x K
cell). Task 3 adds config C: `place_doors` / `DoorPlacement` /
`door_adj_valid_factory`, placing two "keys->doors" rectangles that block
roadmap edges until the corresponding waypoint is completed -- a genuinely
stage-dependent adjacency that euclid-based heuristics cannot see, plugged
into `product_oracle` / `astar_product` / `eval_cell` via the existing
`adj_valid` / `adj_valid_factory` hooks.

New-file-only; reuses `continuous_prm_common` (worlds, PRM, INF) and
`continuous_prm_c7_hard_maps` (hard-map suites) by import. Does not modify
either. See docs/superpowers/plans/2026-07-07-c11-headroom-probe.md.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

import continuous_prm_common as C
import continuous_prm_c7_hard_maps as H7


@dataclass
class C11ProbeConfig:
    """Shared knobs for the C11 headroom probe (mission sampling + measurement)."""

    roadmap_nodes: int = 192
    roadmap_k: int = 7
    min_start_goal_dist_frac: float = 0.45
    min_separation_frac: float = 0.25
    min_separation_frac_relaxed: float = 0.15
    max_sample_attempts: int = 200
    n_worlds: int = 25
    budgets: Sequence[int] = (100, 200, 400, 800, 1600, 3200)
    k_values: Sequence[int] = (2, 4, 8)

    # Keys->doors config C knobs (Task 3).
    n_doors: int = 2
    door_halfwidth_frac: float = 0.02
    door_halfheight_frac: float = 0.10
    door_t_range: tuple = (0.3, 0.7)
    door_max_attempts: int = 20
    door_min_blocked_edges: int = 3


# ---------------------------------------------------------------------------
# Mission layer
# ---------------------------------------------------------------------------

def sample_mission(rm, world, K: int, seed: int, cfg: Optional[C11ProbeConfig] = None) -> List[int]:
    """Sample K distinct roadmap-node waypoints for a compositional mission.

    Waypoints are drawn seeded from nodes connected to the goal (excluding
    start node 0 and goal node 1), subject to a minimum pairwise euclidean
    separation of `min_separation_frac * side` (retried up to
    `max_sample_attempts` times; relaxed to `min_separation_frac_relaxed *
    side` if unfillable at the stricter threshold). Returns a list of K
    distinct roadmap node indices in visiting order.

    Raises RuntimeError if K waypoints satisfying even the relaxed separation
    cannot be found -- callers should treat this as "skip this world".
    """
    cfg = cfg or C11ProbeConfig()
    side = float(world.side_len)
    rng = np.random.RandomState(seed)

    candidates = np.where(rm.connected_to_goal)[0]
    candidates = candidates[(candidates != 0) & (candidates != 1)]
    if len(candidates) < K:
        raise RuntimeError(f"only {len(candidates)} eligible nodes, need {K}")

    for min_sep_frac in (cfg.min_separation_frac, cfg.min_separation_frac_relaxed):
        min_sep = min_sep_frac * side
        wp = _try_sample_separated(rm, candidates, K, min_sep, rng, cfg.max_sample_attempts)
        if wp is not None:
            return wp
    raise RuntimeError(f"could not sample {K} waypoints with min separation even relaxed")


def _try_sample_separated(rm, candidates: np.ndarray, K: int, min_sep: float,
                           rng: np.random.RandomState, max_attempts: int) -> Optional[List[int]]:
    for _ in range(max_attempts):
        chosen = rng.choice(candidates, size=K, replace=False)
        pts = rm.points[chosen]
        ok = True
        for a in range(K):
            for b in range(a + 1, K):
                if float(np.linalg.norm(pts[a] - pts[b])) < min_sep:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return [int(x) for x in chosen]
    return None


# ---------------------------------------------------------------------------
# Product graph transition rule
# ---------------------------------------------------------------------------

def transition_stage(s: int, j: int, wp: Sequence[int]) -> int:
    """Forward transition rule: moving onto roadmap node `j` while at stage
    `s` advances to stage `s+1` iff `s < K` and `j == wp[s]`; otherwise stage
    is unchanged. Waypoints are distinct, so no transition ever cascades
    (advancing at most one stage per edge traversal)."""
    K = len(wp)
    if s < K and j == wp[s]:
        return s + 1
    return s


def mission_target(s: int, wp: Sequence[int], goal_idx: int = 1) -> int:
    """`tgt(s)`: the next thing to reach from stage s -- the s-th waypoint if
    stages remain, else the goal node."""
    K = len(wp)
    return wp[s] if s < K else goal_idx


# ---------------------------------------------------------------------------
# Exact backward-Dijkstra oracle over the product graph
# ---------------------------------------------------------------------------

def product_oracle(rm, wp: Sequence[int], adj_valid: Optional[Callable[[int, int, int], bool]] = None,
                    goal_idx: int = 1, start_idx: int = 0) -> np.ndarray:
    """Exact cost-to-go h*(i, s) for every product state, via backward
    Dijkstra from the product goal state (goal_idx, K).

    Product state: (node i, stage s), s in 0..K = number of waypoints
    completed. Forward transition along physical roadmap edge i->j (cost =
    euclidean edge weight from rm.adj) is `transition_stage(s, j, wp)`.

    `adj_valid(i, j, s)` is an optional stage-dependent edge-validity hook:
    when provided, the physical edge i<->j is only usable while departing
    node i at stage s (i.e. usable for the forward transition FROM (i, s)).
    Defaults to None, meaning every edge is valid at every stage (Task 3
    supplies a real predicate for the keys->doors config).

    Backward search direction / predecessor rule (see report): from a popped
    product state (j, s'), for every physical roadmap neighbor i of j with
    edge weight w, two distinct predecessor states are considered:
      * Case A ("non-completing" arrival): predecessor (i, s') is valid iff
        moving i->j while at stage s' would NOT complete a waypoint, i.e.
        `transition_stage(s', j, wp) == s'` (equivalently: s' >= K or
        j != wp[s']).
      * Case B ("completing" arrival): predecessor (i, s'-1) is valid iff
        s' >= 1 and moving i->j while at stage s'-1 completes waypoint
        s'-1, i.e. `j == wp[s'-1]` (equivalently transition_stage(s'-1, j,
        wp) == s').
    Both cases can be simultaneously valid for the same physical edge i-j
    (they name different product states, (i,s') and (i,s'-1), reached via
    the same edge under different departure stages), so both are always
    checked independently. When `adj_valid` is supplied, a candidate
    predecessor (i, s) is only relaxed if `adj_valid(i, j, s)` holds (the
    edge must be usable when DEPARTING (i, s) toward j).

    Returns an (K+1, N) array `oracle[s, i] = h*(i, s)`; unreachable entries
    are `C.INF`.
    """
    K = len(wp)
    N = rm.points.shape[0]
    dist = np.full((K + 1, N), C.INF, dtype=np.float64)
    dist[K, goal_idx] = 0.0
    heap: List = [(0.0, K, goal_idx)]

    while heap:
        d, s_pop, j = heapq.heappop(heap)
        if d != dist[s_pop, j]:
            continue

        for i, w in rm.adj[j]:
            # Case A: predecessor (i, s_pop), edge i->j does not complete a
            # waypoint when departing stage s_pop.
            if transition_stage(s_pop, j, wp) == s_pop:
                if adj_valid is None or adj_valid(i, j, s_pop):
                    nd = d + w
                    if nd < dist[s_pop, i]:
                        dist[s_pop, i] = nd
                        heapq.heappush(heap, (nd, s_pop, i))

            # Case B: predecessor (i, s_pop - 1), edge i->j completes
            # waypoint (s_pop - 1) when departing that stage.
            if s_pop >= 1 and transition_stage(s_pop - 1, j, wp) == s_pop:
                s_pred = s_pop - 1
                if adj_valid is None or adj_valid(i, j, s_pred):
                    nd = d + w
                    if nd < dist[s_pred, i]:
                        dist[s_pred, i] = nd
                        heapq.heappush(heap, (nd, s_pred, i))

    return dist


def mission_reachable(oracle: np.ndarray) -> bool:
    """Whether the product goal is reachable from the product start (0, 0).

    `oracle[0, 0]` is `C.INF` (1e30) when unreachable, not `float('inf')`, so
    plain `np.isfinite` is a trap here (1e30 IS finite). Use this helper
    everywhere reachability from the start is checked.
    """
    return float(oracle[0, 0]) < C.INF / 10.0


# ---------------------------------------------------------------------------
# Admissible heuristics
# ---------------------------------------------------------------------------

def h_next(rm, wp: Sequence[int], goal_idx: int = 1) -> Callable[[int, int], float]:
    """h_next(i, s) = ||p_i - p_tgt(s)||, tgt(s) = wp[s] if s < K else goal.

    Admissible by the triangle inequality: the straight-line distance to the
    very next target under-estimates any path cost to it (and hence to the
    full remaining mission, since further legs only add non-negative cost).
    """
    points = rm.points

    def _h(i: int, s: int) -> float:
        tgt = mission_target(s, wp, goal_idx)
        return float(np.linalg.norm(points[i] - points[tgt]))

    return _h


def h_legsum(rm, wp: Sequence[int], goal_idx: int = 1) -> Callable[[int, int], float]:
    """h_legsum(i, s) = ||p_i - p_tgt(s)|| + sum of straight-line remaining
    leg lengths along the chain wp[s] -> wp[s+1] -> ... -> wp[K-1] -> goal.

    Admissible: each term is a straight-line lower bound on the true cost of
    its corresponding leg, and h_legsum >= h_next (it is h_next plus
    additional non-negative terms), while still <= h*(i, s).
    """
    K = len(wp)
    points = rm.points
    chain = list(wp) + [goal_idx]

    # Precompute leg[t] = ||p_chain[t] - p_chain[t+1]|| for t = 0..K-1
    # (chain[0]=wp[0], ..., chain[K-1]=wp[K-1], chain[K]=goal).
    leg_lengths = [
        float(np.linalg.norm(points[chain[t]] - points[chain[t + 1]]))
        for t in range(K)
    ]
    # remaining_from[s] = sum_{t=s}^{K-1} leg_lengths[t], for s in 0..K.
    remaining_from = [0.0] * (K + 1)
    for s in range(K - 1, -1, -1):
        remaining_from[s] = leg_lengths[s] + remaining_from[s + 1]

    def _h(i: int, s: int) -> float:
        tgt = mission_target(s, wp, goal_idx)
        to_next = float(np.linalg.norm(points[i] - points[tgt]))
        return to_next + remaining_from[s]

    return _h


# ---------------------------------------------------------------------------
# Task 2 -- product A*, binding-budget calibration, matched three-arm eval.
# ---------------------------------------------------------------------------

def astar_product(adj, wp: Sequence[int], h, budget: int,
                   adj_valid: Optional[Callable[[int, int, int], bool]] = None,
                   start=(0, 0)) -> dict:
    """Budget-limited A* on the product graph, mirroring `C.astar_search`'s
    conventions exactly but keyed on state `(node, stage)`.

    `h` is indexed `h[s, i]` (an (K+1, N) ndarray -- both the exact oracle
    array and materialized `h_next`/`h_legsum` arrays satisfy this).

    `adj_valid(i, j, s)`, if given, is checked with the DEPARTURE stage: an
    edge i->j is usable from state (i, s) iff `adj_valid(i, j, s)` -- the same
    convention `product_oracle` documents (Case A/B there validate the
    predecessor's departure stage; forward, that is simply the current
    state's stage). Defaults to None (every edge always usable).

    Goal state is `(1, K)`, K = len(wp). Returns the same dict shape as
    `C.astar_search`: `{"found", "cost", "expansions", "closed"}`, with
    `closed` = the total count of closed product states.
    """
    K = len(wp)
    N = len(adj)
    start_i, start_s = start
    goal_idx, goal_stage = 1, K

    g = np.full((K + 1, N), C.INF, dtype=np.float64)
    g[start_s, start_i] = 0.0
    closed = np.zeros((K + 1, N), dtype=np.bool_)

    h_start = float(h[start_s, start_i])
    heap: List = [(h_start, 0.0, start_s, start_i)]
    expansions = 0

    while heap and expansions < budget:
        _, cur_g, s, i = heapq.heappop(heap)
        if closed[s, i]:
            continue
        if cur_g != g[s, i]:
            continue
        closed[s, i] = True
        expansions += 1
        if i == goal_idx and s == goal_stage:
            return {"found": True, "cost": float(g[s, i]), "expansions": expansions,
                    "closed": int(closed.sum())}
        for j, w in adj[i]:
            if adj_valid is not None and not adj_valid(i, j, s):
                continue
            s2 = transition_stage(s, j, wp)
            if closed[s2, j]:
                continue
            ng = cur_g + w
            if ng < g[s2, j]:
                hv = float(h[s2, j])
                if hv >= C.INF / 10.0:
                    # Unreachable under this heuristic -- do not push a dead
                    # successor (matches admissible pruning; the oracle arm
                    # needs this to avoid pushing states behind closed doors).
                    continue
                g[s2, j] = ng
                f = ng + hv
                heapq.heappush(heap, (f, ng, s2, j))

    return {"found": False, "cost": float("nan"), "expansions": expansions,
            "closed": int(closed.sum())}


def calibrate_binding_budget(records: Sequence[dict], budgets: Sequence[int]) -> tuple:
    """Pick the binding budget for one (config, K) cell from its records.

    The binding budget is the LOWEST budget in `budgets` where the
    `h_legsum` arm's success rate (fraction of found=True records at that
    budget) is >= 0.05. If no budget qualifies, returns the largest budget
    with the DEGENERATE flag set. Returns `(budget, degenerate)`.
    """
    budgets = tuple(sorted(budgets))
    for b in budgets:
        legsum_at_b = [r for r in records if r.get("arm") == "h_legsum" and r.get("budget") == b]
        if not legsum_at_b:
            continue
        success_rate = sum(1 for r in legsum_at_b if r["found"]) / len(legsum_at_b)
        if success_rate >= 0.05:
            return (b, False)
    return (budgets[-1], True)


def eval_cell(spec_name: str, config_idx: int, K: int, n_worlds: int = 25,
              budgets: Sequence[int] = (100, 200, 400, 800, 1600, 3200),
              adj_valid_factory=None, max_world_attempts: int = 200) -> List[dict]:
    """Matched three-arm eval (`h_next`, `h_legsum`, `h_oracle`) for one
    (config, K) cell: `n_worlds` valid worlds, every arm run at every budget
    on the SAME (rm, wp, adj_valid) per world.

    World seeds follow the pre-registered formula
    `seed = 1234 + 7919*world_idx + 104729*config_idx + 15485863*K`, trying
    `world_idx = 0, 1, 2, ...` (capped at `max_world_attempts`) until
    `n_worlds` valid worlds are collected. A world is skipped (world_idx
    advances, attempt counted) if `C.build_world`/`C.build_prm` fail,
    `sample_mission` raises RuntimeError, the optional `adj_valid_factory`
    fails to place doors, or the product goal is unreachable under the exact
    oracle (`mission_reachable`).

    Returns one dict per (world x arm x budget), each carrying `config`,
    `config_idx`, `K`, `world_idx`, `seed`, `arm`, `budget`, `found`, `cost`,
    `expansions`, `closed`, `opt_cost`. Any found cost is asserted to equal
    `opt_cost` within 1e-6 (all three heuristics are admissible, so this
    catches wiring bugs at run time). Deterministic: identical inputs
    produce identical records (no RNG outside the seed formula).

    Note: a record's `world_idx` is the valid-world ordinal (0..n_worlds-1)
    at which it was collected, NOT the seed-formula attempt index (worlds
    skipped for build/mission/door failures do not consume a `world_idx`
    slot) -- the recorded `seed` field is the ground truth for reproducing
    any individual record's (world, roadmap, mission) triple.
    """
    H7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    spec = specs[spec_name]
    roadmap_cfg = C.RoadmapConfig(n_nodes=192, k_neighbors=7)

    records: List[dict] = []
    world_idx = 0
    attempts = 0
    worlds_found = 0

    while worlds_found < n_worlds:
        if attempts >= max_world_attempts:
            raise RuntimeError(
                f"exhausted {max_world_attempts} attempts finding valid worlds for "
                f"{spec_name} config_idx={config_idx} K={K}: only found {worlds_found}/{n_worlds}"
            )
        seed = 1234 + 7919 * world_idx + 104729 * config_idx + 15485863 * K
        attempts += 1
        world_idx += 1

        world = C.build_world(spec, seed, 0.45)
        if world is None:
            continue
        rm = C.build_prm(world, roadmap_cfg, seed + 17)
        if rm is None:
            continue
        try:
            wp = sample_mission(rm, world, K, seed)
        except RuntimeError:
            continue

        adj_valid = None
        if adj_valid_factory is not None:
            try:
                adj_valid = adj_valid_factory(rm, world, wp, seed)
            except RuntimeError:
                # Placement exhausted its resample budget (e.g. door
                # validation never succeeded on this world) -- same
                # skip-world convention as sample_mission's RuntimeError.
                continue
            if adj_valid is None:
                continue

        oracle = product_oracle(rm, wp, adj_valid)
        if not mission_reachable(oracle):
            continue

        opt_cost = float(oracle[0, 0])
        K_ = len(wp)
        N = rm.points.shape[0]
        h_next_fn = h_next(rm, wp)
        h_legsum_fn = h_legsum(rm, wp)
        h_next_arr = np.array([[h_next_fn(i, s) for i in range(N)] for s in range(K_ + 1)])
        h_legsum_arr = np.array([[h_legsum_fn(i, s) for i in range(N)] for s in range(K_ + 1)])

        arms = (("h_next", h_next_arr), ("h_legsum", h_legsum_arr), ("h_oracle", oracle))
        for budget in budgets:
            for arm_name, h_arr in arms:
                res = astar_product(rm.adj, wp, h_arr, budget, adj_valid=adj_valid)
                if res["found"]:
                    assert _approx_eq(res["cost"], opt_cost), (
                        f"{spec_name} config_idx={config_idx} K={K} world_idx={worlds_found} "
                        f"arm={arm_name} budget={budget}: found cost {res['cost']} != "
                        f"opt_cost {opt_cost} (admissible heuristics must recover the optimum)"
                    )
                records.append({
                    "config": spec_name,
                    "config_idx": config_idx,
                    "K": K,
                    "world_idx": worlds_found,
                    "seed": seed,
                    "arm": arm_name,
                    "budget": budget,
                    "found": bool(res["found"]),
                    "cost": float(res["cost"]),
                    "expansions": int(res["expansions"]),
                    "closed": int(res["closed"]),
                    "opt_cost": opt_cost,
                })
        worlds_found += 1

    return records


def _approx_eq(a: float, b: float, abs_tol: float = 1e-6) -> bool:
    """Local abs-tolerance equality check (avoids a pytest import in the
    library module; used only for the internal optimality sanity assert)."""
    return abs(a - b) <= abs_tol


# ---------------------------------------------------------------------------
# Task 3 -- keys->doors stage-dependent adjacency.
# ---------------------------------------------------------------------------
#
# D = 2 doors, d in {0, 1}. Door d's key is waypoint d: the edges it blocks
# are usable only once the mission has completed waypoint d, i.e. usable at
# stage s > d; blocked for s <= d. Doors are placed as thin rectangles across
# the straight-line corridor from wp[d] to the mission's NEXT target after
# wp[d] (wp[d+1] if it exists, else the goal node), and only ever REMOVE
# roadmap edges from consideration -- the world, its obstacles, and the
# roadmap itself are never modified; a door is purely an edge mask consumed
# through `adj_valid(i, j, s)`.

Rect = tuple  # (xmin, ymin, xmax, ymax)


@dataclass
class DoorPlacement:
    """Result of placing config C's keys->doors on one (rm, world, wp).

    `rects[d]` is door d's axis-aligned rectangle `(xmin, ymin, xmax, ymax)`.
    `blocked[d]` is the frozenset of undirected roadmap edges `{i, j}`
    (stored as sorted `(min(i,j), max(i,j))` tuples, matching
    `continuous_prm_common.build_prm`'s own edge-set convention) that door d's
    rectangle intersects. `adj_valid(i, j, s)` is the stage-dependent
    edge-validity predicate: False iff edge {i,j} is blocked by some door d
    with `s <= d`, True otherwise.
    """

    rects: List[Rect]
    blocked: List[frozenset]

    def adj_valid(self, i: int, j: int, s: int) -> bool:
        key = (i, j) if i <= j else (j, i)
        for d, blocked_d in enumerate(self.blocked):
            if s <= d and key in blocked_d:
                return False
        return True


def _segment_intersects_rect(p1: np.ndarray, p2: np.ndarray, rect: Rect) -> bool:
    """Standard segment-vs-AABB test (slab / Liang-Barsky clipping).

    Returns True iff the closed segment p1->p2 has any point (including
    either endpoint) inside or on the boundary of `rect`. Handles the
    degenerate zero-length segment (p1 == p2) as a point-in-rect test.
    """
    xmin, ymin, xmax, ymax = rect
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx, dy = x2 - x1, y2 - y1

    if dx == 0.0 and dy == 0.0:
        return xmin <= x1 <= xmax and ymin <= y1 <= ymax

    t_enter, t_exit = 0.0, 1.0
    for delta, lo, hi, coord in ((dx, xmin, xmax, x1), (dy, ymin, ymax, y1)):
        if delta == 0.0:
            # Segment is parallel to this axis's slab boundaries: if the
            # constant coordinate already lies outside [lo, hi], no
            # intersection is possible for any t; otherwise this slab
            # imposes no constraint on t.
            if coord < lo or coord > hi:
                return False
            continue
        t_lo = (lo - coord) / delta
        t_hi = (hi - coord) / delta
        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo
        t_enter = max(t_enter, t_lo)
        t_exit = min(t_exit, t_hi)
        if t_enter > t_exit:
            return False

    return t_enter <= t_exit


def _edges_blocked_by_rect(rm, rect: Rect) -> frozenset:
    """The undirected roadmap edges whose segment intersects `rect`."""
    blocked = set()
    points = rm.points
    N = points.shape[0]
    for i in range(N):
        for j, _w in rm.adj[i]:
            if j <= i:
                continue  # visit each undirected edge once
            if _segment_intersects_rect(points[i], points[j], rect):
                blocked.add((i, j))
    return frozenset(blocked)


def _point_in_rect(p: np.ndarray, rect: Rect) -> bool:
    """Inclusive point-in-rect test (boundary contact counts as contains --
    the conservative choice: a door whose edge merely touches a special
    point still gets rejected and resampled, rather than risking start/goal/
    waypoint positions sitting exactly on a door's edge)."""
    xmin, ymin, xmax, ymax = rect
    return xmin <= float(p[0]) <= xmax and ymin <= float(p[1]) <= ymax


def _door_rect_contains_any_special_point(rect: Rect, rm, wp: Sequence[int],
                                           start_idx: int = 0, goal_idx: int = 1) -> bool:
    """Whether `rect` contains the start, the goal, or any waypoint position."""
    special = [start_idx, goal_idx] + list(wp)
    return any(_point_in_rect(rm.points[idx], rect) for idx in special)


def _make_door_rect(p_from: np.ndarray, p_to: np.ndarray, t: float, side: float,
                     cfg: C11ProbeConfig) -> Rect:
    """Axis-aligned door rectangle centered at parameter `t` along the
    straight segment p_from->p_to: thin along the corridor's dominant axis,
    wide across it (half-extents `door_halfwidth_frac*side` /
    `door_halfheight_frac*side`, swapped depending on which axis dominates).
    """
    cx = float(p_from[0]) + t * (float(p_to[0]) - float(p_from[0]))
    cy = float(p_from[1]) + t * (float(p_to[1]) - float(p_from[1]))
    dx = abs(float(p_to[0]) - float(p_from[0]))
    dy = abs(float(p_to[1]) - float(p_from[1]))

    half_thin = cfg.door_halfwidth_frac * side
    half_wide = cfg.door_halfheight_frac * side
    if dx >= dy:
        # Corridor mostly horizontal: thin in x, wide in y.
        half_x, half_y = half_thin, half_wide
    else:
        # Corridor mostly vertical: thin in y, wide in x.
        half_x, half_y = half_wide, half_thin

    return (cx - half_x, cy - half_y, cx + half_x, cy + half_y)


def _door_endpoints(wp: Sequence[int], d: int, goal_idx: int = 1) -> tuple:
    """(from_node, to_node) for door d: from wp[d] to the NEXT mission target
    after wp[d] -- wp[d+1] if it exists, else the goal node."""
    K = len(wp)
    from_node = wp[d]
    to_node = wp[d + 1] if d + 1 < K else goal_idx
    return from_node, to_node


def place_doors(rm, world, wp: Sequence[int], seed: int,
                 cfg: Optional[C11ProbeConfig] = None) -> DoorPlacement:
    """Place and validate config C's `cfg.n_doors` (default 2) keys->doors.

    Door d is a thin rectangle across the straight corridor from wp[d] to
    the next mission target after wp[d] (wp[d+1], or the goal if d is the
    last waypoint). Door d blocks its intersected roadmap edges for stages
    s <= d (its key is completing waypoint d) and reopens them for s > d.

    Deterministic in (rm, world, wp, seed): attempt 0 centers every door at
    t=0.5 along its corridor; if validation fails, a seeded RNG
    (`np.random.default_rng(seed + 7717)`) draws a fresh `t ~
    U(*cfg.door_t_range)` independently for EACH door on each retry, up to
    `cfg.door_max_attempts` attempts total.

    Validation (all must hold for an attempt to be accepted):
      1. Each door blocks >= cfg.door_min_blocked_edges roadmap edges.
      2. No door's rectangle contains the start, the goal, or any waypoint
         position (inclusive/boundary-counts-as-contains test).
      3. The mission stays feasible under the resulting adj_valid
         (`mission_reachable(product_oracle(rm, wp, adj_valid))`).

    Raises RuntimeError if no attempt validates within `door_max_attempts`
    -- callers should treat this as "skip this world" (same convention as
    `sample_mission`).
    """
    cfg = cfg or C11ProbeConfig()
    side = float(world.side_len)
    D = cfg.n_doors
    K = len(wp)
    if K < D:
        raise RuntimeError(f"need at least {D} waypoints to place {D} doors, got K={K}")

    rng = np.random.default_rng(seed + 7717)

    for attempt in range(cfg.door_max_attempts):
        ts = [0.5] * D if attempt == 0 else [
            float(rng.uniform(cfg.door_t_range[0], cfg.door_t_range[1])) for _ in range(D)
        ]

        rects: List[Rect] = []
        for d in range(D):
            from_node, to_node = _door_endpoints(wp, d)
            rect = _make_door_rect(rm.points[from_node], rm.points[to_node], ts[d], side, cfg)
            rects.append(rect)

        if any(_door_rect_contains_any_special_point(r, rm, wp) for r in rects):
            continue

        blocked = [_edges_blocked_by_rect(rm, r) for r in rects]
        if any(len(b) < cfg.door_min_blocked_edges for b in blocked):
            continue

        placement = DoorPlacement(rects=rects, blocked=blocked)
        oracle = product_oracle(rm, wp, placement.adj_valid)
        if not mission_reachable(oracle):
            continue

        return placement

    raise RuntimeError(
        f"could not place {D} valid doors within {cfg.door_max_attempts} attempts "
        f"(K={K}, seed={seed})"
    )


def door_adj_valid_factory(rm, world, wp: Sequence[int], seed: int,
                            cfg: Optional[C11ProbeConfig] = None):
    """`adj_valid_factory` conforming to `eval_cell`'s hook contract.

    Called as `adj_valid_factory(rm, world, wp, seed)` (see `eval_cell`'s
    world loop): returns the placement's `adj_valid` callable on success.
    `place_doors`'s RuntimeError (placement exhausted its resample attempts)
    is allowed to propagate -- `eval_cell` catches it as a world-skip, the
    same convention already used for `sample_mission`'s RuntimeError.

    Pass this directly as `eval_cell("C_hard_maze", 2, K, ...,
    adj_valid_factory=door_adj_valid_factory)` to run config C.
    """
    placement = place_doors(rm, world, wp, seed, cfg)
    return placement.adj_valid
