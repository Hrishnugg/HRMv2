#!/usr/bin/env python3
"""C11 G0-H headroom probe -- Task 1: mission layer, product graph, exact oracle.

Measures whether compositional missions (visit K ordered waypoints, then reach
the goal) on the existing hard-map roadmaps create real heuristic headroom for
A* search, ahead of any learning. This module implements only the ground-truth
machinery: mission sampling, the product-graph transition rule, an exact
backward-Dijkstra oracle over the product graph, and the two admissible
heuristics (`h_next`, `h_legsum`) that A* (Task 2) will be compared against.

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


# ---------------------------------------------------------------------------
# Mission layer
# ---------------------------------------------------------------------------

def sample_mission(rm, world, K: int, seed: int) -> List[int]:
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
    cfg = C11ProbeConfig()
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
