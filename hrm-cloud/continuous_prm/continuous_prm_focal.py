"""Bounded-suboptimal focal A* (A*epsilon) for static weighted graphs (PRM).

Pure graph primitive: depends only on an adjacency list + per-node arrays, so
it imports nothing from continuous_prm_common. Mirrors astar_search's return
shape: {"found", "cost", "expansions", "closed"}.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def focal_astar_search(
    adj: List[List[Tuple[int, float]]],
    euclid_h: np.ndarray,
    rank_h: np.ndarray,
    budget: int,
    w: float = 1.0,
    start_idx: int = 0,
    goal_idx: int = 1,
) -> Dict[str, Any]:
    """Bounded-suboptimal focal A* (A*epsilon) on a static weighted graph.

    OPEN is ordered by the admissible f = g + euclid_h (euclid_h must be
    admissible+consistent, e.g. straight-line distance on a PRM). The FOCAL
    set {n in OPEN : f(n) <= w * f_min} is expanded by minimum rank_h (the
    learned cost-to-go estimate), tie-broken by (rank_h, f, insertion_counter).
    Returns the same dict shape as astar_search; cost <= w * optimal.
    """
    if w < 1.0:
        raise ValueError(f"focal w must be >= 1.0, got {w}")
    n = len(adj)
    g = np.full(n, np.inf, dtype=np.float64)
    g[start_idx] = 0.0
    # OPEN entries: (f, g, node, counter). counter breaks ties deterministically.
    counter = 0
    open_entries: List[Tuple[float, float, int, int]] = [
        (float(euclid_h[start_idx]), 0.0, start_idx, counter)
    ]
    closed = np.zeros(n, dtype=np.bool_)
    expansions = 0
    while open_entries and expansions < budget:
        # Drop stale entries (node closed, or g superseded) from the front-set view.
        live = [e for e in open_entries if not closed[e[2]] and e[1] == g[e[2]]]
        if not live:
            break
        open_entries = live
        f_min = min(e[0] for e in open_entries)
        threshold = w * f_min
        focal_set = [e for e in open_entries if e[0] <= threshold + 1e-12]
        # Select by (rank_h, f, counter).
        best = min(focal_set, key=lambda e: (float(rank_h[e[2]]), e[0], e[3]))
        open_entries.remove(best)
        _, cur_g, u, _ = best
        if closed[u] or cur_g != g[u]:
            continue
        closed[u] = True
        expansions += 1
        if u == goal_idx:
            return {"found": True, "cost": float(g[u]), "expansions": expansions, "closed": int(closed.sum())}
        for v, ew in adj[u]:
            if closed[v]:
                continue
            ng = g[u] + ew
            if ng < g[v]:
                g[v] = ng
                counter += 1
                open_entries.append((ng + float(euclid_h[v]), ng, v, counter))
    return {"found": False, "cost": float("nan"), "expansions": expansions, "closed": int(closed.sum())}
