"""Space-time A* and backward space-time Dijkstra over a static PRM with moving obstacles (C8).

Pure functions over numpy; agent is a point moving at v_agent (m/s) with time step dt (s).
State = (node_index, t_step) where real time = t_step * dt.
Cost = arrival time-step (makespan minimisation).
"""
from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Tuple

import numpy as np


def _edge_steps(length: float, v_agent: float, dt: float) -> int:
    return max(1, int(math.ceil(length / (v_agent * dt))))


def space_time_astar_prm(adj, points, dyn, h_table, budget, v_agent, dt, t_max,
                         start=0, goal=1) -> Dict[str, Any]:
    """Space-time A* on a static PRM with deterministically moving obstacles.

    `h_table[node, t]` is the estimated remaining TIME-TO-GO in time-steps (NOT
    arrival time). For an admissible exact oracle, derive it via
    ``oracle_time_to_go(backward_spacetime_dijkstra(...))`` -- do NOT pass the raw
    arrival-time table from ``backward_spacetime_dijkstra`` (that is ``t + remaining``,
    which overestimates time-to-go and is inadmissible).

    Parameters
    ----------
    adj : List[List[Tuple[int, float]]]
        Adjacency list: adj[u] = [(v, euclidean_length), ...]
    points : np.ndarray, shape (N, 2)
        Node positions.
    dyn : Dynamics
        Moving-obstacle feasibility checker.
    h_table : np.ndarray, shape (N, T+1)
        Admissible heuristic = estimated remaining time-to-go in time-steps
        (e.g. zeros, or ``oracle_time_to_go(backward_spacetime_dijkstra(...))``).
    budget : int
        Maximum node expansions before giving up.
    v_agent : float
        Agent speed in distance units per second.
    dt : float
        Time-step duration in seconds.
    t_max : int
        Maximum allowed time-step index.
    start, goal : int
        Node indices.

    Returns
    -------
    dict with keys: found (bool), arrival (int), expansions (int), closed (int).
    """
    Tcap = int(h_table.shape[1]) - 1

    def hval(node: int, t: int) -> float:
        return float(h_table[node, min(t, Tcap)])

    # g[(node, t)] = best known arrival time-step (= t itself for cost=time)
    g: Dict[Tuple[int, int], int] = {(start, 0): 0}
    counter = 0
    # heap entries: (f, counter, node, t_step)
    openq: List[Tuple[float, int, int, int]] = [(hval(start, 0), counter, start, 0)]
    closed: set = set()
    expansions = 0

    while openq and expansions < budget:
        f, _, u, t = heapq.heappop(openq)
        if (u, t) in closed:
            continue
        closed.add((u, t))
        expansions += 1

        if u == goal:
            return {
                "found": True,
                "arrival": int(t),
                "expansions": expansions,
                "closed": len(closed),
            }

        # --- wait action (stay at u for 1 step) ---
        if t + 1 <= t_max and dyn.node_free(points[u], t * dt, (t + 1) * dt, samples=8):
            nt = t + 1
            if (u, nt) not in closed and nt < g.get((u, nt), 1 << 30):
                g[(u, nt)] = nt
                counter += 1
                heapq.heappush(openq, (nt + hval(u, nt), counter, u, nt))

        # --- move actions ---
        for v, length in adj[u]:
            steps = _edge_steps(float(length), v_agent, dt)
            nt = t + steps
            if nt > t_max:
                continue
            if not dyn.edge_free(points[u], points[v], t * dt, nt * dt,
                                  samples=max(8, 4 * steps)):
                continue
            if (v, nt) in closed:
                continue
            if nt < g.get((v, nt), 1 << 30):
                g[(v, nt)] = nt
                counter += 1
                heapq.heappush(openq, (nt + hval(v, nt), counter, v, nt))

    return {"found": False, "arrival": -1, "expansions": expansions, "closed": len(closed)}


def space_time_focal_prm(adj, points, dyn, euclid_time, rank_h, budget, v_agent, dt, t_max,
                         w=1.0, start=0, goal=1) -> Dict[str, Any]:
    """Bounded-suboptimal focal space-time A*epsilon on a static PRM with moving obstacles.

    OPEN is ordered by the admissible f = g + euclid_time[node], where g = arrival
    time-step and euclid_time[node] is the Euclidean lower bound in time-step units
    (e.g. euclid(node, goal) / v_agent / dt), independent of t. The FOCAL band
    {entries : f <= w * f_min + 1e-12} selects expansions by minimising
    (rank_h[node, min(t, Tcap)], f, counter), where rank_h[node, t] is a
    time-to-go estimate used as the band ranker (not required to be admissible).

    The returned makespan satisfies arrival <= w * optimal_arrival when euclid_time
    is admissible (i.e. euclid_time[node] <= true remaining time-steps to goal).

    Uses rebuild-live OPEN + lazy deletion: before each expansion, stale and closed
    entries are filtered, f_min is recomputed over the live set, and the FOCAL band
    is selected fresh. State is (node, t_step) with a (node, t_step) closed set.
    Wait/move feasibility is identical to space_time_astar_prm.

    Parameters
    ----------
    adj : List[List[Tuple[int, float]]]
        Adjacency list.
    points : np.ndarray, shape (N, 2)
        Node positions.
    dyn : Dynamics
        Moving-obstacle feasibility checker.
    euclid_time : np.ndarray, shape (N,)
        Admissible per-node time-to-go lower bound in time-step units.
    rank_h : np.ndarray, shape (N, t_max+1)
        Learned time-to-go estimate used to rank entries within the FOCAL band.
    budget : int
        Maximum node expansions before giving up.
    v_agent : float
        Agent speed in distance units per second.
    dt : float
        Time-step duration in seconds.
    t_max : int
        Maximum allowed time-step index.
    w : float
        Suboptimality factor (>= 1.0). w=1.0 reduces to standard A*.
    start, goal : int
        Node indices.

    Returns
    -------
    dict with keys: found (bool), arrival (int), expansions (int), closed (int).
    """
    if w < 1.0:
        raise ValueError(f"focal w must be >= 1.0, got {w}")
    if not np.all(np.isfinite(euclid_time)):
        raise ValueError("euclid_time must be fully finite")
    if not np.all(np.isfinite(rank_h)):
        raise ValueError("rank_h must be fully finite")

    Tcap = int(rank_h.shape[1]) - 1

    # g[(node, t)] = best known arrival time-step
    g: Dict[Tuple[int, int], int] = {(start, 0): 0}
    counter = 0
    # OPEN entries: (f, g_val, node, t_step, counter)
    # f = g_val + euclid_time[node]  (admissible ordering)
    f_start = float(euclid_time[start])
    open_entries: List[Tuple[float, int, int, int, int]] = [
        (f_start, 0, start, 0, counter)
    ]
    closed: set = set()
    expansions = 0

    while open_entries and expansions < budget:
        # Rebuild live: discard stale (g changed) or already-closed entries.
        live = [
            e for e in open_entries
            if (e[2], e[3]) not in closed and e[1] == g.get((e[2], e[3]), 1 << 30)
        ]
        if not live:
            break
        open_entries = live

        f_min = min(e[0] for e in open_entries)
        threshold = w * f_min + 1e-12
        focal_set = [e for e in open_entries if e[0] <= threshold]

        # Select by (rank_h[node, min(t, Tcap)], f, counter)
        best = min(focal_set, key=lambda e: (float(rank_h[e[2], min(e[3], Tcap)]), e[0], e[4]))
        open_entries.remove(best)

        f_val, g_val, u, t, _ = best
        state = (u, t)
        if state in closed or g_val != g.get(state, 1 << 30):
            continue
        closed.add(state)
        expansions += 1

        if u == goal:
            return {
                "found": True,
                "arrival": int(t),
                "expansions": expansions,
                "closed": len(closed),
            }

        # --- wait action (stay at u for 1 step) ---
        if t + 1 <= t_max and dyn.node_free(points[u], t * dt, (t + 1) * dt, samples=8):
            nt = t + 1
            nstate = (u, nt)
            if nstate not in closed and nt < g.get(nstate, 1 << 30):
                g[nstate] = nt
                counter += 1
                nf = float(nt) + float(euclid_time[u])
                open_entries.append((nf, nt, u, nt, counter))

        # --- move actions ---
        for v, length in adj[u]:
            steps = _edge_steps(float(length), v_agent, dt)
            nt = t + steps
            if nt > t_max:
                continue
            if not dyn.edge_free(points[u], points[v], t * dt, nt * dt,
                                  samples=max(8, 4 * steps)):
                continue
            nstate = (v, nt)
            if nstate in closed:
                continue
            if nt < g.get(nstate, 1 << 30):
                g[nstate] = nt
                counter += 1
                nf = float(nt) + float(euclid_time[v])
                open_entries.append((nf, nt, v, nt, counter))

    return {"found": False, "arrival": -1, "expansions": expansions, "closed": len(closed)}


def backward_spacetime_dijkstra(adj, points, dyn, v_agent, dt, t_max, goal=1) -> np.ndarray:
    """Backward space-time Dijkstra to compute the optimal-arrival oracle.

    hstar[node, t] = minimum arrival time-step at goal reachable from (node, t),
    or +inf if unreachable within t_max.

    The DP sweeps t from t_max down to 0. Moves strictly increase t, so
    hstar[v, nt] with nt > t is already computed when we compute hstar[u, t].
    Wait (t -> t+1) also references hstar[u, t+1] which is computed first.
    Therefore a single backward sweep is exact.

    Parameters
    ----------
    adj, points, dyn, v_agent, dt, t_max, goal : same semantics as A*.

    Returns
    -------
    np.ndarray of shape (N, t_max+1), dtype float64.
    """
    n = len(adj)
    INF = float("inf")
    hstar = np.full((n, t_max + 1), INF, dtype=np.float64)

    for t in range(t_max, -1, -1):
        for u in range(n):
            if u == goal:
                hstar[u, t] = float(t)
                continue

            best = INF

            # wait: (u, t) -> (u, t+1)
            if t + 1 <= t_max and dyn.node_free(points[u], t * dt, (t + 1) * dt, samples=8):
                best = min(best, hstar[u, t + 1])

            # move: (u, t) -> (v, t+steps)
            for v, length in adj[u]:
                steps = _edge_steps(float(length), v_agent, dt)
                nt = t + steps
                if nt <= t_max and dyn.edge_free(
                    points[u], points[v], t * dt, nt * dt,
                    samples=max(8, 4 * steps)
                ):
                    best = min(best, hstar[v, nt])

            hstar[u, t] = best

    return hstar


def oracle_time_to_go(hstar: np.ndarray, t_max: int) -> np.ndarray:
    """Convert arrival-time table into remaining time-to-go.

    ttg[node, t] = hstar[node, t] - t  (steps remaining to goal).
    Unreachable entries (inf) are filled with a finite upper bound.
    Negative values (shouldn't happen for consistent DP) are clipped to 0.

    Parameters
    ----------
    hstar : np.ndarray, shape (N, T+1)
        Output of backward_spacetime_dijkstra.
    t_max : int
        Horizon used when computing hstar (used to compute the fill value).

    Returns
    -------
    np.ndarray of shape (N, T+1), dtype float64, all finite and non-negative.
    """
    t_grid = np.arange(hstar.shape[1], dtype=np.float64)[None, :]  # (1, T+1)
    ttg = hstar - t_grid
    finite = np.isfinite(ttg)
    fill = (float(ttg[finite].max()) + (t_max + 1)) if finite.any() else float(2 * (t_max + 1))
    ttg = np.where(finite, ttg, fill)
    return np.maximum(ttg, 0.0)
