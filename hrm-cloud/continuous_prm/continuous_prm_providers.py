"""Heuristic providers for the C7 integration comparison.

A provider maps (world, roadmap, goal_idx) -> a finite, non-negative per-node
heuristic array h[N]. The planner consumes h directly (astar) or as the focal
ranker alongside the admissible Euclid array (focal_astar).
"""
from __future__ import annotations

import abc

import numpy as np

import continuous_prm_common as C


def _finite_fill(vals: np.ndarray, fallback: float) -> np.ndarray:
    out = np.array(vals, dtype=np.float64)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = fallback
    return np.maximum(out, 0.0)


class HeuristicProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def node_h(self, world: "C.World", roadmap: "C.Roadmap", goal_idx: int = 1) -> np.ndarray:
        ...


class EuclidProvider(HeuristicProvider):
    name = "euclid"

    def node_h(self, world, roadmap, goal_idx: int = 1) -> np.ndarray:
        goal = roadmap.points[goal_idx]
        d = np.linalg.norm(roadmap.points - goal[None, :], axis=1)
        return _finite_fill(d, fallback=0.0)


class OracleProvider(HeuristicProvider):
    """Exact graph cost-to-go (the minimal-expansion ceiling for A* on this graph)."""
    name = "oracle"

    def node_h(self, world, roadmap, goal_idx: int = 1) -> np.ndarray:
        # NOTE: common.py uses INF = 1e30 (a large FINITE sentinel) for
        # unreachable Dijkstra distances, so np.isfinite is NOT a valid
        # connectivity test. Follow the codebase convention (dist < INF/10).
        dij = np.array(C.dijkstra_to_goal(roadmap.adj, goal_idx=goal_idx), dtype=np.float64)
        connected = dij < C.INF / 10.0
        fill = float(dij[connected].max() + world.side_len) if connected.any() else 10.0 * world.side_len
        dij[~connected] = fill
        return _finite_fill(dij, fallback=fill)
