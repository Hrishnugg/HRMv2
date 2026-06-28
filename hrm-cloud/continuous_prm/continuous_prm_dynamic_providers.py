"""Time-aware heuristic providers returning h[node, t] tables in time-step units (C8)."""
from __future__ import annotations
import abc
import numpy as np
import continuous_prm_spacetime as ST


def euclid_time_row(roadmap, v_agent, goal_idx=1) -> np.ndarray:
    """Per-node straight-line time-to-go (seconds) = euclid(node,goal)/v_agent."""
    goal = roadmap.points[goal_idx]
    return np.linalg.norm(roadmap.points - goal[None, :], axis=1) / float(v_agent)


def _finite_fill(a: np.ndarray, fallback: float) -> np.ndarray:
    out = np.array(a, dtype=np.float64)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = fallback
    return np.maximum(out, 0.0)


class SpaceTimeHeuristicProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def h_table(self, world, roadmap, dyn, v_agent, dt, t_max, goal_idx: int = 1) -> np.ndarray:
        """Return h[node, t] (shape (N, t_max+1)) = estimated remaining time-to-go in TIME-STEPS."""
        ...


class EuclidTimeProvider(SpaceTimeHeuristicProvider):
    name = "euclid"

    def h_table(self, world, roadmap, dyn, v_agent, dt, t_max, goal_idx=1):
        row = euclid_time_row(roadmap, v_agent, goal_idx) / float(dt)  # seconds -> time-steps
        table = np.repeat(row[:, None], t_max + 1, axis=1)
        return _finite_fill(table, 0.0)


class OracleProvider(SpaceTimeHeuristicProvider):
    name = "oracle"

    def h_table(self, world, roadmap, dyn, v_agent, dt, t_max, goal_idx=1):
        hstar = ST.backward_spacetime_dijkstra(roadmap.adj, roadmap.points, dyn, v_agent, dt, t_max, goal_idx)
        ttg = ST.oracle_time_to_go(hstar, t_max)
        finite = np.isfinite(ttg)
        fill = float(ttg[finite].max() + (t_max + 1)) if finite.any() else float(2 * (t_max + 1))
        return _finite_fill(ttg, fill)
