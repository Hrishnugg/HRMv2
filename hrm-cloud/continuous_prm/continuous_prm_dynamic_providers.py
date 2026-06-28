"""Time-aware heuristic providers returning h[node, t] tables in time-step units (C8)."""
from __future__ import annotations
import abc
import numpy as np
import continuous_prm_spacetime as ST
from continuous_prm_spacetime import (
    space_time_astar_prm,
    space_time_focal_prm,
    backward_spacetime_dijkstra,
)


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


def run_world_arms_spacetime(world, roadmap, dyn, providers: dict, budgets, w_values,
                             v_agent, dt, t_max, goal_idx=1, start_idx=0):
    """Run every (provider, mode, budget[, w]) arm on one shared world+roadmap+dynamics.

    astar mode: space_time_astar_prm with the provider's h_table.
    focal mode: space_time_focal_prm with admissible euclid_time order + provider table as ranker.
    Per-provider nonfinite guard: a provider whose h_table raises FloatingPointError is
    recorded as nonfinite arms (found=False) for all its budget/w arms, not crashing the world.
    suboptimality = arrival / optimal_arrival (optimal from backward_spacetime_dijkstra[start,0]).
    Returns a flat list of record dicts.
    """
    # optimal arrival (makespan) once:
    hstar = backward_spacetime_dijkstra(
        roadmap.adj, roadmap.points, dyn, v_agent, dt, t_max, goal_idx
    )
    opt = float(hstar[start_idx, 0])  # may be inf if unsolvable

    # admissible euclid_time in time-step units, for focal ordering:
    euclid_t = euclid_time_row(roadmap, v_agent, goal_idx) / float(dt)

    # precompute each provider's table (nonfinite-robust):
    tables = {}
    nonfinite: set = set()
    for name, prov in providers.items():
        try:
            tables[name] = prov.h_table(world, roadmap, dyn, v_agent, dt, t_max, goal_idx)
        except FloatingPointError:
            nonfinite.add(name)

    records = []
    for name in providers:
        if name in nonfinite:
            for b in budgets:
                records.append(_arm_record_st(name, "astar", None, b,
                    {"found": False, "arrival": -1, "expansions": 0, "closed": 0}, opt, nonfinite=1))
                for w in w_values:
                    records.append(_arm_record_st(name, "focal", float(w), b,
                        {"found": False, "arrival": -1, "expansions": 0, "closed": 0}, opt, nonfinite=1))
            continue
        h = tables[name]
        for b in budgets:
            r = space_time_astar_prm(
                roadmap.adj, roadmap.points, dyn, h, int(b),
                v_agent, dt, t_max, start_idx, goal_idx
            )
            records.append(_arm_record_st(name, "astar", None, b, r, opt))
            for w in w_values:
                rf = space_time_focal_prm(
                    roadmap.adj, roadmap.points, dyn, euclid_t, h, int(b),
                    v_agent, dt, t_max, float(w), start_idx, goal_idx
                )
                records.append(_arm_record_st(name, "focal", float(w), b, rf, opt))
    return records


def _arm_record_st(provider, mode, w, budget, res, opt, nonfinite=0):
    found = bool(res["found"])
    arrival = int(res["arrival"]) if found else -1
    sub = (arrival / opt) if (found and np.isfinite(opt) and opt > 0) else float("nan")
    return {
        "provider": provider, "mode": mode, "w": w, "budget": int(budget),
        "found": found, "expansions": int(res["expansions"]), "arrival": arrival,
        "optimal_arrival": float(opt) if np.isfinite(opt) else float("nan"),
        "suboptimality": sub, "closed": int(res["closed"]), "nonfinite": int(nonfinite),
    }
