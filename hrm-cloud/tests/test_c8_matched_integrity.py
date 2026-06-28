import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "continuous_prm"))

import continuous_prm_common as C
import continuous_prm_dynamics as D
import continuous_prm_dynamic_providers as P


def _toy_roadmap():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
    adj = [[(2, 1.0)], [(2, 1.0)], [(0, 1.0), (1, 1.0)]]
    dist = C.dijkstra_to_goal(adj, goal_idx=1)
    return C.Roadmap(points=points, adj=adj, dist_to_goal=dist,
                     connected_to_goal=(dist < C.INF / 10.0))


class _W:
    side_len = 1.0


def test_run_arms_records_and_suboptimality():
    rm = _toy_roadmap()
    mc = D.MovingCircle(ax=1.6, ay=0.0, bx=1.6, by=4.0, period=2.0, radius=0.4)  # forces a wait
    dyn = D.Dynamics([mc])
    providers = {"euclid": P.EuclidTimeProvider(), "oracle": P.OracleProvider()}
    recs = P.run_world_arms_spacetime(_W(), rm, dyn, providers, budgets=[2000], w_values=[1.0, 1.5],
                                      v_agent=1.0, dt=1.0, t_max=40, goal_idx=1, start_idx=0)
    names = {(r["provider"], r["mode"], r.get("w")) for r in recs}
    assert ("euclid", "astar", None) in names
    assert ("oracle", "astar", None) in names
    assert ("euclid", "focal", 1.0) in names
    assert ("oracle", "focal", 1.5) in names
    for r in recs:
        assert {"provider", "mode", "budget", "found", "expansions", "arrival",
                "optimal_arrival", "suboptimality", "nonfinite"}.issubset(r)
        if r["found"]:
            assert r["suboptimality"] >= 1.0 - 1e-6
            if r["mode"] == "focal":
                assert r["suboptimality"] <= r["w"] + 1e-6


def test_run_arms_handles_nonfinite_provider():
    rm = _toy_roadmap(); dyn = D.Dynamics([])
    class _Bad(P.SpaceTimeHeuristicProvider):
        name = "bad"
        def h_table(self, world, roadmap, dyn, v_agent, dt, t_max, goal_idx=1):
            raise FloatingPointError("boom")
    providers = {"euclid": P.EuclidTimeProvider(), "bad": _Bad()}
    recs = P.run_world_arms_spacetime(_W(), rm, dyn, providers, [2000], [1.0],
                                      1.0, 1.0, 40, 1, 0)
    bad = [r for r in recs if r["provider"] == "bad"]
    assert bad and all((r["found"] is False) and r["nonfinite"] == 1 for r in bad)
    eu = [r for r in recs if r["provider"] == "euclid"]
    assert eu and all(r["nonfinite"] == 0 for r in eu)


def test_matched_worlds_identical_across_seeds():
    # determinism: same dynamics + same roadmap -> identical arm records
    rm = _toy_roadmap(); dyn = D.Dynamics([])
    providers = {"euclid": P.EuclidTimeProvider(), "oracle": P.OracleProvider()}
    a = P.run_world_arms_spacetime(_W(), rm, dyn, providers, [2000], [1.0], 1.0, 1.0, 40, 1, 0)
    b = P.run_world_arms_spacetime(_W(), rm, dyn, providers, [2000], [1.0], 1.0, 1.0, 40, 1, 0)
    assert a == b
