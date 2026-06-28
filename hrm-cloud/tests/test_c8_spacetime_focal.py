import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "continuous_prm"))

import continuous_prm_spacetime as ST  # noqa: E402
import continuous_prm_dynamics as D  # noqa: E402


def _toy():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
    adj = [[(2, 1.0)], [(2, 1.0)], [(0, 1.0), (1, 1.0)]]
    return points, adj


def _euclid_time(points, v_agent=1.0, dt=1.0, goal=1):
    return np.linalg.norm(points - points[goal][None, :], axis=1) / v_agent / dt


def test_focal_w1_matches_optimal_arrival():
    points, adj = _toy(); dyn = D.Dynamics([])
    et = _euclid_time(points)
    rank = np.zeros((3, 50))
    opt = ST.space_time_astar_prm(adj, points, dyn, ST.oracle_time_to_go(
        ST.backward_spacetime_dijkstra(adj, points, dyn, 1.0, 1.0, 40, 1), 40),
        5000, 1.0, 1.0, 40, 0, 1)
    res = ST.space_time_focal_prm(adj, points, dyn, et, rank, 5000, 1.0, 1.0, 40, w=1.0, start=0, goal=1)
    assert res["found"] and res["arrival"] == opt["arrival"]


def test_focal_bound_never_violated_with_dynamics():
    points, adj = _toy()
    mc = D.MovingCircle(ax=1.6, ay=0.0, bx=1.6, by=4.0, period=2.0, radius=0.4)  # forces a wait
    dyn = D.Dynamics([mc])
    et = _euclid_time(points)
    rng = np.random.default_rng(0); rank = rng.random((3, 50))
    htab = ST.oracle_time_to_go(ST.backward_spacetime_dijkstra(adj, points, dyn, 1.0, 1.0, 40, 1), 40)
    opt = ST.space_time_astar_prm(adj, points, dyn, htab, 5000, 1.0, 1.0, 40, 0, 1)
    w = 2.0
    res = ST.space_time_focal_prm(adj, points, dyn, et, rank, 5000, 1.0, 1.0, 40, w=w, start=0, goal=1)
    assert res["found"] and res["arrival"] <= w * opt["arrival"] + 1e-9


def test_focal_guards():
    points, adj = _toy(); dyn = D.Dynamics([])
    et = _euclid_time(points); rank = np.zeros((3, 50))
    import pytest
    with pytest.raises(ValueError):
        ST.space_time_focal_prm(adj, points, dyn, et, rank, 100, 1.0, 1.0, 40, w=0.9)
    bad = et.copy(); bad[0] = np.inf
    with pytest.raises(ValueError):
        ST.space_time_focal_prm(adj, points, dyn, bad, rank, 100, 1.0, 1.0, 40, w=1.0)


def test_focal_determinism():
    points, adj = _toy(); dyn = D.Dynamics([])
    et = _euclid_time(points); rank = np.linspace(0, 1, 150).reshape(3, 50)
    a = ST.space_time_focal_prm(adj, points, dyn, et, rank, 5000, 1.0, 1.0, 40, w=1.3, start=0, goal=1)
    b = ST.space_time_focal_prm(adj, points, dyn, et, rank, 5000, 1.0, 1.0, 40, w=1.3, start=0, goal=1)
    assert a == b
