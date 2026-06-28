import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "continuous_prm"))

import continuous_prm_spacetime as ST  # noqa: E402
import continuous_prm_dynamics as D  # noqa: E402


def _toy():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]])  # 0=start,1=goal,2=mid
    adj = [[(2, 1.0)], [(2, 1.0)], [(0, 1.0), (1, 1.0)]]
    return points, adj


def test_spacetime_astar_no_dynamics_reaches_goal():
    points, adj = _toy()
    dyn = D.Dynamics([])
    h = np.zeros((3, 50))
    res = ST.space_time_astar_prm(adj, points, dyn, h, budget=1000, v_agent=1.0, dt=1.0,
                                  t_max=40, start=0, goal=1)
    assert res["found"]
    assert res["arrival"] == 2


def test_spacetime_astar_must_wait_for_patroller():
    points, adj = _toy()
    mc = D.MovingCircle(ax=1.0, ay=0.0, bx=1.0, by=5.0, period=6.0, radius=0.3)
    dyn = D.Dynamics([mc])
    h = np.zeros((3, 50))
    res = ST.space_time_astar_prm(adj, points, dyn, h, budget=5000, v_agent=1.0, dt=1.0,
                                  t_max=40, start=0, goal=1)
    assert res["found"]
    assert res["arrival"] >= 2


def test_backward_dijkstra_matches_astar_optimal_arrival():
    points, adj = _toy()
    mc = D.MovingCircle(ax=1.0, ay=0.0, bx=1.0, by=5.0, period=6.0, radius=0.3)
    dyn = D.Dynamics([mc])
    htab = ST.backward_spacetime_dijkstra(adj, points, dyn, v_agent=1.0, dt=1.0, t_max=40, goal=1)
    opt = ST.space_time_astar_prm(adj, points, dyn, htab, budget=5000, v_agent=1.0, dt=1.0,
                                  t_max=40, start=0, goal=1)
    assert np.isclose(htab[0, 0], opt["arrival"])
    assert opt["found"] and opt["expansions"] <= 12


def test_spacetime_budget_and_completeness():
    points, adj = _toy()
    dyn = D.Dynamics([])
    h = np.zeros((3, 50))
    tiny = ST.space_time_astar_prm(adj, points, dyn, h, budget=1, v_agent=1.0, dt=1.0,
                                   t_max=40, start=0, goal=1)
    assert tiny["found"] is False and tiny["expansions"] <= 1


def test_oracle_time_to_go_zero_at_goal():
    points, adj = _toy()
    dyn = D.Dynamics([])
    htab = ST.backward_spacetime_dijkstra(adj, points, dyn, v_agent=1.0, dt=1.0, t_max=40, goal=1)
    ttg = ST.oracle_time_to_go(htab, t_max=40)
    assert np.all(ttg[1, :] == 0.0)            # goal node: zero time-to-go at all t
    assert np.all(np.isfinite(ttg))            # reachable graph -> all finite
    assert ttg[0, 0] == htab[0, 0]             # start at t=0: time-to-go == arrival (t=0)
