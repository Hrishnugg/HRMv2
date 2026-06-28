import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "continuous_prm"))

import continuous_prm_common as C
import continuous_prm_spacetime as ST
import continuous_prm_dynamics as D
import continuous_prm_dynamic_providers as P


def _toy_roadmap():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]])  # 0=start,1=goal,2=mid
    adj = [[(2, 1.0)], [(2, 1.0)], [(0, 1.0), (1, 1.0)]]
    dist = C.dijkstra_to_goal(adj, goal_idx=1)
    return C.Roadmap(points=points, adj=adj, dist_to_goal=dist,
                     connected_to_goal=(dist < C.INF / 10.0))


class _W:
    side_len = 1.0


def test_euclid_time_table_shape_admissible_and_constant():
    rm = _toy_roadmap(); dyn = D.Dynamics([])
    h = P.EuclidTimeProvider().h_table(_W(), rm, dyn, v_agent=1.0, dt=1.0, t_max=40, goal_idx=1)
    assert h.shape == (3, 41)
    assert np.all(np.isfinite(h)) and np.all(h >= 0)
    assert np.allclose(h, h[:, :1])  # t-independent (constant rows)
    orc = P.OracleProvider().h_table(_W(), rm, dyn, 1.0, 1.0, 40, 1)
    assert np.all(h <= orc + 1e-9)  # euclid-time is an admissible lower bound


def test_oracle_table_equals_backward_dijkstra_ttg():
    rm = _toy_roadmap(); dyn = D.Dynamics([])
    h = P.OracleProvider().h_table(_W(), rm, dyn, 1.0, 1.0, 40, 1)
    hstar = ST.backward_spacetime_dijkstra(rm.adj, rm.points, dyn, 1.0, 1.0, 40, 1)
    ttg = ST.oracle_time_to_go(hstar, 40)
    assert np.allclose(h, ttg)
    assert np.all(h[1, :] == 0.0)  # goal node: zero time-to-go at all t


def test_oracle_makes_search_minimal():
    rm = _toy_roadmap(); dyn = D.Dynamics([])
    h = P.OracleProvider().h_table(_W(), rm, dyn, 1.0, 1.0, 40, 1)
    res = ST.space_time_astar_prm(rm.adj, rm.points, dyn, h, 5000, 1.0, 1.0, 40, 0, 1)
    assert res["found"] and res["arrival"] == 2 and res["expansions"] <= 6


def test_provider_names():
    assert P.EuclidTimeProvider().name == "euclid"
    assert P.OracleProvider().name == "oracle"
