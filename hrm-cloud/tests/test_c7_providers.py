import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "continuous_prm"))

import numpy as np
import continuous_prm_common as C
import continuous_prm_providers as P


def _tiny_world_and_prm():
    spec = C.build_anchor_specs()["C_open"]
    for seed in range(50):
        world = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
        if world is None:
            continue
        rm = C.build_prm(world, C.RoadmapConfig(n_nodes=64, k_neighbors=8), seed=seed)
        if rm is not None:
            return world, rm
    raise RuntimeError("could not build a tiny world")


def test_euclid_provider_admissible():
    world, rm = _tiny_world_and_prm()
    h = P.EuclidProvider().node_h(world, rm, goal_idx=1)
    assert h.shape == (rm.points.shape[0],)
    assert np.all(np.isfinite(h)) and np.all(h >= -1e-9)
    conn = rm.connected_to_goal
    if conn.any():
        # Euclid <= exact graph cost-to-go on connected nodes (admissible).
        assert np.all(h[conn] <= rm.dist_to_goal[conn] + 1e-6)


def test_oracle_provider_equals_dijkstra():
    world, rm = _tiny_world_and_prm()
    h = P.OracleProvider().node_h(world, rm, goal_idx=1)
    dij = C.dijkstra_to_goal(rm.adj, goal_idx=1)
    conn = dij < C.INF / 10.0
    assert np.allclose(h[conn], dij[conn], atol=1e-9)
    assert np.all(np.isfinite(h))  # disconnected nodes filled finite, not inf/nan


def test_oracle_makes_astar_optimal():
    world, rm = _tiny_world_and_prm()
    h = P.OracleProvider().node_h(world, rm, goal_idx=1)
    res = C.astar_search(rm.adj, h, budget=10_000)
    assert res["found"]
    assert np.isclose(res["cost"], rm.dist_to_goal[0], rtol=1e-6)


def test_oracle_fills_disconnected_finite():
    adj = [[(1, 1.0)], [(0, 1.0)], []]  # node 2 disconnected from goal (idx 1)
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    dist = C.dijkstra_to_goal(adj, goal_idx=1)
    rm = C.Roadmap(points=pts, adj=adj, dist_to_goal=dist,
                   connected_to_goal=(dist < C.INF / 10.0))

    class _W:
        side_len = 1.0

    h = P.OracleProvider().node_h(_W(), rm, goal_idx=1)
    assert np.all(np.isfinite(h))
    assert h[1] == 0.0
    assert h[2] > h[0]  # disconnected node filled larger than any reachable cost
