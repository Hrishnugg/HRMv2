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


def test_scalar_provider_additive_and_bounded():
    import continuous_prm_c7_hard_maps as M
    M.install_c7_hard_maps()
    spec = C.build_anchor_specs()["C_hard_spiral"]
    world = rm = None
    for seed in range(60):
        world = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
        if world is None:
            continue
        rm = C.build_prm(world, C.RoadmapConfig(n_nodes=128, k_neighbors=7), seed=seed)
        if rm is not None and rm.connected_to_goal[0]:
            break
    assert world is not None and rm is not None
    prov = P.ScalarResidualProvider.untrained_for_test(world)
    h = prov.node_h(world, rm, goal_idx=1)
    euclid = P.EuclidProvider().node_h(world, rm, goal_idx=1)
    assert h.shape == euclid.shape
    assert np.all(np.isfinite(h))
    assert np.all(h >= euclid - 1e-6)  # additive, non-negative residual
    assert np.all((h - euclid) <= world.side_len * prov.max_norm_residual + 1e-4)  # bounded


def test_value_field_provider_matches_c6_field_integration():
    import torch
    import continuous_prm_c6_heatmap_value_field as C6
    import continuous_prm_c7_hard_maps as M
    M.install_c7_hard_maps()
    spec = C.build_anchor_specs()["C_hard_spiral"]
    world = rm = None
    for seed in range(60):
        world = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
        if world is None:
            continue
        rm = C.build_prm(world, C.RoadmapConfig(n_nodes=128, k_neighbors=7), seed=seed)
        if rm is not None and rm.connected_to_goal[0]:
            break
    assert world is not None and rm is not None
    device = torch.device("cpu")
    grid_size = 64
    model = C6.build_model("hrm").to(device).eval()  # untrained field model
    prov = P.ValueFieldProvider(model, grid_size=grid_size, device=device, backbone="hrm")
    h = prov.node_h(world, rm, goal_idx=1)
    # parity with the shared helper used by evaluate_shard
    ex = C6.make_heatmap_example(world, grid_size)
    euclid = np.linalg.norm(rm.points - world.goal[None, :], axis=1).astype(np.float64)
    h_ref, _ = C6.field_node_heuristic(model, ex["x"], world, rm, euclid, device)
    assert np.allclose(h, h_ref, atol=1e-6)
    assert np.all(np.isfinite(h)) and np.all(h >= euclid - 1e-6)  # additive residual >= 0
