import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_common as C
import continuous_prm_c13_state_heuristic as C13


def _undirected_adj(points, edges):
    adj = [[] for _ in range(len(points))]
    for a, b in edges:
        w = float(np.linalg.norm(points[a] - points[b]))
        adj[a].append((b, w))
        adj[b].append((a, w))
    return adj


def _manual_graph():
    points = np.array(
        [
            [0.0, 0.0],  # start
            [1.0, 0.0],  # goal
            [0.4, 0.4],
            [0.8, 0.2],
        ],
        dtype=np.float64,
    )
    adj = _undirected_adj(points, [(0, 2), (2, 3), (3, 1), (0, 3)])
    return points, adj


def test_constant_minus_e_semantics_are_explicit():
    audit = C13.semantics_audit()
    assert audit["max_rank_vs_euclid_error"] < 1.0e-12
    assert audit["constant_h_range"] < 1.0e-12

    e = np.array([0.0, 0.2, 0.7, math.sqrt(2.0)])
    value = C13.goal_proximity_value(e, 1.0)
    assert np.allclose(C13.proximity_value_to_cost_rank(value, 1.0), e)
    assert np.allclose(C13.literal_constant_residual_h(e, 1.0), math.sqrt(2.0))


def test_one_step_backup_dominates_euclid_and_is_admissible_consistent():
    points, adj = _manual_graph()
    e = C13.euclidean_to_goal(points, points[1])
    h1 = C13.one_step_euclidean_backup(points, adj)
    oracle = C.dijkstra_to_goal(adj, goal_idx=1)
    props = C13.one_step_property_audit(points, adj, oracle=oracle)

    assert np.all(h1 + 1.0e-12 >= e)
    assert np.all(h1 <= oracle + 1.0e-12)
    assert h1[1] == 0.0
    assert props["dominance_violation"] < 1.0e-12
    assert props["admissibility_violation"] < 1.0e-12
    assert props["consistency_violation"] < 1.0e-12
    assert props["goal_abs"] == 0.0


def test_bounded_backup_curve_stays_valid_and_approaches_oracle():
    points = np.array(
        [
            [0.0, 0.0], [1.0, 0.0],
            [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
        ],
        dtype=np.float64,
    )
    # A four-edge detour: deeper backups must propagate more of its cost.
    adj = _undirected_adj(points, [(0, 2), (2, 3), (3, 4), (4, 1)])

    oracle = C.dijkstra_to_goal(adj, goal_idx=1)
    previous = C13.euclidean_to_goal(points, points[1])
    gaps = []
    for depth in (0, 1, 2, 4):
        h = C13.bounded_euclidean_backup(points, adj, depth)
        assert np.all(h + 1.0e-12 >= previous)
        assert np.all(h <= oracle + 1.0e-12)
        for u, nbrs in enumerate(adj):
            for v, w in nbrs:
                assert h[u] <= w + h[v] + 1.0e-12
        gaps.append(float(np.mean(oracle - h)))
        previous = h
    assert np.allclose(
        C13.bounded_euclidean_backup(points, adj, 1),
        C13.one_step_euclidean_backup(points, adj),
    )
    assert gaps[-1] < gaps[1]


def test_target_construction_cannot_read_oracle_distance():
    points, adj = _manual_graph()

    class NoOracleRoadmap:
        def __init__(self):
            self.points = points
            self.adj = adj

        @property
        def dist_to_goal(self):
            raise AssertionError("C13 target attempted to read the shortest-path oracle")

    roadmap = NoOracleRoadmap()
    target = C13.one_step_residual_target(roadmap.points, roadmap.adj, side_len=1.0)
    assert target.shape == (len(points),)
    assert np.all(np.isfinite(target))
    assert np.all(target >= 0.0)


def test_local_features_ignore_start_descriptor_and_far_obstacles():
    points = np.array([[0.5, 0.5], [0.8, 0.5], [0.58, 0.54]], dtype=np.float64)
    adj = _undirected_adj(points, [(0, 2), (2, 1)])
    cfg = C13.LocalStateConfig(
        sensor_radius_frac=0.15,
        num_rays=8,
        ray_steps=12,
        max_neighbors=8,
    )
    base = C.World(
        spec_name="locality",
        side_len=1.0,
        obstacles=[],
        start=np.array([0.1, 0.1]),
        goal=points[1].copy(),
        descriptor=np.zeros(8, dtype=np.float32),
        meta={"mode": "base"},
    )
    changed = C.World(
        spec_name="different_name",
        side_len=1.0,
        obstacles=[C.Obstacle(kind="circle", cx=0.9, cy=0.9, radius=0.02)],
        start=np.array([0.95, 0.05]),
        goal=points[1].copy(),
        descriptor=np.ones(8, dtype=np.float32),
        meta={"mode": "different"},
    )

    x0 = C13.local_state_sequence(base, points, adj, 0, cfg)
    x1 = C13.local_state_sequence(changed, points, adj, 0, cfg)
    assert x0.shape == (cfg.seq_len, cfg.token_dim)
    assert np.allclose(x0, x1, atol=0.0)


def test_density_grid_contains_exact_plus_ten_percent_setting():
    cfg = C13.C13Config()
    nodes = C13.parse_int_csv(cfg.density_nodes)
    assert cfg.train_nodes == 192
    assert 211 in nodes
    assert round(cfg.train_nodes * 1.10) == 211
