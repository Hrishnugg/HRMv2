import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "continuous_prm"))

import continuous_prm_common as C  # noqa: E402
import continuous_prm_c6_heatmap_value_field as C6  # noqa: E402


def _world(obstacles=None, start=(0.1, 0.1), goal=(0.9, 0.9)):
    return C.World(
        spec_name="test",
        side_len=1.0,
        obstacles=list(obstacles or []),
        start=np.array(start, dtype=np.float64),
        goal=np.array(goal, dtype=np.float64),
        descriptor=np.zeros(8, dtype=np.float32),
        meta={},
    )


def test_empty_grid_dijkstra_approximates_euclidean():
    world = _world()
    _, free, _ = C6.rasterize_world(world, 64)
    dist = C6.grid_dijkstra_to_goal(world, free)
    start_cell = C6.point_to_cell(world.start, world.side_len, 64)
    expected = float(np.linalg.norm(world.goal - world.start))
    assert math.isfinite(float(dist[start_cell]))
    assert abs(float(dist[start_cell]) - expected) / expected < 0.15


def test_wall_with_gap_routes_through_gap():
    obstacles = [
        C.Obstacle("rect", 0.5, 0.22, hw=0.025, hh=0.22),
        C.Obstacle("rect", 0.5, 0.78, hw=0.025, hh=0.22),
    ]
    world = _world(obstacles, start=(0.1, 0.5), goal=(0.9, 0.5))
    _, free, _ = C6.rasterize_world(world, 64)
    dist = C6.grid_dijkstra_to_goal(world, free)
    path = C6.trace_path_mask(world, free, dist)
    start_cell = C6.point_to_cell(world.start, world.side_len, 64)
    gap_cell = C6.point_to_cell(np.array([0.5, 0.5]), world.side_len, 64)
    assert math.isfinite(float(dist[start_cell]))
    assert path[gap_cell] > 0.0


def test_bilinear_interpolation_uses_cell_centers():
    world = _world(start=(0.1, 0.1), goal=(0.9, 0.9))
    grid = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    val = C6.interpolate_grid_values(grid, world, np.array([[0.5, 0.5]], dtype=np.float64))
    assert np.allclose(val, [1.5])


def test_oracle_grid_heuristic_changes_astar_on_hard_fixture():
    C6.install_c5_hard_runtime()
    spec = C.build_anchor_specs()["C_hard_maze"]
    cfg = C.RoadmapConfig(n_nodes=128, k_neighbors=6)
    changed = False
    for seed in range(1234, 1290):
        world = C.build_world(spec, seed, cfg.min_start_goal_dist_frac)
        if world is None:
            continue
        roadmap = C.build_prm(world, cfg, seed=seed + 17)
        if roadmap is None or not roadmap.connected_to_goal[0]:
            continue
        ex = C6.make_heatmap_example(world, 48)
        euclid = np.linalg.norm(roadmap.points - world.goal[None, :], axis=1)
        oracle = C6.make_oracle_heuristic(world, roadmap, ex["grid_distance_world"], euclid)
        e = C.astar_search(roadmap.adj, euclid, 96)
        o = C.astar_search(roadmap.adj, oracle, 96)
        changed = (e["found"] != o["found"]) or (e["expansions"] != o["expansions"])
        if changed:
            break
    assert changed


def test_field_model_forward_shapes():
    x = torch.randn(2, C6.INPUT_CHANNELS, 48, 48)
    models = [
        C6.UNetField(in_channels=C6.INPUT_CHANNELS, base=8),
        C6.ONLSTMField(in_channels=C6.INPUT_CHANNELS, hidden_dim=32, layers=1, chunk_size=8),
        C6.HRMField(in_channels=C6.INPUT_CHANNELS, hidden_dim=32, layers=1, heads=4, steps=1),
    ]
    for model in models:
        y = model(x)
        p = model.forward_path(x)
        assert tuple(y.shape) == (2, 1, 48, 48)
        assert tuple(p.shape) == (2, 1, 48, 48)


def test_dataset_shard_schema_roundtrip(tmp_path):
    x = np.zeros((2, C6.INPUT_CHANNELS, 16, 16), dtype=np.float32)
    target = np.zeros((2, 16, 16), dtype=np.float32)
    mask = np.ones((2, 16, 16), dtype=np.bool_)
    path = np.zeros((2, 16, 16), dtype=np.float32)
    npz = tmp_path / "toy_train.npz"
    np.savez_compressed(npz, x=x, target_residual=target, target_distance=target, free_mask=mask, path_mask=path)
    ds = C6.HeatmapDataset([npz])
    row = ds[0]
    assert len(ds) == 2
    assert tuple(row["x"].shape) == (C6.INPUT_CHANNELS, 16, 16)
    assert tuple(row["target_residual"].shape) == (16, 16)
    assert tuple(row["free_mask"].shape) == (16, 16)


def test_significance_analyzer_claim_candidate(tmp_path):
    rows = []
    for i in range(12):
        base_found = 1 if i < 6 else 0
        rows.append(
            {
                "suite": "S",
                "budget": 144,
                "world_index": i,
                "method": "euclidean",
                "found": base_found,
                "expansions": 144,
                "cost_ratio": 1.0,
                "oracle_cost": 1.0,
            }
        )
        rows.append(
            {
                "suite": "S",
                "budget": 144,
                "world_index": i,
                "method": "unet",
                "found": 1,
                "expansions": 80,
                "cost_ratio": 1.0,
                "oracle_cost": 1.0,
            }
        )
    sig, md = C6.analyze_significance(rows, tmp_path)
    out = C6.read_csv(sig)
    claim = [r for r in out if r["method"] == "unet"][0]
    assert int(claim["paired_gains"]) == 6
    assert float(claim["delta"]) == 0.5
    assert int(claim["claim_candidate"]) == 1
    assert md.exists()
