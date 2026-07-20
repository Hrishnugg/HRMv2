from __future__ import annotations

from pathlib import Path

import numpy as np

import continuous_prm_common as C
import continuous_prm_c13_lhbl_c7_comparison as X
import continuous_prm_c13_shared_queue as Q
import continuous_prm_c7_hard_maps as M7
import continuous_prm_c7_integration_compare as C7


def test_development_only_selection_is_frozen() -> None:
    cfg = X.ComparisonConfig()
    X.resolve_paths(cfg)
    report = X.build_selection_report(cfg)

    low = report["low_distortion"]
    primary = report["primary_throughput"]
    assert (low["iteration"], low["alpha"]) == (8, 0.75)
    assert (primary["iteration"], primary["alpha"]) == (8, 1.0)
    assert low["wins"] == primary["wins"] == 48
    assert low["max_cost_ratio"] <= 1.05
    assert primary["max_cost_ratio"] <= 1.10
    assert report["frozen_before_c7_current_model_evaluation"] is True


def test_astar_parent_bookkeeping_preserves_c7_search() -> None:
    adj = [
        [(2, 1.0), (3, 1.2)],
        [(2, 1.0), (3, 0.9)],
        [(0, 1.0), (1, 1.0), (3, 0.25)],
        [(0, 1.2), (1, 0.9), (2, 0.25)],
    ]
    for heuristic in (
        np.asarray([2.0, 0.0, 1.0, 0.9]),
        np.asarray([0.0, 0.0, 9.0, 0.0]),
    ):
        for budget in range(1, 6):
            expected = C.astar_search(adj, heuristic, budget)
            observed = X.astar_with_path(adj, heuristic, budget)
            assert observed["found"] == expected["found"]
            assert observed["expansions"] == expected["expansions"]
            assert observed["closed"] == expected["closed"]
            if expected["found"]:
                assert observed["cost"] == expected["cost"]
                assert Q.validate_path(
                    adj, observed["path"], observed["cost"]
                )["valid"]


def test_seed_aware_iterator_matches_original_c7_recipe() -> None:
    cfg = X.ComparisonConfig(worlds=1)
    M7.install_c7_hard_maps(cfg.sector_tokens)
    specs = C.build_anchor_specs()
    roadmap_cfg = C.RoadmapConfig(
        n_nodes=cfg.roadmap_nodes, k_neighbors=cfg.roadmap_k
    )
    original_cfg = C7.apply_scale_preset(
        C7.C7Config(
            eval_worlds=1,
            roadmap_nodes=cfg.roadmap_nodes,
            roadmap_k=cfg.roadmap_k,
            seed=cfg.seed,
            scale="local",
        )
    )
    original = next(
        C7.iter_matched_worlds(
            specs[X.ALL_SUITES[0]], 0, original_cfg, roadmap_cfg, 1
        )
    )
    seeded = next(
        X.iter_c7_worlds_with_seed(
            specs[X.ALL_SUITES[0]], 0, cfg, roadmap_cfg
        )
    )
    world_index, world_seed, roadmap_seed, world, roadmap = seeded
    assert world_index == original[0] == 0
    assert world_seed == world.meta["seed"]
    assert roadmap_seed == world_seed + 17
    assert np.array_equal(world.start, original[1].start)
    assert np.array_equal(world.goal, original[1].goal)
    assert np.array_equal(roadmap.points, original[2].points)
    assert roadmap.adj == original[2].adj


def test_map_and_current_boundaries_are_not_conflated() -> None:
    assert X._boundary_for_provider("field_hrm") == (
        "complete_64x64_occupancy_goal_raster"
    )
    assert "global_obstacle_list" in X._boundary_for_provider("scalar_hrm")
    assert X.CURRENT_PRIMARY not in X.MAP_FIELD_ARMS
    assert Path(X.__file__).name == "continuous_prm_c13_lhbl_c7_comparison.py"
