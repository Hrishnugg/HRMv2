import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_c13_lhbl as H


def _adj(edges, n):
    graph = [[] for _ in range(n)]
    for a, b, weight in edges:
        graph[a].append((b, float(weight)))
        graph[b].append((a, float(weight)))
    return graph


def _corridor_graph():
    points = np.array(
        [[0.0, 0.0], [4.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    )
    edges = [(0, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 1, 1.5)]
    return points, _adj(edges, len(points))


def test_first_lhbl_backup_matches_the_euclidean_exit_stub_target():
    points, graph = _corridor_graph()
    euclid = np.linalg.norm(points - points[1][None, :], axis=1)
    values, diagnostics = H.limited_horizon_values(
        points, graph, points[1], euclid, radius=1.5
    )
    assert values[0] == pytest.approx(3.0 + np.sqrt(5.0))
    assert diagnostics[0]["observed_nodes"] == 3
    assert diagnostics[0]["exit_actions"] == 1
    assert not diagnostics[0]["fallback"]
    assert np.all(values + 1.0e-9 >= euclid)
    assert values[1] == pytest.approx(0.0)


def test_lhbl_target_uses_only_bootstraps_on_visible_exit_actions():
    points, graph = _corridor_graph()
    euclid = np.linalg.norm(points - points[1][None, :], axis=1)
    first, _ = H.limited_horizon_values(
        points, graph, points[1], euclid, radius=1.5
    )
    beyond = euclid.copy()
    beyond[5] += 100.0
    second, _ = H.limited_horizon_values(
        points, graph, points[1], beyond, radius=1.5
    )
    assert second[0] == pytest.approx(first[0])
    exposed = euclid.copy()
    exposed[4] += 2.0
    third, _ = H.limited_horizon_values(
        points, graph, points[1], exposed, radius=1.5
    )
    assert third[0] == pytest.approx(first[0] + 2.0)


def test_generated_cohort_replays_the_source_world_and_roadmap_recipe():
    study = H.I.StudyConfig(
        suite="C_hard_maze",
        train_worlds=1,
        val_worlds=1,
        eval_worlds=1,
        roadmap_nodes=192,
        roadmap_k=7,
        seed=1234,
        max_world_retries=200,
    )
    local = H.C13.LocalStateConfig(
        sensor_radius_frac=0.20, num_rays=2, ray_steps=2, max_neighbors=4
    )
    bundle = H.rebuild_generated_bundles(study, local, "train", 1, 0)[0]
    assert bundle.world_seed == 2_332_917
    assert len(bundle.roadmap.points) == 192
    assert sum(len(row) for row in bundle.roadmap.adj) // 2 == 687
    assert bundle.features.shape == (192, 7, 16)


def test_attention_pool_is_invariant_to_non_summary_token_row_order():
    torch.manual_seed(7)
    model = H.AttentionPoolRanker(token_dim=16, hidden_dim=32, max_output=4.0)
    model.eval()
    features = torch.zeros((1, 8, 16), dtype=torch.float32)
    features[:, 0] = torch.randn((1, 16))
    features[:, 1:5] = torch.randn((1, 4, 16))
    order = torch.tensor([0, 3, 1, 4, 2, 5, 6, 7])
    with torch.no_grad():
        first = model(features)
        second = model(features.index_select(1, order))
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)


def _row(world, focal_delta, control_delta):
    return {
        "model": "masked_pool",
        "iteration": 4,
        "alpha": 0.5,
        "focal_w": 1.1,
        "world_index": world,
        "certified": True,
        "bound_violation_eval_only": False,
        "path_valid": True,
        "max_expansions_per_state": 1,
        "expansions": 10 + focal_delta,
        "euclid_focal_expansions": 10,
        "delta_vs_euclid_focal": focal_delta,
        "same_search_euclid_expansions": 12,
        "delta_vs_same_search_euclid": control_delta,
        "direct_learned_astar_expansions": 7,
        "direct_learned_astar_cost_ratio_eval_only": 1.05,
        "final_cost_ratio_eval_only": 1.0,
        "rank_eligible_choice_rate": 0.5,
    }


def test_lhbl_gate_requires_stable_wins_against_both_controls():
    rows = [
        _row(index, focal, control)
        for index, (focal, control) in enumerate(
            zip((-2, -2, -2, -2, -2, 1), (-3, -3, -3, -3, -3, 1))
        )
    ]
    summary = H.summarize_search(rows, 0.80)[0]
    assert summary["gate_pass"]
    assert summary["focal_wins"] == 5
    assert summary["same_search_euclid_wins"] == 5
    verdict = H.build_verdict([summary], 1.10)
    assert verdict["fresh_replication_required"]
    assert verdict["selected_candidate"]["model"] == "masked_pool"


def test_lhbl_gate_rejects_an_uncertified_apparent_win():
    rows = [_row(index, -2, -3) for index in range(6)]
    rows[0]["certified"] = False
    summary = H.summarize_search(rows, 0.80)[0]
    assert summary["safety_failures"] == 1
    assert not summary["gate_pass"]
