import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_common as C
import continuous_prm_c13_local_escape as G


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


def test_local_escape_raises_euclidean_using_only_the_observed_frontier():
    points, graph = _corridor_graph()
    observation = G.extract_local_observation(
        points, graph, current_idx=0, goal_point=points[1], radius=1.5
    )
    result = G.exact_local_escape(observation)
    assert result["observed_nodes"] == 3
    assert result["frontier_nodes"] == 1
    assert not result["fallback"]
    assert result["value"] > 4.0
    assert result["value"] == pytest.approx(2.0 + np.sqrt(10.0))


def test_local_escape_is_invariant_to_unobserved_graph_changes():
    points, graph = _corridor_graph()
    first = G.exact_local_escape(
        G.extract_local_observation(points, graph, 0, points[1], radius=1.5)
    )
    changed_points = points.copy()
    changed_points[4] = [8.0, 8.0]
    changed_points[5] = [9.0, 8.0]
    changed_graph = _adj(
        [(0, 2, 1.0), (2, 3, 1.0), (3, 4, 9.0), (4, 5, 7.0), (5, 1, 6.0)],
        len(points),
    )
    second = G.exact_local_escape(
        G.extract_local_observation(
            changed_points, changed_graph, 0, changed_points[1], radius=1.5
        )
    )
    assert second["value"] == pytest.approx(first["value"])
    assert second["observed_nodes"] == first["observed_nodes"]
    assert second["frontier_nodes"] == first["frontier_nodes"]


def test_local_escape_field_is_euclidean_dominating_and_admissible():
    points, graph = _corridor_graph()
    field, diagnostics = G.compute_local_escape_field(
        points, graph, points[1], radius=1.5
    )
    euclid = np.linalg.norm(points - points[1][None, :], axis=1)
    oracle = C.dijkstra_to_goal(graph, goal_idx=1)
    assert np.all(field + 1.0e-9 >= euclid)
    assert np.all(field <= oracle + 1.0e-9)
    assert field[1] == pytest.approx(0.0)
    assert not any(row["fallback"] for row in diagnostics)


def _search_row(radius, focal_delta, control_delta):
    return {
        "radius_frac": float(radius),
        "focal_w": 1.10,
        "certified": True,
        "bound_violation_eval_only": False,
        "path_valid": True,
        "anchor_lower_bound_exceeds_optimal_eval_only": False,
        "expansion_accounting_valid": True,
        "expansions": 10 + int(focal_delta),
        "rank_expansions": 3,
        "anchor_expansions": 7 + int(focal_delta),
        "rank_eligibility_checks": 10,
        "rank_eligible_choices": 3,
        "euclid_focal_expansions": 10,
        "delta_vs_euclid_focal": float(focal_delta),
        "same_search_euclid_expansions": 12,
        "delta_vs_same_search_euclid": float(control_delta),
        "shared_oracle_expansions": 6,
        "direct_local_astar_expansions": 9,
        "direct_local_astar_cost_ratio_eval_only": 1.0,
        "final_cost_ratio_eval_only": 1.0,
    }


def _target_summary(radius):
    return {
        "radius_frac": float(radius),
        "euclid_dominance_violations": 0,
        "fallback_nodes_reachable": 0,
        "fallback_nodes_unreachable": 0,
        "oracle_admissibility_violations_eval_only": 0,
        "goal_boundary_violation": False,
        "fallback_nodes": 0,
        "local_nodes_mean": 20.0,
        "positive_residual_rate": 0.8,
        "start_h_over_oracle_eval_only": 0.6,
        "edge_consistency_violations": 2,
        "target_seconds": 0.01,
    }


def test_local_escape_gate_requires_stable_focal_and_control_wins():
    rows = [
        _search_row(0.20, focal, control)
        for focal, control in zip((-2, -2, -2, -2, -2, 1), (-3, -3, -3, -3, -3, 1))
    ]
    targets = [_target_summary(0.20) for _ in range(6)]
    summary = G.summarize_rows(rows, targets, 0.80)[0]
    assert summary["gate_pass"]
    assert summary["focal_wins"] == 5
    assert summary["same_search_euclid_wins"] == 5



def test_unreachable_fallbacks_are_recorded_but_do_not_fail_the_gate():
    rows = [_search_row(0.20, -2, -3) for _ in range(6)]
    targets = [_target_summary(0.20) for _ in range(6)]
    for target in targets:
        target["fallback_nodes"] = 4
        target["fallback_nodes_unreachable"] = 4
    summary = G.summarize_rows(rows, targets, 0.80)[0]
    assert summary["target_validity_failures"] == 0
    assert summary["gate_pass"]

def test_local_escape_verdict_requires_fresh_replication():
    summary = {
        "radius_frac": 0.20,
        "focal_w": 1.10,
        "gate_pass": True,
        "expansions_mean": 8.0,
    }
    verdict = G.build_verdict([summary], 1.10, 0.20)
    assert verdict["primary_gate_pass"]
    assert verdict["fresh_world_replication_required"]
    assert verdict["authorization"].startswith("replicate_primary_radius")
