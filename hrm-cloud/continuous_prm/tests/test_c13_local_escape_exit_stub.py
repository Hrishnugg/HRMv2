import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_common as C
import continuous_prm_c13_local_escape_exit_stub as G2


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


def test_exit_stub_pays_the_visible_outgoing_edge_and_endpoint_terminal():
    points, graph = _corridor_graph()
    observation = G2.extract_local_observation(
        points, graph, current_idx=0, goal_point=points[1], radius=1.5
    )
    result = G2.exact_local_escape(observation)
    assert result["observed_nodes"] == 3
    assert result["frontier_nodes"] == 1
    assert result["exit_actions"] == 1
    assert not result["fallback"]
    assert result["value"] == pytest.approx(3.0 + np.sqrt(5.0))


def test_exit_stub_is_invariant_to_topology_beyond_the_exposed_action():
    points, graph = _corridor_graph()
    first = G2.exact_local_escape(
        G2.extract_local_observation(points, graph, 0, points[1], radius=1.5)
    )
    changed_points = points.copy()
    changed_points[5] = [9.0, 8.0]
    changed_graph = _adj(
        [(0, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 7.0), (5, 1, 6.0)],
        len(points),
    )
    second = G2.exact_local_escape(
        G2.extract_local_observation(
            changed_points, changed_graph, 0, changed_points[1], radius=1.5
        )
    )
    assert second["value"] == pytest.approx(first["value"])
    assert second["observed_nodes"] == first["observed_nodes"]
    assert second["exit_actions"] == first["exit_actions"]


def test_exposed_exit_endpoint_geometry_is_part_of_the_current_observation():
    points, graph = _corridor_graph()
    first = G2.exact_local_escape(
        G2.extract_local_observation(points, graph, 0, points[1], radius=1.5)
    )
    changed_points = points.copy()
    changed_points[4] = [2.5, 1.0]
    second = G2.exact_local_escape(
        G2.extract_local_observation(
            changed_points, graph, 0, changed_points[1], radius=1.5
        )
    )
    assert second["value"] != pytest.approx(first["value"])


def test_exit_stub_field_is_euclidean_dominating_and_admissible():
    points, graph = _corridor_graph()
    field, diagnostics = G2.compute_local_escape_field(
        points, graph, points[1], radius=1.5
    )
    euclid = np.linalg.norm(points - points[1][None, :], axis=1)
    oracle = C.dijkstra_to_goal(graph, goal_idx=1)
    assert np.all(field + 1.0e-9 >= euclid)
    assert np.all(field <= oracle + 1.0e-9)
    assert field[1] == pytest.approx(0.0)
    assert not any(row["fallback"] for row in diagnostics)
