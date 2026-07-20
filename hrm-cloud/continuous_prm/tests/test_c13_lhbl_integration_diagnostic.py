import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_c13_identifiability as I
import continuous_prm_c13_lhbl_integration_diagnostic as D
import continuous_prm_c13_lhbl_focal_reopen_diagnostic as F
import continuous_prm_c13_shared_queue as Q


def _graph():
    points = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    graph = [[] for _ in points]
    for a, b, cost in ((0, 2, 1.0), (2, 3, 1.0), (3, 1, 2.0)):
        graph[a].append((b, cost))
        graph[b].append((a, cost))
    return points, graph


def test_path_enabled_focal_matches_the_existing_search_for_every_secondary():
    points, graph = _graph()
    euclid = np.linalg.norm(points - points[1][None, :], axis=1)
    rank = np.array([7.0, 0.0, 3.0, 1.0])
    for mode in ("h", "fhat", "residual"):
        expected = I.focal_search_with_secondary(
            graph, euclid, rank, len(points), 1.10, mode
        )
        actual = D.focal_search_with_path(
            graph, euclid, rank, len(points), 1.10, mode
        )
        assert actual["found"] == expected["found"]
        assert actual["cost"] == expected["cost"]
        assert actual["expansions"] == expected["expansions"]
        assert Q.validate_path(graph, actual["path"], actual["cost"])["valid"]
        assert actual["cost"] <= 1.10 * actual["anchor_f_min_at_return"] + 1e-9



def test_reopening_focal_repairs_a_state_when_a_better_g_arrives():
    graph = [[] for _ in range(4)]
    for a, b, cost in ((0, 2, 1.05), (0, 3, 1.0), (3, 2, 0.01), (2, 1, 1.0)):
        graph[a].append((b, cost))
        graph[b].append((a, cost))
    anchor = np.zeros(4, dtype=np.float64)
    rank = np.array([0.0, 100.0, 0.0, 10.0])
    unsafe = D.focal_search_with_path(graph, anchor, rank, 8, 1.10, "h")
    repaired = F.focal_search_with_path(graph, anchor, rank, 8, 1.10, "h")
    assert unsafe["cost"] == 2.05
    assert repaired["cost"] == 2.01
    assert repaired["max_expansions_per_state"] == 2
    assert Q.validate_path(graph, repaired["path"], repaired["cost"])["valid"]

def test_integration_selection_requires_a_pass_at_both_densities():
    rows = [
        {"mode": "h", "alpha": 0.75, "density": 192, "gate_pass": True, "delta_mean": -3.0},
        {"mode": "h", "alpha": 0.75, "density": 211, "gate_pass": True, "delta_mean": -2.0},
        {"mode": "fhat", "alpha": 1.0, "density": 192, "gate_pass": True, "delta_mean": -9.0},
        {"mode": "fhat", "alpha": 1.0, "density": 211, "gate_pass": False, "delta_mean": 1.0},
    ]
    verdict = D.select_candidate(rows)
    assert verdict["fresh_replication_required"]
    assert verdict["selected_candidate"]["mode"] == "h"
    assert verdict["selected_candidate"]["alpha"] == 0.75
