import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_common as C
import continuous_prm_c13_shared_queue as Q


def _adj(edges, n):
    graph = [[] for _ in range(n)]
    for a, b, weight in edges:
        graph[a].append((b, float(weight)))
        graph[b].append((a, float(weight)))
    return graph


def _summary_row(delta, *, certified=True, path_valid=True, violation=False):
    return {
        "focal_w": 1.10,
        "delta_vs_euclid_focal": float(delta),
        "certified": bool(certified),
        "bound_violation_eval_only": bool(violation),
        "path_valid": bool(path_valid),
        "anchor_lower_bound_exceeds_optimal_eval_only": False,
        "expansions": 5,
        "rank_expansions": 3,
        "anchor_expansions": 2,
        "duplicate_state_expansions": 1,
        "euclid_focal_expansions": 7,
        "euclid_astar_expansions": 8,
        "independent_certifier_expansions": 11,
        "saved_vs_independent_certifier": 6,
        "final_cost_ratio_eval_only": 1.02,
    }


def test_shared_queue_reuses_a_rank_expansion_in_the_anchor_search():
    graph = _adj(
        [(0, 2, 5.0), (0, 3, 1.0), (3, 2, 1.0), (2, 1, 10.0)],
        4,
    )
    anchor = np.array([12.0, 0.0, 10.0, 11.0])
    rank = np.array([0.0, 1000.0, 0.0, 100.0])

    result = Q.shared_anchor_certified_search(
        graph,
        anchor,
        rank,
        w=1.10,
        budget=8,
    )

    assert result["certified"]
    assert result["final_cost"] == pytest.approx(12.0)
    assert result["expansions"] == 4
    assert result["rank_expansions"] == 2
    assert result["anchor_expansions"] == 2
    assert result["duplicate_state_expansions"] == 1
    assert result["max_expansions_per_state"] == 2
    assert result["improvements_after_expansion"] == 1
    assert Q.validate_path(graph, result["path"], result["final_cost"])["valid"]


def test_shared_queue_refuses_to_claim_a_bound_without_search_budget():
    graph = _adj([(0, 1, 1.0)], 2)
    result = Q.shared_anchor_certified_search(
        graph,
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        w=1.10,
        budget=0,
    )
    assert not result["certified"]
    assert not result["found"]
    assert result["expansions"] == 0


def test_shared_queue_validates_anchor_and_rank_inputs():
    graph = _adj([(0, 2, 1.0), (2, 1, 1.0)], 3)
    with pytest.raises(ValueError, match="inconsistent"):
        Q.shared_anchor_certified_search(
            graph,
            np.array([3.0, 0.0, 1.0]),
            np.array([2.0, 0.0, 1.0]),
            w=1.10,
            budget=6,
        )
    with pytest.raises(ValueError, match="rank heuristic must be finite"):
        Q.shared_anchor_certified_search(
            graph,
            np.array([2.0, 0.0, 1.0]),
            np.array([2.0, 0.0, np.nan]),
            w=1.10,
            budget=6,
        )


def test_path_validation_rejects_missing_edges_and_wrong_costs():
    graph = _adj([(0, 2, 1.0), (2, 1, 1.0)], 3)
    assert Q.validate_path(graph, [0, 2, 1], 2.0)["valid"]
    assert not Q.validate_path(graph, [0, 1], 2.0)["valid"]
    assert not Q.validate_path(graph, [0, 2, 1], 3.0)["valid"]


def test_returned_witness_is_frozen_to_the_certified_g_label():
    # Regression for mutable-parent corruption after a goal incumbent exists.
    # The returned witness must remain tied to the incumbent's stored g-label.
    graph = _adj(
        [
            (0, 2, .7432943165), (2, 3, 1.2811169349),
            (3, 4, .2077056615), (4, 5, .8024119879),
            (5, 6, 2.2851056690), (6, 7, .8367988507),
            (7, 8, 2.2751746567), (8, 1, 1.6264782312),
            (0, 3, 2.0924019340), (0, 5, 1.8234153967),
            (0, 6, 2.5738506434), (0, 7, 1.5822577315),
            (0, 8, 2.1116869034), (1, 2, 2.4515238178),
            (1, 3, .6921783591), (2, 4, .5145307312),
            (2, 5, 1.9640611132), (2, 6, .4686658297),
            (2, 7, 2.5584876079), (3, 6, 1.4681265347),
            (3, 7, 2.3496673361), (4, 6, 2.9139197153),
            (4, 8, .3211936068),
        ],
        9,
    )
    rank = np.array([
        5.3616128426, 0.0, 21.5392637127, .4705503496, 4.8992529093,
        15.9696684507, 14.5799956402, 24.9559033612, 9.4551967864,
    ])
    result = Q.shared_anchor_certified_search(
        graph, np.zeros(9), rank, w=2.0, budget=18,
    )
    assert result["certified"]
    assert result["improvements_after_expansion"] == 1
    assert Q.validate_path(graph, result["path"], result["final_cost"])["valid"]

@pytest.mark.parametrize("w", [1.0, 1.10, 1.25])
def test_random_graph_certificates_respect_the_independent_optimum(w):
    rng = np.random.default_rng(90210 + int(100 * w))
    for _ in range(20):
        n = 9
        order = [0, *range(2, n), 1]
        edges = []
        used = set()
        for a, b in zip(order[:-1], order[1:]):
            weight = float(rng.uniform(0.2, 3.0))
            edges.append((a, b, weight))
            used.add((min(a, b), max(a, b)))
        for a in range(n):
            for b in range(a + 1, n):
                if (a, b) in used or rng.random() >= 0.30:
                    continue
                edges.append((a, b, float(rng.uniform(0.2, 3.0))))
        graph = _adj(edges, n)
        optimal_h = C.dijkstra_to_goal(graph, goal_idx=1)
        arbitrary_rank = rng.uniform(0.0, 20.0, size=n)
        arbitrary_rank[1] = 0.0

        result = Q.shared_anchor_certified_search(
            graph,
            optimal_h,
            arbitrary_rank,
            w=w,
            budget=2 * n,
        )
        path_check = Q.validate_path(graph, result["path"], result["final_cost"])

        assert result["certified"]
        assert path_check["valid"]
        assert result["lower_bound"] <= optimal_h[0] + 1.0e-9
        assert result["final_cost"] <= w * optimal_h[0] + 1.0e-9
        assert result["max_expansions_per_state"] <= 2
        if w == 1.0:
            assert result["final_cost"] == pytest.approx(optimal_h[0])


def test_summary_gate_keeps_the_five_of_six_contract():
    rows = [_summary_row(delta) for delta in (-2, -2, -2, -2, -2, 1)]
    summary = Q.summarize_rows(rows, required_win_fraction=0.80)[0]
    assert summary["required_wins"] == 5
    assert summary["wins"] == 5
    assert summary["gate_pass"]

    rows[0] = _summary_row(-2, certified=False)
    assert not Q.summarize_rows(rows, required_win_fraction=0.80)[0]["gate_pass"]


def test_gate_verdict_authorizes_only_after_oracle_passes():
    passed = Q.build_gate_verdict(
        [{"focal_w": 1.10, "gate_pass": True}],
        primary_w=1.10,
    )
    assert passed["verdict"] == "shared_queue_oracle_gate_pass"
    assert passed["authorization"] == "run_exact_rollout_target_next"

    failed = Q.build_gate_verdict(
        [{"focal_w": 1.10, "gate_pass": False}],
        primary_w=1.10,
    )
    assert failed["verdict"] == "shared_queue_oracle_gate_fail"
    assert failed["authorization"].startswith("do_not_test_target")
