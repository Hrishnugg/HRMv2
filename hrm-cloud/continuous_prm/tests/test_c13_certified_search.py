import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_c13_certified_search as S


def _adj(edges, n):
    graph = [[] for _ in range(n)]
    for a, b, weight in edges:
        graph[a].append((b, float(weight)))
        graph[b].append((a, float(weight)))
    return graph


def _summary_row(provider, delta, *, certified=True, violation=False):
    return {
        "provider": provider,
        "focal_w": 1.10,
        "delta_total_vs_euclid_focal": float(delta),
        "certified": bool(certified),
        "bound_violation_eval_only": bool(violation),
        "phase1_expansions": 2,
        "phase1_reopens": 0,
        "phase1_cost_ratio_eval_only": 1.02,
        "certificate_expansions": 3,
        "total_expansions": 5,
        "euclid_focal_expansions": 7,
        "final_cost_ratio_eval_only": 1.02,
        "anchor_goal_popped": False,
        "anchor_improved_incumbent": False,
    }


def test_anchor_validation_accepts_consistent_and_rejects_inconsistent():
    graph = _adj([(0, 2, 1.0), (2, 1, 1.0), (0, 1, 3.0)], 3)
    check = S.validate_consistent_anchor(graph, np.array([2.0, 0.0, 1.0]))
    assert check["max_consistency_violation"] <= 1.0e-12

    with pytest.raises(ValueError, match="inconsistent"):
        S.validate_consistent_anchor(graph, np.array([3.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="zero at the goal"):
        S.validate_consistent_anchor(graph, np.array([2.0, 0.1, 1.0]))


def test_inadmissible_phase_one_reopens_after_a_better_path():
    graph = _adj(
        [(0, 2, 5.0), (0, 3, 1.0), (3, 2, 1.0), (2, 1, 10.0)],
        4,
    )
    arbitrary_rank = np.array([0.0, 1000.0, 0.0, 100.0])
    result = S.inadmissible_astar_incumbent(graph, arbitrary_rank, budget=10)

    assert result["found"]
    assert result["cost"] == pytest.approx(12.0)
    assert result["reopens"] == 1
    assert result["expansions"] == 5


def test_anchor_can_certify_an_incumbent_before_any_expansion():
    graph = _adj([(0, 1, 10.0)], 2)
    result = S.certify_incumbent_with_anchor(
        graph,
        np.array([10.0, 0.0]),
        incumbent_cost=10.0,
        w=1.10,
        budget=0,
    )

    assert result["certified"]
    assert result["expansions"] == 0
    assert not result["anchor_goal_popped"]
    assert result["proof"] == "incumbent_le_w_times_anchor_open_lower_bound"


def test_anchor_improves_a_bad_incumbent_before_certifying():
    graph = _adj([(0, 2, 1.0), (2, 1, 1.0), (0, 1, 5.0)], 3)
    result = S.certify_incumbent_with_anchor(
        graph,
        np.array([2.0, 0.0, 1.0]),
        incumbent_cost=4.0,
        w=1.10,
        budget=3,
    )

    assert result["certified"]
    assert result["final_cost"] == pytest.approx(2.0)
    assert result["anchor_goal_popped"]
    assert result["anchor_improved_incumbent"]
    assert result["certificate_ratio"] <= 1.10 + 1.0e-12


def test_anchor_refuses_a_certificate_when_budget_is_insufficient():
    graph = _adj([(0, 2, 1.0), (2, 1, 1.0), (0, 1, 5.0)], 3)
    result = S.certify_incumbent_with_anchor(
        graph,
        np.array([2.0, 0.0, 1.0]),
        incumbent_cost=4.0,
        w=1.10,
        budget=0,
    )

    assert not result["certified"]
    assert result["found"]
    assert result["final_cost"] == pytest.approx(4.0)
    assert result["proof"] == "budget_exhausted_without_certificate"


def test_gate_requires_five_of_six_wins_and_no_proof_failures():
    rows = [
        _summary_row("oracle_eval_only", delta)
        for delta in (-2, -2, -2, -2, -2, 1)
    ]
    summary = S.summarize_certified_rows(rows, required_win_fraction=0.80)[0]
    assert summary["required_wins"] == 5
    assert summary["wins"] == 5
    assert summary["gate_pass"]

    rows[0] = _summary_row("oracle_eval_only", -2, certified=False)
    failed = S.summarize_certified_rows(rows, required_win_fraction=0.80)[0]
    assert not failed["gate_pass"]

    four_wins = [
        _summary_row("oracle_eval_only", delta)
        for delta in (-3, -3, -3, -3, 1, 1)
    ]
    failed = S.summarize_certified_rows(four_wins, required_win_fraction=0.80)[0]
    assert not failed["gate_pass"]


def test_gate_verdict_separates_integration_target_and_representation_failures():
    def row(provider, gate_pass):
        return {"provider": provider, "focal_w": 1.10, "gate_pass": gate_pass}

    oracle_failure = S.build_gate_verdict(
        [row("oracle_eval_only", False), row("rollout_exact", True)], 1.10
    )
    assert oracle_failure["verdict"].startswith("reject_simple_certifier")

    target_failure = S.build_gate_verdict(
        [row("oracle_eval_only", True), row("rollout_exact", False)], 1.10
    )
    assert target_failure["verdict"].startswith("reject_current_rollout_target")

    representation_failure = S.build_gate_verdict(
        [
            row("oracle_eval_only", True),
            row("rollout_exact", True),
            row("hrm_padded", False),
        ],
        1.10,
    )
    assert representation_failure["verdict"].startswith("exact_target_passes")

    learned_pass = S.build_gate_verdict(
        [
            row("oracle_eval_only", True),
            row("rollout_exact", True),
            row("hrm_padded", True),
        ],
        1.10,
    )
    assert learned_pass["verdict"] == "provisional_learned_certified_gate_pass"
