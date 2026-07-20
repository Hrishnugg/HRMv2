import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_c13_shared_queue_target as T


def _row(delta, *, certified=True, path_valid=True, accounting=True, violation=False):
    return {
        "focal_w": 1.10,
        "delta_vs_euclid_focal": float(delta),
        "certified": bool(certified),
        "bound_violation_eval_only": bool(violation),
        "path_valid": bool(path_valid),
        "anchor_lower_bound_exceeds_optimal_eval_only": False,
        "expansion_accounting_valid": bool(accounting),
        "expansions": 5,
        "rank_expansions": 3,
        "anchor_expansions": 2,
        "duplicate_state_expansions": 0,
        "rank_eligibility_checks": 5,
        "rank_eligible_choices": 3,
        "rollout_label_rate": 0.75,
        "rollout_rank_vs_oracle_spearman_eval_only": 0.60,
        "rollout_rank_to_oracle_ratio_start_eval_only": 3.0,
        "euclid_focal_expansions": 7,
        "euclid_astar_expansions": 8,
        "shared_oracle_expansions": 4,
        "delta_vs_shared_oracle": 1,
        "independent_exact_total_expansions": 11,
        "saved_vs_independent_exact": 6,
        "final_cost_ratio_eval_only": 1.02,
    }


def test_exact_target_gate_keeps_the_five_of_six_contract():
    rows = [_row(delta) for delta in (-2, -2, -2, -2, -2, 1)]
    summary = T.summarize_rows(rows, required_win_fraction=0.80)[0]
    assert summary["required_wins"] == 5
    assert summary["wins"] == 5
    assert summary["gate_pass"]
    assert summary["rank_eligible_choice_rate"] == pytest.approx(0.60)


@pytest.mark.parametrize(
    "replacement",
    [
        _row(-2, certified=False),
        _row(-2, path_valid=False),
        _row(-2, accounting=False),
        _row(-2, violation=True),
    ],
)
def test_exact_target_gate_rejects_safety_or_accounting_failure(replacement):
    rows = [_row(-2) for _ in range(6)]
    rows[0] = replacement
    assert not T.summarize_rows(rows, required_win_fraction=0.80)[0]["gate_pass"]


def test_exact_target_gate_verdict_controls_model_authorization():
    passed = T.build_gate_verdict(
        [{"focal_w": 1.10, "gate_pass": True}],
        primary_w=1.10,
    )
    assert passed["verdict"] == "shared_queue_exact_rollout_gate_pass"
    assert passed["authorization"] == "run_frozen_learned_providers_next"

    failed = T.build_gate_verdict(
        [{"focal_w": 1.10, "gate_pass": False}],
        primary_w=1.10,
    )
    assert failed["verdict"] == "shared_queue_exact_rollout_gate_fail"
    assert failed["authorization"].startswith("repair_target_alignment")
