import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_c13_shared_queue_calibration as F


def _row(provider, alpha, focal_delta, control_delta):
    return {
        "provider": provider,
        "alpha": float(alpha),
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
        "same_search_euclid_expansions": 11,
        "delta_vs_same_search_euclid": float(control_delta),
        "c13e_exact_expansions": 13,
        "shared_oracle_expansions": 8,
        "final_cost_ratio_eval_only": 1.0,
    }


def test_calibrated_rank_has_exact_endpoints_and_preserves_residual_order():
    euclid = np.array([1.0, 2.0, 3.0])
    rollout = np.array([5.0, 4.0, 9.0])
    assert np.allclose(F.calibrated_rank(euclid, rollout, 0.0), euclid)
    assert np.allclose(F.calibrated_rank(euclid, rollout, 1.0), rollout)
    blended = F.calibrated_rank(euclid, rollout, 0.25)
    assert np.allclose(blended - euclid, 0.25 * (rollout - euclid))


def test_calibrated_rank_rejects_invalid_scale_or_target():
    with pytest.raises(ValueError, match="alpha"):
        F.calibrated_rank(np.array([1.0]), np.array([2.0]), 1.5)
    with pytest.raises(ValueError, match="undercut"):
        F.calibrated_rank(np.array([2.0]), np.array([1.0]), 0.5)


def test_calibration_gate_requires_focal_and_same_search_control_gains():
    candidate = [
        _row("rollout_blend_a0p10", 0.10, focal, control)
        for focal, control in zip((-2, -2, -2, -2, -2, 1), (-1, -1, -1, -1, 0, 0))
    ]
    summary = F.summarize_rows(candidate, 0.80, 4)[0]
    assert summary["focal_wins"] == 5
    assert summary["euclid_rank_wins"] == 4
    assert summary["calibration_gate_pass"]

    no_control_gain = [
        _row("rollout_blend_a0p10", 0.10, -2, 0) for _ in range(6)
    ]
    assert not F.summarize_rows(no_control_gain, 0.80, 4)[0][
        "calibration_gate_pass"
    ]


def test_verdict_requires_fresh_replication_for_a_passing_candidate():
    summaries = [
        {
            "provider": "euclid_rank",
            "alpha": 0.0,
            "focal_w": 1.10,
            "calibration_gate_pass": False,
            "expansions_mean": 11.0,
        },
        {
            "provider": "rollout_exact",
            "alpha": 1.0,
            "focal_w": 1.10,
            "calibration_gate_pass": False,
            "expansions_mean": 13.0,
        },
        {
            "provider": "rollout_blend_a0p10",
            "alpha": 0.10,
            "focal_w": 1.10,
            "calibration_gate_pass": True,
            "expansions_mean": 9.0,
        },
    ]
    verdict = F.build_verdict(summaries, 1.10)
    assert verdict["calibration_candidate_found"]
    assert verdict["selected_candidate"]["alpha"] == pytest.approx(0.10)
    assert verdict["authorization"].startswith("replicate_fixed_alpha")


def test_verdict_advances_to_local_target_when_calibration_fails():
    summaries = [
        {
            "provider": "euclid_rank",
            "alpha": 0.0,
            "focal_w": 1.10,
            "calibration_gate_pass": False,
            "expansions_mean": 11.0,
        },
        {
            "provider": "rollout_exact",
            "alpha": 1.0,
            "focal_w": 1.10,
            "calibration_gate_pass": False,
            "expansions_mean": 13.0,
        },
    ]
    verdict = F.build_verdict(summaries, 1.10)
    assert not verdict["calibration_candidate_found"]
    assert verdict["authorization"] == "advance_to_exact_bounded_local_escape_target"
