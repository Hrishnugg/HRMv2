from __future__ import annotations

import copy

import continuous_prm_c13_hrm_alignment as O
import continuous_prm_c13_hrm_substitution as N
import continuous_prm_c13_identifiability as I
import continuous_prm_c13_lhbl_generated_v3 as H
import continuous_prm_c13_lhbl_multisuite as J


def _rows(
    summary_improvements: dict[tuple[int, float], int] | None = None,
    trimmed_improvement: int = 10,
    flat_improvement: int = 12,
) -> list[dict]:
    summary_improvements = summary_improvements or {
        (iteration, alpha): 20
        for iteration in (4, 6, 8)
        for alpha in (1.0, 1.5)
    }
    rows: list[dict] = []
    for suite_index, suite in enumerate(J.DEV_SUITES):
        for world in range(4):
            seed = 10_000 + suite_index * 100 + world
            common = {
                "phase": "development",
                "suite": suite,
                "world_seed": seed,
                "found": True,
                "path_valid": True,
            }
            field = 100 + suite_index
            rows.extend(
                [
                    {
                        **common,
                        "arm": "euclid",
                        "expansions": field + 20,
                        "cost_ratio_eval_only": 1.0,
                    },
                    {
                        **common,
                        "arm": "field_hrm",
                        "expansions": field,
                        "cost_ratio_eval_only": 1.028,
                    },
                    {
                        **common,
                        "arm": "scalar_hrm",
                        "expansions": field + 5,
                        "cost_ratio_eval_only": 1.035,
                    },
                ]
            )
            for (iteration, alpha), improvement in summary_improvements.items():
                rows.extend(
                    [
                        {
                            **common,
                            "arm": N._arm_name(
                                O.SUMMARY_FAMILY, iteration, alpha
                            ),
                            "expansions": field - improvement,
                            "cost_ratio_eval_only": 1.030,
                        },
                        {
                            **common,
                            "arm": N._arm_name(
                                O.TRIMMED_FAMILY, iteration, alpha
                            ),
                            "expansions": field - trimmed_improvement,
                            "cost_ratio_eval_only": 1.031,
                        },
                        {
                            **common,
                            "arm": N._arm_name(O.FLAT_FAMILY, iteration, alpha),
                            "expansions": field - flat_improvement,
                            "cost_ratio_eval_only": 1.029,
                        },
                    ]
                )
    return rows


def test_alignment_operating_points_and_intervention_are_frozen() -> None:
    cfg = O.HrmAlignmentConfig()
    assert N._parse_iterations(cfg.candidate_iterations) == [4, 6, 8]
    assert cfg.alphas == "1.00,1.50"
    assert cfg.primary_iteration == 8
    assert cfg.primary_alpha == 1.5
    assert cfg.sensor_radius_frac == 0.2
    assert cfg.confirmation_worlds_per_suite == 24
    assert cfg.confirmation_seed_offset == 20_000_000
    assert cfg.train_device == "cuda"
    assert cfg.evaluation_device == "cpu"
    assert O.SUMMARY_FAMILY == "hrm_summary_last"
    assert O.TRIMMED_FAMILY == "hrm_trimmed"


def test_summary_last_and_trimmed_start_from_identical_parameters() -> None:
    source = H.LHBLConfig(seed=17_413, hidden_dim=64)
    study = I.StudyConfig(hidden_dim=64)
    audit = O.initialization_audit(source, study)
    assert audit["all_initial_tensors_equal"] is True
    assert audit["trimmed_state_sha256"] == audit["summary_last_state_sha256"]
    assert audit["trimmed_parameters"] == audit["summary_last_parameters"]
    assert audit["trimmed_readout_mode"] == "trimmed"
    assert audit["summary_last_readout_mode"] == "summary_last"


def test_bound_families_restore_imported_module_globals() -> None:
    original = (N.HRM_FAMILY, N.FLAT_FAMILY)
    with O.bound_n_families(O.SUMMARY_FAMILY, O.FLAT_FAMILY):
        assert N.HRM_FAMILY == O.SUMMARY_FAMILY
        assert N.FLAT_FAMILY == O.FLAT_FAMILY
    assert (N.HRM_FAMILY, N.FLAT_FAMILY) == original


def test_candidate_passes_method_and_direct_readout_gates() -> None:
    cfg = O.HrmAlignmentConfig(bootstrap_replicates=2_000)
    summary = O.candidate_summary(cfg, _rows(), 8, 1.5, 24)
    assert summary["method_gate_pass"] is True
    assert summary["readout_gate_pass"] is True
    assert summary["overall_gate_pass"] is True
    assert summary["delta_vs_field_hrm_ci95_high"] < 0.0
    assert summary["delta_vs_trimmed_ci95_high"] < 0.0
    assert summary["negative_suites"] == 6


def test_method_compatibility_without_trimmed_win_does_not_pass_alignment() -> None:
    cfg = O.HrmAlignmentConfig(bootstrap_replicates=2_000)
    rows = _rows(
        summary_improvements={
            (iteration, alpha): 9
            for iteration in (4, 6, 8)
            for alpha in (1.0, 1.5)
        },
        trimmed_improvement=10,
    )
    summary = O.candidate_summary(cfg, rows, 8, 1.5, 24)
    assert summary["method_gate_pass"] is True
    assert summary["readout_gate_pass"] is False
    assert summary["overall_gate_pass"] is False
    assert summary["delta_vs_trimmed_mean"] > 0.0


def test_beating_flat_expansions_is_reported_but_not_required() -> None:
    cfg = O.HrmAlignmentConfig(bootstrap_replicates=2_000)
    summary = O.candidate_summary(
        cfg,
        _rows(flat_improvement=25),
        8,
        1.5,
        24,
    )
    assert summary["delta_vs_flat_mean"] > 0.0
    assert summary["overall_gate_pass"] is True


def test_flat_relative_cost_failure_rejects_method_gate() -> None:
    cfg = O.HrmAlignmentConfig(bootstrap_replicates=2_000)
    rows = _rows()
    target = N._arm_name(O.SUMMARY_FAMILY, 8, 1.5)
    for row in rows:
        if row["arm"] == target:
            row["cost_ratio_eval_only"] = 1.036
    summary = O.candidate_summary(cfg, rows, 8, 1.5, 24)
    assert summary["method_gate_pass"] is False
    assert (
        summary["method_gate_conditions"]["mean_cost_within_flat_margin"]
        is False
    )


def test_selection_uses_preregistered_tie_break_order() -> None:
    cfg = O.HrmAlignmentConfig(bootstrap_replicates=2_000)
    improvements = {
        (4, 1.0): 13,
        (4, 1.5): 14,
        (6, 1.0): 16,
        (6, 1.5): 24,
        (8, 1.0): 18,
        (8, 1.5): 20,
    }
    candidates, verdict = O.summarize_development(
        cfg,
        _rows(summary_improvements=improvements),
        [4, 6, 8],
        [1.0, 1.5],
        24,
    )
    assert len(candidates) == 6
    assert verdict["gate_pass"] is True
    assert verdict["selected_candidate"]["iteration"] == 6
    assert verdict["selected_candidate"]["alpha"] == 1.5
    assert verdict["fixed_primary_cell"]["iteration"] == 8
    assert verdict["fixed_primary_cell"]["alpha"] == 1.5


def test_duplicate_pair_comparison_ignores_timing_only() -> None:
    row = {
        "phase": "development",
        "suite": "C_hard_maze",
        "arm": "field_hrm",
        "family": "field_hrm",
        "runtime_information_boundary": "map",
        "suite_world_index": 0,
        "world_index": 0,
        "world_seed": 1,
        "roadmap_seed": 2,
        "expansions": 10,
        "found": True,
        "path_valid": True,
        "cost": 1.25,
        "optimal": 1.2,
        "cost_ratio_eval_only": 1.25 / 1.2,
        "model_seconds": 0.1,
    }
    changed_time = copy.deepcopy(row)
    changed_time["model_seconds"] = 99.0
    assert O._same_result(row, changed_time) is True
    changed_result = copy.deepcopy(row)
    changed_result["expansions"] = 11
    assert O._same_result(row, changed_result) is False

