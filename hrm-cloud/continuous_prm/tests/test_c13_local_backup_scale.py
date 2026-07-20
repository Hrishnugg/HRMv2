from __future__ import annotations

import continuous_prm_c13_lhbl_multisuite as J
import continuous_prm_c13_local_backup_scale as L


def test_scale_gate_selects_fastest_safe_alpha() -> None:
    cfg = L.ScaleConfig(bootstrap_replicates=2_000)
    rows = []
    for alpha, improvement, cost in ((1.0, 8, 1.03), (1.5, 16, 1.07)):
        for suite_index, suite in enumerate(J.DEV_SUITES):
            for world in range(cfg.worlds_per_suite):
                field = 100 + suite_index
                current = field - improvement
                rows.append(
                    {
                        "suite": suite,
                        "world_index": suite_index * cfg.worlds_per_suite + world,
                        "world_seed": 1000 + suite_index * cfg.worlds_per_suite + world,
                        "alpha": alpha,
                        "found": True,
                        "path_valid": True,
                        "current_expansions": current,
                        "current_cost_ratio_eval_only": cost,
                        "field_hrm_expansions": field,
                        "field_hrm_cost_ratio_eval_only": 1.04,
                        "delta_vs_field_hrm": current - field,
                        "scalar_hrm_expansions": field - 1,
                        "scalar_hrm_cost_ratio_eval_only": 1.05,
                        "delta_vs_scalar_hrm": current - (field - 1),
                        "inference_seconds": 0.01,
                        "local_backup_seconds": 0.02,
                    }
                )
    cfg.alphas = "1.00,1.50"
    _, _, verdict = L.summarize(cfg, rows)
    assert verdict["gate_pass"] is True
    assert verdict["selected_candidate"]["alpha"] == 1.5
    assert verdict["authorization"] == (
        "confirm_selected_alpha_on_seed_offset_15000000"
    )


def test_scale_gate_rejects_alpha_above_cost_ceiling() -> None:
    cfg = L.ScaleConfig(bootstrap_replicates=2_000, alphas="2.00")
    rows = []
    for suite_index, suite in enumerate(J.DEV_SUITES):
        for world in range(cfg.worlds_per_suite):
            field = 100
            rows.append(
                {
                    "suite": suite,
                    "world_index": suite_index * cfg.worlds_per_suite + world,
                    "world_seed": 2000 + suite_index * cfg.worlds_per_suite + world,
                    "alpha": 2.0,
                    "found": True,
                    "path_valid": True,
                    "current_expansions": 60,
                    "current_cost_ratio_eval_only": 1.11 if not rows else 1.05,
                    "field_hrm_expansions": field,
                    "field_hrm_cost_ratio_eval_only": 1.04,
                    "delta_vs_field_hrm": -40,
                    "scalar_hrm_expansions": 99,
                    "scalar_hrm_cost_ratio_eval_only": 1.05,
                    "delta_vs_scalar_hrm": -39,
                    "inference_seconds": 0.01,
                    "local_backup_seconds": 0.02,
                }
            )
    _, _, verdict = L.summarize(cfg, rows)
    assert verdict["gate_pass"] is False
