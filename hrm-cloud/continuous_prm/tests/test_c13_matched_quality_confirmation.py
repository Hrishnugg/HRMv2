from __future__ import annotations

import continuous_prm_c13_lhbl_multisuite as J
import continuous_prm_c13_matched_quality_confirmation as M


def _synthetic_rows(cfg: M.ConfirmationConfig, current_cost: float = 1.040) -> list[dict]:
    rows = []
    for suite_index, suite in enumerate(J.DEV_SUITES):
        for world in range(cfg.worlds_per_suite):
            seed = 10_000 + suite_index * cfg.worlds_per_suite + world
            common = {
                "suite": suite,
                "world_index": suite_index * cfg.worlds_per_suite + world,
                "world_seed": seed,
                "found": True,
                "path_valid": True,
                "bound_violation_eval_only": False,
                "certificate_violation": False,
            }
            rows.extend(
                [
                    {
                        **common,
                        "arm": M.CURRENT_ARM,
                        "expansions": 80,
                        "cost_ratio_eval_only": current_cost,
                    },
                    {
                        **common,
                        "arm": "field_hrm",
                        "expansions": 100,
                        "cost_ratio_eval_only": 1.038,
                    },
                    {
                        **common,
                        "arm": M.SAFETY_ARM,
                        "expansions": 110,
                        "cost_ratio_eval_only": 1.050,
                    },
                ]
            )
    return rows


def test_matched_quality_gate_passes_fixed_empirical_pareto_result() -> None:
    cfg = M.ConfirmationConfig(worlds_per_suite=2, bootstrap_replicates=2_000)
    rows = _synthetic_rows(cfg)
    pairs = M.pairwise_comparisons(rows, cfg)
    verdict = M.build_verdict(rows, pairs, cfg)
    assert verdict["gate_pass"] is True
    assert verdict["bounded_control"]["pass"] is True
    assert verdict["negative_suites"] == 6
    assert verdict["authorization"] == (
        "document_confirmed_current_state_matched_quality_improvement"
    )


def test_matched_quality_gate_rejects_relative_mean_cost_failure() -> None:
    cfg = M.ConfirmationConfig(worlds_per_suite=2, bootstrap_replicates=2_000)
    rows = _synthetic_rows(cfg, current_cost=1.044)
    pairs = M.pairwise_comparisons(rows, cfg)
    verdict = M.build_verdict(rows, pairs, cfg)
    assert verdict["gate_pass"] is False
    assert verdict["conditions"]["current_mean_cost_within_0_005_of_field_hrm"] is False


def test_matched_quality_gate_rejects_relative_max_cost_failure() -> None:
    cfg = M.ConfirmationConfig(worlds_per_suite=2, bootstrap_replicates=2_000)
    rows = _synthetic_rows(cfg)
    current = [row for row in rows if row["arm"] == M.CURRENT_ARM]
    current[0]["cost_ratio_eval_only"] = 1.059
    pairs = M.pairwise_comparisons(rows, cfg)
    verdict = M.build_verdict(rows, pairs, cfg)
    assert verdict["gate_pass"] is False
    assert verdict["conditions"]["current_max_cost_within_0_02_of_field_hrm"] is False


def test_confirmation_operating_point_is_frozen() -> None:
    cfg = M.ConfirmationConfig()
    assert cfg.current_iteration == 8
    assert cfg.current_alpha == 1.50
    assert cfg.sensor_radius_frac == 0.20
    assert cfg.seed_offset == 15_000_000
    assert cfg.worlds_per_suite == 24
    assert cfg.budget == 384
    assert cfg.safety_iteration == 4
    assert cfg.safety_alpha == 0.50
    assert cfg.safety_w == 1.10
    assert M.CURRENT_BOUNDARY.startswith("current_goal_geometry")
    assert "occupancy" not in M.CURRENT_BOUNDARY
