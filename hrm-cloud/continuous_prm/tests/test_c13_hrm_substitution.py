from __future__ import annotations

import copy

import continuous_prm_c13_hrm_substitution as N
import continuous_prm_c13_lhbl_multisuite as J


def _rows(
    cfg: N.HrmSubstitutionConfig,
    improvements: dict[tuple[int, float], int] | None = None,
    flat_improvement: int = 12,
) -> list[dict]:
    improvements = improvements or {
        (iteration, alpha): 10
        for iteration in (4, 6, 8)
        for alpha in (1.0, 1.5)
    }
    rows: list[dict] = []
    for suite_index, suite in enumerate(J.DEV_SUITES):
        for world in range(4):
            seed = 10_000 + suite_index * 100 + world
            common = {
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
            for (iteration, alpha), improvement in improvements.items():
                rows.extend(
                    [
                        {
                            **common,
                            "arm": N._arm_name(
                                N.HRM_FAMILY, iteration, alpha
                            ),
                            "expansions": field - improvement,
                            "cost_ratio_eval_only": 1.030,
                        },
                        {
                            **common,
                            "arm": N._arm_name(
                                N.FLAT_FAMILY, iteration, alpha
                            ),
                            "expansions": field - flat_improvement,
                            "cost_ratio_eval_only": 1.031,
                        },
                    ]
                )
    return rows


def test_hrm_substitution_operating_points_are_frozen() -> None:
    cfg = N.HrmSubstitutionConfig()
    assert N._parse_iterations(cfg.candidate_iterations) == [4, 6, 8]
    assert cfg.alphas == "1.00,1.50"
    assert cfg.primary_iteration == 8
    assert cfg.primary_alpha == 1.50
    assert cfg.sensor_radius_frac == 0.20
    assert cfg.confirmation_worlds_per_suite == 24
    assert cfg.confirmation_seed_offset == 20_000_000
    assert cfg.train_device == "cuda"
    assert cfg.evaluation_device == "cpu"


def test_candidate_passes_field_gate_and_architecture_win() -> None:
    cfg = N.HrmSubstitutionConfig(bootstrap_replicates=2_000)
    rows = _rows(
        cfg,
        improvements={
            (iteration, alpha): 18
            for iteration in (4, 6, 8)
            for alpha in (1.0, 1.5)
        },
        flat_improvement=12,
    )
    summary = N.candidate_summary(cfg, rows, 8, 1.5, 24)
    assert summary["field_gate_pass"] is True
    assert summary["architecture_win"] is True
    assert summary["negative_suites"] == 6
    assert summary["delta_vs_field_hrm_ci95_high"] < 0.0
    assert summary["delta_vs_flat_ci95_high"] < 0.0


def test_field_compatibility_does_not_imply_architecture_win() -> None:
    cfg = N.HrmSubstitutionConfig(bootstrap_replicates=2_000)
    rows = _rows(cfg, flat_improvement=15)
    summary = N.candidate_summary(cfg, rows, 8, 1.5, 24)
    assert summary["field_gate_pass"] is True
    assert summary["architecture_win"] is False
    assert summary["delta_vs_flat_mean"] > 0.0


def test_development_selection_uses_preregistered_tie_break_order() -> None:
    cfg = N.HrmSubstitutionConfig(bootstrap_replicates=2_000)
    improvements = {
        (4, 1.0): 8,
        (4, 1.5): 9,
        (6, 1.0): 11,
        (6, 1.5): 20,
        (8, 1.0): 12,
        (8, 1.5): 10,
    }
    rows = _rows(cfg, improvements=improvements)
    candidates, verdict = N.summarize_development(
        cfg, rows, [4, 6, 8], [1.0, 1.5], 24
    )
    assert len(candidates) == 6
    assert verdict["gate_pass"] is True
    assert verdict["selected_candidate"]["iteration"] == 6
    assert verdict["selected_candidate"]["alpha"] == 1.5
    assert verdict["fixed_primary_cell"]["iteration"] == 8
    assert verdict["fixed_primary_cell"]["alpha"] == 1.5


def test_development_gate_rejects_relative_cost_failure() -> None:
    cfg = N.HrmSubstitutionConfig(bootstrap_replicates=2_000)
    rows = _rows(cfg)
    target = N._arm_name(N.HRM_FAMILY, 8, 1.5)
    for row in rows:
        if row["arm"] == target:
            row["cost_ratio_eval_only"] = 1.034
    summary = N.candidate_summary(cfg, rows, 8, 1.5, 24)
    assert summary["field_gate_pass"] is False
    assert (
        summary["field_gate_conditions"]["mean_cost_within_field_margin"]
        is False
    )


def test_source_replay_check_includes_cache_hash_and_reuse_status() -> None:
    record = {
        "split": "development",
        "suite": "C_hard_maze",
        "suite_world_index": 0,
        "global_world_index": 0,
        "world_seed": 123,
        "roadmap_seed": 140,
        "nodes": 192,
        "edges": 700,
        "cache": "cache.npz",
        "cache_sha256": "abc",
        "cache_status": "reused",
    }
    N._verify_replay_records("development", [record], [record])
    changed = copy.deepcopy(record)
    changed["cache_sha256"] = "def"
    try:
        N._verify_replay_records("development", [changed], [record])
    except RuntimeError:
        pass
    else:
        raise AssertionError("changed cache hash should fail the replay audit")
    created = copy.deepcopy(record)
    created["cache_status"] = "created"
    try:
        N._verify_replay_records("development", [created], [record])
    except RuntimeError:
        pass
    else:
        raise AssertionError("new cache creation should fail the replay audit")
