from __future__ import annotations

from pathlib import Path

import continuous_prm_c13_lhbl_multisuite as J


def test_training_configuration_preserves_current_state_contract() -> None:
    cfg = J.MultiSuiteConfig()
    train = J.training_config(cfg)
    assert train.models == "flat_mlp"
    assert train.train_worlds == 96
    assert train.validation_worlds == 24
    assert train.sensor_radius_frac == 0.20
    assert train.num_rays == train.ray_steps == 32
    assert train.max_neighbors == 24
    assert train.alphas == "0.25,0.50,0.75,1.00"


def test_feature_cache_is_reused_without_changing_features(tmp_path: Path) -> None:
    cfg = J.MultiSuiteConfig(
        out_dir=str(tmp_path / "run"),
        train_worlds_per_suite=1,
        roadmap_nodes=64,
        roadmap_k=7,
        num_rays=4,
        ray_steps=4,
        max_neighbors=8,
        max_world_retries=50,
    )
    first, first_records, _ = J.build_balanced_bundles(
        cfg, "smoke", ("C_hard_maze",), 1, 123_000
    )
    second, second_records, _ = J.build_balanced_bundles(
        cfg, "smoke", ("C_hard_maze",), 1, 123_000
    )
    assert first_records[0]["cache_status"] == "created"
    assert second_records[0]["cache_status"] == "reused"
    assert first_records[0]["cache_sha256"] == second_records[0]["cache_sha256"]
    assert first[0].features.shape == second[0].features.shape == (64, 13, 16)
    assert (first[0].features == second[0].features).all()


def test_development_gate_selects_only_preregistered_pass() -> None:
    cfg = J.MultiSuiteConfig(
        development_worlds_per_suite=4,
        bootstrap_replicates=2_000,
    )
    rows = []
    for iteration in (1, 2):
        for suite_index, suite in enumerate(J.DEV_SUITES):
            for world in range(4):
                field = 100 + suite_index
                current = field - (10 if iteration == 2 else -3)
                rows.append(
                    {
                        "suite": suite,
                        "world_index": suite_index * 4 + world,
                        "world_seed": 1000 + suite_index * 4 + world,
                        "iteration": iteration,
                        "alpha": 1.0,
                        "current_expansions": current,
                        "current_cost_ratio_eval_only": 1.02,
                        "field_hrm_expansions": field,
                        "field_hrm_cost_ratio_eval_only": 1.03,
                        "delta_vs_field_hrm": current - field,
                        "scalar_hrm_expansions": field - 1,
                        "scalar_hrm_cost_ratio_eval_only": 1.04,
                        "delta_vs_scalar_hrm": current - (field - 1),
                    }
                )
    _, pooled, verdict = J.summarize_candidates(cfg, rows)
    assert len(pooled) == 2
    assert verdict["gate_pass"] is True
    assert verdict["selected_candidate"]["iteration"] == 2
    assert verdict["authorization"] == (
        "run_fixed_candidate_on_untouched_six_suite_seed_block"
    )
