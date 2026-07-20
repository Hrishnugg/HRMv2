from __future__ import annotations

import continuous_prm_c13_local_bellman_integration as K
import continuous_prm_c13_lhbl_multisuite as J


def _rows(label: str, family: str, improvement: int) -> list[dict]:
    rows = []
    for suite_index, suite in enumerate(J.DEV_SUITES):
        for world in range(4):
            field = 100 + suite_index
            current = field - improvement
            rows.append(
                {
                    "suite": suite,
                    "world_index": suite_index * 4 + world,
                    "world_seed": 1000 + suite_index * 4 + world,
                    "model_label": label,
                    "family": family,
                    "iteration": 8,
                    "alpha": 1.0,
                    "found": True,
                    "path_valid": True,
                    "current_expansions": current,
                    "current_cost_ratio_eval_only": 1.03,
                    "field_hrm_expansions": field,
                    "field_hrm_cost_ratio_eval_only": 1.04,
                    "delta_vs_field_hrm": current - field,
                    "scalar_hrm_expansions": field - 2,
                    "scalar_hrm_cost_ratio_eval_only": 1.05,
                    "delta_vs_scalar_hrm": current - (field - 2),
                    "inference_seconds": 0.01,
                    "local_backup_seconds": 0.02,
                }
            )
    return rows


def test_local_backup_gate_prefers_stronger_passing_candidate() -> None:
    cfg = K.LocalBackupConfig(bootstrap_replicates=2_000)
    rows = _rows("maze_i8", "maze", 6) + _rows(
        "multisuite_i8", "multisuite", 10
    )
    cells, pooled, verdict = K.summarize(cfg, rows)
    assert len(cells) == 12
    assert len(pooled) == 2
    assert verdict["gate_pass"] is True
    assert verdict["selected_candidate"]["model_label"] == "multisuite_i8"
    assert verdict["authorization"] == (
        "run_fixed_local_bellman_candidate_on_seed_offset_15000000"
    )


def test_local_backup_gate_rejects_cost_regime_violation() -> None:
    cfg = K.LocalBackupConfig(bootstrap_replicates=2_000)
    rows = _rows("multisuite_i8", "multisuite", 10)
    rows[0]["current_cost_ratio_eval_only"] = 1.11
    _, _, verdict = K.summarize(cfg, rows)
    assert verdict["gate_pass"] is False
    assert verdict["selected_candidate"] is None
