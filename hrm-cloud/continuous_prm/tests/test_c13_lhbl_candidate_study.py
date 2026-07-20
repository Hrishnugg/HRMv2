import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_c13_lhbl_candidate_study as C


def _cell(variant, iteration, alpha, cohort, density, gate, mean=-2.0):
    return {
        "variant": variant,
        "iteration": iteration,
        "alpha": alpha,
        "cohort": cohort,
        "density": density,
        "gate_pass": gate,
        "safety_failures": 0,
        "delta_mean": mean,
    }


def _pooled(variant, iteration, alpha, density, gate, mean):
    return {
        "variant": variant,
        "iteration": iteration,
        "alpha": alpha,
        "density": density,
        "gate_pass": gate,
        "safety_failures": 0,
        "delta_mean": mean,
    }


def test_selection_prefers_an_all_four_cells_candidate_over_a_better_pooled_only_one():
    cells = []
    for cohort in ("cohort_a", "cohort_b"):
        for density in (192, 211):
            cells.append(_cell("model", 7, 0.25, cohort, density, True))
            cells.append(
                _cell(
                    "model_plus_local_backup",
                    8,
                    0.5,
                    cohort,
                    density,
                    not (cohort == "cohort_b" and density == 192),
                    mean=-8.0,
                )
            )
    pooled = [
        _pooled("model", 7, 0.25, density, True, -3.0)
        for density in (192, 211)
    ] + [
        _pooled("model_plus_local_backup", 8, 0.5, density, True, -9.0)
        for density in (192, 211)
    ]
    verdict = C.select_candidate(cells, pooled)
    assert verdict["selection_tier"] == "all_four_cells"
    assert verdict["selected_candidate"]["variant"] == "model"
    assert verdict["selected_candidate"]["iteration"] == 7


def test_selection_allows_pooled_gate_only_when_every_cell_mean_is_negative_and_safe():
    cells = [
        _cell("model", 6, 0.25, cohort, density, False, mean=-1.0)
        for cohort in ("cohort_a", "cohort_b")
        for density in (192, 211)
    ]
    pooled = [
        _pooled("model", 6, 0.25, density, True, -4.0)
        for density in (192, 211)
    ]
    verdict = C.select_candidate(cells, pooled)
    assert verdict["selection_tier"] == "pooled_24_worlds"
    assert verdict["fresh_replication_required"]

    cells[0]["delta_mean"] = 0.5
    verdict = C.select_candidate(cells, pooled)
    assert verdict["selection_tier"] == "none"
    assert not verdict["fresh_replication_required"]
