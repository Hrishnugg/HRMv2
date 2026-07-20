import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_c13_lhbl_replication as R


def _row(density, focal_delta=-2, same_delta=-5, safety=False):
    return {
        "density": density,
        "delta_vs_euclid_focal": focal_delta,
        "delta_vs_same_search_euclid": same_delta,
        "safety_failure": safety,
        "expansions": 100,
        "euclid_focal_expansions": 100 - focal_delta,
        "same_search_euclid_expansions": 100 - same_delta,
        "final_cost_ratio_eval_only": 1.01,
        "direct_learned_astar_cost_ratio_eval_only": 1.02,
        "inference_seconds": 0.01,
        "learned_search_seconds": 0.02,
    }


def test_bootstrap_mean_ci_is_deterministic_and_brackets_the_sample_mean():
    first = R.bootstrap_mean_ci([-5, -4, -3, -2], 2_000, 17)
    second = R.bootstrap_mean_ci([-5, -4, -3, -2], 2_000, 17)
    assert first == second
    assert first[0] <= -3.5 <= first[1]


def test_replication_gate_requires_both_fixed_densities():
    passing = [
        {"density": 192, "gate_pass": True},
        {"density": 211, "gate_pass": True},
    ]
    assert R.build_verdict(passing, [192, 211])["gate_pass"]
    assert not R.build_verdict(passing[:1], [192, 211])["gate_pass"]
    passing[1]["gate_pass"] = False
    assert not R.build_verdict(passing, [192, 211])["gate_pass"]


def test_replication_summary_enforces_ten_of_twelve_wins_at_each_density():
    cfg = R.ReplicationConfig(bootstrap_replicates=1_000)
    rows = [_row(density) for density in (192, 211) for _ in range(12)]
    summaries = R.summarize_replication(rows, cfg)
    assert all(row["required_wins"] == 10 for row in summaries)
    assert all(row["gate_pass"] for row in summaries)

    failing = list(rows)
    failing[:3] = [_row(192, focal_delta=1, same_delta=1) for _ in range(3)]
    summaries = R.summarize_replication(failing, cfg)
    density_192 = next(row for row in summaries if row["density"] == 192)
    assert density_192["focal_wins"] == 9
    assert not density_192["gate_pass"]


def test_locked_defaults_match_the_development_selection():
    cfg = R.ReplicationConfig()
    assert cfg.worlds == 12
    assert cfg.densities == "192,211"
    assert cfg.alpha == pytest.approx(1.0)
    assert cfg.focal_w == pytest.approx(1.10)
    assert cfg.required_win_fraction == pytest.approx(0.80)
