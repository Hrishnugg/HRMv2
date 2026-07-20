import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_c13_lhbl_focal_fresh2 as F2


def test_fresh2_verdict_requires_both_locked_densities():
    summaries = [
        {"density": 192, "gate_pass": True},
        {"density": 211, "gate_pass": True},
    ]
    assert F2.build_verdict(summaries, [192, 211])["gate_pass"]
    summaries[1]["gate_pass"] = False
    assert not F2.build_verdict(summaries, [192, 211])["gate_pass"]


def test_fresh2_defaults_are_the_fixed_matched_control_candidate():
    cfg = F2.Fresh2Config()
    assert cfg.worlds == 12
    assert cfg.densities == "192,211"
    assert cfg.mode == "fhat"
    assert cfg.alpha == 0.25
    assert cfg.focal_w == 1.10
    assert cfg.fresh_seed_offset == 2_700_000
