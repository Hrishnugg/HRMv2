import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_c13_lhbl_focal_fresh3 as F3


def test_fresh3_verdict_requires_both_density_gates():
    summaries = [
        {"density": 192, "gate_pass": True},
        {"density": 211, "gate_pass": True},
    ]
    assert F3.build_verdict(summaries, [192, 211])["gate_pass"]
    summaries[0]["gate_pass"] = False
    assert not F3.build_verdict(summaries, [192, 211])["gate_pass"]


def test_fresh3_defaults_lock_iteration_four_and_the_new_seed_range():
    cfg = F3.Fresh3Config()
    assert cfg.checkpoint.endswith("flat_mlp_iteration_04.pt")
    assert cfg.mode == "fhat"
    assert cfg.alpha == 0.50
    assert cfg.worlds == 12
    assert cfg.densities == "192,211"
    assert cfg.fresh_seed_offset == 3_600_000
