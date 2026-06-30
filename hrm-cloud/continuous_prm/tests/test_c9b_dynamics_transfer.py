import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parents[1]
import numpy as np
import continuous_prm_c9b_dynamics_transfer as C9B
import continuous_prm_common as C
import continuous_prm_c9_transfer as C9


def test_config_sources_and_suites():
    cfg = C9B.C9bConfig()
    assert cfg.backbones == "scalar_hrm,scalar_onlstm,field_unet"
    assert cfg.targets == "C_dyn_maze_dense,C_dyn_crossing,C_dyn_rooms_large"
    assert set(cfg.awareness_list()) == {"aware", "blind"}
    srcs = C9B.resolve_sources(cfg)
    assert set(srcs) == {(b, a) for b in ("scalar_hrm", "scalar_onlstm", "field_unet") for a in ("aware", "blind")}
    C9B.install()
    specs = C.build_anchor_specs()
    for t in C9B._parse_csv(cfg.targets):
        assert t in specs


def test_adapt_test_disjoint():
    C9B.install()
    cfg = C9B.C9bConfig(n_test=4, seed=7)
    adapt = C9B.adapt_world_seeds("C_dyn_crossing", K=4, seed_idx=0, cfg=cfg)
    test = C9B.test_world_seeds("C_dyn_crossing", cfg)
    assert len(adapt) == 4 and len(test) == 4
    assert set(adapt).isdisjoint(set(test))
    fa = {C9.world_fingerprint(C9B._build_world_only("C_dyn_crossing", s)) for s in adapt}
    ft = {C9.world_fingerprint(C9B._build_world_only("C_dyn_crossing", s)) for s in test}
    assert fa.isdisjoint(ft)
