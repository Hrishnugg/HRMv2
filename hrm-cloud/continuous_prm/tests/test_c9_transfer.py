import os, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parents[1]

import continuous_prm_c9_transfer as C9


def test_c9config_defaults():
    cfg = C9.C9Config()
    assert cfg.targets == "C_hard_maze_dense,C_hard_bugtrap,C_hard_rooms_large"
    assert cfg.backbones == "hrm,onlstm"
    assert cfg.k_grid == "0,1,2,4,8,16,32"
    assert cfg.n_adapt_seeds >= 1
    assert cfg.source_dir.endswith("c7_local")


@pytest.mark.skipif(not (HERE / "runs/c7_local/checkpoints/avgbase__hrm.pt").exists(),
                    reason="C7 avgbase base checkpoint not present")
def test_load_source_base_real():
    import torch
    dev = torch.device("cpu")
    base = C9.load_source_base(HERE / "runs/c7_local", "hrm", dev)
    assert base.ckpt_path.exists()
    assert base.feature_cfg.token_dim > 0
    assert base.backbone_cfg.name == "hrm"
    assert base.model is not None
