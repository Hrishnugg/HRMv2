import os, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parents[1]

import continuous_prm_c9_transfer as C9
import continuous_prm_common as C
import continuous_prm_c7_hard_maps as H7


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


def test_world_fingerprint_and_split_disjoint():
    import numpy as np
    H7.install_c7_hard_maps()  # noqa: F821  (H7 imported in module under test; re-import here)
    specs = C.build_anchor_specs()
    spec = specs["C_hard_bugtrap"]
    cfg = C9.C9Config(n_test=6)
    rmcfg = C.RoadmapConfig(n_nodes=cfg.roadmap_nodes, k_neighbors=cfg.roadmap_k)

    test_fps = set()
    for _, world, _ in C9.iter_test_worlds(spec, suite_idx=1, cfg=cfg, roadmap_cfg=rmcfg, n_test=cfg.n_test):
        test_fps.add(C9.world_fingerprint(world))
    assert len(test_fps) == cfg.n_test

    adapt_fps = set()
    for s in (0, 1):
        seed = C9.adapt_seed(spec.name, K=4, adapt_seed_idx=s, base_seed=cfg.seed)
        for fp in C9.adapt_world_fingerprints(spec, n_worlds=4, nodes_per_world=8,
                                              roadmap_cfg=rmcfg, feature_cfg=C.FeatureConfig(), seed=seed):
            adapt_fps.add(fp)
    assert adapt_fps.isdisjoint(test_fps)


def test_adapt_seed_deterministic():
    a = C9.adapt_seed("C_hard_bugtrap", 4, 2, 1234)
    b = C9.adapt_seed("C_hard_bugtrap", 4, 2, 1234)
    c = C9.adapt_seed("C_hard_bugtrap", 4, 3, 1234)
    assert a == b and a != c
