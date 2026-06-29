import os, sys
import dataclasses as _dc
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


def dataclasses_replace_epochs(train_cfg, epochs):
    return _dc.replace(train_cfg, base_epochs=int(epochs))


def _tiny_npz(tmp_path, feature_cfg, n=24):
    import numpy as np
    x = np.random.RandomState(0).randn(n, feature_cfg.seq_len, feature_cfg.token_dim).astype("float32")
    y = np.abs(np.random.RandomState(1).randn(n)).astype("float32")
    p = tmp_path / "tiny.npz"
    np.savez_compressed(p, x=x, y=y, euclid=np.ones(n, "float32"), side=np.ones(n, "float32"))
    return p


@pytest.mark.skipif(not (HERE / "runs/c7_local/checkpoints/avgbase__hrm.pt").exists(),
                    reason="base missing")
def test_train_scalar_model_full_ft_vs_scratch(tmp_path):
    import torch
    dev = torch.device("cpu")
    base = C9.load_source_base(HERE / "runs/c7_local", "hrm", dev)
    npz = _tiny_npz(tmp_path, base.feature_cfg)
    tcfg = dataclasses_replace_epochs(base.train_cfg, 1)
    ck_ft = tmp_path / "ft.pt"
    C9.train_scalar_model(base.backbone_cfg, npz, ck_ft, base.feature_cfg, tcfg, dev, seed=0, init_ckpt=base.ckpt_path)
    assert ck_ft.exists()
    ck_sc = tmp_path / "sc.pt"
    C9.train_scalar_model(base.backbone_cfg, npz, ck_sc, base.feature_cfg, tcfg, dev, seed=0, init_ckpt=None)
    assert ck_sc.exists()
    pl = torch.load(ck_ft, map_location="cpu")
    assert {"model", "backbone_cfg", "feature_cfg", "train_cfg"} <= set(pl)


@pytest.mark.skipif(not (HERE / "runs/c7_local/checkpoints/avgbase__hrm.pt").exists(), reason="base missing")
def test_load_scalar_provider_base_and_full_ft(tmp_path):
    import torch, numpy as np
    dev = torch.device("cpu")
    base = C9.load_source_base(HERE / "runs/c7_local", "hrm", dev)
    prov0 = C9.load_scalar_provider(base.ckpt_path, dev)
    assert prov0.name == "scalar_hrm"
    npz = _tiny_npz(tmp_path, base.feature_cfg)
    tcfg = dataclasses_replace_epochs(base.train_cfg, 1)
    ck = tmp_path / "ft.pt"
    C9.train_scalar_model(base.backbone_cfg, npz, ck, base.feature_cfg, tcfg, dev, seed=0, init_ckpt=base.ckpt_path)
    prov1 = C9.load_scalar_provider(ck, dev)
    H7.install_c7_hard_maps(); specs = C.build_anchor_specs()
    rmcfg = C.RoadmapConfig(n_nodes=64, k_neighbors=7)
    w = C.build_world(specs["C_hard_bugtrap"], 7, rmcfg.min_start_goal_dist_frac)
    rm = C.build_prm(w, rmcfg, seed=24)
    h = prov1.node_h(w, rm, goal_idx=1)
    assert np.isfinite(h).all() and h.shape[0] == rm.points.shape[0]


@pytest.mark.skipif(not (HERE / "runs/c7_local/checkpoints/avgbase__hrm.pt").exists(), reason="base missing")
def test_run_adapt_smoke(tmp_path):
    import torch
    cfg = C9.C9Config(
        source_dir=str(HERE / "runs/c7_local"), out_dir=str(tmp_path / "c9"),
        targets="C_hard_bugtrap", backbones="hrm", k_grid="0,1", n_adapt_seeds=1,
        n_test=4, adapt_epochs=1, roadmap_nodes=192, roadmap_k=7, cpu=True, seed=7,
    )
    man = C9.run_adapt(cfg, torch.device("cpu"))
    arms = {(a["K"], a["method"]) for a in man["arms"]}
    assert (1, "lora") in arms and (1, "full_ft") in arms and (1, "scratch") in arms
    assert not any(a["K"] == 0 for a in man["arms"])
    for a in man["arms"]:
        assert Path(a["ckpt"]).exists()


@pytest.mark.skipif(not (HERE / "runs/c7_local/checkpoints/avgbase__hrm.pt").exists(), reason="base missing")
def test_run_eval_smoke(tmp_path):
    import torch, csv
    cfg = C9.C9Config(source_dir=str(HERE / "runs/c7_local"), out_dir=str(tmp_path / "c9"),
                      targets="C_hard_bugtrap", backbones="hrm", k_grid="0,1", n_adapt_seeds=1,
                      n_test=4, adapt_epochs=1, roadmap_nodes=192, roadmap_k=7, budgets="200,400",
                      w_values="1.0", cpu=True, seed=7)
    dev = torch.device("cpu")
    C9.run_adapt(cfg, dev)
    raw = C9.run_eval(cfg, dev)
    rows = list(csv.DictReader(open(raw, newline="")))
    provs = {r["provider"] for r in rows}
    assert "euclid" in provs and "oracle" in provs
    assert any(p.startswith("zeroshot_hrm") for p in provs)
    assert any(p.startswith("lora_hrm") for p in provs)
    assert all("target" in r and "K" in r for r in rows)


def _write_synth_raw(path):
    import csv
    rows = []
    def row(provider, method, K, wi, exp, found=True):
        return dict(target="C_hard_bugtrap", K=K, seed=0, method=method, backbone="hrm",
                    suite="C_hard_bugtrap", world_index=wi, provider=provider, mode="astar",
                    w="", budget=400, found=found, expansions=exp, closed=exp, cost=1.0,
                    optimal=1.0, suboptimality=1.0, nonfinite=0)
    for wi in (0, 1):
        rows.append(row("euclid", "euclid", -1, wi, 100))
        rows.append(row("oracle", "oracle", -1, wi, 20))
        rows.append(row("lora_hrm_K4_s0", "lora", 4, wi, 40))
        rows.append(row("scratch_hrm_K4_s0", "scratch", 4, wi, 80))
        rows.append(row("zeroshot_hrm", "zero_shot", 0, wi, 60))
    cols = C9.RAW_COLS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in cols})


def test_analyze_curve_ordering(tmp_path):
    import csv
    raw = tmp_path / "raw.csv"; _write_synth_raw(raw)
    out = C9.analyze_from_raw(raw, tmp_path, seed=0, targets=["C_hard_bugtrap"], backbones=["hrm"])
    cur = {(r["method"], int(r["K"])): float(r["exp_ratio_median"])
           for r in csv.DictReader(open(out["curves"], newline="")) if r["exp_ratio_median"]}
    assert cur[("lora", 4)] < cur[("scratch", 4)]   # 0.4 < 0.8
    assert Path(out["comparisons"]).exists()
    assert Path(out["significance"]).exists()


@pytest.mark.skipif(not (HERE / "runs/c7_local/checkpoints/avgbase__hrm.pt").exists(), reason="base missing")
def test_run_full_smoke(tmp_path):
    import torch
    cfg = C9.C9Config(source_dir=str(HERE / "runs/c7_local"), out_dir=str(tmp_path / "c9"),
                      targets="C_hard_bugtrap", backbones="hrm", k_grid="0,1", n_adapt_seeds=1,
                      n_test=4, adapt_epochs=1, roadmap_nodes=192, roadmap_k=7, budgets="200,400",
                      w_values="1.0", cpu=True, seed=7)
    out = C9.run_full(cfg, torch.device("cpu"))
    assert Path(out["curves"]).exists() and Path(out["comparisons"]).exists() and Path(out["significance"]).exists()
