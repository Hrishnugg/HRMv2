import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parents[1]
import continuous_prm_c9h_transfer as C9H
import continuous_prm_c6_heatmap_value_field as C6
import continuous_prm_common as C


def test_conv_lora_identity_then_frozen_base():
    import torch
    torch.manual_seed(0)
    unet = C6.build_model("unet", in_channels=8)
    x = torch.randn(1, 8, 64, 64)
    with torch.no_grad():
        base_out = unet(x).clone()
    n = C9H.apply_conv_lora(unet, rank=4, alpha=1.0)
    assert n >= 10
    with torch.no_grad():
        lora_out = unet(x)
    assert torch.allclose(base_out, lora_out, atol=1e-5)  # B=0 => identity at init
    C.set_lora_trainable(unet)
    trainable = [nm for nm, p in unet.named_parameters() if p.requires_grad]
    assert trainable and all((".A" in nm or ".B" in nm) for nm in trainable)


def test_conv_lora_changes_output_after_step():
    import torch
    unet = C6.build_model("unet", in_channels=8)
    C9H.apply_conv_lora(unet, rank=4, alpha=1.0)
    C.set_lora_trainable(unet)
    x = torch.randn(1, 8, 64, 64)
    base_out = unet(x).detach().clone()
    opt = torch.optim.SGD([p for p in unet.parameters() if p.requires_grad], lr=1.0)
    loss = unet(x).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    assert not torch.allclose(base_out, unet(x), atol=1e-6)


def test_c9hconfig_defaults():
    cfg = C9H.C9hConfig()
    assert cfg.backbones == "hrm,onlstm,unet"
    assert cfg.methods == "lora_bounded,lora_unbounded,full_ft,scratch"
    assert cfg.k_grid == "1,4,16"
    assert cfg.n_adapt_seeds == 3
    assert cfg.epochs == 10 and abs(cfg.lr - 2e-4) < 1e-12
    assert cfg.source_dir.endswith("c7_local")


import dataclasses
import continuous_prm_c9_transfer as C9


@pytest.mark.skipif(not (HERE/"runs/c7_local/checkpoints/avgbase__hrm.pt").exists(), reason="base missing")
def test_scalar_lora_bounded_vs_unbounded(tmp_path):
    import torch, numpy as np
    dev = torch.device("cpu")
    base = C9.load_source_base(HERE/"runs/c7_local", "hrm", dev)
    n=24
    x=np.random.RandomState(0).randn(n, base.feature_cfg.seq_len, base.feature_cfg.token_dim).astype("float32")
    y=np.abs(np.random.RandomState(1).randn(n)).astype("float32")
    npz=tmp_path/"t.npz"; np.savez_compressed(npz, x=x, y=y, euclid=np.ones(n,"float32"), side=np.ones(n,"float32"))
    tcfg = dataclasses.replace(base.train_cfg, base_epochs=2, lr=2e-4)
    ckb = C9H.train_scalar_lora(base.backbone_cfg, npz, tmp_path/"b.pt", base.feature_cfg, tcfg, dev, seed=0,
                                init_ckpt=base.ckpt_path, rank=8, alpha=1.0, bounded=True)
    cku = C9H.train_scalar_lora(base.backbone_cfg, npz, tmp_path/"u.pt", base.feature_cfg, tcfg, dev, seed=0,
                                init_ckpt=base.ckpt_path, rank=8, alpha=1.0, bounded=False)
    pb = torch.load(ckb, map_location="cpu"); pu = torch.load(cku, map_location="cpu")
    assert pb["bounded"] is True and pu["bounded"] is False
    assert "lora_rank" in pb and pb["max_norm_residual"] != pu["max_norm_residual"]
    # loader round-trips both
    provb = C9H.load_scalar_provider_c9h(ckb, dev)
    assert provb.name == "scalar_hrm"


import continuous_prm_c7_hard_maps as H7


@pytest.mark.skipif(not (HERE/"runs/c7_local/checkpoints/c6_heatmap__unet.pt").exists(), reason="field base missing")
def test_field_train_and_provider(tmp_path):
    import torch, numpy as np
    dev = torch.device("cpu")
    H7.install_c7_hard_maps(); specs = C.build_anchor_specs()
    c6cfg = C9H._c6_cfg(C9H.C9hConfig(epochs=1, grid_size=64))
    npz = C9H.collect_field_adapt(specs["C_hard_bugtrap"], tmp_path/"ds", "adapt_t", 2, c6cfg, seed=7)
    base = HERE/"runs/c7_local/checkpoints/c6_heatmap__unet.pt"
    ck_ft = C9H.train_field_model("unet", [npz], tmp_path/"ft.pt", c6cfg, dev, seed=0, init_ckpt=base, lora_rank=0, alpha=1.0)
    ck_lo = C9H.train_field_model("unet", [npz], tmp_path/"lo.pt", c6cfg, dev, seed=0, init_ckpt=base, lora_rank=8, alpha=1.0)
    rmcfg = C.RoadmapConfig(n_nodes=64, k_neighbors=7)
    w = None; rm = None
    for s in (11, 7, 13, 24, 31, 5):
        w = C.build_world(specs["C_hard_bugtrap"], s, rmcfg.min_start_goal_dist_frac)
        if w is None: continue
        rm = C.build_prm(w, rmcfg, seed=s+17)
        if rm is not None and rm.connected_to_goal[0]: break
    assert rm is not None and rm.connected_to_goal[0]
    for ck in (ck_ft, ck_lo):
        prov = C9H.load_field_provider(ck, dev, grid_size=64, bounded=True)
        h = prov.node_h(w, rm, goal_idx=1)
        assert np.isfinite(h).all() and h.shape[0] == rm.points.shape[0]


@pytest.mark.skipif(not (HERE/"runs/c7_local/checkpoints/avgbase__hrm.pt").exists(), reason="base missing")
def test_run_adapt_smoke(tmp_path):
    import torch
    cfg = C9H.C9hConfig(source_dir=str(HERE/"runs/c7_local"), out_dir=str(tmp_path/"c9h"),
                        targets="C_hard_bugtrap", backbones="hrm,unet",
                        methods="lora_bounded,lora_unbounded,full_ft,scratch",
                        k_grid="1", n_adapt_seeds=1, n_test=4, epochs=1,
                        roadmap_nodes=192, roadmap_k=7, cpu=True, seed=7)
    man = C9H.run_adapt(cfg, torch.device("cpu"))
    got = {(a["backbone"], a["method"]) for a in man["arms"]}
    assert ("hrm", "lora_bounded") in got and ("hrm", "lora_unbounded") in got
    assert ("unet", "lora_bounded") in got and ("unet", "full_ft") in got
    # bounded flag set correctly
    assert all(a["bounded"] == ("unbounded" not in a["method"]) for a in man["arms"])
    for a in man["arms"]:
        assert Path(a["ckpt"]).exists()


@pytest.mark.skipif(not (HERE/"runs/c7_local/checkpoints/avgbase__hrm.pt").exists(), reason="base missing")
def test_run_eval_smoke(tmp_path):
    import torch, csv
    cfg = C9H.C9hConfig(source_dir=str(HERE/"runs/c7_local"), out_dir=str(tmp_path/"c9h"),
                        targets="C_hard_bugtrap", backbones="hrm,unet",
                        methods="lora_bounded,full_ft,scratch", k_grid="1", n_adapt_seeds=1,
                        n_test=4, epochs=1, budgets="200,400", w_values="1.0", cpu=True, seed=7)
    dev = torch.device("cpu"); C9H.run_adapt(cfg, dev); raw = C9H.run_eval(cfg, dev)
    provs = {r["provider"] for r in csv.DictReader(open(raw, newline=""))}
    assert "euclid" in provs and "oracle" in provs
    assert any(p.startswith("zeroshot_hrm") for p in provs)
    assert any(p.startswith("zeroshot_unet") for p in provs)
    assert any(p.startswith("lora_bounded_unet") for p in provs)
    assert any(p.startswith("full_ft_hrm") for p in provs)


def _c9h_synth_raw(path):
    import csv
    rows = []
    def row(provider, method, K, wi, exp, backbone="unet", found=True):
        return dict(target="C_hard_bugtrap", K=K, seed=0, method=method, backbone=backbone,
                    suite="C_hard_bugtrap", world_index=wi, provider=provider, mode="astar",
                    w="", budget=400, found=found, expansions=exp, closed=exp, cost=1.0,
                    optimal=1.0, suboptimality=1.0, nonfinite=0)
    for wi in (0, 1):
        rows.append(row("euclid", "euclid", -1, wi, 100, backbone=""))
        rows.append(row("oracle", "oracle", -1, wi, 20, backbone=""))
        rows.append(row("lora_bounded_unet_K4_s0", "lora_bounded", 4, wi, 40))
        rows.append(row("lora_unbounded_unet_K4_s0", "lora_unbounded", 4, wi, 70))
        rows.append(row("full_ft_unet_K4_s0", "full_ft", 4, wi, 50))
        rows.append(row("scratch_unet_K4_s0", "scratch", 4, wi, 90))
        rows.append(row("zeroshot_unet", "zero_shot", 0, wi, 60))
    import continuous_prm_c9_transfer as C9
    cols = C9.RAW_COLS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in cols})


def test_analyze_c9h_curves_and_methods(tmp_path):
    import csv
    raw = tmp_path / "raw.csv"; _c9h_synth_raw(raw)
    out = C9H.analyze_from_raw_c9h(raw, tmp_path, seed=0, targets=["C_hard_bugtrap"],
                                   backbones=["unet"], methods=["lora_bounded", "lora_unbounded", "full_ft", "scratch"])
    rowsout = list(csv.DictReader(open(out["curves"], newline="")))
    cur = {(r["method"], int(r["K"])): float(r["exp_ratio_median"]) for r in rowsout if r["exp_ratio_median"]}
    assert cur[("lora_bounded", 4)] < cur[("lora_unbounded", 4)]  # 0.4 < 0.7
    assert cur[("full_ft", 4)] < cur[("lora_unbounded", 4)]       # 0.5 < 0.7
    assert Path(out["comparisons"]).exists() and Path(out["significance"]).exists()
    # bounded-vs-unbounded section present
    assert "bounded" in Path(out["comparisons"]).read_text().lower()
