import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parents[1]
import numpy as np
import continuous_prm_c10_interp as C10
import continuous_prm_common as C
import continuous_prm_c9_transfer as C9
import continuous_prm_c9h_transfer as C9H


def test_family_grid_specs_and_bracketing():
    specs = C10.c10_family_specs()
    for fam in C10.SOURCE_FAMILIES + C10.TARGET_FAMILIES:
        assert fam in specs
        s = specs[fam]
        assert s.mode == C10.HARD_MODE and s.name in ("C_hard_maze", "C_hard_rooms")
    src_cent = {f: C10.family_descriptor_centroid(specs[f], n=6, seed=1) for f in C10.SOURCE_FAMILIES}
    for tgt in C10.TARGET_FAMILIES:
        z_t = C10.family_descriptor_centroid(specs[tgt], n=6, seed=2)
        ok, viol = C10.bracketing_ok(z_t, list(src_cent.values()))
        assert isinstance(ok, bool) and isinstance(viol, list)
    z = C10.family_descriptor_centroid(specs["C10_maze_tgt"], n=8, seed=3)
    ok, viol = C10.bracketing_ok(z, [C10.family_descriptor_centroid(specs[f], n=8, seed=3) for f in C10.SOURCE_FAMILIES])
    assert isinstance(ok, bool)


def test_rbf_weights():
    cent = [np.array([0.,0.]), np.array([1.,0.]), np.array([0.,1.])]
    z = np.array([0.1, 0.1])
    w = C10.rbf_weights(z, cent, sigma=1.0)
    assert abs(float(w.sum()) - 1.0) < 1e-6 and w.shape == (3,)
    assert int(np.argmax(w)) == 0  # nearest centroid gets most weight
    w0 = C10.rbf_weights(z, cent, sigma=1e-4)  # sigma->0 => one-hot on nearest
    assert w0[0] > 0.99
    assert C10.nearest_weights(z, cent).tolist() == [1.0, 0.0, 0.0]
    u = C10.uniform_weights(3)
    assert np.allclose(u, 1/3)


@pytest.mark.skipif(not (HERE/"runs/c7_local/checkpoints/avgbase__hrm.pt").exists(), reason="base missing")
def test_train_source_experts_smoke(tmp_path):
    import torch
    cfg = C10.C10Config(source_dir=str(HERE/"runs/c7_local"), out_dir=str(tmp_path/"c10"),
                        backbones="hrm", n_src_worlds=2, n_centroid_worlds=4, epochs=1,
                        roadmap_nodes=192, roadmap_k=7, cpu=True, seed=7)
    man = C10.train_source_experts(cfg, torch.device("cpu"),
                                   only_families=["C10_maze_d1", "C10_rooms_s20"])
    assert len(man["experts"]) == 2
    for e in man["experts"]:
        assert Path(e["ckpt"]).exists() and "centroid" in e and len(e["centroid"]) == 8


@pytest.mark.skipif(not (HERE/"runs/c7_local/checkpoints/avgbase__hrm.pt").exists(), reason="base missing")
def test_weight_merge_baker(tmp_path):
    import torch, numpy as np
    dev = torch.device("cpu")
    base = C9.load_source_base(HERE/"runs/c7_local", "hrm", dev)
    def tiny(seed):
        n = 16
        x = np.random.RandomState(seed).randn(n, base.feature_cfg.seq_len, base.feature_cfg.token_dim).astype("float32")
        y = np.abs(np.random.RandomState(seed + 1).randn(n)).astype("float32")
        p = tmp_path / f"t{seed}.npz"
        np.savez_compressed(p, x=x, y=y, euclid=np.ones(n, "float32"), side=np.ones(n, "float32"))
        return p
    import dataclasses as dc
    tcfg = dc.replace(base.train_cfg, base_epochs=1, lr=2e-4)
    e0 = C9H.train_scalar_lora(base.backbone_cfg, tiny(0), tmp_path/"e0.pt", base.feature_cfg, tcfg, dev, seed=0, init_ckpt=base.ckpt_path, rank=8, alpha=1.0, bounded=True)
    e1 = C9H.train_scalar_lora(base.backbone_cfg, tiny(5), tmp_path/"e1.pt", base.feature_cfg, tcfg, dev, seed=0, init_ckpt=base.ckpt_path, rank=8, alpha=1.0, bounded=True)
    merged = C10.bake_weight_merge(base.ckpt_path, [e0, e1], np.array([1.0, 0.0]), tmp_path/"m.pt", dev)
    pm = C9H.load_scalar_provider_c9h(merged, dev)
    pe = C9H.load_scalar_provider_c9h(e0, dev)
    xb = torch.randn(4, base.feature_cfg.seq_len, base.feature_cfg.token_dim)
    with torch.no_grad():
        import numpy as _np
        a = pm.model(xb).detach().numpy()
        b = pe.model(xb).detach().numpy()
    assert _np.allclose(a, b, atol=1e-4)
    # k=1 branch: w=[0,1] reproduces expert-1's forward (exercises the accumulation path)
    m1 = C10.bake_weight_merge(base.ckpt_path, [e0, e1], np.array([0.0, 1.0]), tmp_path/"m1.pt", dev)
    p1 = C9H.load_scalar_provider_c9h(m1, dev); pe1 = C9H.load_scalar_provider_c9h(e1, dev)
    with torch.no_grad():
        assert _np.allclose(p1.model(xb).detach().numpy(), pe1.model(xb).detach().numpy(), atol=1e-4)
    # uniform bake == base + 0.5·Δ0 + 0.5·Δ1 on a sample target (weight-space; forward is nonlinear in weights)
    mh = C10.bake_weight_merge(base.ckpt_path, [e0, e1], np.array([0.5, 0.5]), tmp_path/"mh.pt", dev)
    def _delta(sd, key):
        A = sd[f"{key}.parametrizations.weight.0.A"]; B = sd[f"{key}.parametrizations.weight.0.B"]
        return float(sd[f"{key}.parametrizations.weight.0.adapter_scale"]) * (B @ A)
    ms = torch.load(mh, map_location="cpu")["model"]; s0 = torch.load(e0, map_location="cpu")["model"]; s1 = torch.load(e1, map_location="cpu")["model"]; bs = torch.load(base.ckpt_path, map_location="cpu")["model"]
    k = "backbone.L_blocks.0.ffn.w1"
    assert torch.allclose(ms[k + ".weight"], bs[k + ".weight"] + 0.5 * _delta(s0, k) + 0.5 * _delta(s1, k), atol=1e-5)
