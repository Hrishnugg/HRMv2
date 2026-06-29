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
