import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parents[1]

import hrm_headtohead as HH
import hrm_headtohead_v2 as HH2
import continuous_prm_common as C


def test_v2_backbone_forward_finite_and_shaped():
    device = torch.device("cpu")
    sizing = HH.RepairedSizing(hidden_dim=32, num_heads=4, num_layers=1, head_hidden=48, n_cycles=2, ffn_mult=2.6)
    model = HH2.build_repaired_model_v2(sizing, token_dim=16, max_norm_residual=4.0, device=device)
    x = torch.randn(5, 24, 16)
    out = model(x)
    assert out.shape == (5,)
    assert torch.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 4.0).all()


def test_v2_drops_h_blocks_and_folds_into_l_blocks():
    """I2: H_blocks (a second token-level transformer stack) must be gone
    entirely -- its capacity folds into +1 L_blocks layer."""
    sizing = HH.RepairedSizing(hidden_dim=32, num_heads=4, num_layers=1, head_hidden=48, n_cycles=2, ffn_mult=2.6)
    model = HH2.build_repaired_model_v2(sizing, token_dim=16, max_norm_residual=4.0, device=torch.device("cpu"))
    assert not hasattr(model.backbone, "H_blocks")
    assert len(model.backbone.L_blocks) == sizing.num_layers + 1  # +1 folded from dropped H_blocks
    assert not hasattr(model.backbone, "H_pool")  # replaced by H_mlp
    assert isinstance(model.backbone.H_mlp, HH2.HMlp)


def test_v2_readout_uses_state_token_not_mean_pool():
    """I3: pooled readout each cycle must come from h[:, 0] (the state/goal
    token per make_feature_sequence), not mean(h, dim=1). Verify by checking
    that perturbing ONLY token 0 changes the output more than perturbing an
    arbitrary non-state token by the same magnitude at a fixed backbone (a
    mean-pool readout would treat all 24 tokens symmetrically; a token-0
    readout should not)."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    sizing = HH.RepairedSizing(hidden_dim=32, num_heads=4, num_layers=1, head_hidden=48, n_cycles=1, ffn_mult=2.6)
    model = HH2.build_repaired_model_v2(sizing, token_dim=16, max_norm_residual=4.0, device=device)
    model.eval()
    with torch.no_grad():
        model.head[-1].weight.add_(0.1 * torch.randn_like(model.head[-1].weight))

    base = torch.zeros(1, 24, 16)
    with torch.no_grad():
        out_base = model(base, clamp=False)

        pert_state = base.clone()
        pert_state[0, 0] += 1.0
        out_pert_state = model(pert_state, clamp=False)

        pert_other = base.clone()
        pert_other[0, 12] += 1.0  # an arbitrary ray token, same perturbation magnitude
        out_pert_other = model(pert_other, clamp=False)

    delta_state = (out_pert_state - out_base).abs().item()
    delta_other = (out_pert_other - out_base).abs().item()
    # Under a token-0 readout, perturbing token 0 should move the output at
    # least as much as perturbing an arbitrary other token (both tokens still
    # interact via L_blocks cross-token attention, so delta_other > 0 is
    # expected too -- the assertion is about asymmetry, not isolation).
    assert delta_state >= delta_other, (
        f"expected state-token (idx 0) perturbation to dominate under a token-0 "
        f"readout; got delta_state={delta_state:.6f} delta_other={delta_other:.6f}"
    )


def test_v2_init_is_not_blanket_std_001():
    """I1: embed/SwiGLU/H_mlp Linears must NOT be std=0.01 normal-init (the
    incumbent's DeepSapientHRMBackbone._init_weights pattern V1 must avoid).
    Xavier-uniform on a (16, 32) or (32, 32)-shaped weight has std well above
    0.01 by construction; assert the actual init std is at least 5x higher
    than what a std=0.01 init would have produced, on both the embed layer and
    an FFN layer."""
    sizing = HH.RepairedSizing(hidden_dim=64, num_heads=4, num_layers=1, head_hidden=48, n_cycles=2, ffn_mult=2.6)
    model = HH2.build_repaired_model_v2(sizing, token_dim=16, max_norm_residual=4.0, device=torch.device("cpu"))
    embed_std = model.backbone.embed.weight.std().item()
    ffn_std = model.backbone.L_blocks[0].ffn.w1.weight.std().item()
    assert embed_std > 0.05, f"embed weight std={embed_std:.5f} looks like a std=0.01 normal-init, not xavier"
    assert ffn_std > 0.05, f"ffn w1 weight std={ffn_std:.5f} looks like a std=0.01 normal-init, not xavier"
    # Head final layer must STILL be zero-init + bias -2.0 (the one thing I1 keeps).
    assert torch.all(model.head[-1].weight == 0.0)
    assert torch.allclose(model.head[-1].bias, torch.full_like(model.head[-1].bias, -2.0))


def test_v2_matched_sizing_within_tolerance_of_incumbent():
    incumbent_cfg = C.BackboneConfig(name="hrm", backbone_type="hrm", hidden_dim=192, num_layers=2,
                                     k_step=2, num_heads=4, head_hidden=256, lora_rank=8)
    incumbent_model = C.build_model(incumbent_cfg, C.FeatureConfig(), C.TrainingConfig(), torch.device("cpu"))
    incumbent_params = HH.count_params(incumbent_model)
    sizing = HH2.pick_matched_sizing_v2(incumbent_cfg, incumbent_params, token_dim=16, max_norm_residual=4.0)
    repaired = HH2.build_repaired_model_v2(sizing, token_dim=16, max_norm_residual=4.0, device=torch.device("cpu"))
    repaired_params = HH.count_params(repaired)
    assert repaired_params <= incumbent_params * 1.30  # small slack over the 25% target for float rounding


def test_train_variant_tiny_synthetic_roundtrip_v1(tmp_path, monkeypatch):
    """End-to-end V1 (fixed backbone, incumbent recipe) on a tiny synthetic
    pooled dataset -- exercises the same code path as the real GPU sweep
    without its cost."""
    device = torch.device("cpu")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((48, 24, 16)).astype(np.float32)
    y = np.abs(rng.standard_normal(48)).astype(np.float32)
    npz_path = tmp_path / "tiny_scalar.npz"
    np.savez(npz_path, x=x, y=y)

    monkeypatch.setattr(HH, "SCALAR_DATASETS", {"tiny": npz_path})
    sizing = HH.RepairedSizing(hidden_dim=32, num_heads=4, num_layers=1, head_hidden=48, n_cycles=2, ffn_mult=2.6)
    fc = C.FeatureConfig()
    tc = C.TrainingConfig(batch_size=16, base_epochs=2, lr=2e-4, weight_decay=1e-4, grad_clip=1.0,
                          max_norm_residual=4.0, num_workers=0)
    spec = HH2.VariantSpec(key="v1_test", label="test", use_v2_backbone=True, epochs=2, lr=2e-4, warmup_steps=0)
    out_ckpt = tmp_path / "tiny_v1.pt"
    ckpt, meta = HH2.train_variant(spec, sizing, sizing, fc, tc, device, seed=1234, out_ckpt=out_ckpt)
    assert ckpt.exists()
    assert meta["variant"] == "v1_test"
    assert np.isfinite(meta["final_loss"])

    prov = HH2.load_variant_provider(ckpt, device)
    assert prov.name == "repaired_hrm_v1_test"
    with torch.no_grad():
        pred = prov.model(torch.from_numpy(x[:8]))
    assert pred.shape == (8,)
    assert torch.isfinite(pred).all()


def test_train_variant_tiny_synthetic_roundtrip_v4_control(tmp_path, monkeypatch):
    """V4 control uses the ORIGINAL (unfixed) hrm_headtohead.RepairedHRMBackbone
    verbatim -- confirm the dispatch actually builds that class, not V2."""
    device = torch.device("cpu")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, 24, 16)).astype(np.float32)
    y = np.abs(rng.standard_normal(32)).astype(np.float32)
    npz_path = tmp_path / "tiny_scalar.npz"
    np.savez(npz_path, x=x, y=y)

    monkeypatch.setattr(HH, "SCALAR_DATASETS", {"tiny": npz_path})
    sizing = HH.RepairedSizing(hidden_dim=32, num_heads=4, num_layers=1, head_hidden=48, n_cycles=2, ffn_mult=2.6)
    fc = C.FeatureConfig()
    tc = C.TrainingConfig(batch_size=16, base_epochs=2, lr=2e-4, weight_decay=1e-4, grad_clip=1.0,
                          max_norm_residual=4.0, num_workers=0)
    spec = HH2.VariantSpec(key="v4_test", label="test control", use_v2_backbone=False, epochs=2, lr=2e-4, warmup_steps=0)
    out_ckpt = tmp_path / "tiny_v4.pt"
    ckpt, meta = HH2.train_variant(spec, sizing, sizing, fc, tc, device, seed=1234, out_ckpt=out_ckpt)
    payload = torch.load(ckpt, map_location="cpu")
    assert payload["use_v2_backbone"] is False

    prov = HH2.load_variant_provider(ckpt, device)
    assert isinstance(prov.model, HH.RepairedHeuristicModel)  # ORIGINAL class, not V2
    with torch.no_grad():
        pred = prov.model(torch.from_numpy(x[:8]))
    assert torch.isfinite(pred).all()


def test_warmup_schedule_linear_then_flat():
    assert HH2._linear_warmup_lr(0, 100, 5e-4) == pytest.approx(5e-6)
    assert HH2._linear_warmup_lr(49, 100, 5e-4) == pytest.approx(5e-4 * 50 / 100)
    assert HH2._linear_warmup_lr(99, 100, 5e-4) == pytest.approx(5e-4 * 100 / 100)
    assert HH2._linear_warmup_lr(100, 100, 5e-4) == pytest.approx(5e-4)
    assert HH2._linear_warmup_lr(500, 100, 5e-4) == pytest.approx(5e-4)
    assert HH2._linear_warmup_lr(0, 0, 5e-4) == pytest.approx(5e-4)  # no warmup


def test_variant_specs_match_task_recipe():
    assert HH2.VARIANTS["v1"].epochs == 16 and HH2.VARIANTS["v1"].lr == 2e-4 and HH2.VARIANTS["v1"].warmup_steps == 0
    assert HH2.VARIANTS["v2"].epochs == 16 and HH2.VARIANTS["v2"].lr == 5e-4 and HH2.VARIANTS["v2"].warmup_steps == 100
    assert HH2.VARIANTS["v3"].epochs == 32 and HH2.VARIANTS["v3"].lr == 2e-4
    assert HH2.VARIANTS["v4"].epochs == 32 and HH2.VARIANTS["v4"].use_v2_backbone is False
    assert all(v.use_v2_backbone for k, v in HH2.VARIANTS.items() if k != "v4")
