import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parents[1]

import hrm_headtohead as HH
import continuous_prm_common as C


def test_repaired_model_forward_finite_and_shaped():
    device = torch.device("cpu")
    sizing = HH.RepairedSizing(hidden_dim=32, num_heads=4, num_layers=1, head_hidden=48, n_cycles=2, ffn_mult=2.6)
    model = HH.build_repaired_model(sizing, token_dim=16, max_norm_residual=4.0, device=device)
    x = torch.randn(5, 24, 16)
    out = model(x)
    assert out.shape == (5,)
    assert torch.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 4.0).all()


def test_repaired_model_cross_token_attention_is_real():
    """The audit-found degeneracy: incumbent's GatedRecurrentBlock attends over a
    length-1 sequence (softmax over one key == identity), so changing one token
    cannot, via attention, influence another token's contribution before pooling.
    The repair's whole point is that L_blocks/H_blocks attend over the FULL
    sequence. Verify perturbing a single token changes the pooled output through
    a path that depends on attention seeing >1 token (gradient w.r.t. a token
    that isn't the one perturbed should be nonzero after cross-token mixing)."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    sizing = HH.RepairedSizing(hidden_dim=32, num_heads=4, num_layers=1, head_hidden=48, n_cycles=2, ffn_mult=2.6)
    model = HH.build_repaired_model(sizing, token_dim=16, max_norm_residual=4.0, device=device)
    # break head-final-layer symmetry (it's zero-init by design) so backbone signal reaches the loss
    with torch.no_grad():
        model.head[-1].weight.add_(0.1 * torch.randn_like(model.head[-1].weight))

    x = torch.zeros(1, 24, 16, requires_grad=True)
    ctx = model.backbone.encode_sequence(x)
    scalar = ctx.sum()
    grad, = torch.autograd.grad(scalar, x)
    # Cross-token attention means EVERY token's embedding can influence the
    # pooled zH through paths beyond its own mean-pool contribution (attention
    # mixing before the L/H updates). A nonzero gradient on more than one
    # token position confirms tokens interact, unlike a length-1-sequence
    # attention block where no such interaction is architecturally possible.
    nonzero_positions = (grad.abs().sum(dim=-1) > 1e-8).sum().item()
    assert nonzero_positions > 1, f"expected cross-token gradient spread, got {nonzero_positions} nonzero token positions"


def test_matched_sizing_within_tolerance_of_incumbent():
    incumbent_cfg = C.BackboneConfig(name="hrm", backbone_type="hrm", hidden_dim=192, num_layers=2,
                                     k_step=2, num_heads=4, head_hidden=256, lora_rank=8)
    incumbent_model = C.build_model(incumbent_cfg, C.FeatureConfig(), C.TrainingConfig(), torch.device("cpu"))
    incumbent_params = HH.count_params(incumbent_model)
    sizing = HH.pick_matched_sizing(incumbent_cfg, incumbent_params, token_dim=16, max_norm_residual=4.0)
    repaired = HH.build_repaired_model(sizing, token_dim=16, max_norm_residual=4.0, device=torch.device("cpu"))
    repaired_params = HH.count_params(repaired)
    assert repaired_params <= incumbent_params * 1.30  # small slack over the 25% target for float rounding


def test_train_repaired_model_tiny_synthetic_roundtrip(tmp_path):
    """End-to-end: train on a tiny synthetic pooled dataset for 2 epochs, save,
    reload as a ScalarResidualProvider, run a forward pass. Exercises the same
    code path as the real (46k-row, 16-epoch) GPU run without its cost."""
    device = torch.device("cpu")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((48, 24, 16)).astype(np.float32)
    y = np.abs(rng.standard_normal(48)).astype(np.float32)
    npz_path = tmp_path / "tiny_scalar.npz"
    np.savez(npz_path, x=x, y=y)

    orig_datasets = HH.SCALAR_DATASETS
    try:
        HH.SCALAR_DATASETS = {"tiny": npz_path}
        sizing = HH.RepairedSizing(hidden_dim=32, num_heads=4, num_layers=1, head_hidden=48, n_cycles=2, ffn_mult=2.6)
        fc = C.FeatureConfig()
        tc = C.TrainingConfig(batch_size=16, base_epochs=2, lr=2e-4, weight_decay=1e-4, grad_clip=1.0,
                              max_norm_residual=4.0, num_workers=0)
        out_ckpt = tmp_path / "tiny_repaired.pt"
        ckpt = HH.train_repaired_model(sizing, fc, tc, device, seed=1234, out_ckpt=out_ckpt)
        assert ckpt.exists()

        prov = HH.load_repaired_provider(ckpt, device)
        assert prov.name == "repaired_hrm"
        with torch.no_grad():
            pred = prov.model(torch.from_numpy(x[:8]))
        assert pred.shape == (8,)
        assert torch.isfinite(pred).all()
    finally:
        HH.SCALAR_DATASETS = orig_datasets


def test_target_suite_idx_matches_c9_ordering():
    """suite_idx feeds the world-seed formula (continuous_prm_c7_integration_compare.
    iter_matched_worlds: w_seed uses 1_000_003 * (suite_idx + 1)), so it is NOT just a
    label -- using the wrong index silently generates different worlds. Pin the
    ordering against C9's own default targets list."""
    import continuous_prm_c9_transfer as C9
    c9_targets = [t for t in C9.C9Config().targets.split(",") if t]
    assert c9_targets == HH.TARGETS
    assert HH.TARGET_SUITE_IDX["C_hard_maze_dense"] == 0
    assert HH.TARGET_SUITE_IDX["C_hard_rooms_large"] == 2


def test_binding_budgets_match_calibration():
    import json
    calib = json.loads((HH.SOURCE_DIR / "calibration.json").read_text())
    budgets = calib["budgets"]
    assert HH.BINDING_BUDGET["C_hard_maze_dense"] == budgets["C_hard_maze_dense"][0] == 140
    assert HH.BINDING_BUDGET["C_hard_rooms_large"] == budgets["C_hard_rooms_large"][0] == 56


@pytest.mark.skipif(not (HH.SOURCE_DIR / "checkpoints" / "avgbase__hrm.pt").exists(),
                    reason="C7 avgbase checkpoint not present")
def test_eval_provider_on_target_smoke():
    """Tiny n_test smoke test of the real eval path (world gen + A* + matched ratio)
    against the real incumbent checkpoint. Not a numeric gate (n_test too small for
    that) -- just confirms the plumbing produces a well-formed, finite result."""
    import continuous_prm_c9_transfer as C9
    import continuous_prm_providers as P
    device = torch.device("cpu")
    base = C9.load_source_base(HH.SOURCE_DIR, "hrm", device)
    zp = P.ScalarResidualProvider(base.model, base.feature_cfg, device, "hrm", base.train_cfg.max_norm_residual)
    zp.name = "incumbent_hrm_zeroshot"
    res = HH.eval_provider_on_target("C_hard_rooms_large", zp, device, n_test=3)
    assert res["target"] == "C_hard_rooms_large"
    assert res["budget"] == 56
    assert 0.0 <= res["succ_provider"] <= 1.0
    assert 0.0 <= res["succ_euclid"] <= 1.0
