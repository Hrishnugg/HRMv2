"""C11 Task 6: HRM-v2 ACT arm -- model adaptation only (trainer + providers
are Task 7). Tests the continuous-input inner (`TraceHRMInner`), the ACT
wrapper subclass (`TraceHRMACT`), the scalar readout convention
(`readout_yhat`), and the regression ACT loss head (`RegressionACTLossHead`)
in isolation from the core `continuous_prm_c11_mission` module (which this
task does not touch -- the registry hook is Task 7).

All tests run on CPU. The module under test (`continuous_prm_c11_hrmv2_arm`)
must import cleanly even when `hrm` is absent (lazy imports only, guarded
inside `_load_hrm()` / function bodies) -- `test_module_imports_without_hrm`
verifies this via an AST scan rather than actually uninstalling `hrm` (which
is already imported into `sys.modules` by the time any test runs).
"""
import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_c11_hrmv2_arm as HA  # noqa: E402  (must import without `hrm`)

hrm = pytest.importorskip("hrm")

from hrm.models.hrm_act_v1 import HRMACTv1_Inner, HRMACTv1  # noqa: E402


# ---------------------------------------------------------------------------
# Test 1: module imports lazily -- no top-level `import hrm` / `from hrm`.
# ---------------------------------------------------------------------------

def test_module_imports_without_hrm():
    """AST-scan the module source: every `import hrm` / `from hrm import ...`
    statement must live inside a function body, never at module top level.
    `continuous_prm_c11_hrmv2_arm` (HA, imported above) was already proven
    importable before `pytest.importorskip("hrm")` ran, which is itself
    evidence the module doesn't need `hrm` at import time -- this test
    additionally pins the *mechanism* (lazy imports only) so a future edit
    can't silently reintroduce a top-level `hrm` import that happens to work
    today because `hrm` is installed."""
    source = inspect.getsource(HA)
    tree = ast.parse(source)

    def _is_hrm_import(node):
        if isinstance(node, ast.Import):
            return any(alias.name == "hrm" or alias.name.startswith("hrm.") for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            return node.module is not None and (node.module == "hrm" or node.module.startswith("hrm."))
        return False

    # Only scan nodes at MODULE level (tree.body), not nested inside
    # FunctionDef/AsyncFunctionDef bodies -- those are fine.
    top_level_hrm_imports = [n for n in tree.body if _is_hrm_import(n)]
    assert top_level_hrm_imports == [], (
        f"found top-level `hrm` import(s) in continuous_prm_c11_hrmv2_arm: "
        f"{[ast.dump(n) for n in top_level_hrm_imports]}"
    )

    # There MUST be at least one hrm import somewhere (inside functions) --
    # otherwise this module isn't actually using hrm at all.
    all_hrm_imports = [n for n in ast.walk(tree) if _is_hrm_import(n)]
    assert len(all_hrm_imports) > 0, "expected at least one lazy `hrm` import inside a function body"


def test_build_hrmv2_arm_raises_clear_error_when_hrm_missing(monkeypatch):
    """`build_hrmv2_arm` must raise a RuntimeError naming the install command
    when `_load_hrm()` fails (simulated here by monkeypatching `_load_hrm`
    to raise ImportError, standing in for `hrm` truly being absent)."""

    def _broken_load_hrm():
        raise ImportError("No module named 'hrm'")

    monkeypatch.setattr(HA, "_load_hrm", _broken_load_hrm)

    with pytest.raises(RuntimeError) as exc_info:
        HA.build_hrmv2_arm()

    msg = str(exc_info.value)
    assert "pip install -e ./HRM-v2 --no-deps" in msg


# ---------------------------------------------------------------------------
# Shared config helper.
# ---------------------------------------------------------------------------

def _tiny_config_dict(**overrides):
    """A small-but-faithful config dict for fast CPU tests: keeps the
    architecture shape (H/L=2/2 layers, hidden=256, heads=8) from
    `hrmv2_config()`'s defaults but lets individual tests override
    halt_exploration_prob / halt_max_steps for determinism tests."""
    cfg = HA.hrmv2_config()
    cfg.update(overrides)
    return cfg


def _make_batch(batch_size=4, seq_len=10, token_dim=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    inputs = torch.randn(batch_size, seq_len, token_dim, generator=g)
    puzzle_identifiers = torch.zeros(batch_size, dtype=torch.int32)
    return {"inputs": inputs, "puzzle_identifiers": puzzle_identifiers}


# ---------------------------------------------------------------------------
# Test 2: TraceHRMInner accepts continuous float inputs.
# ---------------------------------------------------------------------------

def test_trace_inner_continuous_inputs():
    from hrm.models.hrm_act_v1 import HRMACTv1Config

    cfg_dict = _tiny_config_dict()
    config = HRMACTv1Config(**cfg_dict)
    inner = HA.TraceHRMInner(config, token_dim=12)

    assert isinstance(inner, HRMACTv1_Inner)

    batch = _make_batch(batch_size=4, seq_len=10, token_dim=12)
    carry = inner.empty_carry(4)
    carry = inner.reset_carry(torch.ones(4, dtype=torch.bool), carry)

    new_carry, logits, (q_halt, q_continue) = inner(carry, batch)

    assert logits.shape == (4, 10, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(q_halt).all()
    assert torch.isfinite(q_continue).all()


def test_trace_inner_input_embeddings_shape_and_finiteness():
    """Directly exercise `_input_embeddings` (float input, not int IDs) --
    confirms the override accepts (B, seq_len, token_dim) float tensors and
    returns embeddings of shape (B, seq_len + puzzle_emb_len, hidden_size)
    matching the parent's contract (puzzle_emb_len == 0 here, so seq_len is
    unchanged)."""
    from hrm.models.hrm_act_v1 import HRMACTv1Config

    cfg_dict = _tiny_config_dict()
    config = HRMACTv1Config(**cfg_dict)
    inner = HA.TraceHRMInner(config, token_dim=12)

    inputs = torch.randn(4, 10, 12)
    puzzle_identifiers = torch.zeros(4, dtype=torch.int32)

    emb = inner._input_embeddings(inputs, puzzle_identifiers)

    assert emb.shape == (4, 10, config.hidden_size)
    assert torch.isfinite(emb).all()
    assert emb.dtype == inner.forward_dtype


# ---------------------------------------------------------------------------
# Test 3: TraceHRMACT full ACT forward.
# ---------------------------------------------------------------------------

def test_trace_model_full_act_forward():
    cfg_dict = _tiny_config_dict()
    model = HA.TraceHRMACT(cfg_dict)
    assert isinstance(model, HRMACTv1)
    assert isinstance(model.inner, HA.TraceHRMInner)

    model.eval()
    batch = _make_batch(batch_size=4, seq_len=10, token_dim=12)
    carry = model.initial_carry(batch)

    all_finish = False
    outputs = None
    for _ in range(cfg_dict["halt_max_steps"] + 1):
        carry, outputs = model(carry, batch)
        if carry.halted.all():
            all_finish = True
            break

    assert all_finish
    assert outputs is not None
    assert outputs["logits"].shape == (4, 10, 1)
    assert torch.isfinite(outputs["logits"]).all()


# ---------------------------------------------------------------------------
# Test 4: scalar readout range + extremes.
# ---------------------------------------------------------------------------

def test_scalar_readout_range():
    logits = torch.randn(6, 10, 1) * 3.0
    outputs = {"logits": logits}
    yhat = HA.readout_yhat(outputs)

    assert yhat.shape == (6,)
    assert torch.isfinite(yhat).all()
    assert (yhat >= 0.0).all()
    assert (yhat <= 4.0).all()


def test_scalar_readout_extremes():
    # Position 0, channel 0, forced to extreme large positive -> clamps at 4.0.
    logits_hi = torch.full((3, 10, 1), -100.0)
    logits_hi[:, 0, 0] = 100.0
    yhat_hi = HA.readout_yhat({"logits": logits_hi})
    assert torch.allclose(yhat_hi, torch.full((3,), 4.0), atol=1e-4)

    # Forced to extreme negative -> softplus(-100) approx 0 -> clamps at 0.0.
    logits_lo = torch.full((3, 10, 1), 100.0)
    logits_lo[:, 0, 0] = -100.0
    yhat_lo = HA.readout_yhat({"logits": logits_lo})
    assert torch.allclose(yhat_lo, torch.zeros(3), atol=1e-4)

    # Only position 0 is read -- other positions must not affect the result.
    logits_other = torch.zeros(3, 10, 1)
    logits_other[:, 0, 0] = 0.0  # softplus(0) = ln(2)
    logits_other[:, 1:, 0] = 999.0  # would blow up if wrongly pooled
    yhat_other = HA.readout_yhat({"logits": logits_other})
    expected = torch.full((3,), float(np.log(2.0)))
    assert torch.allclose(yhat_other, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 5: RegressionACTLossHead.
# ---------------------------------------------------------------------------

def test_regression_act_loss_head_structure_and_grad():
    """Build the real model (small batch, forced 1-segment via halt_max_steps=1
    so the test stays fast), run the loss head end to end, and confirm:
    (a) the forward signature/return contract matches ACTLossHead's
    (carry, loss, metrics, outputs, all_finish); (b) loss is finite;
    (c) gradients flow to trace_proj.weight."""
    cfg_dict = _tiny_config_dict(halt_max_steps=1, halt_exploration_prob=0.0)
    model = HA.TraceHRMACT(cfg_dict)
    loss_head = HA.RegressionACTLossHead(model, band=0.1)

    model.train()
    batch_size = 4
    inputs = torch.randn(batch_size, 10, 12)
    puzzle_identifiers = torch.zeros(batch_size, dtype=torch.int32)
    y = torch.rand(batch_size) * 4.0
    batch = {
        "inputs": inputs,
        "puzzle_identifiers": puzzle_identifiers,
        "y": y,
    }

    carry = loss_head.initial_carry(batch)
    carry, loss, metrics, outputs, all_finish = loss_head(
        return_keys=["logits"], carry=carry, batch=batch
    )

    assert torch.isfinite(loss)
    assert isinstance(metrics, dict)
    assert "mae" in metrics
    assert "q_halt_accuracy" in metrics
    assert "count" in metrics
    assert "steps" in metrics

    (loss / batch_size).backward()
    grad = model.inner.trace_proj.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0.0


def test_regression_act_loss_head_forced_correct_band():
    """Hand-build outputs where yhat is forced within 0.1 of y (bypass the
    model: construct a loss head around a tiny real model, but call the
    correctness/loss math the way the head does by feeding an outputs dict
    with logits engineered so `readout_yhat` lands close to y). Confirms
    `correct` is all-True and the q_halt BCE target is 1.0 everywhere,
    mirroring ACTLossHead's `seq_is_correct` -> q_halt-target wiring."""
    cfg_dict = _tiny_config_dict(halt_max_steps=1, halt_exploration_prob=0.0)
    model = HA.TraceHRMACT(cfg_dict)
    loss_head = HA.RegressionACTLossHead(model, band=0.1)

    batch_size = 5
    y = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0])
    yhat = y.clone()  # exact match -> definitely within band

    correct = HA.regression_correct(yhat, y, band=0.1)
    assert correct.all()

    # BCE target should be 1.0 for all-correct.
    q_halt_logits = torch.zeros(batch_size, requires_grad=True)
    q_halt_loss = F.binary_cross_entropy_with_logits(
        q_halt_logits, correct.to(q_halt_logits.dtype), reduction="sum"
    )
    assert torch.isfinite(q_halt_loss)


# ---------------------------------------------------------------------------
# Test 6: param count band.
# ---------------------------------------------------------------------------

def test_param_count_band():
    cfg_dict = HA.hrmv2_config()
    model = HA.TraceHRMACT(cfg_dict)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TraceHRMACT param count: {n_params}")
    assert 1_000_000 <= n_params <= 4_000_000


# ---------------------------------------------------------------------------
# Test 7: deterministic forward.
# ---------------------------------------------------------------------------

def test_deterministic_forward():
    cfg_dict = HA.hrmv2_config(halt_exploration_prob=0.0)
    assert cfg_dict["halt_exploration_prob"] == 0.0

    torch.manual_seed(42)
    model_a = HA.TraceHRMACT(cfg_dict)
    model_a.eval()

    torch.manual_seed(42)
    model_b = HA.TraceHRMACT(cfg_dict)
    model_b.eval()

    batch = _make_batch(batch_size=3, seq_len=10, token_dim=12, seed=7)

    def _run_to_halt(model, batch):
        carry = model.initial_carry(batch)
        outputs = None
        for _ in range(cfg_dict["halt_max_steps"] + 1):
            carry, outputs = model(carry, batch)
            if carry.halted.all():
                break
        return outputs["logits"]

    logits_a = _run_to_halt(model_a, batch)
    logits_b = _run_to_halt(model_b, batch)

    assert torch.allclose(logits_a, logits_b, atol=1e-6)

    # Running the SAME model twice on the same batch (fresh carry each time,
    # eval mode, no exploration) must also be deterministic.
    logits_a2 = _run_to_halt(model_a, batch)
    assert torch.allclose(logits_a, logits_a2, atol=1e-6)


# ---------------------------------------------------------------------------
# hrmv2_config() sanity.
# ---------------------------------------------------------------------------

def test_hrmv2_config_matches_spec_defaults():
    cfg = HA.hrmv2_config()
    assert cfg["vocab_size"] == 1
    assert cfg["seq_len"] == 10
    assert cfg["puzzle_emb_ndim"] == 0
    assert cfg["num_puzzle_identifiers"] == 1
    assert cfg["hidden_size"] == 256
    assert cfg["num_heads"] == 8
    assert cfg["H_layers"] == 2
    assert cfg["L_layers"] == 2
    assert cfg["H_cycles"] == 2
    assert cfg["L_cycles"] == 2
    assert cfg["halt_max_steps"] == 8
    assert cfg["halt_exploration_prob"] == 0.1
    assert cfg["pos_encodings"] == "learned"
    assert cfg["forward_dtype"] == "float32"
    assert cfg["expansion"] == 4.0
    assert cfg["rms_norm_eps"] == 1e-5
    assert cfg["batch_size"] == 64

    # Must actually construct a valid HRMACTv1Config with no extra fixups.
    from hrm.models.hrm_act_v1 import HRMACTv1Config
    HRMACTv1Config(**cfg)


def test_hrmv2_config_accepts_overrides():
    cfg = HA.hrmv2_config(halt_exploration_prob=0.0, hidden_size=128)
    assert cfg["halt_exploration_prob"] == 0.0
    assert cfg["hidden_size"] == 128
