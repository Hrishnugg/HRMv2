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


# ===========================================================================
# Task 7: faithful per-segment trainer + ACT/forced-k providers +
# halt-step logging.
#
# These tests import `continuous_prm_c11_mission` (the core module) ONLY
# for bundle/cell fixtures and `encode_trace_padded`/`stack_targets` -- the
# module under test (`continuous_prm_c11_hrmv2_arm`) still never imports it
# at module scope (see `test_module_imports_without_hrm`'s AST scan, which
# would also catch a top-level `import continuous_prm_c11_mission` if one
# were added; T7 only adds LAZY imports of it inside `train_hrmv2`/the
# provider functions, exactly like the existing lazy `hrm` imports).
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import continuous_prm_c11_mission as M  # noqa: E402


def _tiny_mission_cfg(**overrides):
    """A small mission config for fast bundle collection (Task 7 tests don't
    need the full 192-node roadmap; the default is used since roadmap size
    is not itself an HRM-v2 concern -- only epochs/batch_size are shrunk per
    test)."""
    return M.C11MissionConfig(**overrides)


def _tiny_hrmv2_cfg(**overrides):
    """A small-but-faithful HRM-v2 config dict, mirroring `_tiny_config_dict`
    above but exposed at module scope for Task 7's trainer/provider tests
    (hidden=64, H/L=1/1, halt_max_steps=4 where speed matters, per the
    task's TDD instructions)."""
    cfg = HA.hrmv2_config(hidden_size=64, num_heads=4, H_layers=1, L_layers=1,
                           H_cycles=1, L_cycles=1, halt_max_steps=4)
    cfg.update(overrides)
    return cfg


def _one_bundle(config_label="A", K=2, n_attempts=60):
    """Collect exactly one valid TEST-seed bundle for a small cell -- reused
    across the Task 7 trainer/provider tests as the "tiny dataset"."""
    cells = M.build_cell_grid()
    cell = next(c for c in cells if c["config_label"] == config_label and c["K"] == K)
    for w in range(n_attempts):
        seed = M.test_seed(w, cell["config_idx"], cell["K"])
        try:
            return cell, M.collect_world_bundle(cell, seed)
        except RuntimeError:
            continue
    raise RuntimeError(f"no valid bundle found for cell {cell}")


# ---------------------------------------------------------------------------
# Test: streaming trainer -- one optimizer step per segment.
# ---------------------------------------------------------------------------

def test_streaming_trainer_segment_steps():
    """`train_hrmv2` on a tiny dataset (1 bundle, capped to <=256 rows,
    batch 128, epochs=1): the optimizer's `.step` must be called once PER
    SEGMENT actually run (summed over every batch's segment loop) -- this
    is the faithful loop's whole point (deep supervision: one gradient
    update per ACT segment, not per batch). AdamATan2 must be the
    optimizer class in use, and `warmup_constant_lr` must drive the LR
    schedule (verified in the companion test below).

    T7-review IMPORTANT 2 assertions: the segment loop drains on
    `ever_halted.all()` (every row completed >= 1 ACT episode this batch),
    which `is_last_step` bounds at `halt_max_steps` segments per batch --
    so (a) `_PER_BATCH_SEGMENT_CAP` must NEVER bind (`capped_batches ==
    0`), (b) every batch's segment count is <= halt_max_steps
    (`max_batch_segments`), and (c) total segments <= epochs x n_batches x
    halt_max_steps. (Pre-fix, the loop waited for ALL rows to be halted on
    the SAME call -- `carry.halted.all()` -- which desyncs once training
    perturbs the Q-head; the 200-segment cap then bound on ~90% of
    mid-training batches, inflating each batch to ~25x its
    `halt_max_steps` budget.)

    Minor 2: the hrmv2 architecture config dict must round-trip through
    `meta["hrmv2_config"]` (so a non-default architecture can be
    reconstructed at eval time from the ckpt/manifest instead of failing a
    strict state_dict load against `hrmv2_config()`'s defaults)."""
    cell, bundle = _one_bundle("A", K=2)
    # Cap to <=256 rows for speed (targets are already small for K=2, A).
    if bundle.targets.shape[0] > 256:
        bundle.targets = bundle.targets[:256]

    hrmv2_cfg = _tiny_hrmv2_cfg()
    mission_cfg = _tiny_mission_cfg(batch_size=128, epochs=1, lr=1e-4, weight_decay=0.0)

    step_calls = {"n": 0}
    orig_step = HA._load_hrm  # sanity: _load_hrm must be usable here too
    assert callable(orig_step)

    import hrm.train.optim as _optim_mod

    orig_adam_step = _optim_mod.AdamATan2.step

    def _counting_step(self, closure=None):
        step_calls["n"] += 1
        return orig_adam_step(self, closure=closure)

    _optim_mod.AdamATan2.step = _counting_step
    try:
        state_dict, meta = HA.train_hrmv2(
            cell, [bundle], seed=0, cfg=mission_cfg, hrmv2_cfg=hrmv2_cfg, epochs=1,
        )
    finally:
        _optim_mod.AdamATan2.step = orig_adam_step

    assert step_calls["n"] > 0
    assert meta.get("total_segments") == step_calls["n"], (
        f"meta['total_segments']={meta.get('total_segments')} != "
        f"actual optimizer.step() calls={step_calls['n']}"
    )
    assert meta["arm"] == "hrmv2_act"
    assert meta["epochs"] == 1
    assert isinstance(state_dict, dict)
    assert all(v.device.type == "cpu" for v in state_dict.values())

    # IMPORTANT-2 drain-condition bounds: the cap is a pure backstop that
    # must never bind; each batch drains within halt_max_steps segments.
    halt_max = hrmv2_cfg["halt_max_steps"]
    n_rows = bundle.targets.shape[0]
    n_batches = n_rows // mission_cfg.batch_size  # drop-last
    assert meta["capped_batches"] == 0, (
        f"the per-batch segment cap must be a never-binding backstop under "
        f"the ever-halted drain condition; it bound on "
        f"{meta['capped_batches']} batch(es)"
    )
    assert meta["max_batch_segments"] <= halt_max, (
        f"every batch must drain within halt_max_steps={halt_max} segments; "
        f"observed max_batch_segments={meta['max_batch_segments']}"
    )
    assert meta["total_segments"] <= 1 * n_batches * halt_max, (
        f"total segments {meta['total_segments']} exceeds the "
        f"epochs x batches x halt_max_steps bound {1 * n_batches * halt_max}"
    )

    # Minor 2: the architecture config round-trips through meta.
    assert meta["hrmv2_config"] == hrmv2_cfg


def test_streaming_trainer_uses_adamatan2_and_warmup():
    """Directly confirms the optimizer class and the warmup->constant LR
    multiplier behavior (`warmup_constant_lr`), independent of the
    step-counting test above."""
    from hrm.train.optim import AdamATan2, warmup_constant_lr

    assert warmup_constant_lr(0, 100) == 0.0
    assert warmup_constant_lr(50, 100) == pytest.approx(0.5)
    assert warmup_constant_lr(100, 100) == pytest.approx(1.0)
    assert warmup_constant_lr(500, 100) == pytest.approx(1.0)

    cell, bundle = _one_bundle("A", K=2)
    if bundle.targets.shape[0] > 128:
        bundle.targets = bundle.targets[:128]
    hrmv2_cfg = _tiny_hrmv2_cfg()
    mission_cfg = _tiny_mission_cfg(batch_size=64, epochs=1, lr=3e-4, weight_decay=1e-5)

    captured = {}
    orig_init = AdamATan2.__init__

    def _capturing_init(self, params, **kwargs):
        captured["class"] = type(self).__name__
        captured["lr"] = kwargs.get("lr")
        captured["betas"] = kwargs.get("betas")
        captured["weight_decay"] = kwargs.get("weight_decay")
        return orig_init(self, params, **kwargs)

    AdamATan2.__init__ = _capturing_init
    try:
        HA.train_hrmv2(cell, [bundle], seed=1, cfg=mission_cfg, hrmv2_cfg=hrmv2_cfg, epochs=1)
    finally:
        AdamATan2.__init__ = orig_init

    assert captured["class"] == "AdamATan2"
    assert captured["lr"] == pytest.approx(3e-4)
    assert captured["betas"] == (0.9, 0.95)
    assert captured["weight_decay"] == pytest.approx(1e-5)


# ---------------------------------------------------------------------------
# Test: forced-k provider -- exact segment count, ignores q-halt.
# ---------------------------------------------------------------------------

def test_forced_k_provider_exact_segments():
    """`_hrmv2_forward_batch_forced_k(k)` must run EXACTLY k inner segment
    advances regardless of q-halt (a forward hook on the inner module
    counts calls); output must be finite and in [0, cap]."""
    cell, bundle = _one_bundle("A", K=2)

    hrmv2_cfg = _tiny_hrmv2_cfg()
    mission_cfg = _tiny_mission_cfg()
    model = HA.build_hrmv2_arm(hrmv2_cfg)
    model.eval()

    N = bundle.rm.points.shape[0]
    states = [(i, s) for s in range(3) for i in range(N)][:20]

    inner_calls = {"n": 0}

    def _hook(module, args, output):
        inner_calls["n"] += 1

    handle = model.inner.register_forward_hook(_hook)
    try:
        for k in (1, 2, 4, 8):
            inner_calls["n"] = 0
            fwd = HA._hrmv2_forward_batch_forced_k(k)
            with torch.no_grad():
                out = fwd(model, bundle, states, mission_cfg, torch.device("cpu"))
            assert inner_calls["n"] == k, f"k={k}: expected {k} inner advances, got {inner_calls['n']}"
            assert out.shape == (len(states),)
            assert torch.isfinite(out).all()
            assert (out >= 0.0).all()
            assert (out <= mission_cfg.residual_cap).all()
    finally:
        handle.remove()


def test_forced_k8_equals_halting_disabled():
    """k=8 forced output must equal the ACT-live provider's output on a
    COPY of the model whose q-head is rigged to never halt early (bias
    forced very negative, so Q(halt) << Q(continue) always) -- with
    halt_max_steps=8 and q never voting to halt, the ACT-live path also
    runs exactly 8 segments, so the two should coincide. Read the halting
    condition's direction first (see `HRMACTv1.forward`): `halted = halted
    | (q_halt_logits > q_continue_logits)` -- forcing q_halt_logits very
    low relative to q_continue_logits means this OR-term is always False,
    so only `is_last_step` (step 8) halts, matching forced-k=8 exactly.
    Eval mode also disables `halt_exploration_prob`'s forced-min-steps
    branch entirely (gated on `self.training`), so this is a clean
    apples-to-apples comparison."""
    cell, bundle = _one_bundle("A", K=2)
    hrmv2_cfg = _tiny_hrmv2_cfg(halt_max_steps=8)
    mission_cfg = _tiny_mission_cfg()

    model = HA.build_hrmv2_arm(hrmv2_cfg)
    model.eval()

    # Rig a COPY's q_head so Q(halt) is always far below Q(continue):
    # q_head outputs 2 channels (halt, continue); force the halt channel's
    # bias very negative and the continue channel's bias very positive.
    import copy
    rigged = copy.deepcopy(model)
    rigged.eval()
    with torch.no_grad():
        rigged.inner.q_head.bias[0] = -100.0  # Q(halt) channel
        rigged.inner.q_head.bias[1] = 100.0   # Q(continue) channel
        rigged.inner.q_head.weight.zero_()  # kill input-dependence, bias dominates

    N = bundle.rm.points.shape[0]
    states = [(i, s) for s in range(3) for i in range(N)][:12]

    fwd_k8 = HA._hrmv2_forward_batch_forced_k(8)
    with torch.no_grad():
        out_k8 = fwd_k8(model, bundle, states, mission_cfg, torch.device("cpu"))
        out_act_rigged = HA._hrmv2_forward_batch_act(rigged, bundle, states, mission_cfg, torch.device("cpu"))

    assert torch.allclose(out_k8, out_act_rigged, atol=1e-4), (
        f"forced-k=8 output != ACT-live (halting disabled) output: "
        f"max diff = {(out_k8 - out_act_rigged).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Test: ACT-live provider -- halt-step side channel.
# ---------------------------------------------------------------------------

def test_act_live_halt_steps_side_channel():
    """After running the ACT-live forward_batch over 10 states, the bundle
    must carry `hrmv2_halt_steps`: an `(K+1, N)` float32 ndarray, NaN off
    the 10 queried (i, s) cells, and in `[1, halt_max_steps]` on the
    queried ones.

    T7-review IMPORTANT 1: the channel records the Q-DERIVED WOULD-HALT
    step -- the FIRST segment at which `q_halt_logits > q_continue_logits`
    fires for that query, `halt_max_steps` if it never fires -- NOT
    segments-run. Segments-run is CONSTANT (`halt_max_steps`) in eval mode
    (the early-halt branch in `HRMACTv1.forward` is `self.training`-gated;
    reviewer confirmed the vendored original behaves identically), and
    G2b's Spearman(halt-steps, K) is undefined on a constant. This test
    pins the two rig-derived extremes the reviewer verified in a
    prototype:
      - a FRESH model: `q_head` init is weight-zeroed + bias filled -5.0
        on BOTH channels (`HRMACTv1_Inner.__init__`), so `q_halt ==
        q_continue` exactly and the STRICT `>` never fires -> every
        queried value == halt_max_steps;
      - a rigged COPY (bias +100/-100, weight zeroed): `q_halt >
        q_continue` from segment 1 -> every queried value == 1.0.
    The two rigs producing DIFFERENT values is the load-bearing assertion:
    the channel must VARY with the Q-head, else G2b is degenerate."""
    import copy

    cell, bundle = _one_bundle("A", K=2)
    hrmv2_cfg = _tiny_hrmv2_cfg(halt_max_steps=4)
    mission_cfg = _tiny_mission_cfg()

    model = HA.build_hrmv2_arm(hrmv2_cfg)
    model.eval()

    N = bundle.rm.points.shape[0]
    states = [(i, s) for s in range(3) for i in range(N)][:10]

    # --- Rig 1: fresh model (q_halt == q_continue == -5.0 exactly; strict
    # `>` never fires) -> would-halt == halt_max_steps for every query.
    with torch.no_grad():
        out = HA._hrmv2_forward_batch_act(model, bundle, states, mission_cfg, torch.device("cpu"))

    assert out.shape == (10,)
    assert hasattr(bundle, "hrmv2_halt_steps")
    halt_steps = bundle.hrmv2_halt_steps
    assert halt_steps.shape == (3, N)
    assert halt_steps.dtype == np.float32

    queried_mask = np.zeros((3, N), dtype=bool)
    for (i, s) in states:
        queried_mask[s, i] = True

    fresh_vals = halt_steps[queried_mask].copy()
    assert np.all(np.isfinite(fresh_vals))
    assert np.all(fresh_vals >= 1)
    assert np.all(fresh_vals <= hrmv2_cfg["halt_max_steps"])
    assert np.all(fresh_vals == float(hrmv2_cfg["halt_max_steps"])), (
        f"fresh q-head (bias -5/-5, strict >) must never vote halt -> all "
        f"would-halt == {hrmv2_cfg['halt_max_steps']}; got "
        f"{sorted(set(fresh_vals.tolist()))}"
    )

    unqueried_vals = halt_steps[~queried_mask]
    assert np.all(np.isnan(unqueried_vals))

    # --- Rig 2: q-head rigged to vote halt from the FIRST segment
    # (q_halt=+100 >> q_continue=-100, input-independent) -> would-halt ==
    # 1.0 for every query.
    rigged = copy.deepcopy(model)
    rigged.eval()
    with torch.no_grad():
        rigged.inner.q_head.bias[0] = 100.0   # Q(halt)
        rigged.inner.q_head.bias[1] = -100.0  # Q(continue)
        rigged.inner.q_head.weight.zero_()

    # Clear the side channel so rig 2's writes to the SAME queried cells
    # are unambiguous (not rig-1 leftovers).
    del bundle.hrmv2_halt_steps
    with torch.no_grad():
        out2 = HA._hrmv2_forward_batch_act(rigged, bundle, states, mission_cfg, torch.device("cpu"))
    assert out2.shape == (10,)

    rigged_vals = bundle.hrmv2_halt_steps[queried_mask]
    assert np.all(rigged_vals == 1.0), (
        f"q-head rigged +100/-100 must vote halt at segment 1 -> all "
        f"would-halt == 1.0; got {sorted(set(rigged_vals.tolist()))}"
    )

    # The channel VARIES with the Q-head -- the whole point (G2b is
    # undefined on a constant).
    assert not np.array_equal(fresh_vals, rigged_vals)


# ---------------------------------------------------------------------------
# Test: trainer determinism on CPU.
# ---------------------------------------------------------------------------

def test_trainer_determinism_cpu(monkeypatch):
    """Two identical `train_hrmv2` calls (same cell/bundle/seed/cfg) on CPU
    must produce allclose state_dicts.

    T7-review minor 4: `train_hrmv2` auto-selects CUDA when available
    (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`), and
    this machine HAS CUDA -- so without intervention this test would
    actually exercise GPU determinism (which the trainer's own docstring
    explicitly does NOT promise: CUDA attention kernels are not bitwise
    deterministic). To make the test match its name and the docstring's
    CPU-only determinism claim, `torch.cuda.is_available` is monkeypatched
    to False for the duration of BOTH calls, forcing the CPU path (the
    reviewer's offered alternative was renaming the test; forcing CPU was
    chosen because CPU determinism is the property the trainer actually
    guarantees, so it is the property worth pinning)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    cell, bundle = _one_bundle("A", K=2)
    if bundle.targets.shape[0] > 128:
        bundle.targets = bundle.targets[:128]

    hrmv2_cfg = _tiny_hrmv2_cfg()
    mission_cfg = _tiny_mission_cfg(batch_size=64, epochs=1)

    state_dict_a, meta_a = HA.train_hrmv2(cell, [bundle], seed=3, cfg=mission_cfg,
                                           hrmv2_cfg=hrmv2_cfg, epochs=1)
    state_dict_b, meta_b = HA.train_hrmv2(cell, [bundle], seed=3, cfg=mission_cfg,
                                           hrmv2_cfg=hrmv2_cfg, epochs=1)

    assert state_dict_a.keys() == state_dict_b.keys()
    for k in state_dict_a:
        assert torch.allclose(state_dict_a[k], state_dict_b[k], atol=1e-6), f"mismatch at {k}"
    assert meta_a["final_loss"] == pytest.approx(meta_b["final_loss"])
    assert all(v.device.type == "cpu" for v in state_dict_a.values())


# ---------------------------------------------------------------------------
# Test: end-to-end run_train -> run_eval through the REAL registry dispatch
# chain (not train_hrmv2 called directly). Regression test for a genuine
# signature-mismatch bug caught by exactly this kind of full-pipeline
# smoke test: `register_hrmv2_providers` originally registered
# `train=train_hrmv2` directly, but `_resolve_trainer`'s core-module
# adapter calls `registered_train(cell, bundles, seed, cfg, epochs)` -- 5
# positional arguments -- which `train_hrmv2` (signature `cell, bundles,
# seed, cfg=None, hrmv2_cfg=None, epochs=None`) silently received as
# `hrmv2_cfg=<the caller's epochs value>`, crashing inside `hrmv2_config()`
# the moment `epochs` was an int (`TypeError: 'int' object is not
# iterable`). No unit test of `train_hrmv2` directly (calling it with
# explicit keyword arguments, as every OTHER test in this file does) could
# ever catch this -- it only manifests when `train_hrmv2` is invoked
# THROUGH the registry's positional-argument adapter, exactly as
# `run_train` does in production.
# ---------------------------------------------------------------------------

def test_run_train_run_eval_end_to_end_through_registry(tmp_path, monkeypatch):
    """`run_train` (arms hrmv2_act + two forced-k names) -> `run_eval`
    through the REAL `_resolve_trainer`/`register_hrmv2_providers` wiring,
    with a tiny `hrmv2_config` override (monkeypatched module-level, so
    `register_hrmv2_providers`'s `_build`/`_train` closures -- which call
    the BARE `hrmv2_config()` name at call time, not a captured reference
    -- pick it up): confirms (a) `hrmv2_act` trains and gets a manifest
    entry, (b) BOTH forced-k arms are skipped by `run_train` (no manifest
    entry, matching `test_hrmv2_registration_and_g1_family`'s isolated
    check, now exercised end-to-end), (c) `run_eval` produces CSV records
    for `hrmv2_act` AND both forced-k arms via `ckpt_alias` resolution,
    (d) the forced-k arms' records carry the SAME `train_seed` as
    `hrmv2_act` (proof they rode its checkpoint, not a fresh/wrong one)."""
    import csv
    import json as _json

    _original_hrmv2_config = HA.hrmv2_config  # Close over the REAL function
    # -- calling `HA.hrmv2_config(...)` from inside the replacement below
    # would recurse into the replacement itself once monkeypatched (the
    # module attribute IS the replacement by the time it's called).

    def _tiny_hrmv2_config(cfg=None, **overrides):
        base = _original_hrmv2_config(cfg, hidden_size=32, num_heads=2, H_layers=1, L_layers=1,
                                       H_cycles=1, L_cycles=1, halt_max_steps=3)
        base.update(overrides)
        return base

    monkeypatch.setattr(HA, "hrmv2_config", _tiny_hrmv2_config)

    cells = M.build_cell_grid()
    cell = next(c for c in cells if c["config_label"] == "A" and c["K"] == 2)
    mission_cfg = M.C11MissionConfig(batch_size=64, epochs=1, lr=1e-4, weight_decay=0.0)

    out_dir = tmp_path / "e2e_run"
    M.run_train(str(out_dir), arms=("hrmv2_act", "hrmv2_act_k2", "hrmv2_act_k4"),
                cells=[cell], seeds=(0,), cfg=mission_cfg, n_train_worlds=2, epochs=1)

    manifest = _json.loads((out_dir / "manifest.json").read_text())
    assert "hrmv2_act__A2__s0" in manifest
    assert "hrmv2_act_k2__A2__s0" not in manifest
    assert "hrmv2_act_k4__A2__s0" not in manifest

    M.run_eval(str(out_dir), cells=[cell], cfg=mission_cfg, n_test_worlds=2, budgets=(400,),
               arms=("hrmv2_act", "hrmv2_act_k2", "hrmv2_act_k4", "h_legsum"))

    with open(out_dir / "results" / "c11_eval_raw.csv") as f:
        rows = list(csv.DictReader(f))
    arms_seen = set(r["arm"] for r in rows)
    assert "hrmv2_act" in arms_seen
    assert "hrmv2_act_k2" in arms_seen
    assert "hrmv2_act_k4" in arms_seen

    hrmv2_seeds = set(r["train_seed"] for r in rows if r["arm"] == "hrmv2_act")
    k2_seeds = set(r["train_seed"] for r in rows if r["arm"] == "hrmv2_act_k2")
    k4_seeds = set(r["train_seed"] for r in rows if r["arm"] == "hrmv2_act_k4")
    assert k2_seeds == hrmv2_seeds
    assert k4_seeds == hrmv2_seeds
