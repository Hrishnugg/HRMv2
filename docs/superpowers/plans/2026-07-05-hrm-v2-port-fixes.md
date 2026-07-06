# HRM-v2 Port Fixes + Maze Revalidation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the HRM-v2 Blackwell port's training-procedure and ops defects identified in `HRM-v2/PORT_FIDELITY_AUDIT.md` (D1/D2/D3/D4/D5, B1/B2, C2), guard fidelity with an original-vs-port parity test, and revalidate by re-training Maze-Hard 30×30 — expecting exact-accuracy to move from 25.4% toward the paper's ≈75%, avg-steps < 16, and q_halt decoupled from 1−exact.

**Architecture:** All changes inside `HRM-v2/` (the port) — the vendored original at repo-root `models/` is FROZEN ground truth (never edit). New `src/hrm/train/losses.py` (verbatim ACTLossHead port) and `src/hrm/train/optim.py` (pure-PyTorch AdamATan2 + warmup schedule); `ops/attention.py` gets a layout contract; `models/sparse_embedding.py` optimizer reverts to original logic with forward-side tail handling; both train scripts get the original's streaming-carry, one-segment-per-optimizer-step loop.

**Tech Stack:** PyTorch 2.9.0+cu130, Python 3.13, RTX 5090 (flash-attn absent → SDPA fallback path is live). `python -m pytest HRM-v2/tests -q` from repo root.

**Conventions:** Branch `hrm-v2-fixes`. NEVER modify repo-root `models/`, `config/`, `dataset/` (vendored original) or the WIP files (`hrm-cloud/continuous_prm/continuous_prm_common.py`, `hrm-cloud/transfer_astar_heuristic_clean_parallel_fixed.py`). Stage only files under `HRM-v2/` (+ this plan). TDD per task.

**Audit cross-refs:** `HRM-v2/PORT_FIDELITY_AUDIT.md` §B1/B2 (attention), §C2 (sparse-emb), §D1–D6 (training), §F (fix list). Original ground truth: `models/hrm/hrm_act_v1.py`, `models/layers.py:98-135` (attention), `models/losses.py` (ACTLossHead), `models/sparse_embedding.py`.

---

## Task 1: `sdpa` layout contract + ImportError-only fallback (B1, B2)

**Files:** Modify `HRM-v2/src/hrm/ops/attention.py`; Modify `HRM-v2/tests/test_attention.py`.

- [ ] **Step 1: failing test.** REPLACE the two `(B,H,S,D)`-layout sdpa tests in `tests/test_attention.py` with contract tests (the API contract becomes: q,k,v are ALWAYS `(batch, seqlen, num_heads, head_dim)`). Add the killer case seq_len < num_heads, verified against a manual reference:
```python
def _reference_attention(q, k, v):  # (B,S,H,D) -> (B,S,H,D), plain softmax attention
    import torch
    qh = q.permute(0, 2, 1, 3); kh = k.permute(0, 2, 1, 3); vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    return (torch.softmax(scores, dim=-1) @ vh).permute(0, 2, 1, 3)

def test_sdpa_seq_shorter_than_heads():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 8, 16) for _ in range(3))   # S=4 < H=8 — the audit B1 case
    out = sdpa(q, k, v)
    ref = _reference_attention(q, k, v)
    assert out.shape == q.shape and torch.allclose(out, ref, atol=1e-5)

def test_sdpa_seq_longer_than_heads():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 64, 8, 16) for _ in range(3))
    assert torch.allclose(sdpa(q, k, v), _reference_attention(q, k, v), atol=1e-5)

def test_sdpa_seq_equal_heads():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 8, 8, 16) for _ in range(3))
    assert torch.allclose(sdpa(q, k, v), _reference_attention(q, k, v), atol=1e-5)
```
- [ ] **Step 2: run** `python -m pytest HRM-v2/tests/test_attention.py -q` → the S<H and S==H cases FAIL (wrong-axis attention).
- [ ] **Step 3: implement.** In `ops/attention.py`: delete the `needs_transpose` heuristic; document + enforce the contract:
```python
def sdpa(q, k, v, attn_mask=None, is_causal=False, dropout_p=0.0):
    """q, k, v: (batch, seqlen, num_heads, head_dim) — the HRM layout contract (matches flash-attn)."""
    q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    return out.transpose(1, 2)
```
And in `attention(...)`: change `except (ImportError, Exception)` → `except ImportError` (B2); keep the dtype/cuda gating. Update any other test in the file that passed `(B,H,S,D)` to the new contract.
- [ ] **Step 4: run** `python -m pytest HRM-v2/tests/test_attention.py -q` → all pass. **Step 5: commit** `fix(hrm-v2): sdpa layout contract (B,S,H,D) — remove shape-guess heuristic; ImportError-only flash fallback`.

---

## Task 2: Port `ACTLossHead` verbatim (D1, D3, D4)

**Files:** Create `HRM-v2/src/hrm/train/losses.py`; Test `HRM-v2/tests/test_losses.py`. (`src/hrm/train/__init__.py` exists; re-export from it.)

- [ ] **Step 1: failing test** (`tests/test_losses.py`) — parity vs the ORIGINAL `models/losses.py` (pure torch, importable by path) + a functional check that q_halt_loss exists:
```python
import sys, types, importlib.util
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m

orig_losses = _load("orig_losses", REPO / "models" / "losses.py")
from hrm.train.losses import stablemax_cross_entropy, softmax_cross_entropy, ACTLossHead, IGNORE_LABEL_ID

def test_ce_parity_with_original():
    torch.manual_seed(0)
    logits = torch.randn(4, 12, 11); labels = torch.randint(0, 11, (4, 12)); labels[0, :6] = -100
    for fn_o, fn_p in ((orig_losses.stablemax_cross_entropy, stablemax_cross_entropy),
                       (orig_losses.softmax_cross_entropy, softmax_cross_entropy)):
        assert torch.allclose(fn_o(logits, labels), fn_p(logits, labels).to(fn_o(logits, labels).dtype), atol=1e-6)

class _DummyACT(torch.nn.Module):
    """Returns canned outputs shaped like HRMACTv1 so ACTLossHead semantics can be asserted."""
    def __init__(self, logits, qh, qc, labels, halted, steps, target_qc=None):
        super().__init__(); self.o = dict(logits=logits, q_halt_logits=qh, q_continue_logits=qc)
        if target_qc is not None: self.o["target_q_continue"] = target_qc
        from dataclasses import dataclass
        self.carry = types.SimpleNamespace(current_data={"labels": labels}, halted=halted, steps=steps)
    def forward(self, carry, batch): return self.carry, self.o

def test_act_loss_head_includes_q_halt_term():
    torch.manual_seed(0)
    B, S, V = 3, 8, 7
    logits = torch.randn(B, S, V, requires_grad=True)
    qh = torch.randn(B, requires_grad=True); qc = torch.randn(B, requires_grad=True)
    labels = torch.randint(0, V, (B, S))
    dummy = _DummyACT(logits, qh, qc, labels, halted=torch.ones(B, dtype=torch.bool), steps=torch.ones(B, dtype=torch.int32))
    head = ACTLossHead(dummy, loss_type="softmax_cross_entropy")
    carry, loss, metrics, _, all_halted = head(return_keys=[], carry=None, batch=None)
    loss.backward()
    assert qh.grad is not None and qh.grad.abs().sum() > 0      # D1: q_halt gets gradient
    assert "q_halt_loss" in metrics and "lm_loss" in metrics
    # divisor semantics: per-sequence mean over valid tokens, then sum (not global mean) — D3
    per_tok = torch.nn.functional.cross_entropy(logits.detach().float().view(-1, V), labels.view(-1), reduction="none").view(B, S)
    expected_lm = (per_tok / S).sum()
    assert torch.allclose(metrics["lm_loss"].float(), expected_lm, rtol=1e-4)
```
- [ ] **Step 2: RED** (`ModuleNotFoundError: hrm.train.losses`).
- [ ] **Step 3: implement** — copy `models/losses.py` VERBATIM into `src/hrm/train/losses.py` (it imports only torch/F/typing — no repo-specific imports; keep `IGNORE_LABEL_ID = -100`, `s`, `log_stablemax`, `stablemax_cross_entropy`, `softmax_cross_entropy`, `ACTLossHead` including the `0.5 * (q_halt_loss + q_continue_loss)` weighting, sum reductions, per-sequence `loss_divisor`, and the metrics dict). Re-export in `src/hrm/train/__init__.py`. Do NOT "improve" anything — verbatim is the point.
- [ ] **Step 4: GREEN** `python -m pytest HRM-v2/tests/test_losses.py -q`. **Step 5: commit** `feat(hrm-v2): port ACTLossHead + stablemax/softmax CE verbatim from models/losses.py (restores q_halt_loss, per-seq divisor, 0.5 weighting)`.

---

## Task 3: Sparse-embedding optimizer — revert to original logic; forward owns variable batches (C2)

**Files:** Modify `HRM-v2/src/hrm/models/sparse_embedding.py`; Test `HRM-v2/tests/test_models.py` (append).

Design: delete the optimizer's `actual_batch_size` inference entirely (revert `step` to the original `models/sparse_embedding.py:63-95` logic — full-tensor path). Handle short batches in `forward`: after copying the live rows, fill the TAIL of `local_ids` with `inputs[0]` (a valid, in-batch id) so stale tail rows contribute `sign(0)=0` gradient to a real id and never resurrect stale ids; tail `local_weights` rows keep zero grad.

- [ ] **Step 1: failing tests** (append to `tests/test_models.py`):
```python
def test_sparse_emb_midbatch_zero_grad_row_not_dropped():
    torch.manual_seed(0)
    emb = CastedSparseEmbedding(10, 4, batch_size=3, init_std=0.5, cast_to=torch.float32); emb.train()
    ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    out = emb(ids)
    loss = out[0].sum() + out[2].sum()          # row 1 (id=2) gets EXACTLY zero grad
    loss.backward()
    w_before = emb.weights.clone()
    opt = CastedSparseEmbeddingSignSGD_Distributed([emb.weights, emb.local_weights, emb.local_ids], world_size=1, lr=0.1, weight_decay=0.0)
    opt.step()
    assert not torch.allclose(emb.weights[1], w_before[1])      # id 1 updated
    assert not torch.allclose(emb.weights[3], w_before[3])      # id 3 (tail after zero-grad row) NOT dropped — the C2 bug
    assert torch.allclose(emb.weights[2], w_before[2])          # zero-grad id: sign(0)=0, wd=0 -> unchanged

def test_sparse_emb_short_batch_no_stale_id_updates():
    torch.manual_seed(0)
    emb = CastedSparseEmbedding(10, 4, batch_size=4, init_std=0.5, cast_to=torch.float32); emb.train()
    emb(torch.tensor([7, 8, 9, 6], dtype=torch.int32))          # populate all 4 rows
    out = emb(torch.tensor([1, 2], dtype=torch.int32))          # short batch: rows 2,3 are stale
    out.sum().backward()
    w_before = emb.weights.clone()
    opt = CastedSparseEmbeddingSignSGD_Distributed([emb.weights, emb.local_weights, emb.local_ids], world_size=1, lr=0.1, weight_decay=0.5)
    opt.step()
    assert not torch.allclose(emb.weights[1], w_before[1]) and not torch.allclose(emb.weights[2], w_before[2])
    for stale in (7, 8, 9, 6):                                   # stale ids must receive NO update, NO weight decay
        assert torch.allclose(emb.weights[stale], w_before[stale])
```
- [ ] **Step 2: RED** (first test: id-3 assertion fails under current prefix-slice heuristic; second: stale ids decay under current code).
- [ ] **Step 3: implement.** `forward` (training branch): after the existing copy of live rows, add
```python
        if actual_batch_size < self.local_ids.shape[0]:
            self.local_ids[actual_batch_size:].fill_(int(inputs[0].item()))
```
`step`: revert to the original body — remove lines 155–164 (both heuristics) and the `[:actual_batch_size]` slicing; call `_sparse_emb_signsgd_dist(local_weights_grad, local_ids, weights, ...)` on the full tensors, exactly as `models/sparse_embedding.py:85-95`. NOTE: `zero_grad` between steps is the caller's job (train scripts already do it); grads for unused tail rows are zero after `backward()`, so `sign(0)=0` and the tail contributes nothing.
- [ ] **Step 4: GREEN** `python -m pytest HRM-v2/tests/test_models.py -q` (whole file — no regressions). **Step 5: commit** `fix(hrm-v2): sparse-emb optimizer reverts to original full-batch logic; forward fills stale tail ids (C2)`.

---

## Task 4: AdamATan2 (pure PyTorch) + constant-LR-after-warmup (D5)

**Files:** Create `HRM-v2/src/hrm/train/optim.py`; Test `HRM-v2/tests/test_optim.py`.

- [ ] **Step 1: failing test:**
```python
import torch
from hrm.train.optim import AdamATan2, warmup_constant_lr

def test_adam_atan2_steps_and_is_scale_invariant_in_eps():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = AdamATan2([p], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)
    before = p.detach().clone()
    (p.square().sum()).backward()
    opt.step()
    assert not torch.allclose(p.detach(), before)
    assert torch.isfinite(p).all()

def test_adam_atan2_zero_grad_only_decays():
    p = torch.nn.Parameter(torch.ones(4))
    opt = AdamATan2([p], lr=0.1, betas=(0.9, 0.95), weight_decay=0.5)
    p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.allclose(p.detach(), torch.full((4,), 1.0 * (1 - 0.1 * 0.5)))  # atan2(0, x)=0 -> pure decoupled decay

def test_warmup_constant():
    assert warmup_constant_lr(0, 100) == 0.0 and warmup_constant_lr(50, 100) == 0.5
    assert warmup_constant_lr(100, 100) == 1.0 and warmup_constant_lr(5000, 100) == 1.0
```
- [ ] **Step 2: RED. Step 3: implement** (`a`/`b` constants per the adam-atan2 formulation the official HRM depends on):
```python
import math, torch

class AdamATan2(torch.optim.Optimizer):
    """Adam with atan2-based update (scale-invariant, epsilon-free) + decoupled weight decay.
    update = a * atan2(m_hat, b * sqrt(v_hat));  a = 1.2732395447351628, b = 1.0."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, a=1.2732395447351628, b=1.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a, b=b))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, (b1, b2), wd, a, b = group["lr"], group["betas"], group["weight_decay"], group["a"], group["b"]
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1; t = st["step"]
                m, v = st["exp_avg"], st["exp_avg_sq"]
                m.lerp_(g, 1 - b1); v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1 ** t); v_hat = v / (1 - b2 ** t)
                if wd != 0: p.mul_(1 - lr * wd)                       # decoupled (AdamW-style)
                p.add_(torch.atan2(m_hat, b * v_hat.sqrt()), alpha=-lr * a)
        return None

def warmup_constant_lr(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0: return 1.0
    return min(1.0, step / warmup_steps)
```
Export from `src/hrm/train/__init__.py`.
- [ ] **Step 4: GREEN** `python -m pytest HRM-v2/tests/test_optim.py -q`. **Step 5: commit** `feat(hrm-v2): pure-PyTorch AdamATan2 + warmup-then-constant LR (paper recipe, no CUDA-ext dependency)`.

---

## Task 5: Faithful training loop in both train scripts (D1, D2, D5, D6)

**Files:** Modify `HRM-v2/train_maze_optimized.py`, `HRM-v2/train_sudoku.py`. (Tests: the loop's semantics are guarded by T2/T4 units + a smoke here.)

Replace the loss/loop machinery in BOTH scripts (keep dataset classes, configs, checkpointing, eval-metrics printing):

- [ ] **Step 1: delete** the local `stablemax_cross_entropy`, `train_step`, and the inner `for _ in range(halt_max_steps)` loops (train AND the cosine schedule); import instead: `from hrm.train import ACTLossHead, IGNORE_LABEL_ID, AdamATan2, warmup_constant_lr`.
- [ ] **Step 2: implement the original streaming loop** (one segment per optimizer step, persistent carry, deep supervision):
```python
    loss_head = ACTLossHead(model, loss_type="stablemax_cross_entropy")
    carry = None
    global_step = 0
    for batch in train_dataset:                       # loader already streams epochs
        batch = {k: v.to(config.device) for k, v in batch.items()}
        if carry is None:
            carry = loss_head.initial_carry(batch)    # all-halted -> slots fill from this batch

        lr_mult = warmup_constant_lr(global_step, config.warmup_steps)
        for pg in optimizer.param_groups: pg["lr"] = config.lr * lr_mult
        if puzzle_emb_optimizer:
            for pg in puzzle_emb_optimizer.param_groups: pg["lr"] = config.puzzle_emb_lr * lr_mult

        optimizer.zero_grad(set_to_none=False)        # sparse-emb grads must exist as zeros for SignSGD
        if puzzle_emb_optimizer: puzzle_emb_optimizer.zero_grad(set_to_none=False)

        carry, loss, metrics, _, _all_halted = loss_head(return_keys=[], carry=carry, batch=batch)
        (loss / config.batch_size).backward()         # per-sequence scale (loss is sum over sequences)
        optimizer.step()
        if puzzle_emb_optimizer: puzzle_emb_optimizer.step()
        global_step += 1
        # metrics: accumulate metrics["count"]-weighted accuracy/exact_accuracy/q_halt_accuracy/steps as before
```
Config changes (both scripts): `weight_decay: float = 1.0` (was 0.1), optimizer = `AdamATan2(main_params, lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)` (was AdamW), **remove `clip_grad_norm_`**, remove `cosine_schedule`. Keep puzzle-emb SignSGD wiring as-is (3 explicit tensors). NOTE: `metrics` values are tensors — `.item()` them for logging; `metrics["count"]` can be 0 early (nothing halted yet) — guard division. Eval loops stay as-is structurally but must call `loss_head` (or keep calling `model` directly and compute metrics as before — simpler: reuse the loss head with a fresh carry per eval batch, iterate `halt_max_steps` times, accumulate metrics where halted; in eval mode halting = max-steps, matching the original's fixed-compute eval).
- [ ] **Step 3: smoke test** (append `tests/test_train_smoke.py`): build a tiny HRMACTv1 (hidden 64, heads 4, seq 16, vocab 8, H/L cycles 1/1, layers 1/1, halt_max_steps 3, batch 4, puzzle_emb_ndim 0, forward_dtype float32) + ACTLossHead + AdamATan2; feed 8 random batches through the STREAMING loop (copy the loop shape from the script); assert: loss finite every step, ≥1 parameter changed, and `q_head.weight.grad` is non-None with nonzero norm on at least one step (D1 regression guard at the loop level).
- [ ] **Step 4:** `python -m pytest HRM-v2/tests -q` (all green, incl. earlier tasks). **Step 5: commit** `fix(hrm-v2): faithful training loop — streaming carry, one segment per optimizer step (deep supervision), ACTLossHead, AdamATan2 + constant LR, wd 1.0, no grad clip`.

---

## Task 6: Original-vs-port parity test (F6 — guards everything)

**Files:** Create `HRM-v2/tests/test_parity_original.py`.

- [ ] **Step 1: implement the harness** (this test is the deliverable; it must FAIL if any future drift breaks numerics):
```python
import sys, types, importlib.util
from pathlib import Path
import pytest, torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]

# --- flash-attn shim (original models/layers.py imports it unconditionally) ---
def _shim_flash():
    def flash_attn_func(q, k, v, causal=False, **kw):   # (B,S,H,D) contract, SDPA-backed
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)
        return o.transpose(1, 2)
    for name in ("flash_attn_interface", "flash_attn"):
        m = types.ModuleType(name); m.flash_attn_func = flash_attn_func
        sys.modules.setdefault(name, m)

def _load(name, path, pkg_path=None):
    if pkg_path is not None:
        pkg = types.ModuleType(name); pkg.__path__ = [str(pkg_path)]; sys.modules[name] = pkg; return pkg
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m

@pytest.fixture(scope="module")
def orig():
    _shim_flash()
    _load("models", None, pkg_path=REPO / "models")
    _load("models.common", REPO / "models" / "common.py")
    _load("models.layers", REPO / "models" / "layers.py")
    _load("models.sparse_embedding", REPO / "models" / "sparse_embedding.py")
    _load("models.hrm", None, pkg_path=REPO / "models" / "hrm")
    return _load("models.hrm.hrm_act_v1", REPO / "models" / "hrm" / "hrm_act_v1.py")

def _cfg(seq_len, num_heads, hidden):
    return dict(batch_size=3, seq_len=seq_len, puzzle_emb_ndim=0, num_puzzle_identifiers=1, vocab_size=11,
                H_cycles=2, L_cycles=2, H_layers=1, L_layers=1, hidden_size=hidden, expansion=4.0,
                num_heads=num_heads, pos_encodings="rope", halt_max_steps=4, halt_exploration_prob=0.0,
                forward_dtype="float32")

@pytest.mark.parametrize("seq_len,num_heads,hidden", [(32, 4, 64), (4, 8, 64)])  # 2nd = audit B1 regime (S < H)
def test_inner_forward_parity(orig, seq_len, num_heads, hidden):
    from hrm.models.hrm_act_v1 import HRMACTv1
    torch.manual_seed(0)
    cfg = _cfg(seq_len, num_heads, hidden)
    o = orig.HierarchicalReasoningModel_ACTV1(cfg); p = HRMACTv1(cfg)
    missing, unexpected = p.load_state_dict(o.state_dict(), strict=False)
    assert not missing and not unexpected                     # identical parameterization
    o.eval(); p.eval()
    batch = {"inputs": torch.randint(0, 11, (3, seq_len)), "puzzle_identifiers": torch.zeros(3, dtype=torch.int32)}
    co = o.inner.reset_carry(torch.ones(3, dtype=torch.bool), o.inner.empty_carry(3))
    cp = p.inner.reset_carry(torch.ones(3, dtype=torch.bool), p.inner.empty_carry(3))
    with torch.no_grad():
        _, lo, (qho, qco) = o.inner(co, batch)
        _, lp, (qhp, qcp) = p.inner(cp, batch)
    assert torch.allclose(lo, lp, atol=1e-5), (lo - lp).abs().max()
    assert torch.allclose(qho, qhp, atol=1e-5) and torch.allclose(qco, qcp, atol=1e-5)
```
NOTES for the implementer: (a) `use_flash=True` in the port will attempt flash first — on this box flash-attn is NOT installed for the PORT path (real ImportError → SDPA), while the ORIGINAL path uses the shim (SDPA-backed) — both effectively SDPA in float32 → tight tolerance is correct. If flash IS importable, force the SDPA path for the test by constructing port blocks with `use_flash=False`... simpler: monkeypatch `hrm.ops.attention.flash_attention` to raise ImportError within this test module. (b) `empty_carry` original returns CPU tensors without device arg — fine on CPU. (c) If `load_state_dict(strict=False)` reports mismatched keys, DO NOT rename port modules — report the mismatch (that would itself be an audit finding). Expect exact key parity per the audit.
- [ ] **Step 2: run** `python -m pytest HRM-v2/tests/test_parity_original.py -q` → 2 passed (the S<H case passes only because Task 1 landed — this test would have caught B1). If tolerance fails at 1e-5 (bf16-free float32 path should be well within), investigate — do NOT loosen beyond 1e-4 without diagnosing.
- [ ] **Step 3: commit** `test(hrm-v2): original-vs-port numerical parity (flash shim + models pkg registration; covers S<H regime)`.

---

## Task 7: Rebuild maze dataset + revalidation run + writeup

**Files:** run artifacts under `HRM-v2/checkpoints/`, report `HRM-v2/RETRAIN_RESULTS.md`, status update in `HRM-v2/PORT_FIDELITY_AUDIT.md`.

- [ ] **Step 1: dataset.** `data/maze-30x30-hard-1k` is absent. Build it with the vendored original builder: read `dataset/build_maze_dataset.py` for its CLI (it fetches sapientinc HF data), run it from repo root so the output lands at the path `train_maze_optimized.py` expects (`../data/maze-30x30-hard-1k` relative to HRM-v2 → `<repo>/data/maze-30x30-hard-1k`; note the script runs with CWD=HRM-v2 — verify the relative path resolution and pass an absolute `--data-path` if the script supports it / adjust config default if needed). Confirm `train/dataset.json` + npy files exist after build.
- [ ] **Step 2: launch the run** (background, ~2–3 h wall clock on the 5090; the new loop does 1 segment/optimizer-step, so step count for comparable episode exposure ≈ old_steps × halt_max_steps — set total steps ≈ 50k or run the same epochs=? — match the ORIGINAL paper recipe closest available: keep the script's epoch-based streaming and simply let it run the configured epochs; log every 100 steps).
- [ ] **Step 3: gates** (from the audit §F7): (G-a) exact_accuracy substantially above the broken run's 25.4%, trending toward the paper's ≈75% band; (G-b) **avg steps < 16** with a nontrivial distribution once q_halt trains; (G-c) `q_halt_accuracy` decoupled from `1 − exact_accuracy` (the frozen-head identity broken). Honest reporting either way — if gates miss, diagnose (LR, steps, dataset variance) before concluding.
- [ ] **Step 4: writeup + commit.** `HRM-v2/RETRAIN_RESULTS.md`: config, curves (steps vs exact-acc/avg-steps/q_halt-acc), gate verdicts, before/after table vs the broken run and the paper. Append a status block to `PORT_FIDELITY_AUDIT.md` ("fixes landed @ SHAs; revalidation result"). Commit docs (checkpoints stay untracked).

---

## Self-review

**Spec coverage:** audit F1→T2, F2→T5, F3(B1/B2)→T1, F4(C2)→T3, F5(D5)→T4+T5, F6→T6, F7→T7. D3/D4 land inside T2 (verbatim head) + T5 (uses it). C1 (nn.Buffer) deliberately deferred — the `_apply` approach is compatible with the port's construction order; documented in the audit, out of scope here (YAGNI).
**Placeholder scan:** all code steps carry real code; T5's loop is complete; T7 is a run task with explicit gates; the one "read the CLI" item (dataset builder) is a legitimate runtime discovery, bounded and verifiable.
**Type consistency:** `ACTLossHead(model, loss_type=...)` / `loss_head(return_keys=[], carry=..., batch=...)` matches `models/losses.py:49-57` (kwargs → `self.model(**model_kwargs)` → `HRMACTv1.forward(carry, batch)`); `AdamATan2` + `warmup_constant_lr` names consistent T4→T5; `sdpa` contract consistent T1→T6 (shim uses the same (B,S,H,D)).
**Frozen ground truth:** no task touches repo-root `models/`, `config/`, `dataset/` sources (T7 only *runs* the dataset builder).
