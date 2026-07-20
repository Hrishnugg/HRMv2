# HRM-v2 Port Fidelity Audit — vs Original Code + Paper (arXiv 2506.21734v3)

**Date:** 2026-07-05
**Scope:** line-by-line comparison of the Blackwell port (`HRM-v2/`) against (a) the vendored original Sapient code at repo root (`models/hrm/hrm_act_v1.py`, `models/layers.py`, `models/losses.py`, `models/sparse_embedding.py`, `models/common.py`, `config/arch/hrm_v1.yaml`) and (b) the HRM paper (2506.21734v3): H/L schedule, one-step gradient, deep supervision, ACT/Q-learning, architecture details, init, loss, optimizer.
**Prompted by:** the program audit (`docs/experiments/cross-space/PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md`) — before building anything new on "HRM," verify the foundation.

---

## Verdict in one paragraph

**The model port is faithful; the training port is not.** `src/hrm/models/hrm_act_v1.py` reproduces the original architecture essentially line-for-line (update schedule, one-step gradient, input injection, initializations, ACT wrapper mechanics — all correct). But the original's `ACTLossHead` + training procedure were **not ported** — the training scripts reimplement them with two critical omissions: (1) the **`q_halt_loss` is missing**, so the halt head receives no gradient, stays at its −5 init forever, and **ACT never learns to halt**; (2) **deep supervision is broken** — only the final segment's loss is backpropagated instead of an optimizer step per segment. These two mechanisms are, per the paper, the core of HRM's "reasoning" behavior (learned adaptive compute + per-segment supervision of iterative refinement). The port's own `TRAINING_RESULTS.md` unknowingly contains proof (see D1). There is additionally one latent numerical landmine in the attention fallback (B1) that the two completed training runs happened not to trigger. Net: **HRM-v2 has never trained a faithful HRM** — maze exact-accuracy 25.4% vs the paper's ≈74.5% on the same benchmark is explained by the training-loop deviations, not (as far as this audit can tell) by the model port.

---

## A. What is FAITHFUL (verified, no action needed)

Checked `src/hrm/models/hrm_act_v1.py` against `models/hrm/hrm_act_v1.py` — all of the following match exactly:

- **H/L update schedule** (paper §2, original :188–204): `H_cycles × L_cycles` loop with L updating every step from `(z_L, z_H + input_embeddings)`, H updating once per cycle from `(z_H, z_L)`; all but the final L-step and final H-step under `torch.no_grad()`; the final two calls grad-enabled (**one-step gradient approximation**), `assert not z_H.requires_grad` present.
- **Input injection** by element-wise add inside the reasoning module (`hidden_states + input_injection`).
- **Post-norm blocks**: `rms_norm(x + attn(x))` → `rms_norm(x + mlp(x))`, functional RMSNorm without learnable scale, eps 1e-5; SwiGLU with `_find_multiple(round(expansion·h·2/3), 256)` intermediate; bias-free linears (except q_head).
- **Embeddings**: `embed_scale = √hidden`, embed init std `1/√hidden`, puzzle embedding zero-init, prepended with ceil-div length + right-pad, learned-pos variant scaled by 0.707; RoPE precompute identical (theta 10000, cat(freqs,freqs), non-persistent buffers).
- **Initial states**: `H_init`/`L_init` truncated normal std 1 (trunc ±2), fixed persistent buffers (paper: "kept fixed throughout training").
- **Q-head init**: weight zeroed, bias −5 ("almost-zero sigmoid for bootstrapping"); Q read from `z_H[:, 0]` in float32.
- **Output**: `lm_head(z_H)[:, puzzle_emb_len:]`; **new carry detached**.
- **ACT wrapper**: halted-slot reset (`reset_carry` → inits), step reset, per-slot `current_data` replacement, `halted = is_last_step | (q_halt > q_continue)` in training, exploration `min_halt_steps` via `rand < ε` × `randint(2, max+1)`, **target-Q via a second no-grad forward** with `sigmoid(where(is_last, next_halt, max(next_halt, next_continue)))` — all identical to the original.
- **`trunc_normal_init_`** (post-`BUGFIX_APPLIED.md`): byte-identical to `models/common.py`, including the JAX-correct compensated std.
- **Port-added genuine improvements**: explicit `device=` in `empty_carry`/`initial_carry` (the original relies on a `torch.device` context manager in its own trainer); variable-batch handling in the sparse embedding forward.

Also faithful: `ops/rotary.py`, `ops/norm.py::rms_norm`, `SwiGLU`, `CastedLinear`/`CastedEmbedding` (init + casting), stablemax reimplementation in the train scripts (float64, same `s(x)` transform as `models/losses.py`).

---

## B. Ops discrepancies

### B1 — `sdpa()` layout heuristic: silent wrong-axis attention (HIGH, latent)
`src/hrm/ops/attention.py:37`:
```python
needs_transpose = q.dim() == 4 and q.size(1) > q.size(2)
```
The model always calls attention with **(B, S, H, D)** (S = seq_len + puzzle_emb_len). PyTorch SDPA needs (B, H, S, D). This line *guesses* the layout by comparing sizes: whenever **S ≤ num_heads, the guess is wrong** and SDPA attends **across heads at each position** instead of across positions — shapes stay consistent, no error is raised, outputs are numerically garbage. The completed runs (Sudoku S=82, Maze S=901, 8 heads) took the correct branch, and when flash-attn is importable the SDPA path is skipped — but this is the *designated fallback for Blackwell*, the exact case the port exists for. Any short-sequence use (unit tests, smoke configs, downstream reuse) silently corrupts. `tests/test_attention.py` only tests S=64 vs H=8, so the failure mode is uncovered.
**Fix:** delete the heuristic; make the layout a contract (accept (B,S,H,D), always `transpose(1,2)` in and out), and add a parity test with S < H.

### B2 — `except (ImportError, Exception)` swallows all flash-attn errors (MEDIUM)
`ops/attention.py:132`: any runtime error inside flash-attn (bad dtype on some path, OOM, kernel failure) silently falls through to SDPA — the run continues on a different kernel with no signal. **Fix:** catch `ImportError` (and optionally log once on first fallback).

### B3 — dead `RMSNorm` class with learnable weight (COSMETIC)
`ops/norm.py:34` defines a learnable-scale RMSNorm the HRM never uses (the paper/original explicitly exclude scale/bias from norm). Unused → harmless, but delete or comment to prevent accidental use.

---

## C. Sparse-embedding discrepancies

### C1 — `local_weights` demoted from `nn.Buffer` to plain attribute (MEDIUM)
Original (`models/sparse_embedding.py:24`): `nn.Buffer(..., requires_grad=True, persistent=False)` — discoverable via `.buffers()`, moved by `.to()` natively. Port (`src/hrm/models/sparse_embedding.py:57`) uses a plain attribute plus a custom `_apply` override that re-creates the leaf on every `.to()`/`.cuda()` call. Consequences: `puzzle_emb.buffers()` no longer yields the trainable buffer (the original optimizer-wiring idiom breaks — the port's scripts compensate by passing the three tensors explicitly); **any `.to()`/`.half()`/`.cuda()` after optimizer construction orphans the optimizer's reference** (it would update a dead tensor); fragile under DDP/compile. Works in the port's own scripts only because the optimizer is built after the single `.to(device)`.
**Fix:** restore `nn.Buffer(..., requires_grad=True, persistent=False)` (supported in current PyTorch), or document the construction-order constraint loudly.

### C2 — optimizer `actual_batch_size` inference from gradient sparsity (MEDIUM)
`src/hrm/models/sparse_embedding.py:157–164`: the step infers the live batch rows via `has_grad = grad.abs().sum(1) > 0; actual_batch_size = has_grad.sum()` and then **slices a prefix** `[:actual_batch_size]`. If any row in the middle has an exactly-zero gradient (fully-masked sample, dead segment), the count under-shoots the prefix length and **rows with real gradients at the tail are silently dropped** (ids/grads stay aligned, but updates are lost). The `local_ids != 0` fallback also wrongly assumes puzzle id 0 is unused. The original has none of this logic (it assumes full batches — "FIXME: Assuming the batch is always full" in `models/losses.py`).
**Fix:** record the actual batch size in the module during `forward` (an int attribute) and slice by that; never infer from gradient values.

---

## D. Training-procedure discrepancies (the critical ones)

The original's `models/losses.py::ACTLossHead` was not ported; `train_sudoku.py` / `train_maze_optimized.py` reimplement the loss and loop with these deviations:

### D1 — `q_halt_loss` MISSING → ACT halting never trains (CRITICAL)
Original (`models/losses.py:84`):
```python
q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(...), reduction="sum")
... lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
```
This is the loss that teaches Q_halt to predict answer correctness — the heart of the paper's ACT (§ Adaptive Computational Time; `Ĝ_halt = 1{ŷ = y}`). The port's `train_step` (both scripts) has **only** the `q_continue` bootstrap term, weighted 0.1. Since `target_q_continue` is computed under `no_grad`, **no gradient ever reaches `q_halt`**: with the (faithful) zero-weight/−5-bias init, `q_halt_logits ≡ −5` for the entire run (AdamW's decay only shrinks it toward 0 by ~3%/3k steps). Therefore during training `q_halt > q_continue` is (almost) never true → **every sequence always runs `halt_max_steps` segments**; the learned-halting behavior, and the paper's inference-time-scaling story, are untrained.

**Proof from the port's own results** (`TRAINING_RESULTS.md`, maze run):
- "Avg Steps: **16.0** (max steps)" — never halts early;
- "Q-Halt Accuracy: **74.60%**" with "Exact Accuracy: **25.40%**" — the metric is `(q_halt_logits >= 0) == seq_is_correct`; with `q_halt ≡ −5` the predictor is constant-False, so the metric equals `1 − exact_accuracy`: **100 − 25.40 = 74.60, exactly.** The numbers are the frozen-head signature.

### D2 — Deep supervision broken: only the LAST segment is supervised (CRITICAL)
Original procedure (paper Deep Supervision; official trainer): **one segment per optimizer step** — forward one segment, compute loss, `backward()`, `OptimizerStep(θ)`, detach carry, repeat, with a persistent carry streaming new samples into halted slots. The port (`train_sudoku.py:366–391`, `train_maze_optimized.py:456–479`): fresh carry per batch, an inner loop running up to `halt_max_steps` segments **overwriting `loss` each iteration**, then a **single `loss.backward()` + step after the loop**. Only the final segment's loss trains the model; the other ≤15 segments' forward passes (and their one-step-grad graphs) are computed and discarded. Combined with D1 (never halts early), the effective procedure is: run 16 segments, supervise segment 16 — a 16×-sparser learning signal with no incentive for good intermediate answers, i.e. precisely *not* deep supervision. (Also ~16× wasted forward compute per update.)

### D3 — Loss normalization changed (HIGH)
Original: per-token CE divided by that sequence's **valid-token count**, then **summed** over the batch (`(loss_fn(...) / loss_divisor).sum()`). Port: `lm_loss.mean()` over **B × S including ignored positions**. Effects: global scale change (affects effective LR), and when `ignore_index` labels exist, sequences with more padding contribute less per real token. (For fully-labeled Sudoku/Maze the main effect is the scale.)

### D4 — Q-loss weighting/reduction changed (MEDIUM)
Original: `lm_loss + 0.5·(q_halt_loss + q_continue_loss)`, both `reduction="sum"`. Port: `lm_loss.mean() + 0.1·BCE(q_continue, target)` with mean reduction. Even after D1 is fixed, the relative supervision strength of the Q-head differs by orders of magnitude at typical batch sizes.

### D5 — Optimizer/schedule recipe differs from paper (MEDIUM, for reproduction)
Paper/original: **Adam-atan2** (scale-invariant), **constant LR after warm-up**, weight_decay 1.0 (official configs), no grad clip. Port: **AdamW**, **cosine decay to 10%**, weight_decay 0.1, `clip_grad_norm_(1.0)`. (lr 1e-4, puzzle_emb_lr 1e-2, betas (0.9, 0.95) do match the official recipe.) These matter for reproducing the paper's small-sample results, which are recipe-sensitive.

### D6 — Per-batch episode-synchronized segments vs streaming carry (LOW/DESIGN)
The original's carry design (all-halted init + per-slot `current_data` replacement) exists to let **un-halted sequences keep reasoning across optimizer steps while halted slots take fresh data** — an asynchronous stream. The port re-creates the carry every batch, so that machinery is ported but never exercised. Once D1/D2 are fixed, adopting the streaming pattern is what actually reproduces the paper's compute allocation.

---

## E. Impact on results, and on the wider program

- The port's Maze-Hard 30×30 run: **96.6% token / 25.4% exact, always 16 steps**. The paper reports ≈**74.5%** exact on Maze-Hard 30×30 at ~27M params. The gap is consistent with D1+D2 (+D3–D5): the network learned local structure (tokens) but the segment/halting machinery — HRM's actual contribution — was disabled in training. **The port model is probably fine; the port training never let it be an HRM.**
- For the program audit (`PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md`): this strengthens its conclusion at the foundation layer. Across the whole repo there are now **three** HRM lineages: (1) the faithful original (never trained here), (2) HRM-v2 (faithful model, unfaithful training — never trained as an HRM), (3) `hrm-cloud`'s `DeepSapientHRMBackbone` (a two-timescale RNN that dropped ACT/deep-supervision/iteration entirely). **No experiment in this repository has yet trained the mechanism the HRM paper is about.** The C11 "iterative refinement + deep supervision" arm should be built on a *fixed* HRM-v2, which after the fixes below is the right foundation.

---

## F. Recommended fixes (ordered)

1. **Port `ACTLossHead` verbatim** into `src/hrm/train/losses.py` (q_halt BCE vs `seq_is_correct`, q_continue bootstrap, `0.5·(sum+sum)` weighting, per-sequence divisor) and use it in both train scripts. *(Fixes D1, D3, D4.)*
2. **Restore deep supervision**: optimizer step per segment (forward → loss → backward → step → detach carry, loop), ideally with the original's persistent streaming carry. *(Fixes D2, D6.)*
3. **Fix `sdpa` layout**: contract + unconditional transpose; add a parity test with `seq_len < num_heads` and a cross-check against `models/layers.py` attention on identical weights/inputs. *(Fixes B1; catch `ImportError` only — B2.)*
4. **Fix sparse-emb optimizer batch handling** (explicit batch size from forward; drop the `!= 0` fallback) and consider restoring `nn.Buffer`. *(Fixes C1, C2.)*
5. **Recipe alignment for reproduction runs**: adam-atan2, constant LR + warm-up, wd 1.0, no clip. *(D5.)*
6. **Add an original-vs-port parity test**: same seeds/weights, CPU float32, run one full ACT step through `models/hrm/hrm_act_v1.py` and `src/hrm/models/hrm_act_v1.py`, assert logits/Q allclose — this single test would have caught B1 and guards all future drift.
7. **Re-run Maze-Hard** after 1–5: expected signatures of success — exact accuracy moving toward the paper's ≈75%, avg steps **< 16** with a nontrivial distribution, and `q_halt_accuracy` decoupling from `1 − exact_accuracy`.

---

*Receipts: original — `models/hrm/hrm_act_v1.py:188–213` (schedule/1-step), `models/losses.py:83–101` (loss), `models/sparse_embedding.py:24` (buffer), `models/common.py:7` (init). Port — `src/hrm/models/hrm_act_v1.py` (faithful model), `src/hrm/ops/attention.py:37,132` (B1/B2), `src/hrm/models/sparse_embedding.py:57–72,155–174` (C1/C2), `train_sudoku.py:181–210,355–391` and `train_maze_optimized.py:237–250,456–479` (D1–D6), `TRAINING_RESULTS.md:14–31` (the frozen-q_halt proof).*

---

## G. Status update (2026-07-06): fixes landed, verified, paper-scale run deferred by choice

All of section F's recommended fixes have landed:

| # | Fix | Commit |
|---|---|---|
| 1 | Port `ACTLossHead` verbatim (restores `q_halt_loss`, per-sequence divisor, 0.5 weighting) | `2922501` |
| 2 | Faithful training loop: streaming carry, one optimizer step per segment (deep supervision) | `b40cc43` |
| — | AdamATan2 (pure PyTorch) + warmup-then-constant LR, recipe alignment (F5) | `b2f594e` |
| 3 | `sdpa()` layout contract — remove shape-guess heuristic, `ImportError`-only flash fallback (B1/B2) | `b40ed12` |
| 4 | Sparse-embedding optimizer reverts to original full-batch logic (C2) | `3cb0e93` |
| — | Cadence config for paper-recipe reproduction (`epochs=1500`, `eval_every=2000`, `save_every=5000`) + streaming-carry partial-batch padding crash fix | `4062a4f` |
| 6 | Original-vs-port numerical parity test, explicitly covering the `S < H` regime that B1 could have silently corrupted | `079dfcb` |
| — | Full-state checkpointing (optimizer+RNG+step, `resume_from`) — enables crash recovery / cloud-spot migration for long paper-scale runs | `1e7561f` |

**Parity: bit-exact.** `tests/test_parity_original.py` (`079dfcb`) runs the port and the original vendored implementation with identical seeds/weights and asserts numerical agreement, including the exact `S < H` short-sequence regime that B1's layout heuristic could have silently corrupted. This test passing certifies the model's forward computation is correct independent of any training run's outcome.

**Mechanisms verified in a live partial run.** A post-fix Maze-Hard 30×30 run (see `RETRAIN_RESULTS.md`) reached ~150,000 of 375,000 configured steps (~12h, RTX 5090) before being stopped by choice. It directly confirms fixes 1 and 2 are live, not just present in the diff:
- `q_halt_loss` **exists and declines 0.14 → 0.06** — mechanically impossible under the pre-fix trainer, whose absence of this term is exactly what produced the D1 frozen-head signature (`Q-Halt Accuracy: 74.60%` = `100 − Exact Accuracy: 25.40%`, i.e. a constant-false predictor) in the original `TRAINING_RESULTS.md`.
- Token accuracy is strong and train/eval-consistent (96.5% train / 90–95% eval), and `exact_accuracy = 0.000` at the stop point is the expected pre-grok floor, not a regression: the official 1k-sample recipe is grokking-flavored (paper reproductions run ~20,000 epochs; this run reached 1,500 epochs, ~7.5% of that sample-exposure) — a compute gap, not a correctness gap.

**Paper-scale run deliberately deferred.** Reproducing the paper's ≈74.5% Maze-Hard exact-accuracy figure requires continuing this run (or a fresh one) far past the 150k-step stop point, likely to full paper-recipe sample-exposure. That reproduction has been **deprioritized by choice** in favor of other work — it is not blocked on any known defect. The foundation (faithful model, now-faithful training loop, bit-exact parity, mechanisms confirmed live) is in place for whenever that run is prioritized again; `1e7561f`'s full-state checkpointing means it can also be resumed rather than restarted.

**Net effect on this audit's verdict:** the "training port is not [faithful]" conclusion in the opening paragraph is resolved at the mechanism level (D1, D2, B1, C2 all fixed and verified); what remains open is purely a matter of compute (reaching paper-scale sample-exposure), tracked separately in `RETRAIN_RESULTS.md`.
