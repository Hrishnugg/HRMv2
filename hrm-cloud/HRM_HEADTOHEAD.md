# HRM head-to-head: incumbent vs a repaired cross-token-attention backbone

**Date:** 2026-07-06
**Scope:** validates two claims from `PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md` §5.2 — (a) our experiments' HRM reproduces prior published numbers by default, and (b) a variant that repairs the audit-found degeneracy (no real cross-token attention) performs same-or-better on our planning benchmarks. Harness: `hrm-cloud/continuous_prm/hrm_headtohead.py` (+ `hrm-cloud/continuous_prm/tests/test_hrm_headtohead.py`). New-file-only; `continuous_prm_common.py` and `transfer_astar_*` were not modified.

---

## The degeneracy under test

`GatedRecurrentBlock` (`continuous_prm_common.py:1066-1083`), used by every L/H block in `DeepSapientHRMBackbone`, applies `nn.MultiheadAttention` to a **length-1 sequence**: `h_norm.unsqueeze(1)` fed as query/key/value. Softmax over a single key is identically 1, so the "4-head attention" is a linear map in disguise — there is no cross-token attention anywhere in the incumbent model. All interaction across the 24-token feature bag (1 self + 6 nearest-obstacle tokens + 16 ray tokens + 1) flows through a token-by-token recurrent scan.

`RepairedHRMBackbone` (new file) keeps the two-timescale L/H structure but replaces the scan with the HRM paper's iterate-on-fixed-input pattern: `L_blocks`/`H_blocks` are standard transformer encoder blocks (`nn.MultiheadAttention` over the **full** 24-token sequence, batch_first, plus a SwiGLU FFN and RMSNorm, reusing `continuous_prm_common.RMSNorm`/`SwiGLU`) run for `n_cycles=2` on the embedded token sequence, with a pooled slow state `zH` injected into every token before each cycle and refined via `H_pool` (a `Linear` over the token-mean) after. A dedicated test (`test_repaired_model_cross_token_attention_is_real`) confirms tokens actually interact: the gradient of the pooled output w.r.t. an all-zero input spreads across more than one token position, which is architecturally impossible in a length-1-sequence attention block.

The head (3-layer MLP, GELU, zero-init final layer with bias −2.0, softplus + clamp to `[0, max_norm_residual]`) is copied verbatim from `ContinuousHeuristicModel` so `continuous_prm_providers.ScalarResidualProvider` works unchanged on the repaired model.

---

## Results

### Part A — discrete reproduction (arm (a) only)

`bench_focal.py --ckpt ckpts/avgbase__hrm__ALL_TASKS.pt --suites OOD_A128_static,OOD_A192_static --seeds 8 --budget 200 --w 1.0 --device cuda`

| Suite | Prior (`FOCAL_SEARCH_RESULTS.md` §4) | Rerun | Deviation |
|---|---|---|---|
| `OOD_A128_static` | exp_ratio 0.85, succ 0.62→0.75 | **exp_ratio 0.85, succ 0.62→0.75** | 0.00 |
| `OOD_A192_static` | exp_ratio 0.94, succ 0.75→0.75 | **exp_ratio 0.94, succ 0.75→0.75** | 0.00 |

Exact match to two decimal places on both suites and both success numbers. **Gate A: PASS.**

Retraining the discrete base for a repaired-backbone arm (b) comparison is out of the ~1h GPU budget by design (discrete HRM training in this program runs for hours, not minutes); the repaired-vs-incumbent comparison is carried entirely by the continuous side below, on the same backbone family (`DeepSapientHRMBackbone`/`GatedRecurrentBlock`) and the same audit finding.

### Part B — continuous head-to-head

Protocol: `continuous_prm_c9_transfer.load_source_base` for the incumbent; zero-shot `ScalarResidualProvider` eval on 30 deterministic TEST worlds/target (`C9.iter_test_worlds`, roadmap 192/k7, `H7.install_c7_hard_maps()`), matched A\* expansion-ratio vs Euclid at the calibrated binding budget, median over both-solved worlds — mirrors `continuous_prm_c9_transfer.py` `run_eval`'s zero-shot path (lines 281-331) and `analyze_from_raw`'s binding-budget/ratio computation exactly.

| Benchmark | Prior (C9_RESULTS.md, zero-shot) | Arm-a rerun (incumbent) | Arm-b (repaired) | Params (a / b) | Verdict |
|---|---|---|---|---|---|
| `C_hard_maze_dense` (budget 140) | 0.650 @ succ 1.00 | **0.6501 @ succ 1.000** | **0.7185 @ succ 0.933** | 2,158,529 / 1,899,905 | repaired WORSE (+0.068 ratio, −6.7pp succ) |
| `C_hard_rooms_large` (budget 56) | 0.771 @ succ 0.97 | **0.7714 @ succ 0.967** | **0.8611 @ succ 0.867** | 2,158,529 / 1,899,905 | repaired WORSE (+0.090 ratio, −10.0pp succ) |

Repaired-model params are 0.88x the incumbent's (1,899,905 vs 2,158,529) — within the ~25% capacity-matching tolerance without needing to shrink the FFN; both backbones use `hidden_dim=192, num_heads=4, num_layers=2, head_hidden=256` (read from the loaded avgbase payload's `backbone_cfg`).

**Gate B1 (reproduction):** PASS, with essentially zero deviation (0.0001 and 0.0004 on the matched ratio; succ matched to 3 decimals). The zero-shot eval is deterministic given the same seed/suite_idx/roadmap config, so this precision is expected, not lucky.

**Gate B2 (same-or-better):** FAIL on both targets. The repaired model's exp-ratio exceeds `incumbent + 0.03` by a wide margin on both (+0.068 maze_dense, +0.090 rooms_large), and success also drops by more than the 0.05 tolerance on both (−6.7pp, −10.0pp). Training converged cleanly (smooth monotonic loss 0.326→0.060 over 16 epochs, no nonfinite losses/predictions) — this is not an artifact of a broken or undertrained repaired model; it trained on the identical pooled data and recipe as the incumbent and simply generalizes worse to the two held-out hard families.

---

## Method notes

- **Data/recipe parity for arm (b):** trained on the exact same pooled scalar data as the incumbent avgbase — `BalancedTaskDataset` over the three `runs/c7_local/datasets_scalar/{C_hard_maze,C_hard_rooms,C_hard_spiral}_train_scalar.npz` files (46,079 rows total, 24×16 tokens each) — using the recipe **recorded in the incumbent checkpoint's own `train_cfg`** (`base_epochs=16, lr=2e-4, weight_decay=1e-4, batch_size=256, grad_clip=1.0, max_norm_residual=4.0`), seed 1234. This was a deliberate mid-task correction: the original brief said "epochs 10," but the checkpoint (`runs/c7_local/checkpoints/avgbase__hrm.pt`) actually recorded `base_epochs=16` — using the checkpoint's real recipe rather than the brief's assumed one gives a truer apples-to-apples comparison, which is why arm (b) trained for 16 epochs.
- **Why discrete arm (b) is out of scope:** the task explicitly scoped retraining the discrete base as out-of-budget. §5.2's degeneracy is present in the exact module Part A's checkpoint runs through: `residual_tasklora_v2.py:980` (`bench_focal.py`'s `CleanHeuristicModel` backbone) has the identical `class GatedRecurrentBlock` calling `self.attn(h_norm.unsqueeze(1), h_norm.unsqueeze(1), h_norm.unsqueeze(1))` — the same length-1-sequence pattern as `continuous_prm_common.py`'s copy. The same pattern also recurs in several other discrete-side files (`hrm_cloudFullScaleRobustFix.py:99`, `hrm_cloudMid.py:85`, `hrm_cloud_8gpu.py:78`, `lstm_hrm_comparison.py:168`; `hrm_cloud_8gpu_v2.py:92` calls `self.attn(h_norm, h_norm, h_norm)` without the unsqueeze and was not investigated further). So the continuous-side finding is evidence about the same underlying architectural choice used by the checkpoint under test in Part A, not a different one.
- **suite_idx is seed-relevant, not just a label:** `iter_matched_worlds`'s world-seed formula includes `1_000_003 * (suite_idx + 1)`, so eval used the full 3-target C9 ordering (`C_hard_maze_dense, C_hard_bugtrap, C_hard_rooms_large` → indices 0, 1, 2) and only scored indices 0 and 2, rather than re-enumerating a 2-target list (which would have silently shifted rooms_large to index 1 and generated different worlds than C9's own eval).

---

## Interpretation

This result is a **second independent confirmation of the audit's formulation thesis (§5.3)**, not a contradiction of the repair's correctness. The repaired backbone demonstrably has real cross-token attention (verified by gradient spread across token positions) and comparable capacity (0.88x params), trained on identical data with the incumbent's own recorded recipe, and converged cleanly — yet it generalizes *worse* to both held-out hard families. Read against §5.3's four lines of evidence (the accidental-MLP-control wins in C8/C9b; only the globally-receptive U-Net shows a persistent edge; non-locality is where all scalar models are weakest; architectures only separate on tasks that demand real sequential computation), this is the expected outcome under the "MLP-complete task" diagnosis: on a per-node smooth-regression task over a small, pre-digested, non-sequential feature bag, adding real cross-token attention is not free — it adds capacity to overfit correlations in the 24-token bag that don't transfer to unseen obstacle geometries, without adding any signal a well-tuned MLP-equivalent (the incumbent's linear-map-in-disguise "attention") couldn't already extract. It does not, on its own, disprove that HRM-as-designed (iterative refinement on a *fixed, non-serialized* input like a map or graph, with deep supervision and adaptive depth — §5.5's binding conditions) could shine; it shows that bolting real attention onto the *existing* formulation — a serialized bag of local geometric features — doesn't help, and here measurably hurts. The fix implied by the audit is at the formulation layer (feed the model the map/graph and let it iterate), not the block-internals layer this experiment tested.

---

## GPU time

Part A: ~13 min. Part B arm-a eval: ~4 min. Part B arm-b train: ~1 min. Part B arm-b eval: ~3 min. **Total: ~21 min of the ~60 min budget.**
