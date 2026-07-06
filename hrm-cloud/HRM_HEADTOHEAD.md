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

---

## Remediation sweep (v2)

**Date:** 2026-07-06. **Scope:** a follow-up audit of arm (b)'s loss (above) found four plausible integration defects independent of "does real cross-token attention help." This section builds a remediated backbone (`RepairedHRMBackboneV2`, `hrm-cloud/continuous_prm/hrm_headtohead_v2.py`), sweeps four training recipes against it plus an epochs-matched control, and asks: was arm (b)'s loss an **integration artifact** (fixable) or **architectural** (the audit's formulation thesis, restated above)? New-file-only, imports `hrm_headtohead.py`'s arm-a incumbent loader, eval harness (`eval_provider_on_target`/`eval_arm`, unmodified), `TARGETS`/`BINDING_BUDGET`/`SOURCE_DIR`, and `pooled_scalar_arrays`. `continuous_prm_common.py`, `transfer_astar_*`, and `hrm_headtohead.py` were not touched.

### The four defects

- **I1 (init):** the original `RepairedHRMBackbone._init_weights` applied a blanket `std=0.01` normal-init to every `nn.Linear`, copied verbatim from the incumbent's `DeepSapientHRMBackbone._init_weights`. That init suits the incumbent's small-signal gated-recurrent design (its residual gate blends a tiny perturbation into a persistent state); on a fresh transformer stack it silences attention and FFN outputs near-completely at init, well below the signal a `smooth_l1_loss` gradient can efficiently climb out of in 16 epochs. **Fix:** `xavier_uniform_` on `embed`, SwiGLU's three `Linear`s, and MHA `out_proj` (PyTorch's own `nn.MultiheadAttention.__init__` already xavier-inits `in_proj_weight` by default — V2 leaves it alone rather than re-initializing it). The head's zero-init final layer + bias −2.0 is kept unchanged (verbatim from `ContinuousHeuristicModel`) — this was never part of the problem; I1 only targets the backbone.
- **I2 (collapsed two-timescale):** the original ran `H_blocks` — a **second full token-level transformer stack**, attention over all 24 tokens again — every cycle as the "H update," then mean-pooled its output through a `Linear` (`H_pool`). That is not a slow/pooled timescale; it is a second L stack with an extra linear on top. The intended two-timescale separation (fast per-token L vs. slow pooled-summary H) never existed in the original arm (b). **Fix:** `H_blocks` is dropped entirely. Each cycle: `hs = h + zH.unsqueeze(1)` (inject pooled state into every token) → `L_blocks` over tokens (cross-token attention, unchanged mechanism) → `h = hs` → `zH = zH + H_mlp(rms_norm(concat[zH, pooled]))`, where `H_mlp` is a 2-layer MLP (`hidden_dim` wide, GELU) operating purely on the two pooled `(B, hidden_dim)` vectors — no token dimension anywhere in the H update. `H_blocks`' capacity is folded into `+1` `L_blocks` layer (`num_layers+1`) so cross-token compute isn't simply deleted, only the mislabeled "H" stack.
- **I3 (mean-pool readout):** the original seeded `zH` via `h.mean(dim=1)` and re-derived the pooled signal for `H_pool` via `h.mean(dim=1)` again every cycle — treating the state/goal token (token 0, which alone encodes `dx, dy, euclidean-dist, position, clearance, line-of-sight, corridor-blockage` relative to the goal; see `make_feature_sequence`) as no more informative than any of the 23 obstacle/ray tokens, diluting the single most task-relevant signal 1-in-24. **Fix:** the per-cycle pooled readout is `rms_norm(h[:, 0])` (state-token readout, not mean-pool), and the backbone's final return is `rms_norm(zH)` before the head (previously the raw un-normalized `zH`).
- **I5 (recipe assumed, not swept):** arm (b) reused the incumbent's recorded recipe (`lr=2e-4`, `base_epochs=16`) without questioning whether a from-scratch transformer needs a different one; final loss 0.060 vs. the incumbent's 0.0498 on identical data left underfitting on the table as an unexamined confound alongside I1–I3.

### Variants

All four variants train on the **identical pooled data** as the incumbent and the original arm (b) — `hrm_headtohead.pooled_scalar_arrays()` (`BalancedTaskDataset` over the same three C7 `datasets_scalar/*.npz` files), seed 1234 — via `hrm_headtohead_v2.train_variant`, a generalization of `hrm_headtohead.train_repaired_model`'s loop (`AdamW`, `smooth_l1_loss`, grad-clip 1.0) with a per-variant epoch count, learning rate, and optional linear warmup.

| Variant | Backbone | Epochs | LR | Warmup | Final train loss | Params (ratio to incumbent) | Train wall |
|---|---|---|---|---|---|---|---|
| — incumbent (`avgbase__hrm`) | `DeepSapientHRMBackbone` | 16 | 2e-4 | — | 0.0498 | 2,158,529 (1.00x) | — |
| — original arm-b (unfixed) | `RepairedHRMBackbone` | 16 | 2e-4 | — | 0.060 | 1,899,905 (0.88x) | ~1 min |
| **V1** | `RepairedHRMBackboneV2` (I1+I2+I3) | 16 | 2e-4 | none | **0.04446** | 1,538,561 (0.71x) | 58.8s |
| **V2** | `RepairedHRMBackboneV2` (I1+I2+I3) | 16 | 5e-4 | 100 steps | **0.03661** | 1,538,561 (0.71x) | 61.6s |
| **V3** | `RepairedHRMBackboneV2` (I1+I2+I3) | 32 | 2e-4 | none | **0.03690** | 1,538,561 (0.71x) | 133.3s |
| **V4** (control) | `RepairedHRMBackbone` (ORIGINAL, unfixed) | 32 | 2e-4 | none | 0.05171 | 1,899,905 (0.88x) | 142.0s |

V2's params are 0.71x the incumbent's — H_blocks' removal (I2) more than offsets the +1 folded L layer, so V1–V3 land comfortably inside the ~25% capacity-matching tolerance (well under it, in fact — V4's 0.88x is the tighter match to the incumbent's 2.16M). All three fixed variants (V1–V3) beat both the incumbent's and the original arm-b's final train loss; V2 (higher LR + warmup) is the best of the sweep. GPU: all four variants trained back-to-back, ~6.6 min total.

### Eval (all four variants evaluated — budget allowed it)

Same protocol as Part B above: 30 deterministic TEST worlds/target, roadmap 192/k7, matched A\* expansion-ratio vs. Euclid at the calibrated binding budget (140 / 56), median over both-solved worlds.

| Variant | `C_hard_maze_dense` ratio @ succ | `C_hard_rooms_large` ratio @ succ | Eval wall |
|---|---|---|---|
| **Incumbent** | 0.6501 @ 1.000 | 0.7714 @ 0.967 | — |
| **Original arm-b** (unfixed, 16ep) | 0.7185 @ 0.933 | 0.8611 @ 0.867 | — |
| **V1** (fixed, incumbent recipe) | 0.6311 @ 1.000 | **1.0417 @ 0.733** | 184.2s |
| **V2** (fixed, lr5e-4+warmup) | 0.7253 @ 1.000 | **1.1277 @ 0.900** | 199.0s |
| **V3** (fixed, 32 epochs) | 0.7478 @ 0.967 | **1.0573 @ 0.700** | 190.6s |
| **V4** (control: unfixed, 32ep) | 0.6860 @ 1.000 | 0.8571 @ 0.867 | 186.6s |

### Gate verdicts

**R1 — does any remediated variant reach ≤ incumbent+0.01 exp-ratio at succ within 0.03 on BOTH targets? FAIL, all variants.** V1 is the closest on `maze_dense` alone (0.6311 ≤ 0.6601, succ exact match) but every fixed variant fails `rooms_large` by a wide margin (best case V1: ratio 1.0417 vs. the 0.7814 ceiling, succ off by 23pp). No variant clears both targets simultaneously.

**R2 — does any variant BEAT the incumbent (strictly lower ratio, ≥ succ) on BOTH targets? FAIL, all variants.** V1 strictly beats the incumbent on `maze_dense` alone (ratio 0.6311 < 0.6501, succ 1.000 = 1.000) — the only single-target strict win anywhere in this sweep, remediated or not — but the same model is far worse than the incumbent on `rooms_large` (ratio 1.0417 vs. 0.7714, succ 0.733 vs. 0.967). No variant wins on both.

**R3 — V4-vs-V3 (and vs. original arm-b): how much of the original gap was underfit vs. integration?** **Mixed, and the honest answer cuts against the loss numbers.** By train loss alone, the fixes (I1–I3) clearly help: V3 (fixed + 2x epochs, loss 0.0369) beats V4 (unfixed + 2x epochs, loss 0.0517), and both beat the original arm-b (unfixed, 16ep, loss 0.060) — so *more epochs alone* recovers part of the gap (0.060→0.0517), and *fixes+more epochs* recovers more (0.060→0.0369). If train loss were the only signal, the verdict would be "mostly integration, with a side of underfit." **But the held-out-generalization numbers say the opposite on `rooms_large`:** V4 (epochs-only, no architecture fix) closes most of the original gap there (ratio +0.090 → +0.086 vs. incumbent) while V3 (epochs + fixes) makes it dramatically *worse* (+0.286 vs. incumbent) — worse than the original 16-epoch unfixed arm-b itself (+0.090). Every I1–I3-fixed variant (V1, V2, V3) pushes `rooms_large`'s matched ratio **above 1.0** — meaning on the median both-solved world, these models are net-harmful as an additive heuristic, expanding *more* nodes than plain Euclidean distance would with zero learning. The unfixed control (V4) never crosses 1.0, matching the incumbent's and original arm-b's pattern. On `maze_dense`, by contrast, the fixes help succ (V1/V2 reach 1.000, matching incumbent) even though the best-loss variant (V2) still has a higher ratio than the incumbent.

### Conclusion: not an integration artifact — the loss curve was misleading

**The four integration defects are real bugs and V2's fixes make the model measurably easier to fit (lower train loss, better `maze_dense` success) — but fixing them does not close the head-to-head gap, and on `rooms_large` it makes generalization actively worse, not better.** This is not the outcome "just an integration bug" would predict (which is: fix it, the loss and the eval numbers move together in the same direction). Instead the fixes decouple loss from held-out generalization: better-fit models transfer worse to `rooms_large`'s obstacle geometry specifically, while the *original*, integration-buggy, worse-fitting arm (b) generalizes closer to (though still worse than) the incumbent on that same target. That asymmetry — real cross-token attention plus a properly-scaled init and a non-collapsed two-timescale structure making the model *fit the 24-token training bag better while generalizing worse to unseen rooms geometry* — is exactly the overfitting signature the original doc's Interpretation section predicted from the audit's "MLP-complete task" diagnosis: added capacity to model correlations in a small, pre-digested, non-sequential feature bag, with no guarantee those correlations are the ones that transfer. R2's one bright spot (V1 strictly beats the incumbent on `maze_dense` alone) is a genuine, reproducible result — but it is a single-target win with a same-model loss on the other target, not a general improvement, so it does not overturn Part B's original FAIL verdict. **Verdict: the loss was architectural (or more precisely, a training-formulation interaction — real attention overfits this particular feature-bag task on this particular held-out family), not primarily an integration artifact.** The four defects were worth fixing on their own engineering merits (I1–I3 are objectively better-designed than what they replaced) and the fixed model is a strictly better implementation of "give the backbone real cross-token attention" — but that better implementation still loses the head-to-head, for a different and more specific reason (rooms_large overfit) than originally suspected (undertrained/miscalibrated). This narrows, rather than resolves, the open question from the original doc's Interpretation section: it is not that real attention never helps here (`maze_dense` succ improved), it is that it helps unevenly across held-out families in a way an incumbent-recipe rerun could not have surfaced without this sweep.

### GPU time (v2 sweep)

Train: V1 58.8s + V2 61.6s + V3 133.3s + V4 142.0s ≈ 6.6 min. Eval (all four, 30 worlds × 2 targets each): V1 184.2s + V2 199.0s + V3 190.6s + V4 186.6s ≈ 12.7 min. **Total: ~19.3 min of the 30 min budget.**
