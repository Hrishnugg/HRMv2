# C9-hardening (C9h) — Matched-compute + Field-backbone Transfer: Design

**Date:** 2026-06-29
**Status:** design (approved); spec for implementation.
**Branch:** `perf/eval-speedup`
**Builds on:** C9 (`continuous_prm_c9_transfer.py`, few-shot transfer) — reused and kept FROZEN; C6/C7 field stack (`UNetField`, `ValueFieldProvider`, `field_node_heuristic`); C3 bounded-residual LoRA. Related: `hrm-cloud/continuous_prm/C9_RESULTS.md`, `C7_RESULTS.md`.

---

## 1. Goal & motivation

C9 established that few-shot transfer works, with a LoRA-vs-full-FT crossover. But C9's "LoRA" confounded four things at once — **low-rank structure + bounded tanh residual + fewer epochs (8 vs 10) + lower LR (1.5e-4 vs 2e-4)** — so the headline sample-efficiency claim is not a clean comparison, and it only tested the scalar backbones (not the C7/C8-strongest field U-Net). C9h makes the claim rigorous:

1. **Matched compute** — every trained adaptation arm uses one identical recipe (same epochs, LR, loss). The only differences are the two axes of interest.
2. **Disentangle the bound** — run **bounded-LoRA** and **unbounded-LoRA** separately, so we learn whether LoRA's low-K robustness comes from the *low-rank structure* or the *tanh clamp*.
3. **Best backbone** — extend the comparison to the **field U-Net** via a new **conv-LoRA**, so the parameter-efficiency story is tested where the heuristic is strongest.

**Claim under test:** at matched compute, on three held-out hard families and three backbones, characterize the adaptation curves of {zero-shot, bounded-LoRA, unbounded-LoRA, full-FT, from-scratch} — isolating low-rank-vs-full-rank and bounded-vs-unbounded, and confirming whether the crossover holds on the field U-Net.

## 2. Scope

C9h is local (RTX 5090) and **new-file-only**: it adds `continuous_prm_c9h_transfer.py` and reuses C9 (imported, not modified) + the C6/C7 field stack. C9's module/results stay frozen. No edits to `continuous_prm_common.py` or `transfer_astar_heuristic_clean_parallel_fixed.py` (user WIP). YAGNI: no cluster preset, no new suites, no parameter-space interpolation (separate follow-up).

## 3. Arm matrix & grid

**Backbones (3):** scalar `hrm`, scalar `onlstm`, field `unet`.
**Methods (4 trained + zero-shot):** `zero_shot` (K=0, the source base), `lora_bounded`, `lora_unbounded`, `full_ft`, `scratch`.
**Matched recipe (all trained arms):** epochs `E=10`, lr `2e-4`, smooth-L1 loss, grad-clip as in `train_avgbase`. The ONLY differences across arms: low-rank (LoRA) vs full-rank (full_ft/scratch), bounded vs unbounded residual, and base-init (lora/full_ft) vs fresh (scratch).
**Targets (3 held-out):** `C_hard_maze_dense`, `C_hard_bugtrap`, `C_hard_rooms_large`.
**K-grid:** `{1, 4, 16}` (3 points spanning C9's observed crossover).
**Adapt seeds:** `3` per (target, K) — CIs pool over (seed × world).
**TEST:** 30 worlds/target, disjoint from ADAPT, matched across all arms; per-target binding budgets reused from `c7_local/calibration.json` (C9 convention: lowest band budget with euclid success ≥ 0.05).

Trained-model count: 3 backbones × 4 methods × 3 targets × 3 K × 3 seeds = **324** (+ zero-shot, untrained). Estimated ~6–8 h local.

## 4. Architecture & reuse

New module `continuous_prm_c9h_transfer.py` imports C9 (`import continuous_prm_c9_transfer as C9`) and the field stack, reusing wherever possible:

- **Reuse from C9 (unchanged):** `load_source_base`, `train_scalar_model` (full_ft/scratch for scalar), `iter_test_worlds`, `world_fingerprint`, `adapt_seed`, `load_scalar_provider`, `RAW_COLS`, `_binding_budget_for`, `analyze_from_raw` (extended call), `_parse_csv`/`_parse_ints`.
- **Reuse from C/C6/C7:** `C.train_expert` (scalar LoRA) — but C9h needs matched epochs/LR and an unbounded option, so scalar LoRA is produced by a **C9h LoRA trainer** (see §6) rather than `train_expert`, to control epochs/LR/bound uniformly. `C.collect_task_dataset` (scalar K-world npz). `C6.make_heatmap_example`, `C6.build_model("unet", in_channels)`, `C6.field_node_heuristic`, `C6.checkpoint_path`; `P.ValueFieldProvider`; the C6 field training loop (mirrored, see §7).

New primitives added by C9h: **conv-LoRA** (§5), a **unified matched-compute LoRA trainer** for scalar + field (§6), and **field transfer arms** (collect/train/provider, §7).

## 5. conv-LoRA (new primitive)

For a `nn.Conv2d` with weight `W` of shape `[out, in, kh, kw]`, add a low-rank update applied to the flattened weight matrix `W2d = W.reshape(out, in*kh*kw)`:

- Parametrize: `W2d_eff = W2d + scale · (B @ A)`, where `A ∈ R[r, in*kh*kw]`, `B ∈ R[out, r]`, `scale = alpha / r`. `A` init small (e.g. 0.01·randn), `B` init zero (so the adapter starts as identity — first forward = base).
- Implement via a `torch.nn.utils.parametrize` registration on the conv's `weight` (forward reshapes back to `[out,in,kh,kw]`), mirroring how `common.SingleAdapterLoRA` wraps Linear. Base `weight` frozen; only `A`/`B` trainable.
- **Bounded variant:** for field, "bounded residual" is on the predicted heatmap output, not the weight; the bound is applied at the heuristic-integration step (`field_node_heuristic` already clips/scales the residual via `max_norm_residual`-equivalent). So for field, `lora_bounded` vs `lora_unbounded` differ in whether the field output residual is clamped at integration; the conv-LoRA weight adapter itself is identical. (Document this asymmetry: scalar's bound is the tanh on the per-node residual head; field's bound is the residual clip in `field_node_heuristic`.)

`apply_conv_lora(unet, rank, alpha)` wraps the U-Net's `DoubleConv` conv layers (the encoder/decoder convs); `set_conv_lora_trainable(unet)` freezes base + unfreezes A/B. A loader applies `apply_conv_lora` before `load_state_dict` for LoRA field checkpoints (parallel to C9's `load_scalar_provider`).

## 6. Matched-compute LoRA trainer (scalar + field)

A single recipe, two backends:
- **Scalar LoRA** (`hrm`/`onlstm`): build model, load base, `C.apply_lora(rank, alpha)`, `C.set_lora_trainable`; train all-trainable (= the LoRA A/B) at `E=10`, lr `2e-4`, smooth-L1; `lora_bounded` keeps the model's `max_norm_residual` tanh bound (the default), `lora_unbounded` sets the residual bound to `inf` (no clamp) for that run. Save with `lora_rank`/`alpha`/`bounded` flags in payload.
- **Field LoRA** (`unet`): build U-Net, load base, `apply_conv_lora(rank, alpha)`, `set_conv_lora_trainable`; train at the same `E`/lr/loss on the field K-world dataset; `lora_bounded`/`lora_unbounded` toggles the integration-time residual clamp recorded in the payload.
- **full_ft / scratch** reuse the matched recipe with all params trainable (full_ft loads base, scratch fresh) — scalar via C9 `train_scalar_model` (already matched-capable via its `train_cfg.base_epochs`/lr), field via the new field trainer (§7).

All arms write a checkpoint payload carrying enough to reload (backbone/feature/in_channels/cfg + lora flags + bounded flag).

## 7. Field transfer arms

- **Source base:** `runs/c7_local/checkpoints/c6_heatmap__unet.pt` (C7 field U-Net trained on maze/rooms/spiral; in_channels=8).
- **Collect:** K-world field dataset via `C6.make_heatmap_example(world, grid_size)` over K target worlds (seeded like C9's `adapt_seed`; disjoint from TEST via the C9 fingerprint convention). Cache per (target,K,seed).
- **Field trainer** (`train_field_model`): mirror C6's per-model training loop (the loop inside `C6.run_train`) — build U-Net (optionally `apply_conv_lora`), optionally load base (full_ft/lora) vs fresh (scratch), train `E` epochs at lr on the K-world field dataset, write to a unique path. (C6.run_train is fixed-path/skip-if-exists/from-scratch, so a mirrored loop is required — exactly analogous to why C9 wrote `train_scalar_model`.)
- **Field provider load:** `ValueFieldProvider` over the loaded U-Net (apply_conv_lora before load for LoRA ckpts), integrated via `field_node_heuristic`; `.name` set to the unique arm key for eval disambiguation.

## 8. Eval & analyze (reuse C9)

Reuse C9's matched-eval structure: per target, build providers {euclid, oracle, per-backbone zero_shot, all adapted arms}, iterate the shared TEST worlds, `P.run_world_arms`, write `RAW_COLS` rows + merged raw CSV. The field providers plug into the same `run_world_arms` (they expose `node_h`). Reuse `analyze_from_raw` (curves/comparisons/significance) — the method axis now has 4 trained methods; the comparisons MD adds the **bounded-vs-unbounded** column and reports per-backbone curves including `unet`.

## 9. Metrics, comparisons, gates

Primary metric + stats identical to C9 (matched A* expansion-ratio vs euclid + success; bootstrap CI; McNemar+BH; single binding budget per target; seed pooling). New pre-registered comparisons:
1. **Matched LoRA vs full-FT** (per backbone, per K) — does the C9 crossover survive equal compute?
2. **Bounded vs unbounded LoRA** — is the low-K robustness from the bound or the low-rank?
3. **Field (U-Net) vs scalar** — does the crossover hold on the strongest backbone?
4. **vs from-scratch / vs zero-shot / vs euclid** — as in C9.

**Gates:**
- **G0 (conv-LoRA + wiring):** unit tests — conv-LoRA apply→save→load round-trips (strict load); a fresh conv-LoRA leaves outputs unchanged (B=0 init) then changes them after a step; base weights frozen / only A,B trainable; bounded≠unbounded produce different heuristics; ADAPT⊥TEST disjointness; matched recipe applied (epochs/lr identical across arms).
- **G1 (base sanity):** field `zero_shot` reproduces the C7 field-OOD direction (U-Net beats euclid on held-out); scalar zero_shot matches C9.
- **G2 (directional):** transfer ≫ scratch at low K; the matched LoRA-vs-FT and bounded-vs-unbounded contrasts are coherent and reported with CIs.

## 10. Scale & risks

Local; ~324 models + eval, ~6–8 h. Risks: (a) **conv-LoRA correctness** — new primitive; mitigated by the G0 round-trip/output-change/frozen-base unit tests before any run. (b) **field training cost** — heavier than scalar; mitigated by K≤16, the C8 static-base cache, and the 3×3×3 grid. (c) **field bound semantics** — field's "bound" is at integration, not in the adapter; documented in §5 and the results.

## 11. Acceptance criteria

- `continuous_prm_c9h_transfer.py` runs `--mode full --scale local` end-to-end, writing per-target adaptation curves (4 methods × 3 backbones), comparisons MD (incl. bounded-vs-unbounded + field), significance MD, and a manifest.
- G0/G1/G2 pass (or any miss explained).
- `C9H_RESULTS.md` written and cross-linked from `C9_RESULTS.md`.
- conv-LoRA unit tests green; matched recipe verified; WIP files never staged; C9 module unchanged.
