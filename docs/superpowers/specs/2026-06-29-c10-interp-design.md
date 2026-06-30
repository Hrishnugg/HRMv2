# C10 — Parameter-space LoRA Interpolation (zero-shot transfer): Design

**Date:** 2026-06-29
**Status:** design (approved; revised for a bracketing source grid); spec for implementation.
**Branch:** `perf/eval-speedup`
**Builds on:** C9/C9h (per-family LoRA adapters, `train_scalar_lora`, matched recipe), the C3/C4 descriptor/expert lineage (`task_descriptor`, RBF weighting), C5/C7 hard-map generators (varied at runtime), and the C7 scalar `avgbase` source base. Reused FROZEN. Related: `hrm-cloud/continuous_prm/C9_RESULTS.md`, `C9H_RESULTS.md`, `continuous_prm_experiment_ladder_repo_coupled.md` (earmarked this as "C9: parameter-space LoRA interpolation").

---

## 1. Goal & motivation

Can we reach an **unseen** task family with **zero labeled target data** by *composing* the adapters we already know how to train? The earmarked idea is **parameter-space interpolation**: train one LoRA expert per *source* family, then for a new family merge the experts' weight-deltas by task-descriptor similarity. C10 tests this against **prediction-space** mixing (C4) and simpler baselines.

Crucially, C10 uses a **bracketing source grid** so the held-out targets sit *inside* the source descriptor hull — making this **genuine interpolation**, not extrapolation. (The C7 held-out families are OOD extremes, so they would be extrapolation; C10 instead defines its own source/target grid.)

"Zero target data" = **no solved/labeled target worlds**. The target's *descriptor* (geometry knobs from `task_descriptor`; no solving) IS used to compute interpolation weights, exactly as in C4 — this is zero-label transfer driven by task metadata.

**Claim under test:** RBF-weighted **weight-space** merge of source-family LoRA experts beats the zero-shot pooled base (and nearest / uniform) on *interior* held-out families with zero target labels — and reveals whether weight-space merging beats **prediction-space** mixing.

## 2. Scope

Local (RTX 5090), **new-file-only**: `continuous_prm_c10_interp.py` (+ a runtime family-grid installer, in the same module). Reuses C9/C9h/C5/C7 (imported, not modified). No edits to `continuous_prm_common.py` / `transfer_astar_*` (user WIP). Scalar backbones only (HRM, ON-LSTM); no field; RBF/nearest/uniform weighting only; **no K-grid** (zero-shot).

## 3. Bracketing source/target family grid

A runtime installer (`install_c10_families()`, composing on `install_c7_hard_maps()` like C8) registers **HARD_MODE variant** AnchorSpecs via `dataclasses.replace` on the C7/C5 hard specs, so the encoder regime matches the `avgbase` base. Two clean continuous axes (each axis is one generator varied along one knob), giving **8 source families** and **3 interior targets**:

**Maze-density axis** (vary `gap_width_frac` ↓ + `extra_clutter_range`/`obstacle_count_range` ↑ on `C_hard_maze`):
- Sources: `C10_maze_d0` (gap 0.18, clutter 2–4), `C10_maze_d1` (gap 0.15, clutter 4–7), `C10_maze_d2` (gap 0.13, clutter 7–11), `C10_maze_d3` (gap 0.11, clutter 10–14).
- Interior target: `C10_maze_tgt` (gap 0.14, clutter 6–9) — between d1 and d2.

**Rooms-scale axis** (vary `side_len` on `C_hard_rooms`):
- Sources: `C10_rooms_s10` (1.0), `C10_rooms_s20` (2.0), `C10_rooms_s30` (3.0), `C10_rooms_s40` (4.0).
- Interior targets: `C10_rooms_t25` (2.5, between s20/s30), `C10_rooms_t35` (3.5, between s30/s40).

All 8 sources form ONE pooled source set; the RBF over descriptors is expected to weight maze-sources for a maze target and rooms-sources for a rooms target (a built-in test of descriptor selectivity). Source count and axes are config knobs (`--source-families`, `--target-families`) so the grid can be widened later.

**Bracketing is verified, not assumed** (Gate G0b): for each target, each descriptor dimension must satisfy `min_k z_k[d] ≤ z_T[d] ≤ max_k z_k[d]` (interior on every axis); a warning is logged for any dimension where it fails.

## 4. Source experts & descriptors

For each source family S (8) and backbone B ∈ {hrm, onlstm}: collect `N_src=48` worlds (`C.collect_task_dataset`, `C9.SCALAR_NODES_PER_WORLD`), train a **bounded** LoRA expert on `avgbase__{B}` via `C9h.train_scalar_lora(rank=8, alpha=1.0, bounded=True, epochs=10, lr=2e-4)`. → 16 source experts. Store each source's **descriptor centroid** `z_S` = mean `C.task_descriptor(spec, world.obstacles)` over its worlds. (16 quick LoRA fine-tunes.)

## 5. Descriptor + RBF weighting

For target T: `z_T` = mean `task_descriptor` over T's TEST worlds (geometry only). RBF weights over the 8 source centroids: `w_k(z_T) = softmax_k(−‖(z_T − z_k)/s‖² / (2σ²))`, `s` = per-dim std across source centroids (epsilon-floored), `σ` = `--rbf-sigma` (default 1.0), weights sum to 1. `nearest` = one-hot argmin; `uniform` = `1/8` each.

## 6. Arms (per target × backbone; ZERO target labels)

| Arm | Mechanism |
|---|---|
| `zero_shot` | pooled `avgbase__{B}` (the bar to beat) |
| `nearest` | nearest-descriptor source expert, baked |
| `uniform_wmerge` | uniform-weight weight-space merge of all 8 |
| `rbf_wmerge` | **RBF-weighted weight-space merge** (the earmarked C10) |
| `rbf_pmix` | **RBF-weighted prediction-space mix** (C4-style head-to-head) |
| `euclid`, `oracle` | baseline / ceiling |

C9 few-shot numbers cited in the writeup as the "with target data" reference ceiling (not a live arm).

## 7. Mechanisms

- **Weight-space merge (baker):** for weights `w`, build a fresh `ContinuousHeuristicModel`, load `avgbase__{B}`; for each LoRA-target Linear, compute `Δ = Σ_k w_k · (alpha/r) · B_k@A_k` from the experts' A/B (load each expert with `apply_lora`, read `.parametrizations.weight.0.{A,B}`), add `Δ.reshape(W.shape)` to the base weight in-place. Save a **plain** merged checkpoint (no LoRA) with backbone/feature/train cfg + `max_norm_residual=4.0`. Eval via `P.ScalarResidualProvider`. (Summed rank-r deltas aren't low-rank → bake into W exactly.)
- **Prediction-space mix (provider):** `_PredMixProvider(HeuristicProvider)` holding the 8 expert models + `w`; `node_h` = `euclid + side_len · clip(Σ_k w_k ŷ_k, 0, 4.0)`. (For efficiency it may keep only the top-m experts by weight; default m=8.)

## 8. Architecture

New `continuous_prm_c10_interp.py`: `install_c10_families()`, `C10Config`, modes `train`/`eval`/`analyze`/`full` + CLI. Reuses `C9.load_source_base`, `C9.iter_test_worlds`, `C9._binding_budget_for`, `C9.RAW_COLS`, `C9._parse_csv`, `C9.SCALAR_NODES_PER_WORLD`, the C9 analyze helpers; `C9h.train_scalar_lora`; `C.collect_task_dataset`/`task_descriptor`/`build_model`/`apply_lora`/`safe_load_state`; `P.ScalarResidualProvider`/`EuclidProvider`/`OracleProvider`/`run_world_arms`. New: family-grid installer, descriptor/RBF weighting + bracketing check, weight-merge baker, `_PredMixProvider`, arm orchestration. Analyze: thin C10 variant of C9's (arm names as the "method" axis; no K; comparisons = the §9 set).

## 9. Grid, metrics, gates

3 interior targets × 2 backbones; eval 30 TEST worlds/target; **calibrate binding budgets fresh** for the new target families (reuse the C8/C9 binding-budget convention: lowest budget with euclid success ≥ 0.05 over a small calib grid) since these are new specs not in `c7_local/calibration.json`. Stats identical to C9 (matched A\* exp-ratio vs euclid + success; bootstrap CI; McNemar+BH).

Pre-registered comparisons: (1) `rbf_wmerge` vs `zero_shot` (interpolation helps?); (2) `rbf_wmerge` vs `uniform_wmerge` vs `nearest` (descriptor-weighting value); (3) `rbf_wmerge` vs `rbf_pmix` (weight vs prediction space); (4) vs euclid/oracle.

**Gates:**
- **G0 (merge correctness):** unit tests — RBF weights sum to 1, →one-hot as σ→0; weight-space bake with `w=e_k` reproduces expert-k's forward (tol); uniform bake == mean of deltas; `_PredMixProvider` with `w=e_k` == expert-k's `node_h`.
- **G0b (bracketing):** each interior target's descriptor is per-axis within [min,max] of the source centroids (logged; warn on violation).
- **G1 (base sanity):** `zero_shot` beats euclid on the interior targets.
- **G2 (directional):** the comparisons reported with CIs; `rbf_wmerge` ≥ `zero_shot` and ≥ `uniform`/`nearest` (or a clear, honest negative).

## 10. Scale & risks

Local, ~1.5 h (16 source experts + small merges + eval + a quick calibrate). Risks: (a) **weight-merge off-distribution** — summed deltas baked into W is exact, but large merged deltas could push the model off; bounded residual (4.0) guards integration. (b) **descriptor selectivity** — if maze/rooms descriptors overlap, RBF may misweight across axes; the per-axis bracketing check + the uniform/nearest baselines make this visible. (c) **two axes only** — maze-density + rooms-scale; bugtrap-style structural transfer is out of scope (noted as future).

## 11. Acceptance criteria

- `continuous_prm_c10_interp.py` runs `--mode full --scale local` end-to-end: family-grid install, 16 source experts + descriptors, fresh target calibration, per-target/backbone arm eval, comparisons MD (weight-vs-prediction + interpolation-vs-baselines), significance MD, manifest, and the bracketing report.
- G0 merge-correctness unit tests green; G0b bracketing verified; G1/G2 pass (or any miss explained).
- `C10_RESULTS.md` written and cross-linked from `C9_RESULTS.md`.
- WIP files never staged; C9/C9h/C5/C7/common unchanged.
