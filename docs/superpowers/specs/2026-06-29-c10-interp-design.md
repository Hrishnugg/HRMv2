# C10 — Parameter-space LoRA Interpolation (zero-shot transfer): Design

**Date:** 2026-06-29
**Status:** design (approved); spec for implementation.
**Branch:** `perf/eval-speedup`
**Builds on:** C9/C9h (per-family LoRA adapters, `train_scalar_lora`, matched recipe), the C3/C4 descriptor/expert lineage (`task_descriptor`, RBF weighting), and the C7 scalar `avgbase` source base. Reused FROZEN. Related: `hrm-cloud/continuous_prm/C9_RESULTS.md`, `C9H_RESULTS.md`, `continuous_prm_experiment_ladder_repo_coupled.md` (which earmarked this as "C9: parameter-space LoRA interpolation").

---

## 1. Goal & motivation

C9/C9h showed per-family adaptation needs *some* target data (LoRA from K worlds; full-FT from more). C10 asks the deeper transfer question: can we reach an **unseen** family with **zero labeled target data** by *composing* the adapters we already know how to train? The earmarked idea is **parameter-space interpolation**: train one LoRA expert per *source* family, then for a new family merge the experts' weight-deltas by task-descriptor similarity. C10 tests this and pits it against **prediction-space** mixing (the C4 mechanism) and simpler baselines.

"Zero target data" = **no solved/labeled target worlds** are used. The target's *descriptor* (geometry knobs — side length, obstacle count/density, narrow-passage indicator; from `task_descriptor`, no solving) IS allowed, exactly as in C4 — this is zero-label transfer driven by task metadata.

**Claim under test:** RBF-weighted weight-space merge of source-family LoRA experts beats the zero-shot pooled base (and nearest-expert / uniform-average) on held-out families, with zero target labels — and the comparison reveals whether merging in **weight space** beats mixing in **prediction space**.

## 2. Scope

Local (RTX 5090), **new-file-only**: adds `continuous_prm_c10_interp.py`; reuses C9/C9h (imported, not modified) + `common.task_descriptor`/`SingleAdapterLoRA`. No edits to `continuous_prm_common.py` or `transfer_astar_*` (user WIP). YAGNI: scalar backbones only (HRM, ON-LSTM); no field; no learned router (RBF/nearest/uniform only); no K-grid (zero-shot).

## 3. Source experts

For each **source** family S ∈ {C_hard_maze, C_hard_rooms, C_hard_spiral} and backbone B ∈ {hrm, onlstm}: collect `N_src=64` worlds (`C.collect_task_dataset`, `C9.SCALAR_NODES_PER_WORLD`) and train a **bounded** LoRA expert on the C7 `avgbase__{B}` base via `C9h.train_scalar_lora(rank=8, alpha=1.0, bounded=True, epochs=10, lr=2e-4)`. → 6 experts. Each source family also stores a **descriptor centroid** `z_S` = mean `C.task_descriptor(spec, world.obstacles)` over its 64 worlds.

## 4. Descriptor + RBF weighting

For a held-out target T, compute `z_T` = mean `task_descriptor` over T's TEST worlds (geometry only, no labels). RBF weights over the 3 source centroids:

`w_k(z_T) = softmax_k( −‖(z_T − z_k)/s‖² / (2σ²) )`, where `s` = per-dimension std across the 3 source centroids (with an epsilon floor), `σ` = `--rbf-sigma` (default 1.0). Weights sum to 1. (Mirrors C4's RBF; reuse the formula.)

- **nearest** = one-hot on `argmin_k ‖(z_T − z_k)/s‖`.
- **uniform** = `[1/3, 1/3, 1/3]`.

## 5. Arms (per held-out target × backbone; all ZERO target labels)

| Arm | Mechanism |
|---|---|
| `zero_shot` | pooled `avgbase__{B}` (no interpolation — the bar to beat) |
| `nearest` | nearest-descriptor source expert, baked (weight-space) |
| `uniform_wmerge` | uniform-weight weight-space merge |
| `rbf_wmerge` | **RBF-weighted weight-space merge** (the earmarked C10) |
| `rbf_pmix` | **RBF-weighted prediction-space mix** (C4-style; the head-to-head) |
| `euclid`, `oracle` | baseline / ceiling |

The C9 *few-shot* numbers (e.g. LoRA@K) are cited in the writeup as the "with target data" reference ceiling — not a live arm (different run).

## 6. Mechanisms

- **Weight-space merge (baker):** for the chosen weights `w`, build a fresh `ContinuousHeuristicModel`, load the `avgbase__{B}` base; for each LoRA-target Linear, compute the per-layer merged delta `Δ = Σ_k w_k · (alpha/r) · B_k@A_k` from the 3 experts' A/B (extracted by re-applying `apply_lora` per expert and reading `.parametrizations.weight.0.{A,B}`), and add `Δ.reshape(W.shape)` to the base weight in-place. Save a **plain** merged checkpoint (no LoRA params) with the backbone/feature/train cfg + `max_norm_residual` (bounded, 4.0). Eval via `P.ScalarResidualProvider`. (Summed rank-r deltas aren't low-rank, so baking into W is the exact, clean representation.)
- **Prediction-space mix (provider):** a `_PredMixProvider(HeuristicProvider)` holding the 3 expert models + weights `w`; `node_h` computes per-node `ŷ = clip(Σ_k w_k · ŷ_k, 0, B)` (each `ŷ_k` from expert k), then `h = euclid + side_len · ŷ` — mirroring `ScalarResidualProvider`'s integration. `.name` = the arm key.

## 7. Architecture

New `continuous_prm_c10_interp.py` (`C10Config`, modes `train`/`eval`/`analyze`/`full` + CLI). Reuses: `C9.load_source_base`, `C9.iter_test_worlds`, `C9.load_scalar_provider`, `C9._binding_budget_for`, `C9.RAW_COLS`, `C9._parse_csv`, `C9.analyze_from_raw` (or a thin C10 analyze for the arm set), `C9.SCALAR_NODES_PER_WORLD`; `C9h.train_scalar_lora`; `C.collect_task_dataset`, `C.task_descriptor`, `C.build_model`, `C.apply_lora`, `C.safe_load_state`; `P.ScalarResidualProvider`/`EuclidProvider`/`OracleProvider`/`run_world_arms`; `C7.bootstrap_median_ci` + `C6.mcnemar_exact_p`/`bh_q_values` (via C9's analyze). New pieces: descriptor/RBF weighting, the weight-merge baker, the `_PredMixProvider`.

## 8. Grid

3 held-out targets (C_hard_maze_dense, C_hard_bugtrap, C_hard_rooms_large) × 2 backbones; eval on 30 TEST worlds/target at the reused C7 binding budgets. **No K-grid** (zero-shot). Models: 6 source experts + per (target,backbone): {nearest, uniform_wmerge, rbf_wmerge} baked merges + rbf_pmix (no bake) = small. ~1 h local.

## 9. Metrics, comparisons, gates

Metric + stats identical to C9 (matched A\* expansion-ratio vs euclid + success; bootstrap CI; McNemar+BH; single binding budget; pooling not needed — zero-shot has one model per arm, but eval still runs 30 TEST worlds). Pre-registered comparisons:
1. **Interpolation vs no-interpolation:** rbf_wmerge vs zero_shot (does composing help?).
2. **Descriptor-weighting value:** rbf_wmerge vs uniform_wmerge vs nearest.
3. **Weight vs prediction space:** rbf_wmerge vs rbf_pmix (the head-to-head).
4. **vs euclid / oracle** (and the C9 few-shot ceiling, referenced).

**Gates:**
- **G0 (merge correctness):** unit tests — RBF weights sum to 1 and reduce to one-hot as σ→0; a weight-space bake with `w=[1,0,0]` reproduces source-expert-0's forward (within tol); uniform bake == mean of the three deltas; `_PredMixProvider` with `w=[1,0,0]` == expert-0's `node_h`.
- **G1 (base sanity):** `zero_shot` reproduces the C9 zero-shot direction (learned ≫ euclid on held-out).
- **G2 (directional):** the three comparisons reported with CIs; `rbf_wmerge` ≥ `zero_shot` (or a clear, honest negative).

## 10. Scale & risks

Local, ~1 h. Risks: (a) **only 3 source families** ⇒ coarse interpolation — a real limitation (more source families would sharpen it); reported honestly. (b) **weight-merge validity** — summed low-rank deltas baked into W is exact (no approximation); the risk is that large merged deltas push the model off-distribution — mitigated by the bounded residual (4.0 clamp) at integration. (c) **descriptor informativeness** — if the 3 source descriptors are too similar, RBF ≈ uniform; the uniform/nearest baselines make this visible.

## 11. Acceptance criteria

- `continuous_prm_c10_interp.py` runs `--mode full --scale local` end-to-end, writing per-target/backbone arm results (curves/summary), a comparisons MD (incl. weight-vs-prediction + interpolation-vs-baselines), significance MD, and a manifest.
- G0 merge-correctness unit tests green; G1/G2 pass (or any miss explained).
- `C10_RESULTS.md` written and cross-linked from `C9_RESULTS.md`.
- WIP files never staged; C9/C9h/common unchanged.
