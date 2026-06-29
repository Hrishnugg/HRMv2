# C9 — Few-shot Transfer Learning for Learned PRM Heuristics: Design

**Date:** 2026-06-29
**Status:** design (approved); spec for implementation.
**Branch:** `perf/eval-speedup`
**Builds on:** C7 (static integration comparison — additive residual ≫ Euclid, generalizes OOD) and C3/C4 (the original bounded-residual task-LoRA / descriptor-mixture transfer lineage). Related: `hrm-cloud/continuous_prm/C7_RESULTS.md`, `hrm-cloud/continuous_prm/C8_RESULTS.md`, `hrm-cloud/continuous_prm/continuous_prm_experiment_ladder_repo_coupled.md`.

---

## 1. Goal & scientific motivation

The north-star is "transfer learning paired with a learned model that beats a purely-algorithmic planner." C7/C8 established the *beats-algorithmic* half decisively (learned additive-residual heuristics cut A\* expansions 15–95% over Euclidean, and generalize **zero-shot** to held-out task families). The untested half is **transfer as adaptation**: given a learned heuristic and only a *few* labeled worlds from a new task family, can we adapt it to that family cheaply — and is a parameter-efficient adapter (bounded-residual LoRA) more sample-efficient than full fine-tuning?

**C9 claim (static phase, C9a):** On the hard static continuous-PRM substrate, a pooled learned heuristic adapts to an *unseen* task family from few labeled worlds. Specifically, with K target worlds:
- **adaptation beats zero-shot** transfer (the C7 OOD number),
- **adaptation beats from-scratch** training on the same K worlds (transfer carries real prior signal),
- **bounded-residual LoRA matches or beats full fine-tune at small K** (parameter-efficient adaptation is more sample-efficient),
- and every adapted arm still **beats the Euclidean baseline**.

This turns C7's "zero-shot generalizes" result into a quantified **adaptation curve** (performance vs K) and gives the north-star's "task-LoRA" mechanism its first rigorous test on hard problems.

## 2. Scope

- **C9a (this spec):** static substrate (the C7 hard suites). Build the full protocol + harness here; it is cheap and the signal is clean (C8 showed time-awareness does not help the heuristic, so static is the right place to study adaptation).
- **C9b (future, separate cycle):** rerun the identical protocol on the C8 dynamic space-time substrate as a generalization check. Out of scope for this spec beyond noting the harness is designed to be reused.

YAGNI: no learned router, no parameter-space LoRA interpolation, no descriptor mixture in C9a — those are separate follow-ups (the original ladder's C4/earmarked-C9). C9a is strictly the few-shot adaptation study.

## 3. Architecture & reuse

C9a is largely an **assembly of existing, validated parts** (new-file-only, like C7/C8 — no edits to `continuous_prm_common.py`, C3, or C7).

- **Source base (transfer source):** the existing C7 pooled scalar checkpoints `runs/c7_local/checkpoints/avgbase__{hrm,onlstm}.pt`. Verified via `runs/c7_local/train_manifest.json`: these were trained on the pooled scalar datasets over `C_hard_maze, C_hard_rooms, C_hard_spiral` (`nodes_per_world=160`) — i.e. the 3 in-distribution families, with the 3 targets genuinely held out. **Fallback:** if a config/feature mismatch is detected at load, retrain a clean pooled base on those 3 families via `common.train_avgbase` (cheap, minutes) and record it in the C9 manifest.
- **Targets (held-out families):** `C_hard_maze_dense`, `C_hard_bugtrap`, `C_hard_rooms_large` (the C7 OOD families; registered by `install_c7_hard_maps()`).
- **Adaptation mechanisms** (all on the scalar `ContinuousHeuristicModel`, where LoRA lives):
  - **LoRA:** reuse `common.train_expert(...)` — loads the frozen base, `apply_lora(rank, alpha)`, `set_lora_trainable(...)`, trains the bounded-residual adapter on the target dataset. (This is exactly the C3 expert mechanism, now driven with a few-shot K-world dataset.)
  - **Full fine-tune:** load the base `state_dict` into a fresh `ContinuousHeuristicModel`, train **all** params on the K-world dataset (a small routine in the C9 orchestrator; same optimizer/loss as `train_avgbase`).
  - **From-scratch:** `common.train_avgbase` on only the K target worlds (fresh init, no source) — the no-transfer control.
  - **Zero-shot:** the source base applied directly (K=0).
- **Eval:** reuse `continuous_prm_providers.ScalarResidualProvider(model, feature_cfg, device, backbone, max_norm_residual)` to turn each (adapted) model into an additive heuristic, plus `EuclidProvider`/`OracleProvider` and `run_world_arms` (matched A\* integrity), and the C7 analyze/stats conventions.

## 4. File structure / components

One new orchestrator file; reuses everything else.

- **Create:** `hrm-cloud/continuous_prm/continuous_prm_c9_transfer.py` — modes:
  - `adapt`: for each (target family × K × adapt-seed × method × backbone) produce and save an adapted model/adapter, plus a manifest. Methods: `lora`, `full_ft`, `scratch`. (Zero-shot needs no checkpoint — it is the source base.)
  - `eval`: matched A\* on the per-target fixed TEST set for every adapted arm + `zero_shot` (source base) + `euclid` + `oracle`. Sharded per (target, K, seed, method, backbone); merged raw CSV.
  - `analyze`: build adaptation curves + the four pre-registered comparisons + stats; write `C9_RESULTS`-style tables.
  - `full`: adapt → eval → analyze.
- **Create:** `hrm-cloud/continuous_prm/tests/test_c9_transfer.py` — unit tests for the seed-split disjointness (ADAPT vs TEST), the K-subset sampler determinism, and the adapt-routine wiring (LoRA params trainable / base frozen; full-FT all trainable; scratch fresh init).
- **Reuse (no edits):** `continuous_prm_common.py` (train_avgbase/train_expert/apply_lora/set_lora_trainable/model_checkpoint_path/ContinuousHeuristicModel), `continuous_prm_providers.py` (ScalarResidualProvider/Euclid/Oracle/run_world_arms), `continuous_prm_c7_hard_maps.py` (suite install), `continuous_prm_c7_integration_compare.py` (stats/analyze helpers — import, don't fork, where practical).

## 5. Protocol & data flow

For each **target family** T:

1. **World pool & split.** Generate a seeded pool of T worlds. Partition by seed into a disjoint **ADAPT pool** and a fixed **TEST set** (default 30 worlds). The split is deterministic and unit-tested for disjointness. TEST is identical across all K/method/seed/backbone arms (matched comparison).
2. **Labels.** For ADAPT and TEST worlds, build the PRM and Dijkstra cost-to-go (the C7 scalar dataset path), producing the normalized detour-residual labels the scalar models train/eval against.
3. **K-grid.** `K ∈ {0, 1, 2, 4, 8, 16, 32}` (default). K=0 = zero-shot.
4. **Adapt seeds.** For each K>0, `n_adapt_seeds` (default 5) independent draws of a K-world subset from the ADAPT pool (seeded sampler). Each (K, seed) → its own adapted model per method/backbone. This yields a distribution over "which K worlds you happened to get," i.e. CIs on each curve point.
5. **Arms.** Per (T, K, seed, backbone ∈ {hrm, onlstm}): `lora`, `full_ft`, `scratch`. Plus `zero_shot` (base, seed-independent), `euclid`, `oracle`.
6. **Eval.** Every arm runs matched A\* on T's TEST set at T's calibrated binding budget (reuse C7 `calibration.json` for the target suites; if absent, run C7 calibrate for the targets). Record expansions/success/suboptimality per test world.

## 6. Metrics, comparisons, gates

**Primary metric:** matched A\* expansion-ratio vs Euclid on solved TEST worlds (lower = better), with success rate and path-suboptimality first-class — identical to C7.

**Headline artifact:** the **adaptation curve** — median expansion-ratio (and success) vs K, with bootstrap CIs, one line per {zero_shot, lora, full_ft, scratch} × backbone × target. Plus a **worlds-to-threshold** summary: the smallest K at which each method reaches, say, 80% of the zero-shot→oracle expansion-gap.

**Pre-registered comparisons (per target, at the binding budget):**
1. **Adaptation vs zero-shot:** lora/full_ft at K>0 vs the K=0 base. (Does adapting help at all?)
2. **Transfer vs no-transfer:** lora vs scratch at each K. (Does the source prior carry signal? Expect lora ≪ scratch, especially at small K.)
3. **Parameter-efficiency:** lora vs full_ft at small K (1–8). (Is the adapter more sample-efficient? Expect lora ≤ full_ft at small K, converging as K grows.)
4. **Gap-to-ceiling & baseline:** all arms vs oracle (floor) and vs euclid (must beat). Plus **HRM vs ON-LSTM** transfer (north-star architecture read: does the hierarchical backbone adapt better?).

**Gates:**
- **G0 (integrity):** matched-set integrity (same TEST worlds across arms; only-both-solved cells in ratios), adapt-routine wiring correct (LoRA: base frozen + adapter trainable; full_ft: all trainable; scratch: fresh init), and adaptation reduces target training loss.
- **G1 (base sanity):** `zero_shot` reproduces the C7 OOD expansion-ratio/success for each target (confirms the reused base is wired correctly).
- **G2 (directional):** comparisons 1–3 point the right way on at least the well-powered targets (adaptation>zero-shot; lora>scratch; lora≤full_ft at small K), and every arm beats euclid.

## 7. Statistics

Identical methodology to C7/C8: paired Wilcoxon on (ratio−1) in ratio-space, seeded bootstrap 95% CIs as the primary inference, small-n guard (n<6 → "n/a"), McNemar+BH for the success grid over the *learned* arms (euclid/oracle excluded), and explicit multiplicity disclosure. Curve-point CIs come from the bootstrap over (adapt-seed × test-world) where applicable. All seeded (`np.random.default_rng(cfg.seed)`) for reproducibility.

## 8. Scale presets

- **local (default, RTX 5090):** 3 targets × K-grid {0,1,2,4,8,16,32} × 5 adapt-seeds × {lora, full_ft, scratch} × {hrm, onlstm}, TEST=30. Each adaptation is tiny (≤32 worlds, few epochs); the cost is the count of adaptations × short eval. Reuse C7 calibrated budgets. This is the validation run.
- **cluster (later):** more adapt-seeds (tighter CIs), TEST=60+, optionally add the field backbone via full-FT, and the C9b dynamic rerun. No code change beyond a preset.

Standing constraints (unchanged): local GPU validation first; **never stage the user's WIP** `continuous_prm_common.py` / `transfer_astar_heuristic_clean_parallel_fixed.py`; `runs/` text artifacts (csv/json/md) committable, binaries/`_shards/`/figures gitignored.

## 9. Risks & mitigations

- **Base reusability** — verified clean (manifest shows the right families); fallback is a cheap clean retrain recorded in the manifest.
- **from-scratch @ K=1 degenerate** — expected and informative (it is the point: transfer should dominate when data is scarce). Reported, not "fixed."
- **Adapt-seed count for tight CIs** — start at 5; if a target's curve is noisy, deepen that target C-style (more seeds) rather than widening.
- **Per-target binding budget** — reuse C7 `calibration.json`; if a target lacks a non-degenerate budget, apply the C8 binding-budget fix convention (skip a 0%-euclid edge).
- **Adaptation overfitting at large K vs base** — bounded residual (`max_norm_residual`, B_max=4.0) and the C3 tanh-bounded LoRA correction guard against off-task blow-up; early-stop/epoch budget kept modest.

## 10. Acceptance criteria

- `continuous_prm_c9_transfer.py` runs `--mode full --scale local` end-to-end and writes: per-target adaptation-curve CSV/MD, the four pre-registered comparison tables, the significance MD, and figures.
- G0/G1/G2 pass on the local run (or any failure is explained).
- `C9_RESULTS.md` written and cross-linked with C7/C8.
- No NaN/Inf heuristics; matched-set integrity holds; WIP files never staged.
