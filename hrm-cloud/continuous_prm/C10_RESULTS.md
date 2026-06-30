# C10 — Parameter-space LoRA Interpolation (zero-label transfer): Results (local validation)

**Date:** 2026-06-29 (local validation, RTX 5090)
**Experiment:** `continuous_prm_c10_interp.py` (spec: `../../docs/superpowers/specs/2026-06-29-c10-interp-design.md`, plan: `../../docs/superpowers/plans/2026-06-29-c10-interp.md`)
**Scope:** Can we reach an *unseen, interior* task family with **zero target labels** by composing per-source-family LoRA experts? Weight-space merge (the earmarked idea) vs prediction-space mix vs nearest/uniform baselines vs the pooled zero-shot base. Builds on C9/C9h (per-family bounded LoRA adapters) and the C3/C4 descriptor/RBF lineage.

Related: [`C9H_RESULTS.md`](C9H_RESULTS.md) (the LoRA-plateau finding this inherits), [`C9_RESULTS.md`](C9_RESULTS.md) (few-shot transfer), [`C7_RESULTS.md`](C7_RESULTS.md) (the `avgbase` source base), [`continuous_prm_experiment_ladder_repo_coupled.md`](continuous_prm_experiment_ladder_repo_coupled.md) (earmarked this as "parameter-space LoRA interpolation").

---

## TL;DR

**A clean null on the central claim, with a clean positive on the machinery.** On a *bracketing* family grid — 8 source families on two continuous axes (maze-density, rooms-scale) chosen so the three held-out targets sit *inside* the source descriptor hull (verified, not assumed) — we trained 16 bounded-LoRA source experts (8 families × 2 backbones) and reached each interior target with **zero target labels** by interpolating the adapters.

1. **The descriptor machinery works (positive).** Every interior target is per-axis **interior** to its own axis's sources (G0b: bracketing_ok=True, no active-dim violations, both backbones). The RBF over task descriptors is **sharply selective**: **98.6–99.8 % of the interpolation mass lands on the target's own axis** (maze target → maze sources, rooms target → rooms sources; cross-axis leakage ≈ 0). Zero-label, descriptor-only weighting correctly identifies the right family.

2. **Interpolation does not beat the zero-shot base (null).** Every interpolation arm reaches the *level* of the (already strong) pooled `avgbase` zero-shot heuristic, but none meaningfully exceeds it. `rbf_wmerge` vs `zero_shot` deltas are tiny and **mixed in sign** (−0.02 on maze; **+0.08 to +0.12 worse** on the ON-LSTM rooms targets), with heavily overlapping CIs.

3. **Descriptor-weighting buys nothing over uniform/nearest.** Despite the 99 % selectivity, `rbf_wmerge` ≈ `uniform_wmerge` ≈ `nearest` (deltas ±0.01–0.11, no consistent winner; uniform is *slightly* ahead in several cells).

4. **Weight-space ≈ prediction-space.** `rbf_wmerge` vs `rbf_pmix` deltas are ≈ 0.00 everywhere — merging adapter weights and mixing their predictions are interchangeable here.

**Why (mechanistic, and predicted by C9h):** each source LoRA is *itself* ≈ zero-shot — C9h showed bounded low-rank adapters plateau at the base level (capacity-limited, never catastrophic). Any weighted composition of plateaued adapters is therefore also ≈ zero-shot. **You cannot interpolate past a ceiling the individual adapters never exceed.** The pooled base already captures essentially all the transferable signal for these interior families, so there is no headroom for interpolation to exploit. The actionable read: zero-label adapter composition is **safe and correctly targeted** (RBF localizes the right family, never catastrophic) but **unnecessary** when the base is already strong — consistent with the program-wide "the cheap signal is already captured" theme (C8/C9/C9h).

---

## Run configuration

| Knob | Value |
|---|---|
| Source base | C7 `avgbase__{hrm,onlstm}` (trained on standard maze/rooms/spiral) |
| Source families (bracketing grid) | maze-density: `C10_maze_d0..d3` (gap 0.18→0.11, clutter 2–4→10–14); rooms-scale: `C10_rooms_s10..s40` (side 1.0→4.0) — 8 total |
| Interior targets (held-out, zero-label) | `C10_maze_tgt` (gap 0.14, clutter 6–9), `C10_rooms_t25` (side 2.5), `C10_rooms_t35` (side 3.5) |
| Backbones | HRM, ON-LSTM (scalar token models; LoRA lives here) |
| Source experts | **16** bounded LoRA (8 families × 2 bb), rank 8, α 1.0, matched recipe (epochs 10, lr 2e-4) |
| Arms (zero target labels) | zero_shot, nearest, uniform_wmerge, **rbf_wmerge**, **rbf_pmix**, + euclid/oracle |
| Weighting | RBF over 8 descriptor centroids (σ=1.0, per-dim std scaled); nearest = one-hot; uniform = 1/8 |
| TEST | 30 worlds/target, matched across arms; binding budget = 150 (all targets) |
| Eval | 19,440 matched arm-records; ~34 min eval (+ ~25 min one-time source-expert training) on one RTX 5090 |

Primary metric: matched A\* expansion-ratio vs euclid on solved TEST worlds at the binding budget (median + bootstrap CI); success first-class (McNemar+BH vs euclid). Target descriptor `z_T` = mean `task_descriptor` over the TEST worlds (geometry only, no solving).

---

## Gate verdicts

- **G0 (merge correctness):** 9/9 unit tests green — RBF sum-to-1 & σ→0 one-hot; weight-merge `w=eₖ` reproduces expert-k's forward; uniform bake == mean of deltas; pred-mix `w=eₖ` == expert-k's `node_h`; **+ a CUDA regression test** (added after a GPU-only device bug, see Caveats). ✅
- **G0b (bracketing + selectivity):** all 3 targets per-axis interior (bracketing_ok=True, no active-dim violations, both backbones); RBF mass on own axis 0.986–0.998. ✅ (genuine interpolation, and the descriptor weighting is correctly selective.)
- **G1 (base sanity):** zero_shot beats euclid on every interior target/backbone — ratios 0.44–0.79; success gains significant (e.g. `rooms_t25` euclid 0.567 → 1.000, McNemar BH q=0.001; `maze_tgt` 0.800 → 1.000, q=0.039). ✅
- **G2 (directional):** **NULL** — `rbf_wmerge` ≈ `zero_shot` (mixed-sign, CI-overlapping); `rbf_wmerge` ≈ `uniform`/`nearest` (descriptor-weighting no edge); `rbf_wmerge` ≈ `rbf_pmix` (weight-space ≈ prediction-space). The earmarked "weight-space interpolation beats the alternatives" claim does **not** hold here — reported as an honest negative with its mechanism (below). ⚠️ (null, explained)

---

## The arm matrix (expansion-ratio vs euclid; lower = better; n = #TEST worlds solved by both)

**Maze-density target (`C10_maze_tgt`, binding budget 150):**
| backbone | zero_shot | nearest | uniform_wmerge | rbf_wmerge | rbf_pmix |
|---|---|---|---|---|---|
| hrm | 0.529 (succ 1.00) | 0.513 | 0.500 | 0.508 (0.97) | 0.519 |
| onlstm | 0.455 (1.00) | 0.442 | 0.451 | 0.443 (0.97) | 0.445 |

**Rooms-scale targets (`C10_rooms_t25` / `t35`, binding budget 150):**
| target / backbone | zero_shot | nearest | uniform_wmerge | rbf_wmerge | rbf_pmix |
|---|---|---|---|---|---|
| t25 / hrm | 0.790 | 0.766 | 0.781 | **0.766** | 0.766 |
| t25 / onlstm | **0.610** | 0.727 | 0.638 | 0.692 | 0.692 |
| t35 / hrm | **0.748** | 0.746 | 0.761 | 0.761 | 0.768 |
| t35 / onlstm | **0.600** | 0.808 | 0.610 | 0.716 | 0.709 |

All arms beat euclid (ratios < 1) with significant success gains (full grid in `runs/c10_local/results/continuous_prm_c10_significance.md`, all BH q ≤ 0.062). The bolded cells are the *best* arm per row — note `zero_shot` or `uniform_wmerge` wins as often as `rbf_wmerge`, and on the ON-LSTM rooms targets plain `zero_shot` is clearly best (rbf_wmerge is 0.08–0.12 *worse*). No arm is a consistent winner; the spread within a row is mostly within the bootstrap CIs.

---

## Reading the result

1. **The bracketing redesign succeeded as an experiment.** Unlike the C7 OOD-extreme held-outs (which would be extrapolation), the C10 targets are genuinely *interior* (G0b verified per-axis on every target), and the RBF correctly concentrates ~99 % of its weight on the right axis with zero labels. So this is a *fair, clean* test of interpolation — and the test returns a null, not a "the setup was wrong" non-result.
2. **The ceiling is the base, not the merge.** zero_shot `avgbase` already achieves 0.44–0.79 expansion-ratios on these hard interior families. C9h established that a bounded low-rank LoRA on a single family plateaus at ≈ this level (robust, capacity-limited). C10 is the natural corollary: a *combination* of such adapters also plateaus at ≈ zero-shot. The low-rank capacity bound that made single-family LoRA robust-but-flat is exactly what caps interpolation.
3. **Mechanism choice is irrelevant here.** Weight-space vs prediction-space, RBF vs uniform vs nearest — none matters, because they are all different ways of combining adapters that individually don't exceed the base. When there's no headroom, the combination rule can't manufacture any.
4. **Backbone-agnostic, consistent with the whole program.** HRM ≈ ON-LSTM in pattern (no hierarchical advantage); the only systematic backbone effect is that ON-LSTM's *zero_shot* on rooms is strong enough that interpolation visibly *hurts* — a mild reminder that composing adapters can drag a strong base toward the (weaker) pooled-adapter mean.

---

## Caveats

1. **GPU-only device bug found and fixed mid-validation (now guarded).** The first local run trained all 16 experts then crashed in `bake_weight_merge`: LoRA `A`/`B` were loaded `map_location="cpu"` while the model weights were on `cuda`, and `.to(dtype)` didn't move device. Every unit test (and the merge-correctness review) ran on CPU, so the mismatch never surfaced. Fixed by moving `A`/`B` to the weight's device; a **CUDA-guarded regression test** (`test_weight_merge_baker_cuda`, w=[0.5,0.5] so both adapters accumulate) now runs on GPU machines. The validated numbers are from the post-fix run.
2. **Two axes only.** maze-density + rooms-scale; structural/bugtrap interpolation is out of scope (a third axis was deferred). The null may not generalize to axes where the base is *weaker* and headroom exists — that is the natural follow-up (see below).
3. **The base is strong by construction.** `avgbase` was trained on standard maze/rooms/spiral and generalizes well to the hard interior variants. A *weaker/narrower* base (less coverage of the target region) might leave headroom where descriptor-weighted interpolation could separate from uniform — untested here.
4. **Bounded low-rank adapters only.** Per C9h the bound is irrelevant and rank-8 is the C-series standard, but a higher-rank or full-FT "expert bank" (which C9h showed *can* exceed zero-shot with target data) was not interpolated — full-FT experts have no low-rank `A/B` to merge in weight space, so weight-space interpolation is intrinsically a low-rank-adapter method.
5. Focal arms (w=1.1) are in the raw CSV but unused by the astar-only curves (as in C7–C9).

---

## Status & next

**C10 complete and validated locally.** Zero-label parameter-space interpolation of LoRA experts is **safe, correctly targeted, and unnecessary** on a base that already covers the interior region: the RBF localizes the right family with ~99 % selectivity, every arm beats euclid with zero target labels, but no interpolation variant beats the pooled zero-shot base, descriptor-weighting matches uniform, and weight-space matches prediction-space. The result is the predicted corollary of the C9h LoRA-plateau: you cannot interpolate past a ceiling the individual adapters never exceed.

Natural next steps (user's call): (a) **weaken the base** (train `avgbase` on a narrower family set that excludes the target region) to create headroom and re-test whether RBF-weighted interpolation then separates from uniform — the experiment that would turn this null into a positive *if* the mechanism has any value; (b) **higher-capacity experts** (rank↑ or per-family full-FT distilled back to a low-rank adapter) so the individual experts exceed zero-shot, then interpolate; (c) a **third (structural) axis** where the base is genuinely weak; (d) cluster-scale confirmation (more seeds/targets). The local evidence already establishes the headline cleanly.

_Artifacts: `runs/c10_local/results/continuous_prm_c10_curves.csv`, `…_comparisons.md`, `…_significance.md`, `…_bracketing.md`, `c10_weights_manifest.json`, `source_manifest.json` (16 experts). Raw per-record CSV (19,440 rows) and the 16 source-expert + merged checkpoints are regenerable locally._
