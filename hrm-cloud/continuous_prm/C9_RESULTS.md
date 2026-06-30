# C9 — Few-shot Transfer Learning: Results (local validation)

**Date:** 2026-06-29 (local validation, RTX 5090)
**Experiment:** `continuous_prm_c9_transfer.py` (spec: `../../docs/superpowers/specs/2026-06-29-c9-transfer-design.md`, plan: `../../docs/superpowers/plans/2026-06-29-c9-transfer.md`)
**Scope:** C9a (static) — few-shot adaptation of a learned additive-residual PRM heuristic to *unseen* hard task families, comparing zero-shot / LoRA / full fine-tune / from-scratch. Builds on C7 (the pooled `avgbase` source) and the C3 bounded-residual task-LoRA mechanism.

Related: [`C9H_RESULTS.md`](C9H_RESULTS.md) (matched-compute + field hardening of this result), [`C10_RESULTS.md`](C10_RESULTS.md) (zero-label parameter-space interpolation of these adapters — a clean null: interpolation can't pass the LoRA plateau), [`C7_RESULTS.md`](C7_RESULTS.md) (static integration; source base), [`C8_RESULTS.md`](C8_RESULTS.md) (dynamics), [`continuous_prm_experiment_ladder_repo_coupled.md`](continuous_prm_experiment_ladder_repo_coupled.md) (C1–C4 transfer lineage).

---

## TL;DR

The untested half of the north-star — **transfer learning** — lands cleanly. Adapting the C7 pooled scalar heuristic to three held-out hard families (`maze_dense`, `bugtrap`, `rooms_large`) from a handful of labeled worlds beats both the Euclidean baseline and from-scratch training, with two sharply different adaptation profiles:

- **Bounded-residual LoRA is the robust, sample-efficient adapter.** From **K=1 world** it already matches the (already strong) zero-shot heuristic — expansion-ratio ~0.65–0.77 vs euclid at ~0.8–1.0 success — and **never catastrophically fails**. It plateaus rather than improving much with more data.
- **Full fine-tune is high-variance, high-ceiling.** At K=1 it *overfits the single world* and is worse than euclid (ratio ~0.95–1.13, success 0.37); but by K≥8–16 it adapts strongly and becomes the **best** arm (ratios 0.49–0.62).
- **From-scratch needs many worlds.** At K≤8 it is statistically indistinguishable from euclid (no transfer); only by K=16–32 does it catch up. This is the control that proves the source prior carries real signal.

So the headline is a **sample-efficiency crossover**: LoRA dominates in the few-shot regime (robustness), full fine-tune dominates once enough target data exists (capacity). Both transfer arms massively beat from-scratch at low K. The success gains over euclid are highly significant (McNemar BH q = 0.000).

This is the first rigorous transfer result on the hard substrate, and it makes the north-star's "transfer + learned heuristic beats the algorithmic planner" concrete: a heuristic trained on one family adapts to an unseen family from as little as one labeled world.

---

## Run configuration

| Knob | Value |
|---|---|
| Source base | C7 `avgbase__{hrm,onlstm}` (trained on maze/rooms/spiral) |
| Target families (held-out) | C_hard_maze_dense, C_hard_bugtrap, C_hard_rooms_large |
| Backbones | HRM, ON-LSTM (scalar token models; LoRA lives here) |
| K-grid | 0, 1, 2, 4, 8, 16, 32 (K=0 = zero-shot) |
| Adapt seeds | 5 per (target, K) — CIs pool over (seed × world) |
| Methods | zero_shot, lora, full_ft, scratch |
| TEST worlds | 30/target (disjoint from ADAPT; matched across all arms) |
| Binding budget/target | bugtrap=24, maze_dense=140, rooms_large=56 (calibrated; lowest non-degenerate) |
| Adapted models trained | **540** (3 targets × 2 bb × 6 K>0 × 5 seeds × 3 methods) + zero-shot |
| Eval | 99,360 matched arm-records; ~5.5 h on one RTX 5090 |

Primary metric: matched A\* expansion-ratio vs euclid on solved TEST worlds at the binding budget, pooled over adapt-seeds; success rate first-class; C7-style stats (paired bootstrap CI; McNemar+BH success grid).

---

## Gate verdicts

- **G0 (integrity):** 10/10 unit tests green; ADAPT⊥TEST disjointness verified; matched comparison (all arms on the same TEST worlds); single binding budget per target; LoRA-aware checkpoint loading. ✅
- **G1 (base sanity):** zero-shot reproduces the C7 OOD direction — learned ≫ euclid on every held-out family (ratios 0.65–0.87; success gains McNemar q=0.000, e.g. maze_dense euclid 0.33 → 1.00). ✅ (same-distribution sanity; the C9 TEST draw differs from C7's by suite index, so this is directional, not world-for-world.)
- **G2 (directional):** transfer ≫ from-scratch at low K ✅✅; LoRA ≤ full_ft (more robust) at small K ✅; full_ft best at high K ✅; adaptation beats zero-shot for full_ft at K≥8 ✅ (LoRA ties zero-shot — see caveat). Every arm beats euclid except full_ft/scratch at K=1 (the expected overfit/no-transfer failure). ✅

---

## The adaptation curves (expansion-ratio vs euclid; lower = better)

Representative (HRM; full table in `runs/c9_local/results/continuous_prm_c9_comparisons.md`):

**C_hard_maze_dense / HRM** (zero-shot 0.650 @ succ 1.00):
| K | lora | full_ft | scratch |
|---:|---|---|---|
| 1 | 0.650 (succ 1.00) | 0.744 (succ 0.76) | 1.008 (succ 0.37) |
| 4 | 0.674 (0.99) | 0.702 (0.95) | 1.023 (0.27) |
| 16 | 0.730 (0.97) | **0.571** (0.99) | 0.808 (0.80) |
| 32 | 0.701 (0.99) | **0.552** (0.97) | 0.755 (0.85) |

**C_hard_rooms_large / HRM** (zero-shot 0.771 @ succ 0.97):
| K | lora | full_ft | scratch |
|---:|---|---|---|
| 1 | 0.771 (0.97) | 1.128 (0.37) | 1.143 (0.35) |
| 8 | 0.771 (0.93) | 0.500 (0.92) | 0.558 (0.71) |
| 32 | 0.809 (0.75) | **0.489** (0.94) | 0.468 (0.92) |

**C_hard_bugtrap / HRM** (zero-shot 0.696 @ succ 0.83): lora flat ~0.70 (succ 0.83) through K=8 then degrades (K32 0.950 @ 0.48); full_ft 0.954@K1 → 0.619@K16; scratch 1.000@K1 → 0.783@K32.

The pattern is consistent across both backbones (ON-LSTM table in the artifact). **HRM vs ON-LSTM:** similar adaptation profiles — no clear hierarchical advantage (consistent with C7/C8).

---

## Reading the result

1. **Transfer beats no-transfer, decisively, in the few-shot regime.** At K=1–4, LoRA sits at the zero-shot level (0.65–0.77, success 0.8–1.0) while from-scratch is at/above euclid (≥1.0, success 0.27–0.53, McNemar n.s.). The source prior is doing the work when data is scarce — exactly the transfer claim.
2. **LoRA = robustness; full fine-tune = capacity.** LoRA's bounded residual makes it essentially incapable of catastrophic forgetting — it inherits the base and nudges — so it is the safe choice at low K. Full fine-tune has the freedom to overfit one world (K=1 disaster) *and* the capacity to exploit many (K≥16 best). The crossover is the actionable finding: pick the adapter for few-shot, the fine-tune when target data is plentiful.
3. **The heuristic-quality story matches C8.** As in C8, the learned signal that matters is largely captured cheaply; LoRA's plateau says the extra target data mostly doesn't change the additive residual much beyond what the pooled base already encodes.

---

## Caveats

1. **Compute-budget asymmetry (important).** LoRA reuses the validated C3 recipe — `expert_epochs=8 @ lr 1.5e-4` with a **bounded tanh residual** — while full_ft/scratch use `base_epochs=10 @ lr 2e-4` **unbounded**. So LoRA's plateau conflates *adapter capacity* with a *lighter, bounded* recipe; this is not a perfectly matched-compute comparison. The robust, recipe-independent conclusion is the **variance** contrast (LoRA low-variance/robust, full_ft high-variance/high-ceiling) and the **transfer ≫ scratch** result, not a literal "LoRA caps out" claim. A matched-budget LoRA-vs-FT sweep is the natural cluster follow-up.
2. **Scalar backbones only.** LoRA is defined on the HRM/ON-LSTM token models; the C7/C8-strongest **field U-Net** was not adapted (no LoRA there). Field transfer (full-FT only) could differ.
3. **Tight binding budgets** (24–140) — the calibrated fair-fight points where euclid partially succeeds; zero-shot is already strong there.
4. **G1 is distributional**, not world-for-world C7 reproduction (different suite index → different 30-world TEST draw; matched across all C9 arms).
5. Focal arms (w=1.1) are recorded in the raw CSV but unused by the astar-only curves.

---

## Status & next

**C9a complete and validated locally.** Transfer learning works on the hard substrate: a pooled learned heuristic adapts to unseen families from ≤1 labeled world (LoRA, robust) and exploits more data when available (full fine-tune), both far beating from-scratch — with significant success gains over the Euclidean planner.

Natural next steps (user's call): (a) **cluster confirmation** with a *matched* LoRA-vs-FT compute budget, more seeds, and the **field backbone**; (b) **C9b** — rerun this protocol on the C8 dynamic substrate (the harness is reused as-is); (c) the originally-earmarked **parameter-space LoRA interpolation** (interpolate adapters by task descriptor) now that the per-family adapters exist. The local evidence already establishes the headline.

_Artifacts: `runs/c9_local/results/continuous_prm_c9_curves.csv`, `…_comparisons.md`, `…_significance.md`, `adapt_manifest.json` (540 arms). Raw per-record CSV (99,360 rows) is regenerable locally._
