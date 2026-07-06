# Continuous-PRM Program — Strategic Step-Back (post-C9b)

**Date:** 2026-07-05
**Companion to:** [`CONTINUOUS_PRM_STORY.md`](CONTINUOUS_PRM_STORY.md) (what happened). This doc is *what it means and where the leverage is now.*
> **Refined by the full-program audit (2026-07-05):** [`../PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md`](../PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md) — a cross-space (discrete + continuous) pattern analysis and an architecture-level second look at HRM/ON-LSTM. It sharpens this memo's #1 thrust: the substrate should not just be "harder" but *hierarchical by construction* (compositional missions on the existing maps, C11), with a measured-headroom gate, an explicit MLP control, and an iterative-refinement arm — because the six-phase architecture tie traces to a formulation an MLP can saturate, not to hierarchy being useless.
**Evidence base:** C6 (value field) → C7 (integration) → C8 (dynamics) → C9 (transfer) → C9h (hardening) → C10 (interpolation) → C9b (transfer under dynamics). All locally validated on one RTX 5090.

---

## 1. The scorecard, sharpened

The original north-star: **transfer learning + a hierarchically-structured model beats a purely algorithmic planner.** After six phases, it decomposes cleanly:

| Sub-claim | Status | Strength of evidence |
|---|---|---|
| A learned heuristic beats the Euclidean planner | **PROVEN** | 15–95% fewer A\* expansions, significant success gains, static + dynamic, in- and out-of-distribution, across 6 phases |
| Transfer works (adapt to unseen families) | **PROVEN** | few-shot ≫ from-scratch (q=0.000) on static (C9), matched-compute (C9h), field U-Net (C9h), and dynamic (C9b) substrates |
| A *hierarchical* model gives the edge | **NEGATIVE** | HRM ≈ ON-LSTM ≈ U-Net in every phase; where one wins it's the *non-hierarchical* conv U-Net |
| Composition/adaptation cleverness adds beyond a strong base | **NEGATIVE** | C10 interpolation ≈ zero-shot; C9b adaptation can't unlock the temporal window; C9h LoRA plateaus |

**Two of the three north-star pillars are answered — one yes, one no. The third (beats-the-planner) is a decisive yes.** So the program has largely *succeeded at its stated goal*, with one clean surprise: the hierarchical-model bet failed.

---

## 2. The three convergent findings — and the single diagnosis behind them

Read individually, the phases produced three recurring results. Read together, they are **not independent** — they share one cause.

**Finding 1 — No architectural edge.** HRM (hierarchical), ON-LSTM (recurrent), U-Net (spatial) are interchangeable as PRM heuristics; the plain conv U-Net is often best. Six phases, zero hierarchical advantage.

**Finding 2 — The cheap/present signal is already sufficient.** This is the theme that keeps recurring in different costumes:
- C8: the *present* obstacle frame predicts time-to-go as well as an explicit *future* window (aware ≈ blind).
- C9h: a *low-rank* adapter plateaus at the pooled-base level (can't specialize much).
- C10: *composing* per-family adapters by descriptor can't beat a plain uniform mix or the zero-shot base.
- C9b: *few-shot adaptation* (even full-FT at K=16) can't make the temporal window help the heuristic.

**Finding 3 — Transfer is robust.** The one thing that reliably *does* help: starting from a pooled prior instead of scratch, especially when target data is scarce.

**The diagnosis: the substrate is near its headroom ceiling.** Findings 1 and 2 are two faces of the same fact. On the maps + roadmaps + binding-budget regime we built, **a competent, cheap learned heuristic already captures most of the advantage that any heuristic could extract.** When the achievable gap between "decent heuristic" and "perfect oracle heuristic" is small, then:
- extra model capacity (hierarchy) has nothing to bite on → Finding 1;
- extra information (future window), extra data (adaptation past a point), and extra cleverness (interpolation) all bump against the same low ceiling → Finding 2;
- but a *prior* still beats *nothing* when you have little data → Finding 3 survives because it operates below the ceiling, in the data-scarce regime.

This is corroborated by the numbers: zero-shot and even from-scratch-at-K≥4 routinely reach 0.9–1.0 success and expansion-ratios of 0.1–0.3 at the binding budgets. There is little room *above* that for a cleverer method to occupy. C9b made this literal: because one dynamic world is a dense space-time supervision set, the "few-shot" regime that produced C9's crossover simply doesn't exist under dynamics — the data ceiling is hit immediately.

**The experiments are well-run. The substrate is the limiting reagent.** We have been extracting diminishing returns from methods on a problem whose achievable heuristic-quality gap is modest.

---

## 3. What this implies

1. **The interesting question has shifted.** "Can a learned, transferable heuristic beat the planner?" is answered — yes, robustly. Continuing to test *variations of method* on the *same substrate* will keep returning near-nulls, because there is no headroom left for them to reveal.

2. **The hierarchy hypothesis isn't refuted in general — it's untested where it could matter.** We showed hierarchy doesn't help *on a substrate where nothing does*. That's weak evidence about hierarchy per se. A fair test of "does structure help" requires a problem with **long-horizon, compositional, or high-dimensional structure** where a good heuristic must do genuine multi-step reasoning — not 2D point-robot roadmaps where the heuristic is a smooth, largely-local function of geometry.

3. **The transfer result is the real, bankable asset.** It's clean, replicated four ways, and publishable on its own. If the goal is a paper, the spine is: *a pooled learned PRM heuristic transfers to unseen families (static and dynamic) from ≤1 labeled world, beating the algorithmic planner; low-rank vs full fine-tune trades sample-efficiency for ceiling in the data-scarce regime; and — a clean negative — neither model hierarchy nor explicit temporal look-ahead adds value once the base is competent.*

---

## 4. Ranked next thrusts

Ordered by leverage — how much each would actually move the needle given the diagnosis above.

### #1 (recommended) — Change the substrate to one with real headroom
**The move:** raise the achievable heuristic-quality gap so that capacity/structure *can* separate. Concretely, one or more of:
- **Higher-dimensional config spaces** — a 3–7-DOF arm / manipulator PRM instead of a 2D point robot. Heuristic estimation becomes genuinely hard (narrow passages in joint space, non-local cost-to-go), and the oracle-vs-Euclid gap is large.
- **Long-horizon / compositional maps** — mazes with many sequential chokepoints, or tasks needing sub-goal decomposition, where a good heuristic must reason multiple steps ahead (the regime where hierarchy *should* help, if it ever does).
- **Tighter budgets / harder OOD** — push binding budgets down so even the best learned heuristic leaves headroom vs the oracle.

**What it resolves:** whether Findings 1 & 2 are *substrate-saturation artifacts* (my hypothesis) or *fundamental to learned planning heuristics*. Either answer is valuable and publishable. This is the single experiment that could *revive* the north-star's model half or *conclusively bury* it.
**Cost:** high (new substrate: arm kinematics / collision, new suites, re-validate the harness). But it's the only thrust that attacks the root cause.

### #2 — A roadmap-native model (GNN), not token/field
**The move:** the PRM *is a graph*; our models (token-sequence HRM/ON-LSTM, grid U-Net) ignore that. A GNN that message-passes over the roadmap to predict per-node cost-to-go matches the problem's structure. This is the most principled "different architecture" test after HRM's null.
**What it resolves:** whether "no architectural edge" is really "HRM specifically doesn't help" vs "no inductive bias helps." A graph-native win would be a genuine model-side result.
**Cost:** medium (new model class, reuse substrate + harness). Pairs naturally with #1 (a GNN on a hard substrate is the strongest single bet).

### #3 — Weaken the base and re-run C10 (close the interpolation question)
**The move:** the still-open C10 follow-up — train a deliberately narrow base that excludes the target region, then re-run interpolation. Does RBF-weighted merging beat uniform *when there's headroom*?
**What it resolves:** turns C10's null from "interpolation adds nothing" into either "interpolation adds nothing even with headroom (fundamental)" or "interpolation works when the base is weak (conditional positive)." Cleanest loose end.
**Cost:** low (reuses the entire C10 harness; one narrow-base training run + a re-eval). Good "cheap resolution" option.

### #4 — Change the objective axis
**The move:** we've optimized *heuristic quality* (expansion-ratio at a binding budget). Other axes may have headroom: **solution optimality / anytime behavior** (the learned heuristic is inadmissible — quantify and control the suboptimality/speed trade), or **the focal-vs-additive integration** at genuinely tight admissible baselines (C7 showed additive wins against loose Euclid; the opposite regime is untested here).
**What it resolves:** whether the win we're leaving on the table is in *plan quality* rather than *search speed*.
**Cost:** low-medium (reuses substrate; new metrics/analysis).

### #5 — Consolidate & publish what exists
**The move:** cluster-scale confirmation (more seeds, all backbones, success-aware composites) to lock C7–C10 + C9b into publication-grade numbers, and write the paper around the transfer result + the clean negatives.
**What it resolves:** nothing new scientifically, but banks the asset. Right choice if the goal is now "ship it."
**Cost:** medium (compute for scale-up; writing).

---

## 5. Recommendation

**Lead with #1 + #2 together: a graph-native heuristic on a higher-headroom substrate (arm/high-DOF or long-horizon).** That single combined bet is the only one that attacks the root cause the diagnosis identifies — it either *revives* the model/hierarchy half of the north-star (structure helps once the problem is hard enough) or *conclusively establishes* the stronger, more surprising claim (learned planning heuristics are architecture-agnostic; the win is purely transfer + integration). Both outcomes are publishable and both end the current pattern of near-nulls.

**If appetite for a big substrate change is low right now,** do #3 first (cheap, closes the C10 interpolation question definitively) as a warm-up, then reassess — but expect it to confirm the saturation diagnosis rather than overturn it.

**If the goal has quietly become "wrap up,"** go to #5: the transfer result + the two clean negatives are a coherent, honest paper today.

The one thing I'd advise *against* is another method-variation on the current 2D substrate. The last three phases (C10, and the two negatives in C9b) have told us, consistently, that there's no headroom left there for cleverness to find.
