# Continuous-PRM Learned Heuristics — The Story So Far (C6 → C10)

**Last updated:** 2026-06-30 (local validation throughout, single RTX 5090)
**Scope:** the continuous-space probabilistic-roadmap (PRM) line of the project, phases C6–C10.
**Per-phase detail:** [`C6_RESULTS.md`](C6_RESULTS.md) · [`C7_RESULTS.md`](C7_RESULTS.md) · [`C8_RESULTS.md`](C8_RESULTS.md) · [`C9_RESULTS.md`](C9_RESULTS.md) · [`C9H_RESULTS.md`](C9H_RESULTS.md) · [`C10_RESULTS.md`](C10_RESULTS.md) · [`C9B_RESULTS.md`](C9B_RESULTS.md) (transfer under dynamics — follow-up to C8+C9)
**Program history:** [`../EXPERIMENT_RESULTS_COMPENDIUM.md`](../EXPERIMENT_RESULTS_COMPENDIUM.md)

---

## The frame

This is one continuous investigation on a single substrate. **Continuous-space PRM planning:** sample a probabilistic roadmap over a hard 2-D map, run A\* on it, and replace or augment the Euclidean heuristic with a *learned* one:

```
h(node) = euclid(node, goal) + side_len · clip(ŷ(node), 0, B)
```

Everything is measured the same way:

- **Primary metric:** matched A\* **expansion-ratio vs Euclidean**, computed only on the instances *both* arms solve, at a calibrated **binding budget** — the node-expansion cap where Euclidean is only ~5–60 % successful, so there is a genuine fight. Lower = fewer expansions = better.
- **Success rate** is first-class, with **McNemar exact + Benjamini-Hochberg** significance on the success grid; expansion ratios get paired Wilcoxon + seeded bootstrap CIs in ratio-space.
- **Three backbones recur:** **HRM** (hierarchical), **ON-LSTM** (ordered-neuron recurrent), **U-Net** (spatial/field).

**The north-star the whole ladder chases:** *transfer learning + a hierarchically-structured model beats a purely algorithmic planner.*

The one-line scorecard, established across C6–C10:

| Claim | Verdict |
|---|---|
| Learned heuristic beats the algorithmic (Euclidean) planner | **YES** (15–95 % fewer expansions, significant success gains, static + dynamic, in- and out-of-distribution) |
| Transfer works (adapt to unseen families) | **YES** (from ≤1 labeled world; transfer ≫ from-scratch, q=0.000) |
| Zero-label transfer by adapter interpolation | **Works, but adds nothing** over a strong base (clean null; the selectivity machinery is sound) |
| A *hierarchical* model gives the edge | **NO** (consistently no advantage; plain conv U-Net often wins) |

---

## C6 — Value-field framing (the rescue)

**Question.** C5 had tried a *per-node scalar residual* and the HRM head **collapsed to a near-constant** — useless. Is the problem the representation rather than the model?

**Method.** Replace the per-node scalar with a **goal-conditioned spatial value field** (a cost-to-go heatmap, bilinearly sampled at PRM nodes) — structure that matches gates and bottlenecks.

**Findings.**
- **The field has a real ceiling:** the oracle cost-to-go heatmap cuts expansions **22–27 %** (Spearman 0.95 to true cost-to-go).
- Undertrained at first (constant fields, Euclidean pinned at 100 %); fixed with a denser roadmap (192/k7, putting Euclidean in the 50–70 % band) and more training.
- Properly trained (96 worlds, 16 epochs), **all three learned fields significantly beat Euclidean.** On `C_hard_maze` at the binding budget, **HRM +0.35 success (0.625 → 0.975, p=1.2e-4, −42 expansions)**, near the oracle's 0.95. **HRM is rescued** — best learned model, reversing its C5 collapse.
- **Multi-suite training closes the OOD gap:** training on maze+rooms lifts rooms 0.70 → 0.875 and the held-out dense-maze variant 0.825 → 1.0.

**Carried forward.** C5's failure was representational/training-specific, not a capacity limit. Focal re-ranking was explicitly *not* needed — additive integration already captured a well-ranked field.

---

## C7 — Integration comparison (the controlled head-to-head)

**Question.** Is the *value field* specifically what's needed, or just *any* well-trained signal integrated additively?

**Method.** Matched comparison over **six hard suites** (maze, rooms, spiral + bugtrap, maze_dense, rooms_large held out) × {additive scalar residual, value field, focal ranker} × {HRM, ON-LSTM, U-Net}, sharing an `avgbase` source model trained on maze/rooms/spiral.

**Findings (load-bearing for everything after).**
- **Learned additive heuristics cut A\* expansions ~15–48 %** over Euclidean across all six suites, and **generalize OOD**.
- **The per-node scalar does *not* reproduce its C5 collapse when trained properly** — confirming C6's diagnosis. The field isn't magic; the C5 scalar was just badly trained.
- **Additive beats focal** here — the *opposite* of the earlier discrete-grid result. The principle: focal helps with a *tight* admissible baseline and a well-ranked-but-miscalibrated signal; against a **loose** baseline like Euclidean the win comes from injecting residual *magnitude* additively, not re-ranking within a band. **Which integration wins depends on baseline tightness.**
- **No hierarchical edge:** HRM ≈ ON-LSTM ≈ U-Net — the first clear sign the "hierarchical model wins" half of the north-star isn't landing.

C7's `avgbase__{hrm,onlstm}` becomes the **source base** reused by C9/C9h/C10.

---

## C8 — Dynamics (does time change the story?)

**Question.** Does the C7 story survive in time, and — the headline — **does explicitly modeling the future obstacle window help?**

**Method.** Lift to **space-time**: state = `(node, t_step)`, deterministically moving patrollers known to the planner, makespan objective, space-time A\* with a backward space-time Dijkstra oracle. Compare a **time-aware** model against a **time-blind (W=0) twin** that sees only the present frame.

**What held (confirmed; significant at the heavy run).**
- Learned time-aware additive heuristics **dominate Euclidean-time: 65–95 % fewer expansions, +0.2 to +0.7 success** (e.g. spiral 0.20 → 0.90, maze 0.30 → 1.00), generalizing OOD.
- **Additive ≫ focal again** (additive ratios 0.05–0.42 vs focal 0.76–0.99) — C7's lesson reproduces under dynamics.

**The spotlight — a robust, mechanistically-explained NEGATIVE.** This took three runs to settle honestly. The soft local run was inconclusive (2 suites favored aware, 3 favored blind). So the suites were **hardened to be genuinely time-coupled** (sealed gates, faster patrollers that flip open/closed *during* approach, W=8 lookahead), then re-run with a **fair fit** (12 epochs, full 1.13M-sample dataset, n≈20):
- **7 significant blind-wins vs 1 aware-win.**
- A direct **mechanistic ablation** (predicted time-to-go vs the exact oracle) sealed it: **aware was marginally *worse* — mean Δ +0.25 steps.** The future window adds no predictive signal *for the heuristic*.
- The `crossing` control (open arena, no chokepoint) behaved correctly: aware ties/loses, as it should.

**Nuance worth keeping:** time-awareness matters for the optimal *plan* (the labels/oracle encode it), but not for the learned *heuristic that guides the search* — the present frame is a near-sufficient predictor of time-to-go. And again the plain conv **U-Net was the strongest backbone**, not the recurrent/hierarchical ones.

---

## C9 — Transfer learning (the untested half of the north-star)

**Question.** Adapt the C7 pooled `avgbase` heuristic to **three held-out hard families** (maze_dense, bugtrap, rooms_large) from K labeled worlds — how, and how cheaply?

**Method.** K ∈ {0,1,2,4,8,16,32}, 5 seeds, arms = **zero-shot / LoRA (bounded residual) / full fine-tune / from-scratch**. 540 adapted models, ~5.5 h.

**The headline — a sample-efficiency crossover.**
- **Zero-shot already beats Euclidean OOD** (ratios 0.65–0.87; success gains McNemar q=0.000, e.g. maze_dense 0.33 → 1.00).
- **Bounded LoRA = robust + sample-efficient:** from **K=1 world** it matches zero-shot and **never catastrophically fails**, but **plateaus** (barely improves with more data).
- **Full fine-tune = high-variance, high-ceiling:** at K=1 it *overfits the single world* (ratio ~1.13, succ 0.37, worse than Euclidean) but by K≥16 it is the **best** arm (ratios 0.49–0.62).
- **From-scratch needs K≥16** to beat Euclidean — the control proving the source prior carries real signal.

**So:** transfer ≫ from-scratch decisively at low K (q=0.000); LoRA wins few-shot via robustness, full-FT wins data-rich via capacity. The north-star's "transfer + learned heuristic beats the algorithmic planner" is now concrete — adapt from as little as one labeled world. (HRM ≈ ON-LSTM, again.)

---

## C9h — Hardening (kill the confounds)

**Question.** C9's LoRA used a lighter recipe than full-FT. Is the crossover real, or a recipe artifact? Does it hold on the field backbone?

**Method.** Re-run **every arm at identical compute** (epochs 10, lr 2e-4); split LoRA into **bounded vs unbounded**; extend to the **field U-Net via a new conv-LoRA** (reusing the shape-agnostic `SingleAdapterLoRA` on Conv2d weights). 324 models, 3 targets × 3 backbones × K{1,4,16}.

**Everything sharpened.**
- **The crossover is real, not a recipe artifact:** at matched compute, full-FT is still worst at K=1 (overfit) and best by K≥4–16 (ratios 0.40–0.59); **LoRA still plateaus ≈ zero-shot.** It is genuinely **low-rank vs full-rank capacity**.
- **The bound is irrelevant:** bounded ≈ unbounded **everywhere** (median Δ = 0.000 ± 0.008 across 27 cells) — the clamp rarely binds for a low-rank adapter, so LoRA's robustness comes from the **low-rank structure**, not the cap.
- **Generalizes to the field U-Net:** conv-LoRA plateaus too; **field full-FT is the single best adapter** (rooms_large 0.404 @ 0.97 success vs zero-shot 0.98 @ 0.67).
- Transfer ≫ scratch holds (q=0.000).

**Crisp summary:** LoRA = robust / sample-efficient / capacity-limited; full fine-tune = high-variance / high-ceiling — across scalar *and* field backbones, bound shown irrelevant.

---

## C10 — Parameter-space LoRA interpolation (the earmarked idea; a clean null)

**Question.** Can you reach an **unseen family with ZERO target labels** by *composing the adapters you already have* — merging per-family LoRA weight-deltas by task-descriptor similarity (`W' = W + Σ wₖ·scaleₖ·BₖAₖ`)?

**Method.** The key move: a **bracketing source grid** so targets are genuine *interpolation*, not extrapolation. Two continuous axes — maze-density (`C10_maze_d0..d3`, gap 0.18 → 0.11) and rooms-scale (`s10..s40`, side 1 → 4) — 8 source families, 3 **interior** targets (maze between d1/d2; rooms 2.5 and 3.5). 16 source experts × 7 arms (zero_shot / nearest / uniform-merge / **rbf-merge** / **rbf-predmix** / euclid / oracle), 19,440 eval rows.

**A clean null on the claim, with a clean positive on the machinery.**
- **The machinery works (G0b):** all targets verified per-axis *interior*, and the RBF over descriptors is **sharply selective — 98.6–99.8 % of weight mass lands on the target's own axis** (maze → maze sources, rooms → rooms, ~0 cross-axis leak). Zero-label, descriptor-only weighting correctly identifies the family.
- **G1:** every arm beats Euclidean (ratios 0.44–0.79; success gains BH q 0.001–0.039).
- **G2 — the null:** interpolation arms reach zero-shot's *level* but **don't beat it**; **RBF-weighting buys nothing over uniform/nearest**; and **weight-space ≈ prediction-space** (Δ≈0). On ON-LSTM rooms, plain zero-shot is *best* and interpolation is 0.08–0.12 worse.

**Why — predicted by C9h.** Each source LoRA is *itself* ≈ zero-shot (C9h's plateau). **Any weighted composition of plateaued adapters is also ≈ zero-shot — you cannot interpolate past a ceiling the individual adapters never exceed.** The pooled base already captures the transferable signal for interior families, so there is no headroom. Zero-label adapter composition is **safe and correctly targeted but unnecessary** when the base is already strong.

*(Engineering footnote: C10 caught a GPU-only device bug in the weight-merge baker — LoRA A/B loaded on CPU, model weights on CUDA — that every CPU unit test and the merge-correctness review missed. Now guarded by a CUDA-only regression test. Lesson: device correctness needs a GPU-guarded test, not just CPU equivalence.)*

---

## The three through-lines

1. **"The cheap signal is already captured" — the dominant theme.** It recurs every phase after C7. C8: the present frame predicts time-to-go as well as the future window. C9/C9h: low-rank LoRA plateaus at the zero-shot level. C10: composing plateaued adapters stays at the plateau. The pooled base repeatedly already holds most of the transferable signal; extra structure (temporal lookahead, more target data through a low-rank adapter, descriptor-weighted adapter mixing) fails to add to it.

2. **No hierarchical edge — anywhere.** HRM ≈ ON-LSTM ≈ U-Net in *every* phase, and where one wins it is usually the plain conv **U-Net** (C8's strongest, C9h's best field adapter). The north-star bet that a *hierarchically-structured* model would win is **not supported** by C6–C10. C6 "rescued" HRM only to competitiveness, not superiority.

3. **Integration depends on baseline tightness.** Additive beats focal throughout C7/C8 precisely because Euclidean is a loose admissible baseline — the inverse of the discrete-grid result. A clean, transferable principle.

---

## Where the frontier is (candidate next directions)

The two halves of the north-star that *landed* are "learned heuristic ≫ algorithmic planner" and "transfer works." The two open/negative results point at the most interesting next experiments:

- **Create headroom for interpolation (turn C10's null into a real test).** C10 couldn't separate RBF from uniform because the base already covers the interior region. Train a **deliberately weaker/narrower base** (one that excludes the target region), then re-run interpolation: if descriptor-weighted merging beats uniform *there*, the mechanism has value; if not, the null is fundamental. This is the single cleanest follow-up.
- **Higher-capacity experts before interpolating.** Since the plateau is the low-rank bound (C9h), interpolate experts that actually exceed zero-shot — higher-rank adapters, or per-family full-FT distilled back into a mergeable low-rank delta.
- **A structural third axis.** Both C10 axes (density, scale) are smooth deformations. A topology/structure axis (e.g. bugtrap-style) where the base is genuinely weak would stress interpolation differently.
- **Revisit the hierarchical question deliberately.** Four phases show no hierarchical edge. Either accept it as a finding (publishable: hierarchy doesn't help PRM heuristics on these substrates) or design a task where hierarchy *should* matter (long-horizon, compositional sub-goals) and test it head-on.
- **Cluster-scale confirmation.** Every result here is local-RTX-5090, single-seed-ish. A multi-seed, all-backbone, full-grid run would harden the publishable claims (especially C8's spotlight negative and the C9/C9h crossover).
- **C9b — transfer under dynamics.** The C9 protocol on the C8 space-time substrate (the harness is reusable) — does the few-shot crossover hold when the target is a dynamic family?

---

## Follow-up: C9b — transfer under dynamics ([`C9B_RESULTS.md`](C9B_RESULTS.md))

Ran the C9/C9h transfer protocol on the C8 space-time substrate (486 adapters; aware + blind sources). Two results, both reinforcing the through-lines:
- **C8's time-aware-vs-blind negative is robust to few-shot adaptation** — even full fine-tune at K=16 target worlds never makes the aware heuristic beat the blind one (0/9 headline cells). You can't adapt your way into the future window mattering for the heuristic. (Theme 1, along a new axis.)
- **The C9/C9h few-shot crossover is out-of-regime under dynamics** — one dynamic world = K × N × (t_max+1) ≈ 25k+ supervised (node,t) targets, so "K=1" is already data-rich: full-FT isn't catastrophic and LoRA doesn't plateau. The crossover is a *data-scarcity* effect, not a universal law. Transfer ≫ from-scratch at K=1 still holds; learned ≫ euclid-time still holds; no hierarchical edge (U-Net best), again.

## Status

All of C6–C10 (plus the C9b dynamics-transfer follow-up) is validated locally. The learned-heuristic and transfer halves of the north-star are established; the *hierarchical-model* half is a consistent negative across all six phases; the time-aware spotlight is a robust negative even under adaptation; and the open frontier is whether descriptor-weighted interpolation can ever beat uniform given a base with real headroom.
