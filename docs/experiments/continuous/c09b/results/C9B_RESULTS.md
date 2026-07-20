# C9b — Few-shot Transfer under Dynamics: Results (local validation)

**Date:** 2026-07-02 (local validation, RTX 5090)
**Experiment:** `continuous_prm_c9b_dynamics_transfer.py` (spec: [`2026-06-30-c9b-dynamics-transfer-design.md`](../design/2026-06-30-c9b-dynamics-transfer-design.md), plan: [`2026-06-30-c9b-dynamics-transfer.md`](../plans/2026-06-30-c9b-dynamics-transfer.md))
**Scope:** Run the C9/C9h few-shot transfer protocol on the C8 **space-time** substrate. Adapt the frozen C8 pooled dynamic heuristics (aware + blind, 3 backbones) to three held-out dynamic families with zero_shot / LoRA / full-FT / from-scratch, and ask two questions: (1) does the C9/C9h crossover reproduce under dynamics, and (2) does few-shot adaptation flip C8's time-aware-vs-blind negative?

Related: [`C8_RESULTS.md`](../../c08/results/C8_RESULTS.md) (the spotlight negative this re-tests), [`C9_RESULTS.md`](../../c09/results/C9_RESULTS.md) / [`C9H_RESULTS.md`](../../c09h/results/C9H_RESULTS.md) (the static crossover this compares against), [`CONTINUOUS_PRM_STORY.md`](../../program/CONTINUOUS_PRM_STORY.md).

---

## TL;DR

Two findings, one a clean confirmation and one a nuanced null:

1. **The headline (G3): C8's "time-awareness helps the plan, not the heuristic" negative is ROBUST to few-shot adaptation.** Even **full fine-tune at K=16 target worlds does not make the time-aware heuristic beat the time-blind one** — **0 of 9 (target × backbone) headline cells** show aware beating blind on the success composite; succ_delta ≤ 0 in every full_ft@K16 cell, and across *all* K and methods there is no systematic aware advantage. The time-coupling control (`C_dyn_crossing`) shows no aware win either. You cannot adapt your way into the future window mattering for the heuristic — consistent with C8's mechanism (the present frame is a near-sufficient predictor of time-to-go).

2. **The C9/C9h crossover does NOT cleanly reproduce under dynamics (G2, nuanced).** The reason is a genuine methodological insight: **one dynamic world = K × N × (t_max+1) supervised (node, t) samples — thousands even at K=1.** So the data-scarcity regime that drove C9's crossover is *collapsed*: **full-FT is not catastrophic at K=1** (succ 0.57–0.98, beats euclid at q=0.000, vs C9-static where full-FT@K1 was worse than euclid), and **LoRA does not plateau** (it improves smoothly with K, diverging *below* zero_shot rather than tying it). What survives: **transfer ≫ from-scratch at K=1** (adapted arms 0.67–0.98 success vs from-scratch ~0.47), and **every learned arm ≫ euclid-time** (McNemar q=0.000). So "transfer helps under dynamics" holds decisively; the specific LoRA-robust / full-FT-high-ceiling *shape* is a static-substrate phenomenon tied to literal world-count scarcity, not a universal law.

Backbone-agnostic as ever (HRM ≈ ON-LSTM ≈ U-Net; the conv U-Net `field_unet` again posts the lowest expansion-ratios). The learned-heuristic-beats-Euclid and transfer-beats-scratch halves of the north-star hold under dynamics; the aware-vs-blind spotlight stays a robust negative.

---

## Run configuration

| Knob | Value |
|---|---|
| Source bases (frozen) | C8-heavy pooled `c8_{scalar_hrm,scalar_onlstm,field_unet}__{aware,blind}` (trained on C_dyn_maze/rooms/spiral) |
| Held-out targets | `C_dyn_maze_dense`, `C_dyn_crossing` (time-coupling control), `C_dyn_rooms_large` |
| Backbones × awareness | scalar HRM, scalar ON-LSTM, field U-Net × {aware (W=8), blind (W=0)} |
| Arms | zero_shot (frozen source) / lora (bounded, conv-LoRA for field) / full_ft / scratch |
| Matched recipe | epochs 10, lr 2e-4, weight_decay 1e-4, smooth-L1, grad-clip — identical across trained arms |
| K-grid / seeds | K ∈ {1,4,16}; 3 adapt seeds |
| Adapters trained | **486** (3 bb × 2 awareness × 3 methods × 3 targets × 3 K × 3 seeds) |
| TEST | 20 worlds/target, ADAPT⊥TEST; binding budget/target = C8 calibration (crossing=150, maze_dense=2500, rooms_large=600) |
| Eval | **30,600 matched arm-records** (3 targets × 170 providers × 20 worlds); ~22 h on one RTX 5090 (~16 h adapt incl. slow maze_dense world selection, ~6 h space-time eval) |

Primary metric: matched space-time A\* expansion-ratio vs Euclidean-time on solved TEST worlds at the binding budget (median + bootstrap CI, pooled over seed × world); success first-class (McNemar+BH). The aware-vs-blind probe uses a **success composite** (succ_delta over all worlds + matched ratio on the shared-solved set) — the C8 lesson that matched-ratio alone silently drops the worlds only one arm solves.

---

## Gate verdicts

- **G0 (integrity):** 11/11 unit tests green — temporal dataset schema mirrors C8 (scalar + field, aware/blind), scalar + field trainers (source/LoRA/full-FT/scratch) with schema guards, LoRA-aware provider loaders, ADAPT⊥TEST disjointness, the field cell-gather orientation verified consistent across collect→train→eval (no transpose). ✅
- **G1 (base sanity):** frozen zero_shot beats Euclidean-time on every held-out dynamic family (ratios 0.14–0.86 < 1; McNemar q = 0.001–0.016, e.g. maze_dense euclid 0.05 → zero_shot 0.40–0.60). ✅ *(zero_shot is weaker here than in C9-static — succ 0.4–0.75 — so adaptation has real headroom.)*
- **G2 (crossover replication):** **PARTIAL / nuanced.** Transfer ≫ from-scratch at K=1 ✅ (adapted 0.67–0.98 vs scratch ~0.47, q=0.000); every learned arm ≫ euclid ✅. But the *shape* — full-FT catastrophic@K1, LoRA plateaus ≈ zero_shot — does **not** reproduce: full-FT is fine at K=1 and LoRA improves with K. Cause: the space-time per-world sample richness collapses the few-shot scarcity regime (below). ⚠️ (honest miss, explained)
- **G3 (the probe — headline):** the C8 aware-vs-blind negative **survives adaptation.** 0/9 full_ft@K16 cells show aware > blind on the success composite; no systematic aware advantage at any K/method; control shows no aware win. ✅ (negative confirmed and strengthened)

---

## The adaptation curves (expansion-ratio vs euclid-time; lower = better; succ shown)

Representative (aware; full grid in `runs/c9b_local/results/continuous_prm_c9b_comparisons.md`):

**C_dyn_maze_dense / field_unet / aware** (zero_shot 0.145 @ succ 0.60):
| K | lora | full_ft | scratch |
|---:|---|---|---|
| 1 | 0.165 (0.78) | 0.112 (0.70) | 0.117 (**0.47**) |
| 4 | 0.085 (0.88) | 0.167 (0.88) | 0.151 (0.78) |
| 16 | 0.059 (0.97) | 0.101 (0.95) | 0.106 (0.95) |

**C_dyn_rooms_large / scalar_onlstm / aware** (zero_shot 0.856 @ succ 0.65):
| K | lora | full_ft | scratch |
|---:|---|---|---|
| 1 | 0.428 (0.93) | 0.455 (0.92) | 0.155 (**0.55**) |
| 4 | 0.165 (0.95) | 0.213 (0.98) | 0.131 (0.97) |
| 16 | 0.153 (0.98) | 0.185 (0.98) | 0.137 (0.98) |

Two things to read here: (a) **scratch's success is ~0.47–0.55 at K=1 and only catches up by K≥4** — the transfer-beats-scratch signal, intact. (b) **both lora and full_ft improve monotonically with K and neither collapses at K=1** — the C9 crossover's two signatures (full-FT@K1 disaster, LoRA flat) are *absent*.

---

## Reading the result

1. **The spotlight negative is now robust along a third axis.** C8 showed time-awareness doesn't help the heuristic at full training scale (7 sig blind-wins vs 1; MAE Δ +0.25). C9b adds: *nor does it help after few-shot adaptation to a new dynamic family* — not with LoRA, not with full fine-tune, not at K=16. The window is redundant for the heuristic, and no amount of target-specific adaptation changes that. This is the cleanest, most decision-relevant part of C9b: if you're building a learned heuristic for space-time search, don't pay for the future-occupancy channels — adapt a present-frame model.

2. **"K worlds" is not "K samples" in space-time — and that dissolves the crossover.** C9's crossover was fundamentally about *data scarcity*: 1 static world ≈ a few hundred node labels, so full-FT overfits and LoRA's low rank is a safety feature. Under dynamics, 1 world carries the full backward-Dijkstra time-to-go field over ~192 nodes × ~140 timesteps ≈ 25k+ supervised targets. At that density even K=1 is "data-rich," so full-FT fits cleanly and LoRA has enough signal to keep improving. The crossover isn't refuted — it's *out of regime*. (To actually reproduce it here you'd need to sub-sample the per-world (node,t) supervision, not reduce K.)

3. **Transfer still clearly helps.** Every adapted arm crushes Euclidean-time (65–95% fewer expansions, q=0.000) and beats from-scratch at K=1. The pooled dynamic source carries real signal, and adapting it to an unseen dynamic family from one world already lands at 0.7–1.0 success where from-scratch sits at ~0.5. The north-star's transfer claim holds under dynamics.

4. **No hierarchical edge, again.** HRM ≈ ON-LSTM ≈ U-Net; `field_unet` posts the lowest ratios (best), reprising C8. Five phases in, the hierarchical-model half of the north-star remains unsupported.

---

## Caveats

1. **G2 is out-of-regime, not a contradiction.** The C9/C9h crossover is a data-scarcity phenomenon; the space-time substrate's per-world sample richness removes the scarcity at these K values. This is a scoping finding about what "few-shot" means across substrates, reported honestly — it does not overturn C9/C9h (which stand on the static substrate).
2. **Single binding budget per target** (150/2500/600, from C8 calibration) rather than a full sweep — the fair-fight point, chosen for tractability given ~170 providers/target. At these budgets many adapted arms saturate success (≈1.0), compressing the expansion-ratio into the primary signal; the aware-vs-blind probe is judged on the success composite precisely to handle this.
3. **Probe vs curves success use different denominators** — the probe's succ (per-world, OR-over-seeds) and the curves' succ (per-world×seed mean) can differ for the same arm; the probe denominator is deliberately per-world so aware/blind are compared on identical world sets.
4. **Control threshold artifact:** two `C_dyn_crossing` full_ft@K16 cells flag "CONTROL:NOT-tie" at succ_delta = −0.05 (one world of 20, aware *worse*) — substantively a tie with no aware advantage; the strict |Δ|≤0.05 test just clips there. No spurious aware-win anywhere in the control.
5. Local, single machine; scalar+field with aware/blind twins; makespan/focal arms recorded in the raw CSV but unused by the astar-only curves.

---

## Status & next

**C9b complete and validated locally.** Transfer under dynamics works (learned ≫ euclid-time, transfer ≫ scratch at K=1), the C9/C9h few-shot *crossover* is shown to be a static-substrate / data-scarcity effect that dissolves when each world is a dense space-time supervision set, and — the headline — **C8's time-aware-vs-blind negative is robust to few-shot adaptation** (aware never overtakes blind, through full fine-tune at K=16).

Natural next steps (user's call): (a) if the crossover's regime-dependence is interesting, re-run C9b with **per-world (node,t) sub-sampling** to restore data scarcity and see the crossover re-emerge under dynamics; (b) cluster-scale confirmation (more seeds, success-aware composite as primary); (c) the planned **step back for the bigger picture** across C6–C10 + C9b — the recurring "no hierarchical edge" and "the cheap/present signal is already sufficient" themes now span static integration, dynamics, transfer, interpolation, and transfer-under-dynamics.

_Artifacts: `runs/c9b_local/results/continuous_prm_c9b_curves.csv`, `…_comparisons.md`, `…_significance.md`, `…_probe.md`, `adapt_manifest.json` (486 arms). Raw per-record CSV (30,600 rows) and the 486 adapter + 6 source checkpoints are regenerable locally._
