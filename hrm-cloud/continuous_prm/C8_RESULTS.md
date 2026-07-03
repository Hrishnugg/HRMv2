# C8 — Dynamics: Results (local validation)

**Date:** 2026-06-28 (local validation, RTX 5090)
**Experiment:** `continuous_prm_c8_dynamics_compare.py` (spec: `../../docs/superpowers/specs/2026-06-27-c8-dynamics-design.md`, plan: `../../docs/superpowers/plans/2026-06-27-c8-dynamics.md`)
**Scope:** Phase 2 — learned heuristics for **space-time** planning on a static PRM with deterministically moving circular obstacles. Extends the C7 integration comparison into the time dimension and adds the **time-aware vs time-blind spotlight**.

Related: [`C7_RESULTS.md`](C7_RESULTS.md) (static integration comparison this builds on), [`C9_RESULTS.md`](C9_RESULTS.md) (few-shot transfer learning), [`C9B_RESULTS.md`](C9B_RESULTS.md) (transfer under dynamics — confirms this spotlight negative is robust even to few-shot adaptation: aware never overtakes blind through full-FT@K16), [`C6_RESULTS.md`](C6_RESULTS.md) (value-field stage), [`../EXPERIMENT_RESULTS_COMPENDIUM.md`](../EXPERIMENT_RESULTS_COMPENDIUM.md) (program history).

---

## TL;DR

On a space-time A\* substrate (state = `(node, t_step)`, moving patrollers known to the planner, makespan objective), a learned **time-aware** heuristic — integrated as an **additive residual** on Euclidean-time — makes the planner dramatically more capable than the Euclidean-time baseline: **success climbs from 0.1–0.4 to ~1.0 and expansions drop 65–92% across the dynamic suites, and it generalizes to held-out OOD suites.** Additive again beats focal, reproducing C7.

But the headline scientific question of C8 — **does modeling the future obstacle window help?** — does **not** land at this local scale. The **time-blind (W=0) twin is competitive with, and in several suites significantly better than, the time-aware model.** Of the paired aware-vs-blind tests, two are significant in favor of *aware* (rooms scalar-HRM, spiral field-U-Net) and three significant in favor of *blind* (maze scalar-HRM, maze field-U-Net, rooms_large field-U-Net); the rest are ties. Net: **no consistent temporal advantage yet.**

This is the expected job of a local validation: it confirms the substrate, the pipeline, and the Euclidean-beating + additive-beating-focal results, and it flags that the temporal spotlight needs **either full-scale training or harder, more time-coupled suites (or both)** before we can claim the future window helps. Two concrete, testable reasons it likely under-delivered here are spelled out below.

> **Update (2026-06-28) — two follow-up runs change the conclusion.** We acted on both reasons: (#2) rebuilt the suites to be genuinely time-coupled, then (#1) re-ran with a fair fit (full data, 12 epochs) and real statistical power (n≈20). Outcome: **Gate 1 (learned ≫ euclid) is now statistically significant**, and **the temporal spotlight is a robust NEGATIVE** — a strong time-blind (present-frame) learner matches or beats the future-aware one (significant blind-wins in 7 cells vs 1 aware-win). The future window helps the *plan*, not the learned *heuristic*. See "[Heavier confirmation](#heavier-confirmation-2026-06-28--fair-fit--firm-n-the-spotlight-is-a-robust-negative)" (decisive) and "[Hardened re-run](#hardened-re-run-2026-06-28--time-coupled-suites-w8)" below. Results: `runs/c8_local_hardened/`, `runs/c8_local_heavy/`.

---

## Run configuration (this validation)

Reduced local preset, tuned for a ~50-minute single-GPU run:

| Knob | Value | Note |
|---|---|---|
| Train suites | C_dyn_maze, C_dyn_rooms, C_dyn_spiral | 36 usable worlds after connectivity filtering |
| Eval suites | + C_dyn_maze_dense, C_dyn_crossing, C_dyn_rooms_large | 3 in-distribution + 3 held-out OOD |
| Epochs | 6 | reduced from 12 |
| Rollout window W | 4 | reduced from 8 |
| Scalar dataset cap | 250,000 | seeded subsample of 582k reachable (node,t) samples |
| Backbones | scalar {hrm, onlstm}, field {unet, hrm} | each with a time-blind (W=0) twin → 8 trained models |
| Eval | 10 worlds/suite, w ∈ {1.0, 1.1}, calibrated binding budget/suite | 3600 arm-records |

Binding budgets (lower edge of the calibrated band): crossing=150, maze=1800, maze_dense=150, rooms=1300, rooms_large=400, spiral=2500.

The reduced knobs (W=4, 250k cap, 6 epochs) are exactly the levers that disadvantage the time-aware arm relative to its blind twin — see "Why the spotlight didn't land."

---

## Directional sanity gate

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | Time-aware learned **beats Euclidean-time** | ✅ PASS (strong) | Success 0.1–0.4 → ~1.0; matched expansion ratios 0.08–0.35 (in-dist); McNemar p 0.004–0.031; where matched-n≥6, Wilcoxon p 0.004–0.008 |
| 2 | Time-aware **beats time-blind** (the spotlight) | ❌ FAIL / inconclusive | Aware/blind median ratio ≈ 1; 2 suites significant *for* aware, 3 significant *for* blind. No consistent advantage at this scale. |
| 3 | Additive **beats focal**, focal makespan **≤ w** | ✅ PASS | Additive astar ratios 0.08–0.35 ≪ focal best-w ratios 0.6–0.99 (reproduces C7); focal suboptimality ≤ 1.1; additive is inadmissible → suboptimality 1.10–1.18 (the expected speed/optimality trade) |
| 4 | **Oracle is the expansion floor** | ✅ PASS | oracle suboptimality = 1.0 and lowest expansions; learned arms capture ~50–96% of the euclid→oracle gap (tiny sub-zero values are n=2 matched-set noise) |

**3 of 4 pass strongly; the spotlight (Gate 2) is the open question.**

---

## Comparison-by-comparison

### 1. Time-aware learned vs Euclidean-time
At each suite's binding budget, every learned arm crushes Euclidean-time on both success and expansions. Exemplar (field-HRM/astar):

| Suite | Euclid succ → arm succ | Median exp ratio (95% CI) | McNemar p |
|---|---|---|---|
| C_dyn_crossing (150) | 0.20 → 1.00 | 0.125 [0.082, 0.169] | 0.008 |
| C_dyn_maze (1800) | 0.40 → 1.00 | 0.152 [0.048, 0.357] | 0.031 |
| C_dyn_rooms (1300) | 0.30 → 1.00 | 0.083 [0.054, 0.094] | 0.016 |
| C_dyn_spiral (2500) | 0.10 → 1.00 | 0.345 | 0.004 |
| C_dyn_rooms_large (400) | 0.40 → 1.00 | 0.540 [0.145, 0.743] | 0.031 |
| C_dyn_maze_dense (150) | 0.00 → 0.00 | n/a (degenerate) | n/a |

At the binding budget the matched set is small (n=1–4 → Wilcoxon "n/a"), but at the next budget up where n≥6 the effect is significant in ratio-space: crossing@250 field-HRM 0.121 (p=0.008), maze@2500 field-HRM 0.132 (p=0.004). **C_dyn_maze_dense is degenerate at its binding budget 150 (both euclid and learned fail); at budget 3500 euclid 0.40 → learned 1.00 (+0.6).** This suite is too hard for the calibrated budget and should have its binding budget raised.

### 2. The spotlight — time-aware vs time-blind (W=0)
Matched expansion ratio of each aware arm vs its blind twin (< 1 means the future window helps):

| Suite | scalar_hrm | scalar_onlstm | field_unet | field_hrm |
|---|---|---|---|---|
| crossing | 1.050 | 0.964 | 1.152 | 1.166 |
| maze | **1.128** (p=0.027) | 0.988 | **1.832** (p=0.014) | 1.099 |
| rooms | **0.723** (p=0.004) | 0.821 (p=0.084) | 1.063 | 1.033 |
| rooms_large | 1.300 (p=0.055) | 0.900 | **1.397** (p=0.027) | 1.002 |
| spiral | 0.925 | 0.933 | **0.763** (p=0.020) | 0.875 |

Bold = significant. **Significant aware-wins: 2 (rooms scalar-HRM, spiral field-U-Net). Significant blind-wins: 3 (maze scalar-HRM, maze field-U-Net, rooms_large field-U-Net).** The future window provides no reliable edge here.

### 3. Additive vs focal (does C7's additive-wins hold under dynamics?)
Yes, decisively. Additive (plain A\* with the learned residual added to euclid-time) far out-cuts the focal ranker:

| Suite | additive astar ratio | focal best-w (1.1) ratio |
|---|---|---|
| crossing | 0.125 (field-HRM) | 0.767 |
| maze | 0.119–0.152 | 0.86–0.92 |
| rooms | 0.083–0.130 | 0.83–0.96 |
| spiral | 0.135–0.345 | 0.93–0.99 |

Same lesson as C7: against a **loose admissible baseline** (Euclidean-time), the win comes from injecting the residual *magnitude* additively, not from re-ranking within a focal band. Cost: the additive heuristic is inadmissible, so makespans run ~10–18% over optimal (suboptimality_mean 1.10–1.18) — focal stays within its w=1.1 bound.

### 5. Gap to the oracle ceiling
Median uncaptured fraction of the euclid→oracle expansion gap (0 = matches oracle, 1 = no better than euclid): field arms sit at **0.04–0.13 in-distribution** (capturing ~87–96% of the gap), rising to ~0.5–0.6 on rooms_large. scalar-ONLSTM on rooms_large is the one arm that goes >1 (worse than euclid) — consistent with its weak success there.

### 6. In-distribution vs held-out (best arm = field-HRM)
| Group | Suite | Median ratio (CI) | Succ Δ vs euclid |
|---|---|---|---|
| in-dist | C_dyn_maze | 0.152 [0.048, 0.357] | +0.60 |
| in-dist | C_dyn_rooms | 0.083 [0.054, 0.094] | +0.70 |
| in-dist | C_dyn_spiral | 0.345 | +0.90 |
| held-out | C_dyn_crossing | 0.125 [0.082, 0.169] | +0.80 |
| held-out | C_dyn_rooms_large | 0.540 [0.145, 0.743] | +0.60 |
| held-out | C_dyn_maze_dense | n/a (degenerate) | 0.00 |

The Euclidean-beating result **generalizes** to two of three OOD suites; maze_dense is uninformative at its current budget.

---

## Why the spotlight didn't land (and how to test it)

Two plausible, non-exclusive explanations — both directly testable:

1. **The aware arm is under-trained at this scale.** The time-aware models carry W=4 extra occupancy channels (field) / a length-5 rollout sequence (scalar) but were trained on the *same* capped 250k samples and *same* 6 epochs as their blind twins. The higher-dimensional input needs more data/epochs to pay off; the blind model, with a lower-dimensional input, fits better under a tight budget. The very knobs chosen to make this run fast (W=4, 250k cap, 6 epochs) penalize aware more than blind. **Test:** full-scale training (W=8, full dataset, ≥12 epochs) on the cluster — give aware a fair fit before concluding.

2. **The suites are not time-coupled enough.** If the patrollers rarely force a genuinely time-dependent detour (i.e., the right action seldom depends on *when* you arrive), then a time-blind heuristic that reads the current occupancy frame is already near-optimal, and there is nothing for the future window to add. The one suite hard enough to stress this — maze_dense — is degenerate at its budget. **Test:** harder, more time-coupled suites (faster patrollers, narrower timing gaps, blocking corridors that open/close) and a raised maze_dense budget.

Until one of these is run, the honest statement is: **learned time-aware heuristics dominate Euclidean-time planning, but we have not yet shown that explicitly modeling the future obstacle window beats a strong time-blind learner.**

---

## Engineering notes

- Two perf bugs found and fixed during bring-up (both behavior-preserving, verified by equivalence tests): (a) the field occupancy stack recomputed a full grid-Dijkstra per (world, t) sample → cached the static base once per world (20 min/epoch → 1–2 s); (b) the scalar dataset enumerated every (node, t) → 1.19M samples → seeded subsample cap at 250k. Post-fix the whole train→eval→analyze run is ~50 min on one RTX 5090.
- Calibration band (`calibration.json`) was computed once at Gate 1 and reused; the eval phase honors the per-suite binding budget.
- Stats methodology matches C7: McNemar+BH on the success/learned-arm grid; paired Wilcoxon + seeded bootstrap CI in ratio-space (exploratory, uncorrected); small-n guard (n<6 → "n/a"); multiplicity disclosed.

---

## Status & recommended next step

**Substrate + pipeline: validated. Euclidean-beating + additive-beating-focal + OOD generalization: confirmed. Temporal spotlight: open (moved toward aware after hardening — see below).**

Recommended next move (user's call — this matches the "full cluster-scale confirmation later" plan): a **cluster-scale run with W=8 / full data / ≥12 epochs on the now time-coupled (hardened) suites**, to give the time-aware vs time-blind question a fair test. If aware still ties blind at full scale on time-coupled suites, that is itself a publishable negative result about when temporal modeling does and doesn't pay off in space-time search.

---

## Hardened re-run (2026-06-28) — time-coupled suites, W=8

Acting on reason #2, the dynamic suites were rebuilt to be genuinely time-coupled, then re-run at W=8 with a fresh recalibration. Results in `runs/c8_local_hardened/`.

**What changed in the suites** (`continuous_prm_c8_dynamic_maps.py`, commits `f813468`, `9174678`):
- **Sealed gates, no passing lane** — new `lateral_frac` knob set to ~0.02 (was a fixed ±0.10 offset that left a lane), so the patroller sweep centers on the corridor and actually blocks it.
- **Faster patrollers** — `period_frac` 0.13–0.14 (was 0.26–0.32), so a gate flips open↔closed *during* the agent's approach. This is the regime only the future window can anticipate.
- **`C_dyn_crossing` kept unchanged as a control** (open arena, no chokepoint) — blind should tie aware there.
- **`maze_dense` re-tuned** (gap 0.13, radius 0.062, period 0.17, t_max 140) to stay solvable (~32%) while strongly time-coupled.
- Suite self-test confirmed the intent: hardened suites show mean arrival-delay 3.8–18.8 steps; the control sits at 0.9.

**Run config:** identical to the first run except **W=8** (was 4 — the patroller phase flips over ~8 steps, so a 4-step lookahead was too short) and a **fresh calibration** (geometry changed). 6 epochs, 250k scalar cap, scalar {hrm,onlstm} + field {unet,hrm} with W=0 twins. ~65 min.

**Recalibrated binding budgets** (5/6 now sane vs the degenerate first run): maze 1800 (euclid 0.4), rooms 1300 (0.1), spiral 2500 (0.2), crossing 150 (0.2), rooms_large 400 (0.2). **`maze_dense` remains extreme** — euclid solves only 0.1 even at 3500, so its lower band edge (150) is still 0/0 degenerate; it needs a higher binding budget (use the 3500 edge) or a dedicated calibration floor.

**Gate 1 (learned ≫ euclid) still passes strongly** on the hardened suites: e.g. field_hrm rooms 0.129 @ +0.9 success (p=0.004), crossing 0.152 @ +0.8 (p=0.008), spiral field_unet 0.152 @ +0.7 (p=0.016), maze field_unet 0.168 @ +0.5.

**The spotlight (Comparison 2) moved toward aware — but did not decisively land.**

- **Strongest new signal — success, on `rooms_large` (hardest in-scope held-out): aware solves more than blind.** field_hrm aware 1.00 vs blind 0.50 (+0.5), scalar_hrm 1.00 vs 0.70 (+0.3), field_unet 0.90 vs 0.80 (+0.1). The future window lets the planner *solve instances blind fails on* — a stronger "helps" than expansion count.
- **Expansion ratios tilted below 1** on the harder suites (rooms_large field_unet 0.687 / scalar_hrm 0.706; spiral scalar 0.853; maze field_unet 0.816) — more aware-leaning than the soft run.
- **But still mixed and mostly sub-significant** at this n (matched 1–10): `scalar_onlstm` regresses (maze 1.300, p=0.006 for *blind*; crossing 1.508, p=0.027), and most other ratios are not significant.

**Methodological catch (important):** the matched expansion ratio is computed only on worlds **both** arms solve, so on rooms_large it *excludes the very worlds where aware's edge is largest* (those only aware solves). There, the **success delta is the truer measure** — and it favors aware. A cluster-scale analysis should report a success-aware composite, not expansion ratio alone, when success rates differ.

**Read:** hardening shifted the needle in the hypothesis's direction — genuine aware *success* advantages appear on the hardest time-coupled suite, and expansion ratios move sub-1 on the harder suites — but the local run (6 epochs, 250k cap, n≤10) still cannot declare a clean win, and one backbone regresses. This now squarely motivates **reason #1**: a cluster full-scale run (full data, ≥12 epochs) on these time-coupled suites, with success-aware reporting, is the test that should settle it.

---

## Heavier confirmation (2026-06-28) — fair fit + firm n: the spotlight is a robust NEGATIVE

To test reason #1 (under-training) we re-ran on the hardened suites with a **fair fit and real statistical power**: **epochs 12** (was 6), **full uncapped scalar dataset = 1,129,536 samples** (was 250k), **train-worlds 24 → 53 usable worlds**, **eval-worlds 20** (was 10 → matched n now reaches 6–20 instead of ≤4), W=8, fresh recalibration with the binding-budget fix. Results in `runs/c8_local_heavy/`. ~3.5 h on one RTX 5090.

**Binding budgets** now all non-degenerate, including **`maze_dense`=2500** (euclid 0.05 — the [`6a3f312`](#) binding fix + more worlds removed the degeneracy). `rooms_large` recalibrated to 600 where euclid is already 0.75 (easier; less headroom).

**Gate 1 (learned ≫ euclid) is now statistically significant, not just directional:**

| Suite | Budget | best field arm | euclid→arm succ | exp ratio (CI) | Wilcoxon p |
|---|---:|---|---|---|---:|
| C_dyn_spiral | 2500 | field_hrm | 0.20→0.90 (+0.70) | **0.046** [0.038, 0.073] | n<6 (McNemar <0.001) |
| C_dyn_maze | 1800 | field_unet | 0.30→1.00 (+0.70) | **0.064** [0.046, 0.088] | 0.031 |
| C_dyn_rooms | 1300 | field_unet | 0.30→1.00 (+0.70) | 0.096 [0.038, 0.191] | 0.031 |
| C_dyn_maze_dense | 2500 | field_unet | 0.05→0.75 (+0.70) | 0.278 | McNemar <0.001 |
| C_dyn_rooms_large | 600 | scalar_hrm | 0.75→0.95 (+0.20) | 0.326 [0.207, 0.532] | **<0.001** (n=14) |
| C_dyn_crossing | 150 | field_unet | 0.30→1.00 (+0.70) | 0.258 [0.132, 0.769] | 0.062 (n=6) |

Learned space-time heuristics cut expansions **65–95%** with **+0.2 to +0.7 success**, significant where n permits. **The main thesis is confirmed under dynamics.** Additive ≫ focal holds again (Comparison 3: additive ratios 0.05–0.42 vs focal best-w 0.76–0.99).

**The spotlight (Comparison 2) is now a robust NEGATIVE.** With a fair fit and n≈20, the time-blind (W=0) twin is competitive-to-better, and the additive expansion ratios of the **blind** variants are consistently **lower (better)** than their aware counterparts:

- **Significant aware-wins: 1** (maze scalar_hrm, 0.823, p=0.021). **Significant blind-wins: 7** (crossing scalar_onlstm 2.13 p<0.001; maze scalar_onlstm 1.24 p=0.001; maze field_hrm 2.21 p<0.001; maze_dense scalar_onlstm 1.29 p=0.008; maze_dense field_unet 1.62 p<0.001; rooms scalar_onlstm 1.51 p=0.021; rooms field_hrm 1.27 p=0.027).
- The blind-vs-aware additive ratios make it stark: maze field_hrm_blind **0.054** vs aware 0.418; rooms field_hrm_blind 0.080 vs aware 0.176; rooms_large field_unet_blind 0.228 vs aware 0.380. The present-frame model is the *better heuristic*.
- The `crossing` control behaves correctly (aware ties or loses — no timing to exploit).

**This refutes reason #1: it is not under-training.** With full data and 12 epochs on genuinely time-coupled suites, the future window still does not help the learned heuristic and frequently hurts. The most consistent explanation: **for heuristic *guidance*, a present-frame learner predicts the (time-aware) cost-to-go about as well as a future-aware one — the window is largely redundant for the heuristic, even though it is essential for the optimal *plan* (the labels/oracle encode it).** A secondary contributor is optimization difficulty: the larger aware input adds variance (the recurrent field_hrm degrades most, maze 2.21×), so the extra channels cost more than they pay back at this scale.

**Comparison 4 nuance:** the convolutional **field_unet is the strongest backbone** (lowest ratios: maze 0.064, spiral 0.076, rooms 0.096), not the recurrent/hierarchical ones — so "recurrent wins when timing matters" is *not* supported here either.

### Mechanistic ablation — heuristic accuracy, aware vs blind

To pin down *why* the window doesn't help search, we measured the thing search depends on directly: the **accuracy of each model's predicted time-to-go** against the exact space-time oracle, on held-out `(node,t)` cells (`continuous_prm_c8_heuristic_accuracy.py`, results `runs/c8_local_heavy/results/c8_heuristic_accuracy.md`, commit `360f20b`). MAE in time-steps, pooled over cells; oracle-vs-oracle sanity = 0.0.

**Result: the future window does not improve heuristic accuracy.** Across 24 (suite, backbone) pairs, aware is more accurate in **11** and less in **13**; **mean Δ(aware−blind) = +0.25 steps** (aware marginally *worse*). The per-pair deltas are small (mostly <1 step) and scatter around zero — there is no systematic accuracy gain from seeing the future. This is the mechanism behind the search negative: if the predicted cost-to-go is no better, the guidance can't be better.

Caveat: MAE measures *calibration*, not search utility directly — on the open/easy suites (crossing, rooms_large) the inadmissible learned overestimators have larger MAE than euclid's tight underestimate yet still search far better (a loose admissible heuristic gives poor guidance). But for the **aware-vs-blind head-to-head** — same integration, same labels — MAE is a fair comparison, and it says the window adds no predictive signal.

### Bottom line for C8

- **Confirmed (now significant):** learned additive space-time heuristics dominate Euclidean-time planning (65–95% fewer expansions, large success gains, generalizes OOD); additive ≫ focal under dynamics, as in C7.
- **Careful negative (robust at heavy-local scale, mechanistically explained):** explicitly modeling the future obstacle window does **not** improve a learned heuristic over a strong time-blind (present-frame) one — in search expansions (7 sig blind-wins vs 1) *and* in direct predicted-time-to-go accuracy (mean Δ +0.25 steps, aware slightly worse). The present frame is a near-sufficient predictor of time-to-go for *guidance*. The publishable nuance: **time-awareness matters for the plan, not for the heuristic that guides the search.**
- **Backbone:** the plain convolutional `field_unet` is the strongest; the recurrent/hierarchical backbones do not win here.
- **Open for cluster:** a definitive publication run (more seeds, all backbones incl. field_onlstm, a success-aware composite metric) would harden these results, but the local evidence — positive thesis + mechanistically-explained negative — already points clearly.
