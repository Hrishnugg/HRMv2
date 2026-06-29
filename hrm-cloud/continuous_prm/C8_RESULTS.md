# C8 — Dynamics: Results (local validation)

**Date:** 2026-06-28 (local validation, RTX 5090)
**Experiment:** `continuous_prm_c8_dynamics_compare.py` (spec: `../../docs/superpowers/specs/2026-06-27-c8-dynamics-design.md`, plan: `../../docs/superpowers/plans/2026-06-27-c8-dynamics.md`)
**Scope:** Phase 2 — learned heuristics for **space-time** planning on a static PRM with deterministically moving circular obstacles. Extends the C7 integration comparison into the time dimension and adds the **time-aware vs time-blind spotlight**.

Related: [`C7_RESULTS.md`](C7_RESULTS.md) (static integration comparison this builds on), [`C6_RESULTS.md`](C6_RESULTS.md) (value-field stage), [`../EXPERIMENT_RESULTS_COMPENDIUM.md`](../EXPERIMENT_RESULTS_COMPENDIUM.md) (program history).

---

## TL;DR

On a space-time A\* substrate (state = `(node, t_step)`, moving patrollers known to the planner, makespan objective), a learned **time-aware** heuristic — integrated as an **additive residual** on Euclidean-time — makes the planner dramatically more capable than the Euclidean-time baseline: **success climbs from 0.1–0.4 to ~1.0 and expansions drop 65–92% across the dynamic suites, and it generalizes to held-out OOD suites.** Additive again beats focal, reproducing C7.

But the headline scientific question of C8 — **does modeling the future obstacle window help?** — does **not** land at this local scale. The **time-blind (W=0) twin is competitive with, and in several suites significantly better than, the time-aware model.** Of the paired aware-vs-blind tests, two are significant in favor of *aware* (rooms scalar-HRM, spiral field-U-Net) and three significant in favor of *blind* (maze scalar-HRM, maze field-U-Net, rooms_large field-U-Net); the rest are ties. Net: **no consistent temporal advantage yet.**

This is the expected job of a local validation: it confirms the substrate, the pipeline, and the Euclidean-beating + additive-beating-focal results, and it flags that the temporal spotlight needs **either full-scale training or harder, more time-coupled suites (or both)** before we can claim the future window helps. Two concrete, testable reasons it likely under-delivered here are spelled out below.

> **Update (2026-06-28) — hardened, time-coupled re-run.** We acted on reason #2 and rebuilt the suites to be genuinely time-coupled, then re-ran (W=8, fresh calibration). The spotlight **moved toward aware but did not decisively land at local scale.** See "[Hardened re-run](#hardened-re-run-2026-06-28--time-coupled-suites-w8)" below. Results: `runs/c8_local_hardened/`.

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
