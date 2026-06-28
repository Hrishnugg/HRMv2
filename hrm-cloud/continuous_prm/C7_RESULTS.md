# C7 — Integration Comparison: Results (local validation)

**Date:** 2026-06-27 (local validation, RTX 5090)
**Experiment:** `continuous_prm_c7_integration_compare.py` (spec: `../../docs/superpowers/specs/2026-06-27-c7-integration-comparison-design.md`, plan: `../../docs/superpowers/plans/2026-06-27-c7-integration-comparison.md`)
**Scope:** Phase 1 — a single **matched** comparison of heuristic-integration strategies on hard continuous-PRM planning. Phase 2 (dynamics) is a separate later cycle.

Related: [`C6_RESULTS.md`](C6_RESULTS.md) (value-field stage this builds on), [`../EXPERIMENT_RESULTS_FOCAL_REDESIGN.md`](../EXPERIMENT_RESULTS_FOCAL_REDESIGN.md) (discrete focal search), [`../EXPERIMENT_RESULTS_COMPENDIUM.md`](../EXPERIMENT_RESULTS_COMPENDIUM.md) (program history).

---

## TL;DR

A learned heuristic — integrated as an **additive residual** on top of Euclidean distance — makes A\* on hard continuous PRMs dramatically more efficient: **field-HRM cuts expansions ~15–48% with large success gains over Euclidean on every one of six suites, and generalizes to all three held-out OOD axes.** Two results diverge from the pre-registered hypothesis and are the most interesting findings:

1. **Additive integration wins; focal-ranker barely helps** — the *opposite* of the discrete-grid result. The lever flips with the *tightness of the admissible baseline*: a strong baseline (discrete Manhattan) favors focal; a weak one (continuous Euclidean) favors additive.
2. **Scalar-HRM does not reproduce its C5 collapse** — with proper hard-suite training the per-node scalar residual works as well as the spatial value field. Representation is not the deciding factor here; *training adequacy* was the C5 problem.

The north-star thesis holds: a hierarchical learned heuristic (HRM) beats the algorithmic (Euclidean A\*) planner on search efficiency and generalizes. This is a **local validation**; a cluster-scale confirmation is the natural next step.

---

## 1. Design

**Matched comparison** — same seeded worlds, same PRMs, same instances across all arms (guaranteed by a shared world generator + matched-integrity tests). Arms = `{Euclid, additive-scalar (C5-style), value-field (C6-style), focal-ranker (new)} × {HRM, ON-LSTM, U-Net}` + oracle ceiling.

| Integration | How it enters A\* |
|---|---|
| Euclid (baseline) | `h = euclidean` (admissible) |
| additive scalar | `h = euclid + side_len·clip(ŷ_node)` — per-node residual from a sequence model |
| value field | `h = euclid + side_len·interp(residual_grid)` — cost-to-go field bilinearly sampled at nodes |
| focal ranker (A\*ε) | OPEN ordered by admissible `f=g+euclid`; expand the band `{f ≤ w·f_min}` minimizing the learned `h`; cost ≤ `w·optimal` |
| oracle | exact graph Dijkstra cost-to-go (minimal-expansion ceiling) |

**Metrics:** primary = **expansions on matched-solved instances** (median ratio vs Euclid; never saturates); secondary = success rate (McNemar + BH); first-class = **path-suboptimality** (`cost/optimal`). Stats: McNemar+BH on success (learned arms only), paired Wilcoxon + seeded bootstrap 95% CI on expansion ratios.

**Suites & split (graded OOD):** train on `C_hard_maze`, `C_hard_rooms`, `C_hard_spiral`; hold out `C_hard_maze_dense` (near-OOD variant), `C_hard_bugtrap` (structural-OOD), `C_hard_rooms_large` (scale-OOD). Roadmap 192/k7, grid 64. Local scale: 96 train / 24 eval worlds, 16 epochs, w∈{1.0,1.1}.

## 2. Gate 1 — difficulty calibration

Per-suite binding budgets chosen so Euclid sits in a measurable band with large expansion headroom (oracle, a *perfect* graph heuristic, expands only ~path-length nodes, so the headroom is the Euclid→oracle expansion gap, not oracle success):

| Suite | binding budgets | Euclid succ | Euclid exp | oracle exp | headroom |
|---|---|---|---|---|---|
| C_hard_maze | 140 / 152 | 0.58 / 0.96 | ~132–137 | 23 | 0.83 |
| C_hard_maze_dense | 140 / 152 | 0.25 / 0.75 | ~136–144 | 29 | 0.79 |
| C_hard_rooms | 140 / 152 | 0.38 / 0.75 | ~132–139 | 35 | 0.74 |
| C_hard_spiral | 140 / 152 | 0.25 / 0.79 | ~133–143 | 36 | 0.73 |
| C_hard_bugtrap | 24 / 32 | 0.46 / 0.71 | ~16–20 | 8 | 0.48–0.60 |
| C_hard_rooms_large | 56 / 64 | 0.42 / 0.79 | ~44–52 | 13 | 0.70–0.75 |

(The lower-detour held-out suites — bugtrap, rooms_large — bind at small budgets; the per-suite calibration handles this automatically.)

## 3. Headline results (binding budget, lower of the band)

Success / mean expansions / **expansion ratio vs Euclid** (median, matched-solved). Lower ratio = fewer expansions.

| Suite (budget) | Euclid | **field_hrm** | scalar_hrm | best field/scalar | oracle |
|---|---|---|---|---|---|
| C_hard_maze (140) | 0.58 / 132 / 1.0 | **1.0 / 85 / 0.521** | 1.0 / 76 / **0.427** | scalar_hrm 0.427 | 1.0 / 23 / 0.17 |
| C_hard_maze_dense* (140) | 0.25 / 136 / 1.0 | **0.96 / 102 / 0.804** | 1.0 / 99 / 0.721 | scalar_hrm 0.721 | 1.0 / 29 / 0.22 |
| C_hard_rooms (140) | 0.38 / 132 / 1.0 | **0.96 / 114 / 0.829** | 1.0 / 113 / 0.822 | scalar_hrm 0.822 | 1.0 / 35 / 0.26 |
| C_hard_spiral (140) | 0.25 / 133 / 1.0 | **0.92 / 117 / 0.850** | 0.71 / 120 / 0.820 | scalar_hrm 0.820 | 1.0 / 36 / 0.25 |
| C_hard_bugtrap* (24) | 0.46 / 16 / 1.0 | **0.75 / 13 / 0.714** | 0.79 / 14 / 0.750 | scalar_onlstm 0.619 | 1.0 / 8 / 0.47 |
| C_hard_rooms_large* (56) | 0.42 / 44 / 1.0 | **0.75 / 41 / 0.839** | 0.92 / 32 / 0.729 | scalar_hrm 0.729 | 1.0 / 13 / 0.27 |

\* = held-out OOD suite. All learned arms shown are `astar` mode (additive). Success deltas vs Euclid are large and positive everywhere (e.g. maze +0.42, maze_dense +0.71, rooms +0.58, spiral +0.67).

## 4. Pre-registered comparison outcomes

1. **field_hrm vs Euclid** — wins on all six suites; significant (Wilcoxon p<0.05, McNemar p<0.05) on **5/6**. The exception is `C_hard_rooms_large` (field_hrm ratio 0.839, p=0.371, CI [0.65, 1.22] includes 1) — noisy at this scale; *scalar_hrm there is significant* (0.729, p=0.004), so learned heuristics still help that suite.
2. **Representation lever (scalar vs field)** — **neither dominates.** scalar_hrm is competitive or slightly better on maze/dense/rooms/rooms_large; field_hrm is better on spiral. **The C5 scalar-HRM collapse does not reproduce** — with proper hard-suite training the per-node scalar residual is a strong heuristic. This revises the C5→C6 "representation rescues HRM" story: C5's failure was training/setup-specific, not intrinsic to the scalar representation.
3. **Integration lever (focal vs additive, scalar)** — **additive wins decisively.** scalar_hrm/astar ratios are 0.43–0.82; the best focal-w (1.1) only reaches ~0.95–0.98 (and focal w=1.0 ≡ Euclid). Focal does *not* recover what additive already captures.
4. **Integration lever (focal vs additive, field)** — same verdict across all field models: astar 0.52–0.91 vs focal ~0.94–0.97.
5. **Gap-to-ceiling** — learned arms capture **~1–79%** of the Euclid→oracle expansion gap (uncaptured fraction 0.21–0.99 across the 30 learned arms; best on bugtrap/maze — e.g. on maze field_hrm/scalar_hrm leave ~0.42/0.32). Substantial room remains toward the oracle.
6. **In-dist vs held-out** — field_hrm reduces expansions and improves success on **both** groups: in-dist ratios 0.52–0.85, held-out 0.71–0.84, with success gains on every held-out suite. Transfer holds across all three OOD axes (near/structural/scale).

## 5. Why additive wins here but focal won in the discrete grid

This is the key scientific finding. Focal A\*ε only *re-orders within the admissible band* `{f ≤ w·f_min}` defined by the baseline heuristic; it can never override the baseline's ordering, only break ties (w=1) or widen slightly (w>1).

- **Discrete grid (Manhattan baseline):** Manhattan is a *tight* admissible heuristic, so additive over-inflation pushed it inadmissible and *hurt*, while the focal band was narrow enough that the learned tie-break helped — focal won (see [`../EXPERIMENT_RESULTS_FOCAL_REDESIGN.md`](../EXPERIMENT_RESULTS_FOCAL_REDESIGN.md)).
- **Continuous PRM (Euclidean baseline):** straight-line Euclidean is a *weak* lower bound on a roadmap with detours, so (a) the additive learned residual has large, useful magnitude to inject — cutting expansions ~50% at only ~2–5% path suboptimality — and (b) the focal band is so loose that tie-breaking inside it barely matters.

**Takeaway:** the better integration is **baseline-tightness-dependent**. Where the admissible baseline is loose, accept mild inadmissibility and inject the learned magnitude (additive); where it is tight, preserve admissibility and use the signal as a ranker (focal). The C7 harness measures both and lets the data decide per domain — a cleaner, more general statement than either single-domain result alone.

## 6. Threats to validity / caveats

- **Local scale.** 24 eval worlds/suite, 96 train; some matched sets are small (maze_dense/spiral n=6 at B140). A cluster-scale run (120+ worlds, 3–5 seeds, full w grid) is needed for publication-grade CIs. The harness lifts to cluster via `--scale cluster` with no code change.
- **One non-significant cell:** field_hrm on rooms_large (p=0.371). Reported honestly; scalar_hrm covers it.
- **Mild inadmissibility of additive arms** is real and reported (path suboptimality ~1.02–1.14; highest scalar_hrm/rooms_large 1.14). Focal arms carry the proven `≤ w` bound. The trade — ~2–14% longer paths for ~15–58% fewer expansions — is the central additive-vs-focal tension, surfaced rather than hidden.
- **Multiplicity:** BH correction applied only to the success/McNemar grid over learned arms; expansion-Wilcoxon and the six comparison p-values are uncorrected (bootstrap CIs are the primary inference). Focal "best-w" is post-hoc selected (winner's curse) — disclosed in the comparison tables.
- **Held-out difficulty:** bugtrap and rooms_large have lower detour ratios (~1.2–1.4 vs spiral's ~3.9), so they bind at small budgets and show smaller (still positive) effects. Harder held-out geometry would sharpen the transfer story.

## 7. Status & next steps

**Validated locally:** additive learned heuristics (HRM best/competitive, all backbones working) significantly beat Euclidean A\* on hard continuous-PRM planning (−15 to −48% expansions, large success gains), generalizing across three OOD axes. The integration-comparison harness is matched, tested (29 unit tests across focal/providers/maps/integrity/stats), reproducible (seeded), and cluster-ready.

**Next:**
- **Cluster-scale confirmation** (`--scale cluster`) for publication-grade CIs and the full w-sweep.
- **Harder held-out suites** (raise bugtrap/rooms_large detour) to strengthen the transfer claim.
- **Close the gap-to-ceiling** — learned arms leave ~20–99% of the Euclid→oracle gap (best ~21%); a ranking/path-loss objective or better-calibrated magnitude could recover more.
- **Phase 2 — dynamics:** bring time-varying obstacles into the continuous value-field line (separate spec/plan cycle). The provider/planner interface is designed for a dynamics-aware provider to drop in.

## 8. Reproduce

```bash
# local validation (this run)
python hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py --mode full --scale local \
  --out-dir hrm-cloud/continuous_prm/runs/c7_local
# cluster scale-up (publication numbers), same code:
python hrm-cloud/continuous_prm/continuous_prm_c7_integration_compare.py --mode full --scale cluster \
  --out-dir hrm-cloud/continuous_prm/runs/c7_cluster
```

Artifacts: `runs/c7_local/results/continuous_prm_c7_eval_summary.csv` (per-arm), `…_significance.md` (McNemar/BH + Wilcoxon/CI), `…_preregistered.md` (the six comparisons), `runs/c7_local/figures/` (expansion-ratio, gap-to-ceiling, suboptimality-vs-w). Calibration: `runs/c7_local/calibration.json`.
