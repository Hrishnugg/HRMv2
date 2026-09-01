# C8-R result: multi-seed replication of fixed-provider dynamic transfer

**Completed:** 2026-07-23
**Design:** [2026-07-23-c8r-multiseed-replication.md](../design/2026-07-23-c8r-multiseed-replication.md) (frozen before any new-seed training)
**Status:** complete local replication; three training seeds × one common frozen 50-map-per-suite cohort

## Protocol executed as preregistered

- Fixed primary provider: field U-Net blind (additive, canonical binding budgets from `c8_local_heavy/calibration.json`, copied verbatim into every eval directory).
- Seeds: canonical 1234 (existing checkpoints) plus full-pipeline retrains 2001 and 2002 (`runs/c8r_seed2001/`, `runs/c8r_seed2002/`; collect + train, field U-Net aware+blind only).
- Common evaluation cohort: seed 999999, 50 maps/suite, generated once (`runs/c8r_fresh_eval/`) and re-evaluated byte-identically for each seed's checkpoints (`runs/c8r_seed2001_eval/`, `runs/c8r_seed2002_eval/`).
- Analysis: `docs/experiments/analysis/c8_fixed_provider_reanalysis.py` per run (map-level, 10k bootstraps, seed 20260723); outputs `c8r_fresh_cohort*`, `c8r_seed2001*`, `c8r_seed2002*`.

## Primary success deltas (blind U-Net vs Euclid; paired, map-level 95% CIs)

| Suite | seed 1234 | seed 2001 | seed 2002 |
|---|---|---|---|
| Crossing | +0.80 [+0.68, +0.90] | +0.86 [+0.74, +0.96] | +0.88 [+0.78, +0.96] |
| Maze | +0.84 [+0.74, +0.94] | +0.86 [+0.76, +0.96] | +0.84 [+0.74, +0.94] |
| Dense maze | +0.64 [+0.50, +0.78] | +0.46 [+0.32, +0.60] | +0.56 [+0.42, +0.70] |
| Rooms | +0.58 [+0.44, +0.72] | +0.58 [+0.44, +0.72] | +0.58 [+0.44, +0.72] |
| Large rooms | +0.18 [+0.08, +0.30] | +0.10 [-0.02, +0.22] | +0.16 [+0.06, +0.26] |
| Spiral | +0.84 [+0.74, +0.94] | +0.84 [+0.74, +0.94] | +0.84 [+0.74, +0.94] |

**17 of 18 seed×suite CIs exclude zero**; the single crossing interval is seed 2001's large-rooms cell, where the Euclid baseline is already at 0.82.

## Matched-solved median expansion ratios (across-seed ranges)

| Suite | seed 1234 | seed 2001 | seed 2002 | across-seed range |
|---|---:|---:|---:|---|
| Crossing | 0.273 | 0.434 | 0.299 | 0.273–0.434 |
| Maze | 0.047 | 0.059 | 0.059 | 0.047–0.059 |
| Dense maze | 0.216 | 0.625 | 0.378 | 0.216–0.625 (n=3) |
| Rooms | 0.121 | 0.135 | 0.117 | 0.117–0.135 |
| Large rooms | 0.250 | 0.548 | 0.255 | 0.250–0.548 |
| Spiral | 0.092 | 0.089 | 0.065 | 0.065–0.092 |

Search-effort reductions are seed-stable and large on maze, rooms, and spiral (medians 0.047–0.135 in every seed) and seed-variable on crossing, dense maze, and large rooms (0.216–0.625). Success replication does not depend on this variance.

## Aware-minus-blind twin contrasts (18 seed×suite cells)

Significant cells only: seed 1234 — crossing +0.080 (aware), dense −0.120 (blind), large rooms −0.200 (blind); seed 2001 — large rooms +0.080 (aware); seed 2002 — none. Totals: 2 aware, 2 blind, 14 null; **large rooms is significantly blind-better in one seed and significantly aware-better in another**. The per-suite future-window effects are not stable across training seeds; the evidence-safe conclusion is training-seed noise rather than information value, strengthening the no-systematic-benefit boundary claim.

## Preregistered readouts

| Readout | seed 2001 | seed 2002 | Verdict |
|---|---|---|---|
| R1: success CI excludes zero in ≥5/6 suites | 5/6 — pass | 6/6 — pass | **pass** |
| R2: deltas within ±0.15 of canonical in ≥5 suites | 5/6 — pass | 6/6 — pass | **pass** |
| R3: no significant aware-over-blind in any suite | fail (large rooms) | pass | **fail as stated; cross-seed sign flip reinterprets the failure as twin-contrast instability** (reported as-is per design) |

No retraining, reselection, or recalibration occurred in response to any readout.

## Claim impact

- The fixed-provider dynamic zero-shot success result is now supported by two disjoint evaluation cohorts and three independent training seeds.
- Expansion-effort magnitude claims must be stated as across-seed ranges on crossing/dense/large-rooms.
- The future-window null upgrades from "no consistent benefit" to "per-suite effects flip sign across seeds," which is stronger evidence of non-systematicity.

## Artifacts

`runs/c8r_fresh_eval/`, `runs/c8r_seed2001{,_eval}/`, `runs/c8r_seed2002{,_eval}/` (checkpoints, logs, raw rows); analysis outputs beside `c8_fixed_provider_reanalysis.py`. Not yet in the master synthesis; scheduled for the next synthesis refresh.

## Addendum (2026-07-25): post-hoc protocol divergence disclosed

Discovered during the paper's label-count audit, after all results above were frozen. The design's step 1 said "64 train worlds ... identical to canonical," but the canonical heavy run (`runs/c8_local_heavy/train_manifest.json`) used `train_worlds=24` per family, yielding 53 usable worlds (17/19/17). The retrains, at 64 candidates per family, yielded **139** (41/50/48, seed 2001) and **149** (40/54/55, seed 2002) usable worlds — roughly 2.7× the canonical training collection (retrain field datasets: 15,429 time-slices vs the canonical's 5,883). Consequences, recorded as-is:

- The success replication (R1/R2) stands, but as replication under independent seeds **and** a larger training collection — not a config-identical rerun. This is arguably stronger evidence for the transfer effect's robustness and weaker evidence about seed-only variance.
- Twin (aware−blind) contrasts between the canonical seed and either retrain confound seed with data scale; the two retrains (2001 vs 2002) remain collection-matched with each other, and their twin-contrast patterns still differ (one significant aware cell vs none), so the twin-instability conclusion survives on the matched pair.
- Paper v5 discloses the divergence (supplement §C multi-seed detail + audit-trail item 5 + limitations); the frozen design document is left unedited per policy.
- Exact reachable-label counts per pipeline: `docs/experiments/analysis/c8_reachable_count.*` (deterministic recount reproducing each run's manifest world counts as a correctness gate).
