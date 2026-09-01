# C13-J preregistration: suite-balanced current-state LHBL

Date frozen: 2026-07-17, after C13-I failed its six-suite gate and before any
C13-J cohort is generated or model is trained.

## Why this experiment is next

C13-I isolated a distribution problem rather than a search-safety problem.
The maze-only C13-H model beat `field_hrm` in mean expansions on the original
maze suite (79.50 versus 84.58), but only one of six suite deltas was negative.
Its pooled mean was 15.36 expansions worse than `field_hrm`, and large rooms
exceeded the predeclared 1.10 direct-search cost regime. The reopening FOCAL
control remained valid and bounded on all 144 roadmaps.

The next intervention is therefore training-distribution coverage. It does not
change the runtime information boundary, add the map raster, tune on the failed
C13-I rows, or weaken the path-quality requirement.

## Locked runtime boundary

The model continues to receive only:

- current-to-goal geometry;
- 32 radius-bounded rays (`radius = 0.20 * side_len`, 32 samples per ray); and
- up to 24 one-hop roadmap actions.

The runtime input still forbids the complete obstacle list, occupancy raster,
world descriptor, global free-space summaries, unbounded line of sight,
`dist_to_goal`, and shortest-path results. The LHBL target remains a local
radius-bounded Bellman backup bootstrapped from a frozen previous-iteration
state model; graph shortest-path distance remains evaluation-only.

## Data and model

- Architecture: flat MLP, matching the strongest stable C13-H family.
- Roadmap: 192 nodes, `k=7`.
- Training suites: `C_hard_maze`, `C_hard_rooms`, `C_hard_spiral`—the same
  three suite distributions used to train C7, so the current-state arm does not
  get broader training-family coverage than the map-derived comparator.
- Training: 32 connected worlds per suite (96 total).
- Bellman-target validation: 8 connected worlds per training suite (24 total).
- Development search comparison: 4 connected worlds from each of all six C7
  suites (24 total), on a disjoint seed block.
- Optimization: 8 outer bootstrap iterations, 5 inner epochs, batch 128,
  hidden width 64, AdamW learning rate `5e-4`, weight decay `1e-4`.
- Additive alpha grid: `0.25, 0.50, 0.75, 1.00`.
- Model seed: 17413. Cohort seed blocks and exact accepted world/roadmap seeds
  are emitted by the runner and must be pairwise disjoint and disjoint from the
  C13-I benchmark.

Features are cached per accepted roadmap only to make interrupted execution
resumable. The cache is an exact serialization of the locked current-state
features, not a source of additional information.

## Development comparison and selection

Every checkpoint/alpha candidate is compared on the 24-world development block
to live `field_hrm` and `scalar_hrm` outputs from the saved C7 checkpoints.
Selection is fixed as follows:

1. require a finite current-state A* result on all 24 roadmaps;
2. require maximum current-state graph-optimal cost ratio at most 1.10;
3. require the pooled paired bootstrap 95% confidence interval upper endpoint
   for `expansions(current) - expansions(field_hrm)` to be below zero;
4. require a negative mean expansion difference in at least four of six suites;
5. among passing candidates, select the lowest pooled current-state mean
   expansions, then lower maximum cost ratio, earlier iteration, and lower
   alpha.

If no candidate passes, C13-J is a failed distribution-only intervention. No
checkpoint or alpha may be chosen merely because it is numerically best, and
the next experiment must change the representation or online integration.

## Confirmation boundary

A development pass authorizes one confirmation only: 24 new connected worlds
per suite from a fourth, untouched seed block, evaluated live against all five
C7 learned providers. Confirmation uses the selected checkpoint/alpha without
modification and the same C13-I solid-improvement gate:

- 144/144 valid current-state paths;
- maximum cost ratio at most 1.10;
- pooled paired 95% CI upper endpoint below zero versus `field_hrm`;
- negative mean expansion delta in at least four of six suites; and
- pooled mean path ratio no more than 0.01 above `field_hrm`.

The formally bounded reopening-FOCAL control is carried along as a safety
operating point but cannot substitute for the direct arm's efficiency gate.

## Integrity

The run must hash this preregistration, implementation, source study manifest,
C7 checkpoints, feature caches, cohort records, all trained checkpoints,
training history, development raw rows, candidate summaries, selection verdict,
verification, and manifest. It must record zero cohort seed overlap, zero C13-I
seed overlap, the exact runtime-information labels, and that neither training
nor target construction reads `dist_to_goal`.
