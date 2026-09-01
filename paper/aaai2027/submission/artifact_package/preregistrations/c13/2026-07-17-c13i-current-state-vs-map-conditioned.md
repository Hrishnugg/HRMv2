# C13-I preregistration: current-state LHBL vs C7 map-derived heuristics

Date frozen: 2026-07-17, before evaluating the C13-H checkpoints on the C7
roadmaps.

## Question

Can a heuristic whose runtime observation is tied to the current search state
produce a solid search improvement over the earlier C7 heuristics that consume
map-derived information, when all arms search the exact same PRM graphs?

This is the comparison requested after the professor's objection that the C7
value field describes the map rather than the planner's current state. It is not
a claim that a partial observation contains the whole map. It tests whether a
state-conditioned model can learn a useful search policy/value signal without
receiving the full occupancy raster or global obstacle list at runtime.

## Information boundaries

The C13-H current-state learner receives, for each candidate state:

- current-to-goal geometry;
- 32 range rays, truncated to radius `0.20 * side_len`, sampled with 32 steps;
- up to 24 one-hop roadmap actions with their local geometry; and
- a learned bootstrap value at the boundary during training.

It does **not** receive the occupancy raster, the complete obstacle list, graph
shortest-path distance, or a whole-map value field at runtime. Its training
target is the limited-horizon Bellman target; `dist_to_goal` is evaluation-only.

C7 references have different, explicitly labeled boundaries:

- `field_*`: the complete `64 x 64` occupancy/goal raster, followed by full-map
  value-field interpolation at roadmap nodes;
- `scalar_*`: map-derived node features that enumerate the world's obstacle
  list for global summaries and angular sectors, plus rays and goal geometry;
- `oracle`: exact graph cost-to-go, an evaluation ceiling rather than a
  deployable arm; and
- `euclid`: goal geometry only.

The primary full-map comparator is `field_hrm`; `field_onlstm` and `field_unet`
are secondary full-map comparators. `scalar_hrm` and `scalar_onlstm` are reported
as map-derived scalar references, not mislabeled as raster models.

## Development-only operating-point selection

Selection uses only the 48 already-observed C13-H development roadmaps in
`c13_lhbl_candidate_study`; no C7 current-model output may be read first. For
each model-only `(iteration, alpha)` direct-additive A* candidate:

1. require exactly 48 finite results;
2. require fewer expansions than Euclidean A* on all 48 roadmaps;
3. require the maximum development path-cost ratio to be no greater than the
   stated ceiling; and
4. among eligible candidates, choose the lowest mean expansion count, breaking
   ties by lower maximum cost ratio, then earlier iteration, then lower alpha.

Two points are frozen because they represent two declared deployment regimes,
not because either was selected on C7:

| Regime | Development ceiling | Frozen checkpoint | Alpha | Development result |
|---|---:|---:|---:|---|
| low distortion | 1.05 | flat MLP iteration 8 | 0.75 | 48/48 wins; mean 104.15 vs 143.94; max ratio 1.037 (exact aggregate is emitted by the runner) |
| matched `w=1.10` throughput | 1.10 | flat MLP iteration 8 | 1.00 | 48/48 wins; mean 81.67 vs 143.94; max ratio 1.066 |

The table's rounded values are descriptive. The comparison runner recomputes
the selection from raw rows and refuses to run if either selected point changes.

The already selected and independently confirmed bounded arm is also frozen:
flat MLP iteration 4, `alpha=0.50`, reopening `fhat` FOCAL, `w=1.10`. This arm
is the safety/control operating point; it was the C13-H fresh3 pass at both 192
and 211 nodes.

## Matched benchmark

- C7 configuration: seed 1234, 192 nodes, `k=7`, 24 connected worlds per suite.
- Suites, in the original C7 order: `C_hard_maze`, `C_hard_maze_dense`,
  `C_hard_rooms`, `C_hard_spiral`, `C_hard_bugtrap`, and
  `C_hard_rooms_large`.
- The generator, retry/skip rule, PRM seed, checkpoints, calibration, and
  historical raw file are the saved `runs/c7_local` artifacts.
- Budgets: both saved calibrated budgets per suite, plus a generous common
  budget of `2 * roadmap_nodes = 384` for complete paired expansion and path
  comparisons. The extra budget does not let no-reopen A* expand more than all
  192 nodes; it also accommodates legitimate re-expansions in the repaired
  bounded FOCAL arm.
- Every heuristic array is computed live from the saved checkpoint on the
  regenerated shared world/roadmap. Saved C7 raw rows are used only for parity
  verification, not substituted for live inference.
- Direct A* uses the exact C7 no-reopen ordering and tie-breaking, augmented only
  with parent pointers so every returned path can be validated.

## Primary analyses frozen before the run

At budget 384, report for every arm and suite:

- solved and valid paths out of 24;
- mean/median expansions and bootstrap 95% confidence intervals;
- paired wins/ties/losses and paired mean expansion difference;
- mean and maximum graph-optimal cost ratio;
- representation/inference time separately from search time; and
- current-state vs each fixed C7 learned provider, never a per-world oracle
  choice of the best learned arm.

The primary comparison is the `matched w=1.10 throughput` current-state arm
against `field_hrm`. The low-distortion and formally bounded arms are mandatory
secondary operating points. Comparisons against all other fixed C7 providers
are reported without silently replacing the primary comparator.

## Solid-improvement gate

The primary gate passes only if the throughput current-state arm:

1. returns 144/144 valid paths at budget 384;
2. has no observed path ratio above 1.10;
3. has a pooled paired bootstrap 95% confidence interval whose upper endpoint
   for `expansions(current) - expansions(field_hrm)` is below zero;
4. has a negative mean expansion difference in at least four of six suites; and
5. has a pooled mean path ratio no more than 0.01 above `field_hrm`.

The low-distortion point has a separate descriptive Pareto check using its 1.05
development regime; it cannot rescue a failed primary gate by post-hoc tuning.
The bounded FOCAL arm must have zero invalid paths, zero `w=1.10` violations,
and zero certificate failures, but it is not required to be the fastest arm.

If the primary gate fails, the result is diagnostic. No alpha, checkpoint,
budget, or suite may be tuned on these rows and then described as confirmation.
The next experiment must train a new current-state model (for example on a
multi-suite cohort) and confirm it on a new seed block.

## Integrity requirements

The run must record SHA-256 hashes for this preregistration, implementations,
C13 candidate-study raw data and checkpoints, all five C7 learned checkpoints,
C7 calibration and historical raw rows, per-suite shards, summaries, verdict,
and verification files. Verification must include exact world counts, seed
uniqueness, path validity, C7 historical parity, and the frozen information
boundary labels above.
