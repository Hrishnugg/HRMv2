# C13-K preregistration: local Bellman integration rescue

Date frozen: 2026-07-17, after C13-J failed and before evaluating any local-
backup candidate on the C13-J development seed block.

## Correction to the earlier interpretation

C13-I and C13-J show that a static learned residual is not enough outside the
maze-only setting. C13-J's suite-balanced static model was still worse than
`field_hrm` in every development suite. This rules out training-distribution
coverage as the sole explanation.

An integration result was present in the pre-C7 C13-H candidate study but was
not carried into C13-I. Under direct additive A*, the iteration-8 maze model
followed by a radius-bounded local Bellman backup achieved:

- 63.7708 mean expansions versus 143.9375 for Euclidean A*;
- 48 wins, 0 ties, 0 losses; and
- mean graph-optimal cost ratio 1.0079, maximum 1.0433.

The corresponding static model used 81.6667 expansions. We previously selected
the static variant because the candidate gate was defined for repaired FOCAL,
where the backup was not best. That did not answer whether the backup was the
right **direct-search integration**. C13-K tests that question explicitly.

## Operator and information boundary

For a queried state `s`, C13-K:

1. observes the same current-to-goal geometry, 32 rays truncated to
   `0.20 * side_len`, and one-hop actions as C13-H/J;
2. forms the learned frozen exit-state value `V(exit)`;
3. runs Dijkstra only inside the physical radius around `s`; and
4. returns the minimum local path cost to the goal (if locally reached) or to a
   boundary-crossing action plus `V(exit)`.

This is a state-conditioned local lookahead. It does not read the occupancy
raster, complete obstacle list, graph `dist_to_goal`, a global Dijkstra result,
or nodes recursively beyond the radius. The implementation may batch the same
per-state query for all roadmap nodes for timing convenience, but each value is
defined solely by that state's bounded neighborhood and frozen boundary values.
Representation and backup time must be reported separately from A* search time.

## Development candidates

The locked C13-J development block (four new worlds from each of six suites) is
used for selection. It is disjoint from training, validation, C13-I, and the
future confirmation block.

Candidates are:

- the original C13-H maze-trained flat-MLP iteration 8; and
- C13-J suite-balanced flat-MLP iterations 1 through 8.

Every model is followed by the same radius-0.20 local Bellman operator. The
only alpha grid is `0.50, 0.75, 1.00`; no radius, depth, target, or architecture
is tuned on these rows. Live `field_hrm` and `scalar_hrm` outputs are recomputed
on the same roadmaps.

## Development gate and selection

A candidate passes only if it:

1. solves all 24 roadmaps;
2. has maximum graph-optimal cost ratio at most 1.10;
3. has a pooled paired bootstrap 95% CI upper endpoint below zero for
   `expansions(current local backup) - expansions(field_hrm)`; and
4. has a negative mean expansion delta in at least four of six suites.

Among passing candidates, select the lowest current mean expansions, then lower
maximum cost ratio, then prefer the suite-balanced model, then earlier iteration
and lower alpha. If no candidate passes, local Bellman integration is rejected
and the next intervention must add genuine online/search-history state.

## Untouched confirmation

A development pass authorizes exactly one 24-world-per-suite confirmation on
seed offset 15,000,000. The selected model, iteration, alpha, radius, and search
algorithm are frozen. All five C7 learned providers are recomputed live.

The solid-improvement gate is the same as C13-I/J:

- 144/144 valid paths;
- maximum cost ratio at most 1.10;
- pooled paired 95% CI upper endpoint below zero versus `field_hrm`;
- negative mean delta in at least four of six suites; and
- pooled mean cost ratio no more than 0.01 above `field_hrm`.

The fixed reopening-FOCAL control remains the formal safety point; the direct
local-backup arm is an empirical path-quality operating point and must not be
described as carrying a formal 1.10 proof.

## Integrity

Hash the preregistration, implementation, C13-H and C13-J checkpoints, C13-J
development feature caches/cohort records, C7 comparator checkpoints, raw
rows, summaries, selection, verification, manifest, and (if authorized) every
fresh-confirmation shard. Record information boundaries and all seed overlaps.
