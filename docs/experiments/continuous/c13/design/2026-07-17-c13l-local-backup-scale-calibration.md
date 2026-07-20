# C13-L preregistration: local-backup scale calibration

Date frozen: 2026-07-17, after C13-K failed its confidence gate and before
generating the C13-L calibration block.

## Motivation fixed from prior evidence

C13-K isolated a promising integration point. The suite-balanced iteration-8
model followed by the radius-0.20 local Bellman backup used 74.7083 expansions
versus 75.9167 for live `field_hrm`, had a negative suite delta in four of six
suites, and retained substantial path-quality headroom. It failed because the
24-world paired confidence interval was `[-8.0417, 5.4583]`, not because its
mean or suite coverage had the wrong sign.

The locked alpha grid ended at 1.00, while the response was strongly monotonic:
alpha 0.75 used 86.3333 expansions and alpha 1.00 used 74.7083. Alpha is the
strength of the locally backed-up residual relative to Euclidean distance. This
suggests an integration calibration issue: the learned/local value ordering is
useful but underweighted.

## What may and may not change

Fixed:

- suite-balanced flat-MLP checkpoint iteration 8 from C13-J;
- current-state features (32 rays, radius 0.20, 32 ray steps, 24 one-hop
  actions);
- one radius-0.20 local Bellman backup with frozen learned exit values;
- direct no-reopen A* matching C7; and
- C7 `field_hrm` as the primary full-map comparator.

The only tested variable is alpha: `1.00, 1.25, 1.50, 2.00`. No checkpoint,
radius, backup depth, architecture, target, suite weighting, or search algorithm
is tuned.

## New calibration block

- Eight connected 192-node, `k=7` roadmaps per each of the six C7 suites
  (48 total).
- Cohort seed offset: 12,500,000 under the C13-J deterministic recipe.
- The block must be disjoint from C13-I, C13-J train/validation/development,
  and all earlier C13-H cohorts.
- Current-state features are cached and hashed per accepted roadmap.
- `field_hrm` and `scalar_hrm` are recomputed live from their saved checkpoints.

## Gate and selection

An alpha passes only if:

1. all 48 returned paths are valid;
2. maximum graph-optimal cost ratio is at most 1.10;
3. the pooled paired bootstrap 95% CI upper endpoint for
   `expansions(current) - expansions(field_hrm)` is below zero; and
4. mean expansion delta is negative in at least four of six suites.

Choose the passing alpha with the lowest mean current expansions, then lower
maximum cost ratio, then lower alpha. If none passes, amplification is rejected
and the next intervention must add online/search-history state.

## Untouched confirmation

A pass authorizes exactly one six-suite confirmation using the selected alpha
on 24 worlds per suite at seed offset 15,000,000. The final gate additionally
requires the pooled mean current path ratio to be no more than 0.01 above
`field_hrm`. All five C7 learned providers and the Euclidean/oracle references
must be recomputed live. No result from the calibration block may be pooled into
confirmation.

## Claim boundary

This remains an empirical direct-search operating point, not a formal 1.10
guarantee. The formally bounded reopening-FOCAL arm remains the safety control.
The runtime value is state-conditioned local lookahead; it does not consume the
occupancy raster, complete obstacle list, global Dijkstra values, or
`dist_to_goal`.
