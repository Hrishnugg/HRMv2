# C13-M preregistration: matched-quality fresh confirmation

Date frozen: 2026-07-17, after the C13-L calibration and reopening mechanism
probe, before generating any confirmation world at seed offset 15,000,000.

## Why this is a distinct confirmation question

C13-L correctly rejected every direct arm under an absolute 1.10 worst-case
cost-ratio ceiling. The post-hoc mechanism probe then showed that reopening did
not remove the rare direct-search outliers, while certified FOCAL removed them
but cost too many expansions.

The C7 learned comparators are themselves unbounded direct A* arms. On the
locked 48-world C13-L block, `field_hrm` had mean/max graph-optimal ratios
1.0418/1.3533. The fixed suite-balanced local-backup arm at alpha 1.50 had
mean/max 1.0381/1.3660 while reducing expansions from 83.0208 to 69.9792. Its
paired expansion improvement was -13.0417 with 95% CI
[-18.6667, -7.6458], and every suite mean was negative.

Thus alpha 1.50 is not a formally bounded arm, but it is a development-stage
Pareto improvement at matched empirical path quality. C13-M asks whether that
result confirms on untouched worlds. The bounded reopening-FOCAL arm is carried
along as a separate safety operating point; the two claims are not conflated.

## Fixed current-state arm

- Model: C13-J suite-balanced flat MLP, iteration 8.
- Runtime observation: current-to-goal geometry, 32 rays truncated at
  `0.20 * side_len` with 32 samples, and up to 24 one-hop actions.
- Integration: one radius-0.20 local Dijkstra/Bellman backup using frozen
  learned exit-state values.
- Residual scale: alpha 1.50.
- Search: the exact C7 no-reopen direct A* ordering, with parent bookkeeping
  only for path validation.
- No occupancy raster, complete obstacle list, world descriptor, global graph
  solution, `dist_to_goal`, or shortest-path result is a runtime input.

The fixed safety control is the independently confirmed C13-H iteration-4
static rank, alpha 0.50, reopening `fhat` FOCAL at `w=1.10`.

## Untouched confirmation block

- 24 connected worlds from each of all six C7 suites (144 total).
- 192 roadmap nodes, `k=7`.
- C13 deterministic seed offset 15,000,000.
- Must have zero overlap with C13-I, all C13-J splits, C13-L calibration,
  original C13-H cohorts, and each other.
- Every current-state feature cache is hashed and graph-checked.
- All five C7 learned providers (`field_hrm`, `field_onlstm`, `field_unet`,
  `scalar_hrm`, `scalar_onlstm`) plus Euclidean and graph-oracle references are
  recomputed live on the shared roadmaps.
- Common generous budget: 384 expansions. No-reopen A* can expand at most 192
  distinct states; the extra budget permits legitimate safety-control reopens.

## Preregistered matched-quality gate

The fixed alpha-1.50 arm confirms only if:

1. it returns 144/144 valid paths;
2. the pooled paired bootstrap 95% CI upper endpoint for
   `expansions(current) - expansions(field_hrm)` is below zero;
3. mean expansion delta versus `field_hrm` is negative in at least four of six
   suites;
4. pooled mean current cost ratio is no more than 0.005 above pooled mean
   `field_hrm` cost ratio; and
5. maximum current cost ratio is no more than 0.02 above maximum `field_hrm`
   cost ratio.

The last two requirements are comparator-relative because this operating point
answers the same direct-A* quality/efficiency question as C7. They do not create
a formal bound. The safety-control report must separately show 144/144 valid
paths, zero observed `w=1.10` violations, and zero certificate violations.

Secondary analyses compare the fixed arm to each other fixed C7 provider; no
per-world best-provider oracle is allowed. Representation time, local-backup
time, provider time, and search time are reported separately.

## Outcome discipline

The 48 C13-L rows are development evidence only and are never pooled into this
confirmation. If C13-M fails any gate condition, the matched-quality claim is
rejected; no alpha, checkpoint, or threshold is changed on these 144 rows.

## Integrity

Hash this preregistration, implementation, selected checkpoint, safety
checkpoint, all C7 checkpoints, every feature cache, cohort records, raw rows,
summaries, verdict, verification, report, manifest, and suite shards. Record
all seed-overlap checks and runtime-information labels.
