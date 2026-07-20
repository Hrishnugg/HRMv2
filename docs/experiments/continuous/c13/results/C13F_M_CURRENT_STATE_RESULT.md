# C13-F through C13-M: current-state local-Bellman heuristic result

**Date:** 2026-07-17  
**Status:** complete; C13-M preregistered fresh confirmation passed  
**Canonical claim:** a fixed heuristic built from bounded observations of each
search state and one radius-bounded local Bellman backup reduces A* expansions
relative to C7's complete-map field HRM on 144 untouched worlds, while also
having lower empirical mean and worst path-cost ratios on that cohort.

## Result in one paragraph

The fixed C13-M arm averages **68.306 expansions** versus **81.264** for the C7
map-conditioned field HRM, a paired reduction of **12.958 expansions** or
**15.95%**. The world-paired bootstrap 95% interval for current minus field-HRM
is **[-16.299, -9.743]**; the current-state arm wins 109 worlds, ties 3, and
loses 32. All six suite mean deltas are negative. Its mean/max graph-optimal
path-cost ratios are **1.02353/1.16244**, compared with
**1.03111/1.33459** for field HRM. All 144 current-state paths are valid. A
separate reopening-FOCAL operating point at `w=1.10` also returns 144/144 valid
paths with zero observed bound or certificate violations. The primary direct
A* result is an empirical matched-quality improvement, not a formal
suboptimality guarantee and not an end-to-end wall-clock speedup.

## What “current-state” means here

The methodological requirement inherited from the professor meeting is that
the heuristic for state `s` be a function of `s` and bounded observations
around `s`, rather than a representation of the entire map or a map-wide
shortest-path solution.

The fixed C13-M provider receives, for every queried roadmap state:

- current-to-goal geometry;
- 32 collision rays truncated at `0.20 * side_len`, with 32 samples per ray;
- up to 24 one-hop collision-free roadmap actions;
- the roadmap nodes and edges inside the same physical radius for one local
  Dijkstra/Bellman backup; and
- frozen learned values at edges leaving that local region.

It does **not** receive a `64 x 64` occupancy raster, the complete obstacle
list, map-wide reachability or clearance channels, `dist_to_goal`, grid
Dijkstra, A*, or any other full-problem solution as a feature or training
target. Graph shortest paths are used only after search for evaluation and
safety auditing.

This is not a “map-free robot” claim. The planner operates on a known PRM, and
the current implementation batches bounded observations for all 192 candidate
states before search. The defensible distinction is **state-conditioned,
radius-bounded information versus complete-map conditioning**, not known map
versus unknown map.

The harness still retains the full obstacle geometry and PRM to generate the
roadmap and to simulate clipped collision rays, one-hop actions, and local
subgraphs. Those global objects are not exposed as model tokens or search
ranks; they are the observation simulator. A locality test changes obstacles
outside the sensing radius and requires the queried state tokens to remain
identical. This is therefore a representation/information-boundary result, not
an implementation of online sensing from an unknown environment.

Normalized absolute current and goal coordinates are part of the state tokens.
The model can consequently learn spatial priors shared by the benchmark suite
families. C13-M establishes fresh-world generalization within those families,
not coordinate-invariant or arbitrary-layout transfer.

### Comparator information boundaries

| Arm | Runtime information |
|---|---|
| C13-M current local backup | Current/goal geometry, bounded rays, one-hop actions, radius-bounded local PRM, frozen learned exit values |
| C13 bounded safety control | Current/goal geometry, bounded rays, one-hop actions |
| C7 `field_*` | Complete `64 x 64` occupancy/goal raster |
| C7 `scalar_*` | Global obstacle-list summaries, sectors, rays, and goal geometry |
| Euclidean | Current and goal geometry only |
| Graph oracle | Exact full-graph cost-to-go; evaluation reference only |

## Fixed method

### Training

The selected model is the C13-J suite-balanced flat MLP at outer iteration 8.
Training uses 96 worlds (32 each from maze, rooms, and spiral), validation uses
24 worlds (8 per training suite), and development uses 24 worlds (4 from each
of all six C7 suites). Splits are disjoint by world seed and do not overlap
C13-I.

The model predicts a normalized nonnegative residual over Euclidean distance.
Targets are produced by repeated local heuristic Bellman learning:

1. start from Euclidean or the prior iteration's frozen prediction;
2. run one local Dijkstra backup inside radius `0.20 * side_len`;
3. terminate locally at the goal or at an exit edge plus the frozen value of
   the outside endpoint; and
4. regress the resulting residual from the bounded state observation.

No iteration reads graph shortest-path cost-to-go as a label. Graph oracle
values appear only in held-out diagnostic correlations and cost ratios.

### Runtime integration

For each state, let `h_E` be Euclidean, `h_M` the learned bootstrap value, and
`B_0.20(h_M)` one radius-bounded local backup. The fixed rank is

```text
h_rank(s) = h_E(s) + 1.50 * (B_0.20(h_M)(s) - h_E(s)).
```

The primary arm uses the exact no-reopen A* ordering used by the learned C7
arms. The independent safety point uses the original one-suite iteration-4
model with alpha `0.50` inside reopening `fhat` FOCAL at `w=1.10`.

The two operating points answer different questions:

- primary: empirical expansion efficiency at path quality matched to the C7
  direct-A* comparators;
- safety: bounded-suboptimality behavior with a direct anchor certificate.

They must not be conflated.

## Why the final design was necessary

C13-F through C13-M isolate scale, target, representation, training
distribution, and search integration rather than treating every failure as a
model-capacity problem.

| Stage | Question | Result | Decision |
|---|---|---|---|
| C13-F | Is the rollout failure only a scale/calibration problem? | No calibration candidate clears the unchanged bounded gate. | Replace the behavior-return objective. |
| C13-G | Does an exact radius-bounded local-escape heuristic have enough bounded-FOCAL headroom? | No. Radius 0.20 loses all six primary comparisons; the exit-stub variant wins only 1/6. | Learn a local Bellman value and examine direct ordering. |
| C13-H | Can local heuristic Bellman learning produce a stable current-state provider? | Yes on fresh maze worlds at both 192 and 211 nodes under bounded FOCAL. | Run a live six-suite C7 comparison. |
| C13-I | Does the one-suite provider beat map-conditioned C7 arms across all suites? | No. It is `+15.36` expansions worse than field HRM pooled and negative in only one suite. | Train across multiple suites. |
| C13-J | Is training distribution alone the missing piece? | No. Static direct integration remains `+16.17` expansions worse on the 24-world development set. | Change online integration. |
| C13-K | Does one local Bellman backup repair ranking? | Nearly. Iteration 8/alpha 1.0 changes the pooled delta to `-1.21`, negative in four suites, but CI crosses zero. | Calibrate local-backup scale on fresh worlds. |
| C13-L | Is there a reproducible efficiency/quality frontier? | Yes empirically at alpha 1.5, but all arms fail the absolute 1.10 worst-case ceiling. | Preregister a comparator-relative matched-quality confirmation and retain a separate bounded control. |
| Mechanism probe | Will reopening remove direct-A* cost outliers? | No; reopening adds work and does not remove the outliers. Certified FOCAL is safe but slower. | Keep direct and bounded operating points separate. |
| C13-M | Does fixed alpha 1.5 confirm against live C7 providers on untouched worlds? | **Yes. All five primary conditions and all three safety-control conditions pass.** | Document the bounded claim scope. |

## C13-F: scale versus ordering

C13-F freezes C13-D's successful shared-state search and varies only the scale
of the exact C13-B rollout rank. The same-search Euclidean rank averages
`138.17` expansions, compared with `129.67` for matched Euclidean FOCAL. The
uncalibrated exact rank averages `131.00`; it beats the same-search Euclidean
rank by `7.17` but remains `+1.33` worse than FOCAL and wins only 2/6 worlds.
No fixed calibration clears both comparison gates. This rules out the narrow
explanation that C13-E failed only because its rank was numerically too large.

Evidence: `runs/c13_shared_queue_calibration/`.

## C13-G: exact local-escape ceilings

The first LoHA*-inspired construction runs exact local Dijkstra inside a
physical radius and terminates at a local goal or an exit edge plus Euclidean
distance. At radius 0.20 it observes 20.79 nodes on average and returns optimal
paths, but averages `136.50` expansions versus `129.67` for matched FOCAL. It
loses all six primary comparisons. Its start heuristic is only `0.431x` graph
oracle on average, so exact locality is too weak under this insertion.

An exit-stub variant uses a locally visible exit plus a frozen continuation
value. It improves the mean to `133.00` expansions and beats same-search
Euclidean on all six worlds, but still produces only 1 win, 1 tie, and 4 losses
against FOCAL. Both implementations pass path, locality, and safety checks.

The failure is informative: exact bounded local information exists, but a
single shallow analytical backup is underpowered in the bounded shared-queue
integration.

Evidence: `runs/c13_local_escape/` and
`runs/c13_local_escape_exit_stub/`.

## C13-H: local heuristic Bellman learning

C13-H iterates the local backup, using the previous model as the frozen exit
value. A one-suite flat MLP is trained on 48 maze worlds and validated on 12.
On the six-world development audit, iteration 6/alpha 1.0 averages `125.67`
expansions versus `129.67` for matched FOCAL and wins all six. Direct A* with
the same learned value averages `92.67` expansions with a maximum cost ratio
of `1.01451`, showing that the learned ordering is substantially stronger than
the bounded secondary-key result.

Because selecting iteration 6 on six worlds would be fragile, the full
candidate study compares checkpoints, alphas, direct values, local backups,
and bounded FOCAL across two 12-world development cohorts at 192 and 211
nodes. It selects the more conservative iteration-4/alpha-0.5 bounded-Focal
arm. A third untouched cohort then passes at both densities:

| Density | Current expansions | Euclidean-control expansions | Paired delta | 95% CI | W/T/L |
|---:|---:|---:|---:|---:|---:|
| 192 | 137.417 | 141.833 | -4.417 | [-6.333, -2.750] | 12/0/0 |
| 211 | 149.917 | 156.833 | -6.917 | [-10.583, -4.000] | 11/1/0 |

All paths pass the `w=1.10` safety checks. This establishes a real one-suite
current-state signal, but not yet superiority to C7's map-conditioned models.

Evidence: `runs/c13_lhbl_flat_48w/`,
`runs/c13_lhbl_candidate_study/`, and
`runs/c13_lhbl_focal_fresh3/`.

## C13-I: live comparison exposes the representation/distribution failure

C13-I reruns all five learned C7 providers, Euclidean, and the graph oracle on
the exact 144 historical C7 worlds. The fixed one-suite current arm is valid
on all 144 worlds, and its bounded control has zero bound/certificate
violations. It nevertheless fails decisively against field HRM:

- current `98.292` versus field HRM `82.931` expansions;
- paired delta `+15.361`, 95% CI `[+12.021, +18.639]`;
- 27 wins, 1 tie, 116 losses;
- negative mean delta in only maze; and
- maximum current path-cost ratio `1.2073`, above the absolute 1.10 gate.

Suite deltas are maze `-5.08`, dense maze `+25.75`, rooms `+16.00`, spiral
`+25.29`, bugtrap `+12.92`, and large rooms `+17.29`. One-suite learning does
not transfer to the complete C7 distribution.

The live replay has six historical parity differences, all in the secondary
field-U-Net arm: two budgets each for maze world 22 (one expansion), rooms
world 19 (one expansion), and spiral world 18 (same expansions, slightly
different path cost). Field HRM, the primary comparator, has exact parity.
The C13-I scientific failure remains interpretable, but strict artifact
verification is marked false rather than hiding the U-Net drift.

Evidence: `runs/c13_lhbl_c7_comparison/`.

## C13-J: multi-suite training is necessary but not sufficient

C13-J uses 96 training worlds from maze, rooms, and spiral; 24 validation
worlds; and a fresh four-per-suite development block. Eight outer Bellman
iterations raise the training target mean from `0.105` to `0.734`. Validation
MAE rises from `0.0436` to `0.0944` as the bootstrapped target becomes larger,
but checkpoints remain finite and provenance-clean.

The strongest static direct-A* cell is iteration 8/alpha 1.0, yet it averages
`92.083` expansions versus `75.917` for field HRM: delta `+16.167`, 95% CI
`[+9.667, +22.667]`, with no suite mean negative. Changing only the training
distribution does not solve the planner mismatch.

Evidence: `runs/c13_lhbl_multisuite/`.

## C13-K: local Bellman integration is the missing mechanism

C13-K applies one radius-bounded Bellman backup at inference, using learned
predictions only as local exit values. With the same C13-J iteration-8 model,
alpha 1.0 changes the 24-world development result from static `+16.17` to
`-1.21` expansions relative to field HRM:

- current `74.708` versus field HRM `75.917`;
- paired 95% CI `[-8.042, +5.458]`;
- negative mean delta in four of six suites; and
- mean/max current cost ratio `1.00482/1.03628`.

The interval crosses zero, so C13-K correctly rejects a claim. Mechanistically,
however, the one local backup is the decisive change: it converts a clearly
inferior static model into a near tie without changing training or adding
complete-map inputs.

Evidence: `runs/c13_local_bellman_integration/`.

## C13-L and the reopening probe: choose the question honestly

C13-L freezes the C13-J iteration-8 model and local radius 0.20, then evaluates
alpha `{1.00, 1.25, 1.50, 2.00}` on a fresh 48-world block, eight per suite.

| Alpha | Current exp. | Field HRM exp. | Delta | 95% CI | Negative suites | Current mean/max cost ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 79.229 | 83.021 | -3.792 | [-9.396, +1.730] | 4 | 1.0220 / 1.2408 |
| 1.25 | 73.250 | 83.021 | -9.771 | [-15.375, -4.417] | 4 | 1.0313 / 1.3730 |
| 1.50 | 69.979 | 83.021 | -13.042 | [-18.667, -7.646] | 6 | 1.0381 / 1.3660 |
| 2.00 | 66.083 | 83.021 | -16.938 | [-22.563, -11.542] | 6 | 1.0522 / 1.3660 |

Every alpha fails the preregistered absolute 1.10 maximum-cost ceiling, so
C13-L's official verdict is rejection. That result must remain recorded.

At alpha 1.50, however, field HRM itself has mean/max ratios
`1.04179/1.35330`; current-state has `1.03805/1.36602`. Current-state is much
faster in expansions, has slightly lower mean cost, and differs in maximum by
only `+0.01272`. This is a legitimate development-stage matched-quality
Pareto point against a comparator that is also unbounded direct A*.

A post-hoc mechanism probe tests whether reopening can remove the direct-A*
outliers. It cannot. At alpha 1.50, reopening-first-goal increases expansions
from `69.98` to `82.98` and still has maximum cost ratio `1.3375`.
Reopening `fhat` FOCAL is safe (max `1.06394`) but rises to `113.38`
expansions, much slower than field HRM. This motivates two distinct operating
points instead of retrofitting a bound onto the direct arm.

Evidence: `runs/c13_local_backup_scale/` and
`runs/c13_reopening_rank_probe/`.

## C13-M preregistered fresh confirmation

### Frozen before data generation

- model: C13-J suite-balanced flat MLP iteration 8;
- local backup: one application, physical radius 0.20;
- residual scale: alpha 1.50;
- search: exact C7 no-reopen direct A*;
- safety control: original iteration 4, alpha 0.50, reopening `fhat` FOCAL,
  `w=1.10`;
- cohort: 24 worlds from each of all six suites, seed offset 15,000,000;
- common budget: 384 expansions; and
- all five learned C7 providers recomputed live.

The 144 confirmation seeds have zero overlap with C13-I, every C13-J split,
C13-L, or the original C13-H train/validation/search cohorts. Every feature
cache is tied to graph node/edge counts and hashed. During evaluation, features
are regenerated live and required to equal the cache exactly; mismatches are
zero.

### Preregistered gate

The primary arm must return 144 valid paths, have pooled paired-CI upper bound
below zero versus field HRM, have negative mean delta in at least four suites,
have mean cost ratio no more than `+0.005` above field HRM, and have maximum
cost ratio no more than `+0.02` above field HRM. The bounded control separately
requires 144 valid paths and zero bound/certificate violations.

Every condition passes.

### Primary comparison by suite

| Suite | Current exp. | Field HRM exp. | Delta | 95% CI | W/T/L | Current cost mean | Field cost mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Maze | 47.625 | 84.375 | -36.750 | [-45.583, -27.833] | 24/0/0 | 1.03361 | 1.01546 |
| Dense maze | 95.375 | 104.500 | -9.125 | [-15.208, -3.125] | 18/1/5 | 1.02723 | 1.01307 |
| Rooms | 98.042 | 106.167 | -8.125 | [-14.333, -1.708] | 16/0/8 | 1.01340 | 1.01466 |
| Spiral | 120.250 | 121.333 | -1.083 | [-5.875, +4.208] | 14/1/9 | 1.01645 | 1.02032 |
| Bugtrap | 14.958 | 24.292 | -9.333 | [-17.792, -2.917] | 18/1/5 | 1.02715 | 1.09596 |
| Large rooms | 33.583 | 46.917 | -13.333 | [-18.917, -7.500] | 19/0/5 | 1.02336 | 1.02720 |
| **Pooled** | **68.306** | **81.264** | **-12.958** | **[-16.299, -9.743]** | **109/3/32** | **1.02353** | **1.03111** |

The spiral suite mean is negative but its suite-specific interval crosses
zero. The claim is pooled with a breadth check, not that every suite is
individually significant.

### Comparison with every learned C7 arm

| Comparator | Comparator exp. | Current minus comparator | 95% CI | W/T/L |
|---|---:|---:|---:|---:|
| Field HRM | 81.264 | -12.958 | [-16.299, -9.743] | 109/3/32 |
| Field ON-LSTM | 87.444 | -19.139 | [-22.306, -15.965] | 119/1/24 |
| Field U-Net | 88.278 | -19.972 | [-23.021, -16.979] | 128/3/13 |
| Scalar HRM | 77.604 | -9.299 | [-12.479, -6.222] | 102/6/36 |
| Scalar ON-LSTM | 76.778 | -8.472 | [-11.785, -5.104] | 96/2/46 |
| Euclidean | 109.340 | -41.035 | [-45.792, -36.326] | 139/0/5 |

No per-world best-provider oracle is used. Each row compares the one fixed
current arm with one fixed provider.

### Path quality and safety

| Arm | Mean cost ratio | Maximum cost ratio |
|---|---:|---:|
| Current local backup, alpha 1.50 | 1.023533 | 1.162438 |
| Field HRM | 1.031112 | 1.334591 |
| Field ON-LSTM | 1.042602 | 1.386084 |
| Field U-Net | 1.033800 | 1.262486 |
| Scalar HRM | 1.055347 | 1.380292 |
| Scalar ON-LSTM | 1.049109 | 1.412050 |
| Bounded current FOCAL | 1.003523 | 1.063860 |

The primary current arm is not formally bounded; its empirical maximum is
well below every learned C7 comparator on this cohort. The bounded control has
zero observed `w=1.10` violations and zero direct certificate violations.

### Runtime accounting

| Arm | Representation/provider s | Model s | Local backup s | Search s |
|---|---:|---:|---:|---:|
| Current local backup | 5.1384 | 0.0008 | 0.0097 | 0.0002 |
| Field HRM | 0.3707 | included | 0 | 0.0003 |
| Scalar HRM | 1.5273 | included | 0 | 0.0003 |

The current prototype is **not an end-to-end runtime win**. Its Python
feature builder scans bounded observations for all 192 states and dominates
elapsed time. The network and local backup are cheap; feature extraction is
the engineering bottleneck. The confirmed scientific result is fewer search
expansions at better empirical path quality under a stricter information
boundary. A vectorized or lazy feature builder must be measured before making
latency or deployment-efficiency claims.

## Mechanism interpretation

The full sequence supports a specific causal story.

1. **Representation is not enough by itself.** C13-J's multi-suite model is
   clearly worse than field HRM when inserted statically.
2. **Training distribution is necessary.** The one-suite model fails five of
   six C7 suites even though it confirms on fresh maze worlds.
3. **Search integration is decisive.** One local Bellman backup changes the
   multi-suite iteration-8 delta from `+16.17` to `-1.21` on the same
   development cohort.
4. **Residual scale exposes a Pareto frontier.** Alpha 1.50 trades a modest
   unbounded path-quality tail for a large expansion reduction, at empirical
   quality matching the equally unbounded C7 learned arms.
5. **Reopening is not the missing fix.** It adds expansions without removing
   the tail. Formal safety requires the slower FOCAL operating point.
6. **The final result comes from representation plus local computation plus
   integration.** It is not evidence that the flat MLP alone is superior to
   HRM, nor that complete-map information is generally harmful.

The largest gain is on maze, where bounded local geometry plus learned exit
values appears especially useful. Bugtrap and large rooms also improve
substantially. Spiral is nearly tied. This heterogeneity argues for reporting
the full suite table rather than only the pooled headline.

## Claim policy

### Supported

- On the fixed 144-world C13-M cohort, the current-state/local-subgraph arm
  reduces direct-A* expansions by 15.95% relative to C7 field HRM, with a
  paired CI excluding zero.
- It has lower pooled mean and maximum empirical path-cost ratios than field
  HRM on that cohort.
- All six suite mean expansion deltas are negative.
- It beats every fixed C7 learned provider in pooled expansions.
- A separate current-state FOCAL operating point passes the observed
  `w=1.10` safety and certificate checks on all 144 worlds.
- The result does not use complete-map or shortest-path supervision for the
  current model.

### Not supported

- A formal bound for the alpha-1.50 direct-A* arm.
- End-to-end latency, energy, or deployment speed superiority.
- Unknown-map or sensor-online navigation; the current harness has a known PRM
  and batches every state's local observation.
- Coordinate-invariant or arbitrary-layout generalization; normalized absolute
  current/goal position is part of the representation, and confirmation uses
  the same six generator families.
- General superiority of flat MLPs, HRMs, local methods, or current-state
  methods outside this benchmark.
- Individual statistical significance in every suite; spiral's interval
  crosses zero.
- A claim that C13-I reproduced every historical C7 neural row exactly; six
  secondary field-U-Net rows drifted.

Recommended publication wording:

> On 144 untouched PRM worlds spanning six static navigation suites, a fixed
> bounded-observation heuristic with one radius-limited Bellman backup reduced
> A* expansions by 15.9% relative to a complete-map field HRM (paired mean
> -12.96 expansions, bootstrap 95% CI [-16.30, -9.74]) while attaining lower
> empirical mean and worst path-cost ratios. The direct arm is not formally
> bounded, and its unoptimized feature construction is slower in wall time.

## Reproducibility and integrity

Primary implementation:

- `hrm-cloud/continuous_prm/continuous_prm_c13_matched_quality_confirmation.py`
- `hrm-cloud/continuous_prm/tests/test_c13_matched_quality_confirmation.py`
- preregistration:
  `docs/experiments/continuous/c13/design/2026-07-17-c13m-matched-quality-confirmation.md`

Primary run:

- `hrm-cloud/continuous_prm/runs/c13_matched_quality_confirmation/`
- `results/confirmation_raw.csv`: 1,296 rows = 144 worlds x 9 arms;
- `results/pairwise_summary.csv`: fixed pairwise comparisons;
- `results/gate_verdict.json`: all five primary conditions pass;
- `results/verification.json`: `integrity_pass=true`;
- `integrity.json`: implementation, preregistration, both current checkpoints,
  all five C7 checkpoints, all 144 feature caches, cohort, raw rows, summaries,
  report, manifest, and six suite shards are hashed.

Selected checkpoint SHA-256 values:

- C13-J iteration 8:
  `39D1E145BF5DEB67E7D3281B784DC36810C06E5A8E6193A68590F012132C91C4`;
- bounded-control iteration 4:
  `DBFD516E3DB8AC616F0A3A48F5323FBF1C12405C178EE50C5792388D70B64742`.

Verification records:

- 144 unique confirmation seeds and exactly 24 worlds per suite;
- zero overlap with all enumerated prior C13 cohorts;
- 1,296/1,296 expected rows;
- zero duplicate world/arm keys;
- zero invalid found paths;
- zero live-feature/cache mismatches;
- all seven C7 providers loaded and evaluated live;
- zero safety-control bound violations; and
- zero safety-control certificate violations.

Focused command:

```bash
cd hrm-cloud/continuous_prm
python -m pytest tests/test_c13_matched_quality_confirmation.py -q
python continuous_prm_c13_matched_quality_confirmation.py
```

The runner is resumable only at whole-world boundaries and refuses to append
to an output whose implementation, preregistration, config, checkpoint, or
arm fingerprint has changed.

## Remaining work

The scientific objective raised in the professor meeting is now satisfied at
the benchmark level: the successful arm is tied to the current search state
and bounded local observation, and it improves over complete-map C7 arms on a
fresh six-suite comparison. The next work is hardening rather than result
search:

1. vectorize or lazily construct local features and rerun timing without
   changing ranks;
2. replicate across independent model seeds and a second 144-world cohort;
3. test 211-node and larger-density multi-suite generalization;
4. decide whether the paper presents the matched-quality direct arm, the
   bounded but slower arm, or both as an explicit Pareto frontier; and
5. review the exact state-information wording with the professor before an
   external draft.

## Literature basis

- [Learning Local Heuristics for Search-Based Navigation Planning
  (LoHA*, ICAPS 2023)](https://ojs.aaai.org/index.php/ICAPS/article/view/27245)
- [Real-Time Adaptive A*](https://cdn.aaai.org/Workshops/2006/WS-06-11/WS06-11-010.pdf)
- [Multi-Heuristic A*](https://publications.ri.cmu.edu/multi-heuristic-a-2)
- [Policy-Guided Heuristic Search with Guarantees](https://ojs.aaai.org/index.php/AAAI/article/view/17469)
- [Learning Heuristic Search via Imitation](https://proceedings.mlr.press/v78/bhardwaj17a/bhardwaj17a.pdf)

