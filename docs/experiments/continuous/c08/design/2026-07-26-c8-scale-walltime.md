# C8-S: Scale study — does the expansion advantage convert to wall time? (frozen design)

**Status: FROZEN 2026-07-26, before any implementation was executed.**
**Chronology label: post-submission extension** — designed after the v12 paper text
was frozen for the AAAI-27 full-paper deadline; intended for rebuttal,
camera-ready, or the ICAPS fallback, never for silent insertion into the
submitted claims.

## Question

The frozen blind field U-Net's table cost is constant in roadmap size (the
model consumes the 64x64 world raster, not the graph; t_max+1 forwards
~1.0 s CPU at 192 nodes), while classical search cost grows with the
space-time graph. Two preregistered questions:

1. **Persistence:** does the zero-shot success/expansion advantage over the
   anchor persist on roadmaps 2.7x-10.7x denser than the 192-node training
   substrate (a structural-generalization test: the checkpoint never saw
   these graph sizes)?
2. **Wall-time crossover:** is there a size at which the learned pipeline's
   end-to-end per-map wall time drops below success-tuned weighted A*'s
   (and the anchor's)?

## Frozen protocol

- Sizes N in {192, 512, 1024, 2048} roadmap nodes, k=7 unchanged; six dynamic
  suites in the canonical order (maze, rooms, spiral, maze_dense, crossing,
  rooms_large); t_max=110, dt, v_agent, generators, and usability rule
  (builds, connects, space-time solvable at t=0) unchanged.
- World streams: `iter_dynamic_worlds` with cfg.seed = 5_000_000 + N
  (disjoint from every prior cohort's seed base by construction); per
  (N, suite): first 10 usable worlds = DEV, next 30 usable = EVAL.
  A serialized-world SHA-256 manifest (same fields as the c8r manifest)
  is written at generation time, before any evaluation.
- Provider: the frozen checkpoint `c8_field__unet_blind.pt`
  (SHA-256 b8378950...) in additive mode. No retraining, no reselection.
- Budgets: anchor-only calibration per (N, suite). Grid = paper grid
  {150,250,400,600,900,1300,1800,2500,3500} scaled by N/192 (rounded to
  nearest 10). Rule: the grid budget whose DEV anchor success is closest to
  the paper's fresh-cohort anchor success at that suite's binding budget
  (operating-point matching across sizes); ties -> smaller.
- Weighted A*: per (N, suite) w over {1.1,1.2,1.5,2,3,5} on DEV only,
  frozen rule (highest success, ties smaller), evaluated once on EVAL.
- SIPP: unbudgeted, hard correctness gate (arrival == backward space-time
  Dijkstra optimum on every instance; abort on mismatch).
- Timing: per EVAL map, fixed arm order (anchor, WA*, learned table,
  learned search, SIPP) in one process; first map per container is warmup
  and excluded; perf_counter; container CPU model recorded; cpu=8
  containers; comparisons are within-map pairs. A separate GPU phase
  measures the U-Net table-build component alone on one L4 (deployment
  note; never mixed with the CPU rows).

## Preregistered readouts

- R1 (persistence): per N, learned-anchor success delta with paired
  map-level 95% CI excluding zero in >= 5/6 suites (exact McNemar + BH
  within each N as the confirmatory family).
- R2 (effort): matched learned/anchor expansion-ratio median < 1 in every
  suite at every N (map-bootstrap CIs; descriptive companion to R1).
- R3 (crossover, primary novel readout): per suite, the smallest N (if any)
  with median within-map ratio (learned end-to-end time)/(WA* end-to-end
  time) < 1, and the same vs the anchor. No crossover by N=2048 is an
  honest bounded negative.
- R4 (reference): SIPP success (expected: feasibility ceiling) and wall
  time per N; reported, never merged with budgeted-arm units.
- R5 (GPU note): L4 table-build time per map (component only).

## Integrity

Phases (gen+labels+manifest, calibrate, tune, eval+timing, gpu-table) are
idempotent and sharded per (N, suite). Raw rows, manifests, calibration and
tuning reports ship in the artifact package. Any deviation from this file
is recorded as a dated amendment, never edited in place.

## Amendment 1 (2026-07-26, pre-v2-execution; v12-review Part IV adopted)

The v1 launch was stopped mid calibrate/tune (no eval rows were produced or
examined) to adopt reviewer refinements; outputs move to runs/c8s2_scale.
Changes: (1) ONE shared fresh world cohort per suite evaluated at every
node count -- a world is accepted iff build_prm builds and connects at all
of {192, 512, 1024, 2048} (k=7; per-size roadmap seed = world seed); this
removes the world-difficulty confound across sizes. Nested point prefixes
were considered and declined: reusing the validated build_prm verbatim at
each size avoids custom-sampler semantic risk; the paired unit is the
world. (2) Eval shards run on one L4 container each; arms per world are
euclid, WA*, learned_cpu, learned_gpu (identical code path, device=cuda,
synchronized), and SIPP; three repeats per world with arm order randomized
per (world, repeat) by a seeded RNG; the first eval world per shard is
flagged warmup. (3) A density probe phase records predicted-vs-true
residual correlation, MAE, and bias over reachable states on five eval
worlds per (size, suite). (4) Frozen crossover definition: the smallest
tested N at which the paired learned_gpu-minus-WA* total-time 95% CI lies
below zero AND learned success is noninferior within 0.05 AND mean path
suboptimality is within +0.02 of WA*'s; the CPU-implementation crossover
is reported alongside under the same rule. A log-log size x method
interaction is the secondary scaling readout. (5) Batched network
forwards were NOT introduced: the sequential-forward implementation is
the submitted system; GPU timing uses the same code path with the device
argument. Roadmap construction time is common to all arms and excluded;
it is reported separately from shard logs.

## Completion (2026-07-27)

Executed in full (24 eval + 24 probe shards). Result:
`docs/experiments/continuous/c08/results/C8S_SCALE_WALLTIME_RESULT.md`;
analysis `docs/experiments/analysis/c8s2_analysis.py`. R1 PASS at 192/512,
frozen-bar FAIL at 1024/2048 via floor/ceiling cells only (dense maze
degenerate at all sizes - closest-target rule picks binding 150 on the
harder shared-world cohort; spiral@2048 anchor-unreachable; rooms-large at
ceiling); every non-degenerate cell q<0.01. R2 PASS everywhere
(ratios 0.06-0.29). R3: spiral crossover GPU@512 / CPU@1024; no other
suite by 2048 (bounded negative), but learned wall-time slopes 0.62-0.91
vs WA* 1.27-1.66 on all non-degenerate suites; maze learned/WA* time
ratio collapses 3.3x -> 1.2x. R4 SIPP = feasibility ceiling, slower than
learned_gpu everywhere at 2048. Probe: correlation size-stable, bias
drifts negative. CPU/GPU: 0 found mismatches (float tie-break exp diffs
<= 10, tallied).

## Amendment 2 (2026-07-27, post-hoc sensitivity recalibration; frozen before execution)

The completed v2 run produced degenerate operating points in seven cells:
dense maze at all four sizes and spiral at 2048 select floor budgets where
no arm solves anything (the anchor-operating-point targets 0.06/0.16 are
unreachable on the harder shared-world cohort under the coarse grid), and
rooms-large at 1024/2048 saturates (anchor 0.97-1.00). This amendment
specifies a SENSITIVITY study at discriminative operating points. It is
post hoc (motivated by observing the degeneracies), success-only (no
timing; R3 is untouched), and descriptive: the frozen R1 verdicts stand
and are reported first.

1. Floor cells {dense@192/512/1024/2048, spiral@2048}: each arm runs ONCE
   per world at the state-space cap n*(t_max+1) with recorded solve
   expansions; success at any budget derives exactly by thresholding
   (prefix determinism). Budget ladder = the scaled grid plus {2x, 4x the
   scaled maximum} (all below the cap). Sensitivity binding = the smallest
   ladder budget whose 10-world dev anchor success lies in [0.30, 0.70];
   if none, the smallest with success >= 0.30; if none, the cell is
   declared anchor-infeasible and learned-vs-anchor is reported at the cap.
2. WA* is re-tuned on the dev worlds at the sensitivity binding over the
   canonical weight grid {1.1, 1.2, 1.5, 2, 3, 5} (highest success, ties
   smaller), from the same capped dev runs by thresholding.
3. Evaluation arms on the 30 eval worlds: anchor, tuned WA*, learned_cpu
   (the submitted implementation; no GPU arm - success differs only by
   float tie-breaks). Readouts per cell at the sensitivity binding:
   learned-anchor and learned-WA* success deltas with paired map CIs and
   exact McNemar p (descriptive; no BH family is claimed).
4. Ceiling cells {rooms-large@1024, 2048}: no new runs. Sensitivity
   binding = the largest original grid budget whose recorded dev anchor
   success is <= 0.70 (from the calibration reports); existing eval rows
   are thresholded down to it (valid: they ran at a larger budget).
5. Outputs: sens_{stage}_{arm}_{size}_{suite}.csv rows and
   senssel_{size}_{suite}.json selection reports on the volume; analysis
   in the frozen analysis script. Optimal arrivals are not recomputed
   (success-only; optimal_arrival = -9 sentinel).
