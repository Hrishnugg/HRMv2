# C8-S v2 result: scale study — expansions vs wall time at 192–2048 nodes

Status: COMPLETE (2026-07-27). Frozen design
`docs/experiments/continuous/c08/design/2026-07-26-c8-scale-walltime.md`
(+ Amendment 1: shared worlds, L4 timing, probe, frozen crossover rule).
Executed on Modal (app continuous-prm-c8s, volume runs/c8s2_scale).
Analysis: `docs/experiments/analysis/c8s2_analysis.py` (+ `.json`,
`_output.md`). Post-submission evidence (rebuttal/camera-ready/ICAPS);
NOT part of the 2026-07-28 submission.

## Realized protocol

One shared fresh cohort per suite (30 eval + 10 dev worlds), each world
accepted only if the roadmap builds and connects at ALL of
{192, 512, 1024, 2048} nodes (k=7; per-size roadmap seed = world seed).
Scope: the cohort is therefore conditioned on connectivity at every
tested density (worlds connectable at the sparsest graph), and PRMs are
independently REBUILT per size, not nested node-prefix graphs.
Budget grid scaled by N/192; anchor-only calibration to the fresh-cohort
anchor operating points; WA* re-tuned per (size, suite) under the frozen
rule. Arms per world: euclid, tuned WA*, learned_cpu, learned_gpu
(identical code path, device only), SIPP; three repeats, arm order
randomized per (world, repeat); first eval world per shard excluded from
timing as warmup. The frozen checkpoint is the paper's blind U-Net
(SHA-asserted at load).

## R1 (persistence of the success advantage)

- N=192: PASS (5/6 suite CIs exclude zero; BH q<=0.002 in those five).
- N=512: PASS (5/6; q<=0.009).
- N=1024: FAIL under the frozen >=5/6 bar (4/6).
- N=2048: FAIL (3/6).

Every failure is a floor or ceiling cell, not a reversal. Dense maze is
degenerate at ALL sizes: on the harder shared-world cohort the anchor
solves 0/10 dev worlds at every grid budget up to 3500 (0.3 at 3500 only),
so the closest-to-0.06 rule picks the grid floor (binding 150) and no arm
solves anything. Spiral@2048 is anchor-unreachable at every scaled budget
(all dev solves >= 37,331), binding lands at 1600, all arms 0.00. Rooms-
large saturates (anchor 0.97/1.00 at 1024/2048 - no headroom). Every
non-degenerate cell across all four sizes is individually significant
(q < 0.01; deltas +0.27 to +1.00, e.g. maze +0.93 and crossing +0.80 at
N=2048). Reading: the success advantage persists to 2048 nodes wherever
the calibrated operating point leaves headroom; the frozen calibration
rule itself degenerates on two floor cells (a calibration FAILURE MODE at scale: anchor-operating-point targets of
0.06/0.16 sit below the 10-world resolution of 0.10, so the frozen
closest-target rule can select budgets where nothing solves - an extreme
form of the paper's "coarse grid" caveat).

## Amendment 2 sensitivity: the degenerate cells at discriminative budgets

Post hoc and descriptive by design (`c8s2_sens_analysis.py`; frozen rule:
smallest ladder budget with 10-world dev anchor success in [0.30, 0.70],
ladder = scaled grid plus {2x, 4x max}; success by thresholding capped
runs). Four of the seven degenerate cells un-degenerate IN THE LEARNED
METHOD'S FAVOR, none reverses:

- dense@192 (binding 3500): anchor 0.27 -> learned 0.83, d +0.57
  [+0.40, +0.73], 17/0 discordant, exact p = 1.5e-5; ties re-tuned WA*x5
  (0.83).
- dense@512 (9330): +0.70 [+0.53, +0.87], 21/0, p = 9.5e-7; ties WA* (0.93).
- dense@1024 (18,670): +0.50 [+0.33, +0.67], 15/0, p = 6.1e-5; ties WA*
  (0.93).
- rooms-large@1024 (ceiling cell, sens binding 3200 from recorded rates):
  anchor 0.43 -> learned 0.97, d +0.53 [+0.37, +0.70], 16/0, p = 3.1e-5;
  also above the original-binding-tuned WA* by +0.23 [+0.10, +0.40]
  (weight not re-tuned at this budget - descriptive).
- Three cells remain uninformative even at sensitivity budgets: dense@2048
  and spiral@2048 land above the band (anchor 0.93/1.00 at the first
  ladder point past the frozen grid - the anchor's budget-success curve at
  2048 is steep enough that the doubling ladder jumps the discriminative
  range), and rooms-large@2048's recorded grid offers no point below
  anchor 0.87. Deltas there are +0.07/+0.00/+0.10, none excluding zero.

Reading: wherever a discriminative operating point exists, the success
advantage holds at every size - including on the suites the frozen
calibration degenerated - and the learned arm ties re-tuned WA* on the
dense floors. The frozen R1 verdicts above are unchanged by this
sensitivity study.

## R2 (effort persistence): PASS at every size

Every defined matched expansion-ratio median is far below 1 at every N
(0.062-0.285; map-bootstrap CIs all below 1). The expansion advantage
remains large through 2048 nodes in every estimable suite-size cell
(per-cell matched n in the analysis output).

## R3 (wall-time crossover; the primary novel readout)

**The preregistered crossover criterion is met on spiral: GPU at N=512
(dt -1.19 s [-1.82, -0.57], success +0.17 in the learned arm's favor,
path within +0.02), and the negative time difference remains at N=1024 on
the same paired cohort (dt -5.64 s [-6.88, -4.41], +0.17); the CPU
implementation meets it at N=1024. Multiplicity companion (added at
review): BH across all 24 suite-by-size GPU time contrasts gives
q = 0.0002 for both spiral cells - the crossover is not a
multiple-comparisons artifact.** The spiral
2048 cell is degenerate (see R1) and reports no crossover. No other suite
reaches the frozen crossover by N=2048 - an honest bounded negative per
the design - but the scaling structure is uniform: ESTIMATED log-log wall-time slopes
are 0.62-0.91 (learned GPU) and 0.43-0.73 (learned CPU) versus 1.27-1.66
(tuned WA*) and 1.02-1.35 (anchor) on the four suites with non-degenerate
size sweeps, and the paired world-bootstrap slope CONTRASTS exclude zero
in every case (WA* minus learned GPU: maze +0.36 [+0.30, +0.43], rooms
+0.38 [+0.32, +0.45], crossing +1.02 [+0.92, +1.12], rooms-large +0.76
[+0.67, +0.86]; anchor-minus-GPU and WA*-minus-CPU likewise positive).
The learned-over-WA* total-time ratio on maze collapses from ~3.3x (192)
to ~1.2x (2048; dt +1.81 s [+0.96, +2.68]). Under the registered
secondary readout, the estimated learned wall-time slopes are lower than
every classical arm's on every non-degenerate suite; the frozen-rule
crossover materializes within the tested range only on spiral. Caveat:
four size points; shards may run on different cloud hosts (the paired
fixed-size contrasts are robust to host variation; the cross-size slopes
are estimates).

## R4 (SIPP reference; own units)

SIPP solves ~everything at every size (0.97-1.00 = feasibility ceiling, as
expected) with wall time growing steeply: 1.7-3.7 s (192) to 17.4-39.0 s
(2048). At N=2048 SIPP's mean wall time exceeds learned_gpu's in every
suite (e.g. dense 36.8 vs 11.6 s; rooms-large 39.0 vs 15.0 s) - different
success semantics (unbudgeted optimal vs budgeted), never merged, but a
striking reference point at scale.

## R5 (GPU table-build component)

Mean per-map table build on L4: 1.40 s (192), 2.32 s (512), 4.42 s (1024),
7.61 s (2048) - grows ~5.4x for 10.7x nodes (the t_max+1 forward count is
size-invariant; growth comes from per-node sampling).

## Probe (residual quality vs size; descriptive)

Median per-world predicted-vs-true correlation is size-STABLE (maze
0.64->0.68, spiral 0.74->0.79, dense 0.79->0.79, rooms 0.49->0.59,
crossing 0.20->0.30 from 192 to 2048) while MAE and under-prediction bias
grow modestly (bias -0.15..-1.07 at 192 to -0.68..-1.46 at 2048). The
ranking signal survives 10x node-count extrapolation; magnitude
calibration drifts. (Not comparable to the paper's MovingAI-probe values:
different cohort and protocol.)

## Integrity

- CPU/GPU arms agree on success in all 720 paired evaluations (zero
  found-flag mismatches); 100 evaluations differ by at most ten
  expansions because device floating-point tie-breaking reorders equal-f
  expansions. R1/R2 use learned_cpu (the submitted implementation); each
  R3 line uses its own arm's success/path outcomes.
- Timing excludes roadmap construction (common to all arms) and warmup
  worlds; repeats averaged within world; per-(world,repeat) randomized arm
  order.
- All 24 (size, suite) eval shards and 24 probe shards completed; success
  and expansions are repeat-identical (asserted).

## Verdict

The paper's core claims scale: the success advantage persists wherever the
operating point has headroom (R1 at 192/512; all non-degenerate cells at
1024/2048), and the expansion advantage is size-stable (R2). The wall-time
story sharpens from "expansions do not convert at 192 nodes" to "the
estimated learned wall-time slopes are lower than every classical arm's
(contrast CIs exclude zero on all non-degenerate suites), the
preregistered crossover criterion is met on spiral by N=512 (BH q=0.0002
across the 24-cell scan), and the maze time ratio closes to ~1.2x WA* by
N=2048." Two calibration-floor
degeneracies (dense at all sizes; spiral at 2048) are protocol artifacts
of the frozen closest-target rule on the harder shared-world cohort,
disclosed as such.

## Pointers

Raw rows: volume `runs/c8s2_scale` (local mirror via `rows_bundle.tar.gz`).
Core: `continuous_prm_c8_scale_walltime.py`; driver:
`continuous_prm_c8_scale_modal.py`; bundler: `c8s2_bundle_rows.py`.
