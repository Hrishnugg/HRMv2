# C8 SIPP baseline: safe-interval planning on the confirmation cohort

**Frozen design date:** 2026-07-25 (before any implementation)
**Purpose:** answer the strongest remaining classical-baseline objection to the
dynamic zero-shot result: how does the fixed learned heuristic compare against
Safe Interval Path Planning (Phillips & Likhachev 2011), the standard strong
dynamic planner, under the same trajectories and instances?

## 1. Method

Discrete-time SIPP on the identical substrate:

- **Safe intervals.** For each roadmap node v, compute the maximal runs of
  time steps t in [0, t_max] at which v is collision-free against the exact
  deterministic patroller trajectories (the same validity predicate the
  space-time planner uses). A search state is (v, interval), annotated with
  its earliest achievable arrival time.
- **Successors.** From (v, I) with earliest arrival t: for each edge (v, u)
  with duration tau(e) steps, the agent may wait within I to any departure
  d >= t (d in I), traverse iff every intermediate sample of the edge at the
  appropriate times is collision-free (same edge-validation predicate as
  space-time A*), and arrives at u at d + tau(e) inside some safe interval of
  u. Standard SIPP: generate at most one successor per (edge, target
  interval) with the earliest feasible arrival.
- **Heuristic and cost.** Cost = arrival time in steps; heuristic = the same
  anchor h0(v) = ||x_v - x_g|| / (v_agent * dt) (admissible for earliest
  arrival). No learned guidance. Goal reached when the goal node is expanded;
  horizon t_max bounds arrivals as in the space-time planner.

## 2. Correctness gate (hard, pre-committed)

SIPP with an admissible heuristic is optimal for earliest arrival, as is the
backward space-time Dijkstra already computed for every world. **Gate:** on
every solved instance, SIPP's arrival must equal the space-time optimal
arrival exactly. Any mismatch voids the run and blocks reporting until the
defect is found. (Smoke: 5 worlds per suite before the full sweep.)

## 3. Evaluation

- Cohort: the frozen 50-map-per-suite confirmation cohort (seed 999999), all
  six suites; instances identical to the fixed-provider evaluation.
- Reported per suite: success within horizon; SIPP expansion counts
  (interval-state expansions -- **a different unit from (v,t) expansions**,
  reported with that caveat, never merged into the space-time expansion
  columns); wall time decomposed into interval construction + search; and,
  for context only, success when SIPP's expansion count is capped at the
  suite's binding budget (unit caveat repeated wherever this appears).
- Comparators quoted alongside (from existing frozen results, no re-runs):
  anchor A*, tuned weighted A*, fixed blind U-Net.
- Wall-time context for the learned arm: measured on a 5-map-per-suite sample
  as feature/raster construction + model inference + search (T_total
  decomposition), CPU condition stated.

## 4. Verdict rules (report as-is)

- R1: SIPP success within horizon (unbudgeted) -- expected at/near ceiling;
  any unsolved-at-horizon instance is reported.
- R2: SIPP interval-expansions vs space-time anchor expansions on jointly
  solved maps (descriptive; different units disclosed).
- R3: honest framing commitment: if SIPP solves everything cheaply in wall
  time, the paper reports that the learned result is search-effort structure
  within the time-expanded formulation, not superiority over the best
  classical dynamic planner.

## 5. Exclusions

No learned-heuristic-inside-SIPP arm; no stochastic dynamics; no tuning of
SIPP (it has no free parameters here beyond the shared substrate); no changes
to any frozen artifact. Runs locally on CPU (light; no GPU).
