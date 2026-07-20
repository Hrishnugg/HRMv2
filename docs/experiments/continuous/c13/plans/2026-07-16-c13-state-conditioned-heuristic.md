# C13 implementation plan

- [x] Trace C6/C7 target construction, runtime inputs, and integration semantics.
- [x] Separate strict geometry state from bounded local-observation state.
- [x] Preregister the `constant-E` semantics control, one-step local relaxation, and density sweep.
- [x] Implement leak-resistant local state features and one-step targets.
- [x] Add semantic, admissibility, locality, and provenance tests.
- [x] Implement dataset collection and split-by-world manifests.
- [x] Run the one-suite bounded-backup target-selection probe.
- [x] Run the preliminary one-suite `N={128,160,192,211,256}` density curve.
- [x] Confirm the working interpretation that bounded observations and non-shortest-path rollout outcomes are allowed.
- [x] Implement fresh-start rollout-return collection, HRM/ON-LSTM training, and Euclidean-anchored FOCAL evaluation.
- [x] Add matched Euclidean/one-step FOCAL controls and complete the one-suite smoke.
- [x] Run the one-suite multi-angle identifiability study across target reliability, representation/readout, learning curves, FOCAL width, exact-target ceilings, and primary-A* diagnostics.
- [x] Test an independent learned-incumbent plus fresh Euclidean-certifier integration; verify the proof and reject it when the oracle ceiling loses all six primary comparisons.
- [x] Implement a shared-state anchored multi-queue integration and pass the unchanged oracle ceiling at `w=1.10` on all six audit worlds.
- [x] Run the exact frozen rollout statistic under the shared integration; it fails the locked `w=1.10` gate (2 wins, 1 tie, 3 losses; mean delta `+1.33`), so learned models remain blocked.
- [x] Split rollout scale from ordering with a training-free same-search Euclidean control and preregistered monotone calibration/blend ablations (C13-F); calibration alone was insufficient.
- [x] Test exact radius-bounded local-escape and exit-stub ceilings (C13-G); both failed the unchanged bounded-FOCAL gate.
- [x] Train the local-heuristic Bellman learner (C13-H), select a fixed bounded operating point, and pass untouched 192- and 211-node one-suite confirmation.
- [x] Run the live six-suite C7 comparison (C13-I); the one-suite model failed outside maze and exposed the distribution problem.
- [x] Train the suite-balanced current-state model (C13-J); static insertion still failed, isolating integration rather than distribution alone.
- [x] Add one radius-bounded local Bellman backup (C13-K) and show the same model moves from clearly worse to a near tie on development worlds.
- [x] Calibrate local-backup scale on a fresh six-suite block (C13-L); reject every arm under the absolute 1.10 maximum gate while identifying a preregistrable matched-quality Pareto point.
- [x] Probe reopening versus direct A*; reopening does not remove the tail, while bounded FOCAL is safe but slower.
- [x] Preregister and pass C13-M on 144 untouched worlds: 68.31 versus 81.26 field-HRM expansions, paired delta -12.96 with 95% CI [-16.30, -9.74], all six suite means negative, and better empirical mean/max path quality.
- [x] Hash and verify the implementation, preregistration, checkpoints, 144 feature caches, 1,296 raw rows, summaries, report, manifest, and suite shards.
- [ ] Review the final state/local-subgraph wording and direct-versus-bounded Pareto framing with the professor before publication-facing claims.

## Final status

C13 has met its benchmark-level scientific objective. The fixed C13-M arm is
conditioned on bounded observations of each search state rather than a complete
map raster or shortest-path label, and it confirms a 15.95% pooled expansion
reduction versus the complete-map C7 field HRM on a disjoint six-suite cohort.
The direct arm is not formally bounded and its current Python feature builder
is slower in wall time; both limitations are explicit in the canonical
[C13-F through C13-M result](../results/C13F_M_CURRENT_STATE_RESULT.md).
