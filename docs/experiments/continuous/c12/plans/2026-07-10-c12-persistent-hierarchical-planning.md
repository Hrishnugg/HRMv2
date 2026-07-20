# C12 Persistent Hierarchical Planning — Implementation Plan

**Date:** 2026-07-10
**Status:** G0-A passed; learned C12-A implementation authorized
**Execution priority:** proceed with C12-A dataset/model Tasks 3 and 5–12; keep C12-B independent and off the C12-A critical path.
**Spec:** `docs/experiments/continuous/c12/design/2026-07-10-c12-persistent-hierarchical-planning-design.md`
**G0-A result:** [canonical result](../results/C12_RESULTS.md), full hash `a11a9997e855854f5bd8`, 800 episodes / 3,200 matched provider rows, all four gates passed.

**Goal:** run two separately gated tests of hierarchy under persistent state/computation:

- C12-A: partially observed slow/fast dynamics, multi-horizon forecasting, and closed-loop receding-horizon PRM planning.
- C12-B: tied iterative value refinement on C11 product graphs.

**Implementation rule:** preserve C8 and C11 as immutable prior stages. Import their public helpers, write C12 artifacts under new run directories, and make every run checkpoint-resumable. Do not reinterpret or overwrite a prior canonical result.

---

## 0. Preconditions and read-first sources

Before implementation:

- [x] User approved the design spec on 2026-07-10; the approval record is in the design document.
- [x] `c11_big` completed all 12 training cells and evaluation; C12 remains independent of its verdict.
- [x] Recorded `git status --short --branch`; the worktree contains substantial user-owned documentation and experiment artifacts. Stage only C12 files unless the user expands scope.
- [x] Environment recorded: Python 3.13.7, PyTorch 2.9.0+cu130, CUDA available on RTX 5090, approximately 28 GB free VRAM and 119 GB free disk at implementation start.
- [x] C8/C11 baseline recorded. The literal command below exposed a pre-existing root-level `PYTHONPATH` collection issue in three unrelated `hrm-cloud/tests` modules; with `hrm-cloud` and `hrm-cloud/continuous_prm` on `PYTHONPATH`, the selected baseline was **157 passed, 108 deselected** in 488.10 seconds.

Read completely before coding:

- `hrm-cloud/continuous_prm/continuous_prm_dynamics.py`
- `hrm-cloud/continuous_prm/continuous_prm_spacetime.py`
- `hrm-cloud/continuous_prm/continuous_prm_c8_dynamic_maps.py`
- `hrm-cloud/continuous_prm/continuous_prm_dynamic_providers.py`
- `hrm-cloud/continuous_prm/continuous_prm_c8_dynamics_compare.py`
- `hrm-cloud/continuous_prm/continuous_prm_c11_headroom.py`
- `hrm-cloud/continuous_prm/continuous_prm_c11_mission.py`
- `hrm-cloud/continuous_prm/continuous_prm_c11_hrmv2_arm.py`
- `hrm-cloud/onlstm_hrm_comparison_presetm_v2.py` (`ONLSTMCell`, HRM blocks, rollout loss, environment, evaluator)
- C8 and C11 design/result documents.

Baseline command:

```powershell
python -m pytest hrm-cloud/continuous_prm/tests hrm-cloud/tests -q -k "c8 or c11"
```

---

## 1. Module and artifact map

### New source modules

| File | Responsibility |
|---|---|
| `hrm-cloud/continuous_prm/continuous_prm_c12_latent_dynamics.py` | Hidden slow/fast simulator, gate schedules, observation schema, episode/suite generation, seed splits. |
| `hrm-cloud/continuous_prm/continuous_prm_c12_closed_loop.py` | `TabulatedDynamics`, path-reconstructing predicted-space-time A*, receding-horizon execution, true-simulator scoring. |
| `hrm-cloud/continuous_prm/continuous_prm_c12_world_model.py` | Shared frame encoder/decoder, snapshot/LSTM/Transformer/ON-LSTM/HRM cores, carry API, losses, checkpoint providers. |
| `hrm-cloud/continuous_prm/continuous_prm_c12_persistent.py` | C12-A CLI and orchestration: `probe|collect|train|forecast-eval|plan-eval|analyze|full`. |
| `hrm-cloud/continuous_prm/continuous_prm_c12_refiner.py` | C12-B K=16 probe, tied/untied graph models, train/eval/analyze CLI. |

### New tests

| File | Coverage |
|---|---|
| `hrm-cloud/continuous_prm/tests/test_c12_latent_dynamics.py` | Regimes, aliasing, observation non-leakage, deterministic split generation. |
| `hrm-cloud/continuous_prm/tests/test_c12_closed_loop.py` | Tabulated collision checks, path reconstruction, first-action execution, collision scoring. |
| `hrm-cloud/continuous_prm/tests/test_c12_world_model.py` | Shared I/O, carry semantics, parameter/compute accounting, model forward/gradient/overfit. |
| `hrm-cloud/continuous_prm/tests/test_c12_persistent.py` | Dataset, manifests, metrics, clustered statistics, gates, CLI smoke. |
| `hrm-cloud/continuous_prm/tests/test_c12_refiner.py` | K=16 probe, tied weights, per-cycle outputs/loss, matched controls, B gates. |

### Canonical result

- `docs/experiments/continuous/c12/results/C12_RESULTS.md`

Do not modify prior C8/C11 source unless a missing reusable primitive is impossible to wrap. Any such change requires a separate regression-tested task and explicit justification before it is made.

---

# C12-A implementation

## Task 1 — Hidden-regime simulator and observation contract

**Files:** create `continuous_prm_c12_latent_dynamics.py`, `test_c12_latent_dynamics.py`.

- [x] Write failing tests first:
  - identical seeds produce bitwise-identical episodes;
  - TRAIN/VALIDATION/TEST seeds are disjoint;
  - fast and slow periods fall in their configured ranges;
  - direction-alias pairs share the same current observation but have divergent 8-step futures;
  - slow-gate and route-junction aliases are constructible;
  - `present_sufficient` exposes phase/direction while challenge suites do not;
  - serialized observations contain no latent regime, phase counter, velocity, future waypoint, or future occupancy field;
  - static map/goal seeds are independent of regime seeds, and counterfactual regime variants are balanced per static world;
  - gate state blocks/unblocks exactly the registered roadmap-edge ids;
  - episode reset clears all latent and visible state.
- [x] Implement dataclasses:
  - `C12DynamicsConfig`;
  - `LatentRegime`/`RegimeSchedule`;
  - `GateSchedule`;
  - `LatentDynamicsState`;
  - `C12Observation`;
  - `C12EpisodeSpec`.
- [x] Implement challenge constructors for `direction_alias`, `slow_gate_phase`, `route_mode_junction`, and `present_sufficient` over C8 map families.
- [x] Keep true simulator methods deterministic: `state_at(t)`, `observe(t)`, `future(t, horizon)`, `node_free`, `edge_free`, `gate_edge_valid`.
- [x] Add a schema audit function that fails if forbidden latent fields enter a model batch.
- [x] Run:

```powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c12_latent_dynamics.py -q
```

**Suggested commit:** `feat(c12): latent slow-fast dynamics and leakage-safe observations`

---

## Task 2 — Predicted dynamics adapter and path-reconstructing planner

**Files:** create `continuous_prm_c12_closed_loop.py`, `test_c12_closed_loop.py`.

- [x] Write failing tests:
  - `TabulatedDynamics` built from exact C8 circle centers matches `Dynamics.node_free/edge_free` on analytic cases;
  - predicted gate states mask the intended edge only at the intended step;
  - path-reconstructing A* returns the same arrival/expansion result as C8 A* on a known exact-future fixture;
  - wait action appears as the first action when a corridor will clear;
  - returned first edge is valid under predicted dynamics;
  - true-simulator scoring detects a collision missed by an incorrect forecast;
  - observations update carry during multi-step edge traversal, while decisions occur only at nodes;
  - episode termination distinguishes goal, collision, horizon, and no-plan.
- [x] Implement `TabulatedDynamics(centers, radii, gate_open, dt)` with C8-compatible feasibility methods.
- [x] Implement `predicted_space_time_astar(...)` with parent pointers and a first-action/path result; use C8’s `_edge_steps` convention and collision sampling, but do not edit C8.
- [x] Implement `run_closed_loop_episode(...)` with one-action receding-horizon execution.
- [x] Record cumulative expansions, planning milliseconds, arrival, collisions, replans, and failure reason.
- [x] Verify fine-grained true-simulator collision sampling is at least 2× denser than planning-time sampling.

**Suggested commit:** `feat(c12): tabulated forecasts and closed-loop space-time planner`

---

## Task 3 — Episode collection, serialization, and split manifests

**Files:** modify `continuous_prm_c12_latent_dynamics.py`; create/modify `continuous_prm_c12_persistent.py`, `test_c12_persistent.py`.

- [x] Define deterministic seed formulas with separate high-order split offsets.
- [x] Build an episode record containing visible frames, exact future targets, static map id, regime stratum label, and privileged fields stored in a separate diagnostic block never passed to models.
- [x] Use compact `.npz` shards plus `dataset_manifest.json`; never pickle executable objects.
- [x] Test shard round-trip, checksums, schema version, split disjointness, shape/mask consistency, and append/resume behavior.
- [x] Add `collect --scale smoke|pilot|full` CLI mode.
- [x] Add `inspect-dataset` mode that prints counts, periods, alias rate, missing masks, and disk estimate without training.
- [x] Refuse to merge shards with different config hashes or schema versions.

**Suggested commit:** `feat(c12): deterministic episode dataset and resumable collection`

---

## Task 4 — G0-A aliasing/history/headroom probe

**Files:** modify `continuous_prm_c12_persistent.py`, `test_c12_persistent.py`.

- [x] Implement present-observation discretization used only for the preregistered alias-pair diagnostic.
- [x] Implement `frozen_frame`, `constant_velocity`, `true_mode`, and `oracle_future` forecast providers.
- [x] Run each provider through the same closed-loop planner on 200 dedicated PROBE episodes per stratum; assert they are disjoint from every final split.
- [x] Write `c12a_headroom_raw.csv`, `c12a_headroom_summary.json`, and a Markdown probe report.
- [x] Implement exact G0-A booleans from the spec; unit-test passing, failing, and boundary synthetic cases.
- [x] `train` and `full` modes refuse to start unless a matching config-hash probe report says G0-A passed. Override requires an explicit `--allow-failed-probe` flag and marks every downstream artifact exploratory.
- [x] Run the real probe before implementing model training. The full probe passed all four gates under hash `a11a9997e855854f5bd8`; proceed to learned-model implementation.

Command:

```powershell
python hrm-cloud/continuous_prm/continuous_prm_c12_persistent.py --mode probe --scale full --out-dir hrm-cloud/continuous_prm/runs/c12_persistent
```

**Suggested commit:** `feat(c12): preregistered memory-headroom authorization probe`

---

## Task 5 — Shared frame encoder, decoder, and flat controls

**Files:** create `continuous_prm_c12_world_model.py`, `test_c12_world_model.py`.

- [x] Write failing tests for:
  - frame encoder and object masks;
  - horizon decoder shapes `(B, 32, max_patrollers, 2)` and gate logits;
  - snapshot arm ignores earlier frames;
  - LSTM carry persists and resets exactly;
  - sliding-window Transformer is causal and sees exactly 16 frames;
  - all outputs finite and masked slots zeroed;
  - loss hand calculation for center Huber + 0.5 gate BCE;
  - gradients flow into frame encoder, temporal core, and decoder;
  - parameter/FLOP accounting is deterministic.
- [x] Implement shared `FrameEncoder` and `DirectHorizonDecoder`.
- [x] Implement the `TemporalCore` protocol and carry-tree helpers.
- [x] Implement `snapshot`, `lstm`, and `temporal_transformer` arms.
- [x] Add automatic width search over a small preregistered candidate list to place flat controls within the parameter/compute tolerance; save the chosen widths before pilot TEST.
- [x] Do not import the old monolithic comparison scripts at runtime; copy only required math with provenance comments and dedicated tests.

**Suggested commit:** `feat(c12): shared forecast model contract and flat temporal controls`

---

## Task 6 — ON-LSTM and streaming HRM temporal cores

**Files:** modify `continuous_prm_c12_world_model.py`, `test_c12_world_model.py`.

- [x] Write failing tests:
  - ON-LSTM master gates are cumulative-softmax ordered and numerically valid;
  - HRM fast state updates every step and slow state updates only at configured cadence;
  - slow state persists across replan boundaries;
  - carry detach preserves values and breaks graph history;
  - episode-boundary reset removes prior-regime information;
  - both slow and fast state receive gradients on a multi-timescale toy sequence;
  - repeated eval with same carry/input is deterministic;
  - reset and window-reencode evaluation paths use the same weights.
- [x] Implement `onlstm` from the validated cumax cell math.
- [x] Implement `hrm_stream` as an explicit stateful H/L core; document how it differs from C11 HRM-v2 ACT.
- [x] Parameter/compute match against flat controls and write counts into a frozen `model_grid.json`.
- [x] Tiny alias-task sanity: hierarchical/flat recurrent arms must overfit; snapshot must retain irreducible error.

**Suggested commit:** `feat(c12): persistent ON-LSTM and two-timescale HRM cores`

---

## Task 7 — Training loop, checkpointing, and collapse diagnostics

**Files:** modify `continuous_prm_c12_world_model.py`, `continuous_prm_c12_persistent.py`, tests.

- [x] Stream contiguous episode chunks with truncated BPTT; never shuffle individual timesteps across episodes.
- [x] Reset carry only on an episode-boundary mask; detach at chunk boundaries.
- [x] Implement the common loss/optimizer/clip recipe and validation-only checkpoint selection.
- [x] Save checkpoints atomically and load-merge-write `manifest.json` entries.
- [x] Record config hash, git commit/dirty flag, split manifest hash, arm, seed, param/FLOP counts, epochs, best validation score, wall time, peak VRAM, gradient summaries, and collapse diagnostics.
- [x] Resume only when checkpoint metadata matches all scientific config fields.
- [x] Tests:
  - deterministic CPU mini-training;
  - tiny-set overfit for all arms;
  - carry reset boundary correctness;
  - interrupted run resumes without duplicate entries;
  - TEST loader is never called by train/checkpoint selection;
  - constant-output predictions trigger a failed-validation diagnostic.

**Suggested commit:** `feat(c12): stateful forecast trainer and atomic resume manifests`

---

## Task 8 — Forecast evaluation and mechanistic metrics

**Files:** modify `continuous_prm_c12_persistent.py`, tests.

- [x] Evaluate identical TEST episodes for every arm/seed.
- [x] Produce raw rows per episode, decision step, horizon bucket, arm, and seed.
- [x] Implement ADE/FDE, gate balanced accuracy/Brier, occupancy recall, and route-critical recall.
- [x] Add privileged latent-regime linear probe trained on TRAIN contexts and evaluated on TEST; label it diagnostic.
- [x] Add `*_reset` and `*_window_reencode` modes without retraining.
- [x] Test all metrics against hand-built arrays and ensure padded identities do not contribute.
- [x] Write `c12a_forecast_raw.csv` append-safely by shard and merge deterministically.

**Suggested commit:** `feat(c12): long-horizon forecast and carry-ablation evaluation`

---

## Task 9 — Learned forecast to closed-loop planner integration

**Files:** modify `continuous_prm_c12_persistent.py`, `continuous_prm_c12_closed_loop.py`, tests.

- [x] Convert every learned prediction into `TabulatedDynamics` using the same radii/gate-edge mapping.
- [x] Run matched closed-loop episodes with shared roadmaps, observations, and true dynamics.
- [x] Batch/cache model inference only where it preserves carry semantics; never call a different observation sequence per arm.
- [x] Test a perfect predictor matches `oracle_future` planner outcomes on fixtures.
- [x] Test a deliberately wrong direction predictor causes the expected first-action/collision change.
- [x] Record failure reason and keep unsuccessful episodes in the raw table.
- [x] Write `c12a_planning_raw.csv` and `c12a_carry_ablation.csv`.

**Suggested commit:** `feat(c12): matched learned-forecast closed-loop evaluation`

---

## Task 10 — World-clustered analysis and C12-A gate writer

**Files:** modify `continuous_prm_c12_persistent.py`, tests.

- [x] Aggregate model seeds within world before primary comparisons.
- [x] Implement seeded world bootstrap and world-level paired sign-flip tests.
- [x] Implement BH correction within each preregistered gate family.
- [x] Preselect the best flat comparator using VALIDATION artifacts and persist the choice before TEST analysis.
- [x] Implement G1-A, G2-A, G3-A, and G4-A as pure functions with boundary-case tests.
- [x] Ensure exploratory slices cannot mutate gate booleans.
- [x] Write `c12a_summary.json`, `c12a_significance.csv`, and a generated C12-A Markdown section.

**Suggested commit:** `feat(c12): clustered inference and preregistered C12-A verdicts`

---

## Task 11 — C12-A end-to-end smoke and pilot freeze

**Files:** no new scientific code expected; bug fixes get separate tests/commits.

- [ ] Run the complete smoke pipeline:

```powershell
python hrm-cloud/continuous_prm/continuous_prm_c12_persistent.py --mode full --scale smoke --out-dir hrm-cloud/continuous_prm/runs/c12_persistent_smoke
```

- [ ] Verify all artifacts exist, contain config hashes, and regenerate identically from raw rows.
- [ ] Run pilot (256 train / 64 validation / 64 development-eval per stratum, one seed) after G0-A passes; pilot evaluation seeds are not final TEST seeds.
- [ ] Freeze in the design amendment: final environment knobs, exact matched widths, confirmation of the spec-pinned optimizer/checkpoint rule, and projected runtime.
- [ ] If projected full runtime exceeds the spec cap, stop for local/remote scope choice.
- [ ] Do not inspect full TEST outcomes while tuning from pilot validation.

**Suggested commit:** `test(c12): persistent-dynamics smoke and frozen pilot configuration`

---

## Task 12 — C12-A full run

- [ ] Confirm no unexpected GPU processes or insufficient disk/VRAM.
- [ ] Collect full TRAIN/VALIDATION/TEST shards once.
- [ ] Train 5 learned arms × 3 seeds from the frozen manifest.
- [ ] Run forecast evaluation, closed-loop evaluation, carry ablations, and analysis.
- [ ] Preserve stdout/stderr logs per arm/seed and a top-level run ledger.
- [ ] If one arm fails, resume it from its last atomic checkpoint; do not restart completed arms.
- [ ] Inspect only pipeline integrity until all primary arms finish.

---

# C12-B implementation

## Task 13 — K=16/deep-propagation probe

**Files:** create `continuous_prm_c12_refiner.py`, `test_c12_refiner.py`.

- [x] Reuse C11 bundle/oracle/graph constructors without mutation.
- [x] Instantiate a C12-owned `C11MissionConfig(k_max=16)` and prove C11’s default K≤8 grid/config remains unchanged.
- [x] Add C12-only deterministic K=16 train/test seed formulas.
- [x] Test mission validity, door validity, graph tensor shapes, finite oracle coverage, and split disjointness.
- [x] Implement graph-distance-to-relevant-transition diagnostic.
- [x] Evaluate h_legsum/oracle across a calibration budget grid and compute G0-B.
- [x] Measure peak RAM and wall time for label generation before accepting a cell.
- [x] Write `c12b_probe_raw.csv` and summary; training refuses a failed config hash.

Command:

```powershell
python hrm-cloud/continuous_prm/continuous_prm_c12_refiner.py --mode probe --out-dir hrm-cloud/continuous_prm/runs/c12_refiner
```

**Suggested commit:** `feat(c12): deep product-graph refinement authorization probe`

---

## Task 14 — Tied refiner and matched graph controls

**Files:** modify `continuous_prm_c12_refiner.py`, `test_c12_refiner.py`.

- [x] Tests:
  - shared block object/parameter ids are identical at every tied cycle;
  - untied control parameters differ by cycle;
  - outputs exist at cycles 1/2/4/8 with correct graph shape;
  - exact deep-supervision weights are 0.1/0.2/0.3/0.4;
  - cycle-1 output equals the single-cycle path;
  - edge removal blocks propagation on a toy graph;
  - parameter and edge-operation reports match hand counts;
  - tiny graph overfit reduces loss at every supervised cycle;
  - gradients reach the shared block from the cycle-8 loss.
- [x] Implement `TiedGraphRefiner`, `ShallowParamMatch`, and `UntiedComputeMatch` using the C11 product graph tensors.
- [x] Keep output normalization/cap and target recipe identical to C11.
- [x] Implement common trainer, atomic checkpoints, manifest metadata, and per-cycle validation.
- [x] Diagnostic cycle 16 is disabled in headline analysis by default.

**Suggested commit:** `feat(c12): tied graph refiner with deep supervision and matched controls`

---

## Task 15 — C12-B train/eval/analyze pipeline

**Files:** modify `continuous_prm_c12_refiner.py`, tests.

- [x] CLI modes implemented as integrated `probe|smoke|pilot|full|analyze` transactions; separate train/eval modes were intentionally not exposed (documented deviation).
- [x] Train authorized cells on identical bundles and seeds.
- [x] Write state metrics per world/seed/cycle: MAE, rank correlation, Bellman residual.
- [x] Run matched product A* for cycles 1/2/4/8 and all controls.
- [x] Aggregate at world level and implement G1-B/G2-B/G3-B pure verdict functions.
- [x] Tests cover monotone-positive, compute-only-positive, flat-negative, success-regression, and failed-G0 cases.
- [x] Smoke:

```powershell
python hrm-cloud/continuous_prm/continuous_prm_c12_refiner.py --mode full --scale smoke --out-dir hrm-cloud/continuous_prm/runs/c12_refiner_smoke
```

- [x] Time one full cell/arm before launching the grid; apply the same compute stop rule as C12-A.

**Suggested commit:** `feat(c12): iterative-refinement evaluation and preregistered verdicts`

---

# Integration and publication artifacts

## Task 16 — Combined results writer and reproducibility audit

**Files:** create `docs/experiments/continuous/c12/results/C12_RESULTS.md`; modify orchestrators only if writer bugs surface.

- [ ] Generate the report from raw artifacts and gate booleans; no hand-entered headline numbers.
- [ ] Keep C12-A and C12-B verdicts in separate sections.
- [ ] Include:
  - verbatim hypotheses/gates;
  - G0 authorization evidence;
  - parameter/FLOP/latency matching table;
  - forecast horizon curves;
  - closed-loop completion/collision/expansion/regret table;
  - carry reset/re-encode mechanism table;
  - refiner quality-vs-cycle curves;
  - all negative and failed-control outcomes;
  - threats, deviations, hardware, runtime, git/config hashes.
- [ ] Regenerate summaries from raw CSV and compare hashes/row counts.
- [ ] Verify no TRAIN/TEST overlap and no forbidden latent observation columns.
- [ ] Run all C12 tests plus C8/C11 regression suites.
- [ ] Add C12 links/status to the experiment index and master synthesis only after canonical results exist.

Verification command:

```powershell
python -m pytest hrm-cloud/continuous_prm/tests hrm-cloud/tests -q -k "c8 or c11 or c12"
```

**Suggested commits:**

- `docs(c12): persistent-dynamics and iterative-refinement results`
- `docs(experiments): integrate C12 into the evidence synthesis`

---

## 2. Full-run expected grid

### C12-A

- pooled training across 3 challenge strata plus present-sufficient control;
- 5 learned arms × 3 seeds = 15 primary checkpoints;
- exact same TEST episodes for all checkpoints;
- 4 core strata plus ID/OOD/scale views;
- reset and window-reencode are checkpoint-sharing eval modes, not additional training runs.

### C12-B

- 2 configs × up to 3 K values × 3 seeds × 4 learned graph arms = up to 72 checkpoints;
- K=16 cells included only after G0-B;
- cycle 1/2/4/8 share a tied-refiner checkpoint.

The timing estimate may reduce no preregistered grid silently. A scope reduction requires a written amendment before TEST evaluation.

---

## 3. Definition of done

C12 is complete only when:

- [x] G0-A and G0-B are reported, including any rejected cells/track;
- [x] all authorized primary arms and 3 seeds have valid checkpoints or explicit failure records;
- [x] raw forecast/planning/refinement rows are complete and matched;
- [x] world-clustered statistics and multiplicity corrections are reproducible;
- [x] every gate resolves mechanically to positive/negative/not-authorized;
- [x] C8/C11 regression suites remain green;
- [x] `C12_RESULTS.md` is generated and linked from the experiment index;
- [x] deviations and optimization failures are included rather than replaced post hoc.

---

## 4. Explicitly deferred follow-ups

Do not fold these into the implementation opportunistically:

- stochastic latent-mode transitions and calibrated uncertainty;
- risk-aware or belief-space planning;
- learned macro-actions/two-level A*;
- multi-goal amortized planning;
- transfer/LoRA on C12;
- high-DOF task-and-motion planning;
- multi-agent prediction/coordination;
- reinforcement learning or direct expansion-policy optimization.

They become C13 candidates only after C12 identifies whether the useful axis is temporal memory, iterative propagation, planner integration, or none of them.
