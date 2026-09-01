# C12 — Persistent Hierarchical Planning: Hidden Regimes and Iterative Refinement

**Date:** 2026-07-10
**Status:** approved; C12-A G0 authorization passed and learned-model work is authorized
**Approval record:** User approved the design on 2026-07-10. Execution is authorized with C12-A through the real G0-A probe as the critical path; C12-B remains an independently gated addendum and is not allowed to delay G0-A.
**G0-A record:** The full 800-episode probe passed all four gates under config hash `a11a9997e855854f5bd8`; see [C12 results](../results/C12_RESULTS.md). This authorizes C12-A model training but is not itself a learned-hierarchy result.
**Scope:** one C12 umbrella with two independently gated tracks:

- **C12-A (primary):** persistent, partially observed, multi-timescale dynamics with learned forecasting and closed-loop replanning.
- **C12-B (mechanism addendum):** weight-tied iterative refinement over the C11 product graph.

The tracks share one thesis—hierarchy should matter when useful state or computation must persist across more than one timescale—but they are analyzed separately. A positive result in one track must not be used to rescue a negative result in the other.

Related evidence:

- `docs/experiments/continuous/c08/results/C8_RESULTS.md`: known deterministic future motion; present-frame heuristics were already close to sufficient.
- `docs/experiments/continuous/c11/results/C11_RESULTS.md`: compositional headroom was real, but one-shot scalar regression produced no hierarchy dose-response and exposed recurrent training fragility.
- `docs/experiments/cross-space/PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md`: architecture separation appeared on forecasting tasks and disappeared on scalar heuristic regression.
- `docs/experiments/discrete/dynamic-world-model/design/ONLSTM_VS_HRM_EXPERIMENT_PRESETM_V2.md`: reusable multi-step forecasting and closed-loop evaluation ideas, but not a publication-grade matched result.

---

## 1. Decision summary

C12 will not ask whether a model with a hierarchical name can regress the same fully observed scalar target better than an MLP. It asks two narrower, falsifiable questions:

1. **Temporal persistence:** when the current observation is deliberately insufficient, can a hierarchical recurrent model retain slow environmental context while tracking fast motion better than parameter- and compute-matched flat temporal models, and does that improve closed-loop planning?
2. **Computational persistence:** when a full product graph is available, can a shared iterative update improve a value field over repeated cycles, and does quality grow with compute on deep missions?

The intended claim is conditional, not universal:

> Hierarchy helps learned planning only when the problem exposes a matching hierarchy—slow latent context over fast dynamics, or repeated value propagation over a structured graph.

C12 is successful scientifically even if both tracks are negative, provided the information/headroom gates pass and the matched controls are valid.

---

## 2. What has and has not been tested

Already tested:

- static and deterministic space-time PRM heuristic regression;
- present-frame versus explicitly supplied future-window inputs;
- additive and focal planner integration;
- U-Net, GNN, MLP, HRM/ON-LSTM trace models, and faithful HRM-v2 ACT;
- transfer, LoRA/full fine-tuning, and adapter interpolation;
- compositional missions through K=8 with exact product-graph labels;
- forced HRM-v2 segment counts k in `{1,2,4,8}`.

Not yet tested:

- persistent hidden state across repeated replans;
- a world where the same present observation corresponds to different futures and different correct planning actions;
- slow latent regimes composed with fast obstacle motion;
- matched flat temporal baselines on that partially observed problem;
- a full-graph, weight-tied refiner with deep supervision at intermediate cycles;
- planner-level benefit from a learned forecast, measured over an entire closed-loop episode rather than one isolated heuristic table.

---

## 3. Research hypotheses and preregistered interpretation

### H-A1 — history is necessary

On the primary C12-A suites, a present-only planner must be measurably worse than a history-informed planner or exact-future oracle. If this is false, C12-A stops before model training; a memory architecture cannot be adjudicated on a Markov-sufficient observation.

### H-A2 — temporal hierarchy improves long-horizon prediction

At matched information, training data, parameter scale, and approximately matched inference compute, at least one hierarchical temporal core (`onlstm`, `hrm_stream`) will beat the strongest flat temporal comparator (`lstm`, sliding-window Transformer) on route-critical long-horizon prediction. The advantage should be larger at long horizons or long regime dwell times than at short horizons.

### H-A3 — prediction improvement reaches the planner

A better forecast must improve closed-loop planning—not merely trajectory MSE. Primary evidence is episode success/collision-free completion and cumulative planning work; forecast metrics are mechanistic secondary evidence.

### H-A4 — persistent carry is causal

The winning recurrent model must lose some of its advantage when its carry is reset at every replan. A sliding-window re-encode control receives the same recent observations and distinguishes information access from cheap persistent reuse.

### H-B1 — iterative graph computation refines value

On deep C11 missions, a tied graph refiner will improve state-level error and search guidance from cycle 1 to cycle 8. The gain should be larger at K=8/16 than at K=2.

### H-B2 — recurrence is more than extra FLOPs

The tied refiner must be compared against both a parameter-matched shallow graph model and an edge-operation/FLOP-matched untied graph model. A win only against the shallow model is an “extra compute helps” result, not a hierarchy result.

---

# Part I — C12-A: persistent partially observed dynamics

## 4. Environment and observability contract

### 4.1 Base geometry and planner reuse

C12-A reuses the C8 continuous PRM suites, roadmap construction, collision checks, and time-discretized planning conventions. It does **not** modify C8 outputs or canonical modules. New C12 modules import them.

The true simulator contains:

- a static PRM and static obstacles;
- fast moving circular patrollers;
- optional slow gates that enable/disable selected roadmap edges;
- a latent regime and phase schedule that determines future patroller routes and gate transitions.

The agent observes the environment at every simulator step and replans only at roadmap nodes. During a multi-step edge traversal, observations still update the model carry, but a new action is selected only on arrival.

### 4.2 Latent regime construction

Primary dynamics are deterministic conditional on a hidden episode seed and latent regime. Stochastic transitions are deferred; the first experiment must not confuse irreducible uncertainty with model failure.

Each world composes a **slow** variable with **fast** motion:

- fast patroller periods: nominally 6–12 simulator steps;
- slow gate/regime periods or dwell times: nominally 32–64 steps;
- episode length: 128 steps;
- history burn-in: 16 steps;
- prediction horizon: 32 steps.

Static map, goal, and latent regime are independently seeded. Each accepted static world is instantiated with balanced counterfactual regime variants, including paired variants that reach an identical visible decision state before diverging. This prevents a snapshot model from inferring the hidden regime from map identity or goal placement.

Three challenge mechanisms are required:

1. **Direction aliasing:** a patroller can occupy the same position while moving in opposite directions. Current position is identical; recent history identifies velocity and future occupancy.
2. **Slow gate phase:** the visible gate state can be identical at two phases, while time-to-transition differs. Earlier gate transitions or a synchronized patroller cue identify phase.
3. **Route-mode junction:** a patroller reaches the same junction under two route modes, but a prior cue in the episode identifies which branch it will take.

The hidden regime, direction, phase counter, future waypoints, and future occupancy are never included in learned-model observations.

Alias fixtures are paired by construction. At each designated alias decision, the two serialized current-observation tensors must match exactly after float32 conversion (with an `atol=1e-6` audit), while their true 8-step occupancy and oracle first action differ. The probe does not discover aliases by choosing a favorable tolerance after generation.

### 4.3 Present-sufficient control

A fourth control suite exposes direction/phase in the current observation. All temporal models should tie there after matching. A hierarchy-only win on this control is treated as an optimization or compute confound, not support for H-A2.

### 4.4 Observation

All learned arms receive the same raw information:

- static occupancy raster;
- current dynamic occupancy raster;
- visible gate-open raster;
- normalized current patroller centers/radii with an identity mask;
- current agent and goal raster channels;
- no velocity, latent mode, phase counter, or future frames.

A shared `FrameEncoder` maps the raster/object observation to `d_model=256`. Temporal cores differ; the spatial encoder and multi-horizon decoder are shared and trained jointly per arm. This isolates temporal inductive bias and avoids the C11 “class-native input contains different information” ambiguity.

### 4.5 Suites and splits

Primary challenge strata:

- `direction_alias`;
- `slow_gate_phase`;
- `route_mode_junction`.

Control:

- `present_sufficient`.

Map families reuse C8 maze/rooms/spiral geometry. Full evaluation contains:

- matched in-distribution worlds;
- held-out initial phase and direction combinations;
- longer-dwell OOD worlds (slow period 1.5–2× training range);
- held-out `rooms_large` scale worlds.

Seeds are disjoint by split and recorded explicitly:

- probe: 200 episodes per stratum in a dedicated `PROBE` seed namespace;
- smoke: 8 train / 4 validation / 4 test episodes per stratum;
- pilot: 256 train / 64 validation / 64 development-eval episodes per stratum;
- full: 3,000 train / 300 validation / 300 test episodes per stratum;
- 3 model-training seeds; the TEST episodes are identical across model seeds.

`PROBE`, smoke, pilot, and final splits use separate seed namespaces. Probe and pilot outcomes are development evidence and never enter the final C12 effect tables. No final TEST outcome may be inspected while choosing architecture widths, checkpoint epoch, or environment difficulty. All such choices use PROBE/TRAIN/VALIDATION/pilot-development artifacts only.

---

## 5. Forecast target, model contract, and training

### 5.1 Target

At every eligible simulator step, predict the next 32 steps of:

- normalized patroller center displacement for each identity;
- gate-open probability for each gate;
- an optional rendered occupancy field used for diagnostics, derived deterministically from predicted centers/gates rather than learned by a separate head.

The conditional future is deterministic in the primary experiment. Mixture-density or diffusion outputs are out of scope unless the deterministic experiment passes and a stochastic extension is separately specified.

### 5.2 Shared decoder and losses

Every learned arm uses the same direct multi-horizon decoder from temporal context to all 32 future steps. Direct decoding is chosen over an arm-specific autoregressive decoder so temporal-core comparisons are not confounded by different rollout machinery.

Primary training loss:

`L = mean_horizon(Huber(center_hat, center_true)) + 0.5 * BCE(gate_logits, gate_true)`

- horizons are uniformly weighted so long-term error cannot be hidden by short-term accuracy;
- identity masks exclude absent patrollers/gates;
- positions are normalized by map side length;
- AdamW `lr=2e-4`, `weight_decay=1e-4`, gradient clipping `1.0`;
- batch = 16 episode streams, truncated-BPTT chunk length = 32, maximum = 20 epochs, validation patience = 4 epochs;
- early stopping/checkpoint selection uses validation route-critical forecast loss, never TEST planning outcomes.

These optimization defaults are shared by all learned arms. If the pilot shows numerical failure rather than ordinary underfitting, changing them requires a design amendment and a complete pilot rerun for every arm; there is no arm-specific learning-rate rescue.

### 5.3 Learned arms

Primary learned arms:

1. `snapshot` — current frame only; no temporal state.
2. `lstm` — flat recurrent control with persistent carry.
3. `temporal_transformer` — re-encodes the most recent 16 frame embeddings at every decision.
4. `onlstm` — persistent ordered-neuron state.
5. `hrm_stream` — persistent two-timescale H/L state with a fixed slow-update cadence.

Algorithmic/reference arms:

- `frozen_frame` — treats current obstacles/gates as constant;
- `constant_velocity` — estimates velocity from the last two observations;
- `true_mode` — privileged hidden-regime predictor, diagnostic only;
- `oracle_future` — exact future simulator rollout, ceiling only.

The temporal learned arms target 3–5M trainable parameters. Parameter counts must be within ±10% where possible. Per-step multiply-add estimates and measured latency are reported. If one flat model cannot match both HRM parameters and compute within tolerance, include separate `lstm_param_match` and `lstm_compute_match` controls.

Fixed structural defaults are `d_model=256`, Transformer depth 4 with 8 heads, ON-LSTM chunk size 8, and HRM slow-state update every 4 visible frames. Hidden widths are chosen only from `{256, 320, 384, 448, 512}` to satisfy the matching rule, then frozen in `model_grid.json` before full TEST evaluation.

### 5.4 Stateful API

Every recurrent temporal core implements:

```text
initial_carry(batch_size, device)
step(frame_embedding, carry) -> context, next_carry
detach_carry(carry)
```

Training streams contiguous episode chunks with truncated BPTT. Carry resets only at a true episode boundary. Evaluation additionally runs:

- `*_reset`: same checkpoint, carry reset at every replan;
- `*_window_reencode`: recent 16 observations replayed from a fresh carry at every replan.

These are evaluation modes, not separately trained models.

### 5.5 Optimization safeguards

Before any full run:

- each arm must overfit a tiny deterministic episode set;
- hidden-state gradients must be nonzero across both fast and slow state;
- constant-output collapse checks run on validation predictions;
- state carry must be deterministic in eval mode;
- per-arm loss curves and gradient norms are written to the manifest;
- no arm-specific training rescue is allowed after TEST results. A failed arm may receive a separately labeled optimization diagnostic, not replacement headline numbers.

---

## 6. Predicted dynamics and closed-loop planning

### 6.1 Tabulated forecast adapter

Each predictor produces a `TabulatedDynamics` object containing predicted centers, radii, and gate states for the next 32 steps. It implements the C8-style `node_free` and `edge_free` contract plus a gate-edge validity query.

The true simulator remains separate. Predicted dynamics are used only for planning; collision and completion are scored against true dynamics.

### 6.2 Receding-horizon execution

At a roadmap node:

1. encode the newest observation and update carry;
2. predict 32 future steps;
3. run space-time A* over the predicted dynamics;
4. execute only the first wait or edge action;
5. advance the true simulator, observe each traversed time step, and repeat.

The C12 planner is new and path-reconstructing. It reuses C8 geometry/feasibility functions but does not change `continuous_prm_spacetime.py`, whose public result currently omits a path.

### 6.3 Episode outcomes

Primary planning outcomes:

- collision-free goal completion within 128 steps;
- collision count and first-collision time;
- cumulative A* expansions over all replans;
- total wall-clock planning time;
- arrival time and regret versus `oracle_future` on worlds both solve;
- number of replans and failed-plan events.

Forecast outcomes are secondary/mechanistic:

- average/final displacement error by horizon bucket `{1–4, 5–16, 17–32}`;
- gate-state balanced accuracy and Brier score;
- route-critical occupancy recall on nodes/edges considered by the oracle path;
- latent-regime linear-probe accuracy from frozen temporal context (diagnostic only).

---

## 7. C12-A gates

### G0-A — memory/headroom authorization (must pass before training)

Run 200 dedicated PROBE episodes per stratum with no learned models. These seeds are disjoint from final TEST. C12-A is authorized only if all are true:

1. **Aliasing exists:** at least 15% of valid planning decision points belong to a constructed alias pair whose current serialized observations match under the fixed `atol=1e-6` audit, but whose 8-step future occupancy and oracle first action differ.
2. **History matters:** `constant_velocity` or `true_mode` improves collision-free completion over `frozen_frame` by at least 0.15 absolute **or** reduces collision-adjusted arrival regret by at least 25% with a world-bootstrap 95% CI excluding zero.
3. **Ceiling exists:** `oracle_future` completes at least 85% of accepted episodes and leaves at least a 20% planning-work/regret gap over the best non-oracle baseline.
4. **Control behaves:** the `present_sufficient` suite has materially lower history headroom than each challenge stratum.

If any condition fails, modify only environment generation/calibration and rerun G0-A. Do not train models on a failed substrate.

### G1-A — long-horizon forecast hierarchy

Select the best flat comparator on VALIDATION before reading TEST. A hierarchical arm is forecast-positive only if:

- it beats both the selected flat recurrent comparator and the sliding-window Transformer on route-critical long-horizon error in at least 2 of 3 challenge strata;
- the world-clustered 95% CI excludes zero after BH correction across the three strata;
- its relative advantage in horizon 17–32 is no smaller than in horizon 1–4;
- it is not significantly worse on the `present_sufficient` control.

This gate is mechanistic; failure does not suppress the planning evaluation.

The registered route-critical error follows the mechanism in each stratum: center ADE for `direction_alias` and `route_mode_junction`, and gate-state Brier error for `slow_gate_phase`. Comparator preselection averages these three otherwise incommensurate errors only after dividing each arm's error by the snapshot arm's error for the same seed and stratum. G1 significance remains a world-clustered comparison of the unnormalized stratum-specific errors. Occupancy recall is retained as a secondary diagnostic, not as the selection statistic.

### G2-A — closed-loop planning hierarchy (primary verdict)

A hierarchical arm is planning-positive only if, versus the preselected best flat comparator:

- it improves collision-free completion in at least 2 of 3 challenge strata with world-clustered paired significance after BH, **or**
- at statistically tied completion/collision rate, it reduces cumulative expansions or arrival regret by at least 15% with a 95% CI excluding zero in at least 2 of 3 strata;
- it has no significant collision-rate regression on any primary stratum;
- the advantage is larger on long-dwell OOD than on the present-sufficient control.

The winning arm and comparison are chosen mechanically from validation rankings; no TEST-cell cherry-picking.

### G3-A — carry mechanism

For any G2-A-positive recurrent arm:

- persistent carry must beat its `*_reset` evaluation in at least 2 challenge strata;
- persistent carry must match or beat `*_window_reencode` planning quality while using less measured encoding compute per replan;
- if reset/re-encode do not change the result, the claimed mechanism is “temporal architecture,” not “persistent state.”

### G4-A — honest closure

- **G0 pass + G2 positive:** hierarchy helps under explicitly history-dependent dynamics; report which temporal core and which regime.
- **G0 pass + G1 positive + G2 negative:** hierarchy predicts better but the planner is insensitive; the result is a forecast/planner-integration mismatch.
- **G0 pass + both negative:** matched temporal hierarchy adds no value even when history is necessary; a strong negative.
- **G0 fail:** no architecture verdict; substrate rejected.

---

# Part II — C12-B: iterative product-graph refinement

## 8. Scope and data

C12-B reuses C11 world generation, product-graph encoding, exact oracle labels, leg-sum baseline, matched A*, and TEST worlds. It does not change or overwrite C11 checkpoints/results.

Primary cells:

- configs A (ordered waypoints) and C (keys/doors);
- K in `{2,8,16}`;
- 40 disjoint TRAIN worlds and the existing 25 TEST worlds where available;
- 3 model seeds.

K=16 requires a new probe and disjoint deterministic seed stream. If mission sampling or product-graph memory is impractical, K=16 is dropped **before training** and C12-B remains `{2,8}` with graph-diameter strata.

C12-B constructs a C12-owned `C11MissionConfig(k_max=16)` for K=16 encodings; it does not change C11’s default `k_max=8`, cell grid, stage embeddings, checkpoints, or canonical analysis.

### G0-B — refinement authorization

For every primary K=16 cell retained:

- at least 20 valid probe worlds;
- oracle/leg-sum matched expansion ratio ≤ 0.30 at a calibrated binding budget;
- median shortest-path/message-passing distance from query states to a relevant mission-transition edge > 8 hops;
- exact product-graph labels fit within the measured memory/time envelope.

Failure removes that cell; it does not authorize easier post-hoc replacements.

---

## 9. Models and training

### 9.1 Tied graph refiner

`TiedGraphRefiner` receives the full C11 product graph, node/edge features, and normalized leg-sum initialization. One shared message/update block is applied recurrently. It emits a residual estimate after cycles `{1,2,4,8}` and can be evaluated diagnostically at cycle 16.

Update contract:

```text
h_0 = encode(node_features, h_legsum)
h_{r+1} = h_r + SharedGraphBlock(h_r, edges, edge_features)
yhat_r = softplus_clamp(Readout(h_r))
```

Training loss uses deep supervision:

`L = 0.1 L_1 + 0.2 L_2 + 0.3 L_4 + 0.4 L_8`

where every `L_r` is the same smooth-L1 residual target used by C11. Bellman residual is reported as a diagnostic but is not added to the headline training loss; adding it later would be a separately labeled objective ablation.

### 9.2 Controls

1. `c11_gnn8` — existing C11 8-round untied graph architecture, retrained on the C12-B cells.
2. `shallow_param_match` — one application of a widened graph block matched to the refiner’s parameter count.
3. `untied_compute_match` — the same number of edge-message operations as the 8-cycle refiner, but untied blocks.
4. `tied_refiner` — primary arm.
5. `h_legsum` and `h_oracle` — algorithmic floor/ceiling.

All learned arms use the C11 matched recipe unless graph batching forces a documented mechanical change: AdamW `lr=2e-4`, `weight_decay=1e-4`, gradient clip `1.0`, smooth-L1 target, 40 epochs, and validation checkpoint selection on identical world sets. Graphs are accumulated to the largest common effective node count that fits every arm; the effective batch and accumulation schedule are identical. Parameter counts, edge operations, peak VRAM, and wall time are reported.

### 9.3 Evaluation

For cycles `{1,2,4,8}`:

- state MAE and rank correlation to exact residual;
- Bellman residual;
- matched A* success and expansion ratio at binding budget;
- prediction/search wall time;
- improvement versus cycle 1 as a function of K and graph-distance stratum.

Cycle 16 is extrapolative and diagnostic only unless separately trained/preregistered before TEST.

---

## 10. C12-B gates

### G1-B — depth-of-compute

Positive only if `tied_refiner`:

- improves monotonically (allowing bootstrap-tied adjacent steps) from cycle 1 to 8 on MAE or expansion ratio;
- has cycle-1 versus cycle-8 world-clustered 95% CIs separated on at least one K=8/16 cell;
- shows a larger cycle benefit at K=8/16 than at K=2.

### G2-B — matched architecture advantage

Hierarchy-positive only if cycle-8 `tied_refiner` beats both `shallow_param_match` and `untied_compute_match` on planning quality in at least one deep cell after multiplicity correction, with no success regression. Beating only the shallow model is recorded as a compute-depth positive, not a tied-hierarchy positive.

### G3-B — honest closure

- **G1 and G2 positive:** shared iterative refinement is the missing useful mechanism.
- **G1 positive, G2 negative:** extra propagation helps, but weight tying/hierarchy adds nothing beyond compute.
- **G1 negative:** even explicit full-graph iterative refinement does not progressively solve the C11 target.
- **G0-B fail:** no deep-refinement verdict for the removed cells.

---

## 11. Statistical design

The TEST world—not `world × model_seed`—is the independent unit.

- Aggregate each model’s 3 seeds within world before primary paired comparisons.
- Use world-clustered bootstrap intervals (resample worlds, retaining all model seeds).
- Use paired world-level sign-flip/permutation tests for continuous outcomes.
- Use a paired world-clustered bootstrap for completion-rate differences instead of treating repeated model seeds as independent McNemar observations.
- BH-correct the preregistered primary suite/cell comparisons within each gate.
- Report raw effect sizes and CIs even when a gate is negative.
- Exploratory slices are labeled and cannot alter a preregistered verdict.

---

## 12. Run stages and compute stops

### Stage 0 — pure probes

- G0-A aliasing/history/headroom probe.
- G0-B K=16/headroom/message-distance probe.

No learned-model implementation proceeds for a failed track.

### Stage 1 — unit and smoke

- CPU unit tests.
- one tiny GPU overfit per arm;
- 8/4/4 smoke episodes and one C12-B cell;
- end-to-end artifact validation.

### Stage 2 — pilot

- 256/64/64 episodes per C12-A stratum, one model seed;
- C12-B A/K=8, one seed;
- freeze environment knobs, learning recipe, widths, and checkpoint rule from validation.

### Stage 3 — full local estimate

Time one training/evaluation unit for every arm. Extrapolate complete GPU and CPU hours. If projected local GPU time exceeds 36 hours or evaluation exceeds 24 CPU-hours, stop and present local-versus-remote execution options; do not silently shrink the grid.

### Stage 4 — full run

- C12-A full splits, 3 model seeds, all ID/OOD/control suites;
- C12-B all authorized cells, 3 seeds;
- checkpoint-resumable manifests and append-safe result shards.

### Stage 5 — analysis and writeup

Write `docs/experiments/continuous/c12/results/C12_RESULTS.md` from computed booleans/effects. Preserve raw records under `runs/c12_persistent/` and `runs/c12_refiner/`.

---

## 13. Artifacts

C12-A:

- `runs/c12_persistent/probe/c12a_headroom_raw.csv`
- `runs/c12_persistent/datasets/dataset_manifest.json`
- `runs/c12_persistent/checkpoints/*.pt`
- `runs/c12_persistent/manifest.json`
- `runs/c12_persistent/results/c12a_forecast_raw.csv`
- `runs/c12_persistent/results/c12a_planning_raw.csv`
- `runs/c12_persistent/results/c12a_carry_ablation.csv`
- `runs/c12_persistent/results/c12a_summary.json`
- `runs/c12_persistent/results/c12a_significance.csv`

C12-B:

- `runs/c12_refiner/probe/c12b_probe_raw.csv`
- `runs/c12_refiner/checkpoints/*.pt`
- `runs/c12_refiner/manifest.json`
- `runs/c12_refiner/results/c12b_state_metrics.csv`
- `runs/c12_refiner/results/c12b_eval_raw.csv`
- `runs/c12_refiner/results/c12b_summary.json`
- `runs/c12_refiner/results/c12b_significance.csv`

Canonical:

- `docs/experiments/continuous/c12/results/C12_RESULTS.md`

---

## 14. Threats to validity and safeguards

- **Task engineered for memory:** that is intentional and conditional. The present-sufficient control and strong flat LSTM/Transformer baselines prevent a universal hierarchy claim.
- **Hidden-state leakage:** latent regime/phase/future fields are excluded by schema tests and serialized observation audits.
- **More compute masquerading as hierarchy:** parameter, FLOP/edge-operation, latency, and reset/re-encode controls are mandatory.
- **Forecast metric disconnected from planning:** G2-A planning is primary; G1-A is mechanistic.
- **Repeated TEST worlds across model seeds:** world-clustered inference is mandatory.
- **Optimization collapse:** overfit tests, validation collapse checks, gradient diagnostics, and no post-TEST rescue runs.
- **Difficulty tuning after outcomes:** all environment calibration uses G0/pilot data only; TEST seeds stay sealed.
- **C8 regression:** C12 imports C8 modules and adds new planner/simulator code; canonical C8 files/results are not mutated.
- **C11 regression:** C12-B writes a separate run directory and canonical report; C11 manifests/checkpoints/results remain immutable.
- **Discrete collision approximation:** keep C8 sub-step collision checks and report `dt`; verify executed actions against finer true-simulator sampling.
- **Scope creep into stochastic POMDPs:** stochastic transitions, uncertainty-aware planning, multi-agent coordination, and multi-goal amortization are follow-ups, not hidden additions to C12.

---

## 15. Out of scope

- claiming hierarchy must win;
- stochastic regime transitions or irreducible future uncertainty;
- reinforcement learning or end-to-end policy optimization;
- multi-agent coordination;
- high-DOF manipulation/TAMP;
- transfer/LoRA on C12 before an architecture/mechanism signal exists;
- planner-level learned macro-actions (a possible C13 if C12-B shows a propagation signal);
- changing C11’s canonical verdict with C12 results.

---

## 16. Approval decisions

Approval of this draft locks the following defaults:

1. C12-A is the primary paper-facing experiment; C12-B is an independent mechanism addendum.
2. Deterministic-but-partially-observed slow/fast dynamics come before stochastic dynamics.
3. Closed-loop episode planning is the primary outcome; forecast loss is mechanistic.
4. The flat LSTM and sliding-window Transformer are mandatory controls.
5. World-clustered inference replaces `world × seed` pseudo-replication.
6. Full runs stop for a compute decision rather than silently reducing scale.

Any change to these six decisions after implementation starts must be recorded as a design amendment before TEST evaluation.

---

## 17. Pre-pilot implementation amendment (2026-07-10)

This amendment was frozen before accepting any pilot checkpoint and before any final TEST evaluation. Superseded pilot shards/checkpoints are retained only as implementation diagnostics.

The first end-to-end development smoke revealed a support error in the environment/metric contract: `direction_alias` and `route_mode_junction` had route-critical events only in horizons 1–16, so the preregistered G1 comparison at horizons 17–32 was undefined in two of three challenge strata. No TEST data existed, and no gate outcome was used to choose a favored architecture. The correction is mechanical:

- direction hazards recur at their seeded 6–12-step fast period, while slow-gate and route-mode events use sparse slow-schedule pulses; all preserve the original 8-step G0 divergence and provide route-critical support in horizons 17–32;
- `route_critical_mask` is derived from the exact target hazard-edge occupancy/closure table rather than from only the first hazard interval;
- a hard unit test requires nonzero horizon-17–32 support at the alias point in every challenge stratum;
- the dynamics/dataset/probe/forecast schemas were advanced, and all earlier smoke artifacts remain diagnostic history rather than canonical evidence.

A second pre-pilot audit found that the provisional `slow_gate_phase` variants closed different route edges at the same future offset. History was relevant, but the mechanism did not yet satisfy the stronger registered claim that identical current gate states have different *time-to-transition*. Before accepting a pilot result, the phase schedule was corrected so paired gates remain visibly open at the alias point but close after different seeded offsets; earlier closed intervals identify the phase. The `heldout_phase_direction` slice swaps this timing/direction correlation. A unit test now requires identical present observations, different transition times, divergent eight-step futures, and divergent oracle actions. The partially started v3 pilot was interrupted before any checkpoint or learned result was accepted, its files were retained as diagnostics, and the schemas advanced again.

A final memory-depth audit then checked the comparison against the *registered* 16-frame Transformer rather than merely against snapshot. In the provisional schedule, the Transformer still saw the slow-gate/route-mode cues, and long gate closures made many route-critical rows trivial after the transition was already visible. The final schedule therefore pins these invariants:

- slow-gate and route-mode variants differ at `t=0`, but every visible frame `t=1..16` is exactly paired-identical at the alias decision; persistent cores receive the cue during burn-in, whereas the exact 16-frame Transformer has dropped it;
- slow gates use short closure pulses rather than 32–64-step continuously closed intervals, so route-critical evaluation measures transition anticipation;
- route-mode pulses recur on the 32–64-step slow schedule, while `direction_alias` remains the intended short-memory challenge/control;
- validation checkpoint loss excludes pre-burn-in frames, preventing the visible `t=0` cue from dominating model selection;
- hard tests pin full-history divergence, exact last-16-frame equality, long-horizon support for both counterfactual variants, and post-pulse reopening.

The partially started v4 pilot was likewise interrupted before accepting its snapshot checkpoint. No architecture outcome from either provisional pilot was inspected or promoted.

The evaluation split is also frozen into deterministic, recorded slices before pilot:

- `matched_id`;
- `long_dwell_ood`, with slow dwell sampled from 1.5–2× the training maximum;
- `heldout_phase_direction`, which swaps the training phase/direction correlation;
- `scale_ood` for `C_dyn_rooms_large`.

TRAIN and VALIDATION remain in-distribution. The final pre-pilot v6/v11 full G0 rerun passed on 800 episodes: 71.3% constructed aliasing, +0.467 history completion gain, 65.1% regret reduction (95% CI 63.0–66.9%), 97.5% oracle completion, a 75.1% ceiling gap, and zero present-sufficient history headroom.

The outcome-independent width search over `{256, 320, 384, 448, 512}` is frozen as:

| arm | width | trainable parameters | estimated multiply-adds/step |
|---|---:|---:|---:|
| snapshot | 512 | 3,007,296 | 5,625,152 |
| LSTM | 512 | 4,444,480 | 7,066,944 |
| temporal Transformer | 512 (`ff=1024`) | 3,798,592 | 6,411,584 |
| ON-LSTM | 448 | 3,713,952 | 6,340,928 |
| streaming HRM | 448 | 4,196,672 | 6,821,184 |

The optimizer/checkpoint rule remains exactly the approved one: AdamW `2e-4`, weight decay `1e-4`, clip `1.0`, 16 episode streams, TBPTT 32, maximum 20 epochs, patience 4, and post-burn-in VALIDATION route-critical loss only. After checkpoints are fixed, the headline flat and hierarchical arms are preselected from persistent VALIDATION at horizons 17–32 using center ADE for direction/route-mode worlds and gate Brier for slow-gate worlds. Each stratum is normalized to its same-seed snapshot error before averaging, so trajectory distance and gate probability are not mixed in raw units. This mechanism-aligned rule was frozen before accepting a pilot checkpoint; it does not use TEST or planning outcomes. CUDA training forces the deterministic math attention backend and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Runtime projection is filled from the corrected smoke/pilot timing before any full launch; the 36-GPU-hour/24-CPU-hour stop rule remains binding.
