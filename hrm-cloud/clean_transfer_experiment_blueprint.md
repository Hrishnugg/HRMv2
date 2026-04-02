# Clean transfer experiment blueprint

## Goal

Build a new transfer-learning experiment for learned A* heuristics on grid maps that cleanly tests:

- **map-size transfer** first,
- **subdued obstacle scaling** second,
- **full fine-tuning vs stage-wise LoRA** under the *same* curriculum,
- and whether the learned model actually changes search behavior.

This experiment should replace the old residual-over-static-distance setup rather than patch it again.

---

## 1. What we will reuse from the repo

### Reuse directly
- **Map families / dynamics / occupancy simulation** from  
  `hrm-cloud/transfer_astar_heuristic_imitation_mapscale_lora.py`
- **ON-LSTM backbone** from  
  `hrm-cloud/onlstm_hrm_comparison_presetm_v2.py`
- **DeepSapientHRM backbone** from  
  `hrm-cloud/onlstm_hrm_comparison_presetm_v2.py`
- **LoRA utilities** from the patched transfer scripts
- **Modal sharded eval / preemption-safe caching** from the patched transfer scripts

### Do not reuse as-is
- the old `HRMEncoder` (GRU + GRUCell)
- the old flat obstacle-slot observation vector
- the old residual target `true_ctg - h_static_shortest_path`
- future-time dynamic occupancy patches indexed directly at node time

---

## 2. Architecture choice

## Backbone comparison

Use these two backbones only for the first clean run:

- **ON-LSTM**
- **DeepSapientHRM**

Do **not** use the tiny GRU-based `HRMEncoder`, and do **not** jump straight to `hrm_act_v1` for this first rebuild.

Reason:
- ON-LSTM and DeepSapientHRM already exist in repo code with clean step/state APIs.
- They are much closer to the intended architecture comparison than the old transfer script.
- `hrm_act_v1` would require a much heavier integration rewrite and would confound the first clean experiment.

## Important caveat about DeepSapientHRM

In `onlstm_hrm_comparison_presetm_v2.py`, the `GatedRecurrentBlock` uses `nn.MultiheadAttention` on a sequence length of 1 at each step. That means its benefit is primarily **hierarchical recurrent gating / H-L update structure**, not standard temporal self-attention over the whole history. This is fine, but it should shape how results are interpreted.

---

## 3. New model input representation

## 3.1 Step encoder input

At each environment step, build a spatial tensor with channels:

1. `walls`
2. `agent`
3. `goal`
4. `gate_closed`
5. `patrollers`
6. `drifters`
7. `x_coord`
8. `y_coord`

Optional ninth channel for later:
- `current_blocked = clip(walls + gate + pat + drift)`

The coordinate channels should be normalized to `[-1, 1]` or `[0, 1]` and broadcast spatially.

## 3.2 Frame encoder

Use a small CNN with adaptive pooling so 32x32 and 64x64 maps share the same embedding size.

Recommended starter:
- Conv(8 -> 32, 3x3)
- Conv(32 -> 64, 3x3, stride=2)
- Conv(64 -> 128, 3x3, stride=2)
- residual / norm / GELU as convenient
- adaptive average pooling
- linear projection to `d_model`

This gives a **global frame embedding** that is size-agnostic.

## 3.3 Sequence encoder

Process the sequence of frame embeddings with:
- ON-LSTM, or
- DeepSapientHRM

The encoder runs **once per environment step**, not once per A* node.

## 3.4 Candidate-node head

The node/query head should take:

- recurrent context vector,
- **static local patch** around the candidate node (walls only),
- **current-time** local dynamic patch around the candidate node (not future-time),
- candidate metadata:
  - `dx_to_goal`
  - `dy_to_goal`
  - `dt_offset`
  - `h_base`
  - node absolute coordinates

Recommended patch design:
- small patch: 15x15
- optional large/coarse patch later if needed

This keeps node scoring fast but avoids the old future-occupancy leakage.

---

## 4. New target and planner coupling

## 4.1 Base heuristic

Use:

- **Manhattan distance** for 4-connected movement + wait

not static shortest-path distance.

Reason:
- On static maps with corridors/walls, Manhattan is still admissible but meaningfully weaker.
- This makes stages 1 and 2 nontrivial instead of teaching the model to predict zero.

## 4.2 Target

Predict:

`delta_target = true_cost_to_go_space_time - h_base`

then clamp to nonnegative in target construction.

Recommended transformed target for training:

`y = log1p(delta_target)`

At inference:
- `delta_pred = expm1(y_pred)` if the head outputs unconstrained log-space values, or
- `delta_pred = softplus(raw)` if using raw-step regression.

Preferred first choice:
- predict **log1p residual**.

## 4.3 Planner use

Use:

`f = g + h_base + alpha * delta_pred`

with:
- **no rounding**
- float32 arithmetic
- no extra clipping beyond the positivity mechanism already built into the output transform

This addresses one of the likely causes of the earlier null results: the learned term being too small to reorder the frontier.

## 4.4 Alpha calibration

Do **not** hard-code `alpha = 1.0` forever.

After each stage, tune `alpha` on a small held-out validation suite:
- candidate set: `{0.5, 1.0, 1.5, 2.0}`
- choose by best success, then lowest expansions as tie-breaker

Use the chosen alpha for final evals at that stage.

---

## 5. Data collection redesign

## 5.1 Oracle labels

Keep the existing reverse dynamic-programming / true-CTG labeling idea. That part is good.

## 5.2 State sampling mixture

Do **not** train only on baseline-A* closed lists.

Per episode, sample training states from a mixture:

- **40% baseline closed-list nodes**
- **30% near-solution-path states**
  - states on or one move from the successful path
- **30% uniformly sampled valid space-time states**
  - free cells, random `t` in horizon

This prevents the dataset from collapsing onto the baseline search distribution.

## 5.3 Stratify by residual magnitude

Oversample states with larger residuals.

Simple rule:
- bucket by `delta_target` into
  - zero / tiny
  - medium
  - large
- sample roughly evenly across buckets for each stage dataset

This reduces the risk that the model learns to predict near-zero everywhere.

## 5.4 Dataset reuse

For each curriculum stage:
- collect the dataset once,
- save it on the Modal volume,
- train both the full-ft and LoRA arms on the *same* dataset.

---

## 6. Losses and training objective

## 6.1 Primary loss

Use **weighted Huber** on `log1p(delta_target)`.

Suggested weighting:
- `w = 1 + 2 * 1[delta_target > 0]`
or
- `w = 1 + min(delta_target / 10, 3)`

The exact weighting can be tuned later; the point is to avoid rewarding trivial near-zero predictions too strongly.

## 6.2 Optional auxiliary loss (not in v1 unless needed)

A pairwise ranking loss on sampled node pairs from the same episode / time slice.

This would directly encourage better frontier ordering, but I would keep it **out of v1** so the first clean rebuild changes only the essentials.

---

## 7. Curriculum

## Core curriculum

### Stage 1: `A32_static`
- family: `A`
- size: `32`
- dynamics: none
- max_steps: `80`
- horizon: `18`

### Stage 2: `A64_static`
- family: `A`
- size: `64`
- dynamics: none
- max_steps: `160`
- horizon: `20`

### Stage 3: `A64_sparseDyn`
- family: `A`
- size: `64`
- subdued dynamics only
- recommended obstacle mix:
  - `1 gate`
  - `1 patroller`
  - `0 drifters`
- max_steps: `170`
- horizon: `20`

## Stretch stage

### Stage 4: `A64_fullDyn`
- family: `A`
- size: `64`
- full dynamic mix
- recommended obstacle mix:
  - `2 gates`
  - `4 patrollers`
  - `4 drifters`
- max_steps: `180`
- horizon: `22`

---

## 8. Evaluation suites

## Core suites
- `ID_A32_static`
- `ID_A64_static`
- `ID_A64_sparseDyn`
- `OOD_B64_static`
- `OOD_C64_static`
- `OOD_B64_sparseDyn`
- `OOD_C64_sparseDyn`

## Stretch suites
- `ID_A64_fullDyn`
- `OOD_B64_fullDyn`
- `OOD_C64_fullDyn`

## Budgets
Use:
- `200`
- `500`
- `2000`

## Episodes
- validation/tuning: `20` per suite
- final reported eval: `100` per suite

---

## 9. Model arms to run in parallel

Keep only the 3M-ish arms in the first clean sweep.

### Full fine-tune
- `onlstm_fullft`
- `hrm_fullft`

### LoRA
- `onlstm_lora`
- `hrm_lora`

## LoRA policy

Stage 1:
- train the full base normally for both arms

Stages 2+:
- **full-ft arm:** continue updating all parameters
- **LoRA arm:** freeze the stage-1 base and train only the new stage adapter stack
  - LoRA on `Linear`
  - LoRA on `Conv2d`
  - optionally LoRA on ON-LSTM / recurrent matrices if implemented cleanly
  - allow bias tuning (`BitFit`) for later stages

This makes the comparison interpretable:
- same stage-1 base
- different transfer mechanism for stages 2+

---

## 10. Metrics that matter

Report the usual:
- success rate
- timeout rate
- avg steps
- avg expansions
- expansions per step

But add the diagnostics we were missing before:

### Model-output diagnostics
- mean / std / max of `delta_pred`
- fraction of nodes with `delta_pred > 0`
- correlation of `delta_pred` with `delta_target`
- calibration by residual bucket

### Planner-influence diagnostics
- fraction of candidate sets whose ordering changes vs baseline
- average top-k rank displacement relative to baseline
- success gain conditioned on high-residual states

Without these, another null result would still be hard to interpret.

---

## 11. Modal implementation plan

## Code structure

Create one new clean experiment file, plus optionally a tiny helper module later.

Recommended entrypoint:

`hrm-cloud/transfer_astar_heuristic_clean_parallel.py`

Internally it should support:
- shared stage dataset collection
- shared cached eval shards
- both transfer modes (`fullft`, `lora`)
- both models (`onlstm`, `hrm`)

## Important inherited fixes
Keep these from the patched scripts:
- sharded eval
- per-episode shard checkpoints
- non-preemptible CPU eval where available
- `RUN_TAG`
- `MODEL_RUN_TAG`
- `VOLUME_NAME`
- preemption-safe orchestration
- cached aggregation of eval shards

## Shared artifacts
Use a run family layout like:

- `/datasets/<stage>.pt`
- `/models/<arm>__<model>__<stage>.pt`
- `/results/eval_shards/...`
- `/results/eval_agg/...`
- `/results/diagnostics/...`

This keeps full-ft and LoRA runs side-by-side under one experiment family.

---

## 12. Practical pitfalls to watch

1. **DeepSapientHRM may need a smaller learning rate** than ON-LSTM.
2. **Map-size scaling can break CNN calibration** if coordinate channels are omitted.
3. **Large 64x64 frame sequences increase memory**; adaptive pooling is important.
4. **Do not feed exact future dynamic occupancy patches** to the node head.
5. **Do not judge success only by success rate**; inspect ordering-change diagnostics too.
6. **Do not add few-shot yet**. First prove the heuristic changes search in zero-shot transfer.

---

## 13. Concrete implementation order

1. Build shared map / occupancy / true-CTG utilities by reusing the mapscale script.
2. Build the new step-frame encoder and node-query head.
3. Integrate ON-LSTM and DeepSapientHRM from `onlstm_hrm_comparison_presetm_v2.py`.
4. Swap target from residual-over-static-distance to residual-over-Manhattan.
5. Implement mixed state sampling and residual bucketing.
6. Add full-ft vs LoRA transfer modes.
7. Add alpha tuning on held-out validation.
8. Add diagnostic logging for `delta_pred` and frontier reordering.
9. Run the core 3-stage curriculum first.
10. Only then enable the stretch `fullDyn` stage.

---

## 14. Recommended default run matrix

Start with this and nothing else:

- models: `onlstm`, `hrm`
- size: ~3M class only
- transfer modes: `fullft`, `lora`
- stages: `A32_static -> A64_static -> A64_sparseDyn`
- stretch: off
- eval suites: core suites only

That gives the cleanest first answer:
- does the new framing work at all?
- does LoRA help or hurt transfer relative to full fine-tune?
- does either model actually change A* search on Family B?

---

## 15. The next build step

The next concrete build step should be:

**implement the shared clean experiment core first, then expose `fullft` and `lora` as two modes on top of it.**

That keeps the comparison controlled and avoids another round of script drift.