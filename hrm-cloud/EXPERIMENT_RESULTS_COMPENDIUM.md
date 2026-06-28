# HRM Modal Experiment Results Compendium

Generated: 2026-06-01

This document summarizes the HRMv2 Modal experiment history I found in the local repo, local downloaded result snapshots, Modal volumes, Notion, and Linear. It focuses on the discrete-grid learned A* heuristic work, then maps the findings to the newer continuous-PRM action items.

## Evidence

Primary evidence sources:

- Modal survey script: `hrm-cloud/modal_experiment_survey.py`
- Full Modal JSON survey output: `modal_downloads/full_survey_sdk_parallel/`
- Survey manifest: `modal_downloads/full_survey_sdk_parallel/manifest.json`
- Survey summary: `modal_downloads/full_survey_sdk_parallel/summary.json` and `summary.md`
- Pre-interruption residual snapshot: `modal_downloads/eval_agg_dir/eval_agg/`
- Repo writeups: `BENCHMARK_RESULTS.md`, `LSTM_VS_HRM_EXPERIMENT.md`, `ONLSTM_VS_HRM_EXPERIMENT_PRESETM_V2.md`, `experiment_writeup_last_two_runs.md`, and `clean_transfer_experiment_blueprint.md`
- Notion sources:
  - [AI/ML Research - Continuous PRM Heuristic Learning](https://www.notion.so/36e0aeaff19881df81cee7cd0391c989)
  - [Research Direction](https://www.notion.so/36e0aeaff19881608306f7a8d65b5f5e)
  - [C3 Residual Task-LoRA Experts Spec](https://www.notion.so/36e0aeaff198812b812dc11b8b6ca00c)
- Linear sources:
  - [Continuous PRM Heuristic Learning (C1-C4)](https://linear.app/hhari-proj-research/project/continuous-prm-heuristic-learning-c1-c4-5f9b36657620)
  - [PRO-14: Check residual prediction stability](https://linear.app/hhari-proj-research/issue/PRO-14/cprm-013-check-residual-prediction-stability)
  - [PRO-15: Train C3 HRM LoRA experts](https://linear.app/hhari-proj-research/issue/PRO-15/cprm-020-train-c3-hrm-lora-experts)
  - [PRO-19: Audit bounded residual behavior](https://linear.app/hhari-proj-research/issue/PRO-19/cprm-024-audit-bounded-residual-behavior)
  - [PRO-24: Write continuous PRM experiment report](https://linear.app/hhari-proj-research/issue/PRO-24/cprm-040-write-continuous-prm-experiment-report)

The exhaustive Modal survey found 15,267 file manifest entries and downloaded 13,671 JSON files with zero download errors. Unless stated otherwise, means below are unweighted means over suite-budget rows; lower average expansions is better. Most final-result rows use 100 evaluation episodes.

## Executive Summary

The most recent HRM-related Modal activity is `residual-tasklora-v2-vol`. A live recheck on 2026-06-01 found its latest file modified at 2026-06-01 10:33:42 UTC, which is 2026-06-01 03:33:42 PT. That activity continued the same residual eval run that had previously shown nonfinite-eval spam. There is no newer HRM/HRMv2 Modal volume in the surveyed set.

Update from live Modal recheck on 2026-06-01: `residual-tasklora-v2-vol` has continued activity after the earlier snapshot. The latest file is now an eval shard modified at 2026-06-01 10:33:42 UTC, or 2026-06-01 03:33:42 PT. This still did not produce a `final_results` file or new aggregate JSONs; the new evidence is additional/updated eval shards.

The residual run was not complete. It has no `final_results` file. The current Modal volume has 227 aggregate eval files; the older clean local snapshot has 222. The five files added after the interrupted rerun should be treated as quarantine evidence until rerun after the prediction cap/nonfinite fix.

The last complete transfer-style discrete results are:

- `transfer-astar-heuristic-clean-parallel-v3-vol`: clean transfer final file, completed 2026-04-05 20:33:03 PT.
- `multitask-astar-heuristic-tasklora-v1-vol`: multitask TaskLoRA final file, completed 2026-04-03 05:44:14 PT.

The strongest completed positive result is not a task expert. It is the pooled HRM average-base model in the multitask TaskLoRA run: `avgbase__hrm` scored 0.612 success versus matched baseline 0.591, a +2.11 percentage point gain, with lower mean expansions. The specialist TaskLoRA experts looked high in absolute terms, but their matched baseline subset was also high; only `tasklora__hrm__A32_static` beat its matched baseline, and only by +2.0 points.

Most learned heuristic experiments did not beat the static Manhattan A* baseline broadly. Clean full fine-tuning and clean LoRA underperformed baseline. CondLoRA was incomplete and underperformed on the completed ONLSTM rows. The original imitation/LoRA and map-scale experiments were near-null.

The residual TaskLoRA v2 direction is still scientifically important, but its current result is partial. The pre-interruption snapshot shows `residtasklora__hrm__A32_static` at 0.706 success versus matched baseline 0.696 (+0.93 points) over 27 suite-budget rows, with slightly fewer expansions. The current Modal volume adds five more aggregate rows, but those were produced during the nonfinite incident and should not be used as clean headline results.

## Modal Volume Chronology

Times are shown in Pacific time for readability. The manifest stores UTC timestamps.

| Volume | Latest file time (PT) | JSONs pulled | Status | Primary evidence |
| --- | ---: | ---: | --- | --- |
| `residual-tasklora-v2-vol` | 2026-05-31 21:06:41 | 2,024 | Latest, partial/interrupted | 227 eval aggregates, no final file |
| `transfer-astar-heuristic-clean-parallel-v3-vol` | 2026-04-05 20:33:03 | 3,983 | Complete | `final_results__A64_moderateDyn.json` |
| `transfer-astar-heuristic-avg-condlora-basis-v1-vol` | 2026-04-03 18:50:45 | 2,417 | Partial | 329 eval aggregates, no final file |
| `multitask-astar-heuristic-tasklora-v1-vol` | 2026-04-03 05:44:14 | 3,438 | Complete | `final_results__multitask_tasklora.json` |
| `transfer-astar-heuristic-clean-parallel-v2-vol` | 2026-03-26 08:37:02 | 52 | Smoke/partial | 16 eval aggregates, 20 episodes only |
| `transfer-astar-heuristic-clean-parallel-v1-vol` | 2026-03-26 01:17:29 | 0 | Training partial | Checkpoints only |
| `transfer-astar-heuristic-imitation-v2-vol-v2` | 2026-03-19 11:13:53 | 1,553 | Complete follow-ups | fixpack and map-scale `results__*.json` |
| `transfer-astar-heuristic-imitation-v2-vol` | 2026-03-19 04:24:10 | 200 | Complete/partial mix | original result and baseline rerun |
| `transfer-astar-heuristic-rl-vol` | 2026-02-23 22:16:30 | 1 | Complete zero-shot | `eval_zero_shot.json` |
| `onlstm-hrm-comparison-presetm-v2-vol` | 2026-02-05 23:35:27 | 1 | Complete | `results.json` |
| `onlstm-hrm-comparison-presetm-vol` | 2026-02-03 20:12:28 | 1 | Complete, superseded | `results.json` |
| `lstm-hrm-comparison-vol` | 2026-01-22 08:58:58 | 1 | Complete | `comparison_results.json` |
| early HRM model volumes | 2025-11-28 to 2025-12-07 | 0 | Model/checkpoint evidence | `.pt`, `.zip`, `.npz` artifacts |

## Shared Transfer-Heuristic Methodology

This section applies most directly to the transfer-style learned A* experiments: imitation/LoRA, clean transfer, CondLoRA, multitask TaskLoRA, and residual TaskLoRA. Earlier obstacle-prediction benchmarks used a different interface and are described separately below.

Common planning interface:

- The planner is receding-horizon space-time A* on a 4-connected grid with a wait action.
- Each agent step replans from the current state using a fixed future occupancy rollout for gates, patrollers, and drifters.
- The baseline heuristic is Manhattan distance to goal.
- Learned models predict a nonnegative residual in step units, not an absolute cost-to-go: `delta_target = max(0, true_space_time_cost_to_go - h_manhattan)`.
- The planner priority is `f = g + h_manhattan + alpha * delta_pred`. In the clean transfer family, `delta_pred` is produced from a `log1p(delta)` head and is scaled by an alpha chosen from `{0.5, 1.0, 1.5, 2.0}`.

Common training data construction:

- Offline datasets are collected by running an oracle/static space-time A* planner on Family-A curriculum tasks.
- Candidate node labels come from the oracle cost-to-go grid at absolute time `t_abs + t_rel`.
- Candidate nodes are sampled from three sources: the planner closed list, states near the true path, and uniformly sampled reachable valid states.
- Later clean/residual scripts stratify selected nodes by residual magnitude buckets so that zero-residual nodes do not dominate training.
- The model loss is a masked Smooth L1/Huber-style loss on `log1p(delta_target)`, weighted upward for larger residuals.

Common learned-heuristic inputs in the clean transfer family:

- Observation history length is 20 frames.
- Each frame has 8 channels: walls, agent position, goal, gate occupancy, patroller occupancy, drifter occupancy, normalized x coordinate, and normalized y coordinate.
- Each queried A* node also gets a local 2-channel static/dynamic occupancy patch. The default patch radius is 7, so the patch is 15x15.
- Node metadata has 6 normalized values: goal dx, goal dy, time offset, Manhattan distance, node x coordinate, and node y coordinate.
- The recurrent backbone runs once per environment step/history window to produce a context vector; a lightweight node head scores many candidate A* nodes from that context.

Common clean-transfer model configuration:

- ONLSTM backbone: frame dim 256, hidden dim 480, 2 layers, chunk size 8, patch dim 160, node hidden dim 320, LoRA rank 24.
- HRM backbone: frame dim 256, hidden dim 256, 2 layers, `k_step=2`, 4 attention heads, patch dim 160, node hidden dim 320, LoRA rank 8.
- Training defaults are batch size 8, `LR_FULL=2e-4`, `LR_LORA=5e-4`, full weight decay `1e-3`, LoRA weight decay `0`, and grad clip norm `1.0`.

Common clean-transfer curriculum:

| Stage | Family | Size | Dynamics | Max steps | Horizon | Samples | Nodes/sample | Epochs | Dynamic objects |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `A32_static` | A | 32 | static | 80 | 18 | 10,000 | 1,400 | 2 | 0 gates, 0 patrollers, 0 drifters |
| `A64_static` | A | 64 | static | 160 | 20 | 15,000 | 2,200 | 3 | 0 gates, 0 patrollers, 0 drifters |
| `A64_sparseDyn` | A | 64 | sparseDyn | 170 | 20 | 20,000 | 2,600 | 3 | 1 gate, 1 patroller, 0 drifters |
| `A64_moderateDyn` | A | 64 | moderateDyn | 175 | 21 | 22,000 | 2,800 | 3 | 1 gate, 2 patrollers, 1 drifter |
| `A64_fullDyn` | A | 64 | fullDyn | 180 | 22 | 25,000 | 3,200 | 3 | 2 gates, 3 patrollers, 2 drifters; optional stretch |

Common final evaluation protocol:

- Default final evaluation is 100 episodes per suite and budgets `{200, 500, 2000}` expansions per replanning step.
- Default alpha tuning uses 20 validation episodes at budget 500.
- The 22-suite non-stretch set includes four ID Family-A tasks, six B/C OOD size-64 tasks, and twelve A-family size-transfer tasks at sizes 96, 128, 192, and 256 across static/sparseDyn/moderateDyn.
- Stretch mode adds fullDyn ID/B/C and A-family size-transfer suites, but the completed headline runs in this compendium are non-stretch unless noted.
- Eval artifacts are sharded by model, suite, budget, alpha, episode range, then aggregated into `eval_agg` rows and sometimes into a `final_results__*.json` file.

## Experiment Results

### Early HRM Scaling And Diffusion Baselines

The early benchmark documentation records the progression from small HRM to scaled HRM. The headline result is the 8-GPU HRM win: HRM at 28.97M parameters achieved 68% success versus the LSTM baseline at 66%. Mid-scale HRM reached 62%, and small HRM reached 52%. The same writeup records diffusion planner v2 at 60% and an earlier diffusion v1 around 64%.

Methodology:

- Task: dynamic 20x20 grid navigation with 12 static obstacles and 6 bouncing dynamic obstacles.
- Learned world-model input: normalized dynamic-obstacle position history with shape `(6 obstacles, 20 timesteps, 2 coordinates)`.
- Learned world-model target: next-step position delta per dynamic obstacle.
- Data collection: simulated episodes of 70 physics steps; samples use `past[-21:-1]` as history and `current - previous` as target.
- Dataset scale: the small run used about 18k episodes and roughly 1.08M samples; the larger run used 60k episodes and roughly 18M samples.
- Planner evaluation: SpaceTimeAStar predicts future obstacle positions for horizon 20, runs A* in `(x, y, t)`, executes the first action, and replans at the next step.
- Success metric: reach the goal without collision within 80 agent steps, reported over 50 episodes in the benchmark writeup.
- HRM scaling: small HRM used hidden 128, 2 layers, 4 heads; mid HRM used hidden 256 with gated recurrent blocks; full HRM used hidden 512, 4 layers, 8 heads.
- Large HRM training: 8x H100 DDP, global batch 16,384, learning rate 6e-4, OneCycleLR, AdamW, BF16, gradient clip 1.0, 30 epochs.
- Diffusion baselines were path generators rather than residual heuristics: they consumed a static 64x64 map plus start/goal and emitted 2-channel path maps, then sampled/refined paths with SDF-style post-processing.

Interpretation: HRM could beat LSTM in the original obstacle-prediction/planner pipeline, but it needed substantial scale and tuned training. This did not automatically transfer to later learned-heuristic A* experiments.

### LSTM vs HRM Comparison

Source volume: `lstm-hrm-comparison-vol`

Methodology:

- Task: the same 20x20 dynamic grid/planner pipeline as the early benchmark, using 12 static obstacles, 6 dynamic obstacles, history length 20, rollout horizon 20, and dynamic obstacle speed 0.7.
- Training data: 60k simulation episodes, approximately 18M next-delta samples.
- Inputs/targets: 20-step normalized position histories predict the next obstacle-position delta.
- Optimizer/protocol: AdamW with OneCycleLR; HRM used gradient clipping.
- Evaluation: 100 fixed evaluation episodes with seed offset `+1000`, shared across all models.
- LSTM tiers: `lstm_300k` hidden 160, 2 layers, batch 4096, LR 1e-3, 30 epochs; `lstm_1m` hidden 290; `lstm_3m` hidden 500; `lstm_10m` hidden 900, 3 layers, batch 2048, LR 5e-4, 40 epochs.
- HRM tiers: `hrm_302k` hidden 128, 2 layers, 4 heads, batch 4096, LR 1e-3, 40 epochs; `hrm_3m` hidden 256, LR 4e-4; `hrm_10m` hidden 384, 3 layers, 6 heads, batch 2048, LR 5e-4.

| Model | Success |
| --- | ---: |
| `lstm_300k` | 0.67 |
| `lstm_1m` | 0.67 |
| `lstm_3m` | 0.69 |
| `lstm_10m` | 0.68 |
| `hrm_302k` | 0.64 |
| `hrm_3m` | 0.71 |
| `hrm_10m` | 0.71 |

Interpretation: HRM 3M/10M was the best in this comparison at 0.71 success, but small HRM lagged the LSTM family. This established HRM as a viable backbone for later experiments.

### ONLSTM vs HRM Preset M

Source volumes: `onlstm-hrm-comparison-presetm-vol`, `onlstm-hrm-comparison-presetm-v2-vol`

Methodology:

- Preset M+ v2 increases difficulty to 32x32 room/corridor maps with max 128 agent steps.
- The planner observes the last 20 positions of each dynamic obstacle and rolls a learned predictor forward for the A* planning horizon.
- Dynamic obstacle mix: 2 gates, 4 patrollers, and 6 drifters.
- Model input adds a local 11x11 binary occupancy patch centered on each obstacle's last observed position; a small CNN encodes that patch before it is concatenated with position features.
- Training target is multi-step autoregressive rollout, `k=5`, with weighted MSE that decays across prediction steps and scheduled sampling.
- Evaluation suites: `random_h20`, `random_h10`, `fixedmap_h20`, and `gatesoff_random_h20`.
- Metrics include planner success/collision/static-collision/timeout, node expansions, one-step MSE, and k-step rollout MSE by obstacle class.
- Model tiers follow the ONLSTM and HRM size families, with the 10M HRM using 4-GPU DDP.

Preset M v1 was a very hard/low-success setup: ONLSTM ranged from 0.15 to 0.20 and HRM ranged from 0.08 to 0.10. Preset M+ v2 expanded to four suites; mean success across suites was:

| Model | Mean success |
| --- | ---: |
| `onlstm_10m` | 0.388 |
| `onlstm_1m` | 0.373 |
| `onlstm_3m` | 0.365 |
| `hrm_302k` | 0.343 |
| `onlstm_300k` | 0.323 |
| `hrm_10m` | 0.265 |
| `hrm_3m` | 0.200 |

Interpretation: ONLSTM was stronger than HRM on this Preset M+ dynamic benchmark. This justified keeping ONLSTM as a comparison backbone in later transfer experiments, even though HRM later looked better in average-base learned-heuristic runs.

### Transfer RL Zero-Shot

Source volume: `transfer-astar-heuristic-rl-vol`

Methodology:

- This was the first transfer-first learned A* run. Despite the "RL" filename, the core planner remained Space-Time A* with a learned delta-h correction.
- Curriculum training used Family A only: `stage1_A32_D0`, `stage2_A32_D1`, `stage3a_A64_D1`, and `stage3b_A64_D2`.
- The planner priority used `f = g + h_static + alpha * delta_h`, with alpha fixed at 1.0 in the stage/eval config.
- Model observations combined a local static-map patch, goal vector, obstacle features, time feature, and static distance. Obstacle features encode relative position, type one-hot, and gate-closed state.
- The recurrent core was either ONLSTM or HRM; the learned head predicted bounded delta-h values for queried nodes.
- Evaluation was zero-shot on six suites: `ID_A32_D1`, `ID_A64_D2`, `OOD_B32_D1`, `OOD_C32_D1`, `OOD_B64_D2`, and `OOD_C64_D2`.
- Suite episode counts were 100 for 32x32/ID-style suites and 120 for the 64x64 B/C D2 suites.
- Few-shot adaptation existed in the script for `TARGET_B64_D2` with K in `{50, 200, 1000}`, but the Modal evidence summarized here is the zero-shot `eval_zero_shot.json`.

| Model | Rows | Mean success | Mean avg expansions |
| --- | ---: | ---: | ---: |
| `baseline_static_astar` | 6 | 0.967 | 143,319 |
| `hrm_3m` | 6 | 0.961 | 142,193 |
| `onlstm_3m` | 6 | 0.956 | 141,255 |

Interpretation: the baseline was already very strong. Learned models slightly reduced expansions but also slightly reduced success, so this was not evidence of a planner-level win.

### Imitation/LoRA Transfer And Map-Scale Follow-Up

Source volumes: `transfer-astar-heuristic-imitation-v2-vol`, `transfer-astar-heuristic-imitation-v2-vol-v2`

Methodology:

- These runs replaced the earlier rollout/RL setup with supervised imitation of a residual heuristic in step units.
- Target definition: `delta_h_target(x,y,t) = max(0, true_cost_to_go(x,y,t) - h_static(x,y))`.
- A* priority used `f = g + h_static + alpha * ReLU(delta_h_pred)`, with alpha normally fixed at 1.0 rather than alpha-tuned per model.
- Training data came from oracle Space-Time A* closed-list nodes, described in the script as the "search graveyard"; this was intended to reduce survivorship bias from training only on successful path states.
- The recurrent model ran once per environment step to produce context, while a node MLP scored thousands of A* candidate nodes from node-centric features.
- Node features included relative goal features and projected dynamic occupancy at the queried future node time.
- Default stage-wise LoRA settings were rank 8, alpha 16, train base weights on stage 1, then freeze base weights and train the newest adapter on later stages.
- Original curriculum suites were `ID_A32_D1`, `ID_A64_D2`, `OOD_B32_D1`, `OOD_C32_D1`, `OOD_B64_D2`, and `OOD_C64_D2`.
- Map-scale follow-up suites were `ID_A32_static`, `ID_A64_static`, `ID_A64_sparseDyn`, `OOD_B64_static`, `OOD_C64_static`, `OOD_B64_sparseDyn`, and `OOD_C64_sparseDyn`.
- Few-shot adaptation fine-tuned on the hardest target suite with K in `{50, 200}` episodes.
- Evaluation used strict per-replan budgets `{200, 500, 2000}` and generally 100 episodes per suite.

Original/fixpack results:

| Model group | Rows | Mean success | Mean avg expansions |
| --- | ---: | ---: | ---: |
| baseline, HRM, ONLSTM zero-shot | 18 each | 0.797 | about 27.4k |
| few-shot variants on hardest suite | 3 each | 0.340 | about 73k |

Map-scale LoRA follow-up:

| Model group | Rows | Mean success | Mean avg expansions |
| --- | ---: | ---: | ---: |
| baseline, HRM, ONLSTM zero-shot | 21 each | 0.803 | about 34.4k |
| few-shot variants on hardest suite | 3 each | 0.340 | about 65k |

Interpretation: both experiments were near-null. The models trained and ran, but planner-level behavior tracked the static baseline. The map-scale run clarified that family-B OOD cases were the hard bottleneck, while family-C and many ID/A-family cases were easy.

### Clean Transfer v3

Source volume: `transfer-astar-heuristic-clean-parallel-v3-vol`

Final file: `final_results__A64_moderateDyn.json`

Methodology:

- This is the clean rebuild proposed by `clean_transfer_experiment_blueprint.md`.
- It uses the shared clean-transfer setup above: Manhattan baseline, residual target `log1p(max(0, true_cost_to_go - Manhattan))`, 20-frame spatial history, node patches, and node metadata.
- Curriculum stages were `A32_static`, `A64_static`, `A64_sparseDyn`, and `A64_moderateDyn`; the optional `A64_fullDyn` stretch stage was not part of the completed headline file.
- The comparison was fully crossed over two backbones and two adaptation modes: `fullft__onlstm`, `fullft__hrm`, `lora__onlstm`, and `lora__hrm`.
- Full fine-tuning updates all model parameters at every curriculum stage.
- LoRA trains the same stage-1 base model, then freezes the inherited base and trains stage-specific LoRA adapters on later stages.
- Alpha is tuned separately for each learned model/stage using validation suites, 20 validation episodes, budget 500, and candidates `{0.5, 1.0, 1.5, 2.0}`.
- Final evaluation includes the baseline on all 22 suites x 3 budgets, and every learned model on all 22 suites x 3 budgets using its selected alpha.
- The Modal implementation is sharded and cacheable: episode ranges are saved as `eval_shards`, merged into `eval_agg`, then into `final_results__A64_moderateDyn.json`.

| Model | Rows | Suites | Mean success | Matched delta vs baseline | Mean avg expansions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_manhattan_astar` | 66 | 22 | 0.590 | reference | 126,707 |
| `fullft__hrm` | 66 | 22 | 0.585 | -0.59 pp | 128,013 |
| `fullft__onlstm` | 66 | 22 | 0.564 | -2.67 pp | 127,228 |
| `lora__hrm` | 66 | 22 | 0.554 | -3.62 pp | 140,852 |
| `lora__onlstm` | 66 | 22 | 0.514 | -7.67 pp | 145,282 |

Interpretation: this was a clean negative result. Full fine-tuning was closer than LoRA, HRM was better than ONLSTM, but every learned variant underperformed matched static Manhattan A*.

### CondLoRA Basis v1

Source volume: `transfer-astar-heuristic-avg-condlora-basis-v1-vol`

There is no final results file. The useful evidence is 329 eval aggregate JSONs.

Methodology:

- This run keeps the clean-transfer data/model interface but changes the adaptation structure.
- Stage-specific training is replaced by a pooled average model, `avgft`, trained on the union of all selected training-stage datasets.
- The conditional LoRA arm, `hyplora`, freezes the pooled average model and adds a set of low-rank LoRA basis matrices.
- A small hypernetwork/controller maps task descriptors into mixture weights over the LoRA bases, producing a task-conditioned residual adapter instead of one manually selected adapter per task.
- Default modes are `TRAIN_TRANSFER_MODES=avgft,hyplora` and `EVAL_TRANSFER_MODES=avgft,hyplora`.
- The pooled stage id is `pooled_train`; `AVG_EPOCHS` and `CONDLORA_EPOCHS` default to the maximum configured stage epoch count unless overridden.
- The number of conditional LoRA bases defaults to `max(4, len(STAGES_TO_RUN))`; controller hidden size defaults to 64 and temperature to 1.0.
- Alpha tuning is performed over validation suites for the pooled stage, again with 20 validation episodes, budget 500, and candidates `{0.5, 1.0, 1.5, 2.0}`.
- Final intended evaluation mirrors clean transfer: baseline plus learned arms over all suites and budgets. The observed Modal volume stopped before producing a final file.

100-episode aggregate rows:

| Model | Rows | Suites | Mean success | Matched delta vs baseline | Mean avg expansions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_manhattan_astar` | 66 | 22 | 0.591 | reference | 126,172 |
| `avgft__onlstm` | 66 | 22 | 0.517 | -7.42 pp | 150,969 |
| `hyplora__onlstm` | 37 | 14 | 0.551 | -7.08 pp | 50,608 |

HRM `avgft` and `hyplora` are only present in 20-episode alpha-tuning rows, not a full 100-episode final evaluation.

Interpretation: incomplete and not favorable. The ONLSTM rows underperform matched baseline. Do not cite this as a completed CondLoRA result.

### Multitask TaskLoRA v1

Source volume: `multitask-astar-heuristic-tasklora-v1-vol`

Final file: `final_results__multitask_tasklora.json`

Methodology:

- This run keeps the clean-transfer data/model interface but changes from stage-wise transfer to multitask specialization.
- The `avgbase` arm trains one pooled model on the union of the selected training tasks with task-balanced sampling.
- The `tasklora` arm freezes the pooled average base and trains one separate LoRA expert for each training stage.
- Default training tasks are `A32_static`, `A64_static`, `A64_sparseDyn`, and `A64_moderateDyn`.
- Default backbones are ONLSTM and HRM, but the important completed positive result is the HRM average base.
- `avgbase` alpha is tuned over the training ID suites; each task expert's alpha is tuned on its matching training ID suite.
- Full-suite evaluation runs baseline and `avgbase` over all 22 suites x 3 budgets.
- By default, task experts are evaluated only on the four training ID suites, not on the full 22-suite set. This is why their absolute success rates must be compared to the matched subset baseline rather than the full-suite baseline.
- The final file records `expert_eval_suites`, stages, run tag, and all row metrics in `final_results__multitask_tasklora.json`.

Full-suite rows:

| Model | Rows | Suites | Mean success | Matched delta vs baseline | Mean avg expansions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_manhattan_astar` | 66 | 22 | 0.591 | reference | 126,630 |
| `avgbase__hrm` | 66 | 22 | 0.612 | +2.11 pp | 122,184 |
| `avgbase__onlstm` | 66 | 22 | 0.545 | -4.59 pp | 139,254 |

Task expert rows are only evaluated on four-suite subsets. The matched baseline for those same suite-budget rows is 0.803 success, so absolute expert success should not be compared directly to the 22-suite baseline mean.

| Expert | Rows | Suites | Mean success | Matched delta vs subset baseline | Mean avg expansions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tasklora__hrm__A32_static` | 12 | 4 | 0.823 | +2.00 pp | 35,590 |
| `tasklora__hrm__A64_static` | 12 | 4 | 0.705 | -9.83 pp | 42,057 |
| `tasklora__hrm__A64_sparseDyn` | 12 | 4 | 0.623 | -18.08 pp | 46,396 |
| `tasklora__hrm__A64_moderateDyn` | 12 | 4 | 0.661 | -14.25 pp | 42,256 |
| `tasklora__onlstm__A32_static` | 12 | 4 | 0.799 | -0.42 pp | 37,598 |
| `tasklora__onlstm__A64_static` | 12 | 4 | 0.723 | -8.08 pp | 41,207 |
| `tasklora__onlstm__A64_sparseDyn` | 12 | 4 | 0.601 | -20.25 pp | 50,069 |
| `tasklora__onlstm__A64_moderateDyn` | 12 | 4 | 0.643 | -16.08 pp | 48,543 |

Interpretation: the pooled HRM average-base is the important completed positive signal. Specialist TaskLoRA did not broadly improve matched performance; the A32 HRM expert was the only specialist that beat its matched baseline.

### Residual TaskLoRA v2

Source volume: `residual-tasklora-v2-vol`

This is the most recent discrete Modal run and the one interrupted after the nonfinite eval issue. It has no final results file.

Live recheck on 2026-06-01:

- Fresh pull: `modal_downloads/residual_latest_20260601/`
- Downloaded JSON files: 2,028.
- Manifest entries for this volume: 2,076, versus 2,072 in the previous full survey.
- New remote paths: 4 eval shards.
- Changed common paths: 20 eval shards with newer modification times/sizes.
- Latest remote file: `residtasklora_hrm_A32_static__OOD_A192_static__B2000__a0.5__eps100__0080_0010.json`, modified 2026-06-01 10:33:42 UTC / 2026-06-01 03:33:42 PT.
- Official aggregates are unchanged: 227 `eval_agg` JSONs, 163 of them 100-episode rows.
- No `final_results__residual_tasklora_v2.json` or other `final_results*.json` file exists in the volume.
- Modal status check showed no active containers; the residual ephemeral app listed zero tasks.

The new/updated shard evidence creates three complete 100-episode shard sets that Modal did not aggregate:

| Local-only shard rollup | Residual success / avg expansions | Matched baseline | Matched avgbase | Nonfinite preds |
| --- | ---: | ---: | ---: | ---: |
| `OOD_A192_sparseDyn`, B=200 | 0.520 / 54,363 | 0.490 / 56,724 | 0.550 / 51,482 | 0 |
| `OOD_A192_static`, B=500 | 0.560 / 125,157 | 0.550 / 127,817 | 0.610 / 115,435 | 43 |
| `OOD_A192_static`, B=2000 | 0.670 / 415,299 | 0.660 / 423,078 | 0.710 / 383,357 | 0 |

Combining official 100-episode `residtasklora__hrm__A32_static` aggregates with those three local-only complete shard rollups gives 34 rows over 12 suites: mean success 0.687, mean expansions 91,161. The matched baseline over those rows is 0.676 success, so the residual expert is +1.09 pp versus baseline. The matched `avgbase__hrm` over those same rows is 0.697 success, so the residual expert is -1.00 pp versus avgbase. Because one local-only row has 43 nonfinite predictions, this combined rollup is diagnostic only and should not be treated as a clean final result.

Methodology:

- Residual TaskLoRA v2 is not identical to multitask TaskLoRA v1. It keeps the pooled average base plus task experts, but the expert is explicitly trained as a bounded correction around the frozen base model.
- Default training/eval arms are `avgbase` and `residtasklora`; storage/display names normalize task expert rows to `residtasklora`.
- Default training tasks are `A32_static`, `A64_static`, `A64_sparseDyn`, and `A64_moderateDyn`; optional fullDyn can be enabled with `ENABLE_A64_FULLDYN`/`INCLUDE_STRETCH_STAGE`.
- The pooled base stage id is `ALL_TASKS`.
- The `avgbase` arm trains on the pooled task distribution, as in multitask v1.
- Each `residtasklora` expert loads the frozen average-base weights, injects one LoRA adapter, and trains only the adapter/bias parameters for its task.
- Before expert training, the script calibrates a residual bound `B` from `abs(target_delta - base_delta)` on that task dataset. Defaults are percentile 99, minimum 16, maximum 128.
- At train and eval time, the expert computes `uncorrected_residual = adapt_delta - base_delta`, then `correction = B * tanh(uncorrected_residual / B)`, then `final_delta = clamp(base_delta + correction, 0, PRED_DELTA_MAX)`.
- Expert loss combines bounded residual fit, total final-delta fit, and a small correction-magnitude penalty: defaults are residual weight 1.0, total weight 0.25, magnitude weight 1e-3.
- Default expert eval is broader than multitask v1: it evaluates Family-A ID suites plus Family-A size-OOD suites by default, unless `EVAL_TASK_EXPERTS_ALL_SUITES=1` disables the Family-A-only filter.
- Numerical guards added in the patched local script include finite checks for base/adapt/final deltas, bounded prediction clamping, nonfinite sanitizer counters, and aggregate diagnostics such as correction saturation, residual target clipping, and nonfinite prediction count.

Pre-interruption local snapshot:

| Model | Rows | Suites | Mean success | Matched delta vs baseline | Mean avg expansions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_manhattan_astar` | 66 | 22 | 0.590 | reference | 126,385 |
| `avgbase__hrm` | 65 | 22 | 0.602 | +1.08 pp | 114,900 |
| `residtasklora__hrm__A32_static` | 27 | 10 | 0.706 | +0.93 pp | 58,802 |

Current Modal volume after the interrupted run:

| Model | Rows | Suites | Mean success | Matched delta vs baseline | Mean avg expansions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_manhattan_astar` | 66 | 22 | 0.590 | reference | 126,385 |
| `avgbase__hrm` | 66 | 22 | 0.600 | +1.02 pp | 125,494 |
| `residtasklora__hrm__A32_static` | 31 | 11 | 0.697 | +1.03 pp | 80,795 |

The current volume differs from the old clean snapshot by five added aggregate JSONs and no changed common aggregate metrics:

- `avgbase_hrm_ALL_TASKS__OOD_A256_moderateDyn__B2000__a0.5__eps100.json`
- `residtasklora_hrm_A32_static__OOD_A128_moderateDyn__B2000__a0.5__eps100.json`
- `residtasklora_hrm_A32_static__OOD_A128_sparseDyn__B2000__a0.5__eps100.json`
- `residtasklora_hrm_A32_static__OOD_A128_static__B2000__a0.5__eps100.json`
- `residtasklora_hrm_A32_static__OOD_A192_static__B200__a0.5__eps100.json`

The interrupted rerun modified 60 residual volume files on 2026-06-01 UTC: 54 eval shards, 5 eval aggregates, and the pooled manifest. The nonfinite terminal spam happened during this run. The aggregate rows do not record nonfinite counts, so absence of `nonfinite_pred_count` in aggregate rows is not enough to trust them.

Interpretation: residual TaskLoRA v2 is the latest and most relevant discrete experiment, but it is partial and contaminated by the interrupted nonfinite run. Use the old 222-file aggregate snapshot as the clean pre-interruption evidence, and rerun modeled evals after the prediction cap/nonfinite tracking fix before publishing final claims.

> **Continuation (2026-06-27) — [`EXPERIMENT_RESULTS_FOCAL_REDESIGN.md`](EXPERIMENT_RESULTS_FOCAL_REDESIGN.md).** The rerun, diagnosis, and redesign that resolve this section. Summary: the learned heuristic turned out to be a near-perfect *ranker* (ρ≈0.99 with true cost-to-go) but a scale-miscalibrated *magnitude*; added onto admissible Manhattan and weighted by a global α tuned on small maps, it was suppressed to ≈Manhattan (α pinned to the floor on all 10 models) — which is exactly the "ordering changed in unhelpful ways" failure noted in Cross-Experiment Conclusions below. Re-integrated as a **focal-search ranker** (`PLANNER=focal`), it delivers ~15% fewer A\* expansions at matched-or-better success on **both** HRM and ON-LSTM bases, while the per-task LoRA experts still add nothing over the base (the bounded residual is too small to reorder nodes), confirming "avgbase > specialist."

## Cross-Experiment Conclusions

Static Manhattan A* is a strong baseline. Many learned residuals either do not change planner behavior or degrade it, especially when the heuristic correction changes ordering in unhelpful ways.

HRM is usually the better learned-heuristic backbone in the later transfer experiments. In clean transfer, HRM full fine-tune was closest to baseline and HRM LoRA beat ONLSTM LoRA. In multitask TaskLoRA, `avgbase__hrm` improved while `avgbase__onlstm` underperformed.

ONLSTM still matters as a comparator. It outperformed HRM on the Preset M+ dynamics benchmark, so it should not be discarded globally. It just did not transfer well to the later learned-heuristic A* interface.

The average-base framing is more promising than specialist LoRA by itself. The best completed result is pooled HRM average-base, not per-task adapters. Specialist residuals may need better routing, better task descriptors, or a different target before they consistently help.

The family-B/OOD distribution shift remains a persistent hard case in the earlier transfer runs. Budget increases often increased expansions without improving success, which means the bottleneck is not simply too little search.

The residual formulation needs strict numerical constraints. The Linear and Notion continuous-PRM tasks explicitly require no NaN/Inf rows, bounded corrections, and nonfinite checks. The discrete residual nonfinite incident is exactly the failure mode those action items are trying to prevent.

## Notion And Linear Action Items

The Notion/Linear work is primarily about the newer continuous PRM direction, not another discrete-grid Modal run. However, those tasks were explicitly motivated by the corrected discrete residual setup.

Relevant Notion requirements:

- The continuous PRM hub says the next step is moving from discrete grid learned A* heuristics to continuous geometric planning with PRM graphs.
- The C3 spec keeps the same average-base plus bounded task-specific residual LoRA equation.
- The C3 acceptance criteria require bounded corrections, no NaN/Inf losses/predictions/heuristics, matched experts not systematically worse than avgbase, and an oracle specialization gap.

Relevant Linear issues:

- PRO-14: inspect residual ranges, heuristic min/max, and nonfinite prediction counts; acceptance requires no NaN/Inf rows or documented/fixed bound issues.
- PRO-15: train C3 HRM LoRA experts after the average-base model exists.
- PRO-19: audit bounded residual behavior; acceptance requires correction magnitude plots and nonfinite checks.
- PRO-24: write the continuous PRM experiment report after the C1-C4 results exist.

Practical consequence: before trusting or extending residual TaskLoRA into continuous PRM, the discrete eval path should fail loudly or record sanitizer counts whenever nonfinite tensors are found. Silently replacing nonfinite values with zero can keep a run alive, but it makes results hard to interpret.

## Recommended Next Steps

1. Quarantine the five residual aggregate JSONs added during the 2026-05-31 PT interrupted run.
2. Keep the 222-file pre-interruption residual aggregate snapshot as the clean current evidence.
3. Rerun residual TaskLoRA modeled evaluation with prediction delta clamping enabled and nonfinite eval counts recorded in every shard and aggregate row.
4. Treat any nonzero sanitizer count as a failed eval row unless there is a deliberate diagnostic reason to keep it.
5. After a clean residual rerun, compare:
   - `avgbase__hrm` versus matched baseline over all 22 suites.
   - each residual expert versus matched baseline on its evaluated suite-budget subset.
   - each residual expert versus `avgbase__hrm` on the same rows.
6. Only then use the discrete result as prior evidence for continuous PRM C3.

Suggested full residual eval rerun command after the local nonfinite fix:

```powershell
$env:VOLUME_NAME="residual-tasklora-v2-vol"
$env:RUN_TAG="residual_tasklora_v2_predcap_rerun"
$env:MODEL_RUN_TAG="residual_tasklora_v2"
$env:SKIP_COLLECT=1
$env:SKIP_TRAIN=1
$env:SKIP_ALPHA_TUNE=1
$env:FORCE_REEVAL_MODELED=1
$env:EVAL_EPISODES=100
$env:EVAL_BUDGETS="200,500,2000"
$env:ONLY_MODELS="hrm"
$env:MAX_PARALLEL_EVAL=32
$env:PRED_DELTA_MAX=2048
$env:SANITIZE_NONFINITE_EVAL=1
$env:SANITIZE_NONFINITE_LOG_LIMIT=100
python -m modal run --detach .\hrm-cloud\residual_tasklora_v2.py
```

## Bottom Line

The latest discrete experiment is `residual_tasklora_v2`, and it was indeed left in a partial state after the Modal/billing/nonfinite interruption. The best completed result before that is multitask HRM average-base, not the specialist TaskLoRA adapters. Residual TaskLoRA v2 has a small positive pre-interruption signal, but it needs a clean rerun with bounded predictions and explicit nonfinite accounting before it should be cited as a final result.
