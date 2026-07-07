# C5 Hard Maps + Rich Obstacle Encoder

## Goal

Find a continuous PRM regime where Euclidean A* is genuinely challenged, then test whether HRM and ONLSTM learned residual heuristics produce statistically significant gains.

Success target:

- Euclidean baseline success should sit around 50-70% at at least one low/medium expansion budget.
- HRM/ONLSTM should beat Euclidean at the same budget with enough evaluation worlds for a defensible proportion test.
- The hard maps must remain connected often enough that failures are search-budget failures, not dataset-generation failures.

## Design

This experiment is implemented as a separate script:

```text
hrm-cloud/continuous_prm/continuous_prm_c5_hard_obstacle_encoder.py
```

The script leaves C1-C4 unchanged. It installs runtime-only extensions over `continuous_prm_common` before delegating to the existing staged runner.

### Hard Suites

- `C_hard_maze`: unit-square staggered-wall maze with three alternating gates and circle clutter.
- `C_hard_maze_dense`: denser unit-square OOD version with four walls, tighter gates, and more clutter.
- `C_hard_rooms`: larger two-unit maze/room layout with five walls and more clutter.

Starts and goals are sampled from opposite side bands, forcing long detours through alternating gates.

### Richer Encoder

The C5 feature path keeps the old feature shape contract but repurposes part of the ray-token budget:

- nearest obstacle tokens: default 12
- exact ray tokens: `num_rays - sector_tokens`
- angular obstacle-sector summary tokens: default 16
- task descriptor token

Sector tokens summarize obstacle count, min/mean clearance, mean radius, and rectangle fraction in angular bins around each roadmap node.

## Calibration Results So Far

### Probe A: `96 nodes / k=5`

Command:

```bash
python continuous_prm_c5_hard_obstacle_encoder.py --stage c1 --mode full --out-dir runs/smoke_c5_hard_baseline --eval-worlds 5 --roadmap-nodes 96 --roadmap-k 5 --budgets 8,16,32,64 --nearest-obstacles 8 --num-rays 24 --sector-tokens 8 --cpu --torch-threads 1
```

Result:

- `C_hard_maze`: 5/5 valid worlds; Euclidean success 0.0 at budgets 8, 16, 32 and 0.2 at budget 64.
- `C_hard_maze_dense`: 0/5 valid connected worlds.
- `C_hard_rooms`: 0/5 valid connected worlds.

Interpretation: too sparse. Good as a stress floor, but not suitable for the main learned-heuristic comparison because connected-world yield is too low.

### Probe B: `192 nodes / k=7`

Command:

```bash
python continuous_prm_c5_hard_obstacle_encoder.py --stage c1 --mode full --out-dir runs/c5_calib_192k7_bands --eval-worlds 20 --roadmap-nodes 192 --roadmap-k 7 --budgets 120,136,152,168,184,200 --nearest-obstacles 8 --num-rays 24 --sector-tokens 8 --cpu --torch-threads 1
```

Result:

| Suite | Valid Worlds | Euclidean @120 | @136 | @152 | @168 |
|---|---:|---:|---:|---:|---:|
| `C_hard_maze` | 20/20 | 0.10 | 0.35 | 0.90 | 1.00 |
| `C_hard_maze_dense` | 20/20 | 0.00 | 0.10 | 0.65 | 1.00 |
| `C_hard_rooms` | 20/20 | 0.00 | 0.30 | 0.90 | 1.00 |

Interpretation: `192/k=7` is the first good calibration point. Budget 136-152 is the transition band. `C_hard_maze_dense` at budget 152 hits the desired 50-70% baseline target directly.

## Round 1 Modal Run

Command:

```bash
python -m modal run continuous_prm_modal.py::run_c5_hard --run-name continuous_prm_c5_hard_r1 --train-worlds 160 --nodes-per-world 192 --eval-worlds 80 --roadmap-nodes 192 --roadmap-k 7 --base-epochs 12 --expert-epochs 10 --backbones hrm,onlstm --train-tasks C_hard_maze,C_hard_rooms --eval-suites C_hard_maze,C_hard_maze_dense,C_hard_rooms --budgets 128,136,144,152,168 --roadmap-shard-worlds 20 --eval-shard-worlds 1 --nearest-obstacles 12 --num-rays 48 --ray-steps 96 --sector-tokens 16
```

Purpose:

- Train HRM and ONLSTM avgbase models on hard anchors.
- Train bounded TaskLoRA experts for both hard training anchors.
- Evaluate Euclidean, avgbase, TaskLoRA, and oracle TaskLoRA on 80 worlds per suite.
- Compute success deltas and statistical tests against Euclidean.

Status: completed.

Modal run:

- `continuous_prm_c5_hard_r1`
- Modal URL: `https://modal.com/apps/synoptica/main/ap-qeQWNtxg2XCOG7lstV3OMS`
- Local artifacts:
  - `runs/continuous_prm_c5_hard_r1/results/continuous_prm_eval_summary.csv`
  - `runs/continuous_prm_c5_hard_r1/results/continuous_prm_eval_raw.csv`
  - `runs/continuous_prm_c5_hard_r1/results/continuous_prm_c5_significance.csv`
  - `runs/continuous_prm_c5_hard_r1/results/continuous_prm_c5_significance.md`

Training outcome:

- ONLSTM avgbase trained cleanly: epoch 12 loss `0.02159`, MAE `0.14594`.
- HRM avgbase failed by saturating at the residual cap: epoch 12 loss `2.61219`, MAE `3.11211`, eval `delta_mean_mean=4.0`.
- HRM TaskLoRA experts did not move: `correction_abs_mean=0.0`; deployable HRM rows matched Euclidean exactly.

### Round 1 Results

The hard-map calibration worked. At budget 144, Euclidean was in the desired target band for two suites:

| Suite | Episodes | Euclidean | ONLSTM avgbase | Delta | Paired gains | McNemar p | BH q | Expansion delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `C_hard_maze` | 80 | 0.525 | 1.000 | +0.475 | 38 | `7.28e-12` | `4.34e-11` | -66.987 |
| `C_hard_maze_dense` | 79 | 0.595 | 0.962 | +0.367 | 29 | `3.73e-09` | `1.92e-08` | -24.519 |

Deployable ONLSTM TaskLoRA variants also passed the claim filter:

| Suite | Budget | Method | Success | Delta | Paired gains | McNemar p | BH q | Expansion delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `C_hard_maze` | 144 | `tasklora/onlstm/C_hard_maze` | 1.000 | +0.475 | 38 | `7.28e-12` | `4.34e-11` | -63.987 |
| `C_hard_maze` | 144 | `tasklora/onlstm/C_hard_rooms` | 1.000 | +0.475 | 38 | `7.28e-12` | `4.34e-11` | -64.625 |
| `C_hard_maze_dense` | 144 | `tasklora/onlstm/C_hard_rooms` | 0.937 | +0.342 | 27 | `1.49e-08` | `7.00e-08` | -21.671 |
| `C_hard_maze_dense` | 144 | `tasklora/onlstm/C_hard_maze` | 0.899 | +0.304 | 24 | `1.19e-07` | `5.28e-07` | -20.608 |

`C_hard_rooms` was harder than the preferred 50-70% baseline band at budgets 128-144 and easier by budget 152, but ONLSTM still showed a strong practical signal:

| Budget | Euclidean | ONLSTM avgbase | Delta | Expansion delta |
|---:|---:|---:|---:|---:|
| 128 | 0.052 | 0.922 | +0.870 | -36.039 |
| 136 | 0.195 | 0.961 | +0.766 | -42.610 |
| 144 | 0.442 | 0.987 | +0.545 | -47.961 |
| 152 | 0.818 | 1.000 | +0.182 | -50.818 |

Claim filter used by `continuous_prm_c5_analyze.py`:

- Compare each method against Euclidean on the same suite and budget.
- Prefer paired McNemar tests when raw rows are available.
- Apply Benjamini-Hochberg correction across all method-vs-Euclidean comparisons.
- Claim candidates must be deployable, full-episode comparisons; `oracle_tasklora` is retained only as a diagnostic.
- Require Euclidean in the 0.50-0.70 target band, at least +0.10 absolute success, and BH q <= 0.05.

Conclusion: C5 produces statistically significant, practically large ONLSTM improvements over Euclidean on genuinely hard maps. It does not yet support an HRM improvement claim.

## HRM Follow-ups

### HRM Tune A: lower LR, smaller HRM, residual cap 2

Command:

```bash
python -m modal run continuous_prm_modal.py::run_c5_hard --run-name continuous_prm_c5_hard_hrm_tune_a --train-worlds 160 --nodes-per-world 192 --eval-worlds 80 --roadmap-nodes 192 --roadmap-k 7 --base-epochs 14 --expert-epochs 8 --backbones hrm --train-tasks C_hard_maze,C_hard_rooms --eval-suites C_hard_maze,C_hard_maze_dense,C_hard_rooms --budgets 128,136,144,152,168 --roadmap-shard-worlds 20 --eval-shard-worlds 1 --nearest-obstacles 12 --num-rays 48 --ray-steps 96 --sector-tokens 16 --lr 0.00005 --expert-lr 0.00005 --grad-clip 0.25 --max-norm-residual 2.0 --hrm-hidden 96 --hrm-layers 1 --hrm-k-step 1 --head-hidden 128
```

Artifacts:

- `runs/continuous_prm_c5_hard_hrm_tune_a/results/continuous_prm_eval_summary.csv`
- `runs/continuous_prm_c5_hard_hrm_tune_a/results/continuous_prm_c5_significance.md`

Result:

- No deployable claim candidates.
- At budget 144, HRM avgbase and TaskLoRA exactly matched Euclidean on all three suites.
- Diagnostics showed a constant residual cap: `delta_mean_mean=2.0`, `correction_abs_mean=0.0`.

### HRM Soft-Cap A: differentiable residual cap

Code change:

- `continuous_prm_c5_hard_obstacle_encoder.py` now supports `--soft-residual-cap`.
- `continuous_prm_modal.py::run_c5_hard` now supports `--soft-residual-cap` and explicit HRM architecture knobs.

Command:

```bash
python -m modal run continuous_prm_modal.py::run_c5_hard --run-name continuous_prm_c5_hard_hrm_softcap_a --train-worlds 160 --nodes-per-world 192 --eval-worlds 80 --roadmap-nodes 192 --roadmap-k 7 --base-epochs 14 --expert-epochs 8 --backbones hrm --train-tasks C_hard_maze,C_hard_rooms --eval-suites C_hard_maze,C_hard_maze_dense,C_hard_rooms --budgets 128,136,144,152,168 --roadmap-shard-worlds 20 --eval-shard-worlds 1 --nearest-obstacles 12 --num-rays 48 --ray-steps 96 --sector-tokens 16 --soft-residual-cap --lr 0.0002 --expert-lr 0.00015 --grad-clip 1.0 --max-norm-residual 4.0 --hrm-hidden 96 --hrm-layers 1 --hrm-k-step 1 --head-hidden 128
```

Artifacts:

- `runs/continuous_prm_c5_hard_hrm_softcap_a/results/continuous_prm_eval_summary.csv`
- `runs/continuous_prm_c5_hard_hrm_softcap_a/results/continuous_prm_c5_significance.md`

Result:

- No deployable claim candidates.
- Full-scale training still jumped to the high-cap plateau: avgbase epoch 14 loss `2.61234`, MAE `3.11226`.
- Expert corrections remained zero: `correction_abs_mean=0.0`, `B=4.0000`.
- At budget 144, HRM avgbase and TaskLoRA again matched Euclidean exactly on all three suites.

Interpretation:

- HRM is not being fairly expressed by the current C5 residual-regression head/training recipe.
- The failure is architectural or optimization related, not a map-difficulty issue: the same data gives strong ONLSTM improvements.
- The most suspicious interaction is the HRM block plus nonnegative residual head collapsing to a constant additive heuristic. A constant positive residual does not change A* ordering, so success and expansions match Euclidean.

## Modal Execution Notes

The HRM soft-cap run had repeated Modal worker preemptions during `C_hard_rooms` collection. Because each collection shard requested 20 connected worlds, preempted workers lost substantial partial progress and restarted from `1/20`.

Next engineering fix for hard-room experiments:

- Use smaller `roadmap_shard_worlds` for C5 hard rooms, likely 5 or 10 instead of 20.
- Prefer shard-level progress checkpointing if we keep long collection shards.
- Consider non-preemptible Modal functions only for confirmed final runs, since they cost more.

## Statistical Plan

Primary tests:

- Compare method vs Euclidean success at the same suite and budget.
- Use a two-proportion z-test for quick screening.
- Report Wilson confidence intervals for success rates.
- Treat the strongest claim as valid only if the improvement is both practically meaningful and statistically significant.

Preferred target:

- Budget where Euclidean is 0.50-0.70 and learned heuristic is at least 0.10 absolute success points higher.
- If available, also require lower mean expansions among solved/all episodes.

## Follow-up Knobs

If Round 1 is too easy:

- Lower budgets to 120-144.
- Increase wall count or reduce gate width for `C_hard_maze_dense`.
- Reduce `roadmap_k` from 7 to 6 while keeping connected-world yield acceptable.

If Round 1 is too hard:

- Raise budget band to 144-176.
- Keep `192/k=7` but reduce dense clutter.
- Train on `C_hard_maze_dense` as a third anchor instead of OOD-only.

If models do not improve:

- Add a learned map-level encoder/router, not only local sector tokens.
- Add supervised labels for bottleneck/gate direction.
- Add curriculum training over budget-hard worlds rather than random connected worlds.
