# Continuous PRM Heuristic Learning

This folder contains the staged continuous-space PRM experiment suite for the
HRM/ONLSTM learned A* heuristic direction.

## Layout

| File | Role |
| --- | --- |
| `continuous_prm_common.py` | Continuous world generation, PRM construction, labels, A* evaluation, models, LoRA, training, and evaluation helpers. |
| `continuous_prm_stage_runner.py` | Unified CLI for `--stage c1`, `c2`, `c3`, `c4`, or `full`. |
| `continuous_prm_c1_baseline.py` | Thin wrapper for C1 PRM/Euclidean baseline validation. |
| `continuous_prm_c2_avgbase.py` | Thin wrapper for C2 pooled avgbase HRM/ONLSTM training and evaluation. |
| `continuous_prm_c3_residual_tasklora_experts.py` | Thin wrapper for C3 bounded residual task-LoRA expert training and diagnostics. |
| `continuous_prm_c4_rbf_mixture.py` | Thin wrapper for C4 nearest/RBF descriptor-conditioned expert mixture evaluation. |
| `continuous_prm_experiment_ladder_repo_coupled.md` | Detailed experiment ladder, run commands, metrics, and reasoning. |

## Smoke Check

```bash
python -m py_compile \
  continuous_prm_common.py \
  continuous_prm_stage_runner.py \
  continuous_prm_c1_baseline.py \
  continuous_prm_c2_avgbase.py \
  continuous_prm_c3_residual_tasklora_experts.py \
  continuous_prm_c4_rbf_mixture.py
```

Run from this directory so the thin wrappers can import
`continuous_prm_stage_runner.py` directly.
