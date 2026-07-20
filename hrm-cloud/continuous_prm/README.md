# Continuous PRM Heuristic Learning

This folder contains the staged continuous-space PRM experiment suite for the
HRM/ONLSTM learned A* heuristic direction.

The complete stage history, findings, designs, plans, and generated-evidence
catalog are in the [continuous experiment documentation](../../docs/experiments/continuous/README.md).

## Layout

| File | Role |
| --- | --- |
| `continuous_prm_common.py` | Continuous world generation, PRM construction, labels, A* evaluation, models, LoRA, training, and evaluation helpers. |
| `continuous_prm_stage_runner.py` | Unified CLI for `--stage c1`, `c2`, `c3`, `c4`, or `full`. |
| `continuous_prm_c1_baseline.py` | Thin wrapper for C1 PRM/Euclidean baseline validation. |
| `continuous_prm_c2_avgbase.py` | Thin wrapper for C2 pooled avgbase HRM/ONLSTM training and evaluation. |
| `continuous_prm_c3_residual_tasklora_experts.py` | Thin wrapper for C3 bounded residual task-LoRA expert training and diagnostics. |
| `continuous_prm_c4_rbf_mixture.py` | Thin wrapper for C4 nearest/RBF descriptor-conditioned expert mixture evaluation. |
| `continuous_prm_c13_state_heuristic.py` | C13 semantics, leak-resistant local-state target collection, bounded-backup target selection, and roadmap-density auditing. |
| `continuous_prm_c13_td_ranker.py` | C13-B fresh-start local-rollout collection, HRM/ON-LSTM regression, and Euclidean-anchored FOCAL evaluation with same-search controls. |
| `continuous_prm_c13_identifiability.py` | Multi-angle target reliability, representation/readout, learning-curve, FOCAL-width, exact-target, and primary-A* diagnostic study. |
| `continuous_prm_c13_certified_search.py` | C13-C learned-incumbent plus fresh Euclidean-A* certification gate using the frozen six-world audit. |
| `continuous_prm_c13_shared_queue.py` | C13-D shared-state Euclidean-anchor/oracle-rank integration gate with direct bound certification. |
| `continuous_prm_c13_shared_queue_target.py` | C13-E exact C13-B rollout rank under the frozen C13-D shared search and five-of-six target gate. |
| `continuous_prm_c13_shared_queue_calibration.py` | C13-F same-search Euclidean and fixed residual-scale calibration controls. |
| `continuous_prm_c13_local_escape.py` | C13-G exact radius-bounded local-escape ceiling. |
| `continuous_prm_c13_local_escape_exit_stub.py` | C13-G exit-value variant of the local-escape ceiling. |
| `continuous_prm_c13_lhbl_generated_v3.py` | C13-H local heuristic Bellman learning, training, and bounded development gate. |
| `continuous_prm_c13_lhbl_c7_comparison.py` | C13-I live nine-provider comparison on all 144 C7 worlds. |
| `continuous_prm_c13_lhbl_multisuite.py` | C13-J suite-balanced current-state training and development comparison. |
| `continuous_prm_c13_local_bellman_integration.py` | C13-K one-step local-Bellman inference integration. |
| `continuous_prm_c13_local_backup_scale.py` | C13-L fresh six-suite residual-scale calibration. |
| `continuous_prm_c13_reopening_rank_probe.py` | Post-hoc direct/reopening/FOCAL mechanism probe. |
| `continuous_prm_c13_matched_quality_confirmation.py` | C13-M untouched 144-world matched-quality confirmation with live C7 providers. |
| `continuous_prm_c13_hrm_substitution.py` | C13-N frozen flat-to-trimmed-HRM architecture substitution and gated development comparison. |
| `continuous_prm_c13_hrm_alignment.py` | C13-O summary-last readout alignment with frozen trimmed/flat controls and confirmation-only-after-pass. |
| [C13-B rollout-ranker smoke](../../docs/experiments/continuous/c13/results/C13B_ROLLOUT_RANKER_SMOKE.md) | Provenance contract, smoke metrics, negative learned-signal verdict, and next identifiability gate. |
| [C13-B identifiability study](../../docs/experiments/continuous/c13/results/C13B_IDENTIFIABILITY_STUDY.md) | Causal classification of integration, target, representation, and missing-information failure modes. |
| [C13-C certified search gate](../../docs/experiments/continuous/c13/results/C13C_CERTIFIED_SEARCH.md) | Negative integration result: even an oracle incumbent cannot repay the duplicated fresh-certifier work at `w=1.10`. |
| [C13-D shared-queue oracle gate](../../docs/experiments/continuous/c13/results/C13D_SHARED_QUEUE_ORACLE.md) | Positive oracle integration ceiling: the shared search beats matched FOCAL on all six primary comparisons. |
| [C13-E shared-queue exact-target gate](../../docs/experiments/continuous/c13/results/C13E_SHARED_QUEUE_EXACT_TARGET.md) | Negative target/calibration result: all paths certify, but exact rollout loses the primary five-of-six expansion gate. |
| [C13-F through C13-M current-state result](../../docs/experiments/continuous/c13/results/C13F_M_CURRENT_STATE_RESULT.md) | Canonical positive result and full mechanism history: 15.95% fewer expansions than complete-map field HRM, with claim and runtime caveats. |
| [C13-N HRM substitution](../../docs/experiments/continuous/c13/results/C13N_HRM_SUBSTITUTION_RESULT.md) | Trimmed HRM retains pooled signal but fails suite breadth and matched-flat path-quality gates. |
| [C13-O HRM alignment](../../docs/experiments/continuous/c13/results/C13O_HRM_ALIGNMENT_RESULT.md) | Summary-last helps at iteration 6 but does not produce a robust or endpoint readout win; no cell authorizes confirmation. |
| [C13 design and initial audit](../../docs/experiments/continuous/c13/results/C13_INITIAL_AUDIT.md) | Current-state constraint, target gate, preliminary depth curve, and +10% density result. |
| [`continuous_prm_experiment_ladder_repo_coupled.md`](../../docs/experiments/continuous/c01-c04/continuous_prm_experiment_ladder_repo_coupled.md) | Detailed experiment ladder, run commands, metrics, and reasoning. |

## Smoke Check

```bash
python -m py_compile \
  continuous_prm_common.py \
  continuous_prm_stage_runner.py \
  continuous_prm_c1_baseline.py \
  continuous_prm_c2_avgbase.py \
  continuous_prm_c3_residual_tasklora_experts.py \
  continuous_prm_c4_rbf_mixture.py \
  continuous_prm_c13_state_heuristic.py \
  continuous_prm_c13_td_ranker.py \
  continuous_prm_c13_identifiability.py \
  continuous_prm_c13_certified_search.py \
  continuous_prm_c13_shared_queue.py \
  continuous_prm_c13_shared_queue_target.py \
  continuous_prm_c13_shared_queue_calibration.py \
  continuous_prm_c13_local_escape.py \
  continuous_prm_c13_local_escape_exit_stub.py \
  continuous_prm_c13_lhbl_generated_v3.py \
  continuous_prm_c13_lhbl_c7_comparison.py \
  continuous_prm_c13_lhbl_multisuite.py \
  continuous_prm_c13_local_bellman_integration.py \
  continuous_prm_c13_local_backup_scale.py \
  continuous_prm_c13_reopening_rank_probe.py \
  continuous_prm_c13_matched_quality_confirmation.py \
  continuous_prm_c13_hrm_substitution.py \
  continuous_prm_c13_hrm_alignment.py
```

Run from this directory so the thin wrappers can import
`continuous_prm_stage_runner.py` directly.

## C13 Focused Verification

```bash
python -m pytest tests/test_c13_state_heuristic.py tests/test_c13_td_ranker.py tests/test_c13_identifiability.py tests/test_c13_certified_search.py tests/test_c13_shared_queue.py tests/test_c13_shared_queue_target.py -q
python continuous_prm_c13_state_heuristic.py \
  --mode relaxation --eval-suites C_hard_maze --eval-worlds 30 \
  --train-nodes 192 --roadmap-k 7 --backup-depths 0,1,2,4,8,16
python continuous_prm_c13_td_ranker.py --mode full --smoke-test --out-dir runs/c13_td_smoke
python continuous_prm_c13_identifiability.py --out-dir runs/c13_identifiability
python continuous_prm_c13_certified_search.py --study-dir runs/c13_identifiability --out-dir runs/c13_certified_search
python continuous_prm_c13_shared_queue.py --study-dir runs/c13_identifiability --independent-dir runs/c13_certified_search --out-dir runs/c13_shared_queue_oracle
python continuous_prm_c13_shared_queue_target.py --study-dir runs/c13_identifiability --independent-dir runs/c13_certified_search --oracle-dir runs/c13_shared_queue_oracle --out-dir runs/c13_shared_queue_rollout
python -m pytest tests/test_c13_lhbl_c7_comparison.py tests/test_c13_lhbl_multisuite.py tests/test_c13_local_bellman_integration.py tests/test_c13_local_backup_scale.py tests/test_c13_matched_quality_confirmation.py -q
python -m pytest tests/test_c13_hrm_substitution.py tests/test_c13_hrm_alignment.py -q
python continuous_prm_c13_matched_quality_confirmation.py
```
