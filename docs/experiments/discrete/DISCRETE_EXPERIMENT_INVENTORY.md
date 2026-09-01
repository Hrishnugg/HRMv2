# Discrete Experiment Inventory (Complete)

**Date:** 2026-07-23
**Purpose:** the exhaustive map of every discrete-grid experiment family in the repository — scripts, Modal volumes, raw artifacts, documentation, and coverage status in the [master synthesis](../MASTER_EXPERIMENT_SYNTHESIS.md) and the AAAI-27 submission. The [results compendium](learned-heuristic/results/EXPERIMENT_RESULTS_COMPENDIUM.md) remains the canonical *chronology with numbers*; this inventory is the *completeness map* over it, built from the full Modal survey (`modal_downloads/full_survey_sdk_parallel/`, 13,671 JSONs parsed across 11 volumes with zero download errors, generated 2026-06-01), the `hrm-cloud/` script tree, and the local snapshot directories.

## Scale at a glance

| Layer | Count |
| --- | --- |
| Experiment families | 15 |
| Cloud experiment/training scripts in `hrm-cloud/` | 31 (plus survey/analysis tooling) |
| Modal volumes surveyed with JSON evidence | 11 (+ early model/checkpoint volumes, 2025-11-28→12-07) |
| JSON result files parsed by the full survey | 13,671 (15,267 manifest entries) |
| Curated headline result files in `modal_downloads/survey_results/` | 8 |
| Local snapshot directories under `modal_downloads/` | 12 (incl. the 222-file clean pre-interruption residual snapshot in `eval_agg_dir/`) |
| Local focal-pilot raw logs in `hrm-cloud/` | 5 (`bench_*.log`) |

## Family-by-family inventory

| # | Family | Scripts | Volumes / artifacts | Documentation | Synthesis / paper coverage |
| --- | --- | --- | --- | --- | --- |
| 1 | **Early HRM world-model scaling lineage** (2025-11-28→12-07): small → mid → full-scale → robust-fix → split → boosted → 8-GPU (×2 variants) obstacle-trajectory forecasters + receding-horizon space-time A* | `hrm_cloud.py`, `hrm_cloudv2.py`, `hrm_cloudv3.py`, `hrm_cloudMid.py`, `hrm_cloudBoosted.py`, `hrm_cloudFullScale.py`, `hrm_cloudFullScaleRobustFix.py`, `hrm_cloudFullScaleSplit.py`, `hrm_cloud_8gpu.py`, `hrm_cloud_8gpu_v2.py` | early model volumes (`.pt`/`.zip`/`.npz`; no JSON rows) | [BENCHMARK_RESULTS.md](dynamic-world-model/results/BENCHMARK_RESULTS.md) digests ~10 training runs into 4 headline rows (52%/62%/68% vs LSTM 66%; oracle 98%) | Synthesis: digest only. Paper: one forecasting row (appendix). Per-variant runs are chronology, not independent claims. |
| 2 | **Diffusion planner baselines** v1/v2 (end-to-end path generators, static-map input) | `diffusion_cloud.py` | early volumes; digest rows | BENCHMARK_RESULTS.md (v1 ≈64%, v2 60% despite 8× params/2× data) | Synthesis: yes. Paper: omitted (information-set not matched to recurrent arms; noted here as the end-to-end contrast). |
| 3 | **LSTM vs HRM matched comparison** (100 shared episodes, 7 tiers) | `lstm_hrm_comparison.py` | `lstm-hrm-comparison-vol` (1 JSON); `survey_results/lstm_hrm_comparison_results.json` | [LSTM_VS_HRM_EXPERIMENT.md](dynamic-world-model/results/LSTM_VS_HRM_EXPERIMENT.md) | Synthesis + paper: yes (71/100 vs 67–69/100, descriptive). |
| 4 | **ON-LSTM vs HRM Preset M, v1 and v2** | `onlstm_hrm_comparison.py`, `onlstm_hrm_comparison_presetm_v2.py` | `onlstm-hrm-comparison-presetm-vol` (v1, superseded), `-presetm-v2-vol`; both `survey_results` JSONs | Compendium (v1: ON-LSTM 0.15–0.20 vs HRM 0.08–0.10; v2 canonical: 0.388 vs 0.265); [design](dynamic-world-model/design/ONLSTM_VS_HRM_EXPERIMENT_PRESETM_V2.md) | Synthesis + paper: v2 only. **v1 recorded here as the superseded harder-calibration predecessor.** |
| 5 | **Transfer RL zero-shot** (first transfer-first learned-A* run; α fixed 1.0) | `transfer_astar_heuristic_rl.py` | `transfer-astar-heuristic-rl-vol`; `survey_results/transfer_rl_eval_zero_shot.json` | Compendium | Synthesis: yes. Paper: folded into the additive-near-null narrative. |
| 6 | **Imitation v2 lineage** (supervised residual imitation; "search graveyard" data) — original, fixpack, map-scale LoRA, **and an empty-model baseline rerun control** | 6 script variants (`transfer_astar_heuristic_imitation_v2_fixed*.py`, incl. `LORA_PATCHED`, `fresh_volume_patched`, `poll_and_runtag_fixed`) + `transfer_astar_heuristic_imitation_mapscale_lora.py` | `transfer-astar-heuristic-imitation-v2-vol` (200), `-vol-v2` (1,553); 4 `survey_results` JSONs incl. `imitation_v2_empty_model_rerun_results` | [experiment_writeup_last_two_runs.md](learned-heuristic/results/experiment_writeup_last_two_runs.md) + compendium | Synthesis: original + map-scale. **Empty-model rerun (baseline sanity control: static A* re-verified through the learned-model code path) recorded here.** Paper: folded. |
| 7 | **Clean transfer parallel v1/v2/v3** (the controlled rebuild; full-FT × LoRA × 2 backbones; α tuned) | `transfer_astar_heuristic_clean_parallel.py`, `_fixed.py` | v1 (checkpoints only, training partial), v2 (52 files, 20-episode smoke), **v3 (3,983 files, canonical completed negative)** | [Blueprint](learned-heuristic/design/clean_transfer_experiment_blueprint.md) + compendium | Synthesis + paper: v3. v1/v2 are partial/smoke chronology, recorded here. |
| 8 | **CondLoRA basis v1** (pooled base + hypernetwork-routed LoRA basis) | `transfer_astar_heuristic_avg_condlora_basis_v1.py` | `transfer-astar-heuristic-avg-condlora-basis-v1-vol` (2,417 files; 329 aggregates; no final file); `condlora_eval_agg_dir/` snapshot | Compendium (incomplete, unfavorable) | Synthesis: appendix-grade. Paper: omitted by design (incomplete). |
| 9 | **Multitask TaskLoRA v1** (pooled `avgbase` + per-task experts) | `multitask_astar_heuristic_tasklora.py` | `multitask-astar-heuristic-tasklora-v1-vol` (3,438 files; final file present); `pooled_manifest__ALL_TASKS.json`; `alphas_dir/`, `eval_agg_dir/` | Compendium (canonical: `avgbase__hrm` +2.11pp; only A32 expert beats subset baseline) | Synthesis + paper: yes. |
| 10 | **Residual TaskLoRA v2** (bounded tanh-corrected experts around frozen base; nonfinite incident; interrupted) | `residual_tasklora_v2.py` | `residual-tasklora-v2-vol` (2,024→2,076 files; 227 aggregates; **no final file**); snapshots: `eval_agg_dir/` (222-file clean pre-interruption), `residual_latest_20260601/`, `residual_tasklora_v2_alphas/`, `residual_tasklora_v2_eval_agg/`, 2 live-recheck manifests | Compendium + [focal redesign](learned-heuristic/results/EXPERIMENT_RESULTS_FOCAL_REDESIGN.md) (clean local re-evaluation supersedes) | Synthesis + paper: via the clean re-evaluation and the experts≈base focal finding. |
| 11 | **Learned focal search** (local RTX 5090 pilot; ranker/magnitude diagnosis; w∈{1.0,1.05,1.1}) | `bench_focal.py`, `bench_eval_episode.py` | local logs `bench_run.log`, `bench_run_dyn.log`, `bench_onlstm.log`, `bench_expert.log`, `bench_expert_a192.log`, `bench_onlstm_expert.log` | [FOCAL_SEARCH_RESULTS.md](learned-heuristic/results/FOCAL_SEARCH_RESULTS.md) + [redesign report](learned-heuristic/results/EXPERIMENT_RESULTS_FOCAL_REDESIGN.md) | Synthesis + paper: yes (6–15% at w=1.0). |
| 12 | **Budget-invariance analysis** (per (model,suite): does success improve B500→B2000?) | `analyze_budget_invariance.py` | consumes `eval_agg` mirrors | script docstring only | **Previously uncited as an artifact.** Its conclusion ("budget increases add work, not success, on hard families") appears in the synthesis; the analysis tool is recorded here. |
| 13 | **Evaluation acceleration** (~39× representative speedup; env-forwarding fix) | eval-path changes + `FAST_EVAL.md` | — | [FAST_EVAL.md](learned-heuristic/operations/FAST_EVAL.md) + [plan](learned-heuristic/plans/2026-06-15-residual-tasklora-eval-speedup.md) | Synthesis: engineering result. |
| 14 | **HRM-v2 direct maze solver** (pre-fix run, fidelity audit, mechanism retrain) | `HRM-v2/` tree | HRM-v2 datasets/checkpoints | [direct-solver docs](direct-solver/) (audit, results, history ×5) | Synthesis + paper (forensics): yes. |
| 15 | **Modal survey infrastructure** (the provenance layer itself) | `modal_experiment_survey.py` | 5 survey snapshots (`full_survey`, `full_survey_raw`, `full_survey_sdk`, `full_survey_sdk_parallel` ← canonical, `full_survey_manifest_check`) + 2 residual live-recheck snapshots; `test_dirget/` | [GENERATED_EVIDENCE.md](GENERATED_EVIDENCE.md) + compendium Evidence section | Synthesis: yes (5 generated survey reports). |

## What this inventory adds beyond the compendium digest

1. **The early scaling lineage is ~10 distinct training runs**, not four: v1→v2→v3, Mid, FullScale (+RobustFix, +Split), Boosted, and two 8-GPU variants. The digest rows are the survivors; the variant scripts document the failed/superseded attempts (robustness fixes, data splits) that the headline numbers absorbed.
2. **Preset M v1** exists as a completed-but-superseded run (both architectures near-floor at 0.08–0.20 success) — the calibration failure that motivated Preset M+ v2, and an early instance of the program's calibrate-before-comparing lesson.
3. **An empty-model baseline rerun control** in the imitation lineage: static A* re-verified through the learned-model code path, guarding against harness bugs masquerading as model effects — an early ancestor of the later twin/control discipline.
4. **Clean transfer v1/v2 partials**: a training-only volume and a 20-episode smoke volume that preceded the canonical v3 — chronology evidence that the "clean negative" was itself run three times before being trusted.
5. **The budget-invariance analysis** as a named artifact behind the "more search does not rescue hard families" claim.
6. **Six imitation script iterations** (patches for LoRA wiring, volume freshness, polling/run-tags) — engineering-history context for why later families standardized on sharded, cacheable, manifest-driven evaluation.

None of these change any synthesis or paper claim: every completed headline number was already sourced from the canonical volumes above. The additions are chronology, controls, superseded predecessors, and tooling — recorded so the discrete program's true scale (15 families, ~31 scripts, 13,671 surveyed result files) is visible and auditable from one document.

## Raw-evidence entry points (discrete, complete)

| Artifact class | Location |
| --- | --- |
| Full survey (canonical) | `modal_downloads/full_survey_sdk_parallel/` (`summary.md` 129,604 lines; `manifest.json`; per-volume trees) |
| Earlier survey snapshots | `modal_downloads/full_survey{,_raw,_sdk,_manifest_check}/` |
| Curated headline JSONs | `modal_downloads/survey_results/` (8 files) |
| Clean pre-interruption residual snapshot | `modal_downloads/eval_agg_dir/` (222 aggregates) |
| Post-incident residual snapshots | `modal_downloads/residual_latest_20260601/`, `residual_tasklora_v2_*/`, `residual_check_live_*/` |
| CondLoRA aggregates | `modal_downloads/condlora_eval_agg_dir/` |
| Clean/multitask finals | `modal_downloads/clean_v3_results/`, `modal_downloads/multitask_results/` |
| Local focal logs | `hrm-cloud/bench_*.log` |
