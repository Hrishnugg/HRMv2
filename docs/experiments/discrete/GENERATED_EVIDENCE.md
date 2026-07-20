# Discrete Generated Evidence Catalog

The authored [discrete compendium](learned-heuristic/results/EXPERIMENT_RESULTS_COMPENDIUM.md) is the preferred synthesis. The generated survey documents below remain in `modal_downloads/` beside the JSON they enumerate.

| Generated document | Role and result |
|---|---|
| [`full_survey_sdk_parallel/summary.md`](../../../modal_downloads/full_survey_sdk_parallel/summary.md) | Main Modal survey: 13,671 JSON files parsed across 11 experiment volumes with no download errors. It contains normalized tables used by the compendium. |
| [`residual_latest_20260601/summary.md`](../../../modal_downloads/residual_latest_20260601/summary.md) | Focused Residual TaskLoRA snapshot: 2,028 files, partial/incomplete run evidence, including aggregate rows and alpha files. |
| [`full_survey_manifest_check/summary.md`](../../../modal_downloads/full_survey_manifest_check/summary.md) | Empty manifest-only diagnostic (0 downloaded/parsed files); retained as survey provenance, not result evidence. |
| [`residual_check_live_manifest/summary.md`](../../../modal_downloads/residual_check_live_manifest/summary.md) | Empty live-manifest diagnostic (0 files); provenance only. |
| [`residual_check_live_manifest_2/summary.md`](../../../modal_downloads/residual_check_live_manifest_2/summary.md) | Second empty live-manifest diagnostic (0 files); provenance only. |

## Raw result entry points

- Model-comparison and early transfer snapshots: [`modal_downloads/survey_results/`](../../../modal_downloads/survey_results/).
- Completed clean-transfer result: [`clean_v3_results/final_results__A64_moderateDyn.json`](../../../modal_downloads/clean_v3_results/final_results__A64_moderateDyn.json).
- Completed multitask TaskLoRA result: [`multitask_results/final_results__multitask_tasklora.json`](../../../modal_downloads/multitask_results/final_results__multitask_tasklora.json).
- Residual aggregate snapshot: [`eval_agg_dir/eval_agg/`](../../../modal_downloads/eval_agg_dir/eval_agg/).

These JSON files are evidence artifacts rather than narrative documentation. Their statuses, matched-baseline caveats, and findings are summarized in the compendium and focal-redesign report.
