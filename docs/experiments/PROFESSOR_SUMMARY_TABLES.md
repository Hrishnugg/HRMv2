# Planning Experiment Summary Tables

**Draft:** 2026-07-13  
**Audience:** professor meeting / paper planning  
**Scope:** current repository evidence, including results completed after the 2026-07-10 meeting

## Technical summary

The two tables below are a first publication-oriented snapshot of the discrete and continuous planning programs. The strongest cross-program result is not a general HRM advantage: learned guidance can reduce search work, but the outcome depends primarily on representation, planner integration, and adaptation regime. ON-LSTM, HRM, and U-Net each lead in different cells.

The tables intentionally preserve negative results and evidence status. They do **not** average incompatible protocols together. Cells use:

- `S`: success rate.
- `E`: mean A* expansions.
- `R`: median matched-solved expansion ratio versus the algorithmic baseline; lower is better.
- `ΔE`: mean expansion difference versus the baseline; negative is better.
- `NR`: not reported in a form suitable for this table.
- `—`: method was not run or is not applicable.

Absolute success and expansion counts should be compared **within a row only**. C7 onward generally emphasizes matched-solved expansion ratios, while earlier stages report unconditional mean expansions.

## Table 1. Discrete-space planning and transfer

| Regime / protocol | Algorithmic baseline | LSTM / ON-LSTM | HRM | Adaptation or specialist result | Evidence-safe reading |
| --- | --- | --- | --- | --- | --- |
| Dynamic obstacles, matched 100-episode forecasting/planning ([result](discrete/dynamic-world-model/results/LSTM_VS_HRM_EXPERIMENT.md)) | No matched algorithm-only row in this protocol; `E=NR` | LSTM 3M: `S=.69`; LSTM 10M: `S=.68`; `E=NR` | HRM 3M and 10M: `S=.71`; `E=NR` | — | Descriptive two-episode HRM/LSTM gap with no uncertainty. It establishes viability, not architectural superiority. |
| Structured DynamicMaze++ Preset M+, four 100-episode suites ([synthesis](MASTER_EXPERIMENT_SYNTHESIS.md#preset-m-structured-dynamics)) | Algorithm-only aggregate not reported | Best ON-LSTM: `S=.3875` (10M); rollout MSE also favors ON-LSTM 3M | Best HRM: `S=.3425` (small); HRM 10M `.265`, HRM 3M `.200` | — | ON-LSTM is stronger in this structured-dynamics benchmark. Raw expansion totals are confounded by earlier failures and should not be used as efficiency evidence. |
| Zero-shot transfer, six ID/OOD suites ([compendium](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_COMPENDIUM.md#transfer-rl-zero-shot)) | Static A*: `S=.967`, `E=143,319` | ON-LSTM 3M: `S=.956`, `E=141,255` | HRM 3M: `S=.961`, `E=142,193` | No target adaptation in the summarized artifact | Learned arms save about 1–2k expansions but lose 0.6–1.1 percentage points of success. This is not a planner-level win. |
| Clean staged transfer, static → sparse/moderate dynamics; 22 suites × 3 budgets ([result](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_COMPENDIUM.md#clean-transfer-v3)) | Manhattan A*: `S=.590`, `E=126,707` | Full FT: `S=.564`, `E=127,228`<br>LoRA: `S=.514`, `E=145,282` | Full FT: `S=.585`, `E=128,013`<br>LoRA: `S=.554`, `E=140,852` | Every learned arm is below its matched baseline; LoRA is worse than full FT for both backbones | Canonical controlled negative. HRM is closer to baseline than ON-LSTM, but no learned method wins. |
| Multitask pooled training over static and dynamic curricula; 22 suites × 3 budgets ([result](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_COMPENDIUM.md#multitask-tasklora-v1)) | Manhattan A*: `S=.591`, `E=126,630` | Pooled ON-LSTM: `S=.545`, `E=139,254` | Pooled HRM: **`S=.612`, `E=122,184`** | Only the HRM A32-static expert beats its matched four-suite baseline: `+2.0 pp`; other expert deltas range from `-0.42` to `-20.25 pp` | Strongest completed discrete additive signal is the pooled HRM prior, not task specialization. Still descriptive; no formal uncertainty. |
| Learned focal ranking, large static OOD maps, safe `w=1.0` ([result](discrete/learned-heuristic/results/FOCAL_SEARCH_RESULTS.md#4-results-local-rtx-5090-8-seeds-budget-200)) | Manhattan A*: A128 `S=.62`; A192 `S=.75` | ON-LSTM focal: A128/A192 `R=.85`; success `.75→.75` in both tested cells | HRM focal: A128 `R=.85`, success `.62→.75`; A192 `R=.94`, success `.75→.75` | HRM and ON-LSTM A32 experts exactly match their pooled bases in tested focal cells | Local pilot: 6–15% fewer expansions with no observed `w=1.0` success regression. The gain is not HRM-specific and has not received a full-suite run. |
| Learned focal ranking, moderate dynamic OOD map, safe `w=1.0` ([result](discrete/learned-heuristic/results/FOCAL_SEARCH_RESULTS.md#4-results-local-rtx-5090-8-seeds-budget-200)) | Manhattan A*: `S=.62` | Not reported in the firmed-up dynamic cell | HRM focal: `R=.93`, success `.62→.62` | `w=1.05` reaches `R=.83` here but causes a success regression on A192 static | Pilot evidence that ranking integration transfers to dynamics; retain `w=1.0` as the evidence-safe operating point. |
| No-obstacle / empty-map matched row | **Gap:** not isolated in a publication-ready common protocol | **Gap** | **Gap** | **Gap** | Earlier easy Family-A cells are saturated, but a clean no-obstacle row with matched models, budgets, expansions, and path cost still needs to be extracted or rerun. |

### Discrete-table interpretation

The most defensible discrete story is an integration result: additive learned residuals were often inert or harmful against Manhattan A*, while the same pooled HRM and ON-LSTM signals became useful as focal-search rankers. Task-specific LoRA did not improve that ranking. The pooled HRM result is the strongest additive positive, but it is one descriptive protocol rather than a general HRM advantage.

## Table 2. Continuous-space PRM planning and transfer

| Regime / protocol | Euclidean / oracle reference | ON-LSTM | HRM | U-Net / other | Evidence-safe reading |
| --- | --- | --- | --- | --- | --- |
| Easy static pooled pilot, nine suites at B100 ([C1–C4 synthesis](MASTER_EXPERIMENT_SYNTHESIS.md#c1c4-pilot-ladder-and-saturation-diagnosis)) | Euclid: `S=.9667`, `E=31.20` | Pooled ON-LSTM: `S=.9778`, `E=22.92` | Pooled HRM: `S=.9528`, `E=27.05` | — | Saturated pilot. Useful as motivation for harder maps, not as headline architecture evidence. |
| Hard static scalar guidance, C5 B144 formal maze/dense cells ([C5 synthesis](MASTER_EXPERIMENT_SYNTHESIS.md#c5-calibrated-hard-maps-expose-a-strong-on-lstm-result-and-an-hrm-failure)) | Euclid maze/dense: `S=.525/.595` | ON-LSTM maze/dense: **`S=1.000/.962`**, `ΔE=-66.987/-24.519` | HRM matches Euclid after constant-cap collapse | — | Strong ON-LSTM result within one Modal run; HRM failure is an optimization/representation failure, not evidence that hierarchy is intrinsically harmful. |
| Hard static value field, C6 maze B144 ([result](continuous/c06/results/C6_RESULTS.md)) | Euclid: `S=.625`<br>Raster oracle: `S=.950`, `ΔE=-45.325` | Field ON-LSTM: `S=.900`, `ΔE=-31.475` | Field HRM: **`S=.975`, `ΔE=-42.375`** | Field U-Net: `S=.950`, `ΔE=-38.050` | Field training rescues HRM in this setup. Local `n=40`, one training seed; it does not prove value fields are always necessary. |
| Static integration, C7 six hard/held-out suites at binding budgets ([result](continuous/c07/results/C7_RESULTS.md); [summary CSV](../../hrm-cloud/continuous_prm/runs/c7_local/results/continuous_prm_c7_eval_summary.csv)) | Euclid `S=.250–.583`; exact-Dijkstra oracle `S=1.0` | Scalar ON-LSTM `S=.750–1.000`; field ON-LSTM `S=.458–1.000` | Scalar HRM `S=.708–1.000`, `R=.427–.822`<br>Field HRM `S=.750–1.000`, `R=.521–.850` | Field U-Net `S=.542–1.000` | Additive guidance beats focal against loose Euclid. HRM beats Euclid but does not consistently beat ON-LSTM, U-Net, or its own scalar control. Additive suboptimality is approximately `1.02–1.14`. |
| Dynamic planning, C8 heavy, six suites ([result](continuous/c08/results/C8_RESULTS.md); [summary CSV](../../hrm-cloud/continuous_prm/runs/c8_local_heavy/results/continuous_prm_c8_eval_summary.csv)) | Euclid `S=.05–.75`; oracle `S=.95–1.00` | Scalar ON-LSTM reaches `S=.45–1.00`; blind variants are often equal or better than aware variants | Best HRM cell is spiral field-HRM: `S=.90`, `R=.046`; scalar/field HRM reach `.55–1.00` success across suites | Selected field U-Net cells: `S=.75–1.00`, `R=.064–.380`; U-Net is strongest overall | Selected learned arms reduce matched expansions about 65–95%. No systematic future-window advantage; there is no recurrent/hierarchical edge. Local, one seed, and some matched sets are very small. |
| Static few-shot transfer, C9, K=`1…32` ([result](continuous/c09/results/C9_RESULTS.md)) | Euclid is ratio `1.0`; zero-shot learned sources are already strong | Maze-dense: zero-shot/LoRA K1 `R=.873, S=.933`; full-FT K16 `.584/.987` | Maze-dense: LoRA K1 `R=.650, S=1.00`; full-FT K16 `.571/.99`<br>Rooms-large: LoRA K1 `.771/.97`; full-FT K8 `.500/.92` | Field U-Net not included in C9 | LoRA generally preserves the base at very low K; full FT has higher variance and a higher data-rich ceiling. Repeated TEST worlds require world-clustered reanalysis. |
| Matched-compute static transfer, C9h, K=`1,4,16` ([result](continuous/c09h/results/C9H_RESULTS.md)) | Euclid is ratio `1.0` | Maze-dense ON-LSTM: LoRA K1 `.873/.922`; full-FT K16 `.597/.978` | Maze-dense HRM: LoRA K1 `.654/1.00`; full-FT K16 `.591/.99` | Rooms-large U-Net: LoRA K1 `.992/.67`; full-FT K4/K16 **`.404/.97`** | Bound contributes almost nothing (`0.000±.008` median ratio delta); low rank explains base preservation/plateau, while full rank exploits more labels. No architecture-level advantage is established. |
| Dynamic transfer, C9b, K=`1,4,16` ([result](continuous/c09b/results/C9B_RESULTS.md)) | Euclid is ratio `1.0`; transferred zero-shot ratios `.14–.86` | Rooms-large aware ON-LSTM: zero-shot `.856/.65`; LoRA K1 `.428/.93`, K16 `.153/.98`; full-FT K1 `.455/.92`, K16 `.185/.98` | Maze-dense aware HRM: zero-shot `.319/.50`; LoRA K1 `.285/.717`, K16 `.126/.95`; full-FT K1 `.197/.75`, K16 `.100/.933` | Maze-dense aware U-Net: zero-shot `.145/.60`; LoRA K1 `.165/.78`, K16 **`.059/.97`**; full-FT K1 `.112/.70`, K16 `.101/.95` | Transfer beats scratch at K1, but one dynamic world supplies roughly 25k labels, so K=1 is not truly sample-scarce. Future-aware variants win `0/9` full-FT K16 cells. |
| Zero-label adapter interpolation, C10 ([result](continuous/c10/results/C10_RESULTS.md)) | Euclid is ratio `1.0`; learned zero-shot sources already beat it | Zero-shot R: maze `.455`, rooms `.610/.600`; RBF merge `.443/.692/.716` | Zero-shot R: maze `.529`, rooms `.790/.748`; RBF merge `.508/.766/.761` | Nearest, uniform, RBF weight merge, and prediction mix all remain near the same base ceiling | Clean null: routing is selective, but no interpolation method consistently improves on the pooled zero-shot model. |

### Continuous-table interpretation

The continuous program has a coherent representation-and-regime story:

1. Easy maps saturate and hide architectural differences.
2. Hard maps make learned guidance useful.
3. Better representation/training can rescue a failed backbone, but simpler models remain competitive.
4. Additive integration works when Euclidean distance is loose; discrete focal ranking works when Manhattan is tight.
5. LoRA is base-preserving under genuinely scarce supervision, while full-rank adaptation exploits richer supervision.
6. Future-window inputs and zero-label adapter interpolation add no consistent benefit.

## Results not folded into the common success/expansion matrix

- **C11 compositional missions:** the metric schema changes from obstacle regime to mission length and completion. The completed architecture test finds no hierarchy/depth dose response. In the scaled addendum, U-Net completion is `.804` versus HRM `.736` at K=2 and `.413` versus `.307` at K=8; HRM K=8 matches the leg-sum baseline on completion and mean expansions. See [C11 results](continuous/c11/results/C11_RESULTS.md).
- **C12 persistent dynamics:** the final frozen pilot completed after the meeting and is development-only. It reports `G0-A PASS`, `G1-A FAIL`, `G2-A FAIL`, `G3-A FAIL`, and `G4-A strong_negative`; persistent temporal hierarchy adds no useful value despite confirmed memory headroom. See the [final pilot analysis](../../hrm-cloud/continuous_prm/runs/c12_persistent_pilot_v6_final/results/C12A_ANALYSIS.md).
- **C13 current-state revalidation:** a fixed bounded-observation/local-Bellman arm confirms on 144 untouched worlds at `68.31` expansions versus `81.26` for complete-map field HRM (paired delta `-12.96`, 95% CI `[-16.30, -9.74]`). All six suite means are negative, and empirical mean/max path-cost ratios are lower. The direct arm is not formally bounded or faster in wall time; a separate `w=1.10` FOCAL control has zero violations. See the [canonical C13-F–M result](continuous/c13/results/C13F_M_CURRENT_STATE_RESULT.md).
- **HRM-v2 direct maze solver:** this is a direct solver rather than an A*-heuristic comparison and should remain outside these planner tables.

## Gaps to close before the next professor review

1. Extract a clean discrete no-obstacle row—or explicitly remove that requested row if it does not match a completed protocol.
2. Add path cost/suboptimality wherever raw data support it; most discrete summaries currently omit path cost.
3. Add world-clustered confidence intervals for the compact C9/C9h/C9b cells; the current values are descriptive curve summaries.
4. Recompute C9/C9h/C9b uncertainty at the world level rather than treating repeated adaptation-seed evaluations as independent worlds.
5. Decide whether the meeting table should show only paper-core C7–C13 results or preserve the broader program history as above.
6. Keep pilot/development-only evidence visually separate from canonical evidence in any slide or paper version.
7. Review C13's exact wording: “bounded observations of each search state on a known PRM,” not “map-free,” and direct matched-quality versus separate bounded FOCAL.

## Source hierarchy used

1. Canonical result documents and their raw/summary artifacts.
2. Completed descriptive reports.
3. Local pilots and development-only analyses.
4. Designs, plans, and historical notes.

The broader evidence audit remains in [MASTER_EXPERIMENT_SYNTHESIS.md](MASTER_EXPERIMENT_SYNTHESIS.md).


