# Master Experiment Evidence Synthesis

**Snapshot:** base inventory 2026-07-10; C12 findings refreshed 2026-07-14; C13-M confirmation plus C13-N/C13-O HRM diagnostics refreshed 2026-07-19; completed valid C13-P persistent-state pilot recorded 2026-07-20; key-quantity world-clustered reanalysis for C9/C9h/C9b/C10/C11 completed 2026-07-20
**Purpose:** working evidence base for a research-paper scaffold covering the discrete-grid and continuous-PRM programs.
**Primary audience:** technical readers who need to distinguish measured results, interpretation, caveats, and unfinished work.

This document is the claim-level companion to the [experiment documentation index](README.md). It consolidates the figures, headline results, negative findings, caveats, and reusable lessons from the repository's experiment record. It is intentionally stricter than several source reports: when a narrative headline is broader than its generated table, the narrower evidence-safe wording is used here and the discrepancy is recorded.

## Coverage and status labels

The documentation review covers:

- Authored designs, plans, result reports, audits, and historical notes through the completed C13-M confirmation, C13-N/C13-O architecture diagnostics, and valid negative C13-P persistent-state pilot.
- **54 generated continuous-run Markdown reports** and **5 generated discrete Modal-survey reports** through the C13-P completion refresh.
- The [complete discrete experiment inventory](discrete/DISCRETE_EXPERIMENT_INVENTORY.md) (added 2026-07-23) maps all 15 discrete experiment families, ~31 cloud scripts, and the 11 surveyed Modal volumes (13,671 result files) onto their documentation and coverage status; it adds chronology/controls not digested in this synthesis (Preset M v1, the imitation empty-model rerun control, clean-transfer v1/v2 partials, the budget-invariance analysis tool, and the ~10-run early scaling lineage) without changing any claim.
- The raw-result layer beneath the reports, including checkpoints, feature caches, cohort manifests, raw paired rows, integrity records, and suite shards; exact artifact counts from the older base inventory are no longer treated as current.

Binary checkpoints, training tensors, and every individual JSON record were inventoried but are not treated as independent narrative evidence. Claims come from canonical result tables and selected raw CSV/JSON spot-checks. Smoke reports are kept as implementation evidence, not counted as independent scientific replications.

Status labels used below:


| Label                            | Meaning                                                                                                         |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Canonical local result**       | Completed analysis with saved result artifacts; generally one machine and one main training seed unless noted.  |
| **Completed descriptive result** | Completed outcome without formal uncertainty or hypothesis tests.                                               |
| **Pilot / directional**          | Useful signal with small samples, partial coverage, or weak provenance.                                         |
| **Negative / null**              | No supported improvement under the tested protocol; not proof of equivalence unless an equivalence test exists. |
| **Historical / superseded**      | Retained for chronology but overridden by a later audit or cleaner run.                                         |
| **Design / methods only**        | Defines the protocol but contributes no independent outcome.                                                    |
| **In progress**                  | Training or evaluation is incomplete; no result claim is allowed.                                               |


## Technical summary

1. **Learned heuristics reliably reduce search effort on calibrated continuous PRMs.** C7 reports roughly **15–48%** fewer matched A* expansions for selected additive learned heuristics across static suites. The C8 heavy run reports about **65–95%** fewer matched expansions on dynamic suites, with corrected success evidence on five of six suites. These are primarily local, one-GPU results; some binding-budget matched sets are small.
2. **The discrete learned-heuristic result is smaller but mechanistically clean.** The pooled discrete model is a near-perfect state ranker (`rho` about **0.987–0.994**) whose magnitude does not scale with map size. Reusing the same weights as a focal tie-breaker at `w=1.0` reduced expansions **6–15%** with no observed success regression in the tested cells. This remains a local pilot with only 3–8 seeds and no full-suite inferential analysis.
3. **Transfer works, but its shape depends on label scarcity.** Static C9/C9h show that low-rank adaptation usually preserves a strong pooled base at very low K, while full-rank adaptation often reaches a lower expansion ratio when enough target data are available. C9h shows bounded and unbounded rank-8 LoRA are nearly identical (`median delta = 0.000 +/- 0.008`), so low-rank capacity—not the clamp—best explains the plateau. Under dynamics, one world supplies roughly **25k+ `(node,t)` labels**, and the static low-data crossover disappears.
4. **No stable hierarchical-model advantage is supported on the current heuristic-regression formulation.** HRM, ON-LSTM, and U-Net all beat algorithmic baselines in different cells, but HRM does not consistently beat the simpler learned models; U-Net is often strongest. Architecture separation did appear on the earlier forecasting task, where the sign was task-dependent: HRM 3M/10M reached **71/100** versus matched LSTM variants at **68–69/100**, while Preset M+ favored ON-LSTM (**0.3875** best mean; **0.2650** for HRM-10M). None of those discrete gaps has formal uncertainty.
5. **The strongest negatives are coherent and useful.** Task-specific adapters generally fail to beat the pooled base; future-window-aware dynamic heuristics show no systematic advantage over present-frame twins; and descriptor-weighted adapter interpolation does not consistently beat zero-shot, uniform, or nearest composition. These are conditional findings about the tested formulations, not universal equivalence claims.
6. **Harness choices repeatedly determine whether learned signal is visible.** The program has three successive lessons: representation (C5 scalar collapse to C6 field recovery), planner integration (discrete additive magnitude to focal ranking), and task formulation. C11 exposes global map/graph/mission structure and finds real shallow architecture separation, yet retaining a one-shot scalar regression target still produces no depth-of-compute advantage. Apparent model failures should not be generalized beyond the information, objective, and integration actually tested.
7. **C11 supplied real compositional headroom but no hierarchy dose-response.** The exact oracle/leg-sum ratios remain **0.082–0.225**. The completed main grid contains **198 checkpoints** and **54,450 evaluation rows**. G1 is negative: no structured arm beats MLP at two mission depths with a non-decreasing gap. Global-input U-Net/GNN arms have genuine shallow-K advantages (reported ratios roughly `0.67–0.87` versus MLP), but the edge dissolves with depth. Forced HRM-v2 compute is flat, and learned halting is significantly anti-correlated with mission depth (`rho=-0.407`, `p≈0.0005`). Several recurrent cells collapse, so their worst deep-K rows are optimization-pathology evidence rather than clean capacity comparisons. The scaled best/worst-arm addendum remains in progress and is excluded from current claims.
8. **C13 turns the professor's current-state constraint into a confirmed result by separating target, representation, distribution, and integration.** The exact behavior return and exact shallow local-escape targets fail even after C13-D repairs the shared-search ceiling. One-suite local-Bellman learning confirms only on maze; suite-balanced training still loses when inserted statically. The decisive mechanism is one radius-bounded Bellman backup at inference. C13-M freezes that model/integration before a disjoint 144-world run: current-state A* averages `68.31` expansions versus `81.26` for complete-map field HRM (paired delta `-12.96`, 95% CI `[-16.30, -9.74]`, 109/3/32 W/T/L), all six suite means are negative, and empirical mean/max path-cost ratios are lower. The direct arm is not formally bounded and its prototype feature construction is slower; a separate `w=1.10` FOCAL control passes all safety certificates.
9. **A literal HRM substitution does not preserve the full C13-M result.** C13-N changes only `flat_mlp -> hrm_trimmed` on the frozen C13-J cohorts and local-backup integration. Its fixed endpoint has useful pooled development signal versus field HRM (`-8.625` expansions, 95% CI `[-16.667, -1.208]`) but only 3/6 suite means improve. Against the matched flat model, the expansion CI crosses zero and HRM's mean/max path-cost ratios are worse by `0.0123/0.0641`. The preregistered gate fails and no fresh confirmation is opened. This points to sequence/readout and optimization alignment, not a simple integration incompatibility.
10. **Summary-last ordering changes HRM behavior but is not the missing mechanism.** C13-O starts trimmed and summary-last HRM from identical tensors and changes only valid-token order. At iteration 6, summary-last improves trimmed HRM by `3.625-3.875` expansions with confidence intervals excluding zero, but neither cell passes the field-HRM method gate. At the fixed iteration-8/alpha-1.50 endpoint, summary-last still beats field HRM in pooled expansions (`-6.542`, 95% CI `[-13.251, -0.333]`) but improves only 3/6 suites, fails matched-flat path-quality margins, and is `+2.083` expansions worse than trimmed with an inconclusive CI. No confirmation is opened. The effect is transient; persistent planning state and moving-target stability remain untested.
11. **Persistent HRM search state fails the same-checkpoint reset ranking gate.** C13-P completes validly after named mechanical repairs and a fresh fingerprint: G0-P passes, including independent reconstruction from promoted raw artifacts, but persistent-minus-reset world-macro MRR is `-0.0294` with 95% CI `[-0.0598, -0.0030]`, top-1 delta `-0.0280`, and only 3/6 suites positive. The frozen verdict is `c13p_no_persistent_ranking_signal`. Descriptive free-running G2-P also fails; no self-bootstrap or confirmation is run. This is a negative result for the frozen target/representation/model/training protocol, not a general HRM or memory rejection.

## Claim readiness for a paper


| Candidate claim                                                 | Evidence-safe status                      | Best evidence                                                                                                                            | Required qualification                                                                                                                      |
| --------------------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Learned heuristics reduce static continuous-PRM search effort   | **Ready with local-validation caveat**    | [C7 results](continuous/c07/results/C7_RESULTS.md) and generated C7 tables                                                               | One main training seed, 24 eval worlds/suite, some matched sets n=6; expansion tests are not multiplicity-corrected.                        |
| Learned heuristics reduce dynamic space-time search effort      | **Ready with local-validation caveat**    | [C8 heavy results](continuous/c08/results/C8_RESULTS.md)                                                                                 | One GPU/seed lineage; dense-maze matched n=1 at the binding budget; success and conditional expansion evidence must remain separate.        |
| A pooled prior improves low-label adaptation                    | **Ready after clustered reanalysis**      | [C9](continuous/c09/results/C9_RESULTS.md), [C9h](continuous/c09h/results/C9H_RESULTS.md), [C9b](continuous/c09b/results/C9B_RESULTS.md) | TEST worlds repeat across adaptation seeds; bootstrap/tests should cluster at world level. Method-vs-method direct tests are mostly absent. |
| Low-rank capacity, not the output clamp, causes LoRA's plateau  | **Strong descriptive mechanism claim**    | [C9h results](continuous/c09h/results/C9H_RESULTS.md)                                                                                    | No formal equivalence test; bounded/unbounded identity is descriptive across 27 cells.                                                      |
| Explicit future-window inputs improve heuristic guidance        | **Not supported**                         | [C8 heavy](continuous/c08/results/C8_RESULTS.md), [C9b](continuous/c09b/results/C9B_RESULTS.md)                                          | Phrase as failure to observe a consistent benefit, not equivalence or proof that future information never helps.                            |
| Descriptor-weighted interpolation improves zero-label transfer  | **Not supported**                         | [C10 results](continuous/c10/results/C10_RESULTS.md)                                                                                     | Only two axes, a strong pooled base, transductive target descriptors, and no direct equivalence/noninferiority tests.                       |
| Hierarchical models are superior planning heuristics            | **Not supported on tested formulations**  | [C11 results](continuous/c11/results/C11_RESULTS.md), [C12 results](continuous/c12/results/C12_RESULTS.md), [program audit](cross-space/PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md), [head-to-head](cross-space/HRM_HEADTOHEAD.md) | Do not generalize to all planning. C12-A's completed one-seed pilot closes `strong_negative`; C12-B improves monotonically with recurrent graph cycles but fails the preregistered K-dose response. A localized tied-control win on C/K=8 is not a general hierarchy result. |
| A bounded-observation local-Bellman heuristic improves over complete-map C7 providers | **Confirmed locally for search expansions at matched empirical path quality** | [C13-F through C13-M result](continuous/c13/results/C13F_M_CURRENT_STATE_RESULT.md), [C13-M preregistration](continuous/c13/design/2026-07-17-c13m-matched-quality-confirmation.md) | One model seed and one 144-world confirmation cohort. The alpha-1.50 direct arm is unbounded and the Python feature builder is slower in wall time; claim expansions/path quality, not formal safety or end-to-end speed. |
| A literal trimmed-HRM substitution preserves the C13-M gain | **Not confirmed** | [C13-N result](continuous/c13/results/C13N_HRM_SUBSTITUTION_RESULT.md), [C13-N preregistration](continuous/c13/design/2026-07-17-c13n-hrm-substitution.md) | Development-only architecture diagnostic. The pooled field-HRM comparison is promising, but suite robustness and matched-MLP path-quality gates fail; no confirmation cohort was evaluated. |
| Moving the summary token to the final HRM readout recovers the C13-M gain | **Not confirmed** | [C13-O result](continuous/c13/results/C13O_HRM_ALIGNMENT_RESULT.md), [C13-O preregistration](continuous/c13/design/2026-07-17-c13o-hrm-summary-last-alignment.md) | Two iteration-6 cells improve trimmed HRM directly, but no cell passes the complete field/flat/readout gate and the fixed endpoint readout comparison reverses; no confirmation cohort was evaluated. |
| Query-level persistent HRM search state improves frontier ranking and search | **Not supported for C13-P** | [C13-P preregistration](continuous/c13/design/2026-07-19-c13p-persistent-search-state.md), [implementation plan](continuous/c13/plans/2026-07-19-c13p-persistent-search-state.md), and [canonical result](continuous/c13/results/C13P_PERSISTENT_SEARCH_RESULT.md) | Valid completed pilot: G0-P passes, but persistent carry is worse than same-checkpoint reset on world-macro MRR (`-0.0294`, 95% CI `[-0.0598, -0.0030]`); descriptive G2-P also fails. No self-bootstrap or confirmation was run. |
| Compositional missions create enough headroom to test hierarchy | **Supported as an oracle-headroom claim** | [C11 headroom](continuous/c11/results/C11_HEADROOM.md)                                                                                   | This is an upper bound; the completed learned arms capture some shallow-K benefit but no hierarchy dose-response.                            |
| Hidden slow/fast dynamics create memory-relevant headroom       | **Supported as a G0 authorization claim** | [C12 results](continuous/c12/results/C12_RESULTS.md)                                                                                    | The strong arm is a privileged mode diagnostic, not a learned hierarchy result. The completed one-seed learned pilot fails forecast, planning, and persistent-carry gates and remains development-only. |
| Structured arms exploit compositional depth better than MLP     | **Not supported**                         | [C11 results](continuous/c11/results/C11_RESULTS.md)                                                                                     | Global-input arms win selectively at shallow K, but no arm meets the preregistered depth dose-response; recurrent collapse and repeated-seed inference require care. |
| Additional HRM-v2 compute improves deep-mission guidance        | **Not supported**                         | [C11 results](continuous/c11/results/C11_RESULTS.md)                                                                                     | Forced k=1/2/4/8 curves are effectively flat; learned halting moves in the opposite direction from the hypothesis.                           |
| The HRM-v2 port reproduces the Maze-Hard paper result           | **Not yet supported**                     | [fidelity audit](discrete/direct-solver/audits/PORT_FIDELITY_AUDIT.md), [retrain](discrete/direct-solver/results/RETRAIN_RESULTS.md)     | Forward parity and live mechanisms are verified, but the paper-scale convergence run is unfinished.                                         |


## Scope, metrics, and comparability

### Evidence hierarchy

1. Canonical result documents plus their generated raw/summary artifacts.
2. Completed descriptive reports without formal statistics.
3. Pilot, smoke, interrupted, or weak-provenance outputs.
4. Designs, plans, synthesis memos, historical notes, and external literature.

Program syntheses such as [The Continuous PRM Story](continuous/program/CONTINUOUS_PRM_STORY.md) and the [program audit](cross-space/PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md) are used to organize interpretation. Numerical claims should cite the underlying stage report.

### Core metric definitions


| Metric                    | Definition and caution                                                                                                                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Success                   | Fraction of episodes/worlds that reach a valid solution within the experiment's search or step budget. Budgets and environments differ across stages, so absolute success should not be compared across unrelated experiments. |
| Expansion count           | A* node expansions. Some documents report means over all attempts; C7 onward usually emphasizes paired ratios on worlds both methods solve.                                                                                    |
| Expansion ratio           | Learned-arm expansions divided by baseline expansions on matched solved worlds. Below 1 is better. This conditions on both methods succeeding and can favor a low-success arm that solves only easier worlds.                  |
| Success delta             | Learned success minus matched baseline success, usually in absolute rate or percentage points.                                                                                                                                 |
| Suboptimality             | Found path cost or makespan divided by exact graph/space-time optimum. Additive learned heuristics are often inadmissible; focal search preserves a configured bound.                                                          |
| K                         | Number of labeled adaptation worlds in C9/C9h/C9b, or mission length in C11. These uses are unrelated and must be labeled.                                                                                                     |
| McNemar/BH q              | Paired success test and Benjamini-Hochberg adjusted p-value, usually learned arm versus Euclid. It is not automatically a direct LoRA-versus-full-FT or aware-versus-blind test.                                               |
| Bootstrap CI / Wilcoxon p | Uncertainty for paired expansion ratios. In C7/C8, Wilcoxon grids are generally exploratory and uncorrected; bootstrap intervals should lead the interpretation.                                                               |


### Cross-stage comparability rules

- Do not merge the **50-episode 68% HRM** result from `BENCHMARK_RESULTS.md` with the **100-episode 71% HRM** result from `LSTM_VS_HRM_EXPERIMENT.md`; they are different protocols.
- C1–C6 primarily emphasize success and mean expansions, while C7 onward uses matched-solved median ratios and paired statistics. A single cross-phase effect-size series would be misleading without reanalysis.
- C9–C10 repeated the same TEST worlds across adaptation seeds. Treat `world x seed` record counts as repeated measurements, not independent-world counts.
- Smoke runs, analyzer checks, and post-hoc oracle expert selection are implementation/ceiling evidence, not independent confirmations.

## Claim audit: source prose versus evidence-safe wording


| Source headline or risk                                       | Evidence-safe wording for the paper                                                                                                                                                    |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C7 says the “hierarchical thesis holds.”                      | HRM learned heuristics beat Euclid, but HRM did not consistently beat ON-LSTM, scalar controls, or U-Net. No hierarchical superiority is established.                                  |
| C6 can be read as proving value fields are necessary.         | The field formulation rescued HRM in the C6 setup; C7 later showed a properly trained scalar model can also perform strongly.                                                          |
| C8 says time-aware models are worse.                          | No systematic future-window benefit was observed: 7 uncorrected significant cells favored blind and 1 favored aware; direct MAE was lower for aware in 11/24 cells and blind in 13/24. |
| C9/C9h say full-FT is worse than Euclid at K=1.               | Prominent HRM cells overfit badly at K=1, but several ON-LSTM/U-Net cells still beat Euclid. The effect is high variance and target/backbone dependent.                                |
| C9 says scratch is indistinguishable from Euclid through K=8. | Scratch frequently has poor low-K success, but several target/backbone/K cells are significant against Euclid.                                                                         |
| C9/C9h describe LoRA as never catastrophic.                   | LoRA usually preserves the base at K=1, but can degrade at larger K on bugtrap and rooms-large.                                                                                        |
| C9b says every zero-shot source significantly beats Euclid.   | Every reported expansion ratio is below 1, but some rooms-large zero-shot success contrasts have q values of 0.463–1.000.                                                              |
| C10 says all learned arms have significant success gains.     | All learned arms have expansion ratios below 1; most success contrasts are significant, while several maze cells have BH q=0.062.                                                      |
| C10 says targets are inside the descriptor hull.              | The implemented check verifies coordinate-wise active-dimension bracketing on the target's own axis; it is weaker than full 8-D convex-hull membership.                                |
| C11's failed K=0 continuity check invalidates all evaluation. | The measurement layer was verified; the failed premise was that C11 K=0 must reproduce a C7 tie despite adding an MLP and class-specific inputs. Main G1/G2 verdicts remain negative, with collapse caveats. |
| Pre-fix HRM-v2 metrics demonstrate ACT reasoning.             | The trainer omitted `q_halt_loss` and deep supervision; `74.60% q-halt accuracy = 100 - 25.40% exact` is the frozen-head signature. Use as a validity failure case only.               |


# Discrete-space evidence

## Dynamic-world forecasting and planning

### Early HRM scaling and diffusion benchmark

**Source:** [BENCHMARK_RESULTS.md](discrete/dynamic-world-model/results/BENCHMARK_RESULTS.md)
**Status:** completed descriptive developmental benchmark.

The task uses a 20x20 dynamic grid with 12 static and 6 bouncing obstacles. A recurrent model receives a 20-step obstacle history, predicts future movement, and supports receding-horizon A*. The principal evaluations use 50 episodes.


| Model/run    | Success     | Important context                                         |
| ------------ | ----------- | --------------------------------------------------------- |
| Small HRM    | 26/50 (52%) | Paired LSTM 36/50 (72%); different early setup.           |
| Mid HRM      | 31/50 (62%) | Paired LSTM 34/50 (68%).                                  |
| 28.97M HRM   | 34/50 (68%) | Paired LSTM 33/50 (66%); one-episode gap, no uncertainty. |
| Diffusion v1 | 32/50 (64%) | About 500k parameters.                                    |
| Diffusion v2 | 30/50 (60%) | Roughly 8x parameters, 2x data/epochs, no gain.           |
| Oracle A*    | 49/50 (98%) | Perfect future information.                               |


**Caveats and learning.** LSTM Mid also reached 68%, so the full HRM was not uniquely best across all runs. Diffusion saw a static map while recurrent models received obstacle histories, making the information sets unequal. The source itself treats the 60%/64% diffusion difference as likely noise and recommends at least 200 evaluation episodes. The reusable lesson is that information access and stable gated recurrence mattered more than simply enlarging a direct path generator.

### Matched LSTM-versus-HRM scale comparison

**Source:** [LSTM_VS_HRM_EXPERIMENT.md](discrete/dynamic-world-model/results/LSTM_VS_HRM_EXPERIMENT.md)
**Status:** completed descriptive result; canonical for this 100-episode protocol.

All models trained on roughly 18M samples from 60k episodes and were evaluated on 100 shared seeds.


| Model     | Parameters | Success    |
| --------- | ---------- | ---------- |
| HRM 3M    | 3,624,194  | **71/100** |
| HRM 10M   | 12,224,642 | **71/100** |
| LSTM 3M   | 3,013,002  | 69/100     |
| LSTM 10M  | 16,230,602 | 68/100     |
| LSTM 300K | 311,362    | 67/100     |
| LSTM 1M   | 1,016,742  | 67/100     |
| HRM small | 907,394    | 64/100     |


**Caveats and learning.** No confidence intervals or tests are reported; the matched HRM/LSTM gap is only two episodes. HRM plateaus from 3M to 10M, LSTM is relatively flat, and small HRM is worse. The defensible interpretation is task-specific scale behavior, not a definitive architecture win.

### Preset M+ structured dynamics

**Sources:** [Preset M+ design](discrete/dynamic-world-model/design/ONLSTM_VS_HRM_EXPERIMENT_PRESETM_V2.md), raw `modal_downloads/survey_results/onlstm_comparison_presetm_v2_results.json`, and the [discrete compendium](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_COMPENDIUM.md).
**Status:** completed descriptive raw result; the design document's results template is stale.

The 32x32 protocol uses room/corridor maps, gates, patrollers, drifters, five-step autoregressive loss, and four 100-episode suites.


| Model        | Mean success across four suites |
| ------------ | ------------------------------- |
| ON-LSTM 10M  | **0.3875**                      |
| ON-LSTM 1M   | 0.3725                          |
| ON-LSTM 3M   | 0.3650                          |
| HRM small    | 0.3425                          |
| ON-LSTM 300K | 0.3225                          |
| HRM 10M      | 0.2650                          |
| HRM 3M       | 0.2000                          |


ON-LSTM 3M also has the lowest mean one-step/five-step rollout MSE (`0.1257/1.1269`) versus HRM 3M (`0.3973/2.6438`). Collisions dominate failures, and lower raw expansion totals for HRM are confounded by earlier termination. This is a strong descriptive counterexample to any blanket claim that HRM's hierarchy is superior.

### External background paper

**Source:** [LSTM-augmented A* paper](discrete/dynamic-world-model/references/lstm-augmented-astar-2025.pdf)
**Status:** external literature, not repository-produced evidence.

The paper reports very low LSTM prediction error (`test MSE 0.0024`, `R2 0.9976`) and a 150-trial robustness mean of `65.60% +/- 15.72` success, with large condition dependence (best `93.5% +/- 4.4`, worst `33.9% +/- 2.1`). It is useful for literature framing around prediction-augmented A*, but its MATLAB simulation and metric conventions are not directly comparable with this repository.

## Discrete learned A* heuristics and transfer

### Historical zero-shot and original LoRA curricula

**Sources:** [two-run writeup](discrete/learned-heuristic/results/experiment_writeup_last_two_runs.md) and [results compendium](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_COMPENDIUM.md).
**Status:** completed negative/descriptive results.

- Transfer-RL zero-shot averaged success `0.967` for Manhattan, `0.961` for HRM 3M, and `0.956` for ON-LSTM 3M. Small expansion reductions came with lower success.
- In the original curriculum, hard family-B suites were unchanged: `OOD_B32_D1` stayed at `0.47` success and `OOD_B64_D2` at `0.34` across baseline and learned variants.
- The map-scale curriculum showed A-family size scaling and sparse dynamics were already easy (`1.00` success), while `OOD_B64_static` remained `0.28` and `OOD_B64_sparseDyn` `0.34` for all methods.
- Few-shot K=50/200 variants did not move the hardest family-B cells. Raising the search budget increased work but did not rescue success.

The documented learning is narrow but valuable: the problem was a specific family shift and an inert residual-to-planner interface, not generic map-size transfer or LoRA alone.

### Clean transfer v3: controlled negative

**Sources:** [compendium](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_COMPENDIUM.md) and `modal_downloads/clean_v3_results/final_results__A64_moderateDyn.json`.
**Status:** completed canonical negative.

The controlled run covers 22 suites, three budgets, and 100 episodes per row. Learned alphas all tuned to the candidate floor `0.5`.


| Model                  | Mean success | Matched delta vs Manhattan | Mean expansions |
| ---------------------- | ------------ | -------------------------- | --------------- |
| Manhattan              | 0.590        | reference                  | 126,707         |
| HRM full fine-tune     | 0.585        | -0.59 pp                   | 128,013         |
| ON-LSTM full fine-tune | 0.564        | -2.67 pp                   | 127,228         |
| HRM LoRA               | 0.554        | -3.62 pp                   | 140,852         |
| ON-LSTM LoRA           | 0.514        | -7.67 pp                   | 145,282         |


Every learned arm underperformed matched Manhattan. No formal uncertainty is reported; means are unweighted over suite-budget rows. Some raw diagnostic fields are NaN although headline success/expansion fields are finite, so diagnostic aggregation must filter undefined values.

### CondLoRA: incomplete and unfavorable

**Source:** [compendium](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_COMPENDIUM.md).
**Status:** incomplete negative; no final file.

The available 329 aggregates cover different subsets. ON-LSTM average fine-tuning reports `0.517` success (`-7.42 pp`), while HyperLoRA reports `0.551` over only 37 rows/14 suites (`-7.08 pp` matched). HyperLoRA's lower mean expansions are not comparable with the full baseline because the evaluated subset is easier. This arm belongs in an appendix or omission log, not a headline table.

### Multitask pooled base: strongest discrete additive positive

**Sources:** [compendium](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_COMPENDIUM.md) and `modal_downloads/multitask_results/final_results__multitask_tasklora.json`.
**Status:** completed descriptive result.


| Model                    | Success   | Matched delta | Mean expansions |
| ------------------------ | --------- | ------------- | --------------- |
| Manhattan                | 0.591     | reference     | 126,630         |
| HRM pooled `avgbase`     | **0.612** | **+2.11 pp**  | **122,184**     |
| ON-LSTM pooled `avgbase` | 0.545     | -4.59 pp      | 139,254         |


Experts were tested only on four easier ID suites, whose matched baseline was `0.803`. Only the HRM A32 expert improved (`+2.00 pp`); the other expert deltas range from `-0.42` to `-20.25 pp`. This is the best completed discrete additive signal, but it remains descriptive and does not establish a general specialist benefit.

### Residual TaskLoRA v2: interrupted result and clean re-evaluation

**Sources:** [compendium](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_COMPENDIUM.md) and [focal redesign report](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_FOCAL_REDESIGN.md).
**Status:** interrupted aggregate superseded by a clean local re-evaluation.

The interrupted run has no final file and includes a nonfinite-prediction cell. The clean follow-up reports pooled HRM `+0.010` success versus Manhattan and the A32 expert `-0.010` versus the pooled base, with `7 wins / 10 ties / 27 losses` over 44 cells. Family B remains near `0.01` for every method; family C is saturated near `1.0`. Only the clean follow-up should be used.

### Learned focal search: ranking rescue

**Sources:** [FOCAL_SEARCH_RESULTS.md](discrete/learned-heuristic/results/FOCAL_SEARCH_RESULTS.md), [focal redesign report](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_FOCAL_REDESIGN.md), [design](discrete/learned-heuristic/design/2026-06-23-learned-focal-search-design.md), and [implementation plan](discrete/learned-heuristic/plans/2026-06-23-learned-focal-search.md).
**Status:** validated local pilot; the two result documents describe the same evidence and must be counted once.

The key diagnostic is a scale-flat magnitude despite strong state ordering:


| Map size | Correlation with true residual | Predicted mean | True residual mean | Pred/true |
| -------- | ------------------------------ | -------------- | ------------------ | --------- |
| 64       | 0.987                          | 88             | 121                | 0.73      |
| 128      | 0.992                          | 105            | 249                | 0.42      |
| 192      | 0.994                          | 94             | 345                | 0.27      |
| 256      | 0.992                          | 96             | 493                | 0.19      |


The source does not specify Pearson versus Spearman for `rho`; retain the generic term correlation until the code is checked.

At `w=1.0`, the learned signal only breaks ties inside the admissible focal band:


| Backbone/suite              | Median expansion ratio | Success baseline to focal |
| --------------------------- | ---------------------- | ------------------------- |
| HRM, A128 static            | 0.85                   | 0.62 to 0.75              |
| HRM, A192 static            | 0.94                   | 0.75 to 0.75              |
| HRM, A128 moderate dynamics | 0.93                   | 0.62 to 0.62              |
| ON-LSTM, A128 static        | 0.85                   | 0.75 to 0.75              |
| ON-LSTM, A192 static        | 0.85                   | 0.75 to 0.75              |


This is a **6–15%** expansion reduction with no observed regression in the tested `w=1.0` cells. At `w=1.05`, A128 dynamic improves more but A192 success falls `0.75 to 0.62`. Experts exactly match the pooled base in the tested cells, showing the bounded specialist correction did not change node order.

**Caveats.** The result is local, 3–8 seeds, budgets 150–200, with no A256 or 22-suite sweep and no formal uncertainty. `w=1.0` should be called empirically regression-free in tested cells, not theoretically zero-risk under a finite expansion budget.

### Evaluation acceleration and provenance

**Source:** [FAST_EVAL.md](discrete/learned-heuristic/operations/FAST_EVAL.md) and its [performance plan](discrete/learned-heuristic/plans/2026-06-15-residual-tasklora-eval-speedup.md).
**Status:** completed engineering result.

The exact-cost diagnostic DP consumed about 61% of episode time even on a 64x64 map. After fixing remote environment propagation, a representative shard improved from roughly 100 seconds/episode to 2.6 seconds/episode, about **39x**, while tests preserved success/steps/expansions. `EVAL_DIAG=0` intentionally drops diagnostic fields. A reproducibility warning follows: before the forwarding fix, shell-set `EVAL_*` knobs were silently ignored by remote workers.

## HRM-v2 direct maze solver and fidelity

### Pre-fix Maze-Hard result is not faithful-HRM evidence

**Source:** [TRAINING_RESULTS.md](discrete/direct-solver/results/TRAINING_RESULTS.md)
**Status:** completed but scientifically superseded.

The 27.27M-parameter run reports `96.64%` eval token accuracy, `25.40%` exact accuracy, `74.60%` q-halt accuracy, and `16.0` average ACT steps. The later audit proves `74.60 = 100 - 25.40`: with the halt logit frozen negative, the metric is a constant-false classifier. The run omitted the q-halt loss and backpropagated only the final segment. Use this as a validity failure case, not a benchmark result.

### Port-fidelity audit and completed mechanism fixes

**Sources:** [PORT_FIDELITY_AUDIT.md](discrete/direct-solver/audits/PORT_FIDELITY_AUDIT.md) and [fix plan](discrete/direct-solver/plans/2026-07-05-hrm-v2-port-fixes.md).
**Status:** canonical validity audit; mechanism defects resolved.

The model forward port was structurally faithful, but the training loop was not. The audit identified missing `q_halt_loss`, broken per-segment deep supervision, an SDPA layout hazard, sparse-embedding fragility, and recipe drift. Subsequent commits restored ACT loss, streaming deep supervision, AdamATan2/warmup recipe behavior, an explicit attention-layout contract, sparse-optimizer handling, full-state checkpointing, and original-versus-port parity tests. Forward parity is bit-exact, including the short-sequence regime that could previously attend across heads.

Bit-exact forward parity certifies computation; it does not establish task convergence.

### Post-fix partial revalidation

**Source:** [RETRAIN_RESULTS.md](discrete/direct-solver/results/RETRAIN_RESULTS.md)
**Status:** mechanism-validation partial run, stopped by choice.

The run stopped around `150k / 375k` configured steps after about 12 hours. `q_halt_loss` declined `0.14 to 0.06`, train token accuracy reached `96.5%`, eval token accuracy was `90–95%`, and exact accuracy remained `0.000`. This proves the repaired loss path and supervision mechanism are live, but does not yet demonstrate adaptive halting quality or Maze-Hard exact-match performance.

The report's epoch/sample-exposure arithmetic is internally inconsistent: a run stopped near 40% of configured steps cannot simultaneously have completed all 1,500 configured epochs under uniform steps/epoch. Correct that calculation before paper use.

### Superseded HRM-v2 history


| Document                                                                          | Reliable contribution                             | Superseded content                                |
| --------------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------- |
| [HRM_V2_REVIEW_REPORT.md](discrete/direct-solver/history/HRM_V2_REVIEW_REPORT.md) | Found the truncated-normal initialization bug.    | “Production ready” and broad fidelity conclusion. |
| [BUGFIX_APPLIED.md](discrete/direct-solver/history/BUGFIX_APPLIED.md)             | Records the corrected truncated-normal PDF bound. | Implied this was the only meaningful defect.      |
| [REVIEW_SUMMARY.md](discrete/direct-solver/history/REVIEW_SUMMARY.md)             | Early review chronology.                          | Claims full original-code parity.                 |
| [SESSION_SUMMARY.md](discrete/direct-solver/history/SESSION_SUMMARY.md)           | Setup, datasets, and launch chronology.           | Early correctness and throughput assumptions.     |
| [TRAINING_STATUS.md](discrete/direct-solver/history/TRAINING_STATUS.md)           | In-progress pre-fix run snapshot.                 | Expected-result claims and pre-audit validation.  |


Historical claims of 95–100% expected paper accuracy conflict with the fidelity audit's quoted approximately 74.5% Maze-Hard exact accuracy. Historical throughput numbers also conflict across documents. These notes belong in a development-history appendix only.

# Continuous-space evidence

## C1–C4: pilot ladder and saturation diagnosis

**Source:** [repo-coupled C1–C4 ladder](continuous/c01-c04/continuous_prm_experiment_ladder_repo_coupled.md) and raw entry points in the [continuous evidence catalog](continuous/GENERATED_EVIDENCE.md).
**Status:** pilot/motivation; C1/C2 have serious-run data, while C3/C4 provenance is weaker than publication standard.

The common substrate is a 2-D point robot, a PRM, exact graph-Dijkstra cost labels, and a learned nonnegative detour residual on top of Euclidean distance.


| Stage                   | Main result                                                                                                                                                                      | Evidence limitation                                                                                                                           |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| C1 Euclidean sanity     | At B100, Euclidean success was already `0.925–1.000` over nine suites (`n=80` each), with `17.35–42.95` mean expansions and cost ratio exactly `1.0`; B200 saturated all suites. | The original suites were too easy to discriminate learned methods.                                                                            |
| C2 pooled bases         | At B100 over nine suites x 40 worlds, Euclid averaged `0.9667` success / `31.20` expansions; pooled HRM `0.9528 / 27.05`; pooled ON-LSTM `0.9778 / 22.92`.                       | Saturated success masks differences. HRM regressed rectangle OOD from `0.95` to `0.80`; no formal inference.                                  |
| C3 TaskLoRA experts     | On four ID anchors at B100, matched experts improved mean expansions only `1.14%` over HRM base and `3.03%` over ON-LSTM base; success was `1.0`.                                | Serious summary is orphaned at `runs/` root; configured C3 directory lacks result/checkpoint provenance. Oracle expert selection is post-hoc. |
| C4 nearest/RBF mixtures | At B100, HRM avgbase/RBF were `26.01/25.95` expansions and ON-LSTM `23.79/23.44`; nearest and RBF were nearly identical across sigma `0.5/1/2`.                                  | Summary CSVs have no accompanying raw rows, config, or inferential report.                                                                    |


**Learning.** The first benchmark was dominated by a strong Euclidean heuristic. Specialist routing added little beyond a pooled base. These stages justify C5's calibrated hard maps but should appear as pilot/appendix evidence, not as the paper's main quantitative claim.

## C5: calibrated hard maps expose a strong ON-LSTM result and an HRM failure

**Source:** [C5 hard-map specification and results](continuous/c05/continuous_prm_c5_hard_obstacle_encoder_spec.md) plus the canonical generated C5 significance report.
**Status:** completed Modal result with paired/corrected success evidence.

The final run used 192 PRM nodes with k=7, 160 training worlds, target 80 evaluation worlds per suite, 12 base epochs, 10 expert epochs, and budgets 128–168. Actual retained episodes were 80 maze, 79 dense, and 77 rooms.


| Suite at B144   | Euclid success | ON-LSTM avgbase | Delta  | Paired gains/losses | McNemar p  | BH q       | Mean expansion delta |
| --------------- | -------------- | --------------- | ------ | ------------------- | ---------- | ---------- | -------------------- |
| Hard maze       | 0.525          | **1.000**       | +0.475 | 38 / 0              | `7.28e-12` | `4.34e-11` | -66.987              |
| Hard maze-dense | 0.595          | **0.962**       | +0.367 | 29 / 0              | `3.73e-9`  | `1.92e-8`  | -24.519              |


Rooms showed a large practical curve (`0.052 to 0.922` at B128; `0.442 to 0.987` at B144), but its Euclidean baseline missed the preregistered 0.50–0.70 target band at the tested budgets and therefore was not a formal claim candidate.

**Negative result.** HRM training saturated at the residual cap (`loss 2.61219`, `MAE 3.11211`, `delta_mean = 4.0`); expert corrections were zero and planner rows matched Euclid. Lower-LR/cap-2 and differentiable soft-cap follow-ups reproduced the constant-cap collapse.

**Caveats and learning.** The ON-LSTM result is strong within one Modal run, but dense/rooms underfilled and oracle TaskLoRA is post-hoc diagnostic evidence only. C5 successfully solved the map-difficulty problem and localized HRM's failure to the objective/representation/optimization path, motivating C6.

## C6: value-field training rescues HRM in this setup

**Sources:** [C6 design](continuous/c06/design/continuous_prm_c6_heatmap_value_field_spec.md), [C6 results](continuous/c06/results/C6_RESULTS.md), and generated `c6_local_big1` / `c6_local_multi1` reports.
**Status:** canonical local representation/training recovery; one main training seed.

The field formulation predicts a goal-conditioned raster residual field and samples it at PRM nodes. The implemented input has eight channels: occupancy, reachable-free mask, clearance, goal Gaussian, start Gaussian, x, y, and normalized Euclidean distance. The design document's shorter channel list is stale.

The undersized diagnostic saturated success for every method, although the raster oracle cut expansions by 22–27%. The meaningful run used 96 training worlds, 40 evaluation worlds, 16 epochs, PRM 192/k7, and budgets 128/144/168.


| Hard maze at B144  | Success   | Delta vs Euclid | Corrected evidence         | Mean expansion delta | Rank correlation to PRM oracle |
| ------------------ | --------- | --------------- | -------------------------- | -------------------- | ------------------------------ |
| Euclid             | 0.625     | reference       | —                          | reference            | 0.894                          |
| HRM field          | **0.975** | +0.350          | p `0.000122`, q `0.000338` | -42.375              | 0.952                          |
| U-Net field        | 0.950     | +0.325          | q `0.000488`               | -38.050              | 0.968                          |
| ON-LSTM field      | 0.900     | +0.275          | p `0.00342`, q `0.00586`   | -31.475              | 0.908                          |
| Raster grid oracle | 0.950     | +0.325          | ceiling diagnostic         | -45.325              | approximate oracle             |


Multi-suite training retained maze HRM at `0.975`, lifted held-out dense maze from `0.825 to 1.000`, and rooms from `0.700 to 0.875` at B144. Dense/rooms were outside the formal Euclid target band, so these are directional generalization results.

**Caveats and learning.** The raster oracle is an approximate field diagnostic, not exact PRM Dijkstra or a minimal-expansion oracle. Runs are local RTX 5090, `n=40`, one training seed; publication-scale 160/80 and multi-seed confirmation were not run. The “Run 2—in progress” heading is stale. C6 supports “field training rescued HRM in this setup,” not “fields are intrinsically necessary”; C7 shows a well-trained scalar can also work. Grid size 56 exposes an unresolved U-Net skip-shape bug; 48/64 are safe.

## C7: integration comparison on static hard PRMs

**Sources:** [C7 design](continuous/c07/design/2026-06-27-c7-integration-comparison-design.md), [plan](continuous/c07/plans/2026-06-27-c7-integration-comparison.md), [results](continuous/c07/results/C7_RESULTS.md), and canonical generated `c7_local` reports.
**Status:** central canonical local static result.

The run compares Euclid, scalar HRM/ON-LSTM, field HRM/ON-LSTM/U-Net, focal variants, and exact graph-Dijkstra on three training families and three held-out families. It uses 96 training worlds/suite, 24 evaluation worlds/suite, 16 epochs, PRM 192/k7, and calibrated lower binding budgets 140 (maze/dense/rooms/spiral), 24 (bugtrap), and 56 (rooms-large).


| Suite       | Euclid success | Field-HRM success | Median expansion ratio [95% CI] | Expansion p | Success BH q | Matched n |
| ----------- | -------------- | ----------------- | ------------------------------- | ----------- | ------------ | --------- |
| Maze        | 0.583          | 1.000             | 0.521 [0.450, 0.627]            | <0.001      | 0.018        | 14        |
| Maze-dense  | 0.250          | 0.958             | 0.804 [0.707, 0.829]            | 0.031       | 0.001        | 6         |
| Rooms       | 0.375          | 0.958             | 0.829 [0.775, 0.885]            | 0.004       | 0.002        | 9         |
| Spiral      | 0.250          | 0.917             | 0.850 [0.742, 0.919]            | 0.031       | 0.001        | 6         |
| Bugtrap     | 0.458          | 0.750             | 0.714 [0.533, 0.800]            | 0.002       | 0.100        | 11        |
| Rooms-large | 0.417          | 0.750             | 0.839 [0.646, 1.222]            | 0.371       | 0.133        | 9         |


The field-HRM reduction spans about **15–48%**, but its corrected success is significant in four of six suites, not all six. Other learned arms provide corrected success evidence on bugtrap and rooms-large. Scalar HRM ratios (`0.427–0.822` across the six suites) are often equal or better than field, overturning the preregistered expectation that scalar HRM would fail.

**Integration result.** Additive learned heuristics beat focal variants against the loose Euclidean baseline. Scalar-HRM additive ratios are `0.427–0.822`, while best post-hoc focal `w=1.1` is roughly `0.789–0.977`; `w=1.0` is Euclid-equivalent. Additive path suboptimality is approximately `1.02–1.14`, trading optimality for search efficiency. This is the inverse of the discrete Manhattan result and supports a baseline-tightness-dependent integration principle.

**Caveats.** One GPU/seed, 24 worlds/suite, and matched n as low as 6. Expansion Wilcoxon p-values and preregistered comparison p-values are uncorrected; bootstrap intervals should lead. Choosing “best focal w” is post-hoc. The implemented field provider predicts an additive residual, despite stale design prose describing direct cost-to-go. “Hierarchical thesis holds” in the source should be discarded: HRM beats Euclid but not simpler learned backbones.

## C8: dynamics confirm learned guidance but not future-window advantage

**Sources:** [C8 design](continuous/c08/design/2026-06-27-c8-dynamics-design.md), [plan](continuous/c08/plans/2026-06-27-c8-dynamics.md), [results](continuous/c08/results/C8_RESULTS.md), and generated `c8_local_heavy` reports.
**Status:** central canonical local dynamic result; use the heavy run, not the initial or intermediate verdicts.

The heavy run uses deterministic moving circles, a space-time PRM state `(node,t)`, backward space-time Dijkstra labels, W=8 aware inputs versus W=0 present-frame twins, 53 usable training worlds, 20 evaluation worlds/suite, 12 epochs, and the full 1,129,536 scalar samples.


| Suite                    | Euclid to selected learned success | Median expansion ratio [95% CI] | Expansion evidence        | Success BH q | Matched n |
| ------------------------ | ---------------------------------- | ------------------------------- | ------------------------- | ------------ | --------- |
| Crossing, field U-Net    | 0.30 to 1.00                       | 0.258 [0.132, 0.769]            | p=0.062                   | 0.001        | 6         |
| Maze, field U-Net        | 0.30 to 1.00                       | 0.064 [0.046, 0.088]            | p=0.031                   | 0.001        | 6         |
| Maze-dense, field U-Net  | 0.05 to 0.75                       | 0.278                           | matched n=1; no inference | 0.001        | 1         |
| Rooms, field U-Net       | 0.30 to 1.00                       | 0.096 [0.038, 0.191]            | p=0.031                   | 0.001        | 6         |
| Rooms-large, field U-Net | 0.75 to 0.90                       | 0.380 [0.132, 0.591]            | p=0.013                   | 0.939        | 14        |
| Spiral, field HRM        | 0.20 to 0.90                       | 0.046 [0.038, 0.073]            | n<6                       | 0.001        | 4         |


The safe headline is that selected learned arms cut conditional matched expansions about **65–95%** and improve success `+0.2 to +0.7`; corrected success is strong on five suites, while rooms-large is supported mainly by expansion evidence. Additive again beats focal (`0.05–0.42` on strong cells versus about `0.76–0.99` for focal).

### Temporal-awareness spotlight

Aware/blind ratio below 1 favors the future-aware model. Across 24 suite/backbone cells, one uncorrected significant cell favors aware (maze scalar HRM, `0.823`, p `0.021`) and seven favor blind, including crossing scalar ON-LSTM `2.127`, maze field HRM `2.210`, and dense field U-Net `1.624`. These p-values are uncorrected.

Direct heuristic-accuracy analysis is more nuanced: aware has lower MAE in 11/24 cells and blind in 13/24; mean `MAE_aware - MAE_blind = +0.248` time steps. The correct claim is **no systematic accuracy or search benefit from the future window**, not that awareness is always harmful. Pooled `(node,t)` cells are not independent world-level replicates, and MAE measures calibration rather than search utility.

**Backbone result.** U-Net is strongest overall; HRM has the single lowest spiral ratio. The defensible conclusion is no recurrent/hierarchical advantage, not that U-Net wins every suite.

**Caveats.** Dense maze is outside a useful calibration band and has matched n=1; rooms-large Euclid success 0.75 is above the intended band. Field ON-LSTM and a success-aware composite remain open. The single-seed and single-cohort caveats are resolved by the 2026-07-23 additions below.

### Fixed-provider reanalysis and C8-R multi-seed replication (2026-07-23)

To remove the per-suite arm selection, a paper-motivated reanalysis fixes **one provider — field U-Net blind** (additive, binding budgets) for every suite ([analysis/c8_fixed_provider_reanalysis.py](analysis/c8_fixed_provider_reanalysis.py)). On the canonical 20-map cohort: success deltas `+0.20..+0.75`, all six paired map-level CIs exclude zero, median matched ratios `0.087–0.246`. On a **fresh 50-map/suite cohort** (generation seed 999999, frozen before any result was observed, provably map-disjoint by the eval-seed offset bound): deltas `+0.18..+0.84`, all six CIs exclude zero, ratios `0.047–0.273`, matched n up to 41 (dense maze n=1→3).

**Budget curves** ([design](continuous/c08/design/2026-07-23-c8-budget-curves.md), [analysis/c8_budget_curves_output.md](analysis/c8_budget_curves_output.md)): one eval pass at 4×binding on the frozen fresh cohort yields exact success-vs-budget curves by expansion thresholding (A* order is budget-independent; solve at B iff solve-expansions ≤ B). Cross-check PASSED: thresholding at binding reproduces every fresh-eval success exactly. Verdict: the headline is not an operating-point artifact — **Euclid needs 1.7–2.8× the binding budget to reach the blind provider's binding-budget success** (crossing 2.49×, maze 2.29×, dense 1.95×, rooms 1.73×, large 1.94×, spiral 2.80×), and all providers converge to the same feasibility ceiling (0.94–1.00) at 4×; on dense the 0.94 ceiling binds even the oracle (horizon-unsolvable maps). Artifacts: `runs/c8r_budget_curves/`; supp figure `fig_budget_curves.pdf`.

**Weighted-A\* control** ([design](continuous/c08/design/2026-07-24-c8-wastar-baseline.md), [analysis/c8_wastar_output.md](analysis/c8_wastar_output.md)): per-suite-tuned inflation $w_h\in[1.1,5]$ (tuned on development, evaluated once on the 50-map confirmation cohort). Verdict as-is: tuned WA\* is a far stronger classical baseline than the anchor (success 0.70–1.00), ties blind U-Net success on 4 suites and **beats it on crossing (−0.08)**; the learned heuristic **wins spiral outright (+0.30 [+0.18,+0.42])** and **expands fewer nodes on jointly solved maps in every suite** (median blind/WA\* ratios 0.217–0.982; ≤0.73 on 4/6); WA\* empirical suboptimality small on these instances (1.003–1.017), blind worse on crossing (1.164)/large (1.082). Claim narrowed accordingly in submission_v3. Artifacts: `runs/c8r_wastar/`.

**C8-R** ([design](continuous/c08/design/2026-07-23-c8r-multiseed-replication.md), [result](continuous/c08/results/C8R_MULTISEED_RESULT.md)) adds two full-pipeline retrains (seeds 2001/2002) evaluated on the same frozen cohort. Verdict: **success replicates — 17 of 18 seed×suite CIs exclude zero** (deltas `+0.10..+0.88`; sole miss is seed-2001 large-rooms, Euclid already 0.82). R1 pass (5/6, 6/6), R2 pass (5/6, 6/6), R3 fails as stated for seed 2001 — but the significant twin cells flip suites and sign across seeds (18 cells: 2 aware, 2 blind, 14 null; large rooms goes from blind-better −0.200 under seed 1234 to aware-better +0.080 under seed 2001), upgrading the future-window null to "per-suite effects are training-seed noise." Effort magnitudes are seed-stable on maze/rooms/spiral (medians `0.047–0.135` every seed) and seed-variable on crossing/dense/large-rooms (`0.216–0.625` across seeds); effort claims must be quoted as across-seed ranges on those suites. Artifacts: `runs/c8r_fresh_eval/`, `runs/c8r_seed2001{,_eval}/`, `runs/c8r_seed2002{,_eval}/`.

## C9: static few-shot transfer

**Sources:** [C9 design](continuous/c09/design/2026-06-29-c9-transfer-design.md), [plan](continuous/c09/plans/2026-06-29-c9-transfer.md), [results](continuous/c09/results/C9_RESULTS.md), and generated C9 comparison/significance reports.
**Status:** canonical local transfer result requiring world-clustered reanalysis.

The protocol adapts C7 pooled HRM/ON-LSTM bases to maze-dense, bugtrap, and rooms-large using K `{0,1,2,4,8,16,32}`, five adaptation seeds, and 30 fixed TEST worlds/target. It trained 540 models and produced 99,360 records.

Representative HRM curves:


| Target/method       | K1 ratio (success) | Higher-K result                       |
| ------------------- | ------------------ | ------------------------------------- |
| Maze-dense LoRA     | 0.650 (1.00)       | 0.730 (0.97) at K16                   |
| Maze-dense full-FT  | 0.744 (0.76)       | **0.571 (0.99)** at K16; 0.552 at K32 |
| Maze-dense scratch  | 1.008 (0.37)       | 0.808 (0.80) at K16                   |
| Rooms-large LoRA    | 0.771 (0.97)       | 0.809 (0.75) at K32                   |
| Rooms-large full-FT | 1.128 (0.37)       | **0.500 (0.92)** at K8; 0.489 at K32  |
| Bugtrap LoRA        | about 0.696 (0.83) | degrades to 0.950 (0.48) at K32       |
| Bugtrap full-FT     | 0.954 (0.37)       | 0.619 (0.77) at K16                   |


Zero-shot success versus Euclid is significant in all six target/backbone cells (`q = 0.000–0.025`). The most defensible result is that LoRA generally preserves a strong pooled base at K=1, scratch frequently has poor low-K success, and full-FT often reaches a lower ratio by K8–K16. It is not universal: ON-LSTM full-FT already beats Euclid in multiple K1 cells, some scratch cells are significant before K16, and LoRA can degrade at higher K.

**Inference caveat.** Repeated evaluations of the same 30 TEST worlds across five adaptation seeds are pooled as 150 observations. Publication inference should cluster/bootstrap at world level with adaptation seed nested within world. The report has no formal direct LoRA-versus-full-FT or transfer-versus-scratch contrast.

## C9h: matched-compute hardening and field transfer

**Sources:** [C9h design](continuous/c09h/design/2026-06-29-c9-hardening-design.md), [plan](continuous/c09h/plans/2026-06-29-c9-hardening.md), [results](continuous/c09h/results/C9H_RESULTS.md), and generated C9h reports.
**Status:** primary local source for the low-rank/full-rank mechanism.

C9h gives every trained arm 10 epochs at LR `2e-4`, splits bounded/unbounded LoRA, and adds field U-Net conv-LoRA. It uses K `{1,4,16}`, three seeds, 30 TEST worlds/target, 324 models, and 61,020 records.

- Bounded-minus-unbounded median expansion-ratio delta is `**0.000 +/- 0.008`** across 27 cells; 22/27 are exactly 0.000 and the largest listed absolute delta is 0.028.
- The strongest result is rooms-large field U-Net full-FT: zero-shot `0.982` at `0.67` success, K4/K16 full-FT `**0.404` at `0.97` success**. U-Net LoRA stays near `0.992–1.002` at `0.67` success.
- HRM exemplars show the capacity transition: maze-dense full-FT `0.805` at K1 to `0.591` at K16; rooms-large `1.083` to `0.490`; bugtrap `1.024` to `0.698`.

The mechanism claim is strong descriptively: the output bound contributes little at rank 8; low-rank structure best explains base-preserving flatness, while full-rank adaptation can exploit more data. It is still target/backbone dependent. Several ON-LSTM/U-Net K1 full-FT cells already beat Euclid, and there is no formal equivalence test for bounded versus unbounded.

**Caveats.** Three K values/seeds, one machine, repeated-world pooling, field zero-shot clamped to 4.0, and different scalar/field meanings of “bounded.”

## C9b: transfer under dynamics

**Sources:** [C9b design](continuous/c09b/design/2026-06-30-c9b-dynamics-transfer-design.md), [plan](continuous/c09b/plans/2026-06-30-c9b-dynamics-transfer.md), [results](continuous/c09b/results/C9B_RESULTS.md), and generated C9b reports.
**Status:** canonical local mechanistic result.

The run adapts six frozen C8 sources (three backbones x aware/blind) to three dynamic targets with K `{1,4,16}`, three seeds, and 20 TEST worlds. It trained 486 adapters and generated 30,600 records.

Two findings survive careful wording:

1. **Adaptation does not unlock future-window value.** At full-FT K16, `0/9` target/backbone cells have a positive aware-minus-blind success delta; deltas are `0.00` or `-0.05`. This is failure to observe an advantage, not an equivalence test.
2. **The static low-data crossover is out of regime.** One dynamic world yields about `192 nodes x ~140 time steps = 25k+` supervised states. Full-FT is no longer systematically catastrophic at K1, and LoRA can keep improving. The crossover is better interpreted as label scarcity, not a universal property of world count.

Representative maze-dense field-U-Net aware ratios are `0.145` zero-shot, `0.165` LoRA K1, `0.059` LoRA K16, `0.112` full-FT K1, and `0.101` full-FT K16. Success rises from `0.60` zero-shot to `0.97` LoRA K16. Scratch can show a low conditional ratio while solving far fewer worlds; transfer-versus-scratch should therefore lead with success.

**Caveats.** Generated tables contradict the authored claim that every zero-shot source significantly beats Euclid on success; rooms-large ON-LSTM-aware zero-shot has q `1.000` and U-Net-aware q `0.463`. The aware/blind probe and curves use different success aggregation conventions. Repeated-world pooling and one budget/target remain.

## C10: zero-label adapter interpolation

**Sources:** [C10 design](continuous/c10/design/2026-06-29-c10-interp-design.md), [plan](continuous/c10/plans/2026-06-29-c10-interp.md), [results](continuous/c10/results/C10_RESULTS.md), and generated bracketing/comparison/significance reports.
**Status:** completed local negative/ablation.

The protocol trains 16 rank-8 source experts across four maze-density and four room-scale source families, then targets three interior points without solved target labels.

**Positive machinery result.** RBF own-axis mass is `0.998` for the maze target, `0.986` for rooms t25, and `0.996` for rooms t35. Routing identifies the correct family axis.

**Planning result.** RBF-merge minus zero-shot expansion-ratio deltas are:


| Target/backbone     | Delta; negative favors RBF |
| ------------------- | -------------------------- |
| Maze / HRM          | -0.022                     |
| Maze / ON-LSTM      | -0.012                     |
| Rooms t25 / HRM     | -0.024                     |
| Rooms t25 / ON-LSTM | +0.082                     |
| Rooms t35 / HRM     | +0.012                     |
| Rooms t35 / ON-LSTM | +0.116                     |


Weight-space versus prediction-space deltas lie roughly between `-0.012` and `+0.007`, and uniform often matches or beats RBF. All learned ratios are below 1, but several maze success cells have BH q `0.062`; “all significant” is incorrect. There is no direct equivalence/noninferiority test, so the correct conclusion is **no consistent improvement observed**.

**Caveats.** Target descriptors are means over TEST-world geometry, making the method zero-label but transductive. The bracketing gate is a coordinate-wise own-axis bounding check, not full convex-hull membership. Only two smooth axes and a strong pooled base are tested. A CUDA device bug was fixed before the reported run and now has a GPU regression test.

## C11: compositional-mission headroom and completed architecture test

### Completed G0-H probe

**Sources:** [probe plan](continuous/c11/plans/2026-07-07-c11-headroom-probe.md) and [headroom results](continuous/c11/results/C11_HEADROOM.md).
**Status:** completed oracle-headroom result.

The probe uses 25 worlds/cell, one mission/world, K `{2,4,8}`, three configurations, and a per-cell binding budget calibrated on the leg-sum baseline.


| Config                   | K=2 ratio (n) | K=4 ratio (n) | K=8 ratio (n)  |
| ------------------------ | ------------- | ------------- | -------------- |
| A: maze waypoints        | 0.155 (6)     | 0.121 (3)     | **0.082 (20)** |
| B: rooms-large waypoints | 0.225 (12)    | 0.208 (4)     | **0.103 (16)** |
| C: maze keys/doors       | 0.144 (5)     | 0.128 (3)     | **0.084 (23)** |


All nine ratios clear the preregistered `<=0.5–0.6` gate, and every config declines monotonically with K. Success gaps are not monotonic. Low-K matched samples are only 3–12; budgets change with K; waypoint sampling favors separated reachable nodes; and the oracle is an upper bound, not a learned result.

### Completed main-grid evaluation

**Sources:** [approved design](continuous/c11/design/2026-07-07-c11-compositional-mission-design.md), [implementation plan](continuous/c11/plans/2026-07-07-c11-mission.md), and [canonical results](continuous/c11/results/C11_RESULTS.md).
**Status:** canonical local result; 198/198 main-grid checkpoints, 54,450 evaluation rows, scaled addendum still in progress.

The completed protocol has 11 cells, 40 TRAIN and 25 TEST worlds/cell, three training seeds, and six trained arms: MLP, FiLM U-Net, product-graph GNN, HRM trace, ON-LSTM trace, and HRM-v2 ACT. Evaluation also includes three reference heuristics and four forced-HRM-v2 providers. The manifest contains **33 checkpoints per trained arm** with no missing checkpoint. The raw evaluation table contains 4,950 rows per learned/forced arm and 1,650 rows per reference arm.


| Gate | Verdict | Evidence-safe reading |
| --- | --- | --- |
| K=0 continuity | **FAIL, diagnosed** | Not a stats/evaluation bug. K=0 reduces the task to C7 geometry but not the experiment: C7 had no MLP and did not give every architecture its C11 input. Real architecture separation plus several collapsed recurrent cells violates the all-tie premise. |
| G1 structure dose-response | **Negative** | No structured arm beats MLP at at least two of K `{2,4,8}` on one config with a non-decreasing gap. |
| G2a forced compute | **Negative / insufficient** | HRM-v2 k=1/2/4/8 expansion and MAE curves are nearly flat in every K=8 config. |
| G2b learned halting | **Negative, inverted** | Pooled Spearman `rho=-0.407`, permutation `p≈0.0005`, `n=675`; mean would-halt steps are 6.79/7.01/5.30 at K=2/4/8. |
| G3 closure | **Neither preregistered branch** | The experiment is neither architecture-agnostic nor hierarchy-positive: global-input architectures separate at shallow K, but no architecture converts mission depth into growing advantage. |


**Architecture result.** The U-Net and GNN—the arms with a global occupancy/product-graph view—show the clearest shallow-K advantages. The diagnosis reports versus-MLP expansion ratios around `0.67–0.87` at K in `{0,2}`; at A/K=0, state MAE is U-Net `0.151`, GNN `0.171`, and MLP `0.251`. By K in `{4,8}`, state error rises across every arm and the advantage disappears. C11 therefore rejects the intended hierarchy dose-response while also rejecting a blanket “all architectures tie” interpretation.

**Optimization pathology.** Five of 33 recurrent/ACT arm-cell combinations show constant-prediction collapse or partial seed collapse: HRM trace A8/C8, ON-LSTM B0, HRM-v2 B0, and two ON-LSTM A8 seeds. The largest reported recurrent deficits—HRM trace ratios `1.526` at A8 and `1.600` at C8—are collapse measurements, not clean expressiveness estimates. Removing K=0 padding makes recurrent predictions worse, so padding is not the cause; models used pad steps as extra recurrent compute.

**Statistical caveat.** The current C11 analyzer pools `world × model_seed` records in several intervals/tests. Because the same 25 TEST worlds repeat across three model seeds, publication-strength inference must aggregate or cluster at world level. The mechanical preregistered verdict is negative, but exact CIs/q-values should be refreshed before submission.

**Scaled addendum.** The separate 12-run `unet_film_big` versus `hrm_trace_big` addendum is now complete: 12/12 checkpoints, 2,700 evaluation rows, and 300 state-MAE rows. Scaling does not reverse the ordering. U-Net completion is `0.804` versus HRM `0.736` at K=2 and `0.413` versus `0.307` at K=8; mean state MAE is `0.395` versus `0.447` at K=2 and `0.906` versus `1.855` at K=8. At K=8, HRM exactly matches the leg-sum baseline's completion and mean expansions (`0.307`, `701.34`), consistent with persistent collapse rather than a recovered hierarchy advantage.

### C12 completed follow-up: persistent dynamics and iterative refinement

**Sources:** [C12 design](continuous/c12/design/2026-07-10-c12-persistent-hierarchical-planning-design.md), [implementation plan](continuous/c12/plans/2026-07-10-c12-persistent-hierarchical-planning.md), and [C12 results](continuous/c12/results/C12_RESULTS.md).
**Status:** complete. C12-A's frozen one-seed pilot is development-only and closes `strong_negative`; C12-B completed the full three-seed A/C × K2/K8 TEST grid after K16 failed its construction gate.

C12-A's frozen full G0 probe supplies 800 episodes and 3,200 matched provider rows. Constructed decision-relevant aliases occur at `71.3%` of paired decisions; the privileged mode/history diagnostic gains `+0.467` completion over frozen present-state prediction and reduces collision-adjusted regret by `65.1%` (world-bootstrap 95% CI `63.0–66.9%`). Oracle completion is `97.5%`, with a `75.1%` ceiling gap, while present-sufficient history headroom is exactly `0.000`. This establishes memory-relevant headroom, not a learned hierarchy win. The completed VALIDATION-selected one-seed pilot fails G1-A forecast, G2-A planning, and G3-A carry; G4-A is `strong_negative`: matched temporal hierarchy adds no value even when history is necessary.

C12-B tests the other major C11 substrate objection: insufficient propagation depth. K16 was dropped before training because only 2/300 A worlds and 1/300 C worlds yielded valid missions (20 required), despite 166–188-hop transition distances and oracle/leg-sum ratios near `0.06–0.07`. The authorized K2/K8 grid trained four graph arms across three seeds (48 checkpoints) and produced two complete 3,200-row TEST tables. The tied refiner's expansion burden improves monotonically from cycle 1 to 8 in every cell: A/K2 `0.859→0.815`, A/K8 `0.624→0.576`, C/K2 `0.867→0.777`, and C/K8 `0.576→0.517`. All cycle-1/8 world-bootstrap intervals separate.

The registered G1-B still fails because the improvement does not grow with K: A's K8-minus-K2 gain is `+0.0038` (95% CI `-0.0199, 0.0281`) and C's is `-0.0305` (`-0.0726, 0.0096`). G2-B passes locally on C/K8: tied cycle 8 beats shallow by `0.0177` normalized burden and untied cycle 8 by `0.0076`, both BH `q=0.000067`, with no completion loss. It does not replicate on A/K8, where untied is better by `0.0031` (`q=0.0168`). The registered G3 closure is therefore negative for the hierarchy-depth claim. The nuance matters: recurrent compute helps bounded search, but not preferentially at greater mission depth, and weight tying is config-dependent.

Value diagnostics further narrow the claim. State MAE improves in three of four cells but slightly worsens on C/K8; Bellman residual worsens from cycle 1 to 8 in every cell. Solved cycle-8 paths average roughly `1.4–2.2%` above oracle cost. C12-B supports a bounded-search-efficiency signal under an inadmissible heuristic, not uniform value-iteration improvement or optimal planning. Integrity is complete: 48/48 checkpoints, 3,200/3,200 expected state and evaluation rows, zero seed overlap, independent reanalysis pass, and 217 green C8/C11/C12 regression tests.

## C13: bounded current-observation local-Bellman heuristic

**Sources:** [C13 design](continuous/c13/design/2026-07-16-c13-state-conditioned-heuristic-design.md), [initial audit](continuous/c13/results/C13_INITIAL_AUDIT.md), [pipeline smoke](continuous/c13/results/C13B_ROLLOUT_RANKER_SMOKE.md), [identifiability study](continuous/c13/results/C13B_IDENTIFIABILITY_STUDY.md), [independent-certifier gate](continuous/c13/results/C13C_CERTIFIED_SEARCH.md), [shared-queue oracle gate](continuous/c13/results/C13D_SHARED_QUEUE_ORACLE.md), [shared-queue exact-target gate](continuous/c13/results/C13E_SHARED_QUEUE_EXACT_TARGET.md), [current-state literature/target decision](continuous/c13/design/2026-07-17-c13-current-state-literature-and-next-target.md), [C13-M preregistration](continuous/c13/design/2026-07-17-c13m-matched-quality-confirmation.md), [C13-F through C13-M canonical result](continuous/c13/results/C13F_M_CURRENT_STATE_RESULT.md), [C13-N preregistration](continuous/c13/design/2026-07-17-c13n-hrm-substitution.md), [C13-N result](continuous/c13/results/C13N_HRM_SUBSTITUTION_RESULT.md), [C13-O preregistration](continuous/c13/design/2026-07-17-c13o-hrm-summary-last-alignment.md), [C13-O result](continuous/c13/results/C13O_HRM_ALIGNMENT_RESULT.md), [C13-P preregistration](continuous/c13/design/2026-07-19-c13p-persistent-search-state.md), [C13-P implementation plan](continuous/c13/plans/2026-07-19-c13p-persistent-search-state.md), and [C13-P result](continuous/c13/results/C13P_PERSISTENT_SEARCH_RESULT.md).
**Status:** complete benchmark-level current-state result; C13-M primary matched-quality and bounded-control gates pass, while C13-N substitution, C13-O summary-last, and C13-P persistent-state development gates fail. C13-P is mechanically valid and contributes a completed negative mechanism result for its frozen instantiation.

C13-A establishes the semantic boundary. Literal `constant-E` cancels when inserted as C6's additive residual, and the admissible one-step Euclidean backup saves only about one expansion. Deeper backups gain useful guidance only by traversing multiple graph layers, crossing the methodological boundary raised in the professor meeting.

C13-B replaces graph-shortest-path supervision with three independent fresh-start local-policy returns per node. The collector resets visit history at every labeled start, takes the median successful return rather than the minimum, and records that no shortest-path result is a feature or label. Its three-world smoke showed negative/near-zero held-out correlations and no stable matched-FOCAL gain.

The completed diagnostic shows that the smoke conflated world diversity with identifiability. A flat MLP's held-out Pearson correlation rises from `-0.036` at three training worlds to `0.561` at six and `0.675` at twelve; HRM reaches `0.750`. Five-neighbor compact-feature regression reaches `0.729`, and cross-world nearest-feature target gaps are `0.516x` random gaps. Bounded local observations therefore contain substantial transferable information, though global maze topology remains partially aliased.

The behavior target is an independent blocker. Ten-rollout split-half reliability is Pearson/Spearman `0.646/0.831`, but behavior costs are a median `3.30x` graph oracle and the exact ten-rollout aggregate worsens `w=1.25` FOCAL by `+5.33` expansions, losing on five of six worlds. More model capacity cannot repair a target whose exact value fails the intended search.

Representation and integration both matter. Padded ON-LSTM has held-out correlation `0.006`, while a separately trained trimmed version reaches `0.489`; padded HRM reaches `0.750` partly by treating sequence length as an implicit degree cue. At the original `w=1.10`, HRM `g+h_hat`, exact rollout `g+h`, and oracle `g+h*` are respectively `+3.50`, `+6.33`, and `+1.33` expansions versus matched Euclidean FOCAL. Yet HRM and trimmed ON-LSTM used as unsafe primary A* heuristics save `64.0` and `62.2` expansions, while raising mean path cost by `14.7%` and `7.0%`. The immediate failure is therefore narrow secondary-key integration; target alignment and explicit mask/degree representation remain separate repairs.

C13-C tests the smallest no-retraining repair: arbitrary-rank A* finds an incumbent, then a completely fresh consistent-Euclidean A* proves `incumbent <= w * min_OPEN(g+h)`. At `w=1.10`, the privileged oracle ceiling averages `23.33 + 117.83 = 141.17` total expansions against `129.67` for matched Euclidean FOCAL, losing all six worlds; even ordinary Euclidean A* is slightly cheaper at `139.17`. All `90/90` provider/world/bound rows certify with zero post-hoc violations. At `w=1.25` the oracle wrapper wins on average (`-13.17` expansions) but only on four of six worlds, missing the declared stability gate. The causal conclusion is integration-first: a fresh certifier throws away phase-1 work. Exact rollout inconsistency and model representation remain downstream issues, not explanations for the oracle failure.

C13-D changes only the integration. A one-anchor/one-rank shared-path search uses a single `g`/parent state, Euclidean fallback, synchronized queue updates, and the same direct anchor certificate. At `w=1.10`, it averages `117.83` anchor plus only `5.00` oracle-rank expansions, totaling `122.83` against `129.67` for matched FOCAL. It wins all six worlds, saves `18.33` expansions versus C13-C, returns optimal paths, and records zero cross-queue duplicate expansions or proof failures. The immediate integration ceiling is therefore repaired; this privileged result does not yet establish exact-rollout or learned-provider utility.

C13-E freezes the successful shared search and changes only the rank to C13-B's replayed exact rollout statistic. At `w=1.10`, all six paths certify with zero proof, path, accounting, or post-hoc bound failures, but total work is `131.00` expansions against `129.67` for matched FOCAL: 2 wins, 1 tie, and 3 losses. The exact arm is still `241.50` expansions cheaper than C13-C's duplicated wrapper, confirming the integration repair. Its start-state value is `4.23x` graph-optimal on average and the rank queue is chosen on only `2.42%` of checks, isolating target scale/alignment as the next blocker. Label coverage is `97.22%`, so missing fills are not the dominant explanation; representation is untested because no model is loaded.

C13-F shows that monotone scale calibration alone does not rescue the behavior-return rank. Exact rollout remains `+1.33` expansions worse than matched FOCAL even though it beats the same-search Euclidean rank by `7.17`.

C13-G tests two exact radius-bounded local-escape constructions. At radius 0.20, the pure local heuristic averages `136.50` expansions and loses all six FOCAL comparisons; the exit-stub variant improves to `133.00` but wins only 1/6. Both are valid and local, but too weak under bounded secondary insertion.

C13-H replaces single-step analytical escape with iterative local heuristic Bellman learning. A one-suite flat MLP passes untouched 192- and 211-node maze confirmations under bounded FOCAL: paired deltas `-4.42` and `-6.92`, with confidence intervals excluding zero and zero safety failures.

C13-I then performs the necessary live C7 comparison. The one-suite arm fails across the full distribution: `98.29` versus `82.93` field-HRM expansions, delta `+15.36` with CI `[+12.02, +18.64]`, and only one negative suite mean. Six secondary field-U-Net historical rows drift; primary field-HRM parity is exact and the drift is retained in the failed integrity flag.

C13-J retrains on 96 suite-balanced worlds but static insertion remains `+16.17` expansions worse on a disjoint 24-world development block. Distribution repair alone is not enough.

C13-K adds one radius-bounded Bellman backup at inference. With the same iteration-8 model, the development delta changes from `+16.17` to `-1.21`, negative in four suites, although its CI `[-8.04, +5.46]` still crosses zero. C13-L therefore calibrates alpha on 48 new worlds. Alpha 1.50 gives `69.98` versus `83.02` field-HRM expansions, delta `-13.04` with CI `[-18.67, -7.65]`, and all six suite means negative. It officially fails the absolute 1.10 maximum-cost gate; comparator-relative mean/max ratios are nevertheless closely matched (`1.0381/1.3660` current versus `1.0418/1.3533` field HRM).

The post-hoc reopening probe does not remove the direct-search tail. Reopening-first-goal adds work and retains outliers; reopening FOCAL is safe (max ratio `1.0639`) but much slower. This justifies separate direct matched-quality and bounded operating points.

C13-M freezes iteration 8, radius 0.20, alpha 1.50, and the no-reopen C7 A* ordering before generating 24 worlds from each of six suites. Across 144 zero-overlap worlds, current averages `68.306` expansions versus `81.264` for field HRM: paired delta `-12.958`, bootstrap 95% CI `[-16.299, -9.743]`, with 109 wins, 3 ties, and 32 losses. Every suite mean is negative. Mean/max cost ratios are `1.02353/1.16244` current versus `1.03111/1.33459` field HRM. All 144 paths validate, and the separate `w=1.10` safety control has zero bound or certificate violations.

All C13-M implementation, preregistration, checkpoints, 144 feature caches, 1,296 raw rows, summaries, report, manifest, and suite shards are hashed; seed overlaps, duplicate keys, invalid paths, and live-feature/cache mismatches are all zero.

Runtime is the main qualification. Current feature construction averages `5.138` seconds per world in the unoptimized Python harness versus `0.371` for field HRM; model inference (`0.0008` s), local backup (`0.0097` s), and search (`0.0002` s) are cheap. C13-M establishes a search-expansion/path-quality result under a stricter information boundary, not end-to-end latency superiority.

C13-N then performs the architecture-only substitution requested after C13-M. On the frozen 24-world development block, iteration 8/alpha 1.50 HRM averages `67.292` expansions versus `75.917` field HRM, delta `-8.625` with CI `[-16.667, -1.208]`, but only maze, rooms, and bugtrap have strictly negative suite means. Against the matched flat model, HRM is `-2.125` expansions with CI `[-5.833, +1.792]`; its mean/max cost ratios are `1.02545/1.10243` versus `1.01314/1.03835` flat. The suite-balance and matched-flat quality conditions fail, so the untouched 144-world seed-offset-20M block is not generated.

The diagnostic narrows the mechanism. The local backup improves HRM rank correlation and produces a significant pooled field-HRM gain, so integration is not fundamentally incompatible. Byte-identical features also support the successful MLP, so raw information absence is insufficient as an explanation. The likely mismatch is the final recurrent readout over an artificial sequence—summary first, angular rays, then edge-length-ordered actions—combined with zero-initialized HRM state for every node rather than persistent planning-time hierarchy. Recorded HRM training and CPU inference are roughly `794x` and `415x` the flat implementation, respectively.

C13-O tests the smallest readout repair while preserving identical initialization, targets, optimizer, and caches. Summary-last significantly improves trimmed HRM at iteration 6: deltas `-3.625` and `-3.875` expansions for alpha 1.00/1.50, with CI upper endpoints below zero and matched path quality. Neither cell robustly beats field HRM. At the fixed iteration-8/alpha-1.50 endpoint, summary-last averages `69.375` expansions versus `75.917` field HRM (delta `-6.542`, CI `[-13.251, -0.333]`) but again improves only three suites. It is `+2.083` expansions worse than trimmed HRM with an inconclusive CI and exceeds flat-relative mean/max cost margins. No cell passes the complete method gate and confirmation remains untouched.

The readout result is therefore partial and checkpoint-dependent. Summary-last endpoint correlation still rises from `0.7813` to `0.8631` after local backup, confirming functional integration, but endpoint validation MAE worsens to `0.11112` versus `0.10056` for trimmed HRM. Order matters, yet it neither stabilizes the moving-target loop nor supplies the absent planning-time hierarchy.

C13-P is the preregistered response to the mechanism gap C13-N and C13-O leave open: both applied HRM independently per node with state recreated for every example. The frozen design initializes one carry per planning query, updates it after each expansion event, and compares persistent versus reset modes of one checkpoint, with static C13-M as baseline. After named mechanical repairs and a fresh fingerprint, G0-P passes. G1-P fails in the harmful direction: persistent-minus-reset MRR is `-0.0294`, 95% CI `[-0.0598, -0.0030]`, top-1 delta `-0.0280`, and only 3/6 suites are positive. Descriptive G2-P also fails its reset/C13-M uncertainty, suite-robustness, and path-quality requirements. Post-hoc raw-logit inspection finds severe persistent-only long-sequence growth and candidate-score collapse, but that diagnostic does not alter the frozen null.

**Decision.** Treat C13-M as the completed local result with explicit Pareto framing: the alpha-1.50 direct arm is the confirmed matched-quality expansion result, while the `w=1.10` reopening-FOCAL arm is the slower bounded control. Treat C13-N, C13-O, and C13-P as negative architecture/mechanism diagnostics, not contradictions of C13-M or general HRM rejections. Do not retune these development blocks, self-bootstrap C13-P, or open confirmation. If persistent state is pursued, preregister a new study that separates causal-state sufficiency from recurrent-state stability and requires a stable offline persistent-over-reset signal before free-running integration. Before external publication, optimize feature construction, add model-seed and cohort replication, and review the exact known-PRM/current-local-subgraph wording with the professor.

## C14: label-count × world-diversity factorial — H-C14 rejected; diversity governs

**Sources:** [design + amendment v2](continuous/c14/design/2026-07-23-c14-label-density-factorial.md), [result](continuous/c14/results/C14_RESULT.md), [analysis output](analysis/c14_analysis_output.md).
**Status:** complete (2026-07-23, Modal L4 fleet); preregistered verdict reported as-is.

180 step-matched adaptations (5 N × conc/8×-dist per amendment v2 × static C9/dynamic C9b maze-dense × LoRA-r8-unbounded/full-FT/scratch × 3 seeds; 2,560 optimizer steps every cell) on the frozen 30/20-map test cohorts. **H-C14 rejected as stated:** on matched-solved ratios full-FT is at or below LoRA at every tested N in all 12 (domain×div×seed) cells — the hypothesized crossover sits below N=256 if anywhere — and the preregistered full_ft×log N interaction is null (−0.0025 [−0.0087, +0.0028]). **The preregistered diversity readout localizes the real effect:** concentrated low-world cells collapse full-FT/scratch success vs zero-shot in BOTH domains (dynamic conc N≤1024: −0.300 [−0.550,−0.050] all seeds; static conc N=256: −0.167..−0.300) while 8×-distributed cells at the SAME N rescue them (dynamic +0.15..+0.20; dist−conc full-FT paired deltas +0.30..+0.48 dynamic, +0.233 static); **LoRA never collapses in any of the 60 cells**; dynamic conc recovers exactly where w_min forces 4 worlds (N=65,536: +0.100 all seeds). Evidence-safe revision: the protective variable at low supervision is world diversity or adapter-restricted capacity, not raw supervised-state count — in K-indexed protocols (C9/C9b) map count and state count co-move, and C14 separates them. Caveats: dynamic matched-ratio n=1 (Euclid 1/20 at binding — dynamic inference rests on paired success), one target family per domain, 3 seeds. Per the frozen design, the submitted paper's claims were not edited in response; C14 reports in the supplement (Section L).

# Cross-space evidence and synthesis

## HRM head-to-head: repairing attention does not improve held-out planning

**Source:** [HRM_HEADTOHEAD.md](cross-space/HRM_HEADTOHEAD.md)
**Status:** completed local negative ablation; one seed/variant.

The discrete reproduction exactly recovers the focal pilot at two-decimal precision: A128 ratio `0.85`, success `0.62 to 0.75`; A192 ratio `0.94`, success `0.75 to 0.75`. No repaired discrete model was trained.

The continuous head-to-head trains the incumbent and a repaired cross-token-attention model on the same 46,079-row pooled set and evaluates 30 deterministic worlds/target:


| Model             | Maze-dense ratio @ success | Rooms-large ratio @ success |
| ----------------- | -------------------------- | --------------------------- |
| Incumbent HRM     | **0.6501 @ 1.000**         | **0.7714 @ 0.967**          |
| Original repaired | 0.7185 @ 0.933             | 0.8611 @ 0.867              |


Remediation improves training loss, but not robustly both targets. V1 reaches `0.6311 @ 1.000` on maze and regresses to `1.0417 @ 0.733` on rooms. Later variants are also worse on rooms. Training-loss improvement therefore does not imply search generalization.

**Caveats.** One training seed, two held-out targets, no direct confidence interval/test, and a post-hoc multi-variant sweep without multiplicity control. Several “parameter-matched” variants are 29% smaller despite prose calling them within approximately 25%. The result supports a formulation/overfit diagnosis, not a general claim that attention or hierarchy is harmful.

## Program audit: three harness layers and seven repeated patterns

**Source:** [PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md](cross-space/PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md)
**Status:** canonical interpretation; not independent numerical evidence.

The audit identifies three harness layers where an apparent model failure can be created:


| Layer               | Observed problem                                                               | Corrective evidence                                                                                                                             |
| ------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Representation      | C5 scalar HRM collapses to a constant.                                         | C6 field training restores useful HRM predictions.                                                                                              |
| Planner integration | Discrete ranker is suppressed/misdirected as additive magnitude.               | Focal tie-breaking recovers a 6–15% expansion win with the same weights.                                                                        |
| Task formulation    | Sequence/hierarchical models receive a local, pre-digested regression problem. | Blind one-token controls tie/beat aware models; U-Net's global view is often strongest; repaired attention fits better but does not generalize. |


Its seven cross-space patterns remain useful discussion hypotheses:

1. Learned gains are mainly efficiency margins.
2. Representation and integration failures can masquerade as model failures.
3. Pooled bases generally outperform specialist adapters.
4. Bounded corrections are safe but weak.
5. Architecture separation appears on forecasting, then vanishes on local heuristic regression.
6. Transfer benefits track supervision scarcity.
7. Cheap/local signals saturate the current substrate.

The audit correctly identified the scalar-regression formulation as a likely bottleneck, but completed C11 and C12 now sharpen the conclusion. Composition, non-local structure, an explicit MLP, global U-Net/GNN views, faithful HRM-v2 ACT, persistent hidden-regime memory, and tied full-graph recurrent refinement still produce no general hierarchy/depth dose-response. Architecture is not irrelevant: global inputs separate at shallow K, C12-B cycles improve bounded search in every cell, and tied refinement wins a fully controlled C/K8 comparison. Yet C12-A's learned pilot is `strong_negative`, C12-B's cycle gain is no larger at K8 than K2, and tying loses to untied compute on A/K8. The remaining opportunity is planner/objective structure rather than another backbone substitution.

## Repository validation gates

**Source:** [VALIDATION_GATES.md](cross-space/VALIDATION_GATES.md)
**Status:** engineering/reproducibility snapshot, not a scientific result.

The recorded gate has **194 passed, 0 failed**, with four HRM-v2 tests skipped because flash-attn was unavailable. The suite is CPU/oracle dominated: HRM-v2 59 passed, discrete 84, and continuous 51. Two H100s would not materially accelerate it; persistent label caching was estimated to save 50–70%.

This snapshot predates C11 and says the HRM head-to-head was pending, so it must not be used as current validation coverage for those later artifacts.

# Validation assessment

## Overall assessment: share internally; paper claims require targeted reanalysis

The evidence is coherent enough to serve as the master source for paper scaffolding. It is **not yet publication-ready as a quantitative results section** because several central comparisons reuse worlds across seeds, several stages have one training seed, and direct method-versus-method uncertainty is missing.

### Calculation and source spot-checks


| Claim checked                  | Independent check                                                                                                                                                         | Result                                          |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| C5 maze B144                   | Raw summary: Euclid `0.525`, ON-LSTM `1.000`, 80 episodes, mean expansions `137.375` vs `70.388`.                                                                         | Verified.                                       |
| C5 dense B144                  | Raw summary: Euclid `0.59494`, ON-LSTM `0.96203`, 79 episodes.                                                                                                            | Verified.                                       |
| C6 maze B144                   | Raw summary: Euclid `0.625`, HRM `0.975`, U-Net `0.950`, ON-LSTM `0.900`; HRM expansions `94.075` vs `136.45`.                                                            | Verified.                                       |
| C8 heavy selected ratios       | Raw summary reproduces maze field-HRM-blind `0.0537`, rooms field-HRM-blind `0.0802`, spiral field-HRM `0.0458`, and rooms-large field-U-Net-blind `0.228`.               | Verified; matched n ranges 4–15 in these cells. |
| C9h bound and field result     | Raw curves reproduce maze HRM zero-shot `0.6501`, bounded K1 `0.6539`, and the rooms-large U-Net full-FT `0.404` result reported by the source.                           | Verified.                                       |
| C11 dose response              | Recomputed from `runs/c11_probe/c11_probe_records.csv` at the documented binding budgets. Rounded medians are A `0.15/0.12/0.08`, B `0.23/0.21/0.10`, C `0.14/0.13/0.08`. | Verified against the more precise report table. |
| C11 main-grid completeness     | Manifest grouping gives 198 checkpoints: 33 each for MLP/U-Net/GNN/HRM trace/ON-LSTM trace/HRM-v2. Raw evaluation grouping gives 54,450 rows with complete arm/config/K/budget/seed coverage. | Verified; statistical reanalysis still needed. |
| Discrete clean/multitask means | Direct raw JSON review reproduces the compendium values; undefined diagnostic fields occur in some rows while headline outcomes remain finite.                            | Verified with diagnostic filtering caveat.      |


### High-priority issues before submission

1. **Recompute C9/C9h/C9b/C10 uncertainty at the TEST-world level.** Adaptation seeds are repeated measurements nested within the same worlds. Report world-clustered bootstrap intervals and direct paired contrasts. *Partially completed 2026-07-20 for the paper-quoted key cells ([analysis](analysis/WORLD_CLUSTERED_REANALYSIS.md)): every verdict survives; the C9 crossover, scratch penalty, rooms-large early catastrophe, and bugtrap LoRA degradation gain world-clustered CIs; C9h's 0.000 bound delta reproduces; C9b softens to "no significant aware advantage (8/9 point deltas ≤0, all CIs cross zero)"; C10 sharpens to significantly-worse-in-2/6-cells. Full-grid refresh of every published table remains.*
2. **Add formal method-versus-method contrasts.** Success-versus-Euclid q-values do not establish LoRA versus full-FT, aware versus blind, or RBF versus zero-shot equivalence/superiority.
3. **Separate success evidence from solved-only expansion ratios.** Especially in C8 dense, C9b scratch, and any cell where methods solve different worlds.
4. **Run publication-scale replication for C7/C8.** More training seeds and evaluation worlds are more valuable than additional architecture variants on the same split.
5. **Recompute C11 inference at the TEST-world level.** The main grid and 12-run scaled addendum are complete, but repeated model seeds are nested within 25 TEST worlds; publication intervals must cluster or aggregate at world level. *Partially completed 2026-07-20 ([analysis](analysis/WORLD_CLUSTERED_REANALYSIS.md)): the G2b halting inversion strengthens at world level (Spearman −0.578, stratified permutation p=5×10⁻⁵, n=225 vs record-level −0.407); the shallow-K global-input advantage is world-clustered-significant at K∈{0,2} in configs A/B but absent in C; every K∈{4,8} cell includes parity. Full-table refresh remains.*
6. **Expand the discrete focal pilot.** Run the full suite/budget matrix with paired uncertainty before elevating the 6–15% result to a headline contribution.
7. **Correct remaining document arithmetic and stale labels.** In particular HRM-v2 sample exposure, C6 “in progress,” old C11 addendum status, pre-C13-M blocker language, and outdated universal-success wording.
8. **Harden C13-M rather than retune it.** Optimize or lazily construct the bounded features, replicate independent model seeds and a second 144-world cohort, and test larger densities. Preserve the distinction between the unbounded matched-quality direct arm and the slower bounded FOCAL control.

### Required caveats in any current external draft

- Most continuous headline results are local RTX 5090 validations with one primary training seed.
- Discrete HRM/LSTM and learned-heuristic comparisons generally lack formal uncertainty.
- The same trained bases and TEST worlds connect many stage results; stages are not independent replications.
- Architecture findings are conditional on input representation, information access, training recipe, and planner integration.
- Null results are failures to observe improvement, not equivalence unless explicitly tested.
- C11 contributes a completed negative architecture/depth result, but its pooled `world × seed` uncertainty needs world-clustered reanalysis; the scaled addendum is complete.
- C13-M contributes a one-model-seed, one-cohort expansion/path-quality result. Its direct arm is not formally bounded, its known-PRM feature builder is not an unknown-map sensor policy, and its unoptimized wall time is slower than field HRM.
- C13-P contributes a completed valid negative mechanism result for its frozen target/representation/model/training protocol: G0-P passes, G1-P fails against same-checkpoint reset, and no self-bootstrap or confirmation was run.

# Paper-scaffolding implications

## Evidence-safe working thesis

> Learned heuristics can substantially reduce search effort and transfer across planning families, but the benefit depends more on representation, planner integration, information access, and supervision regime than on nominal model hierarchy. Strong pooled priors are robust; low-rank specialization, explicit future-window inputs, and adapter interpolation add little once the cheap signal is captured. Even with large compositional oracle headroom, C11 finds shallow global-input advantages but no hierarchy/depth dose response. C13 demonstrates the constructive counterpart: after exact rollout, shallow local targets, one-suite transfer, distribution-only training, and reopening each fail distinct controls, a suite-balanced bounded-observation model plus one local Bellman backup confirms a 15.95% expansion reduction over complete-map field HRM at better empirical path quality. That result is not a formal bound or wall-clock speedup.

## Candidate contributions

1. **A calibrated continuous-PRM benchmark ladder** that moves from saturated pilots to hard static and dynamic search regimes with paired success/expansion evaluation.
2. **A cross-domain integration principle:** additive learned magnitude is effective against a loose Euclidean baseline, while focal ranking rescues a good-but-miscalibrated signal when the admissible baseline is tight.
3. **A transfer-regime result:** pooled priors improve low-label adaptation; rank-8 LoRA preserves the base but is capacity-limited, while full-rank adaptation can exploit richer supervision.
4. **Two careful negatives:** future-window awareness provides no consistent heuristic benefit, and descriptor-weighted adapter composition provides no consistent gain over a strong pooled base.
5. **A validity/formulation analysis:** hierarchy cannot be judged fairly when model mechanisms are disabled, inputs are pre-digested, or the task is locally MLP-complete.
6. **A controlled compositional architecture result:** C11 pairs large oracle headroom with explicit MLP/U-Net/GNN/recurrent/ACT controls and finds architecture-dependent shallow differences but no depth dose-response.
7. **A current-state/local-computation result:** C13-M beats every fixed C7 learned provider in pooled expansions without complete-map or shortest-path supervision, while separating the direct matched-quality and bounded-safety operating points.

## Recommended results narrative

1. Establish saturation and calibration (C1–C5).
2. Show representation/training recovery (C6), then the more general static integration result (C7).
3. Extend to dynamics and present the temporal negative with its MAE mechanism (C8).
4. Present transfer curves and matched-compute mechanism (C9/C9h), then explain why dynamics change the label regime (C9b).
5. Present interpolation as a clean null conditioned on plateaued experts and a strong base (C10).
6. Use the discrete focal result as a cross-domain integration counterpoint.
7. Close the hierarchy audit with C11 and C12: oracle/memory headroom is real, but C11 has no depth response, C12-A closes `strong_negative`, and C12-B's monotone compute gain lacks a K-dose response; retain the localized C/K8 tied-control win as a bounded secondary result.
8. Close with C13's controlled failure-to-success sequence: exact rollout and shallow local targets fail, one-suite and distribution-only repairs fail, local Bellman integration recovers the signal, and C13-M confirms fewer expansions than complete-map C7 providers with explicit path-quality, bound, and runtime caveats.

## Candidate figures and tables


| Item     | Content                                                                                                  | Readiness                                             |
| -------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Figure 1 | Program map: forecasting -> heuristic regression -> static/dynamic transfer -> compositional missions.   | Ready from this synthesis.                            |
| Figure 2 | C5/C6/C7 representation and integration progression, with success and expansion panels.                  | Needs consistent reanalysis/plotting from raw rows.   |
| Figure 3 | C8 dynamic learned-versus-Euclid and aware-versus-blind results, separating success from matched ratios. | Raw data available; add world-level uncertainty.      |
| Figure 4 | C9h K curves for LoRA/full-FT/scratch across targets/backbones.                                          | Raw curves available; add direct clustered contrasts. |
| Figure 5 | C10 RBF mass plus method deltas relative to zero-shot.                                                   | Raw data available; avoid equivalence wording.        |
| Figure 6 | C11 oracle/leg-sum headroom plus learned-arm versus MLP ratios by K, with collapsed cells marked.       | Raw data/report ready; redo learned intervals at world level. |
| Figure 7 | C12-B tied-refiner cycle curves at K2/K8 plus shallow/untied controls and path-cost ratios.              | Canonical three-seed world-clustered tables are ready.         |
| Figure 8 | C13-F–M mechanism ladder plus C13-M per-suite paired expansion deltas and empirical path-quality frontier. | Raw paired rows, preregistered intervals, and canonical table are ready. |
| Table 1  | Discrete forecasting and learned-heuristic evidence, with protocol/sample caveats.                       | Ready descriptively.                                  |
| Table 2  | Claim-audit matrix: reported wording, supported wording, evidence tier.                                  | Ready from this document.                             |
| Appendix | Smoke/provenance catalog, HRM-v2 audit history, validation gates, incomplete arms.                       | Ready after link check.                               |


## Further questions

1. After world-clustered reanalysis, which C9/C9h method differences remain outside practical-equivalence margins?
2. Does the C8 temporal negative persist across independent training seeds and a field ON-LSTM arm?
3. Does descriptor-weighted interpolation help when the pooled base is deliberately narrowed or experts actually exceed zero-shot?
4. Can discrete focal gains survive a full 22-suite Modal sweep and multiple budgets without success loss?
5. Does C11's completed scaled U-Net/HRM ordering survive independent training seeds and world-clustered inference?
6. Why do C12-B cycles reduce bounded-search burden while Bellman residual worsens, and can a cost/admissibility-aware objective preserve the efficiency gain without 1–2% path-cost inflation?
7. Can vectorized or lazy C13 local-feature construction retain bit-identical ranks while closing the wall-time gap, and does the fixed result replicate across model seeds, a second cohort, and larger roadmaps?
8. Which claims should remain one coherent paper, and which—C12, HRM-v2 fidelity, or early world forecasting—belong in an appendix or separate follow-up?
9. Can a separately preregistered persistent-state study prevent long-sequence carry/logit collapse and add the missing causal frontier-update information while preserving C13-P's same-checkpoint reset and integrity controls?

# Authored source-document ledger

This ledger records the contribution of every centralized authored source document. Designs and plans are retained for methods/provenance; they are not counted as independent results.

## Cross-space documents


| Document                                                                                         | Type/status              | Contribution                                                                                                                           |
| ------------------------------------------------------------------------------------------------ | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| [PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md](cross-space/PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md) | Interpretive audit       | Cross-program formulation diagnosis, architecture/input audit, and C11 requirements. Numerical claims must be traced to stage reports. |
| [HRM_HEADTOHEAD.md](cross-space/HRM_HEADTOHEAD.md)                                               | Local result             | Reproduces discrete focal cells; repaired attention fits better in some variants but fails robust held-out generalization.             |
| [VALIDATION_GATES.md](cross-space/VALIDATION_GATES.md)                                           | Engineering verification | 194 passed / 0 failed snapshot; CPU/oracle bottleneck; predates C11 and final head-to-head.                                            |


## Discrete direct-solver documents


| Document                                                                                        | Type/status                 | Contribution                                                                                                                  |
| ----------------------------------------------------------------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [PORT_FIDELITY_AUDIT.md](discrete/direct-solver/audits/PORT_FIDELITY_AUDIT.md)                  | Canonical validity audit    | Identifies missing ACT halt loss, broken deep supervision, attention/sparse issues; appended status records fixes and parity. |
| [2026-07-05-hrm-v2-port-fixes.md](discrete/direct-solver/plans/2026-07-05-hrm-v2-port-fixes.md) | Methods/implementation plan | Defines the parity, ACT, supervision, optimizer, and revalidation work. Full paper-scale gate remains unfinished.             |
| [TRAINING_RESULTS.md](discrete/direct-solver/results/TRAINING_RESULTS.md)                       | Superseded result           | Pre-fix 96.64% token / 25.40% exact / frozen-halt signature; validity failure evidence only.                                  |
| [RETRAIN_RESULTS.md](discrete/direct-solver/results/RETRAIN_RESULTS.md)                         | Partial mechanism result    | q-halt loss declines and token accuracy is high; run stopped before convergence and has inconsistent exposure arithmetic.     |
| [HRM_V2_REVIEW_REPORT.md](discrete/direct-solver/history/HRM_V2_REVIEW_REPORT.md)               | Superseded review           | Finds initialization bug but incorrectly declares broad production readiness.                                                 |
| [BUGFIX_APPLIED.md](discrete/direct-solver/history/BUGFIX_APPLIED.md)                           | Historical bug record       | Correct truncated-normal PDF-bound fix; not the only later defect.                                                            |
| [REVIEW_SUMMARY.md](discrete/direct-solver/history/REVIEW_SUMMARY.md)                           | Superseded summary          | Early parity claim overturned by the fidelity audit.                                                                          |
| [SESSION_SUMMARY.md](discrete/direct-solver/history/SESSION_SUMMARY.md)                         | Historical session record   | Setup and launch chronology; early correctness/throughput claims are unreliable.                                              |
| [TRAINING_STATUS.md](discrete/direct-solver/history/TRAINING_STATUS.md)                         | Historical run snapshot     | In-progress pre-fix status and setup notes; final outcome and audit supersede it.                                             |


## Discrete dynamic-world-model documents


| Document                                                                                                             | Type/status                         | Contribution                                                                                              |
| -------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------- |
| [BENCHMARK_RESULTS.md](discrete/dynamic-world-model/results/BENCHMARK_RESULTS.md)                                    | Developmental result compendium     | 50-episode HRM scaling, LSTM, diffusion, and oracle outcomes; protocols differ and gaps lack uncertainty. |
| [LSTM_VS_HRM_EXPERIMENT.md](discrete/dynamic-world-model/results/LSTM_VS_HRM_EXPERIMENT.md)                          | Completed descriptive result/design | 100 shared-seed scale comparison; HRM 3M/10M 71/100 versus matched LSTM 68–69/100.                        |
| [ONLSTM_VS_HRM_EXPERIMENT_PRESETM_V2.md](discrete/dynamic-world-model/design/ONLSTM_VS_HRM_EXPERIMENT_PRESETM_V2.md) | Design with stale results template  | Defines Preset M+; completed raw result favors ON-LSTM and is summarized elsewhere.                       |
| [lstm-augmented-astar-2025.pdf](discrete/dynamic-world-model/references/lstm-augmented-astar-2025.pdf)               | External literature                 | Background comparison only; not repository-produced evidence.                                             |


## Discrete learned-heuristic documents


| Document                                                                                                                      | Type/status               | Contribution                                                                                                                    |
| ----------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| [experiment_writeup_last_two_runs.md](discrete/learned-heuristic/results/experiment_writeup_last_two_runs.md)                 | Completed negative        | Original and map-scale curricula; learned planners match baseline and isolate family B as the bottleneck.                       |
| [clean_transfer_experiment_blueprint.md](discrete/learned-heuristic/design/clean_transfer_experiment_blueprint.md)            | Methods design            | Defines matched full-FT/LoRA, residual target, alpha calibration, and ordering diagnostics realized by clean v3.                |
| [EXPERIMENT_RESULTS_COMPENDIUM.md](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_COMPENDIUM.md)                       | Canonical chronology      | Consolidates raw Modal history, clean negative, pooled HRM +2.11 pp, incomplete arms, and caveats.                              |
| [DISCRETE_EXPERIMENT_INVENTORY.md](discrete/DISCRETE_EXPERIMENT_INVENTORY.md)                                                 | Completeness map (2026-07-23) | Exhaustive family/script/volume/artifact inventory over the compendium; records superseded runs, controls, and tooling; no claim changes. |
| [2026-06-15-residual-tasklora-eval-speedup.md](discrete/learned-heuristic/plans/2026-06-15-residual-tasklora-eval-speedup.md) | Performance plan          | Defines diagnostic-off, caching, sharding, and budget-pruning work; outcome in `FAST_EVAL.md`.                                  |
| [FAST_EVAL.md](discrete/learned-heuristic/operations/FAST_EVAL.md)                                                            | Engineering result        | About 39x representative remote speedup after env forwarding; headline planner metrics preserved.                               |
| [2026-06-23-learned-focal-search-design.md](discrete/learned-heuristic/design/2026-06-23-learned-focal-search-design.md)      | Algorithm design          | Pre-registers admissible band plus learned within-band ranking and invariants.                                                  |
| [2026-06-23-learned-focal-search.md](discrete/learned-heuristic/plans/2026-06-23-learned-focal-search.md)                     | Implementation plan       | TDD/wiring plan; no independent outcome.                                                                                        |
| [FOCAL_SEARCH_RESULTS.md](discrete/learned-heuristic/results/FOCAL_SEARCH_RESULTS.md)                                         | Local pilot result        | Correlation/magnitude diagnosis, 6–15% expansion cut at w=1, expert=base.                                                       |
| [EXPERIMENT_RESULTS_FOCAL_REDESIGN.md](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_FOCAL_REDESIGN.md)               | Canonical pilot synthesis | Same focal evidence plus clean rerun and infrastructure diagnosis; not an independent replication of `FOCAL_SEARCH_RESULTS.md`. |


## Continuous program and C1–C6 documents


| Document                                                                                                                | Type/status               | Contribution                                                                                            |
| ----------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------- |
| [CONTINUOUS_PRM_STORY.md](continuous/program/CONTINUOUS_PRM_STORY.md)                                                   | Program synthesis         | Narrative through C10/C9b; inherited numbers and conclusions, not independent evidence.                 |
| [CONTINUOUS_PRM_STRATEGY.md](continuous/program/CONTINUOUS_PRM_STRATEGY.md)                                             | Strategy memo             | Headroom/saturation diagnosis and ranked next directions; refined by the cross-space audit.             |
| [continuous_prm_experiment_ladder_repo_coupled.md](continuous/c01-c04/continuous_prm_experiment_ladder_repo_coupled.md) | Design + pilot record     | C1–C4 protocol, smoke checks, saturated serious-run findings, and weak specialization evidence.         |
| [continuous_prm_c5_hard_obstacle_encoder_spec.md](continuous/c05/continuous_prm_c5_hard_obstacle_encoder_spec.md)       | Design + canonical result | Hard-map calibration, strong corrected ON-LSTM result, repeated HRM constant-cap failure.               |
| [continuous_prm_c6_heatmap_value_field_spec.md](continuous/c06/design/continuous_prm_c6_heatmap_value_field_spec.md)    | Design + feasibility      | Field protocol and approximate-oracle probe; implemented channel list differs.                          |
| [C6_RESULTS.md](continuous/c06/results/C6_RESULTS.md)                                                                   | Canonical local result    | Adequate training rescues learned fields and HRM; multi-suite generalization; local/single-seed caveat. |


## Continuous C7–C13 documents


| Document                                                                                                               | Type/status                      | Contribution                                                                                                                 |
| ---------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| [2026-06-27-c7-integration-comparison-design.md](continuous/c07/design/2026-06-27-c7-integration-comparison-design.md) | Preregistered design             | Matched scalar/field and additive/focal matrix; some provider prose is stale relative to implementation.                     |
| [2026-06-27-c7-integration-comparison.md](continuous/c07/plans/2026-06-27-c7-integration-comparison.md)                | Implementation plan              | Providers, calibration, tests, evaluation, statistics, and outputs; no independent result.                                   |
| [C7_RESULTS.md](continuous/c07/results/C7_RESULTS.md)                                                                  | Canonical local result           | Static 15–48% field-HRM expansion cuts, scalar viability, additive-over-focal principle; hierarchy wording overstated.       |
| [2026-06-27-c8-dynamics-design.md](continuous/c08/design/2026-06-27-c8-dynamics-design.md)                             | Preregistered design             | Space-time planner, aware/blind twins, dynamic metrics, and oracle gates.                                                    |
| [2026-06-27-c8-dynamics.md](continuous/c08/plans/2026-06-27-c8-dynamics.md)                                            | Implementation plan              | Dynamic maps, temporal providers, calibration, training/eval/statistics; no independent result.                              |
| [C8_RESULTS.md](continuous/c08/results/C8_RESULTS.md)                                                                  | Canonical heavy local result     | Dynamic learned gains, additive-over-focal, and no systematic future-window benefit; preserves diagnostic/intermediate runs. |
| [2026-07-23-c8r-multiseed-replication.md](continuous/c08/design/2026-07-23-c8r-multiseed-replication.md)               | Preregistered design             | Fixed-provider multi-seed replication protocol: seeds 2001/2002, frozen fresh cohort, R1–R3 readouts, failures-as-is rule.   |
| [2026-07-23-c8-budget-curves.md](continuous/c08/design/2026-07-23-c8-budget-curves.md)                                 | Frozen descriptive design        | Success-vs-budget curves from one 4×binding pass by exact expansion thresholding; report-as-is rule; cross-check passed.      |
| [2026-07-23-c14-label-density-factorial.md](continuous/c14/design/2026-07-23-c14-label-density-factorial.md)           | Preregistered design + amendment | 180-cell N×diversity×domain×method factorial; §7 pre-execution amendment (w_min diversity) after static 160-states arithmetic. |
| [C14_RESULT.md](continuous/c14/results/C14_RESULT.md)                                                                  | Canonical factorial result       | H-C14 rejected as-is (no LoRA-favored ratio regime ≥N=256); diversity-at-fixed-N governs full-FT safety; LoRA never collapses. |
| [C8R_MULTISEED_RESULT.md](continuous/c08/results/C8R_MULTISEED_RESULT.md)                                              | Canonical replication result     | 17/18 seed×suite success CIs exclude zero; twin contrasts flip suite and sign across seeds; effort seed-variable on 3 suites. |
| [2026-06-29-c9-transfer-design.md](continuous/c09/design/2026-06-29-c9-transfer-design.md)                             | Preregistered design             | ADAPT/TEST, K curves, zero-shot/LoRA/full-FT/scratch arms, and transfer gates.                                               |
| [2026-06-29-c9-transfer.md](continuous/c09/plans/2026-06-29-c9-transfer.md)                                            | Implementation plan              | Adapter/training/eval/statistics tasks; no independent result.                                                               |
| [C9_RESULTS.md](continuous/c09/results/C9_RESULTS.md)                                                                  | Canonical local result           | Low-K base preservation and higher-K full-FT capacity; universal crossover prose requires narrowing.                         |
| [2026-06-29-c9-hardening-design.md](continuous/c09h/design/2026-06-29-c9-hardening-design.md)                          | Preregistered design             | Matched compute, bounded/unbounded LoRA, and field conv-LoRA.                                                                |
| [2026-06-29-c9-hardening.md](continuous/c09h/plans/2026-06-29-c9-hardening.md)                                         | Implementation plan              | Conv-LoRA and matched training/eval tasks; no independent result.                                                            |
| [C9H_RESULTS.md](continuous/c09h/results/C9H_RESULTS.md)                                                               | Canonical local mechanism result | Bound delta 0.000 +/- 0.008; full-rank capacity and field-U-Net ceiling; crossover is conditional.                           |
| [2026-06-30-c9b-dynamics-transfer-design.md](continuous/c09b/design/2026-06-30-c9b-dynamics-transfer-design.md)        | Preregistered design             | Ports adaptation to aware/blind dynamics and pre-registers crossover/temporal gates.                                         |
| [2026-06-30-c9b-dynamics-transfer.md](continuous/c09b/plans/2026-06-30-c9b-dynamics-transfer.md)                       | Implementation plan              | Temporal datasets, adapters, success-aware probe, and evaluation; no independent result.                                     |
| [C9B_RESULTS.md](continuous/c09b/results/C9B_RESULTS.md)                                                               | Canonical local result           | 0/9 aware wins at full-FT K16; K=1 dynamics is label-rich; some universal significance prose contradicted by tables.         |
| [2026-06-29-c10-interp-design.md](continuous/c10/design/2026-06-29-c10-interp-design.md)                               | Preregistered design             | Source-family grid and zero-label RBF/nearest/uniform/weight/prediction composition.                                         |
| [2026-06-29-c10-interp.md](continuous/c10/plans/2026-06-29-c10-interp.md)                                              | Implementation plan              | Merge tests, family construction, source experts, evaluation, and analysis; no independent result.                           |
| [C10_RESULTS.md](continuous/c10/results/C10_RESULTS.md)                                                                | Canonical local negative         | Near-perfect axis routing, no consistent planning gain; hull/significance/equivalence wording requires narrowing.            |
| [2026-07-07-c11-headroom-probe.md](continuous/c11/plans/2026-07-07-c11-headroom-probe.md)                              | Preregistered probe plan         | Defines the oracle-versus-leg-sum authorization gate.                                                                        |
| [C11_HEADROOM.md](continuous/c11/results/C11_HEADROOM.md)                                                              | Completed result                 | Ratios 0.082–0.225 and monotonic ratio decline with K; upper-bound evidence only.                                            |
| [2026-07-07-c11-compositional-mission-design.md](continuous/c11/design/2026-07-07-c11-compositional-mission-design.md) | Approved preregistration         | MLP/U-Net/GNN/trace/HRM-v2 architecture test, K dose response, depth-of-compute gates.                                       |
| [2026-07-07-c11-mission.md](continuous/c11/plans/2026-07-07-c11-mission.md)                                            | Implemented plan                 | Main 198-run grid complete; original run-count arithmetic mixed the main grid and 12-run scaled addendum.                     |
| [C11_RESULTS.md](continuous/c11/results/C11_RESULTS.md)                                                                | Canonical local result           | K=0 premise failure diagnosed; G1/G2a/G2b negative; shallow global-input advantage, recurrent collapse, no depth response.    |
| [2026-07-10-c12-persistent-hierarchical-planning-design.md](continuous/c12/design/2026-07-10-c12-persistent-hierarchical-planning-design.md) | Approved preregistration + pre-pilot amendment | Separately gates persistent hidden-regime dynamics and tied product-graph refinement; freezes recurring hazards, OOD slices, matched widths, and compute stop rules before pilot. |
| [2026-07-10-c12-persistent-hierarchical-planning.md](continuous/c12/plans/2026-07-10-c12-persistent-hierarchical-planning.md) | Completed implementation plan | C12-A pilot and the full C12-B G0/smoke/pilot/three-seed sequence are complete; deviations and rejected K16 cells are recorded. |
| [C12_RESULTS.md](continuous/c12/results/C12_RESULTS.md)                                                                | Combined canonical result | C12-A one-seed pilot is `strong_negative`; C12-B cycles improve monotonically but fail the K-dose response, with one localized C/K8 tied-control win and explicit path-cost caveats. |
| [2026-07-16-c13-state-conditioned-heuristic-design.md](continuous/c13/design/2026-07-16-c13-state-conditioned-heuristic-design.md) | Implemented design / gated preregistration | Separates strict geometry from bounded observations and records the fresh-start rollout amendment. |
| [2026-07-16-c13-state-conditioned-heuristic.md](continuous/c13/plans/2026-07-16-c13-state-conditioned-heuristic.md) | Completed gated plan | C13-A through C13-M are complete; the remaining items are feature-time hardening, replication, and professor review of claim wording. |
| [C13_INITIAL_AUDIT.md](continuous/c13/results/C13_INITIAL_AUDIT.md) | Preliminary audit | One-step guidance is valid but weak; deeper gains require multi-layer traversal; +10% density adds work for little path benefit. |
| [C13B_ROLLOUT_RANKER_SMOKE.md](continuous/c13/results/C13B_ROLLOUT_RANKER_SMOKE.md) | Implementation/provenance smoke | Fresh-start median returns avoid shortest-path supervision; the three-world learned-signal gate fails. |
| [C13B_IDENTIFIABILITY_STUDY.md](continuous/c13/results/C13B_IDENTIFIABILITY_STUDY.md) | Completed causal diagnostic | Local signal is learnable, but exact rollout target utility, padding/readout, and narrow FOCAL integration each fail distinct checks. |
| [C13C_CERTIFIED_SEARCH.md](continuous/c13/results/C13C_CERTIFIED_SEARCH.md) | Completed negative integration gate | The proof is correct on all 90 rows, but a fresh Euclidean certifier duplicates too much work: the oracle ceiling loses all six `w=1.10` comparisons. |
| [C13D_SHARED_QUEUE_ORACLE.md](continuous/c13/results/C13D_SHARED_QUEUE_ORACLE.md) | Passed oracle integration ceiling | Shared search state turns the C13-C 0/6 oracle failure into a 6/6 win at `w=1.10`, establishing the ceiling subsequently tested by C13-E. |
| [C13E_SHARED_QUEUE_EXACT_TARGET.md](continuous/c13/results/C13E_SHARED_QUEUE_EXACT_TARGET.md) | Failed exact-target gate | The unchanged shared search certifies all paths, but exact rollout averages `+1.33` expansions and wins only 2/6 primary comparisons; target alignment/calibration blocks learned providers. |
| [2026-07-17-c13-current-state-literature-and-next-target.md](continuous/c13/design/2026-07-17-c13-current-state-literature-and-next-target.md) | Literature/method decision | Routes LoHA*, RTAA*, MHA*, and search-effort learning into C13-F/G/H while enforcing the professor's state-information boundary. |
| [2026-07-17-c13i-current-state-vs-map-conditioned.md](continuous/c13/design/2026-07-17-c13i-current-state-vs-map-conditioned.md) | C13-I preregistration | Freezes the live six-suite comparison that rejects one-suite transfer. |
| [2026-07-17-c13j-multisuite-current-state-training.md](continuous/c13/design/2026-07-17-c13j-multisuite-current-state-training.md) | C13-J preregistration | Freezes multi-suite training and the disjoint development block that rejects distribution-only repair. |
| [2026-07-17-c13k-local-bellman-integration.md](continuous/c13/design/2026-07-17-c13k-local-bellman-integration.md) | C13-K preregistration | Isolates one radius-bounded Bellman backup as the integration mechanism. |
| [2026-07-17-c13l-local-backup-scale-calibration.md](continuous/c13/design/2026-07-17-c13l-local-backup-scale-calibration.md) | C13-L preregistration | Rejects the absolute 1.10 ceiling while exposing the comparator-relative matched-quality frontier. |
| [2026-07-17-c13m-matched-quality-confirmation.md](continuous/c13/design/2026-07-17-c13m-matched-quality-confirmation.md) | C13-M preregistration | Freezes the alpha-1.50 arm, 144-world cohort, five primary gates, and separate bounded control before generation. |
| [2026-07-17-c13n-hrm-substitution.md](continuous/c13/design/2026-07-17-c13n-hrm-substitution.md) | C13-N preregistration | Freezes the architecture-only substitution, six development cells, matched field/flat controls, and confirmation-only-after-pass rule. |
| [C13F_M_CURRENT_STATE_RESULT.md](continuous/c13/results/C13F_M_CURRENT_STATE_RESULT.md) | Canonical completed result | Full C13-F–M mechanism record and confirmed 15.95% expansion reduction versus complete-map field HRM, with path-quality, safety, timing, and information-boundary caveats. |
| [C13N_HRM_SUBSTITUTION_RESULT.md](continuous/c13/results/C13N_HRM_SUBSTITUTION_RESULT.md) | Completed negative architecture diagnostic | Useful pooled HRM signal does not clear suite robustness or matched-flat path-quality gates; no fresh confirmation is run. |
| [2026-07-17-c13o-hrm-summary-last-alignment.md](continuous/c13/design/2026-07-17-c13o-hrm-summary-last-alignment.md) | C13-O preregistration | Freezes the summary-last readout alignment test with identical initialization/targets/optimizer and development-only gates. |
| [C13O_HRM_ALIGNMENT_RESULT.md](continuous/c13/results/C13O_HRM_ALIGNMENT_RESULT.md) | Completed negative architecture diagnostic | Summary-last ordering improves trimmed HRM transiently at iteration 6 but no cell passes the complete field/flat/readout method gate; the fixed-endpoint readout comparison reverses and no confirmation is run. |
| [2026-07-19-c13p-persistent-search-state.md](continuous/c13/design/2026-07-19-c13p-persistent-search-state.md) | C13-P preregistration | Freezes the query-level persistent-carry pilot: persistent-versus-reset modes of one checkpoint, stationary path-frontier target from the frozen C13-M teacher, forbidden-information boundaries, and G0-P/G1-P/G2-P gates. |
| [2026-07-19-c13p-persistent-search-state.md (plan)](continuous/c13/plans/2026-07-19-c13p-persistent-search-state.md) | Completed implementation plan | Frozen staged harness, independent raw-artifact reconstruction, and canonical documentation workflow for the completed pilot. |
| [C13P_PERSISTENT_SEARCH_RESULT.md](continuous/c13/results/C13P_PERSISTENT_SEARCH_RESULT.md) | Completed valid negative mechanism pilot | G0-P passes, but persistent carry loses to same-checkpoint reset on world-macro MRR; no self-bootstrap or confirmation is run. |


# Generated report disposition

The [continuous catalog](continuous/GENERATED_EVIDENCE.md) and [discrete catalog](discrete/GENERATED_EVIDENCE.md) are the exhaustive operational indexes. The table below preserves the original reviewed set and adds key later generated outputs; raw reruns and non-Markdown artifacts are not independent replications.

## Selected continuous generated reports


| Generated report                                                                                                                                   | Disposition                                                                 |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| [C5 canonical significance](../../hrm-cloud/continuous_prm/runs/continuous_prm_c5_hard_r1/results/continuous_prm_c5_significance.md)               | Canonical C5 corrected success/expansion evidence.                          |
| [C5 HRM tune significance](../../hrm-cloud/continuous_prm/runs/continuous_prm_c5_hard_hrm_tune_a/results/continuous_prm_c5_significance.md)        | Negative follow-up; HRM remains equal to Euclid.                            |
| [C5 HRM soft-cap significance](../../hrm-cloud/continuous_prm/runs/continuous_prm_c5_hard_hrm_softcap_a/results/continuous_prm_c5_significance.md) | Negative follow-up; differentiable cap does not fix collapse.               |
| [C5 analyzer smoke](../../hrm-cloud/continuous_prm/runs/smoke_c5_script/results/continuous_prm_c5_significance.md)                                 | Pipeline validation only.                                                   |
| [C6 undersized diagnostic](../../hrm-cloud/continuous_prm/runs/c6_local_run2/results/continuous_prm_c6_significance.md)                            | Saturated success; undertrained learned fields; diagnostic only.            |
| [C6 single-suite canonical](../../hrm-cloud/continuous_prm/runs/c6_local_big1/results/continuous_prm_c6_significance.md)                           | Supports significant field gains and HRM recovery.                          |
| [C6 multi-suite canonical](../../hrm-cloud/continuous_prm/runs/c6_local_multi1/results/continuous_prm_c6_significance.md)                          | Supports dense/rooms generalization after multi-suite training.             |
| [C6 U-Net/oracle smoke](../../hrm-cloud/continuous_prm/runs/smoke_c6_heatmap/results/continuous_prm_c6_significance.md)                            | Feasibility/pipeline only.                                                  |
| [C6 multi-model smoke](../../hrm-cloud/continuous_prm/runs/smoke_c6_field_models/results/continuous_prm_c6_significance.md)                        | Finite/nonconstant output validation only.                                  |
| [C6 fast-path smoke](../../hrm-cloud/continuous_prm/runs/smoke_c6_field_models_fastpath/results/continuous_prm_c6_significance.md)                 | Fast-path equivalence/pipeline evidence only.                               |
| [C7 canonical significance](../../hrm-cloud/continuous_prm/runs/c7_local/results/continuous_prm_c7_significance.md)                                | Canonical success/expansion grid.                                           |
| [C7 canonical preregistered comparisons](../../hrm-cloud/continuous_prm/runs/c7_local/results/continuous_prm_c7_preregistered.md)                  | Canonical six-comparison output; some p-values exploratory/uncorrected.     |
| [C7 early-smoke significance](../../hrm-cloud/continuous_prm/runs/c7_smoke/results/continuous_prm_c7_significance.md)                              | Smoke only.                                                                 |
| [C7 early-smoke comparisons](../../hrm-cloud/continuous_prm/runs/c7_smoke/results/continuous_prm_c7_preregistered.md)                              | Smoke only.                                                                 |
| [C7 full-smoke significance](../../hrm-cloud/continuous_prm/runs/c7_full_smoke/results/continuous_prm_c7_significance.md)                          | Full-pipeline smoke only.                                                   |
| [C7 full-smoke comparisons](../../hrm-cloud/continuous_prm/runs/c7_full_smoke/results/continuous_prm_c7_preregistered.md)                          | Full-pipeline smoke only.                                                   |
| [C7 final-check significance](../../hrm-cloud/continuous_prm/runs/c7_final_check/results/continuous_prm_c7_significance.md)                        | Statistical-writer regression check, not new research evidence.             |
| [C7 final-check comparisons](../../hrm-cloud/continuous_prm/runs/c7_final_check/results/continuous_prm_c7_preregistered.md)                        | Comparison-writer regression check, not a replication.                      |
| [C8 initial significance](../../hrm-cloud/continuous_prm/runs/c8_local/results/continuous_prm_c8_significance.md)                                  | Early learned-positive / temporal-mixed diagnostic.                         |
| [C8 initial comparisons](../../hrm-cloud/continuous_prm/runs/c8_local/results/continuous_prm_c8_preregistered.md)                                  | Early comparison output; superseded by heavy run.                           |
| [C8 hardened significance](../../hrm-cloud/continuous_prm/runs/c8_local_hardened/results/continuous_prm_c8_significance.md)                        | Intermediate time-coupled suite result; directional/mixed.                  |
| [C8 hardened comparisons](../../hrm-cloud/continuous_prm/runs/c8_local_hardened/results/continuous_prm_c8_preregistered.md)                        | Intermediate comparison output; superseded by heavy run.                    |
| [C8 heavy significance](../../hrm-cloud/continuous_prm/runs/c8_local_heavy/results/continuous_prm_c8_significance.md)                              | Canonical heavy success/expansion evidence.                                 |
| [C8 heavy comparisons](../../hrm-cloud/continuous_prm/runs/c8_local_heavy/results/continuous_prm_c8_preregistered.md)                              | Canonical heavy learned/additive/temporal comparisons.                      |
| [C8 heuristic accuracy](../../hrm-cloud/continuous_prm/runs/c8_local_heavy/results/c8_heuristic_accuracy.md)                                       | Mechanistic MAE evidence; pooled cells, no world-level uncertainty.         |
| [C9 significance](../../hrm-cloud/continuous_prm/runs/c9_local/results/continuous_prm_c9_significance.md)                                          | Success versus Euclid; not direct method-versus-method inference.           |
| [C9 comparisons](../../hrm-cloud/continuous_prm/runs/c9_local/results/continuous_prm_c9_comparisons.md)                                            | K curves and descriptive adaptation contrasts.                              |
| [C9h significance](../../hrm-cloud/continuous_prm/runs/c9h_local/results/continuous_prm_c9h_significance.md)                                       | Matched-compute success evidence; repeated-world caveat.                    |
| [C9h comparisons](../../hrm-cloud/continuous_prm/runs/c9h_local/results/continuous_prm_c9h_comparisons.md)                                         | Bounded/unbounded and field/full-rank curves.                               |
| [C9b significance](../../hrm-cloud/continuous_prm/runs/c9b_local/results/continuous_prm_c9b_significance.md)                                       | Dynamic transfer success evidence; contradicts some universal source prose. |
| [C9b comparisons](../../hrm-cloud/continuous_prm/runs/c9b_local/results/continuous_prm_c9b_comparisons.md)                                         | Dynamic K curves and loss of the static crossover.                          |
| [C9b aware/blind probe](../../hrm-cloud/continuous_prm/runs/c9b_local/results/continuous_prm_c9b_probe.md)                                         | Supports the 0/9 K16 aware-win count; aggregation differs from curves.      |
| [C10 significance](../../hrm-cloud/continuous_prm/runs/c10_local/results/continuous_prm_c10_significance.md)                                       | Learned-versus-Euclid success evidence; some q=0.062 cells.                 |
| [C10 comparisons](../../hrm-cloud/continuous_prm/runs/c10_local/results/continuous_prm_c10_comparisons.md)                                         | Descriptive interpolation null and method deltas.                           |
| [C10 bracketing](../../hrm-cloud/continuous_prm/runs/c10_local/results/continuous_prm_c10_bracketing.md)                                           | Own-axis bracketing and RBF selectivity; not a full convex-hull test.       |
| [C11 generated canonical result](../../hrm-cloud/continuous_prm/runs/c11_local/C11_RESULTS.md)                                                      | Byte-identical run output for the canonical C11 result document.            |
| [C11 diagnosis addendum](../../hrm-cloud/continuous_prm/runs/c11_local/diagnosis_addendum.md)                                                       | Diagnosis embedded in the canonical report; duplicate provenance, not an independent result. |
| [C13-I live C7 comparison](../../hrm-cloud/continuous_prm/runs/c13_lhbl_c7_comparison/C13I_RESULTS.md) | Negative one-suite transfer result; six secondary U-Net replay deviations are retained and primary field-HRM parity is exact. |
| [C13-M matched-quality confirmation](../../hrm-cloud/continuous_prm/runs/c13_matched_quality_confirmation/results/C13M_RESULTS.md) | Preregistered positive 144-world result; all primary and bounded-control conditions pass. |
| [C13-N HRM substitution](../../hrm-cloud/continuous_prm/runs/c13_hrm_substitution/results/C13N_RESULT.md) | Preregistered development failure; pooled signal is promising, but suite robustness and matched-flat quality conditions block confirmation. |
| [C13-O summary-last alignment](../../hrm-cloud/continuous_prm/runs/c13_hrm_alignment/results/C13O_RESULT.md) | Preregistered development failure; a transient iteration-6 readout gain does not recover the method gate or persist to the fixed endpoint. |


## Discrete generated reports (5)


| Generated report                                                                               | Disposition                                                                                                      |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| [Full Modal survey](../../modal_downloads/full_survey_sdk_parallel/summary.md)                 | Main provenance survey: 13,671 JSON files parsed across 11 volumes with zero download errors; compendium source. |
| [Later residual snapshot](../../modal_downloads/residual_latest_20260601/summary.md)           | 2,028-file partial residual snapshot; later than the full survey, not an independent experiment.                 |
| [Manifest-only survey diagnostic](../../modal_downloads/full_survey_manifest_check/summary.md) | Zero files; provenance diagnostic only.                                                                          |
| [Live-manifest diagnostic 1](../../modal_downloads/residual_check_live_manifest/summary.md)    | Zero files; no result evidence.                                                                                  |
| [Live-manifest diagnostic 2](../../modal_downloads/residual_check_live_manifest_2/summary.md)  | Zero files; no result evidence.                                                                                  |


## Raw evidence entry points


| Program/stage           | Main raw artifacts                                                                                                  | Use                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Discrete survey history | `[modal_downloads/survey_results/](../../modal_downloads/survey_results/)`                                          | Early forecasting, transfer, and model-comparison JSONs.                     |
| Discrete clean transfer | `[clean_v3_results](../../modal_downloads/clean_v3_results/)`                                                       | Canonical clean-transfer result; filter undefined diagnostic values.         |
| Discrete multitask      | `[multitask_results](../../modal_downloads/multitask_results/)`                                                     | Pooled base and limited-ID expert rows.                                      |
| Continuous C5–C8        | `[continuous_prm/runs](../../hrm-cloud/continuous_prm/runs/)` stage directories                                     | Raw matched rows, summaries, significance outputs, configs, and checkpoints. |
| Continuous C9/C9h/C9b   | `*_eval_raw.csv`, `*_curves.csv`, comparison/significance reports, and adapter manifests in each run directory      | Recompute clustered uncertainty and direct method contrasts.                 |
| Continuous C10          | C10 raw rows, curves, weight/source manifests, comparisons, and bracketing report                                   | Recompute interpolation deltas and direct contrasts.                         |
| C11 headroom            | `[c11_probe_records.csv](../../hrm-cloud/continuous_prm/runs/c11_probe/c11_probe_records.csv)`                      | Completed oracle/leg-sum probe.                                              |
| C11 main                | [manifest](../../hrm-cloud/continuous_prm/runs/c11_local/manifest.json), [evaluation](../../hrm-cloud/continuous_prm/runs/c11_local/results/c11_eval_raw.csv), [state MAE](../../hrm-cloud/continuous_prm/runs/c11_local/results/c11_state_mae.csv), [halt steps](../../hrm-cloud/continuous_prm/runs/c11_local/results/c11_halt_steps.csv) | Completed main-grid result; recompute world-clustered inference from raw rows. |
| C11 scaled addendum     | `[c11_big/manifest.json](../../hrm-cloud/continuous_prm/runs/c11_big/manifest.json)` plus training logs, 12 checkpoints, evaluation, and state-MAE rows | Completed scale diagnostic; U-Net remains ahead of HRM and recurrent collapse persists. |
| C12                     | [`c12_refiner/`](../../hrm-cloud/continuous_prm/runs/c12_refiner/) manifests, 48 checkpoints, two 3,200-row tables, clustered analysis, and integrity records | Reproduce C12-B cycle curves, tied controls, and path-quality caveats. |
| C13-M                   | [raw 1,296 rows](../../hrm-cloud/continuous_prm/runs/c13_matched_quality_confirmation/results/confirmation_raw.csv), [pairwise summary](../../hrm-cloud/continuous_prm/runs/c13_matched_quality_confirmation/results/pairwise_summary.csv), [verdict](../../hrm-cloud/continuous_prm/runs/c13_matched_quality_confirmation/results/gate_verdict.json), and [integrity](../../hrm-cloud/continuous_prm/runs/c13_matched_quality_confirmation/integrity.json) | Reproduce fixed current-versus-C7 comparisons, path quality, cohort separation, and safety checks. |
| C13-N                   | [raw 360 rows](../../hrm-cloud/continuous_prm/runs/c13_hrm_substitution/results/development_raw.csv), [candidate summary](../../hrm-cloud/continuous_prm/runs/c13_hrm_substitution/results/development_candidates.csv), [verdict](../../hrm-cloud/continuous_prm/runs/c13_hrm_substitution/results/development_verdict.json), and [integrity](../../hrm-cloud/continuous_prm/runs/c13_hrm_substitution/integrity.json) | Reproduce the HRM/flat matched development grid, suite gate, path-quality comparison, and blocked-confirmation decision. |
| C13-O                   | [raw 504 rows](../../hrm-cloud/continuous_prm/runs/c13_hrm_alignment/results/development_raw.csv), [candidate summary](../../hrm-cloud/continuous_prm/runs/c13_hrm_alignment/results/development_candidates.csv), [verdict](../../hrm-cloud/continuous_prm/runs/c13_hrm_alignment/results/development_verdict.json), and [integrity](../../hrm-cloud/continuous_prm/runs/c13_hrm_alignment/integrity.json) | Reproduce the summary-last/trimmed/flat development grid, direct readout test, method gate, and blocked-confirmation decision. |
| C13-P                   | [generated report](../../hrm-cloud/continuous_prm/runs/c13_persistent_search/results/C13P_RESULT.md), [raw ranking](../../hrm-cloud/continuous_prm/runs/c13_persistent_search/results/development_ranking_raw.csv), [raw search](../../hrm-cloud/continuous_prm/runs/c13_persistent_search/results/development_search_raw.csv), [verdict](../../hrm-cloud/continuous_prm/runs/c13_persistent_search/results/gate_verdict.json), and [integrity](../../hrm-cloud/continuous_prm/runs/c13_persistent_search/integrity.json) | Reproduce the valid G0-P pass, negative persistent-versus-reset G1-P result, descriptive failed G2-P, and no-self-bootstrap/no-confirmation boundary. |


The report visualization uses the reviewed, executable [C11 headroom SQL rows](analysis/c11_headroom_report.sql), which were checked against the raw probe records.

## Master-document maintenance rule

When new results land, update this document only after:

1. identifying the canonical raw and generated artifacts;
2. separating new evidence from reruns/smokes/duplicate prose;
3. recomputing the headline values at the correct grain;
4. recording direct-comparison uncertainty and multiplicity handling;
5. updating the claim-audit table when source prose changes;
6. stamping the snapshot time, especially for C12/C13 or any later live experiment.
