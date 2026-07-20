# HRMv2 Experiment Documentation Index

**Inventory date:** 2026-07-19
**Scope:** discrete-grid/direct-solver experiments, discrete learned-search experiments, continuous-PRM experiments through the completed C13-M current-state confirmation and C13-N/C13-O HRM architecture diagnostics, and documents that compare the two spaces.

This is the canonical entry point for experiment documentation. Authored designs, plans, audits, and results are centralized here; generated reports remain beside their raw CSV/JSON/checkpoint evidence so reruns do not create a second, drifting copy.
The evidence indexes catalog the generated continuous and discrete reports and their principal raw-artifact entry points.

## Where to start

- For the paper-scaffolding evidence base, read the [Master Experiment Evidence Synthesis](MASTER_EXPERIMENT_SYNTHESIS.md). It consolidates exact figures, caveats, claim-audit corrections, paper readiness, and the disposition of every authored and generated report.
- For the whole program and its cross-space conclusions, read the [program audit](cross-space/PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md).
- For the discrete learned-A\* chronology, read the [discrete results compendium](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_COMPENDIUM.md), then the [focal redesign report](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_FOCAL_REDESIGN.md).
- For the continuous chronology, read [The Story So Far](continuous/program/CONTINUOUS_PRM_STORY.md), then the per-stage result documents.
- For current validation status, read [Cross-Space Validation Gates](cross-space/VALIDATION_GATES.md).
- For generated/raw evidence, use the [discrete evidence catalog](discrete/GENERATED_EVIDENCE.md) and [continuous evidence catalog](continuous/GENERATED_EVIDENCE.md).

## Program findings at a glance

| Area | Best current reading |
|---|---|
| Discrete trajectory forecasting | Architecture mattered, but the sign depended on task and scale: medium/large HRM reached 0.71 vs LSTM 0.67-0.69 in one benchmark, while the harder Preset M+ favored ON-LSTM (mean 0.388 vs HRM 0.265). |
| Discrete learned heuristics | Additive residual integration was near-null or harmful because the model ranked states well but was badly scale-calibrated. Reusing the same model as a focal tie-breaker produced a regression-free **6-15% expansion reduction** at `w=1.0`; task-LoRA experts added no ordering benefit over the pooled base. |
| Continuous static PRM | Hard-map calibration made the comparison meaningful. C5 gave large, significant ON-LSTM gains while HRM collapsed; C6/C7 showed that adequate training and value/scalar formulations both work. C7 additive learned heuristics cut expansions roughly **15-48%** and generalized over near-, structural-, and scale-OOD suites. |
| Continuous dynamics | Learned space-time heuristics beat Euclidean-time by **65-95% fewer expansions** with large success gains, but explicit future-window inputs did not help the heuristic: the fully trained time-blind models tied or beat aware models in most significant cells. |
| Transfer | Static C9/C9h established a real sample-efficiency/capacity crossover: LoRA is robust and flat at very low K; full fine-tuning is unstable at K=1 but wins once enough target data exists. The clamp was irrelevant; low rank caused the plateau. Under dynamics, one world already contains tens of thousands of `(node,t)` labels, so the static crossover disappears, though transfer still beats scratch. |
| Adapter interpolation | C10's descriptor/RBF machinery localized the right family with 98.6-99.8% same-axis mass, but weight-space and prediction-space interpolation did not beat the pooled zero-shot base. It was a clean null caused by composing adapters that individually plateau at the base. |
| Hierarchy hypothesis | No robust HRM/ON-LSTM advantage has appeared on the per-node heuristic-regression formulation; a U-Net is often strongest. The cross-space audit attributes this to a local, pre-digested task that an MLP can saturate, not to a general claim that hierarchy cannot help planning. |
| C11 status | The headroom gate passed, but the completed learned grid and scaled addendum are negative for hierarchy/depth response. Global U-Net/GNN inputs separate at shallow K; every arm degrades with depth, recurrent arms are fragile, and the scaled U-Net remains ahead of scaled HRM. |
| C12 status | C12-A's memory-headroom G0 passes, but its completed one-seed learned pilot fails G1/G2/G3 and closes `strong_negative`. C12-B's 48-checkpoint full grid shows monotone cycle gains but no K-dose response; tied refinement wins both controls only on C/K8, while path costs average 1–2% above oracle. |
| C13 status | C13-M confirms the current-state/local-Bellman arm on 144 untouched six-suite worlds. It averages `68.31` expansions versus `81.26` for complete-map field HRM: paired delta `-12.96`, 95% CI `[-16.30, -9.74]`, 109/3/32 W/T/L, with all six suite means negative. Mean/max path-cost ratios are also lower (`1.0235/1.1624` vs `1.0311/1.3346`). The direct arm is not formally bounded and its prototype feature construction is slower; the separate `w=1.10` FOCAL control has zero violations. C13-N's literal `hrm_trimmed` substitution fails its development gate. C13-O's summary-last alignment produces significant direct gains over trimmed HRM at iteration 6, but no cell recovers the full method gate; the fixed endpoint again improves only 3/6 suites, fails matched-flat path-quality margins, and does not beat trimmed HRM. Neither diagnostic opens confirmation. |

## Folder layout

```text
docs/experiments/
├── README.md                         # this master index
├── cross-space/                      # audits, head-to-heads, shared validation
├── discrete/
│   ├── direct-solver/                # HRM-v2 maze training and fidelity history
│   ├── dynamic-world-model/          # trajectory-prediction planning experiments
│   └── learned-heuristic/            # transfer, TaskLoRA, focal-search line
└── continuous/
    ├── program/                      # cross-phase story and strategy
    ├── c01-c04/ ... c13/             # stage-specific design, plan, results
    └── GENERATED_EVIDENCE.md          # every generated run report
```

Operational READMEs remain next to the code they explain: [HRM cloud](../../hrm-cloud/README.md), [continuous PRM](../../hrm-cloud/continuous_prm/README.md), and [HRM-v2](../../HRM-v2/README.md).

## Cross-space major documents

| Document | Kind | Short finding/result |
|---|---|---|
| [PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md](cross-space/PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md) | Audit/synthesis | Across both spaces, learned wins are efficiency margins, pooled bases beat specialists, and architecture separation vanished at the shift from forecasting to local cost-to-go regression. It argues the existing formulation is MLP-complete and motivates compositional C11. |
| [HRM_HEADTOHEAD.md](cross-space/HRM_HEADTOHEAD.md) | Results | Exactly reproduced the discrete focal result, then showed a repaired cross-token-attention model worse than the incumbent on both continuous held-outs. A remediation sweep improved train loss and one maze cell but failed both-target gates and generalized badly to rooms, supporting the formulation/overfit diagnosis. |
| [VALIDATION_GATES.md](cross-space/VALIDATION_GATES.md) | Verification | Recorded **194 passed, 0 failed** across HRM-v2, discrete cloud, and continuous suites. Validation was CPU/oracle-bound; H100s were not the useful acceleration lever. |

## Discrete experiments

### HRM-v2 direct maze solver and port validation

Current evidence:

| Document | Kind | Short finding/result |
|---|---|---|
| [PORT_FIDELITY_AUDIT.md](discrete/direct-solver/audits/PORT_FIDELITY_AUDIT.md) | Audit | Found the model port structurally faithful but the original training port unfaithful: missing `q_halt_loss`, broken deep supervision, an SDPA layout hazard, and recipe drift meant earlier runs had never trained the paper's ACT mechanism. The appended status records the later fixes. |
| [2026-07-05-hrm-v2-port-fixes.md](discrete/direct-solver/plans/2026-07-05-hrm-v2-port-fixes.md) | Implementation plan | Specifies parity tests, the ACT loss port, one-segment-per-step deep supervision, sparse-embedding repair, and Maze-Hard revalidation. It contains no independent result; the outcome is in `RETRAIN_RESULTS.md`. |
| [TRAINING_RESULTS.md](discrete/direct-solver/results/TRAINING_RESULTS.md) | Historical result | The pre-fix run reported 96.64% token and 25.40% exact accuracy, but the later audit showed the halt head was frozen and deep supervision was absent. Treat these numbers as a historical baseline, not evidence of faithful HRM training. |
| [RETRAIN_RESULTS.md](discrete/direct-solver/results/RETRAIN_RESULTS.md) | Current result | The fixed run showed `q_halt_loss` declining 0.14 to 0.06 and 90-95% eval token accuracy, proving the repaired mechanisms were alive. It was stopped by choice around 150k/375k steps; exact accuracy remained at the expected pre-grok floor, so paper-scale reproduction is still unrun. |

Superseded historical notes are preserved rather than erased:

| Document | Why retained | Short summary |
|---|---|---|
| [HRM_V2_REVIEW_REPORT.md](discrete/direct-solver/history/HRM_V2_REVIEW_REPORT.md) | Superseded review | Correctly found the truncated-normal initialization bug, but incorrectly declared the whole port production-ready; the 2026 fidelity audit overturned that broader conclusion. |
| [BUGFIX_APPLIED.md](discrete/direct-solver/history/BUGFIX_APPLIED.md) | Bug record | Documents the one-line truncated-normal PDF-bound correction and its expected initialization impact. |
| [REVIEW_SUMMARY.md](discrete/direct-solver/history/REVIEW_SUMMARY.md) | Superseded summary | Condenses the early review and repeats its now-superseded claim that training infrastructure fully matched the original. |
| [SESSION_SUMMARY.md](discrete/direct-solver/history/SESSION_SUMMARY.md) | Session record | Captures the original review/setup/training-start session, environment, datasets, and the same early correctness assumptions. |
| [TRAINING_STATUS.md](discrete/direct-solver/history/TRAINING_STATUS.md) | Run snapshot | Historical in-progress status at roughly 32% of the pre-fix 100-epoch maze run; final metrics are in `TRAINING_RESULTS.md`. |

### Dynamic-grid trajectory/world-model experiments

| Document | Kind | Short finding/result |
|---|---|---|
| [BENCHMARK_RESULTS.md](discrete/dynamic-world-model/results/BENCHMARK_RESULTS.md) | Results compendium | Scaling the gated HRM to 28.97M parameters on 8 H100s reached 68% success vs a 66% LSTM baseline; smaller HRMs trailed. It also records that diffusion planners were limited by missing future-obstacle information. |
| [LSTM_VS_HRM_EXPERIMENT.md](discrete/dynamic-world-model/results/LSTM_VS_HRM_EXPERIMENT.md) | Results/design | On the 20x20 dynamic-grid benchmark, HRM 3M/10M reached 0.71 success vs LSTM 0.67-0.69, while small HRM lagged. Its appended Preset M section is a follow-up design, not its final result. |
| [ONLSTM_VS_HRM_EXPERIMENT_PRESETM_V2.md](discrete/dynamic-world-model/design/ONLSTM_VS_HRM_EXPERIMENT_PRESETM_V2.md) | Design/template | Defines the harder 32x32 Preset M+ benchmark, multi-step training, map patches, and ablations. The document itself ends with a results template; the surveyed result, summarized in the compendium, favored ON-LSTM (0.388 mean success) over HRM (0.265). |
| [lstm-augmented-astar-2025.pdf](discrete/dynamic-world-model/references/lstm-augmented-astar-2025.pdf) | Background literature | External paper on LSTM/Kalman-augmented A\* in dynamic environments. It motivated the trajectory-prediction line but is not evidence produced by this repository. |

### Discrete learned-heuristic and transfer line

| Document | Kind | Short finding/result |
|---|---|---|
| [experiment_writeup_last_two_runs.md](discrete/learned-heuristic/results/experiment_writeup_last_two_runs.md) | Results | Both the original and map-scale LoRA curricula were planner-level near-nulls: learned methods tracked static A\*, and the cleaner map-scale run isolated family B as the real OOD bottleneck. |
| [clean_transfer_experiment_blueprint.md](discrete/learned-heuristic/design/clean_transfer_experiment_blueprint.md) | Design | Reframes transfer around a Manhattan residual, global frame encoder, candidate-node head, matched full-FT vs LoRA, alpha calibration, and explicit frontier-order diagnostics. The later clean run still returned a negative. |
| [EXPERIMENT_RESULTS_COMPENDIUM.md](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_COMPENDIUM.md) | Main results compendium | Surveys the Modal history. Clean transfer and CondLoRA were negative/incomplete; pooled `avgbase__hrm` was the strongest completed learned-heuristic signal at +2.11pp vs matched Manhattan, while specialists usually lost. Residual TaskLoRA v2 was partial/contaminated at survey time. |
| [2026-06-15-residual-tasklora-eval-speedup.md](discrete/learned-heuristic/plans/2026-06-15-residual-tasklora-eval-speedup.md) | Performance plan | Defines profiling, a diagnostics-off mode, heuristic caching, budget pruning, and sharding changes while preserving headline metrics. Its measured outcome is documented in `FAST_EVAL.md`. |
| [FAST_EVAL.md](discrete/learned-heuristic/operations/FAST_EVAL.md) | Operations/result | `EVAL_DIAG=0` removes a diagnostic-only dynamic program and enables heuristic caching; after fixing env forwarding, a representative remote shard improved from about 100s/episode to 2.6s/episode (~39x) without changing success/expansion metrics. |
| [2026-06-23-learned-focal-search-design.md](discrete/learned-heuristic/design/2026-06-23-learned-focal-search-design.md) | Design spec | Pre-registers focal A\*ε: admissible Manhattan defines the band and the learned signal only ranks within it, with matched-expansion gates and bounded-suboptimality invariants. |
| [2026-06-23-learned-focal-search.md](discrete/learned-heuristic/plans/2026-06-23-learned-focal-search.md) | Implementation plan | Breaks the focal planner, wiring, tests, and local go/no-go benchmark into TDD tasks. It contains no independent result. |
| [FOCAL_SEARCH_RESULTS.md](discrete/learned-heuristic/results/FOCAL_SEARCH_RESULTS.md) | Focused results | Diagnosed ρ≈0.99 ranking but scale-flat magnitude and showed the same pooled model, used as a focal tie-breaker at `w=1.0`, reduced expansions 6-15% with no success regressions. Experts ranked identically to the base. |
| [EXPERIMENT_RESULTS_FOCAL_REDESIGN.md](discrete/learned-heuristic/results/EXPERIMENT_RESULTS_FOCAL_REDESIGN.md) | Main follow-up report | Integrates the clean rerun, infrastructure repair, rank-vs-magnitude diagnosis, both-backbone focal result, and the negative specialist result. It is the canonical continuation of the compendium. |

## Continuous-PRM experiments

### Program-level documents

| Document | Kind | Short finding/result |
|---|---|---|
| [CONTINUOUS_PRM_STORY.md](continuous/program/CONTINUOUS_PRM_STORY.md) | Main synthesis | Summarizes C6-C10 plus C9b: learned heuristics and transfer work; explicit hierarchy, future-window awareness, and interpolation cleverness do not beat strong simpler baselines on the current substrate. |
| [CONTINUOUS_PRM_STRATEGY.md](continuous/program/CONTINUOUS_PRM_STRATEGY.md) | Strategy | Diagnoses substrate/headroom saturation and ranks next directions: a higher-headroom compositional or high-DOF substrate, graph-native modeling, a weakened-base interpolation retest, broader objectives, or publication-scale consolidation. |

### C1-C4: benchmark and specialization ladder

| Document | Kind | Short finding/result |
|---|---|---|
| [continuous_prm_experiment_ladder_repo_coupled.md](continuous/c01-c04/continuous_prm_experiment_ladder_repo_coupled.md) | Design + smoke record | Defines C1 Euclidean sanity, C2 pooled HRM/ON-LSTM, C3 bounded task-LoRA experts, and C4 nearest/RBF prediction mixtures. All four passed CPU smoke checks; the early easy suites largely saturated, motivating C5's calibrated hard maps. |

### C5: hard maps and richer scalar encoder

| Document | Kind | Short finding/result |
|---|---|---|
| [continuous_prm_c5_hard_obstacle_encoder_spec.md](continuous/c05/continuous_prm_c5_hard_obstacle_encoder_spec.md) | Design + results | Calibrated 192/k7 hard maps into a binding band. ON-LSTM raised success from 0.525 to 1.000 on maze and 0.595 to 0.962 on maze-dense with strong corrected significance; HRM saturated at a constant cap and exactly matched Euclidean despite tuning/soft-cap follow-ups. |

### C6: value-field representation

| Document | Kind | Short finding/result |
|---|---|---|
| [continuous_prm_c6_heatmap_value_field_spec.md](continuous/c06/design/continuous_prm_c6_heatmap_value_field_spec.md) | Design spec | Replaces independent scalar residuals with goal-conditioned fields, adds oracle/U-Net/ON-LSTM/HRM gates and search-aware losses, and records an oracle feasibility probe with large success and expansion gains. |
| [C6_RESULTS.md](continuous/c06/results/C6_RESULTS.md) | Results | With adequate training, all field models significantly beat Euclidean; HRM reached 0.975 vs 0.625 success on maze at B144. Multi-suite training lifted rooms to 0.875 and held-out dense maze to 1.0, reversing C5's HRM collapse. |

### C7: integration comparison

| Document | Kind | Short finding/result |
|---|---|---|
| [2026-06-27-c7-integration-comparison-design.md](continuous/c07/design/2026-06-27-c7-integration-comparison-design.md) | Design spec | Pre-registers a matched scalar-vs-field and additive-vs-focal matrix, calibrated budgets, paired statistics, OOD axes, and oracle-gap accounting. |
| [2026-06-27-c7-integration-comparison.md](continuous/c07/plans/2026-06-27-c7-integration-comparison.md) | Implementation plan | Specifies the providers, hard-map suites, integrity tests, calibration, trainers, evaluation, analysis, and publication outputs. It contains no independent result. |
| [C7_RESULTS.md](continuous/c07/results/C7_RESULTS.md) | Results | Additive learned heuristics cut expansions about 15-48% and improved success across all six suites. Scalar and field were both viable; additive beat focal because Euclidean was loose, establishing baseline-tightness-dependent integration. |

### C8: dynamics and temporal-awareness spotlight

| Document | Kind | Short finding/result |
|---|---|---|
| [2026-06-27-c8-dynamics-design.md](continuous/c08/design/2026-06-27-c8-dynamics-design.md) | Design spec | Extends C7 to space-time A\*, pre-registers aware-vs-blind twins, dynamic suites, makespan/suboptimality metrics, and oracle/integrity gates. |
| [2026-06-27-c8-dynamics.md](continuous/c08/plans/2026-06-27-c8-dynamics.md) | Implementation plan | Defines dynamic maps, time-aware providers, collection/training/evaluation, performance repairs, statistics, and staged local/cluster execution. It contains no independent result. |
| [C8_RESULTS.md](continuous/c08/results/C8_RESULTS.md) | Results | Heavy confirmation showed learned heuristics cutting expansions 65-95% with significant success gains, additive again beating focal, and time-blind models beating aware models in 7 significant cells vs 1 aware win. U-Net was strongest; future occupancy helped the planner/oracle but not the learned heuristic. |

### C9: static few-shot transfer

| Document | Kind | Short finding/result |
|---|---|---|
| [2026-06-29-c9-transfer-design.md](continuous/c09/design/2026-06-29-c9-transfer-design.md) | Design spec | Defines ADAPT/TEST separation, K curves, zero-shot/LoRA/full-FT/scratch arms, held-out families, matched worlds, and transfer gates. |
| [2026-06-29-c9-transfer.md](continuous/c09/plans/2026-06-29-c9-transfer.md) | Implementation plan | Implements the few-shot adapters, loaders, evaluation curves, statistics, and result writeup. It contains no independent result. |
| [C9_RESULTS.md](continuous/c09/results/C9_RESULTS.md) | Results | Established the few-shot crossover: LoRA safely matches the strong base from K=1 but plateaus; full-FT overfits at K=1 then becomes best around K>=8-16; scratch needs much more data. Transfer gains over Euclidean were significant. |

### C9h: matched-compute hardening

| Document | Kind | Short finding/result |
|---|---|---|
| [2026-06-29-c9-hardening-design.md](continuous/c09h/design/2026-06-29-c9-hardening-design.md) | Design spec | Removes C9's compute/bound confounds, adds bounded and unbounded LoRA at matched recipes, and extends adaptation to field U-Net through conv-LoRA. |
| [2026-06-29-c9-hardening.md](continuous/c09h/plans/2026-06-29-c9-hardening.md) | Implementation plan | Specifies conv-LoRA tests, matched-compute trainers, provider loading, evaluation, and comparison reports. It contains no independent result. |
| [C9H_RESULTS.md](continuous/c09h/results/C9H_RESULTS.md) | Results | Confirmed the crossover at matched compute, showed bounded and unbounded LoRA nearly identical (median delta 0.000 +/- 0.008), and found field full-FT's highest ceiling (rooms-large ratio 0.404 at 0.97 success). Low rank, not the clamp, caused robustness and plateauing. |

### C9b: transfer under dynamics

| Document | Kind | Short finding/result |
|---|---|---|
| [2026-06-30-c9b-dynamics-transfer-design.md](continuous/c09b/design/2026-06-30-c9b-dynamics-transfer-design.md) | Design spec | Ports the C9/C9h adaptation matrix to C8's aware/blind space-time substrate and pre-registers both crossover replication and temporal-awareness gates. |
| [2026-06-30-c9b-dynamics-transfer.md](continuous/c09b/plans/2026-06-30-c9b-dynamics-transfer.md) | Implementation plan | Defines temporal datasets, scalar/field adapters, source reuse, ADAPT/TEST integrity, evaluation, and a success-aware aware-vs-blind probe. It contains no independent result. |
| [C9B_RESULTS.md](continuous/c09b/results/C9B_RESULTS.md) | Results | Transfer still beat scratch and Euclidean-time, but the static crossover disappeared because K=1 dynamic world already supplies about 25k `(node,t)` targets. Time-aware models never beat blind in any of 9 full-FT@K16 headline cells. |

### C10: zero-label adapter interpolation

| Document | Kind | Short finding/result |
|---|---|---|
| [2026-06-29-c10-interp-design.md](continuous/c10/design/2026-06-29-c10-interp-design.md) | Design spec | Defines a verified bracketing family grid and compares RBF weight merges, prediction mixes, nearest, uniform, and pooled zero-shot with no target labels. |
| [2026-06-29-c10-interp.md](continuous/c10/plans/2026-06-29-c10-interp.md) | Implementation plan | Specifies merge-correctness tests, family construction, source-expert training, bake/mix providers, evaluation, and analysis. It contains no independent result. |
| [C10_RESULTS.md](continuous/c10/results/C10_RESULTS.md) | Results | RBF weights correctly selected the target's axis with 98.6-99.8% mass and every learned arm beat Euclidean, but interpolation did not beat zero-shot, RBF did not beat uniform/nearest, and weight-space matched prediction-space. |

### C11: compositional missions

| Document | Kind | Short finding/result |
|---|---|---|
| [2026-07-07-c11-headroom-probe.md](continuous/c11/plans/2026-07-07-c11-headroom-probe.md) | Probe plan | Pre-registers the cheap G0-H oracle-vs-leg-sum gate over mission length K before authorizing the full phase. The result is in `C11_HEADROOM.md`. |
| [C11_HEADROOM.md](continuous/c11/results/C11_HEADROOM.md) | Results | The gate passed in all practical terms: oracle/leg-sum ratios were 0.082-0.225 and shrank monotonically with K in every config; config A at K=8 cleared the gate with a 0.082 ratio. |
| [2026-07-07-c11-compositional-mission-design.md](continuous/c11/design/2026-07-07-c11-compositional-mission-design.md) | Design spec | Uses the passed headroom to define compositional waypoint/keys-doors missions, explicit MLP/U-Net/GNN/trace-model/iterative-refiner controls, structure-dose and depth-of-compute gates, and honest closure criteria. |
| [2026-07-07-c11-mission.md](continuous/c11/plans/2026-07-07-c11-mission.md) | Implemented plan | Defines the mission data/oracles, six arms, analysis, tests, and scaled addendum now represented in the canonical result. |
| [C11_RESULTS.md](continuous/c11/results/C11_RESULTS.md) | Results | No preregistered hierarchy/depth dose response. Global-input models lead at shallow K, recurrent/ACT arms are fragile, and forced HRM-v2 compute is flat; the completed scaled addendum preserves U-Net over HRM. |

### C12: persistent hidden-regime planning

| Document | Kind | Short finding/result |
|---|---|---|
| [2026-07-10-c12-persistent-hierarchical-planning-design.md](continuous/c12/design/2026-07-10-c12-persistent-hierarchical-planning-design.md) | Approved design + amendment | Separately gates persistent dynamics and tied refinement; freezes recurring hazards, OOD slices, widths, and compute rules before pilot. |
| [2026-07-10-c12-persistent-hierarchical-planning.md](continuous/c12/plans/2026-07-10-c12-persistent-hierarchical-planning.md) | Completed implementation plan | C12-A pilot and the independently gated C12-B full sequence are complete; K16 rejection and CLI/module deviations are documented. |
| [C12_RESULTS.md](continuous/c12/results/C12_RESULTS.md) | Combined canonical result | C12-A closes `strong_negative` at one-seed pilot scale. C12-B improves with cycles in every cell but fails its K-dose response; C/K8 supplies a localized tied-control win with explicit suboptimal-path caveats. |

### C13: state-conditioned heuristic revalidation

| Document | Kind | Short finding/result |
|---|---|---|
| [2026-07-16-c13-state-conditioned-heuristic-design.md](continuous/c13/design/2026-07-16-c13-state-conditioned-heuristic-design.md) | Implemented design / gated preregistration | Separates strict geometry from bounded local observations, proves why literal `constant-E` cannot be the old additive residual, and adds the fresh-start rollout-value amendment plus same-search controls. |
| [2026-07-16-c13-state-conditioned-heuristic.md](continuous/c13/plans/2026-07-16-c13-state-conditioned-heuristic.md) | Completed gated plan | Tracks the full C13-A through C13-M sequence and records the final benchmark-level completion plus the remaining professor-review and timing-hardening work. |
| [C13_INITIAL_AUDIT.md](continuous/c13/results/C13_INITIAL_AUDIT.md) | Preliminary audit | One-step guidance is valid but too weak to justify HRM/ON-LSTM imitation; deeper backups gain search efficiency only by traversing more graph layers. The +10% density probe slightly improves path cost while increasing build and absolute search work. |
| [C13B_ROLLOUT_RANKER_SMOKE.md](continuous/c13/results/C13B_ROLLOUT_RANKER_SMOKE.md) | Implementation/provenance smoke | Fresh-start rollout labels avoid shortest-path supervision and hidden-history label aliasing. Matched FOCAL controls show no stable learned gain: the pipeline passes, but the signal gate fails. |
| [C13B_IDENTIFIABILITY_STUDY.md](continuous/c13/results/C13B_IDENTIFIABILITY_STUDY.md) | Completed causal diagnostic | Local signal is learnable, but exact rollout values do not improve the intended search, ON-LSTM padding breaks transfer, and narrow FOCAL suppresses useful but unsafe learned ordering. |
| [C13C_CERTIFIED_SEARCH.md](continuous/c13/results/C13C_CERTIFIED_SEARCH.md) | Completed bounded-integration gate | A separate learned-incumbent search plus fresh Euclidean certifier is correct but too duplicative: the oracle ceiling loses all six comparisons at the primary `w=1.10`. |
| [C13D_SHARED_QUEUE_ORACLE.md](continuous/c13/results/C13D_SHARED_QUEUE_ORACLE.md) | Passed oracle integration ceiling | Shared `g`/queue state removes the independent wrapper's duplicated work and wins all six `w=1.10` comparisons, establishing the ceiling subsequently tested by C13-E. |
| [C13E_SHARED_QUEUE_EXACT_TARGET.md](continuous/c13/results/C13E_SHARED_QUEUE_EXACT_TARGET.md) | Failed exact-target gate | The unchanged shared search certifies every path, but exact rollout averages `+1.33` expansions and wins only 2/6 primary comparisons; learned providers remain blocked. |
| [2026-07-17-c13-current-state-literature-and-next-target.md](continuous/c13/design/2026-07-17-c13-current-state-literature-and-next-target.md) | Literature/method decision | Connects LoHA*, RTAA*, MHA*, and search-effort learning to the bounded-observation target and integration sequence. |
| [2026-07-17-c13i-current-state-vs-map-conditioned.md](continuous/c13/design/2026-07-17-c13i-current-state-vs-map-conditioned.md) | C13-I preregistration | Freezes the first live comparison with all C7 map-conditioned providers; the one-suite current model fails outside maze. |
| [2026-07-17-c13j-multisuite-current-state-training.md](continuous/c13/design/2026-07-17-c13j-multisuite-current-state-training.md) | C13-J preregistration | Freezes suite-balanced training and a disjoint six-suite development block; distribution-only repair fails. |
| [2026-07-17-c13k-local-bellman-integration.md](continuous/c13/design/2026-07-17-c13k-local-bellman-integration.md) | C13-K preregistration | Tests one radius-bounded local Bellman backup and isolates integration as the decisive mechanism. |
| [2026-07-17-c13l-local-backup-scale-calibration.md](continuous/c13/design/2026-07-17-c13l-local-backup-scale-calibration.md) | C13-L preregistration | Rejects all direct arms under the absolute 1.10 ceiling while exposing an empirical matched-quality frontier. |
| [2026-07-17-c13m-matched-quality-confirmation.md](continuous/c13/design/2026-07-17-c13m-matched-quality-confirmation.md) | C13-M preregistration | Freezes alpha 1.50, the 144-world untouched cohort, relative path-quality margins, and the separate bounded safety control before generation. |
| [2026-07-17-c13n-hrm-substitution.md](continuous/c13/design/2026-07-17-c13n-hrm-substitution.md) | C13-N preregistration | Freezes an architecture-only `flat_mlp -> hrm_trimmed` substitution, a six-cell development grid, matched field/flat controls, and a confirmation-only-after-pass rule. |
| [2026-07-17-c13o-hrm-summary-last-alignment.md](continuous/c13/design/2026-07-17-c13o-hrm-summary-last-alignment.md) | C13-O preregistration | Moves only the type-tagged summary token to the final valid recurrent position, with identical initialization/training and frozen trimmed/flat controls. |
| [C13F_M_CURRENT_STATE_RESULT.md](continuous/c13/results/C13F_M_CURRENT_STATE_RESULT.md) | Canonical completed result | Documents every C13-F through C13-M mechanism test and the confirmed 15.95% expansion reduction versus complete-map field HRM, with path-quality, timing, and claim-scope caveats. |
| [C13N_HRM_SUBSTITUTION_RESULT.md](continuous/c13/results/C13N_HRM_SUBSTITUTION_RESULT.md) | Completed architecture diagnostic | The fixed HRM endpoint has useful pooled signal but fails suite robustness and matched-MLP path-quality gates; no untouched confirmation is authorized. |
| [C13O_HRM_ALIGNMENT_RESULT.md](continuous/c13/results/C13O_HRM_ALIGNMENT_RESULT.md) | Completed readout-alignment diagnostic | Summary-last helps trimmed HRM at iteration 6 but not at the fixed endpoint; no cell passes the field/flat/readout method gate, so confirmation remains untouched. |

## Evidence and scope notes

- Generated reports are **not duplicated** into this tree. See [continuous generated evidence](continuous/GENERATED_EVIDENCE.md) and [discrete generated evidence](discrete/GENERATED_EVIDENCE.md).
- Raw CSV/JSON/checkpoints are evidence, not authored documentation; their most useful entry points are cataloged beside the generated reports.
- `HRM-v2/` installation, porting, API, and project-summary documents remain with the implementation because they are engineering references rather than experiment records.
- The C6-C10/C9b conclusions are local RTX 5090 validations unless their document says otherwise; many explicitly call for cluster/multi-seed confirmation before publication-grade claims.
- Historical documents are not silently rewritten to agree with later findings. The index labels superseded claims and points to the audit that corrected them.
