# Continuous-PRM Experiment Documentation

Use the [master experiment index](../README.md) for a short finding/result summary for every major document.

- [`program/`](program/) contains the cross-phase narrative and strategy.
- [`c01-c04/`](c01-c04/) contains the original benchmark, pooled-base, TaskLoRA, and RBF-mixture ladder.
- [`c05/`](c05/) through [`c13/`](c13/) contain stage-specific design, implementation plans, and authored results.
- [`GENERATED_EVIDENCE.md`](GENERATED_EVIDENCE.md) catalogs all 51 generated Markdown reports and the main raw-evidence entry points.

The implementation remains under [`hrm-cloud/continuous_prm/`](../../../hrm-cloud/continuous_prm/). Generated reports stay there with their raw outputs by design.

Completed frontier: [C12](c12/results/C12_RESULTS.md) confirms hidden-dynamics memory headroom but closes `strong_negative` for C12-A; C12-B shows recurrent-cycle gains without a K-dose response and only a localized C/K8 tied-control win.

Completed current-state result: [C13-F through C13-M](c13/results/C13F_M_CURRENT_STATE_RESULT.md)
tracks the failed calibration, exact-local, distribution-only, and static-integration
controls through the successful local-Bellman mechanism. On 144 untouched worlds,
the fixed bounded-observation arm averages `68.31` expansions versus `81.26` for
complete-map field HRM (paired delta `-12.96`, 95% CI `[-16.30, -9.74]`) with
lower empirical mean/max path-cost ratios. The direct arm is not formally bounded and its unoptimized feature builder is not a wall-clock win; a separate `w=1.10` FOCAL control passes all 144 safety certificates.

Architecture diagnostics: [C13-N](c13/results/C13N_HRM_SUBSTITUTION_RESULT.md)
replaces only the successful flat value model with `hrm_trimmed`. Its fixed
endpoint has a promising pooled development delta versus field HRM
(`-8.63`, 95% CI `[-16.67, -1.21]`) but improves only 3/6 suite means,
has worse matched-MLP path quality, and fails the preregistered development
gate. No new confirmation cohort was opened.

[C13-O](c13/results/C13O_HRM_ALIGNMENT_RESULT.md) then moves the already
type-tagged summary token to the final recurrent readout position. The fixed
endpoint still beats field HRM in pooled expansions (`-6.54`, 95% CI
`[-13.25, -0.33]`) but again improves only 3/6 suites, fails matched-flat
path-quality margins, and is worse than trimmed HRM by `+2.08` expansions
with an inconclusive CI. Two iteration-6 cells improve trimmed HRM directly,
