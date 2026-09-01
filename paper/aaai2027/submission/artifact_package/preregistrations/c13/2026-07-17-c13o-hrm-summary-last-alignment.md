# C13-O preregistration: summary-last HRM readout alignment

**Frozen:** 2026-07-17, before C13-O model training or development evaluation.

## Question

C13-N showed that `hrm_trimmed` can be integrated mechanically into the successful C13-M local-Bellman-backup method, but it did not robustly preserve the flat model's result. The frozen primary cell beat field HRM in the pooled comparison, yet it won on only three of six suite means, did not beat the matched flat model, and exceeded the preregistered flat-relative path-cost margins.

C13-O tests one specific explanation: the recurrent readout is misaligned with the token order. In `hrm_trimmed`, the summary token is first, followed by angular ray tokens and a variable number of action tokens. The final recurrent context is therefore the last valid action token, and its position changes with node degree. C13-O moves the already type-tagged summary token to the final valid position and changes nothing else.

This is a readout/order-alignment study. It is not a new representation, target, optimizer, search integration, or persistence study.

## Locked intervention

- **Candidate:** `hrm_summary_last`, implemented by moving token 0 to the end of each unpadded sequence immediately before the existing HRM backbone.
- **Frozen recurrent control:** the existing C13-N `hrm_trimmed` checkpoints.
- **Frozen flat control:** the existing C13-J `flat_mlp` checkpoints.
- **Representation:** the exact C13-J/C13-N tensor: one type-tagged summary token, 32 type-tagged angular ray tokens, and up to 24 type-tagged action tokens; width 16 and padded length 57.
- **Graph:** 192 total PRM nodes including start and goal, with 7-nearest-neighbor construction.
- **Observation:** 32 bounded rays, radius 0.2 of side length, and 32 samples per ray.
- **Target:** the existing self-bootstrapped limited-horizon Bellman target. `Roadmap.dist_to_goal` is not read by the training target.
- **Search integration:** predicted bootstrap values, exactly one radius-0.2 local Bellman backup with frozen exit values, then exact no-reopen A* ranked by `euclidean + alpha * (local_value - euclidean)`.
- **Training:** 8 outer iterations, 5 epochs per iteration, AdamW, learning rate 5e-4, weight decay 1e-4, batch size 128, hidden width 64, and model seed 17413 under the existing `+1009` model-seed derivation.
- **Devices:** training on CUDA with TF32 disabled; all search evaluation on CPU.

The `hrm_summary_last` and `hrm_trimmed` modules have identical parameter shapes. The harness must reset the same seed, instantiate both, and prove every initial state tensor is identical before training. The same frozen trainer, target-construction order, data-loader seed, optimizer, and batch schedule are used. The only intended intervention is valid-token order/readout context.

## Frozen cohorts and controls

C13-O reuses the exact saved C13-J cohorts and feature caches in `runs/c13_lhbl_multisuite`:

- training: maze, rooms, and spiral; 32 worlds per suite; seed offset 0;
- validation: maze, rooms, and spiral; 8 worlds per suite; seed offset 5,000,000;
- development: all six suites; 4 worlds per suite; seed offset 10,000,000.

Before training, the harness must:

1. re-hash every input and output named by the frozen C13-J integrity manifest;
2. re-hash every input and output named by the frozen C13-N integrity manifest;
3. replay C13-J bundle metadata and require exact agreement in suite, world seed, roadmap seed, node/edge counts, cache path, and cache SHA-256;
4. require every feature cache to be reused rather than created; and
5. record the source, trainer, model-definition, preregistration, wrapper, and control hashes in a binding artifact.

Any mismatch is a hard stop. C13-O must not overwrite the C13-J or C13-N run directories.

## Development cells

The fixed primary cell is outer iteration 8 with `alpha=1.5`. It is reported regardless of outcome.

The complete frozen grid is:

- outer iteration in `{4, 6, 8}`;
- alpha in `{1.0, 1.5}`.

On each of the 24 development worlds, evaluate `hrm_summary_last`, frozen `hrm_trimmed`, and frozen `flat_mlp` at the matched iteration and alpha, plus Euclidean A*, field HRM, and scalar HRM. Report validity, expansions, path cost, graph-optimal cost ratio, paired bootstrap confidence intervals, per-suite means, prediction dispersion, rank correlation before and after local backup, and representation/model/backup/search timing.

No other checkpoint, alpha, radius, representation, optimizer, target, integration rule, or selection criterion may be introduced after development results are observed.

## Development gate

A candidate authorizes untouched confirmation only if all of the following hold:

1. all 24 summary-last searches and reconstructed paths are valid;
2. the paired summary-last-minus-field-HRM expansion 95% bootstrap confidence-interval upper endpoint is below zero;
3. the summary-last-minus-field-HRM mean expansion delta is negative in at least four of six suite means;
4. summary-last mean and maximum graph-optimal cost ratios are no more than 0.005 and 0.02 above field HRM, respectively;
5. summary-last mean and maximum graph-optimal cost ratios are no more than 0.005 and 0.02 above the matched flat model, respectively; and
6. the paired summary-last-minus-trimmed expansion 95% bootstrap confidence-interval upper endpoint is below zero, while summary-last mean and maximum cost ratios are no more than 0.005 and 0.02 above trimmed HRM.

Conditions 1-5 test whether the aligned HRM preserves the successful method at acceptable path quality. Condition 6 is the direct test of the readout-order hypothesis. Summary-last versus flat expansion results are reported but are not required to claim that the readout fix helps the recurrent model.

If multiple cells pass, select the one with the lowest summary-last mean expansions, then lowest mean cost ratio, then earlier iteration, then lower alpha. The fixed primary cell remains separately reported.

## Untouched confirmation, only if authorized

If and only if a development cell passes all six grouped conditions, evaluate that exact cell on 144 newly generated worlds: 24 per suite across all six suites, using seed offset 20,000,000. World and roadmap seeds must be disjoint from all enumerable prior C13 train, validation, development, calibration, and confirmation cohorts, including the C13-M offset-15,000,000 block.

The selected summary-last model, matched trimmed HRM, matched flat MLP, field HRM, scalar HRM, and Euclidean A* are evaluated on every confirmation world. The same six grouped conditions are applied once. A failure is documented without retuning or another seed block.

## Interpretation boundary

- Passing both development and confirmation supports the narrow claim that summary-last ordering fixes a readout-alignment problem for this HRM implementation in the frozen local-backup method.
- Passing method conditions 1-5 but failing condition 6 supports compatibility, not the readout-order explanation.
- Failing conditions 1-5 rejects this alignment change as a recovery of the C13-M method under the frozen setup.
- Any outcome remains specific to the known-PRM observation simulator, absolute-coordinate representation, current-state bounded rays/actions, and one local Bellman backup. It does not establish map-free sensing, persistent HRM planning state, a formal search bound, wall-clock speedup, or general HRM superiority.

