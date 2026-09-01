# C13-N preregistration: HRM substitution in the successful local-backup method

**Frozen:** 2026-07-17, before C13-N model training or development evaluation.

## Question

C13-M established that a current-state local representation can improve exact, no-reopen A* when a learned value is integrated through one radius-0.2 local Bellman backup. Its successful learned component was a flat MLP. C13-N asks whether the same method remains effective when that learned component is replaced by the repository's recurrent HRM ranker.

This is an architecture-substitution study, not a new representation or integration search. The primary comparison changes only the model family from `flat_mlp` to `hrm_trimmed`.

## Locked method

- **Representation:** one summary token, 32 radial-observation tokens, and up to 24 valid-action tokens; token width 16 and maximum sequence length 57.
- **Observation model:** 32 rays, radius 0.2, 32 samples per ray.
- **Graph:** 192-node continuous PRM, 7-nearest-neighbor construction.
- **Target:** exact graph cost-to-go, learned through the existing self-bootstrapped limited-horizon Bellman loop.
- **HRM model:** `RecurrentRanker("hrm", readout="trimmed")`, whose backbone is `DeepSapientHRMBackbone(hidden_dim=64, k_step=2, num_heads=4, num_layers=1)`.
- **Search integration:** predict bootstrap node values, perform exactly one radius-0.2 local Bellman backup using learned exit values, then rank nodes as `euclidean + alpha * (local_value - euclidean)` in exact no-reopen A*.
- **Primary operating point:** outer iteration 8 and `alpha=1.5`, matching C13-M.
- **Training:** 8 outer iterations, 5 epochs per iteration, AdamW, learning rate 5e-4, weight decay 1e-4, batch size 128, hidden width 64, model seed 17413 under the existing trainer's model-seed derivation.
- **Numerics:** HRM training may run on CUDA for feasibility with TF32 disabled. All development and confirmation search evaluation runs on CPU. The frozen C13-J flat checkpoint remains the canonical flat comparator; the different training backend is recorded as an implementation limitation, not silently treated as an architectural variable.

## Frozen cohorts and cache reuse

C13-N must reuse the exact saved C13-J balanced cohorts and feature caches in `runs/c13_lhbl_multisuite`:

- training: maze, rooms, and spiral; 32 worlds per suite; seed offset 0;
- validation: maze, rooms, and spiral; 8 worlds per suite; seed offset 5,000,000;
- development: all six suites; 4 worlds per suite; seed offset 10,000,000.

Before training, the harness must reconstruct the bundle metadata and verify it against C13-J's saved `cohorts.json`, including suite, world seed, graph seed, node/edge counts, and feature-cache SHA-256. Any mismatch is a hard stop. C13-N must not regenerate or replace C13-J feature caches.

## Development cells

The fixed primary cell is iteration 8, alpha 1.5. It is reported regardless of outcome and is the only cell called a literal C13-M architecture substitution.

A small, preregistered robustness grid is also evaluated on the frozen C13-J development cohort:

- outer iteration in `{4, 6, 8}`;
- alpha in `{1.0, 1.5}`.

Iteration 6 is included because the earlier one-suite HRM diagnostic showed its clearest non-collapsed predictions there; iteration 4 checks an earlier state and iteration 8 is the frozen endpoint. Alpha 1.0 is the unamplified local-backup value and alpha 1.5 is the C13-M operating point. No other iteration, alpha, radius, representation, training setting, checkpoint, or search rule may be selected after seeing C13-N development results.

For every HRM cell, evaluate the frozen flat-MLP checkpoint at the same iteration and alpha, plus the C7 field-HRM and scalar-HRM comparators on the identical worlds. Report path validity, expansions, path cost, graph-optimal cost ratio, paired deltas, paired bootstrap confidence intervals, per-suite means, and representation/inference/backup/search timing.

## Development gates

The following gate determines whether a C13-N HRM candidate is authorized for a fresh confirmation block:

1. all 24 HRM searches and reconstructed paths are valid;
2. paired HRM-minus-field-HRM expansion delta has a 95% bootstrap confidence-interval upper endpoint below zero;
3. HRM-minus-field-HRM mean expansion delta is negative in at least four of six suites;
4. HRM mean graph-optimal cost ratio is no more than 0.005 above field HRM;
5. HRM maximum graph-optimal cost ratio is no more than 0.02 above field HRM.

If multiple robustness cells pass, select the one with the lowest mean expansions, then the lowest mean cost ratio, then the earlier iteration, then the lower alpha. This selection is development-only and must be confirmed untouched.

The flat comparison has a separate interpretation:

- **architecture win:** the paired HRM-minus-flat expansion confidence-interval upper endpoint is below zero while HRM is no worse than flat by the same mean/max cost-ratio margins;
- otherwise, report the paired estimate without claiming that HRM improves on the flat model.

Beating the field comparator can authorize confirmation even if HRM does not beat the flat model. In that case the result supports compatibility of HRM with the method, not superiority of HRM over the MLP.

## Untouched confirmation, only if authorized

If and only if a development cell passes all five gates, evaluate it without retuning on 144 newly generated worlds: 24 per suite across all six suites, using seed offset 20,000,000. The harness must prove that world and graph seeds are disjoint from every C13 cohort it can enumerate, including the C13-M offset-15,000,000 confirmation block.

The selected HRM cell, the matched frozen flat cell, field HRM, scalar HRM, and Euclidean A* are all evaluated on the same confirmation worlds. The confirmation verdict uses the same five gates against field HRM. HRM-versus-flat remains a separately reported paired architecture comparison.

No parameter, checkpoint, seed block, representation, integration rule, or gate may change after development results are observed. A failed development gate ends C13-N without confirmation. A failed confirmation is a rejection, not a prompt for another seed block.

## Claim boundary

Even a positive result would show only that this particular HRM implementation is compatible with the successful current-state/local-backup pipeline on the known PRM observation simulator. It would not establish a formal search bound, a wall-clock speedup, map-free sensing, or general HRM architectural superiority. Absolute coordinates remain present in the representation and can encode suite spatial priors.
