# C13 Initial State-Target and Density Audit

**Status:** preliminary target-selection probe, not a final experiment result  
**Date:** 2026-07-16  
**Suite:** `C_hard_maze`  
**Roadmap:** `k=7`; 192 nodes unless swept

Raw artifacts:

- [bounded-backup depth audit](../../../../../hrm-cloud/continuous_prm/runs/c13_state_probe/results/c13_backup_depth_audit.csv)
- [density audit](../../../../../hrm-cloud/continuous_prm/runs/c13_state_probe/results/c13_density_audit_raw.csv)
- [one-step validity checks](../../../../../hrm-cloud/continuous_prm/runs/c13_state_probe/results/c13_one_step_properties.csv)
- [constant-minus-E semantics check](../../../../../hrm-cloud/continuous_prm/runs/c13_state_probe/results/c13_semantics_audit.json)

## Executive decision

Do **not** begin HRM/ON-LSTM training on the one-step target yet.

The target is leak-resistant, admissible, consistent, and nonzero, but its direct analytical version saves only about one expansion at the canonical density. Training a large sequence model to imitate that target would test approximation capacity without enough planner-level headroom to answer the scientific question.

Keep the one-step arm as the clean current-state/local-action control. Resolve whether observation history or rollout/TD supervision is methodologically allowed before implementing a stronger learned target.

## 1. Constant-minus-E semantics

The numerical semantics audit passed exactly:

- converting the maximized proximity value `C-E` back to a minimized rank reproduces Euclidean (`max error = 0`);
- inserting `C-E` as the existing additive residual produces a constant heuristic (`range = 0`).

Therefore the literal suggestion is a useful geometry-only control, not a drop-in replacement for the C6 residual.

## 2. One-step target validity and signal

The one-step Euclidean Bellman backup passed all checked conditions:

- Euclidean dominance violation: `0`;
- admissibility violation against the evaluation oracle: `0`;
- graph-edge consistency violation: `0`;
- goal boundary violation: `0`.

A two-world collection smoke at 192/k7 produced positive residuals on about 92% of nodes, but the normalized residual was small (`mean ≈ 0.01`, `p95 ≈ 0.05–0.06`).

On a separate 30-connected-world target-selection cohort, the exact one-step heuristic improved 18 worlds, tied 12, never worsened, and saved only `0.97` expansions on average relative to Euclidean.

## 3. Bounded-relaxation depth curve

This diagnostic applies a fixed number of Bellman backups starting from Euclidean. Depths above one are **not** declared current-state-only; they quantify how quickly useful signal requires successor-of-successor traversal.

| Backup depth | Mean expansions | Mean saved vs Euclidean | Mean start `h / h*` | Heuristic computation |
|---:|---:|---:|---:|---:|
| 0 | 138.23 | 0.00 | 0.41 | 0.02 ms |
| 1 | 137.27 | 0.97 | 0.42 | 0.24 ms |
| 2 | 136.30 | 1.93 | 0.43 | 0.47 ms |
| 4 | 134.47 | 3.77 | 0.44 | 0.91 ms |
| 8 | 130.60 | 7.63 | 0.47 | 1.77 ms |
| 16 | 121.93 | 16.30 | 0.52 | 3.51 ms |

All depths had zero measured admissibility and consistency violations. The improvement is roughly proportional to how far information is propagated. A meaningful search gain appears only once the method has traversed multiple graph layers, which is precisely the methodological boundary raised in the meeting.

## 4. Preliminary density curve

The density cohort contains 30 fixed world seeds. A disconnected PRM is retained as a failure rather than silently replaced, so connectivity is part of the outcome.

| Nodes | Connected | Mean edges (connected) | Mean degree | Mean build time | Euclidean expansions | One-step expansions |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 6/30 (20%) | 415.0 | 6.48 | 100.6 ms | 95.67 | 95.50 |
| 160 | 11/30 (37%) | 548.8 | 6.86 | 134.6 ms | 117.64 | 117.00 |
| 192 | 11/30 (37%) | 680.6 | 7.09 | 152.1 ms | 133.36 | 132.27 |
| 211 (+10%) | 11/30 (37%) | 754.3 | 7.15 | 165.9 ms | 148.09 | 146.64 |
| 256 | 12/30 (40%) | 945.4 | 7.39 | 189.5 ms | 178.00 | 177.17 |

Absolute expansions rise with graph size. The fraction of nodes expanded is more stable, so absolute expansions alone should not be interpreted as worse search efficiency across densities.

For the preregistered `192 → 211` comparison, nine worlds were connected at both densities:

- graph-optimal path cost changed by `-0.0097` on average (five of nine improved);
- absolute Euclidean expansions increased by `13.78`;
- expansion fraction changed by only `+0.0028`;
- PRM build time increased by `18.3 ms`.

The +10% setting therefore gives a very small path-quality improvement in this probe, no normalized expansion benefit, and higher construction/search work. This is preliminary because the paired sample is only nine worlds and only one suite was tested.

## 5. Next methodological decision

The remaining question should be stated explicitly to the professor:

> Does “function of the current state” permit bounded local observations and returns learned from environment interaction, provided no full-map shortest-path result is used as an input or label?

If **no**, Euclidean/goal proximity is the honest endpoint and a learned obstacle-aware heuristic is not identifiable from `(position, goal)` alone.

If **yes**, the next defensible learned arm is an observation-conditioned value/ranker trained from rollout or TD targets and integrated through Euclidean-anchored focal search. It should not be called admissible unless separately calibrated or bounded.

## 6. C13-B update

The working interpretation now permits bounded local observations and rollout outcomes that never use a shortest-path result as an input or label. C13-B was therefore implemented with fresh-start Monte Carlo returns and matched FOCAL controls.

See [C13B_ROLLOUT_RANKER_SMOKE.md](C13B_ROLLOUT_RANKER_SMOKE.md) for the pipeline/provenance smoke and [C13B_IDENTIFIABILITY_STUDY.md](C13B_IDENTIFIABILITY_STUDY.md) for the completed causal audit. The latter finds that bounded local observations contain learnable signal, but the current rollout target, padding/readout contract, and narrow FOCAL integration do not produce a stable bounded planning gain. The full multi-suite run remains gated.
