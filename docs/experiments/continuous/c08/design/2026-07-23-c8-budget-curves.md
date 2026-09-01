# C8 budget-curves design: success as a function of expansion budget (descriptive)

**Frozen:** 2026-07-23, before the run is launched.
**Motivation:** reviewer-facing robustness view — show that the fixed-provider dynamic result is not an artifact of the single calibrated binding budget by reporting success-vs-budget curves for every provider on the same frozen cohort.

## Protocol

- **Run:** one evaluation pass at a single high budget per suite equal to 4× the canonical binding budget: crossing 600, maze 7200, dense maze 10000, rooms 5200, large rooms 2400, spiral 10000. Written to `runs/c8r_budget_curves/calibration.json` before launch.
- **Cohort:** the frozen fresh cohort (seed 999999, 50 maps/suite, same suite order) — identical worlds to the replication evals.
- **Checkpoints:** canonical seed-1234 field U-Net aware + blind (copied verbatim from `c8r_fresh_eval/checkpoints/`). No retraining.
- **Providers:** euclid, oracle, field U-Net aware, field U-Net blind; astar mode primary; focal w=1.0 recorded because the harness preset requires a focal band (reported separately, never pooled — mode filter `mode=="astar"`).
- **Derivation:** the space-time A\* expansion order is budget-independent and the loop admits a solve at budget B iff recorded solve expansions ≤ B (`while openq and expansions < budget`, goal test after increment). Success-vs-budget curves for all B ≤ 4×binding are therefore derived exactly from the single high-budget run by thresholding per-map solve expansions; no per-budget re-runs and no interpolation.
- **Readout (descriptive, no new hypothesis gates):** per-suite curves success(B) for each provider at B ∈ a dense grid up to 4×binding, with the binding budget marked; the budget (if any) at which Euclid reaches the learned provider's binding-budget success; the unlimited-horizon ceiling (fraction of maps solvable within t_max) where the budget stops binding.
- **Reporting rule:** curves are reported as-is; no recalibration, reselection, or retraining in response to their shapes. If Euclid closes the gap at moderate budgets on some suite, that is reported verbatim.

## Artifacts

`runs/c8r_budget_curves/` (calibration.json, eval log, raw rows), curve-derivation script + outputs under `docs/experiments/analysis/`.
