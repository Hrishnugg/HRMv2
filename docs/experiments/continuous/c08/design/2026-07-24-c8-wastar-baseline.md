# C8 weighted-A* baseline design (descriptive; reviewer-requested)

**Frozen:** 2026-07-24, before any confirmation-cohort evaluation of this baseline.
**Motivation:** the v2 deep review's strongest baseline objection — the 73–95% effort reduction could partly reflect a weak Euclidean anchor. Weighted A\* is the standard classical way to trade path quality for search effort with no learning; a tuned weighted A\* on identical worlds and budgets is the fair classical comparator.

## Protocol

- **Arm:** space-time A\* with inflated anchor $h_w(v,t) = w_h \cdot h_0(v,t)$, where $h_0$ is the canonical travel-time lower bound. No learned components; identical search code, budgets, and collision model as every other arm.
- **Tuning (development only):** weight grid $w_h \in \{1.1, 1.2, 1.5, 2.0, 3.0, 5.0\}$, evaluated on the development cohort (canonical seed-1234 stream, 20 worlds/suite) at the canonical binding budgets. Per-suite selection rule, frozen now: highest success; ties broken toward the smallest $w_h$ (better path quality). This grants the baseline per-suite tuning — the most favorable variant for the baseline.
- **Confirmation:** the selected per-suite $w_h$ is evaluated once on the frozen 50-map confirmation cohort (seed-999999 stream) at binding budgets. No re-tuning, no reselection.
- **Readouts (report-as-is):** per-suite confirmation success and matched-solved expansion ratio vs the plain anchor; paired comparison vs the fixed blind U-Net on the same worlds (success difference; jointly solved effort); empirical path-cost (arrival) ratios vs optimal arrival joined from the existing confirmation raws.
- **Interpretation rule:** whatever the outcome, it is reported. If tuned weighted A\* matches the learned arm, the paper's claim narrows accordingly.

## Artifacts

`continuous_prm_c8_wastar_baseline.py` (standalone; read-only reuse of the frozen harness), `runs/c8r_wastar/` (development + confirmation raws), summary in `docs/experiments/analysis/c8_wastar_output.md`.
