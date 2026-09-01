# C8 weighted-A* extended-grid check

**Frozen design date:** 2026-07-26 (before execution)
**Chronology, disclosed:** this check was specified after external review
observed that three suites' success-tuned weights (maze, dense maze, spiral)
selected the original grid's maximum, $w_h{=}5$, leaving open whether larger
inflation would change the control.

## Protocol

- Extend the candidate grid to $w_h \in \{1.1, 1.2, 1.5, 2, 3, 5, 7, 10\}$.
- Re-run the development cohort (seed 1234, 20 maps/suite) for the two new
  weights only; merge with the frozen development rows for the original six.
- Apply the identical frozen selection rule (highest development success,
  ties toward smaller $w_h$).
- For any suite whose selected weight changes, evaluate the new weight once
  on the frozen 50-map confirmation cohort; suites whose selection is
  unchanged keep their frozen confirmation rows untouched.

## Readouts (report as-is)

- R1: per suite, whether the extended grid changes the selected weight.
- R2: for changed suites, confirmation success/effort/path quality at the
  new weight versus the frozen $w_h{=}5$ rows and versus the learned arm.
- If no selection changes, the check is reported as a boundary-robustness
  null (the original tuning was not grid-limited).

## Exclusions

No effort-based or lexicographic re-tuning (the control remains
success-tuned, as named in the paper); no changes to frozen artifacts.
