# C8 weighted-A* extended-grid check — result

> **ERRATUM (2026-07-27): the numbers below are VOIDED.** The extgrid runner
> used an alphabetical suite list, so its suite-index-seeded worlds were a
> *sibling* cohort, not the frozen development/confirmation cohorts (exposed
> by a cross-run optimal-arrival fingerprint check: 5/185 matches between the
> SIPP raw, which shared the defect, and the canonical rows). The runner was
> corrected to the canonical order and re-executed under the unchanged frozen
> rules; corrected numbers are appended below. Voided raws:
> `runs/c8r_wastar/voided_wrong_cohort/`.

**Design:** `../design/2026-07-26-c8-wastar-extended-grid.md` (frozen before
execution; chronology disclosed: specified after external review observed
three suites at the original grid maximum $w_h{=}5$).
**Run:** `hrm-cloud/continuous_prm/continuous_prm_c8_wastar_extgrid.py` →
`runs/c8r_wastar/{development_ext_raw.csv, confirmation_ext_raw.csv,
extgrid_report.json}`.
**Analysis:** `docs/experiments/analysis/c8_wastar_extgrid_analysis.py` →
`c8_wastar_extgrid.json`.

## R1 — selection changes under the extended grid {7, 10}

| Suite | frozen $w_h$ | extended $w_h$ | dev success |
|---|---|---|---|
| crossing | 1.5 | 1.5 | 1.00 |
| maze | 5 | 5 | 1.00 |
| **dense maze** | **5** | **7** | **1.00** (vs 0.90 at $w=5$) |
| rooms | 3 | 3 | 1.00 |
| large rooms | 1.5 | 1.5 | 1.00 |
| spiral | 5 | 5 | 0.80 (no larger weight beats it) |

Dense-maze development curve is monotone across the merged grid:
1/2/2/4/12/18/20/20 successes of 20 at $w_h$ = 1.1/1.2/1.5/2/3/5/7/10
(ties-toward-smaller picks 7 over 10).

## R2 — dense maze at $w_h{=}7$ on the frozen confirmation cohort (n=50)

- Success: **45/50** vs 35/50 at frozen $w_h{=}5$ (15/5 discordant, exact
  $p{=}0.041$) and vs **35/50 for the learned blind U-Net** (also 15/5,
  $p{=}0.041$; single post-hoc comparisons, unadjusted).
- Matched-solved learned/w7 expansions: median 1.172 [0.948, 1.479]
  (n=30) — parity.
- Jointly solved WA* mean expansions: 843 at $w{=}7$ vs 1,217 at $w{=}5$.

## Verdict

Five of six suites are boundary-robust nulls (the original tuning was not
grid-limited there); spiral's learned advantage (15/0 discordant) is
grid-robust. Dense maze is grid-limited: under stronger inflation the
success-tuned control overtakes the learned arm on this suite. Paper
framing (v7): the learned advantage concentrates where no tested inflation
closes the gap; the dense-maze effort comparison in the frozen table is
grid-limited and disclosed as such in main + appendix.

## CORRECTED RESULT (canonical cohort, 2026-07-27)

Selections: five suites unchanged; dense maze re-selects w=7 on the canonical
dev cohort (19/20 vs 18/20; merged curve 1/2/2/4/12/18/19/19 of 20).
Confirmation (canonical 50-map cohort): w=7 solves 37/50 vs 35/50 for both
frozen w=5 and the learned arm (discordants 2/0 and 3/1, exact p=0.5/0.625);
learned/w7 matched expansions median 1.399 [0.911, 1.665] (n=34); joint-set
path quality w7 1.015 vs blind 1.008, diff -0.007 [-0.022, +0.004].

**Verdict: boundary-robustness NULL on all six suites** — the original
success-tuned control was not grid-limited anywhere; the voided
sibling-cohort "dense maze reversal" (45/50) does not reproduce. Spiral's
learned advantage is grid-robust.

