# C8 MovingAI diagnosis and rescue: residual probe + few-shot adaptation

**Frozen design date:** 2026-07-27 (before execution).
**Chronology, disclosed:** specified after the frozen MovingAI zero-shot
evaluation returned its negative (learned never beats anchor; loses to tuned
WA\* on dungeon instances) and after external review of v6/v7. The probe
tests the proposed mechanism (out-of-distribution residual miscalibration);
the rescue tests the C9/C14-predicted repair (few-shot adaptation).

## Part 1 — residual probe (descriptive diagnostic)

For the frozen blind U-Net, compare predicted vs true normalized residuals
over reachable $(v,t)$ states:

- $\hat\delta = (H_{\text{learned}} - H_{\text{euclid}})/T$ from the frozen
  provider's h-table; $\delta^* = \mathrm{clip}(\mathrm{clip}(ttg -
  h_0, 0, \infty)/T, 0, 4)$ from backward space--time Dijkstra (the exact
  training target).
- **OOD set:** the 50 frozen MovingAI evaluation instances (25 street, 25
  dao), reconstructed exactly from the recorded instance seeds in
  `runs/c8r_movingai/raw.csv`.
- **In-distribution reference:** the first 8 worlds per procedural suite of
  the frozen 50-map confirmation cohort (seed 999999; 48 maps).
- Per map: Pearson $r$, Spearman $\rho$, MAE, and bias (mean
  $\hat\delta-\delta^*$), over reachable slots. Report per-suite/group
  medians and IQRs.
- **Registered directional expectations (descriptive, no test):** MovingAI
  correlations below every procedural suite; dao at or below street;
  positive OOD bias (over-prediction), consistent with the additive
  residual's inflate-only failure mode.

## Part 2 — few-shot rescue

**Adaptation pool:** the 10 recorded *development* instances per group (the
cohort already used for calibration and WA\* tuning; evaluation instances
remain untouched by any training). $K \in \{1, 2, 4, 8\}$ = the first $K$
development instances in recorded order.

**Arms:** conv-LoRA rank 8 ($\alpha{=}1$) and full fine-tuning, both from
the frozen blind U-Net checkpoint; 2 adaptation seeds (0, 1) varying
initialization and batch order only. Scratch is omitted (scope bound; the
transfer-vs-scratch question on dynamic substrates is characterized in C9b
and C14, and the question here is rescue of the frozen prior).

**Training recipe (exact C14 dynamic recipe, no tuning):** supervision =
all reachable states of the $K$ instances (field schema mirroring the
canonical trainer); exactly 2{,}560 optimizer steps at batch 8, AdamW
lr $2{\times}10^{-4}$, weight decay $10^{-4}$, gradient clip 1.0,
smooth-L1 on the capped normalized residual. Local GPU (disclosed; the
frozen zero-shot rows were CPU and are reused, not recomputed).

**Evaluation:** each arm evaluated once on the frozen 25 evaluation
instances per group at BIGB $=$ 14{,}000 with success derived by expansion
thresholding at the frozen binding budgets (street 600, dao 900). All arms
run through the canonical bounded provider (residual clip 4.0). No
recalibration, no WA\* re-tuning.

**Readouts (report as-is):**
- R1 (primary): paired success delta, adapted $-$ frozen zero-shot, per
  (group, $K$, method), seeds averaged within instances, 10k
  instance-bootstrap 95\% CIs. Declared primary cells: $K{=}8$ full FT per
  group (others descriptive).
- R2: adapted $-$ anchor success delta (does adaptation recover the anchor
  gap?).
- R3: adapted $-$ tuned WA\* success delta against the frozen WA\* rows
  (street $w_h{=}1.5$, dao $w_h{=}2$).
- R4: matched-solved expansion ratios (adapted/anchor and adapted/WA\*).
- R5: path-cost ratios on jointly solved instances.
- R6 (mechanism closure): the Part-1 probe recomputed for the $K{=}8$
  full-FT seed-0 model per group.

**Caveats, pre-stated:** instances nest in $\leq$3 base maps per group;
inference is instance-level and unadjusted; the adaptation pool doubles as
the calibration cohort (development data; disclosed); $n{=}25$ evaluation
instances per group; two adaptation seeds.

## Exclusions

No changes to frozen artifacts; no new instance generation; no budget or
weight re-tuning; the zero-shot, anchor, and WA\* comparison rows are the
frozen `raw.csv` rows.
