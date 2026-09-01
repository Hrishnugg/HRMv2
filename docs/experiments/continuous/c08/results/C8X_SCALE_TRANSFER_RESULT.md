# C8-X v2 result: MovingAI at map scale + adaptation transfer to unseen maps

Status: COMPLETE (2026-07-27). Frozen design
`docs/experiments/continuous/c08/design/2026-07-26-c8-movingai-scale-transfer.md`
(+ Amendment 1: K=8 arithmetic; + Amendment 2: 20+20 maps, pool-dev
calibration rows, exact-N, draws, scratch). Executed on Modal
(app continuous-prm-c8x, volume runs/c8x2_scale). Analysis:
`docs/experiments/analysis/c8x2_analysis.py` (+ `.json`, `_output.md`).
Post-submission evidence (rebuttal/camera-ready/ICAPS); NOT part of the
2026-07-28 submission.

## Realized protocol

- 40 maps selected by the frozen rule (20 street + 20 dao); 37 usable
  (street 20/20, dao 17/20 - brc100d, brc300d, brc502d fail instance
  generation); 16 pool maps supply labels and calibration; the
  confirmatory held-out inference uses the remaining 21 maps (12 street +
  9 dao). All results are for this SINGLE frozen source-map split
  (first-8-usable pool); no split resampling.
- Calibration from POOL-map dev rows only (canonical closest-target rule):
  binding street 400, dao 400; tuned w_h = 3 for both groups.
  (The submitted 5-map study realized 600/900 and w_h 1.5/2: different
  cohort, different operating point; comparisons across the two studies are
  qualitative only.)
- Adaptation: K_TOTAL=8 dev instances per cell from M in {1,2,4,8} pool
  maps, 3 draws for M<8, methods {conv-LoRA r8, full FT, scratch}, 2
  optimizer seeds, exact-N label thinning to N_target=65,536 per category
  (recorded in the manifest before adaptation). 120 cells, all complete.
- Evaluation: every adapted cell evaluated once on all usable maps' frozen
  eval instances (12/map) at BIGB=14,000 with success-at-binding by
  expansion thresholding.

## Preregistered primary family (8 tests, source-map-clustered bootstrap, BH)

Held-out maps only; seeds (and M=1 draws) averaged within map.
Test machinery: p-values are two-sided add-one bootstrap probabilities
from the 10,000-resample source-map-clustered bootstrap; BH within the
declared 8-test family. Amendment 3: the pre-amendment M=1 draws covered only pool maps 0-2
(and the third M=4 draw repeated draw 0's map set), so the frozen B2 rows
below compare M=8 against three particular one-map sets. The
balanced-draw repair (all eight M=1 sets, 72 added cells, stable digest
thinning seeds) is the corrected primary; see the repaired-family table
below.

| Test | delta | 95% CI | p | q (BH/8) |
|---|---:|---|---:|---:|
| B1 street lora (a8 - zs) | +0.076 | [+0.017,+0.132] | .0130 | **.0347** |
| B1 street full | +0.125 | [+0.059,+0.191] | .0002 | **.0016** |
| B1 dao lora | -0.019 | [-0.116,+0.093] | .6703 | .8938 |
| B1 dao full | -0.000 | [-0.116,+0.116] | .9891 | .9891 |
| B2 street lora (a8 - a1) | +0.046 | [-0.012,+0.102] | .1202 | .2404 |
| B2 street full | +0.037 | [+0.012,+0.060] | .0032 | **.0128** |
| B2 dao lora | -0.008 | [-0.040,+0.026] | .6145 | .8938 |
| B2 dao full | +0.003 | [-0.032,+0.046] | .8859 | .9891 |

**Verdict: category-split.** On street, few-shot adaptation from 8 pool maps
transfers to 12 unseen maps (both methods' B1 survive BH; full FT also
shows the M=8-over-M=1 coverage gain, B2 q=.013). On dao, adaptation does
not move held-out success at all (zs 0.694 -> a8 0.676-0.694; all four
tests null). Failure of B1/B2 on dao is reported as the registered bounded
negative.

## Repaired primary family (Amendment 3: balanced M=1, all eight draws)

Post-outcome design repair motivated by the structural imbalance finding
(not by effect direction); does not inherit preregistered status.
B1 unchanged; B2-balanced = M=8 minus the mean over ALL EIGHT M=1 draws
per held-out map; BH within the same 8-test structure:

| Test | delta | 95% CI | q (BH/8) |
|---|---:|---|---:|
| B1 street lora | +0.076 | [+0.017,+0.132] | **.035** |
| B1 street full | +0.125 | [+0.059,+0.191] | **.0016** |
| B1 dao lora / full | -0.019 / -0.000 | | .857 / .989 |
| B2bal street lora | +0.023 | [-0.027,+0.076] | .738 |
| B2bal street full | +0.037 | [+0.011,+0.063] | **.026** |
| B2bal dao lora / full | +0.005 / -0.008 | | .857 |

**The street full-FT coverage effect SURVIVES the balanced repair** at the
same magnitude as the frozen three-draw version (+0.037, q=.026): with
all eight one-map adaptation sets, M=8 coverage still beats M=1 for full
fine-tuning on held-out street maps. Street LoRA's B2 remains null, dao
remains null throughout. Balanced dose (street, 8/4/2/1 draws):
lora .768/.780/.821/.792, full .803/.788/.800/.840, scratch
.849/.838/.859/.868 (endpoint contrast, not a monotone dose response).

## A1: zero-shot boundary at scale (secondary/descriptive, with map CIs)

On held-out maps at the frozen operating point, the FROZEN zero-shot
heuristic now BEATS the anchor in both categories: street +0.097
[+0.042,+0.160] (0.715 vs 0.618), dao +0.204 [+0.093,+0.324] (0.694 vs
0.491). It still clearly loses to tuned weighted A*: street -0.222
[-0.285,-0.174] (WA* 0.938), dao -0.148 [-0.278,-0.019] (WA* 0.843).
Reading: the submitted 5-map anchor-relative negative does not generalize
to the 21-held-out-map cohort at this calibration; the honest external
boundary is versus tuned WA*, which dominates success on external maps in
both categories. (Different binding/cohort from the submitted study; this
does not overwrite that study's frozen verdict on its own cohort.)

## Held-out effort, path quality, and conversion fidelity (descriptive)

Matched-effort ratios on held-out maps (map-level medians, bootstrap
CIs): the zero-shot and adapted arms expand FEWER nodes than the anchor -
street zeroshot 0.449 [0.379, 0.618], a8 full 0.441 [0.398, 0.553]; dao
zeroshot 0.580 [0.531, 1.075], a8 full 0.397 [0.281, 0.463] - but MORE
than tuned WA* (street 1.8-2.3x, dao 1.7-2.5x): WA* wins external effort
as well as success. Path quality on jointly solved instances: learned
arms pay 1.03-1.11 mean suboptimality versus the anchor's optimal 1.000,
and are comparable to WA* (1.02-1.10 vs 1.04-1.05). Conversion fidelity:
street pool and held-out maps are tightly matched (mean free-space 0.750
vs 0.755); dao maps are structurally heterogeneous (free-space 0.128-
0.964; pool mean components 3.5 vs held-out 1.8), context for the dao
transfer null and the three failed dao conversions.

## Descriptive findings

- **B3 (within-pool vs held-out) localizes the dao null:** on dao POOL maps
  (whose dev instances supplied the labels) a8 reaches 0.656-0.664 vs pool
  zs 0.564 (~+0.10), while held-out dao is +/-0.00. The submitted
  within-group dao "full rescue" therefore reflects map-specific
  adaptation: it does not transfer across dao maps. Street shows no such
  gap (pool 0.807-0.828 vs held-out 0.792-0.840).
- **Scratch (descriptive; outside the frozen family):** street scratch is
  the strongest arm (a8 0.868 vs full 0.840, lora 0.792; B1-style +0.153
  [+0.097,+0.215]); dao scratch is at zero-shot parity (0.699). With
  ~65k exact labels per cell, external street maps are label-dense enough
  that transfer holds no advantage over from-scratch training - consistent
  with the program's density story (dynamic C9b K=1). Transfer-beats-
  scratch does NOT reproduce on this substrate at this label budget.
- **Dose curves (held-out mean success by M):** street lora
  .745/.803/.834/.792, full .803/.818/.807/.840, scratch
  .832/.853/.869/.868; dao flat throughout (.645-.745). No street M=1
  concentration harm at K=8/N=65k (unlike the submitted K=1 street
  degradation at 2 instances - different label scale).
- **B4:** adapted arms still lose to tuned WA* on held-out success
  everywhere (street full -0.097 [-0.128,-0.063]; dao full -0.148
  [-0.227,-0.074]) while beating the anchor (+0.174 to +0.222 street,
  +0.185/+0.204 dao).

## Integrity notes and deviations

1. **Mid-run defect, repaired before any adaptation outcome existed:** the
   first v2 launch crashed at the first adapt cell on a dataset-name
   assert (`FS.build_dataset` appends `_K{n}` to the stem). Fix =
   rename-after-build; relaunch resumed idempotently (select/generate/
   zeroshot artifacts reused; the crashed attempt produced no trained cell
   and no thinned dataset). One orphan `*_K8.npz` remains on the volume.
2. **Thinning-RNG reproducibility deviation:** the exact-N mask thinning
   seeds its RNG with `abs(hash(tag))`; Python string hashing is
   process-salted, so the exact retained label subsets are not
   re-derivable from code + manifest (the retained COUNT, 65,536, is exact
   and enforced; every cell was thinned by the same mechanism; no
   comparison conditions on subset identity). Future runs will derive the
   seed from a stable digest. Disclosed rather than repaired post hoc.
3. Primary-family scope: Amendment 2 added scratch without amending the
   frozen 8-test family; scratch is therefore reported descriptively.
4. All 192 cells (120 original + 72 Amendment-3) and 37 zero-shot map
   files completed; row-count and role/phase sanity asserts pass in the
   analysis script.
5. **Preemption + partial-file incident (2026-07-28, repaired):** heavy
   Modal container preemption during repair-cell evaluation (48
   preemptions in ~3 h) collapsed whole-cell throughput; evaluation was
   resharded per (cell, held-out map) with a merge step (Amendment 3
   operational note; repair cells evaluate held-out maps only - no
   readout consumes their pool rows). Five cells killed mid-write in the
   whole-cell attempts left PARTIAL csv files that the resume logic
   counted as complete; a per-cell completeness scan (every held-out map
   present) caught them, the partials were deleted, and the cells
   re-evaluated. The completeness scan is now part of the analysis run.
6. Amendment-3 cells use stable SHA-256-digest thinning seeds; a SHA-256
   manifest of every cell dataset and checkpoint
   (datasets_ckpts_sha256.json) ships on the volume and in the artifact
   package, closing the salted-hash reproducibility gap for the
   pre-amendment cells by archiving their exact materialized subsets.

## Pointers

Raw rows: volume `runs/c8x2_scale` (local mirror `runs/c8x2_scale/` via
`rows_bundle.tar.gz`). Core: `continuous_prm_c8_movingai_scale.py`;
driver: `continuous_prm_c8_movingai_scale_modal.py`;
bundler: `c8x2_bundle_rows.py`.
