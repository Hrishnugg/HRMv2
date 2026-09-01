# C8-X: MovingAI at map scale + adaptation transfer to unseen maps (frozen design)

**Status: FROZEN 2026-07-26, before any implementation was executed.**
**Chronology label: post-submission extension** (same policy as C8-S): designed
after the v12 text freeze; intended for rebuttal, camera-ready, or ICAPS.

## Questions

1. **Zero-shot at scale (A):** does the five-map external boundary result
   (procedural success advantage does not reproduce) persist with 20+
   converted source maps and map-clustered inference?
2. **Adaptation transfer to unseen maps (B, primary):** after few-shot
   adaptation on instances from a *pool* of source maps, does planning
   improve on instances from *held-out* source maps the adaptation never
   saw? The submitted paper's within-group study cannot answer this.
3. **External coverage law (B):** at fixed label count, does drawing the
   adaptation instances from more distinct source maps improve held-out-map
   success (C14's contrast, externally)?

## Frozen protocol

- Source maps: download `street-map.zip` and `dao-map.zip` from
  movingai.com. Selection RULE (not list): street = alphabetically first 12
  files ending `_256.map`; dao = alphabetically first 15 files with both
  dimensions <= 512. Realized lists, per-map usable-instance yield, and
  zero-yield attrition are reported; maps with zero usable instances drop.
- Conversion: the frozen pipeline from the submitted study, unchanged
  (64x64 majority coarsening, row-run rectangle decomposition, trained
  maze-family patroller dynamics, identical usability rule).
- Instances per usable map: 6 DEV (seed base 50,000,000) and 12 EVAL
  (seed base 51,000,000), deterministic per (map index, slot); an instance
  manifest with seeds and source-map SHA-256 hashes is written at
  generation.
- **Phase A:** per category, calibrate budgets on pooled DEV instances
  (paper grid + frozen closest-target rule) and tune WA* on DEV (frozen
  rule); evaluate euclid / WA* / frozen blind U-Net zero-shot once on all
  EVAL instances. Readout A1: learned-anchor success per category with
  source-map-clustered bootstrap; per-map effects reported.
- **Phase B:** pool = alphabetically first 8 usable maps per category;
  held-out = all remaining usable maps (if a category has < 4 held-out
  maps it degrades to within-group-only and says so). Adaptation cells:
  K=16 DEV instances total drawn from M in {1,2,4,8} pool maps (16/M per
  map, first instances in recorded order) x {conv-LoRA r8, full FT}
  (exact C14 recipe, 2,560 steps) x seeds {0,1}, per category. Each
  checkpoint is evaluated once on every held-out map's EVAL instances
  (primary) and every pool map's EVAL instances (within-group secondary).

## Preregistered readouts

Primary family (BH within the 8 tests, source-map-clustered bootstrap):
- B1: adapted(M=8) - zero-shot success on held-out maps > 0, per
  category x method (4 tests).
- B2: adapted(M=8) - adapted(M=1) success on held-out maps > 0, per
  category x method (4 tests).

Secondary/descriptive: A1 zero-shot boundary at scale; B3 within-group vs
held-out gap per cell; B4 adapted vs anchor and vs tuned WA* on held-out
maps; per-map forest views. Failure of B1/B2 is reported as a bounded
negative (within-group adaptation that does not transfer strengthens the
narrow-coverage story; it is not suppressed).

## Integrity

Phases (download+convert+manifest, generate, calibrate+tune, eval-zeroshot,
adapt, eval-adapted) are idempotent; adaptation runs on the Modal L4 fleet
with the C14 house pattern (workers never write the manifest). Raw rows,
checkpoints, manifests, and this design ship in the artifact package.
Deviations become dated amendments.

## Amendment 1 (2026-07-26, pre-execution)

Recorded before any phase ran. The original text set 6 DEV instances per
map but drew K=16 adaptation instances from M=1 pool map, which is
arithmetically impossible. Amended: **K=8** labeled instances total
(matching the submitted study's full-rescue operating point), M in
{1,2,4,8} with 8/M instances per pool map; DEV target = 10 instances per
map; a map is *usable* iff >= 8 DEV and >= 8 EVAL instances build within
the per-slot attempt budget (40 attempts per slot). Pool = alphabetically
first 8 usable maps; held-out = remaining usable maps. Budget selection
and WA* tuning follow the submitted study's mechanism exactly: every arm
runs once at BIGB=14,000 and all lower-budget outcomes derive by expansion
thresholding in the frozen analysis (no separate calibrate/tune phases in
the runner). All other commitments unchanged.

## Amendment 2 (2026-07-26, pre-v2-execution; v12-review Part V adopted)

The v1 launch was stopped during instance generation (no evaluation or
adaptation outcome was produced or examined); outputs move to
runs/c8x2_scale. Changes: (1) selection widened to 20 maps per category
(street rule now admits `_256.map` and `_512.map` files with dims <= 512),
so held-out counts support map-level inference. (2) Pool maps' DEV
instances are evaluated in the zeroshot phase; the frozen analysis selects
budgets and WA* weights ONLY from pool-map dev rows; held-out maps receive
a single frozen evaluation of EVAL instances (fixes a calibration-source
hole: v1 produced no dev rows at all). (3) Exact retained-label matching:
per category, N_target = the largest power of two <= the minimum label
supply over all planned cells, computed from per-instance label counts
recorded at generation (outcome-independent) and written into the manifest
before any adaptation; every cell's dataset mask is thinned to exactly
N_target with a cell-seeded RNG. (4) Map-set draw replication: M in {1,2,4}
runs three deterministic pool rotations (draw d uses maps [(d*M+i) mod 8]);
M=8 has its single draw; two optimizer seeds each; methods now lora / full
/ scratch (scratch = identical training loop from random initialization).
120 adaptation cells total. (5) Conversion fidelity per map (free-space
fraction, 4-connected free components, attempt counts) recorded in the
manifest at generation. Whole-map coarsening remains the primary
conversion (continuity with the submitted study); a crop-based sensitivity
is future work.

## Completion (2026-07-27)

Executed in full (120/120 cells; 37 zero-shot map files). Result:
`docs/experiments/continuous/c08/results/C8X_SCALE_TRANSFER_RESULT.md`;
analysis `docs/experiments/analysis/c8x2_analysis.py`. Primary verdict:
category-split (street B1 both methods + full-FT B2 survive BH; dao all
null = registered bounded negative; dao's earlier within-group rescue is
map-specific per B3). A1 descriptive: zero-shot BEATS the anchor on
held-out external maps in both categories at this cohort/calibration
(street +0.097, dao +0.204, map CIs exclude zero) while losing to tuned
WA* (-0.222/-0.148). Two disclosed deviations (dataset-name assert repair
pre-outcome; salted-hash thinning RNG not re-derivable).

## Amendment 3 (2026-07-27, post-outcome design repair; balanced draws)

A structural review of the completed v2 run found the draw schedule
[(d*M+i) mod 8] unbalanced: M=1 draws 0-2 cover only pool maps {0,1,2},
M=2 draws never use maps {6,7}, and the third M=4 draw repeats draw 0's
map set (differing only in the label subsample). B2 (M=8 vs M=1) therefore
compared full coverage against three particular one-map sets. This
amendment repairs the schedule by EXTENSION, not replacement:

1. DRAWS becomes {1: [0..7], 2: [0,1,2,3], 4: [0,1,2], 8: [0]} - the same
   index formula now yields all eight one-map sets at M=1 and the four
   disjoint pairs at M=2; M=4 keeps its two disjoint halves (draws 0,1)
   plus legacy draw 2 (a subsample replicate of draw 0, excluded from
   balanced summaries and reported separately).
2. New cells (M=1 draws 3-7; M=2 draw 3; x 3 methods x 2 seeds x 2
   categories = 72 adaptations) run under the identical frozen recipe,
   N_target, binding budgets, and held-out evaluation. Existing cells are
   reused untouched (idempotent skip).
3. Thinning-RNG repair: new cells seed the exact-N mask thinning from a
   stable SHA-256 digest of the cell tag (replacing Python's salted
   hash()). Existing cells keep their already-materialized datasets, whose
   SHA-256 hashes are recorded in a manifest (Amendment scope: subsets are
   archived even where not re-derivable).
4. Repaired B2 readout: adapted(M=8) minus the MEAN OVER ALL EIGHT M=1
   draws per held-out map (seeds and draws averaged within map), same four
   tests, BH within the same 8-test family alongside the unchanged B1.
   Chronology honestly labeled: this is a post-outcome repair motivated by
   the structural imbalance finding, not by the observed effect direction;
   the repaired family does not inherit preregistered status and is
   reported as the corrected primary analysis, with the original 3-draw B2
   retained alongside for continuity.
5. Balanced dose curves use M=1 (8 draws), M=2 (4 draws), M=4 (draws 0-1),
   M=8 (1 draw).
6. Operational note (2026-07-28): heavy Modal container preemption (48
   preemptions in ~3 h) made whole-cell held-out evaluation (~25 min per
   cell) Sisyphean. Repair-cell evaluation is resharded per (cell,
   held-out map) into 1-3 minute units, merged afterward into the standard
   ad_{tag}.csv. Repair cells are evaluated on HELD-OUT maps only: no
   registered or descriptive readout consumes repair-cell pool-map rows
   (B3 uses the original M=8 cells, which retain full evaluations).
   Evaluation semantics per row are unchanged.
