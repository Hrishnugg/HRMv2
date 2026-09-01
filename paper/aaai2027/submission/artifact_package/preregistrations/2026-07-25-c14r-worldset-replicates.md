# C14-R: independent world-set replicates for the coverage factorial

**Frozen design date:** 2026-07-25 (before any implementation)
**Purpose:** answer the sharpest external critique of C14: each factorial
cell used one deterministically sampled world set, so the coverage effect
could in principle be specific to the particular worlds drawn (the dynamic
concentrated cells at N <= 16,384 share a single world). C14-R re-draws the
world sets independently and asks whether the collapse/rescue pattern
reproduces.

## 1. Factors (tight scope, chosen before results)

- **Domains/cells.** Dynamic: N in {1,024, 16,384} x {concentrated,
  distributed} (the two decisive collapse levels). Static: N = 256 x
  {concentrated, distributed} (the only static collapse level).
- **World-set draws.** Two NEW independent draws per cell (draw streams
  seeded 20260726 and 20260727), collected by the same w_min rule from fresh
  candidate streams disjoint from the original C14 stream. The original C14
  cells serve as draw 1.
- **Methods.** full fine-tuning, unbounded rank-8 LoRA, scratch -- identical
  recipes, 2,560 matched optimizer steps, one optimization seed (0) per arm
  (the original already showed within-cell seed stability; the manipulated
  and replicated level here is the world set).
- **Arms.** (2 dynamic N x 2 coverage + 1 static N x 2 coverage) x 3 methods
  x 2 draws = 36 adaptations. Evaluation on the SAME frozen C9/C9b test
  cohorts at binding budgets (no new test maps).

## 2. Preregistered readouts (report as-is)

- R1 (collapse reproduction): dynamic concentrated N=1,024 full-FT and
  scratch success deltas vs the frozen source, per draw. Original: -0.300 in
  every seed. Reproduction = paired map CI excluding zero in the harmful
  direction in both new draws.
- R2 (rescue reproduction): dynamic distributed N=1,024 full-FT delta per
  draw. Original: +0.150 to +0.200. Reproduction = no harmful CI, point
  delta >= 0.
- R3 (static): static concentrated N=256 full-FT/scratch harmful vs
  distributed ~0, per draw.
- R4 (LoRA): no significant LoRA collapse in any new cell.
- Primary presentation: per-draw success deltas with map-level CIs, plus the
  direct distributed-minus-concentrated paired contrast per draw. All
  outcomes reportable; no draw may be discarded.

## 3. Compute

Modal app pattern from C14 (same volume, L4 fleet, idempotent phases,
arm-level fan-out). Static N=256 collection is CPU-local (small) and
uploaded; dynamic collection runs remotely. ~36 arms, minutes each.

## 4. Exclusions

No new N levels, targets, methods, or ranks; no reruns of original C14
cells; no changes to frozen cohorts or budgets; the original C14 verdict
text is not edited in response to C14-R (C14-R is reported alongside it).
