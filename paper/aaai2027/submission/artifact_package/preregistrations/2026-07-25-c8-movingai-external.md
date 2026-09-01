# C8 external benchmark: dynamic zero-shot on MovingAI-derived worlds

**Frozen design date:** 2026-07-25 (before any implementation)
**Purpose:** test the fixed dynamic heuristic on externally authored map
geometry (the Sturtevant/MovingAI benchmark set used by Neural A* and much of
the learned-heuristic literature), answering the "all environments are
self-generated" objection with a zero-shot evaluation on maps whose structure
we did not design.

## 1. Substrate adaptation

- **Maps.** Two MovingAI groups: (a) city maps (street grids), (b) dungeon/
  game maps (rooms + corridors), downloaded from movingai.com (plain-text
  .map files, kilobyte scale). Exact files recorded in the run manifest.
- **Conversion.** Each map is coarsened to the canonical 64x64 occupancy
  resolution over a unit square, then blocked cells are decomposed into
  axis-aligned rectangle obstacles (maximal horizontal runs merged
  vertically). The result is a standard World(obstacles, side_len=1.0) --
  every downstream component (PRM builder, collision checks, feature raster,
  space-time planner) runs unchanged. No retraining, no recalibration of the
  model.
- **Dynamics.** The canonical maze-family patroller configuration (3
  patrollers, radius 0.075, span 0.38, period 0.14, lateral 0.02, v_agent
  0.060, dt 1.0, t_max 110) is applied on top of the external geometry,
  identical to the trained dynamic regime. Patroller sweeps are sampled by
  the same generator rules.
- **Instances.** Start/goal sampled in free space with the canonical 0.45
  minimum separation; a world is usable under the same three usability rules
  (builds; roadmap connects; space-time solvable at t=0). Per group: draw
  candidate (map-window, seed) instances until 10 usable development + 25
  usable evaluation instances exist, development first, disjoint seeds.

## 2. Protocol

- **Budget calibration.** Anchor-only, the canonical rule (targets 0.45/0.70
  on the development instances over the canonical grid; smaller selected
  budget binding), frozen before any learned evaluation.
- **Arms.** Euclidean anchor A*; tuned weighted A* (grid {1.1,1.2,1.5,2,3,5}
  tuned on the development instances, frozen rule); the fixed blind U-Net
  (checkpoint SHA b8378950..., additive, untouched).
- **Readouts (report as-is).** Per group: paired success deltas
  (learned - anchor; learned - WA*) with map-level bootstrap CIs and exact
  McNemar; matched-solved expansion ratios on jointly solved instances; mean
  suboptimality on jointly solved instances.

## 3. Verdict rules

- R1 (transfer): learned success >= anchor success per group, paired CI
  reported; any negative delta is reported as a transfer failure on external
  geometry.
- R2 (honesty): the external worlds differ from training in geometry
  provenance only within the same raster/scale regime; this is transfer of
  the fixed model to externally authored structure, not domain transfer to a
  different task, and the paper says so.

## 4. Exclusions

No fine-tuning; no per-map model selection; no map-specific patroller tuning;
one canonical dynamics configuration; CPU-scale evaluation run locally in the
background (model inference on 64x64 rasters is CPU-cheap).
