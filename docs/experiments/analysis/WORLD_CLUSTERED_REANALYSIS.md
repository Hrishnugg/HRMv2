# World-Clustered Reanalysis: Findings and Implications

**Date:** 2026-07-20
**Script:** [`world_clustered_reanalysis.py`](world_clustered_reanalysis.py) (deterministic; seed 20260720; 10k world bootstraps, 20k stratified permutations)
**Raw output:** [`world_clustered_reanalysis_output.md`](world_clustered_reanalysis_output.md)
**Addresses:** master-synthesis high-priority issues #1 and #5 (repeated-world pooling in C9/C9h/C9b/C10/C11) for the key paper-quoted quantities. The full-grid refresh of every published table remains camera-ready work.

## Method

Units are TEST worlds; adaptation/model seeds are averaged within worlds before any inference. Success deltas use world-level percentile bootstraps; expansion ratios use median-of-per-world-ratios over matched worlds (Euclid solved and the learned arm solved at least one seed episode) at the binding budget in additive (`astar`) mode; the C11 halting test aggregates model seeds within worlds and permutes mission length within config (worlds are independent across K cells — verified: distinct world seeds per cell). Estimator conventions differ slightly from the published record-level curves (median of per-record ratios), so point values differ modestly; verdict-level agreement is the question.

A first pass of this script accidentally pooled additive and focal evaluation modes, which badly attenuated the C9 effects; the committed version filters `mode=="astar"`. This is worth recording: **mode/aggregation mixing is itself sufficient to halve an apparent transfer effect.**

## Findings

### C11 learned halting (paper headline number) — survives, strengthens

| Grain | n | Spearman rho | p |
| --- | --- | --- | --- |
| Record level (as published) | 675 | −0.4066 | ≈5×10⁻⁴ (published permutation) |
| World level (seeds averaged) | 225 | **−0.5778** | **5×10⁻⁵** (stratified permutation) |

Seed noise was diluting the inversion; the anti-correlation between learned halting and mission depth is stronger at the correct grain. World-level mean would-halt steps 6.78/7.01/5.30 at K=2/4/8 (matches published 6.79/7.01/5.30).

### C11 shallow-K global-input advantage — real but config-dependent

Ratio-of-means vs MLP with world-clustered CIs: excludes parity at K∈{0,2} in config A (U-Net 0.734/0.846; GNN 0.822/0.923) and config B at K=0 (0.904/0.818) plus B/K2 for GNN (0.805); config C never separates (0.98–0.99 at K=2); every cell at K∈{4,8} includes parity. The published "shallow-K advantage that dissolves with depth" holds, with the refinement that it is carried by configs A and B, not C.

### C9 transfer regimes — orderings confirmed with CIs; story intact

Maze-dense HRM (binding budget 140, additive): LoRA K1 ratio 0.686 [0.590, 0.774] at Δsuccess +0.467 [+0.300, +0.633]; full-FT K1 0.789 [0.751, 0.836] → K16 **0.610 [0.549, 0.670]**, beating LoRA K16 0.786 [0.694, 0.891] with non-overlapping CIs — the low-K-preservation → high-K-capacity crossover is world-clustered-significant. Scratch K1 is significantly *worse than Euclid* on success (−0.167 [−0.300, −0.047]). Rooms-large full-FT K1 is genuinely catastrophic (ratio 1.293 [1.170, 1.503]) and recovers to 0.590 [0.498, 0.712] by K8. Bugtrap LoRA K32 degradation is significant (Δsuccess −0.320 [−0.507, −0.133]; ratio 1.153 [1.012, 1.523]) — "LoRA is usually but not universally base-preserving" is now CI-backed in both directions. Published record-level point values (0.650/0.571/0.500) differ only by estimator convention.

### C9h bound-vs-capacity — reproduced exactly

27 cells: median bounded-minus-unbounded world-level ratio delta **+0.0000**, 15/27 exactly zero, sd 0.0077, max |Δ| 0.0370. The published 0.000±0.008 attribution (capacity, not clamp) stands at the corrected grain.

### C9b aware-vs-blind at full-FT K16 — null holds; phrasing must soften

World-clustered: 8/9 point deltas are zero or negative; the one positive (rooms-large field U-Net, +0.017 [0.000, +0.050]) does not exclude zero; no cell shows a significant aware advantage in either direction. The published "0/9 positive" used the probe's different aggregation; the evidence-safe wording is now "no cell shows a significant aware advantage; point deltas −0.050 to +0.017 with all CIs crossing zero."

### C10 interpolation — null sharpens into a partial negative

World-paired at budget 150: four cells are ≈0 with CIs crossing zero; the two ON-LSTM rooms cells are **significantly worse** than zero-shot (RBF-minus-zero-shot +0.089 [+0.061, +0.121] and +0.098 [+0.076, +0.123]), matching the published +0.082/+0.116 point deltas and giving them uncertainty. "No consistent improvement" upgrades to "never better, and significantly worse in two of six cells."

## Implications

1. **No preregistered verdict flips.** Every conclusion quoted in the AAAI draft survives; two strengthen (halting inversion, C10 negative), one gains a refinement (shallow-K config dependence), one needs softer wording (C9b 0/9 → no-significant-advantage).
2. Paper and synthesis updated accordingly (Limitations no longer says the reanalysis is pending; it reports the outcome and scopes the remaining full-grid refresh).
3. Remaining for camera-ready: full-grid world-clustered refresh of every published C9/C9h/C9b/C10/C11 table (this document covers the paper-quoted cells), plus the unchanged replication items (multi-seed C7/C8).
