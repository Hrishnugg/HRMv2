# C14 result: label-count × world-diversity factorial — H-C14 rejected as stated; diversity, not state count, is the governing variable

**Completed:** 2026-07-23 (Modal L4 fleet; collected per amendment v2)
**Design:** [2026-07-23-c14-label-density-factorial.md](../design/2026-07-23-c14-label-density-factorial.md) (frozen 2026-07-23; §7 amendment adopted pre-execution)
**Raw:** `runs/c14_modal/results/continuous_prm_c14_eval_raw.csv` (14,880 rows — exact expected count); analysis [c14_analysis_output.md](../../../analysis/c14_analysis_output.md) via [c14_analysis.py](../../../analysis/c14_analysis.py)

## Execution

180/180 arms trained (5 N × 2 diversity × 2 domains × 3 methods × 3 seeds), all at the matched 2,560 optimizer steps; static cell datasets collected locally and reused byte-identical on Modal (shared sampled index sets preserved); dynamic pool collected remotely from the deterministic seed streams (32 worlds, ~18.9k states/world; w_min(65536)=4). Evals on the frozen C9 static 30-map and C9b dynamic 20-map test cohorts at canonical binding budgets. No cell was rerun, resampled, or dropped after unblinding.

Measured world counts per cell (the amendment's w_min bookkeeping): static conc 2/6/22/89/356 worlds and dist 16/48/176/712/2848 for N=256..65536; dynamic conc 1/1/1/1/4 and dist 8/8/8/8/32.

## Preregistered verdict: H-C14 **rejected as stated**

- **Analysis 2 (crossover):** N\* = 256 — the smallest grid point — in **all 12** (domain × diversity × seed) cells: on matched-solved median ratios, full fine-tuning is at or below LoRA everywhere on the tested grid, in both domains. The crossover the hypothesis sought to place is below N=256 if it exists at all. (Static bootstrap upper bounds reach log₂N ≈ 9.1–13.4; dynamic CIs are degenerate — see limitation below.)
- **Analysis 3 (regression):** the required full_ft × log₂N interaction is null (−0.0025 [−0.0087, +0.0028]); full_ft × dynamic is −0.0409 [−0.0585, +0.0000] (boundary-compatible with zero). Per the frozen verdict rule, the mechanism claim's interaction requirement fails → **rejected** (not "partially supported": (a)'s support is vacuous when N\* sits on the grid boundary everywhere).
- log₂N main effect −0.0140 [−0.0190, −0.0081]: more states do improve ratios, but identically for both methods.

## The preregistered analysis-4 finding: world diversity at fixed N governs whether full fine-tuning is safe

Success deltas vs the frozen zero-shot source (map-paired):

| Cell | full-FT | scratch | LoRA |
|---|---|---|---|
| dynamic conc N=256 (1 world) | **−0.300** [−0.550,−0.050] ×3 seeds | −0.300 ×3 | +0.050..+0.100 |
| dynamic dist N=256 (8 worlds) | **+0.150..+0.200** | +0.150..+0.200 | +0.200 |
| dynamic conc N=16,384 (1 world, ~86% of its states) | −0.100/−0.150/−0.300 | −0.300 ×3 | 0.000 ×3 |
| dynamic conc N=65,536 (**4 worlds**) | **+0.100 ×3** | +0.100..+0.150 | +0.100..+0.150 |
| static conc N=256 (2 worlds) | **−0.167..−0.300** | −0.333..−0.367 | 0.000..−0.067 |
| static dist N=256 (16 worlds) | **0.000..−0.033** | 0.000..−0.067 | −0.033..−0.067 |

- Concentrated low-world cells collapse full-FT and scratch success in **both domains**; the 8×-distributed cells at the **same N** rescue them (paired dist−conc: static full-FT +0.233 at N=256; dynamic full-FT +0.30..+0.48 for N ≤ 16,384).
- **LoRA never collapses in any cell** — its low-supervision protection is diversity-robustness, visible in success, not in matched-solved ratios.
- The dynamic conc recovery at N=65,536 happens exactly where w_min forces 4 worlds — recovery tracks **world count**, not state count, at fixed steps and fixed N-per-cell protocol.

**Revised evidence-safe statement replacing the supervision-density account:** what protects the transferred prior at low supervision is adapter-restricted capacity (LoRA) or world diversity; raw supervised-state count is the wrong index. The paper's existing claim ("one dynamic map supplies ~25k space–time labels, so the static crossover leaves its low-data regime") requires revision: a single dynamic map's states do not protect full fine-tuning by themselves (conc cells crash at N ≤ 16,384 on 1 world); what changes in the dynamic K-indexed protocol is that map count and state count move together.

## Relation to C9/C9b

- Static: consistent. C9's dense full-FT K=1 (+0.227 vs Euclid) sat well below LoRA K=1 (+0.467) — the same "full-FT loses much of the zero-shot advantage at minimal supervision" seen here as −0.17..−0.30 vs zero-shot at conc N=256.
- Dynamic: C9b's "K=1 full-FT not catastrophic" corresponds to C14's conc N=16,384 borderline cell (1 world, most of its states, matched compute): 2 of 3 seeds are non-catastrophic there. The hard crashes appear only at state subsets (N ≤ 4,096) of a single world — a regime C9b never sampled. No contradiction; C14 refines the boundary.

## Limitations

Dynamic matched-solved ratio cells have n=1 (Euclid solves 1/20 dense test maps at the binding budget), so dynamic ratio CIs are degenerate point masses; all dynamic inference rests on the 20-map paired success deltas. Static matched cells have n=10. One target family per domain (maze-dense), per the frozen design; adaptation-seed count 3; the arms were trained on Modal L4s while sources/datasets were produced locally (recorded in the manifest; the protocol requires no cross-machine bit-identity).

## Artifacts

`runs/c14_modal/` (volume: datasets, 180 checkpoints, results, manifest; results + manifest mirrored locally), `runs/c14_local/datasets/` (static pool + cells, canonical), `continuous_prm_c14_label_density.py`, `continuous_prm_c14_modal.py`, analysis outputs beside `c14_analysis.py`.
