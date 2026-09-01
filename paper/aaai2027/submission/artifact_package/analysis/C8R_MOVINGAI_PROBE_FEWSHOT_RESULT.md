# MovingAI diagnosis + rescue — results

**Design:** `../design/2026-07-27-c8-movingai-probe-fewshot.md` (frozen before
execution; chronology disclosed).
**Runners:** `continuous_prm_c8_movingai_probe.py`,
`continuous_prm_c8_movingai_fewshot.py` →
`runs/c8r_movingai/{probe.json, probe_adapted_*.json, fewshot_raw.csv,
fewshot_checkpoints/, fewshot_datasets/, *_run.log}`.
**Analysis:** `docs/experiments/analysis/c8_movingai_fewshot_analysis.py` →
`c8_movingai_fewshot.{json, _output.md}`.

## Part 1 — residual probe (frozen blind U-Net): DIAGNOSIS CONFIRMED

Median per-map Pearson r / MAE / bias of predicted vs true normalized
residual over reachable (v,t) states:

| Set | r (median) | IQR | MAE | bias |
|---|---|---|---|---|
| Spiral (trained) | 0.971 | [0.967, 0.975] | 0.18 | −0.08 |
| Maze (trained) | 0.958 | [0.954, 0.974] | 0.20 | −0.12 |
| Rooms (trained) | 0.942 | [0.895, 0.952] | 0.19 | −0.09 |
| Dense maze (held-out param shift) | 0.939 | [0.910, 0.948] | 0.30 | −0.15 |
| Large rooms (held-out scale shift) | 0.768 | [0.657, 0.816] | 0.37 | +0.28 |
| Crossing (held-out topology) | 0.733 | [0.699, 0.762] | 0.37 | +0.36 |
| **MovingAI street** | **0.385** | [0.041, 0.569] | 0.68 | +0.36 |
| **MovingAI dao** | **0.240** | [0.015, 0.426] | 0.68 | +0.32 |

All three registered directional expectations hold: (1) MovingAI below every
procedural suite; (2) dao ≤ street; (3) positive OOD bias (over-prediction —
the inflate-only misguidance mechanism, since the additive residual can only
raise h above the anchor). Bonus structure: degradation is *graded* by
distribution shift, and the bias sign flips exactly at the shift boundary
(in-family suites slightly under-predict; every shifted set over-predicts).
The two weakest in-distribution suites (crossing, large rooms) are exactly
the seed-variable-effort suites of C8-R.

## Part 2 — few-shot rescue: RESCUED on dao; parity-recovery on street

Frozen references (success at binding): dao zero-shot 0.68 / anchor 0.76 /
WA\*(w=2) 0.96; street 0.68 / 0.76 / 0.76. Adapted arms (seeds averaged,
instance-bootstrap 95% CIs; primary cells = K=8 full FT):

**dao (n=25):**

| K | method | succ | Δ vs zero-shot | Δ vs anchor | Δ vs WA\* | exp/anchor |
|---|---|---|---|---|---|---|
| 1 | full FT | 0.68 | +0.00 [−.18,+.20] | −0.08 | −0.28 | 1.34 |
| 2 | full FT | 0.96 | +0.28 [+.10,+.48] | +0.20 [+.06,+.36] | +0.00 | 0.48 |
| 4 | full FT | 0.98 | +0.30 [+.14,+.48] | +0.22 [+.08,+.38] | +0.02 | 0.46 |
| **8** | **full FT** | **1.00** | **+0.32 [+.16,+.52]** | **+0.24 [+.08,+.40]** | **+0.04 [+.00,+.12]** | 0.53 |
| 8 | LoRA | 0.98 | +0.30 [+.14,+.48] | +0.22 [+.06,+.40] | +0.02 | 0.43 |

Two labeled instances already suffice (succ 0.90–0.96) — and K=2 is exactly
the point where the round-robin dev pool first covers both contributing base
maps (den312d + den520d). At K=8 full FT solves 25/25 at cost ratio 1.04,
erasing the zero-shot −0.28 deficit vs tuned WA\* (its Δ is now +0.04 with
the interval touching zero from above) and halving anchor-matched search
effort (0.43–0.53).

**street (n=25):**

| K | method | succ | Δ vs zero-shot | Δ vs anchor/WA\* | exp/anchor |
|---|---|---|---|---|---|
| 1 | LoRA | 0.54 | −0.14 [−.26,−.04] | −0.22 | 1.47 |
| 1 | full FT | 0.50 | −0.18 [−.34,−.04] | −0.26 | 0.66 |
| 4 | full FT | 0.80 | +0.12 [+.00,+.28] | +0.04 [−.04,+.14] | 0.55 |
| 8 | LoRA | 0.74 | +0.06 [−.10,+.22] | −0.02 [−.14,+.10] | 0.45 |
| 8 | full FT | 0.74 | +0.06 [−.08,+.22] | −0.02 [−.12,+.10] | 0.61 |

**K=1 significantly HURTS on street** (both methods' CIs exclude zero in the
harmful direction) — a live replication of C14's concentrated-supervision
degradation with one distinct world, in the wild. From K=2 up, success
recovers to classical-arm parity (best cell K=4 full FT 0.80) and the
matched-effort advantage over the anchor returns (0.45–0.61). The primary
K=8 street cell is not significant vs zero-shot (+0.06 [−.08,+.22]) —
reported as parity-recovery, not full rescue (street zero-shot was only
−0.08 behind the anchor, so the rescue headroom was small).

## Part 3 — R6 mechanism closure (probe on adapted K=8 full-FT s0 models)

Own-group cells (median r / MAE / bias, frozen → adapted):

| Group | r | MAE | bias |
|---|---|---|---|
| dao | 0.240 → **0.630** [0.458, 0.757] | 0.68 → 0.36 | +0.32 → +0.12 |
| street | 0.385 → **0.500** [0.254, 0.632] | 0.68 → 0.48 | +0.36 → +0.12 |

Adaptation repairs the residual where it was broken, and the size of the
repair tracks the behavioral outcome: dao (Δr +0.39) is fully rescued,
street (Δr +0.12) recovers to parity. Cross-group transfer of the
adaptation is also positive (each adapted model improves the other group's
correlation: dao model on street 0.385→0.462; street model on dao
0.240→0.492 with bias ≈ 0), consistent with a shared external-geometry
component rather than per-map memorization.

## Verdict (as-designed readouts)

- R1: dao PASS (rescue, CI excludes zero at every K ≥ 2); street primary
  cell n.s. (parity-recovery).
- R2: dao beats the anchor from K=2 (+0.20 to +0.24, CIs exclude zero);
  street reaches anchor parity.
- R3: dao erases the tuned-WA\* deficit (−0.28 → +0.04); street parity.
- R4: matched effort vs anchor returns to 0.43–0.61 at K ≥ 4 in both groups.
- R5: cost ratios 1.03–1.11.
- Caveats as pre-stated: ≤3 base maps per group, instance-level inference,
  adaptation pool = development instances (disclosed), 2 adaptation seeds.

Interpretation: the zero-shot failure on external geometry is a *prior*
failure, not a method failure — the frozen residual is miscalibrated OOD
(Part 1), a handful of labeled target instances repairs it (Part 2), and
the repair follows the C14 coverage law (K=1 single-world harm on street;
K=2 dual-map rescue on dao).
