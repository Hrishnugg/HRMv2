# C11 -- Compositional-Mission PRM Heuristics: Results

**Date:** 2026-07-09
**Branch:** c11-mission
**Spec:** `docs/experiments/continuous/c11/design/2026-07-07-c11-compositional-mission-design.md`
**Plan:** `docs/experiments/continuous/c11/plans/2026-07-07-c11-mission.md`

## Pre-registered gates (verbatim, spec section 1)

**G1 -- dose-response in structure. Primary read: per (config, K), each structured arm vs the MLP control on matched per-world expansion-ratio differences (both-solved worlds) + success McNemar. Continuity control: at K=0 the task reduces exactly to C7 (legsum == euclid, no mission tokens beyond the goal leg) and all arms must statistically tie -- if they don't, it is a formulation/implementation bug and the phase halts for diagnosis, not a result. Positive verdict: >=1 structured arm beats the MLP with BH q < 0.05 at >=2 of the 3 K in {2,4,8} values on >=1 config, with the gap monotone non-decreasing in K. Anything less is a negative for that arm.**

**G2 -- depth-of-compute (HRM-v2 mechanism). (a) Forced-segment curves: eval the trained ACT arm at forced k in {1,2,4,8} segments; positive iff quality (state-level MAE and/or expansion-ratio) improves monotonically with bootstrap-CI separation between k=1 and k=8 on >=1 config at K=8. (b) Learned halting: Spearman correlation of mean halt-steps vs K positive with q < 0.05 ("thinks longer on deeper missions").**

**G3 -- honest closure. If G0-H passed (it did), the I/O exposes real structure, and the MLP control still ties everything: "learned planning heuristics are architecture-agnostic" graduates to a strong publishable claim; the program pivots to the transfer+integration paper with the architecture chapter closed.**

## G1 -- per-cell results

### Config A

| K | Binding budget | Arm | Success | vs-MLP ratio [CI] (n) | vs-legsum ratio (n) |
|---|---|---|---|---|---|
| 0 | 200 | gnn | 1.00 | 0.808 [0.766, 0.853] (n=75) | 0.504 (n=75) |
| 0 | 200 | h_legsum | 1.00 | n/a | n/a |
| 0 | 200 | h_next | 1.00 | n/a | n/a |
| 0 | 200 | h_oracle | 1.00 | n/a | n/a |
| 0 | 200 | hrm_trace | 1.00 | 1.312 [1.234, 1.373] (n=75) | 0.844 (n=75) |
| 0 | 200 | hrmv2_act | 1.00 | 0.927 [0.847, 1.000] (n=75) | 0.597 (n=75) |
| 0 | 200 | mlp | 1.00 | -- (control) | 0.616 (n=75) |
| 0 | 200 | onlstm_trace | 1.00 | 1.063 [1.035, 1.090] (n=75) | 0.709 (n=75) |
| 0 | 200 | unet_film | 1.00 | 0.734 [0.691, 0.765] (n=75) | 0.479 (n=75) |
| 0 | 200 | h_oracle (ceiling) | -- | -- | 0.153 (n=25) |
| 2 | 200 | gnn | 0.60 | 0.861 [0.732, 0.910] (n=33) | 0.692 (n=18) |
| 2 | 200 | h_legsum | 0.24 | n/a | n/a |
| 2 | 200 | h_next | 0.04 | n/a | n/a |
| 2 | 200 | h_oracle | 1.00 | n/a | n/a |
| 2 | 200 | hrm_trace | 0.40 | 1.015 [0.905, 1.056] (n=30) | 0.711 (n=18) |
| 2 | 200 | hrmv2_act | 0.61 | 0.890 [0.727, 0.945] (n=38) | 0.677 (n=18) |
| 2 | 200 | mlp | 0.52 | -- (control) | 0.798 (n=18) |
| 2 | 200 | onlstm_trace | 0.52 | 0.833 [0.795, 0.893] (n=36) | 0.690 (n=18) |
| 2 | 200 | unet_film | 0.63 | 0.726 [0.606, 0.765] (n=34) | 0.470 (n=18) |
| 2 | 200 | h_oracle (ceiling) | -- | -- | 0.155 (n=6) |
| 4 | 400 | gnn | 0.60 | 1.015 [0.977, 1.124] (n=45) | 0.456 (n=9) |
| 4 | 400 | h_legsum | 0.12 | n/a | n/a |
| 4 | 400 | h_next | 0.00 | n/a | n/a |
| 4 | 400 | h_oracle | 1.00 | n/a | n/a |
| 4 | 400 | hrm_trace | 0.57 | 1.161 [1.076, 1.257] (n=40) | 0.436 (n=9) |
| 4 | 400 | hrmv2_act | 0.59 | 1.161 [0.964, 1.254] (n=43) | 0.651 (n=9) |
| 4 | 400 | mlp | 0.61 | -- (control) | 0.495 (n=9) |
| 4 | 400 | onlstm_trace | 0.60 | 0.881 [0.795, 0.953] (n=43) | 0.385 (n=9) |
| 4 | 400 | unet_film | 0.63 | 0.943 [0.801, 1.025] (n=44) | 0.357 (n=9) |
| 4 | 400 | h_oracle (ceiling) | -- | -- | 0.121 (n=3) |
| 8 | 1600 | gnn | 1.00 | 0.994 [0.977, 1.030] (n=75) | 0.644 (n=60) |
| 8 | 1600 | h_legsum | 0.80 | n/a | n/a |
| 8 | 1600 | h_next | 0.80 | n/a | n/a |
| 8 | 1600 | h_oracle | 1.00 | n/a | n/a |
| 8 | 1600 | hrm_trace | 0.80 | 1.526 [1.470, 1.690] (n=60) | 1.000 (n=60) |
| 8 | 1600 | hrmv2_act | 1.00 | 1.035 [0.990, 1.094] (n=75) | 0.655 (n=60) |
| 8 | 1600 | mlp | 1.00 | -- (control) | 0.655 (n=60) |
| 8 | 1600 | onlstm_trace | 0.87 | 1.303 [1.097, 1.498] (n=65) | 1.000 (n=60) |
| 8 | 1600 | unet_film | 1.00 | 1.016 [0.969, 1.028] (n=75) | 0.633 (n=60) |
| 8 | 1600 | h_oracle (ceiling) | -- | -- | 0.082 (n=20) |

### Config B

| K | Binding budget | Arm | Success | vs-MLP ratio [CI] (n) | vs-legsum ratio (n) |
|---|---|---|---|---|---|
| 0 | 100 | gnn | 1.00 | 0.815 [0.789, 0.846] (n=75) | 0.525 (n=75) |
| 0 | 100 | h_legsum | 1.00 | n/a | n/a |
| 0 | 100 | h_next | 1.00 | n/a | n/a |
| 0 | 100 | h_oracle | 1.00 | n/a | n/a |
| 0 | 100 | hrm_trace | 1.00 | 1.038 [0.960, 1.139] (n=75) | 0.723 (n=75) |
| 0 | 100 | hrmv2_act | 1.00 | 1.368 [1.316, 1.550] (n=75) | 1.000 (n=75) |
| 0 | 100 | mlp | 1.00 | -- (control) | 0.684 (n=75) |
| 0 | 100 | onlstm_trace | 1.00 | 1.462 [1.368, 1.605] (n=75) | 1.000 (n=75) |
| 0 | 100 | unet_film | 1.00 | 0.868 [0.815, 0.925] (n=75) | 0.593 (n=75) |
| 0 | 100 | h_oracle (ceiling) | -- | -- | 0.222 (n=25) |
| 2 | 100 | gnn | 0.80 | 0.672 [0.633, 0.775] (n=50) | 0.438 (n=36) |
| 2 | 100 | h_legsum | 0.48 | n/a | n/a |
| 2 | 100 | h_next | 0.00 | n/a | n/a |
| 2 | 100 | h_oracle | 1.00 | n/a | n/a |
| 2 | 100 | hrm_trace | 0.68 | 0.950 [0.861, 1.000] (n=45) | 0.647 (n=36) |
| 2 | 100 | hrmv2_act | 0.47 | 1.045 [0.942, 1.221] (n=32) | 0.780 (n=29) |
| 2 | 100 | mlp | 0.71 | -- (control) | 0.750 (n=33) |
| 2 | 100 | onlstm_trace | 0.69 | 1.206 [0.975, 1.282] (n=47) | 0.826 (n=36) |
| 2 | 100 | unet_film | 0.60 | 0.952 [0.846, 1.212] (n=39) | 0.642 (n=32) |
| 2 | 100 | h_oracle (ceiling) | -- | -- | 0.225 (n=12) |
| 4 | 200 | gnn | 0.44 | 0.916 [0.824, 1.000] (n=21) | 0.539 (n=12) |
| 4 | 200 | h_legsum | 0.16 | n/a | n/a |
| 4 | 200 | h_next | 0.00 | n/a | n/a |
| 4 | 200 | h_oracle | 1.00 | n/a | n/a |
| 4 | 200 | hrm_trace | 0.52 | 1.102 [0.971, 1.347] (n=27) | 0.795 (n=12) |
| 4 | 200 | hrmv2_act | 0.20 | 1.315 [1.080, 1.622] (n=12) | 1.079 (n=8) |
| 4 | 200 | mlp | 0.39 | -- (control) | 0.751 (n=12) |
| 4 | 200 | onlstm_trace | 0.45 | 0.978 [0.899, 1.062] (n=29) | 0.755 (n=12) |
| 4 | 200 | unet_film | 0.48 | 0.879 [0.762, 1.064] (n=24) | 0.647 (n=12) |
| 4 | 200 | h_oracle (ceiling) | -- | -- | 0.208 (n=4) |
| 8 | 800 | gnn | 0.91 | 1.020 [0.931, 1.108] (n=68) | 0.424 (n=48) |
| 8 | 800 | h_legsum | 0.64 | n/a | n/a |
| 8 | 800 | h_next | 0.00 | n/a | n/a |
| 8 | 800 | h_oracle | 1.00 | n/a | n/a |
| 8 | 800 | hrm_trace | 0.84 | 1.038 [0.957, 1.120] (n=63) | 0.501 (n=48) |
| 8 | 800 | hrmv2_act | 0.89 | 1.231 [0.975, 1.305] (n=67) | 0.668 (n=48) |
| 8 | 800 | mlp | 0.96 | -- (control) | 0.522 (n=48) |
| 8 | 800 | onlstm_trace | 0.89 | 0.993 [0.957, 1.092] (n=67) | 0.595 (n=48) |
| 8 | 800 | unet_film | 0.91 | 0.905 [0.821, 0.937] (n=68) | 0.428 (n=48) |
| 8 | 800 | h_oracle (ceiling) | -- | -- | 0.103 (n=16) |

### Config C

| K | Binding budget | Arm | Success | vs-MLP ratio [CI] (n) | vs-legsum ratio (n) |
|---|---|---|---|---|---|
| 2 | 200 | gnn | 0.60 | 0.982 [0.883, 1.052] (n=41) | 0.592 (n=15) |
| 2 | 200 | h_legsum | 0.20 | n/a | n/a |
| 2 | 200 | h_next | 0.04 | n/a | n/a |
| 2 | 200 | h_oracle | 1.00 | n/a | n/a |
| 2 | 200 | hrm_trace | 0.60 | 1.109 [1.031, 1.202] (n=39) | 0.747 (n=15) |
| 2 | 200 | hrmv2_act | 0.57 | 0.945 [0.858, 1.049] (n=38) | 0.429 (n=15) |
| 2 | 200 | mlp | 0.56 | -- (control) | 0.582 (n=15) |
| 2 | 200 | onlstm_trace | 0.60 | 0.911 [0.840, 0.962] (n=42) | 0.500 (n=15) |
| 2 | 200 | unet_film | 0.59 | 0.881 [0.809, 0.961] (n=39) | 0.365 (n=15) |
| 2 | 200 | h_oracle (ceiling) | -- | -- | 0.144 (n=5) |
| 4 | 400 | gnn | 0.52 | 1.117 [1.035, 1.206] (n=39) | 0.461 (n=9) |
| 4 | 400 | h_legsum | 0.12 | n/a | n/a |
| 4 | 400 | h_next | 0.00 | n/a | n/a |
| 4 | 400 | h_oracle | 1.00 | n/a | n/a |
| 4 | 400 | hrm_trace | 0.59 | 1.228 [1.154, 1.285] (n=44) | 0.415 (n=9) |
| 4 | 400 | hrmv2_act | 0.43 | 1.348 [1.155, 1.476] (n=32) | 0.637 (n=9) |
| 4 | 400 | mlp | 0.67 | -- (control) | 0.411 (n=9) |
| 4 | 400 | onlstm_trace | 0.59 | 1.068 [1.010, 1.164] (n=44) | 0.466 (n=9) |
| 4 | 400 | unet_film | 0.60 | 1.100 [1.045, 1.204] (n=43) | 0.450 (n=9) |
| 4 | 400 | h_oracle (ceiling) | -- | -- | 0.128 (n=3) |
| 8 | 1600 | gnn | 1.00 | 1.003 [0.948, 1.042] (n=75) | 0.620 (n=69) |
| 8 | 1600 | h_legsum | 0.92 | n/a | n/a |
| 8 | 1600 | h_next | 0.84 | n/a | n/a |
| 8 | 1600 | h_oracle | 1.00 | n/a | n/a |
| 8 | 1600 | hrm_trace | 0.92 | 1.600 [1.509, 1.647] (n=69) | 1.000 (n=69) |
| 8 | 1600 | hrmv2_act | 1.00 | 1.042 [0.989, 1.078] (n=75) | 0.675 (n=69) |
| 8 | 1600 | mlp | 1.00 | -- (control) | 0.625 (n=69) |
| 8 | 1600 | onlstm_trace | 1.00 | 0.967 [0.926, 0.984] (n=75) | 0.626 (n=69) |
| 8 | 1600 | unet_film | 1.00 | 1.014 [0.980, 1.066] (n=75) | 0.661 (n=69) |
| 8 | 1600 | h_oracle (ceiling) | -- | -- | 0.084 (n=23) |

## K=0 continuity control

**Verdict: FAIL**

Violations (config_label, arm, reason):

- A / gnn: ci_hi=0.8529 < 1.0
- A / hrm_trace: ci_lo=1.234 > 1.0
- A / onlstm_trace: ci_lo=1.035 > 1.0
- A / unet_film: ci_hi=0.7647 < 1.0
- B / gnn: ci_hi=0.8462 < 1.0
- B / hrmv2_act: ci_lo=1.316 > 1.0
- B / onlstm_trace: ci_lo=1.368 > 1.0
- B / unet_film: ci_hi=0.925 < 1.0

## G1 verdict

**Negative.** No structured arm beat the MLP control at >=2 of the 3 K values on any config.

| Arm | Config | K=2 beat | K=4 beat | K=8 beat | Monotone | Status |
|---|---|---|---|---|---|---|
| gnn | A | yes | no | no | no | negative |
| gnn | B | yes | no | no | no | negative |
| gnn | C | no | no | no | no | negative |
| hrm_trace | A | no | no | no | no | negative |
| hrm_trace | B | no | yes | no | no | negative |
| hrm_trace | C | no | no | no | no | negative |
| hrmv2_act | A | yes | no | no | no | negative |
| hrmv2_act | B | no | no | no | no | negative |
| hrmv2_act | C | no | no | no | no | negative |
| onlstm_trace | A | yes | yes | no | no | negative |
| onlstm_trace | B | no | no | no | no | negative |
| onlstm_trace | C | yes | no | yes | no | negative |
| unet_film | A | yes | no | no | no | negative |
| unet_film | B | no | no | yes | no | negative |
| unet_film | C | yes | no | no | no | negative |

## G2a -- forced-segment curves (K=8)

| Config | k=1 ratio | k=2 ratio | k=4 ratio | k=8 ratio | act-live ratio [CI] | k=1 mae | k=2 mae | k=4 mae | k=8 mae | act-live mae |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.660 | 0.655 | 0.662 | 0.655 | 0.655 [0.623, 0.703] | 0.881 | 0.878 | 0.878 | 0.879 | 0.879 |
| B | 0.652 | 0.645 | 0.651 | 0.668 | 0.668 [0.576, 0.721] | 0.356 | 0.356 | 0.356 | 0.356 | 0.356 |
| C | 0.676 | 0.685 | 0.688 | 0.675 | 0.675 [0.622, 0.722] | 0.773 | 0.777 | 0.778 | 0.778 | 0.778 |

**G2a verdict: negative or insufficient data.**

## G2b -- halting vs mission depth

Pooled Spearman rho = -0.407, permutation p = 0.000 (n=675).

**G2b verdict: negative.**

## G3 -- closure

**G3 closure WITHHELD pending diagnosis.** The K=0 continuity control FAILED (see the K=0 section above). Per the pre-registered protocol (spec section 1: at K=0 all arms must statistically tie -- "if they don't, it is a formulation/implementation bug and the phase halts for diagnosis, not a result"), the G1 dose-response verdict line above still renders -- it is computed mechanically from the records -- but it is NOT interpretable as an architecture result until the K=0 separation is diagnosed, and no pre-registered closure claim (all-tie, all-positive, or partial) is rendered for this run.

K=0 violations (config, arm, reason):

- A / gnn: ci_hi=0.8529 < 1.0
- A / hrm_trace: ci_lo=1.234 > 1.0
- A / onlstm_trace: ci_lo=1.035 > 1.0
- A / unet_film: ci_hi=0.7647 < 1.0
- B / gnn: ci_hi=0.8462 < 1.0
- B / hrmv2_act: ci_lo=1.316 > 1.0
- B / onlstm_trace: ci_lo=1.368 > 1.0
- B / unet_film: ci_hi=0.925 < 1.0

The 'Diagnosis addendum' section at the end of this document carries the post-diagnosis interpretation.

## Caveats

- **HRM-v2 optimizer/loop confound (pre-registered accepted).** The HRM-v2 arm's optimizer (adam-atan2 + constant LR + warmup) and per-segment training loop are paper-faithful, not matched to arms 1-4's AdamW/smooth-L1 recipe -- mechanism-faithfulness IS the arm, and this confound was pre-registered as accepted (spec section 5).
- **GNN m=8 rounds (pre-registered).** No post-hoc m tuning; a single pre-registered m-sensitivity appendix run at m in {4,16} on one cell is allowed for interpretation only (spec section 8).
- **Param-count spread within band.** Per-arm parameter counts (from the training manifest, via `meta`): gnn=681,089, hrm_trace=2,901,473, hrmv2_act=3,414,786, mlp=1,274,881, onlstm_trace=1,251,457, unet_film=2,096,521. Every native arm targets the [0.5M, 3.5M] band; the HRM-v2 arm targets [1M, 4M] (spec section 3, item 5) -- these bands overlap but are not identical, a known spread.
- **`door_open_at_s` trace slot (T2-review note).** This trace-token slot is structurally 0.0 for configs without doors, and even for config C it does not directly encode live door state -- door state is carried by leg presence/ordering in the mission trace itself, not by this slot. Documented, not a bug.
- **Eval-time fixed-compute ACT.** The would-halt channel (`hrmv2_halt_steps`) is a readout of the model's OWN halting signal (`q_halt_logits > q_continue_logits`), not an actual variable-compute eval loop -- ACT-live eval still runs to `carry.halted.all()` (or the safety cap), so G2b measures learned depth-preference, not realized eval-time compute savings.
- **Oracle headroom is an upper bound.** `h_oracle` is the exact product-graph shortest path; every learned arm is explicitly inadmissible (the residual formulation has no admissibility guarantee at eval time) -- learned-arm costs are recorded but never asserted optimal, and the oracle row in the G1 tables above is a CEILING, not a target any learned arm is expected to reach.

## Diagnosis addendum

**K=0 continuity failure: diagnosed 2026-07-09. Classification: NOT a measurement bug — a premise failure plus a training pathology, and the corrected reading below supersedes any closure text.**

The eval/stats layer was verified correct end to end (paired ratios hand-recomputed from the raw CSV match the tables to the fourth decimal; K=0 binding budgets calibrated non-degenerate; legsum == euclid at K=0 to machine precision). What failed is the pre-registered premise that K=0 "must reproduce the C7 three-way tie": K=0 reduces the *task* to C7's, but not the *experiment* — C7 never ran an MLP control (the program audit's central complaint) and used a 24-token feature bag, whereas C11 gives each arm its class-native encoding. With a real control in place, the arms separate. **This formulation is not MLP-saturable — the audit's question, answered.**

**Finding 1 — global-input architectures hold a genuine, shallow-K advantage.** unet_film and gnn (the arms that see the world globally: occupancy raster; full product graph) beat the MLP control with CI separation at K in {0,2} (vs-MLP ratios 0.67–0.87) and the advantage dissolves by K in {4,8} (state-MAE roughly triples from K=0 to K=8 for every arm, drowning the constant-factor edge). The advantage tracks state-level fit exactly (A0 MAE: unet 0.151, gnn 0.171, mlp 0.251). It is not a dose-response: depth erodes it rather than amplifying it.

**Finding 2 — the recurrent/ACT arms are training-fragile under the matched recipe.** Five of 33 arm-cell combos collapsed to constant predictions (hrm_trace A8/C8 at the residual cap 4.0 — the manifest's identical final_loss 1.1397 on A8 across arms is exactly the constant-cap loss; onlstm_trace B0 at mean(y); hrmv2_act B0 at 0 with softplus-saturated logits; onlstm A8 2/3 seeds). A constant residual is A*-equivalent to the leg-sum baseline, so the affected cells (including the largest "deficits" in the G1 tables: hrm_trace A8 1.526, C8 1.600, onlstm B0 1.462, hrmv2 B0 1.368) measure an optimization pathology, not architectural expressiveness. Where training succeeded, the recurrent arms mostly tie the control (hrm_trace A0 1.312 and onlstm A0 1.063 are real, non-collapsed fit deficits; onlstm C8 0.967 is a real small win).

**Finding 3 — the pad-asymmetry suspect was tested and rejected.** Stripping the 8 zero-pad rows from K=0 trace inputs makes trained recurrent predictions substantially WORSE (hrm_trace A0 MAE 0.257 -> 0.389; prediction spread collapses ~150x): the models learned to use the zero-input pad steps as extra recurrence compute. The pads are, if anything, an asymmetric favor — and the recurrent arms still lose, strengthening the "real deficit" reading. (hrmv2's architecture pins seq_len=10 via learned positional embeddings; it structurally requires the pads.)

**Finding 4 — G2b is negative in the inverted direction.** The would-halt channel varies (per-world means span 1–8) and is significantly ANTI-correlated with mission depth: Spearman rho = -0.407, p ~= 0.0005 (pooled K in {2,4,8}: mean would-halt 6.79 / 7.01 / 5.30). The trained Q-head votes to halt EARLIER on deeper missions — the opposite of "thinks longer on harder problems."

**Corrected gate readings.** G1 stays negative as pre-registered — no arm converts mission depth into a growing advantage over the MLP control on cells where training succeeded — but the hrm/onlstm K=8 rows must be read as collapse measurements. G2a negative (forced extra segments do not improve quality). G2b negative with an inverted sign. **The withheld G3 closure resolves to neither pre-registered branch:** "architecture-agnostic" is contradicted by the two-sided K=0/K=2 separations, and the dose-response/hierarchy thesis is contradicted by the shrinking gap. The honest claim: *on a compositional substrate with measured 5–12x oracle headroom, learned-heuristic quality is architecture-DEPENDENT (global-input conv/graph models > flat MLP > sequence/recurrent models at shallow depth), but no architecture — including the faithful HRM-v2 ACT mechanism — converts compositional depth into advantage; depth degrades every arm, and the recurrent/ACT class is additionally fragile under a matched recipe.*

**Provenance.** Diagnosis executed as five pre-specified probes (cross-K extraction, pad-sensitivity on trained checkpoints, collapse census, by-hand measurement recomputation, halt-channel characterization); probe scripts under the session scratchpad; no repo code was modified by the diagnosis. The scaled addendum (T10) selects unet_film (best) and hrm_trace (worst) and additionally tests whether 10-20M-param variants escape the collapse attractor.

