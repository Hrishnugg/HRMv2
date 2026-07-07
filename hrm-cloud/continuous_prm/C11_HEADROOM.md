# C11 G0-H Headroom Probe -- Results

**Date:** 2026-07-07
**Purpose:** G0-H gate for C11 (compositional-mission PRM) per `../PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md` §7 -- measure whether compositional missions create real heuristic headroom for A* search BEFORE building the C11 phase, at the cost of a probe rather than a phase.
**Pre-registered gate (verbatim):** the exact oracle must cut ≥40–50% of A* expansions vs the admissible leg-sum baseline at the binding budget (i.e. matched median ratio ≤ 0.5–0.6) on ≥1 config — ideally with the gap GROWING in K (dose-response).
**Spec/plan:** `../../docs/superpowers/plans/2026-07-07-c11-headroom-probe.md`.

## Run configuration

| Knob | Value |
|---|---|
| Config A | `C_hard_maze` waypoint missions, config_idx=0, no adjacency factory |
| Config B | `C_hard_rooms_large` waypoint missions, config_idx=1, no adjacency factory |
| Config C | `C_hard_maze` + keys→doors stage-dependent adjacency, config_idx=2 |
| K grid | [2, 4, 8] |
| Worlds/cell | 25 |
| Budgets grid | [100, 200, 400, 800, 1600, 3200] |
| World seed formula | `seed = 1234 + 7919*world_idx + 104729*config_idx + 15485863*K` |
| Binding-budget rule | lowest budget in the grid with h_legsum success rate >= 0.05 across the cell's worlds; else largest budget, flagged DEGENERATE |
| Wall time | 1.1 min total (serial); 0.36s estimation probe (1 world, config C K=8) |

## Per-cell results

| Config | K | Binding budget | Degenerate | Succ h_next | Succ h_legsum | Succ h_oracle | Med.exp h_next (solved) | Med.exp h_legsum (solved) | Med.exp h_oracle (solved) | Oracle/legsum ratio [IQR] (n) | Next/legsum ratio [IQR] (n) | Success gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 2 | 200 | no | 0.04 | 0.24 | 1.00 | 185.0 | 154.0 | 33.0 | 0.155 [0.147, 0.164] (n=6) | 1.217 [1.217, 1.217] (n=1) | 0.76 |
| A | 4 | 400 | no | 0.00 | 0.12 | 1.00 | n/a | 397.0 | 56.0 | 0.121 [0.108, 0.133] (n=3) | n/a (n=0) | 0.88 |
| A | 8 | 1600 | no | 0.80 | 0.80 | 1.00 | 1457.5 | 1350.5 | 108.0 | 0.082 [0.075, 0.084] (n=20) | 1.083 [1.042, 1.192] (n=20) | 0.20 |
| B | 2 | 100 | no | 0.00 | 0.48 | 1.00 | n/a | 88.5 | 21.0 | 0.225 [0.215, 0.240] (n=12) | n/a (n=0) | 0.52 |
| B | 4 | 200 | no | 0.00 | 0.16 | 1.00 | n/a | 138.5 | 42.0 | 0.208 [0.167, 0.253] (n=4) | n/a (n=0) | 0.84 |
| B | 8 | 800 | no | 0.00 | 0.64 | 1.00 | n/a | 607.0 | 66.0 | 0.103 [0.097, 0.109] (n=16) | n/a (n=0) | 0.36 |
| C | 2 | 200 | no | 0.04 | 0.20 | 1.00 | 167.0 | 170.0 | 34.0 | 0.144 [0.118, 0.167] (n=5) | 1.201 [1.201, 1.201] (n=1) | 0.80 |
| C | 4 | 400 | no | 0.00 | 0.12 | 1.00 | n/a | 322.0 | 65.0 | 0.128 [0.118, 0.137] (n=3) | n/a (n=0) | 0.88 |
| C | 8 | 1600 | no | 0.84 | 0.92 | 1.00 | 1489.0 | 1365.0 | 105.0 | 0.084 [0.072, 0.089] (n=23) | 1.094 [1.047, 1.158] (n=21) | 0.08 |

## Dose-response read

Per config, how the matched oracle/legsum ratio and the success gap move across K=2 -> 4 -> 8. The audit's predicted signature: the oracle's advantage over leg-sum should GROW (ratio shrink, gap widen) as mission length increases, if compositional structure is what creates the headroom (rather than, say, a fixed per-cell offset that doesn't compound).

- **Config A** -- oracle/legsum ratio: K=2: 0.155 (n=6), K=4: 0.121 (n=3), K=8: 0.082 (n=20)
  success gap: K=2: 0.76, K=4: 0.88, K=8: 0.20
  -- read: the ratio shrinks monotonically across K=2->4->8 (the oracle's RELATIVE cut over leg-sum grows with mission length, matching the predicted dose-response signature); the success gap does NOT widen monotonically (both arms' success rates move with the per-cell budget-calibration interaction across K, not a clean widening trend).
- **Config B** -- oracle/legsum ratio: K=2: 0.225 (n=12), K=4: 0.208 (n=4), K=8: 0.103 (n=16)
  success gap: K=2: 0.52, K=4: 0.84, K=8: 0.36
  -- read: the ratio shrinks monotonically across K=2->4->8 (the oracle's RELATIVE cut over leg-sum grows with mission length, matching the predicted dose-response signature); the success gap does NOT widen monotonically (both arms' success rates move with the per-cell budget-calibration interaction across K, not a clean widening trend).
- **Config C** -- oracle/legsum ratio: K=2: 0.144 (n=5), K=4: 0.128 (n=3), K=8: 0.084 (n=23)
  success gap: K=2: 0.80, K=4: 0.88, K=8: 0.08
  -- read: the ratio shrinks monotonically across K=2->4->8 (the oracle's RELATIVE cut over leg-sum grows with mission length, matching the predicted dose-response signature); the success gap does NOT widen monotonically (both arms' success rates move with the per-cell budget-calibration interaction across K, not a clean widening trend).

## Gate verdict

**PASS.** Config A at K=8 achieves a matched oracle/legsum median ratio of 0.082 <= 0.5, clearing the pre-registered gate outright (oracle cuts >= 92% of A* expansions vs leg-sum on matched worlds).

## Honest caveats

- **Probe scale:** 25 worlds/cell, no seeds-over-missions replication (one mission per world; a different waypoint sample on the same world is untested).
- **Waypoint sampling choices:** connected-to-goal candidates only, a minimum pairwise separation constraint (relaxed once if unfillable) -- both bias missions toward well-spread, reachable waypoints rather than adversarial or clustered ones.
- **Single binding budget per cell**, calibrated on h_legsum's success rate alone (not h_oracle's or h_next's), then reused for all three arms.
- **Doors geometry (config C):** 2 doors, fixed half-width/half-height fractions, midpoint-first placement with limited resampling -- a specific, not exhaustively tuned, keys-and-doors construction.
- **h_next / h_legsum are geometric (straight-line), not learned, baselines.** The probe measures ORACLE headroom -- an upper bound on what any learned heuristic could capture, not a claim about what a trained model would actually achieve.
- **n_matched shrinks at low K (min n=3, cell(s) [('A', 4), ('C', 4)]):** the binding budget is calibrated to h_legsum's OWN success rate (lowest budget clearing 5%), so at that budget h_oracle typically solves ~100% of worlds while h_legsum solves only slightly above its calibration floor -- the matched-ratio sample is capped by h_legsum's success count, not a bug. Low-K, low-n_matched cells' ratios are the noisiest in this table and should be weighted accordingly; the K=8 cells (the ones the dose-response read leans on) have the largest, most reliable n_matched in every config.

## Decision implication

PASS -> proceed to C11 phase design: the six arms from the audit (explicit MLP control, U-Net field, GNN over the product graph, HRM/ON-LSTM fed the mission trace, the iterative field refiner, plus the cheap scale-confound addendum), gated by G1 (dose-response in structure, with the n_stages=1 degenerate-to-C7 control), G2 (depth-of-compute vs iteration count k), and G3 (honest closure if everything still ties).

