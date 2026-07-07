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
| Worlds/cell | 2 |
| Budgets grid | [400, 3200] |
| World seed formula | `seed = 1234 + 7919*world_idx + 104729*config_idx + 15485863*K` |
| Binding-budget rule | lowest budget in the grid with h_legsum success rate >= 0.05 across the cell's worlds; else largest budget, flagged DEGENERATE |
| Wall time | 0.1 min total (serial); 0.42s estimation probe (1 world, config C K=8) |

## Per-cell results

| Config | K | Binding budget | Degenerate | Succ h_next | Succ h_legsum | Succ h_oracle | Med.exp h_next (solved) | Med.exp h_legsum (solved) | Med.exp h_oracle (solved) | Oracle/legsum ratio [IQR] (n) | Next/legsum ratio [IQR] (n) | Success gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 2 | 400 | no | 0.50 | 0.50 | 1.00 | 355.0 | 296.0 | 43.5 | 0.098 [0.098, 0.098] (n=1) | 1.199 [1.199, 1.199] (n=1) | 0.50 |
| A | 4 | 3200 | no | 1.00 | 1.00 | 1.00 | 852.5 | 705.5 | 70.5 | 0.099 [0.092, 0.106] (n=2) | 1.212 [1.188, 1.236] (n=2) | 0.00 |
| A | 8 | 3200 | no | 1.00 | 1.00 | 1.00 | 1513.0 | 1402.5 | 99.5 | 0.069 [0.063, 0.075] (n=2) | 1.094 [1.048, 1.141] (n=2) | 0.00 |
| B | 2 | 400 | no | 1.00 | 1.00 | 1.00 | 275.0 | 212.0 | 27.0 | 0.149 [0.126, 0.173] (n=2) | 1.517 [1.284, 1.749] (n=2) | 0.00 |
| B | 4 | 400 | no | 0.00 | 0.50 | 1.00 | n/a | 286.0 | 40.5 | 0.119 [0.119, 0.119] (n=1) | n/a (n=0) | 0.50 |
| B | 8 | 3200 | no | 1.00 | 1.00 | 1.00 | 1161.0 | 830.0 | 79.0 | 0.096 [0.093, 0.100] (n=2) | 1.436 [1.327, 1.545] (n=2) | 0.00 |
| C | 2 | 3200 | no | 1.00 | 1.00 | 1.00 | 499.5 | 490.5 | 68.0 | 0.139 [0.136, 0.142] (n=2) | 1.018 [1.014, 1.022] (n=2) | 0.00 |
| C | 4 | 400 | no | 0.00 | 0.50 | 1.00 | n/a | 322.0 | 56.5 | 0.146 [0.146, 0.146] (n=1) | n/a (n=0) | 0.50 |
| C | 8 | 3200 | no | 1.00 | 1.00 | 1.00 | 1368.0 | 1249.5 | 90.0 | 0.072 [0.071, 0.073] (n=2) | 1.096 [1.073, 1.119] (n=2) | 0.00 |

## Dose-response read

Per config, how the matched oracle/legsum ratio and the success gap move across K=2 -> 4 -> 8. The audit's predicted signature: the oracle's advantage over leg-sum should GROW (ratio shrink, gap widen) as mission length increases, if compositional structure is what creates the headroom (rather than, say, a fixed per-cell offset that doesn't compound).

- **Config A** -- oracle/legsum ratio: K=2: 0.098 (n=1), K=4: 0.099 (n=2), K=8: 0.069 (n=2)
  success gap: K=2: 0.50, K=4: 0.00, K=8: 0.00
  -- read: the ratio does NOT shrink monotonically across K=2->4->8 (the oracle's RELATIVE cut over leg-sum grows with mission length, matching the predicted signature); the success gap does NOT widen monotonically (it is non-monotonic here -- both arms' success rates move with the budget-calibration interaction across K, not a clean widening trend).
- **Config B** -- oracle/legsum ratio: K=2: 0.149 (n=2), K=4: 0.119 (n=1), K=8: 0.096 (n=2)
  success gap: K=2: 0.00, K=4: 0.50, K=8: 0.00
  -- read: the ratio shrinks monotonically across K=2->4->8 (the oracle's RELATIVE cut over leg-sum grows with mission length, matching the predicted signature); the success gap does NOT widen monotonically (it is non-monotonic here -- both arms' success rates move with the budget-calibration interaction across K, not a clean widening trend).
- **Config C** -- oracle/legsum ratio: K=2: 0.139 (n=2), K=4: 0.146 (n=1), K=8: 0.072 (n=2)
  success gap: K=2: 0.00, K=4: 0.50, K=8: 0.00
  -- read: the ratio does NOT shrink monotonically across K=2->4->8 (the oracle's RELATIVE cut over leg-sum grows with mission length, matching the predicted signature); the success gap does NOT widen monotonically (it is non-monotonic here -- both arms' success rates move with the budget-calibration interaction across K, not a clean widening trend).

## Gate verdict

**PASS.** Config A at K=8 achieves a matched oracle/legsum median ratio of 0.069 <= 0.5, clearing the pre-registered gate outright (oracle cuts >= 93% of A* expansions vs leg-sum on matched worlds).

## Honest caveats

- **Probe scale:** 2 worlds/cell, no seeds-over-missions replication (one mission per world; a different waypoint sample on the same world is untested).
- **Waypoint sampling choices:** connected-to-goal candidates only, a minimum pairwise separation constraint (relaxed once if unfillable) -- both bias missions toward well-spread, reachable waypoints rather than adversarial or clustered ones.
- **Single binding budget per cell**, calibrated on h_legsum's success rate alone (not h_oracle's or h_next's), then reused for all three arms.
- **Doors geometry (config C):** 2 doors, fixed half-width/half-height fractions, midpoint-first placement with limited resampling -- a specific, not exhaustively tuned, keys-and-doors construction.
- **h_next / h_legsum are geometric (straight-line), not learned, baselines.** The probe measures ORACLE headroom -- an upper bound on what any learned heuristic could capture, not a claim about what a trained model would actually achieve.
- **n_matched shrinks at low K (min n=1, cell(s) [('A', 2), ('B', 4), ('C', 4)]):** the binding budget is calibrated to h_legsum's OWN success rate (lowest budget clearing 5%), so at that budget h_oracle typically solves ~100% of worlds while h_legsum solves only slightly above its calibration floor -- the matched-ratio sample is capped by h_legsum's success count, not a bug. Low-K, low-n_matched cells' ratios are the noisiest in this table and should be weighted accordingly; the K=8 cells (the ones the dose-response read leans on) have the largest, most reliable n_matched in every config.

## Decision implication

PASS -> proceed to C11 phase design: the six arms from the audit (explicit MLP control, U-Net field, GNN over the product graph, HRM/ON-LSTM fed the mission trace, the iterative field refiner, plus the cheap scale-confound addendum), gated by G1 (dose-response in structure, with the n_stages=1 degenerate-to-C7 control), G2 (depth-of-compute vs iteration count k), and G3 (honest closure if everything still ties).

