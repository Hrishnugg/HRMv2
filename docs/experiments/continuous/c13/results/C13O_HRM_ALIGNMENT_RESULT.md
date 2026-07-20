# C13-O result: summary-last HRM readout alignment

**Preregistration frozen:** 2026-07-17  
**Completed:** 2026-07-19  
**Design:** [2026-07-17-c13o-hrm-summary-last-alignment.md](../design/2026-07-17-c13o-hrm-summary-last-alignment.md)  
**Generated report:** [C13O_RESULT.md](../../../../../hrm-cloud/continuous_prm/runs/c13_hrm_alignment/results/C13O_RESULT.md)  
**Verdict:** `summary_last_alignment_development_gate_failed`  
**Untouched confirmation:** not authorized and not run

## Bottom line

Moving the type-tagged summary token from the first to the final valid recurrent
position does **not** recover the C13-M method or support a robust readout-order
explanation.

At the fixed iteration-8, alpha-1.50 endpoint, summary-last HRM still contains
useful search signal: it averages 69.375 expansions versus 75.917 for field HRM,
a paired delta of -6.542 with bootstrap 95% CI [-13.251, -0.333]. However, only
three of six suite means improve. Its mean/max graph-optimal path-cost ratios
are 1.019248/1.101261 versus 1.013142/1.038347 for the matched flat MLP, exceeding
the preregistered +0.005/+0.02 flat-relative margins.

The direct readout comparison also fails at the endpoint. Summary-last averages
69.375 expansions versus 67.292 for frozen trimmed HRM: delta +2.083, 95% CI
[-1.376, +5.792]. No cell passes the complete method gate, so no candidate is
selected and the offset-20M confirmation block remains untouched.

The claim-safe conclusion is:

> Summary-last ordering produces a real but checkpoint-dependent change in HRM
> search behavior. It improves the trimmed control at iteration 6, but the
> effect does not recover robust field-HRM gains, does not persist to the fixed
> endpoint, and does not restore matched-flat path quality. Readout order is a
> contributing alignment factor, not the missing mechanism by itself.

## Frozen intervention and audit

C13-O changed only the valid-token order seen by the recurrent backbone:

- `hrm_trimmed`: summary, 32 angular rays, then valid one-hop actions;
- `hrm_summary_last`: 32 rays, valid actions, then the same summary token.

The representation already carries explicit one-hot token types. Parameter
shapes, model seed, data-loader seeds, optimizer, self-bootstrap target,
8-outer-iteration x 5-inner-epoch schedule, local radius, Bellman integration,
and no-reopen A* were held fixed.

The pre-training audit established:

- identical initial state tensors for trimmed and summary-last HRM, with the
  same state-dict SHA-256;
- 119,105 parameters in each recurrent model;
- 171/171 C13-J integrity entries checked with zero mismatches;
- 21/21 C13-N control integrity entries checked with zero mismatches;
- exact replay of 96 train, 24 validation, and 24 development worlds;
- all 144 feature caches reused and none created;
- no output-directory overlap with either frozen control.

The original process interruption after iteration 4 did not alter the study.
All four checkpoint hashes, the implementation hash, the preregistration hash,
and the optimizer-bearing training fingerprint were verified before resuming
at iteration 5.

## Training trajectory

| Iteration | Train loss | Validation MAE | Validation target mean | Recorded inner-loop time |
|---:|---:|---:|---:|---:|
| 1 | 0.003746 | 0.052904 | 0.1020 | 1,112.5 s |
| 2 | 0.003425 | 0.056944 | 0.2227 | 1,115.7 s |
| 3 | 0.003968 | 0.062972 | 0.3379 | 1,098.2 s |
| 4 | 0.004980 | 0.068747 | 0.4524 | 1,203.5 s |
| 5 | 0.006325 | 0.080151 | 0.5537 | 1,263.1 s |
| 6 | 0.008603 | 0.098932 | 0.6365 | 1,103.4 s |
| 7 | 0.010097 | 0.108786 | 0.6611 | 1,104.9 s |
| 8 | 0.010869 | 0.111120 | 0.7008 | 1,111.0 s |

Summary-last ends with worse validation MAE than trimmed HRM (0.111120 versus
0.100561) despite nearly identical endpoint validation-target means (0.7008
versus 0.7029). The order change therefore does not stabilize the moving
self-bootstrap optimization. Its recorded inner-loop total is 9,112.2 seconds
(2.53 hours), versus 10,769.5 seconds for trimmed HRM; these implementation
timings exclude target construction and are not architecture-level speed claims.

## Preregistered development grid

All 24 paths are valid for every arm and cell.

| Iter. | Alpha | Summary-last exp. | Delta vs field HRM | 95% CI | Delta vs trimmed | 95% CI | Negative suites | Method | Readout |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 4 | 1.00 | 83.958 | +8.042 | [+0.042, +15.875] | -1.958 | [-5.750, +1.583] | 2/6 | fail | fail |
| 4 | 1.50 | 81.792 | +5.875 | [-2.917, +14.667] | -2.375 | [-6.583, +1.542] | 3/6 | fail | fail |
| 6 | 1.00 | 78.792 | +2.875 | [-5.250, +11.083] | -3.625 | [-7.375, -0.208] | 3/6 | fail | **pass** |
| 6 | 1.50 | 74.417 | -1.500 | [-10.250, +6.876] | -3.875 | [-8.125, -0.167] | 3/6 | fail | **pass** |
| 8 | 1.00 | 77.083 | +1.167 | [-6.083, +8.208] | +1.292 | [-2.208, +4.917] | 2/6 | fail | fail |
| 8 | 1.50 | 69.375 | -6.542 | [-13.251, -0.333] | +2.083 | [-1.376, +5.792] | 3/6 | fail | fail |

No cell passes the method gate. Two iteration-6 cells pass the direct readout
gate: summary-last beats trimmed HRM by 3.625-3.875 mean expansions with CI
upper endpoints just below zero and matched path quality. Neither transfers
that gain into a robust comparison with field HRM: both have only three
negative suite means, and their field-HRM confidence intervals cross zero.

This is the central nuance. Serialization order affects the recurrent model,
but the benefit is transient and insufficient.

## Fixed primary endpoint

At iteration 8 and alpha 1.50:

| Arm | Mean expansions | Mean cost ratio | Maximum cost ratio |
|---|---:|---:|---:|
| summary-last HRM | 69.375 | 1.019248 | 1.101261 |
| trimmed HRM | 67.292 | 1.025447 | 1.102426 |
| flat MLP | 69.417 | 1.013142 | 1.038347 |
| field HRM | 75.917 | 1.033480 | 1.263572 |

Paired expansion results are:

- summary-last minus field HRM: -6.542, 95% CI [-13.251, -0.333],
  14/0/10 wins/ties/losses;
- summary-last minus trimmed HRM: +2.083, 95% CI [-1.376, +5.792],
  9/2/13 wins/ties/losses;
- summary-last minus flat MLP: -0.042, 95% CI [-2.833, +3.043],
  11/3/10 wins/ties/losses.

The expansion comparison with flat is effectively unresolved, while the
flat-relative cost differences are +0.006106 in the mean and +0.062915 in the
maximum. Both exceed the locked margins. Summary-last improves path quality
relative to trimmed at the endpoint, but not enough to satisfy the stronger
matched-flat method contract.

### Fixed-cell suite behavior

| Suite | Summary-last | Trimmed | Flat | Field HRM | Summary-field |
|---|---:|---:|---:|---:|---:|
| bugtrap | 12.25 | 10.25 | 14.75 | 24.25 | -12.00 |
| maze | 47.25 | 41.75 | 45.25 | 77.00 | -29.75 |
| maze dense | 101.25 | 99.00 | 103.50 | 97.50 | +3.75 |
| rooms | 99.00 | 94.00 | 98.25 | 107.75 | -8.75 |
| rooms large | 42.75 | 39.50 | 38.00 | 39.50 | +3.25 |
| spiral | 113.75 | 119.25 | 116.75 | 109.50 | +4.25 |

The same heterogeneous structure remains: strong maze/bugtrap gains, a rooms
gain, and losses on dense maze, large rooms, and spiral.

## Mechanism diagnosis

### Integration remains functional

At iteration 8, mean rank correlation with graph cost-to-go changes as follows
after the common local Bellman backup:

| Family | Static Spearman | After local backup |
|---|---:|---:|
| summary-last HRM | 0.7813 | 0.8631 |
| trimmed HRM | 0.7493 | 0.8568 |
| flat MLP | 0.7906 | 0.8689 |

The local backup improves all three families substantially. The endpoint
summary-last arm also beats field HRM in pooled expansions. This is not a
general integration incompatibility.

### Readout order matters, but is not sufficient

The significant iteration-6 gains over trimmed HRM show that placing the
summary at the readout can improve the recurrent representation. The effect
does not persist to iteration 8 and never produces a cell that passes the
field-HRM suite/uncertainty contract. The failure cannot be reduced to the
position of one token.

### Optimization remains a live failure mode

Summary-last validation error accelerates late in the same moving-target loop,
ending 0.01056 above trimmed HRM. The candidate's transient search improvement
at iteration 6 despite worse MAE also shows that scalar regression error is an
imperfect proxy for ranking quality. Any next model should separate stationary
target learnability from self-bootstrap stability rather than simply tune on
this development block.

### The intended HRM substrate is still absent

Both recurrent controls reset their low/high states for every node example.
They perform hierarchy over a serialized static observation but carry no state
across a local Bellman trace, A* expansions, or planning time. C13-O does not
test persistent hierarchical computation.

## What should happen next

Do not retune C13-O and do not open its confirmation block. The smallest clean
next study is a gated persistent-state diagnostic, not another token-order
variant:

1. define an ordered planning episode whose state is reset once per world/query
   and carried only across causally observed local Bellman/search events;
2. keep inference inputs within the current-state boundary: goal/current
   geometry, bounded rays, one-hop actions, and already observed search events;
3. compare a matched zero-reset HRM with a carry-state HRM under identical
   parameters and compute;
4. add a stationary frozen-target control before the self-bootstrap arm, so a
   persistence effect is not confounded with the moving-target instability;
5. specify queue semantics explicitly, because a stateful scorer is a
   history-conditioned search policy rather than a static heuristic;
6. authorize a full development search only if carry state improves held-out
   ranking/next-expansion behavior without collapse, and retain the same
   field/flat path-quality and suite-balance gates.

State must **not** be carried across arbitrary node order, independent worlds,
or outer training iterations. That would manufacture sequence structure rather
than test planning-time memory.

## Verification and claim boundary

- focused C13-O tests: 9/9 passed;
- combined C13-N/C13-O focused tests: 15/15 passed;
- training checkpoints: 8/8;
- development rows: 504/504 (21 arms x 24 worlds);
- diagnostic rows: 216/216;
- duplicate two-pass comparisons: 216, zero mismatches;
- live-feature/cache mismatches: 0;
- integrity entries rehashed: 31/31, zero mismatches;
- confirmation artifacts: 0, as required after the failed gate;
- live C13-O worker after completion: none.

This remains a known-PRM observation-simulator result with absolute
coordinates. It is not a map-free claim, a formal search bound, a persistent
HRM result, a wall-clock advantage, or evidence about HRM architectures in
general.

