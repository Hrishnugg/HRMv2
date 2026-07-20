# C13-N result: HRM substitution in the successful local-backup method

**Date:** 2026-07-17  
**Preregistered design:** [2026-07-17-c13n-hrm-substitution.md](../design/2026-07-17-c13n-hrm-substitution.md)  
**Verdict:** `hrm_substitution_development_gate_failed`  
**Untouched confirmation:** not authorized and not run

## Bottom line

The method is mechanically compatible with HRM, and the fixed HRM endpoint
contains useful search signal. At iteration 8 and alpha 1.50 it reduced mean
development expansions from 75.917 for field HRM to 67.292, a paired delta of
-8.625 with bootstrap 95% CI [-16.667, -1.208].

That is not enough to say that HRM preserves the C13-M result. Only three of
six suite means were strictly negative, below the preregistered four-suite
requirement. The HRM result also used materially worse paths than the matched
flat MLP, and HRM-versus-MLP expansion uncertainty crossed zero. No candidate
passed all gates, so the fresh 144-world confirmation block remained untouched.

The claim-safe conclusion is:

> The local Bellman integration can consume a trimmed-HRM value model, and the
> fixed endpoint was promising in pooled development results, but this literal
> architecture substitution did not establish robust, matched-quality HRM
> compatibility or HRM superiority.

## Frozen experiment

C13-N reused the exact C13-J artifacts:

- 96 training worlds, 32 each from maze, rooms, and spiral;
- 24 validation worlds, 8 per training suite;
- 24 development worlds, 4 per suite across all six suites;
- all 144 saved feature caches, with no cache regeneration;
- 192-node, k=7 PRMs;
- one summary token, 32 angular ray tokens, and up to 24 one-hop action tokens;
- the same eight outer LHBL iterations, five inner epochs, optimizer, seed,
  hidden width, target, and local radius;
- the same one-radius-0.20 Bellman backup and exact no-reopen A* integration.

The sole intended model change was:

`flat_mlp -> hrm_trimmed`

The HRM was `DeepSapientHRMBackbone(hidden=64, k_step=2, heads=4, layers=1)`
with the final state of the true-length token sequence as its readout.

The source audit checked 171 frozen C13-J artifacts. It replayed 96/24/24
worlds, reused 144/144 caches, and found zero hash, seed-overlap, or feature
mismatches.

## Training behavior

| Iteration | Train loss | Validation MAE | Train target mean | Recorded train time |
|---:|---:|---:|---:|---:|
| 1 | 0.00367 | 0.05099 | 0.1051 | 1,313.0 s |
| 2 | 0.00348 | 0.05673 | 0.2210 | 1,247.8 s |
| 3 | 0.00416 | 0.06061 | 0.3373 | 1,368.9 s |
| 4 | 0.00518 | 0.06821 | 0.4468 | 1,276.7 s |
| 5 | 0.00631 | 0.07346 | 0.5391 | 1,407.4 s |
| 6 | 0.00830 | 0.09263 | 0.6178 | 1,409.1 s |
| 7 | 0.00923 | 0.09342 | 0.6566 | 1,384.4 s |
| 8 | 0.00952 | 0.10056 | 0.7058 | 1,362.1 s |

The moving self-bootstrap target grows across iterations and validation error
degrades. This drift is not unique to HRM: the frozen flat run's target mean
also grew from 0.1051 to 0.7339 and its endpoint validation MAE was 0.09440.
HRM tracked the same moving-target process somewhat less accurately at the
endpoint, so the result does not support blaming the training target alone.

## Preregistered development grid

All 24 paths were valid for every cell.

| Iteration | Alpha | HRM exp. | Delta vs field HRM | 95% CI | Negative suites | Gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 4 | 1.00 | 85.917 | +10.000 | [+2.875, +17.250] | 2/6 | fail |
| 4 | 1.50 | 84.167 | +8.250 | [+0.833, +15.875] | 2/6 | fail |
| 6 | 1.00 | 82.417 | +6.500 | [-2.042, +15.042] | 2/6 | fail |
| 6 | 1.50 | 78.292 | +2.375 | [-6.458, +10.542] | 2/6 | fail |
| 8 | 1.00 | 75.792 | -0.125 | [-8.208, +7.750] | 3/6 | fail |
| 8 | 1.50 | 67.292 | -8.625 | [-16.667, -1.208] | 3/6 | fail |

The fixed C13-M substitution cell was iteration 8, alpha 1.50. It passed four
of the five field-comparator conditions:

- all 24 HRM paths valid: pass;
- pooled expansion CI upper endpoint below zero: pass;
- mean path-cost ratio within 0.005 of field HRM: pass;
- maximum path-cost ratio within 0.02 of field HRM: pass;
- strictly negative mean expansion delta in at least four suites: **fail
  (3/6)**.

### Fixed-cell suite behavior

| Suite | HRM exp. | Flat exp. | Field HRM exp. | HRM-field | HRM-flat |
|---|---:|---:|---:|---:|---:|
| bugtrap | 10.25 | 14.75 | 24.25 | -14.00 | -4.50 |
| maze | 41.75 | 45.25 | 77.00 | -35.25 | -3.50 |
| maze dense | 99.00 | 103.50 | 97.50 | +1.50 | -4.50 |
| rooms | 94.00 | 98.25 | 107.75 | -13.75 | -4.25 |
| rooms large | 39.50 | 38.00 | 39.50 | 0.00 | +1.50 |
| spiral | 119.25 | 116.75 | 109.50 | +9.75 | +2.50 |

The gain is therefore not a single pooled-statistic artifact, but it is
heterogeneous: three suite wins, one tie, and two suite losses against field
HRM.

## HRM versus the matched flat MLP

At iteration 8 and alpha 1.50:

- HRM: 67.292 mean expansions;
- flat MLP: 69.417 mean expansions;
- paired HRM-minus-flat delta: -2.125;
- bootstrap 95% CI: [-5.833, +1.792];
- per-world wins/ties/losses: 11/3/10.

The expansion estimate is inconclusive. Path quality favors the flat model:

| Metric | HRM | Flat MLP | HRM-flat |
|---|---:|---:|---:|
| Mean graph-optimal cost ratio | 1.025447 | 1.013142 | +0.012305 |
| Maximum graph-optimal cost ratio | 1.102426 | 1.038347 | +0.064079 |

Both differences exceed the preregistered architecture non-inferiority
margins of +0.005 mean and +0.02 maximum. The architecture-win condition
therefore fails independently of the field-HRM suite gate.

## What this says about the failure mode

### 1. It is not simply an integration incompatibility

The same local Bellman backup raised endpoint rank correlation for HRM from
0.7493 to 0.8568, and the fixed HRM cell beat field HRM in pooled expansions
with a confidence interval below zero. The integration path is doing useful
work with HRM output.

### 2. The raw representation is not simply missing all necessary information

HRM and the successful flat model received byte-identical features and the
same targets. The flat model's success shows that the local observation
contains enough signal for this development regime. A representation
limitation may still interact with HRM, but it is not an information-absence
explanation by itself.

### 3. The current sequence is poorly aligned with a final-state recurrent readout

The token order is:

1. summary and goal/current-state token;
2. 32 rays in angular order;
3. action tokens sorted by edge length and node id.

`hrm_trimmed` returns the final recurrent state, whose last input is the last
valid action token. Sequence length also changes with node degree. This gives
the recurrent model an artificial causal order across heterogeneous token
types and places the global summary farthest from the readout. The flat MLP
does not have to compress the same information through that ordering.

This is an architecture-representation alignment issue: the information is
present, but the chosen serialization and readout may make it harder for HRM
to use consistently across suites.

### 4. The implementation does not give HRM persistent planning state

Every node example starts the HRM low- and high-level states at zero. Its
hierarchy runs across tokens inside one static observation; it is not carried
across Bellman iterations, A* expansions, or planning time. If HRM's intended
advantage is iterative hierarchical computation, that substrate is absent
from this literal substitution.

### 5. The identical optimization schedule is a clean control, not an HRM optimum

The locked schedule isolates model family, but it need not be equally suitable
for both architectures. HRM has slightly fewer parameters (119,105 versus
129,345) yet is much harder to optimize in this implementation.

Recorded inner-loop training time was 10,769.5 seconds (2.99 hours) for HRM
versus 13.56 seconds for the frozen flat run, about 794x. On CPU development
inference over one 192-node world, iteration-8 HRM averaged 0.2202 seconds
versus 0.000531 seconds for flat, about 415x. These are implementation-specific
measurements, not general architecture benchmarks, but they rule out a
wall-clock benefit for this substitution.

## Recommended next test

Do not reinterpret or retune C13-N. If HRM is pursued, the smallest clean next
study is an architecture-alignment diagnostic, not another confirmation seed:

1. keep the same frozen cohorts, target, and local Bellman integration;
2. compare the current final-state `hrm_trimmed` control with a
   summary-last/type-aware HRM readout;
3. keep ray and action blocks explicit rather than treating the whole static
   observation as one causal token stream;
4. separately test a persistent HRM state across Bellman/search steps if the
   intended claim concerns hierarchical iterative reasoning;
5. preregister any HRM-specific optimization changes and treat them as a new
   method, not an architecture-only substitution.

A fresh confirmation block should be opened only after one such development
method passes the same suite-balance and matched-quality gates.

## Integrity and claim boundary

- focused tests: 12/12 passed;
- C13-N integrity entries rehashed: 21/21, zero mismatches;
- development rows: 360/360;
- diagnostics rows: 144/144;
- feature mismatches: 0;
- training checkpoints: 8/8;
- confirmation artifacts: 0, as required after the failed gate;
- C13-N worker: exited.

This is a known-PRM observation-simulator result with absolute coordinates.
It is not a map-free claim, a formal search bound, a fresh-confirmation result,
or evidence about HRM architectures in general.
