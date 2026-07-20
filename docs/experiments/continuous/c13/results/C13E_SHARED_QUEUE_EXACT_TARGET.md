# C13-E Shared-State Exact Rollout Target Gate

**Date:** 2026-07-17  
**Status:** Primary exact-target gate failed; learned providers remain blocked  
**Primary verdict:** `shared_queue_exact_rollout_gate_fail`

## 1. Artifacts

- [run manifest](../../../../../hrm-cloud/continuous_prm/runs/c13_shared_queue_rollout/manifest.json)
- [integrity hashes](../../../../../hrm-cloud/continuous_prm/runs/c13_shared_queue_rollout/integrity.json)
- [verification record](../../../../../hrm-cloud/continuous_prm/runs/c13_shared_queue_rollout/results/verification.json)
- [gate verdict](../../../../../hrm-cloud/continuous_prm/runs/c13_shared_queue_rollout/results/gate_verdict.json)
- [aggregate summary](../../../../../hrm-cloud/continuous_prm/runs/c13_shared_queue_rollout/results/shared_queue_exact_target_summary.csv)
- [per-world raw results](../../../../../hrm-cloud/continuous_prm/runs/c13_shared_queue_rollout/results/shared_queue_exact_target_raw.csv)
- [target diagnostics](../../../../../hrm-cloud/continuous_prm/runs/c13_shared_queue_rollout/results/shared_queue_exact_target_diagnostics.csv)
- [matched Euclidean baselines](../../../../../hrm-cloud/continuous_prm/runs/c13_shared_queue_rollout/results/shared_queue_exact_target_baselines.csv)
- [implementation](../../../../../hrm-cloud/continuous_prm/continuous_prm_c13_shared_queue_target.py)
- [focused tests](../../../../../hrm-cloud/continuous_prm/tests/test_c13_shared_queue_target.py)
- [passed oracle predecessor](C13D_SHARED_QUEUE_ORACLE.md)

## 2. Question and frozen intervention

C13-D establishes that the one-anchor/one-rank shared-state search has enough
oracle headroom: at `w=1.10`, privileged graph-distance ranking beats matched
Euclidean FOCAL on all six audit worlds. C13-E asks the next causal question:
can the permitted shortest-path-free target reproduce that gain before any
model approximation enters?

Only the rank vector changes. C13-E keeps the same:

- six frozen `C_hard_maze` audit worlds and 192-node roadmaps;
- shared `g`, parent, queue-update, and certificate implementation;
- consistent Euclidean proof anchor;
- search budget and widths `{1.05, 1.10, 1.25}`;
- matched Euclidean-ranked FOCAL baseline; and
- primary all-certified, zero-violation, five-of-six, negative-mean gate at
  `w=1.10`.

The new rank is exactly C13-B's replayed `rollout_exact` vector. A labeled node
uses the median cost of its successful ten deterministic fresh-start local
behavior rollouts. A node with no successful rollout uses C13-B's unchanged
deterministic Euclidean-plus-penalty fill. There is no training or model load.

### Provenance boundary

The rollout policy uses current goal geometry, current one-hop actions, and
visit memory accumulated only after each fresh start. Graph shortest-path
distance is not a feature, label, rank, or certificate input. It is read only
after search for target diagnostics, optimal-cost ratios, and bound checks.

## 3. Primary result

The preregistered `w=1.10` gate fails. All searches certify valid bounded
paths, but the exact target does not reduce expansions stably enough.

| Bound | Shared exact total | Rank exp. | Anchor exp. | Rank-choice rate | Euclid FOCAL | Mean delta | Wins / ties / losses | Shared oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.05 | 133.00 | 2.83 | 130.17 | 2.13% | 136.83 | -3.83 | 6 / 0 / 0 | 130.33 |
| **1.10** | **131.00** | **3.17** | **127.83** | **2.42%** | **129.67** | **+1.33** | **2 / 1 / 3** | **122.83** |
| 1.25 | 126.50 | 5.00 | 121.50 | 3.95% | 127.67 | -1.17 | 3 / 0 / 3 | 99.83 |

The `w=1.05` arm passes its own width-specific gate, but the locked primary
width does not. The `w=1.25` mean is slightly favorable but fails the required
five-of-six stability test. These non-monotone width results cannot replace
the declared primary outcome.

### Primary per-world ledger

| World | Exact total | Rank exp. | Anchor exp. | Rank-choice rate | Euclid FOCAL | Delta | Shared oracle | Exact minus oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 142 | 4 | 138 | 2.82% | 137 | +5 | 131 | +11 |
| 1 | 134 | 2 | 132 | 1.49% | 129 | +5 | 120 | +14 |
| 2 | 138 | 4 | 134 | 2.90% | 138 | 0 | 125 | +13 |
| 3 | 141 | 2 | 139 | 1.42% | 144 | -3 | 137 | +4 |
| 4 | 112 | 4 | 108 | 3.57% | 114 | -2 | 111 | +1 |
| 5 | 119 | 3 | 116 | 2.52% | 116 | +3 | 113 | +6 |

Mean post-hoc cost ratio is `1.0015` and the maximum is `1.0086`. Thus this is
an efficiency/stability failure, not a safety failure.

## 4. Multi-angle diagnosis

### Integration

The old independent-wrapper problem is repaired. At the primary width, the
shared exact arm averages `131.00` expansions versus `372.50` for C13-C's
separate exact-incumbent plus fresh-certificate wrapper, saving `241.50`.
Shared state therefore removes the duplication failure even for this poor
incumbent source.

The remaining `+8.17` expansion gap between exact rollout (`131.00`) and the
shared oracle (`122.83`) is measured under the same search implementation. It
cannot be assigned to the former independent-certifier architecture.

### Target ordering and calibration

The replayed target has useful but imperfect ordering: mean per-world
Spearman correlation with graph cost-to-go is `0.760`. Its absolute scale is
far from the anchor, however. Across worlds, the median node-wise
rollout/oracle ratio averages `3.48`, and the start-state ratio averages
`4.23`.

That scale matters in this integration. The rank queue is eligible only when
its minimum `g+h_rollout` is at most `w` times the Euclidean-anchor lower bound.
At `w=1.10`, exact rollout is selected on only `2.42%` of eligibility checks,
for `3.17` rank expansions per world. Those few expansions do not repay their
cost: the anchor performs `127.83` expansions, compared with `117.83` in the
oracle arm.

This points to a target/alignment and calibration interaction. The behavior
return estimates how expensive the exploratory local policy is, not the
remaining cost of an efficient graph path. A monotone rank can contain useful
direction while its inflated cost scale makes it nearly unusable in an
anchor-eligibility test.

### Representation

Representation is not tested here. The rank is the replayed target itself;
there is no neural approximation, padding, readout, capacity, or optimization
error. Because the exact target misses the primary gate, testing frozen HRM,
MLP, or ON-LSTM providers now would conflate representation error with an
already failing target.

### Missing labels or missing information

Raw rollout medians cover `97.22%` of audit nodes. Three worlds have 100%
coverage, yet two of those three lose the primary baseline. Conversely, one
world with eight filled nodes wins. Penalty filling can hurt individual ranks,
but missing-label coverage is not the dominant six-world explanation.

The broader information limit remains plausible: bounded local observations
cannot fully identify distant maze detours, and the behavior policy adds its
own exploration inefficiency. This run does not separate those two sources.
It does show that more rollout-model capacity alone is not the next justified
move.

## 5. Decision

Do not test the frozen learned providers and do not launch the 192/211-node or
multi-suite runs. The exact permitted target has not cleared the shared-search
gate.

The next causal study should remain training-free and split scale from
ordering before changing representations:

1. add a same-search Euclidean-rank control to measure what the shared
   certificate does without rollout information;
2. test preregistered monotone Euclidean/rollout residual blends under the same
   certificate, using fixed coefficients that do not read graph shortest-path
   values; and
3. only if a scale-preserving or calibrated version clears the unchanged gate,
   decide whether to train toward that objective. Otherwise replace the
   behavior-cost target with a more search-aligned shortest-path-free objective
   such as a local Bellman/TD or pairwise ranking target.

This is a target/calibration gate, not evidence that HRM or another
representation is intrinsically incapable.

## 6. Verification

- audit replay: `1,152` rows, zero mismatches, maximum rollout-median delta `0`;
- raw output: `18/18` expected world/bound rows, zero duplicate keys;
- certification, path, accounting, post-hoc bound, and anchor-lower-bound
  failures: `0`;
- anchor consistency violations and states expanded more than twice: `0`;
- training and model loading: none;
- focused shared-search plus exact-target tests: `15 passed`;
- full C13 regression suite: `40 passed`.

The recurring pytest-cache warning is an environment permission warning after
the tests pass; it does not affect test execution.

Reproduce from the repository root:

```powershell
python hrm-cloud/continuous_prm/continuous_prm_c13_shared_queue_target.py
$c13Tests = Get-ChildItem hrm-cloud/continuous_prm/tests/test_c13*.py | Select-Object -ExpandProperty FullName
python -m pytest $c13Tests -q
```
