# C13-D Shared-State Oracle Integration Gate

**Date:** 2026-07-17  
**Status:** Oracle integration ceiling passed; C13-E exact target subsequently failed  
**Primary verdict:** `shared_queue_oracle_gate_pass`

## 1. Artifacts

- [run manifest](../../../../../hrm-cloud/continuous_prm/runs/c13_shared_queue_oracle/manifest.json)
- [integrity hashes](../../../../../hrm-cloud/continuous_prm/runs/c13_shared_queue_oracle/integrity.json)
- [verification record](../../../../../hrm-cloud/continuous_prm/runs/c13_shared_queue_oracle/results/verification.json)
- [gate verdict](../../../../../hrm-cloud/continuous_prm/runs/c13_shared_queue_oracle/results/gate_verdict.json)
- [aggregate summary](../../../../../hrm-cloud/continuous_prm/runs/c13_shared_queue_oracle/results/shared_queue_oracle_summary.csv)
- [per-world raw results](../../../../../hrm-cloud/continuous_prm/runs/c13_shared_queue_oracle/results/shared_queue_oracle_raw.csv)
- [matched Euclidean baselines](../../../../../hrm-cloud/continuous_prm/runs/c13_shared_queue_oracle/results/shared_queue_oracle_baselines.csv)
- [implementation](../../../../../hrm-cloud/continuous_prm/continuous_prm_c13_shared_queue.py)
- [focused tests](../../../../../hrm-cloud/continuous_prm/tests/test_c13_shared_queue.py)
- [rejected independent-certifier predecessor](C13C_CERTIFIED_SEARCH.md)

## 2. Question and locked gate

C13-C proved that an inadmissible incumbent can be certified correctly, but a
completely fresh Euclidean proof search duplicated too much work. At the
primary `w=1.10`, even the privileged optimal oracle incumbent lost all six
matched comparisons.

C13-D changes only the integration. It performs no training and reuses the
same six frozen `C_hard_maze` audit worlds with 192 roadmap nodes. The first
test is deliberately limited to the privileged graph-distance oracle rank.
The exact rollout statistic and learned checkpoints remain untouched until the
integration ceiling passes.

The primary gate is unchanged:

- `w=1.10`;
- certification on all six worlds;
- zero path, anchor-lower-bound, and post-hoc cost-bound failures;
- fewer total expansions than matched Euclidean-ranked FOCAL on at least five
  of six worlds; and
- a strictly negative mean expansion delta.

## 3. Shared-state search

The implementation is a one-anchor/one-rank shared-path MHA-style search with
an uninflated anchor (`w1=1`):

1. Euclidean `g+h` and oracle `g+h*` queues share one `g` value and parent per
   state.
2. The rank queue expands while its minimum key is at most
   `w * min_anchor`; otherwise the Euclidean anchor queue expands.
3. Expanding a state invalidates its current label in both queues. A later
   better path can insert it into the queue in which it has not yet expanded.
4. A generated goal path is a feasible incumbent, but it is returned only
   when `incumbent <= w * min_anchor`.
5. Every queue expansion and any cross-queue duplicate expansion is counted.

This follows the shared-`g` and anchor-eligibility structure of
[Shared Multi-Heuristic A*](https://cdn.aaai.org/ojs/18306/18306-77-21822-1-2-20210717.pdf),
specialized to one inadmissible queue. The direct incumbent/anchor termination
test supplies the requested bound.

### Provenance boundary

Graph shortest-path distance is used here only as the explicitly privileged
oracle rank and for post-hoc evaluation. The certificate itself depends only
on the consistent Euclidean anchor. No shortest-path value is a model feature,
training label, or learned-provider input in this experiment.

## 4. Result

The oracle integration ceiling passes at every tested bound, including all six
worlds at the primary `w=1.10`.

| Bound | Shared total | Rank exp. | Anchor exp. | Cross-queue duplicates | Euclid FOCAL | Mean delta | Wins | Independent wrapper | Work saved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.05 | 130.33 | 3.83 | 126.50 | 0.00 | 136.83 | -6.50 | 6/6 | 149.83 | 19.50 |
| 1.10 | 122.83 | 5.00 | 117.83 | 0.00 | 129.67 | -6.83 | 6/6 | 141.17 | 18.33 |
| 1.25 | 99.83 | 8.67 | 91.17 | 0.00 | 127.67 | -27.83 | 6/6 | 114.50 | 14.67 |

All returned paths are optimal in this privileged oracle arm: mean and maximum
post-hoc cost ratio are `1.000` at every tested bound.

### Primary per-world ledger

| World | Rank exp. | Anchor exp. | Total | Euclid FOCAL | Delta | Independent wrapper | Work saved |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7 | 124 | 131 | 137 | -6 | 151 | 20 |
| 1 | 7 | 113 | 120 | 129 | -9 | 138 | 18 |
| 2 | 5 | 120 | 125 | 138 | -13 | 141 | 16 |
| 3 | 3 | 134 | 137 | 144 | -7 | 159 | 22 |
| 4 | 4 | 107 | 111 | 114 | -3 | 130 | 19 |
| 5 | 4 | 109 | 113 | 116 | -3 | 128 | 15 |

## 5. Interpretation

The C13-C failure was genuinely an integration failure. Its primary oracle
wrapper paid `23.33` expansions for a separate incumbent search and then
`117.83` for the fresh proof search. C13-D retains the same `117.83` anchor
work but needs only `5.00` oracle-queue expansions because the queues share
paths and search state. The resulting `122.83` total beats the matched baseline
by `6.83` expansions and the independent wrapper by `18.33`.

No audit state is expanded by both queues, and there are no improvements after
an expansion in the oracle run. The gain is therefore not produced by hiding
duplicate work. The anchor performs most of the lower-bound progress; once the
oracle queue becomes eligible, a small number of rank-guided expansions closes
the remaining optimal path.

This result repairs only the integration ceiling. It does **not** show that the
shortest-path-free rollout target is useful, that a learned approximation is
good enough, or that HRM is preferable to another representation.

## 6. Decision

The shared-state integration is authorized. Stop the oracle study here and run
the exact frozen rollout statistic next under the identical search and gate:

1. no retraining or target changes;
2. replace only the privileged oracle rank with the saved exact rollout rank;
3. keep `w=1.10`, all certification checks, five-of-six wins, and negative mean
   delta unchanged; and
4. test frozen learned providers only if the exact target itself passes.

If the exact rollout target fails, the next blocker is target alignment or
calibration rather than integration. If it passes, learned representation can
finally be isolated cleanly.

## 7. Verification

- audit replay: `1,152` rows, zero mismatches, maximum rollout-median delta `0`;
- raw output: `18/18` expected world/bound rows, zero duplicate keys;
- certification, path, accounting, and post-hoc bound failures: `0`;
- anchor lower-bound failures and consistency violations: `0`;
- states expanded more than twice: `0`;
- targeted shared-search tests: `9 passed`, including 60 random graph
  certificates; full C13 regression suite: `34 passed`.

Reproduce from the repository root:

```powershell
python hrm-cloud/continuous_prm/continuous_prm_c13_shared_queue.py
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_shared_queue.py -q
```

## 8. C13-E follow-up

The authorized exact-target substitution is complete. See
[C13E_SHARED_QUEUE_EXACT_TARGET.md](C13E_SHARED_QUEUE_EXACT_TARGET.md).

The shared search remains proof-correct, but the exact rollout rank fails the
locked `w=1.10` gate: `131.00` versus `129.67` mean expansions, with 2
wins, 1 tie, and 3 losses. Frozen learned providers are not authorized. The
next blocker is target alignment/calibration rather than shared-state
