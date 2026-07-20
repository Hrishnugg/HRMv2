# C13-C Certified Incumbent Integration Gate

**Date:** 2026-07-17  
**Status:** Completed; simple independent-certifier design rejected at the primary bound  
**Primary verdict:** `reject_simple_certifier_no_oracle_headroom_at_primary_bound`

## 1. Artifacts

- [run manifest](../../../../../hrm-cloud/continuous_prm/runs/c13_certified_search/manifest.json)
- [integrity hashes](../../../../../hrm-cloud/continuous_prm/runs/c13_certified_search/integrity.json)
- [verification record](../../../../../hrm-cloud/continuous_prm/runs/c13_certified_search/results/verification.json)
- [gate verdict](../../../../../hrm-cloud/continuous_prm/runs/c13_certified_search/results/gate_verdict.json)
- [aggregate summary](../../../../../hrm-cloud/continuous_prm/runs/c13_certified_search/results/certified_search_summary.csv)
- [per-world raw results](../../../../../hrm-cloud/continuous_prm/runs/c13_certified_search/results/certified_search_raw.csv)
- [matched Euclidean baselines](../../../../../hrm-cloud/continuous_prm/runs/c13_certified_search/results/certified_search_baselines.csv)
- [implementation](../../../../../hrm-cloud/continuous_prm/continuous_prm_c13_certified_search.py)
- [focused tests](../../../../../hrm-cloud/continuous_prm/tests/test_c13_certified_search.py)
- [source C13-B diagnostic](C13B_IDENTIFIABILITY_STUDY.md)

## 2. Question and locked design

C13-B found useful but unsafe primary-A* ordering. C13-C tests the smallest
wrapper that could convert that ordering into a certified result without
retraining:

1. Phase 1 runs A* under an arbitrary finite rank, permits reopening, and
   stops at the first feasible goal to obtain an incumbent.
2. Phase 2 starts a completely fresh Euclidean A*. Its Euclidean heuristic is
   checked for consistency on every world. It stops only when

   `incumbent_cost <= w * min_OPEN(g + h_euclid)`

   or when the anchor search itself pops the optimal goal.
3. Total expansions are phase 1 plus phase 2. Repeated work is counted rather
   than credited away.

The study replays the six frozen `C_hard_maze` audit worlds from C13-B, each
with 192 roadmap nodes. It performs no training. Providers are the privileged
graph-distance oracle ceiling, the exact saved rollout statistic, and frozen
flat-MLP, padded-HRM, and trimmed-ON-LSTM checkpoints. Bounds are swept at
`w in {1.05, 1.10, 1.25}`.

The primary `w=1.10` gate requires all of the following:

- certification on all six worlds;
- zero post-hoc bound violations;
- fewer total expansions than matched Euclidean-ranked FOCAL on at least five
  of six worlds; and
- a strictly negative mean total-expansion delta.

### Provenance boundary

Shortest-path distance is not a learned/rollout feature, label, or
certification input. It is used post hoc to validate bounds and path ratios.
The explicitly named `oracle_eval_only` control additionally uses graph
distance as a privileged phase-1 rank, solely to determine whether this
integration has enough headroom to evaluate any realizable target or model.

## 3. Primary result at `w=1.10`

The matched Euclidean-ranked FOCAL baseline averages `129.67` expansions.
Every arm certifies all six returned paths and produces zero bound violations,
but every arm loses the efficiency comparison on all six worlds.

| Phase-1 provider | Phase-1 exp. | Reopens | Certificate exp. | Total exp. | Delta vs Euclid FOCAL | Wins | Mean / max final cost ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle, evaluation only | 23.33 | 0.00 | 117.83 | 141.17 | +11.50 | 0/6 | 1.000 / 1.000 |
| Exact rollout statistic | 238.67 | 133.67 | 133.83 | 372.50 | +242.83 | 0/6 | 1.027 / 1.084 |
| Flat MLP | 134.67 | 57.17 | 136.83 | 271.50 | +141.83 | 0/6 | 1.027 / 1.081 |
| HRM, padded readout | 110.50 | 35.50 | 135.67 | 246.17 | +116.50 | 0/6 | 1.023 / 1.086 |
| ON-LSTM, trimmed readout | 120.50 | 44.50 | 121.50 | 242.00 | +112.33 | 0/6 | 1.028 / 1.072 |

The oracle is the decisive control. Its exact optimal incumbent reduces the
fresh certifier by only `7-18` expansions per world, while obtaining that
incumbent costs `19-27` expansions. The phase-1 cost exceeds the certification
saving on every world:

| World | Oracle phase 1 | Fresh certificate | Total | Euclid FOCAL | Delta |
|---:|---:|---:|---:|---:|---:|
| 0 | 27 | 124 | 151 | 137 | +14 |
| 1 | 25 | 113 | 138 | 129 | +9 |
| 2 | 21 | 120 | 141 | 138 | +3 |
| 3 | 25 | 134 | 159 | 144 | +15 |
| 4 | 23 | 107 | 130 | 114 | +16 |
| 5 | 19 | 109 | 128 | 116 | +12 |

For another scale check, ordinary Euclidean A* averages `139.17` expansions.
Even the oracle wrapper averages `141.17`, so the duplicated design does not
beat ordinary A* either.

## 4. Bound sweep

| Bound | Oracle total exp. | Euclid FOCAL exp. | Mean delta | World wins | Gate |
|---:|---:|---:|---:|---:|---|
| 1.05 | 149.83 | 136.83 | +13.00 | 0/6 | Fail |
| 1.10 | 141.17 | 129.67 | +11.50 | 0/6 | Fail |
| 1.25 | 114.50 | 127.67 | -13.17 | 4/6 | Fail |

The looser `w=1.25` setting creates average oracle headroom, but it is not
stable enough for the declared five-of-six gate. This is useful directional
evidence for a better shared integration, not grounds to move the primary
bound or call the current wrapper successful.

## 5. Causal interpretation from several angles

### Integration

This is the immediate failure. The fresh certifier discards all phase-1 search
state. At `w=1.10`, even an exact optimal incumbent cannot save enough anchor
expansions to repay the separate search that found it. Since the privileged
oracle fails first, learned-model or target quality cannot explain the primary
gate rejection.

### Target

The exact rollout statistic is additionally ill-suited to unrestricted
primary A*: it averages `238.67` phase-1 expansions and `133.67` reopens. That
is consistent with the independent C13-B finding that behavior-policy return
is noisy, inefficient relative to the graph oracle, and not a useful bounded
search rank. It is a real secondary issue, but it is not the cause of the
oracle failure.

### Representation

This run cannot cleanly adjudicate representation at the primary bound because
the integration ceiling fails. The frozen learned ranks are substantially
cheaper than the exact rollout rank in phase 1, and all produce valid bounded
paths, but none can overcome the duplicated certificate work. C13-B remains
the relevant representation result: local signal is learnable, while padding
and readout defects are architecture dependent.

### Missing ingredient

The missing mechanism is shared proof/search work. A useful bounded planner
must let the learned queue find incumbents while a consistent anchor queue
maintains the lower bound over shared `g` values, duplicate detection, and
expansion state. A completely independent second search spends almost the
same work as the baseline and then adds phase 1 on top.

Reopening also exposes severe inconsistency in the non-oracle ranks. A
feasibility-only first phase could omit reopening, as the earlier C13-B
primary-A* diagnostic did, but that cannot rescue this gate: the oracle has
zero reopens and still loses all six primary comparisons.

## 6. Decision and next gate

Reject this simple independent-certifier design at `w=1.10`. Do not retrain the
same models and do not launch the final multi-suite density run.

The next integration-only gate should be a shared-state anchored multi-queue
search:

1. one consistent Euclidean anchor queue owns the proof lower bound;
2. one inadmissible learned/oracle queue proposes expansions and incumbents;
3. both queues share `g` values, closed/reopen accounting, and duplicate work;
4. the same `w=1.10`, all-certified, zero-violation, five-of-six, negative-mean
   gate is applied first to the oracle control;
5. only after the oracle ceiling passes should exact rollout and frozen learned
   ranks be evaluated; and
6. no new target training is justified until the exact permitted target passes
   that integration gate.

## 7. Verification

- audit replay: `1,152` rows, zero mismatches, maximum rollout-median delta `0`;
- raw output: `90/90` expected provider/world/bound rows, zero duplicates;
- phase-1 failures: `0`;
- certification failures: `0`;
- post-hoc bound violations: `0`;
- maximum Euclidean-anchor consistency violation: `0` across all six worlds;
- focused C13-C/C13-B tests: `12 passed`; full C13 regression suite: `25 passed`.

Reproduce from the repository root:

```powershell
python hrm-cloud/continuous_prm/continuous_prm_c13_certified_search.py
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_state_heuristic.py hrm-cloud/continuous_prm/tests/test_c13_td_ranker.py hrm-cloud/continuous_prm/tests/test_c13_identifiability.py hrm-cloud/continuous_prm/tests/test_c13_certified_search.py -q
```

## 8. C13-D follow-up

The shared-state oracle gate is complete. See
[C13D_SHARED_QUEUE_ORACLE.md](C13D_SHARED_QUEUE_ORACLE.md).

Sharing `g`, parents, and queue updates repairs the independent wrapper's
duplication failure. At `w=1.10`, the oracle arm averages `122.83`
expansions versus `129.67` for matched FOCAL and wins all six worlds. This
authorizes the exact frozen rollout statistic next, not learned-model claims.

## 9. C13-E follow-up

The exact-target shared-queue gate is complete. See
[C13E_SHARED_QUEUE_EXACT_TARGET.md](C13E_SHARED_QUEUE_EXACT_TARGET.md).

Shared state reduces the exact rollout arm from `372.50` expansions here to
`131.00`, confirming that duplicated certification was repaired. The exact
rank nevertheless fails the primary matched-FOCAL gate, so target calibration
