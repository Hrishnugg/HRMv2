# C13-B Identifiability and Integration Study

**Status:** completed one-suite diagnostic; representation signal found, current planning gate failed  
**Date:** 2026-07-16  
**Suite:** `C_hard_maze`  
**Roadmap:** 192 nodes, `k=7`

Raw artifacts:

- [study manifest](../../../../../hrm-cloud/continuous_prm/runs/c13_identifiability/manifest.json)
- [target, aliasing, and padding diagnostics](../../../../../hrm-cloud/continuous_prm/runs/c13_identifiability/results/diagnostics.json)
- [representation controls](../../../../../hrm-cloud/continuous_prm/runs/c13_identifiability/results/linear_representation_controls.csv)
- [model learning curves](../../../../../hrm-cloud/continuous_prm/runs/c13_identifiability/results/model_learning_curve.csv)
- [per-node target audit](../../../../../hrm-cloud/continuous_prm/runs/c13_identifiability/results/target_reliability_raw.csv)
- [raw integration sweep](../../../../../hrm-cloud/continuous_prm/runs/c13_identifiability/results/integration_raw.csv)
- [integration summary](../../../../../hrm-cloud/continuous_prm/runs/c13_identifiability/results/integration_summary.csv)
- [study implementation](../../../../../hrm-cloud/continuous_prm/continuous_prm_c13_identifiability.py)
- [focused tests](../../../../../hrm-cloud/continuous_prm/tests/test_c13_identifiability.py)

## Executive verdict

The original C13-B failure is **primarily an integration failure at the declared `w=1.10` FOCAL setting**, but it is not only an integration failure.

The bounded current observation contains learnable information: held-out target correlation reaches `0.750` for HRM, `0.675` for a flat MLP, and `0.729` for a five-neighbor nonparametric control. The same learned estimates also cut ordinary A* expansions by roughly 45–64 when used as primary heuristics. They are therefore not empty or random signals.

FOCAL mostly prevents that signal from affecting the search at `w=1.10`. HRM `g+h_hat` is `+3.5` expansions worse than the matched Euclidean-ranked FOCAL control, the exact ten-rollout target is `+6.3` worse, and even exact Dijkstra `g+h*` is `+1.3` worse. Exact `h*` used as the primary A* heuristic instead reduces mean expansions from `139.2` to `23.3`. The information exists, but the original secondary-key insertion point exposes almost none of its headroom.

Two independent blockers remain:

1. **The behavior-return target is noisy and decision-misaligned.** Its median cost is `3.30x` the graph oracle, and the exact ten-rollout target usually makes FOCAL worse. More training cannot fix a target whose exact values do not improve the intended search.
2. **The sequence representation is brittle.** A padded last-state ON-LSTM has held-out correlation `0.006`, while a separately trained trimmed version reaches `0.489`. HRM benefits from padding because it can exploit sequence length as an implicit degree/cardinality cue, but that is an accidental encoding rather than a sound mask contract.

The result does **not** support declaring the target unidentifiable from bounded observations. It supports a narrower conclusion: local observations identify a useful but incomplete proxy, while global maze detours remain aliased and the present target/integration pair does not convert that proxy into a safe planning gain.

## 1. Protocol

The diagnostic deliberately separates target, representation, and integration questions.

- Training dataset: 12 worlds and 2,261 labeled nodes, each from three independent fresh-start local-policy rollouts; model fitting uses 1,811 nodes and the same-world diagnostic holds out 450.
- Held-out validation: four disjoint worlds, 756 labeled nodes.
- Search audit: six further disjoint worlds, 1,152 nodes, with ten independent rollouts per node.
- Features: fixed 25-token sequence containing one summary token, eight ray tokens, up to 16 one-hop action tokens, and trailing zero padding.
- Models: flat MLP, masked pooling, padded/trimmed/summary-last HRM, and padded/trimmed ON-LSTM.
- Controls: ridge and five-neighbor regressors over summary, ray, action, compact, and full feature views.
- Integration: Euclidean-anchored FOCAL at `w={1.00,1.02,1.05,1.10,1.25,1.50}` with `h`, `g+h`, and residual secondary keys; ordinary A* with each direct estimate as a diagnostic primary heuristic.
- Evaluation-only ceilings: exact graph distance in FOCAL and A*. It is never a training feature or label.

All 1,120 audit nodes connected to the goal received a successful rollout label. The 32 disconnected nodes remain in the raw table but are excluded from connected-node target statistics.

## 2. Is the target reliable and search-aligned?

The ten-rollout audit gives a mixed answer.

| Diagnostic | Result |
| --- | ---: |
| Split-half return Pearson / Spearman | `0.646 / 0.831` |
| Rollout cost vs graph-oracle cost Pearson / Spearman | `0.644 / 0.767` |
| Rollout residual vs oracle residual Pearson / Spearman | `0.629 / 0.769` |
| Median / p90 rollout-to-oracle cost ratio | `3.30 / 8.23` |
| Median / p90 within-node rollout IQR | `3.02 / 11.39` |

The rank ordering is moderately reproducible, especially by Spearman correlation, but absolute returns are high-variance and describe a weak local behavior policy rather than efficient graph cost-to-go.

The decisive target-utility check is to insert the **exact ten-rollout aggregate**, before any model error, into the search. At `w=1.25`, exact rollout `g+h` is `+5.33` expansions worse than Euclidean FOCAL, losing on five of six worlds, with mean path-cost ratio `1.101`. As an unconstrained primary A* heuristic it saves `33.5` expansions on all six worlds, but raises mean path cost by `15.7%` and reaches `24.5%` on the worst world.

Thus the target carries a directional shortcut signal, but it is not aligned with shortest-path-efficient bounded search. Increasing model size or rollout count alone is not a sufficient remedy.

## 3. Is it a representation problem?

There is a real representation problem, but it is architecture-specific rather than a universal absence of signal.

| Model | Train worlds | Same-world Pearson | Held-out-world Pearson | Held-out MAE |
| --- | ---: | ---: | ---: | ---: |
| Flat MLP | 3 | `0.691` | `-0.036` | `0.732` |
| Flat MLP | 6 | `0.712` | `0.561` | `0.548` |
| Flat MLP | 12 | `0.721` | `0.675` | `0.469` |
| HRM, padded last state | 12 | `0.741` | **`0.750`** | **`0.425`** |
| HRM, trimmed sequence | 12 | `0.651` | `0.506` | `0.547` |
| HRM, summary-token readout | 12 | `0.662` | `0.476` | `0.555` |
| ON-LSTM, padded last state | 12 | `0.443` | **`0.006`** | `0.699` |
| ON-LSTM, trimmed sequence | 12 | `0.552` | **`0.489`** | `0.595` |
| Masked pooling | 12 | `0.585` | `0.326` | `0.658` |

The learning curve matters. The flat MLP moves from no cross-world transfer at three training worlds to `0.561` at six and `0.675` at twelve. C13-B's three-world smoke therefore confounded insufficient world diversity with identifiability.

The sequence contains a mean of `16.10` real tokens and `8.90` padding tokens, with 10–23 real tokens across nodes. Removing padding at inference from the trained padded HRM shifts predictions by `0.665` on average and lowers correlation from `0.750` to `0.419`. HRM has learned to use the padded tail and sequence length as a proxy for neighborhood cardinality. Conversely, ON-LSTM does not learn a transferable padded readout; retraining on trimmed sequences repairs much of the failure.

The correct repair is not to preserve accidental zero-padding semantics. It is to provide an explicit token mask and explicit degree/cardinality feature, then aggregate only real action tokens. The flat MLP must remain as a control because it already captures most of the available transferable signal.

## 4. Is important information missing?

Some global information is necessarily missing from a bounded static observation, but the study does not support hard non-identifiability.

The compact five-neighbor regressor reaches held-out Pearson `0.729`, and compact ridge reaches `0.660`. Across 1,200 training nodes, the nearest standardized compact feature vector from a different world has mean target gap `0.482`, versus `0.934` for a random cross-world partner. Nearest-feature aliasing therefore cuts the expected gap roughly in half (`0.516x` random) but does not eliminate it.

The evidence-safe interpretation is:

- rays, goal geometry, and one-hop actions contain substantial transferable information about expected local-policy difficulty;
- global maze topology still creates observation aliases that a static current-state encoder cannot resolve;
- temporal observation/history may reduce those aliases if the professor's state definition permits it, but it should be tested explicitly rather than assumed to be HRM's hidden advantage.

## 5. Is it an integration problem?

Yes. It is the dominant cause of the original `w=1.10` planning result.

### FOCAL-width audit

| Setting | Mean expansions | Delta vs matched Euclidean FOCAL | Mean cost ratio | Paired outcome |
| --- | ---: | ---: | ---: | --- |
| Euclidean rank, `w=1.10` | `129.67` | `0.00` | `1.016` | control |
| Oracle `g+h*`, `w=1.10` | `131.00` | `+1.33` | `1.000` | no exposed headroom |
| HRM `g+h_hat`, `w=1.10` | `133.17` | `+3.50` | `1.036` | worse |
| Exact rollout `g+h`, `w=1.10` | `136.00` | `+6.33` | `1.034` | worse |
| Oracle `g+h*`, `w=1.25` | `116.50` | `-11.17` | `1.000` | 5 wins, 1 loss |
| HRM `g+h_hat`, `w=1.25` | `123.50` | `-4.17` | `1.054` | 3 wins, 1 tie, 2 losses |
| Exact rollout `g+h`, `w=1.25` | `133.00` | `+5.33` | `1.101` | 1 win, 5 losses |
| Flat MLP `h`, `w=1.50` | `105.83` | `-19.33` | `1.086` | 6 wins |

Widening FOCAL allows learned ranks to affect selection, but the apparent speedup spends the allowed path-cost slack and is not stable across architectures. The response is also non-monotonic: widening changes which Euclidean-lower-bound nodes enter FOCAL and interacts with closed-node order, so a larger set does not guarantee that exact `g+h*` reaches the goal in fewer expansions.

Ranking the learned residual by itself does not repair the original configuration. At `w=1.10`, residual-key deltas are `+4.33` for the flat MLP, `+4.67` for HRM, and `+4.17` for trimmed ON-LSTM. The useful quantity is not simply the predicted detour residual.

### Primary-A* diagnostic

| Primary heuristic | Mean expansions | Delta vs Euclidean A* | Mean / max cost ratio | Expansion wins |
| --- | ---: | ---: | ---: | ---: |
| Euclidean | `139.17` | `0.00` | `1.000 / 1.000` | control |
| Oracle `h*` | `23.33` | `-115.83` | `1.000 / 1.000` | `6/6` |
| HRM rollout estimate | `75.17` | `-64.00` | `1.147 / 1.246` | `6/6` |
| Trimmed ON-LSTM | `77.00` | `-62.17` | `1.070 / 1.189` | `6/6` |
| Flat MLP | `77.83` | `-61.33` | `1.182 / 1.427` | `6/6` |
| Exact ten-rollout target | `105.67` | `-33.50` | `1.157 / 1.245` | `6/6` |

This is the cleanest causal result. Learned estimates strongly reorder the graph in a useful direction, but they are overestimating, inconsistent behavior values rather than admissible shortest-path heuristics. Primary A* can exploit them only by giving up the path-quality guarantee. The current FOCAL insertion keeps the guarantee but suppresses most of their useful ordering.

## 6. Causal classification

1. **Primary cause of the original smoke failure: integration.** The declared narrow Euclidean-anchored FOCAL set has almost no oracle-ranking headroom, while the same learned estimates are powerful as unsafe primary heuristics.
2. **Independent blocker: target/policy alignment.** The exact rollout target is inefficient relative to the graph oracle and fails the intended search before model approximation enters.
3. **Real contributing defect: representation/readout.** Padding destroys ON-LSTM transfer and gives HRM an accidental cardinality channel. An explicit mask/degree contract is required.
4. **Partial but not dominant limitation: missing global context.** Static local features cannot fully distinguish global detours, but multiple simple controls show that they do identify substantial transferable signal.
5. **Not supported: “HRM hierarchy is the missing ingredient.”** HRM fits this target best, but the flat MLP and trimmed ON-LSTM expose similar unsafe A* speedups, and no architecture supplies a stable bounded gain at the original setting.

## 7. Decision and next gate

Do **not** run the current C13-B formulation across all suites and densities. A larger run would measure a known target/integration mismatch more precisely.

The next one-suite study should change two contracts while keeping the no-shortest-path-supervision boundary:

1. **Representation repair:** explicit real-token masks, explicit one-hop degree/cardinality, masked action aggregation, and matched flat-MLP/HRM/ON-LSTM controls.
2. **Target-utility gate before training:** the exact held-out rollout statistic must improve the intended bounded search over its matched Euclidean control. If the exact target fails, no learned approximation is authorized.
3. **Behavior-value repair:** reduce the `3.30x` behavior/oracle gap with a stronger permitted local/history-aware rollout policy, or replace raw behavior return with a preregistered decision-aligned statistic such as a fixed-threshold success/cost distribution. Do not select the minimum of repeated rollouts.
4. **Integration repair:** test an anchored multi-queue or multi-heuristic search in which Euclidean retains the bound while the learned estimate gets its own primary queue. This directly targets the large A*-primary signal that the current FOCAL secondary key suppresses.
5. **Authorization gate:** require paired expansion gains with the declared cost bound on the one-suite audit, then repeat at 192 and 211 nodes. Only after both pass should the multi-suite density run resume.

This preserves the professor's methodological constraint: no global shortest-path value is used to train the model. Dijkstra remains an evaluation ceiling that tells us whether a proposed target and integration could have worked—not a source of supervision.

## 8. C13-C follow-up

The next integration-only gate is complete. See
[C13C_CERTIFIED_SEARCH.md](C13C_CERTIFIED_SEARCH.md).

A separate primary-rank incumbent search plus a fresh Euclidean certifier is
proof-correct but too duplicative: at `w=1.10`, even the privileged optimal
oracle incumbent loses all six matched FOCAL comparisons. The next gate must
share search and proof state before target or representation repair is useful.

## 9. C13-D follow-up

[C13D_SHARED_QUEUE_ORACLE.md](C13D_SHARED_QUEUE_ORACLE.md) implements that
shared-state repair and passes the oracle ceiling at `w=1.10`, winning all
six matched comparisons with valid certificates. The exact frozen rollout
statistic is now the next gate; learned representation remains downstream.

## 10. C13-E follow-up

[C13E_SHARED_QUEUE_EXACT_TARGET.md](C13E_SHARED_QUEUE_EXACT_TARGET.md) freezes
the passed C13-D integration and substitutes only this study's replayed exact
rollout rank. At `w=1.10`, all six paths certify, but total expansions average
`131.00` versus `129.67` for matched FOCAL, with 2 wins, 1 tie, and 3
losses. This confirms target alignment/calibration as a blocker before learned
representation is tested.
