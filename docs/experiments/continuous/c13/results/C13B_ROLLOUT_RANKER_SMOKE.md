# C13-B Fresh-Start Rollout Ranker Smoke

**Status:** completed implementation/provenance smoke; learned-signal gate not passed  
**Date:** 2026-07-16  
**Suite:** `C_hard_maze` only  
**Roadmap:** trained at 192 nodes / `k=7`; evaluated at 192 and 211 nodes

Raw artifacts:

- [train dataset metadata](../../../../../hrm-cloud/continuous_prm/runs/c13_td_smoke/datasets/c13_td_train.metadata.json)
- [validation dataset metadata](../../../../../hrm-cloud/continuous_prm/runs/c13_td_smoke/datasets/c13_td_val.metadata.json)
- [raw planner evaluation](../../../../../hrm-cloud/continuous_prm/runs/c13_td_smoke/results/c13_td_evaluation_raw.csv)
- [planner summary](../../../../../hrm-cloud/continuous_prm/runs/c13_td_smoke/results/c13_td_evaluation_summary.csv)
- [evaluation manifest](../../../../../hrm-cloud/continuous_prm/runs/c13_td_smoke/evaluation_manifest.json)

## Executive decision

C13-B is implemented end to end, but this smoke does **not** justify the final multi-suite run yet.

The provenance boundary passes: no shortest-path result is a feature or label, successful labels come from feasible one-hop-policy trajectories, and learned predictions are used primarily as secondary ranks inside Euclidean-anchored FOCAL search. The corrected collector also avoids a subtler state mismatch: every labeled node receives independent fresh-start rollouts with visit history reset, and intermediate returns conditioned on hidden history are not used as labels.

The learned-signal gate fails. ON-LSTM is one expansion better than the matched Euclidean-ranked FOCAL control at 192 nodes but 2.0 expansions worse at 211. HRM is worse at both densities. Neither result is stable enough to scale or claim.

## 1. Target and leakage contract

For each PRM node, C13-B executes three independent behavior-policy rollouts. At every step the policy may inspect only:

- the current node's one-hop outgoing actions and edge costs;
- Euclidean geometry from those actions to the fixed goal;
- its own visit history, used to discourage loops.

The visit history is reset before every labeled start. Only the start node receives that rollout's total realized cost. Successful returns for the same start are aggregated by their median, never their minimum. Thus repeated rollouts estimate the behavior policy's fresh-start return rather than approximate a shortest path.

The regression target is

\[
y(s)=\log\left(1+\frac{\max(0,G_{\pi}(s)-E(s,g))}{L}\right),
\]

where `G_pi` is the successful feasible-rollout return. The logarithm retains the heavy tail without saturating most samples; inference applies the inverse transform before forming the cost rank.

The collector metadata records:

- `shortest_path_target=false`;
- `oracle_policy_access=false`;
- `shortest_path_result_used_as_input=false`;
- `label_history_condition=fresh_start_zero_visit_history_only`.

Dijkstra remains limited to the PRM validity/connectivity filter and evaluation. It is not used by the rollout policy, label builder, feature builder, or model.

## 2. Same-search controls

Learned predictions are not assumed admissible. Their primary integration is a secondary cost-to-go rank inside FOCAL, whose membership is anchored by admissible Euclidean `f` and `w=1.10`.

The evaluation includes two essential same-search controls:

- `euclid_focal_rank`: Euclidean supplies both the admissible anchor and secondary rank;
- `one_step_focal_rank`: Euclidean anchors FOCAL and the analytical one-step backup supplies the secondary rank.

Ordinary Euclidean A* remains the planner baseline. Learned additive A* is diagnostic only and cannot support an optimality claim. Without the same-search controls, the first smoke incorrectly appeared to show a learned gain that was actually explained by FOCAL integration itself.

## 3. Collection and verification

The focused test set passes `13/13`, including:

- every rollout step must correspond to a real graph edge;
- suffix returns must equal the cost of the actually traversed edge sequence;
- a sentinel `dist_to_goal` property must remain unread during label collection;
- aggregation must use the median rather than the minimum;
- fresh-start aggregation must not assign intermediate hidden-history returns;
- HRM and ON-LSTM outputs must remain finite and nonnegative;
- learned FOCAL summaries must compare against the matched Euclidean-ranked FOCAL control.

The train smoke used three worlds and produced `568` labeled node states. Per-world coverage was `100%`, `100%`, and `95.8%`; rollout success was `100%`, `100%`, and `95.8%`. The held-out validation world labeled all `192` nodes with `100%` rollout success. No transformed target hit the configured cap.

## 4. Regression smoke

These are tiny smoke models (`hidden=32`, one recurrent layer, eight epochs), not tuned final fits.

| Model | Validation MAE | Validation correlation | Prediction mean | Target mean |
|---|---:|---:|---:|---:|
| HRM | 0.888 | -0.790 | 0.759 | 1.088 |
| ON-LSTM | 0.714 | -0.006 | 1.042 | 1.088 |

The negative/near-zero correlations are the clearest reason not to interpret planner differences as learned value quality. More epochs or a larger backbone may reduce fit error, but they cannot by themselves resolve target aliasing if bounded local observations do not identify global detour cost.

## 5. Planner smoke

All entries below use budget 192. The 192-node cohort has three worlds. At 211 nodes, one matched roadmap is disconnected, leaving two connected worlds.

| Provider | 192 mean expansions | Delta vs Euclidean FOCAL | 192 mean cost ratio | 211 mean expansions | Delta vs Euclidean FOCAL | 211 mean cost ratio |
|---|---:|---:|---:|---:|---:|---:|
| Euclidean A* | 143.67 | — | 1.000 | 155.50 | — | 1.000 |
| Euclidean FOCAL rank | 134.67 | 0.0 | 1.0179 | 148.50 | 0.0 | 1.0151 |
| One-step FOCAL rank | 134.67 | 0.0 | 1.0184 | 150.50 | +2.0 | 1.0137 |
| HRM rollout rank | 146.67 | +12.0 | 1.0292 | 168.00 | +19.5 | 1.0294 |
| ON-LSTM rollout rank | 133.67 | -1.0 | 1.0272 | 150.50 | +2.0 | 1.0141 |

At budget 96, none of the non-oracle arms completes, so the smoke shows no success-rate separation at the tighter budget.

## 6. Interpretation and next gate

The implementation question is answered: rollout supervision can be made shortest-path-free and integrated without an admissibility claim. The scientific question remains open.

The immediate risk is partial observability. A global maze detour can differ sharply between two nodes with similar bounded rays and one-hop actions. A static current-observation ranker may therefore face irreducible label aliasing even when its labels are clean. HRM does not automatically solve that problem here because its input sequence contains current rays/actions, not a temporal observation history.

Before a full multi-suite density run, require a one-suite learning/identifiability gate:

1. measure per-node return variance across independent rollout seeds;
2. run a train-world learning curve and compare HRM/ON-LSTM with a small MLP control;
3. require positive held-out return correlation;
4. require a learned FOCAL ranker to beat `euclid_focal_rank` at both 192 and 211 nodes without a worse bounded-cost tradeoff;
5. if that gate still fails, treat bounded-current-observation value as non-identifiable and discuss whether temporal observation history is permitted rather than scaling the same static formulation.

Until those conditions pass, C13-B is a validated pipeline with a negative preliminary signal result—not evidence that the learned ranker improves planning.

## 7. Completed follow-up

The requested one-suite gate has now been run. See [C13B_IDENTIFIABILITY_STUDY.md](C13B_IDENTIFIABILITY_STUDY.md).

The three-world smoke was too small to diagnose identifiability: with 12 training worlds, HRM reaches held-out target correlation `0.750` and a flat MLP reaches `0.675`. The planning gate still fails for a more specific reason. At the original `w=1.10`, even exact graph-distance ordering does not beat Euclidean-ranked FOCAL, while the exact ten-rollout target makes it worse. Primary-A* diagnostics show that learned estimates contain strong but unsafe ordering signal. The revised next gate therefore requires target utility and bounded integration to pass before any multi-suite scaling.
