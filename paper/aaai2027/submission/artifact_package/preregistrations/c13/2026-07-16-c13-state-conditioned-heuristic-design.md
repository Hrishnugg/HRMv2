# C13: State-Conditioned Heuristic Revalidation

**Status:** original C13-A-E design complete; later C13-F-M sequence confirms a bounded-observation local-Bellman result  
**Date:** 2026-07-17  
**Motivation:** July 14 professor feedback on roadmap density and the use of obstacle-aware Dijkstra distance in C6/C7.

## 1. Why C6/C7 cannot simply change one formula

C6/C7 currently learn

\[
r_D(x,g,M)=\frac{\max(0,D_M(x,g)-E(x,g))}{L},
\qquad
h(x)=E(x,g)+L\hat r_D(x,g,M),
\]

where `D_M` is an obstacle-aware grid/graph shortest-path distance. The learned field also receives a full raster containing occupancy, reachability, clearance, start/goal channels, coordinates, and Euclidean distance. The arm is therefore map-conditioned in both its supervision and its runtime input.

The suggested replacement `constant - E` cannot be inserted as the same additive residual. If `C` is at least the maximum Euclidean distance, then

\[
h(x)=E(x,g)+(C-E(x,g))=C,
\]

so every node receives the same heuristic and A* collapses toward uniform-cost search. `C-E` is meaningful only as a **goal-proximity value that is maximized**; the current planners consume a **cost-to-go estimate that is minimized**. After converting orientations, its ordering is exactly Euclidean and provides no obstacle information. C13 retains this literal construction only as a semantics control.

## 2. State definitions

C13 separates two interpretations that must not be conflated in the paper.

### G0: strict geometric state

`z_G = (current position, fixed goal, side length)`.

No obstacle information is present. The strongest natural admissible cost-to-go lower bound under Euclidean edge costs is Euclidean distance. `C-E` is an equivalent goal-proximity ordering after the sign/orientation conversion. A learned model is not expected to add map-specific guidance in this arm.

### O1: locally observed planning state

`z_O = (z_G, bounded range readings, locally available one-hop roadmap successors)`.

This corresponds to information available at the current state without a global map traversal. Inputs are restricted to:

- current-to-goal displacement and Euclidean distance;
- normalized current coordinates;
- fixed-radius ray readings, truncated at the sensor boundary;
- collision-free one-hop successor actions exposed by the planner's current graph state.

The successor list is one-hop but is not clipped by sensor radius. Clipping the Bellman backup to only a subset of outgoing graph edges can destroy admissibility when an omitted edge begins the optimal route. The graph action list is already available to the planner at the current node; reading it is different from recursively traversing the graph or querying a shortest-path table.

Explicitly forbidden runtime inputs are the global occupancy raster, reachable-free mask, global clearance field, start channel, task/world descriptor, total obstacle count, global free fraction, unbounded goal line-of-sight, corridor scans, `dist_to_goal`, and any shortest-path result.

## 3. Replacement target: one-step Euclidean relaxation

Let

\[
h_0(s)=E(s,g)
\]

and let `A(s)` contain the collision-free one-hop successor actions available at the current graph state. Define

\[
h_1(s)=
\begin{cases}
0 & s=g,\\
\min_{s'\in A(s)} [c(s,s')+h_0(s')] & A(s)\ne\varnothing,\\
h_0(s) & A(s)=\varnothing.
\end{cases}
\]

The learned target is

\[
r_1(s)=\frac{h_1(s)-h_0(s)}{L}.
\]

This target uses no Dijkstra/A*/all-pairs result and performs no recursive graph traversal. It is one local Bellman backup of a relaxed Euclidean heuristic.

For Euclidean edge costs:

- `h_1 >= h_0` by the triangle inequality;
- `h_1 <= h*` because replacing the successor's true cost-to-go with Euclidean can only lower the one-step optimal backup;
- `h_1` is consistent because `h_1(u) <= c(u,v) + h_0(v) <= c(u,v) + h_1(v)` for every graph edge.

The direct `h_1` arm is therefore a valid admissible heuristic. It is also an essential control: because `h_1` is cheap to compute directly, a learned approximation cannot be credited for information already available to the planner.

## 4. Learned arms and search integration

HRM and ON-LSTM consume the same bounded O1 token sequence and predict the nonnegative normalized residual `r_1`.

The learned prediction is **not guaranteed admissible**, even though its training target is. Therefore:

- primary learned evaluation uses Euclidean-anchored focal A* (`w in {1.05, 1.10, 1.25}`), with the prediction used only to rank nodes inside FOCAL;
- direct additive A* with the learned prediction is a diagnostic arm and cannot support optimality claims;
- the direct analytical `h_1` arm is evaluated with ordinary A*;
- graph Dijkstra is permitted only as an evaluation oracle/connectivity check and is never an input or training label.

## 5. Preregistered arms

| Arm | Training target | Runtime information | Integration | Role |
|---|---|---|---|---|
| `euclid` | none | G0 | A* | primary baseline |
| `goal_proximity` | none (`C-E`) | G0 | orientation-converted rank | equivalence/semantics control |
| `constant_cancel` | none (`E + C-E`) | G0 | A* | cancellation/uniform-cost control |
| `one_step` | none | O1 local successors | A* | direct admissible local relaxation |
| `state_hrm` | `r_1` | O1 tokens | focal, primary | learned state-conditioned ranker |
| `state_onlstm` | `r_1` | O1 tokens | focal, primary | architecture control |
| learned additive variants | `r_1` | O1 tokens | A*, diagnostic | failure/robustness diagnostic only |
| `oracle_eval_only` | none | graph shortest path | A* | evaluation ceiling only |

The historical C7 field may be shown as a separately labelled map-conditioned/oracle-supervised reference, but it is not pooled with the C13 state-conditioned arms.

## 6. Roadmap-density sweep

Train learned models only at the existing C7 density (`192` total nodes, including start and goal; `k=7`). Evaluate the same checkpoints at:

`N in {128, 160, 192, 211, 256}` with `k=7`.

`211` is the rounded `+10%` setting requested in the meeting. For each world, all densities reuse the same world seed and roadmap sampling seed, so the lower-density point sets are prefixes of the higher-density samples. The graph is rebuilt at each density, as it must be for a k-nearest PRM.

Report both total nodes and `N / L^2`; a fixed node count does not imply a fixed spatial density on the larger suites.

The sweep records separate cost categories:

- offline PRM build time and edge count;
- connectivity rate and graph-optimal path cost (evaluation only);
- online heuristic inference time;
- online search time, expansions, success at fixed budgets, and returned path-cost ratio.

Changing `N` while keeping `k=7` tests node density, not branching factor. A secondary `k in {5,7,9}` sweep is allowed only after the primary result because varying both together would confound density with degree.

## 7. Data splits and leakage controls

- Train worlds and evaluation worlds use disjoint seed ranges.
- The train split uses the three C7 training families; near-, structural-, and scale-OOD suites remain held out as in C7.
- Train/validation splitting is by world, never by node.
- Dataset metadata records `target_source=one_step_local_euclidean_backup`, `shortest_path_target=false`, and the sensor radius.
- Unit tests use a roadmap object whose `dist_to_goal` property raises if accessed by target/feature code.
- A feature-invariance test changes obstacles, start state, and descriptors outside the sensing radius and requires identical current-state tokens.

## 8. Gates and claim policy

### G0 — semantics

- `E + (C-E)` is numerically constant.
- Orientation-corrected `C-E` has exactly the same ordering as Euclidean.

### G1 — provenance

- Feature and label construction pass with `dist_to_goal` inaccessible.
- No forbidden global field appears in saved dataset metadata or model input.

### G2 — analytical local relaxation

- `h_1 >= E`, `h_1 <= h*`, `h_1(goal)=0`, and edge consistency hold on the audit set.
- `one_step` never returns a more expensive solution than Euclidean A* when both complete.

### G3 — learned signal

- Compare HRM and ON-LSTM against Euclidean and direct `one_step` on matched worlds.
- A learned win over Euclidean is not enough if it fails to match the direct computation of its own target.
- Learned focal results must report success, expansions, wall time, and suboptimality jointly.

### G4 — density robustness

- Report the full curve; do not select only the best density.
- A density effect is not an architecture effect.

Claim-safe outcomes:

- If only `one_step` helps, conclude that local successor information improves Euclidean but learning it is unnecessary.
- If learned arms help only in-distribution or only at `N=192`, conclude density/graph-artifact sensitivity.
- If HRM improves focal ranking across held-out suites and densities, claim a bounded-local-observation-conditioned ranking benefit—not a map-free optimal heuristic.
- If G0-only geometry is required by the professor, Euclidean/proximity is the complete honest baseline; nontrivial obstacle-aware guidance requires expanding "state" to include observations or history.

## 9. C13-B amendment: fresh-start rollout value

The working interpretation approved for implementation permits bounded local observations and outcomes from environment interaction, provided no full-map shortest-path result is an input or label. Formal professor confirmation is still required before publication-facing claims.

For each PRM node, C13-B generates independent behavior-policy rollouts using only current one-hop actions, goal geometry, and visit memory accumulated after that fresh start. Visit history is reset for every labeled start. Only the start node receives the trajectory's total realized cost; intermediate returns conditioned on hidden history are excluded. Successful returns are aggregated by their median, never their minimum, so repeated exploration does not become a shortest-path solver.

The learned target is `log1p(max(0, rollout_return - Euclidean) / side_len)`. The inverse transform restores cost units before search ranking. This transform preserves target ordering while avoiding saturation from the heavy-tailed behavior-policy returns.

HRM and ON-LSTM still receive only the O1 current-observation tokens. Their predictions are non-admissible secondary ranks inside Euclidean-anchored FOCAL. Evaluation must include `euclid_focal_rank` and `one_step_focal_rank` same-search controls; ordinary Euclidean A* alone cannot separate learning from the benefit of changing the search integration.

The [C13-B smoke](../results/C13B_ROLLOUT_RANKER_SMOKE.md) validates the implementation and provenance. The subsequent [identifiability study](../results/C13B_IDENTIFIABILITY_STUDY.md) finds strong held-out prediction signal but separates exact-target, readout, and FOCAL-insertion failures. [C13-C](../results/C13C_CERTIFIED_SEARCH.md) rejects a separate incumbent search plus fresh Euclidean certifier because the proof search duplicates too much work. [C13-D](../results/C13D_SHARED_QUEUE_ORACLE.md) repairs that ceiling: the shared Euclidean-anchor/oracle-rank search certifies optimal paths and beats matched FOCAL on all six `w=1.10` worlds. [C13-E](../results/C13E_SHARED_QUEUE_EXACT_TARGET.md) freezes that integration and substitutes only the replayed exact rollout rank; it certifies all six paths but loses the primary gate at `131.00` versus `129.67` mean expansions with only 2/6 wins. Target alignment/calibration must now be repaired before learned providers, 192/211-node replication, or the full multi-suite run.

## 10. C13-F through C13-M outcome addendum

The C13-E blocker language above records the state of the program at that gate. Subsequent work follows the literature-routed sequence rather than retuning C13-E. Calibration and exact shallow local-escape controls fail; local heuristic Bellman learning confirms on maze but initially fails the full C7 distribution; suite-balanced training alone also fails. One radius-bounded Bellman backup at inference is the decisive repair.

The preregistered C13-M confirmation evaluates a fixed iteration-8/alpha-1.50 arm on 144 untouched six-suite worlds. It averages `68.31` expansions versus `81.26` for complete-map field HRM, paired delta `-12.96` with 95% CI `[-16.30, -9.74]`, and lower empirical mean/max path-cost ratios. A separate `w=1.10` reopening-FOCAL control has zero observed bound or certificate violations.

The direct result is not formally bounded and its current feature builder is slower in wall time. See the canonical [C13-F through C13-M result](../results/C13F_M_CURRENT_STATE_RESULT.md) for the full mechanism, integrity, and claim-scope record.

