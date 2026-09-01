# C13 Current-State Literature Review and Next-Target Decision

**Date:** 2026-07-17  
**Status:** experiment sequence complete; C13-M fresh confirmation passed  
**Purpose:** choose the next target from mechanisms that are actually compatible
with the professor's current-state constraint.

## 1. Methodological filter

The intended method must be a function of the current planning state, not a
function of a complete obstacle map or a precomputed map-wide distance field.
For C13, the defensible runtime state is:

- current and goal geometry;
- bounded sensor observations centered on the current state;
- collision-free actions or roadmap edges visible inside that bounded region;
- optional recurrent history accumulated by the agent before the current
  state; and
- fixed global constants such as sensor radius and roadmap density.

The method must not read a full occupancy raster, map-wide clearance or
reachability channels, graph `dist_to_goal`, grid Dijkstra, A*, or another
full-problem solution as a runtime input or training label. Graph shortest
paths remain permitted only for held-out evaluation and safety checks.

This is stricter than merely using a scalar output. C7's scalar arm is
node-indexed at inference but still learns a full-map shortest-path residual;
it does not satisfy the target-provenance requirement.

## 2. Primary literature

### 2.1 Local Heuristic A* (LoHA*)

Veerapaneni, Saleem, and Likhachev define a local region around the current
state and predict only the additional cost needed to escape that region. The
local residual is added to a simple global heuristic. Their exact local target
is a multi-goal search to either the goal inside the region or a boundary
state, with the boundary's global heuristic as terminal cost. They then use a
learned approximation in focal search to retain bounded suboptimality.

The key transferable construction is

\[
h_K(s)=\min_{b\in B_K(s)}\{d_K(s,b)+h_0(b)\},
\qquad
r_K(s)=h_K(s)-h_0(s),
\]

where `d_K` uses only transitions observed inside the bounded region and
`h_0` is Euclidean. The paper reports 2–20x node-reduction factors on its
navigation tasks and emphasizes that the local formulation generalizes to new
maps and longer horizons.

Source: [Learning Local Heuristics for Search-Based Navigation Planning,
ICAPS 2023](https://ojs.aaai.org/index.php/ICAPS/article/view/27245).

Relevance to C13: this is the closest direct precedent for a useful
obstacle-aware heuristic whose learned input and target are centered on the
current state rather than the full planning map. It also gives an analytical
exact-local ceiling that can be tested before training.

### 2.2 LRTA* and Real-Time Adaptive A*

Real-time heuristic search restricts lookahead to a local search space reached
from the current state. RTAA* performs a bounded A* lookahead and updates
expanded-state heuristics using the best frontier `f` value. The update is a
local Bellman-style improvement and was evaluated for navigation in unknown
terrain.

Source: [Real-Time Adaptive A*, AAAI Workshop
2006](https://cdn.aaai.org/Workshops/2006/WS-06-11/WS06-11-010.pdf).

Relevance to C13: it supports a target family based on bounded frontier values
or TD/Bellman updates rather than inefficient whole-trajectory behavior cost.
Unlike C13-B's Monte Carlo return, the target directly measures the amount by
which local search raises the current heuristic.

### 2.3 Multi-Heuristic A*

MHA* combines one consistent anchor with arbitrarily inadmissible auxiliary
heuristics while retaining completeness and bounded-suboptimality guarantees.
It is the conceptual basis for C13-D's shared anchor/rank implementation.

Source: [Multi-Heuristic A*, International Journal of Robotics Research
2016](https://publications.ri.cmu.edu/multi-heuristic-a-2).

Relevance to C13: the anchor architecture is appropriate, but C13-E exposes a
scale interaction. The behavior-return rank averages several times the anchor
and is almost never queue-eligible. A same-search Euclidean control and fixed
monotone residual calibration are therefore required before blaming ordering.

### 2.4 Policy-guided search and search-effort objectives

Policy-Guided Heuristic Search separates local action preference from global
cost-to-go estimation and gives search-loss guarantees related to both policy
and heuristic quality. Learning Heuristic Search via Imitation likewise argues
that minimizing cost-estimation error need not minimize expansions and treats
node selection as a sequential decision problem.

Sources:

- [Policy-Guided Heuristic Search with Guarantees, AAAI
  2021](https://ojs.aaai.org/index.php/AAAI/article/view/17469)
- [Learning Heuristic Search via Imitation, CoRL
  2017](https://proceedings.mlr.press/v78/bhardwaj17a.html)

Relevance to C13: a local policy or pairwise action-ranking target is a
scale-free fallback if local value targets remain poorly calibrated. The SAIL
imitation target in the latter paper uses a clairvoyant Dijkstra oracle and is
therefore not admissible under the professor's constraint; only the
search-effort formulation transfers.

## 3. Experiment sequence

### C13-F — scale versus ordering

Freeze the C13-D search and C13-B exact rank. Evaluate:

- same-search Euclidean rank (`alpha=0`);
- `Euclidean + alpha * (rollout_exact - Euclidean)` for fixed
  `alpha in {0.05, 0.10, 0.25, 0.50, 1.00}`; and
- the unchanged matched Euclidean FOCAL and shared-oracle controls.

This is a development-only diagnostic on the existing six audit worlds.
Selection on those worlds cannot authorize a final claim; any chosen fixed
coefficient must be replicated on fresh worlds.

### C13-G — exact bounded local-escape ceiling

Construct a radius-bounded subgraph centered on each current node. Run local
Dijkstra only inside that observed subgraph and terminate at either:

- the goal, if locally visible and reachable; or
- the first edge that exits the observation radius, adding Euclidean distance
  from the outside endpoint to the goal.

This target never traverses the full graph. Every global path must either
reach the local goal or cross one of those exit edges, so the minimum candidate
is a lower bound on global path cost when all locally visible exits are
included. Test locality by perturbing all geometry outside the radius and
requiring an unchanged target.

Evaluate the exact local heuristic before learning it. Sweep radius as a
physical fraction of side length, not graph depth, so the information boundary
is stable across 192 and 211 nodes.

### C13-H — learned current-state provider

Train only if C13-G exposes meaningful exact-local headroom. Inputs are a
masked, permutation-controlled encoding of the bounded local subgraph or
sensor patch; the label is the local-escape residual, never graph-to-goal
distance. Include flat MLP, masked set/graph aggregation, and recurrent HRM as
representation controls. Select by planner metrics, not MSE alone.

If the exact local value ceiling fails, skip value regression and test a
successful-trajectory action policy or pairwise rank objective in the same
bounded search.

## 4. Required comparison with C7

A result is not considered solid merely because it passes the six-world C13
development gate. The final evaluation must use C7's six static suites,
192/k7 roadmaps, matched seeds, binding budgets, and metrics:

- success at the binding budget;
- matched-solved expansion ratio and paired confidence interval;
- path-cost ratio and declared bound;
- heuristic construction/inference time; and
- in-distribution versus near-, structural-, and scale-OOD results.

The current-state method must be compared directly with Euclidean, C7
map-conditioned field HRM, and C7 full-map-supervised scalar HRM on the same
instances. Historical C7 ratios are context, not a substitute for rerunning a
matched harness.

## 5. Claim policy

- A local exact heuristic win establishes current-state/local-observation
  algorithmic headroom, not learned-model superiority.
- A learned win must survive fresh worlds, both 192 and 211 nodes, and held-out
  suites.
- Local obstacle patches must be called bounded observations, not “no map.”
- No result can be described as outperforming C7 until the matched C7 rerun is
  complete.

## 6. Completed outcome

C13-F confirms that rollout calibration alone is insufficient. C13-G rejects both exact shallow local-escape variants. C13-H learns the local Bellman objective and confirms a bounded one-suite arm at 192 and 211 nodes, but C13-I shows that model does not transfer across C7. C13-J's suite-balanced training remains worse when inserted statically. C13-K identifies one inference-time local Bellman backup as the missing integration mechanism, and C13-L exposes a comparator-relative matched-quality frontier while correctly rejecting the absolute 1.10 direct-search ceiling.

C13-M freezes iteration 8, radius 0.20, alpha 1.50, and direct no-reopen A* before generating 144 untouched worlds. Relative to complete-map field HRM, it confirms:

- `68.31` versus `81.26` mean expansions;
- paired delta `-12.96`, 95% CI `[-16.30, -9.74]`;
- all six suite mean deltas negative; and
- lower empirical pooled mean/max path-cost ratios.

The direct arm is an unbounded matched-quality result; the separate `w=1.10` FOCAL arm is the bounded control. See [C13F_M_CURRENT_STATE_RESULT.md](../results/C13F_M_CURRENT_STATE_RESULT.md).
