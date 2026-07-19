# C13-P preregistration: persistent search-state HRM

**Frozen design date:** 2026-07-19
**Status:** approved design; no C13-P implementation, trace generation, model
training, development evaluation, or confirmation has occurred
**Purpose:** test whether HRM state carried across the causal expansion sequence
supplies the planning-time mechanism absent from C13-N and C13-O

## 1. Decision and scope

C13-N and C13-O apply a recurrent model independently to every roadmap node.
Their low- and high-level states are recreated for every example. C13-P changes
that state lifetime: one HRM carry is initialized at the beginning of a
planning query, updated once after each expansion, and reset only after the
query ends.

The approved first study is a scoped pilot. It uses a stationary,
behavior-derived path-frontier target and compares two evaluation modes of one
checkpoint:

- `persistent`: carry survives across expansions in the same query;
- `reset`: carry is zeroed before every expansion event.

The unchanged C13-M current-state/local-Bellman method is the static baseline.
C13-P does not include self-bootstrap or on-policy relabeling. A positive pilot
may authorize a separately preregistered on-policy study; it does not authorize
retuning this pilot after development results.

## 2. Scientific question and hypotheses

### 2.1 Primary question

Does query-level HRM state improve identification of the successful-path
frontier and reduce online search expansions beyond both a same-checkpoint
reset ablation and the unchanged C13-M static method?

### 2.2 Mechanism hypothesis

The causal expansion sequence reveals information about explored branches,
frontier evolution, and search progress that is absent from an independently
encoded node. A persistent fast/slow state can use that information to resolve
otherwise similar local observations.

### 2.3 Falsifiers

The persistence hypothesis is rejected for this formulation if any of the
following occurs:

1. carry does not improve held-out path-frontier ranking over the identical
   checkpoint with reset state;
2. offline ranking improves but free-running online search does not;
3. expansion gains require path-quality regressions beyond the locked margins;
4. the apparent gain depends on forbidden future, full-map, or shortest-path
   information; or
5. the result is not deterministic under an exact duplicate evaluation.

## 3. Claim and information boundaries

### 3.1 Permitted runtime information

C13-P may read only:

- C13-J's existing node-local token representation: current/goal geometry,
  bounded rays, and one-hop actions;
- the radius-0.20 local subgraph and frozen exit values already used by C13-M's
  inference-time local Bellman backup;
- causal search facts observed by the algorithm before the current decision:
  expanded-node identity and representation, `g`, the frozen C13-M base score,
  open count, closed count, and expansion index; and
- fixed constants such as side length, roadmap size, sensor radius, and model
  configuration.

### 3.2 Forbidden information

Neither target construction nor any model-visible computation may read:

- graph `dist_to_goal`;
- grid or graph Dijkstra values;
- a clairvoyant/optimal A* result or another full-problem oracle solution used
  as a label;
- a complete occupancy raster or map-wide reachability/clearance channel;
- data from another world or a previous planning query.

No model input, carry update, candidate score, or online decision may read
future expansion events, future open/closed sets, or the final returned path.
The offline label constructor alone may inspect the completed frozen-behavior
path. That path is not a shortest path and is never presented to the model as
an input.

### 3.3 Claim boundary

The online C13-P arm retains A*-style `g` relaxation and expands a node at most
once, but its priority changes with history. It is a history-conditioned
best-first policy, not a static A* heuristic and not a formally bounded search
method. All path-quality statements are empirical and comparator-relative.

The observation simulator uses a known PRM with absolute coordinates. C13-P is
not a map-free navigation claim, a wall-clock speedup claim, or evidence about
HRM architectures in general.

## 4. Frozen sources and audit

The implementation must use new files and a new output directory:

- implementation: `continuous_prm_c13_persistent_search.py`;
- focused tests: `tests/test_c13_persistent_search.py`;
- output: `runs/c13_persistent_search/`.

The following sources are read-only:

- C13-J manifest, cohorts, feature caches, checkpoints, and integrity manifest
  in `runs/c13_lhbl_multisuite/`;
- the C13-J suite-balanced flat-MLP iteration-8 checkpoint;
- C13-M's fixed radius `0.20`, alpha `1.50`, direct no-reopen integration and
  its integrity/evaluation fingerprints in
  `runs/c13_matched_quality_confirmation/`;
- C13-N/O artifacts as negative architecture controls; and
- this preregistration.

Before trace generation, C13-P must:

1. rehash every C13-J integrity entry and the relevant C13-M fingerprint;
2. replay all cohort records exactly, including world seeds, roadmap seeds,
   node/edge counts, cache paths, and cache hashes;
3. require all feature caches to report `reused`;
4. verify zero overlap between the C13-P output directory and every source
   directory; and
5. save the implementation, preregistration, source-manifest, source-integrity,
   source-checkpoint, and source-cohort hashes in a binding file before any
   learned result is accepted.

Any mismatch is a hard stop. C13-P must not repair or overwrite a frozen source.

## 5. Cohorts and stationary teacher traces

### 5.1 Splits

C13-P reuses C13-J's exact splits:

- training: 96 worlds, 32 each from hard maze, rooms, and spiral;
- validation: 24 worlds, 8 from each training suite; and
- development: 24 worlds, 4 from each of the six C13 development suites.

There is no new cohort search and no seed replacement. Training traces are used
for fitting. Validation traces select the checkpoint. Development traces remain
unread until the checkpoint, evaluation code hash, and gate configuration have
been frozen.

### 5.2 Teacher

The teacher is the frozen C13-M static method:

- C13-J suite-balanced flat MLP, outer iteration 8;
- one radius-0.20 local Bellman backup;
- alpha `1.50`;
- direct no-reopen A* ordering; and
- deterministic existing adjacency and node-id tie behavior.

The teacher is run to a valid returned path. Its priority vector is static
within a world. Graph shortest-path values are not read during trace generation.

### 5.3 Event timing

The start node is expanded deterministically. Each nonterminal training event
then follows this order:

1. pop and close the selected node;
2. relax its outgoing edges using the frozen no-reopen rule;
3. form the resulting open candidate set;
4. record the expanded-node event and causal scalar state; and
5. identify the unique path-frontier label offline after the teacher returns.

The terminal goal expansion produces no candidate-ranking event.

### 5.4 Path-frontier label

For each recorded open set, the positive class is the first not-yet-closed node
on the teacher's eventual returned parent-chain path. It must be present in the
recorded open set and must be unique. Every other currently open node is a
negative candidate for that event.

Dataset generation hard-fails if:

- the teacher does not return a valid path;
- the event has no positive, more than one positive, an empty open set, or a
  positive that is already closed;
- a candidate's recorded `g` or parent does not match deterministic replay; or
- a second trace-generation pass produces different bytes or hashes.

This target is decision-aligned behavior supervision. It does not claim that
the teacher path or selected frontier is globally optimal.

### 5.5 Trace artifacts

Trace shards must preserve, at minimum:

- split, suite, world index, world seed, and roadmap seed;
- event index, expanded node, open node ids, closed count, and open count;
- node-aligned frozen feature-cache references and hashes;
- candidate `g` values and frozen C13-M base ranks;
- positive frontier node id;
- teacher path, cost, expansions, and validity for audit only; and
- a schema version and complete generation fingerprint.

The model loader must expose only the permitted causal fields. Teacher path and
future audit fields remain in a separately named privileged section and are
rejected if passed to model input construction.

## 6. Model contract

### 6.1 Frozen node encoder and base score

The C13-J iteration-8 `flat_mlp` checkpoint is loaded read-only. Its 64-wide
hidden encoder produces a node embedding from the existing padded local token
array. All of its parameters remain frozen.

The C13-M local-Bellman rank is also frozen and is supplied as a scalar feature.
C13-P therefore tests planning-time state without changing C13-M's learned
node representation, target training, or local backup.

### 6.2 Persistent HRM event core

The new event core follows the repository's `DeepSapientHRMBackbone` fast/slow
update contract:

- hidden width: 64;
- one low-level block and one high-level block;
- four attention heads per gated recurrent block;
- high-level update cadence `k=2`;
- model seed: `18423`; and
- explicit carry containing low state, high state, and event index.

An expansion event concatenates:

- the frozen 64-wide embedding of the expanded node;
- `g / side_len`;
- frozen local-Bellman rank divided by `side_len`;
- frozen `g + rank` divided by `side_len`;
- `open_count / roadmap_nodes`;
- `closed_count / roadmap_nodes`; and
- zero-based `event_index / roadmap_nodes`, with the start expansion recorded
  as event zero.

A learned linear projection maps this event into the HRM core. The carry is
updated exactly once per expanded node.

### 6.3 Candidate scorer

Each open candidate is scored from:

- its frozen 64-wide node embedding;
- the current 64-wide HRM context;
- candidate `g / side_len`;
- candidate local-Bellman rank divided by `side_len`; and
- candidate `g + rank` divided by `side_len`.

The candidate head is `Linear -> GELU -> Linear` with hidden width 64 and one
unbounded scalar logit. It is applied vectorwise to the complete open set.
Candidate scoring is pure: it cannot mutate carry, and a permutation of the
candidate enumeration must permute logits identically.

### 6.4 Carry modes

One trained checkpoint supports both modes:

- `persistent`: initialize once at the start of a world and retain carry after
  every expansion;
- `reset`: initialize clean state immediately before each event, perform the
  same one-event HRM update, score the event's complete open set, and discard

Both modes execute the same event encoder, HRM blocks, candidate head, and
complete-open-set rescoring. They differ only in the incoming state. No state
is carried across worlds, duplicate evaluations, training epochs, or outer
optimization steps.

On frozen offline traces, carry and reset score the exact same recorded
candidate sets. In free-running search their candidate sets may diverge after
the first different expansion; the matched contract there is the checkpoint,
architecture, update rule, graph, and compute rule rather than identical
realized search states.


## 7. Training protocol

### 7.1 Objective

For each event, cross-entropy is computed over all current open-node logits,
with the unique teacher path-frontier node as the positive class. The reported
loss is event-weighted; per-world metrics are separately macro-averaged so long
teacher traces do not dominate evaluation.

There is no cost-to-go regression, Dijkstra imitation, self-bootstrap,
on-policy relabeling, or development-driven target update.

### 7.2 Optimization

Locked optimization settings are:

- device: CUDA for training, CPU for official evaluation;
- optimizer: AdamW;
- learning rate: `5e-4`;
- weight decay: `1e-4`;
- gradient-norm clip: `1.0`;
- maximum epochs: 20;
- validation patience: 4 epochs;
- TBPTT chunk: 32 expansion events;
- one world stream at a time;
- deterministic epoch-level world shuffle from model seed `18423`; and
- zero dropout.

Carry resets at a true world boundary. At each 32-event TBPTT boundary, tensor
values persist but are detached before the next chunk. Each chunk's mean event
loss produces one optimizer step. The frozen node encoder remains in evaluation
mode and receives no gradients.

### 7.3 Checkpoint selection and resume

The selected checkpoint is the earliest epoch with minimum validation
event-weighted frontier cross-entropy. Validation MRR and top-1 accuracy are
reported but cannot select the checkpoint. Patience counts full epochs without
strict loss improvement.

A training run may resume only when its complete binding fingerprint matches.
Every checkpoint stores model state, optimizer state, completed epoch, RNG
states, source hashes, and trace-dataset hash. A conflicting partial run is a
hard stop requiring a new output directory, not deletion or silent restart.

Once selected, the checkpoint hash, evaluation implementation hash, and gate
configuration are written before development traces are loaded.

## 8. Evaluation arms and online semantics

### 8.1 Offline ranking arms

Development teacher traces are evaluated with:

- `c13p_persistent`;
- `c13p_reset`, using the identical checkpoint; and
- `c13m_base_rank`, which ranks the same recorded open set by frozen C13-M
  `g + rank` with node-id tie-breaking.

Metrics are frontier cross-entropy, mean reciprocal rank (MRR), top-1 accuracy,
and positive-node rank percentile. Event metrics are first aggregated within a
world; primary inference uses the 24 world-level values.

### 8.2 Free-running online arms

All three arms replay each development roadmap from a clean search state.

For `persistent` and `reset`:

1. expand the start node;
2. update carry according to the selected mode;
3. relax outgoing edges;
4. score every currently open node in one vectorized call;
5. rebuild the priority queue from those logits; and
6. pop the maximum-logit node, breaking exact ties by lower frozen C13-M
   `g + rank` and then lower node id.

The queue is rebuilt after every expansion, so no candidate retains a score
from an older history. Candidate enumeration order cannot influence carry or
priority. Closed nodes never re-enter and no node may expand more than once.
The search stops on the first goal pop, an empty open set, or 192 expansions.

`c13m_base` is the unchanged static C13-M direct no-reopen A* arm. Evaluation
records returned path, validity, cost, graph-optimal cost for evaluation only,
cost ratio, expansions, scorer calls, candidates scored, representation time,
model time, and search bookkeeping time.

## 9. Preregistered development gates

All confidence intervals use 20,000 world-clustered bootstrap resamples.
Bootstrap seeds and exact metric code are frozen in the evaluation binding
before development loads.

### 9.1 G0-P: audit and execution integrity

G0-P passes only if:

- every source, cohort, cache, trace, checkpoint, and binding hash matches;
- trace generation is duplicate-byte deterministic;
- model-input leakage checks pass;
- all expected training, validation, and development worlds are present once;
- both carry modes load the identical checkpoint;
- duplicate official evaluation produces identical rows; and
- all returned paths are valid.

G0-P failure yields `c13p_invalid_no_mechanism_verdict` and stops all claims.

### 9.2 G1-P: persistent-state mechanism

For each world, compute persistent minus reset MRR and top-1 accuracy. G1-P
passes only if all conditions hold:

1. the pooled mean MRR improvement has a 95% CI lower endpoint strictly above
   zero;
2. pooled top-1 accuracy improves by at least `0.02` absolute; and
3. persistent MRR is higher than reset MRR in at least four of six suite means.

Failure yields `c13p_no_persistent_ranking_signal` and blocks any claim about
planning-time HRM memory. Online results remain descriptive diagnostics.

### 9.3 G2-P: free-running search value

For each world, compare expansions for persistent minus reset and persistent
minus unchanged C13-M. G2-P passes only if all conditions hold:

1. all 24 persistent, reset, and C13-M searches return valid paths;
2. the pooled persistent-minus-reset expansion CI upper endpoint is below zero;
3. the pooled persistent-minus-C13-M expansion CI upper endpoint is below zero;
4. at least four of six suite-level expansion means are negative against reset;
5. at least four of six suite-level expansion means are negative against
   C13-M;
6. persistent pooled mean graph-optimal cost ratio is no more than `0.005`
   above C13-M's paired pooled mean; and
7. persistent maximum graph-optimal cost ratio is no more than `0.02` above
   C13-M's paired maximum.

Failure after a G1-P pass yields
`c13p_offline_signal_failed_free_running_search`. This is interpreted as
teacher-forcing/distribution-shift or policy-integration failure, not as proof
that search history contains no information.

### 9.4 Overall verdict and authorization

- G0-P fail: invalid study; repair only the named mechanical defect and rerun
  from a new fingerprint.
- G0-P pass, G1-P fail: persistence hypothesis rejected for this target/model.
- G1-P pass, G2-P fail: offline memory signal only; stop without self-bootstrap.
- G1-P and G2-P pass: `c13p_persistent_search_pilot_passed`.

A pass authorizes design of a separate on-policy/self-bootstrap study and a
fresh untouched confirmation. C13-P itself does not generate confirmation
worlds, select a new hyperparameter, or alter any threshold.

## 10. Run stages and hard stops

1. `audit`: rehash sources, replay cohorts/caches, and write the binding.
2. `trace`: generate training and validation traces twice and require identical
   hashes. Development remains unopened.
3. `smoke`: run hand-built trace/state/queue fixtures and one training-world
   forward/backward smoke. Smoke metrics cannot change the design.
4. `train`: fit the stationary model and freeze the validation-selected
   checkpoint plus evaluation binding.
5. `develop`: generate/replay development traces, evaluate offline ranking and
   online search twice, then compute G0-P/G1-P/G2-P.
6. `report`: write generated and canonical reports from computed gate fields.

Each stage requires the preceding fingerprint. The harness defaults to the
earliest incomplete stage and must refuse to mix artifacts from different
fingerprints.

## 11. Required tests

Tests must be written and observed failing before production implementation.
The focused suite must cover at least:

1. persistent carry changes after consecutive events and is unchanged by
   candidate scoring;
2. reset mode starts each event from the exact clean carry;
3. carry resets between worlds and duplicate evaluations;
4. persistent and reset modes share one state dict and parameter count;
5. candidate permutation equivariance;
6. deterministic queue tie-breaking and complete-open-set rescoring;
7. closed-node exclusion and at-most-once expansion;
8. valid path reconstruction under dynamic priorities;
9. exact one-positive path-frontier labeling on hand-built graphs;
10. failure on missing, duplicate, closed, or non-open positives;
11. privileged audit fields rejected by model-input construction;
12. no `dist_to_goal`, Dijkstra, raster, or future-event dependency in trace
    labels or runtime scoring;
13. TBPTT detaches gradients without resetting carry values;
14. validation-only checkpoint selection with earliest-epoch tie behavior;
15. inclusive/exclusive gate boundaries at `0.02`, `0.005`, `0.02`, and zero-CI
    endpoints;
16. exact source/cache/fingerprint refusal on drift;
17. resumable training accepts only an identical fingerprint; and
18. duplicate trace/evaluation artifact equality.

The implementation must also pass the existing C13-N/O focused tests and
compile without warnings attributable to C13-P.

## 12. Artifacts and integrity

Canonical C13-P artifacts are:

- `runs/c13_persistent_search/manifest.json`;
- `runs/c13_persistent_search/source_audit.json`;
- `runs/c13_persistent_search/traces/trace_manifest.json`;
- `runs/c13_persistent_search/traces/{train,validation,development}/...`;
- `runs/c13_persistent_search/checkpoints/...`;
- `runs/c13_persistent_search/results/training_history.csv`;
- `runs/c13_persistent_search/results/checkpoint_selection.json`;
- `runs/c13_persistent_search/results/development_ranking_raw.csv`;
- `runs/c13_persistent_search/results/development_search_raw.csv`;
- `runs/c13_persistent_search/results/gate_verdict.json`;
- `runs/c13_persistent_search/results/verification.json`;
- `runs/c13_persistent_search/results/C13P_RESULT.md`; and
- `runs/c13_persistent_search/integrity.json`.

The integrity manifest hashes the preregistration, implementation, relevant
test file, every frozen input binding, every trace shard, selected checkpoint,
raw result, verdict, verification record, and generated report.

## 13. Approved exclusions

C13-P does not:

- carry state within an arbitrary node enumeration or across independent
  worlds;
- use insertion-time stale scores or per-branch inherited carries;
- add a Euclidean anchor queue or claim bounded suboptimality;
- train a reset-specific checkpoint;
- tune HRM width, cadence, layers, loss, or online score scale on development;
- use self-bootstrap, DAgger, reinforcement learning, or counterfactual search
  rollouts;
- run a fresh confirmation cohort; or
- revise C13-M's completed verdict.

These exclusions keep the first study causal: it asks whether the same model
benefits from genuine query-level state under a fixed behavior target and a
fully specified dynamic-priority integration.
