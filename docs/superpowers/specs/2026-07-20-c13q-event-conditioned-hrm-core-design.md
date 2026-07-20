# C13-Q Event-Conditioned HRM-Core Design

**Date:** 2026-07-20
**Status:** Approved design; written-spec review pending
**Scope:** C13-Q stability, causal-memory, and identifiability study only
**Predecessor:** C13-P persistent search-state pilot

## 1. Decision

C13-Q will test whether stable state carried across planning events improves
frontier ranking after controlling for current information, temporal label
persistence, representation capacity, and numerical failure.

The primary method is a **reveal-only, fixed-slot, event-conditioned HRM-core
extension**. It reuses the parity-tested HRM post-normalized H/L reasoning core
and one-step-gradient schedule, but adds a new outer event scheduler that
changes the input after every search expansion while optionally retaining H/L
state. It is not an unchanged or end-to-end faithful implementation of the
official HRM ACT wrapper.

C13-Q is deliberately not an online utility or final publication-confirmation
study. A method may advance only through the following sequence:

1. **C13-Q:** stable causal state integration and shortcut-resistant offline
   ranking.
2. **C13-R:** separately specified on-policy or counterfactual target learning
   and matched online FOCAL/reopen integration.
3. **C13-S:** one frozen method on 144 untouched worlds, with expansions as the
   superiority endpoint and path quality as paired noninferiority.

No C13-Q result alone may be described as improving closed-loop planning.

## 2. Evidence motivating the design

C13-P is a valid negative result for its frozen implementation, not a general
rejection of search memory. Its persistent arm was numerically unstable:

- 913 of 1,642 development events contained at least one exact candidate tie;
- 348 events assigned exactly the same score to every candidate;
- persistent logits reached (1.57\times10^{20});
- CPU and GPU rankings diverged after recurrence amplified ordinary
  floating-point differences; and
- the shared recurrent context overwhelmed candidate-specific differences.

The implementation used
`continuous_prm_common.GatedRecurrentBlock`, a streaming gated RNN with an
unbounded returned state. It did not use the official HRM fixed-input H/L
refinement, post-normalized state, deep supervision, or ACT semantics.

A fresh audit of every existing C13-P trace established that the data can
support an exact event-sourced successor:

| Quantity | Value |
|---|---:|
| Worlds | 144 |
| Post-expansion events | 11,995 |
| Nodes per world | 192 |
| Newly opened transitions | 18,501 |
| Improved-g transitions | 3,480 |
| Reparentings | 3,480 |
| Maximum newly opened in one event | 15 |
| Maximum improved in one event | 5 |
| Delta-reconstruction violations | 0 |

Only the declared expanded node leaves the frontier; every new or improved
node's parent is the current expanded node; base ranks never change. These
facts hold only for the current strict-lower-g, no-reopen C13-M teacher.

The target also has a major shortcut risk. The positive path-frontier label is
unchanged from the preceding event in 66.61% of train, 67.57% of validation,
and 68.85% of development events. Every label change occurs when the previous
positive node is expanded. A persistent model can therefore appear useful by
retaining its preceding score. C13-Q must separate genuine switch-event
ranking from this temporal-smoothing baseline.

## 3. Research questions and permissible claims

### 3.1 Primary question

Given identical current candidate information, does stable H/L state carried
across causal search events improve ranking on events where the correct
path-frontier candidate changes?

### 3.2 Secondary questions

1. Is any benefit larger than a one-step previous-score persistence rule?
2. Is persistent state competitive with a separately optimized full-snapshot
   reset model that receives an explicit causal accumulator?
3. Is the effect specific to HRM-style H/L computation or shared by a
   parameter- and compute-matched Set+GRU model?
4. Does post-normalizing the legacy C13-P recurrence remove its observed
   numerical collapse without changing the representation?

### 3.3 Claim ladder

C13-Q may support only the strongest statement justified by its controls:

- **State necessity:** same-checkpoint persistent exceeds reset.
- **State beyond smoothing:** persistent also exceeds previous-score carry on
  switch events.
- **Efficient causal state reconstruction:** persistent is noninferior to the
  separately trained causal full-snapshot model.
- **HRM-specific evidence:** persistent HRM-core exceeds matched Set+GRU.
- **Architecture-neutral statefulness:** HRM-core and Set+GRU tie, while both
  exceed their controls.

An HRM-specific claim is forbidden if Set+GRU ties or wins. An online planning
claim is forbidden in every C13-Q outcome.

## 4. Scope and non-goals

C13-Q includes:

- the existing six-suite, 192-node C13-P train/validation/development worlds;
- exact causal trace replay and a new versioned tensor contract;
- reveal-only fixed-slot observations;
- HRM-core, snapshot-reset, temporal-persistence, Set+GRU, stabilized
  legacy, and frozen C13-M ranking controls;
- multi-seed discovery;
- crossed world/seed inference; and
- numerical, causal, device, and provenance gates.

C13-Q excludes:

- confirmation worlds;
- claims from C13-P development worlds;
- free-running model-guided search as a primary endpoint;
- FOCAL/reopen training traces;
- counterfactual continuation labels;
- selection after inspecting untouched worlds;
- full-map or all-node feature exposure in the primary arm; and
- changes to C13-M's frozen result.

C13-R will define reopened-node transitions, on-policy or counterfactual
labels, the bounded residual integration, and matched FOCAL/reopen search.
C13-S will define the final untouched confirmation.

## 5. Terminology and HRM fidelity boundary

The publication-safe name is:

> event-conditioned HRM-core fixed-slot streaming model

The following description is permitted:

> The model reuses the parity-tested HRM post-normalized H/L transformer
> blocks, H/L cycle schedule, detached one-step-gradient update, and
> continuous-input adapter, while adding a task-specific outer scheduler that
> carries terminal H/L state across changing search-event inputs.

The following descriptions are forbidden:

- “bit-exact HRM”;
- “unchanged HRM”;
- “faithful ACT across search events”; and
- “the official HRM architecture” without qualifying the event adapter.

The stock `HRMACTv1.forward` cannot implement this intervention. It retains
the prior `current_data` while a sample is active and resets H/L when a sample
halts. Cross-event persistence requires changing the input while retaining
H/L, so C13-Q must call the continuous-input inner reasoning module through a
new explicit event lifecycle.

## 6. Causal data contract

### 6.1 World-level provenance

Every trace binds:

- `world_id`, `split`, `suite`, `world_index`;
- `world_seed`, `roadmap_seed`;
- `node_count=192`, `edge_count`;
- `start_node`, `goal_node`, `side_len`;
- feature-cache path and SHA-256;
- graph SHA-256;
- frozen static-feature and encoder SHA-256 values;
- slot-order and permutation-audit SHA-256 values; and
- implementation and schema fingerprints.

Train, validation, and development must be disjoint by world seed, roadmap
seed, graph hash, and feature-cache hash.

### 6.2 Event fields

The model-causal event record contains:

- `event_index`;
- `expanded_node`, `expanded_g`, `expanded_base_rank`;
- complete current `frontier_nodes`, `frontier_g`,
  `frontier_base_rank`;
- derived and validated `newly_opened_nodes` and `improved_nodes`;
- `open_count`, `closed_count`; and
- a 192-element candidate mask.

The delta fields must be derived from consecutive causal snapshots and then
replayed independently. They are invalid if:

- a node other than `expanded_node` disappears;
- an intersecting frontier node changes without a strict g improvement;
- a base rank changes;
- an expanded node was not in the prior frontier, except the initial start
  expansion; or
- applying the delta does not reproduce the current snapshot exactly.

### 6.3 Supervision and audit isolation

The following remain physically separate from model inputs:

- `labels.positive_node`;
- final teacher path, path cost, and expansion count;
- graph shortest-path values;
- future events; and
- replay-only parent arrays.

Primary C13-Q must not promote `open_parent` into model-causal input. Parent
features are excluded because they can shortcut the final parent-chain label.
A later parent-on diagnostic requires a new schema, a parent-off paired
ablation, and an online-prefix construction test.

### 6.4 Prefix causality

For every event (t), the input produced from the completed trace must be
byte-identical to the input produced by executing only the live prefix through
event (t). The constructor must operate when label and privileged sections
are absent. Offline and live-prefix constructors must agree exactly.

## 7. Fixed-slot observation

### 7.1 Permutation-equivariant slots

Raw roadmap node IDs have no cross-world spatial meaning and are also used as
teacher tie-breaks. They are bookkeeping indices only and are never embedded
or converted into positional encodings.

The 192 model slots follow the current world's node-array order so H/L or GRU
state remains aligned with node identity across events. The model uses no
learned absolute position and no RoPE. Relabeling all raw nodes must permute
discrete input fields and state axes exactly. After inverse mapping, learned-arm
raw scores and residuals must agree within the larger of 32 float32 ULPs or
`1e-6` absolute-plus-relative tolerance, and final candidate ordering must be
identical. The identity ordering and every audit permutation are hashed into
provenance.

This choice deliberately avoids spatial sorting. Sorting all nodes by their
coordinates would reveal a node's rank relative to unopened geometry through
its slot position. Coordinates and static embeddings are emitted only after a
node crosses the reveal boundary. Start/goal role flags may be emitted before
reveal because the query endpoints are known, but they carry no coordinate or
local-feature payload while unseen.

### 7.2 Reveal boundary

At event (t), static node features are nonzero only for:

- the node expanded at (t); and
- every node in the current post-expansion frontier.

Unopened and previously unseen nodes receive a learned `UNSEEN` token plus a
false reveal flag. Previously revealed but currently closed nodes are not
re-emitted by the primary event tensor; only the persistent carry can retain
their history.

Exposing cached embeddings for all 192 nodes would combine bounded local
observations across the roadmap and approximate a global map. That
representation is excluded from the primary method and may appear only as an
explicit all-node upper control.

### 7.3 Per-slot fields

The primary tensor for slot (n) contains:

- revealed frozen 64-dimensional C13-M node embedding or `UNSEEN`;
- normalized spatial coordinates when revealed;
- start and goal flags;
- revealed, current-open, current-expanded, newly-opened, and improved flags;
- current (g/L), base-rank/(L), and ((g+rank)/L) for open candidates;
- expanded (g/L) and base-rank/(L) for the expanded slot; and
- an explicit validity mask.

Undefined scalars are zero only when accompanied by a false validity flag.

The primary tensor excludes:

- raw node IDs;
- raw parent IDs or parent embeddings;
- node age;
- event index or normalized clock;
- “time since opened”;
- final path membership; and
- any full-map field.

Age and clock are compressed history and would hand part of the memory task
to the reset arm. They may appear only in a named full-state diagnostic.

### 7.4 Equal current information

Persistent and same-checkpoint reset arms receive exactly the same event tensor
and direct candidate facts. Delta-only input is forbidden for the reset arm:
unchanged frontier candidates must retain their current embedding, g, base
rank, and open status in both arms. The sole intervention is the initial H/L
state for that event.

## 8. Models and controls

### 8.1 Q-A: persistent HRM-core

Q-A uses the continuous-input HRM inner module with:

- 192 permutation-equivariant bookkeeping slots;
- hidden size selected during smoke from the frozen grid `{64,128}`;
- one H layer and one L layer;
- two H cycles and two L cycles per event;
- no absolute position embedding and no RoPE;
- float32 forward computation;
- one fixed inner segment per event for the primary comparison; and
- detached terminal H/L carry passed to the next event.

A hidden-64 feasibility instance of the reused core returns one scalar per
slot, produces finite outputs, and keeps H/L RMS near one. Because the final
model removes positional embeddings, its parameter count is re-measured after
implementation rather than copied from that probe. The hidden-size choice is
frozen before discovery and the measured count is bound into every checkpoint.

Each event receives deep supervision through a frontier cross-entropy loss.
The carry is detached between event updates; no gradient crosses event
boundaries. Training batches interleave ordered world streams, and each
optimizer step supervises the current event for every active stream.

### 8.2 Q-A0: same-checkpoint reset intervention

Q-A0 uses Q-A's exact checkpoint and event tensor. At every event it replaces
the incoming H/L carry with the learned H/L initialization, executes the same
fixed cycles and one segment, and scores the same candidates. Event zero must
be byte-identical between Q-A and Q-A0.

This contrast proves only whether Q-A uses state. It cannot by itself prove
that persistence is better than a separately optimized current-state model.

### 8.3 Q-B: separately trained causal snapshot-reset HRM

Q-B uses the same HRM-core size and fixed per-event compute but resets H/L at
every event. A deterministic causal accumulator supplies the complete revealed
current state:

- revealed/open/closed status;
- current or last observed g and base rank;
- current frontier and expanded-node facts; and
- the same revealed static embeddings.

Q-B remains parent-off, age-off, and clock-off. It is trained independently
with the same train/validation worlds and seed ledger. It controls for reset
information starvation and is the closest official fixed-input HRM-semantics
control.

### 8.4 Q-C-H and Q-C-G: previous-score controls

Q-C-H starts from Q-A0's current-event scores, and Q-C-G starts from Q-D0's.
A candidate that survived unchanged from the preceding frontier retains its
preceding score; newly opened or improved candidates use the corresponding
fresh reset score; the expanded candidate is removed. There are no learned
recurrent parameters.

These controls measure how much the hindsight label's 67–69% persistence can
be exploited by temporal smoothing. Q-A must beat Q-C-H and Q-D must beat
Q-C-G on label-switch events.

### 8.5 Q-D: Set+GRU persistent control

Q-D receives the identical reveal-only tensor and candidate facts. A
permutation-equivariant set encoder aggregates the current revealed set, and a
per-slot GRU retains state across events. Parameter count must be within 10%
of Q-A; measured multiply-adds and event latency must be reported. If exact
parameter matching conflicts with the compute bound, compute matching takes
precedence and both differences are reported.

### 8.6 Q-D0: same-checkpoint Set+GRU reset intervention

Q-D0 resets every GRU state at each event while preserving Q-D's identical
current tensor, checkpoint, and compute. Q-D versus Q-D0 is required before
any architecture-neutral statefulness claim.

### 8.7 Q-D1: separately trained causal snapshot-reset Set control

Q-D1 receives the same complete revealed causal snapshot as Q-B, uses the
Set+GRU family's smoke-selected size and learning rate, and resets GRU state at
every event. It is trained independently on the identical worlds and seed
ledger. Q-D must be noninferior to Q-D1 before the Set+GRU family can advance.

### 8.8 Q-E: stabilized legacy recurrence

Q-E preserves C13-P's event and candidate representation while adding:

- post-state RMS normalization;
- a bounded output residual;
- the same stability instrumentation; and
- the same persistent/reset evaluation.

It is a discovery-only causal ablation. It cannot become the publication
method without passing the same shortcut and equal-information controls.

### 8.9 Q-F: frozen C13-M ranking baseline

Q-F ranks the current frontier by the unchanged C13-M priority with its frozen
deterministic tie rule. It has no learned parameters or state. Every offline
arm is reported against Q-F so that a persistent/reset contrast cannot conceal
absolute regression below the established current-state method. Q-F's raw-ID
tie rule remains frozen. Relabel-invariance checks apply to its numeric
priorities and ordering between unequal-priority groups, not to permutations
inside an exactly tied Q-F group.

### 8.10 ACT ablation

ACT is secondary. The primary comparison uses fixed computation because the
intervention must not change compute. An ACT cell may advance only after the
fixed model passes C13-Q. It repeats the same event input for up to four inner
segments, deep-supervises every segment, trains a correctness-based halt head,
and reports both quality and realized segment count. Any ACT comparison must
match or explicitly normalize total computation.

### 8.11 Frozen training recipe

All trainable primary and control arms use the same optimizer family and event
budget:

- AdamATan2 with betas `(0.9,0.95)` and weight decay `1e-4`;
- 100 optimizer-step linear warmup followed by constant learning rate;
- no gradient clipping; gradient norms are logged and nonfinite gradients fail
  the seed;
- 16 synchronized ordered world streams per optimizer step;
- deterministic world-stream shuffling per epoch;
- one detached event update and one deep-supervision loss per active stream;
- maximum 40 discovery epochs, minimum 12 epochs before stopping, and patience
  8 eligible epochs; and
- an epoch checkpoint written before validation.

The HRM and Set+GRU architecture families each receive the identical six-cell
smoke grid: hidden size `{64,128}` crossed with learning rate
`{1e-4,2e-4,5e-4}`, using the same three fixed seeds, at most six epochs, and
the same frozen smoke subset. Each family selects its own configuration by the
same lexicographic rule: all stability gates, number of stable seeds,
switch-event validation MRR, switch-event validation CE, then the smaller
hidden size and lower learning rate. Q-B inherits Q-A's selected HRM
configuration; Q-D1 inherits Q-D's selected Set+GRU configuration. Q-E retains
its fixed diagnostic recipe. No family receives a larger grid, more seeds, or
more epochs. Both selected recipes and compute budgets are frozen before
discovery.

## 9. Scoring and boundedness

The learned model cannot replace the C13-M priority outright. For candidate $i$:

$
f_i = g_i + r_i^{C13M},
\qquad
\delta_i = \kappa L\tanh(s_i),
\qquad
p_i = f_i + \delta_i .
$

Lower $p_i$ is better. Training frontier cross-entropy always uses
$-p_i/(0.05L)$ with the fixed training scale $\kappa_{train}=0.05$. After
training, validation jointly selects the checkpoint and evaluation scale from
$\kappa_{eval}\in\{0.02,0.05,0.10\}$; no checkpoint is retrained after this
selection. A family's selected $\kappa_{eval}$ is applied unchanged to its
persistent, same-checkpoint reset, temporal-persistence, and snapshot arms and
is locked before development evaluation.

The raw score $s_i$, bounded residual, final priority, and ULP margin are
logged separately. Online C13-R will use C13-M for anchor eligibility and the
learned residual only as a bounded secondary key; C13-Q does not establish
that online contract.

## 10. Target and training objective

C13-Q retains C13-P's stationary path-frontier positive to isolate
architecture, stability, and observation changes. It remains privileged
supervision and never enters runtime inputs.

Training events are divided into:

- **stay:** positive node equals the preceding event's positive node; and
- **switch:** positive node differs.

The training loss macro-averages stay-event and switch-event frontier
cross-entropy within each epoch, so the majority stay class cannot dominate
optimization. Event zero is reported separately and excluded from the
stay/switch contrast.

Checkpoint selection is lexicographic:

1. pass every validation stability and causality gate;
2. maximize switch-event qualified coverage;
3. maximize validation switch-event world-macro MRR;
4. minimize validation switch-event cross-entropy;
5. minimize maximum raw-score magnitude; and
6. choose the earliest epoch only as a final tie-break.

An unstable checkpoint is ineligible regardless of ranking quality.

## 11. Numerical and device eligibility

The stress suite evaluates real traces and deterministic synthetic traces at
prefix lengths `{1,8,16,32,48,64,96,128,192}`.

A checkpoint is ineligible if any stress event has:

- nonfinite input, H/L state, raw score, residual, or priority;
- revealed-token H/L RMS outside `[0.5,2.0]`;
- absolute raw score above `100`;
- a residual outside `[-kappa*L,+kappa*L]`;
- an accidental all-equal final candidate priority for frontier size above
  one;
- fewer than 16 ULPs between the top two final priorities; or
- different equality status, top candidate, or full candidate ordering
  between repeated executions on the same device.

CPU and deterministic GPU evaluation need not be bitwise equal. They must
produce identical candidate ordering on every stress event and identical top
candidate and equality status on every development event.

The qualification denominator is every post-expansion event with frontier
size above one in the evaluated split, including event zero. Coverage is
reported overall and per world for every learned arm. It must be at least 99%
overall and 95% in every validation world. An event is unqualified when its
top-two priority margin is below 16 ULPs or its cross-device top candidate or
equality status disagrees. Qualification and imputation are arm-specific. An
unqualified arm is assigned worst rank $K$, reciprocal rank $1/K$, top-1 zero,
and rank percentile one for that event. A qualified comparator retains its
observed metrics; if both arms are unqualified, both receive the conservative
imputation. Finite cross-entropy from the deterministic CPU reference is
retained as observed. This rule is used in checkpoint selection, development
gates, and raw-artifact reconstruction; unqualified events are never dropped.

## 12. Causality and shortcut gates

Before ranking metrics are read:

1. completed-trace and live-prefix tensors are byte-identical;
2. the constructor succeeds after labels and privileged audit fields are
   removed;
3. reveal-only masks never expose unseen cached embeddings;
4. raw node relabeling permutes tensors and state equivariantly and meets the
   score-tolerance and exact-ranking requirements in Section 7.1; Q-F numeric
   priorities and
   unequal-priority ordering are invariant while its frozen raw-ID tie groups
   are audited separately;
5. event zero is byte-identical between persistent and reset;
6. all event deltas replay exactly;
7. parent, age, and clock fields are absent from the primary input schema;
8. train, validation, and development provenance is disjoint; and
9. every checkpoint and raw output is bound to implementation, schema, data,
   model-state, optimizer, and seed hashes.

## 13. Experimental funnel

### 13.1 Smoke

Use three fixed model seeds and a small frozen train subset for each family's
exact hidden-size and learning-rate grid in Section 8.11. Run at most six
epochs and stop a seed at the first numerical, causality, or device failure.
Ranking cannot authorize a model that fails stability.

### 13.2 Discovery

Use eight fixed model seeds on the complete existing C13-P train split with
each family's smoke-selected configuration. Validation jointly selects the
checkpoint and $\kappa_{eval}$. Development is loaded once after all selections
are frozen.

The existing 24 development worlds are discovery evidence only. They cannot
be reused in C13-R selection or C13-S confirmation.

### 13.3 Inference

Worlds and model seeds are crossed random factors. The primary 20,000-resample
confidence interval independently resamples model seeds and worlds with frozen
RNG seeds, preserving all within-seed, within-world paired arms. Averaging
seeds and bootstrapping only worlds is forbidden unless a fixed multi-model
ensemble is explicitly declared as the deployed method and its multiplied
inference cost is included.

All primary arms use the same eight seed identifiers. For a paired comparison,
the eligible seed set is the intersection of seeds whose two arms pass G0-Q.
At least six paired seeds must remain; otherwise that comparison and its family
fail. A seed that fails one arm is excluded from that paired estimate, listed
with its failure reason, and cannot be replaced. Cross-architecture comparisons
pair the same seed identifiers. Each arm and each required pair must satisfy
the six-of-eight minimum, preventing favorable seed-subset selection.

For an MRR difference $d=method-comparator$ with crossed 95% interval
$[d_{low},d_{high}]$:

- superiority means $d_{low}>0$;
- practical superiority at threshold $m$ means point estimate $d\ge m$ and
  $d_{low}>0$;
- noninferiority with lower margin $-m$ means $d_{low}>-m$;
- comparator superiority means $d_{high}<0$; and
- equivalence margin $m$ means $d_{low}\ge-m$ and $d_{high}\le m$.

Suite results are secondary and interpreted after the pooled effect. No suite
may be dropped after inspection.

## 14. Gates and decision rules

### G0-Q: integrity, causality, and stability

G0-Q passes only if every requirement in Sections 11 and 12 passes. The HRM
family is `(persistent=Q-A, reset=Q-A0, temporal=Q-C-H, snapshot=Q-B)`; the
Set+GRU family is `(persistent=Q-D, reset=Q-D0, temporal=Q-C-G,
snapshot=Q-D1)`. Every arm and required pair in a passing family must retain
at least six of eight fixed seeds under Section 13.3.

### G1-Q: state beyond temporal smoothing

For each family, on development label-switch events:

- persistent minus reset world-macro MRR must show practical superiority at
  threshold `+0.02`;
- persistent minus its previous-score control must show superiority;
- persistent overall MRR must be noninferior to reset with margin `0.005`;
- persistent overall MRR must be noninferior to Q-F with margin `0.005`; and
- at least five of six suite switch-event persistent-minus-reset mean effects
  must be positive.

Stay-event, switch-event, qualified-coverage, and conservative-imputation
metrics are always reported separately. A family failing any line does not
advance even if its pooled unstratified MRR is favorable.

### G2-Q: equal-information and architecture attribution

- Each persistent arm must be noninferior to its separately trained causal
  snapshot arm on switch-event MRR with margin `0.01`.
- Let $d=MRR(Q-A)-MRR(Q-D)$ on switch events after both families pass G1-Q and
  their snapshot noninferiority gate:
  - $d_{low}>0$ permits an HRM-specific claim;
  - $d_{high}<0$ permits a Set+GRU-specific claim;
  - interval equivalence within `[-0.01,+0.01]` permits an
    architecture-neutral statefulness claim; and
  - every other interval is architecture-inconclusive and permits no
    attribution claim.
- Q-E is descriptive unless it independently satisfies the HRM family's G0-Q,
  G1-Q, and G2-Q controls.

### Advancement

C13-R opens if at least one family passes G0-Q, G1-Q, and snapshot
noninferiority. If only one family passes, that persistent arm advances. If
both pass and one is superior, the superior arm advances. If both are
equivalent or architecture-inconclusive, both enter C13-R as prespecified
candidate families and C13-R must freeze one before C13-S. No architecture
claim is made from an inconclusive interval, and no seed or suite is selected
after inspection.

If no model passes, C13-Q closes negatively. Confirmation remains untouched,
and the next design must change the target or information contract rather
than retuning development.

## 15. Required artifacts

C13-Q produces:

- frozen source and cohort audit;
- versioned causal trace dataset and duplicate-generation hashes;
- slot-order and permutation-equivariance ledger;
- model, optimizer, and parameter/compute manifest;
- frozen smoke-grid ledger and selection record;
- training history for every seed;
- checkpoint eligibility table;
- raw event-level ranking rows with stay/switch status;
- numerical and device diagnostics;
- crossed-bootstrap samples and summaries;
- gate verdict;
- independent raw-artifact reconstruction;
- integrity manifest covering code, tests, data, checkpoints, and reports;
- generated result report; and
- canonical claim-safe result document.

Raw rows retain all world, graph, feature-cache, model-seed, checkpoint, arm,
event, candidate, qualification, and timing identities required for
independent reconstruction.

## 16. Component boundaries

Implementation must keep these responsibilities separate:

1. **Trace contract:** schema, delta derivation, replay, reveal-only validation.
2. **Slot encoding:** bookkeeping-slot alignment, inverse mapping, tensor
   construction, permutation tests.
3. **Models:** HRM-core lifecycle, snapshot reset, Set+GRU, stabilized legacy.
4. **Training:** ordered streams, detached one-step updates, balanced
   stay/switch loss, checkpoint eligibility.
5. **Evaluation:** arm execution, temporal-persistence control, qualified
   ranking, crossed inference.
6. **Pipeline:** staged bindings, resumability, reports, integrity, independent
   verification.

No component may read labels or privileged audit data through a shared
catch-all payload.

## 17. Error handling

The pipeline fails closed on:

- schema or hash drift;
- missing or duplicate events/nodes/worlds;
- nonfinite values;
- causal-prefix disagreement;
- reveal-boundary violation;
- raw-ID-dependent learned output outside the explicit Q-F tie-rule exemption;
- checkpoint or optimizer mismatch;
- seed reuse across frozen roles;
- stability/device failure;
- malformed bootstrap cells;
- incomplete raw artifacts; or
- attempted development/confirmation execution before the preceding binding
  is frozen.

A failed stage writes no passing marker. Resumption requires an exact binding
match; otherwise the stage must start in a new output directory.

## 18. Test strategy

Tests must be written before implementation and must observe the intended
failure before production code is added.

Required focused tests cover:

- exact delta reconstruction over hand graphs and frozen trace fixtures;
- prefix-only construction with labels physically removed;
- reveal-only masking and all-node leak rejection;
- bookkeeping-slot preservation and arbitrary raw-node relabel invariance;
- candidate inverse mapping;
- HRM block/cycle parity against the vendored port for identical embeddings;
- changing event input while retaining explicit inner carry;
- persistent/reset event-zero identity;
- carry world-boundary and stale-carry rejection;
- bounded residual algebra;
- stay/switch classification and balanced loss;
- Q-C-H/Q-C-G previous-score control semantics;
- previous-score refresh on newly opened and improved candidates;
- Q-B and Q-D1 snapshot-reset information parity;
- Set+GRU parameter/compute audit and Q-D/Q-D0 state intervention;
- frozen C13-M ranking parity;
- stability and ULP qualification;
- CPU/CPU, GPU/GPU, and CPU/GPU ordering invariants;
- crossed world/seed bootstrap shape and pairing;
- exact gate-boundary behavior;
- stage binding, resume, independent reconstruction, and integrity coverage.

The final focused suite is followed by the existing C13 regression suite and
the HRM-v2 parity/training tests. Windows pytest-cache permission warnings are
reported separately from test failures.

## 19. Execution and user-alert boundary

Smoke tests and bounded pilot training are authorized by approval of this
design and its implementation plan. Before launching the first official
full-scale multi-seed discovery run, execution must stop and alert the user
with:

- the exact command;
- selected configuration and seed ledger;
- expected wall time and compute;
- fresh focused/regression test results; and
- confirmation that no untouched C13-S cohort will be opened.

C13-S requires a separate alert immediately before its untouched full run.

## 20. Success condition

C13-Q succeeds only when current artifacts prove that a numerically stable,
causal persistent model:

1. improves switch-event ranking over same-checkpoint reset;
2. improves over previous-score persistence;
3. remains competitive with a separately optimized causal snapshot model;
4. survives crossed world/seed inference and suite robustness;
5. passes every causal, reveal, permutation, and device gate; and
6. yields a frozen, claim-safe method choice for C13-R.

This is an advancement result, not the active thread's terminal success.
The overall goal remains incomplete until C13-R establishes closed-loop
utility and C13-S verifies one frozen method on untouched worlds.
