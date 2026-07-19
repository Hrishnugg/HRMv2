# C13-P Persistent Search-State HRM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Implement and execute the preregistered C13-P pilot that tests whether one HRM checkpoint benefits from query-level carry across the causal A* expansion sequence, first on stationary teacher traces and then in free-running dynamic-priority search.

**Architecture:** Add one isolated experiment module and one focused test module. Reuse the frozen C13-J flat-MLP encoder and C13-M local-Bellman base rank read-only; generate deterministic path-frontier traces; train a 64-wide fast/slow recurrent event core plus a full-open-set candidate scorer; compare persistent, reset, and unchanged C13-M arms; and derive all verdicts from preregistered world-clustered gates.

**Tech Stack:** Python 3, PyTorch, NumPy, pandas, SciPy, pytest, the existing continuous-PRM modules, canonical JSON/CSV artifacts, and Git.

**Frozen design:** docs/experiments/continuous/c13/design/2026-07-19-c13p-persistent-search-state.md

---

## Execution rules

- Treat the frozen design as authoritative. If this plan and the design disagree, stop and resolve the discrepancy before implementation.
- Work only in:
  - hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py
  - hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
  - hrm-cloud/continuous_prm/runs/c13_persistent_search/
  - docs/experiments/continuous/c13/results/C13P_PERSISTENT_SEARCH_RESULT.md
  - the explicitly named documentation indexes in Task 8
- Never edit, repair, or overwrite C13-J, C13-M, C13-N, or C13-O source artifacts.
- Preserve the existing dirty worktree. Before every commit, inspect git status; stage explicit task-owned paths only; never use git add -A, git add ., or a broad commit.
- Write each focused test first, run it, and observe the specified failure before adding production behavior.
- Do not load development traces until the selected checkpoint hash, evaluation-code hash, bootstrap seeds, and gate configuration have been written to the evaluation binding.
- Do not add self-bootstrap, on-policy relabeling, DAgger, reinforcement learning, a reset-specific model, a fresh confirmation cohort, or a fallback queue.
- Official evaluation runs on CPU. CUDA is required for the frozen training run.
- Use the detailed Section 5.3 event contract consistently in traces and learned online search: pop/close, relax, form the open set, construct and apply the event update, then score that complete open set. This makes Section 8.2's high-level list concrete without changing what information is causal.
- In reset mode, zero the low/high tensors before each event but initialize carry.step to that event's true zero-based event_index. Persistent and reset therefore execute the same k=2 high-level cadence; only remembered tensor values differ.
- Exact C13-M ordering is ascending (g + rank, g, node_id). Learned search orders by descending logit, then ascending (g + rank, g, node_id).
- Duplicate evaluation equality applies to deterministic decision/result fields. Timing fields are checked only for finite, nonnegative values.

## Frozen constants

Put these values in one immutable configuration object and serialize them into every stage binding:

~~~python
SCHEMA_VERSION = "c13p-v1"
MODEL_SEED = 18423
HIDDEN_DIM = 64
NUM_LAYERS = 1
NUM_HEADS = 4
K_STEP = 2
LOCAL_RADIUS = 0.20
LOCAL_ALPHA = 1.50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
MAX_EPOCHS = 20
PATIENCE = 4
TBPTT_EVENTS = 32
MAX_EXPANSIONS = 192
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEEDS = {
    "g1_mrr": 3789372949,
    "g2_exp_reset": 1177043361,
    "g2_exp_c13m": 580060237,
}
TRAIN_WORLDS = 96
VALIDATION_WORLDS = 24
DEVELOPMENT_WORLDS = 24
~~~

The three bootstrap seeds above are the uint32 states from the three children returned by np.random.SeedSequence(18423).spawn(3), in the displayed key order. Serialize the mapping exactly; do not use Python hash values or ambient RNG state.

---

### Task 1: Establish configuration, canonical serialization, and frozen-source audit

**Files:**

- Create: hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py
- Create: hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
- Read only: hrm-cloud/continuous_prm/runs/c13_lhbl_multisuite/
- Read only: hrm-cloud/continuous_prm/runs/c13_matched_quality_confirmation/

- [ ] **Step 1: Write failing tests for configuration and canonical artifacts**

Add tests that:

1. instantiate the default configuration and assert every frozen constant above;
2. assert canonical JSON recursively sorts keys, uses UTF-8, terminates with one newline, rejects NaN/Infinity, and produces identical bytes for differently ordered dictionaries;
3. mutate one byte in a temporary frozen input and assert hash verification fails without modifying the input;
4. point output inside a source directory and assert the disjointness guard fails;
5. provide a cohort record with a changed seed, node count, edge count, cache path, or cache hash and assert replay fails;
6. mark one cache as generated instead of reused and assert the audit hard-fails.

Use temporary fixtures only. Do not inspect the real development payload in these unit tests.

- [ ] **Step 2: Run the focused tests and confirm the import failure**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py -q
~~~

Expected: FAIL because continuous_prm_c13_persistent_search does not exist.

- [ ] **Step 3: Implement the configuration and audit primitives**

Define these public contracts:

~~~python
@dataclass(frozen=True)
class PersistentSearchConfig:
    repo_root: Path
    out_dir: Path
    schema_version: str = SCHEMA_VERSION
    model_seed: int = MODEL_SEED
    hidden_dim: int = HIDDEN_DIM
    num_layers: int = NUM_LAYERS
    num_heads: int = NUM_HEADS
    k_step: int = K_STEP
    local_radius: float = LOCAL_RADIUS
    local_alpha: float = LOCAL_ALPHA
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    grad_clip_norm: float = GRAD_CLIP_NORM
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    tbptt_events: int = TBPTT_EVENTS
    max_expansions: int = MAX_EXPANSIONS
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES

@dataclass(frozen=True)
class SourceContext:
    c13j_root: Path
    c13m_root: Path
    preregistration: Path
    implementation: Path
    source_manifest: Mapping[str, object]
    source_hashes: Mapping[str, str]
    cohort_records: Mapping[str, Sequence[Mapping[str, object]]]
    checkpoint_path: Path
    checkpoint_sha256: str

def resolve_paths(repo_root: Path, out_dir: Path | None = None) -> PersistentSearchConfig
def canonical_json_bytes(value: object) -> bytes
def write_canonical_json(path: Path, value: object) -> str
def sha256_file(path: Path) -> str
def verify_integrity_manifest(root: Path, manifest_path: Path) -> dict[str, str]
def assert_source_output_disjoint(source_roots: Sequence[Path], out_dir: Path) -> None
def replay_cohort_records(source: SourceContext) -> dict[str, list[dict[str, object]]]
def audit_sources(cfg: PersistentSearchConfig) -> SourceContext
~~~

Audit requirements:

- Resolve every path before overlap comparison.
- Rehash every C13-J integrity entry and the relevant C13-M evaluation fingerprint.
- Require the suite-balanced flat-MLP outer-iteration-8 checkpoint.
- Replay all 96/24/24 records, including world seed, roadmap seed, node count, edge count, cache path, and cache hash.
- Require every feature cache status to equal reused.
- Hash the implementation and preregistration.
- Return data in memory; Task 7 owns stage writes.
- Raise a specific ValueError naming the first mismatched field. Never mutate a frozen source.

- [ ] **Step 4: Run the Task 1 tests**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py -q -k "config or canonical or audit or disjoint or cohort"
~~~

Expected: PASS.

- [ ] **Step 5: Commit only the Task 1 files**

Run:

~~~powershell
git add -- hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
git commit --only -m "feat(c13p): add frozen source audit" -- hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
~~~

Expected: one commit containing only the two C13-P files.

---

### Task 2: Generate deterministic teacher search and path-frontier traces

**Files:**

- Modify: hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py
- Modify: hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
- Reuse: hrm-cloud/continuous_prm/continuous_prm_c13_lhbl_c7_comparison.py
- Reuse: hrm-cloud/continuous_prm/continuous_prm_c13_shared_queue.py

- [ ] **Step 1: Write failing hand-graph tests for teacher semantics**

Construct a six-node graph with:

- one returned teacher path;
- at least one attractive off-path branch;
- a relaxation that improves an open node before it is closed; and
- a deterministic priority tie.

Assert:

- heap order is exactly ascending (g + rank, g, node_id);
- the start is expansion zero;
- an event is recorded after pop/close and relaxation;
- the terminal goal pop creates no ranking event;
- each event label is the first not-yet-closed node on the returned parent-chain path;
- the label is unique, open, and not closed;
- recorded g and parent values match a second deterministic replay;
- a missing, duplicate, closed, or non-open positive raises ValueError;
- two serialized passes are byte-identical.

Also monkeypatch shortest-path/Dijkstra helpers to raise if trace generation calls them.

- [ ] **Step 2: Run the trace tests and confirm failure**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py -q -k "teacher or frontier or trace"
~~~

Expected: FAIL because the trace contracts are not implemented.

- [ ] **Step 3: Implement the trace data model**

Use immutable records:

~~~python
@dataclass(frozen=True)
class TraceEvent:
    event_index: int
    expanded_node: int
    expanded_g: float
    expanded_base_rank: float
    open_nodes: Sequence[int]
    open_g: Sequence[float]
    open_parent: Sequence[int | None]
    open_base_rank: Sequence[float]
    open_count: int
    closed_count: int
    positive_node: int

@dataclass(frozen=True)
class TeacherTrace:
    split: str
    suite: str
    world_index: int
    world_seed: int
    roadmap_seed: int
    feature_cache_path: str
    feature_cache_sha256: str
    node_count: int
    edge_count: int
    start_idx: int
    goal_idx: int
    events: Sequence[TraceEvent]
    teacher_path: Sequence[int]
    teacher_cost: float
    teacher_expansions: int
    teacher_valid: bool
~~~

Implement:

~~~python
def generate_teacher_trace(
    graph: Sequence[Sequence[tuple[int, float]]],
    start_idx: int,
    goal_idx: int,
    base_rank: np.ndarray,
    metadata: Mapping[str, object],
) -> TeacherTrace

def validate_teacher_trace(trace: TeacherTrace, graph: Sequence[Sequence[tuple[int, float]]]) -> None
def trace_payload(trace: TeacherTrace) -> dict[str, object]
def trace_from_payload(payload: Mapping[str, object]) -> TeacherTrace
def write_trace_shard(path: Path, traces: Sequence[TeacherTrace], generation_fingerprint: str) -> str
def read_trace_shard(path: Path, expected_fingerprint: str) -> Sequence[TeacherTrace]
~~~

Implementation rules:

1. Use direct no-reopen search: a closed node never re-enters.
2. Store g and parent snapshots after relaxation for each recorded open set.
3. Complete the search and reconstruct the returned parent-chain path before assigning labels.
4. For each event, walk the final path from start to goal and choose the first node not yet closed at that event.
5. Serialize model_causal, labels, replay_audit, and privileged_audit separately. Candidate parent snapshots belong in replay_audit; teacher path, future events, cost, validity, and graph-optimal values belong only in privileged_audit.
6. Canonically sort trace records by split, suite, world_index; retain open-node order only as an audited deterministic enumeration.
7. Hash the schema version, source hashes, cohort record, base-rank hash, and generator implementation into generation_fingerprint.

- [ ] **Step 4: Run trace tests twice**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py -q -k "teacher or frontier or trace"
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py -q -k "teacher or frontier or trace"
~~~

Expected: both runs PASS with the same collected test count.

- [ ] **Step 5: Commit the deterministic trace layer**

Run:

~~~powershell
git add -- hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
git commit --only -m "feat(c13p): add deterministic frontier traces" -- hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
~~~

---

### Task 3: Build the frozen representation and persistent HRM event model

**Files:**

- Modify: hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py
- Modify: hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
- Reuse: hrm-cloud/continuous_prm/continuous_prm_c13_identifiability.py
- Reuse: hrm-cloud/continuous_prm/continuous_prm_c13_lhbl_generated_v3.py
- Reuse: hrm-cloud/continuous_prm/continuous_prm_common.py

- [ ] **Step 1: Write failing representation and state-lifetime tests**

Add tests that assert:

- the C13-J iteration-8 FlatMLPRanker encoder yields shape (nodes, 64);
- all frozen encoder parameters have requires_grad false and remain in eval mode;
- the computed base rank equals euclidean_to_goal + 1.50 * (local_value_radius_0_20 - euclidean_to_goal);
- a clean carry contains zero low/high tensors and step zero;
- two persistent events change carry and advance step from zero to two;
- reset mode supplies zero low/high tensors with step equal to the event's true index, preserving the persistent arm's high-level update cadence;
- candidate scoring leaves every carry tensor and step unchanged;
- permuting candidate rows permutes logits exactly;
- persistent and reset modes use the same model object/state dict and parameter count;
- carry never crosses a world boundary or duplicate evaluation.

- [ ] **Step 2: Run the model tests and confirm failure**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py -q -k "encoder or base_rank or carry or candidate or checkpoint"
~~~

Expected: FAIL because model construction and carry operations are absent.

- [ ] **Step 3: Implement the frozen node representation**

Define:

~~~python
@dataclass(frozen=True)
class PreparedWorld:
    node_tokens: np.ndarray
    node_embeddings: np.ndarray
    euclidean_rank: np.ndarray
    local_values: np.ndarray
    base_rank: np.ndarray

def load_frozen_flat_encoder(source: SourceContext, device: torch.device) -> nn.Module
def prepare_world_representation(
    feature_cache: Mapping[str, np.ndarray],
    graph: Sequence[Sequence[tuple[int, float]]],
    goal_idx: int,
    cfg: PersistentSearchConfig,
    encoder: nn.Module,
) -> PreparedWorld
~~~

Load the existing FlatMLPRanker class and exact iteration-8 state dict, then expose its encoder output without copying or retraining weights. Validate the source state-dict keys, hidden width, token shape, checkpoint hash, and cache hash before inference.

- [ ] **Step 4: Implement explicit HRM carry and pure candidate scoring**

Define:

~~~python
@dataclass(frozen=True)
class HRMCarry:
    low: torch.Tensor
    high: torch.Tensor
    step: int

class PersistentSearchHRM(nn.Module):
    def initial_carry(self, batch_size: int, device: torch.device, dtype: torch.dtype, step: int = 0) -> HRMCarry
    def update_event(self, event_features: torch.Tensor, carry: HRMCarry) -> tuple[torch.Tensor, HRMCarry]
    def score_candidates(
        self,
        candidate_embeddings: torch.Tensor,
        context: torch.Tensor,
        candidate_scalars: torch.Tensor,
    ) -> torch.Tensor
~~~

Use this exact structure:

- event input width 70: expanded embedding 64 plus six normalized scalars;
- Linear(70, 64) event projection;
- one C.GatedRecurrentBlock(64, 4) high block;
- one C.GatedRecurrentBlock(64, 4) low block;
- update high when carry.step % 2 == 0, using carry.low.detach() as the input, matching DeepSapientHRMBackbone's fast-to-slow gradient boundary;
- reset calls initial_carry with step equal to the causal event index, so it does not receive extra high-level updates;
- update low every event from projected event plus current high state;
- return the new low state as context;
- candidate input width 131: embedding 64, context 64, and three normalized scalars;
- candidate head Linear(131, 64), GELU, Linear(64, 1);
- zero dropout and unbounded logits.

Initialize with torch.manual_seed(18423) inside a fork_rng context so construction is reproducible without perturbing ambient RNG state. The fast-to-slow detach above is architectural; training additionally detaches both carried tensors, without changing their values, only at TBPTT boundaries.

- [ ] **Step 5: Add a privileged-field rejection boundary**

Implement:

~~~python
ALLOWED_EVENT_KEYS = frozenset({
    "event_index", "expanded_node", "expanded_g", "expanded_base_rank",
    "open_count", "closed_count",
})
ALLOWED_CANDIDATE_KEYS = frozenset({
    "open_nodes", "open_g", "open_base_rank",
})
MODEL_CAUSAL_KEYS = ALLOWED_EVENT_KEYS | ALLOWED_CANDIDATE_KEYS
FORBIDDEN_INPUT_TOKENS = ("dist_to_goal", "dijkstra", "raster", "teacher_path", "future")

def validate_model_causal_fields(causal_event: Mapping[str, object]) -> None
def event_tensor_from_causal(
    causal_event: Mapping[str, object],
    node_embeddings: torch.Tensor,
    side_len: float,
    roadmap_nodes: int,
) -> torch.Tensor
def candidate_tensors_from_causal(
    causal_event: Mapping[str, object],
    node_embeddings: torch.Tensor,
    side_len: float,
) -> tuple[torch.Tensor, torch.Tensor, Sequence[int]]
~~~

Require the exact MODEL_CAUSAL_KEYS set before tensor construction. Tests must add positive_node, open_parent, a privileged_audit dictionary, and each forbidden token in turn and observe ValueError before any tensor is created.

- [ ] **Step 6: Run all Task 3 tests**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py -q -k "encoder or base_rank or carry or candidate or privileged or forbidden"
~~~

Expected: PASS.

- [ ] **Step 7: Commit the representation and model**

Run:

~~~powershell
git add -- hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
git commit --only -m "feat(c13p): add persistent HRM event model" -- hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
~~~

---

### Task 4: Implement stationary sequential training, TBPTT, and checkpoint selection

**Files:**

- Modify: hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py
- Modify: hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py

- [ ] **Step 1: Write failing objective and training-state tests**

Use tiny synthetic worlds and assert:

- frontier_cross_entropy uses all open candidates and the unique positive index;
- reported training loss is event-weighted;
- per-world MRR/top-1 are macro-averaged separately;
- one world stream begins with clean carry;
- a 33-event world takes two optimizer steps with TBPTT_EVENTS 32;
- at the chunk boundary, carry values are numerically preserved while low/high grad_fn are detached;
- the frozen encoder receives no gradients and remains unchanged after an optimizer step;
- epoch order is a deterministic permutation derived from seed 18423 and epoch number;
- validation loss alone selects the earliest minimum epoch;
- patience stops after four full epochs without strict improvement;
- resume accepts an exact binding and rejects any altered source, trace, model, optimizer, or gate field.

- [ ] **Step 2: Run the training tests and confirm failure**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py -q -k "loss or tbptt or train or validation or patience or resume"
~~~

Expected: FAIL because the trainer is not implemented.

- [ ] **Step 3: Implement event loss and deterministic metrics**

Define:

~~~python
def frontier_cross_entropy(logits: torch.Tensor, candidate_nodes: Sequence[int], positive_node: int) -> torch.Tensor
def rank_of_positive(logits: np.ndarray, candidate_nodes: np.ndarray, positive_node: int, tie_keys: np.ndarray) -> int
def summarize_trace_metrics(event_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]
~~~

For learned logits, sort by descending logit then ascending frozen (g + rank, g, node_id). Compute learned cross-entropy from the raw unbounded logits.
For the base arm, define descriptive logits as -(g + rank) / side_len, compute cross-entropy from those logits, and break exact ranking ties by ascending (g + rank, g, node_id).
Use one-based positive rank; reciprocal_rank = 1 / rank; top1 = 1 when rank == 1; and rank_percentile = (rank - 1) / max(candidate_count - 1, 1), so zero is best. Aggregate events within world before pooled or suite means.

- [ ] **Step 4: Implement the trainer and exact resume contract**

Define:

~~~python
@dataclass(frozen=True)
class CheckpointSelection:
    selected_epoch: int
    selected_validation_loss: float
    checkpoint_path: Path
    checkpoint_sha256: str

def detach_carry(carry: HRMCarry) -> HRMCarry
def deterministic_world_order(world_ids: Sequence[str], seed: int, epoch: int) -> Sequence[str]
def train_one_world(
    model: PersistentSearchHRM,
    trace: TeacherTrace,
    prepared: PreparedWorld,
    optimizer: torch.optim.Optimizer,
    cfg: PersistentSearchConfig,
) -> dict[str, float]
def evaluate_stationary_split(
    traces: Sequence[TeacherTrace],
    prepared_worlds: Mapping[str, PreparedWorld],
    model: PersistentSearchHRM,
    carry_mode: str,
    cfg: PersistentSearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]
def save_training_checkpoint(
    path: Path,
    model: PersistentSearchHRM,
    optimizer: torch.optim.Optimizer,
    completed_epoch: int,
    binding: Mapping[str, object],
) -> str
def load_training_checkpoint(
    path: Path,
    model: PersistentSearchHRM,
    optimizer: torch.optim.Optimizer,
    expected_binding: Mapping[str, object],
) -> int
def select_checkpoint(history: pd.DataFrame) -> CheckpointSelection
def train_stationary_model(
    train_traces: Sequence[TeacherTrace],
    validation_traces: Sequence[TeacherTrace],
    prepared_worlds: Mapping[str, PreparedWorld],
    cfg: PersistentSearchConfig,
    binding: Mapping[str, object],
) -> CheckpointSelection
~~~

Training behavior:

1. Require CUDA before the first optimizer/model allocation for the official train stage.
2. Create AdamW over only PersistentSearchHRM parameters with lr 5e-4 and weight decay 1e-4.
3. Shuffle whole training worlds deterministically per epoch; never shuffle events within a world.
4. Reset carry at each world boundary.
5. Accumulate mean event loss for at most 32 consecutive events, backpropagate, clip global norm at 1.0, step once, zero gradients, and detach carry values.
6. Run validation in persistent mode without gradients after each full epoch.
7. Select the earliest epoch attaining the numeric minimum validation event-weighted cross-entropy.
8. Save model, optimizer, completed epoch, Python/NumPy/Torch CPU/Torch CUDA RNG states, source hashes, trace hash, and complete binding fingerprint.
9. If a partial output exists with a different fingerprint, hard-fail and name a new output directory requirement. Never delete or silently restart it.

- [ ] **Step 5: Run Task 4 tests and a one-world CPU unit smoke**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py -q -k "loss or tbptt or train or validation or patience or resume"
~~~

Expected: PASS. Unit smoke may use CPU; it is not the official train stage.

- [ ] **Step 6: Commit training behavior**

Run:

~~~powershell
git add -- hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
git commit --only -m "feat(c13p): add stationary TBPTT training" -- hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
~~~

---

### Task 5: Implement offline ranking evaluation and G1-P

**Files:**

- Modify: hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py
- Modify: hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py

- [ ] **Step 1: Write failing matched-arm and boundary tests**

Build two short traces and one shared checkpoint. Assert:

- persistent and reset score the exact same recorded open sets;
- reset zeros low/high before every event while retaining the true event index for cadence, then executes the same update and scorer;
- both modes load an identical checkpoint hash and state dict;
- c13m_base_rank applies ascending (g + rank, g, node_id);
- event rows aggregate within world before suite and pooled metrics;
- the clustered bootstrap samples 24 world identifiers, not event rows;
- duplicate bootstrap calls with the frozen seed are identical;
- G1 fails when MRR CI lower equals zero;
- G1 fails when top-1 improvement is below 0.02 and passes exactly at 0.02;
- G1 requires strictly positive suite MRR differences in at least four of six suites.

- [ ] **Step 2: Run offline-evaluation tests and confirm failure**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py -q -k "offline or ranking or bootstrap or g1"
~~~

Expected: FAIL because arm evaluation and G1-P are absent.

- [ ] **Step 3: Implement stationary arm evaluation**

Define:

~~~python
OFFLINE_ARMS = ("c13p_persistent", "c13p_reset", "c13m_base_rank")

def evaluate_offline_arms(
    traces: Sequence[TeacherTrace],
    prepared_worlds: Mapping[str, PreparedWorld],
    model: PersistentSearchHRM,
    checkpoint_sha256: str,
    cfg: PersistentSearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]

def world_clustered_bootstrap(
    paired_world_rows: pd.DataFrame,
    value_column: str,
    resamples: int,
    seed: int,
) -> dict[str, float]
~~~

Raw rows must include split, suite, world_index, event_index, arm, positive_node, candidate_count, cross_entropy, positive_rank, reciprocal_rank, top1, and rank_percentile. Summary rows must expose world, suite, and pooled levels without allowing longer traces to dominate the primary inference.

For each named paired comparison, initialize np.random.default_rng with its bound BOOTSTRAP_SEEDS entry, sample the 24 world rows with replacement into an array of shape (20_000, 24), and compute one paired mean per resample.
Compute the 95% interval with np.quantile(resampled_means, [0.025, 0.975], method="linear"). Store unrounded endpoints and use those unrounded values in gates.

- [ ] **Step 4: Implement an explicit G1 verdict**

Define:

~~~python
def g1_verdict(world_metrics: pd.DataFrame, bootstrap_seed: int, resamples: int) -> dict[str, object]
~~~

Return each primitive comparison and:

~~~python
passes = (
    pooled_mrr_ci_low > 0.0
    and pooled_top1_delta >= 0.02
    and suites_with_positive_mrr >= 4
)
verdict = "c13p_g1_passed" if passes else "c13p_no_persistent_ranking_signal"
~~~

Do not infer G1 from rounded display values.

- [ ] **Step 5: Run Task 5 tests**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py -q -k "offline or ranking or bootstrap or g1"
~~~

Expected: PASS.

- [ ] **Step 6: Commit offline evaluation**

Run:

~~~powershell
git add -- hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
git commit --only -m "feat(c13p): add offline persistence gate" -- hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
~~~

---

### Task 6: Implement dynamic full-open-set search and G2-P

**Files:**

- Modify: hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py
- Modify: hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
- Reuse: hrm-cloud/continuous_prm/continuous_prm_c13_lhbl_c7_comparison.py
- Reuse: hrm-cloud/continuous_prm/continuous_prm_c13_shared_queue.py

- [ ] **Step 1: Write failing dynamic-search tests**

Use a graph where the learned scorer changes preference after a second event. Assert:

- start expands first;
- each nonterminal expansion updates carry exactly once after relaxation and before scoring;
- every currently open node is rescored after every expansion;
- no old score remains in the next queue;
- candidate enumeration permutations yield the same selected node;
- learned exact ties use lower base (g + rank), then lower g, then lower node id;
- reset and persistent use the same checkpoint, event index, update cadence, and compute rule, differing only in incoming low/high tensors;
- closed nodes cannot re-enter and each node expands at most once;
- path reconstruction is valid under history-dependent priorities;
- search stops on goal pop, empty open set, or 192 expansions;
- scorer calls and candidates scored match the actual full-open-set operations;
- duplicate runs have identical deterministic rows;
- timing values are finite and nonnegative but need not match.

- [ ] **Step 2: Run dynamic-search tests and confirm failure**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py -q -k "dynamic or search or rescore or tie or timing"
~~~

Expected: FAIL because history-conditioned search is not implemented.

- [ ] **Step 3: Implement dynamic learned search**

Define:

~~~python
@dataclass(frozen=True)
class SearchResult:
    arm: str
    path: Sequence[int]
    valid: bool
    cost: float
    optimal_cost: float
    cost_ratio: float
    expansions: int
    expanded_nodes: Sequence[int]
    scorer_calls: int
    candidates_scored: int
    representation_seconds: float
    model_seconds: float
    bookkeeping_seconds: float

def dynamic_best_first(
    graph: Sequence[Sequence[tuple[int, float]]],
    prepared: PreparedWorld,
    start_idx: int,
    goal_idx: int,
    model: PersistentSearchHRM,
    carry_mode: str,
    cfg: PersistentSearchConfig,
) -> SearchResult

def static_c13m_search(
    graph: Sequence[Sequence[tuple[int, float]]],
    prepared: PreparedWorld,
    start_idx: int,
    goal_idx: int,
    cfg: PersistentSearchConfig,
) -> SearchResult
def deterministic_result_projection(rows: pd.DataFrame) -> pd.DataFrame
def validate_timing_columns(rows: pd.DataFrame) -> None
~~~

Per learned expansion:

1. pop the selected open node and close it;
2. if it is the goal, reconstruct and return without a candidate event;
3. relax outgoing edges under the frozen direct no-reopen rule;
4. build event features from the expanded node and post-relaxation counts;
5. choose persistent incoming carry or a newly initialized reset carry;
6. update the HRM once;
7. vectorize all current open candidates in node-id-sorted enumeration;
8. score all candidates in one call;
9. rebuild priority entries from (-logit, g + base_rank, g, node_id).

For unchanged C13-M, use the existing direct no-reopen implementation or an equality-tested extraction of it. Its priority is static ascending (g + base_rank, g, node_id). Compute graph-optimal cost only after search for evaluation bookkeeping; never pass it to trace/model/search functions.

- [ ] **Step 4: Implement G2 and overall verdicts**

Define:

~~~python
ONLINE_ARMS = ("c13p_persistent", "c13p_reset", "c13m_base")

def evaluate_online_arms(
    worlds: Sequence[Mapping[str, object]],
    prepared_worlds: Mapping[str, PreparedWorld],
    model: PersistentSearchHRM,
    cfg: PersistentSearchConfig,
) -> pd.DataFrame
def g2_verdict(search_rows: pd.DataFrame, bootstrap_seed: int, resamples: int) -> dict[str, object]
def overall_verdict(g0: Mapping[str, object], g1: Mapping[str, object], g2: Mapping[str, object]) -> str
~~~

G2 passes only when all seven frozen conditions hold:

1. all 72 arm-world searches return valid paths;
2. persistent-minus-reset expansion CI upper endpoint is below zero;
3. persistent-minus-C13-M expansion CI upper endpoint is below zero;
4. at least four of six suite expansion means are negative versus reset;
5. at least four of six suite expansion means are negative versus C13-M;
6. persistent pooled mean cost ratio is at most C13-M paired mean plus 0.005;
7. persistent maximum cost ratio is at most C13-M paired maximum plus 0.02.

Boundary tests must prove: CI upper equal to zero fails; mean margin equal to 0.005 passes; max margin equal to 0.02 passes.

Overall verdict mapping:

- G0 fail: c13p_invalid_no_mechanism_verdict
- G0 pass, G1 fail: c13p_no_persistent_ranking_signal
- G0/G1 pass, G2 fail: c13p_offline_signal_failed_free_running_search
- G0/G1/G2 pass: c13p_persistent_search_pilot_passed

- [ ] **Step 5: Run Task 6 tests**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py -q -k "dynamic or search or rescore or tie or timing or g2 or overall"
~~~

Expected: PASS.

- [ ] **Step 6: Commit dynamic search and gates**

Run:

~~~powershell
git add -- hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
git commit --only -m "feat(c13p): add dynamic search evaluation" -- hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
~~~

---

### Task 7: Add staged orchestration, bindings, verification, and generated reports

**Files:**

- Modify: hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py
- Modify: hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py

- [ ] **Step 1: Write failing stage-integrity tests**

In temporary directories, assert:

- audit writes manifest.json and source_audit.json atomically;
- trace refuses to run without the exact audit fingerprint;
- train refuses to run without byte-identical duplicate train/validation trace hashes;
- development trace files are absent and unopened before evaluation_binding.json exists;
- evaluation binding contains selected checkpoint hash, evaluation implementation hash, bootstrap seeds, gate thresholds, and source/trace hashes;
- develop rejects any post-binding code, config, checkpoint, or source drift;
- report derives prose from gate_verdict.json and cannot override its verdict;
- a conflicting partial stage hard-fails instead of deleting or mixing outputs;
- the integrity manifest covers every canonical artifact in the design;
- duplicate deterministic result projections are byte-identical;
- all expected 96/24/24 worlds appear exactly once.

- [ ] **Step 2: Run orchestration tests and confirm failure**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py -q -k "stage or binding or integrity or report or atomic"
~~~

Expected: FAIL because stage orchestration is absent.

- [ ] **Step 3: Implement stage bindings and atomic writers**

Define:

~~~python
STAGES = ("audit", "trace", "smoke", "train", "develop", "report")

def atomic_write_bytes(path: Path, data: bytes) -> None
def stage_binding(cfg: PersistentSearchConfig, stage: str, inputs: Mapping[str, str]) -> dict[str, object]
def require_stage_binding(path: Path, expected: Mapping[str, object]) -> None
def run_audit_stage(cfg: PersistentSearchConfig) -> None
def run_trace_stage(cfg: PersistentSearchConfig) -> None
def run_smoke_stage(cfg: PersistentSearchConfig) -> None
def run_train_stage(cfg: PersistentSearchConfig) -> None
def run_develop_stage(cfg: PersistentSearchConfig) -> None
def run_report_stage(cfg: PersistentSearchConfig) -> None
def run_pipeline(cfg: PersistentSearchConfig, requested_stage: str | None) -> None
~~~

Write to a sibling temporary file, flush and close it, then replace the destination. A completed stage is immutable: rerunning verifies and returns; mismatches hard-fail. With no requested stage, run only the earliest incomplete stage.

Trace stage:

- generates train and validation twice into separate temporary trees;
- requires equal canonical shard bytes and manifest hashes;
- promotes one copy only after equality;
- does not resolve, read, or hash development trace payloads.

Train stage:

- trains on CUDA;
- freezes the validation-selected checkpoint;
- writes checkpoint_selection.json;
- hashes the evaluation implementation and exact gate payload;
- writes evaluation_binding.json before development is available.

Develop stage:

- verifies evaluation binding first;
- generates/replays development traces twice;
- performs offline and online evaluation twice on CPU;
- compares deterministic projections byte-for-byte;
- checks timing columns separately;
- computes G0, then G1, then G2 without short-circuiting raw descriptive outputs;
- writes raw CSVs, summaries, gate_verdict.json, and verification.json.

- [ ] **Step 4: Implement CLI and report generation**

Use:

~~~python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace
def main(argv: Sequence[str] | None = None) -> int

if __name__ == "__main__":
    raise SystemExit(main())
~~~

Supported commands:

~~~text
--stage audit|trace|smoke|train|develop|report|full
--out-dir PATH
--verify-only
~~~

full runs each missing stage in order but preserves every hard stop. verify-only rehashes all bindings/artifacts, checks counts and duplicate projections, and never trains or rewrites artifacts.

Generate runs/c13_persistent_search/results/C13P_RESULT.md from computed JSON. It must include:

- frozen source/checkpoint/trace fingerprints;
- exact cohort and event counts;
- checkpoint selection and training history;
- pooled and six-suite offline metrics;
- G1 primitive comparisons;
- pooled and six-suite online metrics;
- G2 primitive comparisons and quality margins;
- G0 checks;
- final verdict and claim-safe interpretation;
- an explicit statement that there was no self-bootstrap or confirmation.

- [ ] **Step 5: Run the full focused suite**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py -q
~~~

Expected: PASS.

- [ ] **Step 6: Run neighboring regression tests and compilation**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_hrm_substitution.py hrm-cloud/continuous_prm/tests/test_c13_hrm_alignment.py -q
python -m py_compile hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
~~~

Expected: both commands PASS with no C13-P-attributable warnings.

- [ ] **Step 7: Commit orchestration**

Run:

~~~powershell
git add -- hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
git commit --only -m "feat(c13p): add preregistered pipeline harness" -- hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
~~~

---

### Task 8: Execute the frozen pilot, verify evidence, and synchronize documentation

**Files:**

- Create artifacts under: hrm-cloud/continuous_prm/runs/c13_persistent_search/
- Create: docs/experiments/continuous/c13/results/C13P_PERSISTENT_SEARCH_RESULT.md
- Conditionally modify after preserving existing user edits:
  - docs/experiments/README.md
  - docs/experiments/MASTER_EXPERIMENT_SYNTHESIS.md
  - README.md

- [ ] **Step 1: Record the pre-run source state**

Run:

~~~powershell
git status --short
git rev-parse HEAD
git diff --check
python hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py --stage audit
~~~

Expected:

- existing unrelated changes remain untouched;
- audit either passes and writes only C13-P output, or hard-stops with an exact frozen-source mismatch;
- no training/development artifact exists yet.

- [ ] **Step 2: Generate duplicate train/validation traces**

Run:

~~~powershell
python hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py --stage trace
python hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py --verify-only
~~~

Expected:

- 96 training worlds and 24 validation worlds exactly once;
- every cache marked reused;
- duplicate trace hashes equal;
- no development trace payload has been opened or written.

- [ ] **Step 3: Run the locked smoke and training stages**

Run:

~~~powershell
python hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py --stage smoke
nvidia-smi
python hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py --stage train
~~~

Expected:

- smoke passes without changing configuration;
- CUDA is available;
- training uses the 96 fixed worlds, at most 20 epochs, patience four, and TBPTT 32;
- the earliest minimum validation-loss checkpoint is selected;
- evaluation_binding.json exists before any development trace load.

If CUDA is unavailable, stop and report the mechanical blocker. Do not substitute CPU training or alter the plan.

- [ ] **Step 4: Run duplicate official development evaluation**

Run:

~~~powershell
python hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py --stage develop
python hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py --stage report
python hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py --verify-only
~~~

Expected:

- 24 development worlds, four from each of six suites;
- persistent/reset share the selected checkpoint;
- three offline arms and three online arms are complete;
- all deterministic duplicate rows match;
- timing fields are finite/nonnegative;
- returned paths validate;
- gate_verdict.json contains G0-P, G1-P, G2-P primitives and one frozen final verdict;
- integrity.json covers every canonical artifact.

- [ ] **Step 5: Independently recheck the verdict from raw artifacts**

Run a read-only verification that:

1. recomputes world-level metrics from development_ranking_raw.csv;
2. recomputes path validity, paired expansion differences, and cost-ratio margins from development_search_raw.csv;
3. reruns the 20,000-resample clustered bootstrap using the bound seed;
4. compares every primitive to gate_verdict.json without display rounding;
5. verifies no confirmation, bootstrap-label, or self-training artifact exists.

Add this independent result to verification.json under independent_reanalysis and rerun verify-only. If the independent result differs, G0-P fails; do not hand-edit the verdict.

- [ ] **Step 6: Create the canonical result document**

Copy computed facts, tables, hashes, and verdict from the generated report into docs/experiments/continuous/c13/results/C13P_PERSISTENT_SEARCH_RESULT.md. Add a short interpretation selected by the actual outcome:

- G0 fail: mechanical invalidity only;
- G1 fail: no evidence that this target/model uses persistent state;
- G1 pass/G2 fail: offline memory signal with teacher-forcing or policy-integration failure;
- full pass: persistent search-state pilot passed, authorizing only a separately preregistered on-policy study and untouched confirmation.

Never describe C13-P as bounded A*, map-free navigation, a wall-clock speedup, or a general HRM result.

- [ ] **Step 7: Synchronize indexes without absorbing user-owned edits**

Before editing each existing index, inspect its current diff and HEAD version. Add only:

- a link to the C13-P design, plan, and canonical result;
- the exact frozen final verdict;
- one claim-safe sentence explaining whether the result is a persistent-state mechanism signal, an offline-only signal, a null, or invalid.

If an index contains overlapping uncommitted user changes, preserve them and leave the C13-P hunk uncommitted for explicit review. Do not use a whole-file targeted commit that would capture unrelated changes.

- [ ] **Step 8: Run final verification**

Run:

~~~powershell
python -m pytest hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_hrm_substitution.py hrm-cloud/continuous_prm/tests/test_c13_hrm_alignment.py -q
python -m py_compile hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py hrm-cloud/continuous_prm/tests/test_c13_persistent_search.py
python hrm-cloud/continuous_prm/continuous_prm_c13_persistent_search.py --verify-only
git diff --check
git status --short
~~~

Expected: all tests and verification pass; only known user-owned changes plus explicit C13-P documentation changes remain.

- [ ] **Step 9: Commit task-owned evidence safely**

Stage only new C13-P result/artifact paths allowed by repository policy. Do not commit large checkpoints or ignored run outputs unless the repository's existing experiment-artifact policy explicitly tracks them.

Run for the canonical result:

~~~powershell
git add -- docs/experiments/continuous/c13/results/C13P_PERSISTENT_SEARCH_RESULT.md
git commit --only -m "docs(c13p): record persistent search-state verdict" -- docs/experiments/continuous/c13/results/C13P_PERSISTENT_SEARCH_RESULT.md
~~~

Commit clean index files separately only if their diffs contain no pre-existing user work. Otherwise report their exact uncommitted C13-P hunks in the handoff.

- [ ] **Step 10: Report the evidence, not just completion**

The final handoff must state:

- exact source and selected-checkpoint hashes;
- train/validation/development world and event counts;
- selected epoch and validation loss;
- G0/G1/G2 primitive values and final verdict;
- test counts and commands;
- artifact and canonical-result paths;
- whether any index change remains uncommitted due to the dirty worktree;
- confirmation that no live worker remains and no self-bootstrap or confirmation run occurred.

---

## Completion definition

C13-P is complete only when:

1. every focused and neighboring regression test passes;
2. the frozen source audit and duplicate trace checks pass;
3. training and evaluation bindings match the selected checkpoint and implementation;
4. duplicate official deterministic result fields match and timing values validate;
5. raw results independently reproduce G0-P, G1-P, G2-P, and the final verdict;
6. the integrity manifest verifies;
7. the canonical result document is synchronized with generated evidence; and
8. all implementation/training processes have exited.

Passing the pilot is not part of the completion definition. A valid null or offline-only verdict is a completed preregistered study.
