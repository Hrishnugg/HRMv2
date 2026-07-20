"""Frozen-source bindings for the preregistered C13-P persistent-search pilot."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


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


def resolve_paths(repo_root: Path, out_dir: Path | None = None) -> PersistentSearchConfig:
    """Return the immutable C13-P configuration rooted at ``repo_root``."""
    resolved_root = Path(repo_root).resolve()
    resolved_out = (
        Path(out_dir).resolve()
        if out_dir is not None
        else (resolved_root / "hrm-cloud" / "continuous_prm" / "runs" / "c13_persistent_search").resolve()
    )
    return PersistentSearchConfig(repo_root=resolved_root, out_dir=resolved_out)


def _reject_non_finite(value: object) -> None:
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("canonical JSON rejects NaN")
        if math.isinf(value):
            raise ValueError("canonical JSON rejects Infinity")
    elif isinstance(value, Mapping):
        for child in value.values():
            _reject_non_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_non_finite(child)


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON deterministically as UTF-8 with exactly one trailing newline."""
    _reject_non_finite(value)
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"canonical JSON serialization failed: {exc}") from exc
    return (text + "\n").encode("utf-8")


def write_canonical_json(path: Path, value: object) -> str:
    """Write a canonical JSON artifact and return its SHA-256 digest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(value))
    return sha256_file(destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_frozen_source_path(root: Path, raw_path: object, label: str) -> Path:
    """Resolve a frozen path against the configured source snapshot, never its stale origin."""
    source_root = Path(root).resolve()
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path is missing")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        resolved = (source_root / candidate).resolve()
        if not resolved.is_relative_to(source_root):
            raise ValueError(f"{label}.path escapes configured source root: {raw_path}")
        return resolved
    candidate = candidate.resolve()
    if candidate.is_relative_to(source_root):
        return candidate
    source_anchor = (source_root.parent.name.casefold(), source_root.name.casefold())
    candidate_parts = tuple(part.casefold() for part in candidate.parts)
    matches = [index for index in range(1, len(candidate_parts)) if candidate_parts[index - 1:index + 1] == source_anchor]
    if len(matches) == 1:
        suffix = candidate.parts[matches[0] + 1:]
        rebased = (source_root / Path(*suffix)).resolve()
        if rebased.is_relative_to(source_root):
            return rebased
    continuous_anchor = ("hrm-cloud", "continuous_prm")
    configured_parts = tuple(part.casefold() for part in source_root.parts)
    configured_matches = [index for index in range(1, len(configured_parts)) if configured_parts[index - 1:index + 1] == continuous_anchor]
    candidate_matches = [index for index in range(1, len(candidate_parts)) if candidate_parts[index - 1:index + 1] == continuous_anchor]
    if len(configured_matches) == 1 and len(candidate_matches) == 1:
        continuous_root = Path(*source_root.parts[:configured_matches[0] + 1])
        rebased = (continuous_root / Path(*candidate.parts[candidate_matches[0] + 1:])).resolve()
        if rebased.is_relative_to(continuous_root):
            return rebased
    if len(configured_matches) == 1:
        repo_root = Path(*source_root.parts[:configured_matches[0] - 1])
        doc_matches = [index for index, part in enumerate(candidate_parts) if part == "docs"]
        if len(doc_matches) == 1:
            rebased = (repo_root / Path(*candidate.parts[doc_matches[0]:])).resolve()
            if rebased.is_relative_to(repo_root):
                return rebased
    raise ValueError(f"{label}.path is outside configured frozen source root: {raw_path}")

def _manifest_entry_path(root: Path, entry: Mapping[str, object], label: str) -> Path:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path is missing")
    candidate = Path(raw_path)
    return _resolve_frozen_source_path(root, raw_path, label)


def verify_integrity_manifest(root: Path, manifest_path: Path) -> dict[str, str]:
    """Rehash every named manifest entry without modifying any source artifact."""
    resolved_root = Path(root).resolve()
    payload = _read_json(Path(manifest_path).resolve(), "integrity manifest")
    verified: dict[str, str] = {}
    for section in ("inputs", "outputs"):
        entries = payload.get(section, {})
        if not isinstance(entries, Mapping):
            raise ValueError(f"integrity manifest {section} must be an object")
        for name, raw_entry in entries.items():
            label = str(name)
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"{label} integrity entry must be an object")
            expected = raw_entry.get("sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"{label}.sha256 is invalid")
            candidate = _manifest_entry_path(resolved_root, raw_entry, label)
            if not candidate.is_file():
                raise ValueError(f"{label} is missing: {candidate}")
            observed = sha256_file(candidate)
            if observed != expected:
                raise ValueError(f"{label} SHA-256 mismatch")
            if label in verified and verified[label] != observed:
                raise ValueError(f"{label} has conflicting integrity entries")
            verified[label] = observed
    return verified


def assert_source_output_disjoint(source_roots: Sequence[Path], out_dir: Path) -> None:
    """Reject any source/output nesting after resolving every supplied path."""
    resolved_out = Path(out_dir).resolve()
    for source_root in source_roots:
        resolved_source = Path(source_root).resolve()
        if resolved_out == resolved_source or resolved_out.is_relative_to(resolved_source) or resolved_source.is_relative_to(resolved_out):
            raise ValueError(f"output directory overlaps frozen source: {resolved_out}")


_COHORT_FIELDS = ("world_seed", "roadmap_seed", "nodes", "edges", "cache", "cache_sha256")


def _cohort_reference(source: SourceContext) -> Mapping[str, Sequence[Mapping[str, object]]] | None:
    raw_reference = source.source_manifest.get("cohort_records")
    if raw_reference is None:
        return None
    if not isinstance(raw_reference, Mapping):
        raise ValueError("cohort_records reference must be an object")
    return raw_reference  # type: ignore[return-value]


def replay_cohort_records(source: SourceContext) -> dict[str, list[dict[str, object]]]:
    """Validate and return the frozen C13-J train/validation/development cohorts."""
    reference = _cohort_reference(source)
    replayed: dict[str, list[dict[str, object]]] = {}
    for split in ("train", "validation", "development"):
        records = source.cohort_records.get(split)
        if not isinstance(records, Sequence):
            raise ValueError(f"{split} cohort records are missing")
        expected_records = reference.get(split) if reference is not None else None
        if expected_records is not None and (not isinstance(expected_records, Sequence) or len(records) != len(expected_records)):
            raise ValueError(f"{split} cohort record count changed")
        replayed[split] = []
        for index, raw_record in enumerate(records):
            if not isinstance(raw_record, Mapping):
                raise ValueError(f"{split}[{index}] cohort record must be an object")
            expected_record = expected_records[index] if expected_records is not None else None
            if expected_record is not None and not isinstance(expected_record, Mapping):
                raise ValueError(f"{split}[{index}] frozen cohort record must be an object")
            for field in _COHORT_FIELDS:
                if field not in raw_record:
                    raise ValueError(f"{split}[{index}].{field} is missing")
                if expected_record is not None and raw_record[field] != expected_record.get(field):
                    raise ValueError(f"{split}[{index}].{field} changed")
            if raw_record.get("cache_status") != "reused":
                raise ValueError(f"{split}[{index}].cache_status must equal reused")
            cache_path = _resolve_frozen_source_path(source.c13j_root, raw_record["cache"], f"{split}[{index}].cache")
            if not cache_path.is_file():
                raise ValueError(f"{split}[{index}].cache is missing")
            if sha256_file(cache_path) != raw_record["cache_sha256"]:
                raise ValueError(f"{split}[{index}].cache_sha256 mismatch")
            normalized_record = dict(raw_record)
            normalized_record["cache"] = str(cache_path)
            replayed[split].append(normalized_record)
    return replayed


def _require_checkpoint(c13j_root: Path, integrity: Mapping[str, str]) -> tuple[Path, str]:
    checkpoint = (c13j_root / "checkpoints" / "flat_mlp_iteration_08.pt").resolve()
    if not checkpoint.is_file():
        raise ValueError("suite-balanced flat-MLP outer-iteration-8 checkpoint is missing")
    expected = integrity.get("checkpoint_08")
    observed = sha256_file(checkpoint)
    if expected != observed:
        raise ValueError("suite-balanced flat-MLP outer-iteration-8 checkpoint hash mismatch")
    return checkpoint, observed


def audit_sources(cfg: PersistentSearchConfig) -> SourceContext:
    """Audit C13-J/C13-M source snapshots and build an in-memory frozen context."""
    repo_root = Path(cfg.repo_root).resolve()
    continuous_root = repo_root / "hrm-cloud" / "continuous_prm"
    c13j_root = (continuous_root / "runs" / "c13_lhbl_multisuite").resolve()
    c13m_root = (continuous_root / "runs" / "c13_matched_quality_confirmation").resolve()
    preregistration = (repo_root / "docs" / "experiments" / "continuous" / "c13" / "design" / "2026-07-19-c13p-persistent-search-state.md").resolve()
    implementation = Path(__file__).resolve()
    assert_source_output_disjoint((c13j_root, c13m_root), cfg.out_dir)

    c13j_integrity = verify_integrity_manifest(c13j_root, c13j_root / "integrity.json")
    c13m_integrity = verify_integrity_manifest(c13m_root, c13m_root / "integrity.json")
    fingerprint = (c13m_root / "results" / "evaluation_fingerprint.json").resolve()
    if not fingerprint.is_file() or c13m_integrity.get("fingerprint") != sha256_file(fingerprint):
        raise ValueError("C13-M evaluation_fingerprint changed")
    checkpoint_path, checkpoint_sha256 = _require_checkpoint(c13j_root, c13j_integrity)

    cohorts_payload = _read_json(c13j_root / "results" / "cohorts.json", "C13-J cohorts")
    raw_records = cohorts_payload.get("records")
    if not isinstance(raw_records, Mapping):
        raise ValueError("C13-J cohort records are missing")
    source_manifest = dict(_read_json(c13j_root / "manifest.json", "C13-J manifest"))
    source_manifest["cohort_records"] = copy.deepcopy(raw_records)
    provisional = SourceContext(
        c13j_root=c13j_root,
        c13m_root=c13m_root,
        preregistration=preregistration,
        implementation=implementation,
        source_manifest=source_manifest,
        source_hashes={},
        cohort_records=raw_records,  # type: ignore[arg-type]
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )
    replayed = replay_cohort_records(provisional)
    expected_counts = {"train": TRAIN_WORLDS, "validation": VALIDATION_WORLDS, "development": DEVELOPMENT_WORLDS}
    for split, expected_count in expected_counts.items():
        if len(replayed[split]) != expected_count:
            raise ValueError(f"{split} cohort record count changed")
    if not preregistration.is_file():
        raise ValueError("preregistration is missing")

    source_hashes = {
        "c13j_integrity": sha256_file(c13j_root / "integrity.json"),
        "c13m_evaluation_fingerprint": sha256_file(fingerprint),
        "c13j_cohorts": sha256_file(c13j_root / "results" / "cohorts.json"),
        "checkpoint": checkpoint_sha256,
        "preregistration": sha256_file(preregistration),
        "c13j_manifest": sha256_file(c13j_root / "manifest.json"),
        "implementation": sha256_file(implementation),
    }
    return SourceContext(
        c13j_root=c13j_root,
        c13m_root=c13m_root,
        preregistration=preregistration,
        implementation=implementation,
        source_manifest=source_manifest,
        source_hashes=source_hashes,
        cohort_records=replayed,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )


@dataclass(frozen=True)
class TraceEvent:
    event_index: int
    event_kind: str
    expanded_node: int | None
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


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _trace_metadata(metadata: Mapping[str, object]) -> tuple[str, str, int, int, int, str, str]:
    required = ("split", "suite", "world_index", "world_seed", "roadmap_seed")
    if any(name not in metadata for name in required):
        raise ValueError("trace metadata is incomplete")
    cache_path = metadata.get("feature_cache_path", metadata.get("cache"))
    cache_sha256 = metadata.get("feature_cache_sha256", metadata.get("cache_sha256"))
    if not isinstance(metadata["split"], str) or not metadata["split"]:
        raise ValueError("trace split is invalid")
    if not isinstance(metadata["suite"], str) or not metadata["suite"]:
        raise ValueError("trace suite is invalid")
    if not isinstance(cache_path, str) or not cache_path:
        raise ValueError("trace feature_cache_path is invalid")
    if not isinstance(cache_sha256, str) or len(cache_sha256) != 64:
        raise ValueError("trace feature_cache_sha256 is invalid")
    return (
        metadata["split"], metadata["suite"], _integer(metadata["world_index"], "world_index"),
        _integer(metadata["world_seed"], "world_seed"), _integer(metadata["roadmap_seed"], "roadmap_seed"),
        cache_path, cache_sha256,
    )


def _validated_graph(graph: Sequence[Sequence[tuple[int, float]]], base_rank: object) -> tuple[int, tuple[float, ...]]:
    node_count = len(graph)
    if node_count == 0:
        raise ValueError("teacher graph is empty")
    try:
        ranks = tuple(_finite_float(value, "base_rank") for value in base_rank)  # type: ignore[union-attr]
    except TypeError as exc:
        raise ValueError("base_rank must be a one-dimensional sequence") from exc
    if len(ranks) != node_count:
        raise ValueError("base_rank length does not match graph")
    for source, outgoing in enumerate(graph):
        for target, weight in outgoing:
            if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < node_count:
                raise ValueError(f"graph edge {source} has an invalid target")
            edge_weight = _finite_float(weight, f"graph edge {source}->{target}")
            if edge_weight < 0.0:
                raise ValueError("teacher search requires nonnegative edge weights")
    return node_count, ranks


def _path_from_parent(parent: Sequence[int | None], start_idx: int, goal_idx: int) -> tuple[int, ...]:
    reversed_path: list[int] = []
    node: int | None = goal_idx
    while node is not None:
        reversed_path.append(node)
        if node == start_idx:
            return tuple(reversed(reversed_path))
        node = parent[node]
        if len(reversed_path) > len(parent):
            break
    raise ValueError("teacher returned an invalid parent chain")


def generate_teacher_trace(
    graph: Sequence[Sequence[tuple[int, float]]],
    start_idx: int,
    goal_idx: int,
    base_rank: Sequence[float],
    metadata: Mapping[str, object],
) -> TeacherTrace:
    """Run the static C13-M direct no-reopen teacher without an optimal-path oracle."""
    import heapq

    node_count, ranks = _validated_graph(graph, base_rank)
    if not isinstance(start_idx, int) or not isinstance(goal_idx, int) or not 0 <= start_idx < node_count or not 0 <= goal_idx < node_count:
        raise ValueError("teacher start or goal is outside the graph")
    split, suite, world_index, world_seed, roadmap_seed, cache_path, cache_sha256 = _trace_metadata(metadata)
    g = [math.inf] * node_count
    parent: list[int | None] = [None] * node_count
    g[start_idx] = 0.0
    open_nodes = {start_idx}
    closed: set[int] = set()
    heap: list[tuple[float, float, int]] = [(ranks[start_idx], 0.0, start_idx)]
    raw_events: list[tuple[int, str, int | None, float, float, tuple[int, ...], tuple[float, ...], tuple[int | None, ...], tuple[float, ...], int, frozenset[int]]] = [
        (0, "initialized_frontier", None, 0.0, ranks[start_idx], (start_idx,), (0.0,), (None,), (ranks[start_idx],), 0, frozenset()),
    ]
    expansions = 0
    reached_goal = False

    while heap:
        _, popped_g, node = heapq.heappop(heap)
        if node in closed or node not in open_nodes or popped_g != g[node]:
            continue
        open_nodes.remove(node)
        closed.add(node)
        expansions += 1
        if node == goal_idx:
            reached_goal = True
            break
        for neighbor, raw_weight in graph[node]:
            weight = float(raw_weight)
            if neighbor in closed:
                continue
            candidate_g = g[node] + weight
            if candidate_g < g[neighbor]:
                g[neighbor] = candidate_g
                parent[neighbor] = node
                open_nodes.add(neighbor)
                heapq.heappush(heap, (candidate_g + ranks[neighbor], candidate_g, neighbor))
        ordered_open = tuple(sorted(open_nodes, key=lambda candidate: (g[candidate] + ranks[candidate], g[candidate], candidate)))
        raw_events.append((
            len(raw_events), "post_expansion", node, g[node], ranks[node], ordered_open,
            tuple(g[candidate] for candidate in ordered_open),
            tuple(parent[candidate] for candidate in ordered_open),
            tuple(ranks[candidate] for candidate in ordered_open), len(closed), frozenset(closed),
        ))

    if not reached_goal:
        raise ValueError("teacher did not return a valid path")
    teacher_path = _path_from_parent(parent, start_idx, goal_idx)
    events: list[TraceEvent] = []
    for event_index, event_kind, node, expanded_g, expanded_rank, candidates, candidate_g, candidate_parent, candidate_rank, closed_count, closed_snapshot in raw_events:
        positive = next((path_node for path_node in teacher_path if path_node not in closed_snapshot), None)
        if positive is None or positive not in candidates:
            raise ValueError("teacher event has no open path-frontier positive")
        events.append(TraceEvent(
            event_index=event_index, event_kind=event_kind, expanded_node=node, expanded_g=expanded_g, expanded_base_rank=expanded_rank,
            open_nodes=candidates, open_g=candidate_g, open_parent=candidate_parent, open_base_rank=candidate_rank,
            open_count=len(candidates), closed_count=closed_count, positive_node=positive,
        ))
    trace = TeacherTrace(
        split=split, suite=suite, world_index=world_index, world_seed=world_seed, roadmap_seed=roadmap_seed,
        feature_cache_path=cache_path, feature_cache_sha256=cache_sha256, node_count=node_count,
        edge_count=sum(len(outgoing) for outgoing in graph), start_idx=start_idx, goal_idx=goal_idx,
        events=tuple(events), teacher_path=teacher_path, teacher_cost=g[goal_idx],
        teacher_expansions=expansions, teacher_valid=True,
    )
    validate_teacher_trace(trace, graph)
    return trace


def _event_maps(event: TraceEvent) -> tuple[dict[int, float], dict[int, int | None], dict[int, float]]:
    lengths = (len(event.open_nodes), len(event.open_g), len(event.open_parent), len(event.open_base_rank))
    if any(length != event.open_count for length in lengths):
        raise ValueError("trace event open snapshot lengths are invalid")
    if event.open_count <= 0:
        raise ValueError("trace event open set is empty")
    if len(set(event.open_nodes)) != event.open_count:
        raise ValueError("trace event contains duplicate open nodes")
    candidate_g = {node: _finite_float(value, "trace open_g") for node, value in zip(event.open_nodes, event.open_g)}
    parent = {node: value for node, value in zip(event.open_nodes, event.open_parent)}
    rank = {node: _finite_float(value, "trace open_base_rank") for node, value in zip(event.open_nodes, event.open_base_rank)}
    return candidate_g, parent, rank


def validate_teacher_trace(trace: TeacherTrace, graph: Sequence[Sequence[tuple[int, float]]]) -> None:
    """Validate per-event frontier labels and exact direct-search replay."""
    if trace.node_count != len(graph) or trace.node_count <= 0:
        raise ValueError("trace node_count does not match graph")
    if trace.edge_count != sum(len(outgoing) for outgoing in graph):
        raise ValueError("trace edge_count does not match graph")
    if not trace.teacher_valid:
        raise ValueError("teacher trace is not valid")
    if not (0 <= trace.start_idx < trace.node_count and 0 <= trace.goal_idx < trace.node_count):
        raise ValueError("trace start or goal is invalid")
    if not trace.teacher_path or tuple(trace.teacher_path)[0] != trace.start_idx or tuple(trace.teacher_path)[-1] != trace.goal_idx:
        raise ValueError("trace teacher path is invalid")
    if not trace.events:
        raise ValueError("trace is missing its initialized frontier")

    initial = trace.events[0]
    initial_g, initial_parent, initial_rank = _event_maps(initial)
    if initial.event_index != 0 or initial.event_kind != "initialized_frontier" or initial.expanded_node is not None:
        raise ValueError("trace initialized frontier is invalid")
    if initial.closed_count != 0 or initial.open_count != 1 or set(initial_g) != {trace.start_idx}:
        raise ValueError("trace initialized frontier changed")
    if initial_parent[trace.start_idx] is not None or not math.isclose(initial_g[trace.start_idx], 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("trace initialized frontier replay changed")
    if not math.isclose(initial.expanded_g, 0.0, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(initial.expanded_base_rank, initial_rank[trace.start_idx], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("trace initialized frontier scalars changed")
    if initial.positive_node != trace.start_idx:
        raise ValueError("trace initialized frontier positive is invalid")

    open_g: dict[int, float] = {trace.start_idx: 0.0}
    open_parent: dict[int, int | None] = {trace.start_idx: None}
    open_rank: dict[int, float] = {trace.start_idx: initial_rank[trace.start_idx]}
    replayed_g: dict[int, float] = {trace.start_idx: 0.0}
    replayed_parent: dict[int, int | None] = {trace.start_idx: None}
    closed: set[int] = set()

    for expected_index, event in enumerate(trace.events[1:], start=1):
        if event.event_index != expected_index:
            raise ValueError("trace event index is invalid")
        event_g, event_parent, event_rank = _event_maps(event)
        if event.expanded_node == trace.goal_idx:
            raise ValueError("trace terminal goal pop must not be recorded")
        if event.expanded_node not in open_g:
            raise ValueError("trace replay expanded node is not open")
        priority_node = min(open_g, key=lambda node: (open_g[node] + open_rank[node], open_g[node], node))
        if event.expanded_node != priority_node:
            raise ValueError("trace replay heap order changed")
        if not math.isclose(event.expanded_g, open_g[event.expanded_node], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("trace replay expanded g changed")
        if not math.isclose(event.expanded_base_rank, open_rank[event.expanded_node], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("trace replay expanded rank changed")
        open_g.pop(event.expanded_node)
        open_parent.pop(event.expanded_node)
        open_rank.pop(event.expanded_node)
        closed.add(event.expanded_node)
        if event.closed_count != len(closed):
            raise ValueError("trace closed count changed")
        for neighbor, raw_weight in graph[event.expanded_node]:
            weight = _finite_float(raw_weight, "graph edge weight")
            if weight < 0.0:
                raise ValueError("teacher search requires nonnegative edge weights")
            if neighbor in closed:
                continue
            candidate_g = event.expanded_g + weight
            if candidate_g < open_g.get(neighbor, math.inf):
                if neighbor not in event_g:
                    raise ValueError("trace replay is missing an open candidate")
                open_g[neighbor] = candidate_g
                open_parent[neighbor] = event.expanded_node
                open_rank[neighbor] = event_rank[neighbor]
                replayed_g[neighbor] = candidate_g
                replayed_parent[neighbor] = event.expanded_node
        if set(event_g) != set(open_g):
            raise ValueError("trace replay open candidates changed")
        expected_nodes = tuple(sorted(open_g, key=lambda node: (open_g[node] + open_rank[node], open_g[node], node)))
        if tuple(event.open_nodes) != expected_nodes:
            raise ValueError("trace replay open order changed")
        for node in expected_nodes:
            if not math.isclose(event_g[node], open_g[node], rel_tol=0.0, abs_tol=1e-12) or event_parent[node] != open_parent[node]:
                raise ValueError("trace replay candidate g or parent changed")
            if not math.isclose(event_rank[node], open_rank[node], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("trace replay candidate rank changed")
        if not isinstance(event.positive_node, int):
            raise ValueError("trace positive is missing")
        if event.positive_node in closed:
            raise ValueError("trace positive is already closed")
        if event.positive_node not in event_g:
            raise ValueError("trace positive is not open")
        expected_positive = next((node for node in trace.teacher_path if node not in closed), None)
        if event.positive_node != expected_positive:
            raise ValueError("trace positive does not match the path frontier")

    if trace.teacher_expansions != len(trace.events):
        raise ValueError("trace teacher expansion count changed")
    if trace.goal_idx not in open_g:
        raise ValueError("trace terminal goal is not open")
    final_node = min(open_g, key=lambda node: (open_g[node] + open_rank[node], open_g[node], node))
    if final_node != trace.goal_idx:
        raise ValueError("trace terminal goal pop changed")
    chain_reversed: list[int] = []
    node: int | None = trace.goal_idx
    while node is not None:
        chain_reversed.append(node)
        if node == trace.start_idx:
            break
        node = replayed_parent.get(node)
        if len(chain_reversed) > trace.node_count:
            raise ValueError("trace replay parent chain cycles")
    if not chain_reversed or chain_reversed[-1] != trace.start_idx:
        raise ValueError("trace replay parent chain is incomplete")
    replayed_path = tuple(reversed(chain_reversed))
    if replayed_path != tuple(trace.teacher_path):
        raise ValueError("trace replay parent chain does not match privileged teacher path")
    replayed_cost = 0.0
    for previous, current in zip(replayed_path, replayed_path[1:]):
        if previous not in replayed_g or current not in replayed_g:
            raise ValueError("trace replay parent chain g is missing")
        increment = replayed_g[current] - replayed_g[previous]
        if not any(math.isclose(float(weight), increment, rel_tol=0.0, abs_tol=1e-12) for target, weight in graph[previous] if target == current):
            raise ValueError("trace replay parent chain edge cost changed")
        replayed_cost += increment
    if not math.isclose(replayed_cost, replayed_g[trace.goal_idx], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("trace replay parent chain cost changed")
    if not math.isclose(replayed_cost, _finite_float(trace.teacher_cost, "teacher_cost"), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("trace teacher cost does not match replayed parent-chain path")
def _causal_static_fields(trace: TeacherTrace) -> dict[str, object]:
    return {
        "split": trace.split, "suite": trace.suite, "world_index": trace.world_index,
        "world_seed": trace.world_seed, "roadmap_seed": trace.roadmap_seed,
        "feature_cache_path": trace.feature_cache_path, "feature_cache_sha256": trace.feature_cache_sha256,
        "node_count": trace.node_count, "edge_count": trace.edge_count,
        "start_idx": trace.start_idx, "goal_idx": trace.goal_idx,
    }


def trace_payload(trace: TeacherTrace) -> dict[str, object]:
    """Serialize one isolated model-causal record per event and keep teacher data privileged."""
    static = _causal_static_fields(trace)
    examples: list[dict[str, object]] = []
    for event in trace.events:
        current_event = {
            "event_index": event.event_index, "event_kind": event.event_kind,
            "expanded_node": event.expanded_node, "expanded_g": event.expanded_g,
            "expanded_base_rank": event.expanded_base_rank, "open_nodes": list(event.open_nodes),
            "open_g": list(event.open_g), "open_base_rank": list(event.open_base_rank),
            "open_count": event.open_count, "closed_count": event.closed_count,
        }
        examples.append({
            "model_causal": {**static, "event": current_event},
            "labels": {"positive_node": event.positive_node},
            "replay_audit": {"open_parent": list(event.open_parent)},
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "examples": examples,
        "privileged_audit": {
            "teacher_path": list(trace.teacher_path), "teacher_cost": trace.teacher_cost,
            "teacher_expansions": trace.teacher_expansions, "teacher_valid": trace.teacher_valid,
        },
    }


def trace_from_payload(payload: Mapping[str, object]) -> TeacherTrace:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("trace payload schema version changed")
    examples = payload.get("examples")
    privileged = payload.get("privileged_audit")
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)) or not examples or not isinstance(privileged, Mapping):
        raise ValueError("trace payload examples are invalid")
    static: dict[str, object] | None = None
    events: list[TraceEvent] = []
    static_names = ("split", "suite", "world_index", "world_seed", "roadmap_seed", "feature_cache_path", "feature_cache_sha256", "node_count", "edge_count", "start_idx", "goal_idx")
    for raw_example in examples:
        if not isinstance(raw_example, Mapping):
            raise ValueError("trace payload example is invalid")
        causal = raw_example.get("model_causal")
        labels = raw_example.get("labels")
        replay = raw_example.get("replay_audit")
        if not isinstance(causal, Mapping) or not isinstance(labels, Mapping) or not isinstance(replay, Mapping):
            raise ValueError("trace payload example sections are invalid")
        raw_event = causal.get("event")
        raw_parent = replay.get("open_parent")
        if not isinstance(raw_event, Mapping) or not isinstance(raw_parent, Sequence) or isinstance(raw_parent, (str, bytes)):
            raise ValueError("trace payload event is invalid")
        current_static = {name: causal.get(name) for name in static_names}
        if static is None:
            static = current_static
        elif current_static != static:
            raise ValueError("trace payload static fields changed across examples")
        fields = ("open_nodes", "open_g", "open_base_rank")
        if any(not isinstance(raw_event.get(field), Sequence) or isinstance(raw_event.get(field), (str, bytes)) for field in fields):
            raise ValueError("trace payload candidate fields are invalid")
        event_kind = raw_event.get("event_kind")
        if not isinstance(event_kind, str):
            raise ValueError("trace event_kind is invalid")
        raw_expanded = raw_event.get("expanded_node")
        if raw_expanded is not None and (isinstance(raw_expanded, bool) or not isinstance(raw_expanded, int)):
            raise ValueError("trace expanded_node is invalid")
        events.append(TraceEvent(
            event_index=_integer(raw_event.get("event_index"), "event_index"), event_kind=event_kind,
            expanded_node=raw_expanded, expanded_g=_finite_float(raw_event.get("expanded_g"), "expanded_g"),
            expanded_base_rank=_finite_float(raw_event.get("expanded_base_rank"), "expanded_base_rank"),
            open_nodes=tuple(_integer(value, "open_node") for value in raw_event["open_nodes"]),
            open_g=tuple(_finite_float(value, "open_g") for value in raw_event["open_g"]),
            open_parent=tuple(None if value is None else _integer(value, "open_parent") for value in raw_parent),
            open_base_rank=tuple(_finite_float(value, "open_base_rank") for value in raw_event["open_base_rank"]),
            open_count=_integer(raw_event.get("open_count"), "open_count"),
            closed_count=_integer(raw_event.get("closed_count"), "closed_count"),
            positive_node=_integer(labels.get("positive_node"), "positive_node"),
        ))
    if static is None or not isinstance(static["split"], str) or not isinstance(static["suite"], str) or not isinstance(static["feature_cache_path"], str) or not isinstance(static["feature_cache_sha256"], str):
        raise ValueError("trace payload metadata is invalid")
    teacher_path = privileged.get("teacher_path")
    if not isinstance(teacher_path, Sequence) or isinstance(teacher_path, (str, bytes)):
        raise ValueError("trace payload teacher path is invalid")
    return TeacherTrace(
        split=static["split"], suite=static["suite"], world_index=_integer(static["world_index"], "world_index"),
        world_seed=_integer(static["world_seed"], "world_seed"), roadmap_seed=_integer(static["roadmap_seed"], "roadmap_seed"),
        feature_cache_path=static["feature_cache_path"], feature_cache_sha256=static["feature_cache_sha256"],
        node_count=_integer(static["node_count"], "node_count"), edge_count=_integer(static["edge_count"], "edge_count"),
        start_idx=_integer(static["start_idx"], "start_idx"), goal_idx=_integer(static["goal_idx"], "goal_idx"),
        events=tuple(events), teacher_path=tuple(_integer(value, "teacher_path node") for value in teacher_path),
        teacher_cost=_finite_float(privileged.get("teacher_cost"), "teacher_cost"),
        teacher_expansions=_integer(privileged.get("teacher_expansions"), "teacher_expansions"),
        teacher_valid=privileged.get("teacher_valid") is True,
    )
def trace_generation_fingerprint(
    source_hashes: Mapping[str, str], cohort_record: Mapping[str, object], base_rank: Sequence[float],
) -> str:
    """Bind trace generation to frozen sources, cohort, ranks, schema, and generator code."""
    rank_values = [_finite_float(value, "base_rank") for value in base_rank]
    binding = {
        "schema_version": SCHEMA_VERSION, "source_hashes": dict(source_hashes),
        "cohort_record": dict(cohort_record),
        "base_rank_sha256": hashlib.sha256(canonical_json_bytes(rank_values)).hexdigest(),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    return hashlib.sha256(canonical_json_bytes(binding)).hexdigest()


def write_trace_shard(path: Path, traces: Sequence[TeacherTrace], generation_fingerprint: str) -> str:
    if not isinstance(generation_fingerprint, str) or len(generation_fingerprint) != 64:
        raise ValueError("generation fingerprint is invalid")
    ordered = sorted(traces, key=lambda trace: (trace.split, trace.suite, trace.world_index))
    if len({(trace.split, trace.suite, trace.world_index) for trace in ordered}) != len(ordered):
        raise ValueError("trace shard contains duplicate world records")
    payload = {"schema_version": SCHEMA_VERSION, "generation_fingerprint": generation_fingerprint, "traces": [trace_payload(trace) for trace in ordered]}
    return write_canonical_json(Path(path), payload)


def read_trace_shard(path: Path, expected_fingerprint: str) -> Sequence[TeacherTrace]:
    payload = _read_json(Path(path), "trace shard")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("generation_fingerprint") != expected_fingerprint:
        raise ValueError("trace shard fingerprint or schema changed")
    raw_traces = payload.get("traces")
    if not isinstance(raw_traces, Sequence) or isinstance(raw_traces, (str, bytes)):
        raise ValueError("trace shard records are invalid")
    traces = tuple(trace_from_payload(raw) for raw in raw_traces if isinstance(raw, Mapping))
    if len(traces) != len(raw_traces):
        raise ValueError("trace shard contains a non-object trace")
    if tuple(sorted(traces, key=lambda trace: (trace.split, trace.suite, trace.world_index))) != traces:
        raise ValueError("trace shard records are not canonical")
    return traces
