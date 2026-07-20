"""Frozen-source bindings for the preregistered C13-P persistent-search pilot."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import torch
import pandas as pd
from torch import nn

import continuous_prm_c13_identifiability as I
import continuous_prm_common as C


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
TRAINING_BINDING_SCHEMA_VERSION = "c13p-training-binding-v1"


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
        if event.event_kind != "post_expansion" or event.expanded_node is None:
            raise ValueError("trace event_kind must be post-expansion")
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
        if event_kind not in ("initialized_frontier", "post_expansion"):
            raise ValueError("trace event_kind is invalid")
        raw_expanded = raw_event.get("expanded_node")
        if raw_expanded is not None and (isinstance(raw_expanded, bool) or not isinstance(raw_expanded, int)):
            raise ValueError("trace expanded_node is invalid")
        event_index = _integer(raw_event.get("event_index"), "event_index")
        if event_index == 0:
            if event_kind != "initialized_frontier" or raw_expanded is not None:
                raise ValueError("trace initialized frontier event_kind is invalid")
        elif event_kind != "post_expansion" or raw_expanded is None:
            raise ValueError("trace event_kind must be post-expansion")
        events.append(TraceEvent(
            event_index=event_index, event_kind=event_kind,
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

@dataclass(frozen=True)
class PreparedWorld:
    node_tokens: np.ndarray; node_embeddings: np.ndarray; euclidean_rank: np.ndarray; local_values: np.ndarray; base_rank: np.ndarray

def load_frozen_flat_encoder(source: SourceContext, device: torch.device) -> nn.Module:
    if sha256_file(source.checkpoint_path) != source.checkpoint_sha256: raise ValueError("frozen iteration-8 checkpoint hash mismatch")
    payload = torch.load(source.checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("model_name") != "flat_mlp" or payload.get("iteration") != 8: raise ValueError("checkpoint is not flat-MLP iteration-8")
    state, cfg, lhbl = payload.get("model"), payload.get("model_config"), payload.get("lhbl_config")
    if not isinstance(state, Mapping) or not isinstance(cfg, Mapping) or not isinstance(lhbl, Mapping) or lhbl.get("hidden_dim") != HIDDEN_DIM: raise ValueError("checkpoint metadata changed")
    rays, neighbors = cfg.get("num_rays"), cfg.get("max_neighbors")
    if not isinstance(rays, int) or not isinstance(neighbors, int): raise ValueError("checkpoint token schema changed")
    seq = 1 + rays + neighbors; model = I.FlatMLPRanker(seq, 16, HIDDEN_DIM, 4.0); wanted = model.state_dict()
    if set(state) != set(wanted) or any(not isinstance(v, torch.Tensor) or v.shape != wanted[k].shape for k, v in state.items()): raise ValueError("checkpoint state-dict keys or shapes changed")
    if tuple(state["encoder.0.weight"].shape) != (128, seq * 16): raise ValueError("checkpoint hidden width or token shape changed")
    model.load_state_dict(state, strict=True); encoder = model.encoder.to(device); encoder.requires_grad_(False); encoder.eval()
    setattr(encoder, "c13p_checkpoint_sha256", source.checkpoint_sha256); setattr(encoder, "c13p_token_shape", (seq, 16)); return encoder

def _vector(cache: Mapping[str, np.ndarray], key: str, n: int) -> np.ndarray:
    value = cache.get(key)
    if not isinstance(value, np.ndarray): raise ValueError(f"frozen feature cache is missing {key}")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (n,) or not np.all(np.isfinite(result)): raise ValueError(f"{key} is invalid")
    return result

def prepare_world_representation(feature_cache: Mapping[str, np.ndarray], graph: Sequence[Sequence[tuple[int, float]]], goal_idx: int, cfg: PersistentSearchConfig, encoder: nn.Module) -> PreparedWorld:
    tokens = feature_cache.get("features")
    if not isinstance(tokens, np.ndarray) or tokens.ndim != 3 or tuple(tokens.shape[1:]) != getattr(encoder, "c13p_token_shape", None) or not np.all(np.isfinite(tokens)): raise ValueError("feature cache token shape changed")
    path, digest = feature_cache.get("cache_path", feature_cache.get("feature_cache_path")), feature_cache.get("cache_sha256", feature_cache.get("feature_cache_sha256"))
    if path is None or digest is None: raise ValueError("feature cache path/hash binding is required")
    if not isinstance(path, (str, Path)) or not isinstance(digest, str) or sha256_file(Path(path)) != digest: raise ValueError("feature cache hash mismatch")
    if float(cfg.local_alpha) != LOCAL_ALPHA: raise ValueError("frozen local alpha changed")
    if len(graph) != len(tokens) or not isinstance(goal_idx, int) or not 0 <= goal_idx < len(tokens) or encoder.training or any(p.requires_grad for p in encoder.parameters()): raise ValueError("world or frozen encoder invalid")
    euclid, local = _vector(feature_cache, "euclidean_to_goal", len(tokens)), _vector(feature_cache, "local_value_radius_0_20", len(tokens))
    with torch.no_grad(): embedding = encoder(torch.as_tensor(tokens.astype(np.float32, copy=False), device=next(encoder.parameters()).device).flatten(1))
    if tuple(embedding.shape) != (len(tokens), HIDDEN_DIM): raise ValueError("encoder output width changed")
    return PreparedWorld(tokens, embedding.cpu().numpy(), euclid, local, euclid + LOCAL_ALPHA * (local - euclid))

@dataclass(frozen=True)
class HRMCarry:
    low: torch.Tensor; high: torch.Tensor; step: int

class PersistentSearchHRM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(MODEL_SEED); self.event_projection = nn.Linear(70, 64); self.high_block = C.GatedRecurrentBlock(64, 4); self.low_block = C.GatedRecurrentBlock(64, 4); self.candidate_head = nn.Sequential(nn.Linear(131, 64), nn.GELU(), nn.Linear(64, 1))
    @property
    def persistent_model(self) -> "PersistentSearchHRM": return self
    @property
    def reset_model(self) -> "PersistentSearchHRM": return self
    @property
    def parameter_count(self) -> int: return sum(p.numel() for p in self.parameters())
    def initial_carry(self, batch_size: int, device: torch.device, dtype: torch.dtype, step: int = 0) -> HRMCarry:
        if not isinstance(batch_size, int) or batch_size <= 0 or not isinstance(step, int) or step < 0: raise ValueError("invalid carry")
        low = torch.zeros((batch_size, 64), device=device, dtype=dtype); return HRMCarry(low, low.clone(), step)
    def update_event(self, event_features: torch.Tensor, carry: HRMCarry) -> tuple[torch.Tensor, HRMCarry]:
        if event_features.ndim != 2 or event_features.shape[1] != 70 or carry.low.shape != (event_features.shape[0], 64) or carry.high.shape != carry.low.shape: raise ValueError("event/carry mismatch")
        high = self.high_block(carry.low.detach(), carry.high) if carry.step % 2 == 0 else carry.high; low = self.low_block(self.event_projection(event_features) + high, carry.low); return low, HRMCarry(low, high, carry.step + 1)
    def score_candidates(self, candidate_embeddings: torch.Tensor, context: torch.Tensor, candidate_scalars: torch.Tensor) -> torch.Tensor:
        if candidate_embeddings.ndim != 2 or candidate_embeddings.shape[1] != 64 or candidate_scalars.shape != (len(candidate_embeddings), 3) or context.ndim != 2 or context.shape[1] != 64 or context.shape[0] not in (1, len(candidate_embeddings)): raise ValueError("candidate inputs invalid")
        return self.candidate_head(torch.cat((candidate_embeddings, context.expand(len(candidate_embeddings), -1), candidate_scalars), -1)).squeeze(-1)

ALLOWED_EVENT_KEYS = frozenset({"event_index", "expanded_node", "expanded_g", "expanded_base_rank", "open_count", "closed_count"})
ALLOWED_CANDIDATE_KEYS = frozenset({"open_nodes", "open_g", "open_base_rank"})
MODEL_CAUSAL_KEYS = ALLOWED_EVENT_KEYS | ALLOWED_CANDIDATE_KEYS
FORBIDDEN_INPUT_TOKENS = ("dist_to_goal", "dijkstra", "raster", "teacher_path", "future")
def validate_model_causal_fields(causal_event: Mapping[str, object]) -> None:
    if set(causal_event) != MODEL_CAUSAL_KEYS or any(token in str(key).casefold() for key in causal_event for token in FORBIDDEN_INPUT_TOKENS): raise ValueError("noncausal model input")
def event_tensor_from_causal(causal_event: Mapping[str, object], node_embeddings: torch.Tensor, side_len: float, roadmap_nodes: int) -> torch.Tensor:
    validate_model_causal_fields(causal_event)
    if node_embeddings.ndim != 2 or node_embeddings.shape[1] != 64 or side_len <= 0 or roadmap_nodes <= 0: raise ValueError("event representation invalid")
    node, step = _integer(causal_event["expanded_node"], "expanded_node"), _integer(causal_event["event_index"], "event_index")
    if not 0 <= node < len(node_embeddings) or step < 0: raise ValueError("event node/index invalid")
    g, rank, opened, closed = (_finite_float(causal_event[k], k) for k in ("expanded_g", "expanded_base_rank", "open_count", "closed_count"))
    return torch.cat((node_embeddings[node:node+1], torch.as_tensor([g/side_len, rank/side_len, (g+rank)/side_len, opened/roadmap_nodes, closed/roadmap_nodes, step/roadmap_nodes], dtype=node_embeddings.dtype, device=node_embeddings.device).unsqueeze(0)), -1)
def candidate_tensors_from_causal(causal_event: Mapping[str, object], node_embeddings: torch.Tensor, side_len: float) -> tuple[torch.Tensor, torch.Tensor, Sequence[int]]:
    validate_model_causal_fields(causal_event); raw_nodes, raw_g, raw_rank = causal_event["open_nodes"], causal_event["open_g"], causal_event["open_base_rank"]
    if node_embeddings.ndim != 2 or node_embeddings.shape[1] != 64 or side_len <= 0 or any(not isinstance(v, Sequence) or isinstance(v, (str, bytes)) for v in (raw_nodes, raw_g, raw_rank)) or not raw_nodes or len(raw_nodes) != len(raw_g) or len(raw_nodes) != len(raw_rank): raise ValueError("candidate representation invalid")
    nodes = tuple(_integer(v, "open_node") for v in raw_nodes)
    if len(set(nodes)) != len(nodes) or any(not 0 <= node < len(node_embeddings) for node in nodes): raise ValueError("open candidates invalid")
    g, rank = [_finite_float(v, "open_g") for v in raw_g], [_finite_float(v, "open_base_rank") for v in raw_rank]
    return node_embeddings.index_select(0, torch.as_tensor(nodes, device=node_embeddings.device)), torch.as_tensor([[a/side_len,b/side_len,(a+b)/side_len] for a,b in zip(g,rank)], dtype=node_embeddings.dtype, device=node_embeddings.device), nodes

def reset_carry_for_event(model: PersistentSearchHRM, causal_event: Mapping[str, object], batch_size: int, device: torch.device, dtype: torch.dtype, step: int | None = None) -> HRMCarry:
    validate_model_causal_fields(causal_event); event_index=_integer(causal_event["event_index"],"event_index")
    if event_index < 0 or step is not None and step != event_index: raise ValueError("reset carry step must match causal event_index")
    return model.initial_carry(batch_size,device,dtype,step=event_index)

class PersistentCarryLifecycle:
    """Explicit scope owner with linear, single-use carry transitions."""
    def __init__(self,model: PersistentSearchHRM,evaluation_id: str)->None:
        if not isinstance(evaluation_id,str) or not evaluation_id: raise ValueError("evaluation id is invalid")
        self.model=model; self.evaluation_id=evaluation_id; self._world_id: object|None=None; self._current: HRMCarry|None=None
    def initial_for_world(self,world_id:object,batch_size:int,device:torch.device,dtype:torch.dtype,evaluation_id:str|None=None)->HRMCarry:
        if evaluation_id is not None and evaluation_id != self.evaluation_id: raise ValueError("evaluation id cannot cross lifecycle")
        if self._current is not None:
            if world_id != self._world_id: raise ValueError("carry cannot cross a world boundary")
            raise ValueError("duplicate evaluation/world allocation")
        self._world_id=world_id; self._current=self.model.initial_carry(batch_size,device,dtype); return self._current
    def update(self,event_features:torch.Tensor,carry:HRMCarry)->tuple[torch.Tensor,HRMCarry]:
        if self._current is None: raise ValueError("persistent carry lifecycle is not initialized")
        if carry is not self._current:
            if carry.step < self._current.step: raise ValueError("stale carry update")
            raise ValueError("foreign carry update")
        context,next_carry=self.model.update_event(event_features,carry); self._current=next_carry; return context,next_carry
    def detach_current(self) -> HRMCarry:
        if self._current is None:
            raise ValueError("persistent carry lifecycle is not initialized")
        self._current = detach_carry(self._current)
        return self._current

@dataclass(frozen=True)
class CheckpointSelection:
    selected_epoch: int
    selected_validation_loss: float
    checkpoint_path: Path
    checkpoint_sha256: str


def frontier_cross_entropy(logits: torch.Tensor, candidate_nodes: Sequence[int], positive_node: int) -> torch.Tensor:
    """Cross-entropy over the complete current frontier and its sole path candidate."""
    if logits.ndim != 1 or logits.numel() == 0 or len(candidate_nodes) != logits.numel():
        raise ValueError("frontier logits and candidates are inconsistent")
    nodes = tuple(_integer(node, "candidate node") for node in candidate_nodes)
    if len(set(nodes)) != len(nodes) or not isinstance(positive_node, int) or nodes.count(positive_node) != 1:
        raise ValueError("frontier must contain exactly one positive candidate")
    return torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([nodes.index(positive_node)], device=logits.device))


def rank_of_positive(logits: np.ndarray, candidate_nodes: np.ndarray, positive_node: int, tie_keys: np.ndarray) -> int:
    """Return the one-based stable rank using raw descending logits and frozen ties."""
    scores = np.asarray(logits, dtype=np.float64)
    nodes = np.asarray(candidate_nodes)
    ties = np.asarray(tie_keys, dtype=np.float64)
    if scores.ndim != 1 or nodes.ndim != 1 or len(scores) == 0 or len(scores) != len(nodes) or ties.shape != (len(scores), 3):
        raise ValueError("ranking inputs are inconsistent")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(ties)):
        raise ValueError("ranking inputs must be finite")
    values = tuple(_integer(int(node), "candidate node") for node in nodes)
    if len(set(values)) != len(values) or values.count(positive_node) != 1:
        raise ValueError("ranking requires exactly one positive candidate")
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), tuple(float(value) for value in ties[index])))
    return order.index(values.index(positive_node)) + 1


def summarize_trace_metrics(event_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Keep CE event-weighted while making ranking metrics world-macro summaries."""
    required = {"world_id", "suite", "frontier_cross_entropy", "reciprocal_rank", "top1", "rank_percentile"}
    if event_rows.empty or not required.issubset(event_rows.columns):
        raise ValueError("event metric rows are incomplete")
    numeric = ("frontier_cross_entropy", "reciprocal_rank", "top1", "rank_percentile")
    if not all(np.isfinite(event_rows[column].to_numpy(dtype=float)).all() for column in numeric):
        raise ValueError("event metrics must be finite")
    worlds = event_rows.groupby(["world_id", "suite"], sort=True, as_index=False)[list(numeric)].mean()
    suite_metrics = worlds.groupby("suite", sort=True)[["reciprocal_rank", "top1", "rank_percentile"]].mean()
    summary = {
        "event_weighted_frontier_cross_entropy": float(event_rows["frontier_cross_entropy"].mean()),
        "world_macro_mrr": float(worlds["reciprocal_rank"].mean()),
        "world_macro_top1": float(worlds["top1"].mean()),
        "world_macro_rank_percentile": float(worlds["rank_percentile"].mean()),
        "suite_macro_mrr": float(suite_metrics["reciprocal_rank"].mean()),
        "suite_macro_top1": float(suite_metrics["top1"].mean()),
        "suite_macro_rank_percentile": float(suite_metrics["rank_percentile"].mean()),
    }
    return worlds, summary


def detach_carry(carry: HRMCarry) -> HRMCarry:
    return HRMCarry(carry.low.detach(), carry.high.detach(), carry.step)


def deterministic_world_order(world_ids: Sequence[str], seed: int, epoch: int) -> Sequence[str]:
    if not isinstance(seed, int) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("world-order seed or epoch is invalid")
    ordered = sorted(world_ids)
    if any(not isinstance(world_id, str) or not world_id for world_id in ordered) or len(set(ordered)) != len(ordered):
        raise ValueError("world ids must be unique nonempty strings")
    generator = random.Random((seed << 32) + epoch)
    generator.shuffle(ordered)
    return tuple(ordered)


def _trace_world_id(trace: TeacherTrace) -> str:
    return f"{trace.split}/{trace.suite}/{trace.world_index}"


def _prepared_for_trace(prepared_worlds: Mapping[str, PreparedWorld], trace: TeacherTrace) -> PreparedWorld:
    world_id = _trace_world_id(trace)
    if world_id in prepared_worlds:
        return prepared_worlds[world_id]
    raise ValueError(f"prepared representation is missing for {world_id}")


def _event_causal(event: TraceEvent) -> Mapping[str, object]:
    if event.event_kind == "initialized_frontier":
        if event.expanded_node is not None or not event.open_nodes:
            raise ValueError("initialized frontier event is invalid")
        # Event zero has no popped node; its sole causal observation is its recorded open frontier.
        node, g, rank = event.open_nodes[0], event.open_g[0], event.open_base_rank[0]
    elif event.event_kind == "post_expansion":
        if event.expanded_node is None:
            raise ValueError("post-expansion event is missing its expanded node")
        node, g, rank = event.expanded_node, event.expanded_g, event.expanded_base_rank
    else:
        raise ValueError("trace event kind is invalid")
    return {
        "event_index": event.event_index, "expanded_node": node, "expanded_g": g,
        "expanded_base_rank": rank, "open_count": event.open_count, "closed_count": event.closed_count,
        "open_nodes": event.open_nodes, "open_g": event.open_g, "open_base_rank": event.open_base_rank,
    }


def _trace_side_len(trace: TeacherTrace) -> float:
    # The trace contains node-local quantities only; this fixed normalizer is not a map-wide feature.
    return float(max(1, int(math.ceil(math.sqrt(trace.node_count)))))


def _event_tensors(event: TraceEvent, prepared: PreparedWorld, device: torch.device) -> tuple[Mapping[str, object], torch.Tensor, torch.Tensor, Sequence[int]]:
    causal = _event_causal(event)
    embeddings = torch.as_tensor(prepared.node_embeddings, dtype=torch.float32, device=device)
    side_len = _trace_side_len_from_prepared(prepared)
    event_tensor = event_tensor_from_causal(causal, embeddings, side_len, len(prepared.node_embeddings))
    candidate_embeddings, candidate_scalars, candidate_nodes = candidate_tensors_from_causal(causal, embeddings, side_len)
    return causal, event_tensor, candidate_embeddings, candidate_scalars, candidate_nodes


def _trace_side_len_from_prepared(prepared: PreparedWorld) -> float:
    # C13-J's square-world side is not embedded in the trace; node count is the frozen available scale.
    return float(max(1, int(math.ceil(math.sqrt(len(prepared.node_embeddings))))))


def _event_metric_row(trace: TeacherTrace, event: TraceEvent, logits: torch.Tensor, candidate_nodes: Sequence[int], tie_keys: np.ndarray, arm: str) -> dict[str, object]:
    loss = frontier_cross_entropy(logits, candidate_nodes, event.positive_node)
    rank = rank_of_positive(logits.detach().cpu().numpy(), np.asarray(candidate_nodes), event.positive_node, tie_keys)
    count = len(candidate_nodes)
    return {
        "world_id": _trace_world_id(trace), "suite": trace.suite, "split": trace.split, "arm": arm,
        "event_index": event.event_index, "frontier_cross_entropy": float(loss.detach().cpu()), "rank": rank,
        "reciprocal_rank": 1.0 / rank, "top1": float(rank == 1),
        "rank_percentile": float((rank - 1) / max(count - 1, 1)), "candidate_count": count,
    }


def train_one_world(model: PersistentSearchHRM, trace: TeacherTrace, prepared: PreparedWorld, optimizer: torch.optim.Optimizer, cfg: PersistentSearchConfig) -> dict[str, float]:
    """Train one ordered world stream with one AdamW step per TBPTT chunk."""
    if cfg.tbptt_events <= 0 or not trace.events:
        raise ValueError("training trace or TBPTT configuration is invalid")
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lifecycle = PersistentCarryLifecycle(model, f"train:{_trace_world_id(trace)}")
    carry = lifecycle.initial_for_world(_trace_world_id(trace), 1, device, torch.float32)
    losses: list[torch.Tensor] = []
    total_loss = 0.0
    steps = 0
    for offset, event in enumerate(trace.events):
        if event.event_index != offset:
            raise ValueError("trace events are not in causal order")
        _, event_features, candidate_embeddings, candidate_scalars, candidate_nodes = _event_tensors(event, prepared, device)
        context, carry = lifecycle.update(event_features, carry)
        loss = frontier_cross_entropy(model.score_candidates(candidate_embeddings, context, candidate_scalars), candidate_nodes, event.positive_node)
        losses.append(loss)
        total_loss += float(loss.detach().cpu())
        if len(losses) == cfg.tbptt_events or offset + 1 == len(trace.events):
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            carry = lifecycle.detach_current()
            losses.clear()
            steps += 1
    return {"event_count": float(len(trace.events)), "loss_sum": total_loss, "optimizer_steps": float(steps)}


def evaluate_stationary_split(traces: Sequence[TeacherTrace], prepared_worlds: Mapping[str, PreparedWorld], model: PersistentSearchHRM, carry_mode: str, cfg: PersistentSearchConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if carry_mode not in {"persistent", "reset", "base"}:
        raise ValueError("carry mode is invalid")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for trace in traces:
            prepared = _prepared_for_trace(prepared_worlds, trace)
            lifecycle = PersistentCarryLifecycle(model, f"validation:{carry_mode}:{_trace_world_id(trace)}")
            carry: HRMCarry | None = None
            if carry_mode == "persistent":
                carry = lifecycle.initial_for_world(_trace_world_id(trace), 1, device, torch.float32)
            for offset, event in enumerate(trace.events):
                if event.event_index != offset:
                    raise ValueError("trace events are not in causal order")
                causal, event_features, candidate_embeddings, candidate_scalars, candidate_nodes = _event_tensors(event, prepared, device)
                tie_keys = np.column_stack((np.asarray(event.open_g) + np.asarray(event.open_base_rank), event.open_g, event.open_nodes))
                if carry_mode == "base":
                    logits = -torch.as_tensor(np.asarray(event.open_g) + np.asarray(event.open_base_rank), dtype=torch.float32, device=device) / _trace_side_len_from_prepared(prepared)
                else:
                    if carry_mode == "reset":
                        carry = reset_carry_for_event(model, causal, 1, device, torch.float32)
                        context, _ = model.update_event(event_features, carry)
                    else:
                        assert carry is not None
                        context, carry = lifecycle.update(event_features, carry)
                    logits = model.score_candidates(candidate_embeddings, context, candidate_scalars)
                rows.append(_event_metric_row(trace, event, logits, candidate_nodes, tie_keys, carry_mode))
    if was_training:
        model.train()
    event_rows = pd.DataFrame(rows)
    world_rows, _ = summarize_trace_metrics(event_rows)
    return event_rows, world_rows



OFFLINE_ARMS = ("c13p_persistent", "c13p_reset", "c13m_base_rank")
_OFFLINE_METRICS = ("cross_entropy", "positive_rank", "reciprocal_rank", "top1", "rank_percentile")


def _model_state_sha256(model: PersistentSearchHRM) -> str:
    """Return a stable audit digest for the one shared immutable evaluation model."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _offline_metric_row(trace: TeacherTrace, event: TraceEvent, logits: torch.Tensor, candidate_nodes: Sequence[int], tie_keys: np.ndarray, arm: str, checkpoint_sha256: str, state_sha256: str) -> dict[str, object]:
    row = _event_metric_row(trace, event, logits, candidate_nodes, tie_keys, arm)
    row.update({
        "world_index": trace.world_index,
        "positive_node": event.positive_node,
        "candidate_nodes": tuple(candidate_nodes),
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256": state_sha256,
        "cross_entropy": row["frontier_cross_entropy"],
        "positive_rank": row["rank"],
        "raw_logits": tuple(float(value) for value in logits.detach().cpu().tolist()),
    })
    return row


def _offline_summary(event_rows: pd.DataFrame) -> pd.DataFrame:
    required = {"world_id", "split", "suite", "world_index", "arm", *_OFFLINE_METRICS}
    if event_rows.empty or not required.issubset(event_rows.columns):
        raise ValueError("offline event rows are incomplete")
    if set(event_rows["arm"].unique()) != set(OFFLINE_ARMS):
        raise ValueError("offline event arms are incomplete")
    if event_rows.duplicated(["world_id", "arm", "event_index"]).any():
        raise ValueError("offline event identities are not unique")
    if not all(np.isfinite(event_rows[column].to_numpy(dtype=float)).all() for column in _OFFLINE_METRICS):
        raise ValueError("offline event metrics must be finite")
    world = event_rows.groupby(["world_id", "split", "suite", "world_index", "arm"], sort=True, as_index=False)[list(_OFFLINE_METRICS)].mean()
    world["aggregation_level"] = "world"
    suite = world.groupby(["split", "suite", "arm"], sort=True, as_index=False)[list(_OFFLINE_METRICS)].mean()
    suite["aggregation_level"] = "suite"
    suite["world_id"] = None
    suite["world_index"] = np.nan
    pooled = world.groupby(["split", "arm"], sort=True, as_index=False)[list(_OFFLINE_METRICS)].mean()
    pooled["aggregation_level"] = "pooled"
    pooled["world_id"] = None
    pooled["suite"] = None
    pooled["world_index"] = np.nan
    columns = ["aggregation_level", "world_id", "split", "suite", "world_index", "arm", *_OFFLINE_METRICS]
    return pd.concat((world[columns], suite[columns], pooled[columns]), ignore_index=True)


def evaluate_offline_arms(traces: Sequence[TeacherTrace], prepared_worlds: Mapping[str, PreparedWorld], model: PersistentSearchHRM, checkpoint_sha256: str, cfg: PersistentSearchConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score identical recorded frontiers with persistent, reset, and frozen base order."""
    if not traces or not _is_sha256(checkpoint_sha256):
        raise ValueError("offline traces or checkpoint hash are invalid")
    world_ids = tuple(_trace_world_id(trace) for trace in traces)
    if len(set(world_ids)) != len(world_ids):
        raise ValueError("offline trace world identities must be unique")
    device = next(model.parameters()).device
    state_sha256 = _model_state_sha256(model)
    was_training = model.training
    model.eval()
    rows: list[dict[str, object]] = []
    try:
        with torch.no_grad():
            for trace in sorted(traces, key=lambda item: (item.split, item.suite, item.world_index)):
                if not trace.events:
                    raise ValueError("offline trace has no events")
                prepared = _prepared_for_trace(prepared_worlds, trace)
                persistent = PersistentCarryLifecycle(model, f"offline:persistent:{_trace_world_id(trace)}")
                carry = persistent.initial_for_world(_trace_world_id(trace), 1, device, torch.float32)
                for offset, event in enumerate(trace.events):
                    if event.event_index != offset:
                        raise ValueError("trace events are not in causal order")
                    causal, event_features, candidate_embeddings, candidate_scalars, candidate_nodes = _event_tensors(event, prepared, device)
                    tie_keys = np.column_stack((np.asarray(event.open_g) + np.asarray(event.open_base_rank), event.open_g, event.open_nodes))
                    persistent_context, carry = persistent.update(event_features, carry)
                    persistent_logits = model.score_candidates(candidate_embeddings, persistent_context, candidate_scalars)
                    rows.append(_offline_metric_row(trace, event, persistent_logits, candidate_nodes, tie_keys, OFFLINE_ARMS[0], checkpoint_sha256, state_sha256))
                    reset_carry = reset_carry_for_event(model, causal, 1, device, torch.float32)
                    reset_context, _ = model.update_event(event_features, reset_carry)
                    reset_logits = model.score_candidates(candidate_embeddings, reset_context, candidate_scalars)
                    rows.append(_offline_metric_row(trace, event, reset_logits, candidate_nodes, tie_keys, OFFLINE_ARMS[1], checkpoint_sha256, state_sha256))
                    base_logits = -torch.as_tensor(np.asarray(event.open_g) + np.asarray(event.open_base_rank), dtype=torch.float32, device=device)
                    rows.append(_offline_metric_row(trace, event, base_logits, candidate_nodes, tie_keys, OFFLINE_ARMS[2], checkpoint_sha256, state_sha256))
    finally:
        if was_training:
            model.train()
    events = pd.DataFrame(rows)
    return events, _offline_summary(events)


def world_clustered_bootstrap(paired_world_rows: pd.DataFrame, value_column: str, resamples: int, seed: int) -> dict[str, float | int | tuple[int, int]]:
    """Bootstrap only one already-paired value per official development world."""
    if not isinstance(paired_world_rows, pd.DataFrame) or not isinstance(value_column, str) or not value_column:
        raise ValueError("paired world bootstrap inputs are invalid")
    if not isinstance(resamples, int) or resamples <= 0 or not isinstance(seed, int):
        raise ValueError("paired world bootstrap configuration is invalid")
    required = {"world_id", value_column}
    if not required.issubset(paired_world_rows.columns):
        raise ValueError("paired world bootstrap rows are incomplete")
    rows = paired_world_rows.loc[:, ["world_id", value_column]].copy()
    if len(rows) != DEVELOPMENT_WORLDS:
        raise ValueError("paired world bootstrap requires exactly 24 unique worlds")
    if rows["world_id"].isna().any() or rows["world_id"].duplicated().any():
        raise ValueError("paired world bootstrap world identifiers must be unique")
    if any(not isinstance(value, str) or not value for value in rows["world_id"]):
        raise ValueError("paired world bootstrap world identifiers are invalid")
    rows = rows.sort_values("world_id", kind="stable")
    values = rows[value_column].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("paired world bootstrap values must be finite")
    indices = np.random.default_rng(seed).integers(0, DEVELOPMENT_WORLDS, size=(resamples, DEVELOPMENT_WORLDS))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975], method="linear")
    return {"point_estimate": float(values.mean()), "ci_low": float(low), "ci_high": float(high), "n_worlds": DEVELOPMENT_WORLDS, "sample_shape": (resamples, DEVELOPMENT_WORLDS)}


def _g1_paired_world_rows(world_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"world_id", "suite", "world_index", "arm", "reciprocal_rank", "top1"}
    if not isinstance(world_metrics, pd.DataFrame) or not required.issubset(world_metrics.columns):
        raise ValueError("G1 world metrics are incomplete")
    rows = world_metrics.copy()
    if "aggregation_level" in rows:
        rows = rows.loc[rows["aggregation_level"] == "world"].copy()
    if len(rows) != DEVELOPMENT_WORLDS * len(OFFLINE_ARMS):
        raise ValueError("G1 paired world rows are incomplete")
    if set(rows["arm"].unique()) != set(OFFLINE_ARMS):
        raise ValueError("G1 paired world arms are invalid")
    if rows.duplicated(["world_id", "suite", "world_index", "arm"]).any() or rows.duplicated(["world_id", "arm"]).any():
        raise ValueError("G1 paired world identities must be unique")
    identity = rows.groupby("world_id", sort=True).agg(suite=("suite", "nunique"), world_index=("world_index", "nunique"), arms=("arm", "nunique"))
    if len(identity) != DEVELOPMENT_WORLDS or not (identity["suite"] == 1).all() or not (identity["world_index"] == 1).all() or not (identity["arms"] == len(OFFLINE_ARMS)).all():
        raise ValueError("G1 paired world identities are incomplete")
    if rows["suite"].nunique() != 6:
        raise ValueError("G1 requires six suites")
    if not all(np.isfinite(rows[column].to_numpy(dtype=float)).all() for column in ("reciprocal_rank", "top1")):
        raise ValueError("G1 paired world metrics must be finite")
    pivot = rows.pivot(index=["world_id", "suite", "world_index"], columns="arm", values=["reciprocal_rank", "top1"])
    if pivot.shape[0] != DEVELOPMENT_WORLDS or pivot.isna().any().any():
        raise ValueError("G1 paired world pivot is incomplete")
    paired = pivot.reset_index()
    paired.columns = ["world_id", "suite", "world_index", "base_mrr", "persistent_mrr", "reset_mrr", "base_top1", "persistent_top1", "reset_top1"]
    paired["mrr_delta"] = paired["persistent_mrr"] - paired["reset_mrr"]
    paired["top1_delta"] = paired["persistent_top1"] - paired["reset_top1"]
    return paired


def g1_verdict(world_metrics: pd.DataFrame, bootstrap_seed: int, resamples: int) -> dict[str, object]:
    """Apply the frozen paired-world persistent-minus-reset ranking gate."""
    paired = _g1_paired_world_rows(world_metrics)
    bootstrap = world_clustered_bootstrap(paired, "mrr_delta", resamples, bootstrap_seed)
    suite_deltas = paired.groupby("suite", sort=True)["mrr_delta"].mean()
    pooled_top1_delta = float(paired["top1_delta"].mean())
    suites_with_positive_mrr = int((suite_deltas > 0.0).sum())
    pooled_mrr_ci_low = float(bootstrap["ci_low"])
    passes = pooled_mrr_ci_low > 0.0 and pooled_top1_delta >= 0.02 and suites_with_positive_mrr >= 4
    return {
        "comparison": "c13p_persistent_minus_c13p_reset",
        "pooled_mrr_delta": float(paired["mrr_delta"].mean()),
        "pooled_mrr_ci_low": pooled_mrr_ci_low,
        "pooled_mrr_ci_high": float(bootstrap["ci_high"]),
        "pooled_top1_delta": pooled_top1_delta,
        "suite_mrr_deltas": {str(suite): float(delta) for suite, delta in suite_deltas.items()},
        "suites_with_positive_mrr": suites_with_positive_mrr,
        "bootstrap": bootstrap,
        "passes": passes,
        "verdict": "c13p_g1_passed" if passes else "c13p_no_persistent_ranking_signal",
    }


def _binding_error(reason: str) -> ValueError:
    return ValueError(f"training binding {reason}; new output directory required")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def validate_training_binding(binding: Mapping[str, object]) -> str:
    """Validate the complete versioned frozen training binding before allocation."""
    if not isinstance(binding, Mapping):
        raise _binding_error("is invalid")
    required = {"binding_schema_version", "experiment_schema_version", "source_hashes", "trace_dataset_hash", "trace_generation_fingerprint", "model_config", "optimizer_config", "gate_config"}
    if not required.issubset(binding) or binding.get("binding_schema_version") != TRAINING_BINDING_SCHEMA_VERSION or binding.get("experiment_schema_version") != SCHEMA_VERSION:
        raise _binding_error("is incomplete or has a schema mismatch")
    source_hashes = binding["source_hashes"]
    if not isinstance(source_hashes, Mapping) or not source_hashes or any(not isinstance(name, str) or not name or not _is_sha256(digest) for name, digest in source_hashes.items()):
        raise _binding_error("source hashes are invalid")
    if not _is_sha256(binding["trace_dataset_hash"]) or not _is_sha256(binding["trace_generation_fingerprint"]):
        raise _binding_error("trace hashes are invalid")
    expected_model = {"model_seed": MODEL_SEED, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS, "k_step": K_STEP}
    expected_optimizer = {"name": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "grad_clip_norm": GRAD_CLIP_NORM}
    if binding["model_config"] != expected_model:
        raise _binding_error("model configuration changed")
    if binding["optimizer_config"] != expected_optimizer:
        raise _binding_error("optimizer configuration changed")
    gate_config = binding["gate_config"]
    if not isinstance(gate_config, Mapping) or not isinstance(gate_config.get("schema_version"), str) or not gate_config["schema_version"] or not _is_sha256(gate_config.get("fingerprint")):
        raise _binding_error("gate configuration is invalid")
    return hashlib.sha256(canonical_json_bytes(dict(binding))).hexdigest()
def _binding_fingerprint(binding: Mapping[str, object]) -> str:
    return validate_training_binding(binding)


def save_training_checkpoint(path: Path, model: PersistentSearchHRM, optimizer: torch.optim.Optimizer, completed_epoch: int, binding: Mapping[str, object]) -> str:
    if not isinstance(completed_epoch, int) or completed_epoch < 0:
        raise ValueError("completed epoch is invalid")
    fingerprint = _binding_fingerprint(binding)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "completed_epoch": completed_epoch,
        "binding": copy.deepcopy(dict(binding)), "binding_fingerprint": fingerprint,
        "source_hashes": copy.deepcopy(binding["source_hashes"]), "trace_dataset_hash": binding["trace_dataset_hash"], "trace_generation_fingerprint": binding["trace_generation_fingerprint"], "model_config": copy.deepcopy(binding["model_config"]), "optimizer_config": copy.deepcopy(binding["optimizer_config"]), "gate_config": copy.deepcopy(binding["gate_config"]),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }
    torch.save(payload, destination)
    return sha256_file(destination)


def load_training_checkpoint(path: Path, model: PersistentSearchHRM, optimizer: torch.optim.Optimizer, expected_binding: Mapping[str, object]) -> int:
    expected_fingerprint = _binding_fingerprint(expected_binding)
    try:
        payload = torch.load(Path(path), map_location=next(model.parameters()).device, weights_only=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError("checkpoint is unreadable; new output directory required") from exc
    if not isinstance(payload, Mapping) or payload.get("binding_fingerprint") != expected_fingerprint or canonical_json_bytes(payload.get("binding")) != canonical_json_bytes(dict(expected_binding)):
        raise ValueError("checkpoint binding mismatch; new output directory required")
    if not isinstance(payload.get("completed_epoch"), int) or payload["completed_epoch"] < 0 or not isinstance(payload.get("rng"), Mapping):
        raise ValueError("checkpoint payload is invalid; new output directory required")
    try:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        rng = payload["rng"]
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch_cpu"])
        cuda_states = rng.get("torch_cuda", [])
        if cuda_states:
            if not torch.cuda.is_available():
                raise ValueError("CUDA RNG state cannot resume without CUDA; new output directory required")
            torch.cuda.set_rng_state_all(cuda_states)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and "new output directory" in str(exc):
            raise
        raise ValueError("checkpoint state is invalid; new output directory required") from exc
    return payload["completed_epoch"]


def select_checkpoint(history: pd.DataFrame) -> CheckpointSelection:
    required = {"epoch", "validation_loss", "checkpoint_path", "checkpoint_sha256"}
    if history.empty or not required.issubset(history.columns):
        raise ValueError("checkpoint history is incomplete")
    rows = history.copy()
    rows["validation_loss"] = pd.to_numeric(rows["validation_loss"], errors="raise")
    rows["epoch"] = pd.to_numeric(rows["epoch"], errors="raise")
    if not np.isfinite(rows["validation_loss"]).all() or (rows["epoch"] < 1).any():
        raise ValueError("checkpoint history is invalid")
    minimum = float(rows["validation_loss"].min())
    selected = rows.loc[rows["validation_loss"] == minimum].sort_values("epoch", kind="stable").iloc[0]
    checkpoint = Path(str(selected["checkpoint_path"])).resolve()
    digest = str(selected["checkpoint_sha256"])
    if len(digest) != 64:
        raise ValueError("checkpoint hash is invalid")
    return CheckpointSelection(int(selected["epoch"]), minimum, checkpoint, digest)


def train_stationary_model(train_traces: Sequence[TeacherTrace], validation_traces: Sequence[TeacherTrace], prepared_worlds: Mapping[str, PreparedWorld], cfg: PersistentSearchConfig, binding: Mapping[str, object]) -> CheckpointSelection:
    """Official CUDA-only stationary trainer with durable, exact-fingerprint resume."""
    _binding_fingerprint(binding)
    locked = (cfg.model_seed, cfg.learning_rate, cfg.weight_decay, cfg.grad_clip_norm, cfg.max_epochs, cfg.patience, cfg.tbptt_events)
    expected = (MODEL_SEED, LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP_NORM, MAX_EPOCHS, PATIENCE, TBPTT_EVENTS)
    if locked != expected:
        raise ValueError("official training configuration changed; new output directory required")
    if cfg.hidden_dim != HIDDEN_DIM or cfg.num_layers != NUM_LAYERS or cfg.num_heads != NUM_HEADS or cfg.k_step != K_STEP:
        raise ValueError("official model configuration changed; new output directory required")
    if not torch.cuda.is_available():
        raise RuntimeError("official C13-P training requires CUDA before model or optimizer allocation")
    output = Path(cfg.out_dir)
    checkpoint_dir, result_dir = output / "checkpoints", output / "results"
    latest, history_path = checkpoint_dir / "latest.pt", result_dir / "training_history.csv"
    if output.exists() and not latest.exists() and any(output.iterdir()):
        raise ValueError("partial output is incomplete; new output directory required")
    model = PersistentSearchHRM().to(torch.device("cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = pd.DataFrame()
    completed = 0
    if latest.exists():
        completed = load_training_checkpoint(latest, model, optimizer, binding)
        if not history_path.is_file():
            raise ValueError("partial output history is missing; new output directory required")
        history = pd.read_csv(history_path)
        if history.empty or int(history["epoch"].max()) != completed:
            raise ValueError("partial output history disagrees with checkpoint; new output directory required")
        epochs = [int(value) for value in history.sort_values("epoch", kind="stable")["epoch"]]
        if epochs != list(range(1, completed + 1)):
            raise ValueError("partial output history epochs are invalid; new output directory required")
        select_checkpoint(history)
    trace_by_id = {_trace_world_id(trace): trace for trace in train_traces}
    if len(trace_by_id) != len(train_traces):
        raise ValueError("training traces have duplicate world ids")
    best_loss = math.inf
    stalled = 0
    for _, row in history.sort_values("epoch", kind="stable").iterrows():
        value = float(row["validation_loss"])
        if value < best_loss:
            best_loss, stalled = value, 0
        else:
            stalled += 1
    if completed and stalled >= cfg.patience:
        return select_checkpoint(history)
    for epoch in range(completed + 1, cfg.max_epochs + 1):
        train_loss_sum, train_events = 0.0, 0.0
        for world_id in deterministic_world_order(tuple(trace_by_id), cfg.model_seed, epoch):
            metrics = train_one_world(model, trace_by_id[world_id], _prepared_for_trace(prepared_worlds, trace_by_id[world_id]), optimizer, cfg)
            train_loss_sum += metrics["loss_sum"]; train_events += metrics["event_count"]
        validation_events, _ = evaluate_stationary_split(validation_traces, prepared_worlds, model, "persistent", cfg)
        _, validation_summary = summarize_trace_metrics(validation_events)
        validation_loss = validation_summary["event_weighted_frontier_cross_entropy"]
        epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        digest = save_training_checkpoint(epoch_path, model, optimizer, epoch, binding)
        save_training_checkpoint(latest, model, optimizer, epoch, binding)
        record = {"epoch": epoch, "train_event_weighted_loss": train_loss_sum / train_events, "validation_loss": validation_loss, "validation_world_macro_mrr": validation_summary["world_macro_mrr"], "validation_world_macro_top1": validation_summary["world_macro_top1"], "checkpoint_path": str(epoch_path.resolve()), "checkpoint_sha256": digest}
        history = pd.concat((history, pd.DataFrame([record])), ignore_index=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(history_path, index=False)
        if validation_loss < best_loss:
            best_loss, stalled = validation_loss, 0
        else:
            stalled += 1
        if stalled >= cfg.patience:
            break
    return select_checkpoint(history)
