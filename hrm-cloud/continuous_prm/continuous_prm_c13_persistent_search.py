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
