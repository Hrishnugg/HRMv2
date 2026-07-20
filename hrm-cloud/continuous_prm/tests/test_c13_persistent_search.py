import copy
import hashlib
import json
from pathlib import Path

import pytest

import continuous_prm_c13_persistent_search as P


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(cache: Path, *, seed: int = 7) -> dict[str, object]:
    return {
        "world_seed": seed,
        "roadmap_seed": seed + 17,
        "nodes": 192,
        "edges": 384,
        "cache": str(cache),
        "cache_sha256": _sha256(cache),
        "cache_status": "reused",
    }


def _source_context(tmp_path: Path) -> P.SourceContext:
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"frozen cache")
    records = {split: [_record(cache, seed=index + 7)] for index, split in enumerate(("train", "validation", "development"))}
    reference = copy.deepcopy(records)
    return P.SourceContext(
        c13j_root=tmp_path,
        c13m_root=tmp_path,
        preregistration=tmp_path / "preregistration.md",
        implementation=tmp_path / "implementation.py",
        source_manifest={"cohort_records": reference},
        source_hashes={},
        cohort_records=records,
        checkpoint_path=cache,
        checkpoint_sha256=_sha256(cache),
    )


def test_config_defaults_freeze_every_preregistered_constant(tmp_path: Path) -> None:
    cfg = P.resolve_paths(tmp_path)

    assert (cfg.schema_version, cfg.model_seed, cfg.hidden_dim, cfg.num_layers, cfg.num_heads) == ("c13p-v1", 18423, 64, 1, 4)
    assert (cfg.k_step, cfg.local_radius, cfg.local_alpha) == (2, 0.20, 1.50)
    assert (cfg.learning_rate, cfg.weight_decay, cfg.grad_clip_norm) == (5e-4, 1e-4, 1.0)
    assert (cfg.max_epochs, cfg.patience, cfg.tbptt_events, cfg.max_expansions, cfg.bootstrap_resamples) == (20, 4, 32, 192, 20_000)
    assert P.BOOTSTRAP_SEEDS == {"g1_mrr": 3789372949, "g2_exp_reset": 1177043361, "g2_exp_c13m": 580060237}
    assert (P.TRAIN_WORLDS, P.VALIDATION_WORLDS, P.DEVELOPMENT_WORLDS) == (96, 24, 24)


def test_canonical_json_is_sorted_utf8_newline_terminated_and_finite(tmp_path: Path) -> None:
    first = {"z": ["é", {"b": 2, "a": 1}], "a": True}
    second = {"a": True, "z": ["é", {"a": 1, "b": 2}]}

    encoded = P.canonical_json_bytes(first)
    assert encoded == P.canonical_json_bytes(second)
    assert encoded == b'{"a":true,"z":["\xc3\xa9",{"a":1,"b":2}]}\n'
    output = tmp_path / "canonical.json"
    assert P.write_canonical_json(output, first) == _sha256(output)
    assert output.read_bytes() == encoded
    with pytest.raises(ValueError, match="NaN"):
        P.canonical_json_bytes({"bad": float("nan")})
    with pytest.raises(ValueError, match="Infinity"):
        P.canonical_json_bytes({"bad": float("inf")})


def test_integrity_manifest_detects_mutation_without_repairing_input(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.bin"
    frozen.write_bytes(b"original")
    manifest = tmp_path / "integrity.json"
    manifest.write_text(json.dumps({"inputs": {"frozen": {"path": "frozen.bin", "sha256": _sha256(frozen)}}}), encoding="utf-8")

    assert P.verify_integrity_manifest(tmp_path, manifest)["frozen"] == _sha256(frozen)
    frozen.write_bytes(b"mutated")
    mutated = frozen.read_bytes()
    with pytest.raises(ValueError, match="frozen"):
        P.verify_integrity_manifest(tmp_path, manifest)
    assert frozen.read_bytes() == mutated


def test_integrity_manifest_rebases_stale_absolute_path_to_configured_snapshot(tmp_path: Path) -> None:
    configured = tmp_path / "configured" / "runs" / "c13_lhbl_multisuite"
    original = tmp_path / "primary" / "runs" / "c13_lhbl_multisuite"
    configured.mkdir(parents=True)
    original.mkdir(parents=True)
    configured_payload = configured / "payload.bin"
    stale_payload = original / "payload.bin"
    configured_payload.write_bytes(b"configured snapshot")
    stale_payload.write_bytes(b"stale primary checkout")
    manifest = configured / "integrity.json"
    manifest.write_text(
        json.dumps({"inputs": {"payload": {"path": str(stale_payload), "sha256": _sha256(configured_payload)}}}),
        encoding="utf-8",
    )

    assert P.verify_integrity_manifest(configured, manifest)["payload"] == _sha256(configured_payload)


def test_cohort_replay_rebases_stale_absolute_cache_path_to_configured_snapshot(tmp_path: Path) -> None:
    configured = tmp_path / "configured" / "runs" / "c13_lhbl_multisuite"
    original = tmp_path / "primary" / "runs" / "c13_lhbl_multisuite"
    configured.mkdir(parents=True)
    original.mkdir(parents=True)
    configured_cache = configured / "cache.npz"
    stale_cache = original / "cache.npz"
    configured_cache.write_bytes(b"configured cache")
    stale_cache.write_bytes(b"stale cache")
    record = _record(stale_cache)
    record["cache_sha256"] = _sha256(configured_cache)
    records = {split: [copy.deepcopy(record)] for split in ("train", "validation", "development")}
    source = P.SourceContext(
        c13j_root=configured, c13m_root=configured,
        preregistration=configured / "design.md", implementation=configured / "implementation.py",
        source_manifest={"cohort_records": copy.deepcopy(records)}, source_hashes={},
        cohort_records=records, checkpoint_path=configured_cache, checkpoint_sha256=_sha256(configured_cache),

    )

    assert P.replay_cohort_records(source)["train"][0]["cache"] == str(configured_cache.resolve())

def test_disjointness_rejects_output_nested_in_a_frozen_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="overlaps frozen source"):
        P.assert_source_output_disjoint([source], source / "output")


@pytest.mark.parametrize("field,value", [
    ("world_seed", 999),
    ("roadmap_seed", 1020),
    ("nodes", 193),
    ("edges", 385),
    ("cache", "other-cache.npz"),
    ("cache_sha256", "0" * 64),
])
def test_cohort_replay_rejects_each_changed_frozen_field(tmp_path: Path, field: str, value: object) -> None:
    source = _source_context(tmp_path)
    mutated = copy.deepcopy(source.cohort_records)
    mutated["train"][0][field] = value
    source = P.SourceContext(
        c13j_root=source.c13j_root, c13m_root=source.c13m_root,
        preregistration=source.preregistration, implementation=source.implementation,
        source_manifest=source.source_manifest, source_hashes=source.source_hashes,
        cohort_records=mutated, checkpoint_path=source.checkpoint_path,
        checkpoint_sha256=source.checkpoint_sha256,
    )

    with pytest.raises(ValueError, match=field):
        P.replay_cohort_records(source)


def test_cohort_replay_rejects_generated_cache_status(tmp_path: Path) -> None:
    source = _source_context(tmp_path)
    source.cohort_records["validation"][0]["cache_status"] = "generated"

    with pytest.raises(ValueError, match="cache_status"):
        P.replay_cohort_records(source)


def test_audit_sources_rejects_generated_cache_from_temporary_snapshot(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    continuous = repo / "hrm-cloud" / "continuous_prm"
    c13j = continuous / "runs" / "c13_lhbl_multisuite"
    c13m = continuous / "runs" / "c13_matched_quality_confirmation"
    results = c13j / "results"
    results.mkdir(parents=True)
    c13m_results = c13m / "results"
    c13m_results.mkdir(parents=True)
    cache = c13j / "cache.npz"
    cache.write_bytes(b"cache")
    records = {
        split: [_record(cache, seed=base + index) for index in range(count)]
        for split, base, count in (("train", 100, 96), ("validation", 200, 24), ("development", 300, 24))
    }
    records["train"][0]["cache_status"] = "generated"
    cohorts = results / "cohorts.json"
    cohorts.write_text(json.dumps({"records": records}), encoding="utf-8")
    checkpoint = c13j / "checkpoints" / "flat_mlp_iteration_08.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    integrity = {
        "inputs": {"cache": {"path": str(cache), "sha256": _sha256(cache)}},
        "outputs": {"checkpoint_08": {"path": str(checkpoint), "sha256": _sha256(checkpoint)}},
    }
    (c13j / "integrity.json").write_text(json.dumps(integrity), encoding="utf-8")
    (c13j / "manifest.json").write_text("{}", encoding="utf-8")
    fingerprint = c13m_results / "evaluation_fingerprint.json"
    fingerprint.write_text("{}", encoding="utf-8")
    (c13m / "integrity.json").write_text(json.dumps({"outputs": {"fingerprint": {"path": str(fingerprint), "sha256": _sha256(fingerprint)}}}), encoding="utf-8")
    preregistration = repo / "docs" / "experiments" / "continuous" / "c13" / "design" / "2026-07-19-c13p-persistent-search-state.md"
    preregistration.parent.mkdir(parents=True)
    preregistration.write_text("frozen design", encoding="utf-8")

    with pytest.raises(ValueError, match="cache_status"):
        P.audit_sources(P.resolve_paths(repo))
    records["train"][0]["cache_status"] = "reused"
    cohorts.write_text(json.dumps({"records": records}), encoding="utf-8")

    source = P.audit_sources(P.resolve_paths(repo))
    assert source.source_hashes["c13j_manifest"] == _sha256(c13j / "manifest.json")
