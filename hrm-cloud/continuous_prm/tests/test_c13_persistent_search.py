from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

import continuous_prm_c13_identifiability as I

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


def _teacher_hand_graph() -> tuple[list[list[tuple[int, float]]], np.ndarray, dict[str, object]]:
    """A six-node graph with an open-node relaxation, priority tie, and expanded off-path branch."""
    graph = [[(1, 5.0), (2, 1.0), (3, 2.0)], [(4, 1.0)], [(1, 1.0)], [], [], []]
    base_rank = np.asarray([0.0, 4.0, 4.0, 3.0, 0.0, 0.0], dtype=np.float64)
    metadata: dict[str, object] = {"split": "train", "suite": "hand", "world_index": 3, "world_seed": 101, "roadmap_seed": 202, "feature_cache_path": "frozen/hand-cache.npz", "feature_cache_sha256": "a" * 64}
    return graph, base_rank, metadata


def _hand_trace() -> tuple[P.TeacherTrace, list[list[tuple[int, float]]]]:
    graph, base_rank, metadata = _teacher_hand_graph()
    return P.generate_teacher_trace(graph, 0, 4, base_rank, metadata), graph


def test_teacher_trace_records_initialized_frontier_then_post_relaxation_snapshots() -> None:
    trace, _ = _hand_trace()

    assert trace.teacher_path == (0, 2, 1, 4)
    assert trace.teacher_cost == 3.0
    assert trace.teacher_expansions == 5
    assert [event.event_index for event in trace.events] == [0, 1, 2, 3, 4]
    # Event zero is causal state before the first pop, so it retains the start candidate and label.
    assert trace.events[0].event_kind == "initialized_frontier"
    assert trace.events[0].expanded_node is None
    assert trace.events[0].open_nodes == (0,)
    assert trace.events[0].open_g == (0.0,)
    assert trace.events[0].open_parent == (None,)
    assert trace.events[0].closed_count == 0
    assert trace.events[0].positive_node == 0
    # After expanding the start, 2 and 3 tie on f=5 and g chooses 2; 3 is then actually popped.
    assert all(event.event_kind == "post_expansion" for event in trace.events[1:])
    assert trace.events[1].expanded_node == 0
    assert trace.events[1].open_nodes == (2, 3, 1)
    assert trace.events[2].expanded_node == 2
    assert trace.events[2].open_nodes == (3, 1)
    assert trace.events[2].open_g == (2.0, 2.0)
    assert trace.events[2].open_parent == (0, 2)
    assert trace.events[3].expanded_node == 3
    assert trace.events[3].open_nodes == (1,)
    assert trace.events[4].expanded_node == 1
    assert trace.events[4].open_nodes == (4,)
    assert all(event.expanded_node != trace.goal_idx for event in trace.events)


def test_teacher_trace_allows_repeated_path_frontier_when_off_path_node_is_expanded() -> None:
    trace, graph = _hand_trace()

    assert [event.positive_node for event in trace.events] == [0, 2, 1, 1, 4]
    assert trace.events[2].positive_node == trace.events[3].positive_node == 1
    P.validate_teacher_trace(trace, graph)
    for event in trace.events:
        assert event.positive_node in event.open_nodes
        assert event.open_nodes.count(event.positive_node) == 1


def test_teacher_trace_validation_rejects_missing_closed_and_nonopen_positive() -> None:
    trace, graph = _hand_trace()
    cases = ((2, None, "positive"), (3, 2, "closed"), (1, 5, "open"))
    for event_index, positive_node, match in cases:
        events = list(trace.events)
        events[event_index] = replace(events[event_index], positive_node=positive_node)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=match):
            P.validate_teacher_trace(replace(trace, events=tuple(events)), graph)


def test_teacher_trace_validation_rejects_duplicate_open_candidate() -> None:
    trace, graph = _hand_trace()
    event = trace.events[2]
    duplicate = replace(
        event,
        open_nodes=event.open_nodes + (event.open_nodes[-1],),
        open_g=event.open_g + (event.open_g[-1],),
        open_parent=event.open_parent + (event.open_parent[-1],),
        open_base_rank=event.open_base_rank + (event.open_base_rank[-1],),
        open_count=event.open_count + 1,
    )
    events = list(trace.events)
    events[2] = duplicate
    with pytest.raises(ValueError, match="duplicate"):
        P.validate_teacher_trace(replace(trace, events=tuple(events)), graph)


def test_teacher_trace_replay_rejects_changed_candidate_g_or_parent_snapshot() -> None:
    trace, graph = _hand_trace()
    first = trace.events[2]
    changed_g = replace(first, open_g=(9.0, first.open_g[1]))
    changed_parent = replace(first, open_parent=(2, first.open_parent[1]))
    for changed in (changed_g, changed_parent):
        events = list(trace.events)
        events[2] = changed
        with pytest.raises(ValueError, match="replay"):
            P.validate_teacher_trace(replace(trace, events=tuple(events)), graph)


def test_teacher_trace_replay_rejects_equal_cost_privileged_path_substitution() -> None:
    graph = [[(1, 1.0), (2, 1.0)], [(3, 1.0)], [(3, 1.0)], []]
    metadata = {"split": "train", "suite": "equal", "world_index": 0, "world_seed": 1, "roadmap_seed": 2, "feature_cache_path": "equal.npz", "feature_cache_sha256": "c" * 64}
    trace = P.generate_teacher_trace(graph, 0, 3, np.asarray([0.0, 0.0, 1.0, 0.0]), metadata)
    events = list(trace.events)
    events[1] = replace(events[1], positive_node=2)
    events[2] = replace(events[2], positive_node=2)
    substituted = replace(trace, events=tuple(events), teacher_path=(0, 2, 3))
    with pytest.raises(ValueError, match="parent chain"):
        P.validate_teacher_trace(substituted, graph)


def test_teacher_trace_generation_does_not_call_shortest_path_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    graph, base_rank, metadata = _teacher_hand_graph()
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("trace generation must not call an optimal-path oracle")
    monkeypatch.setattr(P, "dijkstra", forbidden, raising=False)
    monkeypatch.setattr(P, "shortest_path", forbidden, raising=False)
    trace = P.generate_teacher_trace(graph, 0, 4, base_rank, metadata)
    assert trace.teacher_valid is True


def test_teacher_trace_payload_exposes_only_one_causal_event_per_example() -> None:
    trace, _ = _hand_trace()
    payload = P.trace_payload(trace)

    assert "model_causal" not in payload
    examples = payload["examples"]
    assert len(examples) == len(trace.events)
    for index, example in enumerate(examples):
        event = trace.events[index]
        causal = example["model_causal"]
        assert set(causal) == {"split", "suite", "world_index", "world_seed", "roadmap_seed", "feature_cache_path", "feature_cache_sha256", "node_count", "edge_count", "start_idx", "goal_idx", "event"}
        assert "events" not in causal
        assert causal["event"] == {
            "event_index": event.event_index, "event_kind": event.event_kind,
            "expanded_node": event.expanded_node, "expanded_g": event.expanded_g,
            "expanded_base_rank": event.expanded_base_rank, "open_nodes": list(event.open_nodes),
            "open_g": list(event.open_g), "open_base_rank": list(event.open_base_rank),
            "open_count": event.open_count, "closed_count": event.closed_count,
        }
        assert example["labels"] == {"positive_node": event.positive_node}
        assert example["replay_audit"] == {"open_parent": list(event.open_parent)}
    assert payload["privileged_audit"]["teacher_path"] == list(trace.teacher_path)
    assert P.trace_from_payload(payload) == trace

def test_teacher_trace_payload_rejects_noncanonical_or_relabelled_event_kinds() -> None:
    trace, graph = _hand_trace()
    cases = ((0, "post_expansion", "initialized"), (1, "initialized_frontier", "post-expansion"), (1, "forged_kind", "event_kind"))
    for index, event_kind, match in cases:
        payload = copy.deepcopy(P.trace_payload(trace))
        payload["examples"][index]["model_causal"]["event"]["event_kind"] = event_kind
        with pytest.raises(ValueError, match=match):
            P.trace_from_payload(payload)
        events = list(trace.events)
        events[index] = replace(events[index], event_kind=event_kind)
        with pytest.raises(ValueError, match=match):
            P.validate_teacher_trace(replace(trace, events=tuple(events)), graph)

def test_teacher_trace_payload_and_shard_are_byte_identical_across_duplicate_passes() -> None:
    first, _ = _hand_trace()
    second, _ = _hand_trace()
    assert P.canonical_json_bytes(P.trace_payload(first)) == P.canonical_json_bytes(P.trace_payload(second))
    fingerprint = P.trace_generation_fingerprint(source_hashes={"source": "b" * 64}, cohort_record={"world_seed": 101, "roadmap_seed": 202}, base_rank=np.asarray([0.0, 4.0, 4.0, 3.0, 0.0, 0.0]))
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
        left = Path(temporary) / "left.json"
        right = Path(temporary) / "right.json"
        assert P.write_trace_shard(left, [first], fingerprint) == _sha256(left)
        assert P.write_trace_shard(right, [second], fingerprint) == _sha256(right)
        assert left.read_bytes() == right.read_bytes()
        assert P.read_trace_shard(left, fingerprint) == (first,)
def _task3_source(tmp_path: Path) -> P.SourceContext:
    model = I.FlatMLPRanker(3, 16, 64, 4.0); checkpoint = tmp_path / "flat_mlp_iteration_08.pt"
    torch.save({"model": model.state_dict(), "model_name": "flat_mlp", "iteration": 8, "lhbl_config": {"hidden_dim": 64}, "model_config": {"num_rays": 1, "max_neighbors": 1}}, checkpoint)
    return P.SourceContext(tmp_path, tmp_path, tmp_path / "design.md", tmp_path / "impl.py", {}, {}, {}, checkpoint, _sha256(checkpoint))

def _task3_features() -> dict[str, np.ndarray]:
    return {"features": np.arange(96, dtype=np.float32).reshape(2,3,16)/100, "euclidean_to_goal": np.asarray([.8,0.]), "local_value_radius_0_20": np.asarray([1.2,0.])}

def test_persistent_and_reset_carries_share_one_model_and_preserve_true_cadence() -> None:
    model = P.PersistentSearchHRM(); carry = model.initial_carry(1,torch.device("cpu"),torch.float32); assert carry.step == 0 and not torch.count_nonzero(carry.low) and not torch.count_nonzero(carry.high)
    context, first = model.update_event(torch.zeros(1,70),carry); _, second = model.update_event(torch.ones(1,70),first); _, reset = model.update_event(torch.zeros(1,70),model.initial_carry(1,torch.device("cpu"),torch.float32,step=1))
    assert second.step == reset.step == 2 and not torch.equal(first.low,carry.low) and context.shape == (1,64) and model.persistent_model is model.reset_model and model.parameter_count == sum(p.numel() for p in model.parameters())

def test_candidate_scoring_is_pure_and_permutation_equivariant() -> None:
    model=P.PersistentSearchHRM(); context,carry=model.update_event(torch.randn(1,70),model.initial_carry(1,torch.device("cpu"),torch.float32)); before=(carry.low.clone(),carry.high.clone(),carry.step); embeddings=torch.randn(4,64); scalars=torch.randn(4,3); logits=model.score_candidates(embeddings,context,scalars); order=torch.tensor([2,0,3,1])
    assert torch.equal(carry.low,before[0]) and torch.equal(carry.high,before[1]) and carry.step==before[2] and torch.equal(logits[order],model.score_candidates(embeddings[order],context,scalars[order]))

def test_causal_tensor_boundary_rejects_privileged_and_future_fields_before_tensor_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    causal={"event_index":0,"expanded_node":0,"expanded_g":0.,"expanded_base_rank":.5,"open_count":1,"closed_count":0,"open_nodes":[1],"open_g":[.2],"open_base_rank":[.4]}
    for bad in ("positive_node","open_parent","privileged_audit",*P.FORBIDDEN_INPUT_TOKENS):
        invalid=dict(causal); invalid[bad]=0
        with pytest.raises(ValueError): P.validate_model_causal_fields(invalid)
    monkeypatch.setattr(P.torch,"as_tensor",lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError("tensor created")))
    invalid=dict(causal); invalid["future_open_count"]=2
    with pytest.raises(ValueError): P.event_tensor_from_causal(invalid,torch.zeros(2,64),1.,2)

def test_representation_requires_cache_binding_and_locked_alpha(tmp_path: Path) -> None:
    encoder=P.load_frozen_flat_encoder(_task3_source(tmp_path),torch.device("cpu")); cache=_task3_features()
    with pytest.raises(ValueError,match="cache"): P.prepare_world_representation(cache,[[],[]],1,P.resolve_paths(tmp_path),encoder)
    payload=tmp_path/"tokens.npz"; np.savez(payload,features=cache["features"]); cache["cache_path"]=str(payload); cache["cache_sha256"]=_sha256(payload)
    partial=dict(cache); partial.pop("cache_sha256")
    with pytest.raises(ValueError,match="cache"): P.prepare_world_representation(partial,[[],[]],1,P.resolve_paths(tmp_path),encoder)
    bad=dict(cache); bad["cache_sha256"]="0"*64
    with pytest.raises(ValueError,match="cache"): P.prepare_world_representation(bad,[[],[]],1,P.resolve_paths(tmp_path),encoder)
    with pytest.raises(ValueError,match="alpha"): P.prepare_world_representation(cache,[[],[]],1,P.PersistentSearchConfig(tmp_path,tmp_path,local_alpha=1.25),encoder)

def test_reset_carry_uses_true_event_index_and_lifecycle_scopes_are_single_use(monkeypatch: pytest.MonkeyPatch) -> None:
    model=P.PersistentSearchHRM(); calls=[]; forward=model.high_block.forward
    def counted(*args: object,**kwargs: object) -> torch.Tensor:
        calls.append(1); return forward(*args,**kwargs)
    monkeypatch.setattr(model.high_block,"forward",counted)
    event={"event_index":0,"expanded_node":0,"expanded_g":0.,"expanded_base_rank":0.,"open_count":1,"closed_count":0,"open_nodes":[1],"open_g":[0.],"open_base_rank":[0.]}
    carries=[P.reset_carry_for_event(model,{**event,"event_index":i},1,torch.device("cpu"),torch.float32) for i in (0,1,2)]
    assert [c.step for c in carries]==[0,1,2]
    for carry in carries: model.update_event(torch.ones(1,70),carry)
    assert len(calls)==2
    owner=P.PersistentCarryLifecycle(model,"official-a"); assert owner.initial_for_world("world-a",1,torch.device("cpu"),torch.float32).step==0
    with pytest.raises(ValueError,match="world"): owner.initial_for_world("world-b",1,torch.device("cpu"),torch.float32)
    with pytest.raises(ValueError,match="evaluation"): owner.initial_for_world("world-a",1,torch.device("cpu"),torch.float32,evaluation_id="official-b")


def test_lifecycle_rejects_stale_and_foreign_carries() -> None:
    model=P.PersistentSearchHRM(); first=P.PersistentCarryLifecycle(model,"eval-a"); carry=first.initial_for_world("world-a",1,torch.device("cpu"),torch.float32)
    _,next_carry=first.update(torch.ones(1,70),carry)
    with pytest.raises(ValueError): first.update(torch.ones(1,70),carry)
    foreign=P.PersistentCarryLifecycle(model,"eval-b"); other=foreign.initial_for_world("world-b",1,torch.device("cpu"),torch.float32)
    with pytest.raises(ValueError): first.update(torch.ones(1,70),other)
    assert first.update(torch.ones(1,70),next_carry)[1].step==2

def test_frozen_encoder_preparation_accepts_exact_bound_cache(tmp_path: Path) -> None:
    encoder=P.load_frozen_flat_encoder(_task3_source(tmp_path),torch.device("cpu")); cache=_task3_features(); path=tmp_path/"exact-cache.npz"; np.savez(path,features=cache["features"]); cache["cache_path"]=str(path); cache["cache_sha256"]=_sha256(path)
    prepared=P.prepare_world_representation(cache,[[(1,1.)],[(0,1.)]],1,P.resolve_paths(tmp_path),encoder)
    assert prepared.node_embeddings.shape==(2,64) and all(not p.requires_grad for p in encoder.parameters()) and not encoder.training
    np.testing.assert_allclose(prepared.base_rank,prepared.euclidean_rank+1.50*(prepared.local_values-prepared.euclidean_rank))

def test_reset_carry_is_zero_causal_and_cadenced(monkeypatch: pytest.MonkeyPatch) -> None:
    model=P.PersistentSearchHRM(); calls=[]; original=model.high_block.forward
    monkeypatch.setattr(model.high_block,"forward",lambda *a,**k:(calls.append(1),original(*a,**k))[1])
    base={"event_index":0,"expanded_node":0,"expanded_g":0.,"expanded_base_rank":0.,"open_count":1,"closed_count":0,"open_nodes":[1],"open_g":[0.],"open_base_rank":[0.]}
    carries=[P.reset_carry_for_event(model,{**base,"event_index":i},1,torch.device("cpu"),torch.float32) for i in (0,1,2)]
    assert [c.step for c in carries]==[0,1,2] and all(not torch.count_nonzero(c.low) and not torch.count_nonzero(c.high) for c in carries)
    with pytest.raises(ValueError): P.reset_carry_for_event(model,base,1,torch.device("cpu"),torch.float32,step=1)
    for carry in carries: model.update_event(torch.ones(1,70),carry)
    assert len(calls)==2

def test_privileged_audit_dictionary_is_rejected() -> None:
    causal={"event_index":0,"expanded_node":0,"expanded_g":0.,"expanded_base_rank":0.,"open_count":1,"closed_count":0,"open_nodes":[1],"open_g":[0.],"open_base_rank":[0.]}; causal["privileged_audit"]={"teacher_path":[0,1]}
    with pytest.raises(ValueError): P.validate_model_causal_fields(causal)


def _training_trace(event_count: int = 2, *, world_index: int = 0) -> P.TeacherTrace:
    events = []
    for index in range(event_count):
        events.append(P.TraceEvent(
            event_index=index, event_kind="initialized_frontier" if index == 0 else "post_expansion",
            expanded_node=None if index == 0 else 0, expanded_g=0.0, expanded_base_rank=0.0,
            open_nodes=(0, 1), open_g=(0.0, 1.0), open_parent=(None, 0),
            open_base_rank=(0.0, 0.5), open_count=2, closed_count=index, positive_node=0,
        ))
    return P.TeacherTrace("train", "maze", world_index, 1, 2, "cache", "0" * 64, 2, 1, 0, 1, tuple(events), (0, 1), 1.0, event_count, True)


def _prepared_training_world() -> P.PreparedWorld:
    return P.PreparedWorld(
        np.zeros((2, 3, 16), dtype=np.float32), np.zeros((2, 64), dtype=np.float32),
        np.array([0.0, 1.0]), np.array([0.0, 0.5]), np.array([0.0, 0.5]),
    )


def test_frontier_loss_ranking_and_world_macro_metrics() -> None:
    logits = torch.tensor([0.0, 3.0, 1.0], requires_grad=True)
    loss = P.frontier_cross_entropy(logits, (4, 7, 9), 7)
    assert loss.item() == pytest.approx(torch.nn.functional.cross_entropy(logits[None], torch.tensor([1])).item())
    assert P.rank_of_positive(np.array([2.0, 2.0]), np.array([4, 7]), 7, np.array([[2.0, 1.0, 4.0], [2.0, 0.0, 7.0]])) == 1
    worlds, summary = P.summarize_trace_metrics(pd.DataFrame([
        {"world_id": "a", "suite": "x", "frontier_cross_entropy": 0.0, "reciprocal_rank": 1.0, "top1": 1.0, "rank_percentile": 0.0},
        {"world_id": "a", "suite": "x", "frontier_cross_entropy": 2.0, "reciprocal_rank": 0.5, "top1": 0.0, "rank_percentile": 1.0},
        {"world_id": "b", "suite": "x", "frontier_cross_entropy": 8.0, "reciprocal_rank": 1.0, "top1": 1.0, "rank_percentile": 0.0},
    ]))
    assert len(worlds) == 2 and summary["event_weighted_frontier_cross_entropy"] == pytest.approx(10.0 / 3.0)
    assert summary["world_macro_mrr"] == pytest.approx(0.875) and summary["world_macro_top1"] == pytest.approx(0.75)


def test_train_world_tbptt_preserves_values_and_detaches_carry() -> None:
    model = P.PersistentSearchHRM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=P.LEARNING_RATE, weight_decay=P.WEIGHT_DECAY)
    cfg = replace(P.resolve_paths(Path.cwd()), tbptt_events=32)
    detached = []
    original = P.detach_carry
    def observed(carry: P.HRMCarry) -> P.HRMCarry:
        result = original(carry); detached.append((carry, result)); return result
    monkeypatch = pytest.MonkeyPatch(); monkeypatch.setattr(P, "detach_carry", observed)
    try:
        result = P.train_one_world(model, _training_trace(33), _prepared_training_world(), optimizer, cfg)
    finally:
        monkeypatch.undo()
    assert result["optimizer_steps"] == 2 and result["event_count"] == 33 and len(detached) == 2
    for before, after in detached:
        assert torch.equal(before.low, after.low) and torch.equal(before.high, after.high)
        assert after.low.grad_fn is None and after.high.grad_fn is None


def test_checkpoint_resume_requires_an_exact_binding(tmp_path: Path) -> None:
    model = P.PersistentSearchHRM(); optimizer = torch.optim.AdamW(model.parameters(), lr=P.LEARNING_RATE, weight_decay=P.WEIGHT_DECAY)
    binding = _complete_training_binding()
    checkpoint = tmp_path / "checkpoint.pt"
    P.save_training_checkpoint(checkpoint, model, optimizer, 3, binding)
    assert P.load_training_checkpoint(checkpoint, model, optimizer, binding) == 3
    with pytest.raises(ValueError, match="new output directory"):
        P.load_training_checkpoint(checkpoint, model, optimizer, {**binding, "gate": {"version": 2}})




def test_order_checkpoint_selection_and_every_binding_component_are_deterministic(tmp_path: Path) -> None:
    assert P.deterministic_world_order(("b", "a", "c"), 18423, 1) == P.deterministic_world_order(("c", "b", "a"), 18423, 1)
    history = pd.DataFrame([
        {"epoch": 3, "validation_loss": 0.25, "checkpoint_path": str(tmp_path / "three.pt"), "checkpoint_sha256": "3" * 64},
        {"epoch": 1, "validation_loss": 0.25, "checkpoint_path": str(tmp_path / "one.pt"), "checkpoint_sha256": "1" * 64},
        {"epoch": 2, "validation_loss": 0.50, "checkpoint_path": str(tmp_path / "two.pt"), "checkpoint_sha256": "2" * 64},
    ])
    assert P.select_checkpoint(history).selected_epoch == 1
    model = P.PersistentSearchHRM(); optimizer = torch.optim.AdamW(model.parameters(), lr=P.LEARNING_RATE, weight_decay=P.WEIGHT_DECAY)
    binding = _complete_training_binding()
    checkpoint = tmp_path / "exact.pt"; P.save_training_checkpoint(checkpoint, model, optimizer, 1, binding)
    for field, replacement in (("source_hashes", {"source": "changed"}), ("trace_hash", "changed"), ("model", {"seed": 0}), ("optimizer", {"name": "SGD"}), ("gate", {"version": 2})):
        with pytest.raises(ValueError, match="new output directory"):
            P.load_training_checkpoint(checkpoint, model, optimizer, {**binding, field: replacement})


def _complete_training_binding() -> dict[str, object]:
    return {
        "binding_schema_version": "c13p-training-binding-v1",
        "experiment_schema_version": P.SCHEMA_VERSION,
        "source_hashes": {"implementation": "a" * 64, "preregistration": "b" * 64},
        "trace_dataset_hash": "c" * 64,
        "trace_generation_fingerprint": "d" * 64,
        "model_config": {"model_seed": P.MODEL_SEED, "hidden_dim": P.HIDDEN_DIM, "num_layers": P.NUM_LAYERS, "num_heads": P.NUM_HEADS, "k_step": P.K_STEP},
        "optimizer_config": {"name": "AdamW", "learning_rate": P.LEARNING_RATE, "weight_decay": P.WEIGHT_DECAY, "grad_clip_norm": P.GRAD_CLIP_NORM},
        "gate_config": {"schema_version": "c13p-gates-v1", "fingerprint": "e" * 64},
    }


def test_complete_training_binding_rejects_missing_or_drifted_required_groups() -> None:
    binding = _complete_training_binding()
    assert isinstance(P.validate_training_binding(binding), str)
    for field in ("binding_schema_version", "experiment_schema_version", "source_hashes", "trace_dataset_hash", "trace_generation_fingerprint", "model_config", "optimizer_config", "gate_config"):
        incomplete = dict(binding); incomplete.pop(field)
        with pytest.raises(ValueError, match="new output directory"):
            P.validate_training_binding(incomplete)
    for field, replacement in (("source_hashes", {"implementation": "z" * 64}), ("model_config", {"model_seed": 0}), ("optimizer_config", {"name": "SGD"})):
        with pytest.raises(ValueError, match="new output directory"):
            P.validate_training_binding({**binding, field: replacement})


def test_prepared_lookup_requires_canonical_split_suite_world_identity() -> None:
    trace = _training_trace(world_index=4)
    prepared = _prepared_training_world()
    assert P._prepared_for_trace({"train/maze/4": prepared}, trace) is prepared
    for mapping in ({"4": prepared}, {("train", "maze", 4): prepared}, {"train/rooms/4": prepared}):
        with pytest.raises(ValueError, match="prepared representation"):
            P._prepared_for_trace(mapping, trace)


def test_terminal_patience_resume_returns_without_new_training_or_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class CpuPersistentSearchHRM(P.PersistentSearchHRM):
        def to(self, *args: object, **kwargs: object) -> "CpuPersistentSearchHRM":
            return self
    cfg = P.resolve_paths(tmp_path, tmp_path / "run")
    binding = _complete_training_binding()
    model = CpuPersistentSearchHRM(); optimizer = torch.optim.AdamW(model.parameters(), lr=P.LEARNING_RATE, weight_decay=P.WEIGHT_DECAY)
    latest = cfg.out_dir / "checkpoints" / "latest.pt"; latest.parent.mkdir(parents=True)
    P.save_training_checkpoint(latest, model, optimizer, 5, binding)
    rows = []
    for epoch, loss in enumerate((1.0, 2.0, 2.0, 2.0, 2.0), start=1):
        rows.append({"epoch": epoch, "validation_loss": loss, "checkpoint_path": str(cfg.out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"), "checkpoint_sha256": str(epoch) * 64})
    history_path = cfg.out_dir / "results" / "training_history.csv"; history_path.parent.mkdir(parents=True); pd.DataFrame(rows).to_csv(history_path, index=False)
    monkeypatch.setattr(P.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(P, "PersistentSearchHRM", CpuPersistentSearchHRM)
    monkeypatch.setattr(P, "train_one_world", lambda *args, **kwargs: pytest.fail("terminal resume trained another world"))
    selected = P.train_stationary_model((), (), {}, cfg, binding)
    assert selected.selected_epoch == 1 and len(pd.read_csv(history_path)) == 5
    assert not (cfg.out_dir / "checkpoints" / "epoch_006.pt").exists()


def test_task4_has_no_duplicate_top_level_definitions() -> None:
    import ast
    for path in (Path(P.__file__), Path(__file__)):
        names = [node.name for node in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        assert len(names) == len(set(names)), path


def _offline_trace(world_index: int, suite: str, event_count: int) -> P.TeacherTrace:
    trace = _training_trace(event_count, world_index=world_index)
    return replace(trace, split="development", suite=suite)


def _g1_world_metrics(mrr_delta: float, top1_delta: float, positive_suites: int = 6) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in _expected_development_registry():
        suite_index = int(record["suite"].split("-")[1]); suite_delta = mrr_delta if suite_index < positive_suites else -mrr_delta
        for arm, mrr, top1 in (("c13p_persistent", 0.50 + suite_delta, 0.40 + top1_delta), ("c13p_reset", 0.50, 0.40), ("c13m_base_rank", 0.25, 0.20)):
            rows.append({**record, "aggregation_level": "world", "world_id": f"development/{record['suite']}/{record['world_index']}", "arm": arm, "reciprocal_rank": mrr, "top1": top1})
    return pd.DataFrame(rows)


def _official_offline_traces(event_count: int = 2, long_tail: bool = False) -> tuple[P.TeacherTrace, ...]:
    traces: list[P.TeacherTrace] = []
    for ordinal, record in enumerate(_expected_development_registry()):
        trace = _offline_trace(int(record["world_index"]), str(record["suite"]), 3 if long_tail and ordinal else event_count)
        traces.append(replace(trace, world_seed=int(record["world_seed"]), roadmap_seed=int(record["roadmap_seed"]), feature_cache_path=str(record["feature_cache_path"]), feature_cache_sha256=str(record["feature_cache_sha256"])))
    return tuple(traces)


def test_offline_arms_use_matched_recorded_frontiers_and_shared_model_audit() -> None:
    traces = _official_offline_traces()
    prepared = {P._trace_world_id(trace): _prepared_training_world() for trace in traces}
    model = P.PersistentSearchHRM()

    events, summary = P.evaluate_offline_arms(traces, prepared, model, "a" * 64, P.resolve_paths(Path.cwd()), expected_development=_expected_development_registry())

    assert P.OFFLINE_ARMS == ("c13p_persistent", "c13p_reset", "c13m_base_rank")
    assert set(events["arm"]) == set(P.OFFLINE_ARMS)
    assert {"split", "suite", "world_index", "event_index", "arm", "positive_node", "candidate_count", "cross_entropy", "positive_rank", "reciprocal_rank", "top1", "rank_percentile", "candidate_nodes", "checkpoint_sha256", "model_state_sha256"}.issubset(events.columns)
    paired = events[events["arm"].isin(("c13p_persistent", "c13p_reset"))]
    for _, group in paired.groupby(["world_id", "event_index"], sort=True):
        assert len(group) == 2
        assert group["candidate_nodes"].nunique() == 1
        assert group["candidate_count"].nunique() == 1
        assert group["positive_node"].nunique() == 1
        assert group["checkpoint_sha256"].nunique() == 1
        assert group["model_state_sha256"].nunique() == 1
    assert set(summary["aggregation_level"]) == {"world", "suite", "pooled"}


def test_offline_base_rank_orders_frozen_tuple_and_world_macro_beats_trace_length() -> None:
    traces = _official_offline_traces(event_count=1, long_tail=True)
    prepared = {P._trace_world_id(trace): _prepared_training_world() for trace in traces}
    events, summary = P.evaluate_offline_arms(traces, prepared, P.PersistentSearchHRM(), "b" * 64, P.resolve_paths(Path.cwd()), expected_development=_expected_development_registry())

    base = events[(events["arm"] == "c13m_base_rank") & (events["event_index"] == 0)]
    assert set(base["positive_rank"]) == {1}
    pooled = summary[(summary["aggregation_level"] == "pooled") & (summary["arm"] == "c13m_base_rank")].iloc[0]
    worlds = summary[(summary["aggregation_level"] == "world") & (summary["arm"] == "c13m_base_rank")]
    assert pooled["reciprocal_rank"] == pytest.approx(worlds["reciprocal_rank"].mean())


def test_world_clustered_bootstrap_samples_only_exactly_twenty_four_paired_worlds() -> None:
    registry = _expected_development_registry()
    paired = P._g1_paired_world_rows(_g1_world_metrics(0.10, 0.02), expected_development=registry)
    first = P.world_clustered_bootstrap(paired, "mrr_delta", 200, P.BOOTSTRAP_SEEDS["g1_mrr"], expected_development=registry)
    second = P.world_clustered_bootstrap(paired, "mrr_delta", 200, P.BOOTSTRAP_SEEDS["g1_mrr"], expected_development=registry)
    assert first == second and first["n_worlds"] == 24 and first["sample_shape"] == (200, 24)
    with pytest.raises(ValueError, match="24|identity"):
        P.world_clustered_bootstrap(paired.iloc[:-1], "mrr_delta", 200, P.BOOTSTRAP_SEEDS["g1_mrr"], expected_development=registry)
    with pytest.raises(ValueError, match="unique"):
        P.world_clustered_bootstrap(pd.concat((paired, paired.iloc[[0]]), ignore_index=True), "mrr_delta", 200, P.BOOTSTRAP_SEEDS["g1_mrr"], expected_development=registry)


def test_g1_boundaries_use_unrounded_paired_world_metrics() -> None:
    zero = P.g1_verdict(_g1_world_metrics(0.0, 0.02), P.BOOTSTRAP_SEEDS["g1_mrr"], 200, expected_development=_expected_development_registry())
    below = P.g1_verdict(_g1_world_metrics(0.10, 0.019), P.BOOTSTRAP_SEEDS["g1_mrr"], 200, expected_development=_expected_development_registry())
    boundary = P.g1_verdict(_g1_world_metrics(0.10, 0.02), P.BOOTSTRAP_SEEDS["g1_mrr"], 200, expected_development=_expected_development_registry())
    suite_failure = P.g1_verdict(_g1_world_metrics(0.10, 0.02, positive_suites=3), P.BOOTSTRAP_SEEDS["g1_mrr"], 200, expected_development=_expected_development_registry())

    assert zero["pooled_mrr_ci_low"] == 0.0 and zero["verdict"] == "c13p_no_persistent_ranking_signal"
    assert below["pooled_top1_delta"] < 0.02 and below["verdict"] == "c13p_no_persistent_ranking_signal"
    assert boundary["pooled_top1_delta"] == pytest.approx(0.02) and boundary["verdict"] == "c13p_g1_passed"
    assert suite_failure["suites_with_positive_mrr"] == 3 and suite_failure["verdict"] == "c13p_no_persistent_ranking_signal"


def test_g1_rejects_missing_duplicate_or_unpaired_world_arm_identities() -> None:
    metrics = _g1_world_metrics(0.10, 0.02)
    for malformed in (
        metrics.iloc[:-1],
        pd.concat((metrics, metrics.iloc[[0]]), ignore_index=True),
        metrics.assign(arm=metrics["arm"].replace({"c13m_base_rank": "unexpected"})),
    ):
        with pytest.raises(ValueError, match="pair|arm|unique|complete"):
            P.g1_verdict(malformed, P.BOOTSTRAP_SEEDS["g1_mrr"], 20, expected_development=_expected_development_registry())


def _expected_development_registry() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for suite_index in range(6):
        suite = f"suite-{suite_index}"
        for world_index in range(4):
            ordinal = suite_index * 4 + world_index
            records.append({
                "split": "development", "suite": suite, "world_index": world_index,
                "world_seed": 1000 + ordinal, "roadmap_seed": 2000 + ordinal,
                "feature_cache_sha256": f"{ordinal + 1:064x}", "node_count": 2,
                "edge_count": 1, "feature_cache_path": f"cache-{ordinal}.npz",
            })
    return records


def test_offline_identity_registry_rejects_noncanonical_duplicate_and_source_drift() -> None:
    registry = _expected_development_registry()
    assert len(P.validate_expected_development_registry(registry)) == 24
    malformed = [
        registry[:-1],
        [*registry, dict(registry[0])],
        [dict(record, split="train") if index == 0 else record for index, record in enumerate(registry)],
        [dict(record, world_index=9) if index == 0 else record for index, record in enumerate(registry)],
    ]
    for candidate in malformed:
        with pytest.raises(ValueError, match="development|duplicate|24|canonical|suite"):
            P.validate_expected_development_registry(candidate)


def test_offline_summary_rejects_missing_arm_event_and_cross_arm_identity_drift() -> None:
    registry = _expected_development_registry()
    rows: list[dict[str, object]] = []
    for record in registry:
        for arm in P.OFFLINE_ARMS:
            rows.append({
                **record, "world_id": f"development/{record['suite']}/{record['world_index']}",
                "event_index": 0, "arm": arm, "positive_node": 0, "candidate_count": 2,
                "candidate_nodes": (0, 1), "checkpoint_sha256": "a" * 64,
                "model_state_sha256": "b" * 64, "cross_entropy": 1.0,
                "positive_rank": 1.0, "reciprocal_rank": 1.0, "top1": 1.0,
                "rank_percentile": 0.0,
            })
    events = pd.DataFrame(rows)
    assert set(P._offline_summary(events, expected_development=registry)["aggregation_level"]) == {"world", "suite", "pooled"}
    with pytest.raises(ValueError, match="cross-product|arms"):
        P._offline_summary(events.iloc[:-1], expected_development=registry)
    drifted = events.copy(); drifted.loc[drifted.index[1], "feature_cache_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="identity|audit"):
        P._offline_summary(drifted, expected_development=registry)


def test_g1_and_bootstrap_require_the_audited_development_identity_registry() -> None:
    registry = _expected_development_registry()
    metrics = _g1_world_metrics(0.10, 0.02)
    for ordinal, record in enumerate(registry):
        mask = metrics["world_id"] == f"development/{record['suite']}/{record['world_index']}"
        metrics.loc[mask, ["world_seed", "roadmap_seed", "feature_cache_sha256", "node_count", "edge_count", "feature_cache_path"]] = (
            record["world_seed"], record["roadmap_seed"], record["feature_cache_sha256"], record["node_count"], record["edge_count"], record["feature_cache_path"],
        )
    assert P.g1_verdict(metrics, P.BOOTSTRAP_SEEDS["g1_mrr"], 100, expected_development=registry)["verdict"] == "c13p_g1_passed"
    paired = P._g1_paired_world_rows(metrics, expected_development=registry)
    assert P.world_clustered_bootstrap(paired, "mrr_delta", 100, P.BOOTSTRAP_SEEDS["g1_mrr"], expected_development=registry)["n_worlds"] == 24
    arbitrary = paired.copy(); arbitrary["world_id"] = [f"w-{index}" for index in range(24)]
    wrong_seed = paired.copy(); wrong_seed.loc[0, "world_seed"] = -1
    for malformed in (arbitrary, wrong_seed):
        with pytest.raises(ValueError, match="identity|canonical|registry|development"):
            P.world_clustered_bootstrap(malformed, "mrr_delta", 20, P.BOOTSTRAP_SEEDS["g1_mrr"], expected_development=registry)
