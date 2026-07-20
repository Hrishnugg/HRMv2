from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
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

def test_encoder_is_frozen_and_prepared_rank_is_the_locked_local_bellman_formula(tmp_path: Path) -> None:
    encoder = P.load_frozen_flat_encoder(_task3_source(tmp_path), torch.device("cpu")); prepared = P.prepare_world_representation(_task3_features(), [[(1,1.)],[(0,1.)]], 1, P.resolve_paths(tmp_path), encoder)
    assert prepared.node_embeddings.shape == (2,64) and not encoder.training and all(not p.requires_grad for p in encoder.parameters())
    np.testing.assert_allclose(prepared.base_rank, prepared.euclidean_rank + 1.50*(prepared.local_values-prepared.euclidean_rank))

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
    with pytest.raises(ValueError,match="stale"):
        first.update(torch.ones(1,70),carry)
    foreign=P.PersistentCarryLifecycle(model,"eval-b"); other=foreign.initial_for_world("world-b",1,torch.device("cpu"),torch.float32)
    with pytest.raises(ValueError,match="foreign"):
        first.update(torch.ones(1,70),other)
    _,after=first.update(torch.ones(1,70),next_carry)
    assert after.step==2

def test_lifecycle_rejects_stale_and_foreign_carries() -> None:
    model=P.PersistentSearchHRM(); first=P.PersistentCarryLifecycle(model,"eval-a"); carry=first.initial_for_world("world-a",1,torch.device("cpu"),torch.float32)
    _,next_carry=first.update(torch.ones(1,70),carry)
    with pytest.raises(ValueError): first.update(torch.ones(1,70),carry)
    foreign=P.PersistentCarryLifecycle(model,"eval-b"); other=foreign.initial_for_world("world-b",1,torch.device("cpu"),torch.float32)
    with pytest.raises(ValueError): first.update(torch.ones(1,70),other)
    assert first.update(torch.ones(1,70),next_carry)[1].step==2
