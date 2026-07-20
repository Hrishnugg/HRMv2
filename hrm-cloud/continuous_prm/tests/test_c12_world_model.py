import math
import json
import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import continuous_prm_c12_world_model as W
import continuous_prm_c12_latent_dynamics as L
import continuous_prm_c12_persistent as P


def _cfg(**overrides):
    values = dict(
        d_model=32,
        horizon=4,
        max_patrollers=2,
        max_gates=3,
        raster_channels=5,
        core_width=32,
        transformer_depth=2,
        transformer_heads=4,
        transformer_ff_width=64,
        transformer_window=3,
        onlstm_chunk_size=4,
        hrm_slow_cadence=2,
        snapshot_depth=2,
        dropout=0.0,
    )
    values.update(overrides)
    return W.WorldModelConfig(**values)


def _batch(cfg, batch_size=2, identity_mask=None, gate_mask=None):
    torch.manual_seed(3)
    if identity_mask is None:
        identity_mask = torch.ones(batch_size, cfg.max_patrollers)
    if gate_mask is None:
        gate_mask = torch.ones(batch_size, cfg.max_gates)
    return {
        "frame_rasters": torch.rand(batch_size, cfg.raster_channels, 16, 16),
        "centers": torch.rand(batch_size, cfg.max_patrollers, 2),
        "radii": torch.rand(batch_size, cfg.max_patrollers),
        "identity_mask": identity_mask,
        "visible_regime_context": torch.rand(batch_size, 3),
        "visible_regime_mask": torch.ones(batch_size, 1),
        "gate_mask": gate_mask,
    }


def _all_tensors(tree):
    if torch.is_tensor(tree):
        return [tree]
    if isinstance(tree, dict):
        return [item for value in tree.values() for item in _all_tensors(value)]
    if isinstance(tree, (tuple, list)):
        return [item for value in tree for item in _all_tensors(value)]
    return []


def test_frame_encoder_ignores_masked_object_slots():
    cfg = _cfg()
    encoder = W.FrameEncoder(cfg).eval()
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    batch = _batch(cfg, identity_mask=mask)
    changed = {name: value.clone() for name, value in batch.items()}
    changed["centers"][0, 1] = torch.tensor([99.0, -99.0])
    changed["radii"][0, 1] = 99.0
    with torch.no_grad():
        a = encoder.from_batch(batch)
        b = encoder.from_batch(changed)
    assert a.shape == (2, cfg.d_model)
    assert torch.equal(a[0], b[0])
    assert torch.isfinite(a).all()


def test_direct_decoder_shapes_and_zeroes_masked_slots():
    cfg = _cfg()
    decoder = W.DirectHorizonDecoder(cfg)
    identity = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    gates = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    output = decoder(torch.randn(2, cfg.d_model), identity, gates)
    assert output["center_displacements"].shape == (2, 4, 2, 2)
    assert output["gate_logits"].shape == (2, 4, 3)
    assert torch.count_nonzero(output["center_displacements"][0, :, 1]) == 0
    assert torch.count_nonzero(output["gate_logits"][0, :, 1]) == 0
    assert all(torch.isfinite(value).all() for value in output.values())


def test_snapshot_ignores_earlier_frames():
    cfg = _cfg()
    core = W.SnapshotCore(cfg.d_model, cfg.core_width, cfg.snapshot_depth).eval()
    x0, x1 = torch.randn(2, cfg.d_model), torch.randn(2, cfg.d_model)
    _, carry = core.step(x0, core.initial_carry(2, x0.device))
    with_history, _ = core.step(x1, carry)
    fresh, _ = core.step(x1, core.initial_carry(2, x1.device))
    assert torch.equal(with_history, fresh)


def test_lstm_carry_persists_and_resets_exactly():
    cfg = _cfg()
    core = W.LSTMCore(cfg.d_model, cfg.core_width).eval()
    x0, x1 = torch.randn(2, cfg.d_model), torch.randn(2, cfg.d_model)
    _, carry = core.step(x0, core.initial_carry(2, x0.device))
    persistent, _ = core.step(x1, carry)
    initial = core.initial_carry(2, x1.device)
    fresh, _ = core.step(x1, initial)
    assert not torch.allclose(persistent, fresh)
    reset = W.reset_carry(carry, torch.ones(2, dtype=torch.bool), initial)
    reset_output, _ = core.step(x1, reset)
    assert torch.equal(reset_output, fresh)


def test_transformer_is_causal_and_keeps_exact_window():
    cfg = _cfg()
    core = W.SlidingWindowTransformerCore(
        cfg.d_model,
        cfg.transformer_window,
        cfg.transformer_depth,
        cfg.transformer_heads,
        cfg.transformer_ff_width,
        dropout=0.0,
    ).eval()
    xs = [torch.randn(2, cfg.d_model) for _ in range(4)]
    carry = core.initial_carry(2, xs[0].device)
    outputs = []
    for value in xs:
        output, carry = core.step(value, carry)
        outputs.append(output)
    assert torch.equal(carry["history"], torch.stack(xs[-3:], dim=1))
    assert carry["length"].tolist() == [3, 3]

    replay = core.initial_carry(2, xs[0].device)
    for value in xs[:2]:
        replay_output, replay = core.step(value, replay)
    assert torch.equal(outputs[1], replay_output)


def test_forecast_loss_matches_hand_calculation_and_masks_padding():
    pred_centers = torch.zeros(1, 2, 2, 2)
    true_centers = torch.ones_like(pred_centers)
    gate_logits = torch.zeros(1, 2, 2)
    true_gates = torch.ones_like(gate_logits)
    identity = torch.tensor([[1.0, 0.0]])
    gate_mask = torch.tensor([[1.0, 0.0]])
    loss = W.forecast_loss(
        pred_centers,
        gate_logits,
        true_centers,
        true_gates,
        identity,
        gate_mask,
    )
    assert loss["center_huber"].item() == pytest.approx(0.5)
    assert loss["gate_bce"].item() == pytest.approx(math.log(2.0))
    assert loss["total"].item() == pytest.approx(0.5 + 0.5 * math.log(2.0))


def test_gradients_reach_encoder_temporal_core_and_decoder():
    cfg = _cfg()
    model = W.build_world_model("lstm", cfg).train()
    batch = _batch(cfg)
    prediction, _ = model.step(batch, model.initial_carry(2, torch.device("cpu")))
    target_centers = torch.randn_like(prediction["center_displacements"])
    target_gates = torch.randint(
        0, 2, prediction["gate_logits"].shape, dtype=prediction["gate_logits"].dtype
    )
    loss = W.forecast_loss(
        prediction["center_displacements"],
        prediction["gate_logits"],
        target_centers,
        target_gates,
        batch["identity_mask"],
        batch["gate_mask"],
    )["total"]
    loss.backward()
    for module in (model.encoder, model.core, model.decoder):
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
            for parameter in module.parameters()
        )


def test_onlstm_master_gates_are_ordered_and_valid():
    cell = W.ONLSTMCell(input_size=16, hidden_size=32, chunk_size=4)
    forget, update = cell.master_gates(torch.randn(3, 16), torch.randn(3, 32))
    assert torch.all((0.0 <= forget) & (forget <= 1.0))
    assert torch.all((0.0 <= update) & (update <= 1.0))
    assert torch.all(torch.diff(forget, dim=-1) >= -1e-7)
    assert torch.all(torch.diff(update, dim=-1) <= 1e-7)


def test_hrm_fast_updates_every_step_and_slow_only_at_cadence():
    cfg = _cfg(hrm_slow_cadence=3)
    core = W.HRMStreamCore(cfg.d_model, cfg.core_width, cfg.hrm_slow_cadence).eval()
    x = torch.randn(2, cfg.d_model)
    carry0 = core.initial_carry(2, x.device)
    _, carry1 = core.step(x, carry0)
    _, carry2 = core.step(x, carry1)
    _, carry3 = core.step(x, carry2)
    assert not torch.equal(carry0["fast"], carry1["fast"])
    assert torch.equal(carry0["slow"], carry1["slow"])
    assert torch.equal(carry1["slow"], carry2["slow"])
    assert not torch.equal(carry2["slow"], carry3["slow"])
    assert carry3["step"].tolist() == [3, 3]


def test_carry_detach_preserves_values_and_breaks_graph_history():
    cfg = _cfg()
    core = W.HRMStreamCore(cfg.d_model, cfg.core_width, cfg.hrm_slow_cadence)
    x = torch.randn(2, cfg.d_model, requires_grad=True)
    _, carry = core.step(x, core.initial_carry(2, x.device))
    detached = W.detach_carry(carry)
    original_tensors = _all_tensors(carry)
    detached_tensors = _all_tensors(detached)
    assert len(original_tensors) == len(detached_tensors)
    for original, clean in zip(original_tensors, detached_tensors):
        assert torch.equal(original, clean)
        assert clean.grad_fn is None


def test_boundary_reset_only_clears_selected_streams():
    cfg = _cfg()
    core = W.LSTMCore(cfg.d_model, cfg.core_width)
    x = torch.randn(2, cfg.d_model)
    _, carry = core.step(x, core.initial_carry(2, x.device))
    initial = core.initial_carry(2, x.device)
    reset = W.reset_carry(carry, torch.tensor([True, False]), initial)
    for before, after, zero in zip(
        _all_tensors(carry), _all_tensors(reset), _all_tensors(initial)
    ):
        assert torch.equal(after[0], zero[0])
        assert torch.equal(after[1], before[1])


@pytest.mark.parametrize("arm", ["snapshot", "lstm", "temporal_transformer", "onlstm", "hrm_stream"])
def test_all_arms_are_deterministic_and_accounted(arm):
    cfg = _cfg()
    model = W.build_world_model(arm, cfg).eval()
    batch = _batch(cfg)
    carry = model.initial_carry(2, torch.device("cpu"))
    with torch.no_grad():
        first, first_carry = model.step(batch, carry)
        second, second_carry = model.step(batch, carry)
    assert torch.equal(first["center_displacements"], second["center_displacements"])
    assert torch.equal(first["gate_logits"], second["gate_logits"])
    for left, right in zip(_all_tensors(first_carry), _all_tensors(second_carry)):
        assert torch.equal(left, right)
    accounting = W.model_accounting(model, raster_size=16)
    assert accounting == W.model_accounting(model, raster_size=16)
    assert accounting["trainable_parameters"] > 0
    assert accounting["estimated_madds_per_step"] > 0


def test_window_reencode_uses_same_weights_as_fresh_sequential_replay():
    cfg = _cfg(transformer_window=3)
    model = W.build_world_model("lstm", cfg).eval()
    batches = [_batch(cfg) for _ in range(3)]
    with torch.no_grad():
        direct, _ = model.run_sequence(batches)
        replay = model.window_reencode(batches)
    assert torch.equal(direct[-1]["center_displacements"], replay["center_displacements"])
    assert torch.equal(direct[-1]["gate_logits"], replay["gate_logits"])


def test_hrm_fast_and_slow_paths_both_receive_sequence_gradients():
    cfg = _cfg(hrm_slow_cadence=2)
    core = W.HRMStreamCore(cfg.d_model, cfg.core_width, cfg.hrm_slow_cadence)
    carry = core.initial_carry(2, torch.device("cpu"))
    contexts = []
    for _ in range(4):
        context, carry = core.step(torch.randn(2, cfg.d_model), carry)
        contexts.append(context)
    torch.stack(contexts).square().mean().backward()
    assert torch.count_nonzero(core.fast_cell.weight_hh.grad) > 0
    assert torch.count_nonzero(core.slow_cell.weight_hh.grad) > 0


def test_constant_predictions_trigger_failed_validation_diagnostic():
    diagnostic = W.collapse_diagnostics(
        center_sum=0.0,
        center_square_sum=0.0,
        center_count=100,
        gate_sum=50.0,
        gate_square_sum=25.0,
        gate_count=100,
        threshold=1e-6,
    )
    assert diagnostic["collapsed"] is True
    assert diagnostic["validation_status"] == "failed_constant_output"


def test_tiny_alias_sanity_separates_temporal_memory_from_snapshot():
    result = W.tiny_alias_sanity(optimization_steps=100)
    assert result["snapshot"]["accuracy"] == pytest.approx(0.5)
    for arm in ("lstm", "temporal_transformer", "onlstm", "hrm_stream"):
        assert result[arm]["accuracy"] >= 0.95


def test_stateful_training_resumes_atomically_without_test_access_or_duplicates(tmp_path):
    dynamics_cfg = L.C12DynamicsConfig(
        raster_size=8,
        episode_steps=12,
        burn_in=4,
        forecast_horizon=4,
        alias_horizon=4,
    )

    def fixture_builder(stratum, pair_index, map_family, cfg, split):
        return L.make_fixture_pair(stratum, pair_index, cfg=cfg, split=split)

    dataset_root = tmp_path / "dataset_run"
    P.collect_dataset(
        dataset_root,
        scale="smoke",
        cfg=dynamics_cfg,
        counts_by_split={"TRAIN": 2, "VALIDATION": 2, "TEST": 2},
        episodes_per_shard=2,
        episode_builder=fixture_builder,
    )
    store = W.C12ShardStore(dataset_root)
    model_cfg = _cfg(
        d_model=16,
        horizon=4,
        max_patrollers=1,
        max_gates=2,
        core_width=16,
        transformer_heads=4,
        transformer_ff_width=32,
        onlstm_chunk_size=4,
        snapshot_depth=1,
    )
    train_cfg = W.TrainingConfig(
        batch_size=2,
        tbptt_steps=4,
        max_epochs=2,
        patience=2,
        collapse_std_threshold=0.0,
    )
    run_root = tmp_path / "training_run"
    partial = W.train_world_model(
        store,
        run_root,
        "lstm",
        seed=7,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        device="cpu",
        stop_after_epochs=1,
    )
    assert partial["status"] == "running"
    complete = W.train_world_model(
        store,
        run_root,
        "lstm",
        seed=7,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        device="cpu",
    )
    assert complete["status"] == "complete"
    assert complete["epochs_completed"] == 2
    assert set(store.requested_splits) == {"TRAIN", "VALIDATION"}
    assert "TEST" not in store.requested_splits
    manifest = json.loads((run_root / "manifest.json").read_text())
    assert len(manifest["training_runs"]) == 1
    assert Path(complete["last_checkpoint"]).exists()
    assert Path(complete["best_checkpoint"]).exists()
