import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import continuous_prm_c12_latent_dynamics as L
import continuous_prm_c12_persistent as P


def test_baseline_forecast_contracts_and_visible_control():
    hidden, _ = L.make_fixture_pair("direction_alias")
    control, _ = L.make_fixture_pair("present_sufficient")

    frozen = P.FrozenFrameProvider()
    frozen.reset(hidden)
    for t in range(hidden.alias_time - 1, hidden.alias_time + 1):
        frozen.observe(hidden, t, hidden.observe(t))
    frozen_forecast = frozen.forecast(hidden, hidden.alias_time, 8)
    assert np.all(frozen_forecast.centers == frozen_forecast.centers[0])

    cv = P.ConstantVelocityProvider()
    cv.reset(hidden)
    for t in range(hidden.alias_time - 1, hidden.alias_time + 1):
        cv.observe(hidden, t, hidden.observe(t))
    cv_forecast = cv.forecast(hidden, hidden.alias_time, 8)
    assert not np.array_equal(cv_forecast.centers[0], cv_forecast.centers[1])

    true_mode = P.TrueModeProvider()
    true_mode.reset(hidden)
    true_mode.observe(hidden, hidden.alias_time, hidden.observe(hidden.alias_time))
    privileged = true_mode.forecast(hidden, hidden.alias_time, 8)
    oracle = P.OracleFutureProvider().forecast(hidden, hidden.alias_time, 8)
    assert not np.array_equal(privileged.centers, oracle.centers)

    true_mode.reset(control)
    true_mode.observe(control, control.alias_time, control.observe(control.alias_time))
    visible = true_mode.forecast(control, control.alias_time, 8)
    visible_oracle = P.OracleFutureProvider().forecast(control, control.alias_time, 8)
    assert np.array_equal(visible.centers, visible_oracle.centers)
    assert np.array_equal(visible.gate_open, visible_oracle.gate_open)


def _passing_metrics():
    return {
        "alias_rate": 0.15,
        "history_completion_gain": 0.15,
        "history_regret_reduction_frac": 0.0,
        "history_regret_reduction_ci_low": -0.1,
        "oracle_completion": 0.85,
        "ceiling_gap_frac": 0.20,
        "control_headroom": 0.02,
        "challenge_headroom": {
            "direction_alias": 0.20,
            "slow_gate_phase": 0.18,
            "route_mode_junction": 0.22,
        },
        "control_margin_required": 0.10,
    }


def test_g0_gate_thresholds_are_inclusive_at_preregistered_boundaries():
    verdict = P.evaluate_g0_gates(_passing_metrics())
    assert verdict["passed"]
    assert all(verdict["conditions"].values())


@pytest.mark.parametrize(
    "field,value,condition",
    [
        ("alias_rate", 0.1499, "aliasing_exists"),
        ("history_completion_gain", 0.1499, "history_matters"),
        ("oracle_completion", 0.8499, "ceiling_exists"),
        ("ceiling_gap_frac", 0.1999, "ceiling_exists"),
    ],
)
def test_g0_gate_failure_cases(field, value, condition):
    metrics = _passing_metrics()
    metrics[field] = value
    verdict = P.evaluate_g0_gates(metrics)
    assert not verdict["conditions"][condition]
    assert not verdict["passed"]


def test_regret_branch_can_authorize_history_when_completion_gain_is_small():
    metrics = _passing_metrics()
    metrics["history_completion_gain"] = 0.02
    metrics["history_regret_reduction_frac"] = 0.25
    metrics["history_regret_reduction_ci_low"] = 0.01
    assert P.evaluate_g0_gates(metrics)["conditions"]["history_matters"]


def test_small_fixture_probe_writes_canonical_artifacts(tmp_path):
    cfg = L.C12DynamicsConfig(raster_size=16)

    def fixture_builder(stratum, pair_index, map_family, cfg, split):
        return L.make_fixture_pair(stratum, pair_index, cfg=cfg, split=split)

    summary = P.run_probe(
        out_dir=tmp_path,
        scale="smoke",
        cfg=cfg,
        pairs_per_stratum=2,
        episode_builder=fixture_builder,
        bootstrap_samples=200,
    )
    raw_path = tmp_path / "results" / "c12a_headroom_raw.csv"
    summary_path = tmp_path / "results" / "c12a_headroom_summary.json"
    report_path = tmp_path / "results" / "C12A_G0_PROBE_REPORT.md"
    manifest_path = tmp_path / "probe_manifest.json"
    assert raw_path.exists() and summary_path.exists() and report_path.exists() and manifest_path.exists()

    with raw_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 * 2 * 2 * 4  # strata * pairs * variants * providers
    assert {row["provider"] for row in rows} == set(P.PROVIDER_NAMES)
    assert summary["episodes_per_stratum"] == 4
    assert summary["config_hash"]

    with summary_path.open() as f:
        disk_summary = json.load(f)
    assert disk_summary["config_hash"] == summary["config_hash"]


def test_probe_authorization_requires_matching_passed_hash(tmp_path):
    results = tmp_path / "results"
    results.mkdir(parents=True)
    path = results / "c12a_headroom_summary.json"
    path.write_text(json.dumps({"config_hash": "abc", "gates": {"passed": False}}))

    with pytest.raises(RuntimeError, match="did not pass"):
        P.assert_probe_authorized(tmp_path, "abc")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        P.assert_probe_authorized(tmp_path, "def", allow_failed_probe=True)
    loaded = P.assert_probe_authorized(tmp_path, "abc", allow_failed_probe=True)
    assert loaded["exploratory_override"] is True


def test_episode_record_has_fixed_shapes_and_separates_privileged_fields():
    cfg = L.C12DynamicsConfig(raster_size=16, episode_steps=40, burn_in=8, forecast_horizon=12)
    episode, _ = L.make_fixture_pair("route_mode_junction", pair_index=4, cfg=cfg, split="TRAIN")
    record = L.build_episode_record(episode)

    assert record.frame_rasters.shape == (40, 5, 16, 16)
    assert record.centers.shape == (40, 1, 2)
    assert record.radii.shape == (40, 1)
    assert record.identity_mask.shape == (40, 1)
    assert record.visible_regime_context.shape == (40, 3)
    assert record.visible_regime_mask.shape == (40, 1)
    assert record.target_center_displacements.shape == (40, 12, 1, 2)
    assert record.target_gate_open.shape == (40, 12, 2)
    assert record.gate_mask.shape == (40, 2)
    assert record.route_critical_mask.shape == (40, 12)
    assert record.route_edge_midpoints.shape == (2, 2)

    arrays = record.model_arrays()
    assert all(array.dtype != object for array in arrays.values())
    for forbidden in L.FORBIDDEN_MODEL_FIELDS:
        assert forbidden not in arrays
    assert record.privileged_diagnostics["route_mode"] in (-1, 1)
    assert "hazard_edge" in record.privileged_diagnostics
    assert record.metadata["split"] == "TRAIN"
    assert record.metadata["static_map_id"]


def test_present_sufficient_record_contains_visible_context_only_for_control():
    cfg = L.C12DynamicsConfig(raster_size=16, episode_steps=40, burn_in=8, forecast_horizon=12)
    hidden, _ = L.make_fixture_pair("slow_gate_phase", cfg=cfg, split="TRAIN")
    visible, _ = L.make_fixture_pair("present_sufficient", cfg=cfg, split="TRAIN")
    hidden_record = L.build_episode_record(hidden)
    visible_record = L.build_episode_record(visible)
    assert hidden_record.visible_regime_mask.sum() == 0
    assert np.all(hidden_record.visible_regime_context == 0)
    assert visible_record.visible_regime_mask.sum() == cfg.episode_steps
    assert np.any(visible_record.visible_regime_context != 0)


@pytest.mark.parametrize("stratum", L.CHALLENGE_STRATA)
def test_every_challenge_has_route_critical_support_at_horizons_17_to_32(stratum):
    cfg = L.C12DynamicsConfig()
    pair = L.make_fixture_pair(stratum, cfg=cfg, split="TRAIN")
    for episode in pair:
        record = L.build_episode_record(episode)
        long_horizon = record.route_critical_mask[
            episode.alias_time, 16 : cfg.forecast_horizon
        ]
        assert long_horizon.sum() > 0


def test_dataset_shards_round_trip_checksum_resume_and_config_refusal(tmp_path):
    cfg = L.C12DynamicsConfig(raster_size=16, episode_steps=40, burn_in=8, forecast_horizon=12)

    def fixture_builder(stratum, pair_index, map_family, cfg, split):
        return L.make_fixture_pair(stratum, pair_index, cfg=cfg, split=split)

    counts = {"TRAIN": 2, "VALIDATION": 2, "SMOKE": 2}
    manifest = P.collect_dataset(
        out_dir=tmp_path,
        scale="smoke",
        cfg=cfg,
        counts_by_split=counts,
        episodes_per_shard=2,
        episode_builder=fixture_builder,
    )
    assert manifest["status"] == "complete"
    assert manifest["episodes_total"] == len(L.STRATA) * sum(counts.values())
    assert set(manifest["splits"]) == set(counts)
    assert len(manifest["shards"]) == len(L.STRATA) * len(counts)

    for shard in manifest["shards"]:
        path = tmp_path / shard["path"]
        sidecar = tmp_path / shard["diagnostics_path"]
        assert path.exists() and sidecar.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == shard["sha256"]
        with np.load(path, allow_pickle=False) as data:
            assert data["frame_rasters"].shape[0] == 2
            assert data["target_center_displacements"].shape[2] == 12
            assert not any(data[name].dtype == object for name in data.files)

    mtimes = {shard["path"]: (tmp_path / shard["path"]).stat().st_mtime_ns for shard in manifest["shards"]}
    resumed = P.collect_dataset(
        out_dir=tmp_path,
        scale="smoke",
        cfg=cfg,
        counts_by_split=counts,
        episodes_per_shard=2,
        episode_builder=fixture_builder,
    )
    assert resumed["config_hash"] == manifest["config_hash"]
    assert mtimes == {
        shard["path"]: (tmp_path / shard["path"]).stat().st_mtime_ns
        for shard in resumed["shards"]
    }

    changed = L.C12DynamicsConfig(raster_size=20, episode_steps=40, burn_in=8, forecast_horizon=12)
    with pytest.raises(RuntimeError, match="config hash mismatch"):
        P.collect_dataset(
            out_dir=tmp_path,
            scale="smoke",
            cfg=changed,
            counts_by_split=counts,
            episodes_per_shard=2,
            episode_builder=fixture_builder,
        )


def test_dataset_inspection_reports_counts_periods_masks_and_disk(tmp_path):
    cfg = L.C12DynamicsConfig(raster_size=16, episode_steps=40, burn_in=8, forecast_horizon=12)

    def fixture_builder(stratum, pair_index, map_family, cfg, split):
        return L.make_fixture_pair(stratum, pair_index, cfg=cfg, split=split)

    P.collect_dataset(
        out_dir=tmp_path,
        scale="smoke",
        cfg=cfg,
        counts_by_split={"TRAIN": 2},
        episodes_per_shard=2,
        episode_builder=fixture_builder,
    )
    report = P.inspect_dataset(tmp_path)
    assert report["episodes_total"] == 2 * len(L.STRATA)
    assert cfg.fast_period_min <= report["fast_period_min"] <= report["fast_period_max"] <= cfg.fast_period_max
    assert cfg.slow_dwell_min <= report["slow_dwell_min"] <= report["slow_dwell_max"] <= cfg.slow_dwell_max
    assert 0.0 <= report["alias_rate"] <= 1.0
    assert report["missing_identity_slots"] == 0
    assert report["disk_bytes"] > 0


def test_forecast_metrics_match_hand_built_displacements_gates_and_occupancy():
    target = np.zeros((4, 2, 2), dtype=np.float32)
    prediction = target.copy()
    prediction[-1, 0, 0] = 1.0
    target_gates = np.asarray(
        [[True, False], [False, True], [True, False], [False, True]], dtype=np.bool_
    )
    logits = np.where(target_gates, 20.0, -20.0).astype(np.float32)
    rows = P.forecast_bucket_metrics(
        prediction,
        logits,
        target,
        target_gates,
        current_centers=np.asarray([[0.25, 0.25], [0.75, 0.75]], dtype=np.float32),
        radii=np.asarray([0.08, 0.08], dtype=np.float32),
        identity_mask=np.asarray([1, 0], dtype=np.uint8),
        gate_mask=np.asarray([1, 1], dtype=np.uint8),
        route_critical_mask=np.asarray([0, 0, 1, 1], dtype=np.uint8),
        route_edge_midpoints=np.asarray([[0.4, 0.4], [0.6, 0.6]], dtype=np.float32),
        raster_size=16,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["ade_sum"] == pytest.approx(1.0)
    assert row["ade_count"] == 4
    assert row["ade"] == pytest.approx(0.25)
    assert row["fde"] == pytest.approx(1.0)
    assert row["route_critical_ade"] == pytest.approx(0.5)
    assert row["gate_balanced_accuracy"] == pytest.approx(1.0)
    assert row["gate_brier"] < 1e-12
    assert row["route_critical_gate_brier_count"] == 4
    assert row["route_critical_gate_brier"] < 1e-12
    assert row["occupancy_hits"] <= row["occupancy_count"]
    assert row["route_critical_occupancy_hits"] <= row["route_critical_occupancy_count"]


def test_forecast_metrics_exclude_padded_identities_and_gates():
    target = np.zeros((4, 2, 2), dtype=np.float32)
    prediction = target.copy()
    prediction[:, 1] = 999.0
    target_gates = np.zeros((4, 2), dtype=np.float32)
    logits = np.full((4, 2), -20.0, dtype=np.float32)
    logits[:, 1] = 20.0
    row = P.forecast_bucket_metrics(
        prediction,
        logits,
        target,
        target_gates,
        current_centers=np.asarray([[0.25, 0.25], [0.75, 0.75]], dtype=np.float32),
        radii=np.asarray([0.08, 0.08], dtype=np.float32),
        identity_mask=np.asarray([1, 0], dtype=np.uint8),
        gate_mask=np.asarray([1, 0], dtype=np.uint8),
        route_critical_mask=np.ones(4, dtype=np.uint8),
        route_edge_midpoints=np.asarray([[0.4, 0.4], [0.6, 0.6]], dtype=np.float32),
        raster_size=16,
    )[0]
    assert row["ade"] == 0.0
    assert row["ade_count"] == 4
    assert row["gate_brier_count"] == 4
    assert row["gate_brier"] < 1e-12
    assert row["route_critical_gate_brier_count"] == 4
    assert row["route_critical_gate_brier"] < 1e-12


def test_g1_mechanism_metric_uses_gate_brier_only_for_slow_gate_stratum():
    horizon = 32
    target = np.zeros((horizon, 2, 2), dtype=np.float32)
    prediction = target.copy()
    prediction[16:, 0, 0] = 0.5
    prediction[16:, 1, 0] = 999.0
    target_gates = np.zeros((horizon, 2), dtype=np.float32)
    logits = np.full((horizon, 2), -20.0, dtype=np.float32)
    logits[16:, :] = 20.0
    args = (
        prediction,
        logits,
        target,
        target_gates,
        np.asarray([1, 0], dtype=np.uint8),
        np.asarray([1, 0], dtype=np.uint8),
        np.concatenate(
            [np.zeros(16, dtype=np.uint8), np.ones(16, dtype=np.uint8)]
        ),
    )
    center_sum, center_count, center_name = P._g1_mechanism_components(
        "direction_alias", *args
    )
    gate_sum, gate_count, gate_name = P._g1_mechanism_components(
        "slow_gate_phase", *args
    )
    route_sum, route_count, route_name = P._g1_mechanism_components(
        "route_mode_junction", *args
    )
    assert center_sum == pytest.approx(8.0)
    assert center_count == 16
    assert route_sum == pytest.approx(center_sum)
    assert route_count == center_count
    assert gate_sum > 15.9
    assert gate_count == 16
    assert center_name == route_name == "route_critical_center_ade_h17_32"
    assert gate_name == "route_critical_gate_brier_h17_32"


def test_world_bootstrap_sign_flip_and_bh_are_deterministic():
    values = [-3.0, -2.0, -1.0, -4.0]
    first = P.seeded_world_bootstrap(values, seed=9, samples=500)
    second = P.seeded_world_bootstrap(values, seed=9, samples=500)
    assert first == second
    assert first["ci_high"] < 0.0
    assert P.paired_sign_flip_p(values) <= 0.125
    assert P.bh_q_values([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])


def test_forecast_world_aggregation_uses_registered_metric_by_stratum():
    common = {
        "arm": "hrm_stream",
        "carry_mode": "persistent",
        "horizon_bucket": "h17_32",
        "pair_id": "pair-0",
        "eval_condition": "matched_id",
        "seed": 0,
        "route_critical_ade_sum": 4.0,
        "route_critical_ade_count": 2,
        "route_critical_gate_brier_sum": 0.5,
        "route_critical_gate_brier_count": 2,
    }
    aggregate = P._aggregate_forecast_worlds(
        [
            {**common, "stratum": "direction_alias"},
            {**common, "stratum": "slow_gate_phase"},
        ]
    )
    assert aggregate[
        (
            "hrm_stream",
            "persistent",
            "direction_alias",
            "h17_32",
            "pair-0",
            "matched_id",
        )
    ] == pytest.approx(2.0)
    assert aggregate[
        (
            "hrm_stream",
            "persistent",
            "slow_gate_phase",
            "h17_32",
            "pair-0",
            "matched_id",
        )
    ] == pytest.approx(0.25)


def test_g1_g2_g3_and_g4_boundary_contracts():
    g1_comparisons = {
        stratum: {
            "flat_recurrent": {"mean_difference": -1, "ci_high": -0.1, "q_value": 0.05},
            "temporal_transformer": {"mean_difference": -1, "ci_high": -0.1, "q_value": 0.05},
        }
        for stratum in L.CHALLENGE_STRATA[:2]
    }
    g1 = P.evaluate_g1_gate(g1_comparisons, True, True)
    assert g1["passed"] is True

    g2_rows = {
        stratum: {
            "completion_difference": 0.1,
            "completion_ci_low": 0.01,
            "completion_q": 0.05,
            "collision_difference": 0.0,
            "collision_q": 1.0,
        }
        for stratum in L.CHALLENGE_STRATA[:2]
    }
    g2 = P.evaluate_g2_gate(g2_rows, True)
    assert g2["passed"] is True

    reset = {
        stratum: {"quality_difference": 0.1, "q_value": 0.05}
        for stratum in L.CHALLENGE_STRATA[:2]
    }
    g3 = P.evaluate_g3_gate(reset, True, 0.01)
    assert g3["passed"] is True
    assert g3["mechanism"] == "persistent_state"
    assert P.evaluate_g4_closure(True, True, False)["verdict"] == "forecast_planner_mismatch"
    assert P.evaluate_g4_closure(True, False, False)["verdict"] == "strong_negative"
    assert P.evaluate_g4_closure(False, True, True)["verdict"] == "substrate_rejected"


def test_flat_comparator_freezes_from_validation_only(tmp_path):
    entries = [
        {"arm": "lstm", "seed": 0, "status": "complete", "best_validation": 0.3},
        {
            "arm": "temporal_transformer",
            "seed": 0,
            "status": "complete",
            "best_validation": 0.2,
        },
    ]
    frozen = P.freeze_flat_comparator(tmp_path, entries, "dataset-hash")
    assert frozen["selected_best_flat"] == "temporal_transformer"
    assert frozen["selected_flat_recurrent"] == "lstm"
    assert frozen["selection_data"] == "VALIDATION only"
    assert P.freeze_flat_comparator(tmp_path, entries, "dataset-hash") == frozen
    changed = [dict(row) for row in entries]
    changed[0]["best_validation"] = 0.1
    with pytest.raises(RuntimeError, match="would change"):
        P.freeze_flat_comparator(tmp_path, changed, "dataset-hash")


def test_registered_long_horizon_scores_drive_method_preselection(tmp_path):
    entries = [
        {"arm": arm, "seed": 0, "status": "complete", "best_validation": 0.01}
        for arm in ("lstm", "temporal_transformer", "onlstm", "hrm_stream")
    ]
    scores = {
        ("lstm", 0): 0.30,
        ("temporal_transformer", 0): 0.20,
        ("onlstm", 0): 0.25,
        ("hrm_stream", 0): 0.15,
    }
    frozen = P.freeze_flat_comparator(
        tmp_path, entries, "dataset-hash", validation_scores=scores
    )
    assert frozen["selected_best_flat"] == "temporal_transformer"
    assert frozen["selected_hierarchy"] == "hrm_stream"
    assert frozen["schema_version"] == "c12a-comparator-selection-v4"
    assert frozen["ranking"][0]["validation_normalized_g1_error_mean"] == pytest.approx(0.2)
    assert "horizons 17-32" in frozen["criterion"]


def test_latent_linear_probe_recovers_a_separable_regime():
    train_x = np.asarray([[-2.0, 0.0], [-1.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    train_y = np.asarray([0, 0, 1, 1])
    eval_x = np.asarray([[-3.0, 2.0], [3.0, -1.0]])
    eval_y = np.asarray([0, 1])
    result = P.linear_probe_accuracy(train_x, train_y, eval_x, eval_y)
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0


@pytest.mark.parametrize("carry_mode", P.CARRY_MODES)
def test_learned_provider_emits_planner_table_and_tracks_encoding(carry_mode):
    cfg = L.C12DynamicsConfig(
        raster_size=8,
        episode_steps=12,
        burn_in=4,
        forecast_horizon=4,
        alias_horizon=4,
    )
    episode, _ = L.make_fixture_pair("direction_alias", cfg=cfg)
    model_cfg = P.WM.WorldModelConfig(
        d_model=16,
        horizon=4,
        max_patrollers=1,
        max_gates=2,
        core_width=16,
        recurrent_layers=1,
        snapshot_depth=1,
        transformer_window=3,
        transformer_depth=1,
        transformer_heads=4,
        transformer_ff_width=32,
        onlstm_chunk_size=4,
        hrm_slow_cadence=2,
        decoder_width=16,
    )
    model = P.WM.build_world_model("lstm", model_cfg).eval()
    provider = P.LearnedForecastProvider(model, carry_mode=carry_mode)
    provider.reset(episode)
    for t in range(episode.alias_time - 2, episode.alias_time + 1):
        provider.observe(episode, t, episode.observe(t))
    forecast = provider.forecast(episode, episode.alias_time, 4)
    assert forecast.centers.shape == (5, 1, 2)
    assert forecast.gate_open.shape == (5, 2)
    assert provider.inference_calls > 0
    assert provider.encoded_frames > 0
    assert provider.forecast_ms >= 0.0
