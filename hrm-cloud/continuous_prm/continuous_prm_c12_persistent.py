"""C12-A orchestration and preregistered G0-A memory/headroom probe.

No learned model is trained here.  The real ``--mode probe --scale full``
command evaluates four non-learned forecast providers on 200 dedicated PROBE
episodes per stratum, writes raw/summary/report artifacts, and computes the
authorization gates from the approved C12 design.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import continuous_prm_common as C
import continuous_prm_c12_closed_loop as CL
import continuous_prm_c12_latent_dynamics as L
import continuous_prm_c12_world_model as WM


PROVIDER_NAMES: Tuple[str, ...] = (
    "frozen_frame",
    "constant_velocity",
    "true_mode",
    "oracle_future",
)
MAP_FAMILIES: Tuple[str, ...] = (
    "C_dyn_maze",
    "C_dyn_rooms",
    "C_dyn_spiral",
    "C_dyn_rooms_large",
)
SCALE_PAIRS_PER_STRATUM = {"smoke": 2, "pilot": 32, "full": 100}
PROBE_SCHEMA_VERSION = "c12a-g0-v11"
DATASET_SCHEMA_VERSION = "c12a-dataset-v6"
FORECAST_SCHEMA_VERSION = "c12a-forecast-v7"
PLANNING_SCHEMA_VERSION = "c12a-planning-v5"
DATASET_SCALE_COUNTS: Dict[str, Dict[str, int]] = {
    "smoke": {"TRAIN": 8, "VALIDATION": 4, "SMOKE": 4},
    "pilot": {"TRAIN": 256, "VALIDATION": 64, "PILOT": 64},
    "full": {"TRAIN": 3000, "VALIDATION": 300, "TEST": 300},
}
DATASET_EPISODES_PER_SHARD = {"smoke": 16, "pilot": 64, "full": 100}
SCALE_MODEL_SEEDS = {"smoke": (0,), "pilot": (0,), "full": (0, 1, 2)}
EVAL_SPLIT_BY_SCALE = {"smoke": "SMOKE", "pilot": "PILOT", "full": "TEST"}
CARRY_MODES: Tuple[str, ...] = ("persistent", "reset", "window_reencode")
_DATASET_SPLIT_ORDER = ("TRAIN", "VALIDATION", "SMOKE", "PILOT", "TEST")
HORIZON_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("h01_04", 0, 4),
    ("h05_16", 4, 16),
    ("h17_32", 16, 32),
)
FORECAST_RAW_COLS: Tuple[str, ...] = (
    "schema_version",
    "eval_config_hash",
    "dataset_config_hash",
    "checkpoint_hash",
    "episode_id",
    "pair_id",
    "static_map_id",
    "split",
    "stratum",
    "map_family",
    "eval_condition",
    "is_long_dwell_ood",
    "is_scale_ood",
    "is_heldout_combo",
    "variant",
    "arm",
    "seed",
    "carry_mode",
    "decision_step",
    "horizon_bucket",
    "horizon_start",
    "horizon_stop",
    "ade_sum",
    "ade_count",
    "ade",
    "fde_sum",
    "fde_count",
    "fde",
    "route_critical_ade_sum",
    "route_critical_ade_count",
    "route_critical_ade",
    "gate_true_positive",
    "gate_positive_count",
    "gate_true_negative",
    "gate_negative_count",
    "gate_balanced_accuracy",
    "gate_brier_sum",
    "gate_brier_count",
    "gate_brier",
    "route_critical_gate_brier_sum",
    "route_critical_gate_brier_count",
    "route_critical_gate_brier",
    "occupancy_hits",
    "occupancy_count",
    "occupancy_recall",
    "route_critical_occupancy_hits",
    "route_critical_occupancy_count",
    "route_critical_occupancy_recall",
)
PLANNING_RAW_COLS: Tuple[str, ...] = (
    "schema_version",
    "eval_config_hash",
    "dataset_config_hash",
    "checkpoint_hash",
    "arm",
    "seed",
    "carry_mode",
    "episode_id",
    "pair_id",
    "pair_index",
    "variant",
    "stratum",
    "map_family",
    "eval_condition",
    "is_long_dwell_ood",
    "is_scale_ood",
    "is_heldout_combo",
    "map_seed",
    "goal_seed",
    "regime_seed",
    "provider",
    "success",
    "failure_reason",
    "arrival_time",
    "elapsed_steps",
    "collision_adjusted_arrival",
    "collisions",
    "first_collision_time",
    "cumulative_expansions",
    "planning_ms",
    "forecast_ms",
    "encoded_frames",
    "inference_calls",
    "replans",
    "failed_plans",
    "observation_updates",
    "first_action",
    "oracle_first_action_hint",
    "route_short_length",
    "route_long_length",
    "route_ratio",
    "fast_period",
    "slow_dwell",
)


def _retry_atomic_write(callable_: Callable[[], None], attempts: int = 20) -> None:
    """Retry transient Windows sharing violations around atomic replaces."""
    last_error: Optional[PermissionError] = None
    for attempt in range(int(attempts)):
        try:
            callable_()
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(1.0, 0.05 * (attempt + 1)))
    assert last_error is not None
    raise last_error


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _retry_atomic_write(lambda: C.write_json(path, payload))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _retry_atomic_write(lambda: C.write_csv(path, rows))


def _write_text(path: Path, value: str) -> None:
    C.ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")

    def write_once() -> None:
        tmp.write_text(value, encoding="utf-8")
        os.replace(tmp, path)

    _retry_atomic_write(write_once)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_npz_deterministic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a compressed NPZ with stable member order and ZIP timestamps."""
    C.ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")

    def write_once() -> None:
        with zipfile.ZipFile(
            tmp, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name in sorted(arrays):
                array = np.asarray(arrays[name])
                if array.dtype == object:
                    raise ValueError(f"refusing to serialize object array {name!r}")
                buffer = io.BytesIO()
                np.save(buffer, array, allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue(), compresslevel=6)
        os.replace(tmp, path)

    _retry_atomic_write(write_once)


def _dataset_config_hash(
    cfg: L.C12DynamicsConfig,
    scale: str,
    counts: Mapping[str, int],
    episodes_per_shard: int,
    map_families: Sequence[str],
    episode_builder: Callable[..., Tuple[L.C12EpisodeSpec, L.C12EpisodeSpec]],
) -> str:
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dynamics": asdict(cfg),
        "scale": scale,
        "counts": {key: int(counts[key]) for key in sorted(counts)},
        "episodes_per_shard": int(episodes_per_shard),
        "map_families": [str(value) for value in map_families],
        "episode_builder": {
            "module": getattr(episode_builder, "__module__", "unknown"),
            "qualname": getattr(episode_builder, "__qualname__", repr(episode_builder)),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _dataset_counts(
    scale: str, counts_by_split: Optional[Mapping[str, int]]
) -> Dict[str, int]:
    if scale not in DATASET_SCALE_COUNTS:
        raise KeyError(f"unknown dataset scale: {scale!r}")
    source = DATASET_SCALE_COUNTS[scale] if counts_by_split is None else counts_by_split
    counts = {str(split).upper(): int(count) for split, count in source.items()}
    unknown = sorted(set(counts).difference(_DATASET_SPLIT_ORDER))
    if unknown:
        raise KeyError(f"unsupported C12 dataset splits: {', '.join(unknown)}")
    if not counts:
        raise ValueError("dataset must contain at least one split")
    for split, count in counts.items():
        if count <= 0 or count % 2:
            raise ValueError(
                f"{split} episode count must be a positive even number for paired regimes"
            )
    return {split: counts[split] for split in _DATASET_SPLIT_ORDER if split in counts}


def _families_for_split(split: str, map_families: Sequence[str]) -> Tuple[str, ...]:
    families = tuple(str(value) for value in map_families)
    if not families:
        raise ValueError("map_families must not be empty")
    if split in ("TRAIN", "VALIDATION"):
        in_distribution = tuple(value for value in families if value != "C_dyn_rooms_large")
        return in_distribution or families
    return families


def _validate_shard_arrays(
    arrays: Mapping[str, np.ndarray], expected_episodes: Optional[int] = None
) -> None:
    required = {
        "frame_rasters",
        "centers",
        "radii",
        "identity_mask",
        "visible_regime_context",
        "visible_regime_mask",
        "target_center_displacements",
        "target_gate_open",
        "gate_mask",
        "route_critical_mask",
        "route_edge_midpoints",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise RuntimeError(f"C12 dataset shard missing arrays: {', '.join(missing)}")
    episode_counts = {int(np.asarray(arrays[name]).shape[0]) for name in required}
    if len(episode_counts) != 1:
        raise RuntimeError("C12 dataset shard arrays disagree on episode count")
    episodes = next(iter(episode_counts))
    if expected_episodes is not None and episodes != int(expected_episodes):
        raise RuntimeError(
            f"C12 dataset shard episode mismatch ({episodes} != {expected_episodes})"
        )
    for name in required:
        array = np.asarray(arrays[name])
        if array.dtype == object:
            raise RuntimeError(f"C12 dataset shard contains object array {name!r}")
        if not np.isfinite(array.astype(np.float64, copy=False)).all():
            raise RuntimeError(f"C12 dataset shard contains non-finite array {name!r}")
    frames = np.asarray(arrays["frame_rasters"])
    centers = np.asarray(arrays["centers"])
    identity = np.asarray(arrays["identity_mask"])
    target_centers = np.asarray(arrays["target_center_displacements"])
    target_gates = np.asarray(arrays["target_gate_open"])
    gate_mask = np.asarray(arrays["gate_mask"])
    if frames.ndim != 5 or frames.shape[2] != 5:
        raise RuntimeError("frame_rasters must have shape [episode,time,5,height,width]")
    if centers.shape[:3] != identity.shape or target_centers.shape[:2] != centers.shape[:2]:
        raise RuntimeError("center/identity/target time shapes are inconsistent")
    if target_centers.shape[3] != centers.shape[2] or target_centers.shape[-1] != 2:
        raise RuntimeError("center target identity/coordinate shapes are inconsistent")
    if target_gates.shape[:2] != gate_mask.shape[:2]:
        raise RuntimeError("gate target/mask time shapes are inconsistent")
    if target_gates.shape[3] != gate_mask.shape[2]:
        raise RuntimeError("gate target/mask slot shapes are inconsistent")


def _load_npz_arrays(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    return arrays


def collect_dataset(
    out_dir: str | Path,
    scale: str = "full",
    cfg: Optional[L.C12DynamicsConfig] = None,
    counts_by_split: Optional[Mapping[str, int]] = None,
    episodes_per_shard: Optional[int] = None,
    map_families: Sequence[str] = MAP_FAMILIES,
    episode_builder: Callable[..., Tuple[L.C12EpisodeSpec, L.C12EpisodeSpec]] = L.build_challenge_pair,
) -> Dict[str, Any]:
    """Collect paired episodes into checksummed, resumable NPZ shards."""
    cfg = cfg or L.C12DynamicsConfig()
    counts = _dataset_counts(scale, counts_by_split)
    shard_size = (
        DATASET_EPISODES_PER_SHARD[scale]
        if episodes_per_shard is None
        else int(episodes_per_shard)
    )
    if shard_size <= 0 or shard_size % 2:
        raise ValueError("episodes_per_shard must be a positive even number")
    out_dir = Path(out_dir)
    manifest_path = out_dir / "dataset_manifest.json"
    dataset_dir = out_dir / "dataset"
    cfg_hash = _dataset_config_hash(
        cfg, scale, counts, shard_size, map_families, episode_builder
    )

    prior: Dict[str, Any] = C.read_json(manifest_path) if manifest_path.exists() else {}
    if prior:
        if prior.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise RuntimeError(
                "C12 dataset schema version mismatch "
                f"({prior.get('schema_version')} != {DATASET_SCHEMA_VERSION})"
            )
        if prior.get("config_hash") != cfg_hash:
            raise RuntimeError(
                "C12 dataset config hash mismatch "
                f"({prior.get('config_hash')} != {cfg_hash})"
            )
    elif dataset_dir.exists() and any(dataset_dir.rglob("*.npz")):
        raise RuntimeError("untracked C12 dataset shards exist without a manifest")

    prior_by_path = {
        str(entry.get("path")): entry for entry in prior.get("shards", [])
    }
    manifest: Dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "config_hash": cfg_hash,
        "dynamics_config_hash": L.config_hash(cfg),
        "scale": scale,
        "cfg": asdict(cfg),
        "splits": counts,
        "strata": list(L.STRATA),
        "map_families": list(map_families),
        "train_validation_map_families": list(
            _families_for_split("TRAIN", map_families)
        ),
        "episodes_per_shard": shard_size,
        "episodes_total": len(L.STRATA) * sum(counts.values()),
        "status": "running",
        "shards": [],
    }
    _write_json(manifest_path, manifest)

    completed_episodes = 0
    for split, count in counts.items():
        split_families = _families_for_split(split, map_families)
        for stratum in L.STRATA:
            for start in range(0, count, shard_size):
                stop = min(count, start + shard_size)
                shard_index = start // shard_size
                relative = Path("dataset") / split.lower() / stratum / f"shard_{shard_index:05d}.npz"
                diagnostics_relative = relative.with_suffix(".diagnostics.json")
                shard_path = out_dir / relative
                diagnostics_path = out_dir / diagnostics_relative
                expected = stop - start
                prior_entry = prior_by_path.get(relative.as_posix())
                valid_prior = False
                if prior_entry and shard_path.exists() and diagnostics_path.exists():
                    valid_prior = (
                        int(prior_entry.get("episodes", -1)) == expected
                        and _sha256_file(shard_path) == prior_entry.get("sha256")
                        and _sha256_file(diagnostics_path)
                        == prior_entry.get("diagnostics_sha256")
                    )
                    if valid_prior:
                        arrays = _load_npz_arrays(shard_path)
                        _validate_shard_arrays(arrays, expected)

                if valid_prior:
                    entry = dict(prior_entry)
                else:
                    records: List[L.C12EpisodeRecord] = []
                    diagnostic_episodes: List[Dict[str, Any]] = []
                    for pair_index in range(start // 2, stop // 2):
                        family = split_families[pair_index % len(split_families)]
                        left, right = episode_builder(
                            stratum=stratum,
                            pair_index=pair_index,
                            map_family=family,
                            cfg=cfg,
                            split=split,
                        )
                        audit = L.audit_alias_pair(left, right)
                        for episode in (left, right):
                            record = L.build_episode_record(episode)
                            records.append(record)
                            diagnostic_episodes.append(
                                {
                                    "metadata": record.metadata,
                                    "privileged_diagnostics": record.privileged_diagnostics,
                                    "alias_audit": audit,
                                }
                            )
                    arrays = {
                        name: np.stack(
                            [record.model_arrays()[name] for record in records], axis=0
                        )
                        for name in records[0].model_arrays()
                    }
                    _validate_shard_arrays(arrays, expected)
                    _write_npz_deterministic(shard_path, arrays)
                    diagnostics_payload = {
                        "schema_version": DATASET_SCHEMA_VERSION,
                        "config_hash": cfg_hash,
                        "split": split,
                        "stratum": stratum,
                        "shard_index": shard_index,
                        "episodes": diagnostic_episodes,
                    }
                    _write_json(diagnostics_path, diagnostics_payload)
                    entry = {
                        "path": relative.as_posix(),
                        "diagnostics_path": diagnostics_relative.as_posix(),
                        "split": split,
                        "stratum": stratum,
                        "shard_index": shard_index,
                        "episodes": expected,
                        "sha256": _sha256_file(shard_path),
                        "diagnostics_sha256": _sha256_file(diagnostics_path),
                        "bytes": shard_path.stat().st_size,
                        "diagnostics_bytes": diagnostics_path.stat().st_size,
                    }
                manifest["shards"].append(entry)
                completed_episodes += expected
                manifest["episodes_complete"] = completed_episodes
                _write_json(manifest_path, manifest)

    manifest["shards"].sort(
        key=lambda row: (
            _DATASET_SPLIT_ORDER.index(str(row["split"])),
            L.STRATA.index(str(row["stratum"])),
            int(row["shard_index"]),
        )
    )
    manifest["status"] = "complete"
    manifest["episodes_complete"] = manifest["episodes_total"]
    manifest["dataset_bytes"] = sum(
        int(row["bytes"]) + int(row["diagnostics_bytes"])
        for row in manifest["shards"]
    )
    _write_json(manifest_path, manifest)
    return manifest


def inspect_dataset(out_dir: str | Path) -> Dict[str, Any]:
    """Verify every shard and summarize collection integrity without training."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"no C12 dataset manifest at {manifest_path}")
    manifest = C.read_json(manifest_path)
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise RuntimeError(
            "C12 dataset schema version mismatch "
            f"({manifest.get('schema_version')} != {DATASET_SCHEMA_VERSION})"
        )

    split_counts: Dict[str, int] = {}
    stratum_counts: Dict[str, int] = {}
    condition_counts: Dict[str, int] = {}
    fast_periods: List[int] = []
    slow_dwells: List[int] = []
    alias_by_pair: Dict[str, bool] = {}
    regime_seeds: set[int] = set()
    episode_ids: set[str] = set()
    missing_identity_slots = 0
    missing_gate_slots = 0
    visible_context_steps = 0
    disk_bytes = manifest_path.stat().st_size
    episodes_total = 0

    for entry in manifest.get("shards", []):
        path = out_dir / str(entry["path"])
        diagnostics_path = out_dir / str(entry["diagnostics_path"])
        if not path.exists() or not diagnostics_path.exists():
            raise RuntimeError(f"C12 dataset shard artifact missing: {path}")
        if _sha256_file(path) != entry.get("sha256"):
            raise RuntimeError(f"C12 dataset shard checksum mismatch: {path}")
        if _sha256_file(diagnostics_path) != entry.get("diagnostics_sha256"):
            raise RuntimeError(f"C12 dataset diagnostics checksum mismatch: {diagnostics_path}")
        arrays = _load_npz_arrays(path)
        _validate_shard_arrays(arrays, int(entry["episodes"]))
        diagnostics = C.read_json(diagnostics_path)
        if diagnostics.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise RuntimeError(f"mixed schema version in {diagnostics_path}")
        if diagnostics.get("config_hash") != manifest.get("config_hash"):
            raise RuntimeError(f"mixed config hash in {diagnostics_path}")
        diagnostic_rows = diagnostics.get("episodes", [])
        if len(diagnostic_rows) != int(entry["episodes"]):
            raise RuntimeError(f"diagnostic episode count mismatch in {diagnostics_path}")

        n = int(entry["episodes"])
        episodes_total += n
        split = str(entry["split"])
        stratum = str(entry["stratum"])
        split_counts[split] = split_counts.get(split, 0) + n
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + n
        missing_identity_slots += int(np.count_nonzero(arrays["identity_mask"] == 0))
        missing_gate_slots += int(np.count_nonzero(arrays["gate_mask"] == 0))
        visible_context_steps += int(np.count_nonzero(arrays["visible_regime_mask"]))

        for row in diagnostic_rows:
            metadata = row["metadata"]
            privileged = row["privileged_diagnostics"]
            episode_id = str(metadata["episode_id"])
            if episode_id in episode_ids:
                raise RuntimeError(f"duplicate C12 episode id: {episode_id}")
            episode_ids.add(episode_id)
            condition = str(metadata.get("eval_condition", "development_id"))
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
            regime_seed = int(metadata["regime_seed"])
            if regime_seed in regime_seeds:
                raise RuntimeError(f"duplicate C12 regime seed: {regime_seed}")
            regime_seeds.add(regime_seed)
            fast_periods.append(int(privileged["fast_period"]))
            slow_dwells.append(int(privileged["slow_dwell"]))
            audit = row.get("alias_audit", {})
            alias_by_pair[str(metadata["pair_id"])] = bool(audit.get("is_alias", False))
        disk_bytes += path.stat().st_size + diagnostics_path.stat().st_size

    if episodes_total == 0:
        raise RuntimeError("C12 dataset manifest contains no episodes")
    expected_total = int(manifest.get("episodes_total", -1))
    if manifest.get("status") == "complete" and episodes_total != expected_total:
        raise RuntimeError(
            f"C12 dataset total mismatch ({episodes_total} != {expected_total})"
        )
    alias_rate = (
        float(sum(alias_by_pair.values()) / len(alias_by_pair)) if alias_by_pair else 0.0
    )
    bytes_per_episode = float(disk_bytes / episodes_total)
    full_episodes = len(L.STRATA) * sum(DATASET_SCALE_COUNTS["full"].values())
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "config_hash": manifest.get("config_hash"),
        "status": manifest.get("status"),
        "episodes_total": episodes_total,
        "split_counts": split_counts,
        "stratum_counts": stratum_counts,
        "condition_counts": condition_counts,
        "fast_period_min": min(fast_periods),
        "fast_period_max": max(fast_periods),
        "slow_dwell_min": min(slow_dwells),
        "slow_dwell_max": max(slow_dwells),
        "alias_rate": alias_rate,
        "missing_identity_slots": missing_identity_slots,
        "missing_gate_slots": missing_gate_slots,
        "visible_context_steps": visible_context_steps,
        "disk_bytes": disk_bytes,
        "bytes_per_episode": bytes_per_episode,
        "projected_full_disk_bytes": int(math.ceil(bytes_per_episode * full_episodes)),
        "checksums_verified": len(manifest.get("shards", [])),
    }


def _sigmoid_numpy(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    positive = value >= 0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _render_normalized_occupancy(
    centers: np.ndarray,
    radii: np.ndarray,
    identity_mask: np.ndarray,
    gate_open: np.ndarray,
    gate_mask: np.ndarray,
    route_edge_midpoints: np.ndarray,
    raster_size: int,
) -> np.ndarray:
    """Render circle and closed-gate forecasts in normalized map coordinates."""
    centers = np.asarray(centers, dtype=np.float64)
    horizon = int(centers.shape[0])
    size = int(raster_size)
    axis = (np.arange(size, dtype=np.float64) + 0.5) / float(size)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    output = np.zeros((horizon, size, size), dtype=np.bool_)
    for h in range(horizon):
        for identity in range(centers.shape[1]):
            if not bool(identity_mask[identity]):
                continue
            dx = xx - float(centers[h, identity, 0])
            dy = yy - float(centers[h, identity, 1])
            output[h] |= dx * dx + dy * dy <= float(radii[identity]) ** 2
        for gate in range(gate_open.shape[1]):
            if not bool(gate_mask[gate]) or bool(gate_open[h, gate]):
                continue
            midpoint = route_edge_midpoints[gate]
            x = int(np.clip(math.floor(float(midpoint[0]) * size), 0, size - 1))
            y = int(np.clip(math.floor(float(midpoint[1]) * size), 0, size - 1))
            output[h, y, x] = True
    return output


def forecast_bucket_metrics(
    predicted_displacements: np.ndarray,
    gate_logits: np.ndarray,
    target_displacements: np.ndarray,
    target_gate_open: np.ndarray,
    current_centers: np.ndarray,
    radii: np.ndarray,
    identity_mask: np.ndarray,
    gate_mask: np.ndarray,
    route_critical_mask: np.ndarray,
    route_edge_midpoints: np.ndarray,
    raster_size: int,
) -> List[Dict[str, Any]]:
    """Return additive metric components for one episode decision step."""
    predicted = np.asarray(predicted_displacements, dtype=np.float64)
    target = np.asarray(target_displacements, dtype=np.float64)
    gate_logits = np.asarray(gate_logits, dtype=np.float64)
    target_gates = np.asarray(target_gate_open, dtype=np.bool_)
    identity = np.asarray(identity_mask, dtype=np.bool_)
    gates = np.asarray(gate_mask, dtype=np.bool_)
    critical = np.asarray(route_critical_mask, dtype=np.bool_)
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 2:
        raise ValueError("center forecasts must share shape [horizon,identity,2]")
    if gate_logits.shape != target_gates.shape or gate_logits.ndim != 2:
        raise ValueError("gate forecasts must share shape [horizon,gate]")
    horizon = int(predicted.shape[0])
    if critical.shape != (horizon,):
        raise ValueError("route_critical_mask must have shape [horizon]")
    probabilities = _sigmoid_numpy(gate_logits)
    distance = np.linalg.norm(predicted - target, axis=-1)
    current = np.asarray(current_centers, dtype=np.float64)
    absolute_prediction = current[None, :, :] + predicted
    absolute_target = current[None, :, :] + target
    predicted_occupancy = _render_normalized_occupancy(
        absolute_prediction,
        radii,
        identity,
        probabilities >= 0.5,
        gates,
        route_edge_midpoints,
        raster_size,
    )
    target_occupancy = _render_normalized_occupancy(
        absolute_target,
        radii,
        identity,
        target_gates,
        gates,
        route_edge_midpoints,
        raster_size,
    )
    axis = (np.arange(int(raster_size), dtype=np.float64) + 0.5) / float(raster_size)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    route_spatial = np.zeros((int(raster_size), int(raster_size)), dtype=np.bool_)
    corridor_radius = max(
        float(np.max(np.asarray(radii)[identity])) if np.any(identity) else 0.0,
        1.5 / float(raster_size),
    )
    for gate, midpoint in enumerate(np.asarray(route_edge_midpoints, dtype=np.float64)):
        if gate >= gates.shape[0] or not bool(gates[gate]):
            continue
        route_spatial |= (
            (xx - float(midpoint[0])) ** 2 + (yy - float(midpoint[1])) ** 2
            <= corridor_radius**2
        )

    rows: List[Dict[str, Any]] = []
    for bucket, raw_start, raw_stop in HORIZON_BUCKETS:
        start = min(raw_start, horizon)
        stop = min(raw_stop, horizon)
        if start >= stop:
            continue
        identity_grid = np.broadcast_to(identity[None, :], (stop - start, identity.size))
        bucket_distance = distance[start:stop]
        valid_distance = bucket_distance[identity_grid]
        final_distance = distance[stop - 1, identity]
        critical_grid = np.broadcast_to(
            critical[start:stop, None], (stop - start, identity.size)
        ) & identity_grid
        critical_distance = bucket_distance[critical_grid]

        bucket_probabilities = probabilities[start:stop, gates]
        bucket_targets = target_gates[start:stop, gates]
        predicted_open = bucket_probabilities >= 0.5
        positives = bucket_targets
        negatives = ~bucket_targets
        true_positive = int(np.count_nonzero(predicted_open & positives))
        true_negative = int(np.count_nonzero((~predicted_open) & negatives))
        positive_count = int(np.count_nonzero(positives))
        negative_count = int(np.count_nonzero(negatives))
        brier = (bucket_probabilities - bucket_targets.astype(np.float64)) ** 2
        critical_gate_grid = np.broadcast_to(
            critical[start:stop, None], brier.shape
        )
        critical_brier = brier[critical_gate_grid]

        target_occ = target_occupancy[start:stop]
        predicted_occ = predicted_occupancy[start:stop]
        occupancy_count = int(np.count_nonzero(target_occ))
        occupancy_hits = int(np.count_nonzero(target_occ & predicted_occ))
        critical_time = critical[start:stop, None, None]
        critical_target = target_occ & critical_time & route_spatial[None, :, :]
        route_count = int(np.count_nonzero(critical_target))
        route_hits = int(np.count_nonzero(critical_target & predicted_occ))
        balanced = None
        if positive_count and negative_count:
            balanced = 0.5 * (
                true_positive / positive_count + true_negative / negative_count
            )
        rows.append(
            {
                "horizon_bucket": bucket,
                "horizon_start": start + 1,
                "horizon_stop": stop,
                "ade_sum": float(valid_distance.sum()),
                "ade_count": int(valid_distance.size),
                "ade": float(valid_distance.mean()) if valid_distance.size else None,
                "fde_sum": float(final_distance.sum()),
                "fde_count": int(final_distance.size),
                "fde": float(final_distance.mean()) if final_distance.size else None,
                "route_critical_ade_sum": float(critical_distance.sum()),
                "route_critical_ade_count": int(critical_distance.size),
                "route_critical_ade": (
                    float(critical_distance.mean()) if critical_distance.size else None
                ),
                "gate_true_positive": true_positive,
                "gate_positive_count": positive_count,
                "gate_true_negative": true_negative,
                "gate_negative_count": negative_count,
                "gate_balanced_accuracy": balanced,
                "gate_brier_sum": float(brier.sum()),
                "gate_brier_count": int(brier.size),
                "gate_brier": float(brier.mean()) if brier.size else None,
                "route_critical_gate_brier_sum": float(critical_brier.sum()),
                "route_critical_gate_brier_count": int(critical_brier.size),
                "route_critical_gate_brier": (
                    float(critical_brier.mean()) if critical_brier.size else None
                ),
                "occupancy_hits": occupancy_hits,
                "occupancy_count": occupancy_count,
                "occupancy_recall": (
                    occupancy_hits / occupancy_count if occupancy_count else None
                ),
                "route_critical_occupancy_hits": route_hits,
                "route_critical_occupancy_count": route_count,
                "route_critical_occupancy_recall": (
                    route_hits / route_count if route_count else None
                ),
            }
        )
    return rows


def _g1_metric_name(stratum: str) -> str:
    if str(stratum) == "slow_gate_phase":
        return "route_critical_gate_brier_h17_32"
    return "route_critical_center_ade_h17_32"


def _g1_mechanism_components(
    stratum: str,
    predicted_displacements: np.ndarray,
    gate_logits: np.ndarray,
    target_displacements: np.ndarray,
    target_gate_open: np.ndarray,
    identity_mask: np.ndarray,
    gate_mask: np.ndarray,
    route_critical_mask: np.ndarray,
) -> Tuple[float, int, str]:
    """Return additive registered G1 error components for horizons 17--32."""
    predicted = np.asarray(predicted_displacements, dtype=np.float64)
    target = np.asarray(target_displacements, dtype=np.float64)
    logits = np.asarray(gate_logits, dtype=np.float64)
    target_gates = np.asarray(target_gate_open, dtype=np.float64)
    identities = np.asarray(identity_mask, dtype=np.bool_)
    gates = np.asarray(gate_mask, dtype=np.bool_)
    critical = np.asarray(route_critical_mask, dtype=np.bool_)
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("G1 center forecasts must share shape [horizon,identity,2]")
    if logits.shape != target_gates.shape or logits.ndim != 2:
        raise ValueError("G1 gate forecasts must share shape [horizon,gate]")
    horizon = int(predicted.shape[0])
    if logits.shape[0] != horizon or critical.shape != (horizon,):
        raise ValueError("G1 forecast horizons and route-critical mask must align")
    start = min(16, horizon)
    stop = min(32, horizon)
    metric_name = _g1_metric_name(stratum)
    if start >= stop:
        return 0.0, 0, metric_name
    critical_slice = critical[start:stop]
    if str(stratum) == "slow_gate_phase":
        probabilities = _sigmoid_numpy(logits[start:stop])
        brier = (probabilities - target_gates[start:stop]) ** 2
        valid = np.broadcast_to(gates[None, :], brier.shape) & critical_slice[:, None]
        selected = brier[valid]
    else:
        distance = np.linalg.norm(predicted[start:stop] - target[start:stop], axis=-1)
        valid = (
            np.broadcast_to(identities[None, :], distance.shape)
            & critical_slice[:, None]
        )
        selected = distance[valid]
    return float(selected.sum()), int(selected.size), metric_name


def seeded_world_bootstrap(
    values: Sequence[float],
    seed: int = 12012,
    samples: int = 5000,
    confidence: float = 0.95,
) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(array, size=(int(samples), array.size), replace=True).mean(axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(draws, tail)),
        "ci_high": float(np.quantile(draws, 1.0 - tail)),
    }


def paired_sign_flip_p(
    values: Sequence[float], seed: int = 12012, samples: int = 20000
) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array != 0.0)]
    if array.size == 0:
        return 1.0
    observed = abs(float(array.mean()))
    if array.size <= 16:
        count = 1 << int(array.size)
        indices = np.arange(count, dtype=np.uint64)[:, None]
        bits = (indices >> np.arange(array.size, dtype=np.uint64)[None, :]) & 1
        signs = bits.astype(np.float64) * 2.0 - 1.0
        statistics = np.abs((signs * array[None, :]).mean(axis=1))
        return float(np.count_nonzero(statistics >= observed - 1e-15) / count)
    rng = np.random.default_rng(int(seed))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(int(samples), array.size))
    statistics = np.abs((signs * array[None, :]).mean(axis=1))
    return float((1 + np.count_nonzero(statistics >= observed - 1e-15)) / (samples + 1))


def bh_q_values(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("BH correction requires a one-dimensional p-value list")
    if values.size == 0:
        return []
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return [float(value) for value in result]


def evaluate_g1_gate(
    comparisons: Mapping[str, Mapping[str, Mapping[str, float]]],
    long_advantage_not_smaller: bool,
    control_not_worse: bool,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    positive_strata = []
    for stratum in L.CHALLENGE_STRATA:
        row = comparisons.get(stratum, {})
        wins_both = True
        for comparator in ("flat_recurrent", "temporal_transformer"):
            result = row.get(comparator, {})
            wins_both &= bool(
                float(result.get("mean_difference", float("inf"))) < 0.0
                and float(result.get("ci_high", float("inf"))) < 0.0
                and float(result.get("q_value", 1.0)) <= alpha
            )
        if wins_both:
            positive_strata.append(stratum)
    conditions = {
        "beats_both_in_two_strata": len(positive_strata) >= 2,
        "long_advantage_not_smaller": bool(long_advantage_not_smaller),
        "control_not_worse": bool(control_not_worse),
    }
    return {
        "passed": bool(all(conditions.values())),
        "positive_strata": positive_strata,
        "conditions": conditions,
    }


def evaluate_g2_gate(
    comparisons: Mapping[str, Mapping[str, Any]],
    ood_advantage_exceeds_control: bool,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    positive_strata = []
    collision_regressions = []
    for stratum in L.CHALLENGE_STRATA:
        row = comparisons.get(stratum, {})
        completion_positive = bool(
            float(row.get("completion_difference", float("-inf"))) > 0.0
            and float(row.get("completion_ci_low", float("-inf"))) > 0.0
            and float(row.get("completion_q", 1.0)) <= alpha
        )
        efficiency_positive = bool(
            row.get("completion_collision_tied", False)
            and (
                (
                    float(row.get("work_reduction", float("-inf"))) >= 0.15
                    and float(row.get("work_reduction_ci_low", float("-inf"))) > 0.0
                )
                or (
                    float(row.get("expansion_reduction", float("-inf"))) >= 0.15
                    and float(row.get("expansion_reduction_ci_low", float("-inf"))) > 0.0
                )
                or (
                    float(row.get("arrival_reduction", float("-inf"))) >= 0.15
                    and float(row.get("arrival_reduction_ci_low", float("-inf"))) > 0.0
                )
            )
        )
        if completion_positive or efficiency_positive:
            positive_strata.append(stratum)
        collision_regression = bool(
            float(row.get("collision_difference", 0.0)) > 0.0
            and float(row.get("collision_q", 1.0)) <= alpha
        )
        if collision_regression:
            collision_regressions.append(stratum)
    conditions = {
        "positive_in_two_strata": len(positive_strata) >= 2,
        "no_collision_regression": not collision_regressions,
        "ood_advantage_exceeds_control": bool(ood_advantage_exceeds_control),
    }
    return {
        "passed": bool(all(conditions.values())),
        "positive_strata": positive_strata,
        "collision_regressions": collision_regressions,
        "conditions": conditions,
    }


def evaluate_g3_gate(
    reset_comparisons: Mapping[str, Mapping[str, float]],
    window_quality_not_worse: bool,
    window_compute_reduction: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    positive_strata = [
        stratum
        for stratum in L.CHALLENGE_STRATA
        if float(reset_comparisons.get(stratum, {}).get("quality_difference", 0.0)) > 0.0
        and float(reset_comparisons.get(stratum, {}).get("q_value", 1.0)) <= alpha
    ]
    conditions = {
        "persistent_beats_reset_in_two_strata": len(positive_strata) >= 2,
        "matches_or_beats_window": bool(window_quality_not_worse),
        "uses_less_encoding_compute": float(window_compute_reduction) > 0.0,
    }
    passed = bool(all(conditions.values()))
    return {
        "passed": passed,
        "positive_strata": positive_strata,
        "mechanism": "persistent_state" if passed else "temporal_architecture",
        "conditions": conditions,
    }


def evaluate_g4_closure(g0_passed: bool, g1_passed: bool, g2_passed: bool) -> Dict[str, str]:
    if not g0_passed:
        return {"verdict": "substrate_rejected", "interpretation": "No architecture verdict."}
    if g2_passed:
        return {
            "verdict": "hierarchy_helps_planning",
            "interpretation": "Hierarchy helps under explicitly history-dependent dynamics.",
        }
    if g1_passed:
        return {
            "verdict": "forecast_planner_mismatch",
            "interpretation": "Hierarchy predicts better, but planning is insensitive.",
        }
    return {
        "verdict": "strong_negative",
        "interpretation": "Matched temporal hierarchy adds no value even when history is necessary.",
    }


def _gate_edges(future: Mapping[str, np.ndarray]) -> Tuple[L.EdgeId, ...]:
    array = np.asarray(future["gate_edges"], dtype=np.int64)
    if array.size == 0:
        return ()
    return tuple(tuple(int(x) for x in row) for row in array.reshape(-1, 2))


class _HistoryProvider:
    name = "history"

    def __init__(self) -> None:
        self.history: List[Tuple[int, np.ndarray, np.ndarray]] = []

    def reset(self, episode: L.C12EpisodeSpec) -> None:
        self.history.clear()

    def observe(
        self, episode: L.C12EpisodeSpec, t: int, observation: L.C12Observation
    ) -> None:
        # This is a current-state read only.  Privileged fields remain in the
        # episode diagnostic state and never enter ``observation.model_payload``.
        current = episode.future(t, 0)
        self.history.append(
            (
                int(t),
                np.asarray(current["centers"][0], dtype=np.float64).copy(),
                np.asarray(current["gate_open"][0], dtype=np.bool_).copy(),
            )
        )
        if len(self.history) > episode.dynamics.cfg.burn_in + 2:
            self.history.pop(0)

    def _current(self, episode: L.C12EpisodeSpec, t: int) -> Tuple[np.ndarray, np.ndarray, Tuple[L.EdgeId, ...]]:
        if self.history and self.history[-1][0] == int(t):
            centers = self.history[-1][1]
            gates = self.history[-1][2]
            edges = _gate_edges(episode.future(t, 0))
            return centers, gates, edges
        current = episode.future(t, 0)
        return current["centers"][0], current["gate_open"][0], _gate_edges(current)


class FrozenFrameProvider(_HistoryProvider):
    name = "frozen_frame"

    def forecast(self, episode: L.C12EpisodeSpec, t: int, horizon: int) -> CL.TabulatedDynamics:
        centers, gates, edges = self._current(episode, t)
        return CL.TabulatedDynamics(
            centers=np.repeat(centers[None, :, :], int(horizon) + 1, axis=0),
            radii=episode.dynamics.radii,
            gate_open=np.repeat(gates[None, :], int(horizon) + 1, axis=0),
            gate_edges=edges,
            dt=episode.dt,
        )


class ConstantVelocityProvider(_HistoryProvider):
    name = "constant_velocity"

    def forecast(self, episode: L.C12EpisodeSpec, t: int, horizon: int) -> CL.TabulatedDynamics:
        centers, gates, edges = self._current(episode, t)
        velocity = np.zeros_like(centers)
        if len(self.history) >= 2:
            t0, c0, _g0 = self.history[-2]
            t1, c1, _g1 = self.history[-1]
            velocity = (c1 - c0) / max(1, t1 - t0)
        steps = np.arange(int(horizon) + 1, dtype=np.float64)[:, None, None]
        predicted = centers[None, :, :] + steps * velocity[None, :, :]
        predicted = np.clip(predicted, 0.0, episode.world.side_len)
        return CL.TabulatedDynamics(
            centers=predicted,
            radii=episode.dynamics.radii,
            gate_open=np.repeat(gates[None, :], int(horizon) + 1, axis=0),
            gate_edges=edges,
            dt=episode.dt,
        )


class TrueModeProvider(_HistoryProvider):
    """Privileged mode diagnostic with deliberately conservative timing.

    It knows which branch is hazardous but not the exact clearance time.  On
    challenge suites it therefore applies a fixed eight-step clearance margin
    and blocks the hazardous branch for the whole planning horizon.  This is
    sufficient to test whether mode/history can choose a safe route while
    preserving a meaningful exact-future oracle ceiling.  In the
    present-sufficient control the same regime fields are visible now, so this
    provider uses the exact deterministic transition rule without history.
    """

    name = "true_mode"

    def forecast(self, episode: L.C12EpisodeSpec, t: int, horizon: int) -> CL.TabulatedDynamics:
        if episode.stratum == L.CONTROL_STRATUM:
            return CL.exact_future_forecast(episode, t, horizon)
        current = episode.future(t, 0)
        centers = np.repeat(current["centers"], int(horizon) + 1, axis=0)
        gates = np.repeat(current["gate_open"], int(horizon) + 1, axis=0)
        edges = _gate_edges(current)
        hazard = L.canonical_edge(episode.schedule.hazard_edge)
        # Mode identifies the dangerous branch, but this diagnostic has no
        # exact phase clock.  Its pre-registered conservative policy waits
        # until a fixed absolute clearance time.  Expressing the margin in
        # absolute episode time prevents receding-horizon replans from
        # restarting the delay forever.
        clearance_time = episode.alias_time + 8
        remaining_clearance = max(0, clearance_time - int(t))
        if remaining_clearance > 0:
            gates[1 : min(int(horizon), remaining_clearance) + 1, :] = False
        if episode.stratum == "slow_gate_phase":
            for j, edge in enumerate(edges):
                if L.canonical_edge(edge) == hazard:
                    gates[1:, j] = False
        else:
            midpoint = 0.5 * (
                episode.roadmap.points[hazard[0]] + episode.roadmap.points[hazard[1]]
            )
            centers[1:, 0, :] = midpoint
        return CL.TabulatedDynamics(
            centers=centers,
            radii=episode.dynamics.radii,
            gate_open=gates,
            gate_edges=edges,
            dt=episode.dt,
        )


class OracleFutureProvider(_HistoryProvider):
    name = "oracle_future"

    def forecast(self, episode: L.C12EpisodeSpec, t: int, horizon: int) -> CL.TabulatedDynamics:
        return CL.exact_future_forecast(episode, t, horizon)


class LearnedForecastProvider:
    """Stateful adapter from a learned C12 checkpoint to ``TabulatedDynamics``."""

    def __init__(
        self,
        model: WM.C12WorldModel,
        device: str = "cpu",
        carry_mode: str = "persistent",
    ) -> None:
        if carry_mode not in CARRY_MODES:
            raise KeyError(f"unknown learned carry mode: {carry_mode!r}")
        self.model = model
        self.device = device
        self.carry_mode = carry_mode
        self.name = f"{model.arm}_{carry_mode}"
        self.carry: Any = None
        self.history: List[Dict[str, Any]] = []
        self.latest: Optional[Tuple[int, L.C12Observation, Dict[str, Any]]] = None
        self.prediction: Optional[Dict[str, Any]] = None
        self.forecast_ms = 0.0
        self.encoded_frames = 0
        self.inference_calls = 0

    def reset(self, episode: L.C12EpisodeSpec) -> None:
        import torch

        self.model.eval()
        self.carry = self.model.initial_carry(1, torch.device(self.device))
        self.history = []
        self.latest = None
        self.prediction = None
        self.forecast_ms = 0.0
        self.encoded_frames = 0
        self.inference_calls = 0

    def _timed_inference(self, callable_: Callable[[], Any], encoded_frames: int) -> Any:
        import torch

        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(self.device))
        start = time.perf_counter()
        result = callable_()
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(self.device))
        self.forecast_ms += (time.perf_counter() - start) * 1000.0
        self.encoded_frames += int(encoded_frames)
        self.inference_calls += 1
        return result

    def _observation_batch(
        self, episode: L.C12EpisodeSpec, observation: L.C12Observation
    ) -> Dict[str, Any]:
        import torch

        cfg = self.model.cfg
        centers = np.zeros((1, cfg.max_patrollers, 2), dtype=np.float32)
        radii = np.zeros((1, cfg.max_patrollers), dtype=np.float32)
        identity = np.zeros((1, cfg.max_patrollers), dtype=np.float32)
        count = int(observation.centers.shape[0])
        if count > cfg.max_patrollers:
            raise RuntimeError("learned checkpoint has too few patroller slots")
        centers[0, :count] = observation.centers
        radii[0, :count] = observation.radii
        identity[0, :count] = observation.identity_mask
        visible = np.zeros((1, 3), dtype=np.float32)
        visible_mask = np.zeros((1, 1), dtype=np.float32)
        if observation.visible_regime_context is not None:
            visible[0] = observation.visible_regime_context
            visible_mask[0, 0] = 1.0
        gate_mask = np.zeros((1, cfg.max_gates), dtype=np.float32)
        gate_count = len(episode.dynamics.gates.edge_ids)
        if gate_count > cfg.max_gates:
            raise RuntimeError("learned checkpoint has too few gate slots")
        gate_mask[0, :gate_count] = 1.0
        frame = np.stack(
            (
                observation.static_occupancy,
                observation.dynamic_occupancy,
                observation.gate_open_raster,
                observation.agent_goal_raster[0],
                observation.agent_goal_raster[1],
            ),
            axis=0,
        )[None]
        arrays = {
            "frame_rasters": frame,
            "centers": centers,
            "radii": radii,
            "identity_mask": identity,
            "visible_regime_context": visible,
            "visible_regime_mask": visible_mask,
            "gate_mask": gate_mask,
        }
        return {
            name: torch.from_numpy(value).float().to(self.device)
            for name, value in arrays.items()
        }

    @staticmethod
    def _visible_gate_vector(
        episode: L.C12EpisodeSpec, observation: L.C12Observation
    ) -> np.ndarray:
        raster = np.asarray(observation.gate_open_raster)
        size = raster.shape[-1]
        values = []
        for edge in episode.dynamics.gates.edge_ids:
            midpoint = 0.5 * (
                episode.roadmap.points[int(edge[0])]
                + episode.roadmap.points[int(edge[1])]
            )
            x = int(
                np.clip(
                    math.floor(float(midpoint[0]) / episode.world.side_len * size),
                    0,
                    size - 1,
                )
            )
            y = int(
                np.clip(
                    math.floor(float(midpoint[1]) / episode.world.side_len * size),
                    0,
                    size - 1,
                )
            )
            values.append(bool(raster[y, x] >= 0.5))
        return np.asarray(values, dtype=np.bool_)

    def observe(
        self,
        episode: L.C12EpisodeSpec,
        t: int,
        observation: L.C12Observation,
    ) -> None:
        import torch

        batch = self._observation_batch(episode, observation)
        self.latest = (int(t), observation, batch)
        self.history.append(batch)
        self.history = self.history[-self.model.cfg.transformer_window :]
        if self.carry_mode == "persistent":
            with torch.no_grad():
                self.prediction, self.carry = self._timed_inference(
                    lambda: self.model.step(batch, self.carry), encoded_frames=1
                )

    def forecast(
        self, episode: L.C12EpisodeSpec, t: int, horizon: int
    ) -> CL.TabulatedDynamics:
        import torch

        if self.latest is None or self.latest[0] != int(t):
            raise RuntimeError("learned provider forecast requires the matching visible frame")
        _latest_t, observation, batch = self.latest
        with torch.no_grad():
            if self.carry_mode == "reset":
                fresh = self.model.initial_carry(1, torch.device(self.device))
                prediction, _ = self._timed_inference(
                    lambda: self.model.step(batch, fresh), encoded_frames=1
                )
            elif self.carry_mode == "window_reencode":
                prediction = self._timed_inference(
                    lambda: self.model.window_reencode(self.history),
                    encoded_frames=len(self.history),
                )
            else:
                if self.prediction is None:
                    raise RuntimeError("persistent learned provider has no prediction")
                prediction = self.prediction
        if int(horizon) > self.model.cfg.horizon:
            raise RuntimeError("planner requested a horizon longer than the learned decoder")
        patrollers = int(observation.centers.shape[0])
        gates = len(episode.dynamics.gates.edge_ids)
        displacements = (
            prediction["center_displacements"][0, :horizon, :patrollers]
            .detach()
            .cpu()
            .numpy()
        )
        predicted_gates = (
            prediction["gate_logits"][0, :horizon, :gates].detach().cpu().numpy()
            >= 0.0
        )
        return CL.normalized_prediction_to_tabulated(
            current_centers=observation.centers,
            normalized_radii=observation.radii,
            predicted_displacements=displacements,
            predicted_gate_open=predicted_gates,
            current_gate_open=self._visible_gate_vector(episode, observation),
            identity_mask=observation.identity_mask,
            gate_mask=np.ones(gates, dtype=np.uint8),
            gate_edges=episode.dynamics.gates.edge_ids,
            side_len=episode.world.side_len,
            dt=episode.dt,
        )


PROVIDER_FACTORIES: Dict[str, Callable[[], _HistoryProvider]] = {
    "frozen_frame": FrozenFrameProvider,
    "constant_velocity": ConstantVelocityProvider,
    "true_mode": TrueModeProvider,
    "oracle_future": OracleFutureProvider,
}


def probe_config_hash(
    cfg: L.C12DynamicsConfig,
    scale: str,
    pairs_per_stratum: int,
    map_families: Sequence[str] = MAP_FAMILIES,
) -> str:
    payload = {
        "schema": PROBE_SCHEMA_VERSION,
        "cfg": asdict(cfg),
        "scale": str(scale),
        "pairs_per_stratum": int(pairs_per_stratum),
        "map_families": list(map_families),
        "providers": list(PROVIDER_NAMES),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _action_signature(action: Optional[CL.PlanAction]) -> str:
    if action is None:
        return "none"
    return f"{action.kind}:{action.source}->{action.target}:{action.duration}"


def _result_row(
    episode: L.C12EpisodeSpec,
    provider_name: str,
    result: CL.ClosedLoopResult,
) -> Dict[str, Any]:
    success_penalty = episode.dynamics.cfg.episode_steps - episode.alias_time + 32
    score = result.elapsed_steps if result.success else success_penalty
    return {
        "episode_id": f"{episode.pair_id}-v{episode.regime.variant}",
        "pair_id": episode.pair_id,
        "pair_index": episode.pair_index,
        "variant": episode.regime.variant,
        "stratum": episode.stratum,
        "map_family": episode.map_family,
        "eval_condition": episode.diagnostics.get("eval_condition", "development_id"),
        "is_long_dwell_ood": int(bool(episode.diagnostics.get("is_long_dwell_ood", False))),
        "is_scale_ood": int(bool(episode.diagnostics.get("is_scale_ood", False))),
        "is_heldout_combo": int(bool(episode.diagnostics.get("is_heldout_combo", False))),
        "map_seed": episode.map_seed,
        "goal_seed": episode.goal_seed,
        "regime_seed": episode.regime_seed,
        "provider": provider_name,
        "success": int(result.success),
        "failure_reason": result.failure_reason,
        "arrival_time": result.arrival_time,
        "elapsed_steps": result.elapsed_steps,
        "collision_adjusted_arrival": float(score),
        "collisions": result.collisions,
        "first_collision_time": "" if result.first_collision_time is None else result.first_collision_time,
        "cumulative_expansions": result.cumulative_expansions,
        "planning_ms": result.planning_ms,
        "forecast_ms": result.forecast_ms,
        "encoded_frames": result.encoded_frames,
        "inference_calls": result.inference_calls,
        "replans": result.replans,
        "failed_plans": result.failed_plans,
        "observation_updates": result.observation_updates,
        "first_action": _action_signature(result.first_action),
        "oracle_first_action_hint": f"edge:{episode.oracle_first_action_hint[0]}->{episode.oracle_first_action_hint[1]}",
        "route_short_length": episode.route_lengths[0],
        "route_long_length": episode.route_lengths[1],
        "route_ratio": episode.diagnostics.get("route_ratio", float("nan")),
        "fast_period": episode.schedule.fast_period,
        "slow_dwell": episode.schedule.slow_dwell,
    }


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _num(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _provider_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "episodes": 0,
            "completion": float("nan"),
            "collision_rate": float("nan"),
            "collision_adjusted_arrival_mean": float("nan"),
            "arrival_success_mean": float("nan"),
            "expansions_mean": float("nan"),
            "planning_ms_mean": float("nan"),
        }
    successes = np.asarray([_num(row, "success") for row in rows], dtype=np.float64)
    scores = np.asarray([_num(row, "collision_adjusted_arrival") for row in rows], dtype=np.float64)
    arrivals = np.asarray(
        [_num(row, "elapsed_steps") for row in rows if _num(row, "success") >= 0.5],
        dtype=np.float64,
    )
    return {
        "episodes": len(rows),
        "completion": float(np.mean(successes)),
        "collision_rate": float(np.mean([_num(row, "collisions") > 0 for row in rows])),
        "collision_adjusted_arrival_mean": float(np.mean(scores)),
        "arrival_success_mean": float(np.mean(arrivals)) if arrivals.size else float("nan"),
        "expansions_mean": float(np.mean([_num(row, "cumulative_expansions") for row in rows])),
        "planning_ms_mean": float(np.mean([_num(row, "planning_ms") for row in rows])),
    }


def _bootstrap_history(
    rows: Sequence[Mapping[str, Any]],
    history_provider: str,
    samples: int,
    seed: int = 20260710,
) -> Dict[str, float]:
    challenge = [row for row in rows if str(row["stratum"]) in L.CHALLENGE_STRATA]
    by_pair: Dict[str, List[Mapping[str, Any]]] = {}
    for row in challenge:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)
    pair_ids = sorted(by_pair)
    if not pair_ids:
        return {
            "completion_gain_ci_low": float("nan"),
            "completion_gain_ci_high": float("nan"),
            "regret_reduction_ci_low": float("nan"),
            "regret_reduction_ci_high": float("nan"),
        }
    rng = np.random.default_rng(seed)
    completion: List[float] = []
    reduction: List[float] = []
    for _ in range(max(1, int(samples))):
        selected = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        frozen_rows: List[Mapping[str, Any]] = []
        history_rows: List[Mapping[str, Any]] = []
        for pair_id in selected:
            cluster = by_pair[str(pair_id)]
            frozen_rows.extend(row for row in cluster if row["provider"] == "frozen_frame")
            history_rows.extend(row for row in cluster if row["provider"] == history_provider)
        frozen_completion = np.mean([_num(row, "success") for row in frozen_rows])
        history_completion = np.mean([_num(row, "success") for row in history_rows])
        frozen_score = np.mean([_num(row, "collision_adjusted_arrival") for row in frozen_rows])
        history_score = np.mean([_num(row, "collision_adjusted_arrival") for row in history_rows])
        completion.append(float(history_completion - frozen_completion))
        reduction.append(float((frozen_score - history_score) / max(1e-12, frozen_score)))
    return {
        "completion_gain_ci_low": float(np.quantile(completion, 0.025)),
        "completion_gain_ci_high": float(np.quantile(completion, 0.975)),
        "regret_reduction_ci_low": float(np.quantile(reduction, 0.025)),
        "regret_reduction_ci_high": float(np.quantile(reduction, 0.975)),
    }


def evaluate_g0_gates(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the approved inclusive G0-A thresholds as pure booleans."""
    alias_ok = float(metrics["alias_rate"]) >= 0.15
    history_ok = float(metrics["history_completion_gain"]) >= 0.15 or (
        float(metrics["history_regret_reduction_frac"]) >= 0.25
        and float(metrics["history_regret_reduction_ci_low"]) > 0.0
    )
    ceiling_ok = (
        float(metrics["oracle_completion"]) >= 0.85
        and float(metrics["ceiling_gap_frac"]) >= 0.20
    )
    control = float(metrics["control_headroom"])
    margin = float(metrics.get("control_margin_required", 0.10))
    challenge = dict(metrics["challenge_headroom"])
    control_ok = bool(challenge) and all(float(value) - control >= margin for value in challenge.values())
    conditions = {
        "aliasing_exists": bool(alias_ok),
        "history_matters": bool(history_ok),
        "ceiling_exists": bool(ceiling_ok),
        "control_behaves": bool(control_ok),
    }
    return {
        "passed": bool(all(conditions.values())),
        "conditions": conditions,
        "thresholds": {
            "alias_rate_min": 0.15,
            "history_completion_gain_min": 0.15,
            "history_regret_reduction_min": 0.25,
            "oracle_completion_min": 0.85,
            "ceiling_gap_min": 0.20,
            "control_headroom_margin_min": margin,
        },
    }


def _summarize(
    rows: List[Dict[str, Any]],
    alias_audits: List[Dict[str, Any]],
    config_hash: str,
    scale: str,
    pairs_per_stratum: int,
    bootstrap_samples: int,
) -> Dict[str, Any]:
    by_stratum: Dict[str, Dict[str, Any]] = {}
    for stratum in L.STRATA:
        by_stratum[stratum] = {}
        for provider in PROVIDER_NAMES:
            subset = [
                row
                for row in rows
                if row["stratum"] == stratum and row["provider"] == provider
            ]
            by_stratum[stratum][provider] = _provider_stats(subset)

    challenge_rows = [row for row in rows if row["stratum"] in L.CHALLENGE_STRATA]
    challenge_stats = {
        provider: _provider_stats(
            [row for row in challenge_rows if row["provider"] == provider]
        )
        for provider in PROVIDER_NAMES
    }
    history_provider = max(
        ("constant_velocity", "true_mode"),
        key=lambda name: (
            challenge_stats[name]["completion"],
            -challenge_stats[name]["collision_adjusted_arrival_mean"],
        ),
    )
    frozen = challenge_stats["frozen_frame"]
    history = challenge_stats[history_provider]
    oracle = challenge_stats["oracle_future"]
    nonoracle_provider = max(
        ("frozen_frame", "constant_velocity", "true_mode"),
        key=lambda name: (
            challenge_stats[name]["completion"],
            -challenge_stats[name]["collision_adjusted_arrival_mean"],
        ),
    )
    nonoracle = challenge_stats[nonoracle_provider]

    # Require the observed oracle actions, not only the constructor hints, to
    # diverge within each declared alias pair.
    oracle_actions: Dict[str, List[str]] = {}
    for row in rows:
        if row["provider"] == "oracle_future":
            oracle_actions.setdefault(str(row["pair_id"]), []).append(str(row["first_action"]))
    audited_challenge = [a for a in alias_audits if a["stratum"] in L.CHALLENGE_STRATA]
    qualified_pairs = 0
    for audit in audited_challenge:
        actions = oracle_actions.get(str(audit["pair_id"]), [])
        actual_diverges = len(actions) == 2 and len(set(actions)) == 2
        audit["actual_oracle_first_action_diverges"] = actual_diverges
        audit["qualified_alias"] = bool(
            audit["current_match"] and audit["future_diverges"] and actual_diverges
        )
        qualified_pairs += int(audit["qualified_alias"])
    eligible_alias_decisions = 2 * len(audited_challenge)
    alias_rate = (
        2.0 * qualified_pairs / eligible_alias_decisions if eligible_alias_decisions else 0.0
    )

    history_gain = float(history["completion"] - frozen["completion"])
    history_reduction = float(
        (frozen["collision_adjusted_arrival_mean"] - history["collision_adjusted_arrival_mean"])
        / max(1e-12, frozen["collision_adjusted_arrival_mean"])
    )
    bootstrap = _bootstrap_history(rows, history_provider, bootstrap_samples)
    regret_gap = float(
        (nonoracle["collision_adjusted_arrival_mean"] - oracle["collision_adjusted_arrival_mean"])
        / max(1e-12, nonoracle["collision_adjusted_arrival_mean"])
    )
    work_gap = float(
        (nonoracle["expansions_mean"] - oracle["expansions_mean"])
        / max(1e-12, nonoracle["expansions_mean"])
    )
    ceiling_gap = max(regret_gap, work_gap)

    challenge_headroom: Dict[str, float] = {}
    for stratum in L.CHALLENGE_STRATA:
        frozen_completion = by_stratum[stratum]["frozen_frame"]["completion"]
        best_history_completion = max(
            by_stratum[stratum]["constant_velocity"]["completion"],
            by_stratum[stratum]["true_mode"]["completion"],
        )
        challenge_headroom[stratum] = float(best_history_completion - frozen_completion)
    control_headroom = float(
        by_stratum[L.CONTROL_STRATUM]["oracle_future"]["completion"]
        - by_stratum[L.CONTROL_STRATUM]["true_mode"]["completion"]
    )
    metrics = {
        "alias_rate": alias_rate,
        "qualified_alias_pairs": qualified_pairs,
        "eligible_alias_decisions": eligible_alias_decisions,
        "history_provider": history_provider,
        "history_completion_gain": history_gain,
        "history_regret_reduction_frac": history_reduction,
        "history_regret_reduction_ci_low": bootstrap["regret_reduction_ci_low"],
        "history_regret_reduction_ci_high": bootstrap["regret_reduction_ci_high"],
        "history_completion_gain_ci_low": bootstrap["completion_gain_ci_low"],
        "history_completion_gain_ci_high": bootstrap["completion_gain_ci_high"],
        "oracle_completion": float(oracle["completion"]),
        "best_nonoracle_provider": nonoracle_provider,
        "ceiling_regret_gap_frac": regret_gap,
        "ceiling_work_gap_frac": work_gap,
        "ceiling_gap_frac": ceiling_gap,
        "challenge_headroom": challenge_headroom,
        "control_headroom": control_headroom,
        # "Materially lower" was qualitative in the design.  The
        # implementation freezes it before the real probe as a 0.10 absolute
        # completion-headroom margin for every challenge stratum.
        "control_margin_required": 0.10,
    }
    gates = evaluate_g0_gates(metrics)
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "config_hash": config_hash,
        "scale": scale,
        "pairs_per_stratum": int(pairs_per_stratum),
        "episodes_per_stratum": int(2 * pairs_per_stratum),
        "total_episode_provider_rows": len(rows),
        "providers": list(PROVIDER_NAMES),
        "per_stratum": by_stratum,
        "challenge_pooled": challenge_stats,
        "metrics": metrics,
        "gates": gates,
    }


def _report(summary: Mapping[str, Any]) -> str:
    m = summary["metrics"]
    g = summary["gates"]
    status = "PASS — model training authorized" if g["passed"] else "FAIL — calibrate environment only"
    lines = [
        "# C12-A G0 Memory/Headroom Probe",
        "",
        f"**Verdict:** {status}",
        "",
        f"- Config hash: `{summary['config_hash']}`",
        f"- Scale: `{summary['scale']}` ({summary['episodes_per_stratum']} episodes/stratum)",
        f"- Constructed alias rate: {100*m['alias_rate']:.1f}%",
        f"- Best history diagnostic: `{m['history_provider']}`",
        f"- History completion gain vs frozen: {m['history_completion_gain']:+.3f}",
        f"- History collision-adjusted regret reduction: {100*m['history_regret_reduction_frac']:.1f}% "
        f"(bootstrap 95% CI {100*m['history_regret_reduction_ci_low']:.1f}% to "
        f"{100*m['history_regret_reduction_ci_high']:.1f}%)",
        f"- Oracle completion: {100*m['oracle_completion']:.1f}%",
        f"- Oracle ceiling gap vs `{m['best_nonoracle_provider']}`: {100*m['ceiling_gap_frac']:.1f}%",
        f"- Present-sufficient history headroom: {m['control_headroom']:+.3f}",
        "",
        "## Gate conditions",
        "",
    ]
    for name, passed in g["conditions"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(["", "## Challenge headroom", ""])
    for stratum, value in m["challenge_headroom"].items():
        lines.append(f"- `{stratum}`: {value:+.3f}")
    lines.extend(
        [
            "",
            "The probe uses only PROBE namespace seeds. It does not inspect final TEST worlds and does not train a learned model.",
            "",
        ]
    )
    return "\n".join(lines)


def run_probe(
    out_dir: str | Path,
    scale: str = "full",
    cfg: Optional[L.C12DynamicsConfig] = None,
    pairs_per_stratum: Optional[int] = None,
    map_families: Sequence[str] = MAP_FAMILIES,
    episode_builder: Callable[..., Tuple[L.C12EpisodeSpec, L.C12EpisodeSpec]] = L.build_challenge_pair,
    bootstrap_samples: int = 2000,
) -> Dict[str, Any]:
    cfg = cfg or L.C12DynamicsConfig()
    if scale not in SCALE_PAIRS_PER_STRATUM:
        raise KeyError(f"unknown probe scale: {scale!r}")
    n_pairs = SCALE_PAIRS_PER_STRATUM[scale] if pairs_per_stratum is None else int(pairs_per_stratum)
    if n_pairs < 1:
        raise ValueError("pairs_per_stratum must be positive")
    out_dir = Path(out_dir)
    results_dir = C.ensure_dir(out_dir / "results")
    raw_path = results_dir / "c12a_headroom_raw.csv"
    audits_path = results_dir / "c12a_alias_audits.json"
    summary_path = results_dir / "c12a_headroom_summary.json"
    report_path = results_dir / "C12A_G0_PROBE_REPORT.md"
    manifest_path = out_dir / "probe_manifest.json"
    cfg_hash = probe_config_hash(cfg, scale, n_pairs, map_families)

    prior_manifest = C.read_json(manifest_path) if manifest_path.exists() else {}
    if prior_manifest.get("config_hash") == cfg_hash:
        rows = _read_csv(raw_path)
        prior_audits = C.read_json(audits_path).get("audits", []) if audits_path.exists() else []
    else:
        rows = []
        prior_audits = []
    completed = {
        (str(row["pair_id"]), int(float(row["variant"])), str(row["provider"]))
        for row in rows
    }
    audits_by_pair = {str(row["pair_id"]): row for row in prior_audits}
    manifest = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "config_hash": cfg_hash,
        "scale": scale,
        "pairs_per_stratum": n_pairs,
        "episodes_per_stratum": 2 * n_pairs,
        "map_families": list(map_families),
        "providers": list(PROVIDER_NAMES),
        "cfg": asdict(cfg),
        "status": "running",
        "rows_complete": len(rows),
    }
    _write_json(manifest_path, manifest)

    total_pairs = len(L.STRATA) * n_pairs
    completed_pairs = 0
    t0 = time.time()
    for stratum in L.STRATA:
        for pair_index in range(n_pairs):
            family = str(map_families[pair_index % len(map_families)])
            left, right = episode_builder(
                stratum=stratum,
                pair_index=pair_index,
                map_family=family,
                cfg=cfg,
                split="PROBE",
            )
            if left.pair_id not in audits_by_pair:
                audit = L.audit_alias_pair(left, right)
                audit.update({"stratum": stratum, "map_family": family})
                audits_by_pair[left.pair_id] = audit
            for episode in (left, right):
                for provider_name in PROVIDER_NAMES:
                    key = (episode.pair_id, episode.regime.variant, provider_name)
                    if key in completed:
                        continue
                    provider = PROVIDER_FACTORIES[provider_name]()
                    result = CL.run_closed_loop_episode(episode, provider)
                    rows.append(_result_row(episode, provider_name, result))
                    completed.add(key)
            completed_pairs += 1
            if completed_pairs % 5 == 0 or completed_pairs == total_pairs:
                rows.sort(
                    key=lambda row: (
                        str(row["stratum"]),
                        int(float(row["pair_index"])),
                        int(float(row["variant"])),
                        PROVIDER_NAMES.index(str(row["provider"])),
                    )
                )
                _write_csv(raw_path, rows)
                _write_json(audits_path, {"config_hash": cfg_hash, "audits": list(audits_by_pair.values())})
                manifest["rows_complete"] = len(rows)
                manifest["pairs_complete"] = completed_pairs
                manifest["wall_s"] = time.time() - t0
                _write_json(manifest_path, manifest)
            if completed_pairs % max(1, min(10, total_pairs)) == 0:
                print(
                    f"[C12 G0-A] {completed_pairs}/{total_pairs} pairs, "
                    f"{len(rows)} rows, {time.time() - t0:.1f}s",
                    flush=True,
                )

    summary = _summarize(
        rows,
        list(audits_by_pair.values()),
        cfg_hash,
        scale,
        n_pairs,
        bootstrap_samples,
    )
    _write_json(summary_path, summary)
    _write_text(report_path, _report(summary))
    # Persist actual-oracle divergence fields added during summarization.
    _write_json(audits_path, {"config_hash": cfg_hash, "audits": list(audits_by_pair.values())})
    manifest.update(
        {
            "status": "complete",
            "rows_complete": len(rows),
            "pairs_complete": total_pairs,
            "wall_s": time.time() - t0,
            "g0_passed": bool(summary["gates"]["passed"]),
            "summary": str(summary_path),
            "report": str(report_path),
        }
    )
    _write_json(manifest_path, manifest)
    print(_report(summary), flush=True)
    return summary


def assert_probe_authorized(
    out_dir: str | Path,
    expected_hash: str,
    allow_failed_probe: bool = False,
) -> Dict[str, Any]:
    path = Path(out_dir) / "results" / "c12a_headroom_summary.json"
    if not path.exists():
        raise RuntimeError(f"C12-A training refused: probe summary missing at {path}")
    summary = C.read_json(path)
    if summary.get("config_hash") != expected_hash:
        raise RuntimeError(
            "C12-A training refused: probe config hash mismatch "
            f"({summary.get('config_hash')} != {expected_hash})"
        )
    passed = bool(summary.get("gates", {}).get("passed", False))
    if not passed and not allow_failed_probe:
        raise RuntimeError("C12-A training refused: matching G0-A probe did not pass")
    result = dict(summary)
    result["exploratory_override"] = bool(not passed and allow_failed_probe)
    return result


def _model_frame_at(batch: Mapping[str, Any], t: int) -> Dict[str, Any]:
    return {
        "frame_rasters": batch["frame_rasters"][:, t],
        "centers": batch["centers"][:, t],
        "radii": batch["radii"][:, t],
        "identity_mask": batch["identity_mask"][:, t],
        "visible_regime_context": batch["visible_regime_context"][:, t],
        "visible_regime_mask": batch["visible_regime_mask"][:, t],
        "gate_mask": batch["gate_mask"][:, t],
    }


def _merge_csv_shards_with_columns(
    paths: Sequence[Path], output_path: Path, columns: Sequence[str]
) -> None:
    C.ensure_dir(output_path.parent)
    tmp = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(columns))
        writer.writeheader()
        for path in paths:
            with path.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                if tuple(reader.fieldnames or ()) != tuple(columns):
                    raise RuntimeError(f"C12 forecast shard schema mismatch: {path}")
                for row in reader:
                    writer.writerow(row)
    _retry_atomic_write(lambda: os.replace(tmp, output_path))


def _merge_csv_shards(paths: Sequence[Path], output_path: Path) -> None:
    _merge_csv_shards_with_columns(paths, output_path, FORECAST_RAW_COLS)


def _load_trained_models(
    out_dir: Path,
    device: str,
    arms: Optional[Sequence[str]],
    seeds: Optional[Sequence[int]],
) -> List[Tuple[Dict[str, Any], WM.C12WorldModel, str]]:
    import torch

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"C12 training manifest missing at {manifest_path}")
    manifest = C.read_json(manifest_path)
    rows = list(manifest.get("training_runs", []))
    arm_filter = None if arms is None else {str(value) for value in arms}
    seed_filter = None if seeds is None else {int(value) for value in seeds}
    selected = [
        row
        for row in rows
        if (arm_filter is None or str(row["arm"]) in arm_filter)
        and (seed_filter is None or int(row["seed"]) in seed_filter)
    ]
    if not selected:
        raise RuntimeError("no C12 trained checkpoints match forecast evaluation")
    loaded = []
    for entry in sorted(
        selected, key=lambda row: (WM.ARM_NAMES.index(str(row["arm"])), int(row["seed"]))
    ):
        if entry.get("status") != "complete":
            raise RuntimeError(
                f"C12 checkpoint is not complete: {entry['arm']}/seed={entry['seed']}"
            )
        checkpoint_path = Path(str(entry.get("best_checkpoint") or ""))
        if not checkpoint_path.exists():
            candidate = out_dir / checkpoint_path
            if candidate.exists():
                checkpoint_path = candidate
            else:
                raise RuntimeError(f"C12 best checkpoint missing at {checkpoint_path}")
        checkpoint_hash = _sha256_file(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("scientific_hash") != entry.get("scientific_hash"):
            raise RuntimeError(f"C12 checkpoint/manifest hash mismatch at {checkpoint_path}")
        model = WM.build_world_model(
            str(entry["arm"]), WM.WorldModelConfig(**checkpoint["model_cfg"])
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        loaded.append((dict(entry), model, checkpoint_hash))
    return loaded


def _forecast_eval_hash(
    dataset_manifest_hash: str,
    split: str,
    models: Sequence[Tuple[Mapping[str, Any], WM.C12WorldModel, str]],
) -> str:
    payload = {
        "schema_version": FORECAST_SCHEMA_VERSION,
        "dataset_manifest_hash": dataset_manifest_hash,
        "split": split,
        "models": [
            {
                "arm": entry["arm"],
                "seed": int(entry["seed"]),
                "scientific_hash": entry["scientific_hash"],
                "checkpoint_hash": checkpoint_hash,
                "carry_modes": (
                    ["persistent"]
                    if entry["arm"] == "snapshot"
                    else list(CARRY_MODES)
                ),
            }
            for entry, _model, checkpoint_hash in models
        ],
        "horizon_buckets": list(HORIZON_BUCKETS),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def linear_probe_accuracy(
    train_context: np.ndarray,
    train_labels: np.ndarray,
    eval_context: np.ndarray,
    eval_labels: np.ndarray,
    ridge: float = 1.0e-3,
) -> Dict[str, float]:
    """Deterministic frozen-context binary ridge probe."""
    train_context = np.asarray(train_context, dtype=np.float64)
    eval_context = np.asarray(eval_context, dtype=np.float64)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    eval_labels = np.asarray(eval_labels, dtype=np.int64)
    if train_context.ndim != 2 or eval_context.ndim != 2:
        raise ValueError("linear-probe contexts must be matrices")
    if train_context.shape[1] != eval_context.shape[1]:
        raise ValueError("linear-probe train/eval widths differ")
    mean = train_context.mean(axis=0, keepdims=True)
    std = train_context.std(axis=0, keepdims=True)
    std[std < 1.0e-8] = 1.0
    train_x = (train_context - mean) / std
    eval_x = (eval_context - mean) / std
    train_x = np.concatenate((train_x, np.ones((train_x.shape[0], 1))), axis=1)
    eval_x = np.concatenate((eval_x, np.ones((eval_x.shape[0], 1))), axis=1)
    targets = train_labels.astype(np.float64) * 2.0 - 1.0
    regularizer = np.eye(train_x.shape[1], dtype=np.float64) * float(ridge)
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(train_x.T @ train_x + regularizer, train_x.T @ targets)
    predicted = (eval_x @ weights >= 0.0).astype(np.int64)
    accuracy = float(np.mean(predicted == eval_labels))
    recalls = []
    for label in (0, 1):
        mask = eval_labels == label
        if np.any(mask):
            recalls.append(float(np.mean(predicted[mask] == label)))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
    }


def _alias_contexts_for_split(
    out_dir: Path,
    store: WM.C12ShardStore,
    model: WM.C12WorldModel,
    split: str,
    device: str,
    batch_size: int = 64,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    import torch

    contexts: Dict[str, List[np.ndarray]] = {stratum: [] for stratum in L.STRATA}
    labels: Dict[str, List[int]] = {stratum: [] for stratum in L.STRATA}
    for entry in store.manifest["shards"]:
        if str(entry["split"]) != split:
            continue
        with np.load(out_dir / str(entry["path"]), allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
        diagnostics = C.read_json(out_dir / str(entry["diagnostics_path"]))["episodes"]
        count = int(entry["episodes"])
        for start in range(0, count, int(batch_size)):
            stop = min(count, start + int(batch_size))
            numpy_batch = {name: value[start:stop] for name, value in arrays.items()}
            batch = WM.tensorize_episode_batch(numpy_batch, torch.device(device))
            carry = model.initial_carry(stop - start, torch.device(device))
            alias_times = [int(diagnostics[index]["metadata"]["alias_time"]) for index in range(start, stop)]
            max_alias = max(alias_times)
            selected: Dict[int, np.ndarray] = {}
            with torch.no_grad():
                for t in range(max_alias + 1):
                    prediction, carry = model.step(_model_frame_at(batch, t), carry)
                    values = prediction["context"].detach().cpu().numpy()
                    for local, alias_time in enumerate(alias_times):
                        if t == alias_time:
                            selected[local] = values[local].copy()
            for local in range(stop - start):
                metadata = diagnostics[start + local]["metadata"]
                stratum = str(metadata["stratum"])
                contexts[stratum].append(selected[local])
                labels[stratum].append(int(metadata["variant"]))
    return {
        stratum: (
            np.asarray(contexts[stratum], dtype=np.float64),
            np.asarray(labels[stratum], dtype=np.int64),
        )
        for stratum in L.STRATA
    }


def run_latent_regime_probe(
    out_dir: str | Path,
    eval_split: str,
    device: str = "auto",
    arms: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    import torch

    out_dir = Path(out_dir)
    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        if device == "auto"
        else str(device)
    )
    store = WM.C12ShardStore(out_dir, verify=True)
    models = _load_trained_models(out_dir, resolved_device, arms, seeds)
    rows: List[Dict[str, Any]] = []
    for entry, model, checkpoint_hash in models:
        train = _alias_contexts_for_split(
            out_dir, store, model, "TRAIN", resolved_device
        )
        evaluation = _alias_contexts_for_split(
            out_dir, store, model, eval_split, resolved_device
        )
        for stratum in L.STRATA:
            train_x, train_y = train[stratum]
            eval_x, eval_y = evaluation[stratum]
            metrics = linear_probe_accuracy(train_x, train_y, eval_x, eval_y)
            rows.append(
                {
                    "schema_version": FORECAST_SCHEMA_VERSION,
                    "dataset_config_hash": store.manifest.get("config_hash"),
                    "checkpoint_hash": checkpoint_hash,
                    "arm": entry["arm"],
                    "seed": int(entry["seed"]),
                    "stratum": stratum,
                    "train_examples": int(train_y.size),
                    "eval_examples": int(eval_y.size),
                    **metrics,
                    "diagnostic_only": 1,
                }
            )
    path = C.ensure_dir(out_dir / "results") / "c12a_latent_probe.csv"
    _write_csv(path, rows)
    return rows


def run_forecast_evaluation(
    out_dir: str | Path,
    scale: str = "full",
    device: str = "auto",
    arms: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    eval_batch_size: int = 16,
) -> Dict[str, Any]:
    """Evaluate persistent/reset/window carries on identical held-out episodes."""
    import torch

    if scale not in EVAL_SPLIT_BY_SCALE:
        raise KeyError(f"unknown C12 forecast-eval scale: {scale!r}")
    out_dir = Path(out_dir)
    split = EVAL_SPLIT_BY_SCALE[scale]
    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        if device == "auto"
        else str(device)
    )
    store = WM.C12ShardStore(out_dir, verify=True)
    if split not in store.available_splits:
        raise RuntimeError(f"C12 forecast-eval split {split} is absent from the dataset")
    models = _load_trained_models(out_dir, resolved_device, arms, seeds)
    eval_hash = _forecast_eval_hash(store.manifest_hash, split, models)
    results_dir = C.ensure_dir(out_dir / "results")
    shard_dir = C.ensure_dir(results_dir / "forecast_shards")
    manifest_path = results_dir / "forecast_eval_manifest.json"
    raw_path = results_dir / "c12a_forecast_raw.csv"
    prior = C.read_json(manifest_path) if manifest_path.exists() else {}
    if prior and prior.get("eval_config_hash") != eval_hash:
        raise RuntimeError("C12 forecast evaluation config hash mismatch")
    prior_by_job = {str(row["job_id"]): row for row in prior.get("shards", [])}
    manifest: Dict[str, Any] = {
        "schema_version": FORECAST_SCHEMA_VERSION,
        "eval_config_hash": eval_hash,
        "dataset_manifest_hash": store.manifest_hash,
        "dataset_config_hash": store.manifest.get("config_hash"),
        "scale": scale,
        "split": split,
        "device": resolved_device,
        "status": "running",
        "shards": [],
    }
    _write_json(manifest_path, manifest)

    dataset_entries = [
        dict(entry)
        for entry in store.manifest["shards"]
        if str(entry["split"]) == split
    ]
    dataset_entries.sort(
        key=lambda row: (L.STRATA.index(str(row["stratum"])), int(row["shard_index"]))
    )
    output_paths: List[Path] = []
    total_rows = 0
    for training_entry, model, checkpoint_hash in models:
        arm = str(training_entry["arm"])
        seed = int(training_entry["seed"])
        modes = ("persistent",) if arm == "snapshot" else CARRY_MODES
        for carry_mode in modes:
            for dataset_entry in dataset_entries:
                job_id = (
                    f"{arm}__s{seed}__{carry_mode}__{split.lower()}__"
                    f"{dataset_entry['stratum']}__{int(dataset_entry['shard_index']):05d}"
                )
                output_path = shard_dir / f"{job_id}.csv"
                prior_entry = prior_by_job.get(job_id)
                if (
                    prior_entry
                    and output_path.exists()
                    and _sha256_file(output_path) == prior_entry.get("sha256")
                ):
                    shard_manifest = dict(prior_entry)
                else:
                    npz_path = out_dir / str(dataset_entry["path"])
                    diagnostics_path = out_dir / str(dataset_entry["diagnostics_path"])
                    with np.load(npz_path, allow_pickle=False) as payload:
                        arrays = {name: np.asarray(payload[name]) for name in payload.files}
                    diagnostics = C.read_json(diagnostics_path)["episodes"]
                    rows: List[Dict[str, Any]] = []
                    count = int(dataset_entry["episodes"])
                    for batch_start in range(0, count, int(eval_batch_size)):
                        batch_stop = min(count, batch_start + int(eval_batch_size))
                        numpy_batch = {
                            name: value[batch_start:batch_stop]
                            for name, value in arrays.items()
                        }
                        batch = WM.tensorize_episode_batch(
                            numpy_batch, torch.device(resolved_device)
                        )
                        batch_size = batch_stop - batch_start
                        carry = model.initial_carry(
                            batch_size, torch.device(resolved_device)
                        )
                        history: List[Dict[str, Any]] = []
                        steps = int(batch["frame_rasters"].shape[1])
                        with torch.no_grad():
                            for t in range(steps):
                                frame = _model_frame_at(batch, t)
                                history.append(frame)
                                if carry_mode == "persistent":
                                    prediction, carry = model.step(frame, carry)
                                elif carry_mode == "reset":
                                    fresh = model.initial_carry(
                                        batch_size, torch.device(resolved_device)
                                    )
                                    prediction, _ = model.step(frame, fresh)
                                elif carry_mode == "window_reencode":
                                    prediction = model.window_reencode(history)
                                else:
                                    raise AssertionError(carry_mode)
                                predicted_centers = (
                                    prediction["center_displacements"].detach().cpu().numpy()
                                )
                                predicted_gates = prediction["gate_logits"].detach().cpu().numpy()
                                for local_index in range(batch_size):
                                    diagnostic = diagnostics[batch_start + local_index]
                                    metadata = diagnostic["metadata"]
                                    if t < int(metadata["alias_time"]):
                                        continue
                                    metric_rows = forecast_bucket_metrics(
                                        predicted_centers[local_index],
                                        predicted_gates[local_index],
                                        numpy_batch["target_center_displacements"][local_index, t],
                                        numpy_batch["target_gate_open"][local_index, t],
                                        numpy_batch["centers"][local_index, t],
                                        numpy_batch["radii"][local_index, t],
                                        numpy_batch["identity_mask"][local_index, t],
                                        numpy_batch["gate_mask"][local_index, t],
                                        numpy_batch["route_critical_mask"][local_index, t],
                                        numpy_batch["route_edge_midpoints"][local_index],
                                        int(numpy_batch["frame_rasters"].shape[-1]),
                                    )
                                    for metric in metric_rows:
                                        row = {
                                            "schema_version": FORECAST_SCHEMA_VERSION,
                                            "eval_config_hash": eval_hash,
                                            "dataset_config_hash": store.manifest.get("config_hash"),
                                            "checkpoint_hash": checkpoint_hash,
                                            "episode_id": metadata["episode_id"],
                                            "pair_id": metadata["pair_id"],
                                            "static_map_id": metadata["static_map_id"],
                                            "split": split,
                                            "stratum": metadata["stratum"],
                                            "map_family": metadata["map_family"],
                                            "eval_condition": metadata["eval_condition"],
                                            "is_long_dwell_ood": int(metadata["is_long_dwell_ood"]),
                                            "is_scale_ood": int(metadata["is_scale_ood"]),
                                            "is_heldout_combo": int(metadata["is_heldout_combo"]),
                                            "variant": metadata["variant"],
                                            "arm": arm,
                                            "seed": seed,
                                            "carry_mode": carry_mode,
                                            "decision_step": t,
                                            **metric,
                                        }
                                        rows.append(
                                            {name: row.get(name) for name in FORECAST_RAW_COLS}
                                        )
                    _write_csv(output_path, rows)
                    shard_manifest = {
                        "job_id": job_id,
                        "path": str(output_path.relative_to(out_dir)).replace("\\", "/"),
                        "sha256": _sha256_file(output_path),
                        "rows": len(rows),
                        "arm": arm,
                        "seed": seed,
                        "carry_mode": carry_mode,
                        "stratum": dataset_entry["stratum"],
                        "dataset_shard": dataset_entry["path"],
                    }
                manifest["shards"].append(shard_manifest)
                output_paths.append(output_path)
                total_rows += int(shard_manifest["rows"])
                manifest["rows_complete"] = total_rows
                _write_json(manifest_path, manifest)
                print(f"[C12 forecast] {job_id}: {shard_manifest['rows']} rows", flush=True)

    _merge_csv_shards(output_paths, raw_path)
    latent_rows = run_latent_regime_probe(
        out_dir,
        eval_split=split,
        device=resolved_device,
        arms=[entry["arm"] for entry, _model, _hash in models],
        seeds=[int(entry["seed"]) for entry, _model, _hash in models],
    )
    manifest.update(
        {
            "status": "complete",
            "rows_complete": total_rows,
            "raw_path": str(raw_path),
            "raw_sha256": _sha256_file(raw_path),
            "latent_probe_path": str(results_dir / "c12a_latent_probe.csv"),
            "latent_probe_rows": len(latent_rows),
        }
    )
    _write_json(manifest_path, manifest)
    return manifest


def _rebuild_eval_episodes(
    out_dir: Path,
    split: str,
    episode_builder: Callable[..., Tuple[L.C12EpisodeSpec, L.C12EpisodeSpec]],
) -> List[L.C12EpisodeSpec]:
    dataset_manifest = C.read_json(out_dir / "dataset_manifest.json")
    cfg = L.C12DynamicsConfig(**dataset_manifest["cfg"])
    diagnostic_rows: List[Dict[str, Any]] = []
    for entry in dataset_manifest["shards"]:
        if str(entry["split"]) != split:
            continue
        sidecar = C.read_json(out_dir / str(entry["diagnostics_path"]))
        diagnostic_rows.extend(sidecar["episodes"])
    diagnostic_rows.sort(
        key=lambda row: (
            L.STRATA.index(str(row["metadata"]["stratum"])),
            int(row["metadata"]["pair_index"]),
            int(row["metadata"]["variant"]),
        )
    )
    pair_cache: Dict[Tuple[str, int, str], Tuple[L.C12EpisodeSpec, L.C12EpisodeSpec]] = {}
    episodes: List[L.C12EpisodeSpec] = []
    for row in diagnostic_rows:
        metadata = row["metadata"]
        key = (
            str(metadata["stratum"]),
            int(metadata["pair_index"]),
            str(metadata["map_family"]),
        )
        if key not in pair_cache:
            pair_cache[key] = episode_builder(
                stratum=key[0],
                pair_index=key[1],
                map_family=key[2],
                cfg=cfg,
                split=split,
            )
        pair = pair_cache[key]
        episode = pair[int(metadata["variant"])]
        if (
            episode.map_seed != int(metadata["map_seed"])
            or episode.goal_seed != int(metadata["goal_seed"])
            or episode.regime_seed != int(metadata["regime_seed"])
        ):
            raise RuntimeError(f"rebuilt C12 episode does not match dataset: {metadata['episode_id']}")
        episodes.append(episode)
    if not episodes:
        raise RuntimeError(f"C12 planning split {split} contains no episodes")
    return episodes


def _planning_eval_hash(
    dataset_manifest_hash: str,
    split: str,
    models: Sequence[Tuple[Mapping[str, Any], WM.C12WorldModel, str]],
    include_references: bool,
) -> str:
    payload = {
        "schema_version": PLANNING_SCHEMA_VERSION,
        "dataset_manifest_hash": dataset_manifest_hash,
        "split": split,
        "models": [
            {
                "arm": entry["arm"],
                "seed": int(entry["seed"]),
                "scientific_hash": entry["scientific_hash"],
                "checkpoint_hash": checkpoint_hash,
                "carry_modes": (
                    ["persistent"]
                    if entry["arm"] == "snapshot"
                    else list(CARRY_MODES)
                ),
            }
            for entry, _model, checkpoint_hash in models
        ],
        "references": list(PROVIDER_NAMES) if include_references else [],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def run_planning_evaluation(
    out_dir: str | Path,
    scale: str = "full",
    device: str = "auto",
    arms: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    include_references: bool = True,
    episode_builder: Callable[..., Tuple[L.C12EpisodeSpec, L.C12EpisodeSpec]] = L.build_challenge_pair,
) -> Dict[str, Any]:
    """Run matched learned forecasts through the true closed-loop simulator."""
    import torch

    if scale not in EVAL_SPLIT_BY_SCALE:
        raise KeyError(f"unknown C12 planning-eval scale: {scale!r}")
    out_dir = Path(out_dir)
    split = EVAL_SPLIT_BY_SCALE[scale]
    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        if device == "auto"
        else str(device)
    )
    store = WM.C12ShardStore(out_dir, verify=True)
    if split not in store.available_splits:
        raise RuntimeError(f"C12 planning-eval split {split} is absent from the dataset")
    models = _load_trained_models(out_dir, resolved_device, arms, seeds)
    episodes = _rebuild_eval_episodes(out_dir, split, episode_builder)
    eval_hash = _planning_eval_hash(
        store.manifest_hash, split, models, include_references
    )
    results_dir = C.ensure_dir(out_dir / "results")
    shard_dir = C.ensure_dir(results_dir / "planning_shards")
    manifest_path = results_dir / "planning_eval_manifest.json"
    raw_path = results_dir / "c12a_planning_raw.csv"
    carry_path = results_dir / "c12a_carry_ablation.csv"
    prior = C.read_json(manifest_path) if manifest_path.exists() else {}
    if prior and prior.get("eval_config_hash") != eval_hash:
        raise RuntimeError("C12 planning evaluation config hash mismatch")
    prior_by_job = {str(row["job_id"]): row for row in prior.get("shards", [])}
    manifest: Dict[str, Any] = {
        "schema_version": PLANNING_SCHEMA_VERSION,
        "eval_config_hash": eval_hash,
        "dataset_manifest_hash": store.manifest_hash,
        "dataset_config_hash": store.manifest.get("config_hash"),
        "scale": scale,
        "split": split,
        "device": resolved_device,
        "status": "running",
        "shards": [],
    }
    _write_json(manifest_path, manifest)

    jobs: List[Tuple[str, int, str, Optional[WM.C12WorldModel], str]] = []
    if include_references:
        jobs.extend((name, -1, "reference", None, "algorithmic") for name in PROVIDER_NAMES)
    for entry, model, checkpoint_hash in models:
        arm = str(entry["arm"])
        modes = ("persistent",) if arm == "snapshot" else CARRY_MODES
        jobs.extend(
            (arm, int(entry["seed"]), mode, model, checkpoint_hash)
            for mode in modes
        )
    output_paths: List[Path] = []
    carry_paths: List[Path] = []
    total_rows = 0
    for arm, seed, carry_mode, model, checkpoint_hash in jobs:
        job_id = f"{arm}__s{seed}__{carry_mode}__{split.lower()}"
        output_path = shard_dir / f"{job_id}.csv"
        prior_entry = prior_by_job.get(job_id)
        if (
            prior_entry
            and output_path.exists()
            and _sha256_file(output_path) == prior_entry.get("sha256")
        ):
            shard_manifest = dict(prior_entry)
        else:
            rows: List[Dict[str, Any]] = []
            for episode in episodes:
                if model is None:
                    provider = PROVIDER_FACTORIES[arm]()
                    provider_name = arm
                else:
                    provider = LearnedForecastProvider(
                        model, device=resolved_device, carry_mode=carry_mode
                    )
                    provider_name = provider.name
                result = CL.run_closed_loop_episode(episode, provider)
                base = _result_row(episode, provider_name, result)
                row = {
                    "schema_version": PLANNING_SCHEMA_VERSION,
                    "eval_config_hash": eval_hash,
                    "dataset_config_hash": store.manifest.get("config_hash"),
                    "checkpoint_hash": checkpoint_hash,
                    "arm": arm,
                    "seed": seed,
                    "carry_mode": carry_mode,
                    **base,
                }
                rows.append({name: row.get(name) for name in PLANNING_RAW_COLS})
            _write_csv(output_path, rows)
            shard_manifest = {
                "job_id": job_id,
                "path": str(output_path.relative_to(out_dir)).replace("\\", "/"),
                "sha256": _sha256_file(output_path),
                "rows": len(rows),
                "arm": arm,
                "seed": seed,
                "carry_mode": carry_mode,
            }
        manifest["shards"].append(shard_manifest)
        output_paths.append(output_path)
        if model is not None and arm != "snapshot":
            carry_paths.append(output_path)
        total_rows += int(shard_manifest["rows"])
        manifest["rows_complete"] = total_rows
        _write_json(manifest_path, manifest)
        print(f"[C12 planning] {job_id}: {shard_manifest['rows']} episodes", flush=True)

    _merge_csv_shards_with_columns(output_paths, raw_path, PLANNING_RAW_COLS)
    _merge_csv_shards_with_columns(carry_paths, carry_path, PLANNING_RAW_COLS)
    manifest.update(
        {
            "status": "complete",
            "rows_complete": total_rows,
            "raw_path": str(raw_path),
            "raw_sha256": _sha256_file(raw_path),
            "carry_ablation_path": str(carry_path),
            "carry_ablation_sha256": _sha256_file(carry_path),
        }
    )
    _write_json(manifest_path, manifest)
    return manifest


def run_training(
    out_dir: str | Path,
    scale: str = "full",
    cfg: Optional[L.C12DynamicsConfig] = None,
    device: str = "auto",
    probe_dir: Optional[str | Path] = None,
    allow_failed_probe: bool = False,
    arms: Sequence[str] = WM.ARM_NAMES,
    seeds: Optional[Sequence[int]] = None,
    training_cfg: Optional[WM.TrainingConfig] = None,
) -> Dict[str, Any]:
    """Train the frozen model grid after the full G0 authorization probe."""
    cfg = cfg or L.C12DynamicsConfig()
    if scale not in SCALE_MODEL_SEEDS:
        raise KeyError(f"unknown C12 training scale: {scale!r}")
    out_dir = Path(out_dir)
    probe_root = out_dir if probe_dir is None else Path(probe_dir)
    # G0-A is one full preregistered authorization experiment.  Smoke/pilot
    # model scales inherit that authorization rather than looking for weaker
    # scale-specific probe hashes.
    official_probe_hash = probe_config_hash(
        cfg,
        "full",
        SCALE_PAIRS_PER_STRATUM["full"],
        MAP_FAMILIES,
    )
    authorization = assert_probe_authorized(
        probe_root,
        official_probe_hash,
        allow_failed_probe=allow_failed_probe,
    )
    inspection = inspect_dataset(out_dir)
    store = WM.C12ShardStore(out_dir, verify=True)
    if store.manifest.get("scale") != scale:
        raise RuntimeError(
            "C12 training/dataset scale mismatch "
            f"({scale} != {store.manifest.get('scale')})"
        )
    if store.manifest.get("dynamics_config_hash") != L.config_hash(cfg):
        raise RuntimeError("C12 training/dataset dynamics config hash mismatch")
    dimensions = store.dimensions()
    sanity_path = out_dir / "tiny_alias_sanity.json"
    if sanity_path.exists():
        sanity = C.read_json(sanity_path)
    else:
        sanity = {
            "schema_version": "c12a-tiny-alias-v1",
            "results": WM.tiny_alias_sanity(device="cpu"),
        }
        sanity["passed"] = bool(
            sanity["results"]["snapshot"]["accuracy"] <= 0.75
            and all(
                sanity["results"][arm]["accuracy"] >= 0.95
                for arm in ("lstm", "temporal_transformer", "onlstm", "hrm_stream")
            )
        )
        _write_json(sanity_path, sanity)
    if sanity.get("schema_version") != "c12a-tiny-alias-v1" or not sanity.get(
        "passed", False
    ):
        raise RuntimeError("C12 temporal cores failed the frozen tiny-alias sanity")
    base_model_cfg = WM.WorldModelConfig(
        horizon=dimensions["horizon"],
        max_patrollers=dimensions["max_patrollers"],
        max_gates=dimensions["max_gates"],
        raster_channels=dimensions["raster_channels"],
    )
    computed_grid = WM.select_model_grid(
        base_model_cfg, raster_size=dimensions["raster_size"]
    )
    computed_grid.update(
        {
            "dataset_config_hash": store.manifest.get("config_hash"),
            "dynamics_config_hash": L.config_hash(cfg),
            "raster_size": dimensions["raster_size"],
        }
    )
    grid_path = out_dir / "model_grid.json"
    if grid_path.exists():
        prior_grid = C.read_json(grid_path)
        if prior_grid != computed_grid:
            raise RuntimeError("frozen C12 model_grid.json does not match current config")
    else:
        WM.write_model_grid(grid_path, computed_grid)

    selected_seeds = tuple(SCALE_MODEL_SEEDS[scale] if seeds is None else seeds)
    selected_arms = tuple(str(arm) for arm in arms)
    unknown = sorted(set(selected_arms).difference(WM.ARM_NAMES))
    if unknown:
        raise KeyError(f"unknown C12 training arms: {', '.join(unknown)}")
    if not selected_seeds:
        raise ValueError("C12 training requires at least one model seed")
    if device == "auto":
        import torch

        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = str(device)
    train_cfg = training_cfg or WM.TrainingConfig()
    entries: List[Dict[str, Any]] = []
    ledger_path = out_dir / "run_ledger.json"
    ledger: Dict[str, Any] = {
        "schema_version": "c12a-run-ledger-v1",
        "mode": "train",
        "scale": scale,
        "status": "running",
        "dataset_config_hash": store.manifest.get("config_hash"),
        "probe_config_hash": official_probe_hash,
        "exploratory_override": authorization["exploratory_override"],
        "device": resolved_device,
        "arms": list(selected_arms),
        "seeds": [int(seed) for seed in selected_seeds],
        "completed": [],
    }
    _write_json(ledger_path, ledger)
    for arm in selected_arms:
        model_cfg = WM.WorldModelConfig(**computed_grid["arms"][arm]["config"])
        for seed in selected_seeds:
            entry = WM.train_world_model(
                store,
                out_dir,
                arm,
                int(seed),
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                device=resolved_device,
            )
            entries.append(entry)
            ledger["completed"].append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "status": entry["status"],
                    "best_validation": entry.get("best_validation"),
                }
            )
            _write_json(ledger_path, ledger)
            print(
                f"[C12 train] {arm}/seed={seed}: {entry['status']}, "
                f"best_val={entry.get('best_validation')}",
                flush=True,
            )
    failed = [entry for entry in entries if entry["status"] != "complete"]
    comparator_selection: Optional[Dict[str, Any]] = None
    if not failed:
        validation_scores = validation_long_horizon_scores(
            out_dir,
            store,
            resolved_device,
            arms=selected_arms,
            seeds=selected_seeds,
        )
        comparator_selection = freeze_flat_comparator(
            out_dir,
            entries,
            store.manifest_hash,
            validation_scores=validation_scores,
        )
    ledger["status"] = "complete" if not failed else "failed"
    ledger["failed"] = [
        {"arm": entry["arm"], "seed": entry["seed"], "status": entry["status"]}
        for entry in failed
    ]
    _write_json(ledger_path, ledger)
    return {
        "status": ledger["status"],
        "scale": scale,
        "device": resolved_device,
        "dataset": inspection,
        "model_grid": computed_grid,
        "training_runs": entries,
        "comparator_selection": comparator_selection,
    }


def validation_long_horizon_scores(
    out_dir: str | Path,
    store: WM.C12ShardStore,
    device: str,
    arms: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    batch_size: int = 16,
) -> Dict[Tuple[str, int], float]:
    """Rank arms on snapshot-normalized, mechanism-specific G1 error."""
    import torch

    out_dir = Path(out_dir)
    models = _load_trained_models(out_dir, device, arms, seeds)
    validation_entries = [
        dict(entry)
        for entry in store.manifest["shards"]
        if str(entry["split"]) == "VALIDATION"
        and str(entry["stratum"]) in L.CHALLENGE_STRATA
    ]
    if not validation_entries:
        raise RuntimeError("C12 comparator selection requires challenge VALIDATION shards")
    burn_in = int(store.manifest.get("cfg", {}).get("burn_in", 0))
    raw_components: Dict[Tuple[str, int, str], List[float]] = {}
    checkpoint_hashes: Dict[Tuple[str, int], str] = {}
    for entry, model, checkpoint_hash in models:
        arm = str(entry["arm"])
        seed = int(entry["seed"])
        checkpoint_hashes[(arm, seed)] = checkpoint_hash
        for shard in validation_entries:
            stratum = str(shard["stratum"])
            with np.load(out_dir / str(shard["path"]), allow_pickle=False) as payload:
                arrays = {name: np.asarray(payload[name]) for name in payload.files}
            episodes = int(shard["episodes"])
            for start in range(0, episodes, int(batch_size)):
                stop = min(episodes, start + int(batch_size))
                numpy_batch = {
                    name: value[start:stop] for name, value in arrays.items()
                }
                batch = WM.tensorize_episode_batch(
                    numpy_batch, torch.device(device)
                )
                carry = model.initial_carry(stop - start, torch.device(device))
                with torch.no_grad():
                    for t in range(int(batch["frame_rasters"].shape[1])):
                        prediction, carry = model.step(_model_frame_at(batch, t), carry)
                        if t < burn_in:
                            continue
                        predicted_centers = (
                            prediction["center_displacements"].detach().cpu().numpy()
                        )
                        predicted_gates = (
                            prediction["gate_logits"].detach().cpu().numpy()
                        )
                        for local in range(stop - start):
                            metric_sum, metric_count, _ = _g1_mechanism_components(
                                stratum,
                                predicted_centers[local],
                                predicted_gates[local],
                                numpy_batch["target_center_displacements"][local, t],
                                numpy_batch["target_gate_open"][local, t],
                                numpy_batch["identity_mask"][local, t],
                                numpy_batch["gate_mask"][local, t],
                                numpy_batch["route_critical_mask"][local, t],
                            )
                            bucket = raw_components.setdefault(
                                (arm, seed, stratum), [0.0, 0.0]
                            )
                            bucket[0] += metric_sum
                            bucket[1] += metric_count

    raw_errors: Dict[Tuple[str, int, str], float] = {}
    for key, (metric_sum, metric_count) in raw_components.items():
        if metric_count <= 0:
            raise RuntimeError(
                "C12 validation long-horizon metric is undefined for "
                f"{key[0]}/seed={key[1]}/{key[2]}"
            )
        raw_errors[key] = metric_sum / metric_count

    model_keys = sorted(
        checkpoint_hashes,
        key=lambda key: (WM.ARM_NAMES.index(key[0]), key[1]),
    )
    scores: Dict[Tuple[str, int], float] = {}
    normalized_by_key: Dict[Tuple[str, int, str], float] = {}
    for arm, seed in model_keys:
        normalized = []
        for stratum in L.CHALLENGE_STRATA:
            raw_key = (arm, seed, stratum)
            baseline_key = ("snapshot", seed, stratum)
            if raw_key not in raw_errors or baseline_key not in raw_errors:
                raise RuntimeError(
                    "C12 comparator selection requires every arm and the snapshot "
                    f"baseline for seed={seed}/{stratum}"
                )
            baseline = raw_errors[baseline_key]
            if baseline <= 1e-12:
                raise RuntimeError(
                    "C12 snapshot-normalized validation metric is undefined because "
                    f"snapshot error is zero for seed={seed}/{stratum}"
                )
            value = raw_errors[raw_key] / baseline
            normalized_by_key[raw_key] = value
            normalized.append(value)
        scores[(arm, seed)] = float(np.mean(normalized))

    rows: List[Dict[str, Any]] = []
    for arm, seed in model_keys:
        for stratum in L.CHALLENGE_STRATA:
            raw_key = (arm, seed, stratum)
            metric_sum, metric_count = raw_components[raw_key]
            rows.append(
                {
                    "schema_version": "c12a-validation-selection-v3",
                    "dataset_manifest_hash": store.manifest_hash,
                    "dataset_config_hash": store.manifest.get("config_hash"),
                    "checkpoint_hash": checkpoint_hashes[(arm, seed)],
                    "arm": arm,
                    "seed": seed,
                    "stratum": stratum,
                    "raw_metric": _g1_metric_name(stratum),
                    "metric_sum": metric_sum,
                    "metric_count": int(metric_count),
                    "raw_error": raw_errors[raw_key],
                    "snapshot_error": raw_errors[("snapshot", seed, stratum)],
                    "snapshot_normalized_error": normalized_by_key[raw_key],
                    "selection_score_mean": scores[(arm, seed)],
                    "split": "VALIDATION",
                    "challenge_strata_only": 1,
                }
            )
    _write_csv(out_dir / "validation_selection_metrics.csv", rows)
    return scores


def freeze_flat_comparator(
    out_dir: str | Path,
    training_entries: Sequence[Mapping[str, Any]],
    dataset_manifest_hash: str,
    validation_scores: Optional[Mapping[Tuple[str, int], float]] = None,
) -> Dict[str, Any]:
    """Persist the validation-only flat ranking before any TEST evaluation."""
    out_dir = Path(out_dir)
    candidates = ("lstm", "temporal_transformer")
    ranking = []
    for arm in candidates:
        values = [
            float(
                validation_scores[(arm, int(row["seed"]))]
                if validation_scores is not None
                else row["best_validation"]
            )
            for row in training_entries
            if str(row.get("arm")) == arm
            and row.get("status") == "complete"
            and row.get("best_validation") is not None
            and (
                validation_scores is None
                or (arm, int(row["seed"])) in validation_scores
            )
        ]
        if not values:
            raise RuntimeError(f"cannot freeze C12 flat comparator without {arm}")
        ranking.append(
            {
                "arm": arm,
                "validation_normalized_g1_error_mean": float(
                    np.mean(values)
                ),
                "model_seeds": len(values),
            }
        )
    hierarchy_ranking = []
    for arm in ("onlstm", "hrm_stream"):
        values = [
            float(
                validation_scores[(arm, int(row["seed"]))]
                if validation_scores is not None
                else row["best_validation"]
            )
            for row in training_entries
            if str(row.get("arm")) == arm
            and row.get("status") == "complete"
            and row.get("best_validation") is not None
            and (
                validation_scores is None
                or (arm, int(row["seed"])) in validation_scores
            )
        ]
        if values:
            hierarchy_ranking.append(
                {
                    "arm": arm,
                    "validation_normalized_g1_error_mean": float(
                        np.mean(values)
                    ),
                    "model_seeds": len(values),
                }
            )
    hierarchy_ranking.sort(
        key=lambda row: (
            row["validation_normalized_g1_error_mean"],
            row["arm"],
        )
    )
    ranking.sort(
        key=lambda row: (
            row["validation_normalized_g1_error_mean"],
            row["arm"],
        )
    )
    payload = {
        "schema_version": "c12a-comparator-selection-v4",
        "dataset_manifest_hash": dataset_manifest_hash,
        "selection_data": "VALIDATION only",
        "criterion": (
            "mean snapshot-normalized persistent VALIDATION mechanism-specific G1 error "
            "at horizons 17-32 across direction, slow-gate, and route-mode strata and "
            "across model seeds"
            if validation_scores is not None
            else "legacy mean best route-critical composite checkpoint loss"
        ),
        "selected_best_flat": ranking[0]["arm"],
        "selected_flat_recurrent": "lstm",
        "required_transformer_comparator": "temporal_transformer",
        "selected_hierarchy": hierarchy_ranking[0]["arm"] if hierarchy_ranking else None,
        "ranking": ranking,
        "hierarchy_ranking": hierarchy_ranking,
        "frozen_before_test": True,
    }
    path = out_dir / "comparator_selection.json"
    if path.exists():
        prior = C.read_json(path)
        if prior != payload:
            raise RuntimeError("frozen C12 comparator selection would change")
    else:
        _write_json(path, payload)
    return payload


def run_full_pipeline(
    out_dir: str | Path,
    scale: str = "smoke",
    cfg: Optional[L.C12DynamicsConfig] = None,
    device: str = "auto",
    probe_dir: Optional[str | Path] = None,
    allow_failed_probe: bool = False,
    bootstrap_samples: int = 5000,
) -> Dict[str, Any]:
    """Resumable collect -> train -> forecast -> plan -> analyze pipeline."""
    cfg = cfg or L.C12DynamicsConfig()
    out_dir = Path(out_dir)
    probe_root = out_dir if probe_dir is None else Path(probe_dir)
    pipeline_path = out_dir / "pipeline_manifest.json"
    pipeline: Dict[str, Any] = {
        "schema_version": "c12a-pipeline-v1",
        "scale": scale,
        "status": "running",
        "steps": {},
    }
    _write_json(pipeline_path, pipeline)
    dataset = collect_dataset(out_dir, scale=scale, cfg=cfg)
    pipeline["steps"]["collect"] = {
        "status": dataset["status"],
        "config_hash": dataset["config_hash"],
        "episodes": dataset["episodes_total"],
    }
    _write_json(pipeline_path, pipeline)
    training = run_training(
        out_dir,
        scale=scale,
        cfg=cfg,
        device=device,
        probe_dir=probe_root,
        allow_failed_probe=allow_failed_probe,
    )
    pipeline["steps"]["train"] = {
        "status": training["status"],
        "runs": len(training["training_runs"]),
    }
    _write_json(pipeline_path, pipeline)
    if training["status"] != "complete":
        pipeline["status"] = "failed"
        _write_json(pipeline_path, pipeline)
        return pipeline
    forecast = run_forecast_evaluation(
        out_dir, scale=scale, device=device
    )
    pipeline["steps"]["forecast_eval"] = {
        "status": forecast["status"],
        "rows": forecast["rows_complete"],
    }
    _write_json(pipeline_path, pipeline)
    planning = run_planning_evaluation(
        out_dir, scale=scale, device=device
    )
    pipeline["steps"]["plan_eval"] = {
        "status": planning["status"],
        "rows": planning["rows_complete"],
    }
    _write_json(pipeline_path, pipeline)
    analysis = run_analysis(
        out_dir,
        scale=scale,
        probe_dir=probe_root,
        bootstrap_samples=bootstrap_samples,
    )
    pipeline["steps"]["analyze"] = {
        "status": "complete",
        "official_final": analysis["official_final"],
        "verdict": analysis["headline_gates"]["G4_A"]["verdict"],
    }
    pipeline["status"] = "complete"
    _write_json(pipeline_path, pipeline)
    return pipeline


def _stable_analysis_seed(*parts: Any) -> int:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "little")


def _paired_map_comparison(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *seed_parts: Any,
    bootstrap_samples: int,
) -> Dict[str, Any]:
    worlds = sorted(set(left).intersection(right))
    differences = [float(left[world] - right[world]) for world in worlds]
    seed = _stable_analysis_seed(*seed_parts)
    interval = seeded_world_bootstrap(
        differences, seed=seed, samples=bootstrap_samples
    )
    return {
        "worlds": len(worlds),
        "mean_difference": interval["mean"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "p_value": paired_sign_flip_p(differences, seed=seed),
        "differences": differences,
    }


def _aggregate_forecast_worlds(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str, str, str, str, str], float]:
    by_seed: Dict[Tuple[str, str, str, str, str, str, int], List[float]] = {}
    for row in rows:
        if str(row["stratum"]) == "slow_gate_phase":
            sum_column = "route_critical_gate_brier_sum"
            count_column = "route_critical_gate_brier_count"
        else:
            sum_column = "route_critical_ade_sum"
            count_column = "route_critical_ade_count"
        count = int(float(row.get(count_column, 0) or 0))
        if count <= 0:
            continue
        key = (
            str(row["arm"]),
            str(row["carry_mode"]),
            str(row["stratum"]),
            str(row["horizon_bucket"]),
            str(row["pair_id"]),
            str(row.get("eval_condition", "matched_id")),
            int(float(row["seed"])),
        )
        bucket = by_seed.setdefault(key, [0.0, 0.0])
        bucket[0] += float(row[sum_column])
        bucket[1] += count
    across_seeds: Dict[Tuple[str, str, str, str, str, str], List[float]] = {}
    for key, (total, count) in by_seed.items():
        across_seeds.setdefault(key[:-1], []).append(total / count)
    return {key: float(np.mean(values)) for key, values in across_seeds.items()}


def _forecast_world_map(
    aggregate: Mapping[Tuple[str, str, str, str, str, str], float],
    arm: str,
    carry_mode: str,
    stratum: str,
    bucket: str,
    condition: Optional[str] = None,
) -> Dict[str, float]:
    return {
        pair_id: value
        for (candidate, mode, suite, horizon, pair_id, eval_condition), value in aggregate.items()
        if candidate == arm
        and mode == carry_mode
        and suite == stratum
        and horizon == bucket
        and (condition is None or eval_condition == condition)
    }


def _aggregate_planning_worlds(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str, str, str, str, str], float]:
    metrics = (
        "success",
        "collisions",
        "cumulative_expansions",
        "collision_adjusted_arrival",
        "forecast_ms",
        "encoded_frames",
    )
    by_seed: Dict[Tuple[str, str, str, str, str, int], Dict[str, List[float]]] = {}
    for row in rows:
        if int(float(row.get("seed", -1))) < 0:
            continue
        key = (
            str(row["arm"]),
            str(row["carry_mode"]),
            str(row["stratum"]),
            str(row["pair_id"]),
            str(row.get("eval_condition", "matched_id")),
            int(float(row["seed"])),
        )
        target = by_seed.setdefault(
            key,
            {
                name: []
                for name in metrics
                + ("quality", "encoded_frames_per_replan", "forecast_ms_per_replan")
            },
        )
        for metric in metrics:
            target[metric].append(float(row.get(metric, 0.0) or 0.0))
        quality = (
            float(row["success"])
            - float(row["collisions"])
            - float(row["collision_adjusted_arrival"]) / 160.0
        )
        target["quality"].append(quality)
        replans = max(1.0, float(row.get("replans", 0.0) or 0.0))
        target["encoded_frames_per_replan"].append(
            float(row.get("encoded_frames", 0.0) or 0.0) / replans
        )
        target["forecast_ms_per_replan"].append(
            float(row.get("forecast_ms", 0.0) or 0.0) / replans
        )
    across_seeds: Dict[Tuple[str, str, str, str, str, str], List[float]] = {}
    for key, values in by_seed.items():
        for metric, metric_values in values.items():
            across_seeds.setdefault((*key[:-1], metric), []).append(
                float(np.mean(metric_values))
            )
    return {key: float(np.mean(values)) for key, values in across_seeds.items()}


def _planning_world_map(
    aggregate: Mapping[Tuple[str, str, str, str, str, str], float],
    arm: str,
    carry_mode: str,
    stratum: str,
    metric: str,
    condition: Optional[str] = None,
) -> Dict[str, float]:
    return {
        pair_id: value
        for (candidate, mode, suite, pair_id, eval_condition, candidate_metric), value in aggregate.items()
        if candidate == arm
        and mode == carry_mode
        and suite == stratum
        and candidate_metric == metric
        and (condition is None or eval_condition == condition)
    }


def _relative_reduction_map(
    hierarchy: Mapping[str, float], comparator: Mapping[str, float]
) -> Dict[str, float]:
    return {
        world: (float(comparator[world]) - float(hierarchy[world]))
        / max(abs(float(comparator[world])), 1.0e-12)
        for world in set(hierarchy).intersection(comparator)
    }


def _significance_row(
    family: str,
    hierarchy: str,
    comparator: str,
    stratum: str,
    metric: str,
    result: Mapping[str, Any],
    q_value: float,
) -> Dict[str, Any]:
    return {
        "family": family,
        "hierarchy": hierarchy,
        "comparator": comparator,
        "stratum": stratum,
        "metric": metric,
        "worlds": int(result["worlds"]),
        "mean_difference": result["mean_difference"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "p_value": result["p_value"],
        "q_value": q_value,
    }


def run_analysis(
    out_dir: str | Path,
    scale: str = "full",
    probe_dir: Optional[str | Path] = None,
    bootstrap_samples: int = 5000,
) -> Dict[str, Any]:
    """Generate clustered G1--G4 results from immutable raw artifacts."""
    out_dir = Path(out_dir)
    results_dir = C.ensure_dir(out_dir / "results")
    forecast_path = results_dir / "c12a_forecast_raw.csv"
    planning_path = results_dir / "c12a_planning_raw.csv"
    selection_path = out_dir / "comparator_selection.json"
    for required in (forecast_path, planning_path, selection_path):
        if not required.exists():
            raise RuntimeError(f"C12 analysis artifact missing at {required}")
    forecast_rows = _read_csv(forecast_path)
    planning_rows = _read_csv(planning_path)
    selection = C.read_json(selection_path)
    selected_hierarchy = selection.get("selected_hierarchy")
    if selected_hierarchy not in ("onlstm", "hrm_stream"):
        raise RuntimeError("C12 hierarchy was not frozen from validation")
    best_flat = str(selection["selected_best_flat"])
    flat_recurrent = str(selection["selected_flat_recurrent"])
    transformer = str(selection["required_transformer_comparator"])
    significance_rows: List[Dict[str, Any]] = []

    forecast_aggregate = _aggregate_forecast_worlds(forecast_rows)
    all_g1: Dict[str, Any] = {}
    for hierarchy in ("onlstm", "hrm_stream"):
        comparisons: Dict[str, Dict[str, Dict[str, float]]] = {
            stratum: {} for stratum in L.CHALLENGE_STRATA
        }
        for label, comparator in (
            ("flat_recurrent", flat_recurrent),
            ("temporal_transformer", transformer),
        ):
            results = []
            for stratum in L.CHALLENGE_STRATA:
                result = _paired_map_comparison(
                    _forecast_world_map(
                        forecast_aggregate, hierarchy, "persistent", stratum, "h17_32"
                    ),
                    _forecast_world_map(
                        forecast_aggregate, comparator, "persistent", stratum, "h17_32"
                    ),
                    "g1",
                    hierarchy,
                    comparator,
                    stratum,
                    bootstrap_samples=bootstrap_samples,
                )
                results.append(result)
            q_values = bh_q_values([result["p_value"] for result in results])
            for stratum, result, q_value in zip(L.CHALLENGE_STRATA, results, q_values):
                comparisons[stratum][label] = {
                    **{key: value for key, value in result.items() if key != "differences"},
                    "q_value": q_value,
                }
                significance_rows.append(
                    _significance_row(
                        "G1",
                        hierarchy,
                        comparator,
                        stratum,
                        "route_critical_mechanism_error_h17_32",
                        result,
                        q_value,
                    )
                )
        relative: Dict[str, float] = {}
        for bucket in ("h01_04", "h17_32"):
            h_values: Dict[str, float] = {}
            c_values: Dict[str, float] = {}
            for stratum in L.CHALLENGE_STRATA:
                h_values.update(
                    _forecast_world_map(
                        forecast_aggregate, hierarchy, "persistent", stratum, bucket
                    )
                )
                c_values.update(
                    _forecast_world_map(
                        forecast_aggregate, best_flat, "persistent", stratum, bucket
                    )
                )
            reductions = _relative_reduction_map(h_values, c_values)
            relative[bucket] = float(np.mean(list(reductions.values()))) if reductions else float("nan")
        control_result = _paired_map_comparison(
            _forecast_world_map(
                forecast_aggregate,
                hierarchy,
                "persistent",
                L.CONTROL_STRATUM,
                "h17_32",
            ),
            _forecast_world_map(
                forecast_aggregate,
                best_flat,
                "persistent",
                L.CONTROL_STRATUM,
                "h17_32",
            ),
            "g1-control",
            hierarchy,
            bootstrap_samples=bootstrap_samples,
        )
        control_not_worse = not bool(
            control_result["mean_difference"] > 0.0
            and control_result["ci_low"] > 0.0
            and control_result["p_value"] <= 0.05
        )
        gate = evaluate_g1_gate(
            comparisons,
            long_advantage_not_smaller=bool(
                np.isfinite(relative["h17_32"])
                and np.isfinite(relative["h01_04"])
                and relative["h17_32"] >= relative["h01_04"]
            ),
            control_not_worse=control_not_worse,
        )
        all_g1[hierarchy] = {
            "gate": gate,
            "comparisons": comparisons,
            "relative_advantage": relative,
            "control_comparison": {
                key: value for key, value in control_result.items() if key != "differences"
            },
        }

    planning_aggregate = _aggregate_planning_worlds(planning_rows)
    all_g2: Dict[str, Any] = {}
    all_g3: Dict[str, Any] = {}
    for hierarchy in ("onlstm", "hrm_stream"):
        rows_by_stratum: Dict[str, Dict[str, Any]] = {}
        completion_results = []
        collision_results = []
        for stratum in L.CHALLENGE_STRATA:
            completion = _paired_map_comparison(
                _planning_world_map(
                    planning_aggregate, hierarchy, "persistent", stratum, "success"
                ),
                _planning_world_map(
                    planning_aggregate, best_flat, "persistent", stratum, "success"
                ),
                "g2-completion",
                hierarchy,
                stratum,
                bootstrap_samples=bootstrap_samples,
            )
            collision = _paired_map_comparison(
                _planning_world_map(
                    planning_aggregate, hierarchy, "persistent", stratum, "collisions"
                ),
                _planning_world_map(
                    planning_aggregate, best_flat, "persistent", stratum, "collisions"
                ),
                "g2-collision",
                hierarchy,
                stratum,
                bootstrap_samples=bootstrap_samples,
            )
            expansion_reductions = _relative_reduction_map(
                _planning_world_map(
                    planning_aggregate,
                    hierarchy,
                    "persistent",
                    stratum,
                    "cumulative_expansions",
                ),
                _planning_world_map(
                    planning_aggregate,
                    best_flat,
                    "persistent",
                    stratum,
                    "cumulative_expansions",
                ),
            )
            arrival_reductions = _relative_reduction_map(
                _planning_world_map(
                    planning_aggregate,
                    hierarchy,
                    "persistent",
                    stratum,
                    "collision_adjusted_arrival",
                ),
                _planning_world_map(
                    planning_aggregate,
                    best_flat,
                    "persistent",
                    stratum,
                    "collision_adjusted_arrival",
                ),
            )
            expansion_interval = seeded_world_bootstrap(
                list(expansion_reductions.values()),
                seed=_stable_analysis_seed("g2-expansion", hierarchy, stratum),
                samples=bootstrap_samples,
            )
            arrival_interval = seeded_world_bootstrap(
                list(arrival_reductions.values()),
                seed=_stable_analysis_seed("g2-arrival", hierarchy, stratum),
                samples=bootstrap_samples,
            )
            completion_results.append(completion)
            collision_results.append(collision)
            rows_by_stratum[stratum] = {
                "completion_difference": completion["mean_difference"],
                "completion_ci_low": completion["ci_low"],
                "collision_difference": collision["mean_difference"],
                "expansion_reduction": expansion_interval["mean"],
                "expansion_reduction_ci_low": expansion_interval["ci_low"],
                "arrival_reduction": arrival_interval["mean"],
                "arrival_reduction_ci_low": arrival_interval["ci_low"],
            }
        completion_q = bh_q_values([result["p_value"] for result in completion_results])
        collision_q = bh_q_values([result["p_value"] for result in collision_results])
        for index, stratum in enumerate(L.CHALLENGE_STRATA):
            rows_by_stratum[stratum]["completion_q"] = completion_q[index]
            rows_by_stratum[stratum]["collision_q"] = collision_q[index]
            rows_by_stratum[stratum]["completion_collision_tied"] = bool(
                completion_q[index] > 0.05 and collision_q[index] > 0.05
            )
            significance_rows.extend(
                [
                    _significance_row(
                        "G2",
                        hierarchy,
                        best_flat,
                        stratum,
                        "completion",
                        completion_results[index],
                        completion_q[index],
                    ),
                    _significance_row(
                        "G2",
                        hierarchy,
                        best_flat,
                        stratum,
                        "collisions",
                        collision_results[index],
                        collision_q[index],
                    ),
                ]
            )

        def pooled_quality_difference(stratum: str, condition: Optional[str]) -> float:
            comparison = _paired_map_comparison(
                _planning_world_map(
                    planning_aggregate,
                    hierarchy,
                    "persistent",
                    stratum,
                    "quality",
                    condition,
                ),
                _planning_world_map(
                    planning_aggregate,
                    best_flat,
                    "persistent",
                    stratum,
                    "quality",
                    condition,
                ),
                "quality-slice",
                hierarchy,
                stratum,
                condition,
                bootstrap_samples=bootstrap_samples,
            )
            return float(comparison["mean_difference"])

        # Long dwell is a causal environment intervention in the slow-gate
        # stratum; the other strata retain their own registered mechanisms.
        ood_values = [
            pooled_quality_difference("slow_gate_phase", "long_dwell_ood")
        ]
        ood_values = [value for value in ood_values if np.isfinite(value)]
        control_value = pooled_quality_difference(L.CONTROL_STRATUM, None)
        ood_advantage = float(np.mean(ood_values)) if ood_values else float("nan")
        g2_gate = evaluate_g2_gate(
            rows_by_stratum,
            ood_advantage_exceeds_control=bool(
                np.isfinite(ood_advantage)
                and np.isfinite(control_value)
                and ood_advantage > control_value
            ),
        )
        all_g2[hierarchy] = {
            "gate": g2_gate,
            "comparisons": rows_by_stratum,
            "long_dwell_ood_quality_advantage": ood_advantage,
            "control_quality_advantage": control_value,
        }

        reset_results = []
        reset_rows: Dict[str, Dict[str, float]] = {}
        for stratum in L.CHALLENGE_STRATA:
            result = _paired_map_comparison(
                _planning_world_map(
                    planning_aggregate, hierarchy, "persistent", stratum, "quality"
                ),
                _planning_world_map(
                    planning_aggregate, hierarchy, "reset", stratum, "quality"
                ),
                "g3-reset",
                hierarchy,
                stratum,
                bootstrap_samples=bootstrap_samples,
            )
            reset_results.append(result)
        reset_q = bh_q_values([result["p_value"] for result in reset_results])
        for stratum, result, q_value in zip(L.CHALLENGE_STRATA, reset_results, reset_q):
            reset_rows[stratum] = {
                "quality_difference": result["mean_difference"],
                "q_value": q_value,
            }
            significance_rows.append(
                _significance_row(
                    "G3",
                    hierarchy,
                    f"{hierarchy}_reset",
                    stratum,
                    "planning_quality",
                    result,
                    q_value,
                )
            )
        persistent_quality: Dict[str, float] = {}
        window_quality: Dict[str, float] = {}
        persistent_encoded: Dict[str, float] = {}
        window_encoded: Dict[str, float] = {}
        for stratum in L.CHALLENGE_STRATA:
            persistent_quality.update(
                _planning_world_map(
                    planning_aggregate, hierarchy, "persistent", stratum, "quality"
                )
            )
            window_quality.update(
                _planning_world_map(
                    planning_aggregate, hierarchy, "window_reencode", stratum, "quality"
                )
            )
            persistent_encoded.update(
                _planning_world_map(
                    planning_aggregate,
                    hierarchy,
                    "persistent",
                    stratum,
                    "encoded_frames_per_replan",
                )
            )
            window_encoded.update(
                _planning_world_map(
                    planning_aggregate,
                    hierarchy,
                    "window_reencode",
                    stratum,
                    "encoded_frames_per_replan",
                )
            )
        window_comparison = _paired_map_comparison(
            persistent_quality,
            window_quality,
            "g3-window",
            hierarchy,
            bootstrap_samples=bootstrap_samples,
        )
        compute_reductions = _relative_reduction_map(
            persistent_encoded, window_encoded
        )
        compute_reduction = (
            float(np.mean(list(compute_reductions.values())))
            if compute_reductions
            else float("nan")
        )
        g3_gate = evaluate_g3_gate(
            reset_rows,
            window_quality_not_worse=bool(
                window_comparison["mean_difference"] >= 0.0
                or window_comparison["p_value"] > 0.05
            ),
            window_compute_reduction=compute_reduction,
        )
        all_g3[hierarchy] = {
            "gate": g3_gate,
            "reset_comparisons": reset_rows,
            "window_comparison": {
                key: value for key, value in window_comparison.items() if key != "differences"
            },
            "encoding_compute_reduction": compute_reduction,
        }

    probe_root = out_dir if probe_dir is None else Path(probe_dir)
    probe_path = probe_root / "results" / "c12a_headroom_summary.json"
    if not probe_path.exists():
        raise RuntimeError(f"C12 G0 summary missing at {probe_path}")
    g0 = C.read_json(probe_path)
    g0_passed = bool(g0.get("gates", {}).get("passed", False))
    selected_g1 = all_g1[selected_hierarchy]["gate"]
    selected_g2 = all_g2[selected_hierarchy]["gate"]
    selected_g3 = all_g3[selected_hierarchy]["gate"]
    closure = evaluate_g4_closure(
        g0_passed, bool(selected_g1["passed"]), bool(selected_g2["passed"])
    )
    summary = {
        "schema_version": "c12a-analysis-v3",
        "scale": scale,
        "official_final": scale == "full",
        "development_only": scale != "full",
        "dataset_config_hash": C.read_json(out_dir / "dataset_manifest.json").get(
            "config_hash"
        ),
        "forecast_raw_sha256": _sha256_file(forecast_path),
        "planning_raw_sha256": _sha256_file(planning_path),
        "selection": selection,
        "selected_hierarchy": selected_hierarchy,
        "selected_flat": best_flat,
        "g0": {"passed": g0_passed, "config_hash": g0.get("config_hash")},
        "g1": all_g1,
        "g2": all_g2,
        "g3": all_g3,
        "headline_gates": {
            "G1_A": selected_g1,
            "G2_A": selected_g2,
            "G3_A": selected_g3,
            "G4_A": closure,
        },
    }
    summary_path = results_dir / "c12a_summary.json"
    significance_path = results_dir / "c12a_significance.csv"
    report_path = results_dir / "C12A_ANALYSIS.md"
    _write_json(summary_path, summary)
    _write_csv(significance_path, significance_rows)
    lines = [
        "# C12-A Persistent Dynamics Analysis",
        "",
        f"**Status:** {'FINAL' if scale == 'full' else 'DEVELOPMENT ONLY'}",
        "",
        f"- Selected hierarchy (VALIDATION): `{selected_hierarchy}`",
        f"- Selected flat comparator (VALIDATION): `{best_flat}`",
        f"- G0-A: {'PASS' if g0_passed else 'FAIL'}",
        f"- G1-A forecast: {'PASS' if selected_g1['passed'] else 'FAIL'}",
        f"- G2-A planning: {'PASS' if selected_g2['passed'] else 'FAIL'}",
        f"- G3-A carry: {'PASS' if selected_g3['passed'] else 'FAIL'}",
        f"- G4-A closure: `{closure['verdict']}`",
        "",
        closure["interpretation"],
        "",
        "Gate booleans use only preregistered strata and world-clustered primary comparisons; exploratory slices are descriptive only.",
        "",
    ]
    _write_text(report_path, "\n".join(lines))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("probe", "collect", "inspect-dataset", "train", "forecast-eval", "plan-eval", "analyze", "full"),
    )
    parser.add_argument("--scale", choices=tuple(SCALE_PAIRS_PER_STRATUM), default="full")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "c12_persistent",
    )
    parser.add_argument("--pairs-per-stratum", type=int, default=None)
    parser.add_argument("--episodes-per-shard", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--roadmap-nodes", type=int, default=96)
    parser.add_argument("--roadmap-k", type=int, default=7)
    parser.add_argument("--raster-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--probe-dir", type=Path, default=None)
    parser.add_argument("--allow-failed-probe", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = L.C12DynamicsConfig(
        roadmap_nodes=args.roadmap_nodes,
        roadmap_k=args.roadmap_k,
        raster_size=args.raster_size,
    )
    n_pairs = (
        SCALE_PAIRS_PER_STRATUM[args.scale]
        if args.pairs_per_stratum is None
        else int(args.pairs_per_stratum)
    )
    expected_hash = probe_config_hash(cfg, args.scale, n_pairs, MAP_FAMILIES)
    if args.mode == "probe":
        summary = run_probe(
            args.out_dir,
            scale=args.scale,
            cfg=cfg,
            pairs_per_stratum=n_pairs,
            bootstrap_samples=args.bootstrap_samples,
        )
        return 0 if summary["gates"]["passed"] else 2
    if args.mode == "inspect-dataset":
        print(json.dumps(inspect_dataset(args.out_dir), indent=2, sort_keys=True))
        return 0
    if args.mode == "collect":
        manifest = collect_dataset(
            args.out_dir,
            scale=args.scale,
            cfg=cfg,
            episodes_per_shard=args.episodes_per_shard,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.mode == "train":
        summary = run_training(
            args.out_dir,
            scale=args.scale,
            cfg=cfg,
            device=args.device,
            probe_dir=args.probe_dir,
            allow_failed_probe=args.allow_failed_probe,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "complete" else 2
    if args.mode == "forecast-eval":
        summary = run_forecast_evaluation(
            args.out_dir,
            scale=args.scale,
            device=args.device,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.mode == "plan-eval":
        summary = run_planning_evaluation(
            args.out_dir,
            scale=args.scale,
            device=args.device,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.mode == "analyze":
        summary = run_analysis(
            args.out_dir,
            scale=args.scale,
            probe_dir=args.probe_dir,
            bootstrap_samples=args.bootstrap_samples,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.mode == "full":
        summary = run_full_pipeline(
            args.out_dir,
            scale=args.scale,
            cfg=cfg,
            device=args.device,
            probe_dir=args.probe_dir,
            allow_failed_probe=args.allow_failed_probe,
            bootstrap_samples=args.bootstrap_samples,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "complete" else 2

    authorization = assert_probe_authorized(
        args.out_dir, expected_hash, allow_failed_probe=args.allow_failed_probe
    )
    label = "exploratory" if authorization["exploratory_override"] else "authorized"
    raise RuntimeError(
        f"C12-A {args.mode} is {label} by G0-A, but learned-model Tasks 5+ "
        "are intentionally not started by the probe-stage implementation"
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROVIDER_NAMES",
    "MAP_FAMILIES",
    "DATASET_SCHEMA_VERSION",
    "DATASET_SCALE_COUNTS",
    "FrozenFrameProvider",
    "ConstantVelocityProvider",
    "TrueModeProvider",
    "OracleFutureProvider",
    "LearnedForecastProvider",
    "probe_config_hash",
    "evaluate_g0_gates",
    "run_probe",
    "collect_dataset",
    "inspect_dataset",
    "forecast_bucket_metrics",
    "linear_probe_accuracy",
    "seeded_world_bootstrap",
    "paired_sign_flip_p",
    "bh_q_values",
    "evaluate_g1_gate",
    "evaluate_g2_gate",
    "evaluate_g3_gate",
    "evaluate_g4_closure",
    "run_training",
    "run_forecast_evaluation",
    "run_latent_regime_probe",
    "run_planning_evaluation",
    "freeze_flat_comparator",
    "run_full_pipeline",
    "run_analysis",
    "assert_probe_authorized",
    "main",
]
