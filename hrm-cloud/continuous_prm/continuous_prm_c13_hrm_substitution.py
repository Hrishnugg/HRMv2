#!/usr/bin/env python3
"""C13-N architecture-only HRM substitution for the C13-M method.

The experiment reuses C13-J's exact cohorts and feature caches, trains the
repository HRM ranker under the same LHBL schedule as the frozen flat MLP, and
evaluates both through one identical local Bellman backup.  A new confirmation
cohort is generated only when the preregistered development gate passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import continuous_prm_common as C
import continuous_prm_c13_certified_search as S
import continuous_prm_c13_identifiability as I
import continuous_prm_c13_lhbl_c7_comparison as X
import continuous_prm_c13_lhbl_generated_v3 as H
import continuous_prm_c13_lhbl_multisuite as J
import continuous_prm_c13_shared_queue as Q
import continuous_prm_c13_state_heuristic as C13


HRM_FAMILY = "hrm_trimmed"
FLAT_FAMILY = "flat_mlp"
REFERENCE_ARMS = ("euclid", "field_hrm", "scalar_hrm")
CURRENT_BOUNDARY = (
    "current_goal_geometry_bounded_rays_one_hop_actions_plus_"
    "radius_bounded_local_subgraph_and_frozen_exit_values"
)
PREREGISTRATION = (
    "../../docs/experiments/continuous/c13/design/"
    "2026-07-17-c13n-hrm-substitution.md"
)


@dataclass
class HrmSubstitutionConfig:
    multisuite_run_dir: str = "runs/c13_lhbl_multisuite"
    original_run_dir: str = "runs/c13_lhbl_flat_48w"
    study_dir: str = "runs/c13_identifiability"
    c7_run_dir: str = "runs/c7_local"
    c13i_run_dir: str = "runs/c13_lhbl_c7_comparison"
    c13l_run_dir: str = "runs/c13_local_backup_scale"
    c13m_run_dir: str = "runs/c13_matched_quality_confirmation"
    out_dir: str = "runs/c13_hrm_substitution"
    preregistration: str = PREREGISTRATION
    mode: str = "full"
    candidate_iterations: str = "4,6,8"
    alphas: str = "1.00,1.50"
    primary_iteration: int = 8
    primary_alpha: float = 1.50
    sensor_radius_frac: float = 0.20
    required_negative_suites: int = 4
    mean_cost_margin: float = 0.005
    max_cost_margin: float = 0.020
    bootstrap_replicates: int = 20_000
    bootstrap_seed: int = 1_113_337
    confirmation_worlds_per_suite: int = 24
    confirmation_seed_offset: int = 20_000_000
    train_device: str = "cuda"
    evaluation_device: str = "cpu"


def resolve_paths(cfg: HrmSubstitutionConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    for field_name in (
        "multisuite_run_dir",
        "original_run_dir",
        "study_dir",
        "c7_run_dir",
        "c13i_run_dir",
        "c13l_run_dir",
        "c13m_run_dir",
        "out_dir",
        "preregistration",
    ):
        path = Path(getattr(cfg, field_name))
        if not path.is_absolute():
            setattr(cfg, field_name, str((script_dir / path).resolve()))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _parse_iterations(value: str) -> List[int]:
    result = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)) or any(item <= 0 for item in result):
        raise ValueError("candidate_iterations must be unique positive integers")
    return sorted(result)


def _arm_name(family: str, iteration: int, alpha: float) -> str:
    return f"{family}_i{int(iteration):02d}_a{float(alpha):.2f}"


def _checkpoint_path(run_dir: str, family: str, iteration: int) -> Path:
    return (
        Path(run_dir)
        / "checkpoints"
        / f"{family}_iteration_{int(iteration):02d}.pt"
    )


def _load_source_configs(
    cfg: HrmSubstitutionConfig,
) -> Tuple[J.MultiSuiteConfig, H.LHBLConfig, Dict[str, Any]]:
    manifest_path = Path(cfg.multisuite_run_dir) / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("C13-J manifest is required")
    manifest = H._read_json(manifest_path)
    multi_cfg = J.MultiSuiteConfig(**manifest["config"])
    source_train_cfg = H.LHBLConfig(**manifest["lhbl_config"])
    locked = {
        "train_worlds_per_suite": 32,
        "validation_worlds_per_suite": 8,
        "development_worlds_per_suite": 4,
        "roadmap_nodes": 192,
        "roadmap_k": 7,
        "cohort_seed": 17_413,
        "train_seed_offset": 0,
        "validation_seed_offset": 5_000_000,
        "development_seed_offset": 10_000_000,
        "model_seed": 17_413,
        "outer_iterations": 8,
        "inner_epochs": 5,
        "batch_size": 128,
        "hidden_dim": 64,
        "lr": 5.0e-4,
        "weight_decay": 1.0e-4,
        "sensor_radius_frac": 0.20,
        "num_rays": 32,
        "ray_steps": 32,
        "max_neighbors": 24,
    }
    changed = {
        name: (getattr(multi_cfg, name), expected)
        for name, expected in locked.items()
        if getattr(multi_cfg, name) != expected
    }
    if changed:
        raise RuntimeError(f"C13-J frozen configuration drifted: {changed}")
    if source_train_cfg.models != FLAT_FAMILY:
        raise RuntimeError("C13-J source model is no longer flat_mlp")
    if source_train_cfg.seed != multi_cfg.model_seed:
        raise RuntimeError("C13-J source model seed changed")
    if Path(multi_cfg.out_dir).resolve() != Path(cfg.multisuite_run_dir).resolve():
        raise RuntimeError("C13-J manifest points at a different run directory")
    return multi_cfg, source_train_cfg, manifest


def _verify_frozen_integrity(cfg: HrmSubstitutionConfig) -> Dict[str, Any]:
    path = Path(cfg.multisuite_run_dir) / "integrity.json"
    if not path.exists():
        raise RuntimeError("C13-J integrity manifest is required")
    payload = H._read_json(path)
    checked = 0
    mismatches: List[str] = []
    for section in ("inputs", "outputs"):
        for name, record in payload[section].items():
            artifact = Path(record["path"])
            if not artifact.exists():
                mismatches.append(f"{section}/{name}:missing")
                continue
            observed = S.file_sha256(artifact)
            if observed != record["sha256"]:
                mismatches.append(f"{section}/{name}:sha256")
            checked += 1
    if mismatches:
        raise RuntimeError(f"C13-J frozen integrity mismatch: {mismatches}")
    return {
        "integrity_path": str(path),
        "integrity_sha256": S.file_sha256(path),
        "artifacts_checked": checked,
        "mismatches": mismatches,
    }


REPLAY_KEYS = (
    "split",
    "suite",
    "suite_world_index",
    "global_world_index",
    "world_seed",
    "roadmap_seed",
    "nodes",
    "edges",
    "cache",
    "cache_sha256",
)


def _verify_replay_records(
    split: str,
    observed: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
) -> None:
    left = [{key: row[key] for key in REPLAY_KEYS} for row in observed]
    right = [{key: row[key] for key in REPLAY_KEYS} for row in expected]
    if left != right:
        raise RuntimeError(f"C13-J {split} cohort or feature-cache replay changed")
    if any(str(row["cache_status"]) != "reused" for row in observed):
        raise RuntimeError(f"C13-J {split} replay attempted to create a feature cache")


def audit_and_rebuild_source(
    cfg: HrmSubstitutionConfig,
) -> Tuple[
    J.MultiSuiteConfig,
    H.LHBLConfig,
    I.StudyConfig,
    List[H.WorldBundle],
    List[H.WorldBundle],
    List[H.WorldBundle],
    Dict[str, Any],
]:
    multi_cfg, source_train_cfg, source_manifest = _load_source_configs(cfg)
    frozen_integrity = _verify_frozen_integrity(cfg)
    saved = H._read_json(
        Path(cfg.multisuite_run_dir) / "results" / "cohorts.json"
    )
    for split, rows in saved["records"].items():
        for row in rows:
            cache = Path(row["cache"])
            if not cache.exists() or S.file_sha256(cache) != row["cache_sha256"]:
                raise RuntimeError(f"missing or changed C13-J feature cache: {cache}")

    train, train_records, _ = J.build_balanced_bundles(
        multi_cfg,
        "train",
        J.TRAIN_SUITES,
        multi_cfg.train_worlds_per_suite,
        multi_cfg.train_seed_offset,
    )
    validation, validation_records, _ = J.build_balanced_bundles(
        multi_cfg,
        "validation",
        J.TRAIN_SUITES,
        multi_cfg.validation_worlds_per_suite,
        multi_cfg.validation_seed_offset,
    )
    development, development_records, _ = J.build_balanced_bundles(
        multi_cfg,
        "development",
        J.DEV_SUITES,
        multi_cfg.development_worlds_per_suite,
        multi_cfg.development_seed_offset,
    )
    _verify_replay_records("train", train_records, saved["records"]["train"])
    _verify_replay_records(
        "validation", validation_records, saved["records"]["validation"]
    )
    _verify_replay_records(
        "development", development_records, saved["records"]["development"]
    )
    cohort_verification = J.verify_cohorts(
        multi_cfg, train, validation, development
    )
    study_cfg, study_manifest = S.load_study(cfg.study_dir)
    audit = {
        "frozen_integrity": frozen_integrity,
        "source_manifest_sha256": S.file_sha256(
            Path(cfg.multisuite_run_dir) / "manifest.json"
        ),
        "source_cohorts_sha256": S.file_sha256(
            Path(cfg.multisuite_run_dir) / "results" / "cohorts.json"
        ),
        "cohort_verification": cohort_verification,
        "replayed_records": {
            "train": len(train_records),
            "validation": len(validation_records),
            "development": len(development_records),
        },
        "feature_caches_reused": len(train_records)
        + len(validation_records)
        + len(development_records),
        "source_study_manifest": study_manifest,
        "source_c13j_manifest": source_manifest,
    }
    return (
        multi_cfg,
        source_train_cfg,
        study_cfg,
        train,
        validation,
        development,
        audit,
    )


def _hrm_training_config(
    cfg: HrmSubstitutionConfig, source: H.LHBLConfig
) -> H.LHBLConfig:
    result = H.LHBLConfig(**asdict(source))
    result.out_dir = cfg.out_dir
    result.study_dir = cfg.study_dir
    result.models = HRM_FAMILY
    result.device = cfg.train_device
    return result


def _training_fingerprint(
    cfg: HrmSubstitutionConfig,
    train_cfg: H.LHBLConfig,
    model_cfg: I.StudyConfig,
) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "experiment": "C13-N HRM substitution training",
        "training_config": asdict(train_cfg),
        "model_config": asdict(model_cfg),
        "source_cohorts_sha256": S.file_sha256(
            Path(cfg.multisuite_run_dir) / "results" / "cohorts.json"
        ),
        "source_integrity_sha256": S.file_sha256(
            Path(cfg.multisuite_run_dir) / "integrity.json"
        ),
        "preregistration_sha256": S.file_sha256(Path(cfg.preregistration)),
        "trainer_sha256": S.file_sha256(Path(H.__file__).resolve()),
        "model_definition_sha256": S.file_sha256(Path(I.__file__).resolve()),
        "implementation_sha256": S.file_sha256(Path(__file__).resolve()),
    }
    return _json_hash(payload), payload


def train_hrm(
    cfg: HrmSubstitutionConfig,
    source_train_cfg: H.LHBLConfig,
    study_cfg: I.StudyConfig,
    train_bundles: Sequence[H.WorldBundle],
    validation_bundles: Sequence[H.WorldBundle],
) -> Tuple[List[Path], List[Dict[str, Any]], Dict[str, Any]]:
    results_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    checkpoint_dir = C13.ensure_dir(Path(cfg.out_dir) / "checkpoints")
    history_path = results_dir / "training_history.csv"
    state_path = results_dir / "training_state.json"
    train_cfg = _hrm_training_config(cfg, source_train_cfg)
    model_cfg = H.model_config(train_cfg, study_cfg)
    fingerprint, fingerprint_payload = _training_fingerprint(
        cfg, train_cfg, model_cfg
    )
    expected_paths = [
        _checkpoint_path(cfg.out_dir, HRM_FAMILY, iteration)
        for iteration in range(1, int(train_cfg.outer_iterations) + 1)
    ]

    completed = 0
    histories: List[Dict[str, Any]] = []
    if state_path.exists():
        state = H._read_json(state_path)
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError("existing C13-N training fingerprint differs")
        completed = int(state.get("completed_iteration", 0))
        histories = list(_read_csv(history_path))
        if len(histories) != completed:
            raise RuntimeError("C13-N training state/history length mismatch")
        for iteration in range(1, completed + 1):
            if not expected_paths[iteration - 1].exists():
                raise RuntimeError("C13-N completed training checkpoint is missing")
        if state.get("status") == "complete":
            if completed != int(train_cfg.outer_iterations):
                raise RuntimeError("C13-N complete marker has the wrong iteration")
            hashes = state.get("checkpoint_sha256", {})
            for path in expected_paths:
                if hashes.get(path.name) != S.file_sha256(path):
                    raise RuntimeError(f"C13-N checkpoint hash changed: {path}")
            return expected_paths, histories, state
    else:
        existing = list(checkpoint_dir.glob(f"{HRM_FAMILY}_iteration_*.pt"))
        if existing:
            raise RuntimeError(
                "unfingerprinted C13-N checkpoints exist; use a new output directory"
            )
        C13.write_json(
            state_path,
            {
                "status": "running",
                "completed_iteration": 0,
                "fingerprint": fingerprint,
                "fingerprint_payload": fingerprint_payload,
            },
        )

    device = H.resolve_device(cfg.train_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was preregistered but CUDA is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    seed = int(train_cfg.seed) + 1009
    C.set_global_seed(seed)
    model = H.build_lhbl_model(HRM_FAMILY, model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.lr),
        weight_decay=float(train_cfg.weight_decay),
    )
    if completed:
        resume = torch.load(
            expected_paths[completed - 1],
            map_location=device,
            weights_only=False,
        )
        if resume.get("c13n_training_fingerprint") != fingerprint:
            raise RuntimeError("C13-N resume checkpoint fingerprint differs")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])

    for iteration in range(completed + 1, int(train_cfg.outer_iterations) + 1):
        target_model = None if iteration == 1 else model
        x_train, y_train, train_stats = H.bootstrap_and_targets(
            train_bundles, target_model, train_cfg, device
        )
        x_val, y_val, val_stats = H.bootstrap_and_targets(
            validation_bundles, target_model, train_cfg, device
        )
        dataset = TensorDataset(
            torch.from_numpy(x_train), torch.from_numpy(y_train)
        )
        generator = torch.Generator().manual_seed(seed + iteration)
        loader = DataLoader(
            dataset,
            batch_size=int(train_cfg.batch_size),
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        losses: List[float] = []
        started = time.perf_counter()
        model.train()
        for _ in range(int(train_cfg.inner_epochs)):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(xb)
                loss = F.smooth_l1_loss(prediction, yb)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"nonfinite C13-N loss at iteration {iteration}"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_cfg.grad_clip)
                )
                optimizer.step()
                losses.append(float(loss.item()))
        model.eval()
        val_prediction = I.predict_model(model, x_val, device)
        val_error = val_prediction - y_val.astype(np.float64)
        history = {
            "model": HRM_FAMILY,
            "iteration": int(iteration),
            "inner_epochs": int(train_cfg.inner_epochs),
            "train_loss_mean": float(np.mean(losses)),
            "val_mae": float(np.mean(np.abs(val_error))),
            "val_rmse": float(np.sqrt(np.mean(val_error * val_error))),
            "seconds": float(time.perf_counter() - started),
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"val_{key}": value for key, value in val_stats.items()},
        }
        checkpoint = expected_paths[iteration - 1]
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_name": HRM_FAMILY,
                "iteration": int(iteration),
                "lhbl_config": asdict(train_cfg),
                "model_config": asdict(model_cfg),
                "target": "radius_bounded_local_paths_plus_frozen_frontier_value",
                "shortest_path_target": False,
                "c13n_training_fingerprint": fingerprint,
            },
            checkpoint,
        )
        histories.append(history)
        C13.write_csv(history_path, histories)
        C13.write_json(
            state_path,
            {
                "status": (
                    "complete"
                    if iteration == int(train_cfg.outer_iterations)
                    else "running"
                ),
                "completed_iteration": int(iteration),
                "fingerprint": fingerprint,
                "fingerprint_payload": fingerprint_payload,
                "checkpoint_sha256": {
                    path.name: S.file_sha256(path)
                    for path in expected_paths[:iteration]
                },
            },
        )
        print(
            f"[c13n] HRM training iteration={iteration}/"
            f"{train_cfg.outer_iterations} loss={history['train_loss_mean']:.5f} "
            f"val_mae={history['val_mae']:.5f} "
            f"seconds={history['seconds']:.1f}",
            flush=True,
        )
    state = H._read_json(state_path)
    return expected_paths, histories, state


def load_model(
    path: Path, expected_family: str, device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"missing model checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_name") != expected_family:
        raise RuntimeError(f"unexpected model family in checkpoint: {path}")
    if payload.get("shortest_path_target") is not False:
        raise RuntimeError(f"checkpoint target provenance changed: {path}")
    model_cfg = I.StudyConfig(**payload["model_config"])
    model = H.build_lhbl_model(expected_family, model_cfg)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload


def load_candidate_models(
    cfg: HrmSubstitutionConfig,
    iterations: Sequence[int],
    device: torch.device,
) -> Tuple[Dict[Tuple[str, int], torch.nn.Module], Dict[str, Path]]:
    models: Dict[Tuple[str, int], torch.nn.Module] = {}
    paths: Dict[str, Path] = {}
    for family, run_dir in (
        (HRM_FAMILY, cfg.out_dir),
        (FLAT_FAMILY, cfg.multisuite_run_dir),
    ):
        for iteration in iterations:
            path = _checkpoint_path(run_dir, family, iteration)
            model, payload = load_model(path, family, device)
            if int(payload["iteration"]) != int(iteration):
                raise RuntimeError(f"checkpoint iteration mismatch: {path}")
            models[(family, int(iteration))] = model
            paths[f"{family}_iteration_{int(iteration):02d}"] = path
    return models, paths


def _result_row(
    phase: str,
    bundle: H.WorldBundle,
    record: Mapping[str, Any],
    arm: str,
    family: str,
    iteration: int | str,
    alpha: float | str,
    boundary: str,
    result: Mapping[str, Any],
    optimal: float,
    representation_seconds: float,
    model_seconds: float,
    local_backup_seconds: float,
    search_seconds: float,
) -> Dict[str, Any]:
    found = bool(result["found"])
    if found:
        validation = Q.validate_path(
            bundle.roadmap.adj, result["path"], result["cost"]
        )
        path_valid = bool(validation["valid"])
        cost = float(result["cost"])
        ratio = float(cost / optimal)
    else:
        path_valid = False
        cost = float("nan")
        ratio = float("nan")
    return {
        "phase": phase,
        "suite": bundle.suite,
        "suite_world_index": int(record["suite_world_index"]),
        "world_index": int(bundle.world_index),
        "world_seed": int(bundle.world_seed),
        "roadmap_seed": int(record["roadmap_seed"]),
        "arm": arm,
        "family": family,
        "iteration": iteration,
        "alpha": alpha,
        "runtime_information_boundary": boundary,
        "found": found,
        "path_valid": path_valid,
        "expansions": int(result["expansions"]),
        "cost": cost,
        "optimal": float(optimal),
        "cost_ratio_eval_only": ratio,
        "representation_seconds": float(representation_seconds),
        "model_seconds": float(model_seconds),
        "local_backup_seconds": float(local_backup_seconds),
        "search_seconds": float(search_seconds),
    }


def _diagnostic_row(
    phase: str,
    bundle: H.WorldBundle,
    family: str,
    iteration: int,
    learned: np.ndarray,
    local_values: np.ndarray,
    prediction: np.ndarray,
    representation_seconds: float,
    model_seconds: float,
    backup_seconds: float,
    local_diagnostics: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    oracle = np.asarray(bundle.roadmap.dist_to_goal, dtype=np.float64)
    connected = np.isfinite(oracle) & (oracle < C.INF / 10.0)
    return {
        "phase": phase,
        "suite": bundle.suite,
        "world_index": int(bundle.world_index),
        "world_seed": int(bundle.world_seed),
        "family": family,
        "iteration": int(iteration),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction)),
        "static_spearman_eval_only": I.safe_spearman(
            learned[connected], oracle[connected]
        ),
        "local_backup_spearman_eval_only": I.safe_spearman(
            local_values[connected], oracle[connected]
        ),
        "representation_seconds": float(representation_seconds),
        "model_seconds": float(model_seconds),
        "local_backup_seconds": float(backup_seconds),
        "fallback_nodes": int(
            sum(bool(row["fallback"]) for row in local_diagnostics)
        ),
        "observed_nodes_mean": float(
            np.mean([float(row["observed_nodes"]) for row in local_diagnostics])
        ),
        "exit_actions_mean": float(
            np.mean([float(row["exit_actions"]) for row in local_diagnostics])
        ),
    }


def _expected_eval_arms(
    iterations: Sequence[int], alphas: Sequence[float]
) -> set[str]:
    result = set(REFERENCE_ARMS)
    for family in (HRM_FAMILY, FLAT_FAMILY):
        for iteration in iterations:
            for alpha in alphas:
                result.add(_arm_name(family, iteration, alpha))
    return result


def _clean_and_find_completed(
    rows: List[Dict[str, Any]],
    diagnostics: List[Dict[str, Any]],
    expected_arms: set[str],
    iterations: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], set[int]]:
    row_groups: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    diag_groups: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_groups[int(row["world_seed"])].append(row)
    for row in diagnostics:
        diag_groups[int(row["world_seed"])].append(row)
    expected_diag = {
        (family, int(iteration))
        for family in (HRM_FAMILY, FLAT_FAMILY)
        for iteration in iterations
    }
    completed: set[int] = set()
    partial: set[int] = set()
    for seed, group in row_groups.items():
        arms = [str(row["arm"]) for row in group]
        diag = {
            (str(row["family"]), int(row["iteration"]))
            for row in diag_groups.get(seed, [])
        }
        if len(arms) == len(set(arms)) and set(arms) == expected_arms and diag == expected_diag:
            completed.add(seed)
        else:
            partial.add(seed)
    partial.update(set(diag_groups) - completed)
    if partial:
        rows = [row for row in rows if int(row["world_seed"]) not in partial]
        diagnostics = [
            row for row in diagnostics if int(row["world_seed"]) not in partial
        ]
    return rows, diagnostics, completed


def evaluate_phase(
    cfg: HrmSubstitutionConfig,
    phase: str,
    bundles: Sequence[H.WorldBundle],
    records: Sequence[Mapping[str, Any]],
    multi_cfg: J.MultiSuiteConfig,
    iterations: Sequence[int],
    alphas: Sequence[float],
    raw_path: Path,
    diagnostics_path: Path,
    meta_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Path]]:
    device = H.resolve_device(cfg.evaluation_device)
    if str(device) != "cpu":
        raise RuntimeError("C13-N search evaluation is locked to CPU")
    models, model_paths = load_candidate_models(cfg, iterations, device)
    comparators = J._load_c7_comparators(multi_cfg, device)
    checkpoint_hashes = {
        name: S.file_sha256(path) for name, path in model_paths.items()
    }
    fingerprint_payload = {
        "phase": phase,
        "config": asdict(cfg),
        "iterations": list(iterations),
        "alphas": list(alphas),
        "checkpoint_sha256": checkpoint_hashes,
        "preregistration_sha256": S.file_sha256(Path(cfg.preregistration)),
        "implementation_sha256": S.file_sha256(Path(__file__).resolve()),
    }
    fingerprint = _json_hash(fingerprint_payload)
    if meta_path.exists():
        previous = H._read_json(meta_path)
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError(f"existing C13-N {phase} fingerprint differs")
    else:
        C13.write_json(
            meta_path,
            {
                "fingerprint": fingerprint,
                "fingerprint_payload": fingerprint_payload,
            },
        )
    rows: List[Dict[str, Any]] = list(_read_csv(raw_path))
    diagnostics: List[Dict[str, Any]] = list(_read_csv(diagnostics_path))
    expected_arms = _expected_eval_arms(iterations, alphas)
    rows, diagnostics, completed = _clean_and_find_completed(
        rows, diagnostics, expected_arms, iterations
    )
    record_by_seed = {int(row["world_seed"]): row for row in records}
    feature_mismatches = 0

    for ordinal, bundle in enumerate(bundles, start=1):
        if int(bundle.world_seed) in completed:
            continue
        record = record_by_seed[int(bundle.world_seed)]
        roadmap = bundle.roadmap
        optimal = float(roadmap.dist_to_goal[0])
        if not math.isfinite(optimal) or optimal >= C.INF / 10.0:
            raise RuntimeError(f"{phase} accepted a disconnected roadmap")
        world_rows: List[Dict[str, Any]] = []
        world_diagnostics: List[Dict[str, Any]] = []

        representation_started = time.perf_counter()
        live_features = C13.make_local_state_features(
            bundle.world,
            roadmap.points,
            roadmap.adj,
            J.local_config(multi_cfg),
        )
        representation_seconds = float(
            time.perf_counter() - representation_started
        )
        if not np.array_equal(live_features, bundle.features):
            feature_mismatches += 1
            raise RuntimeError(f"{phase} live/cache feature mismatch")
        euclid = C13.euclidean_to_goal(roadmap.points, roadmap.points[1])

        reference_ranks: Dict[str, np.ndarray] = {"euclid": euclid}
        reference_times: Dict[str, float] = {"euclid": 0.0}
        for name, provider in comparators.items():
            started = time.perf_counter()
            reference_ranks[name] = np.asarray(
                provider.node_h(bundle.world, roadmap, 1), dtype=np.float64
            )
            reference_times[name] = float(time.perf_counter() - started)
        for name in REFERENCE_ARMS:
            started = time.perf_counter()
            result = X.astar_with_path(
                roadmap.adj, reference_ranks[name], len(roadmap.points)
            )
            search_seconds = float(time.perf_counter() - started)
            world_rows.append(
                _result_row(
                    phase,
                    bundle,
                    record,
                    name,
                    name,
                    "",
                    "",
                    (
                        "current_goal_geometry"
                        if name == "euclid"
                        else (
                            "complete_64x64_occupancy_goal_raster"
                            if name == "field_hrm"
                            else "global_obstacle_summaries_sectors_rays_goal"
                        )
                    ),
                    result,
                    optimal,
                    reference_times[name],
                    0.0,
                    0.0,
                    search_seconds,
                )
            )

        for family in (HRM_FAMILY, FLAT_FAMILY):
            for iteration in iterations:
                model = models[(family, int(iteration))]
                started = time.perf_counter()
                prediction = np.asarray(
                    I.predict_model(model, live_features, device),
                    dtype=np.float64,
                )
                model_seconds = float(time.perf_counter() - started)
                learned = (
                    euclid + float(bundle.world.side_len) * prediction
                )
                started = time.perf_counter()
                local_values, local_diagnostics = H.limited_horizon_values(
                    roadmap.points,
                    roadmap.adj,
                    roadmap.points[1],
                    learned,
                    float(cfg.sensor_radius_frac)
                    * float(bundle.world.side_len),
                )
                backup_seconds = float(time.perf_counter() - started)
                world_diagnostics.append(
                    _diagnostic_row(
                        phase,
                        bundle,
                        family,
                        int(iteration),
                        learned,
                        local_values,
                        prediction,
                        representation_seconds,
                        model_seconds,
                        backup_seconds,
                        local_diagnostics,
                    )
                )
                for alpha in alphas:
                    rank = euclid + float(alpha) * (local_values - euclid)
                    started = time.perf_counter()
                    result = X.astar_with_path(
                        roadmap.adj, rank, len(roadmap.points)
                    )
                    search_seconds = float(time.perf_counter() - started)
                    world_rows.append(
                        _result_row(
                            phase,
                            bundle,
                            record,
                            _arm_name(family, int(iteration), float(alpha)),
                            family,
                            int(iteration),
                            float(alpha),
                            CURRENT_BOUNDARY,
                            result,
                            optimal,
                            representation_seconds,
                            model_seconds,
                            backup_seconds,
                            search_seconds,
                        )
                    )
        observed_arms = {str(row["arm"]) for row in world_rows}
        if observed_arms != expected_arms:
            raise RuntimeError(f"{phase} arm set is incomplete")
        if any(
            not bool(row["found"]) or not bool(row["path_valid"])
            for row in world_rows
        ):
            raise RuntimeError(f"{phase} search returned an invalid path")
        rows.extend(world_rows)
        diagnostics.extend(world_diagnostics)
        C13.write_csv(raw_path, rows)
        C13.write_csv(diagnostics_path, diagnostics)
        print(
            f"[c13n] {phase} {ordinal}/{len(bundles)} "
            f"{bundle.suite}/{record['suite_world_index']} "
            f"seed={bundle.world_seed}",
            flush=True,
        )
    provenance = {
        "phase": phase,
        "fingerprint": fingerprint,
        "feature_cache_mismatches": feature_mismatches,
        "worlds": len(bundles),
        "rows": len(rows),
        "diagnostics": len(diagnostics),
        "expected_arms_per_world": len(expected_arms),
        "checkpoint_sha256": checkpoint_hashes,
    }
    return rows, diagnostics, provenance, model_paths


def _valid_row(row: Mapping[str, Any]) -> bool:
    return _as_bool(row["found"]) and _as_bool(row["path_valid"])


def _paired(
    rows: Sequence[Mapping[str, Any]],
    left_arm: str,
    right_arm: str,
    scope: str = "POOLED",
) -> Tuple[List[Tuple[Mapping[str, Any], Mapping[str, Any]]], np.ndarray]:
    scoped = [
        row
        for row in rows
        if scope == "POOLED" or str(row["suite"]) == scope
    ]
    lookup = {
        (int(row["world_seed"]), str(row["arm"])): row for row in scoped
    }
    seeds = sorted({int(row["world_seed"]) for row in scoped})
    pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for seed in seeds:
        left = lookup.get((seed, left_arm))
        right = lookup.get((seed, right_arm))
        if left is not None and right is not None and _valid_row(left) and _valid_row(right):
            pairs.append((left, right))
    delta = np.asarray(
        [
            float(left["expansions"]) - float(right["expansions"])
            for left, right in pairs
        ],
        dtype=np.float64,
    )
    return pairs, delta


def candidate_summary(
    cfg: HrmSubstitutionConfig,
    rows: Sequence[Mapping[str, Any]],
    iteration: int,
    alpha: float,
    expected_worlds: int,
) -> Dict[str, Any]:
    hrm_arm = _arm_name(HRM_FAMILY, iteration, alpha)
    flat_arm = _arm_name(FLAT_FAMILY, iteration, alpha)
    field_pairs, field_delta = _paired(rows, hrm_arm, "field_hrm")
    flat_pairs, flat_delta = _paired(rows, hrm_arm, flat_arm)
    if not len(field_delta) or not len(flat_delta):
        raise RuntimeError("C13-N candidate lacks paired rows")
    field_low, field_high = X._bootstrap_mean_ci(
        field_delta,
        cfg.bootstrap_replicates,
        cfg.bootstrap_seed + int(iteration) * 10_000 + int(alpha * 1000),
    )
    flat_low, flat_high = X._bootstrap_mean_ci(
        flat_delta,
        cfg.bootstrap_replicates,
        cfg.bootstrap_seed
        + 500_000
        + int(iteration) * 10_000
        + int(alpha * 1000),
    )
    suite_means: Dict[str, float] = {}
    for suite in J.DEV_SUITES:
        _, suite_delta = _paired(rows, hrm_arm, "field_hrm", suite)
        suite_means[suite] = float(np.mean(suite_delta))
    negative_suites = int(sum(value < 0.0 for value in suite_means.values()))

    hrm_cost = np.asarray(
        [float(left["cost_ratio_eval_only"]) for left, _ in field_pairs]
    )
    field_cost = np.asarray(
        [float(right["cost_ratio_eval_only"]) for _, right in field_pairs]
    )
    flat_hrm_cost = np.asarray(
        [float(left["cost_ratio_eval_only"]) for left, _ in flat_pairs]
    )
    flat_cost = np.asarray(
        [float(right["cost_ratio_eval_only"]) for _, right in flat_pairs]
    )
    validity = len(field_pairs) == expected_worlds and len(flat_pairs) == expected_worlds
    conditions = {
        "all_hrm_paths_valid": validity,
        "field_expansion_ci_upper_below_zero": float(field_high) < 0.0,
        "negative_suite_means_at_least_four": negative_suites
        >= int(cfg.required_negative_suites),
        "mean_cost_within_field_margin": float(np.mean(hrm_cost))
        <= float(np.mean(field_cost)) + float(cfg.mean_cost_margin) + 1.0e-12,
        "max_cost_within_field_margin": float(np.max(hrm_cost))
        <= float(np.max(field_cost)) + float(cfg.max_cost_margin) + 1.0e-12,
    }
    architecture_conditions = {
        "hrm_flat_expansion_ci_upper_below_zero": float(flat_high) < 0.0,
        "mean_cost_within_flat_margin": float(np.mean(flat_hrm_cost))
        <= float(np.mean(flat_cost)) + float(cfg.mean_cost_margin) + 1.0e-12,
        "max_cost_within_flat_margin": float(np.max(flat_hrm_cost))
        <= float(np.max(flat_cost)) + float(cfg.max_cost_margin) + 1.0e-12,
    }
    return {
        "iteration": int(iteration),
        "alpha": float(alpha),
        "worlds": int(len(field_pairs)),
        "hrm_expansions_mean": float(
            np.mean([float(left["expansions"]) for left, _ in field_pairs])
        ),
        "field_hrm_expansions_mean": float(
            np.mean([float(right["expansions"]) for _, right in field_pairs])
        ),
        "delta_vs_field_hrm_mean": float(np.mean(field_delta)),
        "delta_vs_field_hrm_ci95_low": float(field_low),
        "delta_vs_field_hrm_ci95_high": float(field_high),
        "field_wins": int(np.sum(field_delta < 0.0)),
        "field_ties": int(np.sum(field_delta == 0.0)),
        "field_losses": int(np.sum(field_delta > 0.0)),
        "negative_suites": negative_suites,
        "suite_delta_means": suite_means,
        "hrm_cost_ratio_mean": float(np.mean(hrm_cost)),
        "field_hrm_cost_ratio_mean": float(np.mean(field_cost)),
        "hrm_cost_ratio_max": float(np.max(hrm_cost)),
        "field_hrm_cost_ratio_max": float(np.max(field_cost)),
        "flat_expansions_mean": float(
            np.mean([float(right["expansions"]) for _, right in flat_pairs])
        ),
        "delta_vs_flat_mean": float(np.mean(flat_delta)),
        "delta_vs_flat_ci95_low": float(flat_low),
        "delta_vs_flat_ci95_high": float(flat_high),
        "flat_wins": int(np.sum(flat_delta < 0.0)),
        "flat_ties": int(np.sum(flat_delta == 0.0)),
        "flat_losses": int(np.sum(flat_delta > 0.0)),
        "flat_cost_ratio_mean": float(np.mean(flat_cost)),
        "flat_cost_ratio_max": float(np.max(flat_cost)),
        "field_gate_conditions": conditions,
        "field_gate_pass": bool(all(conditions.values())),
        "architecture_conditions": architecture_conditions,
        "architecture_win": bool(all(architecture_conditions.values())),
    }


def summarize_development(
    cfg: HrmSubstitutionConfig,
    rows: Sequence[Mapping[str, Any]],
    iterations: Sequence[int],
    alphas: Sequence[float],
    expected_worlds: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidates = [
        candidate_summary(cfg, rows, iteration, alpha, expected_worlds)
        for iteration in iterations
        for alpha in alphas
    ]
    primary_rows = [
        row
        for row in candidates
        if int(row["iteration"]) == int(cfg.primary_iteration)
        and math.isclose(float(row["alpha"]), float(cfg.primary_alpha))
    ]
    if len(primary_rows) != 1:
        raise RuntimeError("fixed primary C13-N cell is missing")
    passing = [row for row in candidates if bool(row["field_gate_pass"])]
    selected = (
        min(
            passing,
            key=lambda row: (
                float(row["hrm_expansions_mean"]),
                float(row["hrm_cost_ratio_mean"]),
                int(row["iteration"]),
                float(row["alpha"]),
            ),
        )
        if passing
        else None
    )
    verdict = {
        "verdict": (
            "hrm_substitution_development_pass_requires_fresh_confirmation"
            if selected is not None
            else "hrm_substitution_development_gate_failed"
        ),
        "gate_pass": selected is not None,
        "fixed_primary_cell": primary_rows[0],
        "passing_candidates": len(passing),
        "selected_candidate": selected,
        "authorization": (
            "run_selected_hrm_cell_on_seed_offset_20000000"
            if selected is not None
            else "stop_without_confirmation_or_retuning"
        ),
    }
    return candidates, verdict


def _seed_set(rows: Iterable[Mapping[str, Any]]) -> set[int]:
    return {int(row["world_seed"]) for row in rows}


def build_confirmation_cohort(
    cfg: HrmSubstitutionConfig,
    source_multi_cfg: J.MultiSuiteConfig,
) -> Tuple[List[H.WorldBundle], List[Dict[str, Any]], Dict[str, Any]]:
    multi_cfg = J.MultiSuiteConfig(**asdict(source_multi_cfg))
    multi_cfg.out_dir = cfg.out_dir
    multi_cfg.c7_run_dir = cfg.c7_run_dir
    multi_cfg.c13i_run_dir = cfg.c13i_run_dir
    bundles, records, _ = J.build_balanced_bundles(
        multi_cfg,
        "confirmation",
        J.DEV_SUITES,
        int(cfg.confirmation_worlds_per_suite),
        int(cfg.confirmation_seed_offset),
    )
    confirmation = _seed_set(records)
    prior: Dict[str, set[int]] = {}
    c13j = H._read_json(
        Path(cfg.multisuite_run_dir) / "results" / "cohorts.json"
    )
    for split, old_rows in c13j["records"].items():
        prior[f"c13j_{split}"] = _seed_set(old_rows)
    prior["c13i"] = J.c13i_seed_set(source_multi_cfg)

    optional_json = {
        "c13l_alpha_calibration": (
            Path(cfg.c13l_run_dir) / "results" / "calibration_cohort.json",
            "records",
        ),
        "c13m_confirmation": (
            Path(cfg.c13m_run_dir) / "results" / "confirmation_cohort.json",
            "records",
        ),
    }
    for name, (path, key) in optional_json.items():
        if path.exists():
            payload = H._read_json(path)
            prior[name] = _seed_set(payload[key])
    original_path = (
        Path(cfg.original_run_dir) / "results" / "lhbl_cohorts.json"
    )
    if original_path.exists():
        original = H._read_json(original_path)
        for split in ("train", "validation", "development_eval"):
            prior[f"original_{split}"] = _seed_set(original.get(split, []))

    overlaps = {
        name: len(confirmation & old_seeds)
        for name, old_seeds in prior.items()
    }
    expected = int(cfg.confirmation_worlds_per_suite) * len(J.DEV_SUITES)
    suite_counts = {
        suite: sum(str(row["suite"]) == suite for row in records)
        for suite in J.DEV_SUITES
    }
    verification = {
        "seed_offset": int(cfg.confirmation_seed_offset),
        "unique_seeds": len(confirmation),
        "expected_unique_seeds": expected,
        "suite_counts": suite_counts,
        "prior_overlap": overlaps,
    }
    if len(confirmation) != expected:
        raise RuntimeError("C13-N confirmation seed uniqueness failed")
    if any(
        count != int(cfg.confirmation_worlds_per_suite)
        for count in suite_counts.values()
    ):
        raise RuntimeError("C13-N confirmation suite balance failed")
    if any(overlaps.values()):
        raise RuntimeError(f"C13-N confirmation seed overlap: {overlaps}")
    return bundles, records, verification


def confirmation_verdict(
    cfg: HrmSubstitutionConfig,
    rows: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = int(cfg.confirmation_worlds_per_suite) * len(J.DEV_SUITES)
    result = candidate_summary(
        cfg,
        rows,
        int(selected["iteration"]),
        float(selected["alpha"]),
        expected,
    )
    return {
        "verdict": (
            "hrm_substitution_compatibility_confirmed"
            if result["field_gate_pass"]
            else "hrm_substitution_compatibility_not_confirmed"
        ),
        "gate_pass": bool(result["field_gate_pass"]),
        "candidate": result,
        "architecture_win_confirmed": bool(result["architecture_win"]),
        "authorization": (
            "document_confirmed_hrm_compatibility"
            if result["field_gate_pass"]
            else "document_confirmation_failure_without_retuning"
        ),
    }


def write_generated_report(
    path: Path,
    development: Mapping[str, Any],
    confirmation: Mapping[str, Any] | None,
) -> Path:
    primary = development["fixed_primary_cell"]
    selected = development["selected_candidate"]
    lines = [
        "# C13-N HRM substitution",
        "",
        f"Development verdict: {development['verdict']}.",
        "",
        "## Fixed C13-M substitution cell",
        "",
        (
            f"Iteration {primary['iteration']}, alpha {primary['alpha']:.2f}: "
            f"HRM {primary['hrm_expansions_mean']:.3f} expansions, "
            f"field HRM {primary['field_hrm_expansions_mean']:.3f}, paired delta "
            f"{primary['delta_vs_field_hrm_mean']:+.3f} with 95% CI "
            f"[{primary['delta_vs_field_hrm_ci95_low']:+.3f}, "
            f"{primary['delta_vs_field_hrm_ci95_high']:+.3f}]."
        ),
        (
            f"Against the matched flat MLP: delta "
            f"{primary['delta_vs_flat_mean']:+.3f}, 95% CI "
            f"[{primary['delta_vs_flat_ci95_low']:+.3f}, "
            f"{primary['delta_vs_flat_ci95_high']:+.3f}]."
        ),
        f"Field-comparator gate pass: {primary['field_gate_pass']}.",
        f"Architecture-win criterion: {primary['architecture_win']}.",
        "",
        "## Development selection",
        "",
    ]
    if selected is None:
        lines.extend(
            [
                "No preregistered HRM cell passed all five development gates.",
                "The untouched confirmation block was therefore not generated or evaluated.",
            ]
        )
    else:
        lines.extend(
            [
                (
                    f"Selected iteration {selected['iteration']}, alpha "
                    f"{selected['alpha']:.2f}; development delta versus field HRM "
                    f"{selected['delta_vs_field_hrm_mean']:+.3f}."
                ),
                "",
                "## Untouched confirmation",
                "",
            ]
        )
        if confirmation is None:
            lines.append("Confirmation was authorized but not run in this invocation.")
        else:
            candidate = confirmation["candidate"]
            lines.extend(
                [
                    f"Verdict: {confirmation['verdict']}.",
                    (
                        f"Across {candidate['worlds']} worlds, HRM averaged "
                        f"{candidate['hrm_expansions_mean']:.3f} expansions versus "
                        f"{candidate['field_hrm_expansions_mean']:.3f} for field HRM; "
                        f"paired delta {candidate['delta_vs_field_hrm_mean']:+.3f}, "
                        f"95% CI [{candidate['delta_vs_field_hrm_ci95_low']:+.3f}, "
                        f"{candidate['delta_vs_field_hrm_ci95_high']:+.3f}]."
                    ),
                    (
                        f"Matched HRM-minus-flat delta "
                        f"{candidate['delta_vs_flat_mean']:+.3f}, 95% CI "
                        f"[{candidate['delta_vs_flat_ci95_low']:+.3f}, "
                        f"{candidate['delta_vs_flat_ci95_high']:+.3f}]."
                    ),
                    (
                        f"HRM compatibility gate pass: "
                        f"{confirmation['gate_pass']}; architecture win confirmed: "
                        f"{confirmation['architecture_win_confirmed']}."
                    ),
                ]
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This study changes the learned model family, not the representation or search",
            "integration. It remains a known-PRM observation-simulator result with absolute",
            "coordinates, not a formal search bound, a wall-clock claim, or proof of general",
            "HRM superiority.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    return path


def run(cfg: HrmSubstitutionConfig) -> Dict[str, Any]:
    resolve_paths(cfg)
    if cfg.mode not in {"audit", "train", "develop", "full"}:
        raise ValueError("mode must be audit, train, develop, or full")
    iterations = _parse_iterations(cfg.candidate_iterations)
    alphas = C13.parse_float_csv(cfg.alphas)
    if cfg.primary_iteration not in iterations or not any(
        math.isclose(value, cfg.primary_alpha) for value in alphas
    ):
        raise RuntimeError("primary cell must be present in the candidate grid")
    results_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    (
        multi_cfg,
        source_train_cfg,
        study_cfg,
        train,
        validation,
        development,
        source_audit,
    ) = audit_and_rebuild_source(cfg)
    source_audit_path = C13.write_json(
        results_dir / "source_audit.json", source_audit
    )
    if cfg.mode == "audit":
        return {"source_audit": source_audit}

    checkpoints, histories, training_state = train_hrm(
        cfg,
        source_train_cfg,
        study_cfg,
        train,
        validation,
    )
    if cfg.mode == "train":
        return {
            "source_audit": source_audit,
            "training_state": training_state,
            "checkpoints": [str(path) for path in checkpoints],
        }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    saved = H._read_json(
        Path(cfg.multisuite_run_dir) / "results" / "cohorts.json"
    )
    dev_rows, dev_diagnostics, dev_provenance, model_paths = evaluate_phase(
        cfg,
        "development",
        development,
        saved["records"]["development"],
        multi_cfg,
        iterations,
        alphas,
        results_dir / "development_raw.csv",
        results_dir / "development_diagnostics.csv",
        results_dir / "development_meta.json",
    )
    candidates, development_verdict = summarize_development(
        cfg, dev_rows, iterations, alphas, len(development)
    )
    candidate_path = C13.write_csv(
        results_dir / "development_candidates.csv",
        [
            {
                **row,
                "suite_delta_means": json.dumps(
                    row["suite_delta_means"], sort_keys=True
                ),
                "field_gate_conditions": json.dumps(
                    row["field_gate_conditions"], sort_keys=True
                ),
                "architecture_conditions": json.dumps(
                    row["architecture_conditions"], sort_keys=True
                ),
            }
            for row in candidates
        ],
    )
    development_verdict_path = C13.write_json(
        results_dir / "development_verdict.json", development_verdict
    )

    confirmation_result: Dict[str, Any] | None = None
    confirmation_paths: Dict[str, Any] = {}
    if development_verdict["gate_pass"] and cfg.mode == "full":
        bundles, records, cohort_verification = build_confirmation_cohort(
            cfg, multi_cfg
        )
        cohort_path = C13.write_json(
            results_dir / "confirmation_cohort.json",
            {
                "records": records,
                "verification": cohort_verification,
            },
        )
        selected = development_verdict["selected_candidate"]
        confirmation_iterations = [int(selected["iteration"])]
        confirmation_alphas = [float(selected["alpha"])]
        confirmation_multi_cfg = J.MultiSuiteConfig(**asdict(multi_cfg))
        confirmation_multi_cfg.out_dir = cfg.out_dir
        confirmation_rows, confirmation_diagnostics, confirmation_provenance, _ = (
            evaluate_phase(
                cfg,
                "confirmation",
                bundles,
                records,
                confirmation_multi_cfg,
                confirmation_iterations,
                confirmation_alphas,
                results_dir / "confirmation_raw.csv",
                results_dir / "confirmation_diagnostics.csv",
                results_dir / "confirmation_meta.json",
            )
        )
        confirmation_result = confirmation_verdict(
            cfg, confirmation_rows, selected
        )
        confirmation_verdict_path = C13.write_json(
            results_dir / "confirmation_verdict.json", confirmation_result
        )
        confirmation_paths = {
            "cohort": cohort_path,
            "verdict": confirmation_verdict_path,
            "provenance": confirmation_provenance,
        }

    report_path = write_generated_report(
        results_dir / "C13N_RESULT.md",
        development_verdict,
        confirmation_result,
    )
    verification = {
        "source_audit_pass": True,
        "training_status": training_state["status"],
        "training_iterations": len(histories),
        "expected_training_iterations": int(source_train_cfg.outer_iterations),
        "development_worlds": len(development),
        "development_rows": len(dev_rows),
        "development_diagnostics": len(dev_diagnostics),
        "development_provenance": dev_provenance,
        "development_gate_pass": bool(development_verdict["gate_pass"]),
        "confirmation_run": confirmation_result is not None,
        "confirmation_gate_pass": (
            bool(confirmation_result["gate_pass"])
            if confirmation_result is not None
            else None
        ),
        "full_map_runtime_input": False,
        "shortest_path_target": False,
        "training_target_reads_dist_to_goal": False,
        "training_device": cfg.train_device,
        "evaluation_device": cfg.evaluation_device,
    }
    verification["integrity_pass"] = bool(
        verification["training_status"] == "complete"
        and verification["training_iterations"]
        == verification["expected_training_iterations"]
        and dev_provenance["feature_cache_mismatches"] == 0
    )
    verification_path = C13.write_json(
        results_dir / "verification.json", verification
    )
    manifest = {
        "experiment": "C13-N architecture-only HRM substitution",
        "config": asdict(cfg),
        "source_audit": str(source_audit_path),
        "training_state": training_state,
        "development_verdict": development_verdict,
        "confirmation_verdict": confirmation_result,
        "model_checkpoints": {
            name: {
                "path": str(path),
                "sha256": S.file_sha256(path),
            }
            for name, path in model_paths.items()
        },
        "outputs": {
            "candidate_summary": str(candidate_path),
            "development_verdict": str(development_verdict_path),
            "report": str(report_path),
            "verification": str(verification_path),
            **{
                name: str(value)
                for name, value in confirmation_paths.items()
                if isinstance(value, Path)
            },
        },
    }
    manifest_path = C13.write_json(Path(cfg.out_dir) / "manifest.json", manifest)
    output_paths = {
        "source_audit": source_audit_path,
        "training_history": results_dir / "training_history.csv",
        "training_state": results_dir / "training_state.json",
        "development_raw": results_dir / "development_raw.csv",
        "development_diagnostics": results_dir / "development_diagnostics.csv",
        "development_candidates": candidate_path,
        "development_verdict": development_verdict_path,
        "report": report_path,
        "verification": verification_path,
        "manifest": manifest_path,
    }
    for name in (
        "confirmation_cohort",
        "confirmation_raw",
        "confirmation_diagnostics",
        "confirmation_verdict",
    ):
        path = results_dir / f"{name}.json"
        if name in {"confirmation_raw", "confirmation_diagnostics"}:
            path = results_dir / f"{name}.csv"
        if path.exists():
            output_paths[name] = path
    for path in checkpoints:
        output_paths[path.stem] = path
    integrity_path = C13.write_json(
        Path(cfg.out_dir) / "integrity.json",
        {
            "inputs": {
                "implementation": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": S.file_sha256(Path(__file__).resolve()),
                },
                "preregistration": {
                    "path": str(Path(cfg.preregistration)),
                    "sha256": S.file_sha256(Path(cfg.preregistration)),
                },
                "source_c13j_integrity": {
                    "path": str(
                        Path(cfg.multisuite_run_dir) / "integrity.json"
                    ),
                    "sha256": S.file_sha256(
                        Path(cfg.multisuite_run_dir) / "integrity.json"
                    ),
                },
            },
            "outputs": {
                name: {"path": str(path), "sha256": S.file_sha256(path)}
                for name, path in output_paths.items()
            },
        },
    )
    if not verification["integrity_pass"]:
        raise RuntimeError("C13-N verification failed")
    print(
        f"[c13n] {development_verdict['verdict']} "
        f"confirmation="
        f"{confirmation_result['verdict'] if confirmation_result else 'not_run'} "
        f"-> {report_path}",
        flush=True,
    )
    return {
        "development": development_verdict,
        "confirmation": confirmation_result,
        "verification": verification,
        "integrity": str(integrity_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="C13-N architecture-only HRM substitution"
    )
    for field in HrmSubstitutionConfig.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        parser.add_argument(name, type=type(field.default), default=field.default)
    return parser.parse_args()


def main() -> None:
    cfg = HrmSubstitutionConfig(**vars(parse_args()))
    run(cfg)


if __name__ == "__main__":
    main()
