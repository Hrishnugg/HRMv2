#!/usr/bin/env python3
"""C13-M untouched matched-quality confirmation.

This experiment freezes one current-state operating point selected in C13-L:
the C13-J iteration-8 flat model, one radius-0.20 local Bellman backup, and
alpha 1.50 in the exact C7 no-reopen A* integration.  It compares that arm to
all C7 providers on a new suite-balanced cohort and carries the independently
confirmed reopening-FOCAL arm as a separate bounded safety control.
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

import continuous_prm_common as C
import continuous_prm_c13_certified_search as S
import continuous_prm_c13_identifiability as I
import continuous_prm_c13_lhbl_c7_comparison as X
import continuous_prm_c13_lhbl_focal_matched_control_diagnostic as F
import continuous_prm_c13_lhbl_generated_v3 as H
import continuous_prm_c13_lhbl_multisuite as J
import continuous_prm_c13_local_bellman_integration as K
import continuous_prm_c13_shared_queue as Q
import continuous_prm_c13_state_heuristic as C13
import continuous_prm_c7_integration_compare as C7


CURRENT_ARM = "current_local_backup_alpha_1_50"
SAFETY_ARM = "current_bounded_focal_alpha_0_50"
C7_ARMS = (
    "euclid",
    "field_hrm",
    "field_onlstm",
    "field_unet",
    "oracle",
    "scalar_hrm",
    "scalar_onlstm",
)
ALL_ARMS = tuple(sorted(C7_ARMS)) + (CURRENT_ARM, SAFETY_ARM)
PAIRWISE_COMPARATORS = (
    "field_hrm",
    "field_onlstm",
    "field_unet",
    "scalar_hrm",
    "scalar_onlstm",
    "euclid",
)
CURRENT_BOUNDARY = (
    "current_goal_geometry_bounded_rays_one_hop_actions_plus_"
    "radius_bounded_local_subgraph_and_frozen_exit_values"
)
SAFETY_BOUNDARY = "current_goal_geometry_bounded_rays_one_hop_actions"


@dataclass
class ConfirmationConfig:
    multisuite_run_dir: str = "runs/c13_lhbl_multisuite"
    original_run_dir: str = "runs/c13_lhbl_flat_48w"
    c7_run_dir: str = "runs/c7_local"
    c13i_run_dir: str = "runs/c13_lhbl_c7_comparison"
    c13l_run_dir: str = "runs/c13_local_backup_scale"
    out_dir: str = "runs/c13_matched_quality_confirmation"
    preregistration: str = (
        "../../docs/experiments/continuous/c13/design/"
        "2026-07-17-c13m-matched-quality-confirmation.md"
    )
    worlds_per_suite: int = 24
    seed_offset: int = 15_000_000
    roadmap_nodes: int = 192
    roadmap_k: int = 7
    max_world_retries: int = 200
    grid_size: int = 64
    sector_tokens: int = 16
    current_iteration: int = 8
    current_alpha: float = 1.50
    sensor_radius_frac: float = 0.20
    safety_iteration: int = 4
    safety_alpha: float = 0.50
    safety_w: float = 1.10
    budget: int = 384
    required_negative_suites: int = 4
    mean_cost_margin: float = 0.005
    max_cost_margin: float = 0.020
    bootstrap_replicates: int = 20_000
    bootstrap_seed: int = 1_013_337
    device: str = "cpu"


def resolve_paths(cfg: ConfirmationConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    for field_name in (
        "multisuite_run_dir",
        "original_run_dir",
        "c7_run_dir",
        "c13i_run_dir",
        "c13l_run_dir",
        "out_dir",
        "preregistration",
    ):
        path = Path(getattr(cfg, field_name))
        if not path.is_absolute():
            setattr(cfg, field_name, str((script_dir / path).resolve()))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seed_set(rows: Iterable[Mapping[str, Any]]) -> set[int]:
    return {int(row["world_seed"]) for row in rows}


def _csv_seed_set(path: Path) -> set[int]:
    return _seed_set(_read_csv(path))


def _checkpoint_paths(cfg: ConfirmationConfig) -> Dict[str, Path]:
    result = {
        "current_multisuite_iteration_08": (
            Path(cfg.multisuite_run_dir) / "checkpoints" / "flat_mlp_iteration_08.pt"
        ),
        "safety_original_iteration_04": (
            Path(cfg.original_run_dir) / "checkpoints" / "flat_mlp_iteration_04.pt"
        ),
        "c7_field_hrm": Path(cfg.c7_run_dir) / "checkpoints" / "c6_heatmap__hrm.pt",
        "c7_field_onlstm": Path(cfg.c7_run_dir) / "checkpoints" / "c6_heatmap__onlstm.pt",
        "c7_field_unet": Path(cfg.c7_run_dir) / "checkpoints" / "c6_heatmap__unet.pt",
        "c7_scalar_hrm": Path(cfg.c7_run_dir) / "checkpoints" / "avgbase__hrm.pt",
        "c7_scalar_onlstm": Path(cfg.c7_run_dir) / "checkpoints" / "avgbase__onlstm.pt",
    }
    missing = [str(path) for path in result.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing frozen checkpoints: {missing}")
    return result


def _multi_cfg(cfg: ConfirmationConfig) -> J.MultiSuiteConfig:
    return J.MultiSuiteConfig(
        c7_run_dir=cfg.c7_run_dir,
        c13i_run_dir=cfg.c13i_run_dir,
        out_dir=cfg.out_dir,
        roadmap_nodes=int(cfg.roadmap_nodes),
        roadmap_k=int(cfg.roadmap_k),
        max_world_retries=int(cfg.max_world_retries),
        device=cfg.device,
    )


def build_confirmation_cohort(
    cfg: ConfirmationConfig,
) -> Tuple[List[H.WorldBundle], List[Dict[str, Any]], List[Path], Dict[str, Any]]:
    multi_cfg = _multi_cfg(cfg)
    bundles, records, caches = J.build_balanced_bundles(
        multi_cfg,
        "matched_quality_confirmation",
        J.DEV_SUITES,
        int(cfg.worlds_per_suite),
        int(cfg.seed_offset),
    )
    confirmation = {int(bundle.world_seed) for bundle in bundles}
    prior: Dict[str, set[int]] = {}

    c13j = H._read_json(Path(cfg.multisuite_run_dir) / "results" / "cohorts.json")
    for split, rows in c13j["records"].items():
        prior[f"c13j_{split}"] = _seed_set(rows)
    prior["c13i"] = J.c13i_seed_set(multi_cfg)

    c13l = H._read_json(
        Path(cfg.c13l_run_dir) / "results" / "calibration_cohort.json"
    )
    prior["c13l_alpha_calibration"] = _seed_set(c13l["records"])

    original = H._read_json(
        Path(cfg.original_run_dir) / "results" / "lhbl_cohorts.json"
    )
    for split in ("train", "validation", "development_eval"):
        prior[f"original_{split}"] = _seed_set(original.get(split, []))
    prior["original_search_raw"] = _csv_seed_set(
        Path(cfg.original_run_dir) / "results" / "lhbl_search_raw.csv"
    )

    overlaps = {name: len(confirmation & seeds) for name, seeds in prior.items()}
    expected = int(cfg.worlds_per_suite) * len(J.DEV_SUITES)
    suite_counts = {
        suite: sum(bundle.suite == suite for bundle in bundles) for suite in J.DEV_SUITES
    }
    verification = {
        "unique_seeds": len(confirmation),
        "expected_unique_seeds": expected,
        "suite_counts": suite_counts,
        "expected_worlds_per_suite": int(cfg.worlds_per_suite),
        "prior_overlap": overlaps,
    }
    if len(confirmation) != expected:
        raise RuntimeError(
            f"confirmation seed uniqueness failed: {len(confirmation)}/{expected}"
        )
    if any(count != int(cfg.worlds_per_suite) for count in suite_counts.values()):
        raise RuntimeError(f"confirmation suite balance failed: {suite_counts}")
    if any(overlaps.values()):
        raise RuntimeError(f"confirmation seed overlap: {overlaps}")
    return bundles, records, caches, verification


def c7_config(cfg: ConfirmationConfig) -> C7.C7Config:
    comparison = X.ComparisonConfig(
        c7_run_dir=cfg.c7_run_dir,
        out_dir=cfg.out_dir,
        suites=",".join(J.DEV_SUITES),
        worlds=int(cfg.worlds_per_suite),
        roadmap_nodes=int(cfg.roadmap_nodes),
        roadmap_k=int(cfg.roadmap_k),
        grid_size=int(cfg.grid_size),
        sector_tokens=int(cfg.sector_tokens),
        device=cfg.device,
    )
    return X.c7_config(comparison)


def load_frozen_components(
    cfg: ConfirmationConfig,
) -> Tuple[Any, Mapping[str, Any], Any, Mapping[str, Any], Dict[str, Any], Dict[str, Path]]:
    device = H.resolve_device(cfg.device)
    if str(device) != "cpu":
        raise RuntimeError("C13-M is locked to CPU")
    paths = _checkpoint_paths(cfg)
    current_model, current_payload = K._load_model(
        paths["current_multisuite_iteration_08"], device
    )
    safety_model, safety_payload = K._load_model(
        paths["safety_original_iteration_04"], device
    )
    if int(current_payload["iteration"]) != int(cfg.current_iteration):
        raise RuntimeError("current checkpoint iteration changed")
    if int(safety_payload["iteration"]) != int(cfg.safety_iteration):
        raise RuntimeError("safety checkpoint iteration changed")
    if current_payload.get("shortest_path_target") is not False:
        raise RuntimeError("current model target provenance changed")
    if safety_payload.get("shortest_path_target") is not False:
        raise RuntimeError("safety model target provenance changed")
    providers = C7._load_eval_providers(Path(cfg.c7_run_dir), c7_config(cfg), device)
    if set(providers) != set(C7_ARMS):
        raise RuntimeError(
            f"unexpected C7 provider set: {sorted(providers)}; expected {sorted(C7_ARMS)}"
        )
    return current_model, current_payload, safety_model, safety_payload, providers, paths


def _rank_diagnostic(
    bundle: H.WorldBundle,
    arm: str,
    boundary: str,
    rank: np.ndarray,
    representation_seconds: float,
    model_seconds: float,
    local_backup_seconds: float,
    local_diagnostics: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    oracle = np.asarray(bundle.roadmap.dist_to_goal, dtype=np.float64)
    connected = np.isfinite(oracle) & (oracle < C.INF / 10.0)
    local = list(local_diagnostics or [])
    return {
        "suite": bundle.suite,
        "world_index": int(bundle.world_index),
        "world_seed": int(bundle.world_seed),
        "arm": arm,
        "runtime_information_boundary": boundary,
        "rank_vs_oracle_spearman_eval_only": I.safe_spearman(
            np.asarray(rank)[connected], oracle[connected]
        ),
        "overestimate_rate_eval_only": float(
            np.mean(np.asarray(rank)[connected] > oracle[connected] + 1.0e-9)
        ),
        "representation_seconds": float(representation_seconds),
        "model_seconds": float(model_seconds),
        "local_backup_seconds": float(local_backup_seconds),
        "fallback_nodes": int(sum(_as_bool(row["fallback"]) for row in local)),
        "observed_nodes_mean": (
            float(np.mean([float(row["observed_nodes"]) for row in local]))
            if local
            else 0.0
        ),
        "exit_actions_mean": (
            float(np.mean([float(row["exit_actions"]) for row in local]))
            if local
            else 0.0
        ),
    }


def _result_row(
    bundle: H.WorldBundle,
    record: Mapping[str, Any],
    arm: str,
    boundary: str,
    algorithm: str,
    result: Mapping[str, Any],
    optimal: float,
    budget: int,
    representation_seconds: float,
    model_seconds: float,
    local_backup_seconds: float,
    search_seconds: float,
    safety_w: float,
) -> Dict[str, Any]:
    found = bool(result["found"])
    if found:
        path = Q.validate_path(bundle.roadmap.adj, result["path"], result["cost"])
        cost = float(result["cost"])
        cost_ratio = float(cost / optimal)
    else:
        path = {"valid": False, "edges": 0}
        cost = float("nan")
        cost_ratio = float("nan")
    anchor = float(result.get("anchor_f_min_at_return", float("nan")))
    is_safety = arm == SAFETY_ARM
    bound_violation = bool(
        is_safety and (not found or cost > float(safety_w) * optimal + 1.0e-9)
    )
    certificate_violation = bool(
        is_safety
        and (
            not found
            or not math.isfinite(anchor)
            or cost > float(safety_w) * anchor + 1.0e-9
        )
    )
    return {
        "suite": bundle.suite,
        "suite_world_index": int(record["suite_world_index"]),
        "world_index": int(bundle.world_index),
        "world_seed": int(bundle.world_seed),
        "roadmap_seed": int(record["roadmap_seed"]),
        "nodes": int(len(bundle.roadmap.points)),
        "edges": int(sum(len(group) for group in bundle.roadmap.adj) // 2),
        "arm": arm,
        "runtime_information_boundary": boundary,
        "algorithm": algorithm,
        "budget": int(budget),
        "found": found,
        "path_valid": bool(path["valid"]),
        "path_edges": int(path["edges"]),
        "cost": cost,
        "optimal": float(optimal),
        "cost_ratio_eval_only": cost_ratio,
        "expansions": int(result["expansions"]),
        "closed": int(result.get("closed", 0)),
        "max_expansions_per_state": int(result.get("max_expansions_per_state", 1)),
        "anchor_f_min_at_return": anchor if is_safety else "",
        "bound_violation_eval_only": bound_violation,
        "certificate_violation": certificate_violation,
        "representation_seconds": float(representation_seconds),
        "model_seconds": float(model_seconds),
        "local_backup_seconds": float(local_backup_seconds),
        "search_seconds": float(search_seconds),
    }


def _completed_seeds(rows: Sequence[Mapping[str, Any]]) -> set[int]:
    arms: DefaultDict[int, List[str]] = defaultdict(list)
    for row in rows:
        arms[int(row["world_seed"])].append(str(row["arm"]))
    complete: set[int] = set()
    expected = set(ALL_ARMS)
    for seed, observed in arms.items():
        if len(observed) != len(set(observed)):
            raise RuntimeError(f"duplicate raw arm rows for world seed {seed}")
        if set(observed) == expected:
            complete.add(seed)
        else:
            raise RuntimeError(
                f"incomplete persisted world {seed}: {sorted(observed)} vs {sorted(expected)}"
            )
    return complete


def evaluate(
    cfg: ConfirmationConfig,
    bundles: Sequence[H.WorldBundle],
    records: Sequence[Mapping[str, Any]],
    raw_path: Path,
    diagnostics_path: Path,
    meta_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Path]]:
    (
        current_model,
        current_payload,
        safety_model,
        safety_payload,
        providers,
        checkpoints,
    ) = load_frozen_components(cfg)
    device = H.resolve_device(cfg.device)
    fingerprint_payload = {
        "config": asdict(cfg),
        "implementation_sha256": S.file_sha256(Path(__file__).resolve()),
        "preregistration_sha256": S.file_sha256(Path(cfg.preregistration)),
        "checkpoints": {
            name: S.file_sha256(path) for name, path in checkpoints.items()
        },
        "arms": list(ALL_ARMS),
    }
    fingerprint = _json_hash(fingerprint_payload)
    if meta_path.exists():
        previous = H._read_json(meta_path)
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError("existing C13-M shard fingerprint differs")
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
    completed = _completed_seeds(rows)
    record_by_seed = {int(row["world_seed"]): row for row in records}
    feature_mismatches = 0

    for ordinal, bundle in enumerate(bundles, start=1):
        if int(bundle.world_seed) in completed:
            continue
        record = record_by_seed[int(bundle.world_seed)]
        world_started = time.perf_counter()
        roadmap = bundle.roadmap
        optimal = float(roadmap.dist_to_goal[0])
        if not math.isfinite(optimal) or optimal >= C.INF / 10.0:
            raise RuntimeError("accepted C13-M roadmap is disconnected")

        world_rows: List[Dict[str, Any]] = []
        world_diagnostics: List[Dict[str, Any]] = []
        ranks: Dict[str, np.ndarray] = {}
        for name, provider in sorted(providers.items()):
            provider_started = time.perf_counter()
            rank = np.asarray(provider.node_h(bundle.world, roadmap, 1), dtype=np.float64)
            provider_seconds = float(time.perf_counter() - provider_started)
            ranks[name] = rank
            search_started = time.perf_counter()
            result = X.astar_with_path(roadmap.adj, rank, int(cfg.budget))
            search_seconds = float(time.perf_counter() - search_started)
            boundary = X._boundary_for_provider(name)
            world_rows.append(
                _result_row(
                    bundle,
                    record,
                    name,
                    boundary,
                    "no_reopen_astar",
                    result,
                    optimal,
                    cfg.budget,
                    provider_seconds,
                    0.0,
                    0.0,
                    search_seconds,
                    cfg.safety_w,
                )
            )
            world_diagnostics.append(
                _rank_diagnostic(
                    bundle, name, boundary, rank, provider_seconds, 0.0, 0.0
                )
            )

        feature_started = time.perf_counter()
        live_features = C13.make_local_state_features(
            bundle.world, roadmap.points, roadmap.adj, J.local_config(_multi_cfg(cfg))
        )
        feature_seconds = float(time.perf_counter() - feature_started)
        if not np.array_equal(live_features, bundle.features):
            feature_mismatches += 1
            raise RuntimeError(f"live feature/cache mismatch for {bundle.world_seed}")
        euclid = ranks["euclid"]

        current_infer_started = time.perf_counter()
        current_prediction = np.asarray(
            I.predict_model(current_model, live_features, device), dtype=np.float64
        )
        current_model_seconds = float(time.perf_counter() - current_infer_started)
        current_bootstrap = (
            euclid + float(bundle.world.side_len) * current_prediction
        )
        backup_started = time.perf_counter()
        local_values, local_diagnostics = H.limited_horizon_values(
            roadmap.points,
            roadmap.adj,
            roadmap.points[1],
            current_bootstrap,
            float(cfg.sensor_radius_frac) * float(bundle.world.side_len),
        )
        backup_seconds = float(time.perf_counter() - backup_started)
        current_rank = euclid + float(cfg.current_alpha) * (local_values - euclid)
        search_started = time.perf_counter()
        current_result = X.astar_with_path(
            roadmap.adj, current_rank, int(cfg.budget)
        )
        current_search_seconds = float(time.perf_counter() - search_started)
        world_rows.append(
            _result_row(
                bundle,
                record,
                CURRENT_ARM,
                CURRENT_BOUNDARY,
                "no_reopen_astar",
                current_result,
                optimal,
                cfg.budget,
                feature_seconds,
                current_model_seconds,
                backup_seconds,
                current_search_seconds,
                cfg.safety_w,
            )
        )
        world_diagnostics.append(
            _rank_diagnostic(
                bundle,
                CURRENT_ARM,
                CURRENT_BOUNDARY,
                current_rank,
                feature_seconds,
                current_model_seconds,
                backup_seconds,
                local_diagnostics,
            )
        )

        safety_infer_started = time.perf_counter()
        safety_prediction = np.asarray(
            I.predict_model(safety_model, live_features, device), dtype=np.float64
        )
        safety_model_seconds = float(time.perf_counter() - safety_infer_started)
        safety_learned = euclid + float(bundle.world.side_len) * safety_prediction
        safety_rank = euclid + float(cfg.safety_alpha) * (safety_learned - euclid)
        safety_search_started = time.perf_counter()
        safety_result = F.focal_search_with_path(
            roadmap.adj,
            euclid,
            safety_rank,
            int(cfg.budget),
            float(cfg.safety_w),
            "fhat",
        )
        safety_search_seconds = float(time.perf_counter() - safety_search_started)
        world_rows.append(
            _result_row(
                bundle,
                record,
                SAFETY_ARM,
                SAFETY_BOUNDARY,
                "reopening_fhat_focal",
                safety_result,
                optimal,
                cfg.budget,
                feature_seconds,
                safety_model_seconds,
                0.0,
                safety_search_seconds,
                cfg.safety_w,
            )
        )
        world_diagnostics.append(
            _rank_diagnostic(
                bundle,
                SAFETY_ARM,
                SAFETY_BOUNDARY,
                safety_rank,
                feature_seconds,
                safety_model_seconds,
                0.0,
            )
        )

        if {str(row["arm"]) for row in world_rows} != set(ALL_ARMS):
            raise RuntimeError("world arm set is incomplete")
        rows.extend(world_rows)
        diagnostics.extend(world_diagnostics)
        C13.write_csv(raw_path, rows)
        C13.write_csv(diagnostics_path, diagnostics)
        print(
            f"[c13m] {ordinal}/{len(bundles)} {bundle.suite}/"
            f"{record['suite_world_index']} seed={bundle.world_seed} "
            f"current={current_result['expansions']} field_hrm="
            f"{next(row['expansions'] for row in world_rows if row['arm'] == 'field_hrm')} "
            f"elapsed={time.perf_counter() - world_started:.2f}s",
            flush=True,
        )

    provenance = {
        "current_checkpoint_iteration": int(current_payload["iteration"]),
        "safety_checkpoint_iteration": int(safety_payload["iteration"]),
        "current_shortest_path_target": current_payload.get("shortest_path_target"),
        "safety_shortest_path_target": safety_payload.get("shortest_path_target"),
        "feature_cache_mismatches": int(feature_mismatches),
        "fingerprint": fingerprint,
    }
    return rows, diagnostics, provenance, checkpoints


def _scope_groups(
    rows: Sequence[Mapping[str, Any]], scope: str
) -> List[Mapping[str, Any]]:
    return list(rows) if scope == "POOLED" else [row for row in rows if row["suite"] == scope]


def summarize_arms(
    rows: Sequence[Mapping[str, Any]], cfg: ConfirmationConfig
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    scopes = list(J.DEV_SUITES) + ["POOLED"]
    for scope in scopes:
        scoped = _scope_groups(rows, scope)
        for arm in ALL_ARMS:
            group = [row for row in scoped if row["arm"] == arm]
            valid = [row for row in group if _as_bool(row["found"]) and _as_bool(row["path_valid"])]
            expansions = np.asarray([float(row["expansions"]) for row in valid])
            costs = np.asarray([float(row["cost_ratio_eval_only"]) for row in valid])
            low, high = X._bootstrap_mean_ci(
                expansions,
                cfg.bootstrap_replicates,
                cfg.bootstrap_seed + scopes.index(scope) * 1000 + ALL_ARMS.index(arm) * 17,
            )
            output.append(
                {
                    "scope": scope,
                    "arm": arm,
                    "worlds": len(group),
                    "solved_valid": len(valid),
                    "invalid_found_paths": sum(
                        _as_bool(row["found"]) and not _as_bool(row["path_valid"])
                        for row in group
                    ),
                    "expansions_mean": float(np.mean(expansions)) if len(expansions) else float("nan"),
                    "expansions_median": float(np.median(expansions)) if len(expansions) else float("nan"),
                    "expansions_mean_ci95_low": low,
                    "expansions_mean_ci95_high": high,
                    "cost_ratio_mean_eval_only": float(np.mean(costs)) if len(costs) else float("nan"),
                    "cost_ratio_max_eval_only": float(np.max(costs)) if len(costs) else float("nan"),
                    "representation_seconds_mean": float(
                        np.mean([float(row["representation_seconds"]) for row in group])
                    ) if group else float("nan"),
                    "model_seconds_mean": float(
                        np.mean([float(row["model_seconds"]) for row in group])
                    ) if group else float("nan"),
                    "local_backup_seconds_mean": float(
                        np.mean([float(row["local_backup_seconds"]) for row in group])
                    ) if group else float("nan"),
                    "search_seconds_mean": float(
                        np.mean([float(row["search_seconds"]) for row in group])
                    ) if group else float("nan"),
                    "bound_violations": sum(
                        _as_bool(row["bound_violation_eval_only"]) for row in group
                    ),
                    "certificate_violations": sum(
                        _as_bool(row["certificate_violation"]) for row in group
                    ),
                }
            )
    return output


def pairwise_comparisons(
    rows: Sequence[Mapping[str, Any]], cfg: ConfirmationConfig
) -> List[Dict[str, Any]]:
    lookup = {
        (int(row["world_seed"]), str(row["arm"])): row for row in rows
    }
    output: List[Dict[str, Any]] = []
    scopes = list(J.DEV_SUITES) + ["POOLED"]
    for comparator in PAIRWISE_COMPARATORS:
        for scope in scopes:
            seeds = sorted(
                {
                    int(row["world_seed"])
                    for row in rows
                    if scope == "POOLED" or row["suite"] == scope
                }
            )
            pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
            for seed in seeds:
                left = lookup.get((seed, CURRENT_ARM))
                right = lookup.get((seed, comparator))
                if left is None or right is None:
                    continue
                if (
                    _as_bool(left["found"])
                    and _as_bool(left["path_valid"])
                    and _as_bool(right["found"])
                    and _as_bool(right["path_valid"])
                ):
                    pairs.append((left, right))
            delta = np.asarray(
                [float(left["expansions"]) - float(right["expansions"]) for left, right in pairs]
            )
            cost_delta = np.asarray(
                [
                    float(left["cost_ratio_eval_only"])
                    - float(right["cost_ratio_eval_only"])
                    for left, right in pairs
                ]
            )
            low, high = X._bootstrap_mean_ci(
                delta,
                cfg.bootstrap_replicates,
                cfg.bootstrap_seed + PAIRWISE_COMPARATORS.index(comparator) * 10_000 + scopes.index(scope),
            )
            output.append(
                {
                    "scope": scope,
                    "current_arm": CURRENT_ARM,
                    "comparator": comparator,
                    "paired_valid": len(pairs),
                    "delta_expansions_mean": float(np.mean(delta)) if len(delta) else float("nan"),
                    "delta_expansions_ci95_low": low,
                    "delta_expansions_ci95_high": high,
                    "wins": int(np.sum(delta < 0.0)),
                    "ties": int(np.sum(delta == 0.0)),
                    "losses": int(np.sum(delta > 0.0)),
                    "current_expansions_mean": float(
                        np.mean([float(left["expansions"]) for left, _ in pairs])
                    ) if pairs else float("nan"),
                    "comparator_expansions_mean": float(
                        np.mean([float(right["expansions"]) for _, right in pairs])
                    ) if pairs else float("nan"),
                    "current_cost_ratio_mean": float(
                        np.mean([float(left["cost_ratio_eval_only"]) for left, _ in pairs])
                    ) if pairs else float("nan"),
                    "comparator_cost_ratio_mean": float(
                        np.mean([float(right["cost_ratio_eval_only"]) for _, right in pairs])
                    ) if pairs else float("nan"),
                    "cost_ratio_delta_mean": float(np.mean(cost_delta)) if len(cost_delta) else float("nan"),
                }
            )
    return output


def _find_pair(
    pairs: Sequence[Mapping[str, Any]], scope: str, comparator: str
) -> Mapping[str, Any]:
    selected = [
        row for row in pairs if row["scope"] == scope and row["comparator"] == comparator
    ]
    if len(selected) != 1:
        raise RuntimeError(f"missing pairwise row {scope}/{comparator}")
    return selected[0]


def build_verdict(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    cfg: ConfirmationConfig,
) -> Dict[str, Any]:
    expected = int(cfg.worlds_per_suite) * len(J.DEV_SUITES)
    current = [row for row in rows if row["arm"] == CURRENT_ARM]
    field = [row for row in rows if row["arm"] == "field_hrm"]
    safety = [row for row in rows if row["arm"] == SAFETY_ARM]
    current_valid = [row for row in current if _as_bool(row["found"]) and _as_bool(row["path_valid"])]
    field_valid = [row for row in field if _as_bool(row["found"]) and _as_bool(row["path_valid"])]
    pooled = _find_pair(pairs, "POOLED", "field_hrm")
    suite_pairs = [_find_pair(pairs, suite, "field_hrm") for suite in J.DEV_SUITES]
    current_mean = float(np.mean([float(row["cost_ratio_eval_only"]) for row in current_valid]))
    field_mean = float(np.mean([float(row["cost_ratio_eval_only"]) for row in field_valid]))
    current_max = float(np.max([float(row["cost_ratio_eval_only"]) for row in current_valid]))
    field_max = float(np.max([float(row["cost_ratio_eval_only"]) for row in field_valid]))
    negative_suites = int(
        sum(float(row["delta_expansions_mean"]) < 0.0 for row in suite_pairs)
    )
    conditions = {
        "current_valid_paths_expected": len(current_valid) == expected,
        "pooled_expansion_ci_upper_below_zero": float(
            pooled["delta_expansions_ci95_high"]
        ) < 0.0,
        "negative_mean_delta_at_least_four_suites": negative_suites
        >= int(cfg.required_negative_suites),
        "current_mean_cost_within_0_005_of_field_hrm": current_mean
        <= field_mean + float(cfg.mean_cost_margin) + 1.0e-12,
        "current_max_cost_within_0_02_of_field_hrm": current_max
        <= field_max + float(cfg.max_cost_margin) + 1.0e-12,
    }
    safety_conditions = {
        "valid_paths_expected": sum(
            _as_bool(row["found"]) and _as_bool(row["path_valid"]) for row in safety
        ) == expected,
        "bound_violations_zero": sum(
            _as_bool(row["bound_violation_eval_only"]) for row in safety
        ) == 0,
        "certificate_violations_zero": sum(
            _as_bool(row["certificate_violation"]) for row in safety
        ) == 0,
    }
    gate_pass = all(conditions.values())
    return {
        "verdict": (
            "matched_quality_current_state_improvement_confirmed"
            if gate_pass
            else "matched_quality_current_state_improvement_not_confirmed"
        ),
        "gate_pass": bool(gate_pass),
        "current_arm": CURRENT_ARM,
        "primary_comparator": "field_hrm",
        "expected_worlds": expected,
        "conditions": conditions,
        "negative_suites": negative_suites,
        "current_solved_valid": len(current_valid),
        "current_cost_ratio_mean_eval_only": current_mean,
        "field_hrm_cost_ratio_mean_eval_only": field_mean,
        "mean_cost_margin_observed": current_mean - field_mean,
        "current_cost_ratio_max_eval_only": current_max,
        "field_hrm_cost_ratio_max_eval_only": field_max,
        "max_cost_margin_observed": current_max - field_max,
        "pooled_primary_pair": dict(pooled),
        "suite_primary_pairs": [dict(row) for row in suite_pairs],
        "bounded_control": {
            "pass": bool(all(safety_conditions.values())),
            "conditions": safety_conditions,
            "w": float(cfg.safety_w),
        },
        "claim_scope": (
            "empirical_matched_quality_direct_astar_efficiency_not_a_formal_bound"
        ),
        "authorization": (
            "document_confirmed_current_state_matched_quality_improvement"
            if gate_pass
            else "document_preregistered_confirmation_failure_without_retuning"
        ),
    }


def write_report(
    path: Path,
    summaries: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    verdict: Mapping[str, Any],
    cfg: ConfirmationConfig,
) -> Path:
    pooled_summaries = {row["arm"]: row for row in summaries if row["scope"] == "POOLED"}
    primary = verdict["pooled_primary_pair"]
    lines = [
        "# C13-M matched-quality fresh confirmation",
        "",
        f"**Verdict:** `{verdict['verdict']}` (`gate_pass={str(verdict['gate_pass']).lower()}`).",
        "",
        "The fixed current-state arm uses current/goal geometry, bounded rays, one-hop",
        "actions, and one radius-bounded local Bellman backup. It never receives an",
        "occupancy raster, the complete obstacle list, or a global graph solution at runtime.",
        "The C7 `field_*` arms receive the complete 64 x 64 map raster.",
        "",
        "## Preregistered primary comparison",
        "",
        f"Across {primary['paired_valid']} untouched worlds, current-state search averaged ",
        f"{float(primary['current_expansions_mean']):.3f} expansions versus ",
        f"{float(primary['comparator_expansions_mean']):.3f} for `field_hrm`: paired delta ",
        f"{float(primary['delta_expansions_mean']):+.3f}, bootstrap 95% CI ",
        f"[{float(primary['delta_expansions_ci95_low']):+.3f}, ",
        f"{float(primary['delta_expansions_ci95_high']):+.3f}], with ",
        f"{primary['wins']} wins / {primary['ties']} ties / {primary['losses']} losses.",
        "",
        f"Mean graph-optimal cost ratio was {float(verdict['current_cost_ratio_mean_eval_only']):.6f} ",
        f"for current-state versus {float(verdict['field_hrm_cost_ratio_mean_eval_only']):.6f} ",
        f"for `field_hrm`; maxima were {float(verdict['current_cost_ratio_max_eval_only']):.6f} ",
        f"and {float(verdict['field_hrm_cost_ratio_max_eval_only']):.6f}, respectively.",
        "",
        "| Suite | Current exp. | Field HRM exp. | Delta | 95% CI | W/T/L |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in verdict["suite_primary_pairs"]:
        lines.append(
            f"|{row['scope']}|{float(row['current_expansions_mean']):.2f}|"
            f"{float(row['comparator_expansions_mean']):.2f}|"
            f"{float(row['delta_expansions_mean']):+.2f}|"
            f"[{float(row['delta_expansions_ci95_low']):+.2f}, "
            f"{float(row['delta_expansions_ci95_high']):+.2f}]|"
            f"{row['wins']}/{row['ties']}/{row['losses']}|"
        )
    lines.extend(
        [
            "",
            "## Pooled operating points",
            "",
            "| Arm | Valid | Mean exp. | Mean cost ratio | Max cost ratio | Rep. s | Model s | Backup s | Search s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ALL_ARMS:
        row = pooled_summaries[arm]
        lines.append(
            f"|{arm}|{row['solved_valid']}/{row['worlds']}|"
            f"{float(row['expansions_mean']):.2f}|"
            f"{float(row['cost_ratio_mean_eval_only']):.6f}|"
            f"{float(row['cost_ratio_max_eval_only']):.6f}|"
            f"{float(row['representation_seconds_mean']):.4f}|"
            f"{float(row['model_seconds_mean']):.4f}|"
            f"{float(row['local_backup_seconds_mean']):.4f}|"
            f"{float(row['search_seconds_mean']):.4f}|"
        )
    lines.extend(
        [
            "",
            "## Gate conditions",
            "",
        ]
    )
    for condition, passed in verdict["conditions"].items():
        lines.append(f"- `{condition}`: **{str(passed).upper()}**")
    lines.extend(["", "## Separate bounded safety control", ""])
    lines.append(
        f"The reopening `fhat` FOCAL control at w={cfg.safety_w:.2f} passed="
        f"{str(verdict['bounded_control']['pass']).lower()}."
    )
    for condition, passed in verdict["bounded_control"]["conditions"].items():
        lines.append(f"- `{condition}`: **{str(passed).upper()}**")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is a comparator-relative, empirical matched-quality result for the same",
            "unbounded direct-A* setting used by C7's learned arms. It is not a formal",
            "suboptimality guarantee. The bounded FOCAL operating point is reported",
            "separately and must not be conflated with the alpha-1.50 result.",
            "",
            "The C13-L development rows were not pooled into this confirmation and no",
            "checkpoint, alpha, suite, or threshold was selected on these confirmation rows.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(cfg: ConfirmationConfig) -> Dict[str, Any]:
    resolve_paths(cfg)
    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    shard_dir = C13.ensure_dir(result_dir / "suite_shards")
    raw_path = result_dir / "confirmation_raw.csv"
    diagnostics_path = result_dir / "rank_diagnostics.csv"
    meta_path = result_dir / "evaluation_fingerprint.json"

    bundles, records, caches, cohort_verification = build_confirmation_cohort(cfg)
    cohort_path = C13.write_json(
        result_dir / "confirmation_cohort.json",
        {
            "seed_offset": int(cfg.seed_offset),
            "records": records,
            "verification": cohort_verification,
        },
    )
    rows, diagnostics, provenance, checkpoints = evaluate(
        cfg, bundles, records, raw_path, diagnostics_path, meta_path
    )
    summaries = summarize_arms(rows, cfg)
    pairs = pairwise_comparisons(rows, cfg)
    verdict = build_verdict(rows, pairs, cfg)
    summary_path = C13.write_csv(result_dir / "arm_summary.csv", summaries)
    pairs_path = C13.write_csv(result_dir / "pairwise_summary.csv", pairs)
    verdict_path = C13.write_json(result_dir / "gate_verdict.json", verdict)
    suite_shards: Dict[str, Path] = {}
    for suite in J.DEV_SUITES:
        suite_shards[suite] = C13.write_csv(
            shard_dir / f"{suite}.csv", [row for row in rows if row["suite"] == suite]
        )
    report_path = write_report(
        result_dir / "C13M_RESULTS.md", summaries, pairs, verdict, cfg
    )

    expected_worlds = int(cfg.worlds_per_suite) * len(J.DEV_SUITES)
    expected_rows = expected_worlds * len(ALL_ARMS)
    key_counts: DefaultDict[Tuple[int, str], int] = defaultdict(int)
    for row in rows:
        key_counts[(int(row["world_seed"]), str(row["arm"]))] += 1
    current_rows = [row for row in rows if row["arm"] == CURRENT_ARM]
    safety_rows = [row for row in rows if row["arm"] == SAFETY_ARM]
    invalid_found = sum(
        _as_bool(row["found"]) and not _as_bool(row["path_valid"]) for row in rows
    )
    verification = {
        "device": cfg.device,
        "cohort": cohort_verification,
        "provenance": provenance,
        "rows": len(rows),
        "expected_rows": expected_rows,
        "unique_world_seeds": len({int(row["world_seed"]) for row in rows}),
        "expected_worlds": expected_worlds,
        "duplicate_world_arm_keys": sum(count != 1 for count in key_counts.values()),
        "invalid_found_paths": int(invalid_found),
        "all_current_paths_valid": sum(
            _as_bool(row["found"]) and _as_bool(row["path_valid"]) for row in current_rows
        ) == expected_worlds,
        "all_provider_paths_valid": sum(
            _as_bool(row["found"]) and _as_bool(row["path_valid"])
            for row in rows
            if row["arm"] in C7_ARMS
        ) == expected_worlds * len(C7_ARMS),
        "bounded_control_paths_valid": sum(
            _as_bool(row["found"]) and _as_bool(row["path_valid"]) for row in safety_rows
        ) == expected_worlds,
        "bounded_control_bound_violations": sum(
            _as_bool(row["bound_violation_eval_only"]) for row in safety_rows
        ),
        "bounded_control_certificate_violations": sum(
            _as_bool(row["certificate_violation"]) for row in safety_rows
        ),
        "expected_provider_set": list(C7_ARMS),
        "runtime_information": {
            "current": CURRENT_BOUNDARY,
            "safety": SAFETY_BOUNDARY,
            "field": "complete_64x64_occupancy_goal_raster",
            "scalar": "global_obstacle_list_summaries_sectors_plus_rays_goal",
        },
        "current_full_map_runtime_input": False,
        "current_shortest_path_runtime_input": False,
        "current_training_target_reads_dist_to_goal": False,
        "confirmation_rows_not_used_for_selection": True,
        "scientific_gate_pass": bool(verdict["gate_pass"]),
    }
    verification["integrity_pass"] = bool(
        verification["rows"] == verification["expected_rows"]
        and verification["unique_world_seeds"] == expected_worlds
        and verification["duplicate_world_arm_keys"] == 0
        and verification["invalid_found_paths"] == 0
        and verification["all_current_paths_valid"]
        and verification["all_provider_paths_valid"]
        and verification["bounded_control_paths_valid"]
        and verification["bounded_control_bound_violations"] == 0
        and verification["bounded_control_certificate_violations"] == 0
        and provenance["feature_cache_mismatches"] == 0
        and all(value == 0 for value in cohort_verification["prior_overlap"].values())
    )
    verification_path = C13.write_json(result_dir / "verification.json", verification)
    manifest_path = C13.write_json(
        Path(cfg.out_dir) / "manifest.json",
        {
            "experiment": "C13-M matched-quality fresh confirmation",
            "config": asdict(cfg),
            "fixed_operating_point": {
                "model": "suite_balanced_flat_mlp",
                "iteration": int(cfg.current_iteration),
                "alpha": float(cfg.current_alpha),
                "sensor_radius_frac": float(cfg.sensor_radius_frac),
                "local_backup_applications": 1,
                "search": "no_reopen_astar",
            },
            "bounded_control": {
                "model": "original_flat_mlp",
                "iteration": int(cfg.safety_iteration),
                "alpha": float(cfg.safety_alpha),
                "w": float(cfg.safety_w),
                "search": "reopening_fhat_focal",
            },
            "checkpoint_paths": {name: str(path) for name, path in checkpoints.items()},
            "outputs": {
                "cohort": str(cohort_path),
                "raw": str(raw_path),
                "diagnostics": str(diagnostics_path),
                "summary": str(summary_path),
                "pairs": str(pairs_path),
                "verdict": str(verdict_path),
                "verification": str(verification_path),
                "report": str(report_path),
                "suite_shards": {name: str(path) for name, path in suite_shards.items()},
            },
        },
    )

    inputs: Dict[str, Path] = {
        "implementation": Path(__file__).resolve(),
        "preregistration": Path(cfg.preregistration),
        "c13l_raw_selection_evidence": Path(cfg.c13l_run_dir) / "results" / "calibration_raw.csv",
        "c13l_gate_selection_evidence": Path(cfg.c13l_run_dir) / "results" / "gate_verdict.json",
        "c13j_training_verdict": Path(cfg.multisuite_run_dir) / "results" / "gate_verdict.json",
        **checkpoints,
        **{f"feature_cache_{index:03d}": path for index, path in enumerate(caches)},
    }
    outputs: Dict[str, Path] = {
        "cohort": cohort_path,
        "raw": raw_path,
        "diagnostics": diagnostics_path,
        "summary": summary_path,
        "pairs": pairs_path,
        "verdict": verdict_path,
        "verification": verification_path,
        "report": report_path,
        "fingerprint": meta_path,
        "manifest": manifest_path,
        **{f"suite_shard_{name}": path for name, path in suite_shards.items()},
    }
    integrity_path = C13.write_json(
        Path(cfg.out_dir) / "integrity.json",
        {
            "inputs": {
                name: {"path": str(path), "sha256": S.file_sha256(path)}
                for name, path in inputs.items()
            },
            "outputs": {
                name: {"path": str(path), "sha256": S.file_sha256(path)}
                for name, path in outputs.items()
            },
        },
    )
    if not verification["integrity_pass"]:
        raise RuntimeError("C13-M artifact verification failed")
    print(
        f"[c13m] {verdict['verdict']} gate_pass={verdict['gate_pass']} "
        f"safety_pass={verdict['bounded_control']['pass']} -> {verdict_path}",
        flush=True,
    )
    return {
        "verdict": verdict,
        "verification": verification,
        "integrity": str(integrity_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-M matched-quality confirmation")
    parser.add_argument("--out-dir", default=ConfirmationConfig.out_dir)
    parser.add_argument("--device", default=ConfirmationConfig.device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(ConfirmationConfig(out_dir=args.out_dir, device=args.device))


if __name__ == "__main__":
    main()
