#!/usr/bin/env python3
"""C13-K development test for radius-bounded Bellman integration."""
from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

import continuous_prm_common as C
import continuous_prm_c13_certified_search as S
import continuous_prm_c13_identifiability as I
import continuous_prm_c13_lhbl_c7_comparison as X
import continuous_prm_c13_lhbl_generated_v3 as H
import continuous_prm_c13_lhbl_multisuite as J
import continuous_prm_c13_shared_queue as Q
import continuous_prm_c13_state_heuristic as C13


@dataclass
class LocalBackupConfig:
    original_run_dir: str = "runs/c13_lhbl_flat_48w"
    multisuite_run_dir: str = "runs/c13_lhbl_multisuite"
    c7_run_dir: str = "runs/c7_local"
    out_dir: str = "runs/c13_local_bellman_integration"
    preregistration: str = (
        "../../docs/experiments/continuous/c13/design/"
        "2026-07-17-c13k-local-bellman-integration.md"
    )
    alphas: str = "0.50,0.75,1.00"
    sensor_radius_frac: float = 0.20
    cost_ceiling: float = 1.10
    required_negative_suites: int = 4
    bootstrap_replicates: int = 20_000
    bootstrap_seed: int = 813_337
    device: str = "cpu"


def resolve_paths(cfg: LocalBackupConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    for field_name in (
        "original_run_dir",
        "multisuite_run_dir",
        "c7_run_dir",
        "out_dir",
        "preregistration",
    ):
        path = Path(getattr(cfg, field_name))
        if not path.is_absolute():
            setattr(cfg, field_name, str((script_dir / path).resolve()))


def _load_model(path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_name") != "flat_mlp":
        raise RuntimeError(f"unexpected checkpoint model family: {path}")
    if payload.get("shortest_path_target") is not False:
        raise RuntimeError(f"checkpoint target provenance changed: {path}")
    model_cfg = I.StudyConfig(**payload["model_config"])
    model = H.build_lhbl_model("flat_mlp", model_cfg)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload


def load_models(
    cfg: LocalBackupConfig, device: torch.device
) -> Tuple[Dict[str, torch.nn.Module], Dict[str, Dict[str, Any]], Dict[str, Path]]:
    paths: Dict[str, Path] = {
        "maze_i8": Path(cfg.original_run_dir)
        / "checkpoints"
        / "flat_mlp_iteration_08.pt"
    }
    for iteration in range(1, 9):
        paths[f"multisuite_i{iteration}"] = (
            Path(cfg.multisuite_run_dir)
            / "checkpoints"
            / f"flat_mlp_iteration_{iteration:02d}.pt"
        )
    models: Dict[str, torch.nn.Module] = {}
    payloads: Dict[str, Dict[str, Any]] = {}
    for label, path in paths.items():
        model, payload = _load_model(path, device)
        models[label] = model
        payloads[label] = payload
    return models, payloads, paths


def rebuild_development(
    cfg: LocalBackupConfig,
) -> Tuple[List[H.WorldBundle], List[Dict[str, Any]], List[Path]]:
    multi_cfg = J.MultiSuiteConfig(
        c7_run_dir=cfg.c7_run_dir,
        out_dir=cfg.multisuite_run_dir,
        device=cfg.device,
    )
    bundles, records, caches = J.build_balanced_bundles(
        multi_cfg,
        "development",
        J.DEV_SUITES,
        multi_cfg.development_worlds_per_suite,
        multi_cfg.development_seed_offset,
    )
    saved = H._read_json(
        Path(cfg.multisuite_run_dir) / "results" / "cohorts.json"
    )["records"]["development"]
    keys = ("suite", "global_world_index", "world_seed", "roadmap_seed", "nodes", "edges")
    observed = [{key: row[key] for key in keys} for row in records]
    expected = [{key: row[key] for key in keys} for row in saved]
    if observed != expected:
        raise RuntimeError("C13-J development cohort replay changed")
    return bundles, records, caches


def evaluate(
    cfg: LocalBackupConfig,
    bundles: Sequence[H.WorldBundle],
    models: Mapping[str, torch.nn.Module],
    payloads: Mapping[str, Mapping[str, Any]],
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    multi_cfg = J.MultiSuiteConfig(
        c7_run_dir=cfg.c7_run_dir,
        out_dir=cfg.multisuite_run_dir,
        device=cfg.device,
    )
    comparators = J._load_c7_comparators(multi_cfg, device)
    alphas = C13.parse_float_csv(cfg.alphas)
    rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    for ordinal, bundle in enumerate(bundles, start=1):
        roadmap = bundle.roadmap
        euclid = C13.euclidean_to_goal(roadmap.points, roadmap.points[1])
        optimal = float(roadmap.dist_to_goal[0])
        references: Dict[str, Dict[str, Any]] = {}
        for name, provider in comparators.items():
            rank = np.asarray(provider.node_h(bundle.world, roadmap, 1), dtype=np.float64)
            references[name] = X.astar_with_path(
                roadmap.adj, rank, len(roadmap.points)
            )
        for label, model in sorted(models.items()):
            iteration = int(payloads[label]["iteration"])
            family = "multisuite" if label.startswith("multisuite") else "maze"
            inference_started = time.perf_counter()
            prediction = np.asarray(
                I.predict_model(model, bundle.features, device), dtype=np.float64
            )
            inference_seconds = float(time.perf_counter() - inference_started)
            learned = euclid + float(bundle.world.side_len) * prediction
            backup_started = time.perf_counter()
            local_values, local_diagnostics = H.limited_horizon_values(
                roadmap.points,
                roadmap.adj,
                roadmap.points[1],
                learned,
                float(cfg.sensor_radius_frac) * float(bundle.world.side_len),
            )
            backup_seconds = float(time.perf_counter() - backup_started)
            oracle = np.asarray(roadmap.dist_to_goal, dtype=np.float64)
            connected = np.isfinite(oracle) & (oracle < C.INF / 10.0)
            diagnostics.append(
                {
                    "suite": bundle.suite,
                    "world_index": int(bundle.world_index),
                    "world_seed": int(bundle.world_seed),
                    "model_label": label,
                    "family": family,
                    "iteration": iteration,
                    "inference_seconds": inference_seconds,
                    "local_backup_seconds": backup_seconds,
                    "fallback_nodes": int(
                        sum(bool(row["fallback"]) for row in local_diagnostics)
                    ),
                    "observed_nodes_mean": float(
                        np.mean([row["observed_nodes"] for row in local_diagnostics])
                    ),
                    "exit_actions_mean": float(
                        np.mean([row["exit_actions"] for row in local_diagnostics])
                    ),
                    "static_spearman_eval_only": I.safe_spearman(
                        learned[connected], oracle[connected]
                    ),
                    "local_backup_spearman_eval_only": I.safe_spearman(
                        local_values[connected], oracle[connected]
                    ),
                }
            )
            for alpha in alphas:
                rank = euclid + float(alpha) * (local_values - euclid)
                result = X.astar_with_path(
                    roadmap.adj, rank, len(roadmap.points)
                )
                if not result["found"]:
                    raise RuntimeError("local-backup candidate failed on connected graph")
                path = Q.validate_path(
                    roadmap.adj, result["path"], result["cost"]
                )
                if not path["valid"]:
                    raise RuntimeError("local-backup A* returned an invalid path")
                rows.append(
                    {
                        "suite": bundle.suite,
                        "world_index": int(bundle.world_index),
                        "world_seed": int(bundle.world_seed),
                        "model_label": label,
                        "family": family,
                        "iteration": iteration,
                        "alpha": float(alpha),
                        "found": True,
                        "path_valid": True,
                        "current_expansions": int(result["expansions"]),
                        "current_cost_ratio_eval_only": float(
                            result["cost"] / optimal
                        ),
                        "field_hrm_expansions": int(
                            references["field_hrm"]["expansions"]
                        ),
                        "field_hrm_cost_ratio_eval_only": float(
                            references["field_hrm"]["cost"] / optimal
                        ),
                        "delta_vs_field_hrm": int(result["expansions"])
                        - int(references["field_hrm"]["expansions"]),
                        "scalar_hrm_expansions": int(
                            references["scalar_hrm"]["expansions"]
                        ),
                        "scalar_hrm_cost_ratio_eval_only": float(
                            references["scalar_hrm"]["cost"] / optimal
                        ),
                        "delta_vs_scalar_hrm": int(result["expansions"])
                        - int(references["scalar_hrm"]["expansions"]),
                        "inference_seconds": inference_seconds,
                        "local_backup_seconds": backup_seconds,
                    }
                )
        print(
            f"[c13k] development {ordinal}/{len(bundles)} "
            f"{bundle.suite}/{bundle.world_index}",
            flush=True,
        )
    return rows, diagnostics


def summarize(
    cfg: LocalBackupConfig, rows: Sequence[Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    grouped: DefaultDict[Tuple[str, str, int, float, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["model_label"]),
                str(row["family"]),
                int(row["iteration"]),
                float(row["alpha"]),
                str(row["suite"]),
            )
        ].append(row)
    for (label, family, iteration, alpha, suite), group in sorted(grouped.items()):
        delta = np.asarray([float(row["delta_vs_field_hrm"]) for row in group])
        cells.append(
            {
                "model_label": label,
                "family": family,
                "iteration": iteration,
                "alpha": alpha,
                "suite": suite,
                "worlds": len(group),
                "current_expansions_mean": float(
                    np.mean([float(row["current_expansions"]) for row in group])
                ),
                "field_hrm_expansions_mean": float(
                    np.mean([float(row["field_hrm_expansions"]) for row in group])
                ),
                "delta_vs_field_hrm_mean": float(np.mean(delta)),
                "wins": int(np.sum(delta < 0.0)),
                "ties": int(np.sum(delta == 0.0)),
                "losses": int(np.sum(delta > 0.0)),
                "current_cost_ratio_max_eval_only": float(
                    np.max([float(row["current_cost_ratio_eval_only"]) for row in group])
                ),
            }
        )
    pooled: List[Dict[str, Any]] = []
    candidates: DefaultDict[Tuple[str, str, int, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        candidates[
            (
                str(row["model_label"]),
                str(row["family"]),
                int(row["iteration"]),
                float(row["alpha"]),
            )
        ].append(row)
    passing: List[Dict[str, Any]] = []
    for (label, family, iteration, alpha), group in sorted(candidates.items()):
        delta = np.asarray([float(row["delta_vs_field_hrm"]) for row in group])
        low, high = X._bootstrap_mean_ci(
            delta,
            cfg.bootstrap_replicates,
            cfg.bootstrap_seed + iteration * 10_000 + int(alpha * 1000) + len(label),
        )
        suite_means = {
            cell["suite"]: float(cell["delta_vs_field_hrm_mean"])
            for cell in cells
            if cell["model_label"] == label
            and math.isclose(float(cell["alpha"]), alpha)
        }
        negative_suites = int(sum(value < 0.0 for value in suite_means.values()))
        max_cost = float(
            np.max([float(row["current_cost_ratio_eval_only"]) for row in group])
        )
        candidate = {
            "model_label": label,
            "family": family,
            "iteration": iteration,
            "alpha": alpha,
            "worlds": len(group),
            "current_expansions_mean": float(
                np.mean([float(row["current_expansions"]) for row in group])
            ),
            "field_hrm_expansions_mean": float(
                np.mean([float(row["field_hrm_expansions"]) for row in group])
            ),
            "scalar_hrm_expansions_mean": float(
                np.mean([float(row["scalar_hrm_expansions"]) for row in group])
            ),
            "delta_vs_field_hrm_mean": float(np.mean(delta)),
            "delta_vs_field_hrm_ci95_low": low,
            "delta_vs_field_hrm_ci95_high": high,
            "wins": int(np.sum(delta < 0.0)),
            "ties": int(np.sum(delta == 0.0)),
            "losses": int(np.sum(delta > 0.0)),
            "negative_suites": negative_suites,
            "suite_delta_means": suite_means,
            "current_cost_ratio_mean_eval_only": float(
                np.mean([float(row["current_cost_ratio_eval_only"]) for row in group])
            ),
            "current_cost_ratio_max_eval_only": max_cost,
            "gate_pass": bool(
                len(group) == 24
                and max_cost <= cfg.cost_ceiling + 1.0e-12
                and high < 0.0
                and negative_suites >= cfg.required_negative_suites
            ),
        }
        pooled.append(candidate)
        if candidate["gate_pass"]:
            passing.append(candidate)
    selected = (
        min(
            passing,
            key=lambda row: (
                float(row["current_expansions_mean"]),
                float(row["current_cost_ratio_max_eval_only"]),
                0 if row["family"] == "multisuite" else 1,
                int(row["iteration"]),
                float(row["alpha"]),
            ),
        )
        if passing
        else None
    )
    verdict = {
        "verdict": (
            "local_bellman_development_pass_requires_fresh_confirmation"
            if selected is not None
            else "local_bellman_integration_rejected"
        ),
        "gate_pass": selected is not None,
        "selected_candidate": selected,
        "passing_candidates": len(passing),
        "authorization": (
            "run_fixed_local_bellman_candidate_on_seed_offset_15000000"
            if selected is not None
            else "add_genuine_online_search_history_state"
        ),
    }
    return cells, pooled, verdict


def run(cfg: LocalBackupConfig) -> Dict[str, Any]:
    device = H.resolve_device(cfg.device)
    if str(device) != "cpu":
        raise RuntimeError("C13-K is locked to CPU")
    bundles, cohort_records, cache_paths = rebuild_development(cfg)
    models, payloads, checkpoint_paths = load_models(cfg, device)
    rows, diagnostics = evaluate(cfg, bundles, models, payloads, device)
    cells, pooled, verdict = summarize(cfg, rows)
    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    raw_path = C13.write_csv(result_dir / "development_raw.csv", rows)
    diagnostics_path = C13.write_csv(
        result_dir / "provider_diagnostics.csv", diagnostics
    )
    cells_path = C13.write_csv(result_dir / "candidate_suite_summary.csv", cells)
    pooled_path = C13.write_csv(result_dir / "candidate_pooled_summary.csv", pooled)
    verdict_path = C13.write_json(result_dir / "gate_verdict.json", verdict)
    verification = {
        "device": str(device),
        "development_worlds": len(bundles),
        "models": len(models),
        "alphas": C13.parse_float_csv(cfg.alphas),
        "rows": len(rows),
        "expected_rows": int(len(bundles) * len(models) * len(C13.parse_float_csv(cfg.alphas))),
        "invalid_paths": int(sum(not bool(row["path_valid"]) for row in rows)),
        "full_map_runtime_input": False,
        "runtime_information": (
            "current_goal_geometry_bounded_rays_one_hop_actions_plus_"
            "radius_bounded_local_subgraph_and_frozen_exit_values"
        ),
        "shortest_path_target": False,
        "training_target_reads_dist_to_goal": False,
        "development_cohort_matches_c13j": True,
        "confirmation_seed_offset": 15_000_000,
    }
    verification["integrity_pass"] = bool(
        verification["rows"] == verification["expected_rows"]
        and verification["invalid_paths"] == 0
    )
    verification_path = C13.write_json(
        result_dir / "verification.json", verification
    )
    manifest = {
        "experiment": "C13-K radius-bounded Bellman integration",
        "config": asdict(cfg),
        "development_source": "C13-J locked development block",
        "operator": "local_dijkstra_to_goal_or_exit_plus_frozen_learned_value",
        "full_map_runtime_input": False,
        "fresh_confirmation_required": bool(verdict["gate_pass"]),
        "outputs": {
            "raw": str(raw_path),
            "diagnostics": str(diagnostics_path),
            "cells": str(cells_path),
            "pooled": str(pooled_path),
            "verdict": str(verdict_path),
            "verification": str(verification_path),
        },
    }
    manifest_path = C13.write_json(Path(cfg.out_dir) / "manifest.json", manifest)
    inputs = {
        "implementation": Path(__file__).resolve(),
        "preregistration": Path(cfg.preregistration),
        "c13j_cohorts": Path(cfg.multisuite_run_dir) / "results" / "cohorts.json",
        "c13j_verdict": Path(cfg.multisuite_run_dir) / "results" / "gate_verdict.json",
        "c7_field_hrm": Path(cfg.c7_run_dir) / "checkpoints" / "c6_heatmap__hrm.pt",
        "c7_scalar_hrm": Path(cfg.c7_run_dir) / "checkpoints" / "avgbase__hrm.pt",
        **{f"checkpoint_{label}": path for label, path in checkpoint_paths.items()},
        **{f"development_cache_{index:02d}": path for index, path in enumerate(cache_paths)},
    }
    outputs = {
        "raw": raw_path,
        "diagnostics": diagnostics_path,
        "cells": cells_path,
        "pooled": pooled_path,
        "verdict": verdict_path,
        "verification": verification_path,
        "manifest": manifest_path,
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
            "cohort_records": cohort_records,
        },
    )
    if not verification["integrity_pass"]:
        raise RuntimeError("C13-K verification failed")
    print(
        f"[c13k] {verdict['verdict']} selected={verdict['selected_candidate']} "
        f"-> {verdict_path}",
        flush=True,
    )
    return {"verdict": verdict, "verification": verification, "integrity": integrity_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-K local Bellman integration")
    parser.add_argument("--out-dir", default=LocalBackupConfig.out_dir)
    parser.add_argument("--device", default=LocalBackupConfig.device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = LocalBackupConfig(out_dir=args.out_dir, device=args.device)
    resolve_paths(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
