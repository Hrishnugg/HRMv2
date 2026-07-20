#!/usr/bin/env python3
"""C13-L alpha calibration for the fixed local-Bellman integration."""
from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import continuous_prm_common as C
import continuous_prm_c13_certified_search as S
import continuous_prm_c13_identifiability as I
import continuous_prm_c13_lhbl_c7_comparison as X
import continuous_prm_c13_lhbl_generated_v3 as H
import continuous_prm_c13_lhbl_multisuite as J
import continuous_prm_c13_local_bellman_integration as K
import continuous_prm_c13_shared_queue as Q
import continuous_prm_c13_state_heuristic as C13


@dataclass
class ScaleConfig:
    multisuite_run_dir: str = "runs/c13_lhbl_multisuite"
    c13i_run_dir: str = "runs/c13_lhbl_c7_comparison"
    c7_run_dir: str = "runs/c7_local"
    original_run_dir: str = "runs/c13_lhbl_flat_48w"
    out_dir: str = "runs/c13_local_backup_scale"
    preregistration: str = (
        "../../docs/experiments/continuous/c13/design/"
        "2026-07-17-c13l-local-backup-scale-calibration.md"
    )
    worlds_per_suite: int = 8
    seed_offset: int = 12_500_000
    alphas: str = "1.00,1.25,1.50,2.00"
    sensor_radius_frac: float = 0.20
    cost_ceiling: float = 1.10
    required_negative_suites: int = 4
    bootstrap_replicates: int = 20_000
    bootstrap_seed: int = 913_337
    device: str = "cpu"


def resolve_paths(cfg: ScaleConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    for field_name in (
        "multisuite_run_dir",
        "c13i_run_dir",
        "c7_run_dir",
        "original_run_dir",
        "out_dir",
        "preregistration",
    ):
        path = Path(getattr(cfg, field_name))
        if not path.is_absolute():
            setattr(cfg, field_name, str((script_dir / path).resolve()))


def build_calibration_cohort(
    cfg: ScaleConfig,
) -> Tuple[List[H.WorldBundle], List[Dict[str, Any]], List[Path], Dict[str, Any]]:
    multi_cfg = J.MultiSuiteConfig(
        c7_run_dir=cfg.c7_run_dir,
        c13i_run_dir=cfg.c13i_run_dir,
        out_dir=cfg.out_dir,
        device=cfg.device,
    )
    bundles, records, caches = J.build_balanced_bundles(
        multi_cfg,
        "alpha_calibration",
        J.DEV_SUITES,
        cfg.worlds_per_suite,
        cfg.seed_offset,
    )
    seeds = {int(bundle.world_seed) for bundle in bundles}
    old_seeds: Dict[str, set[int]] = {}
    c13j = H._read_json(
        Path(cfg.multisuite_run_dir) / "results" / "cohorts.json"
    )
    for split, rows in c13j["records"].items():
        old_seeds[f"c13j_{split}"] = {int(row["world_seed"]) for row in rows}
    old_seeds["c13i"] = J.c13i_seed_set(multi_cfg)
    original = H._read_json(
        Path(cfg.original_run_dir) / "results" / "lhbl_cohorts.json"
    )
    for split in ("train", "validation", "development_eval"):
        rows = original.get(split, [])
        old_seeds[f"original_{split}"] = {
            int(row["world_seed"]) for row in rows
        }
    overlaps = {name: len(seeds & values) for name, values in old_seeds.items()}
    if len(seeds) != int(cfg.worlds_per_suite * len(J.DEV_SUITES)):
        raise RuntimeError("calibration seed uniqueness failed")
    if any(overlaps.values()):
        raise RuntimeError(f"calibration seed overlap: {overlaps}")
    verification = {
        "unique_seeds": len(seeds),
        "expected_unique_seeds": int(cfg.worlds_per_suite * len(J.DEV_SUITES)),
        "overlap": overlaps,
    }
    return bundles, records, caches, verification


def evaluate(
    cfg: ScaleConfig,
    bundles: Sequence[H.WorldBundle],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path]:
    device = H.resolve_device(cfg.device)
    if str(device) != "cpu":
        raise RuntimeError("C13-L is locked to CPU")
    checkpoint = (
        Path(cfg.multisuite_run_dir)
        / "checkpoints"
        / "flat_mlp_iteration_08.pt"
    )
    model, payload = K._load_model(checkpoint, device)
    if int(payload["iteration"]) != 8:
        raise RuntimeError("C13-L checkpoint is not iteration 8")
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
        infer_started = time.perf_counter()
        prediction = np.asarray(
            I.predict_model(model, bundle.features, device), dtype=np.float64
        )
        inference_seconds = float(time.perf_counter() - infer_started)
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
                "inference_seconds": inference_seconds,
                "local_backup_seconds": backup_seconds,
                "fallback_nodes": int(
                    sum(bool(row["fallback"]) for row in local_diagnostics)
                ),
                "observed_nodes_mean": float(
                    np.mean([row["observed_nodes"] for row in local_diagnostics])
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
                raise RuntimeError("calibration A* failed on connected graph")
            path = Q.validate_path(roadmap.adj, result["path"], result["cost"])
            if not path["valid"]:
                raise RuntimeError("calibration A* path invalid")
            rows.append(
                {
                    "suite": bundle.suite,
                    "world_index": int(bundle.world_index),
                    "world_seed": int(bundle.world_seed),
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
            f"[c13l] calibration {ordinal}/{len(bundles)} "
            f"{bundle.suite}/{bundle.world_index}",
            flush=True,
        )
    return rows, diagnostics, checkpoint


def summarize(
    cfg: ScaleConfig, rows: Sequence[Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    grouped: DefaultDict[Tuple[float, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["alpha"]), str(row["suite"]))].append(row)
    for (alpha, suite), group in sorted(grouped.items()):
        delta = np.asarray([float(row["delta_vs_field_hrm"]) for row in group])
        cells.append(
            {
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
    passing: List[Dict[str, Any]] = []
    for alpha in C13.parse_float_csv(cfg.alphas):
        group = [row for row in rows if math.isclose(float(row["alpha"]), alpha)]
        delta = np.asarray([float(row["delta_vs_field_hrm"]) for row in group])
        low, high = X._bootstrap_mean_ci(
            delta,
            cfg.bootstrap_replicates,
            cfg.bootstrap_seed + int(alpha * 10_000),
        )
        suite_means = {
            row["suite"]: float(row["delta_vs_field_hrm_mean"])
            for row in cells
            if math.isclose(float(row["alpha"]), alpha)
        }
        max_cost = float(
            np.max([float(row["current_cost_ratio_eval_only"]) for row in group])
        )
        candidate = {
            "alpha": float(alpha),
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
            "negative_suites": int(sum(value < 0.0 for value in suite_means.values())),
            "suite_delta_means": suite_means,
            "current_cost_ratio_mean_eval_only": float(
                np.mean([float(row["current_cost_ratio_eval_only"]) for row in group])
            ),
            "current_cost_ratio_max_eval_only": max_cost,
        }
        candidate["gate_pass"] = bool(
            len(group) == cfg.worlds_per_suite * len(J.DEV_SUITES)
            and all(bool(row["path_valid"]) for row in group)
            and max_cost <= cfg.cost_ceiling + 1.0e-12
            and high < 0.0
            and candidate["negative_suites"] >= cfg.required_negative_suites
        )
        pooled.append(candidate)
        if candidate["gate_pass"]:
            passing.append(candidate)
    selected = (
        min(
            passing,
            key=lambda row: (
                float(row["current_expansions_mean"]),
                float(row["current_cost_ratio_max_eval_only"]),
                float(row["alpha"]),
            ),
        )
        if passing
        else None
    )
    verdict = {
        "verdict": (
            "local_backup_scale_selected_requires_fresh_confirmation"
            if selected is not None
            else "local_backup_scale_amplification_rejected"
        ),
        "gate_pass": selected is not None,
        "selected_candidate": selected,
        "passing_candidates": len(passing),
        "fixed_model": {
            "family": "suite_balanced_flat_mlp",
            "iteration": 8,
            "sensor_radius_frac": cfg.sensor_radius_frac,
            "backup_applications": 1,
        },
        "authorization": (
            "confirm_selected_alpha_on_seed_offset_15000000"
            if selected is not None
            else "add_online_search_history_state"
        ),
    }
    return cells, pooled, verdict


def run(cfg: ScaleConfig) -> Dict[str, Any]:
    bundles, cohort_records, caches, cohort_verification = build_calibration_cohort(cfg)
    rows, diagnostics, checkpoint = evaluate(cfg, bundles)
    cells, pooled, verdict = summarize(cfg, rows)
    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    cohort_path = C13.write_json(
        result_dir / "calibration_cohort.json",
        {
            "seed_offset": cfg.seed_offset,
            "records": cohort_records,
            "verification": cohort_verification,
        },
    )
    raw_path = C13.write_csv(result_dir / "calibration_raw.csv", rows)
    diagnostics_path = C13.write_csv(
        result_dir / "provider_diagnostics.csv", diagnostics
    )
    cells_path = C13.write_csv(result_dir / "alpha_suite_summary.csv", cells)
    pooled_path = C13.write_csv(result_dir / "alpha_pooled_summary.csv", pooled)
    verdict_path = C13.write_json(result_dir / "gate_verdict.json", verdict)
    verification = {
        "device": cfg.device,
        "cohort": cohort_verification,
        "rows": len(rows),
        "expected_rows": int(
            cfg.worlds_per_suite
            * len(J.DEV_SUITES)
            * len(C13.parse_float_csv(cfg.alphas))
        ),
        "invalid_paths": int(sum(not bool(row["path_valid"]) for row in rows)),
        "runtime_information": (
            "current_goal_geometry_bounded_rays_one_hop_actions_plus_"
            "radius_bounded_local_subgraph_and_frozen_exit_values"
        ),
        "full_map_runtime_input": False,
        "shortest_path_target": False,
        "training_target_reads_dist_to_goal": False,
        "confirmation_seed_offset": 15_000_000,
    }
    verification["integrity_pass"] = bool(
        verification["rows"] == verification["expected_rows"]
        and verification["invalid_paths"] == 0
        and all(value == 0 for value in cohort_verification["overlap"].values())
    )
    verification_path = C13.write_json(
        result_dir / "verification.json", verification
    )
    manifest_path = C13.write_json(
        Path(cfg.out_dir) / "manifest.json",
        {
            "experiment": "C13-L local-backup alpha calibration",
            "config": asdict(cfg),
            "checkpoint": str(checkpoint),
            "full_map_runtime_input": False,
            "fresh_confirmation_required": bool(verdict["gate_pass"]),
            "outputs": {
                "cohort": str(cohort_path),
                "raw": str(raw_path),
                "pooled": str(pooled_path),
                "verdict": str(verdict_path),
                "verification": str(verification_path),
            },
        },
    )
    inputs = {
        "implementation": Path(__file__).resolve(),
        "preregistration": Path(cfg.preregistration),
        "checkpoint": checkpoint,
        "c13k_verdict": (
            Path(__file__).resolve().parent
            / "runs"
            / "c13_local_bellman_integration"
            / "results"
            / "gate_verdict.json"
        ),
        "c7_field_hrm": Path(cfg.c7_run_dir) / "checkpoints" / "c6_heatmap__hrm.pt",
        "c7_scalar_hrm": Path(cfg.c7_run_dir) / "checkpoints" / "avgbase__hrm.pt",
        **{f"feature_cache_{index:02d}": path for index, path in enumerate(caches)},
    }
    outputs = {
        "cohort": cohort_path,
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
        },
    )
    if not verification["integrity_pass"]:
        raise RuntimeError("C13-L verification failed")
    print(
        f"[c13l] {verdict['verdict']} selected={verdict['selected_candidate']} "
        f"-> {verdict_path}",
        flush=True,
    )
    return {"verdict": verdict, "verification": verification, "integrity": integrity_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-L local backup alpha calibration")
    parser.add_argument("--out-dir", default=ScaleConfig.out_dir)
    parser.add_argument("--device", default=ScaleConfig.device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ScaleConfig(out_dir=args.out_dir, device=args.device)
    resolve_paths(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
