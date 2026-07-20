#!/usr/bin/env python3
"""C13-J suite-balanced current-state LHBL training and development gate."""
from __future__ import annotations

import argparse
import csv
import math
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
import continuous_prm_c13_state_heuristic as C13
import continuous_prm_c7_hard_maps as M7
import continuous_prm_c7_integration_compare as C7


TRAIN_SUITES = ("C_hard_maze", "C_hard_rooms", "C_hard_spiral")
DEV_SUITES = X.ALL_SUITES
PREREGISTRATION = (
    "../../docs/experiments/continuous/c13/design/"
    "2026-07-17-c13j-multisuite-current-state-training.md"
)


@dataclass
class MultiSuiteConfig:
    study_dir: str = "runs/c13_identifiability"
    c7_run_dir: str = "runs/c7_local"
    c13i_run_dir: str = "runs/c13_lhbl_c7_comparison"
    out_dir: str = "runs/c13_lhbl_multisuite"
    preregistration: str = PREREGISTRATION
    train_worlds_per_suite: int = 32
    validation_worlds_per_suite: int = 8
    development_worlds_per_suite: int = 4
    roadmap_nodes: int = 192
    roadmap_k: int = 7
    max_world_retries: int = 200
    cohort_seed: int = 17_413
    train_seed_offset: int = 0
    validation_seed_offset: int = 5_000_000
    development_seed_offset: int = 10_000_000
    model_seed: int = 17_413
    outer_iterations: int = 8
    inner_epochs: int = 5
    batch_size: int = 128
    hidden_dim: int = 64
    lr: float = 5.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    sensor_radius_frac: float = 0.20
    num_rays: int = 32
    ray_steps: int = 32
    max_neighbors: int = 24
    max_norm_residual: float = 4.0
    alphas: str = "0.25,0.50,0.75,1.00"
    focal_w: float = 1.10
    budget_factor: float = 2.0
    cost_ceiling: float = 1.10
    required_negative_suites: int = 4
    bootstrap_replicates: int = 20_000
    bootstrap_seed: int = 713_337
    device: str = "cpu"


def resolve_paths(cfg: MultiSuiteConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    for field_name in (
        "study_dir",
        "c7_run_dir",
        "c13i_run_dir",
        "out_dir",
        "preregistration",
    ):
        path = Path(getattr(cfg, field_name))
        if not path.is_absolute():
            setattr(cfg, field_name, str((script_dir / path).resolve()))


def local_config(cfg: MultiSuiteConfig) -> C13.LocalStateConfig:
    return C13.LocalStateConfig(
        sensor_radius_frac=float(cfg.sensor_radius_frac),
        num_rays=int(cfg.num_rays),
        ray_steps=int(cfg.ray_steps),
        max_neighbors=int(cfg.max_neighbors),
    )


def training_config(cfg: MultiSuiteConfig) -> H.LHBLConfig:
    return H.LHBLConfig(
        study_dir=cfg.study_dir,
        out_dir=cfg.out_dir,
        models="flat_mlp",
        train_worlds=int(cfg.train_worlds_per_suite * len(TRAIN_SUITES)),
        validation_worlds=int(
            cfg.validation_worlds_per_suite * len(TRAIN_SUITES)
        ),
        outer_iterations=int(cfg.outer_iterations),
        inner_epochs=int(cfg.inner_epochs),
        batch_size=int(cfg.batch_size),
        hidden_dim=int(cfg.hidden_dim),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        grad_clip=float(cfg.grad_clip),
        sensor_radius_frac=float(cfg.sensor_radius_frac),
        num_rays=int(cfg.num_rays),
        ray_steps=int(cfg.ray_steps),
        max_neighbors=int(cfg.max_neighbors),
        max_norm_residual=float(cfg.max_norm_residual),
        alphas=str(cfg.alphas),
        focal_ws=str(cfg.focal_w),
        primary_w=float(cfg.focal_w),
        required_win_fraction=0.80,
        budget_factor=float(cfg.budget_factor),
        seed=int(cfg.model_seed),
        device=str(cfg.device),
    )


def _cache_path(cfg: MultiSuiteConfig, split: str, suite: str, world_seed: int) -> Path:
    return (
        Path(cfg.out_dir)
        / "feature_cache"
        / split
        / suite
        / f"{int(world_seed)}.npz"
    )


def _save_cache(path: Path, features: np.ndarray, nodes: int, edges: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.asarray(features, dtype=np.float32),
        nodes=np.asarray([int(nodes)], dtype=np.int64),
        edges=np.asarray([int(edges)], dtype=np.int64),
    )


def _load_cache(path: Path, nodes: int, edges: int, state_cfg: C13.LocalStateConfig) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        cached_nodes = int(payload["nodes"][0])
        cached_edges = int(payload["edges"][0])
    expected_shape = (int(nodes), int(state_cfg.seq_len), int(state_cfg.token_dim))
    if cached_nodes != int(nodes) or cached_edges != int(edges) or features.shape != expected_shape:
        raise RuntimeError(f"feature cache provenance mismatch: {path}")
    return features


def build_balanced_bundles(
    cfg: MultiSuiteConfig,
    split: str,
    suites: Sequence[str],
    worlds_per_suite: int,
    seed_offset: int,
) -> Tuple[List[H.WorldBundle], List[Dict[str, Any]], List[Path]]:
    M7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    state_cfg = local_config(cfg)
    roadmap_cfg = C.RoadmapConfig(
        n_nodes=int(cfg.roadmap_nodes), k_neighbors=int(cfg.roadmap_k)
    )
    bundles: List[H.WorldBundle] = []
    records: List[Dict[str, Any]] = []
    caches: List[Path] = []
    for suite in suites:
        suite_idx = X.ALL_SUITES.index(suite)
        accepted = 0
        candidates = int(worlds_per_suite) * int(cfg.max_world_retries)
        for _, world, world_seed in C13.iter_worlds(
            specs[suite],
            suite_idx,
            candidates,
            int(cfg.cohort_seed) + int(seed_offset),
            retry=int(cfg.max_world_retries),
        ):
            roadmap_seed = int(world_seed) + 17
            roadmap = C.build_prm(world, roadmap_cfg, seed=roadmap_seed)
            if roadmap is None or not roadmap.connected_to_goal[0]:
                continue
            edges = int(sum(len(group) for group in roadmap.adj) // 2)
            cache = _cache_path(cfg, split, suite, world_seed)
            if cache.exists():
                features = _load_cache(
                    cache, len(roadmap.points), edges, state_cfg
                )
                cache_status = "reused"
            else:
                features = C13.make_local_state_features(
                    world, roadmap.points, roadmap.adj, state_cfg
                )
                _save_cache(cache, features, len(roadmap.points), edges)
                cache_status = "created"
            global_index = len(bundles)
            bundles.append(
                H.WorldBundle(
                    split=split,
                    suite=suite,
                    world_index=global_index,
                    world_seed=int(world_seed),
                    world=world,
                    roadmap=roadmap,
                    features=features,
                )
            )
            records.append(
                {
                    "split": split,
                    "suite": suite,
                    "suite_world_index": int(accepted),
                    "global_world_index": int(global_index),
                    "world_seed": int(world_seed),
                    "roadmap_seed": int(roadmap_seed),
                    "nodes": int(len(roadmap.points)),
                    "edges": edges,
                    "cache": str(cache),
                    "cache_status": cache_status,
                    "cache_sha256": S.file_sha256(cache),
                }
            )
            caches.append(cache)
            accepted += 1
            print(
                f"[c13j] {split}/{suite} {accepted}/{worlds_per_suite} "
                f"cache={cache_status}",
                flush=True,
            )
            if accepted >= int(worlds_per_suite):
                break
        if accepted != int(worlds_per_suite):
            raise RuntimeError(
                f"{split}/{suite} under-filled: {accepted}/{worlds_per_suite}"
            )
    return bundles, records, caches


def c13i_seed_set(cfg: MultiSuiteConfig) -> set[int]:
    raw = Path(cfg.c13i_run_dir) / "results" / "c13i_raw.csv"
    if not raw.exists():
        raise RuntimeError("C13-I raw rows are required for seed-overlap exclusion")
    with raw.open(newline="", encoding="utf-8") as handle:
        return {int(row["world_seed"]) for row in csv.DictReader(handle)}


def verify_cohorts(
    cfg: MultiSuiteConfig,
    train: Sequence[H.WorldBundle],
    validation: Sequence[H.WorldBundle],
    development: Sequence[H.WorldBundle],
) -> Dict[str, Any]:
    groups = {
        "train": {int(bundle.world_seed) for bundle in train},
        "validation": {int(bundle.world_seed) for bundle in validation},
        "development": {int(bundle.world_seed) for bundle in development},
    }
    expected = {
        "train": int(cfg.train_worlds_per_suite * len(TRAIN_SUITES)),
        "validation": int(
            cfg.validation_worlds_per_suite * len(TRAIN_SUITES)
        ),
        "development": int(
            cfg.development_worlds_per_suite * len(DEV_SUITES)
        ),
    }
    overlap = {
        "train_validation": len(groups["train"] & groups["validation"]),
        "train_development": len(groups["train"] & groups["development"]),
        "validation_development": len(
            groups["validation"] & groups["development"]
        ),
    }
    old = c13i_seed_set(cfg)
    c13i_overlap = {
        name: len(seeds & old) for name, seeds in groups.items()
    }
    if any(len(groups[name]) != expected[name] for name in groups):
        raise RuntimeError("cohort seed uniqueness failed")
    if any(overlap.values()) or any(c13i_overlap.values()):
        raise RuntimeError("cohort seed overlap failed")
    return {
        "unique_seeds": {name: len(value) for name, value in groups.items()},
        "expected_seeds": expected,
        "cross_split_overlap": overlap,
        "c13i_overlap": c13i_overlap,
    }


def _load_checkpoint_models(
    checkpoint_paths: Sequence[Path], device: torch.device
) -> Dict[int, torch.nn.Module]:
    models: Dict[int, torch.nn.Module] = {}
    for path in checkpoint_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("model_name") != "flat_mlp":
            raise RuntimeError(f"unexpected model family in {path}")
        iteration = int(payload["iteration"])
        model_cfg = I.StudyConfig(**payload["model_config"])
        model = H.build_lhbl_model("flat_mlp", model_cfg)
        model.load_state_dict(payload["model"], strict=True)
        models[iteration] = model.to(device).eval()
    if sorted(models) != list(range(1, max(models) + 1)):
        raise RuntimeError("checkpoint iterations are incomplete")
    return models


def _load_c7_comparators(
    cfg: MultiSuiteConfig, device: torch.device
) -> Dict[str, Any]:
    c7_cfg = C7.apply_scale_preset(
        C7.C7Config(
            grid_size=64,
            roadmap_nodes=int(cfg.roadmap_nodes),
            roadmap_k=int(cfg.roadmap_k),
            eval_worlds=int(cfg.development_worlds_per_suite),
            seed=1234,
            scale="local",
            out_dir=cfg.c7_run_dir,
            cpu=True,
            scalar_backbones="hrm,onlstm",
            field_backbones="unet,onlstm,hrm",
        )
    )
    providers = C7._load_eval_providers(Path(cfg.c7_run_dir), c7_cfg, device)
    return {name: providers[name] for name in ("field_hrm", "scalar_hrm")}


def evaluate_development(
    cfg: MultiSuiteConfig,
    bundles: Sequence[H.WorldBundle],
    checkpoint_paths: Sequence[Path],
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    models = _load_checkpoint_models(checkpoint_paths, device)
    comparators = _load_c7_comparators(cfg, device)
    alphas = C13.parse_float_csv(cfg.alphas)
    rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    for ordinal, bundle in enumerate(bundles, start=1):
        roadmap = bundle.roadmap
        optimal = float(roadmap.dist_to_goal[0])
        euclid = C13.euclidean_to_goal(roadmap.points, roadmap.points[1])
        reference: Dict[str, Dict[str, Any]] = {}
        for name, provider in comparators.items():
            rank = np.asarray(provider.node_h(bundle.world, roadmap, 1), dtype=np.float64)
            result = C.astar_search(roadmap.adj, rank, len(roadmap.points))
            reference[name] = result
        for iteration, model in sorted(models.items()):
            prediction = np.asarray(
                I.predict_model(model, bundle.features, device), dtype=np.float64
            )
            learned = euclid + float(bundle.world.side_len) * prediction
            oracle = np.asarray(roadmap.dist_to_goal, dtype=np.float64)
            connected = np.isfinite(oracle) & (oracle < C.INF / 10.0)
            diagnostics.append(
                {
                    "suite": bundle.suite,
                    "world_index": int(bundle.world_index),
                    "world_seed": int(bundle.world_seed),
                    "iteration": int(iteration),
                    "prediction_mean": float(np.mean(prediction)),
                    "prediction_p95": float(np.percentile(prediction, 95)),
                    "rank_vs_oracle_spearman_eval_only": I.safe_spearman(
                        learned[connected], oracle[connected]
                    ),
                    "overestimate_rate_eval_only": float(
                        np.mean(learned[connected] > oracle[connected] + 1.0e-9)
                    ),
                }
            )
            for alpha in alphas:
                rank = euclid + float(alpha) * (learned - euclid)
                current = C.astar_search(
                    roadmap.adj, rank, len(roadmap.points)
                )
                if not current["found"]:
                    raise RuntimeError("connected development roadmap was not solved")
                rows.append(
                    {
                        "suite": bundle.suite,
                        "world_index": int(bundle.world_index),
                        "world_seed": int(bundle.world_seed),
                        "iteration": int(iteration),
                        "alpha": float(alpha),
                        "current_expansions": int(current["expansions"]),
                        "current_cost_ratio_eval_only": float(
                            current["cost"] / optimal
                        ),
                        "field_hrm_expansions": int(
                            reference["field_hrm"]["expansions"]
                        ),
                        "field_hrm_cost_ratio_eval_only": float(
                            reference["field_hrm"]["cost"] / optimal
                        ),
                        "delta_vs_field_hrm": int(current["expansions"])
                        - int(reference["field_hrm"]["expansions"]),
                        "scalar_hrm_expansions": int(
                            reference["scalar_hrm"]["expansions"]
                        ),
                        "scalar_hrm_cost_ratio_eval_only": float(
                            reference["scalar_hrm"]["cost"] / optimal
                        ),
                        "delta_vs_scalar_hrm": int(current["expansions"])
                        - int(reference["scalar_hrm"]["expansions"]),
                    }
                )
        print(
            f"[c13j] development comparison {ordinal}/{len(bundles)} "
            f"{bundle.suite}/{bundle.world_index}",
            flush=True,
        )
    return rows, diagnostics


def summarize_candidates(
    cfg: MultiSuiteConfig, rows: Sequence[Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    pooled: List[Dict[str, Any]] = []
    grouped: DefaultDict[Tuple[int, float, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["iteration"]), float(row["alpha"]), str(row["suite"]))].append(row)
    for (iteration, alpha, suite), group in sorted(grouped.items()):
        delta = np.asarray([float(row["delta_vs_field_hrm"]) for row in group])
        cells.append(
            {
                "iteration": iteration,
                "alpha": alpha,
                "suite": suite,
                "worlds": len(group),
                "delta_vs_field_hrm_mean": float(np.mean(delta)),
                "wins": int(np.sum(delta < 0.0)),
                "ties": int(np.sum(delta == 0.0)),
                "losses": int(np.sum(delta > 0.0)),
                "current_expansions_mean": float(
                    np.mean([float(row["current_expansions"]) for row in group])
                ),
                "field_hrm_expansions_mean": float(
                    np.mean([float(row["field_hrm_expansions"]) for row in group])
                ),
                "current_cost_ratio_max_eval_only": float(
                    np.max([float(row["current_cost_ratio_eval_only"]) for row in group])
                ),
            }
        )
    by_candidate: DefaultDict[Tuple[int, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_candidate[(int(row["iteration"]), float(row["alpha"]))].append(row)
    passing: List[Dict[str, Any]] = []
    for (iteration, alpha), group in sorted(by_candidate.items()):
        delta = np.asarray([float(row["delta_vs_field_hrm"]) for row in group])
        low, high = X._bootstrap_mean_ci(
            delta,
            cfg.bootstrap_replicates,
            cfg.bootstrap_seed + iteration * 10_000 + int(alpha * 1000),
        )
        suite_means = {
            cell["suite"]: float(cell["delta_vs_field_hrm_mean"])
            for cell in cells
            if int(cell["iteration"]) == iteration
            and math.isclose(float(cell["alpha"]), alpha)
        }
        negative_suites = int(sum(value < 0.0 for value in suite_means.values()))
        max_cost = float(
            np.max([float(row["current_cost_ratio_eval_only"]) for row in group])
        )
        candidate = {
            "iteration": int(iteration),
            "alpha": float(alpha),
            "worlds": int(len(group)),
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
                len(group) == cfg.development_worlds_per_suite * len(DEV_SUITES)
                and max_cost <= cfg.cost_ceiling + 1.0e-12
                and high < 0.0
                and negative_suites >= int(cfg.required_negative_suites)
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
                int(row["iteration"]),
                float(row["alpha"]),
            ),
        )
        if passing
        else None
    )
    verdict = {
        "verdict": (
            "multisuite_development_gate_pass_requires_fresh_confirmation"
            if selected is not None
            else "multisuite_distribution_only_intervention_failed"
        ),
        "gate_pass": selected is not None,
        "selected_candidate": selected,
        "passing_candidates": len(passing),
        "authorization": (
            "run_fixed_candidate_on_untouched_six_suite_seed_block"
            if selected is not None
            else "change_representation_or_online_integration_before_more_confirmation"
        ),
    }
    return cells, pooled, verdict


def _write_cohort_file(
    cfg: MultiSuiteConfig,
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    verification: Mapping[str, Any],
) -> Path:
    return C13.write_json(
        Path(cfg.out_dir) / "results" / "cohorts.json",
        {
            "seed_recipe": "c13_iter_worlds_with_global_c7_suite_index_and_roadmap_seed_plus_17",
            "cohort_seed": int(cfg.cohort_seed),
            "offsets": {
                "train": int(cfg.train_seed_offset),
                "validation": int(cfg.validation_seed_offset),
                "development": int(cfg.development_seed_offset),
            },
            "records": records,
            "verification": verification,
        },
    )


def run(cfg: MultiSuiteConfig, collect_only: bool = False) -> Dict[str, Any]:
    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    study_cfg, source_manifest = S.load_study(cfg.study_dir)
    train, train_records, train_caches = build_balanced_bundles(
        cfg,
        "train",
        TRAIN_SUITES,
        cfg.train_worlds_per_suite,
        cfg.train_seed_offset,
    )
    validation, validation_records, validation_caches = build_balanced_bundles(
        cfg,
        "validation",
        TRAIN_SUITES,
        cfg.validation_worlds_per_suite,
        cfg.validation_seed_offset,
    )
    development, development_records, development_caches = build_balanced_bundles(
        cfg,
        "development",
        DEV_SUITES,
        cfg.development_worlds_per_suite,
        cfg.development_seed_offset,
    )
    cohort_verification = verify_cohorts(cfg, train, validation, development)
    cohort_path = _write_cohort_file(
        cfg,
        {
            "train": train_records,
            "validation": validation_records,
            "development": development_records,
        },
        cohort_verification,
    )
    if collect_only:
        return {"cohorts": cohort_path, "verification": cohort_verification}

    device = H.resolve_device(cfg.device)
    if str(device) != "cpu":
        raise RuntimeError("C13-J is locked to CPU for reproducible comparison")
    lhbl_cfg = training_config(cfg)
    baseline_rows, baselines = H.build_baselines(
        development, [cfg.focal_w], cfg.budget_factor
    )
    histories, search_rows, prediction_rows, checkpoints = H.train_models(
        lhbl_cfg,
        study_cfg,
        train,
        validation,
        development,
        baselines,
        device,
    )
    dev_rows, dev_diagnostics = evaluate_development(
        cfg, development, checkpoints, device
    )
    cells, pooled, verdict = summarize_candidates(cfg, dev_rows)

    baseline_path = C13.write_csv(result_dir / "euclid_baselines.csv", baseline_rows)
    history_path = C13.write_csv(result_dir / "training_history.csv", histories)
    search_path = C13.write_csv(result_dir / "certified_search_diagnostics.csv", search_rows)
    prediction_path = C13.write_csv(
        result_dir / "training_prediction_diagnostics.csv", prediction_rows
    )
    dev_path = C13.write_csv(result_dir / "development_map_comparison_raw.csv", dev_rows)
    dev_diag_path = C13.write_csv(
        result_dir / "development_prediction_diagnostics.csv", dev_diagnostics
    )
    cells_path = C13.write_csv(result_dir / "candidate_suite_summary.csv", cells)
    pooled_path = C13.write_csv(result_dir / "candidate_pooled_summary.csv", pooled)
    verdict_path = C13.write_json(result_dir / "gate_verdict.json", verdict)
    verification = {
        "device": str(device),
        "cohorts": cohort_verification,
        "train_worlds": len(train),
        "validation_worlds": len(validation),
        "development_worlds": len(development),
        "checkpoints": len(checkpoints),
        "expected_checkpoints": int(cfg.outer_iterations),
        "development_rows": len(dev_rows),
        "expected_development_rows": int(
            len(development)
            * cfg.outer_iterations
            * len(C13.parse_float_csv(cfg.alphas))
        ),
        "runtime_information": "current_goal_geometry_bounded_rays_one_hop_actions",
        "full_map_runtime_input": False,
        "shortest_path_target": False,
        "training_target_reads_dist_to_goal": False,
        "development_comparator_information": {
            "field_hrm": "complete_64x64_occupancy_goal_raster",
            "scalar_hrm": "global_obstacle_list_summaries_sectors_plus_rays_goal",
        },
    }
    verification["integrity_pass"] = bool(
        verification["checkpoints"] == verification["expected_checkpoints"]
        and verification["development_rows"]
        == verification["expected_development_rows"]
        and all(
            value == 0
            for value in cohort_verification["cross_split_overlap"].values()
        )
        and all(
            value == 0 for value in cohort_verification["c13i_overlap"].values()
        )
    )
    verification_path = C13.write_json(
        result_dir / "verification.json", verification
    )
    manifest = {
        "experiment": "C13-J suite-balanced current-state LHBL",
        "config": asdict(cfg),
        "lhbl_config": asdict(lhbl_cfg),
        "source_study_manifest": source_manifest,
        "runtime_scope": "current_goal_geometry_bounded_rays_one_hop_actions",
        "training_suites_match_c7": list(TRAIN_SUITES),
        "development_suites": list(DEV_SUITES),
        "fresh_confirmation_required": bool(verdict["gate_pass"]),
        "outputs": {
            "cohorts": str(cohort_path),
            "history": str(history_path),
            "development_raw": str(dev_path),
            "candidate_pooled": str(pooled_path),
            "verdict": str(verdict_path),
            "verification": str(verification_path),
        },
    }
    manifest_path = C13.write_json(Path(cfg.out_dir) / "manifest.json", manifest)

    cache_paths = train_caches + validation_caches + development_caches
    inputs = {
        "implementation": Path(__file__).resolve(),
        "preregistration": Path(cfg.preregistration),
        "source_study_manifest": Path(cfg.study_dir) / "manifest.json",
        "c13i_verdict": Path(cfg.c13i_run_dir) / "results" / "gate_verdict.json",
        "c13i_raw": Path(cfg.c13i_run_dir) / "results" / "c13i_raw.csv",
        "c7_field_hrm": Path(cfg.c7_run_dir) / "checkpoints" / "c6_heatmap__hrm.pt",
        "c7_scalar_hrm": Path(cfg.c7_run_dir) / "checkpoints" / "avgbase__hrm.pt",
        **{f"feature_cache_{index:03d}": path for index, path in enumerate(cache_paths)},
    }
    outputs = {
        "cohorts": cohort_path,
        "baselines": baseline_path,
        "history": history_path,
        "search": search_path,
        "training_predictions": prediction_path,
        "development_raw": dev_path,
        "development_predictions": dev_diag_path,
        "cells": cells_path,
        "pooled": pooled_path,
        "verdict": verdict_path,
        "verification": verification_path,
        "manifest": manifest_path,
        **{f"checkpoint_{index + 1:02d}": path for index, path in enumerate(checkpoints)},
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
        raise RuntimeError("C13-J verification failed")
    print(
        f"[c13j] {verdict['verdict']} gate_pass={verdict['gate_pass']} "
        f"selected={verdict['selected_candidate']} -> {verdict_path}",
        flush=True,
    )
    return {
        "verdict": verdict,
        "verification": verification,
        "integrity": integrity_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-J multisuite current-state LHBL")
    parser.add_argument("--mode", choices=("collect", "full"), default="full")
    parser.add_argument("--out-dir", default=MultiSuiteConfig.out_dir)
    parser.add_argument("--device", default=MultiSuiteConfig.device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MultiSuiteConfig(out_dir=args.out_dir, device=args.device)
    resolve_paths(cfg)
    if cfg.roadmap_nodes != 192 or cfg.roadmap_k != 7:
        raise ValueError("C13-J is locked to the C7 192/k7 roadmap")
    run(cfg, collect_only=args.mode == "collect")


if __name__ == "__main__":
    main()
