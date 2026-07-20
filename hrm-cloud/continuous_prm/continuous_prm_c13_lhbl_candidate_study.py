#!/usr/bin/env python3
"""C13-H candidate study on the two already-observed paired cohorts.

This is development-only.  It separates the learned state representation from
the runtime limited-horizon Bellman operator and from the Euclidean exit-stub
control, using the repaired same-mode FOCAL integration throughout.
"""
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
import continuous_prm_c13_lhbl_focal_matched_control_diagnostic as F
import continuous_prm_c13_lhbl_generated_v3 as H
import continuous_prm_c13_lhbl_replication as R
import continuous_prm_c13_shared_queue as Q
import continuous_prm_c13_state_heuristic as C13
import continuous_prm_c7_hard_maps as M7


@dataclass
class CandidateStudyConfig:
    source_run_dir: str = "runs/c13_lhbl_flat_48w"
    cohort_a_dir: str = "runs/c13_lhbl_fresh_192_211"
    cohort_b_dir: str = "runs/c13_lhbl_focal_fresh2"
    out_dir: str = "runs/c13_lhbl_candidate_study"
    iterations: str = "4,5,6,7,8"
    alphas: str = "0.10,0.25,0.50,0.75,1.00"
    mode: str = "fhat"
    focal_w: float = 1.10
    budget_factor: float = 2.0
    required_win_fraction: float = 0.80
    bootstrap_replicates: int = 20_000
    bootstrap_seed: int = 413_337
    device: str = "auto"


def load_checkpoints(
    cfg: CandidateStudyConfig, device: torch.device
) -> Tuple[Dict[int, torch.nn.Module], H.LHBLConfig, I.StudyConfig, Dict[int, Path]]:
    iterations = R.parse_int_csv(cfg.iterations)
    source_run = Path(cfg.source_run_dir)
    integrity = H._read_json(source_run / "integrity.json")
    models: Dict[int, torch.nn.Module] = {}
    paths: Dict[int, Path] = {}
    training_cfg: H.LHBLConfig | None = None
    model_cfg: I.StudyConfig | None = None
    for iteration in iterations:
        name = f"flat_mlp_iteration_{iteration:02d}.pt"
        path = source_run / "checkpoints" / name
        expected = integrity["checkpoints"][name]["sha256"]
        if S.file_sha256(path) != expected:
            raise RuntimeError(f"checkpoint integrity mismatch: {name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("shortest_path_target") is not False
            or payload.get("model_name") != "flat_mlp"
            or int(payload.get("iteration", -1)) != iteration
        ):
            raise RuntimeError(f"checkpoint provenance mismatch: {name}")
        current_training_cfg = H.LHBLConfig(**payload["lhbl_config"])
        current_model_cfg = I.StudyConfig(**payload["model_config"])
        if training_cfg is None:
            training_cfg = current_training_cfg
            model_cfg = current_model_cfg
        elif H.state_config(training_cfg) != H.state_config(current_training_cfg):
            raise RuntimeError("checkpoint state configurations differ")
        model = H.build_lhbl_model("flat_mlp", current_model_cfg)
        model.load_state_dict(payload["model"], strict=True)
        models[iteration] = model.to(device).eval()
        paths[iteration] = path
    if training_cfg is None or model_cfg is None:
        raise RuntimeError("no checkpoints loaded")
    return models, training_cfg, model_cfg, paths


def replay_cohort(
    cohort_label: str,
    cohort_path: Path,
    local_cfg: C13.LocalStateConfig,
) -> List[H.WorldBundle]:
    payload = H._read_json(cohort_path)
    M7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    bundles: List[H.WorldBundle] = []
    records = sorted(
        payload["roadmaps"], key=lambda row: (int(row["density"]), int(row["world_index"]))
    )
    for row in records:
        world_seed = int(row["world_seed"])
        density = int(row["density"])
        world = C.build_world(specs["C_hard_maze"], world_seed, 0.45)
        if world is None:
            raise RuntimeError(f"could not replay {cohort_label}/{world_seed}")
        roadmap = C.build_prm(
            world,
            C.RoadmapConfig(n_nodes=density, k_neighbors=7),
            seed=int(row["roadmap_seed"]),
        )
        if roadmap is None:
            raise RuntimeError(f"could not replay {cohort_label}/{world_seed}/{density}")
        if (
            len(roadmap.points) != int(row["nodes"])
            or sum(len(group) for group in roadmap.adj) // 2 != int(row["edges"])
        ):
            raise RuntimeError("candidate-study roadmap replay mismatch")
        features = C13.make_local_state_features(
            world, roadmap.points, roadmap.adj, local_cfg
        )
        bundles.append(
            H.WorldBundle(
                split=cohort_label,
                suite="C_hard_maze",
                world_index=int(row["world_index"]),
                world_seed=world_seed,
                world=world,
                roadmap=roadmap,
                features=features,
            )
        )
    if len(bundles) != 24:
        raise RuntimeError(f"{cohort_label} replay has {len(bundles)} rather than 24 roadmaps")
    return bundles


def provider_ranks(
    bundle: H.WorldBundle,
    models: Mapping[int, torch.nn.Module],
    cfg: CandidateStudyConfig,
    training_cfg: H.LHBLConfig,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    roadmap = bundle.roadmap
    euclid = C13.euclidean_to_goal(roadmap.points, roadmap.points[1])
    oracle = np.asarray(roadmap.dist_to_goal, dtype=np.float64)
    connected = np.isfinite(oracle) & (oracle < C.INF / 10.0)
    radius = float(training_cfg.sensor_radius_frac) * float(bundle.world.side_len)
    providers: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    exit_started = time.perf_counter()
    exit_values, exit_diagnostics = H.limited_horizon_values(
        roadmap.points,
        roadmap.adj,
        roadmap.points[1],
        euclid,
        radius,
    )
    exit_seconds = float(time.perf_counter() - exit_started)
    providers.append(
        {
            "variant": "euclid_exit_stub",
            "iteration": 0,
            "rank": exit_values,
            "inference_seconds": 0.0,
            "backup_seconds": exit_seconds,
            "fallback_nodes": int(
                np.sum([bool(row["fallback"]) for row in exit_diagnostics])
            ),
        }
    )

    for iteration, model in sorted(models.items()):
        infer_started = time.perf_counter()
        prediction = I.predict_model(model, bundle.features, device)
        infer_seconds = float(time.perf_counter() - infer_started)
        learned = euclid + float(bundle.world.side_len) * prediction
        providers.append(
            {
                "variant": "model",
                "iteration": int(iteration),
                "rank": learned,
                "inference_seconds": infer_seconds,
                "backup_seconds": 0.0,
                "fallback_nodes": 0,
            }
        )
        backup_started = time.perf_counter()
        backed_up, backup_diagnostics = H.limited_horizon_values(
            roadmap.points,
            roadmap.adj,
            roadmap.points[1],
            learned,
            radius,
        )
        backup_seconds = float(time.perf_counter() - backup_started)
        providers.append(
            {
                "variant": "model_plus_local_backup",
                "iteration": int(iteration),
                "rank": backed_up,
                "inference_seconds": infer_seconds,
                "backup_seconds": backup_seconds,
                "fallback_nodes": int(
                    np.sum([bool(row["fallback"]) for row in backup_diagnostics])
                ),
            }
        )

    for provider in providers:
        rank = np.asarray(provider["rank"], dtype=np.float64)
        diagnostics.append(
            {
                "cohort": bundle.split,
                "density": int(len(roadmap.points)),
                "world_index": int(bundle.world_index),
                "world_seed": int(bundle.world_seed),
                "variant": provider["variant"],
                "iteration": int(provider["iteration"]),
                "inference_seconds": float(provider["inference_seconds"]),
                "backup_seconds": float(provider["backup_seconds"]),
                "fallback_nodes": int(provider["fallback_nodes"]),
                "rank_vs_oracle_spearman_eval_only": I.safe_spearman(
                    rank[connected], oracle[connected]
                ),
                "overestimate_rate_eval_only": float(
                    np.mean(rank[connected] > oracle[connected] + 1.0e-9)
                ),
            }
        )
    return providers, diagnostics


def evaluate(
    bundles: Sequence[H.WorldBundle],
    models: Mapping[int, torch.nn.Module],
    training_cfg: H.LHBLConfig,
    cfg: CandidateStudyConfig,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    alphas = C13.parse_float_csv(cfg.alphas)
    for bundle in bundles:
        roadmap = bundle.roadmap
        euclid = C13.euclidean_to_goal(roadmap.points, roadmap.points[1])
        optimal = float(roadmap.dist_to_goal[0])
        budget = int(math.ceil(cfg.budget_factor * len(roadmap.points)))
        baseline = F.focal_search_with_path(
            roadmap.adj, euclid, euclid, budget, cfg.focal_w, cfg.mode
        )
        baseline_path = Q.validate_path(
            roadmap.adj, baseline["path"], baseline["cost"]
        )
        providers, provider_diagnostics = provider_ranks(
            bundle, models, cfg, training_cfg, device
        )
        diagnostics.extend(provider_diagnostics)
        for provider in providers:
            provider_rank = np.asarray(provider["rank"], dtype=np.float64)
            for alpha in alphas:
                rank = euclid + float(alpha) * (provider_rank - euclid)
                started = time.perf_counter()
                result = F.focal_search_with_path(
                    roadmap.adj, euclid, rank, budget, cfg.focal_w, cfg.mode
                )
                search_seconds = float(time.perf_counter() - started)
                path = Q.validate_path(roadmap.adj, result["path"], result["cost"])
                cost = float(result["cost"])
                direct = C.astar_search(roadmap.adj, rank, len(roadmap.points))
                safety = bool(
                    not bool(baseline["found"])
                    or not bool(baseline_path["valid"])
                    or float(baseline["cost"]) > cfg.focal_w * optimal + 1.0e-9
                    or not bool(result["found"])
                    or not bool(path["valid"])
                    or not math.isfinite(cost)
                    or cost > cfg.focal_w * optimal + 1.0e-9
                    or cost
                    > cfg.focal_w * float(result["anchor_f_min_at_return"]) + 1.0e-9
                )
                rows.append(
                    {
                        "cohort": bundle.split,
                        "density": int(len(roadmap.points)),
                        "world_index": int(bundle.world_index),
                        "world_seed": int(bundle.world_seed),
                        "variant": provider["variant"],
                        "iteration": int(provider["iteration"]),
                        "alpha": float(alpha),
                        "mode": cfg.mode,
                        "focal_w": float(cfg.focal_w),
                        "found": bool(result["found"]),
                        "path_valid": bool(path["valid"]),
                        "cost_ratio_eval_only": cost / optimal,
                        "bound_violation_eval_only": bool(
                            cost > cfg.focal_w * optimal + 1.0e-9
                        ),
                        "expansions": int(result["expansions"]),
                        "max_expansions_per_state": int(
                            result["max_expansions_per_state"]
                        ),
                        "euclid_control_expansions": int(baseline["expansions"]),
                        "delta_vs_euclid_control": int(result["expansions"])
                        - int(baseline["expansions"]),
                        "direct_astar_expansions": int(direct["expansions"]),
                        "direct_astar_cost_ratio_eval_only": float(
                            direct["cost"] / optimal
                        ),
                        "inference_seconds": float(provider["inference_seconds"]),
                        "backup_seconds": float(provider["backup_seconds"]),
                        "search_seconds": search_seconds,
                        "fallback_nodes": int(provider["fallback_nodes"]),
                        "safety_failure": safety,
                    }
                )
    return rows, diagnostics


def summarize_cells(
    rows: Sequence[Mapping[str, Any]], cfg: CandidateStudyConfig
) -> List[Dict[str, Any]]:
    grouped: DefaultDict[Tuple[str, int, float, str, int], List[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[
            (
                str(row["variant"]),
                int(row["iteration"]),
                float(row["alpha"]),
                str(row["cohort"]),
                int(row["density"]),
            )
        ].append(row)
    summaries: List[Dict[str, Any]] = []
    for (variant, iteration, alpha, cohort, density), group in sorted(grouped.items()):
        delta = np.asarray(
            [float(row["delta_vs_euclid_control"]) for row in group],
            dtype=np.float64,
        )
        required = int(math.ceil(cfg.required_win_fraction * len(group)))
        wins = int(np.sum(delta < 0.0))
        safety = int(np.sum([bool(row["safety_failure"]) for row in group]))
        low, high = R.bootstrap_mean_ci(
            delta,
            cfg.bootstrap_replicates,
            cfg.bootstrap_seed
            + density
            + iteration * 100
            + int(round(alpha * 10_000))
            + len(variant) * 100_000
            + len(cohort) * 1_000_000,
        )
        summaries.append(
            {
                "variant": variant,
                "iteration": int(iteration),
                "alpha": float(alpha),
                "cohort": cohort,
                "density": int(density),
                "worlds": int(len(group)),
                "required_wins": int(required),
                "gate_pass": bool(
                    safety == 0
                    and wins >= required
                    and float(np.mean(delta)) < 0.0
                ),
                "safety_failures": int(safety),
                "delta_mean": float(np.mean(delta)),
                "delta_ci95_low": low,
                "delta_ci95_high": high,
                "wins": wins,
                "ties": int(np.sum(delta == 0.0)),
                "losses": int(np.sum(delta > 0.0)),
                "expansions_mean": float(
                    np.mean([float(row["expansions"]) for row in group])
                ),
                "euclid_control_expansions_mean": float(
                    np.mean([float(row["euclid_control_expansions"]) for row in group])
                ),
                "cost_ratio_max_eval_only": float(
                    np.max([float(row["cost_ratio_eval_only"]) for row in group])
                ),
            }
        )
    return summaries


def summarize_pooled(
    rows: Sequence[Mapping[str, Any]], cfg: CandidateStudyConfig
) -> List[Dict[str, Any]]:
    grouped: DefaultDict[Tuple[str, int, float, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["variant"]),
                int(row["iteration"]),
                float(row["alpha"]),
                int(row["density"]),
            )
        ].append(row)
    summaries: List[Dict[str, Any]] = []
    for (variant, iteration, alpha, density), group in sorted(grouped.items()):
        delta = np.asarray(
            [float(row["delta_vs_euclid_control"]) for row in group],
            dtype=np.float64,
        )
        required = int(math.ceil(cfg.required_win_fraction * len(group)))
        wins = int(np.sum(delta < 0.0))
        safety = int(np.sum([bool(row["safety_failure"]) for row in group]))
        low, high = R.bootstrap_mean_ci(
            delta,
            cfg.bootstrap_replicates,
            cfg.bootstrap_seed
            + 50_000_000
            + density
            + iteration * 100
            + int(round(alpha * 10_000))
            + len(variant) * 100_000,
        )
        summaries.append(
            {
                "variant": variant,
                "iteration": int(iteration),
                "alpha": float(alpha),
                "density": int(density),
                "worlds": int(len(group)),
                "required_wins": int(required),
                "gate_pass": bool(
                    safety == 0
                    and wins >= required
                    and float(np.mean(delta)) < 0.0
                ),
                "safety_failures": int(safety),
                "delta_mean": float(np.mean(delta)),
                "delta_ci95_low": low,
                "delta_ci95_high": high,
                "wins": wins,
                "ties": int(np.sum(delta == 0.0)),
                "losses": int(np.sum(delta > 0.0)),
                "cost_ratio_max_eval_only": float(
                    np.max([float(row["cost_ratio_eval_only"]) for row in group])
                ),
            }
        )
    return summaries


def select_candidate(
    cell_summaries: Sequence[Mapping[str, Any]],
    pooled_summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    cell_groups: DefaultDict[Tuple[str, int, float], List[Mapping[str, Any]]] = defaultdict(list)
    pooled_groups: DefaultDict[Tuple[str, int, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in cell_summaries:
        cell_groups[(str(row["variant"]), int(row["iteration"]), float(row["alpha"]))].append(row)
    for row in pooled_summaries:
        pooled_groups[(str(row["variant"]), int(row["iteration"]), float(row["alpha"]))].append(row)
    hard: List[Dict[str, Any]] = []
    pooled: List[Dict[str, Any]] = []
    for key, density_rows in pooled_groups.items():
        cells = cell_groups[key]
        if {int(row["density"]) for row in density_rows} != {192, 211}:
            continue
        candidate = {
            "variant": key[0],
            "iteration": key[1],
            "alpha": key[2],
            "combined_delta_mean": float(
                np.mean([float(row["delta_mean"]) for row in density_rows])
            ),
            "pooled_density_summaries": [dict(row) for row in density_rows],
            "cell_summaries": [dict(row) for row in cells],
        }
        if len(cells) == 4 and all(bool(row["gate_pass"]) for row in cells):
            hard.append(candidate)
        elif (
            all(bool(row["gate_pass"]) for row in density_rows)
            and all(int(row["safety_failures"]) == 0 for row in cells)
            and all(float(row["delta_mean"]) < 0.0 for row in cells)
        ):
            pooled.append(candidate)
    tier = "all_four_cells" if hard else "pooled_24_worlds" if pooled else "none"
    eligible = hard if hard else pooled
    selected = (
        min(
            eligible,
            key=lambda row: (
                float(row["combined_delta_mean"]),
                str(row["variant"]),
                int(row["iteration"]),
                float(row["alpha"]),
            ),
        )
        if eligible
        else None
    )
    return {
        "verdict": (
            "candidate_selected_requires_fresh3"
            if selected is not None
            else "no_candidate_passed_combined_development_gate"
        ),
        "selection_tier": tier,
        "selected_candidate": selected,
        "eligible_candidates": int(len(eligible)),
        "fresh_replication_required": selected is not None,
        "authorization": (
            "replicate_selected_candidate_on_untouched_fresh3"
            if selected is not None
            else "revise_training_objective_or_information_boundary"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-H combined candidate study")
    for field in CandidateStudyConfig.__dataclass_fields__.values():
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=type(field.default),
            default=field.default,
        )
    return parser.parse_args()


def resolve_paths(cfg: CandidateStudyConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    for field_name in ("source_run_dir", "cohort_a_dir", "cohort_b_dir", "out_dir"):
        default = getattr(CandidateStudyConfig, field_name)
        if getattr(cfg, field_name) == default:
            setattr(cfg, field_name, str(script_dir / default))


def main() -> None:
    cfg = CandidateStudyConfig(**vars(parse_args()))
    resolve_paths(cfg)
    if cfg.mode != "fhat" or not math.isclose(cfg.focal_w, 1.10):
        raise ValueError("candidate study is locked to reopening fhat at w=1.10")
    device = R.resolve_device(cfg.device)
    models, training_cfg, model_cfg, checkpoint_paths = load_checkpoints(cfg, device)
    cohort_a_path = Path(cfg.cohort_a_dir) / "results" / "fresh_cohort.json"
    cohort_b_path = Path(cfg.cohort_b_dir) / "results" / "fresh2_cohort.json"
    local_cfg = H.state_config(training_cfg)
    bundles_a = replay_cohort("cohort_a", cohort_a_path, local_cfg)
    bundles_b = replay_cohort("cohort_b", cohort_b_path, local_cfg)
    seeds_a = {bundle.world_seed for bundle in bundles_a}
    seeds_b = {bundle.world_seed for bundle in bundles_b}
    if seeds_a & seeds_b:
        raise RuntimeError("candidate-study cohorts overlap")
    rows, diagnostics = evaluate(
        bundles_a + bundles_b, models, training_cfg, cfg, device
    )
    cell_summaries = summarize_cells(rows, cfg)
    pooled_summaries = summarize_pooled(rows, cfg)
    verdict = select_candidate(cell_summaries, pooled_summaries)

    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    raw_path = C13.write_csv(result_dir / "candidate_raw.csv", rows)
    diagnostics_path = C13.write_csv(
        result_dir / "provider_diagnostics.csv", diagnostics
    )
    cell_path = C13.write_csv(result_dir / "candidate_cell_summary.csv", cell_summaries)
    pooled_path = C13.write_csv(
        result_dir / "candidate_pooled_summary.csv", pooled_summaries
    )
    verdict_path = C13.write_json(result_dir / "gate_verdict.json", verdict)
    expected_providers = 1 + 2 * len(models)
    expected_rows = 48 * expected_providers * len(C13.parse_float_csv(cfg.alphas))
    verification = {
        "device": str(device),
        "cohorts": ["cohort_a", "cohort_b"],
        "cohort_seed_overlap": int(len(seeds_a & seeds_b)),
        "roadmaps": int(len(bundles_a) + len(bundles_b)),
        "providers_per_roadmap": int(expected_providers),
        "rows": int(len(rows)),
        "expected_rows": int(expected_rows),
        "provider_diagnostic_rows": int(len(diagnostics)),
        "expected_provider_diagnostic_rows": int(48 * expected_providers),
        "safety_failures": int(np.sum([bool(row["safety_failure"]) for row in rows])),
        "fallback_nodes": int(np.sum([int(row["fallback_nodes"]) for row in rows])),
        "runtime_information": {
            "model": "current_goal_geometry_bounded_rays_one_hop_actions",
            "model_plus_local_backup": "radius_bounded_local_subgraph_plus_frozen_exit_state_values",
            "euclid_exit_stub": "radius_bounded_local_subgraph_plus_euclidean_exit_values",
        },
        "shortest_path_target": False,
        "training_target_reads_dist_to_goal": False,
        "fresh_replication_required": bool(verdict["fresh_replication_required"]),
    }
    verification_path = C13.write_json(
        result_dir / "verification.json", verification
    )
    if (
        verification["cohort_seed_overlap"]
        or verification["rows"] != verification["expected_rows"]
        or verification["provider_diagnostic_rows"]
        != verification["expected_provider_diagnostic_rows"]
        or verification["safety_failures"]
    ):
        raise RuntimeError("candidate study verification failed")

    inputs = {
        "implementation": Path(__file__).resolve(),
        "training_implementation": Path(H.__file__).resolve(),
        "matched_focal_implementation": Path(F.__file__).resolve(),
        "source_run_manifest": Path(cfg.source_run_dir) / "manifest.json",
        "cohort_a": cohort_a_path,
        "cohort_b": cohort_b_path,
        **{f"checkpoint_{iteration}": path for iteration, path in checkpoint_paths.items()},
    }
    outputs = {
        "raw": raw_path,
        "provider_diagnostics": diagnostics_path,
        "cell_summary": cell_path,
        "pooled_summary": pooled_path,
        "gate": verdict_path,
        "verification": verification_path,
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
    manifest_path = C13.write_json(
        Path(cfg.out_dir) / "manifest.json",
        {
            "experiment": "C13-H combined-cohort representation and runtime-backup study",
            "config": asdict(cfg),
            "source_model_config": asdict(model_cfg),
            "cohort_role": "two_already_observed_cohorts_development_only",
            "selection_policy": "all_four_cells_else_pooled_24_worlds_then_fresh3",
            "verdict": verdict,
            "outputs": {name: str(path) for name, path in outputs.items()},
            "integrity": str(integrity_path),
        },
    )
    print(f"verdict={verdict['verdict']}")
    print(f"selection_tier={verdict['selection_tier']}")
    print(f"selected={verdict['selected_candidate']}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
