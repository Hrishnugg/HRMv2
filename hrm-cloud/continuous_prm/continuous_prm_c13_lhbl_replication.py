#!/usr/bin/env python3
"""Fresh paired-density replication for the selected C13-H checkpoint.

This runner is intentionally evaluation-only.  It binds the fixed development
selection to its recorded checkpoint and training implementation, generates a
new seed range, and evaluates the same current-observation model at 192 and 211
nodes.  No checkpoint, alpha, focal width, or gate is selected on fresh data.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

import continuous_prm_common as C
import continuous_prm_c13_certified_search as S
import continuous_prm_c13_identifiability as I
import continuous_prm_c13_lhbl_generated_v3 as H
import continuous_prm_c13_shared_queue as Q
import continuous_prm_c13_state_heuristic as C13
import continuous_prm_c7_hard_maps as M7


EXPECTED_TRAINING_RUNNER_SHA256 = (
    "c43b147b0b85a223f9169990710cee483e64dc0c9894d64e03cdb46cee83f39f"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "4925f630ecbad6e3d410240ab3da2a73d3ff96c748ee59fb4ab8344a9a65e501"
)


@dataclass
class ReplicationConfig:
    source_run_dir: str = "runs/c13_lhbl_flat_48w"
    checkpoint: str = (
        "runs/c13_lhbl_flat_48w/checkpoints/flat_mlp_iteration_06.pt"
    )
    out_dir: str = "runs/c13_lhbl_fresh_192_211"
    suite: str = "C_hard_maze"
    worlds: int = 12
    densities: str = "192,211"
    roadmap_k: int = 7
    seed: int = 1234
    fresh_seed_offset: int = 1_800_000
    max_world_retries: int = 200
    alpha: float = 1.0
    focal_w: float = 1.10
    budget_factor: float = 2.0
    required_win_fraction: float = 0.80
    bootstrap_replicates: int = 20_000
    bootstrap_seed: int = 91_337
    device: str = "auto"


def parse_int_csv(value: str) -> List[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or any(item <= 0 for item in values):
        raise ValueError("densities must be a nonempty unique list of positive integers")
    return values


def bootstrap_mean_ci(
    values: Sequence[float], replicates: int, seed: int
) -> Tuple[float, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(data) == 0 or not np.all(np.isfinite(data)):
        raise ValueError("bootstrap values must be nonempty and finite")
    if len(data) == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(data), size=(int(replicates), len(data)))
    means = np.mean(data[indices], axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def resolve_device(name: str) -> torch.device:
    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_bound_checkpoint(
    cfg: ReplicationConfig, device: torch.device
) -> Tuple[torch.nn.Module, H.LHBLConfig, I.StudyConfig, Dict[str, Any]]:
    checkpoint = Path(cfg.checkpoint)
    training_runner = Path(H.__file__).resolve()
    if S.file_sha256(training_runner) != EXPECTED_TRAINING_RUNNER_SHA256:
        raise RuntimeError("training-runner hash does not match the selected development run")
    if S.file_sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash does not match the selected development candidate")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("shortest_path_target") is not False:
        raise RuntimeError("checkpoint lacks the no-shortest-path-target provenance flag")
    if payload.get("model_name") != "flat_mlp" or int(payload.get("iteration", -1)) != 6:
        raise RuntimeError("checkpoint is not the fixed flat_mlp iteration-6 candidate")
    training_cfg = H.LHBLConfig(**payload["lhbl_config"])
    model_cfg = I.StudyConfig(**payload["model_config"])
    if int(model_cfg.seed) != int(cfg.seed):
        raise RuntimeError("replication seed base differs from the source study seed")
    model = H.build_lhbl_model(str(payload["model_name"]), model_cfg)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, training_cfg, model_cfg, payload


def source_seed_set(source_run_dir: Path) -> set[int]:
    cohorts = H._read_json(source_run_dir / "results" / "lhbl_cohorts.json")
    seeds = {
        int(row["world_seed"])
        for split in ("train", "validation")
        for row in cohorts[split]
    }
    raw_path = source_run_dir / "results" / "lhbl_search_raw.csv"
    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        seeds.update(int(row["world_seed"]) for row in rows)
    return seeds


def generate_paired_bundles(
    cfg: ReplicationConfig,
    local_cfg: C13.LocalStateConfig,
    forbidden_seeds: set[int],
) -> Tuple[Dict[int, List[H.WorldBundle]], List[Dict[str, Any]], Dict[str, Any]]:
    M7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    if cfg.suite not in specs:
        raise KeyError(f"unknown suite {cfg.suite!r}; have {sorted(specs)}")
    densities = parse_int_csv(cfg.densities)
    requested = int(cfg.worlds)
    bundles: Dict[int, List[H.WorldBundle]] = {density: [] for density in densities}
    records: List[Dict[str, Any]] = []
    candidates_seen = 0
    prefix_failures = 0
    overlap_count = 0
    candidate_limit = requested * int(cfg.max_world_retries)
    for _, world, world_seed in C13.iter_worlds(
        specs[cfg.suite],
        0,
        candidate_limit,
        int(cfg.seed) + int(cfg.fresh_seed_offset),
        retry=int(cfg.max_world_retries),
    ):
        candidates_seen += 1
        roadmaps: Dict[int, C.Roadmap] = {}
        build_seconds: Dict[int, float] = {}
        for density in densities:
            started = time.perf_counter()
            roadmap = C.build_prm(
                world,
                C.RoadmapConfig(
                    n_nodes=int(density), k_neighbors=int(cfg.roadmap_k)
                ),
                seed=int(world_seed) + 17,
            )
            build_seconds[density] = float(time.perf_counter() - started)
            if roadmap is not None:
                roadmaps[density] = roadmap
        if len(roadmaps) != len(densities):
            continue
        ordered = sorted(densities)
        for smaller, larger in zip(ordered, ordered[1:]):
            if not np.array_equal(
                roadmaps[smaller].points, roadmaps[larger].points[:smaller]
            ):
                prefix_failures += 1
        if prefix_failures:
            raise RuntimeError("paired-density point samples are not exact prefixes")
        world_index = len(bundles[ordered[0]])
        if int(world_seed) in forbidden_seeds:
            overlap_count += 1
        for density in densities:
            roadmap = roadmaps[density]
            feature_started = time.perf_counter()
            features = C13.make_local_state_features(
                world, roadmap.points, roadmap.adj, local_cfg
            )
            feature_seconds = float(time.perf_counter() - feature_started)
            bundles[density].append(
                H.WorldBundle(
                    split="fresh_replication",
                    suite=str(cfg.suite),
                    world_index=int(world_index),
                    world_seed=int(world_seed),
                    world=world,
                    roadmap=roadmap,
                    features=features,
                )
            )
            records.append(
                {
                    "world_index": int(world_index),
                    "world_seed": int(world_seed),
                    "roadmap_seed": int(world_seed) + 17,
                    "density": int(density),
                    "nodes": int(len(roadmap.points)),
                    "edges": int(sum(len(row) for row in roadmap.adj) // 2),
                    "build_seconds": build_seconds[density],
                    "feature_seconds": feature_seconds,
                    "prefix_points_verified": True,
                }
            )
        if len(bundles[ordered[0]]) >= requested:
            break
    if any(len(group) != requested for group in bundles.values()):
        raise RuntimeError("fresh paired-density cohort under-filled")
    return bundles, records, {
        "candidate_worlds_seen": int(candidates_seen),
        "accepted_paired_worlds": int(requested),
        "paired_acceptance_rate": float(requested / max(1, candidates_seen)),
        "prefix_failures": int(prefix_failures),
        "source_seed_overlap": int(overlap_count),
    }


def evaluate_fixed_candidate(
    cfg: ReplicationConfig,
    model: torch.nn.Module,
    bundles: Mapping[int, Sequence[H.WorldBundle]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for density in sorted(bundles):
        for bundle in bundles[density]:
            roadmap = bundle.roadmap
            euclid = C13.euclidean_to_goal(roadmap.points, roadmap.points[1])
            inference_started = time.perf_counter()
            prediction = I.predict_model(model, bundle.features, device)
            inference_seconds = float(time.perf_counter() - inference_started)
            learned = euclid + float(bundle.world.side_len) * prediction
            rank = euclid + float(cfg.alpha) * (learned - euclid)
            oracle = np.asarray(roadmap.dist_to_goal, dtype=np.float64)
            connected = np.isfinite(oracle) & (oracle < C.INF / 10.0)
            optimal = float(oracle[0])
            budget = int(math.ceil(float(cfg.budget_factor) * len(roadmap.points)))

            focal_started = time.perf_counter()
            focal = I.focal_search_with_secondary(
                roadmap.adj,
                euclid,
                euclid,
                len(roadmap.points),
                float(cfg.focal_w),
                "h",
            )
            focal_seconds = float(time.perf_counter() - focal_started)
            same_started = time.perf_counter()
            same = Q.shared_anchor_certified_search(
                roadmap.adj,
                euclid,
                euclid,
                float(cfg.focal_w),
                budget,
                validate_anchor=False,
            )
            same_seconds = float(time.perf_counter() - same_started)
            learned_started = time.perf_counter()
            result = Q.shared_anchor_certified_search(
                roadmap.adj,
                euclid,
                rank,
                float(cfg.focal_w),
                budget,
                validate_anchor=False,
            )
            learned_seconds = float(time.perf_counter() - learned_started)
            direct = C.astar_search(roadmap.adj, rank, len(roadmap.points))
            path = Q.validate_path(roadmap.adj, result["path"], result["final_cost"])
            same_path = Q.validate_path(roadmap.adj, same["path"], same["final_cost"])
            final_cost = float(result["final_cost"])
            nonfinite_prediction = bool(not np.all(np.isfinite(prediction)))
            safety_failure = bool(
                nonfinite_prediction
                or not bool(focal["found"])
                or not bool(same["certified"])
                or not bool(same_path["valid"])
                or not bool(result["certified"])
                or not bool(path["valid"])
                or not math.isfinite(final_cost)
                or final_cost > float(cfg.focal_w) * optimal + 1.0e-9
                or int(result["max_expansions_per_state"]) > 2
            )
            rows.append(
                {
                    "suite": bundle.suite,
                    "world_index": int(bundle.world_index),
                    "world_seed": int(bundle.world_seed),
                    "roadmap_seed": int(bundle.world_seed) + 17,
                    "density": int(density),
                    "roadmap_k": int(cfg.roadmap_k),
                    "budget": int(budget),
                    "model": "flat_mlp",
                    "iteration": 6,
                    "alpha": float(cfg.alpha),
                    "focal_w": float(cfg.focal_w),
                    "inference_seconds": inference_seconds,
                    "learned_search_seconds": learned_seconds,
                    "euclid_focal_search_seconds": focal_seconds,
                    "same_search_euclid_seconds": same_seconds,
                    "prediction_mean": float(np.mean(prediction)),
                    "prediction_p95": float(np.percentile(prediction, 95)),
                    "rank_vs_oracle_spearman_eval_only": I.safe_spearman(
                        learned[connected], oracle[connected]
                    ),
                    "oracle_overestimate_rate_eval_only": float(
                        np.mean(learned[connected] > oracle[connected] + 1.0e-9)
                    ),
                    "optimal_cost_eval_only": optimal,
                    "certified": bool(result["certified"]),
                    "proof": result["proof"],
                    "path_valid": bool(path["valid"]),
                    "final_cost": final_cost if math.isfinite(final_cost) else "",
                    "final_cost_ratio_eval_only": (
                        final_cost / optimal if math.isfinite(final_cost) else ""
                    ),
                    "bound_violation_eval_only": bool(
                        not math.isfinite(final_cost)
                        or final_cost > float(cfg.focal_w) * optimal + 1.0e-9
                    ),
                    "expansions": int(result["expansions"]),
                    "rank_expansions": int(result["rank_expansions"]),
                    "anchor_expansions": int(result["anchor_expansions"]),
                    "max_expansions_per_state": int(
                        result["max_expansions_per_state"]
                    ),
                    "euclid_focal_found": bool(focal["found"]),
                    "euclid_focal_expansions": int(focal["expansions"]),
                    "delta_vs_euclid_focal": int(result["expansions"])
                    - int(focal["expansions"]),
                    "same_search_euclid_certified": bool(same["certified"]),
                    "same_search_euclid_path_valid": bool(same_path["valid"]),
                    "same_search_euclid_expansions": int(same["expansions"]),
                    "delta_vs_same_search_euclid": int(result["expansions"])
                    - int(same["expansions"]),
                    "direct_learned_astar_expansions": int(direct["expansions"]),
                    "direct_learned_astar_cost_ratio_eval_only": float(
                        direct["cost"] / optimal
                    ),
                    "nonfinite_prediction": nonfinite_prediction,
                    "safety_failure": safety_failure,
                }
            )
    return rows


def summarize_replication(
    rows: Sequence[Mapping[str, Any]], cfg: ReplicationConfig
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for density in parse_int_csv(cfg.densities):
        group = [row for row in rows if int(row["density"]) == int(density)]
        focal_delta = np.asarray(
            [float(row["delta_vs_euclid_focal"]) for row in group], dtype=np.float64
        )
        same_delta = np.asarray(
            [float(row["delta_vs_same_search_euclid"]) for row in group],
            dtype=np.float64,
        )
        required = int(math.ceil(float(cfg.required_win_fraction) * len(group)))
        focal_low, focal_high = bootstrap_mean_ci(
            focal_delta, cfg.bootstrap_replicates, cfg.bootstrap_seed + density
        )
        same_low, same_high = bootstrap_mean_ci(
            same_delta, cfg.bootstrap_replicates, cfg.bootstrap_seed + 10_000 + density
        )
        safety = int(np.sum([bool(row["safety_failure"]) for row in group]))
        focal_wins = int(np.sum(focal_delta < 0.0))
        same_wins = int(np.sum(same_delta < 0.0))
        focal_mean = float(np.mean(focal_delta))
        same_mean = float(np.mean(same_delta))
        summaries.append(
            {
                "density": int(density),
                "worlds": int(len(group)),
                "required_wins": int(required),
                "gate_pass": bool(
                    len(group) == int(cfg.worlds)
                    and safety == 0
                    and focal_wins >= required
                    and same_wins >= required
                    and focal_mean < 0.0
                    and same_mean < 0.0
                ),
                "safety_failures": int(safety),
                "expansions_mean": float(
                    np.mean([float(row["expansions"]) for row in group])
                ),
                "euclid_focal_expansions_mean": float(
                    np.mean([float(row["euclid_focal_expansions"]) for row in group])
                ),
                "delta_vs_euclid_focal_mean": focal_mean,
                "delta_vs_euclid_focal_ci95_low": focal_low,
                "delta_vs_euclid_focal_ci95_high": focal_high,
                "focal_wins": focal_wins,
                "focal_ties": int(np.sum(focal_delta == 0.0)),
                "focal_losses": int(np.sum(focal_delta > 0.0)),
                "same_search_euclid_expansions_mean": float(
                    np.mean(
                        [float(row["same_search_euclid_expansions"]) for row in group]
                    )
                ),
                "delta_vs_same_search_euclid_mean": same_mean,
                "delta_vs_same_search_euclid_ci95_low": same_low,
                "delta_vs_same_search_euclid_ci95_high": same_high,
                "same_search_euclid_wins": same_wins,
                "same_search_euclid_ties": int(np.sum(same_delta == 0.0)),
                "same_search_euclid_losses": int(np.sum(same_delta > 0.0)),
                "final_cost_ratio_max_eval_only": float(
                    np.max([float(row["final_cost_ratio_eval_only"]) for row in group])
                ),
                "direct_learned_astar_cost_ratio_max_eval_only": float(
                    np.max(
                        [
                            float(row["direct_learned_astar_cost_ratio_eval_only"])
                            for row in group
                        ]
                    )
                ),
                "inference_seconds_mean": float(
                    np.mean([float(row["inference_seconds"]) for row in group])
                ),
                "learned_search_seconds_mean": float(
                    np.mean([float(row["learned_search_seconds"]) for row in group])
                ),
            }
        )
    return summaries


def build_verdict(
    summaries: Sequence[Mapping[str, Any]], required_densities: Sequence[int]
) -> Dict[str, Any]:
    expected = {int(value) for value in required_densities}
    observed = {int(row["density"]) for row in summaries}
    passed = observed == expected and all(bool(row["gate_pass"]) for row in summaries)
    return {
        "verdict": (
            "fresh_192_211_replication_pass"
            if passed
            else "fresh_192_211_replication_fail"
        ),
        "gate_pass": bool(passed),
        "required_densities": sorted(expected),
        "passing_densities": sorted(
            int(row["density"]) for row in summaries if bool(row["gate_pass"])
        ),
        "fixed_selection": {
            "model": "flat_mlp",
            "iteration": 6,
            "alpha": 1.0,
            "focal_w": 1.1,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "authorization": (
            "run_matched_c7_six_suite_comparison"
            if passed
            else "do_not_scale_candidate"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-H fresh 192/211 replication")
    for field in ReplicationConfig.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        parser.add_argument(name, type=type(field.default), default=field.default)
    return parser.parse_args()


def resolve_paths(cfg: ReplicationConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    for field_name in ("source_run_dir", "checkpoint", "out_dir"):
        default = getattr(ReplicationConfig, field_name)
        if getattr(cfg, field_name) == default:
            setattr(cfg, field_name, str(script_dir / default))


def main() -> None:
    cfg = ReplicationConfig(**vars(parse_args()))
    resolve_paths(cfg)
    densities = parse_int_csv(cfg.densities)
    if densities != [192, 211]:
        raise ValueError("locked replication requires densities 192,211")
    if int(cfg.worlds) != 12:
        raise ValueError("locked replication requires 12 paired worlds")
    if not math.isclose(float(cfg.alpha), 1.0) or not math.isclose(
        float(cfg.focal_w), 1.10
    ):
        raise ValueError("locked replication requires alpha=1.0 and focal-w=1.10")
    device = resolve_device(cfg.device)
    model, training_cfg, model_cfg, _ = load_bound_checkpoint(cfg, device)
    forbidden = source_seed_set(Path(cfg.source_run_dir))
    bundles, cohort_rows, cohort_stats = generate_paired_bundles(
        cfg, H.state_config(training_cfg), forbidden
    )
    rows = evaluate_fixed_candidate(cfg, model, bundles, device)
    summaries = summarize_replication(rows, cfg)
    verdict = build_verdict(summaries, densities)

    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    raw_path = C13.write_csv(result_dir / "fresh_replication_raw.csv", rows)
    summary_path = C13.write_csv(
        result_dir / "fresh_replication_summary.csv", summaries
    )
    verdict_path = C13.write_json(result_dir / "gate_verdict.json", verdict)
    cohort_path = C13.write_json(
        result_dir / "fresh_cohort.json",
        {
            "seed_base": int(cfg.seed),
            "fresh_seed_offset": int(cfg.fresh_seed_offset),
            "selection_rule": "accept_only_worlds_with_connected_192_and_211_roadmaps",
            "same_world_and_roadmap_seed_across_densities": True,
            "point_samples_are_density_prefixes": True,
            "stats": cohort_stats,
            "roadmaps": cohort_rows,
        },
    )
    verification = {
        "device": str(device),
        "rows": int(len(rows)),
        "expected_rows": int(cfg.worlds) * len(densities),
        "worlds_per_density": {
            str(density): int(len(bundles[density])) for density in densities
        },
        "source_seed_overlap": int(cohort_stats["source_seed_overlap"]),
        "prefix_failures": int(cohort_stats["prefix_failures"]),
        "safety_failures": int(np.sum([bool(row["safety_failure"]) for row in rows])),
        "checkpoint_sha256": S.file_sha256(Path(cfg.checkpoint)),
        "training_runner_sha256": S.file_sha256(Path(H.__file__).resolve()),
        "shortest_path_target": False,
        "training_target_reads_dist_to_goal": False,
        "runtime_information": "current_goal_geometry_bounded_rays_one_hop_actions",
        "evaluation_oracle_role": "connectivity_cost_ratio_and_rank_audit_only",
        "fixed_before_fresh_data": {
            "model": "flat_mlp",
            "iteration": 6,
            "alpha": float(cfg.alpha),
            "focal_w": float(cfg.focal_w),
            "required_win_fraction": float(cfg.required_win_fraction),
            "worlds": int(cfg.worlds),
            "densities": densities,
        },
        "gate_pass": bool(verdict["gate_pass"]),
    }
    verification_path = C13.write_json(
        result_dir / "verification.json", verification
    )
    if (
        verification["rows"] != verification["expected_rows"]
        or verification["source_seed_overlap"]
        or verification["prefix_failures"]
        or verification["safety_failures"]
    ):
        raise RuntimeError("fresh replication provenance or safety verification failed")

    source_run = Path(cfg.source_run_dir)
    source_paths = {
        "implementation": Path(__file__).resolve(),
        "training_implementation": Path(H.__file__).resolve(),
        "checkpoint": Path(cfg.checkpoint),
        "source_run_manifest": source_run / "manifest.json",
        "source_development_verdict": source_run / "results" / "gate_verdict.json",
        "source_cohorts": source_run / "results" / "lhbl_cohorts.json",
        "source_development_raw": source_run / "results" / "lhbl_search_raw.csv",
    }
    output_paths = {
        "raw": raw_path,
        "summary": summary_path,
        "gate": verdict_path,
        "cohort": cohort_path,
        "verification": verification_path,
    }
    integrity = {
        "inputs": {
            name: {"path": str(path), "sha256": S.file_sha256(path)}
            for name, path in source_paths.items()
        },
        "outputs": {
            name: {"path": str(path), "sha256": S.file_sha256(path)}
            for name, path in output_paths.items()
        },
    }
    integrity_path = C13.write_json(Path(cfg.out_dir) / "integrity.json", integrity)
    manifest = {
        "experiment": "C13-H fixed-checkpoint fresh paired-density replication",
        "config": asdict(cfg),
        "source_model_config": asdict(model_cfg),
        "selection_status": "fixed_from_development_before_fresh_seed_generation",
        "runtime_scope": "current_goal_geometry_bounded_rays_one_hop_actions",
        "full_map_runtime_input": False,
        "shortest_path_target": False,
        "gate_verdict": verdict,
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "integrity": str(integrity_path),
    }
    manifest_path = C13.write_json(Path(cfg.out_dir) / "manifest.json", manifest)
    print(f"verdict={verdict['verdict']}")
    print(f"authorization={verdict['authorization']}")
    print(f"summary={summaries}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
