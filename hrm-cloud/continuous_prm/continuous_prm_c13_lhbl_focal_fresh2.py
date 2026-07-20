#!/usr/bin/env python3
"""Second fresh replication of the fixed C13-H matched-control FOCAL arm."""
from __future__ import annotations

import argparse
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
import continuous_prm_c13_lhbl_focal_matched_control_diagnostic as F
import continuous_prm_c13_lhbl_generated_v3 as H
import continuous_prm_c13_lhbl_replication as R
import continuous_prm_c13_shared_queue as Q
import continuous_prm_c13_state_heuristic as C13


EXPECTED_DIAGNOSTIC_RUNNER_SHA256 = (
    "df3ba82e41a2633a4a590d6b048260104240bd9f2a79109ee75a63d12d9f7f9b"
)
EXPECTED_DIAGNOSTIC_VERDICT_SHA256 = (
    "de5c3986b280b32fbc7b2e8fb5238e647e2fbd01fcd443fc15af42d611f64a0f"
)


@dataclass
class Fresh2Config:
    source_run_dir: str = "runs/c13_lhbl_flat_48w"
    first_replication_dir: str = "runs/c13_lhbl_fresh_192_211"
    diagnostic_dir: str = "runs/c13_lhbl_focal_matched_control_diagnostic"
    checkpoint: str = (
        "runs/c13_lhbl_flat_48w/checkpoints/flat_mlp_iteration_06.pt"
    )
    out_dir: str = "runs/c13_lhbl_focal_fresh2"
    suite: str = "C_hard_maze"
    worlds: int = 12
    densities: str = "192,211"
    roadmap_k: int = 7
    seed: int = 1234
    fresh_seed_offset: int = 2_700_000
    max_world_retries: int = 200
    mode: str = "fhat"
    alpha: float = 0.25
    focal_w: float = 1.10
    budget_factor: float = 2.0
    required_win_fraction: float = 0.80
    bootstrap_replicates: int = 20_000
    bootstrap_seed: int = 291_337
    device: str = "auto"


def validate_fixed_selection(cfg: Fresh2Config) -> None:
    diagnostic_runner = Path(F.__file__).resolve()
    verdict_path = Path(cfg.diagnostic_dir) / "results" / "gate_verdict.json"
    if S.file_sha256(diagnostic_runner) != EXPECTED_DIAGNOSTIC_RUNNER_SHA256:
        raise RuntimeError("matched-control diagnostic runner hash changed")
    if S.file_sha256(verdict_path) != EXPECTED_DIAGNOSTIC_VERDICT_SHA256:
        raise RuntimeError("matched-control diagnostic verdict hash changed")
    verdict = H._read_json(verdict_path)
    selected = verdict.get("selected_candidate") or {}
    if (
        selected.get("mode") != "fhat"
        or not math.isclose(float(selected.get("alpha", -1.0)), 0.25)
        or cfg.mode != "fhat"
        or not math.isclose(float(cfg.alpha), 0.25)
        or not math.isclose(float(cfg.focal_w), 1.10)
    ):
        raise RuntimeError("fresh2 configuration differs from the fixed diagnostic selection")


def forbidden_seed_set(cfg: Fresh2Config) -> set[int]:
    seeds = R.source_seed_set(Path(cfg.source_run_dir))
    cohort = H._read_json(
        Path(cfg.first_replication_dir) / "results" / "fresh_cohort.json"
    )
    seeds.update(int(row["world_seed"]) for row in cohort["roadmaps"])
    return seeds


def evaluate(
    cfg: Fresh2Config,
    model: torch.nn.Module,
    bundles: Mapping[int, Sequence[H.WorldBundle]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for density in sorted(bundles):
        for bundle in bundles[density]:
            roadmap = bundle.roadmap
            euclid = C13.euclidean_to_goal(roadmap.points, roadmap.points[1])
            infer_started = time.perf_counter()
            prediction = I.predict_model(model, bundle.features, device)
            inference_seconds = float(time.perf_counter() - infer_started)
            learned = euclid + float(bundle.world.side_len) * prediction
            rank = euclid + float(cfg.alpha) * (learned - euclid)
            optimal = float(roadmap.dist_to_goal[0])
            budget = int(math.ceil(cfg.budget_factor * len(roadmap.points)))

            baseline_started = time.perf_counter()
            baseline = F.focal_search_with_path(
                roadmap.adj, euclid, euclid, budget, cfg.focal_w, cfg.mode
            )
            baseline_seconds = float(time.perf_counter() - baseline_started)
            learned_started = time.perf_counter()
            result = F.focal_search_with_path(
                roadmap.adj, euclid, rank, budget, cfg.focal_w, cfg.mode
            )
            learned_seconds = float(time.perf_counter() - learned_started)
            legacy = I.focal_search_with_secondary(
                roadmap.adj,
                euclid,
                euclid,
                len(roadmap.points),
                cfg.focal_w,
                cfg.mode,
            )
            astar = C.astar_search(roadmap.adj, euclid, len(roadmap.points))
            direct = C.astar_search(roadmap.adj, rank, len(roadmap.points))
            baseline_path = Q.validate_path(
                roadmap.adj, baseline["path"], baseline["cost"]
            )
            path = Q.validate_path(roadmap.adj, result["path"], result["cost"])
            baseline_cost = float(baseline["cost"])
            cost = float(result["cost"])
            legacy_cost = float(legacy["cost"])
            safety = bool(
                not np.all(np.isfinite(prediction))
                or not bool(baseline["found"])
                or not bool(baseline_path["valid"])
                or baseline_cost > cfg.focal_w * optimal + 1.0e-9
                or baseline_cost
                > cfg.focal_w * float(baseline["anchor_f_min_at_return"]) + 1.0e-9
                or not bool(result["found"])
                or not bool(path["valid"])
                or cost > cfg.focal_w * optimal + 1.0e-9
                or cost > cfg.focal_w * float(result["anchor_f_min_at_return"]) + 1.0e-9
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
                    "mode": cfg.mode,
                    "alpha": float(cfg.alpha),
                    "focal_w": float(cfg.focal_w),
                    "inference_seconds": inference_seconds,
                    "learned_search_seconds": learned_seconds,
                    "euclid_control_search_seconds": baseline_seconds,
                    "found": bool(result["found"]),
                    "path_valid": bool(path["valid"]),
                    "cost": cost,
                    "cost_ratio_eval_only": cost / optimal,
                    "bound_violation_eval_only": bool(
                        cost > cfg.focal_w * optimal + 1.0e-9
                    ),
                    "anchor_certificate_at_return": bool(
                        cost
                        <= cfg.focal_w * float(result["anchor_f_min_at_return"])
                        + 1.0e-9
                    ),
                    "expansions": int(result["expansions"]),
                    "max_expansions_per_state": int(
                        result["max_expansions_per_state"]
                    ),
                    "euclid_control_expansions": int(baseline["expansions"]),
                    "euclid_control_cost_ratio_eval_only": baseline_cost / optimal,
                    "euclid_control_max_expansions_per_state": int(
                        baseline["max_expansions_per_state"]
                    ),
                    "delta_vs_euclid_control": int(result["expansions"])
                    - int(baseline["expansions"]),
                    "legacy_no_reopen_expansions": int(legacy["expansions"]),
                    "legacy_no_reopen_cost_ratio_eval_only": (
                        legacy_cost / optimal if math.isfinite(legacy_cost) else ""
                    ),
                    "legacy_no_reopen_bound_violation_eval_only": bool(
                        not math.isfinite(legacy_cost)
                        or legacy_cost > cfg.focal_w * optimal + 1.0e-9
                    ),
                    "delta_vs_legacy_no_reopen": int(result["expansions"])
                    - int(legacy["expansions"]),
                    "euclid_astar_expansions": int(astar["expansions"]),
                    "delta_vs_euclid_astar": int(result["expansions"])
                    - int(astar["expansions"]),
                    "direct_learned_astar_expansions": int(direct["expansions"]),
                    "direct_learned_astar_cost_ratio_eval_only": float(
                        direct["cost"] / optimal
                    ),
                    "safety_failure": safety,
                }
            )
    return rows


def summarize(
    rows: Sequence[Mapping[str, Any]], cfg: Fresh2Config
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for density in R.parse_int_csv(cfg.densities):
        group = [row for row in rows if int(row["density"]) == density]
        primary = np.asarray(
            [float(row["delta_vs_euclid_control"]) for row in group],
            dtype=np.float64,
        )
        legacy = np.asarray(
            [float(row["delta_vs_legacy_no_reopen"]) for row in group],
            dtype=np.float64,
        )
        astar = np.asarray(
            [float(row["delta_vs_euclid_astar"]) for row in group], dtype=np.float64
        )
        required = int(math.ceil(cfg.required_win_fraction * len(group)))
        low, high = R.bootstrap_mean_ci(
            primary, cfg.bootstrap_replicates, cfg.bootstrap_seed + density
        )
        legacy_low, legacy_high = R.bootstrap_mean_ci(
            legacy, cfg.bootstrap_replicates, cfg.bootstrap_seed + 10_000 + density
        )
        astar_low, astar_high = R.bootstrap_mean_ci(
            astar, cfg.bootstrap_replicates, cfg.bootstrap_seed + 20_000 + density
        )
        wins = int(np.sum(primary < 0.0))
        safety = int(np.sum([bool(row["safety_failure"]) for row in group]))
        output.append(
            {
                "density": int(density),
                "worlds": int(len(group)),
                "required_wins": int(required),
                "gate_pass": bool(
                    len(group) == cfg.worlds
                    and safety == 0
                    and wins >= required
                    and float(np.mean(primary)) < 0.0
                ),
                "safety_failures": int(safety),
                "expansions_mean": float(
                    np.mean([float(row["expansions"]) for row in group])
                ),
                "euclid_control_expansions_mean": float(
                    np.mean([float(row["euclid_control_expansions"]) for row in group])
                ),
                "delta_vs_euclid_control_mean": float(np.mean(primary)),
                "delta_vs_euclid_control_ci95_low": low,
                "delta_vs_euclid_control_ci95_high": high,
                "wins": wins,
                "ties": int(np.sum(primary == 0.0)),
                "losses": int(np.sum(primary > 0.0)),
                "legacy_no_reopen_expansions_mean": float(
                    np.mean([float(row["legacy_no_reopen_expansions"]) for row in group])
                ),
                "delta_vs_legacy_no_reopen_mean": float(np.mean(legacy)),
                "delta_vs_legacy_no_reopen_ci95_low": legacy_low,
                "delta_vs_legacy_no_reopen_ci95_high": legacy_high,
                "legacy_bound_violations": int(
                    np.sum(
                        [
                            bool(row["legacy_no_reopen_bound_violation_eval_only"])
                            for row in group
                        ]
                    )
                ),
                "euclid_astar_expansions_mean": float(
                    np.mean([float(row["euclid_astar_expansions"]) for row in group])
                ),
                "delta_vs_euclid_astar_mean": float(np.mean(astar)),
                "delta_vs_euclid_astar_ci95_low": astar_low,
                "delta_vs_euclid_astar_ci95_high": astar_high,
                "cost_ratio_max_eval_only": float(
                    np.max([float(row["cost_ratio_eval_only"]) for row in group])
                ),
                "direct_learned_astar_cost_ratio_max_eval_only": float(
                    np.max(
                        [
                            float(row["direct_learned_astar_cost_ratio_eval_only"])
                            for row in group
                        ]
                    )
                ),
            }
        )
    return output


def build_verdict(
    summaries: Sequence[Mapping[str, Any]], densities: Sequence[int]
) -> Dict[str, Any]:
    expected = {int(value) for value in densities}
    observed = {int(row["density"]) for row in summaries}
    passed = observed == expected and all(bool(row["gate_pass"]) for row in summaries)
    return {
        "verdict": (
            "fixed_matched_control_fresh2_pass"
            if passed
            else "fixed_matched_control_fresh2_fail"
        ),
        "gate_pass": bool(passed),
        "passing_densities": sorted(
            int(row["density"]) for row in summaries if bool(row["gate_pass"])
        ),
        "fixed_candidate": {
            "model": "flat_mlp",
            "iteration": 6,
            "mode": "fhat",
            "alpha": 0.25,
            "focal_w": 1.10,
            "checkpoint_sha256": R.EXPECTED_CHECKPOINT_SHA256,
        },
        "authorization": (
            "run_matched_c7_six_suite_comparison"
            if passed
            else "do_not_scale_candidate"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-H fixed FOCAL fresh2")
    for field in Fresh2Config.__dataclass_fields__.values():
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=type(field.default),
            default=field.default,
        )
    return parser.parse_args()


def resolve_paths(cfg: Fresh2Config) -> None:
    script_dir = Path(__file__).resolve().parent
    for field_name in (
        "source_run_dir",
        "first_replication_dir",
        "diagnostic_dir",
        "checkpoint",
        "out_dir",
    ):
        default = getattr(Fresh2Config, field_name)
        if getattr(cfg, field_name) == default:
            setattr(cfg, field_name, str(script_dir / default))


def main() -> None:
    cfg = Fresh2Config(**vars(parse_args()))
    resolve_paths(cfg)
    densities = R.parse_int_csv(cfg.densities)
    if densities != [192, 211] or cfg.worlds != 12:
        raise ValueError("locked fresh2 requires 12 worlds at densities 192,211")
    validate_fixed_selection(cfg)
    device = R.resolve_device(cfg.device)
    replication_cfg = R.ReplicationConfig(
        source_run_dir=cfg.source_run_dir,
        checkpoint=cfg.checkpoint,
        suite=cfg.suite,
        worlds=cfg.worlds,
        densities=cfg.densities,
        roadmap_k=cfg.roadmap_k,
        seed=cfg.seed,
        fresh_seed_offset=cfg.fresh_seed_offset,
        max_world_retries=cfg.max_world_retries,
        alpha=cfg.alpha,
        focal_w=cfg.focal_w,
        budget_factor=cfg.budget_factor,
        required_win_fraction=cfg.required_win_fraction,
        device=cfg.device,
    )
    model, training_cfg, model_cfg, _ = R.load_bound_checkpoint(
        replication_cfg, device
    )
    bundles, cohort_rows, cohort_stats = R.generate_paired_bundles(
        replication_cfg,
        H.state_config(training_cfg),
        forbidden_seed_set(cfg),
    )
    rows = evaluate(cfg, model, bundles, device)
    summaries = summarize(rows, cfg)
    verdict = build_verdict(summaries, densities)

    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    raw_path = C13.write_csv(result_dir / "fresh2_raw.csv", rows)
    summary_path = C13.write_csv(result_dir / "fresh2_summary.csv", summaries)
    verdict_path = C13.write_json(result_dir / "gate_verdict.json", verdict)
    cohort_path = C13.write_json(
        result_dir / "fresh2_cohort.json",
        {
            "seed_base": cfg.seed,
            "fresh_seed_offset": cfg.fresh_seed_offset,
            "selection_rule": "accept_only_paired_connected_192_and_211_roadmaps",
            "stats": cohort_stats,
            "roadmaps": cohort_rows,
        },
    )
    verification = {
        "device": str(device),
        "rows": int(len(rows)),
        "expected_rows": int(cfg.worlds * len(densities)),
        "source_seed_overlap": int(cohort_stats["source_seed_overlap"]),
        "prefix_failures": int(cohort_stats["prefix_failures"]),
        "safety_failures": int(np.sum([bool(row["safety_failure"]) for row in rows])),
        "checkpoint_sha256": S.file_sha256(Path(cfg.checkpoint)),
        "diagnostic_runner_sha256": S.file_sha256(Path(F.__file__).resolve()),
        "diagnostic_verdict_sha256": S.file_sha256(
            Path(cfg.diagnostic_dir) / "results" / "gate_verdict.json"
        ),
        "fixed_before_fresh_data": {
            "model": "flat_mlp",
            "iteration": 6,
            "mode": cfg.mode,
            "alpha": cfg.alpha,
            "focal_w": cfg.focal_w,
            "worlds": cfg.worlds,
            "densities": densities,
        },
        "shortest_path_target": False,
        "training_target_reads_dist_to_goal": False,
        "runtime_information": "current_goal_geometry_bounded_rays_one_hop_actions",
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
        raise RuntimeError("fresh2 provenance or safety verification failed")

    inputs = {
        "implementation": Path(__file__).resolve(),
        "matched_control_implementation": Path(F.__file__).resolve(),
        "training_implementation": Path(H.__file__).resolve(),
        "checkpoint": Path(cfg.checkpoint),
        "diagnostic_manifest": Path(cfg.diagnostic_dir) / "manifest.json",
        "diagnostic_verdict": Path(cfg.diagnostic_dir)
        / "results"
        / "gate_verdict.json",
        "first_replication_cohort": Path(cfg.first_replication_dir)
        / "results"
        / "fresh_cohort.json",
    }
    outputs = {
        "raw": raw_path,
        "summary": summary_path,
        "gate": verdict_path,
        "cohort": cohort_path,
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
            "experiment": "C13-H fixed matched-control FOCAL second fresh replication",
            "config": asdict(cfg),
            "source_model_config": asdict(model_cfg),
            "selection_status": "fixed_before_second_fresh_seed_generation",
            "runtime_scope": "current_goal_geometry_bounded_rays_one_hop_actions",
            "full_map_runtime_input": False,
            "shortest_path_target": False,
            "verdict": verdict,
            "outputs": {name: str(path) for name, path in outputs.items()},
            "integrity": str(integrity_path),
        },
    )
    print(f"verdict={verdict['verdict']}")
    print(f"authorization={verdict['authorization']}")
    print(f"summary={summaries}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
