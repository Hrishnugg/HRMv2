#!/usr/bin/env python3
"""Diagnose C13-H rank integration on the failed first fresh cohort.

The cohort in ``c13_lhbl_fresh_192_211`` has already failed the locked shared-
queue gate, so this script treats it as development data.  It keeps the model
checkpoint and focal width fixed while testing ordinary Euclidean-anchored
FOCAL secondary keys.  A selected integration must be confirmed on a new seed
range before it can support any claim.
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
import continuous_prm_c13_lhbl_generated_v3 as H
import continuous_prm_c13_lhbl_replication as R
import continuous_prm_c13_shared_queue as Q
import continuous_prm_c13_state_heuristic as C13
import continuous_prm_c7_hard_maps as M7


@dataclass
class DiagnosticConfig:
    source_replication_dir: str = "runs/c13_lhbl_fresh_192_211"
    source_run_dir: str = "runs/c13_lhbl_flat_48w"
    checkpoint: str = (
        "runs/c13_lhbl_flat_48w/checkpoints/flat_mlp_iteration_06.pt"
    )
    out_dir: str = "runs/c13_lhbl_focal_matched_control_diagnostic"
    modes: str = "h,fhat,residual"
    alphas: str = "0.25,0.50,0.75,1.00"
    focal_w: float = 1.10
    required_win_fraction: float = 0.80
    bootstrap_replicates: int = 20_000
    bootstrap_seed: int = 177_013
    device: str = "auto"


def parse_names(value: str) -> List[str]:
    names = [item.strip() for item in str(value).split(",") if item.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("modes must be a nonempty unique comma-separated list")
    if any(name not in {"h", "fhat", "residual"} for name in names):
        raise ValueError("modes must be drawn from h,fhat,residual")
    return names


def reconstruct_path(
    parent: np.ndarray, start_idx: int = 0, goal_idx: int = 1
) -> List[int]:
    path: List[int] = []
    node = int(goal_idx)
    while node >= 0:
        path.append(node)
        if node == int(start_idx):
            return list(reversed(path))
        node = int(parent[node])
    return []


def focal_search_with_path(
    adj: Sequence[Sequence[Tuple[int, float]]],
    euclid_h: np.ndarray,
    rank_h: np.ndarray,
    budget: int,
    w: float,
    secondary: str,
    start_idx: int = 0,
    goal_idx: int = 1,
) -> Dict[str, Any]:
    """Ordinary FOCAL with a consistent Euclidean anchor and saved path."""

    if float(w) < 1.0 or secondary not in {"h", "fhat", "residual"}:
        raise ValueError("invalid focal configuration")
    n = len(adj)
    g = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)
    g[int(start_idx)] = 0.0
    counter = 0
    opened: List[Tuple[float, float, int, int]] = [
        (float(euclid_h[start_idx]), 0.0, int(start_idx), counter)
    ]
    closed = np.zeros(n, dtype=np.bool_)
    expansion_counts = np.zeros(n, dtype=np.int64)
    expansions = 0
    while opened and expansions < int(budget):
        opened = [
            entry
            for entry in opened
            if not closed[entry[2]] and entry[1] == g[entry[2]]
        ]
        if not opened:
            break
        f_min = min(entry[0] for entry in opened)
        focal = [
            entry for entry in opened if entry[0] <= float(w) * f_min + 1.0e-12
        ]

        def key(entry: Tuple[float, float, int, int]) -> Tuple[float, ...]:
            f_value, g_value, node, insertion = entry
            if secondary == "h":
                primary = float(rank_h[node])
            elif secondary == "fhat":
                primary = float(g_value + rank_h[node])
            else:
                primary = float(rank_h[node] - euclid_h[node])
            return primary, float(rank_h[node]), f_value, insertion

        best = min(focal, key=key)
        opened.remove(best)
        _, current_g, node, _ = best
        if closed[node] or current_g != g[node]:
            continue
        closed[node] = True
        expansions += 1
        expansion_counts[node] += 1
        if node == int(goal_idx):
            return {
                "found": True,
                "cost": float(g[node]),
                "expansions": int(expansions),
                "closed": int(closed.sum()),
                "max_expansions_per_state": int(np.max(expansion_counts)),
                "path": reconstruct_path(parent, start_idx, goal_idx),
                "anchor_f_min_at_return": float(f_min),
            }
        for neighbor_value, edge_cost_value in adj[node]:
            neighbor = int(neighbor_value)
            candidate = float(g[node]) + float(edge_cost_value)
            if candidate + C.EPS < float(g[neighbor]):
                g[neighbor] = candidate
                parent[neighbor] = node
                if closed[neighbor]:
                    closed[neighbor] = False
                counter += 1
                opened.append(
                    (
                        candidate + float(euclid_h[neighbor]),
                        candidate,
                        neighbor,
                        counter,
                    )
                )
    return {
        "found": False,
        "cost": float("nan"),
        "expansions": int(expansions),
        "closed": int(closed.sum()),
        "max_expansions_per_state": int(np.max(expansion_counts)),
        "path": [],
        "anchor_f_min_at_return": float("nan"),
    }


def rebuild_failed_replication_cohort(
    cfg: DiagnosticConfig, local_cfg: C13.LocalStateConfig
) -> Dict[int, List[H.WorldBundle]]:
    cohort = H._read_json(
        Path(cfg.source_replication_dir) / "results" / "fresh_cohort.json"
    )
    M7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    suite = "C_hard_maze"
    by_density: Dict[int, List[H.WorldBundle]] = {192: [], 211: []}
    records = sorted(
        cohort["roadmaps"], key=lambda row: (int(row["density"]), int(row["world_index"]))
    )
    for row in records:
        world_seed = int(row["world_seed"])
        density = int(row["density"])
        world = C.build_world(specs[suite], world_seed, 0.45)
        if world is None:
            raise RuntimeError(f"could not replay world {world_seed}")
        roadmap = C.build_prm(
            world,
            C.RoadmapConfig(n_nodes=density, k_neighbors=7),
            seed=int(row["roadmap_seed"]),
        )
        if roadmap is None:
            raise RuntimeError(f"could not replay roadmap {world_seed}/{density}")
        if (
            len(roadmap.points) != int(row["nodes"])
            or sum(len(group) for group in roadmap.adj) // 2 != int(row["edges"])
        ):
            raise RuntimeError("failed-cohort roadmap replay mismatch")
        features = C13.make_local_state_features(
            world, roadmap.points, roadmap.adj, local_cfg
        )
        by_density[density].append(
            H.WorldBundle(
                split="failed_fresh_diagnostic",
                suite=suite,
                world_index=int(row["world_index"]),
                world_seed=world_seed,
                world=world,
                roadmap=roadmap,
                features=features,
            )
        )
    if any(len(group) != 12 for group in by_density.values()):
        raise RuntimeError("failed replication replay is incomplete")
    return by_density


def run_diagnostic(
    cfg: DiagnosticConfig,
    model: torch.nn.Module,
    bundles: Mapping[int, Sequence[H.WorldBundle]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    alphas = C13.parse_float_csv(cfg.alphas)
    modes = parse_names(cfg.modes)
    for density in sorted(bundles):
        for bundle in bundles[density]:
            roadmap = bundle.roadmap
            euclid = C13.euclidean_to_goal(roadmap.points, roadmap.points[1])
            prediction = I.predict_model(model, bundle.features, device)
            learned = euclid + float(bundle.world.side_len) * prediction
            optimal = float(roadmap.dist_to_goal[0])
            baselines: Dict[str, Dict[str, Any]] = {}
            baseline_paths: Dict[str, Dict[str, Any]] = {}
            for mode in modes:
                baselines[mode] = focal_search_with_path(
                    roadmap.adj,
                    euclid,
                    euclid,
                    int(math.ceil(2.0 * len(roadmap.points))),
                    cfg.focal_w,
                    mode,
                )
                baseline_paths[mode] = Q.validate_path(
                    roadmap.adj,
                    baselines[mode]["path"],
                    baselines[mode]["cost"],
                )
            for alpha in alphas:
                rank = euclid + float(alpha) * (learned - euclid)
                for mode in modes:
                    baseline = baselines[mode]
                    baseline_path = baseline_paths[mode]
                    started = time.perf_counter()
                    result = focal_search_with_path(
                        roadmap.adj,
                        euclid,
                        rank,
                        int(math.ceil(2.0 * len(roadmap.points))),
                        cfg.focal_w,
                        mode,
                    )
                    elapsed = float(time.perf_counter() - started)
                    path = Q.validate_path(roadmap.adj, result["path"], result["cost"])
                    cost = float(result["cost"])
                    safety = bool(
                        not bool(baseline["found"])
                        or not bool(baseline_path["valid"])
                        or not bool(result["found"])
                        or not bool(path["valid"])
                        or not math.isfinite(cost)
                        or cost > float(cfg.focal_w) * optimal + 1.0e-9
                        or cost > float(cfg.focal_w)
                        * float(result["anchor_f_min_at_return"])
                        + 1.0e-9
                    )
                    rows.append(
                        {
                            "density": int(density),
                            "world_index": int(bundle.world_index),
                            "world_seed": int(bundle.world_seed),
                            "mode": mode,
                            "alpha": float(alpha),
                            "focal_w": float(cfg.focal_w),
                            "found": bool(result["found"]),
                            "path_valid": bool(path["valid"]),
                            "cost": cost if math.isfinite(cost) else "",
                            "cost_ratio_eval_only": (
                                cost / optimal if math.isfinite(cost) else ""
                            ),
                            "bound_violation_eval_only": bool(
                                not math.isfinite(cost)
                                or cost > float(cfg.focal_w) * optimal + 1.0e-9
                            ),
                            "anchor_certificate_at_return": bool(
                                math.isfinite(cost)
                                and cost
                                <= float(cfg.focal_w)
                                * float(result["anchor_f_min_at_return"])
                                + 1.0e-9
                            ),
                            "expansions": int(result["expansions"]),
                            "max_expansions_per_state": int(result["max_expansions_per_state"]),
                            "euclid_focal_max_expansions_per_state": int(baseline["max_expansions_per_state"]),
                            "euclid_focal_expansions": int(baseline["expansions"]),
                            "delta_vs_euclid_focal": int(result["expansions"])
                            - int(baseline["expansions"]),
                            "search_seconds": elapsed,
                            "safety_failure": safety,
                        }
                    )
    return rows


def summarize(
    rows: Sequence[Mapping[str, Any]], cfg: DiagnosticConfig
) -> List[Dict[str, Any]]:
    grouped: DefaultDict[Tuple[str, float, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["mode"]), float(row["alpha"]), int(row["density"]))].append(row)
    output: List[Dict[str, Any]] = []
    for (mode, alpha, density), group in sorted(grouped.items()):
        delta = np.asarray(
            [float(row["delta_vs_euclid_focal"]) for row in group], dtype=np.float64
        )
        required = int(math.ceil(cfg.required_win_fraction * len(group)))
        wins = int(np.sum(delta < 0.0))
        safety = int(np.sum([bool(row["safety_failure"]) for row in group]))
        low, high = R.bootstrap_mean_ci(
            delta,
            cfg.bootstrap_replicates,
            cfg.bootstrap_seed + density + int(round(alpha * 1000)) + len(mode) * 10_000,
        )
        output.append(
            {
                "mode": mode,
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
                "expansions_mean": float(
                    np.mean([float(row["expansions"]) for row in group])
                ),
                "euclid_focal_expansions_mean": float(
                    np.mean([float(row["euclid_focal_expansions"]) for row in group])
                ),
                "delta_mean": float(np.mean(delta)),
                "delta_ci95_low": low,
                "delta_ci95_high": high,
                "wins": wins,
                "ties": int(np.sum(delta == 0.0)),
                "losses": int(np.sum(delta > 0.0)),
                "cost_ratio_max_eval_only": float(
                    np.max([float(row["cost_ratio_eval_only"]) for row in group])
                ),
                "search_seconds_mean": float(
                    np.mean([float(row["search_seconds"]) for row in group])
                ),
            }
        )
    return output


def select_candidate(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: DefaultDict[Tuple[str, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(str(row["mode"]), float(row["alpha"]))].append(row)
    eligible: List[Dict[str, Any]] = []
    for (mode, alpha), group in grouped.items():
        if {int(row["density"]) for row in group} != {192, 211}:
            continue
        if all(bool(row["gate_pass"]) for row in group):
            eligible.append(
                {
                    "mode": mode,
                    "alpha": float(alpha),
                    "combined_delta_mean": float(
                        np.mean([float(row["delta_mean"]) for row in group])
                    ),
                    "density_summaries": [dict(row) for row in group],
                }
            )
    selected = (
        min(
            eligible,
            key=lambda row: (
                float(row["combined_delta_mean"]),
                str(row["mode"]),
                float(row["alpha"]),
            ),
        )
        if eligible
        else None
    )
    return {
        "verdict": (
            "integration_candidate_selected_requires_new_fresh_replication"
            if selected is not None
            else "no_focal_integration_candidate_passed_diagnostic_gate"
        ),
        "selected_candidate": selected,
        "passing_candidates": int(len(eligible)),
        "fresh_replication_required": selected is not None,
        "authorization": (
            "replicate_fixed_integration_on_new_seed_range"
            if selected is not None
            else "revise_model_or_objective"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-H matched-control FOCAL diagnostic")
    for field in DiagnosticConfig.__dataclass_fields__.values():
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=type(field.default),
            default=field.default,
        )
    return parser.parse_args()


def resolve_paths(cfg: DiagnosticConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    for field_name in (
        "source_replication_dir",
        "source_run_dir",
        "checkpoint",
        "out_dir",
    ):
        default = getattr(DiagnosticConfig, field_name)
        if getattr(cfg, field_name) == default:
            setattr(cfg, field_name, str(script_dir / default))


def main() -> None:
    cfg = DiagnosticConfig(**vars(parse_args()))
    resolve_paths(cfg)
    replication_cfg = R.ReplicationConfig(
        source_run_dir=cfg.source_run_dir,
        checkpoint=cfg.checkpoint,
        device=cfg.device,
    )
    device = R.resolve_device(cfg.device)
    model, training_cfg, _, _ = R.load_bound_checkpoint(replication_cfg, device)
    bundles = rebuild_failed_replication_cohort(cfg, H.state_config(training_cfg))
    rows = run_diagnostic(cfg, model, bundles, device)
    summaries = summarize(rows, cfg)
    verdict = select_candidate(summaries)

    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    raw_path = C13.write_csv(result_dir / "focal_integration_raw.csv", rows)
    summary_path = C13.write_csv(result_dir / "focal_integration_summary.csv", summaries)
    verdict_path = C13.write_json(result_dir / "gate_verdict.json", verdict)
    verification = {
        "rows": int(len(rows)),
        "expected_rows": int(2 * 12 * len(parse_names(cfg.modes)) * len(C13.parse_float_csv(cfg.alphas))),
        "safety_failures": int(np.sum([bool(row["safety_failure"]) for row in rows])),
        "checkpoint_sha256": S.file_sha256(Path(cfg.checkpoint)),
        "cohort_role": "failed_first_replication_reused_for_integration_diagnosis",
        "runtime_information": "current_goal_geometry_bounded_rays_one_hop_actions",
        "anchor": "consistent_euclidean_focal_lower_bound",
        "control_matching": "same_secondary_mode_and_reopening_search",
        "fresh_replication_required": bool(verdict["fresh_replication_required"]),
    }
    verification_path = C13.write_json(result_dir / "verification.json", verification)
    if verification["rows"] != verification["expected_rows"] or verification["safety_failures"]:
        raise RuntimeError("integration diagnostic verification failed")

    source_replication = Path(cfg.source_replication_dir)
    inputs = {
        "implementation": Path(__file__).resolve(),
        "checkpoint": Path(cfg.checkpoint),
        "training_implementation": Path(H.__file__).resolve(),
        "failed_replication_manifest": source_replication / "manifest.json",
        "failed_replication_cohort": source_replication / "results" / "fresh_cohort.json",
        "failed_replication_verdict": source_replication / "results" / "gate_verdict.json",
    }
    outputs = {
        "raw": raw_path,
        "summary": summary_path,
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
            "experiment": "C13-H bound-safe matched-control FOCAL diagnostic",
            "config": asdict(cfg),
            "cohort_role": "failed_first_replication_now_development_only",
            "selection_requires_new_fresh_replication": True,
            "verdict": verdict,
            "outputs": {name: str(path) for name, path in outputs.items()},
            "integrity": str(integrity_path),
        },
    )
    print(f"verdict={verdict['verdict']}")
    print(f"selected={verdict['selected_candidate']}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
