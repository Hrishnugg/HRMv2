#!/usr/bin/env python3
"""Post-hoc C13-L mechanism probe: no-reopen vs reopening vs bounded FOCAL.

This script is diagnostic only.  Its rows may motivate a new preregistered seed
block but may not be reported as confirmation.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import continuous_prm_common as C
import continuous_prm_c13_identifiability as I
import continuous_prm_c13_lhbl_c7_comparison as X
import continuous_prm_c13_lhbl_generated_v3 as H
import continuous_prm_c13_lhbl_multisuite as J
import continuous_prm_c13_lhbl_focal_matched_control_diagnostic as F
import continuous_prm_c13_local_backup_scale as L
import continuous_prm_c13_local_bellman_integration as K
import continuous_prm_c13_shared_queue as Q
import continuous_prm_c13_state_heuristic as C13


@dataclass
class ProbeConfig:
    multisuite_run_dir: str = "runs/c13_lhbl_multisuite"
    scale_run_dir: str = "runs/c13_local_backup_scale"
    c7_run_dir: str = "runs/c7_local"
    out_dir: str = "runs/c13_reopening_rank_probe"
    alphas: str = "1.00,1.25,1.50,2.00"
    sensor_radius_frac: float = 0.20
    focal_w: float = 1.10
    device: str = "cpu"


def resolve_paths(cfg: ProbeConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    for field_name in (
        "multisuite_run_dir",
        "scale_run_dir",
        "c7_run_dir",
        "out_dir",
    ):
        path = Path(getattr(cfg, field_name))
        if not path.is_absolute():
            setattr(cfg, field_name, str((script_dir / path).resolve()))


def reopening_rank_astar(
    adj: List[List[Tuple[int, float]]],
    heuristic: np.ndarray,
    budget: int,
    start_idx: int = 0,
    goal_idx: int = 1,
) -> Dict[str, Any]:
    """First-goal A* with arbitrary rank and state reopening on improved g."""
    rank = np.asarray(heuristic, dtype=np.float64).reshape(-1)
    n = len(adj)
    if rank.shape != (n,) or not np.all(np.isfinite(rank)):
        raise ValueError("rank must be finite and graph-aligned")
    g = np.full(n, C.INF, dtype=np.float64)
    expanded_g = np.full(n, C.INF, dtype=np.float64)
    paths: List[List[int]] = [[] for _ in range(n)]
    g[start_idx] = 0.0
    paths[start_idx] = [int(start_idx)]
    counter = itertools.count()
    heap: List[Tuple[float, float, int, int]] = [
        (float(rank[start_idx]), 0.0, int(start_idx), next(counter))
    ]
    expansions = 0
    expansion_counts = np.zeros(n, dtype=np.int64)
    while heap and expansions < int(budget):
        _, entry_g, node, _ = heapq.heappop(heap)
        if entry_g != float(g[node]):
            continue
        if entry_g >= float(expanded_g[node]) - C.EPS:
            continue
        expanded_g[node] = entry_g
        expansions += 1
        expansion_counts[node] += 1
        if node == int(goal_idx):
            return {
                "found": True,
                "cost": float(entry_g),
                "expansions": int(expansions),
                "closed": int(np.sum(np.isfinite(expanded_g) & (expanded_g < C.INF / 10.0))),
                "path": list(paths[node]),
                "reexpansions": int(np.sum(np.maximum(expansion_counts - 1, 0))),
                "max_expansions_per_state": int(np.max(expansion_counts)),
            }
        for neighbor, edge_cost in adj[node]:
            new_g = float(entry_g + edge_cost)
            if new_g + C.EPS >= float(g[neighbor]):
                continue
            g[neighbor] = new_g
            paths[neighbor] = list(paths[node]) + [int(neighbor)]
            heapq.heappush(
                heap,
                (
                    new_g + float(rank[neighbor]),
                    new_g,
                    int(neighbor),
                    next(counter),
                ),
            )
    return {
        "found": False,
        "cost": float("nan"),
        "expansions": int(expansions),
        "closed": int(np.sum(np.isfinite(expanded_g) & (expanded_g < C.INF / 10.0))),
        "path": [],
        "reexpansions": int(np.sum(np.maximum(expansion_counts - 1, 0))),
        "max_expansions_per_state": int(np.max(expansion_counts)),
    }


def rebuild_scale_cohort(cfg: ProbeConfig) -> List[H.WorldBundle]:
    scale_cfg = L.ScaleConfig(
        multisuite_run_dir=cfg.multisuite_run_dir,
        c7_run_dir=cfg.c7_run_dir,
        out_dir=cfg.scale_run_dir,
        device=cfg.device,
    )
    bundles, records, _, verification = L.build_calibration_cohort(scale_cfg)
    saved = H._read_json(
        Path(cfg.scale_run_dir) / "results" / "calibration_cohort.json"
    )
    expected_seeds = [int(row["world_seed"]) for row in saved["records"]]
    observed_seeds = [int(row["world_seed"]) for row in records]
    if expected_seeds != observed_seeds or any(verification["overlap"].values()):
        raise RuntimeError("C13-L calibration cohort replay changed")
    return bundles


def run(cfg: ProbeConfig) -> Dict[str, Any]:
    device = H.resolve_device(cfg.device)
    if str(device) != "cpu":
        raise RuntimeError("probe is locked to CPU")
    bundles = rebuild_scale_cohort(cfg)
    checkpoint = (
        Path(cfg.multisuite_run_dir)
        / "checkpoints"
        / "flat_mlp_iteration_08.pt"
    )
    model, payload = K._load_model(checkpoint, device)
    if int(payload["iteration"]) != 8:
        raise RuntimeError("probe checkpoint changed")
    multi_cfg = J.MultiSuiteConfig(
        c7_run_dir=cfg.c7_run_dir,
        out_dir=cfg.multisuite_run_dir,
        device=cfg.device,
    )
    field = J._load_c7_comparators(multi_cfg, device)["field_hrm"]
    rows: List[Dict[str, Any]] = []
    alphas = C13.parse_float_csv(cfg.alphas)
    for ordinal, bundle in enumerate(bundles, start=1):
        roadmap = bundle.roadmap
        euclid = C13.euclidean_to_goal(roadmap.points, roadmap.points[1])
        optimal = float(roadmap.dist_to_goal[0])
        field_result = X.astar_with_path(
            roadmap.adj,
            np.asarray(field.node_h(bundle.world, roadmap, 1), dtype=np.float64),
            len(roadmap.points),
        )
        prediction = np.asarray(
            I.predict_model(model, bundle.features, device), dtype=np.float64
        )
        learned = euclid + float(bundle.world.side_len) * prediction
        local_values, _ = H.limited_horizon_values(
            roadmap.points,
            roadmap.adj,
            roadmap.points[1],
            learned,
            cfg.sensor_radius_frac * float(bundle.world.side_len),
        )
        for alpha in alphas:
            rank = euclid + float(alpha) * (local_values - euclid)
            variants = {
                "no_reopen": X.astar_with_path(
                    roadmap.adj, rank, len(roadmap.points)
                ),
                "reopen_first_goal": reopening_rank_astar(
                    roadmap.adj, rank, 2 * len(roadmap.points)
                ),
                "reopening_fhat_focal": F.focal_search_with_path(
                    roadmap.adj,
                    euclid,
                    rank,
                    2 * len(roadmap.points),
                    cfg.focal_w,
                    "fhat",
                ),
            }
            for variant, result in variants.items():
                found = bool(result["found"])
                path = Q.validate_path(
                    roadmap.adj, result["path"], result["cost"]
                ) if found else {"valid": False}
                anchor_min = float(result.get("anchor_f_min_at_return", float("nan")))
                rows.append(
                    {
                        "suite": bundle.suite,
                        "world_index": int(bundle.world_index),
                        "world_seed": int(bundle.world_seed),
                        "alpha": float(alpha),
                        "variant": variant,
                        "found": found,
                        "path_valid": bool(path["valid"]),
                        "expansions": int(result["expansions"]),
                        "reexpansions": int(result.get("reexpansions", 0)),
                        "max_expansions_per_state": int(
                            result.get("max_expansions_per_state", 1)
                        ),
                        "cost_ratio_eval_only": (
                            float(result["cost"] / optimal) if found else float("nan")
                        ),
                        "bound_violation_eval_only": bool(
                            found and float(result["cost"]) > cfg.focal_w * optimal + 1.0e-9
                        ),
                        "certificate_violation": bool(
                            variant == "reopening_fhat_focal"
                            and found
                            and math.isfinite(anchor_min)
                            and float(result["cost"]) > cfg.focal_w * anchor_min + 1.0e-9
                        ),
                        "field_hrm_expansions": int(field_result["expansions"]),
                        "delta_vs_field_hrm": int(result["expansions"])
                        - int(field_result["expansions"]),
                    }
                )
        print(f"[c13m-probe] {ordinal}/{len(bundles)}", flush=True)

    summaries: List[Dict[str, Any]] = []
    grouped: DefaultDict[Tuple[str, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), float(row["alpha"]))].append(row)
    for (variant, alpha), group in sorted(grouped.items()):
        delta = np.asarray([float(row["delta_vs_field_hrm"]) for row in group])
        summaries.append(
            {
                "variant": variant,
                "alpha": alpha,
                "worlds": len(group),
                "expansions_mean": float(
                    np.mean([float(row["expansions"]) for row in group])
                ),
                "field_hrm_expansions_mean": float(
                    np.mean([float(row["field_hrm_expansions"]) for row in group])
                ),
                "delta_vs_field_hrm_mean": float(np.mean(delta)),
                "wins": int(np.sum(delta < 0.0)),
                "ties": int(np.sum(delta == 0.0)),
                "losses": int(np.sum(delta > 0.0)),
                "cost_ratio_mean_eval_only": float(
                    np.mean([float(row["cost_ratio_eval_only"]) for row in group])
                ),
                "cost_ratio_max_eval_only": float(
                    np.max([float(row["cost_ratio_eval_only"]) for row in group])
                ),
                "bound_violations": int(
                    sum(bool(row["bound_violation_eval_only"]) for row in group)
                ),
                "certificate_violations": int(
                    sum(bool(row["certificate_violation"]) for row in group)
                ),
                "reexpansions_mean": float(
                    np.mean([float(row["reexpansions"]) for row in group])
                ),
            }
        )
    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    raw_path = C13.write_csv(result_dir / "posthoc_raw.csv", rows)
    summary_path = C13.write_csv(result_dir / "posthoc_summary.csv", summaries)
    verification = C13.write_json(
        result_dir / "verification.json",
        {
            "status": "posthoc_mechanism_probe_not_confirmation",
            "worlds": len(bundles),
            "rows": len(rows),
            "invalid_paths": int(sum(not bool(row["path_valid"]) for row in rows)),
            "checkpoint": str(checkpoint),
            "full_map_runtime_input": False,
        },
    )
    print(f"[c13m-probe] -> {summary_path}", flush=True)
    return {"raw": raw_path, "summary": summary_path, "verification": verification}


def main() -> None:
    cfg = ProbeConfig()
    resolve_paths(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
