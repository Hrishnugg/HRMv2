#!/usr/bin/env python3
"""C13-D: shared-state anchored multi-queue oracle gate.

This integration-only study reuses the six frozen C13-B audit worlds and
performs no training.  It tests the privileged graph-distance oracle first,
before any exact-rollout or learned provider is authorized.

The search is a one-anchor/one-rank SMHA-style specialization with ``w1=1``:

* both queues share one ``g`` value and parent per state;
* the Euclidean anchor queue owns the admissible lower bound;
* the oracle rank queue is preferred while its minimum key is at most
  ``w * min_anchor``; otherwise the anchor queue expands;
* expanding a state removes its current label from both queues, while a later
  better path can insert it into the queue in which it has not yet expanded;
* a feasible goal label is returned only when
  ``incumbent <= w * min_anchor``.

The last condition directly certifies the requested cost bound.  Expansions
from both queues and states expanded once by each queue are all counted.
Graph shortest-path distance is used only as the explicitly privileged oracle
rank and for post-hoc evaluation; it is never the proof anchor, a training
target, or a learned feature.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
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
import continuous_prm_c13_state_heuristic as C13


@dataclass
class SharedQueueConfig:
    study_dir: str = "runs/c13_identifiability"
    independent_dir: str = "runs/c13_certified_search"
    out_dir: str = "runs/c13_shared_queue_oracle"
    focal_ws: str = "1.05,1.10,1.25"
    primary_w: float = 1.10
    budget_factor: float = 2.0
    required_win_fraction: float = 0.80


def _clean_heap(
    heap: List[Tuple[float, float, int, int, int]],
    g: np.ndarray,
    version: np.ndarray,
) -> None:
    while heap:
        _, entry_g, node, entry_version, _ = heap[0]
        if entry_version != int(version[node]) or entry_g != float(g[node]):
            heapq.heappop(heap)
            continue
        break


def _reconstruct_path(parent: np.ndarray, start_idx: int, goal_idx: int) -> List[int]:
    path = [int(goal_idx)]
    seen = {int(goal_idx)}
    while path[-1] != int(start_idx):
        predecessor = int(parent[path[-1]])
        if predecessor < 0 or predecessor in seen:
            return []
        path.append(predecessor)
        seen.add(predecessor)
    path.reverse()
    return path


def validate_path(
    adj: List[List[Tuple[int, float]]],
    path: Sequence[int],
    expected_cost: float,
    start_idx: int = 0,
    goal_idx: int = 1,
    tolerance: float = 1.0e-9,
) -> Dict[str, Any]:
    nodes = [int(node) for node in path]
    if not nodes or nodes[0] != int(start_idx) or nodes[-1] != int(goal_idx):
        return {"valid": False, "cost": float("nan"), "edges": 0}
    total = 0.0
    for node, neighbor in zip(nodes[:-1], nodes[1:]):
        costs = [
            float(edge_cost)
            for candidate, edge_cost in adj[node]
            if int(candidate) == int(neighbor)
        ]
        if not costs:
            return {"valid": False, "cost": float("nan"), "edges": 0}
        total += min(costs)
    valid = math.isfinite(float(expected_cost)) and math.isclose(
        total,
        float(expected_cost),
        rel_tol=0.0,
        abs_tol=float(tolerance),
    )
    return {"valid": bool(valid), "cost": float(total), "edges": len(nodes) - 1}


def shared_anchor_certified_search(
    adj: List[List[Tuple[int, float]]],
    anchor_h: np.ndarray,
    rank_h: np.ndarray,
    w: float,
    budget: int,
    start_idx: int = 0,
    goal_idx: int = 1,
    validate_anchor: bool = True,
) -> Dict[str, Any]:
    """Run a shared-g anchor/rank search with a direct anchor certificate.

    This follows the shared-path and queue-eligibility structure of SMHA*,
    specialized to one inadmissible queue and an uninflated anchor.  The direct
    incumbent/anchor termination test is sufficient for the requested ``w``
    bound and is checked before every expansion.
    """

    if float(w) < 1.0:
        raise ValueError("w must be at least one")
    if int(budget) < 0:
        raise ValueError("budget must be nonnegative")
    anchor = np.asarray(anchor_h, dtype=np.float64).reshape(-1)
    rank = np.asarray(rank_h, dtype=np.float64).reshape(-1)
    if len(anchor) != len(adj) or len(rank) != len(adj):
        raise ValueError("heuristic lengths must match the graph")
    if not np.all(np.isfinite(rank)):
        raise ValueError("rank heuristic must be finite")
    if validate_anchor:
        S.validate_consistent_anchor(adj, anchor, goal_idx=goal_idx)

    n = len(adj)
    g = np.full(n, np.inf, dtype=np.float64)
    g[start_idx] = 0.0
    parent = np.full(n, -1, dtype=np.int64)
    # Preserve the witness associated with each current g-label. Mutable
    # ancestor pointers can improve without propagating that improvement into
    # an already generated descendant, so reconstructing through live parents
    # can describe a different cost than the certified incumbent.
    label_paths: List[Tuple[int, ...]] = [tuple() for _ in range(n)]
    label_paths[start_idx] = (int(start_idx),)

    version = np.zeros(n, dtype=np.int64)
    expanded_anchor = np.zeros(n, dtype=np.bool_)
    expanded_rank = np.zeros(n, dtype=np.bool_)
    expansion_count = np.zeros(n, dtype=np.int64)
    counter = 0
    anchor_open: List[Tuple[float, float, int, int, int]] = [
        (float(anchor[start_idx]), 0.0, int(start_idx), 0, counter)
    ]
    rank_open: List[Tuple[float, float, int, int, int]] = [
        (float(rank[start_idx]), 0.0, int(start_idx), 0, counter)
    ]

    incumbent = 0.0 if int(start_idx) == int(goal_idx) else float("inf")
    expansions = 0
    anchor_expansions = 0
    rank_expansions = 0
    generated = 1
    incumbent_updates = int(math.isfinite(incumbent))
    improvements_after_expansion = 0
    rank_eligibility_checks = 0
    rank_eligible_choices = 0
    lower_bound = float(anchor[start_idx])
    started = time.perf_counter()

    while expansions < int(budget):
        _clean_heap(anchor_open, g, version)
        _clean_heap(rank_open, g, version)
        if not anchor_open:
            break

        lower_bound = float(anchor_open[0][0])
        if math.isfinite(incumbent) and incumbent <= float(w) * lower_bound + C.EPS:
            path = list(label_paths[goal_idx])
            return {
                "certified": True,
                "found": True,
                "final_cost": float(incumbent),
                "path": path,
                "lower_bound": float(lower_bound),
                "certificate_ratio": float(incumbent / max(C.EPS, lower_bound)),
                "expansions": int(expansions),
                "anchor_expansions": int(anchor_expansions),
                "rank_expansions": int(rank_expansions),
                "generated": int(generated),
                "incumbent_updates": int(incumbent_updates),
                "improvements_after_expansion": int(improvements_after_expansion),
                "duplicate_state_expansions": int(np.sum(expansion_count > 1)),
                "max_expansions_per_state": int(np.max(expansion_count)),
                "rank_eligibility_checks": int(rank_eligibility_checks),
                "rank_eligible_choices": int(rank_eligible_choices),
                "proof": "incumbent_le_w_times_shared_anchor_open_lower_bound",
                "seconds": float(time.perf_counter() - started),
            }

        use_rank = False
        if rank_open:
            rank_eligibility_checks += 1
            use_rank = bool(
                float(rank_open[0][0]) <= float(w) * lower_bound + C.EPS
            )
        if use_rank:
            rank_eligible_choices += 1
            _, current_g, node, entry_version, _ = heapq.heappop(rank_open)
            queue_name = "rank"
        else:
            _, current_g, node, entry_version, _ = heapq.heappop(anchor_open)
            queue_name = "anchor"
        if entry_version != int(version[node]) or current_g != float(g[node]):
            continue

        version[node] += 1  # invalidates this label in both queues
        if queue_name == "rank":
            if expanded_rank[node]:
                raise RuntimeError("state expanded twice by the rank queue")
            expanded_rank[node] = True
            rank_expansions += 1
        else:
            if expanded_anchor[node]:
                raise RuntimeError("state expanded twice by the anchor queue")
            expanded_anchor[node] = True
            anchor_expansions += 1
        expansions += 1
        expansion_count[node] += 1
        if expansion_count[node] > 2:
            raise RuntimeError("shared-queue state expanded more than twice")

        for neighbor, edge_cost in adj[node]:
            neighbor = int(neighbor)
            new_g = float(g[node]) + float(edge_cost)
            if new_g + C.EPS >= float(g[neighbor]):
                continue
            if expanded_anchor[neighbor]:
                raise RuntimeError(
                    "consistent-anchor invariant violated by a later g improvement"
                )
            if expanded_rank[neighbor] or expanded_anchor[neighbor]:
                improvements_after_expansion += 1
            g[neighbor] = new_g
            parent[neighbor] = int(node)
            label_paths[neighbor] = label_paths[node] + (neighbor,)
            version[neighbor] += 1
            counter += 1
            generated += 1
            if not expanded_anchor[neighbor]:
                heapq.heappush(
                    anchor_open,
                    (
                        new_g + float(anchor[neighbor]),
                        new_g,
                        neighbor,
                        int(version[neighbor]),
                        counter,
                    ),
                )
            if not expanded_rank[neighbor]:
                heapq.heappush(
                    rank_open,
                    (
                        new_g + float(rank[neighbor]),
                        new_g,
                        neighbor,
                        int(version[neighbor]),
                        counter,
                    ),
                )
            if neighbor == int(goal_idx) and new_g + C.EPS < incumbent:
                incumbent = float(new_g)
                incumbent_updates += 1

    return {
        "certified": False,
        "found": math.isfinite(incumbent),
        "final_cost": float(incumbent) if math.isfinite(incumbent) else float("nan"),
        "path": (
            list(label_paths[goal_idx])
            if math.isfinite(incumbent)
            else []
        ),
        "lower_bound": float(lower_bound),
        "certificate_ratio": (
            float(incumbent / max(C.EPS, lower_bound))
            if math.isfinite(incumbent)
            else float("nan")
        ),
        "expansions": int(expansions),
        "anchor_expansions": int(anchor_expansions),
        "rank_expansions": int(rank_expansions),
        "generated": int(generated),
        "incumbent_updates": int(incumbent_updates),
        "improvements_after_expansion": int(improvements_after_expansion),
        "duplicate_state_expansions": int(np.sum(expansion_count > 1)),
        "max_expansions_per_state": int(np.max(expansion_count)),
        "rank_eligibility_checks": int(rank_eligibility_checks),
        "rank_eligible_choices": int(rank_eligible_choices),
        "proof": "budget_or_anchor_exhausted_without_certificate",
        "seconds": float(time.perf_counter() - started),
    }


def load_independent_oracle_rows(
    independent_dir: str | Path,
) -> Dict[Tuple[int, float], Dict[str, str]]:
    path = Path(independent_dir) / "results" / "certified_search_raw.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("provider") == "oracle_eval_only"
        ]
    indexed: Dict[Tuple[int, float], Dict[str, str]] = {}
    for row in rows:
        key = (int(row["world_index"]), float(row["focal_w"]))
        if key in indexed:
            raise ValueError(f"duplicate independent oracle row {key}")
        indexed[key] = row
    return indexed


def evaluate_oracle_gate(
    cfg: SharedQueueConfig,
    study_cfg: I.StudyConfig,
    bundles: Sequence[I.AuditBundle],
    independent_rows: Mapping[Tuple[int, float], Mapping[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    focal_ws = C13.parse_float_csv(cfg.focal_ws)
    rows: List[Dict[str, Any]] = []
    baselines: List[Dict[str, Any]] = []
    anchor_checks: List[Dict[str, Any]] = []

    for bundle in bundles:
        rm = bundle.roadmap
        euclid = C13.euclidean_to_goal(rm.points, rm.points[1])
        oracle = C13._finite_oracle(rm, bundle.world.side_len)
        anchor_check = S.validate_consistent_anchor(rm.adj, euclid)
        anchor_checks.append(
            {
                "world_index": bundle.world_index,
                "world_seed": bundle.world_seed,
                **anchor_check,
            }
        )
        optimal = float(rm.dist_to_goal[0])  # evaluation-only
        euclid_astar = C.astar_search(rm.adj, euclid, len(rm.points))
        budget = max(0, int(math.ceil(float(cfg.budget_factor) * len(rm.points))))

        for focal_w in focal_ws:
            baseline = I.focal_search_with_secondary(
                rm.adj,
                euclid,
                euclid,
                budget=len(rm.points),
                w=focal_w,
                secondary="h",
            )
            baseline_cost = float(baseline["cost"])
            baselines.append(
                {
                    "suite": study_cfg.suite,
                    "world_index": bundle.world_index,
                    "world_seed": bundle.world_seed,
                    "focal_w": float(focal_w),
                    "euclid_focal_found": bool(baseline["found"]),
                    "euclid_focal_expansions": int(baseline["expansions"]),
                    "euclid_focal_cost": baseline_cost,
                    "euclid_focal_cost_ratio_eval_only": baseline_cost / optimal,
                    "euclid_astar_expansions": int(euclid_astar["expansions"]),
                    "euclid_astar_cost": float(euclid_astar["cost"]),
                }
            )

            result = shared_anchor_certified_search(
                rm.adj,
                euclid,
                oracle,
                w=float(focal_w),
                budget=budget,
                validate_anchor=False,
            )
            final_cost = float(result["final_cost"])
            path_check = validate_path(rm.adj, result["path"], final_cost)
            independent = independent_rows[(bundle.world_index, float(focal_w))]
            anchor_lb = float(result["lower_bound"])
            rows.append(
                {
                    "suite": study_cfg.suite,
                    "world_index": bundle.world_index,
                    "world_seed": bundle.world_seed,
                    "provider": "oracle_eval_only",
                    "focal_w": float(focal_w),
                    "certified": bool(result["certified"]),
                    "found": bool(result["found"]),
                    "proof": result["proof"],
                    "final_cost": final_cost if math.isfinite(final_cost) else "",
                    "final_cost_ratio_eval_only": (
                        final_cost / optimal if math.isfinite(final_cost) else ""
                    ),
                    "bound_violation_eval_only": bool(
                        not math.isfinite(final_cost)
                        or final_cost > float(focal_w) * optimal + 1.0e-9
                    ),
                    "anchor_lower_bound": anchor_lb,
                    "anchor_lower_bound_exceeds_optimal_eval_only": bool(
                        anchor_lb > optimal + 1.0e-9
                    ),
                    "certificate_ratio": float(result["certificate_ratio"]),
                    "path_valid": bool(path_check["valid"]),
                    "path_cost": path_check["cost"],
                    "path_edges": int(path_check["edges"]),
                    "expansions": int(result["expansions"]),
                    "rank_expansions": int(result["rank_expansions"]),
                    "anchor_expansions": int(result["anchor_expansions"]),
                    "expansion_accounting_valid": bool(
                        int(result["expansions"])
                        == int(result["rank_expansions"])
                        + int(result["anchor_expansions"])
                    ),
                    "duplicate_state_expansions": int(
                        result["duplicate_state_expansions"]
                    ),
                    "max_expansions_per_state": int(
                        result["max_expansions_per_state"]
                    ),
                    "generated": int(result["generated"]),
                    "incumbent_updates": int(result["incumbent_updates"]),
                    "improvements_after_expansion": int(
                        result["improvements_after_expansion"]
                    ),
                    "rank_eligibility_checks": int(
                        result["rank_eligibility_checks"]
                    ),
                    "rank_eligible_choices": int(result["rank_eligible_choices"]),
                    "rank_eligible_choice_rate": float(
                        result["rank_eligible_choices"]
                        / max(1, result["rank_eligibility_checks"])
                    ),
                    "search_seconds": float(result["seconds"]),
                    "euclid_focal_expansions": int(baseline["expansions"]),
                    "delta_vs_euclid_focal": int(result["expansions"])
                    - int(baseline["expansions"]),
                    "euclid_astar_expansions": int(euclid_astar["expansions"]),
                    "delta_vs_euclid_astar": int(result["expansions"])
                    - int(euclid_astar["expansions"]),
                    "independent_certifier_expansions": int(
                        independent["total_expansions"]
                    ),
                    "saved_vs_independent_certifier": int(
                        independent["total_expansions"]
                    )
                    - int(result["expansions"]),
                }
            )
    return rows, baselines, anchor_checks


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    required_win_fraction: float,
) -> List[Dict[str, Any]]:
    grouped: DefaultDict[float, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["focal_w"])].append(row)

    summaries: List[Dict[str, Any]] = []
    for focal_w, group in sorted(grouped.items()):
        deltas = np.asarray(
            [float(row["delta_vs_euclid_focal"]) for row in group],
            dtype=np.float64,
        )
        wins = int(np.sum(deltas < 0.0))
        ties = int(np.sum(deltas == 0.0))
        losses = int(np.sum(deltas > 0.0))
        required_wins = int(math.ceil(float(required_win_fraction) * len(group)))
        certification_rate = float(np.mean([bool(row["certified"]) for row in group]))
        bound_violations = int(
            np.sum([bool(row["bound_violation_eval_only"]) for row in group])
        )
        path_failures = int(np.sum([not bool(row["path_valid"]) for row in group]))
        lower_bound_failures = int(
            np.sum(
                [
                    bool(row["anchor_lower_bound_exceeds_optimal_eval_only"])
                    for row in group
                ]
            )
        )
        mean_delta = float(np.mean(deltas))
        summaries.append(
            {
                "provider": "oracle_eval_only",
                "focal_w": float(focal_w),
                "worlds": int(len(group)),
                "required_wins": int(required_wins),
                "gate_pass": bool(
                    certification_rate == 1.0
                    and bound_violations == 0
                    and path_failures == 0
                    and lower_bound_failures == 0
                    and wins >= required_wins
                    and mean_delta < 0.0
                ),
                "certification_rate": certification_rate,
                "bound_violations_eval_only": bound_violations,
                "path_failures": path_failures,
                "anchor_lower_bound_failures_eval_only": lower_bound_failures,
                "expansions_mean": float(
                    np.mean([float(row["expansions"]) for row in group])
                ),
                "rank_expansions_mean": float(
                    np.mean([float(row["rank_expansions"]) for row in group])
                ),
                "anchor_expansions_mean": float(
                    np.mean([float(row["anchor_expansions"]) for row in group])
                ),
                "duplicate_state_expansions_mean": float(
                    np.mean(
                        [float(row["duplicate_state_expansions"]) for row in group]
                    )
                ),
                "euclid_focal_expansions_mean": float(
                    np.mean([float(row["euclid_focal_expansions"]) for row in group])
                ),
                "delta_vs_euclid_focal_mean": mean_delta,
                "delta_vs_euclid_focal_median": float(np.median(deltas)),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "euclid_astar_expansions_mean": float(
                    np.mean([float(row["euclid_astar_expansions"]) for row in group])
                ),
                "independent_certifier_expansions_mean": float(
                    np.mean(
                        [
                            float(row["independent_certifier_expansions"])
                            for row in group
                        ]
                    )
                ),
                "saved_vs_independent_certifier_mean": float(
                    np.mean(
                        [float(row["saved_vs_independent_certifier"]) for row in group]
                    )
                ),
                "final_cost_ratio_mean_eval_only": float(
                    np.mean([float(row["final_cost_ratio_eval_only"]) for row in group])
                ),
                "final_cost_ratio_max_eval_only": float(
                    np.max([float(row["final_cost_ratio_eval_only"]) for row in group])
                ),
            }
        )
    return summaries


def build_gate_verdict(
    summaries: Sequence[Mapping[str, Any]],
    primary_w: float,
) -> Dict[str, Any]:
    matches = [
        row
        for row in summaries
        if math.isclose(float(row["focal_w"]), float(primary_w), abs_tol=1.0e-12)
    ]
    if len(matches) != 1:
        raise ValueError("primary shared-queue summary is missing or duplicated")
    primary = dict(matches[0])
    passed = bool(primary["gate_pass"])
    return {
        "primary_w": float(primary_w),
        "oracle_gate_pass": passed,
        "verdict": (
            "shared_queue_oracle_gate_pass"
            if passed
            else "shared_queue_oracle_gate_fail"
        ),
        "primary_summary": primary,
        "authorization": (
            "run_exact_rollout_target_next"
            if passed
            else "do_not_test_target_or_models_repair_integration"
        ),
        "gate_definition": {
            "certification_rate": 1.0,
            "bound_violations": 0,
            "path_failures": 0,
            "anchor_lower_bound_failures": 0,
            "required_win_fraction": "configured; ceil(fraction * worlds)",
            "mean_delta_vs_matched_euclid_focal": "strictly_negative",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-D shared-queue oracle gate")
    parser.add_argument("--study-dir", default=SharedQueueConfig.study_dir)
    parser.add_argument(
        "--independent-dir", default=SharedQueueConfig.independent_dir
    )
    parser.add_argument("--out-dir", default=SharedQueueConfig.out_dir)
    parser.add_argument("--focal-ws", default=SharedQueueConfig.focal_ws)
    parser.add_argument("--primary-w", type=float, default=SharedQueueConfig.primary_w)
    parser.add_argument(
        "--budget-factor", type=float, default=SharedQueueConfig.budget_factor
    )
    parser.add_argument(
        "--required-win-fraction",
        type=float,
        default=SharedQueueConfig.required_win_fraction,
    )
    return parser.parse_args()


def _resolve_default_paths(cfg: SharedQueueConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    study_was_resolved = False
    if cfg.study_dir == SharedQueueConfig.study_dir:
        candidate = script_dir / cfg.study_dir
        if not Path(cfg.study_dir).exists() and candidate.exists():
            cfg.study_dir = str(candidate)
            study_was_resolved = True
    if cfg.independent_dir == SharedQueueConfig.independent_dir:
        candidate = script_dir / cfg.independent_dir
        if not Path(cfg.independent_dir).exists() and candidate.exists():
            cfg.independent_dir = str(candidate)
    if cfg.out_dir == SharedQueueConfig.out_dir and study_was_resolved:
        cfg.out_dir = str(script_dir / cfg.out_dir)


def main() -> None:
    cfg = SharedQueueConfig(**vars(parse_args()))
    _resolve_default_paths(cfg)
    if not (0.0 < float(cfg.required_win_fraction) <= 1.0):
        raise ValueError("required-win-fraction must be in (0, 1]")
    focal_ws = C13.parse_float_csv(cfg.focal_ws)
    if not any(
        math.isclose(float(value), float(cfg.primary_w), abs_tol=1.0e-12)
        for value in focal_ws
    ):
        raise ValueError("primary-w must appear in focal-ws")

    study_cfg, source_manifest = S.load_study(cfg.study_dir)
    bundles = I.collect_audit_bundles(study_cfg)
    replay = S.verify_audit_replay(cfg.study_dir, bundles)
    independent_rows = load_independent_oracle_rows(cfg.independent_dir)
    rows, baselines, anchor_checks = evaluate_oracle_gate(
        cfg,
        study_cfg,
        bundles,
        independent_rows,
    )
    summaries = summarize_rows(rows, cfg.required_win_fraction)
    gate = build_gate_verdict(summaries, cfg.primary_w)

    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    raw_path = C13.write_csv(result_dir / "shared_queue_oracle_raw.csv", rows)
    baseline_path = C13.write_csv(
        result_dir / "shared_queue_oracle_baselines.csv", baselines
    )
    summary_path = C13.write_csv(
        result_dir / "shared_queue_oracle_summary.csv", summaries
    )
    gate_path = C13.write_json(result_dir / "gate_verdict.json", gate)

    keys = [(int(row["world_index"]), float(row["focal_w"])) for row in rows]
    verification = {
        "audit_replay": replay,
        "worlds": int(len(bundles)),
        "focal_ws": focal_ws,
        "raw_rows": int(len(rows)),
        "expected_raw_rows": int(len(bundles) * len(focal_ws)),
        "duplicate_keys": int(len(keys) - len(set(keys))),
        "certification_failures": int(
            np.sum([not bool(row["certified"]) for row in rows])
        ),
        "path_failures": int(np.sum([not bool(row["path_valid"]) for row in rows])),
        "expansion_accounting_failures": int(
            np.sum([not bool(row["expansion_accounting_valid"]) for row in rows])
        ),
        "bound_violations_eval_only": int(
            np.sum([bool(row["bound_violation_eval_only"]) for row in rows])
        ),
        "anchor_lower_bound_failures_eval_only": int(
            np.sum(
                [
                    bool(row["anchor_lower_bound_exceeds_optimal_eval_only"])
                    for row in rows
                ]
            )
        ),
        "states_expanded_more_than_twice": int(
            np.sum([int(row["max_expansions_per_state"]) > 2 for row in rows])
        ),
        "anchor_checks": anchor_checks,
        "maximum_anchor_consistency_violation": float(
            max(float(row["max_consistency_violation"]) for row in anchor_checks)
        ),
        "training_performed": False,
        "shortest_path_target": False,
        "shortest_path_oracle_control": "privileged_rank_queue_only",
        "proof_anchor": "euclidean_consistent_admissible",
    }
    hard_failures = (
        verification["duplicate_keys"]
        + verification["certification_failures"]
        + verification["path_failures"]
        + verification["expansion_accounting_failures"]
        + verification["bound_violations_eval_only"]
        + verification["anchor_lower_bound_failures_eval_only"]
        + verification["states_expanded_more_than_twice"]
    )
    if verification["raw_rows"] != verification["expected_raw_rows"]:
        raise RuntimeError("shared-queue output row count mismatch")
    if hard_failures:
        raise RuntimeError("shared-queue verification invariant failed")
    verification_path = C13.write_json(result_dir / "verification.json", verification)

    source_paths = {
        "implementation": Path(__file__).resolve(),
        "shared_certificate_helper": Path(S.__file__).resolve(),
        "source_study_manifest": Path(cfg.study_dir) / "manifest.json",
        "source_target_audit": Path(cfg.study_dir)
        / "results"
        / "target_reliability_raw.csv",
        "source_independent_raw": Path(cfg.independent_dir)
        / "results"
        / "certified_search_raw.csv",
    }
    output_paths = {
        "raw": raw_path,
        "baselines": baseline_path,
        "summary": summary_path,
        "gate": gate_path,
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
        "experiment": "C13-D shared-state oracle integration gate",
        "runner_config": asdict(cfg),
        "source_study_config": asdict(study_cfg),
        "source_study_experiment": source_manifest.get("experiment"),
        "training_performed": False,
        "shortest_path_target": False,
        "oracle_control": "privileged_graph_distance_rank_queue",
        "proof_anchor": "euclidean_consistent_admissible",
        "search": "one_anchor_one_rank_shared_g_SMHA_style_w1_1",
        "termination": "incumbent_le_w_times_shared_anchor_open_lower_bound",
        "expansion_accounting": "all_queue_expansions_including_cross_queue_duplicates",
        "algorithm_reference": "https://cdn.aaai.org/ojs/18306/18306-77-21822-1-2-20210717.pdf",
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "integrity": str(integrity_path),
    }
    manifest_path = C13.write_json(Path(cfg.out_dir) / "manifest.json", manifest)

    print(f"verdict={gate['verdict']}")
    print(f"authorization={gate['authorization']}")
    for name, path in {
        **output_paths,
        "integrity": integrity_path,
        "manifest": manifest_path,
    }.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
