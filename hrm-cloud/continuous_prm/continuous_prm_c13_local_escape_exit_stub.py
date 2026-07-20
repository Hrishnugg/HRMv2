#!/usr/bin/env python3
"""C13-G2: exact bounded local-escape exit-stub heuristic ceiling.

The target is a LoHA*-style local heuristic.  For each current roadmap node,
the observer exposes only nodes and internal edges within a fixed physical
radius plus the cost and Euclidean terminal value of each immediately outgoing
action. Local Dijkstra terminates at either the goal (if observed) or one of
those exit stubs.

No map-wide raster, graph distance-to-goal, or global search result is used by
the target.  Graph shortest paths are read only after target construction for
admissibility and outcome evaluation.  This file first tests the exact local
ceiling; learning is authorized only after fresh-world replication.
"""
from __future__ import annotations

import argparse
import heapq
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
import continuous_prm_c13_shared_queue as Q
import continuous_prm_c13_shared_queue_target as T
import continuous_prm_c13_state_heuristic as C13


@dataclass
class LocalObservation:
    relative_points: np.ndarray
    euclid_to_goal: np.ndarray
    adjacency: List[List[Tuple[int, float]]]
    frontier: np.ndarray
    exit_terminals: List[List[float]]
    start_local: int
    goal_local: int


@dataclass
class LocalEscapeConfig:
    study_dir: str = "runs/c13_identifiability"
    oracle_dir: str = "runs/c13_shared_queue_oracle"
    out_dir: str = "runs/c13_local_escape_exit_stub"
    radius_fracs: str = "0.10,0.15,0.20,0.25,0.30"
    primary_radius_frac: float = 0.20
    focal_ws: str = "1.05,1.10,1.25"
    primary_w: float = 1.10
    budget_factor: float = 2.0
    required_win_fraction: float = 0.80


def provider_name(radius_frac: float) -> str:
    return f"local_escape_exit_stub_r{float(radius_frac):.2f}".replace(".", "p")


def extract_local_observation(
    points: np.ndarray,
    adj: Sequence[Sequence[Tuple[int, float]]],
    current_idx: int,
    goal_point: np.ndarray,
    radius: float,
    goal_idx: int = 1,
) -> LocalObservation:
    """Simulate a radius-bounded local graph observation.

    The observation exposes no outside topology. For an action leaving the
    radius it exposes only that one-step edge cost and the endpoint's
    Euclidean-to-goal terminal value, as specified by the preregistered target.
    """

    coords = np.asarray(points, dtype=np.float64)
    goal = np.asarray(goal_point, dtype=np.float64).reshape(2)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(adj):
        raise ValueError("points and adjacency must describe the same 2-D graph")
    if not 0 <= int(current_idx) < len(coords):
        raise ValueError("current_idx is outside the graph")
    if not float(radius) > 0.0:
        raise ValueError("radius must be positive")

    center = coords[int(current_idx)]
    inside_mask = np.linalg.norm(coords - center[None, :], axis=1) <= (
        float(radius) + C.EPS
    )
    inside_mask[int(current_idx)] = True
    global_ids = np.flatnonzero(inside_mask).astype(np.int64)
    global_to_local = {int(node): index for index, node in enumerate(global_ids)}
    local_adj: List[List[Tuple[int, float]]] = [[] for _ in global_ids]
    frontier = np.zeros(len(global_ids), dtype=np.bool_)
    exit_terminals: List[List[float]] = [[] for _ in global_ids]

    for local_u, global_u_value in enumerate(global_ids):
        global_u = int(global_u_value)
        for global_v_value, edge_cost in adj[global_u]:
            global_v = int(global_v_value)
            local_v = global_to_local.get(global_v)
            if local_v is None:
                frontier[local_u] = True
                exit_terminals[local_u].append(
                    float(edge_cost) + float(np.linalg.norm(coords[global_v] - goal))
                )
                continue
            local_adj[local_u].append((int(local_v), float(edge_cost)))

    local_points = coords[global_ids]
    return LocalObservation(
        relative_points=local_points - center[None, :],
        euclid_to_goal=np.linalg.norm(local_points - goal[None, :], axis=1),
        adjacency=local_adj,
        frontier=frontier,
        exit_terminals=exit_terminals,
        start_local=int(global_to_local[int(current_idx)]),
        goal_local=int(global_to_local.get(int(goal_idx), -1)),
    )


def exact_local_escape(observation: LocalObservation) -> Dict[str, Any]:
    """Solve the observed local multi-goal problem without global graph data."""

    n = len(observation.adjacency)
    if observation.euclid_to_goal.shape != (n,):
        raise ValueError("local Euclidean values are not node-aligned")
    if observation.frontier.shape != (n,):
        raise ValueError("frontier mask is not node-aligned")
    if len(observation.exit_terminals) != n:
        raise ValueError("exit terminals are not node-aligned")
    if not 0 <= int(observation.start_local) < n:
        raise ValueError("local start index is invalid")

    start = int(observation.start_local)
    distances = np.full(n, np.inf, dtype=np.float64)
    distances[start] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, start)]
    best = float("inf")
    explored = 0

    while heap:
        distance, node = heapq.heappop(heap)
        if distance != float(distances[node]):
            continue
        if distance >= best - C.EPS:
            continue
        explored += 1
        if node == int(observation.goal_local):
            best = min(best, float(distance))
        elif bool(observation.frontier[node]):
            best = min(best, float(distance) + min(observation.exit_terminals[node]))
        for neighbor, edge_cost in observation.adjacency[node]:
            candidate = float(distance) + float(edge_cost)
            if candidate + C.EPS >= float(distances[neighbor]):
                continue
            distances[neighbor] = candidate
            heapq.heappush(heap, (candidate, int(neighbor)))

    fallback = not math.isfinite(best)
    if fallback:
        best = float(observation.euclid_to_goal[start])
    best = max(float(best), float(observation.euclid_to_goal[start]))
    return {
        "value": float(best),
        "fallback": bool(fallback),
        "explored_local_nodes": int(explored),
        "observed_nodes": int(n),
        "frontier_nodes": int(np.sum(observation.frontier)),
        "exit_actions": int(sum(len(values) for values in observation.exit_terminals)),
        "goal_observed": bool(int(observation.goal_local) >= 0),
    }


def compute_local_escape_field(
    points: np.ndarray,
    adj: Sequence[Sequence[Tuple[int, float]]],
    goal_point: np.ndarray,
    radius: float,
    goal_idx: int = 1,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    values = np.empty(len(points), dtype=np.float64)
    diagnostics: List[Dict[str, Any]] = []
    for node in range(len(points)):
        observation = extract_local_observation(
            points,
            adj,
            node,
            goal_point,
            radius,
            goal_idx=goal_idx,
        )
        result = exact_local_escape(observation)
        values[node] = float(result["value"])
        diagnostics.append({"node": int(node), **result})
    values[int(goal_idx)] = 0.0
    return values, diagnostics


def field_diagnostics(
    bundle: I.AuditBundle,
    radius_frac: float,
    field: np.ndarray,
    node_diagnostics: Sequence[Mapping[str, Any]],
    seconds: float,
) -> Dict[str, Any]:
    rm = bundle.roadmap
    euclid = C13.euclidean_to_goal(rm.points, rm.points[1])
    oracle = np.asarray(rm.dist_to_goal, dtype=np.float64)  # evaluation-only
    connected = np.isfinite(oracle) & (oracle < C.INF / 10.0)
    residual = np.asarray(field, dtype=np.float64) - euclid
    fallback = np.asarray(
        [bool(row["fallback"]) for row in node_diagnostics], dtype=np.bool_
    )
    dominance = int(np.sum(residual < -1.0e-9))
    admissibility = int(np.sum(field[connected] > oracle[connected] + 1.0e-9))
    consistency_violations = 0
    max_consistency_violation = 0.0
    for node, neighbors in enumerate(rm.adj):
        for neighbor, edge_cost in neighbors:
            violation = float(field[node]) - (
                float(edge_cost) + float(field[int(neighbor)])
            )
            if violation > 1.0e-9:
                consistency_violations += 1
                max_consistency_violation = max(max_consistency_violation, violation)
    positive = residual > 1.0e-9
    return {
        "suite": str(bundle.node_rows[0]["suite"]),
        "world_index": int(bundle.world_index),
        "world_seed": int(bundle.world_seed),
        "radius_frac": float(radius_frac),
        "radius_abs": float(radius_frac * bundle.world.side_len),
        "nodes": int(len(field)),
        "local_nodes_mean": float(
            np.mean([int(row["observed_nodes"]) for row in node_diagnostics])
        ),
        "local_nodes_max": int(
            np.max([int(row["observed_nodes"]) for row in node_diagnostics])
        ),
        "frontier_nodes_mean": float(
            np.mean([int(row["frontier_nodes"]) for row in node_diagnostics])
        ),
        "exit_actions_mean": float(
            np.mean([int(row["exit_actions"]) for row in node_diagnostics])
        ),
        "local_nodes_explored_mean": float(
            np.mean([int(row["explored_local_nodes"]) for row in node_diagnostics])
        ),
        "fallback_nodes": int(np.sum(fallback)),
        "fallback_nodes_reachable": int(np.sum(fallback & connected)),
        "fallback_nodes_unreachable": int(np.sum(fallback & ~connected)),
        "goal_observed_rate": float(
            np.mean([bool(row["goal_observed"]) for row in node_diagnostics])
        ),
        "positive_residual_rate": float(np.mean(positive)),
        "residual_mean_over_side_len": float(
            np.mean(residual) / float(bundle.world.side_len)
        ),
        "residual_p95_over_side_len": float(
            np.percentile(residual, 95) / float(bundle.world.side_len)
        ),
        "start_h_over_oracle_eval_only": float(field[0] / oracle[0]),
        "field_vs_oracle_spearman_eval_only": I.safe_spearman(
            field[connected], oracle[connected]
        ),
        "euclid_dominance_violations": dominance,
        "oracle_admissibility_violations_eval_only": admissibility,
        "max_oracle_overestimate_eval_only": float(
            max(0.0, np.max(field[connected] - oracle[connected]))
        ),
        "goal_boundary_violation": bool(abs(float(field[1])) > 1.0e-9),
        "edge_consistency_violations": int(consistency_violations),
        "max_edge_consistency_violation": float(max_consistency_violation),
        "target_seconds": float(seconds),
    }


def evaluate_local_escape(
    cfg: LocalEscapeConfig,
    study_cfg: I.StudyConfig,
    bundles: Sequence[I.AuditBundle],
    oracle_rows: Mapping[Tuple[int, float], Mapping[str, str]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    radius_fracs = C13.parse_float_csv(cfg.radius_fracs)
    focal_ws = C13.parse_float_csv(cfg.focal_ws)
    rows: List[Dict[str, Any]] = []
    baselines: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    target_summaries: List[Dict[str, Any]] = []
    anchor_checks: List[Dict[str, Any]] = []

    expected_reference_keys = {
        (int(bundle.world_index), float(focal_w))
        for bundle in bundles
        for focal_w in focal_ws
    }
    missing = sorted(expected_reference_keys - set(oracle_rows))
    if missing:
        raise ValueError(f"shared oracle reference is missing keys: {missing}")

    for bundle in bundles:
        rm = bundle.roadmap
        euclid = C13.euclidean_to_goal(rm.points, rm.points[1])
        optimal = float(rm.dist_to_goal[0])  # evaluation-only
        euclid_astar = C.astar_search(rm.adj, euclid, len(rm.points))
        budget = max(0, int(math.ceil(float(cfg.budget_factor) * len(rm.points))))
        anchor_checks.append(
            {
                "world_index": int(bundle.world_index),
                "world_seed": int(bundle.world_seed),
                **S.validate_consistent_anchor(rm.adj, euclid),
            }
        )

        baseline_by_w: Dict[float, Dict[str, Any]] = {}
        euclid_shared_by_w: Dict[float, Dict[str, Any]] = {}
        for focal_w in focal_ws:
            baseline = I.focal_search_with_secondary(
                rm.adj,
                euclid,
                euclid,
                budget=len(rm.points),
                w=float(focal_w),
                secondary="h",
            )
            shared_control = Q.shared_anchor_certified_search(
                rm.adj,
                euclid,
                euclid,
                w=float(focal_w),
                budget=budget,
                validate_anchor=False,
            )
            baseline_by_w[float(focal_w)] = baseline
            euclid_shared_by_w[float(focal_w)] = shared_control
            baselines.append(
                {
                    "suite": study_cfg.suite,
                    "world_index": int(bundle.world_index),
                    "world_seed": int(bundle.world_seed),
                    "focal_w": float(focal_w),
                    "euclid_focal_expansions": int(baseline["expansions"]),
                    "euclid_focal_cost": float(baseline["cost"]),
                    "euclid_astar_expansions": int(euclid_astar["expansions"]),
                    "euclid_astar_cost": float(euclid_astar["cost"]),
                    "same_search_euclid_expansions": int(
                        shared_control["expansions"]
                    ),
                    "same_search_euclid_certified": bool(
                        shared_control["certified"]
                    ),
                }
            )

        for radius_frac in radius_fracs:
            started = time.perf_counter()
            field, node_diagnostics = compute_local_escape_field(
                rm.points,
                rm.adj,
                rm.points[1],
                float(radius_frac) * float(bundle.world.side_len),
            )
            target_seconds = time.perf_counter() - started
            target_summary = field_diagnostics(
                bundle,
                float(radius_frac),
                field,
                node_diagnostics,
                target_seconds,
            )
            target_summaries.append(target_summary)
            for node_row, node_result in zip(bundle.node_rows, node_diagnostics):
                node = int(node_result["node"])
                target_rows.append(
                    {
                        "suite": study_cfg.suite,
                        "world_index": int(bundle.world_index),
                        "world_seed": int(bundle.world_seed),
                        "radius_frac": float(radius_frac),
                        "node": node,
                        "local_escape": float(field[node]),
                        "euclid": float(euclid[node]),
                        "oracle_eval_only": float(rm.dist_to_goal[node]),
                        "residual": float(field[node] - euclid[node]),
                        "observed_nodes": int(node_result["observed_nodes"]),
                        "frontier_nodes": int(node_result["frontier_nodes"]),
                        "exit_actions": int(node_result["exit_actions"]),
                        "explored_local_nodes": int(
                            node_result["explored_local_nodes"]
                        ),
                        "fallback": bool(node_result["fallback"]),
                        "goal_observed": bool(node_result["goal_observed"]),
                        "source_rollout_label_available": bool(
                            node_row.get("rollout_median") not in (None, "")
                        ),
                    }
                )

            direct_astar = C.astar_search(rm.adj, field, len(rm.points))
            for focal_w in focal_ws:
                key = (int(bundle.world_index), float(focal_w))
                oracle_reference = oracle_rows[key]
                if int(oracle_reference["world_seed"]) != int(bundle.world_seed):
                    raise ValueError(f"oracle reference world mismatch at {key}")
                baseline = baseline_by_w[float(focal_w)]
                shared_control = euclid_shared_by_w[float(focal_w)]
                result = Q.shared_anchor_certified_search(
                    rm.adj,
                    euclid,
                    field,
                    w=float(focal_w),
                    budget=budget,
                    validate_anchor=False,
                )
                final_cost = float(result["final_cost"])
                path_check = Q.validate_path(rm.adj, result["path"], final_cost)
                rows.append(
                    {
                        "suite": study_cfg.suite,
                        "world_index": int(bundle.world_index),
                        "world_seed": int(bundle.world_seed),
                        "provider": provider_name(radius_frac),
                        "radius_frac": float(radius_frac),
                        "focal_w": float(focal_w),
                        "certified": bool(result["certified"]),
                        "found": bool(result["found"]),
                        "proof": result["proof"],
                        "final_cost": (
                            final_cost if math.isfinite(final_cost) else ""
                        ),
                        "final_cost_ratio_eval_only": (
                            final_cost / optimal if math.isfinite(final_cost) else ""
                        ),
                        "bound_violation_eval_only": bool(
                            not math.isfinite(final_cost)
                            or final_cost > float(focal_w) * optimal + 1.0e-9
                        ),
                        "anchor_lower_bound": float(result["lower_bound"]),
                        "anchor_lower_bound_exceeds_optimal_eval_only": bool(
                            float(result["lower_bound"]) > optimal + 1.0e-9
                        ),
                        "certificate_ratio": float(result["certificate_ratio"]),
                        "path_valid": bool(path_check["valid"]),
                        "path_cost": float(path_check["cost"]),
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
                        "rank_eligible_choices": int(
                            result["rank_eligible_choices"]
                        ),
                        "rank_eligible_choice_rate": float(
                            result["rank_eligible_choices"]
                            / max(1, result["rank_eligibility_checks"])
                        ),
                        "search_seconds": float(result["seconds"]),
                        "target_seconds_all_nodes": float(target_seconds),
                        "local_nodes_mean": float(
                            target_summary["local_nodes_mean"]
                        ),
                        "positive_residual_rate": float(
                            target_summary["positive_residual_rate"]
                        ),
                        "start_h_over_oracle_eval_only": float(
                            target_summary["start_h_over_oracle_eval_only"]
                        ),
                        "euclid_focal_expansions": int(baseline["expansions"]),
                        "delta_vs_euclid_focal": int(result["expansions"])
                        - int(baseline["expansions"]),
                        "same_search_euclid_expansions": int(
                            shared_control["expansions"]
                        ),
                        "delta_vs_same_search_euclid": int(result["expansions"])
                        - int(shared_control["expansions"]),
                        "euclid_astar_expansions": int(euclid_astar["expansions"]),
                        "direct_local_astar_expansions": int(
                            direct_astar["expansions"]
                        ),
                        "direct_local_astar_cost_ratio_eval_only": float(
                            direct_astar["cost"] / optimal
                        ),
                        "shared_oracle_expansions": int(
                            oracle_reference["expansions"]
                        ),
                        "delta_vs_shared_oracle": int(result["expansions"])
                        - int(oracle_reference["expansions"]),
                    }
                )
    return rows, baselines, target_rows, target_summaries, anchor_checks


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    target_summaries: Sequence[Mapping[str, Any]],
    required_win_fraction: float,
) -> List[Dict[str, Any]]:
    target_by_radius: DefaultDict[float, List[Mapping[str, Any]]] = defaultdict(list)
    for row in target_summaries:
        target_by_radius[float(row["radius_frac"])].append(row)
    grouped: DefaultDict[Tuple[float, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["radius_frac"]), float(row["focal_w"]))].append(row)

    summaries: List[Dict[str, Any]] = []
    for (radius_frac, focal_w), group in sorted(grouped.items()):
        target_group = target_by_radius[radius_frac]
        focal_deltas = np.asarray(
            [float(row["delta_vs_euclid_focal"]) for row in group]
        )
        control_deltas = np.asarray(
            [float(row["delta_vs_same_search_euclid"]) for row in group]
        )
        required_wins = int(math.ceil(float(required_win_fraction) * len(group)))
        focal_wins = int(np.sum(focal_deltas < 0.0))
        control_wins = int(np.sum(control_deltas < 0.0))
        safety_failures = int(
            np.sum([not bool(row["certified"]) for row in group])
            + np.sum([bool(row["bound_violation_eval_only"]) for row in group])
            + np.sum([not bool(row["path_valid"]) for row in group])
            + np.sum(
                [
                    bool(row["anchor_lower_bound_exceeds_optimal_eval_only"])
                    for row in group
                ]
            )
            + np.sum(
                [not bool(row["expansion_accounting_valid"]) for row in group]
            )
        )
        target_failures = int(
            np.sum(
                [int(row["euclid_dominance_violations"]) for row in target_group]
            )
            + np.sum(
                [
                    int(row["oracle_admissibility_violations_eval_only"])
                    for row in target_group
                ]
            )
            + np.sum([bool(row["goal_boundary_violation"]) for row in target_group])
            + np.sum(
                [
                    int(row.get("fallback_nodes_reachable", row["fallback_nodes"]))
                    for row in target_group
                ]
            )
        )
        mean_focal_delta = float(np.mean(focal_deltas))
        mean_control_delta = float(np.mean(control_deltas))
        total_checks = int(
            np.sum([int(row["rank_eligibility_checks"]) for row in group])
        )
        total_choices = int(
            np.sum([int(row["rank_eligible_choices"]) for row in group])
        )
        summaries.append(
            {
                "provider": provider_name(radius_frac),
                "radius_frac": float(radius_frac),
                "focal_w": float(focal_w),
                "worlds": int(len(group)),
                "required_wins": int(required_wins),
                "gate_pass": bool(
                    safety_failures == 0
                    and target_failures == 0
                    and focal_wins >= required_wins
                    and control_wins >= required_wins
                    and mean_focal_delta < 0.0
                    and mean_control_delta < 0.0
                ),
                "search_safety_failures": safety_failures,
                "target_validity_failures": target_failures,
                "expansions_mean": float(
                    np.mean([float(row["expansions"]) for row in group])
                ),
                "rank_expansions_mean": float(
                    np.mean([float(row["rank_expansions"]) for row in group])
                ),
                "anchor_expansions_mean": float(
                    np.mean([float(row["anchor_expansions"]) for row in group])
                ),
                "rank_eligible_choice_rate": float(
                    total_choices / max(1, total_checks)
                ),
                "euclid_focal_expansions_mean": float(
                    np.mean([float(row["euclid_focal_expansions"]) for row in group])
                ),
                "delta_vs_euclid_focal_mean": mean_focal_delta,
                "focal_wins": focal_wins,
                "focal_ties": int(np.sum(focal_deltas == 0.0)),
                "focal_losses": int(np.sum(focal_deltas > 0.0)),
                "same_search_euclid_expansions_mean": float(
                    np.mean(
                        [float(row["same_search_euclid_expansions"]) for row in group]
                    )
                ),
                "delta_vs_same_search_euclid_mean": mean_control_delta,
                "same_search_euclid_wins": control_wins,
                "same_search_euclid_ties": int(np.sum(control_deltas == 0.0)),
                "same_search_euclid_losses": int(np.sum(control_deltas > 0.0)),
                "shared_oracle_expansions_mean": float(
                    np.mean([float(row["shared_oracle_expansions"]) for row in group])
                ),
                "direct_local_astar_expansions_mean": float(
                    np.mean(
                        [float(row["direct_local_astar_expansions"]) for row in group]
                    )
                ),
                "direct_local_astar_cost_ratio_max_eval_only": float(
                    np.max(
                        [
                            float(row["direct_local_astar_cost_ratio_eval_only"])
                            for row in group
                        ]
                    )
                ),
                "final_cost_ratio_mean_eval_only": float(
                    np.mean([float(row["final_cost_ratio_eval_only"]) for row in group])
                ),
                "final_cost_ratio_max_eval_only": float(
                    np.max([float(row["final_cost_ratio_eval_only"]) for row in group])
                ),
                "local_nodes_mean": float(
                    np.mean([float(row["local_nodes_mean"]) for row in target_group])
                ),
                "positive_residual_rate_mean": float(
                    np.mean(
                        [float(row["positive_residual_rate"]) for row in target_group]
                    )
                ),
                "start_h_over_oracle_mean_eval_only": float(
                    np.mean(
                        [
                            float(row["start_h_over_oracle_eval_only"])
                            for row in target_group
                        ]
                    )
                ),
                "edge_consistency_violations": int(
                    np.sum(
                        [int(row["edge_consistency_violations"]) for row in target_group]
                    )
                ),
                "target_seconds_mean": float(
                    np.mean([float(row["target_seconds"]) for row in target_group])
                ),
            }
        )
    return summaries


def build_verdict(
    summaries: Sequence[Mapping[str, Any]],
    primary_w: float,
    primary_radius_frac: float,
) -> Dict[str, Any]:
    primary = [
        dict(row)
        for row in summaries
        if math.isclose(float(row["focal_w"]), float(primary_w), abs_tol=1.0e-12)
        and math.isclose(
            float(row["radius_frac"]), float(primary_radius_frac), abs_tol=1.0e-12
        )
    ]
    if len(primary) != 1:
        raise ValueError("primary local-escape summary is missing or duplicated")
    passing = [
        dict(row)
        for row in summaries
        if math.isclose(float(row["focal_w"]), float(primary_w), abs_tol=1.0e-12)
        and bool(row["gate_pass"])
    ]
    primary_pass = bool(primary[0]["gate_pass"])
    selected = primary[0] if primary_pass else (
        min(
            passing,
            key=lambda row: (float(row["expansions_mean"]), float(row["radius_frac"])),
        )
        if passing
        else None
    )
    if primary_pass:
        verdict = "primary_exact_local_escape_exit_stub_gate_pass_development_only"
        authorization = "replicate_primary_radius_on_fresh_192_and_211_worlds"
    elif selected is not None:
        verdict = "alternate_local_escape_exit_stub_radius_found_development_only"
        authorization = "replicate_fixed_alternate_radius_on_fresh_worlds"
    else:
        verdict = "exact_local_escape_exit_stub_gate_fail"
        authorization = "move_to_current_state_policy_or_td_target"
    return {
        "primary_w": float(primary_w),
        "primary_radius_frac": float(primary_radius_frac),
        "development_cohort_reused": True,
        "primary_gate_pass": primary_pass,
        "any_radius_gate_pass": bool(passing),
        "verdict": verdict,
        "authorization": authorization,
        "primary_summary": primary[0],
        "selected_candidate": selected,
        "fresh_world_replication_required": selected is not None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-G2 exact local-escape exit-stub ceiling")
    parser.add_argument("--study-dir", default=LocalEscapeConfig.study_dir)
    parser.add_argument("--oracle-dir", default=LocalEscapeConfig.oracle_dir)
    parser.add_argument("--out-dir", default=LocalEscapeConfig.out_dir)
    parser.add_argument("--radius-fracs", default=LocalEscapeConfig.radius_fracs)
    parser.add_argument(
        "--primary-radius-frac",
        type=float,
        default=LocalEscapeConfig.primary_radius_frac,
    )
    parser.add_argument("--focal-ws", default=LocalEscapeConfig.focal_ws)
    parser.add_argument("--primary-w", type=float, default=LocalEscapeConfig.primary_w)
    parser.add_argument(
        "--budget-factor", type=float, default=LocalEscapeConfig.budget_factor
    )
    parser.add_argument(
        "--required-win-fraction",
        type=float,
        default=LocalEscapeConfig.required_win_fraction,
    )
    return parser.parse_args()


def _resolve_default_paths(cfg: LocalEscapeConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    study_was_resolved = False
    for field, default in (
        ("study_dir", LocalEscapeConfig.study_dir),
        ("oracle_dir", LocalEscapeConfig.oracle_dir),
    ):
        value = getattr(cfg, field)
        if value != default:
            continue
        candidate = script_dir / value
        if not Path(value).exists() and candidate.exists():
            setattr(cfg, field, str(candidate))
            if field == "study_dir":
                study_was_resolved = True
    if cfg.out_dir == LocalEscapeConfig.out_dir and study_was_resolved:
        cfg.out_dir = str(script_dir / cfg.out_dir)


def main() -> None:
    cfg = LocalEscapeConfig(**vars(parse_args()))
    _resolve_default_paths(cfg)
    radius_fracs = C13.parse_float_csv(cfg.radius_fracs)
    focal_ws = C13.parse_float_csv(cfg.focal_ws)
    if not any(
        math.isclose(value, cfg.primary_radius_frac, abs_tol=1.0e-12)
        for value in radius_fracs
    ):
        raise ValueError("primary-radius-frac must appear in radius-fracs")
    if not any(
        math.isclose(value, cfg.primary_w, abs_tol=1.0e-12)
        for value in focal_ws
    ):
        raise ValueError("primary-w must appear in focal-ws")

    study_cfg, source_manifest = S.load_study(cfg.study_dir)
    bundles = I.collect_audit_bundles(study_cfg)
    replay = S.verify_audit_replay(cfg.study_dir, bundles)
    oracle_raw = (
        Path(cfg.oracle_dir) / "results" / "shared_queue_oracle_raw.csv"
    )
    oracle_rows = T.load_reference_rows(oracle_raw, "oracle_eval_only")
    rows, baselines, target_rows, target_summaries, anchor_checks = (
        evaluate_local_escape(cfg, study_cfg, bundles, oracle_rows)
    )
    summaries = summarize_rows(rows, target_summaries, cfg.required_win_fraction)
    verdict = build_verdict(summaries, cfg.primary_w, cfg.primary_radius_frac)

    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    raw_path = C13.write_csv(result_dir / "local_escape_search_raw.csv", rows)
    baseline_path = C13.write_csv(
        result_dir / "local_escape_baselines.csv", baselines
    )
    target_raw_path = C13.write_csv(
        result_dir / "local_escape_target_raw.csv", target_rows
    )
    target_summary_path = C13.write_csv(
        result_dir / "local_escape_target_summary.csv", target_summaries
    )
    summary_path = C13.write_csv(
        result_dir / "local_escape_search_summary.csv", summaries
    )
    verdict_path = C13.write_json(result_dir / "gate_verdict.json", verdict)

    keys = [
        (float(row["radius_frac"]), int(row["world_index"]), float(row["focal_w"]))
        for row in rows
    ]
    verification = {
        "audit_replay": replay,
        "worlds": int(len(bundles)),
        "radius_fracs": radius_fracs,
        "focal_ws": focal_ws,
        "raw_rows": int(len(rows)),
        "expected_raw_rows": int(len(bundles) * len(radius_fracs) * len(focal_ws)),
        "target_rows": int(len(target_rows)),
        "expected_target_rows": int(
            len(bundles) * len(radius_fracs) * study_cfg.roadmap_nodes
        ),
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
        "target_euclid_dominance_violations": int(
            np.sum(
                [int(row["euclid_dominance_violations"]) for row in target_summaries]
            )
        ),
        "target_oracle_admissibility_violations_eval_only": int(
            np.sum(
                [
                    int(row["oracle_admissibility_violations_eval_only"])
                    for row in target_summaries
                ]
            )
        ),
        "target_goal_boundary_violations": int(
            np.sum([bool(row["goal_boundary_violation"]) for row in target_summaries])
        ),
        "target_fallback_nodes": int(
            np.sum([int(row["fallback_nodes"]) for row in target_summaries])
        ),
        "target_fallback_nodes_reachable": int(
            np.sum(
                [int(row["fallback_nodes_reachable"]) for row in target_summaries]
            )
        ),
        "target_fallback_nodes_unreachable": int(
            np.sum(
                [int(row["fallback_nodes_unreachable"]) for row in target_summaries]
            )
        ),
        "anchor_checks": anchor_checks,
        "maximum_anchor_consistency_violation": float(
            max(float(row["max_consistency_violation"]) for row in anchor_checks)
        ),
        "development_cohort_reused": True,
        "fresh_replication_required": verdict["fresh_world_replication_required"],
        "training_performed": False,
        "model_loading_performed": False,
        "shortest_path_target": False,
        "shortest_path_use": "posthoc_admissibility_and_outcome_evaluation_only",
        "runtime_information": "radius_bounded_nodes_internal_edges_exit_edge_costs_endpoint_euclidean_goal_geometry",
    }
    if verification["raw_rows"] != verification["expected_raw_rows"]:
        raise RuntimeError("local-escape search row count mismatch")
    if verification["target_rows"] != verification["expected_target_rows"]:
        raise RuntimeError("local-escape target row count mismatch")
    structural_failures = (
        verification["duplicate_keys"]
        + verification["path_failures"]
        + verification["expansion_accounting_failures"]
        + verification["states_expanded_more_than_twice"]
        + verification["target_euclid_dominance_violations"]
        + verification["target_oracle_admissibility_violations_eval_only"]
        + verification["target_goal_boundary_violations"]
        + verification["target_fallback_nodes_reachable"]
    )
    verification_path = C13.write_json(result_dir / "verification.json", verification)
    if structural_failures:
        raise RuntimeError("local-escape structural or target invariant failed")

    source_paths = {
        "implementation": Path(__file__).resolve(),
        "shared_queue_implementation": Path(Q.__file__).resolve(),
        "source_study_manifest": Path(cfg.study_dir) / "manifest.json",
        "source_target_audit": Path(cfg.study_dir)
        / "results"
        / "target_reliability_raw.csv",
        "source_oracle_raw": oracle_raw,
    }
    output_paths = {
        "raw": raw_path,
        "baselines": baseline_path,
        "target_raw": target_raw_path,
        "target_summary": target_summary_path,
        "summary": summary_path,
        "gate": verdict_path,
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
        "experiment": "C13-G2 exact bounded local-escape exit-stub heuristic ceiling",
        "runner_config": asdict(cfg),
        "source_study_config": asdict(study_cfg),
        "source_study_experiment": source_manifest.get("experiment"),
        "training_performed": False,
        "model_loading_performed": False,
        "shortest_path_target": False,
        "target": "local_dijkstra_to_visible_exit_edge_plus_endpoint_euclidean_terminal",
        "runtime_information": "radius_bounded_nodes_internal_edges_exit_edge_costs_endpoint_euclidean_goal_geometry",
        "full_map_runtime_input": False,
        "development_cohort_reused": True,
        "fresh_replication_required": verdict["fresh_world_replication_required"],
        "literature_reference": "https://ojs.aaai.org/index.php/ICAPS/article/view/27245",
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "integrity": str(integrity_path),
    }
    manifest_path = C13.write_json(Path(cfg.out_dir) / "manifest.json", manifest)

    print(f"verdict={verdict['verdict']}")
    print(f"authorization={verdict['authorization']}")
    if verdict["selected_candidate"] is not None:
        print(
            "selected_radius_frac="
            f"{verdict['selected_candidate']['radius_frac']}"
        )
    for name, path in {
        **output_paths,
        "integrity": integrity_path,
        "manifest": manifest_path,
    }.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
