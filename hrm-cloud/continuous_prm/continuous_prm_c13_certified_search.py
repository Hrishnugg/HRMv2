#!/usr/bin/env python3
"""C13-C: certify an inadmissible incumbent with a fresh Euclidean A*.

This is an integration-only diagnostic.  It reuses the frozen C13-B audit
worlds, rollout aggregates, and learned checkpoints; it performs no training.

Phase 1 runs A* with an arbitrary finite rank and permits node reopening.  The
first goal supplies a feasible, potentially suboptimal incumbent.  Phase 2 is
an independent A* whose Euclidean anchor is checked for consistency.  It stops
only when

    incumbent_cost <= w * min_open(g + h_euclid),

which certifies the requested bound because the anchor OPEN minimum is a lower
bound on optimal cost.  Total work is the sum of both phases, including any
duplicated expansions.  Graph shortest-path distance is used post hoc for
bound validation and, only in the explicitly privileged oracle_eval_only arm,
as the phase-1 integration ceiling.  It is never a learned/rollout model
feature, training label, or certification input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import continuous_prm_common as C
import continuous_prm_c13_identifiability as I
import continuous_prm_c13_state_heuristic as C13


DEFAULT_PROVIDERS = "oracle_eval_only,rollout_exact,flat_mlp,hrm_padded,onlstm_trimmed"


@dataclass
class CertifiedSearchConfig:
    study_dir: str = "runs/c13_identifiability"
    out_dir: str = "runs/c13_certified_search"
    providers: str = DEFAULT_PROVIDERS
    focal_ws: str = "1.05,1.10,1.25"
    primary_w: float = 1.10
    phase1_budget_factor: float = 4.0
    anchor_budget_factor: float = 1.0
    required_win_fraction: float = 0.80
    device: str = "auto"


def resolve_device(name: str) -> torch.device:
    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_consistent_anchor(
    adj: List[List[Tuple[int, float]]],
    heuristic: np.ndarray,
    goal_idx: int = 1,
    tolerance: float = 1.0e-9,
) -> Dict[str, float]:
    """Validate the conditions used by the phase-2 lower-bound proof."""

    h = np.asarray(heuristic, dtype=np.float64).reshape(-1)
    if len(h) != len(adj):
        raise ValueError("anchor length does not match graph")
    if not np.all(np.isfinite(h)):
        raise ValueError("anchor must be finite")
    if float(np.min(h)) < -float(tolerance):
        raise ValueError("anchor must be nonnegative")
    if abs(float(h[goal_idx])) > float(tolerance):
        raise ValueError("anchor must be zero at the goal")

    max_violation = 0.0
    min_edge_cost = float("inf")
    for node, neighbors in enumerate(adj):
        for neighbor, edge_cost in neighbors:
            cost = float(edge_cost)
            if not math.isfinite(cost) or cost < 0.0:
                raise ValueError("edge costs must be finite and nonnegative")
            min_edge_cost = min(min_edge_cost, cost)
            max_violation = max(
                max_violation,
                float(h[node]) - (cost + float(h[int(neighbor)])),
            )
    if max_violation > float(tolerance):
        raise ValueError(f"anchor is inconsistent by {max_violation:.6g}")
    return {
        "max_consistency_violation": float(max_violation),
        "goal_value": float(h[goal_idx]),
        "minimum_value": float(np.min(h)),
        "minimum_edge_cost": float(min_edge_cost),
    }


def inadmissible_astar_incumbent(
    adj: List[List[Tuple[int, float]]],
    heuristic: np.ndarray,
    budget: int,
    start_idx: int = 0,
    goal_idx: int = 1,
) -> Dict[str, Any]:
    """Return the first feasible goal under an arbitrary finite heuristic.

    Reopening is required because the learned/rollout rank need not be
    consistent.  No optimality or suboptimality claim is made for this phase.
    """

    h = np.asarray(heuristic, dtype=np.float64).reshape(-1)
    if len(h) != len(adj) or not np.all(np.isfinite(h)):
        raise ValueError("phase-1 heuristic must be finite and match the graph")
    if int(budget) < 0:
        raise ValueError("budget must be nonnegative")

    g = np.full(len(adj), np.inf, dtype=np.float64)
    g[start_idx] = 0.0
    counter = 0
    opened: List[Tuple[float, float, int, int]] = [
        (float(h[start_idx]), 0.0, int(start_idx), counter)
    ]
    closed = np.zeros(len(adj), dtype=np.bool_)
    expansions = 0
    reopens = 0
    generated = 1

    while opened and expansions < int(budget):
        _, current_g, node, _ = heapq.heappop(opened)
        if current_g != g[node] or closed[node]:
            continue
        closed[node] = True
        expansions += 1
        if node == goal_idx:
            return {
                "found": True,
                "cost": float(g[node]),
                "expansions": int(expansions),
                "reopens": int(reopens),
                "generated": int(generated),
                "closed": int(np.sum(closed)),
            }
        for neighbor, edge_cost in adj[node]:
            neighbor = int(neighbor)
            new_g = float(g[node]) + float(edge_cost)
            if new_g + C.EPS < float(g[neighbor]):
                if closed[neighbor]:
                    closed[neighbor] = False
                    reopens += 1
                g[neighbor] = new_g
                counter += 1
                generated += 1
                heapq.heappush(
                    opened,
                    (
                        new_g + float(h[neighbor]),
                        new_g,
                        neighbor,
                        counter,
                    ),
                )
    return {
        "found": False,
        "cost": float("nan"),
        "expansions": int(expansions),
        "reopens": int(reopens),
        "generated": int(generated),
        "closed": int(np.sum(closed)),
    }


def certify_incumbent_with_anchor(
    adj: List[List[Tuple[int, float]]],
    anchor_h: np.ndarray,
    incumbent_cost: float,
    w: float,
    budget: int,
    start_idx: int = 0,
    goal_idx: int = 1,
    validate_anchor: bool = True,
) -> Dict[str, Any]:
    """Certify a feasible incumbent using a fresh consistent-anchor A*."""

    if float(w) < 1.0:
        raise ValueError("w must be at least one")
    if int(budget) < 0:
        raise ValueError("budget must be nonnegative")
    h = np.asarray(anchor_h, dtype=np.float64).reshape(-1)
    if validate_anchor:
        validate_consistent_anchor(adj, h, goal_idx=goal_idx)

    initial_incumbent = (
        float(incumbent_cost) if math.isfinite(float(incumbent_cost)) else float("inf")
    )
    incumbent = initial_incumbent
    g = np.full(len(adj), np.inf, dtype=np.float64)
    g[start_idx] = 0.0
    counter = 0
    opened: List[Tuple[float, float, int, int]] = [
        (float(h[start_idx]), 0.0, int(start_idx), counter)
    ]
    closed = np.zeros(len(adj), dtype=np.bool_)
    expansions = 0
    generated = 1
    lower_bound = float(h[start_idx])

    while opened:
        while opened:
            entry = opened[0]
            _, entry_g, node, _ = entry
            if closed[node] or entry_g != g[node]:
                heapq.heappop(opened)
                continue
            break
        if not opened:
            break

        lower_bound = float(opened[0][0])
        if math.isfinite(incumbent) and incumbent <= float(w) * lower_bound + C.EPS:
            return {
                "certified": True,
                "found": True,
                "final_cost": float(incumbent),
                "lower_bound": float(lower_bound),
                "certificate_ratio": float(incumbent / max(C.EPS, lower_bound)),
                "expansions": int(expansions),
                "generated": int(generated),
                "anchor_goal_popped": False,
                "anchor_improved_incumbent": bool(incumbent + C.EPS < initial_incumbent),
                "proof": "incumbent_le_w_times_anchor_open_lower_bound",
            }
        if expansions >= int(budget):
            break

        _, current_g, node, _ = heapq.heappop(opened)
        if closed[node] or current_g != g[node]:
            continue
        closed[node] = True
        expansions += 1
        if node == goal_idx:
            if float(g[node]) + C.EPS < incumbent:
                incumbent = float(g[node])
            return {
                "certified": True,
                "found": True,
                "final_cost": float(incumbent),
                "lower_bound": float(g[node]),
                "certificate_ratio": float(incumbent / max(C.EPS, float(g[node]))),
                "expansions": int(expansions),
                "generated": int(generated),
                "anchor_goal_popped": True,
                "anchor_improved_incumbent": bool(incumbent + C.EPS < initial_incumbent),
                "proof": "consistent_anchor_goal_pop_is_optimal",
            }
        for neighbor, edge_cost in adj[node]:
            neighbor = int(neighbor)
            if closed[neighbor]:
                continue
            new_g = float(g[node]) + float(edge_cost)
            if new_g + C.EPS < float(g[neighbor]):
                g[neighbor] = new_g
                counter += 1
                generated += 1
                heapq.heappush(
                    opened,
                    (
                        new_g + float(h[neighbor]),
                        new_g,
                        neighbor,
                        counter,
                    ),
                )

    return {
        "certified": False,
        "found": math.isfinite(incumbent),
        "final_cost": float(incumbent) if math.isfinite(incumbent) else float("nan"),
        "lower_bound": float(lower_bound),
        "certificate_ratio": (
            float(incumbent / max(C.EPS, lower_bound))
            if math.isfinite(incumbent)
            else float("nan")
        ),
        "expansions": int(expansions),
        "generated": int(generated),
        "anchor_goal_popped": False,
        "anchor_improved_incumbent": bool(incumbent + C.EPS < initial_incumbent),
        "proof": "budget_exhausted_without_certificate",
    }


def load_study(study_dir: str | Path) -> Tuple[I.StudyConfig, Dict[str, Any]]:
    manifest_path = Path(study_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("shortest_path_target") is not False:
        raise ValueError("source study does not preserve the no-shortest-path target contract")
    study_cfg = I.StudyConfig(**manifest["config"])
    study_cfg.out_dir = str(Path(study_dir))
    return study_cfg, manifest


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_frozen_models(
    study_dir: str | Path,
    study_cfg: I.StudyConfig,
    model_names: Sequence[str],
    device: torch.device,
) -> Tuple[Dict[str, nn.Module], Dict[str, str]]:
    models: Dict[str, nn.Module] = {}
    hashes: Dict[str, str] = {}
    for name in model_names:
        checkpoint = Path(study_dir) / "checkpoints" / f"{name}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        payload = _torch_load(checkpoint)
        if payload.get("model_name") != name:
            raise ValueError(f"checkpoint identity mismatch for {name}")
        if payload.get("shortest_path_target") is not False:
            raise ValueError(f"checkpoint {name} lacks the no-shortest-path provenance flag")
        model = I.build_model(name, study_cfg)
        model.load_state_dict(payload["model"], strict=True)
        model.to(device).eval()
        models[name] = model
        hashes[name] = file_sha256(checkpoint)
    return models, hashes


def verify_audit_replay(
    study_dir: str | Path,
    bundles: Sequence[I.AuditBundle],
    tolerance: float = 1.0e-12,
) -> Dict[str, Any]:
    raw_path = Path(study_dir) / "results" / "target_reliability_raw.csv"
    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    replayed = [row for bundle in bundles for row in bundle.node_rows]
    if len(saved) != len(replayed):
        raise ValueError(f"audit replay row mismatch: {len(replayed)} vs {len(saved)}")

    max_abs_delta = 0.0
    mismatches = 0
    for expected, actual in zip(saved, replayed):
        if (
            int(expected["world_index"]) != int(actual["world_index"])
            or int(expected["world_seed"]) != int(actual["world_seed"])
            or int(expected["node"]) != int(actual["node"])
        ):
            mismatches += 1
            continue
        expected_value = expected.get("rollout_median", "")
        actual_value = actual.get("rollout_median", "")
        if expected_value == "" and actual_value == "":
            continue
        if expected_value == "" or actual_value == "":
            mismatches += 1
            continue
        delta = abs(float(expected_value) - float(actual_value))
        max_abs_delta = max(max_abs_delta, delta)
        if delta > float(tolerance):
            mismatches += 1
    if mismatches:
        raise ValueError(f"audit replay mismatches: {mismatches}")
    return {
        "rows": int(len(saved)),
        "mismatches": int(mismatches),
        "max_abs_rollout_median_delta": float(max_abs_delta),
        "source_sha256": file_sha256(raw_path),
    }


def build_provider_ranks(
    bundle: I.AuditBundle,
    providers: Sequence[str],
    models: Mapping[str, nn.Module],
    study_cfg: I.StudyConfig,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    euclid = C13.euclidean_to_goal(bundle.roadmap.points, bundle.roadmap.points[1])
    ranks: Dict[str, np.ndarray] = {}
    for provider in providers:
        if provider == "oracle_eval_only":
            ranks[provider] = C13._finite_oracle(bundle.roadmap, bundle.world.side_len)
        elif provider == "rollout_exact":
            ranks[provider] = np.asarray(bundle.rollout_rank, dtype=np.float64)
        elif provider in models:
            transformed = I.predict_model(models[provider], bundle.features, device)
            ranks[provider] = euclid + bundle.world.side_len * np.expm1(
                np.clip(transformed, 0.0, study_cfg.max_log_residual)
            )
        else:
            raise KeyError(f"unknown provider {provider!r}")
        if not np.all(np.isfinite(ranks[provider])):
            raise ValueError(f"provider {provider} produced nonfinite rank")
    return ranks


def evaluate_certified_search(
    runner_cfg: CertifiedSearchConfig,
    study_cfg: I.StudyConfig,
    bundles: Sequence[I.AuditBundle],
    models: Mapping[str, nn.Module],
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    providers = C13.parse_csv(runner_cfg.providers)
    focal_ws = C13.parse_float_csv(runner_cfg.focal_ws)
    rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    anchor_checks: List[Dict[str, Any]] = []

    for bundle in bundles:
        rm = bundle.roadmap
        euclid = C13.euclidean_to_goal(rm.points, rm.points[1])
        anchor_check = validate_consistent_anchor(rm.adj, euclid)
        anchor_checks.append(
            {
                "world_index": bundle.world_index,
                "world_seed": bundle.world_seed,
                **anchor_check,
            }
        )
        optimal = float(rm.dist_to_goal[0])  # evaluation-only
        euclid_astar = C.astar_search(rm.adj, euclid, len(rm.points))
        baseline_by_w: Dict[float, Dict[str, Any]] = {}
        for focal_w in focal_ws:
            baseline = I.focal_search_with_secondary(
                rm.adj,
                euclid,
                euclid,
                budget=len(rm.points),
                w=focal_w,
                secondary="h",
            )
            baseline_by_w[float(focal_w)] = baseline
            baseline_cost = float(baseline["cost"])
            baseline_rows.append(
                {
                    "suite": study_cfg.suite,
                    "world_index": bundle.world_index,
                    "world_seed": bundle.world_seed,
                    "focal_w": float(focal_w),
                    "euclid_focal_found": bool(baseline["found"]),
                    "euclid_focal_expansions": int(baseline["expansions"]),
                    "euclid_focal_cost": baseline_cost,
                    "euclid_focal_cost_ratio_eval_only": baseline_cost / optimal,
                    "euclid_focal_bound_violation_eval_only": bool(
                        baseline_cost > float(focal_w) * optimal + 1.0e-9
                    ),
                    "euclid_astar_expansions": int(euclid_astar["expansions"]),
                    "euclid_astar_cost": float(euclid_astar["cost"]),
                }
            )

        ranks = build_provider_ranks(bundle, providers, models, study_cfg, device)
        phase1_budget = max(0, int(math.ceil(runner_cfg.phase1_budget_factor * len(rm.points))))
        anchor_budget = max(0, int(math.ceil(runner_cfg.anchor_budget_factor * len(rm.points))))
        for provider in providers:
            phase1_started = time.perf_counter()
            phase1 = inadmissible_astar_incumbent(
                rm.adj,
                ranks[provider],
                budget=phase1_budget,
            )
            phase1_seconds = time.perf_counter() - phase1_started
            phase1_cost = float(phase1["cost"])
            for focal_w in focal_ws:
                certify_started = time.perf_counter()
                certificate = certify_incumbent_with_anchor(
                    rm.adj,
                    euclid,
                    phase1_cost,
                    w=focal_w,
                    budget=anchor_budget,
                    validate_anchor=False,
                )
                certification_seconds = time.perf_counter() - certify_started
                final_cost = float(certificate["final_cost"])
                baseline = baseline_by_w[float(focal_w)]
                total_expansions = int(phase1["expansions"]) + int(certificate["expansions"])
                rows.append(
                    {
                        "suite": study_cfg.suite,
                        "world_index": bundle.world_index,
                        "world_seed": bundle.world_seed,
                        "provider": provider,
                        "focal_w": float(focal_w),
                        "phase1_found": bool(phase1["found"]),
                        "phase1_cost": phase1_cost if math.isfinite(phase1_cost) else "",
                        "phase1_cost_ratio_eval_only": (
                            phase1_cost / optimal if math.isfinite(phase1_cost) else ""
                        ),
                        "phase1_expansions": int(phase1["expansions"]),
                        "phase1_reopens": int(phase1["reopens"]),
                        "phase1_generated": int(phase1["generated"]),
                        "phase1_seconds": float(phase1_seconds),
                        "certified": bool(certificate["certified"]),
                        "certificate_proof": certificate["proof"],
                        "certificate_lower_bound": float(certificate["lower_bound"]),
                        "certificate_ratio": float(certificate["certificate_ratio"]),
                        "certificate_expansions": int(certificate["expansions"]),
                        "certificate_generated": int(certificate["generated"]),
                        "certificate_seconds": float(certification_seconds),
                        "anchor_goal_popped": bool(certificate["anchor_goal_popped"]),
                        "anchor_improved_incumbent": bool(
                            certificate["anchor_improved_incumbent"]
                        ),
                        "total_expansions": int(total_expansions),
                        "final_cost": final_cost if math.isfinite(final_cost) else "",
                        "final_cost_ratio_eval_only": (
                            final_cost / optimal if math.isfinite(final_cost) else ""
                        ),
                        "bound_violation_eval_only": bool(
                            not math.isfinite(final_cost)
                            or final_cost > float(focal_w) * optimal + 1.0e-9
                        ),
                        "euclid_focal_expansions": int(baseline["expansions"]),
                        "euclid_focal_cost_ratio_eval_only": float(baseline["cost"]) / optimal,
                        "delta_total_vs_euclid_focal": int(total_expansions)
                        - int(baseline["expansions"]),
                        "euclid_astar_expansions": int(euclid_astar["expansions"]),
                        "delta_total_vs_euclid_astar": int(total_expansions)
                        - int(euclid_astar["expansions"]),
                    }
                )
    return rows, baseline_rows, anchor_checks


def summarize_certified_rows(
    rows: Sequence[Mapping[str, Any]],
    required_win_fraction: float,
) -> List[Dict[str, Any]]:
    grouped: DefaultDict[Tuple[str, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["provider"]), float(row["focal_w"]))].append(row)

    summaries: List[Dict[str, Any]] = []
    for (provider, focal_w), group in sorted(grouped.items(), key=lambda item: str(item[0])):
        deltas = np.asarray(
            [float(row["delta_total_vs_euclid_focal"]) for row in group],
            dtype=np.float64,
        )
        required_wins = int(math.ceil(float(required_win_fraction) * len(group)))
        wins = int(np.sum(deltas < 0.0))
        ties = int(np.sum(deltas == 0.0))
        losses = int(np.sum(deltas > 0.0))
        certification_rate = float(np.mean([bool(row["certified"]) for row in group]))
        bound_violations = int(np.sum([bool(row["bound_violation_eval_only"]) for row in group]))
        mean_delta = float(np.mean(deltas))
        summaries.append(
            {
                "provider": provider,
                "focal_w": float(focal_w),
                "worlds": int(len(group)),
                "required_wins": int(required_wins),
                "gate_pass": bool(
                    certification_rate == 1.0
                    and bound_violations == 0
                    and wins >= required_wins
                    and mean_delta < 0.0
                ),
                "certification_rate": certification_rate,
                "bound_violations_eval_only": bound_violations,
                "phase1_expansions_mean": float(
                    np.mean([float(row["phase1_expansions"]) for row in group])
                ),
                "phase1_reopens_mean": float(
                    np.mean([float(row["phase1_reopens"]) for row in group])
                ),
                "phase1_cost_ratio_mean_eval_only": float(
                    np.mean([float(row["phase1_cost_ratio_eval_only"]) for row in group])
                ),
                "certificate_expansions_mean": float(
                    np.mean([float(row["certificate_expansions"]) for row in group])
                ),
                "total_expansions_mean": float(
                    np.mean([float(row["total_expansions"]) for row in group])
                ),
                "euclid_focal_expansions_mean": float(
                    np.mean([float(row["euclid_focal_expansions"]) for row in group])
                ),
                "delta_total_vs_euclid_focal_mean": mean_delta,
                "delta_total_vs_euclid_focal_median": float(np.median(deltas)),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "final_cost_ratio_mean_eval_only": float(
                    np.mean([float(row["final_cost_ratio_eval_only"]) for row in group])
                ),
                "final_cost_ratio_max_eval_only": float(
                    np.max([float(row["final_cost_ratio_eval_only"]) for row in group])
                ),
                "anchor_goal_pop_rate": float(
                    np.mean([bool(row["anchor_goal_popped"]) for row in group])
                ),
                "anchor_improvement_rate": float(
                    np.mean([bool(row["anchor_improved_incumbent"]) for row in group])
                ),
            }
        )
    return summaries


def build_gate_verdict(
    summaries: Sequence[Mapping[str, Any]],
    primary_w: float,
) -> Dict[str, Any]:
    by_provider = {
        str(row["provider"]): dict(row)
        for row in summaries
        if math.isclose(float(row["focal_w"]), float(primary_w), abs_tol=1.0e-12)
    }
    if "oracle_eval_only" not in by_provider or "rollout_exact" not in by_provider:
        raise ValueError("primary gate requires oracle_eval_only and rollout_exact")
    oracle_pass = bool(by_provider["oracle_eval_only"]["gate_pass"])
    exact_pass = bool(by_provider["rollout_exact"]["gate_pass"])
    learned = {
        provider: bool(row["gate_pass"])
        for provider, row in by_provider.items()
        if provider not in {"oracle_eval_only", "rollout_exact"}
    }
    if not oracle_pass:
        verdict = "reject_simple_certifier_no_oracle_headroom_at_primary_bound"
    elif not exact_pass:
        verdict = "reject_current_rollout_target_under_simple_certifier"
    elif any(learned.values()):
        verdict = "provisional_learned_certified_gate_pass"
    else:
        verdict = "exact_target_passes_but_learned_representation_requires_repair"
    return {
        "primary_w": float(primary_w),
        "oracle_gate_pass": oracle_pass,
        "exact_rollout_gate_pass": exact_pass,
        "learned_gate_pass": learned,
        "verdict": verdict,
        "gate_definition": {
            "certification_rate": 1.0,
            "bound_violations": 0,
            "required_win_fraction": "configured; ceil(fraction * worlds)",
            "mean_delta_vs_matched_euclid_focal": "strictly_negative",
            "expansion_accounting": "phase1_plus_fresh_anchor_including_duplicate_work",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-C certified incumbent diagnostic")
    parser.add_argument("--study-dir", default=CertifiedSearchConfig.study_dir)
    parser.add_argument("--out-dir", default=CertifiedSearchConfig.out_dir)
    parser.add_argument("--providers", default=CertifiedSearchConfig.providers)
    parser.add_argument("--focal-ws", default=CertifiedSearchConfig.focal_ws)
    parser.add_argument("--primary-w", type=float, default=CertifiedSearchConfig.primary_w)
    parser.add_argument(
        "--phase1-budget-factor",
        type=float,
        default=CertifiedSearchConfig.phase1_budget_factor,
    )
    parser.add_argument(
        "--anchor-budget-factor",
        type=float,
        default=CertifiedSearchConfig.anchor_budget_factor,
    )
    parser.add_argument(
        "--required-win-fraction",
        type=float,
        default=CertifiedSearchConfig.required_win_fraction,
    )
    parser.add_argument("--device", default=CertifiedSearchConfig.device)
    return parser.parse_args()


def main() -> None:
    runner_cfg = CertifiedSearchConfig(**vars(parse_args()))
    script_dir = Path(__file__).resolve().parent
    if runner_cfg.study_dir == CertifiedSearchConfig.study_dir:
        script_relative_study = script_dir / runner_cfg.study_dir
        if not Path(runner_cfg.study_dir).exists() and script_relative_study.exists():
            runner_cfg.study_dir = str(script_relative_study)
            if runner_cfg.out_dir == CertifiedSearchConfig.out_dir:
                runner_cfg.out_dir = str(script_dir / runner_cfg.out_dir)
    if not (0.0 < runner_cfg.required_win_fraction <= 1.0):
        raise ValueError("required-win-fraction must be in (0, 1]")
    focal_ws = C13.parse_float_csv(runner_cfg.focal_ws)
    if not any(
        math.isclose(float(value), float(runner_cfg.primary_w), abs_tol=1.0e-12)
        for value in focal_ws
    ):
        raise ValueError("primary-w must appear in focal-ws")

    study_cfg, source_manifest = load_study(runner_cfg.study_dir)
    device = resolve_device(runner_cfg.device)
    providers = C13.parse_csv(runner_cfg.providers)
    model_names = [
        provider
        for provider in providers
        if provider not in {"oracle_eval_only", "rollout_exact"}
    ]
    models, checkpoint_hashes = load_frozen_models(
        runner_cfg.study_dir,
        study_cfg,
        model_names,
        device,
    )
    bundles = I.collect_audit_bundles(study_cfg)
    replay = verify_audit_replay(runner_cfg.study_dir, bundles)

    rows, baseline_rows, anchor_checks = evaluate_certified_search(
        runner_cfg,
        study_cfg,
        bundles,
        models,
        device,
    )
    summaries = summarize_certified_rows(rows, runner_cfg.required_win_fraction)
    gate = build_gate_verdict(summaries, runner_cfg.primary_w)

    result_dir = C13.ensure_dir(Path(runner_cfg.out_dir) / "results")
    raw_path = C13.write_csv(result_dir / "certified_search_raw.csv", rows)
    baseline_path = C13.write_csv(result_dir / "certified_search_baselines.csv", baseline_rows)
    summary_path = C13.write_csv(result_dir / "certified_search_summary.csv", summaries)
    gate_path = C13.write_json(result_dir / "gate_verdict.json", gate)

    keys = [
        (int(row["world_index"]), str(row["provider"]), float(row["focal_w"]))
        for row in rows
    ]
    verification = {
        "audit_replay": replay,
        "worlds": int(len(bundles)),
        "providers": providers,
        "focal_ws": focal_ws,
        "raw_rows": int(len(rows)),
        "expected_raw_rows": int(len(bundles) * len(providers) * len(focal_ws)),
        "duplicate_keys": int(len(keys) - len(set(keys))),
        "certification_failures": int(np.sum([not bool(row["certified"]) for row in rows])),
        "phase1_failures": int(np.sum([not bool(row["phase1_found"]) for row in rows])),
        "bound_violations_eval_only": int(
            np.sum([bool(row["bound_violation_eval_only"]) for row in rows])
        ),
        "anchor_checks": anchor_checks,
        "maximum_anchor_consistency_violation": float(
            max(float(row["max_consistency_violation"]) for row in anchor_checks)
        ),
        "shortest_path_training_target": False,
        "shortest_path_role_learned_and_rollout_arms": "posthoc_bound_validation_only",
        "shortest_path_oracle_control": "privileged_phase1_integration_ceiling",
    }
    if verification["raw_rows"] != verification["expected_raw_rows"]:
        raise RuntimeError("certified-search output row count mismatch")
    if verification["duplicate_keys"]:
        raise RuntimeError("duplicate certified-search output keys")
    if (
        verification["phase1_failures"]
        or verification["certification_failures"]
        or verification["bound_violations_eval_only"]
    ):
        raise RuntimeError("certification invariant failed")
    verification_path = C13.write_json(result_dir / "verification.json", verification)

    source_paths = {
        "source_study_manifest": Path(runner_cfg.study_dir) / "manifest.json",
        "implementation": Path(__file__).resolve(),
        "source_target_audit": Path(runner_cfg.study_dir)
        / "results"
        / "target_reliability_raw.csv",
        **{
            f"checkpoint_{name}": Path(runner_cfg.study_dir) / "checkpoints" / f"{name}.pt"
            for name in model_names
        },
    }
    output_paths = {
        "raw": raw_path,
        "baselines": baseline_path,
        "summary": summary_path,
        "gate": gate_path,
        "verification": verification_path,
    }
    integrity = {
        "inputs": {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in source_paths.items()},
        "outputs": {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in output_paths.items()},
        "checkpoint_hashes": checkpoint_hashes,
    }
    integrity_path = C13.write_json(Path(runner_cfg.out_dir) / "integrity.json", integrity)
    manifest = {
        "experiment": "C13-C certified incumbent integration diagnostic",
        "runner_config": asdict(runner_cfg),
        "source_study_config": asdict(study_cfg),
        "source_study_experiment": source_manifest.get("experiment"),
        "training_performed": False,
        "shortest_path_target": False,
        "certification_anchor": "euclidean_consistent_admissible",
        "oracle_control": "privileged_shortest_path_phase1_integration_ceiling",
        "learned_and_rollout_shortest_path_role": "posthoc_bound_validation_only",
        "phase1": "arbitrary_finite_rank_astar_with_reopening_first_goal",
        "phase2": "fresh_euclidean_astar_until_incumbent_le_w_times_open_lower_bound",
        "expansion_accounting": "phase1_plus_phase2_including_duplicate_work",
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "integrity": str(integrity_path),
    }
    manifest_path = C13.write_json(Path(runner_cfg.out_dir) / "manifest.json", manifest)

    print(f"verdict={gate['verdict']}")
    for name, path in {**output_paths, "integrity": integrity_path, "manifest": manifest_path}.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
