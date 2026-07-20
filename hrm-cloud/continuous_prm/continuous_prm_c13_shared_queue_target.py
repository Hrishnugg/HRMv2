#!/usr/bin/env python3
"""C13-E: exact rollout target in the frozen shared-state certified search.

This causal gate changes exactly one component from C13-D: the privileged
graph-distance rank is replaced by C13-B's replayed ``rollout_exact`` vector.
The shared-g search, Euclidean proof anchor, widths, audit worlds, baselines,
budget, certificate, and five-of-six gate remain fixed.

``rollout_exact`` means the successful fresh-start behavior-return median from
ten deterministic rollouts at each node.  Nodes with no successful rollout use
the same deterministic Euclidean-plus-penalty fill constructed by C13-B.  No
training or model loading occurs here.  Graph shortest-path distance is used
only for post-hoc cost/target diagnostics and never as a rank, feature, label,
or certificate input.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import continuous_prm_common as C
import continuous_prm_c13_certified_search as S
import continuous_prm_c13_identifiability as I
import continuous_prm_c13_shared_queue as Q
import continuous_prm_c13_state_heuristic as C13


@dataclass
class ExactTargetConfig:
    study_dir: str = "runs/c13_identifiability"
    independent_dir: str = "runs/c13_certified_search"
    oracle_dir: str = "runs/c13_shared_queue_oracle"
    out_dir: str = "runs/c13_shared_queue_rollout"
    focal_ws: str = "1.05,1.10,1.25"
    primary_w: float = 1.10
    budget_factor: float = 2.0
    required_win_fraction: float = 0.80


def load_reference_rows(
    path: str | Path,
    provider: str,
) -> Dict[Tuple[int, float], Dict[str, str]]:
    """Load one provider from a frozen reference CSV and reject duplicates."""

    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        selected = [
            row for row in csv.DictReader(handle) if row.get("provider") == provider
        ]
    indexed: Dict[Tuple[int, float], Dict[str, str]] = {}
    for row in selected:
        key = (int(row["world_index"]), float(row["focal_w"]))
        if key in indexed:
            raise ValueError(f"duplicate {provider} reference row {key} in {source}")
        indexed[key] = row
    if not indexed:
        raise ValueError(f"provider {provider!r} is absent from {source}")
    return indexed


def _optional_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def target_diagnostics(bundle: I.AuditBundle) -> Dict[str, Any]:
    """Describe target coverage, scale, and ordering without changing the rank."""

    rollout = np.asarray(bundle.rollout_rank, dtype=np.float64)
    if rollout.shape != (len(bundle.roadmap.points),) or not np.all(
        np.isfinite(rollout)
    ):
        raise ValueError("replayed rollout_exact rank must be finite and node-aligned")

    raw = np.asarray(
        [_optional_float(row.get("rollout_median")) for row in bundle.node_rows],
        dtype=np.float64,
    )
    if raw.shape != rollout.shape:
        raise ValueError("target audit rows do not align with the replayed roadmap")

    oracle = np.asarray(bundle.roadmap.dist_to_goal, dtype=np.float64)
    connected = np.isfinite(oracle) & (oracle < C.INF / 10.0)
    labeled = np.isfinite(raw)
    comparable = connected & (oracle > C.EPS)
    euclid = C13.euclidean_to_goal(
        bundle.roadmap.points,
        bundle.roadmap.points[1],
    )
    missing = ~labeled
    fill_residual = rollout[missing] - euclid[missing]

    return {
        "suite": str(bundle.node_rows[0]["suite"]),
        "world_index": int(bundle.world_index),
        "world_seed": int(bundle.world_seed),
        "nodes": int(len(rollout)),
        "rollout_labeled_nodes": int(np.sum(labeled)),
        "rollout_unlabeled_nodes": int(np.sum(missing)),
        "rollout_label_rate": float(np.mean(labeled)),
        "rollout_label_rate_connected_eval_only": float(
            np.mean(labeled[connected])
        ),
        "penalty_fill_residual": (
            float(np.median(fill_residual)) if len(fill_residual) else 0.0
        ),
        "rollout_rank_vs_oracle_spearman_eval_only": I.safe_spearman(
            rollout[connected], oracle[connected]
        ),
        "rollout_rank_to_oracle_ratio_median_eval_only": float(
            np.median(rollout[comparable] / oracle[comparable])
        ),
        "rollout_rank_to_oracle_ratio_start_eval_only": float(
            rollout[0] / oracle[0]
        ),
        "rollout_rank_mae_over_side_len_eval_only": float(
            np.mean(np.abs(rollout[connected] - oracle[connected]))
            / float(bundle.world.side_len)
        ),
    }


def evaluate_exact_target_gate(
    cfg: ExactTargetConfig,
    study_cfg: I.StudyConfig,
    bundles: Sequence[I.AuditBundle],
    independent_rows: Mapping[Tuple[int, float], Mapping[str, str]],
    oracle_rows: Mapping[Tuple[int, float], Mapping[str, str]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Evaluate only the frozen exact rollout rank in the shared search."""

    focal_ws = C13.parse_float_csv(cfg.focal_ws)
    rows: List[Dict[str, Any]] = []
    baselines: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    anchor_checks: List[Dict[str, Any]] = []

    expected_keys = {
        (int(bundle.world_index), float(focal_w))
        for bundle in bundles
        for focal_w in focal_ws
    }
    for label, reference in (
        ("independent rollout_exact", independent_rows),
        ("shared oracle", oracle_rows),
    ):
        missing = sorted(expected_keys - set(reference))
        if missing:
            raise ValueError(f"{label} reference is missing keys: {missing}")

    for bundle in bundles:
        rm = bundle.roadmap
        euclid = C13.euclidean_to_goal(rm.points, rm.points[1])
        rollout_exact = np.asarray(bundle.rollout_rank, dtype=np.float64)
        diagnostic = target_diagnostics(bundle)
        diagnostics.append(diagnostic)
        anchor_check = S.validate_consistent_anchor(rm.adj, euclid)
        anchor_checks.append(
            {
                "world_index": int(bundle.world_index),
                "world_seed": int(bundle.world_seed),
                **anchor_check,
            }
        )
        optimal = float(rm.dist_to_goal[0])  # evaluation-only
        euclid_astar = C.astar_search(rm.adj, euclid, len(rm.points))
        budget = max(0, int(math.ceil(float(cfg.budget_factor) * len(rm.points))))

        for focal_w in focal_ws:
            key = (int(bundle.world_index), float(focal_w))
            independent = independent_rows[key]
            oracle = oracle_rows[key]
            if int(independent["world_seed"]) != int(bundle.world_seed):
                raise ValueError(f"independent reference world mismatch at {key}")
            if int(oracle["world_seed"]) != int(bundle.world_seed):
                raise ValueError(f"oracle reference world mismatch at {key}")

            baseline = I.focal_search_with_secondary(
                rm.adj,
                euclid,
                euclid,
                budget=len(rm.points),
                w=float(focal_w),
                secondary="h",
            )
            baseline_cost = float(baseline["cost"])
            baselines.append(
                {
                    "suite": study_cfg.suite,
                    "world_index": int(bundle.world_index),
                    "world_seed": int(bundle.world_seed),
                    "focal_w": float(focal_w),
                    "euclid_focal_found": bool(baseline["found"]),
                    "euclid_focal_expansions": int(baseline["expansions"]),
                    "euclid_focal_cost": baseline_cost,
                    "euclid_focal_cost_ratio_eval_only": baseline_cost / optimal,
                    "euclid_astar_expansions": int(euclid_astar["expansions"]),
                    "euclid_astar_cost": float(euclid_astar["cost"]),
                }
            )

            result = Q.shared_anchor_certified_search(
                rm.adj,
                euclid,
                rollout_exact,
                w=float(focal_w),
                budget=budget,
                validate_anchor=False,
            )
            final_cost = float(result["final_cost"])
            path_check = Q.validate_path(rm.adj, result["path"], final_cost)
            anchor_lb = float(result["lower_bound"])
            accounting_valid = bool(
                int(result["expansions"])
                == int(result["rank_expansions"])
                + int(result["anchor_expansions"])
            )
            rows.append(
                {
                    "suite": study_cfg.suite,
                    "world_index": int(bundle.world_index),
                    "world_seed": int(bundle.world_seed),
                    "provider": "rollout_exact",
                    "rank_source": "c13b_replayed_rollout_exact",
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
                    "expansion_accounting_valid": accounting_valid,
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
                    "rollout_label_rate": float(diagnostic["rollout_label_rate"]),
                    "rollout_rank_vs_oracle_spearman_eval_only": float(
                        diagnostic["rollout_rank_vs_oracle_spearman_eval_only"]
                    ),
                    "rollout_rank_to_oracle_ratio_start_eval_only": float(
                        diagnostic["rollout_rank_to_oracle_ratio_start_eval_only"]
                    ),
                    "euclid_focal_expansions": int(baseline["expansions"]),
                    "delta_vs_euclid_focal": int(result["expansions"])
                    - int(baseline["expansions"]),
                    "euclid_astar_expansions": int(euclid_astar["expansions"]),
                    "delta_vs_euclid_astar": int(result["expansions"])
                    - int(euclid_astar["expansions"]),
                    "shared_oracle_expansions": int(oracle["expansions"]),
                    "delta_vs_shared_oracle": int(result["expansions"])
                    - int(oracle["expansions"]),
                    "independent_exact_total_expansions": int(
                        independent["total_expansions"]
                    ),
                    "saved_vs_independent_exact": int(
                        independent["total_expansions"]
                    )
                    - int(result["expansions"]),
                }
            )
    return rows, baselines, diagnostics, anchor_checks


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    required_win_fraction: float,
) -> List[Dict[str, Any]]:
    """Summarize each width and apply the locked five-of-six target gate."""

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
        accounting_failures = int(
            np.sum(
                [not bool(row["expansion_accounting_valid"]) for row in group]
            )
        )
        mean_delta = float(np.mean(deltas))
        total_checks = int(
            np.sum([int(row["rank_eligibility_checks"]) for row in group])
        )
        total_choices = int(
            np.sum([int(row["rank_eligible_choices"]) for row in group])
        )
        summaries.append(
            {
                "provider": "rollout_exact",
                "focal_w": float(focal_w),
                "worlds": int(len(group)),
                "required_wins": int(required_wins),
                "gate_pass": bool(
                    certification_rate == 1.0
                    and bound_violations == 0
                    and path_failures == 0
                    and lower_bound_failures == 0
                    and accounting_failures == 0
                    and wins >= required_wins
                    and mean_delta < 0.0
                ),
                "certification_rate": certification_rate,
                "bound_violations_eval_only": bound_violations,
                "path_failures": path_failures,
                "anchor_lower_bound_failures_eval_only": lower_bound_failures,
                "expansion_accounting_failures": accounting_failures,
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
                "rank_eligible_choice_rate": float(
                    total_choices / max(1, total_checks)
                ),
                "rollout_label_rate_mean": float(
                    np.mean([float(row["rollout_label_rate"]) for row in group])
                ),
                "rollout_rank_vs_oracle_spearman_mean_eval_only": float(
                    np.mean(
                        [
                            float(row["rollout_rank_vs_oracle_spearman_eval_only"])
                            for row in group
                        ]
                    )
                ),
                "rollout_rank_to_oracle_ratio_start_mean_eval_only": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "rollout_rank_to_oracle_ratio_start_eval_only"
                                ]
                            )
                            for row in group
                        ]
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
                "shared_oracle_expansions_mean": float(
                    np.mean([float(row["shared_oracle_expansions"]) for row in group])
                ),
                "delta_vs_shared_oracle_mean": float(
                    np.mean([float(row["delta_vs_shared_oracle"]) for row in group])
                ),
                "independent_exact_total_expansions_mean": float(
                    np.mean(
                        [
                            float(row["independent_exact_total_expansions"])
                            for row in group
                        ]
                    )
                ),
                "saved_vs_independent_exact_mean": float(
                    np.mean(
                        [float(row["saved_vs_independent_exact"]) for row in group]
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
        raise ValueError("primary exact-target summary is missing or duplicated")
    primary = dict(matches[0])
    passed = bool(primary["gate_pass"])
    return {
        "primary_w": float(primary_w),
        "exact_rollout_gate_pass": passed,
        "verdict": (
            "shared_queue_exact_rollout_gate_pass"
            if passed
            else "shared_queue_exact_rollout_gate_fail"
        ),
        "primary_summary": primary,
        "authorization": (
            "run_frozen_learned_providers_next"
            if passed
            else "repair_target_alignment_or_calibration_before_models"
        ),
        "gate_definition": {
            "certification_rate": 1.0,
            "bound_violations": 0,
            "path_failures": 0,
            "anchor_lower_bound_failures": 0,
            "expansion_accounting_failures": 0,
            "required_win_fraction": "configured; ceil(fraction * worlds)",
            "mean_delta_vs_matched_euclid_focal": "strictly_negative",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-E shared exact-target gate")
    parser.add_argument("--study-dir", default=ExactTargetConfig.study_dir)
    parser.add_argument("--independent-dir", default=ExactTargetConfig.independent_dir)
    parser.add_argument("--oracle-dir", default=ExactTargetConfig.oracle_dir)
    parser.add_argument("--out-dir", default=ExactTargetConfig.out_dir)
    parser.add_argument("--focal-ws", default=ExactTargetConfig.focal_ws)
    parser.add_argument("--primary-w", type=float, default=ExactTargetConfig.primary_w)
    parser.add_argument(
        "--budget-factor", type=float, default=ExactTargetConfig.budget_factor
    )
    parser.add_argument(
        "--required-win-fraction",
        type=float,
        default=ExactTargetConfig.required_win_fraction,
    )
    return parser.parse_args()


def _resolve_default_paths(cfg: ExactTargetConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    study_was_resolved = False
    for field, default in (
        ("study_dir", ExactTargetConfig.study_dir),
        ("independent_dir", ExactTargetConfig.independent_dir),
        ("oracle_dir", ExactTargetConfig.oracle_dir),
    ):
        value = getattr(cfg, field)
        if value != default:
            continue
        candidate = script_dir / value
        if not Path(value).exists() and candidate.exists():
            setattr(cfg, field, str(candidate))
            if field == "study_dir":
                study_was_resolved = True
    if cfg.out_dir == ExactTargetConfig.out_dir and study_was_resolved:
        cfg.out_dir = str(script_dir / cfg.out_dir)


def main() -> None:
    cfg = ExactTargetConfig(**vars(parse_args()))
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
    independent_raw = (
        Path(cfg.independent_dir) / "results" / "certified_search_raw.csv"
    )
    oracle_raw = (
        Path(cfg.oracle_dir) / "results" / "shared_queue_oracle_raw.csv"
    )
    independent_rows = load_reference_rows(independent_raw, "rollout_exact")
    oracle_rows = load_reference_rows(oracle_raw, "oracle_eval_only")
    rows, baselines, diagnostics, anchor_checks = evaluate_exact_target_gate(
        cfg,
        study_cfg,
        bundles,
        independent_rows,
        oracle_rows,
    )
    summaries = summarize_rows(rows, cfg.required_win_fraction)
    gate = build_gate_verdict(summaries, cfg.primary_w)

    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    raw_path = C13.write_csv(result_dir / "shared_queue_exact_target_raw.csv", rows)
    baseline_path = C13.write_csv(
        result_dir / "shared_queue_exact_target_baselines.csv", baselines
    )
    diagnostic_path = C13.write_csv(
        result_dir / "shared_queue_exact_target_diagnostics.csv", diagnostics
    )
    summary_path = C13.write_csv(
        result_dir / "shared_queue_exact_target_summary.csv", summaries
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
        "reference_independent_exact_rows": int(len(independent_rows)),
        "reference_shared_oracle_rows": int(len(oracle_rows)),
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
        "rank_reconstruction": "identical_bundle.rollout_rank_used_by_c13b",
        "training_performed": False,
        "model_loading_performed": False,
        "shortest_path_target": False,
        "shortest_path_use": "posthoc_target_and_cost_diagnostics_only",
        "proof_anchor": "euclidean_consistent_admissible",
    }
    if verification["raw_rows"] != verification["expected_raw_rows"]:
        raise RuntimeError("exact-target output row count mismatch")
    structural_failures = (
        verification["duplicate_keys"]
        + verification["expansion_accounting_failures"]
        + verification["states_expanded_more_than_twice"]
    )
    if structural_failures:
        raise RuntimeError("exact-target structural verification invariant failed")
    verification_path = C13.write_json(result_dir / "verification.json", verification)

    source_paths = {
        "implementation": Path(__file__).resolve(),
        "shared_queue_implementation": Path(Q.__file__).resolve(),
        "source_study_manifest": Path(cfg.study_dir) / "manifest.json",
        "source_target_audit": Path(cfg.study_dir)
        / "results"
        / "target_reliability_raw.csv",
        "source_independent_raw": independent_raw,
        "source_shared_oracle_raw": oracle_raw,
    }
    output_paths = {
        "raw": raw_path,
        "baselines": baseline_path,
        "diagnostics": diagnostic_path,
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
        "experiment": "C13-E shared-state exact rollout target gate",
        "runner_config": asdict(cfg),
        "source_study_config": asdict(study_cfg),
        "source_study_experiment": source_manifest.get("experiment"),
        "training_performed": False,
        "model_loading_performed": False,
        "shortest_path_target": False,
        "rank_provider": "c13b_replayed_rollout_exact",
        "rank_definition": (
            "successful_fresh_start_behavior_return_median_of_10_with_"
            "c13b_deterministic_penalty_fill"
        ),
        "shortest_path_use": "posthoc_target_and_cost_diagnostics_only",
        "proof_anchor": "euclidean_consistent_admissible",
        "search": "c13d_one_anchor_one_rank_shared_g_unchanged",
        "termination": "incumbent_le_w_times_shared_anchor_open_lower_bound",
        "expansion_accounting": "all_queue_expansions_including_cross_queue_duplicates",
        "causal_change_from_c13d": "rank_only_oracle_to_rollout_exact",
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
