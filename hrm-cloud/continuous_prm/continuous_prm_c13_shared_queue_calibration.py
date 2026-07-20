#!/usr/bin/env python3
"""C13-F: training-free rollout scale-versus-ordering diagnostic.

The six C13 audit worlds are reused only as a development cohort.  The shared
search and Euclidean proof anchor are frozen.  We compare a same-search
Euclidean rank with fixed monotone blends

    Euclidean + alpha * (rollout_exact - Euclidean)

so the rollout ordering is retained while its inflated behavior-cost scale is
attenuated.  No model is trained or loaded, and no shortest-path result enters
the rank.  Selection on this reused cohort is explicitly development-only; a
chosen coefficient must be replicated on fresh worlds before it can authorize
training or a scientific claim.
"""
from __future__ import annotations

import argparse
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
import continuous_prm_c13_shared_queue_target as T
import continuous_prm_c13_state_heuristic as C13


@dataclass
class CalibrationConfig:
    study_dir: str = "runs/c13_identifiability"
    exact_dir: str = "runs/c13_shared_queue_rollout"
    oracle_dir: str = "runs/c13_shared_queue_oracle"
    out_dir: str = "runs/c13_shared_queue_calibration"
    focal_ws: str = "1.05,1.10,1.25"
    primary_w: float = 1.10
    alphas: str = "0.00,0.05,0.10,0.25,0.50,1.00"
    budget_factor: float = 2.0
    required_focal_win_fraction: float = 0.80
    required_euclid_rank_wins: int = 4


def provider_name(alpha: float) -> str:
    if math.isclose(float(alpha), 0.0, abs_tol=1.0e-12):
        return "euclid_rank"
    if math.isclose(float(alpha), 1.0, abs_tol=1.0e-12):
        return "rollout_exact"
    return f"rollout_blend_a{float(alpha):.2f}".replace(".", "p")


def calibrated_rank(
    euclid: np.ndarray,
    rollout_exact: np.ndarray,
    alpha: float,
    tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Attenuate only rollout residual magnitude while preserving ordering."""

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    baseline = np.asarray(euclid, dtype=np.float64).reshape(-1)
    rollout = np.asarray(rollout_exact, dtype=np.float64).reshape(-1)
    if baseline.shape != rollout.shape:
        raise ValueError("euclid and rollout ranks must have matching shapes")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(rollout)):
        raise ValueError("calibration inputs must be finite")
    residual = rollout - baseline
    if float(np.min(residual)) < -float(tolerance):
        raise ValueError("rollout_exact must not undercut Euclidean")
    residual = np.maximum(residual, 0.0)
    return baseline + float(alpha) * residual


def evaluate_calibration(
    cfg: CalibrationConfig,
    study_cfg: I.StudyConfig,
    bundles: Sequence[I.AuditBundle],
    exact_rows: Mapping[Tuple[int, float], Mapping[str, str]],
    oracle_rows: Mapping[Tuple[int, float], Mapping[str, str]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    focal_ws = C13.parse_float_csv(cfg.focal_ws)
    alphas = C13.parse_float_csv(cfg.alphas)
    rows: List[Dict[str, Any]] = []
    baselines: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    anchor_checks: List[Dict[str, Any]] = []

    expected_reference_keys = {
        (int(bundle.world_index), float(focal_w))
        for bundle in bundles
        for focal_w in focal_ws
    }
    for label, reference in (("exact", exact_rows), ("oracle", oracle_rows)):
        missing = sorted(expected_reference_keys - set(reference))
        if missing:
            raise ValueError(f"{label} reference is missing keys: {missing}")

    for bundle in bundles:
        rm = bundle.roadmap
        euclid = C13.euclidean_to_goal(rm.points, rm.points[1])
        diagnostic = T.target_diagnostics(bundle)
        diagnostics.append(diagnostic)
        anchor_checks.append(
            {
                "world_index": int(bundle.world_index),
                "world_seed": int(bundle.world_seed),
                **S.validate_consistent_anchor(rm.adj, euclid),
            }
        )
        optimal = float(rm.dist_to_goal[0])  # evaluation-only
        euclid_astar = C.astar_search(rm.adj, euclid, len(rm.points))
        budget = max(0, int(math.ceil(float(cfg.budget_factor) * len(rm.points))))

        baseline_by_w: Dict[float, Dict[str, Any]] = {}
        for focal_w in focal_ws:
            baseline = I.focal_search_with_secondary(
                rm.adj,
                euclid,
                euclid,
                budget=len(rm.points),
                w=float(focal_w),
                secondary="h",
            )
            baseline_by_w[float(focal_w)] = baseline
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

        for alpha in alphas:
            rank = calibrated_rank(euclid, bundle.rollout_rank, alpha)
            provider = provider_name(alpha)
            for focal_w in focal_ws:
                key = (int(bundle.world_index), float(focal_w))
                exact_reference = exact_rows[key]
                oracle_reference = oracle_rows[key]
                if int(exact_reference["world_seed"]) != int(bundle.world_seed):
                    raise ValueError(f"exact reference world mismatch at {key}")
                if int(oracle_reference["world_seed"]) != int(bundle.world_seed):
                    raise ValueError(f"oracle reference world mismatch at {key}")
                baseline = baseline_by_w[float(focal_w)]
                result = Q.shared_anchor_certified_search(
                    rm.adj,
                    euclid,
                    rank,
                    w=float(focal_w),
                    budget=budget,
                    validate_anchor=False,
                )
                final_cost = float(result["final_cost"])
                path_check = Q.validate_path(rm.adj, result["path"], final_cost)
                anchor_lb = float(result["lower_bound"])
                rows.append(
                    {
                        "suite": study_cfg.suite,
                        "world_index": int(bundle.world_index),
                        "world_seed": int(bundle.world_seed),
                        "provider": provider,
                        "alpha": float(alpha),
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
                        "rank_eligible_choices": int(
                            result["rank_eligible_choices"]
                        ),
                        "rank_eligible_choice_rate": float(
                            result["rank_eligible_choices"]
                            / max(1, result["rank_eligibility_checks"])
                        ),
                        "search_seconds": float(result["seconds"]),
                        "rollout_label_rate": float(
                            diagnostic["rollout_label_rate"]
                        ),
                        "rollout_start_ratio_eval_only": float(
                            diagnostic[
                                "rollout_rank_to_oracle_ratio_start_eval_only"
                            ]
                        ),
                        "euclid_focal_expansions": int(baseline["expansions"]),
                        "delta_vs_euclid_focal": int(result["expansions"])
                        - int(baseline["expansions"]),
                        "euclid_astar_expansions": int(euclid_astar["expansions"]),
                        "delta_vs_euclid_astar": int(result["expansions"])
                        - int(euclid_astar["expansions"]),
                        "c13e_exact_expansions": int(
                            exact_reference["expansions"]
                        ),
                        "delta_vs_c13e_exact": int(result["expansions"])
                        - int(exact_reference["expansions"]),
                        "shared_oracle_expansions": int(
                            oracle_reference["expansions"]
                        ),
                        "delta_vs_shared_oracle": int(result["expansions"])
                        - int(oracle_reference["expansions"]),
                    }
                )

    euclid_control = {
        (int(row["world_index"]), float(row["focal_w"])): int(row["expansions"])
        for row in rows
        if row["provider"] == "euclid_rank"
    }
    if len(euclid_control) != len(bundles) * len(focal_ws):
        raise RuntimeError("same-search Euclidean control is incomplete")
    for row in rows:
        key = (int(row["world_index"]), float(row["focal_w"]))
        control_expansions = int(euclid_control[key])
        row["same_search_euclid_expansions"] = control_expansions
        row["delta_vs_same_search_euclid"] = (
            int(row["expansions"]) - control_expansions
        )
    return rows, baselines, diagnostics, anchor_checks


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    required_focal_win_fraction: float,
    required_euclid_rank_wins: int,
) -> List[Dict[str, Any]]:
    grouped: DefaultDict[Tuple[str, float, float], List[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[
            (str(row["provider"]), float(row["alpha"]), float(row["focal_w"]))
        ].append(row)

    summaries: List[Dict[str, Any]] = []
    for (provider, alpha, focal_w), group in sorted(grouped.items()):
        focal_deltas = np.asarray(
            [float(row["delta_vs_euclid_focal"]) for row in group],
            dtype=np.float64,
        )
        control_deltas = np.asarray(
            [float(row["delta_vs_same_search_euclid"]) for row in group],
            dtype=np.float64,
        )
        focal_wins = int(np.sum(focal_deltas < 0.0))
        control_wins = int(np.sum(control_deltas < 0.0))
        required_focal_wins = int(
            math.ceil(float(required_focal_win_fraction) * len(group))
        )
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
        mean_focal_delta = float(np.mean(focal_deltas))
        mean_control_delta = float(np.mean(control_deltas))
        is_calibrated_candidate = float(alpha) > 0.0 and float(alpha) < 1.0
        safety_pass = bool(
            certification_rate == 1.0
            and bound_violations == 0
            and path_failures == 0
            and lower_bound_failures == 0
            and accounting_failures == 0
        )
        total_checks = int(
            np.sum([int(row["rank_eligibility_checks"]) for row in group])
        )
        total_choices = int(
            np.sum([int(row["rank_eligible_choices"]) for row in group])
        )
        summaries.append(
            {
                "provider": provider,
                "alpha": float(alpha),
                "focal_w": float(focal_w),
                "worlds": int(len(group)),
                "development_candidate": bool(is_calibrated_candidate),
                "safety_pass": safety_pass,
                "calibration_gate_pass": bool(
                    is_calibrated_candidate
                    and safety_pass
                    and focal_wins >= required_focal_wins
                    and mean_focal_delta < 0.0
                    and control_wins >= int(required_euclid_rank_wins)
                    and mean_control_delta < 0.0
                ),
                "required_focal_wins": int(required_focal_wins),
                "required_euclid_rank_wins": int(required_euclid_rank_wins),
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
                "rank_eligible_choice_rate": float(
                    total_choices / max(1, total_checks)
                ),
                "euclid_focal_expansions_mean": float(
                    np.mean([float(row["euclid_focal_expansions"]) for row in group])
                ),
                "delta_vs_euclid_focal_mean": mean_focal_delta,
                "delta_vs_euclid_focal_median": float(np.median(focal_deltas)),
                "focal_wins": focal_wins,
                "focal_ties": int(np.sum(focal_deltas == 0.0)),
                "focal_losses": int(np.sum(focal_deltas > 0.0)),
                "same_search_euclid_expansions_mean": float(
                    np.mean(
                        [float(row["same_search_euclid_expansions"]) for row in group]
                    )
                ),
                "delta_vs_same_search_euclid_mean": mean_control_delta,
                "euclid_rank_wins": control_wins,
                "euclid_rank_ties": int(np.sum(control_deltas == 0.0)),
                "euclid_rank_losses": int(np.sum(control_deltas > 0.0)),
                "c13e_exact_expansions_mean": float(
                    np.mean([float(row["c13e_exact_expansions"]) for row in group])
                ),
                "shared_oracle_expansions_mean": float(
                    np.mean([float(row["shared_oracle_expansions"]) for row in group])
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


def build_verdict(
    summaries: Sequence[Mapping[str, Any]],
    primary_w: float,
) -> Dict[str, Any]:
    primary = [
        dict(row)
        for row in summaries
        if math.isclose(float(row["focal_w"]), float(primary_w), abs_tol=1.0e-12)
    ]
    controls = [row for row in primary if row["provider"] == "euclid_rank"]
    exact = [row for row in primary if row["provider"] == "rollout_exact"]
    if len(controls) != 1 or len(exact) != 1:
        raise ValueError("primary Euclidean or exact control is missing or duplicated")
    passing = [row for row in primary if bool(row["calibration_gate_pass"])]
    selected = (
        min(passing, key=lambda row: (float(row["expansions_mean"]), float(row["alpha"])))
        if passing
        else None
    )
    found = selected is not None
    return {
        "primary_w": float(primary_w),
        "development_cohort_reused": True,
        "calibration_candidate_found": found,
        "verdict": (
            "calibration_mechanism_found_development_only"
            if found
            else "calibration_not_sufficient_move_to_local_objective"
        ),
        "authorization": (
            "replicate_fixed_alpha_on_fresh_worlds_before_models"
            if found
            else "advance_to_exact_bounded_local_escape_target"
        ),
        "selected_candidate": selected,
        "same_search_euclid_control": controls[0],
        "uncalibrated_exact_control": exact[0],
        "gate_definition": {
            "all_safety_checks": "pass",
            "wins_vs_matched_euclid_focal": "configured five-of-six contract",
            "mean_delta_vs_matched_euclid_focal": "strictly_negative",
            "wins_vs_same_search_euclid_rank": "configured absolute count",
            "mean_delta_vs_same_search_euclid_rank": "strictly_negative",
            "fresh_world_replication_required": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-F shared-search calibration")
    parser.add_argument("--study-dir", default=CalibrationConfig.study_dir)
    parser.add_argument("--exact-dir", default=CalibrationConfig.exact_dir)
    parser.add_argument("--oracle-dir", default=CalibrationConfig.oracle_dir)
    parser.add_argument("--out-dir", default=CalibrationConfig.out_dir)
    parser.add_argument("--focal-ws", default=CalibrationConfig.focal_ws)
    parser.add_argument("--primary-w", type=float, default=CalibrationConfig.primary_w)
    parser.add_argument("--alphas", default=CalibrationConfig.alphas)
    parser.add_argument(
        "--budget-factor", type=float, default=CalibrationConfig.budget_factor
    )
    parser.add_argument(
        "--required-focal-win-fraction",
        type=float,
        default=CalibrationConfig.required_focal_win_fraction,
    )
    parser.add_argument(
        "--required-euclid-rank-wins",
        type=int,
        default=CalibrationConfig.required_euclid_rank_wins,
    )
    return parser.parse_args()


def _resolve_default_paths(cfg: CalibrationConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    study_was_resolved = False
    for field, default in (
        ("study_dir", CalibrationConfig.study_dir),
        ("exact_dir", CalibrationConfig.exact_dir),
        ("oracle_dir", CalibrationConfig.oracle_dir),
    ):
        value = getattr(cfg, field)
        if value != default:
            continue
        candidate = script_dir / value
        if not Path(value).exists() and candidate.exists():
            setattr(cfg, field, str(candidate))
            if field == "study_dir":
                study_was_resolved = True
    if cfg.out_dir == CalibrationConfig.out_dir and study_was_resolved:
        cfg.out_dir = str(script_dir / cfg.out_dir)


def main() -> None:
    cfg = CalibrationConfig(**vars(parse_args()))
    _resolve_default_paths(cfg)
    if not 0.0 < float(cfg.required_focal_win_fraction) <= 1.0:
        raise ValueError("required-focal-win-fraction must be in (0, 1]")
    focal_ws = C13.parse_float_csv(cfg.focal_ws)
    alphas = C13.parse_float_csv(cfg.alphas)
    if not any(
        math.isclose(float(value), float(cfg.primary_w), abs_tol=1.0e-12)
        for value in focal_ws
    ):
        raise ValueError("primary-w must appear in focal-ws")
    if not any(math.isclose(alpha, 0.0, abs_tol=1.0e-12) for alpha in alphas):
        raise ValueError("alphas must include 0.0 for the same-search control")
    if not any(math.isclose(alpha, 1.0, abs_tol=1.0e-12) for alpha in alphas):
        raise ValueError("alphas must include 1.0 for exact replay verification")

    study_cfg, source_manifest = S.load_study(cfg.study_dir)
    bundles = I.collect_audit_bundles(study_cfg)
    replay = S.verify_audit_replay(cfg.study_dir, bundles)
    exact_raw = (
        Path(cfg.exact_dir) / "results" / "shared_queue_exact_target_raw.csv"
    )
    oracle_raw = (
        Path(cfg.oracle_dir) / "results" / "shared_queue_oracle_raw.csv"
    )
    exact_rows = T.load_reference_rows(exact_raw, "rollout_exact")
    oracle_rows = T.load_reference_rows(oracle_raw, "oracle_eval_only")
    rows, baselines, diagnostics, anchor_checks = evaluate_calibration(
        cfg,
        study_cfg,
        bundles,
        exact_rows,
        oracle_rows,
    )
    summaries = summarize_rows(
        rows,
        cfg.required_focal_win_fraction,
        cfg.required_euclid_rank_wins,
    )
    verdict = build_verdict(summaries, cfg.primary_w)

    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    raw_path = C13.write_csv(result_dir / "shared_queue_calibration_raw.csv", rows)
    baseline_path = C13.write_csv(
        result_dir / "shared_queue_calibration_baselines.csv", baselines
    )
    diagnostic_path = C13.write_csv(
        result_dir / "shared_queue_calibration_target_diagnostics.csv", diagnostics
    )
    summary_path = C13.write_csv(
        result_dir / "shared_queue_calibration_summary.csv", summaries
    )
    verdict_path = C13.write_json(result_dir / "gate_verdict.json", verdict)

    keys = [
        (str(row["provider"]), int(row["world_index"]), float(row["focal_w"]))
        for row in rows
    ]
    exact_replay_rows = [row for row in rows if row["provider"] == "rollout_exact"]
    exact_replay_expansion_mismatches = int(
        np.sum(
            [
                int(row["expansions"]) != int(row["c13e_exact_expansions"])
                for row in exact_replay_rows
            ]
        )
    )
    verification = {
        "audit_replay": replay,
        "worlds": int(len(bundles)),
        "focal_ws": focal_ws,
        "alphas": alphas,
        "raw_rows": int(len(rows)),
        "expected_raw_rows": int(len(bundles) * len(focal_ws) * len(alphas)),
        "duplicate_keys": int(len(keys) - len(set(keys))),
        "exact_replay_rows": int(len(exact_replay_rows)),
        "exact_replay_expansion_mismatches": exact_replay_expansion_mismatches,
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
        "development_cohort_reused": True,
        "fresh_replication_required": True,
        "training_performed": False,
        "model_loading_performed": False,
        "shortest_path_target": False,
        "shortest_path_use": "posthoc_cost_and_target_diagnostics_only",
    }
    structural_failures = (
        verification["duplicate_keys"]
        + verification["exact_replay_expansion_mismatches"]
        + verification["expansion_accounting_failures"]
        + verification["states_expanded_more_than_twice"]
    )
    if verification["raw_rows"] != verification["expected_raw_rows"]:
        raise RuntimeError("calibration output row count mismatch")
    if structural_failures:
        raise RuntimeError("calibration structural verification failed")
    verification_path = C13.write_json(result_dir / "verification.json", verification)

    source_paths = {
        "implementation": Path(__file__).resolve(),
        "shared_queue_implementation": Path(Q.__file__).resolve(),
        "exact_target_implementation": Path(T.__file__).resolve(),
        "source_study_manifest": Path(cfg.study_dir) / "manifest.json",
        "source_target_audit": Path(cfg.study_dir)
        / "results"
        / "target_reliability_raw.csv",
        "source_exact_raw": exact_raw,
        "source_oracle_raw": oracle_raw,
    }
    output_paths = {
        "raw": raw_path,
        "baselines": baseline_path,
        "target_diagnostics": diagnostic_path,
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
        "experiment": "C13-F training-free shared-search rollout calibration",
        "runner_config": asdict(cfg),
        "source_study_config": asdict(study_cfg),
        "source_study_experiment": source_manifest.get("experiment"),
        "training_performed": False,
        "model_loading_performed": False,
        "shortest_path_target": False,
        "rank_family": "euclidean_plus_fixed_alpha_times_rollout_residual",
        "causal_question": "absolute_scale_versus_rollout_ordering",
        "development_cohort_reused": True,
        "fresh_replication_required": True,
        "selection_policy": "best_passing_primary_candidate_then_fresh_replication",
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "integrity": str(integrity_path),
    }
    manifest_path = C13.write_json(Path(cfg.out_dir) / "manifest.json", manifest)

    print(f"verdict={verdict['verdict']}")
    print(f"authorization={verdict['authorization']}")
    if verdict["selected_candidate"] is not None:
        print(f"selected_provider={verdict['selected_candidate']['provider']}")
        print(f"selected_alpha={verdict['selected_candidate']['alpha']}")
    for name, path in {
        **output_paths,
        "integrity": integrity_path,
        "manifest": manifest_path,
    }.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
