#!/usr/bin/env python3
"""C13-O summary-last HRM readout-alignment experiment.

The study imports the frozen C13-N trainer/evaluator, binds it to explicit model
families, and adds a three-way summary-last/trimmed/flat comparison.  C13-J
cohorts and caches and C13-N trimmed checkpoints remain read-only controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import torch

import continuous_prm_common as C
import continuous_prm_c13_certified_search as S
import continuous_prm_c13_hrm_substitution as N
import continuous_prm_c13_identifiability as I
import continuous_prm_c13_lhbl_generated_v3 as H
import continuous_prm_c13_lhbl_multisuite as J
import continuous_prm_c13_state_heuristic as C13
import continuous_prm_c13_lhbl_c7_comparison as X


SUMMARY_FAMILY = "hrm_summary_last"
TRIMMED_FAMILY = "hrm_trimmed"
FLAT_FAMILY = "flat_mlp"
REFERENCE_ARMS = ("euclid", "field_hrm", "scalar_hrm")
PREREGISTRATION = (
    "../../docs/experiments/continuous/c13/design/"
    "2026-07-17-c13o-hrm-summary-last-alignment.md"
)


@dataclass
class HrmAlignmentConfig(N.HrmSubstitutionConfig):
    trimmed_run_dir: str = "runs/c13_hrm_substitution"
    out_dir: str = "runs/c13_hrm_alignment"
    preregistration: str = PREREGISTRATION


def resolve_paths(cfg: HrmAlignmentConfig) -> None:
    N.resolve_paths(cfg)
    path = Path(cfg.trimmed_run_dir)
    if not path.is_absolute():
        cfg.trimmed_run_dir = str((Path(__file__).resolve().parent / path).resolve())


@contextmanager
def bound_n_families(primary: str, secondary: str) -> Iterator[None]:
    """Temporarily bind the frozen two-family C13-N machinery explicitly."""

    previous_primary = N.HRM_FAMILY
    previous_secondary = N.FLAT_FAMILY
    N.HRM_FAMILY = str(primary)
    N.FLAT_FAMILY = str(secondary)
    try:
        yield
    finally:
        N.HRM_FAMILY = previous_primary
        N.FLAT_FAMILY = previous_secondary


def verify_integrity_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"required integrity manifest is missing: {path}")
    payload = H._read_json(path)
    checked = 0
    mismatches: List[str] = []
    for section in ("inputs", "outputs"):
        for name, record in payload.get(section, {}).items():
            artifact = Path(record["path"])
            if not artifact.exists():
                mismatches.append(f"{section}/{name}:missing")
            elif S.file_sha256(artifact) != str(record["sha256"]):
                mismatches.append(f"{section}/{name}:sha256")
            checked += 1
    if mismatches:
        raise RuntimeError(f"frozen integrity mismatch in {path}: {mismatches}")
    return {
        "path": str(path),
        "sha256": S.file_sha256(path),
        "artifacts_checked": checked,
        "mismatches": mismatches,
    }


def _state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialization_audit(
    source_train_cfg: H.LHBLConfig,
    study_cfg: I.StudyConfig,
) -> Dict[str, Any]:
    model_cfg = H.model_config(source_train_cfg, study_cfg)
    seed = int(source_train_cfg.seed) + 1009
    C.set_global_seed(seed)
    trimmed = H.build_lhbl_model(TRIMMED_FAMILY, model_cfg)
    C.set_global_seed(seed)
    summary = H.build_lhbl_model(SUMMARY_FAMILY, model_cfg)
    left = trimmed.state_dict()
    right = summary.state_dict()
    keys_equal = list(left) == list(right)
    tensor_mismatches = [
        name
        for name in left
        if name not in right or not torch.equal(left[name], right[name])
    ]
    result = {
        "seed": seed,
        "keys_equal": keys_equal,
        "tensor_mismatches": tensor_mismatches,
        "all_initial_tensors_equal": bool(keys_equal and not tensor_mismatches),
        "trimmed_state_sha256": _state_dict_sha256(trimmed),
        "summary_last_state_sha256": _state_dict_sha256(summary),
        "trimmed_parameters": int(sum(p.numel() for p in trimmed.parameters())),
        "summary_last_parameters": int(sum(p.numel() for p in summary.parameters())),
        "trimmed_readout_mode": str(trimmed.readout_mode),
        "summary_last_readout_mode": str(summary.readout_mode),
    }
    if not result["all_initial_tensors_equal"]:
        raise RuntimeError("summary-last and trimmed HRM initial states differ")
    if result["trimmed_parameters"] != result["summary_last_parameters"]:
        raise RuntimeError("summary-last and trimmed HRM parameter counts differ")
    return result


def training_config(
    cfg: HrmAlignmentConfig, source: H.LHBLConfig
) -> H.LHBLConfig:
    with bound_n_families(SUMMARY_FAMILY, FLAT_FAMILY):
        return N._hrm_training_config(cfg, source)


def write_or_verify_binding(
    cfg: HrmAlignmentConfig,
    source_train_cfg: H.LHBLConfig,
    study_cfg: I.StudyConfig,
    source_audit: Mapping[str, Any],
    control_audit: Mapping[str, Any],
    init_audit: Mapping[str, Any],
) -> Tuple[Path, Dict[str, Any]]:
    train_cfg = training_config(cfg, source_train_cfg)
    payload = {
        "experiment": "C13-O summary-last HRM readout alignment",
        "intervention": "move_summary_token_to_final_valid_position",
        "candidate_family": SUMMARY_FAMILY,
        "frozen_recurrent_control": TRIMMED_FAMILY,
        "frozen_flat_control": FLAT_FAMILY,
        "training_config": asdict(train_cfg),
        "model_config": asdict(H.model_config(train_cfg, study_cfg)),
        "initialization_audit": dict(init_audit),
        "source_audit": {
            "source_integrity_sha256": source_audit["frozen_integrity"][
                "integrity_sha256"
            ],
            "source_cohorts_sha256": source_audit["source_cohorts_sha256"],
            "feature_caches_reused": source_audit["feature_caches_reused"],
        },
        "trimmed_control_integrity": dict(control_audit),
        "hashes": {
            "preregistration": S.file_sha256(Path(cfg.preregistration)),
            "wrapper": S.file_sha256(Path(__file__).resolve()),
            "imported_frozen_trainer": S.file_sha256(Path(N.__file__).resolve()),
            "lhbl_trainer": S.file_sha256(Path(H.__file__).resolve()),
            "model_definition": S.file_sha256(Path(I.__file__).resolve()),
        },
        "note": (
            "The imported C13-N trainer is bound to hrm_summary_last; its own "
            "fingerprint remains nested in training_state.json."
        ),
    }
    binding = {
        "fingerprint": N._json_hash(payload),
        "fingerprint_payload": payload,
    }
    path = Path(cfg.out_dir) / "results" / "training_binding.json"
    if path.exists():
        previous = H._read_json(path)
        if previous != binding:
            raise RuntimeError("existing C13-O training binding differs")
    else:
        C13.write_json(path, binding)
    return path, binding


def train_summary_last(
    cfg: HrmAlignmentConfig,
    source_train_cfg: H.LHBLConfig,
    study_cfg: I.StudyConfig,
    train_bundles: Sequence[H.WorldBundle],
    validation_bundles: Sequence[H.WorldBundle],
) -> Tuple[List[Path], List[Dict[str, Any]], Dict[str, Any]]:
    with bound_n_families(SUMMARY_FAMILY, FLAT_FAMILY):
        return N.train_hrm(
            cfg,
            source_train_cfg,
            study_cfg,
            train_bundles,
            validation_bundles,
        )


def _pair_config(
    cfg: HrmAlignmentConfig,
    primary_run_dir: str,
    secondary_run_dir: str,
) -> HrmAlignmentConfig:
    result = HrmAlignmentConfig(**asdict(cfg))
    result.out_dir = str(primary_run_dir)
    result.multisuite_run_dir = str(secondary_run_dir)
    return result


def _same_result(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    text_fields = (
        "phase",
        "suite",
        "arm",
        "family",
        "runtime_information_boundary",
    )
    integer_fields = (
        "suite_world_index",
        "world_index",
        "world_seed",
        "roadmap_seed",
        "expansions",
    )
    float_fields = ("cost", "optimal", "cost_ratio_eval_only")
    if any(str(left[key]) != str(right[key]) for key in text_fields):
        return False
    if any(int(left[key]) != int(right[key]) for key in integer_fields):
        return False
    if N._as_bool(left["found"]) != N._as_bool(right["found"]):
        return False
    if N._as_bool(left["path_valid"]) != N._as_bool(right["path_valid"]):
        return False
    return all(
        math.isclose(float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1e-12)
        for key in float_fields
    )


def _expected_arms(
    iterations: Sequence[int], alphas: Sequence[float]
) -> set[str]:
    arms = set(REFERENCE_ARMS)
    for family in (SUMMARY_FAMILY, TRIMMED_FAMILY, FLAT_FAMILY):
        for iteration in iterations:
            for alpha in alphas:
                arms.add(N._arm_name(family, iteration, alpha))
    return arms


def evaluate_three_way(
    cfg: HrmAlignmentConfig,
    phase: str,
    bundles: Sequence[H.WorldBundle],
    records: Sequence[Mapping[str, Any]],
    multi_cfg: J.MultiSuiteConfig,
    iterations: Sequence[int],
    alphas: Sequence[float],
    results_dir: Path,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
    Dict[str, Path],
]:
    prefix = str(phase)
    summary_flat_cfg = _pair_config(
        cfg, cfg.out_dir, cfg.multisuite_run_dir
    )
    with bound_n_families(SUMMARY_FAMILY, FLAT_FAMILY):
        base_rows, base_diag, base_meta, base_models = N.evaluate_phase(
            summary_flat_cfg,
            phase,
            bundles,
            records,
            multi_cfg,
            iterations,
            alphas,
            results_dir / f"{prefix}_summary_flat_raw.csv",
            results_dir / f"{prefix}_summary_flat_diagnostics.csv",
            results_dir / f"{prefix}_summary_flat_meta.json",
        )

    trimmed_summary_cfg = _pair_config(
        cfg, cfg.trimmed_run_dir, cfg.out_dir
    )
    with bound_n_families(TRIMMED_FAMILY, SUMMARY_FAMILY):
        extra_rows, extra_diag, extra_meta, extra_models = N.evaluate_phase(
            trimmed_summary_cfg,
            phase,
            bundles,
            records,
            multi_cfg,
            iterations,
            alphas,
            results_dir / f"{prefix}_trimmed_summary_raw.csv",
            results_dir / f"{prefix}_trimmed_summary_diagnostics.csv",
            results_dir / f"{prefix}_trimmed_summary_meta.json",
        )

    base_lookup = {
        (int(row["world_seed"]), str(row["arm"])): row for row in base_rows
    }
    overlap_checks = 0
    mismatches: List[str] = []
    for row in extra_rows:
        key = (int(row["world_seed"]), str(row["arm"]))
        if key in base_lookup:
            overlap_checks += 1
            if not _same_result(base_lookup[key], row):
                mismatches.append(f"{key[0]}:{key[1]}")
    if mismatches:
        raise RuntimeError(f"duplicate pair evaluation changed results: {mismatches[:8]}")

    rows = list(base_rows) + [
        row for row in extra_rows if str(row["family"]) == TRIMMED_FAMILY
    ]
    diagnostics = list(base_diag) + [
        row for row in extra_diag if str(row["family"]) == TRIMMED_FAMILY
    ]
    expected = _expected_arms(iterations, alphas)
    for bundle in bundles:
        seed = int(bundle.world_seed)
        world_rows = [row for row in rows if int(row["world_seed"]) == seed]
        observed = [str(row["arm"]) for row in world_rows]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise RuntimeError(f"C13-O {phase} arm set is incomplete for seed {seed}")
    expected_diag = len(bundles) * 3 * len(iterations)
    if len(diagnostics) != expected_diag:
        raise RuntimeError(
            f"C13-O {phase} diagnostics count {len(diagnostics)} != {expected_diag}"
        )

    raw_path = C13.write_csv(results_dir / f"{prefix}_raw.csv", rows)
    diagnostic_path = C13.write_csv(
        results_dir / f"{prefix}_diagnostics.csv", diagnostics
    )
    provenance = {
        "phase": phase,
        "worlds": len(bundles),
        "rows": len(rows),
        "diagnostics": len(diagnostics),
        "expected_arms_per_world": len(expected),
        "duplicate_pair_checks": overlap_checks,
        "duplicate_pair_mismatches": mismatches,
        "feature_cache_mismatches": int(
            base_meta["feature_cache_mismatches"]
        )
        + int(extra_meta["feature_cache_mismatches"]),
        "pair_fingerprints": {
            "summary_flat": base_meta["fingerprint"],
            "trimmed_summary": extra_meta["fingerprint"],
        },
        "canonical_raw": str(raw_path),
        "canonical_diagnostics": str(diagnostic_path),
        "wrapper_sha256": S.file_sha256(Path(__file__).resolve()),
    }
    meta_path = C13.write_json(results_dir / f"{prefix}_meta.json", provenance)
    provenance["canonical_meta"] = str(meta_path)
    models = dict(base_models)
    models.update(extra_models)
    return rows, diagnostics, provenance, models


def _paired_stats(
    cfg: HrmAlignmentConfig,
    rows: Sequence[Mapping[str, Any]],
    left_arm: str,
    right_arm: str,
    seed_offset: int,
    scope: str = "POOLED",
) -> Tuple[
    List[Tuple[Mapping[str, Any], Mapping[str, Any]]],
    np.ndarray,
    float,
    float,
]:
    pairs, delta = N._paired(rows, left_arm, right_arm, scope)
    if not len(delta):
        raise RuntimeError(f"missing paired rows: {left_arm} vs {right_arm}")
    low, high = X._bootstrap_mean_ci(
        delta,
        int(cfg.bootstrap_replicates),
        int(cfg.bootstrap_seed) + int(seed_offset),
    )
    return pairs, delta, float(low), float(high)


def _costs(
    pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [float(left["cost_ratio_eval_only"]) for left, _ in pairs],
            dtype=np.float64,
        ),
        np.asarray(
            [float(right["cost_ratio_eval_only"]) for _, right in pairs],
            dtype=np.float64,
        ),
    )


def _wtl(delta: np.ndarray) -> Dict[str, int]:
    return {
        "wins": int(np.sum(delta < 0.0)),
        "ties": int(np.sum(delta == 0.0)),
        "losses": int(np.sum(delta > 0.0)),
    }


def candidate_summary(
    cfg: HrmAlignmentConfig,
    rows: Sequence[Mapping[str, Any]],
    iteration: int,
    alpha: float,
    expected_worlds: int,
) -> Dict[str, Any]:
    summary_arm = N._arm_name(SUMMARY_FAMILY, iteration, alpha)
    trimmed_arm = N._arm_name(TRIMMED_FAMILY, iteration, alpha)
    flat_arm = N._arm_name(FLAT_FAMILY, iteration, alpha)
    seed_cell = int(iteration) * 10_000 + int(float(alpha) * 1000)
    field_pairs, field_delta, field_low, field_high = _paired_stats(
        cfg, rows, summary_arm, "field_hrm", seed_cell
    )
    flat_pairs, flat_delta, flat_low, flat_high = _paired_stats(
        cfg, rows, summary_arm, flat_arm, 500_000 + seed_cell
    )
    trimmed_pairs, trimmed_delta, trimmed_low, trimmed_high = _paired_stats(
        cfg, rows, summary_arm, trimmed_arm, 1_000_000 + seed_cell
    )
    summary_cost, field_cost = _costs(field_pairs)
    _, flat_cost = _costs(flat_pairs)
    _, trimmed_cost = _costs(trimmed_pairs)
    suite_means: Dict[str, float] = {}
    for suite in J.DEV_SUITES:
        _, delta = N._paired(rows, summary_arm, "field_hrm", suite)
        if not len(delta):
            raise RuntimeError(f"missing suite pair for {suite}")
        suite_means[suite] = float(np.mean(delta))
    negative_suites = int(sum(value < 0.0 for value in suite_means.values()))
    validity = all(
        len(pairs) == int(expected_worlds)
        for pairs in (field_pairs, flat_pairs, trimmed_pairs)
    )
    method_conditions = {
        "all_summary_last_paths_valid": validity,
        "field_expansion_ci_upper_below_zero": field_high < 0.0,
        "negative_suite_means_at_least_four": negative_suites
        >= int(cfg.required_negative_suites),
        "mean_cost_within_field_margin": float(np.mean(summary_cost))
        <= float(np.mean(field_cost)) + float(cfg.mean_cost_margin) + 1e-12,
        "max_cost_within_field_margin": float(np.max(summary_cost))
        <= float(np.max(field_cost)) + float(cfg.max_cost_margin) + 1e-12,
        "mean_cost_within_flat_margin": float(np.mean(summary_cost))
        <= float(np.mean(flat_cost)) + float(cfg.mean_cost_margin) + 1e-12,
        "max_cost_within_flat_margin": float(np.max(summary_cost))
        <= float(np.max(flat_cost)) + float(cfg.max_cost_margin) + 1e-12,
    }
    readout_conditions = {
        "trimmed_expansion_ci_upper_below_zero": trimmed_high < 0.0,
        "mean_cost_within_trimmed_margin": float(np.mean(summary_cost))
        <= float(np.mean(trimmed_cost)) + float(cfg.mean_cost_margin) + 1e-12,
        "max_cost_within_trimmed_margin": float(np.max(summary_cost))
        <= float(np.max(trimmed_cost)) + float(cfg.max_cost_margin) + 1e-12,
    }
    return {
        "iteration": int(iteration),
        "alpha": float(alpha),
        "worlds": len(field_pairs),
        "summary_last_expansions_mean": float(
            np.mean([float(left["expansions"]) for left, _ in field_pairs])
        ),
        "summary_last_cost_ratio_mean": float(np.mean(summary_cost)),
        "summary_last_cost_ratio_max": float(np.max(summary_cost)),
        "field_hrm_expansions_mean": float(
            np.mean([float(right["expansions"]) for _, right in field_pairs])
        ),
        "field_hrm_cost_ratio_mean": float(np.mean(field_cost)),
        "field_hrm_cost_ratio_max": float(np.max(field_cost)),
        "delta_vs_field_hrm_mean": float(np.mean(field_delta)),
        "delta_vs_field_hrm_ci95_low": field_low,
        "delta_vs_field_hrm_ci95_high": field_high,
        "field_wtl": _wtl(field_delta),
        "negative_suites": negative_suites,
        "suite_delta_means": suite_means,
        "flat_expansions_mean": float(
            np.mean([float(right["expansions"]) for _, right in flat_pairs])
        ),
        "flat_cost_ratio_mean": float(np.mean(flat_cost)),
        "flat_cost_ratio_max": float(np.max(flat_cost)),
        "delta_vs_flat_mean": float(np.mean(flat_delta)),
        "delta_vs_flat_ci95_low": flat_low,
        "delta_vs_flat_ci95_high": flat_high,
        "flat_wtl": _wtl(flat_delta),
        "trimmed_expansions_mean": float(
            np.mean([float(right["expansions"]) for _, right in trimmed_pairs])
        ),
        "trimmed_cost_ratio_mean": float(np.mean(trimmed_cost)),
        "trimmed_cost_ratio_max": float(np.max(trimmed_cost)),
        "delta_vs_trimmed_mean": float(np.mean(trimmed_delta)),
        "delta_vs_trimmed_ci95_low": trimmed_low,
        "delta_vs_trimmed_ci95_high": trimmed_high,
        "trimmed_wtl": _wtl(trimmed_delta),
        "method_gate_conditions": method_conditions,
        "method_gate_pass": bool(all(method_conditions.values())),
        "readout_gate_conditions": readout_conditions,
        "readout_gate_pass": bool(all(readout_conditions.values())),
        "overall_gate_pass": bool(
            all(method_conditions.values()) and all(readout_conditions.values())
        ),
    }


def summarize_development(
    cfg: HrmAlignmentConfig,
    rows: Sequence[Mapping[str, Any]],
    iterations: Sequence[int],
    alphas: Sequence[float],
    expected_worlds: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidates = [
        candidate_summary(cfg, rows, iteration, alpha, expected_worlds)
        for iteration in iterations
        for alpha in alphas
    ]
    primary = [
        row
        for row in candidates
        if int(row["iteration"]) == int(cfg.primary_iteration)
        and math.isclose(float(row["alpha"]), float(cfg.primary_alpha))
    ]
    if len(primary) != 1:
        raise RuntimeError("fixed primary C13-O cell is missing")
    passing = [row for row in candidates if bool(row["overall_gate_pass"])]
    selected = (
        min(
            passing,
            key=lambda row: (
                float(row["summary_last_expansions_mean"]),
                float(row["summary_last_cost_ratio_mean"]),
                int(row["iteration"]),
                float(row["alpha"]),
            ),
        )
        if passing
        else None
    )
    verdict = {
        "verdict": (
            "summary_last_alignment_development_pass_requires_confirmation"
            if selected is not None
            else "summary_last_alignment_development_gate_failed"
        ),
        "gate_pass": selected is not None,
        "fixed_primary_cell": primary[0],
        "method_passing_candidates": int(
            sum(bool(row["method_gate_pass"]) for row in candidates)
        ),
        "readout_passing_candidates": int(
            sum(bool(row["readout_gate_pass"]) for row in candidates)
        ),
        "overall_passing_candidates": len(passing),
        "selected_candidate": selected,
        "authorization": (
            "run_selected_cell_on_seed_offset_20000000"
            if selected is not None
            else "stop_without_confirmation_or_retuning"
        ),
    }
    return candidates, verdict


def _roadmap_seed_groups(cfg: HrmAlignmentConfig) -> Dict[str, set[int]]:
    groups: Dict[str, set[int]] = {}
    c13j = H._read_json(
        Path(cfg.multisuite_run_dir) / "results" / "cohorts.json"
    )
    for split, rows in c13j["records"].items():
        groups[f"c13j_{split}"] = {
            int(row["roadmap_seed"]) for row in rows if "roadmap_seed" in row
        }
    optional = {
        "c13l_alpha_calibration": Path(cfg.c13l_run_dir)
        / "results"
        / "calibration_cohort.json",
        "c13m_confirmation": Path(cfg.c13m_run_dir)
        / "results"
        / "confirmation_cohort.json",
    }
    for name, path in optional.items():
        if path.exists():
            rows = H._read_json(path).get("records", [])
            groups[name] = {
                int(row["roadmap_seed"])
                for row in rows
                if "roadmap_seed" in row
            }
    return groups


def build_confirmation_cohort(
    cfg: HrmAlignmentConfig,
    source_multi_cfg: J.MultiSuiteConfig,
) -> Tuple[List[H.WorldBundle], List[Dict[str, Any]], Dict[str, Any]]:
    bundles, records, verification = N.build_confirmation_cohort(
        cfg, source_multi_cfg
    )
    current = {int(row["roadmap_seed"]) for row in records}
    overlaps = {
        name: len(current & seeds)
        for name, seeds in _roadmap_seed_groups(cfg).items()
    }
    verification = dict(verification)
    verification["unique_roadmap_seeds"] = len(current)
    verification["roadmap_seed_prior_overlap"] = overlaps
    if len(current) != len(records):
        raise RuntimeError("C13-O confirmation roadmap seeds are not unique")
    if any(overlaps.values()):
        raise RuntimeError(f"C13-O confirmation roadmap seed overlap: {overlaps}")
    return bundles, records, verification


def confirmation_verdict(
    cfg: HrmAlignmentConfig,
    rows: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = int(cfg.confirmation_worlds_per_suite) * len(J.DEV_SUITES)
    result = candidate_summary(
        cfg,
        rows,
        int(selected["iteration"]),
        float(selected["alpha"]),
        expected,
    )
    return {
        "verdict": (
            "summary_last_readout_alignment_confirmed"
            if result["overall_gate_pass"]
            else "summary_last_readout_alignment_not_confirmed"
        ),
        "gate_pass": bool(result["overall_gate_pass"]),
        "method_gate_pass": bool(result["method_gate_pass"]),
        "readout_gate_pass": bool(result["readout_gate_pass"]),
        "candidate": result,
        "authorization": (
            "document_confirmed_readout_alignment"
            if result["overall_gate_pass"]
            else "document_confirmation_failure_without_retuning"
        ),
    }


def _candidate_csv_rows(
    candidates: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    nested = {
        "field_wtl",
        "flat_wtl",
        "trimmed_wtl",
        "suite_delta_means",
        "method_gate_conditions",
        "readout_gate_conditions",
    }
    return [
        {
            key: json.dumps(value, sort_keys=True) if key in nested else value
            for key, value in row.items()
        }
        for row in candidates
    ]


def write_generated_report(
    path: Path,
    development: Mapping[str, Any],
    confirmation: Mapping[str, Any] | None,
) -> Path:
    primary = development["fixed_primary_cell"]
    selected = development["selected_candidate"]
    lines = [
        "# C13-O summary-last HRM alignment",
        "",
        f"Development verdict: {development['verdict']}.",
        "",
        "## Fixed primary cell",
        "",
        (
            f"Iteration {primary['iteration']}, alpha {primary['alpha']:.2f}: "
            f"summary-last {primary['summary_last_expansions_mean']:.3f}, "
            f"trimmed HRM {primary['trimmed_expansions_mean']:.3f}, flat MLP "
            f"{primary['flat_expansions_mean']:.3f}, and field HRM "
            f"{primary['field_hrm_expansions_mean']:.3f} mean expansions."
        ),
        (
            f"Summary-last minus field: {primary['delta_vs_field_hrm_mean']:+.3f}, "
            f"95% CI [{primary['delta_vs_field_hrm_ci95_low']:+.3f}, "
            f"{primary['delta_vs_field_hrm_ci95_high']:+.3f}]."
        ),
        (
            f"Summary-last minus trimmed: {primary['delta_vs_trimmed_mean']:+.3f}, "
            f"95% CI [{primary['delta_vs_trimmed_ci95_low']:+.3f}, "
            f"{primary['delta_vs_trimmed_ci95_high']:+.3f}]."
        ),
        (
            f"Summary-last minus flat: {primary['delta_vs_flat_mean']:+.3f}, "
            f"95% CI [{primary['delta_vs_flat_ci95_low']:+.3f}, "
            f"{primary['delta_vs_flat_ci95_high']:+.3f}]."
        ),
        f"Method gate pass: {primary['method_gate_pass']}.",
        f"Direct readout gate pass: {primary['readout_gate_pass']}.",
        "",
        "## Development selection",
        "",
    ]
    if selected is None:
        lines.extend(
            [
                "No preregistered cell passed both the method and direct readout gates.",
                "The untouched confirmation cohort was not generated or evaluated.",
            ]
        )
    else:
        lines.extend(
            [
                (
                    f"Selected iteration {selected['iteration']}, alpha "
                    f"{selected['alpha']:.2f}."
                ),
                "",
                "## Untouched confirmation",
                "",
            ]
        )
        if confirmation is None:
            lines.append("Confirmation was authorized but not run in this invocation.")
        else:
            cell = confirmation["candidate"]
            lines.extend(
                [
                    f"Verdict: {confirmation['verdict']}.",
                    (
                        f"Across {cell['worlds']} worlds, summary-last minus field "
                        f"was {cell['delta_vs_field_hrm_mean']:+.3f} expansions "
                        f"(95% CI [{cell['delta_vs_field_hrm_ci95_low']:+.3f}, "
                        f"{cell['delta_vs_field_hrm_ci95_high']:+.3f}])."
                    ),
                    (
                        f"Summary-last minus trimmed was "
                        f"{cell['delta_vs_trimmed_mean']:+.3f} "
                        f"(95% CI [{cell['delta_vs_trimmed_ci95_low']:+.3f}, "
                        f"{cell['delta_vs_trimmed_ci95_high']:+.3f}])."
                    ),
                ]
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This study changes valid-token order/readout context only. It does not test",
            "persistent planning state, map-free sensing, formal search guarantees,",
            "wall-clock speedup, or general HRM superiority.",
        ]
    )
    C13.ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _existing_output_paths(results_dir: Path) -> Dict[str, Path]:
    candidates = {
        path.stem: path
        for path in results_dir.iterdir()
        if path.is_file() and path.name not in {"integrity.json"}
    }
    return candidates


def run(cfg: HrmAlignmentConfig) -> Dict[str, Any]:
    resolve_paths(cfg)
    if cfg.mode not in {"audit", "train", "develop", "full"}:
        raise ValueError("mode must be audit, train, develop, or full")
    iterations = N._parse_iterations(cfg.candidate_iterations)
    alphas = C13.parse_float_csv(cfg.alphas)
    if cfg.primary_iteration not in iterations or not any(
        math.isclose(value, cfg.primary_alpha) for value in alphas
    ):
        raise RuntimeError("primary cell must be in the candidate grid")
    if Path(cfg.out_dir).resolve() in {
        Path(cfg.multisuite_run_dir).resolve(),
        Path(cfg.trimmed_run_dir).resolve(),
    }:
        raise RuntimeError("C13-O output directory must not overwrite a control")
    results_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")

    (
        multi_cfg,
        source_train_cfg,
        study_cfg,
        train,
        validation,
        development,
        source_audit,
    ) = N.audit_and_rebuild_source(cfg)
    control_audit = verify_integrity_manifest(
        Path(cfg.trimmed_run_dir) / "integrity.json"
    )
    init_audit = initialization_audit(source_train_cfg, study_cfg)
    source_audit = dict(source_audit)
    source_audit["c13n_trimmed_control"] = control_audit
    source_audit["initialization_audit"] = init_audit
    source_audit_path = C13.write_json(
        results_dir / "source_audit.json", source_audit
    )
    binding_path, binding = write_or_verify_binding(
        cfg,
        source_train_cfg,
        study_cfg,
        source_audit,
        control_audit,
        init_audit,
    )
    if cfg.mode == "audit":
        return {
            "source_audit": source_audit,
            "training_binding": binding,
        }

    checkpoints, histories, training_state = train_summary_last(
        cfg, source_train_cfg, study_cfg, train, validation
    )
    if cfg.mode == "train":
        return {
            "source_audit": source_audit,
            "training_state": training_state,
            "checkpoints": [str(path) for path in checkpoints],
        }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    saved = H._read_json(
        Path(cfg.multisuite_run_dir) / "results" / "cohorts.json"
    )
    dev_rows, dev_diagnostics, dev_provenance, model_paths = evaluate_three_way(
        cfg,
        "development",
        development,
        saved["records"]["development"],
        multi_cfg,
        iterations,
        alphas,
        results_dir,
    )
    candidates, development_verdict = summarize_development(
        cfg, dev_rows, iterations, alphas, len(development)
    )
    candidate_path = C13.write_csv(
        results_dir / "development_candidates.csv",
        _candidate_csv_rows(candidates),
    )
    development_verdict_path = C13.write_json(
        results_dir / "development_verdict.json", development_verdict
    )

    confirmation_result: Dict[str, Any] | None = None
    confirmation_provenance: Dict[str, Any] | None = None
    if development_verdict["gate_pass"] and cfg.mode == "full":
        confirmation_bundles, confirmation_records, cohort_verification = (
            build_confirmation_cohort(cfg, multi_cfg)
        )
        C13.write_json(
            results_dir / "confirmation_cohort.json",
            {
                "records": confirmation_records,
                "verification": cohort_verification,
            },
        )
        selected = development_verdict["selected_candidate"]
        selected_iterations = [int(selected["iteration"])]
        selected_alphas = [float(selected["alpha"])]
        confirmation_multi_cfg = J.MultiSuiteConfig(**asdict(multi_cfg))
        confirmation_multi_cfg.out_dir = cfg.out_dir
        confirmation_rows, _, confirmation_provenance, confirmation_models = (
            evaluate_three_way(
                cfg,
                "confirmation",
                confirmation_bundles,
                confirmation_records,
                confirmation_multi_cfg,
                selected_iterations,
                selected_alphas,
                results_dir,
            )
        )
        model_paths.update(confirmation_models)
        confirmation_result = confirmation_verdict(
            cfg, confirmation_rows, selected
        )
        C13.write_json(
            results_dir / "confirmation_verdict.json", confirmation_result
        )

    report_path = results_dir / "C13O_RESULT.md"
    write_generated_report(report_path, development_verdict, confirmation_result)
    expected_dev_rows = len(development) * (
        len(REFERENCE_ARMS) + 3 * len(iterations) * len(alphas)
    )
    expected_dev_diagnostics = len(development) * 3 * len(iterations)
    verification = {
        "source_audit_pass": True,
        "c13n_control_artifacts_checked": control_audit["artifacts_checked"],
        "initial_states_identical": init_audit["all_initial_tensors_equal"],
        "training_status": training_state["status"],
        "training_iterations": len(histories),
        "expected_training_iterations": int(source_train_cfg.outer_iterations),
        "development_worlds": len(development),
        "development_rows": len(dev_rows),
        "expected_development_rows": expected_dev_rows,
        "development_diagnostics": len(dev_diagnostics),
        "expected_development_diagnostics": expected_dev_diagnostics,
        "development_provenance": dev_provenance,
        "development_gate_pass": bool(development_verdict["gate_pass"]),
        "confirmation_run": confirmation_result is not None,
        "confirmation_gate_pass": (
            bool(confirmation_result["gate_pass"])
            if confirmation_result is not None
            else None
        ),
        "confirmation_provenance": confirmation_provenance,
        "full_map_runtime_input": False,
        "shortest_path_target": False,
        "training_target_reads_dist_to_goal": False,
        "intervention": "summary_token_moved_to_final_valid_position",
        "training_device": cfg.train_device,
        "evaluation_device": cfg.evaluation_device,
    }
    verification["integrity_pass"] = bool(
        verification["initial_states_identical"]
        and verification["training_status"] == "complete"
        and verification["training_iterations"]
        == verification["expected_training_iterations"]
        and verification["development_rows"]
        == verification["expected_development_rows"]
        and verification["development_diagnostics"]
        == verification["expected_development_diagnostics"]
        and dev_provenance["feature_cache_mismatches"] == 0
        and not dev_provenance["duplicate_pair_mismatches"]
    )
    verification_path = C13.write_json(
        results_dir / "verification.json", verification
    )
    manifest = {
        "experiment": "C13-O summary-last HRM readout alignment",
        "config": asdict(cfg),
        "training_binding": binding,
        "source_audit": str(source_audit_path),
        "training_state": training_state,
        "development_verdict": development_verdict,
        "confirmation_verdict": confirmation_result,
        "model_checkpoints": {
            name: {"path": str(path), "sha256": S.file_sha256(path)}
            for name, path in model_paths.items()
        },
        "outputs": {
            "training_binding": str(binding_path),
            "development_candidates": str(candidate_path),
            "development_verdict": str(development_verdict_path),
            "report": str(report_path),
            "verification": str(verification_path),
        },
    }
    manifest_path = C13.write_json(Path(cfg.out_dir) / "manifest.json", manifest)
    output_paths = _existing_output_paths(results_dir)
    output_paths["manifest"] = manifest_path
    for path in checkpoints:
        output_paths[path.stem] = path
    integrity_path = C13.write_json(
        Path(cfg.out_dir) / "integrity.json",
        {
            "inputs": {
                "implementation": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": S.file_sha256(Path(__file__).resolve()),
                },
                "imported_c13n_trainer": {
                    "path": str(Path(N.__file__).resolve()),
                    "sha256": S.file_sha256(Path(N.__file__).resolve()),
                },
                "preregistration": {
                    "path": str(Path(cfg.preregistration)),
                    "sha256": S.file_sha256(Path(cfg.preregistration)),
                },
                "source_c13j_integrity": {
                    "path": str(Path(cfg.multisuite_run_dir) / "integrity.json"),
                    "sha256": S.file_sha256(
                        Path(cfg.multisuite_run_dir) / "integrity.json"
                    ),
                },
                "trimmed_c13n_integrity": {
                    "path": str(Path(cfg.trimmed_run_dir) / "integrity.json"),
                    "sha256": S.file_sha256(
                        Path(cfg.trimmed_run_dir) / "integrity.json"
                    ),
                },
            },
            "outputs": {
                name: {"path": str(path), "sha256": S.file_sha256(path)}
                for name, path in output_paths.items()
            },
        },
    )
    if not verification["integrity_pass"]:
        raise RuntimeError("C13-O verification failed")
    print(
        f"[c13o] {development_verdict['verdict']} "
        f"confirmation="
        f"{confirmation_result['verdict'] if confirmation_result else 'not_run'} "
        f"-> {report_path}",
        flush=True,
    )
    return {
        "development": development_verdict,
        "confirmation": confirmation_result,
        "verification": verification,
        "integrity": str(integrity_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("audit", "train", "develop", "full"), default="full"
    )
    parser.add_argument("--out-dir", default=HrmAlignmentConfig.out_dir)
    parser.add_argument("--train-device", default=HrmAlignmentConfig.train_device)
    parser.add_argument(
        "--evaluation-device", default=HrmAlignmentConfig.evaluation_device
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=HrmAlignmentConfig.bootstrap_replicates,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = HrmAlignmentConfig(
        mode=str(args.mode),
        out_dir=str(args.out_dir),
        train_device=str(args.train_device),
        evaluation_device=str(args.evaluation_device),
        bootstrap_replicates=int(args.bootstrap_replicates),
    )
    run(cfg)


if __name__ == "__main__":
    main()

