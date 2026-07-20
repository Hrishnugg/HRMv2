#!/usr/bin/env python3
"""Fresh3 confirmation of C13-H iteration 4 / alpha 0.5."""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

import continuous_prm_c13_certified_search as S
import continuous_prm_c13_lhbl_candidate_study as D
import continuous_prm_c13_lhbl_focal_fresh2 as F2
import continuous_prm_c13_lhbl_focal_matched_control_diagnostic as F
import continuous_prm_c13_lhbl_generated_v3 as H
import continuous_prm_c13_lhbl_replication as R
import continuous_prm_c13_state_heuristic as C13


EXPECTED_STUDY_RUNNER_SHA256 = (
    "d497cca4241ec680b136cedbed2d8bebb7c7ea4632cb5bb0791d7e11041d0664"
)
EXPECTED_STUDY_VERDICT_SHA256 = (
    "d29efc11acbd681359a4ddbe6eaacd9018ecc9808631b9d706269a6f7fc15fab"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "dbfd516e3db8ac616f0a3a48f5323fbf1c12405c178ee50c5792388d70b64742"
)


@dataclass
class Fresh3Config:
    source_run_dir: str = "runs/c13_lhbl_flat_48w"
    cohort_a_dir: str = "runs/c13_lhbl_fresh_192_211"
    cohort_b_dir: str = "runs/c13_lhbl_focal_fresh2"
    candidate_study_dir: str = "runs/c13_lhbl_candidate_study"
    checkpoint: str = (
        "runs/c13_lhbl_flat_48w/checkpoints/flat_mlp_iteration_04.pt"
    )
    out_dir: str = "runs/c13_lhbl_focal_fresh3"
    suite: str = "C_hard_maze"
    worlds: int = 12
    densities: str = "192,211"
    roadmap_k: int = 7
    seed: int = 1234
    fresh_seed_offset: int = 3_600_000
    max_world_retries: int = 200
    mode: str = "fhat"
    alpha: float = 0.50
    focal_w: float = 1.10
    budget_factor: float = 2.0
    required_win_fraction: float = 0.80
    bootstrap_replicates: int = 20_000
    bootstrap_seed: int = 513_337
    device: str = "auto"


def validate_selection(cfg: Fresh3Config) -> None:
    verdict_path = Path(cfg.candidate_study_dir) / "results" / "gate_verdict.json"
    if S.file_sha256(Path(D.__file__).resolve()) != EXPECTED_STUDY_RUNNER_SHA256:
        raise RuntimeError("candidate-study runner hash changed")
    if S.file_sha256(verdict_path) != EXPECTED_STUDY_VERDICT_SHA256:
        raise RuntimeError("candidate-study verdict hash changed")
    if S.file_sha256(Path(cfg.checkpoint)) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("iteration-4 checkpoint hash changed")
    selected = H._read_json(verdict_path).get("selected_candidate") or {}
    if (
        selected.get("variant") != "model"
        or int(selected.get("iteration", -1)) != 4
        or not math.isclose(float(selected.get("alpha", -1.0)), 0.50)
        or cfg.mode != "fhat"
        or not math.isclose(cfg.alpha, 0.50)
        or not math.isclose(cfg.focal_w, 1.10)
    ):
        raise RuntimeError("fresh3 differs from the selected fixed candidate")


def forbidden_seeds(cfg: Fresh3Config) -> set[int]:
    seeds = R.source_seed_set(Path(cfg.source_run_dir))
    for directory, filename in (
        (cfg.cohort_a_dir, "fresh_cohort.json"),
        (cfg.cohort_b_dir, "fresh2_cohort.json"),
    ):
        cohort = H._read_json(Path(directory) / "results" / filename)
        seeds.update(int(row["world_seed"]) for row in cohort["roadmaps"])
    return seeds


def build_verdict(
    summaries: Sequence[Mapping[str, Any]], densities: Sequence[int]
) -> Dict[str, Any]:
    expected = {int(value) for value in densities}
    observed = {int(row["density"]) for row in summaries}
    passed = observed == expected and all(bool(row["gate_pass"]) for row in summaries)
    return {
        "verdict": "fixed_candidate_fresh3_pass" if passed else "fixed_candidate_fresh3_fail",
        "gate_pass": bool(passed),
        "passing_densities": sorted(
            int(row["density"]) for row in summaries if bool(row["gate_pass"])
        ),
        "fixed_candidate": {
            "variant": "model",
            "model": "flat_mlp",
            "iteration": 4,
            "mode": "fhat",
            "alpha": 0.50,
            "focal_w": 1.10,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "authorization": (
            "run_matched_c7_six_suite_comparison"
            if passed
            else "revise_candidate_before_multi_suite_scaling"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-H fixed candidate fresh3")
    for field in Fresh3Config.__dataclass_fields__.values():
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=type(field.default),
            default=field.default,
        )
    return parser.parse_args()


def resolve_paths(cfg: Fresh3Config) -> None:
    script_dir = Path(__file__).resolve().parent
    for field_name in (
        "source_run_dir",
        "cohort_a_dir",
        "cohort_b_dir",
        "candidate_study_dir",
        "checkpoint",
        "out_dir",
    ):
        default = getattr(Fresh3Config, field_name)
        if getattr(cfg, field_name) == default:
            setattr(cfg, field_name, str(script_dir / default))


def main() -> None:
    cfg = Fresh3Config(**vars(parse_args()))
    resolve_paths(cfg)
    densities = R.parse_int_csv(cfg.densities)
    if densities != [192, 211] or cfg.worlds != 12:
        raise ValueError("fresh3 is locked to 12 paired worlds at 192 and 211")
    validate_selection(cfg)
    device = R.resolve_device(cfg.device)
    study_cfg = D.CandidateStudyConfig(
        source_run_dir=cfg.source_run_dir,
        iterations="4",
        device=cfg.device,
    )
    models, training_cfg, model_cfg, checkpoint_paths = D.load_checkpoints(
        study_cfg, device
    )
    if checkpoint_paths[4].resolve() != Path(cfg.checkpoint).resolve():
        raise RuntimeError("loaded checkpoint path differs from the locked candidate")
    generation_cfg = R.ReplicationConfig(
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
    bundles, cohort_rows, cohort_stats = R.generate_paired_bundles(
        generation_cfg, H.state_config(training_cfg), forbidden_seeds(cfg)
    )
    rows = F2.evaluate(cfg, models[4], bundles, device)
    summaries = F2.summarize(rows, cfg)
    verdict = build_verdict(summaries, densities)

    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    raw_path = C13.write_csv(result_dir / "fresh3_raw.csv", rows)
    summary_path = C13.write_csv(result_dir / "fresh3_summary.csv", summaries)
    verdict_path = C13.write_json(result_dir / "gate_verdict.json", verdict)
    cohort_path = C13.write_json(
        result_dir / "fresh3_cohort.json",
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
        "candidate_study_runner_sha256": S.file_sha256(Path(D.__file__).resolve()),
        "candidate_study_verdict_sha256": S.file_sha256(
            Path(cfg.candidate_study_dir) / "results" / "gate_verdict.json"
        ),
        "fixed_before_fresh_data": verdict["fixed_candidate"],
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
        raise RuntimeError("fresh3 provenance or safety verification failed")

    inputs = {
        "implementation": Path(__file__).resolve(),
        "candidate_study_implementation": Path(D.__file__).resolve(),
        "matched_focal_implementation": Path(F.__file__).resolve(),
        "training_implementation": Path(H.__file__).resolve(),
        "checkpoint": Path(cfg.checkpoint),
        "candidate_study_manifest": Path(cfg.candidate_study_dir) / "manifest.json",
        "candidate_study_verdict": Path(cfg.candidate_study_dir)
        / "results"
        / "gate_verdict.json",
        "cohort_a": Path(cfg.cohort_a_dir) / "results" / "fresh_cohort.json",
        "cohort_b": Path(cfg.cohort_b_dir) / "results" / "fresh2_cohort.json",
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
            "experiment": "C13-H fixed iteration-4 matched-control fresh3",
            "config": asdict(cfg),
            "source_model_config": asdict(model_cfg),
            "selection_status": "fixed_before_fresh3_seed_generation",
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
