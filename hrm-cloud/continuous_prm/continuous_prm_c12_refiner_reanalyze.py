"""Independent regeneration check for the canonical C12-B derived results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

import continuous_prm_c12_refiner as R
import continuous_prm_c12_refiner_pipeline as P


def _semantic_equal(left, right) -> bool:
    """Compare regenerated JSON while tolerating CSV round-trip ulps."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _semantic_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _semantic_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def run(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    result_dir = out_dir / "results"
    state_path = result_dir / "c12b_state_metrics.csv"
    eval_path = result_dir / "c12b_eval_raw.csv"
    canonical_summary = result_dir / "c12b_summary.json"
    canonical_significance = result_dir / "c12b_significance.csv"
    for path in (state_path, eval_path, canonical_summary, canonical_significance):
        if not path.exists():
            raise RuntimeError(f"missing canonical C12-B artifact: {path}")

    state_rows = pd.read_csv(state_path).to_dict(orient="records")
    eval_rows = pd.read_csv(eval_path).to_dict(orient="records")
    summary, significance = P.analyze_results(state_rows, eval_rows, scale="full")

    check_dir = result_dir / "reanalysis"
    candidate_summary = check_dir / "c12b_summary.json"
    candidate_significance = check_dir / "c12b_significance.csv"
    R._write_json_atomic(candidate_summary, summary)
    R._write_csv_atomic(candidate_significance, significance)

    existing_summary_obj = json.loads(canonical_summary.read_text(encoding="utf-8"))
    candidate_summary_obj = json.loads(candidate_summary.read_text(encoding="utf-8"))
    summary_equal = _semantic_equal(existing_summary_obj, candidate_summary_obj)

    existing_sig = pd.read_csv(canonical_significance).sort_index(axis=1)
    candidate_sig = pd.read_csv(candidate_significance).sort_index(axis=1)
    significance_equal = True
    try:
        pd.testing.assert_frame_equal(
            existing_sig,
            candidate_sig,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError:
        significance_equal = False

    verification = {
        "schema_version": R.SCHEMA_VERSION,
        "status": "pass" if summary_equal and significance_equal else "fail",
        "source_rows": {"state": len(state_rows), "evaluation": len(eval_rows)},
        "source_sha256": {
            "state": P._sha256(state_path),
            "evaluation": P._sha256(eval_path),
        },
        "summary": {
            "semantic_equal": summary_equal,
            "canonical_sha256": P._sha256(canonical_summary),
            "regenerated_sha256": P._sha256(candidate_summary),
        },
        "significance": {
            "table_equal": significance_equal,
            "canonical_rows": len(existing_sig),
            "regenerated_rows": len(candidate_sig),
            "canonical_sha256": P._sha256(canonical_significance),
            "regenerated_sha256": P._sha256(candidate_significance),
        },
    }
    R._write_json_atomic(check_dir / "verification.json", verification)
    if verification["status"] != "pass":
        raise AssertionError(f"C12-B reanalysis mismatch: {verification}")
    return verification

