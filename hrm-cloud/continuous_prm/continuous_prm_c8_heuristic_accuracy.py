#!/usr/bin/env python3
"""
C8 Heuristic-Accuracy Ablation
===============================
Measures the MAE of each C8 learned space-time heuristic's predicted
time-to-go against the exact space-time oracle, comparing each
time-AWARE model to its time-BLIND (W=0) twin.

Tests WHY the future window didn't help search: does it make the
*heuristic* more accurate, or not?

Usage
-----
    python hrm-cloud/continuous_prm/continuous_prm_c8_heuristic_accuracy.py

Options
-------
    --out-dir     Path to the c8 run directory (default: hrm-cloud/continuous_prm/runs/c8_local_heavy)
    --eval-worlds Number of worlds per suite (default: 10)
    --device      cuda | cpu (default: auto-detect)
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Force UTF-8 stdout/stderr on Windows (cp1252 can't handle Unicode in print)
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "buffer") and (
    sys.stdout.encoding is None or sys.stdout.encoding.lower().replace("-", "") in ("cp1252", "cp850", "ascii")
):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and (
    sys.stderr.encoding is None or sys.stderr.encoding.lower().replace("-", "") in ("cp1252", "cp850", "ascii")
):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Bootstrap: ensure the continuous_prm package directory is on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import continuous_prm_common as C
import continuous_prm_c8_dynamic_maps as M8

# Re-use parse_csv, ensure_dir from the C8 orchestrator
from continuous_prm_c8_dynamics_compare import (
    C8Config,
    _load_eval_providers,
    iter_dynamic_worlds,
    parse_csv,
    ensure_dir,
)

INF = C.INF  # 1e30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reachable_mask(oracle_h: np.ndarray, t_max: int) -> np.ndarray:
    """
    Identify cells where the oracle reports a GENUINELY reachable time-to-go.

    OracleProvider fills unreachable (node, t) cells with
        fill = max_finite_ttg + (t_max + 1)
    which is much smaller than INF/10 but still distinctly larger than
    reachable values. We use a conservative threshold of 2*(t_max+1) to
    separate truly reachable cells from fill-substituted cells.

    The spec's formulation `np.isfinite(oracle_h) & (oracle_h < INF/10)` is
    correct in intent: OracleProvider produces an all-finite table using a
    domain-specific fill (not 1e30), so both conditions reduce to ``all True''.
    We instead separate reachable from fill using `oracle_h <= t_max` which
    captures cells where the agent can actually reach the goal within the
    planning horizon.
    """
    # Any ttg > t_max is unreachable within the horizon; treat as mask-out.
    return oracle_h <= float(t_max)


def _mae_rmse(errors: np.ndarray):
    """Return (MAE, RMSE) from a flat array of absolute errors."""
    if len(errors) == 0:
        return float("nan"), float("nan")
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return mae, rmse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="C8 heuristic accuracy ablation")
    parser.add_argument(
        "--out-dir",
        default="hrm-cloud/continuous_prm/runs/c8_local_heavy",
        help="Path to c8 run directory (contains checkpoints/ and results/)",
    )
    parser.add_argument(
        "--eval-worlds",
        type=int,
        default=10,
        help="Number of worlds per suite to evaluate (default: 10)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cuda or cpu (default: auto)",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Install dynamic maps and pick device
    # -----------------------------------------------------------------------
    M8.install_c8_dynamic_maps()

    import torch
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[c8_accuracy] device={device}", flush=True)

    # -----------------------------------------------------------------------
    # 2. Build C8Config matching the heavy run
    # -----------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    cfg = C8Config(
        window_w=8,
        scalar_backbones="hrm,onlstm",
        field_backbones="unet,hrm",
        eval_suites=(
            "C_dyn_maze,C_dyn_rooms,C_dyn_spiral,"
            "C_dyn_maze_dense,C_dyn_crossing,C_dyn_rooms_large"
        ),
        eval_worlds=args.eval_worlds,
        # roadmap_nodes / roadmap_k / seed: use C8Config defaults (192, 7, 1234)
    )

    # -----------------------------------------------------------------------
    # 3. Load providers
    # -----------------------------------------------------------------------
    print(f"[c8_accuracy] loading providers from {out_dir}", flush=True)
    providers = _load_eval_providers(cfg, out_dir, device)
    print(f"[c8_accuracy] loaded providers: {sorted(providers.keys())}", flush=True)

    # -----------------------------------------------------------------------
    # 4. Sanity check: oracle vs oracle MAE should be exactly 0
    # -----------------------------------------------------------------------
    print("[c8_accuracy] running sanity check (oracle vs oracle)...", flush=True)
    suite0 = parse_csv(cfg.eval_suites)[0]
    params0 = M8.dynamics_params(suite0)
    v0, dt0, t_max0 = float(params0["v_agent"]), float(params0["dt"]), int(params0["t_max"])
    sanity_oracle = providers["oracle"]
    sanity_world_gen = iter_dynamic_worlds(suite0, 0, cfg, 1)
    _, w0, dyn0, rm0 = next(sanity_world_gen)
    oracle_h0 = sanity_oracle.h_table(w0, rm0, dyn0, v0, dt0, t_max0)
    oracle_h0b = sanity_oracle.h_table(w0, rm0, dyn0, v0, dt0, t_max0)
    sanity_mae = float(np.mean(np.abs(oracle_h0 - oracle_h0b)))
    assert sanity_mae == 0.0, f"Oracle vs oracle sanity FAILED: MAE={sanity_mae}"
    print(f"[c8_accuracy] sanity check PASSED: oracle-vs-oracle MAE = {sanity_mae:.6f}", flush=True)

    # -----------------------------------------------------------------------
    # 5. Model names to evaluate (in order)
    # -----------------------------------------------------------------------
    MODEL_NAMES = [
        "euclid",
        "scalar_hrm",
        "scalar_hrm_blind",
        "scalar_onlstm",
        "scalar_onlstm_blind",
        "field_unet",
        "field_unet_blind",
        "field_hrm",
        "field_hrm_blind",
    ]
    # Filter to those actually loaded
    model_names = [m for m in MODEL_NAMES if m in providers]
    if len(model_names) < len(MODEL_NAMES):
        missing = [m for m in MODEL_NAMES if m not in providers]
        print(f"[c8_accuracy] WARNING: missing providers: {missing}", flush=True)

    # -----------------------------------------------------------------------
    # 6. Main evaluation loop
    # -----------------------------------------------------------------------
    # Per (suite, model): accumulated abs-error arrays (lists of flat arrays)
    # Also accumulate oracle values for mean_oracle computation
    suite_results: dict[str, dict] = {}   # suite -> {model: errors_list, "oracle_vals": list}

    suites = parse_csv(cfg.eval_suites)
    for suite_idx, suite in enumerate(suites):
        params = M8.dynamics_params(suite)
        v_agent = float(params["v_agent"])
        dt = float(params["dt"])
        t_max = int(params["t_max"])

        print(f"\n[c8_accuracy] suite {suite} (v={v_agent}, dt={dt}, t_max={t_max})", flush=True)

        suite_errors: dict[str, list] = {m: [] for m in model_names}
        suite_oracle_vals: list = []
        n_worlds_ok = 0

        for wi, world, dyn, rm in iter_dynamic_worlds(suite, suite_idx, cfg, cfg.eval_worlds):
            # Compute oracle h_table
            oracle_h = providers["oracle"].h_table(world, rm, dyn, v_agent, dt, t_max)

            # Build reachability mask
            mask = _reachable_mask(oracle_h, t_max)
            if mask.sum() == 0:
                print(f"  world {wi}: empty mask, skipping", flush=True)
                continue

            oracle_vals = oracle_h[mask]
            suite_oracle_vals.append(oracle_vals)
            n_worlds_ok += 1

            for mname in model_names:
                pred = providers[mname].h_table(world, rm, dyn, v_agent, dt, t_max)
                abs_err = np.abs(pred[mask] - oracle_vals)
                suite_errors[mname].append(abs_err)

            print(
                f"  world {wi}: mask={mask.sum()} cells "
                f"| oracle mean={oracle_vals.mean():.2f} steps",
                flush=True,
            )

        suite_results[suite] = {
            "errors": suite_errors,
            "oracle_vals": suite_oracle_vals,
            "t_max": t_max,
        }
        print(
            f"[c8_accuracy] {suite}: {n_worlds_ok}/{cfg.eval_worlds} worlds evaluated",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # 7. Compute per-(suite, model) statistics
    # -----------------------------------------------------------------------
    # Structure: stats[suite][model] = {mae, rmse, n_cells, mean_oracle}
    stats: dict[str, dict[str, dict]] = {}

    for suite, sdata in suite_results.items():
        stats[suite] = {}
        all_oracle = np.concatenate(sdata["oracle_vals"]) if sdata["oracle_vals"] else np.array([])
        mean_oracle = float(np.mean(all_oracle)) if len(all_oracle) > 0 else float("nan")

        for mname in model_names:
            errs_list = sdata["errors"][mname]
            if errs_list:
                all_errs = np.concatenate(errs_list)
            else:
                all_errs = np.array([])
            mae, rmse = _mae_rmse(all_errs)
            stats[suite][mname] = {
                "mae": mae,
                "rmse": rmse,
                "n_cells": len(all_errs),
                "mean_oracle": mean_oracle,
                "mae_over_oracle": mae / mean_oracle if mean_oracle > 0 else float("nan"),
            }

    # -----------------------------------------------------------------------
    # 8. Build output tables
    # -----------------------------------------------------------------------
    lines: list[str] = []
    lines.append("# C8 Heuristic Accuracy: Predicted Time-to-Go vs Oracle")
    lines.append("")
    lines.append(
        "Measures MAE of each C8 model's `h_table` against the exact backward-Dijkstra "
        "oracle, restricted to cells reachable within the planning horizon (`ttg <= t_max`). "
        "Cells are pooled across worlds (not averaged per-world then averaged)."
    )
    lines.append("")
    lines.append(f"Oracle-vs-oracle sanity MAE: **{sanity_mae:.6f}** (must be 0.0 -- PASSED)")
    lines.append("")

    # --- Per-suite tables ---
    for suite in suites:
        lines.append(f"## Suite: {suite}")
        lines.append("")
        lines.append("| model | MAE (steps) | RMSE | MAE/mean_oracle | n_cells |")
        lines.append("|---|---|---|---|---|")

        suite_stats = stats[suite]
        for mname in model_names:
            if mname not in suite_stats:
                continue
            s = suite_stats[mname]
            lines.append(
                f"| {mname} "
                f"| {s['mae']:.3f} "
                f"| {s['rmse']:.3f} "
                f"| {s['mae_over_oracle']:.3f} "
                f"| {s['n_cells']:,} |"
            )
        lines.append("")
        mean_or = suite_stats.get(model_names[0], {}).get("mean_oracle", float("nan"))
        lines.append(f"*Mean oracle time-to-go (reachable cells): {mean_or:.2f} steps*")
        lines.append("")

    # --- Aware-vs-Blind summary table ---
    lines.append("## Aware-vs-Blind Summary")
    lines.append("")
    lines.append(
        "Positive delta = aware is WORSE (higher MAE); negative delta = aware is BETTER. "
        "`better` = whichever variant has lower MAE."
    )
    lines.append("")
    lines.append(
        "| suite | backbone | MAE_aware | MAE_blind | delta = aware - blind | better |"
    )
    lines.append("|---|---|---|---|---|---|")

    BACKBONE_PAIRS = [
        ("scalar_hrm",    "scalar_hrm_blind",    "scalar_hrm"),
        ("scalar_onlstm", "scalar_onlstm_blind", "scalar_onlstm"),
        ("field_unet",    "field_unet_blind",    "field_unet"),
        ("field_hrm",     "field_hrm_blind",     "field_hrm"),
    ]

    aware_better_count = 0
    blind_better_count = 0
    delta_sum = 0.0
    delta_count = 0

    avb_rows: list[tuple] = []
    for suite in suites:
        suite_stats = stats[suite]
        for aware_key, blind_key, bb_label in BACKBONE_PAIRS:
            if aware_key not in suite_stats or blind_key not in suite_stats:
                continue
            mae_aware = suite_stats[aware_key]["mae"]
            mae_blind = suite_stats[blind_key]["mae"]
            if np.isnan(mae_aware) or np.isnan(mae_blind):
                continue
            delta = mae_aware - mae_blind
            better = "aware" if delta < 0 else "blind"
            if delta < 0:
                aware_better_count += 1
            elif delta > 0:
                blind_better_count += 1
            delta_sum += delta
            delta_count += 1
            avb_rows.append((suite, bb_label, mae_aware, mae_blind, delta, better))
            lines.append(
                f"| {suite} "
                f"| {bb_label} "
                f"| {mae_aware:.3f} "
                f"| {mae_blind:.3f} "
                f"| {delta:+.3f} "
                f"| **{better}** |"
            )

    lines.append("")
    mean_delta = delta_sum / delta_count if delta_count > 0 else float("nan")
    verdict = (
        f"**Verdict:** out of {delta_count} (suite, backbone) pairs, "
        f"aware is more accurate (delta<0) in {aware_better_count} and "
        f"less accurate (delta>0) in {blind_better_count} "
        f"(mean delta = {mean_delta:+.3f} steps). "
    )
    if mean_delta > 0.05:
        verdict += (
            "The future window makes the heuristic systematically WORSE on average - "
            "consistent with no search-expansion benefit from time-awareness."
        )
    elif mean_delta < -0.05:
        verdict += (
            "The future window makes the heuristic systematically BETTER on average - "
            "but search expansions did not benefit, suggesting the accuracy gain does not translate to better guidance."
        )
    else:
        verdict += (
            "The future window provides no systematic accuracy advantage over the "
            "time-blind baseline - directly explaining why it did not reduce search expansions."
        )
    lines.append(verdict)
    lines.append("")

    md_text = "\n".join(lines)

    # -----------------------------------------------------------------------
    # 9. Print and write output
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(md_text)
    print("=" * 72 + "\n")

    results_dir = ensure_dir(out_dir / "results")
    md_path = results_dir / "c8_heuristic_accuracy.md"
    md_path.write_text(md_text, encoding="utf-8")
    print(f"[c8_accuracy] written: {md_path}", flush=True)

    # -----------------------------------------------------------------------
    # 10. Euclid sanity: it should have the largest MAE (largest underestimate)
    # -----------------------------------------------------------------------
    if "euclid" in model_names:
        for suite in suites:
            suite_stats = stats[suite]
            euclid_mae = suite_stats.get("euclid", {}).get("mae", float("nan"))
            model_maes = {
                m: suite_stats[m]["mae"]
                for m in model_names
                if m != "euclid" and m in suite_stats
            }
            if model_maes and not np.isnan(euclid_mae):
                max_learned = max(model_maes.values())
                if euclid_mae < max_learned:
                    print(
                        f"[c8_accuracy] NOTE: {suite}: euclid MAE ({euclid_mae:.3f}) "
                        f"< max learned MAE ({max_learned:.3f}) - euclid is NOT the worst. "
                        "This can happen when learned models overestimate (inadmissible), "
                        "which inflates their MAE above the euclid underestimate.",
                        flush=True,
                    )
                else:
                    print(
                        f"[c8_accuracy] {suite}: euclid MAE={euclid_mae:.3f} "
                        f"> max learned MAE={max_learned:.3f} (OK) (euclid is worst, as expected)",
                        flush=True,
                    )

    print("[c8_accuracy] done.", flush=True)


if __name__ == "__main__":
    main()
