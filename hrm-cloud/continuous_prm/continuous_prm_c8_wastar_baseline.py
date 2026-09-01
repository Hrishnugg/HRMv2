#!/usr/bin/env python3
"""Weighted-A* classical baseline for the C8 dynamic substrate.

Standalone, read-only reuse of the frozen harness (no changes to the C8
module): regenerates the development (seed 1234, 20/suite) and confirmation
(seed 999999, 50/suite) cohorts via the canonical world iterator, runs
space-time A* with an inflated anchor h_w = w_h * h0 at the canonical binding
budgets, tunes w_h per suite on development (highest success, ties -> smaller
w_h; rule frozen in the design note), and evaluates the tuned weight once on
the confirmation cohort.

Design: docs/experiments/continuous/c08/design/2026-07-24-c8-wastar-baseline.md
Usage:  python continuous_prm_c8_wastar_baseline.py [--phase dev|conf|both]
Output: runs/c8r_wastar/{development,confirmation}_raw.csv + tuned_weights.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import continuous_prm_c8_dynamics_compare as M8C
import continuous_prm_c8_dynamic_maps as M8MAPS
import continuous_prm_dynamic_providers as DP
import continuous_prm_spacetime as ST

SUITES = ["C_dyn_maze", "C_dyn_rooms", "C_dyn_spiral",
          "C_dyn_maze_dense", "C_dyn_crossing", "C_dyn_rooms_large"]
BINDING = {"C_dyn_crossing": 150, "C_dyn_maze": 1800, "C_dyn_maze_dense": 2500,
           "C_dyn_rooms": 1300, "C_dyn_rooms_large": 600, "C_dyn_spiral": 2500}
WEIGHTS = [1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
OUT = Path("runs/c8r_wastar")
COLS = ["suite", "world_index", "w_h", "budget", "found", "expansions", "arrival"]


def run_cohort(seed: int, n_worlds: int, weights, tag: str) -> Path:
    cfg = M8C.C8Config(seed=int(seed), eval_worlds=int(n_worlds))
    cfg = M8C.apply_scale_preset(cfg)
    cfg.eval_worlds = int(n_worlds)  # preset must not override the cohort size
    rows = []
    for suite_idx, suite in enumerate(SUITES):
        params = M8MAPS.dynamics_params(suite)
        v_agent, dt, t_max = float(params["v_agent"]), float(params["dt"]), int(params["t_max"])
        budget = BINDING[suite]
        anchor = DP.EuclidTimeProvider()
        for wi, world, dyn, rm in M8C.iter_dynamic_worlds(suite, suite_idx, cfg, n_worlds):
            h0 = anchor.h_table(world, rm, dyn, v_agent, dt, t_max, goal_idx=1)
            for w_h in weights[suite] if isinstance(weights, dict) else weights:
                res = ST.space_time_astar_prm(
                    rm.adj, rm.points, dyn, h0 * float(w_h), int(budget),
                    v_agent, dt, t_max, 0, 1)
                rows.append(dict(suite=suite, world_index=wi, w_h=float(w_h),
                                 budget=int(budget), found=bool(res["found"]),
                                 expansions=int(res["expansions"]),
                                 arrival=int(res["arrival"])))
        print(f"[wastar] {tag} {suite}: done ({n_worlds} worlds)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"{tag}_raw.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[wastar] wrote {out_csv} ({len(rows)} rows)", flush=True)
    return out_csv


def tune(dev_csv: Path) -> dict:
    """Frozen rule: per suite, highest success; ties -> smallest w_h."""
    by = {}
    with open(dev_csv, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["suite"], float(r["w_h"]))
            by.setdefault(key, []).append(r["found"] in ("True", "true", "1"))
    tuned = {}
    for suite in SUITES:
        best = None
        for w_h in WEIGHTS:
            succ = float(np.mean(by[(suite, w_h)])) if (suite, w_h) in by else float("nan")
            if best is None or succ > best[1] + 1e-12:
                best = (w_h, succ)
        tuned[suite] = {"w_h": best[0], "dev_success": best[1]}
        print(f"[wastar] tuned {suite}: w_h={best[0]} (dev success {best[1]:.2f})", flush=True)
    with open(OUT / "tuned_weights.json", "w") as f:
        json.dump(tuned, f, indent=1)
    return tuned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["dev", "conf", "both"], default="both")
    args = p.parse_args()
    M8MAPS.install_c8_dynamic_maps()
    if args.phase in ("dev", "both"):
        dev_csv = run_cohort(1234, 20, WEIGHTS, "development")
        tune(dev_csv)
    if args.phase in ("conf", "both"):
        tuned = json.load(open(OUT / "tuned_weights.json"))
        weights = {s: [tuned[s]["w_h"]] for s in SUITES}
        run_cohort(999999, 50, weights, "confirmation")


if __name__ == "__main__":
    main()
