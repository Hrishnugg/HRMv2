#!/usr/bin/env python3
"""Extended-grid weighted-A* check (frozen design 2026-07-26).

Runs ONLY the two new weights {7, 10} on the development cohort, merges with
the frozen development rows, re-applies the identical success-tuned selection
rule over the extended grid, and (only if a suite's selection changes) runs
the changed weight once on the frozen confirmation cohort. Read-only with
respect to all frozen artifacts; new rows are written beside them.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import continuous_prm_c8_dynamic_maps as M8MAPS
import continuous_prm_c8_dynamics_compare as M8C
import continuous_prm_dynamic_providers as DP
import continuous_prm_spacetime as ST

HERE = Path(__file__).parent
OUT = HERE / "runs" / "c8r_wastar"
SUITES = ["C_dyn_crossing", "C_dyn_maze", "C_dyn_maze_dense",
          "C_dyn_rooms", "C_dyn_rooms_large", "C_dyn_spiral"]
BINDING = {"C_dyn_crossing": 150, "C_dyn_maze": 1800, "C_dyn_maze_dense": 2500,
           "C_dyn_rooms": 1300, "C_dyn_rooms_large": 600, "C_dyn_spiral": 2500}
OLD_WEIGHTS = [1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
NEW_WEIGHTS = [7.0, 10.0]
COLS = ["suite", "world_index", "w_h", "budget", "found", "expansions", "arrival"]


def run_weights(seed: int, n_worlds: int, weights, tag: str) -> Path:
    cfg = M8C.C8Config(seed=int(seed), eval_worlds=int(n_worlds))
    cfg = M8C.apply_scale_preset(cfg)
    cfg.eval_worlds = int(n_worlds)
    anchor = DP.EuclidTimeProvider()
    rows = []
    for suite_idx, suite in enumerate(SUITES):
        if isinstance(weights, dict) and suite not in weights:
            continue
        params = M8MAPS.dynamics_params(suite)
        v_agent, dt, t_max = params["v_agent"], params["dt"], int(params["t_max"])
        budget = BINDING[suite]
        for wi, world, dyn, rm in M8C.iter_dynamic_worlds(suite, suite_idx, cfg, n_worlds):
            h0 = anchor.h_table(world, rm, dyn, v_agent, dt, t_max, goal_idx=1)
            ws = weights[suite] if isinstance(weights, dict) else weights
            for w_h in ws:
                res = ST.space_time_astar_prm(rm.adj, rm.points, dyn,
                                              h0 * float(w_h), int(budget),
                                              v_agent, dt, t_max, 0, 1)
                rows.append(dict(suite=suite, world_index=wi, w_h=float(w_h),
                                 budget=int(budget), found=bool(res["found"]),
                                 expansions=int(res["expansions"]),
                                 arrival=int(res["arrival"])))
        print(f"[extgrid] {tag} {suite}: done", flush=True)
    out_csv = OUT / f"{tag}_raw.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_csv


def main():
    M8MAPS.install_c8_dynamic_maps()
    # New-weight development rows (idempotent: reuse if already present).
    if not (OUT / "development_ext_raw.csv").exists():
        run_weights(1234, 20, NEW_WEIGHTS, "development_ext")
    # Merge with frozen development rows and re-select.
    succ = {}
    for path in (OUT / "development_raw.csv", OUT / "development_ext_raw.csv"):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["suite"], float(r["w_h"]))
                succ.setdefault(key, []).append(r["found"] in ("True", "true", "1"))
    all_w = OLD_WEIGHTS + NEW_WEIGHTS
    old_tuned = json.load(open(OUT / "tuned_weights.json"))
    changed = {}
    report = {}
    for suite in SUITES:
        best = None
        for w_h in all_w:
            sc = float(np.mean(succ[(suite, w_h)])) if (suite, w_h) in succ else -1.0
            if best is None or sc > best[1] + 1e-12:
                best = (w_h, sc)
        report[suite] = dict(extended_selection=best[0], dev_success=best[1],
                             frozen_selection=old_tuned[suite]["w_h"])
        if abs(best[0] - float(old_tuned[suite]["w_h"])) > 1e-9:
            changed[suite] = best[0]
        print(f"[extgrid] {suite}: frozen w={old_tuned[suite]['w_h']:g} -> "
              f"extended w={best[0]:g} (dev succ {best[1]:.2f})"
              f"{'  CHANGED' if suite in changed else ''}", flush=True)

    if changed:
        conf_csv = run_weights(999999, 50, {s: [w] for s, w in changed.items()},
                               "confirmation_ext")
        report["confirmation_ext_csv"] = str(conf_csv)
    with open(OUT / "extgrid_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
