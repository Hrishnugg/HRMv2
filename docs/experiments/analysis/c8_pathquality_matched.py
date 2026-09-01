"""Matched path quality: fixed blind U-Net vs Euclid anchor on jointly solved maps.

v11-review must-fix 7: the headline learned-vs-anchor path-cost evidence
(per-suite means 1.008-1.164) lacked a common-set definition, paired
comparator values, intervals, and n. This script computes, per suite at the
binding budget on the frozen fresh 50-map cohort (astar mode):

  - jointly solved n (learned AND euclid found=True)
  - learned mean suboptimality (arrival/optimal_arrival) on the joint set
  - euclid mean suboptimality on the joint set (expected exactly 1.0:
    admissible anchor => optimal arrival when solved; verified, not assumed)
  - paired mean difference with 10k map-level percentile bootstrap CI

Deterministic: numpy default_rng(20260723). Outputs c8_pathquality_matched
.md/.json next to this script.
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "hrm-cloud", "continuous_prm", "runs",
    "c8r_fresh_eval", "results", "continuous_prm_c8_eval_raw.csv"))
RNG = np.random.default_rng(20260723)
NBOOT = 10_000

BINDING = {"C_dyn_crossing": 150, "C_dyn_maze": 1800, "C_dyn_maze_dense": 2500,
           "C_dyn_rooms": 1300, "C_dyn_rooms_large": 600, "C_dyn_spiral": 2500}
LABELS = {"C_dyn_crossing": "Crossing", "C_dyn_maze": "Maze",
          "C_dyn_maze_dense": "Dense maze", "C_dyn_rooms": "Rooms",
          "C_dyn_rooms_large": "Large rooms", "C_dyn_spiral": "Spiral"}
ORDER = ["C_dyn_crossing", "C_dyn_maze", "C_dyn_maze_dense",
         "C_dyn_rooms", "C_dyn_rooms_large", "C_dyn_spiral"]
LEARNED, ANCHOR = "field_unet_blind", "euclid"

per = defaultdict(dict)  # (suite, provider) -> world -> (found, arrival, opt)
for r in csv.DictReader(open(RAW, newline="")):
    if r["mode"] != "astar" or int(r["budget"]) != BINDING[r["suite"]]:
        continue
    if r["provider"] not in (LEARNED, ANCHOR):
        continue
    per[(r["suite"], r["provider"])][int(r["world_index"])] = (
        r["found"] == "True", float(r["arrival"]), float(r["optimal_arrival"]))

out = {"raw": RAW, "learned": LEARNED, "anchor": ANCHOR, "suites": {}}
lines = ["# Matched path quality: blind U-Net vs Euclid (fresh cohort, binding budgets)",
         "", "| Suite | joint n | learned subopt | anchor subopt | paired diff [95% CI] |",
         "|---|---:|---:|---:|---|"]
anchor_optimal_everywhere = True
for suite in ORDER:
    lw, aw = per[(suite, LEARNED)], per[(suite, ANCHOR)]
    joint = sorted(w for w in lw if lw[w][0] and aw.get(w, (False,))[0])
    ls = np.array([lw[w][1] / lw[w][2] for w in joint])
    asub = np.array([aw[w][1] / aw[w][2] for w in joint])
    # verify anchor optimality on ALL its solved maps, not just joint
    a_solved = [w for w in aw if aw[w][0]]
    a_all = np.array([aw[w][1] / aw[w][2] for w in a_solved])
    a_opt = bool(np.allclose(a_all, 1.0, atol=1e-9))
    anchor_optimal_everywhere &= a_opt
    diffs = ls - asub
    idx = RNG.integers(0, len(diffs), size=(NBOOT, len(diffs)))
    boots = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out["suites"][suite] = {
        "n_joint": len(joint), "learned_subopt_mean": float(ls.mean()),
        "anchor_subopt_mean": float(asub.mean()),
        "anchor_optimal_on_all_solved": a_opt,
        "paired_diff_mean": float(diffs.mean()),
        "paired_diff_ci": [float(lo), float(hi)]}
    lines.append(
        f"| {LABELS[suite]} | {len(joint)} | {ls.mean():.3f} | {asub.mean():.3f} "
        f"| {diffs.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] |")
lines += ["", f"Anchor arrival == optimal arrival on every anchor-solved map "
          f"(all suites): {anchor_optimal_everywhere}"]
out["anchor_optimal_everywhere"] = anchor_optimal_everywhere

open(os.path.join(HERE, "c8_pathquality_matched.json"), "w").write(
    json.dumps(out, indent=1))
open(os.path.join(HERE, "c8_pathquality_matched_output.md"), "w",
     encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
