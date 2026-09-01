"""Weighted-A* baseline vs anchor vs fixed blind U-Net on the confirmation cohort.

Joins runs/c8r_wastar/confirmation_raw.csv with the existing fresh-cohort raw
(euclid + field_unet_blind rows, astar mode, binding budgets; same worlds by
construction). Reports per suite: success for all three arms; paired
blind-minus-wastar success delta (10k map bootstraps); matched-solved median
expansion ratios (each learned/classical arm vs the anchor, and blind vs
wastar directly); and mean empirical suboptimality (arrival / optimal arrival)
on solved maps. Report-as-is per the frozen design note.
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "hrm-cloud",
                                     "continuous_prm", "runs"))
WASTAR = os.path.join(RUNS, "c8r_wastar", "confirmation_raw.csv")
FRESH = os.path.join(RUNS, "c8r_fresh_eval", "results", "continuous_prm_c8_eval_raw.csv")
TUNED = os.path.join(RUNS, "c8r_wastar", "tuned_weights.json")

SUITES = ["C_dyn_crossing", "C_dyn_maze", "C_dyn_maze_dense",
          "C_dyn_rooms", "C_dyn_rooms_large", "C_dyn_spiral"]
LABELS = {"C_dyn_crossing": "Crossing", "C_dyn_maze": "Maze",
          "C_dyn_maze_dense": "Dense maze", "C_dyn_rooms": "Rooms",
          "C_dyn_rooms_large": "Large rooms", "C_dyn_spiral": "Spiral"}
BINDING = {"C_dyn_crossing": 150, "C_dyn_maze": 1800, "C_dyn_maze_dense": 2500,
           "C_dyn_rooms": 1300, "C_dyn_rooms_large": 600, "C_dyn_spiral": 2500}
BOOT, SEED = 10_000, 20260724


def main():
    tuned = json.load(open(TUNED))
    wa = {}
    with open(WASTAR, newline="") as f:
        for r in csv.DictReader(f):
            wa[(r["suite"], int(r["world_index"]))] = (
                r["found"] in ("True", "true", "1"), float(r["expansions"]),
                float(r["arrival"]))
    eu, bl, opt = {}, {}, {}
    with open(FRESH, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("mode") != "astar":
                continue
            if int(float(r["budget"])) != BINDING[r["suite"]]:
                continue
            key = (r["suite"], int(float(r["world_index"])))
            rec = (r["found"] in ("True", "true", "1"), float(r["expansions"]),
                   float(r["arrival"]))
            if r["provider"] == "euclid":
                eu[key] = rec
            elif r["provider"] == "field_unet_blind":
                bl[key] = rec
            o = r.get("optimal_arrival", "")
            try:
                o = float(o)
                if np.isfinite(o) and o > 0:
                    opt[key] = o
            except (TypeError, ValueError):
                pass

    lines = ["# Weighted-A* baseline on the confirmation cohort", ""]
    lines.append("| Suite | $w_h$ | WA* succ | Anchor succ | Blind U-Net succ | "
                 "Blind$-$WA* dsucc [CI] | WA*/anchor ratio (n) | Blind/WA* ratio (n) | "
                 "WA* subopt | Blind subopt |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    out = {"suites": {}}
    for suite in SUITES:
        worlds = sorted(w for (s, w) in wa if s == suite)
        wsucc = np.mean([wa[(suite, w)][0] for w in worlds])
        esucc = np.mean([eu[(suite, w)][0] for w in worlds])
        bsucc = np.mean([bl[(suite, w)][0] for w in worlds])
        deltas = [float(bl[(suite, w)][0]) - float(wa[(suite, w)][0]) for w in worlds]
        rng = np.random.default_rng(SEED)
        bs = [float(np.mean([deltas[i] for i in rng.integers(0, len(deltas), len(deltas))]))
              for _ in range(BOOT)]
        ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        # matched ratios
        wa_r = [wa[(suite, w)][1] / eu[(suite, w)][1] for w in worlds
                if wa[(suite, w)][0] and eu[(suite, w)][0]]
        bw_r = [bl[(suite, w)][1] / wa[(suite, w)][1] for w in worlds
                if bl[(suite, w)][0] and wa[(suite, w)][0]]
        # suboptimality on solved maps with known optimum
        wa_s = [wa[(suite, w)][2] / opt[(suite, w)] for w in worlds
                if wa[(suite, w)][0] and (suite, w) in opt]
        bl_s = [bl[(suite, w)][2] / opt[(suite, w)] for w in worlds
                if bl[(suite, w)][0] and (suite, w) in opt]
        row = dict(w_h=tuned[suite]["w_h"], wastar_succ=float(wsucc),
                   euclid_succ=float(esucc), blind_succ=float(bsucc),
                   blind_minus_wastar=float(np.mean(deltas)), ci=ci,
                   wastar_ratio=float(np.median(wa_r)) if wa_r else None,
                   wastar_ratio_n=len(wa_r),
                   blind_over_wastar_ratio=float(np.median(bw_r)) if bw_r else None,
                   blind_over_wastar_n=len(bw_r),
                   wastar_subopt=float(np.mean(wa_s)) if wa_s else None,
                   blind_subopt=float(np.mean(bl_s)) if bl_s else None)
        out["suites"][suite] = row
        lines.append(
            f"| {LABELS[suite]} | {row['w_h']:g} | {wsucc:.2f} | {esucc:.2f} | {bsucc:.2f} | "
            f"{np.mean(deltas):+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] | "
            f"{(f'{row['wastar_ratio']:.3f}' if wa_r else 'n/a')} ({len(wa_r)}) | "
            f"{(f'{row['blind_over_wastar_ratio']:.3f}' if bw_r else 'n/a')} ({len(bw_r)}) | "
            f"{(f'{row['wastar_subopt']:.3f}' if wa_s else 'n/a')} | "
            f"{(f'{row['blind_subopt']:.3f}' if bl_s else 'n/a')} |")
    lines.append("")
    lines.append("Blind/WA* ratio < 1 means the learned heuristic expands fewer nodes than "
                 "tuned weighted A* on jointly solved maps. Suboptimality = arrival / optimal arrival.")
    md = os.path.join(HERE, "c8_wastar_output.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(HERE, "c8_wastar.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
