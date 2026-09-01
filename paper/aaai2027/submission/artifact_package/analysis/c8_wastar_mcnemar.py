"""Exact McNemar tests and effort-ratio CIs for the weighted-A* control.

Extends c8_wastar_analysis.py with the inference the v4 review demanded:
  1. Paired blind-vs-WA* success: per-suite discordant counts, exact
     two-sided binomial McNemar p, Benjamini-Hochberg q across the six suites
     (the same success-testing policy the paper applies elsewhere).
  2. Map-resampled bootstrap CIs (10k, seed 20260724 -- same stream as the
     WA* analysis) for the median Blind/WA* and WA*/anchor expansion ratios
     on jointly solved maps.
  3. Suboptimality means with per-arm n and bootstrap CIs.

Same inputs as c8_wastar_analysis.py; read-only.
"""
import csv
import json
import math
import os

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


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p for discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * cdf)


def bh(ps):
    order = np.argsort(ps)
    m = len(ps)
    qs = [0.0] * m
    prev = 1.0
    for rank_from_end, idx in enumerate(reversed(order)):
        rank = m - rank_from_end
        q = min(prev, ps[idx] * m / rank)
        qs[idx] = q
        prev = q
    return qs


def boot_ci(vals, stat, rng):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return None
    reps = [float(stat(vals[rng.integers(0, len(vals), len(vals))]))
            for _ in range(BOOT)]
    return (float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5)))


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

    rows = {}
    ps = []
    for suite in SUITES:
        worlds = sorted(w for (s, w) in wa if s == suite)
        b_only = sum(1 for w in worlds if bl[(suite, w)][0] and not wa[(suite, w)][0])
        w_only = sum(1 for w in worlds if wa[(suite, w)][0] and not bl[(suite, w)][0])
        p = exact_mcnemar(b_only, w_only)
        ps.append(p)
        rng = np.random.default_rng(SEED)
        bw_r = [bl[(suite, w)][1] / wa[(suite, w)][1] for w in worlds
                if bl[(suite, w)][0] and wa[(suite, w)][0]]
        wa_r = [wa[(suite, w)][1] / eu[(suite, w)][1] for w in worlds
                if wa[(suite, w)][0] and eu[(suite, w)][0]]
        wa_s = [wa[(suite, w)][2] / opt[(suite, w)] for w in worlds
                if wa[(suite, w)][0] and (suite, w) in opt]
        bl_s = [bl[(suite, w)][2] / opt[(suite, w)] for w in worlds
                if bl[(suite, w)][0] and (suite, w) in opt]
        rows[suite] = {
            "n_maps": len(worlds),
            "blind_only": b_only, "wastar_only": w_only, "p_exact": p,
            "blind_over_wastar_median": float(np.median(bw_r)) if bw_r else None,
            "blind_over_wastar_ci": boot_ci(bw_r, np.median, rng),
            "blind_over_wastar_n": len(bw_r),
            "wastar_over_anchor_median": float(np.median(wa_r)) if wa_r else None,
            "wastar_over_anchor_ci": boot_ci(wa_r, np.median, rng),
            "wastar_over_anchor_n": len(wa_r),
            "wastar_subopt_mean": float(np.mean(wa_s)) if wa_s else None,
            "wastar_subopt_ci": boot_ci(wa_s, np.mean, rng),
            "wastar_subopt_n": len(wa_s),
            "blind_subopt_mean": float(np.mean(bl_s)) if bl_s else None,
            "blind_subopt_ci": boot_ci(bl_s, np.mean, rng),
            "blind_subopt_n": len(bl_s),
            "w_h": tuned[suite]["w_h"],
        }
    qs = bh(ps)
    for suite, q in zip(SUITES, qs):
        rows[suite]["q_bh"] = q

    lines = [
        "# WA* vs blind U-Net: exact McNemar + effort-ratio CIs "
        "(confirmation cohort, binding budgets)",
        "",
        "| Suite | Discordant (learned-only / WA*-only) | exact p | BH q | "
        "Blind/WA* median ratio [CI] (n) | WA*/anchor median ratio [CI] (n) | "
        "Subopt WA* [CI] | Subopt learned [CI] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for suite in SUITES:
        r = rows[suite]
        def ci_s(ci, fmt="{:.3f}"):
            return "n/a" if ci is None else f"[{fmt.format(ci[0])},{fmt.format(ci[1])}]"
        lines.append(
            f"| {LABELS[suite]} | {r['blind_only']}/{r['wastar_only']} "
            f"| {r['p_exact']:.2e} | {r['q_bh']:.2e} "
            f"| {r['blind_over_wastar_median']:.3f} {ci_s(r['blind_over_wastar_ci'])} "
            f"({r['blind_over_wastar_n']}) "
            f"| {r['wastar_over_anchor_median']:.3f} {ci_s(r['wastar_over_anchor_ci'])} "
            f"({r['wastar_over_anchor_n']}) "
            f"| {r['wastar_subopt_mean']:.3f} {ci_s(r['wastar_subopt_ci'])} "
            f"| {r['blind_subopt_mean']:.3f} {ci_s(r['blind_subopt_ci'])} |")
    lines += [
        "",
        "Exact two-sided binomial McNemar on discordant maps; BH across the six "
        "suites. Ratio CIs: 10k map-resampled bootstraps of the median on "
        "jointly solved maps (seed 20260724). Suboptimality: mean arrival / "
        "optimal arrival on solved maps with known optimum, bootstrap CI of the mean.",
    ]
    with open(os.path.join(HERE, "c8_wastar_mcnemar_output.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "c8_wastar_mcnemar.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
