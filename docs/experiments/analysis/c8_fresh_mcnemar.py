"""Exact McNemar tests + BH correction for the fresh confirmation cohort.

Paired euclid vs field_unet_blind success at binding budgets (astar mode),
per suite; two-sided exact binomial test on discordant pairs; BH across the
six suites. Substantiates the main paper's 'corrected tests agree' sentence.
"""
import csv
import os
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(HERE, "..", "..", "..", "hrm-cloud",
                                    "continuous_prm", "runs", "c8r_fresh_eval",
                                    "results", "continuous_prm_c8_eval_raw.csv"))
BINDING = {"C_dyn_crossing": 150, "C_dyn_maze": 1800, "C_dyn_maze_dense": 2500,
           "C_dyn_rooms": 1300, "C_dyn_rooms_large": 600, "C_dyn_spiral": 2500}
LABELS = {"C_dyn_crossing": "Crossing", "C_dyn_maze": "Maze",
          "C_dyn_maze_dense": "Dense maze", "C_dyn_rooms": "Rooms",
          "C_dyn_rooms_large": "Large rooms", "C_dyn_spiral": "Spiral"}


def exact_mcnemar(b, c):
    """Two-sided exact binomial p for discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main():
    eu, bl = {}, {}
    with open(RAW, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("mode") != "astar":
                continue
            if int(float(r["budget"])) != BINDING[r["suite"]]:
                continue
            key = (r["suite"], int(float(r["world_index"])))
            found = r["found"] in ("True", "true", "1")
            if r["provider"] == "euclid":
                eu[key] = found
            elif r["provider"] == "field_unet_blind":
                bl[key] = found
    rows = []
    for suite in BINDING:
        worlds = sorted(w for (s, w) in eu if s == suite)
        b = sum(1 for w in worlds if bl[(suite, w)] and not eu[(suite, w)])
        c = sum(1 for w in worlds if eu[(suite, w)] and not bl[(suite, w)])
        p = exact_mcnemar(b, c)
        rows.append([suite, b, c, p])
    # BH correction across the six suites
    order = sorted(range(len(rows)), key=lambda i: rows[i][3])
    m = len(rows)
    qs = [0.0] * m
    prev = 1.0
    for rank_pos in range(m - 1, -1, -1):
        i = order[rank_pos]
        q = min(prev, rows[i][3] * m / (rank_pos + 1))
        qs[i] = q
        prev = q
    print("| Suite | blind-only | euclid-only | exact p | BH q |")
    print("|---|---|---|---|---|")
    for (suite, b, c, p), q in zip(rows, qs):
        print(f"| {LABELS[suite]} | {b} | {c} | {p:.2e} | {q:.2e} |")


if __name__ == "__main__":
    main()
