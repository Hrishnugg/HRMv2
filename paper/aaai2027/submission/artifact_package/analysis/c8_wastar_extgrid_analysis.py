"""Extended-grid WA* follow-up: dense maze at the re-selected w=7.

Compares the confirmation-cohort dense-maze rows at the extended-grid
selection (w=7) against (a) the frozen w=5 rows and (b) the learned blind
U-Net arm, with exact McNemar p-values on discordant maps and matched-solved
expansion ratios. Reported as an update to the success-tuned control.
"""
import csv
import json
import os
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "hrm-cloud",
                                     "continuous_prm", "runs"))
SUITE = "C_dyn_maze_dense"
BUDGET = 2500


def wastar_rows(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["suite"] == SUITE:
                out[int(r["world_index"])] = (
                    r["found"] in ("True", "true", "1"), int(r["expansions"]))
    return out


def blind_rows():
    out = {}
    p = os.path.join(RUNS, "c8r_fresh_eval", "results",
                     "continuous_prm_c8_eval_raw.csv")
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            if (r["suite"] == SUITE and r["mode"] == "astar"
                    and r["provider"] == "field_unet_blind"
                    and int(float(r["budget"])) == BUDGET):
                out[int(float(r["world_index"]))] = (
                    r["found"] in ("True", "true", "1"),
                    int(float(r["expansions"])))
    return out


def mcnemar(a, b):
    """Exact two-sided binomial on discordant pairs (a-only, b-only)."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def main():
    w5 = wastar_rows(os.path.join(RUNS, "c8r_wastar", "confirmation_raw.csv"))
    w7 = wastar_rows(os.path.join(RUNS, "c8r_wastar", "confirmation_ext_raw.csv"))
    bl = blind_rows()
    worlds = sorted(set(w5) & set(w7) & set(bl))
    rep = {"n_maps": len(worlds),
           "w5_success": sum(w5[w][0] for w in worlds),
           "w7_success": sum(w7[w][0] for w in worlds),
           "blind_success": sum(bl[w][0] for w in worlds)}

    a = sum(1 for w in worlds if w7[w][0] and not w5[w][0])
    b = sum(1 for w in worlds if w5[w][0] and not w7[w][0])
    rep["w7_vs_w5"] = dict(w7_only=a, w5_only=b, p_exact=mcnemar(a, b))
    a = sum(1 for w in worlds if w7[w][0] and not bl[w][0])
    b = sum(1 for w in worlds if bl[w][0] and not w7[w][0])
    rep["w7_vs_blind"] = dict(w7_only=a, blind_only=b, p_exact=mcnemar(a, b))

    joint = [w for w in worlds if w7[w][0] and bl[w][0]]
    ratios = np.array([bl[w][1] / w7[w][1] for w in joint])
    rng = np.random.default_rng(20260726)
    meds = [float(np.median(ratios[rng.integers(0, len(ratios), len(ratios))]))
            for _ in range(10_000)]
    rep["blind_over_w7_expansions"] = dict(
        n=len(joint), median=float(np.median(ratios)),
        ci=[float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))])

    out = os.path.join(HERE, "c8_wastar_extgrid.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
