"""C8-S Amendment 2 sensitivity analysis (post-hoc, success-only, descriptive).

Floor cells (dense@all sizes, spiral@2048): success at the sensitivity
binding (senssel_*.json, frozen rule) from capped sens_eval rows by
thresholding; readouts = learned-anchor and learned-WA*(w*) paired deltas
with 10k map-bootstrap CIs and exact McNemar p (descriptive; no BH family).
Ceiling cells (rooms-large@1024/2048): sensitivity binding = largest
original grid budget with recorded dev anchor success <= 0.70 (calibration
reports); existing eval rows thresholded down (learned_cpu arm).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
RUN = ROOT / "runs" / "c8s2_scale"
OUT_JSON = HERE.parent / "c8s2_sens_analysis.json"
OUT_MD = HERE.parent / "c8s2_sens_analysis_output.md"

FLOOR = [(192, "C_dyn_maze_dense"), (512, "C_dyn_maze_dense"),
         (1024, "C_dyn_maze_dense"), (2048, "C_dyn_maze_dense"),
         (2048, "C_dyn_spiral")]
CEILING = [(1024, "C_dyn_rooms_large"), (2048, "C_dyn_rooms_large")]
NBOOT = 10_000
RNG = np.random.default_rng(20260728)


def rows(path):
    with open(path, newline="") as f:
        rs = list(csv.DictReader(f))
    for r in rs:
        r["found"] = r["found"] == "True"
        r["expansions"] = int(r["expansions"])
        r["world_index"] = int(r["world_index"])
    return rs


def solved(rs, budget):
    return {r["world_index"]: (r["found"] and r["expansions"] <= budget)
            for r in rs}


def paired(a, b):
    ws = sorted(a)
    da = np.array([float(a[w]) for w in ws])
    db = np.array([float(b[w]) for w in ws])
    d = da - db
    idx = RNG.integers(0, len(d), size=(NBOOT, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    x = int(np.sum(da > db))
    y = int(np.sum(db > da))
    p = binomtest(min(x, y), x + y, 0.5).pvalue if x + y else 1.0
    return dict(delta=float(d.mean()), ci=[float(lo), float(hi)],
                discordant=[x, y], p=float(p),
                succ_a=float(da.mean()), succ_b=float(db.mean()))


def main():
    res = {"floor": {}, "ceiling": {}}
    md = ["# C8-S Amendment 2 sensitivity output (post hoc, descriptive)",
          ""]
    for n, suite in FLOOR:
        sel = json.loads((RUN / f"senssel_{n}_{suite}.json").read_text())
        b = int(sel["binding"])
        eu = solved(rows(RUN / f"sens_eval_euclid_{n}_{suite}.csv"), b)
        wa = solved(rows(RUN / f"sens_eval_wastar_sel_{n}_{suite}.csv"), b)
        le = solved(rows(RUN / f"sens_eval_learned_cpu_{n}_{suite}.csv"), b)
        va = paired(le, eu)
        vw = paired(le, wa)
        res["floor"][f"{n}_{suite}"] = dict(
            binding=b, status=sel["status"], w_star=sel["w_star"],
            vs_anchor=va, vs_wastar=vw)
        s = suite.replace("C_dyn_", "")
        md += [f"- {s}@{n} (binding {b}, {sel['status']}, w*="
               f"{sel['w_star']:g}): anchor {va['succ_b']:.2f} learned "
               f"{va['succ_a']:.2f} d={va['delta']:+.2f} "
               f"[{va['ci'][0]:+.2f},{va['ci'][1]:+.2f}] "
               f"disc {va['discordant'][0]}/{va['discordant'][1]} "
               f"p={va['p']:.4g}; WA* {vw['succ_b']:.2f} "
               f"d={vw['delta']:+.2f} [{vw['ci'][0]:+.2f},"
               f"{vw['ci'][1]:+.2f}] p={vw['p']:.4g}"]
    for n, suite in CEILING:
        cal = json.loads((RUN / f"calib_{n}_{suite}.json").read_text())
        cands = [int(k) for k, v in cal["rates"].items() if v <= 0.70]
        if not cands:
            md += [f"- rooms_large@{n}: no grid budget with dev anchor "
                   f"<= 0.70 (floor rate "
                   f"{min(cal['rates'].values()):.2f}); skipped"]
            res["ceiling"][f"{n}_{suite}"] = dict(binding=None)
            continue
        b = max(cands)
        ev = rows(RUN / f"eval_{n}_{suite}.csv")
        first = {}
        for r in ev:
            key = (r["world_index"], r["arm"])
            if key not in first:
                first[key] = r
        worlds = sorted({r["world_index"] for r in ev})

        def s_at(arm):
            return {w: (first[(w, arm)]["found"]
                        and first[(w, arm)]["expansions"] <= b)
                    for w in worlds}

        va = paired(s_at("learned_cpu"), s_at("euclid"))
        vw = paired(s_at("learned_cpu"), s_at("wastar"))
        res["ceiling"][f"{n}_{suite}"] = dict(binding=b, vs_anchor=va,
                                              vs_wastar=vw)
        md += [f"- rooms_large@{n} (sens binding {b} from recorded rates): "
               f"anchor {va['succ_b']:.2f} learned {va['succ_a']:.2f} "
               f"d={va['delta']:+.2f} [{va['ci'][0]:+.2f},"
               f"{va['ci'][1]:+.2f}] disc {va['discordant'][0]}/"
               f"{va['discordant'][1]} p={va['p']:.4g}; original-w WA* "
               f"{vw['succ_b']:.2f} d={vw['delta']:+.2f} "
               f"[{vw['ci'][0]:+.2f},{vw['ci'][1]:+.2f}] (note: WA* weight "
               f"tuned at the original binding, not re-tuned)"]
    OUT_JSON.write_text(json.dumps(res, indent=1))
    OUT_MD.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
