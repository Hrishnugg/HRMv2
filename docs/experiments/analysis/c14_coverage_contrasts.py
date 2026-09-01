"""C14 direct coverage contrasts: distributed minus concentrated, every cell.

For each (domain, N, method): the paired per-map success difference between
the distributed and concentrated arms (adaptation seeds averaged within maps
first, per the paper's inference policy), with a 10k map-bootstrap CI
(seed 20260723). 30 rows = 2 domains x 5 N x 3 methods. This is the compact
global readout of the factorial's central treatment effect requested by
external review; per-cell deltas vs the source remain in c14_analysis.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_RUN = sys.argv[1] if len(sys.argv) > 1 else "c14_modal"
RUNS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "hrm-cloud",
                                     "continuous_prm", "runs"))
RAW = os.path.join(RUNS, _RUN, "results", "continuous_prm_c14_eval_raw.csv")
BINDING = {"static": 140, "dynamic": 2500}
BOOT, SEED = 10_000, 20260723


def main():
    per = defaultdict(lambda: defaultdict(list))  # (dom,N,div,method) -> world -> [found...]
    with open(RAW, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("mode") != "astar":
                continue
            dom = r["domain"]
            if int(float(r["budget"])) != BINDING[dom]:
                continue
            if r["method"] not in ("lora", "full_ft", "scratch"):
                continue
            key = (dom, int(float(r["N"])), r["diversity"], r["method"])
            per[key][int(float(r["world_index"]))].append(
                str(r["found"]) in ("True", "true", "1"))

    avg = {k: {w: float(np.mean(v)) for w, v in d.items()} for k, d in per.items()}
    rows = []
    lines = ["# C14 direct coverage contrasts (distributed - concentrated)",
             "",
             "Paired per-map success differences, adaptation seeds averaged "
             "within maps, 10k map bootstraps. 30 cells; marginal intervals "
             "(one declared family per domain x method, 5 N levels each).",
             "",
             "| Domain | Method | N | dist-conc | 95% CI | n maps |",
             "|---|---|---|---|---|---|"]
    for dom in ("static", "dynamic"):
        for method in ("full_ft", "lora", "scratch"):
            for N in (256, 1024, 4096, 16384, 65536):
                kc, kd = (dom, N, "conc", method), (dom, N, "dist", method)
                if kc not in avg or kd not in avg:
                    continue
                worlds = sorted(set(avg[kc]) & set(avg[kd]))
                d = np.array([avg[kd][w] - avg[kc][w] for w in worlds])
                rng = np.random.default_rng(SEED)
                reps = [float(np.mean(d[rng.integers(0, len(d), len(d))]))
                        for _ in range(BOOT)]
                lo, hi = np.percentile(reps, 2.5), np.percentile(reps, 97.5)
                rows.append(dict(domain=dom, method=method, N=N,
                                 delta=float(np.mean(d)),
                                 ci=[float(lo), float(hi)], n=len(worlds)))
                mark = "*" if lo > 0 or hi < 0 else ""
                lines.append(f"| {dom} | {method} | {N} | {np.mean(d):+.3f}{mark} "
                             f"| [{lo:+.3f},{hi:+.3f}] | {len(worlds)} |")
    lines += ["", "*: marginal 95% interval excludes zero."]
    with open(os.path.join(HERE, "c14_coverage_contrasts.json"), "w") as f:
        json.dump(rows, f, indent=1)
    md = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "c14_coverage_contrasts_output.md"), "w",
              encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
