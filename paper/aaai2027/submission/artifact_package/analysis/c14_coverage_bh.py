"""BH-adjusted inference for the 30 C14 coverage contrasts.

Per cell: paired sign-flip (randomization) p-value on the per-map
distributed-concentrated success differences (seeds averaged within maps,
20k flips, seed 20260728), then Benjamini-Hochberg across the 30 cells.
Companion to c14_coverage_contrasts.py (which reports marginal bootstrap
CIs); addresses the v7 review's multiplicity objection.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_RUN = sys.argv[1] if len(sys.argv) > 1 else "c14_modal"
RAW = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "hrm-cloud", "continuous_prm", "runs", _RUN,
    "results", "continuous_prm_c14_eval_raw.csv"))
BINDING = {"static": 140, "dynamic": 2500}
FLIPS, SEED = 20_000, 20260728


def main():
    per = defaultdict(lambda: defaultdict(list))
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
    avg = {k: {w: float(np.mean(v)) for w, v in d.items()}
           for k, d in per.items()}

    cells = []
    rng = np.random.default_rng(SEED)
    for dom in ("static", "dynamic"):
        for method in ("full_ft", "lora", "scratch"):
            for N in (256, 1024, 4096, 16384, 65536):
                kc, kd = (dom, N, "conc", method), (dom, N, "dist", method)
                worlds = sorted(set(avg[kc]) & set(avg[kd]))
                d = np.array([avg[kd][w] - avg[kc][w] for w in worlds])
                obs = float(np.mean(d))
                signs = rng.choice([-1.0, 1.0], size=(FLIPS, len(d)))
                null = (signs * d[None, :]).mean(axis=1)
                p = float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1)
                          / (FLIPS + 1))
                cells.append(dict(domain=dom, method=method, N=N,
                                  delta=obs, p=p))
    # Benjamini-Hochberg across the 30 cells
    order = np.argsort([c["p"] for c in cells])
    m = len(cells)
    q = [None] * m
    prev = 1.0
    for rank_pos, idx in list(enumerate(order))[::-1]:
        val = cells[idx]["p"] * m / (rank_pos + 1)
        prev = min(prev, val)
        q[idx] = prev
    for c, qq in zip(cells, q):
        c["q_bh"] = float(qq)

    sig = [c for c in cells if c["q_bh"] < 0.05]
    lines = ["# C14 coverage contrasts: BH-adjusted sign-flip inference", "",
             "| Domain | Method | N | dist-conc | p (sign-flip) | q (BH, 30) |",
             "|---|---|---|---|---|---|"]
    for c in cells:
        mark = "*" if c["q_bh"] < 0.05 else ""
        lines.append(f"| {c['domain']} | {c['method']} | {c['N']} | "
                     f"{c['delta']:+.3f} | {c['p']:.4f} | "
                     f"{c['q_bh']:.4f}{mark} |")
    lines += ["", f"q<0.05 cells: {len(sig)} of 30"]
    ft_sc = [c for c in cells if c["method"] in ("full_ft", "scratch")]
    low = [c for c in ft_sc if (c["domain"] == "static" and c["N"] == 256)
           or (c["domain"] == "dynamic" and c["N"] <= 16384)]
    high = [c for c in ft_sc if c not in low]
    lines.append(f"FT/scratch cells with w_min<=2: "
                 f"{sum(1 for c in low if c['q_bh'] < 0.05)}/{len(low)} at q<0.05; "
                 f"with w_min>=4: {sum(1 for c in high if c['q_bh'] < 0.05)}"
                 f"/{len(high)}")
    with open(os.path.join(HERE, "c14_coverage_bh.json"), "w") as f:
        json.dump(cells, f, indent=1)
    md = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "c14_coverage_bh_output.md"), "w",
              encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
