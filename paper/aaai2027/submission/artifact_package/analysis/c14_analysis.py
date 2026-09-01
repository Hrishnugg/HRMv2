"""C14 preregistered analysis (design 2026-07-23 + amendment v2).

Inputs: runs/<run>/results/continuous_prm_c14_eval_raw.csv (astar rows only).
Binding budgets: static = lowest calibrated band budget (140); dynamic = the
single binding budget the eval ran (2500).

Implements the four preregistered readouts:
 1. per-cell map-level success delta vs zero_shot + matched-solved median
    expansion ratio vs euclid, with 10k map-bootstrap CIs (seed 20260723);
 2. crossover N*(domain, diversity, seed): first log-N grid point where the
    isotonic-decreasing (PAVA) full-FT ratio curve is <= the LoRA curve,
    linearly interpolated in log N between grid points; map-bootstrap CI;
 3. map-level OLS: ratio ~ logN + method dummies + domain + diversity
    + method x logN + method x domain, map-clustered bootstrap CIs
    (H-C14 needs full_ft x logN significant AND full_ft x domain ~ 0);
 4. diversity readout: dist - conc paired-map success/ratio deltas at fixed N.

Usage: python c14_analysis.py [run_dir_name]
Outputs: c14_analysis_output.md + c14_analysis.json beside this file.
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_RUN = sys.argv[1] if len(sys.argv) > 1 else "c14_local"
RUNS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "hrm-cloud",
                                     "continuous_prm", "runs"))
RAW = os.path.join(RUNS, _RUN, "results", "continuous_prm_c14_eval_raw.csv")

BINDING = {"static": 140, "dynamic": 2500}
BOOT = 10_000
SEED = 20260723
rng_global = np.random.default_rng(SEED)


def load_rows():
    rows = []
    with open(RAW, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("mode") != "astar":
                continue
            dom = r["domain"]
            if int(float(r["budget"])) != BINDING[dom]:
                continue
            rows.append(dict(
                domain=dom, method=r["method"],
                N=int(float(r["N"])) if r["N"] not in ("", None) else -1,
                div=r["diversity"], seed=int(float(r["seed"])) if r["seed"] else -1,
                world=int(float(r["world_index"])),
                found=str(r["found"]) in ("True", "true", "1"),
                exp=float(r["expansions"]),
            ))
    return rows


def index_rows(rows):
    """-> per domain: euclid[world]=(found,exp); zero[world]; cells[(N,div,method,seed)][world]"""
    out = {}
    for dom in ("static", "dynamic"):
        dr = [r for r in rows if r["domain"] == dom]
        eu = {r["world"]: (r["found"], r["exp"]) for r in dr if r["method"] == "euclid"}
        zs = {r["world"]: (r["found"], r["exp"]) for r in dr if r["method"] == "zero_shot"}
        cells = defaultdict(dict)
        for r in dr:
            if r["method"] in ("lora", "full_ft", "scratch"):
                cells[(r["N"], r["div"], r["method"], r["seed"])][r["world"]] = (r["found"], r["exp"])
        out[dom] = dict(euclid=eu, zero=zs, cells=dict(cells))
    return out


def cell_stats(arm, euclid, zero, worlds):
    """success delta vs zero_shot; matched median ratio vs euclid, on `worlds`."""
    dsucc = np.mean([float(arm[w][0]) - float(zero[w][0]) for w in worlds])
    ratios = [arm[w][1] / euclid[w][1] for w in worlds
              if arm[w][0] and euclid[w][0] and euclid[w][1] > 0]
    med = float(np.median(ratios)) if ratios else float("nan")
    return float(dsucc), med, len(ratios)


def boot_ci(fn, worlds, B=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    worlds = list(worlds)
    vals = []
    for _ in range(B):
        sample = [worlds[i] for i in rng.integers(0, len(worlds), len(worlds))]
        v = fn(sample)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def pava_decreasing(y):
    """Pool-adjacent-violators for a DEC sequence (returns the monotone fit)."""
    y = [float(v) for v in y]
    # fit increasing on negated values
    vals = [-v for v in y]
    w = [1.0] * len(vals)
    blocks = [[v, wt, 1] for v, wt in zip(vals, w)]  # value, weight, count
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2, c2 = out.pop()
            v1, w1, c1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    fit = []
    for v, _wt, c in out:
        fit.extend([-v] * c)
    return fit


def crossover_logN(Ns, lora_med, full_med):
    """First log2-N at which PAVA-monotone full <= lora; linear interp between
    grid points; None if no crossing in range. NaN cells are dropped pairwise."""
    pts = [(math.log2(n), lo, fu) for n, lo, fu in zip(Ns, lora_med, full_med)
           if not (math.isnan(lo) or math.isnan(fu))]
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    lo = pava_decreasing([p[1] for p in pts])
    fu = pava_decreasing([p[2] for p in pts])
    diff = [f - l for f, l in zip(fu, lo)]  # <= 0 means full wins
    if diff[0] <= 0:
        return xs[0]
    for i in range(1, len(diff)):
        if diff[i] <= 0:
            d0, d1 = diff[i - 1], diff[i]
            t = d0 / (d0 - d1) if d0 != d1 else 1.0
            return xs[i - 1] + t * (xs[i] - xs[i - 1])
    return None


def ols(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def main():
    rows = load_rows()
    idx = index_rows(rows)
    out = {"run": _RUN, "cells": [], "crossovers": [], "regression": {},
           "diversity": []}
    lines = [f"# C14 preregistered analysis (run `{_RUN}`)", ""]

    # ---- 1. per-cell stats -------------------------------------------------
    lines.append("## Per-cell statistics (success delta vs zero-shot; matched median ratio vs euclid)")
    lines.append("")
    lines.append("| domain | N | div | method | seed | dsucc [CI] | ratio [CI] | n |")
    lines.append("|---|---|---|---|---|---|---|---|")
    cell_meds = {}
    for dom, d in idx.items():
        eu, zs = d["euclid"], d["zero"]
        worlds = sorted(set(eu) & set(zs))
        for (N, div, method, s), arm in sorted(d["cells"].items()):
            ws = [w for w in worlds if w in arm]
            dsucc, med, n = cell_stats(arm, eu, zs, ws)
            ci_d = boot_ci(lambda smp: np.mean([float(arm[w][0]) - float(zs[w][0]) for w in smp]), ws)
            def _med(smp):
                rr = [arm[w][1] / eu[w][1] for w in smp if arm[w][0] and eu[w][0] and eu[w][1] > 0]
                return float(np.median(rr)) if rr else None
            ci_r = boot_ci(_med, ws)
            cell_meds[(dom, N, div, method, s)] = med
            out["cells"].append(dict(domain=dom, N=N, div=div, method=method, seed=s,
                                     dsucc=dsucc, dsucc_ci=ci_d, ratio=med, ratio_ci=ci_r,
                                     matched_n=n))
            lines.append(f"| {dom} | {N} | {div} | {method} | {s} | "
                         f"{dsucc:+.3f} [{ci_d[0]:+.3f},{ci_d[1]:+.3f}] | "
                         f"{med:.3f} [{ci_r[0]:.3f},{ci_r[1]:.3f}] | {n} |")
    lines.append("")

    # ---- 2. crossovers -----------------------------------------------------
    lines.append("## Crossover N* (PAVA-monotone full-FT <= LoRA, log2-N axis)")
    lines.append("")
    Ns_all = sorted({k[0] for d in idx.values() for k in d["cells"]})
    seeds_all = sorted({k[3] for d in idx.values() for k in d["cells"]})
    divs_all = sorted({k[1] for d in idx.values() for k in d["cells"]})
    for dom, d in idx.items():
        eu = d["euclid"]
        worlds_dom = sorted(eu)
        if not worlds_dom or not d["cells"]:
            continue
        for div in divs_all:
            for s in seeds_all:
                lo = [cell_meds.get((dom, N, div, "lora", s), float("nan")) for N in Ns_all]
                fu = [cell_meds.get((dom, N, div, "full_ft", s), float("nan")) for N in Ns_all]
                x = crossover_logN(Ns_all, lo, fu)

                def _xover(smp):
                    lo_b, fu_b = [], []
                    for N in Ns_all:
                        arm_l = d["cells"].get((N, div, "lora", s))
                        arm_f = d["cells"].get((N, div, "full_ft", s))
                        for arm, acc in ((arm_l, lo_b), (arm_f, fu_b)):
                            if arm is None:
                                acc.append(float("nan"))
                                continue
                            rr = [arm[w][1] / eu[w][1] for w in smp
                                  if w in arm and arm[w][0] and eu[w][0] and eu[w][1] > 0]
                            acc.append(float(np.median(rr)) if rr else float("nan"))
                    return crossover_logN(Ns_all, lo_b, fu_b)
                ci = boot_ci(_xover, worlds_dom, B=2000, seed=SEED + 7)
                out["crossovers"].append(dict(domain=dom, div=div, seed=s,
                                              log2N_star=x, ci=ci))
                xs = f"{x:.2f}" if x is not None else "none<=maxN"
                lines.append(f"- {dom}/{div}/s{s}: log2 N* = {xs} "
                             f"[{ci[0]:.2f},{ci[1]:.2f}] "
                             f"(N* ~ {2**x:.0f})" if x is not None else
                             f"- {dom}/{div}/s{s}: no crossing within tested N")
    lines.append("")

    # ---- 3. regression -----------------------------------------------------
    lines.append("## Map-level regression (lora/full_ft/scratch rows)")
    lines.append("")
    Xr, yr, wmap = [], [], []
    for dom, d in idx.items():
        eu = d["euclid"]
        for (N, div, method, s), arm in d["cells"].items():
            for w, (f, e) in arm.items():
                if not (f and eu.get(w, (False, 0))[0] and eu[w][1] > 0):
                    continue
                ratio = e / eu[w][1]
                Xr.append([1.0, math.log2(N),
                           1.0 if method == "full_ft" else 0.0,
                           1.0 if method == "scratch" else 0.0,
                           1.0 if dom == "dynamic" else 0.0,
                           1.0 if div == "dist" else 0.0,
                           (math.log2(N) if method == "full_ft" else 0.0),
                           (1.0 if (method == "full_ft" and dom == "dynamic") else 0.0)])
                yr.append(ratio)
                wmap.append((dom, w))
    names = ["intercept", "log2N", "full_ft", "scratch", "dynamic", "dist",
             "full_ft_x_log2N", "full_ft_x_dynamic"]
    beta = ols(Xr, yr)
    # map-clustered bootstrap over (domain, world) clusters
    clusters = sorted(set(wmap))
    Xa, ya = np.asarray(Xr), np.asarray(yr)
    cl_idx = {c: [i for i, cw in enumerate(wmap) if cw == c] for c in clusters}
    rng = np.random.default_rng(SEED + 13)
    betas = []
    for _ in range(2000):
        smp = [clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]
        rows_i = [i for c in smp for i in cl_idx[c]]
        betas.append(ols(Xa[rows_i], ya[rows_i]))
    betas = np.asarray(betas)
    for j, nm in enumerate(names):
        lo, hi = np.percentile(betas[:, j], [2.5, 97.5])
        out["regression"][nm] = dict(beta=float(beta[j]), ci=[float(lo), float(hi)])
        lines.append(f"- {nm}: {beta[j]:+.4f} [{lo:+.4f},{hi:+.4f}]")
    lines.append("")
    lines.append("H-C14 requires full_ft_x_log2N to exclude zero (negative) and "
                 "full_ft_x_dynamic to be compatible with zero.")
    lines.append("")

    # ---- 4. diversity readout ---------------------------------------------
    lines.append("## Diversity (dist - conc) paired-map deltas at fixed N")
    lines.append("")
    for dom, d in idx.items():
        eu = d["euclid"]
        for N in Ns_all:
            for method in ("lora", "full_ft"):
                deltas = []
                for s in seeds_all:
                    a_c = d["cells"].get((N, "conc", method, s))
                    a_d = d["cells"].get((N, "dist", method, s))
                    if not a_c or not a_d:
                        continue
                    common = sorted(set(a_c) & set(a_d))
                    deltas.extend(float(a_d[w][0]) - float(a_c[w][0]) for w in common)
                if deltas:
                    m = float(np.mean(deltas))
                    lines.append(f"- {dom} N={N} {method}: dist-conc success {m:+.3f} "
                                 f"(n={len(deltas)} map-seed pairs)")
                    out["diversity"].append(dict(domain=dom, N=N, method=method,
                                                 dsucc=m, n=len(deltas)))
    lines.append("")

    md = os.path.join(HERE, "c14_analysis_output.md")
    js = os.path.join(HERE, "c14_analysis.json")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n".join(lines[-40:]))
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
