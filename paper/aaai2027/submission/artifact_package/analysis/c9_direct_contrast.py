"""Direct paired full-FT minus LoRA contrasts from the C9 raw rows.

Answers the v2 deep review's objection that the crossover claim relies on
disjoint marginal intervals: computes the map-paired FT-LoRA difference in
(i) success and (ii) matched expansion ratio (vs euclid), at K in {1, 16},
per (target, backbone), with map-clustered 10k bootstraps (adaptation seeds
averaged within maps first; astar mode; binding budgets).

Usage: python c9_direct_contrast.py [run_dir_name]
Outputs: c9_direct_contrast_output.md + .json beside this file.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_RUN = sys.argv[1] if len(sys.argv) > 1 else "c9_local"
RUNS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "hrm-cloud",
                                     "continuous_prm", "runs"))
RAW = os.path.join(RUNS, _RUN, "results", "continuous_prm_c9_eval_raw.csv")

BOOT = 10_000
SEED = 20260723
KS = (1, 16)
METHODS = ("full_ft", "lora")


def load_rows():
    rows = []
    with open(RAW, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("mode") != "astar":
                continue
            rows.append(dict(
                target=r["target"], K=int(float(r["K"])) if r["K"] else -1,
                seed=int(float(r["seed"])) if r["seed"] not in ("", None) else -1,
                method=r["method"], backbone=r["backbone"],
                provider=r.get("provider", ""),
                world=int(float(r["world_index"])),
                budget=int(float(r["budget"])),
                found=str(r["found"]) in ("True", "true", "1"),
                exp=float(r["expansions"]),
            ))
    return rows


def binding_budget(rows, target, floor=0.05):
    """Lowest budget with euclid success >= floor; else highest (mirrors C9)."""
    by_b = defaultdict(list)
    for r in rows:
        if r["target"] == target and r["provider"] == "euclid":
            by_b[r["budget"]].append(r["found"])
    budgets = sorted(by_b)
    for b in budgets:
        if np.mean(by_b[b]) >= floor:
            return b
    return budgets[-1]


def main():
    rows = load_rows()
    targets = sorted({r["target"] for r in rows})
    backbones = sorted({r["backbone"] for r in rows if r["backbone"]})
    out = {"run": _RUN, "contrasts": []}
    lines = ["# Direct paired full-FT $-$ LoRA contrasts (C9 raw, map-clustered)", ""]
    lines.append("| target | backbone | K | n maps | d success [95% CI] | d ratio [95% CI] | matched n |")
    lines.append("|---|---|---|---|---|---|---|")
    win_ft = win_lora = 0
    for target in targets:
        trows = [r for r in rows if r["target"] == target]
        bb = binding_budget(trows, target)
        eu = {}
        for r in trows:
            if r["provider"] == "euclid" and r["budget"] == bb:
                eu[r["world"]] = (r["found"], r["exp"])
        for backbone in backbones:
            for K in KS:
                # per-map per-method: average over adaptation seeds within map
                per = {m: defaultdict(lambda: ([], [])) for m in METHODS}
                for r in trows:
                    if (r["method"] in METHODS and r["backbone"] == backbone
                            and r["K"] == K and r["budget"] == bb):
                        f, e = per[r["method"]][r["world"]]
                        f.append(1.0 if r["found"] else 0.0)
                        e.append(r["exp"] if r["found"] else np.nan)
                maps = sorted(set(per["full_ft"]) & set(per["lora"]) & set(eu))
                if not maps:
                    continue

                def stats(sample):
                    dsucc, dratio = [], []
                    for w in sample:
                        ft_f, ft_e = per["full_ft"][w]
                        lo_f, lo_e = per["lora"][w]
                        dsucc.append(np.mean(ft_f) - np.mean(lo_f))
                        if eu[w][0] and eu[w][1] > 0:
                            fte = np.nanmean(ft_e)
                            loe = np.nanmean(lo_e)
                            if np.isfinite(fte) and np.isfinite(loe):
                                dratio.append((fte - loe) / eu[w][1])
                    return (float(np.mean(dsucc)),
                            float(np.mean(dratio)) if dratio else float("nan"),
                            len(dratio))

                ds, dr, mn = stats(maps)
                rng = np.random.default_rng(SEED)
                bs_s, bs_r = [], []
                for _ in range(BOOT):
                    smp = [maps[i] for i in rng.integers(0, len(maps), len(maps))]
                    s_, r_, _ = stats(smp)
                    bs_s.append(s_)
                    if not np.isnan(r_):
                        bs_r.append(r_)
                ci_s = (float(np.percentile(bs_s, 2.5)), float(np.percentile(bs_s, 97.5)))
                ci_r = ((float(np.percentile(bs_r, 2.5)), float(np.percentile(bs_r, 97.5)))
                        if bs_r else (float("nan"), float("nan")))
                if not np.isnan(dr):
                    if dr < 0:
                        win_ft += 1
                    elif dr > 0:
                        win_lora += 1
                out["contrasts"].append(dict(
                    target=target, backbone=backbone, K=K, n_maps=len(maps),
                    dsucc=ds, dsucc_ci=ci_s, dratio=dr, dratio_ci=ci_r, matched_n=mn))
                lines.append(
                    f"| {target} | {backbone} | {K} | {len(maps)} | "
                    f"{ds:+.3f} [{ci_s[0]:+.3f},{ci_s[1]:+.3f}] | "
                    f"{dr:+.3f} [{ci_r[0]:+.3f},{ci_r[1]:+.3f}] | {mn} |")
    lines.append("")
    lines.append(f"Ratio-delta direction count (all cells): FT better {win_ft}, LoRA better {win_lora}.")
    lines.append("Negative d ratio = full-FT expands less than LoRA (relative to euclid on triple-matched maps).")
    md = os.path.join(HERE, "c9_direct_contrast_output.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(HERE, "c9_direct_contrast.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
