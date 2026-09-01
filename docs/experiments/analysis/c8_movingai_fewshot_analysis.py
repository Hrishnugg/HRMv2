"""Few-shot rescue analysis on the MovingAI external benchmark.

Joins runs/c8r_movingai/fewshot_raw.csv (adapted arms) with the frozen
raw.csv evaluation rows (zero-shot / anchor / tuned WA*), derives success at
the frozen binding budgets by expansion thresholding, and reports per
(group, K, method): paired success deltas vs zero-shot (R1, primary at K=8
full FT), vs anchor (R2), vs tuned WA* (R3), matched-solved expansion
ratios (R4), and path-cost ratios (R5). Adaptation seeds are averaged
within instances before instance-level bootstrap (10k, seed 20260727).
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "hrm-cloud",
                                     "continuous_prm", "runs", "c8r_movingai"))
BINDING = {"street": 600, "dao": 900}
WSEL = {"street": "wastar_1.5", "dao": "wastar_2"}
KS = [1, 2, 4, 8]
METHODS = ["lora", "full_ft"]
BOOT, SEED = 10_000, 20260727


def solved_at(row, B):
    return row["solved_bigb"] and row["expansions"] <= B


def load_frozen():
    """group -> arm -> instance -> row (evaluation phase)."""
    out = defaultdict(lambda: defaultdict(dict))
    with open(os.path.join(RUNS, "raw.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["phase"] != "eval":
                continue
            row = dict(solved_bigb=r["solved_bigb"] in ("True", "true", "1"),
                       expansions=int(r["expansions"]),
                       arrival=int(r["arrival"]),
                       optimal=int(r["optimal_arrival"]))
            out[r["group"]][r["arm"]][int(r["instance"])] = row
    return out


def load_fewshot():
    """(group, K, method) -> seed -> instance -> row."""
    out = defaultdict(lambda: defaultdict(dict))
    with open(os.path.join(RUNS, "fewshot_raw.csv"), newline="") as f:
        for r in csv.DictReader(f):
            row = dict(solved_bigb=r["solved_bigb"] in ("True", "true", "1"),
                       expansions=int(r["expansions"]),
                       arrival=int(r["arrival"]),
                       optimal=int(r["optimal_arrival"]))
            key = (r["group"], int(r["K"]), r["method"])
            out[key][int(r["adapt_seed"])][int(r["instance"])] = row
    return out


def boot_ci(diffs):
    rng = np.random.default_rng(SEED)
    d = np.asarray(diffs, dtype=np.float64)
    reps = [float(np.mean(d[rng.integers(0, len(d), len(d))]))
            for _ in range(BOOT)]
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def med_ci(vals):
    rng = np.random.default_rng(SEED + 1)
    v = np.asarray(vals, dtype=np.float64)
    reps = [float(np.median(v[rng.integers(0, len(v), len(v))]))
            for _ in range(BOOT)]
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def main():
    frozen = load_frozen()
    fs = load_fewshot()
    results = []
    lines = ["# MovingAI few-shot rescue", ""]
    for group in sorted(frozen):
        B = BINDING[group]
        base = frozen[group]
        inst = sorted(base["euclid"])
        zs = {i: float(solved_at(base["field_unet_blind"][i], B)) for i in inst}
        eu = {i: float(solved_at(base["euclid"][i], B)) for i in inst}
        wa = {i: float(solved_at(base[WSEL[group]][i], B)) for i in inst}
        lines += [f"## {group} (binding {B}, n={len(inst)}; frozen success: "
                  f"zero-shot {np.mean(list(zs.values())):.2f}, anchor "
                  f"{np.mean(list(eu.values())):.2f}, WA* "
                  f"{np.mean(list(wa.values())):.2f})", ""]
        for K in KS:
            for method in METHODS:
                seeds = fs.get((group, K, method))
                if not seeds:
                    continue
                ad = {i: float(np.mean([float(solved_at(seeds[s][i], B))
                                        for s in sorted(seeds)]))
                      for i in inst}
                succ = float(np.mean(list(ad.values())))
                d_zs = [ad[i] - zs[i] for i in inst]
                d_eu = [ad[i] - eu[i] for i in inst]
                d_wa = [ad[i] - wa[i] for i in inst]
                # matched-solved ratios + path cost (per-seed rows, instance-avg)
                r_eu, r_wa, cost = [], [], []
                for i in inst:
                    per_eu, per_wa, per_c = [], [], []
                    for s in sorted(seeds):
                        row = seeds[s][i]
                        if solved_at(row, B):
                            per_c.append(row["arrival"] / max(1, row["optimal"]))
                            if eu[i]:
                                per_eu.append(row["expansions"]
                                              / base["euclid"][i]["expansions"])
                            if wa[i]:
                                per_wa.append(row["expansions"]
                                              / base[WSEL[group]][i]["expansions"])
                    if per_eu:
                        r_eu.append(float(np.mean(per_eu)))
                    if per_wa:
                        r_wa.append(float(np.mean(per_wa)))
                    if per_c:
                        cost.append(float(np.mean(per_c)))
                rec = dict(
                    group=group, K=K, method=method, succ=succ,
                    d_zeroshot=float(np.mean(d_zs)), d_zeroshot_ci=boot_ci(d_zs),
                    d_anchor=float(np.mean(d_eu)), d_anchor_ci=boot_ci(d_eu),
                    d_wastar=float(np.mean(d_wa)), d_wastar_ci=boot_ci(d_wa),
                    ratio_vs_anchor=(float(np.median(r_eu)) if r_eu else None),
                    ratio_vs_anchor_ci=(med_ci(r_eu) if len(r_eu) > 2 else None),
                    ratio_vs_anchor_n=len(r_eu),
                    ratio_vs_wastar=(float(np.median(r_wa)) if r_wa else None),
                    ratio_vs_wastar_ci=(med_ci(r_wa) if len(r_wa) > 2 else None),
                    ratio_vs_wastar_n=len(r_wa),
                    cost_ratio=(float(np.mean(cost)) if cost else None))
                results.append(rec)
                lines.append(
                    f"- K={K} {method}: succ {succ:.2f} | dZS "
                    f"{rec['d_zeroshot']:+.2f} {rec['d_zeroshot_ci']} | dAnchor "
                    f"{rec['d_anchor']:+.2f} {rec['d_anchor_ci']} | dWA* "
                    f"{rec['d_wastar']:+.2f} {rec['d_wastar_ci']} | exp/anchor "
                    f"{rec['ratio_vs_anchor']} (n={rec['ratio_vs_anchor_n']}) | "
                    f"exp/WA* {rec['ratio_vs_wastar']} "
                    f"(n={rec['ratio_vs_wastar_n']}) | cost {rec['cost_ratio']}")
        lines.append("")
    with open(os.path.join(HERE, "c8_movingai_fewshot.json"), "w") as f:
        json.dump(results, f, indent=1)
    md = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "c8_movingai_fewshot_output.md"), "w",
              encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
