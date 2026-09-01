"""MovingAI external-benchmark analysis (frozen design 2026-07-25).

From runs/c8r_movingai/raw.csv: per group, (1) calibrate the binding budget on
development instances (anchor-only, canonical rule: grid budgets closest to
targets 0.45/0.70, ties smaller, binding = smaller selected); (2) tune w_h on
development instances at the binding budget (highest success, ties smaller);
(3) evaluate on the evaluation instances: success at binding for anchor /
tuned WA* / fixed blind U-Net, paired deltas with 10k instance-bootstrap CIs
and exact McNemar, matched-solved expansion ratios, and matched (jointly
solved) mean suboptimality with the paired per-instance difference.

Solve-at-budget is derived by thresholding recorded solve expansions
(prefix-deterministic search; same rule as the budget curves).
"""
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "hrm-cloud", "continuous_prm",
    "runs", "c8r_movingai", "raw.csv"))
GRID = [150, 250, 400, 600, 900, 1300, 1800, 2500, 3500]
WEIGHTS = [1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
GROUPS = ["street", "dao"]
BOOT, SEED = 10_000, 20260725


def solved_at(row, B):
    return row["solved_bigb"] and row["expansions"] <= B


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n
    return min(1.0, 2.0 * cdf)


def boot_ci(vals, stat, rng):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return None
    reps = [float(stat(vals[rng.integers(0, len(vals), len(vals))]))
            for _ in range(BOOT)]
    return (float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5)))


def main():
    rows = []
    with open(RAW, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                group=r["group"], phase=r["phase"], instance=int(r["instance"]),
                map=r["map"], arm=r["arm"],
                solved_bigb=r["solved_bigb"] in ("True", "true", "1"),
                expansions=int(r["expansions"]), arrival=int(r["arrival"]),
                optimal=int(r["optimal_arrival"])))

    out = {}
    lines = ["# MovingAI external benchmark (dynamic zero-shot)", ""]
    for g in GROUPS:
        dev = [r for r in rows if r["group"] == g and r["phase"] == "dev"]
        ev = [r for r in rows if r["group"] == g and r["phase"] == "eval"]
        dev_e = [r for r in dev if r["arm"] == "euclid"]
        succ_at = {B: np.mean([solved_at(r, B) for r in dev_e]) for B in GRID}
        chosen = []
        for tgt in (0.45, 0.70):
            best = min(GRID, key=lambda B: (abs(succ_at[B] - tgt), B))
            if best not in chosen:
                chosen.append(best)
        binding = min(chosen)
        wa_succ = {}
        for w in WEIGHTS:
            dw = [r for r in dev if r["arm"] == f"wastar_{w:g}"]
            wa_succ[w] = np.mean([solved_at(r, binding) for r in dw])
        w_star = max(WEIGHTS, key=lambda w: (wa_succ[w], -w))

        by = {arm: {r["instance"]: r for r in ev if r["arm"] == arm}
              for arm in ("euclid", f"wastar_{w_star:g}", "field_unet_blind")}
        inst = sorted(by["euclid"])
        eu = {i: solved_at(by["euclid"][i], binding) for i in inst}
        wa = {i: solved_at(by[f"wastar_{w_star:g}"][i], binding) for i in inst}
        le = {i: solved_at(by["field_unet_blind"][i], binding) for i in inst}
        rng = np.random.default_rng(SEED)

        def paired(a, b):
            d = [float(a[i]) - float(b[i]) for i in inst]
            ci = boot_ci(d, np.mean, rng)
            b_only = sum(1 for i in inst if a[i] and not b[i])
            c_only = sum(1 for i in inst if b[i] and not a[i])
            return dict(delta=float(np.mean(d)), ci=ci,
                        discordant=[b_only, c_only],
                        p_mcnemar=exact_mcnemar(b_only, c_only))

        def ratio(a_rows, b_rows, a_ok, b_ok):
            joint = [i for i in inst if a_ok[i] and b_ok[i]]
            vals = [a_rows[i]["expansions"] / b_rows[i]["expansions"] for i in joint]
            return dict(median=float(np.median(vals)) if vals else None,
                        ci=boot_ci(vals, np.median, rng), n=len(joint))

        def subopt(a_rows, a_ok, joint):
            return [a_rows[i]["arrival"] / a_rows[i]["optimal"] for i in joint
                    if a_rows[i]["optimal"] > 0]

        jl = [i for i in inst if le[i] and wa[i]]
        sub_l = subopt(by["field_unet_blind"], le, jl)
        sub_w = subopt(by[f"wastar_{w_star:g}"], wa, jl)
        d_sub = [a - b for a, b in zip(sub_l, sub_w)]

        res = dict(
            binding=binding, w_star=w_star,
            n_eval=len(inst),
            succ=dict(euclid=float(np.mean([eu[i] for i in inst])),
                      wastar=float(np.mean([wa[i] for i in inst])),
                      learned=float(np.mean([le[i] for i in inst]))),
            learned_vs_euclid=paired(le, eu),
            learned_vs_wastar=paired(le, wa),
            ratio_learned_euclid=ratio(by["field_unet_blind"], by["euclid"], le, eu),
            ratio_learned_wastar=ratio(by["field_unet_blind"],
                                       by[f"wastar_{w_star:g}"], le, wa),
            subopt_joint=dict(
                learned=float(np.mean(sub_l)) if sub_l else None,
                wastar=float(np.mean(sub_w)) if sub_w else None,
                diff_mean=float(np.mean(d_sub)) if d_sub else None,
                diff_ci=boot_ci(d_sub, np.mean, rng), n=len(jl)),
        )
        out[g] = res
        lve, lvw = res["learned_vs_euclid"], res["learned_vs_wastar"]
        rle, rlw = res["ratio_learned_euclid"], res["ratio_learned_wastar"]
        sj = res["subopt_joint"]
        lines += [
            f"## {g} (binding {binding}, tuned w_h={w_star:g}, n={len(inst)})",
            "",
            f"- success: anchor {res['succ']['euclid']:.2f} | WA* "
            f"{res['succ']['wastar']:.2f} | learned {res['succ']['learned']:.2f}",
            f"- learned-anchor: {lve['delta']:+.2f} "
            f"[{lve['ci'][0]:+.2f},{lve['ci'][1]:+.2f}], discordant "
            f"{lve['discordant'][0]}/{lve['discordant'][1]}, exact p="
            f"{lve['p_mcnemar']:.2e}",
            f"- learned-WA*: {lvw['delta']:+.2f} "
            f"[{lvw['ci'][0]:+.2f},{lvw['ci'][1]:+.2f}], discordant "
            f"{lvw['discordant'][0]}/{lvw['discordant'][1]}, exact p="
            f"{lvw['p_mcnemar']:.2e}",
            f"- matched ratio learned/anchor: {rle['median']:.3f} "
            f"[{rle['ci'][0]:.3f},{rle['ci'][1]:.3f}] (n={rle['n']})"
            if rle["median"] is not None else "- matched ratio learned/anchor: n/a",
            f"- matched ratio learned/WA*: {rlw['median']:.3f} "
            f"[{rlw['ci'][0]:.3f},{rlw['ci'][1]:.3f}] (n={rlw['n']})"
            if rlw["median"] is not None else "- matched ratio learned/WA*: n/a",
            f"- joint subopt learned {sj['learned']:.3f} vs WA* {sj['wastar']:.3f}, "
            f"paired diff {sj['diff_mean']:+.3f} "
            f"[{sj['diff_ci'][0]:+.3f},{sj['diff_ci'][1]:+.3f}] (n={sj['n']})"
            if sj["learned"] is not None else "- joint subopt: n/a",
            "",
        ]

    with open(os.path.join(HERE, "c8_movingai_analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    md = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "c8_movingai_analysis_output.md"), "w",
              encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
