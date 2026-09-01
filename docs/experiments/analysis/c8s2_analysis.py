"""C8-S v2 frozen analysis (design 2026-07-26-c8-scale-walltime + Amendment 1).

Registered readouts on runs/c8s2_scale rows (shared worlds across sizes
{192,512,1024,2048}; arms euclid / wastar / learned_cpu / learned_gpu /
sipp; 3 repeats, randomized order; first eval world per shard = warmup,
excluded from timing only):

R1 persistence: per N, learned-anchor success delta; exact McNemar + BH
   within each N (6 suites = the confirmatory family); paired map CI;
   pass = CI excludes zero in >= 5/6 suites.
R2 effort: matched learned/anchor expansion-ratio median < 1 per suite per
   N (map-bootstrap CI; descriptive companion).
R3 crossover (primary novel): smallest N where paired learned_gpu - WA*
   total-time 95% map CI < 0 AND success noninferior within 0.05 AND mean
   path suboptimality within +0.02 of WA*; CPU alongside; log-log size x
   method slope contrast as secondary.
R4 SIPP reference (own units, never merged). R5 GPU table-build component.
Probe: predicted-vs-true residual correlation / MAE / bias per size.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
RUN = ROOT / "runs" / "c8s2_scale"
OUT_JSON = HERE.parent / "c8s2_analysis.json"
OUT_MD = HERE.parent / "c8s2_analysis_output.md"

SIZES = (192, 512, 1024, 2048)
SUITES = ("C_dyn_maze", "C_dyn_rooms", "C_dyn_spiral", "C_dyn_maze_dense",
          "C_dyn_crossing", "C_dyn_rooms_large")
SHORT = {s: s.replace("C_dyn_", "") for s in SUITES}
NBOOT = 10_000
RNG = np.random.default_rng(20260727)


def load(size, suite):
    p = RUN / f"eval_{size}_{suite}.csv"
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["found"] = r["found"] == "True"
        r["warmup"] = r["warmup"] == "True"
        for k in ("expansions", "arrival", "optimal_arrival", "world_index",
                  "repeat", "budget"):
            r[k] = int(r[k])
        for k in ("t_total_s", "t_table_s", "t_search_s"):
            r[k] = float(r[k])
    return rows


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, 0.5).pvalue * 1.0


def bh(pairs):
    items = sorted(pairs.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 1.0
    for rank in range(m, 0, -1):
        k, p = items[rank - 1]
        prev = min(prev, p * m / rank)
        out[k] = prev
    return out


def boot_mean_ci(vals):
    a = np.asarray(vals, dtype=float)
    idx = RNG.integers(0, len(a), size=(NBOOT, len(a)))
    means = a[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(a.mean()), float(lo), float(hi)


def boot_mean_ci_p(vals):
    """Mean + percentile CI + two-sided add-one bootstrap p (vs zero)."""
    a = np.asarray(vals, dtype=float)
    idx = RNG.integers(0, len(a), size=(NBOOT, len(a)))
    means = a[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_le = (np.sum(means <= 0.0) + 1) / (NBOOT + 1)
    p_ge = (np.sum(means >= 0.0) + 1) / (NBOOT + 1)
    p = float(min(1.0, 2 * min(p_le, p_ge)))
    return float(a.mean()), float(lo), float(hi), p


def boot_median_ci(vals):
    a = np.asarray(vals, dtype=float)
    idx = RNG.integers(0, len(a), size=(NBOOT, len(a)))
    meds = np.median(a[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(np.median(a)), float(lo), float(hi)


def main():
    res = {"sizes": {}, "probe": {}, "R3": {}}
    md = ["# C8-S v2 analysis output", ""]
    per = {}  # (size, suite) -> per-world dict
    div = {}  # cpu/gpu divergence tally

    for size in SIZES:
        for suite in SUITES:
            rows = load(size, suite)
            arms = sorted({r["arm"] for r in rows})
            assert set(arms) == {"euclid", "wastar", "learned_cpu",
                                 "learned_gpu", "sipp"}, (size, suite, arms)
            worlds = sorted({r["world_index"] for r in rows})
            by = defaultdict(dict)  # world -> arm -> dict
            for w in worlds:
                for arm in arms:
                    rs = [r for r in rows
                          if r["world_index"] == w and r["arm"] == arm]
                    assert rs, (size, suite, w, arm)
                    f0, e0 = rs[0]["found"], rs[0]["expansions"]
                    assert all(r["found"] == f0 and r["expansions"] == e0
                               for r in rs), ("nondet", size, suite, w, arm)
                    tim = [r for r in rs if not r["warmup"]]
                    tvals = [r["t_total_s"] for r in (tim or rs)]
                    by[w][arm] = dict(
                        found=f0, exp=e0, arrival=rs[0]["arrival"],
                        opt=rs[0]["optimal_arrival"],
                        t=float(np.mean(tvals)),
                        t_table=float(np.mean(
                            [r["t_table_s"] for r in (tim or rs)])))
            # learned_cpu vs learned_gpu: same code path, but device float
            # differences can reorder ties -> tally divergence, don't assert
            fm = sum(1 for w in worlds
                     if by[w]["learned_cpu"]["found"]
                     != by[w]["learned_gpu"]["found"])
            em = [abs(by[w]["learned_cpu"]["exp"] - by[w]["learned_gpu"]["exp"])
                  for w in worlds]
            div.setdefault(size, {})[suite] = dict(
                found_mismatch=fm, exp_mismatch=int(np.sum(np.array(em) > 0)),
                max_abs_exp_diff=int(max(em)) if em else 0)
            per[(size, suite)] = by

    # ---- R1 + R2 per size ----
    for size in SIZES:
        pv, entry = {}, {}
        for suite in SUITES:
            by = per[(size, suite)]
            worlds = sorted(by)
            le = np.array([by[w]["learned_cpu"]["found"] for w in worlds])
            eu = np.array([by[w]["euclid"]["found"] for w in worlds])
            b = int(np.sum(le & ~eu))
            c = int(np.sum(~le & eu))
            pv[suite] = mcnemar_exact(b, c)
            d, lo, hi = boot_mean_ci(le.astype(float) - eu.astype(float))
            joint = [w for w in worlds
                     if by[w]["learned_cpu"]["found"]
                     and by[w]["euclid"]["found"]]
            ratios = [by[w]["learned_cpu"]["exp"] / by[w]["euclid"]["exp"]
                      for w in joint]
            if len(ratios) >= 2:
                rm, rlo, rhi = boot_median_ci(ratios)
            elif ratios:
                rm, rlo, rhi = float(ratios[0]), math.nan, math.nan
            else:
                rm = rlo = rhi = math.nan
            entry[suite] = dict(
                n=len(worlds), succ_learned=float(le.mean()),
                succ_euclid=float(eu.mean()), delta=d, ci=[lo, hi],
                discordant=[b, c], ratio=rm, ratio_ci=[rlo, rhi],
                n_joint=len(joint))
        qv = bh(pv)
        n_excl = sum(1 for s in SUITES
                     if entry[s]["ci"][0] > 0 or entry[s]["ci"][1] < 0)
        n_q = sum(1 for s in SUITES if qv[s] < 0.05)
        r2_all = all(entry[s]["ratio"] < 1 for s in SUITES
                     if not math.isnan(entry[s]["ratio"]))
        for s in SUITES:
            entry[s]["p"] = pv[s]
            entry[s]["q"] = qv[s]
        res["sizes"][size] = dict(
            suites=entry, R1_ci_excl=n_excl, R1_q_sig=n_q,
            R1_pass=bool(n_excl >= 5), R2_all_below_1=bool(r2_all))

    # ---- R3 crossover + secondary scaling ----
    for suite in SUITES:
        sd = {}
        for size in SIZES:
            by = per[(size, suite)]
            worlds = sorted(by)
            cell = {}
            for learned_arm in ("learned_gpu", "learned_cpu"):
                dt = [by[w][learned_arm]["t"] - by[w]["wastar"]["t"]
                      for w in worlds]
                m, lo, hi, dt_p = boot_mean_ci_p(dt)
                le = np.array([by[w][learned_arm]["found"] for w in worlds])
                wa = np.array([by[w]["wastar"]["found"] for w in worlds])
                sdelta = float(le.mean() - wa.mean())
                joint = [w for w in worlds
                         if by[w][learned_arm]["found"]
                         and by[w]["wastar"]["found"]]
                if joint:
                    sub_l = float(np.mean(
                        [by[w][learned_arm]["arrival"] / by[w][learned_arm]["opt"]
                         for w in joint]))
                    sub_w = float(np.mean(
                        [by[w]["wastar"]["arrival"] / by[w]["wastar"]["opt"]
                         for w in joint]))
                else:
                    sub_l = sub_w = math.nan
                ok = (hi < 0.0 and sdelta >= -0.05
                      and (math.isnan(sub_l) or sub_l <= sub_w + 0.02))
                cell[learned_arm] = dict(
                    dt_mean=m, dt_ci=[lo, hi], dt_p=dt_p, succ_delta=sdelta,
                    subopt=[sub_l, sub_w], crossover=bool(ok))
            cell["t_means"] = {
                arm: float(np.mean([by[w][arm]["t"] for w in sorted(by)]))
                for arm in ("euclid", "wastar", "learned_cpu",
                            "learned_gpu", "sipp")}
            sd[size] = cell
        cross = {}
        for learned_arm in ("learned_gpu", "learned_cpu"):
            hit = [n for n in SIZES if sd[n][learned_arm]["crossover"]]
            cross[learned_arm] = hit[0] if hit else None
        # secondary: log-log slope contrast (world-mean times)
        slopes = {}
        for arm in ("learned_gpu", "learned_cpu", "wastar", "euclid"):
            xs = np.log([float(n) for n in SIZES])
            ys = np.log([sd[n]["t_means"][arm] for n in SIZES])
            slopes[arm] = float(np.polyfit(xs, ys, 1)[0])
        res["R3"][suite] = dict(per_size=sd, crossover_N=cross,
                                loglog_slopes=slopes)

    # multiplicity companion (review item): BH across the 24 suite x size
    # dt contrasts per learned arm; the frozen crossover rule is unchanged,
    # q-values reported alongside
    for learned_arm in ("learned_gpu", "learned_cpu"):
        pv = {f"{s}|{n}": res["R3"][s]["per_size"][n][learned_arm]["dt_p"]
              for s in SUITES for n in SIZES}
        qv = bh(pv)
        for key, q in qv.items():
            s, n = key.split("|")
            res["R3"][s]["per_size"][int(n)][learned_arm]["dt_q"] = q

    # slope-contrast bootstrap (review item): paired world resampling on
    # the four suites with non-degenerate size sweeps
    NONDEG = ("C_dyn_maze", "C_dyn_rooms", "C_dyn_crossing",
              "C_dyn_rooms_large")
    xs = np.log([float(n) for n in SIZES])
    res["slope_boot"] = {}
    for suite in NONDEG:
        worlds = sorted(per[(192, suite)])
        tmat = {arm: np.array([[per[(n, suite)][w][arm]["t"] for w in worlds]
                               for n in SIZES])
                for arm in ("learned_gpu", "learned_cpu", "wastar", "euclid")}
        idx = RNG.integers(0, len(worlds), size=(NBOOT, len(worlds)))
        out = {}
        for arm_a, arm_b in (("wastar", "learned_gpu"),
                             ("wastar", "learned_cpu"),
                             ("euclid", "learned_gpu")):
            contrasts = np.empty(NBOOT)
            for bi in range(NBOOT):
                sel = idx[bi]
                ya = np.log(tmat[arm_a][:, sel].mean(axis=1))
                yb = np.log(tmat[arm_b][:, sel].mean(axis=1))
                contrasts[bi] = (np.polyfit(xs, ya, 1)[0]
                                 - np.polyfit(xs, yb, 1)[0])
            ya = np.log(tmat[arm_a].mean(axis=1))
            yb = np.log(tmat[arm_b].mean(axis=1))
            point = float(np.polyfit(xs, ya, 1)[0]
                          - np.polyfit(xs, yb, 1)[0])
            lo, hi = np.percentile(contrasts, [2.5, 97.5])
            out[f"{arm_a}-minus-{arm_b}"] = [point, float(lo), float(hi)]
        res["slope_boot"][suite] = out

    # ---- R4 / R5 / probe ----
    r4 = {}
    for size in SIZES:
        row = {}
        for suite in SUITES:
            by = per[(size, suite)]
            worlds = sorted(by)
            row[suite] = dict(
                succ=float(np.mean([by[w]["sipp"]["found"] for w in worlds])),
                t=float(np.mean([by[w]["sipp"]["t"] for w in worlds])))
        r4[size] = row
    res["R4_sipp"] = r4
    res["R5_gpu_table_s"] = {
        size: float(np.mean(
            [per[(size, s)][w]["learned_gpu"]["t_table"]
             for s in SUITES for w in per[(size, s)]]))
        for size in SIZES}

    for size in SIZES:
        pr = {}
        for suite in SUITES:
            p = RUN / f"probe_{size}_{suite}.csv"
            with open(p, newline="") as f:
                rows = list(csv.DictReader(f))
            pr[suite] = dict(
                r=float(np.median([float(r["pearson_r"]) for r in rows])),
                mae=float(np.median([float(r["mae"]) for r in rows])),
                bias=float(np.median([float(r["bias"]) for r in rows])))
        res["probe"][size] = pr

    # ---- markdown ----
    for size in SIZES:
        e = res["sizes"][size]
        md += [f"## N={size}: R1 CI-excl {e['R1_ci_excl']}/6 "
               f"(q<.05 in {e['R1_q_sig']}/6) "
               f"{'PASS' if e['R1_pass'] else 'FAIL'}; "
               f"R2 all-ratios<1 {e['R2_all_below_1']}"]
        for s in SUITES:
            x = e["suites"][s]
            md += [f"- {SHORT[s]}: succ {x['succ_euclid']:.2f}->"
                   f"{x['succ_learned']:.2f} d={x['delta']:+.2f} "
                   f"[{x['ci'][0]:+.2f},{x['ci'][1]:+.2f}] q={x['q']:.4g}; "
                   f"ratio {x['ratio']:.3f} "
                   f"[{x['ratio_ci'][0]:.3f},{x['ratio_ci'][1]:.3f}] "
                   f"(n_joint {x['n_joint']})"]
        md += [""]
    md += ["## R3 crossover (frozen rule; smallest N or none)", ""]
    for suite in SUITES:
        r3 = res["R3"][suite]
        gpu_n, cpu_n = r3["crossover_N"]["learned_gpu"], \
            r3["crossover_N"]["learned_cpu"]
        md += [f"- {SHORT[suite]}: GPU {gpu_n} CPU {cpu_n}; "
               f"slopes gpu {r3['loglog_slopes']['learned_gpu']:.2f} "
               f"cpu {r3['loglog_slopes']['learned_cpu']:.2f} "
               f"wa* {r3['loglog_slopes']['wastar']:.2f} "
               f"eu {r3['loglog_slopes']['euclid']:.2f}"]
        for size in SIZES:
            c = r3["per_size"][size]
            g = c["learned_gpu"]
            md += [f"    N={size}: gpu-wa* dt {g['dt_mean']:+.3f}s "
                   f"[{g['dt_ci'][0]:+.3f},{g['dt_ci'][1]:+.3f}] "
                   f"q={g.get('dt_q', float('nan')):.4g} "
                   f"succd {g['succ_delta']:+.2f} "
                   f"sub {g['subopt'][0]:.3f}/{g['subopt'][1]:.3f} "
                   f"x={g['crossover']}; t: eu {c['t_means']['euclid']:.3f} "
                   f"wa {c['t_means']['wastar']:.3f} "
                   f"cpu {c['t_means']['learned_cpu']:.3f} "
                   f"gpu {c['t_means']['learned_gpu']:.3f} "
                   f"sipp {c['t_means']['sipp']:.3f}"]
    md += ["", "## Slope contrasts (paired world bootstrap; non-degenerate suites)", ""]
    for suite, out in res["slope_boot"].items():
        md += [f"- {SHORT[suite]}: " + "; ".join(
            f"{k} {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]"
            for k, v in out.items())]
    md += ["", "## R4 SIPP (succ | mean t per suite)", ""]
    for size in SIZES:
        md += [f"- N={size}: " + "; ".join(
            f"{SHORT[s]} {r4[size][s]['succ']:.2f}|{r4[size][s]['t']:.2f}s"
            for s in SUITES)]
    md += ["", "## R5 GPU table-build mean (s): " + ", ".join(
        f"N={n} {res['R5_gpu_table_s'][n]:.3f}" for n in SIZES), ""]
    md += ["## Probe (median r | MAE | bias)", ""]
    for size in SIZES:
        md += [f"- N={size}: " + "; ".join(
            f"{SHORT[s]} {res['probe'][size][s]['r']:.2f}|"
            f"{res['probe'][size][s]['mae']:.2f}|"
            f"{res['probe'][size][s]['bias']:+.2f}" for s in SUITES)]

    res["cpu_gpu_divergence"] = div
    tot_f = sum(v["found_mismatch"] for d in div.values() for v in d.values())
    tot_e = sum(v["exp_mismatch"] for d in div.values() for v in d.values())
    mx = max(v["max_abs_exp_diff"] for d in div.values() for v in d.values())
    md += ["", f"## CPU/GPU divergence: found mismatches {tot_f}, "
           f"worlds with exp diff {tot_e}, max |d exp| {mx}"]

    OUT_JSON.write_text(json.dumps(res, indent=1))
    OUT_MD.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
