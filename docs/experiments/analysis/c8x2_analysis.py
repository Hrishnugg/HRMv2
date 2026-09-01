"""C8-X v2 frozen analysis (design 2026-07-26 + Amendments 1-2).

From runs/c8x2_scale/ rows: per category, (1) calibrate the binding budget on
POOL-map development rows only (anchor-only, canonical rule: grid budgets
closest to targets 0.45/0.70, ties smaller, binding = smaller selected);
(2) tune w_h on the same pool-dev rows at binding (highest success, ties
smaller); (3) preregistered primary family, BH within 8 tests,
source-map-clustered bootstrap on HELD-OUT maps:
  B1 adapted(M=8) - zeroshot success > 0, per category x {lora, full};
  B2 adapted(M=8) - adapted(M=1) success > 0, per category x {lora, full}.
Scratch was added by Amendment 2 without amending the primary family; all
scratch comparisons are descriptive. A1/B3/B4 descriptive as registered.
Solve-at-budget derives from thresholding recorded solve expansions
(prefix-deterministic search; BIGB = 14000).
Units: source maps; optimizer seeds (and M=1 draws) averaged within map
before resampling. Interval and test machinery: 10,000-resample
source-map-clustered percentile bootstrap; p-values are two-sided add-one
bootstrap probabilities (2*min(P(mean<=0), P(mean>=0))); BH within the
declared 8-test family.

Amendment 3 support: when the balanced draws exist (M=1 draws 0-7, M=2
draws 0-3), the script additionally reports the repaired B2 (M=8 minus the
mean over ALL EIGHT M=1 draws) and a repaired BH family (B1 + repaired
B2), plus balanced dose curves (M=4 uses disjoint draws 0-1; legacy draw 2
is a subsample replicate of draw 0 and is excluded from balanced
summaries). Descriptive additions per the review: held-out matched-effort
and path-quality tables and conversion-fidelity summaries.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
RUN = ROOT / "runs" / "c8x2_scale"
OUT_JSON = HERE.parent / "c8x2_analysis.json"
OUT_MD = HERE.parent / "c8x2_analysis_output.md"

CALIB_GRID = [150, 250, 400, 600, 900, 1300, 1800, 2500, 3500]
WEIGHTS = [1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
TARGETS = (0.45, 0.70)
CATS = ("street", "dao")
PRIMARY_METHODS = ("lora", "full")
ALL_METHODS = ("lora", "full", "scratch")
NBOOT = 10_000
RNG = np.random.default_rng(20260727)


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["found"] = r["found"] == "True"
        r["expansions"] = int(r["expansions"])
        r["slot"] = int(r["slot"])
        r["arrival"] = int(r["arrival"])
        r["optimal_arrival"] = int(r["optimal_arrival"])
    return rows


def solved_at(row, budget):
    return bool(row["found"] and row["expansions"] <= budget)


def tagged(cat, m, draw, method, seed):
    return f"{cat}_M{m}_d{draw}_{method}_s{seed}"


def boot_ci_p(per_map: np.ndarray):
    """Mean of per-map values; 10k map bootstrap; percentile CI + two-sided p."""
    n = len(per_map)
    idx = RNG.integers(0, n, size=(NBOOT, n))
    means = per_map[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_le = (np.sum(means <= 0.0) + 1) / (NBOOT + 1)
    p_ge = (np.sum(means >= 0.0) + 1) / (NBOOT + 1)
    p = float(min(1.0, 2 * min(p_le, p_ge)))
    return float(per_map.mean()), float(lo), float(hi), p


def bh(pvals: dict):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    q = {}
    prev = 1.0
    for rank in range(m, 0, -1):
        name, p = items[rank - 1]
        val = min(prev, p * m / rank)
        q[name] = val
        prev = val
    return q


def main():
    man = {c: json.loads((RUN / f"instances_{c}.json").read_text())
           for c in CATS}
    zs = defaultdict(list)  # cat -> rows
    for c in CATS:
        for p in sorted(RUN.glob(f"zs_{c}_*.csv")):
            zs[c].extend(load_rows(p))
    ad = {}  # tag -> rows
    for p in sorted(RUN.glob("ad_*.csv")):
        ad[p.stem[3:]] = load_rows(p)

    res = {"design": "2026-07-26-c8-movingai-scale-transfer + A1 + A2",
           "cats": {}}
    md = ["# C8-X v2 analysis output", ""]

    for c in CATS:
        m = man[c]
        pool, usable = list(m["pool"]), list(m["usable"])
        heldout = [x for x in usable if x not in pool]
        rows = zs[c]
        # sanity: arms present, roles consistent
        arms = sorted({r["arm"] for r in rows})
        assert "euclid" in arms and "zeroshot" in arms, arms
        dev = [r for r in rows if r["phase"] == "dev"]
        assert all(r["role"] == "pool" for r in dev)

        # (1) binding budget from pool-dev anchor rows
        dev_eu = [r for r in dev if r["arm"] == "euclid"]
        succ_at = {b: float(np.mean([solved_at(r, b) for r in dev_eu]))
                   for b in CALIB_GRID}
        chosen = []
        for tgt in TARGETS:
            best = min(CALIB_GRID,
                       key=lambda b: (abs(succ_at[b] - tgt), b))
            chosen.append(best)
        binding = min(chosen)

        # (2) tune w on pool-dev at binding
        wa_succ = {}
        for w in WEIGHTS:
            dw = [r for r in dev if r["arm"] == f"wastar_{w:g}"]
            wa_succ[w] = float(np.mean([solved_at(r, binding) for r in dw]))
        w_star = max(WEIGHTS, key=lambda w: (wa_succ[w], -w))

        def map_succ(rws, arm, maps, budget=binding):
            """arm success per map over eval rows (mean over instances)."""
            per = {}
            for name in maps:
                sub = [r for r in rws if r["map"] == name
                       and r["phase"] == "eval" and r["arm"] == arm]
                assert sub, (c, arm, name)
                per[name] = float(np.mean([solved_at(r, budget) for r in sub]))
            return per

        zs_h = map_succ(rows, "zeroshot", heldout)
        eu_h = map_succ(rows, "euclid", heldout)
        wa_h = map_succ(rows, f"wastar_{w_star:g}", heldout)

        def adapted_map_succ(mm, draws, method, maps):
            """seed- (and draw-) averaged adapted success per map."""
            per = defaultdict(list)
            for d in draws:
                for s in (0, 1):
                    t = tagged(c, mm, d, method, s)
                    sub = ad[t]
                    for name in maps:
                        rs = [r for r in sub if r["map"] == name]
                        assert rs, (t, name)
                        per[name].append(
                            float(np.mean([solved_at(r, binding)
                                           for r in rs])))
            return {k: float(np.mean(v)) for k, v in per.items()}

        def avail_draws(mm, method):
            return [d for d in range(8)
                    if (RUN / f"ad_{tagged(c, mm, d, method, 0)}.csv").exists()]

        # primary family pieces
        prim = {}
        desc = {}
        for method in ALL_METHODS:
            a8 = adapted_map_succ(8, [0], method, heldout)
            a1 = adapted_map_succ(1, [0, 1, 2], method, heldout)
            d_b1 = np.array([a8[n] - zs_h[n] for n in heldout])
            d_b2 = np.array([a8[n] - a1[n] for n in heldout])
            b1 = boot_ci_p(d_b1)
            b2 = boot_ci_p(d_b2)
            entry = {
                "a8_mean": float(np.mean(list(a8.values()))),
                "a1_mean": float(np.mean(list(a1.values()))),
                "zeroshot_mean": float(np.mean(list(zs_h.values()))),
                "B1_delta": b1[0], "B1_ci": [b1[1], b1[2]], "B1_p": b1[3],
                "B2_delta": b2[0], "B2_ci": [b2[1], b2[2]], "B2_p": b2[3],
                "a8_per_map": a8, "a1_per_map": a1,
            }
            m1_draws = avail_draws(1, method)
            if len(m1_draws) == 8:  # Amendment 3 balanced repair present
                a1b = adapted_map_succ(1, m1_draws, method, heldout)
                d_b2b = np.array([a8[n] - a1b[n] for n in heldout])
                b2b = boot_ci_p(d_b2b)
                entry.update({
                    "a1_balanced_mean": float(np.mean(list(a1b.values()))),
                    "B2bal_delta": b2b[0], "B2bal_ci": [b2b[1], b2b[2]],
                    "B2bal_p": b2b[3]})
            if method in PRIMARY_METHODS:
                prim[method] = entry
            else:
                desc[method] = entry

        # descriptive: M-dose curve on held-out; B4 vs anchor/WA*; B3 gap
        # (balanced draws where present: M=1 all avail, M=2 0-3, M=4 0-1)
        dose = {}
        for method in ALL_METHODS:
            dose[method] = {}
            for mm, want in ((1, list(range(8))), (2, [0, 1, 2, 3]),
                             (4, [0, 1]), (8, [0])):
                draws = [d for d in want if d in avail_draws(mm, method)]
                amap = adapted_map_succ(mm, draws, method, heldout)
                dose[method][mm] = float(np.mean(list(amap.values())))

        # descriptive effort + path quality on held-out maps (review item)
        def arm_recs(arm_rows, name):
            return [r for r in arm_rows if r["map"] == name
                    and r["phase"] == "eval"]

        def zs_arm_rows(arm):
            return [r for r in rows if r["arm"] == arm]

        def ad_rows(mm, method):
            out = []
            for d in ([0] if mm == 8 else avail_draws(mm, method)):
                for s in (0, 1):
                    out.extend(ad[tagged(c, mm, d, method, s)])
            return out

        def effort_path(rows_a, rows_b):
            """Per held-out map: median exp ratio a/b and mean subopt pair on
            instances solved by both at binding; then map-level summaries."""
            ratios, sub_a, sub_b = [], [], []
            for name in heldout:
                ra = arm_recs(rows_a, name)
                rb = arm_recs(rows_b, name)
                by_seed_b = defaultdict(list)
                for r in rb:
                    by_seed_b[r["seed"]].append(r)
                m_ratios, m_sa, m_sb = [], [], []
                for r in ra:
                    if not solved_at(r, binding):
                        continue
                    for r2 in by_seed_b.get(r["seed"], []):
                        if solved_at(r2, binding):
                            m_ratios.append(r["expansions"] / r2["expansions"])
                            if r["optimal_arrival"] > 0:
                                m_sa.append(r["arrival"]
                                            / r["optimal_arrival"])
                                m_sb.append(r2["arrival"]
                                            / r2["optimal_arrival"])
                if m_ratios:
                    ratios.append(float(np.median(m_ratios)))
                if m_sa:
                    sub_a.append(float(np.mean(m_sa)))
                    sub_b.append(float(np.mean(m_sb)))
            out = {"n_maps": len(ratios)}
            if len(ratios) >= 2:
                m, lo, hi = (float(np.median(ratios)),) + tuple(
                    np.percentile(np.median(
                        np.array(ratios)[RNG.integers(0, len(ratios),
                                                      (NBOOT, len(ratios)))],
                        axis=1), [2.5, 97.5]))
                out["ratio"] = [m, float(lo), float(hi)]
            if sub_a:
                out["subopt"] = [float(np.mean(sub_a)), float(np.mean(sub_b))]
            return out

        eff = {}
        eu_rows = zs_arm_rows("euclid")
        wa_rows = zs_arm_rows(f"wastar_{w_star:g}")
        eff["zeroshot_vs_anchor"] = effort_path(zs_arm_rows("zeroshot"),
                                                eu_rows)
        eff["zeroshot_vs_wastar"] = effort_path(zs_arm_rows("zeroshot"),
                                                wa_rows)
        for method in PRIMARY_METHODS:
            eff[f"a8_{method}_vs_anchor"] = effort_path(
                ad_rows(8, method), eu_rows)
            eff[f"a8_{method}_vs_wastar"] = effort_path(
                ad_rows(8, method), wa_rows)

        # conversion fidelity: pool vs held-out (review item)
        def fid(names):
            ff = [m["maps"][n]["free_frac"] for n in names]
            cc = [m["maps"][n]["components"] for n in names]
            return {"free_frac": [float(np.mean(ff)), float(np.min(ff)),
                                  float(np.max(ff))],
                    "components": [float(np.mean(cc)), int(np.min(cc)),
                                   int(np.max(cc))]}

        failed = [n for n, v in m["maps"].items() if not v["usable"]]
        fidelity = {"pool": fid(pool), "heldout": fid(heldout),
                    "failed_maps": failed}
        b4 = {}
        for method in PRIMARY_METHODS:
            a8 = prim[method]["a8_per_map"]
            va = boot_ci_p(np.array([a8[n] - eu_h[n] for n in heldout]))
            vw = boot_ci_p(np.array([a8[n] - wa_h[n] for n in heldout]))
            b4[method] = {"vs_anchor": va[:3], "vs_wastar": vw[:3]}
        b3 = {}
        for method in PRIMARY_METHODS:
            a8_pool = adapted_map_succ(8, [0], method, pool)
            zs_pool = map_succ(rows, "zeroshot", pool)
            b3[method] = {
                "pool_a8": float(np.mean(list(a8_pool.values()))),
                "pool_zs": float(np.mean(list(zs_pool.values()))),
                "heldout_a8": prim[method]["a8_mean"],
                "heldout_zs": prim[method]["zeroshot_mean"],
            }

        # A1 descriptive: zero-shot boundary at scale (held-out maps)
        d_anchor = boot_ci_p(np.array([zs_h[n] - eu_h[n] for n in heldout]))
        d_wa = boot_ci_p(np.array([zs_h[n] - wa_h[n] for n in heldout]))
        a1_desc = {
            "n_heldout_maps": len(heldout), "n_pool": len(pool),
            "binding": binding, "w_star": w_star,
            "anchor_dev_succ_at_binding": succ_at[binding],
            "zeroshot_mean": float(np.mean(list(zs_h.values()))),
            "euclid_mean": float(np.mean(list(eu_h.values()))),
            "wastar_mean": float(np.mean(list(wa_h.values()))),
            "delta_vs_anchor": d_anchor[:3], "delta_vs_wastar": d_wa[:3],
            "n_target": m.get("n_target"),
            "per_map": {n: {"zs": zs_h[n], "eu": eu_h[n], "wa": wa_h[n]}
                        for n in heldout},
        }

        res["cats"][c] = {"A1": a1_desc, "primary": prim,
                          "scratch_descriptive": desc, "dose": dose,
                          "B3": b3, "B4": b4, "effort_path": eff,
                          "fidelity": fidelity}

    # BH across the frozen 8-test family
    pvals = {}
    for c in CATS:
        for method in PRIMARY_METHODS:
            pvals[f"B1_{c}_{method}"] = res["cats"][c]["primary"][method]["B1_p"]
            pvals[f"B2_{c}_{method}"] = res["cats"][c]["primary"][method]["B2_p"]
    qvals = bh(pvals)
    res["primary_family"] = {k: {"p": pvals[k], "q": qvals[k]}
                             for k in sorted(pvals)}

    # Amendment 3 repaired family (B1 unchanged + balanced B2), when present
    have_bal = all("B2bal_p" in res["cats"][c]["primary"][me]
                   for c in CATS for me in PRIMARY_METHODS)
    if have_bal:
        pv2 = {}
        for c in CATS:
            for me in PRIMARY_METHODS:
                e = res["cats"][c]["primary"][me]
                pv2[f"B1_{c}_{me}"] = e["B1_p"]
                pv2[f"B2bal_{c}_{me}"] = e["B2bal_p"]
        qv2 = bh(pv2)
        res["repaired_family"] = {k: {"p": pv2[k], "q": qv2[k]}
                                  for k in sorted(pv2)}

    # markdown
    for c in CATS:
        r = res["cats"][c]
        a = r["A1"]
        md += [f"## {c} (binding {a['binding']}, tuned w_h={a['w_star']:g}, "
               f"{a['n_heldout_maps']} held-out maps, N_target {a['n_target']})",
               "",
               f"- A1 zero-shot (held-out): learned {a['zeroshot_mean']:.3f} "
               f"vs anchor {a['euclid_mean']:.3f} vs WA* {a['wastar_mean']:.3f}; "
               f"d_anchor {a['delta_vs_anchor'][0]:+.3f} "
               f"[{a['delta_vs_anchor'][1]:+.3f},{a['delta_vs_anchor'][2]:+.3f}], "
               f"d_WA* {a['delta_vs_wastar'][0]:+.3f} "
               f"[{a['delta_vs_wastar'][1]:+.3f},{a['delta_vs_wastar'][2]:+.3f}]"]
        for method in PRIMARY_METHODS:
            e = r["primary"][method]
            q1 = qvals[f"B1_{c}_{method}"]
            q2 = qvals[f"B2_{c}_{method}"]
            md += [f"- {method}: a8 {e['a8_mean']:.3f} a1 {e['a1_mean']:.3f} "
                   f"zs {e['zeroshot_mean']:.3f}; "
                   f"B1 {e['B1_delta']:+.3f} [{e['B1_ci'][0]:+.3f},"
                   f"{e['B1_ci'][1]:+.3f}] p={e['B1_p']:.4f} q={q1:.4f}; "
                   f"B2 {e['B2_delta']:+.3f} [{e['B2_ci'][0]:+.3f},"
                   f"{e['B2_ci'][1]:+.3f}] p={e['B2_p']:.4f} q={q2:.4f}"]
        e = r["scratch_descriptive"]["scratch"]
        md += [f"- scratch (descriptive): a8 {e['a8_mean']:.3f} "
               f"a1 {e['a1_mean']:.3f}; B1-style {e['B1_delta']:+.3f} "
               f"[{e['B1_ci'][0]:+.3f},{e['B1_ci'][1]:+.3f}]"]
        md += [f"- dose (held-out mean succ by M): " + "; ".join(
            f"{meth} " + " ".join(f"M{mm}={r['dose'][meth][mm]:.3f}"
                                  for mm in (1, 2, 4, 8))
            for meth in ALL_METHODS)]
        for method in PRIMARY_METHODS:
            g = r["B3"][method]
            v = r["B4"][method]
            md += [f"- {method} B3 pool-vs-heldout a8: {g['pool_a8']:.3f} vs "
                   f"{g['heldout_a8']:.3f} (zs {g['pool_zs']:.3f}/"
                   f"{g['heldout_zs']:.3f}); B4 a8-vs-anchor "
                   f"{v['vs_anchor'][0]:+.3f} [{v['vs_anchor'][1]:+.3f},"
                   f"{v['vs_anchor'][2]:+.3f}], a8-vs-WA* "
                   f"{v['vs_wastar'][0]:+.3f} [{v['vs_wastar'][1]:+.3f},"
                   f"{v['vs_wastar'][2]:+.3f}]"]
        if "B2bal_p" in r["primary"]["full"]:
            for method in PRIMARY_METHODS:
                e = r["primary"][method]
                q = res.get("repaired_family", {}).get(
                    f"B2bal_{c}_{method}", {}).get("q", float("nan"))
                md += [f"- {method} B2 BALANCED (8 draws): a1bal "
                       f"{e['a1_balanced_mean']:.3f}; delta "
                       f"{e['B2bal_delta']:+.3f} [{e['B2bal_ci'][0]:+.3f},"
                       f"{e['B2bal_ci'][1]:+.3f}] p={e['B2bal_p']:.4f} "
                       f"q={q:.4f}"]
        for k, v in r["effort_path"].items():
            ratio = v.get("ratio")
            sub = v.get("subopt")
            md += [f"- effort/path {k}: ratio "
                   + (f"{ratio[0]:.3f} [{ratio[1]:.3f},{ratio[2]:.3f}]"
                      if ratio else "n/a")
                   + (f"; subopt {sub[0]:.3f}/{sub[1]:.3f}" if sub else "")
                   + f" (n_maps {v['n_maps']})"]
        fp, fh = r["fidelity"]["pool"], r["fidelity"]["heldout"]
        md += [f"- fidelity pool free {fp['free_frac'][0]:.3f} "
               f"[{fp['free_frac'][1]:.3f}-{fp['free_frac'][2]:.3f}] "
               f"comp {fp['components'][0]:.1f}; heldout free "
               f"{fh['free_frac'][0]:.3f} "
               f"[{fh['free_frac'][1]:.3f}-{fh['free_frac'][2]:.3f}] "
               f"comp {fh['components'][0]:.1f}; failed: "
               + (", ".join(r["fidelity"]["failed_maps"]) or "none")]
        md += [""]

    OUT_JSON.write_text(json.dumps(res, indent=1))
    OUT_MD.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
