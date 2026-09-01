"""C14-R analysis: independent world-set replicates (frozen design 2026-07-25).

Inputs: runs/c14r_draw{2,3}/results/continuous_prm_c14r_eval_raw.csv
(astar rows at binding budgets; methods euclid / zero_shot / lora / full_ft /
scratch; one optimization seed per arm).

Per (draw, domain, N, coverage, method): map-level success delta vs the frozen
zero-shot source with a 10k map-bootstrap CI; plus the direct
distributed-minus-concentrated paired contrast per (draw, domain, N, method).
Preregistered readouts R1-R4 evaluated verbatim. Original C14 cells = draw 1
(quoted from c14_analysis.json for context, not recomputed).
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "hrm-cloud",
                                     "continuous_prm", "runs"))
BINDING = {"static": 140, "dynamic": 2500}
DRAWS = [2, 3]
BOOT, SEED = 10_000, 20260725


def load(draw):
    path = os.path.join(RUNS, f"c14r_draw{draw}", "results",
                        "continuous_prm_c14r_eval_raw.csv")
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("mode") != "astar":
                continue
            dom = r["domain"]
            if int(float(r["budget"])) != BINDING[dom]:
                continue
            rows.append(dict(
                domain=dom, method=r["method"],
                N=int(float(r["N"])) if r["N"] not in ("", None) else -1,
                div=r["diversity"],
                world=int(float(r["world_index"])),
                found=str(r["found"]) in ("True", "true", "1")))
    return rows


def boot_ci(vals, rng):
    vals = np.asarray(vals, dtype=float)
    reps = [float(np.mean(vals[rng.integers(0, len(vals), len(vals))]))
            for _ in range(BOOT)]
    return (float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5)))


def main():
    out = {"draws": {}}
    lines = ["# C14-R: independent world-set replicates", ""]
    verdict_bits = []
    for draw in DRAWS:
        try:
            rows = load(draw)
        except FileNotFoundError:
            print(f'[c14r] draw {draw} results not present yet; skipping')
            continue
        res = {}
        for dom in ("static", "dynamic"):
            dr = [r for r in rows if r["domain"] == dom]
            zs = {r["world"]: r["found"] for r in dr if r["method"] == "zero_shot"}
            cells = defaultdict(dict)
            for r in dr:
                if r["method"] in ("lora", "full_ft", "scratch"):
                    cells[(r["N"], r["div"], r["method"])][r["world"]] = r["found"]
            for (N, div, method), arm in sorted(cells.items()):
                worlds = sorted(set(arm) & set(zs))
                d = [float(arm[w]) - float(zs[w]) for w in worlds]
                rng = np.random.default_rng(SEED + draw)
                res[f"{dom}|{N}|{div}|{method}"] = dict(
                    dsucc=float(np.mean(d)), ci=boot_ci(d, rng), n=len(worlds))
            # direct dist - conc paired contrast
            for N in sorted({k[0] for k in cells}):
                for method in ("full_ft", "lora", "scratch"):
                    kc, kd = (N, "conc", method), (N, "dist", method)
                    if kc in cells and kd in cells:
                        worlds = sorted(set(cells[kc]) & set(cells[kd]))
                        d = [float(cells[kd][w]) - float(cells[kc][w]) for w in worlds]
                        rng = np.random.default_rng(SEED + draw)
                        res[f"{dom}|{N}|dist-conc|{method}"] = dict(
                            delta=float(np.mean(d)), ci=boot_ci(d, rng), n=len(worlds))
        out["draws"][draw] = res

        lines.append(f"## Draw {draw}")
        lines.append("")
        for key in sorted(res):
            v = res[key]
            if "dist-conc" in key:
                lines.append(f"- {key}: {v['delta']:+.3f} "
                             f"[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] (n={v['n']})")
            else:
                lines.append(f"- {key}: {v['dsucc']:+.3f} "
                             f"[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
        lines.append("")

        # Preregistered readouts
        def cell(dom, N, div, m):
            return out["draws"][draw].get(f"{dom}|{N}|{div}|{m}")

        r1 = all(cell("dynamic", 1024, "conc", m) and
                 cell("dynamic", 1024, "conc", m)["ci"][1] < 0
                 for m in ("full_ft", "scratch"))
        r2c = cell("dynamic", 1024, "dist", "full_ft")
        r2 = bool(r2c and r2c["dsucc"] >= 0 and r2c["ci"][0] > -1e-9 or
                  (r2c and r2c["dsucc"] >= 0 and r2c["ci"][1] >= 0))
        r2 = bool(r2c and r2c["dsucc"] >= 0 and not (r2c["ci"][1] < 0))
        r3_conc_bad = all(cell("static", 256, "conc", m) and
                          cell("static", 256, "conc", m)["ci"][1] < 0
                          for m in ("full_ft", "scratch"))
        lora_cells = [v for k, v in out["draws"][draw].items()
                      if k.endswith("|lora") and "dist-conc" not in k]
        r4 = all(not (v["ci"][1] < 0) for v in lora_cells)
        verdict_bits.append((draw, r1, r2, r3_conc_bad, r4))

    lines.append("## Preregistered readouts")
    lines.append("")
    for draw, r1, r2, r3, r4 in verdict_bits:
        lines.append(f"- draw {draw}: R1 (dynamic conc N=1024 FT+scratch harmful CI) "
                     f"{'PASS' if r1 else 'FAIL'}; R2 (dynamic dist N=1024 FT rescued) "
                     f"{'PASS' if r2 else 'FAIL'}; R3 (static conc N=256 harmful) "
                     f"{'PASS' if r3 else 'FAIL'}; R4 (no significant LoRA drop) "
                     f"{'PASS' if r4 else 'FAIL'}")

    with open(os.path.join(HERE, "c14r_analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    md = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "c14r_analysis_output.md"), "w",
              encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
