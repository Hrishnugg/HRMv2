"""Fixed-provider C8 dynamic-transfer reanalysis.

Addresses the selected-arms objection: designates ONE primary provider --
field U-Net *blind* (no future-motion input) in additive mode at each suite's
binding budget -- chosen without reference to per-suite target outcomes, and
recomputes all headline quantities at the map level from the canonical
c8_local_heavy raw rows. Also computes the aware-vs-blind direct differences
for the same fixed family, and the per-suite best arm as a secondary,
explicitly-labeled selection.

Statistics: maps are the independent unit (20 per suite). Success deltas use
10k-resample percentile bootstraps over maps. Matched-solved ratios use the
per-map ratio on jointly solved maps (median, bootstrap CI, n reported).
Deterministic: numpy default_rng(20260723).

Outputs: c8_fixed_provider_output.md and c8_fixed_provider.json (for figures).
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
_RUN = sys.argv[1] if len(sys.argv) > 1 else "c8_local_heavy"
_PREFIX = sys.argv[2] if len(sys.argv) > 2 else "c8_fixed_provider"
RAW = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "hrm-cloud", "continuous_prm", "runs",
    _RUN, "results", "continuous_prm_c8_eval_raw.csv"))
RNG = np.random.default_rng(20260723)
NBOOT = 10_000

BINDING = {"C_dyn_crossing": 150, "C_dyn_maze": 1800, "C_dyn_maze_dense": 2500,
           "C_dyn_rooms": 1300, "C_dyn_rooms_large": 600, "C_dyn_spiral": 2500}
SUITE_LABELS = {"C_dyn_crossing": "Crossing", "C_dyn_maze": "Maze",
                "C_dyn_maze_dense": "Dense maze", "C_dyn_rooms": "Rooms",
                "C_dyn_rooms_large": "Large rooms", "C_dyn_spiral": "Spiral"}
PRIMARY = "field_unet_blind"
AWARE_TWIN = "field_unet"
LEARNED = ["scalar_hrm", "scalar_hrm_blind", "scalar_onlstm", "scalar_onlstm_blind",
           "field_unet", "field_unet_blind", "field_hrm", "field_hrm_blind"]


def boot_ci(vals, stat):
    vals = np.asarray(vals, dtype=float)
    idx = RNG.integers(0, len(vals), size=(NBOOT, len(vals)))
    stats = np.array([stat(vals[row]) for row in idx])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


rows = list(csv.DictReader(open(RAW, newline="")))
# index: (suite, provider) -> world -> (found, expansions) at binding budget, astar mode
data = defaultdict(dict)
for r in rows:
    if r["mode"] != "astar":
        continue
    if int(r["budget"]) != BINDING[r["suite"]]:
        continue
    data[(r["suite"], r["provider"])][int(r["world_index"])] = (
        r["found"] == "True", float(r["expansions"]))

out_md = [f"# Fixed-provider C8 analysis: {_RUN}", "",
          f"Primary provider: `{PRIMARY}` (additive mode, binding budgets), fixed across all suites.",
          "Statistics are map-level (n=20 maps/suite; 10k bootstraps; seed 20260723).", ""]
out = {"primary": PRIMARY, "suites": {}}

out_md.append("## Primary: field U-Net blind vs Euclid")
out_md.append("| Suite | Euclid succ. | Blind U-Net succ. | Dsucc [95% CI] | matched median ratio [95% CI] | matched n |")
out_md.append("|---|---:|---:|---|---|---:|")
for suite in BINDING:
    eu = data[(suite, "euclid")]
    pr = data[(suite, PRIMARY)]
    worlds = sorted(eu)
    eu_s = np.array([1.0 if eu[w][0] else 0.0 for w in worlds])
    pr_s = np.array([1.0 if pr[w][0] else 0.0 for w in worlds])
    deltas = pr_s - eu_s
    dlo, dhi = boot_ci(deltas, np.mean)
    ratios = [pr[w][1] / eu[w][1] for w in worlds if eu[w][0] and pr[w][0]]
    if ratios:
        rmed = float(np.median(ratios))
        rlo, rhi = boot_ci(ratios, np.median)
    else:
        rmed, rlo, rhi = float("nan"), float("nan"), float("nan")
    out["suites"][suite] = {
        "label": SUITE_LABELS[suite],
        "euclid_success": float(eu_s.mean()), "primary_success": float(pr_s.mean()),
        "dsucc": float(deltas.mean()), "dsucc_ci": [dlo, dhi],
        "ratio_median": rmed, "ratio_ci": [rlo, rhi], "matched_n": len(ratios)}
    out_md.append(f"| {SUITE_LABELS[suite]} | {eu_s.mean():.2f} | {pr_s.mean():.2f} | "
                  f"{deltas.mean():+.2f} [{dlo:+.2f}, {dhi:+.2f}] | "
                  f"{rmed:.3f} [{rlo:.3f}, {rhi:.3f}] | {len(ratios)} |")

out_md.append("")
out_md.append("## Aware minus blind (field U-Net twins), map-paired")
out_md.append("| Suite | Dsucc (aware-blind) [95% CI] | D median ratio on jointly solved |")
out_md.append("|---|---|---|")
for suite in BINDING:
    aw = data[(suite, AWARE_TWIN)]
    bl = data[(suite, PRIMARY)]
    worlds = sorted(aw)
    ds = np.array([(1.0 if aw[w][0] else 0.0) - (1.0 if bl[w][0] else 0.0) for w in worlds])
    lo, hi = boot_ci(ds, np.mean)
    eu = data[(suite, "euclid")]
    joint = [w for w in worlds if aw[w][0] and bl[w][0] and eu[w][0]]
    if joint:
        dr = float(np.median([aw[w][1] / eu[w][1] for w in joint]) -
                   np.median([bl[w][1] / eu[w][1] for w in joint]))
    else:
        dr = float("nan")
    out["suites"][suite]["aware_minus_blind_succ"] = float(ds.mean())
    out["suites"][suite]["aware_minus_blind_succ_ci"] = [lo, hi]
    out["suites"][suite]["aware_minus_blind_ratio_delta"] = dr
    out_md.append(f"| {SUITE_LABELS[suite]} | {ds.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {dr:+.3f} |")

out_md.append("")
out_md.append("## Secondary: per-suite best arm (post-hoc selection, labeled as such)")
out_md.append("| Suite | Best arm | succ. | matched median ratio | matched n |")
out_md.append("|---|---|---:|---:|---:|")
for suite in BINDING:
    eu = data[(suite, "euclid")]
    worlds = sorted(eu)
    best = None
    for prov in LEARNED:
        pv = data.get((suite, prov))
        if not pv:
            continue
        succ = np.mean([1.0 if pv[w][0] else 0.0 for w in worlds])
        ratios = [pv[w][1] / eu[w][1] for w in worlds if eu[w][0] and pv[w][0]]
        if not ratios:
            continue
        med = float(np.median(ratios))
        # selection rule for the secondary table: highest success, ratio as tiebreak
        key = (-succ, med)
        if best is None or key < best[0]:
            best = (key, prov, succ, med, len(ratios))
    _, prov, succ, med, n = best
    out["suites"][suite]["best_arm"] = {"provider": prov, "success": float(succ),
                                        "ratio_median": med, "matched_n": n}
    out_md.append(f"| {SUITE_LABELS[suite]} | {prov} | {succ:.2f} | {med:.3f} | {n} |")

with open(os.path.join(HERE, _PREFIX + "_output.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(out_md) + "\n")
with open(os.path.join(HERE, _PREFIX + ".json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("\n".join(out_md))
