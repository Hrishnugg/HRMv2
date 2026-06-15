#!/usr/bin/env python3
"""Scan existing eval_agg JSONs and report, per (model, suite), whether success
improves from B500 -> B2000. Suites that are flat are candidates for dropping
B2000 (the most expensive budget). Reads a local mirror of the Modal volume's
results/eval_agg directory.

eval_agg JSON schema (from residual_tasklora_v2.py ~line 2731): each file is
{"row": {"model","suite","budget","alpha","episodes","success_rate",
         "avg_expansions", ...}, "metric_sums": {...}, "diag": {...},
 "complete": true}

Usage:
  python hrm-cloud/analyze_budget_invariance.py --agg-dir /path/to/results/eval_agg
  python hrm-cloud/analyze_budget_invariance.py --agg-dir DIR --lo 200 --hi 500
"""
import argparse, glob, json, os
from collections import defaultdict


def _extract(d):
    # Real schema nests under "row"; fall back to top-level for forward-compat.
    r = d.get("row", d)
    try:
        return (str(r["model"]), str(r["suite"]), int(r["episodes"]),
                int(r["budget"]), float(r["success_rate"]),
                float(r.get("avg_expansions", 0.0)))
    except (KeyError, TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg-dir", required=True, help="dir of eval_agg/*.json")
    ap.add_argument("--eps", type=float, default=0.02,
                    help="success delta below which budgets are 'flat'")
    ap.add_argument("--lo", type=int, default=500, help="lower budget to compare")
    ap.add_argument("--hi", type=int, default=2000, help="higher budget to compare")
    args = ap.parse_args()

    # key (model, suite, episodes) -> {budget: (success_rate, avg_expansions)}
    rows = defaultdict(dict)
    n_files = n_parsed = 0
    for path in glob.glob(os.path.join(args.agg_dir, "**", "*.json"), recursive=True):
        n_files += 1
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        ex = _extract(d)
        if ex is None:
            continue
        model, suite, episodes, budget, succ, exps = ex
        rows[(model, suite, episodes)][budget] = (succ, exps)
        n_parsed += 1

    print(f"[scan] {n_files} files, {n_parsed} parsed rows, "
          f"{len(rows)} (model,suite,episodes) groups")

    flat, helps = [], []
    for (model, suite, episodes), by_b in sorted(rows.items()):
        if args.lo in by_b and args.hi in by_b:
            s_lo, e_lo = by_b[args.lo]
            s_hi, e_hi = by_b[args.hi]
            delta = s_hi - s_lo
            rec = (model, suite, episodes, s_lo, s_hi, delta, e_lo, e_hi)
            (flat if delta <= args.eps else helps).append(rec)

    print(f"\n=== B{args.hi} HELPS (keep) - success gain > {args.eps} ===")
    for model, suite, ep, s_lo, s_hi, d, e_lo, e_hi in sorted(helps, key=lambda r: -r[5]):
        print(f"  {suite:28s} {model:32s} ep={ep} "
              f"B{args.lo}={s_lo:.3f} B{args.hi}={s_hi:.3f} (+{d:.3f}) "
              f"exp {e_lo:.0f}->{e_hi:.0f}")
    print(f"\n=== FLAT: B{args.hi} ~= B{args.lo} (candidate to drop B{args.hi}) ===")
    for model, suite, ep, s_lo, s_hi, d, e_lo, e_hi in flat:
        print(f"  {suite:28s} {model:32s} ep={ep} "
              f"B{args.lo}={s_lo:.3f} B{args.hi}={s_hi:.3f} ({d:+.3f})")

    helped_suites = {suite for _, suite, *_ in helps}
    flat_suites = {suite for _, suite, *_ in flat}
    safe_to_drop = sorted(flat_suites - helped_suites)
    print(f"\nSuites flat for ALL models with both budgets "
          f"(safe to drop B{args.hi}): {safe_to_drop}")


if __name__ == "__main__":
    main()
