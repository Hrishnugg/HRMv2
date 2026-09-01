"""C8 success-vs-budget curves from the single high-budget eval pass.

Derivation basis (see design 2026-07-23-c8-budget-curves.md): the space-time
A* expansion order is budget-independent and a map is solved at budget B iff
its recorded solve expansions E satisfy E <= B. One eval at 4x the binding
budget therefore yields exact success(B) for every B <= 4x binding by
thresholding, with no re-runs and no interpolation.

Cross-check: thresholding at the canonical binding budget must reproduce the
c8r_fresh_eval per-suite success EXACTLY (same worlds, same checkpoints,
prefix-deterministic search). A mismatch indicates a bug and fails the run.

Usage: python c8_budget_curves.py [run_dir_name] [fresh_run_dir_name]
Outputs: c8_budget_curves_output.md + c8_budget_curves.json next to this file.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_RUN = sys.argv[1] if len(sys.argv) > 1 else "c8r_budget_curves"
_FRESH = sys.argv[2] if len(sys.argv) > 2 else "c8r_fresh_eval"
RUNS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "hrm-cloud",
                                     "continuous_prm", "runs"))
RAW = os.path.join(RUNS, _RUN, "results", "continuous_prm_c8_eval_raw.csv")
FRESH_RAW = os.path.join(RUNS, _FRESH, "results", "continuous_prm_c8_eval_raw.csv")

SUITES = ["C_dyn_crossing", "C_dyn_maze", "C_dyn_maze_dense",
          "C_dyn_rooms", "C_dyn_rooms_large", "C_dyn_spiral"]
LABELS = {"C_dyn_crossing": "Crossing", "C_dyn_maze": "Maze",
          "C_dyn_maze_dense": "Dense maze", "C_dyn_rooms": "Rooms",
          "C_dyn_rooms_large": "Large rooms", "C_dyn_spiral": "Spiral"}
BINDING = {"C_dyn_crossing": 150, "C_dyn_maze": 1800, "C_dyn_maze_dense": 2500,
           "C_dyn_rooms": 1300, "C_dyn_rooms_large": 600, "C_dyn_spiral": 2500}
HIGH = {s: 4 * b for s, b in BINDING.items()}
PROVIDERS = ["euclid", "field_unet_blind", "field_unet", "oracle"]


def load_astar(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("mode") != "astar":
                continue
            rows.append(dict(
                suite=r["suite"], provider=r["provider"],
                world=int(float(r["world_index"])), budget=int(float(r["budget"])),
                found=str(r["found"]) in ("True", "true", "1"),
                expansions=int(float(r["expansions"])),
            ))
    return rows


def success_at(rows_pw, B):
    """rows_pw: {world: (found_high, expansions)}; success iff found and E <= B."""
    n = len(rows_pw)
    s = sum(1 for f, e in rows_pw.values() if f and e <= B)
    return s / n if n else float("nan")


def main():
    rows = load_astar(RAW)
    fresh = load_astar(FRESH_RAW)

    # Index: suite -> provider -> world -> (found, expansions) at the high budget.
    idx = {}
    for r in rows:
        if r["budget"] != HIGH[r["suite"]]:
            continue
        idx.setdefault(r["suite"], {}).setdefault(r["provider"], {})[r["world"]] = (
            r["found"], r["expansions"])

    # Fresh-eval success at the binding budget, for the cross-check.
    fidx = {}
    for r in fresh:
        if r["budget"] != BINDING[r["suite"]]:
            continue
        fidx.setdefault(r["suite"], {}).setdefault(r["provider"], {})[r["world"]] = r["found"]

    out = {"run": _RUN, "suites": {}}
    lines = ["# C8 success-vs-budget curves (derived from one 4x-binding pass)", ""]
    lines.append(f"Run `{_RUN}`; thresholding rule success(B) = [found and expansions <= B].")
    lines.append("")
    xcheck_fail = []
    for suite in SUITES:
        prov_map = idx.get(suite, {})
        n_worlds = len(next(iter(prov_map.values()), {}))
        b_star, b_hi = BINDING[suite], HIGH[suite]
        # Budget grid: every observed solve-expansion count plus the anchors,
        # so the step curve is exact.
        grid = {1, b_star, b_hi}
        for pw in prov_map.values():
            for f, e in pw.values():
                if f:
                    grid.add(int(e))
        grid = sorted(g for g in grid if g <= b_hi)

        curves = {}
        for prov in PROVIDERS:
            pw = prov_map.get(prov, {})
            if not pw:
                continue
            curves[prov] = {
                "budgets": grid,
                "success": [round(success_at(pw, g), 4) for g in grid],
                "success_at_binding": success_at(pw, b_star),
                "success_at_high": success_at(pw, b_hi),
            }
            # Cross-check vs fresh eval at binding (euclid + twins only; the
            # fresh run also had these exact providers).
            fw = fidx.get(suite, {}).get(prov, {})
            if fw:
                fresh_succ = sum(1 for v in fw.values() if v) / len(fw)
                if abs(fresh_succ - curves[prov]["success_at_binding"]) > 1e-9:
                    xcheck_fail.append((suite, prov, fresh_succ,
                                        curves[prov]["success_at_binding"]))

        # Euclid catch-up: smallest B <= 4x binding at which euclid reaches the
        # blind provider's binding-budget success (None if never).
        blind_at_star = curves.get("field_unet_blind", {}).get("success_at_binding")
        catch = None
        if blind_at_star is not None and "euclid" in curves:
            pw = prov_map["euclid"]
            solves = sorted(e for f, e in pw.values() if f)
            for e in solves:
                if success_at(pw, e) >= blind_at_star:
                    catch = int(e)
                    break
        out["suites"][suite] = {
            "label": LABELS[suite], "binding": b_star, "high": b_hi,
            "n_worlds": n_worlds, "curves": curves, "euclid_catchup_budget": catch,
        }
        lines.append(f"## {LABELS[suite]} (binding {b_star}, high {b_hi}, n={n_worlds})")
        for prov in PROVIDERS:
            c = curves.get(prov)
            if not c:
                continue
            lines.append(
                f"- {prov}: success {c['success_at_binding']:.2f} @ binding -> "
                f"{c['success_at_high']:.2f} @ 4x binding")
        lines.append(
            f"- euclid catch-up to blind@binding ({blind_at_star:.2f}): "
            + (f"budget {catch} ({catch/b_star:.2f}x binding)" if catch is not None
               else "not reached within 4x binding"))
        lines.append("")

    lines.append("## Binding-budget cross-check vs c8r_fresh_eval")
    if xcheck_fail:
        lines.append("**FAIL** — thresholded success != fresh-eval success:")
        for suite, prov, fs, ts in xcheck_fail:
            lines.append(f"- {LABELS[suite]} {prov}: fresh {fs:.3f} vs thresholded {ts:.3f}")
    else:
        lines.append("PASS — thresholding at the binding budget reproduces every "
                     "fresh-eval success value exactly (all suites, all providers).")
    lines.append("")

    md = os.path.join(HERE, "c8_budget_curves_output.md")
    js = os.path.join(HERE, "c8_budget_curves.json")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\n".join(lines))
    if xcheck_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
