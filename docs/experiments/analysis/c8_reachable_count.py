"""Exact reachable-label recount for the three C8 dynamic training pipelines.

The paper (v4 and earlier) reported 1,129,536 "supervised samples, masked to
reachable" for the canonical dynamic run. But 1,129,536 = 53 x 192 x 111 is
exactly the UNMASKED node-time slot count (53 usable worlds, 192 roadmap
nodes, t_max+1 = 111 time steps): the training loss masks to states with a
finite backward space-time Dijkstra value (`reachable = isfinite(hstar) &
(hstar < 1e29)`), so the true supervised-label count is strictly smaller.

This script re-runs the deterministic collection stage for each training
pipeline (no training, no checkpoints touched) and counts the mask exactly.

Pipelines (from runs/*/train_manifest.json):
  canonical (seed 1234): train_worlds=24/suite -> 53 usable (17/19/17)
  seed 2001:             train_worlds=64/suite -> 139 usable (41/50/48)
  seed 2002:             train_worlds=64/suite -> 149 usable (40/54/55)

Correctness gate: the recount must reproduce each run's per-suite usable-world
counts exactly, or the script aborts without writing outputs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "hrm-cloud" / "continuous_prm"))

import continuous_prm_c8_dynamics_compare as C8  # noqa: E402

# Same install sequence as the training runner's main(): registers the six
# dynamic suites (composing on the C7-hard and C5-hard runtimes) so
# C.build_anchor_specs() can resolve the training families.
C8.M8.install_c8_dynamic_maps()

PIPELINES = [
    ("canonical_1234", 1234, 24,
     {"C_dyn_maze": 17, "C_dyn_rooms": 19, "C_dyn_spiral": 17}),
    ("seed2001", 2001, 64,
     {"C_dyn_maze": 41, "C_dyn_rooms": 50, "C_dyn_spiral": 48}),
    ("seed2002", 2002, 64,
     {"C_dyn_maze": 40, "C_dyn_rooms": 54, "C_dyn_spiral": 55}),
]


def main() -> None:
    results = {}
    for name, seed, train_worlds, expected in PIPELINES:
        t0 = time.time()
        cfg = C8.C8Config(seed=seed, train_worlds=train_worlds, cpu=True)
        labelsets, counts = C8._collect_labelsets(cfg)
        counts = dict(counts)
        if counts != expected:
            raise SystemExit(
                f"ABORT {name}: recount usable counts {counts} != "
                f"manifest {expected} -- config mismatch, numbers unusable"
            )

        per_world = []
        for ls in labelsets:
            mask = np.asarray(ls["reachable"], dtype=bool)
            per_world.append({
                "suite": ls["suite"],
                "world_seed": int(ls["seed"]),
                "n_nodes": int(mask.shape[0]),
                "t_slots": int(mask.shape[1]),
                "slots": int(mask.size),
                "reachable": int(mask.sum()),
            })

        suites = sorted({w["suite"] for w in per_world})
        by_suite = {
            s: {
                "worlds": sum(1 for w in per_world if w["suite"] == s),
                "slots": sum(w["slots"] for w in per_world if w["suite"] == s),
                "reachable": sum(w["reachable"] for w in per_world if w["suite"] == s),
            }
            for s in suites
        }
        reach = [w["reachable"] for w in per_world]
        total_slots = sum(w["slots"] for w in per_world)
        total_reach = sum(reach)
        results[name] = {
            "seed": seed,
            "train_worlds_per_suite": train_worlds,
            "usable_worlds": len(per_world),
            "usable_per_suite": counts,
            "all_worlds_192_nodes": all(w["n_nodes"] == 192 for w in per_world),
            "all_worlds_111_slots": all(w["t_slots"] == 111 for w in per_world),
            "total_slots": total_slots,
            "total_reachable": total_reach,
            "reachable_fraction": total_reach / total_slots,
            "per_world_reachable_min": int(min(reach)),
            "per_world_reachable_mean": float(np.mean(reach)),
            "per_world_reachable_max": int(max(reach)),
            "by_suite": by_suite,
            "per_world": per_world,
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(f"[recount] {name}: worlds={len(per_world)} slots={total_slots} "
              f"reachable={total_reach} ({100*total_reach/total_slots:.1f}%) "
              f"in {results[name]['elapsed_s']}s", flush=True)

    out_json = HERE / "c8_reachable_count.json"
    out_json.write_text(json.dumps(results, indent=1))

    lines = [
        "# Reachable supervised-label recount (C8 dynamic training pipelines)",
        "",
        "Slots = usable worlds x 192 nodes x 111 time steps (all worlds verified "
        "at exactly 192 nodes and t_max=110). Reachable = finite backward "
        "space-time Dijkstra value (the training loss mask).",
        "",
        "| Pipeline | Worlds (mz/rm/sp) | Slots | Reachable | % | Per-world reachable (min/mean/max) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        c = r["usable_per_suite"]
        lines.append(
            f"| {name} | {r['usable_worlds']} "
            f"({c['C_dyn_maze']}/{c['C_dyn_rooms']}/{c['C_dyn_spiral']}) "
            f"| {r['total_slots']:,} | {r['total_reachable']:,} "
            f"| {100*r['reachable_fraction']:.1f}% "
            f"| {r['per_world_reachable_min']:,} / {r['per_world_reachable_mean']:,.0f} / "
            f"{r['per_world_reachable_max']:,} |"
        )
    lines += [
        "",
        "Gate: per-suite usable-world counts reproduce each run's "
        "train_manifest.json exactly (asserted at runtime).",
    ]
    (HERE / "c8_reachable_count_output.md").write_text("\n".join(lines) + "\n")
    print("[recount] wrote c8_reachable_count.json / c8_reachable_count_output.md",
          flush=True)


if __name__ == "__main__":
    main()
