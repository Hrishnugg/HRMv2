#!/usr/bin/env python3
"""Serialized-world hash manifest for the frozen confirmation cohort.

For every (suite, world_index) of the canonical 50-map confirmation cohort
(seed 999999, canonical suite order), hashes the complete world identity:
generation seed, start, goal, static obstacle geometry, patroller dynamics
parameters, and the roadmap (node coordinates + adjacency). The corrected
control runners (SIPP, extended grid, wall time, probe reference) share this
enumeration code path verbatim, so instance identity holds by construction;
this manifest makes it independently checkable, with optimal-arrival
equality as a separate semantic check. Ships in the artifact package.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import continuous_prm_c8_dynamic_maps as M8MAPS
import continuous_prm_c8_dynamics_compare as M8C

HERE = Path(__file__).parent
OUT = HERE / "runs" / "c8r_world_manifest.json"
# Canonical eval order (== C8Config.eval_suites).
SUITES = ["C_dyn_maze", "C_dyn_rooms", "C_dyn_spiral",
          "C_dyn_maze_dense", "C_dyn_crossing", "C_dyn_rooms_large"]


def h(arr) -> str:
    b = np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes()
    return hashlib.sha256(b).hexdigest()


def world_record(suite, wi, world, dyn, rm) -> dict:
    obs = []
    for o in world.obstacles:
        obs.append([ord(o.kind[0]), o.cx, o.cy,
                    getattr(o, "radius", 0.0) or 0.0,
                    getattr(o, "hw", 0.0) or 0.0,
                    getattr(o, "hh", 0.0) or 0.0])
    pat = []
    for p in getattr(dyn, "circles", []):
        pat.append([p.ax, p.ay, p.bx, p.by, p.radius, p.period])
    parts = {
        "start_goal": h(np.concatenate([world.start, world.goal])),
        "obstacles": h(np.asarray(obs)),
        "patrollers": h(np.asarray(pat)) if pat else "none",
        "roadmap_points": h(rm.points),
        "roadmap_adj": hashlib.sha256(
            json.dumps([[int(j) for j, _ in row] for row in rm.adj])
            .encode()).hexdigest(),
    }
    whole = hashlib.sha256(
        "|".join(f"{k}:{v}" for k, v in sorted(parts.items())).encode()
    ).hexdigest()
    return dict(suite=suite, world_index=int(wi), world_sha256=whole, **parts)


def main():
    M8MAPS.install_c8_dynamic_maps()
    cfg = M8C.C8Config(seed=999999, eval_worlds=50)
    cfg = M8C.apply_scale_preset(cfg)
    cfg.eval_worlds = 50
    records = []
    for si, suite in enumerate(SUITES):
        for wi, world, dyn, rm in M8C.iter_dynamic_worlds(suite, si, cfg, 50):
            records.append(world_record(suite, wi, world, dyn, rm))
        print(f"[manifest] {suite}: done ({len(records)} total)", flush=True)
    with open(OUT, "w") as f:
        json.dump(dict(cohort_seed=999999, suite_order=SUITES,
                       n_instances=len(records), records=records), f, indent=1)
    print(f"[manifest] wrote {OUT} ({len(records)} instances)", flush=True)


if __name__ == "__main__":
    main()
