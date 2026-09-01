#!/usr/bin/env python3
"""External benchmark: dynamic zero-shot on MovingAI-derived worlds.

Frozen design: docs/experiments/continuous/c08/design/2026-07-25-c8-movingai-external.md

Pipeline: MovingAI .map (street: Berlin/Boston/Paris 256x256; dao: den312d/
den520d/brc202d) -> majority-coarsened 64x64 occupancy over the unit square ->
axis-aligned rectangle obstacles (per-row run merging) -> standard World ->
canonical maze-family patrollers placed by the frozen generator rules ->
evaluated with the untouched pipeline (192-node k=7 PRM, space-time A*,
binding-budget protocol).

Arms: Euclidean anchor; weighted A* over the canonical grid; the fixed blind
U-Net (checkpoint untouched). Every search runs once at BIGB = 4x the largest
grid budget; success at any budget B is derived exactly by thresholding
recorded solve expansions (prefix determinism, as in the budget curves).
Calibration (canonical closest-target rule, binding = lower selected budget)
and weighted-A* tuning use only development instances.

Read-only with respect to all frozen artifacts. CPU only.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import continuous_prm_c8_dynamic_maps as M8MAPS
import continuous_prm_c8_dynamics_compare as M8C
import continuous_prm_c6_heatmap_value_field as C6
import continuous_prm_common as C
import continuous_prm_dynamic_providers as DP
import continuous_prm_spacetime as ST

HERE = Path(__file__).parent
OUT = HERE / "runs" / "c8r_movingai"
MAPS_DIR = OUT / "maps"
GROUPS = {
    "street": ["Berlin_0_256.map", "Boston_0_256.map", "Paris_0_256.map"],
    "dao": ["den312d.map", "den520d.map", "brc202d.map"],
}
GRID = 64
CALIB_GRID = [150, 250, 400, 600, 900, 1300, 1800, 2500, 3500]
WEIGHTS = [1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
BIGB = 14000  # 4x the largest grid budget; threshold for any lower budget
N_DEV, N_EVAL = 10, 25
DEV_SEED0, EVAL_SEED0 = 30_000_000, 31_000_000
BLOCKED = set("@OTW")  # octile: trees and water blocked for ground agents

COLS = ["group", "phase", "instance", "map", "seed", "arm", "solved_bigb",
        "expansions", "arrival", "optimal_arrival"]


def parse_map(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    h = int(lines[1].split()[1]); w = int(lines[2].split()[1])
    rows = lines[4:4 + h]
    g = np.zeros((h, w), dtype=bool)
    for i, row in enumerate(rows):
        for j, ch in enumerate(row[:w]):
            g[i, j] = ch in BLOCKED
    return g


def coarsen(g: np.ndarray, n: int = GRID) -> np.ndarray:
    h, w = g.shape
    out = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            r0, r1 = (i * h) // n, max((i * h) // n + 1, ((i + 1) * h) // n)
            c0, c1 = (j * w) // n, max((j * w) // n + 1, ((j + 1) * w) // n)
            out[i, j] = g[r0:r1, c0:c1].mean() > 0.5
    return out


def rects_from_grid(occ: np.ndarray) -> List[C.Obstacle]:
    n = occ.shape[0]
    cell = 1.0 / n
    rects: List[C.Obstacle] = []
    for i in range(n):
        j = 0
        while j < n:
            if occ[i, j]:
                j0 = j
                while j < n and occ[i, j]:
                    j += 1
                cx = (j0 + j) / 2.0 * cell
                cy = (i + 0.5) * cell
                rects.append(C.Obstacle("rect", cx, cy,
                                        hw=(j - j0) / 2.0 * cell, hh=cell / 2.0))
            else:
                j += 1
    return rects


def _point_free(p: np.ndarray, rects: List[C.Obstacle], margin: float = 0.01) -> bool:
    for o in rects:
        if (abs(p[0] - o.cx) <= o.hw + margin and
                abs(p[1] - o.cy) <= o.hh + margin):
            return False
    return True


def build_instance(group: str, occ_by_map: Dict[str, np.ndarray], seed: int):
    """Deterministic instance: map round-robin by seed, start/goal rejection
    sampling, canonical usability rules. Returns (world, dyn, rm, mapname) or None."""
    import random
    names = GROUPS[group]
    mapname = names[seed % len(names)]
    occ = occ_by_map[mapname]
    rects = rects_from_grid(occ)
    rng = random.Random(seed)
    start = goal = None
    for _ in range(500):
        a = np.array([rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.98)])
        b = np.array([rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.98)])
        if (np.linalg.norm(a - b) >= 0.45 and _point_free(a, rects)
                and _point_free(b, rects)):
            start, goal = a, b
            break
    if start is None:
        return None
    world = C.World(spec_name=f"movingai_{group}", side_len=1.0, obstacles=rects,
                    start=start, goal=goal, descriptor=np.zeros(8),
                    meta={"seed": seed, "map": mapname, "mode": "external"})
    rm = C.build_prm(world, C.RoadmapConfig(n_nodes=192, k_neighbors=7), seed=seed + 17)
    if rm is None or not rm.connected_to_goal[0]:
        return None
    dyn = M8MAPS._place_patrollers(world, "C_dyn_maze", seed)
    params = M8MAPS.dynamics_params("C_dyn_maze")
    hstar = ST.backward_spacetime_dijkstra(
        rm.adj, rm.points, dyn, params["v_agent"], params["dt"], params["t_max"], goal=1)
    if not (np.isfinite(hstar[0, 0]) and hstar[0, 0] < 1e29):
        return None
    return world, dyn, rm, mapname, hstar


def load_blind_provider(device):
    ck = HERE / "runs" / "c8_local_heavy" / "checkpoints" / "c8_field__unet_blind.pt"
    pl = torch.load(ck, map_location="cpu", weights_only=True)
    model = C6.build_model(pl["backbone"], in_channels=pl["in_channels"])
    model.load_state_dict(pl["model"]); model.to(device).eval()
    return DP.ValueFieldTemporalProvider(
        model, pl["grid_size"], device, pl["backbone"], pl["window_w"], time_blind=True)


def run_phase(group, occ_by_map, phase, n_needed, seed0, provider, anchor, writer):
    params = M8MAPS.dynamics_params("C_dyn_maze")
    v_agent, dt, t_max = params["v_agent"], params["dt"], int(params["t_max"])
    found, seed, attempts = 0, seed0, 0
    while found < n_needed and attempts < 4000:
        inst = build_instance(group, occ_by_map, seed)
        seed += 1
        attempts += 1
        if inst is None:
            continue
        world, dyn, rm, mapname, hstar = inst
        opt = int(hstar[0, 0])
        h_e = anchor.h_table(world, rm, dyn, v_agent, dt, t_max, goal_idx=1)
        arms = [("euclid", h_e)]
        arms += [(f"wastar_{w:g}", h_e * float(w)) for w in WEIGHTS]
        h_l = provider.h_table(world, rm, dyn, v_agent, dt, t_max, goal_idx=1)
        arms.append(("field_unet_blind", h_l))
        for arm_name, h in arms:
            res = ST.space_time_astar_prm(rm.adj, rm.points, dyn, h, BIGB,
                                          v_agent, dt, t_max, 0, 1)
            writer.writerow(dict(
                group=group, phase=phase, instance=found, map=mapname,
                seed=seed - 1, arm=arm_name, solved_bigb=bool(res["found"]),
                expansions=int(res["expansions"]),
                arrival=int(res["arrival"]), optimal_arrival=opt))
        found += 1
        if found % 5 == 0:
            print(f"[movingai] {group}/{phase}: {found}/{n_needed} "
                  f"(attempts {attempts})", flush=True)
    if found < n_needed:
        raise SystemExit(f"ABORT {group}/{phase}: only {found}/{n_needed} usable "
                         f"instances in {attempts} attempts")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["smoke", "full"], default="full")
    args = p.parse_args()
    M8MAPS.install_c8_dynamic_maps()
    device = torch.device("cpu")
    provider = load_blind_provider(device)
    anchor = DP.EuclidTimeProvider()
    occ = {g: {m: coarsen(parse_map(MAPS_DIR / m)) for m in ms}
           for g, ms in GROUPS.items()}
    for g, ms in occ.items():
        for m, o in ms.items():
            print(f"[movingai] {g}/{m}: blocked {o.mean():.2f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    n_dev, n_eval = (2, 2) if args.phase == "smoke" else (N_DEV, N_EVAL)
    tag = "smoke" if args.phase == "smoke" else "raw"
    with open(OUT / f"{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for g in GROUPS:
            run_phase(g, occ[g], "dev", n_dev, DEV_SEED0, provider, anchor, w)
            run_phase(g, occ[g], "eval", n_eval, EVAL_SEED0, provider, anchor, w)
    print(f"[movingai] wrote {OUT / (tag + '.csv')}", flush=True)


if __name__ == "__main__":
    main()
