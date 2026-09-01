#!/usr/bin/env python3
"""Wall-time decomposition for the dynamic arms (5 maps/suite sample, CPU).

Per instance: T_htable (feature/raster construction + U-Net inference over all
t_max+1 steps, as the provider builds its heuristic table), T_forward (one
timed model forward on a representative stack, reported so the inference
share of T_htable can be estimated), T_search_learned, T_search_anchor,
T_search_wastar (tuned per-suite weights). Searches run at the binding
budgets. Read-only; CPU condition recorded.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import continuous_prm_c8_dynamic_maps as M8MAPS
import continuous_prm_c8_dynamics_compare as M8C
import continuous_prm_c6_heatmap_value_field as C6
import continuous_prm_dynamic_providers as DP
import continuous_prm_spacetime as ST

HERE = Path(__file__).parent
OUT = HERE / "runs" / "c8r_timing"
# Canonical eval order (matches C8Config.eval_suites; suite_idx seeds worlds).
SUITES = ["C_dyn_maze", "C_dyn_rooms", "C_dyn_spiral",
          "C_dyn_maze_dense", "C_dyn_crossing", "C_dyn_rooms_large"]
_OLD_SUITES = ["C_dyn_crossing", "C_dyn_maze", "C_dyn_maze_dense",
          "C_dyn_rooms", "C_dyn_rooms_large", "C_dyn_spiral"]
BINDING = {"C_dyn_crossing": 150, "C_dyn_maze": 1800, "C_dyn_maze_dense": 2500,
           "C_dyn_rooms": 1300, "C_dyn_rooms_large": 600, "C_dyn_spiral": 2500}
N_SAMPLE = 5
COLS = ["suite", "world_index", "t_htable_s", "t_forward_ms",
        "t_search_learned_s", "t_search_anchor_s", "t_search_wastar_s"]


def main():
    M8MAPS.install_c8_dynamic_maps()
    device = torch.device("cpu")
    ck = HERE / "runs" / "c8_local_heavy" / "checkpoints" / "c8_field__unet_blind.pt"
    pl = torch.load(ck, map_location="cpu", weights_only=True)
    model = C6.build_model(pl["backbone"], in_channels=pl["in_channels"])
    model.load_state_dict(pl["model"]); model.to(device).eval()
    prov = DP.ValueFieldTemporalProvider(
        model, pl["grid_size"], device, pl["backbone"], pl["window_w"], time_blind=True)
    anchor = DP.EuclidTimeProvider()
    tuned = json.load(open(HERE / "runs" / "c8r_wastar" / "tuned_weights.json"))

    cfg = M8C.C8Config(seed=999999, eval_worlds=50)
    cfg = M8C.apply_scale_preset(cfg)
    cfg.eval_worlds = 50
    rows = []
    for suite_idx, suite in enumerate(SUITES):
        params = M8MAPS.dynamics_params(suite)
        v_agent, dt, t_max = params["v_agent"], params["dt"], int(params["t_max"])
        budget = BINDING[suite]
        w_h = float(tuned[suite]["w_h"])
        n = 0
        for wi, world, dyn, rm in M8C.iter_dynamic_worlds(suite, suite_idx, cfg, 50):
            if n >= N_SAMPLE:
                break
            n += 1
            t0 = time.perf_counter()
            h_l = prov.h_table(world, rm, dyn, v_agent, dt, t_max, goal_idx=1)
            t1 = time.perf_counter()
            # one representative forward for the inference share
            sb = DP.compute_field_static_base(world, prov.grid_size)
            x = DP.build_field_occupancy_stack(world, dyn, prov.grid_size, 0, 0,
                                               dt, static_base=sb)
            f0 = time.perf_counter()
            _ = C6.predict_residual_grid(model, x, device)
            f1 = time.perf_counter()
            h_e = anchor.h_table(world, rm, dyn, v_agent, dt, t_max, goal_idx=1)
            s0 = time.perf_counter()
            ST.space_time_astar_prm(rm.adj, rm.points, dyn, h_l, budget,
                                    v_agent, dt, t_max, 0, 1)
            s1 = time.perf_counter()
            ST.space_time_astar_prm(rm.adj, rm.points, dyn, h_e, budget,
                                    v_agent, dt, t_max, 0, 1)
            s2 = time.perf_counter()
            ST.space_time_astar_prm(rm.adj, rm.points, dyn, h_e * w_h, budget,
                                    v_agent, dt, t_max, 0, 1)
            s3 = time.perf_counter()
            rows.append(dict(
                suite=suite, world_index=wi,
                t_htable_s=round(t1 - t0, 4),
                t_forward_ms=round(1000 * (f1 - f0), 3),
                t_search_learned_s=round(s1 - s0, 4),
                t_search_anchor_s=round(s2 - s1, 4),
                t_search_wastar_s=round(s3 - s2, 4)))
        print(f"[timing] {suite}: {n} instances", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "timing_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    arr = {k: np.array([r[k] for r in rows]) for k in COLS[2:]}
    summary = {k: dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
               for k, v in arr.items()}
    summary["cpu_condition"] = "local CPU (Windows, PyTorch CPU build), single process"
    with open(OUT / "timing_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
