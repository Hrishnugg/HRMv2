#!/usr/bin/env python3
"""Residual probe: predicted-vs-true residual quality, OOD vs in-distribution.

Frozen design: docs/experiments/continuous/c08/design/
2026-07-27-c8-movingai-probe-fewshot.md (Part 1).

For the frozen blind U-Net, compares the provider's normalized residual
table against the exact backward space-time Dijkstra target on (a) the 50
frozen MovingAI evaluation instances (reconstructed from recorded seeds) and
(b) the first 8 confirmation-cohort worlds per procedural suite. Optionally
probes an adapted checkpoint (--ckpt) for the R6 mechanism readout.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import continuous_prm_c8_dynamic_maps as M8MAPS
import continuous_prm_c8_dynamics_compare as M8C
import continuous_prm_c8_movingai_external as MAI
import continuous_prm_c6_heatmap_value_field as C6
import continuous_prm_dynamic_providers as DP
import continuous_prm_spacetime as ST

HERE = Path(__file__).parent
RAW = HERE / "runs" / "c8r_movingai" / "raw.csv"
OUT_DIR = HERE / "runs" / "c8r_movingai"
SUITES = ["C_dyn_crossing", "C_dyn_maze", "C_dyn_maze_dense",
          "C_dyn_rooms", "C_dyn_rooms_large", "C_dyn_spiral"]
N_REF_PER_SUITE = 8
CAP = 4.0


def recorded_seeds(phase: str) -> dict:
    """group -> ordered list of recorded instance seeds for the phase."""
    out = {}
    with open(RAW, newline="") as f:
        for r in csv.DictReader(f):
            if r["phase"] == phase and r["arm"] == "euclid":
                out.setdefault(r["group"], []).append(int(r["seed"]))
    return out


def probe_metrics(h_learned, h_euclid, hstar, ttg, T_scale):
    """Per-map metrics over reachable slots, in normalized residual units."""
    reach = np.isfinite(hstar) & (hstar < 1e29)
    pred = (h_learned - h_euclid[:, None]) / T_scale
    true = np.clip(np.clip(ttg - h_euclid[:, None], 0.0, None) / T_scale,
                   0.0, CAP)
    p, t = pred[reach], true[reach]
    if len(p) < 10 or float(np.std(t)) == 0.0:
        return None
    pr = float(np.corrcoef(p, t)[0, 1])
    rp = np.argsort(np.argsort(p)).astype(np.float64)
    rt = np.argsort(np.argsort(t)).astype(np.float64)
    sr = float(np.corrcoef(rp, rt)[0, 1])
    return dict(pearson=pr, spearman=sr,
                mae=float(np.mean(np.abs(p - t))),
                bias=float(np.mean(p - t)),
                mean_true=float(np.mean(t)), mean_pred=float(np.mean(p)),
                n_slots=int(reach.sum()))


def probe_map(provider, anchor, world, rm, dyn, params, hstar=None):
    v_agent, dt, t_max = params["v_agent"], params["dt"], int(params["t_max"])
    if hstar is None:
        hstar = ST.backward_spacetime_dijkstra(
            rm.adj, rm.points, dyn, v_agent, dt, t_max, goal=1)
    ttg = ST.oracle_time_to_go(hstar, t_max)
    h_e = anchor.h_table(world, rm, dyn, v_agent, dt, t_max, goal_idx=1)
    h_l = provider.h_table(world, rm, dyn, v_agent, dt, t_max, goal_idx=1)
    T_scale = float(world.side_len) / v_agent / dt
    return probe_metrics(h_l, h_e[:, 0] if h_e.ndim == 2 else h_e,
                         hstar, ttg, T_scale)


def load_provider(device, ckpt: Path | None):
    if ckpt is None:
        return MAI.load_blind_provider(device)
    pl = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = C6.build_model(pl["backbone"], in_channels=pl["in_channels"])
    if pl.get("lora_rank"):
        import continuous_prm_c9h_transfer as C9H
        C9H.apply_conv_lora(model, rank=int(pl["lora_rank"]),
                            alpha=float(pl.get("alpha", 1.0)))
    model.load_state_dict(pl["model"])
    model.to(device).eval()
    return DP.ValueFieldTemporalProvider(model, pl["grid_size"], device,
                                         pl["backbone"], pl["window_w"],
                                         time_blind=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="adapted checkpoint to probe instead of the frozen source")
    ap.add_argument("--groups-only", action="store_true",
                    help="skip the procedural reference (for R6 re-probes)")
    ap.add_argument("--tag", default="probe")
    args = ap.parse_args()

    M8MAPS.install_c8_dynamic_maps()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    provider = load_provider(device, Path(args.ckpt) if args.ckpt else None)
    anchor = DP.EuclidTimeProvider()
    occ = {g: {m: MAI.coarsen(MAI.parse_map(MAI.MAPS_DIR / m)) for m in ms}
           for g, ms in MAI.GROUPS.items()}
    params_mai = M8MAPS.dynamics_params("C_dyn_maze")

    rows = []
    for group, seeds in recorded_seeds("eval").items():
        for i, seed in enumerate(seeds):
            inst = MAI.build_instance(group, occ[group], seed)
            if inst is None:
                raise SystemExit(f"instance reconstruction failed: {group} seed {seed}")
            world, dyn, rm, mapname, hstar = inst
            m = probe_map(provider, anchor, world, rm, dyn, params_mai, hstar)
            if m:
                rows.append(dict(setname=group, map=mapname, index=i, **m))
            print(f"[probe] {group} {i + 1}/{len(seeds)}", flush=True)

    if not args.groups_only:
        cfg = M8C.C8Config(seed=999999, eval_worlds=N_REF_PER_SUITE)
        cfg = M8C.apply_scale_preset(cfg)
        cfg.eval_worlds = N_REF_PER_SUITE
        for si, suite in enumerate(SUITES):
            params = M8MAPS.dynamics_params(suite)
            for wi, world, dyn, rm in M8C.iter_dynamic_worlds(
                    suite, si, cfg, N_REF_PER_SUITE):
                m = probe_map(provider, anchor, world, rm, dyn, params)
                if m:
                    rows.append(dict(setname=suite, map=suite, index=wi, **m))
            print(f"[probe] ref {suite}: done", flush=True)

    by = {}
    for r in rows:
        by.setdefault(r["setname"], []).append(r)
    summary = {}
    for k, v in sorted(by.items()):
        summary[k] = {
            "n_maps": len(v),
            "pearson_median": float(np.median([r["pearson"] for r in v])),
            "pearson_iqr": [float(np.percentile([r["pearson"] for r in v], q))
                            for q in (25, 75)],
            "spearman_median": float(np.median([r["spearman"] for r in v])),
            "mae_median": float(np.median([r["mae"] for r in v])),
            "bias_median": float(np.median([r["bias"] for r in v])),
            "mean_true_median": float(np.median([r["mean_true"] for r in v])),
            "mean_pred_median": float(np.median([r["mean_pred"] for r in v])),
        }
        s = summary[k]
        print(f"[probe] {k}: r={s['pearson_median']:.3f} "
              f"[{s['pearson_iqr'][0]:.3f},{s['pearson_iqr'][1]:.3f}] "
              f"rho={s['spearman_median']:.3f} mae={s['mae_median']:.3f} "
              f"bias={s['bias_median']:+.3f} (n={s['n_maps']})", flush=True)

    out = OUT_DIR / f"{args.tag}.json"
    with open(out, "w") as f:
        json.dump(dict(ckpt=str(args.ckpt), per_map=rows, summary=summary),
                  f, indent=1)
    print(f"[probe] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
