#!/usr/bin/env python3
"""Few-shot rescue on the MovingAI external benchmark.

Frozen design: docs/experiments/continuous/c08/design/
2026-07-27-c8-movingai-probe-fewshot.md (Part 2).

Adapts the frozen blind U-Net on the first K recorded *development*
instances per group (conv-LoRA r8 / full FT, exact C14 dynamic recipe:
2,560 steps, batch 8, AdamW 2e-4/1e-4, clip 1.0, smooth-L1 on the capped
normalized residual), then evaluates each arm once on the frozen 25
evaluation instances per group at BIGB with the frozen binding thresholds.
Zero-shot / anchor / WA* comparison rows are the frozen raw.csv rows and are
not recomputed. Idempotent per phase; checkpoints and eval rows are skipped
if present.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

import continuous_prm_c8_dynamic_maps as M8MAPS
import continuous_prm_c8_dynamics_compare as M8C
import continuous_prm_c8_movingai_external as MAI
import continuous_prm_c6_heatmap_value_field as C6
import continuous_prm_c9h_transfer as C9H
import continuous_prm_common as C
import continuous_prm_dynamic_providers as DP
import continuous_prm_spacetime as ST

HERE = Path(__file__).parent
OUT = HERE / "runs" / "c8r_movingai"
CKPT_DIR = OUT / "fewshot_checkpoints"
DS_DIR = OUT / "fewshot_datasets"
RAW = OUT / "raw.csv"
FS_RAW = OUT / "fewshot_raw.csv"

KS = [1, 2, 4, 8]
METHODS = ["lora", "full_ft"]
SEEDS = [0, 1]
TOTAL_STEPS = 2560
BATCH = 8
LR, WD = 2.0e-4, 1.0e-4
RANK, ALPHA = 8, 1.0
CAP = 4.0
BIGB = MAI.BIGB

COLS = ["group", "K", "method", "adapt_seed", "instance", "map", "seed",
        "arm", "solved_bigb", "expansions", "arrival", "optimal_arrival"]


def recorded_seeds(phase: str) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    with open(RAW, newline="") as f:
        for r in csv.DictReader(f):
            if r["phase"] == phase and r["arm"] == "euclid":
                out.setdefault(r["group"], []).append(int(r["seed"]))
    return out


def build_instances(group: str, occ_by_map, seeds: List[int]):
    out = []
    for s in seeds:
        inst = MAI.build_instance(group, occ_by_map, s)
        if inst is None:
            raise SystemExit(f"instance reconstruction failed: {group} seed {s}")
        out.append((s,) + inst)  # (seed, world, dyn, rm, mapname, hstar)
    return out


def build_dataset(group: str, insts, K: int) -> Path:
    """Field-schema dataset over ALL reachable states of the first K
    instances (mirrors the canonical field trainer's per-(world,t) samples)."""
    out_npz = DS_DIR / f"{group}_K{K}.npz"
    if out_npz.exists():
        return out_npz
    params = M8MAPS.dynamics_params("C_dyn_maze")
    v_agent, dt, t_max = params["v_agent"], params["dt"], int(params["t_max"])
    G = 64
    occs, cells_l, targets_l, masks_l = [], [], [], []
    for (seed, world, dyn, rm, mapname, hstar) in insts[:K]:
        ttg = ST.oracle_time_to_go(hstar, t_max)
        euclid_steps = DP.euclid_time_row(rm, v_agent, goal_idx=1) / dt
        T_scale = float(world.side_len) / v_agent / dt
        resid = np.clip(ttg - euclid_steps[:, None], 0.0, None) / T_scale
        resid = np.clip(resid, 0.0, CAP).astype(np.float32)
        reach = (np.isfinite(hstar) & (hstar < 1e29))
        cells = M8C._build_field_node_cells(rm, float(world.side_len), G)
        static_base = DP.compute_field_static_base(world, G)
        for t in range(t_max + 1):
            x_t = DP.build_field_occupancy_stack(world, dyn, G, t, 0, dt,
                                                 static_base=static_base)
            occs.append(x_t.astype(np.float32))
            cells_l.append(cells.astype(np.int64))
            targets_l.append(resid[:, t])
            masks_l.append(reach[:, t])
    C.ensure_dir(DS_DIR)
    np.savez(out_npz, occ=np.stack(occs), cells=np.stack(cells_l),
             target=np.stack(targets_l), mask=np.stack(masks_l))
    print(f"[fewshot] dataset {out_npz.name}: {len(occs)} samples, "
          f"{int(np.stack(masks_l).sum())} supervised states", flush=True)
    return out_npz


def train_arm(npz_path: Path, method: str, s: int, out_ckpt: Path, device):
    if out_ckpt.exists():
        return out_ckpt
    z = np.load(npz_path, allow_pickle=False)
    occ, cells, target, mask = z["occ"], z["cells"], z["target"], z["mask"]

    src_ckpt = HERE / "runs" / "c8_local_heavy" / "checkpoints" / "c8_field__unet_blind.pt"
    src = torch.load(src_ckpt, map_location="cpu", weights_only=True)
    arm_seed = 977_000 + 131 * s + (7 if method == "lora" else 0) + len(occ)
    C.set_global_seed(arm_seed)
    model = C6.build_model(src["backbone"], in_channels=int(src["in_channels"])).to(device)
    C.safe_load_state(model, src_ckpt)
    if method == "lora":
        C9H.apply_conv_lora(model, rank=RANK, alpha=ALPHA)
        C.set_lora_trainable(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)

    live = np.where(mask.any(axis=1))[0]
    rng = np.random.default_rng(arm_seed + 1)
    model.train()
    losses = []
    for step in range(TOTAL_STEPS):
        bidx = live[rng.integers(0, len(live), BATCH)]
        occ_b = torch.from_numpy(np.ascontiguousarray(occ[bidx])).to(device=device, dtype=torch.float32)
        cells_b = torch.from_numpy(np.ascontiguousarray(cells[bidx])).to(device)
        target_b = torch.from_numpy(np.ascontiguousarray(target[bidx])).to(device=device, dtype=torch.float32)
        mask_b = torch.from_numpy(np.ascontiguousarray(mask[bidx])).to(device)
        pred_grid = C6.model_output_residual(model(occ_b))
        if not torch.isfinite(pred_grid).all():
            raise FloatingPointError(f"fewshot {method}: non-finite grid")
        B, G, _ = pred_grid.shape
        ix = cells_b[..., 0].clamp(0, G - 1)
        iy = cells_b[..., 1].clamp(0, G - 1)
        pred_nodes = torch.gather(pred_grid.reshape(B, G * G), 1, ix * G + iy)
        m = mask_b.bool()
        loss = F.smooth_l1_loss(pred_nodes[m], target_b[m])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"fewshot {method}: non-finite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.item()))
    payload = {
        "model": model.state_dict(),
        "in_channels": int(src["in_channels"]),
        "window_w": 0,
        "grid_size": int(src["grid_size"]),
        "backbone": src["backbone"],
        "method": method,
        "total_steps": TOTAL_STEPS,
        "final_loss": float(np.mean(losses[-50:])),
    }
    if method == "lora":
        payload["lora_rank"] = RANK
        payload["alpha"] = ALPHA
    C.ensure_dir(out_ckpt.parent)
    torch.save(payload, out_ckpt)
    print(f"[fewshot] trained {out_ckpt.name} (final loss "
          f"{payload['final_loss']:.4f})", flush=True)
    return out_ckpt


def make_provider(ckpt: Path, device):
    pl = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = C6.build_model(pl["backbone"], in_channels=int(pl["in_channels"]))
    if pl.get("lora_rank"):
        C9H.apply_conv_lora(model, rank=int(pl["lora_rank"]),
                            alpha=float(pl.get("alpha", 1.0)))
    model.load_state_dict(pl["model"])
    model.to(device).eval()
    return DP.ValueFieldTemporalProvider(model, int(pl["grid_size"]), device,
                                         pl["backbone"], 0, time_blind=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["label", "adapt", "eval", "all"],
                    default="all")
    args = ap.parse_args()
    M8MAPS.install_c8_dynamic_maps()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fewshot] device: {device}", flush=True)
    occ = {g: {m: MAI.coarsen(MAI.parse_map(MAI.MAPS_DIR / m)) for m in ms}
           for g, ms in MAI.GROUPS.items()}
    dev_seeds = recorded_seeds("dev")
    eval_seeds = recorded_seeds("eval")
    groups = sorted(dev_seeds)

    if args.phase in ("label", "all"):
        for g in groups:
            insts = build_instances(g, occ[g], dev_seeds[g][:max(KS)])
            for K in KS:
                build_dataset(g, insts, K)

    if args.phase in ("adapt", "all"):
        for g in groups:
            for K in KS:
                for method in METHODS:
                    for s in SEEDS:
                        train_arm(DS_DIR / f"{g}_K{K}.npz", method, s,
                                  CKPT_DIR / f"{g}_K{K}_{method}_s{s}.pt",
                                  device)

    if args.phase in ("eval", "all"):
        params = M8MAPS.dynamics_params("C_dyn_maze")
        v_agent, dt, t_max = params["v_agent"], params["dt"], int(params["t_max"])
        done = set()
        if FS_RAW.exists():
            with open(FS_RAW, newline="") as f:
                for r in csv.DictReader(f):
                    done.add((r["group"], int(r["K"]), r["method"],
                              int(r["adapt_seed"])))
        mode = "a" if FS_RAW.exists() else "w"
        with open(FS_RAW, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            if mode == "w":
                w.writeheader()
            for g in groups:
                insts = build_instances(g, occ[g], eval_seeds[g])
                for K in KS:
                    for method in METHODS:
                        for s in SEEDS:
                            if (g, K, method, s) in done:
                                continue
                            provider = make_provider(
                                CKPT_DIR / f"{g}_K{K}_{method}_s{s}.pt", device)
                            for i, (seed, world, dyn, rm, mapname, hstar) in enumerate(insts):
                                h = provider.h_table(world, rm, dyn, v_agent,
                                                     dt, t_max, goal_idx=1)
                                res = ST.space_time_astar_prm(
                                    rm.adj, rm.points, dyn, h, BIGB,
                                    v_agent, dt, t_max, 0, 1)
                                w.writerow(dict(
                                    group=g, K=K, method=method, adapt_seed=s,
                                    instance=i, map=mapname, seed=seed,
                                    arm=f"adapted_{method}",
                                    solved_bigb=bool(res["found"]),
                                    expansions=int(res["expansions"]),
                                    arrival=int(res["arrival"]),
                                    optimal_arrival=int(hstar[0, 0])))
                            f.flush()
                            print(f"[fewshot] eval {g} K={K} {method} s{s}: done",
                                  flush=True)
    print("[fewshot] all phases complete", flush=True)


if __name__ == "__main__":
    main()
