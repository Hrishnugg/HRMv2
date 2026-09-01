#!/usr/bin/env python3
"""C8-X v2: MovingAI at map scale + adaptation transfer to unseen maps.

Frozen design 2026-07-26-c8-movingai-scale-transfer.md, Amendments 1-2.
Amendment 2 (v12-review refinements, adopted before any v2 execution):
selection widened to 20 maps/category (street includes _512 files); pool
maps' DEV instances are evaluated in the zeroshot phase (budget/weight
selection in the frozen analysis uses ONLY pool-dev rows; held-out maps
receive a single frozen evaluation of EVAL instances); exact retained-label
matching via a per-category N_target (largest power of two <= the minimum
label supply over all planned cells, computed from outcome-independent
manifest counts before any adaptation) with seeded mask thinning; three
map-set draws (pool rotations) for M in {1,2,4} plus the single M=8 draw;
methods lora / full / scratch (scratch = same loop from random init);
conversion-fidelity stats recorded per map.

Phases: select, generate (+ plan embedded), zeroshot, adapt, evaladapted.
Outputs under $C8X_ROOT/runs/c8x2_scale/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import continuous_prm_c8_dynamic_maps as M8MAPS
import continuous_prm_c8_movingai_external as MAI
import continuous_prm_c8_movingai_fewshot as FS
import continuous_prm_c6_heatmap_value_field as C6
import continuous_prm_c9h_transfer as C9H
import continuous_prm_common as C
import continuous_prm_dynamic_providers as DP
import continuous_prm_spacetime as ST

M8MAPS.install_c8_dynamic_maps()

HERE = Path(__file__).parent
ROOT = Path(os.environ.get("C8X_ROOT", str(HERE)))
OUT = ROOT / "runs" / "c8x2_scale"
SRC_ZIPS = {"street": "street-map.zip", "dao": "dao-map.zip"}
CATS = ["street", "dao"]
N_PER_CAT = 20
N_DEV, N_EVAL = 10, 12
USABLE_MIN = 8
K_TOTAL = 8
MS = [1, 2, 4, 8]
# Amendment 3 (2026-07-27): balanced draws. M=1 covers all eight pool maps,
# M=2 the four disjoint pairs; M=4 keeps the two disjoint halves plus the
# legacy draw 2 (subsample replicate of draw 0, excluded from balanced
# summaries). Pre-amendment cells are reused untouched (idempotent skip).
DRAWS = {1: [0, 1, 2, 3, 4, 5, 6, 7], 2: [0, 1, 2, 3], 4: [0, 1, 2], 8: [0]}
METHODS = ["lora", "full", "scratch"]
SEEDS = [0, 1]
DEV0, EVAL0 = 50_000_000, 51_000_000
STRIDE_MAP, STRIDE_SLOT = 20_000, 40
BIGB = MAI.BIGB
ZS_COLS = ["cat", "map", "role", "phase", "slot", "seed", "arm", "found",
           "expansions", "arrival", "optimal_arrival"]


def _dims_ok(path: Path) -> bool:
    lines = path.read_text().splitlines()
    return max(int(lines[1].split()[1]), int(lines[2].split()[1])) <= 512


def _components(occ: np.ndarray) -> int:
    free = ~occ
    seen = np.zeros_like(free, dtype=bool)
    n = 0
    H, W = free.shape
    for i in range(H):
        for j in range(W):
            if free[i, j] and not seen[i, j]:
                n += 1
                q = deque([(i, j)])
                seen[i, j] = True
                while q:
                    a, b = q.popleft()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if (0 <= x < H and 0 <= y < W and free[x, y]
                                and not seen[x, y]):
                            seen[x, y] = True
                            q.append((x, y))
    return n


def phase_select() -> Path:
    out = OUT / "selection.json"
    if out.exists():
        print("[c8x2] selection.json exists; skip", flush=True)
        return out
    OUT.mkdir(parents=True, exist_ok=True)
    maps_dir = OUT / "maps"
    maps_dir.mkdir(exist_ok=True)
    sel = {}
    for cat in CATS:
        z = ROOT / "runs" / "c8r_movingai" / "maps" / SRC_ZIPS[cat]
        if not z.exists():
            z = HERE / "runs" / "c8r_movingai" / "maps" / SRC_ZIPS[cat]
        ext = OUT / f"src_{cat}"
        if not ext.exists():
            with zipfile.ZipFile(z) as f:
                f.extractall(ext)
        cand = sorted(p for p in ext.rglob("*.map"))
        if cat == "street":
            chosen = [p for p in cand
                      if (p.name.endswith("_256.map")
                          or p.name.endswith("_512.map"))
                      and _dims_ok(p)][:N_PER_CAT]
        else:
            chosen = [p for p in cand if _dims_ok(p)][:N_PER_CAT]
        entry = {}
        for p in chosen:
            dst = maps_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
            entry[p.name] = hashlib.sha256(dst.read_bytes()).hexdigest()
        sel[cat] = entry
        print(f"[c8x2] {cat}: selected {len(entry)} maps", flush=True)
    out.write_text(json.dumps(sel, indent=1))
    return out


def _occ(cat: str, name: str) -> np.ndarray:
    return MAI.coarsen(MAI.parse_map(OUT / "maps" / name))


def _inst(cat: str, name: str, occ: np.ndarray, seed: int):
    saved = MAI.GROUPS
    MAI.GROUPS = {cat: [name]}
    try:
        return MAI.build_instance(cat, {name: occ}, seed)
    finally:
        MAI.GROUPS = saved


def _label_count(hstar: np.ndarray) -> int:
    return int((np.isfinite(hstar) & (hstar < 1e29)).sum())


def _cell_maps(pool, m: int, draw: int):
    return [pool[(draw * m + i) % len(pool)] for i in range(m)]


def phase_generate(cat: str) -> Path:
    out = OUT / f"instances_{cat}.json"
    if out.exists():
        print(f"[c8x2] {out.name} exists; skip", flush=True)
        return out
    sel = json.loads((OUT / "selection.json").read_text())[cat]
    names = sorted(sel)
    manifest = {}
    for mi, name in enumerate(names):
        occ = _occ(cat, name)
        rec = {"sha": sel[name], "dev": [], "dev_labels": [], "eval": [],
               "attempts": 0, "free_frac": round(float((~occ).mean()), 4),
               "components": _components(occ)}
        for role, base, want in (("dev", DEV0, N_DEV), ("eval", EVAL0, N_EVAL)):
            for slot in range(want):
                for a in range(STRIDE_SLOT):
                    seed = base + mi * STRIDE_MAP + slot * STRIDE_SLOT + a
                    rec["attempts"] += 1
                    inst = _inst(cat, name, occ, seed)
                    if inst is not None:
                        rec[role].append(seed)
                        if role == "dev":
                            rec["dev_labels"].append(_label_count(inst[4]))
                        break
        rec["usable"] = (len(rec["dev"]) >= USABLE_MIN and
                         len(rec["eval"]) >= USABLE_MIN)
        manifest[name] = rec
        print(f"[c8x2] {cat}/{name}: dev {len(rec['dev'])}/{N_DEV} eval "
              f"{len(rec['eval'])}/{N_EVAL} usable={rec['usable']} "
              f"free={rec['free_frac']}", flush=True)
    usable = [n for n in names if manifest[n]["usable"]]
    pool, heldout = usable[:8], usable[8:]
    supplies = []
    for m in MS:
        for d in DRAWS[m]:
            per = K_TOTAL // m
            supply = sum(sum(manifest[nm]["dev_labels"][:per])
                         for nm in _cell_maps(pool, m, d))
            supplies.append(supply)
    n_target = 1 << int(np.floor(np.log2(min(supplies)))) if supplies else 0
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cat": cat, "maps": manifest, "usable": usable,
                               "pool": pool, "heldout": heldout,
                               "n_target": n_target,
                               "cell_supplies_min": min(supplies)
                               if supplies else 0}, indent=1))
    print(f"[c8x2] {cat}: usable {len(usable)} (pool 8, heldout "
          f"{len(heldout)}); N_target={n_target}", flush=True)
    return out


def load_learned(device):
    _ensure_ckpt(ROOT)
    saved = MAI.HERE
    MAI.HERE = ROOT if (ROOT / "runs" / "c8_local_heavy").exists() else HERE
    try:
        return MAI.load_blind_provider(torch.device(device))
    finally:
        MAI.HERE = saved


def _ensure_ckpt(root: Path):
    f = "c8_field__unet_blind.pt"
    dst = root / "runs" / "c8_local_heavy" / "checkpoints"
    src = root / "runs" / "c14_sources" / "c8_local_heavy" / "checkpoints"
    if not (dst / f).exists() and (src / f).exists():
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f, dst / f)


def _eval_rows(cat, name, role, phase, occ, seeds, arms, writer):
    p = M8MAPS.dynamics_params("C_dyn_maze")
    v_agent, dt, t_max = float(p["v_agent"]), float(p["dt"]), int(p["t_max"])
    for slot, seed in enumerate(seeds):
        inst = _inst(cat, name, occ, seed)
        if inst is None:
            raise SystemExit(f"reconstruction failed {cat}/{name} seed {seed}")
        world, dyn, rm, mapname, hstar = inst
        opt = int(hstar[0, 0])
        for arm_name, provider, w in arms:
            h = provider.h_table(world, rm, dyn, v_agent, dt, t_max,
                                 goal_idx=1)
            if w:
                h = h * float(w)
            res = ST.space_time_astar_prm(rm.adj, rm.points, dyn, h, BIGB,
                                          v_agent, dt, t_max, 0, 1)
            writer.writerow(dict(cat=cat, map=name, role=role, phase=phase,
                                 slot=slot, seed=seed, arm=arm_name,
                                 found=bool(res["found"]),
                                 expansions=int(res["expansions"]),
                                 arrival=int(res["arrival"]),
                                 optimal_arrival=opt))


def phase_zeroshot(cat: str, name: str, device: str = "cpu") -> Path:
    out = OUT / f"zs_{cat}_{name}.csv"
    if out.exists():
        print(f"[c8x2] {out.name} exists; skip", flush=True)
        return out
    man = json.loads((OUT / f"instances_{cat}.json").read_text())
    if name not in man["usable"]:
        print(f"[c8x2] {cat}/{name} not usable; skip", flush=True)
        return out
    role = "pool" if name in man["pool"] else "heldout"
    occ = _occ(cat, name)
    anchor = DP.EuclidTimeProvider()
    learned = load_learned(device)
    arms = [("euclid", anchor, None)]
    arms += [(f"wastar_{w:g}", anchor, w) for w in MAI.WEIGHTS]
    arms += [("zeroshot", learned, None)]
    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=ZS_COLS)
        wtr.writeheader()
        if role == "pool":
            _eval_rows(cat, name, role, "dev", occ,
                       man["maps"][name]["dev"], arms, wtr)
        _eval_rows(cat, name, role, "eval", occ,
                   man["maps"][name]["eval"], arms, wtr)
    print(f"[c8x2] wrote {out.name}", flush=True)
    return out


def _thin_mask(npz_path: Path, n_target: int, rng_seed: int):
    z = np.load(npz_path, allow_pickle=False)
    occ, cells, target, mask = (z["occ"], z["cells"], z["target"],
                                z["mask"].copy())
    total = int(mask.sum())
    if total < n_target:
        raise SystemExit(f"ABORT: label supply {total} < N_target {n_target}")
    if total > n_target:
        idx = np.argwhere(mask)
        keep = np.random.default_rng(rng_seed).choice(len(idx), n_target,
                                                      replace=False)
        newmask = np.zeros_like(mask)
        for k in keep:
            newmask[idx[k][0], idx[k][1]] = True
        mask = newmask
    np.savez(npz_path, occ=occ, cells=cells, target=target, mask=mask)
    return int(mask.sum())


def _train(npz_path: Path, method: str, s: int, out_ckpt: Path, device):
    """lora/full delegate to FS.train_arm; scratch = same loop, random init."""
    if method in ("lora", "full"):
        saved = FS.HERE
        if (ROOT / "runs" / "c8_local_heavy" / "checkpoints").exists():
            FS.HERE = ROOT
        try:
            return FS.train_arm(npz_path, method, s, out_ckpt, device)
        finally:
            FS.HERE = saved
    if out_ckpt.exists():
        return out_ckpt
    z = np.load(npz_path, allow_pickle=False)
    occ, cells, target, mask = z["occ"], z["cells"], z["target"], z["mask"]
    src_ckpt = ROOT / "runs" / "c8_local_heavy" / "checkpoints" / \
        "c8_field__unet_blind.pt"
    src = torch.load(src_ckpt, map_location="cpu", weights_only=True)
    arm_seed = 977_000 + 131 * s + 13 + len(occ)
    C.set_global_seed(arm_seed)
    model = C6.build_model(src["backbone"],
                           in_channels=int(src["in_channels"])).to(device)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=FS.LR, weight_decay=FS.WD)
    live = np.where(mask.any(axis=1))[0]
    rng = np.random.default_rng(arm_seed + 1)
    model.train()
    losses = []
    for step in range(FS.TOTAL_STEPS):
        bidx = live[rng.integers(0, len(live), FS.BATCH)]
        occ_b = torch.from_numpy(np.ascontiguousarray(occ[bidx])).to(
            device=device, dtype=torch.float32)
        cells_b = torch.from_numpy(np.ascontiguousarray(cells[bidx])).to(device)
        target_b = torch.from_numpy(np.ascontiguousarray(target[bidx])).to(
            device=device, dtype=torch.float32)
        mask_b = torch.from_numpy(np.ascontiguousarray(mask[bidx])).to(device)
        pred_grid = C6.model_output_residual(model(occ_b))
        if not torch.isfinite(pred_grid).all():
            raise FloatingPointError("scratch: non-finite grid")
        B, G, _ = pred_grid.shape
        ix = cells_b[..., 0].clamp(0, G - 1)
        iy = cells_b[..., 1].clamp(0, G - 1)
        pred_nodes = torch.gather(pred_grid.reshape(B, G * G), 1, ix * G + iy)
        m = mask_b.bool()
        loss = F.smooth_l1_loss(pred_nodes[m], target_b[m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.item()))
    payload = {"model": model.state_dict(),
               "in_channels": int(src["in_channels"]), "window_w": 0,
               "grid_size": int(src["grid_size"]), "backbone": src["backbone"],
               "method": "scratch", "total_steps": FS.TOTAL_STEPS,
               "final_loss": float(np.mean(losses[-50:]))}
    C.ensure_dir(out_ckpt.parent)
    torch.save(payload, out_ckpt)
    print(f"[c8x2] trained {out_ckpt.name} (scratch, final "
          f"{payload['final_loss']:.4f})", flush=True)
    return out_ckpt


def cell_tag(cat, m, draw, method, seed):
    return f"{cat}_M{m}_d{draw}_{method}_s{seed}"


def phase_adapt(cat, m, draw, method, seed, device="cuda") -> Path:
    _ensure_ckpt(ROOT)
    tag = cell_tag(cat, m, draw, method, seed)
    ck = OUT / "ckpts" / f"adapt_{tag}.pt"
    if ck.exists():
        print(f"[c8x2] {ck.name} exists; skip", flush=True)
        return ck
    man = json.loads((OUT / f"instances_{cat}.json").read_text())
    per_map = K_TOTAL // m
    insts = []
    for name in _cell_maps(man["pool"], m, draw):
        occ = _occ(cat, name)
        for seed_i in man["maps"][name]["dev"][:per_map]:
            inst = _inst(cat, name, occ, seed_i)
            if inst is None:
                raise SystemExit(f"dev reconstruction failed {name}/{seed_i}")
            insts.append((seed_i,) + inst)
    npz = OUT / "datasets" / f"cell_{tag}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    if not npz.exists():
        saved = FS.DS_DIR
        FS.DS_DIR = npz.parent
        try:
            built = FS.build_dataset(f"cell_{tag}", insts, len(insts))
        finally:
            FS.DS_DIR = saved
        if built != npz:  # FS.build_dataset appends _K{n} to the stem
            built.replace(npz)
        # Amendment 3: stable digest seed (Python's hash() is process-salted;
        # pre-amendment cells keep their materialized datasets, hashes in the
        # artifact manifest)
        kept = _thin_mask(npz, int(man["n_target"]),
                          rng_seed=int.from_bytes(
                              hashlib.sha256(tag.encode()).digest()[:4],
                              "big"))
        print(f"[c8x2] {tag}: thinned to {kept} labels "
              f"(N_target {man['n_target']})", flush=True)
    dev = torch.device(device if (device != "cuda"
                                  or torch.cuda.is_available()) else "cpu")
    ck.parent.mkdir(parents=True, exist_ok=True)
    _train(npz, method, seed, ck, dev)
    return ck


def phase_evaladapted(cat, m, draw, method, seed, device="cpu",
                      only_map: str = "") -> Path:
    """Full-cell evaluation, or (preemption-resilient path, Amendment 3
    operational note) a single-map part when only_map is given."""
    tag = cell_tag(cat, m, draw, method, seed)
    out = (OUT / f"ad_{tag}.csv" if not only_map
           else OUT / f"adpart_{tag}__{only_map}.csv")
    if out.exists():
        print(f"[c8x2] {out.name} exists; skip", flush=True)
        return out
    ck = OUT / "ckpts" / f"adapt_{tag}.pt"
    man = json.loads((OUT / f"instances_{cat}.json").read_text())
    provider = FS.make_provider(ck, torch.device(device))
    arms = [(f"adapted_{tag}", provider, None)]
    names = [only_map] if only_map else man["usable"]
    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=ZS_COLS)
        wtr.writeheader()
        for name in names:
            role = "pool" if name in man["pool"] else "heldout"
            _eval_rows(cat, name, role, "eval", _occ(cat, name),
                       man["maps"][name]["eval"], arms, wtr)
    print(f"[c8x2] wrote {out.name}", flush=True)
    return out


def phase_mergeparts(cat, m, draw, method, seed, expected: int) -> Path:
    """Concatenate adpart files into ad_{tag}.csv (held-out-only cells)."""
    tag = cell_tag(cat, m, draw, method, seed)
    out = OUT / f"ad_{tag}.csv"
    if out.exists():
        print(f"[c8x2] {out.name} exists; skip", flush=True)
        return out
    parts = sorted(OUT.glob(f"adpart_{tag}__*.csv"))
    assert len(parts) == expected, (tag, len(parts), expected)
    rows = []
    for p in parts:
        with open(p, newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=ZS_COLS)
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    print(f"[c8x2] merged {len(parts)} parts -> {out.name}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True,
                   choices=["select", "generate", "zeroshot", "adapt",
                            "evaladapted"])
    p.add_argument("--cat", default="")
    p.add_argument("--map", default="")
    p.add_argument("--m", type=int, default=0)
    p.add_argument("--draw", type=int, default=0)
    p.add_argument("--method", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.phase == "select":
        phase_select()
    elif a.phase == "generate":
        phase_generate(a.cat)
    elif a.phase == "zeroshot":
        phase_zeroshot(a.cat, a.map, a.device)
    elif a.phase == "adapt":
        phase_adapt(a.cat, a.m, a.draw, a.method, a.seed, a.device)
    elif a.phase == "evaladapted":
        phase_evaladapted(a.cat, a.m, a.draw, a.method, a.seed, a.device)


if __name__ == "__main__":
    main()
