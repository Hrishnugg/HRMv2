#!/usr/bin/env python3
"""C8-S v2: scale study with shared worlds across graph sizes.

Frozen design 2026-07-26-c8-scale-walltime.md + Amendment 1 (v12-review
refinements, adopted before any v2 execution):

- ONE fresh world cohort per suite (cfg.seed = 5_000_000, the canonical
  iter_dynamic_worlds seed formula), evaluated at ALL four node counts:
  a world is accepted iff build_prm builds AND connects at every size in
  {192, 512, 1024, 2048} (k=7, per-size roadmap seed = world seed). This
  removes the world-difficulty confound across sizes; roadmaps are
  independently sampled per size via the existing validated builder
  (nested prefixes declined: reusing build_prm verbatim beats a custom
  sampler's semantic risk).
- Timing: eval shards run on one L4 container (CPU classical arms + GPU
  learned variant on identical hardware); arms per world = euclid, WA*,
  learned_cpu (table+search), learned_gpu (same code path, device=cuda),
  SIPP (hard-gated). Three repeats per world, arm order randomized per
  (world, repeat) with a seeded RNG; first world per shard flagged warmup.
- Density probe phase: predicted-vs-true residual correlation / MAE /
  bias over reachable (v,t) states on 5 eval worlds per (size, suite).
- Crossover (frozen analysis definition): smallest n whose paired
  learned_gpu-minus-WA* total-time 95% CI lies below zero with success
  noninferior within 0.05 and mean path suboptimality within +0.02.

Phases: manifest (per suite), calib (per size,suite), tune, eval, probe.
Outputs under runs/c8s2_scale/. All phases idempotent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

import continuous_prm_c8_dynamic_maps as M8MAPS
import continuous_prm_c8_dynamics_compare as M8C
import continuous_prm_c8_sipp_baseline as SIPPB
import continuous_prm_c6_heatmap_value_field as C6
import continuous_prm_common as C
import continuous_prm_dynamic_providers as DP
import continuous_prm_spacetime as ST

M8MAPS.install_c8_dynamic_maps()

HERE = Path(__file__).parent
ROOT = Path(os.environ.get("C8S_ROOT", str(HERE)))
OUT = ROOT / "runs" / "c8s2_scale"
SUITES = ["C_dyn_maze", "C_dyn_rooms", "C_dyn_spiral",
          "C_dyn_maze_dense", "C_dyn_crossing", "C_dyn_rooms_large"]
SIZES = [192, 512, 1024, 2048]
PAPER_GRID = [150, 250, 400, 600, 900, 1300, 1800, 2500, 3500]
WEIGHTS = [1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
ANCHOR_TARGET = {"C_dyn_maze": 0.12, "C_dyn_rooms": 0.42, "C_dyn_spiral": 0.16,
                 "C_dyn_maze_dense": 0.06, "C_dyn_crossing": 0.12,
                 "C_dyn_rooms_large": 0.82}
N_DEV, N_EVAL = 10, 30
SEED_BASE = 5_000_000
N_REPEATS = 3
CKPT_SHA = ("b8378950545f8abdbf06d59568b5d8ab"
            "6069884c1c038d23a41a14b6dc17fb6f")

EVAL_COLS = ["size", "suite", "world_index", "warmup", "repeat", "order",
             "arm", "found", "expansions", "arrival", "optimal_arrival",
             "budget", "w", "t_table_s", "t_search_s", "t_total_s", "cpu",
             "gpu"]
PROBE_COLS = ["size", "suite", "world_index", "n_states", "pearson_r",
              "mae", "bias"]


def scaled_grid(n: int):
    return [max(10, int(round(b * n / 192 / 10.0)) * 10) for b in PAPER_GRID]


def iter_shared_worlds(suite: str, count: int):
    """Yield (wi, world, dyn, {n: rm}) for worlds valid at ALL sizes."""
    suite_idx = SUITES.index(suite)
    valid, attempt = 0, 0
    while valid < count and attempt < count * 60:
        seed = (SEED_BASE + 880_000 + 1_000_003 * (suite_idx + 1)
                + (valid + 1) * 7919 + attempt)
        attempt += 1
        res = M8MAPS.build_dynamic_world(suite, seed)
        if res is None:
            continue
        world, dyn = res
        rms = {}
        ok = True
        for n in SIZES:
            rm = C.build_prm(world, C.RoadmapConfig(n_nodes=n, k_neighbors=7),
                             seed=seed)
            if rm is None or not bool(rm.connected_to_goal[0]):
                ok = False
                break
            rms[n] = rm
        if not ok:
            continue
        yield valid, world, dyn, rms
        valid += 1


def world_hash(world, dyn, rms, tag: str) -> str:
    parts = [tag, f"start={world.start.tolist()}",
             f"goal={world.goal.tolist()}"]
    for o in world.obstacles:
        parts.append(f"obs={o.kind},{o.cx:.9f},{o.cy:.9f},"
                     f"{getattr(o, 'radius', 0.0):.9f},"
                     f"{getattr(o, 'hw', 0.0):.9f},{getattr(o, 'hh', 0.0):.9f}")
    for c in dyn.circles:
        parts.append(f"pat={c.ax:.9f},{c.ay:.9f},{c.bx:.9f},{c.by:.9f},"
                     f"{c.radius:.9f},{c.period:.9f}")
    for n in SIZES:
        parts.append(f"pts{n}=" + hashlib.sha256(
            np.ascontiguousarray(rms[n].points).tobytes()).hexdigest())
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def load_learned(device):
    cands = [ROOT / "runs" / "c8_local_heavy" / "checkpoints" /
             "c8_field__unet_blind.pt",
             ROOT / "runs" / "c14_sources" / "c8_local_heavy" / "checkpoints" /
             "c8_field__unet_blind.pt"]
    ck = next(p for p in cands if p.exists())
    got = hashlib.sha256(ck.read_bytes()).hexdigest()
    if got != CKPT_SHA:
        raise SystemExit(f"ABORT: checkpoint hash mismatch at {ck}: {got}")
    pl = torch.load(ck, map_location="cpu", weights_only=True)
    model = C6.build_model(pl["backbone"], in_channels=pl["in_channels"])
    model.load_state_dict(pl["model"])
    model.to(device).eval()
    return DP.ValueFieldTemporalProvider(model, pl["grid_size"], device,
                                         pl["backbone"], pl["window_w"],
                                         time_blind=True)


def suite_dims(suite: str):
    p = M8MAPS.dynamics_params(suite)
    return float(p["v_agent"]), float(p["dt"]), int(p["t_max"])


def phase_manifest(suite: str) -> Path:
    out = OUT / f"manifest_{suite}.json"
    if out.exists():
        print(f"[c8s2] {out.name} exists; skip", flush=True)
        return out
    entries = {}
    for wi, world, dyn, rms in iter_shared_worlds(suite, N_DEV + N_EVAL):
        role = "dev" if wi < N_DEV else "eval"
        entries[str(wi)] = {"role": role,
                            "hash": world_hash(world, dyn, rms,
                                               f"{SEED_BASE};{suite}")}
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"suite": suite, "seed": SEED_BASE,
                               "sizes": SIZES, "worlds": entries}, indent=1))
    print(f"[c8s2] wrote {out.name} ({len(entries)} shared worlds)", flush=True)
    return out


def phase_calib(n: int, suite: str) -> Path:
    out = OUT / f"calib_{n}_{suite}.json"
    if out.exists():
        print(f"[c8s2] {out.name} exists; skip", flush=True)
        return out
    v_agent, dt, t_max = suite_dims(suite)
    grid = scaled_grid(n)
    anchor = DP.EuclidTimeProvider()
    solves = []
    for wi, world, dyn, rms in iter_shared_worlds(suite, N_DEV):
        rm = rms[n]
        h = anchor.h_table(world, rm, dyn, v_agent, dt, t_max, goal_idx=1)
        res = ST.space_time_astar_prm(rm.adj, rm.points, dyn, h, grid[-1],
                                      v_agent, dt, t_max, 0, 1)
        solves.append(int(res["expansions"]) if res["found"] else grid[-1] + 1)
    OUT.mkdir(parents=True, exist_ok=True)
    target = ANCHOR_TARGET[suite]
    rates = {b: float(np.mean([s <= b for s in solves])) for b in grid}
    best_b = min(grid, key=lambda b: (abs(rates[b] - target), b))
    out.write_text(json.dumps({"size": n, "suite": suite, "budget": best_b,
                               "target": target, "rates": rates,
                               "dev_solve_expansions": solves}, indent=1))
    print(f"[c8s2] calib {n}/{suite}: budget={best_b} "
          f"(dev {rates[best_b]:.2f} vs target {target})", flush=True)
    return out


def phase_tune(n: int, suite: str) -> Path:
    out = OUT / f"tune_{n}_{suite}.json"
    if out.exists():
        print(f"[c8s2] {out.name} exists; skip", flush=True)
        return out
    budget = json.loads((OUT / f"calib_{n}_{suite}.json").read_text())["budget"]
    v_agent, dt, t_max = suite_dims(suite)
    anchor = DP.EuclidTimeProvider()
    succ = {}
    for w in WEIGHTS:
        wins = 0
        for wi, world, dyn, rms in iter_shared_worlds(suite, N_DEV):
            rm = rms[n]
            h = anchor.h_table(world, rm, dyn, v_agent, dt, t_max, goal_idx=1)
            res = ST.space_time_astar_prm(rm.adj, rm.points, dyn, w * h,
                                          budget, v_agent, dt, t_max, 0, 1)
            wins += int(bool(res["found"]))
        succ[w] = wins / N_DEV
    best_w = max(WEIGHTS, key=lambda w: (succ[w], -w))
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"size": n, "suite": suite, "budget": budget,
                               "w": best_w, "dev_success": succ}, indent=1))
    print(f"[c8s2] tune {n}/{suite}: w={best_w}", flush=True)
    return out


def phase_eval(n: int, suite: str, smoke: int = 0) -> Path:
    out = OUT / (f"eval_{n}_{suite}.csv" if not smoke
                 else f"smoke_eval_{n}_{suite}.csv")
    if out.exists() and not smoke:
        print(f"[c8s2] {out.name} exists; skip", flush=True)
        return out
    n_eval = smoke if smoke else N_EVAL
    budget = json.loads((OUT / f"calib_{n}_{suite}.json").read_text())["budget"]
    wsel = json.loads((OUT / f"tune_{n}_{suite}.json").read_text())["w"]
    v_agent, dt, t_max = suite_dims(suite)
    anchor = DP.EuclidTimeProvider()
    learned_cpu = load_learned("cpu")
    has_gpu = torch.cuda.is_available()
    learned_gpu = load_learned("cuda") if has_gpu else None
    cpu = platform.processor() or platform.machine()
    gpu = torch.cuda.get_device_name(0) if has_gpu else ""
    order_rng = np.random.default_rng(20260726 + n + SUITES.index(suite))
    rows, gate_failures = [], 0
    for wi, world, dyn, rms in iter_shared_worlds(suite, N_DEV + n_eval):
        if wi < N_DEV:
            continue
        rm = rms[n]
        warmup = int(wi == N_DEV)
        hstar = ST.backward_spacetime_dijkstra(rm.adj, rm.points, dyn,
                                               v_agent, dt, t_max, goal=1)
        opt = hstar[0, 0]
        opt_arrival = int(opt) if np.isfinite(opt) and opt < 1e29 else -1
        h_anchor = anchor.h_table(world, rm, dyn, v_agent, dt, t_max,
                                  goal_idx=1)

        def run_arm(arm):
            if arm == "euclid":
                t0 = time.perf_counter()
                r = ST.space_time_astar_prm(rm.adj, rm.points, dyn, h_anchor,
                                            budget, v_agent, dt, t_max, 0, 1)
                t1 = time.perf_counter()
                return r, "", 0.0, t1 - t0
            if arm == "wastar":
                t0 = time.perf_counter()
                r = ST.space_time_astar_prm(rm.adj, rm.points, dyn,
                                            wsel * h_anchor, budget,
                                            v_agent, dt, t_max, 0, 1)
                t1 = time.perf_counter()
                return r, wsel, 0.0, t1 - t0
            if arm in ("learned_cpu", "learned_gpu"):
                prov = learned_cpu if arm == "learned_cpu" else learned_gpu
                t0 = time.perf_counter()
                h = prov.h_table(world, rm, dyn, v_agent, dt, t_max,
                                 goal_idx=1)
                if arm == "learned_gpu":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                r = ST.space_time_astar_prm(rm.adj, rm.points, dyn, h, budget,
                                            v_agent, dt, t_max, 0, 1)
                t2 = time.perf_counter()
                return r, "", t1 - t0, t2 - t1
            if arm == "sipp":
                dynp = SIPPB._PredCounter(dyn)
                t0 = time.perf_counter()
                iv = SIPPB.build_intervals(dynp, rm.points, dt, t_max)
                t1 = time.perf_counter()
                r = SIPPB.sipp(rm.adj, rm.points, dynp, iv, h_anchor[:, 0],
                               v_agent, dt, t_max, 0, 1)
                t2 = time.perf_counter()
                return r, "", t1 - t0, t2 - t1
            raise ValueError(arm)

        arms = ["euclid", "wastar", "learned_cpu", "sipp"]
        if has_gpu:
            arms.append("learned_gpu")
        for rep in range(N_REPEATS):
            order = list(order_rng.permutation(arms))
            for oi, arm in enumerate(order):
                r, w, tt, tsr = run_arm(arm)
                if arm == "sipp" and rep == 0:
                    ok = ((r["found"] and r["arrival"] == opt_arrival) or
                          (not r["found"] and opt_arrival < 0))
                    if not ok:
                        gate_failures += 1
                        print(f"[c8s2] SIPP GATE FAIL {n}/{suite}/{wi}",
                              flush=True)
                rows.append(dict(
                    size=n, suite=suite, world_index=wi, warmup=warmup,
                    repeat=rep, order=oi, arm=arm, found=bool(r["found"]),
                    expansions=int(r["expansions"]), arrival=int(r["arrival"]),
                    optimal_arrival=opt_arrival, budget=budget, w=w,
                    t_table_s=round(tt, 5), t_search_s=round(tsr, 5),
                    t_total_s=round(tt + tsr, 5), cpu=cpu, gpu=gpu))
    if gate_failures:
        raise SystemExit(f"ABORT: {gate_failures} SIPP gate failures "
                         f"{n}/{suite}")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=EVAL_COLS)
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    print(f"[c8s2] wrote {out.name} ({len(rows)} rows; gates clean)",
          flush=True)
    return out


def phase_probe(n: int, suite: str) -> Path:
    out = OUT / f"probe_{n}_{suite}.csv"
    if out.exists():
        print(f"[c8s2] {out.name} exists; skip", flush=True)
        return out
    v_agent, dt, t_max = suite_dims(suite)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learned = load_learned(device)
    anchor = DP.EuclidTimeProvider()
    rows = []
    for wi, world, dyn, rms in iter_shared_worlds(suite, N_DEV + 5):
        if wi < N_DEV:
            continue
        rm = rms[n]
        hstar = ST.backward_spacetime_dijkstra(rm.adj, rm.points, dyn,
                                               v_agent, dt, t_max, goal=1)
        ttg = ST.oracle_time_to_go(hstar, t_max)
        reach = np.isfinite(ttg) & (ttg < 1e29)
        h_anchor = anchor.h_table(world, rm, dyn, v_agent, dt, t_max,
                                  goal_idx=1)
        T_scale = float(world.side_len) / v_agent / dt
        true_resid = np.clip(ttg - h_anchor, 0.0, 4.0 * T_scale) / T_scale
        h_pred = learned.h_table(world, rm, dyn, v_agent, dt, t_max,
                                 goal_idx=1)
        pred_resid = (h_pred - h_anchor) / T_scale
        t, p = true_resid[reach], pred_resid[reach]
        r = float(np.corrcoef(t, p)[0, 1]) if len(t) > 1 else float("nan")
        rows.append(dict(size=n, suite=suite, world_index=wi,
                         n_states=int(reach.sum()), pearson_r=round(r, 4),
                         mae=round(float(np.mean(np.abs(p - t))), 4),
                         bias=round(float(np.mean(p - t)), 4)))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=PROBE_COLS)
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    print(f"[c8s2] wrote {out.name}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True,
                   choices=["manifest", "calib", "tune", "eval", "probe",
                            "sens", "senssel"])
    p.add_argument("--size", type=int, default=0)
    p.add_argument("--suite", default="")
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--stage", default="dev")
    p.add_argument("--arm", default="euclid")
    a = p.parse_args()
    if a.phase == "manifest":
        phase_manifest(a.suite)
    elif a.phase == "calib":
        phase_calib(a.size, a.suite)
    elif a.phase == "tune":
        phase_tune(a.size, a.suite)
    elif a.phase == "eval":
        phase_eval(a.size, a.suite, a.smoke)
    elif a.phase == "probe":
        phase_probe(a.size, a.suite)
    elif a.phase == "sens":
        phase_sens(a.size, a.suite, a.stage, a.arm)
    elif a.phase == "senssel":
        sens_select(a.size, a.suite)


# --- Amendment 2 (2026-07-27): sensitivity recalibration, success-only ---

SENS_FLOOR_CELLS = [(192, "C_dyn_maze_dense"), (512, "C_dyn_maze_dense"),
                    (1024, "C_dyn_maze_dense"), (2048, "C_dyn_maze_dense"),
                    (2048, "C_dyn_spiral")]
SENS_WEIGHTS = [1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
SENS_BAND = (0.30, 0.70)


def sens_cap(n: int, suite: str) -> int:
    _, _, t_max = suite_dims(suite)
    return n * (t_max + 1)


def sens_ladder(n: int, suite: str):
    g = scaled_grid(n)
    cap = sens_cap(n, suite)
    return sorted(b for b in set(g + [2 * g[-1], 4 * g[-1]]) if b <= cap)


def phase_sens(n: int, suite: str, stage: str, arm: str) -> Path:
    """One arm over dev or eval worlds at the state cap (Amendment 2).
    Success at any lower budget derives by thresholding solve expansions."""
    out = OUT / f"sens_{stage}_{arm}_{n}_{suite}.csv"
    if out.exists():
        print(f"[c8s2] {out.name} exists; skip", flush=True)
        return out
    v_agent, dt, t_max = suite_dims(suite)
    cap = sens_cap(n, suite)
    anchor = DP.EuclidTimeProvider()
    learned = load_learned("cpu") if arm == "learned_cpu" else None
    w = None
    if arm == "wastar_sel":
        sel = json.loads((OUT / f"senssel_{n}_{suite}.json").read_text())
        w = float(sel["w_star"])
    elif arm.startswith("wastar_"):
        w = float(arm.split("_", 1)[1])
    lo, hi = (0, N_DEV) if stage == "dev" else (N_DEV, N_DEV + N_EVAL)
    rows = []
    for wi, world, dyn, rms in iter_shared_worlds(suite, N_DEV + N_EVAL):
        if not (lo <= wi < hi):
            continue
        rm = rms[n]
        h = (learned if learned is not None else anchor).h_table(
            world, rm, dyn, v_agent, dt, t_max, goal_idx=1)
        if w is not None:
            h = w * h
        r = ST.space_time_astar_prm(rm.adj, rm.points, dyn, h, cap,
                                    v_agent, dt, t_max, 0, 1)
        rows.append(dict(size=n, suite=suite, world_index=wi, warmup=0,
                         repeat=0, order=0, arm=arm, found=bool(r["found"]),
                         expansions=int(r["expansions"]),
                         arrival=int(r["arrival"]), optimal_arrival=-9,
                         budget=cap, w="" if w is None else w,
                         t_table_s=0.0, t_search_s=0.0, t_total_s=0.0,
                         cpu="", gpu=""))
        print(f"[c8s2] sens {stage}/{arm} {n}/{suite} world {wi} "
              f"found={rows[-1]['found']} exp={rows[-1]['expansions']}",
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=EVAL_COLS)
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    print(f"[c8s2] wrote {out.name}", flush=True)
    return out


def sens_select(n: int, suite: str) -> Path:
    """Frozen Amendment-2 rule: smallest ladder budget with dev anchor
    success in [0.30, 0.70]; else smallest with >= 0.30; else
    anchor-infeasible (binding = cap). w* re-tuned at the binding."""
    out = OUT / f"senssel_{n}_{suite}.json"
    if out.exists():
        print(f"[c8s2] {out.name} exists; skip", flush=True)
        return out

    def solve_exps(arm):
        p = OUT / f"sens_dev_{arm}_{n}_{suite}.csv"
        with open(p, newline="") as f:
            return [(r["found"] == "True", int(r["expansions"]))
                    for r in csv.DictReader(f)]

    def succ_at(recs, budget):
        return float(np.mean([f and e <= budget for f, e in recs]))

    cap = sens_cap(n, suite)
    ladder = sens_ladder(n, suite)
    eu = solve_exps("euclid")
    rates = {b: succ_at(eu, b) for b in ladder}
    in_band = [b for b in ladder
               if SENS_BAND[0] <= rates[b] <= SENS_BAND[1]]
    above = [b for b in ladder if rates[b] >= SENS_BAND[0]]
    if in_band:
        binding, status = in_band[0], "in_band"
    elif above:
        binding, status = above[0], "above_band"
    else:
        binding, status = cap, "anchor_infeasible"
    w_rates = {}
    for wv in SENS_WEIGHTS:
        w_rates[f"{wv:g}"] = succ_at(solve_exps(f"wastar_{wv:g}"), binding)
    w_star = max(SENS_WEIGHTS, key=lambda wv: (w_rates[f"{wv:g}"], -wv))
    out.write_text(json.dumps(dict(
        size=n, suite=suite, cap=cap, ladder=ladder, anchor_rates=rates,
        binding=binding, status=status, w_rates=w_rates, w_star=w_star),
        indent=1))
    print(f"[c8s2] senssel {n}/{suite}: binding {binding} ({status}) "
          f"w*={w_star:g}", flush=True)
    return out


if __name__ == "__main__":
    main()
