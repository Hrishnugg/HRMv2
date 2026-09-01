#!/usr/bin/env python3
"""SIPP baseline on the C8 confirmation cohort (frozen design 2026-07-25).

Discrete-time Safe Interval Path Planning on the identical substrate, using
the space-time planner's exact collision predicates:

- waiting at u over [t, t+1) requires dyn.node_free(points[u], t*dt, (t+1)*dt,
  samples=8) -- identical to the space-time A* wait action;
- traversing (u, v) departing at d requires dyn.edge_free(points[u], points[v],
  d*dt, (d+tau)*dt, samples=max(8, 4*tau)) -- identical to the move action;
- arriving at a node never requires a node check (pass-through), matching the
  space-time graph, so both planners search the same reachability relation and
  must agree on earliest arrival.

Intervals: per node, maximal runs of wait-safe steps plus singleton
pass-through intervals for non-wait-safe steps. A SIPP state is
(node, interval); from earliest arrival a inside interval [s, r] the feasible
departures are [a, r+1] (each waited step must be wait-safe), or {a} alone in
a singleton. Successors take the earliest feasible departure per
(edge, target-interval); f = arrival + h0 with the canonical Euclidean
time-lower-bound anchor, which is admissible for earliest arrival.

Correctness gate (hard): on every instance, SIPP's earliest arrival must equal
the backward space-time Dijkstra optimum hstar[start, 0] (both optimal), and
solvability must agree. Any mismatch aborts the run.

Read-only with respect to all frozen artifacts. CPU only.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import continuous_prm_c8_dynamic_maps as M8MAPS
import continuous_prm_c8_dynamics_compare as M8C
import continuous_prm_spacetime as ST
import continuous_prm_dynamic_providers as DP

# Canonical eval order (MUST match C8Config.eval_suites so suite_idx-derived
# world seeds reproduce the frozen confirmation cohort; a prior version used
# an alphabetical list and evaluated a sibling cohort -- disclosed erratum).
SUITES = ["C_dyn_maze", "C_dyn_rooms", "C_dyn_spiral",
          "C_dyn_maze_dense", "C_dyn_crossing", "C_dyn_rooms_large"]
BINDING = {"C_dyn_crossing": 150, "C_dyn_maze": 1800, "C_dyn_maze_dense": 2500,
           "C_dyn_rooms": 1300, "C_dyn_rooms_large": 600, "C_dyn_spiral": 2500}
OUT = Path(__file__).parent / "runs" / "c8r_sipp"

COLS = ["suite", "world_index", "found", "arrival", "optimal_arrival",
        "gate_ok", "sipp_expansions", "predicate_calls", "n_intervals",
        "t_intervals_s", "t_search_s"]


class _PredCounter:
    def __init__(self, dyn):
        self.dyn = dyn
        self.calls = 0

    def node_free(self, *a, **k):
        self.calls += 1
        return self.dyn.node_free(*a, **k)

    def edge_free(self, *a, **k):
        self.calls += 1
        return self.dyn.edge_free(*a, **k)


def build_intervals(dynp, points, dt: float, t_max: int) -> List[List[Tuple[int, int, bool]]]:
    """Per node: list of (start, end, waitable) intervals partitioning [0, t_max].

    Waitable interval [s, r]: every step t in [s, r] satisfies the wait
    predicate over [t, t+1). Non-wait-safe steps become singleton
    pass-through intervals (waitable=False).
    """
    n = points.shape[0]
    out: List[List[Tuple[int, int, bool]]] = []
    for u in range(n):
        safe = [dynp.node_free(points[u], t * dt, (t + 1) * dt, samples=8)
                for t in range(t_max + 1)]
        iv: List[Tuple[int, int, bool]] = []
        t = 0
        while t <= t_max:
            if safe[t]:
                r = t
                while r + 1 <= t_max and safe[r + 1]:
                    r += 1
                iv.append((t, r, True))
                t = r + 1
            else:
                iv.append((t, t, False))
                t += 1
        out.append(iv)
    return out


def _interval_index(intervals: List[Tuple[int, int, bool]], t: int) -> int:
    lo, hi = 0, len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, r, _ = intervals[mid]
        if t < s:
            hi = mid - 1
        elif t > r:
            lo = mid + 1
        else:
            return mid
    raise ValueError(f"time {t} outside interval partition")


def sipp(adj, points, dynp, intervals, h0, v_agent: float, dt: float,
         t_max: int, start: int = 0, goal: int = 1,
         max_expansions: int = 1_000_000) -> Dict[str, int]:
    """SIPP earliest-arrival search. h0[node] = time-to-go lower bound (steps)."""
    start_iv = _interval_index(intervals[start], 0)
    best: Dict[Tuple[int, int], int] = {(start, start_iv): 0}
    counter = 0
    openq: List[Tuple[float, int, int, int, int]] = [
        (float(h0[start]), counter, start, start_iv, 0)]
    closed: set = set()
    expansions = 0

    while openq and expansions < max_expansions:
        f, _, u, ui, a = heapq.heappop(openq)
        if (u, ui) in closed:
            continue
        closed.add((u, ui))
        expansions += 1
        if u == goal:
            return {"found": True, "arrival": int(a), "expansions": expansions}

        s, r, waitable = intervals[u][ui]
        d_hi_wait = (r + 1) if waitable else a  # latest departure via waiting
        for v, length in adj[u]:
            tau = ST._edge_steps(float(length), v_agent, dt)
            d_lo = a
            d_hi = min(d_hi_wait, t_max - tau)
            if d_hi < d_lo:
                continue
            # Waiting past r requires step r to be wait-safe (it is, within a
            # waitable interval); departing exactly at r+1 is legal because the
            # wait through [r, r+1) stayed inside the interval.
            for vi, (vs, vr, _vw) in enumerate(intervals[v]):
                if (v, vi) in closed:
                    continue
                lo = max(d_lo, vs - tau)
                hi = min(d_hi, vr - tau)
                if hi < lo:
                    continue
                arr = None
                for d in range(lo, hi + 1):
                    if dynp.edge_free(points[u], points[v], d * dt,
                                      (d + tau) * dt, samples=max(8, 4 * tau)):
                        arr = d + tau
                        break
                if arr is None:
                    continue
                if arr < best.get((v, vi), 1 << 30):
                    best[(v, vi)] = arr
                    counter += 1
                    heapq.heappush(openq, (arr + float(h0[v]), counter, v, vi, arr))
    return {"found": False, "arrival": -1, "expansions": expansions}


def run_cohort(seed: int, n_worlds: int, tag: str, smoke: int = 0) -> Path:
    cfg = M8C.C8Config(seed=int(seed), eval_worlds=int(n_worlds))
    cfg = M8C.apply_scale_preset(cfg)
    cfg.eval_worlds = int(n_worlds)
    anchor = DP.EuclidTimeProvider()
    rows = []
    gate_failures = 0
    for suite_idx, suite in enumerate(SUITES):
        params = M8MAPS.dynamics_params(suite)
        v_agent, dt, t_max = float(params["v_agent"]), float(params["dt"]), int(params["t_max"])
        count = 0
        for wi, world, dyn, rm in M8C.iter_dynamic_worlds(suite, suite_idx, cfg, n_worlds):
            if smoke and count >= smoke:
                break
            count += 1
            h_tab = anchor.h_table(world, rm, dyn, v_agent, dt, t_max, goal_idx=1)
            h0 = h_tab[:, 0]  # constant in t
            hstar = ST.backward_spacetime_dijkstra(
                rm.adj, rm.points, dyn, v_agent, dt, t_max, goal=1)
            opt = hstar[0, 0]
            opt_arrival = int(opt) if np.isfinite(opt) and opt < 1e29 else -1

            dynp = _PredCounter(dyn)
            t0 = time.perf_counter()
            intervals = build_intervals(dynp, rm.points, dt, t_max)
            t1 = time.perf_counter()
            res = sipp(rm.adj, rm.points, dynp, intervals, h0,
                       v_agent, dt, t_max, 0, 1)
            t2 = time.perf_counter()

            gate_ok = ((res["found"] and res["arrival"] == opt_arrival) or
                       (not res["found"] and opt_arrival < 0))
            if not gate_ok:
                gate_failures += 1
                print(f"[sipp] GATE FAIL {suite} world {wi}: sipp="
                      f"{res['arrival'] if res['found'] else 'unsolved'} "
                      f"opt={opt_arrival}", flush=True)
            rows.append(dict(
                suite=suite, world_index=wi, found=bool(res["found"]),
                arrival=int(res["arrival"]), optimal_arrival=opt_arrival,
                gate_ok=bool(gate_ok), sipp_expansions=int(res["expansions"]),
                predicate_calls=int(dynp.calls),
                n_intervals=int(sum(len(iv) for iv in intervals)),
                t_intervals_s=round(t1 - t0, 4), t_search_s=round(t2 - t1, 4)))
        print(f"[sipp] {tag} {suite}: {count} worlds done", flush=True)

    if gate_failures:
        raise SystemExit(f"ABORT: {gate_failures} correctness-gate failures -- "
                         "results are not reportable")
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"{tag}_raw.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[sipp] wrote {out_csv} ({len(rows)} rows; all gates passed)", flush=True)
    return out_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["smoke", "conf"], default="conf")
    args = p.parse_args()
    M8MAPS.install_c8_dynamic_maps()
    if args.phase == "smoke":
        run_cohort(999999, 50, "smoke", smoke=5)
    else:
        run_cohort(999999, 50, "confirmation")


if __name__ == "__main__":
    main()
