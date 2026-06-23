#!/usr/bin/env python3
"""Local matched-expansion benchmark for learned focal search (no Modal compute).

Compares, on identical instances (same seeds):
  - baseline: Manhattan A* (model=None, PLANNER=astar)
  - focal-learned: focal search ordered by the model's signal (PLANNER=focal)
across map scales and a sweep of w. Reports per-(suite,w) median expansion ratio
(focal/baseline) and success rates. Headline metric: ratio < 1 means fewer expansions.

Setup (one-time): download a checkpoint locally, e.g.
  python -m modal volume get residual-tasklora-v2-vol \
    residual_tasklora_v2/runs/residual_tasklora_v2/models/avgbase__hrm__ALL_TASKS.pt ./ckpts/

Usage:
  python hrm-cloud/bench_focal.py --ckpt ckpts/avgbase__hrm__ALL_TASKS.pt \
    --suites ID_A64_static,OOD_A128_static,OOD_A192_static --seeds 5 --budget 500 \
    --w 1.0,1.1,1.25,1.5,2.0 --device cuda
"""
import argparse, statistics as st
import torch
import residual_tasklora_v2 as R
from residual_tasklora_v2 import BackboneConfig, CleanHeuristicModel


def load_model(ckpt, device):
    payload = torch.load(ckpt, map_location="cpu")
    cfg = BackboneConfig(**payload["cfg"])
    m = CleanHeuristicModel(cfg)
    m.load_state_dict(payload["model_state"], strict=False)
    m.to(device).eval()
    m.arm = "avgbase"
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suites", default="ID_A64_static,OOD_A128_static,OOD_A192_static")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--w", default="1.0,1.1,1.25,1.5,2.0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    R.EVAL_DIAG = False  # fast path; we only need outcomes
    suites = {s.suite_id: s for s in R.build_eval_suites(True, 100)}
    model = load_model(args.ckpt, args.device)
    ws = [float(x) for x in args.w.split(",")]
    print(f"device={args.device} budget={args.budget} seeds={args.seeds}")
    print(f"{'suite':20s} {'w':>5s} {'exp_ratio(med)':>14s} {'succ_base':>9s} {'succ_focal':>10s}")
    for sid in args.suites.split(","):
        s = suites[sid]
        R.PLANNER = "astar"
        base = [R.run_policy_episode(s, seed=i, model=None, alpha=1.0, max_expansions=args.budget, device="cpu")
                for i in range(args.seeds)]
        for w in ws:
            R.PLANNER = "focal"; R.FOCAL_W = w
            foc = [R.run_policy_episode(s, seed=i, model=model, alpha=1.0, max_expansions=args.budget, device=args.device)
                   for i in range(args.seeds)]
            ratios = [f["expansions"] / max(1, b["expansions"]) for f, b in zip(foc, base)]
            sb = sum(b["success"] for b in base) / len(base)
            sf = sum(f["success"] for f in foc) / len(foc)
            print(f"{sid:20s} {w:5.2f} {st.median(ratios):14.2f} {sb:9.2f} {sf:10.2f}")


if __name__ == "__main__":
    main()
