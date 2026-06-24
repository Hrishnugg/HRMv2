#!/usr/bin/env python3
"""Local matched-expansion benchmark for learned focal search (no Modal compute).

Compares, on identical instances (same seeds):
  - baseline: Manhattan A* (model=None, PLANNER=astar)
  - focal-base: focal search ordered by the base model's signal (PLANNER=focal)
  - focal-expert [optional]: focal search ordered by a LoRA expert model (PLANNER=focal)
across map scales and a sweep of w. Reports per-(suite,w) median expansion ratio
(focal/baseline) and success rates. Headline metric: ratio < 1 means fewer expansions.

Setup (one-time): download checkpoints locally, e.g.
  python -m modal volume get residual-tasklora-v2-vol \
    residual_tasklora_v2/runs/residual_tasklora_v2/models/avgbase__hrm__ALL_TASKS.pt ./ckpts/

Usage (base only):
  python hrm-cloud/bench_focal.py --ckpt ckpts/avgbase__hrm__ALL_TASKS.pt \
    --suites ID_A64_static,OOD_A128_static,OOD_A192_static --seeds 5 --budget 500 \
    --w 1.0,1.1,1.25,1.5,2.0 --device cuda

Usage (base + expert):
  python hrm-cloud/bench_focal.py --ckpt ckpts/avgbase__hrm__ALL_TASKS.pt \
    --expert-ckpt ckpts/residtasklora__hrm__A32_static.pt \
    --base-ckpt ckpts/avgbase__hrm__ALL_TASKS.pt \
    --suites ID_A64_static --seeds 5 --budget 500 --w 1.0 --device cuda
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


def load_expert(expert_ckpt, base_ckpt, device):
    # Mirrors residual_tasklora_v2._load_model_for_eval's residtasklora branch, but uses
    # a LOCAL base checkpoint instead of the remote /data path in metrics["base_model_path"].
    ep = torch.load(expert_ckpt, map_location="cpu")
    cfg = BackboneConfig(**ep["cfg"])
    bp = torch.load(base_ckpt, map_location="cpu")
    base_cfg = BackboneConfig(**bp["cfg"])
    base_model = CleanHeuristicModel(base_cfg)
    base_model.load_state_dict(bp["model_state"], strict=False)
    base_model.to(device).eval()
    for p in base_model.parameters():
        p.requires_grad = False
    adapt = CleanHeuristicModel(cfg)
    R._apply_stacked_lora(adapt, cfg.lora_rank, R.LORA_ALPHA, 1,
                          include_conv=R.LORA_ON_CONV, include_attn=R.LORA_ON_ATTN,
                          init_scale=R.LORA_INIT_SCALE)
    adapt.load_state_dict(ep["model_state"], strict=False)
    adapt.to(device).eval()
    for p in adapt.parameters():
        p.requires_grad = False
    return R.BoundedResidualExpertPolicy(
        base_model=base_model, adapt_model=adapt,
        bound_B=float(ep["metrics"].get("bound_B", R.RESIDUAL_BOUND_MAX)),
        metadata=ep["metrics"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--expert-ckpt", default="", help="Path to LoRA expert checkpoint (optional)")
    ap.add_argument("--base-ckpt", default="", help="Local base ckpt for expert loading; defaults to --ckpt")
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

    expert_model = None
    if args.expert_ckpt:
        expert_model = load_expert(
            args.expert_ckpt,
            args.base_ckpt or args.ckpt,
            args.device,
        )

    print(f"device={args.device} budget={args.budget} seeds={args.seeds}")
    if expert_model is not None:
        print(f"{'suite':20s} {'w':>5s} {'base_ratio':>10s} {'base_sc':>8s} {'exp_ratio':>10s} {'exp_sc':>8s}")
    else:
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

            if expert_model is not None:
                R.PLANNER = "focal"; R.FOCAL_W = w
                exp = [R.run_policy_episode(s, seed=i, model=expert_model, alpha=1.0, max_expansions=args.budget, device=args.device)
                       for i in range(args.seeds)]
                exp_ratios = [e["expansions"] / max(1, b["expansions"]) for e, b in zip(exp, base)]
                se = sum(e["success"] for e in exp) / len(exp)
                print(f"{sid:20s} {w:5.2f} {st.median(ratios):10.2f} {sf:8.2f} {st.median(exp_ratios):10.2f} {se:8.2f}")
            else:
                print(f"{sid:20s} {w:5.2f} {st.median(ratios):14.2f} {sb:9.2f} {sf:10.2f}")


if __name__ == "__main__":
    main()
