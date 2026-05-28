#!/usr/bin/env python3
"""
Gradated continuous-PRM experiment runner for experiments C1-C4.

This runner imports `continuous_prm_common.py`, which contains the continuous
2D PRM world generator, feature encoder, HRM/ONLSTM models, LoRA machinery,
training helpers, and the C3 expert-matrix evaluator. The runner adds a clean
stage interface:

  C1: Euclidean PRM baseline only.
  C2: pooled avgbase HRM/ONLSTM + Euclidean baseline.
  C3: bounded residual LoRA experts + matched/cross/oracle diagnostics.
  C4: descriptor-based RBF mixture of learned residual LoRA experts.

The files are intentionally self-contained enough for Modal/local use, but are
kept in small modules so later repo coupling can mount `hrm-cloud/*.py` without
needing the whole previous grid code path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import continuous_prm_common as C


STAGES = ("c1", "c2", "c3", "c4", "full")


def parse_args(default_stage: str = "full") -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradated continuous PRM experiments C1-C4")
    p.add_argument("--stage", type=str, default=default_stage, choices=STAGES)
    default_out_dirs = {
        "c1": "runs/continuous_prm_c1_baseline",
        "c2": "runs/continuous_prm_c2_avgbase",
        "c3": "runs/continuous_prm_c3_residual_tasklora",
        "c4": "runs/continuous_prm_c4_rbf_mixture",
        "full": "runs/continuous_prm_full_ladder",
    }
    p.add_argument("--out-dir", type=str, default=default_out_dirs.get(default_stage, "runs/continuous_prm_gradated"))
    p.add_argument("--checkpoint-dir", type=str, default="", help="For C4 eval, directory containing C3 avgbase/expert checkpoints; outputs still go to --out-dir")
    p.add_argument("--mode", type=str, default="full", choices=["full", "collect", "train", "eval"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--train-tasks", type=str, default=C.DEFAULT_TRAIN_TASKS)
    p.add_argument("--eval-suites", type=str, default=C.DEFAULT_EVAL_SUITES)
    p.add_argument("--backbones", type=str, default="hrm,onlstm")
    p.add_argument("--lora-alphas", type=str, default="1.0")
    p.add_argument("--train-worlds", type=int, default=100)
    p.add_argument("--nodes-per-world", type=int, default=160)
    p.add_argument("--eval-worlds", type=int, default=40)
    p.add_argument("--roadmap-nodes", type=int, default=256)
    p.add_argument("--roadmap-k", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--base-epochs", type=int, default=10)
    p.add_argument("--expert-epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--expert-lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--budgets", type=str, default="100,200,500,1000")
    p.add_argument("--nearest-obstacles", type=int, default=6)
    p.add_argument("--num-rays", type=int, default=16)
    p.add_argument("--ray-steps", type=int, default=80)
    p.add_argument("--max-norm-residual", type=float, default=4.0)
    p.add_argument("--residual-bound-quantile", type=float, default=0.95)
    p.add_argument("--residual-bound-floor", type=float, default=0.08)
    p.add_argument("--correction-l2", type=float, default=1.0e-3)
    p.add_argument("--train-bias-with-lora", action="store_true")
    p.add_argument("--eval-experts-all-suites", type=int, default=1)
    p.add_argument("--no-oracle-expert", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=int(os.environ.get("TORCH_THREADS", "1")), help="CPU torch thread count; default 1 avoids severe oversubscription in PRM smoke/local runs")

    # Backbone size knobs. These mirror continuous_prm_common.py.
    p.add_argument("--head-hidden", type=int, default=256)
    p.add_argument("--hrm-hidden", type=int, default=192)
    p.add_argument("--hrm-layers", type=int, default=2)
    p.add_argument("--hrm-k-step", type=int, default=2)
    p.add_argument("--hrm-heads", type=int, default=4)
    p.add_argument("--onlstm-hidden", type=int, default=256)
    p.add_argument("--onlstm-layers", type=int, default=2)
    p.add_argument("--onlstm-chunk-size", type=int, default=8)

    # C4 mixture controls.
    p.add_argument("--rbf-sigma", type=float, default=1.0, help="RBF bandwidth in standardized descriptor space")
    p.add_argument("--rbf-topk", type=int, default=0, help="If >0, keep only top-k nearest anchor experts in C4")
    p.add_argument("--rbf-desc-samples", type=int, default=96, help="Samples used to estimate each anchor descriptor centroid")
    p.add_argument("--rbf-include-ood-flag", action="store_true", help="Include the descriptor OOD bit in C4 distances; off by default")
    p.add_argument("--include-expert-matrix", action="store_true", help="For C4, also evaluate every individual expert and oracle expert")

    p.add_argument("--make-figures", action="store_true", help="Generate stage success-rate figures; off by default for lightweight smoke/CI runs")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    # Start from the common smoke values, then make C4 valid on tiny artifacts.
    C.apply_smoke_overrides(args)
    if args.smoke_test:
        args.rbf_desc_samples = min(int(getattr(args, "rbf_desc_samples", 96)), 16)
        args.rbf_sigma = float(getattr(args, "rbf_sigma", 1.0))
        args.rbf_topk = int(getattr(args, "rbf_topk", 0))


def runtime_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    apply_smoke_overrides(args)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
        try:
            torch.set_num_interop_threads(max(1, min(4, int(args.torch_threads))))
        except RuntimeError:
            pass
    if bool(getattr(args, "make_figures", False)):
        os.environ["CONTINUOUS_PRM_MAKE_FIGURES"] = "1"
    C.set_global_seed(int(args.seed))
    out_dir = C.ensure_dir(args.out_dir)
    checkpoint_dir = Path(str(getattr(args, "checkpoint_dir", "") or "")) if str(getattr(args, "checkpoint_dir", "") or "") else out_dir
    asset_dir = checkpoint_dir if checkpoint_dir.exists() else out_dir
    data_dir = C.ensure_dir(out_dir / "datasets")
    C.ensure_dir(out_dir / "logs")
    specs = C.build_anchor_specs()
    train_tasks = C.parse_csv(args.train_tasks)
    eval_suites = C.parse_csv(args.eval_suites)
    backbones = C.parse_csv(args.backbones)
    lora_alphas = [float(x) for x in C.parse_csv(args.lora_alphas)]
    for name in train_tasks + eval_suites:
        if name not in specs:
            raise ValueError(f"unknown task/suite {name!r}; available={sorted(specs)}")
    roadmap_cfg, feature_cfg, train_cfg, eval_cfg = C.build_configs_from_args(args)
    backbone_cfgs = C.build_backbone_configs(args)
    for b in backbones:
        if b not in backbone_cfgs:
            raise ValueError(f"unknown backbone {b!r}; available={sorted(backbone_cfgs)}")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    C.write_json(out_dir / f"config__{args.stage}.json", {
        "stage": args.stage,
        "args": vars(args),
        "asset_dir": str(asset_dir),
        "roadmap_cfg": asdict(roadmap_cfg),
        "feature_cfg": asdict(feature_cfg),
        "train_cfg": asdict(train_cfg),
        "eval_cfg": asdict(eval_cfg),
        "train_tasks": train_tasks,
        "eval_suites": eval_suites,
        "backbones": backbones,
        "lora_alphas": lora_alphas,
    })
    print(f"[{C.now_str()}] stage={args.stage} out_dir={out_dir}")
    print(f"[{C.now_str()}] device={device} train_tasks={train_tasks} eval_suites={eval_suites} backbones={backbones}")
    return {
        "out_dir": out_dir,
        "asset_dir": asset_dir,
        "data_dir": data_dir,
        "specs": specs,
        "train_tasks": train_tasks,
        "eval_suites": eval_suites,
        "backbones": backbones,
        "lora_alphas": lora_alphas,
        "roadmap_cfg": roadmap_cfg,
        "feature_cfg": feature_cfg,
        "train_cfg": train_cfg,
        "eval_cfg": eval_cfg,
        "backbone_cfgs": backbone_cfgs,
        "device": device,
    }


def collect_training_datasets(args: argparse.Namespace, rt: Dict[str, Any]) -> Dict[str, Path]:
    task_npz: Dict[str, Path] = {}
    data_dir: Path = rt["data_dir"]
    specs: Dict[str, C.AnchorSpec] = rt["specs"]
    if args.mode in ("full", "collect", "train"):
        for i, task in enumerate(rt["train_tasks"]):
            task_npz[task] = C.collect_task_dataset(
                specs[task],
                out_dir=data_dir,
                split="train",
                n_worlds=int(args.train_worlds),
                nodes_per_world=int(args.nodes_per_world),
                roadmap_cfg=rt["roadmap_cfg"],
                feature_cfg=rt["feature_cfg"],
                seed=int(args.seed) + 10_000 * (i + 1),
            )
    else:
        for task in rt["train_tasks"]:
            p = data_dir / f"{task}_train.npz"
            if not p.exists():
                raise FileNotFoundError(f"missing dataset {p}; run --mode collect or --mode full first")
            task_npz[task] = p
    return task_npz


def ensure_avgbase_models(args: argparse.Namespace, rt: Dict[str, Any], task_npz: Dict[str, Path]) -> Dict[str, Path]:
    ckpts: Dict[str, Path] = {}
    for backbone in rt["backbones"]:
        cfg = rt["backbone_cfgs"][backbone]
        ckpt = C.model_checkpoint_path(rt["out_dir"], backbone, "avgbase")
        if args.mode in ("full", "train") or not ckpt.exists():
            ckpt = C.train_avgbase(
                cfg,
                task_npz,
                rt["out_dir"],
                rt["feature_cfg"],
                rt["train_cfg"],
                rt["device"],
                seed=int(args.seed) + 100,
            )
        if not ckpt.exists():
            raise FileNotFoundError(f"missing avgbase checkpoint {ckpt}")
        ckpts[backbone] = ckpt
    return ckpts


def ensure_expert_models(args: argparse.Namespace, rt: Dict[str, Any], task_npz: Dict[str, Path], base_ckpts: Dict[str, Path]) -> None:
    for backbone in rt["backbones"]:
        cfg = rt["backbone_cfgs"][backbone]
        base_ckpt = base_ckpts[backbone]
        for alpha in rt["lora_alphas"]:
            for j, task in enumerate(rt["train_tasks"]):
                ckpt = C.model_checkpoint_path(rt["out_dir"], backbone, "expert", task=task, alpha=alpha)
                if args.mode in ("full", "train") or not ckpt.exists():
                    C.train_expert(
                        cfg,
                        task,
                        task_npz[task],
                        base_ckpt,
                        rt["out_dir"],
                        rt["feature_cfg"],
                        rt["train_cfg"],
                        rt["device"],
                        seed=int(args.seed) + 1_000 + 37 * j,
                        alpha=float(alpha),
                    )
                if not ckpt.exists():
                    raise FileNotFoundError(f"missing expert checkpoint {ckpt}")


def _stage_results_paths(out_dir: Path, stage: str) -> Tuple[Path, Path, Path]:
    res_dir = C.ensure_dir(out_dir / "results")
    return (
        res_dir / f"continuous_prm_{stage}_raw.csv",
        res_dir / f"continuous_prm_{stage}_summary.csv",
        res_dir / f"continuous_prm_{stage}_summary.json",
    )


def evaluate_baseline_avgbase(
    rt: Dict[str, Any],
    stage: str,
    include_avgbase: bool,
    seed: int,
) -> Path:
    records: List[Dict[str, Any]] = []
    base_models: Dict[str, C.ContinuousHeuristicModel] = {}
    if include_avgbase:
        for backbone in rt["backbones"]:
            cfg = rt["backbone_cfgs"][backbone]
            ckpt = C.model_checkpoint_path(rt["out_dir"], backbone, "avgbase")
            base_models[backbone] = C.load_base_model(cfg, rt["feature_cfg"], rt["train_cfg"], ckpt, rt["device"])

    for suite in rt["eval_suites"]:
        spec = rt["specs"][suite]
        print(f"[{C.now_str()}] {stage}: evaluating suite {suite}", flush=True)
        valid_worlds = 0
        attempts = 0
        while valid_worlds < rt["eval_cfg"].eval_worlds and attempts < rt["eval_cfg"].eval_worlds * rt["eval_cfg"].max_eval_world_retries:
            attempts += 1
            w_seed = seed + 1_000_003 * (rt["eval_suites"].index(suite) + 1) + attempts * 7919
            bundle = C.make_eval_world_bundle(spec, rt["roadmap_cfg"], rt["feature_cfg"], w_seed, retries=1)
            if bundle is None:
                continue
            world = bundle.world
            roadmap = bundle.roadmap
            features = bundle.features
            euclid_h = bundle.euclidean_h
            oracle_cost = bundle.oracle_cost
            valid_worlds += 1

            method_heuristics: List[Tuple[str, str, str, float, np.ndarray, Dict[str, float]]] = []
            zero_h = np.zeros_like(euclid_h, dtype=np.float64)
            method_heuristics.append(("dijkstra", "", "", 0.0, zero_h, {"delta_mean": 0.0, "correction_abs_mean": 0.0}))
            method_heuristics.append(("euclidean", "", "", 0.0, euclid_h, {"delta_mean": 0.0, "correction_abs_mean": 0.0}))
            if include_avgbase:
                for backbone, model in base_models.items():
                    base_delta_norm = C.predict_norm_residuals(model, features, rt["device"], batch_size=2048)
                    base_delta_norm = np.clip(base_delta_norm, 0.0, rt["train_cfg"].max_norm_residual)
                    base_h = euclid_h + base_delta_norm * world.side_len
                    method_heuristics.append(("avgbase", backbone, "", 0.0, base_h, {"delta_mean": float(np.mean(base_delta_norm)), "correction_abs_mean": 0.0}))

            for method, backbone, expert_task, alpha, h, diag in method_heuristics:
                nonfinite = int(not np.isfinite(h).all())
                if nonfinite:
                    h = np.nan_to_num(h, nan=rt["train_cfg"].max_norm_residual * world.side_len, posinf=rt["train_cfg"].max_norm_residual * world.side_len, neginf=0.0)
                for budget in rt["eval_cfg"].budgets:
                    res = C.astar_search(roadmap.adj, h, budget=int(budget))
                    found = bool(res["found"])
                    cost = float(res["cost"])
                    ratio = float(cost / oracle_cost) if found and oracle_cost > C.EPS else float("nan")
                    records.append({
                        "stage": stage,
                        "suite": suite,
                        "world_index": valid_worlds - 1,
                        "method": method,
                        "backbone": backbone,
                        "expert_task": expert_task,
                        "alpha": alpha,
                        "budget": int(budget),
                        "found": int(found),
                        "cost": cost,
                        "cost_ratio": ratio,
                        "oracle_cost": oracle_cost,
                        "expansions": int(res["expansions"]),
                        "closed": int(res["closed"]),
                        "roadmap_nodes": int(roadmap.points.shape[0]),
                        "roadmap_edges": int(sum(len(a) for a in roadmap.adj) // 2),
                        "obstacle_count": int(len(world.obstacles)),
                        "side_len": float(world.side_len),
                        "heuristic_max": float(np.max(h)) if len(h) else float("nan"),
                        "heuristic_min": float(np.min(h)) if len(h) else float("nan"),
                        "nonfinite_heuristic": nonfinite,
                        **diag,
                    })
        print(f"[{C.now_str()}] {stage}: suite {suite}: valid_worlds={valid_worlds}/{rt['eval_cfg'].eval_worlds}", flush=True)

    raw_path, summary_path, json_path = _stage_results_paths(rt["out_dir"], stage)
    C.write_csv(raw_path, records)
    summary = C.aggregate_eval_records(records)
    C.write_csv(summary_path, summary)
    C.write_json(json_path, {"rows": summary})
    make_stage_figures(summary_path, rt["out_dir"], stage)
    print(f"[{C.now_str()}] {stage}: wrote {summary_path}")
    return summary_path


def estimate_anchor_descriptors(
    specs: Dict[str, C.AnchorSpec],
    train_tasks: Sequence[str],
    seed: int,
    samples: int,
    include_ood_flag: bool,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    refs: Dict[str, np.ndarray] = {}
    rng = random.Random(seed)
    for task in train_tasks:
        spec = specs[task]
        descs: List[np.ndarray] = []
        for _ in range(max(1, int(samples))):
            obs = C.generate_obstacles(spec, rng)
            d = C.task_descriptor(spec, obs).astype(np.float64)
            if not include_ood_flag:
                d = d[:-1]
            descs.append(d)
        refs[task] = np.mean(np.stack(descs, axis=0), axis=0)
    mat = np.stack([refs[t] for t in train_tasks], axis=0)
    scale = np.std(mat, axis=0)
    scale = np.where(scale < 1.0e-4, 1.0, scale)
    return refs, mat, scale


def rbf_weights(
    z: np.ndarray,
    train_tasks: Sequence[str],
    ref_mat: np.ndarray,
    scale: np.ndarray,
    sigma: float,
    topk: int = 0,
) -> np.ndarray:
    z = z.astype(np.float64)
    if z.shape[0] != ref_mat.shape[1]:
        z = z[: ref_mat.shape[1]]
    dz = (ref_mat - z[None, :]) / scale[None, :]
    dist2 = np.sum(dz * dz, axis=1)
    if int(topk) > 0 and int(topk) < len(train_tasks):
        keep = np.argsort(dist2)[: int(topk)]
        mask = np.ones(len(train_tasks), dtype=bool)
        mask[keep] = False
        dist2[mask] = np.inf
    sigma = max(float(sigma), 1.0e-6)
    w = np.exp(-0.5 * dist2 / (sigma * sigma))
    if not np.isfinite(w).all() or float(np.sum(w)) <= 1.0e-12:
        w = np.ones(len(train_tasks), dtype=np.float64)
    w = w / float(np.sum(w))
    return w.astype(np.float64)


def evaluate_rbf_mixture(args: argparse.Namespace, rt: Dict[str, Any], seed: int) -> Path:
    records: List[Dict[str, Any]] = []
    refs, ref_mat, scale = estimate_anchor_descriptors(
        rt["specs"],
        rt["train_tasks"],
        seed=int(args.seed) + 33_333,
        samples=int(args.rbf_desc_samples),
        include_ood_flag=bool(args.rbf_include_ood_flag),
    )
    C.write_json(rt["out_dir"] / "results" / "c4_anchor_descriptor_refs.json", {
        "train_tasks": list(rt["train_tasks"]),
        "refs": {k: [float(x) for x in v] for k, v in refs.items()},
        "scale": [float(x) for x in scale],
        "rbf_sigma": float(args.rbf_sigma),
        "rbf_topk": int(args.rbf_topk),
        "include_ood_flag": bool(args.rbf_include_ood_flag),
    })

    base_models: Dict[str, C.ContinuousHeuristicModel] = {}
    expert_cache: Dict[Tuple[str, str, float], Tuple[C.ContinuousHeuristicModel, float]] = {}
    for backbone in rt["backbones"]:
        cfg = rt["backbone_cfgs"][backbone]
        base_ckpt = C.model_checkpoint_path(rt.get("asset_dir", rt["out_dir"]), backbone, "avgbase")
        base_models[backbone] = C.load_base_model(cfg, rt["feature_cfg"], rt["train_cfg"], base_ckpt, rt["device"])

    def get_expert(backbone: str, task: str, alpha: float) -> Tuple[C.ContinuousHeuristicModel, float]:
        key = (backbone, task, float(alpha))
        if key in expert_cache:
            return expert_cache[key]
        cfg = rt["backbone_cfgs"][backbone]
        base_ckpt = C.model_checkpoint_path(rt.get("asset_dir", rt["out_dir"]), backbone, "avgbase")
        expert_ckpt = C.model_checkpoint_path(rt.get("asset_dir", rt["out_dir"]), backbone, "expert", task=task, alpha=alpha)
        expert_model, bound = C.load_expert_model(cfg, rt["feature_cfg"], rt["train_cfg"], base_ckpt, expert_ckpt, rt["device"])
        expert_cache[key] = (expert_model, float(bound))
        return expert_cache[key]

    for suite in rt["eval_suites"]:
        spec = rt["specs"][suite]
        print(f"[{C.now_str()}] c4: evaluating suite {suite}", flush=True)
        valid_worlds = 0
        attempts = 0
        while valid_worlds < rt["eval_cfg"].eval_worlds and attempts < rt["eval_cfg"].eval_worlds * rt["eval_cfg"].max_eval_world_retries:
            attempts += 1
            w_seed = seed + 1_000_003 * (rt["eval_suites"].index(suite) + 1) + attempts * 7919
            bundle = C.make_eval_world_bundle(spec, rt["roadmap_cfg"], rt["feature_cfg"], w_seed, retries=1)
            if bundle is None:
                continue
            valid_worlds += 1
            world = bundle.world
            roadmap = bundle.roadmap
            features = bundle.features
            euclid_h = bundle.euclidean_h
            oracle_cost = bundle.oracle_cost
            z = world.descriptor.astype(np.float64)
            if not bool(args.rbf_include_ood_flag):
                z = z[:-1]
            weights = rbf_weights(z, rt["train_tasks"], ref_mat, scale, sigma=float(args.rbf_sigma), topk=int(args.rbf_topk))
            weights_json = json.dumps({task: float(w) for task, w in zip(rt["train_tasks"], weights)}, sort_keys=True)
            nearest_weights = rbf_weights(z, rt["train_tasks"], ref_mat, scale, sigma=float(args.rbf_sigma), topk=1)
            nearest_idx = int(np.argmax(nearest_weights))
            nearest_task = str(rt["train_tasks"][nearest_idx])
            nearest_weights_json = json.dumps({task: float(w) for task, w in zip(rt["train_tasks"], nearest_weights)}, sort_keys=True)

            method_heuristics: List[Tuple[str, str, str, float, np.ndarray, Dict[str, Any]]] = []
            method_heuristics.append(("euclidean", "", "", 0.0, euclid_h, {"delta_mean": 0.0, "correction_abs_mean": 0.0, "mixture_weights": ""}))

            for backbone in rt["backbones"]:
                base_model = base_models[backbone]
                bpred = C.predict_norm_residuals(base_model, features, rt["device"], batch_size=2048)
                bpred = np.clip(bpred, 0.0, rt["train_cfg"].max_norm_residual)
                base_h = euclid_h + bpred * world.side_len
                method_heuristics.append(("avgbase", backbone, "", 0.0, base_h, {"delta_mean": float(np.mean(bpred)), "correction_abs_mean": 0.0, "mixture_weights": ""}))

                for alpha in rt["lora_alphas"]:
                    corrections: List[np.ndarray] = []
                    individual_rows: List[Tuple[str, np.ndarray, float, float]] = []
                    for task in rt["train_tasks"]:
                        expert_model, bound = get_expert(backbone, task, float(alpha))
                        epred = C.predict_norm_residuals(expert_model, features, rt["device"], batch_size=2048)
                        corr = bound * np.tanh((epred - bpred) / max(float(bound), C.EPS))
                        corrections.append(corr)
                        final_delta = np.clip(bpred + corr, 0.0, rt["train_cfg"].max_norm_residual)
                        h = euclid_h + final_delta * world.side_len
                        individual_rows.append((task, h, float(np.mean(final_delta)), float(np.mean(np.abs(corr)))))
                    corr_mat = np.stack(corrections, axis=0)

                    nearest_corr = corr_mat[nearest_idx]
                    nearest_delta = np.clip(bpred + nearest_corr, 0.0, rt["train_cfg"].max_norm_residual)
                    h_nearest = euclid_h + nearest_delta * world.side_len
                    method_heuristics.append((
                        "nearest_tasklora",
                        backbone,
                        nearest_task,
                        float(alpha),
                        h_nearest,
                        {
                            "delta_mean": float(np.mean(nearest_delta)),
                            "correction_abs_mean": float(np.mean(np.abs(nearest_corr))),
                            "mixture_weights": nearest_weights_json,
                            "rbf_sigma": float(args.rbf_sigma),
                            "rbf_topk": 1,
                        },
                    ))

                    mix_corr = np.sum(weights[:, None] * corr_mat, axis=0)
                    final_delta = np.clip(bpred + mix_corr, 0.0, rt["train_cfg"].max_norm_residual)
                    h_mix = euclid_h + final_delta * world.side_len
                    method_heuristics.append((
                        "rbf_mix_tasklora",
                        backbone,
                        "<rbf_mix>",
                        float(alpha),
                        h_mix,
                        {
                            "delta_mean": float(np.mean(final_delta)),
                            "correction_abs_mean": float(np.mean(np.abs(mix_corr))),
                            "mixture_weights": weights_json,
                            "rbf_sigma": float(args.rbf_sigma),
                            "rbf_topk": int(args.rbf_topk),
                        },
                    ))
                    if bool(args.include_expert_matrix):
                        for task, h_ind, dmean, cmean in individual_rows:
                            method_heuristics.append((
                                "tasklora",
                                backbone,
                                task,
                                float(alpha),
                                h_ind,
                                {"delta_mean": dmean, "correction_abs_mean": cmean, "mixture_weights": ""},
                            ))

            world_rows_start = len(records)
            for method, backbone, expert_task, alpha, h, diag in method_heuristics:
                nonfinite = int(not np.isfinite(h).all())
                if nonfinite:
                    h = np.nan_to_num(h, nan=rt["train_cfg"].max_norm_residual * world.side_len, posinf=rt["train_cfg"].max_norm_residual * world.side_len, neginf=0.0)
                for budget in rt["eval_cfg"].budgets:
                    res = C.astar_search(roadmap.adj, h, budget=int(budget))
                    found = bool(res["found"])
                    cost = float(res["cost"])
                    ratio = float(cost / oracle_cost) if found and oracle_cost > C.EPS else float("nan")
                    records.append({
                        "stage": "c4",
                        "suite": suite,
                        "world_index": valid_worlds - 1,
                        "method": method,
                        "backbone": backbone,
                        "expert_task": expert_task,
                        "alpha": alpha,
                        "budget": int(budget),
                        "found": int(found),
                        "cost": cost,
                        "cost_ratio": ratio,
                        "oracle_cost": oracle_cost,
                        "expansions": int(res["expansions"]),
                        "closed": int(res["closed"]),
                        "roadmap_nodes": int(roadmap.points.shape[0]),
                        "roadmap_edges": int(sum(len(a) for a in roadmap.adj) // 2),
                        "obstacle_count": int(len(world.obstacles)),
                        "side_len": float(world.side_len),
                        "heuristic_max": float(np.max(h)) if len(h) else float("nan"),
                        "heuristic_min": float(np.min(h)) if len(h) else float("nan"),
                        "nonfinite_heuristic": nonfinite,
                        **diag,
                    })

            if bool(args.include_expert_matrix) and rt["eval_cfg"].eval_oracle_expert:
                world_rows = records[world_rows_start:]
                for backbone in rt["backbones"]:
                    for alpha in rt["lora_alphas"]:
                        candidates_all = [r for r in world_rows if r["method"] == "tasklora" and r["backbone"] == backbone and float(r["alpha"]) == float(alpha)]
                        for budget in rt["eval_cfg"].budgets:
                            candidates = [r for r in candidates_all if int(r["budget"]) == int(budget)]
                            if not candidates:
                                continue
                            succ = [r for r in candidates if int(r["found"]) == 1]
                            if succ:
                                best = min(succ, key=lambda r: (float(r["expansions"]), float(r.get("cost_ratio", 999.0))))
                                records.append({**best, "method": "oracle_tasklora", "expert_task": str(best["expert_task"])})
                            else:
                                rep = min(candidates, key=lambda r: float(r["expansions"]))
                                records.append({**rep, "method": "oracle_tasklora", "expert_task": "<none>", "found": 0, "cost": float("nan"), "cost_ratio": float("nan")})
        print(f"[{C.now_str()}] c4: suite {suite}: valid_worlds={valid_worlds}/{rt['eval_cfg'].eval_worlds}", flush=True)

    raw_path, summary_path, json_path = _stage_results_paths(rt["out_dir"], "c4")
    C.write_csv(raw_path, records)
    summary = C.aggregate_eval_records(records)
    C.write_csv(summary_path, summary)
    C.write_json(json_path, {"rows": summary})
    make_stage_figures(summary_path, rt["out_dir"], "c4")
    print(f"[{C.now_str()}] c4: wrote {summary_path}")
    return summary_path


def make_stage_figures(summary_csv: Path, out_dir: Path, stage: str) -> None:
    if str(os.environ.get("CONTINUOUS_PRM_MAKE_FIGURES", "0")).lower() not in ("1", "true", "yes", "on"):
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    rows: List[Dict[str, str]] = []
    with open(summary_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return
    fig_dir = C.ensure_dir(out_dir / "figures" / stage)
    preferred_methods = {"euclidean", "avgbase", "nearest_tasklora", "rbf_mix_tasklora", "oracle_tasklora"}
    for suite in sorted(set(r["suite"] for r in rows)):
        plot_rows = [r for r in rows if r["suite"] == suite and r["method"] in preferred_methods]
        if not plot_rows:
            continue
        keys = sorted(set((r["method"], r.get("backbone", ""), r.get("alpha", "0")) for r in plot_rows))
        plt.figure(figsize=(8.5, 5.0))
        for method, backbone, alpha in keys:
            series = [r for r in plot_rows if r["method"] == method and r.get("backbone", "") == backbone and r.get("alpha", "0") == alpha]
            series.sort(key=lambda r: int(float(r["budget"])))
            xs = [int(float(r["budget"])) for r in series]
            ys = [float(r["success_rate"]) for r in series]
            label = method if not backbone else f"{method}:{backbone}"
            if method in ("nearest_tasklora", "rbf_mix_tasklora", "oracle_tasklora"):
                try:
                    label += f":a={float(alpha):g}"
                except Exception:
                    pass
            plt.plot(xs, ys, marker="o", label=label)
        plt.xlabel("A* expansion budget")
        plt.ylabel("Success rate")
        plt.title(f"{stage.upper()} continuous PRM success: {suite}")
        plt.ylim(-0.03, 1.03)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / f"success__{C.sanitize_name(suite)}.png", dpi=160)
        plt.close()


def run_stage(args: argparse.Namespace) -> Optional[Path]:
    rt = runtime_from_args(args)
    stage = args.stage

    if stage == "full":
        # Full means execute the ladder in order using shared artifacts.
        args_c1 = argparse.Namespace(**vars(args)); args_c1.stage = "c1"
        evaluate_baseline_avgbase(rt, "c1", include_avgbase=False, seed=int(args.seed) + 200_000)
        task_npz = collect_training_datasets(args, rt)
        base_ckpts = ensure_avgbase_models(args, rt, task_npz)
        evaluate_baseline_avgbase(rt, "c2", include_avgbase=True, seed=int(args.seed) + 210_000)
        ensure_expert_models(args, rt, task_npz, base_ckpts)
        if args.mode in ("full", "eval"):
            C.evaluate_all(
                out_dir=rt["out_dir"],
                specs=rt["specs"],
                train_tasks=rt["train_tasks"],
                eval_suites=rt["eval_suites"],
                backbone_cfgs=rt["backbone_cfgs"],
                backbones=rt["backbones"],
                feature_cfg=rt["feature_cfg"],
                roadmap_cfg=rt["roadmap_cfg"],
                train_cfg=rt["train_cfg"],
                eval_cfg=rt["eval_cfg"],
                device=rt["device"],
                seed=int(args.seed) + 220_000,
                lora_alphas=rt["lora_alphas"],
            )
        return evaluate_rbf_mixture(args, rt, seed=int(args.seed) + 230_000)

    if stage == "c1":
        if args.mode in ("train", "collect"):
            print(f"[{C.now_str()}] c1 has no train/collect phase; running baseline eval instead.")
        return evaluate_baseline_avgbase(rt, "c1", include_avgbase=False, seed=int(args.seed) + 200_000)

    if stage == "c2":
        if args.mode in ("full", "collect", "train"):
            task_npz = collect_training_datasets(args, rt)
            if args.mode in ("full", "train"):
                ensure_avgbase_models(args, rt, task_npz)
        if args.mode in ("full", "eval"):
            return evaluate_baseline_avgbase(rt, "c2", include_avgbase=True, seed=int(args.seed) + 210_000)
        return None

    if stage == "c3":
        if args.mode in ("full", "collect", "train"):
            task_npz = collect_training_datasets(args, rt)
            base_ckpts = ensure_avgbase_models(args, rt, task_npz)
            ensure_expert_models(args, rt, task_npz, base_ckpts)
        if args.mode in ("full", "eval"):
            return C.evaluate_all(
                out_dir=rt["out_dir"],
                specs=rt["specs"],
                train_tasks=rt["train_tasks"],
                eval_suites=rt["eval_suites"],
                backbone_cfgs=rt["backbone_cfgs"],
                backbones=rt["backbones"],
                feature_cfg=rt["feature_cfg"],
                roadmap_cfg=rt["roadmap_cfg"],
                train_cfg=rt["train_cfg"],
                eval_cfg=rt["eval_cfg"],
                device=rt["device"],
                seed=int(args.seed) + 220_000,
                lora_alphas=rt["lora_alphas"],
            )
        return None

    if stage == "c4":
        if args.mode in ("full", "collect", "train"):
            task_npz = collect_training_datasets(args, rt)
            base_ckpts = ensure_avgbase_models(args, rt, task_npz)
            ensure_expert_models(args, rt, task_npz, base_ckpts)
        if args.mode in ("full", "eval"):
            return evaluate_rbf_mixture(args, rt, seed=int(args.seed) + 230_000)
        return None

    raise ValueError(f"unknown stage {stage}")


def main(default_stage: str = "full") -> None:
    args = parse_args(default_stage=default_stage)
    run_stage(args)


if __name__ == "__main__":
    main()
