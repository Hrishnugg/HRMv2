#!/usr/bin/env python3
"""
Modal orchestration for continuous PRM C3/C4.

This file keeps the local staged runner intact and adds coarse-grained Modal
parallelism around the expensive loops:

- dataset shards run independently per training task;
- avgbase models train per backbone;
- residual LoRA experts train per backbone/task/alpha;
- C3/C4 eval shards run per suite/world range and are merged afterward.

Examples:

    modal run continuous_prm_modal.py::run_c3

    modal run continuous_prm_modal.py::run_c4 --checkpoint-run-name continuous_prm_c3_modal

    modal run continuous_prm_modal.py::run_c3 --run-name continuous_prm_c3_modal_big \
      --train-worlds 300 --eval-worlds 80 --roadmap-nodes 384 --roadmap-k 18 \
      --lora-alphas 0.5,1.0,1.5,2.0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import modal


APP_NAME = "continuous-prm-heuristic-learning"
VOLUME_NAME = "continuous-prm-heuristic-learning-vol"
VOLUME_ROOT = "/vol"
REMOTE_CODE_DIR = "/app/continuous_prm"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    )
    .add_local_dir(
        Path(__file__).parent,
        remote_path=REMOTE_CODE_DIR,
        copy=True,
        ignore=["__pycache__", "**/__pycache__", "*.pyc", "runs", "runs/**"],
    )
)


def _import_common(params: Optional[Dict[str, Any]] = None):
    import sys

    sys.path.insert(0, REMOTE_CODE_DIR)
    import continuous_prm_common as C  # type: ignore
    if params and str(params.get("profile", "")) == "c5_hard":
        import continuous_prm_c5_hard_obstacle_encoder as C5  # type: ignore

        C5.SECTOR_TOKENS = int(params.get("sector_tokens", 16))
        C5.SOFT_RESIDUAL_CAP = bool(params.get("soft_residual_cap", False))
        C5.install_runtime_extensions()

    return C


def _import_c6():
    import sys

    sys.path.insert(0, REMOTE_CODE_DIR)
    import continuous_prm_c6_heatmap_value_field as C6  # type: ignore

    return C6


def _csv(spec: str) -> List[str]:
    return [x.strip() for x in str(spec).split(",") if x.strip()]


def _float_csv(spec: str) -> List[float]:
    return [float(x) for x in _csv(spec)]


def _run_dir(run_name: str) -> Path:
    return Path(VOLUME_ROOT) / "runs" / run_name


def _base_params(
    *,
    train_worlds: int,
    nodes_per_world: int,
    eval_worlds: int,
    roadmap_nodes: int,
    roadmap_k: int,
    base_epochs: int,
    expert_epochs: int,
    backbones: str,
    lora_alphas: str,
    budgets: str,
    seed: int,
    train_tasks: str,
    eval_suites: str,
    batch_size: int,
    lr: float,
    expert_lr: float,
    weight_decay: float,
    grad_clip: float,
    max_norm_residual: float,
    residual_bound_quantile: float,
    residual_bound_floor: float,
    correction_l2: float,
    roadmap_shard_worlds: int,
    eval_shard_worlds: int,
    rbf_sigma: float,
    rbf_topk: int,
    rbf_desc_samples: int,
    include_expert_matrix: bool,
    cpu: bool,
    torch_threads: int,
    profile: str = "default",
    nearest_obstacles: int = 6,
    num_rays: int = 16,
    ray_steps: int = 80,
    sector_tokens: int = 0,
    soft_residual_cap: bool = False,
    head_hidden: int = 256,
    hrm_hidden: int = 192,
    hrm_layers: int = 2,
    hrm_k_step: int = 2,
    hrm_heads: int = 4,
    onlstm_hidden: int = 256,
    onlstm_layers: int = 2,
    onlstm_chunk_size: int = 8,
) -> Dict[str, Any]:
    return {
        "profile": str(profile),
        "train_worlds": int(train_worlds),
        "nodes_per_world": int(nodes_per_world),
        "eval_worlds": int(eval_worlds),
        "roadmap_nodes": int(roadmap_nodes),
        "roadmap_k": int(roadmap_k),
        "base_epochs": int(base_epochs),
        "expert_epochs": int(expert_epochs),
        "backbones": backbones,
        "lora_alphas": lora_alphas,
        "budgets": budgets,
        "seed": int(seed),
        "train_tasks": train_tasks,
        "eval_suites": eval_suites,
        "batch_size": int(batch_size),
        "lr": float(lr),
        "expert_lr": float(expert_lr),
        "weight_decay": float(weight_decay),
        "grad_clip": float(grad_clip),
        "max_norm_residual": float(max_norm_residual),
        "residual_bound_quantile": float(residual_bound_quantile),
        "residual_bound_floor": float(residual_bound_floor),
        "correction_l2": float(correction_l2),
        "roadmap_shard_worlds": int(roadmap_shard_worlds),
        "eval_shard_worlds": int(eval_shard_worlds),
        "rbf_sigma": float(rbf_sigma),
        "rbf_topk": int(rbf_topk),
        "rbf_desc_samples": int(rbf_desc_samples),
        "include_expert_matrix": bool(include_expert_matrix),
        "cpu": bool(cpu),
        "torch_threads": int(torch_threads),
        "nearest_obstacles": int(nearest_obstacles),
        "num_rays": int(num_rays),
        "ray_steps": int(ray_steps),
        "sector_tokens": int(sector_tokens),
        "soft_residual_cap": bool(soft_residual_cap),
        "head_hidden": int(head_hidden),
        "hrm_hidden": int(hrm_hidden),
        "hrm_layers": int(hrm_layers),
        "hrm_k_step": int(hrm_k_step),
        "hrm_heads": int(hrm_heads),
        "onlstm_hidden": int(onlstm_hidden),
        "onlstm_layers": int(onlstm_layers),
        "onlstm_chunk_size": int(onlstm_chunk_size),
    }


def _args(params: Dict[str, Any], *, stage: str, out_dir: Path, checkpoint_dir: Optional[Path] = None) -> argparse.Namespace:
    C = _import_common(params)
    return argparse.Namespace(
        stage=stage,
        out_dir=str(out_dir),
        checkpoint_dir=str(checkpoint_dir or ""),
        mode="full",
        seed=int(params["seed"]),
        train_tasks=params["train_tasks"],
        eval_suites=params["eval_suites"],
        backbones=params["backbones"],
        lora_alphas=params["lora_alphas"],
        train_worlds=int(params["train_worlds"]),
        nodes_per_world=int(params["nodes_per_world"]),
        eval_worlds=int(params["eval_worlds"]),
        roadmap_nodes=int(params["roadmap_nodes"]),
        roadmap_k=int(params["roadmap_k"]),
        batch_size=int(params["batch_size"]),
        base_epochs=int(params["base_epochs"]),
        expert_epochs=int(params["expert_epochs"]),
        lr=float(params["lr"]),
        expert_lr=float(params["expert_lr"]),
        weight_decay=float(params["weight_decay"]),
        grad_clip=float(params["grad_clip"]),
        budgets=params["budgets"],
        nearest_obstacles=int(params.get("nearest_obstacles", 6)),
        num_rays=int(params.get("num_rays", 16)),
        ray_steps=int(params.get("ray_steps", 80)),
        max_norm_residual=float(params["max_norm_residual"]),
        residual_bound_quantile=float(params["residual_bound_quantile"]),
        residual_bound_floor=float(params["residual_bound_floor"]),
        correction_l2=float(params["correction_l2"]),
        train_bias_with_lora=False,
        eval_experts_all_suites=1,
        no_oracle_expert=False,
        cpu=bool(params["cpu"]),
        num_workers=0,
        torch_threads=int(params["torch_threads"]),
        head_hidden=int(params.get("head_hidden", 256)),
        hrm_hidden=int(params.get("hrm_hidden", 192)),
        hrm_layers=int(params.get("hrm_layers", 2)),
        hrm_k_step=int(params.get("hrm_k_step", 2)),
        hrm_heads=int(params.get("hrm_heads", 4)),
        onlstm_hidden=int(params.get("onlstm_hidden", 256)),
        onlstm_layers=int(params.get("onlstm_layers", 2)),
        onlstm_chunk_size=int(params.get("onlstm_chunk_size", 8)),
        rbf_sigma=float(params["rbf_sigma"]),
        rbf_topk=int(params["rbf_topk"]),
        rbf_desc_samples=int(params["rbf_desc_samples"]),
        rbf_include_ood_flag=False,
        include_expert_matrix=bool(params["include_expert_matrix"]),
        make_figures=False,
        smoke_test=False,
        DEFAULT_TRAIN_TASKS=C.DEFAULT_TRAIN_TASKS,
        DEFAULT_EVAL_SUITES=C.DEFAULT_EVAL_SUITES,
    )


def _runtime(params: Dict[str, Any], *, stage: str, out_dir: Path, checkpoint_dir: Optional[Path] = None) -> Dict[str, Any]:
    import torch

    C = _import_common(params)
    args = _args(params, stage=stage, out_dir=out_dir, checkpoint_dir=checkpoint_dir)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    C.set_global_seed(int(args.seed))
    specs = C.build_anchor_specs()
    roadmap_cfg, feature_cfg, train_cfg, eval_cfg = C.build_configs_from_args(args)
    backbone_cfgs = C.build_backbone_configs(args)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    return {
        "args": args,
        "specs": specs,
        "train_tasks": _csv(args.train_tasks),
        "eval_suites": _csv(args.eval_suites),
        "backbones": _csv(args.backbones),
        "lora_alphas": _float_csv(args.lora_alphas),
        "roadmap_cfg": roadmap_cfg,
        "feature_cfg": feature_cfg,
        "train_cfg": train_cfg,
        "eval_cfg": eval_cfg,
        "backbone_cfgs": backbone_cfgs,
        "device": device,
        "out_dir": out_dir,
        "asset_dir": checkpoint_dir or out_dir,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    C = _import_common()
    C.write_csv(path, rows)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _merge_npz(shard_paths: Sequence[Path], out_npz: Path, meta_paths: Sequence[Path], out_meta: Path, task: str, split: str) -> None:
    import numpy as np

    xs: List[Any] = []
    ys: List[Any] = []
    eus: List[Any] = []
    sides: List[Any] = []
    worlds: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []
    for shard_npz, shard_meta in zip(shard_paths, meta_paths):
        arr = np.load(shard_npz)
        xs.append(arr["x"])
        ys.append(arr["y"])
        eus.append(arr["euclid"])
        sides.append(arr["side"])
        meta = json.loads(shard_meta.read_text(encoding="utf-8"))
        metas.append(meta)
        worlds.extend(meta.get("worlds", []))
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    eu = np.concatenate(eus, axis=0).astype(np.float32)
    side = np.concatenate(sides, axis=0).astype(np.float32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, x=x, y=y, euclid=eu, side=side)
    meta0 = metas[0] if metas else {}
    _write_json(out_meta, {
        "task": task,
        "split": split,
        "n_worlds_requested": int(sum(int(m.get("n_worlds_requested", 0)) for m in metas)),
        "n_worlds_collected": int(sum(int(m.get("n_worlds_collected", 0)) for m in metas)),
        "n_samples": int(x.shape[0]),
        "seq_len": int(meta0.get("seq_len", 0)),
        "token_dim": int(meta0.get("token_dim", 0)),
        "target_residual_norm_mean": float(np.mean(y)) if len(y) else float("nan"),
        "target_residual_norm_p95": float(np.quantile(y, 0.95)) if len(y) else float("nan"),
        "worlds": worlds,
        "shards": [str(p) for p in shard_paths],
        "spec": meta0.get("spec", {}),
    })


def _collect_jobs(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    shard_worlds = max(1, int(params["roadmap_shard_worlds"]))
    for task_idx, task in enumerate(_csv(params["train_tasks"])):
        n = int(params["train_worlds"])
        n_shards = int(math.ceil(n / shard_worlds))
        for shard_idx in range(n_shards):
            start = shard_idx * shard_worlds
            count = min(shard_worlds, n - start)
            jobs.append({"task": task, "task_idx": task_idx, "shard_idx": shard_idx, "worlds": count})
    return jobs


def _eval_jobs(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    shard_worlds = max(1, int(params["eval_shard_worlds"]))
    for suite_idx, suite in enumerate(_csv(params["eval_suites"])):
        n = int(params["eval_worlds"])
        n_shards = int(math.ceil(n / shard_worlds))
        for shard_idx in range(n_shards):
            start = shard_idx * shard_worlds
            count = min(shard_worlds, n - start)
            jobs.append({"suite": suite, "suite_idx": suite_idx, "shard_idx": shard_idx, "world_start": start, "worlds": count})
    return jobs


def _c6_params(
    *,
    train_worlds: int,
    eval_worlds: int,
    grid_size: int,
    roadmap_nodes: int,
    roadmap_k: int,
    epochs: int,
    models: str,
    train_tasks: str,
    eval_suites: str,
    budgets: str,
    seed: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    loss_rank_weight: float,
    loss_path_weight: float,
    loss_consistency_weight: float,
    rank_pairs: int,
    max_world_retries: int,
    sector_tokens: int,
    dataset_shard_worlds: int,
    eval_shard_worlds: int,
    cpu: bool,
    torch_threads: int,
    make_figures: bool,
) -> Dict[str, Any]:
    return {
        "grid_size": int(grid_size),
        "train_worlds": int(train_worlds),
        "eval_worlds": int(eval_worlds),
        "roadmap_nodes": int(roadmap_nodes),
        "roadmap_k": int(roadmap_k),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "grad_clip": float(grad_clip),
        "loss_rank_weight": float(loss_rank_weight),
        "loss_path_weight": float(loss_path_weight),
        "loss_consistency_weight": float(loss_consistency_weight),
        "rank_pairs": int(rank_pairs),
        "train_tasks": train_tasks,
        "eval_suites": eval_suites,
        "budgets": budgets,
        "models": models,
        "seed": int(seed),
        "max_world_retries": int(max_world_retries),
        "sector_tokens": int(sector_tokens),
        "dataset_shard_worlds": int(dataset_shard_worlds),
        "eval_shard_worlds": int(eval_shard_worlds),
        "torch_threads": int(torch_threads),
        "cpu": bool(cpu),
        "make_figures": bool(make_figures),
    }


def _c6_collect_jobs(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    shard_worlds = max(1, int(params["dataset_shard_worlds"]))
    for task_idx, task in enumerate(_csv(params["train_tasks"])):
        n = int(params["train_worlds"])
        n_shards = int(math.ceil(n / shard_worlds))
        for shard_idx in range(n_shards):
            start = shard_idx * shard_worlds
            count = min(shard_worlds, n - start)
            jobs.append({"task": task, "task_idx": task_idx, "shard_idx": shard_idx, "worlds": count})
    return jobs


def _c6_eval_jobs(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    shard_worlds = max(1, int(params["eval_shard_worlds"]))
    for suite_idx, suite in enumerate(_csv(params["eval_suites"])):
        n = int(params["eval_worlds"])
        n_shards = int(math.ceil(n / shard_worlds))
        for shard_idx in range(n_shards):
            start = shard_idx * shard_worlds
            count = min(shard_worlds, n - start)
            jobs.append({"suite": suite, "suite_idx": suite_idx, "shard_idx": shard_idx, "world_start": start, "worlds": count})
    return jobs


@app.function(image=image, timeout=60 * 60 * 6, volumes={VOLUME_ROOT: volume})
def collect_dataset_shard(run_name: str, params: Dict[str, Any], job: Dict[str, Any]) -> str:
    C = _import_common(params)
    volume.reload()
    run_dir = _run_dir(run_name)
    rt = _runtime(params, stage="c3", out_dir=run_dir)
    task = str(job["task"])
    shard_idx = int(job["shard_idx"])
    shard_dir = run_dir / "datasets" / "_shards" / task / f"shard_{shard_idx:04d}"
    spec = rt["specs"][task]
    seed = int(params["seed"]) + 10_000 * (int(job["task_idx"]) + 1) + 1_000_000 * shard_idx
    path = C.collect_task_dataset(
        spec,
        out_dir=shard_dir,
        split="train",
        n_worlds=int(job["worlds"]),
        nodes_per_world=int(params["nodes_per_world"]),
        roadmap_cfg=rt["roadmap_cfg"],
        feature_cfg=rt["feature_cfg"],
        seed=seed,
    )
    volume.commit()
    return str(path)


@app.function(image=image, timeout=60 * 20, volumes={VOLUME_ROOT: volume})
def merge_dataset(run_name: str, params: Dict[str, Any], task: str) -> str:
    volume.reload()
    run_dir = _run_dir(run_name)
    shard_root = run_dir / "datasets" / "_shards" / task
    shard_dirs = sorted([p for p in shard_root.glob("shard_*") if p.is_dir()])
    shard_npz = [p / f"{task}_train.npz" for p in shard_dirs]
    shard_meta = [p / f"{task}_train.json" for p in shard_dirs]
    missing = [str(p) for p in shard_npz + shard_meta if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing dataset shard artifacts: {missing[:8]}")
    out_npz = run_dir / "datasets" / f"{task}_train.npz"
    out_meta = run_dir / "datasets" / f"{task}_train.json"
    _merge_npz(shard_npz, out_npz, shard_meta, out_meta, task, "train")
    volume.commit()
    return str(out_npz)


@app.function(image=image, gpu="L4", timeout=60 * 60 * 4, volumes={VOLUME_ROOT: volume})
def train_avgbase_remote(run_name: str, params: Dict[str, Any], backbone: str) -> str:
    C = _import_common(params)
    volume.reload()
    run_dir = _run_dir(run_name)
    rt = _runtime(params, stage="c3", out_dir=run_dir)
    task_npz = {task: run_dir / "datasets" / f"{task}_train.npz" for task in rt["train_tasks"]}
    ckpt = C.train_avgbase(
        rt["backbone_cfgs"][backbone],
        task_npz,
        run_dir,
        rt["feature_cfg"],
        rt["train_cfg"],
        rt["device"],
        seed=int(params["seed"]) + 100,
    )
    volume.commit()
    return str(ckpt)


@app.function(image=image, gpu="L4", timeout=60 * 60 * 4, volumes={VOLUME_ROOT: volume})
def train_expert_remote(run_name: str, params: Dict[str, Any], backbone: str, task: str, alpha: float) -> str:
    C = _import_common(params)
    volume.reload()
    run_dir = _run_dir(run_name)
    rt = _runtime(params, stage="c3", out_dir=run_dir)
    ckpt = C.train_expert(
        rt["backbone_cfgs"][backbone],
        task,
        run_dir / "datasets" / f"{task}_train.npz",
        C.model_checkpoint_path(run_dir, backbone, "avgbase"),
        run_dir,
        rt["feature_cfg"],
        rt["train_cfg"],
        rt["device"],
        seed=int(params["seed"]) + 1_000 + 37 * rt["train_tasks"].index(task),
        alpha=float(alpha),
    )
    volume.commit()
    return str(ckpt)


def _load_c3_models(C, rt: Dict[str, Any], run_dir: Path):
    base_models: Dict[str, Any] = {}
    expert_cache: Dict[Tuple[str, str, float], Tuple[Any, float]] = {}
    for backbone in rt["backbones"]:
        cfg = rt["backbone_cfgs"][backbone]
        base_models[backbone] = C.load_base_model(cfg, rt["feature_cfg"], rt["train_cfg"], C.model_checkpoint_path(run_dir, backbone, "avgbase"), rt["device"])

    def get_expert(backbone: str, task: str, alpha: float):
        key = (backbone, task, float(alpha))
        if key not in expert_cache:
            cfg = rt["backbone_cfgs"][backbone]
            expert, bound = C.load_expert_model(
                cfg,
                rt["feature_cfg"],
                rt["train_cfg"],
                C.model_checkpoint_path(run_dir, backbone, "avgbase"),
                C.model_checkpoint_path(run_dir, backbone, "expert", task=task, alpha=alpha),
                rt["device"],
            )
            expert_cache[key] = (expert, float(bound))
        return expert_cache[key]

    return base_models, get_expert


@app.function(image=image, gpu="L4", timeout=60 * 60 * 4, volumes={VOLUME_ROOT: volume})
def eval_c3_shard(run_name: str, params: Dict[str, Any], job: Dict[str, Any]) -> str:
    import numpy as np

    C = _import_common(params)
    volume.reload()
    run_dir = _run_dir(run_name)
    rt = _runtime(params, stage="c3", out_dir=run_dir)
    suite = str(job["suite"])
    suite_idx = int(job["suite_idx"])
    shard_idx = int(job["shard_idx"])
    world_start = int(job["world_start"])
    count = int(job["worlds"])
    spec = rt["specs"][suite]
    base_models, get_expert = _load_c3_models(C, rt, run_dir)
    records: List[Dict[str, Any]] = []
    valid_worlds = 0
    attempts = 0
    while valid_worlds < count and attempts < count * rt["eval_cfg"].max_eval_world_retries:
        attempts += 1
        logical_world_idx = world_start + valid_worlds
        w_seed = int(params["seed"]) + 220_000 + 1_000_003 * (suite_idx + 1) + (logical_world_idx + 1) * 7919 + attempts
        bundle = C.make_eval_world_bundle(spec, rt["roadmap_cfg"], rt["feature_cfg"], w_seed, retries=1)
        if bundle is None:
            continue
        valid_worlds += 1
        world = bundle.world
        roadmap = bundle.roadmap
        features = bundle.features
        euclid_h = bundle.euclidean_h
        oracle_cost = bundle.oracle_cost
        method_heuristics: List[Tuple[str, str, str, float, Any, Dict[str, float]]] = [
            ("euclidean", "", "", 0.0, euclid_h, {"delta_mean": 0.0, "correction_abs_mean": 0.0})
        ]
        for backbone in rt["backbones"]:
            base_model = base_models[backbone]
            bpred = C.predict_norm_residuals(base_model, features, rt["device"], batch_size=2048)
            bpred = np.clip(bpred, 0.0, rt["train_cfg"].max_norm_residual)
            method_heuristics.append(("avgbase", backbone, "", 0.0, euclid_h + bpred * world.side_len, {"delta_mean": float(np.mean(bpred)), "correction_abs_mean": 0.0}))
            for alpha in rt["lora_alphas"]:
                for task in rt["train_tasks"]:
                    expert, bound = get_expert(backbone, task, alpha)
                    epred = C.predict_norm_residuals(expert, features, rt["device"], batch_size=2048)
                    correction = bound * np.tanh((epred - bpred) / max(bound, C.EPS))
                    final_delta = np.clip(bpred + correction, 0.0, rt["train_cfg"].max_norm_residual)
                    method_heuristics.append((
                        "tasklora",
                        backbone,
                        task,
                        float(alpha),
                        euclid_h + final_delta * world.side_len,
                        {"delta_mean": float(np.mean(final_delta)), "correction_abs_mean": float(np.mean(np.abs(correction)))},
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
                records.append({
                    "suite": suite,
                    "world_index": logical_world_idx,
                    "method": method,
                    "backbone": backbone,
                    "expert_task": expert_task,
                    "alpha": float(alpha),
                    "budget": int(budget),
                    "found": int(found),
                    "cost": cost,
                    "cost_ratio": float(cost / oracle_cost) if found and oracle_cost > C.EPS else float("nan"),
                    "oracle_cost": float(oracle_cost),
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
        if rt["eval_cfg"].eval_oracle_expert:
            world_rows = records[world_rows_start:]
            for backbone in rt["backbones"]:
                for alpha in rt["lora_alphas"]:
                    task_rows = [r for r in world_rows if r["method"] == "tasklora" and r["backbone"] == backbone and float(r["alpha"]) == float(alpha)]
                    for budget in rt["eval_cfg"].budgets:
                        candidates = [r for r in task_rows if int(r["budget"]) == int(budget)]
                        succ = [r for r in candidates if int(r["found"]) == 1]
                        if succ:
                            best = min(succ, key=lambda r: (float(r["expansions"]), float(r.get("cost_ratio", 999.0))))
                            records.append({**best, "method": "oracle_tasklora", "expert_task": str(best["expert_task"])})
                        elif candidates:
                            rep = min(candidates, key=lambda r: float(r["expansions"]))
                            records.append({**rep, "method": "oracle_tasklora", "expert_task": "<none>", "found": 0, "cost": float("nan"), "cost_ratio": float("nan")})
    out = run_dir / "results" / "_shards" / "c3" / suite / f"shard_{shard_idx:04d}.csv"
    _write_csv(out, records)
    volume.commit()
    return str(out)


@app.function(image=image, timeout=60 * 20, volumes={VOLUME_ROOT: volume})
def merge_c3_eval(run_name: str, params: Dict[str, Any]) -> str:
    C = _import_common(params)
    volume.reload()
    run_dir = _run_dir(run_name)
    rows: List[Dict[str, Any]] = []
    for p in sorted((run_dir / "results" / "_shards" / "c3").glob("*/*.csv")):
        rows.extend(_read_csv(p))
    raw_path = run_dir / "results" / "continuous_prm_eval_raw.csv"
    summary_path = run_dir / "results" / "continuous_prm_eval_summary.csv"
    _write_csv(raw_path, rows)
    summary = C.aggregate_eval_records(rows)
    _write_csv(summary_path, summary)
    _write_json(run_dir / "results" / "continuous_prm_eval_summary.json", {"rows": summary})
    volume.commit()
    return str(summary_path)


def _c4_weights(C, rt: Dict[str, Any], params: Dict[str, Any]):
    import numpy as np
    import random

    refs: Dict[str, np.ndarray] = {}
    rng = random.Random(int(params["seed"]) + 33_333)
    for task in rt["train_tasks"]:
        spec = rt["specs"][task]
        descs = []
        for _ in range(max(1, int(params["rbf_desc_samples"]))):
            obs = C.generate_obstacles(spec, rng)
            descs.append(C.task_descriptor(spec, obs).astype(np.float64)[:-1])
        refs[task] = np.mean(np.stack(descs, axis=0), axis=0)
    ref_mat = np.stack([refs[t] for t in rt["train_tasks"]], axis=0)
    scale = np.std(ref_mat, axis=0)
    scale = np.where(scale < 1.0e-4, 1.0, scale)
    return refs, ref_mat, scale


def _rbf_weights(z, ref_mat, scale, sigma: float, topk: int = 0):
    import numpy as np

    dz = (ref_mat - z[None, :]) / scale[None, :]
    dist2 = np.sum(dz * dz, axis=1)
    if int(topk) > 0 and int(topk) < len(dist2):
        keep = np.argsort(dist2)[: int(topk)]
        mask = np.ones(len(dist2), dtype=bool)
        mask[keep] = False
        dist2[mask] = np.inf
    w = np.exp(-0.5 * dist2 / max(float(sigma), 1.0e-6) ** 2)
    if not np.isfinite(w).all() or float(np.sum(w)) <= 1.0e-12:
        w = np.ones(len(dist2), dtype=np.float64)
    return w / float(np.sum(w))


@app.function(image=image, gpu="L4", timeout=60 * 60 * 4, volumes={VOLUME_ROOT: volume})
def eval_c4_shard(run_name: str, checkpoint_run_name: str, params: Dict[str, Any], job: Dict[str, Any]) -> str:
    import numpy as np

    C = _import_common(params)
    volume.reload()
    out_dir = _run_dir(run_name)
    asset_dir = _run_dir(checkpoint_run_name)
    rt = _runtime(params, stage="c4", out_dir=out_dir, checkpoint_dir=asset_dir)
    suite = str(job["suite"])
    suite_idx = int(job["suite_idx"])
    shard_idx = int(job["shard_idx"])
    world_start = int(job["world_start"])
    count = int(job["worlds"])
    spec = rt["specs"][suite]
    refs, ref_mat, scale = _c4_weights(C, rt, params)
    base_models, get_expert = _load_c3_models(C, rt, asset_dir)
    records: List[Dict[str, Any]] = []
    valid_worlds = 0
    attempts = 0
    while valid_worlds < count and attempts < count * rt["eval_cfg"].max_eval_world_retries:
        attempts += 1
        logical_world_idx = world_start + valid_worlds
        w_seed = int(params["seed"]) + 230_000 + 1_000_003 * (suite_idx + 1) + (logical_world_idx + 1) * 7919 + attempts
        bundle = C.make_eval_world_bundle(spec, rt["roadmap_cfg"], rt["feature_cfg"], w_seed, retries=1)
        if bundle is None:
            continue
        valid_worlds += 1
        world = bundle.world
        roadmap = bundle.roadmap
        features = bundle.features
        euclid_h = bundle.euclidean_h
        oracle_cost = bundle.oracle_cost
        z = world.descriptor.astype(np.float64)[:-1]
        weights = _rbf_weights(z, ref_mat, scale, float(params["rbf_sigma"]), int(params["rbf_topk"]))
        nearest_weights = _rbf_weights(z, ref_mat, scale, float(params["rbf_sigma"]), 1)
        nearest_idx = int(np.argmax(nearest_weights))
        method_heuristics: List[Tuple[str, str, str, float, Any, Dict[str, Any]]] = [
            ("euclidean", "", "", 0.0, euclid_h, {"delta_mean": 0.0, "correction_abs_mean": 0.0, "mixture_weights": ""})
        ]
        weights_json = json.dumps({task: float(w) for task, w in zip(rt["train_tasks"], weights)}, sort_keys=True)
        nearest_weights_json = json.dumps({task: float(w) for task, w in zip(rt["train_tasks"], nearest_weights)}, sort_keys=True)
        for backbone in rt["backbones"]:
            base_model = base_models[backbone]
            bpred = C.predict_norm_residuals(base_model, features, rt["device"], batch_size=2048)
            bpred = np.clip(bpred, 0.0, rt["train_cfg"].max_norm_residual)
            method_heuristics.append(("avgbase", backbone, "", 0.0, euclid_h + bpred * world.side_len, {"delta_mean": float(np.mean(bpred)), "correction_abs_mean": 0.0, "mixture_weights": ""}))
            for alpha in rt["lora_alphas"]:
                corrections = []
                individual = []
                for task in rt["train_tasks"]:
                    expert, bound = get_expert(backbone, task, alpha)
                    epred = C.predict_norm_residuals(expert, features, rt["device"], batch_size=2048)
                    corr = bound * np.tanh((epred - bpred) / max(bound, C.EPS))
                    corrections.append(corr)
                    final_delta = np.clip(bpred + corr, 0.0, rt["train_cfg"].max_norm_residual)
                    individual.append((task, euclid_h + final_delta * world.side_len, float(np.mean(final_delta)), float(np.mean(np.abs(corr)))))
                corr_mat = np.stack(corrections, axis=0)
                nearest_corr = corr_mat[nearest_idx]
                nearest_delta = np.clip(bpred + nearest_corr, 0.0, rt["train_cfg"].max_norm_residual)
                method_heuristics.append(("nearest_tasklora", backbone, rt["train_tasks"][nearest_idx], float(alpha), euclid_h + nearest_delta * world.side_len, {
                    "delta_mean": float(np.mean(nearest_delta)),
                    "correction_abs_mean": float(np.mean(np.abs(nearest_corr))),
                    "mixture_weights": nearest_weights_json,
                    "rbf_sigma": float(params["rbf_sigma"]),
                    "rbf_topk": 1,
                }))
                mix_corr = np.sum(weights[:, None] * corr_mat, axis=0)
                mix_delta = np.clip(bpred + mix_corr, 0.0, rt["train_cfg"].max_norm_residual)
                method_heuristics.append(("rbf_mix_tasklora", backbone, "<rbf_mix>", float(alpha), euclid_h + mix_delta * world.side_len, {
                    "delta_mean": float(np.mean(mix_delta)),
                    "correction_abs_mean": float(np.mean(np.abs(mix_corr))),
                    "mixture_weights": weights_json,
                    "rbf_sigma": float(params["rbf_sigma"]),
                    "rbf_topk": int(params["rbf_topk"]),
                }))
                if bool(params["include_expert_matrix"]):
                    for task, h_ind, dmean, cmean in individual:
                        method_heuristics.append(("tasklora", backbone, task, float(alpha), h_ind, {"delta_mean": dmean, "correction_abs_mean": cmean, "mixture_weights": ""}))
        world_rows_start = len(records)
        for method, backbone, expert_task, alpha, h, diag in method_heuristics:
            nonfinite = int(not np.isfinite(h).all())
            if nonfinite:
                h = np.nan_to_num(h, nan=rt["train_cfg"].max_norm_residual * world.side_len, posinf=rt["train_cfg"].max_norm_residual * world.side_len, neginf=0.0)
            for budget in rt["eval_cfg"].budgets:
                res = C.astar_search(roadmap.adj, h, budget=int(budget))
                found = bool(res["found"])
                cost = float(res["cost"])
                records.append({
                    "stage": "c4",
                    "suite": suite,
                    "world_index": logical_world_idx,
                    "method": method,
                    "backbone": backbone,
                    "expert_task": expert_task,
                    "alpha": float(alpha),
                    "budget": int(budget),
                    "found": int(found),
                    "cost": cost,
                    "cost_ratio": float(cost / oracle_cost) if found and oracle_cost > C.EPS else float("nan"),
                    "oracle_cost": float(oracle_cost),
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
        if bool(params["include_expert_matrix"]) and rt["eval_cfg"].eval_oracle_expert:
            world_rows = records[world_rows_start:]
            for backbone in rt["backbones"]:
                for alpha in rt["lora_alphas"]:
                    candidates_all = [r for r in world_rows if r["method"] == "tasklora" and r["backbone"] == backbone and float(r["alpha"]) == float(alpha)]
                    for budget in rt["eval_cfg"].budgets:
                        candidates = [r for r in candidates_all if int(r["budget"]) == int(budget)]
                        succ = [r for r in candidates if int(r["found"]) == 1]
                        if succ:
                            best = min(succ, key=lambda r: (float(r["expansions"]), float(r.get("cost_ratio", 999.0))))
                            records.append({**best, "method": "oracle_tasklora", "expert_task": str(best["expert_task"])})
                        elif candidates:
                            rep = min(candidates, key=lambda r: float(r["expansions"]))
                            records.append({**rep, "method": "oracle_tasklora", "expert_task": "<none>", "found": 0, "cost": float("nan"), "cost_ratio": float("nan")})
    out = out_dir / "results" / "_shards" / "c4" / suite / f"shard_{shard_idx:04d}.csv"
    _write_csv(out, records)
    volume.commit()
    return str(out)


@app.function(image=image, timeout=60 * 20, volumes={VOLUME_ROOT: volume})
def merge_c4_eval(run_name: str, params: Dict[str, Any]) -> str:
    C = _import_common(params)
    volume.reload()
    run_dir = _run_dir(run_name)
    rows: List[Dict[str, Any]] = []
    for p in sorted((run_dir / "results" / "_shards" / "c4").glob("*/*.csv")):
        rows.extend(_read_csv(p))
    raw_path = run_dir / "results" / "continuous_prm_c4_raw.csv"
    summary_path = run_dir / "results" / "continuous_prm_c4_summary.csv"
    _write_csv(raw_path, rows)
    summary = C.aggregate_eval_records(rows)
    _write_csv(summary_path, summary)
    _write_json(run_dir / "results" / "continuous_prm_c4_summary.json", {"rows": summary})
    volume.commit()
    return str(summary_path)


@app.function(image=image, timeout=60 * 60 * 6, volumes={VOLUME_ROOT: volume})
def collect_c6_dataset_shard(run_name: str, params: Dict[str, Any], job: Dict[str, Any]) -> str:
    C6 = _import_c6()
    volume.reload()
    run_dir = _run_dir(run_name)
    cfg = C6.config_from_params(params)
    if int(cfg.torch_threads) > 0:
        import torch

        torch.set_num_threads(int(cfg.torch_threads))
    C6.install_c5_hard_runtime(cfg.sector_tokens)
    specs = C6.C.build_anchor_specs()
    task = str(job["task"])
    shard_idx = int(job["shard_idx"])
    shard_dir = run_dir / "datasets" / "_shards" / task / f"shard_{shard_idx:04d}"
    seed = int(cfg.seed) + 10_000 * (int(job["task_idx"]) + 1) + 1_000_000 * shard_idx
    path = C6.collect_dataset(specs[task], shard_dir, "train", int(job["worlds"]), cfg, seed=seed)
    volume.commit()
    return str(path)


@app.function(image=image, timeout=60 * 20, volumes={VOLUME_ROOT: volume})
def merge_c6_dataset(run_name: str, params: Dict[str, Any], task: str) -> str:
    C6 = _import_c6()
    volume.reload()
    run_dir = _run_dir(run_name)
    shard_root = run_dir / "datasets" / "_shards" / task
    shard_dirs = sorted([p for p in shard_root.glob("shard_*") if p.is_dir()])
    shard_npz = [p / f"{task}_train.npz" for p in shard_dirs]
    shard_meta = [p / f"{task}_train.json" for p in shard_dirs]
    missing = [str(p) for p in shard_npz + shard_meta if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing C6 dataset shard artifacts: {missing[:8]}")
    out_npz = run_dir / "datasets" / f"{task}_train.npz"
    out_meta = run_dir / "datasets" / f"{task}_train.json"
    C6.merge_dataset_shards(shard_npz, out_npz, shard_meta, out_meta, task)
    volume.commit()
    return str(out_npz)


@app.function(image=image, gpu="L4", timeout=60 * 60 * 4, volumes={VOLUME_ROOT: volume})
def train_c6_model_remote(run_name: str, params: Dict[str, Any], model_name: str) -> str:
    import torch

    C6 = _import_c6()
    volume.reload()
    run_dir = _run_dir(run_name)
    cfg = C6.config_from_params(params)
    if int(cfg.torch_threads) > 0:
        torch.set_num_threads(int(cfg.torch_threads))
    device = torch.device("cpu" if bool(cfg.cpu) or not torch.cuda.is_available() else "cuda")
    tasks = C6.parse_csv(cfg.train_tasks)
    paths = [run_dir / "datasets" / f"{task}_train.npz" for task in tasks]
    ckpt = C6.train_model(model_name, paths, run_dir, cfg, device)
    volume.commit()
    return str(ckpt)


@app.function(image=image, gpu="L4", timeout=60 * 60 * 4, volumes={VOLUME_ROOT: volume})
def eval_c6_shard(run_name: str, params: Dict[str, Any], job: Dict[str, Any]) -> str:
    C6 = _import_c6()
    volume.reload()
    run_dir = _run_dir(run_name)
    cfg = C6.config_from_params(params)
    out = C6.evaluate_shard(
        run_dir,
        cfg,
        str(job["suite"]),
        int(job["suite_idx"]),
        int(job["world_start"]),
        int(job["worlds"]),
        int(job["shard_idx"]),
    )
    volume.commit()
    return str(out)


@app.function(image=image, timeout=60 * 20, volumes={VOLUME_ROOT: volume})
def merge_c6_eval(run_name: str, params: Dict[str, Any]) -> str:
    C6 = _import_c6()
    volume.reload()
    run_dir = _run_dir(run_name)
    summary = C6.merge_eval_shards(run_dir)
    volume.commit()
    return str(summary)


@app.function(image=image, timeout=60 * 5, volumes={VOLUME_ROOT: volume})
def inspect_run_artifacts(run_name: str) -> Dict[str, Any]:
    volume.reload()
    run_dir = _run_dir(run_name)
    checkpoints = sorted(str(p.relative_to(run_dir)) for p in (run_dir / "checkpoints").glob("*.pt"))
    datasets = sorted(str(p.relative_to(run_dir)) for p in (run_dir / "datasets").glob("*_train.npz"))
    results = sorted(str(p.relative_to(run_dir)) for p in (run_dir / "results").glob("*.csv"))
    c3_shards = sorted(str(p.relative_to(run_dir)) for p in (run_dir / "results" / "_shards" / "c3").glob("*/*.csv"))
    c4_shards = sorted(str(p.relative_to(run_dir)) for p in (run_dir / "results" / "_shards" / "c4").glob("*/*.csv"))
    c6_shards = sorted(str(p.relative_to(run_dir)) for p in (run_dir / "results" / "_shards" / "c6").glob("*/*.csv"))
    return {
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "datasets": datasets,
        "checkpoints": checkpoints,
        "results": results,
        "c3_shard_count": len(c3_shards),
        "c4_shard_count": len(c4_shards),
        "c6_shard_count": len(c6_shards),
        "c3_shards_sample": c3_shards[:10],
        "c4_shards_sample": c4_shards[:10],
        "c6_shards_sample": c6_shards[:10],
    }


@app.local_entrypoint()
def run_c3(
    run_name: str = "continuous_prm_c3_modal",
    train_worlds: int = 120,
    nodes_per_world: int = 160,
    eval_worlds: int = 40,
    roadmap_nodes: int = 256,
    roadmap_k: int = 14,
    base_epochs: int = 10,
    expert_epochs: int = 8,
    backbones: str = "hrm,onlstm",
    lora_alphas: str = "1.0",
    budgets: str = "100,200,500,1000",
    seed: int = 1234,
    train_tasks: str = "C_open,C_clutter,C_narrow,C_large_clutter",
    eval_suites: str = "C_open,C_clutter,C_narrow,C_large_clutter,C_extra_dense,C_tiny_passage,C_large_open,C_large_narrow,C_rectangles",
    batch_size: int = 256,
    lr: float = 2.0e-4,
    expert_lr: float = 1.5e-4,
    weight_decay: float = 1.0e-4,
    grad_clip: float = 1.0,
    max_norm_residual: float = 4.0,
    residual_bound_quantile: float = 0.95,
    residual_bound_floor: float = 0.08,
    correction_l2: float = 1.0e-3,
    roadmap_shard_worlds: int = 20,
    eval_shard_worlds: int = 10,
    cpu: bool = False,
    torch_threads: int = 1,
) -> None:
    params = _base_params(
        train_worlds=train_worlds,
        nodes_per_world=nodes_per_world,
        eval_worlds=eval_worlds,
        roadmap_nodes=roadmap_nodes,
        roadmap_k=roadmap_k,
        base_epochs=base_epochs,
        expert_epochs=expert_epochs,
        backbones=backbones,
        lora_alphas=lora_alphas,
        budgets=budgets,
        seed=seed,
        train_tasks=train_tasks,
        eval_suites=eval_suites,
        batch_size=batch_size,
        lr=lr,
        expert_lr=expert_lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        max_norm_residual=max_norm_residual,
        residual_bound_quantile=residual_bound_quantile,
        residual_bound_floor=residual_bound_floor,
        correction_l2=correction_l2,
        roadmap_shard_worlds=roadmap_shard_worlds,
        eval_shard_worlds=eval_shard_worlds,
        rbf_sigma=1.0,
        rbf_topk=0,
        rbf_desc_samples=96,
        include_expert_matrix=True,
        cpu=cpu,
        torch_threads=torch_threads,
    )
    print(f"run={run_name} volume={VOLUME_NAME}")
    collect_jobs = _collect_jobs(params)
    print(f"collecting {len(collect_jobs)} dataset shards")
    list(collect_dataset_shard.map([run_name] * len(collect_jobs), [params] * len(collect_jobs), collect_jobs))
    tasks = _csv(train_tasks)
    print(f"merging datasets: {tasks}")
    list(merge_dataset.map([run_name] * len(tasks), [params] * len(tasks), tasks))
    bbs = _csv(backbones)
    print(f"training avgbase: {bbs}")
    list(train_avgbase_remote.map([run_name] * len(bbs), [params] * len(bbs), bbs))
    expert_jobs = [(b, t, a) for b in bbs for t in tasks for a in _float_csv(lora_alphas)]
    print(f"training experts: {len(expert_jobs)}")
    list(train_expert_remote.map(
        [run_name] * len(expert_jobs),
        [params] * len(expert_jobs),
        [j[0] for j in expert_jobs],
        [j[1] for j in expert_jobs],
        [j[2] for j in expert_jobs],
    ))
    eval_jobs = _eval_jobs(params)
    print(f"evaluating C3 shards: {len(eval_jobs)}", flush=True)
    list(eval_c3_shard.map([run_name] * len(eval_jobs), [params] * len(eval_jobs), eval_jobs))
    summary = merge_c3_eval.remote(run_name, params)
    print(f"C3 summary: {summary}", flush=True)


@app.local_entrypoint()
def run_c5_hard(
    run_name: str = "continuous_prm_c5_hard_modal",
    train_worlds: int = 160,
    nodes_per_world: int = 192,
    eval_worlds: int = 80,
    roadmap_nodes: int = 192,
    roadmap_k: int = 7,
    base_epochs: int = 12,
    expert_epochs: int = 10,
    backbones: str = "hrm,onlstm",
    lora_alphas: str = "1.0",
    budgets: str = "128,136,144,152,168",
    seed: int = 1234,
    train_tasks: str = "C_hard_maze,C_hard_rooms",
    eval_suites: str = "C_hard_maze,C_hard_maze_dense,C_hard_rooms",
    batch_size: int = 256,
    lr: float = 2.0e-4,
    expert_lr: float = 1.5e-4,
    weight_decay: float = 1.0e-4,
    grad_clip: float = 1.0,
    max_norm_residual: float = 4.0,
    residual_bound_quantile: float = 0.95,
    residual_bound_floor: float = 0.08,
    correction_l2: float = 1.0e-3,
    roadmap_shard_worlds: int = 20,
    eval_shard_worlds: int = 1,
    nearest_obstacles: int = 12,
    num_rays: int = 48,
    ray_steps: int = 96,
    sector_tokens: int = 16,
    soft_residual_cap: bool = False,
    head_hidden: int = 256,
    hrm_hidden: int = 192,
    hrm_layers: int = 2,
    hrm_k_step: int = 2,
    hrm_heads: int = 4,
    onlstm_hidden: int = 256,
    onlstm_layers: int = 2,
    onlstm_chunk_size: int = 8,
    cpu: bool = False,
    torch_threads: int = 1,
) -> None:
    params = _base_params(
        train_worlds=train_worlds,
        nodes_per_world=nodes_per_world,
        eval_worlds=eval_worlds,
        roadmap_nodes=roadmap_nodes,
        roadmap_k=roadmap_k,
        base_epochs=base_epochs,
        expert_epochs=expert_epochs,
        backbones=backbones,
        lora_alphas=lora_alphas,
        budgets=budgets,
        seed=seed,
        train_tasks=train_tasks,
        eval_suites=eval_suites,
        batch_size=batch_size,
        lr=lr,
        expert_lr=expert_lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        max_norm_residual=max_norm_residual,
        residual_bound_quantile=residual_bound_quantile,
        residual_bound_floor=residual_bound_floor,
        correction_l2=correction_l2,
        roadmap_shard_worlds=roadmap_shard_worlds,
        eval_shard_worlds=eval_shard_worlds,
        rbf_sigma=1.0,
        rbf_topk=0,
        rbf_desc_samples=96,
        include_expert_matrix=True,
        cpu=cpu,
        torch_threads=torch_threads,
        profile="c5_hard",
        nearest_obstacles=nearest_obstacles,
        num_rays=num_rays,
        ray_steps=ray_steps,
        sector_tokens=sector_tokens,
        soft_residual_cap=soft_residual_cap,
        head_hidden=head_hidden,
        hrm_hidden=hrm_hidden,
        hrm_layers=hrm_layers,
        hrm_k_step=hrm_k_step,
        hrm_heads=hrm_heads,
        onlstm_hidden=onlstm_hidden,
        onlstm_layers=onlstm_layers,
        onlstm_chunk_size=onlstm_chunk_size,
    )
    print(f"run={run_name} profile=c5_hard volume={VOLUME_NAME}", flush=True)
    collect_jobs = _collect_jobs(params)
    print(f"collecting {len(collect_jobs)} C5 dataset shards", flush=True)
    list(collect_dataset_shard.map([run_name] * len(collect_jobs), [params] * len(collect_jobs), collect_jobs))
    tasks = _csv(train_tasks)
    print(f"merging C5 datasets: {tasks}", flush=True)
    list(merge_dataset.map([run_name] * len(tasks), [params] * len(tasks), tasks))
    bbs = _csv(backbones)
    print(f"training C5 avgbase: {bbs}", flush=True)
    list(train_avgbase_remote.map([run_name] * len(bbs), [params] * len(bbs), bbs))
    expert_jobs = [(b, t, a) for b in bbs for t in tasks for a in _float_csv(lora_alphas)]
    print(f"training C5 experts: {len(expert_jobs)}", flush=True)
    list(train_expert_remote.map(
        [run_name] * len(expert_jobs),
        [params] * len(expert_jobs),
        [j[0] for j in expert_jobs],
        [j[1] for j in expert_jobs],
        [j[2] for j in expert_jobs],
    ))
    eval_jobs = _eval_jobs(params)
    print(f"evaluating C5 shards: {len(eval_jobs)}", flush=True)
    list(eval_c3_shard.map([run_name] * len(eval_jobs), [params] * len(eval_jobs), eval_jobs))
    summary = merge_c3_eval.remote(run_name, params)
    print(f"C5 summary: {summary}", flush=True)


@app.local_entrypoint()
def run_c4(
    run_name: str = "continuous_prm_c4_modal",
    checkpoint_run_name: str = "continuous_prm_c3_modal",
    eval_worlds: int = 40,
    roadmap_nodes: int = 256,
    roadmap_k: int = 14,
    backbones: str = "hrm,onlstm",
    lora_alphas: str = "1.0",
    budgets: str = "100,200,500,1000",
    seed: int = 1234,
    train_tasks: str = "C_open,C_clutter,C_narrow,C_large_clutter",
    eval_suites: str = "C_open,C_clutter,C_narrow,C_large_clutter,C_extra_dense,C_tiny_passage,C_large_open,C_large_narrow,C_rectangles",
    batch_size: int = 256,
    rbf_sigma: float = 1.0,
    rbf_topk: int = 0,
    rbf_desc_samples: int = 96,
    include_expert_matrix: bool = False,
    eval_shard_worlds: int = 10,
    cpu: bool = False,
    torch_threads: int = 1,
) -> None:
    params = _base_params(
        train_worlds=0,
        nodes_per_world=160,
        eval_worlds=eval_worlds,
        roadmap_nodes=roadmap_nodes,
        roadmap_k=roadmap_k,
        base_epochs=10,
        expert_epochs=8,
        backbones=backbones,
        lora_alphas=lora_alphas,
        budgets=budgets,
        seed=seed,
        train_tasks=train_tasks,
        eval_suites=eval_suites,
        batch_size=batch_size,
        lr=2.0e-4,
        expert_lr=1.5e-4,
        weight_decay=1.0e-4,
        grad_clip=1.0,
        max_norm_residual=4.0,
        residual_bound_quantile=0.95,
        residual_bound_floor=0.08,
        correction_l2=1.0e-3,
        roadmap_shard_worlds=20,
        eval_shard_worlds=eval_shard_worlds,
        rbf_sigma=rbf_sigma,
        rbf_topk=rbf_topk,
        rbf_desc_samples=rbf_desc_samples,
        include_expert_matrix=include_expert_matrix,
        cpu=cpu,
        torch_threads=torch_threads,
    )
    print(f"run={run_name} checkpoint_run={checkpoint_run_name} volume={VOLUME_NAME}")
    jobs = _eval_jobs(params)
    print(f"evaluating C4 shards: {len(jobs)}")
    list(eval_c4_shard.map([run_name] * len(jobs), [checkpoint_run_name] * len(jobs), [params] * len(jobs), jobs))
    summary = merge_c4_eval.remote(run_name, params)
    print(f"C4 summary: {summary}")


@app.local_entrypoint()
def run_c6_heatmap(
    run_name: str = "continuous_prm_c6_heatmap_r1",
    train_worlds: int = 160,
    eval_worlds: int = 80,
    grid_size: int = 64,
    roadmap_nodes: int = 192,
    roadmap_k: int = 7,
    epochs: int = 16,
    models: str = "oracle,unet,onlstm,hrm",
    train_tasks: str = "C_hard_maze,C_hard_rooms",
    eval_suites: str = "C_hard_maze,C_hard_maze_dense,C_hard_rooms",
    budgets: str = "128,136,144,152,168",
    seed: int = 1234,
    batch_size: int = 16,
    lr: float = 2.0e-4,
    weight_decay: float = 1.0e-4,
    grad_clip: float = 1.0,
    loss_rank_weight: float = 0.2,
    loss_path_weight: float = 0.05,
    loss_consistency_weight: float = 0.02,
    rank_pairs: int = 256,
    max_world_retries: int = 120,
    sector_tokens: int = 16,
    dataset_shard_worlds: int = 10,
    eval_shard_worlds: int = 1,
    cpu: bool = False,
    torch_threads: int = 1,
    make_figures: bool = True,
) -> None:
    params = _c6_params(
        train_worlds=train_worlds,
        eval_worlds=eval_worlds,
        grid_size=grid_size,
        roadmap_nodes=roadmap_nodes,
        roadmap_k=roadmap_k,
        epochs=epochs,
        models=models,
        train_tasks=train_tasks,
        eval_suites=eval_suites,
        budgets=budgets,
        seed=seed,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        loss_rank_weight=loss_rank_weight,
        loss_path_weight=loss_path_weight,
        loss_consistency_weight=loss_consistency_weight,
        rank_pairs=rank_pairs,
        max_world_retries=max_world_retries,
        sector_tokens=sector_tokens,
        dataset_shard_worlds=dataset_shard_worlds,
        eval_shard_worlds=eval_shard_worlds,
        cpu=cpu,
        torch_threads=torch_threads,
        make_figures=make_figures,
    )
    print(f"run={run_name} profile=c6_heatmap volume={VOLUME_NAME}", flush=True)
    collect_jobs = _c6_collect_jobs(params)
    print(f"collecting C6 dataset shards: {len(collect_jobs)}", flush=True)
    list(collect_c6_dataset_shard.map([run_name] * len(collect_jobs), [params] * len(collect_jobs), collect_jobs))
    tasks = _csv(train_tasks)
    print(f"merging C6 datasets: {tasks}", flush=True)
    list(merge_c6_dataset.map([run_name] * len(tasks), [params] * len(tasks), tasks))
    learned_models = [m for m in _csv(models) if m != "oracle"]
    print(f"training C6 models: {learned_models}", flush=True)
    if learned_models:
        list(train_c6_model_remote.map([run_name] * len(learned_models), [params] * len(learned_models), learned_models))
    eval_jobs = _c6_eval_jobs(params)
    print(f"evaluating C6 shards: {len(eval_jobs)}", flush=True)
    list(eval_c6_shard.map([run_name] * len(eval_jobs), [params] * len(eval_jobs), eval_jobs))
    summary = merge_c6_eval.remote(run_name, params)
    print(f"C6 summary: {summary}", flush=True)


@app.local_entrypoint()
def run_c3_eval(
    run_name: str = "continuous_prm_c3_modal",
    eval_worlds: int = 40,
    roadmap_nodes: int = 256,
    roadmap_k: int = 14,
    backbones: str = "hrm,onlstm",
    lora_alphas: str = "1.0",
    budgets: str = "100,200,500,1000",
    seed: int = 1234,
    train_tasks: str = "C_open,C_clutter,C_narrow,C_large_clutter",
    eval_suites: str = "C_open,C_clutter,C_narrow,C_large_clutter,C_extra_dense,C_tiny_passage,C_large_open,C_large_narrow,C_rectangles",
    batch_size: int = 256,
    eval_shard_worlds: int = 10,
    cpu: bool = False,
    torch_threads: int = 1,
) -> None:
    """Resume only the C3 evaluation phase from existing Modal checkpoints."""
    params = _base_params(
        train_worlds=0,
        nodes_per_world=160,
        eval_worlds=eval_worlds,
        roadmap_nodes=roadmap_nodes,
        roadmap_k=roadmap_k,
        base_epochs=10,
        expert_epochs=8,
        backbones=backbones,
        lora_alphas=lora_alphas,
        budgets=budgets,
        seed=seed,
        train_tasks=train_tasks,
        eval_suites=eval_suites,
        batch_size=batch_size,
        lr=2.0e-4,
        expert_lr=1.5e-4,
        weight_decay=1.0e-4,
        grad_clip=1.0,
        max_norm_residual=4.0,
        residual_bound_quantile=0.95,
        residual_bound_floor=0.08,
        correction_l2=1.0e-3,
        roadmap_shard_worlds=20,
        eval_shard_worlds=eval_shard_worlds,
        rbf_sigma=1.0,
        rbf_topk=0,
        rbf_desc_samples=96,
        include_expert_matrix=True,
        cpu=cpu,
        torch_threads=torch_threads,
    )
    jobs = _eval_jobs(params)
    print(f"run={run_name} evaluating C3 shards: {len(jobs)}", flush=True)
    list(eval_c3_shard.map([run_name] * len(jobs), [params] * len(jobs), jobs))
    summary = merge_c3_eval.remote(run_name, params)
    print(f"C3 summary: {summary}", flush=True)


@app.local_entrypoint()
def run_c3_eval_one(
    run_name: str = "continuous_prm_c3_modal",
    suite: str = "C_open",
    shard_idx: int = 0,
    eval_worlds: int = 10,
    roadmap_nodes: int = 256,
    roadmap_k: int = 14,
    backbones: str = "hrm,onlstm",
    lora_alphas: str = "1.0",
    budgets: str = "100,200,500,1000",
    seed: int = 1234,
    train_tasks: str = "C_open,C_clutter,C_narrow,C_large_clutter",
    eval_suites: str = "C_open,C_clutter,C_narrow,C_large_clutter,C_extra_dense,C_tiny_passage,C_large_open,C_large_narrow,C_rectangles",
    batch_size: int = 256,
    cpu: bool = False,
    torch_threads: int = 2,
) -> None:
    params = _base_params(
        train_worlds=120,
        nodes_per_world=160,
        eval_worlds=eval_worlds,
        roadmap_nodes=roadmap_nodes,
        roadmap_k=roadmap_k,
        base_epochs=10,
        expert_epochs=8,
        backbones=backbones,
        lora_alphas=lora_alphas,
        budgets=budgets,
        seed=seed,
        train_tasks=train_tasks,
        eval_suites=eval_suites,
        batch_size=batch_size,
        lr=2.0e-4,
        expert_lr=1.5e-4,
        weight_decay=1.0e-4,
        grad_clip=1.0,
        max_norm_residual=4.0,
        residual_bound_quantile=0.95,
        residual_bound_floor=0.08,
        correction_l2=1.0e-3,
        roadmap_shard_worlds=20,
        eval_shard_worlds=eval_worlds,
        rbf_sigma=1.0,
        rbf_topk=0,
        rbf_desc_samples=96,
        include_expert_matrix=True,
        cpu=cpu,
        torch_threads=torch_threads,
    )
    suites = _csv(params["eval_suites"])
    if suite not in suites:
        raise ValueError(f"unknown suite {suite!r}; expected one of {suites}")
    job = {
        "suite": suite,
        "suite_idx": suites.index(suite),
        "shard_idx": int(shard_idx),
        "world_start": int(shard_idx) * int(eval_worlds),
        "worlds": int(eval_worlds),
    }
    print(f"run={run_name} evaluating one C3 shard: {job}", flush=True)
    out = eval_c3_shard.remote(run_name, params, job)
    print(f"C3 shard: {out}", flush=True)


@app.local_entrypoint()
def inspect_run(run_name: str = "continuous_prm_c3_modal") -> None:
    payload = inspect_run_artifacts.remote(run_name)
    print(json.dumps(payload, indent=2, sort_keys=True))
