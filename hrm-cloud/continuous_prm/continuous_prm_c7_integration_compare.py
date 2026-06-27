#!/usr/bin/env python3
"""
C7 Integration Comparison — orchestrator skeleton.

Wires together the C7 heuristic-provider pipeline (ScalarResidual + ValueField)
against the three C5-hard suites and the three new C7-hostile suites, running
matched-integrity A* on PRM graphs under multiple expansion budgets and focal
weights.

Modes
-----
collect   — generate roadmap worlds + run reference A* (Task 8)
train     — fit scalar-residual and value-field models (Task 8)
eval      — sharded evaluation of all arms (Task 9)
calibrate — per-suite budget calibration (Task 10)
analyze   — aggregate stats + pre-registered comparisons (Task 11)
full      — collect → train → calibrate → eval → analyze (Tasks 8-11)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import continuous_prm_common as C  # noqa: F401 — used by later tasks
import continuous_prm_providers as P  # noqa: F401 — used by later tasks
import continuous_prm_focal as focal  # noqa: F401 — used by later tasks
import continuous_prm_c7_hard_maps as M7

# Lazy imports for heavy modules (torch, C6 helpers) go inside functions to
# avoid pulling in GPU setup at argparse time.


# ---------------------------------------------------------------------------
# Helpers (mirrors C6 style; we re-export from C6 so future tasks can import
# from this module directly without depending on C6's internal structure)
# ---------------------------------------------------------------------------

from continuous_prm_c6_heatmap_value_field import (  # noqa: F401
    ensure_dir,
    parse_csv,
    parse_int_csv,
    write_json,
    now_str,
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class C7Config:
    # Grid / roadmap geometry
    grid_size: int = 64
    roadmap_nodes: int = 192
    roadmap_k: int = 7

    # Suite selection
    train_tasks: str = "C_hard_maze,C_hard_rooms,C_hard_spiral"
    eval_suites: str = (
        "C_hard_maze,C_hard_maze_dense,C_hard_rooms,"
        "C_hard_spiral,C_hard_bugtrap,C_hard_rooms_large"
    )

    # Model families to benchmark
    scalar_backbones: str = "hrm,onlstm"
    field_backbones: str = "unet,onlstm,hrm"

    # Expansion budgets (fallback until calibration overrides per-suite)
    budgets: str = "128,144,168"

    # Focal weight grid (filled by scale preset when empty)
    w_values: str = ""

    # World / training counts (0 = filled by scale preset)
    eval_worlds: int = 0
    train_worlds: int = 0
    epochs: int = 0

    # Misc
    seed: int = 1234
    scale: str = "local"
    mode: str = "full"
    out_dir: str = "runs/c7_local"
    cpu: bool = False
    sector_tokens: int = 16
    budget_grid_size: int = 0
    make_figures: bool = True


# ---------------------------------------------------------------------------
# Scale presets
# ---------------------------------------------------------------------------

def apply_scale_preset(cfg: C7Config) -> C7Config:
    if cfg.scale == "local":
        cfg.eval_worlds = cfg.eval_worlds or 24
        cfg.train_worlds = cfg.train_worlds or 96
        cfg.epochs = cfg.epochs or 16
        cfg.w_values = cfg.w_values or "1.0,1.1"
        cfg.budget_grid_size = cfg.budget_grid_size or 2
    else:  # cluster
        cfg.eval_worlds = cfg.eval_worlds or 120
        cfg.train_worlds = cfg.train_worlds or 160
        cfg.epochs = cfg.epochs or 24
        cfg.w_values = cfg.w_values or "1.0,1.05,1.1,1.25"
        cfg.budget_grid_size = cfg.budget_grid_size or 3
    return cfg


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="C7 Integration Comparison: scalar-residual vs value-field heuristics"
    )
    p.add_argument(
        "--mode",
        choices=("collect", "train", "eval", "analyze", "full", "calibrate"),
        default="full",
    )
    p.add_argument(
        "--scale",
        choices=("local", "cluster"),
        default="local",
    )
    p.add_argument("--out-dir", type=str, default="runs/c7_local")
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--roadmap-nodes", type=int, default=192)
    p.add_argument("--roadmap-k", type=int, default=7)
    p.add_argument("--train-tasks", type=str, default="C_hard_maze,C_hard_rooms,C_hard_spiral")
    p.add_argument(
        "--eval-suites",
        type=str,
        default=(
            "C_hard_maze,C_hard_maze_dense,C_hard_rooms,"
            "C_hard_spiral,C_hard_bugtrap,C_hard_rooms_large"
        ),
    )
    p.add_argument("--scalar-backbones", type=str, default="hrm,onlstm")
    p.add_argument("--field-backbones", type=str, default="unet,onlstm,hrm")
    p.add_argument("--budgets", type=str, default="128,144,168")
    # Preset-filled fields default to 0/"" so the preset can fill them
    p.add_argument("--w-values", type=str, default="")
    p.add_argument("--eval-worlds", type=int, default=0)
    p.add_argument("--train-worlds", type=int, default=0)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--sector-tokens", type=int, default=16)
    p.add_argument("--budget-grid-size", type=int, default=0)
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> C7Config:
    return C7Config(
        grid_size=int(args.grid_size),
        roadmap_nodes=int(args.roadmap_nodes),
        roadmap_k=int(args.roadmap_k),
        train_tasks=str(args.train_tasks),
        eval_suites=str(args.eval_suites),
        scalar_backbones=str(args.scalar_backbones),
        field_backbones=str(args.field_backbones),
        budgets=str(args.budgets),
        w_values=str(args.w_values),
        eval_worlds=int(args.eval_worlds),
        train_worlds=int(args.train_worlds),
        epochs=int(args.epochs),
        seed=int(args.seed),
        scale=str(args.scale),
        mode=str(args.mode),
        out_dir=str(args.out_dir),
        cpu=bool(args.cpu),
        sector_tokens=int(args.sector_tokens),
        budget_grid_size=int(args.budget_grid_size),
        make_figures=not bool(args.no_figures),
    )


# ---------------------------------------------------------------------------
# Collect / train (Task 8)
# ---------------------------------------------------------------------------

# Scalar collection node count. Mirrors the production C1/C2/C3 PRM training
# runs (continuous_prm_modal defaults / config__c*.json all use 160).
SCALAR_NODES_PER_WORLD = 160

# Field datasets live under out_dir/"datasets" (C6.collect_all's own path). Scalar
# datasets MUST live in a SEPARATE directory: C6 writes per-grid occupancy npz to
# {name}_train.npz, while the scalar path writes per-node feature/residual npz —
# same filename, incompatible format. Both early-return if the npz exists, so a
# shared directory would silently load the wrong data. Keep them apart.
SCALAR_DATASET_DIRNAME = "datasets_scalar"
SCALAR_SPLIT = "train_scalar"


def _pick_device(cfg: C7Config):
    import torch
    return torch.device("cpu" if cfg.cpu or not torch.cuda.is_available() else "cuda")


def _scalar_backbone_cfg(name: str) -> "C.BackboneConfig":
    """Build a scalar BackboneConfig from a backbone name.

    Mirrors ScalarResidualProvider.untrained_for_test in continuous_prm_providers:
    a minimal argparse.Namespace carrying exactly the fields build_backbone_configs
    reads, then index by backbone name. We use the production-scale hidden dims
    (the provider's test uses tiny dims; for real training we want the common.py
    build_backbone_configs defaults, which read these Namespace fields).
    """
    ns = argparse.Namespace(
        hrm_hidden=192,
        hrm_layers=2,
        hrm_k_step=2,
        hrm_heads=4,
        head_hidden=256,
        onlstm_hidden=192,
        onlstm_layers=2,
        onlstm_chunk_size=8,
    )
    configs = C.build_backbone_configs(ns)
    if name not in configs:
        raise ValueError(f"unknown scalar backbone {name!r}; have {sorted(configs)}")
    return configs[name]


def _c6config_from_c7(cfg: C7Config) -> "object":
    """Build a C6Config (field path) from the C7 config.

    Field models only: cfg.field_backbones (NO 'oracle'). Eval-only C6Config
    fields (eval_worlds/eval_suites/budgets) are irrelevant for collect/train and
    keep their C6 defaults.
    """
    import continuous_prm_c6_heatmap_value_field as C6
    return C6.C6Config(
        grid_size=int(cfg.grid_size),
        train_worlds=int(cfg.train_worlds),
        roadmap_nodes=int(cfg.roadmap_nodes),
        roadmap_k=int(cfg.roadmap_k),
        epochs=int(cfg.epochs),
        train_tasks=str(cfg.train_tasks),
        models=str(cfg.field_backbones),
        seed=int(cfg.seed),
        sector_tokens=int(cfg.sector_tokens),
        cpu=bool(cfg.cpu),
        make_figures=False,
    )


def _collect_field(out_dir: Path, cfg: C7Config) -> Dict[str, Path]:
    """Collect field (C6) datasets into out_dir/'datasets'.

    We deliberately do NOT call C6.collect_all: it calls install_c5_hard_runtime
    at its start (restoring the C5-only build_anchor_specs), so a C7-only suite
    such as C_hard_spiral is no longer registered and the lookup KeyErrors.
    Instead we replicate collect_all's loop (same datasets dir, "train" split,
    seed = seed + 10_000*(idx+1)) using C6.collect_dataset, which performs no
    runtime install. The C7 maps must already be installed by the caller so
    C.build_anchor_specs returns all six suites.
    """
    import continuous_prm_c6_heatmap_value_field as C6
    c6cfg = _c6config_from_c7(cfg)
    M7.install_c7_hard_maps(cfg.sector_tokens)
    specs = C.build_anchor_specs()
    datasets_dir = ensure_dir(out_dir / "datasets")
    paths: Dict[str, Path] = {}
    tasks = parse_csv(cfg.train_tasks)
    print(f"[{now_str()}] C7 collect: field datasets (tasks={cfg.train_tasks}) -> {datasets_dir}", flush=True)
    for idx, task in enumerate(tasks):
        if task not in specs:
            raise KeyError(f"field collect: unknown task {task!r}; have {sorted(specs)}")
        paths[task] = C6.collect_dataset(
            specs[task],
            datasets_dir,
            "train",
            int(cfg.train_worlds),
            c6cfg,
            seed=int(cfg.seed) + 10_000 * (idx + 1),
        )
    print(f"[{now_str()}] C7 collect: field datasets done -> {[str(p) for p in paths.values()]}", flush=True)
    return paths


def _collect_scalar(out_dir: Path, cfg: C7Config) -> Dict[str, Path]:
    """Collect scalar (C5/common) per-node datasets into a SEPARATE directory.

    Requires the C7 hard maps to be installed (so C.build_anchor_specs and the
    hard feature encoder C.make_features_for_roadmap are active). Uses distinct
    per-task seeds mirroring C6.collect_all's seed + 10_000*(idx+1) scheme.
    """
    scalar_dir = ensure_dir(out_dir / SCALAR_DATASET_DIRNAME)
    specs = C.build_anchor_specs()
    roadmap_cfg = C.RoadmapConfig(n_nodes=int(cfg.roadmap_nodes), k_neighbors=int(cfg.roadmap_k))
    feature_cfg = C.FeatureConfig()
    paths: Dict[str, Path] = {}
    tasks = parse_csv(cfg.train_tasks)
    print(
        f"[{now_str()}] C7 collect: scalar datasets (tasks={cfg.train_tasks}, "
        f"nodes_per_world={SCALAR_NODES_PER_WORLD}) -> {scalar_dir}",
        flush=True,
    )
    for idx, task in enumerate(tasks):
        if task not in specs:
            raise KeyError(f"scalar collect: unknown task {task!r}; have {sorted(specs)}")
        # NOTE: collect_task_dataset early-returns if the npz+meta already exist and writes
        # the npz non-atomically; if a collection is interrupted after both land, delete the
        # datasets_scalar dir before re-running to avoid reusing a truncated file.
        paths[task] = C.collect_task_dataset(
            specs[task],
            scalar_dir,
            SCALAR_SPLIT,
            int(cfg.train_worlds),
            SCALAR_NODES_PER_WORLD,
            roadmap_cfg,
            feature_cfg,
            seed=int(cfg.seed) + 10_000 * (idx + 1),
        )
    print(f"[{now_str()}] C7 collect: scalar datasets done -> {[str(p) for p in paths.values()]}", flush=True)
    return paths


def run_collect(out_dir: Path, cfg: C7Config) -> Dict[str, Dict[str, Path]]:
    """Collect BOTH dataset families. Returns {'field': {...}, 'scalar': {...}}."""
    field_paths = _collect_field(out_dir, cfg)
    # Defensive: ensure the C7 suites + hard feature encoder are active before
    # scalar collection (C_hard_spiral lives in the C7 runtime). install is idempotent.
    M7.install_c7_hard_maps(cfg.sector_tokens)
    scalar_paths = _collect_scalar(out_dir, cfg)
    return {"field": field_paths, "scalar": scalar_paths}


def _train_field(out_dir: Path, cfg: C7Config, dataset_paths: Dict[str, Path], device) -> Dict[str, Path]:
    """Train each field backbone over the union of field datasets (C6 path)."""
    import continuous_prm_c6_heatmap_value_field as C6
    c6cfg = _c6config_from_c7(cfg)
    print(f"[{now_str()}] C7 train: field backbones={cfg.field_backbones}", flush=True)
    C6.run_train(out_dir, c6cfg, dataset_paths, device)
    ckpts: Dict[str, Path] = {}
    for name in parse_csv(cfg.field_backbones):
        if name == "oracle":
            continue
        ckpts[name] = C6.checkpoint_path(out_dir, name)
    return ckpts


def _train_scalar(out_dir: Path, cfg: C7Config, dataset_paths: Dict[str, Path], device) -> Dict[str, Path]:
    """Train each scalar avgbase backbone over the pooled scalar datasets."""
    feature_cfg = C.FeatureConfig()
    train_cfg = C.TrainingConfig(base_epochs=int(cfg.epochs))
    ckpts: Dict[str, Path] = {}
    print(f"[{now_str()}] C7 train: scalar avgbase backbones={cfg.scalar_backbones}", flush=True)
    for name in parse_csv(cfg.scalar_backbones):
        backbone_cfg = _scalar_backbone_cfg(name)
        ckpts[name] = C.train_avgbase(
            backbone_cfg,
            dataset_paths,
            out_dir,
            feature_cfg,
            train_cfg,
            device,
            seed=int(cfg.seed),
        )
    return ckpts


def run_train_all(out_dir: Path, cfg: C7Config, device) -> Dict[str, object]:
    """Collect (if needed) then train BOTH model families; write train_manifest.json."""
    collected = run_collect(out_dir, cfg)
    # Runtime-install ordering: install_c7_hard_maps() (called in run_collect / main)
    # composes on top of install_c5_hard_runtime and re-applies the C7 suite wrappers.
    # Neither C6.run_train nor C.train_avgbase calls install_c5_hard_runtime, so the
    # C7 suites stay registered for the whole train run; field-then-scalar order is safe.
    field_ckpts = _train_field(out_dir, cfg, collected["field"], device)
    scalar_ckpts = _train_scalar(out_dir, cfg, collected["scalar"], device)

    manifest = {
        "stage": "c7_train",
        "timestamp": now_str(),
        "train_tasks": parse_csv(cfg.train_tasks),
        "field": {
            "backbones": [m for m in parse_csv(cfg.field_backbones) if m != "oracle"],
            "datasets": {task: str(p) for task, p in collected["field"].items()},
            "checkpoints": {
                name: str(p) for name, p in field_ckpts.items() if Path(p).exists()
            },
        },
        "scalar": {
            "backbones": parse_csv(cfg.scalar_backbones),
            "nodes_per_world": SCALAR_NODES_PER_WORLD,
            "datasets": {task: str(p) for task, p in collected["scalar"].items()},
            "checkpoints": {
                name: str(p) for name, p in scalar_ckpts.items() if Path(p).exists()
            },
        },
    }
    manifest_path = Path(out_dir) / "train_manifest.json"
    write_json(manifest_path, manifest)
    print(f"[{now_str()}] C7 train: wrote manifest -> {manifest_path}", flush=True)
    print(
        f"[{now_str()}] C7 train: field checkpoints={list(manifest['field']['checkpoints'].values())} "
        f"scalar checkpoints={list(manifest['scalar']['checkpoints'].values())}",
        flush=True,
    )
    return manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    cfg = apply_scale_preset(cfg)

    # Install all six suites (3 C5-hard + 3 C7) into the common registry.
    M7.install_c7_hard_maps(cfg.sector_tokens)

    out_dir = ensure_dir(cfg.out_dir)

    print(
        f"[{now_str()}] C7 mode={cfg.mode} scale={cfg.scale} "
        f"out_dir={out_dir} cpu={cfg.cpu} "
        f"eval_worlds={cfg.eval_worlds} train_worlds={cfg.train_worlds} "
        f"epochs={cfg.epochs} w_values={cfg.w_values}",
        flush=True,
    )

    if cfg.mode in ("collect",):
        device = _pick_device(cfg)
        print(f"[{now_str()}] C7 collect: device={device}", flush=True)
        run_collect(out_dir, cfg)
    elif cfg.mode == "train":
        device = _pick_device(cfg)
        print(f"[{now_str()}] C7 train: device={device}", flush=True)
        run_train_all(out_dir, cfg, device)
    elif cfg.mode == "eval":
        raise NotImplementedError("C7 eval mode: implemented in Task 9")
    elif cfg.mode == "calibrate":
        raise NotImplementedError("C7 calibrate mode: implemented in Task 10")
    elif cfg.mode == "analyze":
        raise NotImplementedError("C7 analyze mode: implemented in Task 11")
    elif cfg.mode == "full":
        raise NotImplementedError("C7 full mode: implemented in Tasks 8-11")
    else:
        raise ValueError(f"unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
