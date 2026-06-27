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
        raise NotImplementedError("C7 collect mode: implemented in Task 8")
    elif cfg.mode == "train":
        raise NotImplementedError("C7 train mode: implemented in Task 8")
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
