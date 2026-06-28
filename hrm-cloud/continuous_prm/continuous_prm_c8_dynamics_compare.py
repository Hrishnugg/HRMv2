#!/usr/bin/env python3
"""
C8 Dynamics Comparison — orchestrator skeleton.

Wires together the C8 space-time heuristic-provider pipeline (ScalarTemporal +
ValueFieldTemporal) against the three C8-dynamic train suites and the three new
C8-dynamic held-out suites, running matched-integrity space-time A* on PRM
graphs under multiple expansion budgets and focal weights.

Modes
-----
collect   — generate roadmap worlds + run reference space-time A* (Task 10)
train     — fit scalar-temporal and value-field-temporal models (Task 10)
eval      — sharded evaluation of all arms (Task 11)
calibrate — per-suite budget calibration (Task 12)
analyze   — aggregate stats + pre-registered comparisons (Task 13)
full      — collect → train → calibrate → eval → analyze (Tasks 10-13)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import continuous_prm_common as C
import continuous_prm_dynamic_providers as P
import continuous_prm_spacetime as ST
import continuous_prm_dynamics as D
import continuous_prm_c8_dynamic_maps as M8

# Lazy imports for heavy modules (torch, C6 helpers) go inside functions to
# avoid pulling in GPU setup at argparse time.


# ---------------------------------------------------------------------------
# Helpers (mirrors C7 style; re-export from C6 so future tasks can import
# from this module directly without depending on C6's internal structure)
# ---------------------------------------------------------------------------

from continuous_prm_c6_heatmap_value_field import (  # noqa: F401
    ensure_dir,
    parse_csv,
    parse_int_csv,
    write_csv,
    write_json,
    read_csv,
    now_str,
    mcnemar_exact_p,
    bh_q_values,
    sanitize_name,
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class C8Config:
    # Grid / roadmap geometry
    grid_size: int = 64
    roadmap_nodes: int = 192
    roadmap_k: int = 7

    # Suite selection
    train_tasks: str = "C_dyn_maze,C_dyn_rooms,C_dyn_spiral"
    eval_suites: str = (
        "C_dyn_maze,C_dyn_rooms,C_dyn_spiral,"
        "C_dyn_maze_dense,C_dyn_crossing,C_dyn_rooms_large"
    )

    # Model families to benchmark
    scalar_backbones: str = "hrm,onlstm"
    field_backbones: str = "unet,onlstm,hrm"

    # Expansion budgets (fallback until calibration overrides per-suite).
    # Larger than the C7 default because space-time graphs are denser.
    budgets: str = "2000"

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
    out_dir: str = "runs/c8_local"
    cpu: bool = False
    budget_grid_size: int = 0
    make_figures: bool = True

    # Dynamics-specific knobs
    # window_w: rollout window length for temporal heuristics (number of time
    # steps a scalar/field model looks back when building the feature vector).
    window_w: int = 8
    # k_patrollers: number of nearest patrollers used in scalar feature vectors.
    k_patrollers: int = 4

    # NOTE: per-suite v_agent / dt / t_max are NOT global config — they come
    # from M8.dynamics_params(suite) and are read per suite in eval/calibrate/
    # train modes. They vary by suite geometry and patroller density, so a
    # single global value would be incorrect.


# ---------------------------------------------------------------------------
# Scale presets
# (Local is smaller than C7 because each space-time eval is heavier —
# dynamics worlds include temporal graphs that are ~t_max times larger.)
# ---------------------------------------------------------------------------

def apply_scale_preset(cfg: C8Config) -> C8Config:
    if cfg.scale == "local":
        cfg.eval_worlds = cfg.eval_worlds or 16
        cfg.train_worlds = cfg.train_worlds or 64
        cfg.epochs = cfg.epochs or 12
        cfg.w_values = cfg.w_values or "1.0,1.1"
        cfg.budget_grid_size = cfg.budget_grid_size or 2
    else:  # cluster
        cfg.eval_worlds = cfg.eval_worlds or 80
        cfg.train_worlds = cfg.train_worlds or 120
        cfg.epochs = cfg.epochs or 20
        cfg.w_values = cfg.w_values or "1.0,1.05,1.1,1.25"
        cfg.budget_grid_size = cfg.budget_grid_size or 3
    return cfg


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "C8 Dynamics Comparison: scalar-temporal vs value-field-temporal "
            "heuristics on dynamic moving-obstacle suites"
        )
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
    p.add_argument("--out-dir", type=str, default="runs/c8_local")
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--roadmap-nodes", type=int, default=192)
    p.add_argument("--roadmap-k", type=int, default=7)
    p.add_argument("--train-tasks", type=str, default="C_dyn_maze,C_dyn_rooms,C_dyn_spiral")
    p.add_argument(
        "--eval-suites",
        type=str,
        default=(
            "C_dyn_maze,C_dyn_rooms,C_dyn_spiral,"
            "C_dyn_maze_dense,C_dyn_crossing,C_dyn_rooms_large"
        ),
    )
    p.add_argument("--scalar-backbones", type=str, default="hrm,onlstm")
    p.add_argument("--field-backbones", type=str, default="unet,onlstm,hrm")
    p.add_argument("--budgets", type=str, default="2000")
    # Preset-filled fields default to 0/"" so the preset can fill them
    p.add_argument("--w-values", type=str, default="")
    p.add_argument("--eval-worlds", type=int, default=0)
    p.add_argument("--train-worlds", type=int, default=0)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--budget-grid-size", type=int, default=0)
    p.add_argument("--no-figures", action="store_true")
    # Dynamics-specific knobs
    p.add_argument(
        "--window-w",
        type=int,
        default=8,
        help="Rollout window length for temporal heuristics (time steps looked back).",
    )
    p.add_argument(
        "--k-patrollers",
        type=int,
        default=4,
        help="Number of nearest patrollers included in scalar feature vectors.",
    )
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> C8Config:
    return C8Config(
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
        budget_grid_size=int(args.budget_grid_size),
        make_figures=not bool(args.no_figures),
        window_w=int(args.window_w),
        k_patrollers=int(args.k_patrollers),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    cfg = apply_scale_preset(cfg)

    # Install all dynamic suites (composes on C7 hard maps + C5 hard runtime)
    # into the common registry so C.build_anchor_specs() returns all six C8
    # dynamic suites alongside the static C7/C5 suites.
    M8.install_c8_dynamic_maps()

    out_dir = ensure_dir(cfg.out_dir)

    print(
        f"[{now_str()}] C8 mode={cfg.mode} scale={cfg.scale} "
        f"out_dir={out_dir} cpu={cfg.cpu} "
        f"eval_worlds={cfg.eval_worlds} train_worlds={cfg.train_worlds} "
        f"epochs={cfg.epochs} w_values={cfg.w_values} "
        f"window_w={cfg.window_w} k_patrollers={cfg.k_patrollers}",
        flush=True,
    )

    if cfg.mode == "collect":
        raise NotImplementedError("C8 collect: Task 10")
    elif cfg.mode == "train":
        raise NotImplementedError("C8 train: Task 10")
    elif cfg.mode == "eval":
        raise NotImplementedError("C8 eval: Task 11")
    elif cfg.mode == "calibrate":
        raise NotImplementedError("C8 calibrate: Task 12")
    elif cfg.mode == "analyze":
        raise NotImplementedError("C8 analyze: Task 13")
    elif cfg.mode == "full":
        raise NotImplementedError("C8 full: Tasks 10-13")
    else:
        raise ValueError(f"unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
