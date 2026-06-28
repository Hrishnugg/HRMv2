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
import csv
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
    write_csv,
    write_json,
    read_csv,
    finite_mean,
    now_str,
    mcnemar_exact_p,
    bh_q_values,
    sanitize_name,
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
# Eval (Task 9): unified, matched, sharded multi-arm evaluation
# ---------------------------------------------------------------------------

def _load_eval_providers(out_dir: Path, cfg: C7Config, device) -> Dict[str, "P.HeuristicProvider"]:
    """Build every arm's provider ONCE, keyed by provider.name.

    Always present: euclid, oracle. Plus one scalar_<bb> per scalar backbone with
    a trained avgbase checkpoint, and one field_<bb> per field backbone with a
    trained C6 checkpoint. Missing checkpoints are skipped with a log line.
    """
    import torch
    import continuous_prm_c6_heatmap_value_field as C6

    # Ensure C7 suites + hard encoder are active before any model construction.
    M7.install_c7_hard_maps(cfg.sector_tokens)

    providers: Dict[str, "P.HeuristicProvider"] = {}
    eu = P.EuclidProvider(); providers[eu.name] = eu        # "euclid"
    orc = P.OracleProvider(); providers[orc.name] = orc     # "oracle"

    for bb in parse_csv(cfg.scalar_backbones):
        ckpt = C.model_checkpoint_path(out_dir, bb, "avgbase")
        if not ckpt.exists():
            print(f"[c7] skip scalar {bb}: no checkpoint {ckpt}", flush=True)
            continue
        payload = torch.load(ckpt, map_location="cpu")
        bbcfg = C.BackboneConfig(**payload["backbone_cfg"])
        fcfg = C.FeatureConfig(**payload["feature_cfg"])
        tcfg = C.TrainingConfig(**payload["train_cfg"])
        model = C.load_base_model(bbcfg, fcfg, tcfg, ckpt, device)
        prov = P.ScalarResidualProvider(model, fcfg, device, bb, tcfg.max_norm_residual)
        providers[prov.name] = prov                         # "scalar_<bb>"

    for bb in parse_csv(cfg.field_backbones):
        if bb == "oracle":
            continue
        ckpt = C6.checkpoint_path(out_dir, bb)
        if not ckpt.exists():
            print(f"[c7] skip field {bb}: no checkpoint {ckpt}", flush=True)
            continue
        model = C6.load_model(bb, ckpt, device)
        prov = P.ValueFieldProvider(model, cfg.grid_size, device, bb)
        providers[prov.name] = prov                         # "field_<bb>"

    # Re-assert C7 maps defensively (model loading touches no runtime install, but
    # keep parity with the plan in case a future loader path does).
    M7.install_c7_hard_maps(cfg.sector_tokens)
    return providers


def _load_per_suite_budgets(out_dir: Path, cfg: C7Config):
    """Return a callable suite -> list[int] of per-suite expansion budgets.

    If out_dir/calibration.json exists (written by Task 10), per-suite budgets are
    read from it. Expected shape (Task 10 must match this):
        {"budgets": {suite: [b1, b2, ...]}, ...}
    Suites absent from the calibration map fall back to parse_int_csv(cfg.budgets).
    """
    import json
    fallback = [int(b) for b in parse_int_csv(cfg.budgets)]
    calib_budgets: Dict[str, list] = {}
    calib_path = Path(out_dir) / "calibration.json"
    if calib_path.exists():
        try:
            calib = json.loads(calib_path.read_text())
            calib_budgets = dict(calib.get("budgets", {}) or {})
        except (ValueError, OSError) as exc:
            print(f"[c7] eval: failed to read {calib_path} ({exc}); using fallback budgets", flush=True)

    def per_suite_budgets(suite: str):
        raw = calib_budgets.get(suite)
        if raw:
            return [int(b) for b in raw]
        return list(fallback)

    return per_suite_budgets


def iter_matched_worlds(spec, suite_idx, cfg, roadmap_cfg, n_worlds, retry=200):
    """Yield (world_index, world, roadmap) for up to n_worlds connected worlds,
    using the deterministic eval seed formula. Shared by eval and calibrate.

    The seed formula and skip rules (invalid world / disconnected PRM) match the
    original run_eval loop EXACTLY so that eval and calibrate observe identical
    worlds for a given (suite, world_index). Callers that need under-fill
    detection should count yielded worlds and compare against n_worlds.
    """
    valid = 0
    attempt = 0
    while valid < n_worlds and attempt < n_worlds * retry:
        attempt += 1
        w_seed = int(cfg.seed) + 770_000 + 1_000_003 * (suite_idx + 1) + (valid + 1) * 7919 + attempt
        world = C.build_world(spec, w_seed, roadmap_cfg.min_start_goal_dist_frac)
        if world is None:
            continue
        rm = C.build_prm(world, roadmap_cfg, seed=w_seed + 17)
        if rm is None or not rm.connected_to_goal[0]:
            continue
        yield valid, world, rm
        valid += 1


def run_eval(out_dir: Path, cfg: C7Config, device) -> Path:
    """Matched, sharded multi-arm eval across eval suites; write per-suite shard
    CSVs + a merged raw CSV. Returns the merged-CSV path.

    Worlds+PRMs are generated per suite from a seeded retry loop and shared across
    ALL arms (matched). Records carry provider/mode/w/budget (from run_world_arms)
    plus suite/world_index added here; world.meta is intentionally NOT spread in.
    """
    out_dir = Path(out_dir)
    providers = _load_eval_providers(out_dir, cfg, device)
    per_suite_budgets = _load_per_suite_budgets(out_dir, cfg)
    print(
        f"[{now_str()}] C7 eval: providers={sorted(providers)} "
        f"eval_worlds={cfg.eval_worlds} w_values={cfg.w_values}",
        flush=True,
    )

    roadmap_cfg = C.RoadmapConfig(n_nodes=cfg.roadmap_nodes, k_neighbors=cfg.roadmap_k)
    w_values = [float(x) for x in parse_csv(cfg.w_values)]
    specs = C.build_anchor_specs()
    for suite_idx, suite in enumerate(parse_csv(cfg.eval_suites)):
        if suite not in specs:
            raise KeyError(f"eval: unknown suite {suite!r}; have {sorted(specs)}")
        spec = specs[suite]
        budgets = per_suite_budgets(suite)  # list[int]
        records = []
        valid = 0
        for world_index, world, rm in iter_matched_worlds(
            spec, suite_idx, cfg, roadmap_cfg, cfg.eval_worlds
        ):
            recs = P.run_world_arms(world, rm, providers, budgets, w_values, goal_idx=1)
            for r in recs:
                r["suite"] = suite
                r["world_index"] = world_index
            records.extend(recs)
            valid = world_index + 1
        if valid < cfg.eval_worlds:
            print(f"[c7] WARNING: {suite} under-filled: {valid}/{cfg.eval_worlds} worlds", flush=True)
        shard = ensure_dir(Path(out_dir) / "results" / "_shards" / "c7" / suite) / "shard_0000.csv"
        write_csv(shard, records)
        print(f"[c7] eval {suite}: {valid} worlds, {len(records)} arm-records -> {shard}", flush=True)

    # Merge into one raw CSV (no stats; that is Task 11). Bound the merge to EXACTLY
    # the suites this run evaluated — a glob would silently pull in stale shards from
    # prior runs with different --eval-suites (or an interrupted run), corrupting the
    # comparison.
    merged_rows: list = []
    for suite in parse_csv(cfg.eval_suites):
        shard_csv = Path(out_dir) / "results" / "_shards" / "c7" / suite / "shard_0000.csv"
        if shard_csv.exists():
            with open(shard_csv, newline="") as fh:
                merged_rows.extend(csv.DictReader(fh))
    merged_path = Path(out_dir) / "results" / "continuous_prm_c7_eval_raw.csv"
    write_csv(merged_path, merged_rows)
    print(f"[{now_str()}] C7 eval: merged {len(merged_rows)} rows -> {merged_path}", flush=True)
    return merged_path


# ---------------------------------------------------------------------------
# Calibrate (Task 10): per-suite binding-budget band selection (Gate 1)
# ---------------------------------------------------------------------------

# Fine budget grid swept by calibrate. Dense near ~144 (the steep Euclid success
# transition for the maze-family suites) AND near ~64 (the easy suites saturate
# Euclid early), then coarser up to 320 so the [0.45, 0.70] success targets are
# bracketed on both sides for every reasonable suite geometry.
CALIB_GRID = [24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 140, 152, 164, 176, 192, 224, 264, 320]


def _select_band_budgets(grid, euclid_success, k):
    """Pick k in-band budgets from `grid` by matching euclid success to targets.

    Targets are k points evenly spaced in [0.45, 0.70]. For each target, pick the
    grid budget whose euclid_success is closest (ties -> smaller budget, since
    `grid` is ascending). Dedupe preserving ascending budget order; if dedupe
    leaves < k, fill with the next-closest unused grid budgets (by distance to the
    nearest target). Returns a sorted ascending list of length min(k, len(grid)).
    """
    import numpy as np

    k = max(1, min(int(k), len(grid)))
    targets = np.linspace(0.45, 0.70, k)
    chosen: list = []
    for t in targets:
        # closest grid budget by |euclid_success - t|; ascending grid => smaller
        # budget wins ties because argmin returns the first minimum.
        best = int(np.argmin([abs(euclid_success[i] - t) for i in range(len(grid))]))
        if grid[best] not in chosen:
            chosen.append(grid[best])
    if len(chosen) < k:
        # rank unused grid budgets by distance to the NEAREST target, fill ascending.
        def nearest_target_dist(i):
            return min(abs(euclid_success[i] - t) for t in targets)
        remaining = sorted(
            (i for i in range(len(grid)) if grid[i] not in chosen),
            key=lambda i: (nearest_target_dist(i), grid[i]),
        )
        for i in remaining:
            if len(chosen) >= k:
                break
            chosen.append(grid[i])
    return sorted(chosen)


def run_calibrate(out_dir: Path, cfg: C7Config) -> Path:
    """Gate 1: sweep CALIB_GRID with ONLY Euclid + Oracle A* per suite and pick the
    binding-budget band. Writes out_dir/calibration.json (the per-suite "budgets"
    map T9's run_eval consumes) plus diagnostics.

    No learned models are loaded — euclid + oracle are pure geometry/graph, so this
    runs fast on CPU. Worlds come from iter_matched_worlds, so the budgets are
    calibrated on the SAME worlds eval will later score.
    """
    import numpy as np

    out_dir = Path(out_dir)
    M7.install_c7_hard_maps(cfg.sector_tokens)
    specs = C.build_anchor_specs()
    roadmap_cfg = C.RoadmapConfig(n_nodes=cfg.roadmap_nodes, k_neighbors=cfg.roadmap_k)
    euclid_provider = P.EuclidProvider()
    oracle_provider = P.OracleProvider()

    print(
        f"[{now_str()}] C7 calibrate: grid={CALIB_GRID} eval_worlds={cfg.eval_worlds} "
        f"budget_grid_size={cfg.budget_grid_size} suites={cfg.eval_suites}",
        flush=True,
    )

    budgets_map: Dict[str, list] = {}
    measurements: Dict[str, list] = {}
    warnings: list = []

    for suite_idx, suite in enumerate(parse_csv(cfg.eval_suites)):
        if suite not in specs:
            raise KeyError(f"calibrate: unknown suite {suite!r}; have {sorted(specs)}")
        spec = specs[suite]

        # Per-budget tallies: found counts + sum of expansions over SOLVED instances
        # (so mean expansions are computed only where the method actually reached the
        # goal, which is what the euclid-vs-oracle expansion gap is about).
        euclid_found = [0] * len(CALIB_GRID)
        oracle_found = [0] * len(CALIB_GRID)
        euclid_exp_sum = [0.0] * len(CALIB_GRID)
        oracle_exp_sum = [0.0] * len(CALIB_GRID)
        n_worlds = 0
        for _, world, rm in iter_matched_worlds(spec, suite_idx, cfg, roadmap_cfg, cfg.eval_worlds):
            euclid_h = euclid_provider.node_h(world, rm, goal_idx=1)
            oracle_h = oracle_provider.node_h(world, rm, goal_idx=1)
            for bi, b in enumerate(CALIB_GRID):
                re = C.astar_search(rm.adj, euclid_h, int(b))
                if re["found"]:
                    euclid_found[bi] += 1
                    euclid_exp_sum[bi] += int(re["expansions"])
                ro = C.astar_search(rm.adj, oracle_h, int(b))
                if ro["found"]:
                    oracle_found[bi] += 1
                    oracle_exp_sum[bi] += int(ro["expansions"])
            n_worlds += 1

        if n_worlds < cfg.eval_worlds:
            print(
                f"[c7] WARNING: calibrate {suite} under-filled: {n_worlds}/{cfg.eval_worlds} worlds",
                flush=True,
            )
            warnings.append(f"{suite}: under-filled {n_worlds}/{cfg.eval_worlds} worlds")

        denom = max(1, n_worlds)
        euclid_success = [f / denom for f in euclid_found]
        oracle_success = [f / denom for f in oracle_found]
        # Mean expansions over SOLVED instances (None where that method solved 0).
        euclid_exp = [
            (euclid_exp_sum[bi] / euclid_found[bi]) if euclid_found[bi] > 0 else None
            for bi in range(len(CALIB_GRID))
        ]
        oracle_exp = [
            (oracle_exp_sum[bi] / oracle_found[bi]) if oracle_found[bi] > 0 else None
            for bi in range(len(CALIB_GRID))
        ]

        chosen = _select_band_budgets(CALIB_GRID, euclid_success, cfg.budget_grid_size)
        idx_of = {b: i for i, b in enumerate(CALIB_GRID)}
        chosen_eu = [round(euclid_success[idx_of[b]], 3) for b in chosen]
        chosen_eu_full = [euclid_success[idx_of[b]] for b in chosen]

        def _headroom_at(b):
            """1 - oracle_exp/euclid_exp on euclid-solved means; None if not computable."""
            i = idx_of[b]
            ee, oe = euclid_exp[i], oracle_exp[i]
            if ee is None or oe is None or ee <= 0:
                return None
            return 1.0 - (oe / ee)

        chosen_headroom = [_headroom_at(b) for b in chosen]
        chosen_eu_exp = [euclid_exp[idx_of[b]] for b in chosen]
        chosen_or_exp = [oracle_exp[idx_of[b]] for b in chosen]

        # Refined warnings (oracle saturating at 1.0 is EXPECTED for the exact
        # graph-Dijkstra oracle; the real signal is the expansion gap):
        #  (1) no chosen budget has euclid success in [0.35, 0.95] -> no usable
        #      difficulty point, AND
        #  (2) best chosen budget's headroom < 0.15 -> oracle barely cheaper than
        #      euclid, so little room for learned models to help.
        if not any(0.35 <= es <= 0.95 for es in chosen_eu_full):
            msg = (
                f"{suite}: no usable difficulty point (no chosen budget has euclid_success "
                f"in [0.35, 0.95]; euclid@chosen={chosen_eu})"
            )
            print(f"[c7] WARNING: {msg}", flush=True)
            warnings.append(msg)
        valid_headroom = [h for h in chosen_headroom if h is not None]
        best_headroom = max(valid_headroom) if valid_headroom else None
        if best_headroom is not None and best_headroom < 0.15:
            msg = (
                f"{suite}: low expansion headroom (best chosen headroom={best_headroom:.3f} "
                f"< 0.15; oracle barely cheaper than euclid)"
            )
            print(f"[c7] WARNING: {msg}", flush=True)
            warnings.append(msg)

        budgets_map[suite] = chosen
        measurements[suite] = [
            {"budget": int(b), "euclid": round(euclid_success[bi], 4),
             "oracle": round(oracle_success[bi], 4),
             "euclid_exp": (round(euclid_exp[bi], 2) if euclid_exp[bi] is not None else None),
             "oracle_exp": (round(oracle_exp[bi], 2) if oracle_exp[bi] is not None else None),
             "worlds": int(n_worlds)}
            for bi, b in enumerate(CALIB_GRID)
        ]

        def _fmt(vals, nd=2):
            return [(round(v, nd) if v is not None else None) for v in vals]

        print(
            f"[c7] calibrate {suite}: chosen={chosen} euclid@chosen={chosen_eu} "
            f"euclid_exp@chosen={_fmt(chosen_eu_exp)} oracle_exp@chosen={_fmt(chosen_or_exp)} "
            f"headroom@chosen={_fmt(chosen_headroom, 3)}",
            flush=True,
        )

    calib = {
        "budgets": budgets_map,
        "grid": CALIB_GRID,
        "measurements": measurements,
        "warnings": warnings,
    }
    calib_path = out_dir / "calibration.json"
    write_json(calib_path, calib)
    print(
        f"[{now_str()}] C7 calibrate: wrote {calib_path} "
        f"({len(budgets_map)} suites, {len(warnings)} warnings)",
        flush=True,
    )
    return calib_path


# ---------------------------------------------------------------------------
# Analyze (Task 11): summary CSV + significance MD + six pre-registered
# comparisons MD + figures
# ---------------------------------------------------------------------------

# An "arm" is (provider, mode, w). The euclid baseline arm is always astar.
EUCLID_BASELINE = ("euclid", "astar", None)
# Providers that are baseline/ceiling, not "learned arms" (used by comparison 5).
NON_LEARNED = {"euclid", "oracle"}
# In-distribution (trained) vs held-out (OOD) suites for comparison 6 (spec 6.5).
IN_DIST_SUITES = ("C_hard_maze", "C_hard_rooms", "C_hard_spiral")
HELD_OUT_SUITES = ("C_hard_maze_dense", "C_hard_bugtrap", "C_hard_rooms_large")


def _coerce_bool(v) -> bool:
    """Robustly coerce a CSV cell to bool (handles 'True'/'1'/'true'/True/1)."""
    return str(v).strip().lower() in ("1", "true", "yes")


def _coerce_float(v):
    """Coerce a CSV cell to float; '' / None / non-numeric -> nan."""
    if v is None:
        return float("nan")
    s = str(v).strip()
    if s == "" or s.lower() in ("none", "nan"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _coerce_w(v):
    """Coerce a w cell to float, or None for astar rows (blank/None)."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fmt_w(w) -> str:
    return "" if w is None else f"{float(w):g}"


def _fmt_num(v, nd: int = 3) -> str:
    """Format a possibly-None/nan float for markdown."""
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    import math as _m
    if not _m.isfinite(f):
        return "n/a"
    return f"{f:.{nd}f}"


def load_raw_rows(out_dir: Path):
    """Read continuous_prm_c7_eval_raw.csv and coerce CSV strings to typed fields.

    Returns a list of dicts with: suite, world_index(int), provider, mode,
    w(float|None), budget(int), found(bool), expansions(int), cost(float),
    optimal(float), suboptimality(float), nonfinite(bool), closed(int).
    """
    raw_path = Path(out_dir) / "results" / "continuous_prm_c7_eval_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"analyze: missing raw eval CSV {raw_path}; run --mode eval first"
        )
    rows = read_csv(raw_path)
    out = []
    for r in rows:
        out.append(
            {
                "suite": str(r.get("suite", "")),
                "world_index": int(float(r.get("world_index", 0))),
                "provider": str(r.get("provider", "")),
                "mode": str(r.get("mode", "")),
                "w": _coerce_w(r.get("w")),
                "budget": int(float(r.get("budget", 0))),
                "found": _coerce_bool(r.get("found")),
                "expansions": int(float(r.get("expansions", 0) or 0)),
                "cost": _coerce_float(r.get("cost")),
                "optimal": _coerce_float(r.get("optimal")),
                "suboptimality": _coerce_float(r.get("suboptimality")),
                "nonfinite": _coerce_bool(r.get("nonfinite")),
                "closed": int(float(r.get("closed", 0) or 0)),
            }
        )
    return out


def _binding_budget(suite: str, suite_budgets, calib_budgets) -> int:
    """Pick the binding budget for a suite: the LOWER of the suite's calibrated
    budgets if calibration.json has them, else the first budget seen in the data."""
    raw = calib_budgets.get(suite)
    if raw:
        try:
            return int(min(int(b) for b in raw))
        except (TypeError, ValueError):
            pass
    seen = sorted(suite_budgets.get(suite, []))
    return int(seen[0]) if seen else 0


def _load_calib_budgets(out_dir: Path):
    """Return {suite: [budgets]} from out_dir/calibration.json, or {} if absent."""
    import json
    calib_path = Path(out_dir) / "calibration.json"
    if not calib_path.exists():
        return {}
    try:
        calib = json.loads(calib_path.read_text())
        return dict(calib.get("budgets", {}) or {})
    except (ValueError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Stats helpers (no new heavy deps; scipy optional)
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank_p(diffs):
    """Paired Wilcoxon signed-rank two-sided p-value over `diffs` (arm - euclid,
    or any paired difference). Uses scipy.stats.wilcoxon if importable; otherwise
    a normal-approximation with tie/zero correction. Returns nan when undefined
    (e.g. < 1 nonzero diff). The caller pairs the inputs."""
    import numpy as np

    d = np.asarray([x for x in diffs if np.isfinite(x)], dtype=np.float64)
    nz = d[d != 0.0]
    if nz.size < 1:
        return float("nan")
    # Prefer scipy when available (exact for small n, continuity-corrected normal otherwise).
    try:
        from scipy.stats import wilcoxon  # type: ignore
        # zero_method="wilcox" drops zeros (matches our manual fallback).
        res = wilcoxon(d, zero_method="wilcox", alternative="two-sided", mode="auto")
        return float(res.pvalue)
    except Exception:
        pass
    # Manual normal approximation with tie correction.
    n = nz.size
    ranks = _rankdata_average(np.abs(nz))
    w_plus = float(np.sum(ranks[nz > 0]))
    w_minus = float(np.sum(ranks[nz < 0]))
    t = min(w_plus, w_minus)
    mean_t = n * (n + 1) / 4.0
    # tie correction term
    _, counts = np.unique(np.abs(nz), return_counts=True)
    tie_term = float(np.sum(counts ** 3 - counts))
    var_t = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_t <= 0:
        return float("nan")
    # continuity correction
    z = (t - mean_t + 0.5) / (var_t ** 0.5)
    # two-sided p from standard normal CDF
    p = 2.0 * 0.5 * (1.0 + _erf(-abs(z) / (2.0 ** 0.5)))
    return float(min(1.0, max(0.0, p)))


def _rankdata_average(x):
    """Average ranks (1-based) with ties averaged — like scipy.stats.rankdata."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _erf(x: float) -> float:
    import math
    return math.erf(x)


def bootstrap_median_ci(values, seed: int, n_boot: int = 1000, lo: float = 2.5, hi: float = 97.5):
    """Seeded bootstrap percentile CI on the median of `values`.

    Resamples with replacement n_boot times using np.random.default_rng(seed),
    returns (median, ci_lo, ci_hi). nan triple if no finite values."""
    import numpy as np
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    med = float(np.median(arr))
    if arr.size == 1:
        return (med, med, med)
    rng = np.random.default_rng(int(seed))
    boot = np.empty(n_boot, dtype=np.float64)
    idx_n = arr.size
    for i in range(n_boot):
        sample = arr[rng.integers(0, idx_n, size=idx_n)]
        boot[i] = np.median(sample)
    return (med, float(np.percentile(boot, lo)), float(np.percentile(boot, hi)))


# ---------------------------------------------------------------------------
# Indexing helpers over the typed raw rows
# ---------------------------------------------------------------------------

def _index_rows(rows):
    """Index typed raw rows two ways:
      arms[suite][budget] = sorted list of distinct (provider, mode, w) arms.
      lookup[(suite, budget, provider, mode, w, world_index)] = row.
    Also returns the set of suites and per-suite sorted budget list.
    """
    arms: Dict = {}
    lookup: Dict = {}
    suite_budgets: Dict = {}
    for r in rows:
        suite, budget = r["suite"], r["budget"]
        arm = (r["provider"], r["mode"], r["w"])
        arms.setdefault(suite, {}).setdefault(budget, set()).add(arm)
        lookup[(suite, budget, r["provider"], r["mode"], r["w"], r["world_index"])] = r
        suite_budgets.setdefault(suite, set()).add(budget)
    arms_sorted = {
        s: {b: sorted(v, key=lambda a: (a[0], a[1], (a[2] if a[2] is not None else -1.0)))
            for b, v in bd.items()}
        for s, bd in arms.items()
    }
    suite_budgets_sorted = {s: sorted(bs) for s, bs in suite_budgets.items()}
    return arms_sorted, lookup, suite_budgets_sorted


def _worlds_for(rows, suite, budget, provider, mode, w):
    """Set of world_index where this arm has a row at (suite, budget)."""
    out = set()
    for r in rows:
        if (r["suite"] == suite and r["budget"] == budget and r["provider"] == provider
                and r["mode"] == mode and r["w"] == w):
            out.add(r["world_index"])
    return out


def _arm_rows_by_world(rows, suite, budget, provider, mode, w):
    """Map world_index -> row for one arm at (suite, budget)."""
    out = {}
    for r in rows:
        if (r["suite"] == suite and r["budget"] == budget and r["provider"] == provider
                and r["mode"] == mode and r["w"] == w):
            out[r["world_index"]] = r
    return out


# ---------------------------------------------------------------------------
# 1. Summary CSV
# ---------------------------------------------------------------------------

def build_summary(rows, cfg: C7Config):
    """One row per (suite, provider, mode, w, budget). See plan output #1.

    NOTE: the per-node Spearman-to-oracle diagnostic is intentionally OMITTED here
    — analyze only has per-instance results (no per-node h), so a node-level
    Spearman would require re-running the providers (out of scope for a CSV-only
    analyze). It remains available in the eval/diagnostic path.
    """
    import numpy as np

    # Group rows by arm key.
    groups: Dict = {}
    for r in rows:
        key = (r["suite"], r["provider"], r["mode"], r["w"], r["budget"])
        groups.setdefault(key, []).append(r)

    # Precompute euclid-astar expansions per (suite, budget, world) for the ratio.
    euclid_exp: Dict = {}
    for r in rows:
        if r["provider"] == "euclid" and r["mode"] == "astar":
            if r["found"]:
                euclid_exp[(r["suite"], r["budget"], r["world_index"])] = float(r["expansions"])

    out = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], (k[3] if k[3] is not None else -1.0), k[4])):
        suite, provider, mode, w, budget = key
        grp = groups[key]
        n = len({r["world_index"] for r in grp})
        solved = [r for r in grp if r["found"]]
        succ = (len(solved) / n) if n else float("nan")
        exps = [float(r["expansions"]) for r in solved]
        subs = [float(r["suboptimality"]) for r in solved if np.isfinite(r["suboptimality"])]
        # exp ratio vs euclid over matched-solved (both euclid-astar AND this arm solved).
        ratios = []
        for r in solved:
            eu = euclid_exp.get((suite, budget, r["world_index"]))
            if eu is not None and eu > 0:
                ratios.append(float(r["expansions"]) / eu)
        out.append(
            {
                "suite": suite,
                "provider": provider,
                "mode": mode,
                "w": _fmt_w(w),
                "budget": int(budget),
                "n": int(n),
                "success_rate": round(succ, 4) if np.isfinite(succ) else "",
                "expansions_mean": round(float(np.mean(exps)), 3) if exps else "",
                "expansions_median": round(float(np.median(exps)), 3) if exps else "",
                "suboptimality_mean": round(float(np.mean(subs)), 5) if subs else "",
                "suboptimality_p95": round(float(np.percentile(subs, 95)), 5) if subs else "",
                "exp_ratio_vs_euclid_median": round(float(np.median(ratios)), 4) if ratios else "",
                "matched_solved_n": int(len(ratios)),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 2. Significance MD (McNemar/BH success + Wilcoxon/bootstrap expansions)
# ---------------------------------------------------------------------------

def _success_significance(rows, suite_budgets):
    """McNemar paired (arm-astar vs euclid-astar) on SUCCESS per (suite, budget),
    BH-corrected across the whole grid. gain = arm found & euclid not; loss = euclid
    found & arm not, paired over ALL shared worlds. Returns list of dict rows."""
    import numpy as np

    comparisons = []
    pvals = []
    for suite in sorted(suite_budgets):
        for budget in suite_budgets[suite]:
            eu_by_world = _arm_rows_by_world(rows, suite, budget, "euclid", "astar", None)
            if not eu_by_world:
                continue
            # all non-euclid arms present at this (suite, budget), any mode/w.
            arm_keys = sorted(
                {(r["provider"], r["mode"], r["w"]) for r in rows
                 if r["suite"] == suite and r["budget"] == budget and r["provider"] != "euclid"},
                key=lambda a: (a[0], a[1], (a[2] if a[2] is not None else -1.0)),
            )
            for (provider, mode, w) in arm_keys:
                arm_by_world = _arm_rows_by_world(rows, suite, budget, provider, mode, w)
                shared = sorted(set(eu_by_world) & set(arm_by_world))
                if not shared:
                    continue
                gain = sum(1 for wi in shared if arm_by_world[wi]["found"] and not eu_by_world[wi]["found"])
                loss = sum(1 for wi in shared if eu_by_world[wi]["found"] and not arm_by_world[wi]["found"])
                eu_succ = float(np.mean([1.0 if eu_by_world[wi]["found"] else 0.0 for wi in shared]))
                arm_succ = float(np.mean([1.0 if arm_by_world[wi]["found"] else 0.0 for wi in shared]))
                p = mcnemar_exact_p(gain, loss)
                pvals.append(p)
                comparisons.append(
                    {
                        "suite": suite, "budget": int(budget),
                        "provider": provider, "mode": mode, "w": w,
                        "n": len(shared),
                        "euclid_success": eu_succ, "arm_success": arm_succ,
                        "delta": arm_succ - eu_succ,
                        "gain": gain, "loss": loss, "mcnemar_p": p,
                    }
                )
    qvals = bh_q_values(pvals)
    for row, q in zip(comparisons, qvals):
        row["bh_q"] = q
    return comparisons


def _expansion_significance(rows, suite_budgets, cfg: C7Config):
    """For each non-euclid arm vs euclid-astar at (suite, budget): on the matched
    set (worlds euclid-astar solved AND the arm solved), report median exp_ratio,
    paired Wilcoxon signed-rank p (arm_exp vs euclid_exp), and bootstrap 95% CI on
    the median ratio. Returns list of dict rows."""
    import numpy as np

    out = []
    for suite in sorted(suite_budgets):
        for budget in suite_budgets[suite]:
            eu_by_world = _arm_rows_by_world(rows, suite, budget, "euclid", "astar", None)
            if not eu_by_world:
                continue
            arm_keys = sorted(
                {(r["provider"], r["mode"], r["w"]) for r in rows
                 if r["suite"] == suite and r["budget"] == budget and r["provider"] != "euclid"},
                key=lambda a: (a[0], a[1], (a[2] if a[2] is not None else -1.0)),
            )
            for (provider, mode, w) in arm_keys:
                arm_by_world = _arm_rows_by_world(rows, suite, budget, provider, mode, w)
                # matched set: euclid solved AND arm solved.
                matched = sorted(
                    wi for wi in (set(eu_by_world) & set(arm_by_world))
                    if eu_by_world[wi]["found"] and arm_by_world[wi]["found"]
                    and float(eu_by_world[wi]["expansions"]) > 0
                )
                if not matched:
                    out.append(
                        {"suite": suite, "budget": int(budget), "provider": provider,
                         "mode": mode, "w": w, "n_matched": 0,
                         "median_ratio": None, "wilcoxon_p": float("nan"),
                         "ci_lo": None, "ci_hi": None}
                    )
                    continue
                ratios = [float(arm_by_world[wi]["expansions"]) / float(eu_by_world[wi]["expansions"])
                          for wi in matched]
                diffs = [float(arm_by_world[wi]["expansions"]) - float(eu_by_world[wi]["expansions"])
                         for wi in matched]
                wp = wilcoxon_signed_rank_p(diffs)
                med, ci_lo, ci_hi = bootstrap_median_ci(ratios, seed=int(cfg.seed))
                out.append(
                    {"suite": suite, "budget": int(budget), "provider": provider,
                     "mode": mode, "w": w, "n_matched": len(matched),
                     "median_ratio": med, "wilcoxon_p": wp,
                     "ci_lo": ci_lo, "ci_hi": ci_hi}
                )
    return out


def write_significance_md(out_dir: Path, success_rows, expansion_rows):
    def arm_label(r):
        base = f"{r['provider']}/{r['mode']}"
        return base + (f"/w={_fmt_w(r['w'])}" if r["w"] is not None else "")

    lines = ["# C7 Integration Comparison — Significance", ""]
    lines += [
        "## Success: McNemar (arm vs euclid/astar), BH-corrected across the grid",
        "",
        "|Suite|Budget|Arm|n|Euclid succ|Arm succ|Delta|Gain|Loss|McNemar p|BH q|",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not success_rows:
        lines.append("|<none>|-|-|-|-|-|-|-|-|-|-|")
    for r in sorted(success_rows, key=lambda x: (x["suite"], x["budget"], arm_label(x))):
        lines.append(
            f"|{r['suite']}|{r['budget']}|{arm_label(r)}|{r['n']}|"
            f"{_fmt_num(r['euclid_success'])}|{_fmt_num(r['arm_success'])}|{_fmt_num(r['delta'])}|"
            f"{r['gain']}|{r['loss']}|{_fmt_num(r['mcnemar_p'])}|{_fmt_num(r['bh_q'])}|"
        )
    lines += [
        "",
        "## Expansions: matched-set median ratio (arm/euclid) + Wilcoxon p + bootstrap 95% CI",
        "",
        "|Suite|Budget|Arm|n matched|Median ratio|95% CI|Wilcoxon p|",
        "|---|---:|---|---:|---:|---|---:|",
    ]
    if not expansion_rows:
        lines.append("|<none>|-|-|-|-|-|-|")
    for r in sorted(expansion_rows, key=lambda x: (x["suite"], x["budget"], arm_label(x))):
        ci = (f"[{_fmt_num(r['ci_lo'])}, {_fmt_num(r['ci_hi'])}]"
              if r["ci_lo"] is not None else "n/a")
        lines.append(
            f"|{r['suite']}|{r['budget']}|{arm_label(r)}|{r['n_matched']}|"
            f"{_fmt_num(r['median_ratio'])}|{ci}|{_fmt_num(r['wilcoxon_p'])}|"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- McNemar pairs each arm against `euclid/astar` on success over shared worlds;",
        "  gain = arm found & euclid not, loss = euclid found & arm not. BH q corrects across",
        "  all arm-vs-euclid success comparisons in this grid.",
        "- The expansion ratio uses the *matched set* (worlds euclid AND the arm both solved).",
        "  Median ratio < 1 means the arm expands fewer nodes than euclid. The Wilcoxon p tests",
        "  paired (arm_exp - euclid_exp); the bootstrap CI is a seeded percentile CI on the median ratio.",
    ]
    md_path = Path(out_dir) / "results" / "continuous_prm_c7_significance.md"
    ensure_dir(md_path.parent)
    # Force UTF-8 (Windows write_text defaults to cp1252, which mangles em-dashes).
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# 3. Pre-registered comparisons MD (the six from spec 6.9)
# ---------------------------------------------------------------------------

def _matched_ratio_and_success(rows, suite, budget, provider, mode, w, cfg):
    """Helper: for one arm vs euclid/astar at (suite, budget), return a dict with
    median exp_ratio (matched), wilcoxon p, CI, success delta, McNemar p, and n.
    Returns None if euclid is absent at this (suite, budget) or the arm has no rows."""
    import numpy as np
    eu_by_world = _arm_rows_by_world(rows, suite, budget, "euclid", "astar", None)
    arm_by_world = _arm_rows_by_world(rows, suite, budget, provider, mode, w)
    if not eu_by_world or not arm_by_world:
        return None
    shared = sorted(set(eu_by_world) & set(arm_by_world))
    if not shared:
        return None
    gain = sum(1 for wi in shared if arm_by_world[wi]["found"] and not eu_by_world[wi]["found"])
    loss = sum(1 for wi in shared if eu_by_world[wi]["found"] and not arm_by_world[wi]["found"])
    eu_succ = float(np.mean([1.0 if eu_by_world[wi]["found"] else 0.0 for wi in shared]))
    arm_succ = float(np.mean([1.0 if arm_by_world[wi]["found"] else 0.0 for wi in shared]))
    mp = mcnemar_exact_p(gain, loss)
    matched = sorted(
        wi for wi in shared
        if eu_by_world[wi]["found"] and arm_by_world[wi]["found"]
        and float(eu_by_world[wi]["expansions"]) > 0
    )
    if matched:
        ratios = [float(arm_by_world[wi]["expansions"]) / float(eu_by_world[wi]["expansions"]) for wi in matched]
        diffs = [float(arm_by_world[wi]["expansions"]) - float(eu_by_world[wi]["expansions"]) for wi in matched]
        med, ci_lo, ci_hi = bootstrap_median_ci(ratios, seed=int(cfg.seed))
        wp = wilcoxon_signed_rank_p(diffs)
    else:
        med = ci_lo = ci_hi = None
        wp = float("nan")
    return {
        "n": len(shared), "n_matched": len(matched),
        "euclid_success": eu_succ, "arm_success": arm_succ, "success_delta": arm_succ - eu_succ,
        "gain": gain, "loss": loss, "mcnemar_p": mp,
        "median_ratio": med, "ci_lo": ci_lo, "ci_hi": ci_hi, "wilcoxon_p": wp,
    }


def _best_focal_w(rows, suite, budget, provider, cfg):
    """For a provider's focal arms at (suite, budget), pick the w with the lowest
    matched-median exp_ratio vs euclid. Returns (w, stats) or (None, None)."""
    ws = sorted({r["w"] for r in rows
                 if r["suite"] == suite and r["budget"] == budget
                 and r["provider"] == provider and r["mode"] == "focal" and r["w"] is not None})
    best = None
    for w in ws:
        st = _matched_ratio_and_success(rows, suite, budget, provider, "focal", w, cfg)
        if st is None or st["median_ratio"] is None:
            continue
        if best is None or st["median_ratio"] < best[1]["median_ratio"]:
            best = (w, st)
    return best if best is not None else (None, None)


def _present_arms(rows):
    """Set of providers present in the data."""
    return {r["provider"] for r in rows}


def write_preregistered_md(out_dir: Path, rows, cfg: C7Config, binding):
    """Write the six pre-registered comparison sections. `binding` maps suite ->
    binding budget. Each section degrades gracefully (notes when an arm/suite is
    absent) instead of crashing."""
    present = _present_arms(rows)
    suites = sorted(binding)
    lines = ["# C7 Integration Comparison — Pre-registered Comparisons", ""]
    lines.append(
        "Binding budget per suite (lower of the calibrated band, else first budget seen): "
        + ", ".join(f"{s}={binding[s]}" for s in suites)
    )
    lines.append("")

    def ratio_cell(st):
        if st is None or st.get("median_ratio") is None:
            return "n/a"
        ci = f"[{_fmt_num(st['ci_lo'])}, {_fmt_num(st['ci_hi'])}]"
        return f"{_fmt_num(st['median_ratio'])} {ci}"

    # --- Comparison 1: field_hrm/astar vs euclid/astar ----------------------
    lines += ["## 1. field_hrm/astar vs euclid/astar (success + expansions), per suite", ""]
    if "field_hrm" not in present:
        lines += ["_field_hrm absent — skipped._", ""]
    else:
        lines += [
            "|Suite|Budget|n|Euclid succ|Arm succ|Succ delta|McNemar p|n matched|Median ratio (95% CI)|Wilcoxon p|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
        for s in suites:
            b = binding[s]
            st = _matched_ratio_and_success(rows, s, b, "field_hrm", "astar", None, cfg)
            if st is None:
                lines.append(f"|{s}|{b}|_arm/euclid absent_|||||||")
                continue
            lines.append(
                f"|{s}|{b}|{st['n']}|{_fmt_num(st['euclid_success'])}|{_fmt_num(st['arm_success'])}|"
                f"{_fmt_num(st['success_delta'])}|{_fmt_num(st['mcnemar_p'])}|{st['n_matched']}|"
                f"{ratio_cell(st)}|{_fmt_num(st['wilcoxon_p'])}|"
            )
        lines.append("")

    # --- Comparison 2: scalar_hrm/astar vs field_hrm/astar (representation) --
    lines += ["## 2. scalar_hrm/astar vs field_hrm/astar — the representation lever", ""]
    if "scalar_hrm" not in present and "field_hrm" not in present:
        lines += ["_neither scalar_hrm nor field_hrm present — skipped._", ""]
    else:
        lines += [
            "Each arm's expansion ratio + success vs euclid, side by side (does the field beat the scalar?).",
            "",
            "|Suite|Budget|Arm|Arm succ vs euclid (delta)|n matched|Median ratio (95% CI)|Wilcoxon p|",
            "|---|---:|---|---:|---:|---|---:|",
        ]
        for s in suites:
            b = binding[s]
            for prov in ("scalar_hrm", "field_hrm"):
                if prov not in present:
                    lines.append(f"|{s}|{b}|{prov}|_absent_||||")
                    continue
                st = _matched_ratio_and_success(rows, s, b, prov, "astar", None, cfg)
                if st is None:
                    lines.append(f"|{s}|{b}|{prov}|_euclid/arm absent_||||")
                    continue
                lines.append(
                    f"|{s}|{b}|{prov}|{_fmt_num(st['arm_success'])} ({_fmt_num(st['success_delta'])})|"
                    f"{st['n_matched']}|{ratio_cell(st)}|{_fmt_num(st['wilcoxon_p'])}|"
                )
        lines.append("")

    # --- Comparison 3: scalar_hrm/focal(best w) vs scalar_hrm/astar ----------
    lines += ["## 3. scalar_hrm/focal (best w) vs scalar_hrm/astar — the integration lever", ""]
    if "scalar_hrm" not in present:
        lines += ["_scalar_hrm absent — skipped._", ""]
    else:
        lines += [
            "Both expressed as median exp_ratio vs euclid on their matched sets (does focal help the scalar?).",
            "",
            "|Suite|Budget|scalar/astar ratio (CI)|best w|scalar/focal ratio (CI)|focal Wilcoxon p|",
            "|---|---:|---|---:|---|---:|",
        ]
        for s in suites:
            b = binding[s]
            st_a = _matched_ratio_and_success(rows, s, b, "scalar_hrm", "astar", None, cfg)
            bw, st_f = _best_focal_w(rows, s, b, "scalar_hrm", cfg)
            lines.append(
                f"|{s}|{b}|{ratio_cell(st_a)}|{_fmt_w(bw) if bw is not None else 'n/a'}|"
                f"{ratio_cell(st_f)}|{_fmt_num(st_f['wilcoxon_p']) if st_f else 'n/a'}|"
            )
        lines.append("")

    # --- Comparison 4: field_*/focal vs field_*/astar -----------------------
    lines += ["## 4. field_*/focal (best w) vs field_*/astar — focal vs additive on the field models", ""]
    field_provs = sorted(p for p in present if p.startswith("field_"))
    if not field_provs:
        lines += ["_no field_* arms present — skipped._", ""]
    else:
        lines += [
            "|Suite|Budget|Provider|astar ratio (CI)|best w|focal ratio (CI)|focal Wilcoxon p|",
            "|---|---:|---|---|---:|---|---:|",
        ]
        for s in suites:
            b = binding[s]
            for prov in field_provs:
                st_a = _matched_ratio_and_success(rows, s, b, prov, "astar", None, cfg)
                bw, st_f = _best_focal_w(rows, s, b, prov, cfg)
                lines.append(
                    f"|{s}|{b}|{prov}|{ratio_cell(st_a)}|{_fmt_w(bw) if bw is not None else 'n/a'}|"
                    f"{ratio_cell(st_f)}|{_fmt_num(st_f['wilcoxon_p']) if st_f else 'n/a'}|"
                )
        lines.append("")

    # --- Comparison 5: each learned arm vs oracle/astar (gap-to-ceiling) -----
    lines += ["## 5. Each learned arm vs oracle/astar — gap-to-ceiling", ""]
    if "oracle" not in present:
        lines += ["_oracle absent — skipped._", ""]
    else:
        lines += [
            "Median over the triple-matched set (euclid, oracle, arm all solved) of",
            "`(arm_exp - oracle_exp) / (euclid_exp - oracle_exp)` — the fraction of the",
            "euclid->oracle expansion gap left *uncaptured* (0 = matches oracle, 1 = no better than euclid).",
            "",
            "|Suite|Budget|Arm|n triple-matched|Median uncaptured-gap fraction|",
            "|---|---:|---|---:|---:|",
        ]
        learned = sorted(p for p in present if p not in NON_LEARNED)
        any_row = False
        for s in suites:
            b = binding[s]
            eu_by_world = _arm_rows_by_world(rows, s, b, "euclid", "astar", None)
            or_by_world = _arm_rows_by_world(rows, s, b, "oracle", "astar", None)
            for prov in learned:
                # astar arm only (direct integration) for the gap-to-ceiling.
                arm_by_world = _arm_rows_by_world(rows, s, b, prov, "astar", None)
                fracs = _gap_to_ceiling_fracs(eu_by_world, or_by_world, arm_by_world)
                if not fracs:
                    lines.append(f"|{s}|{b}|{prov}/astar|0|n/a|")
                    continue
                import numpy as np
                lines.append(f"|{s}|{b}|{prov}/astar|{len(fracs)}|{_fmt_num(float(np.median(fracs)))}|")
                any_row = True
        if not any_row:
            lines.append("|<none>|-|-|-|-|")
        lines.append("")

    # --- Comparison 6: in-distribution vs held-out (field_hrm) ---------------
    lines += ["## 6. In-distribution vs held-out — field_hrm exp_ratio + success vs euclid", ""]
    if "field_hrm" not in present:
        lines += ["_field_hrm absent — skipped._", ""]
    else:
        lines += [
            f"In-distribution (trained): {', '.join(IN_DIST_SUITES)}.  "
            f"Held-out (OOD): {', '.join(HELD_OUT_SUITES)}.",
            "",
            "|Group|Suite|Budget|n matched|Median ratio (95% CI)|Succ delta vs euclid|",
            "|---|---|---:|---:|---|---:|",
        ]
        for group_name, group in (("in-dist", IN_DIST_SUITES), ("held-out", HELD_OUT_SUITES)):
            present_in_group = [s for s in group if s in binding]
            if not present_in_group:
                lines.append(f"|{group_name}|_no suites in this run_|-|-|-|-|")
                continue
            for s in present_in_group:
                b = binding[s]
                st = _matched_ratio_and_success(rows, s, b, "field_hrm", "astar", None, cfg)
                if st is None:
                    lines.append(f"|{group_name}|{s}|{b}|_arm/euclid absent_|-|-|")
                    continue
                lines.append(
                    f"|{group_name}|{s}|{b}|{st['n_matched']}|{ratio_cell(st)}|"
                    f"{_fmt_num(st['success_delta'])}|"
                )
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Each comparison uses the per-suite binding budget. Arms or suites absent from this run",
        "  are skipped with a note rather than crashing.",
        "- Comparisons 3 and 4 pick the focal `w` with the lowest matched-median exp_ratio vs euclid.",
        "- Bootstrap CIs are seeded (`np.random.default_rng(cfg.seed)`), so this analysis is reproducible.",
    ]
    md_path = Path(out_dir) / "results" / "continuous_prm_c7_preregistered.md"
    ensure_dir(md_path.parent)
    # Force UTF-8 (Windows write_text defaults to cp1252, which mangles em-dashes).
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def _gap_to_ceiling_fracs(eu_by_world, or_by_world, arm_by_world):
    """List of (arm_exp - oracle_exp)/(euclid_exp - oracle_exp) over worlds where
    euclid, oracle, and arm all solved and the euclid->oracle gap is positive."""
    fracs = []
    shared = set(eu_by_world) & set(or_by_world) & set(arm_by_world)
    for wi in shared:
        e, o, a = eu_by_world[wi], or_by_world[wi], arm_by_world[wi]
        if not (e["found"] and o["found"] and a["found"]):
            continue
        eu_exp = float(e["expansions"]); or_exp = float(o["expansions"]); arm_exp = float(a["expansions"])
        denom = eu_exp - or_exp
        if denom <= 0:
            continue
        fracs.append((arm_exp - or_exp) / denom)
    return fracs


# ---------------------------------------------------------------------------
# 4. Figures (guarded, like C6's maybe_make_figures)
# ---------------------------------------------------------------------------

def maybe_make_figures(out_dir: Path, rows, cfg: C7Config, binding):
    """Write three simple figures if matplotlib is importable; skip silently
    otherwise. (a) expansion-ratio-vs-euclid bars per suite at binding budget;
    (b) gap-to-oracle (arm vs oracle expansions); (c) suboptimality-vs-w for
    focal arms."""
    if not cfg.make_figures:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[c7] analyze: matplotlib unavailable; skipping figures", flush=True)
        return []
    import numpy as np

    fig_dir = ensure_dir(Path(out_dir) / "figures")
    written = []
    suites = sorted(binding)
    present = _present_arms(rows)
    learned = sorted(p for p in present if p not in NON_LEARNED)

    # (a) expansion-ratio bars per suite at the binding budget (astar arms).
    try:
        for s in suites:
            b = binding[s]
            labels, vals = [], []
            for prov in learned:
                st = _matched_ratio_and_success(rows, s, b, prov, "astar", None, cfg)
                if st is not None and st["median_ratio"] is not None:
                    labels.append(prov)
                    vals.append(st["median_ratio"])
            if not labels:
                continue
            plt.figure(figsize=(6, 3.5))
            plt.bar(range(len(labels)), vals, color="steelblue")
            plt.axhline(1.0, color="k", linestyle="--", linewidth=1)
            plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
            plt.ylabel("median exp ratio vs euclid")
            plt.title(f"{s} @ budget {b}")
            plt.tight_layout()
            p = fig_dir / f"exp_ratio__{sanitize_name(s)}.png"
            plt.savefig(p, dpi=130); plt.close()
            written.append(p)
    except Exception as exc:  # pragma: no cover - figure robustness
        print(f"[c7] analyze: figure (a) failed: {exc}", flush=True)

    # (b) gap-to-oracle: arm vs oracle median expansions per suite at binding budget.
    try:
        if "oracle" in present:
            for s in suites:
                b = binding[s]
                or_by_world = _arm_rows_by_world(rows, s, b, "oracle", "astar", None)
                or_exps = [float(r["expansions"]) for r in or_by_world.values() if r["found"]]
                or_med = float(np.median(or_exps)) if or_exps else float("nan")
                labels, vals = [], []
                for prov in learned:
                    arm_by_world = _arm_rows_by_world(rows, s, b, prov, "astar", None)
                    exps = [float(r["expansions"]) for r in arm_by_world.values() if r["found"]]
                    if exps:
                        labels.append(prov); vals.append(float(np.median(exps)))
                if not labels:
                    continue
                plt.figure(figsize=(6, 3.5))
                plt.bar(range(len(labels)), vals, color="indianred")
                if np.isfinite(or_med):
                    plt.axhline(or_med, color="green", linestyle="--", linewidth=1.2, label=f"oracle={or_med:.0f}")
                    plt.legend()
                plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
                plt.ylabel("median expansions (solved)")
                plt.title(f"gap-to-oracle: {s} @ budget {b}")
                plt.tight_layout()
                p = fig_dir / f"gap_to_oracle__{sanitize_name(s)}.png"
                plt.savefig(p, dpi=130); plt.close()
                written.append(p)
    except Exception as exc:  # pragma: no cover
        print(f"[c7] analyze: figure (b) failed: {exc}", flush=True)

    # (c) suboptimality-vs-w curve for focal arms (pooled over suites at binding budget).
    try:
        focal_provs = sorted({r["provider"] for r in rows if r["mode"] == "focal"})
        plotted = False
        plt.figure(figsize=(6, 3.5))
        for prov in focal_provs:
            ws = sorted({r["w"] for r in rows if r["provider"] == prov and r["mode"] == "focal" and r["w"] is not None})
            xs, ys = [], []
            for w in ws:
                subs = []
                for s in suites:
                    b = binding[s]
                    for r in rows:
                        if (r["suite"] == s and r["budget"] == b and r["provider"] == prov
                                and r["mode"] == "focal" and r["w"] == w and r["found"]
                                and np.isfinite(r["suboptimality"])):
                            subs.append(float(r["suboptimality"]))
                if subs:
                    xs.append(w); ys.append(float(np.mean(subs)))
            if xs:
                plt.plot(xs, ys, marker="o", label=prov)
                plotted = True
        if plotted:
            plt.xlabel("focal w"); plt.ylabel("mean suboptimality (solved)")
            plt.title("suboptimality vs focal w (binding budgets)")
            plt.legend(fontsize=8)
            plt.tight_layout()
            p = fig_dir / "suboptimality_vs_w.png"
            plt.savefig(p, dpi=130); plt.close()
            written.append(p)
        else:
            plt.close()
    except Exception as exc:  # pragma: no cover
        print(f"[c7] analyze: figure (c) failed: {exc}", flush=True)

    if written:
        print(f"[c7] analyze: wrote {len(written)} figures -> {fig_dir}", flush=True)
    return written


# ---------------------------------------------------------------------------
# Analyze orchestrator
# ---------------------------------------------------------------------------

def run_analyze(out_dir: Path, cfg: C7Config) -> Dict[str, Path]:
    """Read the raw eval CSV and write summary CSV, significance MD, pre-registered
    MD, and figures. Returns the paths written."""
    out_dir = Path(out_dir)
    rows = load_raw_rows(out_dir)
    print(f"[{now_str()}] C7 analyze: loaded {len(rows)} raw rows", flush=True)

    _, _, suite_budgets = _index_rows(rows)
    calib_budgets = _load_calib_budgets(out_dir)
    binding = {s: _binding_budget(s, suite_budgets, calib_budgets) for s in suite_budgets}

    # 1. Summary CSV
    summary = build_summary(rows, cfg)
    summary_path = Path(out_dir) / "results" / "continuous_prm_c7_eval_summary.csv"
    write_csv(summary_path, summary)
    print(f"[{now_str()}] C7 analyze: wrote summary ({len(summary)} arm-rows) -> {summary_path}", flush=True)

    # 2. Significance MD
    success_rows = _success_significance(rows, suite_budgets)
    expansion_rows = _expansion_significance(rows, suite_budgets, cfg)
    sig_path = write_significance_md(out_dir, success_rows, expansion_rows)
    print(f"[{now_str()}] C7 analyze: wrote significance -> {sig_path}", flush=True)

    # 3. Pre-registered comparisons MD
    prereg_path = write_preregistered_md(out_dir, rows, cfg, binding)
    print(f"[{now_str()}] C7 analyze: wrote pre-registered comparisons -> {prereg_path}", flush=True)

    # 4. Figures (guarded)
    figs = maybe_make_figures(out_dir, rows, cfg, binding)

    return {
        "summary": summary_path,
        "significance": sig_path,
        "preregistered": prereg_path,
        "figures": figs,
    }


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
        device = _pick_device(cfg)
        print(f"[{now_str()}] C7 eval: device={device}", flush=True)
        run_eval(out_dir, cfg, device)
    elif cfg.mode == "calibrate":
        # Euclid + Oracle only (pure geometry/graph); no neural nets, no device.
        print(f"[{now_str()}] C7 calibrate: euclid+oracle sweep (CPU)", flush=True)
        run_calibrate(out_dir, cfg)
    elif cfg.mode == "analyze":
        # Reads out_dir/results/continuous_prm_c7_eval_raw.csv (no models / device).
        print(f"[{now_str()}] C7 analyze: stats + pre-registered comparisons (CPU)", flush=True)
        run_analyze(out_dir, cfg)
    elif cfg.mode == "full":
        # full = collect -> train -> calibrate -> eval -> analyze. full mode is not
        # wired here on purpose: the heavy collect/train stages require a GPU and a
        # multi-hour budget; run the individual modes (collect/train/calibrate/eval/
        # analyze) so each stage is checkpointed and resumable.
        raise NotImplementedError(
            "C7 full mode: run --mode collect/train/calibrate/eval/analyze individually "
            "(each stage checkpoints to out_dir so a long run is resumable)"
        )
    else:
        raise ValueError(f"unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
