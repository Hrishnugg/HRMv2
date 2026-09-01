#!/usr/bin/env python3
"""Modal orchestration for C14 (label-count x world-diversity factorial).

Burst-parallel execution of the C14 grid per the frozen design + amendment v2:
the LOCALLY collected static cell datasets are uploaded byte-identical (this
preserves the shared sampled index sets), the dynamic pool/cells are collected
remotely from the same deterministic seed streams, the 180 adaptation arms
fan out one-per-container at matched 2,560 steps, and the evals run on the
frozen C9/C9b test cohorts (dynamic eval sharded by test world). Checkpoints
resume for free: an arm whose checkpoint exists on the volume is skipped.

Usage:
    modal run continuous_prm_c14_modal.py::run_c14 --phase all
    modal run continuous_prm_c14_modal.py::run_c14 --phase adapt --limit-arms 2   # smoke
    modal volume get continuous-prm-heuristic-learning-vol runs/c14_modal/results ...

Phases: upload | collect | adapt | eval | all (each idempotent).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

APP_NAME = "continuous-prm-c14"
VOLUME_NAME = "continuous-prm-heuristic-learning-vol"
VOLUME_ROOT = "/vol"
REMOTE_CODE_DIR = "/app/continuous_prm"
RUN_NAME = "c14_modal"

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
        ignore=["__pycache__", "**/__pycache__", "*.pyc", "runs", "runs/**",
                "hrm-cloud", "tests", ".pytest_cache", ".pytest-c13p-fresh-history-red"],
    )
)

N_GRID = "256,1024,4096,16384,65536"
N_SEEDS = 3
TOTAL_STEPS = 2560
N_TEST_STATIC = 30
N_TEST_DYNAMIC = 20


def _remote_cfg():
    """C14Config rooted at the volume (import only works inside containers)."""
    import sys
    sys.path.insert(0, REMOTE_CODE_DIR)
    import continuous_prm_c14_label_density as X

    cfg = X.C14Config(
        static_source_dir=f"{VOLUME_ROOT}/runs/c14_sources/c7_local",
        dynamic_source_dir=f"{VOLUME_ROOT}/runs/c14_sources/c8_local_heavy",
        out_dir=f"{VOLUME_ROOT}/runs/{RUN_NAME}",
        n_grid=N_GRID, n_seeds=N_SEEDS, total_steps=TOTAL_STEPS,
        n_test_static=N_TEST_STATIC, n_test_dynamic=N_TEST_DYNAMIC,
    )
    return X, cfg


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.function(image=image, timeout=60 * 60 * 3, volumes={VOLUME_ROOT: volume}, cpu=4.0)
def collect_dynamic_remote() -> str:
    """Dynamic pool + cell collection from the deterministic seed streams."""
    X, cfg = _remote_cfg()
    volume.reload()
    man = X._load_manifest(cfg)
    X.collect_dynamic_cells(cfg, man)
    volume.commit()
    return "dynamic cells collected"


@app.function(image=image, gpu="L4", timeout=60 * 60 * 2, volumes={VOLUME_ROOT: volume})
def train_arm_remote(arm: Dict[str, Any]) -> Dict[str, Any]:
    """Train one (domain, N, div, method, seed) arm; skipped if ckpt exists."""
    X, cfg = _remote_cfg()
    volume.reload()
    man = X._load_manifest(cfg)
    domain, N, div = arm["domain"], int(arm["N"]), arm["div"]
    method, s = arm["method"], int(arm["seed"])
    key = X.cell_key(domain, N, div)
    cell = man["cells"].get(key)
    if cell is None:
        raise RuntimeError(f"cell {key} not collected")
    ck = X.arm_ckpt(cfg, domain, N, div, method, s)
    if not ck.exists():
        # Cell npz paths were recorded under the local run layout at collection
        # time; resolve by basename under this run's datasets dir on the volume.
        npz = Path(cfg.out_dir) / "datasets" / Path(cell["npz"]).name
        if domain == "static":
            X.train_static_arm(cfg, npz, method, s, ck, _device())
        else:
            X.train_dynamic_arm(cfg, npz, method, s, ck, _device())
        volume.commit()
    return dict(domain=domain, N=N, diversity=div, method=method, seed=s, ckpt=str(ck))


@app.function(image=image, timeout=60 * 15, volumes={VOLUME_ROOT: volume})
def finalize_manifest_remote() -> str:
    """Rebuild manifest['arms'] by scanning the checkpoints directory."""
    X, cfg = _remote_cfg()
    volume.reload()
    man = X._load_manifest(cfg)
    arms = []
    for p in sorted((Path(cfg.out_dir) / "checkpoints").glob("c14__*.pt")):
        parts = p.stem.split("__")  # c14, domain, N{n}, div, method, s{k}
        arms.append(dict(domain=parts[1], N=int(parts[2][1:]), diversity=parts[3],
                         method=parts[4], seed=int(parts[5][1:]), ckpt=str(p)))
    man["arms"] = arms
    X._save_manifest(cfg, man)
    volume.commit()
    return f"{len(arms)} arms in manifest"


@app.function(image=image, gpu="L4", timeout=60 * 60 * 2, volumes={VOLUME_ROOT: volume}, cpu=8.0)
def eval_static_world_remote(world_pos: int) -> str:
    """One static test world x all providers (feature building dominates and is
    per-provider, so sharding by world is the parallelism that matters)."""
    import csv as _csv
    X, cfg = _remote_cfg()
    volume.reload()
    man = X._load_manifest(cfg)
    rows = X.eval_static(cfg, man, _device(), only_worlds=[int(world_pos)])
    shard_dir = Path(cfg.out_dir) / "results" / "_shards" / "static"
    shard_dir.mkdir(parents=True, exist_ok=True)
    with open(shard_dir / f"world_{int(world_pos):03d}.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=X.RAW_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in X.RAW_COLS})
    volume.commit()
    return f"static world {world_pos}: {len(rows)} rows"


@app.function(image=image, gpu="L4", timeout=60 * 60 * 2, volumes={VOLUME_ROOT: volume})
def eval_dynamic_world_remote(world_pos: int) -> str:
    import csv as _csv
    X, cfg = _remote_cfg()
    volume.reload()
    man = X._load_manifest(cfg)
    rows = X.eval_dynamic(cfg, man, _device(), only_worlds=[int(world_pos)])
    shard_dir = Path(cfg.out_dir) / "results" / "_shards" / "dynamic"
    shard_dir.mkdir(parents=True, exist_ok=True)
    with open(shard_dir / f"world_{int(world_pos):03d}.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=X.RAW_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in X.RAW_COLS})
    volume.commit()
    return f"world {world_pos}: {len(rows)} rows"


@app.function(image=image, timeout=60 * 15, volumes={VOLUME_ROOT: volume})
def merge_eval_remote() -> str:
    import csv as _csv
    X, cfg = _remote_cfg()
    volume.reload()
    res = Path(cfg.out_dir) / "results"
    for domain in ("static", "dynamic"):
        rows: List[dict] = []
        for p in sorted((res / "_shards" / domain).glob("world_*.csv")):
            with open(p, newline="") as f:
                rows.extend(_csv.DictReader(f))
        with open(res / f"c14_{domain}_raw.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=X.RAW_COLS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in X.RAW_COLS})
    all_rows: List[dict] = []
    for name in ("c14_static_raw.csv", "c14_dynamic_raw.csv"):
        p = res / name
        if p.exists():
            with open(p, newline="") as f:
                all_rows.extend(_csv.DictReader(f))
    with open(res / "continuous_prm_c14_eval_raw.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=X.RAW_COLS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in X.RAW_COLS})
    volume.commit()
    return f"merged {len(all_rows)} rows"


def _local_paths():
    here = Path(__file__).parent
    return {
        "cells_dir": here / "runs" / "c14_local" / "datasets",
        "manifest": here / "runs" / "c14_local" / "c14_manifest.json",
        "c7_ckpt": here / "runs" / "c7_local" / "checkpoints" / "avgbase__hrm.pt",
        "c7_calib": here / "runs" / "c7_local" / "calibration.json",
        "c8_ckpt": here / "runs" / "c8_local_heavy" / "checkpoints" / "c8_field__unet_blind.pt",
        "c8_calib": here / "runs" / "c8_local_heavy" / "calibration.json",
    }


def _upload():
    """Push static cell datasets, cleaned manifest, and frozen sources."""
    lp = _local_paths()
    man = json.loads(lp["manifest"].read_text())
    # Cells only; arms retrain on Modal for uniform provenance. Rewrite each
    # cell's npz path to the volume layout (basename under the run datasets dir).
    cells = {}
    for key, cell in man["cells"].items():
        c = dict(cell)
        c["npz"] = f"{VOLUME_ROOT}/runs/{RUN_NAME}/datasets/{Path(cell['npz']).name}"
        cells[key] = c
    cleaned = {"cells": cells, "arms": [], "sources": {
        "static": "c14_sources/c7_local/checkpoints/avgbase__hrm.pt",
        "dynamic": "c14_sources/c8_local_heavy/checkpoints/c8_field__unet_blind.pt",
        "provenance": "collected locally 2026-07-23; adapted on Modal L4 fleet",
    }, "config": man.get("config", {})}
    tmp = lp["manifest"].with_suffix(".modal.json")
    tmp.write_text(json.dumps(cleaned, indent=1))

    with volume.batch_upload(force=True) as batch:
        for p in sorted(lp["cells_dir"].glob("static__*.npz")):
            batch.put_file(p, f"runs/{RUN_NAME}/datasets/{p.name}")
        batch.put_file(tmp, f"runs/{RUN_NAME}/c14_manifest.json")
        batch.put_file(lp["c7_ckpt"], "runs/c14_sources/c7_local/checkpoints/avgbase__hrm.pt")
        batch.put_file(lp["c7_calib"], "runs/c14_sources/c7_local/calibration.json")
        batch.put_file(lp["c8_ckpt"], "runs/c14_sources/c8_local_heavy/checkpoints/c8_field__unet_blind.pt")
        batch.put_file(lp["c8_calib"], "runs/c14_sources/c8_local_heavy/calibration.json")
    print("upload complete (10 static cells + manifest + sources)")


def _arm_list(limit: int = 0) -> List[Dict[str, Any]]:
    arms = []
    for domain in ("static", "dynamic"):
        for N in [int(x) for x in N_GRID.split(",")]:
            for div in ("conc", "dist"):
                for method in ("lora", "full_ft", "scratch"):
                    for s in range(N_SEEDS):
                        arms.append(dict(domain=domain, N=N, div=div, method=method, seed=s))
    return arms[:limit] if limit > 0 else arms


@app.local_entrypoint()
def run_c14(phase: str = "all", limit_arms: int = 0) -> None:
    if phase in ("upload", "all"):
        _upload()
    if phase in ("collect", "all"):
        print(collect_dynamic_remote.remote())
    if phase in ("adapt", "all"):
        arms = _arm_list(limit_arms)
        # Static arms only need static cells (already uploaded); dynamic arms
        # need the collect phase to have completed.
        if limit_arms > 0:
            arms = [a for a in arms if a["domain"] == "static"][:limit_arms]
            print(f"smoke: {len(arms)} static arms")
        print(f"training {len(arms)} arms on L4 fleet")
        done = list(train_arm_remote.map(arms))
        print(f"trained/verified {len(done)} arms")
        print(finalize_manifest_remote.remote())
    if phase in ("eval", "all"):
        static_results = list(eval_static_world_remote.map(range(N_TEST_STATIC)))
        print(f"static eval: {len(static_results)} world shards")
        dyn_results = list(eval_dynamic_world_remote.map(range(N_TEST_DYNAMIC)))
        print(f"dynamic eval: {len(dyn_results)} world shards")
        print(merge_eval_remote.remote())
    print("phase(s) complete")
