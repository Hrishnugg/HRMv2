#!/usr/bin/env python3
"""Modal orchestration for C14-R (independent world-set replicates).

Frozen design: docs/experiments/continuous/c14/design/2026-07-25-c14r-worldset-replicates.md

Per draw d in {2, 3}: collect NEW static and dynamic pools with
draw_offset=d (only the pool seed streams change; test cohorts, budgets, and
sampling rules are byte-identical to C14), build the restricted cells
(static N=256 x {conc, dist}; dynamic N in {1024, 16384} x {conc, dist}),
train 18 arms ({6 cells} x {lora, full_ft, scratch} x seed 0) at the matched
2,560 steps, and evaluate on the SAME frozen C9/C9b test cohorts.

Sources are already on the volume from the original C14 run
(runs/c14_sources/...). Each draw is fully idempotent under
runs/c14r_draw{d}/.

Usage:
    python -m modal run continuous_prm_c14r_modal.py::run_c14r --draw 2 --phase all
    python -m modal run continuous_prm_c14r_modal.py::run_c14r --draw 3 --phase all
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import modal

APP_NAME = "continuous-prm-c14r"
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
        ignore=["__pycache__", "**/__pycache__", "*.pyc", "runs", "runs/**",
                "hrm-cloud", "tests", ".pytest_cache"],
    )
)

STATIC_N_GRID = "256"
DYNAMIC_N_GRID = "1024,16384"
N_SEEDS = 1
TOTAL_STEPS = 2560
N_TEST_STATIC = 30
N_TEST_DYNAMIC = 20


def _remote_cfg(draw: int, domain: str):
    """C14Config for one draw and one domain's collection grid."""
    import sys
    sys.path.insert(0, REMOTE_CODE_DIR)
    import continuous_prm_c14_label_density as X

    n_grid = STATIC_N_GRID if domain == "static" else DYNAMIC_N_GRID
    cfg = X.C14Config(
        static_source_dir=f"{VOLUME_ROOT}/runs/c14_sources/c7_local",
        dynamic_source_dir=f"{VOLUME_ROOT}/runs/c14_sources/c8_local_heavy",
        out_dir=f"{VOLUME_ROOT}/runs/c14r_draw{int(draw)}",
        domains=domain, n_grid=n_grid, n_seeds=N_SEEDS,
        total_steps=TOTAL_STEPS, n_test_static=N_TEST_STATIC,
        n_test_dynamic=N_TEST_DYNAMIC, draw_offset=int(draw),
    )
    return X, cfg


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.function(image=image, timeout=60 * 60 * 3, volumes={VOLUME_ROOT: volume}, cpu=8.0)
def collect_remote(draw: int) -> str:
    """Collect BOTH domains' pools and cells for this draw (CPU)."""
    X, cfg_s = _remote_cfg(draw, "static")
    volume.reload()
    man = X._load_manifest(cfg_s)
    man.setdefault("sources", {}).update({
        "static": "c14_sources/c7_local/checkpoints/avgbase__hrm.pt",
        "dynamic": "c14_sources/c8_local_heavy/checkpoints/c8_field__unet_blind.pt",
        "provenance": f"C14-R draw {int(draw)}: independent world streams "
                      f"(draw_offset={int(draw)}), collected on Modal 2026-07-25",
    })
    X.collect_static_cells(cfg_s, man)
    _X, cfg_d = _remote_cfg(draw, "dynamic")
    X.collect_dynamic_cells(cfg_d, man)
    X._save_manifest(cfg_s, man)
    volume.commit()
    return f"draw {draw}: cells collected ({len(man['cells'])})"


@app.function(image=image, gpu="L4", timeout=60 * 60 * 2, volumes={VOLUME_ROOT: volume})
def train_arm_remote(arm: Dict[str, Any]) -> Dict[str, Any]:
    draw = int(arm["draw"])
    domain, N, div = arm["domain"], int(arm["N"]), arm["div"]
    method, s = arm["method"], int(arm["seed"])
    X, cfg = _remote_cfg(draw, domain)
    volume.reload()
    man = X._load_manifest(cfg)
    key = X.cell_key(domain, N, div)
    cell = man["cells"].get(key)
    if cell is None:
        raise RuntimeError(f"draw {draw} cell {key} not collected")
    ck = X.arm_ckpt(cfg, domain, N, div, method, s)
    if not ck.exists():
        npz = Path(cfg.out_dir) / "datasets" / Path(cell["npz"]).name
        if domain == "static":
            X.train_static_arm(cfg, npz, method, s, ck, _device())
        else:
            X.train_dynamic_arm(cfg, npz, method, s, ck, _device())
        volume.commit()
    return dict(draw=draw, domain=domain, N=N, diversity=div, method=method,
                seed=s, ckpt=str(ck))


@app.function(image=image, timeout=60 * 15, volumes={VOLUME_ROOT: volume})
def finalize_manifest_remote(draw: int) -> str:
    X, cfg = _remote_cfg(draw, "static")
    volume.reload()
    man = X._load_manifest(cfg)
    arms = []
    for p in sorted((Path(cfg.out_dir) / "checkpoints").glob("c14__*.pt")):
        parts = p.stem.split("__")
        arms.append(dict(domain=parts[1], N=int(parts[2][1:]), diversity=parts[3],
                         method=parts[4], seed=int(parts[5][1:]), ckpt=str(p)))
    man["arms"] = arms
    X._save_manifest(cfg, man)
    volume.commit()
    return f"draw {draw}: {len(arms)} arms in manifest"


@app.function(image=image, gpu="L4", timeout=60 * 60 * 2, volumes={VOLUME_ROOT: volume}, cpu=8.0)
def eval_static_world_remote(spec: Dict[str, Any]) -> str:
    import csv as _csv
    draw, world_pos = int(spec["draw"]), int(spec["world"])
    X, cfg = _remote_cfg(draw, "static")
    volume.reload()
    man = X._load_manifest(cfg)
    rows = X.eval_static(cfg, man, _device(), only_worlds=[world_pos])
    shard_dir = Path(cfg.out_dir) / "results" / "_shards" / "static"
    shard_dir.mkdir(parents=True, exist_ok=True)
    with open(shard_dir / f"world_{world_pos:03d}.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=X.RAW_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in X.RAW_COLS})
    volume.commit()
    return f"draw {draw} static world {world_pos}: {len(rows)} rows"


@app.function(image=image, gpu="L4", timeout=60 * 60 * 2, volumes={VOLUME_ROOT: volume})
def eval_dynamic_world_remote(spec: Dict[str, Any]) -> str:
    import csv as _csv
    draw, world_pos = int(spec["draw"]), int(spec["world"])
    X, cfg = _remote_cfg(draw, "dynamic")
    volume.reload()
    man = X._load_manifest(cfg)
    rows = X.eval_dynamic(cfg, man, _device(), only_worlds=[world_pos])
    shard_dir = Path(cfg.out_dir) / "results" / "_shards" / "dynamic"
    shard_dir.mkdir(parents=True, exist_ok=True)
    with open(shard_dir / f"world_{world_pos:03d}.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=X.RAW_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in X.RAW_COLS})
    volume.commit()
    return f"draw {draw} dynamic world {world_pos}: {len(rows)} rows"


@app.function(image=image, timeout=60 * 15, volumes={VOLUME_ROOT: volume})
def merge_eval_remote(draw: int) -> str:
    import csv as _csv
    X, cfg = _remote_cfg(draw, "static")
    volume.reload()
    res = Path(cfg.out_dir) / "results"
    all_rows: List[dict] = []
    for domain in ("static", "dynamic"):
        rows: List[dict] = []
        shards = res / "_shards" / domain
        if shards.exists():
            for p in sorted(shards.glob("world_*.csv")):
                with open(p, newline="") as f:
                    rows.extend(_csv.DictReader(f))
        with open(res / f"c14r_{domain}_raw.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=X.RAW_COLS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in X.RAW_COLS})
        all_rows.extend(rows)
    with open(res / "continuous_prm_c14r_eval_raw.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=X.RAW_COLS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in X.RAW_COLS})
    volume.commit()
    return f"draw {draw}: merged {len(all_rows)} rows"


def _arm_list(draw: int) -> List[Dict[str, Any]]:
    arms = []
    cells = ([("static", int(n)) for n in STATIC_N_GRID.split(",")] +
             [("dynamic", int(n)) for n in DYNAMIC_N_GRID.split(",")])
    for domain, N in cells:
        for div in ("conc", "dist"):
            for method in ("lora", "full_ft", "scratch"):
                for s in range(N_SEEDS):
                    arms.append(dict(draw=int(draw), domain=domain, N=N,
                                     div=div, method=method, seed=s))
    return arms


@app.local_entrypoint()
def run_c14r(draw: int = 2, phase: str = "all") -> None:
    if phase in ("collect", "all"):
        print(collect_remote.remote(draw))
    if phase in ("adapt", "all"):
        arms = _arm_list(draw)
        print(f"draw {draw}: training {len(arms)} arms on L4 fleet")
        done = list(train_arm_remote.map(arms))
        print(f"draw {draw}: trained/verified {len(done)} arms")
        print(finalize_manifest_remote.remote(draw))
    if phase in ("eval", "all"):
        st = list(eval_static_world_remote.map(
            [dict(draw=draw, world=w) for w in range(N_TEST_STATIC)]))
        print(f"draw {draw}: static eval {len(st)} shards")
        dy = list(eval_dynamic_world_remote.map(
            [dict(draw=draw, world=w) for w in range(N_TEST_DYNAMIC)]))
        print(f"draw {draw}: dynamic eval {len(dy)} shards")
        print(merge_eval_remote.remote(draw))
    print(f"draw {draw}: phase(s) complete")
