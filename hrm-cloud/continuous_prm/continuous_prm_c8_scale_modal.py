"""Modal driver for C8-S v2 (design + Amendment 1).

Launch:  python -m modal run continuous_prm_c8_scale_modal.py::run_all
Eval and probe shards run on L4 containers (CPU classical arms + GPU
learned variant timed on identical hardware within each map).
"""
from pathlib import Path

import modal

APP_NAME = "continuous-prm-c8s"
VOLUME_NAME = "continuous-prm-heuristic-learning-vol"
VOLUME_ROOT = "/vol"
REMOTE_CODE_DIR = "/app/continuous_prm"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2.0", "numpy>=1.24.0")
    .add_local_dir(
        Path(__file__).parent,
        remote_path=REMOTE_CODE_DIR,
        copy=True,
        ignore=["__pycache__", "**/__pycache__", "*.pyc", "runs", "runs/**",
                "tests", ".pytest_cache"],
    )
)

SIZES = [192, 512, 1024, 2048]
SUITES = ["C_dyn_maze", "C_dyn_rooms", "C_dyn_spiral",
          "C_dyn_maze_dense", "C_dyn_crossing", "C_dyn_rooms_large"]


def _enter():
    import os
    import sys
    os.environ["C8S_ROOT"] = VOLUME_ROOT
    sys.path.insert(0, REMOTE_CODE_DIR)
    import continuous_prm_c8_scale_walltime as S
    return S


@app.function(image=image, timeout=60 * 60 * 2, cpu=4.0,
              volumes={VOLUME_ROOT: volume})
def manifest_remote(suite: str) -> str:
    S = _enter()
    S.phase_manifest(suite)
    volume.commit()
    return f"{suite} manifest"


@app.function(image=image, timeout=60 * 60 * 3, cpu=8.0,
              volumes={VOLUME_ROOT: volume})
def calib_tune_remote(shard: dict) -> str:
    S = _enter()
    S.phase_calib(shard["size"], shard["suite"])
    volume.commit()
    S.phase_tune(shard["size"], shard["suite"])
    volume.commit()
    return f"{shard['size']}/{shard['suite']} calibrated+tuned"


@app.function(image=image, gpu="L4", timeout=60 * 60 * 8, cpu=8.0,
              volumes={VOLUME_ROOT: volume})
def eval_remote(shard: dict) -> str:
    S = _enter()
    S.phase_eval(shard["size"], shard["suite"])
    volume.commit()
    return f"{shard['size']}/{shard['suite']} evaluated"


@app.function(image=image, gpu="L4", timeout=60 * 60 * 2,
              volumes={VOLUME_ROOT: volume})
def probe_remote(shard: dict) -> str:
    S = _enter()
    S.phase_probe(shard["size"], shard["suite"])
    volume.commit()
    return f"{shard['size']}/{shard['suite']} probed"


@app.function(image=image, timeout=60 * 60 * 6, cpu=8.0,
              volumes={VOLUME_ROOT: volume})
def sens_remote(shard: dict) -> str:
    S = _enter()
    S.phase_sens(shard["size"], shard["suite"], shard["stage"], shard["arm"])
    volume.commit()
    return f"{shard['size']}/{shard['suite']}/{shard['stage']}/{shard['arm']}"


@app.function(image=image, timeout=60 * 30, cpu=2.0,
              volumes={VOLUME_ROOT: volume})
def sens_select_remote(cell: dict) -> str:
    S = _enter()
    S.sens_select(cell["size"], cell["suite"])
    volume.commit()
    return f"{cell['size']}/{cell['suite']} selected"


@app.local_entrypoint()
def sensitivity():
    """Amendment 2 (2026-07-27): floor-cell sensitivity recalibration."""
    cells = [{"size": n, "suite": s} for (n, s) in [
        (192, "C_dyn_maze_dense"), (512, "C_dyn_maze_dense"),
        (1024, "C_dyn_maze_dense"), (2048, "C_dyn_maze_dense"),
        (2048, "C_dyn_spiral")]]
    dev_arms = ["euclid"] + [f"wastar_{w:g}"
                             for w in (1.1, 1.2, 1.5, 2.0, 3.0, 5.0)]
    dev = [dict(c, stage="dev", arm=a) for c in cells for a in dev_arms]
    print(f"[c8s2-modal] sens dev ({len(dev)} shards)...", flush=True)
    for r in sens_remote.map(dev):
        print(" ", r, flush=True)
    print("[c8s2-modal] sens select (5 cells)...", flush=True)
    for r in sens_select_remote.map(cells):
        print(" ", r, flush=True)
    ev = [dict(c, stage="eval", arm=a) for c in cells
          for a in ("euclid", "wastar_sel", "learned_cpu")]
    print(f"[c8s2-modal] sens eval ({len(ev)} shards)...", flush=True)
    for r in sens_remote.map(ev):
        print(" ", r, flush=True)
    print("[c8s2-modal] SENSITIVITY COMPLETE", flush=True)


@app.local_entrypoint()
def run_all():
    shards = [{"size": n, "suite": s} for n in SIZES for s in SUITES]
    print("[c8s2-modal] manifests (6 suites)...", flush=True)
    for r in manifest_remote.map(SUITES):
        print(" ", r, flush=True)
    print("[c8s2-modal] calibrate+tune (24 shards)...", flush=True)
    for r in calib_tune_remote.map(shards):
        print(" ", r, flush=True)
    print("[c8s2-modal] eval (24 shards, L4)...", flush=True)
    for r in eval_remote.map(shards):
        print(" ", r, flush=True)
    print("[c8s2-modal] probe (24 shards, L4)...", flush=True)
    for r in probe_remote.map(shards):
        print(" ", r, flush=True)
    print("[c8s2-modal] ALL PHASES COMPLETE", flush=True)
