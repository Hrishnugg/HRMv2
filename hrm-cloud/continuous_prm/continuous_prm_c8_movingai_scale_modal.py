"""Modal driver for C8-X (frozen design 2026-07-26-c8-movingai-scale-transfer.md).

Launch:  python -m modal run continuous_prm_c8_movingai_scale_modal.py::run_all

Phases run sequentially; shards fan out with .map. The source zips are
uploaded to the volume once from the local repo (driver side) if absent.
"""
from pathlib import Path

import modal

APP_NAME = "continuous-prm-c8x"
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

CATS = ["street", "dao"]
MS = [1, 2, 4, 8]
# Amendment 3 (2026-07-27): balanced draws (must mirror the core module)
DRAWS = {1: [0, 1, 2, 3, 4, 5, 6, 7], 2: [0, 1, 2, 3], 4: [0, 1, 2], 8: [0]}
METHODS = ["lora", "full", "scratch"]
SEEDS = [0, 1]


def _enter():
    import os
    import sys
    os.environ["C8X_ROOT"] = VOLUME_ROOT
    sys.path.insert(0, REMOTE_CODE_DIR)
    import continuous_prm_c8_movingai_scale as X
    return X


@app.function(image=image, timeout=60 * 30, cpu=2.0,
              volumes={VOLUME_ROOT: volume})
def select_remote() -> str:
    X = _enter()
    X.phase_select()
    volume.commit()
    return "selected"


@app.function(image=image, timeout=60 * 60 * 4, cpu=8.0,
              volumes={VOLUME_ROOT: volume})
def generate_remote(cat: str) -> str:
    X = _enter()
    X.phase_generate(cat)
    volume.commit()
    return f"{cat} generated"


@app.function(image=image, timeout=60 * 60 * 4, cpu=8.0,
              volumes={VOLUME_ROOT: volume})
def zeroshot_remote(shard: dict) -> str:
    X = _enter()
    X.phase_zeroshot(shard["cat"], shard["map"], device="cpu")
    volume.commit()
    return f"zs {shard['cat']}/{shard['map']}"


@app.function(image=image, gpu="L4", timeout=60 * 60 * 3,
              volumes={VOLUME_ROOT: volume})
def adapt_remote(cell: dict) -> str:
    X = _enter()
    X.phase_adapt(cell["cat"], cell["m"], cell["draw"], cell["method"],
                  cell["seed"], device="cuda")
    volume.commit()
    return f"adapt {cell}"


@app.function(image=image, timeout=60 * 60 * 5, cpu=8.0,
              volumes={VOLUME_ROOT: volume})
def evaladapted_remote(cell: dict) -> str:
    X = _enter()
    X.phase_evaladapted(cell["cat"], cell["m"], cell["draw"], cell["method"],
                        cell["seed"], device="cpu")
    volume.commit()
    return f"ad {cell}"


@app.function(image=image, timeout=60 * 20, cpu=4.0,
              volumes={VOLUME_ROOT: volume})
def evalpart_remote(shard: dict) -> str:
    X = _enter()
    X.phase_evaladapted(shard["cat"], shard["m"], shard["draw"],
                        shard["method"], shard["seed"], device="cpu",
                        only_map=shard["map"])
    volume.commit()
    return f"part {shard['map']} {shard['cat']}_M{shard['m']}_d{shard['draw']}_{shard['method']}_s{shard['seed']}"


@app.function(image=image, timeout=60 * 20, cpu=2.0,
              volumes={VOLUME_ROOT: volume})
def merge_remote(cell: dict) -> str:
    X = _enter()
    X.phase_mergeparts(cell["cat"], cell["m"], cell["draw"], cell["method"],
                       cell["seed"], cell["expected"])
    volume.commit()
    return f"merged {cell['cat']}_M{cell['m']}_d{cell['draw']}_{cell['method']}_s{cell['seed']}"


@app.function(image=image, timeout=60 * 10, cpu=2.0,
              volumes={VOLUME_ROOT: volume})
def list_state_remote() -> dict:
    """Missing ad_ cells + held-out map lists, computed volume-side."""
    import json as _json
    from pathlib import Path as _P
    root = _P(VOLUME_ROOT) / "runs" / "c8x2_scale"
    state = {"heldout": {}, "missing": []}
    for cat in CATS:
        man = _json.loads((root / f"instances_{cat}.json").read_text())
        state["heldout"][cat] = [x for x in man["usable"]
                                 if x not in man["pool"]]
    for c in CATS:
        for m in MS:
            for d in DRAWS[m]:
                for me in ("lora", "full", "scratch"):
                    for s in (0, 1):
                        tag = f"{c}_M{m}_d{d}_{me}_s{s}"
                        if not (root / f"ad_{tag}.csv").exists():
                            state["missing"].append(
                                dict(cat=c, m=m, draw=d, method=me, seed=s))
    return state


@app.local_entrypoint()
def repair_evals():
    """Amendment 3 operational path: per-(cell, held-out map) eval shards
    (preemption-resilient), then merge into ad_{tag}.csv."""
    state = list_state_remote.remote()
    missing = state["missing"]
    print(f"[c8x-modal] missing ad_ cells: {len(missing)}", flush=True)
    shards = [dict(cell, map=nm) for cell in missing
              for nm in state["heldout"][cell["cat"]]]
    print(f"[c8x-modal] eval parts ({len(shards)} shards)...", flush=True)
    for r in evalpart_remote.map(shards):
        print(" ", r, flush=True)
    merges = [dict(cell, expected=len(state["heldout"][cell["cat"]]))
              for cell in missing]
    print(f"[c8x-modal] merging {len(merges)} cells...", flush=True)
    for r in merge_remote.map(merges):
        print(" ", r, flush=True)
    print("[c8x-modal] ALL PHASES COMPLETE", flush=True)


@app.function(image=image, timeout=60 * 30, cpu=2.0,
              volumes={VOLUME_ROOT: volume})
def list_usable_remote() -> dict:
    import json
    X = _enter()
    out = {}
    for cat in CATS:
        man = json.loads((X.OUT / f"instances_{cat}.json").read_text())
        out[cat] = man["usable"]
    return out


@app.local_entrypoint()
def upload_zips():
    """One-time: put the local source zips on the volume."""
    import subprocess
    import sys
    for z in ("street-map.zip", "dao-map.zip"):
        local = Path(__file__).parent / "runs" / "c8r_movingai" / "maps" / z
        remote = f"runs/c8r_movingai/maps/{z}"
        r = subprocess.run([sys.executable, "-m", "modal", "volume", "put",
                            VOLUME_NAME, str(local), remote],
                           capture_output=True, text=True)
        print(z, r.returncode, (r.stdout + r.stderr)[-200:])


@app.local_entrypoint()
def run_all():
    print("[c8x-modal] select...", flush=True)
    print(" ", select_remote.remote(), flush=True)
    print("[c8x-modal] generate...", flush=True)
    for r in generate_remote.map(CATS):
        print(" ", r, flush=True)
    usable = list_usable_remote.remote()
    shards = [{"cat": c, "map": m} for c in CATS for m in usable[c]]
    print(f"[c8x-modal] zeroshot ({len(shards)} maps)...", flush=True)
    for r in zeroshot_remote.map(shards):
        print(" ", r, flush=True)
    cells = [{"cat": c, "m": m, "draw": d, "method": me, "seed": s}
             for c in CATS for m in MS for d in DRAWS[m]
             for me in METHODS for s in SEEDS]
    print(f"[c8x-modal] adapt ({len(cells)} cells, L4)...", flush=True)
    for r in adapt_remote.map(cells):
        print(" ", r, flush=True)
    print("[c8x-modal] eval adapted...", flush=True)
    for r in evaladapted_remote.map(cells):
        print(" ", r, flush=True)
    print("[c8x-modal] ALL PHASES COMPLETE", flush=True)
