"""SHA-256 manifest of every c8x2 cell dataset + checkpoint on the volume."""
import modal

app = modal.App("c8x2-sha")
vol = modal.Volume.from_name("continuous-prm-heuristic-learning-vol")
img = modal.Image.debian_slim(python_version="3.11")


@app.function(image=img, volumes={"/vol": vol}, timeout=1800)
def manifest() -> str:
    import hashlib
    import json
    from pathlib import Path
    root = Path("/vol/runs/c8x2_scale")
    out = {}
    for sub, pat in (("datasets", "*.npz"), ("ckpts", "*.pt")):
        d = root / sub
        if not d.exists():
            continue
        for p in sorted(d.glob(pat)):
            out[f"{sub}/{p.name}"] = hashlib.sha256(
                p.read_bytes()).hexdigest()
    (root / "datasets_ckpts_sha256.json").write_text(
        json.dumps(out, indent=1))
    vol.commit()
    return f"{len(out)} artifacts hashed -> datasets_ckpts_sha256.json"


@app.local_entrypoint()
def main():
    print(manifest.remote())
