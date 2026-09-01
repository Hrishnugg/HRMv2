"""Bundle c8x2_scale CSV rows + manifests on the Modal volume into one tarball."""
import modal

app = modal.App("c8x2-bundle")
vol = modal.Volume.from_name("continuous-prm-heuristic-learning-vol")
img = modal.Image.debian_slim(python_version="3.11")


@app.function(image=img, volumes={"/vol": vol}, timeout=600)
def bundle() -> str:
    import tarfile
    from pathlib import Path

    root = Path("/vol/runs/c8x2_scale")
    out = root / "rows_bundle.tar.gz"
    names = sorted(
        [p for p in root.iterdir()
         if p.suffix in (".csv", ".json") and p.name != "rows_bundle.tar.gz"])
    with tarfile.open(out, "w:gz") as tf:
        for p in names:
            tf.add(p, arcname=p.name)
    vol.commit()
    return f"{len(names)} files -> {out.name} ({out.stat().st_size} bytes)"


@app.local_entrypoint()
def main():
    print(bundle.remote())
