"""Delete the five partial ad_ cell files on the volume."""
import modal

app = modal.App("c8x2-delpartial")
vol = modal.Volume.from_name("continuous-prm-heuristic-learning-vol")
img = modal.Image.debian_slim(python_version="3.11")

BAD = ["ad_dao_M1_d4_scratch_s1.csv", "ad_dao_M1_d6_lora_s0.csv",
       "ad_dao_M1_d7_lora_s1.csv", "ad_street_M1_d3_full_s1.csv",
       "ad_street_M2_d3_full_s0.csv"]


@app.function(image=img, volumes={"/vol": vol}, timeout=300)
def clean() -> str:
    from pathlib import Path
    root = Path("/vol/runs/c8x2_scale")
    n = 0
    for name in BAD:
        p = root / name
        if p.exists():
            p.unlink()
            n += 1
    vol.commit()
    return f"deleted {n}/{len(BAD)}"


@app.local_entrypoint()
def main():
    print(clean.remote())
