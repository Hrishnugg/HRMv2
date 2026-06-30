import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parents[1]
import numpy as np
import continuous_prm_c10_interp as C10
import continuous_prm_common as C


def test_family_grid_specs_and_bracketing():
    specs = C10.c10_family_specs()
    for fam in C10.SOURCE_FAMILIES + C10.TARGET_FAMILIES:
        assert fam in specs
        s = specs[fam]
        assert s.mode == C10.HARD_MODE and s.name in ("C_hard_maze", "C_hard_rooms")
    src_cent = {f: C10.family_descriptor_centroid(specs[f], n=6, seed=1) for f in C10.SOURCE_FAMILIES}
    for tgt in C10.TARGET_FAMILIES:
        z_t = C10.family_descriptor_centroid(specs[tgt], n=6, seed=2)
        ok, viol = C10.bracketing_ok(z_t, list(src_cent.values()))
        assert isinstance(ok, bool) and isinstance(viol, list)
    z = C10.family_descriptor_centroid(specs["C10_maze_tgt"], n=8, seed=3)
    ok, viol = C10.bracketing_ok(z, [C10.family_descriptor_centroid(specs[f], n=8, seed=3) for f in C10.SOURCE_FAMILIES])
    assert isinstance(ok, bool)


def test_rbf_weights():
    cent = [np.array([0.,0.]), np.array([1.,0.]), np.array([0.,1.])]
    z = np.array([0.1, 0.1])
    w = C10.rbf_weights(z, cent, sigma=1.0)
    assert abs(float(w.sum()) - 1.0) < 1e-6 and w.shape == (3,)
    assert int(np.argmax(w)) == 0  # nearest centroid gets most weight
    w0 = C10.rbf_weights(z, cent, sigma=1e-4)  # sigma->0 => one-hot on nearest
    assert w0[0] > 0.99
    assert C10.nearest_weights(z, cent).tolist() == [1.0, 0.0, 0.0]
    u = C10.uniform_weights(3)
    assert np.allclose(u, 1/3)


@pytest.mark.skipif(not (HERE/"runs/c7_local/checkpoints/avgbase__hrm.pt").exists(), reason="base missing")
def test_train_source_experts_smoke(tmp_path):
    import torch
    cfg = C10.C10Config(source_dir=str(HERE/"runs/c7_local"), out_dir=str(tmp_path/"c10"),
                        backbones="hrm", n_src_worlds=2, n_centroid_worlds=4, epochs=1,
                        roadmap_nodes=192, roadmap_k=7, cpu=True, seed=7)
    man = C10.train_source_experts(cfg, torch.device("cpu"),
                                   only_families=["C10_maze_d1", "C10_rooms_s20"])
    assert len(man["experts"]) == 2
    for e in man["experts"]:
        assert Path(e["ckpt"]).exists() and "centroid" in e and len(e["centroid"]) == 8
