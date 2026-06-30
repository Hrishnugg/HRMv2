import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parents[1]
import numpy as np
import continuous_prm_c9b_dynamics_transfer as C9B
import continuous_prm_common as C
import continuous_prm_c9_transfer as C9


def test_config_sources_and_suites():
    cfg = C9B.C9bConfig()
    assert cfg.backbones == "scalar_hrm,scalar_onlstm,field_unet"
    assert cfg.targets == "C_dyn_maze_dense,C_dyn_crossing,C_dyn_rooms_large"
    assert set(cfg.awareness_list()) == {"aware", "blind"}
    srcs = C9B.resolve_sources(cfg)
    assert set(srcs) == {(b, a) for b in ("scalar_hrm", "scalar_onlstm", "field_unet") for a in ("aware", "blind")}
    C9B.install()
    specs = C.build_anchor_specs()
    for t in C9B._parse_csv(cfg.targets):
        assert t in specs


def test_adapt_test_disjoint():
    C9B.install()
    cfg = C9B.C9bConfig(n_test=4, seed=7)
    adapt = C9B.adapt_world_seeds("C_dyn_crossing", K=4, seed_idx=0, cfg=cfg)
    test = C9B.test_world_seeds("C_dyn_crossing", cfg)
    assert len(adapt) == 4 and len(test) == 4
    assert set(adapt).isdisjoint(set(test))
    fa = {C9.world_fingerprint(C9B._build_world_only("C_dyn_crossing", s)) for s in adapt}
    ft = {C9.world_fingerprint(C9B._build_world_only("C_dyn_crossing", s)) for s in test}
    assert fa.isdisjoint(ft)


@pytest.mark.skipif(not (HERE / "runs/c8_local_heavy/checkpoints/c8_scalar__hrm.pt").exists(), reason="c8 sources missing")
def test_temporal_dataset_shapes(tmp_path):
    C9B.install()
    cfg = C9B.C9bConfig(out_dir=str(tmp_path / "c9b"))
    seeds = C9B.adapt_world_seeds("C_dyn_crossing", K=2, seed_idx=0, cfg=cfg)
    sa = C9B.collect_temporal_dataset("C_dyn_crossing", seeds, backbone="scalar_hrm", window_w=8, k_patrollers=4, grid_size=64, out_npz=tmp_path / "sa.npz")
    sb = C9B.collect_temporal_dataset("C_dyn_crossing", seeds, backbone="scalar_hrm", window_w=0, k_patrollers=4, grid_size=64, out_npz=tmp_path / "sb.npz")
    A = np.load(sa); B = np.load(sb)
    assert A["x"].ndim == 3 and A["x"].shape[1] == 9      # (M, W+1=9, token_dim)
    assert B["x"].shape[1] == 1                            # blind seq dim 1
    assert A["y"].shape[0] == A["x"].shape[0] and A["x"].shape[0] > 0


@pytest.mark.skipif(not (HERE / "runs/c8_local_heavy/checkpoints/c8_field__unet.pt").exists(), reason="c8 sources missing")
def test_temporal_dataset_field_shapes(tmp_path):
    C9B.install()
    cfg = C9B.C9bConfig(out_dir=str(tmp_path / "c9b"))
    seeds = C9B.adapt_world_seeds("C_dyn_crossing", K=2, seed_idx=0, cfg=cfg)
    f = C9B.collect_temporal_dataset("C_dyn_crossing", seeds, backbone="field_unet", window_w=8,
                                     k_patrollers=4, grid_size=64, out_npz=tmp_path / "f.npz")
    F = np.load(f)
    assert F["occ"].ndim == 4 and F["occ"].shape[1] == 16          # 8 + W(=8)
    assert F["occ"].shape[2] == 64 and F["occ"].shape[3] == 64
    assert F["occ"].shape[0] > 0
    assert F["cells"].shape[0] == F["occ"].shape[0] and F["cells"].shape[2] == 2
    assert F["target"].shape == F["mask"].shape == F["cells"].shape[:2]
    assert F["mask"].dtype == np.bool_


@pytest.mark.skipif(not (HERE/"runs/c8_local_heavy/checkpoints/c8_scalar__hrm.pt").exists(), reason="c8 sources missing")
def test_scalar_trainer_methods(tmp_path):
    C9B.install(); import torch
    cfg = C9B.C9bConfig(out_dir=str(tmp_path/"c9b"), epochs=1, cpu=True)
    seeds = C9B.adapt_world_seeds("C_dyn_crossing", K=2, seed_idx=0, cfg=cfg)
    npz = C9B.collect_temporal_dataset("C_dyn_crossing", seeds, backbone="scalar_hrm", window_w=8,
                                       k_patrollers=4, grid_size=64, out_npz=tmp_path/"d.npz")
    src = HERE/"runs/c8_local_heavy/checkpoints/c8_scalar__hrm.pt"
    for method in ("lora", "full_ft", "scratch"):
        ck = C9B.train_scalar_temporal(npz, tmp_path/f"{method}.pt", source_ckpt=src,
                                       method=method, cfg=cfg, device=torch.device("cpu"), seed=0)
        payload = torch.load(ck, map_location="cpu")
        assert payload["window_w"] == 8 and payload["method"] == method
        assert payload["k_patrollers"] == 4 and payload["token_dim"] == 20
        if method == "lora":
            assert payload["lora_rank"] == 8


@pytest.mark.skipif(not (HERE/"runs/c8_local_heavy/checkpoints/c8_field__unet.pt").exists(), reason="c8 sources missing")
def test_field_trainer_methods(tmp_path):
    C9B.install(); import torch
    cfg = C9B.C9bConfig(out_dir=str(tmp_path/"c9b"), epochs=1, cpu=True)
    seeds = C9B.adapt_world_seeds("C_dyn_crossing", K=2, seed_idx=0, cfg=cfg)
    npz = C9B.collect_temporal_dataset("C_dyn_crossing", seeds, backbone="field_unet", window_w=8,
                                       k_patrollers=4, grid_size=64, out_npz=tmp_path/"f.npz")
    src = HERE/"runs/c8_local_heavy/checkpoints/c8_field__unet.pt"
    for method in ("lora", "full_ft", "scratch"):
        ck = C9B.train_field_temporal(npz, tmp_path/f"{method}.pt", source_ckpt=src,
                                      method=method, cfg=cfg, device=torch.device("cpu"), seed=0)
        p = torch.load(ck, map_location="cpu")
        assert p["in_channels"] == 16 and p["window_w"] == 8 and p["method"] == method
        if method == "lora":
            assert p["lora_rank"] == 8


@pytest.mark.skipif(not (HERE/"runs/c8_local_heavy/checkpoints/c8_scalar__hrm.pt").exists(), reason="c8 sources missing")
def test_provider_loaders(tmp_path):
    C9B.install(); import torch
    import continuous_prm_c8_dynamic_maps as M8MAPS
    import continuous_prm_common as C
    dev = torch.device("cpu")
    src = HERE/"runs/c8_local_heavy/checkpoints/c8_scalar__hrm.pt"
    prov = C9B.load_temporal_provider(src, backbone="scalar_hrm", device=dev)
    assert prov.name.startswith("scalar_hrm")
    # smoke a forward on one valid crossing world (use _collect_world_labels for a guaranteed-valid world+rm+dyn)
    seeds = C9B.test_world_seeds("C_dyn_crossing", C9B.C9bConfig(n_test=1))
    lab = C9B._collect_world_labels_memo("C_dyn_crossing", seeds[0], 64)
    ht = prov.h_table(lab["world"], lab["rm"], lab["dyn"], lab["params"]["v_agent"], lab["params"]["dt"], int(lab["params"]["t_max"]))
    assert ht.shape[0] == lab["rm"].points.shape[0]


@pytest.mark.skipif(not (HERE/"runs/c8_local_heavy/checkpoints/c8_scalar__hrm.pt").exists(), reason="c8 sources missing")
def test_run_adapt_smoke(tmp_path):
    import torch
    cfg = C9B.C9bConfig(out_dir=str(tmp_path/"c9b"), backbones="scalar_hrm", awareness="aware,blind",
                        methods="lora,scratch", k_grid="1", n_adapt_seeds=1, n_test=4, epochs=1, cpu=True, seed=7)
    man = C9B.run_adapt(cfg, torch.device("cpu"), only_targets=["C_dyn_crossing"])
    assert len(man["arms"]) == 4   # 1 target x 1 bb x 2 awareness x 2 methods x 1 K x 1 seed
    for a in man["arms"]:
        assert Path(a["ckpt"]).exists() and a["awareness"] in ("aware","blind") and a["method"] in ("lora","scratch")
    assert (Path(cfg.out_dir)/"adapt_manifest.json").exists()


@pytest.mark.skipif(not (HERE/"runs/c8_local_heavy/checkpoints/c8_scalar__hrm.pt").exists(), reason="c8 sources missing")
def test_run_eval_smoke(tmp_path):
    import torch, csv
    cfg = C9B.C9bConfig(out_dir=str(tmp_path/"c9b"), backbones="scalar_hrm", awareness="aware,blind",
                        methods="lora,scratch", k_grid="1", n_adapt_seeds=1, n_test=3, epochs=1,
                        budgets="150,250", cpu=True, seed=7)
    C9B.run_adapt(cfg, torch.device("cpu"), only_targets=["C_dyn_crossing"])
    raw = C9B.run_eval(cfg, torch.device("cpu"), only_targets=["C_dyn_crossing"])
    rows = list(csv.DictReader(open(raw, newline="")))
    assert rows
    provs = {r["provider"] for r in rows}
    assert "euclid" in provs and "oracle" in provs
    assert any(p.startswith("zeroshot_scalar_hrm_aware") for p in provs)
    assert any(p.startswith("lora_scalar_hrm_blind") for p in provs)
    for r in rows:
        assert r["target"] == "C_dyn_crossing" and r["method"] and r["awareness"] in ("aware","blind","")
