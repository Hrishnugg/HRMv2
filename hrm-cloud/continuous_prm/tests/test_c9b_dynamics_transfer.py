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


# -----------------------------------------------------------------------------
# Task 9 — analyze mode (synthetic, CPU-only, no models)
# -----------------------------------------------------------------------------
#
# Builds a synthetic raw CSV by hand using the EXACT C9B_RAW_COLS schema (no
# eval/training involved) for 1 target (C_dyn_crossing), 1 backbone
# (scalar_hrm), both awareness arms, euclid + the 4 C9b method arms, K in
# {1, 16} for the trained arms (zero_shot fixed at K=0), at a single budget
# (150) over a handful of worlds. full_ft is constructed so that at K=16
# "aware" beats "blind" (lower expansions AND higher success), giving the
# probe a clean positive cell to detect; C_dyn_crossing itself is the time-
# coupling CONTROL (see c8-dynamics memory), so elsewhere (zero_shot/lora/
# scratch, and full_ft at K=1) aware and blind are built to be a ~tie.

def _write_c9b_raw_row(wri, **kw):
    row = {k: "" for k in C9B.C9B_RAW_COLS}
    row.update(kw)
    wri.writerow(row)


def _build_synthetic_c9b_raw(path):
    import csv as _csv
    target = "C_dyn_crossing"
    backbone = "scalar_hrm"
    budget = 150
    n_worlds = 6
    euclid_exp = {wi: 100.0 + 5.0 * wi for wi in range(n_worlds)}

    with open(path, "w", newline="") as f:
        wri = _csv.DictWriter(f, fieldnames=C9B.C9B_RAW_COLS)
        wri.writeheader()

        # --- euclid: always found, defines the matched-ratio denominator ----
        for wi in range(n_worlds):
            _write_c9b_raw_row(
                wri, target=target, backbone="", awareness="", method="euclid", K=-1,
                seed=-1, world_index=wi, provider="euclid", mode="astar", w="",
                budget=budget, found=True, expansions=euclid_exp[wi], arrival=10.0,
                optimal_arrival=10.0, suboptimality=1.0, closed=20, nonfinite=0,
            )

        # --- oracle: also always found (not used by the probe but part of the
        # real schema; included so analyze code that scans all rows is exercised
        # against a representative file). ------------------------------------
        for wi in range(n_worlds):
            _write_c9b_raw_row(
                wri, target=target, backbone="", awareness="", method="oracle", K=-1,
                seed=-1, world_index=wi, provider="oracle", mode="astar", w="",
                budget=budget, found=True, expansions=euclid_exp[wi] * 0.5, arrival=9.0,
                optimal_arrival=9.0, suboptimality=1.0, closed=15, nonfinite=0,
            )

        def arm_block(method, K, aware_ratio, blind_ratio, aware_succ_n, blind_succ_n,
                       provider_suffix):
            # aware_ratio/blind_ratio: matched expansion-ratio (vs euclid) on solved worlds.
            # aware_succ_n/blind_succ_n: number of worlds (out of n_worlds) that are FOUND.
            for awareness, ratio, succ_n in (
                ("aware", aware_ratio, aware_succ_n), ("blind", blind_ratio, blind_succ_n)
            ):
                provider = f"{provider_suffix}_{backbone}_{awareness}_K{K}_s0"
                for wi in range(n_worlds):
                    found = wi < succ_n
                    exp = euclid_exp[wi] * ratio if found else ""
                    _write_c9b_raw_row(
                        wri, target=target, backbone=backbone, awareness=awareness,
                        method=method, K=K, seed=0, world_index=wi, provider=provider,
                        mode="astar", w="", budget=budget, found=found,
                        expansions=exp, arrival=(11.0 if found else ""),
                        optimal_arrival=10.0, suboptimality=(1.1 if found else ""),
                        closed=25, nonfinite=0,
                    )

        # zero_shot: K fixed at 0, aware ~= blind (tie; control stays a tie).
        arm_block("zero_shot", 0, 0.9, 0.9, n_worlds, n_worlds, "zeroshot")

        # lora: K in {1, 16}, aware ~= blind throughout.
        for K in (1, 16):
            arm_block("lora", K, 0.8, 0.8, n_worlds, n_worlds, "lora")

        # scratch: K in {1, 16}, much worse than euclid (ratio > 1), aware ~= blind.
        for K in (1, 16):
            arm_block("scratch", K, 1.5, 1.5, n_worlds - 2, n_worlds - 2, "scratch")

        # full_ft: the headline cell. K=1 aware~=blind (tie, matches the control's
        # expectation pre-adaptation); K=16 aware BEATS blind (lower ratio + more
        # successes) -- this is the pre-registered positive the probe should flag.
        arm_block("full_ft", 1, 1.0, 1.0, n_worlds - 1, n_worlds - 1, "full_ft")
        arm_block("full_ft", 16, 0.6, 1.1, n_worlds, n_worlds - 2, "full_ft")

    return path


def test_analyze_c9b(tmp_path):
    raw = _build_synthetic_c9b_raw(tmp_path / "c9b_raw.csv")
    res = tmp_path / "results"
    out = C9B.analyze_from_raw_c9b(
        raw, res, seed=1,
        targets=["C_dyn_crossing"], backbones=["scalar_hrm"], awareness=["aware", "blind"],
    )
    assert set(out) == {"curves", "comparisons", "significance", "probe"}
    for p in out.values():
        assert Path(p).exists()

    import csv as _csv
    curve_rows = list(_csv.DictReader(open(out["curves"], newline="")))
    assert curve_rows
    expected_cols = ["target", "backbone", "awareness", "arm", "K", "binding_budget",
                      "n_matched", "exp_ratio_median", "ci_lo", "ci_hi", "success"]
    assert list(curve_rows[0].keys()) == expected_cols
    # full_ft K=16 aware row should show a lower exp_ratio_median than blind.
    aware16 = next(r for r in curve_rows if r["arm"] == "full_ft" and r["K"] == "16" and r["awareness"] == "aware")
    blind16 = next(r for r in curve_rows if r["arm"] == "full_ft" and r["K"] == "16" and r["awareness"] == "blind")
    assert float(aware16["exp_ratio_median"]) < float(blind16["exp_ratio_median"])
    assert float(aware16["success"]) > float(blind16["success"])

    comp_text = Path(out["comparisons"]).read_text(encoding="utf-8")
    assert "C_dyn_crossing" in comp_text and "scalar_hrm" in comp_text

    sig_text = Path(out["significance"]).read_text(encoding="utf-8")
    assert "aware" in sig_text and "blind" in sig_text

    probe_text = Path(out["probe"]).read_text(encoding="utf-8")
    # the headline full_ft K=16 aware-vs-blind cell must be named explicitly.
    assert "full_ft" in probe_text and "16" in probe_text
    assert "C_dyn_crossing" in probe_text


# -----------------------------------------------------------------------------
# Task 10 — full mode + CLI + scale presets (CPU smoke, end-to-end)
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not (HERE/"runs/c8_local_heavy/checkpoints/c8_scalar__hrm.pt").exists(), reason="c8 sources missing")
def test_run_full_smoke(tmp_path):
    import torch
    cfg = C9B.C9bConfig(out_dir=str(tmp_path/"c9b"), mode="full", scale="smoke", cpu=True, seed=7)
    cfg = C9B.apply_scale(cfg)
    res = C9B.run_full(cfg, torch.device("cpu"))
    for key in ("curves", "comparisons", "significance", "probe"):
        assert Path(res[key]).exists()
