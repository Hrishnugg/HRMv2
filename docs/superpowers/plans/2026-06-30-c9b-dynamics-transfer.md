# C9b — Few-shot Transfer under Dynamics: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `continuous_prm_c9b_dynamics_transfer.py` — adapt the frozen C8 pooled space-time heuristics (aware + blind, 3 backbones) to held-out **dynamic** families with zero_shot / LoRA / full-FT / from-scratch, and measure (1) whether the C9/C9h crossover reproduces under dynamics and (2) whether adaptation flips C8's time-aware-vs-blind negative.

**Architecture:** New-file-only. The temporal **feature builders, providers, space-time eval, oracle, suite installer, calibrate** (C8) and the **ADAPT/TEST split + analysis/stats** (C9) are reused directly. The one thing C8 does *not* expose reusably is a trainable loop with `init_ckpt`+LoRA injection — so C9b writes its own **temporal trainers** (scalar + field) that load a C8 source, apply `C.apply_lora` / `C9h.apply_conv_lora` (or full-FT / scratch), and train at the C9h matched recipe (epochs 10, lr 2e-4). Awareness = W>0 vs W=0 inputs, carried through every arm.

**Tech Stack:** Python, PyTorch, NumPy. Reuse modules (import as shown): `continuous_prm_common` (`C`), `continuous_prm_c8_dynamics_compare` (`M8`), `continuous_prm_c8_dynamic_maps` (`M8MAPS`), `continuous_prm_dynamic_providers` (`DP`), `continuous_prm_spacetime` (`ST`), `continuous_prm_dynamics` (`D`), `continuous_prm_c9_transfer` (`C9`), `continuous_prm_c9h_transfer` (`C9H`).

**Conventions:** Windows; `python` / `python -m pytest`. Branch `c9b-dynamics-transfer` (already created). NEVER stage `continuous_prm_common.py` / `transfer_astar_heuristic_clean_parallel_fixed.py` (user WIP). Stage only the new C9b files + runs/ text artifacts. Seeded RNG. Do NOT modify C8/C9/C9h/common.

**Verified API (from the Explore pass — confirm by reading source before trusting):**
- `M8MAPS.install_c8_dynamic_maps() -> None` (c8_dynamic_maps.py:473); after it, `C.build_anchor_specs()` returns specs incl. `C_dyn_{maze,rooms,spiral,maze_dense,crossing,rooms_large}`.
- `M8MAPS.build_dynamic_world(suite, seed) -> Optional[(C.World, D.Dynamics)]` (:450); `M8MAPS.dynamics_params(suite) -> {v_agent, dt, t_max, ...}` (:214).
- `M8._collect_world_labels(suite, seed, cfg) -> dict|None` (c8_dynamics_compare.py:174) with keys `world, rm, dyn, params, ttg (N,t_max+1), node_residual (N,t_max+1), reachable (N,t_max+1) bool`. Uses `ST.backward_spacetime_dijkstra(rm.adj, rm.points, dyn, v_agent, dt, t_max, goal=1)` + `ST.oracle_time_to_go(hstar, t_max)`.
- `DP.build_scalar_temporal_features(world, roadmap, dyn, t_max, dt, window_w, k_patrollers, goal_idx=1) -> np.ndarray (N, t_max+1, W+1, token_dim)`, token_dim = 4 + 4*K (c9b uses k_patrollers from the source ckpt). W=0 → seq dim 1 (blind).
- `DP.build_field_occupancy_stack(world, dyn, grid_size, t, window_w, dt, static_base=None) -> np.ndarray (8+W, G, G)`.
- `DP.ScalarTemporalProvider(model, device, backbone, window_w, k_patrollers, max_norm_residual, time_blind=False)`; `DP.ValueFieldTemporalProvider(model, grid_size, device, backbone, window_w, max_norm_residual=4.0, time_blind=False)`; `DP.EuclidTimeProvider()`; `DP.OracleProvider()`. Each `.h_table(world, roadmap, dyn, v_agent, dt, t_max, goal_idx=1) -> (N, t_max+1)`; `.name` like `scalar_hrm` / `scalar_hrm_blind`.
- `DP.run_world_arms_spacetime(world, roadmap, dyn, providers: dict, budgets, w_values, v_agent, dt, t_max, goal_idx=1, start_idx=0) -> list[dict]` with keys `provider, mode, found, expansions, arrival, optimal_arrival, suboptimality, closed, nonfinite, budget, w`.
- C8 source payloads — scalar: `{model, backbone_cfg, window_w, k_patrollers, token_dim, max_norm_residual, backbone}`; field: `{model, in_channels(=8+W), window_w, grid_size, backbone}`.
- C8 training loops to MIRROR: `M8._train_scalar(labelsets, cfg, device, window_w=None, suffix="")` (:348) and `M8._train_field(...)` (:584) — read these for the exact model-build, smooth-L1-on-reachable-mask, optimizer, and save; C9b reuses the *body* but adds `init_ckpt` load + `apply_lora`/full-FT/scratch + recipe params.
- `C.apply_lora(model, rank, alpha, init_scale=0.01) -> int`; `C9H.apply_conv_lora(unet, rank, alpha, init_scale=0.01) -> int`; `C.iter_lora_targets(model)`; `C.build_model(...)` / the scalar backbone builder used by `_train_scalar`; `C.safe_load_state(model, path)`.
- C9 reuse: `C9.world_fingerprint(world)` (:83), `C9.adapt_seed(target,K,idx,base_seed)` (:97), `C9._parse_csv/_parse_ints` (:214), `C9._binding_budget_for(rows,target,euclid_floor=0.05)` (:347), `C9._astar/_is_found/_euclid_exp_by_world`, `C9.analyze_from_raw` pattern + stats imports `mcnemar_exact_p`/`bh_q_values` (C6), `bootstrap_median_ci` (C7).
- C8 source checkpoints (frozen): `runs/c8_local_heavy/checkpoints/c8_{scalar__hrm,scalar__onlstm,field__unet}{,_blind}.pt`.

---

## File structure

| File | Responsibility |
|---|---|
| Create `hrm-cloud/continuous_prm/continuous_prm_c9b_dynamics_transfer.py` | `C9bConfig`; source resolution; temporal dataset collection; temporal trainers (scalar+field) with init_ckpt+LoRA/full-FT/scratch; provider loaders; ADAPT/TEST split; adapt/eval/analyze/full + CLI. |
| Create `hrm-cloud/continuous_prm/tests/test_c9b_dynamics_transfer.py` | suites install + source paths; ADAPT⊥TEST; temporal dataset shapes (aware/blind, scalar/field); trainer round-trip + LoRA-applied; provider load + forward; eval smoke; analyze (synthetic). |
| Reuse (no edits) | C8 (`_dynamics_compare`, `_dynamic_maps`), `dynamic_providers`, `spacetime`, `dynamics`, C9, C9h, common. |
| Output `runs/c9b_local/{checkpoints,datasets,results}` | adapters, raw/curves/comparisons/significance/probe MD, manifests. |

---

## Task 1: Module skeleton + C9bConfig + source resolution + suites

**Files:** Create `continuous_prm_c9b_dynamics_transfer.py`; Test `tests/test_c9b_dynamics_transfer.py`.

- [ ] **Step 1: failing test**
```python
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parents[1]
import numpy as np
import continuous_prm_c9b_dynamics_transfer as C9B
import continuous_prm_common as C


def test_config_sources_and_suites():
    cfg = C9B.C9bConfig()
    assert cfg.backbones == "scalar_hrm,scalar_onlstm,field_unet"
    assert cfg.targets == "C_dyn_maze_dense,C_dyn_crossing,C_dyn_rooms_large"
    assert set(cfg.awareness_list()) == {"aware", "blind"}
    # source resolution: 3 backbones x {aware,blind}
    srcs = C9B.resolve_sources(cfg)
    assert set(srcs) == {(b, a) for b in ("scalar_hrm", "scalar_onlstm", "field_unet") for a in ("aware", "blind")}
    # suites install + target specs exist
    C9B.install()
    specs = C.build_anchor_specs()
    for t in C9B._parse_csv(cfg.targets):
        assert t in specs
```

- [ ] **Step 2: run, expect fail** (`ModuleNotFoundError`). `python -m pytest hrm-cloud/continuous_prm/tests/test_c9b_dynamics_transfer.py::test_config_sources_and_suites -q`

- [ ] **Step 3: implement** the header + config + helpers:
```python
#!/usr/bin/env python3
"""C9b: few-shot transfer under dynamics. Adapt frozen C8 pooled space-time
heuristics (aware + blind) to held-out dynamic families via zero_shot/LoRA/full_ft/scratch.
New-file-only; reuses C8 + C9/C9h + common. See docs/superpowers/specs/2026-06-30-c9b-dynamics-transfer-design.md.
"""
from __future__ import annotations
import argparse, dataclasses, csv as _csv, json, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import continuous_prm_common as C
import continuous_prm_c8_dynamics_compare as M8
import continuous_prm_c8_dynamic_maps as M8MAPS
import continuous_prm_dynamic_providers as DP
import continuous_prm_spacetime as ST
import continuous_prm_dynamics as D
import continuous_prm_c9_transfer as C9
import continuous_prm_c9h_transfer as C9H

BACKBONES = ["scalar_hrm", "scalar_onlstm", "field_unet"]   # field_unet => field substrate; others scalar
AWARENESS = ["aware", "blind"]


def install():
    M8MAPS.install_c8_dynamic_maps()


def _parse_csv(s): return C9._parse_csv(s)
def _parse_ints(s): return C9._parse_ints(s)
def now_str() -> str: return C.now_str()


@dataclass
class C9bConfig:
    source_dir: str = "hrm-cloud/continuous_prm/runs/c8_local_heavy"
    out_dir: str = "hrm-cloud/continuous_prm/runs/c9b_local"
    backbones: str = "scalar_hrm,scalar_onlstm,field_unet"
    targets: str = "C_dyn_maze_dense,C_dyn_crossing,C_dyn_rooms_large"
    awareness: str = "aware,blind"
    methods: str = "lora,full_ft,scratch"      # zero_shot handled separately (frozen source)
    k_grid: str = "1,4,16"
    n_adapt_seeds: int = 3
    n_test: int = 20
    rank: int = 8
    alpha: float = 1.0
    epochs: int = 10
    lr: float = 2.0e-4
    grid_size: int = 64
    budgets: str = ""                          # empty => from source_dir/calibration.json
    w_values: str = "1.0,1.1"
    seed: int = 1234
    scale: str = "local"
    mode: str = "full"
    cpu: bool = False
    retrain_sources: bool = False

    def awareness_list(self): return _parse_csv(self.awareness)


def _src_ckpt(cfg: C9bConfig, backbone: str, awareness: str) -> Path:
    # C8 names: c8_scalar__hrm.pt / c8_field__unet_blind.pt ...
    sub, bb = backbone.split("_", 1)          # "scalar","hrm" | "field","unet"
    suffix = "_blind" if awareness == "blind" else ""
    return Path(cfg.source_dir) / "checkpoints" / f"c8_{sub}__{bb}{suffix}.pt"


def resolve_sources(cfg: C9bConfig) -> Dict[Tuple[str, str], Path]:
    out = {}
    for b in _parse_csv(cfg.backbones):
        for a in cfg.awareness_list():
            out[(b, a)] = _src_ckpt(cfg, b, a)
    return out


def _is_field(backbone: str) -> bool: return backbone.startswith("field")
```

- [ ] **Step 4: run, expect pass.** `python -m pytest hrm-cloud/continuous_prm/tests/test_c9b_dynamics_transfer.py -q`. If `dynamics_params`/`build_anchor_specs` keys differ, adapt and report.

- [ ] **Step 5: commit**
```bash
git add hrm-cloud/continuous_prm/continuous_prm_c9b_dynamics_transfer.py hrm-cloud/continuous_prm/tests/test_c9b_dynamics_transfer.py
git commit -m "feat(c9b): module skeleton + C9bConfig + source resolution + suite install"
```

---

## Task 2: ADAPT/TEST split + disjointness (space-time generator)

**Files:** Modify module + test. Deterministic per-target ADAPT and TEST world lists from the C8 dynamic generator, provably disjoint.

- [ ] **Step 1: failing test**
```python
def test_adapt_test_disjoint():
    C9B.install()
    cfg = C9B.C9bConfig(n_test=4, seed=7)
    adapt = C9B.adapt_world_seeds("C_dyn_crossing", K=4, seed_idx=0, cfg=cfg)
    test = C9B.test_world_seeds("C_dyn_crossing", cfg)
    assert len(adapt) == 4 and len(test) == 4
    assert set(adapt).isdisjoint(set(test))   # disjoint world seeds => disjoint worlds
    # fingerprints of built worlds are disjoint too
    fa = {C9.world_fingerprint(C9B._build_world_only("C_dyn_crossing", s)) for s in adapt}
    ft = {C9.world_fingerprint(C9B._build_world_only("C_dyn_crossing", s)) for s in test}
    assert fa.isdisjoint(ft)
```

- [ ] **Step 2: RED. Step 3: implement** — TEST seeds drawn from a fixed high block, ADAPT seeds from a disjoint per-(K,idx) block via `C9.adapt_seed`; both filtered to seeds that build a valid solvable world (mirror C8's acceptance). Read `M8._collect_world_labels` for the acceptance rule (world not None, rm connected, reachable cells exist); reuse it lightly via a build-only helper:
```python
def _build_world_only(suite: str, seed: int):
    res = M8MAPS.build_dynamic_world(suite, int(seed))
    return None if res is None else res[0]


def _valid_world_seed(suite: str, seed: int) -> bool:
    res = M8MAPS.build_dynamic_world(suite, int(seed))
    if res is None:
        return False
    world, dyn = res
    rm = C.build_prm(world, C.RoadmapConfig(n_nodes=192, k_neighbors=7), seed=int(seed) + 17)
    return rm is not None and bool(rm.connected_to_goal[0])


def test_world_seeds(target: str, cfg: C9bConfig) -> List[int]:
    rng = np.random.default_rng(10_000_000 + (int(hashlib.md5(target.encode()).hexdigest()[:6], 16) % 1_000_000))
    out, tries = [], 0
    while len(out) < cfg.n_test and tries < cfg.n_test * 200:
        s = int(rng.integers(0, 2**31 - 1)); tries += 1
        if _valid_world_seed(target, s):
            out.append(s)
    return out


def adapt_world_seeds(target: str, K: int, seed_idx: int, cfg: C9bConfig) -> List[int]:
    base = C9.adapt_seed(target, K, seed_idx, cfg.seed)
    rng = np.random.default_rng(base)
    test_set = set(test_world_seeds(target, cfg))
    out, tries = [], 0
    while len(out) < K and tries < K * 400:
        s = int(rng.integers(0, 2**31 - 1)); tries += 1
        if s in test_set:
            continue
        if _valid_world_seed(target, s):
            out.append(s)
    return out
```

- [ ] **Step 4: GREEN. Step 5: commit** `feat(c9b): ADAPT/TEST world split + disjointness for dynamic generator`.

---

## Task 3: Temporal dataset collection (aware + blind, scalar + field)

**Files:** Modify module + test. Build a K-shot ADAPT dataset of temporal features + space-time-oracle residual targets, for a given backbone (scalar or field) and awareness (W or 0).

- [ ] **Step 1: failing test**
```python
@pytest.mark.skipif(not (HERE/"runs/c8_local_heavy/checkpoints/c8_scalar__hrm.pt").exists(), reason="c8 sources missing")
def test_temporal_dataset_shapes(tmp_path):
    C9B.install()
    cfg = C9B.C9bConfig(out_dir=str(tmp_path/"c9b"))
    seeds = C9B.adapt_world_seeds("C_dyn_crossing", K=2, seed_idx=0, cfg=cfg)
    # scalar aware (W from source) and blind (W=0)
    sa = C9B.collect_temporal_dataset("C_dyn_crossing", seeds, backbone="scalar_hrm", window_w=8, k_patrollers=2, grid_size=64, out_npz=tmp_path/"sa.npz")
    sb = C9B.collect_temporal_dataset("C_dyn_crossing", seeds, backbone="scalar_hrm", window_w=0, k_patrollers=2, grid_size=64, out_npz=tmp_path/"sb.npz")
    import numpy as np
    A = np.load(sa); B = np.load(sb)
    assert A["x"].ndim == 3 and A["x"].shape[1] == 9      # (M, W+1=9, token_dim)
    assert B["x"].shape[1] == 1                            # blind seq dim 1
    assert A["y"].shape[0] == A["x"].shape[0] and A["x"].shape[0] > 0
```

- [ ] **Step 2: RED. Step 3: implement** — for each adapt world: `lab = M8._collect_world_labels(target, seed, _c8cfg(cfg))`; build features via `DP.build_scalar_temporal_features(...)` or `DP.build_field_occupancy_stack(...)` per t; flatten to per-(node,t) samples masked by `lab["reachable"]`, target = `lab["node_residual"]`. Read `M8._train_scalar`/`_train_field` (c8_dynamics_compare.py:348/584) for the EXACT feature→sample flattening + reachable masking they use at train time and mirror it so the adapter sees the same representation as the source. Provide `_c8cfg(cfg)` that builds an `M8.C8Config` with the right grid_size/k_patrollers/window. Save npz keys `x`, `y` (+ `meta` window_w/backbone). Scalar x = (M, W+1, token_dim); field x = (M, 8+W, G, G) with per-sample (node→? no: field is per (world,t) grid). NOTE: field datasets are per-(world,t) grids with the residual *field* as target — mirror `_train_field`'s dataset exactly (it may store grids + sampled-node targets); do not invent a schema.

- [ ] **Step 4: GREEN (shapes match what the trainers/providers expect). Step 5: commit** `feat(c9b): temporal dataset collection (aware/blind, scalar/field)`.

---

## Task 4: Scalar temporal trainer (init_ckpt + lora/full_ft/scratch + bounded)

**Files:** Modify module + test. Mirror `M8._train_scalar`'s model-build + loss loop; add source load, method branch, recipe params.

- [ ] **Step 1: failing test**
```python
@pytest.mark.skipif(not (HERE/"runs/c8_local_heavy/checkpoints/c8_scalar__hrm.pt").exists(), reason="c8 sources missing")
def test_scalar_trainer_methods(tmp_path):
    C9B.install(); import torch
    cfg = C9B.C9bConfig(out_dir=str(tmp_path/"c9b"), epochs=1, cpu=True)
    seeds = C9B.adapt_world_seeds("C_dyn_crossing", K=2, seed_idx=0, cfg=cfg)
    npz = C9B.collect_temporal_dataset("C_dyn_crossing", seeds, backbone="scalar_hrm", window_w=8, k_patrollers=2, grid_size=64, out_npz=tmp_path/"d.npz")
    src = HERE/"runs/c8_local_heavy/checkpoints/c8_scalar__hrm.pt"
    for method in ("lora", "full_ft", "scratch"):
        ck = C9B.train_scalar_temporal(npz, tmp_path/f"{method}.pt", source_ckpt=(src if method!="scratch" else None),
                                       method=method, cfg=cfg, device=torch.device("cpu"), seed=0)
        payload = torch.load(ck, map_location="cpu")
        assert payload["window_w"] == 8 and payload["method"] == method
        if method == "lora":
            assert payload["lora_rank"] == 8
```

- [ ] **Step 2: RED. Step 3: implement** `train_scalar_temporal(dataset_npz, out_ckpt, source_ckpt, method, cfg, device, seed) -> Path`:
  - Load source payload (for `backbone_cfg`, `window_w`, `k_patrollers`, `token_dim`, `max_norm_residual`).
  - Build the scalar model exactly as `M8._train_scalar` does (same backbone builder + token_dim/seq_len). Read :348 for the constructor.
  - `method=="scratch"`: random init (no source load). Else `C.safe_load_state(model, source_ckpt)`.
  - `method=="lora"`: `C.apply_lora(model, rank=cfg.rank, alpha=cfg.alpha)`, freeze non-LoRA (mirror `C9H.train_scalar_lora` :82 for the freeze + the bounded handling — bounded=True here, finite `max_norm_residual`).
  - Train: AdamW(lr=cfg.lr), smooth-L1 on the reachable-masked residual, `cfg.epochs` epochs, grad-clip — mirror `M8._train_scalar`'s loop body.
  - Save payload: `{model, backbone_cfg, window_w, k_patrollers, token_dim, max_norm_residual, backbone, method, lora_rank(if lora), alpha}` so the provider loader (Task 6) can reconstruct.

- [ ] **Step 4: GREEN. Step 5: commit** `feat(c9b): scalar temporal trainer (source/LoRA/full-FT/scratch, matched recipe)`.

---

## Task 5: Field temporal trainer (conv-LoRA + full_ft + scratch)

**Files:** Modify module + test. Mirror `M8._train_field`; add source load + conv-LoRA/full-FT/scratch.

- [ ] **Step 1: failing test**
```python
@pytest.mark.skipif(not (HERE/"runs/c8_local_heavy/checkpoints/c8_field__unet.pt").exists(), reason="c8 sources missing")
def test_field_trainer_methods(tmp_path):
    C9B.install(); import torch
    cfg = C9B.C9bConfig(out_dir=str(tmp_path/"c9b"), epochs=1, cpu=True)
    seeds = C9B.adapt_world_seeds("C_dyn_crossing", K=2, seed_idx=0, cfg=cfg)
    npz = C9B.collect_temporal_dataset("C_dyn_crossing", seeds, backbone="field_unet", window_w=8, k_patrollers=2, grid_size=64, out_npz=tmp_path/"f.npz")
    src = HERE/"runs/c8_local_heavy/checkpoints/c8_field__unet.pt"
    for method in ("lora", "full_ft", "scratch"):
        ck = C9B.train_field_temporal(npz, tmp_path/f"{method}.pt", source_ckpt=(src if method!="scratch" else None),
                                      method=method, cfg=cfg, device=torch.device("cpu"), seed=0)
        p = torch.load(ck, map_location="cpu")
        assert p["in_channels"] == 16 and p["window_w"] == 8 and p["method"] == method
```

- [ ] **Step 2: RED. Step 3: implement** `train_field_temporal(dataset_npz, out_ckpt, source_ckpt, method, cfg, device, seed) -> Path`:
  - Load source payload for `in_channels (8+W)`, `window_w`, `grid_size`.
  - Build the U-Net exactly as `M8._train_field` (:584) builds it with `in_channels=8+W`.
  - `scratch`: random init; else `C.safe_load_state(model, source_ckpt)`.
  - `lora`: `C9H.apply_conv_lora(model, rank=cfg.rank, alpha=cfg.alpha)`, freeze non-LoRA (mirror `C9H.train_field_model` :158).
  - Train mirroring `M8._train_field`'s loss loop (smooth-L1 / the C6-style field loss it uses — read it; do not substitute a different loss), AdamW(lr=cfg.lr), cfg.epochs.
  - Save `{model, in_channels, window_w, grid_size, backbone, method, lora_rank(if lora), alpha}`.

- [ ] **Step 4: GREEN. Step 5: commit** `feat(c9b): field temporal trainer (source/conv-LoRA/full-FT/scratch)`.

---

## Task 6: Provider loaders (scalar + field, aware/blind, LoRA-aware)

**Files:** Modify module + test.

- [ ] **Step 1: failing test**
```python
@pytest.mark.skipif(not (HERE/"runs/c8_local_heavy/checkpoints/c8_scalar__hrm.pt").exists(), reason="c8 sources missing")
def test_provider_loaders(tmp_path):
    C9B.install(); import torch
    dev = torch.device("cpu")
    src = HERE/"runs/c8_local_heavy/checkpoints/c8_scalar__hrm.pt"
    prov = C9B.load_temporal_provider(src, backbone="scalar_hrm", device=dev)   # frozen source => zero_shot arm
    assert prov.name.startswith("scalar_hrm")
    # smoke a forward on one crossing world
    seeds = C9B.test_world_seeds("C_dyn_crossing", C9B.C9bConfig(n_test=1))
    w, dyn = M8MAPS.build_dynamic_world("C_dyn_crossing", seeds[0])
    rm = C.build_prm(w, C.RoadmapConfig(n_nodes=192, k_neighbors=7), seed=seeds[0]+17)
    pp = M8MAPS.dynamics_params("C_dyn_crossing")
    ht = prov.h_table(w, rm, dyn, pp["v_agent"], pp["dt"], int(pp["t_max"]))
    assert ht.shape[0] == rm.points.shape[0]
```

- [ ] **Step 2: RED. Step 3: implement** `load_temporal_provider(ckpt, backbone, device, bounded=True) -> provider`:
  - Load payload; detect awareness from `window_w` (>0 aware else blind) → `time_blind = (window_w == 0)`.
  - Scalar: build model (backbone_cfg/token_dim), `if "lora_rank" in payload: C.apply_lora(...)`, `model.load_state_dict(payload["model"])`, return `DP.ScalarTemporalProvider(model, device, bb, window_w, k_patrollers, max_norm_residual, time_blind=time_blind)`.
  - Field: build U-Net (in_channels), `if "lora_rank" in payload: C9H.apply_conv_lora(...)`, load state, return `DP.ValueFieldTemporalProvider(model, grid_size, device, bb, window_w, max_norm_residual=(4.0 if bounded else float("inf")), time_blind=time_blind)`.
  - (Mirror C8 `_load_eval_providers` :836 for the exact build+load order.)

- [ ] **Step 4: GREEN. Step 5: commit** `feat(c9b): temporal provider loaders (LoRA-aware, aware/blind)`.

---

## Task 7: adapt mode + manifest

**Files:** Modify module + test. Per (target × backbone × awareness × method × K × seed): collect dataset, train adapter, record. zero_shot is the frozen source (no training).

- [ ] **Step 1: failing smoke test**
```python
@pytest.mark.skipif(not (HERE/"runs/c8_local_heavy/checkpoints/c8_scalar__hrm.pt").exists(), reason="c8 sources missing")
def test_run_adapt_smoke(tmp_path):
    import torch
    cfg = C9B.C9bConfig(out_dir=str(tmp_path/"c9b"), backbones="scalar_hrm", awareness="aware,blind",
                        methods="lora,scratch", k_grid="1", n_adapt_seeds=1, n_test=4, epochs=1, cpu=True, seed=7)
    man = C9B.run_adapt(cfg, torch.device("cpu"), only_targets=["C_dyn_crossing"])
    # 1 target x 1 bb x 2 awareness x 2 methods x 1 K x 1 seed = 4 adapters
    assert len(man["arms"]) == 4
    for a in man["arms"]:
        assert Path(a["ckpt"]).exists() and a["awareness"] in ("aware","blind") and a["method"] in ("lora","scratch")
```

- [ ] **Step 2: RED. Step 3: implement** `run_adapt(cfg, device, only_targets=None) -> dict`: loop targets→backbones→awareness→K→seed→method; `window_w = src_window(backbone, awareness)` (read from the source payload: aware→source W, blind→0); collect dataset once per (target,backbone,awareness,K,seed) and reuse across methods; dispatch `train_scalar_temporal`/`train_field_temporal` by `_is_field`; arm ckpt path `c9b__{target}__{backbone}__{awareness}__{method}__K{K}__s{seed}.pt`; `if not ck.exists()` guard (resumable). Write `adapt_manifest.json` with arms (target, backbone, awareness, method, K, seed, ckpt, window_w) + the source paths for the zero_shot arm.

- [ ] **Step 4: GREEN. Step 5: commit** `feat(c9b): adapt mode (temporal adapters per target×backbone×awareness×method×K×seed) + manifest`.

---

## Task 8: eval mode (space-time arms over budget grid)

**Files:** Modify module + test. Per target: build providers = euclid + oracle + per (backbone,awareness) frozen zero_shot source + every adapted arm; eval on TEST worlds over the budget grid; raw CSV.

- [ ] **Step 1:** define `C9B_RAW_COLS = ["target","backbone","awareness","method","K","seed","world_index","provider","mode","w","budget","found","expansions","arrival","optimal_arrival","suboptimality","closed","nonfinite"]`.

- [ ] **Step 2: smoke test** — after `run_adapt` smoke, `run_eval(cfg, only_targets=["C_dyn_crossing"])` (n_test=4, budgets="150,250"); assert rows exist, every row has target+backbone+awareness+method, and provider set includes `euclid`, `oracle`, a `zero_shot` arm and a `lora` arm.

- [ ] **Step 3: implement** `run_eval(cfg, device, only_targets=None) -> Path` mirroring C8 eval + C9 eval:
  - budgets: `_parse_ints(cfg.budgets)` or read `source_dir/calibration.json` (`{suite:{...}}` → per-target binding band; reuse C8's selection or `C9._binding_budget_for` later in analyze — for eval just run the full per-target budget list).
  - For each target: materialize TEST worlds once (`test_world_seeds` → build world+dyn+rm); for each (backbone, awareness): load the frozen source as `zero_shot` provider (name `zeroshot_{backbone}_{awareness}`), and each adapted arm via `load_temporal_provider` (name `{method}_{backbone}_{awareness}_K{K}_s{seed}`). Stamp meta dict per provider (method, backbone, awareness, K, seed). Always add `DP.EuclidTimeProvider()`, `DP.OracleProvider()`.
  - Eval each world: `DP.run_world_arms_spacetime(world, rm, dyn, providers, budgets, w_values, pp["v_agent"], pp["dt"], int(pp["t_max"]))`; `r.update(target=..., world_index=..., **meta)`; write shard + merged `results/continuous_prm_c9b_eval_raw.csv` with `C9B_RAW_COLS`.
  - NOTE the eval is large; print per-target progress with `now_str()`.

- [ ] **Step 4: GREEN. Step 5: commit** `feat(c9b): eval mode (space-time arms incl frozen zero_shot, over budget grid)`.

---

## Task 9: analyze mode (crossover curves + aware-vs-blind success-composite probe)

**Files:** Modify module + test. Reuse C9's binding-budget + matched-ratio logic; add the awareness dimension and the success composite.

- [ ] **Step 1: implement** `analyze_from_raw_c9b(raw_csv, out_dir, seed, targets, backbones, awareness) -> dict`:
  - binding budget per target via `C9._binding_budget_for(rows, target)` (euclid rows; the col is `provider=="euclid"`, `mode=="astar"`).
  - **Replication curves:** for each (target, backbone, awareness, method ∈ {zero_shot,lora,full_ft,scratch}, K): matched exp-ratio vs euclid (`C9._euclid_exp_by_world` + `bootstrap_median_ci`) + success at the binding budget. Write `continuous_prm_c9b_curves.csv` (cols target,backbone,awareness,method,K,binding_budget,n_matched,exp_ratio_median,ci_lo,ci_hi,success).
  - **Crossover comparisons MD:** per (target,backbone,awareness) the K-curve for lora/full_ft/scratch vs zero_shot (does full_ft overtake by K≥4–16; lora flat; both ≫ scratch at low K).
  - **Significance MD:** McNemar+BH of each arm vs euclid at the binding budget (reuse `mcnemar_exact_p`/`bh_q_values`).
  - **Probe MD (the headline):** per (target,backbone,method,K) compare **aware vs blind** with a SUCCESS COMPOSITE: success delta (aware−blind over all TEST worlds) AND matched exp-ratio on the shared-solved set; flag any cell where they disagree; call out whether `full_ft` at K=16 has aware>blind while `C_dyn_crossing` (control) stays a tie. Write `continuous_prm_c9b_probe.md`.

- [ ] **Step 2: test** — synthetic rows (euclid + zero_shot/lora/full_ft/scratch × aware/blind, 1 target, 1 backbone, K∈{1,16}, a few worlds, with full_ft K16 aware better than blind) → assert curves CSV exists, probe.md exists and reports the aware>blind full_ft@K16 cell, comparisons+significance exist.

- [ ] **Step 3: implement + GREEN. Step 4: commit** `feat(c9b): analyze (crossover curves + significance + aware-vs-blind success-composite probe)`.

---

## Task 10: full mode + CLI + scale presets

**Files:** Modify module; Test smoke.

- [ ] **Step 1:** implement `run_full` (optional source check / `--retrain-sources` via a thin wrapper around `M8`'s train path if a source ckpt is missing → else error with a clear message; default reuse), `apply_scale` (`smoke`: 1 target C_dyn_crossing, backbones scalar_hrm, awareness aware,blind, methods lora,scratch, k_grid 1, n_adapt_seeds 1, n_test 4, epochs 1, budgets 150,250, cpu; `local`: defaults), `build_argparser`/`config_from_args`/`main`+`__main__` (flags for every knob incl `--retrain-sources` store_true, `--source-dir`, `--targets`, `--backbones`, `--awareness`, `--methods`, `--k-grid`, `--n-adapt-seeds`, `--n-test`, `--budgets`, `--cpu`).
- [ ] **Step 2: test** `test_run_full_smoke` (cpu) asserts curves+comparisons+probe exist.
- [ ] **Step 3: GREEN. Step 4:** CPU smoke from CLI to `runs/c9b_smoke` then delete it. **Step 5: commit** `feat(c9b): full mode + CLI + scale presets`.

---

## Task 11: local run + C9B_RESULTS + gates

- [ ] **Source check:** confirm the 6 C8 sources exist (`runs/c8_local_heavy/checkpoints/`); if any missing, run with `--retrain-sources` (adds source-training time).
- [ ] **Local run (background, monitor; failure tokens must avoid `cuda`/`oom` substrings):**
```bash
python -u hrm-cloud/continuous_prm/continuous_prm_c9b_dynamics_transfer.py --mode full --scale local \
  --backbones scalar_hrm,scalar_onlstm,field_unet --awareness aware,blind --methods lora,full_ft,scratch \
  --k-grid 1,4,16 --n-adapt-seeds 3 --n-test 20 --w-values 1.0,1.1 \
  --out-dir hrm-cloud/continuous_prm/runs/c9b_local
```
- [ ] **Gates + writeup:** G0 (unit tests green incl LoRA round-trip + ADAPT⊥TEST + aware/blind shapes), G1 (zero_shot beats euclid-time on the 3 held-out dynamic families), G2 (crossover: full_ft worst@K1 best@K≥4–16, lora≈zero_shot, both ≫ scratch — per backbone/awareness), G3 (the probe: aware-vs-blind success composite — does full_ft@K16 flip C8's negative; crossing stays a tie). Write `C9B_RESULTS.md` (arm matrix, crossover verdict, the aware/blind probe verdict honest either way, caveats: matched-ratio-hides-aware-wins handled by composite, source-recipe asymmetry, runtime), cross-link from `C8_RESULTS.md`, `C9H_RESULTS.md`, `CONTINUOUS_PRM_STORY.md`. Commit text artifacts only (curves/comparisons/significance/probe + manifests + the MD + cross-links); confirm WIP unstaged.

---

## Self-review

**Spec coverage:** §1 two questions → G2 (replication) + G3 (probe) in T9/T11; §3 sources (reuse 6 C8 ckpts + retrain fallback) → T1/T10; §4 targets/disjointness/budgets → T2/T8; §5 arms/recipe/grid (486 adapters) → T4/T5/T7; §6 metrics/stats/gates + success composite → T9; §7 architecture (reuse + new temporal trainers) → T3–T6; §8 risks → T11 notes. Covered.

**Placeholder scan:** the heavy training-loop *bodies* (T4/T5) and the feature→sample flattening (T3) are specified as "mirror C8 `_train_scalar`/`_train_field` (exact file:line), add init_ckpt+LoRA+recipe" rather than transcribed line-for-line — deliberate (the implementer reads the source to avoid transcription drift), the same pattern that built C7–C10 successfully; every NEW function has an exact signature, the reuse calls are concrete (file:line), and the tests pin the observable contract (shapes, payload keys, arm counts, provider names). No vague error-handling/validation placeholders.

**Type consistency:** `C9bConfig` fields stable across tasks; `collect_temporal_dataset(target, seeds, backbone, window_w, k_patrollers, grid_size, out_npz)` (T3) consumed by T7; `train_scalar_temporal`/`train_field_temporal(dataset_npz, out_ckpt, source_ckpt, method, cfg, device, seed)` (T4/T5) consumed by T7; `load_temporal_provider(ckpt, backbone, device, bounded=True)` (T6) consumed by T8; arm names `{zero_shot,lora,full_ft,scratch}` × `{aware,blind}` consistent T7→T8→T9; `C9B_RAW_COLS` (T8) read by T9. Consistent.

**Known risk (carried from the Explore pass):** C8's `_train_scalar`/`_train_field` are inlined, so T3–T5 must read those bodies to reproduce the exact dataset schema + loss + model build. If the field dataset is per-(world,t) grids rather than per-sample rows, T3's field branch must store grids (not the scalar (M,…) layout) — the test pins scalar shapes; the field schema is whatever `_train_field` consumes. The T3/T5 implementer must verify the field schema against `_train_field` before finalizing.
