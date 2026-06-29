# C9-hardening (C9h) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `continuous_prm_c9h_transfer.py` — a matched-compute few-shot transfer study that disentangles LoRA (bounded vs unbounded, low-rank vs full-rank) and extends it to the field U-Net via new conv-LoRA.

**Architecture:** New-file-only; reuse C9 (frozen) + C6/C7 field stack + `common` LoRA. conv-LoRA reuses `common.SingleAdapterLoRA` (shape-agnostic) via a new Conv2d target-iterator; matched recipe (E=10, lr=2e-4) applied to every trained arm; arms = {zero_shot, lora_bounded, lora_unbounded, full_ft, scratch} × {hrm, onlstm (scalar), unet (field)} on 3 targets × K{1,4,16} × 3 seeds.

**Tech Stack:** Python, PyTorch, NumPy. Reuse: `continuous_prm_c9_transfer` (alias `C9`), `continuous_prm_common` (`C`), `continuous_prm_providers` (`P`), `continuous_prm_c6_heatmap_value_field` (`C6`), `continuous_prm_c7_hard_maps` (`H7`).

**Conventions:** Windows; `python` / `python -m pytest`. NEVER stage `continuous_prm_common.py` / `transfer_astar_heuristic_clean_parallel_fixed.py` (user WIP). Stage only the new C9h files + runs/ text artifacts (csv/json/md). `runs/.gitignore` excludes `*.pt`/`*.npz`/`figures/`/`_shards/`. Seeded RNG. Do NOT modify C9/C6/C7/common.

**Verified reuse API:**
- `C.SingleAdapterLoRA(base_weight, rank, alpha, init_scale=0.01)` — flattens `base_weight` to `[out, numel//out]`, `W_eff = W + (alpha/r)·(B@A)`, `A=[r,in]` (init_scale·randn), `B=[out,r]` (zeros → starts as identity); shape-agnostic (reshapes back). Registered via `torch.nn.utils.parametrize.register_parametrization(module, "weight", SingleAdapterLoRA(...), unsafe=True)`.
- `C.set_lora_trainable(module, train_bias=False)` — freezes all; unfreezes params whose name contains `.parametrizations.` and `.A`/`.B`. Works for ANY parametrized weight (conv included).
- `C.apply_lora(module, rank, alpha, init_scale=0.01)` — Linear/MHA only (used for scalar).
- `C.ContinuousHeuristicModel.max_norm_residual` (settable float); `forward(x, clamp=True)` does `clamp(softplus(raw), 0, max_norm_residual)`.
- `C6.INPUT_CHANNELS == 8`; `C6.build_model("unet", in_channels=8) -> UNetField`; `UNetField` Conv2d submodules: `e1..e4,b,d1..d4` each a `DoubleConv` with `.net[0]` and `.net[3]` = `nn.Conv2d`; plus `residual_head`,`path_head` = `nn.Conv2d(base,1,1)`; (`u1..u4` are `ConvTranspose2d`).
- `C6.checkpoint_path(out_dir, name) -> .../checkpoints/c6_heatmap__{name}.pt`.
- `C6.collect_dataset(spec, out_dir, split, n_worlds, cfg: C6Config, seed) -> Path` (npz keys x[N,8,G,G], target_residual[N,G,G], target_distance, free_mask, path_mask).
- `C6.HeatmapDataset(paths)` → items dict {x[8,G,G], target_residual[G,G], target_distance, free_mask, path_mask}.
- `C6.train_model(name, dataset_paths, out_dir, cfg, device) -> Path` — the field loop to MIRROR: `AdamW(lr=cfg.lr, weight_decay=cfg.weight_decay)`, per batch `value=masked_smooth_l1(model_output_residual(model(xb)), yb["target_residual"], yb["free_mask"])` + `cfg.loss_rank_weight*ranking_loss(pred_raw, yb["target_distance"], yb["free_mask"], cfg.rank_pairs)` + `cfg.loss_path_weight*path_bce_loss_from_logits(model(xb), yb["path_mask"], yb["free_mask"])` + `cfg.loss_consistency_weight*consistency_loss(model_output_residual(pred_raw), yb["x"], yb["free_mask"])`, grad-clip `cfg.grad_clip`. Helpers are C6 module-level: `C6.masked_smooth_l1`, `C6.ranking_loss`, `C6.path_bce_loss_from_logits`, `C6.consistency_loss`, `C6.model_output_residual`.
- `C6.C6Config` fields incl. grid_size=64, epochs, lr, weight_decay, grad_clip, batch_size, loss_*_weight, rank_pairs, max_world_retries.
- `C6.field_node_heuristic(model, x, world, roadmap, euclid_h, device) -> (h[N], pred_resid_norm[G,G])`: `h = euclid_h + side_len·max(0, interp(max(0, pred_resid)))` (≥0 clip, posinf→10).
- `P.ValueFieldProvider(model, grid_size, device, backbone)`; `.name=f"field_{backbone}"`; `.node_h(world, roadmap, goal_idx=1)`.
- `P.run_world_arms(world, rm, providers, budgets, w_values, goal_idx=1)` — uses any provider with `.node_h`; records keyed by the providers-dict key.
- C9 (frozen, import as `C9`): `C9.load_source_base`, `C9.train_scalar_model(backbone_cfg,npz,out_ckpt,feature_cfg,train_cfg,device,seed,init_ckpt=None)`, `C9.iter_test_worlds`, `C9.world_fingerprint`, `C9.adapt_seed(target,K,idx,base_seed)`, `C9.load_scalar_provider`, `C9._binding_budget_for`, `C9.RAW_COLS`, `C9._parse_csv`, `C9._parse_ints`. Source bases: scalar `c7_local/checkpoints/avgbase__{hrm,onlstm}.pt`; field `c7_local/checkpoints/c6_heatmap__unet.pt`.

---

## File structure

| File | Responsibility |
|---|---|
| Create `hrm-cloud/continuous_prm/continuous_prm_c9h_transfer.py` | conv-LoRA; scalar LoRA (bounded/unbounded) trainer + loader; field collect/train/provider (full-FT/scratch/conv-LoRA, bounded/unbounded); C9hConfig; adapt/eval/analyze/full + CLI. |
| Create `hrm-cloud/continuous_prm/tests/test_c9h_transfer.py` | conv-LoRA round-trip/identity/frozen-base; bounded≠unbounded (scalar+field); field train/load/node_h; adapt/eval/analyze smokes. |
| Reuse (no edits) | `continuous_prm_c9_transfer.py`, `continuous_prm_common.py`, `continuous_prm_providers.py`, `continuous_prm_c6_heatmap_value_field.py`, `continuous_prm_c7_hard_maps.py`. |
| Output `runs/c9h_local/{checkpoints,datasets,results}` | adapters, K-world npz, curves/comparisons/significance, manifest. |

---

## Task 1: conv-LoRA primitive

**Files:** Create `continuous_prm_c9h_transfer.py`; Test `tests/test_c9h_transfer.py`.

- [ ] **Step 1: failing test**
```python
# tests/test_c9h_transfer.py
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parents[1]
import continuous_prm_c9h_transfer as C9H
import continuous_prm_c6_heatmap_value_field as C6
import continuous_prm_common as C


def test_conv_lora_identity_then_frozen_base():
    import torch
    torch.manual_seed(0)
    unet = C6.build_model("unet", in_channels=8)
    x = torch.randn(1, 8, 64, 64)
    with torch.no_grad():
        base_out = unet(x).clone()
    n = C9H.apply_conv_lora(unet, rank=4, alpha=1.0)
    assert n >= 10  # many Conv2d wrapped
    with torch.no_grad():
        lora_out = unet(x)
    # B initialised to zero => adapter is identity at init
    assert torch.allclose(base_out, lora_out, atol=1e-5)
    # only A/B trainable after set_lora_trainable
    C.set_lora_trainable(unet)
    trainable = [nm for nm, p in unet.named_parameters() if p.requires_grad]
    assert trainable and all((".A" in nm or ".B" in nm) for nm in trainable)


def test_conv_lora_changes_output_after_step():
    import torch
    unet = C6.build_model("unet", in_channels=8)
    C9H.apply_conv_lora(unet, rank=4, alpha=1.0)
    C.set_lora_trainable(unet)
    x = torch.randn(1, 8, 64, 64)
    base_out = unet(x).detach().clone()
    opt = torch.optim.SGD([p for p in unet.parameters() if p.requires_grad], lr=1.0)
    loss = unet(x).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    assert not torch.allclose(base_out, unet(x), atol=1e-6)
```

- [ ] **Step 2: run, expect fail** (`AttributeError: apply_conv_lora`). `python -m pytest hrm-cloud/continuous_prm/tests/test_c9h_transfer.py -q`

- [ ] **Step 3: implement** (module header + conv-LoRA):
```python
#!/usr/bin/env python3
"""C9h: matched-compute transfer hardening (LoRA bounded/unbounded vs full-FT,
scalar + field conv-LoRA). New-file-only; reuses C9 (frozen) + C6/C7 field stack.
See docs/superpowers/specs/2026-06-29-c9-hardening-design.md.
"""
from __future__ import annotations
import argparse, dataclasses, csv as _csv, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

import continuous_prm_common as C
import continuous_prm_providers as P
import continuous_prm_c6_heatmap_value_field as C6
import continuous_prm_c7_hard_maps as H7
import continuous_prm_c9_transfer as C9


def _iter_conv2d(module: nn.Module):
    for sub in module.modules():
        if isinstance(sub, nn.Conv2d):
            yield sub


def apply_conv_lora(unet: nn.Module, rank: int, alpha: float, init_scale: float = 0.01) -> int:
    """Register a shape-agnostic SingleAdapterLoRA parametrization on every Conv2d.weight
    in the U-Net. Reuses common.SingleAdapterLoRA (handles [out,in,kh,kw] via flatten)."""
    wrapped = 0
    for conv in _iter_conv2d(unet):
        if parametrize.is_parametrized(conv, "weight"):
            continue
        w = conv.weight
        parametrize.register_parametrization(
            conv, "weight", C.SingleAdapterLoRA(w.data, rank, alpha, init_scale=init_scale), unsafe=True)
        wrapped += 1
    return wrapped
```

- [ ] **Step 4: run, expect pass** (2 passed). `python -m pytest hrm-cloud/continuous_prm/tests/test_c9h_transfer.py -q`. If `register_parametrization` on `weight` errors for ConvTranspose2d, note `_iter_conv2d` yields only `nn.Conv2d` (ConvTranspose2d is a different class) — fine. If `SingleAdapterLoRA` flatten assumes 2D, confirm it uses `numel()//out` (it does) so 4D conv weights work.

- [ ] **Step 5: commit**
```bash
git add hrm-cloud/continuous_prm/continuous_prm_c9h_transfer.py hrm-cloud/continuous_prm/tests/test_c9h_transfer.py
git commit -m "feat(c9h): conv-LoRA primitive (SingleAdapterLoRA on U-Net Conv2d)"
```

---

## Task 2: C9hConfig + matched-recipe helper

**Files:** Modify `continuous_prm_c9h_transfer.py`; Test same.

- [ ] **Step 1: failing test**
```python
def test_c9hconfig_defaults():
    cfg = C9H.C9hConfig()
    assert cfg.backbones == "hrm,onlstm,unet"
    assert cfg.methods == "lora_bounded,lora_unbounded,full_ft,scratch"
    assert cfg.k_grid == "1,4,16"
    assert cfg.n_adapt_seeds == 3
    assert cfg.epochs == 10 and abs(cfg.lr - 2e-4) < 1e-12
    assert cfg.source_dir.endswith("c7_local")
```

- [ ] **Step 2: run, expect fail.**

- [ ] **Step 3: implement** (append):
```python
@dataclass
class C9hConfig:
    source_dir: str = "hrm-cloud/continuous_prm/runs/c7_local"
    out_dir: str = "hrm-cloud/continuous_prm/runs/c9h_local"
    targets: str = "C_hard_maze_dense,C_hard_bugtrap,C_hard_rooms_large"
    backbones: str = "hrm,onlstm,unet"        # scalar hrm/onlstm + field unet
    methods: str = "lora_bounded,lora_unbounded,full_ft,scratch"  # + zero_shot implied
    k_grid: str = "1,4,16"
    n_adapt_seeds: int = 3
    n_test: int = 30
    epochs: int = 10          # MATCHED recipe for ALL trained arms
    lr: float = 2.0e-4        # MATCHED
    rank: int = 8
    alpha: float = 1.0
    grid_size: int = 64
    roadmap_nodes: int = 192
    roadmap_k: int = 7
    budgets: str = ""         # "" => reuse source calibration.json per target
    w_values: str = "1.0,1.1"
    seed: int = 1234
    scale: str = "local"
    mode: str = "full"
    cpu: bool = False


def _is_field(backbone: str) -> bool:
    return backbone == "unet"


def now_str() -> str:
    return C.now_str()
```

- [ ] **Step 4: run, expect pass.**
- [ ] **Step 5: commit** `feat(c9h): C9hConfig + matched-recipe defaults`.

---

## Task 3: scalar LoRA trainer (bounded/unbounded) + loader

**Files:** Modify module + test.

- [ ] **Step 1: failing test**
```python
@pytest.mark.skipif(not (HERE/"runs/c7_local/checkpoints/avgbase__hrm.pt").exists(), reason="base missing")
def test_scalar_lora_bounded_vs_unbounded(tmp_path):
    import torch, numpy as np
    dev = torch.device("cpu")
    base = C9.load_source_base(HERE/"runs/c7_local", "hrm", dev)
    # tiny npz (reuse C9-style)
    n=24; x=np.random.RandomState(0).randn(n, base.feature_cfg.seq_len, base.feature_cfg.token_dim).astype("float32")
    y=np.abs(np.random.RandomState(1).randn(n)).astype("float32")
    npz=tmp_path/"t.npz"; np.savez_compressed(npz, x=x, y=y, euclid=np.ones(n,"float32"), side=np.ones(n,"float32"))
    tcfg = dataclasses.replace(base.train_cfg, base_epochs=2, lr=2e-4)
    ckb = C9H.train_scalar_lora(base.backbone_cfg, npz, tmp_path/"b.pt", base.feature_cfg, tcfg, dev, seed=0,
                                init_ckpt=base.ckpt_path, rank=8, alpha=1.0, bounded=True)
    cku = C9H.train_scalar_lora(base.backbone_cfg, npz, tmp_path/"u.pt", base.feature_cfg, tcfg, dev, seed=0,
                                init_ckpt=base.ckpt_path, rank=8, alpha=1.0, bounded=False)
    pb = torch.load(ckb, map_location="cpu"); pu = torch.load(cku, map_location="cpu")
    assert pb["bounded"] is True and pu["bounded"] is False
    assert "lora_rank" in pb and pb["max_norm_residual"] != pu["max_norm_residual"]
```

- [ ] **Step 2: run, expect fail.**

- [ ] **Step 3: implement** (append). Mirror `C9.train_scalar_model`'s loop but: load base, `C.apply_lora`, `C.set_lora_trainable`, set `model.max_norm_residual = inf` when `not bounded`, save lora flags:
```python
def train_scalar_lora(backbone_cfg, dataset_npz, out_ckpt, feature_cfg, train_cfg, device,
                      seed, init_ckpt, rank, alpha, bounded: bool):
    out_ckpt = Path(out_ckpt)
    x, y = C.load_npz_arrays(dataset_npz)
    ds = C.ArrayDataset(x, y)
    loader = C.make_loader(ds, train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)
    model = C.build_model(backbone_cfg, feature_cfg, train_cfg, device)
    if init_ckpt is not None:
        C.safe_load_state(model, Path(init_ckpt))
    max_resid = float("inf") if not bounded else float(train_cfg.max_norm_residual)
    model.max_norm_residual = max_resid
    n_wrapped = C.apply_lora(model, rank=int(rank), alpha=float(alpha))
    C.set_lora_trainable(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    C.set_global_seed(int(seed))
    import torch.nn.functional as F
    for _ in range(train_cfg.base_epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb)
            if not torch.isfinite(loss): raise RuntimeError("nonfinite c9h scalar-lora loss")
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, train_cfg.grad_clip)
            opt.step()
    payload = {"model": model.state_dict(), "backbone_cfg": asdict(backbone_cfg),
               "feature_cfg": asdict(feature_cfg), "train_cfg": asdict(train_cfg),
               "lora_rank": int(rank), "alpha": float(alpha), "bounded": bool(bounded),
               "max_norm_residual": max_resid}
    C.ensure_dir(out_ckpt.parent); torch.save(payload, out_ckpt)
    return out_ckpt


def load_scalar_provider_c9h(ckpt, device):
    payload = torch.load(Path(ckpt), map_location="cpu")
    bb = C.BackboneConfig(**payload["backbone_cfg"]); fc = C.FeatureConfig(**payload["feature_cfg"])
    tc = C.TrainingConfig(**payload["train_cfg"])
    model = C.build_model(bb, fc, tc, device)
    if "lora_rank" in payload:
        C.apply_lora(model, rank=int(payload["lora_rank"]), alpha=float(payload["alpha"]))
    model.load_state_dict(payload["model"], strict=True)
    mnr = float(payload.get("max_norm_residual", tc.max_norm_residual))
    model.max_norm_residual = mnr
    model.eval()
    return P.ScalarResidualProvider(model, fc, device, bb.name, mnr)
```
Note: `np.clip(..., 0, inf)` in ScalarResidualProvider is fine (inf upper bound = no clamp).

- [ ] **Step 4: run, expect pass.**
- [ ] **Step 5: commit** `feat(c9h): scalar LoRA trainer (bounded/unbounded) + provider loader`.

---

## Task 4: field collect + field trainer (full-FT/scratch/conv-LoRA) + field provider loader

**Files:** Modify module + test. This is the largest task.

- [ ] **Step 1: failing test**
```python
@pytest.mark.skipif(not (HERE/"runs/c7_local/checkpoints/c6_heatmap__unet.pt").exists(), reason="field base missing")
def test_field_train_and_provider(tmp_path):
    import torch, numpy as np
    dev = torch.device("cpu")
    H7.install_c7_hard_maps(); specs = C.build_anchor_specs()
    c6cfg = C9H._c6_cfg(C9H.C9hConfig(epochs=1, grid_size=64))
    npz = C9H.collect_field_adapt(specs["C_hard_bugtrap"], tmp_path/"ds", "adapt_t", n_worlds=2, c6cfg=c6cfg, seed=7)
    base = HERE/"runs/c7_local/checkpoints/c6_heatmap__unet.pt"
    # full-FT (init from base, no lora) and conv-LoRA arms
    ck_ft = C9H.train_field_model("unet", [npz], tmp_path/"ft.pt", c6cfg, dev, seed=0, init_ckpt=base, lora_rank=0, alpha=1.0)
    ck_lo = C9H.train_field_model("unet", [npz], tmp_path/"lo.pt", c6cfg, dev, seed=0, init_ckpt=base, lora_rank=8, alpha=1.0)
    for ck, bounded in ((ck_ft, True), (ck_lo, True)):
        prov = C9H.load_field_provider(ck, dev, grid_size=64, bounded=bounded)
        w = C.build_world(specs["C_hard_bugtrap"], 11, C.RoadmapConfig(n_nodes=64, k_neighbors=7).min_start_goal_dist_frac)
        rm = C.build_prm(w, C.RoadmapConfig(n_nodes=64, k_neighbors=7), seed=24)
        h = prov.node_h(w, rm, goal_idx=1)
        assert np.isfinite(h).all() and h.shape[0] == rm.points.shape[0]
```
(If world seed 11 / prm seed 24 disconnects, try other small seeds — keep the finiteness/shape asserts.)

- [ ] **Step 2: run, expect fail.**

- [ ] **Step 3: implement** (append). `_c6_cfg`, `collect_field_adapt`, `train_field_model` (mirror `C6.train_model` loop + init + conv-LoRA + unique path), `load_field_provider` (+ bounded clamp):
```python
def _c6_cfg(cfg: C9hConfig) -> "C6.C6Config":
    return C6.C6Config(grid_size=int(cfg.grid_size), epochs=int(cfg.epochs), lr=float(cfg.lr),
                       roadmap_nodes=int(cfg.roadmap_nodes), roadmap_k=int(cfg.roadmap_k), seed=int(cfg.seed))


def collect_field_adapt(spec, ds_dir, split, n_worlds, c6cfg, seed) -> Path:
    return C6.collect_dataset(spec, Path(ds_dir), split, int(n_worlds), c6cfg, seed=int(seed))


def train_field_model(name, dataset_paths, out_ckpt, c6cfg, device, seed, init_ckpt, lora_rank, alpha):
    """Mirror C6.train_model's loop with optional init-from-base + conv-LoRA + unique out path."""
    out_ckpt = Path(out_ckpt)
    ds = C6.HeatmapDataset(list(dataset_paths))
    loader = torch.utils.data.DataLoader(ds, batch_size=int(c6cfg.batch_size), shuffle=True, num_workers=0)
    model = C6.build_model(name).to(device)
    if init_ckpt is not None:
        pl = torch.load(Path(init_ckpt), map_location="cpu")
        model.load_state_dict(pl["model"] if "model" in pl else pl, strict=True)
    if int(lora_rank) > 0:
        apply_conv_lora(model, rank=int(lora_rank), alpha=float(alpha))
        C.set_lora_trainable(model)
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=float(c6cfg.lr), weight_decay=float(c6cfg.weight_decay))
    C.set_global_seed(int(seed))
    for _ in range(int(c6cfg.epochs)):
        model.train()
        for batch in loader:
            xb = batch["x"].to(device)
            free = batch["free_mask"].to(device)
            tr = batch["target_residual"].to(device); td = batch["target_distance"].to(device)
            pm = batch["path_mask"].to(device)
            pred_raw = model(xb)
            value = C6.masked_smooth_l1(C6.model_output_residual(pred_raw), tr, free)
            rank = C6.ranking_loss(pred_raw, td, free, int(c6cfg.rank_pairs))
            path = C6.path_bce_loss_from_logits(model(xb), pm, free)
            cons = C6.consistency_loss(C6.model_output_residual(pred_raw), xb, free)
            loss = value + c6cfg.loss_rank_weight*rank + c6cfg.loss_path_weight*path + c6cfg.loss_consistency_weight*cons
            if not torch.isfinite(loss): raise RuntimeError("nonfinite c9h field loss")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(c6cfg.grad_clip)); opt.step()
    payload = {"model": model.state_dict(), "model_name": name, "lora_rank": int(lora_rank),
               "alpha": float(alpha), "grid_size": int(c6cfg.grid_size)}
    C.ensure_dir(out_ckpt.parent); torch.save(payload, out_ckpt)
    return out_ckpt


class _BoundedFieldProvider(P.ValueFieldProvider):
    """ValueFieldProvider with an optional integration-time upper clamp on the residual."""
    def __init__(self, model, grid_size, device, backbone, max_resid):
        super().__init__(model, grid_size, device, backbone)
        self.max_resid = float(max_resid)
    def node_h(self, world, roadmap, goal_idx: int = 1):
        from continuous_prm_providers import euclid_to_goal
        euclid = euclid_to_goal(roadmap, goal_idx)
        x = C6.make_heatmap_example(world, self.grid_size)["x"]
        h, _ = C6.field_node_heuristic(self.model, x, world, roadmap, euclid, self.device)
        if np.isfinite(self.max_resid):
            # re-clamp residual contribution: h = euclid + min(h-euclid, side*max_resid)
            extra = np.minimum(h - euclid, world.side_len * self.max_resid)
            h = euclid + np.maximum(0.0, extra)
        return np.maximum(h, 0.0)


def load_field_provider(ckpt, device, grid_size, bounded: bool):
    pl = torch.load(Path(ckpt), map_location="cpu")
    model = C6.build_model(pl.get("model_name", "unet"))
    if int(pl.get("lora_rank", 0)) > 0:
        apply_conv_lora(model, rank=int(pl["lora_rank"]), alpha=float(pl["alpha"]))
    model.load_state_dict(pl["model"], strict=True); model.to(device); model.eval()
    max_resid = 4.0 if bounded else float("inf")
    return _BoundedFieldProvider(model, int(grid_size), device, "unet", max_resid)
```
(Verify `euclid_to_goal` is importable from `continuous_prm_providers`; the reference shows `ValueFieldProvider.node_h` imports/uses it. If it's a module-level function, `from continuous_prm_providers import euclid_to_goal` works.)

- [ ] **Step 4: run, expect pass.** Confirm `C6.masked_smooth_l1`/`ranking_loss`/`path_bce_loss_from_logits`/`consistency_loss`/`model_output_residual` are C6 module-level (read C6 if a name differs; adapt the call).
- [ ] **Step 5: commit** `feat(c9h): field collect + field trainer (full-FT/scratch/conv-LoRA) + bounded field provider`.

---

## Task 5: adapt mode + manifest

**Files:** Modify module + test.

- [ ] **Step 1: failing smoke test** (1 target, hrm+unet, K{1}, 1 seed, methods all; cpu):
```python
@pytest.mark.skipif(not (HERE/"runs/c7_local/checkpoints/avgbase__hrm.pt").exists(), reason="base missing")
def test_run_adapt_smoke(tmp_path):
    import torch
    cfg = C9H.C9hConfig(source_dir=str(HERE/"runs/c7_local"), out_dir=str(tmp_path/"c9h"),
                        targets="C_hard_bugtrap", backbones="hrm,unet",
                        methods="lora_bounded,lora_unbounded,full_ft,scratch",
                        k_grid="1", n_adapt_seeds=1, n_test=4, epochs=1,
                        roadmap_nodes=192, roadmap_k=7, cpu=True, seed=7)
    man = C9H.run_adapt(cfg, torch.device("cpu"))
    got = {(a["backbone"], a["method"]) for a in man["arms"]}
    assert ("hrm","lora_bounded") in got and ("hrm","lora_unbounded") in got
    assert ("unet","lora_bounded") in got and ("unet","full_ft") in got
    for a in man["arms"]: assert Path(a["ckpt"]).exists()
```

- [ ] **Step 2: run, expect fail.**

- [ ] **Step 3: implement** `run_adapt`. For each target × backbone × K × seed × method: pick source base (scalar `avgbase__{bb}` via `C9.load_source_base`; field `c6_heatmap__unet.pt`), collect the K-world dataset (scalar via `C.collect_task_dataset` with `C9.adapt_seed`+`C9.SCALAR_NODES_PER_WORLD`; field via `collect_field_adapt`), then dispatch by (backbone family, method):
  - scalar full_ft/scratch → `C9.train_scalar_model(..., init_ckpt=base.ckpt_path | None)` with `train_cfg = dataclasses.replace(base.train_cfg, base_epochs=cfg.epochs, lr=cfg.lr)`.
  - scalar lora_bounded/lora_unbounded → `train_scalar_lora(..., bounded=…)`.
  - field full_ft/scratch → `train_field_model("unet", [npz], ck, c6cfg, …, init_ckpt=base|None, lora_rank=0)`.
  - field lora_bounded/lora_unbounded → `train_field_model("unet", [npz], ck, c6cfg, …, init_ckpt=base, lora_rank=cfg.rank)` (bounded flag recorded for eval, not training).
  Unique ckpt path: `checkpoints/c9h__{target}__{bb}__K{K}__s{seed}__{method}.pt`. Append arm dict {target,K,seed,method,backbone,ckpt,bounded}. Write `adapt_manifest.json`. (Mirror `C9.run_adapt` structure.)

- [ ] **Step 4: run, expect pass + full file green.**
- [ ] **Step 5: commit** `feat(c9h): adapt mode (scalar+field, 4 methods incl. bounded/unbounded) + manifest`.

---

## Task 6: eval mode (matched A* with scalar+field providers)

**Files:** Modify module + test.

- [ ] **Step 1: failing smoke** (reuse Task 5 adapt; assert field + scalar arms appear in raw CSV):
```python
@pytest.mark.skipif(not (HERE/"runs/c7_local/checkpoints/avgbase__hrm.pt").exists(), reason="base missing")
def test_run_eval_smoke(tmp_path):
    import torch, csv
    cfg = C9H.C9hConfig(source_dir=str(HERE/"runs/c7_local"), out_dir=str(tmp_path/"c9h"),
                        targets="C_hard_bugtrap", backbones="hrm,unet",
                        methods="lora_bounded,full_ft,scratch", k_grid="1", n_adapt_seeds=1,
                        n_test=4, epochs=1, budgets="200,400", w_values="1.0", cpu=True, seed=7)
    dev=torch.device("cpu"); C9H.run_adapt(cfg, dev); raw=C9H.run_eval(cfg, dev)
    provs={r["provider"] for r in csv.DictReader(open(raw,newline=""))}
    assert any(p.startswith("zeroshot_hrm") for p in provs)
    assert any(p.startswith("zeroshot_unet") for p in provs)
    assert any(p.startswith("lora_bounded_unet") for p in provs)
```

- [ ] **Step 2: run, expect fail.**

- [ ] **Step 3: implement** `run_eval` mirroring `C9.run_eval` but the provider set per target includes: euclid, oracle; per scalar backbone a `zeroshot_{bb}` (`C9.load_source_base` → `P.ScalarResidualProvider`); a `zeroshot_unet` (field base → `load_field_provider(base, …, bounded=True)`); and per manifest arm a provider via `load_scalar_provider_c9h` (scalar) or `load_field_provider(…, bounded=arm["bounded"])` (field), registered under unique key `{method}_{backbone}_K{K}_s{seed}` with `meta` stamping (K/seed/method/backbone). Use `C9.RAW_COLS`, `C9.iter_test_worlds`, `P.run_world_arms`, budgets via cfg.budgets or `c7_local/calibration.json`. Write per-target shard + merged `results/continuous_prm_c9h_eval_raw.csv`.

- [ ] **Step 4: run, expect pass.**
- [ ] **Step 5: commit** `feat(c9h): eval mode (scalar+field providers; matched A* on TEST)`.

---

## Task 7: analyze mode (5-method curves + bounded-vs-unbounded + field)

**Files:** Modify module + test.

- [ ] **Step 1: failing test** (synthetic rows; assert curves CSV has lora_bounded & lora_unbounded & full_ft for unet; comparisons + significance files exist). Build synthetic rows with `C9.RAW_COLS`, methods incl. `lora_bounded`/`lora_unbounded`, backbone `unet`, one budget, euclid+oracle, 2 worlds.

- [ ] **Step 2: run, expect fail.**

- [ ] **Step 3: implement** `analyze_from_raw_c9h(raw_csv, out_dir, seed, targets, backbones, methods)` — copy C9's `analyze_from_raw` structure (single binding budget via `C9._binding_budget_for`; pool over (world×seed); `bootstrap_median_ci`; McNemar+BH) but with the configurable `methods` list (`["zero_shot"]+methods`). Comparisons MD adds a **bounded-vs-unbounded** section (lora_bounded vs lora_unbounded per backbone/K) and includes `unet` rows. `run_analyze(cfg)` wraps it reading `results/continuous_prm_c9h_eval_raw.csv`. (Import `bootstrap_median_ci` from C9's imports or from `continuous_prm_c7_integration_compare`; `mcnemar_exact_p`/`bh_q_values` from C6.)

- [ ] **Step 4: run, expect pass.**
- [ ] **Step 5: commit** `feat(c9h): analyze mode (5-method curves + bounded-vs-unbounded + field)`.

---

## Task 8: full mode + CLI + local run + C9H_RESULTS.md

**Files:** Modify module; Create `C9H_RESULTS.md`; Modify `C9_RESULTS.md` (cross-link).

- [ ] **Step 1: failing end-to-end smoke** (`run_full` on cpu, 1 target, hrm+unet, K{1}, 1 seed, 2 methods; assert curves+comparisons+significance exist).
- [ ] **Step 2: run, expect fail.**
- [ ] **Step 3: implement** `run_full` (adapt→eval→analyze), `apply_scale` (local: n_adapt_seeds or 3, n_test or 30; smoke: tiny), `config_from_args`, `build_argparser` (auto-flags from C9hConfig; `store_true` for `cpu`), `main()` + `__main__` (dispatch adapt/eval/analyze/full).
- [ ] **Step 4: run, expect pass + full C9h suite green.** CLI sanity: `python -c "...build_argparser().parse_args(['--mode','full','--scale','smoke','--cpu'])..."`.
- [ ] **Step 5: commit** `feat(c9h): full mode + CLI + scale presets`.

- [ ] **Step 6: CPU smoke** (delete after):
```bash
python hrm-cloud/continuous_prm/continuous_prm_c9h_transfer.py --mode full --scale smoke \
  --targets C_hard_bugtrap --backbones hrm,unet --methods lora_bounded,full_ft,scratch \
  --budgets 200,400 --w-values 1.0 --out-dir hrm-cloud/continuous_prm/runs/c9h_smoke --cpu
```
Expect curves/comparisons/significance written. Then `rm -rf runs/c9h_smoke`.

- [ ] **Step 7: local GPU run** (background; monitor like C9, filter avoiding `oom`/`cuda` substrings):
```bash
python -u hrm-cloud/continuous_prm/continuous_prm_c9h_transfer.py --mode full --scale local \
  --targets C_hard_maze_dense,C_hard_bugtrap,C_hard_rooms_large --backbones hrm,onlstm,unet \
  --methods lora_bounded,lora_unbounded,full_ft,scratch --k-grid 1,4,16 --n-adapt-seeds 3 \
  --n-test 30 --w-values 1.0,1.1 --out-dir hrm-cloud/continuous_prm/runs/c9h_local
```

- [ ] **Step 8: gates + writeup.** G0 (conv-LoRA unit tests green), G1 (field/scalar zero_shot beat euclid), G2 (matched LoRA-vs-FT + bounded-vs-unbounded coherent). Write `C9H_RESULTS.md` (mirror C9_RESULTS: arm matrix, the 2×2 disentangling, field crossover, gate verdicts, caveats), add a cross-link to `C9_RESULTS.md`. Commit text artifacts only (curves/comparisons/significance/manifest + the two MDs); confirm WIP unstaged.
```bash
git add hrm-cloud/continuous_prm/C9H_RESULTS.md hrm-cloud/continuous_prm/C9_RESULTS.md \
  hrm-cloud/continuous_prm/runs/c9h_local/results/continuous_prm_c9h_curves.csv \
  hrm-cloud/continuous_prm/runs/c9h_local/results/continuous_prm_c9h_comparisons.md \
  hrm-cloud/continuous_prm/runs/c9h_local/results/continuous_prm_c9h_significance.md \
  hrm-cloud/continuous_prm/runs/c9h_local/adapt_manifest.json
git commit -m "results(c9h): matched-compute + field conv-LoRA transfer validation + C9H_RESULTS"
```

---

## Self-review

**Spec coverage:** §3 arm matrix/grid → T2,T5; §4 reuse + new module → all; §5 conv-LoRA → T1; §5 field bounded/unbounded (integration clamp) → T4 `_BoundedFieldProvider`; §6 matched LoRA trainer (scalar) → T3, (field) → T4/T5; §7 field arms → T4,T5; §8 eval/analyze reuse → T6,T7; §9 gates → T8; §10 risks (conv-LoRA correctness) → T1 tests. All covered.

**Placeholder scan:** every code step has concrete code; the only "mirror C9/C6" steps (T5 run_adapt, T6 run_eval, T7 analyze) give the exact reuse calls + record schema + unique-key convention — the implementer follows the frozen C9 equivalents. No "TBD"/"handle edge cases".

**Type consistency:** `apply_conv_lora`/`set_lora_trainable` (reused) consistent T1→T4; `train_scalar_lora` signature stable T3→T5; `train_field_model(name, dataset_paths, out_ckpt, c6cfg, device, seed, init_ckpt, lora_rank, alpha)` stable T4→T5; `load_field_provider(ckpt, device, grid_size, bounded)` / `load_scalar_provider_c9h(ckpt, device)` stable T3/T4→T6; `C9hConfig` fields used identically across tasks; arm-key convention `{method}_{backbone}_K{K}_s{seed}` + `zeroshot_{bb}` defined T6, parsed by T7 via the meta-stamped columns (not by name). Consistent.
