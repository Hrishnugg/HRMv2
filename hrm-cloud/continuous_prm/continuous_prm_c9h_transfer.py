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
    in the U-Net (reuses common.SingleAdapterLoRA; B init zero => identity at init).
    Returns the number of wrapped conv weights. set_lora_trainable then trains only A/B."""
    wrapped = 0
    for conv in _iter_conv2d(unet):
        if parametrize.is_parametrized(conv, "weight"):
            continue
        w = conv.weight
        parametrize.register_parametrization(
            conv, "weight", C.SingleAdapterLoRA(w.data, rank, alpha, init_scale=init_scale), unsafe=True)
        wrapped += 1
    return wrapped


@dataclass
class C9hConfig:
    source_dir: str = "hrm-cloud/continuous_prm/runs/c7_local"
    out_dir: str = "hrm-cloud/continuous_prm/runs/c9h_local"
    targets: str = "C_hard_maze_dense,C_hard_bugtrap,C_hard_rooms_large"
    backbones: str = "hrm,onlstm,unet"
    methods: str = "lora_bounded,lora_unbounded,full_ft,scratch"
    k_grid: str = "1,4,16"
    n_adapt_seeds: int = 3
    n_test: int = 30
    epochs: int = 10
    lr: float = 2.0e-4
    rank: int = 8
    alpha: float = 1.0
    grid_size: int = 64
    roadmap_nodes: int = 192
    roadmap_k: int = 7
    budgets: str = ""
    w_values: str = "1.0,1.1"
    seed: int = 1234
    scale: str = "local"
    mode: str = "full"
    cpu: bool = False


def _is_field(backbone: str) -> bool:
    return backbone == "unet"


def now_str() -> str:
    return C.now_str()


# ---------------------------------------------------------------------------
# Task 3 — Matched-compute scalar LoRA trainer (bounded / unbounded)
# ---------------------------------------------------------------------------

def train_scalar_lora(backbone_cfg, dataset_npz, out_ckpt, feature_cfg, train_cfg, device,
                      seed, init_ckpt, rank, alpha, bounded: bool):
    """Matched-compute scalar LoRA fine-tune. bounded=True keeps the model's finite
    max_norm_residual clamp; bounded=False sets it to inf (no clamp)."""
    import torch.nn.functional as F
    out_ckpt = Path(out_ckpt)
    x, y = C.load_npz_arrays(dataset_npz)
    ds = C.ArrayDataset(x, y)
    loader = C.make_loader(ds, train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)
    model = C.build_model(backbone_cfg, feature_cfg, train_cfg, device)
    if init_ckpt is not None:
        C.safe_load_state(model, Path(init_ckpt))
    max_resid = float(train_cfg.max_norm_residual) if bounded else float("inf")
    model.max_norm_residual = max_resid
    C.apply_lora(model, rank=int(rank), alpha=float(alpha))
    C.set_lora_trainable(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    C.set_global_seed(int(seed))
    for _ in range(train_cfg.base_epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite c9h scalar-lora loss")
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
    """Load a scalar checkpoint (avgbase/full-FT/scratch/LoRA, bounded/unbounded) into a provider."""
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


# ---------------------------------------------------------------------------
# Task 4 — Field U-Net transfer arms (collect, train, bounded provider)
# ---------------------------------------------------------------------------

def _c6_cfg(cfg) -> "C6.C6Config":
    """Build a minimal C6Config from a C9hConfig, forwarding grid/epoch/lr/roadmap settings."""
    return C6.C6Config(
        grid_size=int(cfg.grid_size),
        epochs=int(cfg.epochs),
        lr=float(cfg.lr),
        roadmap_nodes=int(cfg.roadmap_nodes),
        roadmap_k=int(cfg.roadmap_k),
        seed=int(cfg.seed),
    )


def collect_field_adapt(spec, ds_dir, split: str, n_worlds: int, c6cfg, seed: int) -> Path:
    """Collect a C6-format heatmap dataset for one anchor spec.  Returns the npz path."""
    return C6.collect_dataset(spec, Path(ds_dir), split, int(n_worlds), c6cfg, seed=int(seed))


def train_field_model(
    name: str,
    dataset_paths,
    out_ckpt,
    c6cfg,
    device,
    seed: int,
    init_ckpt,
    lora_rank: int,
    alpha: float,
) -> Path:
    """Train (or fine-tune) a C6 field model.

    Mirrors C6.train_model's loss loop exactly:
      - uses forward_both() for a single forward pass yielding (residual_raw, path_logits)
      - model_output_residual(raw) applies softplus+squeeze
      - ranking_loss receives the post-softplus prediction (not raw)
      - path_bce_loss_from_logits receives path_logits (not a second model(x) call)
    Optional conv-LoRA via apply_conv_lora when lora_rank > 0.
    """
    out_ckpt = Path(out_ckpt)
    ds = C6.HeatmapDataset(list(dataset_paths))
    loader = torch.utils.data.DataLoader(ds, batch_size=int(c6cfg.batch_size), shuffle=True, num_workers=0)
    model = C6.build_model(name).to(device)
    if init_ckpt is not None:
        pl = torch.load(Path(init_ckpt), map_location="cpu")
        sd = pl["model"] if isinstance(pl, dict) and "model" in pl else pl
        model.load_state_dict(sd, strict=True)
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
            xb = batch["x"].to(device=device, dtype=torch.float32)
            free = batch["free_mask"].to(device=device)
            tr = batch["target_residual"].to(device=device, dtype=torch.float32)
            td = batch["target_distance"].to(device=device, dtype=torch.float32)
            pm = batch["path_mask"].to(device=device, dtype=torch.float32)
            # Single forward — use forward_both if available (same as C6.train_model)
            if hasattr(model, "forward_both"):
                raw_pred, path_logits = model.forward_both(xb)
            else:
                raw_pred = model(xb)
                path_logits = model.forward_path(xb) if hasattr(model, "forward_path") else torch.zeros_like(raw_pred)
            pred = C6.model_output_residual(raw_pred)
            value = C6.masked_smooth_l1(pred, tr, free)
            rank = C6.ranking_loss(pred, td, free, int(c6cfg.rank_pairs))
            path = C6.path_bce_loss_from_logits(path_logits, pm, free)
            cons = C6.consistency_loss(pred, xb, free)
            loss = (
                value
                + float(c6cfg.loss_rank_weight) * rank
                + float(c6cfg.loss_path_weight) * path
                + float(c6cfg.loss_consistency_weight) * cons
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite c9h field loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(c6cfg.grad_clip))
            opt.step()
    payload = {
        "model": model.state_dict(),
        "model_name": name,
        "lora_rank": int(lora_rank),
        "alpha": float(alpha),
        "grid_size": int(c6cfg.grid_size),
    }
    C.ensure_dir(out_ckpt.parent)
    torch.save(payload, out_ckpt)
    return out_ckpt


class _BoundedFieldProvider(P.ValueFieldProvider):
    """ValueFieldProvider with an optional residual clamp (bounded=True limits extra residual)."""

    def __init__(self, model, grid_size: int, device, backbone: str, max_resid: float):
        super().__init__(model, grid_size, device, backbone)
        self.max_resid = float(max_resid)

    def node_h(self, world, roadmap, goal_idx: int = 1) -> np.ndarray:
        from continuous_prm_providers import euclid_to_goal
        euclid = euclid_to_goal(roadmap, goal_idx)
        x = C6.make_heatmap_example(world, self.grid_size)["x"]
        h, _ = C6.field_node_heuristic(self.model, x, world, roadmap, euclid, self.device)
        if np.isfinite(self.max_resid):
            # Clamp the additive residual term: extra = h - euclid <= side_len * max_resid
            extra = np.asarray(h, dtype=np.float64) - euclid
            extra = np.minimum(extra, float(world.side_len) * self.max_resid)
            h = euclid + np.maximum(0.0, extra)
        return np.maximum(np.asarray(h, dtype=np.float64), 0.0)


def load_field_provider(ckpt, device, grid_size: int, bounded: bool) -> "_BoundedFieldProvider":
    """Load a field checkpoint (full-FT or conv-LoRA) and wrap it in a _BoundedFieldProvider."""
    pl = torch.load(Path(ckpt), map_location="cpu")
    model = C6.build_model(pl.get("model_name", "unet"))
    if int(pl.get("lora_rank", 0)) > 0:
        apply_conv_lora(model, rank=int(pl["lora_rank"]), alpha=float(pl["alpha"]))
    model.load_state_dict(pl["model"], strict=True)
    model.to(device)
    model.eval()
    max_resid = 4.0 if bounded else float("inf")
    return _BoundedFieldProvider(model, int(grid_size), device, pl.get("model_name", "unet"), max_resid)


# ---------------------------------------------------------------------------
# Task 5 — adapt mode (4 methods × scalar+field backbones, writes adapt_manifest.json)
# ---------------------------------------------------------------------------

def arm_ckpt_path(out_dir, target, K, seed, method, backbone):
    return Path(out_dir) / "checkpoints" / f"c9h__{target}__{backbone}__K{K}__s{seed}__{method}.pt"


def _field_base_ckpt(source_dir) -> Path:
    return Path(source_dir) / "checkpoints" / "c6_heatmap__unet.pt"


def run_adapt(cfg: C9hConfig, device) -> dict:
    H7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    out_dir = Path(cfg.out_dir)
    ds_dir = out_dir / "datasets"
    C.ensure_dir(out_dir / "checkpoints"); C.ensure_dir(ds_dir)
    rmcfg = C.RoadmapConfig(n_nodes=cfg.roadmap_nodes, k_neighbors=cfg.roadmap_k)
    c6cfg = _c6_cfg(cfg)
    Ks = [k for k in C9._parse_ints(cfg.k_grid) if k > 0]
    methods = C9._parse_csv(cfg.methods)
    arms = []
    for target in C9._parse_csv(cfg.targets):
        spec = specs[target]
        for backbone in C9._parse_csv(cfg.backbones):
            field = _is_field(backbone)
            base = None if field else C9.load_source_base(Path(cfg.source_dir), backbone, device)
            tcfg = None if field else dataclasses.replace(base.train_cfg, base_epochs=int(cfg.epochs), lr=float(cfg.lr))
            for K in Ks:
                for s in range(int(cfg.n_adapt_seeds)):
                    aseed = C9.adapt_seed(target, K, s, cfg.seed)
                    if field:
                        npz = collect_field_adapt(spec, ds_dir, f"fadapt_{target}_K{K}_s{s}", K, c6cfg, seed=aseed)
                    else:
                        npz = C.collect_task_dataset(spec, ds_dir, f"adapt_{target}_K{K}_s{s}", K,
                                                     C9.SCALAR_NODES_PER_WORLD, rmcfg, base.feature_cfg, seed=aseed)
                    for method in methods:
                        ck = arm_ckpt_path(out_dir, target, K, s, method, backbone)
                        bounded = ("unbounded" not in method)  # lora_bounded/full_ft/scratch => bounded=True; lora_unbounded => False
                        if not ck.exists():
                            if field:
                                lora_rank = int(cfg.rank) if method.startswith("lora") else 0
                                init = None if method == "scratch" else _field_base_ckpt(cfg.source_dir)
                                train_field_model("unet", [npz], ck, c6cfg, device, seed=aseed,
                                                  init_ckpt=init, lora_rank=lora_rank, alpha=float(cfg.alpha))
                            else:
                                if method.startswith("lora"):
                                    train_scalar_lora(base.backbone_cfg, npz, ck, base.feature_cfg, tcfg, device,
                                                      seed=aseed, init_ckpt=base.ckpt_path, rank=int(cfg.rank),
                                                      alpha=float(cfg.alpha), bounded=bounded)
                                else:
                                    init = base.ckpt_path if method == "full_ft" else None
                                    C9.train_scalar_model(base.backbone_cfg, npz, ck, base.feature_cfg, tcfg, device,
                                                          seed=aseed, init_ckpt=init)
                        arms.append(dict(target=target, K=K, seed=s, method=method, backbone=backbone,
                                         ckpt=str(ck), bounded=bool(bounded)))
                    print(f"[{now_str()}] c9h adapt: {target} {backbone} K={K} s={s} done", flush=True)
    manifest = {"arms": arms, "targets": C9._parse_csv(cfg.targets), "backbones": C9._parse_csv(cfg.backbones),
                "methods": methods, "k_grid": C9._parse_ints(cfg.k_grid), "n_adapt_seeds": int(cfg.n_adapt_seeds), "seed": int(cfg.seed)}
    C.write_json(out_dir / "adapt_manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Task 6 — eval mode (scalar + field providers; matched A* on TEST)
# ---------------------------------------------------------------------------

import csv as _csv


def _target_budgets(cfg: C9hConfig, target: str) -> List[int]:
    if str(cfg.budgets).strip():
        return C9._parse_ints(cfg.budgets)
    calib = Path(cfg.source_dir) / "calibration.json"
    if calib.exists():
        b = (json.loads(calib.read_text()).get("budgets", {}) or {}).get(target)
        if b:
            return [int(x) for x in b]
    raise RuntimeError(f"no budgets for {target}: pass --budgets or provide {calib}")


def run_eval(cfg: C9hConfig, device) -> Path:
    H7.install_c7_hard_maps(); specs = C.build_anchor_specs()
    out_dir = Path(cfg.out_dir); res_dir = out_dir / "results"; shard_dir = res_dir / "_shards"
    C.ensure_dir(shard_dir)
    rmcfg = C.RoadmapConfig(n_nodes=cfg.roadmap_nodes, k_neighbors=cfg.roadmap_k)
    w_values = [float(x) for x in C9._parse_csv(cfg.w_values)]
    manifest = json.loads((out_dir / "adapt_manifest.json").read_text())
    all_rows = []
    for suite_idx, target in enumerate(C9._parse_csv(cfg.targets)):
        spec = specs[target]; budgets = _target_budgets(cfg, target)
        providers = {"euclid": P.EuclidProvider(), "oracle": P.OracleProvider()}
        meta = {"euclid": {}, "oracle": {}}
        # zero-shot per backbone
        for backbone in C9._parse_csv(cfg.backbones):
            if _is_field(backbone):
                zp = load_field_provider(_field_base_ckpt(cfg.source_dir), device, grid_size=cfg.grid_size, bounded=True)
            else:
                base = C9.load_source_base(Path(cfg.source_dir), backbone, device)
                zp = P.ScalarResidualProvider(base.model, base.feature_cfg, device, backbone, base.train_cfg.max_norm_residual)
            zp.name = f"zeroshot_{backbone}"
            providers[zp.name] = zp
            meta[zp.name] = dict(K=0, seed=-1, method="zero_shot", backbone=backbone)
        # adapted arms
        for a in manifest["arms"]:
            if a["target"] != target:
                continue
            if _is_field(a["backbone"]):
                prov = load_field_provider(Path(a["ckpt"]), device, grid_size=cfg.grid_size, bounded=bool(a["bounded"]))
            else:
                prov = load_scalar_provider_c9h(Path(a["ckpt"]), device)
            key = f'{a["method"]}_{a["backbone"]}_K{a["K"]}_s{a["seed"]}'
            prov.name = key; providers[key] = prov
            meta[key] = dict(K=a["K"], seed=a["seed"], method=a["method"], backbone=a["backbone"])
        rows = []
        for world_index, world, rm in C9.iter_test_worlds(spec, suite_idx, cfg, rmcfg, cfg.n_test):
            recs = P.run_world_arms(world, rm, providers, budgets, w_values, goal_idx=1)
            for r in recs:
                m = meta.get(r["provider"], dict(K=-1, seed=-1, method=r["provider"], backbone=""))
                r.update(dict(target=target, suite=target, world_index=world_index, **m))
                rows.append(r)
        sp = shard_dir / f"{target}.csv"
        with open(sp, "w", newline="") as f:
            wri = _csv.DictWriter(f, fieldnames=C9.RAW_COLS); wri.writeheader()
            for r in rows: wri.writerow({k: r.get(k, "") for k in C9.RAW_COLS})
        all_rows.extend(rows)
        print(f"[{now_str()}] c9h eval {target}: {len(rows)} rows", flush=True)
    raw = res_dir / "continuous_prm_c9h_eval_raw.csv"
    with open(raw, "w", newline="") as f:
        wri = _csv.DictWriter(f, fieldnames=C9.RAW_COLS); wri.writeheader()
        for r in all_rows: wri.writerow({k: r.get(k, "") for k in C9.RAW_COLS})
    print(f"[{now_str()}] c9h eval: merged {len(all_rows)} rows -> {raw}", flush=True)
    return raw
