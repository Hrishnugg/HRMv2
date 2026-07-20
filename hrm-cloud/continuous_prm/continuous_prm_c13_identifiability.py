#!/usr/bin/env python3
"""C13-B multi-angle identifiability and integration study.

This study is intentionally diagnostic.  It separates four questions:

1. Is the fresh-start rollout target stable and related to true route cost?
2. Can the declared bounded current observation identify that target?
3. Do padding/readout choices prevent recurrent models from extracting it?
4. Does FOCAL consume a useful rank in the right way?

Shortest-path distances appear only in explicitly labelled evaluation audits.
They are never used to collect training labels or fit a model.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

import continuous_prm_common as C
import continuous_prm_c13_state_heuristic as C13
import continuous_prm_c13_td_ranker as TD
import continuous_prm_c7_hard_maps as M7


@dataclass
class StudyConfig:
    out_dir: str = "runs/c13_identifiability"
    suite: str = "C_hard_maze"
    train_worlds: int = 12
    val_worlds: int = 4
    eval_worlds: int = 6
    roadmap_nodes: int = 192
    roadmap_k: int = 7
    rollouts_per_start: int = 3
    audit_rollouts_per_start: int = 10
    max_steps_factor: int = 8
    sensor_radius_frac: float = 0.20
    num_rays: int = 8
    ray_steps: int = 8
    max_neighbors: int = 16
    max_log_residual: float = 4.0
    hidden_dim: int = 32
    epochs: int = 12
    batch_size: int = 128
    lr: float = 5.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    focal_ws: str = "1.00,1.02,1.05,1.10,1.25,1.50"
    seed: int = 1234
    max_world_retries: int = 200
    device: str = "auto"

    def state_cfg(self) -> C13.LocalStateConfig:
        return C13.LocalStateConfig(
            sensor_radius_frac=float(self.sensor_radius_frac),
            num_rays=int(self.num_rays),
            ray_steps=int(self.ray_steps),
            max_neighbors=int(self.max_neighbors),
        )

    def policy_cfg(self, rollouts_per_start: Optional[int] = None) -> TD.RolloutPolicyConfig:
        return TD.RolloutPolicyConfig(
            rollouts_per_start=int(
                self.rollouts_per_start if rollouts_per_start is None else rollouts_per_start
            ),
            max_steps_factor=int(self.max_steps_factor),
            epsilon=0.02,
            temperature=0.02,
            revisit_penalty_frac=0.50,
            reverse_penalty_frac=0.15,
            max_norm_residual=float(self.max_log_residual),
        )


MODEL_SCHEDULE: Dict[str, Tuple[int, ...]] = {
    "flat_mlp": (3, 6, 12),
    "masked_pool": (3, 6, 12),
    "hrm_padded": (3, 12),
    "hrm_trimmed": (3, 12),
    "hrm_summary_last": (12,),
    "onlstm_padded": (12,),
    "onlstm_trimmed": (12,),
}


def resolve_device(name: str) -> torch.device:
    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    targ = np.asarray(target, dtype=np.float64).reshape(-1)
    keep = np.isfinite(pred) & np.isfinite(targ)
    pred, targ = pred[keep], targ[keep]
    if not len(pred):
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "prediction_mean": float("nan"),
            "target_mean": float("nan"),
        }
    err = pred - targ
    return {
        "n": int(len(pred)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "pearson": safe_pearson(pred, targ),
        "spearman": safe_spearman(pred, targ),
        "prediction_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(targ)),
    }


def sequence_lengths(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(x, torch.Tensor):
        return torch.any(torch.abs(x) > 0.0, dim=-1).sum(dim=1)
    arr = np.asarray(x)
    return (np.abs(arr).sum(axis=2) > 0.0).sum(axis=1)


# ---------------------------------------------------------------------------
# Representation variants
# ---------------------------------------------------------------------------


class PositiveHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, max_output: float):
        super().__init__()
        self.max_output = float(max_output)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.softplus(self.net(x).squeeze(-1)), 0.0, self.max_output)


class FlatMLPRanker(nn.Module):
    def __init__(self, seq_len: int, token_dim: int, hidden_dim: int, max_output: float):
        super().__init__()
        flat_dim = int(seq_len) * int(token_dim)
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.head = PositiveHead(hidden_dim, hidden_dim, max_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x.flatten(1)))


class MaskedPoolRanker(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, max_output: float):
        super().__init__()
        self.token_encoder = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.GELU())
        self.head = PositiveHead(hidden_dim * 3, hidden_dim, max_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.any(torch.abs(x) > 0.0, dim=-1)
        encoded = self.token_encoder(x)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(encoded.dtype)
        mean = torch.sum(encoded * mask.unsqueeze(-1), dim=1) / denom
        masked = encoded.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        maximum = torch.max(masked, dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        summary = encoded[:, 0]
        return self.head(torch.cat([summary, mean, maximum], dim=-1))


class RecurrentRanker(nn.Module):
    def __init__(
        self,
        backbone_type: str,
        readout_mode: str,
        token_dim: int,
        hidden_dim: int,
        max_output: float,
    ):
        super().__init__()
        self.backbone_type = str(backbone_type)
        self.readout_mode = str(readout_mode)
        self.max_output = float(max_output)
        if backbone_type == "hrm":
            self.backbone = C.DeepSapientHRMBackbone(
                input_dim=token_dim,
                hidden_dim=hidden_dim,
                k_step=2,
                num_heads=4,
                num_layers=1,
            )
        elif backbone_type == "onlstm":
            self.backbone = C.ONLSTMBackbone(
                input_dim=token_dim,
                hidden_dim=hidden_dim,
                num_layers=1,
                chunk_size=8,
            )
        else:
            raise ValueError(backbone_type)
        self.head = PositiveHead(hidden_dim, hidden_dim, max_output)

    def encode(self, x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        selected = str(mode or self.readout_mode)
        if selected == "padded":
            return self.backbone.encode_sequence(x)
        if selected not in {"trimmed", "summary_last"}:
            raise ValueError(selected)
        lengths = sequence_lengths(x)
        contexts = torch.zeros(
            (x.shape[0], self.backbone.hidden_dim),
            device=x.device,
            dtype=x.dtype,
        )
        for length_t in torch.unique(lengths, sorted=True):
            length = int(length_t.item())
            indices = torch.nonzero(lengths == length_t, as_tuple=False).squeeze(-1)
            seq = x.index_select(0, indices)[:, :length]
            if selected == "summary_last" and length > 1:
                seq = torch.cat([seq[:, 1:], seq[:, :1]], dim=1)
            chunk = self.backbone.encode_sequence(seq)
            contexts = contexts.index_copy(0, indices, chunk)
        return contexts

    def forward_mode(self, x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        return self.head(self.encode(x, mode=mode))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_mode(x, self.readout_mode)


def build_model(name: str, cfg: StudyConfig) -> nn.Module:
    seq_len = cfg.state_cfg().seq_len
    token_dim = cfg.state_cfg().token_dim
    if name == "flat_mlp":
        return FlatMLPRanker(seq_len, token_dim, cfg.hidden_dim, cfg.max_log_residual)
    if name == "masked_pool":
        return MaskedPoolRanker(token_dim, cfg.hidden_dim, cfg.max_log_residual)
    if name.startswith("hrm_"):
        return RecurrentRanker(
            "hrm", name.removeprefix("hrm_"), token_dim, cfg.hidden_dim, cfg.max_log_residual
        )
    if name.startswith("onlstm_"):
        return RecurrentRanker(
            "onlstm", name.removeprefix("onlstm_"), token_dim, cfg.hidden_dim, cfg.max_log_residual
        )
    raise KeyError(name)


def predict_model(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    mode_override: Optional[str] = None,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            if mode_override is not None and isinstance(model, RecurrentRanker):
                pred = model.forward_mode(xb, mode_override)
            else:
                pred = model(xb)
            outputs.append(pred.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float64)


# ---------------------------------------------------------------------------
# Dataset, linear controls, and learning curve
# ---------------------------------------------------------------------------


def collect_study_datasets(cfg: StudyConfig) -> Tuple[Path, Path]:
    td_cfg = TD.C13TDConfig(
        out_dir=cfg.out_dir,
        train_suites=cfg.suite,
        eval_suites=cfg.suite,
        train_worlds=cfg.train_worlds,
        val_worlds=cfg.val_worlds,
        eval_worlds=cfg.eval_worlds,
        train_nodes=cfg.roadmap_nodes,
        density_nodes=str(cfg.roadmap_nodes),
        roadmap_k=cfg.roadmap_k,
        seed=cfg.seed,
        max_world_retries=cfg.max_world_retries,
        sensor_radius_frac=cfg.sensor_radius_frac,
        num_rays=cfg.num_rays,
        ray_steps=cfg.ray_steps,
        max_neighbors=cfg.max_neighbors,
        rollouts_per_start=cfg.rollouts_per_start,
        max_steps_factor=cfg.max_steps_factor,
        max_norm_residual=cfg.max_log_residual,
    )
    train = TD.collect_rollout_dataset(
        td_cfg,
        "train",
        [cfg.suite],
        cfg.train_worlds,
        seed_offset=0,
    )
    val = TD.collect_rollout_dataset(
        td_cfg,
        "val",
        [cfg.suite],
        cfg.val_worlds,
        seed_offset=500_000,
    )
    return train, val


def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def split_selected_worlds(
    world_id: np.ndarray,
    n_worlds: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    selected_worlds = np.sort(np.unique(world_id))[: int(n_worlds)]
    rng = np.random.default_rng(int(seed) + int(n_worlds) * 101)
    train_parts: List[np.ndarray] = []
    within_parts: List[np.ndarray] = []
    for world in selected_worlds:
        indices = np.flatnonzero(world_id == world)
        shuffled = rng.permutation(indices)
        n_within = max(1, int(round(0.20 * len(shuffled))))
        within_parts.append(shuffled[:n_within])
        train_parts.append(shuffled[n_within:])
    return np.concatenate(train_parts), np.concatenate(within_parts)


def feature_views(x: np.ndarray, num_rays: int) -> Dict[str, np.ndarray]:
    arr = np.asarray(x, dtype=np.float64)
    neighbor_start = 1 + int(num_rays)
    mask = np.abs(arr).sum(axis=2) > 0.0
    neighbors = arr[:, neighbor_start:, 4:]
    neighbor_mask = mask[:, neighbor_start:]
    denom = neighbor_mask.sum(axis=1, keepdims=True).clip(min=1)
    mean_neighbor = (neighbors * neighbor_mask[:, :, None]).sum(axis=1) / denom
    centered = (neighbors - mean_neighbor[:, None, :]) * neighbor_mask[:, :, None]
    std_neighbor = np.sqrt((centered * centered).sum(axis=1) / denom)
    max_neighbor = np.max(
        np.where(neighbor_mask[:, :, None], neighbors, -np.inf), axis=1
    )
    max_neighbor[~np.isfinite(max_neighbor)] = 0.0
    compact = np.concatenate(
        [
            arr[:, 0, 4:],
            arr[:, 1:neighbor_start, 4:].reshape(len(arr), -1),
            mean_neighbor,
            std_neighbor,
            max_neighbor,
            neighbor_mask.sum(axis=1, keepdims=True),
        ],
        axis=1,
    )
    return {
        "summary": arr[:, 0, 4:],
        "rays": arr[:, 1:neighbor_start].reshape(len(arr), -1),
        "actions": arr[:, neighbor_start:].reshape(len(arr), -1),
        "compact": compact,
        "full": arr.reshape(len(arr), -1),
    }


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Dict[str, np.ndarray]:
    features = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    z = (features - mean) / scale
    design = np.concatenate([np.ones((len(z), 1)), z], axis=1)
    regularizer = np.eye(design.shape[1]) * float(alpha)
    regularizer[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    return {"mean": mean, "scale": scale, "coef": coef}


def predict_ridge(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["scale"]
    design = np.concatenate([np.ones((len(z), 1)), z], axis=1)
    return np.clip(design @ model["coef"], 0.0, None)


def predict_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    train_z = (x_train - mean) / scale
    eval_z = (x_eval - mean) / scale
    distances = cdist(eval_z, train_z, metric="sqeuclidean")
    k_eff = min(int(k), len(train_z))
    nearest = np.argpartition(distances, kth=k_eff - 1, axis=1)[:, :k_eff]
    return np.mean(np.asarray(y_train)[nearest], axis=1)


def run_linear_and_aliasing_controls(
    cfg: StudyConfig,
    train_data: Mapping[str, np.ndarray],
    val_data: Mapping[str, np.ndarray],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Mapping[str, np.ndarray]]]:
    train_idx, within_idx = split_selected_worlds(
        train_data["world_id"], cfg.train_worlds, cfg.seed + 55
    )
    train_views = feature_views(train_data["x"], cfg.num_rays)
    val_views = feature_views(val_data["x"], cfg.num_rays)
    rows: List[Dict[str, Any]] = []
    ridge_models: Dict[str, Mapping[str, np.ndarray]] = {}
    for view_name in ("summary", "rays", "actions", "compact", "full"):
        model = fit_ridge(train_views[view_name][train_idx], train_data["y"][train_idx])
        ridge_models[view_name] = model
        for split, indices, views, target in (
            ("same_world_nodes", within_idx, train_views, train_data["y"]),
            ("heldout_worlds", np.arange(len(val_data["y"])), val_views, val_data["y"]),
        ):
            metrics = regression_metrics(
                predict_ridge(model, views[view_name][indices]), target[indices]
            )
            rows.append({"model": f"ridge_{view_name}", "split": split, **metrics})

    for view_name in ("summary", "compact"):
        for split, indices, views, target in (
            ("same_world_nodes", within_idx, train_views, train_data["y"]),
            ("heldout_worlds", np.arange(len(val_data["y"])), val_views, val_data["y"]),
        ):
            pred = predict_knn(
                train_views[view_name][train_idx],
                train_data["y"][train_idx],
                views[view_name][indices],
                k=5,
            )
            rows.append(
                {"model": f"knn5_{view_name}", "split": split, **regression_metrics(pred, target[indices])}
            )

    # Cross-world nearest-neighbor target gaps estimate observation aliasing.
    alias_rng = np.random.default_rng(cfg.seed + 909)
    all_indices = np.arange(len(train_data["y"]))
    if len(all_indices) > 1200:
        all_indices = np.sort(alias_rng.choice(all_indices, size=1200, replace=False))
    compact = train_views["compact"][all_indices]
    target = train_data["y"][all_indices].astype(np.float64)
    worlds = train_data["world_id"][all_indices]
    mean = compact.mean(axis=0)
    scale = compact.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    distance = cdist((compact - mean) / scale, (compact - mean) / scale, metric="sqeuclidean")
    distance[worlds[:, None] == worlds[None, :]] = np.inf
    nearest = np.argmin(distance, axis=1)
    nearest_gap = np.abs(target - target[nearest])
    random_partner = np.empty(len(worlds), dtype=np.int64)
    for idx, world in enumerate(worlds):
        candidates = np.flatnonzero(worlds != world)
        random_partner[idx] = int(alias_rng.choice(candidates))
    random_gap = np.abs(target - target[random_partner])
    aliasing = {
        "n": int(len(target)),
        "nearest_cross_world_feature_distance_mean": float(np.mean(np.sqrt(distance[np.arange(len(target)), nearest]))),
        "nearest_cross_world_target_gap_mean": float(np.mean(nearest_gap)),
        "nearest_cross_world_target_gap_median": float(np.median(nearest_gap)),
        "random_cross_world_target_gap_mean": float(np.mean(random_gap)),
        "nearest_gap_fraction_of_random": float(np.mean(nearest_gap) / max(C.EPS, np.mean(random_gap))),
    }
    return rows, aliasing, ridge_models


def train_one_model(
    cfg: StudyConfig,
    name: str,
    n_worlds: int,
    train_data: Mapping[str, np.ndarray],
    val_data: Mapping[str, np.ndarray],
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    train_idx, within_idx = split_selected_worlds(
        train_data["world_id"], n_worlds, cfg.seed + 701
    )
    seed = cfg.seed + sum(ord(ch) for ch in name) + n_worlds * 1009
    C.set_global_seed(seed)
    model = build_model(name, cfg).to(device)
    ds = TensorDataset(
        torch.from_numpy(train_data["x"][train_idx].astype(np.float32)),
        torch.from_numpy(train_data["y"][train_idx].astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed + 1)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_mae = float("inf")
    history: List[Dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses: List[float] = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss for {name}/{n_worlds}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
        heldout = regression_metrics(
            predict_model(model, val_data["x"], device), val_data["y"]
        )
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **heldout})
        if heldout["mae"] < best_mae:
            best_mae = heldout["mae"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError(f"no valid epoch for {name}/{n_worlds}")
    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    train_metrics = regression_metrics(
        predict_model(model, train_data["x"][train_idx], device),
        train_data["y"][train_idx],
    )
    within_metrics = regression_metrics(
        predict_model(model, train_data["x"][within_idx], device),
        train_data["y"][within_idx],
    )
    heldout_metrics = regression_metrics(
        predict_model(model, val_data["x"], device), val_data["y"]
    )
    row: Dict[str, Any] = {
        "model": name,
        "train_worlds": int(n_worlds),
        "train_samples": int(len(train_idx)),
        "same_world_samples": int(len(within_idx)),
        "best_epoch": int(best_epoch),
    }
    for prefix, metrics in (
        ("train", train_metrics),
        ("same_world", within_metrics),
        ("heldout_world", heldout_metrics),
    ):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    row["history"] = history
    print(
        f"study {name} worlds={n_worlds}: same_corr={within_metrics['pearson']:.3f} "
        f"heldout_corr={heldout_metrics['pearson']:.3f} epoch={best_epoch}",
        flush=True,
    )
    return model, row


def run_model_learning_curve(
    cfg: StudyConfig,
    train_data: Mapping[str, np.ndarray],
    val_data: Mapping[str, np.ndarray],
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, nn.Module]]:
    rows: List[Dict[str, Any]] = []
    final_models: Dict[str, nn.Module] = {}
    checkpoint_dir = C13.ensure_dir(Path(cfg.out_dir) / "checkpoints")
    for name, requested_counts in MODEL_SCHEDULE.items():
        counts = tuple(count for count in requested_counts if count <= cfg.train_worlds)
        if not counts:
            counts = (cfg.train_worlds,)
        for n_worlds in counts:
            model, row = train_one_model(
                cfg, name, n_worlds, train_data, val_data, device
            )
            history = row.pop("history")
            rows.append(row)
            if n_worlds == max(counts):
                final_models[name] = model
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_name": name,
                        "config": asdict(cfg),
                        "target_source": TD.TARGET_SOURCE,
                        "target_transform": TD.TARGET_TRANSFORM,
                        "shortest_path_target": False,
                        "history": history,
                        "metrics": row,
                    },
                    checkpoint_dir / f"{name}.pt",
                )
    return rows, final_models


# ---------------------------------------------------------------------------
# Target reliability and integration audit
# ---------------------------------------------------------------------------


@dataclass
class AuditBundle:
    world: C.World
    roadmap: C.Roadmap
    features: np.ndarray
    rollout_rank: np.ndarray
    node_rows: List[Dict[str, Any]]
    world_index: int
    world_seed: int


def collect_audit_bundles(cfg: StudyConfig) -> List[AuditBundle]:
    M7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    if cfg.suite not in specs:
        raise KeyError(cfg.suite)
    bundles: List[AuditBundle] = []
    candidate_worlds = cfg.eval_worlds * cfg.max_world_retries
    policy = cfg.policy_cfg(cfg.audit_rollouts_per_start)
    for _, world, world_seed in C13.iter_worlds(
        specs[cfg.suite],
        0,
        candidate_worlds,
        cfg.seed + 900_000,
        retry=cfg.max_world_retries,
    ):
        rm = C.build_prm(
            world,
            C.RoadmapConfig(n_nodes=cfg.roadmap_nodes, k_neighbors=cfg.roadmap_k),
            seed=world_seed + 17,
        )
        if rm is None:
            continue
        features = C13.make_local_state_features(
            world, rm.points, rm.adj, cfg.state_cfg()
        )
        euclid = C13.euclidean_to_goal(rm.points, rm.points[1])
        rollout_rank = np.full(len(rm.points), np.nan, dtype=np.float64)
        rows: List[Dict[str, Any]] = []
        for node in range(len(rm.points)):
            costs: List[float] = []
            costs_by_rollout: List[float] = []
            for rollout_index in range(cfg.audit_rollouts_per_start):
                rollout_seed = (
                    world_seed * 1_000_003
                    + node * 10_007
                    + rollout_index * 97
                    + 71_003
                )
                episode = TD.local_behavior_rollout(
                    rm.points,
                    rm.adj,
                    node,
                    world.side_len,
                    random.Random(rollout_seed),
                    policy,
                )
                if episode.found:
                    costs.append(episode.total_cost)
                    costs_by_rollout.append(episode.total_cost)
                else:
                    costs_by_rollout.append(float("nan"))
            half = max(1, cfg.audit_rollouts_per_start // 2)
            first = [value for value in costs_by_rollout[:half] if math.isfinite(value)]
            second = [value for value in costs_by_rollout[half:] if math.isfinite(value)]
            median = float(np.median(costs)) if costs else float("nan")
            if math.isfinite(median):
                rollout_rank[node] = median
            oracle = float(rm.dist_to_goal[node])
            connected = math.isfinite(oracle) and oracle < C.INF / 10.0
            raw_residual = (
                max(0.0, median - float(euclid[node])) / float(world.side_len)
                if math.isfinite(median)
                else float("nan")
            )
            oracle_residual = (
                max(0.0, oracle - float(euclid[node])) / float(world.side_len)
                if connected
                else float("nan")
            )
            rows.append(
                {
                    "suite": cfg.suite,
                    "world_index": len(bundles),
                    "world_seed": world_seed,
                    "node": node,
                    "connected_eval_only": connected,
                    "euclid": float(euclid[node]),
                    "oracle_eval_only": oracle if connected else "",
                    "oracle_residual_eval_only": oracle_residual if connected else "",
                    "rollout_successes": len(costs),
                    "rollout_success_rate": len(costs) / cfg.audit_rollouts_per_start,
                    "rollout_median": median if math.isfinite(median) else "",
                    "rollout_iqr": float(np.subtract(*np.percentile(costs, [75, 25]))) if costs else "",
                    "rollout_half_a": float(np.median(first)) if first else "",
                    "rollout_half_b": float(np.median(second)) if second else "",
                    "rollout_to_oracle_ratio": median / oracle
                    if connected and math.isfinite(median) and oracle > C.EPS
                    else "",
                    "rollout_residual": raw_residual if math.isfinite(raw_residual) else "",
                    "rollout_target": math.log1p(raw_residual)
                    if math.isfinite(raw_residual)
                    else "",
                }
            )
        finite = np.isfinite(rollout_rank)
        penalty = (
            float(np.max(rollout_rank[finite] - euclid[finite]) + world.side_len)
            if finite.any()
            else 4.0 * world.side_len
        )
        rollout_rank[~finite] = euclid[~finite] + penalty
        bundles.append(
            AuditBundle(
                world=world,
                roadmap=rm,
                features=features,
                rollout_rank=rollout_rank,
                node_rows=rows,
                world_index=len(bundles),
                world_seed=world_seed,
            )
        )
        print(
            f"audit world {len(bundles)}/{cfg.eval_worlds}: "
            f"rollout_label_rate={float(np.mean(finite)):.3f}",
            flush=True,
        )
        if len(bundles) >= cfg.eval_worlds:
            break
    if len(bundles) < cfg.eval_worlds:
        raise RuntimeError(f"audit under-filled: {len(bundles)}/{cfg.eval_worlds}")
    return bundles


def summarize_target_audit(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def values(key: str) -> np.ndarray:
        return np.array(
            [float(row[key]) if row.get(key, "") != "" else np.nan for row in rows],
            dtype=np.float64,
        )

    connected = np.array([bool(row["connected_eval_only"]) for row in rows])
    rollout = values("rollout_median")
    oracle = values("oracle_eval_only")
    euclid = values("euclid")
    half_a = values("rollout_half_a")
    half_b = values("rollout_half_b")
    rollout_residual = values("rollout_residual")
    oracle_residual = values("oracle_residual_eval_only")
    ratio = values("rollout_to_oracle_ratio")
    iqr = values("rollout_iqr")
    valid = connected & np.isfinite(rollout)
    return {
        "nodes": len(rows),
        "connected_nodes_eval_only": int(np.sum(connected)),
        "rollout_labeled_connected_nodes": int(np.sum(valid)),
        "rollout_label_rate_on_connected": float(np.mean(np.isfinite(rollout[connected]))),
        "split_half_return_pearson": safe_pearson(half_a[valid], half_b[valid]),
        "split_half_return_spearman": safe_spearman(half_a[valid], half_b[valid]),
        "rollout_vs_oracle_cost_pearson": safe_pearson(rollout[valid], oracle[valid]),
        "rollout_vs_oracle_cost_spearman": safe_spearman(rollout[valid], oracle[valid]),
        "rollout_vs_euclid_cost_pearson": safe_pearson(rollout[valid], euclid[valid]),
        "rollout_vs_oracle_residual_pearson": safe_pearson(
            rollout_residual[valid], oracle_residual[valid]
        ),
        "rollout_vs_oracle_residual_spearman": safe_spearman(
            rollout_residual[valid], oracle_residual[valid]
        ),
        "rollout_to_oracle_ratio_median": float(np.nanmedian(ratio[valid])),
        "rollout_to_oracle_ratio_p90": float(np.nanquantile(ratio[valid], 0.90)),
        "rollout_iqr_median": float(np.nanmedian(iqr[valid])),
        "rollout_iqr_p90": float(np.nanquantile(iqr[valid], 0.90)),
    }


def focal_search_with_secondary(
    adj: List[List[Tuple[int, float]]],
    euclid_h: np.ndarray,
    rank_h: np.ndarray,
    budget: int,
    w: float,
    secondary: str = "h",
    start_idx: int = 0,
    goal_idx: int = 1,
) -> Dict[str, Any]:
    if w < 1.0:
        raise ValueError("w must be >= 1")
    if secondary not in {"h", "fhat", "residual"}:
        raise ValueError(secondary)
    n = len(adj)
    g = np.full(n, np.inf, dtype=np.float64)
    g[start_idx] = 0.0
    counter = 0
    opened: List[Tuple[float, float, int, int]] = [
        (float(euclid_h[start_idx]), 0.0, start_idx, counter)
    ]
    closed = np.zeros(n, dtype=np.bool_)
    expansions = 0
    while opened and expansions < int(budget):
        opened = [entry for entry in opened if not closed[entry[2]] and entry[1] == g[entry[2]]]
        if not opened:
            break
        f_min = min(entry[0] for entry in opened)
        focal = [entry for entry in opened if entry[0] <= float(w) * f_min + 1.0e-12]

        def secondary_key(entry: Tuple[float, float, int, int]) -> Tuple[float, ...]:
            f_value, g_value, node, insertion = entry
            if secondary == "h":
                primary = float(rank_h[node])
            elif secondary == "fhat":
                primary = float(g_value + rank_h[node])
            else:
                primary = float(rank_h[node] - euclid_h[node])
            return primary, float(rank_h[node]), f_value, insertion

        best = min(focal, key=secondary_key)
        opened.remove(best)
        _, current_g, node, _ = best
        if closed[node] or current_g != g[node]:
            continue
        closed[node] = True
        expansions += 1
        if node == goal_idx:
            return {
                "found": True,
                "cost": float(g[node]),
                "expansions": expansions,
                "closed": int(closed.sum()),
            }
        for neighbor, edge_cost in adj[node]:
            if closed[neighbor]:
                continue
            new_g = g[node] + float(edge_cost)
            if new_g < g[neighbor]:
                g[neighbor] = new_g
                counter += 1
                opened.append(
                    (
                        new_g + float(euclid_h[neighbor]),
                        new_g,
                        int(neighbor),
                        counter,
                    )
                )
    return {
        "found": False,
        "cost": float("nan"),
        "expansions": expansions,
        "closed": int(closed.sum()),
    }


def run_integration_audit(
    cfg: StudyConfig,
    bundles: Sequence[AuditBundle],
    models: Mapping[str, nn.Module],
    ridge_models: Mapping[str, Mapping[str, np.ndarray]],
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    focal_ws = C13.parse_float_csv(cfg.focal_ws)
    for bundle in bundles:
        rm = bundle.roadmap
        world = bundle.world
        euclid = C13.euclidean_to_goal(rm.points, rm.points[1])
        one_step = C13.one_step_euclidean_backup(rm.points, rm.adj)
        oracle = C13._finite_oracle(rm, world.side_len)
        optimal = float(rm.dist_to_goal[0])
        ranks: Dict[str, np.ndarray] = {
            "euclid": euclid,
            "one_step": one_step,
            "oracle_eval_only": oracle,
            "rollout_exact": bundle.rollout_rank,
        }
        for alpha in (0.25, 0.50, 0.75):
            ranks[f"rollout_blend_{alpha:.2f}"] = (
                (1.0 - alpha) * euclid + alpha * bundle.rollout_rank
            )
        for name, model in models.items():
            transformed = predict_model(model, bundle.features, device)
            decoded = euclid + world.side_len * np.expm1(
                np.clip(transformed, 0.0, cfg.max_log_residual)
            )
            ranks[name] = decoded
            ranks[f"{name}_blend_0.50"] = 0.5 * euclid + 0.5 * decoded
        compact = feature_views(bundle.features, cfg.num_rays)["compact"]
        ridge_pred = predict_ridge(ridge_models["compact"], compact)
        ranks["ridge_compact"] = euclid + world.side_len * np.expm1(
            np.clip(ridge_pred, 0.0, cfg.max_log_residual)
        )

        baseline_focal_ws: set[float] = set()
        for provider, rank in ranks.items():
            modes = ("h",)
            if provider in {"oracle_eval_only", "rollout_exact", "ridge_compact"} or provider in models:
                modes = ("h", "fhat", "residual")
            for focal_w in focal_ws:
                for secondary in modes:
                    t0 = time.perf_counter()
                    result = focal_search_with_secondary(
                        rm.adj,
                        euclid,
                        rank,
                        budget=len(rm.points),
                        w=focal_w,
                        secondary=secondary,
                    )
                    elapsed = time.perf_counter() - t0
                    if provider == "euclid" and secondary == "h":
                        baseline_focal_ws.add(float(focal_w))
                    cost = float(result["cost"])
                    rows.append(
                        {
                            "suite": cfg.suite,
                            "world_index": bundle.world_index,
                            "world_seed": bundle.world_seed,
                            "provider": provider,
                            "secondary": secondary,
                            "focal_w": float(focal_w),
                            "found": bool(result["found"]),
                            "expansions": int(result["expansions"]),
                            "cost": cost if math.isfinite(cost) else "",
                            "cost_ratio": cost / optimal if math.isfinite(cost) else "",
                            "search_seconds": elapsed,
                        }
                    )
        # Primary-heuristic A* checks whether a useful estimate is being lost
        # specifically by FOCAL's Euclidean-anchored secondary-key insertion.
        astar_ranks: Dict[str, np.ndarray] = {
            "euclid_astar": euclid,
            "one_step_astar_primary": one_step,
            "oracle_astar_eval_only": oracle,
            "rollout_exact_astar_primary": bundle.rollout_rank,
            "ridge_compact_astar_primary": ranks["ridge_compact"],
        }
        for name in models:
            astar_ranks[f"{name}_astar_primary"] = ranks[name]
        for provider, heuristic in astar_ranks.items():
            result = C.astar_search(rm.adj, heuristic, len(rm.points))
            cost = float(result["cost"])
            rows.append(
                {
                    "suite": cfg.suite,
                    "world_index": bundle.world_index,
                    "world_seed": bundle.world_seed,
                    "provider": provider,
                    "secondary": "astar_primary",
                    "focal_w": "",
                    "found": bool(result["found"]),
                    "expansions": int(result["expansions"]),
                    "cost": cost if math.isfinite(cost) else "",
                    "cost_ratio": cost / optimal if math.isfinite(cost) else "",
                    "search_seconds": 0.0,
                }
            )
        if baseline_focal_ws != set(focal_ws):
            raise RuntimeError("Euclidean focal baseline missing for one or more weights")

    baseline = {
        (int(row["world_index"]), float(row["focal_w"])): float(row["expansions"])
        for row in rows
        if row["provider"] == "euclid" and row["secondary"] == "h"
    }
    baseline_astar = {
        int(row["world_index"]): float(row["expansions"])
        for row in rows
        if row["provider"] == "euclid_astar"
    }
    grouped: DefaultDict[Tuple[str, str, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["provider"]), str(row["secondary"]), row["focal_w"])].append(row)
    summary: List[Dict[str, Any]] = []
    for (provider, secondary, focal_w), group in sorted(
        grouped.items(), key=lambda item: str(item[0])
    ):
        ratios = [float(row["cost_ratio"]) for row in group if row["cost_ratio"] != ""]
        deltas = (
            [
                float(row["expansions"])
                - baseline[(int(row["world_index"]), float(focal_w))]
                for row in group
            ]
            if focal_w != ""
            else []
        )
        astar_deltas = (
            [
                float(row["expansions"]) - baseline_astar[int(row["world_index"])]
                for row in group
            ]
            if focal_w == ""
            else []
        )
        summary.append(
            {
                "provider": provider,
                "secondary": secondary,
                "focal_w": focal_w,
                "worlds": len(group),
                "success_rate": float(np.mean([bool(row["found"]) for row in group])),
                "expansions_mean": float(np.mean([float(row["expansions"]) for row in group])),
                "expansion_delta_vs_euclid_focal_mean": float(np.mean(deltas)) if deltas else "",
                "expansion_delta_vs_euclid_astar_mean": float(np.mean(astar_deltas)) if astar_deltas else "",
                "cost_ratio_mean": float(np.mean(ratios)) if ratios else "",
                "cost_ratio_max": float(np.max(ratios)) if ratios else "",
            }
        )
    return rows, summary


def padding_diagnostics(
    cfg: StudyConfig,
    val_data: Mapping[str, np.ndarray],
    models: Mapping[str, nn.Module],
    device: torch.device,
) -> Dict[str, Any]:
    lengths = np.asarray(sequence_lengths(val_data["x"]), dtype=np.int64)
    result: Dict[str, Any] = {
        "sequence_length": int(val_data["x"].shape[1]),
        "real_tokens_mean": float(np.mean(lengths)),
        "padding_tokens_mean": float(val_data["x"].shape[1] - np.mean(lengths)),
        "real_tokens_min": int(np.min(lengths)),
        "real_tokens_max": int(np.max(lengths)),
    }
    for name in ("hrm_padded", "onlstm_padded"):
        if name not in models:
            continue
        model = models[name]
        padded = predict_model(model, val_data["x"], device)
        trimmed = predict_model(model, val_data["x"], device, mode_override="trimmed")
        result[name] = {
            "padded_metrics": regression_metrics(padded, val_data["y"]),
            "same_weights_trimmed_metrics": regression_metrics(trimmed, val_data["y"]),
            "mean_abs_prediction_shift": float(np.mean(np.abs(padded - trimmed))),
            "max_abs_prediction_shift": float(np.max(np.abs(padded - trimmed))),
        }
    return result


def write_outputs(
    cfg: StudyConfig,
    linear_rows: Sequence[Dict[str, Any]],
    aliasing: Mapping[str, Any],
    curve_rows: Sequence[Dict[str, Any]],
    target_rows: Sequence[Dict[str, Any]],
    target_summary: Mapping[str, Any],
    integration_rows: Sequence[Dict[str, Any]],
    integration_summary: Sequence[Dict[str, Any]],
    padding: Mapping[str, Any],
) -> Dict[str, Path]:
    out = C13.ensure_dir(Path(cfg.out_dir) / "results")
    paths = {
        "linear": C13.write_csv(out / "linear_representation_controls.csv", list(linear_rows)),
        "curve": C13.write_csv(out / "model_learning_curve.csv", list(curve_rows)),
        "target_raw": C13.write_csv(out / "target_reliability_raw.csv", list(target_rows)),
        "integration_raw": C13.write_csv(out / "integration_raw.csv", list(integration_rows)),
        "integration_summary": C13.write_csv(out / "integration_summary.csv", list(integration_summary)),
        "diagnostics": C13.write_json(
            out / "diagnostics.json",
            {
                "target": dict(target_summary),
                "aliasing": dict(aliasing),
                "padding": dict(padding),
            },
        ),
    }
    paths["manifest"] = C13.write_json(
        Path(cfg.out_dir) / "manifest.json",
        {
            "experiment": "C13-B identifiability study",
            "config": asdict(cfg),
            "target_source": TD.TARGET_SOURCE,
            "target_transform": TD.TARGET_TRANSFORM,
            "shortest_path_target": False,
            "oracle_role": "target-correlation_and_search-ceiling_evaluation_only",
            "questions": [
                "target_reliability",
                "bounded-observation_identifiability",
                "padding_and_readout_representation",
                "focal_secondary_integration",
            ],
            "outputs": {key: str(value) for key, value in paths.items()},
        },
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-B multi-angle identifiability study")
    parser.add_argument("--out-dir", default=StudyConfig.out_dir)
    parser.add_argument("--suite", default=StudyConfig.suite)
    parser.add_argument("--train-worlds", type=int, default=StudyConfig.train_worlds)
    parser.add_argument("--val-worlds", type=int, default=StudyConfig.val_worlds)
    parser.add_argument("--eval-worlds", type=int, default=StudyConfig.eval_worlds)
    parser.add_argument("--roadmap-nodes", type=int, default=StudyConfig.roadmap_nodes)
    parser.add_argument("--roadmap-k", type=int, default=StudyConfig.roadmap_k)
    parser.add_argument("--rollouts-per-start", type=int, default=StudyConfig.rollouts_per_start)
    parser.add_argument(
        "--audit-rollouts-per-start", type=int, default=StudyConfig.audit_rollouts_per_start
    )
    parser.add_argument("--epochs", type=int, default=StudyConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=StudyConfig.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=StudyConfig.hidden_dim)
    parser.add_argument("--lr", type=float, default=StudyConfig.lr)
    parser.add_argument("--focal-ws", default=StudyConfig.focal_ws)
    parser.add_argument("--seed", type=int, default=StudyConfig.seed)
    parser.add_argument("--device", default=StudyConfig.device)
    return parser.parse_args()


def main() -> None:
    cfg = StudyConfig(**vars(parse_args()))
    device = resolve_device(cfg.device)
    C.set_global_seed(cfg.seed)
    train_path, val_path = collect_study_datasets(cfg)
    train_data = load_dataset(train_path)
    val_data = load_dataset(val_path)
    linear_rows, aliasing, ridge_models = run_linear_and_aliasing_controls(
        cfg, train_data, val_data
    )
    curve_rows, models = run_model_learning_curve(
        cfg, train_data, val_data, device
    )
    bundles = collect_audit_bundles(cfg)
    target_rows = [row for bundle in bundles for row in bundle.node_rows]
    target_summary = summarize_target_audit(target_rows)
    integration_rows, integration_summary = run_integration_audit(
        cfg, bundles, models, ridge_models, device
    )
    padding = padding_diagnostics(cfg, val_data, models, device)
    outputs = write_outputs(
        cfg,
        linear_rows,
        aliasing,
        curve_rows,
        target_rows,
        target_summary,
        integration_rows,
        integration_summary,
        padding,
    )
    for key, value in outputs.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
