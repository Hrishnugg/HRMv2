#!/usr/bin/env python3
"""C13-H: current-state limited-horizon Bellman learning.

The training target is a radius-bounded Bellman backup.  A local Dijkstra
search uses only edges inside the current observation and bootstraps every
visible exit action from a frozen state-value model evaluated at that next
state.  It never reads graph ``dist_to_goal`` or another full-problem solution.

At runtime each learned provider receives only current/goal geometry, bounded
rays, and one-hop actions.  A consistent Euclidean anchor certifies every
returned path; learned values only rank the shared auxiliary queue.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import continuous_prm_common as C
import continuous_prm_c13_certified_search as S
import continuous_prm_c13_identifiability as I
import continuous_prm_c13_shared_queue as Q
import continuous_prm_c13_state_heuristic as C13
import continuous_prm_c7_hard_maps as M7


@dataclass
class LHBLConfig:
    study_dir: str = "runs/c13_identifiability"
    out_dir: str = "runs/c13_lhbl"
    models: str = "flat_mlp,masked_pool,hrm_trimmed"
    train_worlds: int = 12
    validation_worlds: int = 4
    outer_iterations: int = 8
    inner_epochs: int = 5
    batch_size: int = 128
    hidden_dim: int = 64
    lr: float = 5.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    sensor_radius_frac: float = 0.20
    num_rays: int = 32
    ray_steps: int = 32
    max_neighbors: int = 24
    max_norm_residual: float = 4.0
    alphas: str = "0.25,0.50,1.00"
    focal_ws: str = "1.05,1.10,1.25"
    primary_w: float = 1.10
    required_win_fraction: float = 0.80
    budget_factor: float = 2.0
    seed: int = 7413
    device: str = "auto"


@dataclass
class WorldBundle:
    split: str
    suite: str
    world_index: int
    world_seed: int
    world: C.World
    roadmap: C.Roadmap
    features: np.ndarray


class AttentionPoolRanker(nn.Module):
    """Permutation-safe token interaction model with a summary readout."""

    def __init__(self, token_dim: int, hidden_dim: int, max_output: float):
        super().__init__()
        if int(hidden_dim) % 4 != 0:
            raise ValueError("attention hidden_dim must be divisible by four")
        self.input = nn.Linear(int(token_dim), int(hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=4,
            dim_feedforward=int(hidden_dim) * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = I.PositiveHead(int(hidden_dim), int(hidden_dim), float(max_output))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = ~torch.any(torch.abs(x) > 0.0, dim=-1)
        padding[:, 0] = False
        encoded = self.encoder(self.input(x), src_key_padding_mask=padding)
        return self.head(encoded[:, 0])


def build_lhbl_model(name: str, cfg: I.StudyConfig) -> nn.Module:
    if name == "attention_pool":
        return AttentionPoolRanker(
            cfg.state_cfg().token_dim, cfg.hidden_dim, cfg.max_log_residual
        )
    return I.build_model(name, cfg)


def resolve_device(name: str) -> torch.device:
    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_names(value: str) -> List[str]:
    names = [item.strip() for item in str(value).split(",") if item.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("models must be a nonempty unique comma-separated list")
    return names


def state_config(cfg: LHBLConfig) -> C13.LocalStateConfig:
    return C13.LocalStateConfig(
        sensor_radius_frac=float(cfg.sensor_radius_frac),
        num_rays=int(cfg.num_rays),
        ray_steps=int(cfg.ray_steps),
        max_neighbors=int(cfg.max_neighbors),
    )


def model_config(cfg: LHBLConfig, study_cfg: I.StudyConfig) -> I.StudyConfig:
    configured = I.StudyConfig(**asdict(study_cfg))
    configured.sensor_radius_frac = float(cfg.sensor_radius_frac)
    configured.num_rays = int(cfg.num_rays)
    configured.ray_steps = int(cfg.ray_steps)
    configured.max_neighbors = int(cfg.max_neighbors)
    configured.hidden_dim = int(cfg.hidden_dim)
    configured.max_log_residual = float(cfg.max_norm_residual)
    configured.batch_size = int(cfg.batch_size)
    configured.lr = float(cfg.lr)
    configured.weight_decay = float(cfg.weight_decay)
    configured.grad_clip = float(cfg.grad_clip)
    return configured


def limited_horizon_values(
    points: np.ndarray,
    adj: Sequence[Sequence[Tuple[int, float]]],
    goal_point: np.ndarray,
    bootstrap_values: np.ndarray,
    radius: float,
    goal_idx: int = 1,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Apply one radius-bounded Bellman backup without a global solution."""

    pts = np.asarray(points, dtype=np.float64)
    bootstrap = np.asarray(bootstrap_values, dtype=np.float64).reshape(-1)
    goal = np.asarray(goal_point, dtype=np.float64).reshape(2)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) != len(adj):
        raise ValueError("points and adjacency must describe one 2-D graph")
    if bootstrap.shape != (len(pts),) or not np.all(np.isfinite(bootstrap)):
        raise ValueError("bootstrap values must be finite and node-aligned")
    if float(radius) <= 0.0:
        raise ValueError("radius must be positive")

    euclid = np.linalg.norm(pts - goal[None, :], axis=1)
    values = np.empty(len(pts), dtype=np.float64)
    diagnostics: List[Dict[str, Any]] = []
    for start in range(len(pts)):
        inside_mask = np.linalg.norm(pts - pts[start][None, :], axis=1) <= (
            float(radius) + C.EPS
        )
        inside_mask[start] = True
        inside_ids = np.flatnonzero(inside_mask).astype(np.int64)
        local_of = {int(node): index for index, node in enumerate(inside_ids)}
        distances = np.full(len(inside_ids), np.inf, dtype=np.float64)
        distances[local_of[start]] = 0.0
        heap: List[Tuple[float, int]] = [(0.0, int(start))]
        best = float("inf")
        explored = 0
        exit_actions = 0
        while heap:
            distance, node = __import__("heapq").heappop(heap)
            local_node = local_of[node]
            if distance != float(distances[local_node]):
                continue
            if distance >= best - C.EPS:
                continue
            explored += 1
            if node == int(goal_idx):
                best = min(best, float(distance))
            for neighbor_value, edge_cost_value in adj[node]:
                neighbor = int(neighbor_value)
                edge_cost = float(edge_cost_value)
                local_neighbor = local_of.get(neighbor)
                if local_neighbor is None:
                    exit_actions += 1
                    best = min(
                        best,
                        float(distance) + edge_cost + float(bootstrap[neighbor]),
                    )
                    continue
                candidate = float(distance) + edge_cost
                if candidate + C.EPS >= float(distances[local_neighbor]):
                    continue
                distances[local_neighbor] = candidate
                __import__("heapq").heappush(heap, (candidate, neighbor))
        fallback = not math.isfinite(best)
        if fallback:
            best = float(euclid[start])
        values[start] = max(float(euclid[start]), float(best))
        diagnostics.append(
            {
                "node": int(start),
                "observed_nodes": int(len(inside_ids)),
                "explored_nodes": int(explored),
                "exit_actions": int(exit_actions),
                "fallback": bool(fallback),
            }
        )
    values[int(goal_idx)] = 0.0
    return values, diagnostics


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rebuild_metadata_bundles(
    metadata_path: Path,
    local_cfg: C13.LocalStateConfig,
) -> List[WorldBundle]:
    metadata = _read_json(metadata_path)
    M7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    bundles: List[WorldBundle] = []
    for row in metadata["worlds"]:
        suite = str(row["suite"])
        world_seed = int(row["world_seed"])
        world = C.build_world(specs[suite], world_seed, 0.45)
        if world is None:
            raise RuntimeError(f"could not replay {suite}/{world_seed}")
        rm = C.build_prm(
            world,
            C.RoadmapConfig(
                n_nodes=int(metadata["roadmap_nodes"]),
                k_neighbors=int(metadata["roadmap_k"]),
            ),
            seed=int(row["roadmap_seed"]),
        )
        if rm is None:
            raise RuntimeError(f"could not replay roadmap {suite}/{world_seed}")
        edges = sum(len(neighbors) for neighbors in rm.adj) // 2
        if len(rm.points) != int(row["nodes"]) or edges != int(row["edges"]):
            raise RuntimeError(f"metadata replay mismatch {suite}/{world_seed}")
        features = C13.make_local_state_features(world, rm.points, rm.adj, local_cfg)
        bundles.append(
            WorldBundle(
                split=str(metadata["split"]),
                suite=suite,
                world_index=int(row["world_id"]),
                world_seed=world_seed,
                world=world,
                roadmap=rm,
                features=features,
            )
        )
    return bundles


def rebuild_generated_bundles(
    study_cfg: I.StudyConfig,
    local_cfg: C13.LocalStateConfig,
    split: str,
    worlds: int,
    seed_offset: int,
) -> List[WorldBundle]:
    """Build a deterministic train/validation cohort from the source study recipe."""

    requested = int(worlds)
    if requested <= 0:
        raise ValueError("generated cohort world count must be positive")
    M7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    suite = str(study_cfg.suite)
    if suite not in specs:
        raise KeyError(f"unknown suite {suite!r}; have {sorted(specs)}")
    bundles: List[WorldBundle] = []
    candidates = requested * int(study_cfg.max_world_retries)
    for _, world, world_seed in C13.iter_worlds(
        specs[suite],
        0,
        candidates,
        int(study_cfg.seed) + int(seed_offset),
        retry=int(study_cfg.max_world_retries),
    ):
        rm = C.build_prm(
            world,
            C.RoadmapConfig(
                n_nodes=int(study_cfg.roadmap_nodes),
                k_neighbors=int(study_cfg.roadmap_k),
            ),
            seed=int(world_seed) + 17,
        )
        if rm is None:
            continue
        features = C13.make_local_state_features(world, rm.points, rm.adj, local_cfg)
        bundles.append(
            WorldBundle(
                split=str(split),
                suite=suite,
                world_index=len(bundles),
                world_seed=int(world_seed),
                world=world,
                roadmap=rm,
                features=features,
            )
        )
        if len(bundles) >= requested:
            break
    if len(bundles) != requested:
        raise RuntimeError(
            f"{split} generated cohort under-filled: {len(bundles)}/{requested}"
        )
    return bundles


def rebuild_eval_bundles(
    study_cfg: I.StudyConfig,
    local_cfg: C13.LocalStateConfig,
) -> List[WorldBundle]:
    M7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    bundles: List[WorldBundle] = []
    candidates = int(study_cfg.eval_worlds) * int(study_cfg.max_world_retries)
    for _, world, world_seed in C13.iter_worlds(
        specs[study_cfg.suite],
        0,
        candidates,
        int(study_cfg.seed) + 900_000,
        retry=int(study_cfg.max_world_retries),
    ):
        rm = C.build_prm(
            world,
            C.RoadmapConfig(
                n_nodes=int(study_cfg.roadmap_nodes),
                k_neighbors=int(study_cfg.roadmap_k),
            ),
            seed=world_seed + 17,
        )
        if rm is None:
            continue
        features = C13.make_local_state_features(world, rm.points, rm.adj, local_cfg)
        bundles.append(
            WorldBundle(
                split="development_eval",
                suite=str(study_cfg.suite),
                world_index=len(bundles),
                world_seed=int(world_seed),
                world=world,
                roadmap=rm,
                features=features,
            )
        )
        if len(bundles) >= int(study_cfg.eval_worlds):
            break
    if len(bundles) != int(study_cfg.eval_worlds):
        raise RuntimeError("development evaluation replay under-filled")
    return bundles


def bootstrap_and_targets(
    bundles: Sequence[WorldBundle],
    model: Optional[torch.nn.Module],
    cfg: LHBLConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    fallback_nodes = 0
    observed_nodes: List[int] = []
    exit_actions: List[int] = []
    for bundle in bundles:
        rm = bundle.roadmap
        euclid = C13.euclidean_to_goal(rm.points, rm.points[1])
        if model is None:
            bootstrap = euclid.copy()
        else:
            prediction = I.predict_model(model, bundle.features, device)
            bootstrap = euclid + float(bundle.world.side_len) * prediction
        bootstrap[1] = 0.0
        values, diagnostics = limited_horizon_values(
            rm.points,
            rm.adj,
            rm.points[1],
            bootstrap,
            float(cfg.sensor_radius_frac) * float(bundle.world.side_len),
        )
        residual = np.clip(
            (values - euclid) / float(bundle.world.side_len),
            0.0,
            float(cfg.max_norm_residual),
        )
        residual[1] = 0.0
        xs.append(bundle.features.astype(np.float32))
        ys.append(residual.astype(np.float32))
        fallback_nodes += int(np.sum([bool(row["fallback"]) for row in diagnostics]))
        observed_nodes.extend(int(row["observed_nodes"]) for row in diagnostics)
        exit_actions.extend(int(row["exit_actions"]) for row in diagnostics)
    y = np.concatenate(ys)
    return np.concatenate(xs), y, {
        "samples": int(len(y)),
        "target_mean": float(np.mean(y)),
        "target_p95": float(np.percentile(y, 95)),
        "target_clip_rate": float(np.mean(y >= float(cfg.max_norm_residual) - 1e-9)),
        "positive_rate": float(np.mean(y > 1e-9)),
        "fallback_nodes": int(fallback_nodes),
        "observed_nodes_mean": float(np.mean(observed_nodes)),
        "exit_actions_mean": float(np.mean(exit_actions)),
    }


def build_baselines(
    bundles: Sequence[WorldBundle],
    focal_ws: Sequence[float],
    budget_factor: float,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, float], Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    indexed: Dict[Tuple[int, float], Dict[str, Any]] = {}
    for bundle in bundles:
        rm = bundle.roadmap
        euclid = C13.euclidean_to_goal(rm.points, rm.points[1])
        astar = C.astar_search(rm.adj, euclid, len(rm.points))
        budget = int(math.ceil(float(budget_factor) * len(rm.points)))
        for focal_w in focal_ws:
            focal = I.focal_search_with_secondary(
                rm.adj, euclid, euclid, len(rm.points), float(focal_w), "h"
            )
            shared = Q.shared_anchor_certified_search(
                rm.adj, euclid, euclid, float(focal_w), budget, validate_anchor=False
            )
            row = {
                "suite": bundle.suite,
                "world_index": int(bundle.world_index),
                "world_seed": int(bundle.world_seed),
                "focal_w": float(focal_w),
                "euclid_astar_expansions": int(astar["expansions"]),
                "euclid_focal_expansions": int(focal["expansions"]),
                "euclid_focal_cost": float(focal["cost"]),
                "same_search_euclid_expansions": int(shared["expansions"]),
                "same_search_euclid_certified": bool(shared["certified"]),
            }
            rows.append(row)
            indexed[(bundle.world_index, float(focal_w))] = row
    return rows, indexed


def evaluate_checkpoint(
    model_name: str,
    iteration: int,
    model: torch.nn.Module,
    bundles: Sequence[WorldBundle],
    baselines: Mapping[Tuple[int, float], Mapping[str, Any]],
    cfg: LHBLConfig,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    alphas = C13.parse_float_csv(cfg.alphas)
    focal_ws = C13.parse_float_csv(cfg.focal_ws)
    for bundle in bundles:
        rm = bundle.roadmap
        euclid = C13.euclidean_to_goal(rm.points, rm.points[1])
        prediction = I.predict_model(model, bundle.features, device)
        learned = euclid + float(bundle.world.side_len) * prediction
        oracle = np.asarray(rm.dist_to_goal, dtype=np.float64)  # evaluation-only
        connected = np.isfinite(oracle) & (oracle < C.INF / 10.0)
        diagnostics.append(
            {
                "model": model_name,
                "iteration": int(iteration),
                "world_index": int(bundle.world_index),
                "world_seed": int(bundle.world_seed),
                "prediction_mean": float(np.mean(prediction)),
                "prediction_p95": float(np.percentile(prediction, 95)),
                "prediction_clip_rate": float(
                    np.mean(prediction >= float(cfg.max_norm_residual) - 1e-9)
                ),
                "rank_vs_oracle_spearman_eval_only": I.safe_spearman(
                    learned[connected], oracle[connected]
                ),
                "oracle_overestimate_rate_eval_only": float(
                    np.mean(learned[connected] > oracle[connected] + 1e-9)
                ),
                "start_rank_over_oracle_eval_only": float(learned[0] / oracle[0]),
            }
        )
        budget = int(math.ceil(float(cfg.budget_factor) * len(rm.points)))
        for alpha in alphas:
            rank = euclid + float(alpha) * (learned - euclid)
            direct = C.astar_search(rm.adj, rank, len(rm.points))
            for focal_w in focal_ws:
                result = Q.shared_anchor_certified_search(
                    rm.adj,
                    euclid,
                    rank,
                    float(focal_w),
                    budget,
                    validate_anchor=False,
                )
                final_cost = float(result["final_cost"])
                path = Q.validate_path(rm.adj, result["path"], final_cost)
                baseline = baselines[(bundle.world_index, float(focal_w))]
                optimal = float(oracle[0])
                rows.append(
                    {
                        "suite": bundle.suite,
                        "world_index": int(bundle.world_index),
                        "world_seed": int(bundle.world_seed),
                        "model": model_name,
                        "iteration": int(iteration),
                        "alpha": float(alpha),
                        "focal_w": float(focal_w),
                        "certified": bool(result["certified"]),
                        "found": bool(result["found"]),
                        "proof": result["proof"],
                        "final_cost": final_cost if math.isfinite(final_cost) else "",
                        "final_cost_ratio_eval_only": (
                            final_cost / optimal if math.isfinite(final_cost) else ""
                        ),
                        "bound_violation_eval_only": bool(
                            not math.isfinite(final_cost)
                            or final_cost > float(focal_w) * optimal + 1e-9
                        ),
                        "path_valid": bool(path["valid"]),
                        "expansions": int(result["expansions"]),
                        "rank_expansions": int(result["rank_expansions"]),
                        "anchor_expansions": int(result["anchor_expansions"]),
                        "max_expansions_per_state": int(
                            result["max_expansions_per_state"]
                        ),
                        "rank_eligible_choice_rate": float(
                            result["rank_eligible_choices"]
                            / max(1, result["rank_eligibility_checks"])
                        ),
                        "euclid_focal_expansions": int(
                            baseline["euclid_focal_expansions"]
                        ),
                        "delta_vs_euclid_focal": int(result["expansions"])
                        - int(baseline["euclid_focal_expansions"]),
                        "same_search_euclid_expansions": int(
                            baseline["same_search_euclid_expansions"]
                        ),
                        "delta_vs_same_search_euclid": int(result["expansions"])
                        - int(baseline["same_search_euclid_expansions"]),
                        "euclid_astar_expansions": int(
                            baseline["euclid_astar_expansions"]
                        ),
                        "direct_learned_astar_expansions": int(direct["expansions"]),
                        "direct_learned_astar_cost_ratio_eval_only": float(
                            direct["cost"] / optimal
                        ),
                    }
                )
    return rows, diagnostics


def summarize_search(
    rows: Sequence[Mapping[str, Any]], required_win_fraction: float
) -> List[Dict[str, Any]]:
    grouped: DefaultDict[Tuple[str, int, float, float], List[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[
            (
                str(row["model"]),
                int(row["iteration"]),
                float(row["alpha"]),
                float(row["focal_w"]),
            )
        ].append(row)
    summaries: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        model, iteration, alpha, focal_w = key
        focal_delta = np.asarray([float(row["delta_vs_euclid_focal"]) for row in group])
        control_delta = np.asarray(
            [float(row["delta_vs_same_search_euclid"]) for row in group]
        )
        required = int(math.ceil(float(required_win_fraction) * len(group)))
        safety = int(
            np.sum([not bool(row["certified"]) for row in group])
            + np.sum([bool(row["bound_violation_eval_only"]) for row in group])
            + np.sum([not bool(row["path_valid"]) for row in group])
            + np.sum([int(row["max_expansions_per_state"]) > 2 for row in group])
        )
        focal_wins = int(np.sum(focal_delta < 0))
        control_wins = int(np.sum(control_delta < 0))
        focal_mean = float(np.mean(focal_delta))
        control_mean = float(np.mean(control_delta))
        summaries.append(
            {
                "model": model,
                "iteration": int(iteration),
                "alpha": float(alpha),
                "focal_w": float(focal_w),
                "worlds": int(len(group)),
                "required_wins": int(required),
                "gate_pass": bool(
                    safety == 0
                    and focal_wins >= required
                    and control_wins >= required
                    and focal_mean < 0.0
                    and control_mean < 0.0
                ),
                "safety_failures": int(safety),
                "expansions_mean": float(
                    np.mean([float(row["expansions"]) for row in group])
                ),
                "euclid_focal_expansions_mean": float(
                    np.mean([float(row["euclid_focal_expansions"]) for row in group])
                ),
                "delta_vs_euclid_focal_mean": focal_mean,
                "focal_wins": focal_wins,
                "focal_ties": int(np.sum(focal_delta == 0)),
                "focal_losses": int(np.sum(focal_delta > 0)),
                "same_search_euclid_expansions_mean": float(
                    np.mean(
                        [float(row["same_search_euclid_expansions"]) for row in group]
                    )
                ),
                "delta_vs_same_search_euclid_mean": control_mean,
                "same_search_euclid_wins": control_wins,
                "same_search_euclid_ties": int(np.sum(control_delta == 0)),
                "same_search_euclid_losses": int(np.sum(control_delta > 0)),
                "direct_learned_astar_expansions_mean": float(
                    np.mean(
                        [float(row["direct_learned_astar_expansions"]) for row in group]
                    )
                ),
                "direct_learned_astar_cost_ratio_max_eval_only": float(
                    np.max(
                        [
                            float(row["direct_learned_astar_cost_ratio_eval_only"])
                            for row in group
                        ]
                    )
                ),
                "final_cost_ratio_max_eval_only": float(
                    np.max([float(row["final_cost_ratio_eval_only"]) for row in group])
                ),
                "rank_eligible_choice_rate": float(
                    np.mean([float(row["rank_eligible_choice_rate"]) for row in group])
                ),
            }
        )
    return summaries


def build_verdict(
    summaries: Sequence[Mapping[str, Any]], primary_w: float
) -> Dict[str, Any]:
    primary = [
        dict(row)
        for row in summaries
        if math.isclose(float(row["focal_w"]), float(primary_w), abs_tol=1e-12)
        and bool(row["gate_pass"])
    ]
    selected = (
        min(
            primary,
            key=lambda row: (
                float(row["expansions_mean"]),
                str(row["model"]),
                int(row["iteration"]),
                float(row["alpha"]),
            ),
        )
        if primary
        else None
    )
    return {
        "verdict": (
            "lhbl_development_gate_pass_requires_fresh_replication"
            if selected is not None
            else "lhbl_development_gate_fail"
        ),
        "authorization": (
            "replicate_selected_candidate_on_fresh_192_and_211_worlds"
            if selected is not None
            else "revise_target_or_representation_before_scaling"
        ),
        "primary_w": float(primary_w),
        "development_cohort_reused": True,
        "passing_candidates": int(len(primary)),
        "selected_candidate": selected,
        "fresh_replication_required": selected is not None,
    }


def train_models(
    cfg: LHBLConfig,
    study_cfg: I.StudyConfig,
    train_bundles: Sequence[WorldBundle],
    val_bundles: Sequence[WorldBundle],
    eval_bundles: Sequence[WorldBundle],
    baselines: Mapping[Tuple[int, float], Mapping[str, Any]],
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Path]]:
    histories: List[Dict[str, Any]] = []
    search_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    checkpoint_paths: List[Path] = []
    model_cfg = model_config(cfg, study_cfg)
    checkpoint_dir = C13.ensure_dir(Path(cfg.out_dir) / "checkpoints")

    for model_offset, name in enumerate(parse_names(cfg.models)):
        seed = int(cfg.seed) + 1009 * (model_offset + 1)
        C.set_global_seed(seed)
        model = build_lhbl_model(name, model_cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
        )
        for iteration in range(1, int(cfg.outer_iterations) + 1):
            target_model = None if iteration == 1 else model
            x_train, y_train, train_stats = bootstrap_and_targets(
                train_bundles, target_model, cfg, device
            )
            x_val, y_val, val_stats = bootstrap_and_targets(
                val_bundles, target_model, cfg, device
            )
            dataset = TensorDataset(
                torch.from_numpy(x_train), torch.from_numpy(y_train)
            )
            generator = torch.Generator().manual_seed(seed + iteration)
            loader = DataLoader(
                dataset,
                batch_size=int(cfg.batch_size),
                shuffle=True,
                num_workers=0,
                generator=generator,
            )
            losses: List[float] = []
            started = time.perf_counter()
            model.train()
            for _ in range(int(cfg.inner_epochs)):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(xb)
                    loss = F.smooth_l1_loss(prediction, yb)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"nonfinite LHBL loss for {name}/{iteration}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
                    optimizer.step()
                    losses.append(float(loss.item()))
            model.eval()
            val_prediction = I.predict_model(model, x_val, device)
            val_error = val_prediction - y_val.astype(np.float64)
            histories.append(
                {
                    "model": name,
                    "iteration": int(iteration),
                    "inner_epochs": int(cfg.inner_epochs),
                    "train_loss_mean": float(np.mean(losses)),
                    "val_mae": float(np.mean(np.abs(val_error))),
                    "val_rmse": float(np.sqrt(np.mean(val_error * val_error))),
                    "seconds": float(time.perf_counter() - started),
                    **{f"train_{key}": value for key, value in train_stats.items()},
                    **{f"val_{key}": value for key, value in val_stats.items()},
                }
            )
            checkpoint = checkpoint_dir / f"{name}_iteration_{iteration:02d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": name,
                    "iteration": int(iteration),
                    "lhbl_config": asdict(cfg),
                    "model_config": asdict(model_cfg),
                    "target": "radius_bounded_local_paths_plus_frozen_frontier_value",
                    "shortest_path_target": False,
                },
                checkpoint,
            )
            checkpoint_paths.append(checkpoint)
            checkpoint_rows, checkpoint_diagnostics = evaluate_checkpoint(
                name,
                iteration,
                model,
                eval_bundles,
                baselines,
                cfg,
                device,
            )
            search_rows.extend(checkpoint_rows)
            prediction_rows.extend(checkpoint_diagnostics)
            print(
                f"lhbl {name} iteration={iteration}/{cfg.outer_iterations}: "
                f"target={train_stats['target_mean']:.3f} "
                f"val_mae={histories[-1]['val_mae']:.3f}",
                flush=True,
            )
    return histories, search_rows, prediction_rows, checkpoint_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-H limited-horizon Bellman learning")
    for field in LHBLConfig.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        default = field.default
        if isinstance(default, bool):
            parser.add_argument(name, action="store_true", default=default)
        else:
            parser.add_argument(name, type=type(default), default=default)
    return parser.parse_args()


def _resolve_paths(cfg: LHBLConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    if cfg.study_dir == LHBLConfig.study_dir:
        candidate = script_dir / cfg.study_dir
        if candidate.exists():
            cfg.study_dir = str(candidate)
            if cfg.out_dir == LHBLConfig.out_dir:
                cfg.out_dir = str(script_dir / cfg.out_dir)


def main() -> None:
    cfg = LHBLConfig(**vars(parse_args()))
    _resolve_paths(cfg)
    if int(cfg.outer_iterations) <= 0 or int(cfg.inner_epochs) <= 0:
        raise ValueError("training iteration counts must be positive")
    if int(cfg.train_worlds) <= 0 or int(cfg.validation_worlds) <= 0:
        raise ValueError("training and validation world counts must be positive")
    alphas = C13.parse_float_csv(cfg.alphas)
    focal_ws = C13.parse_float_csv(cfg.focal_ws)
    if not any(math.isclose(value, cfg.primary_w, abs_tol=1e-12) for value in focal_ws):
        raise ValueError("primary-w must appear in focal-ws")
    if not all(0.0 < value <= 1.0 for value in alphas):
        raise ValueError("alphas must lie in (0, 1]")

    study_cfg, source_manifest = S.load_study(cfg.study_dir)
    local_cfg = state_config(cfg)
    train_metadata = Path(cfg.study_dir) / "datasets" / "c13_td_train.metadata.json"
    val_metadata = Path(cfg.study_dir) / "datasets" / "c13_td_val.metadata.json"
    if int(cfg.train_worlds) == int(study_cfg.train_worlds):
        train_bundles = rebuild_metadata_bundles(train_metadata, local_cfg)
        train_cohort_source = "source_metadata_replay"
    else:
        train_bundles = rebuild_generated_bundles(
            study_cfg, local_cfg, "train", cfg.train_worlds, 0
        )
        train_cohort_source = "deterministic_source_recipe"
    if int(cfg.validation_worlds) == int(study_cfg.val_worlds):
        val_bundles = rebuild_metadata_bundles(val_metadata, local_cfg)
        validation_cohort_source = "source_metadata_replay"
    else:
        val_bundles = rebuild_generated_bundles(
            study_cfg, local_cfg, "validation", cfg.validation_worlds, 500_000
        )
        validation_cohort_source = "deterministic_source_recipe"
    eval_bundles = rebuild_eval_bundles(study_cfg, local_cfg)
    baseline_rows, baselines = build_baselines(eval_bundles, focal_ws, cfg.budget_factor)
    device = resolve_device(cfg.device)
    histories, search_rows, prediction_rows, checkpoints = train_models(
        cfg,
        study_cfg,
        train_bundles,
        val_bundles,
        eval_bundles,
        baselines,
        device,
    )
    summaries = summarize_search(search_rows, cfg.required_win_fraction)
    verdict = build_verdict(summaries, cfg.primary_w)

    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    cohort_path = C13.write_json(
        result_dir / "lhbl_cohorts.json",
        {
            "experiment": "C13-H",
            "train_source": train_cohort_source,
            "validation_source": validation_cohort_source,
            "source_study_seed": int(study_cfg.seed),
            "train_seed_offset": 0,
            "validation_seed_offset": 500_000,
            "world_generator": "continuous_prm_c13_state_heuristic.iter_worlds",
            "roadmap_seed_rule": "world_seed_plus_17",
            "train": [
                {
                    "split": bundle.split,
                    "suite": bundle.suite,
                    "world_index": int(bundle.world_index),
                    "world_seed": int(bundle.world_seed),
                    "roadmap_seed": int(bundle.world_seed) + 17,
                    "nodes": int(len(bundle.roadmap.points)),
                    "edges": int(sum(len(row) for row in bundle.roadmap.adj) // 2),
                }
                for bundle in train_bundles
            ],
            "validation": [
                {
                    "split": bundle.split,
                    "suite": bundle.suite,
                    "world_index": int(bundle.world_index),
                    "world_seed": int(bundle.world_seed),
                    "roadmap_seed": int(bundle.world_seed) + 17,
                    "nodes": int(len(bundle.roadmap.points)),
                    "edges": int(sum(len(row) for row in bundle.roadmap.adj) // 2),
                }
                for bundle in val_bundles
            ],
        },
    )
    history_path = C13.write_csv(result_dir / "lhbl_training_history.csv", histories)
    baseline_path = C13.write_csv(result_dir / "lhbl_baselines.csv", baseline_rows)
    raw_path = C13.write_csv(result_dir / "lhbl_search_raw.csv", search_rows)
    summary_path = C13.write_csv(result_dir / "lhbl_search_summary.csv", summaries)
    prediction_path = C13.write_csv(
        result_dir / "lhbl_prediction_diagnostics.csv", prediction_rows
    )
    verdict_path = C13.write_json(result_dir / "gate_verdict.json", verdict)

    models = parse_names(cfg.models)
    expected_rows = (
        len(models)
        * int(cfg.outer_iterations)
        * len(alphas)
        * len(focal_ws)
        * len(eval_bundles)
    )
    train_world_seeds = {b.world_seed for b in train_bundles}
    validation_world_seeds = {b.world_seed for b in val_bundles}
    cohort_seed_failures = (
        int(len(train_world_seeds) != len(train_bundles))
        + int(len(validation_world_seeds) != len(val_bundles))
        + int(bool(train_world_seeds & validation_world_seeds))
    )
    verification = {
        "models": models,
        "device": str(device),
        "train_worlds": int(len(train_bundles)),
        "validation_worlds": int(len(val_bundles)),
        "train_cohort_source": train_cohort_source,
        "validation_cohort_source": validation_cohort_source,
        "unique_train_world_seeds": int(len(train_world_seeds)),
        "unique_validation_world_seeds": int(len(validation_world_seeds)),
        "train_validation_seed_overlap": int(len(train_world_seeds & validation_world_seeds)),
        "cohort_seed_failures": int(cohort_seed_failures),
        "development_eval_worlds": int(len(eval_bundles)),
        "search_rows": int(len(search_rows)),
        "expected_search_rows": int(expected_rows),
        "history_rows": int(len(histories)),
        "expected_history_rows": int(len(models) * int(cfg.outer_iterations)),
        "prediction_rows": int(len(prediction_rows)),
        "expected_prediction_rows": int(
            len(models) * int(cfg.outer_iterations) * len(eval_bundles)
        ),
        "checkpoint_count": int(len(checkpoints)),
        "certification_failures": int(
            np.sum([not bool(row["certified"]) for row in search_rows])
        ),
        "path_failures": int(np.sum([not bool(row["path_valid"]) for row in search_rows])),
        "bound_violations_eval_only": int(
            np.sum([bool(row["bound_violation_eval_only"]) for row in search_rows])
        ),
        "states_expanded_more_than_twice": int(
            np.sum([int(row["max_expansions_per_state"]) > 2 for row in search_rows])
        ),
        "nonfinite_training_metrics": int(
            np.sum(
                [
                    not math.isfinite(float(row["train_loss_mean"]))
                    or not math.isfinite(float(row["val_mae"]))
                    for row in histories
                ]
            )
        ),
        "shortest_path_target": False,
        "training_target_reads_dist_to_goal": False,
        "shortest_path_use": "development_evaluation_and_roadmap_connectivity_only",
        "runtime_information": "current_goal_geometry_bounded_rays_one_hop_actions",
        "training_information": "bounded_local_subgraph_edges_and_frozen_successor_values",
        "development_cohort_reused": True,
        "fresh_replication_required": verdict["fresh_replication_required"],
    }
    for actual, expected, label in (
        (verification["search_rows"], verification["expected_search_rows"], "search"),
        (verification["history_rows"], verification["expected_history_rows"], "history"),
        (
            verification["prediction_rows"],
            verification["expected_prediction_rows"],
            "prediction",
        ),
    ):
        if actual != expected:
            raise RuntimeError(f"LHBL {label} row count mismatch")
    verification_path = C13.write_json(result_dir / "verification.json", verification)
    safety_failures = (
        verification["certification_failures"]
        + verification["path_failures"]
        + verification["bound_violations_eval_only"]
        + verification["states_expanded_more_than_twice"]
        + verification["nonfinite_training_metrics"]
        + verification["cohort_seed_failures"]
    )
    if safety_failures:
        raise RuntimeError("LHBL safety or numerical verification failed")

    source_paths = {
        "implementation": Path(__file__).resolve(),
        "shared_queue": Path(Q.__file__).resolve(),
        "source_study_manifest": Path(cfg.study_dir) / "manifest.json",
    }
    if train_cohort_source == "source_metadata_replay":
        source_paths["train_metadata"] = train_metadata
    if validation_cohort_source == "source_metadata_replay":
        source_paths["validation_metadata"] = val_metadata
    output_paths = {
        "history": history_path,
        "baselines": baseline_path,
        "raw": raw_path,
        "summary": summary_path,
        "predictions": prediction_path,
        "gate": verdict_path,
        "verification": verification_path,
        "cohorts": cohort_path,
    }
    integrity = {
        "inputs": {
            name: {"path": str(path), "sha256": S.file_sha256(path)}
            for name, path in source_paths.items()
        },
        "outputs": {
            name: {"path": str(path), "sha256": S.file_sha256(path)}
            for name, path in output_paths.items()
        },
        "checkpoints": {
            path.name: {"path": str(path), "sha256": S.file_sha256(path)}
            for path in checkpoints
        },
    }
    integrity_path = C13.write_json(Path(cfg.out_dir) / "integrity.json", integrity)
    manifest = {
        "experiment": "C13-H current-state limited-horizon Bellman learning",
        "runner_config": asdict(cfg),
        "source_study_config": asdict(study_cfg),
        "source_study_experiment": source_manifest.get("experiment"),
        "train_cohort_source": train_cohort_source,
        "validation_cohort_source": validation_cohort_source,
        "cohort_seed_source": "source_study_config",
        "target": "bounded_local_paths_plus_frozen_frontier_state_value",
        "target_network_update": "frozen_for_each_outer_iteration",
        "shortest_path_target": False,
        "training_target_reads_dist_to_goal": False,
        "runtime_scope": "current_goal_geometry_bounded_rays_one_hop_actions",
        "full_map_runtime_input": False,
        "development_cohort_reused": True,
        "fresh_replication_required": verdict["fresh_replication_required"],
        "literature_reference": "https://ojs.aaai.org/index.php/AAAI/article/view/41023",
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "integrity": str(integrity_path),
    }
    manifest_path = C13.write_json(Path(cfg.out_dir) / "manifest.json", manifest)
    print(f"verdict={verdict['verdict']}")
    print(f"authorization={verdict['authorization']}")
    print(f"selected={verdict['selected_candidate']}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
