"""Heuristic providers for the C7 integration comparison.

A provider maps (world, roadmap, goal_idx) -> a finite, non-negative per-node
heuristic array h[N]. The planner consumes h directly (astar) or as the focal
ranker alongside the admissible Euclid array (focal_astar).
"""
from __future__ import annotations

import abc
import argparse

import numpy as np
import torch

import continuous_prm_common as C
import continuous_prm_c5_hard_obstacle_encoder as C5
import continuous_prm_focal as focal


def _finite_fill(vals: np.ndarray, fallback: float) -> np.ndarray:
    out = np.array(vals, dtype=np.float64)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = fallback
    return np.maximum(out, 0.0)


def euclid_to_goal(roadmap: "C.Roadmap", goal_idx: int = 1) -> np.ndarray:
    """Per-node straight-line distance to the goal node (finite, >= 0)."""
    goal = roadmap.points[goal_idx]
    d = np.linalg.norm(roadmap.points - goal[None, :], axis=1)
    return _finite_fill(d, fallback=0.0)


class HeuristicProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def node_h(self, world: "C.World", roadmap: "C.Roadmap", goal_idx: int = 1) -> np.ndarray:
        ...


class EuclidProvider(HeuristicProvider):
    name = "euclid"

    def node_h(self, world, roadmap, goal_idx: int = 1) -> np.ndarray:
        return euclid_to_goal(roadmap, goal_idx)


class OracleProvider(HeuristicProvider):
    """Exact graph cost-to-go (the minimal-expansion ceiling for A* on this graph)."""
    name = "oracle"

    def node_h(self, world, roadmap, goal_idx: int = 1) -> np.ndarray:
        # NOTE: common.py uses INF = 1e30 (a large FINITE sentinel) for
        # unreachable Dijkstra distances, so np.isfinite is NOT a valid
        # connectivity test. Follow the codebase convention (dist < INF/10).
        dij = np.array(C.dijkstra_to_goal(roadmap.adj, goal_idx=goal_idx), dtype=np.float64)
        connected = dij < C.INF / 10.0
        fill = float(dij[connected].max() + world.side_len) if connected.any() else 10.0 * world.side_len
        dij[~connected] = fill
        return _finite_fill(dij, fallback=fill)


class ScalarResidualProvider(HeuristicProvider):
    """Additive per-node scalar residual: ``h = euclid + side_len * clip(yhat, 0, B)``.

    ``yhat`` is the normalized per-node residual predicted by a C5 sequence model
    (``continuous_prm_common.ContinuousHeuristicModel``). This is the C5-style
    additive integration: features come from the C5 hard obstacle encoder, the
    model predicts a (already softplus+clamped, but re-clipped here for safety)
    nonnegative residual in [0, B], and the heuristic adds ``side_len * yhat`` to
    the admissible Euclidean distance. The result is finite and >= euclid.
    """

    def __init__(self, model, feature_cfg, device, backbone: str, max_norm_residual: float):
        self.model = model
        self.feature_cfg = feature_cfg
        self.device = device
        self.max_norm_residual = float(max_norm_residual)
        self.name = f"scalar_{backbone}"

    def node_h(self, world, roadmap, goal_idx: int = 1) -> np.ndarray:
        feats = C5.make_hard_features_for_roadmap(world, roadmap, self.feature_cfg)
        yhat = C.predict_norm_residuals(self.model, feats, self.device)
        yhat = np.clip(np.asarray(yhat, dtype=np.float64), 0.0, self.max_norm_residual)
        euclid = euclid_to_goal(roadmap, goal_idx)
        h = euclid + world.side_len * yhat
        if not np.all(np.isfinite(h)):
            raise FloatingPointError("ScalarResidualProvider produced non-finite h")
        return np.maximum(h, 0.0)

    @classmethod
    def untrained_for_test(cls, world, device=None):
        """Tiny random (untrained) model on CPU for unit tests (no checkpoint).

        The model's input dim is ``feature_cfg.token_dim`` (the per-token width),
        which is exactly what ``C5.make_hard_features_for_roadmap`` emits. Using the
        default ``FeatureConfig`` (token_dim=16) keeps the model input aligned with
        the installed C5-hard encoder, matching how C5 builds model + feature config
        together (``build_model(cfg, feature_cfg, ...)`` wires ``token_dim``).
        """
        device = device or torch.device("cpu")
        feature_cfg = C.FeatureConfig()
        # Minimal argparse.Namespace with exactly the fields build_backbone_configs
        # reads (it constructs both the "hrm" and "onlstm" configs); we only use the
        # "hrm" entry. Kept small so the test stays fast on CPU.
        ns = argparse.Namespace(
            hrm_hidden=32,
            hrm_layers=1,
            hrm_k_step=2,
            hrm_heads=4,
            head_hidden=48,
            onlstm_hidden=32,
            onlstm_layers=1,
            onlstm_chunk_size=8,
        )
        bb = C.build_backbone_configs(ns)["hrm"]
        train_cfg = C.TrainingConfig()
        model = C.build_model(bb, feature_cfg, train_cfg, device).eval()
        return cls(model, feature_cfg, device, backbone="hrm", max_norm_residual=train_cfg.max_norm_residual)


class ValueFieldProvider(HeuristicProvider):
    """C6 value-field integration: h = euclid + side_len * interp(field residual).

    Uses the same ``field_node_heuristic`` shared helper that ``evaluate_shard``
    calls, guaranteeing numerical parity with the C6 offline benchmark.
    Covers learned backbones (unet / onlstm / hrm); oracle/graph-dijkstra ceiling
    is handled separately by ``OracleProvider``.
    """

    def __init__(self, model, grid_size: int, device, backbone: str):
        self.model = model
        self.grid_size = int(grid_size)
        self.device = device
        self.name = f"field_{backbone}"

    def node_h(self, world, roadmap, goal_idx: int = 1) -> np.ndarray:
        import continuous_prm_c6_heatmap_value_field as C6
        euclid = euclid_to_goal(roadmap, goal_idx)
        x = C6.make_heatmap_example(world, self.grid_size)["x"]
        h, _ = C6.field_node_heuristic(self.model, x, world, roadmap, euclid, self.device)
        if not np.all(np.isfinite(h)):
            raise FloatingPointError("ValueFieldProvider produced non-finite h")
        return np.maximum(h, 0.0)


def run_world_arms(world, roadmap, providers: dict, budgets, w_values, goal_idx: int = 1, start_idx: int = 0):
    """Run every (provider, mode, budget[, w]) arm on one shared world+roadmap.

    - astar mode for all providers (additive/direct integration).
    - focal mode for all providers, for each w in w_values (Euclid + collapsed
      signals degrade to A* automatically).
    Returns a flat list of record dicts with matched suboptimality vs the graph optimal.
    """
    opt = float(roadmap.dist_to_goal[start_idx])  # graph optimal cost from start
    if "euclid" in providers:
        euclid_h = providers["euclid"].node_h(world, roadmap, goal_idx)
    else:
        euclid_h = euclid_to_goal(roadmap, goal_idx)
    node_h = {name: prov.node_h(world, roadmap, goal_idx) for name, prov in providers.items()}
    records = []
    for name in providers:
        h = node_h[name]
        for b in budgets:
            r = C.astar_search(roadmap.adj, h, budget=int(b), start_idx=start_idx, goal_idx=goal_idx)
            records.append(_arm_record(name, "astar", None, b, r, opt))
            for w in w_values:
                rf = focal.focal_astar_search(roadmap.adj, euclid_h=euclid_h, rank_h=h,
                                              budget=int(b), w=float(w), start_idx=start_idx, goal_idx=goal_idx)
                records.append(_arm_record(name, "focal", float(w), b, rf, opt))
    return records


def _arm_record(provider, mode, w, budget, res, opt):
    found = bool(res["found"])
    cost = float(res["cost"]) if found else float("nan")
    sub = (cost / opt) if (found and opt > 0) else float("nan")
    return {
        "provider": provider, "mode": mode, "w": w, "budget": int(budget),
        "found": found, "expansions": int(res["expansions"]), "cost": cost,
        "optimal": opt, "suboptimality": sub,
    }
