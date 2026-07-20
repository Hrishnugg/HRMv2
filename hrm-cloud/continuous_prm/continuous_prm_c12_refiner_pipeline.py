"""Training, evaluation, inference, and artifact pipeline for C12-B.

The model definitions and G0-B probe live in ``continuous_prm_c12_refiner``.
This companion module keeps the execution layer reviewable without changing
the frozen C11 implementation.  It is imported lazily by the C12-B CLI.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

import continuous_prm_common as C
import continuous_prm_c11_headroom as C11P
import continuous_prm_c11_mission as C11
import continuous_prm_c12_refiner as R


PILOT_EVAL_BASE = 2_450_000_001
SUCCESS_NONINFERIORITY_MARGIN = -0.05
BOOTSTRAP_DRAWS = 10_000
PERMUTATION_DRAWS = 20_000


@dataclass
class GraphSample:
    bundle: C11.WorldBundle
    node_feats: np.ndarray
    edge_index: np.ndarray
    edge_feats: np.ndarray
    h_legsum_norm: np.ndarray
    target_flat_ids: np.ndarray
    target_y: np.ndarray
    final_transition_hops: float

    @property
    def n_nodes(self) -> int:
        return int(self.node_feats.shape[0])


@dataclass
class GraphBatch:
    node_feats: torch.Tensor
    edge_index: torch.Tensor
    edge_feats: torch.Tensor
    h_legsum_norm: torch.Tensor
    target_flat_ids: torch.Tensor
    target_y: torch.Tensor


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    R._write_json_atomic(Path(path), payload)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    R._write_csv_atomic(Path(path), rows)


def _torch_save_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_seed(seed: int, cell: Mapping[str, object]) -> int:
    return 120_012 + 10_007 * (int(seed) + 1) + 101 * int(cell["config_idx"]) + int(cell["K"])


def _pilot_eval_seed(w: int, config_idx: int, K: int) -> int:
    return R._seed_from_base(PILOT_EVAL_BASE, w, config_idx, K)


def _collect_pilot_eval(
    cell: Mapping[str, object], n_worlds: int, cfg: R.C12RefinerConfig
) -> Tuple[List[C11.WorldBundle], List[dict]]:
    bundles: List[C11.WorldBundle] = []
    ledger: List[dict] = []
    attempt = 0
    while len(bundles) < n_worlds:
        if attempt >= cfg.max_world_attempts:
            raise RuntimeError(
                f"pilot-eval collection found {len(bundles)}/{n_worlds} valid worlds "
                f"for {dict(cell)} after {attempt} attempts"
            )
        seed = _pilot_eval_seed(attempt, int(cell["config_idx"]), int(cell["K"]))
        row = {
            "split": "pilot_eval",
            "attempt_idx": attempt,
            "seed": seed,
            "config": cell["config_label"],
            "K": int(cell["K"]),
            "accepted": False,
            "reason": "",
        }
        attempt += 1
        t0 = time.perf_counter()
        try:
            bundle = C11.collect_world_bundle(dict(cell), seed, cfg)
        except RuntimeError as exc:
            row["collect_wall_s"] = time.perf_counter() - t0
            row["reason"] = str(exc)
            ledger.append(row)
            continue
        row["collect_wall_s"] = time.perf_counter() - t0
        row["accepted"] = True
        row["valid_world_idx"] = len(bundles)
        ledger.append(row)
        bundles.append(bundle)
    return bundles, ledger


def prepare_sample(bundle: C11.WorldBundle, cfg: R.C12RefinerConfig) -> GraphSample:
    graph = C11.encode_product_graph(bundle, cfg)
    N = int(bundle.rm.points.shape[0])
    K = len(bundle.wp)
    hl = np.asarray(
        [[bundle.hl(i, s) for i in range(N)] for s in range(K + 1)], dtype=np.float32
    ).reshape(-1)
    hl /= float(bundle.world.side_len)
    target_i = bundle.targets[:, 0].astype(np.int64)
    target_s = bundle.targets[:, 1].astype(np.int64)
    target_ids = target_s * N + target_i
    return GraphSample(
        bundle=bundle,
        node_feats=graph["node_feats"].astype(np.float32, copy=False),
        edge_index=graph["edge_index"].astype(np.int64, copy=False),
        edge_feats=graph["edge_feats"].astype(np.float32, copy=False),
        h_legsum_norm=hl,
        target_flat_ids=target_ids,
        target_y=bundle.targets[:, 2].astype(np.float32),
        final_transition_hops=R.final_transition_distance(graph["edge_index"], N, K),
    )


def prepare_samples(
    bundles: Sequence[C11.WorldBundle], cfg: R.C12RefinerConfig
) -> List[GraphSample]:
    return [prepare_sample(bundle, cfg) for bundle in bundles]


def batch_samples(samples: Sequence[GraphSample], device: torch.device) -> GraphBatch:
    if not samples:
        raise ValueError("cannot batch zero graph samples")
    node_parts: List[np.ndarray] = []
    edge_parts: List[np.ndarray] = []
    edge_feat_parts: List[np.ndarray] = []
    hl_parts: List[np.ndarray] = []
    target_id_parts: List[np.ndarray] = []
    target_y_parts: List[np.ndarray] = []
    offset = 0
    for sample in samples:
        node_parts.append(sample.node_feats)
        edge_parts.append(sample.edge_index + offset)
        edge_feat_parts.append(sample.edge_feats)
        hl_parts.append(sample.h_legsum_norm)
        target_id_parts.append(sample.target_flat_ids + offset)
        target_y_parts.append(sample.target_y)
        offset += sample.n_nodes
    return GraphBatch(
        node_feats=torch.from_numpy(np.concatenate(node_parts, axis=0)).to(device),
        edge_index=torch.from_numpy(np.concatenate(edge_parts, axis=1)).to(device),
        edge_feats=torch.from_numpy(np.concatenate(edge_feat_parts, axis=0)).to(device),
        h_legsum_norm=torch.from_numpy(np.concatenate(hl_parts, axis=0)).to(device),
        target_flat_ids=torch.from_numpy(np.concatenate(target_id_parts)).to(device),
        target_y=torch.from_numpy(np.concatenate(target_y_parts)).to(device),
    )


def pack_sample_indices(
    samples: Sequence[GraphSample], order: Sequence[int], node_budget: int
) -> List[List[int]]:
    packs: List[List[int]] = []
    current: List[int] = []
    current_nodes = 0
    for idx in order:
        n = samples[int(idx)].n_nodes
        if current and current_nodes + n > node_budget:
            packs.append(current)
            current = []
            current_nodes = 0
        current.append(int(idx))
        current_nodes += n
    if current:
        packs.append(current)
    return packs


def supervised_loss(
    outputs: Mapping[int, torch.Tensor], batch: GraphBatch, cfg: R.C12RefinerConfig
) -> torch.Tensor:
    available = sorted(outputs)
    if set(available).issubset(cfg.deep_supervision_weights):
        weights = {cycle: cfg.deep_supervision_weights[cycle] for cycle in available}
        norm = sum(weights.values())
        weights = {cycle: value / norm for cycle, value in weights.items()}
    else:
        weights = {cycle: 1.0 / len(available) for cycle in available}
    losses = []
    for cycle in available:
        preds = outputs[cycle][batch.target_flat_ids]
        losses.append(
            weights[cycle]
            * F.smooth_l1_loss(preds, batch.target_y, beta=cfg.smooth_l1_beta)
        )
    return sum(losses)


def _evaluate_loss(
    model: nn.Module,
    samples: Sequence[GraphSample],
    cfg: R.C12RefinerConfig,
    device: torch.device,
) -> float:
    model.eval()
    losses: List[float] = []
    order = list(range(len(samples)))
    with torch.no_grad():
        for pack in pack_sample_indices(samples, order, cfg.graph_accum_nodes):
            batch = batch_samples([samples[i] for i in pack], device)
            outputs = model(
                batch.node_feats, batch.edge_index, batch.edge_feats, batch.h_legsum_norm
            )
            losses.append(float(supervised_loss(outputs, batch, cfg).item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_arm(
    arm: str,
    cell: Mapping[str, object],
    model_seed: int,
    train_samples: Sequence[GraphSample],
    val_samples: Sequence[GraphSample],
    cfg: R.C12RefinerConfig,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
    *,
    force: bool = False,
) -> dict:
    if checkpoint_path.exists() and not force:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return dict(payload["meta"], resumed=True)

    seed_value = _model_seed(model_seed, cell)
    random.seed(seed_value)
    np.random.seed(seed_value % (2**32))
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    rng = np.random.default_rng(seed_value)

    model = R.build_arm(arm, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    best_val = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[dict] = []
    started = time.perf_counter()

    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(len(train_samples)).tolist()
        packs = pack_sample_indices(train_samples, order, cfg.graph_accum_nodes)
        epoch_losses: List[float] = []
        epoch_t0 = time.perf_counter()
        for pack in packs:
            batch = batch_samples([train_samples[i] for i in pack], device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch.node_feats, batch.edge_index, batch.edge_feats, batch.h_legsum_norm
            )
            loss = supervised_loss(outputs, batch, cfg)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss: arm={arm}, cell={dict(cell)}, seed={model_seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        val_loss = _evaluate_loss(model, val_samples, cfg, device)
        train_loss = float(np.mean(epoch_losses))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "wall_s": time.perf_counter() - epoch_t0,
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(
                f"[C12-B train] {arm} {cell['config_label']}/K{cell['K']} seed={model_seed} "
                f"epoch={epoch + 1}/{epochs} train={train_loss:.5f} val={val_loss:.5f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("training completed without a validation checkpoint")
    wall_s = time.perf_counter() - started
    meta = {
        "schema_version": R.SCHEMA_VERSION,
        "arm": arm,
        "config": cell["config_label"],
        "K": int(cell["K"]),
        "model_seed": int(model_seed),
        "seed_value": seed_value,
        "epochs": int(epochs),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "first_train_loss": history[0]["train_loss"],
        "final_train_loss": history[-1]["train_loss"],
        "parameter_count": R.parameter_count(model),
        "edge_applications": int(model.edge_applications),
        "train_wall_s": wall_s,
        "device": str(device),
        "resumed": False,
    }
    _torch_save_atomic(
        checkpoint_path,
        {
            "schema_version": R.SCHEMA_VERSION,
            "state_dict": best_state,
            "meta": meta,
            "history": history,
            "config": R._json_safe(asdict(cfg)),
        },
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return meta


def load_model(
    arm: str, checkpoint_path: Path, cfg: R.C12RefinerConfig, device: torch.device
) -> Tuple[nn.Module, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = R.build_arm(arm, cfg)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, payload["meta"]


def predict_sample(
    model: nn.Module, sample: GraphSample, device: torch.device
) -> Dict[int, np.ndarray]:
    batch = batch_samples([sample], device)
    with torch.no_grad():
        outputs = model(
            batch.node_feats, batch.edge_index, batch.edge_feats, batch.h_legsum_norm
        )
    return {
        int(cycle): values.detach().cpu().numpy().astype(np.float64)
        for cycle, values in outputs.items()
    }


def _bellman_residual(
    h_flat: np.ndarray, sample: GraphSample, finite_mask: np.ndarray
) -> float:
    src = sample.edge_index[0]
    dst = sample.edge_index[1]
    usable = finite_mask[src] & finite_mask[dst]
    if not np.any(usable):
        return float("nan")
    src_u = src[usable]
    dst_u = dst[usable]
    side = float(sample.bundle.world.side_len)
    edge_cost = sample.edge_feats[usable, 0].astype(np.float64) * side
    candidates = edge_cost + h_flat[dst_u]
    best = np.full(h_flat.shape[0], np.inf, dtype=np.float64)
    np.minimum.at(best, src_u, candidates)
    valid = finite_mask & np.isfinite(best)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(h_flat[valid] - best[valid])) / side)


def state_metrics(
    h_flat: np.ndarray, sample: GraphSample
) -> Tuple[float, float, float]:
    oracle = sample.bundle.oracle.reshape(-1).astype(np.float64)
    finite = oracle < C.INF / 10.0
    side = float(sample.bundle.world.side_len)
    mae = float(np.mean(np.abs(h_flat[finite] - oracle[finite])) / side)
    rho = float(spearmanr(h_flat[finite], oracle[finite]).statistic)
    if not math.isfinite(rho):
        rho = 0.0
    bellman = _bellman_residual(h_flat, sample, finite)
    return mae, rho, bellman


def _reference_fields(sample: GraphSample) -> Dict[str, np.ndarray]:
    bundle = sample.bundle
    K = len(bundle.wp)
    N = int(bundle.rm.points.shape[0])
    hl = np.asarray([[bundle.hl(i, s) for i in range(N)] for s in range(K + 1)])
    return {"h_legsum": hl.reshape(-1), "h_oracle": bundle.oracle.reshape(-1)}


def _evaluate_field(
    *,
    field_flat: np.ndarray,
    sample: GraphSample,
    arm: str,
    cycle: int,
    model_seed: int,
    config: str,
    K: int,
    world_idx: int,
    binding_budget: int,
) -> Tuple[dict, dict]:
    bundle = sample.bundle
    N = int(bundle.rm.points.shape[0])
    field = field_flat.reshape(K + 1, N)
    mae, rho, bellman = state_metrics(field_flat, sample)
    result = C11P.astar_product(
        bundle.rm.adj, bundle.wp, field, binding_budget, bundle.adj_valid
    )
    common = {
        "config": config,
        "K": K,
        "world_idx": world_idx,
        "world_seed": int(bundle.seed),
        "model_seed": model_seed,
        "arm": arm,
        "cycle": cycle,
        "binding_budget": binding_budget,
        "final_transition_hops": sample.final_transition_hops,
    }
    state_row = dict(common)
    state_row.update(
        {"state_mae": mae, "rank_corr": rho, "bellman_residual": bellman}
    )
    eval_row = dict(common)
    eval_row.update(
        {
            "found": bool(result["found"]),
            "cost": float(result["cost"]) if result["found"] else "",
            "optimal_cost": float(bundle.oracle[0, 0]),
            "cost_ratio": (
                float(result["cost"]) / float(bundle.oracle[0, 0])
                if result["found"] else ""
            ),
            "expansions": int(result["expansions"]),
            "closed": int(result["closed"]),
            "expansion_burden": float(result["expansions"]) / float(binding_budget),
            "completion": 1.0 if result["found"] else 0.0,
        }
    )
    return state_row, eval_row


def evaluate_cell_seed(
    *,
    cell: Mapping[str, object],
    model_seed: int,
    eval_samples: Sequence[GraphSample],
    cfg: R.C12RefinerConfig,
    checkpoint_dir: Path,
    device: torch.device,
    include_references: bool,
) -> Tuple[List[dict], List[dict]]:
    state_rows: List[dict] = []
    eval_rows: List[dict] = []
    config = str(cell["config_label"])
    K = int(cell["K"])
    binding_budget = int(cfg.binding_budgets[(config, K)])

    if include_references:
        for world_idx, sample in enumerate(eval_samples):
            for reference, field in _reference_fields(sample).items():
                sr, er = _evaluate_field(
                    field_flat=field,
                    sample=sample,
                    arm=reference,
                    cycle=0,
                    model_seed=-1,
                    config=config,
                    K=K,
                    world_idx=world_idx,
                    binding_budget=binding_budget,
                )
                state_rows.append(sr)
                eval_rows.append(er)

    for arm in R.ARM_NAMES:
        ckpt = checkpoint_dir / f"{arm}_config{config}_K{K}_seed{model_seed}.pt"
        model, _meta = load_model(arm, ckpt, cfg, device)
        for world_idx, sample in enumerate(eval_samples):
            outputs = predict_sample(model, sample, device)
            side = float(sample.bundle.world.side_len)
            hl = sample.h_legsum_norm.astype(np.float64) * side
            for cycle, residual in outputs.items():
                field = hl + side * residual
                sr, er = _evaluate_field(
                    field_flat=field,
                    sample=sample,
                    arm=arm,
                    cycle=cycle,
                    model_seed=model_seed,
                    config=config,
                    K=K,
                    world_idx=world_idx,
                    binding_budget=binding_budget,
                )
                state_rows.append(sr)
                eval_rows.append(er)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return state_rows, eval_rows


def _bootstrap_ci(values: np.ndarray, *, seed: int, draws: int = BOOTSTRAP_DRAWS) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(draws, values.size))
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def _bootstrap_difference_independent(
    deep: np.ndarray, shallow: np.ndarray, *, seed: int
) -> Tuple[float, float]:
    deep = np.asarray(deep, dtype=np.float64)
    shallow = np.asarray(shallow, dtype=np.float64)
    rng = np.random.default_rng(seed)
    di = rng.integers(0, deep.size, size=(BOOTSTRAP_DRAWS, deep.size))
    si = rng.integers(0, shallow.size, size=(BOOTSTRAP_DRAWS, shallow.size))
    diffs = deep[di].mean(axis=1) - shallow[si].mean(axis=1)
    return (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)))


def _sign_flip_p(values: np.ndarray, *, seed: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(PERMUTATION_DRAWS, values.size))
    null = np.abs((signs * values[None, :]).mean(axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (PERMUTATION_DRAWS + 1))


def _bh_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=np.float64)
    running = 1.0
    for rank_idx in range(n - 1, -1, -1):
        original_idx = order[rank_idx]
        rank = rank_idx + 1
        running = min(running, float(p[original_idx]) * n / rank)
        adjusted[original_idx] = min(1.0, running)
    return adjusted.tolist()


def analyze_results(
    state_rows: Sequence[dict], eval_rows: Sequence[dict], *, scale: str
) -> Tuple[dict, List[dict]]:
    state = pd.DataFrame(state_rows)
    raw = pd.DataFrame(eval_rows)
    learned = raw[raw["model_seed"] >= 0].copy()
    averaged = (
        learned.groupby(["config", "K", "world_idx", "world_seed", "arm", "cycle"], as_index=False)
        .agg(
            expansion_burden=("expansion_burden", "mean"),
            completion=("completion", "mean"),
            expansions=("expansions", "mean"),
            final_transition_hops=("final_transition_hops", "first"),
        )
    )
    significance: List[dict] = []
    g1_cells: List[dict] = []
    improvements: Dict[Tuple[str, int], np.ndarray] = {}

    for config in sorted(averaged["config"].unique().tolist()):
        for K in sorted(averaged["K"].unique().tolist()):
            tied = averaged[
                (averaged["config"] == config)
                & (averaged["K"] == K)
                & (averaged["arm"] == "tied_refiner")
            ]
            pivot = tied.pivot(index="world_idx", columns="cycle", values="expansion_burden")
            if not all(cycle in pivot for cycle in (1, 2, 4, 8)):
                continue
            pivot = pivot.dropna()
            improvement = (pivot[1] - pivot[8]).to_numpy(dtype=np.float64)
            improvements[(config, int(K))] = improvement
            imp_ci = _bootstrap_ci(improvement, seed=1200 + int(K) + ord(config[0]))
            adjacent = []
            monotonic = True
            for prev, later in ((1, 2), (2, 4), (4, 8)):
                later_minus_prev = (pivot[later] - pivot[prev]).to_numpy(dtype=np.float64)
                ci = _bootstrap_ci(later_minus_prev, seed=1300 + prev * 10 + later + int(K))
                # A statistically resolved worsening has CI_low > 0.  Ties
                # and unresolved small changes are permitted by G1-B.
                no_resolved_worsening = ci[0] <= 0.0
                monotonic = monotonic and no_resolved_worsening
                adjacent.append(
                    {
                        "from": prev,
                        "to": later,
                        "mean_later_minus_earlier": float(later_minus_prev.mean()),
                        "ci95": list(ci),
                        "no_resolved_worsening": no_resolved_worsening,
                    }
                )
            g1_cells.append(
                {
                    "config": config,
                    "K": int(K),
                    "n_worlds": int(len(pivot)),
                    "cycle_means": {str(c): float(pivot[c].mean()) for c in (1, 2, 4, 8)},
                    "monotonic_ties_allowed": monotonic,
                    "cycle1_minus_cycle8": float(improvement.mean()),
                    "cycle1_minus_cycle8_ci95": list(imp_ci),
                    "cycle1_vs_cycle8_separated": imp_ci[0] > 0.0,
                    "adjacent": adjacent,
                }
            )

    g1_by_config: List[dict] = []
    for config in sorted(averaged["config"].unique().tolist()):
        deep = next((row for row in g1_cells if row["config"] == config and row["K"] == 8), None)
        imp8 = improvements.get((config, 8))
        imp2 = improvements.get((config, 2))
        if deep is None or imp8 is None or imp2 is None:
            g1_by_config.append({"config": config, "passed": False, "reason": "missing K2/K8 cycle rows"})
            continue
        dose_ci = _bootstrap_difference_independent(
            imp8, imp2, seed=1400 + ord(config[0])
        )
        passed = bool(
            deep["monotonic_ties_allowed"]
            and deep["cycle1_vs_cycle8_separated"]
            and dose_ci[0] > 0.0
        )
        g1_by_config.append(
            {
                "config": config,
                "passed": passed,
                "deep_K": 8,
                "mean_deep_minus_K2_improvement": float(imp8.mean() - imp2.mean()),
                "deep_minus_K2_ci95": list(dose_ci),
            }
        )
    g1 = any(row["passed"] for row in g1_by_config)

    # G2-B primary family: tied cycle 8 versus shallow cycle 1 and untied
    # cycle 8, in each deep (K=8) config.  Seeds are averaged within world;
    # worlds are the independent units for CIs and sign-flip tests.
    comparisons: List[dict] = []
    for config in sorted(averaged["config"].unique().tolist()):
        cell = averaged[(averaged["config"] == config) & (averaged["K"] == 8)]
        tied = cell[(cell["arm"] == "tied_refiner") & (cell["cycle"] == 8)].set_index("world_idx")
        for control, control_cycle in (("shallow_param_match", 1), ("untied_compute_match", 8)):
            comp = cell[(cell["arm"] == control) & (cell["cycle"] == control_cycle)].set_index("world_idx")
            common = sorted(set(tied.index) & set(comp.index))
            burden_gain = (
                comp.loc[common, "expansion_burden"].to_numpy(dtype=np.float64)
                - tied.loc[common, "expansion_burden"].to_numpy(dtype=np.float64)
            )
            completion_diff = (
                tied.loc[common, "completion"].to_numpy(dtype=np.float64)
                - comp.loc[common, "completion"].to_numpy(dtype=np.float64)
            )
            gain_ci = _bootstrap_ci(burden_gain, seed=1500 + len(comparisons))
            completion_ci = _bootstrap_ci(completion_diff, seed=1600 + len(comparisons))
            comparisons.append(
                {
                    "family": "G2-B",
                    "config": config,
                    "K": 8,
                    "treatment": "tied_refiner_cycle8",
                    "control": f"{control}_cycle{control_cycle}",
                    "n_worlds": len(common),
                    "mean_control_minus_tied_burden": float(burden_gain.mean()),
                    "burden_gain_ci95_low": gain_ci[0],
                    "burden_gain_ci95_high": gain_ci[1],
                    "p_raw": _sign_flip_p(burden_gain, seed=1700 + len(comparisons)),
                    "mean_tied_minus_control_completion": float(completion_diff.mean()),
                    "completion_diff_ci95_low": completion_ci[0],
                    "completion_diff_ci95_high": completion_ci[1],
                }
            )
    adjusted = _bh_adjust([row["p_raw"] for row in comparisons])
    for row, p_adj in zip(comparisons, adjusted):
        row["p_bh"] = p_adj
        row["planning_better"] = bool(
            row["mean_control_minus_tied_burden"] > 0
            and row["burden_gain_ci95_low"] > 0
            and p_adj < 0.05
        )
        row["no_success_regression"] = bool(
            row["completion_diff_ci95_low"] >= SUCCESS_NONINFERIORITY_MARGIN
        )
        significance.append(row)

    g2_cells: List[dict] = []
    for config in sorted({row["config"] for row in comparisons}):
        rows = [row for row in comparisons if row["config"] == config]
        passed = len(rows) == 2 and all(
            row["planning_better"] and row["no_success_regression"] for row in rows
        )
        g2_cells.append({"config": config, "K": 8, "passed": passed})
    g2 = any(row["passed"] for row in g2_cells)

    state_summary = (
        state.groupby(["config", "K", "arm", "cycle"], as_index=False)
        .agg(
            state_mae_mean=("state_mae", "mean"),
            rank_corr_mean=("rank_corr", "mean"),
            bellman_residual_mean=("bellman_residual", "mean"),
        )
        .to_dict(orient="records")
    )

    strata: List[dict] = []
    tied_rows = averaged[averaged["arm"] == "tied_refiner"].copy()
    for (config, K), group in tied_rows.groupby(["config", "K"]):
        pivot = group.pivot(index="world_idx", columns="cycle", values="expansion_burden")
        distances = group.groupby("world_idx")["final_transition_hops"].first()
        common = pivot.dropna().index.intersection(distances.index)
        if len(common) < 3 or 1 not in pivot or 8 not in pivot:
            continue
        ranks = distances.loc[common].rank(method="first")
        labels = pd.qcut(ranks, q=3, labels=["near", "middle", "far"])
        improvement = pivot.loc[common, 1] - pivot.loc[common, 8]
        for label in ("near", "middle", "far"):
            mask = labels == label
            strata.append(
                {
                    "config": config,
                    "K": int(K),
                    "distance_stratum": label,
                    "n_worlds": int(mask.sum()),
                    "mean_cycle1_minus_cycle8_burden": float(improvement[mask].mean()),
                }
            )

    probe_authorized = True  # analysis only runs over G0-authorized cells
    verdict = R.c12b_verdict(g0=probe_authorized, g1=g1, g2=g2)
    summary = {
        "schema_version": R.SCHEMA_VERSION,
        "scale": scale,
        "development_only": scale != "full",
        "independent_unit": "world; model seeds averaged within world",
        "primary_planning_metric": "expansions / binding_budget; lower is better",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "permutation_draws": PERMUTATION_DRAWS,
        "multiplicity": "Benjamini-Hochberg over four G2-B deep-cell comparisons",
        "success_noninferiority_margin": SUCCESS_NONINFERIORITY_MARGIN,
        "gates": {
            "G1_B": {"passed": g1, "cells": g1_cells, "by_config": g1_by_config},
            "G2_B": {"passed": g2, "cells": g2_cells},
            "G3_B": verdict,
        },
        "state_summary": state_summary,
        "distance_strata": strata,
    }
    return summary, significance


def tiny_overfit_all_arms(
    sample: GraphSample, cfg: R.C12RefinerConfig, device: torch.device, steps: int = 30
) -> List[dict]:
    rows: List[dict] = []
    batch = batch_samples([sample], device)
    for arm_idx, arm in enumerate(R.ARM_NAMES):
        torch.manual_seed(1900 + arm_idx)
        model = R.build_arm(arm, cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        losses: List[float] = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch.node_feats, batch.edge_index, batch.edge_feats, batch.h_legsum_norm
            )
            loss = supervised_loss(outputs, batch, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
        passed = min(losses[-5:]) < losses[0]
        rows.append(
            {
                "arm": arm,
                "steps": steps,
                "first_loss": losses[0],
                "final_loss": losses[-1],
                "best_last5_loss": min(losses[-5:]),
                "passed": passed,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not all(row["passed"] for row in rows):
        raise AssertionError(f"tiny-overfit failure: {rows}")
    return rows


def _scale_contract(scale: str, cfg: R.C12RefinerConfig) -> dict:
    cells = [cell for cell in R.build_cell_grid(cfg) if int(cell["K"]) in (2, 8)]
    if scale == "smoke":
        cells = [cell for cell in cells if cell["config_label"] == "A" and cell["K"] == 8]
        return {"cells": cells, "model_seeds": (0,), "n_train": 2, "n_val": 1, "n_eval": 2, "epochs": 3}
    if scale == "pilot":
        cells = [cell for cell in cells if cell["config_label"] == "A" and cell["K"] == 8]
        return {
            "cells": cells,
            "model_seeds": (0,),
            "n_train": cfg.n_train_worlds,
            "n_val": cfg.n_val_worlds,
            "n_eval": cfg.n_test_worlds,
            "epochs": cfg.epochs,
        }
    if scale == "full":
        return {
            "cells": cells,
            "model_seeds": cfg.train_seeds,
            "n_train": cfg.n_train_worlds,
            "n_val": cfg.n_val_worlds,
            "n_eval": cfg.n_test_worlds,
            "epochs": cfg.epochs,
        }
    raise ValueError(f"unknown scale {scale!r}")


def _checkpoint_name(arm: str, cell: Mapping[str, object], seed: int) -> str:
    return f"{arm}_config{cell['config_label']}_K{cell['K']}_seed{seed}.pt"


def _validate_probe(out_dir: Path) -> dict:
    path = out_dir / "probe" / "c12b_probe_summary.json"
    if not path.exists():
        raise RuntimeError(f"G0-B summary is required before training: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError("G0-B summary is not complete")
    authorized = {(row["config"], int(row["K"])) for row in payload["authorized_cells"]}
    expected = {("A", 2), ("A", 8), ("C", 2), ("C", 8)}
    if not expected.issubset(authorized):
        raise RuntimeError(f"required K2/K8 cells are not authorized: {authorized}")
    return payload


def _validate_runtime_gate(out_dir: Path) -> dict:
    path = out_dir / "results" / "pilot" / "runtime_projection.json"
    if not path.exists():
        raise RuntimeError(f"pilot runtime gate is required before full: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("full_authorized", False):
        raise RuntimeError(
            "full grid is stopped by the preregistered runtime cap; choose local/remote explicitly"
        )
    return payload


def _integrity(
    *,
    scale: str,
    contract: Mapping[str, object],
    checkpoint_dir: Path,
    state_rows: Sequence[dict],
    eval_rows: Sequence[dict],
    result_dir: Path,
) -> dict:
    cells = contract["cells"]
    seeds = contract["model_seeds"]
    n_eval = int(contract["n_eval"])
    expected_checkpoints = len(cells) * len(seeds) * len(R.ARM_NAMES)
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    expected_learned_per_world = 1 + 1 + 4 + 4
    expected_learned = len(cells) * len(seeds) * n_eval * expected_learned_per_world
    expected_references = len(cells) * n_eval * len(R.REFERENCE_NAMES)
    expected_rows = expected_learned + expected_references

    eval_keys = [
        (
            row["config"], row["K"], row["world_idx"], row["model_seed"], row["arm"], row["cycle"]
        )
        for row in eval_rows
    ]
    state_keys = [
        (
            row["config"], row["K"], row["world_idx"], row["model_seed"], row["arm"], row["cycle"]
        )
        for row in state_rows
    ]
    checks = {
        "checkpoint_count": len(checkpoints) == expected_checkpoints,
        "eval_row_count": len(eval_rows) == expected_rows,
        "state_row_count": len(state_rows) == expected_rows,
        "eval_keys_unique": len(eval_keys) == len(set(eval_keys)),
        "state_keys_unique": len(state_keys) == len(set(state_keys)),
        "state_metrics_finite": all(
            math.isfinite(float(row[name]))
            for row in state_rows
            for name in ("state_mae", "rank_corr", "bellman_residual")
        ),
    }
    artifacts = {}
    for name in ("c12b_state_metrics.csv", "c12b_eval_raw.csv", "c12b_summary.json", "c12b_significance.csv"):
        path = result_dir / name
        artifacts[name] = {"exists": path.exists(), "sha256": _sha256(path) if path.exists() else None}
        checks[f"artifact_{name}"] = path.exists()
    return {
        "scale": scale,
        "passed": all(checks.values()),
        "checks": checks,
        "expected": {
            "checkpoints": expected_checkpoints,
            "rows": expected_rows,
            "learned_rows": expected_learned,
            "reference_rows": expected_references,
        },
        "observed": {
            "checkpoints": len(checkpoints),
            "eval_rows": len(eval_rows),
            "state_rows": len(state_rows),
        },
        "artifacts": artifacts,
    }


def run_experiment(
    *,
    out_dir: Path,
    scale: str,
    device_name: str,
    force: bool = False,
) -> dict:
    cfg = R.C12RefinerConfig()
    out_dir = Path(out_dir)
    probe = _validate_probe(out_dir)
    if scale == "full":
        _validate_runtime_gate(out_dir)
    contract = _scale_contract(scale, cfg)
    device = _device(device_name)
    checkpoint_dir = out_dir / "checkpoints"
    result_dir = out_dir / "results" if scale == "full" else out_dir / "results" / scale
    dataset_dir = out_dir / "datasets" / scale
    manifest_path = out_dir / "manifest.json"
    archive_manifest = out_dir / "manifests" / f"{scale}.json"
    run_started = time.perf_counter()

    manifest = {
        "schema_version": R.SCHEMA_VERSION,
        "track": "C12-B",
        "scale": scale,
        "status": "running",
        "device": str(device),
        "probe_raw_sha256": probe["artifacts"]["raw_sha256"],
        "contract": R._json_safe(contract),
        "training": [],
        "failures": [],
    }
    _write_json(manifest_path, manifest)

    all_state_rows: List[dict] = []
    all_eval_rows: List[dict] = []
    total_train_wall_s = 0.0
    smoke_overfit: Optional[List[dict]] = None

    try:
        for cell in contract["cells"]:
            label = f"{cell['config_label']}_K{cell['K']}"
            print(f"[C12-B data] collecting {label}", flush=True)
            train_bundles, train_ledger = R.collect_cell_dataset(
                cell, "train", int(contract["n_train"]), cfg
            )
            val_bundles, val_ledger = R.collect_cell_dataset(
                cell, "validation", int(contract["n_val"]), cfg
            )
            if scale == "full":
                eval_bundles, eval_ledger = R.collect_cell_dataset(
                    cell, "test", int(contract["n_eval"]), cfg
                )
            else:
                eval_bundles, eval_ledger = _collect_pilot_eval(
                    cell, int(contract["n_eval"]), cfg
                )
            _write_csv(dataset_dir / f"{label}_train_seed_ledger.csv", train_ledger)
            _write_csv(dataset_dir / f"{label}_validation_seed_ledger.csv", val_ledger)
            _write_csv(dataset_dir / f"{label}_{'test' if scale == 'full' else 'pilot_eval'}_seed_ledger.csv", eval_ledger)
            train_samples = prepare_samples(train_bundles, cfg)
            val_samples = prepare_samples(val_bundles, cfg)
            eval_samples = prepare_samples(eval_bundles, cfg)

            if scale == "smoke" and smoke_overfit is None:
                smoke_overfit = tiny_overfit_all_arms(train_samples[0], cfg, device)
                _write_json(result_dir / "tiny_overfit.json", smoke_overfit)

            for model_seed in contract["model_seeds"]:
                for arm in R.ARM_NAMES:
                    ckpt = checkpoint_dir / _checkpoint_name(arm, cell, int(model_seed))
                    meta = train_arm(
                        arm,
                        cell,
                        int(model_seed),
                        train_samples,
                        val_samples,
                        cfg,
                        device,
                        int(contract["epochs"]),
                        ckpt,
                        force=force,
                    )
                    total_train_wall_s += 0.0 if meta.get("resumed") else float(meta["train_wall_s"])
                    manifest["training"].append(meta)
                    _write_json(manifest_path, manifest)

                state_rows, eval_rows = evaluate_cell_seed(
                    cell=cell,
                    model_seed=int(model_seed),
                    eval_samples=eval_samples,
                    cfg=cfg,
                    checkpoint_dir=checkpoint_dir,
                    device=device,
                    include_references=int(model_seed) == int(contract["model_seeds"][0]),
                )
                all_state_rows.extend(state_rows)
                all_eval_rows.extend(eval_rows)

        _write_csv(result_dir / "c12b_state_metrics.csv", all_state_rows)
        _write_csv(result_dir / "c12b_eval_raw.csv", all_eval_rows)
        summary, significance = analyze_results(all_state_rows, all_eval_rows, scale=scale)
        _write_json(result_dir / "c12b_summary.json", summary)
        # A zero-row significance table is possible in smoke/pilot.  Keep a
        # valid artifact with an explicit noninferential record.
        if not significance:
            significance = [{"family": "G2-B", "status": "insufficient_cells", "scale": scale}]
        _write_csv(result_dir / "c12b_significance.csv", significance)

        elapsed_s = time.perf_counter() - run_started
        if scale == "pilot":
            full_multiplier = 12.0  # 4 authorized cells * 3 seeds / 1 pilot cell-seed
            if device.type == "cuda":
                projected_gpu_h = total_train_wall_s * full_multiplier / 3600.0
                projected_cpu_h = max(0.0, elapsed_s - total_train_wall_s) * full_multiplier / 3600.0
            else:
                projected_gpu_h = 0.0
                projected_cpu_h = elapsed_s * full_multiplier / 3600.0
            runtime = {
                "measured_pilot_wall_s": elapsed_s,
                "measured_training_wall_s": total_train_wall_s,
                "projection_multiplier": full_multiplier,
                "projected_gpu_hours": projected_gpu_h,
                "projected_cpu_hours": projected_cpu_h,
                "gpu_cap_hours": cfg.max_projected_gpu_hours,
                "cpu_cap_hours": cfg.max_projected_cpu_hours,
                "full_authorized": bool(
                    projected_gpu_h <= cfg.max_projected_gpu_hours
                    and projected_cpu_h <= cfg.max_projected_cpu_hours
                ),
            }
            _write_json(result_dir / "runtime_projection.json", runtime)
            manifest["runtime_gate"] = runtime

        integrity = _integrity(
            scale=scale,
            contract=contract,
            checkpoint_dir=checkpoint_dir,
            state_rows=all_state_rows,
            eval_rows=all_eval_rows,
            result_dir=result_dir,
        )
        _write_json(result_dir / "integrity.json", integrity)
        if not integrity["passed"]:
            raise AssertionError(f"artifact integrity failed: {integrity}")

        manifest["status"] = "complete"
        manifest["elapsed_s"] = elapsed_s
        manifest["result_dir"] = str(result_dir)
        manifest["integrity"] = integrity
        _write_json(manifest_path, manifest)
        _write_json(archive_manifest, manifest)
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        manifest["elapsed_s"] = time.perf_counter() - run_started
        _write_json(manifest_path, manifest)
        _write_json(archive_manifest, manifest)
        raise


def run_mode(args) -> dict:
    if args.mode not in ("smoke", "pilot", "full"):
        raise ValueError(f"pipeline cannot run mode {args.mode!r}")
    return run_experiment(
        out_dir=Path(args.out_dir),
        scale=args.mode,
        device_name=args.device,
        force=bool(args.force),
    )
