"""C12-B: tied iterative value refinement on C11 product graphs.

This module is deliberately C12-owned.  It imports the frozen C11 world,
oracle, graph, and A* interfaces but never changes C11 defaults or artifacts.

Execution order is enforced by the CLI:

1. ``probe`` runs the pre-data K=16 G0-B authorization gate.
2. ``smoke`` exercises every arm and the artifact contract.
3. ``pilot`` runs A/K=8 with one model seed and writes a runtime projection.
4. ``full`` is permitted only after the probe and runtime gates authorize it.

The product edges encoded by C11 point in the direction of feasible forward
motion.  Cost-to-go information therefore propagates from each edge's
destination back to its source.  All C12-B matched refinement controls use
that same reverse-value direction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

import continuous_prm_common as C
import continuous_prm_c11_headroom as C11P
import continuous_prm_c11_mission as C11


SCHEMA_VERSION = "c12b-v1"
SPLITS: Tuple[str, ...] = ("train", "test", "validation", "probe")
ARM_NAMES: Tuple[str, ...] = (
    "c11_gnn8",
    "shallow_param_match",
    "untied_compute_match",
    "tied_refiner",
)
REFERENCE_NAMES: Tuple[str, ...] = ("h_legsum", "h_oracle")


@dataclass
class C12RefinerConfig(C11.C11MissionConfig):
    """Frozen C12-B constants.

    The memory/time limits below are feasibility guards, not learned-result
    thresholds.  They are intentionally generous for a 17*192-state graph
    and were fixed before the K=16 probe was inspected.
    """

    k_values: Tuple[int, ...] = (2, 8, 16)
    k_max: int = 16
    seq_max: int = 18
    n_val_worlds: int = 10

    report_cycles: Tuple[int, ...] = (1, 2, 4, 8)
    refinement_cycles: int = 8
    diagnostic_cycles: int = 16
    deep_supervision_weights: Dict[int, float] = field(
        default_factory=lambda: {1: 0.1, 2: 0.2, 4: 0.3, 8: 0.4}
    )
    graph_accum_nodes: int = 8192

    probe_budgets: Tuple[int, ...] = (100, 200, 400, 800, 1600, 3200, 6400, 12800)
    g0_min_worlds: int = 20
    g0_max_expansion_ratio: float = 0.30
    g0_min_median_hops: float = 8.0  # strict: median must be > 8
    g0_max_label_wall_s: float = 60.0
    g0_max_peak_rss_bytes: int = 8 * 1024**3
    g0_max_graph_bytes: int = 256 * 1024**2

    # The agreed execution caps.  A measured pilot projection above either
    # threshold stops before the full grid; the grid is never silently cut.
    max_projected_gpu_hours: float = 36.0
    max_projected_cpu_hours: float = 24.0


def build_cell_grid(cfg: Optional[C12RefinerConfig] = None) -> List[dict]:
    """Return the preregistered A/C x K={2,8,16} grid in stable order."""
    cfg = cfg or C12RefinerConfig()
    defs = (
        ("A", "C_hard_maze", 0, False),
        ("C", "C_hard_maze", 2, True),
    )
    return [
        {
            "config_label": label,
            "spec_name": spec,
            "config_idx": idx,
            "K": int(K),
            "doors": doors,
        }
        for label, spec, idx, doors in defs
        for K in cfg.k_values
    ]


# C11 TRAIN/TEST streams are reused at K={2,8}.  These new bases are all
# below NumPy's uint32 seed ceiling even after offsets, and are disjoint from
# both C11 and one another over the registered attempt envelope.
_K16_TRAIN_BASE = 1_400_000_001
_K16_TEST_BASE = 1_700_000_001
_VALIDATION_BASE = 2_100_000_001
_PROBE_BASE = 2_800_000_001


def _seed_from_base(base: int, w: int, config_idx: int, K: int) -> int:
    seed = int(base + 7919 * int(w) + 104729 * int(config_idx) + 15485863 * int(K))
    if not (0 <= seed < 2**32):
        raise ValueError(f"seed outside uint32 range: {seed}")
    return seed


def world_seed(split: str, w: int, config_idx: int, K: int) -> int:
    """Return the frozen candidate seed for a split/cell/attempt."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split == "train" and K in (2, 8):
        return C11.train_seed(w, config_idx, K)
    if split == "test" and K in (2, 8):
        return C11.test_seed(w, config_idx, K)
    if split == "train":
        base = _K16_TRAIN_BASE
    elif split == "test":
        base = _K16_TEST_BASE
    elif split == "validation":
        base = _VALIDATION_BASE
    else:
        base = _PROBE_BASE
    return _seed_from_base(base, w, config_idx, K)


def collect_cell_dataset(
    cell: Mapping[str, object],
    split: str,
    n_worlds: int,
    cfg: Optional[C12RefinerConfig] = None,
    *,
    allow_partial: bool = False,
) -> Tuple[List[C11.WorldBundle], List[dict]]:
    """Collect valid C11 bundles and an auditable candidate-seed ledger."""
    cfg = cfg or C12RefinerConfig()
    bundles: List[C11.WorldBundle] = []
    ledger: List[dict] = []
    attempt = 0
    while len(bundles) < int(n_worlds):
        if attempt >= cfg.max_world_attempts:
            if allow_partial:
                break
            raise RuntimeError(
                f"only found {len(bundles)}/{n_worlds} valid {split} worlds for {dict(cell)} "
                f"after {attempt} attempts"
            )
        seed = world_seed(split, attempt, int(cell["config_idx"]), int(cell["K"]))
        row = {
            "split": split,
            "attempt_idx": attempt,
            "seed": seed,
            "config": str(cell["config_label"]),
            "K": int(cell["K"]),
            "accepted": False,
            "reason": "",
        }
        attempt += 1
        collect_t0 = time.perf_counter()
        try:
            bundle = C11.collect_world_bundle(dict(cell), seed, cfg)
        except RuntimeError as exc:
            row["collect_wall_s"] = time.perf_counter() - collect_t0
            row["reason"] = str(exc)
            ledger.append(row)
            continue
        row["collect_wall_s"] = time.perf_counter() - collect_t0
        row["peak_rss_bytes"] = _process_peak_rss()
        row["accepted"] = True
        row["valid_world_idx"] = len(bundles)
        ledger.append(row)
        bundles.append(bundle)
    return bundles, ledger


def reverse_mean_aggregate(values_at_destination: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Mean destination values/messages onto each forward edge's source."""
    n_nodes = int(values_at_destination.shape[0])
    if edge_index.numel() == 0:
        return torch.zeros_like(values_at_destination)
    src, dst = edge_index[0], edge_index[1]
    gathered = values_at_destination[dst]
    out = torch.zeros_like(values_at_destination)
    out.index_add_(0, src, gathered)
    degree = torch.zeros(n_nodes, device=out.device, dtype=out.dtype)
    degree.index_add_(0, src, torch.ones_like(src, dtype=out.dtype))
    shape = (n_nodes,) + (1,) * (out.ndim - 1)
    return out / degree.clamp(min=1.0).reshape(shape)


def final_transition_distance(edge_index: np.ndarray, n_roadmap_nodes: int, K: int) -> float:
    """Directed hops from canonical start to the final-transition edge.

    A relevant edge has a source in stage K-1 and destination in stage K.
    The returned distance is to that edge's source.  This definition was
    frozen before the probe because reverse value propagation needs at least
    that many recurrent rounds for final-transition information to affect
    the start state.  Unreachable sources return infinity.
    """
    if K <= 0:
        return 0.0
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.size == 0:
        return float("inf")
    srcs = edge_index[0]
    dsts = edge_index[1]
    target_sources = set(
        int(src)
        for src, dst in zip(srcs.tolist(), dsts.tolist())
        if src // n_roadmap_nodes == K - 1 and dst // n_roadmap_nodes == K
    )
    if not target_sources:
        return float("inf")
    adjacency: List[List[int]] = [[] for _ in range((K + 1) * n_roadmap_nodes)]
    for src, dst in zip(srcs.tolist(), dsts.tolist()):
        adjacency[int(src)].append(int(dst))
    dist = [-1] * len(adjacency)
    dist[0] = 0
    q: deque[int] = deque([0])
    while q:
        node = q.popleft()
        if node in target_sources:
            return float(dist[node])
        for nxt in adjacency[node]:
            if dist[nxt] < 0:
                dist[nxt] = dist[node] + 1
                q.append(nxt)
    return float("inf")


def evaluate_g0b_cell(
    *,
    valid_worlds: int,
    expansion_ratio: Optional[float],
    median_final_transition_hops: float,
    max_label_wall_s: float,
    max_peak_rss_bytes: int,
    max_graph_bytes: int,
    degenerate_budget: bool,
    cfg: Optional[C12RefinerConfig] = None,
) -> dict:
    """Evaluate every pre-registered G0-B conjunct for one K=16 cell."""
    cfg = cfg or C12RefinerConfig()
    checks = {
        "enough_valid_worlds": int(valid_worlds) >= cfg.g0_min_worlds,
        "oracle_headroom": expansion_ratio is not None
        and math.isfinite(float(expansion_ratio))
        and float(expansion_ratio) <= cfg.g0_max_expansion_ratio,
        "deep_enough": float(median_final_transition_hops) > cfg.g0_min_median_hops,
        "label_time_feasible": float(max_label_wall_s) <= cfg.g0_max_label_wall_s,
        "rss_feasible": int(max_peak_rss_bytes) <= cfg.g0_max_peak_rss_bytes,
        "graph_memory_feasible": int(max_graph_bytes) <= cfg.g0_max_graph_bytes,
        "binding_budget_nondegenerate": not bool(degenerate_budget),
    }
    return {"passed": all(checks.values()), "checks": checks}


class SharedGraphBlock(nn.Module):
    """One reverse-value message/update block."""

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = int(hidden)
        self.message = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_feats: torch.Tensor) -> torch.Tensor:
        n_nodes = int(h.shape[0])
        if edge_index.numel() == 0:
            agg = torch.zeros_like(h)
        else:
            src, dst = edge_index[0], edge_index[1]
            msg = self.message(torch.cat([h[dst], edge_feats], dim=-1))
            agg = torch.zeros(n_nodes, self.hidden, device=h.device, dtype=h.dtype)
            agg.index_add_(0, src, msg)
            degree = torch.zeros(n_nodes, device=h.device, dtype=h.dtype)
            degree.index_add_(0, src, torch.ones_like(src, dtype=h.dtype))
            agg = agg / degree.clamp(min=1.0)[:, None]
        return self.update(torch.cat([h, agg], dim=-1))


class _RefinerBase(nn.Module):
    def __init__(self, cfg: C12RefinerConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden = int(cfg.gnn_hidden)
        self.cap = float(cfg.residual_cap)
        # 14 C11 node features plus normalized h_legsum initialization.
        self.encoder = nn.Linear(15, self.hidden)
        self.readout = nn.Sequential(
            nn.Linear(self.hidden, self.hidden), nn.GELU(), nn.Linear(self.hidden, 1)
        )

    def encode(self, node_feats: torch.Tensor, h_legsum: torch.Tensor) -> torch.Tensor:
        if h_legsum.ndim == 1:
            h_legsum = h_legsum[:, None]
        return self.encoder(torch.cat([node_feats, h_legsum], dim=-1))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.softplus(self.readout(h).squeeze(-1)), 0.0, self.cap)


class TiedGraphRefiner(_RefinerBase):
    """Apply one shared block recurrently and expose cycles 1/2/4/8."""

    def __init__(self, cfg: Optional[C12RefinerConfig] = None):
        cfg = cfg or C12RefinerConfig()
        super().__init__(cfg)
        self.block = SharedGraphBlock(self.hidden)
        self.edge_applications = int(cfg.refinement_cycles)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        h_legsum: torch.Tensor,
        max_cycles: Optional[int] = None,
    ) -> Dict[int, torch.Tensor]:
        cycles = int(max_cycles or self.cfg.refinement_cycles)
        if cycles < 1 or cycles > self.cfg.diagnostic_cycles:
            raise ValueError(f"max_cycles must be in [1,{self.cfg.diagnostic_cycles}], got {cycles}")
        emit = set(c for c in self.cfg.report_cycles if c <= cycles)
        if cycles == self.cfg.diagnostic_cycles:
            emit.add(cycles)
        h = self.encode(node_feats, h_legsum)
        out: Dict[int, torch.Tensor] = {}
        for cycle in range(1, cycles + 1):
            h = h + self.block(h, edge_index, edge_feats)
            if cycle in emit:
                out[cycle] = self.decode(h)
        return out


class ShallowParamMatch(_RefinerBase):
    """One application of the same-size block; exactly parameter matched."""

    def __init__(self, cfg: Optional[C12RefinerConfig] = None):
        cfg = cfg or C12RefinerConfig()
        super().__init__(cfg)
        self.block = SharedGraphBlock(self.hidden)
        self.edge_applications = 1

    def forward(self, node_feats, edge_index, edge_feats, h_legsum, max_cycles=None):
        del max_cycles
        h = self.encode(node_feats, h_legsum)
        h = h + self.block(h, edge_index, edge_feats)
        return {1: self.decode(h)}


class UntiedComputeMatch(_RefinerBase):
    """Eight distinct blocks with the same edge-operation count as tied."""

    def __init__(self, cfg: Optional[C12RefinerConfig] = None):
        cfg = cfg or C12RefinerConfig()
        super().__init__(cfg)
        self.blocks = nn.ModuleList([SharedGraphBlock(self.hidden) for _ in range(cfg.refinement_cycles)])
        self.edge_applications = int(cfg.refinement_cycles)

    def forward(self, node_feats, edge_index, edge_feats, h_legsum, max_cycles=None):
        cycles = int(max_cycles or self.cfg.refinement_cycles)
        if cycles > len(self.blocks):
            raise ValueError("untied control has only the registered eight blocks")
        h = self.encode(node_feats, h_legsum)
        out: Dict[int, torch.Tensor] = {}
        for cycle, block in enumerate(self.blocks[:cycles], start=1):
            h = h + block(h, edge_index, edge_feats)
            if cycle in self.cfg.report_cycles:
                out[cycle] = self.decode(h)
        return out


class C11GNN8Control(nn.Module):
    """The exact C11 8-round forward-message architecture, retrained."""

    def __init__(self, cfg: Optional[C12RefinerConfig] = None):
        super().__init__()
        cfg = cfg or C12RefinerConfig()
        self.model = C11.ProductGraphGNN(cfg)
        self.edge_applications = int(cfg.gnn_rounds)

    def forward(self, node_feats, edge_index, edge_feats, h_legsum=None, max_cycles=None):
        del h_legsum, max_cycles
        return {8: self.model(node_feats, edge_index, edge_feats)}


def build_arm(name: str, cfg: Optional[C12RefinerConfig] = None) -> nn.Module:
    cfg = cfg or C12RefinerConfig()
    builders = {
        "c11_gnn8": C11GNN8Control,
        "shallow_param_match": ShallowParamMatch,
        "untied_compute_match": UntiedComputeMatch,
        "tied_refiner": TiedGraphRefiner,
    }
    if name not in builders:
        raise ValueError(f"unknown arm {name!r}; expected one of {ARM_NAMES}")
    return builders[name](cfg)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def c12b_verdict(*, g0: bool, g1: bool, g2: bool) -> dict:
    if not g0:
        return {"code": "not_authorized", "text": "G0-B failed; no deep-refinement verdict is authorized."}
    if g1 and g2:
        return {
            "code": "shared_refinement_positive",
            "text": "Shared iterative refinement is the missing propagation mechanism on at least one deep cell.",
        }
    if g1:
        return {
            "code": "propagation_only",
            "text": "Additional propagation helps, but weight tying has no demonstrated advantage.",
        }
    return {
        "code": "no_progressive_refinement",
        "text": "Iterative refinement does not progressively solve the registered target.",
    }


def _materialize_legsum(bundle: C11.WorldBundle) -> np.ndarray:
    K = len(bundle.wp)
    N = int(bundle.rm.points.shape[0])
    return np.asarray([[bundle.hl(i, s) for i in range(N)] for s in range(K + 1)], dtype=np.float64)


def _process_peak_rss() -> int:
    info = psutil.Process(os.getpid()).memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def _graph_bytes(graph: Mapping[str, np.ndarray]) -> int:
    return int(sum(int(v.nbytes) for v in graph.values()))


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_safe(obj: object) -> object:
    """Recursively normalize dataclass payloads for strict JSON."""
    if isinstance(obj, Mapping):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _write_json_atomic(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _matched_expansion_ratio(rows: Sequence[dict], binding_budget: int) -> Tuple[Optional[float], int]:
    by_arm: Dict[str, Dict[int, dict]] = {name: {} for name in REFERENCE_NAMES}
    for row in rows:
        if int(row["budget"]) != int(binding_budget) or not bool(row["found"]):
            continue
        by_arm[str(row["arm"])][int(row["world_idx"])] = row
    matched = sorted(set(by_arm["h_legsum"]) & set(by_arm["h_oracle"]))
    ratios = [
        float(by_arm["h_oracle"][w]["expansions"]) / float(by_arm["h_legsum"][w]["expansions"])
        for w in matched
        if int(by_arm["h_legsum"][w]["expansions"]) > 0
    ]
    return (float(statistics.median(ratios)) if ratios else None, len(ratios))


def _probe_cell(cell: dict, cfg: C12RefinerConfig) -> Tuple[List[dict], dict, List[dict]]:
    bundles, ledger = collect_cell_dataset(
        cell, "probe", cfg.g0_min_worlds, cfg, allow_partial=True
    )
    accepted_ledger = {int(row["seed"]): row for row in ledger if bool(row["accepted"])}
    rows: List[dict] = []
    per_world: List[dict] = []
    for world_idx, bundle in enumerate(bundles):
        # Oracle/targets were created during collection; the accepted seed
        # ledger carries that wall time.  Graph encoding is measured here.
        h_legsum = _materialize_legsum(bundle)
        graph_t0 = time.perf_counter()
        graph = C11.encode_product_graph(bundle, cfg)
        graph_wall_s = time.perf_counter() - graph_t0
        label_wall_s = float(accepted_ledger[int(bundle.seed)]["collect_wall_s"])
        graph_bytes = _graph_bytes(graph)
        peak_rss = max(
            _process_peak_rss(),
            int(accepted_ledger[int(bundle.seed)].get("peak_rss_bytes", 0)),
        )
        hops = final_transition_distance(graph["edge_index"], int(bundle.rm.points.shape[0]), len(bundle.wp))
        if not math.isfinite(hops):
            raise RuntimeError(f"final transition is unreachable in accepted world seed={bundle.seed}")

        world_meta = {
            "config": cell["config_label"],
            "K": int(cell["K"]),
            "world_idx": world_idx,
            "seed": int(bundle.seed),
            "label_wall_s": label_wall_s,
            "graph_wall_s": graph_wall_s,
            "final_transition_hops": hops,
            "graph_bytes": graph_bytes,
            "peak_rss_bytes": peak_rss,
            "n_product_nodes": int(graph["node_feats"].shape[0]),
            "n_product_edges": int(graph["edge_index"].shape[1]),
        }
        per_world.append(world_meta)

        for arm, heuristic in (("h_legsum", h_legsum), ("h_oracle", bundle.oracle)):
            for budget in cfg.probe_budgets:
                result = C11P.astar_product(
                    bundle.rm.adj,
                    bundle.wp,
                    heuristic,
                    int(budget),
                    bundle.adj_valid,
                )
                row = dict(world_meta)
                row.update({
                    "arm": arm,
                    "budget": int(budget),
                    "found": bool(result["found"]),
                    "cost": float(result["cost"]) if result["found"] else "",
                    "expansions": int(result["expansions"]),
                    "closed": int(result["closed"]),
                    "opt_cost": float(bundle.oracle[0, 0]),
                })
                if result["found"] and abs(float(result["cost"]) - float(bundle.oracle[0, 0])) > 1e-6:
                    raise AssertionError(f"A* cost mismatch for seed={bundle.seed}, arm={arm}, budget={budget}")
                rows.append(row)

    if rows:
        binding_budget, degenerate = C11P.calibrate_binding_budget(rows, cfg.probe_budgets)
        ratio, n_matched = _matched_expansion_ratio(rows, binding_budget)
    else:
        binding_budget, degenerate, ratio, n_matched = cfg.probe_budgets[-1], True, None, 0
    median_hops = (
        float(statistics.median(float(r["final_transition_hops"]) for r in per_world))
        if per_world else -1.0
    )
    max_label = max((float(r["label_wall_s"]) for r in per_world), default=0.0)
    max_rss = max((int(r["peak_rss_bytes"]) for r in per_world), default=_process_peak_rss())
    max_graph = max((int(r["graph_bytes"]) for r in per_world), default=0)
    gate = evaluate_g0b_cell(
        valid_worlds=len(per_world),
        expansion_ratio=ratio,
        median_final_transition_hops=median_hops,
        max_label_wall_s=max_label,
        max_peak_rss_bytes=max_rss,
        max_graph_bytes=max_graph,
        degenerate_budget=degenerate,
        cfg=cfg,
    )
    summary = {
        "config": cell["config_label"],
        "K": int(cell["K"]),
        "valid_worlds": len(per_world),
        "attempts": len(ledger),
        "required_valid_worlds": cfg.g0_min_worlds,
        "binding_budget": int(binding_budget),
        "degenerate_budget": bool(degenerate),
        "matched_oracle_legsum_median_ratio": ratio,
        "n_matched": n_matched,
        "median_final_transition_hops": median_hops,
        "max_label_wall_s": max_label,
        "max_peak_rss_bytes": max_rss,
        "max_graph_bytes": max_graph,
        "gate": gate,
    }
    return rows, summary, ledger


def run_probe(out_dir: Path, cfg: Optional[C12RefinerConfig] = None) -> dict:
    """Run and freeze both A/K16 and C/K16 G0-B cells."""
    cfg = cfg or C12RefinerConfig()
    out_dir = Path(out_dir)
    probe_dir = out_dir / "probe"
    raw_rows: List[dict] = []
    ledger_rows: List[dict] = []
    cells: List[dict] = []
    started = time.time()
    for cell in (c for c in build_cell_grid(cfg) if c["K"] == 16):
        print(f"[C12-B G0] collecting {cell['config_label']}/K16", flush=True)
        rows, summary, ledger = _probe_cell(cell, cfg)
        raw_rows.extend(rows)
        ledger_rows.extend(ledger)
        cells.append(summary)
        print(
            f"[C12-B G0] {cell['config_label']}/K16: pass={summary['gate']['passed']} "
            f"ratio={summary['matched_oracle_legsum_median_ratio']} "
            f"median_hops={summary['median_final_transition_hops']}",
            flush=True,
        )

    raw_path = probe_dir / "c12b_probe_raw.csv"
    ledger_path = probe_dir / "c12b_probe_seed_ledger.csv"
    _write_csv_atomic(raw_path, raw_rows)
    _write_csv_atomic(ledger_path, ledger_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "track": "C12-B",
        "started_unix": started,
        "completed_unix": time.time(),
        "frozen_definition": {
            "distance": "directed hops from flat start 0 to source of an edge entering final stage K",
            "value_message_direction": "forward-edge destination to source",
            "thresholds": {
                "min_valid_worlds": cfg.g0_min_worlds,
                "max_oracle_legsum_median_ratio": cfg.g0_max_expansion_ratio,
                "median_final_transition_hops_strictly_greater_than": cfg.g0_min_median_hops,
                "max_label_wall_s": cfg.g0_max_label_wall_s,
                "max_peak_rss_bytes": cfg.g0_max_peak_rss_bytes,
                "max_graph_bytes": cfg.g0_max_graph_bytes,
            },
        },
        "cells": cells,
        "authorized_cells": [
            {"config": label, "K": K}
            for label in ("A", "C")
            for K in (2, 8)
        ] + [
            {"config": str(cell["config"]), "K": 16}
            for cell in cells
            if bool(cell["gate"]["passed"])
        ],
        "k16_all_passed": all(bool(cell["gate"]["passed"]) for cell in cells),
        "artifacts": {
            "raw_csv": str(raw_path),
            "raw_sha256": _sha256(raw_path),
            "seed_ledger_csv": str(ledger_path),
            "seed_ledger_sha256": _sha256(ledger_path),
        },
        "config": asdict(cfg),
    }
    _write_json_atomic(probe_dir / "c12b_probe_summary.json", summary)
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "smoke", "pilot", "full", "analyze"), required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/c12_refiner"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true", help="replace completed artifacts for the selected mode")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    if args.mode == "probe":
        summary_path = args.out_dir / "probe" / "c12b_probe_summary.json"
        if summary_path.exists() and not args.force:
            raise SystemExit(f"completed probe artifact already exists: {summary_path}; pass --force to replace")
        summary = run_probe(args.out_dir)
        print(json.dumps({"status": summary["status"], "k16_all_passed": summary["k16_all_passed"]}, indent=2))
        return
    if args.mode in ("smoke", "pilot", "full"):
        from continuous_prm_c12_refiner_pipeline import run_mode

        manifest = run_mode(args)
        print(json.dumps({
            "status": manifest["status"], "scale": manifest["scale"],
            "result_dir": manifest["result_dir"],
        }, indent=2))
        return
    if args.mode == "analyze":
        from continuous_prm_c12_refiner_reanalyze import run

        verification = run(args.out_dir)
        print(json.dumps({
            "status": verification["status"], "source_rows": verification["source_rows"],
        }, indent=2))
        return


if __name__ == "__main__":
    main()
