#!/usr/bin/env python3
"""C13-I live current-state versus C7 map-derived comparison.

The direct operating points are selected only from the frozen C13-H candidate
study.  Every C7 heuristic is then recomputed live on the exact saved C7 world
recipe, and its result is checked against the historical C7 raw row.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np

import continuous_prm_common as C
import continuous_prm_c13_certified_search as S
import continuous_prm_c13_identifiability as I
import continuous_prm_c13_lhbl_candidate_study as D
import continuous_prm_c13_lhbl_focal_matched_control_diagnostic as F
import continuous_prm_c13_lhbl_generated_v3 as H
import continuous_prm_c13_lhbl_replication as R
import continuous_prm_c13_shared_queue as Q
import continuous_prm_c13_state_heuristic as C13
import continuous_prm_c7_hard_maps as M7
import continuous_prm_c7_integration_compare as C7


EXPECTED_CANDIDATE_RAW_SHA256 = (
    "727a4fbc2765c26d73112588c3fa4d095f269fda348191048a7cf0bfc9e86750"
)
EXPECTED_CHECKPOINT_04_SHA256 = (
    "dbfd516e3db8ac616f0a3a48f5323fbf1c12405c178ee50c5792388d70b64742"
)
EXPECTED_CHECKPOINT_08_SHA256 = (
    "ff00bd0ba33aafb06274c442e5b5cedacfcc7a27a376e377bfdbec07337a662c"
)

CURRENT_LOW = "current_lhbl_low_distortion"
CURRENT_PRIMARY = "current_lhbl_throughput"
CURRENT_BOUNDED = "current_lhbl_bounded_focal"
CURRENT_ARMS = (CURRENT_LOW, CURRENT_PRIMARY, CURRENT_BOUNDED)
ALL_SUITES = (
    "C_hard_maze",
    "C_hard_maze_dense",
    "C_hard_rooms",
    "C_hard_spiral",
    "C_hard_bugtrap",
    "C_hard_rooms_large",
)
MAP_FIELD_ARMS = ("field_hrm", "field_onlstm", "field_unet")
MAP_SCALAR_ARMS = ("scalar_hrm", "scalar_onlstm")
PRIMARY_COMPARATOR = "field_hrm"


@dataclass
class ComparisonConfig:
    source_run_dir: str = "runs/c13_lhbl_flat_48w"
    candidate_study_dir: str = "runs/c13_lhbl_candidate_study"
    c7_run_dir: str = "runs/c7_local"
    out_dir: str = "runs/c13_lhbl_c7_comparison"
    preregistration: str = (
        "../../docs/experiments/continuous/c13/design/"
        "2026-07-17-c13i-current-state-vs-map-conditioned.md"
    )
    suites: str = ",".join(ALL_SUITES)
    worlds: int = 24
    roadmap_nodes: int = 192
    roadmap_k: int = 7
    seed: int = 1234
    grid_size: int = 64
    sector_tokens: int = 16
    max_world_retries: int = 200
    generous_budget_factor: float = 2.0
    low_cost_ceiling: float = 1.05
    primary_cost_ceiling: float = 1.10
    low_iteration: int = 8
    low_alpha: float = 0.75
    primary_iteration: int = 8
    primary_alpha: float = 1.00
    bounded_iteration: int = 4
    bounded_alpha: float = 0.50
    bounded_w: float = 1.10
    bootstrap_replicates: int = 20_000
    bootstrap_seed: int = 613_337
    device: str = "cpu"


def resolve_paths(cfg: ComparisonConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    for field_name in (
        "source_run_dir",
        "candidate_study_dir",
        "c7_run_dir",
        "out_dir",
        "preregistration",
    ):
        value = Path(getattr(cfg, field_name))
        if not value.is_absolute():
            setattr(cfg, field_name, str((script_dir / value).resolve()))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _coerce_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"expected finite value, got {value!r}")
    return result


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def candidate_statistics(candidate_raw: Path) -> List[Dict[str, Any]]:
    groups: DefaultDict[Tuple[int, float], List[Mapping[str, str]]] = defaultdict(list)
    for row in _read_csv(candidate_raw):
        if row["variant"] != "model":
            continue
        groups[(int(row["iteration"]), float(row["alpha"]))].append(row)
    stats: List[Dict[str, Any]] = []
    for (iteration, alpha), rows in sorted(groups.items()):
        direct = np.asarray(
            [_finite_float(row["direct_astar_expansions"]) for row in rows],
            dtype=np.float64,
        )
        euclid = np.asarray(
            [_finite_float(row["euclid_control_expansions"]) for row in rows],
            dtype=np.float64,
        )
        costs = np.asarray(
            [_finite_float(row["direct_astar_cost_ratio_eval_only"]) for row in rows],
            dtype=np.float64,
        )
        delta = direct - euclid
        stats.append(
            {
                "variant": "model",
                "iteration": int(iteration),
                "alpha": float(alpha),
                "roadmaps": int(len(rows)),
                "mean_expansions": float(np.mean(direct)),
                "mean_euclid_expansions": float(np.mean(euclid)),
                "mean_delta": float(np.mean(delta)),
                "wins": int(np.sum(delta < 0.0)),
                "ties": int(np.sum(delta == 0.0)),
                "losses": int(np.sum(delta > 0.0)),
                "mean_cost_ratio": float(np.mean(costs)),
                "max_cost_ratio": float(np.max(costs)),
            }
        )
    return stats


def select_for_ceiling(
    stats: Sequence[Mapping[str, Any]], ceiling: float, expected_roadmaps: int = 48
) -> Dict[str, Any]:
    eligible = [
        dict(row)
        for row in stats
        if int(row["roadmaps"]) == int(expected_roadmaps)
        and int(row["wins"]) == int(expected_roadmaps)
        and int(row["ties"]) == 0
        and int(row["losses"]) == 0
        and float(row["max_cost_ratio"]) <= float(ceiling) + 1.0e-12
    ]
    if not eligible:
        raise RuntimeError(f"no direct operating point satisfies ceiling {ceiling}")
    selected = min(
        eligible,
        key=lambda row: (
            float(row["mean_expansions"]),
            float(row["max_cost_ratio"]),
            int(row["iteration"]),
            float(row["alpha"]),
        ),
    )
    selected["cost_ceiling"] = float(ceiling)
    selected["eligible_candidates"] = int(len(eligible))
    return selected


def build_selection_report(cfg: ComparisonConfig) -> Dict[str, Any]:
    candidate_raw = Path(cfg.candidate_study_dir) / "results" / "candidate_raw.csv"
    if S.file_sha256(candidate_raw) != EXPECTED_CANDIDATE_RAW_SHA256:
        raise RuntimeError("candidate-study raw hash changed")
    stats = candidate_statistics(candidate_raw)
    low = select_for_ceiling(stats, cfg.low_cost_ceiling)
    primary = select_for_ceiling(stats, cfg.primary_cost_ceiling)
    expected_low = (int(cfg.low_iteration), float(cfg.low_alpha))
    expected_primary = (int(cfg.primary_iteration), float(cfg.primary_alpha))
    observed_low = (int(low["iteration"]), float(low["alpha"]))
    observed_primary = (int(primary["iteration"]), float(primary["alpha"]))
    if observed_low != expected_low or observed_primary != expected_primary:
        raise RuntimeError(
            "development-only selection changed: "
            f"low={observed_low}, primary={observed_primary}"
        )
    checkpoints = {
        "iteration_04": Path(cfg.source_run_dir)
        / "checkpoints"
        / "flat_mlp_iteration_04.pt",
        "iteration_08": Path(cfg.source_run_dir)
        / "checkpoints"
        / "flat_mlp_iteration_08.pt",
    }
    observed_hashes = {name: S.file_sha256(path) for name, path in checkpoints.items()}
    if observed_hashes["iteration_04"] != EXPECTED_CHECKPOINT_04_SHA256:
        raise RuntimeError("iteration-4 checkpoint hash changed")
    if observed_hashes["iteration_08"] != EXPECTED_CHECKPOINT_08_SHA256:
        raise RuntimeError("iteration-8 checkpoint hash changed")
    return {
        "selection_data": str(candidate_raw),
        "selection_data_sha256": S.file_sha256(candidate_raw),
        "rule": {
            "required_roadmaps": 48,
            "required_pairwise_wins": 48,
            "ranking": [
                "lowest_mean_expansions",
                "lowest_max_cost_ratio",
                "earlier_iteration",
                "lower_alpha",
            ],
        },
        "low_distortion": low,
        "primary_throughput": primary,
        "bounded_control": {
            "iteration": int(cfg.bounded_iteration),
            "alpha": float(cfg.bounded_alpha),
            "algorithm": "reopening_fhat_focal",
            "w": float(cfg.bounded_w),
            "fresh3_confirmed": True,
        },
        "checkpoint_hashes": observed_hashes,
        "all_candidate_statistics": stats,
        "frozen_before_c7_current_model_evaluation": True,
    }


def _reconstruct_path(parent: np.ndarray, start_idx: int, goal_idx: int) -> List[int]:
    path = [int(goal_idx)]
    seen = {int(goal_idx)}
    while path[-1] != int(start_idx):
        predecessor = int(parent[path[-1]])
        if predecessor < 0 or predecessor in seen:
            return []
        path.append(predecessor)
        seen.add(predecessor)
    path.reverse()
    return path


def astar_with_path(
    adj: List[List[Tuple[int, float]]],
    heuristic: np.ndarray,
    budget: int,
    start_idx: int = 0,
    goal_idx: int = 1,
) -> Dict[str, Any]:
    """C7's exact no-reopen A* ordering, with parent bookkeeping only."""
    h = np.asarray(heuristic, dtype=np.float64).reshape(-1)
    n = len(adj)
    if h.shape != (n,) or not np.all(np.isfinite(h)):
        raise ValueError("heuristic must be a finite vector aligned with the graph")
    g = np.full(n, C.INF, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)
    g[start_idx] = 0.0
    heap: List[Tuple[float, float, int]] = [(float(h[start_idx]), 0.0, start_idx)]
    closed = np.zeros(n, dtype=np.bool_)
    expansions = 0
    while heap and expansions < int(budget):
        _, cur_g, node = heapq.heappop(heap)
        if closed[node] or cur_g != g[node]:
            continue
        closed[node] = True
        expansions += 1
        if node == goal_idx:
            return {
                "found": True,
                "cost": float(g[node]),
                "expansions": int(expansions),
                "closed": int(closed.sum()),
                "path": _reconstruct_path(parent, start_idx, goal_idx),
            }
        for neighbor, edge_cost in adj[node]:
            if closed[neighbor]:
                continue
            new_g = float(g[node] + edge_cost)
            if new_g < g[neighbor]:
                g[neighbor] = new_g
                parent[neighbor] = int(node)
                heapq.heappush(
                    heap,
                    (new_g + float(h[neighbor]), new_g, int(neighbor)),
                )
    return {
        "found": False,
        "cost": float("nan"),
        "expansions": int(expansions),
        "closed": int(closed.sum()),
        "path": [],
    }


def iter_c7_worlds_with_seed(
    spec: C.AnchorSpec,
    suite_idx: int,
    cfg: ComparisonConfig,
    roadmap_cfg: C.RoadmapConfig,
) -> Iterator[Tuple[int, int, int, C.World, C.Roadmap]]:
    valid = 0
    attempt = 0
    while valid < int(cfg.worlds) and attempt < int(cfg.worlds) * int(cfg.max_world_retries):
        attempt += 1
        world_seed = (
            int(cfg.seed)
            + 770_000
            + 1_000_003 * (int(suite_idx) + 1)
            + (valid + 1) * 7919
            + attempt
        )
        world = C.build_world(spec, world_seed, roadmap_cfg.min_start_goal_dist_frac)
        if world is None:
            continue
        roadmap_seed = world_seed + 17
        roadmap = C.build_prm(world, roadmap_cfg, seed=roadmap_seed)
        if roadmap is None or not roadmap.connected_to_goal[0]:
            continue
        yield valid, world_seed, roadmap_seed, world, roadmap
        valid += 1


def c7_config(cfg: ComparisonConfig) -> C7.C7Config:
    result = C7.C7Config(
        grid_size=int(cfg.grid_size),
        roadmap_nodes=int(cfg.roadmap_nodes),
        roadmap_k=int(cfg.roadmap_k),
        eval_suites=str(cfg.suites),
        scalar_backbones="hrm,onlstm",
        field_backbones="unet,onlstm,hrm",
        eval_worlds=int(cfg.worlds),
        seed=int(cfg.seed),
        scale="local",
        out_dir=str(cfg.c7_run_dir),
        cpu=True,
        sector_tokens=int(cfg.sector_tokens),
    )
    return C7.apply_scale_preset(result)


def calibrated_budgets(cfg: ComparisonConfig) -> Dict[str, List[int]]:
    payload = H._read_json(Path(cfg.c7_run_dir) / "calibration.json")
    result: Dict[str, List[int]] = {}
    generous = int(math.ceil(cfg.generous_budget_factor * cfg.roadmap_nodes))
    for suite in C13.parse_csv(cfg.suites):
        saved = [int(value) for value in payload["budgets"][suite]]
        result[suite] = sorted(set(saved + [generous]))
    return result


def historical_lookup(cfg: ComparisonConfig) -> Dict[Tuple[str, int, str, int], Dict[str, str]]:
    path = Path(cfg.c7_run_dir) / "results" / "continuous_prm_c7_eval_raw.csv"
    lookup: Dict[Tuple[str, int, str, int], Dict[str, str]] = {}
    for row in _read_csv(path):
        if row["mode"] != "astar":
            continue
        key = (
            row["suite"],
            int(row["world_index"]),
            row["provider"],
            int(row["budget"]),
        )
        lookup[key] = row
    return lookup


def parity_record(
    suite: str,
    world_index: int,
    provider: str,
    budget: int,
    result: Mapping[str, Any],
    historical: Mapping[Tuple[str, int, str, int], Mapping[str, str]],
) -> Dict[str, Any] | None:
    expected = historical.get((suite, int(world_index), provider, int(budget)))
    if expected is None:
        return None
    expected_found = _coerce_bool(expected["found"])
    live_found = bool(result["found"])
    found_equal = expected_found == live_found
    expansions_equal = int(expected["expansions"]) == int(result["expansions"])
    if expected_found and live_found:
        cost_close = math.isclose(
            float(expected["cost"]),
            float(result["cost"]),
            rel_tol=0.0,
            abs_tol=1.0e-8,
        )
    else:
        cost_close = found_equal
    return {
        "suite": suite,
        "world_index": int(world_index),
        "provider": provider,
        "budget": int(budget),
        "expected_found": bool(expected_found),
        "live_found": bool(live_found),
        "expected_expansions": int(expected["expansions"]),
        "live_expansions": int(result["expansions"]),
        "found_equal": bool(found_equal),
        "expansions_equal": bool(expansions_equal),
        "cost_close": bool(cost_close),
        "parity_pass": bool(found_equal and expansions_equal and cost_close),
    }


def _rank_diagnostic(
    suite: str,
    world_index: int,
    world_seed: int,
    arm: str,
    boundary: str,
    rank: np.ndarray,
    oracle: np.ndarray,
    representation_seconds: float,
    model_seconds: float,
) -> Dict[str, Any]:
    connected = np.isfinite(oracle) & (oracle < C.INF / 10.0)
    return {
        "suite": suite,
        "world_index": int(world_index),
        "world_seed": int(world_seed),
        "arm": arm,
        "runtime_information_boundary": boundary,
        "rank_vs_oracle_spearman_eval_only": I.safe_spearman(
            np.asarray(rank)[connected], np.asarray(oracle)[connected]
        ),
        "overestimate_rate_eval_only": float(
            np.mean(np.asarray(rank)[connected] > np.asarray(oracle)[connected] + 1.0e-9)
        ),
        "representation_seconds": float(representation_seconds),
        "model_seconds": float(model_seconds),
    }


def _completed_worlds(
    rows: Sequence[Mapping[str, Any]], arm_count: int, budget_count: int
) -> set[int]:
    counts: DefaultDict[int, int] = defaultdict(int)
    for row in rows:
        counts[int(row["world_index"])] += 1
    expected = int(arm_count) * int(budget_count)
    return {world for world, count in counts.items() if count == expected}


def _suite_paths(out_dir: Path, suite: str) -> Dict[str, Path]:
    root = C13.ensure_dir(out_dir / "results" / "_shards" / "c13i" / suite)
    return {
        "raw": root / "raw.csv",
        "diagnostics": root / "provider_diagnostics.csv",
        "parity": root / "historical_parity.csv",
        "meta": root / "meta.json",
    }


def _read_existing(path: Path) -> List[Dict[str, str]]:
    return _read_csv(path) if path.exists() and path.stat().st_size else []


def _boundary_for_provider(name: str) -> str:
    if name.startswith("field_"):
        return "complete_64x64_occupancy_goal_raster"
    if name.startswith("scalar_"):
        return "global_obstacle_list_summaries_sectors_plus_rays_goal"
    if name == "oracle":
        return "exact_full_graph_cost_to_go_evaluation_only"
    if name == "euclid":
        return "current_goal_geometry_only"
    raise KeyError(name)


def run_eval(cfg: ComparisonConfig, selected_suites: Sequence[str]) -> Dict[str, Any]:
    import torch

    selection = build_selection_report(cfg)
    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    selection_path = C13.write_json(result_dir / "operating_point_selection.json", selection)
    device = R.resolve_device(cfg.device)
    if str(device) != "cpu":
        raise RuntimeError("C13-I is locked to CPU for exact C7 historical parity")

    study_cfg = D.CandidateStudyConfig(
        source_run_dir=cfg.source_run_dir,
        iterations=f"{cfg.bounded_iteration},{cfg.primary_iteration}",
        device=cfg.device,
    )
    models, training_cfg, _, checkpoint_paths = D.load_checkpoints(study_cfg, device)
    local_cfg = H.state_config(training_cfg)
    providers = C7._load_eval_providers(Path(cfg.c7_run_dir), c7_config(cfg), device)
    required_providers = {
        "euclid",
        "oracle",
        *MAP_FIELD_ARMS,
        *MAP_SCALAR_ARMS,
    }
    if set(providers) != required_providers:
        raise RuntimeError(
            f"unexpected C7 provider set: {sorted(providers)}; expected {sorted(required_providers)}"
        )
    budgets_by_suite = calibrated_budgets(cfg)
    historical = historical_lookup(cfg)
    M7.install_c7_hard_maps(cfg.sector_tokens)
    specs = C.build_anchor_specs()
    roadmap_cfg = C.RoadmapConfig(
        n_nodes=int(cfg.roadmap_nodes), k_neighbors=int(cfg.roadmap_k)
    )
    arms = sorted(providers) + list(CURRENT_ARMS)
    fingerprint_payload = {
        "config": asdict(cfg),
        "selection_sha256": S.file_sha256(selection_path),
        "implementation_sha256": S.file_sha256(Path(__file__).resolve()),
        "c7_calibration_sha256": S.file_sha256(Path(cfg.c7_run_dir) / "calibration.json"),
        "c7_raw_sha256": S.file_sha256(
            Path(cfg.c7_run_dir) / "results" / "continuous_prm_c7_eval_raw.csv"
        ),
        "checkpoint_hashes": selection["checkpoint_hashes"],
    }
    fingerprint = _json_hash(fingerprint_payload)

    for suite in selected_suites:
        suite_idx = ALL_SUITES.index(suite)
        paths = _suite_paths(Path(cfg.out_dir), suite)
        if paths["meta"].exists():
            previous = H._read_json(paths["meta"])
            if previous.get("fingerprint") != fingerprint:
                raise RuntimeError(f"existing shard fingerprint differs for {suite}")
        else:
            C13.write_json(
                paths["meta"],
                {
                    "suite": suite,
                    "fingerprint": fingerprint,
                    "fingerprint_payload": fingerprint_payload,
                    "budgets": budgets_by_suite[suite],
                    "arms": arms,
                },
            )
        rows: List[Dict[str, Any]] = list(_read_existing(paths["raw"]))
        diagnostics: List[Dict[str, Any]] = list(_read_existing(paths["diagnostics"]))
        parity: List[Dict[str, Any]] = list(_read_existing(paths["parity"]))
        completed = _completed_worlds(rows, len(arms), len(budgets_by_suite[suite]))
        if len(completed) == int(cfg.worlds):
            print(f"[c13i] {suite}: complete shard already present", flush=True)
            continue
        for world_index, world_seed, roadmap_seed, world, roadmap in iter_c7_worlds_with_seed(
            specs[suite], suite_idx, cfg, roadmap_cfg
        ):
            if world_index in completed:
                continue
            world_started = time.perf_counter()
            optimal = float(roadmap.dist_to_goal[0])
            if not math.isfinite(optimal) or optimal >= C.INF / 10.0:
                raise RuntimeError("accepted C7 roadmap is disconnected")
            oracle = np.asarray(roadmap.dist_to_goal, dtype=np.float64)

            ranks: Dict[str, np.ndarray] = {}
            rank_times: Dict[str, Tuple[float, float]] = {}
            for name, provider in sorted(providers.items()):
                started = time.perf_counter()
                ranks[name] = np.asarray(
                    provider.node_h(world, roadmap, goal_idx=1), dtype=np.float64
                )
                rank_times[name] = (float(time.perf_counter() - started), 0.0)
                diagnostics.append(
                    _rank_diagnostic(
                        suite,
                        world_index,
                        world_seed,
                        name,
                        _boundary_for_provider(name),
                        ranks[name],
                        oracle,
                        rank_times[name][0],
                        0.0,
                    )
                )
            euclid = ranks["euclid"]

            feature_started = time.perf_counter()
            features = C13.make_local_state_features(
                world, roadmap.points, roadmap.adj, local_cfg
            )
            feature_seconds = float(time.perf_counter() - feature_started)
            predictions: Dict[int, np.ndarray] = {}
            prediction_times: Dict[int, float] = {}
            for iteration in sorted(models):
                started = time.perf_counter()
                predictions[iteration] = np.asarray(
                    I.predict_model(models[iteration], features, device), dtype=np.float64
                )
                prediction_times[iteration] = float(time.perf_counter() - started)
            learned_08 = euclid + float(world.side_len) * predictions[cfg.primary_iteration]
            learned_04 = euclid + float(world.side_len) * predictions[cfg.bounded_iteration]
            ranks[CURRENT_LOW] = euclid + float(cfg.low_alpha) * (learned_08 - euclid)
            ranks[CURRENT_PRIMARY] = euclid + float(cfg.primary_alpha) * (learned_08 - euclid)
            ranks[CURRENT_BOUNDED] = euclid + float(cfg.bounded_alpha) * (learned_04 - euclid)
            rank_times[CURRENT_LOW] = (
                feature_seconds,
                prediction_times[cfg.primary_iteration],
            )
            rank_times[CURRENT_PRIMARY] = (
                feature_seconds,
                prediction_times[cfg.primary_iteration],
            )
            rank_times[CURRENT_BOUNDED] = (
                feature_seconds,
                prediction_times[cfg.bounded_iteration],
            )
            for name in CURRENT_ARMS:
                diagnostics.append(
                    _rank_diagnostic(
                        suite,
                        world_index,
                        world_seed,
                        name,
                        "current_goal_geometry_bounded_rays_one_hop_actions",
                        ranks[name],
                        oracle,
                        rank_times[name][0],
                        rank_times[name][1],
                    )
                )

            world_rows: List[Dict[str, Any]] = []
            world_parity: List[Dict[str, Any]] = []
            for arm in arms:
                for budget in budgets_by_suite[suite]:
                    search_started = time.perf_counter()
                    if arm == CURRENT_BOUNDED:
                        result = F.focal_search_with_path(
                            roadmap.adj,
                            euclid,
                            ranks[arm],
                            int(budget),
                            float(cfg.bounded_w),
                            "fhat",
                        )
                        algorithm = "reopening_fhat_focal"
                    else:
                        result = astar_with_path(
                            roadmap.adj, ranks[arm], int(budget)
                        )
                        algorithm = "no_reopen_astar"
                    search_seconds = float(time.perf_counter() - search_started)
                    found = bool(result["found"])
                    if found:
                        path_check = Q.validate_path(
                            roadmap.adj, result["path"], result["cost"]
                        )
                        cost = float(result["cost"])
                        cost_ratio = cost / optimal
                    else:
                        path_check = {"valid": False, "cost": float("nan"), "edges": 0}
                        cost = float("nan")
                        cost_ratio = float("nan")
                    anchor_f_min = float(result.get("anchor_f_min_at_return", float("nan")))
                    bound_violation = bool(
                        arm == CURRENT_BOUNDED
                        and found
                        and cost > float(cfg.bounded_w) * optimal + 1.0e-9
                    )
                    certificate_violation = bool(
                        arm == CURRENT_BOUNDED
                        and found
                        and math.isfinite(anchor_f_min)
                        and cost > float(cfg.bounded_w) * anchor_f_min + 1.0e-9
                    )
                    world_rows.append(
                        {
                            "suite": suite,
                            "world_index": int(world_index),
                            "world_seed": int(world_seed),
                            "roadmap_seed": int(roadmap_seed),
                            "nodes": int(len(roadmap.points)),
                            "edges": int(sum(len(group) for group in roadmap.adj) // 2),
                            "arm": arm,
                            "runtime_information_boundary": (
                                "current_goal_geometry_bounded_rays_one_hop_actions"
                                if arm in CURRENT_ARMS
                                else _boundary_for_provider(arm)
                            ),
                            "algorithm": algorithm,
                            "budget": int(budget),
                            "found": bool(found),
                            "path_valid": bool(path_check["valid"]),
                            "path_edges": int(path_check["edges"]),
                            "cost": cost,
                            "optimal": optimal,
                            "cost_ratio_eval_only": cost_ratio,
                            "expansions": int(result["expansions"]),
                            "closed": int(result.get("closed", 0)),
                            "max_expansions_per_state": int(
                                result.get("max_expansions_per_state", 1)
                            ),
                            "anchor_f_min_at_return": anchor_f_min,
                            "bound_violation_eval_only": bool(bound_violation),
                            "certificate_violation": bool(certificate_violation),
                            "representation_seconds": float(rank_times[arm][0]),
                            "model_seconds": float(rank_times[arm][1]),
                            "search_seconds": search_seconds,
                        }
                    )
                    if arm in providers:
                        check = parity_record(
                            suite,
                            world_index,
                            arm,
                            budget,
                            result,
                            historical,
                        )
                        if check is not None:
                            world_parity.append(check)
            rows.extend(world_rows)
            parity.extend(world_parity)
            C13.write_csv(paths["raw"], rows)
            C13.write_csv(paths["diagnostics"], diagnostics)
            C13.write_csv(paths["parity"], parity)
            print(
                f"[c13i] {suite} {world_index + 1}/{cfg.worlds}: "
                f"feature={feature_seconds:.3f}s total={time.perf_counter() - world_started:.3f}s "
                f"rows={len(rows)} parity_fail={sum(not _coerce_bool(r['parity_pass']) for r in parity)}",
                flush=True,
            )
        completed = _completed_worlds(rows, len(arms), len(budgets_by_suite[suite]))
        if len(completed) != int(cfg.worlds):
            raise RuntimeError(f"{suite} under-filled: {len(completed)}/{cfg.worlds}")

    return {
        "selection": selection,
        "selection_path": selection_path,
        "providers": sorted(providers),
        "checkpoint_paths": {str(key): str(value) for key, value in checkpoint_paths.items()},
        "local_state_config": asdict(local_cfg),
    }


def _bootstrap_mean_ci(
    values: Sequence[float], replicates: int, seed: int, confidence: float = 0.95
) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=np.float64)
    for start in range(0, int(replicates), 1000):
        count = min(1000, int(replicates) - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        samples[start : start + count] = np.mean(array[indices], axis=1)
    tail = (1.0 - float(confidence)) * 50.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return float(low), float(high)


def _row_bool(row: Mapping[str, Any], key: str) -> bool:
    value = row[key]
    return value if isinstance(value, bool) else _coerce_bool(value)


def summarize_arms(
    rows: Sequence[Mapping[str, Any]], cfg: ComparisonConfig, generous_budget: int
) -> List[Dict[str, Any]]:
    groups: DefaultDict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["budget"]) == int(generous_budget):
            groups[(str(row["suite"]), str(row["arm"]))].append(row)
    result: List[Dict[str, Any]] = []
    for (suite, arm), group in sorted(groups.items()):
        solved = [row for row in group if _row_bool(row, "found") and _row_bool(row, "path_valid")]
        expansions = np.asarray([float(row["expansions"]) for row in solved])
        costs = np.asarray([float(row["cost_ratio_eval_only"]) for row in solved])
        ci_low, ci_high = _bootstrap_mean_ci(
            expansions,
            cfg.bootstrap_replicates,
            cfg.bootstrap_seed + C13.parse_csv(cfg.suites).index(suite) * 1000 + len(arm),
        )
        result.append(
            {
                "suite": suite,
                "arm": arm,
                "worlds": int(len(group)),
                "solved_valid": int(len(solved)),
                "invalid_found_paths": int(
                    sum(_row_bool(row, "found") and not _row_bool(row, "path_valid") for row in group)
                ),
                "expansions_mean": float(np.mean(expansions)) if len(expansions) else float("nan"),
                "expansions_median": float(np.median(expansions)) if len(expansions) else float("nan"),
                "expansions_mean_ci95_low": ci_low,
                "expansions_mean_ci95_high": ci_high,
                "cost_ratio_mean_eval_only": float(np.mean(costs)) if len(costs) else float("nan"),
                "cost_ratio_max_eval_only": float(np.max(costs)) if len(costs) else float("nan"),
                "representation_seconds_mean": float(
                    np.mean([float(row["representation_seconds"]) for row in group])
                ),
                "model_seconds_mean": float(
                    np.mean([float(row["model_seconds"]) for row in group])
                ),
                "search_seconds_mean": float(
                    np.mean([float(row["search_seconds"]) for row in group])
                ),
                "bound_violations": int(sum(_row_bool(row, "bound_violation_eval_only") for row in group)),
                "certificate_violations": int(sum(_row_bool(row, "certificate_violation") for row in group)),
            }
        )
    return result


def pairwise_comparisons(
    rows: Sequence[Mapping[str, Any]], cfg: ComparisonConfig, generous_budget: int
) -> List[Dict[str, Any]]:
    lookup: Dict[Tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        if int(row["budget"]) == int(generous_budget):
            lookup[(str(row["suite"]), int(row["world_index"]), str(row["arm"]))] = row
    comparators = list(MAP_FIELD_ARMS) + list(MAP_SCALAR_ARMS) + ["euclid"]
    result: List[Dict[str, Any]] = []
    suites = C13.parse_csv(cfg.suites)
    for current in CURRENT_ARMS:
        for comparator in comparators:
            for label, group_suites in [(suite, [suite]) for suite in suites] + [("POOLED", suites)]:
                pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
                for suite in group_suites:
                    for world_index in range(int(cfg.worlds)):
                        left = lookup.get((suite, world_index, current))
                        right = lookup.get((suite, world_index, comparator))
                        if left is None or right is None:
                            continue
                        if (
                            _row_bool(left, "found")
                            and _row_bool(left, "path_valid")
                            and _row_bool(right, "found")
                            and _row_bool(right, "path_valid")
                        ):
                            pairs.append((left, right))
                delta = np.asarray(
                    [float(left["expansions"]) - float(right["expansions"]) for left, right in pairs],
                    dtype=np.float64,
                )
                cost_delta = np.asarray(
                    [
                        float(left["cost_ratio_eval_only"])
                        - float(right["cost_ratio_eval_only"])
                        for left, right in pairs
                    ],
                    dtype=np.float64,
                )
                seed = (
                    cfg.bootstrap_seed
                    + len(current) * 10_000
                    + len(comparator) * 100
                    + (suites.index(label) if label in suites else 99)
                )
                low, high = _bootstrap_mean_ci(delta, cfg.bootstrap_replicates, seed)
                result.append(
                    {
                        "scope": label,
                        "current_arm": current,
                        "comparator": comparator,
                        "paired_valid": int(len(pairs)),
                        "delta_expansions_mean": float(np.mean(delta)) if len(delta) else float("nan"),
                        "delta_expansions_ci95_low": low,
                        "delta_expansions_ci95_high": high,
                        "wins": int(np.sum(delta < 0.0)),
                        "ties": int(np.sum(delta == 0.0)),
                        "losses": int(np.sum(delta > 0.0)),
                        "current_expansions_mean": float(
                            np.mean([float(left["expansions"]) for left, _ in pairs])
                        ) if pairs else float("nan"),
                        "comparator_expansions_mean": float(
                            np.mean([float(right["expansions"]) for _, right in pairs])
                        ) if pairs else float("nan"),
                        "current_cost_ratio_mean": float(
                            np.mean([float(left["cost_ratio_eval_only"]) for left, _ in pairs])
                        ) if pairs else float("nan"),
                        "comparator_cost_ratio_mean": float(
                            np.mean([float(right["cost_ratio_eval_only"]) for _, right in pairs])
                        ) if pairs else float("nan"),
                        "cost_ratio_delta_mean": float(np.mean(cost_delta)) if len(cost_delta) else float("nan"),
                    }
                )
    return result


def _find_pair(
    pairs: Sequence[Mapping[str, Any]], scope: str, current: str, comparator: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in pairs
        if row["scope"] == scope
        and row["current_arm"] == current
        and row["comparator"] == comparator
    ]
    if len(matches) != 1:
        raise RuntimeError(f"missing pairwise row: {scope}/{current}/{comparator}")
    return matches[0]


def build_verdict(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
    cfg: ComparisonConfig,
    generous_budget: int,
) -> Dict[str, Any]:
    primary_rows = [
        row
        for row in rows
        if int(row["budget"]) == int(generous_budget) and row["arm"] == CURRENT_PRIMARY
    ]
    field_rows = [
        row
        for row in rows
        if int(row["budget"]) == int(generous_budget) and row["arm"] == PRIMARY_COMPARATOR
    ]
    bounded_rows = [
        row
        for row in rows
        if int(row["budget"]) == int(generous_budget) and row["arm"] == CURRENT_BOUNDED
    ]
    pooled = _find_pair(pairs, "POOLED", CURRENT_PRIMARY, PRIMARY_COMPARATOR)
    suite_pairs = [
        _find_pair(pairs, suite, CURRENT_PRIMARY, PRIMARY_COMPARATOR)
        for suite in C13.parse_csv(cfg.suites)
    ]
    primary_solved = sum(
        _row_bool(row, "found") and _row_bool(row, "path_valid") for row in primary_rows
    )
    primary_max_cost = max(
        float(row["cost_ratio_eval_only"])
        for row in primary_rows
        if _row_bool(row, "found") and _row_bool(row, "path_valid")
    )
    primary_mean_cost = float(
        np.mean([float(row["cost_ratio_eval_only"]) for row in primary_rows])
    )
    field_mean_cost = float(
        np.mean([float(row["cost_ratio_eval_only"]) for row in field_rows])
    )
    negative_suites = int(
        sum(float(row["delta_expansions_mean"]) < 0.0 for row in suite_pairs)
    )
    conditions = {
        "primary_valid_paths_144_of_144": int(primary_solved) == 144,
        "primary_max_cost_ratio_at_most_1_10": primary_max_cost <= 1.10 + 1.0e-12,
        "pooled_expansion_ci_upper_below_zero": float(
            pooled["delta_expansions_ci95_high"]
        ) < 0.0,
        "negative_mean_delta_at_least_four_suites": negative_suites >= 4,
        "primary_mean_cost_within_0_01_of_field_hrm": primary_mean_cost
        <= field_mean_cost + 0.01 + 1.0e-12,
    }
    bounded_conditions = {
        "valid_paths_144_of_144": sum(
            _row_bool(row, "found") and _row_bool(row, "path_valid") for row in bounded_rows
        ) == 144,
        "bound_violations_zero": sum(
            _row_bool(row, "bound_violation_eval_only") for row in bounded_rows
        ) == 0,
        "certificate_violations_zero": sum(
            _row_bool(row, "certificate_violation") for row in bounded_rows
        ) == 0,
    }
    parity_failures = int(sum(not _row_bool(row, "parity_pass") for row in parity_rows))
    gate_pass = all(conditions.values()) and parity_failures == 0
    return {
        "verdict": (
            "current_state_solid_improvement_over_field_hrm"
            if gate_pass
            else "current_state_does_not_pass_preregistered_c7_improvement_gate"
        ),
        "gate_pass": bool(gate_pass),
        "primary_arm": CURRENT_PRIMARY,
        "primary_comparator": PRIMARY_COMPARATOR,
        "conditions": conditions,
        "negative_suites": negative_suites,
        "primary_solved_valid": int(primary_solved),
        "primary_max_cost_ratio_eval_only": primary_max_cost,
        "primary_mean_cost_ratio_eval_only": primary_mean_cost,
        "field_hrm_mean_cost_ratio_eval_only": field_mean_cost,
        "pooled_primary_pair": dict(pooled),
        "suite_primary_pairs": [dict(row) for row in suite_pairs],
        "bounded_control": {
            "pass": bool(all(bounded_conditions.values())),
            "conditions": bounded_conditions,
        },
        "historical_parity_failures": parity_failures,
        "authorization": (
            "document_c13i_as_confirmed_current_state_improvement"
            if gate_pass
            else "train_new_multisuite_current_state_model_then_use_new_seed_block"
        ),
    }


def write_report(
    path: Path,
    summaries: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    verdict: Mapping[str, Any],
    cfg: ComparisonConfig,
) -> Path:
    lines = [
        "# C13-I current-state vs C7 map-derived result",
        "",
        f"**Verdict:** `{verdict['verdict']}` (`gate_pass={str(verdict['gate_pass']).lower()}`).",
        "",
        "The current-state arms receive only current/goal geometry, radius-bounded rays,",
        "and one-hop actions at runtime. `field_*` receives the complete 64 x 64 map raster;",
        "`scalar_*` receives global obstacle-list-derived summaries and sectors.",
        "",
        "## Generous-budget results (384 expansions)",
        "",
        "| Suite | Arm | Valid | Mean exp. | 95% CI | Mean cost ratio | Max cost ratio |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    wanted = set(CURRENT_ARMS) | {"field_hrm", "scalar_hrm", "euclid", "oracle"}
    for row in summaries:
        if row["arm"] not in wanted:
            continue
        lines.append(
            f"|{row['suite']}|{row['arm']}|{row['solved_valid']}/{row['worlds']}|"
            f"{float(row['expansions_mean']):.2f}|"
            f"[{float(row['expansions_mean_ci95_low']):.2f}, {float(row['expansions_mean_ci95_high']):.2f}]|"
            f"{float(row['cost_ratio_mean_eval_only']):.4f}|"
            f"{float(row['cost_ratio_max_eval_only']):.4f}|"
        )
    lines += [
        "",
        "## Primary paired comparison",
        "",
        "| Scope | Paired | Delta exp. current-field_hrm | 95% CI | W/T/L | Cost-ratio delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    primary_pairs = [
        row
        for row in pairs
        if row["current_arm"] == CURRENT_PRIMARY and row["comparator"] == PRIMARY_COMPARATOR
    ]
    order = {suite: idx for idx, suite in enumerate(C13.parse_csv(cfg.suites))}
    order["POOLED"] = 999
    for row in sorted(primary_pairs, key=lambda item: order[item["scope"]]):
        lines.append(
            f"|{row['scope']}|{row['paired_valid']}|{float(row['delta_expansions_mean']):.3f}|"
            f"[{float(row['delta_expansions_ci95_low']):.3f}, {float(row['delta_expansions_ci95_high']):.3f}]|"
            f"{row['wins']}/{row['ties']}/{row['losses']}|{float(row['cost_ratio_delta_mean']):.4f}|"
        )
    lines += [
        "",
        "## Frozen gate",
        "",
    ]
    for name, passed in verdict["conditions"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines += [
        "",
        "The bounded reopening-FOCAL control is reported separately because its safety",
        "certificate and its expansion efficiency answer different questions.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_analyze(cfg: ComparisonConfig) -> Dict[str, Any]:
    suites = C13.parse_csv(cfg.suites)
    all_rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    parity: List[Dict[str, Any]] = []
    shard_paths: Dict[str, Path] = {}
    for suite in suites:
        paths = _suite_paths(Path(cfg.out_dir), suite)
        for key in ("raw", "diagnostics", "parity", "meta"):
            if not paths[key].exists():
                raise RuntimeError(f"missing {suite} shard artifact: {paths[key]}")
            shard_paths[f"{suite}_{key}"] = paths[key]
        all_rows.extend(_read_csv(paths["raw"]))
        diagnostics.extend(_read_csv(paths["diagnostics"]))
        parity.extend(_read_csv(paths["parity"]))

    budgets_by_suite = calibrated_budgets(cfg)
    arm_names = sorted({str(row["arm"]) for row in all_rows})
    expected_rows = sum(
        int(cfg.worlds) * len(budgets_by_suite[suite]) * len(arm_names)
        for suite in suites
    )
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"raw row count {len(all_rows)} != expected {expected_rows}")
    generous = int(math.ceil(cfg.generous_budget_factor * cfg.roadmap_nodes))
    summaries = summarize_arms(all_rows, cfg, generous)
    pairs = pairwise_comparisons(all_rows, cfg, generous)
    verdict = build_verdict(all_rows, pairs, parity, cfg, generous)

    result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
    raw_path = C13.write_csv(result_dir / "c13i_raw.csv", all_rows)
    diagnostic_path = C13.write_csv(result_dir / "provider_diagnostics.csv", diagnostics)
    parity_path = C13.write_csv(result_dir / "historical_parity.csv", parity)
    summary_path = C13.write_csv(result_dir / "arm_summary.csv", summaries)
    pairs_path = C13.write_csv(result_dir / "pairwise_summary.csv", pairs)
    verdict_path = C13.write_json(result_dir / "gate_verdict.json", verdict)
    report_path = write_report(
        Path(cfg.out_dir) / "C13I_RESULTS.md", summaries, pairs, verdict, cfg
    )

    seed_pairs = {
        (str(row["suite"]), int(row["world_index"]), int(row["world_seed"]))
        for row in all_rows
    }
    path_invalid = int(
        sum(_row_bool(row, "found") and not _row_bool(row, "path_valid") for row in all_rows)
    )
    parity_failures = int(sum(not _row_bool(row, "parity_pass") for row in parity))
    selection = H._read_json(result_dir / "operating_point_selection.json")
    verification = {
        "rows": int(len(all_rows)),
        "expected_rows": int(expected_rows),
        "suites": suites,
        "worlds_per_suite": int(cfg.worlds),
        "unique_suite_world_seed_records": int(len(seed_pairs)),
        "expected_unique_suite_world_seed_records": int(len(suites) * cfg.worlds),
        "arms": arm_names,
        "path_invalid_found_rows": path_invalid,
        "historical_parity_checks": int(len(parity)),
        "historical_parity_failures": parity_failures,
        "bounded_bound_violations": int(
            sum(
                row["arm"] == CURRENT_BOUNDED
                and int(row["budget"]) == generous
                and _row_bool(row, "bound_violation_eval_only")
                for row in all_rows
            )
        ),
        "bounded_certificate_violations": int(
            sum(
                row["arm"] == CURRENT_BOUNDED
                and int(row["budget"]) == generous
                and _row_bool(row, "certificate_violation")
                for row in all_rows
            )
        ),
        "selection_fixed_before_c7": bool(
            selection["frozen_before_c7_current_model_evaluation"]
        ),
        "shortest_path_target": False,
        "training_target_reads_dist_to_goal": False,
        "runtime_information": {
            "current": "current_goal_geometry_bounded_rays_one_hop_actions",
            "field": "complete_64x64_occupancy_goal_raster",
            "scalar": "global_obstacle_list_summaries_sectors_plus_rays_goal",
        },
        "integrity_pass": bool(
            len(all_rows) == expected_rows
            and len(seed_pairs) == len(suites) * cfg.worlds
            and path_invalid == 0
            and parity_failures == 0
        ),
    }
    verification_path = C13.write_json(result_dir / "verification.json", verification)
    manifest = {
        "experiment": "C13-I current-state LHBL versus live C7 map-derived heuristics",
        "config": asdict(cfg),
        "selection": str(result_dir / "operating_point_selection.json"),
        "preregistration": cfg.preregistration,
        "primary_arm": CURRENT_PRIMARY,
        "primary_comparator": PRIMARY_COMPARATOR,
        "full_map_runtime_input_for_current": False,
        "outputs": {
            "raw": str(raw_path),
            "diagnostics": str(diagnostic_path),
            "parity": str(parity_path),
            "summary": str(summary_path),
            "pairwise": str(pairs_path),
            "verdict": str(verdict_path),
            "verification": str(verification_path),
            "report": str(report_path),
        },
    }
    manifest_path = C13.write_json(Path(cfg.out_dir) / "manifest.json", manifest)

    c7_inputs = {
        "c7_calibration": Path(cfg.c7_run_dir) / "calibration.json",
        "c7_historical_raw": Path(cfg.c7_run_dir)
        / "results"
        / "continuous_prm_c7_eval_raw.csv",
        **{
            f"c7_checkpoint_{path.stem}": path
            for path in sorted((Path(cfg.c7_run_dir) / "checkpoints").glob("*.pt"))
        },
    }
    inputs = {
        "implementation": Path(__file__).resolve(),
        "preregistration": Path(cfg.preregistration),
        "candidate_raw": Path(cfg.candidate_study_dir) / "results" / "candidate_raw.csv",
        "checkpoint_04": Path(cfg.source_run_dir)
        / "checkpoints"
        / "flat_mlp_iteration_04.pt",
        "checkpoint_08": Path(cfg.source_run_dir)
        / "checkpoints"
        / "flat_mlp_iteration_08.pt",
        **c7_inputs,
        **shard_paths,
    }
    outputs = {
        "raw": raw_path,
        "diagnostics": diagnostic_path,
        "parity": parity_path,
        "summary": summary_path,
        "pairwise": pairs_path,
        "verdict": verdict_path,
        "verification": verification_path,
        "report": report_path,
        "manifest": manifest_path,
        "selection": result_dir / "operating_point_selection.json",
    }
    integrity = {
        "inputs": {
            name: {"path": str(path), "sha256": S.file_sha256(path)}
            for name, path in inputs.items()
        },
        "outputs": {
            name: {"path": str(path), "sha256": S.file_sha256(path)}
            for name, path in outputs.items()
        },
    }
    integrity_path = C13.write_json(Path(cfg.out_dir) / "integrity.json", integrity)
    if not verification["integrity_pass"]:
        raise RuntimeError("C13-I verification failed; preserved artifacts for diagnosis")
    print(
        f"[c13i] {verdict['verdict']} gate_pass={verdict['gate_pass']} "
        f"parity={len(parity) - parity_failures}/{len(parity)} -> {report_path}",
        flush=True,
    )
    return {
        "verdict": verdict,
        "verification": verification,
        "integrity": integrity_path,
        "report": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C13-I current-state vs C7 comparison")
    parser.add_argument("--mode", choices=("select", "eval", "analyze", "full"), default="full")
    parser.add_argument("--only-suites", type=str, default="")
    parser.add_argument("--out-dir", type=str, default=ComparisonConfig.out_dir)
    parser.add_argument("--device", type=str, default=ComparisonConfig.device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ComparisonConfig(out_dir=args.out_dir, device=args.device)
    resolve_paths(cfg)
    all_suites = list(ALL_SUITES)
    selected_suites = C13.parse_csv(args.only_suites) if args.only_suites else all_suites
    unknown = sorted(set(selected_suites) - set(all_suites))
    if unknown:
        raise ValueError(f"unknown C7 suites: {unknown}")
    if args.mode == "select":
        result_dir = C13.ensure_dir(Path(cfg.out_dir) / "results")
        path = C13.write_json(
            result_dir / "operating_point_selection.json", build_selection_report(cfg)
        )
        print(f"[c13i] frozen development-only operating points -> {path}", flush=True)
        return
    if args.mode in {"eval", "full"}:
        run_eval(cfg, selected_suites)
    if args.mode in {"analyze", "full"}:
        if selected_suites != all_suites:
            raise ValueError("analysis is locked to all six suites")
        run_analyze(cfg)


if __name__ == "__main__":
    main()
