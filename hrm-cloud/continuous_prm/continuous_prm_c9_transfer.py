#!/usr/bin/env python3
"""C9a: few-shot transfer learning for learned PRM heuristics.

Adapt the C7 pooled scalar base (avgbase) to held-out hard families from K worlds,
comparing zero-shot / LoRA / full fine-tune / from-scratch on HRM + ON-LSTM, and
report adaptation curves. New-file-only; reuses C7/C3 machinery. See
docs/superpowers/specs/2026-06-29-c9-transfer-design.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import numpy as np

import continuous_prm_common as C
import continuous_prm_providers as P
import continuous_prm_c7_integration_compare as C7
import continuous_prm_c7_hard_maps as H7


def now_str() -> str:
    return C.now_str()


@dataclass
class C9Config:
    source_dir: str = "hrm-cloud/continuous_prm/runs/c7_local"
    out_dir: str = "hrm-cloud/continuous_prm/runs/c9_local"
    targets: str = "C_hard_maze_dense,C_hard_bugtrap,C_hard_rooms_large"
    backbones: str = "hrm,onlstm"
    k_grid: str = "0,1,2,4,8,16,32"
    n_adapt_seeds: int = 5
    n_test: int = 30
    alpha: float = 1.0
    adapt_epochs: int = 0
    roadmap_nodes: int = 192
    roadmap_k: int = 7
    budgets: str = ""
    w_values: str = "1.0,1.1"
    seed: int = 1234
    scale: str = "local"
    mode: str = "full"
    cpu: bool = False


@dataclass
class SourceBase:
    model: object
    backbone_cfg: object
    feature_cfg: object
    train_cfg: object
    ckpt_path: Path
    backbone: str


def load_source_base(source_dir, backbone: str, device) -> SourceBase:
    """Load the C7 avgbase checkpoint + its configs for `backbone`."""
    import torch
    ckpt = Path(source_dir) / "checkpoints" / f"avgbase__{backbone}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"source base not found: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu")
    backbone_cfg = C.BackboneConfig(**payload["backbone_cfg"])
    feature_cfg = C.FeatureConfig(**payload["feature_cfg"])
    train_cfg = C.TrainingConfig(**payload["train_cfg"])
    model = C.load_base_model(backbone_cfg, feature_cfg, train_cfg, ckpt, device)
    return SourceBase(model, backbone_cfg, feature_cfg, train_cfg, ckpt, backbone)


# ---------------------------------------------------------------------------
# Task 2 — ADAPT/TEST world-split helpers
# ---------------------------------------------------------------------------

def world_fingerprint(world) -> tuple:
    """Stable identity of a world (start, goal, obstacle centers)."""
    start = tuple(np.round(np.asarray(world.start, dtype=np.float64), 6))
    goal = tuple(np.round(np.asarray(world.goal, dtype=np.float64), 6))
    obs = tuple(sorted((round(float(getattr(o, "cx", 0.0)), 6), round(float(getattr(o, "cy", 0.0)), 6)) for o in world.obstacles))
    return (round(float(world.side_len), 6), start, goal, obs)


def iter_test_worlds(spec, suite_idx: int, cfg: C9Config, roadmap_cfg, n_test: int):
    """TEST worlds for a target = the C7 deterministic eval worlds (yields (idx, world, rm))."""
    c7cfg = C7.C7Config(seed=int(cfg.seed), roadmap_nodes=cfg.roadmap_nodes, roadmap_k=cfg.roadmap_k)
    yield from C7.iter_matched_worlds(spec, suite_idx, c7cfg, roadmap_cfg, n_test)


def adapt_seed(target: str, K: int, adapt_seed_idx: int, base_seed: int) -> int:
    """Deterministic (cross-process stable), well-separated seed for an
    ADAPT(target, K, adapt_seed_idx) collection. Cross-process stability matters
    because collect_task_dataset caches by file path: a salted hash would let a
    resume reuse an npz built under a different seed."""
    h = int(hashlib.md5(target.encode()).hexdigest()[:4], 16)
    # 100_000*h % 7_000_000 is a coarse per-target offset; the actual separation
    # between collections comes from the 1_009*K + idx terms.
    return int(base_seed) + 5_000_000 + 100_000 * h % 7_000_000 + 1_009 * int(K) + int(adapt_seed_idx)


def adapt_world_fingerprints(spec, n_worlds, nodes_per_world, roadmap_cfg, feature_cfg, seed):
    """Replays collect_task_dataset's world-generation loop to expose ADAPT world fingerprints
    (for the disjointness test). Mirrors C.collect_task_dataset world acceptance rules."""
    import random as _random
    rng = _random.Random(int(seed))
    fps, done, attempts = [], 0, 0
    while done < n_worlds and attempts < n_worlds * 100:
        attempts += 1
        w_seed = rng.randint(0, 2**31 - 1)
        world = C.build_world(spec, w_seed, roadmap_cfg.min_start_goal_dist_frac)
        if world is None:
            continue
        rm = C.build_prm(world, roadmap_cfg, seed=w_seed + 17)
        if rm is None or not rm.connected_to_goal[0]:
            continue
        connected_idxs = np.where(rm.connected_to_goal)[0]
        connected_idxs = connected_idxs[connected_idxs != 1]
        if len(connected_idxs) < max(12, nodes_per_world // 4):
            continue
        # mirror collect_task_dataset's rng.sample so replayed worlds match the real ADAPT collection
        if nodes_per_world > 1:
            rng.sample([int(i) for i in connected_idxs], k=min(nodes_per_world - 1, len(connected_idxs)))
        # (collect_task_dataset's later finite-residual gate cannot drop connected
        # nodes, so it draws no rng and need not be replicated here.)
        fps.append(world_fingerprint(world))
        done += 1
    return fps


# ---------------------------------------------------------------------------
# Task 3 — per-(K, seed) scalar model training
# ---------------------------------------------------------------------------

def train_scalar_model(backbone_cfg, dataset_npz, out_ckpt, feature_cfg,
                       train_cfg, device, seed: int, init_ckpt=None):
    """Train an additive-residual scalar model on a single npz dataset.
    init_ckpt=None => from-scratch; a path => full fine-tune from those weights.
    Mirrors C.train_avgbase's optimizer/loss/loop; writes to out_ckpt (no skip guard)."""
    out_ckpt = Path(out_ckpt)
    x, y = C.load_npz_arrays(dataset_npz)
    ds = C.ArrayDataset(x, y)
    loader = C.make_loader(ds, train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)
    model = C.build_model(backbone_cfg, feature_cfg, train_cfg, device)
    if init_ckpt is not None:
        C.safe_load_state(model, Path(init_ckpt))
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    C.set_global_seed(int(seed))
    history = []
    for epoch in range(1, train_cfg.base_epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            if not torch.isfinite(pred).all():
                raise RuntimeError("nonfinite c9 predictions")
            loss = F.smooth_l1_loss(pred, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite c9 loss")
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch, "loss": C.finite_mean(losses), "seconds": time.time() - t0})
    payload = {
        "model": model.state_dict(),
        "backbone_cfg": asdict(backbone_cfg),
        "feature_cfg": asdict(feature_cfg),
        "train_cfg": asdict(train_cfg),
        "init_ckpt": (str(init_ckpt) if init_ckpt is not None else None),
        "history": history,
    }
    C.ensure_dir(out_ckpt.parent)
    torch.save(payload, out_ckpt)
    return out_ckpt


# ---------------------------------------------------------------------------
# Task 4 — load any scalar checkpoint into a ScalarResidualProvider
# ---------------------------------------------------------------------------

def load_scalar_provider(ckpt, device):
    """Load a scalar checkpoint (avgbase / full-FT / scratch / LoRA-expert) into a ScalarResidualProvider.
    For LoRA experts, apply_lora must run before load_state_dict (the state_dict carries LoRA params)."""
    payload = torch.load(Path(ckpt), map_location="cpu")
    backbone_cfg = C.BackboneConfig(**payload["backbone_cfg"])
    feature_cfg = C.FeatureConfig(**payload["feature_cfg"])
    train_cfg = C.TrainingConfig(**payload["train_cfg"])
    model = C.build_model(backbone_cfg, feature_cfg, train_cfg, device)
    if "lora_rank" in payload and "alpha" in payload:
        C.apply_lora(model, rank=int(payload["lora_rank"]), alpha=float(payload["alpha"]), init_scale=0.01)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return P.ScalarResidualProvider(model, feature_cfg, device, backbone_cfg.name, train_cfg.max_norm_residual)


# ---------------------------------------------------------------------------
# Task 5 — adapt mode (lora / full_ft / scratch per target x K x seed x backbone)
# ---------------------------------------------------------------------------

SCALAR_NODES_PER_WORLD = C7.SCALAR_NODES_PER_WORLD  # match how the avgbase base was trained


def _parse_csv(s): return [t for t in str(s).split(",") if t != ""]
def _parse_ints(s): return [int(t) for t in _parse_csv(s)]


def arm_ckpt_path(out_dir, target: str, K: int, seed: int, method: str, backbone: str) -> Path:
    return Path(out_dir) / "checkpoints" / f"c9__{target}__K{K}__s{seed}__{method}__{backbone}.pt"


def run_adapt(cfg: C9Config, device) -> dict:
    H7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    out_dir = Path(cfg.out_dir)
    ds_dir = out_dir / "datasets"
    C.ensure_dir(out_dir / "checkpoints"); C.ensure_dir(ds_dir)
    rmcfg = C.RoadmapConfig(n_nodes=cfg.roadmap_nodes, k_neighbors=cfg.roadmap_k)
    Ks = [k for k in _parse_ints(cfg.k_grid) if k > 0]
    arms = []
    for target in _parse_csv(cfg.targets):
        spec = specs[target]
        for backbone in _parse_csv(cfg.backbones):
            base = load_source_base(Path(cfg.source_dir), backbone, device)
            tcfg = base.train_cfg if cfg.adapt_epochs <= 0 else dataclasses.replace(base.train_cfg, base_epochs=int(cfg.adapt_epochs))
            for K in Ks:
                for s in range(int(cfg.n_adapt_seeds)):
                    aseed = adapt_seed(target, K, s, cfg.seed)
                    npz = C.collect_task_dataset(
                        spec, ds_dir, f"adapt_{target}_K{K}_s{s}", K, SCALAR_NODES_PER_WORLD,
                        rmcfg, base.feature_cfg, seed=aseed)
                    lora_ck = C.train_expert(
                        base.backbone_cfg, f"{target}_K{K}_s{s}", npz, base.ckpt_path,
                        out_dir, base.feature_cfg, tcfg, device, seed=aseed, alpha=float(cfg.alpha))
                    arms.append(dict(target=target, K=K, seed=s, method="lora", backbone=backbone, ckpt=str(lora_ck)))
                    for method, init in (("full_ft", base.ckpt_path), ("scratch", None)):
                        ck = arm_ckpt_path(out_dir, target, K, s, method, backbone)
                        if not ck.exists():
                            train_scalar_model(base.backbone_cfg, npz, ck, base.feature_cfg, tcfg, device, seed=aseed, init_ckpt=init)
                        arms.append(dict(target=target, K=K, seed=s, method=method, backbone=backbone, ckpt=str(ck)))
                    print(f"[{now_str()}] c9 adapt: {target} {backbone} K={K} s={s} done", flush=True)
    manifest = {"arms": arms, "targets": _parse_csv(cfg.targets), "backbones": _parse_csv(cfg.backbones),
                "k_grid": _parse_ints(cfg.k_grid), "n_adapt_seeds": int(cfg.n_adapt_seeds), "seed": int(cfg.seed)}
    C.write_json(out_dir / "adapt_manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Task 6 — eval mode (run TEST worlds, per-arm providers, raw CSV output)
# ---------------------------------------------------------------------------

import csv as _csv

RAW_COLS = ["target", "K", "seed", "method", "backbone", "suite", "world_index",
            "provider", "mode", "w", "budget", "found", "expansions", "closed",
            "cost", "optimal", "suboptimality", "nonfinite"]


def _target_budgets(cfg: C9Config, target: str) -> List[int]:
    if str(cfg.budgets).strip():
        return _parse_ints(cfg.budgets)
    calib = Path(cfg.source_dir) / "calibration.json"
    if calib.exists():
        import json
        b = (json.loads(calib.read_text()).get("budgets", {}) or {}).get(target)
        if b:
            return [int(x) for x in b]
    raise RuntimeError(f"no budgets for {target}: pass --budgets or provide {calib}")


def run_eval(cfg: C9Config, device) -> Path:
    H7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    out_dir = Path(cfg.out_dir)
    res_dir = out_dir / "results"; shard_dir = res_dir / "_shards"
    C.ensure_dir(shard_dir)
    rmcfg = C.RoadmapConfig(n_nodes=cfg.roadmap_nodes, k_neighbors=cfg.roadmap_k)
    w_values = [float(x) for x in _parse_csv(cfg.w_values)]
    import json
    manifest = json.loads((out_dir / "adapt_manifest.json").read_text())
    all_rows: List[dict] = []
    for suite_idx, target in enumerate(_parse_csv(cfg.targets)):
        spec = specs[target]
        budgets = _target_budgets(cfg, target)
        providers: Dict[str, object] = {"euclid": P.EuclidProvider(), "oracle": P.OracleProvider()}
        meta: Dict[str, dict] = {"euclid": {}, "oracle": {}}
        for backbone in _parse_csv(cfg.backbones):
            base = load_source_base(Path(cfg.source_dir), backbone, device)
            zp = P.ScalarResidualProvider(base.model, base.feature_cfg, device, backbone, base.train_cfg.max_norm_residual)
            zp.name = f"zeroshot_{backbone}"
            providers[zp.name] = zp
            meta[zp.name] = dict(K=0, seed=-1, method="zero_shot", backbone=backbone)
        for a in manifest["arms"]:
            if a["target"] != target:
                continue
            prov = load_scalar_provider(Path(a["ckpt"]), device)
            key = f'{a["method"]}_{a["backbone"]}_K{a["K"]}_s{a["seed"]}'
            prov.name = key
            providers[key] = prov
            meta[key] = dict(K=a["K"], seed=a["seed"], method=a["method"], backbone=a["backbone"])
        rows = []
        for world_index, world, rm in iter_test_worlds(spec, suite_idx, cfg, rmcfg, cfg.n_test):
            recs = P.run_world_arms(world, rm, providers, budgets, w_values, goal_idx=1)
            for r in recs:
                m = meta.get(r["provider"], dict(K=-1, seed=-1, method=r["provider"], backbone=""))
                r.update(dict(target=target, suite=target, world_index=world_index, **m))
                rows.append(r)
        sp = shard_dir / f"{target}.csv"
        with open(sp, "w", newline="") as f:
            wri = _csv.DictWriter(f, fieldnames=RAW_COLS); wri.writeheader()
            for r in rows:
                wri.writerow({k: r.get(k, "") for k in RAW_COLS})
        all_rows.extend(rows)
        print(f"[{now_str()}] c9 eval {target}: {len(rows)} rows", flush=True)
    raw = res_dir / "continuous_prm_c9_eval_raw.csv"
    with open(raw, "w", newline="") as f:
        wri = _csv.DictWriter(f, fieldnames=RAW_COLS); wri.writeheader()
        for r in all_rows:
            wri.writerow({k: r.get(k, "") for k in RAW_COLS})
    print(f"[{now_str()}] c9 eval: merged {len(all_rows)} rows -> {raw}", flush=True)
    return raw


# ---------------------------------------------------------------------------
# Task 7 — analyze mode (adaptation curves CSV + comparisons + significance MD)
# ---------------------------------------------------------------------------

from continuous_prm_c6_heatmap_value_field import mcnemar_exact_p, bh_q_values
from continuous_prm_c7_integration_compare import wilcoxon_signed_rank_p, bootstrap_median_ci


def _load_rows(raw_csv): return list(_csv.DictReader(open(raw_csv, newline="")))
def _astar(rows): return [r for r in rows if r.get("mode") == "astar"]
def _is_found(r): return str(r.get("found")) in ("True", "1", "true")


def _euclid_exp_by_world(rows, target):
    out = {}
    for r in _astar(rows):
        if r["target"] == target and r["provider"] == "euclid" and _is_found(r):
            out[int(r["world_index"])] = float(r["expansions"])
    return out


def analyze_from_raw(raw_csv, out_dir, seed, targets, backbones):
    rows = _load_rows(raw_csv)
    res_dir = Path(out_dir); C.ensure_dir(res_dir)
    methods = ["zero_shot", "lora", "full_ft", "scratch"]
    curves = []
    for target in targets:
        eu = _euclid_exp_by_world(rows, target)
        for backbone in backbones:
            for method in methods:
                by_K, succ_by_K = {}, {}
                for r in _astar(rows):
                    if r["target"] != target or r["method"] != method:
                        continue
                    if method != "zero_shot" and r.get("backbone") != backbone:
                        continue
                    if method == "zero_shot" and r.get("backbone") not in ("", backbone):
                        continue
                    K = int(r["K"]); wi = int(r["world_index"]); found = _is_found(r)
                    succ_by_K.setdefault(K, []).append(1 if found else 0)
                    if found and wi in eu and eu[wi] > 0:
                        by_K.setdefault(K, []).append(float(r["expansions"]) / eu[wi])
                for K in sorted(set(list(by_K) + list(succ_by_K))):
                    med, lo, hi = bootstrap_median_ci(by_K.get(K, []), seed=seed)
                    succ = float(np.mean(succ_by_K.get(K, []))) if succ_by_K.get(K) else float("nan")
                    curves.append(dict(target=target, backbone=backbone, method=method, K=K,
                                       n_matched=len(by_K.get(K, [])), exp_ratio_median=med,
                                       ci_lo=lo, ci_hi=hi, success=succ))
    curves_csv = res_dir / "continuous_prm_c9_curves.csv"
    cols = ["target", "backbone", "method", "K", "n_matched", "exp_ratio_median", "ci_lo", "ci_hi", "success"]
    with open(curves_csv, "w", newline="") as f:
        wri = _csv.DictWriter(f, fieldnames=cols); wri.writeheader()
        for c in curves:
            wri.writerow({k: ("" if c[k] is None or (isinstance(c[k], float) and not np.isfinite(c[k])) else c[k]) for k in cols})
    comp_md = res_dir / "continuous_prm_c9_comparisons.md"
    _write_comparisons_md(comp_md, curves)
    sig_md = res_dir / "continuous_prm_c9_significance.md"
    _write_significance_md(sig_md, rows, targets, backbones, methods)
    print(f"[{now_str()}] c9 analyze: wrote {curves_csv}, {comp_md}, {sig_md}", flush=True)
    return {"curves": curves_csv, "comparisons": comp_md, "significance": sig_md}


def _write_comparisons_md(path, curves):
    by = {(c["target"], c["backbone"], c["method"], c["K"]): c for c in curves}
    lines = ["# C9 Transfer — Pre-registered Comparisons", "",
             "Adaptation curves: matched A* expansion-ratio vs euclid (median, 95% CI) per K.",
             "lora vs scratch = transfer helps; lora vs full_ft = sample-efficiency; vs K=0 = adaptation helps. Lower = fewer expansions.", ""]
    targets = sorted({c["target"] for c in curves}); backbones = sorted({c["backbone"] for c in curves})
    Ks = sorted({c["K"] for c in curves})
    for target in targets:
        for backbone in backbones:
            lines += [f"## {target} / {backbone}", "|K|zero_shot|lora|full_ft|scratch|", "|---:|---|---|---|---|"]
            for K in Ks:
                cells = []
                for m in ("zero_shot", "lora", "full_ft", "scratch"):
                    c = by.get((target, backbone, m, K))
                    cells.append("n/a" if not c or not np.isfinite(c["exp_ratio_median"])
                                 else f'{c["exp_ratio_median"]:.3f} [{c["ci_lo"]:.3f},{c["ci_hi"]:.3f}] (succ {c["success"]:.2f}, n{c["n_matched"]})')
                lines.append(f"|{K}|" + "|".join(cells) + "|")
            lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_significance_md(path, rows, targets, backbones, methods):
    """McNemar + BH success grid: learned arms vs euclid, per (target, backbone, method, K)."""
    astar_rows = _astar(rows)
    # Build euclid found-by-world index: {target: {wi: bool}}
    eu_found: Dict[str, Dict[int, bool]] = {}
    for r in astar_rows:
        if r["provider"] == "euclid":
            eu_found.setdefault(r["target"], {})[int(r["world_index"])] = _is_found(r)

    comparisons = []
    pvals = []
    for target in targets:
        eu_by_wi = eu_found.get(target, {})
        for backbone in backbones:
            for method in methods:
                # Collect all K values present for this arm
                Ks_present: Dict[int, Dict[int, bool]] = {}  # K -> {wi: found}
                for r in astar_rows:
                    if r["target"] != target or r["method"] != method:
                        continue
                    if method != "zero_shot" and r.get("backbone") != backbone:
                        continue
                    if method == "zero_shot" and r.get("backbone") not in ("", backbone):
                        continue
                    K = int(r["K"]); wi = int(r["world_index"])
                    Ks_present.setdefault(K, {})[wi] = _is_found(r)
                for K, arm_by_wi in sorted(Ks_present.items()):
                    shared = sorted(set(eu_by_wi) & set(arm_by_wi))
                    n = len(shared)
                    if n == 0:
                        continue
                    gain = sum(1 for wi in shared if arm_by_wi[wi] and not eu_by_wi.get(wi, False))
                    loss = sum(1 for wi in shared if eu_by_wi.get(wi, False) and not arm_by_wi[wi])
                    eu_succ = float(np.mean([1.0 if eu_by_wi.get(wi, False) else 0.0 for wi in shared]))
                    arm_succ = float(np.mean([1.0 if arm_by_wi[wi] else 0.0 for wi in shared]))
                    p = mcnemar_exact_p(gain, loss)
                    pvals.append(p)
                    comparisons.append(dict(target=target, backbone=backbone, method=method, K=K,
                                            n=n, euclid_succ=eu_succ, arm_succ=arm_succ,
                                            gain=gain, loss=loss, mcnemar_p=p))
    qvals = bh_q_values(pvals)
    for row, q in zip(comparisons, qvals):
        row["bh_q"] = q

    def _fmt_p(p, disc):
        if disc < 2:
            return "n/a"
        return f"{p:.4f}" if p is not None and np.isfinite(p) else "n/a"

    def _fmt_num(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:.3f}"

    lines = [
        "# C9 Transfer — Significance",
        "",
        "McNemar exact test (learned arm found & euclid not = gain; euclid found & arm not = loss).",
        "BH correction applied across all comparisons in this table.",
        "n/a when fewer than 2 discordant pairs.",
        "",
        "|target|backbone|method|K|n|euclid_succ|arm_succ|gain|loss|McNemar p|BH q|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not comparisons:
        lines.append("|<none>|-|-|-|-|-|-|-|-|-|-|")
    for r in comparisons:
        disc = r["gain"] + r["loss"]
        lines.append(
            f"|{r['target']}|{r['backbone']}|{r['method']}|{r['K']}|{r['n']}|"
            f"{_fmt_num(r['euclid_succ'])}|{_fmt_num(r['arm_succ'])}|{r['gain']}|{r['loss']}|"
            f"{_fmt_p(r['mcnemar_p'], disc)}|{_fmt_num(r.get('bh_q'))}|"
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_analyze(cfg: C9Config) -> dict:
    raw = Path(cfg.out_dir) / "results" / "continuous_prm_c9_eval_raw.csv"
    return analyze_from_raw(raw, Path(cfg.out_dir) / "results", seed=int(cfg.seed),
                            targets=_parse_csv(cfg.targets), backbones=_parse_csv(cfg.backbones))
