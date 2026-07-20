#!/usr/bin/env python3
"""C10: parameter-space LoRA interpolation for zero-shot transfer to interior held-out
families. New-file-only; reuses C9/C9h + C5/C7 generators + common.
See docs/experiments/continuous/c10/design/2026-06-29-c10-interp-design.md.
"""
from __future__ import annotations
import argparse, dataclasses, csv as _csv, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import continuous_prm_common as C
import continuous_prm_providers as P
import continuous_prm_c5_hard_obstacle_encoder as C5
import continuous_prm_c7_hard_maps as H7
import continuous_prm_c9_transfer as C9
import continuous_prm_c9h_transfer as C9H

HARD_MODE = C5.HARD_MODE

SOURCE_FAMILIES = [
    "C10_maze_d0", "C10_maze_d1", "C10_maze_d2", "C10_maze_d3",
    "C10_rooms_s10", "C10_rooms_s20", "C10_rooms_s30", "C10_rooms_s40",
]
TARGET_FAMILIES = ["C10_maze_tgt", "C10_rooms_t25", "C10_rooms_t35"]


def _maze(base, gap, clutter):
    return dataclasses.replace(base, name="C_hard_maze", mode=HARD_MODE,
                               gap_width_frac=gap, extra_clutter_range=clutter,
                               obstacle_count_range=clutter, rectangle_count_range=(3, 3), is_ood=False)


def _rooms(base, side):
    return dataclasses.replace(base, name="C_hard_rooms", mode=HARD_MODE, side_len=float(side), is_ood=False)


def c10_family_specs() -> Dict[str, "C.AnchorSpec"]:
    H7.install_c7_hard_maps()
    base = C.build_anchor_specs()
    mz, rm = base["C_hard_maze"], base["C_hard_rooms"]
    return {
        "C10_maze_d0": _maze(mz, 0.18, (2, 4)),
        "C10_maze_d1": _maze(mz, 0.15, (4, 7)),
        "C10_maze_d2": _maze(mz, 0.13, (7, 11)),
        "C10_maze_d3": _maze(mz, 0.11, (10, 14)),
        "C10_maze_tgt": _maze(mz, 0.14, (6, 9)),
        "C10_rooms_s10": _rooms(rm, 1.0),
        "C10_rooms_s20": _rooms(rm, 2.0),
        "C10_rooms_s30": _rooms(rm, 3.0),
        "C10_rooms_s40": _rooms(rm, 4.0),
        "C10_rooms_t25": _rooms(rm, 2.5),
        "C10_rooms_t35": _rooms(rm, 3.5),
    }


def family_descriptor_centroid(spec, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    descs, tries = [], 0
    while len(descs) < n and tries < n * 50:
        tries += 1
        w = C.build_world(spec, int(rng.integers(0, 2**31 - 1)), 0.45)
        if w is not None:
            descs.append(np.asarray(w.descriptor, dtype=np.float64))
    if not descs:
        raise RuntimeError(f"no worlds for {spec.name}")
    return np.mean(np.stack(descs, 0), axis=0)


def bracketing_ok(z_t, source_centroids, rel_tol: float = 0.05) -> Tuple[bool, List[int]]:
    """Check whether target centroid ``z_t`` is bracketed by the source hull.

    Degenerate (inactive) dims -- where the sources barely vary
    (``spread <= 1e-6``) -- are SKIPPED: a near-constant dim cannot bracket and
    sampling noise there would otherwise produce spurious violations. On active
    dims a small tolerance proportional to the per-dim spread is allowed, so a
    violation is flagged only if ``z_t[d] < lo[d] - rel_tol*spread[d]`` or
    ``z_t[d] > hi[d] + rel_tol*spread[d]``. ``viol`` lists only active-dim
    violations.
    """
    Z = np.stack([np.asarray(c, dtype=np.float64) for c in source_centroids], 0)
    lo, hi = Z.min(0), Z.max(0)
    spread = hi - lo
    z = np.asarray(z_t, dtype=np.float64)
    viol = []
    for d in range(z.shape[0]):
        if spread[d] <= 1e-6:
            continue  # inactive dim: sources don't vary, cannot bracket
        tol = rel_tol * spread[d]
        if z[d] < lo[d] - tol or z[d] > hi[d] + tol:
            viol.append(int(d))
    return (len(viol) == 0, viol)


@dataclass
class C10Config:
    source_dir: str = "hrm-cloud/continuous_prm/runs/c7_local"
    out_dir: str = "hrm-cloud/continuous_prm/runs/c10_local"
    backbones: str = "hrm,onlstm"
    n_src_worlds: int = 48
    n_centroid_worlds: int = 24
    n_test: int = 30
    rank: int = 8
    alpha: float = 1.0
    rbf_sigma: float = 1.0
    epochs: int = 10
    lr: float = 2.0e-4
    roadmap_nodes: int = 192
    roadmap_k: int = 7
    budgets: str = "150,250,400,600,900,1300"   # eval grid; binding budget chosen in analyze
    w_values: str = "1.0,1.1"
    seed: int = 1234
    scale: str = "local"
    mode: str = "full"
    cpu: bool = False
    source_families: str = ""   # CSV; empty => SOURCE_FAMILIES
    target_families: str = ""   # CSV; empty => TARGET_FAMILIES


def now_str() -> str:
    return C.now_str()


def rbf_weights(z_t, centroids, sigma: float) -> np.ndarray:
    Z = np.stack([np.asarray(c, dtype=np.float64) for c in centroids], 0)
    s = Z.std(0); s = np.where(s < 1e-6, 1.0, s)        # per-dim scale, epsilon-floored
    d2 = (((np.asarray(z_t, dtype=np.float64)[None, :] - Z) / s) ** 2).sum(1)
    logits = -d2 / (2.0 * max(1e-12, float(sigma) ** 2))
    logits -= logits.max()
    w = np.exp(logits); w /= w.sum()
    return w


def nearest_weights(z_t, centroids) -> np.ndarray:
    Z = np.stack([np.asarray(c, dtype=np.float64) for c in centroids], 0)
    s = Z.std(0); s = np.where(s < 1e-6, 1.0, s)
    d2 = (((np.asarray(z_t, dtype=np.float64)[None, :] - Z) / s) ** 2).sum(1)
    w = np.zeros(Z.shape[0]); w[int(np.argmin(d2))] = 1.0
    return w


def uniform_weights(n: int) -> np.ndarray:
    return np.full(int(n), 1.0 / int(n), dtype=np.float64)


# ---------------------------------------------------------------------------
# Task 3 — Source-expert training + descriptor centroids
# ---------------------------------------------------------------------------

import hashlib


def _stable_offset(s: str) -> int:
    """Deterministic per-string seed offset (Python's hash() is salted across processes)."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) & 0xFFFF


def _expert_ckpt(out_dir, family, backbone):
    return Path(out_dir) / "checkpoints" / f"c10_src__{family}__{backbone}.pt"


def train_source_experts(cfg: C10Config, device, only_families=None) -> dict:
    specs = c10_family_specs()
    fams = only_families or SOURCE_FAMILIES
    out_dir = Path(cfg.out_dir); ds_dir = out_dir / "datasets"
    C.ensure_dir(out_dir / "checkpoints"); C.ensure_dir(ds_dir)
    rmcfg = C.RoadmapConfig(n_nodes=cfg.roadmap_nodes, k_neighbors=cfg.roadmap_k)
    experts = []
    for backbone in C9._parse_csv(cfg.backbones):
        base = C9.load_source_base(Path(cfg.source_dir), backbone, device)
        tcfg = dataclasses.replace(base.train_cfg, base_epochs=int(cfg.epochs), lr=float(cfg.lr))
        for fam in fams:
            spec = specs[fam]
            seed = int(cfg.seed) + _stable_offset(fam)
            npz = C.collect_task_dataset(spec, ds_dir, f"src_{fam}", int(cfg.n_src_worlds),
                                         C9.SCALAR_NODES_PER_WORLD, rmcfg, base.feature_cfg, seed=seed)
            ck = _expert_ckpt(out_dir, fam, backbone)
            if not ck.exists():
                C9H.train_scalar_lora(base.backbone_cfg, npz, ck, base.feature_cfg, tcfg, device,
                                      seed=seed, init_ckpt=base.ckpt_path, rank=int(cfg.rank),
                                      alpha=float(cfg.alpha), bounded=True)
            cent = family_descriptor_centroid(spec, int(cfg.n_centroid_worlds), seed + 1)
            experts.append(dict(family=fam, backbone=backbone, ckpt=str(ck),
                                centroid=[float(v) for v in cent]))
            print(f"[{now_str()}] c10 source expert: {fam} {backbone} done", flush=True)
    man = {"experts": experts, "source_families": fams, "backbones": C9._parse_csv(cfg.backbones)}
    C.write_json(out_dir / "source_manifest.json", man)
    return man


# ---------------------------------------------------------------------------
# Task 4 — Weight-merge baker
# ---------------------------------------------------------------------------

def bake_weight_merge(base_ckpt, expert_ckpts, weights, out_ckpt, device) -> Path:
    """Bake W' = W_base + Σ_k w_k · scale_k · (B_k @ A_k) into a plain merged checkpoint.

    Loads the plain avgbase weights, adds the weighted LoRA deltas for every
    LoRA target, and saves a plain checkpoint (no LoRA params) that can be
    loaded by load_scalar_provider_c9h without apply_lora.

    MHA in_proj_weight special case: iter_lora_targets yields
    name="…attn.in_proj" and sub=MultiheadAttention, but PyTorch registers the
    parametrization on the MHA module itself, so the state-dict prefix is
    name[:-len(".in_proj")] (i.e., "…attn") not "…attn.in_proj".
    """
    import torch.nn as nn

    base_payload = torch.load(Path(base_ckpt), map_location="cpu")
    bb = C.BackboneConfig(**base_payload["backbone_cfg"])
    fc = C.FeatureConfig(**base_payload["feature_cfg"])
    tc = C.TrainingConfig(**base_payload["train_cfg"])
    model = C.build_model(bb, fc, tc, device)
    C.safe_load_state(model, Path(base_ckpt))   # plain base weights at {layer}.weight

    states = [torch.load(Path(p), map_location="cpu")["model"] for p in expert_ckpts]
    w = np.asarray(weights, dtype=np.float64)

    seen: set = set()
    for name, sub, attr in C.iter_lora_targets(model):
        key_id = (id(sub), attr)
        if key_id in seen:
            continue
        seen.add(key_id)

        # Derive the state-dict prefix that PyTorch actually used when
        # registering the parametrization.  For MHA in_proj_weight the
        # parametrization is registered on the MHA module (sub), whose
        # named_modules() name does NOT have the ".in_proj" suffix that
        # iter_lora_targets appends to name.
        if isinstance(sub, nn.MultiheadAttention) and attr == "in_proj_weight":
            sd_prefix = name[: -len(".in_proj")]
        else:
            sd_prefix = name

        W = getattr(sub, attr).data    # [out, in] or [3*embed, embed] for MHA

        delta = torch.zeros_like(W)
        for k, st in enumerate(states):
            if abs(float(w[k])) < 1e-12:
                continue
            A = st[f"{sd_prefix}.parametrizations.{attr}.0.A"].to(W.device, W.dtype)
            B = st[f"{sd_prefix}.parametrizations.{attr}.0.B"].to(W.device, W.dtype)
            scale = float(st[f"{sd_prefix}.parametrizations.{attr}.0.adapter_scale"])
            delta = delta + float(w[k]) * scale * (B @ A).reshape(W.shape)

        with torch.no_grad():
            getattr(sub, attr).add_(delta.to(W.device, W.dtype))

    payload = {
        "model": model.state_dict(),
        "backbone_cfg": asdict(bb),
        "feature_cfg": asdict(fc),
        "train_cfg": asdict(tc),
        "max_norm_residual": float(tc.max_norm_residual),
    }
    C.ensure_dir(Path(out_ckpt).parent)
    torch.save(payload, Path(out_ckpt))
    return Path(out_ckpt)


# ---------------------------------------------------------------------------
# Task 5 — Prediction-mix provider
# ---------------------------------------------------------------------------

class _PredMixProvider(P.HeuristicProvider):
    """RBF-weighted prediction-space mix: h = euclid + side*clip(Σ w_k ŷ_k, 0, B)."""
    def __init__(self, models, weights, feature_cfg, device, max_resid, name):
        self.models = models; self.w = np.asarray(weights, dtype=np.float64)
        self.feature_cfg = feature_cfg; self.device = device
        self.max_resid = float(max_resid); self.name = name

    def node_h(self, world, roadmap, goal_idx: int = 1) -> np.ndarray:
        feats = C5.make_hard_features_for_roadmap(world, roadmap, self.feature_cfg)
        acc = None
        for wk, m in zip(self.w, self.models):
            if abs(float(wk)) < 1e-12:
                continue
            yk = np.asarray(C.predict_norm_residuals(m, feats, self.device), dtype=np.float64)
            acc = (wk * yk) if acc is None else acc + wk * yk
        if acc is None:
            acc = np.zeros(roadmap.points.shape[0])
        yhat = np.clip(acc, 0.0, self.max_resid)
        euclid = P.euclid_to_goal(roadmap, goal_idx)
        h = euclid + world.side_len * yhat
        if not np.all(np.isfinite(h)):
            raise FloatingPointError("_PredMixProvider produced non-finite h")
        return np.maximum(h, 0.0)


def make_pred_mix_provider(expert_ckpts, weights, device, name) -> "_PredMixProvider":
    """Build a prediction-mix provider from a list of expert checkpoint paths and blend weights.

    Loads each expert via load_scalar_provider_c9h (applies LoRA, sets max_norm_residual),
    extracts the model, and wraps all experts in a _PredMixProvider.  feature_cfg and
    max_norm_residual are taken from the last loaded expert (all experts share the same
    backbone config so these values are identical across experts).
    """
    models, fc, mnr = [], None, 4.0
    for p in expert_ckpts:
        prov = C9H.load_scalar_provider_c9h(Path(p), device)   # builds + loads expert model (LoRA applied)
        models.append(prov.model); fc = prov.feature_cfg; mnr = prov.max_norm_residual
    return _PredMixProvider(models, weights, fc, device, mnr, name)


# ---------------------------------------------------------------------------
# Task 6 — Eval mode (interpolation arms on TEST worlds over budget grid)
# ---------------------------------------------------------------------------

C10_RAW_COLS = ["target", "method", "backbone", "suite", "world_index", "provider",
                "mode", "w", "budget", "found", "expansions", "closed", "cost",
                "optimal", "suboptimality", "nonfinite"]


def run_eval(cfg: C10Config, device, only_targets=None) -> Path:
    """For each TARGET family × backbone build the 7 arms, eval on TEST worlds
    over the budget grid, and write a raw CSV + a weights manifest.

    Arms per (target × backbone):
      euclid, oracle (shared, no backbone)
      zeroshot_{bb}, nearest_{bb}, uniform_wmerge_{bb}, rbf_wmerge_{bb}, rbf_pmix_{bb}
    """
    H7.install_c7_hard_maps()
    specs = c10_family_specs()
    out_dir = Path(cfg.out_dir)
    res_dir = out_dir / "results"
    shard_dir = res_dir / "_shards"
    C.ensure_dir(shard_dir)
    C.ensure_dir(out_dir / "checkpoints")
    rmcfg = C.RoadmapConfig(n_nodes=cfg.roadmap_nodes, k_neighbors=cfg.roadmap_k)
    budgets = C9._parse_ints(cfg.budgets)
    w_values = [float(x) for x in C9._parse_csv(cfg.w_values)]
    manifest = json.loads((out_dir / "source_manifest.json").read_text())
    src_fams = manifest["source_families"]
    targets = only_targets or TARGET_FAMILIES
    all_rows: List[dict] = []
    wman: List[dict] = []
    for suite_idx, target in enumerate(targets):
        spec = specs[target]
        # Materialise test worlds ONCE for determinism + speed (z_T and eval reuse same list)
        test_worlds = list(C9.iter_test_worlds(spec, suite_idx, cfg, rmcfg, cfg.n_test))
        z_T = np.mean(
            np.stack([np.asarray(w.descriptor, dtype=np.float64) for (_, w, _) in test_worlds], 0),
            axis=0,
        )
        # Shared providers (no backbone dependency)
        providers: Dict[str, object] = {
            "euclid": P.EuclidProvider(),
            "oracle": P.OracleProvider(),
        }
        meta: Dict[str, dict] = {
            "euclid": dict(method="euclid", backbone=""),
            "oracle": dict(method="oracle", backbone=""),
        }
        for backbone in C9._parse_csv(cfg.backbones):
            base = C9.load_source_base(Path(cfg.source_dir), backbone, device)
            # Build parallel (expert_ckpts, centroids) lists ordered by src_fams from manifest
            byfam = {e["family"]: e for e in manifest["experts"] if e["backbone"] == backbone}
            expert_ckpts = [byfam[f]["ckpt"] for f in src_fams]
            centroids = [np.asarray(byfam[f]["centroid"], dtype=np.float64) for f in src_fams]
            # Compute interpolation weights from z_T and centroids
            w_rbf = rbf_weights(z_T, centroids, cfg.rbf_sigma)
            w_near = nearest_weights(z_T, centroids)
            w_unif = uniform_weights(len(src_fams))
            # Arm 1: zero-shot (plain base, no merging)
            zp = P.ScalarResidualProvider(
                base.model, base.feature_cfg, device, backbone, base.train_cfg.max_norm_residual
            )
            zp.name = f"zeroshot_{backbone}"
            providers[zp.name] = zp
            meta[zp.name] = dict(method="zero_shot", backbone=backbone)
            # Arms 2-4: weight-merge (nearest / uniform / rbf)
            for arm, wv in (
                ("nearest", w_near),
                ("uniform_wmerge", w_unif),
                ("rbf_wmerge", w_rbf),
            ):
                mck = out_dir / "checkpoints" / f"c10_merge__{target}__{arm}__{backbone}.pt"
                bake_weight_merge(base.ckpt_path, expert_ckpts, wv, mck, device)
                prov = C9H.load_scalar_provider_c9h(mck, device)
                prov.name = f"{arm}_{backbone}"
                providers[prov.name] = prov
                meta[prov.name] = dict(method=arm, backbone=backbone)
            # Arm 5: prediction-mix (rbf)
            pm = make_pred_mix_provider(expert_ckpts, w_rbf, device, name=f"rbf_pmix_{backbone}")
            providers[pm.name] = pm
            meta[pm.name] = dict(method="rbf_pmix", backbone=backbone)
            # Record weights manifest entry for this (target, backbone)
            wman.append(dict(
                target=target,
                backbone=backbone,
                z_T=[float(v) for v in z_T],
                source_families=list(src_fams),
                centroids=[[float(v) for v in c] for c in centroids],
                rbf=[float(v) for v in w_rbf],
                nearest=[float(v) for v in w_near],
                uniform=[float(v) for v in w_unif],
            ))
        rows: List[dict] = []
        for world_index, world, rm in test_worlds:
            recs = P.run_world_arms(world, rm, providers, budgets, w_values, goal_idx=1)
            for r in recs:
                m = meta.get(r["provider"], dict(method=r["provider"], backbone=""))
                r.update(dict(target=target, suite=target, world_index=world_index, **m))
                rows.append(r)
        sp = shard_dir / f"{target}.csv"
        with open(sp, "w", newline="") as f:
            wri = _csv.DictWriter(f, fieldnames=C10_RAW_COLS)
            wri.writeheader()
            for r in rows:
                wri.writerow({k: r.get(k, "") for k in C10_RAW_COLS})
        all_rows.extend(rows)
        print(f"[{now_str()}] c10 eval {target}: {len(rows)} rows", flush=True)
    raw = res_dir / "continuous_prm_c10_eval_raw.csv"
    with open(raw, "w", newline="") as f:
        wri = _csv.DictWriter(f, fieldnames=C10_RAW_COLS)
        wri.writeheader()
        for r in all_rows:
            wri.writerow({k: r.get(k, "") for k in C10_RAW_COLS})
    C.write_json(res_dir / "c10_weights_manifest.json", {"entries": wman})
    print(f"[{now_str()}] c10 eval: merged {len(all_rows)} rows -> {raw}", flush=True)
    return raw


# ---------------------------------------------------------------------------
# Task 7 — Analyze mode (curves CSV + comparisons + significance + bracketing)
# ---------------------------------------------------------------------------

from continuous_prm_c6_heatmap_value_field import mcnemar_exact_p, bh_q_values
from continuous_prm_c7_integration_compare import bootstrap_median_ci

C10_ARMS = ["zero_shot", "nearest", "uniform_wmerge", "rbf_wmerge", "rbf_pmix"]


def _axis_of(name: str) -> str:
    return "maze" if "maze" in name else "rooms"


def analyze_from_raw_c10(raw_csv, out_dir, seed, targets, backbones, weights_manifest=None) -> dict:
    rows = C9._load_rows(raw_csv)
    res_dir = Path(out_dir); C.ensure_dir(res_dir)
    binding = {t: C9._binding_budget_for(rows, t) for t in targets}
    curves = []
    for target in targets:
        bb_budget = binding[target]
        eu = C9._euclid_exp_by_world(rows, target, bb_budget) if bb_budget is not None else {}
        for backbone in backbones:
            for arm in C10_ARMS:
                ratios, succ = [], []
                for r in C9._astar(rows):
                    if r["target"] != target or r["method"] != arm or r.get("backbone") != backbone:
                        continue
                    if bb_budget is None or int(r["budget"]) != bb_budget:
                        continue
                    wi = int(r["world_index"]); found = C9._is_found(r)
                    succ.append(1 if found else 0)
                    if found and wi in eu and eu[wi] > 0:
                        ratios.append(float(r["expansions"]) / eu[wi])
                med, lo, hi = bootstrap_median_ci(ratios, seed=seed)
                curves.append(dict(target=target, backbone=backbone, arm=arm, binding_budget=bb_budget,
                                   n_matched=len(ratios), exp_ratio_median=med, ci_lo=lo, ci_hi=hi,
                                   success=(float(np.mean(succ)) if succ else float("nan"))))
    curves_csv = res_dir / "continuous_prm_c10_curves.csv"
    cols = ["target", "backbone", "arm", "binding_budget", "n_matched", "exp_ratio_median", "ci_lo", "ci_hi", "success"]
    with open(curves_csv, "w", newline="") as f:
        wri = _csv.DictWriter(f, fieldnames=cols); wri.writeheader()
        for c in curves:
            wri.writerow({k: ("" if c[k] is None or (isinstance(c[k], float) and not np.isfinite(c[k])) else c[k]) for k in cols})
    comp_md = res_dir / "continuous_prm_c10_comparisons.md"
    _write_comparisons_md_c10(comp_md, curves, binding)
    sig_md = res_dir / "continuous_prm_c10_significance.md"
    _write_significance_md_c10(sig_md, rows, targets, backbones, binding)
    brk_md = res_dir / "continuous_prm_c10_bracketing.md"
    _write_bracketing_md_c10(brk_md, weights_manifest)
    print(f"[{now_str()}] c10 analyze: wrote {curves_csv}, {comp_md}, {sig_md}, {brk_md}", flush=True)
    return {"curves": curves_csv, "comparisons": comp_md, "significance": sig_md, "bracketing": brk_md}


def _write_comparisons_md_c10(path, curves, binding):
    by = {(c["target"], c["backbone"], c["arm"]): c for c in curves}
    binding_str = ", ".join(f"{t}={b}" for t, b in sorted(binding.items()))
    lines = ["# C10 Interpolation — Pre-registered Comparisons", "",
             "Matched A* expansion-ratio vs euclid (median, 95% CI) at the per-target binding budget. Lower = fewer expansions.",
             "Key contrasts: rbf_wmerge vs zero_shot (interpolation helps?), vs nearest/uniform (descriptor-weighting value), vs rbf_pmix (weight-space vs prediction-space).",
             f"Binding budgets: {binding_str}.", ""]
    targets = sorted({c["target"] for c in curves}); backbones = sorted({c["backbone"] for c in curves})
    def cell(c):
        return "n/a" if not c or not np.isfinite(c["exp_ratio_median"]) else \
            f'{c["exp_ratio_median"]:.3f} [{c["ci_lo"]:.3f},{c["ci_hi"]:.3f}] (succ {c["success"]:.2f}, n{c["n_matched"]})'
    for target in targets:
        for backbone in backbones:
            lines += [f"## {target} / {backbone}", "|arm|exp-ratio (median [CI]) |", "|---|---|"]
            for arm in C10_ARMS:
                lines.append(f"|{arm}|{cell(by.get((target, backbone, arm)))}|")
            # explicit rbf_wmerge deltas
            rw = by.get((target, backbone, "rbf_wmerge"))
            if rw and np.isfinite(rw["exp_ratio_median"]):
                for other in ("zero_shot", "nearest", "uniform_wmerge", "rbf_pmix"):
                    o = by.get((target, backbone, other))
                    if o and np.isfinite(o["exp_ratio_median"]):
                        d = rw["exp_ratio_median"] - o["exp_ratio_median"]
                        verdict = "rbf_wmerge better" if d < 0 else ("tie" if abs(d) < 1e-9 else f"{other} better")
                        lines.append(f"- rbf_wmerge vs {other}: Δ={d:+.3f} ({verdict})")
            lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_significance_md_c10(path, rows, targets, backbones, binding):
    """McNemar + BH success grid: each arm vs euclid at the binding budget (paired over worlds)."""
    astar_rows = C9._astar(rows)
    eu_found = {}
    for r in astar_rows:
        if r["provider"] != "euclid":
            continue
        bbd = binding.get(r["target"])
        if bbd is None or int(r["budget"]) != bbd:
            continue
        eu_found.setdefault(r["target"], {})[int(r["world_index"])] = C9._is_found(r)
    comparisons, pvals = [], []
    for target in targets:
        bbd = binding.get(target); eu_by_wi = eu_found.get(target, {})
        for backbone in backbones:
            for arm in C10_ARMS:
                pairs = []
                for r in astar_rows:
                    if r["target"] != target or r["method"] != arm or r.get("backbone") != backbone:
                        continue
                    if bbd is None or int(r["budget"]) != bbd:
                        continue
                    wi = int(r["world_index"])
                    if wi in eu_by_wi:
                        pairs.append((wi, C9._is_found(r)))
                n = len(pairs)
                if n == 0:
                    continue
                gain = sum(1 for (wi, af) in pairs if af and not eu_by_wi[wi])
                loss = sum(1 for (wi, af) in pairs if eu_by_wi[wi] and not af)
                eu_succ = float(np.mean([1.0 if eu_by_wi[wi] else 0.0 for (wi, _) in pairs]))
                arm_succ = float(np.mean([1.0 if af else 0.0 for (_, af) in pairs]))
                p = mcnemar_exact_p(gain, loss); pvals.append(p)
                comparisons.append(dict(target=target, backbone=backbone, arm=arm, n=n,
                                        euclid_succ=eu_succ, arm_succ=arm_succ, gain=gain, loss=loss, mcnemar_p=p))
    qvals = bh_q_values(pvals)
    for row, q in zip(comparisons, qvals):
        row["bh_q"] = q
    def _fp(p, disc): return "n/a" if disc < 2 else (f"{p:.4f}" if p is not None and np.isfinite(p) else "n/a")
    def _fn(v): return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"
    binding_str = ", ".join(f"{t}={b}" for t, b in sorted(binding.items()))
    lines = ["# C10 Interpolation — Significance", "",
             "McNemar exact (arm found & euclid not = gain; euclid found & arm not = loss). BH across the table. n/a if <2 discordant.",
             f"At the per-target binding budget ({binding_str}); pairs over worlds.", "",
             "|target|backbone|arm|n|euclid_succ|arm_succ|gain|loss|McNemar p|BH q|",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    if not comparisons:
        lines.append("|<none>|-|-|-|-|-|-|-|-|-|")
    for r in comparisons:
        disc = r["gain"] + r["loss"]
        lines.append(f"|{r['target']}|{r['backbone']}|{r['arm']}|{r['n']}|{_fn(r['euclid_succ'])}|{_fn(r['arm_succ'])}|{r['gain']}|{r['loss']}|{_fp(r['mcnemar_p'], disc)}|{_fn(r.get('bh_q'))}|")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_bracketing_md_c10(path, weights_manifest):
    lines = ["# C10 — Bracketing (G0b) + RBF descriptor selectivity", "",
             "Per target: per-AXIS bracketing (target descriptor interior to its OWN axis's source centroids) and the RBF weight mass landing on the target's own axis (descriptor selectivity).", ""]
    if weights_manifest is None or not Path(weights_manifest).exists():
        lines.append("_No weights manifest available._")
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        return
    man = json.loads(Path(weights_manifest).read_text())
    for e in man.get("entries", []):
        target, backbone = e["target"], e["backbone"]
        z_T = np.asarray(e["z_T"], dtype=np.float64)
        fams = e["source_families"]; cents = [np.asarray(c, dtype=np.float64) for c in e["centroids"]]
        rbf = np.asarray(e["rbf"], dtype=np.float64)
        tgt_axis = _axis_of(target)
        same_idx = [i for i, f in enumerate(fams) if _axis_of(f) == tgt_axis]
        if same_idx:
            ok, viol = bracketing_ok(z_T, [cents[i] for i in same_idx])
        else:
            ok, viol = (False, [])
        same_mass = float(rbf[same_idx].sum()) if same_idx else 0.0
        lines += [f"## {target} / {backbone}",
                  f"- axis: **{tgt_axis}**; same-axis sources: {[fams[i] for i in same_idx]}",
                  f"- per-axis bracketing_ok: **{ok}** (active-dim violations: {viol})",
                  f"- RBF mass on own axis: **{same_mass:.3f}** (selectivity: {'good' if same_mass >= 0.5 else 'weak'})",
                  f"- RBF weights: " + ", ".join(f"{f}={w:.3f}" for f, w in zip(fams, rbf)), ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_analyze(cfg: C10Config) -> dict:
    res = Path(cfg.out_dir) / "results"
    raw = res / "continuous_prm_c10_eval_raw.csv"
    wman = res / "c10_weights_manifest.json"
    return analyze_from_raw_c10(raw, res, seed=int(cfg.seed),
                                targets=(C9._parse_csv(cfg.target_families) or list(TARGET_FAMILIES)),
                                backbones=C9._parse_csv(cfg.backbones),
                                weights_manifest=(wman if wman.exists() else None))


# ---------------------------------------------------------------------------
# Task 8 — Full mode + CLI + scale presets
# ---------------------------------------------------------------------------

def apply_scale(cfg: C10Config) -> C10Config:
    """Scale presets. 'local' = the full RTX-5090 grid (defaults already local-scale).
    'smoke' = tiny CPU end-to-end (1 target, 2 source families, 1 epoch)."""
    if cfg.scale == "smoke":
        return dataclasses.replace(
            cfg, backbones="hrm", n_src_worlds=2, n_centroid_worlds=4, n_test=4, epochs=1,
            budgets="200,400", source_families="C10_maze_d1,C10_rooms_s20",
            target_families="C10_maze_tgt", cpu=True)
    return cfg  # 'local' uses the dataclass defaults (the full grid)


def run_full(cfg: C10Config, device) -> dict:
    fams = C9._parse_csv(cfg.source_families) or list(SOURCE_FAMILIES)
    tgts = C9._parse_csv(cfg.target_families) or list(TARGET_FAMILIES)
    print(f"[{now_str()}] c10 full: sources={fams} targets={tgts} backbones={cfg.backbones}", flush=True)
    train_source_experts(cfg, device, only_families=fams)
    run_eval(cfg, device, only_targets=tgts)
    res = analyze_from_raw_c10(
        Path(cfg.out_dir) / "results" / "continuous_prm_c10_eval_raw.csv",
        Path(cfg.out_dir) / "results", seed=int(cfg.seed), targets=tgts,
        backbones=C9._parse_csv(cfg.backbones),
        weights_manifest=(Path(cfg.out_dir) / "results" / "c10_weights_manifest.json"))
    return res


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="C10 parameter-space LoRA interpolation")
    d = C10Config()
    ap.add_argument("--mode", choices=["train", "eval", "analyze", "full"], default=d.mode)
    ap.add_argument("--scale", choices=["local", "smoke"], default=d.scale)
    ap.add_argument("--source-dir", default=d.source_dir)
    ap.add_argument("--out-dir", default=d.out_dir)
    ap.add_argument("--backbones", default=d.backbones)
    ap.add_argument("--source-families", default=d.source_families)
    ap.add_argument("--target-families", default=d.target_families)
    ap.add_argument("--n-src-worlds", type=int, default=d.n_src_worlds)
    ap.add_argument("--n-centroid-worlds", type=int, default=d.n_centroid_worlds)
    ap.add_argument("--n-test", type=int, default=d.n_test)
    ap.add_argument("--rank", type=int, default=d.rank)
    ap.add_argument("--alpha", type=float, default=d.alpha)
    ap.add_argument("--rbf-sigma", type=float, default=d.rbf_sigma)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--roadmap-nodes", type=int, default=d.roadmap_nodes)
    ap.add_argument("--roadmap-k", type=int, default=d.roadmap_k)
    ap.add_argument("--budgets", default=d.budgets)
    ap.add_argument("--w-values", default=d.w_values)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--cpu", action="store_true", default=d.cpu)
    return ap


def config_from_args(args) -> C10Config:
    return C10Config(
        mode=args.mode, scale=args.scale, source_dir=args.source_dir, out_dir=args.out_dir,
        backbones=args.backbones, source_families=args.source_families, target_families=args.target_families,
        n_src_worlds=args.n_src_worlds, n_centroid_worlds=args.n_centroid_worlds, n_test=args.n_test,
        rank=args.rank, alpha=args.alpha, rbf_sigma=args.rbf_sigma, epochs=args.epochs, lr=args.lr,
        roadmap_nodes=args.roadmap_nodes, roadmap_k=args.roadmap_k, budgets=args.budgets,
        w_values=args.w_values, seed=args.seed, cpu=args.cpu)


def main(argv=None):
    args = build_argparser().parse_args(argv)
    cfg = apply_scale(config_from_args(args))
    device = torch.device("cpu") if cfg.cpu or not torch.cuda.is_available() else torch.device("cuda")
    print(f"[{now_str()}] c10 mode={cfg.mode} scale={cfg.scale} device={device}", flush=True)
    if cfg.mode == "train":
        train_source_experts(cfg, device, only_families=(C9._parse_csv(cfg.source_families) or None))
    elif cfg.mode == "eval":
        run_eval(cfg, device, only_targets=(C9._parse_csv(cfg.target_families) or None))
    elif cfg.mode == "analyze":
        run_analyze(cfg)
    else:
        run_full(cfg, device)


if __name__ == "__main__":
    main()
