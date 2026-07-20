#!/usr/bin/env python3
"""HRM head-to-head validation: incumbent DeepSapientHRMBackbone (audit-found
degeneracy: MultiheadAttention over a length-1 sequence, i.e. no cross-token
attention) vs a RepairedHRMBackbone that runs REAL cross-token attention.

Purpose (see docs/experiments/cross-space/PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md Sec 5.2): verify
(a) the incumbent reproduces prior C9 zero-shot numbers on the continuous-PRM
hard targets, and (b) a capacity-matched variant with genuine cross-token
attention performs same-or-better under an identical eval protocol. New-file
only; reuses C9/C7/C5 machinery via import. Does NOT modify continuous_prm_common.py,
transfer_astar_*, or any existing experiment module.

Arm (a) reproduction: continuous_prm_c9_transfer.load_source_base + zero-shot
ScalarResidualProvider eval, mirrored from C9's run_eval (continuous_prm_c9_transfer.py:281-331).

Arm (b) repaired: RepairedHRMBackbone iterates L/H transformer blocks (attention
over the FULL 24-token sequence) on a fixed input, instead of scanning tokens
one at a time through length-1-sequence attention. Trained on the same pooled
scalar data (BalancedTaskDataset over the three datasets_scalar/*.npz files)
with the recipe recorded in the incumbent avgbase checkpoint's train_cfg.

Usage:
  python hrm_headtohead.py --mode eval_a --device cuda
  python hrm_headtohead.py --mode train_b --device cuda
  python hrm_headtohead.py --mode eval_b --device cuda
  python hrm_headtohead.py --mode all --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import continuous_prm_common as C
import continuous_prm_providers as P
import continuous_prm_c7_hard_maps as H7
import continuous_prm_c9_transfer as C9

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "runs" / "c7_local"
OUT_DIR = HERE / "runs" / "hrm_headtohead_local"
TARGETS = ["C_hard_maze_dense", "C_hard_bugtrap", "C_hard_rooms_large"]  # full C9 ordering; suite_idx is seed-relevant
EVAL_TARGETS = ["C_hard_maze_dense", "C_hard_rooms_large"]  # what we actually score
TARGET_SUITE_IDX = {t: i for i, t in enumerate(TARGETS)}
BINDING_BUDGET = {"C_hard_maze_dense": 140, "C_hard_rooms_large": 56}  # confirmed vs runs/c7_local/calibration.json
N_TEST_WORLDS = 30
ROADMAP_NODES = 192
ROADMAP_K = 7
SCALAR_DATASETS = {
    "C_hard_maze": SOURCE_DIR / "datasets_scalar" / "C_hard_maze_train_scalar.npz",
    "C_hard_rooms": SOURCE_DIR / "datasets_scalar" / "C_hard_rooms_train_scalar.npz",
    "C_hard_spiral": SOURCE_DIR / "datasets_scalar" / "C_hard_spiral_train_scalar.npz",
}
REPAIRED_SEED = 1234


# -----------------------------------------------------------------------------
# Arm (b): repaired backbone — real cross-token attention over the fixed
# 24-token sequence, instead of a token-by-token recurrent scan through
# length-1-sequence attention (the audit-found degeneracy).
# -----------------------------------------------------------------------------

class TransformerEncoderBlockRT(nn.Module):
    """Standard post-norm encoder block: MHA over the FULL sequence + SwiGLU FFN.
    Reuses common's RMSNorm/SwiGLU. batch_first MHA so input is (B, S, H)."""

    def __init__(self, dim: int, num_heads: int, ffn_mult: float = 2.6):
        super().__init__()
        self.norm1 = C.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = C.RMSNorm(dim)
        self.ffn = C.SwiGLU(dim, int(dim * ffn_mult))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, H). Cross-token attention: every token attends to every
        # other token in the sequence (S = seq_len, NOT 1) — this is the
        # mechanism the incumbent GatedRecurrentBlock lacks.
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class RepairedHRMBackbone(nn.Module):
    """Two-timescale HRM computation with REAL cross-token attention: instead of
    a token-by-token recurrent scan with length-1 MHA (continuous_prm_common's
    DeepSapientHRMBackbone / GatedRecurrentBlock), iterate L/H transformer blocks
    (attention across ALL tokens) on the fixed token sequence -- the HRM paper's
    iterate-on-fixed-input pattern, sized to match the incumbent.

    forward: h = embed(x) -> (B, S, H); zH = h.mean(1) -> (B, H) slow state.
    For n_cycles: inject zH into every token, run L_blocks (cross-token attn),
    then refine zH via a pooled (mean-over-tokens) update through H_pool/H_blocks.
    Returns zH (B, H), the slow state, as the pooled sequence encoding -- matching
    DeepSapientHRMBackbone.encode_sequence's (B, H) contract.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, n_cycles: int = 2,
                 ffn_mult: float = 2.6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cycles = int(n_cycles)
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.L_blocks = nn.ModuleList([TransformerEncoderBlockRT(hidden_dim, num_heads, ffn_mult) for _ in range(num_layers)])
        self.H_blocks = nn.ModuleList([TransformerEncoderBlockRT(hidden_dim, num_heads, ffn_mult) for _ in range(num_layers)])
        # H update: pooled refinement of the slow state through a Linear, as specced.
        self.H_pool = nn.Linear(hidden_dim, hidden_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq, token_dim). Mirrors the spec pseudocode exactly:
        #   h = embed(x); zH = h.mean(1)
        #   for _ in range(n_cycles):
        #       hs = h + zH.unsqueeze(1); for blk in L_blocks: hs = blk(hs)
        #       h = hs; zH = zH + H_pool(h)   # H_pool takes the token-mean of h
        # H_blocks runs once per cycle on the post-L token states so the
        # declared H_blocks module list (two-timescale separation: L refines
        # tokens, H_blocks re-contextualizes at the pooled/slow level before
        # the H_pool update) is exercised rather than left unused dead weight;
        # H_blocks also gets real cross-token attention over the full sequence.
        h = self.embed(x)               # (B, S, H) token states
        zH = h.mean(dim=1)              # (B, H) slow state, pooled over tokens
        for _ in range(self.n_cycles):
            hs = h + zH.unsqueeze(1)    # inject slow state into every token
            for blk in self.L_blocks:
                hs = blk(hs)            # cross-token attention over the FULL sequence
            for blk in self.H_blocks:
                hs = blk(hs)            # H stack: also real cross-token attention
            h = hs
            zH = zH + self.H_pool(h.mean(dim=1))  # H update: pooled refinement
        return zH


class RepairedHeuristicModel(nn.Module):
    """Same head + softplus + clamp contract as continuous_prm_common.ContinuousHeuristicModel,
    with RepairedHRMBackbone swapping in for DeepSapientHRMBackbone. Head/forward
    logic copied verbatim from ContinuousHeuristicModel so P.ScalarResidualProvider
    works unchanged."""

    def __init__(self, token_dim: int, hidden_dim: int, num_heads: int, num_layers: int,
                 head_hidden: int, max_norm_residual: float = 4.0, n_cycles: int = 2, ffn_mult: float = 2.6):
        super().__init__()
        self.max_norm_residual = float(max_norm_residual)
        self.backbone = RepairedHRMBackbone(token_dim, hidden_dim, num_heads, num_layers, n_cycles=n_cycles, ffn_mult=ffn_mult)
        # --- copied from ContinuousHeuristicModel.__init__ (continuous_prm_common.py) ---
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )
        last = self.head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, -2.0)

    def forward(self, x_seq: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        # --- copied from ContinuousHeuristicModel.forward (continuous_prm_common.py) ---
        ctx = self.backbone.encode_sequence(x_seq)
        raw = self.head(ctx).squeeze(-1)
        out = F.softplus(raw)
        if clamp:
            out = torch.clamp(out, min=0.0, max=self.max_norm_residual)
        return out


@dataclass
class RepairedSizing:
    hidden_dim: int
    num_heads: int
    num_layers: int
    head_hidden: int
    n_cycles: int
    ffn_mult: float


def build_repaired_model(sizing: RepairedSizing, token_dim: int, max_norm_residual: float, device) -> RepairedHeuristicModel:
    model = RepairedHeuristicModel(
        token_dim=token_dim, hidden_dim=sizing.hidden_dim, num_heads=sizing.num_heads,
        num_layers=sizing.num_layers, head_hidden=sizing.head_hidden,
        max_norm_residual=max_norm_residual, n_cycles=sizing.n_cycles, ffn_mult=sizing.ffn_mult,
    )
    return model.to(device)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def pick_matched_sizing(incumbent_backbone_cfg: C.BackboneConfig, incumbent_params: int, token_dim: int,
                        max_norm_residual: float, tol: float = 0.25) -> RepairedSizing:
    """Match capacity to the incumbent (hidden_dim, num_heads, num_layers read from
    the loaded avgbase payload). If the naive repaired model is >tol larger, shrink
    FFN width until within tolerance; report both counts either way."""
    hidden_dim = incumbent_backbone_cfg.hidden_dim
    num_heads = incumbent_backbone_cfg.num_heads
    num_layers = incumbent_backbone_cfg.num_layers
    head_hidden = incumbent_backbone_cfg.head_hidden
    for ffn_mult in (2.6, 1.8, 1.2, 0.8, 0.5):
        sizing = RepairedSizing(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
                                head_hidden=head_hidden, n_cycles=2, ffn_mult=ffn_mult)
        probe = build_repaired_model(sizing, token_dim, max_norm_residual, torch.device("cpu"))
        n = count_params(probe)
        del probe
        if n <= incumbent_params * (1.0 + tol):
            return sizing
    # fall through: return the smallest tried, caller reports the (larger) mismatch honestly.
    return sizing


# -----------------------------------------------------------------------------
# Train / load arm (b)
# -----------------------------------------------------------------------------

def pooled_scalar_arrays() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Pool the three C7 per-suite scalar datasets exactly as C.train_avgbase did
    for the incumbent avgbase (BalancedTaskDataset over a {task: (x, y)} dict)."""
    missing = [str(p) for p in SCALAR_DATASETS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"pooled scalar datasets missing (expected from C7 run): {missing}")
    return {task: C.load_npz_arrays(path) for task, path in SCALAR_DATASETS.items()}


def train_repaired_model(sizing: RepairedSizing, feature_cfg: C.FeatureConfig, train_cfg: C.TrainingConfig,
                         device, seed: int, out_ckpt: Path) -> Path:
    """Train RepairedHeuristicModel on the SAME pooled data + recipe as the
    incumbent avgbase. Mirrors C.train_avgbase's loop (BalancedTaskDataset,
    AdamW, smooth-L1, grad-clip) with the repaired model swapped in."""
    arrays = pooled_scalar_arrays()
    ds = C.BalancedTaskDataset(arrays)
    loader = C.make_loader(ds, train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)
    model = build_repaired_model(sizing, feature_cfg.token_dim, train_cfg.max_norm_residual, device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    C.set_global_seed(int(seed))
    history: List[dict] = []
    for epoch in range(1, train_cfg.base_epochs + 1):
        model.train()
        losses: List[float] = []
        maes: List[float] = []
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            if not torch.isfinite(pred).all():
                raise RuntimeError("nonfinite repaired-model predictions")
            loss = F.smooth_l1_loss(pred, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite repaired-model loss")
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            losses.append(float(loss.item()))
            maes.append(float(torch.mean(torch.abs(pred.detach() - yb)).item()))
        row = {"epoch": epoch, "loss": C.finite_mean(losses), "mae": C.finite_mean(maes), "seconds": time.time() - t0}
        history.append(row)
        print(f"[{C.now_str()}] repaired-hrm epoch {epoch}/{train_cfg.base_epochs}: loss={row['loss']:.5f} mae={row['mae']:.5f}", flush=True)
    payload = {
        "model": model.state_dict(),
        "sizing": asdict(sizing),
        "feature_cfg": asdict(feature_cfg),
        "train_cfg": asdict(train_cfg),
        "tasks": list(SCALAR_DATASETS.keys()),
        "history": history,
        "seed": int(seed),
    }
    C.ensure_dir(out_ckpt.parent)
    torch.save(payload, out_ckpt)
    return out_ckpt


def load_repaired_provider(ckpt_path: Path, device) -> P.ScalarResidualProvider:
    payload = torch.load(ckpt_path, map_location="cpu")
    sizing = RepairedSizing(**payload["sizing"])
    feature_cfg = C.FeatureConfig(**payload["feature_cfg"])
    train_cfg = C.TrainingConfig(**payload["train_cfg"])
    model = build_repaired_model(sizing, feature_cfg.token_dim, train_cfg.max_norm_residual, device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    prov = P.ScalarResidualProvider(model, feature_cfg, device, "hrmx", train_cfg.max_norm_residual)
    prov.name = "repaired_hrm"
    return prov


# -----------------------------------------------------------------------------
# Eval: mirrors C9's run_eval zero-shot path (continuous_prm_c9_transfer.py:281-331)
# for a single provider against euclid, on the two hard targets, with the
# matched-ratio-vs-euclid computed the C9/C9-analyze way (binding budget, both-
# solved worlds, median ratio).
# -----------------------------------------------------------------------------

def eval_provider_on_target(target: str, provider, device, n_test: int = N_TEST_WORLDS) -> dict:
    H7.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    spec = specs[target]
    suite_idx = TARGET_SUITE_IDX[target]  # index into the FULL 3-target C9 ordering (seed-relevant)
    budget = BINDING_BUDGET[target]
    rmcfg = C.RoadmapConfig(n_nodes=ROADMAP_NODES, k_neighbors=ROADMAP_K)
    cfg = C9.C9Config(roadmap_nodes=ROADMAP_NODES, roadmap_k=ROADMAP_K, seed=1234)
    providers = {"euclid": P.EuclidProvider(), provider.name: provider}
    rows = []
    for world_index, world, rm in C9.iter_test_worlds(spec, suite_idx, cfg, rmcfg, n_test):
        recs = P.run_world_arms(world, rm, providers, budgets=[budget], w_values=[], goal_idx=1)
        for r in recs:
            r["world_index"] = world_index
            rows.append(r)
    astar_rows = [r for r in rows if r["mode"] == "astar"]

    def is_found(r): return bool(r["found"])

    euclid_by_world = {r["world_index"]: r for r in astar_rows if r["provider"] == "euclid"}
    prov_by_world = {r["world_index"]: r for r in astar_rows if r["provider"] == provider.name}
    succ_euclid = sum(1 for r in euclid_by_world.values() if is_found(r)) / max(1, len(euclid_by_world))
    succ_prov = sum(1 for r in prov_by_world.values() if is_found(r)) / max(1, len(prov_by_world))
    ratios = []
    for wi, pr in prov_by_world.items():
        er = euclid_by_world.get(wi)
        if er is None or not is_found(pr) or not is_found(er) or float(er["expansions"]) <= 0:
            continue
        ratios.append(float(pr["expansions"]) / float(er["expansions"]))
    matched_ratio = float(np.median(ratios)) if ratios else float("nan")
    return {
        "target": target, "provider": provider.name, "budget": budget, "n_test": n_test,
        "n_both_solved": len(ratios), "succ_provider": succ_prov, "succ_euclid": succ_euclid,
        "matched_exp_ratio_vs_euclid": matched_ratio,
    }


def eval_arm(name: str, provider_factory, device, targets: List[str] = EVAL_TARGETS) -> List[dict]:
    results = []
    for target in targets:
        provider = provider_factory(target)
        res = eval_provider_on_target(target, provider, device)
        res["arm"] = name
        results.append(res)
        print(f"[{C.now_str()}] {name} {target}: matched_ratio={res['matched_exp_ratio_vs_euclid']:.4f} "
              f"succ={res['succ_provider']:.3f} (euclid succ={res['succ_euclid']:.3f}, n_both_solved={res['n_both_solved']})",
              flush=True)
    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def run_eval_a(device) -> List[dict]:
    base = C9.load_source_base(SOURCE_DIR, "hrm", device)
    zp = P.ScalarResidualProvider(base.model, base.feature_cfg, device, "hrm", base.train_cfg.max_norm_residual)
    zp.name = "incumbent_hrm_zeroshot"
    results = eval_arm("incumbent_hrm_zeroshot", lambda target: zp, device)
    for r in results:
        r["params"] = count_params(base.model)
    return results


def run_train_b(device) -> Tuple[Path, dict]:
    base = C9.load_source_base(SOURCE_DIR, "hrm", device)  # for backbone_cfg / feature_cfg / train_cfg only
    incumbent_params = count_params(base.model)
    sizing = pick_matched_sizing(base.backbone_cfg, incumbent_params, base.feature_cfg.token_dim, base.train_cfg.max_norm_residual)
    probe = build_repaired_model(sizing, base.feature_cfg.token_dim, base.train_cfg.max_norm_residual, torch.device("cpu"))
    repaired_params = count_params(probe)
    del probe
    ratio = repaired_params / max(1, incumbent_params)
    print(f"[{C.now_str()}] incumbent params={incumbent_params} repaired params={repaired_params} ratio={ratio:.3f} sizing={sizing}", flush=True)
    out_ckpt = OUT_DIR / "checkpoints" / "repaired_hrm.pt"
    ckpt = train_repaired_model(sizing, base.feature_cfg, base.train_cfg, device, seed=REPAIRED_SEED, out_ckpt=out_ckpt)
    meta = {"incumbent_params": incumbent_params, "repaired_params": repaired_params, "ratio": ratio, "sizing": asdict(sizing)}
    C.write_json(OUT_DIR / "results" / "param_counts.json", meta)
    return ckpt, meta


def run_eval_b(device, ckpt_path: Optional[Path] = None) -> List[dict]:
    ckpt_path = ckpt_path or (OUT_DIR / "checkpoints" / "repaired_hrm.pt")
    prov = load_repaired_provider(ckpt_path, device)
    results = eval_arm("repaired_hrm", lambda target: prov, device)
    payload = torch.load(ckpt_path, map_location="cpu")
    sizing = RepairedSizing(**payload["sizing"])
    feature_cfg = C.FeatureConfig(**payload["feature_cfg"])
    train_cfg = C.TrainingConfig(**payload["train_cfg"])
    m = build_repaired_model(sizing, feature_cfg.token_dim, train_cfg.max_norm_residual, torch.device("cpu"))
    n_params = count_params(m)
    del m
    for r in results:
        r["params"] = n_params
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval_a", "train_b", "eval_b", "all"], default="all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    C.ensure_dir(OUT_DIR / "results")

    all_results: dict = {}
    if args.mode in ("eval_a", "all"):
        all_results["arm_a"] = run_eval_a(device)
        C.write_json(OUT_DIR / "results" / "arm_a_eval.json", {"results": all_results["arm_a"]})
    if args.mode in ("train_b", "all"):
        ckpt, meta = run_train_b(device)
        all_results["train_b_meta"] = meta
    if args.mode in ("eval_b", "all"):
        all_results["arm_b"] = run_eval_b(device)
        C.write_json(OUT_DIR / "results" / "arm_b_eval.json", {"results": all_results["arm_b"]})

    C.write_json(OUT_DIR / "results" / "headtohead_summary.json", all_results)
    print(f"[{C.now_str()}] hrm_headtohead {args.mode} done -> {OUT_DIR / 'results'}", flush=True)


if __name__ == "__main__":
    main()
