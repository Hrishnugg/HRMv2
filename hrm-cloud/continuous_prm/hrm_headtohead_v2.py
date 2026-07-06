#!/usr/bin/env python3
"""Remediation sweep for the RepairedHRMBackbone that LOST the head-to-head in
hrm_headtohead.py (arm (b): maze_dense 0.7185@0.933 vs incumbent 0.6501@1.000;
rooms_large 0.8611@0.867 vs incumbent 0.7714@0.967 -- see HRM_HEADTOHEAD.md).

A follow-up audit found four integration defects in that arm (b) that plausibly
inhibited it independent of the "repaired cross-token attention" architectural
change under test:

  I1 (init):        blanket std=0.01 normal-init on EVERY nn.Linear (copied from
                     the incumbent's DeepSapientHRMBackbone._init_weights) silences
                     a transformer's attention+FFN at init -- that init suits a
                     small-signal GATED recurrent design, not a fresh transformer.
  I2 (two-timescale): the original's zH update ran H_blocks -- ANOTHER full
                     token-level transformer stack -- over the post-L token
                     states every cycle, so "H" was never actually a pooled slow
                     state; it was a second L stack. This collapses the intended
                     two-timescale separation (fast token-level L vs slow
                     pooled-summary H) into "two L stacks in series."
  I3 (readout):      mean-pooling over all 24 tokens (1 state/goal token + 6
                     obstacle tokens + 16 ray tokens + 1) to seed zH, and again
                     for the H_pool update, treats the goal-relative state token
                     (token 0; see continuous_prm_common.make_feature_sequence)
                     as no more informative than an arbitrary ray sample --
                     diluting the single most task-relevant token 1-in-24.
  I5 (recipe):       arm (b) reused the incumbent's recipe (lr=2e-4, epochs=16)
                     unexamined; final train loss 0.060 vs the incumbent's 0.0498
                     on the identical data suggests under-convergence may also be
                     contributing, independent of I1-I3.

This file builds RepairedHRMBackboneV2 (I1+I2+I3 fixed) and sweeps four training
variants (V1: fixes only, incumbent recipe; V2: fixes + higher-lr+warmup; V3:
fixes + 2x epochs; V4: control = ORIGINAL unfixed RepairedHRMBackbone + 2x
epochs) to separate "the fixes matter" from "it was just undertrained" from
"the architectural change (real cross-token attention on a serialized feature
bag) is what hurts, independent of integration bugs."

New-file only; imports hrm_headtohead.py (arm-a incumbent, eval harness,
TARGETS/BINDING_BUDGET/SOURCE_DIR, pooled_scalar_arrays) and
continuous_prm_common.py unchanged. Does NOT modify continuous_prm_common.py,
transfer_astar_*, or hrm_headtohead.py.

Usage:
  python hrm_headtohead_v2.py --mode train --variants v1,v2,v3,v4 --device cuda
  python hrm_headtohead_v2.py --mode eval --variants best,v4 --device cuda
  python hrm_headtohead_v2.py --mode all --variants v1,v2,v3,v4 --eval-all --device cuda
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import continuous_prm_common as C
import continuous_prm_providers as P
import hrm_headtohead as HH

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "runs" / "hrm_headtohead_v2_local"
EVAL_TARGETS = HH.EVAL_TARGETS  # ["C_hard_maze_dense", "C_hard_rooms_large"]


# -----------------------------------------------------------------------------
# I1 + I2 + I3: RepairedHRMBackboneV2
# -----------------------------------------------------------------------------

def _xavier_init_transformer_linears(module: nn.Module) -> None:
    """I1: xavier_uniform for MHA out_proj and SwiGLU/embed Linears; leave MHA
    in_proj_weight untouched (PyTorch's nn.MultiheadAttention.__init__ already
    applies xavier_uniform_ to in_proj_weight by default -- re-initializing it
    here would be redundant, not wrong, but the spec says "just don't clobber
    it", i.e. don't overwrite it with something worse; xavier-on-xavier is a
    no-op modulo RNG draw so we skip it explicitly for clarity). All plain
    nn.Linear weights (embed, SwiGLU w1/w2/w3, H_mlp) get xavier_uniform_.
    Biases zero. This intentionally does NOT touch the head (kept separately
    zero-init + bias -2.0, per I1's "keep ONLY the head's zero-init" clause)."""
    for m in module.modules():
        if isinstance(m, nn.MultiheadAttention):
            # in_proj already xavier via PyTorch default init -- do not touch.
            nn.init.xavier_uniform_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)


class TransformerEncoderBlockRT2(nn.Module):
    """Same block as hrm_headtohead.TransformerEncoderBlockRT (post-norm MHA +
    SwiGLU over the full token sequence), duplicated here (not imported) so its
    weight init is fully controlled by RepairedHRMBackboneV2's I1 pass rather
    than depending on hrm_headtohead's module staying untouched (it must not be
    modified) while still letting this file own the init contract end-to-end."""

    def __init__(self, dim: int, num_heads: int, ffn_mult: float = 2.6):
        super().__init__()
        self.norm1 = C.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = C.RMSNorm(dim)
        self.ffn = C.SwiGLU(dim, int(dim * ffn_mult))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class HMlp(nn.Module):
    """I2's H update network: 2-layer MLP, hidden_dim wide, taking
    concat[zH, pooled] (2*hidden_dim) -> hidden_dim -> hidden_dim."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class RepairedHRMBackboneV2(nn.Module):
    """I1 (init) + I2 (true two-timescale) + I3 (state-token readout) applied to
    hrm_headtohead.RepairedHRMBackbone.

    forward per cycle (I2, verbatim from the remediation spec):
        hs = h + zH.unsqueeze(1)          # inject pooled slow state into every token
        for blk in L_blocks: hs = blk(hs) # cross-token attn over the FULL sequence
        h = hs
        pooled = rms_norm(h[:, 0])        # I3: state/goal-token readout, not mean-pool
        zH = zH + H_mlp(rms_norm(concat[zH, pooled]))

    H_blocks is DROPPED ENTIRELY (I2: "H must operate on the pooled slow state,
    not re-process tokens" -- a second token-level transformer stack is not a
    slow-timescale operation, it is more L). Its capacity is folded into +1
    L_blocks layer (num_layers+1) to stay within ~25% of the incumbent's
    2,158,529 params -- see pick_matched_sizing_v2 below, which reports the
    resulting count so the tradeoff is visible rather than assumed.

    Final return: rms_norm(zH) (I3's "final return rms_norm(zH) before the
    head"), matching ContinuousHeuristicModel's contract of returning a
    (B, hidden_dim) pooled sequence encoding to the head.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, n_cycles: int = 2,
                 ffn_mult: float = 2.6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cycles = int(n_cycles)
        self.embed = nn.Linear(input_dim, hidden_dim)
        # I2: fold H_blocks' capacity into L_blocks (+1 layer) since H_blocks is
        # dropped entirely as a token-level stack.
        self.L_blocks = nn.ModuleList(
            [TransformerEncoderBlockRT2(hidden_dim, num_heads, ffn_mult) for _ in range(num_layers + 1)]
        )
        self.readout_norm = C.RMSNorm(hidden_dim)   # I3: norm on h[:, 0] pooled readout
        self.h_update_norm = C.RMSNorm(2 * hidden_dim)  # norm on concat[zH, pooled] before H_mlp
        self.H_mlp = HMlp(hidden_dim)                # I2: pooled-state update network
        self.final_norm = C.RMSNorm(hidden_dim)      # I3: rms_norm(zH) before the head
        self._apply_i1_init()

    def _apply_i1_init(self) -> None:
        # I1: xavier_uniform for embed + SwiGLU Linears + H_mlp Linears; MHA
        # out_proj xavier'd explicitly, in_proj left at PyTorch's own xavier
        # default. NOTHING in this backbone gets the incumbent's blanket
        # std=0.01 normal-init -- that is the entire point of I1.
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.zeros_(self.embed.bias)
        for blk in self.L_blocks:
            nn.init.xavier_uniform_(blk.ffn.w1.weight)
            nn.init.xavier_uniform_(blk.ffn.w2.weight)
            nn.init.xavier_uniform_(blk.ffn.w3.weight)
        _xavier_init_transformer_linears(self)
        nn.init.xavier_uniform_(self.H_mlp.fc1.weight)
        nn.init.zeros_(self.H_mlp.fc1.bias)
        nn.init.xavier_uniform_(self.H_mlp.fc2.weight)
        nn.init.zeros_(self.H_mlp.fc2.bias)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)               # (B, S, H) token states
        zH = h.mean(dim=1)              # (B, H) initial slow state (seed only; NOT re-derived by mean-pool per cycle -- I3 fixes the per-cycle readout)
        for _ in range(self.n_cycles):
            hs = h + zH.unsqueeze(1)    # inject slow state into every token (I2)
            for blk in self.L_blocks:
                hs = blk(hs)            # cross-token attention over the FULL sequence; H_blocks dropped (I2)
            h = hs
            pooled = self.readout_norm(h[:, 0])  # I3: state/goal-token readout, not mean-pool
            h_update_in = self.h_update_norm(torch.cat([zH, pooled], dim=-1))
            zH = zH + self.H_mlp(h_update_in)    # I2: H operates on the pooled slow state only
        return self.final_norm(zH)      # I3: rms_norm(zH) before the head


class RepairedHeuristicModelV2(nn.Module):
    """Same head + softplus + clamp contract as ContinuousHeuristicModel /
    hrm_headtohead.RepairedHeuristicModel. Head init (zero-init final layer,
    bias -2.0) is the ONE piece of the original's init that I1 says to KEEP --
    copied verbatim, unchanged."""

    def __init__(self, token_dim: int, hidden_dim: int, num_heads: int, num_layers: int,
                 head_hidden: int, max_norm_residual: float = 4.0, n_cycles: int = 2, ffn_mult: float = 2.6):
        super().__init__()
        self.max_norm_residual = float(max_norm_residual)
        self.backbone = RepairedHRMBackboneV2(token_dim, hidden_dim, num_heads, num_layers, n_cycles=n_cycles, ffn_mult=ffn_mult)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )
        # I1: "keep ONLY the head's zero-init + bias -2.0 (copy from the
        # original wrapper)" -- the two non-final head Linears keep PyTorch's
        # default init (untouched), matching the original wrapper's behavior
        # exactly (it never touched head[0]/head[2] either -- only head[-1]).
        last = self.head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, -2.0)

    def forward(self, x_seq: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        ctx = self.backbone.encode_sequence(x_seq)
        raw = self.head(ctx).squeeze(-1)
        out = F.softplus(raw)
        if clamp:
            out = torch.clamp(out, min=0.0, max=self.max_norm_residual)
        return out


def build_repaired_model_v2(sizing: "HH.RepairedSizing", token_dim: int, max_norm_residual: float, device) -> RepairedHeuristicModelV2:
    model = RepairedHeuristicModelV2(
        token_dim=token_dim, hidden_dim=sizing.hidden_dim, num_heads=sizing.num_heads,
        num_layers=sizing.num_layers, head_hidden=sizing.head_hidden,
        max_norm_residual=max_norm_residual, n_cycles=sizing.n_cycles, ffn_mult=sizing.ffn_mult,
    )
    return model.to(device)


def pick_matched_sizing_v2(incumbent_backbone_cfg: C.BackboneConfig, incumbent_params: int, token_dim: int,
                           max_norm_residual: float, tol: float = 0.25) -> "HH.RepairedSizing":
    """Same shrink-FFN-until-in-tolerance search as hrm_headtohead.pick_matched_sizing,
    but probing build_repaired_model_v2 (V2 has +1 L layer folded from the dropped
    H_blocks, so its param count differs from V1's naive sizing at the same ffn_mult)."""
    hidden_dim = incumbent_backbone_cfg.hidden_dim
    num_heads = incumbent_backbone_cfg.num_heads
    num_layers = incumbent_backbone_cfg.num_layers
    head_hidden = incumbent_backbone_cfg.head_hidden
    sizing = None
    for ffn_mult in (2.6, 1.8, 1.2, 0.8, 0.5):
        sizing = HH.RepairedSizing(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
                                   head_hidden=head_hidden, n_cycles=2, ffn_mult=ffn_mult)
        probe = build_repaired_model_v2(sizing, token_dim, max_norm_residual, torch.device("cpu"))
        n = HH.count_params(probe)
        del probe
        if n <= incumbent_params * (1.0 + tol):
            return sizing
    return sizing


# -----------------------------------------------------------------------------
# I5: recipe sweep. Variants share the identical pooled dataset (reuses
# hrm_headtohead.pooled_scalar_arrays, i.e. the SAME BalancedTaskDataset the
# incumbent and the original arm (b) trained on) and seed 1234.
# -----------------------------------------------------------------------------

@dataclass
class VariantSpec:
    key: str
    label: str
    use_v2_backbone: bool          # False => V4 control: ORIGINAL unfixed RepairedHRMBackbone
    epochs: int
    lr: float
    warmup_steps: int = 0          # 0 => no warmup


VARIANTS: Dict[str, VariantSpec] = {
    "v1": VariantSpec(key="v1", label="V1 (fixes 1-3, incumbent recipe)", use_v2_backbone=True, epochs=16, lr=2e-4, warmup_steps=0),
    "v2": VariantSpec(key="v2", label="V2 (V1 + lr 5e-4, 100-step warmup)", use_v2_backbone=True, epochs=16, lr=5e-4, warmup_steps=100),
    "v3": VariantSpec(key="v3", label="V3 (V1 + epochs 32)", use_v2_backbone=True, epochs=32, lr=2e-4, warmup_steps=0),
    "v4": VariantSpec(key="v4", label="V4 control (ORIGINAL unfixed backbone + epochs 32)", use_v2_backbone=False, epochs=32, lr=2e-4, warmup_steps=0),
}


def _build_model_for_variant(spec: VariantSpec, sizing_v2: "HH.RepairedSizing", sizing_v1: "HH.RepairedSizing",
                             token_dim: int, max_norm_residual: float, device) -> nn.Module:
    if spec.use_v2_backbone:
        return build_repaired_model_v2(sizing_v2, token_dim, max_norm_residual, device)
    # V4 control: the ORIGINAL (unfixed) RepairedHRMBackbone from hrm_headtohead.py,
    # unmodified, at its own (V1-incompatible) sizing since it still has H_blocks.
    return HH.build_repaired_model(sizing_v1, token_dim, max_norm_residual, device)


def _linear_warmup_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps <= 0 or step >= warmup_steps:
        return base_lr
    return base_lr * float(step + 1) / float(warmup_steps)


def train_variant(spec: VariantSpec, sizing_v2: "HH.RepairedSizing", sizing_v1: "HH.RepairedSizing",
                  feature_cfg: C.FeatureConfig, base_train_cfg: C.TrainingConfig, device, seed: int,
                  out_ckpt: Path) -> Tuple[Path, dict]:
    """Train one variant on the SAME pooled data (hrm_headtohead.pooled_scalar_arrays,
    i.e. hrm_headtohead.SCALAR_DATASETS -- the identical C7 avgbase pooled scalar
    files) with per-variant epochs/lr/warmup. Mirrors hrm_headtohead.train_repaired_model's
    loop (BalancedTaskDataset, AdamW, smooth-L1, grad-clip) exactly, generalized
    over the model-build call and an optional linear warmup schedule."""
    arrays = HH.pooled_scalar_arrays()
    ds = C.BalancedTaskDataset(arrays)
    loader = C.make_loader(ds, base_train_cfg.batch_size, shuffle=True, num_workers=base_train_cfg.num_workers)
    model = _build_model_for_variant(spec, sizing_v2, sizing_v1, feature_cfg.token_dim, base_train_cfg.max_norm_residual, device)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=base_train_cfg.weight_decay)
    C.set_global_seed(int(seed))
    history: List[dict] = []
    global_step = 0
    t_start = time.time()
    for epoch in range(1, spec.epochs + 1):
        model.train()
        losses: List[float] = []
        maes: List[float] = []
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            cur_lr = _linear_warmup_lr(global_step, spec.warmup_steps, spec.lr)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            if not torch.isfinite(pred).all():
                raise RuntimeError(f"[{spec.key}] nonfinite predictions")
            loss = F.smooth_l1_loss(pred, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"[{spec.key}] nonfinite loss")
            loss.backward()
            if base_train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), base_train_cfg.grad_clip)
            opt.step()
            global_step += 1
            losses.append(float(loss.item()))
            maes.append(float(torch.mean(torch.abs(pred.detach() - yb)).item()))
        row = {"epoch": epoch, "loss": C.finite_mean(losses), "mae": C.finite_mean(maes), "seconds": time.time() - t0, "lr": cur_lr}
        history.append(row)
        print(f"[{C.now_str()}] {spec.key} epoch {epoch}/{spec.epochs}: loss={row['loss']:.5f} mae={row['mae']:.5f} lr={cur_lr:.2e}", flush=True)
    total_seconds = time.time() - t_start
    train_cfg_used = replace(base_train_cfg, base_epochs=spec.epochs, lr=spec.lr)
    n_params = HH.count_params(model)
    payload = {
        "model": model.state_dict(),
        "sizing": asdict(sizing_v2 if spec.use_v2_backbone else sizing_v1),
        "feature_cfg": asdict(feature_cfg),
        "train_cfg": asdict(train_cfg_used),
        "variant_key": spec.key,
        "use_v2_backbone": spec.use_v2_backbone,
        "tasks": list(HH.SCALAR_DATASETS.keys()),
        "history": history,
        "seed": int(seed),
        "params": n_params,
        "total_train_seconds": total_seconds,
    }
    C.ensure_dir(out_ckpt.parent)
    torch.save(payload, out_ckpt)
    meta = {
        "variant": spec.key, "label": spec.label, "final_loss": history[-1]["loss"], "final_mae": history[-1]["mae"],
        "epochs": spec.epochs, "lr": spec.lr, "warmup_steps": spec.warmup_steps, "use_v2_backbone": spec.use_v2_backbone,
        "params": n_params, "total_train_seconds": total_seconds,
    }
    return out_ckpt, meta


def load_variant_provider(ckpt_path: Path, device) -> P.ScalarResidualProvider:
    payload = torch.load(ckpt_path, map_location="cpu")
    use_v2 = bool(payload["use_v2_backbone"])
    feature_cfg = C.FeatureConfig(**payload["feature_cfg"])
    train_cfg = C.TrainingConfig(**payload["train_cfg"])
    if use_v2:
        sizing = HH.RepairedSizing(**payload["sizing"])
        model = build_repaired_model_v2(sizing, feature_cfg.token_dim, train_cfg.max_norm_residual, device)
    else:
        sizing = HH.RepairedSizing(**payload["sizing"])
        model = HH.build_repaired_model(sizing, feature_cfg.token_dim, train_cfg.max_norm_residual, device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    prov = P.ScalarResidualProvider(model, feature_cfg, device, "hrmx_v2", train_cfg.max_norm_residual)
    prov.name = f"repaired_hrm_{payload['variant_key']}"
    return prov


# -----------------------------------------------------------------------------
# Eval: reuses hrm_headtohead.eval_provider_on_target / eval_arm verbatim (same
# 30-world, matched-budget, matched-exp-ratio-vs-euclid protocol as arm (a)/(b)).
# -----------------------------------------------------------------------------

def eval_variant(variant_key: str, ckpt_path: Path, device, targets: List[str] = EVAL_TARGETS) -> List[dict]:
    prov = load_variant_provider(ckpt_path, device)
    results = HH.eval_arm(f"repaired_hrm_{variant_key}", lambda target: prov, device, targets=targets)
    payload = torch.load(ckpt_path, map_location="cpu")
    for r in results:
        r["params"] = int(payload["params"])
        r["final_train_loss"] = float(payload["history"][-1]["loss"])
        r["variant"] = variant_key
    return results


# -----------------------------------------------------------------------------
# CLI orchestration
# -----------------------------------------------------------------------------

def ckpt_path_for(variant_key: str) -> Path:
    return OUT_DIR / "checkpoints" / f"repaired_hrm_{variant_key}.pt"


def run_train(variant_keys: List[str], device) -> Dict[str, dict]:
    base = None
    # Only need base's backbone_cfg/feature_cfg/train_cfg + incumbent param count;
    # arm-a incumbent model weights are not touched.
    import continuous_prm_c9_transfer as C9
    base = C9.load_source_base(HH.SOURCE_DIR, "hrm", torch.device("cpu"))
    incumbent_params = HH.count_params(base.model)
    sizing_v2 = pick_matched_sizing_v2(base.backbone_cfg, incumbent_params, base.feature_cfg.token_dim, base.train_cfg.max_norm_residual)
    sizing_v1 = HH.pick_matched_sizing(base.backbone_cfg, incumbent_params, base.feature_cfg.token_dim, base.train_cfg.max_norm_residual)
    print(f"[{C.now_str()}] incumbent_params={incumbent_params} sizing_v2={sizing_v2} sizing_v1(control)={sizing_v1}", flush=True)

    metas: Dict[str, dict] = {}
    for key in variant_keys:
        spec = VARIANTS[key]
        out_ckpt = ckpt_path_for(key)
        t0 = time.time()
        _, meta = train_variant(spec, sizing_v2, sizing_v1, base.feature_cfg, base.train_cfg, device, seed=HH.REPAIRED_SEED, out_ckpt=out_ckpt)
        meta["wall_seconds"] = time.time() - t0
        meta["params_ratio_to_incumbent"] = meta["params"] / max(1, incumbent_params)
        metas[key] = meta
        print(f"[{C.now_str()}] TRAIN DONE {key}: final_loss={meta['final_loss']:.5f} params={meta['params']} "
              f"(ratio={meta['params_ratio_to_incumbent']:.3f}) wall={meta['wall_seconds']:.1f}s", flush=True)
    C.write_json(OUT_DIR / "results" / "train_summary.json", {"incumbent_params": incumbent_params, "variants": metas})
    return metas


def run_eval(variant_keys: List[str], device) -> Dict[str, List[dict]]:
    all_results: Dict[str, List[dict]] = {}
    for key in variant_keys:
        ckpt = ckpt_path_for(key)
        if not ckpt.exists():
            print(f"[{C.now_str()}] SKIP eval {key}: checkpoint not found at {ckpt}", flush=True)
            continue
        t0 = time.time()
        results = eval_variant(key, ckpt, device)
        wall = time.time() - t0
        for r in results:
            r["eval_wall_seconds"] = wall
        all_results[key] = results
        C.write_json(OUT_DIR / "results" / f"eval_{key}.json", {"results": results})
        print(f"[{C.now_str()}] EVAL DONE {key} in {wall:.1f}s", flush=True)
    C.write_json(OUT_DIR / "results" / "eval_summary.json", all_results)
    return all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--variants", default="v1,v2,v3,v4", help="comma-separated subset of v1,v2,v3,v4")
    ap.add_argument("--eval-variants", default=None, help="override which variants to eval (default: same as --variants)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    C.ensure_dir(OUT_DIR / "results")

    variant_keys = [v.strip() for v in args.variants.split(",") if v.strip()]
    eval_keys = [v.strip() for v in args.eval_variants.split(",")] if args.eval_variants else variant_keys

    summary: dict = {}
    if args.mode in ("train", "all"):
        summary["train"] = run_train(variant_keys, device)
    if args.mode in ("eval", "all"):
        summary["eval"] = run_eval(eval_keys, device)

    C.write_json(OUT_DIR / "results" / "v2_summary.json", summary)
    print(f"[{C.now_str()}] hrm_headtohead_v2 {args.mode} done -> {OUT_DIR / 'results'}", flush=True)


if __name__ == "__main__":
    main()
