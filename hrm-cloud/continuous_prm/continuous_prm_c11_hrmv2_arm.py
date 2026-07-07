#!/usr/bin/env python3
"""C11 Task 6: HRM-v2 ACT arm -- model adaptation ONLY (the trainer +
providers are Task 7; the core registry hook into
`continuous_prm_c11_mission.py` is also Task 7). This module never imports
or modifies anything under `HRM-v2/` at module scope -- the whole
adaptation lives here, entirely via subclassing the installed `hrm` package
(`pip install -e ./HRM-v2 --no-deps`), imported lazily inside function
bodies so this module (and everything that imports it) stays importable on
a machine without `hrm` installed.

This is C11's sixth arm: the FULL HRM-v2 mechanism (hierarchical H/L
cycles, deep supervision per segment, ACT Q-learning halting) trained on
OUR substrate (product-graph heuristic regression) for the first time. The
other five arms (MLP control, FiLM U-Net, product-graph GNN, HRM-trace,
ON-LSTM-trace -- see `continuous_prm_c11_mission.py`) either have no
sequence mechanism at all or reuse the *trace-token backbone*
(`continuous_prm_common.ContinuousHeuristicModel`) without the paper's ACT
halting / deep-supervision training loop. This arm is the mechanism itself.

Three pieces, read together against `HRM-v2/src/hrm/models/hrm_act_v1.py`
and `HRM-v2/src/hrm/train/losses.py` (both READ IN FULL before writing this
file -- see the plan's Task 6 "READ FIRST" list):

1. `TraceHRMInner(HRMACTv1_Inner)` -- swaps the parent's token-ID embedding
   lookup (`embed_tokens(input.to(torch.int32))`) for a continuous linear
   projection (`trace_proj: Linear(token_dim, hidden_size)`) of FLOAT trace
   tokens (B, seq_len, token_dim) -- see `continuous_prm_c11_mission.py`'s
   `TRACE_TOKEN_LAYOUT` (dim 12) / `encode_trace_padded` (seq_max 10) for
   what these tokens look like; this module does not import that one (T6
   doesn't touch the mission module), it just matches its published
   contract. Everything downstream of `_input_embeddings` (H/L reasoning,
   q_head, lm_head, carries) is inherited UNCHANGED from `HRMACTv1_Inner`.

2. `TraceHRMACT(HRMACTv1)` -- the ACT wrapper, unchanged except it
   constructs a `TraceHRMInner` instead of `HRMACTv1_Inner`.
   `initial_carry`/`forward` are inherited untouched: they only ever touch
   `batch["inputs"]` for `.shape[0]` (batch size) and `.device`/dtype-like
   bookkeeping (`torch.empty_like`, `torch.where`) -- never an
   int-specific op (no embedding lookup happens outside
   `_input_embeddings`) -- confirmed by reading `HRMACTv1.initial_carry`
   and `HRMACTv1.forward` in full; float `inputs` flow through unmodified.

3. `RegressionACTLossHead(nn.Module)` -- mirrors `ACTLossHead`'s exact
   forward signature/return contract
   (`(carry, loss, metrics, outputs, all_finish)`, same
   `lm_loss + 0.5*(q_halt_loss + q_continue_loss)` composition, same
   per-sequence-style sum-reduction conventions) but swaps classification
   for regression: `lm_loss` -> `smooth_l1_loss(yhat, y, reduction="sum")`
   where `yhat = readout_yhat(outputs)` (the query-token, position-0,
   softplus-clamped scalar readout -- NOT mean-pooled; this is the
   remediation lesson from `program-audit-c11`'s hierarchy-vs-substrate
   finding: a degenerate seq-len-1 "blind" control tied because pooling
   erased structure, so readout must come from a SPECIFIC state token),
   and `seq_is_correct` -> `(yhat - y).abs() <= band` (pre-registered
   band = 0.1, side-length units, matching the residual-cap convention:
   `y` and `yhat` both live in `[0, cfg.residual_cap] = [0, 4.0]`).

Config: `hrmv2_config()` returns the exact dict `HRMACTv1Config` needs
(every required field the Pydantic model demands, read directly off
`HRMACTv1Config.model_fields` -- see the module-level comment above
`hrmv2_config` for the full field-by-field justification), pinned at the
plan's paper-faithful defaults (hidden=256, H/L=2/2 layers, H/L cycles=2/2,
halt_max_steps=8, exploration=0.1, learned positionals, float32 forward
dtype -- this machine has no `flash_attn`, so `hrm`'s `attention()` wrapper
falls through to its `sdpa()` path automatically for ANY non-CUDA or
non-fp16/bf16 tensor, confirmed by reading
`HRM-v2/src/hrm/ops/attention.py`: `is_suitable_dtype = q.dtype in
[float16, bfloat16]` is False for our float32 CPU tensors regardless of
device, so `flash_attention()` -- and therefore any `import flash_attn` --
is never even attempted).

See docs/superpowers/plans/2026-07-07-c11-mission.md (Task 6) and
docs/superpowers/specs/2026-07-07-c11-compositional-mission-design.md
(section 5) for the authoritative spec.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    # Only for type checkers / IDEs -- never executed, so this does not
    # violate the "no top-level `hrm` import" contract at runtime.
    from hrm.models.hrm_act_v1 import (
        HRMACTv1,
        HRMACTv1_Inner,
        HRMACTv1Config,
        HRMACTv1InnerCarry,
    )

_INSTALL_CMD = "pip install -e ./HRM-v2 --no-deps"


def _load_hrm():
    """Lazy import of the `hrm` symbols this module needs. Called from
    inside every function/method that touches `hrm` -- never at module
    scope, so `import continuous_prm_c11_hrmv2_arm` succeeds even when
    `hrm` is not installed. Returns a small namespace-like tuple of the
    exact symbols used below; raises the underlying ImportError to the
    caller (callers that want a friendlier message wrap this in
    `build_hrmv2_arm`)."""
    from hrm.models.hrm_act_v1 import (
        HRMACTv1,
        HRMACTv1_Inner,
        HRMACTv1Config,
        HRMACTv1InnerCarry,
    )

    return HRMACTv1, HRMACTv1_Inner, HRMACTv1Config, HRMACTv1InnerCarry


def build_hrmv2_arm(cfg: Optional[dict] = None) -> "TraceHRMACT":
    """Construct a `TraceHRMACT` from `hrmv2_config(cfg)`. Raises a clear
    RuntimeError naming the install command if `hrm` is not importable --
    the single user-facing entry point that should be called by code that
    doesn't want to handle ImportError itself (e.g. the eventual T7
    registry hook)."""
    try:
        _load_hrm()
    except ImportError as exc:
        raise RuntimeError(
            "the `hrm` package (HRM-v2) is not installed. Install it with:\n"
            f"    {_INSTALL_CMD}\n"
            f"(underlying error: {exc})"
        ) from exc

    config_dict = hrmv2_config(cfg)
    return TraceHRMACT(config_dict)


# ---------------------------------------------------------------------------
# hrmv2_config: the HRMACTv1Config dict for TraceHRMACT.
# ---------------------------------------------------------------------------
#
# Every field below is REQUIRED by `HRMACTv1Config` (a pydantic BaseModel)
# except where noted "has a pydantic default" -- confirmed by reading
# `HRMACTv1Config.model_fields` directly (all 18 fields; the model has no
# `extra="forbid"`/`"allow"` override, so only fields the class declares are
# meaningful):
#   REQUIRED (no pydantic default): batch_size, seq_len,
#     num_puzzle_identifiers, vocab_size, H_cycles, L_cycles, H_layers,
#     L_layers, hidden_size, num_heads, halt_max_steps,
#     halt_exploration_prob.
#   HAS PYDANTIC DEFAULT (pinned here anyway, for an explicit/reproducible
#     config rather than relying on the class's defaults silently):
#     puzzle_emb_ndim=0, expansion=4.0, pos_encodings="rope"->"learned"
#     (OVERRIDDEN per spec), rms_norm_eps=1e-5, rope_theta=10000.0 (kept at
#     its default -- irrelevant since pos_encodings="learned" never
#     constructs a RotaryEmbedding, see `HRMACTv1_Inner.__init__`'s
#     if/elif on `pos_encodings`), forward_dtype="bfloat16"->"float32"
#     (OVERRIDDEN: this machine trains/evals on CPU without flash_attn, and
#     float32 is the CPU-safe, parity-clean choice the sibling
#     ContinuousHeuristicModel arms also use).
#
# No field conflict: `seq_len=10` is the RAW trace length (query token +
# up to 9 leg tokens, per `TRACE_TOKEN_LAYOUT`/`encode_trace_padded`'s
# `cfg.seq_max=10`); `puzzle_emb_ndim=0` forces `puzzle_emb_len =
# ceil(0 / hidden_size) = 0` (see `HRMACTv1_Inner.__init__`:
# `-(0 // -hidden_size) == 0`), so `total_seq_len = seq_len +
# puzzle_emb_len == 10` exactly -- no puzzle-embedding prefix shifts the
# sequence, matching the plan's "puzzle_emb_len == 0 path" instruction.
# `vocab_size=1`: the lm_head projects hidden_size -> 1, i.e. ONE scalar
# channel per position (not a token-classification vocabulary) -- this is
# the regression readout channel, read out at position 0 only (see
# `readout_yhat`).
def hrmv2_config(cfg: Optional[dict] = None, **overrides: Any) -> dict:
    """Return the config dict for `TraceHRMACT` / `HRMACTv1Config`. `cfg`
    (if given) is merged first, then `overrides` (kwargs) on top -- both
    optional, both take precedence over the pinned defaults below in the
    order given, so `hrmv2_config(halt_exploration_prob=0.0)` and
    `hrmv2_config({"halt_exploration_prob": 0.0})` both work."""
    config_dict: Dict[str, Any] = {
        # Data dimensions.
        "batch_size": 64,
        "seq_len": 10,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "vocab_size": 1,
        # Hierarchical cycles.
        "H_cycles": 2,
        "L_cycles": 2,
        # Layers.
        "H_layers": 2,
        "L_layers": 2,
        # Transformer architecture.
        "hidden_size": 256,
        "expansion": 4.0,
        "num_heads": 8,
        "pos_encodings": "learned",
        # Normalization / RoPE (rope_theta unused under "learned" but kept
        # explicit at its class default for a fully-pinned config).
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        # ACT halting.
        "halt_max_steps": 8,
        "halt_exploration_prob": 0.1,
        # Precision.
        "forward_dtype": "float32",
    }
    if cfg:
        config_dict.update(cfg)
    if overrides:
        config_dict.update(overrides)
    return config_dict


# ---------------------------------------------------------------------------
# TraceHRMInner: continuous-input subclass of HRMACTv1_Inner.
# ---------------------------------------------------------------------------

class TraceHRMInner(nn.Module):
    """Subclass of `hrm.models.hrm_act_v1.HRMACTv1_Inner` (base resolved
    lazily at __init__ time via `_load_hrm()`, since class bases can't be
    named at module scope without a top-level `hrm` import). Replaces the
    parent's discrete token-ID embedding path with a continuous linear
    projection of float trace tokens; everything else (H/L reasoning,
    q_head, lm_head, carries) is inherited from the parent unchanged.

    NOTE ON THE BASE-CLASS TRICK: Python requires a class's bases to be
    resolvable at class-definition time, which conflicts with "no
    top-level `hrm` import." We resolve this the standard lazy-subclass
    way: `TraceHRMInner` is defined once, lazily, the first time it's
    needed, and cached at module scope under this same name via
    `_get_trace_hrm_inner_cls()`. Callers should use the module-level
    `TraceHRMInner` name (see the bottom of this section, where it's bound
    to the lazily-built class via a module `__getattr__` hook) -- from the
    outside this is indistinguishable from an ordinary class.
    """


def _build_trace_hrm_inner_cls():
    _, HRMACTv1_Inner, _, _ = _load_hrm()

    class _TraceHRMInner(HRMACTv1_Inner):
        """Continuous-input `HRMACTv1_Inner`: `_input_embeddings` accepts
        FLOAT trace tokens (B, seq_len, token_dim) instead of int token
        IDs, projecting them into hidden_size via a learned linear layer
        (`trace_proj`) instead of an embedding-table lookup. The parent's
        positional-encoding arithmetic (`pos_encodings == "learned"` path,
        `puzzle_emb_len == 0`) is replicated EXACTLY -- same scale
        constant (0.707106781 = 1/sqrt(2)), same operation order (add pos
        THEN scale by embed_scale) -- copied line-for-line from
        `HRMACTv1_Inner._input_embeddings`, not reconstructed from memory."""

        def __init__(self, config, token_dim: int = 12):
            super().__init__(config)
            self.token_dim = int(token_dim)

            # `trace_proj`: Linear(token_dim, hidden_size). Init std mirrors
            # the parent's embedding-scale convention: `CastedEmbedding`
            # (what `embed_tokens`/`embed_pos` use) is truncated-normal
            # initialized at `std = 1/embed_scale = 1/sqrt(hidden_size)`
            # (see `HRMACTv1_Inner.__init__`: `embed_init_std = 1.0 /
            # self.embed_scale`). We mirror that SPIRIT -- keep the
            # projected embedding's initial variance in the same ballpark
            # as the token/positional embeddings it's added to/replaces --
            # by trunc-normal-initializing `trace_proj.weight` at the same
            # `std = 1/sqrt(hidden_size)` (via `hrm`'s own
            # `trunc_normal_init_` helper, the same one `CastedEmbedding`
            # and `CastedLinear` use, rather than PyTorch's variance-losing
            # `nn.init.trunc_normal_`). Bias is zero-initialized (matching
            # `CastedLinear`'s own bias convention).
            from hrm.utils.init import trunc_normal_init_

            self.trace_proj = nn.Linear(self.token_dim, config.hidden_size, bias=True)
            with torch.no_grad():
                trunc_normal_init_(self.trace_proj.weight, std=1.0 / math.sqrt(config.hidden_size))
                self.trace_proj.bias.zero_()

        def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
            """`input` is FLOAT (B, seq_len, token_dim) -- NOT int token
            IDs. Replicates `HRMACTv1_Inner._input_embeddings`'s body
            EXACTLY except swapping the `embed_tokens` lookup for
            `trace_proj`, and (per this arm's config: `puzzle_emb_ndim=0`)
            omitting the puzzle-embedding branch (dead code here since
            `self.config.puzzle_emb_ndim > 0` is always False for this
            arm's config; the branch is left out rather than kept-but-
            unreachable, since replicating dead code for its own sake adds
            confusion, not fidelity -- the parent's ACTUAL puzzle-emb
            control flow is exercised nowhere in this arm regardless of
              whether we paste the branch in)."""
            embedding = self.trace_proj(input.to(self.forward_dtype))

            # Position embeddings -- copied verbatim from the parent's
            # "learned" branch: SAME scale constant (0.707106781 =
            # 1/sqrt(2), "to maintain forward variance" per the parent's
            # own comment) and SAME operation order (embedding + pos,
            # THEN multiply by the sqrt(2)-compensating constant, THEN by
              # embed_scale below) -- not reconstructed, copied.
            if self.config.pos_encodings == "learned":
                embedding = 0.707106781 * (
                    embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
                )

            # Scale embeddings -- identical final line to the parent.
            return self.embed_scale * embedding

    _TraceHRMInner.__name__ = "TraceHRMInner"
    _TraceHRMInner.__qualname__ = "TraceHRMInner"
    return _TraceHRMInner


_TRACE_HRM_INNER_CLS_CACHE: Optional[type] = None


def _get_trace_hrm_inner_cls() -> type:
    global _TRACE_HRM_INNER_CLS_CACHE
    if _TRACE_HRM_INNER_CLS_CACHE is None:
        _TRACE_HRM_INNER_CLS_CACHE = _build_trace_hrm_inner_cls()
    return _TRACE_HRM_INNER_CLS_CACHE


class _TraceHRMInnerFactory:
    """Callable placeholder bound to the module-level name `TraceHRMInner`
    (see the `TraceHRMInner = _TraceHRMInnerFactory()` assignment below).
    `TraceHRMInner(config, token_dim=12)` builds (lazily, on first overall
    use, then cached) the real subclass and instantiates it, so callers use
    `TraceHRMInner` exactly like an ordinary class -- `isinstance(x,
    TraceHRMInner)` also works via `__instancecheck__`."""

    def __call__(self, config, token_dim: int = 12):
        cls = _get_trace_hrm_inner_cls()
        return cls(config, token_dim=token_dim)

    def __instancecheck__(self, instance):
        cls = _get_trace_hrm_inner_cls()
        return isinstance(instance, cls)


# Rebind the placeholder class defined above to the callable factory -- from
# here on, `TraceHRMInner` (the module attribute) is the factory instance,
# not the empty placeholder class. `isinstance(x, TraceHRMInner)` works via
# `__instancecheck__` because `TraceHRMInner`'s TYPE (`_TraceHRMInnerFactory`)
# defines it, which Python's `isinstance` honors for any right-hand operand
# object (not just classes).
TraceHRMInner = _TraceHRMInnerFactory()  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# TraceHRMACT: ACT wrapper constructing TraceHRMInner instead of
# HRMACTv1_Inner.
# ---------------------------------------------------------------------------

def _build_trace_hrm_act_cls():
    HRMACTv1, _, HRMACTv1Config, _ = _load_hrm()
    trace_inner_cls = _get_trace_hrm_inner_cls()

    class _TraceHRMACT(HRMACTv1):
        """Subclass of `hrm.models.hrm_act_v1.HRMACTv1` that constructs a
        `TraceHRMInner` instead of the parent's `HRMACTv1_Inner`.
        `initial_carry` and `forward` (the ACT halting logic) are inherited
        UNCHANGED -- read in full: both only ever touch `batch["inputs"]`
        for `.shape[0]` (batch size, `initial_carry`) or via
        dtype/device-agnostic ops (`torch.empty_like`, `torch.where`,
        `forward`'s `new_current_data` construction) -- never an
        int-specific op (no embedding lookup happens anywhere in
        `HRMACTv1.forward`/`initial_carry`; the ONLY embedding lookup in
        the whole model is `HRMACTv1_Inner._input_embeddings`'s
        `embed_tokens(input.to(torch.int32))` line, which THIS subclass's
        inner (`TraceHRMInner`) already overrides). So float `inputs` flow
        through the parent's carry/halting bookkeeping with no
        modification needed -- confirmed by reading, not asserted."""

        def __init__(self, config_dict: dict, token_dim: int = 12):
            # nn.Module.__init__ (skip HRMACTv1.__init__, which would
            # build the wrong inner class) -- mirrors HRMACTv1.__init__'s
            # own body but swaps HRMACTv1_Inner for TraceHRMInner.
            nn.Module.__init__(self)
            self.config = HRMACTv1Config(**config_dict)
            self.inner = trace_inner_cls(self.config, token_dim=token_dim)

    _TraceHRMACT.__name__ = "TraceHRMACT"
    _TraceHRMACT.__qualname__ = "TraceHRMACT"
    return _TraceHRMACT


_TRACE_HRM_ACT_CLS_CACHE: Optional[type] = None


def _get_trace_hrm_act_cls() -> type:
    global _TRACE_HRM_ACT_CLS_CACHE
    if _TRACE_HRM_ACT_CLS_CACHE is None:
        _TRACE_HRM_ACT_CLS_CACHE = _build_trace_hrm_act_cls()
    return _TRACE_HRM_ACT_CLS_CACHE


class _TraceHRMACTFactory:
    """Same lazy-factory pattern as `_TraceHRMInnerFactory` above, bound to
    the module-level name `TraceHRMACT`."""

    def __call__(self, config_dict: dict, token_dim: int = 12):
        cls = _get_trace_hrm_act_cls()
        return cls(config_dict, token_dim=token_dim)

    def __instancecheck__(self, instance):
        cls = _get_trace_hrm_act_cls()
        return isinstance(instance, cls)


TraceHRMACT = _TraceHRMACTFactory()  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# readout_yhat: scalar readout convention.
# ---------------------------------------------------------------------------

def readout_yhat(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """`torch.clamp(F.softplus(outputs["logits"][:, 0, 0]), 0.0, 4.0)`.

    Position 0 of the trace sequence is ALWAYS the query token (see
    `TRACE_TOKEN_LAYOUT` / `encode_trace` in `continuous_prm_c11_mission.py`:
    token 0 = `[x/side, y/side, 8 rays, s/k_max, k_remaining/k_max]`, i.e.
    the state (i, s) being queried) -- so reading the scalar prediction off
    position 0's lm_head channel is a STATE-TOKEN readout, not a mean-pool
    over the whole sequence. This is the remediation lesson from the
    hierarchy-vs-substrate program audit (MEMORY `program-audit-c11`): a
    degenerate seq-len-1 "blind" MLP control TIED with the hierarchical
    arms in an earlier formulation partly because mean-pooling across a
    24-token feature bag erased the very sequence structure the hierarchy
    was supposed to exploit. Reading a SPECIFIC, semantically-anchored
    token (the query state itself) instead of pooling keeps the readout
    tied to what's actually being asked ("what's the residual heuristic
    value AT this product-graph state"), matching every other arm's
    per-state prediction contract.

    `vocab_size=1`, so `logits[:, 0, 0]` is exactly the model's single
    scalar output channel at the query position -- shape (B,). Softplus
    keeps the raw logit non-negative (residuals are non-negative by
    admissibility, per the spec's target construction:
    `y = clip((oracle - hl) / side_len, 0, cap)`, `y >= 0` before
    clipping); clamp to `[0, cap=4.0]` matches the SAME
    `residual_cap`/`max_norm_residual` convention every other C11 arm uses
    (`continuous_prm_c11_mission._softplus_clamp` /
    `continuous_prm_common.ContinuousHeuristicModel.forward`)."""
    return torch.clamp(F.softplus(outputs["logits"][:, 0, 0]), 0.0, 4.0)


def regression_correct(yhat: torch.Tensor, y: torch.Tensor, band: float = 0.1) -> torch.Tensor:
    """`(yhat.detach() - y).abs() <= band` -- the pre-registered
    correctness band (0.1, side-length units) used both by
    `RegressionACTLossHead` (q_halt BCE target) and directly by tests."""
    return (yhat.detach() - y).abs() <= band


# ---------------------------------------------------------------------------
# RegressionACTLossHead: mirrors ACTLossHead's structure for regression.
# ---------------------------------------------------------------------------

def _build_regression_act_loss_head_cls():
    """Lazily build the `RegressionACTLossHead` class. `nn.Module` itself
    needs no lazy import (torch is a hard top-level dependency of this
    module already), so unlike `TraceHRMInner`/`TraceHRMACT` this class
    doesn't need a `hrm`-derived base class -- it's built lazily anyway,
    for symmetry and because its docstring/behavior is defined entirely in
    terms of mirroring `hrm.train.losses.ACTLossHead`'s structure (read in
    full as part of this task), which is worth confirming is importable at
    the point this class is actually used."""
    _load_hrm()  # Confirms `hrm` is present; symbols not otherwise needed.

    class _RegressionACTLossHead(nn.Module):
        """Mirrors `hrm.train.losses.ACTLossHead`'s structure (forward
        signature `(return_keys, **model_kwargs) -> (carry, loss, metrics,
        outputs, all_finish)`, same `lm_loss + 0.5*(q_halt_loss +
        q_continue_loss)` composition, same sum-reduction /
        detach-for-metrics conventions -- read `ACTLossHead.forward` in
        full and this implementation copies its control flow line-for-line
        except for exactly these substitutions:

        - `lm_loss` (classification cross-entropy summed then divided by a
          per-sequence `loss_divisor`) -> `F.smooth_l1_loss(yhat, y,
          reduction="sum")` where `yhat = readout_yhat(outputs)`. There is
          no `loss_divisor` here: the original's divisor exists because
          classification losses are per-TOKEN (divided by the number of
          valid label tokens per sequence before summing, so multi-token
          sequences don't dominate); our regression target is a single
          SCALAR per sequence (one residual value per product-graph
          state), so there is nothing to divide by within a sequence --
          the `reduction="sum"` sums over the batch exactly the way the
          original's `(... / loss_divisor).sum()` sums over the batch
          AFTER its own per-sequence normalization. The T7 trainer applies
          the SAME batch-level scaling the original training loop applies
          (`(loss / batch_size).backward()`), so this sum-over-batch
          convention composes identically.
        - `seq_is_correct` -> `regression_correct(yhat, y, band)` (default
          band 0.1, side-length units, pre-registered).
        - `q_halt_loss` / `q_continue_loss`: IDENTICAL structure and 0.5
          weighting to the original -- same
          `binary_cross_entropy_with_logits(..., reduction="sum")` calls,
          same `target_q_continue` bootstrap consumption, only the
          correctness SOURCE differs (regression band vs. exact-match).
        - Metrics dict: keeps `count`, `q_halt_accuracy`, `steps` from the
          original (they generalize as-is to a scalar-target regression
          setting) and adds `mae` (mean absolute error, SAME sum-then-
          divide-by-batch-size convention the trainer applies to the main
          loss, i.e. reported as a raw batch-sum here -- consistent with
          how `lm_loss`/`q_halt_loss` are also reported as raw sums, not
          pre-divided, in the original `metrics.update(...)` block)."""

        def __init__(self, model: nn.Module, band: float = 0.1):
            super().__init__()
            self.model = model
            self.band = float(band)

        def initial_carry(self, *args, **kwargs):
            return self.model.initial_carry(*args, **kwargs)  # type: ignore[attr-defined]

        def forward(
            self,
            return_keys,
            **model_kwargs,
        ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
            new_carry, outputs = self.model(**model_kwargs)
            y = new_carry.current_data["y"]

            yhat = readout_yhat(outputs)

            with torch.no_grad():
                correct = regression_correct(yhat, y, band=self.band)

                # Metrics (halted) -- same `valid_metrics` gating as the
                # original (only sequences that have actually halted this
                # step contribute to the reported metrics).
                valid_metrics = new_carry.halted
                metrics = {
                    "count": valid_metrics.sum(),
                    "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == correct)).sum(),
                    "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
                    "mae": torch.where(valid_metrics, (yhat - y).abs(), torch.zeros_like(y)).sum(),
                }

            # Losses.
            lm_loss = F.smooth_l1_loss(yhat, y, reduction="sum")
            q_halt_loss = F.binary_cross_entropy_with_logits(
                outputs["q_halt_logits"], correct.to(outputs["q_halt_logits"].dtype), reduction="sum"
            )

            metrics.update({
                "lm_loss": lm_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            })

            q_continue_loss = 0
            if "target_q_continue" in outputs:
                q_continue_loss = F.binary_cross_entropy_with_logits(
                    outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum"
                )
                metrics["q_continue_loss"] = q_continue_loss.detach()

            detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

            return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()

    _RegressionACTLossHead.__name__ = "RegressionACTLossHead"
    _RegressionACTLossHead.__qualname__ = "RegressionACTLossHead"
    return _RegressionACTLossHead


_REGRESSION_ACT_LOSS_HEAD_CLS_CACHE: Optional[type] = None


def _get_regression_act_loss_head_cls() -> type:
    global _REGRESSION_ACT_LOSS_HEAD_CLS_CACHE
    if _REGRESSION_ACT_LOSS_HEAD_CLS_CACHE is None:
        _REGRESSION_ACT_LOSS_HEAD_CLS_CACHE = _build_regression_act_loss_head_cls()
    return _REGRESSION_ACT_LOSS_HEAD_CLS_CACHE


class _RegressionACTLossHeadFactory:
    """Same lazy-factory pattern as the two above, bound to the
    module-level name `RegressionACTLossHead`."""

    def __call__(self, model: nn.Module, band: float = 0.1):
        cls = _get_regression_act_loss_head_cls()
        return cls(model, band=band)

    def __instancecheck__(self, instance):
        cls = _get_regression_act_loss_head_cls()
        return isinstance(instance, cls)


RegressionACTLossHead = _RegressionACTLossHeadFactory()  # type: ignore[assignment,misc]


__all__ = [
    "build_hrmv2_arm",
    "hrmv2_config",
    "TraceHRMInner",
    "TraceHRMACT",
    "readout_yhat",
    "regression_correct",
    "RegressionACTLossHead",
]
