#!/usr/bin/env python3
"""C11 Task 6: HRM-v2 ACT arm -- model adaptation (`TraceHRMInner`,
`TraceHRMACT`, `readout_yhat`, `RegressionACTLossHead`). Task 7 (this file,
extended): the faithful per-segment trainer (`train_hrmv2`), the ACT-live
and forced-k eval providers (`_hrmv2_forward_batch_act`,
`_hrmv2_forward_batch_forced_k`), and the registration hook
(`register_hrmv2_providers`) that the core module
(`continuous_prm_c11_mission.py`) calls, post-import, to wire this arm into
the shared provider registry. This module never imports or modifies
anything under `HRM-v2/` at module scope -- the whole adaptation lives
here, entirely via subclassing the installed `hrm` package (`pip install -e
./HRM-v2 --no-deps`), imported lazily inside function bodies so this module
(and everything that imports it) stays importable on a machine without
`hrm` installed. T7 additionally uses `continuous_prm_c11_mission`'s
encoders (`encode_trace_padded`) and dataset types (`WorldBundle`,
`stack_targets`) -- per the SAME isolation discipline, that import is ALSO
lazy (inside `train_hrmv2`/the provider functions), never at module scope,
so this module still imports cleanly standalone. The two lazy-import
targets (`hrm`, `continuous_prm_c11_mission`) are independent: either can
be absent without breaking the other's import.

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

See docs/experiments/continuous/c11/plans/2026-07-07-c11-mission.md (Tasks 6-7) and
docs/experiments/continuous/c11/design/2026-07-07-c11-compositional-mission-design.md
(section 5) for the authoritative spec.
"""
from __future__ import annotations

import copy
import math
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    # Only for type checkers / IDEs -- never executed, so this does not
    # violate the "no top-level `hrm`" / "no top-level
    # `continuous_prm_c11_mission`" import contracts at runtime.
    from hrm.models.hrm_act_v1 import (
        HRMACTv1,
        HRMACTv1_Inner,
        HRMACTv1Config,
        HRMACTv1InnerCarry,
    )
    from continuous_prm_c11_mission import C11MissionConfig, WorldBundle

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
    `TraceHRMInner` name -- see the bottom of this section, where the name
    `TraceHRMInner` is REBOUND (a plain module-level assignment,
    `TraceHRMInner = _TraceHRMInnerFactory()`) from this placeholder class
    to an INSTANCE of `_TraceHRMInnerFactory`, a small callable class whose
    `__call__` builds-and-instantiates the real lazy subclass and whose
    `__instancecheck__` makes `isinstance(x, TraceHRMInner)` work against
    it. The mechanism is factory-INSTANCE rebinding of the module
    attribute, not a module-level `__getattr__` hook (Python 3.7+'s
    PEP 562 mechanism, which this module does NOT use) -- from the
    outside, calling `TraceHRMInner(config, token_dim=12)` looks
    indistinguishable from an ordinary class either way, but the
    implementation is "the name now points at a callable object," not "a
    module-level `__getattr__` intercepts attribute lookups."
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
            # T6-review minor: `_input_embeddings` below deliberately OMITS
            # the parent's puzzle-embedding branch (see that method's own
            # docstring) rather than keeping it dead-but-present -- correct
            # ONLY when `puzzle_emb_ndim == 0` (so `puzzle_emb_len == 0` and
            # no puzzle-embedding prefix is ever expected in the sequence).
            # A future config with `puzzle_emb_ndim > 0` would silently
            # produce a shape-shifted sequence (the parent's own forward
            # expects `total_seq_len = seq_len + puzzle_emb_len`) with no
            # error until some downstream shape mismatch -- this assert
            # fails LOUDLY and immediately at construction time instead.
            assert config.puzzle_emb_ndim == 0, (
                f"TraceHRMInner requires puzzle_emb_ndim == 0 (got "
                f"{config.puzzle_emb_ndim}): _input_embeddings omits the "
                f"parent's puzzle-embedding branch entirely, which is only "
                f"correct when puzzle embeddings are disabled -- see "
                f"_input_embeddings's docstring for why the branch is left "
                f"out rather than kept-but-unreachable."
            )
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
        # T6-review minor: `hrm` absent means `_get_trace_hrm_inner_cls()`
        # raises ImportError (via `_load_hrm()`) -- an `isinstance` check
        # must not propagate that as a crash. Without `hrm` installed,
        # nothing in the process could possibly BE a `TraceHRMInner`
        # (the class itself couldn't have been built), so `False` is the
        # semantically correct answer, not just a safe fallback.
        try:
            cls = _get_trace_hrm_inner_cls()
        except ImportError:
            return False
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
        # T6-review minor: see `_TraceHRMInnerFactory.__instancecheck__`'s
        # comment -- `hrm` absent must yield `False`, not an ImportError
        # crash, from an `isinstance` check.
        try:
            cls = _get_trace_hrm_act_cls()
        except ImportError:
            return False
        return isinstance(instance, cls)


TraceHRMACT = _TraceHRMACTFactory()  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# readout_yhat: scalar readout convention.
# ---------------------------------------------------------------------------

def readout_yhat(outputs: Dict[str, torch.Tensor], cap: float = 4.0) -> torch.Tensor:
    """`torch.clamp(F.softplus(outputs["logits"][:, 0, 0]), 0.0, cap)`.

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
    clipping); clamp to `[0, cap]` (default 4.0, T6's original hardcoded
    value -- now a parameter per the T6 review: `cap` should be THREADABLE
    from `cfg.residual_cap` rather than hardcoded, so a future
    `residual_cap` change cannot silently desync this arm's readout from
    the other 5 arms' `_softplus_clamp(cap=cfg.residual_cap)` convention.
    T7's providers pass `cap=cfg.residual_cap` explicitly;
    `RegressionACTLossHead` keeps the 4.0 default, matching
    `hrmv2_config()`'s own pinned defaults, since the loss head has no
    `cfg` object to thread a value from) matches the SAME
    `residual_cap`/`max_norm_residual` convention every other C11 arm uses
    (`continuous_prm_c11_mission._softplus_clamp` /
    `continuous_prm_common.ContinuousHeuristicModel.forward`)."""
    return torch.clamp(F.softplus(outputs["logits"][:, 0, 0]), 0.0, float(cap))


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
          setting) and adds `mae`: `sum(|yhat - y|)` over ONLY the rows
          that halted THIS segment (`torch.where(valid_metrics, ...,
          0).sum()`, `valid_metrics = new_carry.halted`), reported as a
          raw sum here -- consistent with how `lm_loss`/`q_halt_loss` are
          also reported as raw (unnormalized) sums in the original
          `metrics.update(...)` block, never pre-divided inside the loss
          head. T6-review fix: the divisor to recover a true per-row mean
          from this raw sum is `metrics["count"]` (the number of ROWS that
          halted this segment, i.e. `valid_metrics.sum()` -- the SAME
          halted-gated count `q_halt_accuracy`/`steps` are also gated by),
          NOT `batch_size`: a mid-training segment typically has only SOME
          rows halt (the rest continue to the next segment), so `count <=
          batch_size` in general and dividing by `batch_size` would
          under-count the true mean whenever `count < batch_size`. Callers
          that want a per-row MAE must compute `mae / count.clamp_min(1)`
          themselves (mirroring how `accuracy`/`exact_accuracy`/
          `q_halt_accuracy` are similarly divided by `count` OUTSIDE the
          loss head, e.g. in `evaluate()`'s accumulation loop in
          `train_maze_optimized.py`)."""

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
        # T6-review minor: see `_TraceHRMInnerFactory.__instancecheck__`'s
        # comment -- `hrm` absent must yield `False`, not an ImportError
        # crash, from an `isinstance` check.
        try:
            cls = _get_regression_act_loss_head_cls()
        except ImportError:
            return False
        return isinstance(instance, cls)


RegressionACTLossHead = _RegressionACTLossHeadFactory()  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Task 7: lazy import of the core mission module (encoders + dataset types).
# ---------------------------------------------------------------------------

def _load_mission():
    """Lazy import of `continuous_prm_c11_mission` -- called from inside
    every T7 function that needs `encode_trace_padded`/`stack_targets`/
    `WorldBundle`/`C11MissionConfig`, never at module scope, so this module
    keeps importing cleanly even when the core module (or, transitively,
    `hrm-cloud`'s other C-series dependencies) is unavailable for some
    reason. In the NORMAL case (the core module is what imports THIS
    module, via its `register_hrmv2_providers()` hook), by the time any T7
    function actually runs, `continuous_prm_c11_mission` is already fully
    loaded and sitting in `sys.modules` -- this is a plain cached-module
    lookup, not a re-execution of that module's top level -- so there is no
    circular-import deadlock: the core module's own top-level `import
    continuous_prm_c11_hrmv2_arm as _HA; _HA.register_hrmv2_providers()`
    only REGISTERS callables (never calls `train_hrmv2`/a provider
    function immediately), and those callables' own bodies don't run until
    long after both modules have finished importing."""
    import continuous_prm_c11_mission as _M

    return _M


# ---------------------------------------------------------------------------
# Task 7: train_hrmv2 -- the faithful per-segment trainer.
# ---------------------------------------------------------------------------
#
# RECIPE SPLIT (spec section 5's "recipe deviation, accepted and flagged" --
# this is documented here, at the trainer itself, not just in the spec,
# since it is THE thing under test by this arm):
#
#   MATCHED to the native `train_arm` recipe (arms 1-4), so this arm's data
#   diet and epoch budget are directly comparable to theirs:
#     - Data: `stack_targets(bundles)` -- the SAME pooled (i, s, y,
#       bundle_idx) rows every native arm trains on, from the SAME
#       `WorldBundle`s (`run_train` collects bundles once per cell and
#       reuses them across every arm touching that cell, hrmv2 included).
#     - Epochs: `cfg.epochs` (40, overridable via the `epochs` param) --
#       same default as the native recipe.
#     - Shuffle: the SAME per-epoch `np.random.default_rng` shuffle,
#       seeded identically (`10007*(seed+1) + 101*config_idx + K`, via
#       `_training_seed`-equivalent arithmetic reproduced here since this
#       module doesn't import `continuous_prm_c11_mission._training_seed`
#       -- a private helper of that module -- but the FORMULA itself is
#       public per the plan's "Matched recipe" section and reproduced
#       verbatim below).
#     - Loss target: smooth-L1 on ŷ vs y (the loss the native recipe also
#       optimizes) -- BUT computed by `RegressionACTLossHead`'s own
#       `lm_loss` term (`reduction="sum"`, not the native trainer's
#       `reduction="mean"` via `F.smooth_l1_loss(preds, y_batch,
#       beta=cfg.smooth_l1_beta)` with PyTorch's default `beta=1.0` --
#       `RegressionACTLossHead` is deliberately NOT parameterized by
#       `cfg.smooth_l1_beta`, since it mirrors `ACTLossHead`'s own
#       hardcoded-beta convention faithfully rather than threading a
#       recipe knob through the loss head -- the two betas coincide at
#       their shared default of 1.0 in this arm's actual runs, so this is
#       a documentation-only distinction, not a live divergence, UNLESS
#       `cfg.smooth_l1_beta` is ever overridden away from 1.0 for the
#       native arms, in which case this arm's loss would silently NOT
#       track that override -- flagged here rather than hidden).
#
#   PAPER-FAITHFUL (deliberately UNMATCHED to the native recipe -- this
#   deviation IS the arm; it is the mechanism under test, not a bug):
#     - Optimizer: `AdamATan2` (`hrm.train.optim.AdamATan2`, scale-invariant
#       atan2-based update) instead of `AdamW` -- the paper's own optimizer,
#       per `train_maze_optimized.py`.
#     - LR schedule: warmup-then-constant (`warmup_constant_lr`, 100-step
#       warmup) instead of the native recipe's flat LR -- also per
#       `train_maze_optimized.py` (whose own `warmup_steps=500` is a
#       maze-scale value; 100 is this arm's own choice, small relative to
#       the total step count of a 40-epoch run over even a handful of
#       bundles, since HRM-v2 batches are much smaller in row-count terms
#       than the maze's ~48k-step budget).
#     - Grad-clip: NONE (the native recipe's `cfg.grad_clip=1.0` via
#       `clip_grad_norm_` is a native-recipe-only knob; the paper's
#       training script has no grad-clip call, so none is added here --
#       another faithfulness choice, not an oversight).
#     - THE SEGMENT LOOP ITSELF: one optimizer step PER ACT SEGMENT (deep
#       supervision), not one step per batch. This is the paper's actual
#       training mechanism and the reason this arm exists at all -- see
#       the loop body below.
#
#   DOCUMENTED DEVIATION FROM THE LITERAL STREAMING SCRIPT
#   (`train_maze_optimized.py`'s `main()`): that script keeps ONE carry
#   PERSISTENT ACROSS THE ENTIRE TRAINING RUN (created once before the
#   first batch, threaded through every subsequent batch via the
#   dataloader's `for batch_idx, batch in enumerate(pbar)` loop, reset only
#   per-ROW via the model's own `reset_carry(carry.halted, ...)` whenever
#   that row individually halts -- so different rows in the same tensor can
#   be mid-episode on wildly different data at any given step). This
#   trainer instead creates a FRESH carry once PER BATCH (`carry =
#   model.initial_carry(batch)`), then loops segments until every row has
#   COMPLETED at least one ACT episode on that batch (the `ever_halted`
#   DRAIN CONDITION -- see the loop body and `_PER_BATCH_SEGMENT_CAP`'s
#   history comment for why the naive "until all rows are halted on the
#   same call" reading of the spec's `all_finish` is unreachable in
#   practice) before moving to the next -- i.e. every row in a batch
#   STARTS its episode together (batch-synchronized fresh carry), every
#   row COMPLETES >= 1 full episode per batch (bounded at halt_max_steps
#   segments by `is_last_step`), and rows that halt early may begin a
#   partial second episode on the same data before the batch drains
#   (their extra segments contribute normal gradient updates, exactly as
#   the streaming script's persistent carry would also have them do).
#   This matches the epoch/shuffle/batch STRUCTURE the other 5 arms use
#   (so "40 epochs over the pooled TRAIN rows, shuffled per epoch" means
#   the same thing for every arm) rather than the maze script's single
#   unbounded `IterableDataset` stream. This is a DELIBERATE,
#   PRE-REGISTERED deviation from the literal script (per spec section 5:
#   "the segmented training loop [is] paper-faithful" refers to the
#   SEGMENT MECHANISM -- one optimizer step per segment, carry persisting
#   ACROSS SEGMENTS within an episode -- not to the streaming script's
#   specific cross-BATCH carry lifetime, which is an artifact of that
#   script's unbounded-stream data loader, orthogonal to the mechanism
#   under test here). The mechanism that actually matters for G2 (does
#   deep supervision / ACT halting help) -- one gradient update per
#   segment, real multi-segment episodes, live Q-learning halting during
#   training -- is preserved exactly; only the batch-boundary bookkeeping
#   is adapted to fit this phase's shared epoch/shuffle/batch harness.
#
# PARTIAL FINAL BATCH: dropped (`drop_last`-equivalent), per the spec's
# explicit choice among the offered alternatives ("dropping <=1023 of ~50k
# rows/epoch is cleaner"). No padding-by-cycling, no validity mask.
#
# CONFIG BATCH_SIZE vs ACTUAL BATCH SIZE: `hrmv2_config()`'s
# `batch_size=64` field is NOT threaded into the actual per-step tensor
# size. Verified directly (not assumed) by running a 200-row batch through
# `model.initial_carry`/`model.forward` with `config.batch_size=64` still
# set: both derive every shape purely from `batch["inputs"].shape[0]`
# (`HRMACTv1.initial_carry`: `batch_size = batch["inputs"].shape[0]`;
# `HRMACTv1_Inner.empty_carry(batch_size)` takes that value as a plain
# argument) -- `config.batch_size` is read NOWHERE in the forward/carry
# path for this arm's configuration (`puzzle_emb_ndim=0` means the ONE
# place batch_size-must-match-config would matter, `CastedSparseEmbedding`
# for puzzle embeddings, never gets constructed at all -- see
# `HRMACTv1_Inner.__init__`'s `if self.config.puzzle_emb_ndim > 0:` guard).
# So `hrmv2_config()`'s `batch_size=64` stays as-is (cosmetic/unused for
# this arm), and real batches are `cfg.batch_size` (1024, the mission
# config's field, matched to the native recipe) rows each.

def _training_seed_formula(seed: int, cell: dict) -> int:
    """The SAME pre-registered training-seed formula
    `continuous_prm_c11_mission._training_seed` uses (`10007*(seed+1) +
    101*config_idx + K`), reproduced here rather than imported (that
    function is a private helper of the core module, and this module's
    lazy-import discipline is about avoiding a HARD dependency on the core
    module at IMPORT time, not about refusing to depend on its PUBLIC
    behavior at call time -- but a private underscore-prefixed helper is
    exactly the kind of internal detail that should be duplicated-with-
    attribution rather than reached into)."""
    return 10007 * (int(seed) + 1) + 101 * int(cell["config_idx"]) + int(cell["K"])


# Per-batch segment cap -- a PURE BACKSTOP behind the segment loop's real
# terminator, the ever-halted DRAIN CONDITION (see the loop body in
# `train_hrmv2`). History, because both halves were discovered the hard
# way rather than designed in from the spec:
#
#   1. The spec's literal per-batch "loop until all_finish" instruction
#      implicitly assumes `carry.halted.all()` (every row halted ON THE
#      SAME call) is reachable in a bounded number of calls. It is NOT,
#      once training has run even a few optimizer steps:
#      `HRMACTv1.forward`'s halting is PER-ROW -- a row that halts via the
#      Q-driven branch (`q_halt_logits > q_continue_logits`, active
#      whenever `self.training`) gets `reset_carry`'d on the VERY NEXT
#      call and restarts a fresh episode on ITS OWN step-clock,
#      permanently desynchronized from every other row's clock (each row
#      is still GUARANTEED to halt again within `halt_max_steps` of ITS
#      OWN restart, via `is_last_step` -- see `HRMACTv1.forward`'s
#      algebra: `min_halt_steps <= halt_max_steps` always, so
#      `is_last_step` can never be suppressed by the exploration AND-mask
#      -- but "guaranteed to halt eventually, on its own clock" is not the
#      same as "all rows halt on the SAME absolute segment call").
#      Empirically: clean convergence up to n=16; verified NOT to converge
#      within 3000 segments (27.8s) on real encoded data at n=128 (the
#      tiny-test batch size) once a few optimizer steps had nudged the
#      Q-head -- and with ~10%/row exploration, P(all rows aligned-halted
#      on one call) ~ 0.9^B, which is ~0 at the recipe's REAL
#      `C11MissionConfig.batch_size` of 1024. Left uncapped, one batch can
#      run indefinitely; this cap was originally added as the fix.
#
#   2. The T7 review then measured that with the cap as the ONLY
#      terminator, it BOUND on ~90% of mid-training batches (14/16
#      measured) -- i.e. nearly every batch burned the full 200 segments
#      (~25x the intended `halt_max_steps=8` budget of `hrmv2_config()`'s
#      default; the earlier claim here that the cap "essentially never
#      binds" was FALSE, extrapolated from pre-training behavior), rows
#      re-ran ~25-50 episodes on the same data, and T9's 33 full-scale
#      runs would have inflated accordingly. The fix is the DRAIN
#      CONDITION now in the loop: exit when `ever_halted.all()` -- every
#      row has COMPLETED >= 1 ACT episode this batch -- which
#      `is_last_step` bounds at `halt_max_steps` segments (all first
#      episodes start synchronized on the fresh per-batch carry), while
#      preserving one-optimizer-step-per-segment and live Q-dynamics.
#
# Under the drain condition this cap is genuinely unreachable
# (`halt_max_steps` is 8 at the paper-faithful default, 200 = 25x that;
# tests assert zero capped batches and per-batch segment counts <=
# halt_max_steps) -- it exists purely to bound the damage of some future
# regression in the drain logic or the parent package's halting semantics,
# turning a would-be infinite loop into a loud, finite, diagnosable one
# (`meta["capped_batches"]` > 0 is the tell).
_PER_BATCH_SEGMENT_CAP = 200


def train_hrmv2(cell: dict, bundles: Sequence["WorldBundle"], seed: int,
                 cfg: Optional["C11MissionConfig"] = None,
                 hrmv2_cfg: Optional[dict] = None,
                 epochs: Optional[int] = None,
                 ) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Train a fresh `TraceHRMACT` on `bundles`' pooled TRAIN targets, via
    the faithful per-segment loop (see the module-level comment block
    above for the full recipe-split rationale). Each batch's segment loop
    terminates on the ever-halted DRAIN CONDITION -- every row has
    completed >= 1 ACT episode -- which `is_last_step` bounds at
    `halt_max_steps` segments per batch (see the loop body's comment and
    `_PER_BATCH_SEGMENT_CAP`'s history comment: per-row ACT halting
    desynchronizes across a batch once training perturbs the Q-head, so
    the naive "every row halted on the SAME call" condition is
    unreachable at real batch sizes, and a fixed cap alone inflated every
    batch to ~25x its segment budget). Returns `(state_dict_cpu, meta)`
    -- the SAME `(state_dict, meta)` contract `train_arm` returns, so
    this function is a drop-in `train` entry for
    `register_provider_builder`; `meta` additionally carries the
    drain-condition diagnostics (`capped_batches`, `max_batch_segments`,
    `batch_segments_hist`) and the architecture config
    (`hrmv2_config`).

    `hrmv2_cfg`: the `HRMACTv1Config` dict (from `hrmv2_config(...)`,
    typically with test-time overrides for speed); defaults to
    `hrmv2_config()`'s paper-faithful full-size defaults when omitted.
    `cfg`: the `C11MissionConfig` (data/epochs/batch_size/lr/weight_decay
    -- the MATCHED half of the recipe split); defaults to
    `C11MissionConfig()`.

    Seeding: `torch.manual_seed`/`torch.cuda.manual_seed_all` and the numpy
    shuffle RNG are BOTH seeded with `_training_seed_formula(seed, cell)`
    -- identical formula and identical dual-seeding convention to
    `train_arm`'s own seeding (see that function's docstring), so two
    `train_hrmv2` calls with identical inputs produce allclose state_dicts
    on CPU (bit-determinism, like `train_arm`, is CPU-only -- CUDA attention
    kernels are not bitwise deterministic across runs)."""
    _M = _load_mission()
    HRMACTv1, HRMACTv1_Inner, HRMACTv1Config, HRMACTv1InnerCarry = _load_hrm()
    from hrm.train.optim import AdamATan2, warmup_constant_lr

    cfg = cfg or _M.C11MissionConfig()
    hrmv2_cfg = dict(hrmv2_cfg) if hrmv2_cfg is not None else hrmv2_config()
    n_epochs = int(epochs) if epochs is not None else int(cfg.epochs)

    train_seed_val = _training_seed_formula(seed, cell)
    torch.manual_seed(train_seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed_val)
    shuffle_rng = np.random.default_rng(train_seed_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_hrmv2_arm(hrmv2_cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    loss_head = RegressionACTLossHead(model, band=0.1)

    opt = AdamATan2(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: warmup_constant_lr(step, 100))

    targets = _M.stack_targets(bundles)  # (n, 4): (i, s, y, bundle_idx)
    n_rows = targets.shape[0]
    all_i = targets[:, 0].astype(np.int64)
    all_s = targets[:, 1].astype(np.int64)
    all_y = targets[:, 2].astype(np.float32)
    all_bidx = targets[:, 3].astype(np.int64)

    batch_size = int(cfg.batch_size)
    n_full_batches = n_rows // batch_size  # drop-last: partial tail dropped.

    t0 = time.time()
    first_epoch_loss: Optional[float] = None
    final_loss: float = float("nan")
    total_segments = 0
    mean_halt_steps_final_epoch: float = float("nan")
    # Drain-condition diagnostics (T7-review IMPORTANT 2): how many
    # batches hit the backstop cap (must be 0 -- asserted in tests), the
    # largest per-batch segment count (must be <= halt_max_steps under the
    # ever-halted drain), and a compact {segment_count: n_batches}
    # histogram (bounded at halt_max_steps + 1 distinct keys, so it stays
    # tiny in the manifest even for full 40-epoch runs).
    capped_batches = 0
    max_batch_segments = 0
    batch_segments_hist: Dict[int, int] = {}

    model.train()
    for epoch in range(n_epochs):
        order = shuffle_rng.permutation(n_rows)
        epoch_seg_losses: List[float] = []
        epoch_halt_steps: List[float] = []

        for b in range(n_full_batches):
            batch_idx = order[b * batch_size:(b + 1) * batch_size]
            rows_i = all_i[batch_idx]
            rows_s = all_s[batch_idx]
            rows_bidx = all_bidx[batch_idx]
            y_batch = torch.from_numpy(all_y[batch_idx]).to(device)

            # Per-arm batch encoding, grouped by bundle (mirrors
            # `_forward_batch_by_bundle`'s bundle-grouping convention --
            # `encode_trace_padded` is called once per DISTINCT bundle
            # present in the batch, not once per row).
            n = batch_idx.shape[0]
            padded = np.zeros((n, cfg.seq_max, cfg.token_dim), dtype=np.float32)
            for b_idx in np.unique(rows_bidx):
                b_idx = int(b_idx)
                row_mask = rows_bidx == b_idx
                states = list(zip(rows_i[row_mask].tolist(), rows_s[row_mask].tolist()))
                bundle_padded, _mask = _M.encode_trace_padded(bundles[b_idx], states, cfg)
                padded[row_mask] = bundle_padded

            inputs_t = torch.from_numpy(padded).to(device)
            puzzle_ids_t = torch.zeros(n, dtype=torch.int32, device=device)
            batch = {"inputs": inputs_t, "puzzle_identifiers": puzzle_ids_t, "y": y_batch}

            # THE FAITHFUL LOOP: carry created ONCE per batch; loop
            # `loss_head(...)` (one ACT segment per call); ONE optimizer
            # step PER SEGMENT (deep supervision -- the mechanism under
            # test).
            #
            # DRAIN CONDITION (T7-review IMPORTANT 2): the loop exits when
            # `ever_halted.all()` -- every row has COMPLETED >= 1 ACT
            # episode this batch -- NOT when `carry.halted.all()` (all
            # rows halted on the SAME call). The original all-on-the-same-
            # call condition is unreachable in practice: `initial_carry`
            # starts every row's episode synchronized, but the FIRST row
            # to halt early (Q-vote or exploration path, active whenever
            # `self.training`) gets `reset_carry`'d on the very next call
            # and restarts on its OWN step-clock, permanently
            # desynchronized -- and with ~10%/row exploration,
            # P(all B rows aligned-halted on one call) ~ 0.9^B ~ 0 at the
            # real B=1024, so the old loop's 200-segment cap was the
            # de-facto terminator on ~90% of mid-training batches
            # (measured 14/16 by the T7 review), inflating every batch to
            # ~25x its `halt_max_steps` segment budget and re-running rows
            # ~25-50 episodes on the same data. The drain condition is
            # bounded: all first episodes START synchronized (fresh
            # carry), and `is_last_step` unconditionally halts any
            # still-running row at its episode step `halt_max_steps`
            # (never suppressible -- `min_halt_steps <= halt_max_steps`
            # always, see `HRMACTv1.forward`'s exploration mask algebra),
            # so every row's FIRST halt lands within the batch's first
            # `halt_max_steps` segments and the loop drains by then.
            # Early-halting rows may begin a second episode before the
            # batch drains; those partial-episode segments still
            # contribute normal gradient updates on the same data
            # (harmless, and exactly what the streaming script's
            # persistent carry would also do to them). One-optimizer-step-
            # per-segment and live Q-learning dynamics are preserved
            # untouched. `_PER_BATCH_SEGMENT_CAP` remains as a pure
            # backstop only -- genuinely never binding now (asserted by
            # `test_streaming_trainer_segment_steps`: zero capped batches,
            # every batch's segment count <= halt_max_steps).
            carry = loss_head.initial_carry(batch)
            ever_halted = torch.zeros(n, dtype=torch.bool, device=device)
            batch_halt_steps: List[float] = []
            batch_segments = 0
            while not bool(ever_halted.all()) and batch_segments < _PER_BATCH_SEGMENT_CAP:
                carry, loss, metrics, _outputs, _all_finish_t = loss_head(
                    return_keys=[], carry=carry, batch=batch
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite loss for arm 'hrmv2_act'")
                (loss / n).backward()
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=False)
                total_segments += 1
                batch_segments += 1
                epoch_seg_losses.append(float(loss.item()) / n)
                count = float(metrics["count"].item())
                if count > 0:
                    batch_halt_steps.append(float(metrics["steps"].item()) / count)
                ever_halted = ever_halted | carry.halted

            if batch_segments >= _PER_BATCH_SEGMENT_CAP and not bool(ever_halted.all()):
                capped_batches += 1
            max_batch_segments = max(max_batch_segments, batch_segments)
            batch_segments_hist[batch_segments] = batch_segments_hist.get(batch_segments, 0) + 1

            epoch_halt_steps.extend(batch_halt_steps)

        epoch_mean = float(np.mean(epoch_seg_losses)) if epoch_seg_losses else float("nan")
        if epoch == 0:
            first_epoch_loss = epoch_mean
        final_loss = epoch_mean
        if epoch_halt_steps:
            mean_halt_steps_final_epoch = float(np.mean(epoch_halt_steps))

    wall_s = time.time() - t0

    state_dict_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    meta = {
        "arm": "hrmv2_act",
        "config_label": cell.get("config_label"),
        "K": int(cell["K"]),
        "seed": int(seed),
        "epochs": n_epochs,
        "param_count": int(param_count),
        "first_epoch_loss": first_epoch_loss,
        "final_loss": final_loss,
        "wall_s": wall_s,
        "total_segments": total_segments,
        "mean_halt_steps_final_epoch": mean_halt_steps_final_epoch,
        # Drain-condition diagnostics (T7-review IMPORTANT 2) -- see the
        # loop body's comment: capped_batches must be 0 in practice.
        "capped_batches": int(capped_batches),
        "max_batch_segments": int(max_batch_segments),
        "batch_segments_hist": dict(batch_segments_hist),
        # T7-review minor 2: persist the ARCHITECTURE config alongside the
        # weights, so a non-default architecture (e.g. a scaled T10
        # variant or a test-sized model) can be reconstructed at eval time
        # from the ckpt/manifest instead of failing `make_provider`'s
        # strict state_dict load against `hrmv2_config()`'s defaults --
        # today the registered `build` still constructs the default
        # architecture (a mismatched ckpt fails LOUDLY at load, which is
        # the acceptable current behavior), but the config traveling with
        # the ckpt makes the future fix a pure consumer-side change. All
        # values are scalars/strings, so this JSON-serializes cleanly into
        # the manifest.
        "hrmv2_config": dict(hrmv2_cfg),
    }
    return state_dict_cpu, meta


# ---------------------------------------------------------------------------
# Task 7: eval providers -- ACT-live and forced-k.
# ---------------------------------------------------------------------------
#
# Both providers share the SAME batch-building step (encode every queried
# (i, s) state's trace tokens via `encode_trace_padded`, grouped by bundle
# -- trivial here since `forward_batch`'s contract is single-bundle, so no
# grouping is actually needed, unlike the training loop's multi-bundle
# batches) and differ only in the SEGMENT LOOP driving the model:
#   - ACT-live: the model's OWN halting (`model.forward`'s ACT wrapper),
#     run in eval mode. IMPORTANT FINDING (verified by reading
#     `HRMACTv1.forward` AND by direct execution, not assumed): in eval
#     mode (`self.training == False`), the EARLY-halt branch (`halted =
#     halted | (q_halt_logits > q_continue_logits)`) and the exploration
#     branch are BOTH gated on `self.training` and therefore NEVER fire --
#     only `is_last_step` (`new_steps >= halt_max_steps`) can halt a
#     sequence in eval mode. This means the ACT-live eval provider ALWAYS
#     runs exactly `halt_max_steps` segments for EVERY query, regardless of
#     the trained Q-head's actual halt/continue preferences -- Q-driven
#     early stopping is a TRAINING-time exploration/bootstrapping
#     mechanism in this port, not a mechanism the port exposes at eval
#     time. This is a real, load-bearing property of the FROZEN, installed
#     `hrm` package (confirmed empirically: `model.eval()` +
#     `halt_exploration_prob=0.5` + running to `carry.halted.all()` always
#     takes exactly `halt_max_steps` iterations, independent of q-head
#     values) -- not a bug in this arm's code, and not something this
#     phase is permitted to patch (`HRM-v2/` is frozen). Practical
#     consequence (T7-review IMPORTANT 1): SEGMENTS-RUN is therefore a
#     constant and useless as a halting readout -- so the per-query side
#     channel (`bundle.hrmv2_halt_steps`) records the Q-DERIVED WOULD-HALT
#     step instead: the first segment at which `q_halt_logits >
#     q_continue_logits` fires (the exact comparison the training-mode
#     early-halt branch uses), `halt_max_steps` if never -- computed
#     inside the SAME segment loop from logits every segment call already
#     produces (zero extra compute), and genuinely VARYING with what the
#     Q-head learned, which is what gate G2b's halting-vs-K correlation
#     needs (Spearman on a constant is undefined). See
#     `_hrmv2_forward_batch_act`'s docstring for the full rationale. The
#     provider's yhat path is implemented EXACTLY as specified ("eval
#     mode, the model's own halting") -- this comment documents what that
#     mechanism actually does, verified rather than assumed.
#   - Forced-k: bypasses `model.forward`'s ACT wrapper ENTIRELY, driving
#     `model.inner` directly for exactly k calls (each call is "one
#     segment" by the same definition `train_maze_optimized.py`'s own
#     comment uses: "ONE segment per optimizer step... forward one segment
#     through the loss head" -- `model.inner.forward` IS that one-segment
#     unit; the ACT wrapper (`model.forward`) adds only the
#     halt/continue/exploration/target-Q bookkeeping around it). This
#     genuinely ignores q-halt (there is no q-based branching in this
#     path at all -- q values aren't even read), giving TRUE k-segment
#     compute regardless of what the trained Q-head would have decided.

def _build_batch_for_states(model: "TraceHRMACT", bundle: "WorldBundle",
                             states: Sequence[Tuple[int, int]], cfg: "C11MissionConfig",
                             device: torch.device) -> Dict[str, torch.Tensor]:
    """Build the `{"inputs", "puzzle_identifiers"}` batch dict for
    `states` on `bundle`, via `encode_trace_padded` (the SAME encoder every
    other trace-consuming arm uses -- identical information per the I/O
    contract, spec section 3)."""
    _M = _load_mission()
    padded, _mask = _M.encode_trace_padded(bundle, states, cfg)
    inputs_t = torch.from_numpy(padded).to(device)
    n = len(states)
    puzzle_ids_t = torch.zeros(n, dtype=torch.int32, device=device)
    return {"inputs": inputs_t, "puzzle_identifiers": puzzle_ids_t}


def _hrmv2_forward_batch_act(model: "TraceHRMACT", bundle: "WorldBundle",
                              states: Sequence[Tuple[int, int]], cfg: "C11MissionConfig",
                              device: torch.device) -> torch.Tensor:
    """ACT-LIVE provider: run the model's own halting (eval mode) segment
    loop until `all_finish`, reading out via `readout_yhat(outputs,
    cap=cfg.residual_cap)`.

    SIDE-CHANNEL (T7-review IMPORTANT 1): stashes the per-query Q-DERIVED
    WOULD-HALT step on `bundle.hrmv2_halt_steps` (an `(K+1, N)` float32
    ndarray, NaN off the queried states) -- the FIRST segment index (1-
    based) at which the model's own halting signal, `q_halt_logits >
    q_continue_logits` (the exact comparison `HRMACTv1.forward`'s training-
    mode early-halt branch uses), fires for that query; `halt_max_steps`
    if it never fires. Values are therefore in `[1, halt_max_steps]`.

    WHY would-halt and not segments-run: in eval mode
    (`model.training == False`) the early-halt branch and the exploration
    branch of `HRMACTv1.forward` are BOTH `self.training`-gated and never
    fire, so segments-run is CONSTANT -- every query runs exactly
    `halt_max_steps` segments regardless of the trained Q-head (verified
    empirically on both a rigged q-head and a trained model; the vendored
    original at repo-root `models/hrm/hrm_act_v1.py` behaves identically
    per the T7 review). A constant channel makes gate G2b -- Spearman
    correlation of halt-steps vs mission depth K -- undefined. The
    Q-derived would-halt step is the paper-legitimate "learned halting"
    readout under this port's fixed-compute eval contract: it is computed
    from the SAME q logits the training-mode halting actually branches on,
    read out of the SAME segment loop this provider already runs (zero
    extra forward passes -- `outputs["q_halt_logits"]`/
    `["q_continue_logits"]` are produced by every segment call
    unconditionally), and it VARIES with what the Q-head learned, which is
    the thing G2b is trying to measure. `yhat` (the actual heuristic
    prediction) is unaffected: it is still read out at each row's REAL
    halt (which in eval mode is always the final, `is_last_step`-forced
    segment), exactly as before."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            batch = _build_batch_for_states(model, bundle, states, cfg, device)
            carry = model.initial_carry(batch)

            # All bookkeeping tensors live on `device` throughout the loop
            # (matching `carry.halted`/`outputs[...]`'s device) -- moved to
            # CPU/numpy only once, after the loop, when stashing the
            # side channel. Mixing CPU bookkeeping with a CUDA `carry`
            # would raise a device mismatch the moment `device` is not
            # "cpu" (this arm's tests run on CPU, but `make_provider`/
            # `predict_field` derive `device` from the model's OWN device,
            # which is CUDA whenever a trained checkpoint is loaded onto a
            # CUDA model -- so this must be device-correct regardless of
            # what this run happens to use).
            would_halt_out = torch.zeros(len(states), dtype=torch.float32, device=device)
            yhat_out = torch.zeros(len(states), dtype=torch.float32, device=device)
            remaining = torch.ones(len(states), dtype=torch.bool, device=device)
            undecided = torch.ones(len(states), dtype=torch.bool, device=device)

            halt_max_steps = int(model.config.halt_max_steps)
            seg_idx = 0
            for _ in range(halt_max_steps + 1):
                carry, outputs = model(carry, batch)
                seg_idx += 1

                # Q-derived would-halt vote: record the FIRST segment at
                # which q_halt > q_continue (strict >, matching the
                # training-mode branch's own comparison) for each
                # still-undecided query. Zero extra compute -- both logits
                # are already in `outputs` on every segment call.
                vote = outputs["q_halt_logits"] > outputs["q_continue_logits"]
                newly_decided = vote & undecided
                if newly_decided.any():
                    would_halt_out[newly_decided] = float(seg_idx)
                    undecided = undecided & ~newly_decided

                newly_halted = carry.halted & remaining
                if newly_halted.any():
                    yhat_row = readout_yhat(outputs, cap=float(cfg.residual_cap))
                    idx = newly_halted.nonzero(as_tuple=True)[0]
                    yhat_out[idx] = yhat_row[idx]
                    remaining = remaining & ~newly_halted
                if carry.halted.all():
                    break

            if remaining.any():
                # Should not happen (halt_max_steps bounds the loop above
                # at halt_max_steps+1 iterations, and `is_last_step` alone
                # guarantees `carry.halted.all()` by step halt_max_steps),
                # but guard rather than silently returning stale zeros.
                raise RuntimeError(
                    f"{int(remaining.sum())} states never halted within "
                    f"{halt_max_steps} ACT-live segments"
                )

            # Queries whose q-vote never fired record halt_max_steps (the
            # "would run the full budget" reading -- consistent with what
            # training-mode halting would have done to them: only
            # is_last_step would have stopped them).
            if undecided.any():
                would_halt_out[undecided] = float(halt_max_steps)
    finally:
        model.train(was_training)

    _stash_halt_steps(bundle, states, would_halt_out.detach().cpu().numpy())
    return yhat_out


def _stash_halt_steps(bundle: "WorldBundle", states: Sequence[Tuple[int, int]],
                       halt_steps: np.ndarray) -> None:
    """Store-on-bundle pattern (matching `WorldBundle.node_rays`/
    `field_grids`'s established convention): `bundle.hrmv2_halt_steps` is
    an `(K+1, N)` float32 ndarray, created NaN-filled on first use and
    updated in place on every call (so repeated ACT-live provider calls on
    the same bundle, e.g. across multiple budgets in `run_eval`, keep
    accumulating/overwriting the SAME queried cells rather than each call
    starting over from an all-NaN array -- consistent with every other
    query in the same eval run sharing one halt-step map per bundle).

    CROSS-CALL STALENESS CAVEAT (T7-review minor 3): because cells are
    only OVERWRITTEN when re-queried (never invalidated wholesale), a
    SECOND, differently-configured model (different seed/checkpoint/
    halt_max_steps) that queries only a SUBSET of a bundle's states leaves
    the non-requeried cells holding the FIRST model's values -- the map is
    "last writer per cell", not "one coherent model per map". Within one
    `run_eval` this cannot mix models (each provider queries the FULL
    (K+1) x N state set via `predict_field`, overwriting every cell), but
    a consumer doing partial ad-hoc queries across models must clear the
    attribute (`del bundle.hrmv2_halt_steps`) between models, as the
    side-channel test itself does between its two q-head rigs."""
    K = len(bundle.wp)
    N = bundle.rm.points.shape[0]
    existing = getattr(bundle, "hrmv2_halt_steps", None)
    if existing is None or existing.shape != (K + 1, N):
        existing = np.full((K + 1, N), np.nan, dtype=np.float32)
    for row, (i, s) in enumerate(states):
        existing[s, i] = halt_steps[row]
    bundle.hrmv2_halt_steps = existing


def _hrmv2_forward_batch_forced_k(k: int) -> Callable[..., torch.Tensor]:
    """Returns a `forward_batch(model, bundle, states, cfg, device)`
    callable that runs EXACTLY `k` inner segment advances, ignoring q-halt
    entirely (`model.inner` is driven directly -- the ACT wrapper's
    halt/continue/exploration logic never runs, so there is no q-based
    branching to ignore in the first place; this IS what "ignore q-halt"
    means operationally). `k` is captured by closure; the returned
    callable's signature matches the `forward_batch` contract exactly, so
    it plugs directly into `register_provider_builder`."""
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    def _forward_batch(model: "TraceHRMACT", bundle: "WorldBundle",
                        states: Sequence[Tuple[int, int]], cfg: "C11MissionConfig",
                        device: torch.device) -> torch.Tensor:
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                batch = _build_batch_for_states(model, bundle, states, cfg, device)
                n = len(states)
                inner_carry = model.inner.empty_carry(n)
                inner_carry = model.inner.reset_carry(
                    torch.ones(n, dtype=torch.bool, device=device), inner_carry
                )
                outputs = None
                logits = None
                for _ in range(k):
                    inner_carry, logits, (_q_halt, _q_continue) = model.inner(inner_carry, batch)
                outputs = {"logits": logits}
                yhat = readout_yhat(outputs, cap=float(cfg.residual_cap))
        finally:
            model.train(was_training)
        return yhat

    return _forward_batch


# ---------------------------------------------------------------------------
# Task 7: registration hook -- called by the core module post-import.
# ---------------------------------------------------------------------------

_HRMV2_FORCED_K_VALUES: Tuple[int, ...] = (1, 2, 4, 8)

_HRMV2_REGISTERED = False


def register_hrmv2_providers() -> None:
    """Register `hrmv2_act` (ACT-live, WITH a trainer) and
    `hrmv2_act_k{1,2,4,8}` (forced-k, EVAL-ONLY -- no trainer, `ckpt_alias=
    "hrmv2_act"`) into the core module's provider registry
    (`continuous_prm_c11_mission.register_provider_builder`). Called by the
    core module itself, post-import, wrapped in a try/except ImportError.

    EAGERLY calls `_load_hrm()` FIRST, before registering anything --
    THIS is the load-bearing line for the core module's guard contract
    (`try: ... register_hrmv2_providers() except ImportError: pass`). An
    earlier version of this function deferred ALL `hrm`-touching to
    provider-CONSTRUCTION time (the `build(cfg)` closures below, which
    only run when `run_train`/`run_eval`/`make_provider` actually build a
    model) -- which meant `register_hrmv2_providers()` itself NEVER raised
    ImportError even when `hrm` was completely absent, so the core
    module's registration hook silently registered `hrmv2_act` anyway
    (with `build`/`forward_batch` closures that would only fail LATER, the
    first time something tried to actually use them) -- a real bug caught
    by `test_hrmv2_registration_import_error_only_guard`'s subprocess test
    (which blocks `hrm` at `sys.meta_path` and asserts `hrmv2_act` is NOT
    registered): the pre-fix core module imported "successfully" but with
    a registry entry that lied about `hrm` availability. Calling
    `_load_hrm()` up front makes this function fail FAST and VISIBLY the
    moment `hrm` is unavailable, exactly where the core module's
    try/except expects the failure to occur.

    Idempotent: calling this twice (e.g. two test processes each importing
    the core module) is a no-op the second time (`PROVIDER_BUILDERS`
    already has the names; `register_provider_builder` would otherwise be
    fine with re-registering the SAME names to the SAME callables anyway,
    since only NATIVE arm names are rejected -- but the module-level guard
    keeps this cheap and avoids rebuilding closures pointlessly)."""
    global _HRMV2_REGISTERED
    if _HRMV2_REGISTERED:
        return

    _load_hrm()  # Eager check -- see the docstring above; must run FIRST.
    _M = _load_mission()

    def _build(cfg: "C11MissionConfig") -> "TraceHRMACT":
        return build_hrmv2_arm(hrmv2_config())

    def _train(cell: dict, bundles: Sequence["WorldBundle"], seed: int,
               cfg: Optional["C11MissionConfig"] = None, epochs: Optional[int] = None
               ) -> Tuple[Dict[str, torch.Tensor], dict]:
        """Adapter binding `train_hrmv2`'s `hrmv2_cfg` parameter (which has
        NO slot in the registry's `train(cell, bundles, seed, cfg, epochs)`
        contract -- exactly 5 positional parameters, none of them a
        model-architecture config) to the paper-faithful default
        (`hrmv2_config()`, called FRESH on every invocation rather than
        once at registration time, so it always reflects the CURRENT
        `hrmv2_config` -- relevant for tests that monkeypatch it). Passing
        `train_hrmv2` directly as `train=train_hrmv2` (an earlier version
        of this registration did exactly that) is a signature mismatch:
        `_resolve_trainer`'s adapter in the core module calls
        `registered_train(cell, bundles, seed, cfg, epochs)` -- 5
        positional arguments -- which `train_hrmv2` receives as `(cell,
        bundles, seed, cfg=<cfg>, hrmv2_cfg=<epochs value>)`, silently
        binding the CALLER's `epochs` argument to `train_hrmv2`'s
        `hrmv2_cfg` parameter (both are keyword-or-positional in that
        slot) -- a genuine, silent parameter-mismatch bug caught by an
        end-to-end `run_train` smoke test (train_hrmv2 crashed inside
        `hrmv2_config()` with `TypeError: 'int' object is not iterable`
        the moment `epochs` was an int, since `hrmv2_cfg` expects a
        dict-or-None). This thin wrapper is the fix: it matches the
        registry's EXACT positional contract and threads `hrmv2_cfg`
        itself, rather than relying on positional-argument coincidence."""
        return train_hrmv2(cell, bundles, seed, cfg=cfg, hrmv2_cfg=hrmv2_config(), epochs=epochs)

    _M.register_provider_builder(
        "hrmv2_act", _build, _hrmv2_forward_batch_act, train=_train,
    )

    for k in _HRMV2_FORCED_K_VALUES:
        _M.register_provider_builder(
            f"hrmv2_act_k{k}", _build, _hrmv2_forward_batch_forced_k(k),
            train=None, ckpt_alias="hrmv2_act",
        )

    _HRMV2_REGISTERED = True


__all__ = [
    "build_hrmv2_arm",
    "hrmv2_config",
    "TraceHRMInner",
    "TraceHRMACT",
    "readout_yhat",
    "regression_correct",
    "RegressionACTLossHead",
    "train_hrmv2",
    "register_hrmv2_providers",
]
