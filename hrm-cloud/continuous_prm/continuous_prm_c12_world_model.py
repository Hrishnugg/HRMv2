"""Shared C12-A frame encoder, direct decoder, and persistent temporal cores.

The five learned arms differ only in ``TemporalCore``.  They consume the same
visible frame payload and use the same direct multi-horizon decoder.  Carry is
an explicit tree of tensors so training can reset it only at episode boundaries
and detach it exactly at truncated-BPTT boundaries.
"""
from __future__ import annotations

import json
import hashlib
import os
import random
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
# Required by CUDA >= 10.2 for reproducible cuBLAS reductions.  This must be
# present before the first CUDA handle is created; setting it at import time
# keeps smoke/pilot/full behavior aligned.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import Tensor, nn
import torch.nn.functional as F


ARM_NAMES: Tuple[str, ...] = (
    "snapshot",
    "lstm",
    "temporal_transformer",
    "onlstm",
    "hrm_stream",
)
WIDTH_CANDIDATES: Tuple[int, ...] = (256, 320, 384, 448, 512)


@dataclass(frozen=True)
class WorldModelConfig:
    d_model: int = 256
    horizon: int = 32
    max_patrollers: int = 1
    max_gates: int = 2
    raster_channels: int = 5
    core_width: int = 512
    recurrent_layers: int = 2
    snapshot_depth: int = 8
    transformer_window: int = 16
    transformer_depth: int = 4
    transformer_heads: int = 8
    transformer_ff_width: int = 1024
    onlstm_chunk_size: int = 8
    hrm_slow_cadence: int = 4
    decoder_width: Optional[int] = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "d_model": self.d_model,
            "horizon": self.horizon,
            "max_patrollers": self.max_patrollers,
            "max_gates": self.max_gates,
            "raster_channels": self.raster_channels,
            "core_width": self.core_width,
            "recurrent_layers": self.recurrent_layers,
            "snapshot_depth": self.snapshot_depth,
            "transformer_window": self.transformer_window,
            "transformer_depth": self.transformer_depth,
            "transformer_heads": self.transformer_heads,
            "transformer_ff_width": self.transformer_ff_width,
            "onlstm_chunk_size": self.onlstm_chunk_size,
            "hrm_slow_cadence": self.hrm_slow_cadence,
        }
        invalid = [name for name, value in positive.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"world-model fields must be positive: {', '.join(invalid)}")
        if self.d_model % self.transformer_heads:
            raise ValueError("d_model must be divisible by transformer_heads")
        if self.core_width % self.onlstm_chunk_size:
            raise ValueError("core_width must be divisible by onlstm_chunk_size")
        if self.decoder_width is not None and self.decoder_width <= 0:
            raise ValueError("decoder_width must be positive when provided")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    batch_size: int = 16
    tbptt_steps: int = 32
    max_epochs: int = 20
    patience: int = 4
    min_delta: float = 0.0
    collapse_std_threshold: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer settings")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")
        for name in ("batch_size", "tbptt_steps", "max_epochs", "patience"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_delta < 0.0 or self.collapse_std_threshold < 0.0:
            raise ValueError("validation thresholds must be non-negative")


def _module_dtype(module: nn.Module) -> torch.dtype:
    parameter = next(module.parameters(), None)
    return torch.float32 if parameter is None else parameter.dtype


class FrameEncoder(nn.Module):
    """Encode the shared visible raster/object payload into one frame token."""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        raster_dim = max(32, cfg.d_model // 2)
        object_dim = max(16, cfg.d_model // 4)
        visible_dim = max(8, cfg.d_model // 8)
        self.raster_net = nn.Sequential(
            nn.Conv2d(cfg.raster_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, raster_dim),
            nn.SiLU(),
        )
        self.object_net = nn.Sequential(
            nn.Linear(3, object_dim),
            nn.SiLU(),
            nn.Linear(object_dim, object_dim),
            nn.SiLU(),
        )
        self.visible_net = nn.Sequential(
            nn.Linear(4, visible_dim),
            nn.SiLU(),
            nn.Linear(visible_dim, visible_dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(raster_dim + object_dim + visible_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(
        self,
        frame_rasters: Tensor,
        centers: Tensor,
        radii: Tensor,
        identity_mask: Tensor,
        visible_regime_context: Tensor,
        visible_regime_mask: Tensor,
    ) -> Tensor:
        raster = self.raster_net(frame_rasters.float())
        object_inputs = torch.cat((centers.float(), radii.float().unsqueeze(-1)), dim=-1)
        object_tokens = self.object_net(object_inputs)
        identity = identity_mask.to(object_tokens.dtype).unsqueeze(-1)
        object_summary = (object_tokens * identity).sum(dim=1) / identity.sum(
            dim=1
        ).clamp_min(1.0)
        visible_mask = visible_regime_mask.to(visible_regime_context.dtype)
        visible_inputs = torch.cat(
            (visible_regime_context.float() * visible_mask.float(), visible_mask.float()),
            dim=-1,
        )
        visible = self.visible_net(visible_inputs)
        return self.fusion(torch.cat((raster, object_summary, visible), dim=-1))

    def from_batch(self, batch: Mapping[str, Tensor]) -> Tensor:
        return self(
            batch["frame_rasters"],
            batch["centers"],
            batch["radii"],
            batch["identity_mask"],
            batch["visible_regime_context"],
            batch["visible_regime_mask"],
        )


class DirectHorizonDecoder(nn.Module):
    """Decode all future horizons directly from one temporal context."""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        width = int(cfg.decoder_width or max(cfg.d_model, cfg.core_width))
        self.trunk = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.center_head = nn.Linear(
            width, cfg.horizon * cfg.max_patrollers * 2
        )
        self.gate_head = nn.Linear(width, cfg.horizon * cfg.max_gates)

    def forward(
        self, context: Tensor, identity_mask: Tensor, gate_mask: Tensor
    ) -> Dict[str, Tensor]:
        batch = context.shape[0]
        hidden = self.trunk(context)
        centers = self.center_head(hidden).reshape(
            batch, self.cfg.horizon, self.cfg.max_patrollers, 2
        )
        gates = self.gate_head(hidden).reshape(
            batch, self.cfg.horizon, self.cfg.max_gates
        )
        centers = centers * identity_mask.to(centers.dtype)[:, None, :, None]
        gates = gates * gate_mask.to(gates.dtype)[:, None, :]
        return {"center_displacements": centers, "gate_logits": gates}


class TemporalCore(nn.Module, ABC):
    @abstractmethod
    def initial_carry(self, batch_size: int, device: torch.device) -> Any:
        raise NotImplementedError

    @abstractmethod
    def step(self, frame_embedding: Tensor, carry: Any) -> Tuple[Tensor, Any]:
        raise NotImplementedError

    def detach_carry(self, carry: Any) -> Any:
        return detach_carry(carry)


class SnapshotCore(TemporalCore):
    def __init__(self, d_model: int, width: int, depth: int = 8) -> None:
        super().__init__()
        self.input = nn.Linear(d_model, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU())
            for _ in range(int(depth))
        )
        self.output = nn.Linear(width, d_model)

    def initial_carry(self, batch_size: int, device: torch.device) -> None:
        return None

    def step(self, frame_embedding: Tensor, carry: Any) -> Tuple[Tensor, None]:
        hidden = F.silu(self.input(frame_embedding))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden), None


class LSTMCore(TemporalCore):
    def __init__(
        self, d_model: int, width: int, layers: int = 2, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.layers = nn.ModuleList(
            nn.LSTMCell(d_model if layer == 0 else width, width)
            for layer in range(int(layers))
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(width, d_model)

    def initial_carry(
        self, batch_size: int, device: torch.device
    ) -> Tuple[Tuple[Tensor, Tensor], ...]:
        dtype = _module_dtype(self)
        return tuple(
            (
                torch.zeros(batch_size, self.width, device=device, dtype=dtype),
                torch.zeros(batch_size, self.width, device=device, dtype=dtype),
            )
            for _ in self.layers
        )

    def step(
        self, frame_embedding: Tensor, carry: Tuple[Tuple[Tensor, Tensor], ...]
    ) -> Tuple[Tensor, Tuple[Tuple[Tensor, Tensor], ...]]:
        value = frame_embedding
        next_carry = []
        for index, (cell, state) in enumerate(zip(self.layers, carry)):
            h, c = cell(value, state)
            next_carry.append((h, c))
            value = self.dropout(h) if index + 1 < len(self.layers) else h
        return self.output(value), tuple(next_carry)


class SlidingWindowTransformerCore(TemporalCore):
    def __init__(
        self,
        d_model: int,
        window: int = 16,
        depth: int = 4,
        heads: int = 8,
        ff_width: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.window = int(window)
        self.position = nn.Parameter(torch.zeros(window, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=depth, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )

    def initial_carry(self, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        dtype = _module_dtype(self)
        return {
            "history": torch.zeros(
                batch_size, self.window, self.d_model, device=device, dtype=dtype
            ),
            "length": torch.zeros(batch_size, device=device, dtype=torch.long),
        }

    def step(
        self, frame_embedding: Tensor, carry: Mapping[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        history = torch.cat(
            (carry["history"][:, 1:], frame_embedding.unsqueeze(1)), dim=1
        )
        length = torch.clamp(carry["length"] + 1, max=self.window)
        positions = torch.arange(self.window, device=frame_embedding.device)[None, :]
        padding = positions < (self.window - length)[:, None]
        # The buffer itself is the causal boundary: it contains only frames
        # already observed at this online step.  We consume only the newest
        # output token, so it may attend to every earlier token in the buffer.
        # Omitting an intra-buffer triangular mask also avoids fully-masked
        # leading padding queries during short prefixes.
        encoded = self.encoder(
            history + self.position[None, :, :],
            src_key_padding_mask=padding,
        )
        return encoded[:, -1], {"history": history, "length": length}


def cumax(value: Tensor) -> Tensor:
    # Floating accumulation can put the last entry a few ulps above one.
    return torch.cumsum(torch.softmax(value, dim=-1), dim=-1).clamp_(0.0, 1.0)


class ONLSTMCell(nn.Module):
    """Ordered-neuron LSTM cell using the validated cumulative-softmax gates."""

    def __init__(self, input_size: int, hidden_size: int, chunk_size: int = 8) -> None:
        super().__init__()
        if hidden_size % chunk_size:
            raise ValueError("ON-LSTM hidden size must be divisible by chunk size")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.chunk_size = int(chunk_size)
        self.chunks = hidden_size // chunk_size
        self.projection = nn.Linear(
            input_size + hidden_size, 4 * hidden_size + 2 * self.chunks
        )

    def _project(self, value: Tensor, hidden: Tensor) -> Tensor:
        return self.projection(torch.cat((value, hidden), dim=-1))

    def _master_from_projected(self, projected: Tensor) -> Tuple[Tensor, Tensor]:
        offset = 4 * self.hidden_size
        master_forget = cumax(projected[:, offset : offset + self.chunks])
        master_update = 1.0 - cumax(projected[:, offset + self.chunks :])
        return (
            master_forget.repeat_interleave(self.chunk_size, dim=-1),
            master_update.repeat_interleave(self.chunk_size, dim=-1),
        )

    def master_gates(self, value: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        return self._master_from_projected(self._project(value, hidden))

    def forward(
        self, value: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        hidden, cell = state
        projected = self._project(value, hidden)
        gates = projected[:, : 4 * self.hidden_size]
        forget, update, output, candidate = gates.chunk(4, dim=-1)
        forget = torch.sigmoid(forget)
        update = torch.sigmoid(update)
        output = torch.sigmoid(output)
        candidate = torch.tanh(candidate)
        master_forget, master_update = self._master_from_projected(projected)
        overlap = master_forget * master_update
        forget = forget * overlap + (master_forget - overlap)
        update = update * overlap + (master_update - overlap)
        next_cell = forget * cell + update * candidate
        next_hidden = output * torch.tanh(next_cell)
        return next_hidden, next_cell


class ONLSTMCore(TemporalCore):
    def __init__(
        self,
        d_model: int,
        width: int,
        layers: int = 2,
        chunk_size: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.layers = nn.ModuleList(
            ONLSTMCell(
                d_model if layer == 0 else width,
                width,
                chunk_size=chunk_size,
            )
            for layer in range(int(layers))
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(width, d_model)

    def initial_carry(
        self, batch_size: int, device: torch.device
    ) -> Tuple[Tuple[Tensor, Tensor], ...]:
        dtype = _module_dtype(self)
        return tuple(
            (
                torch.zeros(batch_size, self.width, device=device, dtype=dtype),
                torch.zeros(batch_size, self.width, device=device, dtype=dtype),
            )
            for _ in self.layers
        )

    def step(
        self, frame_embedding: Tensor, carry: Tuple[Tuple[Tensor, Tensor], ...]
    ) -> Tuple[Tensor, Tuple[Tuple[Tensor, Tensor], ...]]:
        value = frame_embedding
        next_carry = []
        for index, (cell, state) in enumerate(zip(self.layers, carry)):
            h, c = cell(value, state)
            next_carry.append((h, c))
            value = self.dropout(h) if index + 1 < len(self.layers) else h
        return self.output(value), tuple(next_carry)


class HRMStreamCore(TemporalCore):
    """Explicit persistent fast/slow core, distinct from C11 HRM-v2 ACT.

    The fast GRU updates for every visible frame.  The slow GRU updates only
    after each fixed ``slow_cadence`` block and persists between replans.  No
    adaptive-computation halting or per-query inner loop is used here.
    """

    def __init__(self, d_model: int, width: int, slow_cadence: int = 4) -> None:
        super().__init__()
        self.width = int(width)
        self.slow_cadence = int(slow_cadence)
        self.input = nn.Linear(d_model, width)
        self.fast_cell = nn.GRUCell(2 * width, width)
        self.slow_cell = nn.GRUCell(width, width)
        self.output = nn.Sequential(
            nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, d_model)
        )

    def initial_carry(self, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        dtype = _module_dtype(self)
        return {
            "fast": torch.zeros(batch_size, self.width, device=device, dtype=dtype),
            "slow": torch.zeros(batch_size, self.width, device=device, dtype=dtype),
            "step": torch.zeros(batch_size, device=device, dtype=torch.long),
        }

    def step(
        self, frame_embedding: Tensor, carry: Mapping[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        projected = F.silu(self.input(frame_embedding))
        fast = self.fast_cell(torch.cat((projected, carry["slow"]), dim=-1), carry["fast"])
        slow_candidate = self.slow_cell(fast, carry["slow"])
        next_step = carry["step"] + 1
        update = (next_step % self.slow_cadence == 0).unsqueeze(-1)
        slow = torch.where(update, slow_candidate, carry["slow"])
        context = self.output(torch.cat((fast, slow), dim=-1))
        return context, {"fast": fast, "slow": slow, "step": next_step}


def detach_carry(carry: Any) -> Any:
    if torch.is_tensor(carry):
        return carry.detach()
    if isinstance(carry, dict):
        return {key: detach_carry(value) for key, value in carry.items()}
    if isinstance(carry, tuple):
        return tuple(detach_carry(value) for value in carry)
    if isinstance(carry, list):
        return [detach_carry(value) for value in carry]
    return carry


def reset_carry(carry: Any, boundary_mask: Tensor, initial: Any) -> Any:
    """Reset selected batch streams while preserving every other carry value."""
    if torch.is_tensor(carry):
        if carry.ndim == 0 or carry.shape[0] != boundary_mask.shape[0]:
            return carry
        mask = boundary_mask.to(device=carry.device, dtype=torch.bool)
        mask = mask.reshape(mask.shape[0], *([1] * (carry.ndim - 1)))
        return torch.where(mask, initial.to(carry.device), carry)
    if isinstance(carry, dict):
        return {
            key: reset_carry(value, boundary_mask, initial[key])
            for key, value in carry.items()
        }
    if isinstance(carry, tuple):
        return tuple(
            reset_carry(value, boundary_mask, clean)
            for value, clean in zip(carry, initial)
        )
    if isinstance(carry, list):
        return [
            reset_carry(value, boundary_mask, clean)
            for value, clean in zip(carry, initial)
        ]
    return initial if carry is None else carry


class C12WorldModel(nn.Module):
    def __init__(
        self,
        arm: str,
        cfg: WorldModelConfig,
        encoder: FrameEncoder,
        core: TemporalCore,
        decoder: DirectHorizonDecoder,
    ) -> None:
        super().__init__()
        self.arm = arm
        self.cfg = cfg
        self.encoder = encoder
        self.core = core
        self.decoder = decoder

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initial_carry(self, batch_size: int, device: torch.device) -> Any:
        return self.core.initial_carry(batch_size, device)

    def step(
        self, batch: Mapping[str, Tensor], carry: Any
    ) -> Tuple[Dict[str, Tensor], Any]:
        frame = self.encoder.from_batch(batch)
        context, next_carry = self.core.step(frame, carry)
        prediction = self.decoder(context, batch["identity_mask"], batch["gate_mask"])
        prediction["context"] = context
        return prediction, next_carry

    def run_sequence(
        self,
        batches: Sequence[Mapping[str, Tensor]],
        carry: Any = None,
        boundary_masks: Optional[Sequence[Optional[Tensor]]] = None,
        detach_every: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Tensor]], Any]:
        if not batches:
            raise ValueError("run_sequence requires at least one frame batch")
        batch_size = int(batches[0]["frame_rasters"].shape[0])
        if carry is None:
            carry = self.initial_carry(batch_size, batches[0]["frame_rasters"].device)
        outputs: List[Dict[str, Tensor]] = []
        for index, batch in enumerate(batches):
            if boundary_masks is not None and boundary_masks[index] is not None:
                initial = self.initial_carry(batch_size, batch["frame_rasters"].device)
                carry = reset_carry(carry, boundary_masks[index], initial)
            output, carry = self.step(batch, carry)
            outputs.append(output)
            if detach_every is not None and (index + 1) % int(detach_every) == 0:
                carry = detach_carry(carry)
        return outputs, carry

    def window_reencode(
        self, batches: Sequence[Mapping[str, Tensor]]
    ) -> Dict[str, Tensor]:
        if not batches:
            raise ValueError("window_reencode requires at least one frame")
        window = list(batches[-self.cfg.transformer_window :])
        outputs, _ = self.run_sequence(window, carry=None)
        return outputs[-1]


def build_world_model(
    arm: str, cfg: Optional[WorldModelConfig] = None
) -> C12WorldModel:
    cfg = cfg or WorldModelConfig()
    if arm not in ARM_NAMES:
        raise KeyError(f"unknown C12 world-model arm: {arm!r}")
    if arm == "snapshot":
        core: TemporalCore = SnapshotCore(
            cfg.d_model, cfg.core_width, depth=cfg.snapshot_depth
        )
    elif arm == "lstm":
        core = LSTMCore(
            cfg.d_model,
            cfg.core_width,
            layers=cfg.recurrent_layers,
            dropout=cfg.dropout,
        )
    elif arm == "temporal_transformer":
        core = SlidingWindowTransformerCore(
            cfg.d_model,
            window=cfg.transformer_window,
            depth=cfg.transformer_depth,
            heads=cfg.transformer_heads,
            ff_width=cfg.transformer_ff_width,
            dropout=cfg.dropout,
        )
    elif arm == "onlstm":
        core = ONLSTMCore(
            cfg.d_model,
            cfg.core_width,
            layers=cfg.recurrent_layers,
            chunk_size=cfg.onlstm_chunk_size,
            dropout=cfg.dropout,
        )
    else:
        core = HRMStreamCore(
            cfg.d_model, cfg.core_width, slow_cadence=cfg.hrm_slow_cadence
        )
    return C12WorldModel(
        arm,
        cfg,
        FrameEncoder(cfg),
        core,
        DirectHorizonDecoder(cfg),
    )


def _masked_reduction(value: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    expanded = mask.expand_as(value)
    numerator = (value * expanded).sum()
    denominator = expanded.sum()
    return numerator / denominator.clamp_min(1.0), numerator, denominator


def forecast_loss(
    predicted_centers: Tensor,
    gate_logits: Tensor,
    target_centers: Tensor,
    target_gate_open: Tensor,
    identity_mask: Tensor,
    gate_mask: Tensor,
    horizon_mask: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Uniform-horizon Huber(center) + 0.5 BCE(gate), with slot masks."""
    if predicted_centers.shape != target_centers.shape:
        raise ValueError("predicted and target center shapes differ")
    if gate_logits.shape != target_gate_open.shape:
        raise ValueError("predicted and target gate shapes differ")
    center_element = F.huber_loss(
        predicted_centers, target_centers, reduction="none", delta=1.0
    )
    center_mask = identity_mask.to(center_element.dtype)[:, None, :, None]
    gate_weight = gate_mask.to(gate_logits.dtype)[:, None, :]
    if horizon_mask is not None:
        if horizon_mask.shape != predicted_centers.shape[:2]:
            raise ValueError("horizon_mask must have shape [batch,horizon]")
        center_mask = center_mask * horizon_mask.to(center_element.dtype)[:, :, None, None]
        gate_weight = gate_weight * horizon_mask.to(gate_logits.dtype)[:, :, None]
    center_loss, center_numerator, center_denominator = _masked_reduction(
        center_element, center_mask
    )
    gate_element = F.binary_cross_entropy_with_logits(
        gate_logits, target_gate_open.to(gate_logits.dtype), reduction="none"
    )
    gate_loss, gate_numerator, gate_denominator = _masked_reduction(
        gate_element, gate_weight
    )
    total = center_loss + 0.5 * gate_loss
    return {
        "total": total,
        "center_huber": center_loss,
        "gate_bce": gate_loss,
        "center_numerator": center_numerator,
        "center_denominator": center_denominator,
        "gate_numerator": gate_numerator,
        "gate_denominator": gate_denominator,
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class C12ShardStore:
    """Verified shard reader that shuffles episodes, never timesteps."""

    def __init__(self, root: str | Path, verify: bool = True) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "dataset_manifest.json"
        if not self.manifest_path.exists():
            raise RuntimeError(f"C12 dataset manifest missing at {self.manifest_path}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete":
            raise RuntimeError("C12 training requires a complete dataset manifest")
        self.manifest_hash = _sha256_path(self.manifest_path)
        self.requested_splits: List[str] = []
        if verify:
            self.verify()

    @property
    def available_splits(self) -> Tuple[str, ...]:
        return tuple(str(value) for value in self.manifest.get("splits", {}))

    def _entries(self, split: str) -> List[Dict[str, Any]]:
        split = str(split).upper()
        entries = [
            dict(entry)
            for entry in self.manifest.get("shards", [])
            if str(entry.get("split")) == split
        ]
        if not entries:
            raise KeyError(f"C12 dataset split {split!r} is unavailable")
        return entries

    def verify(self) -> None:
        seen = set()
        for entry in self.manifest.get("shards", []):
            relative = str(entry["path"])
            if relative in seen:
                raise RuntimeError(f"duplicate shard path in manifest: {relative}")
            seen.add(relative)
            path = self.root / relative
            if not path.exists():
                raise RuntimeError(f"C12 dataset shard missing at {path}")
            if _sha256_path(path) != entry.get("sha256"):
                raise RuntimeError(f"C12 dataset checksum mismatch at {path}")
            with np.load(path, allow_pickle=False) as payload:
                if any(payload[name].dtype == object for name in payload.files):
                    raise RuntimeError(f"object array found in C12 shard {path}")
                counts = {int(payload[name].shape[0]) for name in payload.files}
            if counts != {int(entry["episodes"])}:
                raise RuntimeError(f"episode count mismatch in C12 shard {path}")

    def dimensions(self) -> Dict[str, int]:
        first = dict(self.manifest["shards"][0])
        path = self.root / str(first["path"])
        with np.load(path, allow_pickle=False) as payload:
            return {
                "episode_steps": int(payload["frame_rasters"].shape[1]),
                "raster_channels": int(payload["frame_rasters"].shape[2]),
                "raster_size": int(payload["frame_rasters"].shape[3]),
                "horizon": int(payload["target_center_displacements"].shape[2]),
                "max_patrollers": int(payload["centers"].shape[2]),
                "max_gates": int(payload["gate_mask"].shape[2]),
            }

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        seed: int,
        shuffle: bool,
    ) -> Iterator[Dict[str, np.ndarray]]:
        split = str(split).upper()
        self.requested_splits.append(split)
        entries = self._entries(split)
        rng = np.random.default_rng(int(seed))
        if shuffle:
            rng.shuffle(entries)
        for entry in entries:
            path = self.root / str(entry["path"])
            with np.load(path, allow_pickle=False) as payload:
                arrays = {name: np.asarray(payload[name]) for name in payload.files}
            count = int(entry["episodes"])
            indices = np.arange(count, dtype=np.int64)
            if shuffle:
                rng.shuffle(indices)
            for start in range(0, count, int(batch_size)):
                selection = indices[start : start + int(batch_size)]
                # Advanced indexing materializes a compact batch, allowing the
                # source shard to be released before the next optimizer step.
                yield {name: array[selection] for name, array in arrays.items()}


_FLOAT_BATCH_FIELDS = frozenset(
    {
        "frame_rasters",
        "centers",
        "radii",
        "identity_mask",
        "visible_regime_context",
        "visible_regime_mask",
        "target_center_displacements",
        "target_gate_open",
        "gate_mask",
        "route_critical_mask",
        "route_edge_midpoints",
    }
)


def tensorize_episode_batch(
    batch: Mapping[str, np.ndarray], device: torch.device
) -> Dict[str, Tensor]:
    result: Dict[str, Tensor] = {}
    for name, value in batch.items():
        tensor = torch.from_numpy(np.asarray(value))
        if name in _FLOAT_BATCH_FIELDS:
            tensor = tensor.float()
        result[name] = tensor.to(device=device, non_blocking=device.type == "cuda")
    return result


def _frame_at(batch: Mapping[str, Tensor], t: int) -> Dict[str, Tensor]:
    return {
        "frame_rasters": batch["frame_rasters"][:, t],
        "centers": batch["centers"][:, t],
        "radii": batch["radii"][:, t],
        "identity_mask": batch["identity_mask"][:, t],
        "visible_regime_context": batch["visible_regime_context"][:, t],
        "visible_regime_mask": batch["visible_regime_mask"][:, t],
        "gate_mask": batch["gate_mask"][:, t],
    }


def collapse_diagnostics(
    center_sum: float,
    center_square_sum: float,
    center_count: int,
    gate_sum: float,
    gate_square_sum: float,
    gate_count: int,
    threshold: float = 1.0e-6,
) -> Dict[str, Any]:
    def std(total: float, square: float, count: int) -> float:
        if count <= 1:
            return 0.0
        mean = total / count
        return float(max(0.0, square / count - mean * mean) ** 0.5)

    center_std = std(center_sum, center_square_sum, center_count)
    gate_std = std(gate_sum, gate_square_sum, gate_count)
    collapsed = bool(center_std <= threshold and gate_std <= threshold)
    return {
        "center_prediction_std": center_std,
        "gate_probability_std": gate_std,
        "center_values": int(center_count),
        "gate_values": int(gate_count),
        "threshold": float(threshold),
        "collapsed": collapsed,
        "validation_status": "failed_constant_output" if collapsed else "passed",
    }


def run_forecast_epoch(
    model: C12WorldModel,
    store: C12ShardStore,
    split: str,
    cfg: TrainingConfig,
    device: torch.device,
    epoch_seed: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    route_critical_only: bool = False,
) -> Dict[str, Any]:
    """Run one contiguous episode-stream epoch with exact TBPTT boundaries."""
    training = optimizer is not None
    model.train(training)
    center_numerator = 0.0
    center_denominator = 0.0
    gate_numerator = 0.0
    gate_denominator = 0.0
    center_sum = center_square_sum = 0.0
    gate_sum = gate_square_sum = 0.0
    center_count = gate_count = 0
    grad_norms: List[float] = []
    episode_count = 0
    step_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for numpy_batch in store.iter_batches(
            split,
            batch_size=cfg.batch_size,
            seed=epoch_seed,
            shuffle=training,
        ):
            batch = tensorize_episode_batch(numpy_batch, device)
            batch_size = int(batch["frame_rasters"].shape[0])
            steps = int(batch["frame_rasters"].shape[1])
            episode_count += batch_size
            carry = model.initial_carry(batch_size, device)
            for chunk_start in range(0, steps, cfg.tbptt_steps):
                chunk_stop = min(steps, chunk_start + cfg.tbptt_steps)
                chunk_losses: List[Tensor] = []
                if training:
                    assert optimizer is not None
                    optimizer.zero_grad(set_to_none=True)
                for t in range(chunk_start, chunk_stop):
                    prediction, carry = model.step(_frame_at(batch, t), carry)
                    step_count += batch_size
                    validation_burn_in = int(
                        store.manifest.get("cfg", {}).get("burn_in", 0)
                    )
                    if route_critical_only and t < validation_burn_in:
                        # Carry still consumes the full visible prefix, but
                        # checkpoint selection begins only at eligible planning
                        # decisions after the registered burn-in.
                        continue
                    horizon_mask = (
                        batch["route_critical_mask"][:, t]
                        if route_critical_only
                        else None
                    )
                    losses = forecast_loss(
                        prediction["center_displacements"],
                        prediction["gate_logits"],
                        batch["target_center_displacements"][:, t],
                        batch["target_gate_open"][:, t],
                        batch["identity_mask"][:, t],
                        batch["gate_mask"][:, t],
                        horizon_mask=horizon_mask,
                    )
                    if not route_critical_only or (
                        losses["center_denominator"].item() > 0
                        and losses["gate_denominator"].item() > 0
                    ):
                        chunk_losses.append(losses["total"])
                    center_numerator += float(losses["center_numerator"].detach())
                    center_denominator += float(losses["center_denominator"].detach())
                    gate_numerator += float(losses["gate_numerator"].detach())
                    gate_denominator += float(losses["gate_denominator"].detach())
                    if not training:
                        centers = prediction["center_displacements"].detach()
                        center_mask = batch["identity_mask"][:, t, None, :, None].bool()
                        valid_centers = centers.masked_select(center_mask.expand_as(centers))
                        probabilities = torch.sigmoid(prediction["gate_logits"].detach())
                        gate_mask = batch["gate_mask"][:, t, None, :].bool()
                        valid_gates = probabilities.masked_select(
                            gate_mask.expand_as(probabilities)
                        )
                        center_sum += float(valid_centers.sum())
                        center_square_sum += float((valid_centers * valid_centers).sum())
                        gate_sum += float(valid_gates.sum())
                        gate_square_sum += float((valid_gates * valid_gates).sum())
                        center_count += int(valid_centers.numel())
                        gate_count += int(valid_gates.numel())

                if training and chunk_losses:
                    chunk_loss = torch.stack(chunk_losses).mean()
                    chunk_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.gradient_clip
                    )
                    grad_norms.append(float(grad_norm.detach()))
                    optimizer.step()
                carry = detach_carry(carry)

    center_loss = center_numerator / max(1.0, center_denominator)
    gate_loss = gate_numerator / max(1.0, gate_denominator)
    result: Dict[str, Any] = {
        "split": str(split).upper(),
        "episodes": episode_count,
        "stream_steps": step_count,
        "route_critical_only": bool(route_critical_only),
        "center_huber": center_loss,
        "gate_bce": gate_loss,
        "total": center_loss + 0.5 * gate_loss,
        "gradient_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "gradient_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
        "optimizer_steps": len(grad_norms),
    }
    if not training:
        result["collapse"] = collapse_diagnostics(
            center_sum,
            center_square_sum,
            center_count,
            gate_sum,
            gate_square_sum,
            gate_count,
            threshold=cfg.collapse_std_threshold,
        )
    return result


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad or not trainable_only
        )
    )


def estimate_madds_per_step(model: C12WorldModel, raster_size: int = 32) -> int:
    """Deterministic architecture-level multiply-add proxy for matching arms."""
    total = 0
    spatial = int(raster_size)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            spatial = (spatial + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
            total += int(module.weight.numel() * spatial * spatial)
        elif isinstance(module, (nn.Linear, nn.LSTMCell, nn.GRUCell)):
            total += int(
                sum(
                    parameter.numel()
                    for name, parameter in module.named_parameters(recurse=False)
                    if "weight" in name
                )
            )
    # MultiheadAttention's in-projection is a raw parameter rather than a
    # child Linear, so explicitly account for it once per transformer layer.
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                total += int(module.in_proj_weight.numel())
    return int(total)


def model_accounting(model: C12WorldModel, raster_size: int = 32) -> Dict[str, int]:
    return {
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "total_parameters": count_parameters(model, trainable_only=False),
        "encoder_parameters": count_parameters(model.encoder),
        "core_parameters": count_parameters(model.core),
        "decoder_parameters": count_parameters(model.decoder),
        "estimated_madds_per_step": estimate_madds_per_step(model, raster_size),
    }


def tiny_alias_sanity(
    device: str | torch.device = "cpu",
    seed: int = 12012,
    optimization_steps: int = 160,
) -> Dict[str, Dict[str, float]]:
    """Verify temporal cores can fit history labels that snapshot cannot see."""
    device = torch.device(device)
    results: Dict[str, Dict[str, float]] = {}
    labels = torch.tensor([0.0, 1.0] * 16, device=device)
    first = torch.zeros(labels.shape[0], 16, device=device)
    first[:, 0] = labels * 2.0 - 1.0
    aliased_now = torch.zeros_like(first)
    cfg = WorldModelConfig(
        d_model=16,
        horizon=1,
        max_patrollers=1,
        max_gates=1,
        core_width=32,
        recurrent_layers=1,
        snapshot_depth=2,
        transformer_window=2,
        transformer_depth=1,
        transformer_heads=4,
        transformer_ff_width=64,
        onlstm_chunk_size=4,
        hrm_slow_cadence=2,
        decoder_width=32,
        dropout=0.0,
    )
    for arm_index, arm in enumerate(ARM_NAMES):
        torch.manual_seed(int(seed) + arm_index)
        model = build_world_model(arm, cfg).to(device)
        core = model.core
        head = nn.Linear(cfg.d_model, 1).to(device)
        optimizer = torch.optim.Adam(
            list(core.parameters()) + list(head.parameters()), lr=1.0e-2
        )
        loss_value = float("nan")
        for _ in range(int(optimization_steps)):
            carry = core.initial_carry(labels.shape[0], device)
            _context, carry = core.step(first, carry)
            context, _carry = core.step(aliased_now, carry)
            logits = head(context).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())
        with torch.no_grad():
            carry = core.initial_carry(labels.shape[0], device)
            _context, carry = core.step(first, carry)
            context, _carry = core.step(aliased_now, carry)
            predicted = (head(context).squeeze(-1) >= 0.0).float()
            accuracy = float((predicted == labels).float().mean())
        results[arm] = {"accuracy": accuracy, "loss": loss_value}
    return results


def select_model_grid(
    base_cfg: Optional[WorldModelConfig] = None,
    candidates: Sequence[int] = WIDTH_CANDIDATES,
    target_parameters: int = 4_000_000,
    tolerance_range: Tuple[int, int] = (3_000_000, 5_000_000),
    raster_size: int = 32,
) -> Dict[str, Any]:
    """Choose preregistered widths by parameter distance, never by outcomes."""
    base_cfg = base_cfg or WorldModelConfig()
    rows: Dict[str, Any] = {}
    for arm in ARM_NAMES:
        choices = []
        for width in candidates:
            kwargs: Dict[str, Any] = {"core_width": int(width)}
            if arm == "temporal_transformer":
                kwargs["transformer_ff_width"] = 2 * int(width)
            cfg = replace(base_cfg, **kwargs)
            model = build_world_model(arm, cfg)
            accounting = model_accounting(model, raster_size=raster_size)
            in_range = (
                tolerance_range[0]
                <= accounting["trainable_parameters"]
                <= tolerance_range[1]
            )
            choices.append((not in_range, abs(accounting["trainable_parameters"] - target_parameters), width, cfg, accounting))
        _, _, width, cfg, accounting = min(choices, key=lambda row: row[:3])
        rows[arm] = {
            "width": int(width),
            "config": asdict(cfg),
            **accounting,
            "within_3m_5m": bool(
                tolerance_range[0]
                <= accounting["trainable_parameters"]
                <= tolerance_range[1]
            ),
        }
    return {
        "schema_version": "c12a-model-grid-v1",
        "selection_rule": {
            "candidate_widths": [int(value) for value in candidates],
            "target_parameters": int(target_parameters),
            "tolerance_range": [int(value) for value in tolerance_range],
        },
        "arms": rows,
    }


def write_model_grid(path: str | Path, grid: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(grid, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _scientific_training_hash(
    store: C12ShardStore,
    arm: str,
    seed: int,
    model_cfg: WorldModelConfig,
    train_cfg: TrainingConfig,
) -> str:
    payload = {
        "dataset_manifest_hash": store.manifest_hash,
        "dataset_config_hash": store.manifest.get("config_hash"),
        "arm": arm,
        "seed": int(seed),
        "model_cfg": asdict(model_cfg),
        "training_cfg": asdict(train_cfg),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    last_error: Optional[PermissionError] = None
    for attempt in range(20):
        try:
            torch.save(dict(payload), tmp)
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(1.0, 0.05 * (attempt + 1)))
    assert last_error is not None
    raise last_error


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _git_state() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def _capture_rng_state(device: torch.device) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng_state(payload: Mapping[str, Any], device: torch.device) -> None:
    if "torch_cpu" in payload:
        torch.set_rng_state(payload["torch_cpu"].cpu())
    if device.type == "cuda" and "torch_cuda" in payload:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["torch_cuda"]])


def _update_training_manifest(out_dir: Path, entry: Mapping[str, Any]) -> None:
    path = out_dir / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "c12a-training-manifest-v1":
            raise RuntimeError("C12 training manifest schema mismatch")
    else:
        manifest = {"schema_version": "c12a-training-manifest-v1", "training_runs": []}
    key = (str(entry["arm"]), int(entry["seed"]))
    rows = [
        row
        for row in manifest.get("training_runs", [])
        if (str(row.get("arm")), int(row.get("seed", -1))) != key
    ]
    rows.append(dict(entry))
    rows.sort(key=lambda row: (ARM_NAMES.index(str(row["arm"])), int(row["seed"])))
    manifest["training_runs"] = rows
    _atomic_json(path, manifest)


def train_world_model(
    store: C12ShardStore,
    out_dir: str | Path,
    arm: str,
    seed: int,
    model_cfg: Optional[WorldModelConfig] = None,
    train_cfg: Optional[TrainingConfig] = None,
    device: str | torch.device = "cpu",
    stop_after_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Train/resume one arm using TRAIN and VALIDATION only.

    ``stop_after_epochs`` is an operational test/maintenance limit and is not a
    scientific hyperparameter.  A limited invocation remains ``running`` and
    resumes from the next atomic epoch checkpoint without duplicate manifests.
    """
    if arm not in ARM_NAMES:
        raise KeyError(f"unknown C12 training arm: {arm!r}")
    model_cfg = model_cfg or WorldModelConfig()
    train_cfg = train_cfg or TrainingConfig()
    device = torch.device(device)
    out_dir = Path(out_dir)
    science_hash = _scientific_training_hash(
        store, arm, seed, model_cfg, train_cfg
    )
    checkpoint_dir = out_dir / "checkpoints"
    last_path = checkpoint_dir / f"c12a__{arm}__seed{int(seed)}.last.pt"
    best_path = checkpoint_dir / f"c12a__{arm}__seed{int(seed)}.best.pt"

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device)
        # The fused memory-efficient attention backward is nondeterministic on
        # current CUDA.  The math kernel is slower but preserves the frozen
        # model-seed contract used for C12 comparisons.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    model = build_world_model(arm, model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    accounting = model_accounting(
        model, raster_size=store.dimensions()["raster_size"]
    )
    git = _git_state()
    history: List[Dict[str, Any]] = []
    best_validation = float("inf")
    best_epoch: Optional[int] = None
    stale_epochs = 0
    start_epoch = 0
    prior_wall = 0.0

    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("scientific_hash") != science_hash:
            raise RuntimeError(
                "C12 checkpoint scientific config mismatch "
                f"({checkpoint.get('scientific_hash')} != {science_hash})"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = list(checkpoint.get("history", []))
        best_validation = float(checkpoint.get("best_validation", float("inf")))
        best_epoch = checkpoint.get("best_epoch")
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        prior_wall = float(checkpoint.get("wall_seconds", 0.0))
        _restore_rng_state(checkpoint.get("rng_state", {}), device)
        if checkpoint.get("status") in ("complete", "failed_validation_collapse"):
            entry = dict(checkpoint["manifest_entry"])
            _update_training_manifest(out_dir, entry)
            return entry

    invocation_limit = train_cfg.max_epochs
    if stop_after_epochs is not None:
        if int(stop_after_epochs) <= 0:
            raise ValueError("stop_after_epochs must be positive")
        invocation_limit = min(
            train_cfg.max_epochs, start_epoch + int(stop_after_epochs)
        )
    t0 = time.time()
    early_stopped = False
    last_validation: Dict[str, Any] = {}

    for epoch in range(start_epoch, invocation_limit):
        train_metrics = run_forecast_epoch(
            model,
            store,
            "TRAIN",
            train_cfg,
            device,
            epoch_seed=int(seed) * 100_000 + epoch,
            optimizer=optimizer,
            route_critical_only=False,
        )
        validation_metrics = run_forecast_epoch(
            model,
            store,
            "VALIDATION",
            train_cfg,
            device,
            epoch_seed=int(seed) * 100_000 + epoch,
            optimizer=None,
            route_critical_only=True,
        )
        last_validation = validation_metrics
        collapsed = bool(validation_metrics["collapse"]["collapsed"])
        score = float(validation_metrics["total"])
        improved = bool(
            not collapsed and score < best_validation - train_cfg.min_delta
        )
        if improved:
            best_validation = score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "selected": improved,
            }
        )
        wall = prior_wall + time.time() - t0
        peak_vram = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        common_payload: Dict[str, Any] = {
            "schema_version": "c12a-checkpoint-v1",
            "scientific_hash": science_hash,
            "dataset_manifest_hash": store.manifest_hash,
            "dataset_config_hash": store.manifest.get("config_hash"),
            "arm": arm,
            "seed": int(seed),
            "model_cfg": asdict(model_cfg),
            "training_cfg": asdict(train_cfg),
            "model_accounting": accounting,
            "git": git,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "stale_epochs": stale_epochs,
            "history": history,
            "wall_seconds": wall,
            "peak_vram_bytes": peak_vram,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": _capture_rng_state(device),
        }
        if improved:
            best_payload = dict(common_payload)
            best_payload["status"] = "best"
            _atomic_torch_save(best_path, best_payload)

        early_stopped = stale_epochs >= train_cfg.patience
        exhausted = epoch + 1 >= train_cfg.max_epochs
        operational_stop = epoch + 1 >= invocation_limit and not exhausted
        if early_stopped or exhausted:
            status = "complete" if best_epoch is not None else "failed_validation_collapse"
        elif operational_stop:
            status = "running"
        else:
            status = "running"
        entry = {
            "arm": arm,
            "seed": int(seed),
            "status": status,
            "scientific_hash": science_hash,
            "dataset_manifest_hash": store.manifest_hash,
            "dataset_config_hash": store.manifest.get("config_hash"),
            "last_checkpoint": str(last_path),
            "best_checkpoint": str(best_path) if best_path.exists() or improved else None,
            "epochs_completed": epoch + 1,
            "best_epoch": best_epoch,
            "best_validation": best_validation if best_epoch is not None else None,
            "wall_seconds": wall,
            "peak_vram_bytes": peak_vram,
            "parameters": accounting["trainable_parameters"],
            "estimated_madds_per_step": accounting["estimated_madds_per_step"],
            "last_gradient_norm_mean": train_metrics["gradient_norm_mean"],
            "last_gradient_norm_max": train_metrics["gradient_norm_max"],
            "collapse": validation_metrics["collapse"],
            "git": git,
        }
        common_payload["status"] = status
        common_payload["manifest_entry"] = entry
        _atomic_torch_save(last_path, common_payload)
        _update_training_manifest(out_dir, entry)
        if early_stopped:
            break

    if not history and start_epoch >= train_cfg.max_epochs:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        entry = dict(checkpoint["manifest_entry"])
    else:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        entry = dict(checkpoint["manifest_entry"])
    if any(split not in ("TRAIN", "VALIDATION") for split in store.requested_splits):
        raise AssertionError("training accessed a forbidden non-development split")
    return entry


__all__ = [
    "ARM_NAMES",
    "WIDTH_CANDIDATES",
    "WorldModelConfig",
    "TrainingConfig",
    "FrameEncoder",
    "DirectHorizonDecoder",
    "TemporalCore",
    "SnapshotCore",
    "LSTMCore",
    "SlidingWindowTransformerCore",
    "ONLSTMCell",
    "ONLSTMCore",
    "HRMStreamCore",
    "C12WorldModel",
    "cumax",
    "detach_carry",
    "reset_carry",
    "build_world_model",
    "forecast_loss",
    "C12ShardStore",
    "tensorize_episode_batch",
    "collapse_diagnostics",
    "run_forecast_epoch",
    "train_world_model",
    "count_parameters",
    "estimate_madds_per_step",
    "model_accounting",
    "tiny_alias_sanity",
    "select_model_grid",
    "write_model_grid",
]
