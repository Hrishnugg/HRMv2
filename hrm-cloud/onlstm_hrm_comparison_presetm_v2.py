"""
ON-LSTM vs HRM (DynamicMaze++ Preset M+) — Experiment v2

Builds on the Preset M ON-LSTM vs HRM experiment with four improvements:

1) Richer evaluation metrics:
   - Failure breakdown (collision / timeout / static collision)
   - Planning cost proxy (A* node expansions)
   - One-step prediction MSE (overall + per obstacle class)
   - k-step rollout MSE (overall + per obstacle class)

2) Multi-step training objective (+ scheduled sampling):
   - Train with k-step autoregressive rollout loss (k=ENV_CONFIG['k_rollout'])
   - Scheduled sampling transitions from teacher-forcing to autoregressive

3) Local map context as input:
   - Each sample includes an 11×11 binary occupancy patch centered on the obstacle's last observed position
   - A lightweight CNN encodes the patch, concatenated to (x,y) at each timestep

4) Evaluation ablations:
   - Random vs fixed-map evaluation
   - Gates on/off
   - Planning horizon sweep (10 vs 20)

Usage:
    modal run hrm-cloud/onlstm_hrm_comparison_presetm_v2.py
"""

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import heapq
import os
import json
import copy
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from collections import deque

# -----------------------------------------------------------------------------
# Modal app / image / volume
# -----------------------------------------------------------------------------

app = modal.App("onlstm-hrm-comparison-dynamicmaze-presetm-v2")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch>=2.4.0", "numpy", "gymnasium", "tqdm")
)

# New volume so we don't overwrite the v1 experiment artifacts
vol = modal.Volume.from_name("onlstm-hrm-comparison-presetm-v2-vol", create_if_missing=True)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for a single model variant."""
    name: str
    model_type: str  # "onlstm" or "hrm"
    hidden_dim: int
    num_layers: int
    num_heads: int = 4      # HRM only
    chunk_size: int = 5     # ON-LSTM only (hidden_dim must be divisible by chunk_size)
    patch_embed_dim: int = 16
    batch_size: int = 4096
    lr: float = 1e-3
    epochs: int = 40
    gpu: str = "A10"


# ON-LSTM variants — parameter tiers aligned with prior LSTM buckets (approx)
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "onlstm_300k": ModelConfig("onlstm_300k", "onlstm", hidden_dim=155, num_layers=2,
                                chunk_size=5, patch_embed_dim=16, batch_size=4096, lr=1e-3, epochs=30, gpu="H100"),
    "onlstm_1m": ModelConfig("onlstm_1m", "onlstm", hidden_dim=275, num_layers=2,
                              chunk_size=5, patch_embed_dim=16, batch_size=4096, lr=1e-3, epochs=30, gpu="H100"),
    "onlstm_3m": ModelConfig("onlstm_3m", "onlstm", hidden_dim=475, num_layers=2,
                              chunk_size=5, patch_embed_dim=16, batch_size=4096, lr=1e-3, epochs=30, gpu="H100"),
    "onlstm_10m": ModelConfig("onlstm_10m", "onlstm", hidden_dim=860, num_layers=3,
                               chunk_size=5, patch_embed_dim=16, batch_size=2048, lr=5e-4, epochs=40, gpu="H100"),

    # HRM variants — unchanged tiers
    "hrm_302k": ModelConfig("hrm_302k", "hrm", hidden_dim=128, num_layers=2, num_heads=4,
                             patch_embed_dim=16, batch_size=4096, lr=1e-3, epochs=40, gpu="B200"),
    "hrm_3m": ModelConfig("hrm_3m", "hrm", hidden_dim=256, num_layers=2, num_heads=4,
                           patch_embed_dim=16, batch_size=4096, lr=4e-4, epochs=40, gpu="B200"),
    "hrm_10m": ModelConfig("hrm_10m", "hrm", hidden_dim=384, num_layers=3, num_heads=6,
                            patch_embed_dim=16, batch_size=2048, lr=5e-4, epochs=40, gpu="B200:4"),
}

# Environment + training config (Preset M+)
ENV_CONFIG: Dict[str, Any] = {
    "grid_size": 32,

    # Dynamic obstacles (total)
    "n_dynamic": 12,
    "n_patrollers": 4,
    "n_drifters": 6,
    "n_gates": 2,

    # Predictor / planner windows
    "obs_history": 20,
    "pred_horizon": 20,  # default; overridden per evaluation suite

    # Training objective improvements
    "k_rollout": 5,      # multi-step loss horizon (k-step rollout)
    "patch_size": 11,    # local occupancy patch centered on obstacle last observed position (odd)

    # Episode / dataset generation
    "data_episodes": 60000,
    "physics_steps_per_episode": 70,

    # Eval
    "eval_episodes": 100,
    "max_agent_steps": 128,
    "collision_radius": 0.8,

    # Map generation (rooms + corridors)
    "n_rooms": 8,
    "room_min_size": 4,
    "room_max_size": 7,
    "room_padding": 1,

    # Drifter dynamics
    "drifter_regime_min": 8,
    "drifter_regime_max": 20,
    "drifter_p_forward": 0.75,

    # Gate schedule
    "gate_period_min": 10,
    "gate_period_max": 18,
    "gate_open_min": 4,
    "gate_open_max": 6,

    # Evaluation metrics
    "eval_rollout_mse_k": 5,  # set 0 to disable rollout-mse computation
}

# Evaluation suites (ablations)
EVAL_SUITES: List[Dict[str, Any]] = [
    # Baseline: random maps, horizon 20, gates on
    {"name": "random_h20", "map_mode": "random", "pred_horizon": 20,
     "n_gates": 2, "n_patrollers": 4, "n_drifters": 6, "eval_episodes": 100},

    # Horizon sweep: random maps, horizon 10
    {"name": "random_h10", "map_mode": "random", "pred_horizon": 10,
     "n_gates": 2, "n_patrollers": 4, "n_drifters": 6, "eval_episodes": 100},

    # Fixed-map: same map every episode, horizon 20 (dynamics vary)
    {"name": "fixedmap_h20", "map_mode": "fixed", "map_seed": 12345, "pred_horizon": 20,
     "n_gates": 2, "n_patrollers": 4, "n_drifters": 6, "eval_episodes": 100},

    # Gates-off: keep total obstacle count constant by reallocating to drifters
    {"name": "gatesoff_random_h20", "map_mode": "random", "pred_horizon": 20,
     "n_gates": 0, "n_patrollers": 4, "n_drifters": 8, "eval_episodes": 100},
]

# Checkpointing config
CHECKPOINT_EVERY = 5

# Artifact paths (new base folder to avoid collisions with v1)
PATHS = {
    "data_dir": "/data/onlstm_comparison_presetm_v2/episodes",
    "merged_data": "/data/onlstm_comparison_presetm_v2/merged.pt",
    "models_dir": "/data/onlstm_comparison_presetm_v2/models",
    "checkpoints_dir": "/data/onlstm_comparison_presetm_v2/checkpoints",
    "results": "/data/onlstm_comparison_presetm_v2/results.json",
}

# =============================================================================
# MODEL ARCHITECTURES (ON-LSTM, HRM) + PATCH CONDITIONING
# =============================================================================

class ONLSTMCell(nn.Module):
    """Ordered Neurons LSTM cell (chunked ON-LSTM)."""

    def __init__(self, input_dim: int, hidden_dim: int, chunk_size: int = 5):
        super().__init__()
        assert hidden_dim % chunk_size == 0, "hidden_dim must be divisible by chunk_size"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.n_chunks = hidden_dim // chunk_size

        out_dim = 4 * hidden_dim + 2 * self.n_chunks
        self.lin = nn.Linear(input_dim + hidden_dim, out_dim)

    @staticmethod
    def cumax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return torch.cumsum(F.softmax(x, dim=dim), dim=dim)

    def forward(self, x: torch.Tensor, state):
        h_prev, c_prev = state
        gates = self.lin(torch.cat([x, h_prev], dim=-1))

        H = self.hidden_dim
        i, f, o, g = gates[:, : 4 * H].chunk(4, dim=-1)
        f_hat_lin, i_hat_lin = gates[:, 4 * H :].chunk(2, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        f_hat = self.cumax(f_hat_lin)              # (B, n_chunks)
        i_hat = 1.0 - self.cumax(i_hat_lin)        # (B, n_chunks)

        f_hat = f_hat.repeat_interleave(self.chunk_size, dim=-1)  # (B, H)
        i_hat = i_hat.repeat_interleave(self.chunk_size, dim=-1)  # (B, H)

        omega = f_hat * i_hat
        f = f * omega + (f_hat - omega)
        i = i * omega + (i_hat - omega)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ONLSTMPredictor(nn.Module):
    """ON-LSTM sequence model that predicts next-step delta."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        chunk_size: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.cells.append(ONLSTMCell(in_dim, hidden_dim, chunk_size=chunk_size))

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq, _ = x.size()
        device = x.device
        dtype = x.dtype

        hs = [torch.zeros(b, self.hidden_dim, device=device, dtype=dtype) for _ in range(self.num_layers)]
        cs = [torch.zeros(b, self.hidden_dim, device=device, dtype=dtype) for _ in range(self.num_layers)]

        for t in range(seq):
            inp = x[:, t, :]
            for li, cell in enumerate(self.cells):
                hs[li], cs[li] = cell(inp, (hs[li], cs[li]))
                inp = hs[li]
                if self.dropout > 0 and li < self.num_layers - 1:
                    inp = F.dropout(inp, p=self.dropout, training=self.training)

        return self.fc(hs[-1])


    def init_state(self, batch_size: int, device=None, dtype=None):
        """Initialize recurrent state (hs, cs) for stateful rollouts."""
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        hs = [torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype) for _ in range(self.num_layers)]
        cs = [torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype) for _ in range(self.num_layers)]
        return (hs, cs)

    def step(self, x_t: torch.Tensor, state, t: int = 0):
        """Single-step update. Returns (pred_delta, new_state)."""
        hs, cs = state
        inp = x_t
        for li, cell in enumerate(self.cells):
            hs[li], cs[li] = cell(inp, (hs[li], cs[li]))
            inp = hs[li]
            if self.dropout > 0 and li < self.num_layers - 1:
                inp = F.dropout(inp, p=self.dropout, training=self.training)

        out = self.fc(hs[-1])
        return out, (hs, cs)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_dtype = x.dtype
        x_f32 = x.float()
        norm = x_f32.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return ((x_f32 / norm) * self.scale * self.g).to(x_dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GatedRecurrentBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, int(dim * 2.6))
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, x, state):
        h = (x + state) * 0.7071
        res = h
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm.unsqueeze(1), h_norm.unsqueeze(1), h_norm.unsqueeze(1))
        h = res + attn_out.squeeze(1)
        candidate = h + self.ffn(self.norm2(h))
        z = torch.sigmoid(self.gate(torch.cat([candidate, state], dim=-1)))
        return z * candidate + (1 - z) * state


class DeepSapientHRM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 k_step: int = 2, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.k_step = k_step
        self.hidden_dim = hidden_dim
        self.embed = nn.Linear(input_dim, hidden_dim)

        self.L_blocks = nn.ModuleList([GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.H_blocks = nn.ModuleList([GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)])

        self.head = nn.Linear(hidden_dim, output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        b, seq, _ = x.size()
        h_L = [torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype) for _ in range(len(self.L_blocks))]
        h_H = [torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype) for _ in range(len(self.H_blocks))]

        for t in range(seq):
            curr_in = self.embed(x[:, t, :])

            if t % self.k_step == 0:
                h_in = h_L[-1].detach()
                for i, blk in enumerate(self.H_blocks):
                    h_H[i] = blk(h_in, h_H[i])
                    h_in = h_H[i]

            l_in = curr_in + h_H[-1]
            for i, blk in enumerate(self.L_blocks):
                h_L[i] = blk(l_in, h_L[i])
                l_in = h_L[i]

        return self.head(h_L[-1])


    def init_state(self, batch_size: int, device=None, dtype=None):
        """Initialize recurrent state (h_L, h_H) for stateful rollouts."""
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        h_L = [torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype) for _ in range(len(self.L_blocks))]
        h_H = [torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype) for _ in range(len(self.H_blocks))]
        return (h_L, h_H)

    def step(self, x_t: torch.Tensor, state, t: int):
        """Single-step update. Returns (pred_delta, new_state)."""
        h_L, h_H = state
        curr_in = self.embed(x_t)

        if t % self.k_step == 0:
            h_in = h_L[-1].detach()
            for i, blk in enumerate(self.H_blocks):
                h_H[i] = blk(h_in, h_H[i])
                h_in = h_H[i]

        l_in = curr_in + h_H[-1]
        for i, blk in enumerate(self.L_blocks):
            h_L[i] = blk(l_in, h_L[i])
            l_in = h_L[i]

        out = self.head(h_L[-1])
        return out, (h_L, h_H)


class PatchEncoder(nn.Module):
    """Lightweight CNN that encodes a (P×P) binary occupancy patch into an embedding."""

    def __init__(self, patch_size: int, embed_dim: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.proj = nn.Linear(16, embed_dim)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        # patch: (B, 1, P, P), values in {0,1}
        h = F.relu(self.conv1(patch))
        h = F.relu(self.conv2(h))
        # global average pool
        h = h.mean(dim=(-1, -2))  # (B, 16)
        return self.proj(h)       # (B, embed_dim)


class PatchConditionedPredictor(nn.Module):
    """Wraps a base predictor to condition on a local occupancy patch."""

    def __init__(self, base: nn.Module, patch_size: int, patch_embed_dim: int):
        super().__init__()
        self.base = base
        self.patch_encoder = PatchEncoder(patch_size, patch_embed_dim)

    def forward(self, pos_hist: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        # pos_hist: (B, T, 2), patch: (B, P, P) uint8/bool/float
        # Ensure patch dtype matches model dtype (important for bfloat16 HRM eval).
        if patch.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            patch = patch.float()
        patch = patch.to(pos_hist.dtype)
        patch = patch.unsqueeze(1)  # (B, 1, P, P)
        emb = self.patch_encoder(patch)  # (B, E)
        emb_seq = emb.unsqueeze(1).expand(-1, pos_hist.size(1), -1)
        x = torch.cat([pos_hist, emb_seq.to(pos_hist.dtype)], dim=-1)
        return self.base(x)


    def encode_patch(self, patch: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Encode a (B,P,P) patch once into (B,E) embedding."""
        if patch.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            patch = patch.float()
        patch = patch.to(dtype)
        patch = patch.unsqueeze(1)  # (B,1,P,P)
        return self.patch_encoder(patch)  # (B,E)

    def init_state(self, batch_size: int, device=None, dtype=None):
        if not hasattr(self.base, "init_state"):
            raise RuntimeError("Base model does not support init_state() for stateful rollout.")
        return self.base.init_state(batch_size, device=device, dtype=dtype)

    def step(self, pos_t: torch.Tensor, patch_emb: torch.Tensor, state, t: int):
        """One stateful step given current position and cached patch embedding."""
        x_t = torch.cat([pos_t, patch_emb.to(pos_t.dtype)], dim=-1)
        # Both bases expose .step(x_t, state, t) (ON-LSTM ignores t)
        return self.base.step(x_t, state, t)



def create_model(config: ModelConfig) -> nn.Module:
    input_dim = 2 + config.patch_embed_dim
    if config.model_type == "onlstm":
        base = ONLSTMPredictor(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=2,
            num_layers=config.num_layers,
            chunk_size=config.chunk_size,
            dropout=0.0,
        )
    else:
        base = DeepSapientHRM(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=2,
            k_step=2,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
        )
    return PatchConditionedPredictor(base, ENV_CONFIG["patch_size"], config.patch_embed_dim)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================================================================
# CHECKPOINTING HELPERS
# =============================================================================

def save_checkpoint(model: nn.Module, optimizer, scheduler, epoch: int,
                    loss: float, model_name: str, is_ddp: bool = False):
    os.makedirs(PATHS['checkpoints_dir'], exist_ok=True)
    checkpoint_path = f"{PATHS['checkpoints_dir']}/{model_name}_checkpoint.pt"

    model_state = model.module.state_dict() if is_ddp else model.state_dict()

    checkpoint = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_loss': loss,
    }

    torch.save(checkpoint, checkpoint_path)
    vol.commit()
    print(f"   💾 Checkpoint saved: epoch {epoch}, loss {loss:.6f}")


def load_checkpoint(model: nn.Module, optimizer, scheduler, model_name: str,
                    device='cuda', is_ddp: bool = False, verbose: bool = True) -> int:
    checkpoint_path = f"{PATHS['checkpoints_dir']}/{model_name}_checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        return 0

    if verbose:
        print("   📂 Found checkpoint, resuming training...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if is_ddp:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    if verbose:
        print(f"   ✓ Resumed from epoch {checkpoint['epoch']}, loss {best_loss:.6f}")
    return start_epoch


def cleanup_checkpoint(model_name: str):
    checkpoint_path = f"{PATHS['checkpoints_dir']}/{model_name}_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        vol.commit()
        print("   🗑️ Checkpoint cleaned up")

# =============================================================================
# ENVIRONMENT (DynamicMaze++ Preset M)
# =============================================================================

def _neighbors4(r: int, c: int):
    return ((r, c + 1), (r, c - 1), (r + 1, c), (r - 1, c))


def _bfs_path(static_map: np.ndarray, start: tuple, goal: tuple):
    H, W = static_map.shape
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
        return None
    if static_map[sr, sc] == 1 or static_map[gr, gc] == 1:
        return None
    q = deque()
    q.append((sr, sc))
    parent = {(sr, sc): None}
    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            path = []
            cur = (r, c)
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and static_map[nr, nc] == 0:
                if (nr, nc) not in parent:
                    parent[(nr, nc)] = (r, c)
                    q.append((nr, nc))
    return None


def _carve_L(static_map: np.ndarray, a: tuple, b: tuple, rng: np.random.Generator):
    r1, c1 = a
    r2, c2 = b
    if rng.random() < 0.5:
        for c in range(min(c1, c2), max(c1, c2) + 1):
            static_map[r1, c] = 0
        for r in range(min(r1, r2), max(r1, r2) + 1):
            static_map[r, c2] = 0
    else:
        for r in range(min(r1, r2), max(r1, r2) + 1):
            static_map[r, c1] = 0
        for c in range(min(c1, c2), max(c1, c2) + 1):
            static_map[r2, c] = 0


def extract_local_patch(static_map: np.ndarray, center_rc: Tuple[float, float], patch_size: int) -> np.ndarray:
    """Return (patch_size, patch_size) uint8 patch. 1=wall, 0=free."""
    r = int(round(float(center_rc[0])))
    c = int(round(float(center_rc[1])))
    rad = patch_size // 2
    patch = np.ones((patch_size, patch_size), dtype=np.uint8)
    H, W = static_map.shape
    for i in range(-rad, rad + 1):
        for j in range(-rad, rad + 1):
            rr = r + i
            cc = c + j
            if 0 <= rr < H and 0 <= cc < W:
                patch[i + rad, j + rad] = static_map[rr, cc].astype(np.uint8)
    return patch


class DynamicGridEnv:
    """DynamicMaze++ (Preset M) environment."""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = dict(config)
        self.size = int(config["grid_size"])
        self.n_dyn = int(config["n_dynamic"])
        # defer reset until called

    def reset(self, seed=None, map_seed=None, dyn_seed=None):
        # Map RNG can be fixed independently from dynamics RNG
        if map_seed is None:
            map_seed = seed
        if dyn_seed is None:
            dyn_seed = seed

        rng_map = np.random.default_rng(map_seed)
        self.rng = np.random.default_rng(dyn_seed)

        # 1) Static map
        self.static_map, rooms = self._generate_rooms_and_corridors(rng_map)

        # 2) Fixed start/goal
        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.goal_pos = np.array([float(self.size - 1), float(self.size - 1)], dtype=np.float32)
        self.static_map[0, 0] = 0
        self.static_map[self.size - 1, self.size - 1] = 0

        # Ensure connectivity
        path = _bfs_path(self.static_map, (0, 0), (self.size - 1, self.size - 1))
        if path is None:
            _carve_L(self.static_map, (0, 0), (self.size - 1, self.size - 1), rng_map)
            path = _bfs_path(self.static_map, (0, 0), (self.size - 1, self.size - 1))
        self._main_path = path or [(0, 0), (self.size - 1, self.size - 1)]

        # 3) Dynamic obstacles (order is stable: gates first, then patrollers, then drifters)
        self.dynamic_obs = []
        self._init_gates(rng_map)  # may carve alcoves -> should depend on map RNG for fixed-map suite
        self._init_patrollers(rooms)
        self._init_drifters()

        return self._get_obs()

    def clone(self):
        """Fast clone for rollout-MSE evaluation.

        Copies RNG state but avoids deep-copying large immutable structures
        (e.g., patroller routes) every step.
        """
        new = DynamicGridEnv.__new__(DynamicGridEnv)
        new.cfg = self.cfg
        new.size = self.size
        new.n_dyn = self.n_dyn
        new.static_map = self.static_map.copy()
        new.agent_pos = self.agent_pos.copy()
        new.goal_pos = self.goal_pos.copy()
        new._main_path = list(self._main_path)

        new.dynamic_obs = []
        for o in self.dynamic_obs:
            typ = o["type"]
            if typ == "gate":
                new_o = {
                    "type": "gate",
                    "closed": o["closed"],
                    "open": o["open"],
                    "is_closed": o["is_closed"],
                    "timer": o["timer"],
                    "closed_len": o["closed_len"],
                    "open_len": o["open_len"],
                    "pos": o["pos"].copy(),
                }
            elif typ == "patroller":
                new_o = {
                    "type": "patroller",
                    "route": o["route"],           # shared (immutable)
                    "idx": o["idx"],
                    "dwell": o["dwell"],
                    "waypoints": o["waypoints"],   # shared (immutable)
                    "pos": o["pos"].copy(),
                }
            else:  # drifter
                new_o = {
                    "type": "drifter",
                    "cell": o["cell"],
                    "heading": o["heading"],
                    "mode": o["mode"],
                    "mode_steps": o["mode_steps"],
                    "pos": o["pos"].copy(),
                }
            new.dynamic_obs.append(new_o)

        new.rng = np.random.default_rng()
        new.rng.bit_generator.state = copy.deepcopy(self.rng.bit_generator.state)
        return new

    def get_types(self) -> List[str]:
        return [o["type"] for o in self.dynamic_obs]

    def _generate_rooms_and_corridors(self, rng: np.random.Generator):
        S = self.size
        cfg = self.cfg
        static_map = np.ones((S, S), dtype=np.int8)
        rooms = []
        attempts = 0
        max_attempts = cfg["n_rooms"] * 40

        def overlaps(x, y, w, h):
            pad = cfg["room_padding"]
            for rx, ry, rw, rh in rooms:
                if (x < rx + rh + pad and x + h + pad > rx and
                    y < ry + rw + pad and y + w + pad > ry):
                    return True
            return False

        while len(rooms) < cfg["n_rooms"] and attempts < max_attempts:
            attempts += 1
            w = int(rng.integers(cfg["room_min_size"], cfg["room_max_size"] + 1))
            h = int(rng.integers(cfg["room_min_size"], cfg["room_max_size"] + 1))
            x = int(rng.integers(1, max(2, S - h - 1)))
            y = int(rng.integers(1, max(2, S - w - 1)))

            if overlaps(x, y, w, h):
                continue

            static_map[x:x + h, y:y + w] = 0
            rooms.append((x, y, w, h))

        if not rooms:
            static_map[:, :] = 0
            rooms = [(1, 1, S - 2, S - 2)]

        centers = []
        for x, y, w, h in rooms:
            centers.append((x + h // 2, y + w // 2))

        for i in range(1, len(centers)):
            _carve_L(static_map, centers[i - 1], centers[i], rng)

        start = (0, 0)
        goal = (S - 1, S - 1)

        def nearest_center(p):
            pr, pc = p
            best = centers[0]
            best_d = abs(best[0] - pr) + abs(best[1] - pc)
            for cc in centers[1:]:
                d = abs(cc[0] - pr) + abs(cc[1] - pc)
                if d < best_d:
                    best_d = d
                    best = cc
            return best

        _carve_L(static_map, start, nearest_center(start), rng)
        _carve_L(static_map, goal, nearest_center(goal), rng)

        static_map[0, 0] = 0
        static_map[S - 1, S - 1] = 0
        return static_map, centers

    def _random_free_cell(self):
        S = self.size
        for _ in range(10000):
            r = int(self.rng.integers(0, S))
            c = int(self.rng.integers(0, S))
            if self.static_map[r, c] == 0 and (r, c) not in [(0, 0), (S - 1, S - 1)]:
                return (r, c)
        for r in range(S):
            for c in range(S):
                if self.static_map[r, c] == 0 and (r, c) not in [(0, 0), (S - 1, S - 1)]:
                    return (r, c)
        return (0, 0)

    def _init_gates(self, rng_map: np.random.Generator):
        cfg = self.cfg
        path = self._main_path
        S = self.size

        # if gates are disabled, skip entirely
        if int(cfg.get("n_gates", 0)) <= 0:
            return

        if len(path) < 10:
            gate_cells = [path[len(path) // 2]]
        else:
            i1 = max(2, int(len(path) * 0.33))
            i2 = min(len(path) - 3, int(len(path) * 0.66))
            gate_cells = [path[i1], path[i2]] if i1 != i2 else [path[i1]]

        while len(gate_cells) < cfg["n_gates"] and len(path) > 4:
            idx = int(rng_map.integers(2, len(path) - 2))
            cell = path[idx]
            if cell not in gate_cells:
                gate_cells.append(cell)
        gate_cells = gate_cells[: cfg["n_gates"]]

        for gc in gate_cells:
            gr, gc2 = gc

            candidates = []
            for nr, nc in _neighbors4(gr, gc2):
                if 0 <= nr < S and 0 <= nc < S and self.static_map[nr, nc] == 1:
                    candidates.append((nr, nc))
            if candidates:
                open_pos = candidates[int(rng_map.integers(0, len(candidates)))]
                self.static_map[open_pos[0], open_pos[1]] = 0
            else:
                neigh_free = []
                for nr, nc in _neighbors4(gr, gc2):
                    if 0 <= nr < S and 0 <= nc < S and self.static_map[nr, nc] == 0 and (nr, nc) != (gr, gc2):
                        neigh_free.append((nr, nc))
                open_pos = neigh_free[0] if neigh_free else (gr, gc2)

            # Timing uses dynamics RNG (self.rng) so dynamics vary even in fixed-map suite
            period = int(self.rng.integers(cfg["gate_period_min"], cfg["gate_period_max"] + 1))
            open_len = int(self.rng.integers(cfg["gate_open_min"], cfg["gate_open_max"] + 1))
            closed_len = max(1, period - open_len)

            gate = {
                "type": "gate",
                "closed": (gr, gc2),
                "open": open_pos,
                "is_closed": True,
                "timer": int(self.rng.integers(1, closed_len + 1)),
                "closed_len": closed_len,
                "open_len": open_len,
                "pos": np.array([float(gr), float(gc2)], dtype=np.float32),
            }
            self.dynamic_obs.append(gate)

    def _init_patrollers(self, room_centers):
        cfg = self.cfg
        anchors = room_centers if room_centers else [self._random_free_cell()]
        for _ in range(int(cfg["n_patrollers"])):
            n_wp = int(self.rng.integers(3, 6))
            waypoints = []
            for _k in range(n_wp):
                wp = anchors[int(self.rng.integers(0, len(anchors)))]
                if wp not in waypoints:
                    waypoints.append(wp)
            if len(waypoints) < 2:
                waypoints = [anchors[0], anchors[-1]]

            route = []
            ok = True
            for i in range(len(waypoints)):
                a = waypoints[i]
                b = waypoints[(i + 1) % len(waypoints)]
                seg = _bfs_path(self.static_map, a, b)
                if seg is None:
                    ok = False
                    break
                if route:
                    route.extend(seg[1:])
                else:
                    route.extend(seg)
            if (not ok) or (len(route) < 2):
                route = [self._random_free_cell() for _ in range(30)]

            start_idx = int(self.rng.integers(0, len(route)))

            pat = {
                "type": "patroller",
                "route": route,
                "idx": start_idx,
                "dwell": 0,
                "waypoints": set(waypoints),
                "pos": np.array([float(route[start_idx][0]), float(route[start_idx][1])], dtype=np.float32),
            }
            self.dynamic_obs.append(pat)

    def _init_drifters(self):
        cfg = self.cfg
        headings = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        modes = ["left", "right", "random"]
        for _ in range(int(cfg["n_drifters"])):
            r, c = self._random_free_cell()
            dr, dc = headings[int(self.rng.integers(0, len(headings)))]
            mode = modes[int(self.rng.integers(0, len(modes)))]
            steps = int(self.rng.integers(cfg["drifter_regime_min"], cfg["drifter_regime_max"] + 1))
            d = {
                "type": "drifter",
                "cell": (r, c),
                "heading": (dr, dc),
                "mode": mode,
                "mode_steps": steps,
                "pos": np.array([float(r), float(c)], dtype=np.float32),
            }
            self.dynamic_obs.append(d)

    @staticmethod
    def _turn_left(h):
        dr, dc = h
        return (-dc, dr)

    @staticmethod
    def _turn_right(h):
        dr, dc = h
        return (dc, -dr)

    @staticmethod
    def _reverse(h):
        dr, dc = h
        return (-dr, -dc)

    def step_physics(self):
        cfg = self.cfg
        S = self.size
        for o in self.dynamic_obs:
            typ = o["type"]

            if typ == "gate":
                o["timer"] -= 1
                if o["timer"] <= 0:
                    if o["is_closed"]:
                        o["is_closed"] = False
                        o["timer"] = o["open_len"]
                        r, c = o["open"]
                        o["pos"][:] = (float(r), float(c))
                    else:
                        o["is_closed"] = True
                        jitter = int(self.rng.integers(-1, 2))
                        o["timer"] = max(1, o["closed_len"] + jitter)
                        r, c = o["closed"]
                        o["pos"][:] = (float(r), float(c))

            elif typ == "patroller":
                if o["dwell"] > 0:
                    o["dwell"] -= 1
                else:
                    o["idx"] = (o["idx"] + 1) % len(o["route"])
                    r, c = o["route"][o["idx"]]
                    o["pos"][:] = (float(r), float(c))
                    cell = (r, c)
                    if cell in o["waypoints"] and self.rng.random() < 0.35:
                        o["dwell"] = int(self.rng.integers(1, 3))

            elif typ == "drifter":
                r, c = o["cell"]
                heading = o["heading"]
                mode = o["mode"]

                o["mode_steps"] -= 1
                if o["mode_steps"] <= 0:
                    o["mode"] = ["left", "right", "random"][int(self.rng.integers(0, 3))]
                    o["mode_steps"] = int(self.rng.integers(cfg["drifter_regime_min"], cfg["drifter_regime_max"] + 1))
                    mode = o["mode"]

                forward = heading
                left = self._turn_left(heading)
                right = self._turn_right(heading)
                rev = self._reverse(heading)

                def can_move(h):
                    dr, dc = h
                    nr, nc = r + dr, c + dc
                    return 0 <= nr < S and 0 <= nc < S and self.static_map[nr, nc] == 0

                chosen = None
                if can_move(forward) and self.rng.random() < cfg["drifter_p_forward"]:
                    chosen = forward
                else:
                    if mode == "left":
                        for cand in (left, forward, right, rev):
                            if can_move(cand):
                                chosen = cand
                                break
                    elif mode == "right":
                        for cand in (right, forward, left, rev):
                            if can_move(cand):
                                chosen = cand
                                break
                    else:
                        cands = [d for d in (forward, left, right, rev) if can_move(d)]
                        if cands:
                            chosen = cands[int(self.rng.integers(0, len(cands)))]

                if chosen is None:
                    chosen = (0, 0)

                dr, dc = chosen
                nr, nc = r + dr, c + dc
                if 0 <= nr < S and 0 <= nc < S and self.static_map[nr, nc] == 0:
                    r, c = nr, nc

                o["cell"] = (r, c)
                o["heading"] = chosen if chosen != (0, 0) else heading
                o["pos"][:] = (float(r), float(c))

        return self._get_obs()

    def _get_obs(self):
        return np.array([o["pos"] for o in self.dynamic_obs], dtype=np.float32)

# =============================================================================
# SPACE-TIME A* (uses patch-conditioned predictor)
# =============================================================================

class SpaceTimeAStar:
    def __init__(self, env: DynamicGridEnv, model: nn.Module, device: torch.device,
                 pred_horizon: int, patch_size: int):
        self.env = env
        self.model = model
        self.device = device
        self.pred_horizon = int(pred_horizon)
        self.patch_size = int(patch_size)
        self.model.eval()

    def _compute_patches(self, last_positions: np.ndarray) -> np.ndarray:
        # last_positions: (n_dyn,2) in grid coords
        patches = [extract_local_patch(self.env.static_map, tuple(last_positions[i]), self.patch_size)
                   for i in range(last_positions.shape[0])]
        return np.stack(patches, axis=0)  # (n_dyn,P,P)

    def predict_future(self, obs_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """Predict future obstacle positions for horizon.

        Returns:
            future_positions: (pred_horizon, n_dyn, 2) in grid coords
            first_delta: (n_dyn,2) in *normalized* coords (kept for backward compatibility; not used for one-step MSE)
            expansions_dummy: kept for compatibility
        """
        # obs_history: (n_dyn, T, 2) grid coords
        last_pos = obs_history[:, -1, :]  # (n_dyn,2)
        patches_np = self._compute_patches(last_pos)

        # Normalize positions for the predictor
        curr_seq = torch.tensor(obs_history / self.env.size, dtype=torch.float32, device=self.device)
        patch_t = torch.tensor(patches_np, dtype=torch.uint8, device=self.device)

        # Match model dtype
        curr_seq = curr_seq.to(next(self.model.parameters()).dtype)

        B, T, _ = curr_seq.shape

        # Encode patch once and warm-start state over the observed history
        with torch.no_grad():
            patch_emb = self.model.encode_patch(patch_t, dtype=curr_seq.dtype)
            state = self.model.init_state(B, device=self.device, dtype=curr_seq.dtype)

            pred_delta = None
            for t in range(T):
                pred_delta, state = self.model.step(curr_seq[:, t, :], patch_emb, state, t)

            first_delta = pred_delta  # normalized delta at the first rollout step

            future_obs = []
            curr_pos = curr_seq[:, -1, :]
            for s in range(self.pred_horizon):
                next_pos = curr_pos + pred_delta
                future_obs.append((next_pos.float().cpu().numpy() * self.env.size))
                curr_pos = next_pos
                pred_delta, state = self.model.step(curr_pos, patch_emb, state, T + s)

        future_obs = np.array(future_obs, dtype=np.float32)  # (H,n_dyn,2)
        first_delta_np = first_delta.float().cpu().numpy() if first_delta is not None else np.zeros_like(last_pos, dtype=np.float32)
        return future_obs, first_delta_np, 0

    def get_next_action(self, start, goal, obs_history: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray, np.ndarray, int]:
        """Return (next_cell, future_obs, first_pred_next, expansions)."""
        future_obs, first_delta, _ = self.predict_future(obs_history)

        # Predicted next positions for one-step MSE metric (grid coords)
        # Use the first step of the predicted rollout to avoid unit mismatches.
        pred_next = future_obs[0] if len(future_obs) > 0 else obs_history[:, -1, :].copy()

        # A* in (r,c,t)
        start_node = (int(start[0]), int(start[1]), 0)
        pq = [(0, 0, start_node)]
        g_score = {start_node: 0}
        came_from = {}
        best_node, min_h = None, float('inf')
        expansions = 0

        while pq:
            f, g, curr_node = heapq.heappop(pq)
            expansions += 1
            r, c, t = curr_node

            if (r, c) == (int(goal[0]), int(goal[1])):
                nxt = self._trace(came_from, curr_node, start_node)
                return nxt, future_obs, pred_next, expansions

            if t >= self.pred_horizon - 1:
                h = abs(r - goal[0]) + abs(c - goal[1])
                if h < min_h:
                    min_h = h
                    best_node = curr_node
                continue

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:
                nr, nc, nt = r + dr, c + dc, t + 1

                if not (0 <= nr < self.env.size and 0 <= nc < self.env.size):
                    continue
                if self.env.static_map[nr, nc] == 1:
                    continue
                if np.any(np.linalg.norm(future_obs[nt] - np.array([nr, nc]), axis=1) < 1.0):
                    continue

                new_g = g + 1
                neigh = (nr, nc, nt)
                if new_g < g_score.get(neigh, float('inf')):
                    g_score[neigh] = new_g
                    h = abs(nr - goal[0]) + abs(nc - goal[1])
                    heapq.heappush(pq, (new_g + h, new_g, neigh))
                    came_from[neigh] = curr_node

        if best_node:
            nxt = self._trace(came_from, best_node, start_node)
            return nxt, future_obs, pred_next, expansions

        return (int(start[0]), int(start[1])), future_obs, pred_next, expansions

    def _trace(self, came_from, curr, start):
        path = []
        while curr in came_from:
            path.append(curr)
            curr = came_from[curr]
        return (path[-1][0], path[-1][1]) if path else (int(start[0]), int(start[1]))


# =============================================================================
# MODAL FUNCTIONS - DATA COLLECTION
# =============================================================================

def _load_chunk(filepath: str):
    """Load a data chunk (top-level for pickling)."""
    x, p, y = torch.load(filepath, weights_only=False)
    return (
        np.array(x, dtype=np.float32),
        np.array(p, dtype=np.uint8),
        np.array(y, dtype=np.float32),
    )


@app.function(image=image, volumes={"/data": vol}, cpu=1.0, timeout=7200)
def collect_data_chunk(worker_id: int, n_episodes: int) -> str:
    """Collect trajectory data with local patches and k-step targets."""
    cfg = ENV_CONFIG
    env = DynamicGridEnv(cfg)
    X, P, Y = [], [], []

    H = int(cfg["obs_history"])
    K = int(cfg["k_rollout"])
    PS = int(cfg["patch_size"])
    steps_per_ep = int(cfg["physics_steps_per_episode"])

    for ep in range(n_episodes):
        env.reset(seed=worker_id * 10_000 + ep)
        hist = []

        for _t in range(steps_per_ep):
            obs = env.step_physics()
            hist.append(obs)

            if len(hist) >= H + K:
                # History window ends at time t-K (the last observed position for the sample)
                past = np.array(hist[-(H + K):-K], dtype=np.float32) / env.size  # (H,n_dyn,2)

                # Positions for computing true deltas: times (t-K) .. t  (K+1 positions)
                pos_seq = np.array(hist[-(K + 1):], dtype=np.float32) / env.size  # (K+1,n_dyn,2)
                deltas = pos_seq[1:] - pos_seq[:-1]  # (K,n_dyn,2)

                centers = (pos_seq[0] * env.size)  # (n_dyn,2) in grid coords at time t-K

                for j in range(env.n_dyn):
                    X.append(past[:, j, :])
                    P.append(extract_local_patch(env.static_map, tuple(centers[j]), PS))
                    Y.append(deltas[:, j, :])

    os.makedirs(PATHS['data_dir'], exist_ok=True)
    filepath = f"{PATHS['data_dir']}/chunk_{worker_id}.pt"
    torch.save((X, P, Y), filepath)
    vol.commit()
    return filepath


@app.function(image=image, volumes={"/data": vol}, cpu=8.0, memory=65536, timeout=7200)
def merge_chunks(chunk_files: List[str]) -> str:
    """Merge data chunks into a single tensor dataset."""
    print(f"--> Merging {len(chunk_files)} chunks...")

    with ProcessPoolExecutor(8) as exe:
        results = list(exe.map(_load_chunk, chunk_files))

    print("--> Concatenating arrays...")
    X = np.concatenate([r[0] for r in results], axis=0)  # float32
    P = np.concatenate([r[1] for r in results], axis=0)  # uint8
    Y = np.concatenate([r[2] for r in results], axis=0)  # float32

    print(f"--> Total samples: {len(X):,}")

    # Basic sanitization
    valid = ~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(Y).any(axis=(1, 2))
    if not valid.all():
        print(f"⚠️ Dropped {(~valid).sum()} corrupted samples")
        X = X[valid]
        P = P[valid]
        Y = Y[valid]

    X_t = torch.from_numpy(X)  # float32
    P_t = torch.from_numpy(P)  # uint8
    Y_t = torch.from_numpy(Y)  # float32 (N,K,2)

    torch.save((X_t, P_t, Y_t), PATHS["merged_data"])
    vol.commit()

    print(f"✅ Saved {len(X_t):,} samples to {PATHS['merged_data']}")
    return PATHS["merged_data"]


@app.function(image=image, volumes={"/data": vol}, cpu=1.0)
def check_cached_data() -> bool:
    vol.reload()
    return os.path.exists(PATHS["merged_data"])


@app.function(image=image, volumes={"/data": vol}, cpu=1.0)
def check_completed_models() -> Dict[str, bool]:
    vol.reload()
    status = {}
    for name in MODEL_CONFIGS.keys():
        model_path = f"{PATHS['models_dir']}/{name}.pt"
        status[name] = os.path.exists(model_path)
    return status

# =============================================================================
# TRAINING (multi-step loss + scheduled sampling)
# =============================================================================

def teacher_forcing_prob(epoch: int, total_epochs: int) -> float:
    """Linear decay of teacher forcing prob over first 60% of training."""
    decay_portion = 0.6
    if total_epochs <= 1:
        return 0.0
    frac = epoch / max(1, int(total_epochs * decay_portion))
    return float(max(0.0, 1.0 - frac))


def rollout_loss(model: nn.Module, pos_hist: torch.Tensor, patch: torch.Tensor, true_deltas: torch.Tensor,
                 k: int, p_teacher: float, weights: torch.Tensor) -> torch.Tensor:
    """k-step rollout loss with scheduled sampling (FAST stateful implementation).

    Important: this avoids calling model() k times over a full history window.
    Instead, we:
      1) encode the local patch once,
      2) warm-start the recurrent state by stepping through the H history positions,
      3) autoregressively roll out k steps using the cached recurrent state.

    This reduces compute from O(k*H) forwards to O(H+k) steps.
    """
    # pos_hist: (B,H,2), patch: (B,P,P), true_deltas: (B,k,2)
    B, H, _ = pos_hist.shape
    device = pos_hist.device
    dtype = pos_hist.dtype

    # DDP wraps the model and does not expose custom helper methods.
    # Unwrap to access encode_patch/init_state/step.
    m = model.module if hasattr(model, "module") else model

    # Encode patch once (B,E)
    patch_emb = m.encode_patch(patch, dtype=dtype)

    # Warm-start recurrent state on the observed history
    state = m.init_state(B, device=device, dtype=dtype)
    pred_delta = None
    for t in range(H):
        pred_delta, state = m.step(pos_hist[:, t, :], patch_emb, state, t)

    # Autoregressive rollout with scheduled sampling
    curr_pos = pos_hist[:, -1, :]
    total = 0.0

    for s in range(k):
        td = true_deltas[:, s, :].to(dtype)
        mse = F.mse_loss(pred_delta, td, reduction="mean")
        total = total + weights[s] * mse

        true_next = curr_pos + td
        pred_next = curr_pos + pred_delta

        if p_teacher >= 1.0:
            next_pos = true_next
        elif p_teacher <= 0.0:
            next_pos = pred_next
        else:
            mask = (torch.rand(B, device=device) < p_teacher).unsqueeze(-1)
            next_pos = torch.where(mask, true_next, pred_next)

        curr_pos = next_pos
        pred_delta, state = m.step(curr_pos, patch_emb, state, H + s)

    return total





@app.function(image=image, gpu="H100", volumes={"/data": vol}, timeout=86400)
def train_onlstm_model(model_name: str, merged_path: str):
    from tqdm import tqdm
    from torch.amp import autocast, GradScaler
    import time

    vol.reload()
    cfg = ENV_CONFIG
    config = MODEL_CONFIGS[model_name]

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} (multi-step + patch)")
    print(f"{'='*60}")

    print(f"--> Loading data from {merged_path}...")
    X, P, Y = torch.load(merged_path, weights_only=False)
    print(f"Dataset: {len(X):,} samples")

    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, P, Y),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    model = create_model(config).cuda()
    params = count_parameters(model)
    print(f"Model parameters: {params:,}")

    opt = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=config.lr,
        steps_per_epoch=len(dl),
        epochs=config.epochs
    )
    scaler = GradScaler('cuda')

    start_epoch = load_checkpoint(model, opt, scheduler, model_name, device='cuda')

    # Loss weights (geometric decay)
    K = int(cfg["k_rollout"])
    weights = torch.tensor([0.9 ** s for s in range(K)], device='cuda', dtype=torch.float32)

    model.train()
    start_time = time.time()

    for ep in range(start_epoch, config.epochs):
        p_teacher = teacher_forcing_prob(ep, config.epochs)
        ep_loss = 0.0
        pbar = tqdm(dl, desc=f"Ep {ep+1}/{config.epochs} tf={p_teacher:.2f}", leave=False)

        for bx, bp, by in pbar:
            bx = bx.cuda(non_blocking=True)
            bp = bp.cuda(non_blocking=True)
            by = by.cuda(non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast('cuda'):
                loss = rollout_loss(model, bx, bp, by, K, p_teacher, weights)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            ep_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = ep_loss / max(1, len(dl))
        print(f"Epoch {ep+1}/{config.epochs} | Loss: {avg_loss:.6f} | teacher_forcing={p_teacher:.2f}")

        if (ep + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, opt, scheduler, ep, avg_loss, model_name)

    train_time = time.time() - start_time
    print(f"Training time: {train_time/60:.1f} min")

    os.makedirs(PATHS['models_dir'], exist_ok=True)
    model_path = f"{PATHS['models_dir']}/{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    vol.commit()

    cleanup_checkpoint(model_name)

    print(f"✅ Saved to {model_path}")
    return {"name": model_name, "params": params, "train_time": train_time}


@app.function(image=image, gpu="B200", volumes={"/data": vol}, timeout=86400)
def train_hrm_model(model_name: str, merged_path: str):
    from tqdm import tqdm
    import time

    vol.reload()
    cfg = ENV_CONFIG
    config = MODEL_CONFIGS[model_name]

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} (multi-step + patch)")
    print(f"{'='*60}")

    print(f"--> Loading data from {merged_path}...")
    X, P, Y = torch.load(merged_path, weights_only=False)
    print(f"Dataset: {len(X):,} samples")

    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, P, Y),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    model = create_model(config).cuda().to(torch.bfloat16)
    params = count_parameters(model)
    print(f"Model parameters: {params:,}")

    opt = optim.AdamW(model.parameters(), lr=config.lr, fused=True)
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=config.lr,
        steps_per_epoch=len(dl),
        epochs=config.epochs
    )

    start_epoch = load_checkpoint(model, opt, scheduler, model_name, device='cuda')

    K = int(cfg["k_rollout"])
    weights = torch.tensor([0.9 ** s for s in range(K)], device='cuda', dtype=torch.float32)

    model.train()
    start_time = time.time()

    for ep in range(start_epoch, config.epochs):
        p_teacher = teacher_forcing_prob(ep, config.epochs)
        ep_loss = 0.0
        valid_batches = 0
        pbar = tqdm(dl, desc=f"Ep {ep+1}/{config.epochs} tf={p_teacher:.2f}", leave=False)

        for bx, bp, by in pbar:
            bx = bx.cuda(non_blocking=True).to(torch.bfloat16)
            bp = bp.cuda(non_blocking=True)
            by = by.cuda(non_blocking=True).to(torch.bfloat16)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = rollout_loss(model, bx, bp, by, K, p_teacher, weights)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            ep_loss += float(loss.item())
            valid_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = ep_loss / max(1, valid_batches)
        print(f"Epoch {ep+1}/{config.epochs} | Loss: {avg_loss:.6f} | teacher_forcing={p_teacher:.2f}")

        if (ep + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, opt, scheduler, ep, avg_loss, model_name)

    train_time = time.time() - start_time
    print(f"Training time: {train_time/60:.1f} min")

    os.makedirs(PATHS['models_dir'], exist_ok=True)
    model_path = f"{PATHS['models_dir']}/{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    vol.commit()

    cleanup_checkpoint(model_name)

    print(f"✅ Saved to {model_path}")
    return {"name": model_name, "params": params, "train_time": train_time}

# =============================================================================
# DDP TRAINING FOR hrm_10m (4× GPU)
# =============================================================================

def ddp_setup(rank: int, world_size: int):
    import torch.distributed as dist
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def ddp_cleanup():
    import torch.distributed as dist
    dist.destroy_process_group()


def ddp_train_worker(rank: int, world_size: int, merged_path: str, config_dict: dict):
    from tqdm import tqdm
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    import time

    ddp_setup(rank, world_size)

    config = ModelConfig(**config_dict)
    cfg = ENV_CONFIG

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Training {config.name.upper()} with {world_size}x GPU DDP (multi-step + patch)")
        print(f"{'='*60}")

    if rank == 0:
        print("--> Loading data...")
    X, P, Y = torch.load(merged_path, weights_only=False, map_location='cpu')

    dataset = torch.utils.data.TensorDataset(X, P, Y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    model = create_model(config).to(rank).to(torch.bfloat16)
    model = DDP(model, device_ids=[rank])

    if rank == 0:
        params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {params:,}")
        print(f"Global batch size: {config.batch_size * world_size}")
        print(f"Batches per epoch: {len(dl)}")

    scaled_lr = config.lr * (world_size ** 0.5)
    opt = optim.AdamW(model.parameters(), lr=scaled_lr, fused=True)
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=scaled_lr,
        steps_per_epoch=len(dl),
        epochs=config.epochs
    )

    device = torch.device('cuda', rank)
    start_epoch = load_checkpoint(
        model, opt, scheduler, config.name,
        device=device, is_ddp=True, verbose=(rank == 0)
    )

    start_epoch_tensor = torch.tensor([start_epoch], dtype=torch.int64, device=device)
    dist.broadcast(start_epoch_tensor, src=0)
    start_epoch = int(start_epoch_tensor.item())
    dist.barrier()

    K = int(cfg["k_rollout"])
    weights = torch.tensor([0.9 ** s for s in range(K)], device=device, dtype=torch.float32)

    model.train()
    start_time = time.time()

    for ep in range(start_epoch, config.epochs):
        sampler.set_epoch(ep)
        ep_loss = torch.zeros(1, device=device)
        valid_batches = 0

        p_teacher = teacher_forcing_prob(ep, config.epochs)

        pbar = tqdm(dl, desc=f"Ep {ep+1}/{config.epochs} tf={p_teacher:.2f}", disable=(rank != 0), leave=False)
        for bx, bp, by in pbar:
            bx = bx.to(device, non_blocking=True).to(torch.bfloat16)
            bp = bp.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True).to(torch.bfloat16)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = rollout_loss(model, bx, bp, by, K, p_teacher, weights)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            ep_loss += loss.detach()
            valid_batches += 1
            if rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.6f}")

        dist.all_reduce(ep_loss, op=dist.ReduceOp.SUM)

        if rank == 0:
            avg_loss = ep_loss.item() / (world_size * max(1, valid_batches))
            print(f"Epoch {ep+1}/{config.epochs} | Loss: {avg_loss:.6f} | teacher_forcing={p_teacher:.2f}")
            if (ep + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(model, opt, scheduler, ep, avg_loss, config.name, is_ddp=True)

        dist.barrier()

    train_time = time.time() - start_time

    if rank == 0:
        print(f"Training time: {train_time/60:.1f} min")
        os.makedirs(PATHS['models_dir'], exist_ok=True)
        model_path = f"{PATHS['models_dir']}/{config.name}.pt"
        torch.save(model.module.state_dict(), model_path)
        print(f"✅ Saved to {model_path}")
        cleanup_checkpoint(config.name)

    ddp_cleanup()
    return train_time


@app.function(
    image=image,
    gpu="B200:4",
    volumes={"/data": vol},
    timeout=86400,
    memory=65536
)
def train_hrm_10m_ddp(merged_path: str):
    import torch.multiprocessing as mp
    import time

    vol.reload()
    config = MODEL_CONFIGS["hrm_10m"]
    world_size = torch.cuda.device_count()
    print(f"🚀 Launching {world_size}-GPU DDP Training for hrm_10m")

    config_dict = {
        "name": config.name,
        "model_type": config.model_type,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "chunk_size": config.chunk_size,
        "patch_embed_dim": config.patch_embed_dim,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "epochs": config.epochs,
        "gpu": config.gpu,
    }

    start_time = time.time()
    mp.spawn(
        ddp_train_worker,
        args=(world_size, merged_path, config_dict),
        nprocs=world_size,
        join=True
    )
    train_time = time.time() - start_time
    vol.commit()

    # Count params for return
    model = create_model(config)
    params = count_parameters(model)
    print("✅ hrm_10m DDP training complete")
    return {"name": "hrm_10m", "params": params, "train_time": train_time}

# =============================================================================
# EVALUATION (metrics + ablations)
# =============================================================================

def make_suite_env_config(base_cfg: Dict[str, Any], suite: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    # override obstacle composition
    cfg["n_gates"] = int(suite.get("n_gates", cfg.get("n_gates", 0)))
    cfg["n_patrollers"] = int(suite.get("n_patrollers", cfg.get("n_patrollers", 0)))
    cfg["n_drifters"] = int(suite.get("n_drifters", cfg.get("n_drifters", 0)))
    cfg["n_dynamic"] = int(cfg["n_gates"] + cfg["n_patrollers"] + cfg["n_drifters"])
    return cfg


@app.cls(image=image, gpu="A10", volumes={"/data": vol}, max_containers=30, timeout=3600)
class Evaluator:
    """Loads all models once per container and evaluates tasks."""

    @modal.enter()
    def setup(self):
        self.device = torch.device("cuda")
        self.models = {}

        for name, config in MODEL_CONFIGS.items():
            model_path = f"{PATHS['models_dir']}/{name}.pt"
            if os.path.exists(model_path):
                model = create_model(config).to(self.device)
                state_dict = torch.load(model_path, weights_only=True)
                model.load_state_dict(state_dict)

                if config.model_type == "hrm":
                    model = model.to(torch.bfloat16)

                model.eval()
                self.models[name] = model
                print(f"✓ Loaded {name}")
            else:
                print(f"✗ Missing {name}")

    @modal.method()
    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run one (suite, seed, model) evaluation task."""
        suite = task["suite"]
        seed = int(task["seed"])
        model_name = task["model"]

        model = self.models[model_name]

        suite_cfg = make_suite_env_config(ENV_CONFIG, suite)

        env = DynamicGridEnv(suite_cfg)

        # Map seeding logic
        map_mode = suite.get("map_mode", "random")
        if map_mode == "fixed":
            map_seed = int(suite.get("map_seed", 12345))
            dyn_seed = seed + 10_000
        else:
            map_seed = seed + 1_000
            dyn_seed = seed + 10_000

        env.reset(seed=None, map_seed=map_seed, dyn_seed=dyn_seed)
        types = env.get_types()

        planner = SpaceTimeAStar(
            env=env,
            model=model,
            device=self.device,
            pred_horizon=int(suite.get("pred_horizon", ENV_CONFIG["pred_horizon"])),
            patch_size=int(ENV_CONFIG["patch_size"])
        )

        H = int(ENV_CONFIG["obs_history"])
        max_steps = int(ENV_CONFIG["max_agent_steps"])
        collision_r = float(ENV_CONFIG["collision_radius"])
        rollout_k = int(ENV_CONFIG.get("eval_rollout_mse_k", 0))

        # Initial history for obstacles
        hist = [env.step_physics() for _ in range(H)]

        # Error accumulators (SSE + count)
        one_sse = 0.0
        one_cnt = 0
        one_type_sse = {"gate": 0.0, "patroller": 0.0, "drifter": 0.0}
        one_type_cnt = {"gate": 0, "patroller": 0, "drifter": 0}

        roll_sse = 0.0
        roll_cnt = 0
        roll_type_sse = {"gate": 0.0, "patroller": 0.0, "drifter": 0.0}
        roll_type_cnt = {"gate": 0, "patroller": 0, "drifter": 0}

        expansions_total = 0
        failure_type = "timeout"
        success = 0
        steps_taken = 0

        for step in range(max_steps):
            steps_taken = step + 1

            h_np = np.array(hist[-H:], dtype=np.float32).transpose(1, 0, 2)  # (n_dyn,H,2)

            # Optional rollout-mse: clone env BEFORE planning / stepping
            true_future = None
            if rollout_k > 0:
                env_clone = env.clone()
                true_future = np.array([env_clone.step_physics() for _ in range(rollout_k)], dtype=np.float32)  # (K,n_dyn,2)

            # Plan
            (nr, nc), future_obs, pred_next, expansions = planner.get_next_action(env.agent_pos, env.goal_pos, h_np)
            expansions_total += expansions

            env.agent_pos = np.array([float(nr), float(nc)], dtype=np.float32)

            # Step physics (this produces the true next obstacle positions)
            obs_next = env.step_physics()
            hist.append(obs_next)

            # One-step prediction SSE (pred_next is model's predicted next pos at this planning step)
            err = (pred_next - obs_next).astype(np.float32)  # (n_dyn,2)
            one_sse += float(np.sum(err ** 2))
            one_cnt += int(err.size)

            for i, typ in enumerate(types):
                if typ in one_type_sse:
                    e_i = err[i]
                    one_type_sse[typ] += float(np.sum(e_i ** 2))
                    one_type_cnt[typ] += int(e_i.size)

            # k-step rollout SSE
            if rollout_k > 0 and true_future is not None:
                pred_future_k = future_obs[:rollout_k].astype(np.float32)  # (K,n_dyn,2)
                rerr = pred_future_k - true_future
                roll_sse += float(np.sum(rerr ** 2))
                roll_cnt += int(rerr.size)

                for i, typ in enumerate(types):
                    if typ in roll_type_sse:
                        e_i = rerr[:, i, :]
                        roll_type_sse[typ] += float(np.sum(e_i ** 2))
                        roll_type_cnt[typ] += int(e_i.size)

            # Goal check
            if np.linalg.norm(env.agent_pos - env.goal_pos) < 0.5:
                success = 1
                failure_type = "success"
                break

            # Static collision check
            if env.static_map[int(nr), int(nc)] == 1:
                failure_type = "static_collision"
                break

            # Dynamic collision check
            if np.any(np.linalg.norm(obs_next - env.agent_pos, axis=1) < collision_r):
                failure_type = "collision"
                break

        return {
            "suite": suite["name"],
            "seed": seed,
            "model": model_name,
            "success": int(success),
            "failure_type": failure_type,
            "steps": int(steps_taken),
            "expansions": int(expansions_total),
            "one_sse": float(one_sse),
            "one_cnt": int(one_cnt),
            "one_type_sse": one_type_sse,
            "one_type_cnt": one_type_cnt,
            "roll_sse": float(roll_sse),
            "roll_cnt": int(roll_cnt),
            "roll_type_sse": roll_type_sse,
            "roll_type_cnt": roll_type_cnt,
        }


@app.function(image=image, volumes={"/data": vol}, cpu=2.0, timeout=1800)
def aggregate_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate all per-task episode rows into a structured results JSON."""
    # Build nested aggregations: suites -> model -> stats
    suites: Dict[str, Any] = {}
    meta = {
        "grid_size": ENV_CONFIG["grid_size"],
        "obs_history": ENV_CONFIG["obs_history"],
        "patch_size": ENV_CONFIG["patch_size"],
        "k_rollout_train": ENV_CONFIG["k_rollout"],
        "eval_rollout_mse_k": ENV_CONFIG.get("eval_rollout_mse_k", 0),
        "models": {name: {
            "type": cfg.model_type,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "chunk_size": cfg.chunk_size,
            "patch_embed_dim": cfg.patch_embed_dim,
            "params": count_parameters(create_model(cfg)),
        } for name, cfg in MODEL_CONFIGS.items()}
    }

    # Map suite name -> suite config
    suite_cfg_map = {s["name"]: s for s in EVAL_SUITES}

    for row in rows:
        sname = row["suite"]
        mname = row["model"]
        suites.setdefault(sname, {
            "config": suite_cfg_map.get(sname, {}),
            "models": {}
        })
        m = suites[sname]["models"].setdefault(mname, {
            "type": MODEL_CONFIGS[mname].model_type,
            "successes": 0,
            "total": 0,
            "failures": {"collision": 0, "timeout": 0, "static_collision": 0},
            "steps_sum": 0,
            "steps_sum_success": 0,
            "success_count": 0,
            "expansions_sum": 0,
            "one_sse": 0.0,
            "one_cnt": 0,
            "one_type_sse": {"gate": 0.0, "patroller": 0.0, "drifter": 0.0},
            "one_type_cnt": {"gate": 0, "patroller": 0, "drifter": 0},
            "roll_sse": 0.0,
            "roll_cnt": 0,
            "roll_type_sse": {"gate": 0.0, "patroller": 0.0, "drifter": 0.0},
            "roll_type_cnt": {"gate": 0, "patroller": 0, "drifter": 0},
        })

        m["total"] += 1
        m["successes"] += int(row["success"])
        m["steps_sum"] += int(row["steps"])
        m["expansions_sum"] += int(row.get("expansions", 0))

        if row["success"]:
            m["steps_sum_success"] += int(row["steps"])
            m["success_count"] += 1
        else:
            ft = row.get("failure_type", "timeout")
            if ft in m["failures"]:
                m["failures"][ft] += 1
            else:
                m["failures"]["timeout"] += 1

        m["one_sse"] += float(row["one_sse"])
        m["one_cnt"] += int(row["one_cnt"])

        for typ in m["one_type_sse"].keys():
            m["one_type_sse"][typ] += float(row["one_type_sse"].get(typ, 0.0))
            m["one_type_cnt"][typ] += int(row["one_type_cnt"].get(typ, 0))

        m["roll_sse"] += float(row.get("roll_sse", 0.0))
        m["roll_cnt"] += int(row.get("roll_cnt", 0))

        for typ in m["roll_type_sse"].keys():
            m["roll_type_sse"][typ] += float(row.get("roll_type_sse", {}).get(typ, 0.0))
            m["roll_type_cnt"][typ] += int(row.get("roll_type_cnt", {}).get(typ, 0))

    # Finalize derived metrics
    for sname, sdata in suites.items():
        for mname, m in sdata["models"].items():
            total = max(1, m["total"])
            m["success_rate"] = m["successes"] / total
            m["avg_steps_all"] = m["steps_sum"] / total
            m["avg_steps_success"] = (m["steps_sum_success"] / max(1, m["success_count"])) if m["success_count"] > 0 else None
            m["avg_astar_expansions"] = m["expansions_sum"] / total

            m["one_step_mse"] = (m["one_sse"] / max(1, m["one_cnt"])) if m["one_cnt"] > 0 else None
            m["one_step_mse_by_type"] = {
                typ: (m["one_type_sse"][typ] / max(1, m["one_type_cnt"][typ])) if m["one_type_cnt"][typ] > 0 else None
                for typ in m["one_type_sse"].keys()
            }

            m["rollout_mse_k"] = (m["roll_sse"] / max(1, m["roll_cnt"])) if m["roll_cnt"] > 0 else None
            m["rollout_mse_k_by_type"] = {
                typ: (m["roll_type_sse"][typ] / max(1, m["roll_type_cnt"][typ])) if m["roll_type_cnt"][typ] > 0 else None
                for typ in m["roll_type_sse"].keys()
            }

    final = {
        "experiment": "onlstm_vs_hrm_presetm_v2",
        "generated": "" + str(__import__('datetime').datetime.utcnow().isoformat()) + "Z",
        "meta": meta,
        "suites": suites,
    }

    with open(PATHS["results"], "w") as f:
        json.dump(final, f, indent=2)
    vol.commit()
    return final

# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    print("=" * 70)
    print("ON-LSTM vs HRM — Preset M+ (v2): Multi-step + Patch + Metrics + Ablations")
    print("=" * 70)
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print(f"Data episodes: {ENV_CONFIG['data_episodes']:,}")
    print(f"k-rollout (train): {ENV_CONFIG['k_rollout']}")
    print(f"Patch size: {ENV_CONFIG['patch_size']}×{ENV_CONFIG['patch_size']}")
    print("\nEvaluation suites:")
    for s in EVAL_SUITES:
        print(f" - {s['name']}: map_mode={s['map_mode']}, horizon={s['pred_horizon']}, episodes={s.get('eval_episodes', ENV_CONFIG['eval_episodes'])}")
    print()

    # -------------------------------------------------------------------------
    # Step 1: Data
    # -------------------------------------------------------------------------
    print("📦 Step 1: Data Preparation")
    if check_cached_data.remote():
        print("   ✓ Using cached merged dataset")
        merged_path = PATHS["merged_data"]
    else:
        print(f"   Collecting {ENV_CONFIG['data_episodes']:,} episodes...")
        n_workers = 100
        eps_per_worker = ENV_CONFIG['data_episodes'] // n_workers
        chunks = list(collect_data_chunk.map(range(n_workers), kwargs={'n_episodes': eps_per_worker}))
        print(f"   Merging {len(chunks)} chunks...")
        merged_path = merge_chunks.remote(chunks)

    # -------------------------------------------------------------------------
    # Step 2: Training
    # -------------------------------------------------------------------------
    print("\n🏋️ Step 2: Training Models")
    model_status = check_completed_models.remote()
    completed = [k for k, v in model_status.items() if v]
    to_train = [k for k, v in model_status.items() if not v]
    print(f"   Completed: {completed}")
    print(f"   To train: {to_train}")
    print(f"   Checkpointing: every {CHECKPOINT_EVERY} epochs")

    handles = []
    for name, cfg in MODEL_CONFIGS.items():
        if model_status.get(name, False):
            print(f"   ✓ {name} already complete, skipping")
            continue
        print(f"   🚀 Launching {name}...")
        if cfg.model_type == "onlstm":
            handle = train_onlstm_model.spawn(name, merged_path)
        elif name == "hrm_10m":
            handle = train_hrm_10m_ddp.spawn(merged_path)
        else:
            handle = train_hrm_model.spawn(name, merged_path)
        handles.append((name, handle))

    for name, handle in handles:
        print(f"   ⏳ Waiting for {name}...")
        result = handle.get()
        print(f"   ✓ {name} complete: {result['params']:,} params, {result['train_time']/60:.1f} min")

    # -------------------------------------------------------------------------
    # Step 3: Evaluation (ablations)
    # -------------------------------------------------------------------------
    print("\n📊 Step 3: Evaluation (all suites)")
    evaluator = Evaluator()

    tasks: List[Dict[str, Any]] = []
    for suite in EVAL_SUITES:
        n_eps = int(suite.get("eval_episodes", ENV_CONFIG["eval_episodes"]))
        for seed in range(n_eps):
            for model_name in MODEL_CONFIGS.keys():
                tasks.append({"suite": suite, "seed": seed, "model": model_name})

    print(f"   Total eval tasks: {len(tasks):,} (suites × episodes × models)")
    rows = list(evaluator.run_task.map(tasks))

    # -------------------------------------------------------------------------
    # Step 4: Aggregate & Save
    # -------------------------------------------------------------------------
    print("\n📈 Step 4: Aggregation")
    results = aggregate_results.remote(rows)

    print("\nResults saved to:", PATHS["results"])
    print("\nSuites summarized:")
    for sname, sdata in results["suites"].items():
        print(f" - {sname}")
        # print top 3 by success rate
        items = list(sdata["models"].items())
        items.sort(key=lambda kv: -kv[1].get("success_rate", 0.0))
        for mname, md in items[:3]:
            print(f"     {mname}: {md['success_rate']*100:.1f}% (one-step MSE={md['one_step_mse']})")

