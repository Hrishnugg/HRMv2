"""
LSTM vs HRM Comprehensive Comparison Study

Trains and evaluates:
- 4 LSTM variants: 300K, 1M, 3M, 10M parameters
- 3 HRM variants: 302K, 3.5M, 10M parameters

All models are evaluated on Space-Time A* pathfinding in dynamic environments.

Usage:
    modal run hrm-cloud/lstm_hrm_comparison.py
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
from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

app = modal.App("lstm-hrm-comparison-study")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch>=2.4.0", "numpy", "gymnasium", "tqdm")
)

vol = modal.Volume.from_name("lstm-hrm-comparison-vol", create_if_missing=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model variant."""
    name: str
    model_type: str  # "lstm" or "hrm"
    hidden_dim: int
    num_layers: int
    num_heads: int = 4  # Only used for HRM
    batch_size: int = 4096
    lr: float = 1e-3
    epochs: int = 30
    gpu: str = "A10"


# Model configurations
MODEL_CONFIGS = {
    # LSTM variants (4)
    "lstm_300k": ModelConfig("lstm_300k", "lstm", hidden_dim=160, num_layers=2, 
                             batch_size=4096, lr=1e-3, epochs=30, gpu="A10"),
    "lstm_1m": ModelConfig("lstm_1m", "lstm", hidden_dim=290, num_layers=2,
                           batch_size=4096, lr=1e-3, epochs=30, gpu="A10"),
    "lstm_3m": ModelConfig("lstm_3m", "lstm", hidden_dim=500, num_layers=2,
                           batch_size=4096, lr=1e-3, epochs=30, gpu="A10"),
    "lstm_10m": ModelConfig("lstm_10m", "lstm", hidden_dim=900, num_layers=3,
                            batch_size=2048, lr=5e-4, epochs=40, gpu="A10"),
    
    # HRM variants (3)
    "hrm_302k": ModelConfig("hrm_302k", "hrm", hidden_dim=128, num_layers=2, num_heads=4,
                            batch_size=4096, lr=1e-3, epochs=40, gpu="B200"),
    "hrm_3m": ModelConfig("hrm_3m", "hrm", hidden_dim=256, num_layers=2, num_heads=4,
                          batch_size=4096, lr=4e-4, epochs=40, gpu="B200"),
    "hrm_10m": ModelConfig("hrm_10m", "hrm", hidden_dim=384, num_layers=3, num_heads=6,
                           batch_size=2048, lr=5e-4, epochs=40, gpu="B200:4"),  # Multi-GPU DDP
}

# Environment config
ENV_CONFIG = {
    "grid_size": 20,
    "n_static": 12,
    "n_dynamic": 6,
    "obs_history": 20,
    "pred_horizon": 20,
    "data_episodes": 60000,
    "eval_episodes": 100,
}

# Checkpointing config
CHECKPOINT_EVERY = 5  # Save checkpoint every N epochs

PATHS = {
    "data_dir": "/data/comparison_episodes",
    "merged_data": "/data/comparison_merged.pt",
    "models_dir": "/data/comparison_models",
    "checkpoints_dir": "/data/comparison_checkpoints",
    "results": "/data/comparison_results.json",
}

# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class LSTMPredictor(nn.Module):
    """Scalable LSTM for trajectory prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class RMSNorm(nn.Module):
    """RMS Normalization with FP32 upcast for stability."""
    
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
    """SwiGLU activation for FFN."""
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GatedRecurrentBlock(nn.Module):
    """
    Gated Recurrent Transformer Block (GTrXL-style).
    
    Key features:
    - Variance scaling (0.7071) to prevent state explosion
    - Learned gating for selective memory retention
    - FP32 RMSNorm for AMP stability
    """
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, int(dim * 2.6))
        self.gate = nn.Linear(dim * 2, dim)
    
    def forward(self, x, state):
        # Variance scaling - critical for deep recurrence
        h = (x + state) * 0.7071
        
        # Self-attention
        res = h
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm.unsqueeze(1), h_norm.unsqueeze(1), h_norm.unsqueeze(1))
        h = res + attn_out.squeeze(1)
        
        # FFN
        candidate = h + self.ffn(self.norm2(h))
        
        # Gated update - allows selective forgetting
        z = torch.sigmoid(self.gate(torch.cat([candidate, state], dim=-1)))
        return z * candidate + (1 - z) * state


class DeepSapientHRM(nn.Module):
    """
    Hierarchical Reasoning Model with System 1/System 2 architecture.
    
    - L-blocks (System 1): Fast, reactive processing every timestep
    - H-blocks (System 2): Slow, deliberate processing every K timesteps
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 k_step: int = 2, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.k_step = k_step
        self.hidden_dim = hidden_dim
        self.embed = nn.Linear(input_dim, hidden_dim)
        
        # Deep stacks for both systems
        self.L_blocks = nn.ModuleList([
            GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.H_blocks = nn.ModuleList([
            GatedRecurrentBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.head = nn.Linear(hidden_dim, output_dim)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        b, seq, _ = x.size()
        
        # Initialize states for all layers
        h_L = [torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype) 
               for _ in range(len(self.L_blocks))]
        h_H = [torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype) 
               for _ in range(len(self.H_blocks))]
        
        for t in range(seq):
            curr_in = self.embed(x[:, t, :])
            
            # System 2 (H-Module): Slow, deliberate processing
            if t % self.k_step == 0:
                h_in = h_L[-1].detach()  # Gradient detach for stability
                for i, blk in enumerate(self.H_blocks):
                    h_H[i] = blk(h_in, h_H[i])
                    h_in = h_H[i]
            
            # System 1 (L-Module): Fast, reactive processing
            l_in = curr_in + h_H[-1]
            for i, blk in enumerate(self.L_blocks):
                h_L[i] = blk(l_in, h_L[i])
                l_in = h_L[i]
        
        return self.head(h_L[-1])


def create_model(config: ModelConfig) -> nn.Module:
    """Factory function to create model from config."""
    if config.model_type == "lstm":
        return LSTMPredictor(2, config.hidden_dim, 2, config.num_layers)
    else:
        return DeepSapientHRM(2, config.hidden_dim, 2, 
                             k_step=2, 
                             num_heads=config.num_heads,
                             num_layers=config.num_layers)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# CHECKPOINTING HELPERS
# =============================================================================

def save_checkpoint(model: nn.Module, optimizer, scheduler, epoch: int, 
                   loss: float, model_name: str, is_ddp: bool = False):
    """Save training checkpoint to volume."""
    os.makedirs(PATHS['checkpoints_dir'], exist_ok=True)
    checkpoint_path = f"{PATHS['checkpoints_dir']}/{model_name}_checkpoint.pt"
    
    # Handle DDP model (unwrap module)
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
                   device='cuda', is_ddp: bool = False) -> int:
    """
    Load checkpoint if exists.
    
    Returns:
        start_epoch: Epoch to resume from (0 if no checkpoint)
    """
    checkpoint_path = f"{PATHS['checkpoints_dir']}/{model_name}_checkpoint.pt"
    
    if not os.path.exists(checkpoint_path):
        return 0
    
    print(f"   📂 Found checkpoint, resuming training...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle DDP model (load into module)
    if is_ddp:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    best_loss = checkpoint['best_loss']
    
    print(f"   ✓ Resumed from epoch {checkpoint['epoch']}, loss {best_loss:.6f}")
    return start_epoch


def cleanup_checkpoint(model_name: str):
    """Delete checkpoint after successful training completion."""
    checkpoint_path = f"{PATHS['checkpoints_dir']}/{model_name}_checkpoint.pt"
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        vol.commit()
        print(f"   🗑️ Checkpoint cleaned up")


# =============================================================================
# ENVIRONMENT
# =============================================================================

class DynamicGridEnv:
    """Dynamic grid environment with bouncing obstacles."""
    
    def __init__(self, config: Dict):
        self.size = config["grid_size"]
        self.n_dyn = config["n_dynamic"]
        self.reset()
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Static obstacles
        self.static_map = np.zeros((self.size, self.size))
        self.static_map[0, 0] = 0
        self.static_map[self.size - 1, self.size - 1] = 0
        
        for _ in range(self.size):
            r, c = np.random.randint(0, self.size, 2)
            if (r, c) != (0, 0) and (r, c) != (self.size - 1, self.size - 1):
                self.static_map[r, c] = 1.0
        
        # Dynamic obstacles
        self.dynamic_obs = []
        for _ in range(self.n_dyn):
            while True:
                pos = np.random.randint(0, self.size, 2).astype(float)
                if self.static_map[int(pos[0]), int(pos[1])] == 0:
                    vel = np.random.randn(2)
                    norm = np.linalg.norm(vel)
                    if norm < 1e-4:
                        vel = np.array([0.1, 0.1])
                    else:
                        vel = vel / norm * 0.7
                    self.dynamic_obs.append({'pos': pos, 'vel': vel})
                    break
        
        self.agent_pos = np.array([0., 0.])
        self.goal_pos = np.array([self.size - 1., self.size - 1.])
        return self._get_obs()
    
    def step_physics(self):
        """Update obstacle positions with bouncing."""
        for o in self.dynamic_obs:
            o['pos'] += o['vel']
            for i in range(2):
                if o['pos'][i] < 0 or o['pos'][i] >= self.size:
                    o['vel'][i] *= -1
                    o['pos'][i] = np.clip(o['pos'][i], 0, self.size - 0.01)
        return self._get_obs()
    
    def _get_obs(self):
        return np.array([o['pos'] for o in self.dynamic_obs])


# =============================================================================
# SPACE-TIME A* PLANNER
# =============================================================================

class SpaceTimeAStar:
    """A* planner using predicted obstacle trajectories."""
    
    def __init__(self, env: DynamicGridEnv, model: nn.Module, device: torch.device,
                 pred_horizon: int = 20):
        self.env = env
        self.model = model
        self.device = device
        self.pred_horizon = pred_horizon
        self.model.eval()
    
    def get_next_action(self, start, goal, obs_history):
        """Plan next action using predicted obstacle positions."""
        # Prepare input tensor
        curr = torch.tensor(obs_history / self.env.size, dtype=torch.float32).to(self.device)
        curr = curr.to(next(self.model.parameters()).dtype)
        
        # Predict future obstacle positions
        future_obs = []
        with torch.no_grad():
            for _ in range(self.pred_horizon):
                delta = self.model(curr)
                next_pos_norm = curr[:, -1, :] + delta.to(curr.dtype)
                future_obs.append((next_pos_norm.float().cpu().numpy() * self.env.size))
                curr = torch.cat([curr[:, 1:, :], next_pos_norm.unsqueeze(1)], dim=1)
        
        future_obs = np.array(future_obs)
        
        # A* search through space-time
        start_node = (int(start[0]), int(start[1]), 0)
        pq = [(0, 0, start_node)]
        g_score = {start_node: 0}
        came_from = {}
        best_node, min_h = None, float('inf')
        
        while pq:
            f, g, curr_node = heapq.heappop(pq)
            r, c, t = curr_node
            
            # Goal reached
            if (r, c) == (int(goal[0]), int(goal[1])):
                return self._trace(came_from, curr_node, start_node)
            
            # Horizon reached - pick best node
            if t >= self.pred_horizon - 1:
                h = abs(r - goal[0]) + abs(c - goal[1])
                if h < min_h:
                    min_h = h
                    best_node = curr_node
                continue
            
            # Expand neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:
                nr, nc, nt = r + dr, c + dc, t + 1
                
                # Bounds check
                if not (0 <= nr < self.env.size and 0 <= nc < self.env.size):
                    continue
                
                # Static obstacle check
                if self.env.static_map[nr, nc] == 1:
                    continue
                
                # Dynamic obstacle check
                if np.any(np.linalg.norm(future_obs[nt] - np.array([nr, nc]), axis=1) < 1.0):
                    continue
                
                new_g = g + 1
                neigh = (nr, nc, nt)
                
                if new_g < g_score.get(neigh, float('inf')):
                    g_score[neigh] = new_g
                    h = abs(nr - goal[0]) + abs(nc - goal[1])
                    heapq.heappush(pq, (new_g + h, new_g, neigh))
                    came_from[neigh] = curr_node
        
        # Return best node if goal not reached
        if best_node:
            return self._trace(came_from, best_node, start_node)
        
        return (int(start[0]), int(start[1]))
    
    def _trace(self, came_from, curr, start):
        """Trace path back to start and return first step."""
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
    x, y = torch.load(filepath, weights_only=False)
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


@app.function(image=image, volumes={"/data": vol}, cpu=1.0)
def collect_data_chunk(worker_id: int, n_episodes: int) -> str:
    """Collect trajectory data for training."""
    env = DynamicGridEnv(ENV_CONFIG)
    X, Y = [], []
    
    for _ in range(n_episodes):
        env.reset()
        hist = []
        
        for _ in range(70):  # Steps per episode
            hist.append(env.step_physics())
            
            if len(hist) > ENV_CONFIG['obs_history']:
                past = np.array(hist[-ENV_CONFIG['obs_history'] - 1:-1]) / env.size
                future = np.array(hist[-1]) / env.size
                prev = np.array(hist[-2]) / env.size
                
                # Create per-obstacle samples
                for j in range(env.n_dyn):
                    X.append(past[:, j, :])
                    Y.append(future[j, :] - prev[j, :])  # Predict delta
    
    # Save chunk
    os.makedirs(PATHS['data_dir'], exist_ok=True)
    filepath = f"{PATHS['data_dir']}/chunk_{worker_id}.pt"
    torch.save((X, Y), filepath)
    vol.commit()
    
    return filepath


@app.function(image=image, volumes={"/data": vol}, cpu=8.0, memory=65536, timeout=1800)
def merge_chunks(chunk_files: List[str]) -> str:
    """Merge all data chunks into single tensor file."""
    print(f"--> Merging {len(chunk_files)} chunks...")
    
    with ProcessPoolExecutor(8) as exe:
        results = list(exe.map(_load_chunk, chunk_files))
    
    print("--> Concatenating arrays...")
    X = np.concatenate([r[0] for r in results], axis=0)
    Y = np.concatenate([r[1] for r in results], axis=0)
    
    print(f"--> Total samples: {len(X):,}")
    
    # Sanitize - remove any NaN values
    valid = ~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(Y).any(axis=1)
    if not valid.all():
        print(f"⚠️ Dropped {(~valid).sum()} corrupted samples")
        X = X[valid]
        Y = Y[valid]
    
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)
    
    torch.save((X_t, Y_t), PATHS["merged_data"])
    vol.commit()
    
    print(f"✅ Saved {len(X_t):,} samples to {PATHS['merged_data']}")
    return PATHS["merged_data"]


@app.function(image=image, volumes={"/data": vol}, cpu=1.0)
def check_cached_data() -> bool:
    """Check if merged data exists."""
    vol.reload()
    return os.path.exists(PATHS["merged_data"])


@app.function(image=image, volumes={"/data": vol}, cpu=1.0)
def check_completed_models() -> Dict[str, bool]:
    """Check which models have completed training (have final .pt file)."""
    vol.reload()
    status = {}
    for name in MODEL_CONFIGS.keys():
        model_path = f"{PATHS['models_dir']}/{name}.pt"
        status[name] = os.path.exists(model_path)
    return status


# =============================================================================
# MODAL FUNCTIONS - TRAINING
# =============================================================================

@app.function(image=image, gpu="H100", volumes={"/data": vol}, timeout=43200)
def train_lstm_model(model_name: str, merged_path: str):
    """Train an LSTM model on A10 GPU with checkpointing."""
    from tqdm import tqdm
    from torch.amp import autocast, GradScaler
    import time
    
    vol.reload()  # Ensure we see latest checkpoints
    
    config = MODEL_CONFIGS[model_name]
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load data
    print(f"--> Loading data from {merged_path}...")
    X, Y = torch.load(merged_path, weights_only=False)
    print(f"Dataset: {len(X):,} samples")
    
    # Create dataloader
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, Y),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create model
    model = create_model(config).cuda()
    params = count_parameters(model)
    print(f"Model parameters: {params:,}")
    
    # Optimizer and scheduler
    opt = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=config.lr, 
        steps_per_epoch=len(dl), 
        epochs=config.epochs
    )
    scaler = GradScaler('cuda')
    
    # Load checkpoint if exists
    start_epoch = load_checkpoint(model, opt, scheduler, model_name, device='cuda')
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for ep in range(start_epoch, config.epochs):
        ep_loss = 0
        pbar = tqdm(dl, desc=f"Ep {ep+1}/{config.epochs}", leave=False)
        
        for bx, by in pbar:
            bx, by = bx.cuda(non_blocking=True), by.cuda(non_blocking=True)
            
            opt.zero_grad()
            with autocast('cuda'):
                pred = model(bx)
                loss = nn.MSELoss()(pred, by)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            
            ep_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        
        avg_loss = ep_loss / len(dl)
        print(f"Epoch {ep+1}/{config.epochs} | Loss: {avg_loss:.6f}")
        
        # Save checkpoint every N epochs
        if (ep + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, opt, scheduler, ep, avg_loss, model_name)
    
    train_time = time.time() - start_time
    print(f"Training time: {train_time/60:.1f} min")
    
    # Save final model
    os.makedirs(PATHS['models_dir'], exist_ok=True)
    model_path = f"{PATHS['models_dir']}/{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    vol.commit()
    
    # Cleanup checkpoint after successful training
    cleanup_checkpoint(model_name)
    
    print(f"✅ Saved to {model_path}")
    return {"name": model_name, "params": params, "train_time": train_time}


@app.function(image=image, gpu="B200", volumes={"/data": vol}, timeout=36000)
def train_hrm_model(model_name: str, merged_path: str):
    """Train an HRM model on B200 GPU with BF16 and checkpointing."""
    from tqdm import tqdm
    import time
    
    vol.reload()  # Ensure we see latest checkpoints
    
    config = MODEL_CONFIGS[model_name]
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load data
    print(f"--> Loading data from {merged_path}...")
    X, Y = torch.load(merged_path, weights_only=False)
    print(f"Dataset: {len(X):,} samples")
    
    # Create dataloader
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, Y),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create model with BF16
    model = create_model(config).cuda().to(torch.bfloat16)
    params = count_parameters(model)
    print(f"Model parameters: {params:,}")
    
    # Optimizer and scheduler
    opt = optim.AdamW(model.parameters(), lr=config.lr, fused=True)
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=config.lr,
        steps_per_epoch=len(dl),
        epochs=config.epochs
    )
    
    # Load checkpoint if exists
    start_epoch = load_checkpoint(model, opt, scheduler, model_name, device='cuda')
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for ep in range(start_epoch, config.epochs):
        ep_loss = 0
        valid_batches = 0
        pbar = tqdm(dl, desc=f"Ep {ep+1}/{config.epochs}", leave=False)
        
        for bx, by in pbar:
            bx = bx.cuda(non_blocking=True).to(torch.bfloat16)
            by = by.cuda(non_blocking=True).to(torch.bfloat16)
            
            opt.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(bx)
                loss = nn.MSELoss()(pred, by)
            
            # Skip NaN batches
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            
            ep_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        
        avg_loss = ep_loss / max(valid_batches, 1)
        print(f"Epoch {ep+1}/{config.epochs} | Loss: {avg_loss:.6f}")
        
        # Save checkpoint every N epochs
        if (ep + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, opt, scheduler, ep, avg_loss, model_name)
    
    train_time = time.time() - start_time
    print(f"Training time: {train_time/60:.1f} min")
    
    # Save final model
    os.makedirs(PATHS['models_dir'], exist_ok=True)
    model_path = f"{PATHS['models_dir']}/{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    vol.commit()
    
    # Cleanup checkpoint after successful training
    cleanup_checkpoint(model_name)
    
    print(f"✅ Saved to {model_path}")
    return {"name": model_name, "params": params, "train_time": train_time}


# =============================================================================
# DDP TRAINING FOR LARGE HRM (hrm_10m)
# =============================================================================

def ddp_setup(rank: int, world_size: int):
    """Initialize distributed training."""
    import torch.distributed as dist
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def ddp_cleanup():
    """Cleanup distributed training."""
    import torch.distributed as dist
    dist.destroy_process_group()


def ddp_train_worker(rank: int, world_size: int, merged_path: str, config_dict: dict):
    """DDP training worker for each GPU with checkpointing."""
    from tqdm import tqdm
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    import time
    
    ddp_setup(rank, world_size)
    
    # Recreate config from dict
    config = ModelConfig(**config_dict)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Training {config.name.upper()} with {world_size}x GPU DDP")
        print(f"{'='*60}")
    
    # Load data
    if rank == 0:
        print(f"--> Loading data...")
    X, Y = torch.load(merged_path, weights_only=False, map_location='cpu')
    
    dataset = torch.utils.data.TensorDataset(X, Y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # Per-GPU batch size (global batch = per_gpu * world_size)
    per_gpu_batch = config.batch_size
    
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_gpu_batch,
        shuffle=False,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create model with BF16
    model = DeepSapientHRM(
        2, config.hidden_dim, 2,
        k_step=2,
        num_heads=config.num_heads,
        num_layers=config.num_layers
    ).to(rank).to(torch.bfloat16)
    
    model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {params:,}")
        print(f"Global batch size: {per_gpu_batch * world_size}")
        print(f"Batches per epoch: {len(dl)}")
    
    # Optimizer with sqrt LR scaling for multi-GPU
    scaled_lr = config.lr * (world_size ** 0.5)  # sqrt scaling for RNNs
    opt = optim.AdamW(model.parameters(), lr=scaled_lr, fused=True)
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=scaled_lr,
        steps_per_epoch=len(dl),
        epochs=config.epochs
    )
    
    # Load checkpoint if exists (rank 0 loads, then broadcast start_epoch)
    start_epoch_tensor = torch.zeros(1, dtype=torch.int64, device=rank)
    if rank == 0:
        start_epoch = load_checkpoint(model, opt, scheduler, config.name, 
                                      device=rank, is_ddp=True)
        start_epoch_tensor[0] = start_epoch
    
    # Broadcast start_epoch to all ranks
    dist.broadcast(start_epoch_tensor, src=0)
    start_epoch = int(start_epoch_tensor.item())
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for ep in range(start_epoch, config.epochs):
        sampler.set_epoch(ep)  # Important for shuffling
        ep_loss = torch.zeros(1).to(rank)
        valid_batches = 0
        
        pbar = tqdm(dl, desc=f"Ep {ep+1}/{config.epochs}", disable=(rank != 0), leave=False)
        
        for bx, by in pbar:
            bx = bx.to(rank, non_blocking=True).to(torch.bfloat16)
            by = by.to(rank, non_blocking=True).to(torch.bfloat16)
            
            opt.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(bx)
                loss = nn.MSELoss()(pred, by)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            
            ep_loss += loss
            valid_batches += 1
            
            if rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.6f}")
        
        # Aggregate loss across GPUs
        dist.all_reduce(ep_loss, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            avg_loss = ep_loss.item() / (world_size * max(valid_batches, 1))
            print(f"Epoch {ep+1}/{config.epochs} | Loss: {avg_loss:.6f}")
            
            # Save checkpoint every N epochs (rank 0 only)
            if (ep + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(model, opt, scheduler, ep, avg_loss, config.name, is_ddp=True)
        
        # Sync all ranks after checkpoint
        dist.barrier()
    
    train_time = time.time() - start_time
    
    # Save final model (rank 0 only)
    if rank == 0:
        print(f"Training time: {train_time/60:.1f} min")
        os.makedirs(PATHS['models_dir'], exist_ok=True)
        model_path = f"{PATHS['models_dir']}/{config.name}.pt"
        torch.save(model.module.state_dict(), model_path)
        print(f"✅ Saved to {model_path}")
        
        # Cleanup checkpoint after successful training
        cleanup_checkpoint(config.name)
    
    ddp_cleanup()
    return train_time


@app.function(
    image=image,
    gpu="B200:4",  # 4x B200 GPUs
    volumes={"/data": vol},
    timeout=36000,
    memory=65536
)
def train_hrm_10m_ddp(merged_path: str):
    """Train hrm_10m with 4-GPU DDP and checkpointing."""
    import torch.multiprocessing as mp
    import time
    
    vol.reload()  # Ensure we see latest checkpoints
    
    config = MODEL_CONFIGS["hrm_10m"]
    world_size = torch.cuda.device_count()
    
    print(f"🚀 Launching {world_size}-GPU DDP Training for hrm_10m")
    
    # Convert config to dict for multiprocessing
    config_dict = {
        "name": config.name,
        "model_type": config.model_type,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "epochs": config.epochs,
        "gpu": config.gpu,
    }
    
    start_time = time.time()
    
    # Spawn DDP workers
    mp.spawn(
        ddp_train_worker,
        args=(world_size, merged_path, config_dict),
        nprocs=world_size,
        join=True
    )
    
    train_time = time.time() - start_time
    vol.commit()
    
    # Count params for return
    model = DeepSapientHRM(2, config.hidden_dim, 2, k_step=2, 
                          num_heads=config.num_heads, num_layers=config.num_layers)
    params = count_parameters(model)
    
    print(f"✅ hrm_10m DDP training complete")
    return {"name": "hrm_10m", "params": params, "train_time": train_time}


# =============================================================================
# MODAL FUNCTIONS - EVALUATION
# =============================================================================

@app.cls(image=image, gpu="A10", volumes={"/data": vol}, max_containers=20)
class Evaluator:
    """Evaluator class that loads all models and runs episodes."""
    
    @modal.enter()
    def setup(self):
        """Load all trained models."""
        self.device = torch.device("cuda")
        self.models = {}
        
        for name, config in MODEL_CONFIGS.items():
            model_path = f"{PATHS['models_dir']}/{name}.pt"
            if os.path.exists(model_path):
                model = create_model(config).to(self.device)
                
                # Load weights
                state_dict = torch.load(model_path, weights_only=True)
                model.load_state_dict(state_dict)
                
                # HRM models use BF16
                if config.model_type == "hrm":
                    model = model.to(torch.bfloat16)
                
                model.eval()
                self.models[name] = model
                print(f"✓ Loaded {name}")
            else:
                print(f"✗ Missing {name}")
    
    @modal.method()
    def run_episode(self, seed: int) -> Dict[str, int]:
        """Run one evaluation episode for all models."""
        results = {}
        
        for name, model in self.models.items():
            env = DynamicGridEnv(ENV_CONFIG)
            env.reset(seed=seed + 1000)  # Offset seed for eval
            
            planner = SpaceTimeAStar(env, model, self.device, ENV_CONFIG['pred_horizon'])
            
            # Build initial history
            hist = [env.step_physics() for _ in range(ENV_CONFIG['obs_history'])]
            
            # Run episode
            success = 0
            for step in range(80):  # Max steps
                # Get observation history
                h_np = np.array(hist[-ENV_CONFIG['obs_history']:]).transpose(1, 0, 2)
                
                # Plan next action
                nr, nc = planner.get_next_action(env.agent_pos, env.goal_pos, h_np)
                env.agent_pos = np.array([float(nr), float(nc)])
                
                # Step physics
                obs = env.step_physics()
                hist.append(obs)
                
                # Check goal reached
                if np.linalg.norm(env.agent_pos - env.goal_pos) < 0.5:
                    success = 1
                    break
                
                # Check static collision
                if env.static_map[int(nr), int(nc)] == 1:
                    break
                
                # Check dynamic collision
                if np.any(np.linalg.norm(obs - env.agent_pos, axis=1) < 0.8):
                    break
            
            results[name] = success
        
        return results


@app.function(image=image, volumes={"/data": vol}, cpu=1.0, timeout=300)
def aggregate_results(all_results: List[Dict[str, int]]) -> Dict[str, Any]:
    """Aggregate evaluation results and save."""
    
    # Count successes per model
    model_successes = {name: 0 for name in MODEL_CONFIGS.keys()}
    
    for result in all_results:
        for name, success in result.items():
            model_successes[name] += success
    
    total_episodes = len(all_results)
    
    # Calculate success rates
    results = {}
    for name, config in MODEL_CONFIGS.items():
        successes = model_successes[name]
        results[name] = {
            "type": config.model_type,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "successes": successes,
            "total": total_episodes,
            "success_rate": successes / total_episodes if total_episodes > 0 else 0,
        }
    
    # Save results
    with open(PATHS['results'], 'w') as f:
        json.dump(results, f, indent=2)
    vol.commit()
    
    return results


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    """Main entrypoint - orchestrates data collection, training, and evaluation."""
    
    print("=" * 70)
    print("LSTM vs HRM COMPREHENSIVE COMPARISON STUDY")
    print("=" * 70)
    print(f"Models to train: {list(MODEL_CONFIGS.keys())}")
    print(f"Evaluation episodes: {ENV_CONFIG['eval_episodes']}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Data Collection / Loading
    # -------------------------------------------------------------------------
    print("📦 Step 1: Data Preparation")
    
    if check_cached_data.remote():
        print("   ✓ Using cached data")
        merged_path = PATHS["merged_data"]
    else:
        print(f"   Collecting {ENV_CONFIG['data_episodes']:,} episodes...")
        
        # Parallel data collection (100 workers × 600 episodes each)
        n_workers = 100
        eps_per_worker = ENV_CONFIG['data_episodes'] // n_workers
        
        chunks = list(collect_data_chunk.map(
            range(n_workers), 
            kwargs={'n_episodes': eps_per_worker}
        ))
        
        print(f"   Merging {len(chunks)} chunks...")
        merged_path = merge_chunks.remote(chunks)
    
    # -------------------------------------------------------------------------
    # Step 2: Training (skip completed models, resume from checkpoint if needed)
    # -------------------------------------------------------------------------
    print("\n🏋️ Step 2: Training Models")
    
    # Check which models are already complete
    model_status = check_completed_models.remote()
    completed = [k for k, v in model_status.items() if v]
    to_train = [k for k, v in model_status.items() if not v]
    
    print(f"   Completed: {completed}")
    print(f"   To train: {to_train}")
    print(f"   Checkpointing: every {CHECKPOINT_EVERY} epochs")
    
    # Launch training jobs only for incomplete models
    handles = []
    
    for name, config in MODEL_CONFIGS.items():
        if model_status.get(name, False):
            print(f"   ✓ {name} already complete, skipping")
            continue
        
        print(f"   🚀 Launching {name}...")
        
        if config.model_type == "lstm":
            handle = train_lstm_model.spawn(name, merged_path)
        elif name == "hrm_10m":
            # Use DDP for hrm_10m (multi-GPU)
            handle = train_hrm_10m_ddp.spawn(merged_path)
        else:
            handle = train_hrm_model.spawn(name, merged_path)
        
        handles.append((name, handle))
    
    # Wait for all training to complete
    for name, handle in handles:
        print(f"   ⏳ Waiting for {name}...")
        result = handle.get()
        print(f"   ✓ {name} complete: {result['params']:,} params, {result['train_time']/60:.1f} min")
    
    # -------------------------------------------------------------------------
    # Step 3: Evaluation
    # -------------------------------------------------------------------------
    print(f"\n📊 Step 3: Evaluation ({ENV_CONFIG['eval_episodes']} episodes)")
    
    evaluator = Evaluator()
    all_results = list(evaluator.run_episode.map(range(ENV_CONFIG['eval_episodes'])))
    
    # -------------------------------------------------------------------------
    # Step 4: Results
    # -------------------------------------------------------------------------
    print("\n📈 Step 4: Results")
    
    results = aggregate_results.remote(all_results)
    
    # Print results table
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"{'Model':<12} {'Type':<6} {'Params':<12} {'Success Rate':<15}")
    print("-" * 70)
    
    # Sort by type then by success rate
    sorted_results = sorted(results.items(), 
                           key=lambda x: (x[1]['type'], -x[1]['success_rate']))
    
    for name, data in sorted_results:
        config = MODEL_CONFIGS[name]
        params = count_parameters(create_model(config))
        rate = data['success_rate']
        successes = data['successes']
        total = data['total']
        
        print(f"{name:<12} {data['type']:<6} {params:>10,}  {successes}/{total} ({rate*100:.1f}%)")
    
    print("=" * 70)
    print(f"\nResults saved to: {PATHS['results']}")


if __name__ == "__main__":
    # For local testing
    print("Run with: modal run lstm_hrm_comparison.py")
