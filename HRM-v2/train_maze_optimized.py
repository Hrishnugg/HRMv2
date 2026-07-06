"""
Optimized training script for HRM-ACT-v1 on Maze puzzles.

Features:
- Large batch sizes for full GPU utilization
- Multi-worker data loading for CPU efficiency
- W&B integration for experiment tracking
- Live training visualizations
- Gradient accumulation for even larger effective batches
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
from tqdm import tqdm
import wandb

# Add parent directory to path to import original utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.common import PuzzleDatasetMetadata

# Add src to path for HRM-v2 imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
from hrm.models import HRMACTv1, CastedSparseEmbeddingSignSGD_Distributed
from hrm.train import ACTLossHead, IGNORE_LABEL_ID, AdamATan2, warmup_constant_lr


def pad_batch_to_full_size(
    np_batch: Dict[str, np.ndarray],
    batch_size: int,
    pad_id: int,
    blank_identifier_id: int = 0,
) -> Dict[str, np.ndarray]:
    """Pad a short (tail) batch up to `batch_size` rows.

    Mirrors the original loader (repo-root puzzle_dataset.py:104-113):
    `inputs` pad with the dataset's pad token, `labels` with IGNORE_LABEL_ID
    (padded rows contribute zero LM loss; the loss head's loss_counts /
    valid_metrics gating keeps them out of the metrics), and
    `puzzle_identifiers` with the blank identifier. Required because the
    streaming training loop keeps a persistent carry whose batch dimension is
    fixed by the FIRST batch - an unpadded tail batch would crash the
    halted-slot data replacement (torch.where shape mismatch).
    """
    count = np_batch["puzzle_identifiers"].size
    if count >= batch_size:
        return np_batch
    pad_size = batch_size - count
    pad_values = {
        "inputs": pad_id,
        "labels": IGNORE_LABEL_ID,
        "puzzle_identifiers": blank_identifier_id,
    }
    return {
        k: np.pad(v, ((0, pad_size),) + ((0, 0),) * (v.ndim - 1), constant_values=pad_values[k])
        for k, v in np_batch.items()
    }


def save_full_checkpoint(path, model, optimizer, global_step, model_config):
    """Save a full-state checkpoint: model + optimizer (AdamATan2 exp_avg/
    exp_avg_sq/step per param) + RNG state + global_step.

    The puzzle-embedding SignSGD optimizer is stateless (no momentum/variance
    buffers - see CastedSparseEmbeddingSignSGD_Distributed), so there is
    nothing to save for it beyond what's already in model_state_dict.

    The streaming carry (persistent halted-slot state across batches) is
    intentionally NOT part of this checkpoint: on resume it restarts fresh
    (in-flight episodes are re-begun), which is acceptable per the original
    streaming design - see the resume_from wiring in main() below.
    """
    torch.save({
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "format_version": 2,
    }, path)


def load_full_checkpoint(path, model, optimizer=None, device="cuda"):
    """Load a checkpoint saved by save_full_checkpoint, restoring model
    weights, optimizer state (if an optimizer is given and the checkpoint has
    it), and RNG state. Returns the global_step to resume from.

    Tolerates missing keys so pre-existing weights-only checkpoints (format_version
    absent, only model_state_dict + step) remain loadable - only the pieces
    present in the file are restored.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ckpt.get("torch_rng_state") is not None:
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state"):
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng_state"]])
    return int(ckpt.get("global_step", ckpt.get("step", 0)))


class OptimizedPuzzleDataset(IterableDataset):
    """Optimized puzzle dataset with multi-worker support."""
    
    def __init__(self, dataset_path: str, split: str, batch_size: int, epochs: int = 1, seed: int = 42):
        super().__init__()
        self.dataset_path = dataset_path
        self.split = split
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        
        # Load metadata
        with open(os.path.join(dataset_path, split, "dataset.json"), "r") as f:
            self.metadata = PuzzleDatasetMetadata(**json.load(f))
        
        # Get data paths (will be loaded per-worker)
        self.split_dir = os.path.join(dataset_path, split)
        self.data_files = {
            "inputs": os.path.join(self.split_dir, "all__inputs.npy"),
            "labels": os.path.join(self.split_dir, "all__labels.npy"),
            "puzzle_ids": os.path.join(self.split_dir, "all__puzzle_identifiers.npy"),
            "puzzle_indices": os.path.join(self.split_dir, "all__puzzle_indices.npy"),
        }
        
        # Load metadata only (not full data yet)
        self.num_examples = None
        print(f"Initialized {split} dataset:")
        print(f"  Dataset path: {dataset_path}")
        print(f"  Vocab size: {self.metadata.vocab_size}")
    
    def _load_data(self, worker_id: int):
        """Load data in worker process."""
        self.inputs = np.load(self.data_files["inputs"], mmap_mode="r")
        self.labels = np.load(self.data_files["labels"], mmap_mode="r")
        self.puzzle_ids = np.load(self.data_files["puzzle_ids"])
        self.num_examples = len(self.inputs)
    
    def __iter__(self):
        # Load data in worker
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        self._load_data(worker_id)
        
        for epoch in range(self.epochs):
            # Shuffle with per-epoch seed
            rng = np.random.default_rng(self.seed + epoch)
            indices = rng.permutation(self.num_examples)
            
            # Shard across workers
            if worker_info:
                indices = indices[worker_id::worker_info.num_workers]
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]

                # Get batch data (numpy), padding any short tail batch up to
                # batch_size - the streaming loop's persistent carry needs a
                # fixed batch dimension on every yielded batch
                np_batch = {
                    "inputs": self.inputs[batch_indices].astype(np.int32),
                    "labels": self.labels[batch_indices].astype(np.int32),
                    "puzzle_identifiers": self.puzzle_ids[batch_indices].astype(np.int32),
                }
                np_batch = pad_batch_to_full_size(
                    np_batch, self.batch_size,
                    pad_id=self.metadata.pad_id,
                    blank_identifier_id=self.metadata.blank_identifier_id,
                )
                batch = {k: torch.from_numpy(v) for k, v in np_batch.items()}

                # Handle ignore labels (pad rows are already IGNORE_LABEL_ID,
                # which is never a vocab-space id, so this remap skips them)
                if self.metadata.ignore_label_id is not None:
                    batch["labels"] = torch.where(
                        batch["labels"] == self.metadata.ignore_label_id,
                        IGNORE_LABEL_ID,
                        batch["labels"]
                    )

                yield batch


@dataclass
class TrainConfig:
    """Optimized training configuration for RTX 5090."""
    # Data
    data_path: str = "../data/maze-30x30-hard-1k"
    
    # Model - Maze 30x30
    batch_size: int = 32  # Optimized for 30x30 mazes (900 seq_len is large!)
    seq_len: int = 900  # 30x30 maze
    vocab_size: int = 6  # Maze charset: "# SGo" + padding/special tokens
    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4.0
    H_cycles: int = 2
    L_cycles: int = 2
    H_layers: int = 4
    L_layers: int = 4
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    puzzle_emb_ndim: int = 0  # Disabled for maze (only 1 puzzle type)
    
    # Training
    # 1 segment per optimizer step now: 1k examples / bs 32 with 8-worker
    # sharding = 32 steps/epoch (incl. padded tails) -> 1500 epochs = ~48k
    # steps (~3h on the 5090 at ~0.23s/step)
    epochs: int = 1500
    lr: float = 1e-4
    puzzle_emb_lr: float = 1e-2  # official recipe (inert for maze: puzzle_emb_ndim=0)
    weight_decay: float = 1.0
    puzzle_emb_weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 500

    # Data loading (multi-worker)
    num_workers: int = 8  # 8 workers for 32 virtual cores
    prefetch_factor: int = 4  # Prefetch 4 batches per worker

    # Evaluation
    eval_every: int = 2000  # Evaluate every 2000 steps (steps are ~16x faster now)
    eval_batches: int = 50  # Number of eval batches
    
    # W&B
    use_wandb: bool = False  # Disabled for now
    wandb_project: str = "hrm-v2-maze"
    wandb_run_name: Optional[str] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoint
    checkpoint_dir: str = "checkpoints/maze"
    save_every: int = 5000  # Save checkpoint every 5000 steps
    resume_from: str = ""  # Path to a full-state checkpoint to resume from (empty = start fresh)


def evaluate(loss_head, dataloader, device, max_batches: Optional[int] = None):
    """Evaluate model on dataset.

    Fixed-compute eval, matching the original: a fresh (all-halted) carry per
    eval batch, then `halt_max_steps` forward segments through the loss head
    under no_grad (in eval mode the model only halts at max steps, since ACT
    exploration/early-halt behavior in `HRMACTv1.forward` is gated on
    `self.training`). Metrics accumulate only over slots that halted on a
    given segment (the loss head's own `count`-gated metrics), same as the
    original's per-sequence accounting.
    """
    model = loss_head.model
    model.eval()

    all_metrics = {
        "count": 0.0,
        "accuracy": 0.0,
        "exact_accuracy": 0.0,
        "q_halt_accuracy": 0.0,
        "steps": 0.0,
        "lm_loss": 0.0,
    }

    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            # Move batch to device and initialize a fresh carry for this batch
            batch = {k: v.to(device) for k, v in batch.items()}
            carry = loss_head.initial_carry(batch)

            # Run halt_max_steps segments (fixed-compute eval)
            for _ in range(model.config.halt_max_steps):
                carry, _, metrics, _, all_halted = loss_head(return_keys=[], carry=carry, batch=batch)

                count = metrics["count"].item()
                if count > 0:
                    all_metrics["count"] += count
                    all_metrics["accuracy"] += metrics["accuracy"].item()
                    all_metrics["exact_accuracy"] += metrics["exact_accuracy"].item()
                    all_metrics["q_halt_accuracy"] += metrics["q_halt_accuracy"].item()
                    all_metrics["steps"] += metrics["steps"].item()
                all_metrics["lm_loss"] += metrics["lm_loss"].item()

                if bool(all_halted):
                    break

            num_batches += 1

    # Average metrics
    if all_metrics["count"] > 0:
        all_metrics["accuracy"] /= all_metrics["count"]
        all_metrics["exact_accuracy"] /= all_metrics["count"]
        all_metrics["q_halt_accuracy"] /= all_metrics["count"]
        all_metrics["steps"] /= all_metrics["count"]

    if num_batches > 0:
        all_metrics["lm_loss"] /= num_batches

    model.train()
    return all_metrics


def main():
    """Main training loop."""
    config = TrainConfig()
    
    # Initialize W&B
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"maze-bs{config.batch_size}-lr{config.lr}",
            config=vars(config),
        )
    
    print("=" * 60)
    print("HRM-v2 Optimized Training: Maze 30x30")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size} (optimized for RTX 5090)")
    print(f"Data workers: {config.num_workers}")
    print(f"Learning rate: {config.lr}")
    print(f"Epochs: {config.epochs}")
    print(f"W&B tracking: {config.use_wandb}")
    print()
    
    # Create model
    model_config = {
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "vocab_size": config.vocab_size,
        "num_puzzle_identifiers": 1,  # Maze dataset has only 1 puzzle type
        "hidden_size": config.hidden_size,
        "num_heads": config.num_heads,
        "expansion": config.expansion,
        "H_cycles": config.H_cycles,
        "L_cycles": config.L_cycles,
        "H_layers": config.H_layers,
        "L_layers": config.L_layers,
        "halt_max_steps": config.halt_max_steps,
        "halt_exploration_prob": config.halt_exploration_prob,
        "puzzle_emb_ndim": config.puzzle_emb_ndim,
        "pos_encodings": "rope",
        "forward_dtype": "bfloat16",
    }
    
    print("Creating model...")
    model = HRMACTv1(model_config).to(config.device)
    model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Create datasets with multi-worker support
    print("Setting up data loaders...")
    train_dataset = OptimizedPuzzleDataset(
        config.data_path, "train", config.batch_size, epochs=config.epochs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,  # Dataset already returns batches
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True if config.num_workers > 0 else False,
        pin_memory=True,
    )
    
    eval_dataset = OptimizedPuzzleDataset(
        config.data_path, "test", config.batch_size, epochs=1
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=None,
        num_workers=2,  # Fewer workers for eval
        pin_memory=True,
    )
    
    print(f"✅ Train loader: {config.num_workers} workers, prefetch={config.prefetch_factor}")
    print(f"✅ Eval loader: 2 workers")
    print()
    
    # Loss head (restores q_halt_loss + per-sequence divisor + 0.5 weighting - D1/D3/D4)
    loss_head = ACTLossHead(model, loss_type="stablemax_cross_entropy")

    # Setup optimizers
    main_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("inner.puzzle_emb")
    ]

    optimizer = AdamATan2(
        main_params,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )

    # Puzzle embedding optimizer (SignSGD)
    puzzle_emb_optimizer = None
    if hasattr(model.inner, "puzzle_emb") and config.puzzle_emb_ndim > 0:
        puzzle_emb_params = [
            model.inner.puzzle_emb.weights,
            model.inner.puzzle_emb.local_weights,
            model.inner.puzzle_emb.local_ids,
        ]
        puzzle_emb_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
            puzzle_emb_params,
            world_size=1,
            lr=config.puzzle_emb_lr,
            weight_decay=config.puzzle_emb_weight_decay
        )
        print("✅ Sparse embedding optimizer enabled (SignSGD)")

    # Resume from a full-state checkpoint if requested (model + optimizer +
    # RNG state). Note: the streaming carry below is NOT restored - it always
    # starts fresh on resume, so any episodes that were in-flight in halted
    # slots at the time of the last checkpoint are simply re-begun. This is
    # acceptable per the original streaming design (see save_full_checkpoint).
    if config.resume_from:
        global_step = load_full_checkpoint(config.resume_from, model, optimizer, config.device)
        print(f"✅ Resumed from checkpoint: {config.resume_from} (global_step={global_step})")
    else:
        global_step = 0

    # Training loop
    print("Starting training...")
    print()

    running_metrics = {
        "lm_loss": 0.0,
        "q_halt_loss": 0.0,
        "accuracy": 0.0,
        "exact_accuracy": 0.0,
        "steps": 0.0,
    }
    running_count = 0
    carry = None

    try:
        pbar = tqdm(train_loader, desc="Training", dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device (persistent carry streams new samples into
            # halted slots - the carry is created ONCE and never reset here)
            batch = {k: v.to(config.device) for k, v in batch.items()}
            if carry is None:
                carry = loss_head.initial_carry(batch)

            # Learning rate schedule: warmup then constant (D5 - no cosine decay)
            lr_mult = warmup_constant_lr(global_step, config.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr * lr_mult
            if puzzle_emb_optimizer:
                for param_group in puzzle_emb_optimizer.param_groups:
                    param_group["lr"] = config.puzzle_emb_lr * lr_mult

            # set_to_none=False: the sparse-emb SignSGD optimizer must always see
            # a (possibly zero) grad tensor on local_weights, never None.
            optimizer.zero_grad(set_to_none=False)
            if puzzle_emb_optimizer:
                puzzle_emb_optimizer.zero_grad(set_to_none=False)

            # ONE segment per optimizer step (deep supervision - D2/D6): forward
            # one segment through the loss head, backward, step, done. No inner
            # halt_max_steps loop here - halting is the model's own carry state,
            # threaded across iterations of this very loop.
            carry, loss, metrics, _, all_halted = loss_head(return_keys=[], carry=carry, batch=batch)
            (loss / config.batch_size).backward()

            optimizer.step()
            if puzzle_emb_optimizer:
                puzzle_emb_optimizer.step()

            global_step += 1

            # Accumulate metrics (tensors -> .item(); count can be 0 when
            # nothing halted on this segment)
            count = metrics["count"].item()
            if count > 0:
                running_metrics["lm_loss"] += metrics["lm_loss"].item()
                running_metrics["q_halt_loss"] += metrics["q_halt_loss"].item()
                running_metrics["accuracy"] += metrics["accuracy"].item()
                running_metrics["exact_accuracy"] += metrics["exact_accuracy"].item()
                running_metrics["steps"] += metrics["steps"].item()
                running_count += count

            # Update progress bar
            if running_count > 0:
                pbar.set_postfix({
                    "loss": f"{running_metrics['lm_loss'] / running_count:.3f}",
                    "acc": f"{running_metrics['accuracy'] / running_count:.3f}",
                    "exact": f"{running_metrics['exact_accuracy'] / running_count:.3f}",
                })

            # Evaluation and logging
            if (batch_idx + 1) % config.eval_every == 0:
                # Compute training metrics
                train_metrics = {}
                if running_count > 0:
                    for k, v in running_metrics.items():
                        train_metrics[f"train/{k}"] = v / running_count
                    running_metrics = {k: 0.0 for k in running_metrics}
                    running_count = 0

                # Run evaluation
                print("\n" + "=" * 60)
                print(f"Evaluation at step {global_step}")
                print("=" * 60)

                eval_metrics_raw = evaluate(loss_head, eval_loader, config.device, max_batches=config.eval_batches)
                eval_metrics = {f"eval/{k}": v for k, v in eval_metrics_raw.items()}

                # Print metrics
                print("Training metrics:")
                for k, v in train_metrics.items():
                    print(f"  {k}: {v:.4f}")

                print("\nEvaluation metrics:")
                for k, v in eval_metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
                print()

                # Log to W&B
                if config.use_wandb:
                    wandb.log({
                        **train_metrics,
                        **eval_metrics,
                        "step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "puzzle_emb_lr": config.puzzle_emb_lr if puzzle_emb_optimizer else 0,
                    }, step=global_step)

                model.train()

            # Save checkpoint (full-state: model + optimizer + RNG - enables resume_from)
            if (global_step % config.save_every) == 0:
                checkpoint_dir = Path(config.checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
                save_full_checkpoint(checkpoint_path, model, optimizer, global_step, model_config)

                print(f"💾 Checkpoint saved: {checkpoint_path}")

        print("\n✅ Training complete!")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    finally:
        # Save final checkpoint (full-state: model + optimizer + RNG)
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
        save_full_checkpoint(checkpoint_path, model, optimizer, global_step, model_config)

        print(f"\n💾 Final checkpoint saved: {checkpoint_path}")
        
        if config.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()

