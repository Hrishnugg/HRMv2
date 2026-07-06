"""
Training script for HRM-ACT-v1 on Sudoku puzzles.

This script trains the HRM-v2 model on the Sudoku-Extreme dataset.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
from tqdm import tqdm

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


class SimplePuzzleDataset(IterableDataset):
    """Simplified puzzle dataset for single-GPU training."""
    
    def __init__(self, dataset_path: str, split: str, batch_size: int, epochs: int = 1):
        super().__init__()
        self.dataset_path = dataset_path
        self.split = split
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Load metadata
        with open(os.path.join(dataset_path, split, "dataset.json"), "r") as f:
            self.metadata = PuzzleDatasetMetadata(**json.load(f))
        
        # Load data
        split_dir = os.path.join(dataset_path, split)
        self.inputs = np.load(os.path.join(split_dir, "all__inputs.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(split_dir, "all__labels.npy"), mmap_mode="r")
        self.puzzle_ids = np.load(os.path.join(split_dir, "all__puzzle_identifiers.npy"))
        self.puzzle_indices = np.load(os.path.join(split_dir, "all__puzzle_indices.npy"))
        
        print(f"Loaded {split} dataset:")
        print(f"  Total examples: {len(self.inputs)}")
        print(f"  Num puzzles: {self.puzzle_indices.shape[0] - 1}")
        print(f"  Vocab size: {self.metadata.vocab_size}")
    
    def __iter__(self):
        for epoch in range(self.epochs):
            # Shuffle examples
            indices = np.random.permutation(len(self.inputs))
            
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
    """Training configuration."""
    # Data
    data_path: str = "../data/sudoku-extreme-1k-aug-1000"
    
    # Model
    batch_size: int = 16
    seq_len: int = 81  # Sudoku 9x9
    vocab_size: int = 11  # 0-10 (0=blank, 1-9=digits, 10=padding/ignore)
    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4.0
    H_cycles: int = 2
    L_cycles: int = 2
    H_layers: int = 4
    L_layers: int = 4
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    puzzle_emb_ndim: int = 512  # Same as hidden_size
    
    # Training
    epochs: int = 100
    lr: float = 1e-4
    puzzle_emb_lr: float = 1e-2
    weight_decay: float = 1.0
    puzzle_emb_weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 100
    
    # Evaluation
    eval_every: int = 500  # steps are ~16x faster now (1 segment per step)
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoint
    checkpoint_dir: str = "checkpoints"


def evaluate(loss_head, dataset, device, max_steps: Optional[int] = None):
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
        for batch_idx, batch in enumerate(dataset):
            if max_steps and batch_idx >= max_steps:
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
    
    print("=" * 60)
    print("HRM-v2 Training: Sudoku-Extreme")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print(f"Epochs: {config.epochs}")
    print()
    
    # Create model
    model_config = {
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "vocab_size": config.vocab_size,
        "num_puzzle_identifiers": 1000,  # Sudoku dataset has 1000 puzzles
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
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = SimplePuzzleDataset(
        config.data_path, "train", config.batch_size, epochs=config.epochs
    )
    eval_dataset = SimplePuzzleDataset(
        config.data_path, "test", config.batch_size, epochs=1
    )
    
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

    # Puzzle embedding optimizer (using SignSGD for sparse updates)
    puzzle_emb_optimizer = None
    if hasattr(model.inner, "puzzle_emb") and config.puzzle_emb_ndim > 0:
        # The SignSGD optimizer needs: weights (buffer), local_weights (buffer with grad), local_ids (buffer)
        puzzle_emb_params = [
            model.inner.puzzle_emb.weights,
            model.inner.puzzle_emb.local_weights,
            model.inner.puzzle_emb.local_ids,
        ]
        puzzle_emb_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
            puzzle_emb_params,
            world_size=1,  # Single GPU
            lr=config.puzzle_emb_lr,
            weight_decay=config.puzzle_emb_weight_decay
        )

    # Training loop
    print("Starting training...")
    print()

    global_step = 0
    running_metrics = {k: 0.0 for k in ["lm_loss", "q_halt_loss", "accuracy", "exact_accuracy", "steps"]}
    running_count = 0
    carry = None

    try:
        for batch_idx, batch in enumerate(tqdm(train_dataset, desc="Training")):
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

            # Evaluation
            if (batch_idx + 1) % config.eval_every == 0:
                # Print training metrics
                if running_count > 0:
                    print(f"\n[Step {global_step}] Training metrics:")
                    for k, v in running_metrics.items():
                        print(f"  {k}: {v / running_count:.4f}")
                    running_metrics = {k: 0.0 for k in running_metrics}
                    running_count = 0

                # Run evaluation
                print("Running evaluation...")
                eval_metrics = evaluate(loss_head, eval_dataset, config.device, max_steps=20)
                print("Evaluation metrics:")
                for k, v in eval_metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
                print()

                model.train()

        print("\nTraining complete!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final checkpoint
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / "hrm_sudoku_final.pt"
    torch.save({
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "step": global_step,
    }, checkpoint_path)
    
    print(f"\nCheckpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()

