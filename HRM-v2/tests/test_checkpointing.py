"""Round-trip tests for full-state checkpointing (save_full_checkpoint /
load_full_checkpoint) added to train_maze_optimized.py and train_sudoku.py.

Prior to this, checkpoints only saved model_state_dict + step, so a crash or
migration lost optimizer momentum/variance and RNG state, forcing training to
effectively restart (Adam's exp_avg/exp_avg_sq need to be warm, not zeroed).

This test builds a tiny HRMACTv1 + AdamATan2, runs a few real training-ish
steps, saves a full checkpoint, restores it into a FRESH model+optimizer, and
verifies the fresh copy is bit-for-bit equivalent in the ways that matter for
resuming: params match, global_step matches, optimizer state (exp_avg) is
non-empty and matches, and one more step from either copy (after seeding RNG
identically) produces the same result - i.e. resuming is indistinguishable
from having never stopped.

The streaming carry is intentionally NOT part of the checkpoint - the module
docstring on the checkpoint helpers notes it restarts on resume (see plan).
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from hrm.models import HRMACTv1
from hrm.train import ACTLossHead, IGNORE_LABEL_ID, AdamATan2

_SCRIPT_DIR = Path(__file__).resolve().parent.parent


def _load_train_script(name):
    """Import a train script by file path (they are scripts, not packaged).

    Reuses the stub pattern from test_train_smoke.py: train_maze_optimized.py
    has an unconditional `import wandb` at module scope; wandb is not
    installed in this environment and use_wandb defaults to False (so it's
    never called), so a bare module stub is enough to import the script.
    """
    if name == "train_maze_optimized":
        sys.modules.setdefault("wandb", types.ModuleType("wandb"))
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_model_config():
    return {
        "batch_size": 4,
        "seq_len": 16,
        "vocab_size": 8,
        "num_puzzle_identifiers": 1,
        "hidden_size": 64,
        "num_heads": 4,
        "expansion": 4.0,
        "H_cycles": 1,
        "L_cycles": 1,
        "H_layers": 1,
        "L_layers": 1,
        "halt_max_steps": 2,
        "halt_exploration_prob": 0.1,
        "puzzle_emb_ndim": 0,
        "pos_encodings": "rope",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "forward_dtype": "float32",
    }


def _make_batch(batch_size=4, seq_len=16, vocab_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g, dtype=torch.int32)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g, dtype=torch.int32)
    labels[:, :2] = IGNORE_LABEL_ID
    puzzle_identifiers = torch.zeros((batch_size,), dtype=torch.int32)
    return {"inputs": inputs, "labels": labels, "puzzle_identifiers": puzzle_identifiers}


def _build_model_optimizer(model_config, seed):
    torch.manual_seed(seed)
    model = HRMACTv1(model_config)
    model.train()
    main_params = [p for n, p in model.named_parameters() if not n.startswith("inner.puzzle_emb")]
    optimizer = AdamATan2(main_params, lr=1e-2, betas=(0.9, 0.95), weight_decay=1.0)
    return model, optimizer


def _train_n_steps(model, optimizer, n, start_step=0):
    """Run n training-ish steps (fresh carry each step - carry persistence is
    covered by test_train_smoke.py and is orthogonal to checkpointing)."""
    loss_head = ACTLossHead(model, loss_type="stablemax_cross_entropy")
    global_step = start_step
    for i in range(n):
        batch = _make_batch(seed=global_step)
        carry = loss_head.initial_carry(batch)
        optimizer.zero_grad(set_to_none=False)
        carry, loss, metrics, _, _all_halted = loss_head(return_keys=[], carry=carry, batch=batch)
        (loss / 4).backward()
        optimizer.step()
        global_step += 1
    return global_step


@pytest.fixture(params=["train_maze_optimized", "train_sudoku"])
def train_mod(request):
    return _load_train_script(request.param)


def test_save_full_checkpoint_writes_expected_keys(train_mod, tmp_path):
    model_config = _make_model_config()
    model, optimizer = _build_model_optimizer(model_config, seed=0)
    _train_n_steps(model, optimizer, 3)

    ckpt_path = tmp_path / "ckpt.pt"
    train_mod.save_full_checkpoint(str(ckpt_path), model, optimizer, 3, model_config)

    assert ckpt_path.exists()
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    for key in (
        "model_config", "model_state_dict", "optimizer_state_dict",
        "global_step", "torch_rng_state", "cuda_rng_state", "format_version",
    ):
        assert key in raw, f"missing key {key!r} in saved checkpoint"
    assert raw["global_step"] == 3
    assert raw["format_version"] == 2
    assert raw["model_config"] == model_config


def test_load_full_checkpoint_restores_global_step(train_mod, tmp_path):
    model_config = _make_model_config()
    model, optimizer = _build_model_optimizer(model_config, seed=0)
    global_step = _train_n_steps(model, optimizer, 3)

    ckpt_path = tmp_path / "ckpt.pt"
    train_mod.save_full_checkpoint(str(ckpt_path), model, optimizer, global_step, model_config)

    fresh_model, fresh_optimizer = _build_model_optimizer(model_config, seed=999)
    restored_step = train_mod.load_full_checkpoint(
        str(ckpt_path), fresh_model, fresh_optimizer, device="cpu"
    )

    assert restored_step == 3


def test_load_full_checkpoint_restores_params_bit_for_bit(train_mod, tmp_path):
    model_config = _make_model_config()
    model, optimizer = _build_model_optimizer(model_config, seed=0)
    global_step = _train_n_steps(model, optimizer, 3)

    ckpt_path = tmp_path / "ckpt.pt"
    train_mod.save_full_checkpoint(str(ckpt_path), model, optimizer, global_step, model_config)

    # Fresh model+optimizer with a DIFFERENT seed, so any match after loading
    # is attributable to the checkpoint, not to coincidence.
    fresh_model, fresh_optimizer = _build_model_optimizer(model_config, seed=999)
    train_mod.load_full_checkpoint(str(ckpt_path), fresh_model, fresh_optimizer, device="cpu")

    orig_params = dict(model.named_parameters())
    fresh_params = dict(fresh_model.named_parameters())
    assert orig_params.keys() == fresh_params.keys()
    for name in orig_params:
        assert torch.allclose(orig_params[name].detach(), fresh_params[name].detach()), (
            f"param {name!r} did not round-trip"
        )


def test_load_full_checkpoint_restores_optimizer_state(train_mod, tmp_path):
    model_config = _make_model_config()
    model, optimizer = _build_model_optimizer(model_config, seed=0)
    global_step = _train_n_steps(model, optimizer, 3)

    # Sanity: optimizer state is actually populated (exp_avg/exp_avg_sq/step
    # per param) before we even try to round-trip it.
    assert len(optimizer.state) > 0
    sample_param = next(iter(optimizer.state))
    assert optimizer.state[sample_param]["step"] == 3
    sample_exp_avg_before = optimizer.state[sample_param]["exp_avg"].detach().clone()
    assert sample_exp_avg_before.abs().sum().item() > 0, "exp_avg is all-zero; test would be vacuous"

    ckpt_path = tmp_path / "ckpt.pt"
    train_mod.save_full_checkpoint(str(ckpt_path), model, optimizer, global_step, model_config)

    fresh_model, fresh_optimizer = _build_model_optimizer(model_config, seed=999)
    assert len(fresh_optimizer.state) == 0, "fresh optimizer should start with empty state"

    train_mod.load_full_checkpoint(str(ckpt_path), fresh_model, fresh_optimizer, device="cpu")

    assert len(fresh_optimizer.state) > 0, "optimizer state not restored"

    # Match up state by parameter position/name rather than object identity,
    # since fresh_optimizer's param tensors are distinct objects from optimizer's.
    orig_named = dict(model.named_parameters())
    fresh_named = dict(fresh_model.named_parameters())
    orig_name_by_param = {id(p): n for n, p in orig_named.items()}
    sample_name = orig_name_by_param[id(sample_param)]

    fresh_param = fresh_named[sample_name]
    fresh_state = fresh_optimizer.state[fresh_param]
    assert fresh_state["step"] == 3
    assert torch.allclose(fresh_state["exp_avg"], sample_exp_avg_before)
    assert torch.allclose(
        fresh_state["exp_avg_sq"], optimizer.state[sample_param]["exp_avg_sq"]
    )


def test_resumed_training_step_matches_uninterrupted_step(train_mod, tmp_path):
    """The real point of full-state checkpointing: a 4th step taken on the
    resumed (fresh) copy must match a 4th step taken on the original copy,
    given the same batch and identical RNG seeding. If only model weights
    were restored (the old behavior), Adam's zeroed exp_avg/exp_avg_sq on the
    fresh optimizer would take a differently-shaped step and this would fail.
    """
    model_config = _make_model_config()
    model, optimizer = _build_model_optimizer(model_config, seed=0)
    global_step = _train_n_steps(model, optimizer, 3)

    ckpt_path = tmp_path / "ckpt.pt"
    train_mod.save_full_checkpoint(str(ckpt_path), model, optimizer, global_step, model_config)

    fresh_model, fresh_optimizer = _build_model_optimizer(model_config, seed=999)
    restored_step = train_mod.load_full_checkpoint(
        str(ckpt_path), fresh_model, fresh_optimizer, device="cpu"
    )
    assert restored_step == 3

    # Take one more step on BOTH copies using the identical batch (seeded by
    # the same global_step value), then compare resulting params.
    torch.manual_seed(12345)
    _train_n_steps(model, optimizer, 1, start_step=global_step)

    torch.manual_seed(12345)
    _train_n_steps(fresh_model, fresh_optimizer, 1, start_step=restored_step)

    orig_params = dict(model.named_parameters())
    fresh_params = dict(fresh_model.named_parameters())
    for name in orig_params:
        assert torch.allclose(orig_params[name].detach(), fresh_params[name].detach(), atol=1e-6), (
            f"param {name!r} diverged after the resumed step"
        )


def test_load_full_checkpoint_tolerates_old_weights_only_format(train_mod, tmp_path):
    """format_version==2 checkpoints must remain loadable AND old-style
    weights-only checkpoints (model_state_dict + step, no optimizer/RNG keys)
    must still load without raising - load_full_checkpoint tolerates missing
    keys so pre-existing checkpoints on disk aren't stranded."""
    model_config = _make_model_config()
    model, optimizer = _build_model_optimizer(model_config, seed=0)
    _train_n_steps(model, optimizer, 5)

    old_style_path = tmp_path / "old_style.pt"
    torch.save({
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "step": 5,
    }, str(old_style_path))

    fresh_model, fresh_optimizer = _build_model_optimizer(model_config, seed=999)
    restored_step = train_mod.load_full_checkpoint(
        str(old_style_path), fresh_model, fresh_optimizer, device="cpu"
    )

    assert restored_step == 5
    orig_params = dict(model.named_parameters())
    fresh_params = dict(fresh_model.named_parameters())
    for name in orig_params:
        assert torch.allclose(orig_params[name].detach(), fresh_params[name].detach())
    # No optimizer_state_dict in the old-style file -> fresh_optimizer's state
    # is left alone (empty, since fresh_optimizer never stepped).
    assert len(fresh_optimizer.state) == 0


def test_load_full_checkpoint_works_without_optimizer_argument(train_mod, tmp_path):
    """load_full_checkpoint(path, model, device=...) with no optimizer must
    still restore the model and return the step (e.g. inference-only loads)."""
    model_config = _make_model_config()
    model, optimizer = _build_model_optimizer(model_config, seed=0)
    global_step = _train_n_steps(model, optimizer, 2)

    ckpt_path = tmp_path / "ckpt.pt"
    train_mod.save_full_checkpoint(str(ckpt_path), model, optimizer, global_step, model_config)

    fresh_model, _ = _build_model_optimizer(model_config, seed=999)
    restored_step = train_mod.load_full_checkpoint(str(ckpt_path), fresh_model, device="cpu")

    assert restored_step == 2
    orig_params = dict(model.named_parameters())
    fresh_params = dict(fresh_model.named_parameters())
    for name in orig_params:
        assert torch.allclose(orig_params[name].detach(), fresh_params[name].detach())
