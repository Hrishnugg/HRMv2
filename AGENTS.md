# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

HRM (Hierarchical Reasoning Model) is a pure ML/AI research project — no web servers, databases, or microservices. It has two sub-projects:

- **Root project** (original HRM): requires CUDA GPU + FlashAttention. Cannot import `models/layers.py` on CPU due to hard FlashAttention dependency.
- **HRM-v2** (`HRM-v2/`): modernized rebuild with CPU fallback via PyTorch SDPA. This is the primary development target for CPU-only environments.

### Running without a GPU

The Cloud VM has no NVIDIA GPU. HRM-v2 works fully on CPU (tests, model forward passes). The root project's `models/layers.py` fails to import without FlashAttention/CUDA — this is by design, not a bug.

The `adam-atan2` pip package in `requirements.txt` requires CUDA to build. Skip it on CPU-only setups; it is only needed for training.

### Key commands (HRM-v2)

All commands run from `/workspace/HRM-v2`:

- **Lint**: `ruff check src/ tests/` (pre-existing style issues exist; the codebase compiles and tests pass)
- **Tests**: `pytest tests/ -v` (GPU-specific tests auto-skip on CPU)
- **Install dev**: `pip install -e ".[dev]"`

### Notes

- PyTorch is installed as CPU-only (`--index-url https://download.pytorch.org/whl/cpu`). Do not attempt to install CUDA PyTorch.
- `$HOME/.local/bin` must be on PATH for `ruff`, `pytest`, `torchrun`, etc.
- Root project dependencies (minus `adam-atan2`) install via `pip install -r requirements.txt` after PyTorch is present.
