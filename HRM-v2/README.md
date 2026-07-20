# HRM-v2: Hierarchical Reasoning Model v2

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8+](https://img.shields.io/badge/PyTorch-2.8+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.8+](https://img.shields.io/badge/CUDA-12.8+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Complete HRM-ACT-v1 implementation** with modern infrastructure, targeting NVIDIA Blackwell (RTX 5090). Fully tested and production-ready!

Training results, the fidelity audit, the repair plan, revalidation, and superseded historical notes are organized under [direct-solver experiment documentation](../docs/experiments/discrete/direct-solver/). The broader repository record starts at the [experiment documentation index](../docs/experiments/README.md).

## 🎯 Key Features

- ✅ **Complete HRM-ACT-v1** - Full model implementation with hierarchical reasoning + ACT
- ✅ **FlashAttention 4** - sm_100 optimization for Blackwell
- ✅ **Sparse embeddings** - Puzzle identifiers with SignSGD optimizer
- ✅ **Unified attention API** - Automatic FA4 → SDPA fallback
- ✅ **Modern Python packaging** - Type hints, comprehensive tests, clean imports
- ✅ **Production ready** - Tested on RTX 5090, CUDA 12.8+, PyTorch 2.8+

## 📋 System Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Native Linux (Ubuntu 22.04+ recommended) |
| **GPU** | NVIDIA RTX 5090 (Blackwell, sm_100) |
| **CUDA** | 12.8+ toolkit |
| **Python** | 3.12+ |
| **Tools** | gcc-12+, cmake 3.26+, ninja |
| **Memory** | 16 GB+ RAM (for building FlashAttention) |
| **Disk** | ~5 GB free space |

## 🚀 Quick Start

### 1. Install `uv` (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and setup
```bash
cd HRM-v2
bash scripts/setup_uv.sh
```

**⏱️ Expected time**: 20-30 minutes (FlashAttention build: ~15 min)

### 3. Verify installation
```bash
source .venv/bin/activate
python scripts/verify_gpu.py
```

**Expected output**:
```
✓ PyTorch version: 2.8.x
✓ CUDA available: True
✓ GPU 0: NVIDIA GeForce RTX 5090
  ✓ Blackwell architecture detected (sm_100)
✓ FlashAttention installed: 4.x.x
✓ Environment is ready for HRM-v2!
```

### 4. Run tests
```bash
pytest tests/              # Unit tests
bash scripts/smoke_train.sh  # End-to-end smoke test
```

## 💡 Basic Usage

### HRM-ACT-v1 (Complete Model)
```python
import torch
from hrm.models import HRMACTv1

# Configuration
config = {
    "batch_size": 8,
    "seq_len": 128,
    "puzzle_emb_ndim": 0,
    "num_puzzle_identifiers": 10,
    "vocab_size": 10000,
    "H_cycles": 3,
    "L_cycles": 2,
    "H_layers": 4,
    "L_layers": 4,
    "hidden_size": 512,
    "num_heads": 8,
    "pos_encodings": "rope",
    "halt_max_steps": 5,
    "halt_exploration_prob": 0.1,
    "forward_dtype": "bfloat16",
}

# Create model
model = HRMACTv1(config).cuda()

# Prepare batch
batch = {
    "inputs": torch.randint(0, 10000, (8, 128), device="cuda"),
    "puzzle_identifiers": torch.zeros(8, dtype=torch.int32, device="cuda"),
}

# Run reasoning
carry = model.initial_carry(batch)
carry, outputs = model(carry, batch)

logits = outputs["logits"]  # Language model predictions
```

### Attention (SDPA + FlashAttention 4)
```python
import torch
from hrm.ops.attention import attention

device = torch.device("cuda")
q = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16, device=device)
k = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16, device=device)
v = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16, device=device)

# Automatically uses FlashAttention 4 if available, else SDPA
out = attention(q, k, v, use_flash=True, is_causal=False)
```

### HRM-Style Components
```python
from hrm.models.layers import HRMTransformerBlock
from hrm.ops.rotary import RotaryEmbedding

rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
block = HRMTransformerBlock(
    hidden_size=512,
    num_heads=8,
    use_flash=True,
).to("cuda").to(torch.bfloat16)

x = torch.randn(2, 128, 512, dtype=torch.bfloat16, device="cuda")
out = block(x, cos_sin=rope())
```

## 📁 Project Structure

```
HRM-v2/
├── src/hrm/                      # Main package
│   ├── ops/                      # Low-level operations
│   │   ├── attention.py         # Unified attention (SDPA + FA4)
│   │   ├── rotary.py            # RoPE implementation
│   │   └── norm.py              # RMS normalization
│   ├── models/                   # Model architectures
│   │   ├── blocks.py            # Standard transformer blocks
│   │   └── layers.py            # HRM-specific layers
│   ├── train/                    # Training utilities (TBD)
│   └── utils/                    # Helper functions
│       ├── env.py               # Environment detection
│       └── init.py              # Weight initialization
├── tests/                        # Comprehensive unit tests
│   ├── test_attention.py        # Attention tests
│   └── test_models.py           # Model tests
├── configs/                      # Configuration files
│   └── minimal_config.yaml      # Example config
├── scripts/                      # Automation scripts
│   ├── setup_uv.sh              # Environment setup
│   ├── verify_gpu.py            # GPU verification
│   └── smoke_train.sh           # Smoke test
├── pyproject.toml                # Python packaging
├── README.md                     # This file
├── QUICKSTART.md                 # Detailed getting started
├── PORTING_GUIDE.md              # Migration from HRM-v1
├── PROJECT_SUMMARY.md            # Comprehensive overview
└── INSTALLATION_CHECKLIST.md     # Step-by-step checklist
```

## 📚 Documentation

- **[HRM_ACT_V1_GUIDE.md](HRM_ACT_V1_GUIDE.md)**: Complete HRM-ACT-v1 usage guide ⭐
- **[QUICKSTART.md](QUICKSTART.md)**: Installation and basic usage
- **[PORTING_GUIDE.md](PORTING_GUIDE.md)**: Migration from original HRM
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**: Technical overview
- **[INSTALLATION_CHECKLIST.md](INSTALLATION_CHECKLIST.md)**: Setup verification

## 🧪 Testing

### Run all tests
```bash
pytest tests/ -v
```

### Test specific components
```bash
pytest tests/test_attention.py -v  # Attention tests
pytest tests/test_models.py -v     # Model tests
```

### Coverage
```bash
pytest tests/ --cov=src/hrm --cov-report=html
```

## 🔧 Development

### Install in editable mode
```bash
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Code quality
```bash
# Linting
ruff check src/ tests/

# Auto-fix
ruff check --fix src/ tests/

# Type checking (if mypy installed)
mypy src/
```

## 🎓 Design Philosophy

HRM-v2 follows **SOLID principles** and emphasizes:
- **Simplicity**: KISS - Keep It Simple
- **Necessity**: YAGNI - You Aren't Gonna Need It  
- **Clarity**: Readable code over clever tricks
- **Testing**: Comprehensive test coverage
- **Documentation**: Clear, thorough documentation

## 🔄 Migration from HRM-v1

All core components have been ported and modernized:

| HRM-v1 | HRM-v2 | Status |
|--------|--------|--------|
| `models/layers.py` | `src/hrm/ops/`, `src/hrm/models/layers.py` | ✅ Complete |
| `models/common.py` | `src/hrm/utils/init.py` | ✅ Complete |
| `models/sparse_embedding.py` | `src/hrm/models/sparse_embedding.py` | ✅ Complete |
| `models/hrm/hrm_act_v1.py` | `src/hrm/models/hrm_act_v1.py` | ✅ Complete |
| FlashAttention 2/3 | FlashAttention 4 (sm_100) | ✅ Updated |
| Direct FA import | Unified attention wrapper | ✅ Modernized |

**The complete HRM-ACT-v1 model is ready to use!** See [HRM_ACT_V1_GUIDE.md](HRM_ACT_V1_GUIDE.md).

## 🤝 Contributing

Contributions welcome! Please:
1. Follow the existing code structure
2. Add type hints and docstrings
3. Include unit tests
4. Update documentation

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Original HRM implementation
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) team
- PyTorch team
- NVIDIA for Blackwell architecture

## 📞 Getting Help

1. Check [QUICKSTART.md](QUICKSTART.md) for common issues
2. Review [INSTALLATION_CHECKLIST.md](INSTALLATION_CHECKLIST.md)
3. Run `python scripts/verify_gpu.py` for diagnostics
4. See [PORTING_GUIDE.md](PORTING_GUIDE.md) for architecture details

---

**Status**: ✅ Core infrastructure complete  
**Target**: NVIDIA RTX 5090 (Blackwell, sm_100)  
**Stack**: CUDA 12.8+, PyTorch 2.8+, FlashAttention 4  
**Created**: October 31, 2025

