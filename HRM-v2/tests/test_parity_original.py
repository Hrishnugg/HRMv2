"""Original-vs-port numerical parity test (F6 — guards everything).

The vendored original at repo-root `models/` is FROZEN ground truth. This test
loads it by file path (it has no `__init__.py`, so `models`/`models.hrm` are
registered as namespace packages), shims `flash_attn` (module-level import in
`models/layers.py`), and asserts the port (`hrm.models.hrm_act_v1.HRMACTv1`)
produces byte-for-byte-parameterized, numerically-matching output on an
identical batch with reset carries.

NEVER modify repo-root `models/` — it is read-only ground truth for this test.
"""

import sys
import types
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]


# --- flash-attn shim (original models/layers.py imports it unconditionally) ---
def _shim_flash():
    def flash_attn_func(q, k, v, causal=False, **kw):  # (B,S,H,D) contract, SDPA-backed
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal
        )
        return o.transpose(1, 2)

    for name in ("flash_attn_interface", "flash_attn"):
        m = types.ModuleType(name)
        m.flash_attn_func = flash_attn_func
        sys.modules.setdefault(name, m)


def _load(name, path, pkg_path=None):
    if pkg_path is not None:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(pkg_path)]
        sys.modules[name] = pkg
        return pkg
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def orig():
    _shim_flash()
    _load("models", None, pkg_path=REPO / "models")
    _load("models.common", REPO / "models" / "common.py")
    _load("models.layers", REPO / "models" / "layers.py")
    _load("models.sparse_embedding", REPO / "models" / "sparse_embedding.py")
    _load("models.hrm", None, pkg_path=REPO / "models" / "hrm")
    return _load("models.hrm.hrm_act_v1", REPO / "models" / "hrm" / "hrm_act_v1.py")


@pytest.fixture(autouse=True)
def _force_port_sdpa(monkeypatch):
    """Force the PORT side onto the SDPA path too, deterministically.

    flash-attn is not installed in this environment, so `hrm.ops.attention.attention`
    already falls through to `sdpa` for float32/CPU tensors before it would even try
    `flash_attention`. We still monkeypatch `flash_attention` to raise ImportError so
    this test stays deterministic if flash-attn is ever installed in this environment
    (per implementer note (a) in the plan) — both sides are then float32 SDPA, so a
    tight tolerance is the correct expectation, not a workaround.

    NOTE: `hrm.ops.__init__` does `from .attention import attention`, which rebinds the
    `attention` attribute on the `hrm.ops` package object, shadowing the submodule
    reference — so `import hrm.ops.attention as m` yields the *function*, not the
    module. Fetch the real module via `sys.modules` instead, which is where
    `ops/attention.py`'s own `attention()` resolves the `flash_attention` name from.
    """
    import hrm.ops.attention  # ensure it's imported/registered
    attn_mod = sys.modules["hrm.ops.attention"]

    def _raise(*a, **k):
        raise ImportError("flash-attn forced unavailable for parity test determinism")

    monkeypatch.setattr(attn_mod, "flash_attention", _raise)


def _cfg(seq_len, num_heads, hidden):
    return dict(
        batch_size=3, seq_len=seq_len, puzzle_emb_ndim=0, num_puzzle_identifiers=1, vocab_size=11,
        H_cycles=2, L_cycles=2, H_layers=1, L_layers=1, hidden_size=hidden, expansion=4.0,
        num_heads=num_heads, pos_encodings="rope", halt_max_steps=4, halt_exploration_prob=0.0,
        forward_dtype="float32",
    )


@pytest.mark.parametrize("seq_len,num_heads,hidden", [(32, 4, 64), (4, 8, 64)])  # 2nd = audit B1 regime (S < H)
def test_inner_forward_parity(orig, seq_len, num_heads, hidden):
    from hrm.models.hrm_act_v1 import HRMACTv1

    torch.manual_seed(0)
    cfg = _cfg(seq_len, num_heads, hidden)
    o = orig.HierarchicalReasoningModel_ACTV1(cfg)
    p = HRMACTv1(cfg)
    missing, unexpected = p.load_state_dict(o.state_dict(), strict=False)
    assert not missing and not unexpected  # identical parameterization
    o.eval(); p.eval()
    batch = {"inputs": torch.randint(0, 11, (3, seq_len)), "puzzle_identifiers": torch.zeros(3, dtype=torch.int32)}
    co = o.inner.reset_carry(torch.ones(3, dtype=torch.bool), o.inner.empty_carry(3))
    cp = p.inner.reset_carry(torch.ones(3, dtype=torch.bool), p.inner.empty_carry(3))
    with torch.no_grad():
        _, lo, (qho, qco) = o.inner(co, batch)
        _, lp, (qhp, qcp) = p.inner(cp, batch)
    assert torch.allclose(lo, lp, atol=1e-5), (lo - lp).abs().max()
    assert torch.allclose(qho, qhp, atol=1e-5) and torch.allclose(qco, qcp, atol=1e-5)


def test_parity_has_teeth(orig):
    """Perturbing one port weight must break parity — guards against vacuous passes."""
    from hrm.models.hrm_act_v1 import HRMACTv1

    torch.manual_seed(0)
    cfg = _cfg(32, 4, 64)
    o = orig.HierarchicalReasoningModel_ACTV1(cfg); p = HRMACTv1(cfg)
    p.load_state_dict(o.state_dict(), strict=False)
    with torch.no_grad():
        p.inner.lm_head.weight[0, 0] += 1.0  # single-element perturbation
    o.eval(); p.eval()
    batch = {"inputs": torch.randint(0, 11, (3, 32)), "puzzle_identifiers": torch.zeros(3, dtype=torch.int32)}
    co = o.inner.reset_carry(torch.ones(3, dtype=torch.bool), o.inner.empty_carry(3))
    cp = p.inner.reset_carry(torch.ones(3, dtype=torch.bool), p.inner.empty_carry(3))
    with torch.no_grad():
        _, lo, _ = o.inner(co, batch); _, lp, _ = p.inner(cp, batch)
    assert not torch.allclose(lo, lp, atol=1e-5)
