"""Phase 1/2 correctness: EVAL_DIAG=0 must not change A* outcomes, and must
skip the diagnostics-only cost DP. Static suite => fully deterministic."""
import importlib
import numpy as np
import pytest

import residual_tasklora_v2 as R


def _static_suite():
    # ID_A32_static: no gates/patrollers/drifters => deterministic, fast.
    for s in R.build_eval_suites(include_stretch=False, eval_episodes=10):
        if s.suite_id == "ID_A32_static":
            return s
    raise AssertionError("ID_A32_static not found")


def _run(seed, budget):
    return R.run_policy_episode(_static_suite(), seed=seed, model=None,
                                alpha=1.0, max_expansions=budget, device="cpu")


def test_diag_off_matches_diag_on_baseline():
    suite = _static_suite()
    for seed in range(4):
        R.EVAL_DIAG = True
        on = _run(seed, 200)
        R.EVAL_DIAG = False
        off = _run(seed, 200)
        assert (on["success"], on["steps"], on["expansions"]) == \
               (off["success"], off["steps"], off["expansions"]), \
               f"seed={seed} diverged on/off: {on} vs {off}"
    R.EVAL_DIAG = True  # restore default


def test_diag_off_skips_cost_dp(monkeypatch):
    # When diag is off, compute_true_cost_to_goal must not be called at all.
    def boom(*a, **k):
        raise AssertionError("compute_true_cost_to_goal called with EVAL_DIAG=0")
    monkeypatch.setattr(R, "compute_true_cost_to_goal", boom)
    R.EVAL_DIAG = False
    try:
        res = _run(seed=0, budget=200)
        assert "steps" in res
    finally:
        R.EVAL_DIAG = True


class _DummyDeterministicModel:
    """Returns a deterministic delta per (x,y,t_rel)-ish input; no torch params.
    Mimics the minimal interface run_policy_episode uses: encode_obs_sequence +
    predict_delta_from_ctx. No predict_components_from_ctx (exercises simple path)."""
    arm = "avgbase"

    def encode_obs_sequence(self, obs_seq):
        import torch
        return torch.zeros((obs_seq.shape[0], 8), dtype=torch.float32)

    def predict_delta_from_ctx(self, ctx, node_patch, node_meta):
        import torch
        # delta = sum of meta channels => deterministic function of node meta only.
        return node_meta.sum(dim=-1)


def test_cache_path_matches_uncached_for_dummy_model():
    suite = _static_suite()
    m = _DummyDeterministicModel()
    for seed in range(3):
        R.EVAL_DIAG = True   # uncached path
        on = R.run_policy_episode(suite, seed=seed, model=m, alpha=1.0, max_expansions=300, device="cpu")
        R.EVAL_DIAG = False  # cached path
        off = R.run_policy_episode(suite, seed=seed, model=m, alpha=1.0, max_expansions=300, device="cpu")
        assert (on["success"], on["steps"], on["expansions"]) == \
               (off["success"], off["steps"], off["expansions"]), f"seed={seed}: {on} vs {off}"
    R.EVAL_DIAG = True


class _DummyComponentsModel:
    """Exercises the predict_components_from_ctx branch (the real production path).

    The returned dict must satisfy:
      - parts["final_delta"]          : shape [1, N] tensor  (read by both paths)
      - parts["base_delta"]           : shape [1, N] tensor  (read by diag-on path, line 4620)
      - parts["correction"]           : shape [1, N] tensor  (read by diag-on path, line 4621)
      - parts["uncorrected_residual"] : shape [1, N] tensor  (read by diag-on path, line 4622)
      - parts["bound_B"]              : scalar float          (read by diag-on path, line 4623)

    All values are finite so _sanitize_residual_parts_for_eval and
    _require_finite_scalar pass without alteration.

    final_delta is a deterministic function of node_meta (sum of meta channels),
    so cache hits on the fast path reproduce the same value as the uncached path.
    """
    arm = "avgbase"

    def encode_obs_sequence(self, obs_seq):
        import torch
        return torch.zeros((obs_seq.shape[0], 8), dtype=torch.float32)

    def predict_components_from_ctx(self, ctx, node_patch, node_meta):
        import torch
        # node_meta: [1, N, NODE_META_DIM] — sum across last dim gives [1, N]
        final_delta = node_meta.sum(dim=-1).float()          # shape [1, N], finite, deterministic
        N = final_delta.shape[1]
        base_delta = torch.ones(1, N, dtype=torch.float32) * 0.5
        correction = torch.zeros(1, N, dtype=torch.float32)
        uncorrected_residual = torch.zeros(1, N, dtype=torch.float32)
        bound_B = 1.0   # finite scalar — passes _require_finite_scalar
        return {
            "final_delta": final_delta,
            "base_delta": base_delta,
            "correction": correction,
            "uncorrected_residual": uncorrected_residual,
            "bound_B": bound_B,
        }


def test_cache_path_matches_uncached_for_components_model():
    """EVAL_DIAG=True (uncached, diagnostics on) and EVAL_DIAG=False (cached fast
    path) must produce identical (success, steps, expansions) when the model
    exposes predict_components_from_ctx — the real production branch."""
    suite = _static_suite()
    m = _DummyComponentsModel()
    for seed in range(3):
        R.EVAL_DIAG = True   # uncached, runs full diagnostics + components path
        on = R.run_policy_episode(suite, seed=seed, model=m, alpha=1.0, max_expansions=300, device="cpu")
        R.EVAL_DIAG = False  # cached fast path, also uses components branch
        off = R.run_policy_episode(suite, seed=seed, model=m, alpha=1.0, max_expansions=300, device="cpu")
        assert (on["success"], on["steps"], on["expansions"]) == \
               (off["success"], off["steps"], off["expansions"]), \
               f"seed={seed} diverged on-vs-off: on={on} off={off}"
    R.EVAL_DIAG = True  # restore
