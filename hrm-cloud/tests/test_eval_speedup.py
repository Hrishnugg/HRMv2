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
