"""Lock the math of the C7 analyze-mode statistics helpers.

Covers wilcoxon_signed_rank_p (vs scipy), bootstrap_median_ci (determinism +
bracketing + empty-set), and the small-n / degenerate guards. Mirrors the
sys.path bootstrap used by the other C7 tests (and test_c6_heatmap_value_field).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "continuous_prm"))

import math

import numpy as np
import pytest

import continuous_prm_c7_integration_compare as A


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank
# ---------------------------------------------------------------------------

def test_wilcoxon_matches_scipy_on_fixed_array():
    """On a fixed nonzero-difference array, our helper matches scipy.stats.wilcoxon
    with method="auto" + the default continuity behaviour (correction=True)."""
    scipy_stats = pytest.importorskip("scipy.stats")
    diffs = [0.12, -0.05, 0.30, 0.18, -0.02, 0.25, 0.09, 0.40]
    got = A.wilcoxon_signed_rank_p(diffs)
    expected = float(
        scipy_stats.wilcoxon(
            diffs, zero_method="wilcox", alternative="two-sided", method="auto"
        ).pvalue
    )
    assert math.isfinite(got)
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_wilcoxon_all_zero_returns_nan_without_raising():
    # All-zero differences -> no nonzero pairs -> undefined p (nan), no exception.
    assert math.isnan(A.wilcoxon_signed_rank_p([0.0, 0.0, 0.0]))
    assert math.isnan(A.wilcoxon_signed_rank_p([]))


def test_wilcoxon_manual_fallback_close_to_scipy(monkeypatch):
    """With scipy import blocked, the manual normal-approx fallback still returns
    a finite p in the same ballpark as scipy (sanity, not exact)."""
    scipy_stats = pytest.importorskip("scipy.stats")
    diffs = [0.12, -0.05, 0.30, 0.18, -0.02, 0.25, 0.09, 0.40, 0.15, -0.03]
    p_scipy = float(
        scipy_stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="auto").pvalue
    )

    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    p_manual = A.wilcoxon_signed_rank_p(diffs)
    assert math.isfinite(p_manual)
    # normal approx on small n: same order of magnitude / within a generous band.
    assert abs(p_manual - p_scipy) < 0.1


# ---------------------------------------------------------------------------
# Bootstrap median CI
# ---------------------------------------------------------------------------

def test_bootstrap_is_deterministic_for_same_seed():
    vals = [0.8, 0.9, 1.0, 1.1, 1.2, 0.95, 1.05, 0.85]
    a = A.bootstrap_median_ci(vals, seed=1234)
    b = A.bootstrap_median_ci(vals, seed=1234)
    assert a == b
    # different seed should (essentially always) differ on at least one endpoint.
    c = A.bootstrap_median_ci(vals, seed=9999)
    assert (a[1], a[2]) != (c[1], c[2])


def test_bootstrap_ci_brackets_point_median():
    vals = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
    med, lo, hi = A.bootstrap_median_ci(vals, seed=1234)
    assert med == pytest.approx(float(np.median(vals)))
    assert lo <= med <= hi


def test_bootstrap_empty_returns_nan_triple_without_raising():
    med, lo, hi = A.bootstrap_median_ci([], seed=1)
    assert math.isnan(med) and math.isnan(lo) and math.isnan(hi)
    # all-nonfinite input is treated as empty too.
    med2, lo2, hi2 = A.bootstrap_median_ci([float("nan"), float("inf")], seed=1)
    assert math.isnan(med2) and math.isnan(lo2) and math.isnan(hi2)


def test_bootstrap_single_value_collapses_to_point():
    med, lo, hi = A.bootstrap_median_ci([1.5], seed=42)
    assert (med, lo, hi) == (1.5, 1.5, 1.5)


# ---------------------------------------------------------------------------
# p-value formatting + small-n guard
# ---------------------------------------------------------------------------

def test_fmt_p_small_n_guard():
    assert A._fmt_p(0.01, n=2) == f"n/a (n<{A.MIN_N_FOR_P})"
    # n at/above threshold formats normally.
    assert A._fmt_p(0.04, n=A.MIN_N_FOR_P) == "0.040"


def test_fmt_p_tiny_value_and_nan():
    assert A._fmt_p(0.0005) == "<0.001"
    assert A._fmt_p(0.0) == "0.000"  # exactly zero is not in (0, 0.001)
    assert A._fmt_p(float("nan")) == "n/a"
    assert A._fmt_p(None) == "n/a"
