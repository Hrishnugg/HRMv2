"""Unit tests for _binding_budget (difficulty-aware binding-budget selection).

Pure unit tests: no GPU, no I/O, no calibration files on disk.
All dicts are passed in directly.
"""
import sys
import os

# Allow importing the orchestrator module without its heavy optional deps.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from continuous_prm_c8_dynamics_compare import _binding_budget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calib_budgets(suite, band):
    """Build a calib_budgets dict with one entry."""
    return {suite: band}


def _calib_meas(suite, euclid_by_budget):
    """Build a calib_measurements dict with one suite's measurement list."""
    return {suite: [{"budget": b, "euclid": v} for b, v in euclid_by_budget.items()]}


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestBindingBudgetNormalSuite:
    """Normal suite: lower edge qualifies — should return the lower edge."""

    def test_lower_edge_qualifies(self):
        suite = "C_dyn_maze"
        cb = _calib_budgets(suite, [1800, 2500])
        cm = _calib_meas(suite, {1800: 0.4, 2500: 0.8})
        result = _binding_budget(suite, {suite: [1800, 2500]}, cb, cm)
        assert result == 1800, f"Expected 1800, got {result}"


class TestBindingBudgetDegenerateLowerEdge:
    """C_dyn_maze_dense pattern: band [150, 3500], euclid 0.0 at 150, 0.1 at 3500.
    Lower edge is degenerate — should skip 150 and return 3500."""

    def test_skips_degenerate_lower_returns_upper(self):
        suite = "C_dyn_maze_dense"
        cb = _calib_budgets(suite, [150, 3500])
        cm = _calib_meas(suite, {150: 0.0, 3500: 0.1})
        result = _binding_budget(suite, {suite: [150, 3500]}, cb, cm)
        assert result == 3500, f"Expected 3500, got {result}"

    def test_skips_degenerate_lower_exact_floor_qualifies(self):
        """Budget at exactly the floor (0.05) should qualify."""
        suite = "C_dyn_maze_dense"
        cb = _calib_budgets(suite, [150, 3500])
        cm = _calib_meas(suite, {150: 0.0, 3500: 0.05})
        result = _binding_budget(suite, {suite: [150, 3500]}, cb, cm)
        assert result == 3500, f"Expected 3500, got {result}"


class TestBindingBudgetAllDegenerate:
    """All budgets in band are below floor — should return max(band)."""

    def test_all_degenerate_returns_max(self):
        suite = "C_dyn_maze_dense"
        cb = _calib_budgets(suite, [150, 250])
        cm = _calib_meas(suite, {150: 0.0, 250: 0.0})
        result = _binding_budget(suite, {suite: [150, 250]}, cb, cm)
        assert result == 250, f"Expected 250, got {result}"


class TestBindingBudgetNoMeasurements:
    """When calib_measurements is None (not passed), old behavior: return min(band)."""

    def test_none_measurements_returns_min_band(self):
        suite = "C_dyn_maze_dense"
        cb = _calib_budgets(suite, [150, 3500])
        # No measurements passed — old behavior
        result = _binding_budget(suite, {suite: [150, 3500]}, cb, calib_measurements=None)
        assert result == 150, f"Expected 150 (old behavior), got {result}"

    def test_no_measurements_arg_defaults_to_none(self):
        """Default signature (no calib_measurements) behaves like old behavior."""
        suite = "C_dyn_maze_dense"
        cb = _calib_budgets(suite, [150, 3500])
        result = _binding_budget(suite, {suite: [150, 3500]}, cb)
        assert result == 150, f"Expected 150 (old behavior), got {result}"


class TestBindingBudgetNoBand:
    """No band in calib_budgets for this suite — falls back to min of suite_budgets."""

    def test_no_band_falls_back_to_suite_budgets(self):
        suite = "C_dyn_maze_dense"
        cb = {}  # No band for this suite
        cm = _calib_meas(suite, {150: 0.0, 3500: 0.1})
        result = _binding_budget(suite, {suite: [150, 3500]}, cb, cm)
        assert result == 150, f"Expected 150 (fallback to sorted suite_budgets[0]), got {result}"

    def test_no_band_and_no_measurements_falls_back(self):
        suite = "C_dyn_crossing"
        result = _binding_budget(suite, {suite: [200, 400, 600]}, {}, None)
        assert result == 200, f"Expected 200, got {result}"


class TestBindingBudgetEdgeCases:
    """Robustness: bad/missing values must not crash."""

    def test_measurement_missing_euclid_key_treated_as_zero(self):
        suite = "C_dyn_rooms"
        cb = _calib_budgets(suite, [1300, 1800])
        # 'euclid' key missing in one measurement
        cm = {suite: [{"budget": 1300}, {"budget": 1800, "euclid": 0.3}]}
        result = _binding_budget(suite, {suite: [1300, 1800]}, cb, cm)
        # 1300 has no euclid -> treated as 0.0 -> doesn't qualify
        # 1800 has 0.3 >= 0.05 -> qualifies
        assert result == 1800, f"Expected 1800, got {result}"

    def test_measurement_none_euclid_treated_as_zero(self):
        suite = "C_dyn_rooms"
        cb = _calib_budgets(suite, [1300, 1800])
        cm = {suite: [{"budget": 1300, "euclid": None}, {"budget": 1800, "euclid": 0.6}]}
        result = _binding_budget(suite, {suite: [1300, 1800]}, cb, cm)
        assert result == 1800, f"Expected 1800, got {result}"

    def test_single_budget_in_band_qualifies(self):
        suite = "C_dyn_spiral"
        cb = _calib_budgets(suite, [2500])
        cm = _calib_meas(suite, {2500: 0.5})
        result = _binding_budget(suite, {suite: [2500]}, cb, cm)
        assert result == 2500, f"Expected 2500, got {result}"

    def test_single_budget_in_band_degenerate(self):
        """Single-element band that doesn't qualify: returns max(band) = itself."""
        suite = "C_dyn_spiral"
        cb = _calib_budgets(suite, [2500])
        cm = _calib_meas(suite, {2500: 0.0})
        result = _binding_budget(suite, {suite: [2500]}, cb, cm)
        assert result == 2500, f"Expected 2500, got {result}"
