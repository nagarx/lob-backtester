"""
Tests for return-based metrics.

Tests TotalReturn and AnnualReturn with:
- Formula verification against hand calculations
- Edge cases (empty, NaN, Inf)
- Boundary conditions
"""

import numpy as np
import pytest

from lobbacktest.metrics.returns import AnnualReturn, TotalReturn


class TestTotalReturn:
    """Test TotalReturn metric."""

    def test_formula_positive_returns(self):
        """Verify: TotalReturn = prod(1 + r) - 1"""
        returns = np.array([0.01, 0.02, 0.03])

        # Hand calculation: (1.01 * 1.02 * 1.03) - 1 = 0.061106
        expected = np.prod(1 + returns) - 1

        metric = TotalReturn()
        result = metric.compute(returns, {})

        assert abs(result["TotalReturn"] - expected) < 1e-10, \
            f"Expected {expected}, got {result['TotalReturn']}"

    def test_formula_mixed_returns(self):
        """Test with positive and negative returns."""
        returns = np.array([0.05, -0.03, 0.02, -0.01])

        expected = np.prod(1 + returns) - 1

        metric = TotalReturn()
        result = metric.compute(returns, {})

        assert abs(result["TotalReturn"] - expected) < 1e-10

    def test_empty_returns(self):
        """Empty returns should give 0."""
        returns = np.array([])

        metric = TotalReturn()
        result = metric.compute(returns, {})

        assert result["TotalReturn"] == 0.0

    def test_single_return(self):
        """Single return period."""
        returns = np.array([0.05])

        metric = TotalReturn()
        result = metric.compute(returns, {})

        assert abs(result["TotalReturn"] - 0.05) < 1e-10

    def test_zero_returns(self):
        """All zero returns."""
        returns = np.zeros(10)

        metric = TotalReturn()
        result = metric.compute(returns, {})

        assert result["TotalReturn"] == 0.0

    def test_large_loss(self):
        """Large loss that doesn't exceed -100%."""
        returns = np.array([-0.5, -0.5])  # -75% total

        expected = np.prod(1 + returns) - 1  # = 0.25 - 1 = -0.75

        metric = TotalReturn()
        result = metric.compute(returns, {})

        assert abs(result["TotalReturn"] - expected) < 1e-10

    def test_custom_name(self):
        """Test custom metric name."""
        metric = TotalReturn(name="CumulativeReturn")
        result = metric.compute(np.array([0.01]), {})

        assert "CumulativeReturn" in result


class TestAnnualReturn:
    """Test AnnualReturn metric."""

    def test_formula_one_year(self):
        """One year of data should give same as total return."""
        # 252 daily returns
        returns = np.ones(252) * 0.001  # 0.1% daily

        total_return = np.prod(1 + returns) - 1

        metric = AnnualReturn(trading_days_per_year=252, periods_per_day=1)
        result = metric.compute(returns, {})

        # With one year of data, annualized = total
        assert abs(result["AnnualReturn"] - total_return) < 1e-6

    def test_formula_half_year(self):
        """Half year should extrapolate correctly."""
        # 126 daily returns
        returns = np.ones(126) * 0.001

        total_return = np.prod(1 + returns) - 1
        # Annualize: (1 + TR)^(252/126) - 1 = (1 + TR)^2 - 1
        expected = (1 + total_return) ** 2 - 1

        metric = AnnualReturn(trading_days_per_year=252, periods_per_day=1)
        result = metric.compute(returns, {})

        assert abs(result["AnnualReturn"] - expected) < 1e-6

    def test_empty_returns(self):
        """Empty returns should give 0."""
        returns = np.array([])

        metric = AnnualReturn()
        result = metric.compute(returns, {})

        assert result["AnnualReturn"] == 0.0

    def test_total_loss(self):
        """Total loss (-100%) should return -1."""
        returns = np.array([-1.0])  # -100% in one period

        metric = AnnualReturn()
        result = metric.compute(returns, {})

        assert result["AnnualReturn"] == -1.0

    def test_near_total_loss(self):
        """Near-total loss should not overflow."""
        returns = np.array([-0.99])  # -99% in one period

        metric = AnnualReturn()
        result = metric.compute(returns, {})

        # Should return finite value, not NaN or Inf
        assert np.isfinite(result["AnnualReturn"])
        # Should be negative (still a loss when annualized)
        assert result["AnnualReturn"] < 0

    def test_uses_context_total_return(self):
        """Should use TotalReturn from context if available."""
        returns = np.array([0.01, 0.02])

        # Pre-computed total return
        context = {"TotalReturn": 0.05}

        metric = AnnualReturn(trading_days_per_year=252, periods_per_day=1)
        result = metric.compute(returns, context)

        # Should use context value, not recompute
        # Annual = (1 + 0.05)^(252/2) - 1
        expected = (1.05) ** 126 - 1

        assert abs(result["AnnualReturn"] - expected) < 0.01  # Large due to compounding

    def test_custom_annualization_factors(self):
        """Test with custom trading days and periods."""
        returns = np.ones(100) * 0.001

        # 252 trading days, 100 periods per day
        metric = AnnualReturn(trading_days_per_year=252, periods_per_day=100)
        result = metric.compute(returns, {})

        # Should compute correctly
        assert np.isfinite(result["AnnualReturn"])

    def test_custom_name(self):
        """Test custom metric name."""
        metric = AnnualReturn(name="CAGR")
        result = metric.compute(np.array([0.01]), {})

        assert "CAGR" in result


class TestReturnMetricsEdgeCases:
    """Edge case tests for return metrics."""

    def test_nan_in_returns(self):
        """NaN in returns should be handled gracefully (return 0)."""
        returns = np.array([0.01, np.nan, 0.02])

        metric = TotalReturn()
        result = metric.compute(returns, {})

        # validate_returns rejects arrays with NaN, returns 0
        assert result["TotalReturn"] == 0.0

    def test_inf_in_returns(self):
        """Inf in returns should be handled gracefully (return 0)."""
        returns = np.array([0.01, np.inf, 0.02])

        metric = TotalReturn()
        result = metric.compute(returns, {})

        # validate_returns rejects arrays with Inf, returns 0
        assert result["TotalReturn"] == 0.0

    def test_very_small_returns(self):
        """Very small returns should not underflow."""
        returns = np.ones(1000) * 1e-10

        metric = TotalReturn()
        result = metric.compute(returns, {})

        # Should be very close to sum of returns for small values
        assert np.isfinite(result["TotalReturn"])
        assert result["TotalReturn"] > 0

    def test_alternating_up_down(self):
        """Alternating gains and losses of same magnitude lose money."""
        # +10% then -10% = net loss
        returns = np.array([0.1, -0.1])

        metric = TotalReturn()
        result = metric.compute(returns, {})

        # 1.1 * 0.9 = 0.99, so -1%
        expected = 1.1 * 0.9 - 1
        assert abs(result["TotalReturn"] - expected) < 1e-10

