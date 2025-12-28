"""
Tests for risk metrics.

Tests follow RULE.md testing philosophy:
- Formula tests: Verify math with hand-calculated expected values
- Edge case tests: Handle NaN, Inf, empty arrays, zero division
- Boundary tests: Verify behavior at threshold boundaries
"""

import numpy as np
import pytest

from lobbacktest.metrics.risk import (
    CalmarRatio,
    MaxDrawdown,
    SharpeRatio,
    SortinoRatio,
)


class TestSharpeRatio:
    """Tests for SharpeRatio metric."""

    def test_sharpe_ratio_formula(self):
        """
        Verify: SR = mean(r) / std(r) * sqrt(periods_per_year)

        Hand-calculated example:
        - returns = [0.01, -0.005, 0.02, 0.003]
        - mean = 0.007
        - std = 0.00932 (population)
        - SR = 0.007 / 0.00932 * sqrt(252) = 11.91 (daily annualized)
        """
        returns = np.array([0.01, -0.005, 0.02, 0.003])

        # Compute expected (annual, assuming 1 period per day)
        mean = np.mean(returns)
        std = np.std(returns, ddof=0)  # Population std
        expected_sr = (mean / std) * np.sqrt(252)

        metric = SharpeRatio(trading_days_per_year=252, periods_per_day=1)
        result = metric.compute(returns, {})

        assert abs(result["SharpeRatio"] - expected_sr) < 1e-10, (
            f"Expected {expected_sr}, got {result['SharpeRatio']}"
        )

    def test_sharpe_ratio_with_config_override(self):
        """Test that context parameters override defaults."""
        returns = np.array([0.01, 0.02, 0.01, 0.02])

        metric = SharpeRatio(trading_days_per_year=252, periods_per_day=1)

        # Override via context
        context = {"annualization_factor": 1.0}  # No annualization
        result = metric.compute(returns, context)

        # Without annualization: mean/std
        expected = np.mean(returns) / np.std(returns, ddof=0)
        assert abs(result["SharpeRatio"] - expected) < 1e-10

    def test_sharpe_ratio_empty_returns(self):
        """Test handling of empty returns array."""
        metric = SharpeRatio()
        result = metric.compute(np.array([]), {})
        assert result["SharpeRatio"] == 0.0

    def test_sharpe_ratio_single_return(self):
        """Test handling of single return (std undefined)."""
        metric = SharpeRatio()
        result = metric.compute(np.array([0.01]), {})
        assert result["SharpeRatio"] == 0.0

    def test_sharpe_ratio_zero_std(self):
        """Test handling of zero standard deviation."""
        # All returns are the same
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        metric = SharpeRatio()
        result = metric.compute(returns, {})
        assert result["SharpeRatio"] == 0.0

    def test_sharpe_ratio_negative_mean(self):
        """Test that negative mean produces negative Sharpe."""
        returns = np.array([-0.01, -0.02, -0.01, -0.02])
        metric = SharpeRatio()
        result = metric.compute(returns, {})
        assert result["SharpeRatio"] < 0

    def test_sharpe_ratio_all_positive(self):
        """Test that all positive returns produce positive Sharpe."""
        returns = np.array([0.01, 0.02, 0.015, 0.025])
        metric = SharpeRatio()
        result = metric.compute(returns, {})
        assert result["SharpeRatio"] > 0


class TestSortinoRatio:
    """Tests for SortinoRatio metric."""

    def test_sortino_ratio_formula(self):
        """
        Verify: Sortino = mean(r) / downside_std(r) * sqrt(periods_per_year)

        Where downside_std = sqrt(mean(min(r, 0)^2))
        """
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])

        # Hand-calculate
        mean = np.mean(returns)
        negative = np.minimum(returns, 0)
        downside_var = np.mean(negative**2)
        downside_std = np.sqrt(downside_var)
        expected = (mean / downside_std) * np.sqrt(252)  # Daily

        metric = SortinoRatio(trading_days_per_year=252, periods_per_day=1)
        result = metric.compute(returns, {})

        assert abs(result["SortinoRatio"] - expected) < 1e-10, (
            f"Expected {expected}, got {result['SortinoRatio']}"
        )

    def test_sortino_ratio_no_negative_returns(self):
        """Test handling when there are no negative returns."""
        returns = np.array([0.01, 0.02, 0.015, 0.025])
        metric = SortinoRatio()
        result = metric.compute(returns, {})
        # Should return large positive (capped at 100)
        assert result["SortinoRatio"] == 100.0

    def test_sortino_ratio_all_negative(self):
        """Test with all negative returns."""
        returns = np.array([-0.01, -0.02, -0.015, -0.025])
        metric = SortinoRatio()
        result = metric.compute(returns, {})
        assert result["SortinoRatio"] < 0

    def test_sortino_ratio_empty_returns(self):
        """Test handling of empty returns."""
        metric = SortinoRatio()
        result = metric.compute(np.array([]), {})
        assert result["SortinoRatio"] == 0.0


class TestMaxDrawdown:
    """Tests for MaxDrawdown metric."""

    def test_max_drawdown_formula(self):
        """
        Verify: MDD = max((peak - equity) / peak)

        Test case: equity = [100, 110, 100, 120, 110, 130]
        - Peak at 110, drops to 100: DD = 10/110 = 9.09%
        - Peak at 120, drops to 110: DD = 10/120 = 8.33%
        - Max DD = 9.09%
        """
        # Build returns that produce this equity curve
        equity = np.array([100.0, 110.0, 100.0, 120.0, 110.0, 130.0])
        returns = np.diff(equity) / equity[:-1]

        context = {"equity_curve": equity}

        metric = MaxDrawdown()
        result = metric.compute(returns, context)

        expected_dd = 10.0 / 110.0  # 9.09%
        assert abs(result["MaxDrawdown"] - expected_dd) < 0.001, (
            f"Expected {expected_dd}, got {result['MaxDrawdown']}"
        )

    def test_max_drawdown_monotonic_increase(self):
        """Test that monotonically increasing equity has zero drawdown."""
        equity = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        returns = np.diff(equity) / equity[:-1]

        context = {"equity_curve": equity}

        metric = MaxDrawdown()
        result = metric.compute(returns, context)

        assert result["MaxDrawdown"] == 0.0

    def test_max_drawdown_total_loss(self):
        """Test 100% loss produces max drawdown of 1.0."""
        equity = np.array([100.0, 50.0, 25.0, 10.0, 0.01])
        returns = np.diff(equity) / equity[:-1]

        context = {"equity_curve": equity}

        metric = MaxDrawdown()
        result = metric.compute(returns, context)

        # Max DD = (100 - 0.01) / 100 ≈ 0.9999
        assert result["MaxDrawdown"] > 0.99

    def test_max_drawdown_from_returns_only(self):
        """Test computing drawdown from returns without equity_curve in context."""
        returns = np.array([0.1, -0.1, 0.2, -0.05])

        # Equity: 1.0 -> 1.1 -> 0.99 -> 1.188 -> 1.129
        # Peak: 1.0, 1.1, 1.1, 1.188, 1.188
        # DD: 0, 0, 0.1/1.1, 0, 0.059/1.188

        metric = MaxDrawdown()
        result = metric.compute(returns, {"initial_capital": 1.0})

        # Max DD = 0.1 / 1.1 = 0.0909
        expected = 0.1 / 1.1
        assert abs(result["MaxDrawdown"] - expected) < 0.01

    def test_max_drawdown_empty_returns(self):
        """Test handling of empty returns."""
        metric = MaxDrawdown()
        result = metric.compute(np.array([]), {})
        assert result["MaxDrawdown"] == 0.0


class TestCalmarRatio:
    """Tests for CalmarRatio metric."""

    def test_calmar_ratio_formula(self):
        """
        Verify: Calmar = AnnualReturn / MaxDrawdown
        """
        # Create known returns with known annual return and drawdown
        returns = np.array([0.001] * 252)  # Daily returns of 0.1%
        equity = 100 * np.cumprod(1 + returns)

        # Annual return: (1.001)^252 - 1 ≈ 28.6%
        annual_return = (1 + 0.001) ** 252 - 1

        # Max drawdown is 0 (monotonic), so Calmar undefined
        # Use context with known values instead
        context = {
            "AnnualReturn": 0.20,  # 20%
            "MaxDrawdown": 0.10,  # 10%
        }

        metric = CalmarRatio()
        result = metric.compute(returns, context)

        expected = 0.20 / 0.10
        assert abs(result["CalmarRatio"] - expected) < 1e-10

    def test_calmar_ratio_zero_drawdown(self):
        """Test that zero drawdown returns 0 (undefined)."""
        context = {
            "AnnualReturn": 0.20,
            "MaxDrawdown": 0.0,
        }

        metric = CalmarRatio()
        result = metric.compute(np.array([0.01, 0.02]), context)
        assert result["CalmarRatio"] == 0.0

    def test_calmar_ratio_negative_return(self):
        """Test negative annual return produces negative Calmar."""
        context = {
            "AnnualReturn": -0.20,  # -20%
            "MaxDrawdown": 0.30,  # 30%
        }

        metric = CalmarRatio()
        result = metric.compute(np.array([-0.01, -0.02]), context)
        assert result["CalmarRatio"] < 0

