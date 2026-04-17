"""
Return-based metrics.

Metrics:
- TotalReturn: Cumulative return over the period
- AnnualReturn: Annualized return
"""

from typing import Any, Dict, Mapping

import numpy as np

from lobbacktest.metrics.base import Metric


class TotalReturn(Metric):
    """
    Total cumulative return over the period.

    Formula:
        total_return = (1 + r_1) * (1 + r_2) * ... * (1 + r_n) - 1

    Or equivalently:
        total_return = final_equity / initial_equity - 1

    Reference:
        Standard financial return calculation
    """

    def __init__(self, *, name: str = None):
        """
        Initialize TotalReturn metric.

        Args:
            name: Optional custom name (default: "TotalReturn")
        """
        self._name = name or "TotalReturn"

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute total return from period returns.

        Args:
            returns: Array of per-period returns
            context: Not used for this metric

        Returns:
            {"TotalReturn": cumulative_return}

        Edge cases:
            - Empty returns: 0.0
            - All zeros: 0.0
        """
        if not self.validate_returns(returns):
            return {self.name: 0.0}

        # Compound returns: (1+r1) * (1+r2) * ... - 1
        cumulative = np.prod(1 + returns) - 1

        return {self.name: float(cumulative)}


class AnnualReturn(Metric):
    """
    Annualized return (CAGR).

    Formula:
        annual_return = (1 + total_return)^(periods_per_year / n_periods) - 1

    Where:
        periods_per_year = trading_days_per_year * periods_per_day
        n_periods = len(returns)

    Reference:
        Standard CAGR formula
    """

    def __init__(
        self,
        *,
        name: str = None,
        trading_days_per_year: float = 252.0,
        periods_per_day: float = 1000.0,
    ):
        """
        Initialize AnnualReturn metric.

        All parameters are keyword-only (see SharpeRatio docstring for rationale).

        Args:
            name: Optional custom name (default: "AnnualReturn")
            trading_days_per_year: Trading days per year (default: 252)
            periods_per_day: Trading periods per day (default: 1000)
        """
        self._name = name or "AnnualReturn"
        self.trading_days_per_year = trading_days_per_year
        self.periods_per_day = periods_per_day

    @property
    def name(self) -> str:
        return self._name

    @property
    def periods_per_year(self) -> float:
        """Total periods per year."""
        return self.trading_days_per_year * self.periods_per_day

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute annualized return.

        Args:
            returns: Array of per-period returns
            context: May contain:
                - "TotalReturn": pre-computed total return (optional)
                - "trading_days_per_year": override default (optional)
                - "periods_per_day": override default (optional)

        Returns:
            {"AnnualReturn": annualized_return}

        Edge cases:
            - Empty returns: 0.0
            - Zero periods: 0.0
            - Negative total (loss): Returns negative annualized
        """
        if not self.validate_returns(returns):
            return {self.name: 0.0}

        n_periods = len(returns)
        if n_periods == 0:
            return {self.name: 0.0}

        # Get total return (from context or compute)
        if "TotalReturn" in context:
            total_return = context["TotalReturn"]
        else:
            total_return = float(np.prod(1 + returns) - 1)

        # Handle negative returns (would cause NaN with fractional exponent)
        if total_return <= -1.0:
            # Total loss - return -1 (100% loss annualized)
            return {self.name: -1.0}

        # Get periods per year (from context or defaults)
        periods_per_year = (
            context.get("trading_days_per_year", self.trading_days_per_year)
            * context.get("periods_per_day", self.periods_per_day)
        )

        # Annualize: (1 + total)^(periods_per_year / n) - 1
        exponent = periods_per_year / n_periods

        # Guard against overflow for very short backtests or extreme returns
        # If exponent is too large, the result would be meaningless anyway
        try:
            if exponent > 1000:
                # For extremely short backtests, just return approximate annualized
                # using continuous compounding: e^(r * periods_per_year / n)
                annual_return = np.exp(np.log1p(total_return) * exponent) - 1
                # Clip to reasonable range
                annual_return = np.clip(annual_return, -1.0, 1e10)
            else:
                annual_return = (1 + total_return) ** exponent - 1
        except (OverflowError, FloatingPointError):
            # If computation fails, return a large positive or negative value
            annual_return = 1e10 if total_return > 0 else -1.0

        return {self.name: float(annual_return)}

