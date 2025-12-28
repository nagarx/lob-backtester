"""
Risk-adjusted performance metrics.

Metrics:
- SharpeRatio: Return per unit of total risk
- SortinoRatio: Return per unit of downside risk
- MaxDrawdown: Maximum peak-to-trough decline
- CalmarRatio: Annual return over max drawdown
"""

from typing import Any, Dict, Mapping

import numpy as np

from lobbacktest.metrics.base import Metric


class SharpeRatio(Metric):
    """
    Annualized Sharpe Ratio.

    Formula:
        SR = mean(r) / std(r) * sqrt(periods_per_year)

    Where:
        r = per-period returns
        periods_per_year = trading_days_per_year * periods_per_day

    Reference:
        Sharpe, W. F. (1966). "Mutual Fund Performance"
        Journal of Business, 39(S1), 119-138.

    Notes:
        - Assumes risk-free rate = 0 (common for HFT backtests)
        - Uses sample std (ddof=0) for consistency with hftbacktest
    """

    def __init__(
        self,
        name: str = None,
        trading_days_per_year: float = 252.0,
        periods_per_day: float = 1000.0,
    ):
        """
        Initialize SharpeRatio metric.

        Args:
            name: Optional custom name (default: "SharpeRatio")
            trading_days_per_year: Trading days per year (default: 252)
            periods_per_day: Trading periods per day (default: 1000)
        """
        self._name = name or "SharpeRatio"
        self.trading_days_per_year = trading_days_per_year
        self.periods_per_day = periods_per_day

    @property
    def name(self) -> str:
        return self._name

    @property
    def annualization_factor(self) -> float:
        """sqrt(periods_per_year) for annualization."""
        return np.sqrt(self.trading_days_per_year * self.periods_per_day)

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute annualized Sharpe ratio.

        Args:
            returns: Array of per-period returns
            context: May contain:
                - "annualization_factor": override default
                - "trading_days_per_year": override default
                - "periods_per_day": override default

        Returns:
            {"SharpeRatio": sharpe_ratio}

        Edge cases:
            - Empty returns: 0.0
            - Zero std: 0.0 (cannot compute ratio)
            - Negative mean with positive std: negative SR
        """
        if not self.validate_returns(returns):
            return {self.name: 0.0}

        if len(returns) < 2:
            return {self.name: 0.0}

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=0)  # Population std

        if std_return == 0 or not np.isfinite(std_return):
            return {self.name: 0.0}

        # Get annualization factor
        if "annualization_factor" in context:
            ann_factor = context["annualization_factor"]
        else:
            trading_days = context.get(
                "trading_days_per_year", self.trading_days_per_year
            )
            periods = context.get("periods_per_day", self.periods_per_day)
            ann_factor = np.sqrt(trading_days * periods)

        sharpe = (mean_return / std_return) * ann_factor

        return {self.name: float(sharpe)}


class SortinoRatio(Metric):
    """
    Annualized Sortino Ratio.

    Like Sharpe, but only penalizes downside volatility.

    Formula:
        Sortino = mean(r) / downside_std(r) * sqrt(periods_per_year)

    Where:
        downside_std = sqrt(mean(min(r, 0)^2))

    Reference:
        Sortino, F., & van der Meer, R. (1991).
        "Downside risk". Journal of Portfolio Management.

    Notes:
        - Uses 0 as the target return (MAR = 0)
        - More appropriate for asymmetric return distributions
    """

    def __init__(
        self,
        name: str = None,
        trading_days_per_year: float = 252.0,
        periods_per_day: float = 1000.0,
    ):
        """
        Initialize SortinoRatio metric.

        Args:
            name: Optional custom name (default: "SortinoRatio")
            trading_days_per_year: Trading days per year (default: 252)
            periods_per_day: Trading periods per day (default: 1000)
        """
        self._name = name or "SortinoRatio"
        self.trading_days_per_year = trading_days_per_year
        self.periods_per_day = periods_per_day

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute annualized Sortino ratio.

        Args:
            returns: Array of per-period returns
            context: May contain annualization parameters

        Returns:
            {"SortinoRatio": sortino_ratio}

        Edge cases:
            - No negative returns: Uses small epsilon to avoid Inf
            - All negative: Normal computation
        """
        if not self.validate_returns(returns):
            return {self.name: 0.0}

        if len(returns) < 2:
            return {self.name: 0.0}

        mean_return = np.mean(returns)

        # Downside deviation: sqrt(mean of squared negative returns)
        negative_returns = np.minimum(returns, 0)
        downside_var = np.mean(negative_returns**2)
        downside_std = np.sqrt(downside_var)

        # Handle case with no downside
        if downside_std == 0 or not np.isfinite(downside_std):
            # No downside risk - return large positive if mean > 0, 0 otherwise
            if mean_return > 0:
                return {self.name: 100.0}  # Cap at reasonable value
            else:
                return {self.name: 0.0}

        # Get annualization factor
        if "annualization_factor" in context:
            ann_factor = context["annualization_factor"]
        else:
            trading_days = context.get(
                "trading_days_per_year", self.trading_days_per_year
            )
            periods = context.get("periods_per_day", self.periods_per_day)
            ann_factor = np.sqrt(trading_days * periods)

        sortino = (mean_return / downside_std) * ann_factor

        return {self.name: float(sortino)}


class MaxDrawdown(Metric):
    """
    Maximum Drawdown.

    The maximum observed loss from a peak to a trough.

    Formula:
        MDD = max((peak_i - equity_i) / peak_i) for all i

    Where:
        peak_i = max(equity_0, equity_1, ..., equity_i)

    Reference:
        Standard risk measure used in portfolio management

    Notes:
        - Returns value between 0 and 1 (0% to 100%)
        - Requires equity_curve in context (computed from returns if not present)
    """

    def __init__(self, name: str = None):
        """
        Initialize MaxDrawdown metric.

        Args:
            name: Optional custom name (default: "MaxDrawdown")
        """
        self._name = name or "MaxDrawdown"

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute maximum drawdown.

        Args:
            returns: Array of per-period returns
            context: May contain:
                - "equity_curve": Pre-computed equity curve
                - "initial_capital": Starting capital (default: 1.0)

        Returns:
            {"MaxDrawdown": max_drawdown} where max_drawdown in [0, 1]

        Edge cases:
            - Monotonically increasing: 0.0 (no drawdown)
            - Total loss: 1.0 (100% drawdown)
        """
        if not self.validate_returns(returns):
            return {self.name: 0.0}

        # Get or compute equity curve
        if "equity_curve" in context:
            equity = context["equity_curve"]
        else:
            # Compute from returns
            initial = context.get("initial_capital", 1.0)
            equity = initial * np.cumprod(1 + returns)

        if len(equity) == 0:
            return {self.name: 0.0}

        # Compute running maximum (peak)
        peak = np.maximum.accumulate(equity)

        # Compute drawdown at each point
        drawdown = (peak - equity) / peak

        # Handle division by zero (peak = 0)
        drawdown = np.where(peak > 0, drawdown, 0.0)

        max_dd = float(np.max(drawdown))

        # Clamp to [0, 1]
        max_dd = max(0.0, min(1.0, max_dd))

        return {self.name: max_dd}


class CalmarRatio(Metric):
    """
    Calmar Ratio.

    Annualized return divided by maximum drawdown.

    Formula:
        Calmar = AnnualReturn / MaxDrawdown

    Reference:
        Young, T. W. (1991). "Calmar Ratio: A Smoother Tool"
        Futures Magazine.

    Notes:
        - Higher is better
        - Undefined if MaxDrawdown = 0 (returns 0)
    """

    def __init__(
        self,
        name: str = None,
        trading_days_per_year: float = 252.0,
        periods_per_day: float = 1000.0,
    ):
        """
        Initialize CalmarRatio metric.

        Args:
            name: Optional custom name (default: "CalmarRatio")
            trading_days_per_year: For annualization (default: 252)
            periods_per_day: For annualization (default: 1000)
        """
        self._name = name or "CalmarRatio"
        self.trading_days_per_year = trading_days_per_year
        self.periods_per_day = periods_per_day

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute Calmar ratio.

        Args:
            returns: Array of per-period returns
            context: May contain:
                - "AnnualReturn": Pre-computed annual return
                - "MaxDrawdown": Pre-computed max drawdown

        Returns:
            {"CalmarRatio": calmar_ratio}

        Edge cases:
            - Zero drawdown: 0.0 (undefined)
            - Negative return: negative Calmar
        """
        if not self.validate_returns(returns):
            return {self.name: 0.0}

        # Get or compute annual return
        if "AnnualReturn" in context:
            annual_return = context["AnnualReturn"]
        else:
            # Compute annual return
            total_return = float(np.prod(1 + returns) - 1)
            if total_return <= -1.0:
                annual_return = -1.0
            else:
                n_periods = len(returns)
                periods_per_year = self.trading_days_per_year * self.periods_per_day
                exponent = periods_per_year / n_periods
                annual_return = (1 + total_return) ** exponent - 1

        # Get or compute max drawdown
        if "MaxDrawdown" in context:
            max_dd = context["MaxDrawdown"]
        else:
            # Compute max drawdown
            equity = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(equity)
            drawdown = np.where(peak > 0, (peak - equity) / peak, 0.0)
            max_dd = float(np.max(drawdown))

        # Compute Calmar
        if max_dd == 0 or not np.isfinite(max_dd):
            return {self.name: 0.0}

        calmar = annual_return / max_dd

        return {self.name: float(calmar)}

