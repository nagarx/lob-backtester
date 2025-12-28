"""
Fluent statistics API for backtest results.

Inspired by hftbacktest's Stats pattern, provides chainable
operations for computing and displaying statistics.

Example:
    >>> stats = (
    ...     BacktestStats(result)
    ...         .with_book_size(100_000)
    ...         .daily()
    ...         .compute()
    ... )
    >>> print(stats.summary())
    >>> stats.plot()
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from lobbacktest.metrics.base import Metric
from lobbacktest.metrics.prediction import DirectionalAccuracy, SignalRate
from lobbacktest.metrics.returns import AnnualReturn, TotalReturn
from lobbacktest.metrics.risk import CalmarRatio, MaxDrawdown, SharpeRatio, SortinoRatio
from lobbacktest.metrics.trading import (
    AverageLoss,
    AverageWin,
    PayoffRatio,
    ProfitFactor,
    WinRate,
)
from lobbacktest.types import BacktestResult


@dataclass
class StatsSummary:
    """
    Summary statistics from a backtest.

    Contains computed metrics and metadata for display.
    """

    metrics: Dict[str, float]
    period: str  # "full", "daily", "monthly"
    book_size: Optional[float]
    n_periods: int
    n_trades: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metrics": self.metrics,
            "period": self.period,
            "book_size": self.book_size,
            "n_periods": self.n_periods,
            "n_trades": self.n_trades,
        }


class BacktestStats:
    """
    Fluent API for computing backtest statistics.

    Usage:
        >>> stats = BacktestStats(result)
        >>> stats = stats.with_book_size(100_000).compute()
        >>> print(stats.summary())

    Or chained:
        >>> stats = (
        ...     BacktestStats(result)
        ...         .with_book_size(100_000)
        ...         .daily()
        ...         .compute()
        ... )
    """

    def __init__(self, result: BacktestResult):
        """
        Initialize BacktestStats.

        Args:
            result: BacktestResult from a backtest run
        """
        self._result = result
        self._book_size: Optional[float] = None
        self._period: str = "full"
        self._metrics: List[Metric] = []
        self._computed: Optional[StatsSummary] = None

    def with_book_size(self, book_size: float) -> "BacktestStats":
        """
        Set the book size for normalization.

        Args:
            book_size: Capital/book size in USD

        Returns:
            self for chaining
        """
        self._book_size = book_size
        return self

    def daily(self) -> "BacktestStats":
        """
        Aggregate statistics daily.

        Returns:
            self for chaining
        """
        self._period = "daily"
        return self

    def monthly(self) -> "BacktestStats":
        """
        Aggregate statistics monthly.

        Returns:
            self for chaining
        """
        self._period = "monthly"
        return self

    def full(self) -> "BacktestStats":
        """
        Compute statistics for full period (default).

        Returns:
            self for chaining
        """
        self._period = "full"
        return self

    def with_metrics(self, metrics: List[Metric]) -> "BacktestStats":
        """
        Add custom metrics to compute.

        Args:
            metrics: List of Metric instances

        Returns:
            self for chaining
        """
        self._metrics.extend(metrics)
        return self

    def compute(self) -> "BacktestStats":
        """
        Compute all statistics.

        Returns:
            self for chaining
        """
        # Get default metrics if none specified
        if not self._metrics:
            self._metrics = [
                TotalReturn(),
                AnnualReturn(),
                SharpeRatio(),
                SortinoRatio(),
                MaxDrawdown(),
                CalmarRatio(),
                WinRate(),
                ProfitFactor(),
                AverageWin(),
                AverageLoss(),
                PayoffRatio(),
            ]

            # Add prediction metrics if labels available
            if self._result.labels is not None:
                self._metrics.extend([
                    DirectionalAccuracy(),
                    SignalRate(),
                ])

        # Build context
        context = {
            "equity_curve": self._result.equity_curve,
            "trade_pnls": self._get_trade_pnls(),
            "predictions": self._result.predictions,
            "labels": self._result.labels,
            "initial_capital": self._result.initial_capital,
        }

        if self._book_size:
            context["book_size"] = self._book_size

        # Compute metrics
        computed = {}
        for metric in self._metrics:
            result = metric.compute(self._result.returns, context)
            computed.update(result)
            context.update(result)

        # Create summary
        self._computed = StatsSummary(
            metrics=computed,
            period=self._period,
            book_size=self._book_size,
            n_periods=len(self._result.returns),
            n_trades=self._result.total_trades,
        )

        return self

    def _get_trade_pnls(self) -> np.ndarray:
        """Extract trade P&Ls from result."""
        return self._result.trade_pnls

    def summary(self) -> str:
        """
        Generate formatted summary string.

        Returns:
            Multi-line string with statistics
        """
        if self._computed is None:
            self.compute()

        lines = [
            "=" * 60,
            "BACKTEST STATISTICS",
            "=" * 60,
            f"Period: {self._computed.period}",
            f"Data points: {self._computed.n_periods:,}",
            f"Total trades: {self._computed.n_trades:,}",
        ]

        if self._computed.book_size:
            lines.append(f"Book size: ${self._computed.book_size:,.2f}")

        lines.append("-" * 60)
        lines.append("METRICS:")

        # Group metrics by category
        returns_metrics = ["TotalReturn", "AnnualReturn"]
        risk_metrics = ["SharpeRatio", "SortinoRatio", "MaxDrawdown", "CalmarRatio"]
        trading_metrics = [
            "WinRate",
            "ProfitFactor",
            "AverageWin",
            "AverageLoss",
            "PayoffRatio",
            "Expectancy",
        ]
        prediction_metrics = ["DirectionalAccuracy", "SignalRate"]

        for category, names in [
            ("Returns", returns_metrics),
            ("Risk", risk_metrics),
            ("Trading", trading_metrics),
            ("Prediction", prediction_metrics),
        ]:
            category_metrics = {
                k: v for k, v in self._computed.metrics.items() if k in names
            }
            if category_metrics:
                lines.append(f"\n  {category}:")
                for name, value in category_metrics.items():
                    if isinstance(value, float):
                        if "Rate" in name or "Accuracy" in name or name == "WinRate":
                            lines.append(f"    {name:20s} {value * 100:+.2f}%")
                        elif "Drawdown" in name:
                            lines.append(f"    {name:20s} {value * 100:.2f}%")
                        elif "Return" in name:
                            lines.append(f"    {name:20s} {value * 100:+.2f}%")
                        else:
                            lines.append(f"    {name:20s} {value:+.4f}")
                    else:
                        lines.append(f"    {name:20s} {value}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def plot(self, figsize: tuple = (12, 8)):
        """
        Generate equity curve and position plot.

        Args:
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Equity curve
        ax1 = axes[0]
        ax1.plot(self._result.equity_curve, label="Equity", color="blue")
        ax1.axhline(
            y=self._result.initial_capital,
            color="gray",
            linestyle="--",
            label="Initial Capital",
        )
        ax1.set_ylabel("Equity ($)")
        ax1.legend(loc="upper left")
        ax1.set_title("Equity Curve")
        ax1.grid(True, alpha=0.3)

        # Position
        ax2 = axes[1]
        ax2.fill_between(
            range(len(self._result.positions)),
            self._result.positions,
            0,
            alpha=0.5,
            color="green",
            where=self._result.positions > 0,
            label="Long",
        )
        ax2.fill_between(
            range(len(self._result.positions)),
            self._result.positions,
            0,
            alpha=0.5,
            color="red",
            where=self._result.positions < 0,
            label="Short",
        )
        ax2.set_ylabel("Position")
        ax2.legend(loc="upper left")
        ax2.set_title("Position")
        ax2.grid(True, alpha=0.3)

        # Returns distribution
        ax3 = axes[2]
        ax3.hist(self._result.returns * 100, bins=50, color="blue", alpha=0.7)
        ax3.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax3.set_xlabel("Period")
        ax3.set_ylabel("Return (%)")
        ax3.set_title("Returns Distribution")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @property
    def metrics(self) -> Dict[str, float]:
        """Get computed metrics."""
        if self._computed is None:
            self.compute()
        return self._computed.metrics

    @property
    def result(self) -> BacktestResult:
        """Get underlying BacktestResult."""
        return self._result

