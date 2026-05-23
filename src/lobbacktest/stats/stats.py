"""
Fluent statistics API for backtest results.

Inspired by hftbacktest's Stats pattern, provides chainable
operations for computing and displaying statistics.

Example:
    >>> stats = (
    ...     BacktestStats(result, periods_per_day=390.0)
    ...         .with_book_size(100_000)
    ...         .compute()
    ... )
    >>> print(stats.summary())
    >>> stats.plot()

Note:
    ``.daily()`` / ``.monthly()`` raise ``NotImplementedError`` until
    ``BacktestResult`` exposes ``timestamps_ns``. See FIND-040 in
    ``VALIDATION_FINDINGS_2026_05_14.md``.

HF-2 closure (2026-05-22, sister of #PY-263 BacktestConfig 2026-05-21):
    ``BacktestStats`` accepts ``periods_per_day`` at construction (or via
    ``.with_periods_per_day(...)``). When omitted, ``.compute()`` emits a
    ``DeprecationWarning`` and the metric chain falls back to the legacy
    1000.0 default (matching event-based 1000-events/sample sampling).
    At time-based sub-daily bins (e.g., 60s = 390 periods/day), the
    legacy default silently inflates Sharpe/Sortino/Calmar/AnnualReturn
    by ~1.6018x (sqrt(1000/390)).

    The engine path at ``vectorized.py:623-664`` already propagates
    ``BacktestConfig.resolved_periods_per_day`` via the BacktestContext
    dict; this closure extends the same discipline to the operator-facing
    fluent ``BacktestStats`` API which builds its own context dict (the
    fluent API does NOT go through the engine, so #PY-263's engine-level
    closure does not transitively close this surface).
"""

import warnings
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
        ...         .compute()
        ... )

    Note:
        ``.daily()`` / ``.monthly()`` raise ``NotImplementedError`` until
        ``BacktestResult`` exposes ``timestamps_ns``. See FIND-040.
    """

    def __init__(
        self,
        result: BacktestResult,
        *,
        periods_per_day: Optional[float] = None,
    ):
        """
        Initialize BacktestStats.

        Args:
            result: BacktestResult from a backtest run
            periods_per_day: Optional explicit periods_per_day for
                annualization (Sharpe/Sortino/Calmar/AnnualReturn).
                Sister-closure of #PY-263 BacktestConfig fix (2026-05-21).
                When None (default), ``.compute()`` emits
                ``DeprecationWarning`` and the metric chain falls back to
                legacy default 1000.0 (matches event-based 1000-events/
                sample sampling). At time-based sub-daily bins (e.g., 60s
                = 390 periods/day) this default produces ~1.6018x
                inflated Sharpe via sqrt(1000/390). Pass explicit value
                (e.g., ``BacktestConfig.resolved_periods_per_day``) to
                silence + compute correctly. Keyword-only per
                ``SharpeRatio`` C1-positional-trap convention.
        """
        # Q3 ASYMMETRY fix (2026-05-22): mid-impl gate flagged that `__init__`
        # accepted `periods_per_day=0.0` silently while `with_periods_per_day`
        # raised — inconsistent with §5 fail-fast. Validate at construction
        # so silent zero-injection at risk.py:119 (`sqrt(252*0)=0` → Sharpe=0)
        # is caught at the contract boundary.
        if periods_per_day is not None and periods_per_day <= 0:
            raise ValueError(
                f"BacktestStats: periods_per_day must be > 0 if specified, "
                f"got {periods_per_day}"
            )
        self._result = result
        self._book_size: Optional[float] = None
        self._periods_per_day: Optional[float] = periods_per_day
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

    def with_periods_per_day(self, periods_per_day: float) -> "BacktestStats":
        """
        Set periods_per_day for annualization (sister of #PY-263 closure).

        Args:
            periods_per_day: Trading periods per day. For time-based bins
                pass ``BacktestConfig.resolved_periods_per_day``
                (mode-aware dispatch from #PY-263). For event-based
                sampling pass 1000.0 explicitly to silence the legacy
                default warning.

        Returns:
            self for chaining

        Notes:
            Idempotent: subsequent calls overwrite the prior value
            (last-call-wins). Matches ``.with_book_size()`` semantics.
        """
        if periods_per_day <= 0:
            raise ValueError(
                f"BacktestStats.with_periods_per_day: periods_per_day "
                f"must be > 0, got {periods_per_day}"
            )
        self._periods_per_day = periods_per_day
        return self

    def daily(self) -> "BacktestStats":
        """
        Daily period aggregation — NOT YET SUPPORTED.

        Raises:
            NotImplementedError: ``BacktestResult`` does not currently carry
                ``timestamps_ns``, so daily aggregation cannot be computed.
                Use ``.compute()`` for full-corpus metrics. Track at FIND-040.
        """
        raise NotImplementedError(
            "BacktestStats.daily() requires per-period timestamps on BacktestResult; "
            "BacktestResult does not currently carry timestamps_ns. Daily aggregation is "
            "not yet supported. Use .compute() for full-corpus metrics instead. "
            "Track at FIND-040 in lob-backtester/VALIDATION_FINDINGS_2026_05_14.md."
        )

    def monthly(self) -> "BacktestStats":
        """
        Monthly period aggregation — NOT YET SUPPORTED.

        Raises:
            NotImplementedError: ``BacktestResult`` does not currently carry
                ``timestamps_ns``, so monthly aggregation cannot be computed.
                Use ``.compute()`` for full-corpus metrics. Track at FIND-040.
        """
        raise NotImplementedError(
            "BacktestStats.monthly() requires per-period timestamps on BacktestResult; "
            "BacktestResult does not currently carry timestamps_ns. Monthly aggregation is "
            "not yet supported. Use .compute() for full-corpus metrics instead. "
            "Track at FIND-040 in lob-backtester/VALIDATION_FINDINGS_2026_05_14.md."
        )

    def full(self) -> "BacktestStats":
        """
        Period selector — full-corpus (default).

        No-op; ``.compute()`` returns full-corpus metrics regardless of period.
        Kept for fluent-API symmetry with future ``.daily()`` / ``.monthly()``.

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

        # HF-2 (2026-05-22): sister-closure of #PY-263 BacktestConfig
        # 2026-05-21. Inject periods_per_day into the context dict so
        # that AnnualReturn (returns.py:171), SharpeRatio (risk.py:118),
        # SortinoRatio (risk.py:227), and CalmarRatio (transitively via
        # context["AnnualReturn"]) ALL read the explicit value instead
        # of falling back to each metric's class default of 1000.0.
        #
        # Pre-fix this surface was the operator-facing fluent-API gap
        # left open after #PY-263's engine-path closure: the engine at
        # vectorized.py:623-664 propagates BacktestConfig.resolved_periods_per_day
        # via BacktestContext, but BacktestStats.compute() builds its
        # OWN context dict and constructed metrics with their 1000.0
        # default — silently inflating annualized metrics ~1.6018x at
        # 60s time-based bins (sqrt(1000/390)).
        if self._periods_per_day is not None:
            context["periods_per_day"] = self._periods_per_day
        else:
            warnings.warn(
                "BacktestStats.compute(): periods_per_day not specified; "
                "annualized metrics (SharpeRatio, SortinoRatio, CalmarRatio, "
                "AnnualReturn) will fall back to legacy default 1000.0 "
                "(event-based 1000-events/sample). At time-based sub-daily "
                "bins (e.g., 60s = 390 periods/day), this silently inflates "
                "Sharpe/Sortino/Calmar/AnnualReturn by ~1.6018x via "
                "sqrt(1000/390). Pass explicit periods_per_day=<value> to "
                "BacktestStats(...) or call .with_periods_per_day(<value>) "
                "to silence + compute correctly. Sister-closure of #PY-263 "
                "(2026-05-21 BacktestConfig mode-aware dispatch); see "
                "BacktestConfig.resolved_periods_per_day.",
                DeprecationWarning,
                stacklevel=2,
            )

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

