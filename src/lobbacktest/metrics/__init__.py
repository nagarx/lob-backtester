"""
Performance metrics for backtesting.

This module provides composable metrics following the ABC pattern:
- Metric: Base class for all metrics
- Risk metrics: Sharpe, Sortino, MaxDrawdown, Calmar
- Trading metrics: WinRate, ProfitFactor, PayoffRatio
- Prediction metrics: DirectionalAccuracy, SignalRate

Usage:
    >>> from lobbacktest.metrics import SharpeRatio, MaxDrawdown
    >>> sr = SharpeRatio()
    >>> result = sr.compute(returns, context={})
    >>> print(result["SharpeRatio"])
"""

from lobbacktest.metrics.base import Metric, MetricResult
from lobbacktest.metrics.returns import AnnualReturn, TotalReturn
from lobbacktest.metrics.risk import CalmarRatio, MaxDrawdown, SharpeRatio, SortinoRatio
from lobbacktest.metrics.trading import (
    AverageLoss,
    AverageWin,
    Expectancy,
    PayoffRatio,
    ProfitFactor,
    WinRate,
)
from lobbacktest.metrics.prediction import (
    DirectionalAccuracy,
    DownPrecision,
    SignalRate,
    UpPrecision,
)

__all__ = [
    # Base
    "Metric",
    "MetricResult",
    # Returns
    "TotalReturn",
    "AnnualReturn",
    # Risk
    "SharpeRatio",
    "SortinoRatio",
    "MaxDrawdown",
    "CalmarRatio",
    # Trading
    "WinRate",
    "ProfitFactor",
    "AverageWin",
    "AverageLoss",
    "PayoffRatio",
    "Expectancy",
    # Prediction
    "DirectionalAccuracy",
    "SignalRate",
    "UpPrecision",
    "DownPrecision",
]

