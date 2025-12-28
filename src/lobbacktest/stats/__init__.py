"""
Statistics and aggregation module.

Provides fluent API for computing and displaying backtest statistics.

Usage:
    >>> from lobbacktest.stats import BacktestStats
    >>> stats = BacktestStats(result).daily().compute()
    >>> print(stats.summary())
    >>> stats.plot()
"""

from lobbacktest.stats.stats import BacktestStats

__all__ = ["BacktestStats"]

