"""
Statistics and aggregation module.

Provides fluent API for computing and displaying backtest statistics.

Usage:
    >>> from lobbacktest.stats import BacktestStats
    >>> stats = BacktestStats(result).compute()
    >>> print(stats.summary())
    >>> stats.plot()

Note:
    ``.daily()`` / ``.monthly()`` raise ``NotImplementedError`` until
    ``BacktestResult`` exposes ``timestamps_ns``. See FIND-040 in
    ``VALIDATION_FINDINGS_2026_05_14.md``.
"""

from lobbacktest.stats.stats import BacktestStats

__all__ = ["BacktestStats"]

