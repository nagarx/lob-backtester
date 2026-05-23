"""
Statistics and aggregation module.

Provides fluent API for computing and displaying backtest statistics.

Usage:
    >>> from lobbacktest.stats import BacktestStats
    >>> stats = BacktestStats(result, periods_per_day=390.0).compute()
    >>> print(stats.summary())
    >>> stats.plot()

Note:
    ``.daily()`` / ``.monthly()`` raise ``NotImplementedError`` until
    ``BacktestResult`` exposes ``timestamps_ns``. See FIND-040 in
    ``VALIDATION_FINDINGS_2026_05_14.md``.

    HF-2 closure (2026-05-22, sister of #PY-263): pass explicit
    ``periods_per_day=<value>`` to compute correct annualization.
    Omitting it emits ``DeprecationWarning`` and falls back to legacy
    1000.0 — silently inflates Sharpe/Sortino/Calmar/AnnualReturn
    ~1.6018x at 60s time-based bins. See ``BacktestStats`` docstring
    for full semantics.
"""

from lobbacktest.stats.stats import BacktestStats

__all__ = ["BacktestStats"]

