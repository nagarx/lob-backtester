"""
Reporting and visualization module.

Provides utilities for generating reports and plots from backtest results.

Usage:
    >>> from lobbacktest.reports import generate_report, plot_equity_curve
    >>> report = generate_report(result)
    >>> fig = plot_equity_curve(result)
"""

from lobbacktest.reports.summary import generate_report, comparison_table
from lobbacktest.reports.plots import (
    plot_equity_curve,
    plot_returns_distribution,
    plot_drawdown,
    plot_comparison,
)

__all__ = [
    "generate_report",
    "comparison_table",
    "plot_equity_curve",
    "plot_returns_distribution",
    "plot_drawdown",
    "plot_comparison",
]

