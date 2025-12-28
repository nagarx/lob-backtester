"""
Backtest execution engines.

This module provides the core backtest execution logic:
- Backtester: Main entry point for running backtests
- VectorizedEngine: Numpy-based fast execution

Usage:
    >>> from lobbacktest.engine import Backtester
    >>> backtester = Backtester(config)
    >>> result = backtester.run(data, strategy)
"""

from lobbacktest.engine.vectorized import BacktestData, Backtester, VectorizedEngine

__all__ = [
    "BacktestData",
    "Backtester",
    "VectorizedEngine",
]

