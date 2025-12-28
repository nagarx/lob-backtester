"""
LOB-Backtester: Standalone backtesting library for LOB prediction models.

This library provides a vectorized backtesting engine for evaluating
direction prediction models trained with lob-model-trainer.

Example:
    >>> from lobbacktest import Backtester, DirectionStrategy
    >>> from lobbacktest.config import BacktestConfig
    >>>
    >>> config = BacktestConfig(initial_capital=100_000)
    >>> strategy = DirectionStrategy(predictions)
    >>> backtester = Backtester(config)
    >>> result = backtester.run(data, strategy)
    >>> print(result.summary())
"""

from lobbacktest.version import __version__

# Core types
from lobbacktest.types import (
    BacktestResult,
    Position,
    PositionSide,
    Trade,
    TradeSide,
)

# Configuration
from lobbacktest.config import BacktestConfig, CostConfig

# Engine
from lobbacktest.engine import BacktestData, Backtester

# Strategies
from lobbacktest.strategies import DirectionStrategy, Strategy, ThresholdStrategy

# Metrics
from lobbacktest.metrics import (
    CalmarRatio,
    Expectancy,
    MaxDrawdown,
    Metric,
    ProfitFactor,
    SharpeRatio,
    SortinoRatio,
    TotalReturn,
    WinRate,
)

# Stats
from lobbacktest.stats import BacktestStats

__all__ = [
    # Version
    "__version__",
    # Types
    "BacktestResult",
    "Position",
    "PositionSide",
    "Trade",
    "TradeSide",
    # Config
    "BacktestConfig",
    "CostConfig",
    # Engine
    "BacktestData",
    "Backtester",
    # Strategies
    "Strategy",
    "DirectionStrategy",
    "ThresholdStrategy",
    # Metrics
    "Metric",
    "SharpeRatio",
    "SortinoRatio",
    "MaxDrawdown",
    "CalmarRatio",
    "TotalReturn",
    "WinRate",
    "ProfitFactor",
    "Expectancy",
    # Stats
    "BacktestStats",
]

