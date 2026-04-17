"""
Trading strategies for backtesting.

Strategies:
- DirectionStrategy: Trade based on Up/Down class predictions
- ThresholdStrategy: Trade only above confidence threshold
- ReadabilityStrategy: Confidence-gated (agreement + confirmation + spread)
- RegressionStrategy: Continuous bps return predictions with magnitude gate

Usage:
    >>> from lobbacktest.strategies import RegressionStrategy, RegressionStrategyConfig
    >>> strategy = RegressionStrategy(predicted_returns, spreads, prices,
    ...     config=RegressionStrategyConfig(min_return_bps=5.0))
"""

from lobbacktest.strategies.base import Signal, SignalOutput, Strategy
from lobbacktest.strategies.direction import DirectionStrategy, ThresholdStrategy
from lobbacktest.strategies.readability import ReadabilityStrategy, ReadabilityConfig
from lobbacktest.strategies.regression import RegressionStrategy, RegressionStrategyConfig
from lobbacktest.strategies.hybrid import ReadabilityHybridStrategy, ReadabilityHybridConfig
from lobbacktest.strategies.holding import (
    HoldingPolicy,
    HorizonAlignedPolicy,
    create_holding_policy,
)

__all__ = [
    "Strategy",
    "Signal",
    "SignalOutput",
    "DirectionStrategy",
    "ThresholdStrategy",
    "ReadabilityStrategy",
    "ReadabilityConfig",
    "RegressionStrategy",
    "RegressionStrategyConfig",
    "ReadabilityHybridStrategy",
    "ReadabilityHybridConfig",
    "HoldingPolicy",
    "HorizonAlignedPolicy",
    "create_holding_policy",
]

