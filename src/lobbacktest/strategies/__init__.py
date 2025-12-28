"""
Trading strategies for backtesting.

This module provides strategy implementations:
- Strategy: Abstract base class
- DirectionStrategy: Trade based on Up/Down predictions
- ThresholdStrategy: Trade only above confidence threshold

Usage:
    >>> from lobbacktest.strategies import DirectionStrategy
    >>> strategy = DirectionStrategy(predictions)
    >>> signals = strategy.generate_signals(prices)
"""

from lobbacktest.strategies.base import Signal, Strategy
from lobbacktest.strategies.direction import DirectionStrategy, ThresholdStrategy

__all__ = [
    "Strategy",
    "Signal",
    "DirectionStrategy",
    "ThresholdStrategy",
]

