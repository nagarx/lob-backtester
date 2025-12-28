"""
Base classes for trading strategies.

A strategy converts model predictions into trading signals.
The engine then executes these signals.

Design Philosophy:
- Strategies are stateless (no memory between calls)
- Strategies output signals, not trades (engine handles execution)
- Strategies are composable (can combine multiple strategies)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


class Signal(IntEnum):
    """
    Trading signal from strategy.

    Values:
        SELL (-1): Enter/increase short position
        HOLD (0): Maintain current position (no trade)
        BUY (1): Enter/increase long position
        EXIT (2): Close current position (regardless of direction)
    """

    SELL = -1
    HOLD = 0
    BUY = 1
    EXIT = 2


@dataclass
class SignalOutput:
    """
    Output from strategy signal generation.

    Attributes:
        signals: Array of Signal values (shape: N)
        confidence: Optional confidence scores (shape: N)
        metadata: Optional additional information
    """

    signals: np.ndarray  # Shape: (N,), dtype: int
    confidence: Optional[np.ndarray] = None  # Shape: (N,), dtype: float
    metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        """Validate signal output."""
        if self.signals.ndim != 1:
            raise ValueError(f"signals must be 1D, got shape {self.signals.shape}")
        if self.confidence is not None:
            if self.confidence.shape != self.signals.shape:
                raise ValueError(
                    f"confidence shape {self.confidence.shape} != "
                    f"signals shape {self.signals.shape}"
                )

    def __len__(self) -> int:
        return len(self.signals)


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Strategies convert predictions into trading signals that
    the backtest engine can execute.

    Example:
        >>> class MyStrategy(Strategy):
        ...     def __init__(self, predictions):
        ...         self.predictions = predictions
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return "MyStrategy"
        ...
        ...     def generate_signals(self, prices, index=None):
        ...         # Buy on Up, Sell on Down, Hold on Stable
        ...         signals = np.where(
        ...             self.predictions == 1, Signal.BUY,
        ...             np.where(self.predictions == -1, Signal.SELL, Signal.HOLD)
        ...         )
        ...         return SignalOutput(signals=signals)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this strategy.

        Returns:
            Strategy name for logging and comparison
        """
        raise NotImplementedError

    @abstractmethod
    def generate_signals(
        self,
        prices: np.ndarray,
        index: Optional[int] = None,
    ) -> SignalOutput:
        """
        Generate trading signals from prices.

        Args:
            prices: Price series (shape: N)
            index: Optional current index (for online/streaming mode)
                   If None, generate signals for entire series

        Returns:
            SignalOutput with signals array and optional confidence

        Notes:
            - Signals should be one of Signal enum values
            - Can use self.predictions set during __init__
            - Should handle edge cases (empty prices, etc.)
        """
        raise NotImplementedError

    def validate_predictions(
        self,
        predictions: np.ndarray,
        expected_length: int,
    ) -> bool:
        """
        Validate predictions array.

        Args:
            predictions: Predictions to validate
            expected_length: Expected array length

        Returns:
            True if predictions are valid

        Checks:
            - Array is 1D
            - Length matches expected
            - Values are finite
        """
        if predictions is None:
            return False
        if predictions.ndim != 1:
            return False
        if len(predictions) != expected_length:
            return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

