"""
Direction-based trading strategies.

Strategies:
- DirectionStrategy: Simple direction following (Up -> Buy, Down -> Sell)
- ThresholdStrategy: Trade only when prediction confidence exceeds threshold
"""

from typing import Optional

import numpy as np

from lobbacktest.strategies.base import Signal, SignalOutput, Strategy

# Label constants (consistent with lob-model-trainer)
LABEL_DOWN = -1
LABEL_STABLE = 0
LABEL_UP = 1

# Shifted labels (PyTorch CrossEntropyLoss format)
SHIFTED_LABEL_DOWN = 0
SHIFTED_LABEL_STABLE = 1
SHIFTED_LABEL_UP = 2


class DirectionStrategy(Strategy):
    """
    Simple direction-following strategy.

    Converts model predictions directly to trading signals:
        - Up prediction -> BUY
        - Down prediction -> SELL
        - Stable prediction -> HOLD

    Attributes:
        predictions: Model class predictions (shape: N)
        shifted: If True, predictions use shifted labels (0/1/2)

    Example:
        >>> predictions = np.array([1, 0, -1, 1, 0])  # Up, Stable, Down, Up, Stable
        >>> strategy = DirectionStrategy(predictions)
        >>> output = strategy.generate_signals(prices)
        >>> print(output.signals)  # [1, 0, -1, 1, 0]
    """

    def __init__(
        self,
        predictions: np.ndarray,
        shifted: bool = False,
        name: str = None,
    ):
        """
        Initialize DirectionStrategy.

        Args:
            predictions: Model class predictions (shape: N)
            shifted: If True, predictions use shifted labels (0/1/2 for Down/Stable/Up)
            name: Optional custom name (default: "DirectionStrategy")
        """
        self.predictions = np.asarray(predictions)
        self.shifted = shifted
        self._name = name or "DirectionStrategy"

        # Set label constants based on shifted flag
        if shifted:
            self.label_down = SHIFTED_LABEL_DOWN
            self.label_stable = SHIFTED_LABEL_STABLE
            self.label_up = SHIFTED_LABEL_UP
        else:
            self.label_down = LABEL_DOWN
            self.label_stable = LABEL_STABLE
            self.label_up = LABEL_UP

    @property
    def name(self) -> str:
        return self._name

    def generate_signals(
        self,
        prices: np.ndarray,
        index: Optional[int] = None,
    ) -> SignalOutput:
        """
        Generate trading signals from predictions.

        Args:
            prices: Price series (used for validation only)
            index: Optional current index (for online mode)

        Returns:
            SignalOutput with signals matching prediction direction

        Mapping:
            - prediction == Up label -> Signal.BUY
            - prediction == Down label -> Signal.SELL
            - prediction == Stable label -> Signal.HOLD
        """
        if index is not None:
            # Online mode: single signal
            pred = self.predictions[index]
            if pred == self.label_up:
                signal = Signal.BUY
            elif pred == self.label_down:
                signal = Signal.SELL
            else:
                signal = Signal.HOLD
            return SignalOutput(signals=np.array([signal]))

        # Batch mode: all signals
        if not self.validate_predictions(self.predictions, len(prices)):
            # Return HOLD for all if predictions invalid
            return SignalOutput(signals=np.zeros(len(prices), dtype=np.int8))

        # Map predictions to signals
        signals = np.where(
            self.predictions == self.label_up,
            Signal.BUY,
            np.where(
                self.predictions == self.label_down,
                Signal.SELL,
                Signal.HOLD,
            ),
        ).astype(np.int8)

        return SignalOutput(signals=signals)


class ThresholdStrategy(Strategy):
    """
    Direction strategy with confidence threshold.

    Only generates trading signals when model confidence exceeds
    a threshold. Otherwise, outputs HOLD.

    This is useful for:
    - Reducing noise from low-confidence predictions
    - Controlling signal rate
    - Focusing on high-quality signals

    Attributes:
        predictions: Model class predictions (shape: N)
        probabilities: Model output probabilities (shape: N, 3)
        threshold: Minimum confidence to trade (default: 0.5)

    Example:
        >>> probs = np.array([[0.3, 0.4, 0.3], [0.1, 0.2, 0.7]])
        >>> preds = np.argmax(probs, axis=1)  # [1, 2] = [Stable, Up]
        >>> strategy = ThresholdStrategy(preds, probs, threshold=0.6)
        >>> output = strategy.generate_signals(prices)
        >>> # Only second prediction (0.7 > 0.6) generates a signal
    """

    def __init__(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        threshold: float = 0.5,
        shifted: bool = True,
        name: str = None,
    ):
        """
        Initialize ThresholdStrategy.

        Args:
            predictions: Model class predictions (shape: N)
            probabilities: Model output probabilities (shape: N, 3)
            threshold: Minimum max probability to trade (default: 0.5)
            shifted: If True, predictions use shifted labels (0/1/2)
            name: Optional custom name
        """
        self.predictions = np.asarray(predictions)
        self.probabilities = np.asarray(probabilities)
        self.threshold = threshold
        self.shifted = shifted
        self._name = name or f"ThresholdStrategy(threshold={threshold})"

        # Validate
        if self.probabilities.ndim != 2 or self.probabilities.shape[1] != 3:
            raise ValueError(
                f"probabilities must have shape (N, 3), got {self.probabilities.shape}"
            )
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        # Set label constants
        if shifted:
            self.label_down = SHIFTED_LABEL_DOWN
            self.label_stable = SHIFTED_LABEL_STABLE
            self.label_up = SHIFTED_LABEL_UP
        else:
            self.label_down = LABEL_DOWN
            self.label_stable = LABEL_STABLE
            self.label_up = LABEL_UP

    @property
    def name(self) -> str:
        return self._name

    def generate_signals(
        self,
        prices: np.ndarray,
        index: Optional[int] = None,
    ) -> SignalOutput:
        """
        Generate signals with confidence thresholding.

        Args:
            prices: Price series
            index: Optional current index

        Returns:
            SignalOutput with:
                - signals: Trading signals (only for high-confidence predictions)
                - confidence: Max probability for each prediction
        """
        if index is not None:
            # Online mode
            pred = self.predictions[index]
            prob = self.probabilities[index]
            max_prob = np.max(prob)

            if max_prob < self.threshold:
                signal = Signal.HOLD
            elif pred == self.label_up:
                signal = Signal.BUY
            elif pred == self.label_down:
                signal = Signal.SELL
            else:
                signal = Signal.HOLD

            return SignalOutput(
                signals=np.array([signal]),
                confidence=np.array([max_prob]),
            )

        # Batch mode
        if len(self.predictions) != len(prices):
            return SignalOutput(
                signals=np.zeros(len(prices), dtype=np.int8),
                confidence=np.zeros(len(prices)),
            )

        # Get max probability for each prediction
        max_probs = np.max(self.probabilities, axis=1)

        # Start with direction-based signals
        signals = np.where(
            self.predictions == self.label_up,
            Signal.BUY,
            np.where(
                self.predictions == self.label_down,
                Signal.SELL,
                Signal.HOLD,
            ),
        ).astype(np.int8)

        # Zero out signals below threshold
        signals = np.where(max_probs >= self.threshold, signals, Signal.HOLD)

        return SignalOutput(
            signals=signals,
            confidence=max_probs,
        )


class ExitOnReverseStrategy(Strategy):
    """
    Strategy that exits position on signal reversal.

    When holding long and signal becomes SELL: exit first, then short.
    When holding short and signal becomes BUY: exit first, then long.

    This ensures clean position transitions without multiple
    overlapping positions.

    Note: This strategy is stateful and should be used with the
    engine's position tracking, not standalone.
    """

    def __init__(
        self,
        predictions: np.ndarray,
        shifted: bool = False,
        name: str = None,
    ):
        """
        Initialize ExitOnReverseStrategy.

        Args:
            predictions: Model class predictions
            shifted: If True, use shifted labels
            name: Optional custom name
        """
        self._base_strategy = DirectionStrategy(predictions, shifted=shifted)
        self._name = name or "ExitOnReverseStrategy"

    @property
    def name(self) -> str:
        return self._name

    def generate_signals(
        self,
        prices: np.ndarray,
        index: Optional[int] = None,
    ) -> SignalOutput:
        """
        Generate signals with exit on reversal.

        The actual exit logic is handled by the engine based on
        current position. This strategy just provides the raw
        direction signals.
        """
        return self._base_strategy.generate_signals(prices, index)

