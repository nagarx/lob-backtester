"""
Tests for direction-based strategies.

Tests verify:
- DirectionStrategy signal generation
- ThresholdStrategy confidence filtering
- Shifted vs non-shifted label handling
"""

import numpy as np
import pytest

from lobbacktest.strategies.base import Signal, SignalOutput
from lobbacktest.strategies.direction import (
    DirectionStrategy,
    ThresholdStrategy,
    LABEL_DOWN,
    LABEL_STABLE,
    LABEL_UP,
    SHIFTED_LABEL_DOWN,
    SHIFTED_LABEL_STABLE,
    SHIFTED_LABEL_UP,
)


class TestDirectionStrategy:
    """Tests for DirectionStrategy."""

    def test_basic_signal_generation(self):
        """Test that predictions map correctly to signals."""
        # Up -> BUY, Down -> SELL, Stable -> HOLD
        predictions = np.array([LABEL_UP, LABEL_STABLE, LABEL_DOWN, LABEL_UP, LABEL_STABLE])
        prices = np.array([100.0, 101.0, 100.5, 102.0, 103.0])

        strategy = DirectionStrategy(predictions, shifted=False)
        output = strategy.generate_signals(prices)

        expected = np.array([Signal.BUY, Signal.HOLD, Signal.SELL, Signal.BUY, Signal.HOLD])
        np.testing.assert_array_equal(output.signals, expected)

    def test_shifted_labels(self):
        """Test signal generation with shifted labels (0/1/2)."""
        # Shifted: 0=Down, 1=Stable, 2=Up
        predictions = np.array([
            SHIFTED_LABEL_UP,
            SHIFTED_LABEL_STABLE,
            SHIFTED_LABEL_DOWN,
        ])
        prices = np.array([100.0, 101.0, 100.5])

        strategy = DirectionStrategy(predictions, shifted=True)
        output = strategy.generate_signals(prices)

        expected = np.array([Signal.BUY, Signal.HOLD, Signal.SELL])
        np.testing.assert_array_equal(output.signals, expected)

    def test_online_mode_single_signal(self):
        """Test generating signals one at a time."""
        predictions = np.array([LABEL_UP, LABEL_DOWN, LABEL_STABLE])
        prices = np.array([100.0, 101.0, 102.0])

        strategy = DirectionStrategy(predictions, shifted=False)

        # Get signal at index 0
        output = strategy.generate_signals(prices, index=0)
        assert len(output.signals) == 1
        assert output.signals[0] == Signal.BUY

        # Get signal at index 1
        output = strategy.generate_signals(prices, index=1)
        assert output.signals[0] == Signal.SELL

    def test_all_up_predictions(self):
        """Test all Up predictions generate all BUY signals."""
        predictions = np.array([LABEL_UP, LABEL_UP, LABEL_UP, LABEL_UP])
        prices = np.array([100.0, 101.0, 102.0, 103.0])

        strategy = DirectionStrategy(predictions, shifted=False)
        output = strategy.generate_signals(prices)

        assert all(s == Signal.BUY for s in output.signals)

    def test_all_down_predictions(self):
        """Test all Down predictions generate all SELL signals."""
        predictions = np.array([LABEL_DOWN, LABEL_DOWN, LABEL_DOWN])
        prices = np.array([100.0, 99.0, 98.0])

        strategy = DirectionStrategy(predictions, shifted=False)
        output = strategy.generate_signals(prices)

        assert all(s == Signal.SELL for s in output.signals)

    def test_all_stable_predictions(self):
        """Test all Stable predictions generate all HOLD signals."""
        predictions = np.array([LABEL_STABLE, LABEL_STABLE, LABEL_STABLE])
        prices = np.array([100.0, 100.1, 100.05])

        strategy = DirectionStrategy(predictions, shifted=False)
        output = strategy.generate_signals(prices)

        assert all(s == Signal.HOLD for s in output.signals)

    def test_length_mismatch_returns_hold(self):
        """Test that length mismatch returns all HOLD."""
        predictions = np.array([LABEL_UP, LABEL_DOWN])  # Length 2
        prices = np.array([100.0, 101.0, 102.0, 103.0])  # Length 4

        strategy = DirectionStrategy(predictions, shifted=False)
        output = strategy.generate_signals(prices)

        # Should return HOLD for all (invalid predictions)
        assert len(output.signals) == 4
        assert all(s == Signal.HOLD for s in output.signals)


class TestThresholdStrategy:
    """Tests for ThresholdStrategy."""

    def test_high_confidence_signals_pass(self):
        """Test that high-confidence predictions generate signals."""
        predictions = np.array([SHIFTED_LABEL_UP, SHIFTED_LABEL_DOWN])
        probabilities = np.array([
            [0.1, 0.2, 0.7],  # 70% Up confidence
            [0.8, 0.1, 0.1],  # 80% Down confidence
        ])
        prices = np.array([100.0, 99.0])

        strategy = ThresholdStrategy(
            predictions, probabilities, threshold=0.6, shifted=True
        )
        output = strategy.generate_signals(prices)

        assert output.signals[0] == Signal.BUY  # 0.7 > 0.6
        assert output.signals[1] == Signal.SELL  # 0.8 > 0.6

    def test_low_confidence_signals_filtered(self):
        """Test that low-confidence predictions become HOLD."""
        predictions = np.array([SHIFTED_LABEL_UP, SHIFTED_LABEL_DOWN])
        probabilities = np.array([
            [0.3, 0.4, 0.3],  # 40% max confidence (Stable)
            [0.4, 0.35, 0.25],  # 40% max confidence (Down)
        ])
        prices = np.array([100.0, 99.0])

        strategy = ThresholdStrategy(
            predictions, probabilities, threshold=0.6, shifted=True
        )
        output = strategy.generate_signals(prices)

        # Both should be HOLD (below threshold)
        assert output.signals[0] == Signal.HOLD
        assert output.signals[1] == Signal.HOLD

    def test_stable_predictions_always_hold(self):
        """Test that Stable predictions are HOLD regardless of confidence."""
        predictions = np.array([SHIFTED_LABEL_STABLE])
        probabilities = np.array([[0.1, 0.8, 0.1]])  # High confidence Stable
        prices = np.array([100.0])

        strategy = ThresholdStrategy(
            predictions, probabilities, threshold=0.5, shifted=True
        )
        output = strategy.generate_signals(prices)

        assert output.signals[0] == Signal.HOLD

    def test_confidence_output(self):
        """Test that confidence scores are included in output."""
        predictions = np.array([SHIFTED_LABEL_UP, SHIFTED_LABEL_DOWN])
        probabilities = np.array([
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
        ])
        prices = np.array([100.0, 99.0])

        strategy = ThresholdStrategy(
            predictions, probabilities, threshold=0.5, shifted=True
        )
        output = strategy.generate_signals(prices)

        assert output.confidence is not None
        np.testing.assert_array_almost_equal(output.confidence, [0.7, 0.6])

    def test_threshold_boundary(self):
        """Test behavior at exact threshold boundary."""
        predictions = np.array([SHIFTED_LABEL_UP])
        probabilities = np.array([[0.2, 0.2, 0.6]])  # Exactly at threshold
        prices = np.array([100.0])

        # At threshold should pass
        strategy = ThresholdStrategy(
            predictions, probabilities, threshold=0.6, shifted=True
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.BUY

        # Just below threshold should be filtered
        strategy = ThresholdStrategy(
            predictions, probabilities, threshold=0.601, shifted=True
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.HOLD

    def test_invalid_probabilities_shape(self):
        """Test that wrong probabilities shape raises error."""
        predictions = np.array([SHIFTED_LABEL_UP])
        probabilities = np.array([[0.3, 0.7]])  # Wrong shape (2 instead of 3)

        with pytest.raises(ValueError, match="shape \\(N, 3\\)"):
            ThresholdStrategy(predictions, probabilities, threshold=0.5)

    def test_invalid_threshold(self):
        """Test that threshold outside [0, 1] raises error."""
        predictions = np.array([SHIFTED_LABEL_UP])
        probabilities = np.array([[0.1, 0.2, 0.7]])

        with pytest.raises(ValueError, match="threshold must be in \\[0, 1\\]"):
            ThresholdStrategy(predictions, probabilities, threshold=1.5)


class TestSignalOutput:
    """Tests for SignalOutput dataclass."""

    def test_valid_signal_output(self):
        """Test creating valid SignalOutput."""
        signals = np.array([Signal.BUY, Signal.HOLD, Signal.SELL])
        output = SignalOutput(signals=signals)

        assert len(output) == 3
        assert output.confidence is None

    def test_signal_output_with_confidence(self):
        """Test SignalOutput with confidence scores."""
        signals = np.array([Signal.BUY, Signal.SELL])
        confidence = np.array([0.8, 0.7])

        output = SignalOutput(signals=signals, confidence=confidence)

        assert len(output) == 2
        np.testing.assert_array_equal(output.confidence, [0.8, 0.7])

    def test_signal_output_shape_mismatch(self):
        """Test that mismatched confidence shape raises error."""
        signals = np.array([Signal.BUY, Signal.SELL])
        confidence = np.array([0.8])  # Wrong shape

        with pytest.raises(ValueError, match="confidence shape"):
            SignalOutput(signals=signals, confidence=confidence)

    def test_signal_output_non_1d(self):
        """Test that non-1D signals raise error."""
        signals = np.array([[Signal.BUY], [Signal.SELL]])  # 2D

        with pytest.raises(ValueError, match="must be 1D"):
            SignalOutput(signals=signals)

