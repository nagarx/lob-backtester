"""
Tests for ReadabilityHybridStrategy.

Tests verify:
- #PY-71 NaN guard discipline (2026-05-15) — fail-closed on NaN input across
  all 4 gates (agreement, confidence, spread, predicted_returns).

Pre-fix: `value <op> threshold` evaluated False on NaN input (IEEE 754
NaN-comparison invariant), allowing NaN signals to PASS gates silently and
trigger trades on garbage. Per hft-rules §8 fail-closed.
"""

import numpy as np
import pytest

from lobbacktest.strategies.base import Signal
from lobbacktest.strategies.hybrid import (
    ReadabilityHybridConfig,
    ReadabilityHybridStrategy,
)
from lobbacktest.strategies.holding import HorizonAlignedPolicy


def _make_hybrid_data(n: int = 20):
    """Create test data for ReadabilityHybridStrategy with controlled gate inputs."""
    predictions = np.full(n, 2, dtype=np.int32)  # All Up (shifted)
    agreement = np.ones(n, dtype=np.float64)
    confirmation = np.full(n, 0.66, dtype=np.float64)
    predicted_returns = np.full(n, 8.0, dtype=np.float64)
    spreads = np.full(n, 0.8, dtype=np.float64)
    prices = np.linspace(100, 101, n)
    return predictions, agreement, confirmation, predicted_returns, spreads, prices


class TestHybridNaNGuards:
    """Tests for #PY-71 NaN guard discipline (2026-05-15)."""

    def test_nan_agreement_rejected(self):
        """NaN agreement_ratio must REJECT entry (hybrid.py:112)."""
        predictions, agreement, confirmation, predicted_returns, spreads, prices = _make_hybrid_data(n=10)
        agreement[0] = np.nan
        config = ReadabilityHybridConfig(min_agreement=1.0, min_confidence=0.65, min_return_bps=5.0)
        strategy = ReadabilityHybridStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, predicted_returns=predicted_returns,
            spreads=spreads, prices=prices, config=config,
            holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.HOLD, (
            f"NaN agreement should fail-closed; got {output.signals[0]}"
        )

    def test_nan_confidence_rejected(self):
        """NaN confirmation_score must REJECT entry (hybrid.py:114)."""
        predictions, agreement, confirmation, predicted_returns, spreads, prices = _make_hybrid_data(n=10)
        confirmation[0] = np.nan
        config = ReadabilityHybridConfig(min_agreement=1.0, min_confidence=0.65, min_return_bps=5.0)
        strategy = ReadabilityHybridStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, predicted_returns=predicted_returns,
            spreads=spreads, prices=prices, config=config,
            holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.HOLD, (
            f"NaN confidence should fail-closed; got {output.signals[0]}"
        )

    def test_nan_spread_rejected(self):
        """NaN spread must REJECT entry (hybrid.py:117)."""
        predictions, agreement, confirmation, predicted_returns, spreads, prices = _make_hybrid_data(n=10)
        spreads[0] = np.nan
        config = ReadabilityHybridConfig(max_spread_bps=1.0, min_return_bps=5.0)
        strategy = ReadabilityHybridStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, predicted_returns=predicted_returns,
            spreads=spreads, prices=prices, config=config,
            holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.HOLD, (
            f"NaN spread should fail-closed; got {output.signals[0]}"
        )

    def test_nan_predicted_returns_rejected(self):
        """NaN predicted_returns must REJECT entry (hybrid.py:121 — NEW Agent J).

        Sister of FIND-046 (regression.py:99). Was missed by original #PY-71
        scope but added per pre-impl adversarial gate Agent J SHIP-BLOCKER #5.
        """
        predictions, agreement, confirmation, predicted_returns, spreads, prices = _make_hybrid_data(n=10)
        predicted_returns[0] = np.nan
        config = ReadabilityHybridConfig(min_return_bps=5.0)
        strategy = ReadabilityHybridStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, predicted_returns=predicted_returns,
            spreads=spreads, prices=prices, config=config,
            holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.HOLD, (
            f"NaN predicted_returns should fail-closed; got {output.signals[0]}"
        )
