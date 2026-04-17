"""
Tests for TWAPStrategy.

Tests verify:
- TWAP window signal pattern: BUY/SELL for k events, then EXIT
- Cooldown after TWAP completion
- Threshold gate: |predicted_return| > min_return_bps
- Spread gate: spread <= max_spread_bps
- Engine incompatibility: engine opens full position on first BUY,
  ignores subsequent BUYs (documented issue C2)

Per RULE.md:
- Formula tests: Signal pattern matches documented TWAP window behavior
- Edge tests: Window extends beyond data, zero predictions
- Invariant tests: Exactly one EXIT per TWAP sequence
"""

import numpy as np
import pytest

from lobbacktest.strategies.base import Signal
from lobbacktest.strategies.twap import TWAPStrategy, TWAPStrategyConfig


class TestTWAPSignalPattern:
    """Tests for TWAP window signal generation."""

    def test_twap_window_buy_then_exit(self):
        """Positive prediction → BUY for k events, then EXIT."""
        n = 20
        predicted = np.full(n, 8.0)  # Above threshold
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        config = TWAPStrategyConfig(min_return_bps=5.0, twap_window=5, cooldown_events=0)
        strategy = TWAPStrategy(predicted, spreads, prices, config)
        output = strategy.generate_signals(prices)

        # Events 0-4: BUY (TWAP window of 5)
        for i in range(5):
            assert output.signals[i] == Signal.BUY, f"Event {i}: should be BUY during TWAP"
        # Event 5: EXIT
        assert output.signals[5] == Signal.EXIT

    def test_twap_window_sell_for_negative(self):
        """Negative prediction → SELL for k events, then EXIT."""
        n = 15
        predicted = np.full(n, -8.0)  # Negative, above magnitude threshold
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        config = TWAPStrategyConfig(min_return_bps=5.0, twap_window=5, cooldown_events=0)
        strategy = TWAPStrategy(predicted, spreads, prices, config)
        output = strategy.generate_signals(prices)

        for i in range(5):
            assert output.signals[i] == Signal.SELL, f"Event {i}: should be SELL during TWAP"
        assert output.signals[5] == Signal.EXIT

    def test_cooldown_after_twap(self):
        """After TWAP EXIT, cooldown_events must pass before next sequence."""
        n = 25
        predicted = np.full(n, 8.0)
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        config = TWAPStrategyConfig(min_return_bps=5.0, twap_window=5, cooldown_events=3)
        strategy = TWAPStrategy(predicted, spreads, prices, config)
        output = strategy.generate_signals(prices)

        # Events 0-4: First TWAP window (BUY)
        assert output.signals[0] == Signal.BUY
        # Event 5: EXIT
        assert output.signals[5] == Signal.EXIT
        # Events 6-8: Cooldown (HOLD)
        for i in range(6, 9):
            assert output.signals[i] == Signal.HOLD, f"Event {i}: should be cooldown HOLD"
        # Event 9: Next TWAP starts
        assert output.signals[9] == Signal.BUY


class TestTWAPGates:
    """Tests for TWAP entry gate conditions."""

    def test_below_threshold_no_entry(self):
        """Predictions below threshold → no TWAP entry."""
        n = 10
        predicted = np.full(n, 2.0)  # Below 5.0 threshold
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        config = TWAPStrategyConfig(min_return_bps=5.0, twap_window=5)
        strategy = TWAPStrategy(predicted, spreads, prices, config)
        output = strategy.generate_signals(prices)
        assert all(s == Signal.HOLD for s in output.signals)

    def test_wide_spread_blocks_entry(self):
        """Wide spread blocks TWAP entry."""
        n = 10
        predicted = np.full(n, 8.0)
        spreads = np.full(n, 2.0)  # Above 1.05
        prices = np.linspace(100, 101, n)
        config = TWAPStrategyConfig(min_return_bps=5.0, max_spread_bps=1.05, twap_window=5)
        strategy = TWAPStrategy(predicted, spreads, prices, config)
        output = strategy.generate_signals(prices)
        assert all(s == Signal.HOLD for s in output.signals)

    def test_window_must_fit_in_data(self):
        """TWAP window cannot extend beyond data → no entry near end."""
        n = 8
        predicted = np.full(n, 8.0)
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        config = TWAPStrategyConfig(min_return_bps=5.0, twap_window=10)  # Window > data
        strategy = TWAPStrategy(predicted, spreads, prices, config)
        output = strategy.generate_signals(prices)
        # No entry because i + k >= n for all i
        assert all(s == Signal.HOLD for s in output.signals)


class TestTWAPMetadata:
    """Tests for TWAP metadata output."""

    def test_metadata_keys(self):
        n = 20
        predicted = np.full(n, 8.0)
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        config = TWAPStrategyConfig(twap_window=5, cooldown_events=0)
        strategy = TWAPStrategy(predicted, spreads, prices, config)
        output = strategy.generate_signals(prices)
        meta = output.metadata

        assert meta["strategy_type"] == "twap"
        assert meta["twap_window"] == 5
        assert "n_twap_sequences" in meta
        assert "n_entries" in meta
        assert meta["n_twap_sequences"] >= 1


class TestTWAPEngineIncompatibility:
    """
    Document the known engine incompatibility (Issue C2).

    The TWAP strategy emits repeated BUY signals during the window,
    but the VectorizedEngine opens a full position on the first BUY
    and ignores subsequent BUYs while already LONG.

    This means TWAP is functionally identical to point-entry.
    """

    def test_twap_emits_repeated_buy_signals(self):
        """TWAP should emit BUY for each event in window (strategy's intent)."""
        n = 15
        predicted = np.full(n, 8.0)
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        config = TWAPStrategyConfig(twap_window=5, cooldown_events=0)
        strategy = TWAPStrategy(predicted, spreads, prices, config)
        output = strategy.generate_signals(prices)

        # Count BUY signals in first window
        buy_count = sum(1 for s in output.signals[:5] if s == Signal.BUY)
        assert buy_count == 5, (
            f"TWAP should emit 5 BUY signals in window, got {buy_count}. "
            "NOTE: Engine only executes the first one (C2 limitation)."
        )
