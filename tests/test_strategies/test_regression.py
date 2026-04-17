"""
Tests for RegressionStrategy.

Tests verify:
- Entry gate: |predicted_return| > threshold AND spread check
- Direction from sign of prediction (positive → BUY, negative → SELL)
- Multi-horizon selection via primary_horizon_idx
- Holding policy integration
- Cooldown behavior
- Metadata correctness

Per RULE.md:
- Formula tests: Entry gate matches |predicted| >= min_return_bps
- Edge tests: Zero predictions, all below threshold, NaN-like scenarios
- Boundary tests: Exactly at threshold
"""

import numpy as np
import pytest

from lobbacktest.strategies.base import Signal
from lobbacktest.strategies.regression import RegressionStrategy, RegressionStrategyConfig
from lobbacktest.strategies.holding import HorizonAlignedPolicy


class TestRegressionEntryGate:
    """Tests for regression entry gate logic."""

    def test_above_threshold_enters(self):
        """Predicted return above threshold → entry."""
        n = 15
        predicted = np.full(n, 8.0)  # 8 bps, above default 5.0
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        config = RegressionStrategyConfig(min_return_bps=5.0)
        policy = HorizonAlignedPolicy(hold_events=5)
        strategy = RegressionStrategy(
            predicted_returns=predicted, spreads=spreads,
            prices=prices, config=config, holding_policy=policy,
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.BUY  # Positive return → BUY

    def test_below_threshold_holds(self):
        """Predicted return below threshold → HOLD (no entry)."""
        n = 10
        predicted = np.full(n, 2.0)  # 2 bps, below default 5.0
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        config = RegressionStrategyConfig(min_return_bps=5.0)
        strategy = RegressionStrategy(
            predicted_returns=predicted, spreads=spreads,
            prices=prices, config=config,
        )
        output = strategy.generate_signals(prices)
        assert all(s == Signal.HOLD for s in output.signals)

    def test_threshold_boundary(self):
        """
        |predicted| exactly at min_return_bps: does NOT pass (strict <).
        Gate formula: abs(predicted) < min_return_bps → reject.
        """
        n = 15
        predicted = np.full(n, 5.0)  # Exactly at threshold
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        config = RegressionStrategyConfig(min_return_bps=5.0)
        strategy = RegressionStrategy(
            predicted_returns=predicted, spreads=spreads,
            prices=prices, config=config, holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        # abs(5.0) < 5.0 is False, so gate passes → enters
        assert output.signals[0] == Signal.BUY

    def test_negative_return_sells(self):
        """Negative predicted return → SELL signal."""
        n = 15
        predicted = np.full(n, -8.0)  # -8 bps
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        config = RegressionStrategyConfig(min_return_bps=5.0)
        strategy = RegressionStrategy(
            predicted_returns=predicted, spreads=spreads,
            prices=prices, config=config, holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.SELL

    def test_wide_spread_blocks_entry(self):
        """Wide spread blocks entry even with strong prediction."""
        n = 10
        predicted = np.full(n, 20.0)  # Strong prediction
        spreads = np.full(n, 2.0)  # Above 1.05 threshold
        prices = np.linspace(100, 101, n)
        config = RegressionStrategyConfig(min_return_bps=5.0, max_spread_bps=1.05)
        strategy = RegressionStrategy(
            predicted_returns=predicted, spreads=spreads,
            prices=prices, config=config,
        )
        output = strategy.generate_signals(prices)
        assert all(s == Signal.HOLD for s in output.signals)


class TestRegressionMultiHorizon:
    """Tests for multi-horizon prediction handling."""

    def test_2d_predictions_uses_primary_horizon(self):
        """With 2D predictions [N, H], uses primary_horizon_idx column."""
        n = 15
        # 3 horizons: H10=low signal, H60=high signal, H300=low signal
        predicted = np.zeros((n, 3))
        predicted[:, 0] = 2.0  # H10: below threshold
        predicted[:, 1] = 10.0  # H60: above threshold
        predicted[:, 2] = 1.0  # H300: below threshold
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        config = RegressionStrategyConfig(min_return_bps=5.0, primary_horizon_idx=1)
        strategy = RegressionStrategy(
            predicted_returns=predicted, spreads=spreads,
            prices=prices, config=config, holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.BUY  # Uses H60 (column 1)

    def test_1d_predictions_work(self):
        """1D predictions [N] are handled correctly."""
        n = 15
        predicted = np.full(n, 8.0)
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        strategy = RegressionStrategy(
            predicted_returns=predicted, spreads=spreads,
            prices=prices, holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.BUY


class TestRegressionHoldingIntegration:
    """Tests for holding policy integration."""

    def test_hold_for_horizon_events_then_exit(self):
        """After entry, hold for hold_events then EXIT."""
        n = 20
        predicted = np.full(n, 8.0)
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        config = RegressionStrategyConfig(min_return_bps=5.0)
        policy = HorizonAlignedPolicy(hold_events=5)
        strategy = RegressionStrategy(
            predicted_returns=predicted, spreads=spreads,
            prices=prices, config=config, holding_policy=policy,
        )
        output = strategy.generate_signals(prices)

        assert output.signals[0] == Signal.BUY
        for i in range(1, 5):
            assert output.signals[i] == Signal.BUY, f"Event {i}: should hold BUY"
        assert output.signals[5] == Signal.EXIT

    def test_cooldown_after_exit(self):
        """Cooldown period prevents immediate re-entry."""
        n = 25
        predicted = np.full(n, 8.0)
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        config = RegressionStrategyConfig(min_return_bps=5.0, cooldown_events=3)
        policy = HorizonAlignedPolicy(hold_events=5)
        strategy = RegressionStrategy(
            predicted_returns=predicted, spreads=spreads,
            prices=prices, config=config, holding_policy=policy,
        )
        output = strategy.generate_signals(prices)

        assert output.signals[0] == Signal.BUY
        assert output.signals[5] == Signal.EXIT
        # Events 6-8: Cooldown
        for i in range(6, 9):
            assert output.signals[i] == Signal.HOLD, f"Event {i}: should be cooldown HOLD"
        # Event 9: Re-entry allowed
        assert output.signals[9] == Signal.BUY


class TestRegressionMetadata:
    """Tests for metadata output."""

    def test_metadata_keys(self):
        n = 15
        predicted = np.full(n, 8.0)
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        strategy = RegressionStrategy(
            predicted_returns=predicted, spreads=spreads,
            prices=prices, holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        meta = output.metadata

        assert meta["strategy_type"] == "regression"
        assert "min_return_bps" in meta
        assert "n_entries" in meta
        assert "n_exits" in meta
        assert "avg_hold_events" in meta
        assert "trade_rate" in meta
        assert 0.0 <= meta["trade_rate"] <= 1.0


class TestRegressionEdgeCases:
    """Edge case tests."""

    def test_zero_predictions_no_entry(self):
        """All predictions = 0 → no entry (below any positive threshold)."""
        n = 10
        predicted = np.zeros(n)
        spreads = np.full(n, 0.8)
        prices = np.linspace(100, 101, n)
        config = RegressionStrategyConfig(min_return_bps=5.0)
        strategy = RegressionStrategy(
            predicted_returns=predicted, spreads=spreads, prices=prices, config=config,
        )
        output = strategy.generate_signals(prices)
        assert all(s == Signal.HOLD for s in output.signals)

    def test_no_spreads_provided(self):
        """Strategy works without spreads (spread gate skipped)."""
        n = 15
        predicted = np.full(n, 8.0)
        prices = np.linspace(100, 101, n)
        strategy = RegressionStrategy(
            predicted_returns=predicted, spreads=None,
            prices=prices, holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.BUY

    def test_confidence_output_is_abs_predictions(self):
        """Confidence output should be |predicted_returns|."""
        n = 10
        predicted = np.array([-8.0, 5.0, -3.0, 10.0, -1.0, 7.0, -6.0, 2.0, 9.0, -4.0])
        prices = np.linspace(100, 101, n)
        strategy = RegressionStrategy(
            predicted_returns=predicted, prices=prices,
        )
        output = strategy.generate_signals(prices)
        np.testing.assert_array_almost_equal(output.confidence, np.abs(predicted))



class TestRegressionDefaults:
    """P4 FIX: Verify corrected default values."""

    def test_default_primary_horizon_idx_is_zero(self):
        """P4: Default should be H10 (index 0), not H60 (index 1)."""
        from lobbacktest.strategies.regression import RegressionStrategyConfig
        config = RegressionStrategyConfig()
        assert config.primary_horizon_idx == 0, (
            f"P4 FIX: primary_horizon_idx should default to 0 (H10), "
            f"got {config.primary_horizon_idx}"
        )

    def test_regression_strategy_uses_label_mapping(self):
        """Phase 2a: Custom label_mapping is used for pred_class derivation."""
        from lobbacktest.labels import SIGNED_MAPPING
        predicted = np.array([5.0, -5.0, 0.0])
        prices = np.array([100.0, 100.0, 100.0])
        strategy = RegressionStrategy(
            predicted_returns=predicted,
            prices=prices,
            label_mapping=SIGNED_MAPPING,
        )
        # With SIGNED_MAPPING: up=1, down=-1
        # Internally, pred_class should use 1 and -1, not 2 and 0
        assert strategy.label_mapping.up == 1
        assert strategy.label_mapping.down == -1
