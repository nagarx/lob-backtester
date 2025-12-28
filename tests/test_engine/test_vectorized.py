"""
Tests for vectorized backtest engine.

Tests verify:
- Basic backtest execution
- Position tracking
- P&L calculation
- Transaction costs
- Edge cases
"""

import numpy as np
import pytest

from lobbacktest.config import BacktestConfig, CostConfig
from lobbacktest.engine.vectorized import Backtester, BacktestData, VectorizedEngine
from lobbacktest.strategies.direction import DirectionStrategy


class TestBacktestData:
    """Tests for BacktestData container."""

    def test_valid_data_creation(self):
        """Test creating valid BacktestData."""
        prices = np.array([100.0, 101.0, 102.0])
        data = BacktestData(prices=prices)

        assert len(data) == 3
        assert data.labels is None

    def test_with_labels(self):
        """Test BacktestData with labels."""
        prices = np.array([100.0, 101.0, 102.0])
        labels = np.array([1, 0, -1])
        data = BacktestData(prices=prices, labels=labels)

        assert len(data) == 3
        np.testing.assert_array_equal(data.labels, labels)

    def test_empty_prices_raises(self):
        """Test that empty prices raise error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            BacktestData(prices=np.array([]))

    def test_non_positive_prices_raises(self):
        """Test that non-positive prices raise error."""
        with pytest.raises(ValueError, match="must be positive"):
            BacktestData(prices=np.array([100.0, -1.0, 102.0]))

    def test_nan_prices_raises(self):
        """Test that NaN prices raise error."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            BacktestData(prices=np.array([100.0, np.nan, 102.0]))


class TestVectorizedEngine:
    """Tests for VectorizedEngine."""

    def _create_simple_backtest(self):
        """Create simple backtest components."""
        prices = np.array([100.0, 101.0, 102.0, 101.0, 100.0])
        predictions = np.array([1, 0, -1, 0, 1])  # Up, Stable, Down, Stable, Up

        config = BacktestConfig(
            initial_capital=10000.0,
            position_size=0.1,  # 10% per trade
            costs=CostConfig(spread_bps=0, slippage_bps=0, commission_per_trade=0),
        )

        data = BacktestData(prices=prices)
        strategy = DirectionStrategy(predictions, shifted=False)

        return config, data, strategy

    def test_basic_backtest_runs(self):
        """Test that basic backtest completes without error."""
        config, data, strategy = self._create_simple_backtest()

        engine = VectorizedEngine(config)
        result = engine.run(data, strategy)

        assert result is not None
        assert len(result.equity_curve) == len(data)
        assert result.initial_capital == 10000.0

    def test_no_trades_on_all_hold(self):
        """Test that all HOLD signals produce no trades."""
        prices = np.array([100.0, 101.0, 102.0, 101.0])
        predictions = np.array([0, 0, 0, 0])  # All Stable

        config = BacktestConfig(initial_capital=10000.0)
        data = BacktestData(prices=prices)
        strategy = DirectionStrategy(predictions, shifted=False)

        engine = VectorizedEngine(config)
        result = engine.run(data, strategy)

        assert result.total_trades == 0
        assert result.final_equity == result.initial_capital

    def test_buy_signal_opens_long(self):
        """Test that BUY signal opens long position."""
        prices = np.array([100.0, 101.0, 102.0])
        predictions = np.array([1, 0, 0])  # Buy first, then hold

        config = BacktestConfig(
            initial_capital=10000.0,
            position_size=0.1,
            costs=CostConfig(spread_bps=0, slippage_bps=0, commission_per_trade=0),
        )
        data = BacktestData(prices=prices)
        strategy = DirectionStrategy(predictions, shifted=False)

        engine = VectorizedEngine(config)
        result = engine.run(data, strategy)

        # Should have at least one trade (the BUY)
        assert result.total_trades >= 1

        # First trade should be BUY
        buy_trades = [t for t in result.trades if t.side.value == 1]
        assert len(buy_trades) >= 1

    def test_sell_signal_opens_short(self):
        """Test that SELL signal opens short position (if allowed)."""
        prices = np.array([100.0, 99.0, 98.0])
        predictions = np.array([-1, 0, 0])  # Sell first, then hold

        config = BacktestConfig(
            initial_capital=10000.0,
            position_size=0.1,
            allow_short=True,
            costs=CostConfig(spread_bps=0, slippage_bps=0, commission_per_trade=0),
        )
        data = BacktestData(prices=prices)
        strategy = DirectionStrategy(predictions, shifted=False)

        engine = VectorizedEngine(config)
        result = engine.run(data, strategy)

        # Should have at least one trade (the SELL)
        assert result.total_trades >= 1

    def test_short_disabled(self):
        """Test that SELL signals are ignored when allow_short=False."""
        prices = np.array([100.0, 99.0, 98.0])
        predictions = np.array([-1, -1, -1])  # All sell

        config = BacktestConfig(
            initial_capital=10000.0,
            allow_short=False,
        )
        data = BacktestData(prices=prices)
        strategy = DirectionStrategy(predictions, shifted=False)

        engine = VectorizedEngine(config)
        result = engine.run(data, strategy)

        # No trades (can't short)
        assert result.total_trades == 0

    def test_transaction_costs_reduce_equity(self):
        """Test that transaction costs reduce final equity."""
        prices = np.array([100.0, 100.0, 100.0])  # Flat prices
        predictions = np.array([1, -1, 1])  # Buy, then sell, then buy

        # With costs
        config_with_costs = BacktestConfig(
            initial_capital=10000.0,
            position_size=0.1,
            costs=CostConfig(spread_bps=10.0, slippage_bps=0, commission_per_trade=0),
        )

        # Without costs
        config_no_costs = BacktestConfig(
            initial_capital=10000.0,
            position_size=0.1,
            costs=CostConfig(spread_bps=0, slippage_bps=0, commission_per_trade=0),
        )

        data = BacktestData(prices=prices)
        strategy = DirectionStrategy(predictions, shifted=False)

        engine_with = VectorizedEngine(config_with_costs)
        engine_without = VectorizedEngine(config_no_costs)

        result_with = engine_with.run(data, strategy)
        result_without = engine_without.run(data, strategy)

        # With costs should have lower equity
        assert result_with.final_equity < result_without.final_equity

    def test_profitable_long_trade(self):
        """Test P&L calculation for profitable long trade."""
        # Price goes up
        prices = np.array([100.0, 110.0, 110.0])
        predictions = np.array([1, -1, 0])  # Buy at 100, sell at 110

        config = BacktestConfig(
            initial_capital=10000.0,
            position_size=1.0,  # 100% position
            costs=CostConfig(spread_bps=0, slippage_bps=0, commission_per_trade=0),
        )
        data = BacktestData(prices=prices)
        strategy = DirectionStrategy(predictions, shifted=False)

        engine = VectorizedEngine(config)
        result = engine.run(data, strategy)

        # Should have profit
        assert result.total_pnl > 0

    def test_losing_long_trade(self):
        """Test P&L calculation for losing long trade."""
        # Price goes down
        prices = np.array([100.0, 90.0, 90.0])
        predictions = np.array([1, -1, 0])  # Buy at 100, sell at 90

        config = BacktestConfig(
            initial_capital=10000.0,
            position_size=1.0,
            costs=CostConfig(spread_bps=0, slippage_bps=0, commission_per_trade=0),
        )
        data = BacktestData(prices=prices)
        strategy = DirectionStrategy(predictions, shifted=False)

        engine = VectorizedEngine(config)
        result = engine.run(data, strategy)

        # Should have loss
        assert result.total_pnl < 0

    def test_equity_curve_length_matches_data(self):
        """Test that equity curve has same length as input data."""
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        predictions = np.array([1, 0, -1, 0, 1])

        config = BacktestConfig(initial_capital=10000.0)
        data = BacktestData(prices=prices)
        strategy = DirectionStrategy(predictions, shifted=False)

        engine = VectorizedEngine(config)
        result = engine.run(data, strategy)

        assert len(result.equity_curve) == 5
        assert len(result.positions) == 5
        assert len(result.returns) == 4  # N-1

    def test_metrics_computed(self):
        """Test that metrics are computed in result."""
        config, data, strategy = self._create_simple_backtest()

        engine = VectorizedEngine(config)
        result = engine.run(data, strategy)

        # Should have standard metrics
        assert "TotalReturn" in result.metrics
        assert "SharpeRatio" in result.metrics
        assert "MaxDrawdown" in result.metrics


class TestBacktester:
    """Tests for Backtester convenience wrapper."""

    def test_backtester_run(self):
        """Test that Backtester.run() works."""
        prices = np.array([100.0, 101.0, 102.0])
        predictions = np.array([1, 0, -1])

        config = BacktestConfig(initial_capital=10000.0)
        data = BacktestData(prices=prices)
        strategy = DirectionStrategy(predictions, shifted=False)

        backtester = Backtester(config)
        result = backtester.run(data, strategy)

        assert result is not None
        assert result.initial_capital == 10000.0

    def test_backtester_run_from_arrays(self):
        """Test convenience method for running from arrays."""
        prices = np.array([100.0, 101.0, 102.0])
        predictions = np.array([1, 0, -1])

        config = BacktestConfig(initial_capital=10000.0)
        backtester = Backtester(config)

        result = backtester.run_from_arrays(prices, predictions, shifted=False)

        assert result is not None
        assert len(result.equity_curve) == 3

