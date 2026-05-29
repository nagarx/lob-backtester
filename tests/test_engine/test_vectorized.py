"""
Tests for vectorized backtest engine.

Tests verify:
- Basic backtest execution
- Position tracking
- P&L calculation
- Transaction costs
- Edge cases
"""

import logging

import numpy as np
import pytest

from lobbacktest.config import BacktestConfig, CostConfig
from lobbacktest.engine.vectorized import Backtester, BacktestData, VectorizedEngine
from lobbacktest.strategies.direction import DirectionStrategy
from lobbacktest.types import TradeSide


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


class TestEnginePnlValueLocked:
    """V-NEW (2026-05-30): VALUE-locked golden P&L tests for VectorizedEngine.

    Pre-existing engine tests assert only the SIGN of P&L (`total_pnl > 0`), so a
    magnitude bug (mis-sizing, single-vs-double cost) would pass silently. These
    lock the EXACT hand-derived P&L on deterministic scenarios chosen to avoid the
    `allow_short` reversal (no SELL signal) and the position-size caps (modest 10%
    position), so the engine's behavior is fully traceable and the expected value
    is unambiguous.

    `@filterwarnings("ignore")` tolerates the orthogonal, out-of-scope warnings
    these minimal scenarios emit (the #PY-263 legacy-annualization
    DeprecationWarning, since these configs set no bin_seconds, and an AnnualReturn
    overflow RuntimeWarning on the tiny equity curve). The asserted quantities
    (total_pnl / final_equity / trade_pnls) are annualization-independent, so
    those warnings cannot affect them.
    """

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_zero_cost_long_pnl_exact(self):
        """BUY 100 sh @ $100 (10% of $100k), engine EOF-closes @ $110, zero cost.
        gross = 100 * (110 - 100) = +$1000 exactly. No SELL → allow_short default
        irrelevant; no sizing cap binds at a 10% position."""
        prices = np.array([100.0, 110.0])
        predictions = np.array([1, 0])  # BUY, HOLD -> engine fabricates EOF close
        config = BacktestConfig(
            initial_capital=100_000.0, position_size=0.1,
            costs=CostConfig(spread_bps=0, slippage_bps=0, commission_per_trade=0),
        )
        result = VectorizedEngine(config).run(
            BacktestData(prices=prices), DirectionStrategy(predictions, shifted=False)
        )
        assert result.total_pnl == pytest.approx(1000.0), (
            f"100sh x $10 move, zero cost = $1000 exactly; got {result.total_pnl}"
        )
        assert result.final_equity == pytest.approx(101000.0)
        assert len(result.trade_pnls) == 1
        assert result.trade_pnls[0] == pytest.approx(1000.0)
        assert result.total_trades == 2  # BUY + fabricated EOF FLAT

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_with_cost_long_pnl_exact(self):
        """Same trade with spread 10 bps: entry cost = 10bps*$10k = $10,
        exit cost = 10bps*$11k = $11 -> trade_pnl = 1000 - 10 - 11 = $979 exactly."""
        prices = np.array([100.0, 110.0])
        predictions = np.array([1, 0])
        config = BacktestConfig(
            initial_capital=100_000.0, position_size=0.1,
            costs=CostConfig(spread_bps=10.0, slippage_bps=0.0),
        )
        result = VectorizedEngine(config).run(
            BacktestData(prices=prices), DirectionStrategy(predictions, shifted=False)
        )
        assert len(result.trade_pnls) == 1
        assert result.trade_pnls[0] == pytest.approx(979.0), (
            f"gross 1000 - entry 10 - exit 11 = 979; got {result.trade_pnls[0]}"
        )

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_multibar_e2e_value_locked(self):
        """Synthetic end-to-end: BUY @ $100, ride down to $90 then up to $110,
        engine EOF-closes @ $110. final_equity = $101000, TotalReturn = 0.01,
        MaxDrawdown = (100000-99000)/100000 = 0.01 (the $90 dip), 2 trades."""
        prices = np.array([100.0, 90.0, 110.0, 110.0])
        predictions = np.array([1, 0, 0, 0])  # BUY then hold; engine EOF-closes
        config = BacktestConfig(
            initial_capital=100_000.0, position_size=0.1,
            costs=CostConfig(spread_bps=0, slippage_bps=0, commission_per_trade=0),
        )
        result = VectorizedEngine(config).run(
            BacktestData(prices=prices), DirectionStrategy(predictions, shifted=False)
        )
        assert result.final_equity == pytest.approx(101000.0)
        assert result.metrics["TotalReturn"] == pytest.approx(0.01)
        assert result.metrics["MaxDrawdown"] == pytest.approx(0.01)
        assert result.total_trades == 2
        assert len(result.trade_pnls) == 1
        assert result.trade_pnls[0] == pytest.approx(1000.0)


class TestTradePnlCosts:
    """P2 FIX: trade_pnls must include BOTH entry and exit costs."""

    def test_trade_pnl_includes_both_entry_and_exit_cost(self):
        """Long trade P&L: gross_pnl - entry_cost - exit_cost.

        BUY at 100 (entry_cost), price rises to 110, SELL (exit_cost).
        gross_pnl = (110 - 100) * size = positive
        trade_pnl = gross_pnl - entry_cost - exit_cost
        """
        prices = np.array([100.0, 110.0, 110.0])
        predictions = np.array([1, -1, 0])  # BUY, SELL, HOLD (signed)

        config = BacktestConfig(
            initial_capital=100_000.0,
            position_size=0.1,
            costs=CostConfig(spread_bps=10.0, slippage_bps=0.0),  # 10 bps = measurable cost
        )
        data = BacktestData(prices=prices)
        strategy = DirectionStrategy(predictions, shifted=False)

        engine = VectorizedEngine(config)
        result = engine.run(data, strategy)

        assert len(result.trade_pnls) >= 1, "Should have at least 1 closed trade"
        # With costs, trade_pnl should be LESS than gross_pnl
        # Gross pnl for a 10% position: 0.1 * 100000 * (110-100)/100 = 1000
        # Entry cost: 10 bps on 10000 notional = 10
        # Exit cost: 10 bps on 11000 notional = 11
        # trade_pnl ≈ 1000 - 10 - 11 = 979
        trade_pnl = result.trade_pnls[0]
        # V-NEW (2026-05-30): value-lock the magnitude (was only `< 1000`). The
        # SELL at idx1 closes the long FIRST (this trade_pnls[0]); gross 1000 -
        # entry cost 10 (10bps*$10k) - exit cost 11 (10bps*$11k) = 979.
        assert trade_pnl == pytest.approx(979.0), (
            f"trade_pnl ({trade_pnl:.2f}) should equal gross 1000 - entry 10 - "
            f"exit 11 = 979 (entry+exit costs both deducted)"
        )
        assert trade_pnl > 0, f"Trade should still be profitable, got {trade_pnl:.2f}"

    def test_trade_pnl_short_includes_entry_cost(self):
        """Short trade P&L includes entry cost (symmetric with long)."""
        prices = np.array([110.0, 100.0, 100.0])
        predictions = np.array([-1, 1, 0])  # SELL (short), BUY (close), HOLD

        config = BacktestConfig(
            initial_capital=100_000.0,
            position_size=0.1,
            allow_short=True,
            costs=CostConfig(spread_bps=10.0, slippage_bps=0.0),
        )
        data = BacktestData(prices=prices)
        strategy = DirectionStrategy(predictions, shifted=False)

        engine = VectorizedEngine(config)
        result = engine.run(data, strategy)

        assert len(result.trade_pnls) >= 1, "Should have closed the short"
        trade_pnl = result.trade_pnls[0]
        # Short profit: (110-100)*size = positive, minus entry+exit costs
        assert trade_pnl > 0, f"Short should be profitable, got {trade_pnl:.2f}"
        # Must be less than gross because costs are deducted
        gross_approx = 0.1 * 100_000 * (110 - 100) / 110
        assert trade_pnl < gross_approx, (
            f"trade_pnl ({trade_pnl:.2f}) should be less than gross ({gross_approx:.2f}) "
            f"due to entry+exit costs"
        )


class TestShortSizingSymmetry:
    """C3 FIX: Shorts and longs must have symmetric sizing and accounting."""

    def test_long_short_sizing_symmetry(self):
        """Same capital → approximately same position size for long and short."""
        # Long: BUY at 100
        prices_long = np.array([100.0, 100.0])
        preds_long = np.array([1, 0])
        # Short: SELL at 100
        prices_short = np.array([100.0, 100.0])
        preds_short = np.array([-1, 0])

        config = BacktestConfig(
            initial_capital=100_000.0,
            position_size=0.1,
            allow_short=True,
            costs=CostConfig(spread_bps=0, slippage_bps=0),
        )

        engine = VectorizedEngine(config)

        result_long = engine.run(
            BacktestData(prices=prices_long),
            DirectionStrategy(preds_long, shifted=False),
        )
        result_short = engine.run(
            BacktestData(prices=prices_short),
            DirectionStrategy(preds_short, shifted=False),
        )

        long_size = result_long.trades[0].size if result_long.trades else 0
        short_size = result_short.trades[0].size if result_short.trades else 0

        assert long_size > 0, "Long should open"
        assert short_size > 0, "Short should open"
        assert long_size == short_size, (
            f"C3 FIX: Long size ({long_size}) should equal short size ({short_size}). "
            f"Both use same capital and position_size fraction."
        )

    def test_round_trip_long_and_short_equal_pnl(self):
        """Long +10% and short profiting from -10% should have same absolute P&L.

        Long: BUY at 100, price goes to 110 → +10% profit
        Short: SELL at 110, price goes to 100 → +10% profit
        With symmetric sizing and costs, P&L should be approximately equal.
        """
        # Long: BUY at 100, SELL at 110
        prices_long = np.array([100.0, 110.0, 110.0])
        preds_long = np.array([1, -1, 0])

        # Short: SELL at 110, BUY at 100
        prices_short = np.array([110.0, 100.0, 100.0])
        preds_short = np.array([-1, 1, 0])

        config = BacktestConfig(
            initial_capital=100_000.0,
            position_size=0.1,
            allow_short=True,
            costs=CostConfig(spread_bps=0, slippage_bps=0),
        )

        engine = VectorizedEngine(config)

        result_long = engine.run(
            BacktestData(prices=prices_long),
            DirectionStrategy(preds_long, shifted=False),
        )
        result_short = engine.run(
            BacktestData(prices=prices_short),
            DirectionStrategy(preds_short, shifted=False),
        )

        long_pnl = result_long.trade_pnls[0] if len(result_long.trade_pnls) > 0 else 0
        short_pnl = result_short.trade_pnls[0] if len(result_short.trade_pnls) > 0 else 0

        assert long_pnl > 0, f"Long should be profitable, got {long_pnl:.2f}"
        assert short_pnl > 0, f"Short should be profitable, got {short_pnl:.2f}"

        # Allow small difference due to different notionals at entry
        ratio = long_pnl / short_pnl if short_pnl > 0 else float('inf')
        assert 0.85 < ratio < 1.15, (
            f"C3 FIX: Long P&L ({long_pnl:.2f}) and short P&L ({short_pnl:.2f}) "
            f"should be approximately equal (ratio={ratio:.3f})"
        )


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


class TestEndOfDataAutoClose:
    """FIND-001 lock tests: engine auto-close on EOF open position must emit Trade(FLAT)."""

    def test_end_of_data_auto_close_emits_trade(self, caplog):
        """FIND-001 lock test: auto-close on open position at EOF must emit Trade(FLAT) + WARN log.

        Pre-2026-05-14: auto-close at vectorized.py:436-442 appended trade_pnls but skipped
        trades.append → ZeroDtePnLTransformer silently dropped final round-trip via the silent
        break at zero_dte.py:269.

        See DESIGN_CLUSTER_D1_E_2026_05_14.md §3.1 + VALIDATION_FINDINGS_2026_05_14.md FIND-001.
        """
        # 3-bar always-in BUY strategy: open at bar 0, hold, never close via signal.
        # Engine must auto-close at bar n - 1 and emit Trade(side=FLAT).
        prices = np.array([100.0, 101.0, 102.0])
        predictions = np.array([1, 1, 1])  # BUY/BUY/BUY (unshifted mapping {-1, 0, 1})

        config = BacktestConfig(
            initial_capital=10_000.0,
            position_size=0.1,  # 10% per trade — well within max_position default
            costs=CostConfig(spread_bps=0.0, slippage_bps=0.0, commission_per_trade=0.0),
        )
        data = BacktestData(prices=prices)
        strategy = DirectionStrategy(predictions, shifted=False)

        engine = VectorizedEngine(config)

        with caplog.at_level(logging.WARNING, logger="lobbacktest.engine.vectorized"):
            result = engine.run(data, strategy)

        # FIND-001: trades has 2 entries (1 BUY entry + 1 FLAT auto-close), not 1
        assert len(result.trades) == 2, (
            f"FIND-001: expected 2 trades (BUY entry + FLAT auto-close); got "
            f"{len(result.trades)}: {[(t.index, t.side.name) for t in result.trades]}"
        )

        # Auto-close emits canonical FLAT side at the last bar (n - 1)
        assert result.trades[-1].side == TradeSide.FLAT
        assert result.trades[-1].price == 102.0
        assert result.trades[-1].index == 2  # n - 1 = len(prices) - 1

        # WARN log emitted with required context (hft-rules §8 observability)
        assert any(
            "fabricated end-of-data close" in record.message for record in caplog.records
        ), "FIND-001 fix must emit WARN log per hft-rules §8 observability"

        # FIND-002: __post_init__ invariant satisfied (1 FLAT trade == 1 trade_pnl).
        # If invariant fails, result construction would have raised before this line.
        assert len(result.trade_pnls) == 1

