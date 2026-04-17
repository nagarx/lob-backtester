"""
Tests for core types.

Tests cover:
- Trade: Validation, properties, edge cases
- Position: Validation, properties, invariants
- BacktestResult: Construction, validation, serialization
"""

import numpy as np
import pytest

from lobbacktest.types import (
    BacktestResult,
    Position,
    PositionSide,
    Trade,
    TradeSide,
)


class TestTrade:
    """Tests for Trade dataclass."""

    def test_valid_trade_creation(self):
        """Test creating a valid trade."""
        trade = Trade(
            index=0,
            side=TradeSide.BUY,
            price=100.0,
            size=10.0,
            cost=0.1,
        )
        assert trade.index == 0
        assert trade.side == TradeSide.BUY
        assert trade.price == 100.0
        assert trade.size == 10.0
        assert trade.cost == 0.1

    def test_trade_notional_property(self):
        """Test notional = price × size."""
        trade = Trade(
            index=0,
            side=TradeSide.BUY,
            price=50.0,
            size=20.0,
            cost=0.0,
        )
        assert trade.notional == 1000.0  # 50 * 20

    def test_trade_signed_size_buy(self):
        """Test signed_size is positive for BUY."""
        trade = Trade(
            index=0,
            side=TradeSide.BUY,
            price=100.0,
            size=10.0,
            cost=0.0,
        )
        assert trade.signed_size == 10.0

    def test_trade_signed_size_sell(self):
        """Test signed_size is negative for SELL."""
        trade = Trade(
            index=0,
            side=TradeSide.SELL,
            price=100.0,
            size=10.0,
            cost=0.0,
        )
        assert trade.signed_size == -10.0

    def test_trade_signed_size_flat(self):
        """Test signed_size is zero for FLAT."""
        trade = Trade(
            index=0,
            side=TradeSide.FLAT,
            price=100.0,
            size=10.0,
            cost=0.0,
        )
        assert trade.signed_size == 0.0

    def test_trade_validation_size_zero(self):
        """Test that size=0 raises ValueError."""
        with pytest.raises(ValueError, match="size must be positive"):
            Trade(
                index=0,
                side=TradeSide.BUY,
                price=100.0,
                size=0.0,
                cost=0.0,
            )

    def test_trade_validation_size_negative(self):
        """Test that negative size raises ValueError."""
        with pytest.raises(ValueError, match="size must be positive"):
            Trade(
                index=0,
                side=TradeSide.BUY,
                price=100.0,
                size=-10.0,
                cost=0.0,
            )

    def test_trade_validation_price_zero(self):
        """Test that price=0 raises ValueError."""
        with pytest.raises(ValueError, match="price must be positive"):
            Trade(
                index=0,
                side=TradeSide.BUY,
                price=0.0,
                size=10.0,
                cost=0.0,
            )

    def test_trade_validation_cost_negative(self):
        """Test that negative cost raises ValueError."""
        with pytest.raises(ValueError, match="cost cannot be negative"):
            Trade(
                index=0,
                side=TradeSide.BUY,
                price=100.0,
                size=10.0,
                cost=-0.1,
            )

    def test_trade_with_timestamp(self):
        """Test trade with optional timestamp."""
        trade = Trade(
            index=0,
            side=TradeSide.BUY,
            price=100.0,
            size=10.0,
            cost=0.0,
            timestamp_ns=1234567890,
        )
        assert trade.timestamp_ns == 1234567890


class TestPosition:
    """Tests for Position dataclass."""

    def test_flat_position_creation(self):
        """Test creating a flat position."""
        pos = Position.flat()
        assert pos.is_flat
        assert not pos.is_long
        assert not pos.is_short
        assert pos.size == 0.0
        assert pos.notional == 0.0

    def test_long_position_creation(self):
        """Test creating a long position."""
        pos = Position(
            side=PositionSide.LONG,
            size=100.0,
            entry_price=50.0,
            entry_index=0,
        )
        assert pos.is_long
        assert not pos.is_flat
        assert not pos.is_short
        assert pos.size == 100.0
        assert pos.notional == 5000.0  # 100 * 50

    def test_short_position_creation(self):
        """Test creating a short position."""
        pos = Position(
            side=PositionSide.SHORT,
            size=50.0,
            entry_price=100.0,
            entry_index=5,
        )
        assert pos.is_short
        assert not pos.is_flat
        assert not pos.is_long
        assert pos.notional == 5000.0

    def test_position_entry_cost_default_zero(self):
        """P2 FIX: entry_cost defaults to 0.0 for backward compatibility."""
        pos = Position(
            side=PositionSide.LONG,
            size=100.0,
            entry_price=50.0,
            entry_index=0,
        )
        assert pos.entry_cost == 0.0, (
            f"Default entry_cost should be 0.0, got {pos.entry_cost}"
        )

    def test_position_entry_cost_stored(self):
        """P2 FIX: entry_cost is stored and retrievable."""
        pos = Position(
            side=PositionSide.LONG,
            size=100.0,
            entry_price=50.0,
            entry_index=0,
            entry_cost=1.5,
        )
        assert pos.entry_cost == 1.5, (
            f"entry_cost should be 1.5, got {pos.entry_cost}"
        )

    def test_flat_position_invariant_size_must_be_zero(self):
        """Test that FLAT position with non-zero size raises error."""
        with pytest.raises(ValueError, match="FLAT position must have size=0"):
            Position(
                side=PositionSide.FLAT,
                size=10.0,
                entry_price=100.0,
                entry_index=0,
            )

    def test_long_position_invariant_size_positive(self):
        """Test that LONG position requires positive size."""
        with pytest.raises(ValueError, match="Non-FLAT position must have size>0"):
            Position(
                side=PositionSide.LONG,
                size=0.0,
                entry_price=100.0,
                entry_index=0,
            )

    def test_long_position_invariant_price_positive(self):
        """Test that LONG position requires positive entry_price."""
        with pytest.raises(ValueError, match="entry_price>0"):
            Position(
                side=PositionSide.LONG,
                size=10.0,
                entry_price=0.0,
                entry_index=0,
            )


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def _create_valid_result(self, n=100):
        """Create a valid BacktestResult for testing."""
        equity_curve = np.linspace(100000, 110000, n)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        positions = np.zeros(n)
        prices = np.linspace(100, 105, n)
        predictions = np.zeros(n, dtype=np.int8)

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            positions=positions,
            trades=[],
            trade_pnls=np.array([]),  # No trades
            prices=prices,
            predictions=predictions,
            labels=None,
            metrics={"SharpeRatio": 1.5},
            config_dict={"initial_capital": 100000},
            initial_capital=100000.0,
            final_equity=float(equity_curve[-1]),
            total_trades=0,
            start_index=0,
            end_index=n - 1,
        )

    def test_valid_result_creation(self):
        """Test creating a valid BacktestResult."""
        result = self._create_valid_result()
        assert len(result.equity_curve) == 100
        assert len(result.returns) == 99
        assert result.initial_capital == 100000.0
        assert result.final_equity == 110000.0

    def test_total_return_property(self):
        """Test total_return = (final - initial) / initial."""
        result = self._create_valid_result()
        expected = (110000 - 100000) / 100000
        assert abs(result.total_return - expected) < 1e-10

    def test_total_pnl_property(self):
        """Test total_pnl = final - initial."""
        result = self._create_valid_result()
        assert result.total_pnl == 10000.0

    def test_max_drawdown_property(self):
        """Test max_drawdown calculation."""
        # Create equity curve with known drawdown
        equity = np.array([100, 110, 105, 115, 100, 120])  # Max DD at 100 from 115
        returns = np.diff(equity) / equity[:-1]

        result = BacktestResult(
            equity_curve=equity.astype(float),
            returns=returns,
            positions=np.zeros(6),
            trades=[],
            trade_pnls=np.array([]),
            prices=np.ones(6) * 100,
            predictions=np.zeros(6),
            labels=None,
            metrics={},
            config_dict={},
            initial_capital=100.0,
            final_equity=120.0,
            total_trades=0,
            start_index=0,
            end_index=5,
        )

        # Max drawdown: (115 - 100) / 115 = 0.1304
        expected_dd = (115 - 100) / 115
        assert abs(result.max_drawdown - expected_dd) < 0.001

    def test_validation_length_mismatch_prices(self):
        """Test that mismatched prices length raises error."""
        with pytest.raises(ValueError, match="prices length"):
            BacktestResult(
                equity_curve=np.ones(100),
                returns=np.ones(99),
                positions=np.ones(100),
                trades=[],
                trade_pnls=np.array([]),
                prices=np.ones(50),  # Wrong length
                predictions=np.ones(100),
                labels=None,
                metrics={},
                config_dict={},
                initial_capital=100.0,
                final_equity=100.0,
                total_trades=0,
                start_index=0,
                end_index=99,
            )

    def test_validation_final_equity_mismatch(self):
        """Test that final_equity must match equity_curve[-1]."""
        with pytest.raises(ValueError, match="final_equity"):
            BacktestResult(
                equity_curve=np.ones(100) * 100,
                returns=np.zeros(99),
                positions=np.ones(100),
                trades=[],
                trade_pnls=np.array([]),
                prices=np.ones(100),
                predictions=np.ones(100),
                labels=None,
                metrics={},
                config_dict={},
                initial_capital=100.0,
                final_equity=200.0,  # Doesn't match equity_curve[-1]
                total_trades=0,
                start_index=0,
                end_index=99,
            )

    def test_to_dict_serialization(self):
        """Test that to_dict produces serializable output."""
        result = self._create_valid_result()
        d = result.to_dict()

        assert "equity_curve" in d
        assert "returns" in d
        assert "metrics" in d
        assert "total_return" in d
        assert "max_drawdown" in d

        # Should be JSON-serializable (lists, not numpy arrays)
        import json

        json.dumps(d)  # Should not raise

    def test_summary_output(self):
        """Test that summary() produces valid string."""
        result = self._create_valid_result()
        summary = result.summary()

        assert isinstance(summary, str)
        assert "BACKTEST SUMMARY" in summary
        assert "Initial Capital" in summary
        assert "Final Equity" in summary
        assert "Total Return" in summary

