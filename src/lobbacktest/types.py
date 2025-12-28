"""
Core types for LOB-Backtester.

This module defines the fundamental data structures used throughout the library:
- Trade: A single executed trade
- Position: Current position state
- BacktestResult: Complete backtest output

All types are immutable dataclasses with clear documentation.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np


class TradeSide(IntEnum):
    """
    Side of a trade execution.

    Values:
        SELL (-1): Sell/Short entry
        FLAT (0): Exit position (close)
        BUY (1): Buy/Long entry
    """

    SELL = -1
    FLAT = 0
    BUY = 1


class PositionSide(IntEnum):
    """
    Current position direction.

    Values:
        SHORT (-1): Short position
        FLAT (0): No position
        LONG (1): Long position
    """

    SHORT = -1
    FLAT = 0
    LONG = 1


@dataclass(frozen=True)
class Trade:
    """
    A single executed trade.

    Attributes:
        index: Sequence index when trade occurred (0-based)
        side: Trade direction (BUY/SELL/FLAT)
        price: Execution price (USD)
        size: Number of shares traded
        cost: Total transaction cost (spread + slippage + commission)
        timestamp_ns: Optional timestamp in nanoseconds since epoch

    Invariants:
        - size > 0 for all trades
        - price > 0 for all trades
        - cost >= 0 (transaction costs cannot be negative)
    """

    index: int
    side: TradeSide
    price: float
    size: float
    cost: float
    timestamp_ns: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate trade invariants."""
        if self.size <= 0:
            raise ValueError(f"Trade size must be positive, got {self.size}")
        if self.price <= 0:
            raise ValueError(f"Trade price must be positive, got {self.price}")
        if self.cost < 0:
            raise ValueError(f"Trade cost cannot be negative, got {self.cost}")

    @property
    def notional(self) -> float:
        """Total trade value before costs: price × size."""
        return self.price * self.size

    @property
    def signed_size(self) -> float:
        """Size with direction: positive for BUY, negative for SELL."""
        return self.size * int(self.side) if self.side != TradeSide.FLAT else 0.0


@dataclass(frozen=True)
class Position:
    """
    Position state at a point in time.

    Attributes:
        side: Position direction (LONG/SHORT/FLAT)
        size: Number of shares held (always positive)
        entry_price: Average entry price
        entry_index: Sequence index when position was opened
        unrealized_pnl: Current unrealized P&L based on mark price

    Invariants:
        - size == 0 if and only if side == FLAT
        - size > 0 for LONG/SHORT positions
        - entry_price > 0 for non-FLAT positions
    """

    side: PositionSide
    size: float
    entry_price: float
    entry_index: int
    unrealized_pnl: float = 0.0

    def __post_init__(self) -> None:
        """Validate position invariants."""
        if self.side == PositionSide.FLAT:
            if self.size != 0:
                raise ValueError(f"FLAT position must have size=0, got {self.size}")
        else:
            if self.size <= 0:
                raise ValueError(f"Non-FLAT position must have size>0, got {self.size}")
            if self.entry_price <= 0:
                raise ValueError(
                    f"Non-FLAT position must have entry_price>0, got {self.entry_price}"
                )

    @classmethod
    def flat(cls) -> "Position":
        """Create a flat (no position) state."""
        return cls(
            side=PositionSide.FLAT,
            size=0.0,
            entry_price=0.0,
            entry_index=-1,
            unrealized_pnl=0.0,
        )

    @property
    def is_flat(self) -> bool:
        """True if no position is held."""
        return self.side == PositionSide.FLAT

    @property
    def is_long(self) -> bool:
        """True if holding a long position."""
        return self.side == PositionSide.LONG

    @property
    def is_short(self) -> bool:
        """True if holding a short position."""
        return self.side == PositionSide.SHORT

    @property
    def notional(self) -> float:
        """Current position value: entry_price × size."""
        return self.entry_price * self.size if not self.is_flat else 0.0


@dataclass
class BacktestResult:
    """
    Complete backtest output.

    Contains all information needed for analysis:
    - Equity curve over time
    - All executed trades with P&L
    - Position history
    - Computed metrics
    - Configuration used

    Attributes:
        equity_curve: Equity value at each time step (shape: N)
        returns: Per-period returns (shape: N-1)
        positions: Position size at each step (+size long, -size short, 0 flat)
        trades: List of all executed trades (opens and closes)
        trade_pnls: P&L per closing trade after costs (shape: num_round_trips)
        prices: Price series used for backtest (shape: N)
        predictions: Model predictions used (shape: N)
        labels: True labels if available (shape: N)
        metrics: Dict of computed metrics
        config_dict: Configuration used (serializable)
        initial_capital: Starting capital
        final_equity: Ending equity value
        total_trades: Number of trades executed (opens + closes)
        start_index: First valid index in the data
        end_index: Last valid index in the data

    Invariants:
        - len(equity_curve) == len(prices) == len(positions)
        - len(returns) == len(equity_curve) - 1
        - final_equity == equity_curve[-1]
        - total_trades == len(trades)
        - len(trade_pnls) == number of round-trip trades (closes)
    """

    equity_curve: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    trades: List[Trade]
    trade_pnls: np.ndarray  # P&L per closing trade (after costs)
    prices: np.ndarray
    predictions: np.ndarray
    labels: Optional[np.ndarray]
    metrics: Dict[str, float]
    config_dict: Dict[str, any]
    initial_capital: float
    final_equity: float
    total_trades: int
    start_index: int
    end_index: int

    def __post_init__(self) -> None:
        """Validate result invariants."""
        n = len(self.equity_curve)
        if n == 0:
            raise ValueError("equity_curve cannot be empty")
        if len(self.prices) != n:
            raise ValueError(
                f"prices length {len(self.prices)} != equity_curve length {n}"
            )
        if len(self.positions) != n:
            raise ValueError(
                f"positions length {len(self.positions)} != equity_curve length {n}"
            )
        if len(self.returns) != n - 1:
            raise ValueError(
                f"returns length {len(self.returns)} != equity_curve length - 1 ({n - 1})"
            )
        if abs(self.final_equity - self.equity_curve[-1]) > 1e-10:
            raise ValueError(
                f"final_equity {self.final_equity} != equity_curve[-1] {self.equity_curve[-1]}"
            )
        if self.total_trades != len(self.trades):
            raise ValueError(
                f"total_trades {self.total_trades} != len(trades) {len(self.trades)}"
            )

    @property
    def total_return(self) -> float:
        """Total return: (final - initial) / initial."""
        if self.initial_capital == 0:
            return 0.0
        return (self.final_equity - self.initial_capital) / self.initial_capital

    @property
    def total_pnl(self) -> float:
        """Total profit/loss: final - initial."""
        return self.final_equity - self.initial_capital

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a fraction (0 to 1)."""
        if len(self.equity_curve) == 0:
            return 0.0
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (peak - self.equity_curve) / peak
        # Handle division by zero when peak is 0
        drawdown = np.where(np.isfinite(drawdown), drawdown, 0.0)
        return float(np.max(drawdown))

    @property
    def n_winning_trades(self) -> int:
        """Number of trades with positive P&L."""
        if len(self.trade_pnls) == 0:
            return 0
        return int(np.sum(self.trade_pnls > 0))

    @property
    def n_losing_trades(self) -> int:
        """Number of trades with negative P&L."""
        if len(self.trade_pnls) == 0:
            return 0
        return int(np.sum(self.trade_pnls < 0))

    def summary(self) -> str:
        """
        Generate a formatted summary of backtest results.

        Returns:
            Multi-line string with key metrics
        """
        lines = [
            "=" * 60,
            "BACKTEST SUMMARY",
            "=" * 60,
            f"Initial Capital:     ${self.initial_capital:,.2f}",
            f"Final Equity:        ${self.final_equity:,.2f}",
            f"Total Return:        {self.total_return * 100:+.2f}%",
            f"Total P&L:           ${self.total_pnl:+,.2f}",
            "-" * 60,
            f"Total Trades:        {self.total_trades}",
            f"  Winning:           {self.n_winning_trades}",
            f"  Losing:            {self.n_losing_trades}",
            f"Max Drawdown:        {self.max_drawdown * 100:.2f}%",
            "-" * 60,
            "METRICS:",
        ]
        for name, value in sorted(self.metrics.items()):
            if isinstance(value, float):
                lines.append(f"  {name:20s} {value:+.4f}")
            else:
                lines.append(f"  {name:20s} {value}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, any]:
        """
        Convert result to a dictionary for serialization.

        Note: numpy arrays are converted to lists.
        """
        return {
            "equity_curve": self.equity_curve.tolist(),
            "returns": self.returns.tolist(),
            "positions": self.positions.tolist(),
            "trades": [
                {
                    "index": t.index,
                    "side": int(t.side),
                    "price": t.price,
                    "size": t.size,
                    "cost": t.cost,
                }
                for t in self.trades
            ],
            "trade_pnls": self.trade_pnls.tolist(),
            "metrics": self.metrics,
            "config": self.config_dict,
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "total_trades": self.total_trades,
            "n_winning_trades": self.n_winning_trades,
            "n_losing_trades": self.n_losing_trades,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
        }

