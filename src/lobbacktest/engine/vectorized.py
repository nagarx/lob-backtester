"""
Backtest engine with per-sample position tracking.

This engine executes backtests using a per-sample loop for explicit position
tracking, with numpy-based metric computation. It assumes instant fill at
the current price (no queue simulation).

Note: Module is named 'vectorized.py' for historical reasons.
The main engine loop is a Python for-loop, not vectorized.

Design Philosophy:
- Position tracking is explicit and auditable (per-sample loop)
- Transaction costs are modeled explicitly (entry + exit costs)
- Short and long positions have symmetric sizing and accounting
- Results include all information needed for analysis
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from lobbacktest.config import BacktestConfig
from lobbacktest.metrics.base import Metric
from lobbacktest.metrics.prediction import DirectionalAccuracy, SignalRate
from lobbacktest.metrics.returns import AnnualReturn, TotalReturn
from lobbacktest.metrics.risk import CalmarRatio, MaxDrawdown, SharpeRatio, SortinoRatio
from lobbacktest.metrics.trading import (
    AverageLoss,
    AverageWin,
    Expectancy,
    PayoffRatio,
    ProfitFactor,
    WinRate,
)
from lobbacktest.strategies.base import Signal, SignalOutput, Strategy
from lobbacktest.types import BacktestResult, Position, PositionSide, Trade, TradeSide


@dataclass
class BacktestData:
    """
    Data container for backtest input.

    Attributes:
        prices: Mid-price series (shape: N)
        labels: True labels if available (shape: N)
        timestamps_ns: Optional timestamps in nanoseconds (shape: N)
        predictions: Model predictions (shape: N), 0=Down, 1=Stable, 2=Up
        spreads: Bid-ask spread in bps (shape: N)
        agreement_ratio: HMHP cross-horizon agreement (shape: N), in [0.333, 1.0]
        confirmation_score: HMHP decoder confidence (shape: N), in [0, 0.667]
    """

    prices: np.ndarray
    labels: Optional[np.ndarray] = None
    timestamps_ns: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    spreads: Optional[np.ndarray] = None
    agreement_ratio: Optional[np.ndarray] = None
    confirmation_score: Optional[np.ndarray] = None
    predicted_returns: Optional[np.ndarray] = None
    regression_labels: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate data."""
        if self.prices.ndim != 1:
            raise ValueError(f"prices must be 1D, got shape {self.prices.shape}")
        if len(self.prices) == 0:
            raise ValueError("prices cannot be empty")
        if not np.all(np.isfinite(self.prices)):
            raise ValueError("prices contains NaN or Inf values")
        if np.any(self.prices <= 0):
            raise ValueError("prices must be positive")

    def __len__(self) -> int:
        return len(self.prices)

    @classmethod
    def from_signal_dir(
        cls,
        signal_dir: str,
        *,
        validate: bool = True,
    ) -> "BacktestData":
        """Load BacktestData from a directory of exported signal arrays.

        Loads .npy files produced by the trainer's signal export scripts
        (export_hmhp_signals.py, export_regression_signals.py, etc.).

        When validate=True (default), reads signal_metadata.json and checks:
        - Required files exist (ContractError if missing)
        - All arrays have aligned first dimension (ContractError if mismatched)
        - No NaN/Inf in required arrays (ContractError if found)
        - Value ranges are sensible (warnings for anomalies)

        Args:
            signal_dir: Path to directory containing .npy signal arrays.
            validate: If True, validate signal contract at load time.
                Set to False for legacy code or manual testing.

        Returns:
            BacktestData populated with all available arrays.

        Raises:
            ContractError: If validation=True and critical contract
                violation detected (missing files, shape mismatch, etc.).
        """
        d = Path(signal_dir)

        # Validate signal contract before loading
        if validate:
            from lobbacktest.data.signal_manifest import SignalManifest

            manifest = SignalManifest.from_signal_dir(d)
            warnings = manifest.validate(d)
            for w in warnings:
                print(f"  ⚠️  Signal validation: {w}")

        prices = np.load(d / "prices.npy")
        labels = np.load(d / "labels.npy") if (d / "labels.npy").exists() else None
        predictions = np.load(d / "predictions.npy") if (d / "predictions.npy").exists() else None
        spreads = np.load(d / "spreads.npy") if (d / "spreads.npy").exists() else None
        agreement = np.load(d / "agreement_ratio.npy") if (d / "agreement_ratio.npy").exists() else None
        confirmation = np.load(d / "confirmation_score.npy") if (d / "confirmation_score.npy").exists() else None
        # Prefer calibrated predictions if available (E6+)
        if (d / "calibrated_returns.npy").exists():
            predicted_returns = np.load(d / "calibrated_returns.npy")
        elif (d / "predicted_returns.npy").exists():
            predicted_returns = np.load(d / "predicted_returns.npy")
        else:
            predicted_returns = None
        regression_labels = np.load(d / "regression_labels.npy") if (d / "regression_labels.npy").exists() else None

        return cls(
            prices=prices,
            labels=labels,
            predictions=predictions,
            spreads=spreads,
            agreement_ratio=agreement,
            confirmation_score=confirmation,
            predicted_returns=predicted_returns,
            regression_labels=regression_labels,
        )


class VectorizedEngine:
    """
    Numpy-based vectorized backtest engine.

    This engine:
    1. Generates signals from strategy
    2. Simulates position changes
    3. Computes P&L and equity curve
    4. Calculates performance metrics

    Assumptions:
    - Instant fill at current price (no slippage beyond configured)
    - No partial fills
    - Single position at a time
    - Position size is fixed fraction of capital
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize the vectorized engine.

        Args:
            config: Backtest configuration
        """
        self.config = config

    def run(
        self,
        data: BacktestData,
        strategy: Strategy,
        metrics: Optional[List[Metric]] = None,
    ) -> BacktestResult:
        """
        Run the backtest.

        Args:
            data: Price and label data
            strategy: Trading strategy
            metrics: Optional list of metrics to compute

        Returns:
            BacktestResult with complete backtest output
        """
        n = len(data)
        prices = data.prices

        # Generate signals
        signal_output = strategy.generate_signals(prices)
        signals = signal_output.signals

        # Initialize tracking arrays
        positions = np.zeros(n, dtype=np.float64)  # +1 long, -1 short, 0 flat
        equity = np.zeros(n, dtype=np.float64)
        equity[0] = self.config.initial_capital

        # Track trades
        trades: List[Trade] = []
        trade_pnls: List[float] = []

        # Current state
        current_position = Position.flat()
        cash = self.config.initial_capital

        # Position size in shares (will be computed per trade)
        # We use fixed fraction of capital for each trade

        # Process each time step
        for i in range(n):
            price = prices[i]
            signal = signals[i]

            # Update unrealized P&L for current position
            if not current_position.is_flat:
                if current_position.is_long:
                    unrealized = (price - current_position.entry_price) * current_position.size
                else:  # Short
                    unrealized = (current_position.entry_price - price) * current_position.size
            else:
                unrealized = 0.0

            # Record current position
            if current_position.is_long:
                positions[i] = current_position.size
            elif current_position.is_short:
                positions[i] = -current_position.size
            else:
                positions[i] = 0.0

            # Process signal
            if signal == Signal.BUY:
                if current_position.is_short:
                    # Close short position first
                    cash_flow, cost, pnl = self._close_position(current_position, price)
                    cash += cash_flow - cost
                    # P2 FIX: Include BOTH entry and exit costs in trade_pnls
                    trade_pnls.append(pnl - cost - current_position.entry_cost)
                    trades.append(
                        Trade(
                            index=i,
                            side=TradeSide.FLAT,
                            price=price,
                            size=current_position.size,
                            cost=cost,
                        )
                    )
                    current_position = Position.flat()

                if current_position.is_flat:
                    # Open long position
                    size = self._compute_position_size(cash, price)
                    if size > 0:
                        position_value = size * price
                        cost = self.config.costs.compute_cost(position_value)
                        # Deduct BOTH position value AND cost from cash
                        # (we're "buying" shares, so cash decreases)
                        cash -= (position_value + cost)
                        current_position = Position(
                            side=PositionSide.LONG,
                            size=size,
                            entry_price=price,
                            entry_index=i,
                            entry_cost=cost,  # P2 FIX: Store entry cost for trade_pnls
                        )
                        trades.append(
                            Trade(
                                index=i,
                                side=TradeSide.BUY,
                                price=price,
                                size=size,
                                cost=cost,
                            )
                        )

            elif signal == Signal.SELL:
                if not self.config.allow_short and current_position.is_flat:
                    # Can't short, skip
                    pass
                else:
                    if current_position.is_long:
                        # Close long position first
                        cash_flow, cost, pnl = self._close_position(current_position, price)
                        cash += cash_flow - cost
                        # P2 FIX: Include BOTH entry and exit costs in trade_pnls
                        trade_pnls.append(pnl - cost - current_position.entry_cost)
                        trades.append(
                            Trade(
                                index=i,
                                side=TradeSide.FLAT,
                                price=price,
                                size=current_position.size,
                                cost=cost,
                            )
                        )
                        current_position = Position.flat()

                    if current_position.is_flat and self.config.allow_short:
                        # Open short position
                        # C3 FIX: Symmetric with longs — deduct BOTH position_value AND cost
                        # Position value acts as margin collateral for the short
                        size = self._compute_position_size(cash, price)
                        if size > 0:
                            position_value = size * price
                            cost = self.config.costs.compute_cost(position_value)
                            # C3 FIX: Deduct position_value as margin + cost (same as longs)
                            cash -= (position_value + cost)
                            current_position = Position(
                                side=PositionSide.SHORT,
                                size=size,
                                entry_price=price,
                                entry_index=i,
                                entry_cost=cost,  # P2 FIX: Store entry cost for trade_pnls
                            )
                            trades.append(
                                Trade(
                                    index=i,
                                    side=TradeSide.SELL,
                                    price=price,
                                    size=size,
                                    cost=cost,
                                )
                            )

            elif signal == Signal.EXIT:
                if not current_position.is_flat:
                    cash_flow, cost, pnl = self._close_position(current_position, price)
                    cash += cash_flow - cost
                    # P2 FIX: Include BOTH entry and exit costs in trade_pnls
                    trade_pnls.append(pnl - cost - current_position.entry_cost)
                    trades.append(
                        Trade(
                            index=i,
                            side=TradeSide.FLAT,
                            price=price,
                            size=current_position.size,
                            cost=cost,
                        )
                    )
                    current_position = Position.flat()

            # Update equity: cash + position value
            # For long: equity = cash + current_market_value
            # For short: equity = cash + (entry_value - current_value) = cash + unrealized_pnl
            if not current_position.is_flat:
                if current_position.is_long:
                    # Long position: we own shares, value = current price * size
                    # Also we paid entry_price * size from cash, so:
                    # equity = cash + current_price * size
                    current_value = price * current_position.size
                    equity[i] = cash + current_value
                else:
                    # Short position: C3 FIX — margin (entry_price * size) deducted at entry.
                    # Equity = cash + margin_held + unrealized_pnl
                    #        = cash + entry_price * size + (entry_price - current_price) * size
                    #        = cash + entry_price * size * 2 - current_price * size
                    # Simplified: equity = cash + margin + pnl
                    margin = current_position.entry_price * current_position.size
                    unrealized = (current_position.entry_price - price) * current_position.size
                    equity[i] = cash + margin + unrealized
            else:
                equity[i] = cash

        # Close any remaining position at end
        if not current_position.is_flat:
            final_price = prices[-1]
            cash_flow, cost, pnl = self._close_position(current_position, final_price)
            cash += cash_flow - cost
            # P2 FIX: Include entry cost in trade_pnls
            trade_pnls.append(pnl - cost - current_position.entry_cost)
            equity[-1] = cash

        # Compute returns
        returns = np.diff(equity) / equity[:-1]
        # Handle division by zero
        returns = np.where(np.isfinite(returns), returns, 0.0)

        # Compute metrics
        computed_metrics = self._compute_metrics(
            returns=returns,
            equity_curve=equity,
            trade_pnls=np.array(trade_pnls),
            predictions=signal_output.signals,
            labels=data.labels,
            metrics=metrics,
        )

        return BacktestResult(
            equity_curve=equity,
            returns=returns,
            positions=positions,
            trades=trades,
            trade_pnls=np.array(trade_pnls),
            prices=prices,
            predictions=signal_output.signals,
            labels=data.labels,
            metrics=computed_metrics,
            config_dict=self.config.to_dict(),
            initial_capital=self.config.initial_capital,
            final_equity=float(equity[-1]),
            total_trades=len(trades),
            start_index=0,
            end_index=n - 1,
        )

    def _compute_position_size(self, capital: float, price: float) -> float:
        """
        Compute position size in shares.

        Position sizing uses a fixed fraction of capital, with multiple safeguards:
        1. Position value cannot exceed max_position * initial_capital
        2. Position value cannot exceed available capital (no leverage)
        3. Number of shares is capped to prevent catastrophic short losses

        The share cap is critical: when price is very low, value-based sizing
        would result in huge share counts. If price then rises 100x, a short
        position would lose 100x the value.

        Args:
            capital: Available capital
            price: Current price

        Returns:
            Number of shares to trade
        """
        if capital <= 0 or price <= 0:
            return 0.0

        # Compute target position value as fraction of CURRENT capital
        target_value = capital * self.config.position_size

        # Cap 1: Position value cannot exceed max_position * INITIAL capital
        max_value = self.config.initial_capital * self.config.max_position
        position_value = min(target_value, max_value)

        # Cap 2: Position value cannot exceed available capital (no leverage)
        position_value = min(position_value, capital * 0.95)  # Keep 5% buffer

        # Convert to shares
        size = position_value / price

        # Cap 3: CRITICAL - limit shares to prevent catastrophic short losses
        # Max shares = max_position * initial_capital / reference_price
        # where reference_price is a "reasonable" price estimate
        # We use the larger of current price and the initial capital / 1000
        # This means: if we started with $100k, we assume prices are roughly $100+
        # so max shares ≈ max_position * 1000 = 200 shares at max_position=0.2
        reference_price = max(price, self.config.initial_capital / 1000)
        max_shares = (self.config.initial_capital * self.config.max_position) / reference_price
        size = min(size, max_shares)

        return max(0.0, size)

    def _close_position(
        self,
        position: Position,
        price: float,
    ) -> Tuple[float, float, float]:
        """
        Close a position and compute proceeds (for longs) or settlement (for shorts).

        For LONG positions:
            - We sell shares at current price
            - Proceeds = price * size (the full value we receive)
            - P&L = (price - entry_price) * size
            
        For SHORT positions:
            - We buy back shares at current price to close
            - P&L = (entry_price - price) * size

        Args:
            position: Position to close
            price: Closing price

        Returns:
            (cash_flow, cost, pnl) tuple where:
            - cash_flow: Amount to add to cash (positive for long sells)
            - cost: Transaction cost (always positive)
            - pnl: Actual profit/loss (price difference * size, before costs)
        """
        if position.is_flat:
            return 0.0, 0.0, 0.0

        cost = self.config.costs.compute_cost(position.size * price)

        if position.is_long:
            # Selling shares: receive full proceeds (return position_value + P&L)
            cash_flow = price * position.size
            # P&L = (exit - entry) * size
            pnl = (price - position.entry_price) * position.size
        else:  # Short
            # C3 FIX: Since we deducted position_value as margin at entry,
            # we now return margin + P&L at close.
            # P&L = (entry - exit) * size (positive when price drops)
            pnl = (position.entry_price - price) * position.size
            # Return margin (entry_price * size) + P&L
            cash_flow = position.entry_price * position.size + pnl

        return cash_flow, cost, pnl

    def _compute_metrics(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
        trade_pnls: np.ndarray,
        predictions: np.ndarray,
        labels: Optional[np.ndarray],
        metrics: Optional[List[Metric]],
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            returns: Per-period returns
            equity_curve: Equity values
            trade_pnls: P&L per trade
            predictions: Strategy signals
            labels: True labels (if available)
            metrics: Optional list of custom metrics

        Returns:
            Dict of metric name to value
        """
        # Build typed context (backward compatible with dict access)
        from lobbacktest.context import BacktestContext

        context = BacktestContext(
            equity_curve=equity_curve,
            trade_pnls=trade_pnls,
            predictions=predictions,
            labels=labels,
            initial_capital=self.config.initial_capital,
            trading_days_per_year=self.config.trading_days_per_year,
            periods_per_day=self.config.periods_per_day,
            annualization_factor=self.config.annualization_factor,
        )

        # Default metrics if none provided
        if metrics is None:
            metrics = [
                TotalReturn(),
                AnnualReturn(
                    trading_days_per_year=self.config.trading_days_per_year,
                    periods_per_day=self.config.periods_per_day,
                ),
                SharpeRatio(
                    trading_days_per_year=self.config.trading_days_per_year,
                    periods_per_day=self.config.periods_per_day,
                ),
                SortinoRatio(
                    trading_days_per_year=self.config.trading_days_per_year,
                    periods_per_day=self.config.periods_per_day,
                ),
                MaxDrawdown(),
                CalmarRatio(
                    trading_days_per_year=self.config.trading_days_per_year,
                    periods_per_day=self.config.periods_per_day,
                ),
                WinRate(),
                ProfitFactor(),
                AverageWin(),
                AverageLoss(),
                PayoffRatio(),
                Expectancy(),
            ]

            # Add prediction metrics if labels available
            if labels is not None:
                metrics.extend([
                    DirectionalAccuracy(),
                    SignalRate(),
                ])

        # Compute all metrics
        result = {}
        for metric in metrics:
            metric_result = metric.compute(returns, context)
            result.update(metric_result)
            # Add to context for dependent metrics
            context.update(metric_result)

        return result


class Backtester:
    """
    Main entry point for running backtests.

    This is a convenience wrapper around VectorizedEngine.

    Example:
        >>> config = BacktestConfig(initial_capital=100_000)
        >>> backtester = Backtester(config)
        >>> result = backtester.run(data, strategy)
        >>> print(result.summary())
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize the backtester.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self._engine = VectorizedEngine(config)

    def run(
        self,
        data: BacktestData,
        strategy: Strategy,
        metrics: Optional[List[Metric]] = None,
    ) -> BacktestResult:
        """
        Run a backtest.

        Args:
            data: BacktestData containing prices and optional labels
            strategy: Trading strategy
            metrics: Optional list of metrics to compute

        Returns:
            BacktestResult with complete output
        """
        return self._engine.run(data, strategy, metrics)

    def run_from_arrays(
        self,
        prices: np.ndarray,
        predictions: np.ndarray,
        labels: Optional[np.ndarray] = None,
        shifted: bool = False,
        metrics: Optional[List[Metric]] = None,
    ) -> BacktestResult:
        """
        Convenience method to run backtest from numpy arrays.

        Args:
            prices: Mid-price series
            predictions: Model predictions
            labels: True labels (optional)
            shifted: If predictions use shifted labels (0/1/2)
            metrics: Optional metrics

        Returns:
            BacktestResult
        """
        from lobbacktest.strategies.direction import DirectionStrategy

        data = BacktestData(prices=prices, labels=labels)
        strategy = DirectionStrategy(predictions, shifted=shifted)
        return self.run(data, strategy, metrics)

