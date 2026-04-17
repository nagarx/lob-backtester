"""
Regression-based trading strategy.

Uses continuous bps return predictions to make trading decisions.
Entry gate: |predicted_return| > min_return_bps AND spread <= max_spread.
Direction: sign(predicted_return) determines BUY/SELL.
Confidence: |predicted_return| serves as confidence.

This is the regression counterpart to ReadabilityStrategy (classification).
Both implement the same Strategy interface and use the same HoldingPolicy
framework, but RegressionStrategy uses continuous predictions instead of
discrete class labels.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lobbacktest.labels import LabelMapping, SHIFTED_MAPPING
from lobbacktest.strategies.base import Signal, SignalOutput, Strategy
from lobbacktest.strategies.holding import (
    HoldingPolicy,
    HoldingState,
    HorizonAlignedPolicy,
)


EPS = 1e-10


@dataclass
class RegressionStrategyConfig:
    """Configuration for regression-based trading.

    Attributes:
        min_return_bps: Minimum |predicted return| to enter a trade.
            Should be set above the option breakeven cost.
            ATM 0DTE call: ~5 bps, ATM put: ~4 bps.
        max_spread_bps: Maximum spread to enter (same as ReadabilityStrategy).
        primary_horizon_idx: Which column of predicted_returns to use (0-indexed).
            For horizons [10, 60, 300]: 0=H10, 1=H60, 2=H300.
        cooldown_events: Wait this many events after exit before re-entering.
    """

    min_return_bps: float = 5.0
    max_spread_bps: float = 1.05
    primary_horizon_idx: int = 0  # P4 FIX: H10 (index 0), not H60 (index 1)
    cooldown_events: int = 0


class RegressionStrategy(Strategy):
    """
    Trading strategy driven by continuous return predictions.

    Entry: |predicted_return[i]| > min_return_bps AND spread <= max_spread
    Direction: predicted_return > 0 -> BUY, < 0 -> SELL
    Exit: via HoldingPolicy (same framework as ReadabilityStrategy)

    Args:
        predicted_returns: Model predictions [N, H] or [N] in bps.
        spreads: Bid-ask spread [N] in bps.
        prices: Mid prices [N] (for P&L computation in holding policy).
        config: RegressionStrategyConfig with thresholds.
        holding_policy: HoldingPolicy for exit decisions.
    """

    def __init__(
        self,
        predicted_returns: np.ndarray,
        spreads: Optional[np.ndarray] = None,
        prices: Optional[np.ndarray] = None,
        config: Optional[RegressionStrategyConfig] = None,
        holding_policy: Optional[HoldingPolicy] = None,
        label_mapping: Optional[LabelMapping] = None,
    ):
        self.config = config or RegressionStrategyConfig()
        self.label_mapping = label_mapping or SHIFTED_MAPPING

        if predicted_returns.ndim == 2:
            self.predictions_bps = predicted_returns[:, self.config.primary_horizon_idx]
        else:
            self.predictions_bps = predicted_returns

        self.spreads = spreads
        self.prices = prices
        self.holding_policy = holding_policy or HorizonAlignedPolicy(hold_events=60)

    @property
    def name(self) -> str:
        return (
            f"Regression(min_ret>={self.config.min_return_bps:.1f}bps,"
            f"spread<={self.config.max_spread_bps:.2f},"
            f"hold={self.holding_policy.policy_name})"
        )

    def _check_entry_gate(self, i: int) -> bool:
        """Check if event i passes all entry gates."""
        if abs(self.predictions_bps[i]) < self.config.min_return_bps:
            return False
        if self.spreads is not None and self.config.max_spread_bps > 0:
            if self.spreads[i] > self.config.max_spread_bps:
                return False
        return True

    def _build_holding_state(
        self, i: int, entry_idx: int, position_side: int,
    ) -> HoldingState:
        entry_price = self.prices[entry_idx] if self.prices is not None else 1.0
        current_price = self.prices[i] if self.prices is not None else 1.0

        if entry_price > EPS:
            price_change_bps = (current_price - entry_price) / entry_price * 10000.0
            unrealized_pnl_bps = price_change_bps * position_side
        else:
            unrealized_pnl_bps = 0.0

        pred_class = self.label_mapping.up if self.predictions_bps[i] > 0 else self.label_mapping.down

        return HoldingState(
            events_held=i - entry_idx,
            entry_prediction=pred_class,
            current_prediction=pred_class,
            current_agreement=1.0,
            current_confirmation=abs(self.predictions_bps[i]) / 20.0,
            current_spread=float(self.spreads[i]) if self.spreads is not None else 0.0,
            entry_price=float(entry_price),
            current_price=float(current_price),
            unrealized_pnl_bps=float(unrealized_pnl_bps),
            position_side=position_side,
        )

    def generate_signals(
        self,
        prices: np.ndarray,
        index: Optional[int] = None,
    ) -> SignalOutput:
        """Generate BUY/SELL/HOLD signals from regression predictions."""
        n = len(prices)
        if self.prices is None:
            self.prices = prices

        signals = np.full(n, Signal.HOLD, dtype=np.int32)

        in_position = False
        position_side = 0
        entry_idx = 0
        cooldown_remaining = 0

        n_entries = 0
        n_exits = 0
        total_hold_events = 0
        exit_reasons = {"policy": 0, "end_of_data": 0}

        for i in range(n):
            if in_position:
                state = self._build_holding_state(i, entry_idx, position_side)

                if self.holding_policy.should_exit(state):
                    signals[i] = Signal.EXIT
                    total_hold_events += i - entry_idx
                    in_position = False
                    position_side = 0
                    cooldown_remaining = self.config.cooldown_events
                    n_exits += 1
                    exit_reasons["policy"] += 1
                else:
                    signals[i] = Signal.BUY if position_side == 1 else Signal.SELL
            else:
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                    continue

                if self._check_entry_gate(i):
                    pred = self.predictions_bps[i]
                    if pred > 0:
                        signals[i] = Signal.BUY
                        in_position = True
                        position_side = 1
                    else:
                        signals[i] = Signal.SELL
                        in_position = True
                        position_side = -1
                    entry_idx = i
                    n_entries += 1

        if in_position:
            total_hold_events += n - 1 - entry_idx
            exit_reasons["end_of_data"] += 1

        avg_hold = total_hold_events / max(n_entries, 1)
        metadata = {
            "strategy_type": "regression",
            "holding_policy": self.holding_policy.policy_name,
            "min_return_bps": self.config.min_return_bps,
            "n_entries": n_entries,
            "n_exits": n_exits,
            "avg_hold_events": round(avg_hold, 1),
            "trade_rate": float(n_entries / max(n, 1)),
            "exit_reasons": exit_reasons,
        }

        confidence = np.abs(self.predictions_bps)

        return SignalOutput(
            signals=signals,
            confidence=confidence,
            metadata=metadata,
        )
