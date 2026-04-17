"""
Readability-first trading strategy with configurable holding policies.

Trades only when the microstructure is readable -- when multi-horizon
signals agree and decoder confidence is high. Uses a HoldingPolicy
to determine when to exit positions, preventing signal flickering.

Entry gates (ALL must pass for NEW position):
  1. agreement_ratio == 1.0 (all horizons agree = readable)
  2. confirmation_score > min_confidence (decoder confident)
  3. spread <= max_spread_bps (orderly book)
  4. prediction is directional (Up or Down, not Stable)

While in a position, the HoldingPolicy controls exit timing.
Gate is NOT re-checked while holding -- only the policy decides exits.

First backtest without holding policy: 14,051 trades, -36.79% return.
With H10-aligned holding (10 events): expected ~1,400 trades.
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
class ReadabilityConfig:
    """Configuration for readability gates and holding behavior.

    Attributes:
        min_agreement: Minimum agreement_ratio to enter (1.0 = all horizons agree).
        min_confidence: Minimum confirmation_score to enter.
        max_spread_bps: Maximum spread in bps to enter.
        require_directional: Only enter on Up/Down predictions, not Stable.
        cooldown_events: Wait this many events after exit before re-entering.
        min_volatility: Minimum order_flow_volatility to enter (None = disabled).
            Higher volatility → larger moves → higher probability of exceeding
            breakeven cost. Profiler: NVDA 1-min return std = 11.6 bps mean,
            but varies 2-3x intraday.
    """

    min_agreement: float = 0.667  # P5 FIX: 2/3 horizons agree (was 1.0 = all must agree)
    min_confidence: float = 0.65
    max_spread_bps: float = 1.05
    require_directional: bool = True
    cooldown_events: int = 0
    min_volatility: Optional[float] = None


class ReadabilityStrategy(Strategy):
    """
    Readability-first strategy with holding policy integration.

    Two modes of operation:
    1. Gate check: For events where we have NO position, check readability
       gates to decide whether to enter.
    2. Holding check: For events where we ARE in a position, consult the
       HoldingPolicy to decide whether to exit.

    This eliminates signal flickering because the gate is only checked
    for new entries, not for position continuation.

    Args:
        predictions: Model predictions [N], 0=Down, 1=Stable, 2=Up.
        agreement_ratio: Cross-horizon agreement [N].
        confirmation_score: Decoder confidence [N].
        spreads: Bid-ask spread in bps [N].
        prices: Mid prices [N] (for unrealized P&L computation).
        config: ReadabilityConfig with gate thresholds.
        holding_policy: HoldingPolicy for exit decisions (default: H10-aligned).
    """

    def __init__(
        self,
        predictions: np.ndarray,
        agreement_ratio: np.ndarray,
        confirmation_score: np.ndarray,
        spreads: Optional[np.ndarray] = None,
        prices: Optional[np.ndarray] = None,
        volatility: Optional[np.ndarray] = None,
        config: Optional[ReadabilityConfig] = None,
        holding_policy: Optional[HoldingPolicy] = None,
        label_mapping: Optional[LabelMapping] = None,
    ):
        self.predictions = predictions
        self.agreement_ratio = agreement_ratio
        self.confirmation_score = confirmation_score
        self.spreads = spreads
        self.prices = prices
        self.volatility = volatility
        self.config = config or ReadabilityConfig()
        self.holding_policy = holding_policy or HorizonAlignedPolicy(hold_events=10)
        self.label_mapping = label_mapping or SHIFTED_MAPPING

    @property
    def name(self) -> str:
        return (
            f"Readability(agree>={self.config.min_agreement:.2f},"
            f"conf>={self.config.min_confidence:.2f},"
            f"spread<={self.config.max_spread_bps:.2f},"
            f"hold={self.holding_policy.policy_name},"
            f"cool={self.config.cooldown_events})"
        )

    def _check_entry_gate(self, i: int) -> bool:
        """Check if event i passes all readability gates for entry."""
        if self.agreement_ratio[i] < self.config.min_agreement:
            return False
        if self.confirmation_score[i] <= self.config.min_confidence:
            return False
        if self.spreads is not None and self.config.max_spread_bps > 0:
            if self.spreads[i] > self.config.max_spread_bps:
                return False
        if self.config.require_directional:
            if not self.label_mapping.is_directional(int(self.predictions[i])):
                return False
        if self.config.min_volatility is not None and self.volatility is not None:
            if self.volatility[i] < self.config.min_volatility:
                return False
        return True

    def _build_holding_state(
        self, i: int, entry_idx: int, entry_pred: int, position_side: int,
    ) -> HoldingState:
        """Build HoldingState for the holding policy at event i."""
        entry_price = self.prices[entry_idx] if self.prices is not None else 1.0
        current_price = self.prices[i] if self.prices is not None else 1.0

        if entry_price > EPS:
            price_change_bps = (current_price - entry_price) / entry_price * 10000.0
            unrealized_pnl_bps = price_change_bps * position_side
        else:
            unrealized_pnl_bps = 0.0

        return HoldingState(
            events_held=i - entry_idx,
            entry_prediction=entry_pred,
            current_prediction=int(self.predictions[i]),
            current_agreement=float(self.agreement_ratio[i]),
            current_confirmation=float(self.confirmation_score[i]),
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
        """
        Generate signals with holding policy integration.

        Event loop:
        - If FLAT and cooldown expired: check entry gate -> BUY/SELL or HOLD
        - If IN POSITION: check holding policy -> continue (BUY/SELL) or EXIT
        - After EXIT: enter cooldown period
        """
        n = len(prices)
        if self.prices is None:
            self.prices = prices

        signals = np.full(n, Signal.HOLD, dtype=np.int32)

        in_position = False
        position_side = 0  # +1 long, -1 short
        entry_idx = 0
        entry_pred = 0
        cooldown_remaining = 0

        n_entries = 0
        n_exits = 0
        n_holds_in_position = 0
        n_gate_pass = 0
        n_gate_fail = 0
        total_hold_events = 0
        exit_reasons = {"policy": 0, "end_of_data": 0}

        for i in range(n):
            if in_position:
                state = self._build_holding_state(i, entry_idx, entry_pred, position_side)

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
                    n_holds_in_position += 1
            else:
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                    signals[i] = Signal.HOLD
                    continue

                if self._check_entry_gate(i):
                    n_gate_pass += 1
                    pred = int(self.predictions[i])
                    if self.label_mapping.is_bullish(pred):
                        signals[i] = Signal.BUY
                        in_position = True
                        position_side = 1
                        entry_idx = i
                        entry_pred = pred
                        n_entries += 1
                    elif self.label_mapping.is_bearish(pred):
                        signals[i] = Signal.SELL
                        in_position = True
                        position_side = -1
                        entry_idx = i
                        entry_pred = pred
                        n_entries += 1
                else:
                    n_gate_fail += 1
                    signals[i] = Signal.HOLD

        if in_position:
            total_hold_events += n - 1 - entry_idx
            exit_reasons["end_of_data"] += 1

        avg_hold = total_hold_events / max(n_entries, 1)

        metadata = {
            "holding_policy": self.holding_policy.policy_name,
            "cooldown_events": self.config.cooldown_events,
            "n_entries": n_entries,
            "n_exits": n_exits,
            "n_holds_in_position": n_holds_in_position,
            "n_gate_pass": n_gate_pass,
            "n_gate_fail": n_gate_fail,
            "avg_hold_events": round(avg_hold, 1),
            "trade_rate": float(n_entries / max(n, 1)),
            "exit_reasons": exit_reasons,
        }

        return SignalOutput(
            signals=signals,
            confidence=self.confirmation_score,
            metadata=metadata,
        )
