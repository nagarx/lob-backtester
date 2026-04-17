"""
Readability Hybrid Strategy: classification gate + regression magnitude.

Combines the best of both approaches:
    - HMHP classification for DIRECTION (93.88% DA at high conviction)
    - Ridge regression for MAGNITUDE (IC=0.616, filters by predicted size)

Entry gates (ALL must pass):
    1. agreement_ratio >= min_agreement (readability gate)
    2. confirmation_score > min_confidence (decoder confidence)
    3. spread <= max_spread_bps (orderly book)
    4. prediction is directional (Up or Down, not Stable)
    5. |predicted_return| >= min_return_bps (magnitude gate -- NEW)

Direction comes from classification (predictions[i]): 0=Down, 2=Up.
Magnitude comes from regression (predicted_returns[i]): continuous bps.

This produces very few, very high-quality trades where BOTH models agree.

Reference:
    - HMHP 128-feat: 93.88% directional accuracy at agreement=1.0, confirm>0.65
    - Ridge: IC=0.616, DA=72.2% (from ablation findings)
    - Combined: direction from classification, magnitude from regression
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
class ReadabilityHybridConfig:
    """Configuration for the readability + magnitude hybrid strategy.

    Attributes:
        min_agreement: Minimum agreement_ratio to enter (1.0 = all horizons agree).
        min_confidence: Minimum confirmation_score to enter.
        max_spread_bps: Maximum spread in bps to enter.
        min_return_bps: Minimum |predicted_return| in bps to enter.
            This is the magnitude gate from regression. Set above breakeven.
        require_directional: Only enter on Up/Down predictions, not Stable.
        cooldown_events: Wait this many events after exit before re-entering.
    """

    min_agreement: float = 1.0
    min_confidence: float = 0.65
    max_spread_bps: float = 1.05
    min_return_bps: float = 3.0
    require_directional: bool = True
    cooldown_events: int = 0


class ReadabilityHybridStrategy(Strategy):
    """Hybrid strategy combining classification readability with regression magnitude.

    Args:
        predictions: HMHP class predictions [N], 0=Down, 1=Stable, 2=Up.
        agreement_ratio: HMHP cross-horizon agreement [N].
        confirmation_score: HMHP decoder confidence [N].
        predicted_returns: Ridge regression predictions [N] in bps.
        spreads: Bid-ask spread [N] in bps.
        prices: Mid prices [N].
        config: ReadabilityHybridConfig with gate thresholds.
        holding_policy: HoldingPolicy for exit decisions.
    """

    def __init__(
        self,
        predictions: np.ndarray,
        agreement_ratio: np.ndarray,
        confirmation_score: np.ndarray,
        predicted_returns: np.ndarray,
        spreads: Optional[np.ndarray] = None,
        prices: Optional[np.ndarray] = None,
        config: Optional[ReadabilityHybridConfig] = None,
        holding_policy: Optional[HoldingPolicy] = None,
        label_mapping: Optional[LabelMapping] = None,
    ):
        self.predictions = predictions
        self.agreement_ratio = agreement_ratio
        self.confirmation_score = confirmation_score
        self.predicted_returns = predicted_returns
        self.spreads = spreads
        self.prices = prices
        self.config = config or ReadabilityHybridConfig()
        self.holding_policy = holding_policy or HorizonAlignedPolicy(hold_events=10)
        self.label_mapping = label_mapping or SHIFTED_MAPPING

    @property
    def name(self) -> str:
        return (
            f"Hybrid(agree>={self.config.min_agreement:.1f},"
            f"conf>={self.config.min_confidence:.2f},"
            f"|ret|>={self.config.min_return_bps:.1f}bps,"
            f"spread<={self.config.max_spread_bps:.2f},"
            f"hold={self.holding_policy.policy_name})"
        )

    def _check_entry_gate(self, i: int) -> bool:
        """Check if event i passes ALL gates (readability + magnitude)."""
        if self.agreement_ratio[i] < self.config.min_agreement:
            return False
        if self.confirmation_score[i] <= self.config.min_confidence:
            return False
        if self.spreads is not None and self.config.max_spread_bps > 0:
            if self.spreads[i] > self.config.max_spread_bps:
                return False
        if self.config.require_directional and self.label_mapping.is_stable(int(self.predictions[i])):
            return False
        if abs(self.predicted_returns[i]) < self.config.min_return_bps:
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

        return HoldingState(
            events_held=i - entry_idx,
            entry_prediction=int(self.predictions[entry_idx]),
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
        """Generate BUY/SELL/HOLD signals from hybrid dual-gate logic."""
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
        n_readability_pass = 0
        n_magnitude_pass = 0
        n_both_pass = 0

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
                else:
                    signals[i] = Signal.BUY if position_side == 1 else Signal.SELL
            else:
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                    continue

                readability_ok = (
                    self.agreement_ratio[i] >= self.config.min_agreement
                    and self.confirmation_score[i] > self.config.min_confidence
                    and (self.spreads is None or self.spreads[i] <= self.config.max_spread_bps)
                    and (not self.config.require_directional or not self.label_mapping.is_stable(int(self.predictions[i])))
                )
                magnitude_ok = abs(self.predicted_returns[i]) >= self.config.min_return_bps

                if readability_ok:
                    n_readability_pass += 1
                if magnitude_ok:
                    n_magnitude_pass += 1

                if readability_ok and magnitude_ok:
                    n_both_pass += 1
                    pred = int(self.predictions[i])
                    if self.label_mapping.is_bullish(pred):
                        signals[i] = Signal.BUY
                        in_position = True
                        position_side = 1
                    elif self.label_mapping.is_bearish(pred):
                        signals[i] = Signal.SELL
                        in_position = True
                        position_side = -1
                    entry_idx = i
                    n_entries += 1

        if in_position:
            total_hold_events += n - 1 - entry_idx

        avg_hold = total_hold_events / max(n_entries, 1)

        metadata = {
            "strategy_type": "readability_hybrid",
            "holding_policy": self.holding_policy.policy_name,
            "min_agreement": self.config.min_agreement,
            "min_confidence": self.config.min_confidence,
            "min_return_bps": self.config.min_return_bps,
            "n_entries": n_entries,
            "n_exits": n_exits,
            "avg_hold_events": round(avg_hold, 1),
            "trade_rate": float(n_entries / max(n, 1)),
            "n_readability_pass": n_readability_pass,
            "n_magnitude_pass": n_magnitude_pass,
            "n_both_pass": n_both_pass,
            "readability_pass_rate": round(n_readability_pass / max(n, 1), 4),
            "magnitude_pass_rate": round(n_magnitude_pass / max(n, 1), 4),
            "both_pass_rate": round(n_both_pass / max(n, 1), 4),
        }

        confidence = np.abs(self.predicted_returns)

        return SignalOutput(
            signals=signals,
            confidence=confidence,
            metadata=metadata,
        )
