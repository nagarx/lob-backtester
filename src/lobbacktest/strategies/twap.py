"""
TWAP (Time-Weighted Average Price) execution strategy for smoothed-average regression.

Matches the TLOB smoothed-average label: the model predicts the average of the
next k mid-price returns. A TWAP execution enters proportionally across those k
intervals, so the execution return approximates the smoothed average.

Instead of entering 100% at time t and exiting at t+k (point-to-point), TWAP:
    - At time t: enter 1/k of position if |predicted_return| > threshold
    - At time t+1: add another 1/k
    - ...
    - At time t+k-1: add final 1/k
    - At time t+k: exit all

The average entry price approximates the TWAP, and the P&L should be closer
to the smoothed-average return the model was trained to predict.

Reference:
    Kolm, Turiel & Westray (2023). Smoothed-average target r_t^(k) = (1/k) * sum(m_{t+i} - m_t) / m_t.
    TWAP execution ensures execution return ~ smoothed-average return.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lobbacktest.strategies.base import Signal, SignalOutput, Strategy


@dataclass
class TWAPStrategyConfig:
    """Configuration for TWAP execution.

    Attributes:
        min_return_bps: Minimum |predicted return| to initiate a TWAP sequence.
        max_spread_bps: Maximum spread to allow entry.
        twap_window: Number of intervals to spread entry across (matches smoothing k).
        cooldown_events: Events to wait after TWAP completion before next entry.
    """

    min_return_bps: float = 3.0
    max_spread_bps: float = 1.05
    twap_window: int = 10
    cooldown_events: int = 5


class TWAPStrategy(Strategy):
    """TWAP execution strategy that matches smoothed-average regression labels.

    Instead of point-to-point entry/exit, spreads entry across twap_window
    intervals. The effective entry price is the TWAP, which should match
    the smoothed-average label the model was trained on.

    Args:
        predicted_returns: Model predictions [N] or [N, H] in bps.
        spreads: Bid-ask spread [N] in bps.
        prices: Mid prices [N].
        config: TWAPStrategyConfig.
    """

    def __init__(
        self,
        predicted_returns: np.ndarray,
        spreads: Optional[np.ndarray] = None,
        prices: Optional[np.ndarray] = None,
        config: Optional[TWAPStrategyConfig] = None,
    ):
        self.config = config or TWAPStrategyConfig()

        if predicted_returns.ndim == 2:
            self.predictions_bps = predicted_returns[:, 0]
        else:
            self.predictions_bps = predicted_returns

        self.spreads = spreads
        self.prices = prices

    @property
    def name(self) -> str:
        return (
            f"TWAP(min_ret>={self.config.min_return_bps:.1f}bps,"
            f"window={self.config.twap_window},"
            f"spread<={self.config.max_spread_bps:.2f})"
        )

    def generate_signals(
        self,
        prices: np.ndarray,
        index: Optional[int] = None,
    ) -> SignalOutput:
        """Generate TWAP signals."""
        n = len(prices)
        if self.prices is None:
            self.prices = prices

        signals = np.full(n, Signal.HOLD, dtype=np.int32)

        in_twap = False
        twap_direction = 0
        twap_start = 0
        twap_entries = 0
        cooldown_remaining = 0
        k = self.config.twap_window

        n_twap_sequences = 0
        n_entries = 0
        total_hold_events = 0

        for i in range(n):
            if in_twap:
                events_since_start = i - twap_start

                if events_since_start < k:
                    if self.spreads is not None and self.spreads[i] > self.config.max_spread_bps:
                        signals[i] = Signal.BUY if twap_direction == 1 else Signal.SELL
                    else:
                        signals[i] = Signal.BUY if twap_direction == 1 else Signal.SELL
                        twap_entries += 1
                else:
                    signals[i] = Signal.EXIT
                    total_hold_events += events_since_start
                    in_twap = False
                    twap_direction = 0
                    cooldown_remaining = self.config.cooldown_events
            else:
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                    continue

                pred = self.predictions_bps[i]
                if abs(pred) < self.config.min_return_bps:
                    continue
                if self.spreads is not None and self.spreads[i] > self.config.max_spread_bps:
                    continue

                if i + k >= n:
                    continue

                in_twap = True
                twap_start = i
                twap_entries = 1
                n_twap_sequences += 1
                n_entries += 1

                if pred > 0:
                    twap_direction = 1
                    signals[i] = Signal.BUY
                else:
                    twap_direction = -1
                    signals[i] = Signal.SELL

        if in_twap:
            total_hold_events += n - 1 - twap_start

        avg_hold = total_hold_events / max(n_twap_sequences, 1)

        metadata = {
            "strategy_type": "twap",
            "twap_window": k,
            "min_return_bps": self.config.min_return_bps,
            "n_twap_sequences": n_twap_sequences,
            "n_entries": n_entries,
            "avg_hold_events": round(avg_hold, 1),
            "trade_rate": float(n_twap_sequences / max(n, 1)),
        }

        confidence = np.abs(self.predictions_bps)

        return SignalOutput(
            signals=signals,
            confidence=confidence,
            metadata=metadata,
        )
