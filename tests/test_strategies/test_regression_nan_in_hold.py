"""
Wave 1D T1-D-003 closure tests (2026-05-17).

Locks NaN guard on RegressionStrategy._build_holding_state pred_class
derivation. Pre-fix `NaN > 0` evaluated False (IEEE 754) → silently picked
`down` → DirectionReversalPolicy detected false reversal → unintended
early EXIT on garbage signal.

Mid-hold NaN injection path is plausible (price-stream gap from feature-
extractor); entry-gate at L107 already guards against entering on NaN
(PRESERVE #25), but mid-hold _build_holding_state is called for every i
where in_position.

Fail-CLOSED: NaN pred → use `stable` sentinel (label_mapping.stable);
prevents NaN-induced phantom exits in-hold.
"""

import numpy as np
import pytest

from lobbacktest.labels import SHIFTED_MAPPING
from lobbacktest.strategies.regression import (
    RegressionStrategy,
    RegressionStrategyConfig,
)


def _build_strategy(predicted_returns: np.ndarray) -> RegressionStrategy:
    """Helper: construct minimal RegressionStrategy for state-building tests."""
    return RegressionStrategy(
        predicted_returns=predicted_returns,
        config=RegressionStrategyConfig(min_return_bps=5.0, max_spread_bps=10.0),
        label_mapping=SHIFTED_MAPPING,
    )


class TestRegressionBuildHoldingStateNaNGuard:
    """T1-D-003: _build_holding_state pred_class + confirmation NaN-guarded."""

    def test_nan_prediction_yields_stable_pred_class(self):
        """NaN prediction in-hold → pred_class = stable (no false reversal)."""
        predictions = np.array([10.0, 8.0, float("nan"), 7.0, 6.0])
        strategy = _build_strategy(predictions)

        # Set prices so entry_price > EPS (so unrealized_pnl_bps path runs cleanly).
        strategy.prices = np.array([100.0, 100.1, 100.2, 100.3, 100.4])

        # Build state at i=2 (NaN prediction); entry was at i=0 (positive 10 bps → up)
        state = strategy._build_holding_state(
            i=2, entry_idx=0, position_side=SHIFTED_MAPPING.up - 1
        )

        # Pre-fix: NaN > 0 → False → pred_class = down → would be a reversal
        # Post-fix: pred_class = stable (no reversal trigger downstream)
        assert state.current_prediction == SHIFTED_MAPPING.stable

    def test_positive_finite_prediction_yields_up(self):
        """Finite positive prediction → pred_class = up (pre-existing behavior)."""
        predictions = np.array([10.0, 8.0, 7.0])
        strategy = _build_strategy(predictions)
        strategy.prices = np.array([100.0, 100.1, 100.2])

        state = strategy._build_holding_state(
            i=2, entry_idx=0, position_side=SHIFTED_MAPPING.up - 1
        )
        assert state.current_prediction == SHIFTED_MAPPING.up

    def test_negative_finite_prediction_yields_down(self):
        """Finite negative prediction → pred_class = down (pre-existing behavior)."""
        predictions = np.array([-10.0, -8.0, -7.0])
        strategy = _build_strategy(predictions)
        strategy.prices = np.array([100.0, 100.1, 100.2])

        state = strategy._build_holding_state(
            i=2, entry_idx=0, position_side=SHIFTED_MAPPING.down - 1
        )
        assert state.current_prediction == SHIFTED_MAPPING.down

    def test_zero_finite_prediction_yields_down(self):
        """Zero prediction → pred_class = down (`> 0` strict; pre-existing).

        Note: Wave 2-H H8 flags zero-P&L trade counting; this test locks
        the existing strict `> 0` semantic for pred_class derivation.
        """
        predictions = np.array([0.0, 0.0, 0.0])
        strategy = _build_strategy(predictions)
        strategy.prices = np.array([100.0, 100.1, 100.2])

        state = strategy._build_holding_state(
            i=2, entry_idx=0, position_side=SHIFTED_MAPPING.up - 1
        )
        # 0.0 > 0 is False → down (per pre-existing strict-> semantic)
        assert state.current_prediction == SHIFTED_MAPPING.down

    def test_nan_confirmation_replaced_with_zero(self):
        """NaN prediction → current_confirmation = 0.0 (not NaN propagated)."""
        predictions = np.array([10.0, 8.0, float("nan")])
        strategy = _build_strategy(predictions)
        strategy.prices = np.array([100.0, 100.1, 100.2])

        state = strategy._build_holding_state(
            i=2, entry_idx=0, position_side=SHIFTED_MAPPING.up - 1
        )
        # Pre-fix: |NaN| / 20.0 = NaN propagated
        # Post-fix: 0.0
        assert state.current_confirmation == 0.0

    def test_inf_prediction_yields_stable(self):
        """Inf is also non-finite → stable (fail-closed)."""
        predictions = np.array([10.0, 8.0, float("inf")])
        strategy = _build_strategy(predictions)
        strategy.prices = np.array([100.0, 100.1, 100.2])

        state = strategy._build_holding_state(
            i=2, entry_idx=0, position_side=SHIFTED_MAPPING.up - 1
        )
        assert state.current_prediction == SHIFTED_MAPPING.stable
