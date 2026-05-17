"""
Wave 1D T1-D-001 + T1-D-002 closure tests (2026-05-17).

Locks NaN guards on holding-policy exit decisions:
- DirectionReversalPolicy.should_exit on current_agreement (T1-D-001)
- StopLossTakeProfitPolicy.should_exit on unrealized_pnl_bps (T1-D-002)

Pre-fix `value <op> threshold` with NaN value evaluated False (IEEE 754
NaN-comparison invariant) → gates silently passed → position rode
indefinitely on broken data. Fail-CLOSED convention mirrors PRESERVE
#25 entry-gate NaN-guard pattern (post-#PY-71+FIND-046+HF-3 closure
2026-05-15): NaN → exit (defensive).

User-risk path: regression.py:122-125 produces NaN unrealized_pnl_bps
when current_price is NaN (price-stream gap from feature-extractor) →
NaN propagates into HoldingState → SL/TP never fires → position rides
indefinitely with meaningless P&L.
"""

import math

import pytest

from lobbacktest.strategies.holding import (
    DirectionReversalPolicy,
    StopLossTakeProfitPolicy,
)


def _make_state(
    events_held: int = 5,
    entry_prediction: int = 2,
    current_prediction: int = 2,
    current_agreement: float = 1.0,
    current_confirmation: float = 0.65,
    current_spread: float = 0.8,
    entry_price: float = 100.0,
    current_price: float = 100.0,
    unrealized_pnl_bps: float = 0.0,
    position_side: int = 1,
):
    """Local factory for HoldingState (avoids cross-file fixture coupling)."""
    from lobbacktest.strategies.holding import HoldingState
    return HoldingState(
        events_held=events_held,
        entry_prediction=entry_prediction,
        current_prediction=current_prediction,
        current_agreement=current_agreement,
        current_confirmation=current_confirmation,
        current_spread=current_spread,
        entry_price=entry_price,
        current_price=current_price,
        unrealized_pnl_bps=unrealized_pnl_bps,
        position_side=position_side,
    )


class TestDirectionReversalNaNGuard:
    """T1-D-001: DirectionReversalPolicy.should_exit guards current_agreement NaN."""

    def test_nan_agreement_triggers_exit_when_require_gate_true(self):
        """NaN current_agreement with require_gate=True → exit (fail-closed).

        Pre-fix Wave 1D: `NaN < 1.0` is False → gated-exit branch silently
        skipped → strategy continued holding broken signal. Post-fix:
        np.isfinite guard triggers exit.
        """
        policy = DirectionReversalPolicy(max_hold_events=60, require_gate=True)
        state = _make_state(
            events_held=5,  # below max_hold (won't trigger by max)
            entry_prediction=2,  # Up
            current_prediction=2,  # No reversal (Up→Up)
            current_agreement=float("nan"),  # broken
        )
        # Pre-fix would return False (NaN < 1.0 = False; no exit trigger)
        # Post-fix returns True (NaN guard fires)
        assert policy.should_exit(state) is True

    def test_nan_agreement_no_exit_when_require_gate_false(self):
        """NaN current_agreement with require_gate=False → no exit.

        The gate is opt-in; when disabled, NaN agreement is irrelevant.
        Verifies fix doesn't over-trigger when feature is off.
        """
        policy = DirectionReversalPolicy(max_hold_events=60, require_gate=False)
        state = _make_state(
            events_held=5,
            entry_prediction=2,
            current_prediction=2,  # No reversal
            current_agreement=float("nan"),
        )
        assert policy.should_exit(state) is False

    def test_inf_agreement_also_triggers_exit_when_gated(self):
        """Inf is also non-finite → guard fires."""
        policy = DirectionReversalPolicy(max_hold_events=60, require_gate=True)
        state = _make_state(
            events_held=5,
            entry_prediction=2,
            current_prediction=2,
            current_agreement=float("inf"),
        )
        assert policy.should_exit(state) is True

    def test_finite_agreement_below_threshold_still_exits(self):
        """Finite agreement < 1.0 still triggers exit (pre-existing behavior)."""
        policy = DirectionReversalPolicy(max_hold_events=60, require_gate=True)
        state = _make_state(
            events_held=5,
            entry_prediction=2,
            current_prediction=2,
            current_agreement=0.5,  # finite, below 1.0
        )
        assert policy.should_exit(state) is True

    def test_finite_agreement_at_threshold_no_exit(self):
        """Finite agreement == 1.0 → no exit (boundary behavior preserved)."""
        policy = DirectionReversalPolicy(max_hold_events=60, require_gate=True)
        state = _make_state(
            events_held=5,
            entry_prediction=2,
            current_prediction=2,  # no reversal
            current_agreement=1.0,
        )
        assert policy.should_exit(state) is False


class TestStopLossTakeProfitNaNGuard:
    """T1-D-002: StopLossTakeProfitPolicy.should_exit guards unrealized_pnl_bps NaN.

    This was the MOST OPERATIONALLY DANGEROUS NaN bypass identified by
    Wave 2-G (CLI-reachable via --holding-type stop_loss_take_profit).
    """

    def test_nan_pnl_triggers_exit(self):
        """NaN unrealized_pnl_bps → exit immediately (fail-closed)."""
        policy = StopLossTakeProfitPolicy(
            stop_loss_bps=10.0, take_profit_bps=20.0, max_hold_events=60
        )
        state = _make_state(
            events_held=5,  # below max_hold
            unrealized_pnl_bps=float("nan"),  # broken
        )
        # Pre-fix: NaN <= -10 = False AND NaN >= 20 = False → no exit
        # Post-fix: np.isfinite guard fires → exit
        assert policy.should_exit(state) is True

    def test_inf_pnl_triggers_exit(self):
        """+Inf unrealized_pnl_bps → exit (also non-finite)."""
        policy = StopLossTakeProfitPolicy(
            stop_loss_bps=10.0, take_profit_bps=20.0, max_hold_events=60
        )
        state = _make_state(events_held=5, unrealized_pnl_bps=float("inf"))
        assert policy.should_exit(state) is True

    def test_neg_inf_pnl_triggers_exit(self):
        """-Inf unrealized_pnl_bps → exit."""
        policy = StopLossTakeProfitPolicy(
            stop_loss_bps=10.0, take_profit_bps=20.0, max_hold_events=60
        )
        state = _make_state(events_held=5, unrealized_pnl_bps=float("-inf"))
        assert policy.should_exit(state) is True

    def test_finite_pnl_below_sl_triggers_exit(self):
        """Finite stop-loss path preserved (pre-existing behavior)."""
        policy = StopLossTakeProfitPolicy(
            stop_loss_bps=10.0, take_profit_bps=20.0, max_hold_events=60
        )
        state = _make_state(events_held=5, unrealized_pnl_bps=-12.0)
        assert policy.should_exit(state) is True

    def test_finite_pnl_above_tp_triggers_exit(self):
        """Finite take-profit path preserved (pre-existing behavior)."""
        policy = StopLossTakeProfitPolicy(
            stop_loss_bps=10.0, take_profit_bps=20.0, max_hold_events=60
        )
        state = _make_state(events_held=5, unrealized_pnl_bps=25.0)
        assert policy.should_exit(state) is True

    def test_finite_pnl_within_band_no_exit(self):
        """Finite P&L within SL/TP band → no exit (pre-existing behavior)."""
        policy = StopLossTakeProfitPolicy(
            stop_loss_bps=10.0, take_profit_bps=20.0, max_hold_events=60
        )
        state = _make_state(events_held=5, unrealized_pnl_bps=5.0)
        assert policy.should_exit(state) is False

    def test_max_hold_takes_precedence_over_nan(self):
        """Max-hold check fires before NaN guard (pre-existing ordering preserved)."""
        policy = StopLossTakeProfitPolicy(
            stop_loss_bps=10.0, take_profit_bps=20.0, max_hold_events=60
        )
        state = _make_state(
            events_held=100,  # above max_hold
            unrealized_pnl_bps=float("nan"),
        )
        # Both conditions fire; max_hold short-circuits first. Result is same (exit).
        assert policy.should_exit(state) is True
