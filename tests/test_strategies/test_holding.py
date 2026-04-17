"""
Tests for holding policy framework.

Tests verify:
- HorizonAlignedPolicy: exit after exactly N events
- DirectionReversalPolicy: exit on direction reversal, max hold, optional gate
- StopLossTakeProfitPolicy: exit on SL/TP thresholds
- CompositePolicy: AND/OR combination of policies
- create_holding_policy factory: dict -> policy construction
- HoldingState: data container for policy decisions

Per RULE.md testing philosophy:
- Formula tests: Verify policy decisions match documented behavior
- Edge tests: Boundary conditions (events_held == hold_events exactly)
- Invariant tests: Policies are stateless (same state -> same decision)
"""

import pytest

from lobbacktest.strategies.holding import (
    CompositePolicy,
    DirectionReversalPolicy,
    HoldingPolicy,
    HoldingState,
    HorizonAlignedPolicy,
    StopLossTakeProfitPolicy,
    create_holding_policy,
)


def _make_state(
    events_held: int = 0,
    entry_prediction: int = 2,
    current_prediction: int = 2,
    current_agreement: float = 1.0,
    current_confirmation: float = 0.65,
    current_spread: float = 0.8,
    entry_price: float = 100.0,
    current_price: float = 100.0,
    unrealized_pnl_bps: float = 0.0,
    position_side: int = 1,
) -> HoldingState:
    """Factory for HoldingState with sensible defaults."""
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


class TestHorizonAlignedPolicy:
    """Tests for HorizonAlignedPolicy."""

    def test_hold_until_horizon(self):
        """Hold for exactly N events, exit at N."""
        policy = HorizonAlignedPolicy(hold_events=10)
        assert not policy.should_exit(_make_state(events_held=0))
        assert not policy.should_exit(_make_state(events_held=5))
        assert not policy.should_exit(_make_state(events_held=9))
        assert policy.should_exit(_make_state(events_held=10))
        assert policy.should_exit(_make_state(events_held=15))

    def test_boundary_exactly_at_horizon(self):
        """events_held == hold_events should trigger exit."""
        policy = HorizonAlignedPolicy(hold_events=10)
        state_before = _make_state(events_held=9)
        state_at = _make_state(events_held=10)
        assert not policy.should_exit(state_before)
        assert policy.should_exit(state_at)

    def test_hold_events_1(self):
        """Minimum hold: exit after 1 event."""
        policy = HorizonAlignedPolicy(hold_events=1)
        assert not policy.should_exit(_make_state(events_held=0))
        assert policy.should_exit(_make_state(events_held=1))

    def test_invalid_hold_events_zero(self):
        """hold_events < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="hold_events must be >= 1"):
            HorizonAlignedPolicy(hold_events=0)

    def test_invalid_hold_events_negative(self):
        with pytest.raises(ValueError, match="hold_events must be >= 1"):
            HorizonAlignedPolicy(hold_events=-5)

    def test_policy_name(self):
        assert HorizonAlignedPolicy(hold_events=10).policy_name == "horizon_aligned_10"
        assert HorizonAlignedPolicy(hold_events=60).policy_name == "horizon_aligned_60"

    def test_to_dict(self):
        d = HorizonAlignedPolicy(hold_events=10).to_dict()
        assert d == {"type": "horizon_aligned", "hold_events": 10}

    def test_stateless_same_input_same_output(self):
        """Same state must always produce same decision (policy is stateless)."""
        policy = HorizonAlignedPolicy(hold_events=10)
        state = _make_state(events_held=10)
        assert policy.should_exit(state) == policy.should_exit(state)


class TestDirectionReversalPolicy:
    """Tests for DirectionReversalPolicy."""

    def test_no_reversal_holds(self):
        """Same direction prediction: keep holding."""
        policy = DirectionReversalPolicy(max_hold_events=60)
        state = _make_state(entry_prediction=2, current_prediction=2, events_held=5)
        assert not policy.should_exit(state)

    def test_reversal_up_to_down(self):
        """Up (2) to Down (0) is a reversal: exit."""
        policy = DirectionReversalPolicy(max_hold_events=60)
        state = _make_state(entry_prediction=2, current_prediction=0, events_held=5)
        assert policy.should_exit(state)

    def test_reversal_down_to_up(self):
        """Down (0) to Up (2) is a reversal: exit."""
        policy = DirectionReversalPolicy(max_hold_events=60)
        state = _make_state(entry_prediction=0, current_prediction=2, events_held=5)
        assert policy.should_exit(state)

    def test_stable_is_not_reversal(self):
        """Stable (1) is NOT a reversal from Up or Down: keep holding."""
        policy = DirectionReversalPolicy(max_hold_events=60)
        state_up_stable = _make_state(entry_prediction=2, current_prediction=1, events_held=5)
        state_down_stable = _make_state(entry_prediction=0, current_prediction=1, events_held=5)
        assert not policy.should_exit(state_up_stable)
        assert not policy.should_exit(state_down_stable)

    def test_max_hold_exit(self):
        """Exit when max_hold_events reached regardless of direction."""
        policy = DirectionReversalPolicy(max_hold_events=60)
        state = _make_state(entry_prediction=2, current_prediction=2, events_held=60)
        assert policy.should_exit(state)

    def test_gate_check_agreement(self):
        """With require_gate=True, exit when agreement drops below 1.0."""
        policy = DirectionReversalPolicy(max_hold_events=60, require_gate=True)
        state_good = _make_state(current_agreement=1.0, events_held=5)
        state_bad = _make_state(current_agreement=0.667, events_held=5)
        assert not policy.should_exit(state_good)
        assert policy.should_exit(state_bad)

    def test_gate_not_checked_by_default(self):
        """Without require_gate, agreement doesn't affect exit."""
        policy = DirectionReversalPolicy(max_hold_events=60, require_gate=False)
        state = _make_state(current_agreement=0.333, events_held=5)
        assert not policy.should_exit(state)

    def test_invalid_max_hold(self):
        with pytest.raises(ValueError, match="max_hold_events must be >= 1"):
            DirectionReversalPolicy(max_hold_events=0)

    def test_policy_name(self):
        assert "direction_reversal_60" in DirectionReversalPolicy(60).policy_name
        assert "_gated" in DirectionReversalPolicy(60, require_gate=True).policy_name

    def test_to_dict(self):
        d = DirectionReversalPolicy(60, require_gate=True).to_dict()
        assert d["type"] == "direction_reversal"
        assert d["max_hold_events"] == 60
        assert d["require_gate"] is True


class TestStopLossTakeProfitPolicy:
    """Tests for StopLossTakeProfitPolicy."""

    def test_normal_hold(self):
        """No SL/TP hit and within max hold: keep holding."""
        policy = StopLossTakeProfitPolicy(stop_loss_bps=10, take_profit_bps=20, max_hold_events=60)
        state = _make_state(unrealized_pnl_bps=5.0, events_held=10)
        assert not policy.should_exit(state)

    def test_stop_loss_triggered(self):
        """
        Exit when unrealized loss exceeds stop_loss_bps.
        Formula: exit if unrealized_pnl_bps <= -stop_loss_bps
        """
        policy = StopLossTakeProfitPolicy(stop_loss_bps=10)
        state = _make_state(unrealized_pnl_bps=-10.0)
        assert policy.should_exit(state)

    def test_stop_loss_boundary(self):
        """Exactly at stop loss threshold: should exit."""
        policy = StopLossTakeProfitPolicy(stop_loss_bps=10)
        state_at = _make_state(unrealized_pnl_bps=-10.0)
        state_just_before = _make_state(unrealized_pnl_bps=-9.99)
        assert policy.should_exit(state_at)
        assert not policy.should_exit(state_just_before)

    def test_take_profit_triggered(self):
        """Exit when unrealized gain exceeds take_profit_bps."""
        policy = StopLossTakeProfitPolicy(take_profit_bps=20)
        state = _make_state(unrealized_pnl_bps=20.0)
        assert policy.should_exit(state)

    def test_take_profit_boundary(self):
        """Exactly at take profit threshold: should exit."""
        policy = StopLossTakeProfitPolicy(take_profit_bps=20)
        state_at = _make_state(unrealized_pnl_bps=20.0)
        state_just_before = _make_state(unrealized_pnl_bps=19.99)
        assert policy.should_exit(state_at)
        assert not policy.should_exit(state_just_before)

    def test_max_hold_exit(self):
        """Exit on max hold even if SL/TP not hit."""
        policy = StopLossTakeProfitPolicy(stop_loss_bps=10, take_profit_bps=20, max_hold_events=60)
        state = _make_state(unrealized_pnl_bps=5.0, events_held=60)
        assert policy.should_exit(state)

    def test_invalid_stop_loss(self):
        with pytest.raises(ValueError, match="stop_loss_bps must be > 0"):
            StopLossTakeProfitPolicy(stop_loss_bps=0)

    def test_invalid_take_profit(self):
        with pytest.raises(ValueError, match="take_profit_bps must be > 0"):
            StopLossTakeProfitPolicy(take_profit_bps=-1)

    def test_to_dict(self):
        d = StopLossTakeProfitPolicy(10, 20, 60).to_dict()
        assert d["stop_loss_bps"] == 10
        assert d["take_profit_bps"] == 20
        assert d["max_hold_events"] == 60


class TestCompositePolicy:
    """Tests for CompositePolicy combining multiple policies."""

    def test_any_mode_exit_on_first(self):
        """Mode='any': exit when ANY sub-policy says exit."""
        horizon = HorizonAlignedPolicy(hold_events=10)
        sl = StopLossTakeProfitPolicy(stop_loss_bps=5, take_profit_bps=100, max_hold_events=100)
        composite = CompositePolicy([horizon, sl], mode="any")

        # Horizon reached (10 events), no SL hit
        state = _make_state(events_held=10, unrealized_pnl_bps=0.0)
        assert composite.should_exit(state)

        # SL hit, horizon not reached
        state = _make_state(events_held=3, unrealized_pnl_bps=-5.0)
        assert composite.should_exit(state)

    def test_all_mode_requires_both(self):
        """Mode='all': exit only when ALL sub-policies say exit."""
        horizon = HorizonAlignedPolicy(hold_events=10)
        sl = StopLossTakeProfitPolicy(stop_loss_bps=5, take_profit_bps=100, max_hold_events=100)
        composite = CompositePolicy([horizon, sl], mode="all")

        # Horizon reached but no SL → don't exit
        state = _make_state(events_held=10, unrealized_pnl_bps=0.0)
        assert not composite.should_exit(state)

        # Both horizon AND SL → exit
        state = _make_state(events_held=10, unrealized_pnl_bps=-5.0)
        assert composite.should_exit(state)

    def test_empty_policies_raises(self):
        with pytest.raises(ValueError, match="requires at least one"):
            CompositePolicy([], mode="any")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            CompositePolicy([HorizonAlignedPolicy(10)], mode="xor")

    def test_nested_composite(self):
        """CompositePolicy can contain other CompositePolicies."""
        inner = CompositePolicy([HorizonAlignedPolicy(10)], mode="any")
        outer = CompositePolicy([inner, StopLossTakeProfitPolicy(5, 20, 100)], mode="any")
        state = _make_state(events_held=10)
        assert outer.should_exit(state)

    def test_to_dict(self):
        composite = CompositePolicy(
            [HorizonAlignedPolicy(10), StopLossTakeProfitPolicy(5, 20, 60)],
            mode="any",
        )
        d = composite.to_dict()
        assert d["type"] == "composite"
        assert d["mode"] == "any"
        assert len(d["policies"]) == 2


class TestCreateHoldingPolicy:
    """Tests for factory function."""

    def test_horizon_aligned_from_dict(self):
        policy = create_holding_policy({"type": "horizon_aligned", "hold_events": 10})
        assert isinstance(policy, HorizonAlignedPolicy)
        assert policy.hold_events == 10

    def test_direction_reversal_from_dict(self):
        policy = create_holding_policy({
            "type": "direction_reversal",
            "max_hold_events": 60,
            "require_gate": True,
        })
        assert isinstance(policy, DirectionReversalPolicy)
        assert policy.max_hold_events == 60
        assert policy.require_gate is True

    def test_stop_loss_take_profit_from_dict(self):
        policy = create_holding_policy({
            "type": "stop_loss_take_profit",
            "stop_loss_bps": 10,
            "take_profit_bps": 20,
            "max_hold_events": 60,
        })
        assert isinstance(policy, StopLossTakeProfitPolicy)
        assert policy.stop_loss_bps == 10

    def test_composite_from_dict(self):
        policy = create_holding_policy({
            "type": "composite",
            "mode": "any",
            "policies": [
                {"type": "horizon_aligned", "hold_events": 10},
                {"type": "stop_loss_take_profit", "stop_loss_bps": 5, "take_profit_bps": 20},
            ],
        })
        assert isinstance(policy, CompositePolicy)
        assert len(policy.policies) == 2

    def test_default_type_is_horizon_aligned(self):
        """Missing 'type' key defaults to horizon_aligned."""
        policy = create_holding_policy({})
        assert isinstance(policy, HorizonAlignedPolicy)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown holding policy type"):
            create_holding_policy({"type": "quantum_exit"})

    def test_round_trip_to_dict_from_dict(self):
        """to_dict() → create_holding_policy() preserves behavior."""
        original = CompositePolicy(
            [HorizonAlignedPolicy(10), StopLossTakeProfitPolicy(5, 20, 60)],
            mode="any",
        )
        rebuilt = create_holding_policy(original.to_dict())
        state = _make_state(events_held=10)
        assert original.should_exit(state) == rebuilt.should_exit(state)


class TestDirectionReversalLabelMapping:
    """Phase 2a: DirectionReversalPolicy with LabelMapping."""

    def test_direction_reversal_with_signed_labels(self):
        """Works with SIGNED_MAPPING: -1=Down, 0=Stable, +1=Up."""
        from lobbacktest.labels import SIGNED_MAPPING
        policy = DirectionReversalPolicy(
            max_hold_events=60,
            label_mapping=SIGNED_MAPPING,
        )
        # Down(-1) → Up(+1) IS a reversal
        state = HoldingState(
            events_held=5,
            entry_prediction=-1,  # Down (signed)
            current_prediction=1,  # Up (signed)
            current_agreement=1.0,
            current_confirmation=0.5,
            current_spread=0.5,
            entry_price=100.0,
            current_price=105.0,
            unrealized_pnl_bps=500.0,
            position_side=-1,
        )
        assert policy.should_exit(state) is True, (
            "Signed Down(-1)→Up(+1) should trigger exit (reversal)"
        )

    def test_direction_reversal_non_reversal_cases(self):
        """Stable→Up and Up→Stable are NOT reversals."""
        from lobbacktest.labels import SHIFTED_MAPPING
        policy = DirectionReversalPolicy(
            max_hold_events=60,
            label_mapping=SHIFTED_MAPPING,
        )

        # Up(2) → Stable(1) is NOT a reversal
        state = HoldingState(
            events_held=5,
            entry_prediction=2,   # Up
            current_prediction=1,  # Stable
            current_agreement=1.0,
            current_confirmation=0.5,
            current_spread=0.5,
            entry_price=100.0,
            current_price=102.0,
            unrealized_pnl_bps=200.0,
            position_side=1,
        )
        assert policy.should_exit(state) is False, (
            "Up→Stable is NOT a reversal, should NOT trigger exit"
        )
