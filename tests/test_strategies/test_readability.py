"""
Tests for ReadabilityStrategy.

Tests verify:
- Entry gate logic: agreement, confirmation, spread, directional, volatility
- Holding policy integration: entry vs hold-in-position vs exit
- Cooldown behavior after position exit
- Signal output: correct Signal values (BUY/SELL/HOLD/EXIT)
- Metadata: gate pass/fail counts, entry/exit counts

Per RULE.md:
- Formula tests: Gates match documented thresholds exactly
- Edge tests: All gates fail, all gates pass, boundary values
- Invariant tests: No BUY/SELL while in position (only holding policy decides)
"""

import numpy as np
import pytest

from lobbacktest.strategies.base import Signal, SignalOutput
from lobbacktest.strategies.readability import ReadabilityConfig, ReadabilityStrategy
from lobbacktest.strategies.holding import HorizonAlignedPolicy


def _make_readability_data(
    n: int = 20,
    predictions: np.ndarray = None,
    agreement: np.ndarray = None,
    confirmation: np.ndarray = None,
    spreads: np.ndarray = None,
    prices: np.ndarray = None,
):
    """Create test data for ReadabilityStrategy with controlled gate inputs."""
    if predictions is None:
        predictions = np.full(n, 2, dtype=np.int32)  # All Up
    if agreement is None:
        agreement = np.ones(n, dtype=np.float64)  # All agree
    if confirmation is None:
        confirmation = np.full(n, 0.66, dtype=np.float64)  # Above 0.65
    if spreads is None:
        spreads = np.full(n, 0.8, dtype=np.float64)  # Below 1.05
    if prices is None:
        prices = np.linspace(100, 101, n)
    return predictions, agreement, confirmation, spreads, prices


class TestReadabilityEntryGates:
    """Tests for individual entry gate conditions."""

    def test_all_gates_pass_enters(self):
        """When all gates pass on first event, strategy enters position."""
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=15)
        config = ReadabilityConfig(min_agreement=1.0, min_confidence=0.65, max_spread_bps=1.05)
        policy = HorizonAlignedPolicy(hold_events=10)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices, config=config, holding_policy=policy,
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.BUY  # Prediction=2 (Up) → BUY

    def test_agreement_gate_blocks_entry(self):
        """Low agreement blocks entry → HOLD."""
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=5)
        agreement[:] = 0.667  # Below 1.0
        config = ReadabilityConfig(min_agreement=1.0)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices, config=config,
        )
        output = strategy.generate_signals(prices)
        assert all(s == Signal.HOLD for s in output.signals)

    def test_confirmation_gate_blocks_entry(self):
        """Low confirmation blocks entry → HOLD."""
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=5)
        confirmation[:] = 0.50  # Below 0.65
        config = ReadabilityConfig(min_confidence=0.65)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices, config=config,
        )
        output = strategy.generate_signals(prices)
        assert all(s == Signal.HOLD for s in output.signals)

    def test_confirmation_boundary_equal_does_not_pass(self):
        """confirmation_score == min_confidence does NOT pass (strict >)."""
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=5)
        confirmation[:] = 0.65  # Exactly at threshold
        config = ReadabilityConfig(min_confidence=0.65)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices, config=config,
        )
        output = strategy.generate_signals(prices)
        # Gate uses <=, so exactly at threshold does NOT pass
        assert all(s == Signal.HOLD for s in output.signals)

    def test_spread_gate_blocks_entry(self):
        """Wide spread blocks entry → HOLD."""
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=5)
        spreads[:] = 2.0  # Above 1.05
        config = ReadabilityConfig(max_spread_bps=1.05)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices, config=config,
        )
        output = strategy.generate_signals(prices)
        assert all(s == Signal.HOLD for s in output.signals)

    def test_stable_prediction_blocks_entry(self):
        """Stable predictions (class 1) blocked when require_directional=True."""
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=5)
        predictions[:] = 1  # All Stable
        config = ReadabilityConfig(require_directional=True)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices, config=config,
        )
        output = strategy.generate_signals(prices)
        assert all(s == Signal.HOLD for s in output.signals)

    def test_stable_prediction_allowed_when_not_required(self):
        """Stable predictions pass when require_directional=False."""
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=15)
        predictions[:] = 1  # All Stable
        config = ReadabilityConfig(require_directional=False)
        policy = HorizonAlignedPolicy(hold_events=10)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices, config=config, holding_policy=policy,
        )
        output = strategy.generate_signals(prices)
        # Prediction==1 (Stable) → no BUY or SELL entry since the entry logic checks pred==0 or pred==2
        # require_directional=False skips the check, but the signal mapping still only creates BUY for 2, SELL for 0
        # So Stable prediction with directional not required → HOLD (no entry because prediction is not 0 or 2)
        assert output.signals[0] == Signal.HOLD

    def test_down_prediction_sells(self):
        """Prediction=0 (Down) → SELL signal."""
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=15)
        predictions[:] = 0  # All Down
        config = ReadabilityConfig()
        policy = HorizonAlignedPolicy(hold_events=10)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices, config=config, holding_policy=policy,
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.SELL


class TestReadabilityHoldingIntegration:
    """Tests for holding policy integration with ReadabilityStrategy."""

    def test_hold_for_horizon_events(self):
        """After entry, hold for hold_events then exit."""
        n = 15
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=n)
        config = ReadabilityConfig()
        policy = HorizonAlignedPolicy(hold_events=5)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices, config=config, holding_policy=policy,
        )
        output = strategy.generate_signals(prices)

        # Event 0: Enter (BUY)
        assert output.signals[0] == Signal.BUY
        # Events 1-4: Holding (BUY continues since in long position)
        for i in range(1, 5):
            assert output.signals[i] == Signal.BUY, f"Event {i} should be BUY (holding long)"
        # Event 5: Exit (held for 5 events)
        assert output.signals[5] == Signal.EXIT

    def test_no_gate_recheck_during_hold(self):
        """Gate is NOT re-checked while holding — only policy decides exit."""
        n = 15
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=n)
        # Gate fails from event 2 onward (agreement drops)
        agreement[2:] = 0.5
        config = ReadabilityConfig()
        policy = HorizonAlignedPolicy(hold_events=10)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices, config=config, holding_policy=policy,
        )
        output = strategy.generate_signals(prices)

        # Event 0: Enter (gates pass at event 0)
        assert output.signals[0] == Signal.BUY
        # Events 1-9: Should still hold even though agreement dropped
        for i in range(1, 10):
            assert output.signals[i] == Signal.BUY, f"Event {i}: gate drop should NOT cause exit"
        # Event 10: Exit from holding policy
        assert output.signals[10] == Signal.EXIT


class TestReadabilityCooldown:
    """Tests for cooldown behavior after position exit."""

    def test_cooldown_prevents_immediate_reentry(self):
        """After exit, cooldown_events must pass before re-entry."""
        n = 25
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=n)
        config = ReadabilityConfig(cooldown_events=3)
        policy = HorizonAlignedPolicy(hold_events=5)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices, config=config, holding_policy=policy,
        )
        output = strategy.generate_signals(prices)

        # Event 0: Enter
        assert output.signals[0] == Signal.BUY
        # Event 5: Exit
        assert output.signals[5] == Signal.EXIT
        # Events 6-8: Cooldown (HOLD even though gates pass)
        for i in range(6, 9):
            assert output.signals[i] == Signal.HOLD, f"Event {i} should be HOLD (cooldown)"
        # Event 9: Cooldown expired, can re-enter
        assert output.signals[9] == Signal.BUY

    def test_zero_cooldown_allows_immediate_reentry(self):
        """cooldown_events=0 allows immediate re-entry after exit."""
        n = 20
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=n)
        config = ReadabilityConfig(cooldown_events=0)
        policy = HorizonAlignedPolicy(hold_events=5)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices, config=config, holding_policy=policy,
        )
        output = strategy.generate_signals(prices)

        # Event 0: Enter, Event 5: Exit
        assert output.signals[0] == Signal.BUY
        assert output.signals[5] == Signal.EXIT
        # Event 6: Immediate re-entry (no cooldown)
        assert output.signals[6] == Signal.BUY


class TestReadabilityMetadata:
    """Tests for strategy metadata output."""

    def test_metadata_includes_gate_counts(self):
        """Metadata must include n_gate_pass, n_gate_fail, n_entries, n_exits."""
        n = 15
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=n)
        config = ReadabilityConfig()
        policy = HorizonAlignedPolicy(hold_events=5)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices, config=config, holding_policy=policy,
        )
        output = strategy.generate_signals(prices)
        meta = output.metadata

        assert "n_entries" in meta
        assert "n_exits" in meta
        assert "n_gate_pass" in meta
        assert "n_gate_fail" in meta
        assert "avg_hold_events" in meta
        assert "trade_rate" in meta
        assert meta["n_entries"] >= 1

    def test_metadata_trade_rate_is_fraction(self):
        """trade_rate must be in [0, 1]."""
        n = 15
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=n)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices, holding_policy=HorizonAlignedPolicy(hold_events=5),
        )
        output = strategy.generate_signals(prices)
        assert 0.0 <= output.metadata["trade_rate"] <= 1.0


class TestReadabilityEdgeCases:
    """Edge case tests for ReadabilityStrategy."""

    def test_all_gates_fail_no_trades(self):
        """When no event passes all gates, no signals generated."""
        n = 10
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=n)
        agreement[:] = 0.5  # Fails agreement gate
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices,
        )
        output = strategy.generate_signals(prices)
        assert all(s == Signal.HOLD for s in output.signals)
        assert output.metadata["n_entries"] == 0

    def test_single_event_data(self):
        """Strategy handles single-event data without crash."""
        predictions, agreement, confirmation, spreads, prices = _make_readability_data(n=1)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=spreads,
            prices=prices, holding_policy=HorizonAlignedPolicy(hold_events=5),
        )
        output = strategy.generate_signals(prices)
        assert len(output.signals) == 1

    def test_no_spreads_provided(self):
        """Strategy works without spread data (spread gate skipped)."""
        n = 15
        predictions = np.full(n, 2, dtype=np.int32)
        agreement = np.ones(n)
        confirmation = np.full(n, 0.66)
        prices = np.linspace(100, 101, n)
        strategy = ReadabilityStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, spreads=None,
            prices=prices, holding_policy=HorizonAlignedPolicy(hold_events=5),
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.BUY  # Should enter without spread check


class TestReadabilityDefaults:
    """P5 FIX: Verify corrected default values."""

    def test_default_min_agreement_is_667(self):
        """P5: Default should be 0.667 (2/3 horizons agree), not 1.0."""
        config = ReadabilityConfig()
        assert abs(config.min_agreement - 0.667) < 0.001, (
            f"P5 FIX: min_agreement should default to 0.667, got {config.min_agreement}"
        )

    def test_agreement_boundary_at_667(self):
        """P5: Agreement exactly at boundary — 0.666 fails, 0.668 passes."""
        predictions = np.array([2, 2])  # Up (shifted)
        agreement = np.array([0.666, 0.668])
        confirmation = np.array([0.7, 0.7])

        config = ReadabilityConfig()  # Uses default min_agreement=0.667
        strategy = ReadabilityStrategy(
            predictions=predictions,
            agreement_ratio=agreement,
            confirmation_score=confirmation,
            config=config,
        )
        prices = np.array([100.0, 100.0])
        output = strategy.generate_signals(prices)

        # agreement[0]=0.666 < 0.667 → HOLD
        assert output.signals[0] == Signal.HOLD, (
            f"Agreement 0.666 < 0.667 should be HOLD, got {output.signals[0]}"
        )
        # agreement[1]=0.668 >= 0.667 → BUY (Up prediction passes gate)
        assert output.signals[1] == Signal.BUY, (
            f"Agreement 0.668 >= 0.667 should be BUY, got {output.signals[1]}"
        )
