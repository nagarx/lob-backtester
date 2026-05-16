"""
Tests for ReadabilityHybridStrategy.

Tests verify:
- #PY-71 NaN guard discipline (2026-05-15) — fail-closed on NaN input across
  all 4 gates (agreement, confidence, spread, predicted_returns).

Pre-fix: `value <op> threshold` evaluated False on NaN input (IEEE 754
NaN-comparison invariant), allowing NaN signals to PASS gates silently and
trigger trades on garbage. Per hft-rules §8 fail-closed.
"""

import numpy as np
import pytest

from lobbacktest.strategies.base import Signal
from lobbacktest.strategies.hybrid import (
    ReadabilityHybridConfig,
    ReadabilityHybridStrategy,
)
from lobbacktest.strategies.holding import HorizonAlignedPolicy


def _make_hybrid_data(n: int = 20):
    """Create test data for ReadabilityHybridStrategy with controlled gate inputs."""
    predictions = np.full(n, 2, dtype=np.int32)  # All Up (shifted)
    agreement = np.ones(n, dtype=np.float64)
    confirmation = np.full(n, 0.66, dtype=np.float64)
    predicted_returns = np.full(n, 8.0, dtype=np.float64)
    spreads = np.full(n, 0.8, dtype=np.float64)
    prices = np.linspace(100, 101, n)
    return predictions, agreement, confirmation, predicted_returns, spreads, prices


class TestHybridNaNGuards:
    """Tests for #PY-71 NaN guard discipline (2026-05-15)."""

    def test_nan_agreement_rejected(self):
        """NaN agreement_ratio must REJECT entry (hybrid.py:112)."""
        predictions, agreement, confirmation, predicted_returns, spreads, prices = _make_hybrid_data(n=10)
        agreement[0] = np.nan
        config = ReadabilityHybridConfig(min_agreement=1.0, min_confidence=0.65, min_return_bps=5.0)
        strategy = ReadabilityHybridStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, predicted_returns=predicted_returns,
            spreads=spreads, prices=prices, config=config,
            holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.HOLD, (
            f"NaN agreement should fail-closed; got {output.signals[0]}"
        )

    def test_nan_confidence_rejected(self):
        """NaN confirmation_score must REJECT entry (hybrid.py:114)."""
        predictions, agreement, confirmation, predicted_returns, spreads, prices = _make_hybrid_data(n=10)
        confirmation[0] = np.nan
        config = ReadabilityHybridConfig(min_agreement=1.0, min_confidence=0.65, min_return_bps=5.0)
        strategy = ReadabilityHybridStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, predicted_returns=predicted_returns,
            spreads=spreads, prices=prices, config=config,
            holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.HOLD, (
            f"NaN confidence should fail-closed; got {output.signals[0]}"
        )

    def test_nan_spread_rejected(self):
        """NaN spread must REJECT entry (hybrid.py:117)."""
        predictions, agreement, confirmation, predicted_returns, spreads, prices = _make_hybrid_data(n=10)
        spreads[0] = np.nan
        config = ReadabilityHybridConfig(max_spread_bps=1.0, min_return_bps=5.0)
        strategy = ReadabilityHybridStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, predicted_returns=predicted_returns,
            spreads=spreads, prices=prices, config=config,
            holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.HOLD, (
            f"NaN spread should fail-closed; got {output.signals[0]}"
        )

    def test_nan_predicted_returns_rejected(self):
        """NaN predicted_returns must REJECT entry (hybrid.py:121 — NEW Agent J).

        Sister of FIND-046 (regression.py:99). Was missed by original #PY-71
        scope but added per pre-impl adversarial gate Agent J SHIP-BLOCKER #5.
        """
        predictions, agreement, confirmation, predicted_returns, spreads, prices = _make_hybrid_data(n=10)
        predicted_returns[0] = np.nan
        config = ReadabilityHybridConfig(min_return_bps=5.0)
        strategy = ReadabilityHybridStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, predicted_returns=predicted_returns,
            spreads=spreads, prices=prices, config=config,
            holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        assert output.signals[0] == Signal.HOLD, (
            f"NaN predicted_returns should fail-closed; got {output.signals[0]}"
        )


class TestHybridGateConsolidation:
    """Regression locks for the 2026-05-16 dead-gate consolidation cycle.

    Pre-cycle: ``_check_entry_gate`` was DEAD CODE (defined at hybrid.py:110-134
    with #PY-71 NaN guards) — ``generate_signals`` used INLINE checks at
    L203-209 that DUPLICATED the method's logic but (a) lacked NaN guards
    (relied on IEEE 754 fail-closed COINCIDENCE — `NaN <op> x` returns False),
    and (b) lacked the ``max_spread_bps > 0`` SENTINEL.

    Post-cycle: ``generate_signals`` calls ``_check_readability_gate(i)`` +
    ``_check_magnitude_gate(i)`` (the 2-method split per pre-impl Agent 2's
    Option C). The composing ``_check_entry_gate`` is back-compat for
    callers expecting the merged-boolean shape (e.g., readability.py:225
    sister convention).

    Tests verify:
      * ``max_spread_bps <= 0`` sentinel now honored uniformly (previously
        the inline path would reject all trades when max_spread_bps=0)
      * Metadata counters ``n_readability_pass`` / ``n_magnitude_pass`` /
        ``n_both_pass`` remain separately tracked
      * The composing ``_check_entry_gate(i)`` returns
        ``_check_readability_gate(i) AND _check_magnitude_gate(i)``
    """

    def test_max_spread_bps_zero_disables_spread_gate(self):
        """``max_spread_bps <= 0`` sentinel: spread filter DISABLED, trades pass.

        Pre-consolidation hidden bug: inline path at hybrid.py:206 did
        `spreads[i] <= max_spread_bps` — when max_spread_bps=0 and spreads[i]>0,
        this is False → reject. The method path at hybrid.py:125 had
        `if max_spread_bps > 0:` guard skipping the check entirely.

        Post-consolidation both paths honor the sentinel. This test locks the
        method-path behavior as the canonical one.
        """
        predictions, agreement, confirmation, predicted_returns, spreads, prices = _make_hybrid_data(n=10)
        # Set spread well above any realistic threshold to verify the sentinel
        # disables the gate (not just makes it pass)
        spreads[:] = 50.0  # 50 bps (huge)
        config = ReadabilityHybridConfig(
            min_agreement=1.0,
            min_confidence=0.65,
            min_return_bps=5.0,
            max_spread_bps=0.0,  # sentinel: disable spread gate
        )
        strategy = ReadabilityHybridStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, predicted_returns=predicted_returns,
            spreads=spreads, prices=prices, config=config,
            holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        # First signal should be BUY (Up direction × all-gates-pass-when-spread-disabled)
        # NOT HOLD (which would mean the spread gate rejected)
        assert output.signals[0] == Signal.BUY, (
            f"max_spread_bps=0 should DISABLE spread gate per method-path sentinel; "
            f"got signal={output.signals[0]} (expected BUY). Pre-consolidation "
            f"inline path would reject (spread 50 > 0)."
        )

    def test_magnitude_gate_counter_separate_from_readability(self):
        """Metadata counters preserved post-2-method-split.

        ``generate_signals`` must track ``n_readability_pass`` and
        ``n_magnitude_pass`` SEPARATELY so operators can diagnose which gate
        dominates rejection. Post-consolidation, ``_check_readability_gate``
        and ``_check_magnitude_gate`` are called independently — failure of
        one does NOT short-circuit the other counter.
        """
        predictions, agreement, confirmation, predicted_returns, spreads, prices = _make_hybrid_data(n=10)
        # First half: readability passes, magnitude FAILS (low predicted_returns)
        predicted_returns[:5] = 1.0  # below min_return_bps=5.0
        # Second half: readability FAILS (low agreement), magnitude passes
        agreement[5:] = 0.5  # below min_agreement=1.0
        config = ReadabilityHybridConfig(
            min_agreement=1.0,
            min_confidence=0.65,
            min_return_bps=5.0,
            max_spread_bps=10.0,  # generous, doesn't reject anything
        )
        strategy = ReadabilityHybridStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, predicted_returns=predicted_returns,
            spreads=spreads, prices=prices, config=config,
            holding_policy=HorizonAlignedPolicy(5),
        )
        output = strategy.generate_signals(prices)
        meta = output.metadata
        # First 5: readability_pass=5 (agreement+confidence+spread all OK),
        #          magnitude_pass=0 (1.0 < 5.0)
        # Last 5: readability_pass=0 (agreement 0.5 < 1.0),
        #         magnitude_pass=5 (8.0 >= 5.0)
        assert meta["n_readability_pass"] == 5, (
            f"Expected 5 readability passes (first half); got {meta['n_readability_pass']}"
        )
        assert meta["n_magnitude_pass"] == 5, (
            f"Expected 5 magnitude passes (last half); got {meta['n_magnitude_pass']}"
        )
        assert meta["n_both_pass"] == 0, (
            f"No event passes BOTH gates; got n_both_pass={meta['n_both_pass']}"
        )

    def test_check_entry_gate_composes_readability_and_magnitude(self):
        """The composing ``_check_entry_gate`` returns AND of split methods.

        Back-compat lock for callers expecting merged-boolean (e.g., the
        readability.py:225 sister convention in the wider strategy framework).
        """
        predictions, agreement, confirmation, predicted_returns, spreads, prices = _make_hybrid_data(n=4)
        # Event 0: BOTH gates pass
        # Event 1: readability fails (low agreement)
        agreement[1] = 0.5
        # Event 2: magnitude fails (low predicted_returns)
        predicted_returns[2] = 1.0
        # Event 3: both fail
        agreement[3] = 0.5
        predicted_returns[3] = 1.0

        config = ReadabilityHybridConfig(
            min_agreement=1.0, min_confidence=0.65, min_return_bps=5.0,
            max_spread_bps=10.0,
        )
        strategy = ReadabilityHybridStrategy(
            predictions=predictions, agreement_ratio=agreement,
            confirmation_score=confirmation, predicted_returns=predicted_returns,
            spreads=spreads, prices=prices, config=config,
            holding_policy=HorizonAlignedPolicy(5),
        )

        # Event 0: True AND True → True
        assert strategy._check_entry_gate(0) is True
        assert strategy._check_readability_gate(0) is True
        assert strategy._check_magnitude_gate(0) is True

        # Event 1: False AND True → False (readability fails)
        assert strategy._check_entry_gate(1) is False
        assert strategy._check_readability_gate(1) is False
        assert strategy._check_magnitude_gate(1) is True

        # Event 2: True AND False → False (magnitude fails)
        assert strategy._check_entry_gate(2) is False
        assert strategy._check_readability_gate(2) is True
        assert strategy._check_magnitude_gate(2) is False

        # Event 3: False AND False → False (both fail)
        assert strategy._check_entry_gate(3) is False
        assert strategy._check_readability_gate(3) is False
        assert strategy._check_magnitude_gate(3) is False
