"""
Wave 1A F2 + Wave 2-H H3 closure tests (2026-05-17).

Locks fail-loud `ValueError` raise on `ZeroDteConfig(prefer_calls=False)`.

The option-P&L formula at `engine/zero_dte.py:354-375` hardcodes ATM-call-
like delta sign; selecting PUT spread cost via the `is_call` flag is
inconsistent with the P&L direction formula. Pre-fix:
- `is_call = entry_trade.side.value < 0` → True for SELL entry (puts)
- `direction = 1 if entry_trade.side.value > 0 else -1` → -1 for SELL (puts)
- `gross_pnl = delta * (move_bps / 10000.0) * entry_price * 100 * contracts`
  uses `delta = self.config.delta` (positive scalar, e.g., 0.50)

For real PUT options, delta should be NEGATIVE (≈ -0.95 for deep ITM put).
The code treats it as positive, multiplying through the `direction = -1`
sign flip. Net effect: PUT P&L gets the right sign BY ACCIDENT (negative ×
negative) but the MAGNITUDE is WRONG because true |Δ_put| can differ
materially from |Δ_call| at the same strike.

Currently latent (all production YAMLs + test fixtures use
prefer_calls=True per pre-impl Agent Y grep), but reachable via Python
API. Raising at construction prevents silent-wrong-result exposure per
hft-rules §5 fail-fast + §8 never silently produce incoherent semantics.

See `#PY-311` for full PUT delta sign-convention plumbing (Phase Z
architectural, ~4-6 hr; deferred).
"""

import pytest

from lobbacktest.config import OpraCalibratedCosts, ZeroDteConfig


class TestZeroDteConfigPreferCallsFalseRaises:
    """Fail-loud on prefer_calls=False per Wave 1A F2 + Wave 2-H H3."""

    def test_prefer_calls_false_raises_value_error(self):
        """ZeroDteConfig(prefer_calls=False) → ValueError at construction."""
        with pytest.raises(ValueError, match=r"prefer_calls=False"):
            ZeroDteConfig(prefer_calls=False)

    def test_prefer_calls_false_error_message_cites_PY311(self):
        """Error message references #PY-311 backlog entry for traceability."""
        with pytest.raises(ValueError) as exc_info:
            ZeroDteConfig(prefer_calls=False)
        assert "#PY-311" in str(exc_info.value)

    def test_prefer_calls_false_error_mentions_phase_z(self):
        """Error message identifies the architectural deferral path."""
        with pytest.raises(ValueError) as exc_info:
            ZeroDteConfig(prefer_calls=False)
        msg = str(exc_info.value)
        assert "Phase Z" in msg or "deferred" in msg.lower()

    def test_prefer_calls_true_default_works(self):
        """Default (prefer_calls=True) constructs cleanly (pre-existing)."""
        config = ZeroDteConfig()
        assert config.prefer_calls is True

    def test_prefer_calls_true_explicit_works(self):
        """Explicit prefer_calls=True works (pre-existing behavior)."""
        config = ZeroDteConfig(prefer_calls=True)
        assert config.prefer_calls is True

    def test_other_params_with_calls_true_works(self):
        """Other validation still fires correctly with prefer_calls=True."""
        config = ZeroDteConfig(
            enabled=True,
            delta=0.95,  # Deep ITM
            opra_costs=OpraCalibratedCosts.deep_itm(),
            prefer_calls=True,
        )
        assert config.delta == 0.95
        assert config.prefer_calls is True

    def test_prefer_calls_false_with_other_valid_params_still_raises(self):
        """prefer_calls=False raises even if other params are valid."""
        with pytest.raises(ValueError, match=r"prefer_calls=False"):
            ZeroDteConfig(
                enabled=True,
                delta=0.50,
                opra_costs=OpraCalibratedCosts(),
                prefer_calls=False,
            )

    def test_prefer_calls_false_allowed_under_bsm(self):
        """B3/B4 (2026-06-19): under payoff_model='bsm', the put-block is LIFTED.
        A real BSM put value is coherent for prefer_calls=False, so the
        linear-delta sign-incoherence that motivated the raise no longer applies.
        The raise remains in force for the linear_delta default (above)."""
        config = ZeroDteConfig(enabled=True, payoff_model="bsm", prefer_calls=False)
        assert config.prefer_calls is False
        assert config.payoff_model == "bsm"
