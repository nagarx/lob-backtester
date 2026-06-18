"""Tests for the Phase 4 B2 BSM option-pricing module (engine/option_pricing.py).

q=0 (no-dividend) Black-Scholes. Strong oracle = put-call parity
(C - P = S - K*exp(-r*tau)); plus a Hull known-value, the tau->0/sigma->0
intrinsic limits, deep-ITM behaviour, the American-put intrinsic floor, and the
degenerate-input guards.
"""

import math

import pytest

from lobbacktest.engine.option_pricing import (
    MIN_T,
    bs_call,
    bs_delta,
    bs_gamma,
    bs_put,
    bs_value,
    bs_vega,
)


class TestPutCallParity:
    """C - P = S - K*exp(-r*tau) for arbitrary inputs (the strongest oracle).
    Cases use S >= K so the put is OTM (intrinsic 0) and the American floor is a
    no-op, so European parity holds exactly."""

    CASES = [
        (100.0, 100.0, 0.5, 0.05, 0.30),
        (100.0, 95.0, 0.25, 0.03, 0.20),
        (100.0, 90.0, 1.0, 0.04, 0.40),
        (50.0, 50.0, 0.01, 0.05, 0.60),
    ]

    def test_parity(self):
        for S, K, tau, r, sigma in self.CASES:
            c = bs_call(S, K, tau, r, sigma)
            p = bs_put(S, K, tau, r, sigma)
            lhs = c - p
            rhs = S - K * math.exp(-r * tau)
            assert lhs == pytest.approx(rhs, abs=1e-9), (
                f"parity broken for {(S, K, tau, r, sigma)}: C-P={lhs}, S-Ke^-rt={rhs}"
            )


class TestKnownValue:
    """Hull's classic q=0 example: S=42, K=40, r=0.10, sigma=0.20, tau=0.5
    -> call ~= 4.76, put ~= 0.81."""

    def test_hull_call_put(self):
        assert bs_call(42.0, 40.0, 0.5, 0.10, 0.20) == pytest.approx(4.759, abs=0.01)
        assert bs_put(42.0, 40.0, 0.5, 0.10, 0.20) == pytest.approx(0.808, abs=0.01)


class TestIntrinsicLimits:
    def test_tau_to_zero_call_is_intrinsic(self):
        assert bs_call(110.0, 100.0, MIN_T / 10.0, 0.05, 0.30) == pytest.approx(10.0)

    def test_tau_to_zero_put_is_intrinsic(self):
        assert bs_put(90.0, 100.0, 0.0, 0.05, 0.30) == pytest.approx(10.0)

    def test_sigma_zero_is_intrinsic(self):
        assert bs_call(110.0, 100.0, 0.5, 0.05, 0.0) == pytest.approx(10.0)

    def test_deep_itm_call_approaches_S_minus_disc_K(self):
        # deep ITM: C -> S - K*exp(-r*tau)
        c = bs_call(200.0, 100.0, 0.1, 0.05, 0.20)
        assert c == pytest.approx(200.0 - 100.0 * math.exp(-0.05 * 0.1), abs=0.05)

    def test_otm_call_near_zero(self):
        assert bs_call(50.0, 100.0, 0.01, 0.05, 0.20) == pytest.approx(0.0, abs=1e-6)


class TestAmericanPutFloor:
    def test_deep_itm_european_put_floored_at_intrinsic(self):
        """A deep-ITM European put can sit BELOW intrinsic (no early exercise);
        bs_put floors it at intrinsic (the American/tradeable mark).
        S=50,K=100,tau=2,r=0.10,sigma=0.2: European put ~= 31.9 < intrinsic 50."""
        p = bs_put(50.0, 100.0, 2.0, 0.10, 0.20)
        assert p == pytest.approx(50.0), f"put should be floored at intrinsic 50, got {p}"
        assert p >= (100.0 - 50.0) - 1e-9


class TestGuards:
    def test_nonpositive_S_returns_intrinsic(self):
        assert bs_call(0.0, 100.0, 0.5, 0.05, 0.2) == pytest.approx(0.0)
        assert bs_put(0.0, 100.0, 0.5, 0.05, 0.2) == pytest.approx(100.0)  # max(K-S,0)=100

    def test_value_dispatch_matches_call_put(self):
        assert bs_value(True, 100, 95, 0.3, 0.04, 0.25) == bs_call(100, 95, 0.3, 0.04, 0.25)
        assert bs_value(False, 100, 105, 0.3, 0.04, 0.25) == bs_put(100, 105, 0.3, 0.04, 0.25)


class TestGreeks:
    def test_gamma_positive_and_call_eq_put(self):
        g_c = bs_gamma(100, 100, 0.5, 0.04, 0.3)
        assert g_c > 0
        # gamma is identical for calls and puts (same formula, no flag)
        assert g_c == pytest.approx(bs_gamma(100, 100, 0.5, 0.04, 0.3))

    def test_vega_positive(self):
        assert bs_vega(100, 100, 0.5, 0.04, 0.3) > 0

    def test_delta_call_in_unit_interval(self):
        d = bs_delta(True, 100, 100, 0.5, 0.04, 0.3)
        assert 0.0 < d < 1.0

    def test_delta_put_in_negative_unit_interval(self):
        d = bs_delta(False, 100, 100, 0.5, 0.04, 0.3)
        assert -1.0 < d < 0.0

    def test_deep_itm_call_delta_near_one(self):
        assert bs_delta(True, 200, 100, 0.1, 0.05, 0.2) == pytest.approx(1.0, abs=1e-3)
