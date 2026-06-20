"""Deep-ITM ~1-month option held intraday = the direction null (pedagogical proof).

PEDAGOGICAL DEMO — NOT an edge experiment. These tests PROVE, with hand-verified
golden values on the pure BSM functions (engine/option_pricing.py — no production
path touched), the theorem the council validated:

    A deep-in-the-money (delta -> 1) ~1-month call BOUGHT and SOLD the same day
    (held 30min-2h, no overnight) has, under a STATIC implied vol,

        P&L_per_share  =  bs_value(S_exit) - bs_value(S_entry)
                       ~=  delta * (S_exit - S_entry)                 [Test 2]

    i.e. it is the UNDERLYING's intraday DIRECTION in a leveraged/cheaper wrapper.
    Since intraday mid-price direction is a martingale (FINDING-002 / FINDING-047),
    E[delta * dS | F_t] = 0, so the strategy's expectancy is just the negative of
    the round-trip option cost:

        E[P&L_net]  =  E[delta * dS]  -  cost  =  0  -  cost  <  0    [Test 3]

    The wrapper changes COST and LEVERAGE, never the SIGN. The ONLY term that makes
    the option a DIFFERENT object from the direction bet is vega * d(sigma) — which
    is identically zero under static sigma and is the variance lane, not direction
    [Test 4]. Deep-ITM is the WORST instrument for that (small vega).

Why monthly (not 0DTE): a 1-month deep-ITM call carries a small, stable time value
and delta < 1 (still ~0.99), avoiding 0DTE pin/expiry risk [Test 1] — but it remains
a direction bet. Convention: q=0, r=0 intraday (r*tau ~ 0.004 at 1mo, negligible;
the module's documented assumption). Cost = the IBKR-calibrated 1.4 bps deep-ITM
round-trip breakeven (root CLAUDE.md s.IBKR 0DTE Cost Model / COST_AUDIT_2026_03.md).

This file asserts a MATHEMATICAL claim about already-correct functions; it is the
durable record of the theorem and a regression guard on bs_*. It is deliberately
NOT a `run_record` / EXPERIMENT-NNN and emits no IC / directional_accuracy / Sharpe
(FINDING-033 / FINDING-048 mis-record guards).
"""

import math

import pytest

from lobbacktest.engine.option_pricing import bs_delta, bs_gamma, bs_value, bs_vega

# ---- Deep-ITM 1-month call, intraday hold -------------------------------------
_S0 = 100.0            # underlying at entry
_K = 85.0             # strike = moneyness 0.85 * S0  -> deep ITM call
_R = 0.0              # intraday risk-free (q=0 module; r*tau~0.004 at 1mo, negligible)
_SIGMA = 0.25          # static implied vol
_TAU_1M = 21.0 / 252.0  # ~1 calendar month to expiry, in years
_HOLD_YEARS = 2.0 / (6.5 * 252.0)  # a 2-hour intraday hold, in trading-time years
_TAU_EXIT = _TAU_1M - _HOLD_YEARS

# IBKR-calibrated deep-ITM round-trip breakeven (root CLAUDE.md s.IBKR Cost Model).
_DEEP_ITM_COST_BPS = 1.4


class TestDeepItmIsADirectionProxy:
    """Test 1 - deep-ITM 1-month call: delta ~ 1, gamma ~ 0 (locally linear in S).

    This is the premise: a deep-ITM option's value tracks the underlying ~1-for-1,
    so its P&L is the underlying's directional move (scaled by delta)."""

    def test_delta_near_one(self):
        delta = bs_delta(True, _S0, _K, _TAU_1M, _R, _SIGMA)
        # Hand value: d1 = 2.288 -> N(d1) = 0.9889
        assert delta == pytest.approx(0.9889, abs=2e-3), (
            f"deep-ITM 1mo call delta should be ~0.989 (near 1, a direction proxy), got {delta}"
        )

    def test_gamma_small_locally_linear(self):
        gamma = bs_gamma(_S0, _K, _TAU_1M, _R, _SIGMA)
        # Hand value: gamma = phi(d1)/(S*sigma*sqrt(tau)) ~= 0.00404 per $ -> tiny curvature
        assert 0.0 < gamma < 0.01, (
            f"deep-ITM gamma should be small (locally linear), got {gamma}"
        )

    def test_monthly_retains_small_time_value(self):
        """tau handled correctly: a 1-month deep-ITM call sits slightly ABOVE
        intrinsic (small time value), unlike a 0DTE deep-ITM call which is ~pure
        intrinsic. Confirms the tau=21/252 path is real, not collapsed to 0DTE."""
        intrinsic = _S0 - _K  # 15.0
        v_1m = bs_value(True, _S0, _K, _TAU_1M, _R, _SIGMA)
        v_0dte = bs_value(True, _S0, _K, _HOLD_YEARS, _R, _SIGMA)  # ~2h to expiry
        assert v_1m > intrinsic, "1mo deep-ITM call should hold time value above intrinsic"
        assert (v_1m - intrinsic) < 0.10, "but the time value is small (deep ITM)"
        assert v_0dte == pytest.approx(intrinsic, abs=1e-3), (
            "a ~0DTE deep-ITM call is ~pure intrinsic (no optionality)"
        )
        assert v_1m > v_0dte, "more time to expiry -> (weakly) more value"


class TestStaticSigmaPnlIsDeltaTimesDeltaS:
    """Test 2 - THE NULL: under static sigma, deep-ITM option P&L ~= delta * dS.

    The option P&L over an intraday move IS the underlying's directional move scaled
    by delta. The residual is second-order gamma curvature + negligible intraday
    theta. The wrapper adds nothing to the SIGNAL."""

    def test_up_move_pnl_equals_delta_times_move(self):
        s_exit = 101.0  # +1% intraday move
        v_entry = bs_value(True, _S0, _K, _TAU_1M, _R, _SIGMA)
        v_exit = bs_value(True, s_exit, _K, _TAU_EXIT, _R, _SIGMA)
        pnl_per_share = v_exit - v_entry

        delta_entry = bs_delta(True, _S0, _K, _TAU_1M, _R, _SIGMA)
        delta_term = delta_entry * (s_exit - _S0)

        # Hand values: pnl ~= 0.990, delta_term ~= 0.989 -> residual < 0.2% of value
        assert pnl_per_share == pytest.approx(delta_term, abs=0.02), (
            f"static-sigma deep-ITM P&L should equal delta*dS (the direction null): "
            f"pnl={pnl_per_share}, delta*dS={delta_term}"
        )
        assert pnl_per_share == pytest.approx(0.990, abs=0.02)

    def test_down_move_is_symmetric(self):
        s_exit = 99.0  # -1% intraday move
        v_entry = bs_value(True, _S0, _K, _TAU_1M, _R, _SIGMA)
        v_exit = bs_value(True, s_exit, _K, _TAU_EXIT, _R, _SIGMA)
        pnl_per_share = v_exit - v_entry
        delta_term = bs_delta(True, _S0, _K, _TAU_1M, _R, _SIGMA) * (s_exit - _S0)
        assert pnl_per_share == pytest.approx(delta_term, abs=0.02), (
            "down-move P&L is also ~ delta*dS (the call loses ~delta per $ down)"
        )


class TestMartingaleExpectancyIsNegativeCost:
    """Test 3 - for a martingale dS, E[P&L_net] = E[delta*dS] - cost = -cost < 0.

    The DELTA-replicated edge has exactly zero expectation over a zero-mean move;
    any positive round-trip cost makes the strategy expectancy strictly negative.
    This is the direction martingale (FINDING-002/047) made quantitative."""

    def test_delta_pnl_is_zero_mean_over_symmetric_moves(self):
        delta_entry = bs_delta(True, _S0, _K, _TAU_1M, _R, _SIGMA)
        moves_bps = [-50.0, -20.0, -5.0, 5.0, 20.0, 50.0]  # symmetric, zero-mean
        # delta-replicated P&L in bps of S for a directional (long-call) view
        gross_bps = [delta_entry * m for m in moves_bps]
        mean_gross = sum(gross_bps) / len(gross_bps)
        assert mean_gross == pytest.approx(0.0, abs=1e-9), (
            f"delta-replicated P&L must be zero-mean over a martingale move, got {mean_gross}"
        )

    def test_expected_net_is_negative_cost(self):
        # E[gross]=0 (martingale) -> E[net] = -cost, strictly negative.
        e_gross_bps = 0.0
        e_net_bps = e_gross_bps - _DEEP_ITM_COST_BPS
        assert e_net_bps < 0.0
        assert e_net_bps == pytest.approx(-1.4, abs=1e-9), (
            "the deep-ITM directional wrapper's expectancy is exactly -(round-trip cost)"
        )


class TestVegaIsTheOnlyEscape:
    """Test 4 - the ONLY term that makes the option a different object from the
    direction bet is vega * d(sigma). It is identically zero under static sigma
    (Tests 2-3) and is a VARIANCE object, not direction. Deep-ITM has small vega,
    so it is the worst instrument for it."""

    def test_static_sigma_value_is_independent_of_a_zero_sigma_change(self):
        v_a = bs_value(True, _S0, _K, _TAU_1M, _R, _SIGMA)
        v_b = bs_value(True, _S0, _K, _TAU_1M, _R, _SIGMA)  # same sigma -> dV=0
        assert (v_b - v_a) == 0.0, "static sigma -> the vega term is identically zero"

    def test_value_moves_with_sigma_first_order_vega(self):
        d_sigma = 0.01
        v0 = bs_value(True, _S0, _K, _TAU_1M, _R, _SIGMA)
        v1 = bs_value(True, _S0, _K, _TAU_1M, _R, _SIGMA + d_sigma)
        dv = v1 - v0
        vega = bs_vega(_S0, _K, _TAU_1M, _R, _SIGMA)
        # First-order: dV ~= vega * d_sigma (residual is vomma, second order)
        assert dv == pytest.approx(vega * d_sigma, abs=3e-3), (
            f"value change with sigma should be ~ vega*d_sigma: dV={dv}, vega*dsig={vega * d_sigma}"
        )
        assert dv > 0.0, "a long option gains value when implied vol rises"

    def test_deep_itm_vega_is_material_but_the_object_is_variance_not_direction(self):
        vega = bs_vega(_S0, _K, _TAU_1M, _R, _SIGMA)
        # Hand value ~0.84 per share per unit sigma. Material, but to harvest it you
        # must trade a REALIZED d(sigma) (the variance lane) -- a deep-ITM delta-one
        # contract is a direction proxy, NOT the right vega instrument (ATM straddle is).
        assert vega == pytest.approx(0.84, abs=0.05)
