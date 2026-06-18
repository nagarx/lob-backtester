"""Black-Scholes-Merton option valuation for the 0DTE / same-day options overlay.

Greenfield (Phase 4 B2, 2026-06-19) — the first option *valuation* code in
lob-backtester (the overlay previously used a linear-delta proxy + an ATM-only
theta bolt-on). Formulas mirror the Rust SSoT
``opra-statistical-profiler/src/options_math/bsm.rs`` (the q=0 / no-dividend
Black-Scholes case; NVDA's dividend yield is ~0 at intraday tau, so q=0 is the
documented assumption here — extend to the cost-of-carry form b=r-q if a
dividend-paying or futures underlying is ever priced).

Inputs:
    S     underlying price
    K     strike
    tau   time to expiry, in YEARS (tau_exit < tau_entry over a hold = theta)
    r     risk-free rate, annualized continuous
    sigma annualized implied volatility

Returns option VALUE per share (multiply by 100 * contracts for dollars).

Guards (mirror bsm.rs): tau < MIN_T, sigma < MIN_IV, or S<=0 / K<=0 -> intrinsic
fallback (never divide by sigma*sqrt(tau)~0). The PUT value is floored at
intrinsic, ``max(European_put, K-S)`` — the American / tradeable mark for
US-listed equity options, since a European put can sit BELOW intrinsic (no early
exercise) but a tradeable American put cannot. The European CALL is >= intrinsic
for q=0, so its floor is a no-op.

References:
    Black & Scholes 1973; Merton 1973. See
    hft-wiki/research/theory/black_scholes_merton_pricing_greeks.md and the
    bsm.rs SSoT.
"""

from __future__ import annotations

import math

MIN_T: float = 1e-6  # ~5.9 trading-seconds expressed in years; below -> intrinsic
MIN_IV: float = 1e-6
_INV_SQRT_2PI: float = 0.3989422804014327  # 1/sqrt(2*pi) = N'(0)
_SQRT_2: float = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF N(x) = 0.5 * erfc(-x / sqrt(2)) (machine precision)."""
    return 0.5 * math.erfc(-x / _SQRT_2)


def _norm_pdf(x: float) -> float:
    """Standard normal PDF N'(x) = exp(-x^2 / 2) / sqrt(2*pi)."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _d1_d2(S: float, K: float, tau: float, r: float, sigma: float) -> tuple[float, float]:
    sqrt_t = math.sqrt(tau)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def _degenerate(S: float, K: float, tau: float, sigma: float) -> bool:
    return S <= 0.0 or K <= 0.0 or tau < MIN_T or sigma < MIN_IV


def bs_value(is_call: bool, S: float, K: float, tau: float, r: float, sigma: float) -> float:
    """BSM option value per share (q=0).

    Intrinsic fallback on degenerate inputs; put floored at intrinsic (the
    American-put mark). Call floor is a no-op (European call >= intrinsic at q=0).
    """
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if _degenerate(S, K, tau, sigma):
        return intrinsic
    d1, d2 = _d1_d2(S, K, tau, r, sigma)
    disc = math.exp(-r * tau)
    if is_call:
        val = S * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
    else:
        val = K * disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    return max(val, intrinsic)


def bs_call(S: float, K: float, tau: float, r: float, sigma: float) -> float:
    return bs_value(True, S, K, tau, r, sigma)


def bs_put(S: float, K: float, tau: float, r: float, sigma: float) -> float:
    return bs_value(False, S, K, tau, r, sigma)


def bs_delta(is_call: bool, S: float, K: float, tau: float, r: float, sigma: float) -> float:
    """dV/dS. Degenerate -> the intrinsic slope (ITM=±1, OTM=0)."""
    if _degenerate(S, K, tau, sigma):
        if is_call:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1, _ = _d1_d2(S, K, tau, r, sigma)
    return _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0


def bs_gamma(S: float, K: float, tau: float, r: float, sigma: float) -> float:
    """d2V/dS2 (identical for calls and puts; > 0 = the convexity term)."""
    if _degenerate(S, K, tau, sigma):
        return 0.0
    d1, _ = _d1_d2(S, K, tau, r, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(tau))


def bs_vega(S: float, K: float, tau: float, r: float, sigma: float) -> float:
    """dV/dsigma (FIRST derivative; identical for calls and puts; > 0)."""
    if _degenerate(S, K, tau, sigma):
        return 0.0
    d1, _ = _d1_d2(S, K, tau, r, sigma)
    return S * math.sqrt(tau) * _norm_pdf(d1)


__all__ = [
    "MIN_T",
    "MIN_IV",
    "bs_value",
    "bs_call",
    "bs_put",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
]
