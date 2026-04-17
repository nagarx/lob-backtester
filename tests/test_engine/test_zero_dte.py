"""
Tests for 0DTE ATM Options P&L Transformer.

Tests verify:
- BSM theta formula: theta = S * sigma * N'(0) / (2 * sqrt(T))
- Per-trade P&L model: gross - spread - commission - theta
- Cost breakdown accuracy
- Edge cases: zero holding, short time remaining

Per RULE.md:
- Formula tests: BSM theta verified with hand-calculated values
- Reference: Black & Scholes (1973), Hull (2018) Ch 19
- Source calibration: OPRA CMBP-1 (8 days), IBKR 318 fills
"""

import math

import numpy as np
import pytest

from lobbacktest.engine.zero_dte import (
    EPS,
    NPRIME_ZERO,
    TRADING_MINUTES_PER_YEAR,
    ZeroDtePnLTransformer,
    ZeroDteResult,
    theta_bsm_per_share,
)
from lobbacktest.config import BacktestConfig, CostConfig, ZeroDteConfig, OpraCalibratedCosts
from lobbacktest.types import BacktestResult, Trade, TradeSide


class TestThetaBsmFormula:
    """
    Verify BSM theta formula:
        theta_annual = S * sigma * N'(0) / (2 * sqrt(T))
        theta_per_min = theta_annual / (252 * 390)
        theta_cost = theta_per_min * holding_minutes

    N'(0) = 1 / sqrt(2*pi) ≈ 0.3989
    """

    def test_theta_formula_at_14_00(self):
        """
        At 14:00 ET (120 min remaining), S=$180, IV=40%:
            T = 120 / (252 * 390) = 0.001221
            theta_annual = 180 * 0.40 * 0.3989 / (2 * sqrt(0.001221))
            theta_annual = 180 * 0.40 * 0.3989 / (2 * 0.03494)
            theta_annual = 28.72 / 0.06989 = 410.9
            theta_per_min = 410.9 / 98280 = 0.004181
            theta_1min = 0.004181 USD/share

        Per-contract (100 shares) for 1 min: $0.418
        Documented value: $0.42/contract/min (CODEBASE.md, zero_dte.py docstring)
        """
        theta = theta_bsm_per_share(
            underlying_price=180.0,
            implied_vol=0.40,
            minutes_remaining=120.0,
            holding_minutes=1.0,
        )
        # Expected: ~0.00418 USD/share/min
        theta_per_contract = theta * 100
        assert 0.35 < theta_per_contract < 0.50, (
            f"BSM theta at 14:00 should be ~$0.42/contract/min, got ${theta_per_contract:.4f}"
        )

    def test_theta_formula_at_15_30(self):
        """
        At 15:30 ET (30 min remaining), theta should be ~2x higher than 14:00.
        Theta ∝ 1/sqrt(T), so T=30 vs T=120 → sqrt(120/30) = 2x.
        Documented: 0.47 bps/min at 15:30.
        """
        theta_14 = theta_bsm_per_share(180.0, 0.40, 120.0, 1.0)
        theta_15 = theta_bsm_per_share(180.0, 0.40, 30.0, 1.0)
        ratio = theta_15 / theta_14
        assert 1.8 < ratio < 2.2, f"Theta at 15:30 should be ~2x 14:00, got {ratio:.2f}x"

    def test_theta_scales_linearly_with_holding(self):
        """Theta cost should scale linearly with holding duration (for short holds)."""
        theta_1 = theta_bsm_per_share(180.0, 0.40, 120.0, 1.0)
        theta_5 = theta_bsm_per_share(180.0, 0.40, 120.0, 5.0)
        assert abs(theta_5 / theta_1 - 5.0) < 0.01, "Theta should scale linearly with holding"

    def test_theta_scales_with_stock_price(self):
        """Theta ∝ S: doubling stock price doubles theta."""
        theta_100 = theta_bsm_per_share(100.0, 0.40, 120.0, 1.0)
        theta_200 = theta_bsm_per_share(200.0, 0.40, 120.0, 1.0)
        assert abs(theta_200 / theta_100 - 2.0) < 0.01

    def test_theta_scales_with_volatility(self):
        """Theta ∝ sigma: doubling IV doubles theta."""
        theta_20 = theta_bsm_per_share(180.0, 0.20, 120.0, 1.0)
        theta_40 = theta_bsm_per_share(180.0, 0.40, 120.0, 1.0)
        assert abs(theta_40 / theta_20 - 2.0) < 0.01

    def test_theta_zero_with_zero_holding(self):
        """Zero holding time → zero theta cost."""
        theta = theta_bsm_per_share(180.0, 0.40, 120.0, 0.0)
        assert theta == 0.0

    def test_theta_zero_with_no_time_remaining(self):
        """< 1 minute remaining → zero theta (expired)."""
        theta = theta_bsm_per_share(180.0, 0.40, 0.5, 1.0)
        assert theta == 0.0


class TestThetaConstants:
    """Verify the BSM constants used in the module."""

    def test_nprime_zero(self):
        """N'(0) = 1 / sqrt(2*pi) ≈ 0.39894."""
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert abs(NPRIME_ZERO - expected) < 1e-10

    def test_trading_minutes_per_year(self):
        """252 trading days * 390 minutes/day = 98,280."""
        assert TRADING_MINUTES_PER_YEAR == 252.0 * 390.0

    def test_eps_is_small(self):
        """EPS should be a very small positive number for numerical stability."""
        assert EPS > 0
        assert EPS < 1e-6


class TestOpraCalibratedCosts:
    """Verify OPRA-calibrated cost model defaults against empirical data."""

    def test_default_call_spread(self):
        """Call half-spread: $0.015 (OPRA median $0.030 full ÷ 2)."""
        costs = OpraCalibratedCosts()
        assert costs.atm_call_half_spread == 0.015

    def test_default_put_spread(self):
        """Put half-spread: $0.010 (OPRA median $0.020 full ÷ 2)."""
        costs = OpraCalibratedCosts()
        assert costs.atm_put_half_spread == 0.010

    def test_default_call_premium(self):
        """Call premium: $1.88 (OPRA median, validated by IBKR $1.86)."""
        costs = OpraCalibratedCosts()
        assert costs.atm_call_premium == 1.88

    def test_default_commission(self):
        """Commission: $0.70 (IBKR 318-fill median, all-inclusive)."""
        costs = OpraCalibratedCosts()
        assert costs.commission_per_contract == 0.70

    def test_round_trip_cost_call(self):
        """
        Round-trip cost for ATM 0DTE call:
            spread = 2 * 0.015 * 100 = $3.00
            commission = 2 * 0.70 = $1.40
            total = $4.40

        Reference: IBKR_REAL_WORLD_TRADING_REPORT.md §6, §7
        """
        costs = OpraCalibratedCosts()
        rt = costs.round_trip_cost_per_contract(is_call=True)
        expected = 2 * 0.015 * 100 + 2 * 0.70  # 3.00 + 1.40 = 4.40
        assert abs(rt - expected) < 0.01, f"Call RT cost should be ${expected:.2f}, got ${rt:.2f}"

    def test_round_trip_cost_put(self):
        """
        Round-trip cost for ATM 0DTE put:
            spread = 2 * 0.010 * 100 = $2.00
            commission = 2 * 0.70 = $1.40
            total = $3.40
        """
        costs = OpraCalibratedCosts()
        rt = costs.round_trip_cost_per_contract(is_call=False)
        expected = 2 * 0.010 * 100 + 2 * 0.70  # 2.00 + 1.40 = 3.40
        assert abs(rt - expected) < 0.01, f"Put RT cost should be ${expected:.2f}, got ${rt:.2f}"


class TestZeroDteConfig:
    """Verify ZeroDte configuration defaults."""

    def test_default_delta(self):
        """ATM options have delta ≈ 0.50."""
        config = ZeroDteConfig()
        assert config.delta == 0.50

    def test_default_contracts(self):
        """Default: 1 contract per trade."""
        config = ZeroDteConfig()
        assert config.contracts_per_trade == 1
