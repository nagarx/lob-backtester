"""
Phase X.3 Empirical Trust — Phase B regression tests.

Tests the silent-zero → NaN migration for 5 high-leverage backtester metric
classes shipped 2026-05-05 (lob-backtester):
  - WinRate (trading.py): empty trade list → NaN (not 0.0)
  - ProfitFactor (trading.py): empty trade list → NaN
  - SharpeRatio (risk.py): empty/short/zero-std returns → NaN
  - SortinoRatio (risk.py): empty/short returns + no-downside-zero-mean → NaN
  - TotalReturn (returns.py): empty/non-finite returns → NaN

Locks the design — preventing accidental regression to the silent-zero
behavior (F-6 generalization caught by 2026-05-05 multi-agent audit).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lobbacktest.metrics.trading import WinRate, ProfitFactor
from lobbacktest.metrics.risk import SharpeRatio, SortinoRatio
from lobbacktest.metrics.returns import TotalReturn


class TestWinRateUndefined:
    def test_empty_trade_list_returns_nan(self):
        m = WinRate()
        result = m.compute(np.array([0.01, -0.005]), context={"trade_pnls": np.array([])})
        assert math.isnan(result["WinRate"]), (
            "WinRate must return NaN for empty trade list (pre-X.3 returned 0.0 "
            "indistinguishable from legitimate 0% WR)"
        )

    def test_all_losers_returns_zero(self):
        """Legitimate 0% WR — defined value, NOT NaN."""
        m = WinRate()
        result = m.compute(np.array([]), context={"trade_pnls": np.array([-1.0, -2.0, -0.5])})
        assert result["WinRate"] == 0.0, "All-losers IS legitimate 0% WR"

    def test_all_winners_returns_one(self):
        """Legitimate 100% WR — defined value."""
        m = WinRate()
        result = m.compute(np.array([]), context={"trade_pnls": np.array([1.0, 2.0, 0.5])})
        assert result["WinRate"] == 1.0


class TestProfitFactorUndefined:
    def test_empty_trade_list_returns_nan(self):
        m = ProfitFactor()
        result = m.compute(np.array([]), context={"trade_pnls": np.array([])})
        assert math.isnan(result["ProfitFactor"])

    def test_no_losses_with_profits_returns_capped_100(self):
        """Documented sentinel for 'great strategy' (no losses + profits)."""
        m = ProfitFactor()
        result = m.compute(np.array([]), context={"trade_pnls": np.array([1.0, 2.0, 0.5])})
        assert result["ProfitFactor"] == 100.0, "Documented cap preserved"

    def test_no_wins_returns_zero(self):
        m = ProfitFactor()
        result = m.compute(np.array([]), context={"trade_pnls": np.array([-1.0, -2.0])})
        # 0 / |sum(losers)| = 0.0; defined value
        assert result["ProfitFactor"] == 0.0


class TestSharpeRatioUndefined:
    def test_empty_returns_returns_nan(self):
        m = SharpeRatio()
        result = m.compute(np.array([]), context={})
        assert math.isnan(result["SharpeRatio"])

    def test_single_sample_returns_nan(self):
        """len(returns) < 2 means std is undefined."""
        m = SharpeRatio()
        result = m.compute(np.array([0.01]), context={})
        assert math.isnan(result["SharpeRatio"])

    def test_zero_std_constant_returns_nan(self):
        """Flat returns → std=0 → Sharpe undefined (cannot compute ratio)."""
        m = SharpeRatio()
        result = m.compute(np.array([0.01, 0.01, 0.01, 0.01]), context={})
        assert math.isnan(result["SharpeRatio"])

    def test_normal_returns_works(self):
        """Verify the NaN guards don't break normal computation."""
        np.random.seed(42)
        m = SharpeRatio()
        returns = np.random.randn(100) * 0.01
        result = m.compute(returns, context={})
        # Should be finite (positive or negative depending on luck)
        assert math.isfinite(result["SharpeRatio"])


class TestSortinoRatioUndefined:
    def test_empty_returns_returns_nan(self):
        m = SortinoRatio()
        result = m.compute(np.array([]), context={})
        assert math.isnan(result["SortinoRatio"])

    def test_single_sample_returns_nan(self):
        m = SortinoRatio()
        result = m.compute(np.array([0.01]), context={})
        assert math.isnan(result["SortinoRatio"])

    def test_no_downside_with_positive_mean_returns_capped_100(self):
        """Documented 'great strategy' sentinel preserved."""
        m = SortinoRatio()
        # All positive returns → no downside → mean > 0 → 100.0 cap
        result = m.compute(np.array([0.01, 0.02, 0.03, 0.005, 0.01]), context={})
        assert result["SortinoRatio"] == 100.0

    def test_no_downside_with_zero_mean_returns_nan(self):
        """0/0 form (no downside + non-positive mean) — genuinely undefined."""
        m = SortinoRatio()
        # All zeros → no downside, mean=0
        result = m.compute(np.array([0.0, 0.0, 0.0, 0.0]), context={})
        # Pre-X.3 returned 0.0 (silent fabrication); now NaN
        assert math.isnan(result["SortinoRatio"])


class TestTotalReturnUndefined:
    def test_empty_returns_returns_nan(self):
        m = TotalReturn()
        result = m.compute(np.array([]), context={})
        assert math.isnan(result["TotalReturn"])

    def test_non_finite_returns_returns_nan(self):
        m = TotalReturn()
        result = m.compute(np.array([0.01, np.nan, 0.02]), context={})
        # validate_returns rejects non-finite arrays → NaN
        assert math.isnan(result["TotalReturn"])

    def test_all_zeros_returns_zero(self):
        """Legitimate no-return — defined value."""
        m = TotalReturn()
        result = m.compute(np.array([0.0, 0.0, 0.0]), context={})
        assert result["TotalReturn"] == 0.0

    def test_normal_returns_works(self):
        m = TotalReturn()
        result = m.compute(np.array([0.10, -0.05, 0.03]), context={})
        # (1.10 * 0.95 * 1.03) - 1 = 0.07635
        assert abs(result["TotalReturn"] - 0.07635) < 1e-4


class TestNaNPropagationToDisplay:
    """Verify NaN survives through the format pipeline (loud, not silent zero)."""

    def test_nan_formats_loudly(self):
        """Default Python format of NaN with :.2f produces 'nan' (visible)."""
        x = float("nan")
        formatted = f"{x:>6.2f}"
        # Must contain "nan" — operator reading table sees it's undefined
        assert "nan" in formatted.lower()

    def test_nan_distinguishable_from_zero(self):
        """NaN != 0.0 — formatted differently → operators not misled."""
        nan_str = f"{float('nan'):>6.2f}"
        zero_str = f"{0.0:>6.2f}"
        assert nan_str != zero_str
