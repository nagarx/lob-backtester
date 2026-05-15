"""
Tests for trading metrics.

Tests verify:
- WinRate formula
- ProfitFactor formula
- AverageWin/AverageLoss
- PayoffRatio
- Edge cases (no trades, all wins, all losses)
"""

import numpy as np
import pytest

from lobbacktest.metrics.trading import (
    AverageLoss,
    AverageWin,
    Expectancy,
    PayoffRatio,
    ProfitFactor,
    WinRate,
)


class TestWinRate:
    """Tests for WinRate metric."""

    def test_win_rate_formula(self):
        """
        Verify: WinRate = num_winning / total

        Test: 3 wins, 2 losses -> 60% win rate
        """
        trade_pnls = np.array([10, -5, 20, -10, 15])  # 3 wins, 2 losses
        context = {"trade_pnls": trade_pnls}

        metric = WinRate()
        result = metric.compute(np.array([]), context)

        expected = 3 / 5
        assert abs(result["WinRate"] - expected) < 1e-10, (
            f"Expected {expected}, got {result['WinRate']}"
        )

    def test_win_rate_all_wins(self):
        """Test 100% win rate."""
        trade_pnls = np.array([10, 20, 30, 40])
        context = {"trade_pnls": trade_pnls}

        metric = WinRate()
        result = metric.compute(np.array([]), context)
        assert result["WinRate"] == 1.0

    def test_win_rate_all_losses(self):
        """Test 0% win rate."""
        trade_pnls = np.array([-10, -20, -30, -40])
        context = {"trade_pnls": trade_pnls}

        metric = WinRate()
        result = metric.compute(np.array([]), context)
        assert result["WinRate"] == 0.0

    def test_win_rate_no_trades(self):
        """Test handling of no trades.

        Phase X.3 Empirical Trust (2026-05-05): empty trade list returns
        NaN (was 0.0 pre-X.3). Indistinguishable from legitimate "all-losers
        0% WR" pre-fix → silent ambiguity. Per hft-rules §8.
        """
        import math
        context = {"trade_pnls": np.array([])}

        metric = WinRate()
        result = metric.compute(np.array([]), context)
        assert math.isnan(result["WinRate"])

    def test_win_rate_zero_pnl_not_counted_as_win(self):
        """Test that zero P&L is not counted as a win."""
        trade_pnls = np.array([10, 0, -5])  # 1 win, 1 zero, 1 loss
        context = {"trade_pnls": trade_pnls}

        metric = WinRate()
        result = metric.compute(np.array([]), context)
        expected = 1 / 3  # Only positive counts as win
        assert abs(result["WinRate"] - expected) < 1e-10


class TestProfitFactor:
    """Tests for ProfitFactor metric."""

    def test_profit_factor_formula(self):
        """
        Verify: ProfitFactor = sum(wins) / abs(sum(losses))

        Test: wins = [10, 20, 30] = 60, losses = [-5, -15] = -20
        PF = 60 / 20 = 3.0
        """
        trade_pnls = np.array([10, -5, 20, -15, 30])
        context = {"trade_pnls": trade_pnls}

        metric = ProfitFactor()
        result = metric.compute(np.array([]), context)

        expected = 60 / 20
        assert abs(result["ProfitFactor"] - expected) < 1e-10, (
            f"Expected {expected}, got {result['ProfitFactor']}"
        )

    def test_profit_factor_no_losses(self):
        """Test capped value when no losses."""
        trade_pnls = np.array([10, 20, 30])
        context = {"trade_pnls": trade_pnls}

        metric = ProfitFactor()
        result = metric.compute(np.array([]), context)
        assert result["ProfitFactor"] == 100.0  # Capped

    def test_profit_factor_no_wins(self):
        """Test zero when no wins."""
        trade_pnls = np.array([-10, -20, -30])
        context = {"trade_pnls": trade_pnls}

        metric = ProfitFactor()
        result = metric.compute(np.array([]), context)
        assert result["ProfitFactor"] == 0.0

    def test_profit_factor_no_trades(self):
        """Test handling of no trades.

        Phase X.3 Empirical Trust (2026-05-05): empty trade list returns
        NaN (was 0.0 pre-X.3). Per hft-rules §8.
        """
        import math
        context = {"trade_pnls": np.array([])}

        metric = ProfitFactor()
        result = metric.compute(np.array([]), context)
        assert math.isnan(result["ProfitFactor"])

    def test_profit_factor_breakeven(self):
        """Test profit factor = 1 for breakeven."""
        trade_pnls = np.array([10, -10, 20, -20])
        context = {"trade_pnls": trade_pnls}

        metric = ProfitFactor()
        result = metric.compute(np.array([]), context)
        expected = 30 / 30
        assert abs(result["ProfitFactor"] - expected) < 1e-10


class TestAverageWin:
    """Tests for AverageWin metric."""

    def test_average_win_formula(self):
        """
        Verify: AverageWin = mean(pnl for pnl > 0)

        Test: wins = [10, 20, 30] -> mean = 20
        """
        trade_pnls = np.array([10, -5, 20, -15, 30])
        context = {"trade_pnls": trade_pnls}

        metric = AverageWin()
        result = metric.compute(np.array([]), context)

        expected = (10 + 20 + 30) / 3
        assert abs(result["AverageWin"] - expected) < 1e-10

    def test_average_win_no_wins(self):
        """Test handling of no winning trades.

        FIND-024 EXT (2026-05-15): no-winners returns NaN (was 0.0 pre-fix).
        Mirrors Phase X.3 WinRate precedent at trading.py:79. Per hft-rules
        §8 — 0.0 was indistinguishable from legitimate "avg-winner-is-zero".
        """
        import math
        trade_pnls = np.array([-10, -20, -30])
        context = {"trade_pnls": trade_pnls}

        metric = AverageWin()
        result = metric.compute(np.array([]), context)
        assert math.isnan(result["AverageWin"])

    def test_average_win_no_trades(self):
        """Test handling of empty trade list (FIND-024 EXT 2026-05-15).

        Empty trade_pnls returns NaN mirroring Phase X.3 WinRate precedent
        at trading.py:79. Per hft-rules §8 — silent 0.0 pre-fix was
        indistinguishable from "average winning trade is zero".
        """
        import math
        context = {"trade_pnls": np.array([])}

        metric = AverageWin()
        result = metric.compute(np.array([]), context)
        assert math.isnan(result["AverageWin"])


class TestAverageLoss:
    """Tests for AverageLoss metric."""

    def test_average_loss_formula(self):
        """
        Verify: AverageLoss = abs(mean(pnl for pnl < 0))

        Test: losses = [-10, -20] -> mean = -15 -> abs = 15
        """
        trade_pnls = np.array([30, -10, 20, -20])
        context = {"trade_pnls": trade_pnls}

        metric = AverageLoss()
        result = metric.compute(np.array([]), context)

        expected = 15.0  # abs((-10 + -20) / 2)
        assert abs(result["AverageLoss"] - expected) < 1e-10

    def test_average_loss_no_losses(self):
        """Test handling of no losing trades.

        FIND-024 EXT (2026-05-15): no-losers returns NaN (was 0.0 pre-fix).
        Mirrors Phase X.3 WinRate precedent at trading.py:79. Per hft-rules
        §8 — 0.0 was indistinguishable from legitimate "avg-loser-is-zero".
        """
        import math
        trade_pnls = np.array([10, 20, 30])
        context = {"trade_pnls": trade_pnls}

        metric = AverageLoss()
        result = metric.compute(np.array([]), context)
        assert math.isnan(result["AverageLoss"])

    def test_average_loss_no_trades(self):
        """Test handling of empty trade list (FIND-024 EXT 2026-05-15).

        Empty trade_pnls returns NaN mirroring Phase X.3 WinRate precedent
        at trading.py:79. Per hft-rules §8 — silent 0.0 pre-fix was
        indistinguishable from "average losing trade is zero".
        """
        import math
        context = {"trade_pnls": np.array([])}

        metric = AverageLoss()
        result = metric.compute(np.array([]), context)
        assert math.isnan(result["AverageLoss"])


class TestPayoffRatio:
    """Tests for PayoffRatio metric."""

    def test_payoff_ratio_formula(self):
        """
        Verify: PayoffRatio = AverageWin / AverageLoss

        Test: avg_win = 20, avg_loss = 10 -> ratio = 2.0
        """
        trade_pnls = np.array([20, -10, 20, -10])
        context = {"trade_pnls": trade_pnls}

        metric = PayoffRatio()
        result = metric.compute(np.array([]), context)

        expected = 20 / 10
        assert abs(result["PayoffRatio"] - expected) < 1e-10

    def test_payoff_ratio_no_losses(self):
        """Test capped value when no losses.

        FIND-024 EXT (2026-05-15): L358 magic 100.0 cap PRESERVED as
        documented sentinel for "infinite reward-to-risk" semantic.
        Migration to math.inf vs NaN deferred to #PY-FIND-024-EXT cycle.
        """
        context = {"AverageWin": 20.0, "AverageLoss": 0.0}

        metric = PayoffRatio()
        result = metric.compute(np.array([]), context)
        assert result["PayoffRatio"] == 100.0

    def test_payoff_ratio_no_trades(self):
        """Test handling of empty trade list (FIND-024 EXT 2026-05-15).

        Empty trade_pnls returns NaN at trading.py:347 (was 0.0 pre-fix).
        Mirrors Phase X.3 WinRate precedent. Per hft-rules §8 — silent 0.0
        pre-fix was indistinguishable from "no winning trades, only losers
        path produces 0/avg_loss = 0.0" semantic.
        """
        import math
        context = {"trade_pnls": np.array([])}

        metric = PayoffRatio()
        result = metric.compute(np.array([]), context)
        assert math.isnan(result["PayoffRatio"])


class TestExpectancy:
    """Tests for Expectancy metric."""

    def test_expectancy_formula(self):
        """
        Verify: Expectancy = WinRate * AvgWin - (1 - WinRate) * AvgLoss

        Test: WR = 0.6, AvgWin = 20, AvgLoss = 10
        E = 0.6 * 20 - 0.4 * 10 = 12 - 4 = 8
        """
        # 3 wins of 20, 2 losses of 10
        trade_pnls = np.array([20, -10, 20, -10, 20])
        context = {"trade_pnls": trade_pnls}

        metric = Expectancy()
        result = metric.compute(np.array([]), context)

        win_rate = 3 / 5
        avg_win = 20.0
        avg_loss = 10.0
        expected = win_rate * avg_win - (1 - win_rate) * avg_loss
        assert abs(result["Expectancy"] - expected) < 1e-10

    def test_expectancy_positive_system(self):
        """Test positive expectancy for profitable system."""
        trade_pnls = np.array([30, -10, 30, -10, 30])  # 60% WR, 30 avg win, 10 avg loss
        context = {"trade_pnls": trade_pnls}

        metric = Expectancy()
        result = metric.compute(np.array([]), context)
        assert result["Expectancy"] > 0

    def test_expectancy_negative_system(self):
        """Test negative expectancy for losing system."""
        trade_pnls = np.array([10, -30, 10, -30, -30])  # 40% WR, 10 avg win, 30 avg loss
        context = {"trade_pnls": trade_pnls}

        metric = Expectancy()
        result = metric.compute(np.array([]), context)
        assert result["Expectancy"] < 0

    def test_expectancy_no_trades(self):
        """Test handling of empty trade list (FIND-024 EXT 2026-05-15).

        Empty trade_pnls returns NaN at trading.py:417 (was 0.0 pre-fix).
        Mirrors Phase X.3 WinRate precedent. Per hft-rules §8 — silent 0.0
        pre-fix was indistinguishable from "perfectly balanced expectancy
        = 0 from neutral system" semantic.
        """
        import math
        context = {"trade_pnls": np.array([])}

        metric = Expectancy()
        result = metric.compute(np.array([]), context)
        assert math.isnan(result["Expectancy"])

