"""
Wave 1C T1.1 + class-coherent fold-in tests (2026-05-17).

Locks Phase X.3 NaN-sentinel convention on BacktestResult derived properties:
- `total_return` property at types.py:259-263 — NaN on initial_capital == 0
- `max_drawdown` property at types.py:271-279 — NaN on empty equity_curve

Pre-fix both properties returned silent 0.0 on degenerate inputs while
their sister metrics (`TotalReturn`, `MaxDrawdown`) returned NaN. Same
backtest reported TWO different values via the two access paths
(`BacktestResult.summary()` reads property; `BacktestStats(...).summary()`
reads metric).

Note: BacktestResult.__post_init__ at types.py:218-222 blocks construction
with empty equity_curve. The empty-equity branch in `max_drawdown` is
therefore DEFENSIVE — testing requires bypassing __post_init__ by
mutating the field after construction (BacktestResult is non-frozen, so
direct field assignment works).

`total_return` NaN-on-zero-capital IS reachable through normal
construction (initial_capital=0 is not blocked by __post_init__).
"""

import math

import numpy as np
import pytest

from lobbacktest.types import BacktestResult


def _valid_minimal_result(initial_capital: float = 100000.0) -> BacktestResult:
    """Minimal valid BacktestResult (passes __post_init__)."""
    return BacktestResult(
        equity_curve=np.array([100000.0, 100000.0]),
        returns=np.array([0.0]),
        positions=np.array([0.0, 0.0]),
        trades=[],
        trade_pnls=np.array([], dtype=np.float64),
        prices=np.array([100.0, 100.0]),
        predictions=np.array([1, 1], dtype=np.int8),
        labels=None,
        metrics={},
        config_dict={},
        initial_capital=initial_capital,
        final_equity=100000.0,
        total_trades=0,
        start_index=0,
        end_index=1,
    )


class TestTotalReturnNaNOnZeroCapital:
    """Class-coherent fold-in: total_return returns NaN when initial_capital==0."""

    def test_zero_initial_capital_returns_nan(self):
        """initial_capital == 0 → NaN (not silent 0.0).

        Pre-fix: returned 0.0 (silent "no return")
        Post-fix: returns NaN (undefined)
        """
        result = BacktestResult(
            equity_curve=np.array([0.0, 0.0]),
            returns=np.array([0.0]),
            positions=np.array([0.0, 0.0]),
            trades=[],
            trade_pnls=np.array([], dtype=np.float64),
            prices=np.array([100.0, 100.0]),
            predictions=np.array([1, 1], dtype=np.int8),
            labels=None,
            metrics={},
            config_dict={},
            initial_capital=0.0,
            final_equity=0.0,
            total_trades=0,
            start_index=0,
            end_index=1,
        )
        assert math.isnan(result.total_return), (
            f"total_return on initial_capital=0 must be NaN, got {result.total_return}"
        )

    def test_positive_initial_capital_break_even_returns_zero(self):
        """Normal case: positive initial, final at break-even → 0.0 (finite)."""
        result = _valid_minimal_result(initial_capital=100000.0)
        # equity_curve=[100k, 100k], final=100k → return 0.0
        assert result.total_return == 0.0
        assert not math.isnan(result.total_return)

    def test_normal_positive_return(self):
        """Normal case: 10% return computes correctly."""
        result = BacktestResult(
            equity_curve=np.array([100000.0, 110000.0]),
            returns=np.array([0.10]),
            positions=np.array([0.0, 0.0]),
            trades=[],
            trade_pnls=np.array([], dtype=np.float64),
            prices=np.array([100.0, 110.0]),
            predictions=np.array([1, 1], dtype=np.int8),
            labels=None,
            metrics={},
            config_dict={},
            initial_capital=100000.0,
            final_equity=110000.0,
            total_trades=0,
            start_index=0,
            end_index=1,
        )
        assert result.total_return == pytest.approx(0.10)


class TestMaxDrawdownNaNOnEmptyEquity:
    """Wave 1C T1.1: max_drawdown returns NaN on empty equity (defensive).

    BacktestResult.__post_init__ blocks empty-equity construction, so this
    branch is defensive. We test by mutating equity_curve to empty after
    construction (legal: dataclass is non-frozen).
    """

    def test_empty_equity_returns_nan(self):
        """Empty equity_curve (post-construction mutation) → NaN.

        Pre-fix returned silent 0.0; post-fix returns NaN to align with
        `MaxDrawdown` metric at risk.py:294 (which returns NaN on empty).
        """
        result = _valid_minimal_result()
        # Bypass __post_init__ to test the property's defensive branch
        result.equity_curve = np.array([], dtype=np.float64)
        val = result.max_drawdown
        assert math.isnan(val), (
            f"max_drawdown on empty equity_curve must be NaN, got {val}"
        )

    def test_monotonic_equity_returns_zero(self):
        """Monotonic increasing equity → 0.0 (no drawdown; finite)."""
        result = BacktestResult(
            equity_curve=np.array([100000.0, 105000.0, 110000.0]),
            returns=np.array([0.05, 0.0476]),
            positions=np.array([0.0, 0.0, 0.0]),
            trades=[],
            trade_pnls=np.array([], dtype=np.float64),
            prices=np.array([100.0, 105.0, 110.0]),
            predictions=np.array([1, 1, 1], dtype=np.int8),
            labels=None,
            metrics={},
            config_dict={},
            initial_capital=100000.0,
            final_equity=110000.0,
            total_trades=0,
            start_index=0,
            end_index=2,
        )
        assert result.max_drawdown == pytest.approx(0.0)
        assert not math.isnan(result.max_drawdown)

    def test_drawdown_computed_correctly(self):
        """100→110→99 sequence → MDD = (110-99)/110 = 0.10."""
        result = BacktestResult(
            equity_curve=np.array([100000.0, 110000.0, 99000.0]),
            returns=np.array([0.10, -0.10]),
            positions=np.array([0.0, 0.0, 0.0]),
            trades=[],
            trade_pnls=np.array([], dtype=np.float64),
            prices=np.array([100.0, 110.0, 99.0]),
            predictions=np.array([1, 1, 1], dtype=np.int8),
            labels=None,
            metrics={},
            config_dict={},
            initial_capital=100000.0,
            final_equity=99000.0,
            total_trades=0,
            start_index=0,
            end_index=2,
        )
        assert result.max_drawdown == pytest.approx(0.10)
