"""Tests for BacktestContext — typed context with dict backward compatibility.

Validates that BacktestContext supports both typed attribute access
(new code) and dict-style access (all existing metrics unchanged).

Reference: BACKTESTER_AUDIT_PLAN.md § Phase 2b
"""

import math

import numpy as np
import pytest

from lobbacktest.context import BacktestContext


class TestBacktestContextTypedAccess:
    """Typed attribute access (new code pattern)."""

    def test_core_data_attributes(self):
        """Core data arrays accessible as typed attributes."""
        equity = np.array([100.0, 101.0, 102.0])
        pnls = np.array([1.0, -0.5])
        ctx = BacktestContext(equity_curve=equity, trade_pnls=pnls)
        np.testing.assert_array_equal(ctx.equity_curve, equity)
        np.testing.assert_array_equal(ctx.trade_pnls, pnls)

    def test_optional_fields_default_none(self):
        """Predictions and labels default to None."""
        ctx = BacktestContext(
            equity_curve=np.array([100.0]),
            trade_pnls=np.array([0.0]),
        )
        assert ctx.predictions is None
        assert ctx.labels is None

    def test_config_defaults(self):
        """Configuration parameters have sensible defaults."""
        ctx = BacktestContext(
            equity_curve=np.array([100.0]),
            trade_pnls=np.array([0.0]),
        )
        assert ctx.initial_capital == 1.0
        assert ctx.trading_days_per_year == 252.0
        assert ctx.periods_per_day == 1000.0

    def test_annualization_factor_computed(self):
        """Annualization factor auto-computed: sqrt(252 * 1000)."""
        ctx = BacktestContext(
            equity_curve=np.array([100.0]),
            trade_pnls=np.array([0.0]),
        )
        expected = math.sqrt(252.0 * 1000.0)
        assert abs(ctx.annualization_factor - expected) < 1e-10, (
            f"Expected annualization_factor={expected:.4f}, got {ctx.annualization_factor:.4f}"
        )

    def test_annualization_factor_explicit(self):
        """Explicit annualization_factor overrides computation."""
        ctx = BacktestContext(
            equity_curve=np.array([100.0]),
            trade_pnls=np.array([0.0]),
            annualization_factor=15.87,
        )
        assert ctx.annualization_factor == 15.87


class TestBacktestContextDictAccess:
    """Dict-style access (backward compatible with all metrics)."""

    def test_getitem_typed_field(self):
        """context['equity_curve'] returns the typed field."""
        equity = np.array([100.0, 101.0])
        ctx = BacktestContext(equity_curve=equity, trade_pnls=np.array([0.0]))
        np.testing.assert_array_equal(ctx["equity_curve"], equity)

    def test_getitem_config_field(self):
        """context['initial_capital'] returns config value."""
        ctx = BacktestContext(
            equity_curve=np.array([100.0]),
            trade_pnls=np.array([0.0]),
            initial_capital=100_000.0,
        )
        assert ctx["initial_capital"] == 100_000.0

    def test_getitem_missing_raises_keyerror(self):
        """context['nonexistent'] raises KeyError (same as dict)."""
        ctx = BacktestContext(
            equity_curve=np.array([100.0]),
            trade_pnls=np.array([0.0]),
        )
        with pytest.raises(KeyError):
            _ = ctx["nonexistent_key"]

    def test_contains_typed_field(self):
        """'equity_curve' in context → True."""
        ctx = BacktestContext(
            equity_curve=np.array([100.0]),
            trade_pnls=np.array([0.0]),
        )
        assert "equity_curve" in ctx
        assert "trade_pnls" in ctx
        assert "initial_capital" in ctx
        assert "nonexistent" not in ctx

    def test_get_with_default(self):
        """context.get('missing', 0.0) returns default."""
        ctx = BacktestContext(
            equity_curve=np.array([100.0]),
            trade_pnls=np.array([0.0]),
        )
        assert ctx.get("nonexistent", 42.0) == 42.0
        assert ctx.get("initial_capital") == 1.0

    def test_private_fields_not_accessible_via_dict(self):
        """Private fields (starting with _) are not exposed via dict access."""
        ctx = BacktestContext(
            equity_curve=np.array([100.0]),
            trade_pnls=np.array([0.0]),
        )
        assert "_computed" not in ctx
        with pytest.raises(KeyError):
            _ = ctx["_computed"]


class TestBacktestContextComputedMetrics:
    """Computed metrics added during iteration (mutable _computed dict)."""

    def test_update_and_retrieve(self):
        """context.update({'SharpeRatio': 1.5}) then context['SharpeRatio'] → 1.5."""
        ctx = BacktestContext(
            equity_curve=np.array([100.0]),
            trade_pnls=np.array([0.0]),
        )
        ctx.update({"SharpeRatio": 1.5, "MaxDrawdown": 0.05})
        assert ctx["SharpeRatio"] == 1.5
        assert ctx["MaxDrawdown"] == 0.05

    def test_computed_overrides_typed_field(self):
        """Computed metric with same name as typed field takes precedence.

        This shouldn't happen in practice, but the protocol should be clear:
        computed metrics (from metric iteration) override typed fields.
        """
        ctx = BacktestContext(
            equity_curve=np.array([100.0]),
            trade_pnls=np.array([0.0]),
            initial_capital=100_000.0,
        )
        # Simulate a metric that stores "initial_capital" (unlikely but tests protocol)
        ctx.update({"initial_capital": 999.0})
        assert ctx["initial_capital"] == 999.0  # Computed takes precedence

    def test_contains_computed_metric(self):
        """'SharpeRatio' in context → True after update."""
        ctx = BacktestContext(
            equity_curve=np.array([100.0]),
            trade_pnls=np.array([0.0]),
        )
        assert "SharpeRatio" not in ctx
        ctx.update({"SharpeRatio": 1.5})
        assert "SharpeRatio" in ctx

    def test_metric_dependency_chain(self):
        """Simulate the engine's metric iteration: TotalReturn → AnnualReturn → CalmarRatio.

        Each metric's result is available to subsequent metrics via context.
        """
        ctx = BacktestContext(
            equity_curve=np.array([100.0, 110.0]),
            trade_pnls=np.array([10.0]),
        )

        # Step 1: TotalReturn computes
        ctx.update({"TotalReturn": 0.10})
        assert ctx["TotalReturn"] == 0.10

        # Step 2: AnnualReturn reads TotalReturn from context
        total_return = ctx.get("TotalReturn", 0.0)
        assert total_return == 0.10
        ctx.update({"AnnualReturn": 0.25})

        # Step 3: CalmarRatio reads both
        annual = ctx.get("AnnualReturn", 0.0)
        assert annual == 0.25
