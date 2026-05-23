"""Tests for BacktestStats fluent API.

FIND-040 lock tests: ``.daily()`` / ``.monthly()`` must raise NotImplementedError
until BacktestResult exposes ``timestamps_ns``; ``.full()`` must remain a no-op
self-return for fluent-API symmetry.

See ``lob-backtester/DESIGN_CLUSTER_D1_E_2026_05_14.md`` §4.1 +
``VALIDATION_FINDINGS_2026_05_14.md`` FIND-040 for context.
"""

import numpy as np
import pytest

from lobbacktest.stats import BacktestStats
from lobbacktest.types import BacktestResult


def _make_minimal_result() -> BacktestResult:
    """Construct a minimal BacktestResult for stats tests.

    All 15 required ``BacktestResult`` fields are populated; ``positions`` is
    ``np.ndarray`` (NOT ``List[Position]``) per ``types.py:204``. The fixture
    satisfies the FIND-002 round-trip pairing invariant trivially via empty
    ``trades`` + empty ``trade_pnls``.
    """
    n = 2
    return BacktestResult(
        equity_curve=np.array([100.0, 105.0]),
        returns=np.array([0.05]),
        positions=np.zeros(n),
        prices=np.array([10.0, 10.5]),
        predictions=np.zeros(n),
        labels=None,
        trades=[],
        trade_pnls=np.array([]),
        metrics={},
        config_dict={},
        initial_capital=100.0,
        final_equity=105.0,
        total_trades=0,
        start_index=0,
        end_index=n - 1,
    )


class TestPeriodAggregationStubs:
    """FIND-040: ``.daily()`` / ``.monthly()`` raise; ``.full()`` remains no-op."""

    def test_daily_raises_not_implemented(self):
        """FIND-040 lock: ``.daily()`` raises NotImplementedError citing the finding."""
        stats = BacktestStats(_make_minimal_result())
        with pytest.raises(NotImplementedError, match="FIND-040"):
            stats.daily()

    def test_monthly_raises_not_implemented(self):
        """FIND-040 lock: ``.monthly()`` raises NotImplementedError citing the finding."""
        stats = BacktestStats(_make_minimal_result())
        with pytest.raises(NotImplementedError, match="FIND-040"):
            stats.monthly()

    def test_full_remains_no_op_self_return(self):
        """``.full()`` is fluent self-return (no-op state setter); not affected by FIND-040."""
        stats = BacktestStats(_make_minimal_result())
        result = stats.full()
        assert result is stats  # fluent self-return
        assert stats._period == "full"  # state set (cosmetic only)


def _make_finite_result(n: int = 100) -> BacktestResult:
    """Construct a finite-equity BacktestResult for HF-2 annualization tests.

    Uses a small positive drift so annualized metrics (SharpeRatio,
    SortinoRatio, CalmarRatio, AnnualReturn) produce finite non-NaN
    values whose magnitude is sensitive to ``periods_per_day``.

    Differs from ``_make_minimal_result()``: that fixture has n=2 + zero
    trades, so annualization is degenerate. This fixture has n=100
    equity points producing finite returns for the periods_per_day-
    sensitivity check.
    """
    import numpy as np
    rng = np.random.default_rng(seed=42)
    returns = rng.normal(loc=0.0005, scale=0.01, size=n - 1)
    equity = np.concatenate([[100000.0], 100000.0 * np.cumprod(1 + returns)])
    return BacktestResult(
        equity_curve=equity,
        returns=returns,
        positions=np.zeros(n),
        prices=np.linspace(100.0, 100.0, n),
        predictions=np.zeros(n),
        labels=None,
        trades=[],
        trade_pnls=np.array([]),
        metrics={},
        config_dict={},
        initial_capital=100000.0,
        final_equity=float(equity[-1]),
        total_trades=0,
        start_index=0,
        end_index=n - 1,
    )


class TestPeriodsPerDayHF2:
    """HF-2 closure (2026-05-22): BacktestStats periods_per_day sister of #PY-263.

    Pre-fix BacktestStats.compute() constructed SharpeRatio/SortinoRatio/
    CalmarRatio/AnnualReturn with their class-default periods_per_day=1000.0
    AND built a plain context dict with NO periods_per_day key, so metric
    .compute() calls fell back to self.periods_per_day=1000.0 — silently
    inflating annualized metrics ~1.6018x at 60s time-based bins. The
    engine path at vectorized.py:623-664 was correctly mode-aware via
    BacktestConfig.resolved_periods_per_day (#PY-263 closure 2026-05-21);
    this surface (the operator-facing fluent BacktestStats API) was the
    sister gap left open.

    Fix wires periods_per_day through context dict + emits
    DeprecationWarning when not explicitly specified.
    """

    def test_default_construction_emits_deprecation_warning(self):
        """``BacktestStats(result).compute()`` without periods_per_day warns."""
        result = _make_finite_result()
        with pytest.warns(DeprecationWarning, match="periods_per_day not specified"):
            BacktestStats(result).compute()

    def test_explicit_periods_per_day_silences_warning(self):
        """Passing ``periods_per_day=X`` at construction silences the warning."""
        import warnings as _w
        result = _make_finite_result()
        with _w.catch_warnings():
            _w.simplefilter("error", DeprecationWarning)
            # Must not raise — explicit value silences the HF-2 warning
            BacktestStats(result, periods_per_day=1000.0).compute()
            BacktestStats(result, periods_per_day=390.0).compute()

    def test_with_periods_per_day_chainable_silences_warning(self):
        """``.with_periods_per_day(X)`` chained before ``.compute()`` silences."""
        import warnings as _w
        result = _make_finite_result()
        with _w.catch_warnings():
            _w.simplefilter("error", DeprecationWarning)
            BacktestStats(result).with_periods_per_day(390.0).compute()

    def test_with_periods_per_day_zero_raises(self):
        """``with_periods_per_day(0.0)`` raises ValueError per §5 fail-fast."""
        result = _make_finite_result()
        stats = BacktestStats(result)
        with pytest.raises(ValueError, match="periods_per_day must be > 0"):
            stats.with_periods_per_day(0.0)
        with pytest.raises(ValueError, match="periods_per_day must be > 0"):
            stats.with_periods_per_day(-100.0)

    def test_with_periods_per_day_returns_self_for_chaining(self):
        """``.with_periods_per_day(X)`` returns self per fluent-API convention."""
        result = _make_finite_result()
        stats = BacktestStats(result)
        returned = stats.with_periods_per_day(390.0)
        assert returned is stats, "with_periods_per_day must return self for fluent chaining"

    def test_explicit_periods_per_day_affects_annualized_metrics(self):
        """Sharpe at 390/day != Sharpe at 1000/day (factor sqrt(1000/390) ~= 1.6018x).

        This is the core HF-2 anti-regression: confirms the explicit
        periods_per_day value propagates through the context dict to
        SharpeRatio.compute() (risk.py:118) rather than falling back to
        the metric's 1000.0 class default.
        """
        import warnings as _w
        result = _make_finite_result()
        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            # Explicit 1000.0 (event-based / legacy default semantics)
            stats_1000 = BacktestStats(result, periods_per_day=1000.0).compute()
            sharpe_1000 = stats_1000.metrics["SharpeRatio"]
            # Explicit 390.0 (60s time-based bins)
            stats_390 = BacktestStats(result, periods_per_day=390.0).compute()
            sharpe_390 = stats_390.metrics["SharpeRatio"]
        # Sharpe ~= mean/std * sqrt(252 * periods_per_day). Ratio:
        # sharpe(1000) / sharpe(390) = sqrt(1000/390) ~= 1.6018
        # Allow 1% tolerance for arithmetic precision noise.
        import math
        expected_ratio = math.sqrt(1000.0 / 390.0)
        actual_ratio = sharpe_1000 / sharpe_390
        assert math.isclose(actual_ratio, expected_ratio, rel_tol=0.01), (
            f"HF-2 regression: Sharpe ratio sqrt(1000/390)~={expected_ratio:.4f} "
            f"but got sharpe_1000={sharpe_1000:.4f} / sharpe_390={sharpe_390:.4f} "
            f"= {actual_ratio:.4f}. Indicates periods_per_day NOT propagating "
            f"through context dict to SharpeRatio.compute()."
        )

    def test_init_periods_per_day_is_keyword_only(self):
        """``periods_per_day`` is keyword-only (mirrors SharpeRatio C1 trap fix)."""
        result = _make_finite_result()
        # Positional-arg attempt must raise TypeError
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            BacktestStats(result, 390.0)  # type: ignore[misc]

    def test_init_periods_per_day_zero_or_negative_raises(self):
        """``BacktestStats(..., periods_per_day=0.0|-X)`` raises ValueError at construction.

        Q3 ASYMMETRY closure (2026-05-22 mid-impl gate): pre-fix
        ``__init__`` silently accepted 0.0 / negative values while
        ``with_periods_per_day`` raised — inconsistent with §5 fail-fast.
        Now construction validates symmetrically.
        """
        result = _make_finite_result()
        with pytest.raises(ValueError, match="periods_per_day must be > 0"):
            BacktestStats(result, periods_per_day=0.0)
        with pytest.raises(ValueError, match="periods_per_day must be > 0"):
            BacktestStats(result, periods_per_day=-100.0)
        # Negative-zero is still <= 0 — must also raise
        with pytest.raises(ValueError, match="periods_per_day must be > 0"):
            BacktestStats(result, periods_per_day=-0.0)
