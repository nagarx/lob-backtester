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
