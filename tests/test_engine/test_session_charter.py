"""
No-overnight charter enforcement in VectorizedEngine (2026-08-15).

WHY THIS FILE EXISTS. The programme's one hard portfolio constraint is that
every position opens AND closes inside the same RTH session. Until 2026-08-15 no
code enforced it: ``loader.py`` built ``day_boundaries`` and nothing read it,
``load()`` concatenated every day into one continuous price array, and the
engine's position loop had no day-edge reset — so a position opened near a day's
end carried into the next day and booked the overnight gap, the largest return
term in the series, as intraday P&L. The only bound on a hold was a config
default (``ZeroDteConfig.max_holding_minutes = 60.0``).

Every published backtest is negative, so this defect INJECTED an out-of-charter
return term rather than removing one — it caused no null. The danger runs the
other way: the first genuinely positive backtest would be untrustworthy in the
most seductive way. These tests are the standing proof that it cannot recur.

Structure mirrors the enforcement's four obligations:
  1. a position that would span a boundary is closed AT the boundary
  2. the forced-close counter is correct and the exit is a real, costed exit
  3. a run WITHOUT boundaries still works and is MARKED unenforced
  4. the post-run identity check fires on a spanning trade
plus the day_boundaries tiling contract that all of the above rest on.
"""

import numpy as np
import pytest

from lobbacktest.config import BacktestConfig, CostConfig
from lobbacktest.data.loader import LoadedData
from lobbacktest.engine.vectorized import Backtester, BacktestData, VectorizedEngine
from lobbacktest.strategies.direction import DirectionStrategy
from lobbacktest.types import Trade, TradeSide


def _free_config(**overrides) -> BacktestConfig:
    """Zero-cost config so P&L arithmetic in assertions stays exact."""
    params = dict(
        initial_capital=10_000.0,
        position_size=0.1,
        costs=CostConfig(spread_bps=0, slippage_bps=0, commission_per_trade=0),
    )
    params.update(overrides)
    return BacktestConfig(**params)


def _three_days(rows_per_day: int = 4):
    """Prices + boundaries for 3 equal days, with a visible overnight gap.

    Each day drifts +1/row; between days the price JUMPS +50. That gap is the
    term a spanning position would illegitimately collect, so it is made large
    enough that any leak is unmistakable in the P&L.
    """
    prices = []
    boundaries = []
    level = 100.0
    cursor = 0
    for _ in range(3):
        for r in range(rows_per_day):
            prices.append(level + r)
        boundaries.append((cursor, cursor + rows_per_day))
        cursor += rows_per_day
        level += 50.0
    return np.array(prices, dtype=np.float64), boundaries


def _round_trips(result):
    """Pair the emitted trades into (entry, exit) tuples."""
    pairs = []
    open_t = None
    for t in result.trades:
        if t.side == TradeSide.FLAT:
            pairs.append((open_t, t))
            open_t = None
        else:
            open_t = t
    return pairs


class TestDayBoundariesContract:
    """day_boundaries must exactly tile [0, len(prices)) or raise.

    Row coordinates are a deterministic identity, not a hint. A boundaries list
    that does not describe THIS array is the same positional-coordinate failure
    that let a multi-source join fabricate IC +1.0000 on 162/162 days, so it
    fails loud rather than being guessed at (hft-rules §8).
    """

    def test_none_is_accepted(self):
        BacktestData(prices=np.array([100.0, 101.0]), day_boundaries=None)

    def test_exact_tiling_is_accepted(self):
        BacktestData(
            prices=np.array([100.0, 101.0, 102.0, 103.0]),
            day_boundaries=[(0, 2), (2, 4)],
        )

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty list is ambiguous"):
            BacktestData(prices=np.array([100.0, 101.0]), day_boundaries=[])

    def test_gap_raises(self):
        with pytest.raises(ValueError, match="tile"):
            BacktestData(
                prices=np.array([100.0, 101.0, 102.0, 103.0]),
                day_boundaries=[(0, 2), (3, 4)],
            )

    def test_overlap_raises(self):
        with pytest.raises(ValueError, match="tile"):
            BacktestData(
                prices=np.array([100.0, 101.0, 102.0, 103.0]),
                day_boundaries=[(0, 3), (2, 4)],
            )

    def test_short_coverage_raises(self):
        """The exact shape of the bug this guards: boundaries from another array."""
        with pytest.raises(ValueError, match="cover 2 rows but prices has 4"):
            BacktestData(
                prices=np.array([100.0, 101.0, 102.0, 103.0]),
                day_boundaries=[(0, 2)],
            )

    def test_empty_day_raises(self):
        with pytest.raises(ValueError, match="empty or reversed"):
            BacktestData(
                prices=np.array([100.0, 101.0]),
                day_boundaries=[(0, 0), (0, 2)],
            )


class TestForcedCloseAtBoundary:
    """Obligation 1+2: close at the day's last row, and count it."""

    def test_position_spanning_boundary_is_closed_at_day_end(self):
        """BUY on day 1 row 0, then HOLD forever."""
        prices, boundaries = _three_days(rows_per_day=4)
        predictions = np.zeros(len(prices), dtype=np.int64)
        predictions[0] = 1  # BUY once; every other row is HOLD

        engine = VectorizedEngine(_free_config())
        result = engine.run(
            BacktestData(prices=prices, day_boundaries=boundaries),
            DirectionStrategy(predictions, shifted=False),
        )

        pairs = _round_trips(result)
        assert len(pairs) == 1, "one entry should produce exactly one round trip"
        entry, exit_ = pairs[0]
        assert entry.index == 0
        # Day 1 is rows 0..3, so the forced exit lands on row 3 — NOT row 11.
        assert exit_.index == 3, (
            f"expected the charter close at row 3 (last row of day 1), got row {exit_.index}"
        )
        assert exit_.side == TradeSide.FLAT
        # And it exits at that row's price (103.0), never at the next day's
        # gapped-open price (150.0).
        assert exit_.price == pytest.approx(103.0)

    def test_forced_close_counter_is_correct(self):
        """Re-enter on each day; 3 days => 2 charter closes + 1 end-of-data."""
        prices, boundaries = _three_days(rows_per_day=4)
        predictions = np.zeros(len(prices), dtype=np.int64)
        predictions[0] = 1  # open day 1
        predictions[4] = 1  # open day 2
        predictions[8] = 1  # open day 3

        engine = VectorizedEngine(_free_config())
        result = engine.run(
            BacktestData(prices=prices, day_boundaries=boundaries),
            DirectionStrategy(predictions, shifted=False),
        )

        # The final day's tail is closed by the pre-existing end-of-data path,
        # so SessionForcedCloses counts exactly the closes that would NOT have
        # happened before the fix.
        assert result.metrics["SessionForcedCloses"] == 2.0
        assert result.metrics["SessionCharterEnforced"] == 1.0

        pairs = _round_trips(result)
        assert [e.index for e, _ in pairs] == [0, 4, 8]
        assert [x.index for _, x in pairs] == [3, 7, 11]

    def test_counter_is_zero_when_nothing_spans(self):
        """A strategy that already exits inside the day triggers no force-close."""
        prices, boundaries = _three_days(rows_per_day=4)
        predictions = np.zeros(len(prices), dtype=np.int64)
        predictions[0] = 1  # open on day 1 row 0
        predictions[1] = -1  # and close it on day 1 row 1

        engine = VectorizedEngine(_free_config(allow_short=False))
        result = engine.run(
            BacktestData(prices=prices, day_boundaries=boundaries),
            DirectionStrategy(predictions, shifted=False),
        )

        assert result.metrics["SessionForcedCloses"] == 0.0
        assert result.metrics["SessionCharterEnforced"] == 1.0

    def test_signal_exit_on_the_last_row_is_not_double_counted(self):
        """A signal-driven close ON the boundary row must not also force-close.

        The force-close runs AFTER signal processing precisely so this cannot
        emit two exits for one position (which would break the round-trip
        pairing invariant in BacktestResult.__post_init__).
        """
        prices, boundaries = _three_days(rows_per_day=4)
        predictions = np.zeros(len(prices), dtype=np.int64)
        predictions[0] = 1
        predictions[3] = -1  # SELL on the last row of day 1

        engine = VectorizedEngine(_free_config(allow_short=False))
        result = engine.run(
            BacktestData(prices=prices, day_boundaries=boundaries),
            DirectionStrategy(predictions, shifted=False),
        )

        assert result.metrics["SessionForcedCloses"] == 0.0
        assert len(_round_trips(result)) == 1

    def test_forced_close_charges_the_same_cost_as_a_signal_exit(self):
        """Obligation: the SAME exit path and cost model, not a free unwind.

        Run A closes on row 3 by signal; run B is force-closed on row 3 by the
        charter. Identical costs are charged, so the two P&Ls must match exactly.
        """
        prices, boundaries = _three_days(rows_per_day=4)
        costed = _free_config(
            costs=CostConfig(spread_bps=5.0, slippage_bps=2.0, commission_per_trade=1.0),
            allow_short=False,
        )

        preds_signal = np.zeros(len(prices), dtype=np.int64)
        preds_signal[0], preds_signal[3] = 1, -1
        signal_run = VectorizedEngine(costed).run(
            BacktestData(prices=prices, day_boundaries=boundaries),
            DirectionStrategy(preds_signal, shifted=False),
        )

        preds_forced = np.zeros(len(prices), dtype=np.int64)
        preds_forced[0] = 1
        forced_run = VectorizedEngine(costed).run(
            BacktestData(prices=prices, day_boundaries=boundaries),
            DirectionStrategy(preds_forced, shifted=False),
        )

        assert signal_run.metrics["SessionForcedCloses"] == 0.0
        assert forced_run.metrics["SessionForcedCloses"] == 1.0
        assert forced_run.trade_pnls[0] == pytest.approx(signal_run.trade_pnls[0])
        assert forced_run.trades[1].cost == pytest.approx(signal_run.trades[1].cost)
        assert forced_run.trades[1].cost > 0.0, "a forced close must still pay costs"


class TestUnenforcedRunIsDistinguishable:
    """Obligation 3: back-compat, but never SILENT back-compat."""

    def test_run_without_boundaries_still_works_and_is_marked_unenforced(self):
        prices, _ = _three_days(rows_per_day=4)
        predictions = np.zeros(len(prices), dtype=np.int64)
        predictions[0] = 1

        engine = VectorizedEngine(_free_config())
        result = engine.run(
            BacktestData(prices=prices),  # no day_boundaries
            DirectionStrategy(predictions, shifted=False),
        )

        assert result.metrics["SessionCharterEnforced"] == 0.0
        assert result.metrics["SessionForcedCloses"] == 0.0
        # And the position DOES span — this pins the pre-2026-08-15 behaviour
        # that back-compat deliberately preserves, so the flag is the only thing
        # standing between a reader and an out-of-charter P&L.
        entry, exit_ = _round_trips(result)[0]
        assert entry.index == 0 and exit_.index == len(prices) - 1

    def test_enforced_and_unenforced_differ_by_the_overnight_gap(self):
        """The two runs are NOT equivalent — the delta is the gap itself."""
        prices, boundaries = _three_days(rows_per_day=4)
        predictions = np.zeros(len(prices), dtype=np.int64)
        predictions[0] = 1

        engine = VectorizedEngine(_free_config())
        strategy = DirectionStrategy(predictions, shifted=False)
        unenforced = engine.run(BacktestData(prices=prices), strategy)
        enforced = engine.run(BacktestData(prices=prices, day_boundaries=boundaries), strategy)

        # Unenforced exits at 203.0 having entered at 100.0 (collecting both
        # +50 gaps); enforced exits at 103.0 the same day.
        assert unenforced.trade_pnls[0] > enforced.trade_pnls[0]
        size = enforced.trades[0].size
        gap_collected = unenforced.trade_pnls[0] - enforced.trade_pnls[0]
        assert gap_collected == pytest.approx(100.0 * size), (
            "the unenforced run's excess P&L should be exactly the two "
            "+50 overnight gaps it was never allowed to hold"
        )

    def test_run_from_arrays_defaults_to_unenforced(self):
        """The bare-array public entry point used by discovery harnesses."""
        prices, _ = _three_days(rows_per_day=4)
        predictions = np.zeros(len(prices), dtype=np.int64)
        predictions[0] = 1

        result = Backtester(_free_config()).run_from_arrays(prices, predictions)
        assert result.metrics["SessionCharterEnforced"] == 0.0

    def test_run_from_arrays_accepts_boundaries(self):
        prices, boundaries = _three_days(rows_per_day=4)
        predictions = np.zeros(len(prices), dtype=np.int64)
        predictions[0] = 1

        result = Backtester(_free_config()).run_from_arrays(
            prices, predictions, day_boundaries=boundaries
        )
        assert result.metrics["SessionCharterEnforced"] == 1.0
        assert result.metrics["SessionForcedCloses"] == 1.0


class TestSpanAssertion:
    """Obligation 4: the post-run identity check must actually fire.

    The force-close alone would be an ASSUMPTION. This check re-derives each
    round trip's entry and exit day from the emitted trade record and disagrees
    loudly — the discipline whose absence let day_boundaries sit unread.
    """

    def test_assertion_fires_on_synthetic_spanning_trade(self):
        engine = VectorizedEngine(_free_config())
        spanning = [
            Trade(index=1, side=TradeSide.BUY, price=100.0, size=1.0, cost=0.0),
            Trade(index=5, side=TradeSide.FLAT, price=150.0, size=1.0, cost=0.0),
        ]
        with pytest.raises(ValueError, match="NO-OVERNIGHT CHARTER VIOLATION"):
            engine._assert_no_trade_spans_session(spanning, [(0, 4), (4, 8)])

    def test_assertion_passes_on_same_day_trade(self):
        engine = VectorizedEngine(_free_config())
        same_day = [
            Trade(index=1, side=TradeSide.BUY, price=100.0, size=1.0, cost=0.0),
            Trade(index=3, side=TradeSide.FLAT, price=103.0, size=1.0, cost=0.0),
        ]
        engine._assert_no_trade_spans_session(same_day, [(0, 4), (4, 8)])

    def test_assertion_is_a_noop_without_boundaries(self):
        engine = VectorizedEngine(_free_config())
        spanning = [
            Trade(index=1, side=TradeSide.BUY, price=100.0, size=1.0, cost=0.0),
            Trade(index=5, side=TradeSide.FLAT, price=150.0, size=1.0, cost=0.0),
        ]
        engine._assert_no_trade_spans_session(spanning, None)

    def test_assertion_rejects_a_non_alternating_record(self):
        """A corrupt trade record makes the span check meaningless; say so."""
        engine = VectorizedEngine(_free_config())
        two_opens = [
            Trade(index=1, side=TradeSide.BUY, price=100.0, size=1.0, cost=0.0),
            Trade(index=2, side=TradeSide.BUY, price=101.0, size=1.0, cost=0.0),
        ]
        with pytest.raises(ValueError, match="Charter check cannot run"):
            engine._assert_no_trade_spans_session(two_opens, [(0, 4)])


class TestLoaderThreadsBoundaries:
    """LoadedData must hand the boundaries it already builds to the engine.

    Before 2026-08-15 all five occurrences of ``day_boundaries`` in loader.py
    were construction — the list was built and never read.
    """

    def test_to_backtest_data_threads_day_boundaries(self):
        prices, boundaries = _three_days(rows_per_day=4)
        loaded = LoadedData(
            sequences=np.zeros((len(prices), 2, 3), dtype=np.float32),
            labels=np.zeros(len(prices), dtype=np.int8),
            prices=prices,
            day_boundaries=boundaries,
            days=["20250203", "20250204", "20250205"],
        )
        assert loaded.to_backtest_data().day_boundaries == boundaries

    def test_threaded_boundaries_actually_enforce(self):
        """End to end: loader -> BacktestData -> engine force-close."""
        prices, boundaries = _three_days(rows_per_day=4)
        loaded = LoadedData(
            sequences=np.zeros((len(prices), 2, 3), dtype=np.float32),
            labels=np.zeros((len(prices), 2), dtype=np.int8),
            prices=prices,
            day_boundaries=boundaries,
            days=["20250203", "20250204", "20250205"],
        )
        predictions = np.zeros(len(prices), dtype=np.int64)
        predictions[0] = 1

        result = VectorizedEngine(_free_config()).run(
            loaded.to_backtest_data(horizon_idx=0),
            DirectionStrategy(predictions, shifted=False),
        )
        assert result.metrics["SessionCharterEnforced"] == 1.0
        assert result.metrics["SessionForcedCloses"] == 1.0
