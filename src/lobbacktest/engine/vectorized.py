"""
Backtest engine with per-sample position tracking.

This engine executes backtests using a per-sample loop for explicit position
tracking, with numpy-based metric computation. It assumes instant fill at
the current price (no queue simulation).

Note: Module is named 'vectorized.py' for historical reasons.
The main engine loop is a Python for-loop, not vectorized.

Design Philosophy:
- Position tracking is explicit and auditable (per-sample loop)
- Transaction costs are modeled explicitly (entry + exit costs)
- Short and long positions have symmetric sizing and accounting
- Results include all information needed for analysis
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lobbacktest.config import BacktestConfig
from lobbacktest.metrics.base import Metric
from lobbacktest.metrics.prediction import DirectionalAccuracy, SignalRate
from lobbacktest.metrics.returns import AnnualReturn, TotalReturn
from lobbacktest.metrics.risk import CalmarRatio, MaxDrawdown, SharpeRatio, SortinoRatio
from lobbacktest.metrics.trading import (
    AverageLoss,
    AverageWin,
    Expectancy,
    PayoffRatio,
    ProfitFactor,
    WinRate,
)
from lobbacktest.strategies.base import Signal, SignalOutput, Strategy
from lobbacktest.types import BacktestResult, Position, PositionSide, Trade, TradeSide

logger = logging.getLogger(__name__)


@dataclass
class BacktestData:
    """
    Data container for backtest input.

    Attributes:
        prices: Mid-price series (shape: N)
        labels: True labels if available (shape: N)
        timestamps_ns: Optional timestamps in nanoseconds (shape: N)
        predictions: Model predictions (shape: N), 0=Down, 1=Stable, 2=Up
        spreads: Bid-ask spread in bps (shape: N)
        agreement_ratio: HMHP cross-horizon agreement (shape: N), in [0.333, 1.0]
        confirmation_score: HMHP decoder confidence (shape: N), in [0, 0.667]
        day_boundaries: Optional [(start_idx, end_idx), ...] half-open row ranges,
            one per trading day, in the coordinates of ``prices``. Supplying this
            ARMS the no-overnight charter (see ``VectorizedEngine.run``). Absent
            (the default), the charter is NOT enforced and positions carry across
            day edges exactly as they did before 2026-08-15.
    """

    prices: np.ndarray
    labels: Optional[np.ndarray] = None
    timestamps_ns: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    spreads: Optional[np.ndarray] = None
    agreement_ratio: Optional[np.ndarray] = None
    confirmation_score: Optional[np.ndarray] = None
    predicted_returns: Optional[np.ndarray] = None
    regression_labels: Optional[np.ndarray] = None
    day_boundaries: Optional[List[Tuple[int, int]]] = None

    def __post_init__(self) -> None:
        """Validate data."""
        if self.prices.ndim != 1:
            raise ValueError(f"prices must be 1D, got shape {self.prices.shape}")
        if len(self.prices) == 0:
            raise ValueError("prices cannot be empty")
        if not np.all(np.isfinite(self.prices)):
            raise ValueError("prices contains NaN or Inf values")
        if np.any(self.prices <= 0):
            raise ValueError("prices must be positive")
        self._validate_day_boundaries()

    def _validate_day_boundaries(self) -> None:
        """Fail loud on a day_boundaries list that does not partition ``prices``.

        The charter fix (2026-08-15) derives session edges from this list. A list
        that does not exactly tile ``[0, len(prices))`` would silently mis-place
        those edges — the same *positional-coordinate* failure mode that let a
        multi-source join fabricate IC +1.0000 on 162/162 days. Row identity is a
        deterministic contract, so a violation raises rather than warns
        (hft-rules §8).
        """
        if self.day_boundaries is None:
            return

        n = len(self.prices)
        if len(self.day_boundaries) == 0:
            raise ValueError(
                "day_boundaries is an empty list. Pass None to run without "
                "charter enforcement; an empty list is ambiguous."
            )

        cursor = 0
        for day_i, bounds in enumerate(self.day_boundaries):
            if len(bounds) != 2:
                raise ValueError(
                    f"day_boundaries[{day_i}] must be a (start_idx, end_idx) pair, got {bounds!r}"
                )
            start, end = int(bounds[0]), int(bounds[1])
            if start != cursor:
                raise ValueError(
                    f"day_boundaries must tile [0, {n}) contiguously: "
                    f"day {day_i} starts at {start}, expected {cursor}. "
                    f"Gaps/overlaps mean the row coordinates do not describe "
                    f"this price array."
                )
            if end <= start:
                raise ValueError(
                    f"day_boundaries[{day_i}] = ({start}, {end}) is empty or "
                    f"reversed; every day must contain at least one row."
                )
            cursor = end

        if cursor != n:
            raise ValueError(
                f"day_boundaries cover {cursor} rows but prices has {n}. "
                f"The boundaries were computed against a different array — "
                f"do not guess an alignment, re-derive them from the loader."
            )

    def __len__(self) -> int:
        return len(self.prices)

    @classmethod
    def from_signal_dir(
        cls,
        signal_dir: str,
        *,
        validate: bool = True,
        expected_fields: Optional[Dict[str, Any]] = None,
    ) -> "BacktestData":
        """Load BacktestData from a directory of exported signal arrays.

        Loads .npy files produced by the trainer's signal export scripts
        (export_hmhp_signals.py, export_regression_signals.py, etc.).

        When validate=True (default), reads signal_metadata.json and checks:
        - Required files exist (ContractError if missing)
        - All arrays have aligned first dimension (ContractError if mismatched)
        - No NaN/Inf in required arrays (ContractError if found)
        - Value ranges are sensible (warnings for anomalies)
        - (Phase II, 2026-04-20) CompatibilityContract producer fingerprint
          self-check (tamper detection)
        - (Phase II hardening SB-1, 2026-04-20) If ``expected_fields`` is
          supplied, each field must match ``manifest.compatibility.<field>``.
          Partial-assertion API: consumer asserts the fields it actually knows
          about (e.g., ``primary_horizon_idx``) and defers everything else to
          the trainer. Typo'd keys raise ``ValueError``.

        Args:
            signal_dir: Path to directory containing .npy signal arrays.
            validate: If True, validate signal contract at load time.
                Set to False for legacy code or manual testing.
            expected_fields: Phase II hardening (2026-04-20 SB-1) — consumer-side
                partial CompatibilityContract assertion. Dict mapping
                ``CompatibilityContract`` field name to expected value. Only
                applied when ``validate=True``. Closes the version-skew
                detection gap left by Phase II (v2.21): producer-side
                fingerprint check was active but consumer-side was never wired
                (backtester never constructed an expected_contract). A partial
                assertion (just ``primary_horizon_idx``) is architecturally
                cleaner than a full 11-field contract the backtester doesn't
                fully know.

        Returns:
            BacktestData populated with all available arrays.

        Raises:
            ContractError: If validation=True and critical contract
                violation detected (missing files, shape mismatch, expected_fields
                mismatch, etc.).
            ValueError: If expected_fields contains a key that is not a
                CompatibilityContract field name (typo detection).
        """
        d = Path(signal_dir)

        # SB-E (Phase II hardening, 2026-04-21): contradictory-args guard.
        # Passing ``expected_fields`` + ``validate=False`` is always a caller
        # bug — the assertion cannot run because ``SignalManifest.validate()``
        # is never invoked. Previous behavior silently dropped the assertion,
        # giving the caller a false sense of version-skew coverage. Per
        # hft-rules §5 fail-fast policy, raise.
        if expected_fields is not None and not validate:
            raise ValueError(
                "BacktestData.from_signal_dir: expected_fields requires "
                "validate=True. Passing validate=False disables SignalManifest."
                "validate(), so expected_fields cannot be asserted. "
                "Either supply validate=True (recommended) or drop expected_fields."
            )

        # Validate signal contract before loading. Retain the manifest beyond
        # validation so the calibration-precedence rule (Phase II D10 fix,
        # 2026-04-20) can use ``manifest.calibration_method`` as the authoritative
        # gate instead of file-existence.
        manifest = None
        if validate:
            from hft_contracts.signal_manifest import SignalManifest

            manifest = SignalManifest.from_signal_dir(d)
            warnings = manifest.validate(d, expected_fields=expected_fields)
            for w in warnings:
                print(f"  ⚠️  Signal validation: {w}")

        prices = np.load(d / "prices.npy", allow_pickle=False)
        labels = (
            np.load(d / "labels.npy", allow_pickle=False) if (d / "labels.npy").exists() else None
        )
        predictions = (
            np.load(d / "predictions.npy", allow_pickle=False)
            if (d / "predictions.npy").exists()
            else None
        )
        spreads = (
            np.load(d / "spreads.npy", allow_pickle=False) if (d / "spreads.npy").exists() else None
        )
        agreement = (
            np.load(d / "agreement_ratio.npy", allow_pickle=False)
            if (d / "agreement_ratio.npy").exists()
            else None
        )
        confirmation = (
            np.load(d / "confirmation_score.npy", allow_pickle=False)
            if (d / "confirmation_score.npy").exists()
            else None
        )

        # Phase II D10 fix (2026-04-20): calibration precedence is MANIFEST-DRIVEN, not
        # file-existence-driven. The OLD pattern silently preferred calibrated_returns.npy
        # whenever the file existed — a stale calibration file from a previous export
        # would silently override the fresh predicted_returns.npy. The manifest's
        # calibration_method field is now the authoritative gate:
        #   - manifest.calibration_method is None      → use predicted_returns.npy
        #   - manifest.calibration_method is not None  → use calibrated_returns.npy
        # SignalManifest.validate() already raises ContractError on orphan files
        # (file exists but manifest claims no calibration, or vice versa), so by the
        # time we reach this point the file/claim alignment is guaranteed.
        # Legacy path (validate=False OR pre-Phase-II manifest without
        # calibration_method): fall back to file-existence semantics for
        # back-compat with R1-R8 ledger signal directories.
        manifest_says_calibrated = manifest is not None and manifest.calibration_method is not None
        if manifest_says_calibrated and (d / "calibrated_returns.npy").exists():
            predicted_returns = np.load(d / "calibrated_returns.npy", allow_pickle=False)
        elif manifest is not None and manifest.calibration_method is None:
            # Manifest EXPLICITLY says no calibration — use predicted regardless
            # of whether a stale calibrated file happens to exist.
            if (d / "predicted_returns.npy").exists():
                predicted_returns = np.load(d / "predicted_returns.npy", allow_pickle=False)
            else:
                predicted_returns = None
        else:
            # Legacy / no-manifest path (pre-Phase-II signal directories + validate=False).
            if (d / "calibrated_returns.npy").exists():
                predicted_returns = np.load(d / "calibrated_returns.npy", allow_pickle=False)
            elif (d / "predicted_returns.npy").exists():
                predicted_returns = np.load(d / "predicted_returns.npy", allow_pickle=False)
            else:
                predicted_returns = None
        regression_labels = (
            np.load(d / "regression_labels.npy", allow_pickle=False)
            if (d / "regression_labels.npy").exists()
            else None
        )

        return cls(
            prices=prices,
            labels=labels,
            predictions=predictions,
            spreads=spreads,
            agreement_ratio=agreement,
            confirmation_score=confirmation,
            predicted_returns=predicted_returns,
            regression_labels=regression_labels,
        )


class VectorizedEngine:
    """
    Numpy-based vectorized backtest engine.

    This engine:
    1. Generates signals from strategy
    2. Simulates position changes
    3. Computes P&L and equity curve
    4. Calculates performance metrics

    Assumptions:
    - Instant fill at current price (no slippage beyond configured)
    - No partial fills
    - Single position at a time
    - Position size is fixed fraction of capital
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize the vectorized engine.

        Args:
            config: Backtest configuration
        """
        self.config = config

    def run(
        self,
        data: BacktestData,
        strategy: Strategy,
        metrics: Optional[List[Metric]] = None,
    ) -> BacktestResult:
        """
        Run the backtest.

        NO-OVERNIGHT CHARTER (2026-08-15). The programme's one hard portfolio
        constraint is that every position opens AND closes inside the same RTH
        session. Until this date NO code enforced it: ``loader.py`` built
        ``day_boundaries`` and nothing read it, the loader concatenated every day
        into one continuous price array, and this loop iterated it with no
        day-edge reset — so a position opened near a day's end carried into the
        next day and booked the OVERNIGHT GAP, the largest return term in the
        series, as intraday P&L. The only bound on a hold was
        ``ZeroDteConfig.max_holding_minutes`` (a config default of 60.0); at 60s
        bars (~390 bars/day) that put ~15.4% of every day within max-hold range
        of a boundary.

        When ``data.day_boundaries`` is supplied, any position still open at the
        LAST row of a day is force-closed at that row's price, through the same
        exit path, cost model and trade-recording as any other exit
        (``_record_close``). Two observability outputs, both on the result:

          * ``metrics["SessionForcedCloses"]`` — how many exits were
            charter-driven rather than signal-driven.
          * ``metrics["SessionCharterEnforced"]`` — 1.0 when boundaries were
            supplied, 0.0 when they were not. An unenforced run must never be
            silently indistinguishable from a compliant one.

        A post-run check then re-derives every round trip's entry and exit day
        and raises if any trade spans a boundary. That is a deterministic
        identity violation, so it fails rather than warns (hft-rules §8).

        When ``data.day_boundaries`` is None the charter is NOT enforced and
        behaviour is byte-identical to the pre-2026-08-15 engine. Discovery
        harnesses that pass bare arrays with no day structure land here.

        Args:
            data: Price and label data. ``data.day_boundaries``, if present,
                arms the charter (see above).
            strategy: Trading strategy
            metrics: Optional list of metrics to compute

        Returns:
            BacktestResult with complete backtest output

        Raises:
            ValueError: If a recorded round trip spans a day boundary despite
                enforcement (an engine-invariant violation, not a user error).
        """
        n = len(data)
        prices = data.prices

        # Charter: last row of each day, excluding the final day. The final
        # day's tail is already handled by the pre-existing end-of-data close
        # below, so including it here would double-count and would silence that
        # path's WARN. Excluding it keeps SessionForcedCloses meaning exactly
        # "closes that would NOT have happened before this fix".
        day_boundaries = data.day_boundaries
        charter_enforced = day_boundaries is not None
        session_close_rows: set = set()
        if day_boundaries is not None:
            session_close_rows = {end - 1 for _, end in day_boundaries[:-1]}
        n_forced_session_closes = 0
        # B1 (2026-06-19): reset the once-per-run guard for the realized-spread
        # fallback WARN (set by _realized_spread_bps on NaN/missing per-row spread).
        self._spread_warn_emitted = False

        # Generate signals
        signal_output = strategy.generate_signals(prices)
        signals = signal_output.signals

        # Initialize tracking arrays
        positions = np.zeros(n, dtype=np.float64)  # +1 long, -1 short, 0 flat
        equity = np.zeros(n, dtype=np.float64)
        equity[0] = self.config.initial_capital

        # Track trades
        trades: List[Trade] = []
        trade_pnls: List[float] = []

        # Current state
        current_position = Position.flat()
        cash = self.config.initial_capital

        # Position size in shares (will be computed per trade)
        # We use fixed fraction of capital for each trade

        # Process each time step
        for i in range(n):
            price = prices[i]
            signal = signals[i]

            # Update unrealized P&L for current position
            if not current_position.is_flat:
                if current_position.is_long:
                    unrealized = (price - current_position.entry_price) * current_position.size
                else:  # Short
                    unrealized = (current_position.entry_price - price) * current_position.size
            else:
                unrealized = 0.0

            # Record current position
            if current_position.is_long:
                positions[i] = current_position.size
            elif current_position.is_short:
                positions[i] = -current_position.size
            else:
                positions[i] = 0.0

            # Process signal
            if signal == Signal.BUY:
                if current_position.is_short:
                    # Close short position first
                    cash, _ = self._record_close(
                        current_position,
                        price,
                        i,
                        cash,
                        trades,
                        trade_pnls,
                        spread_bps_override=self._realized_spread_bps(data, i),
                    )
                    current_position = Position.flat()

                if current_position.is_flat:
                    # Open long position
                    size = self._compute_position_size(cash, price)
                    if size > 0:
                        position_value = size * price
                        cost = self.config.costs.compute_cost(
                            position_value, spread_bps=self._realized_spread_bps(data, i)
                        )
                        # Deduct BOTH position value AND cost from cash
                        # (we're "buying" shares, so cash decreases)
                        cash -= position_value + cost
                        current_position = Position(
                            side=PositionSide.LONG,
                            size=size,
                            entry_price=price,
                            entry_index=i,
                            entry_cost=cost,  # P2 FIX: Store entry cost for trade_pnls
                        )
                        trades.append(
                            Trade(
                                index=i,
                                side=TradeSide.BUY,
                                price=price,
                                size=size,
                                cost=cost,
                            )
                        )

            elif signal == Signal.SELL:
                if not self.config.allow_short and current_position.is_flat:
                    # Can't short, skip
                    pass
                else:
                    if current_position.is_long:
                        # Close long position first
                        cash, _ = self._record_close(
                            current_position,
                            price,
                            i,
                            cash,
                            trades,
                            trade_pnls,
                            spread_bps_override=self._realized_spread_bps(data, i),
                        )
                        current_position = Position.flat()

                    if current_position.is_flat and self.config.allow_short:
                        # Open short position
                        # C3 FIX: Symmetric with longs — deduct BOTH position_value AND cost
                        # Position value acts as margin collateral for the short
                        size = self._compute_position_size(cash, price)
                        if size > 0:
                            position_value = size * price
                            cost = self.config.costs.compute_cost(
                                position_value, spread_bps=self._realized_spread_bps(data, i)
                            )
                            # C3 FIX: Deduct position_value as margin + cost (same as longs)
                            cash -= position_value + cost
                            current_position = Position(
                                side=PositionSide.SHORT,
                                size=size,
                                entry_price=price,
                                entry_index=i,
                                entry_cost=cost,  # P2 FIX: Store entry cost for trade_pnls
                            )
                            trades.append(
                                Trade(
                                    index=i,
                                    side=TradeSide.SELL,
                                    price=price,
                                    size=size,
                                    cost=cost,
                                )
                            )

            elif signal == Signal.EXIT:
                if not current_position.is_flat:
                    cash, _ = self._record_close(
                        current_position,
                        price,
                        i,
                        cash,
                        trades,
                        trade_pnls,
                        spread_bps_override=self._realized_spread_bps(data, i),
                    )
                    current_position = Position.flat()

            # NO-OVERNIGHT CHARTER: force-close at the last row of the day.
            # Placed AFTER signal processing so a signal-driven exit on this same
            # row takes precedence and is never double-counted, and BEFORE the
            # equity update so equity[i] marks the day flat. Uses the same
            # _record_close exit path, cost model and Trade(FLAT) recording as
            # every other exit — there is no second exit path.
            if i in session_close_rows and not current_position.is_flat:
                cash, forced_cost = self._record_close(
                    current_position,
                    price,
                    i,
                    cash,
                    trades,
                    trade_pnls,
                    spread_bps_override=self._realized_spread_bps(data, i),
                )
                logger.info(
                    "Charter force-close at session end: row=%d, size=%g, "
                    "price=%.4f, cost=%.4f. Position was still open at the last "
                    "row of its day; the no-overnight charter forbids carrying "
                    "it across the boundary.",
                    i,
                    current_position.size,
                    price,
                    forced_cost,
                )
                current_position = Position.flat()
                n_forced_session_closes += 1

            # Update equity: cash + position value
            # For long: equity = cash + current_market_value
            # For short: equity = cash + (entry_value - current_value) = cash + unrealized_pnl
            if not current_position.is_flat:
                if current_position.is_long:
                    # Long position: we own shares, value = current price * size
                    # Also we paid entry_price * size from cash, so:
                    # equity = cash + current_price * size
                    current_value = price * current_position.size
                    equity[i] = cash + current_value
                else:
                    # Short position: C3 FIX — margin (entry_price * size) deducted at entry.
                    # Equity = cash + margin_held + unrealized_pnl
                    #        = cash + entry_price * size + (entry_price - current_price) * size
                    #        = cash + entry_price * size * 2 - current_price * size
                    # Simplified: equity = cash + margin + pnl
                    margin = current_position.entry_price * current_position.size
                    unrealized = (current_position.entry_price - price) * current_position.size
                    equity[i] = cash + margin + unrealized
            else:
                equity[i] = cash

        # Close any remaining position at end
        if not current_position.is_flat:
            final_price = prices[-1]
            # FIND-001 fix (2026-05-14): emit Trade(side=FLAT) atomically with trade_pnls.append.
            # Pre-fix: only trade_pnls.append fired; zero_dte.py silent break masked the orphan.
            # See DESIGN_CLUSTER_D1_E_2026_05_14.md §3.1 + VALIDATION_FINDINGS_2026_05_14.md FIND-001.
            # 2026-08-15: routed through _record_close (the single exit path)
            # rather than an inline copy. Arithmetic is unchanged.
            cash, cost = self._record_close(
                current_position,
                final_price,
                n - 1,
                cash,
                trades,
                trade_pnls,
                spread_bps_override=self._realized_spread_bps(data, n - 1),
            )
            equity[-1] = cash
            # hft-rules §8 observability — auto-close should not be silent
            logger.warning(
                "Engine fabricated end-of-data close at bar=%d; strategy did not signal EXIT. "
                "size=%g, price=%.4f, cost=%.4f. Strategies that want signal-driven exit should "
                "emit Trade(side=FLAT) explicitly. See FIND-001.",
                n - 1,
                current_position.size,
                final_price,
                cost,
            )

        # Compute returns
        returns = np.diff(equity) / equity[:-1]
        # Handle division by zero
        returns = np.where(np.isfinite(returns), returns, 0.0)

        # Compute metrics
        computed_metrics = self._compute_metrics(
            returns=returns,
            equity_curve=equity,
            trade_pnls=np.array(trade_pnls),
            predictions=signal_output.signals,
            labels=data.labels,
            metrics=metrics,
        )

        # Charter observability. Injected AFTER _compute_metrics so these two
        # keys are present on every run regardless of the `metrics` list the
        # caller passed (scripts/run_spread_signal_backtest.py passes
        # metrics=[]). A silent behaviour change is exactly what this programme
        # keeps being bitten by: a reader must be able to see both how many
        # exits were charter-driven AND whether the charter ran at all.
        computed_metrics["SessionForcedCloses"] = float(n_forced_session_closes)
        computed_metrics["SessionCharterEnforced"] = 1.0 if charter_enforced else 0.0

        # Deterministic identity check: no round trip may span a day boundary.
        self._assert_no_trade_spans_session(trades, day_boundaries)

        return BacktestResult(
            equity_curve=equity,
            returns=returns,
            positions=positions,
            trades=trades,
            trade_pnls=np.array(trade_pnls),
            prices=prices,
            predictions=signal_output.signals,
            labels=data.labels,
            metrics=computed_metrics,
            config_dict=self.config.to_dict(),
            initial_capital=self.config.initial_capital,
            final_equity=float(equity[-1]),
            total_trades=len(trades),
            start_index=0,
            end_index=n - 1,
        )

    def _record_close(
        self,
        position: Position,
        price: float,
        index: int,
        cash: float,
        trades: List[Trade],
        trade_pnls: List[float],
        *,
        spread_bps_override: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Close ``position`` and record it — THE single exit path.

        Extracted 2026-08-15 so the no-overnight force-close cannot drift from
        signal-driven exits. Every exit in this engine (BUY-reverses-short,
        SELL-reverses-long, EXIT, end-of-data, session force-close) routes here,
        so all five share one cost model, one P&L convention and one
        ``Trade(FLAT)`` emission. The arithmetic is byte-identical to the five
        inline copies it replaced.

        P2 FIX (2026-03-17) preserved: ``trade_pnls`` carries BOTH the entry and
        the exit cost. FIND-001 (2026-05-14) preserved: the ``Trade(FLAT)`` is
        emitted atomically with the ``trade_pnls`` append, so the round-trip
        pairing invariant in ``BacktestResult.__post_init__`` cannot be violated
        by a caller forgetting one of the two.

        Args:
            position: The open position to close (must not be FLAT).
            price: Execution price for the close.
            index: Row index the close is recorded at.
            cash: Cash before the close.
            trades: Trade list, appended in place.
            trade_pnls: Round-trip P&L list, appended in place.
            spread_bps_override: Per-leg realized half-spread, or None for the
                configured flat ``spread_bps``.

        Returns:
            ``(cash_after, cost)`` — cash after settlement, and the exit-leg
            transaction cost (returned so callers can log it).
        """
        cash_flow, cost, pnl = self._close_position(
            position, price, spread_bps_override=spread_bps_override
        )
        cash += cash_flow - cost
        # P2 FIX: Include BOTH entry and exit costs in trade_pnls
        trade_pnls.append(pnl - cost - position.entry_cost)
        trades.append(
            Trade(
                index=index,
                side=TradeSide.FLAT,
                price=price,
                size=position.size,
                cost=cost,
            )
        )
        return cash, cost

    def _assert_no_trade_spans_session(
        self,
        trades: List[Trade],
        day_boundaries: Optional[List[Tuple[int, int]]],
    ) -> None:
        """Fail loud if any recorded round trip crosses a day boundary.

        The charter is enforced by the force-close inside the position loop;
        this re-derives the answer independently from the emitted trade record
        and raises on disagreement. It exists because the programme's recurring
        failure mode is deriving an identity and then never checking it — the
        force-close alone would be an assumption, and an assumption is what let
        ``day_boundaries`` sit unread in ``loader.py`` for the whole life of the
        module.

        No-op when ``day_boundaries`` is None (charter not armed).

        Args:
            trades: The emitted trade list, in ``[open, close, open, close, …]``
                alternation order (the same contract ``ZeroDtePnLTransformer``
                relies on).
            day_boundaries: Validated half-open per-day row ranges, or None.

        Raises:
            ValueError: On a spanning round trip, or on a trade list that does
                not alternate (which would make the span check meaningless).
        """
        if day_boundaries is None or not trades:
            return

        # day_of(row) via the day start offsets: searchsorted is O(log D) and
        # needs no per-row map even at 233 days x ~390 bars.
        day_starts = np.array([start for start, _ in day_boundaries], dtype=np.int64)

        def day_of(row: int) -> int:
            return int(np.searchsorted(day_starts, row, side="right") - 1)

        open_trade: Optional[Trade] = None
        for position_in_list, trade in enumerate(trades):
            if trade.side == TradeSide.FLAT:
                if open_trade is None:
                    raise ValueError(
                        f"Charter check cannot run: trades[{position_in_list}] "
                        f"is a FLAT close with no preceding open. The engine "
                        f"emits strict open/close alternation; a violation here "
                        f"means the trade record itself is corrupt."
                    )
                entry_day = day_of(open_trade.index)
                exit_day = day_of(trade.index)
                if entry_day != exit_day:
                    raise ValueError(
                        f"NO-OVERNIGHT CHARTER VIOLATION: round trip entered at "
                        f"row {open_trade.index} (day index {entry_day}, rows "
                        f"{day_boundaries[entry_day]}) and exited at row "
                        f"{trade.index} (day index {exit_day}, rows "
                        f"{day_boundaries[exit_day]}). The position spanned "
                        f"{exit_day - entry_day} day boundary/boundaries and "
                        f"would have booked the overnight gap as intraday P&L. "
                        f"This is an engine invariant, not a config choice — the "
                        f"session force-close should have prevented it."
                    )
                open_trade = None
            else:
                if open_trade is not None:
                    raise ValueError(
                        f"Charter check cannot run: trades[{position_in_list}] "
                        f"opens while trades[?] at row {open_trade.index} is "
                        f"still open. The engine holds one position at a time "
                        f"and emits strict open/close alternation."
                    )
                open_trade = trade

        if open_trade is not None:
            raise ValueError(
                f"Charter check cannot run: the trade record ends with an "
                f"unclosed open at row {open_trade.index}. The end-of-data "
                f"close should have emitted its Trade(FLAT). See FIND-001."
            )

    def _compute_position_size(self, capital: float, price: float) -> float:
        """
        Compute position size in shares.

        Position sizing uses a fixed fraction of capital, with multiple safeguards:
        1. Position value cannot exceed max_position * initial_capital
        2. Position value cannot exceed available capital (no leverage)
        3. Number of shares is capped to prevent catastrophic short losses

        The share cap is critical: when price is very low, value-based sizing
        would result in huge share counts. If price then rises 100x, a short
        position would lose 100x the value.

        Args:
            capital: Available capital
            price: Current price

        Returns:
            Number of shares to trade
        """
        if capital <= 0 or price <= 0:
            return 0.0

        # Compute target position value as fraction of CURRENT capital
        target_value = capital * self.config.position_size

        # Cap 1: Position value cannot exceed max_position * INITIAL capital
        max_value = self.config.initial_capital * self.config.max_position
        position_value = min(target_value, max_value)

        # Cap 2: Position value cannot exceed available capital (no leverage)
        position_value = min(position_value, capital * 0.95)  # Keep 5% buffer

        # Convert to shares
        size = position_value / price

        # Cap 3: CRITICAL - limit shares to prevent catastrophic short losses
        # Max shares = max_position * initial_capital / reference_price
        # where reference_price is a "reasonable" price estimate
        # We use the larger of current price and the initial capital / 1000
        # This means: if we started with $100k, we assume prices are roughly $100+
        # so max shares ≈ max_position * 1000 = 200 shares at max_position=0.2
        reference_price = max(price, self.config.initial_capital / 1000)
        max_shares = (self.config.initial_capital * self.config.max_position) / reference_price
        size = min(size, max_shares)

        return max(0.0, size)

    def _realized_spread_bps(self, data: "BacktestData", i: int) -> Optional[float]:
        """Per-leg realized spread cost (HALF the per-row quoted bid-ask) in bps,
        or None to use the configured flat ``spread_bps``.

        B1 (2026-06-19): active only when ``config.costs.use_realized_spread``. A
        round-trip crosses the spread twice (buy at the ask, sell at the bid), so
        each leg pays half the quoted spread. Falls back to the flat spread
        (returns None) + WARNs once per run when the per-row spread is unavailable
        / non-finite / negative — observation-tier, never crashes the backtest.
        """
        if not self.config.costs.use_realized_spread:
            return None
        spreads = data.spreads
        if spreads is None or i >= len(spreads):
            if not self._spread_warn_emitted:
                logger.warning(
                    "use_realized_spread=True but per-row spreads are unavailable "
                    "(%s); falling back to flat spread_bps.",
                    "data.spreads is None" if spreads is None else f"len={len(spreads)} <= i={i}",
                )
                self._spread_warn_emitted = True
            return None
        s = float(spreads[i])
        if not np.isfinite(s) or s < 0.0:
            if not self._spread_warn_emitted:
                logger.warning(
                    "use_realized_spread=True but a per-row spread is non-finite/"
                    "negative (spread[%d]=%s); falling back to flat spread_bps for "
                    "affected bars.",
                    i,
                    s,
                )
                self._spread_warn_emitted = True
            return None
        return s / 2.0  # half-spread = per-leg taker crossing cost

    def _close_position(
        self,
        position: Position,
        price: float,
        *,
        spread_bps_override: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Close a position and compute proceeds (for longs) or settlement (for shorts).

        For LONG positions:
            - We sell shares at current price
            - Proceeds = price * size (the full value we receive)
            - P&L = (price - entry_price) * size

        For SHORT positions:
            - We buy back shares at current price to close
            - P&L = (entry_price - price) * size

        Args:
            position: Position to close
            price: Closing price

        Returns:
            (cash_flow, cost, pnl) tuple where:
            - cash_flow: Amount to add to cash (positive for long sells)
            - cost: Transaction cost (always positive)
            - pnl: Actual profit/loss (price difference * size, before costs)
        """
        if position.is_flat:
            return 0.0, 0.0, 0.0

        cost = self.config.costs.compute_cost(position.size * price, spread_bps=spread_bps_override)

        if position.is_long:
            # Selling shares: receive full proceeds (return position_value + P&L)
            cash_flow = price * position.size
            # P&L = (exit - entry) * size
            pnl = (price - position.entry_price) * position.size
        else:  # Short
            # C3 FIX: Since we deducted position_value as margin at entry,
            # we now return margin + P&L at close.
            # P&L = (entry - exit) * size (positive when price drops)
            pnl = (position.entry_price - price) * position.size
            # Return margin (entry_price * size) + P&L
            cash_flow = position.entry_price * position.size + pnl

        return cash_flow, cost, pnl

    def _compute_metrics(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
        trade_pnls: np.ndarray,
        predictions: np.ndarray,
        labels: Optional[np.ndarray],
        metrics: Optional[List[Metric]],
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            returns: Per-period returns
            equity_curve: Equity values
            trade_pnls: P&L per trade
            predictions: Strategy signals
            labels: True labels (if available)
            metrics: Optional list of custom metrics

        Returns:
            Dict of metric name to value
        """
        # Build typed context (backward compatible with dict access)
        from lobbacktest.context import BacktestContext

        # #PY-263 (2026-05-21): config.periods_per_day is now Optional[float] = None
        # (was float = 1000.0). Read resolved_periods_per_day to get the
        # mode-aware-dispatched value: explicit override OR derived from
        # zero_dte.bin_seconds OR legacy 1000.0 fallback with DeprecationWarning.
        # Closes silent Sharpe inflation at sub-daily bins.
        _resolved_ppd = self.config.resolved_periods_per_day
        context = BacktestContext(
            equity_curve=equity_curve,
            trade_pnls=trade_pnls,
            predictions=predictions,
            labels=labels,
            initial_capital=self.config.initial_capital,
            trading_days_per_year=self.config.trading_days_per_year,
            # BacktestContext.periods_per_day field remains ``float = 1000.0`` for
            # backward-compat with dict-protocol metric consumers (risk.py:118
            # ``context.get("periods_per_day", self.periods_per_day)``); engine
            # passes resolved value so consumers see correct annualization.
            periods_per_day=_resolved_ppd,
            annualization_factor=self.config.annualization_factor,
        )

        # Default metrics if none provided — all annualizing metrics receive
        # resolved_periods_per_day for correct sub-daily-bin Sharpe/Sortino/Calmar.
        if metrics is None:
            metrics = [
                TotalReturn(),
                AnnualReturn(
                    trading_days_per_year=self.config.trading_days_per_year,
                    periods_per_day=_resolved_ppd,
                ),
                SharpeRatio(
                    trading_days_per_year=self.config.trading_days_per_year,
                    periods_per_day=_resolved_ppd,
                ),
                SortinoRatio(
                    trading_days_per_year=self.config.trading_days_per_year,
                    periods_per_day=_resolved_ppd,
                ),
                MaxDrawdown(),
                CalmarRatio(
                    trading_days_per_year=self.config.trading_days_per_year,
                    periods_per_day=_resolved_ppd,
                ),
                WinRate(),
                ProfitFactor(),
                AverageWin(),
                AverageLoss(),
                PayoffRatio(),
                Expectancy(),
            ]

            # Add prediction metrics if labels available
            if labels is not None:
                metrics.extend(
                    [
                        DirectionalAccuracy(),
                        SignalRate(),
                    ]
                )

        # Compute all metrics
        result = {}
        for metric in metrics:
            metric_result = metric.compute(returns, context)
            result.update(metric_result)
            # Add to context for dependent metrics
            context.update(metric_result)

        return result


class Backtester:
    """
    Main entry point for running backtests.

    This is a convenience wrapper around VectorizedEngine.

    Example:
        >>> config = BacktestConfig(initial_capital=100_000)
        >>> backtester = Backtester(config)
        >>> result = backtester.run(data, strategy)
        >>> print(result.summary())
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize the backtester.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self._engine = VectorizedEngine(config)

    def run(
        self,
        data: BacktestData,
        strategy: Strategy,
        metrics: Optional[List[Metric]] = None,
    ) -> BacktestResult:
        """
        Run a backtest.

        Args:
            data: BacktestData containing prices and optional labels
            strategy: Trading strategy
            metrics: Optional list of metrics to compute

        Returns:
            BacktestResult with complete output
        """
        return self._engine.run(data, strategy, metrics)

    def run_from_arrays(
        self,
        prices: np.ndarray,
        predictions: np.ndarray,
        labels: Optional[np.ndarray] = None,
        shifted: bool = False,
        metrics: Optional[List[Metric]] = None,
        day_boundaries: Optional[List[Tuple[int, int]]] = None,
    ) -> BacktestResult:
        """
        Convenience method to run backtest from numpy arrays.

        NO-OVERNIGHT CHARTER — OPT-IN HERE, AND OFF BY DEFAULT. This is the
        public bare-array entry point used by discovery harnesses that have no
        day structure to give, so ``day_boundaries`` defaults to None and the
        charter is NOT enforced: positions carry across whatever row edges exist
        in ``prices``, exactly as before 2026-08-15. That is the correct
        behaviour for a caller with a single continuous series, and it is a
        silent-correctness hazard for a caller who concatenated days and forgot
        to say so — which is why an unenforced run is not silently equivalent to
        a compliant one. Every run reports ``metrics["SessionCharterEnforced"]``
        (0.0 here unless boundaries are supplied) alongside
        ``metrics["SessionForcedCloses"]``. Check it before reading a P&L.

        Callers that DID concatenate trading days should pass
        ``day_boundaries``; ``DataLoader.load()`` already builds exactly this
        list and ``LoadedData.to_backtest_data()`` threads it automatically.

        Args:
            prices: Mid-price series
            predictions: Model predictions
            labels: True labels (optional)
            shifted: If predictions use shifted labels (0/1/2)
            metrics: Optional metrics
            day_boundaries: Optional per-day ``(start_idx, end_idx)`` half-open
                row ranges over ``prices``. Must tile ``[0, len(prices))``
                exactly; a partial or overlapping list raises rather than being
                guessed at.

        Returns:
            BacktestResult
        """
        from lobbacktest.strategies.direction import DirectionStrategy

        data = BacktestData(prices=prices, labels=labels, day_boundaries=day_boundaries)
        strategy = DirectionStrategy(predictions, shifted=shifted)
        return self.run(data, strategy, metrics)
