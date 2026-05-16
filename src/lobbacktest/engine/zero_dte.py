"""
0DTE ATM Options P&L Transformer (IBKR + OPRA Calibrated).

Converts equity backtest trades into 0DTE ATM option P&L using:
  - Spreads from OPRA CMBP-1 profiler (8-day NVDA empirical)
  - Commission from 318 real IBKR fills ($0.70/contract all-inclusive)
  - Theta from BSM with empirical IV (replaces broken 10 bps/min constant)

Per-trade option P&L model:
    gross_pnl = delta * (exit_price - entry_price) * 100 * contracts
    spread_cost = 2 * half_spread * 100 * contracts
    commission_cost = 2 * commission_per_contract * contracts
    theta_cost = bsm_theta_per_share(S, sigma, T) * holding_min * 100 * contracts
    option_pnl = gross_pnl - spread_cost - commission_cost - theta_cost

BSM theta for ATM 0DTE (validated against IBKR):
    theta_per_min = S * sigma * N'(0) / (2 * sqrt(T)) / (252 * 390)
    At 14:00 (T=120min): 0.23 bps/min   ($0.42/contract/min on $180 stock)
    At 15:30 (T=30min):  0.47 bps/min   ($0.84/contract/min)
    At 15:50 (T=10min):  0.81 bps/min   ($1.45/contract/min)

Source:
    - opra-statistical-profiler/output_opra_nvda/03_ZeroDteTracker.json
    - IBKR-transactions-trades/IBKR_REAL_WORLD_TRADING_REPORT.md
    - BSM: Black & Scholes (1973), theta = -S*sigma*N'(0) / (2*sqrt(T))
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from lobbacktest.config import ZeroDteConfig, OpraCalibratedCosts
from lobbacktest.types import BacktestResult, Trade, TradeSide


EPS = 1e-12
NPRIME_ZERO = 1.0 / math.sqrt(2.0 * math.pi)
TRADING_MINUTES_PER_YEAR = 252.0 * 390.0


def theta_bsm_per_share(
    underlying_price: float,
    implied_vol: float,
    minutes_remaining: float,
    holding_minutes: float,
) -> float:
    """
    BSM-based theta decay for ATM 0DTE option, in USD per share.

    For ATM options (d1 ≈ 0), theta simplifies to:
        theta_annual = S * sigma * N'(0) / (2 * sqrt(T))
        theta_per_minute = theta_annual / (252 * 390)

    The cost over a holding period is the integral of instantaneous theta,
    but for short holds (minutes) relative to T, linear approximation holds.

    Reference: Black & Scholes (1973), Hull (2018) Ch 19.

    Args:
        underlying_price: Stock price at entry (USD).
        implied_vol: Annualized implied volatility (e.g. 0.40 for 40%).
        minutes_remaining: Minutes until market close at entry.
        holding_minutes: How long the position is held (minutes).

    Returns:
        Theta cost in USD per share of option (multiply by 100 for per-contract).
    """
    if minutes_remaining < 1.0 or holding_minutes <= 0.0:
        return 0.0

    t_years = minutes_remaining / TRADING_MINUTES_PER_YEAR
    if t_years < EPS:
        return 0.0

    theta_annual = underlying_price * implied_vol * NPRIME_ZERO / (2.0 * math.sqrt(t_years))
    theta_per_min = theta_annual / TRADING_MINUTES_PER_YEAR

    return theta_per_min * holding_minutes


@dataclass
class ZeroDteResult:
    """Result of 0DTE IBKR+OPRA-calibrated P&L transformation.

    Attributes:
        equity_result: Original equity BacktestResult
        option_trade_pnls: Per-trade option P&L in USD (shape: n_round_trips)
        option_equity_curve: Cumulative option equity (shape: n_round_trips + 1)
        option_total_return: Total return under option P&L
        option_final_equity: Final equity under option P&L
        spread_costs: Per-trade option spread cost in USD (shape: n_round_trips)
        commission_costs: Per-trade commission in USD (shape: n_round_trips)
        theta_costs: Per-trade theta decay cost in USD (shape: n_round_trips)
        holding_periods_events: Per-trade holding period in events (shape: n_round_trips)
        holding_periods_minutes: Per-trade holding period in minutes (shape: n_round_trips)
        underlying_moves_bps: Per-trade underlying move in bps (shape: n_round_trips)
        is_call: Per-trade flag, True=call, False=put (shape: n_round_trips)
        config: ZeroDteConfig used
    """

    equity_result: BacktestResult
    option_trade_pnls: np.ndarray
    option_equity_curve: np.ndarray
    option_total_return: float
    option_final_equity: float
    spread_costs: np.ndarray
    commission_costs: np.ndarray
    theta_costs: np.ndarray
    holding_periods_events: np.ndarray
    holding_periods_minutes: np.ndarray
    underlying_moves_bps: np.ndarray
    is_call: np.ndarray
    config: ZeroDteConfig

    @property
    def n_trades(self) -> int:
        return len(self.option_trade_pnls)

    @property
    def option_win_rate(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return float((self.option_trade_pnls > 0).mean())

    @property
    def avg_spread_cost(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return float(self.spread_costs.mean())

    @property
    def avg_commission_cost(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return float(self.commission_costs.mean())

    @property
    def avg_theta_cost(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return float(self.theta_costs.mean())

    @property
    def avg_holding_minutes(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return float(self.holding_periods_minutes.mean())

    @property
    def avg_underlying_move_bps(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return float(self.underlying_moves_bps.mean())

    @property
    def total_cost(self) -> float:
        return float(self.spread_costs.sum() + self.commission_costs.sum() + self.theta_costs.sum())

    def summary(self) -> str:
        oc = self.config.opra_costs
        lines = [
            "0DTE IBKR+OPRA-Calibrated Option P&L Summary",
            "=" * 50,
            f"Trades: {self.n_trades}",
            f"Contracts/trade: {self.config.contracts_per_trade}",
            f"Delta: {self.config.delta}",
            f"",
            f"--- Option P&L ---",
            f"Total return: {self.option_total_return:.4%}",
            f"Final equity: ${self.option_final_equity:,.2f}",
            f"Win rate: {self.option_win_rate:.2%}",
            f"Avg P&L/trade: ${float(self.option_trade_pnls.mean()) if self.n_trades > 0 else 0:.4f}",
            f"",
            f"--- Cost Breakdown (per trade avg) ---",
            f"Spread cost: ${self.avg_spread_cost:.4f}",
            f"Commission: ${self.avg_commission_cost:.4f}",
            f"Theta (BSM): ${self.avg_theta_cost:.4f}",
            f"Total cost/trade: ${(self.avg_spread_cost + self.avg_commission_cost + self.avg_theta_cost):.4f}",
            f"",
            f"--- Holding ---",
            f"Avg hold: {self.avg_holding_minutes:.1f} min",
            f"Avg underlying move: {self.avg_underlying_move_bps:+.2f} bps",
            f"",
            f"--- Cost Model (IBKR-validated) ---",
            f"Call half-spread: ${oc.atm_call_half_spread:.3f} (OPRA median)",
            f"Put half-spread: ${oc.atm_put_half_spread:.3f} (OPRA median)",
            f"Commission: ${oc.commission_per_contract:.2f}/contract (IBKR 318-fill median)",
            f"Theta model: BSM (IV={oc.implied_vol:.0%}, entry {oc.entry_minutes_before_close:.0f}min before close)",
            f"Round-trip (call): ${oc.round_trip_cost_per_contract(True):.2f}/contract",
            f"Round-trip (put): ${oc.round_trip_cost_per_contract(False):.2f}/contract",
            f"",
            f"--- Equity Backtest Reference ---",
            f"Equity total return: {self.equity_result.total_return:.4%}",
            f"Equity final: ${self.equity_result.final_equity:,.2f}",
        ]
        return "\n".join(lines)


class ZeroDteAlternationError(ValueError):
    """Raised when ZeroDtePnLTransformer detects a violated alternation contract.

    The transformer requires strict [open, close, open, close, ...] alternation in trades:
    entries (sides BUY|SELL) at even indices, exits (side FLAT) at odd indices.
    Post-FIND-001 fix this is structurally guaranteed by the engine; this exception is
    reserved for future regression detection or external Trade-stream consumers.

    See ``DESIGN_CLUSTER_D1_E_2026_05_14.md`` §3.3 + ``VALIDATION_FINDINGS_2026_05_14.md`` FIND-003.
    """


class ZeroDtePnLTransformer:
    """
    Transforms equity backtest trades into IBKR+OPRA-calibrated 0DTE option P&L.

    Cost model validated against 318 real IBKR fills and BSM theta.

    Args:
        config: ZeroDteConfig with IBKR-calibrated cost model
        events_per_minute: Events per minute for holding period conversion.
            **REQUIRED — no silent default.**

            FIND-NEW-01 closure (2026-05-16): pre-fix this parameter defaulted
            to ``10.0`` (calibrated for event-based corpora ~1000 events/day).
            Time-based corpora (e.g., TB v3p0 60s bins → 1.0 event/min)
            silently inherited the wrong calibration, causing
            ``holding_minutes = events / 10.0`` to under-report hold time by
            ~10x → theta cost under-reported by ~10x → R-17a/R-19/R-16d/R-16e
            absolute cost-economics analyses were all biased. Comparative
            ranking (cross-arm within a single sweep) was preserved because
            the bias applied uniformly.

            Operator MUST specify:
              * Time-based corpora: ``events_per_minute = 60.0 / bin_seconds``
                - TB v3p0 60s bins → ``events_per_minute=1.0``
                - TB v3p0 30s bins → ``events_per_minute=2.0``
                - TB v3p0 5s  bins → ``events_per_minute=12.0``
              * Event-based corpora: operator-supplied true rate (typical
                ~10 events/min for active trading hours).

            CLI helpers ``--bin-seconds`` / ``--events-per-minute`` on
            ``scripts/run_regression_backtest.py`` and
            ``scripts/run_readability_backtest.py`` enforce this at argparse
            time (mutually exclusive, at least one required).

            See ``lob-backtester/VALIDATION_FINDINGS_2026_05_14.md`` FIND-NEW-01
            (filed 2026-05-16) for the full closure narrative.

    Raises:
        ValueError: if ``events_per_minute <= 0`` (fail-loud per hft-rules §5).
    """

    def __init__(
        self,
        config: ZeroDteConfig,
        events_per_minute: float,
    ):
        if events_per_minute <= 0:
            raise ValueError(
                f"events_per_minute must be > 0, got {events_per_minute}. "
                f"FIND-NEW-01 (2026-05-16) closed the silent-default class: "
                f"pre-fix this defaulted to 10.0, miscalibrated for time-based "
                f"corpora (TB v3p0 60s → true 1.0 events/min). Pass an explicit "
                f"value derived from your corpus sampling cadence: "
                f"events_per_minute = 60.0 / bin_seconds."
            )
        self.config = config
        self.events_per_minute = events_per_minute

    def transform(self, result: BacktestResult) -> ZeroDteResult:
        """
        Transform equity BacktestResult into IBKR+OPRA-calibrated ZeroDteResult.

        Args:
            result: Equity backtest result with trades and P&L.

        Returns:
            ZeroDteResult with option-adjusted P&L.
        """
        trades = result.trades
        equity_pnls = result.trade_pnls
        n_round_trips = len(equity_pnls)

        empty = np.array([], dtype=np.float64)
        if n_round_trips == 0:
            return ZeroDteResult(
                equity_result=result,
                option_trade_pnls=empty,
                option_equity_curve=np.array([result.initial_capital]),
                option_total_return=0.0,
                option_final_equity=result.initial_capital,
                spread_costs=empty,
                commission_costs=empty,
                theta_costs=empty,
                holding_periods_events=empty,
                holding_periods_minutes=empty,
                underlying_moves_bps=empty,
                is_call=np.array([], dtype=bool),
                config=self.config,
            )

        # FIND-003 fix (2026-05-14): alternation contract precondition.
        # Post-FIND-001 fix, engine emits Trade(FLAT) symmetrically with trade_pnls.append.
        # Contract: each closed round-trip = 1 entry trade (BUY|SELL) + 1 exit trade (FLAT).
        # Therefore len(trades) MUST equal 2 * n_round_trips. Per hft-rules §5 fail-fast.
        # See DESIGN_CLUSTER_D1_E_2026_05_14.md §3.3 + VALIDATION_FINDINGS_2026_05_14.md FIND-003.
        expected_n_trades = n_round_trips * 2
        if len(trades) != expected_n_trades:
            raise ZeroDteAlternationError(
                f"ZeroDte expects 2 trades per round-trip (open + close); "
                f"got n_round_trips={n_round_trips} (from len(equity_pnls)) "
                f"but len(trades)={len(trades)}, expected {expected_n_trades}. "
                f"Engine should emit Trade(side=FLAT) symmetrically with trade_pnls.append. "
                f"See FIND-001/FIND-002/FIND-003 cluster."
            )

        oc = self.config.opra_costs
        contracts = self.config.contracts_per_trade
        delta = self.config.delta

        option_pnls = np.zeros(n_round_trips, dtype=np.float64)
        spread_costs_arr = np.zeros(n_round_trips, dtype=np.float64)
        commission_costs_arr = np.zeros(n_round_trips, dtype=np.float64)
        theta_costs_arr = np.zeros(n_round_trips, dtype=np.float64)
        holding_events_arr = np.zeros(n_round_trips, dtype=np.float64)
        holding_minutes_arr = np.zeros(n_round_trips, dtype=np.float64)
        underlying_moves_arr = np.zeros(n_round_trips, dtype=np.float64)
        is_call_arr = np.zeros(n_round_trips, dtype=bool)

        for i in range(n_round_trips):
            entry_idx = i * 2
            exit_idx = i * 2 + 1
            # FIND-003 fix (2026-05-14): the precondition above guarantees
            # len(trades) == 2 * n_round_trips, so exit_idx < len(trades) is
            # structurally guaranteed. The pre-2026-05-14 silent ``break`` here
            # was hiding FIND-001 by truncating option-mode round-trips whenever
            # the engine left an open EOF position. Replaced with explicit
            # per-pair alternation assert below (catches reordering bugs +
            # future regressions). See FIND-003.

            entry_trade = trades[entry_idx]
            exit_trade = trades[exit_idx]

            # FIND-003: per-pair alternation assert
            if entry_trade.side == TradeSide.FLAT or exit_trade.side != TradeSide.FLAT:
                raise ZeroDteAlternationError(
                    f"ZeroDte alternation violated at round-trip {i}: "
                    f"entry@{entry_idx}.side={entry_trade.side.name}, "
                    f"exit@{exit_idx}.side={exit_trade.side.name}. "
                    f"Expected entry in {{BUY, SELL}} + exit == FLAT."
                )

            is_call = entry_trade.side.value > 0 if self.config.prefer_calls else entry_trade.side.value < 0
            is_call_arr[i] = is_call

            entry_price = entry_trade.price
            exit_price = exit_trade.price

            holding_events = max(1, abs(exit_trade.index - entry_trade.index))
            holding_minutes = min(
                holding_events / self.events_per_minute,
                self.config.max_holding_minutes,
            )
            holding_events_arr[i] = holding_events
            holding_minutes_arr[i] = holding_minutes

            if entry_price > EPS:
                direction = 1 if entry_trade.side.value > 0 else -1
                move_bps = direction * (exit_price - entry_price) / entry_price * 10000.0
            else:
                move_bps = 0.0
            underlying_moves_arr[i] = move_bps

            gross_pnl = delta * (move_bps / 10000.0) * entry_price * 100 * contracts

            half_sp = oc.half_spread(is_call)
            spread_cost = 2 * half_sp * 100 * contracts
            spread_costs_arr[i] = spread_cost

            comm_cost = 2 * oc.commission_per_contract * contracts
            commission_costs_arr[i] = comm_cost

            theta_cost_per_share = theta_bsm_per_share(
                underlying_price=entry_price,
                implied_vol=oc.implied_vol,
                minutes_remaining=oc.entry_minutes_before_close,
                holding_minutes=holding_minutes,
            )
            theta_cost = theta_cost_per_share * 100 * contracts
            theta_costs_arr[i] = theta_cost

            option_pnls[i] = gross_pnl - spread_cost - comm_cost - theta_cost

        option_equity = result.initial_capital + np.cumsum(
            np.concatenate([[0], option_pnls])
        )

        return ZeroDteResult(
            equity_result=result,
            option_trade_pnls=option_pnls,
            option_equity_curve=option_equity,
            option_total_return=float((option_equity[-1] / result.initial_capital) - 1),
            option_final_equity=float(option_equity[-1]),
            spread_costs=spread_costs_arr,
            commission_costs=commission_costs_arr,
            theta_costs=theta_costs_arr,
            holding_periods_events=holding_events_arr,
            holding_periods_minutes=holding_minutes_arr,
            underlying_moves_bps=underlying_moves_arr,
            is_call=is_call_arr,
            config=self.config,
        )
