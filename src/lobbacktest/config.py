"""
Configuration schema for LOB-Backtester.

This module defines all configuration dataclasses with:
- Clear documentation of each parameter
- Validation logic
- Serialization support
- Sensible defaults

Configuration Philosophy (from RULE.md):
- All thresholds and behaviors via configuration
- No hardcoded magic numbers
- Sensible defaults with full override capability
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import yaml


# Phase 6 6A.6 (2026-04-17): module-level single-source exchange presets.
# Previously duplicated as a dead `CostConfig.EXCHANGE_PRESETS` class-var AND
# an inline dict inside `for_exchange()` — drift hazard. Now `for_exchange()`
# reads this. Derived from mbo-statistical-profiler 233-day NVDA analysis
# (VWES = Volume-Weighted Effective Spread).
_EXCHANGE_PRESETS: Dict[str, Dict[str, float]] = {
    "XNAS": {
        "spread_bps": 1.0,
        "slippage_bps": 1.97,
        "taker_fee_bps": 0.30,
        "maker_rebate_bps": -0.20,
    },
    "ARCX": {
        "spread_bps": 1.0,
        "slippage_bps": 1.10,
        "taker_fee_bps": 0.25,
        "maker_rebate_bps": -0.15,
    },
}


@dataclass
class CostConfig:
    """
    Transaction cost configuration.

    All costs are in basis points (1 bp = 0.01% = 0.0001).

    Attributes:
        spread_bps: Bid-ask spread cost per trade (default: 1.0 bp)
        slippage_bps: Market impact / slippage per trade (default: 0.5 bp)
        commission_per_trade: Fixed commission per trade in USD (default: 0.0)
        exchange: Exchange name for preset costs (optional: "XNAS", "ARCX")
        maker_rebate_bps: Maker rebate in bps (negative = rebate, default: 0.0)
        taker_fee_bps: Taker fee in bps (default: 0.0)

    Exchange-calibrated presets (from mbo-statistical-profiler):
        XNAS: VWES=1.97 bps, spread_bps=1.0, slippage_bps=1.97
        ARCX: VWES=1.10 bps, spread_bps=1.0, slippage_bps=1.10

    Example:
        >>> costs = CostConfig.for_exchange("XNAS")
        >>> costs.total_bps  # 2.97 bps round-trip
    """

    spread_bps: float = 1.0
    slippage_bps: float = 0.5
    commission_per_trade: float = 0.0
    exchange: Optional[str] = None
    maker_rebate_bps: float = 0.0
    taker_fee_bps: float = 0.0

    # Phase 6 6A.6 (2026-04-17): removed dead `EXCHANGE_PRESETS` field.
    # It was a per-instance default_factory dict never read by any method
    # (verified: `grep -rn "EXCHANGE_PRESETS" lob-backtester/` returns only
    # the former definition line). `for_exchange()` had its own inline
    # duplicate dict — drift hazard. Now single source of truth lives at
    # module level (`_EXCHANGE_PRESETS` below), and `for_exchange()` reads it.

    def __post_init__(self) -> None:
        """Validate cost parameters."""
        if self.spread_bps < 0:
            raise ValueError(f"spread_bps must be >= 0, got {self.spread_bps}")
        if self.slippage_bps < 0:
            raise ValueError(f"slippage_bps must be >= 0, got {self.slippage_bps}")
        if self.commission_per_trade < 0:
            raise ValueError(f"commission_per_trade must be >= 0, got {self.commission_per_trade}")

    @classmethod
    def for_exchange(cls, exchange: str) -> "CostConfig":
        """
        Create a CostConfig calibrated to a specific exchange.

        Costs are derived from mbo-statistical-profiler empirical measurements
        (233-day NVDA analysis, VWES = Volume-Weighted Effective Spread).

        Args:
            exchange: "XNAS" or "ARCX"

        Returns:
            CostConfig with exchange-calibrated parameters.
        """
        # Phase 6 6A.6: reads module-level `_EXCHANGE_PRESETS` (single source).
        if exchange not in _EXCHANGE_PRESETS:
            raise ValueError(f"Unknown exchange: {exchange}. Available: {list(_EXCHANGE_PRESETS.keys())}")
        p = _EXCHANGE_PRESETS[exchange]
        return cls(
            spread_bps=p["spread_bps"],
            slippage_bps=p["slippage_bps"],
            taker_fee_bps=p["taker_fee_bps"],
            maker_rebate_bps=p["maker_rebate_bps"],
            exchange=exchange,
        )

    @property
    def total_bps(self) -> float:
        """Total variable cost in basis points (excludes fixed commission)."""
        return self.spread_bps + self.slippage_bps + self.taker_fee_bps

    def compute_cost(self, notional: float) -> float:
        """
        Compute total transaction cost for a trade.

        Args:
            notional: Trade value in USD (price x size)

        Returns:
            Total cost in USD
        """
        variable_cost = notional * (self.total_bps / 10000.0)
        return variable_cost + self.commission_per_trade


@dataclass
class OpraCalibratedCosts:
    """
    OPRA + IBKR-calibrated option cost model for 0DTE ATM options.

    Spreads from OPRA CMBP-1 profiler (8-day NVDA). Commission from
    318 real IBKR fills (account U17259580, Nov 2025 - Mar 2026).
    Theta from BSM with empirical IV.

    Costs are in USD per contract (1 contract = 100 shares of underlying).

    Attributes:
        atm_call_half_spread: Half bid-ask spread for ATM 0DTE calls (USD/share).
            OPRA median: $0.030 full → $0.015 half. Validated by IBKR 2DTE $0.02.
        atm_put_half_spread: Half bid-ask spread for ATM 0DTE puts (USD/share).
            OPRA median: $0.020 full → $0.010 half.
        atm_call_premium: Median ATM 0DTE call premium (USD/share).
            OPRA: $1.88. Validated by IBKR 0DTE fill median $1.86.
        atm_put_premium: Median ATM 0DTE put premium (USD/share).
            OPRA: $1.31.
        commission_per_contract: IBKR all-inclusive per-contract commission (USD).
            Empirical median from 318 fills: $0.70. Includes broker execution,
            clearing, third-party execution, and regulatory fees.
            For 0DTE specifically: $0.63 (57-fill median).
        implied_vol: Annualized implied volatility for BSM theta calculation.
            OPRA GreeksTracker median for ATM 0DTE: ~0.40.
        entry_minutes_before_close: Minutes before market close at typical entry.
            Default 120 = 14:00 ET entry (2 hours before 16:00 close).

    Source:
        - opra-statistical-profiler/output_opra_nvda/03_ZeroDteTracker.json
        - IBKR-transactions-trades/IBKR_REAL_WORLD_TRADING_REPORT.md
    """

    atm_call_half_spread: float = 0.015
    atm_put_half_spread: float = 0.010
    atm_call_premium: float = 1.88
    atm_put_premium: float = 1.31
    commission_per_contract: float = 0.70
    implied_vol: float = 0.40
    entry_minutes_before_close: float = 120.0

    def __post_init__(self) -> None:
        if self.atm_call_half_spread < 0:
            raise ValueError(f"atm_call_half_spread must be >= 0, got {self.atm_call_half_spread}")
        if self.atm_put_half_spread < 0:
            raise ValueError(f"atm_put_half_spread must be >= 0, got {self.atm_put_half_spread}")
        if self.commission_per_contract < 0:
            raise ValueError(
                f"commission_per_contract must be >= 0, got {self.commission_per_contract}"
            )
        if self.implied_vol <= 0:
            raise ValueError(f"implied_vol must be > 0, got {self.implied_vol}")

    def half_spread(self, is_call: bool) -> float:
        """Half-spread for entry or exit, in USD per share of option."""
        return self.atm_call_half_spread if is_call else self.atm_put_half_spread

    def entry_premium(self, is_call: bool) -> float:
        """Median ATM premium at entry, in USD per share of option."""
        return self.atm_call_premium if is_call else self.atm_put_premium

    def round_trip_cost_per_contract(self, is_call: bool) -> float:
        """
        Total round-trip cost per contract in USD (excluding theta).

        Formula: 2 × (half_spread × 100 + commission)
        """
        spread_per_leg = self.half_spread(is_call) * 100
        return 2 * (spread_per_leg + self.commission_per_contract)

    @classmethod
    def deep_itm(
        cls,
        *,
        implied_vol: float = 0.25,
        entry_minutes_before_close: float = 120.0,
    ) -> "OpraCalibratedCosts":
        """Deep ITM option costs (delta ~0.95).

        Spreads are tighter (deep ITM options have narrower markets),
        theta is negligible. Commission is the same.

        #PY-273 closure (2026-05-16): default ``implied_vol=0.25``
        reflects OPRA empirical Deep ITM IV-skew (~0.20-0.30 vs ATM's
        0.40). Pre-#PY-273 default (0.40) inherited ATM IV and
        overestimated Deep ITM theta by ~60-100% per BSM
        ``θ = -S·σ·N'(0) / (2·√T)`` linearity in σ. NOTE: the
        underlying BSM formula at ``engine/zero_dte.py:77`` remains
        ATM-only (uses ``N'(0) = 0.3989`` regardless of moneyness);
        a full d1/N'(d1) correction requires strike K plumbing
        through the trade dataclass and is deferred to a dedicated
        Phase Z architectural cycle (see #PY-271 + #PY-273 backlog).

        #PY-274 closure (2026-05-16): keyword-only parameters allow
        CLI flag pass-through via ``scripts/run_regression_backtest.py``
        and ``scripts/run_readability_backtest.py`` (``--implied-vol``
        and ``--entry-minutes-before-close``).

        Args:
            implied_vol: Annualized IV for BSM theta (default 0.25;
                OPRA empirical Deep ITM median per
                ``opra-statistical-profiler/output_opra_nvda/``).
                Operator-overridable via CLI for sensitivity sweeps.
            entry_minutes_before_close: Minutes before market close
                at typical entry (default 120 = 14:00 ET). Operator-
                overridable via CLI when signal timing differs.

        Source:
            - IBKR-transactions-trades/COST_AUDIT_2026_03.md
            - opra-statistical-profiler/output_opra_nvda/03_ZeroDteTracker.json
        Breakeven: 1.4 bps at delta=0.95 on $180 NVDA (no theta).
        """
        return cls(
            atm_call_half_spread=0.005,
            atm_put_half_spread=0.005,
            atm_call_premium=20.0,  # deep ITM premium ~$20
            atm_put_premium=20.0,
            commission_per_contract=0.70,
            implied_vol=implied_vol,
            entry_minutes_before_close=entry_minutes_before_close,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atm_call_half_spread": self.atm_call_half_spread,
            "atm_put_half_spread": self.atm_put_half_spread,
            "atm_call_premium": self.atm_call_premium,
            "atm_put_premium": self.atm_put_premium,
            "commission_per_contract": self.commission_per_contract,
            "implied_vol": self.implied_vol,
            "entry_minutes_before_close": self.entry_minutes_before_close,
        }


@dataclass
class ZeroDteConfig:
    """
    Configuration for 0DTE ATM options backtest simulation.

    Models the costs and constraints of trading 0DTE options using
    underlying equity signals. The cost model uses OPRA-calibrated
    empirical data (OpraCalibratedCosts) when available.

    Attributes:
        enabled: Whether to simulate 0DTE option costs (default: False)
        delta: ATM option delta (default: 0.50)
        opra_costs: OPRA-calibrated per-contract cost model
        max_holding_minutes: Maximum holding period in minutes (default: 60)
        target_holding_minutes: Target holding period (default: 15)
        contracts_per_trade: Number of option contracts per trade (default: 1)
        prefer_calls: True → enter calls on Up signals; False → enter puts (default: True)
        entry_window_start_et: Earliest entry time ET (default: "14:00")
        entry_window_end_et: Latest entry time ET (default: "15:30")
        events_per_minute: FIND-NEW-01 closure (2026-05-16) — sampling cadence
            for ``ZeroDtePnLTransformer.holding_minutes = events / events_per_minute``
            derivation. ``None`` means "must be supplied at transformer-construction
            time" (e.g., via ``--events-per-minute`` CLI on the production scripts).
            When both ``events_per_minute`` and ``bin_seconds`` are non-None,
            ``__post_init__`` raises ``ValueError`` (mutually exclusive per
            hft-rules §5). When only ``bin_seconds`` is set, the transformer
            derives ``events_per_minute = 60.0 / bin_seconds`` at construction.
        bin_seconds: FIND-NEW-01 closure (2026-05-16) — sampling bin width in
            seconds (time-based corpora). Sister of ``events_per_minute``; one
            of the two must be supplied for ``ZeroDteConfig.enabled=True``
            YAML configs consumed by ``ExperimentRunner``. For TB v3p0 60s
            corpora: ``bin_seconds: 60`` (→ ``events_per_minute=1.0``).
    """

    enabled: bool = False
    delta: float = 0.50
    opra_costs: OpraCalibratedCosts = field(default_factory=OpraCalibratedCosts)
    max_holding_minutes: float = 60.0
    target_holding_minutes: float = 15.0
    contracts_per_trade: int = 1
    prefer_calls: bool = True
    entry_window_start_et: str = "14:00"
    entry_window_end_et: str = "15:30"
    events_per_minute: Optional[float] = None
    bin_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.delta <= 0.0 or self.delta > 1.0:
            raise ValueError(f"delta must be in (0, 1], got {self.delta}")
        if self.max_holding_minutes <= 0:
            raise ValueError(f"max_holding_minutes must be > 0, got {self.max_holding_minutes}")
        if self.contracts_per_trade < 1:
            raise ValueError(f"contracts_per_trade must be >= 1, got {self.contracts_per_trade}")
        # Wave 2-H H3 + Wave 1A F2 closure (2026-05-17): fail-loud on
        # `prefer_calls=False`. The option-P&L formula at
        # engine/zero_dte.py:354-375 hardcodes ATM-call-like delta sign;
        # selecting PUT spread cost via is_call flag is inconsistent with the
        # P&L direction formula. Currently latent (all production YAMLs +
        # tests use prefer_calls=True, verified via grep), but reachable via
        # Python API. Raising at construction prevents silent-wrong-result
        # exposure per hft-rules §5 fail-fast + §8 never silently produce
        # incoherent semantics. See #PY-311 for full PUT delta sign-convention
        # plumbing (4-6 hr Phase Z architectural; deferred).
        if not self.prefer_calls:
            raise ValueError(
                "ZeroDteConfig: prefer_calls=False is not yet supported. The "
                "option-P&L formula at engine/zero_dte.py:354-375 is "
                "ATM-call-only; PUT delta sign convention is not wired through "
                "Trade dataclass. Selecting prefer_calls=False would produce "
                "mathematically incoherent option-P&L (PUT spread cost paired "
                "with CALL-like P&L direction). See #PY-311 (Phase Z deferred)."
            )
        # FIND-NEW-01 closure (2026-05-16): mutex + type + non-negative
        # validation on sampling-cadence fields. Both can be None (operator
        # passes events_per_minute directly at ZeroDtePnLTransformer
        # construction); but if BOTH are set, fail-loud per hft-rules §5.
        # MF-1 (HIGH, mid-impl gate): explicit isinstance guard prevents
        # YAML type-coercion silent failures (e.g., `events_per_minute: "abc"`
        # → otherwise raises confusing TypeError at the `<= 0` comparison
        # below; this guard raises ValueError with FIND-NEW-01 context).
        if self.events_per_minute is not None and self.bin_seconds is not None:
            raise ValueError(
                "ZeroDteConfig: events_per_minute and bin_seconds are mutually "
                "exclusive (FIND-NEW-01 closure 2026-05-16). Supply exactly one "
                "OR neither (pass at transformer-construction time)."
            )
        if self.events_per_minute is not None:
            if not isinstance(self.events_per_minute, (int, float)) or isinstance(
                self.events_per_minute, bool
            ):
                raise ValueError(
                    f"events_per_minute must be numeric (int/float), got "
                    f"{type(self.events_per_minute).__name__}={self.events_per_minute!r} "
                    f"(FIND-NEW-01 closure 2026-05-16). YAML configs supplying "
                    f"string values likely need quotes removed or numeric "
                    f"casting in the producer side."
                )
            if self.events_per_minute <= 0:
                raise ValueError(
                    f"events_per_minute must be > 0, got {self.events_per_minute}"
                )
        if self.bin_seconds is not None:
            if not isinstance(self.bin_seconds, (int, float)) or isinstance(
                self.bin_seconds, bool
            ):
                raise ValueError(
                    f"bin_seconds must be numeric (int/float), got "
                    f"{type(self.bin_seconds).__name__}={self.bin_seconds!r} "
                    f"(FIND-NEW-01 closure 2026-05-16)."
                )
            if self.bin_seconds <= 0:
                raise ValueError(
                    f"bin_seconds must be > 0, got {self.bin_seconds}"
                )

    @property
    def resolved_events_per_minute(self) -> Optional[float]:
        """FIND-NEW-01 closure (2026-05-16): resolve sampling cadence from YAML.

        Returns the explicit ``events_per_minute`` if set, else derives from
        ``bin_seconds`` (``= 60.0 / bin_seconds``), else ``None`` (operator
        must pass at transformer-construction time).
        """
        if self.events_per_minute is not None:
            return self.events_per_minute
        if self.bin_seconds is not None:
            return 60.0 / self.bin_seconds
        return None


@dataclass
class BacktestConfig:
    """
    Main backtest configuration.

    Attributes:
        initial_capital: Starting capital in USD (default: 100,000)
        position_size: Position size as fraction of capital (default: 0.1 = 10%)
        max_position: Maximum position as fraction of capital (default: 1.0 = 100%)
        costs: Transaction cost configuration
        allow_short: Whether to allow short positions (default: True)
        fill_price: Price used for fills - "close" or "midpoint" (default: "close")
        stop_loss_pct: Optional stop-loss as percentage (e.g., 0.02 = 2%)
        take_profit_pct: Optional take-profit as percentage
        trading_days_per_year: For annualization (default: 252)
        periods_per_day: Trading periods per day (#PY-263 closure 2026-05-21:
            ``Optional[float] = None``, was ``float = 1000.0``). When None,
            ``resolved_periods_per_day`` derives the value from
            ``zero_dte.bin_seconds`` (RTH 6.5 hr × 3600 s = 23400 s → 390 at 60s
            bins) OR falls back to legacy 1000.0 with ``DeprecationWarning``
            per hft-rules §8. Mutex with ``zero_dte.bin_seconds`` per §5
            fail-fast (mirrors ``ZeroDteConfig`` L349-353 events_per_minute/
            bin_seconds mutex). Closes silent Sharpe inflation at sub-daily
            bins (sqrt(1000/390) = 1.6018x at 60s).

    Invariants:
        - initial_capital > 0
        - 0 < position_size <= max_position <= 1.0
        - trading_days_per_year > 0
        - periods_per_day > 0 (when set explicitly); resolved_periods_per_day always > 0
        - NOT both (periods_per_day, zero_dte.bin_seconds) set simultaneously
    """

    initial_capital: float = 100_000.0
    position_size: float = 0.1
    max_position: float = 1.0
    costs: CostConfig = field(default_factory=CostConfig)
    zero_dte: ZeroDteConfig = field(default_factory=ZeroDteConfig)
    allow_short: bool = True
    fill_price: Literal["close", "midpoint"] = "close"
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trading_days_per_year: float = 252.0
    # #PY-263 (2026-05-21): Optional[float] = None enables mode-aware dispatch
    # via ``resolved_periods_per_day`` property. Legacy float = 1000.0 default
    # caused silent ~1.6x Sharpe inflation at 60s bins per Sharpe scaling
    # (sqrt(periods/yr)). All 5 engine sites + 4 scripts + experiment.py
    # migrated to read ``resolved_periods_per_day``. Consumers reading the
    # raw ``periods_per_day`` field directly will see None and must migrate.
    periods_per_day: Optional[float] = None
    min_confidence: Optional[float] = None
    min_agreement: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # FIND-070 closure (2026-05-14): `min_agreement` / `min_confidence` are
        # declared on the BacktestConfig dataclass schema for legacy YAML
        # compat (`BacktestConfig.from_dict` / `load_yaml` path), but NOT
        # consumed by `ExperimentRunner._build_backtest_config` and NOT read
        # by the engine. The live home for these gate values is the `strategy:`
        # block (consumed by `ReadabilityStrategy` via
        # `ExperimentRunner._build_strategy:354-355`). Emit a
        # `DeprecationWarning` when operators set non-None values here so that
        # the wrong-block placement is machine-visible BEFORE the 2026-10-31
        # field-removal cycle (see PHASE_P_BACKLOG.md FIND-070 closure
        # follow-up).
        import warnings
        if self.min_agreement is not None:
            warnings.warn(
                "BacktestConfig.min_agreement is DEPRECATED: this field is not "
                "consumed by ExperimentRunner._build_backtest_config nor read "
                "by the engine. The live home is the `strategy:` block "
                "(consumed by ReadabilityStrategy via _build_strategy). "
                "Migrate to: `strategy: {type: readability, min_agreement: "
                "<value>}` and remove from `backtest:` block. Scheduled for "
                "removal 2026-10-31.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.min_confidence is not None:
            warnings.warn(
                "BacktestConfig.min_confidence is DEPRECATED: this field is not "
                "consumed by ExperimentRunner._build_backtest_config nor read "
                "by the engine. The live home is the `strategy:` block "
                "(consumed by ReadabilityStrategy via _build_strategy). "
                "Migrate to: `strategy: {type: readability, min_confidence: "
                "<value>}` and remove from `backtest:` block. Scheduled for "
                "removal 2026-10-31.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be > 0, got {self.initial_capital}")
        if not (0 < self.position_size <= 1.0):
            raise ValueError(f"position_size must be in (0, 1], got {self.position_size}")
        if not (0 < self.max_position <= 1.0):
            raise ValueError(f"max_position must be in (0, 1], got {self.max_position}")
        if self.position_size > self.max_position:
            raise ValueError(
                f"position_size ({self.position_size}) cannot exceed "
                f"max_position ({self.max_position})"
            )
        if self.trading_days_per_year <= 0:
            raise ValueError(f"trading_days_per_year must be > 0, got {self.trading_days_per_year}")
        # #PY-263 (2026-05-21): periods_per_day is now Optional[float] = None.
        # Validate only when explicitly set; derivation handled by
        # ``resolved_periods_per_day`` property.
        if self.periods_per_day is not None and self.periods_per_day <= 0:
            raise ValueError(f"periods_per_day must be > 0 if set, got {self.periods_per_day}")
        # #PY-263 mutex per hft-rules §5 fail-fast (mirrors ZeroDteConfig L349-353
        # events_per_minute/bin_seconds mutex pattern). Explicit periods_per_day
        # and zero_dte.bin_seconds are mutually exclusive — both specify the
        # same physical quantity (periods per trading day) and would silently
        # produce drift if both set.
        if (
            self.periods_per_day is not None
            and self.zero_dte is not None
            and self.zero_dte.bin_seconds is not None
        ):
            raise ValueError(
                "BacktestConfig: periods_per_day and zero_dte.bin_seconds are "
                "mutually exclusive (#PY-263 mutex). Specify ONE: either "
                "explicit periods_per_day (legacy override) OR "
                "zero_dte.bin_seconds (auto-derives periods_per_day = "
                "23400/bin_seconds for RTH 6.5 hr × 3600 s). Got "
                f"periods_per_day={self.periods_per_day}, "
                f"zero_dte.bin_seconds={self.zero_dte.bin_seconds}."
            )
        if self.stop_loss_pct is not None and self.stop_loss_pct <= 0:
            raise ValueError(f"stop_loss_pct must be > 0 if set, got {self.stop_loss_pct}")
        if self.take_profit_pct is not None and self.take_profit_pct <= 0:
            raise ValueError(f"take_profit_pct must be > 0 if set, got {self.take_profit_pct}")
        if self.fill_price not in ("close", "midpoint"):
            raise ValueError(f"fill_price must be 'close' or 'midpoint', got {self.fill_price}")
        # FIND-058 PARTIAL CLOSURE (#PY-NEW, 2026-05-24): fill_price='midpoint' is
        # DEAD CODE. The field is declared + validated + serialized, but the engine
        # (`engine/vectorized.py` + `engine/zero_dte.py`) NEVER reads it (grep -rn
        # "fill_price\|midpoint" on engine/ returns ZERO matches). Fills always use
        # the time-series price (typically mid_price from feature index 40 of the
        # signal export). 2 production YAMLs silently set `fill_price: midpoint`
        # and get the default close-fill behavior with no operator notice.
        # FIND-058 PARTIAL: REMOVE THIS WARN WHEN engine reads self.fill_price
        # (see vectorized.py + zero_dte.py — Phase 8+ scope ~2-3 hr); until then,
        # surface the silent lie per hft-rules §8 (never silently drop/clamp/fix).
        # Sister-precedent: FIND-070 closure for min_agreement/min_confidence
        # (L466-489 above). Use DeprecationWarning + stacklevel=2 + actionable
        # migration guidance.
        if self.fill_price == "midpoint":
            warnings.warn(
                "BacktestConfig.fill_price='midpoint' is DEAD CODE (FIND-058 "
                "PARTIAL): this field is declared in the schema and validated, "
                "but the engine (vectorized.py + zero_dte.py) NEVER reads it. "
                "Fills always use the time-series price (typically mid_price "
                "from feature index 40). Setting fill_price='midpoint' "
                "silently provides NO benefit — your backtest runs as if "
                "fill_price='close'. Either: (a) wire fill_price into engine "
                "fill semantics (Phase 8+ scope ~2-3 hr), or (b) remove "
                "'fill_price: midpoint' from your YAML (default 'close' is "
                "operationally equivalent today). Scheduled for removal "
                "2026-10-31 if not wired.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def resolved_periods_per_day(self) -> float:
        """Resolve periods_per_day with mode-aware dispatch.

        Phase Y / #PY-263 closure (2026-05-21): closes silent Sharpe inflation
        at sub-daily bins (sqrt(1000/390) = 1.6018x at 60s bins; sqrt(1000/780)
        ≈ 1.13x at 30s bins; etc.). Per hft-rules §8, fallback to legacy
        ``1000.0`` emits ``DeprecationWarning`` so silent degradation is
        machine-visible.

        Resolution precedence:
        1. **Explicit** ``periods_per_day`` (operator override; legacy YAML compat)
        2. **Derive** from ``zero_dte.bin_seconds`` (RTH 6.5 hr × 3600 s = 23400 s)
           e.g. 60s bins → 390, 30s → 780, 5s → 4680
        3. **Legacy fallback** ``1000.0`` with ``DeprecationWarning`` per §8

        Mutex (case 1 vs 2) enforced in ``__post_init__`` per §5 fail-fast.
        """
        if self.periods_per_day is not None:
            return float(self.periods_per_day)
        if self.zero_dte is not None and self.zero_dte.bin_seconds is not None:
            # RTH = 6.5 hr × 3600 s = 23400 s; periods_per_day = RTH / bin_seconds
            return 23400.0 / float(self.zero_dte.bin_seconds)
        # Legacy default fallback — emit observability warning per §8
        import warnings
        warnings.warn(
            "BacktestConfig.resolved_periods_per_day: neither explicit "
            "periods_per_day nor zero_dte.bin_seconds set; falling back to "
            "legacy default 1000.0 which inflates Sharpe by sqrt(1000/X) at "
            "sub-daily bins (e.g. ~1.6x at 60s, ~1.13x at 30s, ~0.46x at 5s). "
            "Set bin_seconds on zero_dte block for auto-derive, OR set "
            "explicit periods_per_day on BacktestConfig, to silence and "
            "produce correct annualization (#PY-263 closure 2026-05-21).",
            DeprecationWarning,
            stacklevel=2,
        )
        return 1000.0

    @property
    def annualization_factor(self) -> float:
        """
        Factor to annualize per-period metrics.

        Returns:
            sqrt(trading_days_per_year * resolved_periods_per_day)

        #PY-263 (2026-05-21): uses ``resolved_periods_per_day`` to honor
        mode-aware dispatch — reads explicit override OR derives from
        ``zero_dte.bin_seconds`` OR falls back to legacy 1000.0 with
        DeprecationWarning. Closes silent Sharpe inflation at sub-daily bins.
        """
        import numpy as np

        return np.sqrt(self.trading_days_per_year * self.resolved_periods_per_day)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a serializable dictionary."""
        result = {
            "initial_capital": self.initial_capital,
            "position_size": self.position_size,
            "max_position": self.max_position,
            "costs": {
                "spread_bps": self.costs.spread_bps,
                "slippage_bps": self.costs.slippage_bps,
                "commission_per_trade": self.costs.commission_per_trade,
                "exchange": self.costs.exchange,
                "taker_fee_bps": self.costs.taker_fee_bps,
                "maker_rebate_bps": self.costs.maker_rebate_bps,
            },
            "allow_short": self.allow_short,
            "fill_price": self.fill_price,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "trading_days_per_year": self.trading_days_per_year,
            "periods_per_day": self.periods_per_day,
            "min_confidence": self.min_confidence,
            "min_agreement": self.min_agreement,
        }
        if self.zero_dte.enabled:
            result["zero_dte"] = {
                "enabled": self.zero_dte.enabled,
                "delta": self.zero_dte.delta,
                "max_holding_minutes": self.zero_dte.max_holding_minutes,
                "target_holding_minutes": self.zero_dte.target_holding_minutes,
                "contracts_per_trade": self.zero_dte.contracts_per_trade,
                "prefer_calls": self.zero_dte.prefer_calls,
                "entry_window_start_et": self.zero_dte.entry_window_start_et,
                "entry_window_end_et": self.zero_dte.entry_window_end_et,
                "opra_costs": self.zero_dte.opra_costs.to_dict(),
                # FIND-NEW-01 closure (2026-05-16): emit sampling-cadence
                # fields so YAML round-trip preserves operator-set values.
                # Only emit when non-None to keep YAML compact for legacy
                # configs that pass events_per_minute at the transformer.
                **(
                    {"events_per_minute": self.zero_dte.events_per_minute}
                    if self.zero_dte.events_per_minute is not None
                    else {}
                ),
                **(
                    {"bin_seconds": self.zero_dte.bin_seconds}
                    if self.zero_dte.bin_seconds is not None
                    else {}
                ),
            }
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BacktestConfig":
        """Create configuration from a dictionary."""
        costs_dict = d.get("costs", {})
        exchange = costs_dict.get("exchange")
        if exchange and exchange in ("XNAS", "ARCX"):
            costs = CostConfig.for_exchange(exchange)
        else:
            costs = CostConfig(
                spread_bps=costs_dict.get("spread_bps", 1.0),
                slippage_bps=costs_dict.get("slippage_bps", 0.5),
                commission_per_trade=costs_dict.get("commission_per_trade", 0.0),
                exchange=exchange,
                taker_fee_bps=costs_dict.get("taker_fee_bps", 0.0),
                maker_rebate_bps=costs_dict.get("maker_rebate_bps", 0.0),
            )

        dte_dict = d.get("zero_dte", {})
        opra_dict = dte_dict.get("opra_costs", {})
        # HF-1 closure (2026-05-16 LATE; Bundle 1 hygiene post Option B Path B'):
        # mode-aware IV default mirroring OpraCalibratedCosts.deep_itm() factory
        # at config.py:209. Reads zero_dte.delta to detect regime:
        #   delta >= 0.90 → Deep ITM (IV=0.25 per #PY-273 OPRA empirical median;
        #     closes ~60-100% theta overestimation when YAML omits implied_vol)
        #   else → ATM (IV=0.40 preserved; correct for atm_call_premium=1.88
        #     and atm_put_premium=1.31 — class default at L173 unchanged)
        # Operator-explicit YAML override always wins via .get() fallback.
        # Sister to experiment.py:_build_zero_dte_config (same pattern).
        _delta = dte_dict.get("delta", 0.50)
        _iv_default = 0.25 if _delta >= 0.90 else 0.40
        opra_costs = OpraCalibratedCosts(
            atm_call_half_spread=opra_dict.get("atm_call_half_spread", 0.015),
            atm_put_half_spread=opra_dict.get("atm_put_half_spread", 0.010),
            atm_call_premium=opra_dict.get("atm_call_premium", 1.88),
            atm_put_premium=opra_dict.get("atm_put_premium", 1.31),
            commission_per_contract=opra_dict.get("commission_per_contract", 0.70),
            implied_vol=opra_dict.get("implied_vol", _iv_default),
            entry_minutes_before_close=opra_dict.get("entry_minutes_before_close", 120.0),
        )
        zero_dte = ZeroDteConfig(
            enabled=dte_dict.get("enabled", False),
            delta=dte_dict.get("delta", 0.50),
            opra_costs=opra_costs,
            max_holding_minutes=dte_dict.get("max_holding_minutes", 60.0),
            target_holding_minutes=dte_dict.get("target_holding_minutes", 15.0),
            contracts_per_trade=dte_dict.get("contracts_per_trade", 1),
            prefer_calls=dte_dict.get("prefer_calls", True),
            entry_window_start_et=dte_dict.get("entry_window_start_et", "14:00"),
            entry_window_end_et=dte_dict.get("entry_window_end_et", "15:30"),
            # FIND-NEW-01 closure (2026-05-16): read sampling cadence from YAML.
            # ``None`` default preserves legacy YAML configs that supply
            # ``events_per_minute`` directly at transformer construction.
            events_per_minute=dte_dict.get("events_per_minute"),
            bin_seconds=dte_dict.get("bin_seconds"),
        )

        return cls(
            initial_capital=d.get("initial_capital", 100_000.0),
            position_size=d.get("position_size", 0.1),
            max_position=d.get("max_position", 1.0),
            costs=costs,
            zero_dte=zero_dte,
            allow_short=d.get("allow_short", True),
            fill_price=d.get("fill_price", "close"),
            stop_loss_pct=d.get("stop_loss_pct"),
            take_profit_pct=d.get("take_profit_pct"),
            trading_days_per_year=d.get("trading_days_per_year", 252.0),
            # #PY-263 (2026-05-21): default None (was 1000.0) enables mode-aware
            # dispatch via ``resolved_periods_per_day`` property. Explicit YAML
            # value preserves legacy behavior; absence triggers derivation from
            # ``zero_dte.bin_seconds`` or DeprecationWarning fallback per §8.
            periods_per_day=d.get("periods_per_day"),
            min_confidence=d.get("min_confidence"),
            min_agreement=d.get("min_agreement"),
        )

    @classmethod
    def load_yaml(cls, path: str) -> "BacktestConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    def save_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
