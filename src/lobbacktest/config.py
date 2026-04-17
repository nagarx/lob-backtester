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
from typing import Dict, List, Optional, Literal
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
    def deep_itm(cls) -> "OpraCalibratedCosts":
        """Deep ITM option costs (delta ~0.95).

        Spreads are tighter (deep ITM options have narrower markets),
        theta is negligible. Commission is the same.

        Source: IBKR-transactions-trades/COST_AUDIT_2026_03.md
        Breakeven: 1.4 bps at delta=0.95 on $180 NVDA (no theta).
        """
        return cls(
            atm_call_half_spread=0.005,
            atm_put_half_spread=0.005,
            atm_call_premium=20.0,  # deep ITM premium ~$20
            atm_put_premium=20.0,
            commission_per_contract=0.70,
            implied_vol=0.40,
            entry_minutes_before_close=120.0,
        )

    def to_dict(self) -> Dict[str, any]:
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

    def __post_init__(self) -> None:
        if self.delta <= 0.0 or self.delta > 1.0:
            raise ValueError(f"delta must be in (0, 1], got {self.delta}")
        if self.max_holding_minutes <= 0:
            raise ValueError(f"max_holding_minutes must be > 0, got {self.max_holding_minutes}")
        if self.contracts_per_trade < 1:
            raise ValueError(f"contracts_per_trade must be >= 1, got {self.contracts_per_trade}")


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
        periods_per_day: Approximate trading periods per day (default: 1000)

    Invariants:
        - initial_capital > 0
        - 0 < position_size <= max_position <= 1.0
        - trading_days_per_year > 0
        - periods_per_day > 0
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
    periods_per_day: float = 1000.0
    min_confidence: Optional[float] = None
    min_agreement: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
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
        if self.periods_per_day <= 0:
            raise ValueError(f"periods_per_day must be > 0, got {self.periods_per_day}")
        if self.stop_loss_pct is not None and self.stop_loss_pct <= 0:
            raise ValueError(f"stop_loss_pct must be > 0 if set, got {self.stop_loss_pct}")
        if self.take_profit_pct is not None and self.take_profit_pct <= 0:
            raise ValueError(f"take_profit_pct must be > 0 if set, got {self.take_profit_pct}")
        if self.fill_price not in ("close", "midpoint"):
            raise ValueError(f"fill_price must be 'close' or 'midpoint', got {self.fill_price}")

    @property
    def annualization_factor(self) -> float:
        """
        Factor to annualize per-period metrics.

        Returns:
            sqrt(trading_days_per_year * periods_per_day)
        """
        import numpy as np

        return np.sqrt(self.trading_days_per_year * self.periods_per_day)

    def to_dict(self) -> Dict[str, any]:
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
            }
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, any]) -> "BacktestConfig":
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
        opra_costs = OpraCalibratedCosts(
            atm_call_half_spread=opra_dict.get("atm_call_half_spread", 0.015),
            atm_put_half_spread=opra_dict.get("atm_put_half_spread", 0.010),
            atm_call_premium=opra_dict.get("atm_call_premium", 1.88),
            atm_put_premium=opra_dict.get("atm_put_premium", 1.31),
            commission_per_contract=opra_dict.get("commission_per_contract", 0.70),
            implied_vol=opra_dict.get("implied_vol", 0.40),
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
            periods_per_day=d.get("periods_per_day", 1000.0),
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


@dataclass
class ComparisonConfig:
    """
    Configuration for comparing multiple models.

    Attributes:
        models: Dict mapping model name to predictions array
        baseline_name: Optional baseline model name for relative comparisons
    """

    models: Dict[str, any] = field(default_factory=dict)
    baseline_name: Optional[str] = None
