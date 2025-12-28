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


@dataclass
class CostConfig:
    """
    Transaction cost configuration.

    All costs are in basis points (1 bp = 0.01% = 0.0001).

    Attributes:
        spread_bps: Bid-ask spread cost per trade (default: 1.0 bp)
        slippage_bps: Market impact / slippage per trade (default: 0.5 bp)
        commission_per_trade: Fixed commission per trade in USD (default: 0.0)

    Example:
        For a $100 stock with spread_bps=1.0:
        - Cost per share = $100 * 0.0001 = $0.01
        - Round-trip cost = $0.02 per share
    """

    spread_bps: float = 1.0
    slippage_bps: float = 0.5
    commission_per_trade: float = 0.0

    def __post_init__(self) -> None:
        """Validate cost parameters."""
        if self.spread_bps < 0:
            raise ValueError(f"spread_bps must be >= 0, got {self.spread_bps}")
        if self.slippage_bps < 0:
            raise ValueError(f"slippage_bps must be >= 0, got {self.slippage_bps}")
        if self.commission_per_trade < 0:
            raise ValueError(
                f"commission_per_trade must be >= 0, got {self.commission_per_trade}"
            )

    @property
    def total_bps(self) -> float:
        """Total variable cost in basis points (excludes fixed commission)."""
        return self.spread_bps + self.slippage_bps

    def compute_cost(self, notional: float) -> float:
        """
        Compute total transaction cost for a trade.

        Args:
            notional: Trade value in USD (price × size)

        Returns:
            Total cost in USD
        """
        variable_cost = notional * (self.total_bps / 10000.0)
        return variable_cost + self.commission_per_trade


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
    allow_short: bool = True
    fill_price: Literal["close", "midpoint"] = "close"
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trading_days_per_year: float = 252.0
    periods_per_day: float = 1000.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.initial_capital <= 0:
            raise ValueError(
                f"initial_capital must be > 0, got {self.initial_capital}"
            )
        if not (0 < self.position_size <= 1.0):
            raise ValueError(
                f"position_size must be in (0, 1], got {self.position_size}"
            )
        if not (0 < self.max_position <= 1.0):
            raise ValueError(
                f"max_position must be in (0, 1], got {self.max_position}"
            )
        if self.position_size > self.max_position:
            raise ValueError(
                f"position_size ({self.position_size}) cannot exceed "
                f"max_position ({self.max_position})"
            )
        if self.trading_days_per_year <= 0:
            raise ValueError(
                f"trading_days_per_year must be > 0, got {self.trading_days_per_year}"
            )
        if self.periods_per_day <= 0:
            raise ValueError(
                f"periods_per_day must be > 0, got {self.periods_per_day}"
            )
        if self.stop_loss_pct is not None and self.stop_loss_pct <= 0:
            raise ValueError(
                f"stop_loss_pct must be > 0 if set, got {self.stop_loss_pct}"
            )
        if self.take_profit_pct is not None and self.take_profit_pct <= 0:
            raise ValueError(
                f"take_profit_pct must be > 0 if set, got {self.take_profit_pct}"
            )
        if self.fill_price not in ("close", "midpoint"):
            raise ValueError(
                f"fill_price must be 'close' or 'midpoint', got {self.fill_price}"
            )

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
        return {
            "initial_capital": self.initial_capital,
            "position_size": self.position_size,
            "max_position": self.max_position,
            "costs": {
                "spread_bps": self.costs.spread_bps,
                "slippage_bps": self.costs.slippage_bps,
                "commission_per_trade": self.costs.commission_per_trade,
            },
            "allow_short": self.allow_short,
            "fill_price": self.fill_price,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "trading_days_per_year": self.trading_days_per_year,
            "periods_per_day": self.periods_per_day,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, any]) -> "BacktestConfig":
        """Create configuration from a dictionary."""
        costs_dict = d.get("costs", {})
        costs = CostConfig(
            spread_bps=costs_dict.get("spread_bps", 1.0),
            slippage_bps=costs_dict.get("slippage_bps", 0.5),
            commission_per_trade=costs_dict.get("commission_per_trade", 0.0),
        )
        return cls(
            initial_capital=d.get("initial_capital", 100_000.0),
            position_size=d.get("position_size", 0.1),
            max_position=d.get("max_position", 1.0),
            costs=costs,
            allow_short=d.get("allow_short", True),
            fill_price=d.get("fill_price", "close"),
            stop_loss_pct=d.get("stop_loss_pct"),
            take_profit_pct=d.get("take_profit_pct"),
            trading_days_per_year=d.get("trading_days_per_year", 252.0),
            periods_per_day=d.get("periods_per_day", 1000.0),
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

