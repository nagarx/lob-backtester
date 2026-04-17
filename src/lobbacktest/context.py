"""Typed backtest context with dict-protocol backward compatibility.

Provides type-safe access to backtest data while maintaining full backward
compatibility with the existing ``context["key"]`` pattern used by all
15+ metrics. No metric code needs to change.

Usage (new code — typed, IDE autocomplete):
    equity = context.equity_curve
    pnls = context.trade_pnls

Usage (old code — backward compatible, all metrics unchanged):
    equity = context["equity_curve"]
    sharpe = context.get("SharpeRatio", 0.0)

The engine builds a BacktestContext, passes it to metrics as before, and
calls ``context.update(metric_result)`` after each metric computes.
Subsequent metrics can read prior results via ``context["MetricName"]``.

Reference:
    BACKTESTER_AUDIT_PLAN.md § Phase 2b
    engine/vectorized.py lines 494-550 (context build + metric iteration)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class BacktestContext:
    """Typed context for metric computation.

    Supports both typed attribute access AND dict-style access.
    Metrics can use ``context["key"]`` (backward compatible) or
    ``context.equity_curve`` (typed, IDE-friendly).

    Core data (set at creation, conceptually immutable):
        equity_curve: Equity values at each timestep [N].
        trade_pnls: P&L per closed trade [M] (includes entry+exit costs).
        predictions: Model predictions [N] (optional, for classification metrics).
        labels: True labels [N] (optional, for classification metrics).

    Configuration (set at creation, with defaults matching BacktestConfig):
        initial_capital: Starting capital in USD.
        trading_days_per_year: For annualization (default 252).
        periods_per_day: Samples per trading day (default 1000).
        annualization_factor: sqrt(trading_days_per_year * periods_per_day).

    Computed metrics (added during iteration by engine):
        Stored in _computed dict. E.g., CalmarRatio reads
        context["AnnualReturn"] pre-computed by AnnualReturn metric.
    """

    # --- Core data arrays ---
    equity_curve: np.ndarray
    trade_pnls: np.ndarray
    predictions: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None

    # --- Configuration with sensible defaults ---
    initial_capital: float = 1.0
    trading_days_per_year: float = 252.0
    periods_per_day: float = 1000.0
    annualization_factor: float = 0.0

    def __post_init__(self) -> None:
        """Compute derived fields."""
        if self.annualization_factor == 0.0:
            self.annualization_factor = math.sqrt(
                self.trading_days_per_year * self.periods_per_day
            )
        # Internal dict for computed metric results during iteration.
        # Not a dataclass field to avoid it appearing in repr/eq.
        object.__setattr__(self, "_computed", {})

    # --- Dict protocol (backward compatibility for all metrics) ---

    def __getitem__(self, key: str) -> Any:
        """``context["equity_curve"]`` returns typed field or computed metric.

        Raises:
            KeyError: If key not found in typed fields or computed metrics.
        """
        computed: Dict[str, Any] = object.__getattribute__(self, "_computed")
        if key in computed:
            return computed[key]
        if not key.startswith("_") and hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        """``"equity_curve" in context`` returns True."""
        if not isinstance(key, str):
            return False
        computed: Dict[str, Any] = object.__getattribute__(self, "_computed")
        return key in computed or (not key.startswith("_") and hasattr(self, key))

    def get(self, key: str, default: Any = None) -> Any:
        """``context.get("TotalReturn", 0.0)`` — same semantics as dict.get().

        Args:
            key: Attribute name or computed metric name.
            default: Value to return if key not found.

        Returns:
            The value for key, or default if not found.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, d: dict) -> None:
        """``context.update({"SharpeRatio": 1.5})`` — stores computed metrics.

        Called by the engine after each metric computes, so subsequent
        metrics can read prior results (e.g., CalmarRatio reads AnnualReturn).

        Args:
            d: Dict of metric name → computed value.
        """
        computed: Dict[str, Any] = object.__getattribute__(self, "_computed")
        computed.update(d)
