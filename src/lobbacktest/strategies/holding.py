"""
Holding policy framework for position exit decisions.

Determines when to exit a position once entered, solving the signal
flickering problem where readability gates pass/fail on consecutive
events and cause excessive trading.

The fundamental insight: the HMHP model predicts H10 (10 events ahead).
The holding policy should be aligned to the prediction horizon --
hold for at least as long as the model's forecast window.

Policies:
    HorizonAlignedPolicy: Hold for exactly N events (default: H10 = 10)
    DirectionReversalPolicy: Hold until direction reverses or max hold
    StopLossTakeProfitPolicy: Exit on SL/TP thresholds or max hold
    CompositePolicy: Combine policies with AND/OR logic

Reference:
    First backtest without holding: 14,051 trades, -36.79% return
    despite 95.50% directional accuracy. Signal flickering is the cause.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from lobbacktest.labels import LabelMapping, SHIFTED_MAPPING


@dataclass
class HoldingState:
    """State passed to HoldingPolicy at each event while in a position.

    Attributes:
        events_held: Number of events since position entry.
        entry_prediction: Model prediction at entry (0=Down, 1=Stable, 2=Up).
        current_prediction: Current model prediction.
        current_agreement: Current agreement_ratio from HMHP.
        current_confirmation: Current confirmation_score from HMHP.
        current_spread: Current bid-ask spread in bps.
        entry_price: Mid price at position entry.
        current_price: Current mid price.
        unrealized_pnl_bps: Unrealized P&L in basis points of entry price.
        position_side: +1 for long, -1 for short.
    """

    events_held: int
    entry_prediction: int
    current_prediction: int
    current_agreement: float
    current_confirmation: float
    current_spread: float
    entry_price: float
    current_price: float
    unrealized_pnl_bps: float
    position_side: int  # +1 long, -1 short


class HoldingPolicy(ABC):
    """Abstract base for position exit decisions."""

    @property
    @abstractmethod
    def policy_name(self) -> str:
        """Unique name for this policy configuration."""
        ...

    @abstractmethod
    def should_exit(self, state: HoldingState) -> bool:
        """Return True if the position should be closed."""
        ...

    def to_dict(self) -> dict:
        """Serialize policy config for registry/logging."""
        return {"type": self.policy_name}


class HorizonAlignedPolicy(HoldingPolicy):
    """
    Hold for exactly N events, aligned to the prediction horizon.

    The model predicts direction H events ahead. Holding for exactly H
    events ensures each trade uses exactly one prediction window.
    After H events, exit and re-evaluate.

    This is the most principled policy because it matches execution
    to what the model was actually trained to predict.

    Args:
        hold_events: Number of events to hold (default: 10 for H10).
    """

    def __init__(self, hold_events: int = 10):
        if hold_events < 1:
            raise ValueError(f"hold_events must be >= 1, got {hold_events}")
        self.hold_events = hold_events

    @property
    def policy_name(self) -> str:
        return f"horizon_aligned_{self.hold_events}"

    def should_exit(self, state: HoldingState) -> bool:
        return state.events_held >= self.hold_events

    def to_dict(self) -> dict:
        return {"type": "horizon_aligned", "hold_events": self.hold_events}


class DirectionReversalPolicy(HoldingPolicy):
    """
    Hold until the model reverses direction or max holding period.

    Stays in position as long as the model predicts the same direction
    (or Stable). Only exits when:
    1. Model predicts the OPPOSITE direction (Up->Down or Down->Up)
    2. Max holding period reached
    3. Optionally: readability gate fails (require_gate=True)

    Args:
        max_hold_events: Maximum events before forced exit (default: 60 for H60).
        require_gate: If True, also exit when agreement drops below 1.0.
    """

    def __init__(
        self,
        max_hold_events: int = 60,
        require_gate: bool = False,
        label_mapping: Optional[LabelMapping] = None,
    ):
        if max_hold_events < 1:
            raise ValueError(f"max_hold_events must be >= 1, got {max_hold_events}")
        self.max_hold_events = max_hold_events
        self.require_gate = require_gate
        self.label_mapping = label_mapping or SHIFTED_MAPPING

    @property
    def policy_name(self) -> str:
        gate_str = "_gated" if self.require_gate else ""
        return f"direction_reversal_{self.max_hold_events}{gate_str}"

    def should_exit(self, state: HoldingState) -> bool:
        if state.events_held >= self.max_hold_events:
            return True

        reversed_direction = self.label_mapping.is_reversal(
            state.entry_prediction, state.current_prediction
        )
        if reversed_direction:
            return True

        if self.require_gate and state.current_agreement < 1.0:
            return True

        return False

    def to_dict(self) -> dict:
        return {
            "type": "direction_reversal",
            "max_hold_events": self.max_hold_events,
            "require_gate": self.require_gate,
        }


class StopLossTakeProfitPolicy(HoldingPolicy):
    """
    Exit on stop-loss, take-profit, or max holding period.

    Implements risk management gates independent of model predictions.
    Uses unrealized P&L in basis points relative to entry price.

    Args:
        stop_loss_bps: Exit if unrealized loss exceeds this (positive value).
        take_profit_bps: Exit if unrealized gain exceeds this.
        max_hold_events: Maximum events before forced exit.
    """

    def __init__(
        self,
        stop_loss_bps: float = 10.0,
        take_profit_bps: float = 20.0,
        max_hold_events: int = 60,
    ):
        if stop_loss_bps <= 0:
            raise ValueError(f"stop_loss_bps must be > 0, got {stop_loss_bps}")
        if take_profit_bps <= 0:
            raise ValueError(f"take_profit_bps must be > 0, got {take_profit_bps}")
        self.stop_loss_bps = stop_loss_bps
        self.take_profit_bps = take_profit_bps
        self.max_hold_events = max_hold_events

    @property
    def policy_name(self) -> str:
        return f"sltp_sl{self.stop_loss_bps:.0f}_tp{self.take_profit_bps:.0f}"

    def should_exit(self, state: HoldingState) -> bool:
        if state.events_held >= self.max_hold_events:
            return True
        if state.unrealized_pnl_bps <= -self.stop_loss_bps:
            return True
        if state.unrealized_pnl_bps >= self.take_profit_bps:
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "type": "stop_loss_take_profit",
            "stop_loss_bps": self.stop_loss_bps,
            "take_profit_bps": self.take_profit_bps,
            "max_hold_events": self.max_hold_events,
        }


class CompositePolicy(HoldingPolicy):
    """
    Combine multiple holding policies.

    Mode 'any': exit when ANY sub-policy says exit (OR logic, more conservative)
    Mode 'all': exit when ALL sub-policies say exit (AND logic, holds longer)

    Example: HorizonAligned(10) + StopLoss(10bps) with mode='any'
    = exit after 10 events OR if loss exceeds 10 bps, whichever comes first.

    Args:
        policies: List of HoldingPolicy instances.
        mode: 'any' (OR) or 'all' (AND).
    """

    def __init__(self, policies: List[HoldingPolicy], mode: str = "any"):
        if not policies:
            raise ValueError("CompositePolicy requires at least one sub-policy")
        if mode not in ("any", "all"):
            raise ValueError(f"mode must be 'any' or 'all', got {mode}")
        self.policies = policies
        self.mode = mode

    @property
    def policy_name(self) -> str:
        sub_names = "+".join(p.policy_name for p in self.policies)
        return f"composite_{self.mode}({sub_names})"

    def should_exit(self, state: HoldingState) -> bool:
        results = [p.should_exit(state) for p in self.policies]
        if self.mode == "any":
            return any(results)
        return all(results)

    def to_dict(self) -> dict:
        return {
            "type": "composite",
            "mode": self.mode,
            "policies": [p.to_dict() for p in self.policies],
        }


def create_holding_policy(config: dict) -> HoldingPolicy:
    """
    Factory: create a HoldingPolicy from a config dict.

    Args:
        config: Dict with 'type' key and policy-specific parameters.
            Examples:
                {"type": "horizon_aligned", "hold_events": 10}
                {"type": "direction_reversal", "max_hold_events": 60}
                {"type": "stop_loss_take_profit", "stop_loss_bps": 10, "take_profit_bps": 20}
                {"type": "composite", "mode": "any", "policies": [...]}

    Returns:
        Configured HoldingPolicy instance.
    """
    policy_type = config.get("type", "horizon_aligned")

    if policy_type == "horizon_aligned":
        return HorizonAlignedPolicy(
            hold_events=config.get("hold_events", 10),
        )
    elif policy_type == "direction_reversal":
        return DirectionReversalPolicy(
            max_hold_events=config.get("max_hold_events", 60),
            require_gate=config.get("require_gate", False),
        )
    elif policy_type == "stop_loss_take_profit":
        return StopLossTakeProfitPolicy(
            stop_loss_bps=config.get("stop_loss_bps", 10.0),
            take_profit_bps=config.get("take_profit_bps", 20.0),
            max_hold_events=config.get("max_hold_events", 60),
        )
    elif policy_type == "composite":
        sub_configs = config.get("policies", [])
        sub_policies = [create_holding_policy(sc) for sc in sub_configs]
        return CompositePolicy(
            policies=sub_policies,
            mode=config.get("mode", "any"),
        )
    else:
        raise ValueError(f"Unknown holding policy type: {policy_type}")
