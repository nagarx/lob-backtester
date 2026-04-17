"""Centralized label encoding for all backtester strategies.

This module is the SINGLE SOURCE OF TRUTH for label value mappings
in the backtester. All strategies and metrics that reference specific
label values (Up, Down, Stable) MUST import from here.

Supports two encoding conventions:
    - **Signed** {-1=Down, 0=Stable, +1=Up}: Original pipeline convention
    - **Shifted** {0=Down, 1=Stable, 2=Up}: PyTorch CrossEntropyLoss convention

The shifted encoding is the default for backtesting because model
predictions use CrossEntropyLoss (shifted +1 from signed convention).

Reference:
    hft-contracts/src/hft_contracts/labels.py — canonical label contracts
    CLAUDE.md § Label Encoding
"""

from dataclasses import dataclass


# --- Module-level constants (same as direction.py, prediction.py) ---

# Signed convention (original pipeline: {-1, 0, +1})
LABEL_DOWN: int = -1
LABEL_STABLE: int = 0
LABEL_UP: int = 1

# Shifted convention (PyTorch: {0, 1, 2})
SHIFTED_LABEL_DOWN: int = 0
SHIFTED_LABEL_STABLE: int = 1
SHIFTED_LABEL_UP: int = 2


@dataclass(frozen=True)
class LabelMapping:
    """Immutable label encoding for strategy parametrization.

    Every strategy that checks prediction values against Up/Down/Stable
    should use this mapping instead of hardcoded integers.

    Usage:
        mapping = LabelMapping.shifted()  # For model predictions (0/1/2)
        if predictions[i] == mapping.up:
            signal = Signal.BUY
        elif predictions[i] == mapping.down:
            signal = Signal.SELL

    Attributes:
        down: Integer value representing bearish/down prediction.
        stable: Integer value representing neutral/stable prediction.
        up: Integer value representing bullish/up prediction.
        shifted: Whether this uses the shifted (PyTorch) convention.
    """

    down: int
    stable: int
    up: int
    shifted: bool

    @classmethod
    def from_shifted(cls, shifted: bool = True) -> "LabelMapping":
        """Create mapping from shifted flag.

        Args:
            shifted: If True, use {0=Down, 1=Stable, 2=Up}.
                     If False, use {-1=Down, 0=Stable, +1=Up}.
        """
        if shifted:
            return cls(
                down=SHIFTED_LABEL_DOWN,
                stable=SHIFTED_LABEL_STABLE,
                up=SHIFTED_LABEL_UP,
                shifted=True,
            )
        return cls(
            down=LABEL_DOWN,
            stable=LABEL_STABLE,
            up=LABEL_UP,
            shifted=False,
        )

    def is_directional(self, label: int) -> bool:
        """True if label represents a directional prediction (Up or Down)."""
        return label == self.up or label == self.down

    def is_bullish(self, label: int) -> bool:
        """True if label represents an Up/bullish prediction."""
        return label == self.up

    def is_bearish(self, label: int) -> bool:
        """True if label represents a Down/bearish prediction."""
        return label == self.down

    def is_stable(self, label: int) -> bool:
        """True if label represents a Stable/neutral prediction."""
        return label == self.stable

    def is_reversal(self, entry_label: int, current_label: int) -> bool:
        """True if current prediction reverses entry direction.

        A reversal occurs when:
            - Entry was Up, current is Down
            - Entry was Down, current is Up

        Used by DirectionReversalPolicy in holding.py.
        """
        return (
            (entry_label == self.down and current_label == self.up)
            or (entry_label == self.up and current_label == self.down)
        )

    def directional_values(self) -> tuple:
        """Return (down, up) tuple for in/not-in checks."""
        return (self.down, self.up)


# --- Canonical instances ---

#: Default mapping for backtesting (model outputs use shifted labels)
SHIFTED_MAPPING = LabelMapping.from_shifted(shifted=True)

#: Signed mapping for raw pipeline labels
SIGNED_MAPPING = LabelMapping.from_shifted(shifted=False)
