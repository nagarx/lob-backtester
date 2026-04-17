"""Tests for centralized label encoding (labels.py).

Validates the LabelMapping class that all strategies depend on for
interpreting model predictions. A bug here silently breaks every strategy.

Reference: BACKTESTER_AUDIT_PLAN.md § H1 (hardcoded label values)
"""

import pytest

from lobbacktest.labels import (
    LABEL_DOWN,
    LABEL_STABLE,
    LABEL_UP,
    SHIFTED_LABEL_DOWN,
    SHIFTED_LABEL_STABLE,
    SHIFTED_LABEL_UP,
    LabelMapping,
    SHIFTED_MAPPING,
    SIGNED_MAPPING,
)


class TestLabelMappingValues:
    """Verify label encoding constants match pipeline contract."""

    def test_shifted_mapping_values(self):
        """Shifted convention: {0=Down, 1=Stable, 2=Up} for PyTorch CE loss."""
        m = LabelMapping.from_shifted(shifted=True)
        assert m.down == 0, f"Shifted down should be 0, got {m.down}"
        assert m.stable == 1, f"Shifted stable should be 1, got {m.stable}"
        assert m.up == 2, f"Shifted up should be 2, got {m.up}"
        assert m.shifted is True

    def test_signed_mapping_values(self):
        """Signed convention: {-1=Down, 0=Stable, +1=Up} for pipeline."""
        m = LabelMapping.from_shifted(shifted=False)
        assert m.down == -1, f"Signed down should be -1, got {m.down}"
        assert m.stable == 0, f"Signed stable should be 0, got {m.stable}"
        assert m.up == 1, f"Signed up should be 1, got {m.up}"
        assert m.shifted is False

    def test_canonical_instances(self):
        """Module-level SHIFTED_MAPPING and SIGNED_MAPPING exist."""
        assert SHIFTED_MAPPING is not None
        assert SIGNED_MAPPING is not None
        assert SHIFTED_MAPPING.shifted is True
        assert SIGNED_MAPPING.shifted is False
        assert SHIFTED_MAPPING.up == 2
        assert SIGNED_MAPPING.up == 1


class TestLabelMappingPredicates:
    """Verify predicate methods used by all strategies."""

    def test_is_directional_shifted(self):
        """Shifted: 0=Down and 2=Up are directional, 1=Stable is not."""
        m = SHIFTED_MAPPING
        assert m.is_directional(0) is True, "Down (0) should be directional"
        assert m.is_directional(1) is False, "Stable (1) should NOT be directional"
        assert m.is_directional(2) is True, "Up (2) should be directional"

    def test_is_directional_signed(self):
        """Signed: -1=Down and +1=Up are directional, 0=Stable is not."""
        m = SIGNED_MAPPING
        assert m.is_directional(-1) is True, "Down (-1) should be directional"
        assert m.is_directional(0) is False, "Stable (0) should NOT be directional"
        assert m.is_directional(1) is True, "Up (+1) should be directional"

    def test_is_bullish_bearish_stable(self):
        """Each label maps to exactly one predicate."""
        m = SHIFTED_MAPPING
        # Up = bullish
        assert m.is_bullish(2) is True
        assert m.is_bearish(2) is False
        assert m.is_stable(2) is False
        # Down = bearish
        assert m.is_bearish(0) is True
        assert m.is_bullish(0) is False
        assert m.is_stable(0) is False
        # Stable = stable
        assert m.is_stable(1) is True
        assert m.is_bullish(1) is False
        assert m.is_bearish(1) is False

    def test_is_reversal_all_combinations(self):
        """Test all 9 (entry, current) combinations for reversal detection.

        Only (Down→Up) and (Up→Down) are reversals.
        Everything else (same-to-same, any-to-Stable, Stable-to-any) is NOT.
        """
        m = SHIFTED_MAPPING
        # TRUE reversals
        assert m.is_reversal(0, 2) is True, "Down→Up IS a reversal"
        assert m.is_reversal(2, 0) is True, "Up→Down IS a reversal"
        # NOT reversals: same direction
        assert m.is_reversal(0, 0) is False, "Down→Down is NOT a reversal"
        assert m.is_reversal(2, 2) is False, "Up→Up is NOT a reversal"
        assert m.is_reversal(1, 1) is False, "Stable→Stable is NOT a reversal"
        # NOT reversals: involving Stable
        assert m.is_reversal(0, 1) is False, "Down→Stable is NOT a reversal"
        assert m.is_reversal(2, 1) is False, "Up→Stable is NOT a reversal"
        assert m.is_reversal(1, 0) is False, "Stable→Down is NOT a reversal"
        assert m.is_reversal(1, 2) is False, "Stable→Up is NOT a reversal"

    def test_is_reversal_signed_mapping(self):
        """Reversal detection works with signed labels too."""
        m = SIGNED_MAPPING
        assert m.is_reversal(-1, 1) is True, "Signed: Down(-1)→Up(+1) IS a reversal"
        assert m.is_reversal(1, -1) is True, "Signed: Up(+1)→Down(-1) IS a reversal"
        assert m.is_reversal(-1, 0) is False, "Signed: Down→Stable is NOT"
        assert m.is_reversal(0, 1) is False, "Signed: Stable→Up is NOT"

    def test_frozen_immutability(self):
        """LabelMapping is frozen — cannot be accidentally mutated."""
        m = SHIFTED_MAPPING
        with pytest.raises(AttributeError):
            m.up = 99
