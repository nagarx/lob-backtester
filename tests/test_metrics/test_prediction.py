"""
Tests for prediction quality metrics.

Tests verify:
- DirectionalAccuracy: accuracy on non-Stable predictions
- SignalRate: fraction of non-Stable predictions
- UpPrecision / DownPrecision: per-class precision
- Shifted vs unshifted label handling

Per RULE.md:
- Formula tests: Hand-calculated examples with known outcomes
- Edge tests: All Stable predictions, all correct, all wrong
- Boundary tests: Shifted (0/1/2) vs unshifted (-1/0/1) label encoding
"""

import numpy as np
import pytest

from lobbacktest.metrics.prediction import (
    DirectionalAccuracy,
    DownPrecision,
    SignalRate,
    UpPrecision,
)


class TestDirectionalAccuracy:
    """Tests for DirectionalAccuracy metric."""

    def test_formula_unshifted(self):
        """
        DirectionalAcc = correct_directional / total_directional

        Labels: -1=Down, 0=Stable, 1=Up (unshifted)
        Only count where BOTH label and prediction are directional.

        Hand-calculated:
            preds:  [1,  -1,  0,  1, -1]
            labels: [1,  -1,  0, -1,  1]
            directional pairs: (1,1), (-1,-1), (1,-1), (-1,1) = 4 pairs
            correct: (1,1), (-1,-1) = 2
            DA = 2/4 = 0.50
        """
        preds = np.array([1, -1, 0, 1, -1])
        labels = np.array([1, -1, 0, -1, 1])
        context = {"predictions": preds, "labels": labels}

        metric = DirectionalAccuracy(shifted=False)
        result = metric.compute(np.array([0.01, -0.01, 0.0, 0.01, -0.01]), context)
        assert abs(result["DirectionalAccuracy"] - 0.50) < 1e-10

    def test_formula_shifted(self):
        """
        Shifted labels: 0=Down, 1=Stable, 2=Up
        Same logic but with shifted values.

        Hand-calculated:
            preds:  [2, 0, 1, 2, 0]
            labels: [2, 0, 1, 0, 2]
            directional pairs: (2,2), (0,0), (2,0), (0,2) = 4 pairs
            correct: (2,2), (0,0) = 2
            DA = 2/4 = 0.50
        """
        preds = np.array([2, 0, 1, 2, 0])
        labels = np.array([2, 0, 1, 0, 2])
        context = {"predictions": preds, "labels": labels}

        metric = DirectionalAccuracy(shifted=True)
        result = metric.compute(np.array([0.01] * 5), context)
        assert abs(result["DirectionalAccuracy"] - 0.50) < 1e-10

    def test_perfect_directional_accuracy(self):
        """All directional predictions correct → DA = 1.0."""
        preds = np.array([1, -1, 1, -1])
        labels = np.array([1, -1, 1, -1])
        context = {"predictions": preds, "labels": labels}

        metric = DirectionalAccuracy(shifted=False)
        result = metric.compute(np.array([0.01] * 4), context)
        assert result["DirectionalAccuracy"] == 1.0

    def test_all_stable_returns_default(self):
        """All Stable predictions → no directional pairs → return default."""
        preds = np.array([0, 0, 0, 0])
        labels = np.array([1, -1, 0, 1])
        context = {"predictions": preds, "labels": labels}

        metric = DirectionalAccuracy(shifted=False)
        result = metric.compute(np.array([0.01] * 4), context)
        # When no directional predictions, typically returns 0.5 or similar
        assert 0.0 <= result["DirectionalAccuracy"] <= 1.0

    def test_no_labels_returns_default(self):
        """Missing labels in context → graceful fallback."""
        context = {"predictions": np.array([1, -1, 1])}
        metric = DirectionalAccuracy(shifted=False)
        result = metric.compute(np.array([0.01] * 3), context)
        assert "DirectionalAccuracy" in result

    def test_keyword_only_constructor(self):
        """Positional args must raise TypeError (Fix 1 enforcement)."""
        with pytest.raises(TypeError):
            DirectionalAccuracy("custom_name", True)


class TestSignalRate:
    """Tests for SignalRate metric."""

    def test_formula_unshifted(self):
        """
        SignalRate = count(pred != Stable) / total

        Hand-calculated:
            preds: [1, -1, 0, 1, 0]
            non-stable: 3 (1, -1, 1)
            total: 5
            rate = 3/5 = 0.60
        """
        preds = np.array([1, -1, 0, 1, 0])
        context = {"predictions": preds}

        metric = SignalRate(shifted=False)
        result = metric.compute(np.array([0.01] * 5), context)
        assert abs(result["SignalRate"] - 0.60) < 1e-10

    def test_formula_shifted(self):
        """Shifted labels: Stable=1. preds=[2,0,1,2,1] → 3/5 = 0.60."""
        preds = np.array([2, 0, 1, 2, 1])
        context = {"predictions": preds}

        metric = SignalRate(shifted=True)
        result = metric.compute(np.array([0.01] * 5), context)
        assert abs(result["SignalRate"] - 0.60) < 1e-10

    def test_all_stable(self):
        """All Stable → rate = 0.0."""
        preds = np.array([0, 0, 0, 0])
        context = {"predictions": preds}
        metric = SignalRate(shifted=False)
        result = metric.compute(np.array([0.01] * 4), context)
        assert result["SignalRate"] == 0.0

    def test_no_stable(self):
        """No Stable predictions → rate = 1.0."""
        preds = np.array([1, -1, 1, -1])
        context = {"predictions": preds}
        metric = SignalRate(shifted=False)
        result = metric.compute(np.array([0.01] * 4), context)
        assert result["SignalRate"] == 1.0


class TestUpPrecision:
    """Tests for UpPrecision metric."""

    def test_formula(self):
        """
        UpPrecision = TP_up / (TP_up + FP_up)

        Hand-calculated (unshifted, Up=1):
            preds:  [1,  1, -1, 0,  1]
            labels: [1, -1,  1, 0,  1]
            Predicted Up: indices 0, 1, 4
            TP_up (pred=Up AND label=Up): indices 0, 4 → count=2
            FP_up (pred=Up AND label≠Up): index 1 → count=1
            Precision = 2 / (2+1) = 0.667
        """
        preds = np.array([1, 1, -1, 0, 1])
        labels = np.array([1, -1, 1, 0, 1])
        context = {"predictions": preds, "labels": labels}

        metric = UpPrecision(shifted=False)
        result = metric.compute(np.array([0.01] * 5), context)
        assert abs(result["UpPrecision"] - 2 / 3) < 1e-10


class TestDownPrecision:
    """Tests for DownPrecision metric."""

    def test_formula(self):
        """
        DownPrecision = TP_down / (TP_down + FP_down)

        Hand-calculated (unshifted, Down=-1):
            preds:  [-1, -1,  1,  0, -1]
            labels: [-1,  1, -1,  0, -1]
            Predicted Down: indices 0, 1, 4
            TP_down (pred=Down AND label=Down): indices 0, 4 → count=2
            FP_down (pred=Down AND label≠Down): index 1 → count=1
            Precision = 2 / (2+1) = 0.667
        """
        preds = np.array([-1, -1, 1, 0, -1])
        labels = np.array([-1, 1, -1, 0, -1])
        context = {"predictions": preds, "labels": labels}

        metric = DownPrecision(shifted=False)
        result = metric.compute(np.array([0.01] * 5), context)
        assert abs(result["DownPrecision"] - 2 / 3) < 1e-10
