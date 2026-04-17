"""
ML prediction quality metrics.

Metrics:
- DirectionalAccuracy: Accuracy on Up/Down predictions (ignoring Stable)
- SignalRate: Fraction of non-Stable predictions
- UpPrecision: Precision for Up predictions
- DownPrecision: Precision for Down predictions

These metrics evaluate the model's prediction quality independent of
trading execution.
"""

from typing import Any, Dict, Mapping

import numpy as np

from lobbacktest.metrics.base import Metric

# Label constants (consistent with lob-model-trainer)
LABEL_DOWN = -1
LABEL_STABLE = 0
LABEL_UP = 1

# Shifted labels (PyTorch CrossEntropyLoss format)
SHIFTED_LABEL_DOWN = 0
SHIFTED_LABEL_STABLE = 1
SHIFTED_LABEL_UP = 2


class DirectionalAccuracy(Metric):
    """
    Accuracy on directional predictions (Up/Down only).

    Ignores Stable predictions and labels to focus on
    the model's ability to predict price direction.

    Formula:
        DirectionalAcc = correct_directional / total_directional

    Where:
        directional = (label != Stable) AND (prediction != Stable)

    Reference:
        Common in LOB prediction literature (DeepLOB, TLOB)

    Notes:
        - More relevant for trading than overall accuracy
        - Requires "predictions" and "labels" in context
    """

    def __init__(self, *, name: str = None, shifted: bool = False):
        """
        Initialize DirectionalAccuracy metric.

        Args:
            name: Optional custom name (default: "DirectionalAccuracy")
            shifted: If True, use shifted labels (0/1/2 instead of -1/0/1)
        """
        self._name = name or "DirectionalAccuracy"
        self.shifted = shifted

        # Select appropriate label constants
        if shifted:
            self.label_down = SHIFTED_LABEL_DOWN
            self.label_stable = SHIFTED_LABEL_STABLE
            self.label_up = SHIFTED_LABEL_UP
        else:
            self.label_down = LABEL_DOWN
            self.label_stable = LABEL_STABLE
            self.label_up = LABEL_UP

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute directional accuracy.

        Args:
            returns: Not used directly
            context: Must contain:
                - "predictions": Model predictions (shape: N)
                - "labels": True labels (shape: N)

        Returns:
            {"DirectionalAccuracy": accuracy} where accuracy in [0, 1]

        Edge cases:
            - No directional predictions: 0.0
            - All Stable: 0.0
        """
        predictions = context.get("predictions")
        labels = context.get("labels")

        if predictions is None or labels is None:
            return {self.name: 0.0}

        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        if len(predictions) != len(labels):
            return {self.name: 0.0}

        # Filter to directional (non-Stable) samples
        # Both prediction and label must be directional
        directional_mask = (labels != self.label_stable) & (
            predictions != self.label_stable
        )

        if np.sum(directional_mask) == 0:
            return {self.name: 0.0}

        correct = np.sum(
            predictions[directional_mask] == labels[directional_mask]
        )
        total = np.sum(directional_mask)

        accuracy = correct / total

        return {self.name: float(accuracy)}


class SignalRate(Metric):
    """
    Fraction of non-Stable predictions.

    Measures how often the model makes a directional call.

    Formula:
        SignalRate = (num_Up + num_Down) / total

    Reference:
        Useful for understanding model behavior

    Notes:
        - High signal rate = aggressive model
        - Low signal rate = conservative model
    """

    def __init__(self, *, name: str = None, shifted: bool = False):
        """
        Initialize SignalRate metric.

        Args:
            name: Optional custom name (default: "SignalRate")
            shifted: If True, use shifted labels
        """
        self._name = name or "SignalRate"
        self.shifted = shifted
        self.label_stable = SHIFTED_LABEL_STABLE if shifted else LABEL_STABLE

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute signal rate.

        Args:
            returns: Not used directly
            context: Must contain "predictions"

        Returns:
            {"SignalRate": rate} where rate in [0, 1]
        """
        predictions = context.get("predictions")

        if predictions is None:
            return {self.name: 0.0}

        predictions = np.asarray(predictions)

        if len(predictions) == 0:
            return {self.name: 0.0}

        n_signals = np.sum(predictions != self.label_stable)
        rate = n_signals / len(predictions)

        return {self.name: float(rate)}


class UpPrecision(Metric):
    """
    Precision for Up predictions.

    When the model predicts Up, how often is it correct?

    Formula:
        UpPrecision = TP_up / (TP_up + FP_up)

    Where:
        TP_up = predicted Up and actual Up
        FP_up = predicted Up but actual was not Up

    Reference:
        Standard precision metric focused on Up class
    """

    def __init__(self, *, name: str = None, shifted: bool = False):
        """
        Initialize UpPrecision metric.

        Args:
            name: Optional custom name (default: "UpPrecision")
            shifted: If True, use shifted labels
        """
        self._name = name or "UpPrecision"
        self.shifted = shifted
        self.label_up = SHIFTED_LABEL_UP if shifted else LABEL_UP

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute Up precision.

        Args:
            returns: Not used directly
            context: Must contain "predictions" and "labels"

        Returns:
            {"UpPrecision": precision} where precision in [0, 1]

        Edge cases:
            - No Up predictions: 0.0
        """
        predictions = context.get("predictions")
        labels = context.get("labels")

        if predictions is None or labels is None:
            return {self.name: 0.0}

        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        # Predicted Up
        up_pred_mask = predictions == self.label_up

        if np.sum(up_pred_mask) == 0:
            return {self.name: 0.0}

        # True positives: predicted Up and actual Up
        tp = np.sum(labels[up_pred_mask] == self.label_up)
        total_pred_up = np.sum(up_pred_mask)

        precision = tp / total_pred_up

        return {self.name: float(precision)}


class DownPrecision(Metric):
    """
    Precision for Down predictions.

    When the model predicts Down, how often is it correct?

    Formula:
        DownPrecision = TP_down / (TP_down + FP_down)

    Reference:
        Standard precision metric focused on Down class
    """

    def __init__(self, *, name: str = None, shifted: bool = False):
        """
        Initialize DownPrecision metric.

        Args:
            name: Optional custom name (default: "DownPrecision")
            shifted: If True, use shifted labels
        """
        self._name = name or "DownPrecision"
        self.shifted = shifted
        self.label_down = SHIFTED_LABEL_DOWN if shifted else LABEL_DOWN

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute Down precision.

        Args:
            returns: Not used directly
            context: Must contain "predictions" and "labels"

        Returns:
            {"DownPrecision": precision} where precision in [0, 1]

        Edge cases:
            - No Down predictions: 0.0
        """
        predictions = context.get("predictions")
        labels = context.get("labels")

        if predictions is None or labels is None:
            return {self.name: 0.0}

        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        # Predicted Down
        down_pred_mask = predictions == self.label_down

        if np.sum(down_pred_mask) == 0:
            return {self.name: 0.0}

        # True positives: predicted Down and actual Down
        tp = np.sum(labels[down_pred_mask] == self.label_down)
        total_pred_down = np.sum(down_pred_mask)

        precision = tp / total_pred_down

        return {self.name: float(precision)}


class ConfusionMetrics(Metric):
    """
    Compute full confusion matrix metrics.

    Returns multiple metrics:
        - Overall accuracy
        - Per-class precision
        - Per-class recall
        - Per-class F1

    Reference:
        Standard multi-class classification metrics
    """

    def __init__(self, *, name: str = None, shifted: bool = False):
        """
        Initialize ConfusionMetrics.

        Args:
            name: Optional custom name (default: "ConfusionMetrics")
            shifted: If True, use shifted labels
        """
        self._name = name or "ConfusionMetrics"
        self.shifted = shifted

        if shifted:
            self.labels = [SHIFTED_LABEL_DOWN, SHIFTED_LABEL_STABLE, SHIFTED_LABEL_UP]
            self.names = ["Down", "Stable", "Up"]
        else:
            self.labels = [LABEL_DOWN, LABEL_STABLE, LABEL_UP]
            self.names = ["Down", "Stable", "Up"]

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute confusion matrix metrics.

        Returns:
            {
                "Accuracy": overall_accuracy,
                "Precision_Down": ...,
                "Precision_Stable": ...,
                "Precision_Up": ...,
                "Recall_Down": ...,
                ...
            }
        """
        predictions = context.get("predictions")
        labels = context.get("labels")

        if predictions is None or labels is None:
            return {self.name: 0.0}

        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        result = {}

        # Overall accuracy
        result["Accuracy"] = float(np.mean(predictions == labels))

        # Per-class metrics
        for label_val, label_name in zip(self.labels, self.names):
            # Precision: TP / (TP + FP)
            pred_mask = predictions == label_val
            if np.sum(pred_mask) > 0:
                precision = np.sum(labels[pred_mask] == label_val) / np.sum(pred_mask)
            else:
                precision = 0.0
            result[f"Precision_{label_name}"] = float(precision)

            # Recall: TP / (TP + FN)
            true_mask = labels == label_val
            if np.sum(true_mask) > 0:
                recall = np.sum(predictions[true_mask] == label_val) / np.sum(true_mask)
            else:
                recall = 0.0
            result[f"Recall_{label_name}"] = float(recall)

            # F1: 2 * (P * R) / (P + R)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            result[f"F1_{label_name}"] = float(f1)

        return result

