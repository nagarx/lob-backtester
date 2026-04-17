"""
Regression prediction quality metrics for backtesting.

Measures how well the model's continuous predictions correlate with
actual returns — the fundamental measure of regression signal quality.
"""

import numpy as np
from typing import Any, Dict, Mapping, Optional

from lobbacktest.metrics.base import Metric


class PredictionMSE(Metric):
    """Mean squared error between predicted and realized returns."""

    def __init__(self, predicted: np.ndarray, actual: np.ndarray):
        self._predicted = predicted
        self._actual = actual

    @property
    def name(self) -> str:
        return "PredictionMSE"

    def compute(self, returns: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Mapping[str, float]:
        mse = float(np.mean((self._predicted - self._actual) ** 2))
        return {"PredictionMSE": mse}


class PredictionCorrelation(Metric):
    """Pearson correlation between predicted and realized returns."""

    def __init__(self, predicted: np.ndarray, actual: np.ndarray):
        self._predicted = predicted
        self._actual = actual

    @property
    def name(self) -> str:
        return "PredictionCorrelation"

    def compute(self, returns: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Mapping[str, float]:
        if len(self._predicted) < 3:
            return {"PredictionCorrelation": 0.0}
        corr = np.corrcoef(self._predicted, self._actual)[0, 1]
        return {"PredictionCorrelation": float(corr) if np.isfinite(corr) else 0.0}


class PredictionIC(Metric):
    """Spearman rank correlation (Information Coefficient) between predicted and realized."""

    def __init__(self, predicted: np.ndarray, actual: np.ndarray):
        self._predicted = predicted
        self._actual = actual

    @property
    def name(self) -> str:
        return "PredictionIC"

    def compute(self, returns: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Mapping[str, float]:
        if len(self._predicted) < 3:
            return {"PredictionIC": 0.0}
        try:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(self._predicted, self._actual)
            return {"PredictionIC": float(corr) if np.isfinite(corr) else 0.0}
        except ImportError:
            return {"PredictionIC": 0.0}


class DirectionalAccuracy(Metric):
    """Fraction of trades where sign(predicted) matches sign(actual)."""

    def __init__(self, predicted: np.ndarray, actual: np.ndarray):
        self._predicted = predicted
        self._actual = actual

    @property
    def name(self) -> str:
        return "DirectionalAccuracy"

    def compute(self, returns: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Mapping[str, float]:
        mask = (self._predicted != 0) & (self._actual != 0)
        if mask.sum() == 0:
            return {"DirectionalAccuracy": 0.5}
        acc = float((np.sign(self._predicted[mask]) == np.sign(self._actual[mask])).mean())
        return {"DirectionalAccuracy": acc}
