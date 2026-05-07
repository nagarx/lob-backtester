"""
Regression prediction quality metrics for backtesting.

Measures how well the model's continuous predictions correlate with
actual returns — the fundamental measure of regression signal quality.

#PY-63 (2026-05-07): split-semantics on silent-NaN per hft-rules §8.
  - Input-NaN/Inf raises ValueError (caller invariant violation;
    upstream producer was supposed to fail-loud per #38/#39/#40)
  - Constant-array NaN from corr computation returns 0.0 with
    RuntimeWarning (legitimate edge case — variance undefined)
  - scipy ImportError now raises (was silently 0.0; scipy is in
    pyproject.toml [dev], absence indicates env-bug per hft-rules §5)
"""

import numpy as np
import warnings
from typing import Any, Dict, Mapping, Optional

from lobbacktest.metrics.base import Metric


def _assert_finite_pair(
    predicted: np.ndarray, actual: np.ndarray, metric_name: str
) -> None:
    """Fail-loud on NaN/Inf inputs per hft-rules §8.

    #PY-63 (2026-05-07): predicted/actual NaN at a metric boundary is a
    caller-invariant violation — upstream producers (#38/#39/#40) are
    supposed to fail-loud, so reaching a metric with NaN inputs means a
    path bypassed the producer assertions. Delegates to the SSoT helper
    `hft_contracts.validation.assert_finite_array` per reuse-first §0.
    """
    from hft_contracts.validation import assert_finite_array
    assert_finite_array(
        predicted,
        name=f"{metric_name}.predicted",
        extra_diagnostic=(
            "Fix upstream producer (signal exporter, calibration, "
            "or trainer save path)."
        ),
    )
    assert_finite_array(
        actual,
        name=f"{metric_name}.actual",
        extra_diagnostic="Investigate label generation.",
    )


class PredictionMSE(Metric):
    """Mean squared error between predicted and realized returns."""

    def __init__(self, predicted: np.ndarray, actual: np.ndarray):
        self._predicted = predicted
        self._actual = actual

    @property
    def name(self) -> str:
        return "PredictionMSE"

    def compute(self, returns: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Mapping[str, float]:
        _assert_finite_pair(self._predicted, self._actual, "PredictionMSE")
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
        _assert_finite_pair(self._predicted, self._actual, "PredictionCorrelation")
        corr = np.corrcoef(self._predicted, self._actual)[0, 1]
        if not np.isfinite(corr):
            # Constant-array edge: variance=0 → corrcoef undefined.
            # Legitimate per §8 "expected anomalies" — return 0.0 with
            # tracked diagnostic (RuntimeWarning).
            warnings.warn(
                "PredictionCorrelation: correlation undefined "
                "(constant input array — zero variance); returning 0.0",
                RuntimeWarning,
                stacklevel=2,
            )
            return {"PredictionCorrelation": 0.0}
        return {"PredictionCorrelation": float(corr)}


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
        _assert_finite_pair(self._predicted, self._actual, "PredictionIC")
        try:
            from scipy.stats import spearmanr
        except ImportError as exc:
            # scipy is in pyproject.toml [dev]; missing = env-bug per §5.
            # Was silently returning 0.0 — masked dependency-resolution
            # failure as "model has no skill".
            raise ImportError(
                "PredictionIC requires scipy (declared in pyproject.toml). "
                "Install with `pip install scipy>=1.10` or "
                "`pip install -e '.[dev]'`."
            ) from exc
        corr, _ = spearmanr(self._predicted, self._actual)
        if not np.isfinite(corr):
            # Constant-array edge case (legitimate; same rationale as
            # PredictionCorrelation above).
            warnings.warn(
                "PredictionIC: Spearman correlation undefined "
                "(constant input array — tied ranks); returning 0.0",
                RuntimeWarning,
                stacklevel=2,
            )
            return {"PredictionIC": 0.0}
        return {"PredictionIC": float(corr)}


class DirectionalAccuracy(Metric):
    """Fraction of trades where sign(predicted) matches sign(actual)."""

    def __init__(self, predicted: np.ndarray, actual: np.ndarray):
        self._predicted = predicted
        self._actual = actual

    @property
    def name(self) -> str:
        return "DirectionalAccuracy"

    def compute(self, returns: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Mapping[str, float]:
        _assert_finite_pair(self._predicted, self._actual, "DirectionalAccuracy")
        mask = (self._predicted != 0) & (self._actual != 0)
        if mask.sum() == 0:
            # Legitimate edge: all predictions/actuals are exactly zero
            # → no directional information. Return chance baseline.
            return {"DirectionalAccuracy": 0.5}
        acc = float((np.sign(self._predicted[mask]) == np.sign(self._actual[mask])).mean())
        return {"DirectionalAccuracy": acc}
