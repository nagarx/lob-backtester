"""
Base classes for metrics.

This module defines the Metric ABC pattern inspired by hftbacktest.
Metrics are composable, configurable, and extensible.

Design Principles:
1. Each metric has a single responsibility
2. Metrics receive context dict with previously computed values
3. Metrics return dict of {name: value} pairs
4. Handle edge cases explicitly (NaN, empty arrays, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np


@dataclass
class MetricResult:
    """
    Result of a metric computation.

    Attributes:
        name: Metric name
        value: Computed value
        metadata: Optional additional information
    """

    name: str
    value: float
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class Metric(ABC):
    """
    Abstract base class for computing performance metrics.

    Metrics follow the pattern from hftbacktest:
    - Receive returns array and context dict
    - Return dict of {name: value} pairs
    - Can reference previously computed metrics via context

    Example:
        >>> class MyMetric(Metric):
        ...     @property
        ...     def name(self) -> str:
        ...         return "MyMetric"
        ...
        ...     def compute(self, returns, context):
        ...         return {self.name: np.mean(returns) * 100}
        >>>
        >>> metric = MyMetric()
        >>> result = metric.compute(np.array([0.01, 0.02]), {})
        >>> print(result["MyMetric"])
        1.5
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this metric.

        Returns:
            Metric name (used as key in results dict)
        """
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """
        Compute the metric from returns data.

        Args:
            returns: Array of per-period returns (shape: N)
            context: Dict containing:
                - Previously computed metrics
                - Configuration parameters (e.g., annualization_factor)
                - Additional data (equity_curve, trades, etc.)

        Returns:
            Dict mapping metric name to computed value

        Raises:
            ValueError: If required context keys are missing

        Note:
            - Handle empty arrays gracefully (return 0 or NaN as appropriate)
            - Check for NaN/Inf values and handle appropriately
            - Use context.get() with defaults for optional dependencies
        """
        raise NotImplementedError

    def validate_returns(self, returns: np.ndarray) -> bool:
        """
        Validate returns array for computation.

        Args:
            returns: Array to validate

        Returns:
            True if returns are valid for computation

        Checks:
            - Array is not empty
            - Array has finite values
            - Array is 1-dimensional
        """
        if returns is None or len(returns) == 0:
            return False
        if returns.ndim != 1:
            return False
        if not np.all(np.isfinite(returns)):
            return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class CompositeMetric(Metric):
    """
    A metric composed of multiple sub-metrics.

    Useful for computing related metrics together and sharing
    intermediate computations.

    Example:
        >>> class RiskMetrics(CompositeMetric):
        ...     def __init__(self):
        ...         self.sharpe = SharpeRatio()
        ...         self.sortino = SortinoRatio()
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return "RiskMetrics"
        ...
        ...     def compute(self, returns, context):
        ...         result = {}
        ...         result.update(self.sharpe.compute(returns, context))
        ...         result.update(self.sortino.compute(returns, context))
        ...         return result
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        raise NotImplementedError

