"""
Data loading and preprocessing for backtesting.

This module provides utilities for loading data exported by
feature-extractor-MBO-LOB:

- DataLoader: Load sequences, labels, normalization params
- PriceExtractor: Extract and denormalize mid-prices from features
- BacktestData: Container for backtest input data

Usage:
    >>> from lobbacktest.data import DataLoader
    >>> loader = DataLoader("path/to/exports", split="test")
    >>> data = loader.load()
"""

from lobbacktest.data.loader import DataLoader
from lobbacktest.data.prices import PriceExtractor

__all__ = [
    "DataLoader",
    "PriceExtractor",
]

