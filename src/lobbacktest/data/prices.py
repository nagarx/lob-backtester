"""
Price extraction and denormalization.

Utilities for extracting mid-prices from feature sequences and
denormalizing them using exported normalization parameters.

Data Contract (from feature-extractor-MBO-LOB):
    - Sequences: shape (N, 100, 98) or (N, 100, 40) float32
    - Feature layout (GROUPED):
        - [0:10]: Ask prices (10 levels)
        - [10:20]: Ask sizes (10 levels)
        - [20:30]: Bid prices (10 levels)
        - [30:40]: Bid sizes (10 levels)
        - [40]: Mid-price (if derived features enabled)
    - Normalization: Market-structure Z-score
        - price_means: [10] values (per level, shared ask+bid)
        - price_stds: [10] values
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import json

import numpy as np


# Feature indices (consistent with feature-extractor-MBO-LOB)
ASK_PRICES_START = 0
ASK_PRICES_END = 10
BID_PRICES_START = 20
BID_PRICES_END = 30
MID_PRICE_INDEX = 40  # Derived feature index


@dataclass
class NormalizationParams:
    """
    Normalization parameters from feature export.

    Used to denormalize prices for backtesting.

    Attributes:
        strategy: Normalization strategy name
        price_means: Mean values for prices per level (shape: 10)
        price_stds: Std values for prices per level (shape: 10)
        size_means: Mean values for sizes (shape: 20)
        size_stds: Std values for sizes (shape: 20)
        sample_count: Number of samples used for stats
        levels: Number of LOB levels
    """

    strategy: str
    price_means: np.ndarray  # Shape: (10,)
    price_stds: np.ndarray  # Shape: (10,)
    size_means: np.ndarray  # Shape: (20,)
    size_stds: np.ndarray  # Shape: (20,)
    sample_count: int
    levels: int

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "NormalizationParams":
        """Load normalization params from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            strategy=data.get("strategy", "market_structure_zscore"),
            price_means=np.array(data.get("price_means", [])),
            price_stds=np.array(data.get("price_stds", [])),
            size_means=np.array(data.get("size_means", [])),
            size_stds=np.array(data.get("size_stds", [])),
            sample_count=data.get("sample_count", 0),
            levels=data.get("levels", 10),
        )

    def denormalize_prices(
        self,
        normalized_prices: np.ndarray,
        level: int = 0,
    ) -> np.ndarray:
        """
        Denormalize prices for a specific level.

        Args:
            normalized_prices: Normalized price values
            level: LOB level (0-9, default: 0 for best price)

        Returns:
            Denormalized prices in original units
        """
        if level >= len(self.price_means):
            raise ValueError(f"Level {level} exceeds available levels {len(self.price_means)}")

        mean = self.price_means[level]
        std = self.price_stds[level]

        return normalized_prices * std + mean


class PriceExtractor:
    """
    Extract and denormalize prices from feature sequences.

    This class handles the conversion from normalized feature
    sequences to real mid-prices for backtesting.

    Example:
        >>> extractor = PriceExtractor(norm_params)
        >>> prices = extractor.extract_mid_prices(sequences)
    """

    def __init__(
        self,
        norm_params: Optional[NormalizationParams] = None,
        use_derived_mid: bool = True,
    ):
        """
        Initialize PriceExtractor.

        Args:
            norm_params: Normalization parameters (required for denormalization)
            use_derived_mid: If True, use pre-computed mid-price from derived features
                            If False, compute mid from ask/bid prices
        """
        self.norm_params = norm_params
        self.use_derived_mid = use_derived_mid

    def extract_mid_prices(
        self,
        sequences: np.ndarray,
        denormalize: bool = True,
    ) -> np.ndarray:
        """
        Extract mid-prices from feature sequences.

        Takes the LAST timestep of each sequence (most recent data point).

        Args:
            sequences: Feature sequences (shape: N, T, F)
            denormalize: If True, convert back to original price units

        Returns:
            Mid-prices (shape: N)

        Notes:
            - Uses last timestep of each sequence (index -1)
            - If use_derived_mid=True and F >= 41, uses derived mid-price
            - Otherwise, computes mid from (ask_L0 + bid_L0) / 2
        """
        if sequences.ndim != 3:
            raise ValueError(f"Expected 3D sequences, got shape {sequences.shape}")

        n_sequences, seq_len, n_features = sequences.shape

        # Get last timestep
        last_features = sequences[:, -1, :]  # Shape: (N, F)

        # Extract mid-prices
        if self.use_derived_mid and n_features > MID_PRICE_INDEX:
            # Use pre-computed mid-price from derived features
            normalized_mid = last_features[:, MID_PRICE_INDEX]

            if denormalize and self.norm_params is not None:
                # Mid-price uses level 0 normalization
                mid_prices = self.norm_params.denormalize_prices(normalized_mid, level=0)
            else:
                mid_prices = normalized_mid
        else:
            # Compute mid from ask/bid prices
            ask_l0 = last_features[:, ASK_PRICES_START]  # Best ask
            bid_l0 = last_features[:, BID_PRICES_START]  # Best bid

            if denormalize and self.norm_params is not None:
                # Denormalize both
                ask_denorm = self.norm_params.denormalize_prices(ask_l0, level=0)
                bid_denorm = self.norm_params.denormalize_prices(bid_l0, level=0)
                mid_prices = (ask_denorm + bid_denorm) / 2
            else:
                mid_prices = (ask_l0 + bid_l0) / 2

        return mid_prices

    def extract_price_series(
        self,
        sequences: np.ndarray,
        denormalize: bool = True,
    ) -> np.ndarray:
        """
        Extract price series for each sequence timestep.

        Returns prices for ALL timesteps, not just the last one.
        Useful for intra-sequence analysis.

        Args:
            sequences: Feature sequences (shape: N, T, F)
            denormalize: If True, convert to original units

        Returns:
            Price series (shape: N, T)
        """
        if sequences.ndim != 3:
            raise ValueError(f"Expected 3D sequences, got shape {sequences.shape}")

        n_sequences, seq_len, n_features = sequences.shape

        if self.use_derived_mid and n_features > MID_PRICE_INDEX:
            # Use derived mid-price
            normalized_mids = sequences[:, :, MID_PRICE_INDEX]  # Shape: (N, T)

            if denormalize and self.norm_params is not None:
                prices = self.norm_params.denormalize_prices(normalized_mids, level=0)
            else:
                prices = normalized_mids
        else:
            # Compute from ask/bid
            ask_l0 = sequences[:, :, ASK_PRICES_START]
            bid_l0 = sequences[:, :, BID_PRICES_START]

            if denormalize and self.norm_params is not None:
                ask_denorm = self.norm_params.denormalize_prices(ask_l0, level=0)
                bid_denorm = self.norm_params.denormalize_prices(bid_l0, level=0)
                prices = (ask_denorm + bid_denorm) / 2
            else:
                prices = (ask_l0 + bid_l0) / 2

        return prices

    def extract_spread(
        self,
        sequences: np.ndarray,
        denormalize: bool = True,
    ) -> np.ndarray:
        """
        Extract bid-ask spread from sequences.

        Args:
            sequences: Feature sequences (shape: N, T, F)
            denormalize: If True, convert to original units

        Returns:
            Spread at last timestep (shape: N)
        """
        if sequences.ndim != 3:
            raise ValueError(f"Expected 3D sequences, got shape {sequences.shape}")

        last_features = sequences[:, -1, :]

        ask_l0 = last_features[:, ASK_PRICES_START]
        bid_l0 = last_features[:, BID_PRICES_START]

        if denormalize and self.norm_params is not None:
            ask_denorm = self.norm_params.denormalize_prices(ask_l0, level=0)
            bid_denorm = self.norm_params.denormalize_prices(bid_l0, level=0)
            spread = ask_denorm - bid_denorm
        else:
            spread = ask_l0 - bid_l0

        return spread

