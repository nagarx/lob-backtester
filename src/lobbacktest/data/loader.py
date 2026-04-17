"""
Data loading utilities for backtesting.

Loads data exported by feature-extractor-MBO-LOB and
converts it to BacktestData format.

Expected directory structure:
    exports/
    ├── dataset_manifest.json
    ├── train/
    │   ├── {date}_sequences.npy
    │   ├── {date}_labels.npy
    │   ├── {date}_metadata.json
    │   └── {date}_normalization.json
    ├── val/
    └── test/

Normalization boundary contract:
    The Rust exporter writes raw (un-normalized) features by default.
    The {date}_normalization.json contains identity placeholders (means=0, stds=1).
    The trainer normalizes a COPY of the data for model input.
    The backtester reads the ORIGINAL raw .npy and extracts prices directly.
    Therefore the backtester does NOT need the trainer's normalization stats.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union
import json
import logging

import numpy as np

from lobbacktest.data.prices import NormalizationParams, PriceExtractor
from lobbacktest.engine.vectorized import BacktestData

from hft_contracts.validation import (
    ContractError,
    validate_export_contract,
)

logger = logging.getLogger(__name__)


@dataclass
class DayData:
    """
    Data for a single trading day.

    Attributes:
        date: Date string (YYYYMMDD)
        sequences: Feature sequences (shape: N, T, F)
        labels: Labels (shape: N) or (N, H) for multi-horizon
        prices: Denormalized mid-prices (shape: N)
        metadata: Metadata from export
    """

    date: str
    sequences: np.ndarray
    labels: np.ndarray
    prices: np.ndarray
    metadata: Dict


@dataclass
class LoadedData:
    """
    Complete loaded dataset.

    Attributes:
        sequences: All sequences concatenated (shape: total_N, T, F)
        labels: All labels concatenated (shape: total_N) or (total_N, H)
        prices: All prices concatenated (shape: total_N)
        day_boundaries: List of (start_idx, end_idx) for each day
        days: List of date strings
    """

    sequences: np.ndarray
    labels: np.ndarray
    prices: np.ndarray
    day_boundaries: List[Tuple[int, int]]
    days: List[str]

    def __len__(self) -> int:
        return len(self.prices)

    @property
    def n_days(self) -> int:
        return len(self.days)

    def to_backtest_data(
        self,
        horizon_idx: int = 0,
    ) -> BacktestData:
        """
        Convert to BacktestData for backtesting.

        Args:
            horizon_idx: Which horizon to use for labels (if multi-horizon)

        Returns:
            BacktestData instance
        """
        labels = self.labels
        if labels.ndim == 2:
            # Multi-horizon labels, select one
            labels = labels[:, horizon_idx]

        return BacktestData(
            prices=self.prices,
            labels=labels,
        )


class DataLoader:
    """
    Load exported data for backtesting.

    This loader handles the data format from feature-extractor-MBO-LOB:
    - NumPy sequences and labels
    - JSON metadata and normalization params
    - Multi-day datasets with train/val/test splits

    Example:
        >>> loader = DataLoader("path/to/exports", split="test")
        >>> data = loader.load()
        >>> print(f"Loaded {len(data)} samples from {data.n_days} days")
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: Literal["train", "val", "test"] = "test",
        horizon_idx: int = 0,
    ):
        """
        Initialize DataLoader.

        Args:
            data_dir: Path to export directory
            split: Data split to load ("train", "val", "test")
            horizon_idx: Which horizon to use for multi-horizon labels
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.horizon_idx = horizon_idx

        self._validate_dir()

    def _validate_dir(self) -> None:
        """Validate directory structure."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

    def load(self) -> LoadedData:
        """
        Load all data for the split.

        Validates export contract on the first day's metadata, then loads
        all days with shape assertions.

        Returns:
            LoadedData with concatenated sequences, labels, and prices

        Raises:
            FileNotFoundError: If required files are missing.
            ContractError: If export metadata fails validation.
        """
        split_dir = self.data_dir / self.split

        seq_files = sorted(split_dir.glob("*_sequences.npy"))
        if not seq_files:
            raise FileNotFoundError(f"No sequence files found in {split_dir}")

        all_sequences = []
        all_labels = []
        all_prices = []
        day_boundaries = []
        days = []
        contract_validated = False

        current_idx = 0

        for seq_file in seq_files:
            date = seq_file.stem.replace("_sequences", "")

            sequences = np.load(seq_file)

            label_file = split_dir / f"{date}_labels.npy"
            if not label_file.exists():
                raise FileNotFoundError(
                    f"Missing labels for {date}: {label_file}. "
                    "Backtester requires labels for performance evaluation."
                )
            labels = np.load(label_file)

            if sequences.shape[0] != labels.shape[0]:
                raise ContractError(
                    f"Sequence-label alignment violation for {date}: "
                    f"sequences.shape[0]={sequences.shape[0]} != "
                    f"labels.shape[0]={labels.shape[0]}"
                )

            meta_file = split_dir / f"{date}_metadata.json"
            metadata: Dict = {}
            if meta_file.exists():
                with open(meta_file) as f:
                    metadata = json.load(f)

            if not contract_validated and metadata.get("schema_version"):
                warnings = validate_export_contract(
                    metadata, strict_completeness=False
                )
                for w in warnings:
                    logger.warning("Contract warning (%s): %s", date, w)
                contract_validated = True

            norm_file = split_dir / f"{date}_normalization.json"
            norm_params = (
                NormalizationParams.from_json(norm_file)
                if norm_file.exists()
                else None
            )

            extractor = PriceExtractor(norm_params)
            prices = extractor.extract_mid_prices(sequences, denormalize=True)

            start_idx = current_idx
            end_idx = current_idx + len(sequences)
            day_boundaries.append((start_idx, end_idx))
            days.append(date)
            current_idx = end_idx

            all_sequences.append(sequences)
            all_labels.append(labels)
            all_prices.append(prices)

        return LoadedData(
            sequences=np.concatenate(all_sequences, axis=0),
            labels=np.concatenate(all_labels, axis=0),
            prices=np.concatenate(all_prices, axis=0),
            day_boundaries=day_boundaries,
            days=days,
        )

    def load_day(self, date: str) -> DayData:
        """
        Load data for a single day.

        Args:
            date: Date string (YYYYMMDD)

        Returns:
            DayData for the specified date

        Raises:
            FileNotFoundError: If sequences or labels are missing.
            ContractError: If export metadata fails validation.
        """
        split_dir = self.data_dir / self.split

        seq_file = split_dir / f"{date}_sequences.npy"
        if not seq_file.exists():
            raise FileNotFoundError(f"Sequence file not found: {seq_file}")

        sequences = np.load(seq_file)

        label_file = split_dir / f"{date}_labels.npy"
        if not label_file.exists():
            raise FileNotFoundError(
                f"Missing labels for {date}: {label_file}. "
                "Backtester requires labels for performance evaluation."
            )
        labels = np.load(label_file)

        if sequences.shape[0] != labels.shape[0]:
            raise ContractError(
                f"Sequence-label alignment violation for {date}: "
                f"sequences.shape[0]={sequences.shape[0]} != "
                f"labels.shape[0]={labels.shape[0]}"
            )

        meta_file = split_dir / f"{date}_metadata.json"
        metadata: Dict = {}
        if meta_file.exists():
            with open(meta_file) as f:
                metadata = json.load(f)

        if metadata.get("schema_version"):
            warnings = validate_export_contract(
                metadata, strict_completeness=False
            )
            for w in warnings:
                logger.warning("Contract warning (%s): %s", date, w)

        norm_file = split_dir / f"{date}_normalization.json"
        norm_params = (
            NormalizationParams.from_json(norm_file)
            if norm_file.exists()
            else None
        )

        extractor = PriceExtractor(norm_params)
        prices = extractor.extract_mid_prices(sequences, denormalize=True)

        return DayData(
            date=date,
            sequences=sequences,
            labels=labels,
            prices=prices,
            metadata=metadata,
        )

    def list_days(self) -> List[str]:
        """
        List available days in the split.

        Returns:
            List of date strings
        """
        split_dir = self.data_dir / self.split
        seq_files = sorted(split_dir.glob("*_sequences.npy"))
        return [f.stem.replace("_sequences", "") for f in seq_files]

    def get_manifest(self) -> Optional[Dict]:
        """
        Load dataset manifest if available.

        Returns:
            Manifest dict or None
        """
        manifest_file = self.data_dir / "dataset_manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                return json.load(f)
        return None

