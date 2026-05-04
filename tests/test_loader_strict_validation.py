"""C-4 regression tests for backtester DataLoader strict export validation.

Phase O Cycle 1 consumer-side hardening (2026-05-04). Pre-C-4 the backtester
loader (`lobbacktest.data.loader.DataLoader.load()` and `.load_day()`) used
truthiness-based gates around `schema_version` (`metadata.get("schema_version")`),
which silently skipped validation when the key was missing or empty. The
fix brings the backtester boundary into parity with the trainer's
`load_split_data` post-C-3:
  * fail-loud on missing schema_version (every day)
  * day-context wrap on downstream validator raises
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hft_contracts.validation import ContractError
from lobbacktest.data.loader import DataLoader


NUM_SEQS = 8
SEQ_LEN = 20
NUM_FEATURES = 98


def _write_day(
    split_dir: Path,
    day: str,
    *,
    schema_version: str | None = "3.0",
    write_metadata: bool = True,
    rng: np.random.Generator | None = None,
) -> None:
    """Write one synthetic day's worth of files matching the backtester's
    expected layout."""
    if rng is None:
        rng = np.random.default_rng(42)

    seqs = rng.standard_normal((NUM_SEQS, SEQ_LEN, NUM_FEATURES)).astype(np.float32)
    labels = rng.integers(low=-1, high=2, size=NUM_SEQS, endpoint=True).astype(np.int64)
    np.save(split_dir / f"{day}_sequences.npy", seqs)
    np.save(split_dir / f"{day}_labels.npy", labels)

    if write_metadata:
        meta: dict = {"day": day, "n_sequences": NUM_SEQS, "n_features": NUM_FEATURES}
        if schema_version is not None:
            meta["schema_version"] = schema_version
        with open(split_dir / f"{day}_metadata.json", "w") as f:
            json.dump(meta, f)


@pytest.fixture
def export_dir_v3p0(tmp_path: Path) -> Path:
    """Two-day v3.0-clean test split."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    _write_day(test_dir, "20250203", schema_version="3.0")
    _write_day(test_dir, "20250204", schema_version="3.0")
    return tmp_path


@pytest.fixture
def export_dir_legacy_v22(tmp_path: Path) -> Path:
    """Two-day pre-Phase-O test split."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    _write_day(test_dir, "20250203", schema_version="2.2")
    _write_day(test_dir, "20250204", schema_version="2.2")
    return tmp_path


@pytest.fixture
def export_dir_first_clean_then_corrupt(tmp_path: Path) -> Path:
    """Day 1 v3.0, day 2 v2.2 (corrupt). Pre-C-4 the day-1-only flag would
    have silently allowed day 2 to slip through."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    _write_day(test_dir, "20250203", schema_version="3.0")
    _write_day(test_dir, "20250204", schema_version="2.2")
    return tmp_path


@pytest.fixture
def export_dir_missing_metadata_at_day2(tmp_path: Path) -> Path:
    """Day 1 v3.0 with metadata; day 2 missing metadata.json."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    _write_day(test_dir, "20250203", schema_version="3.0")
    _write_day(test_dir, "20250204", write_metadata=False)
    return tmp_path


class TestDataLoaderLoadStrictValidation:
    """C-4: DataLoader.load() validates every day's metadata."""

    def test_v3p0_export_loads_cleanly(self, export_dir_v3p0: Path) -> None:
        loader = DataLoader(export_dir_v3p0, split="test")
        data = loader.load()
        assert data.sequences.shape[0] == 2 * NUM_SEQS

    def test_legacy_v22_export_raises(self, export_dir_legacy_v22: Path) -> None:
        loader = DataLoader(export_dir_legacy_v22, split="test")
        with pytest.raises(ContractError) as excinfo:
            loader.load()
        assert "20250203" in str(excinfo.value), (
            f"day name must appear in error, got {excinfo.value!r}"
        )

    def test_first_clean_then_corrupt_raises_on_day2(
        self, export_dir_first_clean_then_corrupt: Path
    ) -> None:
        """Day-1-only validation would have missed day 2's corruption."""
        loader = DataLoader(export_dir_first_clean_then_corrupt, split="test")
        with pytest.raises(ContractError) as excinfo:
            loader.load()
        assert "20250204" in str(excinfo.value), (
            f"day 2's name must appear in error, got {excinfo.value!r}"
        )

    def test_missing_metadata_day2_raises(
        self, export_dir_missing_metadata_at_day2: Path
    ) -> None:
        loader = DataLoader(export_dir_missing_metadata_at_day2, split="test")
        with pytest.raises(ContractError) as excinfo:
            loader.load()
        assert "20250204" in str(excinfo.value), (
            f"day 2's name must appear in error, got {excinfo.value!r}"
        )


class TestDataLoaderLoadDayStrictValidation:
    """C-4: DataLoader.load_day() also validates fail-loud."""

    def test_load_day_v3p0_passes(self, export_dir_v3p0: Path) -> None:
        loader = DataLoader(export_dir_v3p0, split="test")
        day_data = loader.load_day("20250203")
        assert day_data.sequences.shape[0] == NUM_SEQS

    def test_load_day_v22_raises(self, export_dir_legacy_v22: Path) -> None:
        loader = DataLoader(export_dir_legacy_v22, split="test")
        # Tight regex (V.A.5 audit feedback): match the diagnostic signature
        # `'2.2' != expected '3.0'` rather than the permissive `schema version`
        # which would also match the missing-key path's error string.
        with pytest.raises(
            ContractError, match=r"'2\.2' != expected '3\.0'"
        ):
            loader.load_day("20250203")

    def test_load_day_missing_metadata_raises(
        self, export_dir_missing_metadata_at_day2: Path
    ) -> None:
        loader = DataLoader(export_dir_missing_metadata_at_day2, split="test")
        with pytest.raises(ContractError, match=r"no 'schema_version' field"):
            loader.load_day("20250204")
