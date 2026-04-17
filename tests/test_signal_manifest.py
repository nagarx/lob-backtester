"""Tests for SignalManifest — signal validation at load time.

Validates that signal exports are checked for file existence, shape alignment,
value ranges, and metadata consistency BEFORE the engine runs.

Reference: BACKTESTER_AUDIT_PLAN.md § M6 (from_signal_dir loads without validation)
"""

import json
from pathlib import Path

import numpy as np
import pytest

from lobbacktest.data.signal_manifest import ContractError, SignalManifest
from lobbacktest.engine.vectorized import BacktestData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_signal_dir(
    tmp_path: Path,
    n: int = 100,
    *,
    include_predictions: bool = False,
    include_predicted_returns: bool = False,
    include_labels: bool = False,
    include_spreads: bool = False,
    include_agreement: bool = False,
    include_confirmation: bool = False,
    include_metadata: bool = False,
    metadata_overrides: dict = None,
) -> Path:
    """Create a synthetic signal directory for testing."""
    rng = np.random.RandomState(42)
    d = tmp_path / "signals"
    d.mkdir(parents=True, exist_ok=True)

    # Always create prices (required by all types)
    np.save(d / "prices.npy", rng.uniform(100, 200, size=n).astype(np.float64))

    if include_predictions:
        np.save(d / "predictions.npy", rng.choice([0, 1, 2], size=n).astype(np.int64))
    if include_predicted_returns:
        np.save(d / "predicted_returns.npy", rng.randn(n).astype(np.float64) * 5.0)
    if include_labels:
        np.save(d / "labels.npy", rng.choice([0, 1, 2], size=n).astype(np.int64))
    if include_spreads:
        np.save(d / "spreads.npy", rng.uniform(0.5, 2.0, size=n).astype(np.float64))
    if include_agreement:
        np.save(d / "agreement_ratio.npy", rng.uniform(0.333, 1.0, size=n).astype(np.float64))
    if include_confirmation:
        np.save(d / "confirmation_score.npy", rng.uniform(0, 0.667, size=n).astype(np.float64))

    if include_metadata:
        meta = {
            "model_type": "test_model",
            "split": "test",
            "total_samples": n,
            "horizons": [10, 60, 300],
            "exported_at": "2026-03-17T00:00:00Z",
            "metrics": {"r2": 0.464, "ic": 0.677},
        }
        if metadata_overrides:
            meta.update(metadata_overrides)
        with open(d / "signal_metadata.json", "w") as f:
            json.dump(meta, f)

    return d


# ---------------------------------------------------------------------------
# Signal Type Detection
# ---------------------------------------------------------------------------


class TestSignalTypeDetection:
    """Signal type auto-detected from file existence."""

    def test_classification_type(self, tmp_path: Path):
        d = _create_signal_dir(tmp_path, include_predictions=True)
        manifest = SignalManifest.from_signal_dir(d)
        assert manifest.signal_type == "classification"

    def test_regression_type(self, tmp_path: Path):
        d = _create_signal_dir(tmp_path, include_predicted_returns=True)
        manifest = SignalManifest.from_signal_dir(d)
        assert manifest.signal_type == "regression"

    def test_hybrid_type(self, tmp_path: Path):
        d = _create_signal_dir(tmp_path, include_predictions=True, include_predicted_returns=True)
        manifest = SignalManifest.from_signal_dir(d)
        assert manifest.signal_type == "hybrid"


class TestManifestFromMetadata:
    """Manifest parsed from signal_metadata.json."""

    def test_parses_metadata_fields(self, tmp_path: Path):
        d = _create_signal_dir(
            tmp_path, include_predictions=True, include_metadata=True,
        )
        manifest = SignalManifest.from_signal_dir(d)
        assert manifest.model_type == "test_model"
        assert manifest.split == "test"
        assert manifest.n_samples == 100
        assert manifest.horizons == [10, 60, 300]
        assert manifest.export_timestamp == "2026-03-17T00:00:00Z"

    def test_parses_model_metrics(self, tmp_path: Path):
        d = _create_signal_dir(
            tmp_path, include_predictions=True, include_metadata=True,
        )
        manifest = SignalManifest.from_signal_dir(d)
        assert manifest.model_metrics is not None
        assert abs(manifest.model_metrics["r2"] - 0.464) < 1e-6
        assert abs(manifest.model_metrics["ic"] - 0.677) < 1e-6

    def test_infers_from_files_when_no_metadata(self, tmp_path: Path):
        d = _create_signal_dir(tmp_path, include_predictions=True)
        manifest = SignalManifest.from_signal_dir(d)
        assert manifest.n_samples == 100
        assert manifest.model_type == "unknown"


# ---------------------------------------------------------------------------
# Validation: Critical Errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    """Critical issues raise ContractError."""

    def test_missing_required_file_raises(self, tmp_path: Path):
        """prices.npy missing → ContractError."""
        d = tmp_path / "signals"
        d.mkdir()
        np.save(d / "predictions.npy", np.array([0, 1, 2]))
        # No prices.npy!

        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match="Required signal file missing"):
            manifest.validate(d)

    def test_shape_mismatch_raises(self, tmp_path: Path):
        """predictions [100] + prices [99] → ContractError."""
        d = tmp_path / "signals"
        d.mkdir()
        np.save(d / "prices.npy", np.ones(99))
        np.save(d / "predictions.npy", np.zeros(100, dtype=np.int64))

        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match="Shape mismatch"):
            manifest.validate(d)

    def test_nan_in_required_array_raises(self, tmp_path: Path):
        """NaN in prices → ContractError."""
        d = tmp_path / "signals"
        d.mkdir()
        prices = np.array([100.0, float("nan"), 102.0])
        np.save(d / "prices.npy", prices)
        np.save(d / "predictions.npy", np.array([0, 1, 2]))

        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match="Non-finite values"):
            manifest.validate(d)

    def test_metadata_sample_count_mismatch_raises(self, tmp_path: Path):
        """Metadata says 1000 samples, arrays have 100 → ContractError."""
        d = _create_signal_dir(
            tmp_path,
            n=100,
            include_predictions=True,
            include_metadata=True,
            metadata_overrides={"total_samples": 1000},
        )
        manifest = SignalManifest.from_signal_dir(d)
        with pytest.raises(ContractError, match="Sample count mismatch"):
            manifest.validate(d)


# ---------------------------------------------------------------------------
# Validation: Warnings
# ---------------------------------------------------------------------------


class TestValidationWarnings:
    """Non-critical issues produce warnings, not errors."""

    def test_optional_file_missing_warns(self, tmp_path: Path):
        """spreads missing → warning in returned list."""
        d = _create_signal_dir(tmp_path, include_predictions=True)
        manifest = SignalManifest.from_signal_dir(d)
        warnings = manifest.validate(d)
        assert any("spreads" in w for w in warnings), (
            f"Expected warning about missing spreads, got: {warnings}"
        )

    def test_clean_data_no_warnings(self, tmp_path: Path):
        """Fully valid directory with all files → empty warnings list."""
        d = _create_signal_dir(
            tmp_path,
            include_predictions=True,
            include_labels=True,
            include_spreads=True,
            include_agreement=True,
            include_confirmation=True,
            include_metadata=True,
        )
        manifest = SignalManifest.from_signal_dir(d)
        warnings = manifest.validate(d)
        # Only warnings should be about non-present optional files
        critical_warnings = [w for w in warnings if "missing" not in w.lower()]
        assert len(critical_warnings) == 0, (
            f"Expected no critical warnings, got: {critical_warnings}"
        )


# ---------------------------------------------------------------------------
# Integration with BacktestData
# ---------------------------------------------------------------------------


class TestBacktestDataValidation:
    """Integration: from_signal_dir with validate=True."""

    def test_from_signal_dir_validates_by_default(self, tmp_path: Path):
        """from_signal_dir(validate=True) catches shape mismatch."""
        d = tmp_path / "signals"
        d.mkdir()
        np.save(d / "prices.npy", np.ones(99))
        np.save(d / "predictions.npy", np.zeros(100, dtype=np.int64))

        with pytest.raises(ContractError):
            BacktestData.from_signal_dir(str(d), validate=True)

    def test_from_signal_dir_skip_validation(self, tmp_path: Path):
        """from_signal_dir(validate=False) skips validation."""
        d = _create_signal_dir(tmp_path, include_predictions=True)
        # Should not raise even without optional files
        data = BacktestData.from_signal_dir(str(d), validate=False)
        assert len(data) == 100

    def test_from_signal_dir_valid_regression(self, tmp_path: Path):
        """Valid regression signal dir loads successfully."""
        d = _create_signal_dir(
            tmp_path,
            include_predicted_returns=True,
            include_spreads=True,
            include_metadata=True,
        )
        data = BacktestData.from_signal_dir(str(d), validate=True)
        assert len(data) == 100
        assert data.predicted_returns is not None


class TestManifestSummary:
    """Human-readable summary output."""

    def test_summary_not_empty(self, tmp_path: Path):
        d = _create_signal_dir(
            tmp_path, include_predictions=True, include_metadata=True,
        )
        manifest = SignalManifest.from_signal_dir(d)
        summary = manifest.summary()
        assert "classification" in summary
        assert "100" in summary


class TestFeatureSetRefContentHashRegex:
    """Phase 6 6A.9 regression guards — `feature_set_ref.content_hash` must
    match `^[a-f0-9]{64}$` (SHA-256 hex lowercase) to be accepted. Malformed
    content_hash silently drops the entire `feature_set_ref` to None (fail-
    safe read-only consumer; trainer-side is authoritative).

    Module-level `_CONTENT_HASH_RE` is the single regex source.
    """

    def _valid_hash(self) -> str:
        return "a" * 64  # 64 lowercase hex chars

    def test_valid_hash_accepted(self, tmp_path: Path):
        d = _create_signal_dir(
            tmp_path,
            include_predictions=True,
            include_metadata=True,
            metadata_overrides={
                "feature_set_ref": {
                    "name": "nvda_98_stable_v1",
                    "content_hash": self._valid_hash(),
                }
            },
        )
        manifest = SignalManifest.from_signal_dir(d)
        assert manifest.feature_set_ref is not None
        assert manifest.feature_set_ref["name"] == "nvda_98_stable_v1"
        assert manifest.feature_set_ref["content_hash"] == self._valid_hash()

    def test_uppercase_hex_rejected(self, tmp_path: Path):
        """Contract is lowercase-hex (hft_contracts.canonical_hash.sha256_hex
        output); uppercase variants are malformed."""
        d = _create_signal_dir(
            tmp_path,
            include_predictions=True,
            include_metadata=True,
            metadata_overrides={
                "feature_set_ref": {
                    "name": "nvda_98_stable_v1",
                    "content_hash": "A" * 64,  # uppercase
                }
            },
        )
        manifest = SignalManifest.from_signal_dir(d)
        # Fail-safe: silent drop to None rather than accept malformed ref.
        assert manifest.feature_set_ref is None

    def test_short_hash_rejected(self, tmp_path: Path):
        """63 chars (not 64) → rejected."""
        d = _create_signal_dir(
            tmp_path,
            include_predictions=True,
            include_metadata=True,
            metadata_overrides={
                "feature_set_ref": {
                    "name": "nvda_98_stable_v1",
                    "content_hash": "a" * 63,  # one too short
                }
            },
        )
        manifest = SignalManifest.from_signal_dir(d)
        assert manifest.feature_set_ref is None

    def test_non_hex_rejected(self, tmp_path: Path):
        """Non-hex characters → rejected."""
        d = _create_signal_dir(
            tmp_path,
            include_predictions=True,
            include_metadata=True,
            metadata_overrides={
                "feature_set_ref": {
                    "name": "nvda_98_stable_v1",
                    "content_hash": "z" * 64,  # invalid hex character
                }
            },
        )
        manifest = SignalManifest.from_signal_dir(d)
        assert manifest.feature_set_ref is None

    def test_missing_feature_set_ref_is_none(self, tmp_path: Path):
        """Absent feature_set_ref in metadata → None (back-compat for
        manifests predating Phase 4 4c.4)."""
        d = _create_signal_dir(
            tmp_path,
            include_predictions=True,
            include_metadata=True,
        )
        manifest = SignalManifest.from_signal_dir(d)
        assert manifest.feature_set_ref is None

    def test_regex_is_module_level(self):
        """Phase 6 6A.9 hardening: regex must be compiled ONCE at import
        (not re-compiled per call). Prevents prior inline `_re.compile(...)`
        pattern inside `_from_metadata` from being reintroduced."""
        from lobbacktest.data.signal_manifest import _CONTENT_HASH_RE
        import re
        assert isinstance(_CONTENT_HASH_RE, re.Pattern)
        assert _CONTENT_HASH_RE.pattern == r"^[a-f0-9]{64}$"
