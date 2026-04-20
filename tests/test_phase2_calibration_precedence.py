"""Phase II D10 activation: backtester manifest-driven calibration precedence.

Locks the D10 fix (validation report 2026-04-20):
    - When ``signal_metadata.json::calibration_method`` is set → use
      ``calibrated_returns.npy``.
    - When ``calibration_method`` is explicitly None → use ``predicted_returns.npy``
      even if ``calibrated_returns.npy`` happens to exist (guards against a stale
      calibration file silently overriding fresh predictions).
    - Legacy signals (pre-Phase-II, no calibration_method field) fall back to
      file-existence precedence for back-compat.

Depends on SignalManifest's orphan-file guard (raises ContractError when the
calibrated file and manifest claim disagree), so happy-path tests only exercise
aligned configurations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lobbacktest.engine.vectorized import BacktestData


def _write_regression_signal_dir(
    path: Path,
    *,
    n: int = 16,
    calibration_method: str | None = None,
    write_calibrated: bool = False,
    calibrated_values: np.ndarray | None = None,
    predicted_values: np.ndarray | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)

    np.save(path / "prices.npy", np.abs(np.random.RandomState(0).randn(n)) + 100.0)
    np.save(path / "spreads.npy", np.abs(np.random.RandomState(1).randn(n)) + 0.8)

    if predicted_values is None:
        predicted_values = np.random.RandomState(2).randn(n)
    np.save(path / "predicted_returns.npy", predicted_values)
    np.save(path / "regression_labels.npy", np.random.RandomState(3).randn(n))

    if write_calibrated:
        if calibrated_values is None:
            calibrated_values = np.random.RandomState(4).randn(n)
        np.save(path / "calibrated_returns.npy", calibrated_values)

    meta = {
        "signal_type": "regression",
        "model_type": "tlob_regression",
        "split": "test",
        "total_samples": n,
        "horizons": [10, 60, 300],
        "exported_at": "2026-04-20T00:00:00+00:00",
        "checkpoint": "/tmp/ckpt.pt",
    }
    if calibration_method is not None:
        meta["calibration_method"] = calibration_method
    (path / "signal_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path


class TestManifestDrivenCalibration:
    """Backtester reads the calibration gate from the manifest, not the filesystem."""

    def test_calibration_method_set_loads_calibrated_returns(self, tmp_path):
        """manifest.calibration_method='variance_match' + calibrated file exists → calibrated wins."""
        calibrated = np.full(16, 42.0)  # distinctive marker
        predicted = np.zeros(16)
        d = _write_regression_signal_dir(
            tmp_path / "cal",
            calibration_method="variance_match",
            write_calibrated=True,
            calibrated_values=calibrated,
            predicted_values=predicted,
        )
        data = BacktestData.from_signal_dir(d, validate=True)
        assert data.predicted_returns is not None
        np.testing.assert_allclose(data.predicted_returns, calibrated), (
            "Expected calibrated_returns.npy (42.0) but got predicted_returns.npy (0.0). "
            "D10 regression: manifest-driven precedence failed."
        )

    def test_calibration_method_none_loads_predicted_returns(self, tmp_path):
        """manifest.calibration_method=None + no calibrated file → predicted is used.

        The aligned no-calibration case — SignalManifest.validate() accepts this.
        The buggy old behavior wouldn't differ here (no calibrated file), but this
        locks the contract for the manifest-driven path.
        """
        predicted = np.full(16, 7.0)
        d = _write_regression_signal_dir(
            tmp_path / "nocal",
            calibration_method=None,
            write_calibrated=False,
            predicted_values=predicted,
        )
        data = BacktestData.from_signal_dir(d, validate=True)
        np.testing.assert_allclose(data.predicted_returns, predicted)

    def test_legacy_signal_falls_back_to_file_existence(self, tmp_path):
        """Pre-Phase-II signal directory (no calibration_method field) + validate=False.

        Falls back to the OLD file-existence behavior. Locks the back-compat path
        for R1-R8 ledger-era signal directories.
        """
        calibrated = np.full(16, 99.0)
        predicted = np.zeros(16)
        # No calibration_method in metadata → legacy manifest.
        d = _write_regression_signal_dir(
            tmp_path / "legacy",
            calibration_method=None,
            write_calibrated=True,
            calibrated_values=calibrated,
            predicted_values=predicted,
        )
        # validate=False bypasses the orphan-file guard (legacy path).
        data = BacktestData.from_signal_dir(d, validate=False)
        # Legacy behavior: calibrated file existence wins.
        np.testing.assert_allclose(data.predicted_returns, calibrated)

    def test_orphan_calibrated_file_raises_via_validate(self, tmp_path):
        """Manifest says no calibration + calibrated file present → ContractError.

        This is the validation-layer guard from SignalManifest; kept here as a
        regression test to confirm the BacktestData.from_signal_dir path surfaces it.
        """
        from hft_contracts.validation import ContractError

        d = _write_regression_signal_dir(
            tmp_path / "orphan",
            calibration_method=None,
            write_calibrated=True,
        )
        with pytest.raises(ContractError, match=r"[Oo]rphan calibrated"):
            BacktestData.from_signal_dir(d, validate=True)
