"""Phase II hardening SB-1 (2026-04-20): backtester expected_fields wiring.

Locks the Phase II version-skew detection closure:
    Phase II v2.21 shipped the producer side (trainer emits CompatibilityContract
    + fingerprint) but not the consumer side (backtester never constructed an
    expected_contract). SB-1 extended SignalManifest.validate() with a partial-
    assertion ``expected_fields`` kwarg and wired it into ``BacktestData.
    from_signal_dir`` + ``ExperimentRunner._expected_compatibility_fields``.

Tests here verify:
    1. ``BacktestData.from_signal_dir(signal_dir, expected_fields={...})``
       raises ContractError on mismatch.
    2. ``ExperimentRunner._expected_compatibility_fields`` extracts
       ``primary_horizon_idx`` from the regression-strategy config and supplies
       it only when the user explicitly set it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from hft_contracts.compatibility import CompatibilityContract
from hft_contracts.validation import ContractError

from lobbacktest.engine.vectorized import BacktestData
from lobbacktest.experiment import ExperimentRunner


def _base_contract(**overrides: Any) -> CompatibilityContract:
    defaults = dict(
        contract_version="2.2",
        schema_version="2.2",
        feature_count=98,
        window_size=100,
        feature_layout="default",
        data_source="mbo_lob",
        label_strategy_hash="a" * 64,
        calibration_method=None,
        primary_horizon_idx=0,
        horizons=(10, 60, 300),
        normalization_strategy="none",
    )
    defaults.update(overrides)
    return CompatibilityContract(**defaults)


def _write_regression_signal_dir(
    path: Path,
    contract: CompatibilityContract,
    n_samples: int = 16,
) -> Path:
    """Produce a minimal regression signal directory on disk."""
    path.mkdir(parents=True, exist_ok=True)

    # Regression NPYs — required: prices, predicted_returns
    rng = np.random.RandomState(0)
    np.save(path / "prices.npy", np.abs(rng.randn(n_samples)) + 100.0)
    np.save(path / "predicted_returns.npy", rng.randn(n_samples, 3))
    np.save(path / "regression_labels.npy", rng.randn(n_samples, 3))
    np.save(path / "spreads.npy", np.abs(rng.randn(n_samples)) * 0.5 + 1.0)

    block = {
        "contract_version": contract.contract_version,
        "schema_version": contract.schema_version,
        "feature_count": contract.feature_count,
        "window_size": contract.window_size,
        "feature_layout": contract.feature_layout,
        "data_source": contract.data_source,
        "label_strategy_hash": contract.label_strategy_hash,
        "calibration_method": contract.calibration_method,
        "primary_horizon_idx": contract.primary_horizon_idx,
        "horizons": list(contract.horizons) if contract.horizons else None,
        "normalization_strategy": contract.normalization_strategy,
    }
    meta: Dict[str, Any] = {
        "signal_type": "regression",
        "model_type": "hmhp_regression",
        "split": "test",
        "total_samples": n_samples,
        "horizons": list(contract.horizons) if contract.horizons else None,
        "exported_at": "2026-04-20T00:00:00+00:00",
        "checkpoint": "/tmp/ckpt.pt",
        "compatibility": block,
        "compatibility_fingerprint": contract.fingerprint(),
    }
    (path / "signal_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path


class TestFromSignalDirExpectedFieldsKwarg:
    """BacktestData.from_signal_dir(..., expected_fields=...) wires through."""

    def test_matching_primary_horizon_idx_passes(self, tmp_path):
        c = _base_contract(primary_horizon_idx=0)
        d = _write_regression_signal_dir(tmp_path / "match", c)
        # Matches → loads cleanly
        data = BacktestData.from_signal_dir(
            str(d), validate=True, expected_fields={"primary_horizon_idx": 0}
        )
        assert data.prices.shape[0] == 16

    def test_mismatched_primary_horizon_idx_raises(self, tmp_path):
        c = _base_contract(primary_horizon_idx=0)
        d = _write_regression_signal_dir(tmp_path / "mismatch", c)
        with pytest.raises(ContractError, match="primary_horizon_idx"):
            BacktestData.from_signal_dir(
                str(d), validate=True, expected_fields={"primary_horizon_idx": 1}
            )

    def test_no_expected_fields_still_validates_tamper(self, tmp_path):
        """Without expected_fields, producer fingerprint check still runs (tamper detection)."""
        c = _base_contract()
        d = _write_regression_signal_dir(tmp_path / "tamper_ok", c)
        data = BacktestData.from_signal_dir(str(d), validate=True)
        assert data.prices.shape[0] == 16

    def test_typo_key_raises_valueerror(self, tmp_path):
        """Typo'd field name raises ValueError (fail-loud typo detection)."""
        c = _base_contract()
        d = _write_regression_signal_dir(tmp_path / "typo", c)
        with pytest.raises(ValueError, match="primary_horizen_idx"):
            BacktestData.from_signal_dir(
                str(d),
                validate=True,
                expected_fields={"primary_horizen_idx": 0},  # typo
            )

    # --- Phase II hardening post-audit (2026-04-21) SB-E: contradictory-args guard ---

    def test_expected_fields_with_validate_false_raises(self, tmp_path):
        """expected_fields + validate=False is a caller bug — fail loud per hft-rules §5.

        Previously: silently dropped the assertion (validate(False) never called
        SignalManifest.validate, so expected_fields never ran). Now: raises
        ValueError so the caller sees they asked for an impossible combination.
        """
        c = _base_contract(primary_horizon_idx=0)
        d = _write_regression_signal_dir(tmp_path / "ef_vfalse", c)
        with pytest.raises(ValueError, match="validate=True"):
            BacktestData.from_signal_dir(
                str(d),
                validate=False,
                expected_fields={"primary_horizon_idx": 0},
            )

    def test_validate_false_without_expected_fields_still_works(self, tmp_path):
        """validate=False alone (no expected_fields) is a valid legacy-load path."""
        c = _base_contract()
        d = _write_regression_signal_dir(tmp_path / "vfalse_only", c)
        data = BacktestData.from_signal_dir(str(d), validate=False)
        assert data.prices.shape[0] == 16


class TestExperimentRunnerExpectedFields:
    """ExperimentRunner._expected_compatibility_fields extracts from config."""

    def test_regression_strategy_with_explicit_phi_extracts(self):
        config = {
            "experiment": {"name": "test"},
            "strategy": {"type": "regression", "primary_horizon_idx": 2},
        }
        runner = ExperimentRunner(config)
        fields = runner._expected_compatibility_fields()
        assert fields == {"primary_horizon_idx": 2}

    def test_regression_strategy_without_phi_returns_none(self):
        """If user doesn't set primary_horizon_idx, don't assert it (avoids false positive)."""
        config = {
            "experiment": {"name": "test"},
            "strategy": {"type": "regression"},  # no primary_horizon_idx
        }
        runner = ExperimentRunner(config)
        fields = runner._expected_compatibility_fields()
        assert fields is None

    def test_readability_strategy_returns_none(self):
        """Classification strategies don't know primary_horizon_idx."""
        config = {
            "experiment": {"name": "test"},
            "strategy": {"type": "readability"},
        }
        runner = ExperimentRunner(config)
        fields = runner._expected_compatibility_fields()
        assert fields is None

    def test_default_regression_no_explicit_phi(self):
        """Default regression strategy, no primary_horizon_idx declared → None."""
        config = {"experiment": {"name": "test"}}
        runner = ExperimentRunner(config)
        fields = runner._expected_compatibility_fields()
        assert fields is None
