"""Phase R-17 F1: regression tests for --manifest argparse + ledger-linkage block
in run_regression_backtest.py + run_spread_signal_backtest.py.

Closes #PY-129 producer-side: orchestrator unconditionally injects --manifest
into all 3 backtester scripts at hft-ops/src/hft_ops/stages/backtesting.py:137-138
but pre-R-17 only readability accepted it.

Test scope:
- F1a: argparse accepts --manifest (smoke via --help)
- F1b: ledger record written when --manifest supplied with valid YAML
- F1c: ledger NOT written when --manifest omitted (backward-compat)
- F1d: --manifest nonexistent path logs warning, doesn't crash
- F1e: spread_signal script accepts --manifest with explicit unused notice

Per H1 agent ground-truth review:
- regression script has NO `run_id` variable → uses args.name as ledger key
- regression script's `all_results` is a list of 8 threshold dicts → emits
  hybrid record (top-level best-of + full all_thresholds breakdown)
- spread_signal script lacks ledger-write architecture → accept+notice only
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
REGRESSION_SCRIPT = SCRIPTS_DIR / "run_regression_backtest.py"
SPREAD_SIGNAL_SCRIPT = SCRIPTS_DIR / "run_spread_signal_backtest.py"
READABILITY_SCRIPT = SCRIPTS_DIR / "run_readability_backtest.py"


# =============================================================================
# F1 Smoke Tests — argparse contract via --help
# =============================================================================


class TestArgparseManifestAcceptance:
    """Verify all 3 backtester scripts accept --manifest after Phase R-17 F1.

    Phase R-17 closes #PY-129: pre-R-17, orchestrator injected --manifest into
    ALL 3 scripts but only readability accepted it. R-17 producer-side fix adds
    --manifest to regression + spread_signal scripts. Smoke via --help is the
    cheapest contract test.
    """

    def test_regression_script_accepts_manifest_arg(self):
        """run_regression_backtest.py --help shows --manifest flag."""
        result = subprocess.run(
            [sys.executable, str(REGRESSION_SCRIPT), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"--help failed: {result.stderr}"
        assert "--manifest" in result.stdout, (
            "regression script argparse does NOT declare --manifest (R-17 F1 regression)"
        )
        assert "ledger-linkage" in result.stdout or "ledger linkage" in result.stdout, (
            "--manifest help text should mention ledger-linkage purpose"
        )

    def test_spread_signal_script_accepts_manifest_arg(self):
        """run_spread_signal_backtest.py --help shows --manifest (accept-and-notice)."""
        result = subprocess.run(
            [sys.executable, str(SPREAD_SIGNAL_SCRIPT), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"--help failed: {result.stderr}"
        assert "--manifest" in result.stdout, (
            "spread_signal script does NOT declare --manifest (R-17 F1 regression)"
        )
        # spread_signal accepts but doesn't write — help text should clarify
        assert ("orchestrator compatibility" in result.stdout
                or "does NOT write" in result.stdout
                or "unused" in result.stdout), (
            "--manifest help text should clarify the no-op for spread_signal"
        )

    def test_readability_script_still_accepts_manifest_arg(self):
        """Readability already had --manifest pre-R-17 — verify F1 didn't regress it."""
        result = subprocess.run(
            [sys.executable, str(READABILITY_SCRIPT), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"--help failed: {result.stderr}"
        assert "--manifest" in result.stdout

    def test_regression_argparse_has_no_collision_with_existing_flags(self):
        """F1 adds --manifest; verify no name clash with existing regression flags."""
        result = subprocess.run(
            [sys.executable, str(REGRESSION_SCRIPT), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        # Pre-R-17 flags must all still be present
        expected_flags = [
            "--signals", "--name", "--exchange", "--initial-capital",
            "--position-size", "--max-spread-bps", "--hold-events",
            "--zero-dte", "--commission", "--implied-vol",
            "--entry-minutes-before-close", "--delta", "--deep-itm",
            "--output-dir", "--primary-horizon-idx",
        ]
        for flag in expected_flags:
            assert flag in result.stdout, (
                f"Pre-R-17 flag {flag} missing from regression script "
                f"argparse — F1 may have accidentally removed it"
            )


# =============================================================================
# F1 Integration Tests — ledger-linkage record content
# =============================================================================


def _construct_mock_signal_dir(target_dir: Path, n_samples: int = 100) -> Path:
    """Create minimal valid signal directory for regression backtest."""
    import numpy as np
    target_dir.mkdir(parents=True, exist_ok=True)
    # Multi-horizon predicted_returns (N, H) shape for regression
    np.save(target_dir / "predicted_returns.npy", np.random.randn(n_samples, 3).astype(np.float64))
    np.save(target_dir / "regression_labels.npy", np.random.randn(n_samples, 3).astype(np.float64))
    np.save(target_dir / "spreads.npy", np.abs(np.random.randn(n_samples).astype(np.float64)) + 0.5)
    np.save(target_dir / "prices.npy", 100.0 + np.cumsum(np.random.randn(n_samples).astype(np.float64) * 0.01))
    metadata = {
        "schema_version": "3.0",
        "contract_version": "3.0",
        "model_type": "temporal_ridge",
        "model_name": "TemporalRidge",
        "parameters": 100,
        "signal_type": "regression",
        "split": "test",
        "total_samples": n_samples,
        "checkpoint": "/tmp/mock.pkl",
        "horizons": [10, 60, 300],
    }
    (target_dir / "signal_metadata.json").write_text(json.dumps(metadata))
    return target_dir


@pytest.mark.integration
class TestRegressionLedgerLinkage:
    """Phase R-17 F1: regression script writes hft-ops ledger record when --manifest supplied.

    Mock-signal integration: creates minimal valid signal directory + minimal
    manifest YAML, invokes script, verifies ledger record written with expected
    structure (hybrid: top-level best-of + all_thresholds breakdown).
    """

    def test_ledger_record_written_when_manifest_supplied(self, tmp_path: Path):
        """F1b: --manifest <valid_yaml> triggers ledger-linkage record write."""
        # Construct mock signal directory
        signal_dir = _construct_mock_signal_dir(tmp_path / "signals" / "test")

        # Construct minimal valid manifest YAML
        manifest_path = tmp_path / "experiments" / "test_manifest.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            "experiment:\n"
            "  name: test_F1_integration\n"
        )

        # Output directory
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Invoke regression script
        result = subprocess.run(
            [
                sys.executable, str(REGRESSION_SCRIPT),
                "--signals", str(signal_dir),
                "--name", "F1_integration_run",
                "--exchange", "XNAS",
                "--output-dir", str(output_dir),
                "--manifest", str(manifest_path),
                "--no-zero-dte",  # skip 0DTE path to simplify; uses spot-leg
            ],
            capture_output=True, text=True, timeout=120,
        )

        # Script should complete successfully
        assert result.returncode == 0, (
            f"Regression script failed: stdout={result.stdout[-500:]} "
            f"stderr={result.stderr[-500:]}"
        )

        # Verify ledger record at expected path: <manifest.parent.parent>/ledger/runs/<exp>_backtest_<name>.json
        ledger_path = manifest_path.parent.parent / "ledger" / "runs"
        record_path = ledger_path / "test_F1_integration_backtest_F1_integration_run.json"
        assert record_path.exists(), (
            f"Ledger record NOT written at {record_path}. "
            f"stdout tail: {result.stdout[-500:]}"
        )

        # Verify record schema (hybrid: top-level best-of + all_thresholds breakdown)
        with record_path.open() as f:
            record = json.load(f)

        # Required top-level keys
        required_keys = {
            "experiment_name", "stage", "status", "run_id",
            "best_threshold", "best_option_return_pct", "best_option_win_rate",
            "best_n_entries", "best_total_return", "best_win_rate", "best_sharpe_ratio",
            "all_thresholds", "holding_policy", "exchange", "zero_dte_enabled",
            "signal_dir", "manifest",
        }
        missing = required_keys - set(record.keys())
        assert not missing, f"Record missing required keys: {missing}"

        # Structural assertions
        assert record["experiment_name"] == "test_F1_integration"
        assert record["stage"] == "backtesting"
        assert record["status"] == "completed"
        assert record["run_id"] == "F1_integration_run"
        assert record["exchange"] == "XNAS"
        assert record["zero_dte_enabled"] is False
        assert isinstance(record["all_thresholds"], list)
        assert len(record["all_thresholds"]) == 8  # 8 thresholds per script L239-248

        # Per-threshold record structure (R-16a hypothesis-aligned)
        for t in record["all_thresholds"]:
            assert "label" in t
            assert "n_entries" in t
            assert "win_rate" in t
            assert "total_return" in t
            assert "option_return_pct" in t
            assert "option_win_rate" in t

    def test_ledger_NOT_written_when_manifest_omitted(self, tmp_path: Path):
        """F1c: backward-compat — omitting --manifest preserves pre-R-17 behavior."""
        signal_dir = _construct_mock_signal_dir(tmp_path / "signals" / "test")
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        # NO ledger directory should exist before
        # (would create if --manifest supplied)
        ledger_root = tmp_path / "ledger"

        result = subprocess.run(
            [
                sys.executable, str(REGRESSION_SCRIPT),
                "--signals", str(signal_dir),
                "--name", "F1_no_manifest_run",
                "--exchange", "XNAS",
                "--output-dir", str(output_dir),
                "--no-zero-dte",
                # NO --manifest
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"Regression script failed without --manifest: {result.stderr[-500:]}"
        )
        # Standard output file should still exist
        assert (output_dir / "F1_no_manifest_run.json").exists()
        # NO ledger directory created
        assert not ledger_root.exists(), (
            "Ledger directory should NOT exist when --manifest omitted "
            "(backward-compat regression)"
        )

    def test_manifest_nonexistent_path_logs_warning_no_crash(self, tmp_path: Path):
        """F1d: --manifest <nonexistent> path doesn't crash; script completes
        with warning (mirrors readability:352-353 readonly behavior)."""
        signal_dir = _construct_mock_signal_dir(tmp_path / "signals" / "test")
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        nonexistent_manifest = tmp_path / "does_not_exist.yaml"
        # File deliberately not created

        result = subprocess.run(
            [
                sys.executable, str(REGRESSION_SCRIPT),
                "--signals", str(signal_dir),
                "--name", "F1_nonexistent_manifest",
                "--exchange", "XNAS",
                "--output-dir", str(output_dir),
                "--manifest", str(nonexistent_manifest),
                "--no-zero-dte",
            ],
            capture_output=True, text=True, timeout=120,
        )
        # Script should succeed (warning, not crash) — matches readability pattern
        assert result.returncode == 0, (
            f"Script crashed on nonexistent --manifest path: {result.stderr[-500:]}"
        )
        # Standard output still produced
        assert (output_dir / "F1_nonexistent_manifest.json").exists()


# =============================================================================
# F1 Spread Signal Acceptance — accept + notice
# =============================================================================


class TestSpreadSignalManifestAcceptance:
    """F1e: spread_signal accepts --manifest with explicit notice that
    ledger-linkage is unused (no §5 violation; documented no-op).

    spread_signal lacks per-run ledger architecture (uses module-level
    constants for output config; iterates 5 variants × 5 thresholds × ...).
    Accept-and-notice preserves orchestrator compat without misleading.
    """

    def test_spread_signal_argparse_accepts_orchestrator_args_smoke(self):
        """spread_signal --help reveals --manifest accept-and-ignore semantics."""
        result = subprocess.run(
            [sys.executable, str(SPREAD_SIGNAL_SCRIPT), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"--help failed: {result.stderr}"
        assert "--manifest" in result.stdout
        # Notice text should clarify the no-op
        assert any(phrase in result.stdout for phrase in [
            "orchestrator compatibility",
            "does NOT write",
            "unused",
        ]), "--manifest help text must clarify it's accepted but unused"
