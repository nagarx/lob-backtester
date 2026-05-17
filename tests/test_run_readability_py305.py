"""
#PY-305 closure tests (2026-05-17).

Locks sentinel-None mode-aware default at `scripts/run_readability_backtest.py:123`
(--implied-vol) + `:125` (--entry-minutes-before-close).

Pre-fix: `default=0.40` hardcoded silently inherited ATM IV when operator
passed `--delta 0.95` (Deep ITM) without `--implied-vol` → ~60-100% theta
overstatement → readability strategy wrongly rejected at cost-floor gate.

Post-fix: CLI defaults are None (sentinel); explicit operator value wins;
otherwise IV resolves mode-aware (0.25 if delta >= 0.90 else 0.40).
Class-coherent with `run_regression_backtest.py:389-415` pattern.

Test methodology: argparse-only invocation via subprocess. Script will
fail (no signals) but we capture stdout BEFORE the failure to verify
the mode-aware print line. Signal-dir failure exits before backtest runs;
we only need to see the mode line in early output.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Locate the script (sibling of tests/)
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
READABILITY_SCRIPT = SCRIPTS_DIR / "run_readability_backtest.py"


def _construct_mock_signal_dir(target_dir: Path, n_samples: int = 100) -> Path:
    """Minimal signal-dir for readability backtest (predictions + agreement + confirmation)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / "predictions.npy", np.random.randint(0, 3, size=n_samples).astype(np.int64))
    np.save(target_dir / "agreement_ratio.npy", np.random.uniform(0.5, 1.0, size=n_samples).astype(np.float64))
    np.save(target_dir / "confirmation_score.npy", np.random.uniform(0.3, 0.9, size=n_samples).astype(np.float64))
    np.save(target_dir / "spreads.npy", np.abs(np.random.randn(n_samples).astype(np.float64)) + 0.5)
    np.save(target_dir / "prices.npy", 100.0 + np.cumsum(np.random.randn(n_samples).astype(np.float64) * 0.01))
    np.save(target_dir / "labels.npy", np.random.randint(0, 3, size=n_samples).astype(np.int64))
    metadata = {
        "schema_version": "3.0",
        "contract_version": "3.0",
        "model_type": "hmhp",
        "model_name": "HMHP",
        "parameters": 100,
        "signal_type": "classification",
        "split": "test",
        "total_samples": n_samples,
        "checkpoint": "/tmp/mock.pkl",
        "horizons": [10, 60, 300],
    }
    (target_dir / "signal_metadata.json").write_text(json.dumps(metadata))
    return target_dir


class TestPy305ModeAwareDefault:
    """#PY-305: sentinel-None resolution honors --delta for IV default."""

    @pytest.mark.integration
    def test_deep_itm_delta_uses_iv_025_mode_aware_default(self, tmp_path: Path):
        """--delta 0.95 (no --implied-vol) → IV=0.25 mode-aware default."""
        signal_dir = _construct_mock_signal_dir(tmp_path / "signals" / "test")
        result = subprocess.run(
            [
                sys.executable, str(READABILITY_SCRIPT),
                "--signals", str(signal_dir),
                "--exchange", "XNAS",
                "--delta", "0.95",  # Deep ITM regime
                "--bin-seconds", "60",  # FIND-NEW-01 required
            ],
            capture_output=True, text=True, timeout=60,
        )
        # Script may complete or partial-fail; we only need to verify
        # the mode line appears in stdout BEFORE any failure
        assert "Mode: delta=0.95" in result.stdout, (
            f"Mode line missing or wrong delta; stdout: {result.stdout[-1500:]}"
        )
        assert "IV=0.25" in result.stdout, (
            f"#PY-305 mode-aware default IV=0.25 not in mode line; "
            f"stdout: {result.stdout[-1500:]}"
        )
        assert "[#PY-305 mode-aware default]" in result.stdout, (
            "Mode line should mark IV as #PY-305 mode-aware default"
        )

    @pytest.mark.integration
    def test_atm_delta_uses_iv_040_mode_aware_default(self, tmp_path: Path):
        """--delta 0.50 (no --implied-vol) → IV=0.40 ATM default."""
        signal_dir = _construct_mock_signal_dir(tmp_path / "signals" / "test")
        result = subprocess.run(
            [
                sys.executable, str(READABILITY_SCRIPT),
                "--signals", str(signal_dir),
                "--exchange", "XNAS",
                "--delta", "0.50",  # ATM regime (default)
                "--bin-seconds", "60",
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert "Mode: delta=0.5" in result.stdout, (
            f"Mode line missing or wrong delta; stdout: {result.stdout[-1500:]}"
        )
        assert "IV=0.40" in result.stdout, (
            f"#PY-305 ATM mode-aware default IV=0.40 not in mode line; "
            f"stdout: {result.stdout[-1500:]}"
        )

    @pytest.mark.integration
    def test_explicit_iv_override_wins(self, tmp_path: Path):
        """--implied-vol 0.20 explicit → IV=0.20 [CLI override] (overrides mode default)."""
        signal_dir = _construct_mock_signal_dir(tmp_path / "signals" / "test")
        result = subprocess.run(
            [
                sys.executable, str(READABILITY_SCRIPT),
                "--signals", str(signal_dir),
                "--exchange", "XNAS",
                "--delta", "0.95",  # Deep ITM (would normally → 0.25)
                "--implied-vol", "0.20",  # explicit override
                "--bin-seconds", "60",
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert "IV=0.20" in result.stdout, (
            f"Explicit --implied-vol 0.20 not honored; stdout: {result.stdout[-1500:]}"
        )
        assert "[CLI override]" in result.stdout, (
            "Explicit IV should be tagged [CLI override]"
        )

    @pytest.mark.integration
    def test_boundary_delta_090_uses_deep_itm_default(self, tmp_path: Path):
        """--delta 0.90 (boundary) → IV=0.25 (>=0.90 is Deep ITM)."""
        signal_dir = _construct_mock_signal_dir(tmp_path / "signals" / "test")
        result = subprocess.run(
            [
                sys.executable, str(READABILITY_SCRIPT),
                "--signals", str(signal_dir),
                "--exchange", "XNAS",
                "--delta", "0.90",  # Exactly at boundary
                "--bin-seconds", "60",
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert "IV=0.25" in result.stdout, (
            f"Boundary delta=0.90 should resolve to Deep ITM IV=0.25; "
            f"stdout: {result.stdout[-1500:]}"
        )

    @pytest.mark.integration
    def test_below_boundary_delta_089_uses_atm_default(self, tmp_path: Path):
        """--delta 0.89 → IV=0.40 (just below boundary, still ATM)."""
        signal_dir = _construct_mock_signal_dir(tmp_path / "signals" / "test")
        result = subprocess.run(
            [
                sys.executable, str(READABILITY_SCRIPT),
                "--signals", str(signal_dir),
                "--exchange", "XNAS",
                "--delta", "0.89",  # Just below 0.90 boundary
                "--bin-seconds", "60",
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert "IV=0.40" in result.stdout, (
            f"Boundary delta=0.89 should resolve to ATM IV=0.40; "
            f"stdout: {result.stdout[-1500:]}"
        )
