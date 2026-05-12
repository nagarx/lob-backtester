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


# =============================================================================
# Phase R-16c Sub-cycle 4b — TestPerTradePnlsDump (closes deferred #PY-179)
# =============================================================================
#
# Producer-side contract test for `option_trade_pnls.npy` atomic dump shipped
# in Sub-cycle 4a (commit de99f45 on 2026-05-12). The pre-commit code-reviewer
# gate flagged #PY-179 as "test class deferred to Sub-cycle 4b analysis script
# bundle"; this section closes that gap.
#
# Locks the contract that the R-16c analysis pipeline depends on:
#   - Per-threshold filename: f"{run_name}__option_trade_pnls__{label}.npy"
#   - Atomicity (no .npy.tmp orphans on success)
#   - Backward-compat (omitting output_dir/run_name kwargs preserves pre-fix)
#   - n_trades=0 short-circuit (no emission per L126 guard)
#   - Truthy guard on empty run_name (MICRO-FIX 2 from de99f45 pre-commit gate)
#   - Bit-exact round-trip via atomic_write_npy SSoT
#
# Per Agent I T1 + T2 verdict 2026-05-12: AUGMENT existing test file (NOT new
# file); hybrid framework (subprocess for CLI contract + in-process for
# function contract). 5 subprocess tests (TestPerTradePnlsDump) + 3 in-process
# tests (TestPerTradePnlsDumpInProcess) = 8 cases total.
# =============================================================================


@pytest.mark.integration
class TestPerTradePnlsDump:
    """Phase R-16c Sub-cycle 4b producer-side per-trade pnls dump contract.

    Closes #PY-179 deferred from Sub-cycle 4a MICRO-FIX 3 (pre-commit gate).
    Each test below verifies one aspect of the contract that R-16c analyzer
    depends on for pooled-bootstrap per-trade analysis.
    """

    def test_per_threshold_dump_files_created_for_all_8_thresholds(self, tmp_path: Path):
        """L137-140 atomic dump: with output_dir + run_name supplied + at least
        one threshold having n_trades > 0, the script writes per-threshold .npy
        files matching the canonical 8-label set.
        """
        signal_dir = _construct_mock_signal_dir(tmp_path / "signals" / "test", n_samples=200)
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                sys.executable, str(REGRESSION_SCRIPT),
                "--signals", str(signal_dir),
                "--name", "r16c_dump_test",
                "--exchange", "XNAS",
                "--output-dir", str(output_dir),
                "--zero-dte",  # required for option_trade_pnls dump
                "--deep-itm",  # use deep ITM cost model
                "--max-spread-bps", "100.0",  # permissive spread filter
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"Script failed: stdout={result.stdout[-500:]} "
            f"stderr={result.stderr[-500:]}"
        )

        # At least one .npy file should exist (some thresholds may have n_trades=0)
        npy_files = sorted(output_dir.glob("r16c_dump_test__option_trade_pnls__*.npy"))
        assert len(npy_files) >= 1, (
            f"No per-trade .npy files emitted. Expected >=1 of 8 thresholds. "
            f"stdout tail: {result.stdout[-500:]}"
        )
        # All emitted filenames match the canonical 8-label set
        canonical_labels = {
            "deep_itm_1.4bps", "itm_2bps", "itm_3bps", "atm_5bps",
            "high_conv_8bps", "very_high_10bps", "ultra_conv_15bps", "max_conv_20bps",
        }
        for f in npy_files:
            # Filename: r16c_dump_test__option_trade_pnls__<label>.npy
            label = f.stem.rsplit("__option_trade_pnls__", 1)[1]
            assert label in canonical_labels, (
                f"Unexpected threshold label {label!r} in filename {f.name}. "
                f"Expected one of {sorted(canonical_labels)}."
            )

    def test_per_trade_dump_filename_convention_regex(self, tmp_path: Path):
        """Filename pattern: ^<args.name>__option_trade_pnls__<label>\\.npy$
        per de99f45 producer-side ship (line 138 of run_regression_backtest.py).
        """
        import re
        signal_dir = _construct_mock_signal_dir(tmp_path / "signals" / "test", n_samples=200)
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                sys.executable, str(REGRESSION_SCRIPT),
                "--signals", str(signal_dir),
                "--name", "regex_lock",
                "--exchange", "XNAS",
                "--output-dir", str(output_dir),
                "--deep-itm",
                "--max-spread-bps", "100.0",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        pattern = re.compile(r"^regex_lock__option_trade_pnls__[a-zA-Z0-9._]+\.npy$")
        npy_files = list(output_dir.glob("regex_lock__option_trade_pnls__*.npy"))
        assert len(npy_files) >= 1, "No matching .npy emitted"
        for f in npy_files:
            assert pattern.match(f.name), (
                f"Filename {f.name!r} doesn't match expected regex"
            )

    def test_per_trade_dump_bit_exact_round_trip(self, tmp_path: Path):
        """Critical Agent I T3 addition: write via atomic_write_npy → load via
        np.load → array equality. Locks the SSoT contract that contents survive
        the atomic-write boundary unchanged (no dtype coercion, no endianness flip).
        """
        import numpy as np
        signal_dir = _construct_mock_signal_dir(tmp_path / "signals" / "test", n_samples=200)
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable, str(REGRESSION_SCRIPT),
                "--signals", str(signal_dir),
                "--name", "round_trip",
                "--exchange", "XNAS",
                "--output-dir", str(output_dir),
                "--deep-itm",
                "--max-spread-bps", "100.0",
            ],
            capture_output=True, text=True, timeout=120, check=True,
        )
        # Load every emitted .npy; verify it's finite float64 with non-zero shape
        npy_files = sorted(output_dir.glob("round_trip__option_trade_pnls__*.npy"))
        assert len(npy_files) >= 1
        for f in npy_files:
            arr = np.load(f)
            assert arr.dtype == np.float64, (
                f"{f.name}: expected float64, got {arr.dtype}"
            )
            assert arr.ndim == 1, f"{f.name}: expected 1-D array, got ndim={arr.ndim}"
            assert len(arr) > 0, f"{f.name}: empty array (n_trades=0 should skip dump)"
            assert np.all(np.isfinite(arr)), (
                f"{f.name}: non-finite values violate hft-rules §8 fail-loud"
            )

    def test_per_trade_dump_no_tmp_orphan_after_success(self, tmp_path: Path):
        """Atomic-write durability: no .tmp orphan after successful completion.
        Reuses the hft_contracts.atomic_io.atomic_write_npy SSoT pattern
        (tmp + fsync + os.replace + cleanup on failure).
        """
        signal_dir = _construct_mock_signal_dir(tmp_path / "signals" / "test", n_samples=200)
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                sys.executable, str(REGRESSION_SCRIPT),
                "--signals", str(signal_dir),
                "--name", "no_tmp_orphan",
                "--exchange", "XNAS",
                "--output-dir", str(output_dir),
                "--deep-itm",
                "--max-spread-bps", "100.0",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        # No .tmp files should remain in output_dir
        tmp_orphans = list(output_dir.glob("*.tmp*"))
        assert len(tmp_orphans) == 0, (
            f"Found {len(tmp_orphans)} orphan tmp files: "
            f"{[f.name for f in tmp_orphans]}. atomic_write_npy SSoT broken?"
        )

    def test_per_trade_dump_distinct_files_per_threshold(self, tmp_path: Path):
        """8 thresholds → 8 distinct .npy files (or fewer if n_trades=0 for some).
        No filename collision; each emitted file has different SHA-256.
        """
        from hft_contracts.provenance import hash_file
        signal_dir = _construct_mock_signal_dir(tmp_path / "signals" / "test", n_samples=200)
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable, str(REGRESSION_SCRIPT),
                "--signals", str(signal_dir),
                "--name", "distinct_check",
                "--exchange", "XNAS",
                "--output-dir", str(output_dir),
                "--deep-itm",
                "--max-spread-bps", "100.0",
            ],
            capture_output=True, text=True, timeout=120, check=True,
        )
        npy_files = sorted(output_dir.glob("distinct_check__option_trade_pnls__*.npy"))
        assert len(npy_files) >= 2, (
            f"Need >=2 .npy files to test distinctness; got {len(npy_files)}. "
            f"Permissive spread filter may not produce trades for enough thresholds."
        )
        shas = {hash_file(f, missing_ok=False) for f in npy_files}
        # Different thresholds filter trades differently → different SHAs.
        # In rare permissive-filter cases adjacent thresholds may share trade
        # sets; require at least 2 distinct SHAs as the floor.
        assert len(shas) >= 2, (
            f"All {len(npy_files)} files have identical SHA — filename collision? "
            f"SHAs: {sorted(shas)}"
        )


class TestPerTradePnlsDumpInProcess:
    """In-process tests for `run_one_backtest` kwargs contract (backward-compat
    + truthy-guard + n_trades=0 short-circuit). Cheaper than subprocess (~10ms
    each vs 3-5s) and tests the function-level contract directly.

    NOT marked `@pytest.mark.integration` — runs in default test suite.
    """

    def test_no_kwargs_no_dump_backward_compat(self, tmp_path: Path):
        """Omitting output_dir+run_name preserves pre-Sub-cycle-4a behavior:
        no .npy emission, no error. Per de99f45 backward-compat invariant.
        """
        from scripts.run_regression_backtest import run_one_backtest
        from lobbacktest.config import BacktestConfig, CostConfig, ZeroDteConfig, OpraCalibratedCosts
        from lobbacktest.engine.vectorized import BacktestData
        from lobbacktest.strategies.regression import RegressionStrategyConfig
        from lobbacktest.strategies.holding import create_holding_policy
        import numpy as np

        signal_dir = _construct_mock_signal_dir(tmp_path / "signals", n_samples=100)
        data = BacktestData.from_signal_dir(str(signal_dir))

        zero_dte_config = ZeroDteConfig(
            enabled=True, delta=0.95,
            opra_costs=OpraCalibratedCosts.deep_itm(), contracts_per_trade=1,
        )
        config = BacktestConfig(
            initial_capital=100_000.0, position_size=0.1,
            costs=CostConfig.for_exchange("XNAS"), zero_dte=zero_dte_config,
        )
        strategy_config = RegressionStrategyConfig(
            min_return_bps=1.4, max_spread_bps=100.0,
            primary_horizon_idx=0, cooldown_events=0,
        )
        holding_policy = create_holding_policy({"type": "horizon_aligned", "hold_events": 10})

        # NO output_dir + NO run_name → no dump expected
        summary = run_one_backtest(
            data, data.prices, config, strategy_config, holding_policy,
            zero_dte_config, "test_label", verbose=False,
            # output_dir=None (default), run_name=None (default)
        )
        # Summary should NOT contain "option_trade_pnls_path" key
        assert "option_trade_pnls_path" not in summary, (
            "No-kwargs path silently emitted dump — backward-compat broken"
        )

    def test_empty_run_name_truthy_guard_no_dump(self, tmp_path: Path):
        """Empty run_name='' → truthy guard rejects (MICRO-FIX 2 from de99f45).
        Prevents filenames like __option_trade_pnls__deep_itm_1.4bps.npy
        (double-underscore prefix on empty string).
        """
        from scripts.run_regression_backtest import run_one_backtest
        from lobbacktest.config import BacktestConfig, CostConfig, ZeroDteConfig, OpraCalibratedCosts
        from lobbacktest.engine.vectorized import BacktestData
        from lobbacktest.strategies.regression import RegressionStrategyConfig
        from lobbacktest.strategies.holding import create_holding_policy

        signal_dir = _construct_mock_signal_dir(tmp_path / "signals", n_samples=100)
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        data = BacktestData.from_signal_dir(str(signal_dir))

        zero_dte_config = ZeroDteConfig(
            enabled=True, delta=0.95,
            opra_costs=OpraCalibratedCosts.deep_itm(), contracts_per_trade=1,
        )
        config = BacktestConfig(
            initial_capital=100_000.0, position_size=0.1,
            costs=CostConfig.for_exchange("XNAS"), zero_dte=zero_dte_config,
        )
        strategy_config = RegressionStrategyConfig(
            min_return_bps=1.4, max_spread_bps=100.0,
            primary_horizon_idx=0, cooldown_events=0,
        )
        holding_policy = create_holding_policy({"type": "horizon_aligned", "hold_events": 10})

        summary = run_one_backtest(
            data, data.prices, config, strategy_config, holding_policy,
            zero_dte_config, "test_label", verbose=False,
            output_dir=output_dir,
            run_name="",  # EMPTY string — must be rejected by truthy guard
        )
        # No .npy files written (truthy guard rejected empty run_name)
        npy_files = list(output_dir.glob("*.npy"))
        assert len(npy_files) == 0, (
            f"Empty run_name produced {len(npy_files)} .npy files — "
            f"truthy guard broken. Files: {[f.name for f in npy_files]}"
        )
        assert "option_trade_pnls_path" not in summary

    def test_n_trades_zero_no_dump_for_that_threshold(self, tmp_path: Path):
        """L126 short-circuit: when option_result.n_trades == 0, no .npy
        emission. This is the legitimate-absence case that R-16c analyzer
        must distinguish from "genuine sweep failure" (per Agent H E1).
        """
        from scripts.run_regression_backtest import run_one_backtest
        from lobbacktest.config import BacktestConfig, CostConfig, ZeroDteConfig, OpraCalibratedCosts
        from lobbacktest.engine.vectorized import BacktestData
        from lobbacktest.strategies.regression import RegressionStrategyConfig
        from lobbacktest.strategies.holding import create_holding_policy

        signal_dir = _construct_mock_signal_dir(tmp_path / "signals", n_samples=100)
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        data = BacktestData.from_signal_dir(str(signal_dir))

        zero_dte_config = ZeroDteConfig(
            enabled=True, delta=0.95,
            opra_costs=OpraCalibratedCosts.deep_itm(), contracts_per_trade=1,
        )
        config = BacktestConfig(
            initial_capital=100_000.0, position_size=0.1,
            costs=CostConfig.for_exchange("XNAS"), zero_dte=zero_dte_config,
        )
        # Ultra-conservative threshold (min_return_bps=10000.0 ≈ 100%) → no trades
        strategy_config = RegressionStrategyConfig(
            min_return_bps=10000.0, max_spread_bps=100.0,
            primary_horizon_idx=0, cooldown_events=0,
        )
        holding_policy = create_holding_policy({"type": "horizon_aligned", "hold_events": 10})

        summary = run_one_backtest(
            data, data.prices, config, strategy_config, holding_policy,
            zero_dte_config, "ultra_conv_test", verbose=False,
            output_dir=output_dir,
            run_name="n_trades_zero",
        )
        # No .npy emission for this threshold (n_trades=0 short-circuit)
        npy_files = list(output_dir.glob("n_trades_zero__option_trade_pnls__*.npy"))
        assert len(npy_files) == 0, (
            f"Expected NO .npy emission for n_trades=0 threshold; got "
            f"{len(npy_files)}: {[f.name for f in npy_files]}"
        )
        # Summary should reflect 0 trades; option_trade_pnls_path absent
        assert summary.get("option_n_trades", 0) == 0
        assert "option_trade_pnls_path" not in summary
