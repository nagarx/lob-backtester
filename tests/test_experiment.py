"""Tests for ExperimentRunner — config-driven backtest orchestration.

Validates that the experiment runner correctly loads configs, builds strategies,
executes backtests, registers results, and aggregates sweep results.

Reference: BACKTESTER_AUDIT_PLAN.md § Phase 3b
"""

import json
from pathlib import Path

import numpy as np
import pytest

from lobbacktest.experiment import ExperimentResult, ExperimentRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_regression_signal_dir(tmp_path: Path, n: int = 200) -> Path:
    """Create a valid regression signal directory for testing."""
    rng = np.random.RandomState(42)
    d = tmp_path / "signals"
    d.mkdir(parents=True)

    prices = rng.uniform(150, 200, size=n).astype(np.float64)
    np.save(d / "prices.npy", prices)
    np.save(d / "predicted_returns.npy", rng.randn(n).astype(np.float64) * 5.0)
    np.save(d / "regression_labels.npy", rng.randn(n).astype(np.float64) * 5.0)
    np.save(d / "spreads.npy", rng.uniform(0.5, 1.5, size=n).astype(np.float64))

    meta = {
        "model_type": "tlob_regression",
        "split": "test",
        "total_samples": n,
        "horizons": [10, 60, 300],
        "metrics": {"r2": 0.464, "ic": 0.677},
    }
    with open(d / "signal_metadata.json", "w") as f:
        json.dump(meta, f)

    return d


def _make_regression_config(signal_dir: Path, tmp_path: Path) -> dict:
    """Create a minimal experiment config dict."""
    return {
        "experiment": {
            "name": "test_experiment",
            "description": "Unit test experiment",
        },
        "signals": {"dir": str(signal_dir)},
        "backtest": {
            "initial_capital": 10_000,
            "position_size": 0.1,
            "allow_short": False,
            "exchange": "XNAS",
        },
        "strategy": {
            "type": "regression",
            "min_return_bps": 1.0,
            "max_spread_bps": 5.0,
            "primary_horizon_idx": 0,
        },
        "holding": {"type": "horizon_aligned", "hold_events": 10},
        "zero_dte": {"enabled": False},
        "output": {"dir": str(tmp_path / "registry"), "save_equity_curve": False},
    }


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def test_from_dict(self, tmp_path: Path):
        """Config loaded from dict."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        runner = ExperimentRunner(config)
        assert runner.experiment_name == "test_experiment"

    def test_from_yaml(self, tmp_path: Path):
        """Config loaded from YAML file."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)

        import yaml

        yaml_path = tmp_path / "experiment.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)

        runner = ExperimentRunner.from_yaml(str(yaml_path))
        assert runner.experiment_name == "test_experiment"


# ---------------------------------------------------------------------------
# Single Run
# ---------------------------------------------------------------------------


class TestSingleRun:
    def test_regression_run_completes(self, tmp_path: Path):
        """Single regression backtest runs to completion."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        runner = ExperimentRunner(config)
        result = runner.run()

        assert result.n_runs == 1
        assert len(result.runs) == 1
        assert result.runs[0]["metrics"] is not None
        assert "TotalReturn" in result.runs[0]["metrics"]

    def test_run_registers_to_registry(self, tmp_path: Path):
        """Results automatically registered to BacktestRegistry."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        runner = ExperimentRunner(config)
        result = runner.run()

        assert len(result.registry_ids) == 1
        assert result.registry_ids[0] != ""

        # Verify registry dir has files
        registry_dir = tmp_path / "registry"
        assert registry_dir.exists()

    def test_run_with_zero_dte(self, tmp_path: Path):
        """0DTE transformation applied when enabled.

        Post-FIND-NEW-01 closure (2026-05-16): `events_per_minute` is now
        required in the zero_dte YAML block (no silent default at the
        ZeroDtePnLTransformer). Supplied as 10.0 here for back-compat with
        the original pre-fix calibration (event-based ~1000/day → ~10/min).
        """
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        config["zero_dte"] = {
            "enabled": True,
            "delta": 0.50,
            "commission_per_contract": 0.70,
            "events_per_minute": 10.0,  # FIND-NEW-01 closure
        }
        runner = ExperimentRunner(config)
        result = runner.run()

        assert result.runs[0].get("option_metrics") is not None
        assert "option_total_return" in result.runs[0]["option_metrics"]


# ---------------------------------------------------------------------------
# #PY-226 — ZeroDteConfig builder nested-fallback + fail-loud
# ---------------------------------------------------------------------------


class TestZeroDteConfigBuildPY226:
    """Regression tests for #PY-226 (2026-05-14): _build_zero_dte_config now
    accepts `zero_dte:` block at top-level OR nested under `backtest:` to
    match production readability YAMLs. Fails loud on both-defined per
    hft-rules §8 ("never silently drop").

    Mirrors lob-backtester/configs/nvda_readability_first_xnas.yaml + _arcx
    YAML structure where `zero_dte:` lives under `backtest:` and `opra_costs:`
    lives under `zero_dte:`.
    """

    def test_zero_dte_top_level_legacy_path(self, tmp_path: Path):
        """Legacy: zero_dte: at top-level (matches pre-#PY-226 test fixtures
        in this file). Back-compat preserved."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        config["zero_dte"] = {
            "delta": 0.55,
            "commission_per_contract": 0.85,
        }
        runner = ExperimentRunner(config)
        zd_config = runner._build_zero_dte_config()
        assert zd_config.delta == 0.55
        assert zd_config.opra_costs.commission_per_contract == 0.85

    def test_zero_dte_nested_under_backtest(self, tmp_path: Path):
        """NEW #PY-226: zero_dte: nested under backtest: (production YAML pattern)."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        # Remove top-level zero_dte (set by _make_regression_config); add nested:
        del config["zero_dte"]
        config["backtest"]["zero_dte"] = {
            "delta": 0.55,
            "opra_costs": {
                "commission_per_contract": 0.85,
                "implied_vol": 0.42,
                "entry_minutes_before_close": 90.0,
            },
            "contracts_per_trade": 2,
        }
        runner = ExperimentRunner(config)
        zd_config = runner._build_zero_dte_config()
        assert zd_config.delta == 0.55
        assert zd_config.opra_costs.commission_per_contract == 0.85
        assert zd_config.opra_costs.implied_vol == 0.42
        assert zd_config.opra_costs.entry_minutes_before_close == 90.0
        assert zd_config.contracts_per_trade == 2

    def test_zero_dte_both_locations_raises(self, tmp_path: Path):
        """NEW #PY-226: fail-loud per hft-rules §8 when zero_dte: defined at
        BOTH top-level AND nested under backtest:."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        # Both locations populated — ambiguity:
        config["zero_dte"] = {"delta": 0.50}
        config["backtest"]["zero_dte"] = {"delta": 0.60}
        runner = ExperimentRunner(config)
        with pytest.raises(ValueError, match="zero_dte:.*BOTH.*top-level.*backtest"):
            runner._build_zero_dte_config()

    def test_zero_dte_opra_field_both_locations_raises(self, tmp_path: Path):
        """NEW #PY-226: fail-loud when opra_costs field defined at BOTH zd
        top-level AND nested under opra_costs:."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        del config["zero_dte"]
        config["backtest"]["zero_dte"] = {
            "commission_per_contract": 0.85,  # top-level of zd
            "opra_costs": {
                "commission_per_contract": 0.95,  # nested — conflict
            },
        }
        runner = ExperimentRunner(config)
        with pytest.raises(ValueError, match="commission_per_contract.*BOTH.*top-level.*opra_costs"):
            runner._build_zero_dte_config()

    def test_zero_dte_neither_location_defaults(self, tmp_path: Path):
        """Back-compat: neither top-level nor nested zero_dte: defined →
        all defaults preserved (does NOT fail-loud since 0 production callers
        rely on the missing-block path; see #PY-226 LATENT classification).

        HF-1 invariant (2026-05-16 LATE): default delta=0.50 (ATM regime) →
        IV inherits 0.40 (ATM-correct). The HF-1 mode-aware default at
        experiment.py:_build_zero_dte_config only changes the regime when
        delta >= 0.90 (Deep ITM); ATM regime unchanged."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        del config["zero_dte"]
        # backtest.zero_dte also not set
        runner = ExperimentRunner(config)
        zd_config = runner._build_zero_dte_config()
        assert zd_config.delta == 0.50
        assert zd_config.opra_costs.commission_per_contract == 0.70
        assert zd_config.opra_costs.implied_vol == 0.40
        assert zd_config.opra_costs.entry_minutes_before_close == 120.0
        assert zd_config.contracts_per_trade == 1


class TestPy263ExperimentRunnerAnnualization:
    """G1b / #PY-263 (2026-05-30): ExperimentRunner._build_backtest_config must
    thread the cadence-bearing zero_dte (from the YAML's ``zero_dte.bin_seconds``)
    into the metrics ``BacktestConfig`` so ``resolved_periods_per_day`` derives
    the correct sub-daily annualization (390 at 60s) instead of the legacy 1000.0
    fallback — closing the #PY-263 silent-Sharpe-inflation class on the
    ExperimentRunner path (sister to the regression/readability scripts). Before
    G1b, ``_build_backtest_config`` omitted ``zero_dte=`` so this asserted 1000.0
    (the bug). Gap-closure proof + regression-lock.
    """

    def test_bin_seconds_derives_390(self, tmp_path: Path):
        """zero_dte.bin_seconds=60 (no explicit periods_per_day) → 390.0."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        config["zero_dte"] = {"bin_seconds": 60}
        runner = ExperimentRunner(config)
        bt_config = runner._build_backtest_config()
        assert bt_config.resolved_periods_per_day == 390.0, (
            "G1b/#PY-263: zero_dte.bin_seconds=60 must derive "
            "resolved_periods_per_day = 23400/60 = 390.0 via the metrics "
            "BacktestConfig, NOT the legacy 1000.0 fallback. Got "
            f"{bt_config.resolved_periods_per_day}."
        )
        # annualization_factor = sqrt(trading_days_per_year * resolved_ppd)
        assert bt_config.annualization_factor == pytest.approx((252.0 * 390.0) ** 0.5)

    def test_events_per_minute_stays_on_1000_fallback(self, tmp_path: Path):
        """Event-based path (events_per_minute, no bin_seconds): resolved_
        periods_per_day only derives from bin_seconds, so event-based corpora
        keep the documented 1000.0 fallback — G1b does NOT change this."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        config["zero_dte"] = {"events_per_minute": 10.0}
        runner = ExperimentRunner(config)
        bt_config = runner._build_backtest_config()
        assert bt_config.resolved_periods_per_day == 1000.0

    def test_explicit_periods_per_day_and_bin_seconds_raises(self, tmp_path: Path):
        """Mutex (config.py): a config setting BOTH backtest.periods_per_day AND
        zero_dte.bin_seconds fail-louds per hft-rules §5 (both specify the same
        physical quantity). G1b now surfaces this at _build_backtest_config time."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        config["backtest"]["periods_per_day"] = 1000.0
        config["zero_dte"] = {"bin_seconds": 60}
        runner = ExperimentRunner(config)
        with pytest.raises(ValueError, match="mutually exclusive"):
            runner._build_backtest_config()


class TestHf1ModeAwareIvDefault:
    """HF-1 closure (2026-05-16 LATE; Bundle 1 hygiene post Option B Path B'):
    YAML-reader paths inherit mode-aware IV default mirroring
    OpraCalibratedCosts.deep_itm() factory (#PY-273 closed factory at
    config.py:209 but BacktestConfig.from_dict + _build_zero_dte_config
    YAML-reader sites STILL inherited hard-coded 0.40 ATM default).

    Mode-discrimination via delta>=0.90:
      - delta >= 0.90 → Deep ITM regime → IV=0.25 (per OPRA empirical median)
      - delta <  0.90 → ATM regime → IV=0.40 (preserved for back-compat)
      - Operator-explicit YAML override always wins.

    Covers experiment.py:_build_zero_dte_config path (production
    orchestrator path via ExperimentRunner.from_yaml).
    """

    def test_deep_itm_delta_inherits_iv_025_default(self, tmp_path: Path):
        """HF-1: zero_dte.delta=0.95 (Deep ITM regime) with omitted
        implied_vol → factory-aligned default 0.25 (NOT ATM's 0.40)."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        # Override delta to Deep ITM; omit implied_vol (inherits default).
        config["zero_dte"] = {
            "enabled": True,
            "delta": 0.95,
        }
        runner = ExperimentRunner(config)
        zd_config = runner._build_zero_dte_config()
        assert zd_config.delta == 0.95
        assert zd_config.opra_costs.implied_vol == 0.25, (
            f"HF-1: delta=0.95 (Deep ITM) with omitted implied_vol should "
            f"inherit 0.25 (OpraCalibratedCosts.deep_itm() factory default "
            f"per #PY-273) not 0.40 (ATM legacy default). "
            f"Got {zd_config.opra_costs.implied_vol}."
        )

    def test_atm_delta_inherits_iv_040_default(self, tmp_path: Path):
        """HF-1: zero_dte.delta=0.50 (ATM regime) with omitted implied_vol
        → 0.40 default preserved (ATM-correct for atm_call_premium=1.88
        + atm_put_premium=1.31 per class default at config.py:173).
        Regression-locks ATM correctness; only Deep ITM regime changes."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        config["zero_dte"] = {
            "enabled": True,
            "delta": 0.50,
        }
        runner = ExperimentRunner(config)
        zd_config = runner._build_zero_dte_config()
        assert zd_config.delta == 0.50
        assert zd_config.opra_costs.implied_vol == 0.40, (
            f"HF-1: delta=0.50 (ATM) preserves IV=0.40 default. "
            f"Got {zd_config.opra_costs.implied_vol}."
        )

    def test_explicit_implied_vol_overrides_mode_aware_default(self, tmp_path: Path):
        """HF-1: operator-explicit YAML implied_vol wins regardless of
        delta. Closes silent-override-drop class — explicit user intent
        must NOT be silently replaced by mode-aware default."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        # Deep ITM regime BUT explicit override (top-level zd path)
        config["zero_dte"] = {
            "enabled": True,
            "delta": 0.95,
            "implied_vol": 0.30,  # explicit (between 0.25 and 0.40)
        }
        runner = ExperimentRunner(config)
        zd_config = runner._build_zero_dte_config()
        assert zd_config.opra_costs.implied_vol == 0.30, (
            f"HF-1: explicit YAML implied_vol=0.30 must win regardless "
            f"of delta=0.95 Deep ITM regime. Got "
            f"{zd_config.opra_costs.implied_vol}."
        )

    def test_nested_opra_costs_override_wins_on_deep_itm(self, tmp_path: Path):
        """HF-1 micro-fix Agent X HIGH-1: nested `opra_costs.implied_vol`
        override path must also win on Deep ITM regime — closes the
        load-bearing `_opra_field` nested-or-top-level dispatch at
        experiment.py:633-646. Without this test, a future refactor of
        `_opra_field` could silently drop nested overrides on Deep ITM."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        # Deep ITM regime + nested override (production-canonical
        # placement per opra_costs: block convention)
        config["zero_dte"] = {
            "enabled": True,
            "delta": 0.95,
            "opra_costs": {
                "implied_vol": 0.40,  # explicit ATM-IV on Deep ITM
                                       # (sensitivity-sweep use case)
            },
        }
        runner = ExperimentRunner(config)
        zd_config = runner._build_zero_dte_config()
        assert zd_config.opra_costs.implied_vol == 0.40, (
            f"HF-1: nested opra_costs.implied_vol=0.40 must win over "
            f"mode-aware default 0.25 (delta=0.95). Got "
            f"{zd_config.opra_costs.implied_vol}."
        )

    def test_deep_itm_boundary_delta_090_inherits_025(self, tmp_path: Path):
        """HF-1 boundary case: delta=0.90 (exactly at threshold) → Deep
        ITM IV=0.25. Threshold is INCLUSIVE (>=0.90) per fix design.
        Closes ambiguity: '>0.90' vs '>=0.90' would give different
        results at exactly 0.90."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        config["zero_dte"] = {
            "enabled": True,
            "delta": 0.90,
        }
        runner = ExperimentRunner(config)
        zd_config = runner._build_zero_dte_config()
        assert zd_config.opra_costs.implied_vol == 0.25


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


class TestSweep:
    def test_sweep_produces_multiple_results(self, tmp_path: Path):
        """Sweeping 3 threshold values → 3 runs."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        config["sweep"] = {"min_return_bps": [1.0, 3.0, 5.0]}
        runner = ExperimentRunner(config)
        result = runner.run()

        assert result.n_runs == 3
        assert len(result.runs) == 3
        assert result.sweep_parameter == "min_return_bps"

        # Each run should have different sweep_value
        values = [r["sweep_value"] for r in result.runs]
        assert values == [1.0, 3.0, 5.0]

    def test_sweep_all_registered(self, tmp_path: Path):
        """All sweep runs registered to registry."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        config["sweep"] = {"min_return_bps": [1.0, 5.0]}
        runner = ExperimentRunner(config)
        result = runner.run()

        assert len(result.registry_ids) == 2
        assert all(rid != "" for rid in result.registry_ids)


# ---------------------------------------------------------------------------
# Result Aggregation
# ---------------------------------------------------------------------------


class TestExperimentResult:
    def test_summary_not_empty(self):
        """summary() returns non-empty formatted output."""
        result = ExperimentResult(
            experiment_name="test",
            n_runs=2,
            runs=[
                {"name": "run1", "metrics": {"TotalReturn": 0.05, "SharpeRatio": 1.2}},
                {"name": "run2", "metrics": {"TotalReturn": -0.02, "SharpeRatio": -0.5}},
            ],
        )
        summary = result.summary()
        assert "test" in summary
        assert "2 runs" in summary

    def test_best_by_metric(self):
        """best_by() returns run with highest metric value."""
        result = ExperimentResult(
            experiment_name="test",
            n_runs=3,
            runs=[
                {"name": "a", "metrics": {"TotalReturn": 0.05}},
                {"name": "b", "metrics": {"TotalReturn": 0.10}},
                {"name": "c", "metrics": {"TotalReturn": -0.02}},
            ],
        )
        best = result.best_by("TotalReturn")
        assert best["name"] == "b"

    def test_best_by_drawdown_minimizes(self):
        """best_by('MaxDrawdown') returns lowest absolute drawdown."""
        result = ExperimentResult(
            experiment_name="test",
            n_runs=2,
            runs=[
                {"name": "a", "metrics": {"MaxDrawdown": -0.15}},
                {"name": "b", "metrics": {"MaxDrawdown": -0.05}},
            ],
        )
        best = result.best_by("MaxDrawdown")
        assert best["name"] == "b"

    def test_empty_result_summary(self):
        """Empty result produces informative message."""
        result = ExperimentResult(experiment_name="empty", n_runs=0, runs=[])
        summary = result.summary()
        assert "No runs" in summary


# ---------------------------------------------------------------------------
# Strategy Type Detection
# ---------------------------------------------------------------------------


class TestStrategyTypes:
    def test_unknown_strategy_raises(self, tmp_path: Path):
        """Unknown strategy type raises ValueError."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        config["strategy"]["type"] = "nonexistent_strategy"
        runner = ExperimentRunner(config)

        with pytest.raises(ValueError, match="Unknown strategy type"):
            runner.run()
