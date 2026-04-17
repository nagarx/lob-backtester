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
        """0DTE transformation applied when enabled."""
        signal_dir = _create_regression_signal_dir(tmp_path)
        config = _make_regression_config(signal_dir, tmp_path)
        config["zero_dte"] = {"enabled": True, "delta": 0.50, "commission_per_contract": 0.70}
        runner = ExperimentRunner(config)
        result = runner.run()

        assert result.runs[0].get("option_metrics") is not None
        assert "option_total_return" in result.runs[0]["option_metrics"]


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
