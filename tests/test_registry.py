"""
Tests for BacktestRegistry.

Tests verify:
- Registration creates run directory and index entry
- Round-trip: register → list → get preserves data
- Append-only: new registrations don't overwrite old ones
- Compare: markdown table generation

Per RULE.md:
- Invariant tests: Append-only property
- Edge tests: Empty registry, single run
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from lobbacktest.registry import BacktestRegistry


class TestBacktestRegistry:
    """Tests for BacktestRegistry."""

    @pytest.fixture
    def registry_dir(self, tmp_path):
        """Create a temporary registry directory."""
        d = tmp_path / "backtests"
        d.mkdir()
        return d

    @pytest.fixture
    def registry(self, registry_dir):
        """Create a BacktestRegistry in temp directory."""
        return BacktestRegistry(str(registry_dir))

    def _register(self, registry, name="test_run"):
        """Register a minimal backtest run."""
        config = {"initial_capital": 100000, "position_size": 0.1}
        metrics = {"TotalReturn": -0.05, "SharpeRatio": -2.5, "MaxDrawdown": 0.10,
                    "WinRate": 0.45, "total_trades": 100, "total_return": -0.05,
                    "win_rate": 0.45, "sharpe_ratio": -2.5, "max_drawdown": 0.10}
        signal_meta = {"model_name": "TLOB", "total_samples": 50000}
        return registry.register(
            name=name,
            config_dict=config,
            metrics=metrics,
            signal_metadata=signal_meta,
        )

    def test_register_creates_run(self, registry):
        """Registration creates run directory and index entry."""
        run_id = self._register(registry)
        assert run_id is not None
        assert len(run_id) > 0
        assert registry.count() == 1

    def test_list_all_returns_registered_runs(self, registry):
        """list_all() returns all registered run IDs."""
        r1 = self._register(registry, "run1")
        r2 = self._register(registry, "run2")

        all_ids = registry.list_all()
        assert r1 in all_ids
        assert r2 in all_ids
        assert len(all_ids) == 2

    def test_get_returns_result(self, registry):
        """get(run_id) returns the full result dict."""
        run_id = self._register(registry, "test_get")
        retrieved = registry.get(run_id)
        assert retrieved is not None
        assert retrieved["name"] == "test_get"

    def test_append_only(self, registry):
        """New registrations don't overwrite old ones."""
        r1 = self._register(registry, "first")
        r2 = self._register(registry, "second")

        assert registry.count() == 2
        assert registry.get(r1)["name"] == "first"
        assert registry.get(r2)["name"] == "second"

    def test_empty_registry(self, registry):
        """Empty registry returns empty list."""
        assert registry.count() == 0
        assert registry.list_all() == []

    def test_compare_returns_table(self, registry):
        """compare() returns a markdown comparison table."""
        r1 = self._register(registry, "run1")
        r2 = self._register(registry, "run2")

        table = registry.compare([r1, r2])
        assert isinstance(table, str)
        assert "run1" in table or "run2" in table
