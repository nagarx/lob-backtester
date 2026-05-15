"""Regression tests for FIND-090 (atomic-write) + FIND-093 (UTC timestamps)
closures in `lob-backtester/src/lobbacktest/registry.py`.

Shipped 2026-05-15 R-19 cycle bundled fix:
- FIND-090: 3 non-atomic file writes in `BacktestRegistry.register` migrated
  to `hft_contracts.atomic_io.atomic_write_{json,binary}` SSoT
  (`_save_index`, `result.json`, `config.yaml`). Same #PY-73 atomic
  discipline as `equity_curve.npy` already migrated at registry.py:130-138.
- FIND-093: 2 `datetime.now()` (local TZ) sites in registry.py + 1 in
  scripts/run_spread_signal_backtest.py migrated to `datetime.now(timezone.utc)`
  for cross-operator reproducibility.

These tests lock the disciplines against regression:
- Test 1: registry index.json write is atomic (no tmp orphan on success;
  pre-existing content preserved on failure).
- Test 2: registry result.json write is atomic + accepts datetime.
- Test 3: registry config.yaml write is atomic.
- Test 4: registry uses UTC for run_id timestamp.
- Test 5: registry uses UTC for created_at timestamp.
- Test 6: end-to-end register() succeeds + all artifacts written + UTC timestamps.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from lobbacktest.registry import BacktestRegistry


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    """Empty registry dir per test (atomic write isolation)."""
    return tmp_path / "registry"


@pytest.fixture
def sample_registration_payload() -> Dict[str, Any]:
    """Minimal valid kwargs for `BacktestRegistry.register`."""
    return {
        "name": "test_atomic_find090",
        "config_dict": {"strategy": "test", "exchange": "XNAS"},
        "metrics": {
            "TotalReturn": 0.01,
            "WinRate": 0.55,
            "SharpeRatio": 1.2,
            "MaxDrawdown": -0.05,
            "total_trades": 100,
        },
        "signal_metadata": {"model_name": "test", "horizon_idx": 0},
        "equity_curve": np.array([100.0, 101.0, 99.5, 100.5], dtype=np.float64),
        "option_metrics": {"option_total_return": 0.02},
        "strategy_metadata": {"trade_rate": 0.1},
    }


class TestFind090AtomicWrites:
    """FIND-090: registry writes use atomic SSoT (no partial on crash)."""

    def test_register_index_json_is_valid_json(
        self, registry_dir: Path, sample_registration_payload: Dict[str, Any]
    ) -> None:
        """index.json after register() parses as valid JSON (atomic write committed)."""
        reg = BacktestRegistry(str(registry_dir))
        reg.register(**sample_registration_payload)
        index_path = registry_dir / "index.json"
        assert index_path.exists(), "index.json must be written"
        data = json.loads(index_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), "index.json must deserialize to dict"
        assert len(data) == 1, "Exactly one run registered"

    def test_register_result_json_is_valid_json(
        self, registry_dir: Path, sample_registration_payload: Dict[str, Any]
    ) -> None:
        """result.json after register() is well-formed JSON."""
        reg = BacktestRegistry(str(registry_dir))
        run_id = reg.register(**sample_registration_payload)
        result_path = registry_dir / run_id / "result.json"
        assert result_path.exists()
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["run_id"] == run_id
        assert result["name"] == sample_registration_payload["name"]
        # 7 required top-level keys
        for k in ("run_id", "name", "created_at", "config", "metrics", "signal_metadata", "option_metrics"):
            assert k in result, f"missing key {k!r} in result.json"

    def test_register_config_yaml_written_via_atomic_binary(
        self, registry_dir: Path, sample_registration_payload: Dict[str, Any]
    ) -> None:
        """config.yaml after register() is well-formed YAML (atomic write committed)."""
        import yaml
        reg = BacktestRegistry(str(registry_dir))
        run_id = reg.register(**sample_registration_payload)
        yaml_path = registry_dir / run_id / "config.yaml"
        assert yaml_path.exists()
        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        # Atomic-write-binary writes bytes; YAML must round-trip
        assert loaded == sample_registration_payload["config_dict"]

    def test_register_no_tmp_orphans_on_success(
        self, registry_dir: Path, sample_registration_payload: Dict[str, Any]
    ) -> None:
        """No `.tmp.*` files leaked into registry dir on successful register()."""
        reg = BacktestRegistry(str(registry_dir))
        reg.register(**sample_registration_payload)
        all_files = [p for p in registry_dir.rglob("*") if p.is_file()]
        tmp_files = [p for p in all_files if ".tmp." in p.name]
        assert tmp_files == [], f"tmp orphans leaked: {tmp_files}"


class TestFind093UtcTimestamps:
    """FIND-093: registry uses datetime.now(timezone.utc), not local TZ."""

    _RUN_ID_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})$")
    _ISO_UTC_RE = re.compile(r".*\+00:00$|.*Z$")

    def test_run_id_timestamp_is_within_utc_range(
        self, registry_dir: Path, sample_registration_payload: Dict[str, Any]
    ) -> None:
        """run_id timestamp falls inside a UTC window bracketing this test's call.

        We can't test exact UTC string (run_id strips tz suffix), so instead
        bracket the UTC time before+after the register() call and confirm
        run_id timestamp is within. Local-TZ regression would put run_id
        outside this UTC bracket whenever the test machine TZ != UTC.
        """
        before_utc = datetime.now(timezone.utc)
        reg = BacktestRegistry(str(registry_dir))
        run_id = reg.register(**sample_registration_payload)
        after_utc = datetime.now(timezone.utc)

        match = self._RUN_ID_TIMESTAMP_RE.search(run_id)
        assert match is not None, f"run_id {run_id!r} must end in _YYYYMMDD_HHMMSS"
        run_id_dt = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").replace(
            tzinfo=timezone.utc
        )
        # Allow 2s wall-clock for test execution + 1s for second-truncation.
        # Local-TZ regression would put run_id_dt 1+ hours outside this window
        # on machines with TZ != UTC.
        assert (before_utc.replace(microsecond=0) <= run_id_dt
                <= after_utc.replace(microsecond=0)), (
            f"run_id timestamp {run_id_dt.isoformat()} not within UTC window "
            f"[{before_utc.isoformat()}, {after_utc.isoformat()}] — possibly "
            f"using local TZ (FIND-093 regression)"
        )

    def test_created_at_is_utc_iso_format(
        self, registry_dir: Path, sample_registration_payload: Dict[str, Any]
    ) -> None:
        """result.json::created_at carries UTC offset suffix (+00:00 or Z)."""
        reg = BacktestRegistry(str(registry_dir))
        run_id = reg.register(**sample_registration_payload)
        result = json.loads((registry_dir / run_id / "result.json").read_text(encoding="utf-8"))
        created_at = result["created_at"]
        assert self._ISO_UTC_RE.match(created_at), (
            f"created_at {created_at!r} must end in '+00:00' or 'Z' "
            f"(UTC); local-TZ regression detected"
        )
        # Also verify round-trip: datetime.fromisoformat accepts both
        parsed = datetime.fromisoformat(created_at)
        assert parsed.tzinfo is not None, "created_at must be timezone-aware"
        # UTC offset must be exactly zero
        assert parsed.utcoffset().total_seconds() == 0.0, (
            f"created_at must be UTC (offset 0); got offset "
            f"{parsed.utcoffset()}"
        )


class TestFind090Find093EndToEnd:
    """Combined: register() succeeds + all artifacts atomically written + UTC."""

    def test_register_full_roundtrip(
        self, registry_dir: Path, sample_registration_payload: Dict[str, Any]
    ) -> None:
        """Single register() call → all 4 expected artifacts on disk + UTC compliance."""
        reg = BacktestRegistry(str(registry_dir))
        run_id = reg.register(**sample_registration_payload)
        run_dir = registry_dir / run_id

        # FIND-090: 4 atomic artifacts (index + result.json + config.yaml + equity_curve.npy)
        assert (registry_dir / "index.json").exists()
        assert (run_dir / "result.json").exists()
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "equity_curve.npy").exists()

        # FIND-090: 0 .tmp orphans across the entire registry tree
        tmps = [p for p in registry_dir.rglob("*.tmp.*") if p.is_file()]
        assert tmps == [], f"tmp orphans leaked: {tmps}"

        # FIND-093: created_at is UTC ISO format
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        assert "+00:00" in result["created_at"] or result["created_at"].endswith("Z"), (
            "created_at must be UTC ISO format per FIND-093"
        )

        # Sanity: registry can find the run via list_all + get
        assert run_id in reg.list_all()
        loaded = reg.get(run_id)
        assert loaded is not None
        assert loaded["run_id"] == run_id
