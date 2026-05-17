"""
Wave 2-H H1 closure tests (2026-05-17).

Locks `secrets.token_hex(4)` collision-resistant suffix on
`BacktestRegistry.register()` run_id generation.

Pre-fix: `run_id = f"{name}_{strftime('%Y%m%d_%H%M%S')}"` had 1-second
granularity. Two `register()` calls within the same wall-clock second
produced identical run_id → `run_dir.mkdir(parents=True, exist_ok=True)`
did NOT error → second call's `atomic_write_json(result.json)` SILENTLY
overwrote first call's artifact → `_index[run_id]` dict-key assignment
also overwrote first record. Reachable for parametric sweeps and
orchestrated hft-ops sweep at high rate.

Post-fix: append 8-hex `secrets.token_hex(4)` suffix → ~4.3 billion-way
collision space within same wall-clock second.

Test strategy: fire 100 fast successive `register()` calls and assert
all 100 produce distinct run_ids + distinct result.json files exist
on disk.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest

from lobbacktest.registry import BacktestRegistry

# Same payload pattern as test_registry_atomic_writes_and_utc.py
_RUN_ID_RE_POST_H1 = re.compile(
    r"^[A-Za-z0-9_-]+_\d{8}_\d{6}_[0-9a-f]{8}$"
)


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    """Empty registry dir for collision testing."""
    d = tmp_path / "registry_collision"
    d.mkdir()
    return d


@pytest.fixture
def minimal_payload() -> Dict[str, Any]:
    """Minimal register() payload."""
    import numpy as np

    return {
        "name": "collision_test",
        "config_dict": {"k": "v"},
        "metrics": {
            "TotalReturn": 0.01,
            "WinRate": 0.5,
            "MaxDrawdown": 0.05,
            "SharpeRatio": 1.0,
        },
        "signal_metadata": {"sample": True},
        "equity_curve": None,
    }


class TestRunIdFormatPostH1:
    """Lock the new run_id format `{name}_YYYYMMDD_HHMMSS_<8hex>`."""

    def test_run_id_matches_post_h1_regex(
        self, registry_dir: Path, minimal_payload: Dict[str, Any]
    ) -> None:
        """run_id has 8-hex collision-resistant suffix appended."""
        reg = BacktestRegistry(str(registry_dir))
        run_id = reg.register(**minimal_payload)
        assert _RUN_ID_RE_POST_H1.match(run_id), (
            f"run_id {run_id!r} must match "
            f"{{name}}_{{YYYYMMDD_HHMMSS}}_{{8hex}} post-Wave-2-H-H1 closure"
        )

    def test_run_id_suffix_is_lowercase_hex(
        self, registry_dir: Path, minimal_payload: Dict[str, Any]
    ) -> None:
        """secrets.token_hex returns lowercase; lock that contract."""
        reg = BacktestRegistry(str(registry_dir))
        run_id = reg.register(**minimal_payload)
        suffix = run_id.split("_")[-1]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix), (
            f"Suffix {suffix!r} must be 8 lowercase hex chars"
        )


class TestCollisionResistanceUnderFastSuccessive:
    """The actual collision-resistance test — fast successive register() calls."""

    def test_100_fast_registers_produce_100_distinct_run_ids(
        self, registry_dir: Path, minimal_payload: Dict[str, Any]
    ) -> None:
        """100 calls within the same wall-clock second → 100 distinct run_ids.

        Pre-fix this test would fail with N << 100 distinct ids because
        all calls within the same second collided to the same strftime
        bucket. Post-fix, 8-hex suffix gives ~4.3B unique slots within
        any 1-second window.
        """
        n_calls = 100
        reg = BacktestRegistry(str(registry_dir))
        run_ids = []
        for _ in range(n_calls):
            rid = reg.register(**minimal_payload)
            run_ids.append(rid)

        # All distinct
        assert len(set(run_ids)) == n_calls, (
            f"Wave 2-H H1 regression: expected {n_calls} distinct run_ids "
            f"from fast successive register() calls; got "
            f"{len(set(run_ids))} unique out of {n_calls} total"
        )

        # All directories exist on disk (no overwrites)
        for rid in run_ids:
            run_dir = registry_dir / rid
            assert run_dir.is_dir(), (
                f"run_dir {run_dir} missing — overwrite hazard present"
            )
            result_path = run_dir / "result.json"
            assert result_path.is_file(), (
                f"result.json missing at {result_path}"
            )

    def test_index_preserves_all_records(
        self, registry_dir: Path, minimal_payload: Dict[str, Any]
    ) -> None:
        """_index dict carries all N records (pre-fix: silently overwritten)."""
        n_calls = 50
        reg = BacktestRegistry(str(registry_dir))
        for _ in range(n_calls):
            reg.register(**minimal_payload)

        # Verify on-disk index.json has all N records (round-trip)
        index_path = registry_dir / "index.json"
        assert index_path.is_file()
        index = json.loads(index_path.read_text())
        # _save_index writes a list (per BacktestSummary serialization)
        # or dict (per _index attribute). Either way, count must be N.
        if isinstance(index, list):
            assert len(index) == n_calls
        elif isinstance(index, dict):
            assert len(index) == n_calls
        else:
            pytest.fail(f"index.json has unexpected root type: {type(index)}")
