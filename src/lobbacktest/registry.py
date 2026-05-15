"""
Backtest experiment registry.

Append-only storage for backtest runs with config, metrics, and provenance.
Enables comparison across different strategy configurations and models.

Design:
    registry_dir/
    ├── index.json        ← Quick-lookup metadata for all runs
    ├── {run_id}.json     ← Full result per run
    └── ...
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# FIND-090 closure (2026-05-15 R-19 cycle bundled fix): registry writes
# (_save_index, result.json, config.yaml) migrate to atomic SSoT to close
# SIGKILL-mid-write corruption hazard. Sister of #PY-73 atomic_write_npy
# already in use at L145-150 for equity_curve.npy.
from hft_contracts.atomic_io import atomic_write_binary, atomic_write_json

logger = logging.getLogger(__name__)


@dataclass
class BacktestSummary:
    """Summary of a single backtest run for quick comparison."""

    run_id: str
    name: str
    created_at: str
    model_name: str
    strategy_name: str
    exchange: str

    total_trades: int
    total_return: float
    final_equity: float
    max_drawdown: float
    win_rate: float
    sharpe_ratio: float
    trade_rate: float

    option_total_return: Optional[float] = None
    option_win_rate: Optional[float] = None

    extra: Dict[str, Any] = field(default_factory=dict)


class BacktestRegistry:
    """
    Append-only registry for backtest runs.

    Args:
        base_dir: Directory for storing run results.
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_dir / "index.json"
        self._index: Dict[str, Dict[str, Any]] = {}
        if self._index_path.exists():
            with open(self._index_path) as f:
                self._index = json.load(f)

    def _save_index(self) -> None:
        # FIND-090 closure: atomic-write SSoT (#PY-73 sister; 2026-05-15
        # R-19 cycle bundled fix). Pre-fix: SIGKILL mid-write of
        # _index_path corrupts the per-run index lookup, blocking all
        # subsequent registry.compare() / list_all() queries.
        #
        # DELIBERATE deviation from atomic_write_json SSoT canonical
        # `sort_keys=True` default (hft-contracts/atomic_io.py:38-44):
        # registry index.json semantically tracks experiments in
        # TEMPORAL order. Run_ids are `{name}_YYYYMMDD_HHMMSS`, and
        # operator-facing readers (compare() output, manual ledger
        # browse) expect insertion order = temporal order. Activating
        # `sort_keys=True` would alphabetize the entire historical file
        # on next write — equivalent for same-name runs but cosmetic
        # churn across different name prefixes. Preserves pre-fix
        # `json.dump` implicit default. NOT content-addressable, NOT a
        # golden fixture — exempt from SSoT byte-stability rationale.
        atomic_write_json(self._index_path, self._index, sort_keys=False)

    def register(
        self,
        name: str,
        config_dict: Dict[str, Any],
        metrics: Dict[str, float],
        signal_metadata: Dict[str, Any],
        equity_curve: Optional[np.ndarray] = None,
        option_metrics: Optional[Dict[str, float]] = None,
        strategy_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a backtest run.

        Args:
            name: Human-readable run name.
            config_dict: Full backtest config (serializable).
            metrics: Computed metrics (sharpe, max_dd, win_rate, etc.).
            signal_metadata: Model provenance from signal export.
            equity_curve: Optional equity curve array.
            option_metrics: Optional 0DTE option metrics.
            strategy_metadata: Optional strategy gate stats.

        Returns:
            Run ID for reference.
        """
        # FIND-093 closure (2026-05-15 R-19 cycle bundled fix): datetime.now()
        # without tzinfo returns local-machine TZ → cross-operator reproducibility
        # break (timestamp comparison across operators with different TZ is
        # silently inconsistent). Use UTC explicitly per pipeline convention
        # (hft-contracts.timestamp_utils + Phase A.5.1 ISO-8601 SSoT).
        # `run_id` uses strftime without tz suffix — UTC value still serializes
        # deterministically (e.g., "20260515_142030") matching prior format.
        now_utc = datetime.now(timezone.utc)
        run_id = f"{name}_{now_utc.strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "run_id": run_id,
            "name": name,
            "created_at": now_utc.isoformat(),
            "config": config_dict,
            "metrics": metrics,
            "signal_metadata": signal_metadata,
            "strategy_metadata": strategy_metadata or {},
            "option_metrics": option_metrics or {},
        }

        # FIND-090 closure: atomic-write JSON for result.json (default=str
        # internal at atomic_write_json:191 honors datetime/Enum/Path values
        # the same as pre-fix json.dump(..., default=str)).
        atomic_write_json(run_dir / "result.json", result, sort_keys=False)

        # FIND-090 closure: atomic-write YAML via atomic_write_binary +
        # bytes-encoding lambda. yaml.dump emits str; encode to bytes for
        # the BinaryIO file handle.
        yaml_bytes = yaml.dump(config_dict, default_flow_style=False).encode("utf-8")
        atomic_write_binary(run_dir / "config.yaml", lambda f: f.write(yaml_bytes))

        if equity_curve is not None:
            # #PY-73 atomic write — SIGKILL mid-write would corrupt the
            # equity-curve artifact downstream analytics consume.
            # Migrated 2026-05-11 (hft-contracts v2.7.0).
            from hft_contracts.atomic_io import atomic_write_npy
            atomic_write_npy(run_dir / "equity_curve.npy", equity_curve)

        # 2026-05-05 P0 fix: read PascalCase metric keys (canonical from
        # `VectorizedEngine._compute_metrics` at `vectorized.py:646-651`
        # which dict-keys by `metric.name` = class name e.g. `WinRate`,
        # `SharpeRatio`, `TotalReturn`, `MaxDrawdown`). Pre-fix: registry
        # silently stored zero for these because callers spread
        # `**result.metrics` (PascalCase) but the registry read lowercase
        # keys (`metrics.get('win_rate')` etc.) — same class of bug as the
        # `run_regression_backtest.py` summary-table bug. Companion fix:
        # PascalCase preferred, lowercase fallback preserved for callers
        # that explicitly pass lowercase kwargs (back-compat).
        self._index[run_id] = {
            "name": name,
            "created_at": result["created_at"],
            "total_trades": metrics.get("total_trades", 0),
            "total_return": metrics.get("TotalReturn", metrics.get("total_return", 0)),
            "win_rate": metrics.get("WinRate", metrics.get("win_rate", 0)),
            "sharpe_ratio": metrics.get("SharpeRatio", metrics.get("sharpe_ratio", 0)),
            "max_drawdown": metrics.get("MaxDrawdown", metrics.get("max_drawdown", 0)),
            "trade_rate": (strategy_metadata or {}).get("trade_rate", 0),
            "option_total_return": (option_metrics or {}).get("option_total_return"),
        }
        self._save_index()

        logger.info(f"Registered backtest: {run_id}")
        return run_id

    def list_all(self) -> List[str]:
        """List all run IDs."""
        return list(self._index.keys())

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load full result for a run."""
        result_path = self.base_dir / run_id / "result.json"
        if not result_path.exists():
            return None
        with open(result_path) as f:
            return json.load(f)

    def compare(self, run_ids: Optional[List[str]] = None) -> str:
        """
        Generate markdown comparison table.

        Args:
            run_ids: Specific runs to compare (None = all).

        Returns:
            Markdown table string.
        """
        ids = run_ids or list(self._index.keys())
        if not ids:
            return "No backtest runs found."

        lines = [
            "| Run | Trades | Return | Win Rate | Sharpe | MaxDD | Trade Rate | Option Return |",
            "|---|---|---|---|---|---|---|---|",
        ]

        for rid in ids:
            meta = self._index.get(rid, {})
            opt_ret = meta.get("option_total_return")
            opt_str = f"{opt_ret:.2%}" if opt_ret is not None else "N/A"
            lines.append(
                f"| {meta.get('name', rid)[:30]} "
                f"| {meta.get('total_trades', 0)} "
                f"| {meta.get('total_return', 0):.2%} "
                f"| {meta.get('win_rate', 0):.2%} "
                f"| {meta.get('sharpe_ratio', 0):.2f} "
                f"| {meta.get('max_drawdown', 0):.2%} "
                f"| {meta.get('trade_rate', 0):.1%} "
                f"| {opt_str} |"
            )

        return "\n".join(lines)

    def count(self) -> int:
        return len(self._index)
