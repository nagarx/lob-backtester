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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

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
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

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
        run_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "run_id": run_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "config": config_dict,
            "metrics": metrics,
            "signal_metadata": signal_metadata,
            "strategy_metadata": strategy_metadata or {},
            "option_metrics": option_metrics or {},
        }

        with open(run_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        with open(run_dir / "config.yaml", "w") as f:
            import yaml
            yaml.dump(config_dict, f, default_flow_style=False)

        if equity_curve is not None:
            np.save(run_dir / "equity_curve.npy", equity_curve)

        self._index[run_id] = {
            "name": name,
            "created_at": result["created_at"],
            "total_trades": metrics.get("total_trades", 0),
            "total_return": metrics.get("total_return", 0),
            "win_rate": metrics.get("win_rate", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
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
