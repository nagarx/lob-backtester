"""Config-driven backtest experiment orchestrator.

Replaces manual script chaining (load → build → run → save) with a
single YAML-driven runner that validates inputs, executes backtests
(including parameter sweeps), and registers results automatically.

Usage:
    runner = ExperimentRunner.from_yaml("configs/experiment.yaml")
    result = runner.run()
    print(result.summary())

Or from dict:
    runner = ExperimentRunner(config_dict)
    result = runner.run()

Reference:
    BACKTESTER_AUDIT_PLAN.md § Phase 3b
    CLAUDE.md § Pipeline Overview
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from lobbacktest.config import BacktestConfig, CostConfig, OpraCalibratedCosts, ZeroDteConfig
from lobbacktest.data.signal_manifest import SignalManifest
from lobbacktest.engine.vectorized import BacktestData, VectorizedEngine
from lobbacktest.engine.zero_dte import ZeroDtePnLTransformer
from lobbacktest.registry import BacktestRegistry
from lobbacktest.strategies.direction import DirectionStrategy
from lobbacktest.strategies.holding import (
    HoldingPolicy,
    HorizonAlignedPolicy,
    DirectionReversalPolicy,
    StopLossTakeProfitPolicy,
    create_holding_policy,
)
from lobbacktest.strategies.readability import ReadabilityConfig, ReadabilityStrategy
from lobbacktest.strategies.regression import RegressionStrategy, RegressionStrategyConfig


@dataclass
class ExperimentResult:
    """Aggregated result from one or more backtest runs.

    Attributes:
        experiment_name: Name from config.
        n_runs: Number of runs (1 if no sweep, >1 if sweep).
        runs: Per-run results (config params + metrics).
        registry_ids: BacktestRegistry run IDs for each run.
        sweep_parameter: Which parameter was swept (None if single run).
    """

    experiment_name: str
    n_runs: int
    runs: List[Dict[str, Any]]
    registry_ids: List[str] = field(default_factory=list)
    sweep_parameter: Optional[str] = None

    def summary(self) -> str:
        """Human-readable markdown summary table."""
        if not self.runs:
            return f"Experiment '{self.experiment_name}': No runs completed."

        lines = [
            f"=== {self.experiment_name} ({self.n_runs} runs) ===",
            "",
        ]

        # Build table header from first run's metrics
        first = self.runs[0]
        metric_keys = [k for k in first.get("metrics", {}).keys()]
        option_keys = [k for k in first.get("option_metrics", {}).keys()]

        # Sweep column
        sweep_col = self.sweep_parameter or "run"

        header_parts = [f"| {sweep_col}"]
        for k in metric_keys[:6]:  # Limit to 6 metrics for readability
            header_parts.append(f" {k}")
        if option_keys:
            header_parts.append(" opt_return")
        header = " | ".join(header_parts) + " |"
        sep = "|" + "---|" * (len(header_parts))

        lines.append(header)
        lines.append(sep)

        for run in self.runs:
            sweep_val = run.get("sweep_value", run.get("name", "—"))
            parts = [f"| {sweep_val}"]
            for k in metric_keys[:6]:
                val = run.get("metrics", {}).get(k, 0)
                parts.append(f" {val:.4f}" if isinstance(val, float) else f" {val}")
            if option_keys:
                opt_ret = run.get("option_metrics", {}).get("option_total_return", 0)
                parts.append(f" {opt_ret:.2%}" if isinstance(opt_ret, float) else " —")
            lines.append(" | ".join(parts) + " |")

        return "\n".join(lines)

    def best_by(self, metric: str) -> Optional[Dict[str, Any]]:
        """Return the run with the best value for a given metric.

        For return metrics (containing 'return'), higher is better.
        For risk metrics (containing 'drawdown'), lower absolute is better.
        Default: higher is better.

        Args:
            metric: Metric name (e.g., "TotalReturn", "SharpeRatio").

        Returns:
            The best run dict, or None if no runs.
        """
        if not self.runs:
            return None

        def get_val(run):
            return run.get("metrics", {}).get(metric, float("-inf"))

        if "drawdown" in metric.lower():
            return min(self.runs, key=lambda r: abs(get_val(r)))
        return max(self.runs, key=get_val)


class ExperimentRunner:
    """Config-driven backtest experiment orchestrator.

    Loads a YAML or dict config, validates signal inputs, executes
    one or more backtests (with optional parameter sweep), and
    registers all results to BacktestRegistry.

    Args:
        config: Dict with experiment configuration. See YAML schema
            in plan documentation for full reference.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_name = config.get("experiment", {}).get("name", "unnamed")

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentRunner":
        """Load experiment config from YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            ExperimentRunner ready to execute.
        """
        import yaml

        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    def run(self) -> ExperimentResult:
        """Execute the experiment: load → validate → run → register → aggregate.

        Returns:
            ExperimentResult with all runs and their metrics.
        """
        # 1. Load signals
        signal_dir = Path(self.config.get("signals", {}).get("dir", ""))
        data = BacktestData.from_signal_dir(str(signal_dir), validate=True)

        # Load signal metadata for provenance
        signal_metadata = self._load_signal_metadata(signal_dir)

        # 2. Build base config
        backtest_cfg = self._build_backtest_config()

        # 3. Determine strategy params
        strategy_config = self.config.get("strategy", {})
        strategy_type = strategy_config.get("type", "regression")
        base_params = {
            k: v for k, v in strategy_config.items() if k != "type"
        }

        # 4. Check for sweep
        sweep_config = self.config.get("sweep", {})
        if sweep_config:
            runs = self._run_sweep(data, backtest_cfg, strategy_type, base_params, sweep_config, signal_metadata)
        else:
            run = self._run_single(data, backtest_cfg, strategy_type, base_params, signal_metadata)
            runs = [run]

        # 5. Determine sweep parameter name
        sweep_param = None
        if sweep_config:
            sweep_param = list(sweep_config.keys())[0] if sweep_config else None

        return ExperimentResult(
            experiment_name=self.experiment_name,
            n_runs=len(runs),
            runs=runs,
            registry_ids=[r.get("registry_id", "") for r in runs],
            sweep_parameter=sweep_param,
        )

    def _run_sweep(
        self,
        data: BacktestData,
        backtest_cfg: BacktestConfig,
        strategy_type: str,
        base_params: dict,
        sweep_config: dict,
        signal_metadata: dict,
    ) -> List[Dict[str, Any]]:
        """Run parameter sweep — one backtest per parameter value."""
        results = []
        for param_name, values in sweep_config.items():
            if not isinstance(values, list):
                continue
            for value in values:
                params = {**base_params, param_name: value}
                run = self._run_single(
                    data, backtest_cfg, strategy_type, params, signal_metadata,
                )
                run["sweep_param"] = param_name
                run["sweep_value"] = value
                results.append(run)
        return results

    def _run_single(
        self,
        data: BacktestData,
        backtest_cfg: BacktestConfig,
        strategy_type: str,
        params: dict,
        signal_metadata: dict,
    ) -> Dict[str, Any]:
        """Execute a single backtest run."""
        # Build strategy
        strategy = self._build_strategy(data, strategy_type, params)

        # Run engine
        engine = VectorizedEngine(backtest_cfg)
        result = engine.run(data, strategy)

        # Optional: 0DTE transform
        zero_dte_config = self.config.get("zero_dte", {})
        option_metrics = {}
        if zero_dte_config.get("enabled", False):
            transformer = ZeroDtePnLTransformer(self._build_zero_dte_config())
            zero_dte_result = transformer.transform(result)
            option_metrics = {
                "option_total_return": zero_dte_result.option_total_return,
                "option_win_rate": zero_dte_result.option_win_rate,
            }

        # Register to registry
        output_config = self.config.get("output", {})
        registry_dir = output_config.get("dir", "outputs/backtests")
        registry = BacktestRegistry(registry_dir)

        run_name = f"{self.experiment_name}_{strategy.name}"
        registry_id = registry.register(
            name=run_name,
            config_dict=self._serialize_config(params),
            metrics=result.metrics,
            signal_metadata=signal_metadata,
            option_metrics=option_metrics if option_metrics else None,
            equity_curve=result.equity_curve if output_config.get("save_equity_curve", False) else None,
        )

        return {
            "name": run_name,
            "registry_id": registry_id,
            "strategy": strategy.name,
            "params": params,
            "metrics": result.metrics,
            "option_metrics": option_metrics,
            "total_trades": result.total_trades,
            "final_equity": result.final_equity,
        }

    def _build_strategy(
        self, data: BacktestData, strategy_type: str, params: dict,
    ):
        """Build strategy from type + params. Reuses existing classes."""
        holding_policy = self._build_holding_policy()

        if strategy_type == "regression":
            return RegressionStrategy(
                predicted_returns=data.predicted_returns,
                spreads=data.spreads,
                prices=data.prices,
                config=RegressionStrategyConfig(
                    min_return_bps=params.get("min_return_bps", 5.0),
                    max_spread_bps=params.get("max_spread_bps", 1.05),
                    primary_horizon_idx=params.get("primary_horizon_idx", 0),
                    cooldown_events=params.get("cooldown_events", 0),
                ),
                holding_policy=holding_policy,
            )
        elif strategy_type == "readability":
            return ReadabilityStrategy(
                predictions=data.predictions,
                agreement_ratio=data.agreement_ratio,
                confirmation_score=data.confirmation_score,
                spreads=data.spreads,
                prices=data.prices,
                config=ReadabilityConfig(
                    min_agreement=params.get("min_agreement", 0.667),
                    min_confidence=params.get("min_confidence", 0.65),
                    max_spread_bps=params.get("max_spread_bps", 1.05),
                ),
                holding_policy=holding_policy,
            )
        elif strategy_type == "direction":
            return DirectionStrategy(
                data.predictions,
                shifted=params.get("shifted", True),
            )
        else:
            raise ValueError(f"Unknown strategy type: '{strategy_type}'")

    def _build_holding_policy(self) -> HoldingPolicy:
        """Build holding policy from config."""
        holding_cfg = self.config.get("holding", {})
        policy_type = holding_cfg.get("type", "horizon_aligned")
        hold_events = holding_cfg.get("hold_events", 10)

        if policy_type == "horizon_aligned":
            return HorizonAlignedPolicy(hold_events=hold_events)
        elif policy_type == "direction_reversal":
            return DirectionReversalPolicy(max_hold_events=hold_events)
        elif policy_type == "stop_loss_take_profit":
            return StopLossTakeProfitPolicy(
                max_hold_events=hold_events,
                stop_loss_bps=holding_cfg.get("stop_loss_bps", 10.0),
                take_profit_bps=holding_cfg.get("take_profit_bps", 20.0),
            )
        return HorizonAlignedPolicy(hold_events=hold_events)

    def _build_backtest_config(self) -> BacktestConfig:
        """Build BacktestConfig from experiment config."""
        bt = self.config.get("backtest", {})
        exchange = bt.get("exchange", "XNAS")

        return BacktestConfig(
            initial_capital=bt.get("initial_capital", 100_000.0),
            position_size=bt.get("position_size", 0.1),
            allow_short=bt.get("allow_short", False),
            costs=CostConfig.for_exchange(exchange),
            trading_days_per_year=bt.get("trading_days_per_year", 252.0),
            periods_per_day=bt.get("periods_per_day", 1000.0),
        )

    def _build_zero_dte_config(self) -> ZeroDteConfig:
        """Build ZeroDteConfig from experiment config."""
        zd = self.config.get("zero_dte", {})
        return ZeroDteConfig(
            enabled=True,
            delta=zd.get("delta", 0.50),
            opra_costs=OpraCalibratedCosts(
                commission_per_contract=zd.get("commission_per_contract", 0.70),
                implied_vol=zd.get("implied_vol", 0.40),
                entry_minutes_before_close=zd.get("entry_minutes_before_close", 120.0),
            ),
            contracts_per_trade=zd.get("contracts_per_trade", 1),
        )

    def _load_signal_metadata(self, signal_dir: Path) -> dict:
        """Load signal_metadata.json for provenance tracking."""
        meta_path = signal_dir / "signal_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {"source": str(signal_dir), "metadata_available": False}

    def _serialize_config(self, strategy_params: dict) -> dict:
        """Serialize full experiment config for registry storage."""
        return {
            "experiment": self.config.get("experiment", {}),
            "backtest": self.config.get("backtest", {}),
            "strategy": {
                **self.config.get("strategy", {}),
                **strategy_params,
            },
            "holding": self.config.get("holding", {}),
            "zero_dte": self.config.get("zero_dte", {}),
        }
