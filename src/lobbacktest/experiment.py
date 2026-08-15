"""Config-driven backtest experiment orchestrator.

Replaces manual script chaining (load → build → run → save) with a
single YAML-driven runner that validates inputs, executes backtests
(including parameter sweeps), and registers results automatically.

STATUS (2026-05-30): NOT on the hft-ops orchestrator path. The production
pipeline's ``backtesting`` stage shells out to the standalone scripts under
``scripts/`` (``run_regression_backtest.py`` / ``run_readability_backtest.py`` /
``run_spread_signal_backtest.py``) via subprocess — see
``hft-ops/src/hft_ops/stages/backtesting.py``. ``ExperimentRunner`` is
exercised by this repo's own test suite; treat the scripts as the production
entry points. (Documented to avoid the hft-rules §11 drift hazard of two
divergent run paths — both now close #PY-263 annualization, but the scripts
are what hft-ops actually invokes.)

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

import dataclasses
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hft_contracts.compatibility import CompatibilityContract
from hft_contracts.signal_manifest import SignalManifest
from lobbacktest.config import BacktestConfig, CostConfig, OpraCalibratedCosts, ZeroDteConfig
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


# FIND-070 closure (2026-05-14): per hft-rules §8 "Never silently drop, clamp,
# or 'fix' data without recording diagnostics." Mirrors hft-ops Phase 7.5 R5
# idiom at commit 3dd3ccb. Pre-FIND-070, ExperimentRunner silently dropped any
# YAML key not enumerated below — most notably, production YAMLs
# configs/nvda_readability_first_{xnas,arcx}.yaml declared `min_agreement: 1.0`
# + `min_confidence: 0.65` under `backtest:` block where `_build_strategy`
# (def at line 415; readability params consumed at lines 499-500) reads from
# `strategy:` block only, so the values evaporated silently and the runner
# used defaults `0.667`/`0.65` (per readability.py:54 P5 FIX 2026-03-17).
# LATENT in production (those YAMLs are not currently runnable via
# ExperimentRunner — they lack `signals.dir` so the runner would crash before
# reaching the gate); the fix is FUTURE-PROTECTION for operators copying the
# YAML pattern.
_KNOWN_BACKTEST_KEYS = frozenset(
    {
        # Fields consumed by _build_backtest_config + BacktestConfig dataclass schema
        "initial_capital",
        "position_size",
        "max_position",
        "costs",  # sub-dict; ExperimentRunner reads via CostConfig.for_exchange(exchange)
        "zero_dte",  # nested location accepted per #PY-226 closure 2026-05-14
        "allow_short",
        "fill_price",
        "stop_loss_pct",
        "take_profit_pct",
        "trading_days_per_year",
        "periods_per_day",
        "exchange",  # top-level override read by _build_backtest_config
        # DEPRECATED — declared on BacktestConfig dataclass (config.py:312-313)
        # but NOT consumed by _build_backtest_config + NOT read by engine. Live
        # home for these gate values is the `strategy:` block (consumed by
        # ReadabilityStrategy via _build_strategy:354-355). Listed here so the
        # generic WARN does NOT fire (legacy schema acceptance), but
        # _build_strategy emits a precise ValueError when readability strategy
        # is built with these in the wrong block. Slated for removal 2026-10-31
        # under separate cycle (see PHASE_P_BACKLOG.md #PY-NEW filed alongside
        # this fix).
        "min_confidence",
        "min_agreement",
    }
)

# Strategy-specific known keys, per _build_strategy branches. `type` is the
# discriminator on every set.
_KNOWN_STRATEGY_KEYS_REGRESSION = frozenset(
    {
        "type",
        "min_return_bps",
        "max_spread_bps",
        "primary_horizon_idx",
        "cooldown_events",
    }
)
_KNOWN_STRATEGY_KEYS_READABILITY = frozenset(
    {
        "type",
        "min_agreement",
        "min_confidence",
        "max_spread_bps",
    }
)
_KNOWN_STRATEGY_KEYS_DIRECTION = frozenset({"type", "shifted"})

_STRATEGY_KEY_SETS: Dict[str, frozenset] = {
    "regression": _KNOWN_STRATEGY_KEYS_REGRESSION,
    "readability": _KNOWN_STRATEGY_KEYS_READABILITY,
    "direction": _KNOWN_STRATEGY_KEYS_DIRECTION,
}

# Holding policy keys, per _build_holding_policy.
_KNOWN_HOLDING_KEYS = frozenset(
    {
        "type",
        "hold_events",
        "stop_loss_bps",
        "take_profit_bps",
    }
)

# `signals:` block keys. Until 2026-08-15 this block was read for `dir` ONLY
# (`run()`), with no unknown-key diagnostic — so a typo'd `signals.expects:`
# was silently ignored. Enumerated here so `_warn_unknown_yaml_keys` covers it
# like every other block (FIND-070 idiom).
_KNOWN_SIGNALS_KEYS = frozenset({"dir", "expect"})

# The 11 identity fields of the producer-side CompatibilityContract, derived
# from the dataclass itself rather than re-typed here. hft-rules §1: the
# contract has ONE home (`hft_contracts.compatibility`); a hand-copied list
# would drift silently the first time a field is added upstream.
_CONTRACT_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(CompatibilityContract))


def _warn_unknown_yaml_keys(
    block_name: str,
    raw: Dict[str, Any],
    known: frozenset,
) -> None:
    """Emit ``RuntimeWarning`` when ``raw`` has keys not in ``known``.

    FIND-070 closure (2026-05-14). Mirrors hft-ops Phase 7.5 R5 idiom at
    commit ``3dd3ccb`` per hft-rules §8. Operator-visible diagnostic;
    construction proceeds after the warn.

    The ``stacklevel=3`` assumes the 2-hop call chain
    ``_warn_unknown_yaml_keys -> _build_{backtest_config,holding_policy,
    strategy} -> caller``; if the helper is moved deeper or invoked from a
    different call shape, the stacklevel must be updated.
    """
    unknown = set(raw.keys()) - known
    if unknown:
        warnings.warn(
            f"ExperimentRunner: silently dropping unknown YAML keys "
            f"{sorted(unknown)!r} under `{block_name}:` block (not declared on "
            f"schema). Known keys: {sorted(known)!r}. If these are typos, "
            f"please correct them. If they belong in a different block, please "
            f"relocate per lob-backtester/CLAUDE.md FIND-070 closure.",
            RuntimeWarning,
            stacklevel=3,
        )


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
        signals_block = self.config.get("signals", {}) or {}
        _warn_unknown_yaml_keys("signals", signals_block, _KNOWN_SIGNALS_KEYS)
        signal_dir = Path(signals_block.get("dir", ""))

        # Read the manifest BEFORE validating. It is needed twice: for
        # provenance (as always) and — since 2026-08-15 — to decide which
        # expectations are checkable. Ordering matters: an expectation about a
        # field the producer never declared must be grandfathered, and that
        # cannot be known without the manifest in hand. Pure read; safe to hoist.
        signal_metadata = self._load_signal_metadata(signal_dir)

        # Phase II hardening SB-1 (2026-04-20): wire backtester consumer-side
        # CompatibilityContract partial assertion. `primary_horizon_idx` is
        # DERIVED from the strategy config; every other field must be DECLARED
        # under `signals.expect:` (the backtester cannot compute a
        # label_strategy_hash). Anything neither derived nor declared is still
        # trusted via the manifest's producer fingerprint self-check. This
        # catches the silent-version-skew case where a backtester configured
        # for H10 accidentally loads signals produced for H60 (different
        # primary_horizon_idx, identical producer fingerprint).
        expected_fields = self._expected_compatibility_fields(
            manifest_compat=signal_metadata.get("compatibility"),
        )
        data = BacktestData.from_signal_dir(
            str(signal_dir), validate=True, expected_fields=expected_fields
        )

        # 2. Build base config
        backtest_cfg = self._build_backtest_config()

        # 3. Determine strategy params
        strategy_config = self.config.get("strategy", {})
        strategy_type = strategy_config.get("type", "regression")
        base_params = {k: v for k, v in strategy_config.items() if k != "type"}

        # 4. Check for sweep
        sweep_config = self.config.get("sweep", {})
        if sweep_config:
            runs = self._run_sweep(
                data, backtest_cfg, strategy_type, base_params, sweep_config, signal_metadata
            )
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
                    data,
                    backtest_cfg,
                    strategy_type,
                    params,
                    signal_metadata,
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
            # FIND-NEW-01 closure (2026-05-16): resolve sampling cadence from
            # the YAML zero_dte block (or backtest.zero_dte nested form per
            # #PY-226 closure). ZeroDteConfig.resolved_events_per_minute
            # returns None when neither bin_seconds nor events_per_minute is
            # set in YAML — fail-loud per hft-rules §5 with actionable
            # migration message pointing at the correct YAML schema.
            built_zd_config = self._build_zero_dte_config()
            events_per_minute = built_zd_config.resolved_events_per_minute
            if events_per_minute is None:
                raise ValueError(
                    "ExperimentRunner: zero_dte.enabled=True requires either "
                    "'bin_seconds' or 'events_per_minute' in the zero_dte: "
                    "block (FIND-NEW-01 closure 2026-05-16). Pre-fix the "
                    "engine silently defaulted to events_per_minute=10.0, "
                    "miscalibrated for time-based corpora (TB v3p0 60s → "
                    "true 1.0 events/min, causing ~10x theta cost "
                    "understatement).\n"
                    "Migrate YAML to:\n"
                    "  zero_dte:\n"
                    "    enabled: true\n"
                    "    bin_seconds: 60  # for TB v3p0 60s corpora\n"
                    "    # OR\n"
                    "    events_per_minute: 1.0  # equivalent escape hatch\n"
                    "See lob-backtester/VALIDATION_FINDINGS_2026_05_14.md "
                    "FIND-NEW-01 for the full closure narrative."
                )
            transformer = ZeroDtePnLTransformer(
                built_zd_config, events_per_minute=events_per_minute
            )
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
            equity_curve=result.equity_curve
            if output_config.get("save_equity_curve", False)
            else None,
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

    def _expected_compatibility_fields(
        self,
        manifest_compat: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Derive consumer-side partial CompatibilityContract assertion from config.

        Phase II hardening SB-1 (2026-04-20) wired exactly ONE field —
        ``primary_horizon_idx`` — because it is the only contract field the
        backtester can *derive*. That stayed true, and it is why this consumer
        asserted essentially nothing for four months: the other ten fields
        (``label_strategy_hash``, ``feature_count``, ``window_size``, …) are
        trainer-side facts the backtester has no way to compute.

        2026-08-15 (R3): a field the consumer cannot DERIVE it can still
        DECLARE. The `signals.expect:` block lets the operator pin any of the
        11 contract fields explicitly::

            signals:
              dir: "…/signals/test"
              expect:
                label_strategy_hash: "7299e11a…"   # 64-hex, from the trainer
                feature_count: 98

        The governing principle: **if the config states an expectation about
        the signals it consumes, that expectation is ASSERTED, not assumed.**
        Nothing is asserted by default — a field absent from both the strategy
        config and `signals.expect:` is still trusted via the manifest's own
        producer fingerprint, exactly as before.

        Args:
            manifest_compat: The manifest's ``compatibility`` block, when
                already loaded. Used only for per-field grandfathering (see
                below). ``None`` disables that filtering — the historical
                no-arg call shape is preserved.

        Returns:
            Non-empty dict of field→expected value, or ``None`` when nothing
            is assertable. (``SignalManifest.validate`` rejects an empty dict
            as a caller-side logic error, so ``None`` is the correct "no
            assertions" signal.)

        Raises:
            ValueError: ``signals.expect:`` names a key that is not a
                CompatibilityContract field (typo defence — fail loud per
                hft-rules §5), or an explicit declaration contradicts the
                value derived from the strategy config.
        """
        strategy_config = self.config.get("strategy", {})
        strategy_type = strategy_config.get("type", "regression")
        expected: Dict[str, Any] = {}

        # --- (1) DERIVED: the one field the backtester computes itself. ---
        # RegressionStrategy knows primary_horizon_idx; only assert when the
        # user explicitly set it in config (not accepting the class default).
        if strategy_type == "regression" and "primary_horizon_idx" in strategy_config:
            expected["primary_horizon_idx"] = strategy_config["primary_horizon_idx"]

        # --- (2) DECLARED: `signals.expect:` operator assertions. ---
        signals_block = self.config.get("signals", {}) or {}
        declared = signals_block.get("expect") or {}
        if declared:
            unknown = set(declared) - _CONTRACT_FIELD_NAMES
            if unknown:
                raise ValueError(
                    f"`signals.expect:` names non-contract field(s) "
                    f"{sorted(unknown)!r}. Valid fields: "
                    f"{sorted(_CONTRACT_FIELD_NAMES)!r}. This is a config bug — "
                    f"an unrecognised key would otherwise assert nothing at all."
                )
            # A declaration that contradicts the derived value is ambiguous:
            # we cannot know which the operator meant. Fail rather than pick.
            for key, declared_val in declared.items():
                if key in expected and expected[key] != declared_val:
                    raise ValueError(
                        f"Config contradicts itself on `{key}`: "
                        f"strategy.{key}={expected[key]!r} but "
                        f"signals.expect.{key}={declared_val!r}. "
                        f"Remove one."
                    )
            expected.update(declared)

        # --- (3) GRANDFATHER ON ABSENCE, FAIL ON DISAGREEMENT. ---
        # Three contract fields are optional on the producer side
        # (`_compatibility_from_dict` reads calibration_method /
        # primary_horizon_idx / horizons with .get(), so they arrive as None),
        # and `calibration_method` is None on 8 of the 9 signal dirs currently
        # on disk. Upstream compares with `!=`, so None-vs-expected reads as a
        # MISMATCH and raises — i.e. asserting such a field would hard-fail a
        # historical export for a fact its producer never claimed.
        #
        # ABSENCE is not disagreement: the producer said nothing, so there is
        # nothing to contradict. We drop those expectations here and WARN that
        # the check did not run. A field PRESENT on both sides and differing is
        # left in place and hard-errors upstream — that is a real version skew.
        if manifest_compat is not None and expected:
            ungrandfathered = {
                k: v for k, v in expected.items() if manifest_compat.get(k) is not None
            }
            skipped = sorted(set(expected) - set(ungrandfathered))
            if skipped:
                warnings.warn(
                    f"Signal manifest does not declare {skipped!r}; those "
                    f"expectations were NOT verified (grandfathered — the "
                    f"producer predates the field). Fields still checked: "
                    f"{sorted(ungrandfathered)!r}. Re-export signals with a "
                    f"current trainer to enable the full check.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            expected = ungrandfathered

        # Future-extensibility: if other strategies gain shape-determining
        # knowledge (e.g., hybrid strategy explicitly declares horizons), add
        # field extraction here. Kept narrow for now to minimize false-positive
        # skew detections on legitimate runs.

        return expected if expected else None

    def _build_strategy(
        self,
        data: BacktestData,
        strategy_type: str,
        params: dict,
    ):
        """Build strategy from type + params. Reuses existing classes.

        FIND-070 closure (2026-05-14): adds two diagnostic gates:

        1. ``RuntimeWarning`` on unknown keys in the ``strategy:`` block per
           hft-rules §8 (mirrors hft-ops Phase 7.5 R5 idiom).
        2. ``ValueError`` (fail-loud per hft-rules §5) when a readability
           strategy is requested but ``min_agreement`` / ``min_confidence``
           are declared under ``backtest:`` instead of ``strategy:`` — the
           pre-FIND-070 silent-drop class. Error message embeds a concrete
           migration hint pointing at the correct YAML schema.
        """
        # FIND-070 Step 1: Warn on unknown strategy-block keys (typo defence).
        # Only fires for KNOWN strategy types; unknown types fall through to the
        # bottom-of-method ValueError (more actionable) so we suppress the
        # generic WARN in that case to keep the operator-facing diagnostic clean.
        strategy_block = self.config.get("strategy", {})
        if isinstance(strategy_block, dict) and strategy_type in _STRATEGY_KEY_SETS:
            known_for_type = _STRATEGY_KEY_SETS[strategy_type]
            _warn_unknown_yaml_keys("strategy", strategy_block, known_for_type)

        # FIND-070 Step 2: Fail-loud detection of wrong-block placement of
        # readability gate values. Pre-FIND-070, the prod YAMLs declared
        # `backtest.min_agreement` + `backtest.min_confidence`; runner silently
        # used readability defaults (0.667 / 0.65 per readability.py:54).
        # Per hft-rules §5 fail-fast with a precise migration error.
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
            backtest_block = self.config.get("backtest", {})
            wrong_block_keys: List[str] = []
            if isinstance(backtest_block, dict):
                if (
                    "min_agreement" not in params
                    and backtest_block.get("min_agreement") is not None
                ):
                    wrong_block_keys.append("min_agreement")
                if (
                    "min_confidence" not in params
                    and backtest_block.get("min_confidence") is not None
                ):
                    wrong_block_keys.append("min_confidence")
            if wrong_block_keys:
                quoted = ", ".join(repr(k) for k in wrong_block_keys)
                raise ValueError(
                    f"FIND-070: readability gate parameter(s) [{quoted}] found "
                    f"under `backtest:` block, but ExperimentRunner reads these "
                    f"from `strategy:` block. Pre-fix this silently used the "
                    f"readability defaults (0.667 / 0.65 per readability.py:54 "
                    f"P5 FIX) instead of the YAML's declared values.\n"
                    f"Migrate to:\n"
                    f"  strategy:\n"
                    f"    type: readability\n"
                    f"    min_agreement: <value>\n"
                    f"    min_confidence: <value>\n"
                    f"and remove the same keys from the `backtest:` block. See "
                    f"lob-backtester/CLAUDE.md FIND-070 closure 2026-05-14."
                )
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
        """Build holding policy from config.

        FIND-070 closure (2026-05-14): emits ``RuntimeWarning`` on unknown
        ``holding:`` block keys per hft-rules §8 (mirrors hft-ops Phase 7.5 R5
        idiom).
        """
        holding_cfg = self.config.get("holding", {})
        if isinstance(holding_cfg, dict):
            _warn_unknown_yaml_keys("holding", holding_cfg, _KNOWN_HOLDING_KEYS)
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
        """Build ``BacktestConfig`` from experiment config.

        FIND-070 closure (2026-05-14): emits ``RuntimeWarning`` on unknown
        ``backtest:`` block keys per hft-rules §8 (mirrors hft-ops Phase 7.5
        R5 idiom at commit ``3dd3ccb``).
        """
        bt = self.config.get("backtest", {})
        if isinstance(bt, dict):
            _warn_unknown_yaml_keys("backtest", bt, _KNOWN_BACKTEST_KEYS)
        exchange = bt.get("exchange", "XNAS")

        # #PY-263 (2026-05-30): thread the cadence-bearing zero_dte (built by
        # _build_zero_dte_config from the YAML's zero_dte.bin_seconds) into the
        # metrics BacktestConfig so resolved_periods_per_day derives
        # 23400/bin_seconds (390 at 60s) instead of the legacy 1000.0 fallback —
        # closing the same #PY-263 silent-Sharpe-inflation class V1 closed for
        # run_regression_backtest.py, here on the ExperimentRunner path. The
        # engine reads config.zero_dte ONLY via resolved_periods_per_day /
        # annualization_factor / to_dict (verified: vectorized.py never branches
        # the equity result on zero_dte.enabled — the 0DTE transform is post-hoc
        # via a SEPARATE ZeroDtePnLTransformer), so this is annualization-only;
        # the holding/theta/cost P&L is unchanged. Mutex (config.py:571-584): a
        # YAML setting BOTH backtest.periods_per_day AND zero_dte.bin_seconds
        # fail-louds (correct per hft-rules §5; no current config/test sets both).
        # NOTE: this now calls _build_zero_dte_config() unconditionally, so a
        # config with an *ambiguous* zero_dte block (top-level AND nested) now
        # fail-louds here rather than being silently ignored — more correct per §5.
        return BacktestConfig(
            initial_capital=bt.get("initial_capital", 100_000.0),
            position_size=bt.get("position_size", 0.1),
            allow_short=bt.get("allow_short", False),
            costs=CostConfig.for_exchange(exchange),
            zero_dte=self._build_zero_dte_config(),
            trading_days_per_year=bt.get("trading_days_per_year", 252.0),
            # #PY-263 (2026-05-21): default None (was 1000.0) enables mode-aware
            # dispatch via ``BacktestConfig.resolved_periods_per_day``. Explicit
            # YAML value preserves legacy override; absence triggers derivation
            # from ``zero_dte.bin_seconds`` or DeprecationWarning fallback.
            periods_per_day=bt.get("periods_per_day"),
        )

    def _build_zero_dte_config(self) -> ZeroDteConfig:
        """Build ZeroDteConfig from experiment config.

        #PY-226 (2026-05-14): accepts `zero_dte:` block at top-level OR nested under
        `backtest:` to match production readability YAMLs (nvda_readability_first_xnas
        + _arcx) which nest the block. Pre-#PY-226 reader-side path-mismatch silently
        dropped the YAML's `zero_dte:` block + fell through to all defaults.

        Sub-structure: per BacktestConfig.from_dict at config.py:405-415, opra-cost
        fields (commission_per_contract / implied_vol / entry_minutes_before_close)
        live under `zd.opra_costs:` (nested) in real YAMLs. This reader supports
        BOTH locations (nested-then-top-level fallback) for back-compat with
        existing tests that put fields at `zd` top-level.

        Fail-loud per hft-rules §8 on both-defined ambiguities (top-level AND nested
        `zero_dte:` block) — closes silent-drop class sister to FIND-070.
        """
        top_zd = self.config.get("zero_dte")
        backtest_block = self.config.get("backtest", {})
        nested_zd = backtest_block.get("zero_dte") if isinstance(backtest_block, dict) else None

        if top_zd and nested_zd:
            raise ValueError(
                "zero_dte: defined BOTH at top-level AND nested under backtest:. "
                "Choose ONE location. Recommended: nested under backtest: "
                "(matches lob-backtester/configs/nvda_readability_first_*.yaml)."
            )

        # Pick whichever is defined; fall through to {} if neither (back-compat
        # for default-disabled zero_dte test fixtures + ExperimentRunner callers
        # that don't set the block at all).
        zd: Dict[str, Any] = nested_zd if nested_zd else (top_zd or {})
        if not isinstance(zd, dict):
            zd = {}

        # Sub-structure: opra_costs nested block (per BacktestConfig.from_dict
        # pattern at config.py:405-415). Fields can ALSO appear at zd top-level
        # (legacy test fixtures); read nested first, fall through to top-level.
        opra_block = zd.get("opra_costs", {})
        if not isinstance(opra_block, dict):
            opra_block = {}

        def _opra_field(field: str, default):
            nested_val = opra_block.get(field)
            top_val = zd.get(field)
            if nested_val is not None and top_val is not None:
                raise ValueError(
                    f"zero_dte: field '{field}' defined BOTH at zd top-level AND "
                    f"nested under opra_costs:. Choose ONE location. "
                    f"Recommended: nested under opra_costs:."
                )
            if nested_val is not None:
                return nested_val
            if top_val is not None:
                return top_val
            return default

        # HF-1 closure (2026-05-16 LATE; Bundle 1 hygiene post Option B Path B'):
        # mode-aware IV default mirroring BacktestConfig.from_dict pattern at
        # config.py:566. Reads delta from zd block (top-level OR nested per
        # legacy fixtures) to detect regime:
        #   delta >= 0.90 → Deep ITM (IV=0.25 per #PY-273 OPRA empirical median)
        #   else → ATM (IV=0.40 preserved for ATM-regime back-compat)
        # Operator-explicit YAML override wins via _opra_field nested-or-top-
        # level dispatch (mutex enforced via raise on both-defined).
        _delta = zd.get("delta", 0.50)
        _iv_default = 0.25 if _delta >= 0.90 else 0.40
        return ZeroDteConfig(
            enabled=True,
            delta=_delta,
            opra_costs=OpraCalibratedCosts(
                commission_per_contract=_opra_field("commission_per_contract", 0.70),
                implied_vol=_opra_field("implied_vol", _iv_default),
                entry_minutes_before_close=_opra_field("entry_minutes_before_close", 120.0),
            ),
            contracts_per_trade=zd.get("contracts_per_trade", 1),
            # FIND-NEW-01 closure (2026-05-16): pass sampling-cadence YAML
            # fields through to ZeroDteConfig. ZeroDteConfig.__post_init__
            # validates mutex; ZeroDteConfig.resolved_events_per_minute
            # derives the effective value (None if neither set → caller
            # must supply at transformer-construction time).
            events_per_minute=zd.get("events_per_minute"),
            bin_seconds=zd.get("bin_seconds"),
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
