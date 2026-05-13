#!/usr/bin/env python3
"""
Regression backtest runner with IBKR-calibrated costs.

Uses continuous bps return predictions from a regression model
to generate trading signals. Tests at multiple breakeven thresholds:
  - Deep ITM: ~1.4 bps breakeven
  - ITM: ~3.0 bps breakeven
  - ATM: ~4.7 bps breakeven

Usage:
    python scripts/run_regression_backtest.py \
        --signals ../lob-model-trainer/outputs/experiments/nvda_tlob_128feat_regression_h10/signals/test/ \
        --name tlob_regression_h10 --exchange XNAS
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# R-16c F1 (2026-05-12): persist per-trade option P&L array for downstream
# pooled-bootstrap statistical analysis per X2 pre-impl design gate. Uses
# hft_contracts.atomic_io.atomic_write_npy SSoT (Class A per CLAUDE.md) to
# match the atomic-write discipline shipped in #PY-73 closure (2026-05-11).
# Placed with third-party imports (above sys.path.insert) since hft_contracts
# is a pip-installed sibling package, NOT a path-shim'd local sibling.
from hft_contracts.atomic_io import atomic_write_npy

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lobbacktest.config import BacktestConfig, CostConfig, ZeroDteConfig, OpraCalibratedCosts
from lobbacktest.engine.vectorized import BacktestData, VectorizedEngine
from lobbacktest.engine.zero_dte import ZeroDtePnLTransformer
from lobbacktest.strategies.regression import RegressionStrategy, RegressionStrategyConfig
from lobbacktest.strategies.holding import create_holding_policy
from lobbacktest.metrics import (
    SharpeRatio, SortinoRatio, MaxDrawdown, CalmarRatio,
    TotalReturn, WinRate, ProfitFactor, Expectancy,
)


def run_one_backtest(
    data, prices, config, strategy_config, holding_policy,
    zero_dte_config, label, verbose=True, output_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
):
    """Run a single backtest with given strategy config and return results.

    Phase R-16c F1 (2026-05-12): when ``output_dir`` + ``run_name`` are supplied
    AND ``zero_dte_config.enabled`` AND ``option_result.n_trades > 0``, persists
    the per-trade ``option_trade_pnls`` array atomically to
    ``output_dir / f"{run_name}__option_trade_pnls__{label}.npy"`` for downstream
    pooled-bootstrap statistical analysis. Backwards-compatible: omitting
    ``output_dir``/``run_name`` preserves pre-R-16c behavior (no .npy emission).
    """
    strategy = RegressionStrategy(
        predicted_returns=data.predicted_returns,
        spreads=data.spreads,
        prices=data.prices,
        config=strategy_config,
        holding_policy=holding_policy,
    )

    engine = VectorizedEngine(config)
    tdy = config.trading_days_per_year
    ppd = config.periods_per_day
    all_metrics = [
        SharpeRatio(trading_days_per_year=tdy, periods_per_day=ppd),
        SortinoRatio(trading_days_per_year=tdy, periods_per_day=ppd),
        MaxDrawdown(),
        CalmarRatio(trading_days_per_year=tdy, periods_per_day=ppd),
        TotalReturn(), WinRate(), ProfitFactor(), Expectancy(),
    ]
    # Pre-generate signals to capture strategy metadata (n_entries, avg_hold_events)
    signal_output = strategy.generate_signals(data.prices)
    strategy_meta = signal_output.metadata

    result = engine.run(data, strategy, metrics=all_metrics)

    summary = {
        "label": label,
        "min_return_bps": strategy_config.min_return_bps,
        "max_spread_bps": strategy_config.max_spread_bps,
        "holding_policy": holding_policy.policy_name,
        "strategy_name": strategy.name,
    }

    for k, v in result.metrics.items():
        summary[k] = round(v, 4) if isinstance(v, float) else v

    summary["n_entries"] = strategy_meta.get("n_entries", result.total_trades // 2)
    summary["trade_rate"] = round(strategy_meta.get("trade_rate", 0), 4)
    summary["avg_hold_events"] = strategy_meta.get("avg_hold_events", 0)

    if verbose:
        print(f"\n  --- {label} ---")
        print(f"  Strategy: {strategy.name}")
        print(f"  Trades: {summary['n_entries']}, Rate: {summary['trade_rate']:.3f}")
        print(f"  Avg hold: {summary['avg_hold_events']:.1f} events")
        # 2026-05-05 P0 fix: metric keys are PascalCase (class names, per
        # vectorized.py:646-651 `_compute_metrics` returns {metric.name: value}).
        # Pre-fix this loop used lowercase_snake keys (`total_return`, `win_rate`, etc.)
        # → silently dropped EVERY metric (the `if k in summary` guard always False).
        # All R9-R14 backtests printed with `--no-zero-dte` AND with `--deep-itm`
        # showed empty inline metrics + 0.0000 in summary table per the same key-case
        # bug. Fix: use canonical PascalCase keys.
        for k in ["TotalReturn", "SharpeRatio", "SortinoRatio", "MaxDrawdown",
                   "WinRate", "ProfitFactor", "Expectancy"]:
            if k in summary:
                print(f"  {k}: {summary[k]:.4f}")

    if zero_dte_config.enabled:
        transformer = ZeroDtePnLTransformer(zero_dte_config)
        option_result = transformer.transform(result)
        summary["option_final_equity"] = round(option_result.option_final_equity, 2)
        summary["option_return_pct"] = round(option_result.option_total_return * 100, 2)
        summary["option_n_trades"] = option_result.n_trades
        # 2026-05-05 P0 fix: persist option_win_rate + option_avg_pnl into summary
        # (pre-fix these only printed inline; downstream JSON consumers + summary
        # table couldn't access them). Conditional on n_trades > 0 because empty
        # trade lists would produce NaN means / undefined win rates.
        if option_result.n_trades > 0:
            summary["option_win_rate"] = round(option_result.option_win_rate, 4)
            summary["option_avg_pnl"] = round(float(option_result.option_trade_pnls.mean()), 4)

            # Phase R-16c F1 (2026-05-12): atomic dump of per-trade option pnls.
            # Required for R-16c pooled-per-trade bootstrap CI (X2 pre-registered
            # analysis primitive). Backwards-compatible: only fires when caller
            # supplies output_dir + run_name. Truthy `run_name` guard rejects
            # empty string per pre-commit code-reviewer MICRO-FIX 2 (would
            # produce filenames with double-underscore prefix on empty string).
            # Filename convention parallel to `output_dir / f"{run_name}.json"`.
            if output_dir is not None and run_name:
                pnls_path = output_dir / f"{run_name}__option_trade_pnls__{label}.npy"
                atomic_write_npy(pnls_path, option_result.option_trade_pnls)
                summary["option_trade_pnls_path"] = pnls_path.name
        if verbose:
            print(f"  --- 0DTE Option P&L ---")
            print(f"  Final equity: ${option_result.option_final_equity:,.2f}")
            print(f"  Return: {option_result.option_total_return:.2%}")
            print(f"  Trades: {option_result.n_trades}")
            if option_result.n_trades > 0:
                print(f"  Win rate: {option_result.option_win_rate:.2%}")
                print(f"  Avg P&L/trade: ${float(option_result.option_trade_pnls.mean()):.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Regression Backtest Runner")
    parser.add_argument("--signals", type=str, required=True)
    parser.add_argument("--name", type=str, default="regression_backtest")
    parser.add_argument("--exchange", type=str, default="XNAS", choices=["XNAS", "ARCX"])
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--position-size", type=float, default=0.1)
    parser.add_argument("--max-spread-bps", type=float, default=1.05)

    parser.add_argument("--hold-events", type=int, default=10)

    parser.add_argument("--zero-dte", action="store_true", default=True)
    parser.add_argument("--no-zero-dte", dest="zero_dte", action="store_false")
    parser.add_argument("--commission", type=float, default=0.70)
    parser.add_argument("--implied-vol", type=float, default=0.40)
    parser.add_argument("--entry-minutes-before-close", type=float, default=120.0)
    parser.add_argument("--delta", type=float, default=0.50,
                        help="Option delta (0.50=ATM, 0.95=deep ITM)")
    parser.add_argument("--deep-itm", action="store_true", default=False,
                        help="Use deep ITM costs (delta=0.95, spread=$0.005)")

    parser.add_argument("--output-dir", type=str, default="outputs/backtests/")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to hft-ops experiment manifest YAML. When supplied, "
                             "writes a ledger-linkage record at "
                             "<manifest_parent>/ledger/runs/<exp_name>_backtest_<args.name>.json "
                             "for cross-tool traceability (Phase R-17 #PY-129 closure). "
                             "Backward-compatible: omitting --manifest preserves pre-Phase-R-17 behavior.")

    # Phase V.A.5 (2026-04-21): Phase II CompatibilityContract version-skew
    # detection for standalone-script callers. Closes the gap left by SB-1
    # (which wired the orchestrator-driven path via ExperimentRunner.
    # _expected_compatibility_fields but left script callers bypassing the
    # partial-assertion API). Optional — the default (None) leaves
    # validate=True tamper detection active but skips the
    # primary_horizon assertion, matching pre-V.A.5 behavior for legacy
    # scripts that don't care about version-skew.
    parser.add_argument(
        "--primary-horizon-idx",
        type=int,
        default=None,
        help=(
            "Phase II SB-1 partial-assertion check: if supplied, verifies "
            "signal_metadata.compatibility.primary_horizon_idx matches the "
            "given value. Skipped when omitted (backward-compatible)."
        ),
    )
    args = parser.parse_args()

    signal_dir = Path(args.signals)
    if not signal_dir.exists():
        print(f"ERROR: Signal directory not found: {signal_dir}")
        sys.exit(1)

    print("=" * 70)
    print("  REGRESSION BACKTEST")
    print("=" * 70)
    print(f"  Signals: {signal_dir}")
    print(f"  Exchange: {args.exchange}")

    metadata_path = signal_dir / "signal_metadata.json"
    signal_metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            signal_metadata = json.load(f)
        print(f"  Model: {signal_metadata.get('model_type', '?')}")
        print(f"  Samples: {signal_metadata.get('total_samples', '?'):,}")
        m = signal_metadata.get("metrics", {})
        print(f"  Model R²={m.get('r2', '?')}, IC={m.get('ic', '?')}")

    # Phase B Step 1 ship-blocker fix (R-16d horizon-axis sweep prereq, 2026-05-13):
    # Auto-discover primary_horizon_idx from signal_metadata.compatibility when
    # --primary-horizon-idx flag was NOT explicitly provided. Enables horizon-axis
    # sweeps (e.g., R-16d {H10, H60}) to author manifests without per-axis-value
    # extra_args overrides (an hft-ops manifest schema limitation). Each grid
    # point's signal_metadata.json already carries the correct primary_horizon_idx
    # from the trainer's signal_export step. Explicit flag still wins; auto-discover
    # is fallback only — preserves backward-compat with R9-R14 + R-16a + R-16c
    # invocations that pass the flag explicitly.
    effective_primary_horizon_idx = args.primary_horizon_idx
    discovery_source = "explicit"
    if effective_primary_horizon_idx is None:
        compat = signal_metadata.get("compatibility") or {}
        discovered = compat.get("primary_horizon_idx")
        if discovered is not None:
            effective_primary_horizon_idx = int(discovered)
            discovery_source = "auto-discovered"

    expected_fields = (
        {"primary_horizon_idx": effective_primary_horizon_idx}
        if effective_primary_horizon_idx is not None
        else None
    )
    data = BacktestData.from_signal_dir(
        str(signal_dir),
        expected_fields=expected_fields,
    )
    if expected_fields is not None:
        print(
            f"  Phase II check: primary_horizon_idx={effective_primary_horizon_idx} "
            f"({discovery_source}) ✓"
        )
    n = len(data)
    print(f"  Loaded {n:,} samples")

    pred = data.predicted_returns
    print(f"  Predictions: mean={pred.mean():+.3f}, std={pred.std():.3f}, "
          f"range=[{pred.min():.1f}, {pred.max():.1f}]")

    spreads_data = data.spreads
    if spreads_data is not None:
        print(f"  Spreads: mean={spreads_data.mean():.3f}, median={np.median(spreads_data):.3f} bps")

    costs = CostConfig.for_exchange(args.exchange)
    if args.deep_itm:
        opra_costs = OpraCalibratedCosts.deep_itm()
        opra_costs.commission_per_contract = args.commission
        delta = 0.95
        print(f"  Mode: DEEP ITM (delta={delta}, half_spread=$0.005)")
    else:
        opra_costs = OpraCalibratedCosts(
            commission_per_contract=args.commission,
            implied_vol=args.implied_vol,
            entry_minutes_before_close=args.entry_minutes_before_close,
        )
        delta = args.delta
        print(f"  Mode: ATM (delta={delta}, half_spread=${opra_costs.atm_call_half_spread})")
    zero_dte_config = ZeroDteConfig(
        enabled=args.zero_dte,
        delta=delta,
        opra_costs=opra_costs,
        contracts_per_trade=1,
    )
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        position_size=args.position_size,
        costs=costs,
        zero_dte=zero_dte_config,
    )

    holding_config = {"type": "horizon_aligned", "hold_events": args.hold_events}
    holding_policy = create_holding_policy(holding_config)

    thresholds = [
        ("deep_itm_1.4bps", 1.4),
        ("itm_2bps", 2.0),
        ("itm_3bps", 3.0),
        ("atm_5bps", 5.0),
        ("high_conv_8bps", 8.0),
        ("very_high_10bps", 10.0),
        ("ultra_conv_15bps", 15.0),
        ("max_conv_20bps", 20.0),
    ]

    # Phase R-16c F1 (2026-05-12): set up output_dir BEFORE the per-threshold
    # loop so each iteration can dump option_trade_pnls.npy atomically.
    # Previously output_dir.mkdir() ran AFTER the loop (line ~270), which
    # meant the dump couldn't happen inside run_one_backtest. Lifting this
    # 3-line block ~12 lines earlier is idempotent + safe.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for label, min_ret in thresholds:
        strategy_config = RegressionStrategyConfig(
            min_return_bps=min_ret,
            max_spread_bps=args.max_spread_bps,
            primary_horizon_idx=0,
            cooldown_events=0,
        )
        result = run_one_backtest(
            data, data.prices, config, strategy_config, holding_policy,
            zero_dte_config, label,
            output_dir=output_dir, run_name=args.name,
        )
        all_results.append(result)

    print(f"\n{'=' * 90}")
    print(f"  SUMMARY: {args.name}")
    print(f"{'=' * 90}")
    # 2026-05-05 P0 fix: split table into two metric groups for clarity:
    # SPOT-leg metrics (stock equity, computed by VectorizedEngine via all_metrics
    #   list — WinRate over trade_pnls is the equity-curve win rate)
    # OPTION-leg metrics (0DTE option P&L from ZeroDtePnLTransformer — separate
    #   population because option leverage + theta + spread vs stock returns)
    # Pre-fix these were conflated in one row → readers wrongly compared.
    print(f"  {'Threshold':<20} {'Trades':>7} {'Rate':>7} | "
          f"{'SpotWR%':>7} {'Sharpe':>8} {'SpotRet%':>9} | "
          f"{'OptWR%':>7} {'OptRet':>8}")
    print(f"  {'-' * 20} {'-' * 7} {'-' * 7} + "
          f"{'-' * 7} {'-' * 8} {'-' * 9} + "
          f"{'-' * 7} {'-' * 8}")

    for r in all_results:
        # PascalCase keys per vectorized.py:646-651 _compute_metrics dict-keying.
        # Pre-fix used lowercase_snake → silent zero across all R9-R14 BACKTEST_INDEX
        # entries. Use .get(..., 0) defensively for flexibility on metric availability.
        spot_wr = r.get("WinRate", 0) * 100  # WinRate is fraction; convert to %
        spot_sharpe = r.get("SharpeRatio", 0)
        spot_total = r.get("TotalReturn", 0) * 100  # TotalReturn is fraction; convert
        opt_ret = r.get("option_return_pct", 0)
        opt_wr = r.get("option_win_rate", 0) * 100  # option_win_rate is fraction
        print(f"  {r['label']:<20} {r['n_entries']:>7} {r['trade_rate']:>7.3f} | "
              f"{spot_wr:>6.2f}% {spot_sharpe:>+8.2f} {spot_total:>+8.2f}% | "
              f"{opt_wr:>6.2f}% {opt_ret:>+7.2f}%")

    output_file = output_dir / f"{args.name}.json"
    with open(output_file, "w") as f:
        json.dump({
            "name": args.name,
            "exchange": args.exchange,
            "signal_dir": str(signal_dir),
            "signal_metadata": signal_metadata,
            "holding_policy": holding_policy.policy_name,
            "zero_dte_enabled": args.zero_dte,
            "results": all_results,
        }, f, indent=2)
    print(f"\n  Saved results to {output_file}")

    # Phase R-17 F1 (2026-05-11): #PY-129 producer-side ledger linkage.
    # Mirrors run_readability_backtest.py:326-353 with regression-script
    # adaptations per H1 agent ground-truth review:
    #   - Regression script lacks `run_id` variable (no BacktestRegistry) →
    #     uses args.name as the natural ledger key.
    #   - Regression script's `all_results` is a list of 8 threshold dicts
    #     (NOT a single BacktestResult aggregate) → emits hybrid record:
    #     top-level summary from best-by-OptRet + full all_thresholds breakdown.
    #   - Adds `option_return_pct` + `option_win_rate` (R-16a execution-aligned
    #     metrics; rationale per Agent B 2026-05-11 caveat: "execution-aligned
    #     cost gate" is the PRIMARY scientific signal).
    #   - Narrower exception class set per hft-rules §8 — no silent swallow.
    #   - Inline float() conversions on numpy scalars to avoid default=str
    #     silent-coerce hazard (hft-rules §8 violation in readability template).
    if args.manifest:
        try:
            import yaml as _yaml
            manifest_path = Path(args.manifest)
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest_data = _yaml.safe_load(f)
                manifest_exp_name = manifest_data.get("experiment", {}).get("name", "unknown")
                ledger_path = manifest_path.parent.parent / "ledger" / "runs"
                ledger_path.mkdir(parents=True, exist_ok=True)

                # Pick best across 8 thresholds for top-level summary.
                # Phase R-17 v2 mid-impl refinement (Q2): when --zero-dte enabled,
                # use OptRet (R-16a's PRIMARY metric); else use TotalReturn (spot).
                # Closes silent-coerce hazard where --no-zero-dte yields absent
                # option_return_pct keys, leaving best = arbitrary first-tie.
                if args.zero_dte and any("option_return_pct" in r for r in all_results):
                    best = max(
                        all_results,
                        key=lambda r: float(r.get("option_return_pct", float("-inf"))),
                    )
                else:
                    best = max(
                        all_results,
                        key=lambda r: float(r.get("TotalReturn", float("-inf"))),
                    )

                record = {
                    "experiment_name": manifest_exp_name,
                    "stage": "backtesting",
                    "status": "completed",
                    "run_id": args.name,  # args.name as ledger key (no BacktestRegistry)
                    # Top-level summary (best-of by OptRet)
                    "best_threshold": str(best.get("label", "unknown")),
                    "best_option_return_pct": float(best.get("option_return_pct", 0.0)),
                    "best_option_win_rate": float(best.get("option_win_rate", 0.0)),
                    "best_n_entries": int(best.get("n_entries", 0)),
                    "best_total_return": float(best.get("TotalReturn", 0.0)),
                    "best_win_rate": float(best.get("WinRate", 0.0)),
                    "best_sharpe_ratio": float(best.get("SharpeRatio", 0.0)),
                    # Full per-threshold breakdown (R-16a hypothesis-aligned)
                    "all_thresholds": [
                        {
                            "label": str(r.get("label", "unknown")),
                            "n_entries": int(r.get("n_entries", 0)),
                            "win_rate": float(r.get("WinRate", 0.0)),
                            "total_return": float(r.get("TotalReturn", 0.0)),
                            "option_return_pct": float(r.get("option_return_pct", 0.0)),
                            "option_win_rate": float(r.get("option_win_rate", 0.0)),
                        }
                        for r in all_results
                    ],
                    # Provenance + execution context
                    "holding_policy": holding_policy.policy_name,
                    "exchange": args.exchange,
                    "zero_dte_enabled": args.zero_dte,
                    "signal_dir": str(signal_dir),
                    "manifest": str(manifest_path),
                }
                record_path = ledger_path / f"{manifest_exp_name}_backtest_{args.name}.json"
                with open(record_path, "w") as f:
                    json.dump(record, f, indent=2)
                print(f"  Updated hft-ops ledger: {record_path}")
        except (FileNotFoundError, PermissionError, OSError, KeyError,
                AttributeError, TypeError, _yaml.YAMLError) as e:
            # Narrow exception set per hft-rules §8 — re-raise unexpected types.
            # Phase R-17 v2 mid-impl refinement (Q3): added AttributeError +
            # TypeError to handle realistic chains like
            # `manifest_data.get("experiment", {}).get(...)` when YAML has
            # `experiment: null` or non-dict root.
            print(f"  WARNING: Failed to update hft-ops ledger: {e}")


if __name__ == "__main__":
    main()
