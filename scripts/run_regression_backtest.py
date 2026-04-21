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

import numpy as np

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
    zero_dte_config, label, verbose=True,
):
    """Run a single backtest with given strategy config and return results."""
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
        for k in ["total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown",
                   "win_rate", "profit_factor", "expectancy"]:
            if k in summary:
                print(f"  {k}: {summary[k]:.4f}")

    if zero_dte_config.enabled:
        transformer = ZeroDtePnLTransformer(zero_dte_config)
        option_result = transformer.transform(result)
        summary["option_final_equity"] = round(option_result.option_final_equity, 2)
        summary["option_return_pct"] = round(option_result.option_total_return * 100, 2)
        summary["option_n_trades"] = option_result.n_trades
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

    expected_fields = (
        {"primary_horizon_idx": args.primary_horizon_idx}
        if args.primary_horizon_idx is not None
        else None
    )
    data = BacktestData.from_signal_dir(
        str(signal_dir),
        expected_fields=expected_fields,
    )
    if expected_fields is not None:
        print(f"  Phase II check: primary_horizon_idx={args.primary_horizon_idx} ✓")
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
        )
        all_results.append(result)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {args.name}")
    print(f"{'=' * 70}")
    print(f"  {'Threshold':<20} {'Trades':>7} {'Rate':>7} {'WinRate':>8} "
          f"{'Sharpe':>8} {'TotalRet':>9} {'OptRet':>8}")
    print(f"  {'-' * 20} {'-' * 7} {'-' * 7} {'-' * 8} "
          f"{'-' * 8} {'-' * 9} {'-' * 8}")

    for r in all_results:
        opt_ret = r.get("option_return_pct", 0)
        print(f"  {r['label']:<20} {r['n_entries']:>7} {r['trade_rate']:>7.3f} "
              f"{r.get('win_rate', 0):>8.4f} {r.get('sharpe_ratio', 0):>8.2f} "
              f"{r.get('total_return', 0):>9.4f} {opt_ret:>7.2f}%")

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


if __name__ == "__main__":
    main()
