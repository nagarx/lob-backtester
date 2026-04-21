#!/usr/bin/env python3
"""
End-to-end readability-first backtest runner with configurable holding policies.

Usage:
    # H10-aligned holding (default, recommended)
    python scripts/run_readability_backtest.py \\
        --signals ../lob-model-trainer/outputs/experiments/nvda_hmhp_40feat_h10/signals/test/ \\
        --name h10_hold_xnas --exchange XNAS

    # H60-aligned holding
    python scripts/run_readability_backtest.py \\
        --signals ... --name h60_hold --holding-type horizon_aligned --hold-events 60

    # Direction reversal
    python scripts/run_readability_backtest.py \\
        --signals ... --name reversal --holding-type direction_reversal --max-hold 60

    # Stop-loss / take-profit
    python scripts/run_readability_backtest.py \\
        --signals ... --name sltp --holding-type stop_loss_take_profit \\
        --stop-loss 10 --take-profit 20

    # No holding (original flickering behavior, for comparison)
    python scripts/run_readability_backtest.py \\
        --signals ... --name no_hold --holding-type horizon_aligned --hold-events 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lobbacktest.config import BacktestConfig, CostConfig, ZeroDteConfig, OpraCalibratedCosts
from lobbacktest.engine.vectorized import BacktestData, VectorizedEngine
from lobbacktest.engine.zero_dte import ZeroDtePnLTransformer
from lobbacktest.strategies.readability import ReadabilityStrategy, ReadabilityConfig
from lobbacktest.strategies.holding import create_holding_policy
from lobbacktest.metrics import (
    SharpeRatio, SortinoRatio, MaxDrawdown, CalmarRatio,
    TotalReturn, WinRate, ProfitFactor, Expectancy,
)
from lobbacktest.registry import BacktestRegistry


def build_holding_config(args) -> dict:
    """Build holding policy config dict from CLI args."""
    ht = args.holding_type

    if ht == "horizon_aligned":
        return {"type": "horizon_aligned", "hold_events": args.hold_events}
    elif ht == "direction_reversal":
        return {
            "type": "direction_reversal",
            "max_hold_events": args.max_hold,
            "require_gate": args.require_gate,
        }
    elif ht == "stop_loss_take_profit":
        return {
            "type": "stop_loss_take_profit",
            "stop_loss_bps": args.stop_loss,
            "take_profit_bps": args.take_profit,
            "max_hold_events": args.max_hold,
        }
    elif ht == "composite_horizon_sltp":
        return {
            "type": "composite",
            "mode": "any",
            "policies": [
                {"type": "horizon_aligned", "hold_events": args.hold_events},
                {
                    "type": "stop_loss_take_profit",
                    "stop_loss_bps": args.stop_loss,
                    "take_profit_bps": args.take_profit,
                    "max_hold_events": args.max_hold,
                },
            ],
        }
    else:
        raise ValueError(f"Unknown holding type: {ht}")


def main():
    parser = argparse.ArgumentParser(description="Readability-First Backtest Runner")
    parser.add_argument("--signals", type=str, required=True)
    parser.add_argument("--name", type=str, default="readability_backtest")
    parser.add_argument("--exchange", type=str, default="XNAS", choices=["XNAS", "ARCX"])
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--position-size", type=float, default=0.1)

    parser.add_argument("--min-agreement", type=float, default=1.0)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--max-spread-bps", type=float, default=1.05)

    parser.add_argument("--holding-type", type=str, default="horizon_aligned",
                        choices=["horizon_aligned", "direction_reversal",
                                 "stop_loss_take_profit", "composite_horizon_sltp"])
    parser.add_argument("--hold-events", type=int, default=10)
    parser.add_argument("--max-hold", type=int, default=60)
    parser.add_argument("--stop-loss", type=float, default=10.0)
    parser.add_argument("--take-profit", type=float, default=20.0)
    parser.add_argument("--require-gate", action="store_true", default=False)
    parser.add_argument("--cooldown", type=int, default=0)

    parser.add_argument("--zero-dte", action="store_true", default=True)
    parser.add_argument("--no-zero-dte", dest="zero_dte", action="store_false")
    parser.add_argument("--delta", type=float, default=0.50)
    parser.add_argument("--contracts", type=int, default=1,
                        help="Number of option contracts per trade")
    parser.add_argument("--commission", type=float, default=0.70,
                        help="IBKR all-in commission per contract (USD, from 318-fill median)")
    parser.add_argument("--implied-vol", type=float, default=0.40,
                        help="Annualized IV for BSM theta (OPRA GreeksTracker median)")
    parser.add_argument("--entry-minutes-before-close", type=float, default=120.0,
                        help="Minutes before close at entry (120 = 14:00 ET)")

    parser.add_argument("--output-dir", type=str, default="outputs/backtests/")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to hft-ops experiment manifest YAML")

    # Phase V.A.5 (2026-04-21): Phase II CompatibilityContract version-skew
    # detection for standalone-script callers. See run_regression_backtest.py
    # for the parallel wiring + rationale. Optional — default None preserves
    # pre-V.A.5 behavior (tamper detection only, no partial assertion).
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

    holding_config = build_holding_config(args)
    holding_policy = create_holding_policy(holding_config)

    print("=" * 60)
    print("  READABILITY-FIRST BACKTEST")
    print("=" * 60)
    print(f"  Signals: {signal_dir}")
    print(f"  Exchange: {args.exchange}")
    print(f"  Gates: agree>={args.min_agreement}, conf>{args.min_confidence}, "
          f"spread<={args.max_spread_bps}")
    print(f"  Holding: {holding_policy.policy_name}")
    print(f"  Cooldown: {args.cooldown} events")

    metadata_path = signal_dir / "signal_metadata.json"
    signal_metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            signal_metadata = json.load(f)
        print(f"  Model samples: {signal_metadata.get('total_samples', '?'):,}")

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

    costs = CostConfig.for_exchange(args.exchange)
    opra_costs = OpraCalibratedCosts(
        commission_per_contract=args.commission,
        implied_vol=args.implied_vol,
        entry_minutes_before_close=args.entry_minutes_before_close,
    )
    zero_dte_config = ZeroDteConfig(
        enabled=args.zero_dte,
        delta=args.delta,
        opra_costs=opra_costs,
        contracts_per_trade=args.contracts,
    )
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        position_size=args.position_size,
        costs=costs,
        zero_dte=zero_dte_config,
        min_agreement=args.min_agreement,
        min_confidence=args.min_confidence,
    )

    readability_config = ReadabilityConfig(
        min_agreement=args.min_agreement,
        min_confidence=args.min_confidence,
        max_spread_bps=args.max_spread_bps,
        require_directional=True,
        cooldown_events=args.cooldown,
    )

    strategy = ReadabilityStrategy(
        predictions=data.predictions,
        agreement_ratio=data.agreement_ratio,
        confirmation_score=data.confirmation_score,
        spreads=data.spreads,
        prices=data.prices,
        config=readability_config,
        holding_policy=holding_policy,
    )

    print(f"\n  Running backtest...")
    engine = VectorizedEngine(config)

    tdy = config.trading_days_per_year
    ppd = config.periods_per_day
    all_metrics = [
        SharpeRatio(trading_days_per_year=tdy, periods_per_day=ppd),
        SortinoRatio(trading_days_per_year=tdy, periods_per_day=ppd),
        MaxDrawdown(),
        CalmarRatio(trading_days_per_year=tdy, periods_per_day=ppd),
        TotalReturn(),
        WinRate(),
        ProfitFactor(),
        Expectancy(),
    ]

    result = engine.run(data, strategy, metrics=all_metrics)

    signal_output = strategy.generate_signals(data.prices)
    strat_meta = signal_output.metadata or {}

    print(f"\n{'='*60}")
    print(f"  EQUITY BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Holding policy: {holding_policy.policy_name}")
    print(f"  Total trades: {result.total_trades}")
    print(f"  Entries: {strat_meta.get('n_entries', '?')}")
    print(f"  Avg hold (events): {strat_meta.get('avg_hold_events', '?')}")
    print(f"  Total return: {result.total_return:.2%}")
    print(f"  Final equity: ${result.final_equity:,.2f}")
    print(f"  Max drawdown: {result.max_drawdown:.2%}")
    print(f"  Trade rate: {strat_meta.get('trade_rate', 0):.1%}")

    for key, value in result.metrics.items():
        if key not in ("total_return", "max_drawdown"):
            print(f"  {key}: {value:.4f}")

    if strat_meta.get("exit_reasons"):
        print(f"  Exit reasons: {strat_meta['exit_reasons']}")

    if data.labels is not None and data.predictions is not None:
        gate = (
            (data.agreement_ratio >= args.min_agreement) &
            (data.confirmation_score > args.min_confidence) &
            ((data.predictions == 0) | (data.predictions == 2))
        )
        if data.spreads is not None:
            gate &= data.spreads <= args.max_spread_bps
        gated_preds = data.predictions[gate]
        gated_labels = data.labels[gate]
        if len(gated_preds) > 0:
            dir_mask = np.isin(gated_preds, [0, 2]) & np.isin(gated_labels, [0, 2])
            if dir_mask.sum() > 0:
                dir_acc = (gated_preds[dir_mask] == gated_labels[dir_mask]).mean()
                print(f"\n  Gated directional accuracy: {dir_acc:.2%} ({dir_mask.sum():,} samples)")

    zero_dte_result = None
    if args.zero_dte and result.total_trades > 0:
        print(f"\n{'='*60}")
        print(f"  0DTE OPTION P&L TRANSFORMATION")
        print(f"{'='*60}")
        transformer = ZeroDtePnLTransformer(zero_dte_config)
        zero_dte_result = transformer.transform(result)
        print(zero_dte_result.summary())

    print(f"\n  Registering results...")
    registry = BacktestRegistry(args.output_dir)

    metrics_dict = {
        "total_trades": result.total_trades,
        "total_return": result.total_return,
        "final_equity": result.final_equity,
        "max_drawdown": result.max_drawdown,
        **result.metrics,
    }

    option_metrics = {}
    if zero_dte_result is not None:
        option_metrics = {
            "option_total_return": zero_dte_result.option_total_return,
            "option_final_equity": zero_dte_result.option_final_equity,
            "option_win_rate": zero_dte_result.option_win_rate,
            "avg_theta_cost": zero_dte_result.avg_theta_cost,
        }

    config_dict = config.to_dict()
    config_dict["holding_policy"] = holding_config
    config_dict["readability"] = {
        "min_agreement": args.min_agreement,
        "min_confidence": args.min_confidence,
        "max_spread_bps": args.max_spread_bps,
        "cooldown_events": args.cooldown,
    }

    run_id = registry.register(
        name=args.name,
        config_dict=config_dict,
        metrics=metrics_dict,
        signal_metadata=signal_metadata,
        equity_curve=result.equity_curve,
        option_metrics=option_metrics,
        strategy_metadata=strat_meta,
    )

    print(f"  Registered as: {run_id}")

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
                record = {
                    "experiment_name": manifest_exp_name,
                    "stage": "backtesting",
                    "status": "completed",
                    "run_id": run_id,
                    "holding_policy": holding_policy.policy_name,
                    "total_trades": result.total_trades,
                    "total_return": result.total_return,
                    "win_rate": result.metrics.get("WinRate", 0),
                    "max_drawdown": result.max_drawdown,
                    "manifest": str(manifest_path),
                }
                record_path = ledger_path / f"{manifest_exp_name}_backtest_{args.name}.json"
                with open(record_path, "w") as f:
                    json.dump(record, f, indent=2, default=str)
                print(f"  Updated hft-ops ledger: {record_path}")
        except Exception as e:
            print(f"  WARNING: Failed to update hft-ops ledger: {e}")

    print(f"\n{'='*60}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
