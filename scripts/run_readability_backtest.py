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

# Added 2026-05-15 R-19 cycle C5: atomic-write SSoT for FIND-090
# sister-site closure at L349 (hft-ops ledger linkage). Mirrors
# `run_regression_backtest.py` atomic_io pattern. Placed with
# third-party imports (above sys.path.insert) since hft_contracts is
# a pip-installed sibling package, NOT a path-shim'd local sibling.
from hft_contracts.atomic_io import atomic_write_json

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
    # #PY-305 closure (2026-05-17): sentinel-None mode-aware default. Pre-fix
    # `default=0.40` hardcoded silently inherited ATM IV when operator passed
    # `--delta 0.95` (Deep ITM) without `--implied-vol` → ~60-100% theta
    # overstatement → readability strategy wrongly rejected at cost-floor
    # gate. Mirrors `run_regression_backtest.py:191-211` pattern. Class-
    # coherent fold-in (Wave 1B + pre-impl Agent Z): same treatment for
    # `--entry-minutes-before-close` — also silently inherited stale defaults
    # post FIND-NEW-01 calibration cycle. Resolution at L184 below.
    parser.add_argument("--implied-vol", type=float, default=None,
                        help="Annualized IV for BSM theta. None → mode-aware "
                             "default: 0.25 if --delta >= 0.90 (Deep ITM) "
                             "else 0.40 (ATM). Explicit value overrides.")
    parser.add_argument("--entry-minutes-before-close", type=float, default=None,
                        help="Minutes before close at entry. None → 120.0 "
                             "(default mid-day entry; 14:00 ET).")

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

    # FIND-NEW-01 closure (2026-05-16): mutually-exclusive sampling-cadence
    # flag pair (sister of run_regression_backtest.py). Pre-fix the engine
    # default events_per_minute=10.0 silently miscalibrated TB v3p0 60s
    # backtests (true 1.0 events/min). Fail-loud requires explicit operator
    # calibration when --zero-dte is enabled (the default). See
    # run_regression_backtest.py for full closure narrative +
    # lob-backtester/VALIDATION_FINDINGS_2026_05_14.md FIND-NEW-01.
    cadence_group = parser.add_mutually_exclusive_group(required=False)
    cadence_group.add_argument(
        "--bin-seconds",
        type=float,
        default=None,
        help=(
            "Sampling cadence in seconds (time-based corpora). Derives "
            "events_per_minute = 60.0 / bin_seconds. TB v3p0 60s → 60. "
            "Required when --zero-dte enabled."
        ),
    )
    cadence_group.add_argument(
        "--events-per-minute",
        type=float,
        default=None,
        help=(
            "Events per minute (event-based corpora; escape hatch). "
            "Legacy R9-R14 event-based ~1000/day: pass 10.0. "
            "Required when --zero-dte enabled."
        ),
    )

    args = parser.parse_args()

    # #PY-305 closure (2026-05-17): mode-aware default resolution.
    # CLI defaults are None (sentinel for "use mode-derived default");
    # explicit operator value wins. ATM (delta < 0.90) → IV=0.40 (matches
    # OpraCalibratedCosts class default); Deep ITM (delta >= 0.90) → IV=0.25
    # (matches OpraCalibratedCosts.deep_itm() factory per #PY-273 closure).
    # Class-coherent with `run_regression_backtest.py:389-415` pattern.
    # Note: this script constructs OpraCalibratedCosts DIRECTLY at L260+
    # (no `--deep-itm` flag like the regression script); the mode-aware
    # default just sets the right IV before construction.
    _iv_override = " [CLI override]" if args.implied_vol is not None else ""
    if args.implied_vol is None:
        args.implied_vol = 0.25 if args.delta >= 0.90 else 0.40
        _iv_override = " [#PY-305 mode-aware default]"
    if args.entry_minutes_before_close is None:
        args.entry_minutes_before_close = 120.0

    # FIND-NEW-01 closure (2026-05-16): derive effective events_per_minute.
    # Only required when --zero-dte enabled.
    # MF-2 (LOW, mid-impl gate): explicit Optional[float] annotation matches
    # run_regression_backtest.py:268 — cross-script type-hint consistency.
    from typing import Optional as _Optional
    events_per_minute: _Optional[float] = None
    if args.zero_dte:
        if args.bin_seconds is None and args.events_per_minute is None:
            parser.error(
                "--bin-seconds OR --events-per-minute is required when "
                "--zero-dte is enabled (FIND-NEW-01 closure 2026-05-16). "
                "For TB v3p0 60s use --bin-seconds 60; for legacy event-based "
                "~1000/day use --events-per-minute 10.0. Pass --no-zero-dte "
                "to bypass."
            )
        if args.bin_seconds is not None:
            if args.bin_seconds <= 0:
                parser.error(f"--bin-seconds must be > 0, got {args.bin_seconds}")
            events_per_minute = 60.0 / args.bin_seconds
            cadence_source = f"--bin-seconds {args.bin_seconds}"
        else:
            if args.events_per_minute <= 0:
                parser.error(
                    f"--events-per-minute must be > 0, got {args.events_per_minute}"
                )
            events_per_minute = args.events_per_minute
            cadence_source = f"--events-per-minute {args.events_per_minute}"
        print(
            f"  Sampling cadence: {cadence_source} → "
            f"events_per_minute={events_per_minute:.4f}"
        )

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
    # #PY-305 closure (2026-05-17): operator-facing mode line. Shows the
    # resolved IV + entry-minutes-before-close + delta so operator can
    # verify mode-aware default fired correctly. `_iv_override` tag set
    # at L198+L201 distinguishes CLI explicit override vs mode-aware default.
    print(f"  Mode: delta={args.delta}, IV={args.implied_vol:.2f}{_iv_override}, "
          f"entry_min_to_close={args.entry_minutes_before_close:.0f}")

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
    # R1 / #PY-263 (2026-05-30): thread the time-based sampling cadence into the
    # config so ``BacktestConfig.resolved_periods_per_day`` derives the correct
    # sub-daily annualization (23400 / bin_seconds = 390 at 60s) instead of the
    # legacy 1000.0 fallback that silently inflates equity Sharpe/Sortino/Calmar
    # ~1.6x at 60s bins. Mirrors the V1 fix in run_regression_backtest.py:437-459
    # (V1 closed #PY-263 on the regression sister-script; this closes the
    # readability sister-script V1 left — same silent-Sharpe-inflation class).
    # ``args.bin_seconds`` is None on the --events-per-minute path (event-based
    # corpora stay on the documented fallback; ~1000/day ≈ correct there).
    # Mutex-safe: only bin_seconds is set on the config (NOT events_per_minute,
    # which is passed to the transformer separately below at its construction);
    # the transformer
    # takes events_per_minute explicitly and ignores config.bin_seconds, so this
    # changes annualization ONLY — never the holding/theta math.
    zero_dte_config = ZeroDteConfig(
        enabled=args.zero_dte,
        delta=args.delta,
        opra_costs=opra_costs,
        contracts_per_trade=args.contracts,
        bin_seconds=args.bin_seconds,
    )
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        position_size=args.position_size,
        costs=costs,
        zero_dte=zero_dte_config,
        min_agreement=args.min_agreement,
        min_confidence=args.min_confidence,
    )
    # R1 / #PY-263 (2026-05-30): surface the resolved annualization on the
    # time-based path so operators can confirm the sub-daily fix is active (390
    # at 60s, not the legacy 1000.0 fallback). Uses the resolved_periods_per_day
    # SSoT (no duplicated 23400/bin_seconds math).
    if args.bin_seconds is not None:
        print(
            f"  Annualization: periods_per_day="
            f"{config.resolved_periods_per_day:.1f} "
            f"(mode-aware RTH 23400s / {args.bin_seconds}s bins; #PY-263)"
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
    # #PY-263 (2026-05-21): use resolved_periods_per_day for mode-aware
    # dispatch (closes silent Sharpe inflation at sub-daily bins). See
    # BacktestConfig.resolved_periods_per_day docstring.
    ppd = config.resolved_periods_per_day
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
        # FIND-NEW-01 closure (2026-05-16): pass explicit events_per_minute
        # derived from --bin-seconds OR --events-per-minute CLI flag above
        # (no silent default; mutex group at argparse).
        transformer = ZeroDtePnLTransformer(
            zero_dte_config, events_per_minute=events_per_minute
        )
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
    # G1a / #PY-263 (2026-05-30): persist the resolved annualization comparability
    # key so the saved record self-describes WHICH annualization scaled the
    # Sharpe/Sortino/Calmar (390 at 60s vs the 1000 fallback) — makes runs
    # comparable/auditable from the artifact alone (not just stdout, which
    # hft-ops truncates). Reuses the BacktestConfig properties (no duplicated math).
    config_dict["annualization"] = {
        "resolved_periods_per_day": config.resolved_periods_per_day,
        "annualization_factor": config.annualization_factor,
        "trading_days_per_year": config.trading_days_per_year,
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
                    # G1a / #PY-263 (2026-05-30 sister-symmetry fix): persist the
                    # annualization comparability key into the hft-ops LEDGER record,
                    # top-level to MATCH the regression sister-record
                    # (run_regression_backtest.py:653-654). The original G1a (a646187)
                    # wrote this only into the registry config_dict (nested), leaving
                    # THIS ledger record — the cross-run query surface that
                    # compare_experiments reads — without the key, so readability runs
                    # silently mis-grouped vs regression runs in any annualization-keyed
                    # ledger query. Reuses the BacktestConfig properties (no duplicated
                    # 23400/bin_seconds math, hft-rules §0). float() per the regression
                    # sister-record's numpy-scalar-safe convention.
                    "resolved_periods_per_day": float(config.resolved_periods_per_day),
                    "annualization_factor": float(config.annualization_factor),
                    "manifest": str(manifest_path),
                }
                record_path = ledger_path / f"{manifest_exp_name}_backtest_{args.name}.json"
                # FIND-090 sister-site closure (2026-05-15 R-19 cycle C5):
                # atomic-write SSoT for cross-repo hft-ops ledger linkage.
                # SIGKILL mid-write here corrupts hft-ops ledger state.
                # `sort_keys=True` matches hft-ops SSoT convention.
                # atomic_write_json honors datetime/Enum/Path via internal
                # default=str (per atomic_io.py:191 docstring) — drop-in
                # compatible with pre-fix `default=str` kwarg.
                atomic_write_json(record_path, record, sort_keys=True, indent=2)
                print(f"  Updated hft-ops ledger: {record_path}")
        except Exception as e:
            # NOTE: bare `except Exception` catches `AtomicWriteError` from
            # hft_contracts.atomic_io. Narrower exception tuple (matching
            # run_regression_backtest.py:464 pattern) is OUT OF SCOPE for
            # C5 — broad-except is pre-existing behavior; tightening is a
            # separate hft-rules §8 hygiene cycle.
            print(f"  WARNING: Failed to update hft-ops ledger: {e}")

    print(f"\n{'='*60}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
