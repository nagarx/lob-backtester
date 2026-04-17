#!/usr/bin/env python3
"""
0DTE Backtester with spread_bps Signal — E13 Phase 8.

Uses raw spread_bps (z-scored with training statistics) as the trading signal.
Per-day backtesting for 0DTE correctness (options expire same day).
Deep ITM (delta=0.95) with IBKR-calibrated costs.
Post-hoc theta correction for ATM→deep ITM overestimation.

Decision gate: positive option return with >= 50 trades.

Usage:
    cd lob-backtester && .venv/bin/python scripts/run_spread_signal_backtest.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import norm as scipy_norm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "hft-feature-evaluator" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "hft-contracts" / "src"))

from lobbacktest.config import BacktestConfig, CostConfig, ZeroDteConfig, OpraCalibratedCosts
from lobbacktest.engine.vectorized import BacktestData, VectorizedEngine
from lobbacktest.engine.zero_dte import ZeroDtePnLTransformer
from lobbacktest.strategies.regression import RegressionStrategy, RegressionStrategyConfig
from lobbacktest.strategies.holding import HorizonAlignedPolicy
from hft_evaluator.data.loader import ExportLoader


# =============================================================================
# Constants
# =============================================================================

EXPORT_DIR = str(PROJECT_ROOT / "data" / "exports" / "e5_timebased_60s_point_return")
OUTPUT_DIR = str(SCRIPT_DIR.parent / "outputs" / "backtests" / "spread_bps_signal")
SPREAD_BPS_IDX = 42   # feature index for spread_bps
MID_PRICE_IDX = 40     # feature index for mid_price
TARGET_HORIZON_IDX = 7  # H=60 point return

HOLD_EVENTS = 60       # 60 bins = 60 minutes
DELTA = 0.95           # deep ITM
INITIAL_CAPITAL = 100_000.0

# Threshold sweep
Z_THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0]       # for z-score variants (unbounded)
RANK_THRESHOLDS = [0.0, 0.2, 0.4, 0.6, 0.8]     # for trailing rank (bounded [-1,+1])
TRAILING_RANK_WINDOW = 60                          # 60 bins = 60 minutes

# Ridge comparison features (Subset A from E13)
RIDGE_FEATURES = {
    "spread_bps": 42,
    "total_ask_volume": 44,
    "volume_imbalance": 45,
    "true_ofi": 84,
    "depth_norm_ofi": 85,
}
RIDGE_ALPHA = 1000.0
RIDGE_INDICES = sorted(RIDGE_FEATURES.values())


def log(msg: str) -> None:
    print(f"[backtest] {msg}", flush=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class DaySignal:
    """One day's trading data with signal."""
    date: str
    prices: np.ndarray       # [N] raw USD mid-prices
    z_spread: np.ndarray     # [N] z-scored spread_bps (signal)
    labels: np.ndarray       # [N] point-return bps at H=60
    raw_spread: np.ndarray   # [N] raw spread_bps
    n: int


WARMUP_BINS = 30  # First 30 minutes used as per-day baseline (causal)


def load_days(loader, dates, train_mean, train_std, ridge_beta=None,
              ridge_mu=None, ridge_sigma=None):
    """Load all days with multiple z-score variants and optional Ridge predictions.

    Z-score variants:
      - z_global: z-scored with TRAINING mean/std (between-period level shift issue)
      - z_day_warmup: z-scored with first-30-min per-day baseline (causal, no look-ahead)
      - z_expanding: expanding z-score from sample 0 (causal, noisy early)
    """
    days = []
    for d in dates:
        b = loader.load_day(d)
        prices = np.asarray(b.sequences[:, -1, MID_PRICE_IDX], dtype=np.float64)
        raw_spread = np.asarray(b.sequences[:, -1, SPREAD_BPS_IDX], dtype=np.float64)
        labels = np.asarray(b.labels[:, TARGET_HORIZON_IDX], dtype=np.float64)

        # Global z-score (training statistics) — prone to between-period level shift
        z_global = (raw_spread - train_mean) / max(train_std, 1e-10)

        # Per-day warmup z-score (first WARMUP_BINS as baseline — CAUSAL)
        n = len(raw_spread)
        z_warmup = np.zeros(n, dtype=np.float64)
        if n > WARMUP_BINS:
            warmup_mean = np.mean(raw_spread[:WARMUP_BINS])
            warmup_std = np.std(raw_spread[:WARMUP_BINS])
            if warmup_std < 1e-10:
                warmup_std = 1e-10
            z_warmup[WARMUP_BINS:] = (raw_spread[WARMUP_BINS:] - warmup_mean) / warmup_std
            # Warmup period: z = 0 (no signal → HOLD at any threshold > 0)

        # Expanding z-score (causal, growing window from sample 0)
        z_expanding = np.zeros(n, dtype=np.float64)
        running_sum = 0.0
        running_sq_sum = 0.0
        for t in range(n):
            running_sum += raw_spread[t]
            running_sq_sum += raw_spread[t] ** 2
            count = t + 1
            if count >= 3:
                exp_mean = running_sum / count
                exp_var = running_sq_sum / count - exp_mean ** 2
                exp_std = max(np.sqrt(max(exp_var, 0.0)), 1e-10)
                z_expanding[t] = (raw_spread[t] - exp_mean) / exp_std

        # Trailing-rank signal (BALANCED: ~50% BUY, ~50% SELL by construction)
        z_trailing_rank = np.zeros(n, dtype=np.float64)
        for t in range(TRAILING_RANK_WINDOW, n):
            window = raw_spread[t - TRAILING_RANK_WINDOW:t]
            rank_pct = np.searchsorted(np.sort(window), raw_spread[t]) / len(window) * 100
            z_trailing_rank[t] = (rank_pct - 50) / 50  # range [-1, +1]

        day = DaySignal(
            date=d, prices=prices, z_spread=z_trailing_rank,  # PRIMARY: trailing rank
            labels=labels, raw_spread=raw_spread, n=n,
        )
        day.z_global = z_global
        day.z_warmup = z_warmup
        day.z_expanding = z_expanding
        day.z_trailing_rank = z_trailing_rank

        # Ridge predictions (if model provided)
        if ridge_beta is not None:
            feat = np.asarray(b.sequences[:, -1, :][:, RIDGE_INDICES], dtype=np.float64)
            feat_std = (feat - ridge_mu) / ridge_sigma
            feat_i = np.column_stack([np.ones(len(feat_std)), feat_std])
            day.ridge_preds = feat_i @ ridge_beta
        else:
            day.ridge_preds = None

        days.append(day)
    return days


# =============================================================================
# Per-Day Backtest
# =============================================================================

def run_day_backtest(day, signal_values, config, strategy_config,
                     holding_policy, transformer):
    """Run backtest on a single day. Returns (ZeroDteResult or None, day_meta)."""
    valid = np.all(np.isfinite(
        np.column_stack([day.prices, signal_values, day.labels])
    ), axis=1) & (day.prices > 0)

    if valid.sum() < 20:
        return None, {"date": day.date, "n_valid": int(valid.sum()), "trades": 0}

    data = BacktestData(
        prices=day.prices[valid],
        predicted_returns=signal_values[valid],
        regression_labels=day.labels[valid],
        spreads=None,  # DISABLED: signal IS the spread
    )

    strategy = RegressionStrategy(
        predicted_returns=data.predicted_returns,
        spreads=data.spreads,
        prices=data.prices,
        config=strategy_config,
        holding_policy=holding_policy,
    )

    engine = VectorizedEngine(config)
    result = engine.run(data, strategy, metrics=[])

    meta = {
        "date": day.date,
        "n_valid": int(valid.sum()),
        "trades": len(result.trade_pnls),
    }

    # ALWAYS save equity metrics (before option transformation)
    if len(result.trade_pnls) > 0:
        meta["equity_pnl"] = float(result.trade_pnls.sum())
        meta["equity_trades"] = len(result.trade_pnls)
        meta["equity_win_rate"] = float((result.trade_pnls > 0).mean())
    else:
        meta["equity_pnl"] = 0.0
        meta["equity_trades"] = 0
        meta["equity_win_rate"] = 0.0

    if len(result.trade_pnls) > 0:
        option_result = transformer.transform(result)
        meta["option_pnl"] = float(option_result.option_trade_pnls.sum())
        meta["option_trades"] = option_result.n_trades
        meta["option_win_rate"] = float(option_result.option_win_rate) if option_result.n_trades > 0 else 0.0
        meta["avg_theta_cost"] = float(option_result.avg_theta_cost)
        meta["avg_spread_cost"] = float(option_result.avg_spread_cost)
        meta["avg_commission_cost"] = float(option_result.avg_commission_cost)
        meta["avg_holding_min"] = float(option_result.avg_holding_minutes)
        meta["avg_move_bps"] = float(option_result.avg_underlying_move_bps)
        return option_result, meta
    return None, meta


def run_threshold_sweep(days, config, holding_policy, transformer,
                        thresholds, signal_name, get_signal_fn,
                        allow_short_options=[True, False]):
    """Sweep thresholds for a signal across all days in a split."""
    results = []

    for threshold in thresholds:
        for allow_short in allow_short_options:
            strategy_config = RegressionStrategyConfig(
                min_return_bps=threshold,
                max_spread_bps=9999.0,  # disabled
                primary_horizon_idx=0,
                cooldown_events=0,
            )
            costs = config.get("cost_override", CostConfig.for_exchange("XNAS"))
            bt_config = BacktestConfig(
                initial_capital=config["initial_capital"],
                position_size=config["position_size"],
                costs=costs,
                allow_short=allow_short,
                trading_days_per_year=252.0,
                periods_per_day=245.0,
            )

            all_option_pnls = []
            all_theta_costs = []
            day_metas = []
            n_buys, n_sells = 0, 0

            for day in days:
                signal = get_signal_fn(day)
                opt_result, meta = run_day_backtest(
                    day, signal, bt_config, strategy_config,
                    holding_policy, transformer,
                )
                day_metas.append(meta)
                if opt_result is not None:
                    all_option_pnls.extend(opt_result.option_trade_pnls.tolist())
                    all_theta_costs.extend(opt_result.theta_costs.tolist())
                    n_buys += int(opt_result.is_call.sum())
                    n_sells += int((~opt_result.is_call).sum())

            pnls = np.array(all_option_pnls) if all_option_pnls else np.array([])
            thetas = np.array(all_theta_costs) if all_theta_costs else np.array([])
            n_trades = len(pnls)

            # Aggregate metrics
            total_pnl = float(pnls.sum()) if n_trades > 0 else 0.0
            total_return = total_pnl / config["initial_capital"]
            win_rate = float((pnls > 0).mean()) if n_trades > 0 else 0.0
            avg_pnl = float(pnls.mean()) if n_trades > 0 else 0.0
            avg_theta = float(thetas.mean()) if n_trades > 0 else 0.0

            # Theta correction for deep ITM
            d1 = scipy_norm.ppf(DELTA)
            correction_ratio = 1.0 - scipy_norm.pdf(d1) / scipy_norm.pdf(0)
            theta_correction_per_trade = avg_theta * correction_ratio
            corrected_total_pnl = total_pnl + theta_correction_per_trade * n_trades
            corrected_return = corrected_total_pnl / config["initial_capital"]

            # Per-day P&L (option level)
            daily_pnls = [m.get("option_pnl", 0.0) for m in day_metas]
            daily_pnls_arr = np.array(daily_pnls)
            positive_days = int((daily_pnls_arr > 0).sum())
            trading_days = int((daily_pnls_arr != 0).sum())

            # Sharpe (from daily P&L)
            if trading_days >= 5:
                daily_returns = daily_pnls_arr[daily_pnls_arr != 0] / config["initial_capital"]
                sharpe = float(np.mean(daily_returns) / max(np.std(daily_returns), 1e-10) * np.sqrt(252))
            else:
                sharpe = 0.0

            # Equity-level P&L (before 0DTE transformation — NO theta)
            equity_daily = np.array([m.get("equity_pnl", 0.0) for m in day_metas])
            equity_total = float(equity_daily.sum())
            equity_return = equity_total / config["initial_capital"]
            equity_trades_total = sum(m.get("equity_trades", 0) for m in day_metas)
            equity_wr_vals = [m["equity_win_rate"] for m in day_metas if m.get("equity_trades", 0) > 0]
            equity_win_rate = float(np.mean(equity_wr_vals)) if equity_wr_vals else 0.0
            equity_pos_days = int((equity_daily > 0).sum())
            equity_trading_days = int((equity_daily != 0).sum())
            if equity_trading_days >= 5:
                eq_rets = equity_daily[equity_daily != 0] / config["initial_capital"]
                equity_sharpe = float(np.mean(eq_rets) / max(np.std(eq_rets), 1e-10) * np.sqrt(252))
            else:
                equity_sharpe = 0.0

            label = f"{signal_name}_z{threshold:.1f}_{'ls' if allow_short else 'lo'}"
            entry = {
                "label": label,
                "signal": signal_name,
                "threshold": threshold,
                "allow_short": allow_short,
                "n_trades": n_trades,
                "n_buys": n_buys,
                "n_sells": n_sells,
                "trades_per_day": round(n_trades / max(len(days), 1), 2),
                # As-backtested (conservative — ATM theta)
                "total_pnl_usd": round(total_pnl, 2),
                "total_return_pct": round(total_return * 100, 4),
                "win_rate": round(win_rate, 4),
                "avg_pnl_per_trade": round(avg_pnl, 4),
                "sharpe_annual": round(sharpe, 3),
                # Theta correction
                "avg_theta_cost_usd": round(avg_theta, 4),
                "theta_correction_per_trade": round(theta_correction_per_trade, 4),
                "correction_ratio": round(correction_ratio, 4),
                # Theta-corrected (realistic deep ITM)
                "corrected_total_pnl_usd": round(corrected_total_pnl, 2),
                "corrected_return_pct": round(corrected_return * 100, 4),
                # Per-day
                "positive_days": positive_days,
                "trading_days": trading_days,
                "total_days": len(days),
                "worst_day_pnl": round(float(daily_pnls_arr.min()), 2),
                "best_day_pnl": round(float(daily_pnls_arr.max()), 2),
                # Equity-level P&L (NO theta, direct stock trading)
                "equity_total_pnl_usd": round(equity_total, 2),
                "equity_return_pct": round(equity_return * 100, 4),
                "equity_trades": equity_trades_total,
                "equity_win_rate": round(equity_win_rate, 4),
                "equity_sharpe": round(equity_sharpe, 3),
                "equity_positive_days": equity_pos_days,
                # Per-day details
                "per_day": day_metas,
            }
            results.append(entry)

            log(f"  {label}: trades={n_trades}, "
                f"EQ={equity_return*100:+.2f}% (${equity_total:+.0f}), "
                f"OPT={total_return*100:+.2f}% (corrected: {corrected_return*100:+.2f}%), "
                f"WR_eq={equity_win_rate:.1%}, buys={n_buys}/sells={n_sells}")

    return results


# =============================================================================
# Report Builder
# =============================================================================

def build_report(results):
    """Generate markdown report."""
    lines = []
    w = lines.append

    w("# 0DTE Backtester: spread_bps Signal (E13 Phase 8)")
    w(f"\n> Date: {results.get('analysis_date', 'N/A')}")
    w(f"> Signal: z-scored spread_bps (train mean={results['train_mean']:.4f}, "
      f"std={results['train_std']:.4f})")

    variant_names = ["trailing_rank", "trailing_rank_ibkr", "spread_warmup", "spread_expanding", "spread_global"]

    for split in ["val", "test"]:
        w(f"\n## {split.upper()} Split\n")

        for vname in variant_names:
            vresults = results.get(f"{vname}_{split}", [])
            if not vresults:
                continue
            w(f"\n### {vname}\n")
            w("| Config | Trades | **Equity P&L** | Equity WR | Option P&L | Corrected | Buys/Sells |")
            w("|--------|--------|---------------|-----------|------------|-----------|------------|")
            for r in vresults:
                w(f"| {r['label']} | {r['n_trades']} | "
                  f"**{r.get('equity_return_pct', 0):+.2f}%** | "
                  f"{r.get('equity_win_rate', 0):.1%} | "
                  f"{r['total_return_pct']:+.2f}% | "
                  f"{r['corrected_return_pct']:+.2f}% | "
                  f"{r['n_buys']}/{r['n_sells']} |")

        ridge_results = results.get(f"ridge_{split}", [])
        if ridge_results:
            w("\n### Ridge Model (5-feature, comparison)\n")
            w("| Config | Trades | Return | Corrected | WR | Sharpe |")
            w("|--------|--------|--------|-----------|-----|--------|")
            for r in ridge_results:
                w(f"| {r['label']} | {r['n_trades']} | "
                  f"{r['total_return_pct']:+.2f}% | "
                  f"**{r['corrected_return_pct']:+.2f}%** | "
                  f"{r['win_rate']:.1%} | {r['sharpe_annual']:.2f} |")

        # Best across all variants
        all_split = []
        for vname in variant_names:
            all_split.extend(results.get(f"{vname}_{split}", []))
        if all_split:
            best = max(all_split, key=lambda x: x["corrected_return_pct"])
            w(f"\n**Best spread_bps config**: {best['label']} → "
              f"corrected return **{best['corrected_return_pct']:+.2f}%** "
              f"({best['n_trades']} trades, WR={best['win_rate']:.1%})")

    # Theta analysis
    w("\n## Theta Correction Analysis\n")
    w(f"- ATM theta approximation (N'(0)=0.399) overestimates deep ITM (N'(1.645)=0.103)")
    w(f"- Correction ratio: {results.get('correction_ratio', 0):.4f}")
    first = results.get("spread_val", [{}])[0] if results.get("spread_val") else {}
    if first:
        w(f"- Avg theta cost (ATM): ${first.get('avg_theta_cost_usd', 0):.2f}/trade")
        w(f"- Theta correction: ${first.get('theta_correction_per_trade', 0):.2f}/trade")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="0DTE Backtester: spread_bps Signal")
    parser.add_argument("--export-dir", default=EXPORT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    start = time.time()
    log("0DTE Backtester: spread_bps Signal (E13 Phase 8)")

    export_path = Path(args.export_dir).resolve()
    output_path = Path(args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # === Load Training Data for Z-Score Statistics ===
    log("Loading training data for z-score statistics...")
    train_loader = ExportLoader(str(export_path), "train")
    train_dates = train_loader.list_dates()

    train_spreads = []
    train_features_all = []
    train_labels_all = []
    for d in train_dates:
        b = train_loader.load_day(d)
        train_spreads.append(b.sequences[:, -1, SPREAD_BPS_IDX].astype(np.float64))
        train_features_all.append(b.sequences[:, -1, :][:, RIDGE_INDICES].astype(np.float64))
        train_labels_all.append(b.labels[:, TARGET_HORIZON_IDX].astype(np.float64))

    train_spread_concat = np.concatenate(train_spreads)
    train_mean = float(np.mean(train_spread_concat))
    train_std = float(np.std(train_spread_concat))
    log(f"Training spread_bps: mean={train_mean:.4f}, std={train_std:.4f}, "
        f"n={len(train_spread_concat):,}")

    # === Fit Ridge Model (for comparison) ===
    log("Fitting Ridge model (5-feature, alpha=1000)...")
    X_train = np.vstack(train_features_all)
    y_train = np.concatenate(train_labels_all)
    valid_tr = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
    X_tr, y_tr = X_train[valid_tr], y_train[valid_tr]

    ridge_mu = np.mean(X_tr, axis=0)
    ridge_sigma = np.std(X_tr, axis=0)
    ridge_sigma[ridge_sigma < 1e-10] = 1.0
    X_tr_std = (X_tr - ridge_mu) / ridge_sigma
    n, d = X_tr_std.shape
    X_tr_i = np.column_stack([np.ones(n), X_tr_std])
    I = np.eye(d + 1)
    I[0, 0] = 0.0
    ridge_beta = np.linalg.solve(X_tr_i.T @ X_tr_i + RIDGE_ALPHA * I, X_tr_i.T @ y_tr)
    log(f"Ridge beta: intercept={ridge_beta[0]:.4f}, "
        f"spread_bps={ridge_beta[1]:.4f}")

    # === Load Val + Test Data ===
    log("Loading val and test data...")
    val_loader = ExportLoader(str(export_path), "val")
    test_loader = ExportLoader(str(export_path), "test")
    val_dates = val_loader.list_dates()
    test_dates = test_loader.list_dates()

    val_days = load_days(val_loader, val_dates, train_mean, train_std,
                         ridge_beta, ridge_mu, ridge_sigma)
    test_days = load_days(test_loader, test_dates, train_mean, train_std,
                          ridge_beta, ridge_mu, ridge_sigma)

    log(f"Val: {len(val_days)} days, Test: {len(test_days)} days")

    # === Verification: Z-score sanity ===
    val_z_all = np.concatenate([d.z_spread for d in val_days])
    log(f"Val z-spread: mean={val_z_all.mean():.4f}, std={val_z_all.std():.4f}, "
        f"min={val_z_all.min():.2f}, max={val_z_all.max():.2f}")
    log(f"Val prices: mean={np.mean([d.prices.mean() for d in val_days]):.2f} USD")

    # === 0DTE Configuration ===
    opra_costs = OpraCalibratedCosts.deep_itm()
    zero_dte_config = ZeroDteConfig(
        enabled=True,
        delta=DELTA,
        opra_costs=opra_costs,
        contracts_per_trade=1,
    )
    transformer = ZeroDtePnLTransformer(zero_dte_config, events_per_minute=1.0)
    holding_policy = HorizonAlignedPolicy(hold_events=HOLD_EVENTS)

    bt_config = {
        "initial_capital": INITIAL_CAPITAL,
        "position_size": 0.10,
    }

    # === Run Backtests ===
    all_results = {
        "schema": "spread_bps_backtest_v1",
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "export_dir": str(export_path),
        "train_mean": train_mean,
        "train_std": train_std,
        "delta": DELTA,
        "hold_events": HOLD_EVENTS,
        "correction_ratio": round(
            1.0 - scipy_norm.pdf(scipy_norm.ppf(DELTA)) / scipy_norm.pdf(0), 4
        ),
    }

    # Signal variants to test
    # IBKR equity costs (conservative: market orders, no rebates)
    # Validated: $0.85/side, $1.70 RT, breakeven 0.97 bps
    ibkr_equity_costs = CostConfig(
        spread_bps=0.28,            # half-spread per side: $0.005/$176 * 10000
        slippage_bps=0.0,
        commission_per_trade=0.35,  # IBKR TIERED: $0.0035/share × 100 shares
        taker_fee_bps=0.0,
        maker_rebate_bps=0.0,
    )

    signal_variants = [
        ("trailing_rank", lambda d: d.z_trailing_rank, RANK_THRESHOLDS),
        ("trailing_rank_ibkr", lambda d: d.z_trailing_rank, RANK_THRESHOLDS),  # IBKR costs
        ("spread_warmup", lambda d: d.z_warmup, Z_THRESHOLDS),
        ("spread_expanding", lambda d: d.z_expanding, Z_THRESHOLDS),
        ("spread_global", lambda d: d.z_global, Z_THRESHOLDS),
    ]

    for split_name, days in [("val", val_days), ("test", test_days)]:
        for sig_name, get_fn, thresholds in signal_variants:
            log(f"\n{'='*60}")
            log(f"{sig_name.upper()} — {split_name.upper()} ({len(days)} days)")
            log(f"{'='*60}")

            z_all = np.concatenate([get_fn(d) for d in days])
            log(f"  z-score stats: mean={z_all.mean():.3f}, std={z_all.std():.3f}, "
                f"frac_pos={np.mean(z_all > 0):.1%}")

            # Use IBKR equity costs for _ibkr variants
            sweep_config = dict(bt_config)
            if "ibkr" in sig_name:
                sweep_config["cost_override"] = ibkr_equity_costs

            results = run_threshold_sweep(
                days, sweep_config, holding_policy, transformer,
                thresholds, sig_name,
                get_signal_fn=get_fn,
            )
            all_results[f"{sig_name}_{split_name}"] = results

        log(f"\n{'='*60}")
        log(f"RIDGE MODEL — {split_name.upper()} ({len(days)} days)")
        log(f"{'='*60}")

        ridge_thresholds = [0.0, 1.4, 2.0, 3.0, 5.0]
        ridge_results = run_threshold_sweep(
            days, bt_config, holding_policy, transformer,
            ridge_thresholds, "ridge_5feat",
            get_signal_fn=lambda d: d.ridge_preds,
            allow_short_options=[True],  # Ridge only long-short
        )
        all_results[f"ridge_{split_name}"] = ridge_results

    # === Write Outputs ===
    json_path = output_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    log(f"\nJSON: {json_path}")

    report = build_report(all_results)
    md_path = output_path / "REPORT.md"
    with open(md_path, "w") as f:
        f.write(report)
    log(f"Report: {md_path}")

    elapsed = time.time() - start
    log(f"\nTotal: {elapsed:.1f}s")

    # === Summary ===
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")
    for split_name in ["val", "test"]:
        for vname in ["trailing_rank", "trailing_rank_ibkr", "spread_warmup", "spread_expanding", "spread_global"]:
            vr = all_results.get(f"{vname}_{split_name}", [])
            if vr:
                best = max(vr, key=lambda x: x.get("equity_return_pct", -999))
                log(f"{split_name.upper()} best {vname}: {best['label']} → "
                    f"EQUITY {best.get('equity_return_pct', 0):+.2f}% "
                    f"(${best.get('equity_total_pnl_usd', 0):+.0f}), "
                    f"option corrected {best['corrected_return_pct']:+.2f}%, "
                    f"{best['n_trades']} trades, buys={best['n_buys']}/sells={best['n_sells']}")
        ridge_r = all_results.get(f"ridge_{split_name}", [])
        if ridge_r:
            best_r = max(ridge_r, key=lambda x: x["corrected_return_pct"])
            log(f"{split_name.upper()} best ridge: {best_r['label']} → "
                f"corrected {best_r['corrected_return_pct']:+.2f}% "
                f"({best_r['n_trades']} trades, WR={best_r['win_rate']:.1%})")


if __name__ == "__main__":
    main()
