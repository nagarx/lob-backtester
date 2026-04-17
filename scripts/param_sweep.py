#!/usr/bin/env python3
"""
Parameter sweep for DeepLOB backtest optimization.

Tests combinations of:
- Confidence thresholds: 0.5, 0.6, 0.7, 0.8
- Strategy modes: Long-only, Long/Short
- Transaction costs: Low (0.5 bps), Medium (1.0 bps), High (1.5 bps)

Usage:
    python scripts/param_sweep.py --experiment nvda_h10_weighted_v1 --device cpu
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "lob-model-trainer" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "lob-models" / "src"))

from lobbacktest import (
    BacktestConfig,
    BacktestData,
    Backtester,
    CostConfig,
    ThresholdStrategy,
)


@dataclass
class SweepResult:
    """Result from a single parameter configuration."""
    threshold: float
    allow_short: bool
    spread_bps: float
    slippage_bps: float
    total_return: float
    total_pnl: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    expectancy: float
    total_trades: int
    n_round_trips: int
    directional_accuracy: float


def load_model_and_data(
    experiment_name: str,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load model checkpoint and generate predictions on test data.
    
    Returns:
        (prices, labels, predictions, probabilities)
    """
    import torch
    import torch.nn.functional as F
    
    # Paths
    trainer_dir = Path(__file__).parent.parent.parent / "lob-model-trainer"
    experiment_dir = trainer_dir / "outputs" / "experiments" / experiment_name
    checkpoint_path = experiment_dir / "checkpoints" / "best.pt"
    
    # Data path from experiment config
    import yaml
    config_path = trainer_dir / "configs" / "experiments" / f"{experiment_name}.yaml"
    with open(config_path) as f:
        exp_config = yaml.safe_load(f)
    
    data_dir = Path(exp_config["data"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = trainer_dir / data_dir
    
    # Load normalization params
    norm_file = list((data_dir / "test").glob("*_normalization.json"))[0]
    from lobbacktest.data.prices import NormalizationParams, PriceExtractor
    norm_params = NormalizationParams.from_json(norm_file)
    
    # Load test sequences and labels
    test_dir = data_dir / "test"
    seq_files = sorted(test_dir.glob("*_sequences.npy"))
    label_files = sorted(test_dir.glob("*_labels.npy"))
    
    all_sequences = []
    all_labels = []
    
    for seq_file, label_file in zip(seq_files, label_files):
        sequences = np.load(seq_file)
        labels = np.load(label_file)
        all_sequences.append(sequences)
        all_labels.append(labels)
    
    sequences = np.concatenate(all_sequences, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Handle multi-horizon labels
    horizon_idx = exp_config.get("data", {}).get("horizon_idx", 0)
    if labels.ndim == 2:
        labels = labels[:, horizon_idx]
    
    # Extract prices
    extractor = PriceExtractor(norm_params)
    prices = extractor.extract_mid_prices(sequences, denormalize=True)
    
    # Load model (from lob-models)
    from lobmodels.models.deeplob import create_deeplob
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model with benchmark config
    model = create_deeplob(
        mode="benchmark",
        num_levels=10,
        num_classes=3,
        conv_filters=32,
        inception_filters=64,
        lstm_hidden=64,
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Generate predictions in batches
    # Select first 40 features (LOB only for benchmark mode)
    sequences_lob = sequences[:, :, :40].astype(np.float32)
    
    batch_size = 512
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(sequences_lob), batch_size):
            batch = sequences_lob[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            logits = model(batch_tensor)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
    
    probabilities = np.concatenate(all_probs, axis=0)
    predictions = np.argmax(probabilities, axis=1)
    
    return prices, labels, predictions, probabilities


def run_single_backtest(
    prices: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    allow_short: bool,
    spread_bps: float,
    slippage_bps: float,
) -> SweepResult:
    """Run a single backtest with given parameters."""
    
    config = BacktestConfig(
        initial_capital=100_000,
        position_size=0.1,
        max_position=1.0,
        costs=CostConfig(
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            commission_per_trade=0.0,
        ),
        allow_short=allow_short,
        trading_days_per_year=252,
        periods_per_day=1000,
    )
    
    strategy = ThresholdStrategy(
        predictions=predictions,
        probabilities=probabilities,
        threshold=threshold,
        shifted=True,  # Model outputs 0/1/2
    )
    
    data = BacktestData(prices=prices, labels=labels)
    backtester = Backtester(config)
    result = backtester.run(data, strategy)
    
    # Extract metrics
    metrics = result.metrics
    
    return SweepResult(
        threshold=threshold,
        allow_short=allow_short,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        total_return=result.total_return * 100,
        total_pnl=result.total_pnl,
        sharpe_ratio=metrics.get("SharpeRatio", 0),
        sortino_ratio=metrics.get("SortinoRatio", 0),
        max_drawdown=metrics.get("MaxDrawdown", 0) * 100,
        calmar_ratio=metrics.get("CalmarRatio", 0),
        win_rate=metrics.get("WinRate", 0) * 100,
        profit_factor=metrics.get("ProfitFactor", 0),
        expectancy=metrics.get("Expectancy", 0),
        total_trades=result.total_trades,
        n_round_trips=len(result.trade_pnls),
        directional_accuracy=metrics.get("DirectionalAccuracy", 0) * 100,
    )


def run_parameter_sweep(
    prices: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> List[SweepResult]:
    """Run full parameter sweep."""
    
    # Parameter grid
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    allow_shorts = [False, True]
    cost_configs = [
        (0.0, 0.0, "Zero"),          # 0 bps total (theoretical max)
        (0.1, 0.05, "Minimal"),      # 0.15 bps total
        (0.25, 0.15, "Ultra Low"),   # 0.4 bps total
        (0.5, 0.25, "Low"),          # 0.75 bps total  
    ]
    
    results = []
    total_configs = len(thresholds) * len(allow_shorts) * len(cost_configs)
    current = 0
    
    print(f"\n🔍 Running parameter sweep ({total_configs} configurations)...\n")
    
    for threshold in thresholds:
        for allow_short in allow_shorts:
            for spread_bps, slippage_bps, cost_name in cost_configs:
                current += 1
                mode = "Long/Short" if allow_short else "Long-only"
                
                print(f"  [{current}/{total_configs}] threshold={threshold:.2f}, "
                      f"{mode}, costs={cost_name}", end="")
                
                try:
                    result = run_single_backtest(
                        prices=prices,
                        labels=labels,
                        predictions=predictions,
                        probabilities=probabilities,
                        threshold=threshold,
                        allow_short=allow_short,
                        spread_bps=spread_bps,
                        slippage_bps=slippage_bps,
                    )
                    results.append(result)
                    print(f" → Return: {result.total_return:+.2f}%")
                except Exception as e:
                    print(f" → ERROR: {e}")
    
    return results


def print_results_table(results: List[SweepResult], sort_by: str = "total_return"):
    """Print results in a formatted table."""
    
    # Sort by specified metric (descending)
    sorted_results = sorted(
        results, 
        key=lambda r: getattr(r, sort_by), 
        reverse=True
    )
    
    print("\n" + "=" * 120)
    print(" PARAMETER SWEEP RESULTS (sorted by {})".format(sort_by.replace("_", " ").title()))
    print("=" * 120)
    
    # Header
    print(f"{'Thresh':>6} │ {'Mode':^10} │ {'Costs':^6} │ "
          f"{'Return':>8} │ {'Sharpe':>7} │ {'Sortino':>8} │ "
          f"{'MaxDD':>6} │ {'WinRate':>7} │ {'PF':>5} │ "
          f"{'Trades':>6} │ {'Expect':>8}")
    print("-" * 120)
    
    for r in sorted_results[:25]:  # Top 25
        mode = "L/S" if r.allow_short else "Long"
        cost = f"{r.spread_bps + r.slippage_bps:.1f}bp"
        
        # Color coding for return
        ret_str = f"{r.total_return:+.2f}%"
        
        print(f"{r.threshold:>6.2f} │ {mode:^10} │ {cost:^6} │ "
              f"{ret_str:>8} │ {r.sharpe_ratio:>7.2f} │ {r.sortino_ratio:>8.2f} │ "
              f"{r.max_drawdown:>5.1f}% │ {r.win_rate:>6.1f}% │ {r.profit_factor:>5.2f} │ "
              f"{r.total_trades:>6} │ ${r.expectancy:>7.2f}")
    
    print("=" * 120)


def print_best_configs(results: List[SweepResult]):
    """Print best configuration for each metric."""
    
    metrics = [
        ("total_return", "Total Return", "%", True),
        ("sharpe_ratio", "Sharpe Ratio", "", True),
        ("sortino_ratio", "Sortino Ratio", "", True),
        ("profit_factor", "Profit Factor", "", True),
        ("calmar_ratio", "Calmar Ratio", "", True),
        ("max_drawdown", "Max Drawdown", "%", False),  # Lower is better
    ]
    
    print("\n" + "=" * 80)
    print(" OPTIMAL CONFIGURATIONS BY METRIC")
    print("=" * 80)
    
    for attr, name, unit, higher_better in metrics:
        sorted_results = sorted(
            results,
            key=lambda r: getattr(r, attr),
            reverse=higher_better
        )
        best = sorted_results[0]
        mode = "Long/Short" if best.allow_short else "Long-only"
        cost = f"{best.spread_bps + best.slippage_bps:.2f} bps"
        
        value = getattr(best, attr)
        if unit == "%":
            value_str = f"{value:.2f}%"
        else:
            value_str = f"{value:.3f}"
        
        print(f"\n  📊 Best {name}: {value_str}")
        print(f"     Threshold: {best.threshold:.2f}")
        print(f"     Mode: {mode}")
        print(f"     Costs: {cost}")
        print(f"     Return: {best.total_return:+.2f}% | Sharpe: {best.sharpe_ratio:.2f} | "
              f"WinRate: {best.win_rate:.1f}%")
    
    print("\n" + "=" * 80)


def print_strategy_comparison(results: List[SweepResult]):
    """Compare Long-only vs Long/Short aggregated results."""
    
    long_only = [r for r in results if not r.allow_short]
    long_short = [r for r in results if r.allow_short]
    
    def avg(lst, attr):
        vals = [getattr(r, attr) for r in lst]
        return np.mean(vals) if vals else 0
    
    def best(lst, attr):
        vals = [getattr(r, attr) for r in lst]
        return max(vals) if vals else 0
    
    print("\n" + "=" * 80)
    print(" LONG-ONLY vs LONG/SHORT COMPARISON")
    print("=" * 80)
    print(f"{'Metric':^25} │ {'Long-only (avg)':^20} │ {'Long/Short (avg)':^20}")
    print("-" * 80)
    
    metrics = [
        ("total_return", "Total Return", "%"),
        ("sharpe_ratio", "Sharpe Ratio", ""),
        ("sortino_ratio", "Sortino Ratio", ""),
        ("max_drawdown", "Max Drawdown", "%"),
        ("win_rate", "Win Rate", "%"),
        ("profit_factor", "Profit Factor", ""),
    ]
    
    for attr, name, unit in metrics:
        avg_lo = avg(long_only, attr)
        avg_ls = avg(long_short, attr)
        
        if unit == "%":
            print(f"{name:^25} │ {avg_lo:^19.2f}% │ {avg_ls:^19.2f}%")
        else:
            print(f"{name:^25} │ {avg_lo:^20.3f} │ {avg_ls:^20.3f}")
    
    print("-" * 80)
    
    # Best results
    best_lo = max(long_only, key=lambda r: r.total_return)
    best_ls = max(long_short, key=lambda r: r.total_return)
    
    print(f"\n  🏆 Best Long-only: {best_lo.total_return:+.2f}% @ threshold={best_lo.threshold:.2f}")
    print(f"  🏆 Best Long/Short: {best_ls.total_return:+.2f}% @ threshold={best_ls.threshold:.2f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for DeepLOB backtest")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
    args = parser.parse_args()
    
    print("=" * 80)
    print(f" DeepLOB Parameter Sweep: {args.experiment}")
    print("=" * 80)
    
    # Load data and model
    print("\n📦 Loading model and data...")
    prices, labels, predictions, probabilities = load_model_and_data(
        args.experiment, args.device
    )
    print(f"   Data points: {len(prices):,}")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Probabilities shape: {probabilities.shape}")
    
    # Run sweep
    results = run_parameter_sweep(prices, labels, predictions, probabilities)
    
    # Print results
    print_results_table(results, sort_by="total_return")
    print_best_configs(results)
    print_strategy_comparison(results)
    
    # Summary
    profitable = [r for r in results if r.total_return > 0]
    print(f"\n📈 Profitable configurations: {len(profitable)}/{len(results)} "
          f"({100*len(profitable)/len(results):.1f}%)")
    
    if profitable:
        best = max(profitable, key=lambda r: r.sharpe_ratio)
        mode = "Long/Short" if best.allow_short else "Long-only"
        print(f"\n🎯 RECOMMENDED CONFIG (best risk-adjusted):")
        print(f"   Threshold: {best.threshold:.2f}")
        print(f"   Mode: {mode}")
        print(f"   Costs: {best.spread_bps + best.slippage_bps:.2f} bps")
        print(f"   Expected: Return={best.total_return:+.2f}%, "
              f"Sharpe={best.sharpe_ratio:.2f}, MaxDD={best.max_drawdown:.1f}%")


if __name__ == "__main__":
    main()

