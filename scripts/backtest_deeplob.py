#!/usr/bin/env python3
"""
Backtest a trained DeepLOB model on test data.

Usage:
    python scripts/backtest_deeplob.py --experiment nvda_h10_weighted_v1

This script:
1. Loads the trained model checkpoint
2. Loads all test data from the export directory
3. Generates predictions for each day
4. Runs the backtest with configurable parameters
5. Reports comprehensive metrics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add paths for local packages
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lob-model-trainer" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "lob-models" / "src"))

import torch
from lobmodels.models.deeplob import create_deeplob

from lobbacktest import Backtester, BacktestConfig, CostConfig
from lobbacktest.types import TradeSide
from lobbacktest.engine.vectorized import BacktestData
from lobbacktest.strategies.direction import DirectionStrategy, ThresholdStrategy


def load_model(experiment_dir: Path, device: str = "cpu") -> torch.nn.Module:
    """Load trained DeepLOB model from checkpoint."""
    checkpoint_path = experiment_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
    model.eval()
    model.to(device)

    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    val_loss = checkpoint.get('val_loss')
    print(f"  Val loss: {val_loss:.4f}" if val_loss else "  Val loss: N/A")

    return model


def load_test_data(data_dir: Path, horizon_idx: int = 0) -> Dict[str, dict]:
    """Load all test data by day.
    
    Returns:
        Dict mapping day -> {sequences, labels, prices, normalization}
    """
    test_dir = data_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    days = sorted(set(f.stem.split("_")[0] for f in test_dir.glob("*_sequences.npy")))
    print(f"✓ Found {len(days)} test days")

    data_by_day = {}
    for day in days:
        seq = np.load(test_dir / f"{day}_sequences.npy")
        labels_raw = np.load(test_dir / f"{day}_labels.npy")
        
        with open(test_dir / f"{day}_normalization.json") as f:
            norm = json.load(f)

        # Handle multi-horizon labels
        labels = labels_raw[:, horizon_idx] if labels_raw.ndim == 2 else labels_raw

        # Extract and denormalize prices
        mid_price_idx = 40  # Feature index for mid-price
        prices_norm = seq[:, -1, mid_price_idx]
        
        # Denormalize using level 0 stats
        price_mean = norm["price_means"][0]
        price_std = norm["price_stds"][0] if norm["price_stds"][0] > 0 else 1.0
        prices = prices_norm * price_std + price_mean

        data_by_day[day] = {
            "sequences": seq,
            "labels": labels,
            "prices": prices,
            "n_samples": len(seq),
        }

    total_samples = sum(d["n_samples"] for d in data_by_day.values())
    print(f"  Total samples: {total_samples:,}")

    return data_by_day


def generate_predictions(
    model: torch.nn.Module,
    sequences: np.ndarray,
    device: str = "cpu",
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions and probabilities."""
    # Select first 40 features (LOB only for benchmark mode)
    sequences_lob = sequences[:, :, :40].astype(np.float32)

    predictions = []
    probabilities = []

    with torch.no_grad():
        for i in range(0, len(sequences_lob), batch_size):
            batch = torch.from_numpy(sequences_lob[i : i + batch_size]).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            predictions.append(preds.cpu().numpy())
            probabilities.append(probs.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(probabilities)


def run_backtest(
    prices: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    config: BacktestConfig,
    use_threshold: bool = False,
    probabilities: np.ndarray = None,
    threshold: float = 0.6,
) -> dict:
    """Run backtest and return results."""
    # Create strategy
    if use_threshold and probabilities is not None:
        strategy = ThresholdStrategy(
            predictions=predictions,
            probabilities=probabilities,
            threshold=threshold,
            shifted=True,  # Model outputs {0, 1, 2}
        )
    else:
        strategy = DirectionStrategy(
            predictions=predictions,
            shifted=True,
        )

    # Shift labels from {-1, 0, 1} to {0, 1, 2} for consistency
    labels_shifted = labels + 1

    data = BacktestData(prices=prices, labels=labels_shifted)
    backtester = Backtester(config)
    result = backtester.run(data, strategy)

    return result


def print_results(result, title: str = "Backtest Results"):
    """Print formatted backtest results."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"\n📊 Performance Metrics:")
    print(f"  Total Return:      {result.total_return:>10.2%}")
    print(f"  Total PnL:         ${result.total_pnl:>10,.2f}")
    print(f"  Final Equity:      ${result.final_equity:>10,.2f}")
    print(f"  Max Drawdown:      {result.max_drawdown:>10.2%}")
    
    print(f"\n📈 Risk Metrics:")
    for key in ["SharpeRatio", "SortinoRatio", "CalmarRatio"]:
        if key in result.metrics:
            print(f"  {key}:".ljust(22) + f"{result.metrics[key]:>10.2f}")
    
    print(f"\n🎯 Trading Metrics:")
    for key in ["WinRate", "ProfitFactor", "Expectancy"]:
        if key in result.metrics:
            val = result.metrics[key]
            if key == "Expectancy":
                print(f"  {key}:".ljust(22) + f"${val:>10.2f}")
            else:
                print(f"  {key}:".ljust(22) + f"{val:>10.2f}")
    
    print(f"\n📋 Trade Summary:")
    print(f"  Total Trades:      {result.total_trades:>10,}")
    print(f"  Data Points:       {len(result.prices):>10,}")


def main():
    parser = argparse.ArgumentParser(description="Backtest DeepLOB model")
    parser.add_argument(
        "--experiment",
        type=str,
        default="nvda_h10_weighted_v1",
        help="Experiment name (directory in lob-model-trainer/outputs/experiments/)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: data/exports/nvda_balanced)",
    )
    parser.add_argument(
        "--horizon-idx",
        type=int,
        default=0,
        help="Horizon index for labels (0=h10, 1=h20, 2=h50, 3=h100, 4=h200)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Initial capital in USD",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.1,
        help="Position size as fraction of capital",
    )
    parser.add_argument(
        "--spread-bps",
        type=float,
        default=1.0,
        help="Spread cost in basis points",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.5,
        help="Slippage cost in basis points",
    )
    parser.add_argument(
        "--no-short",
        action="store_true",
        help="Disable short selling",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold (0-1). If set, only trade high-confidence signals",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)",
    )

    args = parser.parse_args()

    # Resolve paths
    experiment_dir = PROJECT_ROOT / "lob-model-trainer" / "outputs" / "experiments" / args.experiment
    if not experiment_dir.exists():
        print(f"❌ Experiment not found: {experiment_dir}")
        sys.exit(1)

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "exports" / "nvda_balanced"
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)

    print(f"\n🚀 DeepLOB Backtest")
    print(f"   Experiment: {args.experiment}")
    print(f"   Data: {data_dir.name}")
    print(f"   Horizon: h={[10, 20, 50, 100, 200][args.horizon_idx]}")

    # Load model and data
    model = load_model(experiment_dir, args.device)
    data_by_day = load_test_data(data_dir, args.horizon_idx)

    # Configure backtest
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        position_size=args.position_size,
        costs=CostConfig(
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
        ),
        allow_short=not args.no_short,
    )

    print(f"\n⚙️  Backtest Configuration:")
    print(f"   Initial Capital: ${config.initial_capital:,.0f}")
    print(f"   Position Size: {config.position_size:.0%}")
    print(f"   Costs: {config.costs.total_bps:.1f} bps")
    print(f"   Short Selling: {'Enabled' if config.allow_short else 'Disabled'}")
    if args.threshold:
        print(f"   Confidence Threshold: {args.threshold:.0%}")

    # Aggregate all test data
    all_prices = []
    all_predictions = []
    all_probabilities = []
    all_labels = []

    print(f"\n📊 Generating predictions...")
    for day, data in data_by_day.items():
        preds, probs = generate_predictions(model, data["sequences"], args.device)
        all_prices.append(data["prices"])
        all_predictions.append(preds)
        all_probabilities.append(probs)
        all_labels.append(data["labels"])

    prices = np.concatenate(all_prices)
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)
    labels = np.concatenate(all_labels)

    print(f"   Generated {len(predictions):,} predictions")

    # Prediction distribution
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"\n📈 Prediction Distribution:")
    label_names = {0: "Down", 1: "Stable", 2: "Up"}
    for u, c in zip(unique, counts):
        print(f"   {label_names[u]}: {c:,} ({100*c/len(predictions):.1f}%)")

    # Run backtest
    print(f"\n⏳ Running backtest...")
    result = run_backtest(
        prices=prices,
        predictions=predictions,
        labels=labels,
        config=config,
        use_threshold=args.threshold is not None,
        probabilities=probabilities if args.threshold else None,
        threshold=args.threshold or 0.6,
    )

    # Print results
    print_results(result, f"Results: {args.experiment}")

    # Debug: analyze trade P&Ls
    trade_pnls = []
    for trade in result.trades:
        if trade.side == TradeSide.FLAT:  # Closing trades
            # P&L is not stored directly, but we can infer from cost
            pass  # Would need to track entry prices
    
    # Count trade types
    from collections import Counter
    trade_sides = Counter(t.side.name for t in result.trades)
    print(f"\n📝 Trade Types:")
    for side, count in trade_sides.items():
        print(f"   {side}: {count}")

    # Compute prediction accuracy
    labels_shifted = labels + 1
    accuracy = (predictions == labels_shifted).mean()
    directional_mask = labels_shifted != 1  # Non-stable
    directional_acc = (predictions[directional_mask] == labels_shifted[directional_mask]).mean()

    print(f"\n🎯 Prediction Accuracy:")
    print(f"   Overall Accuracy:     {accuracy:>10.2%}")
    print(f"   Directional Accuracy: {directional_acc:>10.2%}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()

