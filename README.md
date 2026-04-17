# LOB-Backtester

Standalone backtesting library for evaluating LOB prediction models trained with `lob-model-trainer`.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

## Overview

This library provides a **vectorized backtesting engine** for evaluating direction prediction models on Limit Order Book (LOB) data. It works with the exported data format from `feature-extractor-MBO-LOB` and model checkpoints from `lob-model-trainer`.

### Key Features

- **Per-Sample Engine**: Position tracking with numpy-based metrics (fast for ~50K-sample backtests)
- **Metric ABC Pattern**: Composable, extensible metrics inspired by `hftbacktest`
- **Fluent Stats API**: Chain operations for intuitive usage
- **ML-Focused Metrics**: DirectionalAccuracy, SignalRate, UpPrecision, DownPrecision
- **Trading Metrics**: Sharpe, Sortino, MaxDrawdown, WinRate, ProfitFactor
- **Transaction Costs**: Configurable spread, slippage, commission
- **Multi-Model Comparison**: Compare multiple models side-by-side
- **Comprehensive Testing**: Every module tested to expose implementation issues

## Installation

```bash
# From the lob-backtester directory
pip install -e ".[dev]"

# With PyTorch support (for loading model checkpoints)
pip install -e ".[dev,torch]"
```

## Quick Start

### Basic Backtest

```python
from lobbacktest import (
    Backtester, BacktestConfig, BacktestData, CostConfig,
    DirectionStrategy, ThresholdStrategy,
)
from lobbacktest.data import DataLoader

# Load data
loader = DataLoader(data_dir="path/to/exports", split="test")
data = loader.load()

# Configure backtest
config = BacktestConfig(
    initial_capital=100_000,
    position_size=0.1,  # 10% of capital per trade
    costs=CostConfig(spread_bps=1.0, slippage_bps=0.5),
    allow_short=False,  # Long-only
)

# Simple strategy: map predictions directly to signals
strategy = DirectionStrategy(
    predictions=predictions,  # Model argmax output (0/1/2)
    shifted=True,  # Model outputs {0, 1, 2} not {-1, 0, 1}
)

# OR: Threshold strategy - only trade high-confidence signals
strategy = ThresholdStrategy(
    predictions=predictions,
    probabilities=model_probs,  # Softmax output (N, 3)
    threshold=0.7,  # Only trade if max prob > 70%
    shifted=True,
)

# Run backtest
backtester = Backtester(config)
result = backtester.run(BacktestData(prices=data.prices, labels=data.labels), strategy)

# View results
print(result.summary())
```

### Compare Multiple Models

```python
from lobbacktest import Backtester, BacktestConfig, DirectionStrategy
from lobbacktest.reports import comparison_table, plot_comparison

# Run backtests for each model
config = BacktestConfig(initial_capital=100_000)
backtester = Backtester(config)

results = {}
for name, preds in [("DeepLOB_h10", preds_h10), ("DeepLOB_h100", preds_h100)]:
    strategy = DirectionStrategy(preds, shifted=True)
    results[name] = backtester.run(data, strategy)

# Compare
print(comparison_table(results))
fig = plot_comparison(results)
```

## Architecture

```
lob-backtester/
├── src/lobbacktest/
│   ├── __init__.py          # Public API
│   ├── types.py              # Core types (Position, Trade, BacktestResult)
│   ├── config.py             # Configuration dataclasses
│   │
│   ├── data/                 # Data loading and preprocessing
│   │   ├── loader.py         # Load sequences, labels, normalization params
│   │   └── prices.py         # Denormalize prices from features
│   │
│   ├── strategies/           # Trading strategy implementations
│   │   ├── base.py           # Strategy ABC, Signal enum
│   │   ├── direction.py      # DirectionStrategy, ThresholdStrategy
│   │   ├── readability.py    # ReadabilityStrategy (HMHP agreement + confidence)
│   │   ├── regression.py     # RegressionStrategy (magnitude gate)
│   │   ├── hybrid.py         # ReadabilityHybridStrategy (classification + regression)
│   │   ├── holding.py        # HoldingPolicy, HorizonAlignedPolicy
│   │   └── twap.py           # TWAPStrategy (time-weighted execution)
│   │
│   ├── engine/               # Backtest execution engines
│   │   ├── vectorized.py     # VectorizedEngine, Backtester, BacktestData
│   │   └── zero_dte.py       # ZeroDtePnLTransformer (0DTE options P&L)
│   │
│   ├── metrics/              # Performance metrics
│   │   ├── base.py           # Metric ABC
│   │   ├── returns.py        # Return-based metrics
│   │   ├── risk.py           # Risk metrics (Sharpe, Sortino, Drawdown)
│   │   ├── trading.py        # Trading metrics (WinRate, ProfitFactor)
│   │   ├── prediction.py     # ML prediction metrics (DirectionalAccuracy, SignalRate)
│   │   └── regression_prediction.py  # Regression metrics (MSE, IC, Correlation)
│   │
│   ├── stats/                # Statistics and aggregation
│   │   └── stats.py          # Fluent Stats API
│   │
│   └── reports/              # Reporting and visualization
│       ├── summary.py        # Text summaries
│       └── plots.py          # Equity curves, position charts
│
└── tests/                    # Comprehensive test suite
```

## Data Contract

### Input Format

The backtester expects data exported by `feature-extractor-MBO-LOB`:

```
{split}/
├── {date}_sequences.npy      # Shape: (N, 100, 98) float32
├── {date}_labels.npy         # Shape: (N,) or (N, H) int8
├── {date}_normalization.json # Normalization parameters
└── {date}_metadata.json      # Dataset metadata
```

### Label Encoding

| Value | Meaning |
|-------|---------|
| -1 | Down (price decreased > threshold) |
| 0 | Stable (price within threshold) |
| 1 | Up (price increased > threshold) |

### Model Predictions

Predictions should be numpy arrays with shape `(N,)` containing:
- Class indices `{0, 1, 2}` for Down/Stable/Up, or
- Raw logits/probabilities `(N, 3)` from which we extract argmax

## Configuration

```python
from lobbacktest.config import BacktestConfig

config = BacktestConfig(
    # Capital
    initial_capital=100_000,      # Starting capital (USD)
    position_size=0.1,            # Fraction of capital per trade
    max_position=1.0,             # Max position as fraction of capital
    
    # Transaction costs
    spread_bps=1.0,               # Bid-ask spread in basis points
    slippage_bps=0.5,             # Slippage per trade in bps
    commission_per_trade=0.0,     # Fixed commission per trade
    
    # Risk management
    stop_loss_pct=None,           # Optional stop-loss percentage
    take_profit_pct=None,         # Optional take-profit percentage
    
    # Execution
    allow_short=True,             # Allow short positions
    fill_price="close",           # "close" or "midpoint"
)
```

## Trade P&L Calculation

Trade P&L is computed per **round-trip** (entry + exit):

```
Long Position:
  P&L = (exit_price - entry_price) × size - costs

Short Position:
  P&L = (entry_price - exit_price) × size - costs

Where:
  costs = spread_bps/10000 × notional + slippage_bps/10000 × notional + commission
```

**Key points**:
- `trade_pnls` array contains P&L for each **closed** position (not open/close trades separately)
- `total_trades` counts all trades (opens + closes), so `len(trade_pnls) ≈ total_trades / 2`
- Metrics like `WinRate` and `ProfitFactor` use `trade_pnls`

## Metrics

### Risk Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| `SharpeRatio` | Risk-adjusted return | `mean(r) / std(r) * sqrt(N)` |
| `SortinoRatio` | Downside risk-adjusted | `mean(r) / downside_std(r) * sqrt(N)` |
| `MaxDrawdown` | Maximum peak-to-trough | `max(peak - equity) / peak` |
| `CalmarRatio` | Return over max drawdown | `annual_return / max_drawdown` |

### Trading Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| `WinRate` | Winning trades ratio | `winning_trades / total_trades` |
| `ProfitFactor` | Gross profit / gross loss | `sum(wins) / sum(losses)` |
| `AverageWin` | Average winning trade | `mean(winning_pnl)` |
| `AverageLoss` | Average losing trade | `mean(losing_pnl)` |
| `PayoffRatio` | Avg win / avg loss | `avg_win / avg_loss` |

### Prediction Metrics

| Metric | Description |
|--------|-------------|
| `DirectionalAccuracy` | Up/Down prediction accuracy |
| `SignalRate` | Fraction of non-Stable predictions |
| `UpPrecision` | Precision for Up predictions |
| `DownPrecision` | Precision for Down predictions |

## Testing Philosophy

Every module follows the testing principles from `RULE.md`:

1. **Formula Tests**: Verify calculations against hand-computed examples
2. **Edge Case Tests**: Handle `NaN`, `Inf`, empty arrays, zero division
3. **Boundary Tests**: Test at threshold ± ε
4. **Invariant Tests**: Ensure consistency (e.g., no profit without trades)

```python
# Example: Testing Sharpe Ratio
def test_sharpe_ratio_formula():
    """Sharpe = mean(r) / std(r) * sqrt(N)"""
    returns = np.array([0.01, -0.005, 0.02, 0.003])
    
    # Hand-calculated
    expected_sr = np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    metric = SharpeRatio(trading_days_per_year=252, periods_per_day=1)
    result = metric.compute(returns, {})
    
    assert abs(result["SharpeRatio"] - expected_sr) < 1e-10, \
        f"Expected {expected_sr}, got {result['SharpeRatio']}"
```

## Performance

- **Vectorized**: All computations use numpy, no Python loops
- **Memory efficient**: Streaming computation where possible
- **Fast**: ~10,000 trades/second on standard hardware

## Dependencies

- `numpy>=1.24.0`: Core numerical operations
- `matplotlib>=3.7.0`: Visualization
- `pyyaml>=6.0`: Configuration loading

Optional:
- `torch>=2.0.0`: Load PyTorch model checkpoints

## License

Proprietary - All Rights Reserved.

---

*Last updated: March 16, 2026*

