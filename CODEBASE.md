# LOB-Backtester: Codebase Technical Reference

> **Version**: 0.1.0 | **Tests**: 353 (345 passed + 8 skipped) | **Last Updated**: 2026-04-20 (Phase 7 Stage 7.4 Round 5)  
> **Purpose**: Complete technical details for LLMs and developers to understand, modify, and extend the codebase without prior context.

## State at HEAD (cumulative through Phase 7 Round 5)

- **Phase 0-3 stabilization** (commit `47d0d31`) — typed `BacktestContext`, `ExperimentRunner` orchestration, `ZeroDtePnLTransformer` (IBKR-calibrated 0DTE options P&L), `LabelMapping` SSoT, `HoldingPolicy` composability, P2/P3/P4/P5 fixes.
- **Phase 4 4c.4** (`de94078`) — `SignalManifest.feature_set_ref` propagation from trainer's `signal_metadata.json`; Phase 6 6A.9 `_CONTENT_HASH_RE` producer-consumer regex symmetry.
- **Phase 6 6B.5** (`77f2068`) — `SignalManifest` canonical home co-moved to `hft_contracts.signal_manifest`; this repo's `data/signal_manifest.py` is a thin re-export shim preserving pre-6B.5 imports.
- **Phase 6 final hygiene** (`d642dd1`) — shim lazy `__getattr__` emits `DeprecationWarning` once per symbol per process.
- **Phase 7 post-validation I** (`c28fc53`, `09d4665`) — calendar-driven shim deadline `2026-10-31` replaces prior "version 0.4.0" version-milestone language (hft-ops has no fixed release cadence; calendar gives a concrete migration window).

## Architecture

- **Engine**: Per-sample loop in `engine/vectorized.py` (name is historical; actual algorithm is NOT vectorized). `engine/zero_dte.py` layers IBKR-calibrated 0DTE options P&L transformation on top of the equity backtest.
- **Strategy Pattern**: `Strategy` ABC in `strategies/base.py` + 7 concretes (direction, readability, regression, hybrid, holding policies, twap SKIP). `HoldingPolicy` ABC is composable via `CompositePolicy(mode="any"|"all")`.
- **Metrics ABC**: `metrics/base.py::Metric` + 5 groupings across returns/risk/trading/prediction/regression_prediction.
- **Contract plane via hft_contracts**: `SignalManifest` canonical home at `hft_contracts.signal_manifest` (Phase 6 6B.5); `_CONTENT_HASH_RE` regex imported from same SSoT; label encoding defers to `hft_contracts.labels.LabelContract` for cross-module agreement.
- **IBKR-calibrated 0DTE cost model**: constants calibrated from 316 real NVDA option fills; breakevens 4.9 / 3.8 / 1.4 bps for ATM Call / ATM Put / Deep ITM. Provenance in `engine/zero_dte.py` docstring + `IBKR-transactions-trades/COST_AUDIT_2026_03.md`.
- **Typed + dict hybrid context**: `BacktestContext` is a typed dataclass that also implements `__getitem__` / `__contains__` / `get` / `update` for backward compat with metric consumers written for dict-protocol. Migration path to pure typed access is gradual.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Module Architecture](#2-module-architecture)
3. [Core Data Flow](#3-core-data-flow)
4. [Type System](#4-type-system)
5. [Configuration](#5-configuration)
6. [Strategies](#6-strategies)
7. [Engine](#7-engine)
8. [Metrics](#8-metrics)
9. [Stats API](#9-stats-api)
10. [Data Loading](#10-data-loading)
11. [Reports](#11-reports)
12. [Testing Patterns](#12-testing-patterns)
13. [Integration with Pipeline](#13-integration-with-pipeline)

---

## 1. Project Overview

### Purpose

Standalone backtesting library for evaluating direction prediction models trained with `lob-model-trainer`. Works directly with data exported by `feature-extractor-MBO-LOB`.

### Design Principles

1. **Per-Sample Engine**: Position tracking via Python loop with numpy-based metric computation. (Note: module name `vectorized.py` is historical; the main engine loop is a Python `for i in range(n):` loop, not vectorized.)
2. **Metric ABC Pattern**: Composable, extensible metrics (inspired by `hftbacktest`)
3. **Fluent API**: Chainable operations for intuitive usage
4. **Comprehensive Testing**: Every module tested to expose implementation issues
5. **ML-Focused**: Designed for evaluating direction predictions, not order execution

### Core Dependencies

```toml
[dependencies]
numpy = ">=1.24.0"     # Core numerical operations
matplotlib = ">=3.7.0" # Visualization
pyyaml = ">=6.0"       # Configuration loading
```

---

## 2. Module Architecture

```
src/lobbacktest/
├── __init__.py          # Public API exports
├── version.py           # Version information
├── types.py             # Core types: Trade, Position, BacktestResult
├── config.py            # Configuration: BacktestConfig, CostConfig, ZeroDteConfig, OpraCalibratedCosts
├── registry.py          # Strategy registry
│
├── data/                # Data loading and preprocessing
│   ├── __init__.py
│   ├── loader.py        # DataLoader: Load exported data
│   └── prices.py        # PriceExtractor: Denormalize prices
│
├── strategies/          # Trading strategy implementations
│   ├── __init__.py      # Exports all strategies
│   ├── base.py          # Strategy ABC, Signal enum, SignalOutput
│   ├── direction.py     # DirectionStrategy, ThresholdStrategy
│   ├── readability.py   # ReadabilityStrategy (HMHP agreement + confidence gate)
│   ├── regression.py    # RegressionStrategy (magnitude gate for continuous predictions)
│   ├── hybrid.py        # ReadabilityHybridStrategy (classification direction + regression magnitude)
│   ├── holding.py       # HoldingPolicy, HorizonAlignedPolicy, HoldingState
│   └── twap.py          # TWAPStrategy (time-weighted execution)
│
├── engine/              # Backtest execution
│   ├── __init__.py
│   ├── vectorized.py    # VectorizedEngine, Backtester, BacktestData
│   └── zero_dte.py      # ZeroDtePnLTransformer, ZeroDteResult (0DTE options P&L)
│
├── metrics/             # Performance metrics
│   ├── __init__.py
│   ├── base.py          # Metric ABC, MetricResult
│   ├── returns.py       # TotalReturn, AnnualReturn
│   ├── risk.py          # SharpeRatio, SortinoRatio, MaxDrawdown, CalmarRatio
│   ├── trading.py       # WinRate, ProfitFactor, Expectancy, etc.
│   ├── prediction.py    # DirectionalAccuracy, SignalRate, UpPrecision, DownPrecision
│   └── regression_prediction.py  # PredictionMSE, PredictionCorrelation, PredictionIC
│
├── stats/               # Statistics and aggregation
│   ├── __init__.py
│   └── stats.py         # BacktestStats fluent API
│
├── scripts/             # Runnable scripts
│   ├── backtest_deeplob.py         # DeepLOB backtest runner
│   ├── param_sweep.py              # Parameter sweep (multi-config)
│   ├── run_readability_backtest.py  # Readability strategy backtest
│   └── run_regression_backtest.py   # Regression strategy backtest
│
└── reports/             # Reporting and visualization
    ├── __init__.py
    ├── summary.py       # Text reports, comparison tables
    └── plots.py         # Equity curves, drawdown charts
```

---

## 3. Core Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BACKTEST PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    Exported Data         Model Predictions        Strategy             Engine
        │                       │                     │                   │
        ▼                       ▼                     ▼                   ▼
┌─────────────┐         ┌─────────────┐       ┌─────────────┐     ┌──────────┐
│ DataLoader  │────────▶│ Direction   │──────▶│ Vectorized  │────▶│ Backtest │
│             │         │ Strategy    │       │ Engine      │     │ Result   │
│ sequences   │         │             │       │             │     │          │
│ labels      │         │ predictions │       │ signals     │     │ equity   │
│ prices      │         │ → signals   │       │ → trades    │     │ trades   │
└─────────────┘         └─────────────┘       │ → P&L       │     │ metrics  │
                                              └─────────────┘     └──────────┘
                                                      │
                                              ┌───────┴───────┐
                                              │    Metrics    │
                                              │   Computation │
                                              │               │
                                              │ Sharpe, MDD   │
                                              │ WinRate, etc  │
                                              └───────────────┘
```

---

## 4. Type System

### Trade

```python
@dataclass(frozen=True)
class Trade:
    """A single executed trade."""
    index: int           # Sequence index when trade occurred
    side: TradeSide      # BUY, SELL, or FLAT (close)
    price: float         # Execution price (USD)
    size: float          # Number of shares (always positive)
    cost: float          # Transaction cost (always >= 0)
    timestamp_ns: Optional[int] = None

    @property
    def notional(self) -> float:
        """Trade value: price × size"""

    @property
    def signed_size(self) -> float:
        """Size with direction: + for BUY, - for SELL, 0 for FLAT"""
```

### Position

```python
@dataclass(frozen=True)
class Position:
    """Current position state."""
    side: PositionSide       # LONG, SHORT, or FLAT
    size: float              # Number of shares (positive or 0)
    entry_price: float       # Average entry price
    entry_index: int         # When position opened
    unrealized_pnl: float = 0.0

    @classmethod
    def flat(cls) -> "Position":
        """Create a flat (no position) state."""

    @property
    def is_flat(self) -> bool
    @property
    def is_long(self) -> bool
    @property
    def is_short(self) -> bool
    @property
    def notional(self) -> float
```

### BacktestResult

```python
@dataclass
class BacktestResult:
    """Complete backtest output."""
    equity_curve: np.ndarray     # Shape: (N,)
    returns: np.ndarray          # Shape: (N-1,)
    positions: np.ndarray        # Shape: (N,) - signed position size
    trades: List[Trade]          # All trades (opens + closes)
    trade_pnls: np.ndarray       # P&L per round-trip (closes only)
    prices: np.ndarray           # Shape: (N,)
    predictions: np.ndarray      # Shape: (N,)
    labels: Optional[np.ndarray] # Shape: (N,) if available
    metrics: Dict[str, float]    # Computed metrics
    config_dict: Dict            # Configuration used
    initial_capital: float
    final_equity: float
    total_trades: int            # len(trades), NOT len(trade_pnls)
    start_index: int
    end_index: int

    @property
    def total_return(self) -> float
    @property
    def total_pnl(self) -> float
    @property
    def max_drawdown(self) -> float
    @property
    def n_winning_trades(self) -> int   # sum(trade_pnls > 0)
    @property
    def n_losing_trades(self) -> int    # sum(trade_pnls < 0)
    def summary(self) -> str
    def to_dict(self) -> Dict
```

### Trade P&L Calculation

```python
# Long position P&L (computed when closing)
pnl = (exit_price - entry_price) * size

# Short position P&L (computed when closing)
pnl = (entry_price - exit_price) * size

# trade_pnls stores: pnl - transaction_cost
# This is what WinRate, ProfitFactor, etc. use
```

---

## 5. Configuration

### CostConfig

```python
@dataclass
class CostConfig:
    """Transaction cost configuration (all in basis points)."""
    spread_bps: float = 1.0          # Bid-ask spread per trade
    slippage_bps: float = 0.5        # Market impact
    commission_per_trade: float = 0.0 # Fixed commission (USD)
    exchange: str = ""                # Exchange name for presets
    maker_rebate_bps: float = 0.0
    taker_fee_bps: float = 0.0

    @classmethod
    def for_exchange(cls, exchange: str) -> "CostConfig":
        """Load preset costs for an exchange (XNAS, ARCX)."""
```

### BacktestConfig

```python
@dataclass
class BacktestConfig:
    """Main backtest configuration."""
    initial_capital: float = 100_000.0
    position_size: float = 0.1
    max_position: float = 1.0
    costs: CostConfig = field(default_factory=CostConfig)
    zero_dte: ZeroDteConfig = field(default_factory=ZeroDteConfig)
    allow_short: bool = True
    fill_price: Literal["close", "midpoint"] = "close"
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trading_days_per_year: float = 252.0
    periods_per_day: float = 1000.0
    min_confidence: float = 0.0       # For readability-gated strategies
    min_agreement: float = 0.0        # For readability-gated strategies
```

### ZeroDteConfig and OpraCalibratedCosts

```python
@dataclass
class OpraCalibratedCosts:
    """IBKR-validated 0DTE option cost model."""
    commission_per_contract: float = 0.70   # IBKR median (318 fills)
    implied_vol: float = 0.40
    entry_minutes_before_close: float = 120.0

@dataclass
class ZeroDteConfig:
    """0DTE option P&L transformation configuration."""
    enabled: bool = False
    delta: float = 0.50                     # Option delta (0.50 = ATM)
    opra_costs: OpraCalibratedCosts = field(default_factory=OpraCalibratedCosts)
    contracts_per_trade: int = 1
```

### Validation Rules

| Parameter | Constraint |
|-----------|------------|
| `initial_capital` | > 0 |
| `position_size` | (0, 1] |
| `max_position` | (0, 1] |
| `position_size <= max_position` | Required |
| `stop_loss_pct` | > 0 if set |
| `take_profit_pct` | > 0 if set |
| `fill_price` | "close" or "midpoint" |

---

## 6. Strategies

### Signal Enum

```python
class Signal(IntEnum):
    SELL = -1    # Enter/increase short
    HOLD = 0     # No action
    BUY = 1      # Enter/increase long
    EXIT = 2     # Close current position
```

### Strategy ABC

```python
class Strategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier."""

    @abstractmethod
    def generate_signals(
        self,
        prices: np.ndarray,
        index: Optional[int] = None,
    ) -> SignalOutput:
        """Convert predictions to trading signals."""
```

### DirectionStrategy

Maps predictions directly to signals:
- `Up` → `Signal.BUY`
- `Down` → `Signal.SELL`
- `Stable` → `Signal.HOLD`

```python
strategy = DirectionStrategy(
    predictions=np.array([1, 0, -1, 1]),  # Up, Stable, Down, Up
    shifted=False,  # Use -1/0/1 labels (vs 0/1/2)
)
```

### ThresholdStrategy

Only trades when confidence exceeds threshold:

```python
strategy = ThresholdStrategy(
    predictions=predictions,
    probabilities=model_probs,  # Shape: (N, 3)
    threshold=0.6,  # Only trade if max prob > 60%
    shifted=True,
)
```

### ReadabilityStrategy

Trades only when the HMHP readability gate passes (agreement + confidence + spread):

```python
strategy = ReadabilityStrategy(
    predictions=predictions,        # 0=Down, 1=Stable, 2=Up
    agreement_ratio=agreement,      # HMHP cross-horizon agreement [N]
    confirmation_score=confidence,  # HMHP decoder confidence [N]
    spreads=spreads,                # Bid-ask spread bps [N]
    prices=prices,
    config=ReadabilityConfig(min_agreement=1.0, min_confidence=0.65, max_spread_bps=1.05),
    holding_policy=HorizonAlignedPolicy(hold_events=10),
)
```

### RegressionStrategy

Entry gate based on magnitude of continuous return predictions:

```python
strategy = RegressionStrategy(
    predicted_returns=predicted_returns,  # Continuous bps [N]
    spreads=spreads,
    prices=prices,
    config=RegressionStrategyConfig(min_return_bps=5.0, max_spread_bps=1.05),
)
```

### ReadabilityHybridStrategy

Combines classification direction (HMHP) with regression magnitude (Ridge):

```python
strategy = ReadabilityHybridStrategy(
    predictions=predictions,              # HMHP direction (0/1/2)
    agreement_ratio=agreement,            # HMHP readability gate
    confirmation_score=confidence,        # HMHP confidence gate
    predicted_returns=predicted_returns,  # Ridge magnitude gate
    spreads=spreads, prices=prices,
    config=ReadabilityHybridConfig(min_agreement=1.0, min_confidence=0.65, min_return_bps=5.0),
    holding_policy=HorizonAlignedPolicy(hold_events=60),
)
```

### HoldingPolicy

Controls exit timing for position-holding strategies:

- `HorizonAlignedPolicy(hold_events=N)`: Hold for exactly N events after entry
- `HoldingState`: Tracks events held, unrealized P&L, entry price
- `create_holding_policy(config_dict)`: Factory from YAML config

---

## 7. Engine

### BacktestData

```python
@dataclass
class BacktestData:
    """Input data for backtest. Supports both classification and regression signals."""
    prices: np.ndarray                              # Mid-prices [N]
    labels: Optional[np.ndarray] = None             # True class labels
    timestamps_ns: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None        # Model class predictions [N]
    spreads: Optional[np.ndarray] = None            # Bid-ask spread bps [N]
    agreement_ratio: Optional[np.ndarray] = None    # HMHP agreement [N]
    confirmation_score: Optional[np.ndarray] = None # HMHP confidence [N]
    predicted_returns: Optional[np.ndarray] = None  # Regression predictions bps [N]
    regression_labels: Optional[np.ndarray] = None  # True regression labels bps [N]

    @classmethod
    def from_signal_dir(cls, signal_dir: str) -> "BacktestData":
        """Load all .npy signal files from a directory."""
```

### ZeroDtePnLTransformer

Transforms equity backtest results into 0DTE option P&L using IBKR-calibrated costs:

```python
transformer = ZeroDtePnLTransformer(ZeroDteConfig(
    enabled=True, delta=0.50, contracts_per_trade=1,
    opra_costs=OpraCalibratedCosts(commission_per_contract=0.70, implied_vol=0.40),
))
option_result = transformer.transform(backtest_result)
# option_result.option_total_return, option_result.option_win_rate, etc.
```

### VectorizedEngine

Core backtest execution:

```python
class VectorizedEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(
        self,
        data: BacktestData,
        strategy: Strategy,
        metrics: Optional[List[Metric]] = None,
    ) -> BacktestResult:
        """Execute backtest and return results."""
```

### Backtester (Convenience Wrapper)

```python
class Backtester:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(self, data: BacktestData, strategy: Strategy) -> BacktestResult:
        """Run backtest."""

    def run_from_arrays(
        self,
        prices: np.ndarray,
        predictions: np.ndarray,
        labels: Optional[np.ndarray] = None,
        shifted: bool = False,
    ) -> BacktestResult:
        """Convenience method."""
```

### Position Tracking Logic

```
For each time step:
1. Update unrealized P&L based on current price
2. Record current position
3. Process signal:
   - BUY: Close short (if any), open long
   - SELL: Close long (if any), open short (if allowed)
   - EXIT: Close current position
   - HOLD: No action
4. Update equity = cash + unrealized P&L
```

---

## 8. Metrics

### Metric ABC

```python
class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier."""

    @abstractmethod
    def compute(
        self,
        returns: np.ndarray,
        context: Dict[str, Any],
    ) -> Mapping[str, float]:
        """Compute metric from returns."""
```

### Available Metrics

| Category | Metric | Formula |
|----------|--------|---------|
| **Returns** | `TotalReturn` | `∏(1+r) - 1` |
| | `AnnualReturn` | `(1+TR)^(PPY/N) - 1` |
| **Risk** | `SharpeRatio` | `mean(r) / std(r) × √PPY` |
| | `SortinoRatio` | `mean(r) / downside_std × √PPY` |
| | `MaxDrawdown` | `max((peak - equity) / peak)` |
| | `CalmarRatio` | `AnnualReturn / MaxDrawdown` |
| **Trading** | `WinRate` | `wins / total_trades` |
| | `ProfitFactor` | `gross_profit / gross_loss` |
| | `AverageWin` | `mean(winning_pnl)` |
| | `AverageLoss` | `abs(mean(losing_pnl))` |
| | `PayoffRatio` | `AvgWin / AvgLoss` |
| | `Expectancy` | `WR × AvgWin - (1-WR) × AvgLoss` |
| **Prediction** | `DirectionalAccuracy` | `correct_dir / total_dir` |
| | `SignalRate` | `non_stable / total` |
| | `UpPrecision` | `TP_up / pred_up` |
| | `DownPrecision` | `TP_down / pred_down` |

### Context Keys

Metrics receive a context dict with:

| Key | Description |
|-----|-------------|
| `equity_curve` | Equity values (shape: N) |
| `trade_pnls` | P&L per round-trip trade (after costs) |
| `predictions` | Strategy signals |
| `labels` | True labels (if available) |
| `initial_capital` | Starting capital |
| `annualization_factor` | For annualization |
| `trading_days_per_year` | Default: 252 |
| `periods_per_day` | Default: 1000 |

---

## 9. Stats API

### BacktestStats (Fluent API)

```python
stats = (
    BacktestStats(result)
        .with_book_size(100_000)
        .daily()
        .compute()
)

print(stats.summary())
stats.plot()
```

### Methods

| Method | Description |
|--------|-------------|
| `.with_book_size(n)` | Set capital for normalization |
| `.daily()` | Aggregate by day |
| `.monthly()` | Aggregate by month |
| `.full()` | Use entire period |
| `.with_metrics(list)` | Add custom metrics |
| `.compute()` | Run computation |
| `.summary()` | Get text summary |
| `.plot()` | Generate matplotlib figure |

---

## 10. Data Loading

### DataLoader

Loads data exported by `feature-extractor-MBO-LOB`:

```python
loader = DataLoader(
    data_dir="path/to/exports",
    split="test",  # "train", "val", or "test"
    horizon_idx=0,  # For multi-horizon labels
)
data = loader.load()  # Returns LoadedData
```

### Contract Validation at Load Time

As of v0.1.0, `DataLoader.load()` enforces the pipeline contract:

1. **Labels mandatory**: Raises `FileNotFoundError` if `{date}_labels.npy` is missing (no silent zero-substitution)
2. **Shape alignment**: Raises `ContractError` if `sequences.shape[0] != labels.shape[0]`
3. **Contract validation** (first day only): Calls `validate_export_contract()` which checks schema version, feature count, normalization state, and provenance
4. **Normalization boundary**: The backtester operates on raw (un-normalized) features. The Rust exporter's identity normalization JSON is correct for backtesting. The trainer's normalization stats are NOT used by the backtester.

### LoadedData

```python
@dataclass
class LoadedData:
    sequences: np.ndarray      # Shape: (total_N, T, F)
    labels: np.ndarray         # Shape: (total_N,) or (total_N, H)
    prices: np.ndarray         # Shape: (total_N,) - denormalized
    day_boundaries: List[Tuple[int, int]]
    days: List[str]

    def to_backtest_data(self, horizon_idx=0) -> BacktestData:
        """Convert for backtesting."""
```

### PriceExtractor

Denormalizes prices from feature sequences:

```python
extractor = PriceExtractor(norm_params)
prices = extractor.extract_mid_prices(sequences, denormalize=True)
```

### Feature Layout (from feature-extractor-MBO-LOB)

| Index | Feature |
|-------|---------|
| 0-9 | Ask prices (10 levels) |
| 10-19 | Ask sizes (10 levels) |
| 20-29 | Bid prices (10 levels) |
| 30-39 | Bid sizes (10 levels) |
| 40 | Mid-price (derived) |
| ... | Additional derived features |

---

## 11. Reports

### Text Reports

```python
from lobbacktest.reports import generate_report, comparison_table

# Single result
report = generate_report(result, title="My Backtest")
print(report)

# Compare multiple
table = comparison_table(
    results={"Model A": result_a, "Model B": result_b},
    metrics=["TotalReturn", "SharpeRatio", "MaxDrawdown"],
)
print(table)
```

### Visualization

```python
from lobbacktest.reports import (
    plot_equity_curve,
    plot_returns_distribution,
    plot_drawdown,
    plot_comparison,
)

fig = plot_equity_curve(result)
fig = plot_drawdown(result)
fig = plot_comparison({"A": result_a, "B": result_b}, normalize=True)
```

---

## 12. Testing Patterns

### Test Categories

| Category | Purpose | Example |
|----------|---------|---------|
| **Formula** | Verify math | Hand-calculate Sharpe |
| **Edge** | Handle NaN/Inf/empty | Empty returns → 0 |
| **Boundary** | Threshold behavior | threshold ± ε |
| **Invariant** | Ensure consistency | No profit without trades |

### Example Test

```python
def test_sharpe_ratio_formula():
    """SR = mean(r) / std(r) * sqrt(periods_per_year)"""
    returns = np.array([0.01, -0.005, 0.02, 0.003])

    # Hand-calculated
    mean = np.mean(returns)
    std = np.std(returns, ddof=0)
    expected = (mean / std) * np.sqrt(252)

    metric = SharpeRatio(trading_days_per_year=252, periods_per_day=1)
    result = metric.compute(returns, {})

    assert abs(result["SharpeRatio"] - expected) < 1e-10
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_metrics/ -v

# With coverage
pytest tests/ --cov=lobbacktest
```

---

## 13. Integration with Pipeline

### Data Flow from Training

```
feature-extractor-MBO-LOB    lob-model-trainer       lob-backtester
         │                         │                       │
         │  Export sequences       │  Train model          │
         │  + labels + norm        │                       │
         ▼                         ▼                       ▼
    data/exports/            model.pt              BacktestResult
    ├── train/               checkpoints/          ├── equity_curve
    ├── val/                                       ├── trades
    └── test/                                      └── metrics
```

### Example Integration

```python
# 1. Load exported data
from lobbacktest.data import DataLoader
loader = DataLoader("data/exports/nvda_balanced", split="test")
data = loader.load()

# 2. Load model and generate predictions
import torch
model = torch.load("checkpoints/best_model.pt")
model.eval()
with torch.no_grad():
    logits = model(torch.from_numpy(data.sequences))
    predictions = logits.argmax(dim=-1).numpy()

# 3. Run backtest
from lobbacktest import Backtester, BacktestConfig, DirectionStrategy

config = BacktestConfig(
    initial_capital=100_000,
    position_size=0.1,
)
strategy = DirectionStrategy(predictions, shifted=True)
backtester = Backtester(config)
result = backtester.run(data.to_backtest_data(), strategy)

# 4. Analyze results
print(result.summary())
from lobbacktest.stats import BacktestStats
stats = BacktestStats(result).compute()
stats.plot()
```

---

## Quick Reference

### Key Imports

```python
from lobbacktest import (
    # Core
    Backtester,
    BacktestData,
    BacktestResult,
    BacktestConfig,
    CostConfig,
    # Strategies
    DirectionStrategy,
    ThresholdStrategy,
    # Metrics
    SharpeRatio,
    MaxDrawdown,
    WinRate,
    Expectancy,
    # Stats
    BacktestStats,
)

from lobbacktest.data import DataLoader
from lobbacktest.reports import plot_equity_curve
```

### Default Values

| Parameter | Default | Notes |
|-----------|---------|-------|
| `initial_capital` | 100,000 | USD |
| `position_size` | 0.1 | 10% of capital |
| `spread_bps` | 1.0 | 0.01% |
| `slippage_bps` | 0.5 | 0.005% |
| `trading_days_per_year` | 252 | Standard |
| `periods_per_day` | 1000 | ~1000 sequences/day |

---

*Last updated: March 17, 2026 (v0.2.0 — Phase 1-3b redesign: bug fixes, LabelMapping, BacktestContext, SignalManifest, ExperimentRunner)*

> **Note**: `prices.py` imports feature indices from `hft-contracts`.
> `NormalizationParams.from_json()` validates the `normalization_applied` boundary
> contract. `DataLoader.load()` enforces `validate_export_contract()` at load time,
> requires labels for all days (no silent zero-substitution), and validates
> sequence-label shape alignment via `ContractError`.

