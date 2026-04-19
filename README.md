# LOB-Backtester

Standalone backtesting library for evaluating LOB prediction models trained with `lob-model-trainer`.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Version**: 0.1.0 | **Tests**: 353 (345 passed + 8 skipped)

---

## Overview

This library backtests direction + regression prediction models on Limit Order Book (LOB) signals. It consumes `signal_metadata.json` + signal NPY files from `lob-model-trainer` and produces `BacktestResult`, equity curves, and IBKR-calibrated 0DTE options P&L via the `ZeroDtePnLTransformer`.

### Key Features

- **Per-Sample Engine** — position tracking with numpy-based metrics; fast for ~50K-sample backtests. (Module is named `engine/vectorized.py` for historical reasons — actual algorithm is a per-sample loop.)
- **IBKR-Calibrated 0DTE Options** — real-fill cost model (316 NVDA fills): ATM Call half-spread $0.015, ATM Put $0.010, Deep ITM $0.005; commission $1.40 round-trip; BSM theta per-minute; breakevens 4.9 / 3.8 / 1.4 bps.
- **Metric ABC Pattern** — composable metrics across 5 modules (`returns`, `risk`, `trading`, `prediction`, `regression_prediction`).
- **LabelMapping SSoT** — centralized label encoding (Phase 2a); strategies accept `label_mapping: Optional[LabelMapping]` and default to `SHIFTED_MAPPING`.
- **HoldingPolicy composability** — 4 exit policies + `CompositePolicy` (mode='any'|'all').
- **Phase 3b `ExperimentRunner`** — YAML-config orchestration (load → validate → run → register → aggregate).
- **Phase 4 4c.4 `SignalManifest.feature_set_ref`** — tracks the Phase 4 FeatureSet registry entry; propagated from trainer through the load-time contract validator.
- **Phase 6 6B.5 contract-plane co-move** — `SignalManifest` canonical home in `hft_contracts.signal_manifest`; this repo's `data/signal_manifest.py` is a thin re-export shim (calendar removal deadline 2026-10-31).
- **Phase 2b typed `BacktestContext`** — typed dataclass with dict-style backward compat.

---

## Installation

```bash
cd lob-backtester
pip install -e ".[dev]"
```

---

## Quick Start

**Preferred** — via `hft-ops` orchestrator (single YAML manifest, validated cross-module consistency, ledger tracking):

```bash
cd hft-ops
hft-ops run experiments/e5_60s_huber_cvml_unified.yaml
# → validation → training → post_training_gate → signal_export → backtesting
```

**Direct Python API** (programmatic):

```python
from lobbacktest import Backtester, BacktestConfig
from lobbacktest.config import CostConfig
from lobbacktest.data import BacktestData
from lobbacktest.strategies import DirectionStrategy

# Load signals from lob-model-trainer (validates feature_set_ref, shapes, NaN/Inf at load time)
data = BacktestData.from_signal_dir(
    "../lob-model-trainer/outputs/experiments/e5_60s/signals/test/",
    validate=True,
)

strategy = DirectionStrategy(predictions=data.predictions)

config = BacktestConfig(costs=CostConfig.for_exchange("XNAS"))
backtester = Backtester(strategy, data, config)
result = backtester.run()
print(result.total_return, result.sharpe_ratio, result.win_rate)
```

**CLI Scripts** (under `scripts/` — 6 scripts):

| Script | Purpose |
|--------|---------|
| `scripts/run_regression_backtest.py` | Regression signal backtest (continuous bps predictions) |
| `scripts/run_readability_backtest.py` | Classification signal backtest (HMHP agreement + confidence) |
| `scripts/run_spread_signal_backtest.py` | Spread-based signal backtest |
| `scripts/param_sweep.py` | Parameter grid sweep |
| `scripts/backtest_deeplob.py` | DeepLOB architecture backtest |
| `scripts/e5_regime_filter_test.py` | E5 regime-filter diagnostic |

```bash
python scripts/run_regression_backtest.py \
  --signals ../lob-model-trainer/outputs/experiments/e5_60s/signals/test/ \
  --name e5_round7 --exchange XNAS
```

---

## Data Contract

### Input — Signal Directory

The backtester expects a signal directory emitted by `lob-model-trainer`:

```
signals/test/
├── predictions.npy          # Classification: [N] int32 {0,1,2}
├── predicted_returns.npy    # Regression: [N] or [N,H] float64 basis points
├── calibrated_returns.npy   # (Optional) variance-matched predictions
├── regression_labels.npy    # [N] or [N,H] float64 bps (ground truth)
├── labels.npy               # [N] int32 {0,1,2} (shifted, classification)
├── prices.npy               # [N] float64 USD
├── spreads.npy              # [N] float64 basis points
├── agreement_ratio.npy      # (HMHP) [N] float64 [0.333, 1.0]
├── confirmation_score.npy   # (HMHP) [N] float64 [0, 0.667]
└── signal_metadata.json     # Experiment + checkpoint + horizons + feature_set_ref
```

`SignalManifest.validate()` is called at load time (canonical in `hft_contracts`) — shape alignment, NaN/Inf checks, metadata-trainer cross-consistency, `feature_set_ref.content_hash` regex. Fail-fast.

### Label Encoding — SHIFTED_MAPPING (canonical)

The backtester uses the PyTorch convention via `LabelMapping`:

| Value | Meaning |
|-------|---------|
| 0 | Down (price decreased above threshold) |
| 1 | Stable (price within threshold) |
| 2 | Up (price increased above threshold) |

This matches the trainer's `CrossEntropyLoss`-compatible output (the `{-1, 0, 1}` raw TLOB encoding is shifted +1 at dataset construction time).

```python
from lobbacktest.labels import LabelMapping, SHIFTED_MAPPING

mapping = SHIFTED_MAPPING  # {down: 0, stable: 1, up: 2}
assert mapping.up == 2 and mapping.down == 0

# Predicates — NEVER hardcode 0/1/2 (Phase 2a SSoT)
assert mapping.is_bullish(2) and not mapping.is_bullish(0)
assert mapping.is_directional(2) and not mapping.is_directional(1)
```

---

## Configuration

```python
from lobbacktest.config import BacktestConfig, CostConfig, ReadabilityConfig, HoldingConfig

config = BacktestConfig(
    initial_capital=100_000.0,
    costs=CostConfig.for_exchange("XNAS"),   # per-exchange cost preset (module-level _EXCHANGE_PRESETS, 6A.6)
    min_agreement=0.667,                     # HMHP cross-horizon agreement (P5 default)
    readability=ReadabilityConfig(min_confidence=0.65, max_spread_bps=1.05),
    holding=HoldingConfig(type="horizon_aligned", hold_events=10),
)
```

### 0DTE Options Pricing (IBKR-Calibrated)

```python
from lobbacktest.engine.zero_dte import ZeroDtePnLTransformer
from lobbacktest.config import OpraCalibratedCosts, ZeroDteConfig

costs = OpraCalibratedCosts.deep_itm()   # half_spread=0.005, commission=0.70, theta minimal
# or: OpraCalibratedCosts() — ATM defaults (half_spread=0.015, commission=0.70, IV=0.40)

transformer = ZeroDtePnLTransformer(
    config=ZeroDteConfig(costs=costs, contract_multiplier=100),
)
zero_dte_result = transformer.transform(backtest_result)
# → option_total_return, win_rate, spread_cost, commission_cost, theta_cost
```

IBKR constants calibrated from 316 real NVDA option fills (provenance in `engine/zero_dte.py` docstring + `IBKR-transactions-trades/COST_AUDIT_2026_03.md`). Breakevens: ATM Call 4.9 bps, ATM Put 3.8 bps, Deep ITM 1.4 bps.

---

## Module Structure

```
lob-backtester/
├── src/lobbacktest/
│   ├── __init__.py              # Public API
│   ├── labels.py                # LabelMapping SSoT (Phase 2a)
│   ├── context.py               # BacktestContext typed+dict hybrid (Phase 2b)
│   ├── config.py                # BacktestConfig, CostConfig, ZeroDteConfig, OpraCalibratedCosts
│   ├── types.py                 # Trade, Position (with entry_cost), BacktestResult
│   ├── experiment.py            # ExperimentRunner (Phase 3b)
│   ├── registry.py              # BacktestRegistry
│   │
│   ├── engine/
│   │   ├── vectorized.py        # VectorizedEngine, Backtester (per-sample loop, name is historical)
│   │   └── zero_dte.py          # ZeroDtePnLTransformer (IBKR-calibrated 0DTE P&L)
│   │
│   ├── strategies/
│   │   ├── base.py              # Strategy ABC, Signal enum, SignalOutput
│   │   ├── direction.py         # DirectionStrategy, ThresholdStrategy
│   │   ├── readability.py       # ReadabilityStrategy (HMHP agreement + confidence gate)
│   │   ├── regression.py        # RegressionStrategy (magnitude gate on continuous predictions)
│   │   ├── hybrid.py            # ReadabilityHybridStrategy (classification + regression)
│   │   ├── holding.py           # HoldingPolicy ABC + 4 implementations + CompositePolicy
│   │   └── twap.py              # TWAPStrategy (SKIP — empirically failed, C2 incompatibility)
│   │
│   ├── data/
│   │   ├── loader.py            # DataLoader for trainer exports
│   │   ├── prices.py            # PriceExtractor (denormalize from features)
│   │   └── signal_manifest.py   # Phase 6 6B.5 shim — canonical in hft_contracts.signal_manifest
│   │                            # (removal deadline 2026-10-31; DeprecationWarning emitted per symbol)
│   │
│   ├── metrics/
│   │   ├── base.py              # Metric ABC
│   │   ├── returns.py           # TotalReturn, AnnualReturn
│   │   ├── risk.py              # SharpeRatio, SortinoRatio, MaxDrawdown, CalmarRatio
│   │   ├── trading.py           # WinRate, ProfitFactor, AverageWin, AverageLoss, PayoffRatio, Expectancy
│   │   ├── prediction.py        # DirectionalAccuracy, SignalRate, UpPrecision, DownPrecision, ConfusionMetrics
│   │   └── regression_prediction.py  # PredictionMSE, PredictionCorrelation, PredictionIC
│   │
│   └── stats/
│       └── stats.py             # BacktestStats fluent API
├── scripts/                     # 6 runnable scripts (see Quick Start)
├── configs/                     # YAML experiment configs
├── tests/                       # 353 tests (345 passed + 8 skipped)
└── BACKTEST_INDEX.md            # Living backtest ledger
```

---

## Recent Fixes (Phase 0-7)

| Fix | Impact | Phase |
|---|---|---|
| P2: `trade_pnls` includes entry cost | WinRate / ProfitFactor accurate | 0-3 |
| P3: Short sizing symmetric with longs | Shorts no longer 2x oversized | 0-3 |
| P4: `primary_horizon_idx` defaults to 0 (H10) | Was silently using H60 | 0-3 |
| P5: `min_agreement` defaults to 0.667 | Was 1.0 filtering 90% of signals | 0-3 |
| Phase 2a: LabelMapping centralization | 10 hardcoded label values eliminated | 0-3 |
| Phase 2b: Typed BacktestContext | Dict → typed context (zero breaking) | 0-3 |
| Phase 3a: SignalManifest validation | Signal exports validated at load time | 0-3 |
| Phase 3b: ExperimentRunner orchestration | YAML config → automated experiment flow | 0-3 |
| Phase 4 4c.4: `SignalManifest.feature_set_ref` | FeatureSet registry propagation from trainer | 4 |
| Phase 6 6A.6: `_EXCHANGE_PRESETS` module-level | Dead class-var removed; single SSoT | 6 |
| Phase 6 6A.9: `_CONTENT_HASH_RE` symmetry | Producer-consumer regex parity (imported from `hft_contracts` SSoT) | 6 |
| Phase 6 6B.5: SignalManifest co-move to `hft_contracts` | Cross-module contract at contract plane; shim preserves imports | 6 |
| Phase 6 final hygiene: shim DeprecationWarning | Lazy `__getattr__` emits once per symbol | 6 |
| Phase 7 post-validation I: calendar shim deadline | "version 0.4.0" → 2026-10-31 calendar `_REMOVAL_DATE` | 7 |

See `BACKTEST_INDEX.md` for the living backtest ledger.

---

## Key Constraints (per root CLAUDE.md)

| Constraint | Value | Reason |
|---|---|---|
| Labels | `{0=Down, 1=Stable, 2=Up}` SHIFTED_MAPPING | PyTorch CrossEntropyLoss convention |
| Costs | `CostConfig.for_exchange("XNAS")` | IBKR-calibrated from 316 real fills |
| Position tracking | `entry_cost` on Position | Accurate trade P&L (P2 fix) |
| Metrics | Keyword-only constructors | Prevents positional-arg traps |
| `total_trades` vs `len(trade_pnls)` | `total_trades = len(trades)` (opens+closes); win_rate uses `len(trade_pnls)` (round-trip closes only) | Critical — silently double-counts if conflated |
| Engine | Per-sample loop | Name `vectorized.py` is historical |

---

## Running Tests

```bash
cd lob-backtester
pytest tests/ -v     # expect 345 passed + 8 skipped (real-data gates)
pytest tests/test_engine/test_vectorized.py -v   # engine-only
pytest tests/test_signal_manifest.py -v          # shim + feature_set_ref
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `hft-contracts` | SignalManifest canonical, label contracts, canonical_hash SSoT |
| `numpy>=1.24.0` | Array operations |
| `matplotlib>=3.7.0` | Plots (optional — in `reports/plots.py`) |
| `pyyaml>=6.0` | Config parsing |

---

## Related Libraries

| Library | Role |
|---------|------|
| `hft-ops` | Experiment orchestrator (preferred entry via `hft-ops run <manifest>`) |
| `hft-contracts` | Contract plane — SignalManifest canonical, label_factory, canonical_hash |
| `lob-model-trainer` | Signal producer (emits `signal_metadata.json`, `predictions.npy`, etc.) |
| `feature-extractor-MBO-LOB` | Rust feature extractor — produces the export the trainer consumes |

---

## Documentation

- `CODEBASE.md` — detailed module reference
- `BACKTEST_INDEX.md` — living backtest ledger (round-by-round results)
- Root pipeline docs: `CLAUDE.md`, `PIPELINE_ARCHITECTURE.md`, `DOCUMENTATION_INDEX.md`

---

*Last updated: 2026-04-20 (Phase 7 Stage 7.4 Round 5)*
