# LOB-Backtester

Standalone backtesting library for evaluating LOB prediction models trained with `lob-model-trainer`.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Version**: 0.1.0 | **Tests**: 666 collected (verify exact pass/skip: `pytest --collect-only -q | tail -1`)

> **Pipeline scope (2026-06-02).** This module is part of an **intraday trading research pipeline** — an experiment-first platform for discovering and validating *any* profitable **intraday** trading edge (no overnight positions), across approach classes (microstructure/HFT, scalping, intraday momentum, intraday statistical arbitrage, …) and instruments (equities, futures, same-day options). The pipeline *originated* as a high-frequency NVDA MBO/LOB microstructure system — that origin explains the "HFT" / "LOB" / "MBO" naming here — and that microstructure-direction program is now one (largely-closed) track among many. **Names are historical; the mission is general.** This module's role: the P&L backtester — a generic single-asset linear long/flat/short engine (`VectorizedEngine`) with an optional, separable 0DTE-options overlay (`ZeroDtePnLTransformer`) and IBKR-calibrated costs; reusable for equity/futures intraday directional P&L. For the full mission + approach taxonomy + capability-readiness boundary, see root `CLAUDE.md` §Research Scope & Charter (+ `CROSS_ASSET_OFI_FINDINGS_AND_ISSUES_2026_06_01.md` §9).

---

## Overview

This library backtests direction + regression prediction models on Limit Order Book (LOB) signals. It consumes `signal_metadata.json` + signal NPY files from `lob-model-trainer` and produces `BacktestResult`, equity curves, and IBKR-calibrated 0DTE options P&L via the `ZeroDtePnLTransformer`.

### Key Features

- **Per-Sample Engine** — position tracking with numpy-based metrics; fast for ~50K-sample backtests. (Module is named `engine/vectorized.py` for historical reasons — actual algorithm is a per-sample loop.)
- **IBKR/OPRA-Calibrated 0DTE Options** — commission ($1.40 round-trip) validated from real NVDA IBKR fills; ATM half-spreads ($0.015 call / $0.010 put) from the OPRA CMBP-1 profiler; BSM theta per-minute; ATM breakevens 4.9 / 3.8 bps. **The Deep-ITM $0.005 half-spread → 1.4 bps breakeven is an OPTIMISTIC, UNVALIDATED assumption** (not derived from the fills — real deep-ITM spreads are materially wider, so treat the working breakeven as ~3–7 bps; see `config.py` `OpraCalibratedCosts.deep_itm()` + root FINDING-114 + the `BACKTEST_INDEX.md` freeze note). Default payoff is linear-delta (ATM-call-only); optional BSM call/put re-pricing via `ZeroDteConfig.payoff_model='bsm'` (`engine/option_pricing.py`).
- **Metric ABC Pattern** — composable metrics across 4 modules (`returns`, `risk`, `trading`, `prediction`).
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

**Run paths at a glance** — three distinct ways to drive a backtest:
- **CLI scripts** (`scripts/*.py`) — what the `hft-ops` orchestrator shells out to (its `backtesting` stage runs these via subprocess); the production path.
- **`ExperimentRunner.from_yaml("configs/*.yaml")`** — the config-driven single-run/sweep path (schema in `CODEBASE.md` §5 "ExperimentRunner YAML Schema"); exercised by this repo's tests, **not** on the `hft-ops` path.
- **Direct Python API** — programmatic (`Backtester(config).run(data, strategy)`).

> Annualization caveat: the script paths and the `ExperimentRunner` path derive the annualization factor from the sampling cadence (`periods_per_day` / `zero_dte.bin_seconds` → `BacktestConfig.resolved_periods_per_day`) and were fixed **separately** across those paths (#PY-263). Set a cadence explicitly — the legacy fallback emits a `DeprecationWarning`. See the `CODEBASE.md` State-at-HEAD cycle log rather than re-deriving the saga.

**Preferred** — via `hft-ops` orchestrator (single YAML manifest, validated cross-module consistency, ledger tracking):

```bash
cd hft-ops
hft-ops run experiments/e5_60s_huber_cvml_unified.yaml
# → validation → training → post_training_gate → signal_export → backtesting
```

**Direct Python API** (programmatic):

```python
# BacktestData is re-exported at the top level (the data/ subpackage exports only DataLoader/PriceExtractor)
from lobbacktest import Backtester, BacktestData, BacktestConfig
from lobbacktest.config import CostConfig
from lobbacktest.strategies import DirectionStrategy

# Load signals from lob-model-trainer (validates feature_set_ref, shapes, NaN/Inf at load time)
data = BacktestData.from_signal_dir(
    "../lob-model-trainer/outputs/experiments/e5_60s/signals/test/",
    validate=True,
)

strategy = DirectionStrategy(predictions=data.predictions)

config = BacktestConfig(costs=CostConfig.for_exchange("XNAS"))
backtester = Backtester(config)          # __init__ takes ONLY config
result = backtester.run(data, strategy)  # run() takes (data, strategy[, metrics])
# BacktestResult exposes total_return/total_pnl/max_drawdown as properties;
# Sharpe / win-rate live in the metrics dict (default metric list includes both):
print(result.total_return, result.metrics["SharpeRatio"], result.metrics["WinRate"])
```

**CLI Scripts** (under `scripts/` — 7 scripts):

| Script | Purpose |
|--------|---------|
| `scripts/run_regression_backtest.py` | Regression signal backtest (continuous bps predictions) |
| `scripts/run_readability_backtest.py` | Classification signal backtest (HMHP agreement + confidence) |
| `scripts/run_spread_signal_backtest.py` | Spread-based signal backtest |
| `scripts/param_sweep.py` | Parameter grid sweep |
| `scripts/backtest_deeplob.py` | DeepLOB architecture backtest |
| `scripts/e5_regime_filter_test.py` | E5 regime-filter diagnostic |
| `scripts/check_backtest_index_completeness.py` | BACKTEST_INDEX completeness checker |

```bash
python scripts/run_regression_backtest.py \
  --signals ../lob-model-trainer/outputs/experiments/e5_60s/signals/test/ \
  --name e5_round7 --exchange XNAS
```

**Config-driven (`ExperimentRunner.from_yaml`)** — single-run or one-parameter sweep from a `configs/*.yaml`:

```python
from lobbacktest.experiment import ExperimentRunner

result = ExperimentRunner.from_yaml("configs/e1_deep_itm.yaml").run()  # raises until a zero_dte cadence is added — see caveat below
print(result.summary())            # markdown table across runs
print(result.best_by("TotalReturn"))
```

The YAML block/key schema, the `strategy.type` / `holding.type` enums, and the (one-parameter-at-a-time, not Cartesian) `sweep` semantic are documented in `CODEBASE.md` §5 "ExperimentRunner YAML Schema"; `configs/e1_deep_itm.yaml` + `configs/e1_atm_comparison.yaml` are `ExperimentRunner`-shaped worked examples (they carry `signals.dir` + a `sweep`), but their `zero_dte` blocks predate FIND-NEW-01 and set no `bin_seconds`/`events_per_minute` cadence — which an enabled 0DTE overlay now requires — so `.run()` raises a `ValueError` until a cadence is added (see `CODEBASE.md` §5).

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
from lobbacktest.config import BacktestConfig, CostConfig
# NOTE: ReadabilityConfig lives in lobbacktest.strategies (NOT lobbacktest.config);
# there is no `HoldingConfig` — holding is configured via a HoldingPolicy object.
from lobbacktest.strategies import ReadabilityStrategy, ReadabilityConfig
from lobbacktest.strategies.holding import HorizonAlignedPolicy

config = BacktestConfig(
    initial_capital=100_000.0,
    costs=CostConfig.for_exchange("XNAS"),   # per-exchange cost preset (module-level _EXCHANGE_PRESETS, 6A.6)
)
# Readability / holding gates are configured on the STRATEGY, not on BacktestConfig.
# BacktestConfig has no `readability`/`holding` fields, and `min_agreement` on it is
# DEPRECATED (emits DeprecationWarning; the live home is the strategy):
strategy = ReadabilityStrategy(
    predictions=data.predictions,
    agreement_ratio=data.agreement_ratio,
    confirmation_score=data.confirmation_score,
    config=ReadabilityConfig(min_confidence=0.65, max_spread_bps=1.05),
    holding_policy=HorizonAlignedPolicy(hold_events=10),
)
```

### 0DTE Options Pricing (IBKR-Calibrated)

```python
from lobbacktest.engine.zero_dte import ZeroDtePnLTransformer
from lobbacktest.config import OpraCalibratedCosts, ZeroDteConfig

costs = OpraCalibratedCosts.deep_itm()   # half_spread=0.005 (ASSUMPTION — see caveat below)
# or: OpraCalibratedCosts() — ATM defaults (half_spread=0.015, commission=0.70, IV=0.40)

# ZeroDteConfig uses `opra_costs=` (not `costs=`); the 100x contract multiplier is
# hardcoded in the P&L formula (no `contract_multiplier` field). ZeroDtePnLTransformer
# REQUIRES `events_per_minute` (FIND-NEW-01 — no silent default; 1.0 for 60s bins).
transformer = ZeroDtePnLTransformer(
    ZeroDteConfig(enabled=True, opra_costs=costs),
    events_per_minute=1.0,
)
zero_dte_result = transformer.transform(backtest_result)
# → option_total_return, option_win_rate, avg_spread_cost, avg_commission_cost, avg_theta_cost
```

IBKR commission ($1.40 round-trip) is validated from real NVDA option fills; the ATM half-spreads are from the OPRA CMBP-1 profiler (provenance in `engine/zero_dte.py` docstring + `IBKR-transactions-trades/COST_AUDIT_2026_03.md`). ATM breakevens: Call 4.9 bps, Put 3.8 bps. **The Deep-ITM 1.4 bps breakeven is an OPTIMISTIC, UNVALIDATED assumption** — the $0.005 deep-ITM half-spread is not derived from the fills (those are ATM/near-money) and real deep-ITM spreads are materially wider, so treat the working deep-ITM breakeven as ~3–7 bps (see `config.py` `OpraCalibratedCosts.deep_itm()` + root FINDING-114 + the `BACKTEST_INDEX.md` freeze note).

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
│   │   ├── vectorized.py        # VectorizedEngine, BacktestData, Backtester (per-sample loop, name is historical)
│   │   ├── zero_dte.py          # ZeroDtePnLTransformer (IBKR-calibrated 0DTE P&L; payoff_model linear_delta|bsm)
│   │   └── option_pricing.py    # BSM valuation (bs_value/bs_call/bs_put/bs_delta/bs_gamma/bs_vega; q=0, put floored at intrinsic)
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
│   │   └── prediction.py        # DirectionalAccuracy, SignalRate, UpPrecision, DownPrecision (+ ConfusionMetrics — unexported, dead code, zero callers)
│   │
│   ├── stats/
│   │   └── stats.py             # BacktestStats fluent API
│   └── reports/                 # generate_report, comparison_table + plot_* (summary.py + plots.py)
├── scripts/                     # 7 runnable scripts (see Quick Start)
├── configs/                     # YAML experiment configs
├── tests/                       # 666 collected (verify pass/skip: pytest --collect-only -q | tail -1)
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
pytest tests/ -v     # 666 collected; 16 skipped (real-data + TWAP gates), rest pass
pytest tests/test_engine/test_vectorized.py -v   # engine-only
pytest tests/test_signal_manifest.py -v          # shim + feature_set_ref
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `hft-contracts>=2.7.0` | SignalManifest canonical, label contracts, canonical_hash + atomic_io SSoT |
| `numpy>=1.24.0` | Array operations |
| `scipy>=1.10` | `scipy.stats.norm` (BSM in `scripts/run_spread_signal_backtest.py`) |
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

**Living references** (current ground truth — keep in sync with the code):
- `CODEBASE.md` — detailed module + config reference (its "State at HEAD" cycle log is the running changelog)
- `CLAUDE.md` (this repo) — module structure, design patterns, key constraints, data contract
- `BACKTEST_INDEX.md` — living backtest ledger (round-by-round results)
- `CONTRIBUTING.md` — contribution/field discipline specific to this repo
- Root pipeline docs (monorepo root): `CLAUDE.md`, `PIPELINE_ARCHITECTURE.md`, `DOCUMENTATION_INDEX.md`

**Point-in-time audit / design records** (historical — do **not** read as current state; consult the living references above for what the code does now):
- `VALIDATION_FINDINGS_2026_05_30.md` — full-module re-validation findings (verdict: core SOUND)
- `VALIDATION_FINDINGS_2026_05_14.md` — 3-wave adversarial audit findings brief (large)
- `DESIGN_CLUSTER_D1_E_2026_05_14.md` — detailed design record for the Cluster D.1 + E cycle
- `BACKTESTER_AUDIT_PLAN.md` — the first (2026-03-17) audit plan, **superseded** by the 05-14 findings doc

---

*Last updated: 2026-05-30 (#PY-263 annualization closure: R1 readability-script bin_seconds + G1a self-describing annualization key + G1b ExperimentRunner path)*
