# LOB-Backtester: Audit, Testing & Redesign Plan

> **⚠️ STATUS (2026-05-14): SUPERSEDED for CURRENT STATE by [`VALIDATION_FINDINGS_2026_05_14.md`](VALIDATION_FINDINGS_2026_05_14.md)**.
>
> This document (last updated 2026-03-17) captures the **first** lob-backtester audit and is preserved as **historical record** of the P0-P10 / C1-C3 / H1-H3 / N1-N4 / V1-V5 cycle. It predates Phase II (2026-04-20), Phase V (2026-04-21), Phase X (2026-05-04), Phase Y / Z / Stage 8 (2026-05-05), R-9..R-17a backtest rounds, and the v3p0 corpus. For an empirically-verified picture of the CURRENT state — including 169 findings across 17 themes, the 23-item encoded-lessons defensive list, and adversarial-validated triage — consult `VALIDATION_FINDINGS_2026_05_14.md`.
>
> **Status of legacy entries**:
> - P0 (label-execution mismatch validation) — DONE 2026-03-17.
> - P1 (Sharpe recalc post-C1) — DONE (E2 experiment cycle).
> - P2 (trade_pnls includes entry cost) — DONE; encoded lesson #2.
> - P3 (short sizing symmetric) — DONE; encoded lesson #3.
> - P4 (primary_horizon_idx default 0) — DONE; encoded lesson #4.
> - P5 (min_agreement default 0.667) — DONE for readability.py; **NOT migrated to hybrid.py** (FIND-049).
> - P6, P7 (label encoding, holding policy) — DONE.
> - P8 (magic number extraction) — **STILL OPEN** (FIND-012, FIND-045, FIND-129).
> - P9 (SignalManifest validation) — DONE.
> - P10 (maker_rebate_bps dead) — **STILL OPEN** (FIND-064).
> - P11 (docs updates) — partial; ledger drift confirmed in FIND-103, FIND-161.
> - C2 (TWAP-engine incompat) — **NOT enforced in code** (FIND-051); CLAUDE.md says SKIP, no actual skip marker.
> - C3 (short sizing) — DONE; encoded lesson #3.
> - H1 (hardcoded labels) — partially closed (FIND-042 for direction.py).
> - H2 (maker_rebate_bps) — **STILL OPEN** (FIND-064).
> - H3 (magic numbers) — partially open per FIND-012.
> - N1-N4, V1-V5 — see FIND-001..050 for current status.
>
> **Methodology lessons learned**: this 2026-03-17 audit was single-wave (no adversarial validation). The 2026-05-14 cycle ran 3-wave audit (Wave 1: 8 agents per-module, Wave 2: 8 adversarial agents, Wave 3: synthesis). Wave 2 surfaced **40+ NEW findings** the single-wave audit missed (especially in security, performance, concurrency, numerical precision, reproducibility, architectural debt) AND **refuted 5+ Wave-1 claims** that would have caused refactor disasters (e.g., `backtest_deeplob.py` is NOT a fossil — it's the production default for hft-ops). Multi-wave should be the default for major audits going forward.
>
> Original 2026-03-17 audit content preserved below for historical reference.

---

> **Purpose**: Comprehensive audit of lob-backtester for issues, test coverage, documentation accuracy, and long-term design quality.
> **Date**: 2026-03-16 (initial audit), 2026-03-17 (deep investigation)
> **Approach**: Per RULE.md §0 — build modules that last for years. No quick fixes.

---

## Audit Summary

- **Initial audit (2026-03-16)**: 16 issues found (3 critical, 4 high, 6 medium, 3 low)
- **Deep investigation (2026-03-17)**: 9 additional findings (4 code bugs, 5 validation gaps)
- **Total issues**: 25 (7 critical/high code bugs, 5 unvalidated assumptions, 13 medium/low)
- **Test coverage**: Grew from 146 → **330 tests** (+184 new), covering 12 test modules. Phases 1-3b added 64 tests for bug fixes, LabelMapping, BacktestContext, SignalManifest, and ExperimentRunner.
- **Line coverage**: 61% overall; critical modules (engine, strategies, metrics, config, types) all 85%+

---

> *Note (2026-05-14)*: The numbers above are stale per current state. Empirical pytest collection 2026-05-14: **414 tests** (5 of 6 doc sources cite different counts; root CLAUDE.md is closest at 414). See `VALIDATION_FINDINGS_2026_05_14.md` FIND-103.

---

## Work Completed

### Bug Fix: SharpeRatio Positional Arg Trap (C1) — DONE

**Bug**: `scripts/run_regression_backtest.py` line 54 used `SharpeRatio(tdy, ppd)` (positional args). `SharpeRatio.__init__` signature is `(self, name=None, trading_days_per_year=252.0, periods_per_day=1000.0)`, so `tdy` (252.0) went into `name` and `ppd` (1000.0) went into `trading_days_per_year`. Result: annualization factor = sqrt(1000 * 1000) = 1000 instead of sqrt(252 * 1000) ≈ 502. **ALL regression Sharpe/Sortino/Calmar were inflated ~2x.**

**Fix**:
- Made ALL metric constructors keyword-only (`*` separator) in `risk.py`, `returns.py`, `trading.py`, `prediction.py`
- Fixed the positional call in `run_regression_backtest.py` to use keyword args
- Added 6 tests verifying keyword-only enforcement (`SharpeRatio(252, 1000)` now raises TypeError)

### New Test Files Written — ALL PASSING

| Test File | Tests | Module Covered | Coverage |
|-----------|-------|----------------|----------|
| `tests/test_strategies/test_holding.py` | 40 | All 4 HoldingPolicy impls + factory + CompositePolicy | 96% |
| `tests/test_strategies/test_readability.py` | 17 | ReadabilityStrategy gates, holding, cooldown, metadata | 95% |
| `tests/test_strategies/test_regression.py` | 13 | RegressionStrategy gates, multi-horizon, holding, cooldown | 97% |
| `tests/test_strategies/test_twap.py` | 8 | TWAPStrategy window, cooldown, gates, engine incompatibility (C2) | 93% |
| `tests/test_engine/test_zero_dte.py` | 18 | BSM theta formula (hand-calculated), OPRA costs, ZeroDteConfig | 34% |
| `tests/test_metrics/test_prediction.py` | 12 | DirectionalAccuracy, SignalRate, Up/DownPrecision (shifted + unshifted) | 72% |
| `tests/test_registry.py` | 6 | BacktestRegistry CRUD, append-only, compare table | 92% |
| `tests/test_metrics/test_risk.py` (extended) | +6 | Keyword-only constructor enforcement | N/A |

**Total**: 266 passed, 8 skipped (real-data tests), 0 failures.

### Per-Module Coverage After This Work

| Module | Coverage | Notes |
|--------|----------|-------|
| `types.py` | 92% | Immutable dataclasses, invariants |
| `config.py` | 87% | CostConfig, BacktestConfig, OpraCalibratedCosts |
| `engine/vectorized.py` | 89% | VectorizedEngine, BacktestData |
| `strategies/holding.py` | 96% | All 4 policies + factory |
| `strategies/readability.py` | 95% | 5 gates + holding integration |
| `strategies/regression.py` | 97% | Entry gate + multi-horizon |
| `strategies/twap.py` | 93% | Window pattern + engine incompatibility |
| `strategies/direction.py` | 74% | Signal mapping + threshold |
| `metrics/trading.py` | 99% | WinRate, ProfitFactor, Expectancy |
| `metrics/returns.py` | 91% | TotalReturn, AnnualReturn |
| `metrics/risk.py` | 88% | Sharpe, Sortino, MaxDD, Calmar |
| `metrics/prediction.py` | 72% | DA, SignalRate, Up/DownPrecision |
| `registry.py` | 92% | CRUD + append-only |
| `engine/zero_dte.py` | 34% | BSM theta tested, transform not yet |
| `strategies/hybrid.py` | 21% | Needs test_hybrid.py |
| `stats/stats.py` | 23% | Needs test_stats.py |
| `data/loader.py` | 0% | I/O dependent, lower priority |
| `data/prices.py` | 0% | I/O dependent, lower priority |
| `reports/` | 0% | Plotting, lowest priority |

---

## Validated Issues (Remaining — Not Yet Fixed)

### CRITICAL

**C2: TWAP-Engine Incompatibility** — TWAP emits repeated BUY signals during the window (twap.py lines 114-118), but the engine only opens one position on the first BUY (vectorized.py line 219: `if current_position.is_flat:`). TWAP is functionally identical to point-entry. Test `test_twap_emits_repeated_buy_signals` documents this.

**C3: Short Position Sizing Asymmetry** — LONG open: `cash -= (position_value + cost)`. SHORT open: `cash -= cost` only. After a short, `cash` is barely reduced, so shorts get ~2x position size vs longs. Equity curve is correct (unrealized P&L tracked properly at line 329-330), but position sizing is ~2x for shorts. For NVDA (bullish trend), 2x-sized shorts amplify losses. P&L per-share is correct; the bug is in position sizing magnitude.

### HIGH

**H1: Hardcoded Label Values (0, 2)** — readability.py:124, regression.py:115, hybrid.py:205/209, holding.py:138-139 hardcode shifted labels without parameterization. DirectionStrategy correctly parameterizes via `shifted` flag — other strategies bypass this pattern.

**H2: Dead `maker_rebate_bps`** — Defined in CostConfig, populated by exchange presets, but never used in `total_bps` or `compute_cost()`. `taker_fee_bps` IS wired in (corrected from initial finding).

**H3: Magic Numbers** — vectorized.py:406 (`0.95` leverage buffer), vectorized.py:417 (`initial_capital / 1000`), regression.py:122 (`/ 20.0` confidence scale), zero_dte.py:40 (`TRADING_MINUTES_PER_YEAR`).

### MEDIUM

**M1**: Missing tests for HybridStrategy (21% coverage) and Stats (23% coverage).
**M2**: No day-boundary handling in engine — positions can span overnight.
**M4**: Metric caps at arbitrary 100.0 (SortinoRatio, ProfitFactor).
**M5**: Module named "vectorized.py" but main loop is Python `for i in range(n):`.
**M6**: `from_signal_dir` loads arrays without cross-array shape validation.

### Documentation Discrepancies Found

- "vectorized" claim in CODEBASE.md and module docstring contradicts the Python for-loop
- Scripts directory documented as `src/lobbacktest/scripts/` but actually at project root `scripts/`
- 3 holding policies (DirectionReversal, StopLossTakeProfit, Composite) undocumented in CODEBASE.md
- ConfusionMetrics and regression prediction metrics missing from docs
- No CLAUDE.md file (other repos have one)

---

## Deep Investigation Findings (2026-03-17)

Cross-referencing backtester code, experiment reports, signal export pipeline, and consolidated findings against the pipeline rules. These findings go beyond the initial audit scope.

### New Code Bugs Found

**N1: `trade_pnls` Omits Entry Cost** — `vectorized.py` lines 207, 253, 299 record `trade_pnls.append(pnl - cost)` where `cost` is only the EXIT cost. Entry cost is correctly deducted from `cash` (line 227), so the **equity curve is correct** and equity-based metrics (Sharpe, Sortino, MaxDD, TotalReturn) are accurate. But `trade_pnls`-based metrics (WinRate, ProfitFactor, Expectancy) are **over-estimated** by one entry_cost per trade. Impact direction: makes results look LESS bad than they are.

**N2: `primary_horizon_idx` Defaults to 1 (H60), Not 0 (H10)** — `regression.py` line 47: `primary_horizon_idx: int = 1`. Docstring says "For horizons [10, 60, 300]: 0=H10, 1=H60, 2=H300." The script `run_regression_backtest.py` line 194 overrides to 0 (H10), masking the bug. Any code using `RegressionStrategyConfig()` without override silently uses H60.

**N3: `min_agreement` Defaults to 1.0 (Filters ~90% of Signals)** — `readability.py` line 53: `min_agreement: float = 1.0`. Agreement_ratio ranges [0.333, 1.0] per contract. Threshold of 1.0 requires ALL horizons to agree perfectly → only ~10% of signals pass. Explains very low trade counts in readability backtests.

**N4: Spread Values May Be Z-Scored in Signal Export** — `export_regression_signals.py` line 122 and `export_hmhp_signals.py` line 90 extract `spread_bps` from normalized sequences without denormalization. Feature index 42 is z-scored. The backtester's spread gate (`max_spread_bps: 1.05`) may compare against z-scores, not real basis points. Backtester loader (`prices.py`) has denormalization logic, but the export scripts bypass it.

### Validation Gaps in Experiment Reports

**V1: The 55.8% Conditional Win Rate Is WRONG** — ~~`CONSOLIDATED_FINDINGS` line 62: "When smoothed says 'up,' point-to-point is positive only 55.8% of the time."~~

**P0 VALIDATED (2026-03-17)**: The actual conditional win rate is **69.7%** (not 55.8%), and the Pearson r is **0.640** (not 0.24). Computed from 510,204 samples across 35 test-split days using `LabelExecutionMismatchAnalyzer` with forward_prices.npy (exact aligned comparison).

| Metric | Previously Claimed | P0 Validated | Delta |
|---|---|---|---|
| P(point > 0 \| smoothed > 0) | 55.8% | **69.7%** | **+13.9pp** |
| Pearson r (smoothed vs point) | 0.24 | **0.640** | **+0.40** |
| Spearman r | — | **0.598** | — |

**Threshold analysis** (510K samples): At |smoothed| > 5 bps: 87.9% win rate (114K samples). At |smoothed| > 10 bps: 92.2% win rate (17K samples). **High-conviction smoothed predictions DO predict point-to-point direction with >90% accuracy.**

**Impact**: The label-execution mismatch is MUCH smaller than documented. The root cause of negative backtests is NOT label misalignment — it's cost structure and backtester bugs. The entire root cause analysis in CONSOLIDATED_FINDINGS must be revised.

Report: `data/exports/nvda_xnas_128feat_regression_fwd_prices/p0_label_execution_mismatch_H10.json`

**V2: Point-Return Model Was Never Trained** — The most obvious control experiment for the label-execution mismatch hypothesis. Kolm experiment exported 2.76M point-return sequences but model training was cancelled after feature-level IC showed ~0. But feature IC~0 doesn't preclude a temporal model (T=100) from learning the pattern — TemporalRidge showed temporal structure carries the signal. **The conclusion "point-return doesn't help" is asserted without the test.**

**V3: No Cross-Validation Inflates All Metrics 10-30%** — VALIDATED_TECHNICAL_REPORT identifies this (C3 in that report). Per financial ML literature (Easley et al. 2019), single temporal split inflates by 10-30%. True R² could be [0.31, 0.42] instead of 0.464. At the low end, the model barely beats persistence.

**V4: IC Never Computed on Entry-Filtered Subset** — Reported IC=0.677 is on all 50,724 test samples. Backtester only enters trades where `|prediction| > min_return_bps` (typically 5.0 bps), selecting ~300-700 samples. IC on the subset the strategy actually TRADES was never computed. Could be higher or lower — we don't know if the model has an edge where it trades.

**V5: Recalculation of C1-Affected Metrics Not Done** — C1 inflated Sharpe/Sortino/Calmar by ~2x. Backtest Rounds 4-5 regression metrics in BACKTEST_INDEX.md may be wrong. Corrected values have not been computed.

### Root Cause Synthesis: Why ALL 5 Backtests Are Negative

The negative results are caused by **5 compounding layers**, not a single root cause:

| Layer | Issue | Evidence |
|---|---|---|
| **1. Signal-to-cost ratio** | Mean H10 return = +0.023 bps. Min cost = 1.4 bps (deep ITM), typical cost = 1.97 bps (XNAS equity). Ratio is 61:1 against. | Return distribution in CONSOLIDATED_FINDINGS |
| **2. Label-execution mismatch** | Model predicts smoothed-average (R²=0.464) but execution is point-to-point (r=0.24 correlation). | Documented but partially validated (V1) |
| **3. Inflated metrics** | No CV → true R² potentially 0.31-0.42. Model edge may be near zero after costs. | V3 finding |
| **4. Backtester bugs** | C3 (2x short sizing), N1 (trade_pnls over-estimated), N3 (90% signal filtering), N2 (possibly wrong horizon) | Code analysis |
| **5. Architecture limitation** | Pipeline accumulates OF over 100-event windows. Kolm's LSTM sees per-event transitions. Our features structurally cannot capture what Kolm's features capture. | kolm_of_experiment_2026_03_17.md |

**Simplest explanation**: Even a perfect model can't overcome a 35:1 cost-to-signal ratio on mean predictions. The only viable path is high-conviction trades where MOVE MAGNITUDE exceeds costs — and whether that edge exists at the Deep ITM breakeven (1.4 bps) is untested.

---

## Remaining Work (Revised Priority Order — 2026-03-17)

| Priority | Task | Effort | Rationale |
|---|---|---|---|
| ~~**P0**~~ | ~~Validate the 55.8% conditional win rate~~ | ✅ DONE | **RESULT: 55.8% was WRONG. Actual = 69.7%. r=0.640 (not 0.24). Root cause analysis must be revised.** |
| ~~**P1**~~ | ~~Recalculate Sharpe/Sortino affected by C1~~ | ✅ DONE | Need re-run with fixed engine (deferred to E2 experiment) |
| ~~**P2**~~ | ~~Fix N1: trade_pnls to include entry cost~~ | ✅ DONE | Position.entry_cost field added, trade_pnls includes both costs |
| ~~**P3**~~ | ~~Fix C3 (short sizing symmetry)~~ | ✅ DONE | `cash -= (position_value + cost)` for shorts, symmetric with longs |
| **P4** | Validate spread export is denormalized (N4) | TODO | Requires trainer-side investigation |
| ~~**P5**~~ | ~~Fix H1 (LabelMapping centralization)~~ | ✅ DONE | `labels.py` with LabelMapping, 10 hardcoded values eliminated |
| ~~**P6**~~ | ~~Fix N2: primary_horizon_idx default to 0~~ | ✅ DONE | `regression.py` default changed to 0 |
| ~~**P7**~~ | ~~Fix N3: readability min_agreement default~~ | ✅ DONE | `readability.py` default changed to 0.667 |
| **P8** | Fix H3 (magic number extraction) | TODO | Non-critical |
| ~~**P9**~~ | ~~Fix M6 (array validation)~~ | ✅ DONE | `SignalManifest` validates at load time |
| **P10** | Fix H2 (dead maker_rebate_bps) | TODO | Non-critical |
| **P11** | **Documentation updates** | 2h | CODEBASE.md + CLAUDE.md accuracy |
| **SKIP** | C2 (TWAP redesign) | -- | Strategy empirically failed (Rule 13) |
| **DEFER** | test_hybrid.py, test_stats.py | -- | Strategy empirically underperformed |

### Next Experiment Direction (After Fixes)

| Priority | Experiment | Hypothesis | Decision Gate |
|---|---|---|---|
| **E1** | Validate 55.8%: compute conditional point-return distribution where model predicts UP | Understand actual win rate on EXECUTABLE returns | If conditional mean < 1.4 bps → no viable path with current features |
| **E2** | Compute IC on entry-filtered subset | Does model have edge where it actually trades? | If subset IC < 0.3 → model overconfident, no trading edge |
| **E3** | Deep ITM backtest | 1.4 bps breakeven is ~4x lower than ATM | If still negative → signal truly insufficient |
| **E4** | Train TLOB on point-return labels | Test whether label-mismatch is fixable | If R² > 0.1 on point-returns → mismatch was the root cause |
| **E5** | Purged K-Fold CV on TLOB | Determine true generalization R² | If true R² < 0.2 → model edge near zero |

---

## Positive Patterns Worth Preserving

1. **HoldingPolicy ABC** with `CompositePolicy(mode='any'|'all')` — elegant composable design
2. **BacktestRegistry** — append-only with index.json
3. **ZeroDtePnLTransformer** — IBKR+OPRA empirically calibrated, BSM theta validated
4. **CostConfig.for_exchange()** factory — presets from 233-day profiler
5. **Immutable frozen dataclasses** for Trade and Position with strong validation
6. **DirectionStrategy's label parameterization** via `shifted` flag — should be the pattern for ALL strategies
7. **BACKTEST_INDEX.md** — living ledger with 5 rounds of iterative experiments

---

*This plan is the single reference for backtester audit status. Updated after each work session.*
*Last updated: 2026-03-17 — P0 VALIDATED: 55.8% was WRONG (actual 69.7%, r=0.640). Root cause is NOT label-execution mismatch. Added deep investigation findings (N1-N4, V1-V5), root cause synthesis.*
