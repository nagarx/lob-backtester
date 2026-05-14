# LOB-Backtester — Comprehensive Validation Findings (2026-05-14)

> **Status**: CANONICAL findings doc post 3-wave adversarial audit (16 cumulative agents).
> **Supersedes**: `BACKTESTER_AUDIT_PLAN.md` (2026-03-17) for current state.
> **Companion docs**: `BACKTEST_INDEX.md` (empirical history), `CLAUDE.md` (build + design), `CODEBASE.md` (technical reference).
> **Purpose**: Per-finding technical brief for the upcoming **designing phase**. No design solutions here — only validated issues with full context.
> **Scope**: full `lob-backtester` repo, all dependencies (`hft-contracts`, `hft-metrics`, `hft-statistics`), all producer + consumer surfaces (`lob-model-trainer`, `hft-ops`), and 17 backtest rounds (R1–R17a).

---

## §0 Executive Summary

The backtester is a **load-bearing, mostly-correct module weighed down by accumulating duplication, dead code, and silent-degrade hazards**. Critical correctness gaps are mostly **LATENT** (masked by other design constraints today) but block:
- multi-source signal support (off-exchange path NOT plumbed),
- automated cross-experiment comparison at scale (registry race + non-atomic writes),
- options-native trading (current path is equity → option projection only),
- multi-symbol portfolio support (engine is single-symbol),
- streaming/real-time inference (engine is batch-only).

The fix path is mostly **consolidation + fail-loud retrofits**, not rewrites. None of the recommended fixes requires breaking the **17-round empirical contract** documented in §A.

### Triage matrix (post 3-wave adversarial audit)

| Tier | Count | Notes |
|---|---|---|
| **CONFIRMED CRITICAL** | 11 | Engine accounting gap (C1+NEW-F2), Sweep YAML silent gate drop (N1), Engine single-symbol (F-A1), Options-native path absent (F-A2), Registry race (F-C1+F-C2), `BacktestStats.daily/monthly` stubs (H1), `datetime.now()` non-UTC reproducibility (F-R2) |
| **CONFIRMED HIGH** | 24 | Phase X.3 incomplete migration, `np.load` accepts pickle, Matplotlib leaks, `final_equity` epsilon at large capital, etc. |
| **CONFIRMED MEDIUM/LOW** | 35+ | Doc drift, dead config fields, fossil scripts, naming hygiene |
| **REFUTED-WRONG** | 5 | H11 parity-by-coincidence (locked by frozen-fixture test); M5 ConfusionMetrics ABC (CompositeMetric pattern); O2 BacktestRegistry retirement (orthogonal to hft-ops ledger); U2 OpraCalibratedCosts upstream scope; backtest_deeplob.py is NOT fossil (production default) |
| **REFUTED-OVERSTATED** | 9 | C6 off-exchange (explicit boundary); C7 shim flood (calendar-tracked); U1 BSM upstream (no consumer); O1 ExperimentRunner retire (Python API legitimate); etc. |
| **RECLASSIFIED** | 7 | C8 collision (latent not active); S1 fills 316/318 (documented dual-reconciliation); etc. |
| **NEW findings** (audit missed) | 40+ | Security (4), Concurrency (3+), Numerical fp (5), Reproducibility (3), Memory leaks (3), Architectural debt (8), Tests (3+), Doc drift (7), Hidden (5) |

### Priority order for design phase (impact × likelihood × reversibility)

| Rank | Finding | Reason |
|---|---|---|
| 1 | **FIND-001** Engine end-of-data accounting + FIND-002 invariant gap | Active in latent strategies; silently diverges equity vs option mode |
| 2 | **FIND-070** YAML `backtest.min_agreement/min_confidence` silently dropped by `ExperimentRunner` | Production YAMLs claim "ALL agree" semantic; run silently as "2/3" — already-corrupted real experiments |
| 3 | **FIND-090** Registry race + non-atomic writes (3 sites) | SIGKILL during sweep corrupts ledger; multi-process sweeps will race |
| 4 | **FIND-110** `np.load` allows pickle by default | RCE-via-malicious-`.npy` security hazard |
| 5 | **FIND-120** Architectural: single-symbol + no options-native + no streaming | Blocks 3 distinct future expansion paths |

---

## §1 Methodology — 3-wave audit cycle

### Wave 1 (initial coverage)
Eight parallel agents covered: engine, strategies, metrics, data + signal manifest, config + experiment + registry, scripts + tests, cross-repo consolidation, BACKTEST_INDEX consistency. Produced ~50 findings.

### Wave 2 (adversarial refutation)
Eight parallel agents tasked with **REFUTING** the Wave-1 findings via ground-truth re-verification. Mandate: find benign explanations, hidden context, test coverage. Reclassified ~25 findings (some upgraded to CRITICAL, some downgraded, several refuted outright). Surfaced ~40 NEW findings the original audit missed (especially in security, performance, concurrency, fp precision, reproducibility, and architectural debt).

### Wave 3 (synthesis — this doc)
Reconcile Wave-1 + Wave-2 verdicts. Build canonical findings list with per-finding technical brief. Track:
- **CONFIRMED**: claim holds; Wave-2 found no benign explanation.
- **REFUTED-WRONG**: claim factually contradicted by ground-truth code.
- **REFUTED-OVERSTATED**: claim technically true but severity inflated.
- **RECLASSIFIED**: real but wrong severity.
- **NEW**: surfaced only in Wave-2 hunt.

This 3-wave pattern is established in the pipeline; the root `CLAUDE.md` banner history shows it has consistently caught Wave-1 overstatements (see #PY-207 refutation 2026-05-13 night, BACKTEST_INDEX R-16d Wave-3 reframing of the +2.84% finding). Multi-wave validation should be the default for any large architectural decision.

---

## §2 Findings index

Findings are numbered sequentially. Each gets a stable `FIND-NNN` ID for cross-reference across themes.

| Theme | IDs |
|---|---|
| Engine accounting | FIND-001 … FIND-019 |
| Metrics correctness + namespace | FIND-020 … FIND-040 |
| Strategy + label semantics | FIND-041 … FIND-055 |
| Configuration plane | FIND-056 … FIND-079 |
| Contract plane + signal manifest | FIND-080 … FIND-089 |
| Registry + orchestration | FIND-090 … FIND-099 |
| Tests + coverage + CI | FIND-100 … FIND-109 |
| Security | FIND-110 … FIND-114 |
| Performance + memory | FIND-115 … FIND-119 |
| Concurrency + threading | FIND-130 … FIND-134 |
| Numerical precision + fp | FIND-135 … FIND-144 |
| Reproducibility + determinism | FIND-145 … FIND-149 |
| Architectural debt | FIND-120 … FIND-129 |
| Cross-repo consolidation | FIND-150 … FIND-159 |
| Documentation drift | FIND-160 … FIND-169 |

---

## §3 Engine + types + accounting (FIND-001 … FIND-019)

### FIND-001 — End-of-data auto-close drops the final trade from `trades` but appends to `trade_pnls`

**Status**: CONFIRMED
**Severity**: CRITICAL (latent under default strategies; active under always-in strategies)
**Active vs Latent**: LATENT — most production strategies emit an explicit EXIT at the last bar (HoldingPolicy with `hold_events=N` aligning to corpus length). A strategy that holds past end-of-data fabricates a phantom close.
**Evidence**:
- `src/lobbacktest/engine/vectorized.py:436-442` — `_close_position` is called and `trade_pnls.append(...)` is invoked, but **no** `trades.append(Trade(...))` is emitted for the close. The opening `Trade` (BUY/SELL at lines 337-345/386-394) was appended; the matching close vanishes.
- Downstream propagation: `engine/zero_dte.py:266-273` — `ZeroDtePnLTransformer.transform` iterates `n_round_trips = len(equity_pnls)` and assumes `entry_idx = i*2; exit_idx = i*2 + 1`. When `len(trades) < 2*n_round_trips` (orphaned close), the silent `break` at lines 269-270 drops the final round-trip from option-mode P&L while equity-mode still includes it.

**Reproduction**:
Construct a 3-bar test: `prices=[100,101,102]`, predictions=[BUY, BUY, BUY], no holding-policy-driven EXIT. Engine opens long at bar 0, holds, never closes via signal. At loop exit, end-of-data branch fires. Result: `len(trades) = 1` (the BUY), `len(trade_pnls) = 1`, but the invariant docstring at `types.py:194-200` claims `len(trade_pnls) == number of round-trip trades (closes)`. Operator reads `total_trades = 1` and "Winning: 1" — but the close was never recorded as a trade.

**Root cause**:
The auto-close at lines 436-442 was added as a defensive cleanup (per the P2 fix narrative) but the symmetry with the in-loop trade-recording was missed. The only invariant `BacktestResult.__post_init__` enforces is `total_trades == len(trades)`, NOT the stronger pairing rule.

**Impact**:
- Active when: any strategy holds a position through the last bar without explicit EXIT.
- Blast radius: equity-mode summary metrics are correct (P&L recorded), but option-mode (ZeroDte) silently truncates the final round-trip.
- Worst case: a 1000-trade backtest with the last trade being a 5σ outlier; option mode reports 999 trades, equity mode 1000.

**Edge cases that matter for fix**:
- Reversal-on-same-bar (close short + open long at same `i`) emits **2** trades at the same `i` correctly (lines 310-318 + 337-345). Auto-close emits 1 trade_pnl + 0 trades.
- HoldingPolicy with `hold_events=N` that happens to align with `len(prices) - 1` already emits explicit EXIT (current production path; masks the bug).

**Related**: FIND-002 (invariant gap), FIND-003 (ZeroDte alternation assumption), FIND-004 (phantom-trade cost fabrication).

**Constraints (must preserve)**:
- Long/short P&L symmetry post-C3 fix.
- `trade_pnls.append(pnl - cost - entry_cost)` formula (P2 fix; 4 sites).
- Empirical R6 +45.0% WR / R7 -1.93% / R8 -0.85% reproductions.

**Fix-direction constraints**:
Any fix must (a) emit a `Trade(side=TradeSide.FLAT, ...)` at line 441 to restore symmetry, AND (b) decide whether auto-close fabrication should be a CONFIGURABLE policy (`config.auto_close_on_end: bool = True`) so a strategy can opt out by signaling EXIT explicitly. Option (b) is preferred for hft-rules §8 fail-loud semantics.

**Open questions**:
- Should the engine REQUIRE the strategy to emit explicit EXIT at last bar (fail-loud if open position remains), or fabricate the close (current) but record it correctly?
- For the fabricated close, do we apply transaction costs (current behavior) or treat the open position as "still open at end of evaluation" (returning unrealized P&L only)?

---

### FIND-002 — `BacktestResult.__post_init__` invariant gap: doesn't validate `len(trades) == 2 * len(trade_pnls)`

**Status**: NEW (Adv-1)
**Severity**: CRITICAL (lets FIND-001 ship undetected)
**Active vs Latent**: LATENT (active for FIND-001 scenarios).
**Evidence**: `src/lobbacktest/types.py:218-242`. Validates: equity_curve length, prices/positions length, final_equity == equity_curve[-1] (within 1e-10), `total_trades == len(trades)`. Does NOT validate the stronger pairing `len(trade_pnls) == count(t.side == TradeSide.FLAT for t in trades)`.

**Root cause**: The docstring at types.py:194-200 specifies the invariant in prose but never enforces it.

**Impact**: This invariant gap is what allows FIND-001 to ship today without anyone catching it. Adding it would have surfaced FIND-001 the moment any always-in strategy ran.

**Constraints (must preserve)**:
- Empty equity_curve raise.
- 1e-10 final_equity tolerance (though FIND-138 questions this at large capital).

**Fix-direction constraints**:
Add `n_closes = sum(1 for t in self.trades if t.side == TradeSide.FLAT); assert len(self.trade_pnls) == n_closes` to `__post_init__`. Coordinate with FIND-001 fix — both must ship in same commit.

**Related**: FIND-001, FIND-003.

---

### FIND-003 — `ZeroDtePnLTransformer` assumes strict `[open, close, open, close, …]` alternation, never validates

**Status**: NEW (Adv-1)
**Severity**: HIGH
**Active vs Latent**: LATENT (broken by FIND-001 scenarios).
**Evidence**: `engine/zero_dte.py:266-274` — `for i in range(n_round_trips): entry_idx = i*2; exit_idx = i*2 + 1; entry_trade = trades[entry_idx]; exit_trade = trades[exit_idx]`. No `assert` checks alternation. Silent `break` at line 269-270 hides FIND-001's off-by-one.

**Root cause**: The contract between engine output (`trades`) and ZeroDte input is implicit. Reversal-on-same-bar emits 2 trades at the same `i` correctly (no alternation break). End-of-data auto-close breaks alternation.

**Impact**: silently truncates option-mode results when alternation breaks. Combined with FIND-001 + FIND-002, this is a silent-wrong-result chain.

**Fix-direction constraints**:
After FIND-001/FIND-002 fix, add explicit `assert entry_trade.side != TradeSide.FLAT and exit_trade.side == TradeSide.FLAT` inside the loop. If invariant is broken, raise.

**Related**: FIND-001, FIND-002.

---

### FIND-004 — End-of-data auto-close fabricates trade + applies costs the strategy never signaled

**Status**: CONFIRMED
**Severity**: HIGH (silent cost fabrication; hft-rules §8 violation)
**Active vs Latent**: ACTIVE on any always-in strategy.
**Evidence**: `vectorized.py:436-442` unconditionally calls `_close_position(current_position, prices[-1])`, applies `cost = self.config.costs.compute_cost(...)`, deducts from cash. **The strategy did not request this trade.**

**Root cause**: Defensive engineering: leaving an open position would leak into next backtest's state. But the cost application is silent.

**Impact**:
- For Deep ITM 1.4 bps: a phantom close costs ~$1.40 per share at $100 (cost dwarfs typical pnl). On a 700-trade backtest, the phantom trade is 1 of 700; ~0.14% drag.
- For ATM 4.9 bps: ~$4.40 per share. ~0.7% drag.
- Operator never sees that the cost was applied to a strategy-not-signaled exit.

**Fix-direction constraints**:
Make this a config-driven policy. Either: (a) `config.auto_close_on_end="signal_exit"` (raise if open position at end), (b) `"force_close"` (current — apply costs), (c) `"unrealized"` (no costs; carry unrealized P&L).

**Related**: FIND-001, FIND-002.

---

### FIND-005 — Long/short P&L symmetry post-C3 fix verified correct

**Status**: CONFIRMED-HEALTHY
**Severity**: N/A (preserved invariant)
**Evidence**: 
- Long open: `cash -= (position_value + cost)` (`vectorized.py:329`).
- Short open: `cash -= (position_value + cost)` (`vectorized.py:378` — C3 fix).
- Symbolic walkthrough: open long 10 sh @ $100, cost $1 → cash = -$1001 (from $0); mark at $110 → unrealized=+$100, equity = cash + price × size = -$1001 + $110×10 = $99. Net of initial capital, P&L = +$98. Close: pnl=+$100, cost=$1.10, cash += $1100 - $1.10. Final cash = -$1001 + $1098.90 = $97.90 — equity = $97.90 + 0 = $97.90 ≈ initial - costs + pnl.
- Short symmetric: short 10 @ $100 with $1 cost → cash = -$1001. Mark at $90 → unrealized=+$100, equity = cash + margin + unrealized = -$1001 + $1000 + $100 = $99. Close: pnl=+$100, cost=$0.90, cash_flow = $1000+$100 = $1100, cash += $1100 - $0.90 = $98.10.

**Conclusion**: post-C3 fix is mathematically symmetric.

**Constraints (must preserve)**: This is encoded lesson #3 — DO NOT REVERT `cash -= cost` only for shorts. The whole short backtest history (R-9..R-17a) depends on this.

---

### FIND-006 — Silent NaN→0 in returns calc may hide catastrophic equity collapse

**Status**: REFUTED-OVERSTATED (latent only)
**Severity**: MEDIUM (was HIGH in Wave-1)
**Active vs Latent**: LATENT — Adv-1 walkthrough showed that for SHORT positions with 100% adverse move, `equity[i-1] ≈ -cost ≈ 0`, producing nan/inf returns which are silently clamped. But default `BacktestConfig` has 5% leverage buffer and position-size caps; reaching equity=0 requires a structurally rare path.
**Evidence**: `vectorized.py:445-447`. `returns = np.diff(equity) / equity[:-1]; returns = np.where(np.isfinite(returns), returns, 0.0)`.

**Root cause**: Defensive against fp explosion for downstream Sharpe/Sortino computation. Hft-rules §8 prohibits silent clamps without diagnostics.

**Impact**:
- For nominal NVDA backtests: equity never approaches 0; clamp never fires.
- For extreme drawdown scenarios (short squeeze; pathological strategy): clamp silently masks a blown-up backtest as flat.

**Fix-direction constraints**:
Track `n_nonfinite_returns` counter; surface in `BacktestResult.metrics["diagnostics"]`. Raise on `equity == 0` (or `equity < min_capital` config) per hft-rules §5 fail-loud.

**Related**: FIND-141.

---

### FIND-007 — `Position.entry_cost` defaults to `0.0`; external constructors silently mis-account

**Status**: CONFIRMED-LATENT
**Severity**: LOW (no production bypass exists)
**Active vs Latent**: LATENT — engine always passes `entry_cost=cost`; tests construct without it deliberately (`tests/test_types.py:155, 169, 182` test the default-zero path as designed behavior).
**Evidence**: `types.py:118`. `Position.flat()` factory also uses 0.0.

**Root cause**: Defensive default for the FLAT state.

**Impact**: Only fires if a future caller constructs `Position` outside the engine without setting entry_cost. None exist today.

**Fix-direction constraints**:
Option A: require `entry_cost` (no default) when `side != FLAT`. Breaks `Position.flat()` factory unless coupled with field reorder.
Option B: keep default; add `__post_init__` invariant `if side != FLAT and entry_cost == 0.0: warn`.

---

### FIND-008 — `Position.unrealized_pnl` is dead field

**Status**: CONFIRMED
**Severity**: LOW
**Active vs Latent**: DEAD
**Evidence**: `types.py:117` field declared. Engine computes `unrealized` locally at `vectorized.py:286-292` and never writes back (`Position` is `frozen=True`). The `unrealized_pnl_bps` field on `HoldingState` is a separate concept.
**Fix-direction**: remove from dataclass. Coordinate with serialization (FIND-009).

---

### FIND-009 — `BacktestResult.to_dict` omits `Position.entry_cost` + `unrealized_pnl`; no inverse `from_dict`

**Status**: CONFIRMED (severity demoted)
**Severity**: LOW (no `from_dict` exists, so no round-trip risk)
**Active vs Latent**: DEAD (no current consumer round-trips).
**Evidence**: `types.py:312-342` `to_dict` outputs `{index, side, price, size, cost}` for trades — drops entry_cost. No `from_dict` method exists. BacktestResult is write-only.

**Fix-direction**: serialize entry_cost; OR document explicitly that to_dict is one-way and entry_cost is preserved in `cost` field semantics (P2 fix bakes entry_cost into trade_pnls).

---

### FIND-010 — `_compute_position_size` hidden NVDA-centric coupling via `initial_capital / 1000` reference price

**Status**: CONFIRMED-NEW (Adv-1 added empirical verification)
**Severity**: MEDIUM
**Active vs Latent**: ACTIVE for any underlying < $100 with `initial_capital=$100K`.
**Evidence**: `vectorized.py:519` — `reference_price = max(price, self.config.initial_capital / 1000)`. For `price=$0.50` + `initial=$100K`: ref=$100, `max_shares = max_position × $100K / $100 = 200 × max_position`. Target value was $10K → 20,000 shares; capped at 200 shares = $100. 95% silent under-sizing.

**Root cause**: defensive against catastrophic short losses on penny stocks. Comment line 518-519 acknowledges "we assume prices are roughly $100+" — a hardcoded NVDA-era assumption.

**Impact**:
- For NVDA at $180: ref=$180, no cap binding. ✓
- For any future underlying < $100 with same `initial_capital`: silent under-sizing.

**Fix-direction constraints**:
- Replace Cap-3 with explicit `config.max_shares_per_trade: Optional[int] = None` + assert `price > config.min_reference_price` (raise if violated).
- Document the "$100K/1000 = $100 reference" assumption explicitly.

**Constraints (must preserve)**: NVDA empirical results are unaffected because ref=$180 > $100.

---

### FIND-011 — Position sizing references `initial_capital` for both Cap-1 and Cap-3; drawdown-insensitive

**Status**: CONFIRMED
**Severity**: HIGH (risk-management drift)
**Active vs Latent**: ACTIVE after drawdown.
**Evidence**: `vectorized.py:504, 519` — both `max_value` and `reference_price` use `initial_capital`. Line 501 uses CURRENT `capital` for target. Inconsistent denominator.

**Root cause**: Position-size policy is ambiguous between fixed-fraction (current capital) and fixed-dollar (initial capital).

**Impact**: After a 50% drawdown, engine still permits positions sized off original capital ⇒ effectively 2x leverage.

**Fix-direction constraints**: pick one denominator and document. Fixed-fraction (current capital) is the conventional choice and what the docstring at config.py:285-286 implies.

---

### FIND-012 — `_compute_position_size` magic constants `0.95` leverage buffer and `1000` reference-price divisor

**Status**: CONFIRMED
**Severity**: MEDIUM (hft-rules §2 magic-number violation)
**Evidence**: `vectorized.py:508, 519`.
**Fix-direction**: named module constants with citations; config-driven overrides.

---

### FIND-013 — Reversal-case `cost` variable shadowed between close and re-open

**Status**: NEW (Adv-1)
**Severity**: LOW (code-smell, not correctness)
**Evidence**: `vectorized.py:303-329` — line 306 `_, cost, pnl = self._close_position(...)`; line 307 `cash += cash_flow - cost`; line 309 `trade_pnls.append(pnl - cost - entry_cost)`. Then line 326 opens new long: `cost = self.config.costs.compute_cost(position_value)`. The variable `cost` is overwritten between the close and the new open.

**Root cause**: convenience of variable reuse inside the same signal-handler block.

**Impact**: refactor hazard. If anyone reorders these lines or extracts a helper, the dual semantics of `cost` could silently desync.

**Fix-direction**: rename to `exit_cost` and `entry_cost_new` locally.

---

### FIND-014 — `returns = np.diff(equity) / equity[:-1]` allocates 3x N memory hot-path

**Status**: NEW (Adv-1)
**Severity**: LOW
**Evidence**: `vectorized.py:445-447`. Three array allocations: `np.diff`, division result, `np.where` result.

**Impact**: at 1M events typical, ~24MB peak (8 bytes × 1M × 3). At 5K-100K events (current workload), negligible.

**Fix-direction**: hot-path watch only. Can use in-place ops if needed.

---

### FIND-015 — `np.prod(1 + returns) - 1` overflow risk on long histories

**Status**: NEW (Adv-2)
**Severity**: MEDIUM
**Active vs Latent**: LATENT — current backtests are days/weeks not years.
**Evidence**: `metrics/returns.py:72` (TotalReturn), `metrics/risk.py:402` (CalmarRatio inline fallback). Raw `np.prod` with no log-domain fallback.

**Impact**: For ~10^4 periods with avg return ~1%, product reaches ~10^43 (overflows float64 ~ 10^308 only at extreme tail). More worrying: tail of return distribution (5σ event) can overflow individual `(1 + r)` factors if r > 10^307. Currently bounded by Sharpe calculation context.

**Fix-direction**: log-domain `np.exp(np.sum(np.log1p(returns))) - 1` for long histories. Add explicit overflow guard mirroring `returns.py:183` AnnualReturn pattern (already has it).

**Related**: FIND-138 (CalmarRatio inline divergence has its own overflow path).

---

### FIND-016 — `BacktestResult.__post_init__` final_equity epsilon `1e-10` impossible at large capital

**Status**: NEW (Adv-7)
**Severity**: HIGH (false positives at scale)
**Active vs Latent**: LATENT until backtest runs at >$1M capital.
**Evidence**: `types.py:235-238`. `abs(self.final_equity - self.equity_curve[-1]) > 1e-10`. For float64, eps at $1B value ≈ 1e9 × 2^-52 ≈ 2e-7 — exceeds 1e-10 absolute tolerance. False positives.

**Fix-direction**: `abs(self.final_equity - self.equity_curve[-1]) > max(1e-10, abs(self.final_equity) * 1e-12)` — relative tolerance.

---

### FIND-017 — `_close_position` for shorts uses `entry_price * size + pnl` for cash_flow

**Status**: CONFIRMED-HEALTHY
**Evidence**: `vectorized.py:565-568`. Mathematically correct: returns margin + P&L; matches the C3 symmetric-sizing fix.

---

### FIND-018 — `Trade` and `Position` frozen-dataclass allocation in hot loop

**Status**: NEW (Adv-7)
**Severity**: MEDIUM (latent perf for large backtests)
**Evidence**: `vectorized.py:310-345` etc. Each entry+exit creates 2 `Trade` instances. At 1M samples × ~1% trade rate = 20K allocations × `__post_init__` validation overhead.
**Impact**: Python allocation overhead is ~1μs per dataclass → 20ms total. Negligible at current scale; flagged for >100K-trade futures.

**Fix-direction**: object pool / struct-of-arrays / numba refactor.

---

### FIND-019 — Engine module misnamed `vectorized.py` despite being per-sample loop

**Status**: CONFIRMED (acknowledged in code)
**Severity**: LOW (docstring acknowledges)
**Evidence**: `vectorized.py:1-12` docstring "Module is named 'vectorized.py' for historical reasons. The main engine loop is a Python for-loop, not vectorized."
**Fix-direction**: rename to `engine/_per_sample.py` (mirrors `hft-feature-evaluator` Phase 6 archive precedent) OR actually vectorize (numba; sub-finding of FIND-115).

---

## §4 Metrics correctness + namespace (FIND-020 … FIND-040)

### FIND-020 — Two `DirectionalAccuracy` classes with identical metric name

**Status**: CONFIRMED + RECLASSIFIED (severity downgrade — orphan module)
**Severity**: LOW (was CRITICAL in Wave-1; refuted by Wave-2: `regression_prediction.py` is fully orphan dead code, never imported)
**Active vs Latent**: DEAD
**Evidence**:
- `metrics/prediction.py:31` `class DirectionalAccuracy(Metric)` — imported by engine (`vectorized.py:26`).
- `metrics/regression_prediction.py:137` `class DirectionalAccuracy(Metric)` — **never imported anywhere** in `src/`, `scripts/`, or `tests/` (Adv-2 grep verified).
- `metrics/__init__.py:28-33` re-exports ONLY the `prediction.py` version.

**Root cause**: `regression_prediction.py` is a feature implemented but never wired in. The whole module is orphan code.

**Impact**: latent only if a user manually composes a metric list including both — engine default cannot trigger the collision.

**Fix-direction constraints**:
Option A: delete `regression_prediction.py` entirely (orphan; see FIND-021).
Option B: rename `regression_prediction.DirectionalAccuracy` → `SignDirectionalAccuracy` AND wire into engine for regression strategies AND add duplicate-name guard to engine.

**Open questions**: Was `regression_prediction.py` intended for a future regression-metric-suite that never landed? If so, the module's PredictionIC / PredictionCorrelation / PredictionMSE / DirectionalAccuracy classes should be deleted along with the namespace collision risk.

---

### FIND-021 — `metrics/regression_prediction.py` module is fully orphan dead code

**Status**: NEW (Adv-2)
**Severity**: HIGH (architectural debt accumulation)
**Active vs Latent**: DEAD
**Evidence**: zero imports anywhere in the pipeline (Adv-2 grep). Defines 4 classes with rigorous fail-loud `_assert_finite_pair` boundaries — none consumed.

**Root cause**: Module was created during the 2026-05-07 #PY-63 cycle as a fail-loud upgrade to scipy correlation handling; wiring into the engine's regression-strategy path was never completed.

**Impact**: ~180 LOC unused; maintenance burden + hft-rules §4 "no dead code" violation.

**Fix-direction**: 
Option A: delete entire module + tests.
Option B: wire into engine's regression-strategy metric list (currently engine emits classification metrics only when `labels is not None`; regression analog would emit when `regression_labels is not None`).

**Open questions**: 
- If the regression-metric path is wanted, it should DELEGATE to `hft_metrics.ic.spearman_ic` / `pearson_r` (D1/D2 below) rather than re-implement scipy calls.

**Related**: FIND-150, FIND-151, FIND-152, FIND-153 (cross-repo consolidation candidates).

---

### FIND-022 — `metrics/prediction.py::ConfusionMetrics` is orphan dead code (not in `__init__.py`, not exported)

**Status**: NEW (Adv-2)
**Severity**: MEDIUM
**Evidence**: `prediction.py:339-432` defines `ConfusionMetrics`. Not in `metrics/__init__.py` exports. Zero consumer references.
**Note**: Wave-1 M5 claim that ConfusionMetrics violates the ABC contract was **REFUTED-WRONG** because `metrics/base.py:132-167` defines an explicit `CompositeMetric` pattern that allows multi-key returns. ConfusionMetrics is correctly implementing this pattern.
**Fix-direction**: delete OR expose via `__init__.py` + add per-class tests.

---

### FIND-023 — Strategy-signal-vs-model-prediction semantic mismatch in `DirectionalAccuracy` is INTENTIONAL

**Status**: CONFIRMED-AS-DESIGN
**Severity**: LOW (documentation gap)
**Evidence**:
- `vectorized.py:599-601` sets `context.predictions = signal_output.signals` (strategy signals, not model predictions).
- `metrics/prediction.py:31-126` measures strategy-signal accuracy.
- `Signal.EXIT=2` collides with `SHIFTED_LABEL_UP=2` only when `shifted=True` is passed; engine default is `shifted=False` (mapping `{-1, 0, 1}`), so no collision in default path.

**Root cause**: This is the backtester's job: measure realized-trading-accuracy, NOT model accuracy. Model accuracy is the trainer's responsibility.

**Impact**: latent only if a user manually constructs `DirectionalAccuracy(shifted=True)`.

**Fix-direction**: docstring on `prediction.py` module level should explicitly state "measures STRATEGY signals, NOT model predictions". Optional: add `Signal.EXIT = 99` to remove the latent collision (breaks Signal.IntEnum value contracts; audit consumer code).

**Related**: If a future cycle wants model-prediction accuracy, add a separate `ModelDirectionalAccuracy` metric reading `data.predictions` (raw int32 model output).

---

### FIND-024 — Phase X.3 silent-zero → NaN migration is INCOMPLETE on 4 trading metrics

**Status**: CONFIRMED
**Severity**: HIGH (Phase X.3 closure claim is FALSE)
**Active vs Latent**: ACTIVE on empty-trade backtests.
**Evidence**:
- `metrics/trading.py:79` WinRate empty → `NaN` ✓ (migrated)
- `metrics/trading.py:148` ProfitFactor empty → `NaN` ✓ (migrated)
- `metrics/trading.py:212` AverageWin empty → `0.0` ✗ (NOT migrated)
- `metrics/trading.py:273` AverageLoss empty → `0.0` ✗
- `metrics/trading.py:347` PayoffRatio empty → `0.0` ✗
- `metrics/trading.py:417` Expectancy empty → `0.0` ✗

**Root cause**: Phase X.3 cycle migrated 2 of 6 trading metrics; remaining 4 missed.

**Impact**: An empty-trade backtest reports Expectancy=0 (correct: no edge) which is indistinguishable from a non-empty backtest with exactly-balanced wins/losses (also 0). Hft-rules §8 "never silently clamp".

**Fix-direction**: 1-line each — migrate to `return {self.name: float("nan")}` for `n_total == 0` branches. Add tests pinning the NaN convention.

**Constraints (must preserve)**: existing semantic that "all wins" PayoffRatio=∞ etc. Per FIND-031 the cap-at-100.0 convention separately violates hft-rules.

---

### FIND-025 — `SortinoRatio` magic constant 100.0 cap

**Status**: CONFIRMED
**Severity**: HIGH (magic number; hft-rules §2 violation)
**Evidence**: `metrics/risk.py:213-215`. `if mean_return > 0: return 100.0`.
**Fix-direction**: replace with `float("inf")`. Document at base Metric ABC.

---

### FIND-026 — `ProfitFactor` "no losses" 100.0 cap

**Status**: NEW (Adv-2)
**Severity**: MEDIUM
**Evidence**: `trading.py:158`. Same anti-pattern as FIND-025.
**Fix-direction**: replace with `float("inf")`.

---

### FIND-027 — `CalmarRatio` recomputes `AnnualReturn` inline with divergent overflow handling

**Status**: CONFIRMED
**Severity**: HIGH (silent inf in CalmarRatio when AnnualReturn would clip)
**Evidence**: `metrics/risk.py:400-409` inline AR computation lacks the `exponent > 1000` guard at `returns.py:179-184` AND the `try/except (OverflowError, FloatingPointError)` at `returns.py:188-190` AND the `np.clip(annual_return, -1.0, 1e10)`.

**Active vs Latent**: ACTIVE for short backtests with extreme returns.

**Fix-direction**: CalmarRatio should DELEGATE to `AnnualReturn().compute(returns, context)` for the AR component. Document explicit context dependency.

---

### FIND-028 — `AnnualReturn` `np.clip(... 1e10)` silent clamp

**Status**: CONFIRMED
**Severity**: HIGH
**Evidence**: `returns.py:185`. Silent floor/ceiling.
**Fix-direction**: replace with `float("inf")`. Display layer formats.

---

### FIND-029 — `prediction.DirectionalAccuracy` returns 0.0 on missing context (silent drop)

**Status**: CONFIRMED
**Severity**: HIGH (Phase X.3 missed coverage)
**Evidence**: `prediction.py:101-107, 116-117` — return 0.0 for: (a) missing predictions OR labels, (b) length mismatch, (c) zero directional samples. Same pattern in SignalRate, UpPrecision, DownPrecision, ConfusionMetrics.

**Active vs Latent**: ACTIVE on any backtest that doesn't populate context.predictions/labels.

**Fix-direction**: 
- Missing context → raise per §5 fail-fast.
- Length mismatch → raise per §8.
- Zero directional → NaN per Phase X.3.

---

### FIND-030 — `BacktestResult.max_drawdown` (property) ≠ `risk.MaxDrawdown` (metric) on edge cases

**Status**: CONFIRMED
**Severity**: HIGH (silent inconsistency between summary print and metric)
**Evidence**:
- `types.py:259` empty equity → `0.0`
- `risk.py:294` empty equity → `NaN`
- `types.py:264` peak=0 → `np.where(np.isfinite, ..., 0.0)`
- `risk.py:315` peak=0 → `np.where(peak > 0, ..., 0.0)`

**Fix-direction**: delete property; have `summary()` read `self.metrics.get("MaxDrawdown", float("nan"))`.

---

### FIND-031 — `np.prod(1+returns)` overflow risk (TotalReturn)

**Status**: NEW (Adv-2)
**Severity**: MEDIUM
**Reference**: see FIND-015 — same root cause, different metric site.

---

### FIND-032 — Sortino MAR (Minimum Acceptable Return) hardcoded to 0

**Status**: NEW (Adv-2)
**Severity**: MEDIUM
**Evidence**: `risk.py:143, 207` — no constructor param for `target_return` or `mar`. Forces zero risk-free rate baseline.
**Impact**: cannot test against benchmark/RFR. For HFT context the typical convention is MAR=0 but cost-aware MAR (e.g., 1.4 bps Deep ITM breakeven) would be a defensible alternative.

**Fix-direction**: add `target_return: float = 0.0` constructor param.

---

### FIND-033 — Metric ordering dependency in `_compute_metrics`

**Status**: CONFIRMED
**Severity**: HIGH (fragile)
**Evidence**: `vectorized.py:645-651`. CalmarRatio reads `context["AnnualReturn"]` set by prior AnnualReturn metric. List order matters.

**Fix-direction**: each Metric declares `requires: Set[str]`. Engine topologically sorts; raises on cycle/missing. Or: each metric is self-contained (CalmarRatio takes equity + returns directly).

---

### FIND-034 — `MetricResult` ABC: metrics MUST return `{self.name: value}` but `ConfusionMetrics` returns multi-key

**Status**: REFUTED-WRONG (per Wave-2)
**Severity**: N/A
**Evidence**: `metrics/base.py:132-167` documents `CompositeMetric` pattern explicitly. ConfusionMetrics correctly implements this pattern.
**Note**: this REFUTES the Wave-1 M5 claim. Documentation in `base.py` should make the multi-key allowance more discoverable.

---

### FIND-035 — `BacktestContext.update` is reentrance-vulnerable for duplicate-name metrics

**Status**: NEW (Adv-2)
**Severity**: LOW (engine default doesn't trigger)
**Evidence**: `vectorized.py:647-651`. If two metrics return the same key, the second silently overwrites. Only fires if user manually composes list with both `prediction.DirectionalAccuracy` and (orphan) `regression_prediction.DirectionalAccuracy`.

**Fix-direction**: at engine entry, assert `len({m.name for m in metrics}) == len(metrics)`.

---

### FIND-036 — `regression_prediction.*` metrics require arrays at `__init__` — breaks Metric ABC pattern

**Status**: CONFIRMED (related to FIND-021)
**Severity**: MEDIUM
**Evidence**: `regression_prediction.py:53-65, 70-94, 100-134` — takes `(predicted, actual)` at init; ignores `compute(returns, context)` args.

**Impact**: cannot be composed into engine's default metric loop (engine has no way to inject `predicted/actual`).

**Fix-direction (if wiring rather than deleting)**: refactor to read `context["predicted_returns"]` and `context["regression_labels"]`. Add fields to `BacktestContext` at `context.py:60`.

---

### FIND-037 — `PayoffRatio` context dependency on AverageWin/AverageLoss is order-sensitive

**Status**: NEW (Adv-2)
**Severity**: INFO
**Evidence**: `trading.py:341-343` reads from context (set by prior metrics via update). Fallback path re-computes from trade_pnls — slightly different rounding.

**Fix-direction**: declare requires-dependency (FIND-033 unifies).

---

### FIND-038 — `metrics/__init__.py` doesn't export `regression_prediction` symbols

**Status**: CONFIRMED-HEALTHY (correct given FIND-021)
**Evidence**: `metrics/__init__.py:28-33` exports prediction.py + base.py + returns.py + risk.py + trading.py. Skip is intentional given the orphan status.

---

### FIND-039 — Sharpe/Sortino std `ddof` not documented

**Status**: NEW (clarification)
**Severity**: LOW
**Evidence**: `risk.py:65-67`. Uses `np.std(returns, ddof=0)` (population). Sharpe (1966) is ambiguous; convention for HFT (large N) makes the distinction negligible.

**Fix-direction**: document explicitly in docstring. Add test that hand-calculates with both ddof=0 and ddof=1 to lock the choice.

---

### FIND-040 — `BacktestStats.daily() / .monthly() / .full()` are STUBS

**Status**: CONFIRMED (CRITICAL silent-degrade)
**Severity**: CRITICAL (was H1 in Wave-1; promoted)
**Active vs Latent**: ACTIVE — any caller chaining `.daily()` gets identical output as `.full()`.
**Evidence**: `stats/stats.py:106-124` set `self._period = "daily"` etc.; `compute()` body at lines 149-207 NEVER reads `self._period`. Period only flows to `StatsSummary.period` label.

**Root cause**: aggregation requires per-period timestamp data; `BacktestResult` has no timestamps, only positional index. Calendar aggregation is structurally impossible without contract change.

**Impact**: API actively misleads operators. A "monthly equity curve" report would be identical to "full" because aggregation is dead.

**Fix-direction (no implementation)**:
Option A: raise `NotImplementedError` in `.daily()/.monthly()` with precise message. Cleanest fail-loud per hft-rules §5.
Option B: require timestamps via `BacktestResult.timestamps_ns` field + implement aggregation properly. Larger scope (contract change).

**Open questions**: was BacktestStats meant to be calendar-aware? If yes, the path needs timestamps; if no, the API needs to drop the period methods entirely.

---

## §5 Strategy + label semantics (FIND-041 … FIND-055)

### FIND-041 — Strategies re-emit BUY/SELL while in position, relying on engine idempotency

**Status**: CONFIRMED-DESIGN-CHOICE (per Wave-2)
**Severity**: MEDIUM (implicit contract; no test pins it)
**Evidence**: `readability.py:206`, `regression.py:168`, `hybrid.py:200-217`. Engine no-ops at `vectorized.py:321, 369`.
**Root cause**: strategies emit "what they'd do if not in position" and engine arbitrates state.
**Fix-direction**: add `test_strategy_engine_idempotency_contract` to lock the semantic. Document the contract in `strategies/base.py`. Per hft-rules §4 modularity it would be cleaner for strategies to emit `Signal.HOLD` while holding, but the current contract is defensible.

---

### FIND-042 — `direction.py` re-declares LABEL_* constants (bypasses LabelMapping SSoT)

**Status**: CONFIRMED
**Severity**: MEDIUM (Class A SSoT violation; today values match)
**Evidence**: `strategies/direction.py:16-23`. Re-declares `LABEL_DOWN/STABLE/UP + SHIFTED_LABEL_*` byte-identical to `labels.py:25-32`. NO docstring marks it as backward-compat shim. Comment "consistent with lob-model-trainer" predates Phase 2a.

**Fix-direction**: replace local constants with `from lobbacktest.labels import LABEL_*`. Then migrate constants to import from `hft_contracts.labels` (D3 / FIND-154).

---

### FIND-043 — `DirectionStrategy` API uses `shifted: bool` ≠ other strategies' `label_mapping: LabelMapping`

**Status**: CONFIRMED
**Severity**: MEDIUM (API inconsistency)
**Evidence**: `direction.py:49,61` vs `readability.py:105`, `regression.py:78`, `hybrid.py:98`, `holding.py:128,134,145`.

**Root cause**: `DirectionStrategy` is older API; Phase 2a centralization was not back-applied to direction.

**Fix-direction**: deprecate `shifted: bool` in `DirectionStrategy`/`ThresholdStrategy`; accept `label_mapping: Optional[LabelMapping] = None` defaulting to `SHIFTED_MAPPING`. Mirror the modern API.

---

### FIND-044 — `RegressionStrategy._build_holding_state` silently zero-classifies `predictions_bps[i] == 0`

**Status**: CONFIRMED
**Severity**: HIGH (label drift on exact-zero predictions)
**Evidence**: `regression.py:118` — `pred_class = self.label_mapping.up if self.predictions_bps[i] > 0 else self.label_mapping.down`. Exact zero is silently routed to DOWN.

**Fix-direction**: explicit branching: `> EPS / < -EPS / else stable`. Use named `EPS` constant.

---

### FIND-045 — `RegressionStrategy` confidence scale `abs(pred)/20.0` magic divisor

**Status**: CONFIRMED
**Severity**: MEDIUM (magic number; unbounded for |pred| > 20 bps)
**Evidence**: `regression.py:122`.
**Fix-direction**: parameterize via config; OR scale relative to `min_return_bps`.

---

### FIND-046 — `RegressionStrategy.predictions_bps` NaN passes magnitude gate silently

**Status**: NEW (Adv-2 cross-confirmed)
**Severity**: MEDIUM
**Evidence**: `regression.py:99` — `abs(predictions_bps[i]) < min_return_bps` evaluates `False` for NaN (any comparison with NaN returns False). NaN prediction → gate fails BUT then line 175-185 `if pred > 0:` also False for NaN → SELL branch — silent routing.

**Fix-direction**: `if not np.isfinite(predictions_bps[i]): continue`.

---

### FIND-047 — `readability.py` agreement gate uses `<` while confirmation gate uses `<=` (asymmetric)

**Status**: CONFIRMED
**Severity**: LOW (documentation gap)
**Evidence**: `readability.py:119` `if self.agreement_ratio[i] < self.config.min_agreement:` (strict <) ; line 121 `if self.confirmation_score[i] <= self.config.min_confidence:` (strict <=).
**Test that pins**: `test_strategies/test_readability.py:89-101 test_confirmation_boundary_equal_does_not_pass`.

**Fix-direction**: either align both to `<` (admit equality) or document the asymmetry intentionally.

---

### FIND-048 — `readability.py` module docstring stale (claims `min_agreement == 1.0`)

**Status**: CONFIRMED
**Severity**: LOW (doc drift)
**Evidence**: `readability.py:8-19` docstring vs line 54 default `0.667`.
**Fix-direction**: rewrite docstring to match P5 fix.

---

### FIND-049 — `hybrid.py` default `min_agreement = 1.0` (NOT P5-fixed)

**Status**: CONFIRMED
**Severity**: MEDIUM (sister to readability; P5 not back-applied)
**Evidence**: `hybrid.py:56`. Compare to `readability.py:54` = 0.667.

**Root cause**: P5 fix applied to readability only; hybrid is documented as empirically failed (R5 = -2.67%) so not migrated.

**Fix-direction**: 
Option A: align hybrid default to 0.667.
Option B: explicit `raise NotImplementedError("HYBRID empirically failed R5; see BACKTEST_INDEX")` — fence the module.

---

### FIND-050 — `hybrid.py` and `readability.py` `n_entries`/`n_both_pass` counter drift on Stable predictions

**Status**: CONFIRMED (Wave-2 reclassified to LOW)
**Severity**: LOW (counters are diagnostic; doesn't affect signals)
**Evidence**: counter incremented before `is_bullish/is_bearish` branches. If pred is Stable, `n_both_pass` counts a would-pass event but no entry happens.

**Fix-direction**: increment `n_entries`/`n_both_pass` AFTER successful entry only.

---

### FIND-051 — TWAP marked SKIP in CLAUDE.md but NO actual skip marker in code

**Status**: CONFIRMED (Adv-8 verified)
**Severity**: HIGH (DOC-DISAGREES-WITH-CODE; tests run normally; engine could silently regress)
**Evidence**: `strategies/twap.py` — no `pytest.mark.skip`, no `raise NotImplementedError`. `tests/test_strategies/test_twap.py` — no `pytestmark = pytest.mark.skip`, no `pytest.skip()` body calls. 8 tests run normally. CLAUDE.md L283 claims "marked SKIP — empirically failed, C2 incompatibility".

**Impact**: a future refactor that "fixes" engine C2 incompatibility would silently undo the empirical failure finding. Encoded lesson #14 is NOT actually enforced (per Adv-8).

**Fix-direction**: add `pytestmark = pytest.mark.skip(reason="C2 engine incompatibility — empirically failed R2; see BACKTEST_INDEX")` to `test_twap.py`. Optionally raise `NotImplementedError` at TWAP `__init__` with citation.

**Related**: encoded lesson #14 in §A.

---

### FIND-052 — `holding.py::HoldingState` is not `frozen=True` (other dataclasses are)

**Status**: CONFIRMED
**Severity**: LOW (mutation hazard latent)
**Evidence**: `holding.py:31-56` plain `@dataclass`. Other dataclasses (`Position`, `Trade`, `LabelMapping`) are frozen.

**Fix-direction**: add `frozen=True`.

---

### FIND-053 — `create_holding_policy` factory silently injects `hold_events=10` default

**Status**: CONFIRMED
**Severity**: MEDIUM (silent-config; hft-rules §5)
**Evidence**: `holding.py:275-285` — `config.get("hold_events", 10)` etc.

**Fix-direction**: require explicit fields; fail-loud on missing.

---

### FIND-054 — `Strategy` ABC docstring claims "stateless" but `RegressionStrategy.self.prices` is statement-level state

**Status**: NEW (Adv-5 surfaced); CONFIRMED (Adv-7 corroborated as F-C3)
**Severity**: HIGH-LATENT (will fire under walk-forward CV reuse)
**Active vs Latent**: LATENT — current code always constructs fresh strategy per fold; multi-fold reuse would hit this.
**Evidence**:
- `regression.py:140-141, 174-175`, `readability.py:174-175`, `hybrid.py:158`:
  ```python
  if self.prices is None:
      self.prices = prices
  ```
- `base.py:8` docstring: "Strategies are stateless (no memory between calls)".

**Impact**: a Strategy instance reused across `generate_signals(prices_A)` then `generate_signals(prices_B)` retains `self.prices = prices_A`.

**Fix-direction**: replace conditional with unconditional `self.prices = prices`. OR: enforce strategy single-use via `_signals_generated: bool` guard.

**Related**: encoded lesson #14 (TWAP) parallels strategy-state contract.

---

### FIND-055 — Signal enum values: `Signal.EXIT = 2` collides with `SHIFTED_LABEL_UP = 2`

**Status**: CONFIRMED-LATENT
**Severity**: MEDIUM
**Evidence**: `strategies/base.py:21-35` Signal IntEnum + `labels.py:32` `SHIFTED_LABEL_UP=2`. When `shifted=True` is passed to a Strategy/Metric, an EXIT signal would silently map to "Up" because integer equality.

**Active vs Latent**: LATENT — engine default `shifted=False` maps signals to `{-1, 0, 1}`, no collision.

**Fix-direction**: rename `Signal.EXIT = 99` (or any non-{0,1,2} value). Audit all consumer code for IntEnum-value comparisons.

---

## §6 Configuration plane (FIND-056 … FIND-079)

### FIND-056 — `BacktestConfig.min_confidence` and `min_agreement` are DEAD fields that production YAMLs populate

**Status**: CONFIRMED + ESCALATED (Adv-3 surfaced FIND-070 silent-drop bug)
**Severity**: CRITICAL when combined with FIND-070
**Evidence**:
- `config.py:312-313` — `min_confidence: Optional[float] = None`, `min_agreement: Optional[float] = None`.
- `grep "self.config.min_confidence|self.config.min_agreement" src/` returns ONLY `ReadabilityConfig`/`ReadabilityHybridConfig` references — never `BacktestConfig`.
- Production YAMLs `configs/nvda_readability_first_xnas.yaml:59-60`, `configs/nvda_readability_first_arcx.yaml:43-44` set them under `backtest:` block.
- `scripts/run_readability_backtest.py:198-199` sets the same CLI flag onto BOTH `BacktestConfig(min_agreement=...)` AND `ReadabilityConfig(min_agreement=...)`. The first is silently ignored; the second is authoritative.

**Root cause**: The fields were added pre-Phase-3b but never wired to engine consumers. Production YAMLs use them in good faith.

**Impact**: see FIND-070 for the production silent-misconfig.

**Fix-direction**: pick ONE owner.
- Option A: promote to engine-level (read in `VectorizedEngine` independent of strategy).
- Option B: remove from `BacktestConfig`; force YAMLs to set them under `strategy:` block (mirrors regression strategy YAML schema).

**Related**: FIND-070.

---

### FIND-057 — `BacktestConfig.from_dict` silently overrides user-set cost fields when `exchange` is XNAS/ARCX

**Status**: CONFIRMED (design-intent, but silent override)
**Severity**: MEDIUM
**Evidence**: `config.py:393-394`. Production YAMLs set BOTH `costs.exchange` AND explicit cost fields; the latter are silently dropped.

**Fix-direction**: raise when both are set AND values differ from preset. OR document explicitly.

---

### FIND-058 — `BacktestConfig.{stop_loss_pct, take_profit_pct, fill_price, target_holding_minutes}` are dead

**Status**: CONFIRMED
**Severity**: MEDIUM (config dead surface; production YAMLs set `fill_price: midpoint` and get ignored)
**Evidence**:
- `config.py:288-309, 262` defines these.
- Zero engine reads (`grep "stop_loss_pct|take_profit_pct|fill_price|target_holding_minutes" src/engine/`).
- SL/TP supported via `HoldingPolicy.StopLossTakeProfitPolicy` — a parallel API.

**Fix-direction**: remove from `BacktestConfig`; direct users to `HoldingPolicy`. Audit production YAMLs.

---

### FIND-059 — `ZeroDteConfig.entry_window_start_et`, `entry_window_end_et` are dead fields with no validation

**Status**: CONFIRMED
**Severity**: HIGH (silent-no-op of a configurable behavior)
**Evidence**: `config.py:265-266` declared; `ZeroDtePnLTransformer.transform` (`zero_dte.py:240-310`) never reads them.

**Fix-direction**: 
Option A: remove fields.
Option B: raise `NotImplementedError` in `__post_init__` when user changes from defaults — explicit fail-fast per hft-rules §5.
Option C: implement — but requires per-trade timestamps (currently optional on Trade); deeper scope.

---

### FIND-060 — `ZeroDteConfig.target_holding_minutes` is dead

**Status**: CONFIRMED
**Severity**: LOW
**Evidence**: `config.py:262` declared; zero engine reads. `max_holding_minutes` is used (line 284); `target_holding_minutes` is not.
**Fix-direction**: remove OR implement target-holding-period strategy logic.

---

### FIND-061 — `ZeroDteConfig.__post_init__` does not validate `target_holding_minutes <= max_holding_minutes`

**Status**: NEW (Adv-3)
**Severity**: LOW (dead field, easy validator gap)
**Evidence**: `config.py:268-274`. Even if `target_holding_minutes` were wired (FIND-060), no validator prevents `target > max`.
**Fix-direction**: add validator if/when target_holding_minutes is wired.

---

### FIND-062 — `ZeroDteConfig.delta` has no semantic-range gate

**Status**: NEW (Adv-3)
**Severity**: HIGH-LATENT
**Evidence**: `config.py:269-270` validates `delta > 0 && delta <= 1.0`. But operator-set `delta=0.01` is dramatically different from `delta=0.95` (gamma-trader vs deep ITM). Engine consumes via `delta * (move_bps / 10000.0) * entry_price * 100 * contracts` — at `delta=0.01` produces near-zero P&L silently.

**Fix-direction**: add semantic recommendation in docstring; OR add discrete `OptionRegime` enum (`atm` / `deep_itm` / `otm`) with delta-derived constants.

---

### FIND-063 — `OpraCalibratedCosts.deep_itm()` reuses `atm_call_premium` field name semantically wrong

**Status**: CONFIRMED
**Severity**: LOW (naming hygiene)
**Evidence**: `config.py:215-223` sets `atm_call_premium=20.0` (deep ITM premium) for the deep-ITM factory.
**Fix-direction**: rename field to `call_premium`/`put_premium` OR introduce option-regime dispatch.

---

### FIND-064 — `CostConfig.maker_rebate_bps` is dead (populated by presets, never consumed)

**Status**: CONFIRMED (carry-forward from BACKTESTER_AUDIT_PLAN H2 / P10 since 2026-03-17)
**Severity**: MEDIUM
**Evidence**: `config.py:70` defined; populated by `_EXCHANGE_PRESETS` (XNAS=-0.20, ARCX=-0.15). `total_bps` (line 115-118) sums only `spread + slippage + taker_fee`. `compute_cost` (line 120-131) does not use rebate. Zero callers outside config.py.

**Fix-direction**:
Option A: WIRE rebate into `compute_cost` for maker fills — requires `Trade.liquidity_flag` field.
Option B: REMOVE — current engine treats all fills as taker.

---

### FIND-065 — `compute_cost` ignores commission direction (round-trip = 2× commission)

**Status**: NEW (Adv-7)
**Severity**: MEDIUM (documentation gap, not formula bug)
**Evidence**: `config.py:120-131`. `compute_cost(notional)` = `notional × total_bps/10000 + commission_per_trade`. Engine calls this for BOTH opens AND closes, so round-trip = 2× commission. Docstring says "per_trade" — ambiguous.

**Fix-direction**: rename `commission_per_trade` → `commission_per_fill` and document.

---

### FIND-066 — `OpraCalibratedCosts.entry_premium / atm_*_premium` fields are informational-only

**Status**: NEW (Adv-7)
**Severity**: LOW
**Evidence**: `config.py:170-171` declared; only consumed by `summary()` print (zero_dte.py:188). Not used in P&L calculation.
**Fix-direction**: either wire into a max-loss check (`abs(loss) ≤ premium`) or document as informational-only.

---

### FIND-067 — `ComparisonConfig` is wholly dead and mistyped (`Dict[str, any]` not `Dict[str, Any]`)

**Status**: CONFIRMED
**Severity**: LOW (latent type-checking gap)
**Evidence**: `config.py:457-468`. Zero usages anywhere in the pipeline. `Dict[str, any]` is technically valid only because `any` is callable; Pyright would flag.
**Fix-direction**: delete OR expose via tested comparison module.

---

### FIND-068 — `BacktestConfig.from_dict` hardcoded `costs_dict.get("spread_bps", 1.0)` defaults

**Status**: NEW (Adv-3)
**Severity**: LOW
**Evidence**: `config.py:397+` — YAML can omit cost fields and silently get hardcoded `1.0` bps default. Hardcoded numeric defaults in deserializer violate hft-rules §5 "named constants".

**Fix-direction**: pull defaults from `CostConfig` dataclass field defaults via `dataclasses.fields(CostConfig)` reflection.

---

### FIND-069 — `BacktestConfig.from_dict / to_dict` round-trip non-preserving

**Status**: CONFIRMED
**Severity**: MEDIUM
**Evidence**:
- `to_dict` (config.py:374-385) only emits `zero_dte` block when `enabled=True`. Disabled ZeroDteConfig with non-default fields → fields lost.
- `to_dict` (lines 351-372) OMITS `max_position` field declared at line 303.
- `from_dict` (lines 428-442) does NOT extract `d.get("max_position", ...)`.

**Fix-direction**: always emit `zero_dte` block; add `max_position` to both methods. Deterministic round-trip per hft-rules §7.

---

### FIND-070 — `ExperimentRunner` silently DROPS production-YAML `backtest.min_agreement` / `backtest.min_confidence` ⇒ runs claim "ALL agree" semantic but execute "2/3"

**Status**: NEW (Adv-3) — **MOST CONSEQUENTIAL SILENT-MISCONFIG**
**Severity**: CRITICAL
**Active vs Latent**: ACTIVE — corrupted real experiments
**Evidence**:
- Production YAML `configs/nvda_readability_first_xnas.yaml:14-15` claims "ALL must agree" semantic; sets `backtest.min_agreement: 1.0` and `backtest.min_confidence: 0.65`.
- `ExperimentRunner._build_backtest_config` (`experiment.py:386-398`) only reads 6 fields: `initial_capital`, `position_size`, `allow_short`, `exchange`, `trading_days_per_year`, `periods_per_day`.
- `min_agreement`, `min_confidence` from the `backtest:` block are SILENTLY DROPPED.
- The strategy's `ReadabilityConfig` falls through to its own defaults: `min_agreement=0.667`, `min_confidence=0.65`.
- Operator believes they ran with "ALL agree" (1.0); actually ran "2/3 agree" (0.667).
- Same bug in `arcx.yaml:43-44`.

**Root cause**: FIND-056 dead BacktestConfig fields + `_build_backtest_config` schema mismatch with production YAML. The YAML was authored expecting the fields to work; the wiring was never completed.

**Impact**: any historical experiment run via `ExperimentRunner.from_yaml(configs/nvda_readability_first_*.yaml)` has misleading metadata. The actual gate was 0.667 not 1.0. Trade counts in those runs are ~3× too high relative to the "ALL agree" intent.

**Fix-direction**:
Option A (Quick): extend `_build_backtest_config` to forward `min_agreement` / `min_confidence` to the strategy config.
Option B (Architectural): tighten the YAML schema — move these fields under `strategy:` block and let `_build_strategy` consume them. Fail-loud on unknown `backtest:` fields.

**Open questions**:
- How many runs in `BACKTEST_INDEX.md` were executed via `ExperimentRunner.from_yaml(configs/nvda_readability_first_xnas.yaml)` and are affected? Audit the ledger.
- Is the "ALL agree" semantic actually validated as the right gate, or was the original 1.0 intent a mistake?

**Related**: FIND-056 (dead fields), FIND-073 (production YAMLs functionally orphaned).

---

### FIND-071 — `ExperimentRunner._run_sweep` is single-axis (not Cartesian); silently drops non-list values; no failure policy

**Status**: CONFIRMED (Wave-2 partial-refute: single-axis IS a legitimate design choice)
**Severity**: HIGH for the silent-drop + no-policy parts; design choice for single-axis
**Evidence**: `experiment.py:229-239`.

**Fix-direction**:
- Document the single-axis sweep semantic explicitly OR implement `itertools.product`.
- Fail-loud on non-list values per hft-rules §8.
- Add `--on-failure {continue, abort, retry:N}` policy mirroring hft-ops scheduler.

---

### FIND-072 — `ExperimentRunner._run_sweep` reports first-key-only as `sweep_parameter`

**Status**: NEW (Adv-3)
**Severity**: MEDIUM
**Evidence**: `experiment.py:208` `sweep_param = list(sweep_config.keys())[0]`. If YAML has 2 sweep keys (one is non-list and skipped), `sweep_parameter` records the skipped key, misleading operator.

**Fix-direction**: report all valid sweep keys; align with the iteration loop.

---

### FIND-073 — 4 production `configs/*.yaml` functionally orphaned

**Status**: PART-CONFIRMED
**Severity**: HIGH (silent misleading)
**Evidence**: `configs/nvda_readability_first_xnas.yaml`, `nvda_readability_first_arcx.yaml`, `e1_atm_comparison.yaml`, `e1_deep_itm.yaml`. 
- `ExperimentRunner.from_yaml` loads but with incomplete consumption (FIND-070).
- Standalone scripts (`run_readability_backtest.py`, `run_regression_backtest.py`) use CLI flags NOT YAML.
- No grep hits for `nvda_readability_first` outside the YAML itself.

**Fix-direction**:
Option A: write a thin `scripts/run_experiment.py` that consumes these via `ExperimentRunner.from_yaml`.
Option B: move to `docs/example_configs/` and label as documentation-only.
Option C: extend `_build_backtest_config` schema to consume them fully (FIND-070 fix).

---

### FIND-074 — `ExperimentRunner._build_strategy` rejects `hybrid` and `twap` (despite CLAUDE.md listing)

**Status**: CONFIRMED (Wave-2 reclassified LOW — twap intentional, hybrid orphan)
**Severity**: LOW
**Evidence**: `experiment.py:333-366` dispatches only `regression`, `readability`, `direction`.

**Fix-direction**: extend to support `hybrid` (production-gap; reachable elsewhere); explicitly reject `twap` with cite to BACKTEST_INDEX C2.

---

### FIND-075 — `BacktestConfig.to_dict` does NOT capture `lobbacktest.__version__`

**Status**: NEW (Adv-3)
**Severity**: MEDIUM (replay reproducibility loss; hft-rules §7)
**Evidence**: `config.py:351-386`.

**Fix-direction**: add `lobbacktest_version: __version__` to `to_dict` output + registry result.json + index entry.

---

### FIND-076 — YAML writes don't pin `sort_keys=True`

**Status**: REFUTED-OVERSTATED (PyYAML default is True)
**Severity**: LOW
**Evidence**: `config.py:454`, `registry.py:117`. PyYAML default `sort_keys=True` per Python docs.
**Fix-direction**: defensive `sort_keys=True` for future hardening; not blocking.

---

### FIND-077 — `_load_signal_metadata` silently swallows missing metadata

**Status**: NEW (Adv-3)
**Severity**: LOW
**Evidence**: `experiment.py:414-420`. Returns `{"source": ..., "metadata_available": False}` instead of raising.
**Fix-direction**: raise per hft-rules §5/§8 unless explicit opt-out.

---

### FIND-078 — `BacktestResult.from_dict` does not exist (write-only result)

**Status**: CONFIRMED
**Severity**: LOW
**Note**: relevant to FIND-009 (to_dict omissions cannot cause round-trip miscount because there's no inverse).

---

### FIND-079 — `BacktestConfig.load_yaml` is test-only

**Status**: NEW (informational)
**Evidence**: `config.py:445`. Consumers: only `tests/test_config.py`. Not used by any script or `ExperimentRunner`.
**Fix-direction**: document explicitly OR delete.

---

## §7 Contract plane + signal manifest (FIND-080 … FIND-089)

### FIND-080 — Production code imports from Phase 6 6B.5 deprecated shim

**Status**: CONFIRMED-LOW-SEVERITY (Wave-2 reclassified)
**Severity**: LOW (calendar-tracked; 5+ months runway to 2026-10-31)
**Evidence**: `experiment.py:30`, `engine/vectorized.py:153`. Shim at `data/signal_manifest.py:62-79` emits DeprecationWarning per symbol per process.

**Fix-direction**: re-point both imports to `from hft_contracts.signal_manifest import SignalManifest`. 5-min fix; silences warnings.

---

### FIND-081 — Off-exchange signal exports flow through SEPARATE LOADER (Wave-1 claim refuted)

**Status**: REFUTED-OVERSTATED (Wave-2)
**Severity**: N/A (not a current bug; architectural-gap for future)
**Evidence**:
- `hft-contracts/.../validation.py:509-585` defines `validate_off_exchange_export_contract()` — separate validator.
- `hft-contracts/.../validation.py:588-612` defines `validate_any_export_contract()` — polymorphic dispatcher.
- `hft-feature-evaluator/.../data/loader.py:27,270` consumes off-exchange validator.
- Backtester DataLoader is architecturally MBO-only by design.

**Reframing**: The "silent rejection" framing is wrong. It's an explicit boundary contract. If backtester ever needs off-exchange support, it uses `validate_any_export_contract` (NEW-5 below).

---

### FIND-082 — Producer/consumer canonicalization parity is LOCKED by frozen-fixture test (Wave-1 claim refuted)

**Status**: REFUTED-WRONG (Wave-2)
**Severity**: N/A (healthy)
**Evidence**:
- Producer: `lob-model-trainer/.../export/metadata.py:180-189` uses `asdict` + explicit list-coercion of `horizons`.
- `CompatibilityContract.__post_init__` (`compatibility.py:178-198`) coerces `horizons` list→tuple at construction (canonical in-memory form).
- Consumer recompute: `compatibility.py:218-242` `to_canonical_dict() + sanitize_for_hash` recursive coercion.
- Regression test: `hft-contracts/tests/test_compatibility_contract.py:502-530` `test_compatibility_fingerprint_byte_stability_against_frozen_fixture` — locks bytes.

**Three independent mechanisms enforce parity.** Wave-1's "by coincidence" framing was wrong.

---

### FIND-083 — `_compatibility_from_dict` swallow + tamper-fingerprint = CORRECT design (Wave-1 misread)

**Status**: RECLASSIFIED-AS-FEATURE (Wave-2)
**Severity**: N/A
**Evidence**: `hft-contracts/.../signal_manifest.py:248-257`. Catches malformed-block errors → sets `compatibility=None` while preserving `compat_fp`. validate() at lines 439-444 raises "Tamper indicator" which is the correct diagnostic for "fingerprint exists, block is missing".

**Note**: Wave-1 framing as "swallow with misleading message" was wrong. The error message is diagnostically correct.

---

### FIND-084 — `expected_fields` partial-assertion only handles regression (intentional scope)

**Status**: CONFIRMED-DOCUMENTED (Wave-2)
**Severity**: LOW (legitimate scope)
**Evidence**: `experiment.py:295-325`. Classification strategies don't use `primary_horizon_idx`. Scope-limited by design.

**Fix-direction**: extend to classification when HMHP / similar classification strategies depend on a horizon (currently they don't).

---

### FIND-085 — `validate_export_contract` `strict_completeness=False` default in loader

**Status**: NEW (Adv-4)
**Severity**: MEDIUM
**Evidence**: `loader.py:226-228`. Schema-version is hard-fail; other critical-field absence (n_features, n_sequences, window_size) yields warnings only.

**Fix-direction**: default `strict_completeness=True` for v3p0 corpus; warn on legacy.

---

### FIND-086 — `RegressionStrategy` default `primary_horizon_idx=0` + skipped partial assertion = silent default path

**Status**: NEW (Adv-4)
**Severity**: MEDIUM
**Evidence**: `experiment.py:341` defaults `primary_horizon_idx = params.get("primary_horizon_idx", 0)`. `_expected_compatibility_fields:317` only asserts when user explicitly sets the field.

**Impact**: YAML omitting `primary_horizon_idx` → silent H10 default AND skipped Phase II partial-assertion gate.

**Fix-direction**: require explicit value (no default) in YAML schema; OR always emit the assertion when default fires (warn-only mode).

---

### FIND-087 — `validate_any_export_contract` exists in hft-contracts but not consumed by lob-backtester

**Status**: NEW (Adv-4)
**Severity**: LOW (architectural gap, not current bug)
**Evidence**: `hft-contracts/.../validation.py:588-612`. Backtester `loader.py:39` imports `validate_export_contract` only (MBO-only path).

**Fix-direction**: when off-exchange backtests need to land, migrate loader to use the polymorphic dispatcher. Out of scope today; flagged for future expansion.

---

### FIND-088 — `SignalManifest` reads `signal_metadata.json` via plain `open() + json.load()` (no locking, no retry-on-partial-write)

**Status**: NEW (Adv-4)
**Severity**: LOW (operationally safe — producer is sequential + atomic)
**Evidence**: `hft-contracts/.../signal_manifest.py:199-200`.
**Fix-direction**: documentation only; flagged if multi-process producer ever lands.

---

### FIND-089 — `ContractError` class identity post-REV-2 verified single class

**Status**: CONFIRMED-HEALTHY (Wave-2)
**Severity**: N/A
**Evidence**: `hft-contracts/.../signal_manifest.py:43` `from hft_contracts.validation import ContractError`. Both import paths resolve to same class.

---

## §8 Registry + orchestration (FIND-090 … FIND-099)

### FIND-090 — `BacktestRegistry._save_index` is NOT atomic (sister bug of #PY-73)

**Status**: CONFIRMED-CRITICAL
**Severity**: HIGH (corruption hazard during sweep)
**Active vs Latent**: ACTIVE under SIGKILL or system crash
**Evidence**: `registry.py:69-70`. `open(path, "w") + json.dump(...)` — truncates BEFORE writing. SIGKILL between truncate and flush leaves empty/partial JSON. `__init__:64-66` reads with `json.load` — uncaught `JSONDecodeError` raises on next instantiation.

**Compounding**: `registry.py:112-113` `result.json` + `registry.py:115-117` `config.yaml` are also non-atomic. Only `equity_curve.npy` (line 124) was migrated to `atomic_write_npy` (#PY-73 closure).

**Fix-direction**: migrate 3 sites to `hft_contracts.atomic_io.atomic_write_json`. For YAML, either render to bytes and atomic-write bytes, OR convert config artifact to JSON.

**Constraints (must preserve)**:
- equity_curve.npy already uses atomic_write_npy — preserve.
- Pre-2026-05-05 P0 fix's PascalCase-with-fallback contract at `registry.py:126-141` — preserve.

**Related**: FIND-091 (race condition under parallel writes), FIND-130 (thread-unsafe `_index` mutation), encoded lesson #15.

---

### FIND-091 — `BacktestRegistry._index` mutation not thread-safe

**Status**: NEW (Adv-7)
**Severity**: HIGH (race condition under any parallel sweep)
**Active vs Latent**: LATENT (current sweeps are sequential)
**Evidence**: `registry.py:62-67, 136-147`. `self._index = json.load(f)` at init; `self._index[run_id] = {...}` in register() with no lock. ThreadPoolExecutor sweeping via ExperimentRunner would race.

**Fix-direction**: add file-lock (`filelock` package, already a hft-ops dep). Or document explicitly that BacktestRegistry is single-process only.

---

### FIND-092 — `BacktestRegistry.run_id` second-resolution collision

**Status**: CONFIRMED (Wave-2 reclassified MEDIUM — latent under default sequential sweep)
**Severity**: MEDIUM
**Evidence**: `registry.py:97` `%Y%m%d_%H%M%S`. Two `register()` calls in same second produce identical run_ids.

**Fix-direction**: microsecond precision `%Y%m%d_%H%M%S_%f` + `uuid4().hex[:8]` suffix.

---

### FIND-093 — `BacktestRegistry.register` uses `datetime.now()` (local TZ) — not UTC

**Status**: NEW (Adv-7)
**Severity**: HIGH (cross-machine reproducibility break; hft-rules §3 violation)
**Evidence**: `registry.py:97, 104`. Local timestamps non-portable.

**Fix-direction**: `datetime.now(timezone.utc).strftime(...)`. Add `created_at_utc` field.

---

### FIND-094 — `BacktestRegistry` and hft-ops `ExperimentLedger` are ORTHOGONAL, not duplicates (Wave-1 claim refuted)

**Status**: REFUTED-WRONG (Wave-2)
**Severity**: N/A
**Evidence**:
- `BacktestRegistry` writes per-run `equity_curve.npy` arrays (binary artifacts).
- `hft-ops/.../ledger/runs/*.json` stores scalar metrics + fingerprints.
- Two registries cover orthogonal artifact classes.

**Conclusion**: KEEP BOTH. Document the boundary explicitly in CLAUDE.md as a two-tier ledger architecture (binary vs scalar).

---

### FIND-095 — `ExperimentRunner` parallel orchestration vs hft-ops manifest — both are LEGITIMATE (Wave-1 claim partially refuted)

**Status**: REFUTED-OVERSTATED (Wave-2)
**Severity**: LOW (architectural separation, not duplication)
**Evidence**: 
- hft-ops sweep manifest = YAML-CLI API.
- ExperimentRunner = Python API for interactive notebook / ad-hoc exploration.
- Different surfaces, different audiences.

**Conclusion**: KEEP BOTH, but document the boundary. If unified, defer to a dedicated retirement cycle that verifies hft-ops sweep manifest schema covers ALL ExperimentRunner functionality (including the Python-API niche).

**Related**: FIND-094.

---

### FIND-096 — `BacktestSummary` dataclass at `registry.py:26-49` is defined but never used (parallel to dict literal)

**Status**: NEW (Adv-7)
**Severity**: INFO (parallel-implementation hazard)
**Evidence**: `BacktestRegistry.register` constructs a dict literal at `registry.py:136-146` instead of using `BacktestSummary`. Dead code OR drift hazard.

**Fix-direction**: use `BacktestSummary` consistently, OR delete the dataclass.

---

### FIND-097 — Registry filesystem race when multiple `ExperimentRunner` instances run concurrently

**Status**: NEW (Adv-3)
**Severity**: LOW (compound of FIND-090 + FIND-091)
**Evidence**: `experiment.py:271` default base dir = `outputs/backtests`. `_index` read-then-write without lock.

**Fix-direction**: file-lock OR per-process subdirectory.

---

### FIND-098 — `_save_index` opens with `"w"` not appending — full rewrite on each register

**Status**: NEW (Adv-3)
**Severity**: LOW (compound of FIND-090)
**Evidence**: `registry.py:69`. Combined with non-atomicity makes mid-write SIGKILL CATASTROPHIC.

**Fix-direction**: incremental append OR atomic full rewrite per FIND-090 fix.

---

### FIND-099 — `BacktestSummary` exposes `to_dict` but `BacktestRegistry` ignores it (parallel-impl)

**Status**: NEW (related to FIND-096)
**Severity**: INFO
**Fix-direction**: unify per FIND-096.

---

## §9 Tests + coverage + CI hygiene (FIND-100 … FIND-109)

### FIND-100 — `tests/test_data/` and `tests/test_stats/` are EMPTY directories

**Status**: CONFIRMED
**Severity**: HIGH (silent 0% coverage on key modules)
**Evidence**: `ls tests/test_{data,stats}/` returns only `.` and `..`. `BacktestStats` (stats/stats.py, 358 LOC) has zero test coverage. `PriceExtractor`/`NormalizationParams` (data/prices.py, 285 LOC) has zero direct tests. `DataLoader` (data/loader.py, 369 LOC) has minimal coverage via `test_loader_strict_validation.py` (7 tests for schema-version paths only).

**Fix-direction**: 
- Delete empty dirs (signaling no-coverage intent), OR
- Populate with regression tests: `test_loader.py` (NPY-shape, multi-day loading, missing/duplicate handling); `test_prices.py` (denormalization formula); `test_stats.py` (cumulative, aggregation, streaming-reduction).

---

### FIND-101 — Tautological tests in metrics suite

**Status**: CONFIRMED
**Severity**: MEDIUM (hft-rules §6 violation)
**Evidence**:
- `tests/test_metrics/test_returns.py:24,36` — `expected = np.prod(1 + returns) - 1` mirrors impl byte-for-byte.
- `tests/test_metrics/test_risk.py:59,125,321` — same anti-pattern for Sharpe / Sortino.
- `tests/test_metrics/test_trading.py:254` — Expectancy.

**Fix-direction**: replace with hand-calculated expected values + paper citation per hft-rules §6. e.g., `expected = 0.061106  # Cited: textbook X, eq Y`.

---

### FIND-102 — `@pytest.mark.integration` marker is NOT registered in `pyproject.toml`

**Status**: CONFIRMED
**Severity**: MEDIUM (PytestUnknownMarkWarning + `-m "not integration"` filter unreliable)
**Evidence**: `pyproject.toml:55-59` has no `markers = [...]`. `tests/test_run_regression_backtest_manifest.py:143, 364, 527` use the marker.

**Fix-direction**: add `markers = ["integration: subprocess + real fixture chain (slow); skip on fast CI via -m \"not integration\""]` to `pyproject.toml [tool.pytest.ini_options]`.

---

### FIND-103 — Test count drift across 5 docs

**Status**: CONFIRMED (Wave-2 verified empirical count)
**Severity**: MEDIUM
**Evidence**: Empirical `pytest --collect-only` = **414 tests** (some docs say 415 — grep difference accounts for class-based parametrize).
- `lob-backtester/CLAUDE.md:11`: 359 + 8 = 367 ✗
- `CODEBASE.md:3`: 353 (345 + 8) ✗
- `README.md:7`: 353 ✗
- `BACKTESTER_AUDIT_PLAN.md`: 330 ✗
- root `CLAUDE.md`: 414 ✓

**Fix-direction**: per hft-rules §11 numeric facts must be regenerated, not hand-typed. Replace with `{{TEST_COUNT}}` placeholder + automated refresh, OR delete the claim ("see `pytest --collect-only`").

---

### FIND-104 — Integration tests have no deterministic seed (`torch.manual_seed`, `np.random.seed`, PYTHONHASHSEED)

**Status**: CONFIRMED (Adv-7 verified zero `np.random.seed` calls anywhere)
**Severity**: HIGH (hft-rules §7 violation; tests are non-reproducible)
**Evidence**: `tests/test_integration_real_data.py:326, 452, 463, 482, 486, 500` use `rtol=1e-5, atol=1e-5` but no seed pin. `data_dir` fixture at line 256 picks "the first available day" alphabetically — not reproducible across re-extractions.

**Fix-direction**: pin `torch.manual_seed(42)`, `np.random.seed(42)` at top of integration test class. Capture golden outputs + commit.

---

### FIND-105 — Real-data path uses `torch.load(..., weights_only=False)` (PyTorch ≥2.6 deprecation; RCE risk)

**Status**: NEW (Adv-7)
**Severity**: LOW (test-only path; local checkpoint)
**Evidence**: `test_integration_real_data.py:188`.

**Fix-direction**: pin `weights_only=True` when feasible. For Phase 8C+ checkpoints with numpy globals (e.g., RNG state), document explicit `weights_only=False`.

---

### FIND-106 — Integration test sample-count assertions are loose

**Status**: NEW (Adv-7)
**Severity**: LOW
**Evidence**: `test_integration_real_data.py:288` checks labels are in `[-1, 0, 1]` but doesn't verify SCHEMA version or v3p0 corpus membership. Stale `data/exports/nvda_balanced` path may already be broken on post-Phase-O.

**Fix-direction**: pin schema version check + v3p0 fixture.

---

### FIND-107 — Cross-module synthetic-fixture E2E test absent

**Status**: NEW (Adv-7)
**Severity**: MEDIUM
**Evidence**: Integration coverage is binary: either real-data path (skipped on fast CI) OR synthetic-fixture path (test_signal_manifest_*, test_phase2_*) which test SignalManifest in isolation but never run a full backtest on synthetic fixtures.

`tests/test_run_regression_backtest_manifest.py::TestPerTradePnlsDump` is closest analog but is subprocess-based (slow; marked integration; un-registered marker).

**Fix-direction**: factor `_construct_mock_signal_dir` into session-scoped fixture; add fast in-process E2E synthetic test exercising `BacktestData.from_signal_dir → VectorizedEngine.run → ZeroDtePnLTransformer` without subprocess.

---

### FIND-108 — `pytest-cov` installed but no coverage gate in CI

**Status**: CONFIRMED
**Severity**: LOW
**Evidence**: `pyproject.toml:40` lists `pytest-cov>=4.0`. CI workflow runs `pytest -q --tb=short -o addopts=""` — no `--cov` or `--cov-fail-under`.

**Fix-direction**: set `--cov-fail-under=70%` ratchet in CI (current baseline 61% per BACKTESTER_AUDIT_PLAN, so 70% sets a forward ratchet).

---

### FIND-109 — Phase 6 6D.1 soft-WARN hook regex does NOT cover `lob-backtester/scripts/`

**Status**: CONFIRMED
**Severity**: LOW (hygiene gap)
**Evidence**: `.claude/hooks/check_scripts_header.py` regex covers `(lob-model-trainer|hft-feature-evaluator)/scripts/` only.

**Fix-direction**: extend regex to include `lob-backtester/scripts/`. Add fossil headers per Phase 6 6D.1 convention to: `param_sweep.py` (`# STATUS: experimental fossil`), `e5_regime_filter_test.py` (`# STATUS: experimental fossil`), `run_regression_backtest.py` (`# PRODUCTION INFRA`), etc.

**NOTE**: `backtest_deeplob.py` is **NOT** a fossil — Adv-5 confirmed it's the production default for hft-ops `BacktestingStage.script`. Wave-1 misclassified.

---

## §10 Security (FIND-110 … FIND-114)

### FIND-110 — `np.load` allows pickle by default — RCE-via-malicious-.npy

**Status**: NEW (Adv-7)
**Severity**: HIGH
**Active vs Latent**: LATENT (requires attacker control over signal dir, but path traversal makes this plausible — see FIND-111)
**Evidence**: `vectorized.py:160-200`, `loader.py:190+198`, `registry.py` reads (registry.py:65,161). NumPy `.npy` files with magic header `\x93NUMPY` followed by pickle headers can execute arbitrary code.

**Fix-direction**: explicit `allow_pickle=False` at every `np.load(...)` call site.

**STATUS:CLOSED 2026-05-14** (Cluster H security sweep). Explicit `allow_pickle=False` added at all **25 `np.load(...)` callsites** across the lob-backtester repository:
- `src/lobbacktest/data/loader.py` — 4 sites (lines 190, 198, 284, 292)
- `src/lobbacktest/engine/vectorized.py` — 11 sites (lines 163-168, 187, 192, 198, 200, 203)
- `tests/test_run_regression_backtest_manifest.py` — 1 site (line 636)
- `tests/test_integration_real_data.py` — 2 sites (lines 98, 99)
- `scripts/backtest_deeplob.py` — 2 sites (lines 84, 85)
- `scripts/e5_regime_filter_test.py` — 3 sites (lines 77, 121, 151; `mmap_mode="r"` preserved)
- `scripts/param_sweep.py` — 2 sites (lines 100, 101)

NEW regression-lock test at `tests/test_security/test_np_load_allow_pickle_false.py::TestFind110AllowPickleFalseLock::test_every_np_load_passes_allow_pickle_false` scans every `np.load(` callsite in `src/`, `tests/`, and `scripts/` and fails the suite if any future contribution omits the kwarg. Known limitation: regex-based scan does NOT catch `from numpy import load as _l` aliases or fully-qualified `numpy.load(...)` (codebase convention is `import numpy as np`; zero current occurrences of the aliased forms).

Audit context (Wave 1+2+pre-impl): Pre-Impl Agent 1 verified the production TB v3p0 corpus at `data/exports/nvda_v3p0_tb_pt40_sl20_h30/` loads cleanly with `allow_pickle=False`; every `.npy` file produced by upstream (feature-extractor + trainer signal export) contains pure numeric arrays — ZERO regression surface. The FIND-111 path-traversal sister (path resolution check) remains OPEN as a separate hardening cycle. Locks pickle-RCE vector per hft-rules §8 and Appendix A lesson #29.

---

### FIND-111 — Path-traversal in `signal_dir`

**Status**: NEW (Adv-7)
**Severity**: MEDIUM
**Evidence**: `vectorized.py:131-211` `from_signal_dir(signal_dir)` accepts unsanitized string from CLI args / YAML. No `Path(signal_dir).resolve()` check. Malicious YAML manifest with `signals.dir: "../../../etc/passwd"` would attempt to load arbitrary files (bounded by `.npy` requirement).

**Fix-direction**: resolve and assert path is under a configured `data_root`.

---

### FIND-112 — `yaml.dump` produces a consumer-untrusted sink

**Status**: NEW (Adv-7)
**Severity**: HIGH (registry config.yaml is operator-trusted output, but inputs are partly user-controlled)
**Evidence**: `registry.py:116-117`. `config_dict` mixes safe `BacktestConfig.to_dict()` and arbitrary YAML user input from `ExperimentRunner._serialize_config(params)`. When loaded downstream with `yaml.load` (not `safe_load`), RCE possible.

**Fix-direction**: emit JSON instead of YAML for registry artifacts; OR document "never `yaml.load` this; only `yaml.safe_load`".

---

### FIND-113 — `json.load` on user-controlled paths without size limit

**Status**: NEW (Adv-7)
**Severity**: MEDIUM
**Evidence**: `loader.py:210, 304`, `experiment.py:418`, `registry.py:65, 161`. Memory exhaustion risk.

**Fix-direction**: `os.stat().st_size` precheck before `json.load`.

---

### FIND-114 — `yaml.safe_load` used consistently (verified healthy)

**Status**: HEALTHY (Wave-2 verified)
**Evidence**: `config.py:448`, all CLI YAML loads use `yaml.safe_load`. No `yaml.load` calls in source.

---

## §11 Performance + memory (FIND-115 … FIND-119)

### FIND-115 — Python per-sample loop in `VectorizedEngine.run`

**Status**: NEW (Adv-7) — but acknowledged in code docstring
**Severity**: HIGH (latent perf for large backtests)
**Evidence**: `engine/vectorized.py:281-433`. Python `for i in range(n)` with branching and dataclass allocation per sample. At 1M samples → 30-60s wall time.

**Fix-direction**: numba JIT (large scope) OR strict numpy vectorization of P&L given upfront-computed signals.

**Constraints (must preserve)**: 15 encoded lessons (§A); P&L semantics; trade ordering.

---

### FIND-116 — `plot_comparison` materializes all equity curves in memory

**Status**: NEW (Adv-7)
**Severity**: HIGH (50-model × 1M-sample = 400MB allocation)
**Evidence**: `reports/plots.py:206-266`. `plot_positions:269-326` similar.

**Fix-direction**: streaming/downsampling option for large comparisons.

---

### FIND-117 — `BacktestResult.to_dict()` calls `.tolist()` on equity_curve + positions + returns

**Status**: NEW (Adv-7)
**Severity**: MEDIUM
**Evidence**: `types.py:319-321`. For 1M samples × 4 arrays × 8 bytes = 32MB → 64MB Python lists. Registry serializes → second copy.

**Fix-direction**: serialize as `.npy` binary instead of JSON `.tolist()`. Or downsample.

---

### FIND-118 — `BacktestResult.max_drawdown` is `@property` (recomputed on every access)

**Status**: NEW (Adv-7)
**Severity**: LOW
**Evidence**: `types.py:256-265`. Every print of `result.max_drawdown` re-runs `np.maximum.accumulate(equity_curve)`. The metric is in `result.metrics`.

**Fix-direction**: cache, or delegate to `metrics.MaxDrawdown` (also resolves FIND-030 disagreement).

---

### FIND-119 — Matplotlib figures never closed (memory leak)

**Status**: NEW (Adv-7)
**Severity**: HIGH (memory leak in sweep)
**Evidence**: `reports/plots.py` all 5 plot functions return `fig` but no caller is guaranteed to `plt.close(fig)`. 50-plot sweep → 500MB+ leaked.

**Fix-direction**: callers must `plt.close(fig)` after consumption. Add `@contextmanager` helper OR `with plt.ioff()` discipline. Document in `reports/plots.py` module docstring.

---

## §12 Architectural debt (FIND-120 … FIND-129)

### FIND-120 — Engine is single-symbol; no multi-symbol portfolio P&L

**Status**: NEW (Adv-7) — CRITICAL architectural debt
**Severity**: CRITICAL (long-term)
**Evidence**: `engine/vectorized.py:240-475` `run(data: BacktestData, ...)` takes ONE prices, ONE labels, ONE predictions. No portfolio, correlation, or symbol multiplexing.

**Fix-direction**: design phase needed. Options: (a) `BacktestData.symbol: str` field + portfolio aggregator; (b) per-symbol BacktestResult + composition layer; (c) wholesale redesign.

**Constraints**: must preserve current single-symbol P&L semantics for backward-compat.

**Open questions**: when does multi-symbol become needed? Are correlation effects modeled, or independent runs?

---

### FIND-121 — No options-native execution path (current path is equity→option projection)

**Status**: NEW (Adv-7) — CRITICAL architectural debt
**Severity**: CRITICAL (long-term)
**Evidence**: `engine/zero_dte.py:201-334` PROJECTS option P&L from equity backtest trade pairs. Does NOT model: per-strike orderbook, exercise/assignment, put-call symmetry, early exercise of American options, IV smile, Greeks beyond constant `delta=0.50/0.95`. `prefer_calls` is binary; no put-skew strategy.

**Fix-direction**: design phase. Either (a) options-native `OptionStrategy` + Asset ABC, OR (b) keep projection path + document it explicitly as "approximation for backtest only".

---

### FIND-122 — `from_signal_dir` is batch-only; no streaming readiness

**Status**: NEW (Adv-7)
**Severity**: HIGH (long-term)
**Evidence**: `engine/vectorized.py:80-211` loads everything into memory. No `from_signal_stream`.

**Fix-direction**: design phase. The per-sample loop is already streaming-compatible; only the loader needs a streaming adapter.

---

### FIND-123 — Timezone handling missing on entry/exit windows + DST

**Status**: NEW (Adv-7)
**Severity**: HIGH
**Evidence**: `engine/zero_dte.py:265-266` + `config.py:265-266`. `entry_window_*_et` strings never parsed against actual trade timestamps. DST transitions for 2-hour vs 1-hour pre-close completely absent.

**Fix-direction**: parse via `hft_contracts.timestamp_utils.parse_iso8601_utc` + DST-aware `hft_statistics::time::regime` Rust crate (Python wrapper TBD).

**Related**: FIND-059.

---

### FIND-124 — Currency hardcoded USD throughout

**Status**: NEW (Adv-7)
**Severity**: HIGH (long-term)
**Evidence**: `$` formatting in summaries, `initial_capital=100_000.0` (no currency tag), `commission_per_contract` in USD.

**Fix-direction**: add `currency: str = "USD"` field on BacktestConfig. Format display layer per currency.

---

### FIND-125 — `Trade.size` is `float` (allows fractional shares; US equities don't support)

**Status**: NEW (Adv-7)
**Severity**: MEDIUM (documentation silent)
**Evidence**: `types.py:67-92`.

**Fix-direction**: document explicitly (US equity: round to integer; crypto/FX: float). Add per-instrument size rounding policy.

---

### FIND-126 — `np.diff(equity)/equity[:-1]` mixes intraday and overnight returns

**Status**: NEW (Adv-7)
**Severity**: MEDIUM
**Evidence**: `vectorized.py:445-447`. Returns computed across day boundaries (overnight gaps). Sharpe ratio annualization assumes intra-period returns. Overnight return spikes inflate volatility → understated Sharpe.

**Fix-direction**: detect day-boundary via timestamp; either drop the overnight return OR carry it as a separate metric (overnight Sharpe).

---

### FIND-127 — Maker/taker cost asymmetry declared but not exercised

**Status**: CONFIRMED (sister of FIND-064)
**Severity**: MEDIUM
**Evidence**: `CostConfig.maker_rebate_bps` + `taker_fee_bps` declared separately; `compute_cost` only uses `total_bps` sum. No `Trade.liquidity_flag`.

**Fix-direction**: either wire `liquidity_flag` (large scope) or remove `maker_rebate_bps` field.

---

### FIND-128 — `BacktestResult` keeps full arrays in memory across sweep (no eviction)

**Status**: NEW (Adv-7)
**Severity**: MEDIUM
**Evidence**: `types.py:202-211`. 1M samples × 6 arrays × 8 bytes = 48MB per result. 100-run sweep = 4.8GB.

**Fix-direction**: streaming / disk-backed result OR explicit eviction policy.

---

### FIND-129 — Encoded NVDA-centric assumption: `initial_capital=$100K` → $100 reference price

**Status**: CONFIRMED (sister of FIND-010)
**Severity**: MEDIUM
**Evidence**: `vectorized.py:518-519` comment "we assume prices are roughly $100+".
**Fix-direction**: per FIND-010 — replace with explicit config.

---

## §13 Concurrency + threading (FIND-130 … FIND-134)

### FIND-130 — `BacktestRegistry._index` non-thread-safe (compound with FIND-090, FIND-091)

**Status**: NEW (Adv-7)
**Severity**: HIGH (latent under any parallel sweep)
**Reference**: FIND-091.

---

### FIND-131 — Strategy `self.prices` state-leak across `generate_signals` calls

**Status**: NEW (Adv-5 + Adv-7 cross-confirmed)
**Severity**: HIGH-LATENT
**Reference**: FIND-054.

---

### FIND-132 — `_WARNED: set[str]` module-level mutable state in shim is not thread-local

**Status**: NEW (Adv-7)
**Severity**: LOW (cosmetic; worst-case warning fires twice)
**Evidence**: `data/signal_manifest.py:58`. Module-global; two threads importing shim symbols simultaneously can race.

**Fix-direction**: use `threading.local()` OR document as best-effort.

---

### FIND-133 — `HoldingPolicy.should_exit` may carry state across calls (latent contract drift)

**Status**: NEW (Adv-5)
**Severity**: LOW
**Evidence**: `holding.py` — verified policies are stateless today. Latent if a future subclass adds state.

**Fix-direction**: enforce statelessness via `@final` decorator OR test contract.

---

### FIND-134 — `BacktestContext.update` not reentrant under metric-dependency cycles

**Status**: NEW (Adv-2)
**Severity**: LOW
**Reference**: FIND-035.

---

## §14 Numerical precision + fp (FIND-135 … FIND-144)

### FIND-135 — `final_equity == equity_curve[-1]` epsilon `1e-10` impossible at large capital

**Status**: NEW (Adv-7)
**Reference**: FIND-016.

---

### FIND-136 — Division-by-zero in `RegressionStrategy._build_holding_state` only guarded for entry_price > EPS

**Status**: NEW (Adv-7)
**Severity**: MEDIUM
**Evidence**: `regression.py:112-116`. `entry_price > EPS` guarded; `current_price == 0` not.

**Fix-direction**: add `current_price > EPS` guard.

---

### FIND-137 — `theta_bsm_per_share` floors at expiry (real theta is infinite)

**Status**: NEW (Adv-7)
**Severity**: HIGH (silent under-cost on tail-of-session trades)
**Evidence**: `engine/zero_dte.py:70-80`. `if minutes_remaining < 1.0: return 0.0`. At 15:59:30 (30 sec to close), real theta is many bps/min; floor underestimates cost.

**Fix-direction**: cap at finite-but-large value, OR explicitly raise on `minutes_remaining < 1`.

---

### FIND-138 — `BacktestResult.max_drawdown` masks NaN as 0

**Status**: NEW (Adv-7) — sister of FIND-030
**Severity**: MEDIUM
**Evidence**: `types.py:264`. `np.where(np.isfinite(drawdown), drawdown, 0.0)`.

**Fix-direction**: surface counter of nan-occurrences. Delegate to metric (FIND-030 + FIND-118).

---

### FIND-139 — `compute_cost` ignores commission direction (round-trip = 2× commission)

**Status**: NEW (Adv-7)
**Reference**: FIND-065.

---

### FIND-140 — `np.prod(1+returns)` overflow on long histories

**Status**: NEW (Adv-2/Adv-7)
**Reference**: FIND-015, FIND-031.

---

### FIND-141 — Silent NaN→0 in returns calc

**Status**: REFUTED-OVERSTATED
**Reference**: FIND-006.

---

### FIND-142 — `RegressionStrategy` magnitude gate passes NaN silently

**Status**: NEW (Adv-2)
**Reference**: FIND-046.

---

### FIND-143 — Float32 vs Float64 boundary not documented

**Status**: NEW (Adv-7 candidate)
**Severity**: LOW
**Evidence**: arrays loaded from `.npy` may be float32 (NVDA exports are float32 per `pipeline_contract.toml`); engine internals use float64. Boundary not validated.

**Fix-direction**: document explicit conversion in `BacktestData.__post_init__`.

---

### FIND-144 — `EPS = 1e-12` triplicated across modules

**Status**: NEW (Adv-7)
**Severity**: LOW (cosmetic; values identical)
**Evidence**: `engine/zero_dte.py:38`, `hft_metrics._sanitize.EPS`, root CLAUDE.md cites `EPS = 1e-8` / `FLOAT_CMP_EPS = 1e-10`.

**Fix-direction**: SSoT-driven from `hft_contracts.constants` (new module) OR each module declares with citation.

---

## §15 Reproducibility + determinism (FIND-145 … FIND-149)

### FIND-145 — Zero `np.random.seed` calls anywhere in source tree

**Status**: NEW (Adv-7)
**Severity**: HIGH (any stochastic metric will be non-deterministic)
**Evidence**: grep across `src/lobbacktest/` returns 0 calls.

**Fix-direction**: any stochastic metric (bootstrap CI etc) must take an explicit `seed` param. Document the discipline.

---

### FIND-146 — `BacktestRegistry.register` uses `datetime.now()` local TZ

**Status**: NEW (Adv-7)
**Reference**: FIND-093.

---

### FIND-147 — `sorted(split_dir.glob("*_sequences.npy"))` filesystem-order dependent

**Status**: NEW (Adv-7)
**Severity**: MEDIUM (caught by sort, but fragile)
**Evidence**: `loader.py:175`. `glob` ordering is OS-dependent; `sorted()` saves it.

**Fix-direction**: document the sort-dependence; consider explicit alphanumeric sort OR ISO-date sort.

---

### FIND-148 — `dict.keys()` iteration in `experiment.py:208`

**Status**: NEW (Adv-7)
**Severity**: LOW (Python 3.7+ preserves insertion order)
**Reference**: FIND-072.

---

### FIND-149 — `lobbacktest.__version__` not captured in result metadata

**Status**: NEW (Adv-3)
**Reference**: FIND-075.

---

## §16 Cross-repo consolidation (FIND-150 … FIND-159)

### FIND-150 — `PredictionIC` should delegate to `hft_metrics.ic.spearman_ic`

**Status**: CONFIRMED-WITH-CAVEAT (Wave-2)
**Severity**: MEDIUM (orphan today per FIND-021 — wire OR delete decision)
**Evidence**:
- Backtester: `regression_prediction.py:108-134` calls `scipy.stats.spearmanr` directly.
- Upstream: `hft-metrics/src/hft_metrics/ic.py:55-91` `spearman_ic(x,y) -> tuple[float, float]`. Cluster Z (2026-05-11) migrated to NaN sentinel on failure.

**Drift**: backtester returns `0.0` with RuntimeWarning on degenerate; SSoT returns `(NaN, NaN)`. Migration requires `if not np.isfinite(rho): rho = 0.0` boundary adapter (precedent: `hft-ops/.../ledger/statistical_compare.py:74-82`).

**Fix-direction (if wiring rather than deleting per FIND-021)**: 
```python
from hft_metrics.ic import spearman_ic
rho, _ = spearman_ic(self._predicted, self._actual)
if not np.isfinite(rho):
    warnings.warn(...); return {"PredictionIC": 0.0}
return {"PredictionIC": float(rho)}
```

---

### FIND-151 — `PredictionCorrelation` should delegate to `hft_metrics.ic.pearson_r`

**Status**: CONFIRMED-WITH-CAVEAT (Wave-2)
**Severity**: MEDIUM (same conditions as FIND-150)
**Reference**: same pattern.

---

### FIND-152 — `PredictionMSE` — add `hft_metrics.regression.mean_squared_error` upstream OR keep local

**Status**: CONFIRMED-CONDITIONAL
**Severity**: LOW (only 1 consumer today; below §0 reuse-first threshold of ≥2)
**Evidence**: `hft-metrics/.../regression.py` has MAE, RMSE — but no MSE.

**Fix-direction**: if 2nd consumer emerges (e.g., cv_trainer validation metric extension), add upstream. Else keep local.

---

### FIND-153 — `DirectionalAccuracy` (regression flavor) should delegate to `hft_metrics.regression.directional_accuracy`

**Status**: CONFIRMED (Wave-2)
**Severity**: LOW
**Evidence**: `hft-metrics/src/hft_metrics/regression.py:97-119`.

**Fix-direction**: mechanical migration (if not deleting per FIND-021).

---

### FIND-154 — Label encoding constants duplicated 3× (Wave-1 D3 confirmed)

**Status**: CONFIRMED
**Severity**: MEDIUM (Class A SSoT discipline; today values match)
**Evidence**:
- `lob-backtester/.../labels.py:25-32`
- `lob-backtester/.../metrics/prediction.py:21-28`
- `hft-contracts/.../labels.py:32-67`

**Fix-direction**: backtester `labels.py` becomes re-export shim of `hft_contracts.labels`. `metrics/prediction.py` imports directly. `LabelMapping` dataclass + predicates stay backtester-local.

---

### FIND-155 — BSM theta upstream — REFUTED-OVERSTATED (no upstream module exists; 1 consumer)

**Status**: REFUTED-OVERSTATED (Wave-2)
**Severity**: N/A
**Verdict**: KEEP-LOCAL. Below §0 reuse-first threshold; cross-language SSoT pattern doesn't apply (Rust impl in `opra-statistical-profiler` is different math — full BSM, not ATM theta simplification).

**Reconsider when**: 2nd Python consumer materializes (e.g., live-execution module computing realtime theta cost).

---

### FIND-156 — OpraCalibratedCosts upstream — SCOPE-NARROWED (migrate scalars only, keep dataclass local)

**Status**: REFUTED-WRONG → RECLASSIFIED
**Severity**: MEDIUM
**Evidence**:
- Backtester dataclass: `lob-backtester/.../config.py:135`.
- hft-ops "consumer" is a flat scalar `cost_breakeven_bps: float = 1.4` at `manifest/schema.py:414` — NOT the dataclass.
- Audit conflated "value derived from OpraCalibratedCosts" with "imports OpraCalibratedCosts".

**Fix-direction (SCOPE NARROWED)**:
Move only the 7 scalar constants to `hft_contracts._generated.py` (or new `hft_contracts.cost_constants.py`):
- `BREAKEVEN_BPS_DEEP_ITM = 1.4`
- `BREAKEVEN_BPS_ATM_PUT = 3.8`
- `BREAKEVEN_BPS_ATM_CALL = 4.9`
- `IBKR_COMMISSION_PER_CONTRACT_USD = 0.70`
- `OPRA_ATM_CALL_HALF_SPREAD = 0.015`
- `OPRA_ATM_PUT_HALF_SPREAD = 0.010`
- `OPRA_DEEP_ITM_HALF_SPREAD = 0.005`

Backtester `OpraCalibratedCosts` dataclass imports these constants. hft-ops `cost_breakeven_bps` defaults to `BREAKEVEN_BPS_DEEP_ITM`. Single edit point when IBKR fill set is refreshed.

---

### FIND-157 — `BacktestRegistry` orthogonal to hft-ops ExperimentLedger — KEEP BOTH

**Status**: REFUTED-WRONG (Wave-2)
**Reference**: FIND-094.

---

### FIND-158 — `ExperimentRunner` Python API legitimate vs hft-ops YAML — KEEP BOTH

**Status**: REFUTED-OVERSTATED (Wave-2)
**Reference**: FIND-095.

---

### FIND-159 — `hft_contracts.atomic_io.atomic_write_json` SSoT not fully consumed (registry has 3 sites missing)

**Status**: NEW
**Reference**: FIND-090, FIND-098. Encoded lesson #15 is **DOC-DISAGREES-WITH-CODE** — only equity_curve.npy migrated; result.json, config.yaml, index.json not.

---

## §17 Documentation drift (FIND-160 … FIND-169)

### FIND-160 — Test count cited differently in 5+ docs

**Status**: CONFIRMED
**Reference**: FIND-103.

### FIND-161 — `BACKTESTER_AUDIT_PLAN.md` is 2 months stale (2026-03-17 last update)

**Status**: CONFIRMED
**Severity**: HIGH (doc misrepresents current state)
**Evidence**: `BACKTESTER_AUDIT_PLAN.md:205`. Predates Phase II / V / X / Y / Z + v3p0 corpus + Phase 7.5 orchestrator closures.

**Fix-direction**: update audit-plan with banner pointing at this VALIDATION_FINDINGS doc; preserve historical record but mark superseded.

### FIND-162 — 316 vs 318 IBKR fills — RECLASSIFIED (documented dual-reconciliation)

**Status**: RECLASSIFIED (Wave-2)
**Severity**: LOW
**Evidence**: `IBKR-transactions-trades/COST_AUDIT_2026_03.md:7` documents 316 NVDA + 2 GLD = 318 total. Both numbers are correct in different contexts.
**Fix-direction**: footnote both numbers in code docstrings and BACKTEST_INDEX. Pick one as canonical for any new doc.

### FIND-163 — `lob-backtester/CLAUDE.md` "All strategies use centralized label encoding — never hardcode 0/1/2" is PARTIAL TRUTH

**Status**: CONFIRMED
**Reference**: FIND-042. TRUE for readability/regression/hybrid/holding; FALSE for direction.py and metrics/prediction.py.

### FIND-164 — `readability.py:8-19` module docstring stale (claims `min_agreement == 1.0`)

**Status**: CONFIRMED
**Reference**: FIND-048.

### FIND-165 — `hft-ops/.../stages/backtesting.py:6` docstring lies about `backtest_deeplob.py` as default

**Status**: CONFIRMED (Adv-5)
**Severity**: LOW
**Evidence**: hft-ops orchestrator docstring claims default; reality is operator-specified per manifest.

**Fix-direction**: refresh docstring.

### FIND-166 — `BACKTESTER_AUDIT_PLAN.md` table inconsistency (P0-P10 status partially struck-through)

**Status**: CONFIRMED (carry-forward)
**Severity**: LOW
**Reference**: FIND-161 supersession.

### FIND-167 — CLAUDE.md test breakdown per-file counts drift from actual

**Status**: CONFIRMED (Adv-1)
**Severity**: LOW
**Reference**: FIND-103.

### FIND-168 — `lob-backtester/CLAUDE.md` describes Phase 3b ExperimentRunner "YAML config → automated experiment flow" but 4 production YAMLs are functionally broken (FIND-070, FIND-073)

**Status**: CONFIRMED
**Severity**: HIGH (overstated capability)
**Fix-direction**: refresh CLAUDE.md to match actual state.

### FIND-169 — `zero_dte.py` docstring inconsistent: claims `delta * (exit - entry) * 100 * contracts` formula but code uses `(move_bps/10000)` which embeds direction

**Status**: NEW (Adv-1)
**Severity**: LOW (code is right; docstring is wrong)
**Evidence**: `zero_dte.py:9-14` vs line 296 `gross_pnl = delta * (move_bps / 10000.0) * entry_price * 100 * contracts`.

**Fix-direction**: update docstring.

---

## §A Appendices

### Appendix A: Encoded lessons defensive list (verified)

The 17-round empirical backtest history (R1-R17a) has encoded the following lessons as code constraints. **Each lesson must be preserved by any refactor.**

| # | Lesson | Status | File:Line |
|---|---|---|---|
| 1 | `total_trades = len(trades) ≠ len(trade_pnls)` | **ENFORCED** | `types.py:241` |
| 2 | `trade_pnls` includes BOTH entry + exit costs (P2 fix) | **ENFORCED** | `vectorized.py:309,357,401,441` |
| 3 | Short position sizing symmetric with longs (C3 fix) | **ENFORCED** | `vectorized.py:378` |
| 4 | `primary_horizon_idx: int = 0` default H10 (P4 fix) | **ENFORCED** | `regression.py:48` |
| 5 | `min_agreement: float = 0.667` default (P5 fix) | **ENFORCED** | `readability.py:54` |
| 6 | Metric constructors keyword-only (C1 fix) | **ENFORCED** | `metrics/{risk,returns,trading,prediction}.py` |
| 7 | Strategies consume `LabelMapping` (Phase 2a) | **ENFORCED for readability/regression/hybrid/holding; VIOLATED for direction** | various |
| 8 | `OpraCalibratedCosts.deep_itm()` factory | **ENFORCED** | `config.py:206-219` |
| 9 | ATM defaults: half_spread 0.015/0.010, premium 1.88/1.31 | **ENFORCED** | `config.py:168-171` |
| 10 | BSM theta replaces broken 10 bps/min | **ENFORCED** | `zero_dte.py:43-80` |
| 11 | `atomic_write_npy` per threshold for option_trade_pnls (regression only) | **ENFORCED for regression only**, gap for readability (FIND-S6) | `scripts/run_regression_backtest.py:138-140` |
| 12 | `--primary-horizon-idx` partial-assertion gate (Phase V.A.5) | **ENFORCED** | `data/signal_manifest.py` + V.A.5 scripts |
| 13 | Phase 6 6B.5 shim 2026-10-31 calendar removal | **ENFORCED** | `data/signal_manifest.py:39` |
| 14 | TWAP marked SKIP | **DOC-DISAGREES-WITH-CODE** (no skip marker in source; FIND-051) | `strategies/twap.py` |
| 15 | `atomic_write_json` SSoT for registry indexes (#PY-73) | **DOC-DISAGREES-WITH-CODE** (only equity_curve.npy migrated; FIND-090) | `registry.py:69,113,117` |

**Additional encoded lessons identified by Adv-8 (8 NEW)**:

| # | Lesson | Status | Where |
|---|---|---|---|
| 16 | `BacktestResult` invariant chain (returns length, final_equity tolerance, total_trades) | **ENFORCED** | `types.py:225-242` |
| 17 | C-4 strict schema_version validation in DataLoader | **ENFORCED** | `data/loader.py:217-224` |
| 18 | PascalCase metric key contract in registry result.json | **ENFORCED** with lowercase fallback | `registry.py:126-141` |
| 19 | `Position.entry_cost` field required for P2 fix | **ENFORCED** (default=0.0; FIND-007 latent hazard) | `types.py:118` |
| 20 | `_EXCHANGE_PRESETS` module-level SSoT (Phase 6 6A.6) | **ENFORCED** | `config.py:26-39` |
| 21 | `OpraCalibratedCosts.from_dict` defaults match dataclass defaults | **DRIFT HAZARD** (no test pins; defaults duplicated at config.py:408-414) | — |
| 22 | Phase II tamper-detection 3-way fingerprint check | **ENFORCED upstream**; consumer wiring at `experiment.py::_expected_compatibility_fields` | — |
| 23 | `validate=True` vs `validate=False` semantic | **ENFORCED via tests** | `test_legacy_signal_falls_back_to_file_existence` + `test_orphan_calibrated_file_raises_via_validate` |

**Additional encoded lessons added by Cluster D.1+E (5 NEW, 2026-05-14)** — shipped as Commit 1 (FIND-001/002/003) + Commit 2 (FIND-040 + Lesson #14) per `DESIGN_CLUSTER_D1_E_2026_05_14.md` §11. The design memo internally numbers these "#16-#20" but Appendix A renumbers to "#24-#28" to avoid collision with the existing Adv-8 lessons:

| # | Lesson | Status | File:Line |
|---|---|---|---|
| 24 | Atomic state transitions: `trades.append + trade_pnls.append + equity[i] = cash` must be in the same basic block — auto-close path emits `Trade(side=FLAT)` symmetrically with `trade_pnls.append` | **ENFORCED** | `engine/vectorized.py:438-467` (FIND-001 lock; verified by `tests/test_engine/test_vectorized.py::TestEndOfDataAutoClose::test_end_of_data_auto_close_emits_trade`) |
| 25 | `BacktestResult.__post_init__` enforces P2 round-trip pairing invariant: `len(trade_pnls) == sum(1 for t in trades if t.side == TradeSide.FLAT)` | **ENFORCED** | `types.py:243-256` (FIND-002 lock; verified by `tests/test_types.py::TestBacktestResultRoundTripInvariant::test_post_init_pairing_invariant_violated_raises`) |
| 26 | `ZeroDtePnLTransformer` raises `ZeroDteAlternationError` on odd-length trades + per-pair side mismatch (no silent break) | **ENFORCED** | `engine/zero_dte.py:201-209, 270-278, 308-314` (FIND-003 lock; verified by `tests/test_engine/test_zero_dte.py::TestAlternationContract`) |
| 27 | `BacktestStats.daily()` / `.monthly()` raise `NotImplementedError` until `BacktestResult.timestamps_ns` lands; `.full()` is no-op self-return | **ENFORCED** | `stats/stats.py:113-143` (FIND-040 lock; verified by `tests/test_stats/test_stats.py::TestPeriodAggregationStubs`) |
| 28 | `tests/test_strategies/test_twap.py` carries `pytestmark = pytest.mark.skip(reason="...C2 incompatibility...")` at module scope (replaces stale Lesson #14 "DOC-DISAGREES-WITH-CODE") | **ENFORCED** | `tests/test_strategies/test_twap.py:33-40` (Lesson #14 closure lock; verified by `tests/test_strategies/test_twap_skip_discipline.py::TestTwapSkipDiscipline::test_twap_module_has_pytestmark_skip`) |

**Refactor invariants**: any redesign MUST preserve lessons 1–28 (or explicitly justify breaking them with a new pre-registered experimental finding).

---

### Appendix B: Dead-code candidates (refined post-adversarial)

| Item | Confidence | File:Line | Note |
|---|---|---|---|
| `BacktestStats.daily()/monthly()/full()` (period stubs) | CERTAIN | `stats/stats.py:106-134` | FIND-040 |
| `BacktestStats._get_trade_pnls` (trivial wrapper) | HIGH | `stats/stats.py:209-211` | |
| `ComparisonConfig` (entire class) | CERTAIN | `config.py:457-468` | FIND-067 |
| `BacktestConfig.min_confidence`, `min_agreement` | CONFIRMED-CRITICAL (not just dead — FIND-070) | `config.py:312-313` | FIND-056 + FIND-070 |
| `BacktestConfig.stop_loss_pct, take_profit_pct, fill_price, target_holding_minutes` | CERTAIN | `config.py:288-309, 262` | FIND-058 |
| `ZeroDteConfig.entry_window_*_et, target_holding_minutes` | CERTAIN | `config.py:262, 265-266` | FIND-059, FIND-060 |
| `CostConfig.maker_rebate_bps` | HIGH (populated, never used) | `config.py:70` | FIND-064 |
| `OpraCalibratedCosts.atm_call_premium / atm_put_premium / entry_premium` | MEDIUM (informational-only) | `config.py:170-171` | FIND-066 |
| `tests/test_data/`, `tests/test_stats/` empty dirs | CERTAIN | — | FIND-100 |
| `reports/plots.py::plot_positions` (not in `__all__`) | HIGH | `reports/plots.py:269-326` | |
| `scripts/param_sweep.py` (442 LOC) | HIGH | — | Wave-1 +  Adv-5 confirmed |
| `scripts/e5_regime_filter_test.py` (195 LOC) | HIGH | — | Wave-1 + Adv-5 confirmed |
| `scripts/backtest_deeplob.py` (380 LOC) | **REFUTED-WRONG** — production default for hft-ops | `hft-ops/.../manifest/loader.py:351` | NOT a fossil |
| `configs/nvda_readability_first_*.yaml`, `e1_*.yaml` (4 orphaned) | HIGH | `configs/*.yaml` | FIND-073 |
| `Position.unrealized_pnl` field | MEDIUM | `types.py:117` | FIND-008 |
| `BacktestSummary` dataclass | MEDIUM (parallel-impl with dict literal) | `registry.py:26-49` | FIND-096 |
| `data/prices.py NormalizationParams` denormalization path | LOW (T15 deprecated, but defensive) | `data/prices.py:39-122` | |
| `metrics/regression_prediction.py` (entire module, orphan) | HIGH | — | FIND-021 |
| `metrics/prediction.py::ConfusionMetrics` | MEDIUM | `prediction.py:339-432` | FIND-022 |
| `direction.py:16-23 LABEL_*` local re-declaration | CERTAIN (matches values) | — | FIND-042 |
| `Strategy.validate_predictions` ABC method (no subclass invokes) | LOW | `base.py:128-154` | FIND-D6 (Adv-7) |

---

### Appendix C: Open questions requiring user decision before design phase

**Strategic / scoping**:
1. **End-of-data auto-close policy** (FIND-001..004): should engine REQUIRE explicit EXIT (fail-loud) or FABRICATE close (current)? Audit suggests config-driven `auto_close_on_end`.
2. **`BacktestConfig.min_confidence` / `min_agreement` ownership** (FIND-056 + FIND-070): promote to engine-level OR remove from BacktestConfig?
3. **`ExperimentRunner` + `BacktestRegistry` lifecycle** (FIND-094, FIND-095): KEEP BOTH (orthogonal to hft-ops) per Wave-2 verdict.
4. **`OpraCalibratedCosts` migration scope** (FIND-156): scalar constants only (recommended) vs full dataclass upstream?
5. **316 vs 318 IBKR fills SSoT** (FIND-162): pick one canonical number for ANY new doc; footnote both for back-compat.
6. **`@pytest.mark.integration` registration** (FIND-102): trivial fix; do now?
7. **Empty test dirs**: populate or delete (FIND-100)?
8. **`regression_prediction.py` orphan**: delete or wire (FIND-021)?
9. **TWAP fence** (FIND-051): add `pytestmark = pytest.mark.skip` + `NotImplementedError` raise?
10. **`run_readability_backtest.py` per-trade dump asymmetry** (FIND-S6 / encoded-lesson-11 gap): mirror Sub-cycle 4a pattern proactively?
11. **`Signal.EXIT = 2` ↔ `SHIFTED_LABEL_UP = 2` latent collision** (FIND-055): rename `EXIT = 99`?

**Architectural / long-term**:
12. **Multi-symbol portfolio support** (FIND-120): design redesign required; when?
13. **Options-native execution path** (FIND-121): design phase required; what's the bar (real exchange protocol, OPRA fills replay, etc.)?
14. **Streaming inference readiness** (FIND-122): batch-only today; when streaming needed?
15. **Currency handling** (FIND-124): USD-only ok? When to add `currency` field?
16. **Multi-process sweep safety** (FIND-090, FIND-091, FIND-130): file-lock OR document single-process-only?

**Methodological**:
17. **Multi-wave audit cycle cadence**: should this 3-wave methodology be the standard for all major architectural decisions going forward? (Per pipeline precedent in root CLAUDE.md banner history.)
18. **Encoded-lessons defensive list as test discipline**: should each encoded lesson get a "lock test" that fails if the constraint is violated?

---

### Appendix D: Adversarial agent verdict summary

| Wave-1 claim | Wave-2 verdict | Final | Status |
|---|---|---|---|
| C1 engine end-of-data drop | **CONFIRMED + NEW-F2 invariant gap** | CRITICAL active | FIND-001, FIND-002 |
| C2 two DirectionalAccuracy classes | **REFUTED-OVERSTATED → orphan dead code** | LOW | FIND-020, FIND-021 |
| C3 strategy vs model semantics | **CONFIRMED-AS-DESIGN** | LOW (doc gap) | FIND-023 |
| C4 dead BacktestConfig fields | **CONFIRMED + ESCALATED to CRITICAL (N1 silent-drop)** | CRITICAL active | FIND-056, FIND-070 |
| C5 from_dict silent override | **CONFIRMED design-intent, silent override** | MEDIUM | FIND-057 |
| C6 off-exchange rejection | **REFUTED-OVERSTATED — explicit boundary** | LOW | FIND-081 |
| C7 deprecated shim imports | **CONFIRMED-LOW (calendar-tracked)** | LOW | FIND-080 |
| C8 BacktestRegistry collision | **CONFIRMED-MEDIUM (latent)** | MEDIUM | FIND-092 |
| C9 non-atomic writes | **CONFIRMED HIGH (3 sites)** | HIGH | FIND-090 |
| H1 BacktestStats.daily stubs | **CONFIRMED CRITICAL** | CRITICAL | FIND-040 |
| H2 strategies re-emit while in pos | **CONFIRMED-DESIGN-CHOICE** | MEDIUM | FIND-041 |
| H4 silent NaN→0 | **REFUTED-OVERSTATED (latent)** | MEDIUM | FIND-006 |
| H5 Position.entry_cost default | **CONFIRMED-LATENT** | LOW | FIND-007 |
| H6 BacktestData NaN gap | **REFUTED-OVERSTATED (upstream covers)** | MEDIUM | FIND-006 |
| H8 Phase X.3 incomplete | **CONFIRMED** | HIGH | FIND-024 |
| H10 CalmarRatio divergence | **CONFIRMED** | HIGH | FIND-027 |
| H11 parity by coincidence | **REFUTED-WRONG** (locked by frozen-fixture) | N/A | FIND-082 |
| H12 compatibility error swallow | **RECLASSIFIED-AS-FEATURE** | N/A | FIND-083 |
| H13 regression-only expected_fields | **CONFIRMED legitimate** | LOW | FIND-084 |
| H15 sweep limitations | **CONFIRMED** (single-axis intentional; silent-skip + no-policy real) | HIGH | FIND-071, FIND-072 |
| H16 hybrid/twap rejected | **RECLASSIFIED-LOW** | LOW | FIND-074 |
| H17 readability per-trade dump | **REFUTED-OVERSTATED** (YAGNI today, blocker later) | LOW | FIND-S6 |
| M5 ConfusionMetrics ABC | **REFUTED-WRONG** (CompositeMetric pattern) | N/A | FIND-034 |
| M27 sort_keys not pinned | **REFUTED-OVERSTATED** (PyYAML default True) | LOW | FIND-076 |
| U1 BSM theta upstream | **REFUTED-OVERSTATED** | N/A | FIND-155 |
| U2 OpraCalibratedCosts upstream | **REFUTED-WRONG → scope-narrowed** | MEDIUM | FIND-156 |
| O1 retire ExperimentRunner | **REFUTED-OVERSTATED** (Python API legitimate) | N/A | FIND-095 |
| O2 retire BacktestRegistry | **REFUTED-WRONG** (orthogonal) | N/A | FIND-094 |
| backtest_deeplob.py fossil | **REFUTED-WRONG** (production default!) | N/A | FIND-109 |
| S1 316/318 fills | **RECLASSIFIED** (documented dual-recon.) | LOW | FIND-162 |
| S2 test count drift | **CONFIRMED** | MEDIUM | FIND-103 |

**40+ NEW findings** from Adv-1 through Adv-8 — see FIND-002, FIND-003, FIND-016, FIND-021, FIND-022, FIND-032, FIND-035, FIND-061, FIND-062, FIND-068, FIND-070, FIND-072, FIND-085, FIND-086, FIND-091, FIND-093, FIND-096, FIND-110-114 (security), FIND-115-119 (perf), FIND-120-129 (architectural), FIND-130-134 (concurrency), FIND-135-144 (fp), FIND-145-149 (determinism). Plus FIND-054 strategy state leak, FIND-051 TWAP enforcement gap, etc.

---

### Appendix E: Methodology notes

**Why multi-wave is mandatory for large audits**: the pipeline's root CLAUDE.md banner history shows 3-wave validation consistently catches Wave-1 overstatements. This audit confirms the pattern:
- Wave-1 produced 50 findings.
- Wave-2 refuted/reclassified 25+ of them.
- Wave-2 surfaced 40+ NEW findings the original audit missed (especially in security, performance, concurrency, fp precision, reproducibility, and architectural debt — domains the Wave-1 per-module agents weren't tasked to hunt).

**Time budget**: ~17 cumulative agent-hours for full 3-wave cycle on a ~17K LOC repository with 17 backtest rounds of empirical context. This is high upfront cost but pays off in:
- Avoided refactor disasters (e.g., almost-deleted `backtest_deeplob.py` which is production default).
- Surfaced bugs prior audits missed (e.g., FIND-070 silent-misconfig in production YAMLs).
- Hardened defensive list (15 → 23 encoded lessons; 2 lessons revealed as doc-disagrees-with-code).

**Recommended cadence**: re-run a 3-wave validation cycle quarterly OR before any major architectural change (multi-symbol, options-native, streaming).

---

**End of VALIDATION_FINDINGS_2026_05_14.md**

> Next phase: **DESIGNING**. This doc is the technical brief; the designing phase produces fix designs that preserve the 23 encoded lessons + address the 169 findings.
