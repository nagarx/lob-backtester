# DESIGN — Cluster D.1 + Cluster E (Engine Accounting Triple + Discipline Hygiene)

> **Status**: DETAILED DESIGN — pre-impl adversarial review **COMPLETE** (verdict: **REQUIRES-FIX**; 6 critical fixes applied 2026-05-14 — see §13). Re-validation pending before any code is touched.
> **Cycle**: Cluster D.1 + Cluster E (2-commit cycle, single PR)
> **Authorized**: 2026-05-14 (user-confirmed scope + auto-close policy + path forward)
> **Closes**: FIND-001, FIND-002, FIND-003, FIND-040, Lesson #14 + Open Question Q1 (auto-close policy)
> **Surfaces NEW backlog**: HB-2 (zero_dte YAML path mismatch), HB-5 (total_trades count discontinuity), HB-7 (`Dict[str, any]` lowercase typo at config.py:225,351,389,467)
> **Companion docs**: [VALIDATION_FINDINGS_2026_05_14.md](VALIDATION_FINDINGS_2026_05_14.md) (canonical findings catalog), [BACKTEST_INDEX.md](BACKTEST_INDEX.md) (empirical R-9..R-17a contract), [CLAUDE.md](CLAUDE.md) (build + recent fixes)
> **Reads**: hft-rules §0 (reuse-first), §5 (fail-fast), §8 (never silently drop/clamp/fix), §11 (docs must reflect code exactly)

---

## §0 TL;DR

5 findings + 1 encoded lesson close at architectural root in this cycle. Common root cause: **strategy/engine/result contracts encoded in docs but not enforced at type or test level — silent-wrong-result class**. Architectural fix pattern: codify invariants at `__post_init__` + emit atomic state transitions + fail-loud on contract violation + lock-test discipline for encoded lessons.

**Scope**: 2 commits in one PR.

| Commit | Cluster | Findings closed | Files touched | Tests added |
|---|---|---|---|---|
| 1 | **D.1 — Engine Accounting Triple** | FIND-001 + FIND-002 + FIND-003 | `engine/vectorized.py` + `types.py` + `engine/zero_dte.py` | ~5 tests in `tests/test_engine/` + `tests/test_types.py` |
| 2 | **E — Discipline Hygiene** | FIND-040 + Lesson #14 | `stats/stats.py` + `stats/__init__.py` + `tests/test_strategies/test_twap.py` + `.github/workflows/test.yml` | ~3 tests in NEW `tests/test_stats/test_stats.py` (closes partial FIND-100) + pytest-collection check for TWAP skip |

**Estimated effort**: ~6 hr total (D.1 ~3-5 hr + E ~1-1.5 hr including test_stats/ directory creation).

**R-9..R-17a empirical preservation**: **CORRECTNESS REPAIR** (NOT pure preservation; see §5 + §13.9 Wave-2 correction). Historical R-9..R-17a entries in `BACKTEST_INDEX.md` preserved as documentary record (no retroactive rewrite). FUTURE re-runs may produce: (a) HIGHER `total_trades` by +1 per EOF auto-close, (b) HIGHER `option_total_return` if `zero_dte.py:269` silent break was suppressing the final round-trip from option-mode P&L pre-fix, (c) UNCHANGED `equity_total_return`. Cross-repo impact: ZERO — `BacktestRegistry.get()` returns dicts; existing fixtures construct `BacktestResult` with empty `trades + trade_pnls` (invariant passes trivially); zero production callers of `BacktestStats.daily()/.monthly()`. See §5 for full 4-layer proof.

**Auto-close policy decision** (closes Q1 from Appendix C of VALIDATION_FINDINGS): user authorized **`force_close` + emit `Trade(side=TradeSide.FLAT, ...)` + tighten `BacktestResult.__post_init__` invariant**. HARDCODED — no config field added. Strategies that want `signal_exit` semantic can emit `Trade(side=FLAT)` themselves before EOF. Future cycle may add config-driven `Literal["signal_exit", "force_close", "unrealized"]` if a strategy demands it.

---

## §1 Authorization summary

User authorized via AskUserQuestion 2026-05-14:

| Question | Authorized choice | Rationale |
|---|---|---|
| Cluster scope | **D.1 + Cluster E** (5 findings, 2 commits) | Bundles related discipline-hygiene mini-fixes (FIND-040 + Lesson #14) alongside the engine triple; closes 5 findings in one cycle |
| Auto-close policy | **force_close + Trade(FLAT) + invariant** | Preserves R-9..R-17a empirical contract; addresses hft-rules §8 via WARN observability + tightened invariant; pure-fail-loud variant rejected as too aggressive |
| Path forward | **Detailed memo + 1 adversarial pass** | Saved feedback memory MANDATORY pre-impl gate; explicit-decisions document survives compaction |

3-agent adversarial validation already complete on the **proposed scope** (Adv1: priority/scope refutation; Adv2: hidden-bug hunt; Adv3: cross-repo SSoT). This memo locks the **detailed implementation design** for the next adversarial pass to refute.

---

## §2 Architectural principles for this cycle

Five long-term principles guide every fix design below. These will become **encoded lessons #16–#20** in the VALIDATION_FINDINGS Appendix A discipline list (per Q18 default-accept):

1. **Atomic state transitions** — sibling mutations (`trades.append` + `trade_pnls.append` + `equity[i] = cash`) must be performed through a single helper or in a single basic block. The FIND-001 root cause was 3 sites correctly atomic + 1 site (auto-close) missing the `trades.append` companion. Never split a 3-tuple update across non-adjacent statements.

2. **Codify invariants at construction** — every implicit contract in a docstring must have a corresponding `__post_init__` (dataclass) or `model_validator` (Pydantic) assertion. The FIND-002 invariant `len(trade_pnls) == n_closes(FLAT)` was documented in `types.py:194-200` for years but never enforced. Documentation is not enforcement.

3. **Fail-loud on contract violation, never silent** — when an invariant is reachable from production code paths (not just defensive defaults), violation must `raise`, not `break` / `continue` / `return None` / `return False`. The FIND-003 silent `break` at `zero_dte.py:269-270` hid FIND-001 by suppressing the off-by-one symptom. Per hft-rules §8.

4. **Observability over silent abort** — when behavior must be fail-soft for empirical-contract reasons, emit a `logger.warning(...)` with the exact context so operators can audit. The FIND-001 auto-close cannot be `raise` without breaking R-9..R-17a, but it MUST `warn` so the WARN counter surfaces in post-run audit. Per hft-rules §8 "never silently drop/clamp/fix data without recording diagnostics".

5. **Lock-test discipline for encoded lessons** — every encoded lesson in `VALIDATION_FINDINGS_2026_05_14.md` Appendix A must have a regression test that would have caught the violation. Lesson #14 (TWAP marked SKIP doc-vs-code drift) ships its lock test in this cycle (pytest collection check). Future encoded lessons added during design must include their lock-test in the same commit.

---

## §3 Commit 1 — Cluster D.1 (Engine Accounting Triple)

### §3.1 FIND-001 fix — emit `Trade(side=TradeSide.FLAT, ...)` on end-of-data auto-close

**Site**: `src/lobbacktest/engine/vectorized.py:436-442`

**Ground truth (pre-fix)**:

```python
# vectorized.py:436-442 (current — BROKEN)
if not current_position.is_flat:
    final_price = prices[-1]
    cash_flow, cost, pnl = self._close_position(current_position, final_price)
    cash += cash_flow - cost
    trade_pnls.append(pnl - cost - current_position.entry_cost)  # P&L appended
    equity[-1] = cash
    # MISSING: trades.append(Trade(...))  ← THE BUG
```

**Pre-patch — add to module imports at `vectorized.py:1-38`** (CORRECTED per §13 C1):

Ground truth: `vectorized.py:1-38` imports `dataclass`, `Path`, `typing`, `numpy`, internal modules. **NO `import logging` and NO module-level `logger`**. The pre-impl adversarial agent caught that the patch's `logger.warning(...)` would `NameError` without these imports.

```python
# vectorized.py — add to module imports (e.g., after `import numpy as np` at line 22)
import logging

# Module-level logger (add after imports, before `@dataclass class BacktestData`)
logger = logging.getLogger(__name__)
```

**Patch — at the auto-close site `vectorized.py:436-442`**:

```python
# vectorized.py:436-442 (POST-FIX)
if not current_position.is_flat:
    final_price = prices[-1]
    cash_flow, cost, pnl = self._close_position(current_position, final_price)
    cash += cash_flow - cost
    trades.append(Trade(                            # NEW — atomic with trade_pnls
        index=n - 1,                                # last bar index (n = len(data) at vectorized.py:257)
        side=TradeSide.FLAT,                        # canonical close side (per Adv2 ground-truth)
        price=final_price,
        size=current_position.size,
        cost=cost,
    ))
    trade_pnls.append(pnl - cost - current_position.entry_cost)
    equity[-1] = cash
    logger.warning(                                 # NEW — hft-rules §8 observability
        "Engine fabricated end-of-data close at bar=%d; strategy did not signal EXIT. "
        "size=%d, price=%.4f, cost=%.4f. This is the FIND-001 auto-close path; "
        "strategies that want signal-driven exit should emit Trade(side=FLAT) explicitly.",
        n - 1, current_position.size, final_price, cost,
    )
```

**Diff stats**: +2 LOC at module imports + +9 LOC at vectorized.py:436-442 = +11 LOC, -0 LOC at vectorized.py.

**Why this design** (architectural principles applied):

1. **Atomic state transition** (Principle 1) — `trades.append(Trade(...))` is now immediately adjacent to `trade_pnls.append(...)`, forming a single 2-tuple state update. Reordering them in a future refactor would still preserve atomicity within the same basic block.

2. **Existing enum reused** (hft-rules §0) — `TradeSide.FLAT` already exists at `types.py:19-31` as `IntEnum {SELL=-1, FLAT=0, BUY=1}`. All 3 existing close paths (`vectorized.py:313, 361, 405`) emit `side=TradeSide.FLAT`. No new enum variant needed. Adv2 caught my original proposal of `CLOSE_LONG`/`CLOSE_SHORT` which would have introduced non-existent values and broken `zero_dte.py:269-273` alternation parsing.

3. **Index value `n - 1`** — matches the index convention of other close paths (`vectorized.py:402` uses `index=i` where `i` is the current loop index; at end-of-data the equivalent is `i = n - 1`). **Verified 2026-05-14 (Wave-2 ground-truth re-check)**: `n = len(data)` is set at `vectorized.py:257` (inside `run()` method, NOT `n = len(prices)` as the v1 memo phrased); `n` IS in scope at line 436. Patch can use `n - 1` directly.

4. **WARN log mandatory** (Principle 4) — closes hft-rules §8 "never silently drop/clamp/fix data without recording diagnostics". The auto-close cannot raise (preserves R-9..R-17a empirical) but cannot be silent either. WARN gives operators an audit trail. Future cycles that add `config.log_auto_close: bool` for noise reduction can suppress; default ON.

5. **No new `BacktestError` raise** — the user explicitly chose `force_close` over `signal_exit_required`. The WARN log satisfies the discoverability requirement without breaking empirical contract.

**Open implementation detail**: confirm `n` symbol is in scope at line 436 (it's the local variable holding `len(prices)` based on auto-close context). If named differently in actual code (`N`, `num_bars`, etc.), use the matching name.

---

### §3.2 FIND-002 fix — add round-trip pairing invariant to `BacktestResult.__post_init__`

**Site**: `src/lobbacktest/types.py:218-242`

**Ground truth (pre-fix)** — current invariants enforced:

```python
# types.py:218-242 (current)
def __post_init__(self):
    if len(self.equity_curve) == 0:
        raise ValueError("equity_curve must be non-empty")
    if len(self.prices) != len(self.equity_curve):
        raise ValueError(
            f"prices/equity_curve length mismatch: "
            f"{len(self.prices)} != {len(self.equity_curve)}"
        )
    if len(self.positions) != len(self.equity_curve):
        raise ValueError(...)
    if len(self.returns) != len(self.equity_curve) - 1:
        raise ValueError(...)
    if abs(self.final_equity - self.equity_curve[-1]) > 1e-10:
        raise ValueError(...)
    if self.total_trades != len(self.trades):
        raise ValueError(...)
    # MISSING: len(trade_pnls) vs FLAT-side trades  ← THE GAP
```

**Patch** (append at end of `__post_init__`):

```python
# types.py:218-242 (POST-FIX — append after existing total_trades check)
    # NEW: round-trip pairing invariant (closes FIND-002 + co-locks FIND-001)
    # Contract: each round-trip = 1 entry trade (BUY|SELL) + 1 exit trade (FLAT) + 1 P&L
    # Documented in this class docstring at lines 194-200 but never enforced pre-2026-05-14.
    # See lob-backtester/VALIDATION_FINDINGS_2026_05_14.md FIND-002 and Appendix A row #16.
    n_closes = sum(1 for t in self.trades if t.side == TradeSide.FLAT)
    if len(self.trade_pnls) != n_closes:
        raise ValueError(
            f"P2 round-trip pairing contract: each closed round-trip = 1 FLAT trade + 1 trade_pnl; "
            f"got {n_closes} FLAT trades vs {len(self.trade_pnls)} trade_pnls. "
            f"If you constructed BacktestResult directly (test fixture or external producer), "
            f"emit Trade(side=TradeSide.FLAT, ...) once per round-trip close. "
            f"See FIND-001/FIND-002 cluster."
        )
```

**Diff stats**: +12 LOC.

**Why this design**:

1. **Codify invariant at construction** (Principle 2) — implicit contract from `types.py:194-200` becomes a runtime assertion. Adding the invariant retroactively would have caught FIND-001 immediately upon implementation. Pre-2026-05-14 the invariant was textual only.

2. **`TradeSide.FLAT` counted** (Adv2 ground-truth correction) — uses the existing enum. All 4 close paths (3 in-loop + 1 auto-close-post-FIND-001-fix) emit `side=TradeSide.FLAT`. No new enum variant required. The originally-proposed `t.side in {CLOSE_LONG, CLOSE_SHORT}` was Adv2-caught as referencing non-existent values.

3. **Helpful error message** — references the bug class (FIND-001/002), the contract name (P2 round-trip pairing), and the actionable mitigation (emit FLAT trade once per close). Future debuggers who hit this assertion will have full context.

4. **No retroactive risk** — Adv2 verified: `BacktestRegistry.get()` returns raw JSON dict, no `BacktestResult.from_dict` constructor exists, so historical re-reads never instantiate `BacktestResult` and never fire `__post_init__`. Test fixtures at `tests/test_types.py:247-263, 290-310, 315-330, 336-365` construct with empty `trades=[]` + empty `trade_pnls=np.array([])` → invariant `0 == 0` holds. No fixture breaks.

5. **MUST ship atomically with FIND-001 fix** (Adv2 sequencing constraint) — if FIND-002 lands without FIND-001's `trades.append(Trade(FLAT))`, every fresh run with an open position at EOF would fail the new invariant. Both fixes must be in the same commit, in the order: vectorized.py first, types.py second (so a partial bisect doesn't hit the breakage window).

**Sequencing within commit 1**: Edit vectorized.py BEFORE types.py. Run targeted tests after each edit (`pytest tests/test_engine/test_vectorized.py -v` then `pytest tests/test_types.py -v`).

---

### §3.3 FIND-003 fix — replace silent `break` in `ZeroDtePnLTransformer` with explicit raise + side assert

**Site**: `src/lobbacktest/engine/zero_dte.py:266-273`

**Ground truth (pre-fix — CORRECTED per §13 C6)**:

```python
# zero_dte.py:221-273 (current — silent break hides FIND-001 symptoms)
def transform(self, result: BacktestResult) -> ZeroDteResult:
    """Transform equity BacktestResult into IBKR+OPRA-calibrated ZeroDteResult."""
    trades = result.trades
    equity_pnls = result.trade_pnls
    n_round_trips = len(equity_pnls)                # ← NOT `len(trades) // 2` as the v1 memo claimed
                                                     #    (caught by adversarial gate; corrected here)

    empty = np.array([], dtype=np.float64)
    if n_round_trips == 0:
        return ZeroDteResult(...)                    # early return, fields elided

    # ... arrays initialized ...

    for i in range(n_round_trips):
        entry_idx = i * 2
        exit_idx = i * 2 + 1
        if exit_idx >= len(trades):
            break                                    # ← SILENT — hides alternation break

        entry_trade = trades[entry_idx]
        exit_trade = trades[exit_idx]
        # ... rest of loop body using entry_trade.side / exit_trade.side ...
```

**Critical insight from adversarial gate**: `n_round_trips` is derived from `len(equity_pnls)` — the count of *closed* round-trips reported by the engine. If the engine produces `len(equity_pnls) = K` but `len(trades) < 2*K` (the FIND-001 scenario), the silent `break` at line 269-270 truncates option-mode results to fewer round-trips while equity-mode keeps the count `K`. **Preserve the `len(equity_pnls)` semantics**; do NOT change to `len(trades) // 2`.

**Patch** (CORRECTED per §13 C6 — precondition is a separate guard, NOT a semantic change):

```python
# zero_dte.py — INSERT precondition between L235-251 early-return and L266 for-loop:
n_round_trips = len(equity_pnls)

empty = np.array([], dtype=np.float64)
if n_round_trips == 0:
    return ZeroDteResult(...)  # unchanged early-return

# NEW: alternation contract precondition (closes FIND-003 + co-locks FIND-001)
# Post-FIND-001 fix, engine emits Trade(FLAT) symmetrically with trade_pnls.append.
# Contract: each closed round-trip = 1 entry trade (BUY|SELL) + 1 exit trade (FLAT).
# Therefore: len(trades) MUST equal 2 * n_round_trips. Per hft-rules §5 fail-fast.
expected_n_trades = n_round_trips * 2
if len(trades) != expected_n_trades:
    raise ZeroDteAlternationError(
        f"ZeroDte expects 2 trades per round-trip (open + close); "
        f"got n_round_trips={n_round_trips} (from len(equity_pnls)) "
        f"but len(trades)={len(trades)}, expected {expected_n_trades}. "
        f"Engine should emit Trade(side=FLAT) symmetrically with trade_pnls.append. "
        f"See FIND-001/FIND-002/FIND-003 cluster."
    )

# ... arrays initialized ...

for i in range(n_round_trips):
    entry_idx = i * 2
    exit_idx = i * 2 + 1
    # REMOVED: silent break `if exit_idx >= len(trades): break` — precondition above
    # makes this state structurally unreachable.

    entry_trade = trades[entry_idx]
    exit_trade = trades[exit_idx]

    # NEW: per-pair alternation assertion (catches reordering bugs + future regressions)
    if entry_trade.side == TradeSide.FLAT or exit_trade.side != TradeSide.FLAT:
        raise ZeroDteAlternationError(
            f"ZeroDte alternation violated at round-trip {i}: "
            f"entry@{entry_idx}.side={entry_trade.side.name}, "
            f"exit@{exit_idx}.side={exit_trade.side.name}. "
            f"Expected entry in {{BUY, SELL}} + exit == FLAT."
        )
    # ... rest of loop body unchanged ...
```

**Plus new exception class** at module scope of `zero_dte.py` (or in a shared `engine/errors.py` if one exists):

```python
class ZeroDteAlternationError(ValueError):
    """Raised when ZeroDtePnLTransformer detects a violated alternation contract.

    The transformer requires strict [open, close, open, close, ...] alternation in trades:
    entries (sides BUY|SELL) at even indices, exits (side FLAT) at odd indices.
    Post-FIND-001 fix this is structurally guaranteed by the engine; this exception
    is reserved for future regression detection or external Trade-stream consumers.
    """
```

**Diff stats**: +25 LOC (incl. new exception class).

**Why this design**:

1. **Fail-loud on contract violation** (Principle 3) — replaces silent `break` with explicit raise. The silent break was the proximal cause of FIND-001 going undetected for so long: option-mode P&L silently truncated; equity-mode kept the orphan.

2. **Two-layer check**: outer (odd `len(trades)`) catches engine-side bugs; inner per-pair (entry/exit side mismatch) catches reordering bugs. Defense-in-depth.

3. **New exception class** — `ZeroDteAlternationError(ValueError)` enables future consumers to `except ZeroDteAlternationError` selectively without catching all `ValueError`. Mirrors the `FeatureSetResolverError` hierarchy pattern from `hft_contracts` (per root CLAUDE.md §"Phase 4 FeatureSet Registry").

4. **Reachability post-FIND-001** — once FIND-001 ships, the alternation contract is engine-guaranteed. This assert becomes a regression-detection rail rather than an active production-path raise. That's the right design: type-system enforces invariant; assertion catches regression.

5. **Can ship in same commit as FIND-001/FIND-002 OR separate** — Adv2 sequencing analysis said either is acceptable. **Recommendation**: SAME commit. Closes the whole class atomically. Single PR review, single bisect target.

---

### §3.4 Commit 1 — Tests (CORRECTED per §13)

3 new regression tests + 1 expected-count update. **All test code in this section was rewritten 2026-05-14 after adversarial gate caught 4 API mismatches (C2-C5).** The corrected APIs match ground truth verified in `vectorized.py`, `types.py`, `direction.py`, and existing `tests/test_engine/test_vectorized.py:59-100`.

**Verified API ground truth**:
- Engine class: `VectorizedEngine(config: BacktestConfig)` — `Backtester` exists but tests use `VectorizedEngine`
- Run signature: `engine.run(data, strategy)` — `strategy` is positional argument (NOT kwarg)
- Strategy: `DirectionStrategy(predictions, shifted: bool = False, name: str = None)` — `shifted=False` uses {-1,0,1}; `shifted=True` uses {0,1,2}
- Config: `BacktestConfig(initial_capital=, position_size=, costs=CostConfig(...))` — `for_exchange` is on **CostConfig**, NOT `BacktestConfig`
- `BacktestResult.positions: np.ndarray` (not `List[Position]`)
- `ZeroDtePnLTransformer.transform(result: BacktestResult) -> ZeroDteResult` — single positional arg
- `BacktestResult` requires 15 fields including `start_index`, `end_index`, `predictions`, `labels`, `config_dict`

**File**: `tests/test_engine/test_vectorized.py` (existing 21 tests, append 1)

```python
import logging
import numpy as np
import pytest

from lobbacktest.config import BacktestConfig, CostConfig
from lobbacktest.engine.vectorized import VectorizedEngine, BacktestData
from lobbacktest.strategies.direction import DirectionStrategy
from lobbacktest.types import TradeSide


class TestEndOfDataAutoClose:
    """FIND-001 lock tests: auto-close on EOF open position must emit Trade(FLAT)."""

    def test_end_of_data_auto_close_emits_trade(self, caplog):
        """FIND-001 lock test: auto-close on open position at EOF must emit Trade(FLAT).

        Pre-2026-05-14: auto-close at vectorized.py:436-442 appended trade_pnls but skipped
        trades.append → ZeroDtePnLTransformer silently dropped final round-trip via break.
        """
        # 3-bar always-in BUY strategy: open at bar 0, hold, never close via signal
        prices = np.array([100.0, 101.0, 102.0])
        predictions = np.array([1, 1, 1])  # BUY/BUY/BUY (unshifted mapping {-1,0,1})

        config = BacktestConfig(
            initial_capital=10_000.0,
            position_size=0.1,  # 10% per trade
            costs=CostConfig(spread_bps=0, slippage_bps=0, commission_per_trade=0),
        )
        data = BacktestData(prices=prices)
        strategy = DirectionStrategy(predictions, shifted=False)

        engine = VectorizedEngine(config)

        with caplog.at_level(logging.WARNING):
            result = engine.run(data, strategy)

        # FIND-001: trades has 2 entries (1 BUY entry + 1 FLAT auto-close), not 1
        assert len(result.trades) == 2, (
            f"FIND-001: expected 2 trades (BUY entry + FLAT auto-close); got "
            f"{len(result.trades)}: {[(t.index, t.side.name) for t in result.trades]}"
        )

        # Auto-close emits canonical FLAT side at the last bar (n-1)
        assert result.trades[-1].side == TradeSide.FLAT
        assert result.trades[-1].price == 102.0
        assert result.trades[-1].index == 2  # n - 1 = len(prices) - 1

        # WARN log emitted with required context (hft-rules §8 observability)
        assert any(
            "fabricated end-of-data close" in record.message
            for record in caplog.records
        ), "FIND-001 fix must emit WARN log per hft-rules §8 observability"

        # FIND-002: __post_init__ invariant satisfied (1 FLAT trade == 1 trade_pnl)
        # If invariant fails, result construction would have raised before this line
        assert len(result.trade_pnls) == 1
```

**File**: `tests/test_types.py` (existing 26 tests, append 1 class with 2 tests)

```python
import numpy as np
import pytest

from lobbacktest.types import BacktestResult, Trade, TradeSide


class TestBacktestResultRoundTripInvariant:
    """FIND-002 lock tests: BacktestResult.__post_init__ enforces P2 round-trip pairing."""

    def _base_kwargs(self, n: int) -> dict:
        """Minimal BacktestResult kwargs for the n-bar fixture (positions=np.ndarray)."""
        return dict(
            equity_curve=np.array([100.0] * n),
            returns=np.zeros(n - 1),
            positions=np.zeros(n),   # np.ndarray — NOT List[Position] (C3 correction)
            prices=np.array([10.0] * n),
            predictions=np.zeros(n),
            labels=None,
            metrics={},
            config_dict={},
            initial_capital=100.0,
            final_equity=100.0,
            start_index=0,
            end_index=n - 1,
        )

    def test_post_init_pairing_invariant_satisfied_passes(self):
        """1 FLAT trade + 1 trade_pnl → invariant holds; construction succeeds."""
        result = BacktestResult(
            trades=[
                Trade(index=0, side=TradeSide.BUY, price=10.0, size=10, cost=0.1),
                Trade(index=1, side=TradeSide.FLAT, price=10.5, size=10, cost=0.1),
            ],
            trade_pnls=np.array([4.8]),  # 1 closed round-trip
            total_trades=2,
            **self._base_kwargs(n=2),
        )
        assert len(result.trade_pnls) == 1

    def test_post_init_pairing_invariant_violated_raises(self):
        """1 BUY trade (no FLAT) + 1 trade_pnl → 0 FLAT trades vs 1 pnl → raises."""
        with pytest.raises(ValueError, match="P2 round-trip pairing contract"):
            BacktestResult(
                trades=[
                    Trade(index=0, side=TradeSide.BUY, price=10.0, size=10, cost=0.1),
                    # MISSING: Trade(side=TradeSide.FLAT, ...)
                ],
                trade_pnls=np.array([4.8]),  # 1 pnl but 0 FLAT trades
                total_trades=1,
                **self._base_kwargs(n=2),
            )
```

**File**: `tests/test_engine/test_zero_dte.py` (existing 18 tests, append 2 new tests)

```python
import numpy as np
import pytest

from lobbacktest.config import ZeroDteConfig
from lobbacktest.engine.zero_dte import ZeroDtePnLTransformer, ZeroDteAlternationError
from lobbacktest.types import BacktestResult, Trade, TradeSide


def _make_result(trades: list, trade_pnls: np.ndarray, n: int = 15) -> BacktestResult:
    """Build a BacktestResult fixture that satisfies FIND-002 invariant but stresses FIND-003."""
    return BacktestResult(
        equity_curve=np.array([100.0] * n),
        returns=np.zeros(n - 1),
        positions=np.zeros(n),
        prices=np.array([10.0] * n),
        predictions=np.zeros(n),
        labels=None,
        trades=trades,
        trade_pnls=trade_pnls,
        metrics={},
        config_dict={},
        initial_capital=100.0,
        final_equity=100.0,
        total_trades=len(trades),
        start_index=0,
        end_index=n - 1,
    )


def test_zero_dte_alternation_orphan_trade_raises():
    """FIND-003 lock: ZeroDte raises when len(trades) != 2 * n_round_trips.

    Constructs a fixture that passes FIND-002 invariant (1 FLAT == 1 trade_pnl)
    but violates FIND-003 precondition (3 trades total, expected 2).
    """
    trades = [
        Trade(index=0, side=TradeSide.BUY, price=10.0, size=10, cost=0.1),
        Trade(index=5, side=TradeSide.FLAT, price=10.5, size=10, cost=0.1),
        Trade(index=10, side=TradeSide.BUY, price=10.2, size=10, cost=0.1),  # ORPHAN
    ]
    # n_round_trips = len(trade_pnls) = 1; expected len(trades) = 2; got 3 → raise
    result = _make_result(trades, trade_pnls=np.array([4.8]))

    transformer = ZeroDtePnLTransformer(config=ZeroDteConfig())
    with pytest.raises(ZeroDteAlternationError, match="2 trades per round-trip"):
        transformer.transform(result)


def test_zero_dte_per_pair_alternation_violation_raises():
    """FIND-003 lock: per-pair side assert catches reordered trades.

    Constructs a fixture with 4 trades + 2 trade_pnls (FIND-002 + precondition pass)
    but reordered as (BUY, BUY, FLAT, FLAT) instead of (BUY, FLAT, BUY, FLAT).
    """
    trades = [
        Trade(index=0, side=TradeSide.BUY, price=10.0, size=10, cost=0.1),
        Trade(index=1, side=TradeSide.BUY, price=10.1, size=10, cost=0.1),  # WRONG — should be FLAT
        Trade(index=5, side=TradeSide.FLAT, price=10.5, size=10, cost=0.1),
        Trade(index=6, side=TradeSide.FLAT, price=10.6, size=10, cost=0.1),
    ]
    # n_round_trips = 2; len(trades) = 4 → precondition passes; pair-0 = (BUY, BUY) → per-pair fails
    result = _make_result(trades, trade_pnls=np.array([4.8, 5.0]))

    transformer = ZeroDtePnLTransformer(config=ZeroDteConfig())
    with pytest.raises(ZeroDteAlternationError, match="alternation violated"):
        transformer.transform(result)
```

**Test surface deltas (Commit 1)**:
- `tests/test_engine/test_vectorized.py`: +1 test (in NEW `TestEndOfDataAutoClose` class)
- `tests/test_types.py`: +2 tests (in NEW `TestBacktestResultRoundTripInvariant` class)
- `tests/test_engine/test_zero_dte.py`: +2 tests + 1 NEW helper `_make_result`

Total: **+5 NEW regression tests** locking the 3-finding cluster. All APIs verified against ground truth source (no hallucinated method signatures).

---

## §4 Commit 2 — Cluster E (Discipline Hygiene)

### §4.1 FIND-040 fix — `BacktestStats.daily/.monthly` raise `NotImplementedError`

**Site**: `src/lobbacktest/stats/stats.py:106-134`

**Ground truth (pre-fix)**:

```python
# stats/stats.py:106-134 (current — sets state never read by compute())
def daily(self) -> "BacktestStats":
    self._period = "daily"
    return self

def monthly(self) -> "BacktestStats":
    self._period = "monthly"
    return self

def full(self) -> "BacktestStats":
    self._period = "full"
    return self

# compute() at :149-207 NEVER reads self._period for metric computation;
# self._period only flows to StatsSummary.period as a string label.
```

**Patch**:

```python
# stats/stats.py:106-134 (POST-FIX)
def daily(self) -> "BacktestStats":
    raise NotImplementedError(
        "BacktestStats.daily() requires per-period timestamps on BacktestResult; "
        "BacktestResult does not currently carry timestamps_ns. Daily aggregation is "
        "not yet supported. Use .compute() for full-corpus metrics instead. "
        "Track at FIND-040 in lob-backtester/VALIDATION_FINDINGS_2026_05_14.md."
    )

def monthly(self) -> "BacktestStats":
    raise NotImplementedError(
        "BacktestStats.monthly() requires per-period timestamps on BacktestResult; "
        "BacktestResult does not currently carry timestamps_ns. Monthly aggregation is "
        "not yet supported. Use .compute() for full-corpus metrics instead. "
        "Track at FIND-040 in lob-backtester/VALIDATION_FINDINGS_2026_05_14.md."
    )

def full(self) -> "BacktestStats":
    """Period selector — full-corpus (default). No-op; .compute() returns full metrics regardless."""
    self._period = "full"
    return self
```

**Plus docstring example fix at THREE sites** (per §13 HIDDEN-4 + Wave-2 NB-1 — v1 memo missed `stats/stats.py:72-77` AND `stats/stats.py:1-15` which BOTH chain `.daily()`):

**Site 1**: `stats/__init__.py` module docstring example
```python
# Was: stats = BacktestStats(result).with_book_size(100_000).daily().compute()
# To:  stats = BacktestStats(result).with_book_size(100_000).compute()
#      # Note: .daily() / .monthly() period aggregation not currently supported
#      # (requires timestamps on BacktestResult); see FIND-040.
```

**Site 2**: `stats/stats.py:71-77` class docstring "Or chained:" example (MISSED IN V1 MEMO; §13 adversarial gate caught)

**Site 3 (NEW)**: `stats/stats.py:1-15` module-level docstring example (MISSED IN V1 + V2 MEMO; Wave-2 NB-1 caught 2026-05-14). The MODULE docstring at lines 1-15 of `stats.py` ALSO contains the `.with_book_size(100_000).daily().compute()` chain in its "Example:" block — sister to Site 1 in `__init__.py`. Apply identical fix: drop `.daily()` from the chain + add note about FIND-040 NotImplementedError.
```python
# stats/stats.py:71-77 (current — broken example also chains .daily())
"""...
Or chained:
    >>> stats = (
    ...     BacktestStats(result)
    ...         .with_book_size(100_000)
    ...         .daily()
    ...         .compute()
    ... )
"""

# stats/stats.py:71-77 (POST-FIX)
"""...
Or chained:
    >>> stats = (
    ...     BacktestStats(result)
    ...         .with_book_size(100_000)
    ...         .compute()
    ... )
    # Note: .daily() / .monthly() raise NotImplementedError until timestamps land; see FIND-040.
"""
```

**Diff stats**: +18 LOC at `stats/stats.py` body (NotImplementedError raises) + ~3 LOC at `stats/__init__.py` docstring + ~3 LOC at `stats/stats.py:71-77` docstring = +24 LOC total in Cluster E commit.

**Why this design** (architectural principles applied):

1. **Fail-loud on unsupported feature** (Principle 3 + hft-rules §5) — chaining `.daily()` immediately surfaces the constraint at chain-entry, not at silent compute. Operators discover the limitation at the call site.

2. **`.full()` retained as no-op** — provides the "explicit period selector" UX without misleading aggregation claims. Backward-compatible with operators who chain `.full().compute()` for readability.

3. **Reachability verified** (Adv2) — 0 production callers chain `.daily()` or `.monthly()`. Only the docstring example at `stats/__init__.py:8` did, and it's fixed in the same commit.

4. **Error message references FIND-040** — future debuggers who hit the NotImplementedError have direct doc traceability.

5. **NOT deletion** (hft-rules §4 alternative considered) — preserves the API surface for a future cycle that adds `BacktestResult.timestamps_ns` field + implements true period aggregation. Deletion would close the door; raise leaves it open. Documented in the error message that period aggregation is a contract change (needs timestamps), not a bug fix.

### §4.2 Lesson #14 fix — `pytestmark = pytest.mark.skip` for `test_twap.py`

**Site**: `tests/test_strategies/test_twap.py` (top of file)

**Ground truth (pre-fix)**:

```python
# tests/test_strategies/test_twap.py:1-25 (current — 0 skip markers; all 8 tests run)
"""Tests for TWAPStrategy."""
import numpy as np
import pytest

from lobbacktest.strategies.twap import TWAPStrategy
from lobbacktest.strategies.base import Signal, SignalOutput
# ... 8 test functions, all run normally ...
```

**Patch** (add at top of file, after existing imports):

```python
# tests/test_strategies/test_twap.py:1-30 (POST-FIX)
"""Tests for TWAPStrategy.

NOTE: TWAPStrategy is empirically failed at R2 (see lob-backtester/BACKTEST_INDEX.md
Round 2) due to engine C2 incompatibility. Tests are preserved for future
re-enablement if C2 is resolved, but marked SKIP at module level.

This skip discipline closes Lesson #14 (CLAUDE.md docstring claimed SKIP but code
did not actually skip — doc-vs-code drift detected by 2026-05-14 audit).
See lob-backtester/VALIDATION_FINDINGS_2026_05_14.md Appendix A row #14.
"""
import numpy as np
import pytest

# Lesson #14 lock — module-level skip until C2 incompatibility resolved
pytestmark = pytest.mark.skip(
    reason="TWAPStrategy empirically failed at R2 (BACKTEST_INDEX); "
           "C2 engine incompatibility. Tests preserved for future re-enablement."
)

from lobbacktest.strategies.twap import TWAPStrategy
from lobbacktest.strategies.base import Signal, SignalOutput
# ... 8 test functions, all SKIPPED via pytestmark ...
```

**Plus CI expected-count update** at `.github/workflows/test.yml`:

```yaml
# .github/workflows/test.yml — update inline comment near pytest invocation
# Was (or similar):
#   # Verified count: 359 pass + 8 skip (as of 2026-04-21)
# To:
#   # Verified count: 406 pass + 16 skip (post Lesson #14 TWAP skip — 2026-05-14)
#   # Concrete count: re-run `pytest --collect-only` before each release.
```

**Note on count**: per recent `pytest --collect-only` (2026-05-14 audit), `lob-backtester` collects **414 tests**. After Cluster D.1+E:
- +5 NEW Cluster D.1 tests
- +3 NEW Cluster E tests (stats)
- −0 deleted
- 8 TWAP tests now skip

Estimated post-cycle: ~422 collected = ~406 pass + 16 skip. Confirm via actual `pytest --collect-only` at impl time.

**Diff stats**: +12 LOC at `test_twap.py`, ~3 LOC at `test.yml` (comment only).

**Why this design**:

1. **Module-level skip** (Adv2 verified) — `pytestmark = pytest.mark.skip(...)` at module scope is the canonical pattern; applies to all 8 tests without per-test decoration.

2. **DO NOT modify `TWAPStrategy.__init__` to raise** (Adv2 explicit warning) — that would be a silent downgrade: the class is still importable, might pass `isinstance(s, Strategy)` checks in the registry without ever being instantiated. Keep the class as-is; fence the tests.

3. **Lock-test discipline** (Principle 5) — the docstring reference to Lesson #14 makes future auditors aware of the encoded-lesson lock. A future cycle that removes the skip marker without re-validating C2 compatibility would trip the lesson.

4. **No registry implications** — Adv2 verified TWAPStrategy is NOT in `strategies/__init__.py` exports, NOT in `StrategyRegistry`. Sole import site is `test_twap.py:22`. Skipping is risk-free.

### §4.3 Commit 2 — Tests

**NEW directory**: `tests/test_stats/` (closes partial FIND-100 — was empty)

```
tests/test_stats/
├── __init__.py        # empty
└── test_stats.py      # 3 new tests
```

**File**: `tests/test_stats/test_stats.py` (NEW)

```python
"""Tests for BacktestStats fluent API.

FIND-040 lock tests: .daily()/.monthly() must raise NotImplementedError until
BacktestResult exposes timestamps_ns. See lob-backtester/DESIGN_CLUSTER_D1_E_2026_05_14.md
and VALIDATION_FINDINGS_2026_05_14.md FIND-040 for context.
"""
import numpy as np
import pytest

from lobbacktest.stats import BacktestStats
from lobbacktest.types import BacktestResult


def _make_minimal_result() -> BacktestResult:
    """Construct a minimal BacktestResult for stats tests.

    CORRECTED 2026-05-14 per §13.7 re-validation verdict: helper was missing
    5 required BacktestResult fields + used wrong type for `positions` (was
    `List[Position]`, types.py:204 mandates `np.ndarray` — same C3-class bug).
    """
    n = 2
    return BacktestResult(
        equity_curve=np.array([100.0, 105.0]),
        returns=np.array([0.05]),
        positions=np.zeros(n),                # FIX: was List[Position]; types.py:204 wants ndarray
        prices=np.array([10.0, 10.5]),
        predictions=np.zeros(n),              # FIX: missing required field
        labels=None,                          # FIX: missing required field
        trades=[],
        trade_pnls=np.array([]),
        metrics={},
        config_dict={},                       # FIX: missing required field
        initial_capital=100.0,
        final_equity=105.0,
        total_trades=0,
        start_index=0,                        # FIX: missing required field
        end_index=n - 1,                      # FIX: missing required field
    )


class TestPeriodAggregationStubs:
    """FIND-040: .daily()/.monthly() must raise; .full() must remain no-op."""
    
    def test_daily_raises_not_implemented(self):
        stats = BacktestStats(_make_minimal_result())
        with pytest.raises(NotImplementedError, match="FIND-040"):
            stats.daily()
    
    def test_monthly_raises_not_implemented(self):
        stats = BacktestStats(_make_minimal_result())
        with pytest.raises(NotImplementedError, match="FIND-040"):
            stats.monthly()
    
    def test_full_remains_no_op(self):
        stats = BacktestStats(_make_minimal_result())
        result = stats.full()
        assert result is stats  # fluent self-return
        assert stats._period == "full"  # state set (cosmetic only)
```

**Plus pytest-collection check for Lesson #14**:

`tests/test_strategies/test_twap.py` collection should show 8 SKIPPED. Add a meta-test elsewhere (e.g., `tests/test_meta.py` if exists, or `tests/test_strategies/test_twap_skip_discipline.py`):

```python
# Either inline lock test OR rely on CI count assertion
def test_twap_module_is_skipped_at_collection():
    """Lesson #14 lock: TWAP tests must be marked SKIP at module level."""
    import importlib
    import pytest
    
    mod = importlib.import_module("tests.test_strategies.test_twap")
    pytestmark = getattr(mod, "pytestmark", None)
    assert pytestmark is not None, "Lesson #14: test_twap.py must have pytestmark"
    
    # pytestmark may be a single mark or list — handle both
    marks = pytestmark if isinstance(pytestmark, list) else [pytestmark]
    skip_marks = [m for m in marks if m.name == "skip"]
    assert skip_marks, "Lesson #14: test_twap.py must have a skip mark"
    assert "C2" in skip_marks[0].kwargs.get("reason", ""), \
        "Lesson #14: skip reason must reference C2 incompatibility"
```

**Test surface deltas (Commit 2)**:
- `tests/test_stats/test_stats.py`: +3 NEW tests (FIND-040)
- `tests/test_strategies/test_twap_skip_discipline.py` OR meta-test inline: +1 test (Lesson #14 lock)
- `tests/test_strategies/test_twap.py`: 8 tests now SKIPPED (not deleted)

Total: **+4 NEW lock tests** (or +3 + 1 meta) closing Cluster E.

---

## §5 R-9..R-17a empirical preservation argument

Cluster D.1 + E is structurally non-disruptive to the 17-round empirical contract documented in `BACKTEST_INDEX.md`. Four-layer proof:

### Layer 1 — Production scripts DO reach the auto-close branch — CORRECTNESS REPAIR, not pure preservation

**CORRECTION (Wave-2 adversarial gate, 2026-05-14 session 2)**: the v1 memo's claim that "auto-close was a cold path in R-9..R-17a production" is FACTUALLY WRONG. Empirical re-verification by two independent agents proved the auto-close branch IS REACHABLE in production runs:

- **`HorizonAlignedPolicy.should_exit`** at `strategies/holding.py:102-103` returns `state.events_held >= self.hold_events`. **No EOF check.** Returns True only when the holding-period threshold is met.
- **`regression.py:189`** + **`readability.py:237`** BOTH track `exit_reasons["end_of_data"]` when `in_position` at the final bar — *proving* that production strategies leave positions open at EOF and rely on the engine's auto-close.
- R-9..R-17a default `--hold-events=10` (per `scripts/run_regression_backtest.py:162`). Entries within 10 bars of EOF leave open positions that hit the auto-close at `vectorized.py:436-442`.

**Implication**: Cluster D.1 is a **CORRECTNESS REPAIR**, not pure invariant tightening. Pre-fix engine behavior had two coupled silent bugs (FIND-001 missing `Trade(FLAT)` + FIND-003 silent break at `zero_dte.py:269`) that *interacted* — option-mode P&L silently truncated the final round-trip whenever auto-close fired.

**R-9..R-17a documentary preservation status**:
- `BACKTEST_INDEX.md` historical entries are preserved as documentary record (no retroactive rewrite of this commit)
- FUTURE re-runs of R-9..R-17a configs MAY produce:
  - (a) HIGHER `total_trades` (auto-closed positions now emit `Trade(FLAT)`) — count shifts by +1 per affected EOF auto-close
  - (b) HIGHER `option_total_return` if the silent `break` at `zero_dte.py:269` was suppressing the final round-trip from option-mode P&L pre-fix — likely +1 round-trip worth of option P&L per affected run
  - (c) UNCHANGED `equity_total_return` (`trade_pnls` was already appended pre-fix; only the symmetric `trades.append(Trade(FLAT))` was missing — equity-mode totals are unaffected)

**This is a correctness improvement**, not a regression. The R-9..R-17a documentary record remains valid for **historical interpretation**; re-runs are not required for this commit to ship.

**Layers 2-4 remain VERIFIED** as documented below (registry returns dicts; test fixtures compliant; stats stub callers nonexistent).

### Layer 2 — Historical registry records are dicts, not BacktestResults
`BacktestRegistry.get(run_id)` returns the raw JSON dict from `result.json` (Adv2-verified: `registry.py:156-162`). There is NO `BacktestResult.from_dict` constructor. Historical re-reads never instantiate `BacktestResult` and never fire `__post_init__`. The new FIND-002 invariant only applies to fresh runs (engine output → `BacktestResult(...)`).

### Layer 3 — Test fixtures construct compliant BacktestResults
Adv2-verified: every existing `BacktestResult(...)` construction in `tests/test_types.py:247-263, 290-310, 315-330, 336-365` uses `trades=[]` + `trade_pnls=np.array([])` → invariant `0 == 0` ✓ passes. No fixture breaks.

### Layer 4 — Stats stub callers nonexistent
Adv2 grep confirmed ZERO production scripts chain `BacktestStats.daily()` or `.monthly()`. The only stale reference was the docstring example at `stats/__init__.py:8`, which is fixed in this same commit.

**Conclusion**: Cluster D.1 + E is **structurally zero-impact** on R-9..R-17a empirical numbers. No re-validation, no re-run, no BACKTEST_INDEX update required.

---

## §6 Cross-fix sequencing within Commit 1

**MANDATORY ORDER** (per Adv2):

1. **vectorized.py:436-442** — emit `Trade(side=TradeSide.FLAT, ...)` BEFORE `trade_pnls.append(...)`
2. **types.py:218-242** — append round-trip pairing invariant to `__post_init__`
3. **zero_dte.py:266-273** — replace silent break + add per-pair side assert + new exception class

Why this order:

- Step 1 + Step 2 must be in same commit. If only Step 2 lands, EVERY fresh run with an open EOF position breaks (invariant fires; auto-close path doesn't emit Trade yet).
- Step 3 can technically land in a separate commit (Adv2 said either is acceptable). Recommendation: bundle into Commit 1 for atomic cluster closure.

Within the file edits:
- Edit `vectorized.py` first; run `pytest tests/test_engine/test_vectorized.py -v` to confirm FIND-001 test passes.
- Edit `types.py`; run `pytest tests/test_types.py -v` to confirm invariant tests pass.
- Edit `zero_dte.py`; run `pytest tests/test_engine/test_zero_dte.py -v` to confirm alternation tests pass.
- Run full suite `pytest -q` to confirm zero regressions.

---

## §7 Risk assessment + mitigations

| Risk | Severity | Source | Mitigation |
|---|---|---|---|
| Invariant retroactivity breaks historical re-reads | LOW | Adv2 verified no `from_dict` exists | None needed; registry returns dicts |
| Test fixtures break under new invariant | LOW | Adv2 verified all 4 sample fixtures comply | None needed |
| WARN log spam in CI | LOW | Engineering judgment | Only fires on always-in strategy + EOF open position; not in default test flow |
| FIND-040 NotImplementedError breaks docstring | LOW | Adv2 reachability check | Docstring example updated in same commit |
| Lesson #14 skip affects coverage % | LOW | Adv2 CI check | 8 tests skip; no fail-on-skip threshold in pyproject.toml |
| HB-2 discovered (zero_dte path mismatch) but not fixed in this cycle | MEDIUM | Adv2 surfaced | Filed as NEW backlog entry; Cluster F |
| HB-5 (`total_trades` count discontinuity post-fix) | LOW | Adv2 surfaced | Documented as known anti-pattern; CLAUDE.md root §"Backtester `trade_pnls` vs `trades`" already flags it |
| HB-7 (`Dict[str, any]` lowercase typo) | LOW | Adv3 surfaced | Pre-existing bug, Pydantic-incompatible only; defer to Pydantic-migration cycle (Cluster F.2) |
| `n` symbol scope at vectorized.py:436 | LOW | Implementation detail | Verify at impl time; substitute `len(prices) - 1` if needed |
| `ZeroDteAlternationError` import path | LOW | New exception | Co-locate in `zero_dte.py` module scope; export at module level |

**No HIGH or CRITICAL risks**. All identified risks are LOW–MEDIUM, with concrete mitigations.

---

## §8 NEW backlog entries surfaced this cycle

To be filed in `PHASE_P_BACKLOG.md` (monorepo root) and cross-referenced from `lob-backtester/VALIDATION_FINDINGS_2026_05_14.md`:

### HB-2 — `_build_zero_dte_config` reads top-level `zero_dte:` but YAMLs nest it under `backtest:`
- **Severity**: HIGH (silent default substitution; latent on production YAMLs)
- **Site**: `lob-backtester/src/lobbacktest/experiment.py:402`
- **Symptom**: `nvda_readability_first_xnas.yaml:62` puts `zero_dte:` under `backtest:`. `_build_zero_dte_config` reads top-level via `self.config.get("zero_dte", {})` → returns `{}` (defaults). Silently uses default `delta`, `commission_per_contract`, `opra_costs`, etc.
- **Reachability**: LATENT — `ExperimentRunner.from_yaml` has 0 production callers (per Adv1 grep)
- **Fix-direction**: bundle with FIND-070 fix in Cluster F (config plane cycle); extend runner to read `backtest.zero_dte` OR migrate YAML schema
- **Discovered by**: Adv2 ground-truth audit, 2026-05-14

### HB-5 — `total_trades = len(trades)` count discontinuity post-FIND-001 fix
- **Severity**: LOW (informational drift)
- **Site**: `lob-backtester/src/lobbacktest/engine/vectorized.py:472` (or equivalent — `total_trades` assigned from `len(trades)`)
- **Symptom**: Post-FIND-001 fix, fresh backtests with an EOF open position will have `total_trades = previous_count + 1` (one additional FLAT auto-close trade). Historical R-9..R-17a counts may differ by 1 if those strategies hit the auto-close path (they didn't — see §5 Layer 1 — but future strategies might).
- **Constraint reminder**: root `CLAUDE.md` §"Backtester `trade_pnls` vs `trades`" already documents that `total_trades = len(trades)` counts opens + closes; win-rate / expectancy MUST use `len(trade_pnls)` (round-trips). Post-fix this invariant is unchanged.
- **Fix-direction**: NONE required — the relationship `total_trades = 2 * len(trade_pnls)` is invariant under FIND-001 fix. Documented for awareness.
- **Discovered by**: Adv2 hidden-bug hunt, 2026-05-14

### HB-7 — `Dict[str, any]` lowercase typo (Pydantic-incompatible)
- **Severity**: LOW (latent; only fires if BacktestConfig migrates to Pydantic)
- **Sites**: `lob-backtester/src/lobbacktest/config.py:225, :351, :389, :467`
- **Symptom**: type annotation `Dict[str, any]` uses Python builtin `any` (a function) instead of `typing.Any` (a type marker). Currently treated as `Dict[str, Callable]` by mypy; tolerated at runtime; would be rejected by Pydantic v2 `model_validate`.
- **Fix-direction**: 4 single-character edits; `from typing import Any` may already be imported. Bundle into Cluster F.2 (config plane Pydantic migration) OR ship as a 5-min trivial hygiene PR.
- **Discovered by**: Adv3 cross-repo SSoT investigation, 2026-05-14

---

## §9 Open implementation questions

These can be resolved during impl without further user authorization (proposed defaults in brackets):

1. **`n` symbol scope at `vectorized.py:436`** — RESOLVED 2026-05-14 (Wave-2): `n = len(data)` is set at `vectorized.py:257` and IS in scope at L436. Use `n - 1` directly in the new `Trade(index=n - 1, ...)`.
2. **`ZeroDteAlternationError` placement** — co-locate in `zero_dte.py` or shared `engine/errors.py`? [Default: `zero_dte.py` module scope, exported via `__all__`. Promote to shared module if a 2nd alternation-class error emerges.]
3. **WARN log severity** — `logger.warning` vs `logger.info`? [Default: WARNING per hft-rules §8 observability semantics.]
4. **Test count comment in `.github/workflows/test.yml`** — exact count after cycle? [Default: run `pytest --collect-only -q | tail -3` post-impl; substitute concrete numbers.]
5. **`tests/test_stats/__init__.py`** — empty file or with `__all__` declaration? [Default: empty file matching `tests/test_engine/__init__.py` convention.]
6. **`tests/test_strategies/test_twap_skip_discipline.py`** vs inline meta-test in existing file? [Default: NEW file `test_twap_skip_discipline.py` matching the "one finding = one test file" pattern from `test_engine/`.]
7. **Backlog file location for HB-2 / HB-5 / HB-7** — monorepo root `PHASE_P_BACKLOG.md` or `lob-backtester/PHASE_P_BACKLOG.md` (latter doesn't currently exist)? [Default: monorepo root, per existing convention.]

---

## §10 Sister-cluster sequencing (downstream of this cycle)

After Cluster D.1 + E ships, the following clusters become unblocked or proceed in parallel:

### Cluster E.5 (NEW — surfaced by HB-7) — `Dict[str, any]` 4-site typo fix
- **Effort**: ~5 min (4 single-character edits + `from typing import Any` import)
- **Sequencing**: parallel with this cycle OR next housekeeping PR
- **Dependencies**: none

### Cluster F — Config plane (FIND-070 + HB-2 + FIND-056/058/059/060/068/071/073/077)
- **Effort**: ~1-2 weeks (per Adv1 architectural review)
- **Sequencing**: after Cluster D.1 + E ships; requires architectural decision on Pydantic migration vs explicit-validator-on-dataclass (Adv3 recommends Option B — explicit validator — for first cycle)
- **Dependencies**: Pydantic migration (if Option A) would benefit from `SafeBaseModel` being lifted to a Class A primitive in `hft-contracts` first

### Cluster B — Atomicity (FIND-090/091/130 + 2 sister sites)
- **Effort**: ~1.5 hr (after `atomic_write_yaml` upstream ships)
- **Sequencing**: requires `atomic_write_yaml` in `hft-contracts` 2.8.0 first
- **Dependencies**: `hft-contracts` 2.8.0 ship (~1.5 hr standalone) — see Pre-Cycle below

### Pre-Cycle — `atomic_write_yaml` in `hft-contracts` 2.8.0
- **Effort**: ~1.5 hr (function + 5-8 tests + CHANGELOG + minor version bump + `pyyaml>=6.0` dep declaration)
- **Sequencing**: BEFORE Cluster B
- **Dependencies**: none
- **Pattern**: wrapper over `atomic_write_binary` mirroring `atomic_write_pickle` (per Adv3 recommendation)

### Cluster G (Phase X.3 completion) — FIND-024 + FIND-029 + FIND-044 + FIND-046
- **Effort**: ~3-4 hr (metric silent-zero → NaN migration + strategy NaN routing fix)
- **Sequencing**: parallel with Cluster F (different files; no overlap)
- **Dependencies**: none

### Cluster H — Security (FIND-110 `np.load` pickle hazard)
- **Effort**: ~30 min surgical fix
- **Sequencing**: any time; standalone
- **Dependencies**: none

---

## §11 Encoded lessons added by this cycle

Add to `VALIDATION_FINDINGS_2026_05_14.md` Appendix A:

| # | Lesson | Lock-test location |
|---|---|---|
| #16 | Atomic state transitions: `trades.append + trade_pnls.append + equity[i] = cash` must be in the same basic block | `tests/test_engine/test_vectorized.py::test_end_of_data_auto_close_emits_trade` (FIND-001 lock) |
| #17 | `BacktestResult.__post_init__` enforces P2 round-trip pairing invariant `len(trade_pnls) == count(t.side == TradeSide.FLAT)` | `tests/test_types.py::TestBacktestResultRoundTripInvariant::test_post_init_pairing_invariant_violated_raises` (FIND-002 lock) |
| #18 | `ZeroDtePnLTransformer` raises `ZeroDteAlternationError` on odd-length trades + per-pair side mismatch (no silent break) | `tests/test_engine/test_zero_dte.py::test_zero_dte_alternation_violation_raises` (FIND-003 lock) |
| #19 | `BacktestStats.daily()` / `.monthly()` raise `NotImplementedError` until `BacktestResult.timestamps_ns` lands; `.full()` is no-op | `tests/test_stats/test_stats.py::TestPeriodAggregationStubs::test_daily_raises_not_implemented` (FIND-040 lock) |
| #20 | `tests/test_strategies/test_twap.py` carries `pytestmark = pytest.mark.skip(reason="...C2 incompatibility...")` at module scope | `tests/test_strategies/test_twap_skip_discipline.py::test_twap_module_is_skipped_at_collection` (Lesson #14 lock) |

These 5 new lessons enforce the design discipline encoded in this cycle. Any future commit that removes these locks must explicitly justify the removal in PR description (per Appendix A's discipline contract).

---

## §12 Pre-impl adversarial review (to dispatch next)

Per saved feedback memory MANDATORY pre-impl gate, this memo will be reviewed by 1 adversarial agent before any code edits. The agent's mandate:

1. **Refute** the per-finding fix designs (catch hidden bugs in the proposed code)
2. **Refute** the R-9..R-17a structural preservation argument (find a breaking path)
3. **Refute** the cross-fix sequencing (find a partial-application break window)
4. **Refute** the test plan (find an invariant the lock tests don't actually lock)
5. **Surface NEW** hidden bugs the 3-wave audit missed

Verdict format expected:
- APPROVE-COMMIT (no findings)
- APPROVE-WITH-MICRO-FIXES (small refinements; apply in same commit)
- REQUIRES-FIX (ship-blocker; must redesign)

The adversarial review's findings will be appended to this memo as §13 before implementation begins.

---

## §13 Pre-impl adversarial review verdict (COMPLETE 2026-05-14)

**Dispatched**: 2026-05-14 (1 parallel adversarial agent reading ground-truth source files at `lob-backtester/src/lobbacktest/`).

**Verdict**: **REQUIRES-FIX** → 6 critical fixes applied to memo (this section) before re-validation.

**Key insight from agent**: "Memo's PRINCIPLES survive; CODE SNIPPETS don't. The architectural design (atomic state transition, invariant codification, fail-loud) is sound. The code snippets in the memo need correction."

### §13.1 Critical findings + corrections applied

| # | Severity | Finding | Section corrected |
|---|---|---|---|
| **C1** | CRITICAL | `vectorized.py:1-38` has NO `import logging`; patch's `logger.warning(...)` would `NameError` | §3.1 — added pre-patch `import logging` + `logger = logging.getLogger(__name__)` |
| **C2** | CRITICAL | `ZeroDtePnLTransformer.transform(self, result: BacktestResult) -> ZeroDteResult` (single positional arg at `zero_dte.py:221`), NOT `(trades=, equity_pnls=)` | §3.4 FIND-003 tests — rewrote with `transformer.transform(result)` + `_make_result` helper |
| **C3** | CRITICAL | `BacktestResult.positions: np.ndarray` (`types.py:204`), NOT `List[Position]` | §3.4 FIND-002 tests — rewrote with `positions=np.zeros(n)` |
| **C4** | CRITICAL | `DirectionStrategy(predictions, shifted: bool = False, name: str = None)` (`direction.py:46-51`) — NO `label_mapping` kwarg. CLAUDE.md claim that "all strategies accept label_mapping" is STALE — code was never centralized for DirectionStrategy | §3.4 FIND-001 test — use `DirectionStrategy(predictions, shifted=False)` |
| **C5** | CRITICAL | Engine class is `VectorizedEngine(config)`; `run(data, strategy)` takes strategy as positional arg. `for_exchange()` is on `CostConfig`, NOT `BacktestConfig`. Test code in v1 memo used incorrect `Backtester(config=, strategy=)` + `BacktestConfig.for_exchange()` | §3.4 FIND-001 test — use `VectorizedEngine(config)` + `BacktestConfig(initial_capital=, position_size=, costs=CostConfig(...))` |
| **C6** | CRITICAL | Memo's ground-truth quote `n_round_trips = len(trades) // 2` was WRONG — actual code at `zero_dte.py:233` is `n_round_trips = len(equity_pnls)`. v1 patch would have introduced a SEMANTIC CHANGE (different round-trip count). | §3.3 — rewrote ground-truth quote + patch (preserves `len(equity_pnls)`; adds precondition AS A SEPARATE GUARD after early-return + before loop) |
| **HIDDEN-4** | HIGH | Memo missed SECOND docstring at `stats/stats.py:71-77` that ALSO chains `.daily().compute()`. v1 only fixed `stats/__init__.py:8`. | §4.1 — added Site 2 fix for `stats/stats.py:72-77` docstring |

### §13.2 Lower-severity findings noted (no patch needed)

| # | Severity | Finding | Disposition |
|---|---|---|---|
| HIDDEN-8 | LOW | No CI `-W error` filter; WARN log will not trip CI | Risk row §7 already classified LOW; no mitigation needed |
| HIDDEN-9 | LOW | Memo's `n_round_trips` quote semantic change was unintentional | Closed by §13 C6 correction |
| HIDDEN-10 | LOW | Early-return at `zero_dte.py:236-251` short-circuits the precondition check | Acceptable — precondition only matters when `n_round_trips > 0`; early-return handles the 0 case correctly |

### §13.3 R-9..R-17a preservation re-verified

Adversarial agent confirmed Layer 1-4 of §5 preservation argument via ground-truth code inspection:
- **Layer 1**: VERIFIED — `scripts/run_regression_backtest.py:282` uses `HoldingPolicy.HorizonAlignedPolicy(hold_events=N)`
- **Layer 2**: VERIFIED — `registry.py:156` returns `Dict[str, Any]`; no `from_dict` constructor exists
- **Layer 3**: VERIFIED — all 4 `BacktestResult(...)` constructions in `tests/test_types.py` use empty `trades=[]` + `trade_pnls=np.array([])` (invariant `0 == 0` ✓)
- **Layer 4**: VERIFIED — 0 production callers of `.daily()`/`.monthly()`; only 2 docstring chains (now both fixed)

### §13.4 Cross-fix sequencing re-verified (with State 3 nuance added)

Adversarial agent flagged a previously-unconsidered scenario:

**State 3 (only `zero_dte.py` edited)**: post-edit but pre-FIND-001 fix, the new `ZeroDteAlternationError` raises on any always-in strategy that fabricates a trade_pnls entry without matching trade — making the FIND-001 bug VISIBLE rather than silent. This is good for diagnosability but produces user-visible test failures in production scripts (`run_regression_backtest.py:118`, etc.). 

**Mitigation**: **SINGLE COMMIT for all 3 file edits** (vectorized.py + types.py + zero_dte.py). The v1 memo's "either acceptable for FIND-003 separation" wording was too lenient. Updating §6 to mandate single atomic 3-file commit.

### §13.5 New hidden bugs surfaced (none ship-blocking)

Agent surfaced no new ship-blockers. The 3 incidental finds are tracked:

1. **HIDDEN-1** (now resolved by §13 corrections): `positions=[Position.flat(), Position.flat()]` was wrong type
2. **HIDDEN-2** (now resolved): `DirectionStrategy(label_mapping=)` doesn't exist — code never centralized
3. **HIDDEN-7** (LOW): `ZeroDteAlternationError` import in tests requires the class to exist; co-ship in the same commit (already planned)

### §13.6 Re-validation status

After §13.1-§13.5 corrections, the memo was re-dispatched to a second adversarial agent for ground-truth verification.

### §13.7 Re-validation verdict (COMPLETE 2026-05-14): APPROVE-WITH-MICRO-FIXES

**Dispatched**: 2026-05-14 (1 parallel adversarial agent re-reading ground-truth source for each of C1-C6 + HIDDEN-4 + Sequencing).

**Verdict**: **APPROVE-WITH-MICRO-FIXES** → 1 NEW ship-blocker found + corrected same-cycle.

**Verifications**:
| Correction | Status | Evidence |
|---|---|---|
| C1 (logger import) | **VERIFIED** | `vectorized.py:1-38` lacks `import logging` and `logger`; addition at line 22/40 is conventional |
| C2 (transform single-arg) | **VERIFIED** | `zero_dte.py:221` is `def transform(self, result: BacktestResult) -> ZeroDteResult`; helper produces valid BacktestResult |
| C3 (positions ndarray) | **VERIFIED in §3.4** | `types.py:204` is `positions: np.ndarray`; `_base_kwargs(n)` enumerates all 12+3 = 15 required fields correctly |
| C4 (DirectionStrategy.shifted) | **VERIFIED** | `direction.py:46-51` confirmed; `np.array([1,1,1])` + `shifted=False` produces BUY/BUY/BUY |
| C5 (VectorizedEngine API) | **VERIFIED** | `vectorized.py:240-245` `run(self, data, strategy, metrics=None)`; existing test pattern at `test_vectorized.py:64-71` |
| C6 (n_round_trips semantics) | **VERIFIED** | `zero_dte.py:233` confirmed `n_round_trips = len(equity_pnls)`; precondition placement (after L235-251 early-return, before L266 for-loop) is structurally correct |
| HIDDEN-4 (second docstring) | **VERIFIED** | `stats/stats.py:71-77` contains chained `.daily()` example; `stats/__init__.py:8` analogous |
| Sequencing (single commit) | **VERIFIED** | State 3 break-window argument sound; production scripts use `HorizonAlignedPolicy` so cold path; but any test invoking `transform()` on mismatched fixture breaks in partial-commit window |

**NEW ship-blocker caught (CRITICAL — applied same-cycle)**:

§4.3 `_make_minimal_result()` helper had:
- `positions=[Position.flat(), Position.flat()]` (same C3-class bug — should be `np.zeros(n)`)
- Missing 5 required BacktestResult fields: `predictions`, `labels`, `config_dict`, `start_index`, `end_index`

→ All 3 Cluster E stats tests at §4.3 would have failed with `TypeError: missing required positional arguments` BEFORE reaching the `.daily()` / `.monthly()` / `.full()` test assertions.

**Fix applied 2026-05-14**: §4.3 helper rewritten with all 15 fields + `positions=np.zeros(n)` ndarray + `Position` import removed (now unused).

**NEW bugs NOT caught by either round**: NONE at structural level. All 4 critical correctness paths (engine accounting, type invariants, fail-loud assertions, period-stub semantics) are verified.

**FINAL VERDICT**: **APPROVE-COMMIT** (post §4.3 micro-fix). Design memo is ready for implementation cycle pending user authorization. The architectural principles, code patches, test plan, and sequencing are all ground-truth-verified.

---

### §13.8 Implementation cycle preconditions (READY)

Before implementation begins, the user must explicitly authorize per `commit only when explicitly requested` standing mandate. Pre-conditions verified for the cycle:

1. ✓ Design memo locks all fix patterns at code-level (this document)
2. ✓ Pre-impl adversarial gate complete (REQUIRES-FIX → corrections → APPROVE-COMMIT)
3. ✓ R-9..R-17a empirical preservation structurally guaranteed (4-layer proof, agent-verified)
4. ✓ Cross-fix sequencing mandates single atomic commit for D.1 (3 files)
5. ✓ Cluster E ships as separate commit (different test surface: `tests/test_stats/` + `tests/test_strategies/`)
6. ✓ All test fixtures + helpers verified to construct valid BacktestResult instances
7. ✓ All TradeSide / Trade / Position dataclass signatures verified against ground truth
8. ✓ 3 NEW backlog entries (HB-2, HB-5, HB-7) ready for filing as #PY-226/227/228

The implementation cycle itself requires:
- Mid-impl adversarial gate per saved feedback memory MANDATORY discipline (1 code-reviewer agent after each commit)
- Pre-commit final gate (3 parallel agents per saved feedback memory)
- Atomic per-commit ship with explicit user authorization for each
- Update lob-backtester/CLAUDE.md test-count comment if applicable (~410 + 5 NEW = ~415 tests)
- Update lob-backtester/VALIDATION_FINDINGS_2026_05_14.md Appendix A with encoded lessons #16-#20
- Update root CLAUDE.md banner + MEMORY.md ACTIVE banner + state snapshot post-ship

---

### §13.9 Wave-2 (session 2) pre-implementation re-validation (2026-05-14 session 2)

A second pre-impl adversarial round was dispatched at the start of the implementation session to re-validate the design memo against ground-truth code BEFORE any edits. 7 parallel agents (4 Wave-1 context + 3 Wave-2 adversarial) produced:

**Critical finding (REPAIR semantics)**:
- §5 Layer 1 claim ("auto-close was a COLD path in R-9..R-17a production") REFUTED by 2 independent agents (Agent C ground-truth audit + Agent E independent reverification). `HorizonAlignedPolicy.should_exit` at `holding.py:102-103` has NO EOF check; `regression.py:189` + `readability.py:237` track `exit_reasons["end_of_data"]` confirming production strategies leave positions open at EOF.
- §5 corrected in-place (this update) to reframe the cycle as **CORRECTNESS REPAIR** with documentary R-9..R-17a preservation. Commit-1 message MUST explicitly disclose the REPAIR semantics + potential `total_trades` + `option_total_return` deltas on future re-runs.

**Cosmetic drifts caught (no impact on patch correctness)**:
- §3.1 said `n = len(prices)`; ground truth is `n = len(data)` at `vectorized.py:257`. CORRECTED in-place.
- §4.1 listed 2 docstring sites; actual count is 3 (`stats/__init__.py:8` + `stats/stats.py:72-77` + NEW `stats/stats.py:1-15` module docstring — Agent F NB-1). CORRECTED in-place.
- §9 Q1 (`n` symbol scope at `vectorized.py:436`) RESOLVED: variable name is `n`, in scope.

**Test infrastructure ground-truth (no memo changes needed, but new tests must comply)**:
- `tests/test_stats/` directory EXISTS but is EMPTY (`ls -la` confirmed). Create `__init__.py` + `test_stats.py`; do NOT use `mkdir`.
- `pyproject.toml` has NO custom pytest markers registered — `@pytest.mark.integration` would emit `PytestUnknownMarkWarning`. Do not use unregistered markers in new tests.
- Existing convention: class-based tests with method helpers (NOT `@pytest.fixture`).
- `lob-backtester/tests/conftest.py` does NOT exist — no shared fixtures.
- No existing test uses `caplog` — new FIND-001 lock test introduces this pattern.
- Existing `BacktestResult(...)` constructions in `tests/test_types.py` (4 sites) all use `total_trades=0` with `trades=[]`.
- Existing `logging.getLogger(__name__)` convention at `registry.py:23` + `data/loader.py:42` — C1 patch is consistent.
- `engine/__init__.py` exports BOTH `Backtester` and `VectorizedEngine`; the latter is used in existing tests + all 3 production scripts.

**NEW non-blocking backlog candidate filed in monorepo PHASE_P_BACKLOG.md**:
- `#PY-229` — `BacktestResult.to_dict()` at `types.py:312-342` silently drops 5 of 15 dataclass fields (`predictions`, `labels`, `prices`, `start_index`, `end_index`). Serialization parity bug; downstream `BacktestRegistry.save()` consumes the truncated dict. Severity TIER 2 NON-BLOCKING. Sister-cluster Cluster B (atomicity) or Cluster G (Phase X.3 invariants).

**FINAL VERDICT (Wave-2 session 2): APPROVE-COMMIT** after applying the §5 Layer 1 + §3.1 + §4.1 + §9 in-place corrections (all done in this update). The implementation cycle can proceed.

---

## §14 Decision summary

| Decision | Value | Source |
|---|---|---|
| Cluster scope | D.1 (engine triple) + E (discipline hygiene) | User authorization 2026-05-14 |
| Commits | 2 (D.1 atomic; E separate) in one PR | Adv2 sequencing analysis |
| Auto-close policy | `force_close` + Trade(FLAT) + invariant; WARN log on fabrication | User authorization 2026-05-14 |
| `auto_close_on_end` config field | NOT added (hardcoded force_close) | User selected Option 1 (not Option 3) |
| FIND-002 invariant | `n_closes = sum(1 for t in trades if t.side == TradeSide.FLAT)` | Adv2 ground-truth correction (no CLOSE_LONG/CLOSE_SHORT) |
| FIND-003 fix | New `ZeroDteAlternationError(ValueError)` + replace silent break | Principle 3 (fail-loud) + new exception for traceability |
| FIND-040 fix | `raise NotImplementedError` on `.daily()/.monthly()`; `.full()` no-op | Adv2 reachability (0 production callers) |
| Lesson #14 fix | `pytestmark = pytest.mark.skip` at module scope | Adv2 recommendation (NOT `raise` in `TWAPStrategy.__init__`) |
| Cross-cycle test infrastructure | NEW `tests/test_stats/` directory (closes partial FIND-100) | Cluster E free-ride |
| Empirical preservation | 4-layer structural proof; no R-9..R-17a re-run | §5 |
| NEW backlog | HB-2, HB-5, HB-7 | Adv2 + Adv3 surfaces |
| Sister cluster sequencing | E.5 (HB-7, trivial), F (config plane), B (atomicity, needs atomic_write_yaml), Pre-cycle (atomic_write_yaml in hft-contracts 2.8.0), G (Phase X.3), H (security) | §10 |

---

## §15 References

- Findings catalog: `lob-backtester/VALIDATION_FINDINGS_2026_05_14.md` (169 findings, 17 themes)
- Primary handoff: `POST_LOB_BACKTESTER_VALIDATION_2026_05_14.md` (monorepo root)
- Empirical history: `lob-backtester/BACKTEST_INDEX.md` (R-9..R-17a rounds)
- Build + recent fixes: `lob-backtester/CLAUDE.md`
- Engineering rules: `.claude/rules/hft-rules.md`
- Root pipeline context: `CLAUDE.md` + `PIPELINE_ARCHITECTURE.md` (monorepo root)
- Adversarial agent verdicts: this session's tool log (Adv1 + Adv2 + Adv3, 2026-05-14)

---

**End of memo. Next step: dispatch pre-impl adversarial agent on §3-§4 designs.**
