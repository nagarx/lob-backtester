# LOB-Backtester — Full-Module Re-Validation Findings (2026-05-30)

> **Status**: CANONICAL findings doc post a 7-agent read-only adversarial re-validation + maintainer ground-truthing.
> **Verdict**: **lob-backtester core is SOUND** — zero production-corrupting bugs. The one in-scope defect found was FIXED in commit `0fd41dc` (this cycle). All remaining findings are LOW / non-production / cross-repo.
> **Complements**: `VALIDATION_FINDINGS_2026_05_14.md` (the prior 3-wave audit; this doc re-validates that the post-audit state held + the #PY-263 cycle did not drift). `BACKTEST_INDEX.md` (empirical history), `CODEBASE.md` State-at-HEAD (cycle log), `CONTRIBUTING.md` (field discipline).
> **Purpose**: durable per-finding brief so a future session can pick up any deferred item WITHOUT re-running the validation. No new design solutions — validated state + ranked recommendations + effort estimates.
> **Scope**: full `lob-backtester` repo + its producer/consumer boundaries (`lob-model-trainer` signals, `hft-ops` orchestration, `hft-contracts` SSoT). HEAD at authoring: `0fd41dc` (after the G1a ledger-symmetry fix).
> **Update (2026-07-07)**: the §9.3 CROSS-REPO hft-ops item — the one non-deferred residual — has since **CLOSED**: hft-ops commit `e89c2fd` (2026-05-31) removed the `backtest_deeplob.py` default (`BacktestingStage.script` now `""`; an unset `script:` fails loud at validate-time). Closure annotated in place at §0 / §2-F3 / §6-R1 / §8 / §9.3 / §9.4. The lob-backtester-side R1/R2 fossil deletion remains open.

---

## §0 Executive summary

A 7-agent read-only adversarial cascade (Wave-1 ×5 correctness surfaces + Wave-2 ×2 verdict-attack + design/architecture) plus maintainer Bash/grep ground-truthing attacked the correctness core and **broke nothing**. The money-math, the `#PY-263` G1b annualization-only invariant, and the upstream signal-consumption contract are all re-verified **correct**. Our prior `#PY-263` edits (commits `00638ac` + `a646187`) are correct and did **not** drift from goal.

**Two prior-agent over-amplifications were corrected by ground-truth grep** (the anti-delusion discipline — see §3). The one genuine in-scope defect (the G1a readability-ledger annualization asymmetry) was **fixed this cycle** (`0fd41dc`). Everything else is LOW / non-production / cross-repo and is deferred with full context below.

### Triage matrix

| Tier | Count | Items |
|---|---|---|
| **PRODUCTION-CORRUPTING bug** | **0** | — (the core is sound) |
| **In-scope defect — FIXED this cycle** | 1 | F1 G1a readability-ledger annualization asymmetry (`0fd41dc`) |
| **LOW / non-production — deferred** | 4 | F2 DA/SignalRate-on-Signal-enum (2 fossil scripts only), F4 Cap-3 sub-$100 undersizing (NVDA-unreachable), F5 negative-equity distortion (NVDA-unreachable), F7 ConfusionMetrics dead+untested |
| **MEDIUM — deferred** | 3 | F3 `backtest_deeplob.py` hft-ops-default argparse trap (primary fix CROSS-REPO — **since LANDED in hft-ops, `e89c2fd` 2026-05-31; see §9.3**), F6 `data/prices.py` denorm untested (off production path), F8 no executed e2e golden P&L |
| **Over-amplified prior-agent claims — corrected** | 2 | F6 was rated "CRITICAL feeds all P&L" → actually off-path; F7 was rated "HIGH untested" → actually dead code |
| **Re-verified CORRECT (negative results)** | many | engine money-math, G1b invariant, signal contract, mutexes, the 5 prior #PY-263 tests — see §5/§6 |

---

## §1 Method

Read-only, ground-truth-over-docs. All agents Opus, max effort. Findings required file:line evidence + severity + confidence; speculative claims were forbidden.

- **Wave 1 (5 agents)**: (A) our `a646187` edits + the annualization-only invariant; (B) core engine + 0DTE P&L + cost math; (C) upstream signal-consumption contract; (D) test-coverage sufficiency; (E) deferred-backlog + refuted-non-issues re-validation FROM SCRATCH.
- **Wave 2 (2 agents)**: (W2-A) hostile attack on the soundness verdict (try to break the 5 core claims / find a missed HIGH); (W2-B) design/architecture/long-term-maintainability lens.
- **Maintainer ground-truthing**: grep of every `zero_dte`/`metrics=`/`PriceExtractor`/`ConfusionMetrics` call site; Read of the engine auto-metric block, `data/prices.py`, both scripts' persistence blocks; a stash-revert regression-lock proof; a stash-compare lint-neutrality proof.

---

## §2 Findings (reconciled, de-amplified)

| # | Finding | Severity | Production reach | Status | Evidence |
|---|---|---|---|---|---|
| **F1** | G1a persisted annualization into the readability *registry* `config_dict` (nested) but NOT its hft-ops **ledger record**, while regression carries it top-level → `compare_experiments` cross-run grouping silently mis-handled readability ledger rows | MEDIUM | ledger/audit only (not P&L) | **FIXED `0fd41dc`** | `run_readability_backtest.py:480-491` (was missing) vs `run_regression_backtest.py:653-654` |
| **F2** | `DirectionalAccuracy`/`SignalRate` fed the `Signal` enum `{-1,0,1,2(EXIT)}` as `predictions` vs shifted class labels `{0,1,2}` → garbage metric where it fires | LOW | **None** — fires only in `metrics=None` auto-block; the 3 production scripts pass explicit `metrics=` | DEFERRED (D3) | `vectorized.py:674-679` (+`:479/:491` feed) → `metrics/prediction.py:31-126`; fires only in `param_sweep.py:199` (prints wrong DA at `:220`) + `backtest_deeplob.py:170` (computes its OWN correct DA at `:366-369`, discards the broken one) |
| **F3** | `backtest_deeplob.py` is the hft-ops `BacktestingStage` **default** script, yet its argparse rejects the orchestrator's `--signals/--name/--max-spread-bps/--output-dir` → `SystemExit(2)` (fail-LATE, opaque) | MEDIUM (latent) | **Latent — 3 enabled full-pipeline manifests do NOT override `script:`** (`nvda_tlob_h10_v1`, `nvda_tlob_h100_tb_volscaled`, `nvda_hmhp_tb_volscaled`) → route to the broken default; no e2e-run evidence; fail-loud abort, NOT corruption (corrected 2026-05-30 — see §9.2) | DEFERRED (D1) — **primary fix CROSS-REPO in hft-ops** → **LANDED 2026-05-31 (`e89c2fd` — default removed, unset `script:` fails loud at validate-time; see §9.3)** | `hft-ops/.../manifest/schema.py:277` (default at authoring; since removed) + `stages/backtesting.py:100-168` (flags) vs `scripts/backtest_deeplob.py` argparse `:207-265` |
| **F4** | Cap-3 position sizing silently caps share count for assets priced ≤ `initial_capital/1000` (=$100 default) with NO warning → >50% notional distortion for low-priced assets | LOW (latent) | None (NVDA ~$180; prod `position_size=0.1`) | DEFERRED | `vectorized.py:541-546` (`reference_price = max(price, initial_capital/1000)`) |
| **F5** | Under `position_size=1.0` + an extreme (~100×) adverse move, repeated shorts can drive equity negative; once negative, `returns = diff(equity)/equity[:-1]` is wrong-signed + raw drawdown >1 before the `[0,1]` clamp masks it | LOW (latent) | None (NVDA-unreachable + prod sizing 0.1) | DEFERRED | `vectorized.py` equity/returns construction; reproduced only with adversarial sizing |
| **F6** | `data/prices.py` denormalization is value-untested (mutation `+mean`→`-mean` → 0 test failures) | MEDIUM (test-gap) | **OFF the production path** | DEFERRED | `prices.py:122` math is the correct z-score inverse; used only by test-only `DataLoader` + fossils `param_sweep.py`/`backtest_deeplob.py`; the live scripts consume the trainer's pre-exported `prices.npy` via `from_signal_dir` |
| **F7** | `ConfusionMetrics` is value-untested (mutation `==`→`!=` → 0 test failures) | LOW | None | DEFERRED (delete or quarantine) | `metrics/prediction.py:339` — **dead code: zero callers anywhere** in src/scripts/tests; not exported in any `__init__` (only stale in the `CODEBASE.md`/`README.md` metric inventory) |
| **F8** | No *executed* end-to-end golden test (real signals → known P&L + known Sharpe); the real-data integration test is fully skipped | MEDIUM (test-gap) | — (units golden; only the assembled seam is structurally checked) | DEFERRED | 8 of 16 skips = `test_integration_real_data.py` (torch + data gated); the running e2e tests assert `returncode==0` + persisted-field structure, not golden numbers |

---

## §3 Over-amplified prior-agent claims — CORRECTED (anti-delusion record)

Two Wave-1 findings were rated too high; ground-truth grep de-amplified them. **Recorded so a future session does not chase a phantom.**

- **F6 `data/prices.py` — rated "CRITICAL, feeds all P&L" → actually MEDIUM test-gap, OFF the production path.** The 2 live production scripts (`run_regression`/`run_readability`) consume `data.prices` from `from_signal_dir` (the trainer's pre-exported `prices.npy`) — they never call `PriceExtractor.denormalize_prices`. That denorm is reached only by `DataLoader` (imported by **no** script — test-only) + the 2 fossils. And the math at `prices.py:122` (`normalized*std + mean`) is the correct z-score inverse. So a regression there cannot corrupt a production backtest.
- **F7 `ConfusionMetrics` — rated "HIGH untested" → actually LOW (dead code).** It has zero callers and is not exported in any `__init__`. A bug in unreachable code corrupts nothing. (NB: the 2026-05-14 doc refuted a *different* `ConfusionMetrics` concern — an "ABC/CompositeMetric pattern" worry; this cycle's point is simply that it is unused.)

---

## §4 The fix shipped this cycle (`0fd41dc`, pushed origin/main, CI green run 26691244411)

**F1 — G1a readability-ledger annualization symmetry.** Added `resolved_periods_per_day` + `annualization_factor` top-level to the readability hft-ops ledger record (`run_readability_backtest.py:480-491`), `float()`-cast, reusing the `BacktestConfig` properties (§0 — no duplicated `23400/bin_seconds` math), symmetric with the regression sister-record (`:653-654`). + `TestPy263ReadabilityLedgerAnnualizationPersisted` (a value-locked subprocess+ledger test), **PROVEN to fail without the fix** via a stash-revert of only the script. Suite 624 → 625 (609 pass + 16 skip); zero regressions; zero new lint (6 pre-existing ruff F541 unchanged, stash-compared).

---

## §5 Test-coverage map — "can we rely on each module blindly?"

**YES for the production P&L core; NO for three non-production surfaces.** Coverage classes: GOLDEN (value-locked numeric assertions) / BEHAVIORAL / SMOKE / NONE. Mutation-proofs in brackets.

**Trust blindly (GOLDEN + mutation-resistant):**
- `engine/vectorized.py` [P&L sign-flip → 5 failures], `engine/zero_dte.py` (BSM theta / OPRA cost / SELL→put golden)
- `config.py` (`resolved_periods_per_day`==390 + `annualization_factor` + both mutexes — the #PY-263 lock, 4 independent places)
- `metrics/risk.py` (Sharpe/Sortino/MaxDD/Calmar), `metrics/trading.py` (WinRate/PF/Expectancy — uses `len(trade_pnls)`, not `total_trades`), `metrics/returns.py`, `metrics/prediction.py` DA/SignalRate/Up·DownPrecision portion
- `data/signal_manifest.py` (shape/NaN/Phase-II fingerprint), `data/loader.py` (strict-validation path), `types.py`
- `strategies/{direction,readability,regression,hybrid,holding}.py` (exact-array + gate-boundary + NaN-guard)

**Do NOT trust blindly (gaps — none on a production-corrupting path):**
- `data/prices.py` denormalization — SMOKE only [`+mean`→`-mean` → 0 failures] (F6; off production path)
- `metrics/prediction.py::ConfusionMetrics` — NONE [`==`→`!=` → 0 failures] (F7; dead code)
- `reports/summary.py` (NONE), `reports/plots.py` (NONE) — orphan `reports/` (~519 LOC, 0 callers; prior-cycle N3)
- no executed e2e golden P&L (F8)

**Skips (16):** 8 = `test_integration_real_data.py` (torch + exported-NPY + checkpoint gated — none present in CI) → no executed top-to-bottom golden; 8 = intentional TWAP module-skip (Lesson #14, locked by `test_twap_skip_discipline.py`). Neither hides a *live* regression.

---

## §6 Design / architecture recommendations (ranked; W2-B + reconciliation)

The module is **well-architected and unusually disciplined** (clean `hft_contracts` SSoT reuse, exemplary §8 dead-field quarantine with DeprecationWarning + removal dates, pervasive §5 fail-fast). Standing debt is concentrated in 3 spots. None silently corrupts P&L today.

| Rank | Recommendation | Effort | Scope | Why it matters (years-horizon) |
|---|---|---|---|---|
| R1 | Neutralize the `backtest_deeplob.py` silent-default landmine — delete the torch/DeepLOB fossil so nothing can be the hft-ops `script:` default that argparse-rejects its own caller (F3) | S | lob-backtester (primary fail-loud fix is hft-ops, cross-repo — **landed 2026-05-31 `e89c2fd`, see §9.3**; the fossil deletion is the remaining lob-backtester half) | A stage default that can't be invoked by its caller is a §5 fail-LATE trap; burns the first operator who omits `script:` |
| R2 | Delete the DeepLOB-era fossils (`backtest_deeplob.py`, `param_sweep.py`; likely `e5_regime_filter_test.py`) — they target legacy `nvda_balanced` / `price_means` normalization that no longer exists in v3p0 | S | lob-backtester | Rot that contradicts the current data contract + is where F2/F6 "fire" |
| **R3** | **Unify the two run-paths behind one shared `build_backtest_config(args_or_yaml)` SSoT** consumed by both `scripts/run_*.py` and `ExperimentRunner` | M | lob-backtester | **The one structural years-horizon item.** #PY-263 had to be fixed TWICE (V1 script + G1b ExperimentRunner); two `_iv_default` blocks already diverge. Today mitigated by the `experiment.py:7-15` STATUS banner + tests, not prevented |
| R4 | Fix the G1a key asymmetry (F1) | — | — | **DONE** `0fd41dc` |
| R5 | Quarantine or delete `ConfusionMetrics` (F7) + drop it from the `CODEBASE.md`/`README.md` inventory | S | lob-backtester | The one *un-quarantined* dead symbol (the config dead-fields are exemplary) |
| R6 | Rename `e5_regime_filter_test.py` out of the `*_test.py` namespace (+ fossil header) | S | lob-backtester | `*_test.py` in `scripts/` is a pytest-collection landmine |
| R7 | Add `lob-backtester` to the §4 soft-WARN hook's `SCOPE_REPOS` (`.claude/hooks/check_scripts_header.py:34`) | S | monorepo hook (cross-repo) | This repo's `scripts/` get zero header enforcement — the root cause of R2/R6 |
| (D3) | Fix the `DirectionalAccuracy`/`SignalRate` auto-metric (drop it from the `metrics=None` default, or feed class predictions with the correct `shifted=` flag) (F2) | S | lob-backtester | Real but non-production; metric-hygiene |

---

## §7 Explicitly FINE — do NOT "fix" (resisting invented debt)

- `CostConfig.maker_rebate_bps` inert (taker-only engine; 0DTE discards `CostConfig`) — wiring it would **introduce** a bug. Tested (`test_config.py`). **Leave inert.**
- All config DeprecationWarning dead-fields (`target_holding_minutes`, `entry_window_{start,end}_et`, `fill_price`, `stop_loss_pct`, `take_profit_pct`) — textbook §8 quarantine with removal dates + tests. **Good design, not debt.**
- `strategies/twap.py` skip — deliberate (Lesson #14), tested. **Leave alone.**
- `data/signal_manifest.py` shim over `hft_contracts.signal_manifest` + all `hft_contracts` reuse — clean SSoT. **Leave alone.**
- `run_spread_signal_backtest.py` `periods_per_day=245.0` hardcode + non-orchestratable status — documented intentional. **Leave alone** (just give it a header per R6's grouping).
- **`allow_short=True` default (`config.py:495`) — DO NOT FLIP.** Empirically load-bearing: a SELL on a flat book is skipped when `allow_short=False` (`vectorized.py:350-353`), and `ZeroDtePnLTransformer` maps SELL→put (`zero_dte.py:373`); flipping silently deletes the put leg from every documented R1-R8/E5/E6 result (reproduced: 6 SELL entries → 0). Document-only (prior-cycle D4).

---

## §8 Next-session options

lob-backtester is **trustworthy** and `#PY-263` is fully closed on the 3 production paths. Reasonable next moves:

1. **Move to another repo** — lob-backtester needs nothing urgent.
2. **Design hygiene cycle (lob-backtester)** — highest long-term value is **R3 (unify the two run-paths behind a shared config-builder SSoT)**; then R2/R5/R6 (fossil + dead-code cleanup), F2 (DA-metric fix), and the test gaps (F6 `prices.py` golden + F8 executed e2e golden P&L). Many overlap the prior-cycle dead-code backlog (N2–N9 in `CODEBASE.md`).
3. **Cross-repo items (coordinate with the owning session)** — F3/R1 (`backtest_deeplob.py` hft-ops default fail-loud) + R7 (§4 hook `SCOPE_REPOS`) are primarily hft-ops / monorepo-hook fixes; the lob-backtester-side mitigation is just deleting the deeplob fossil (R2). **[F3/R1's hft-ops half landed 2026-05-31 (`e89c2fd`) — see §9.3; R7 + the R2 fossil deletion remain.]**

---

## §9 Follow-up cycle (2026-05-30, post-compaction re-validation — "are we done?" audit)

A fresh-eye 5-agent adversarial re-validation (run after a session compaction; **anti-anchored** — 4 agents formed judgment from the code *before* reading this doc, 1 audited these conclusions top-down) **re-confirmed the §0 verdict**: zero production-corrupting bugs (the money-math was independently re-derived by hand and stress-probed for boundary/NaN/lookahead failures — the claim *stands*), a sound producer→consumer + orchestration boundary, and no blocking incomplete work in lob-backtester's own code. It also surfaced **two items the prior pass under-rated**, both ground-truthed by the maintainer and closed/corrected here.

### §9.1 — 0DTE assembled-P&L golden lock ADDED (closes the one rely-blindly gap)

The 0DTE Deep-ITM return is the headline output of the module, yet the **assembly** at `zero_dte.py:405-434` had no value-lock — only its components (`theta_bsm_per_share` via `TestThetaBsmFormula`; `round_trip_cost_per_contract` via `TestOpraCalibratedCosts`) did. A `direction` sign-flip (`:388`), an `is_call` ternary inversion (`:373`), a dropped/added cost term, or `gross` using `exit_price` instead of `entry_price` (`:405`) would have silently flipped the reported return and passed the full suite. **Not a bug** (hand-verified correct today by the bug-hunt agent) — a **regression-guard gap** on the most material number this module produces.

**Closed** by `TestZeroDteAssembledPnlGolden` (3 tests, `tests/test_engine/test_zero_dte.py`). A 2-leg fixture (BUY/call 100→101 + SELL/put 100→99, `prefer_calls=True`, `events_per_minute=1.0` → 10-event/10-min hold) value-locks:
- `underlying_moves_bps == [+100.0, +100.0]` — the directional sign for the call (BUY, up) AND the short/put (SELL, down) legs; a `direction` sign-flip gives `[-100, -100]`.
- `is_call == [True, False]` — the BUY→call / SELL→put mapping.
- `spread_costs == [3.0, 2.0]` — the `is_call`→`half_spread` selection (call $0.015 vs put $0.010); a second, independent lock on the mapping.
- `option_trade_pnls == [43.2766…, 44.2766…]` — the full `gross − spread − comm − theta` assembly (`gross = 0.50·0.01·100·100 = $50.00` both legs).
- `option_equity_curve` (shape `(3,)`), `option_final_equity == 100087.553…`, `option_total_return == 0.0008755…` — the cumsum/return aggregate.

Expected values were independently re-derived AND verified bit-exact against the real `transform()` output by a pre-impl gate (`theta = 2.3233619070671425`/contract). The lock was **PROVEN to bite** by a stash-revert mutation test: flipping the `direction` sign → `test_move_bps...` FAILS; dropping the `theta_cost` term → `test_assembled...` FAILS; restoring → `zero_dte.py` byte-identical, all 3 pass. Suite 625 → 628 (612 pass + 16 skip); zero regressions; zero new lint. **The asymmetric fixture prices (exit ≠ entry) are load-bearing** — they make the exit-vs-entry-price gross mutation detectable; do NOT flatten the round-trips.

### §9.2 — F3 reach CORRECTED (the prior "all 9 manifests override `script:`" claim was wrong)

The §2 F3 cell originally read *"None today (all 9 live manifests override `script:`)"*. A ground-truth grep of `hft-ops/experiments/` **refutes this**: **3 enabled full-pipeline manifests do NOT override `script:`** — `nvda_tlob_h10_v1.yaml`, `nvda_tlob_h100_tb_volscaled.yaml`, `nvda_hmhp_tb_volscaled.yaml` (each `stages.backtesting.enabled: true`, no `script:`, carrying the legacy `model_checkpoint`/`data_dir`/`horizon_idx` deeplob interface) → they route to the `schema.py:277` default `backtest_deeplob.py`, whose argparse defines **none** of the orchestrator's `--signals/--name/--max-spread-bps/--output-dir/--manifest` flags → `SystemExit(2)`. It remains **latent** (no ledger evidence these reached the backtest stage end-to-end) and **fail-loud, not corrupting** (aborts the stage; never produces a wrong number). The hmhp `*feat` manifests DO override to `run_readability_backtest.py`. *Lesson: do not inherit a doc's reachability claim without re-grepping the manifests — one Wave-1 agent this cycle repeated the wrong "0 leave the default" claim; a second caught it; the maintainer grep settled it.*

### §9.3 — CROSS-REPO action item for the hft-ops-owning session

The primary fix for §9.2 is **in hft-ops, not lob-backtester** (out of this session's scope, per the standing "work the repo we started in" mandate): (a) change `BacktestingStage.script`'s default away from `backtest_deeplob.py`, (b) have `backtest_deeplob.py` use `parse_known_args` / accept the orchestrator flags, or (c) set `script:` explicitly on the 3 stale manifests. The lob-backtester-side mitigation is the prior-logged R1/R2 (delete the deeplob fossil). **No hft-ops file was modified this cycle.**

**CLOSED (2026-05-31; code-verified 2026-07-07):** hft-ops took option (a) the next day — commit `e89c2fd` ("fix(stages,cli): harvest regression backtest metrics + fail loud on missing backtest script") removed the default: `manifest/schema.py::BacktestingStage.script` is now `script: str = ""` with a "C2 (2026-05-31): NO default" comment, mirrored in `manifest/loader.py::_build_backtesting`, and an enabled backtesting stage with an unset `script:` now fails loud at validate-time — `stages/backtesting.py::BacktestRunner.validate_inputs` appends *"stages.backtesting.script is required (no default); use lob-backtester/scripts/run_regression_backtest.py (regression) or lob-backtester/scripts/run_readability_backtest.py (classification)."* The lob-backtester-side mitigation (R1/R2 — delete the deeplob fossil) remains OPEN.

### §9.4 — verdict

lob-backtester is **complete and trustworthy** by the rely-blindly bar: the production P&L core (equity + 0DTE assembled + costs + annualization + the 6 metrics + signal contract + BacktestResult invariants) is golden-locked and mutation-resistant; the remaining backlog (R3 two-run-path SSoT, dead-code N2-N9, `reports/` orphan, F4/F5 NVDA-unreachable latents, `prices.py` off-path, `ConfusionMetrics` dead) is genuinely optional and safe to defer. The only non-deferred residual is §9.3, which is a cross-repo hft-ops item **(since CLOSED in hft-ops 2026-05-31, `e89c2fd` — see the §9.3 closure note)**.

---

> **Provenance**: this cycle's headline + the fix are also in `CODEBASE.md` State-at-HEAD (committed `0fd41dc`) and the `0fd41dc` commit message. This doc is the full per-finding detail. The §9 follow-up (2026-05-30 post-compaction "are we done?" re-validation) + the §9.1 golden lock + the §9.2 F3 correction are reflected in `CODEBASE.md` State-at-HEAD (pending commit).
