# LOB-Backtester: Codebase Technical Reference

> **Version**: 0.1.0 | **Tests**: 666 collected (verify pass/skip: `pytest --collect-only -q | tail -1`; was 628/43-files at the 2026-05-30 #PY-263 cycle, +38 tests / +2 files from the B1-B4 engine-realism `e08fb89` + deep-ITM demo `6295d82` commits) | **Last Updated**: 2026-05-30 (#PY-263 annualization closure across the 3 production backtest paths — regression + readability scripts + ExperimentRunner [back-compat-default `backtest_deeplob.py` deferred: same gap but emits the DeprecationWarning, non-silent, no live manifest selects it] — R1 `run_readability_backtest.py` `bin_seconds` threading [sister to V1's run_regression fix] + G1a persist `resolved_periods_per_day`/`annualization_factor` into backtest artifacts + G1b `ExperimentRunner._build_backtest_config` zero_dte threading + G1c ExperimentRunner status banner; same-day prior: V1 #PY-263 wiring in `run_regression_backtest.py` [`--bin-seconds` threaded into the config so `resolved_periods_per_day`=390 at 60s, was the 1000.0 fallback inflating equity Sharpe/Sortino/Calmar ~1.6×] + V2 FIND-058-EXT `entry_window_*_et` dead-field warnings + value-locked engine-P&L/e2e golden tests; cumulative through CI-green + hygiene bundle 2026-05-29)  
> **Purpose**: Complete technical details for LLMs and developers to understand, modify, and extend the codebase without prior context.

> **Pipeline scope (2026-06-02).** This module is part of an **intraday trading research pipeline** — an experiment-first platform for discovering and validating *any* profitable **intraday** trading edge (no overnight positions), across approach classes (microstructure/HFT, scalping, intraday momentum, intraday statistical arbitrage, …) and instruments (equities, futures, same-day options). The pipeline *originated* as a high-frequency NVDA MBO/LOB microstructure system — that origin explains the "HFT" / "LOB" / "MBO" naming here — and that microstructure-direction program is now one (largely-closed) track among many. **Names are historical; the mission is general.** This module's role: the P&L backtester — a generic single-asset linear long/flat/short engine (`VectorizedEngine`) with an optional, separable 0DTE-options overlay (`ZeroDtePnLTransformer`) and IBKR-calibrated costs; reusable for equity/futures intraday directional P&L. For the full mission + approach taxonomy + capability-readiness boundary, see root `CLAUDE.md` §Research Scope & Charter (+ `CROSS_ASSET_OFI_FINDINGS_AND_ISSUES_2026_06_01.md` §9).

## State at HEAD (cumulative through Phase 7 Round 5)

- **Phase 0-3 stabilization** (commit `47d0d31`) — typed `BacktestContext`, `ExperimentRunner` orchestration, `ZeroDtePnLTransformer` (IBKR-calibrated 0DTE options P&L), `LabelMapping` SSoT, `HoldingPolicy` composability, P2/P3/P4/P5 fixes.
- **Phase 4 4c.4** (`de94078`) — `SignalManifest.feature_set_ref` propagation from trainer's `signal_metadata.json`; Phase 6 6A.9 `_CONTENT_HASH_RE` producer-consumer regex symmetry.
- **Phase 6 6B.5** (`77f2068`) — `SignalManifest` canonical home co-moved to `hft_contracts.signal_manifest`; this repo's `data/signal_manifest.py` is a thin re-export shim preserving pre-6B.5 imports.
- **Phase 6 final hygiene** (`d642dd1`) — shim lazy `__getattr__` emits `DeprecationWarning` once per symbol per process.
- **Phase 7 post-validation I** (`c28fc53`, `09d4665`) — calendar-driven shim deadline `2026-10-31` replaces prior "version 0.4.0" version-milestone language (hft-ops has no fixed release cadence; calendar gives a concrete migration window).
- **REV 2 hft-contracts public-push follow-up** (`81eb6e9`, 2026-04-20) — backtester shim's `_PUBLIC_NAMES` frozenset extended with `CONTENT_HASH_RE` (REV 2 public name) so `from lobbacktest.data.signal_manifest import CONTENT_HASH_RE` resolves through the Phase 6 6B.5 re-export layer. Legacy `_CONTENT_HASH_RE` (now a `__getattr__`-based DeprecationWarning shim in hft-contracts, removal 2026-10-31) still accessible; `test_regex_is_module_level` extended to verify both names map to the same compiled pattern.
- **Cluster D.1+E close-out** (`ed39e1b`, 2026-05-14) — Closes FIND-001/002/003 (engine accounting triple: Trade(FLAT) on EOF auto-close + round-trip pairing invariant + ZeroDte alternation raise) + FIND-040 (`BacktestStats.daily/monthly` stubs raise NotImplementedError) + Lesson #14 (TWAP tests skipped at module scope via `pytestmark`). Adds 5 NEW Appendix A encoded lessons (#24-#28).
- **Cluster H + #PY-228 + FIND-067** (`b95a33b`, 2026-05-14) — FIND-110 closure: 25 `np.load(...)` callsites migrated to `allow_pickle=False` (pickle-RCE security hazard CLOSED) + AST regression-lock at `tests/test_security/test_np_load_allow_pickle_false.py`. #PY-228 closure: 5 `Dict[str, any]` (lowercase) → `Dict[str, Any]` (typing.Any) sites + alphabetized typing imports + AST regression-lock at `tests/test_type_annotation_discipline.py`. FIND-067 closure: dead `ComparisonConfig` class DELETED (zero monorepo-wide consumers verified pre-impl). Adds Appendix A lessons #29 + #30.
- **R-19 cycle (5 commits + expand)** (`9dcfab0` + `4ef0d7a` + `7cfe8f5`, 2026-05-15) — FIRST cross-architecture single-variable A/B in pipeline history. Round 19a TLOB × TB v3p0 verdict: REFUTE-WITH-ARCHITECTURAL-LIFT (NEW classification; H1 PRIMARY FAILS but H2 PT precision lift +5.8pp materially exceeds R-17a's +0.9pp — empirically REFUTES R-17a Lesson #95 "info-theoretic 22% ceiling" claim). FIND-090 closure (`4ef0d7a`): 3 registry write sites migrated to `hft_contracts.atomic_io.{atomic_write_json, atomic_write_binary}` SSoT. FIND-093 closure (bundled): 3 `datetime.now()` (local TZ) sites → `datetime.now(timezone.utc)` for cross-operator reproducibility. C5 scope expansion (`7cfe8f5`): 3 HIGH-ACTIVE sister sites in `scripts/run_regression_backtest.py:374,460` + `scripts/run_readability_backtest.py:349` migrated to `atomic_write_json` SSoT (hft-ops ledger linkage SIGKILL-corruption hazard CLOSED). +5 NEW AST regression-lock tests (1 spread-script UTC + 4 sister-site discipline) at `tests/test_registry_atomic_writes_and_utc.py` + `tests/test_scripts_atomic_writes.py`. Phase Y composer empirical validation: same `compatibility_fingerprint=dd21d079...` as R-17a (corpus identity preserved across model_type axis), different `model_config_hash=2dc7eeef...` (architectural axis correctly distinguished). Lessons #99-#104 chained from R-17a #94-#98.
- **Option D TIER 1 surgical cluster + NaN aggregator suite** (2026-05-17, single atomic commit pending push) — Closes 6 TIER 1 findings from 10-agent multi-wave audit. **#PY-305 closure**: `scripts/run_readability_backtest.py:123,125` sentinel-None mode-aware default for `--implied-vol` + `--entry-minutes-before-close` (class-coherent with `run_regression_backtest.py:389-415` pattern post-#PY-274; 4th-site sister of IV-inheritance class). **Wave 2-H H1 closure**: `registry.py:127` strftime collision via `secrets.token_hex(4)` 8-hex suffix (closes silent-overwrite on same-second registrations; pattern-recurrence of hft-ops `aeec3b0` parallel-session anti-pattern). **Wave 1A F2 + Wave 2-H H3 closure**: `config.py:__post_init__` fail-loud ValueError on `ZeroDteConfig(prefer_calls=False)` — option-P&L formula at `engine/zero_dte.py:354-375` is ATM-call-only; PUT delta plumbing deferred to #PY-311 Phase Z. **Wave 1C T1.1 closure**: `types.py` `BacktestResult.total_return` + `max_drawdown` properties return NaN on degenerate inputs (Phase X.3 sentinel; align with metric path). **Wave 1D T1-D-001/002/003 closure**: 3 NaN guards in holding/regression policies (`holding.py:151` DirectionReversalPolicy current_agreement + `:198,200` SLTP unrealized_pnl_bps + `regression.py:127` _build_holding_state pred_class) — fail-CLOSED on NaN (exit; mirrors entry-gate PRESERVE #25 convention from #PY-71 closure). **Wave 1F F5 + Wave 2-H H7 closure**: 9 NaN aggregator propagation tests at `tests/test_stats/test_nan_propagation.py` locking BacktestStats does NOT silently coerce NaN → 0 + metric-order independence. Adversarial cascade: 10-agent audit → 3 pre-impl agents (X design + Y cross-module + Z scope) APPROVE-WITH-REVISIONS → 1 mid-impl reviewer APPROVE-WITH-MICRO-FIXES (cite-drift) → 3 pre-commit final (code-reviewer APPROVE-COMMIT + doc-alignment APPROVE-WITH-3-MICRO-FIXES + hft-architect APPROVE-WITH-FOLLOWUP). +49 NEW tests across 7 NEW test files: `test_config_prefer_calls_raise.py` (7), `test_registry_collision_resistant_run_id.py` (4), `test_run_readability_py305.py` (5 integration), `test_stats/test_nan_propagation.py` (9), `test_strategies/test_holding_nan_guards.py` (12), `test_strategies/test_regression_nan_in_hold.py` (6), `test_types_nan_properties.py` (6). Test count 516 → 565 (549 pass + 16 skip); ZERO regressions. Phase Y CompatibilityContract invariant PRESERVED (NONE of 11 fields touched). ZERO new SSoT primitives. 3 NEW backlog filed: #PY-309 (max_drawdown property/metric clamp drift, HIGH), #PY-310 (NaN JSON serialization to_dict, MEDIUM), #PY-311 (prefer_calls=False Phase Z plumbing, MEDIUM-DEFERRED).
- **HF-2 BacktestStats Sharpe sister closure** (commit pending, 2026-05-23) — Sister-closure of `#PY-263` (`ae22b87`, 2026-05-21 BacktestConfig mode-aware `resolved_periods_per_day` dispatch). Closes silent Sharpe/Sortino/Calmar/AnnualReturn ~1.6018x inflation at 60s time-based bins via the operator-facing fluent `BacktestStats` API. Engine path at `vectorized.py:623-664` was correctly mode-aware via `BacktestContext` context propagation, but `BacktestStats.compute()` built its OWN context dict (no `periods_per_day` key) and constructed `SharpeRatio()`/`SortinoRatio()`/`CalmarRatio()`/`AnnualReturn()` with their class-default `periods_per_day=1000.0` — silently falling back via `context.get("periods_per_day", self.periods_per_day)` at `risk.py:118,227` + `returns.py:171`. **Implementation**: `BacktestStats.__init__` accepts keyword-only `periods_per_day: Optional[float] = None` (validates `> 0` if specified per §5; symmetric Q3 fix with chainable setter caught by mid-impl gate); NEW `with_periods_per_day(value)` chainable (idempotent last-call-wins; raises on ≤0); `compute()` injects into context dict OR emits `DeprecationWarning` with actionable migration message citing `BacktestConfig.resolved_periods_per_day` + `sqrt(1000/390)≈1.6018x` factor. **Tests**: +8 NEW `TestPeriodsPerDayHF2` class at `tests/test_stats/test_stats.py` with `_make_finite_result()` n=100 random-walk fixture (`seed=42`) + empirical anti-regression `test_explicit_periods_per_day_affects_annualized_metrics` proving `sharpe(1000)/sharpe(390) ≈ sqrt(1000/390)` within `rel_tol=0.01`. 4 existing `TestNanPropagation*`/`TestMetricOrderIndependence`/`TestMetricsDictHasExpectedKeys` test classes at `tests/test_stats/test_nan_propagation.py:71,130,178,223` decorated with message-scoped `filterwarnings("ignore:BacktestStats.compute.*periods_per_day not specified:DeprecationWarning")` (surgical scope; does NOT mask unrelated DeprecationWarnings). Adversarial cascade: 5 Wave 1 (1A empirical + 1B/1C/1D/1E agents) + 2 Wave 2 (REFUTE + hidden hunt) + 1 pre-impl design gate (APPROVE-WITH-REFINEMENTS) + 1 mid-impl reviewer (APPROVE-WITH-4-MICRO-FIXES same-cycle: Q3 symmetry + Q6 filterwarnings scope + Q7 idempotency doc + Q9 package docstring) + 3 pre-commit final (code-reviewer + doc-alignment + hft-architect ALL APPROVE-COMMIT). HF-1 sister surface (`BacktestContext` direct constructor still uses `1000.0` legacy default; only notebook-research path; 16 existing `test_context.py` tests assert default) DEFERRED to `#PY-NEW-HF1-CONTEXT-DIRECT-CONSTRUCTOR` next-cycle TIER 3 polish per pre-impl gate verdict. Test count: 573 → 581 + 16 skip = 597 collected (+8 NEW). ZERO regressions. ZERO new SSoT primitives. Phase Y CompatibilityContract invariant PRESERVED (NONE of 11 fields touched). Bundle A (`d09f772` + `34ab056` 2026-05-22) intact.
- **Path C-EXTENDED + HF-1 hygiene cycle** (`b5198ae` + `b20537e` + lob-model-trainer `21340e3`, 2026-05-15) — Closes Health Audit findings FIND-024 (silent-zero trading metrics) + #PY-71 (NaN-bypass strategy gates) + FIND-046 (sister of #PY-71) + Wave 1 Agent E hidden findings HF-3 (NaN-bypass in direction.py online mode) + cross-repo HF-1 (#PY-234 Path.resolve→Path.absolute in lob-model-trainer ledger_hook.py). **C2 FIND-024** (`b5198ae`): 6 silent-zero metric sites in `metrics/trading.py` (AvgWin/AvgLoss/PayoffRatio/Expectancy edge cases at L218/224/284/290/362/440) → `float("nan")` mirroring Phase X.3 WinRate precedent at L79. PRESERVED L373 magic 100.0 cap as documented sentinel (deferred migration to #PY-FIND-024-EXT per Agent J vs Agent L disagreement). **C3 #PY-71+FIND-046+HF-3** (`b20537e`): 14 NaN-comparison-bypass gates across 5 strategy files (regression.py L107+L111 + twap.py L121+L144+L147 + hybrid.py L120+L123+L127+L132 + readability.py L127+L130+L134+L141 + direction.py L229) — pre-fix `value <op> threshold` evaluated False on NaN input (IEEE 754 invariant) → garbage trades. NEW `tests/test_strategies/test_hybrid.py` adds full TestHybridNaNGuards class with 4 tests. TWAP sites included for class-coherence (Agent F mandate) despite Lesson #14 module-skip. **HF-1** (cross-repo lob-model-trainer `21340e3`): 5 sites L96/107/118/151/359 in `ledger_hook.py` migrated `.expanduser().resolve()` → `.expanduser().absolute()` for symlinked-deployment fingerprint preservation. Pre-Impl gate (3 parallel agents J/K/L) + Pre-Commit gate (3 parallel agents code-reviewer/doc-alignment/hft-architect) MANDATORY adversarial discipline applied per saved-feedback-memory. MF-1 same-cycle: doc-alignment-auditor found 13 docstring line cites drift L92/101/110/140/343 → updated to ground-truth L96/107/118/151/359. +20 NEW tests cumulative (+15 lob-backtester: 4 FIND-024 + 11 #PY-71+HF-3; +5 lob-model-trainer HF-1 TestHF1PathAbsoluteDiscipline). lob-backtester suite: 469 → 484 (zero regressions).
- **#PY-263 annualization closure across the 3 production backtest paths** (`a646187`, 2026-05-30; pushed `origin/main`, CI run `26677870705` green) — Completes the #PY-263 silent-Sharpe-inflation fix that V1 (`00638ac`, same-day) closed only on `run_regression_backtest.py`. **R1**: `run_readability_backtest.py` threads `bin_seconds=args.bin_seconds` into its `ZeroDteConfig` + observability print (mirrors V1's `run_regression_backtest.py:437-459`) → `resolved_periods_per_day` derives 23400/bin_seconds=390 at 60s, not the 1000.0 fallback (~1.6× = sqrt(1000/390) equity Sharpe/Sortino/Calmar inflation). **G1b**: `experiment.py::_build_backtest_config` passes `zero_dte=self._build_zero_dte_config()` so the ExperimentRunner path also derives correctly; the `config.py:571-584` mutex fail-louds if a YAML sets BOTH `periods_per_day` AND `zero_dte.bin_seconds` (now also fail-louds an *ambiguous* zero_dte block — more correct per §5). **G1a**: persist `resolved_periods_per_day`+`annualization_factor` (reuse the existing BacktestConfig properties — no duplicated 23400/bin_seconds math, §0) into the regression `<name>.json` + hft-ops ledger record + readability `config_dict` so saved runs self-describe which annualization scaled their Sharpe. **G1c**: ExperimentRunner module STATUS banner (§11: hft-ops shells to the scripts, not ExperimentRunner). **Verified invariant** (3 agents + grep): the engine reads `config.zero_dte` ONLY via `resolved_periods_per_day`/`annualization_factor`/`to_dict` — NEVER branches the equity result on `zero_dte.enabled` (the 0DTE transform is a separate post-hoc `ZeroDtePnLTransformer`) → annualization-only; equity-curve/trades/trade_pnls byte-identical. New G1a ledger keys break ZERO hft-ops readers (`ExperimentLedger` reads `ledger/records/` not `ledger/runs/`; 0 field-name hits in `hft-ops/src`). +5 value-locked tests (`TestPy263ExperimentRunnerAnnualization` derives-390/events-fallback/mutex + `TestPy263ReadabilityAnnualization` + `TestPy263AnnualizationPersisted` subprocess locks). Test count 619 → 624 (608 pass + 16 skip); ZERO regressions; zero new lint. Adversarial cascade: Wave-1 ×4 (re-validated each deferred to-do FROM SCRATCH) + Wave-2 ×3 + pre-impl + mid-impl + pre-commit ×3, all APPROVE. **3 candidate "issues" REFUTED from scratch** (don't-fix-non-existent guard): N1 `maker_rebate_bps` (taker-only engine + 0DTE discards `CostConfig` → wiring it would INTRODUCE a bug — leave inert), `run_spread_signal_backtest.py:245` (inert, `metrics=[]`), the original V3/V4 "non-production" framing. **DEFERRED NEXT-CYCLE BACKLOG** (also in the `a646187` commit footer + `POST_LOB_BACKTESTER_PY263_CLOSURE_2026_05_30.md`): **(D1)** `backtest_deeplob.py` is a 4th Sharpe-reporting path with the SAME #PY-263 gap but NON-SILENT (emits the DeprecationWarning), back-compat-default (`hft-ops manifest/schema.py:241-247`), no live manifest selects it — needs new `--bin-seconds` CLI infra (~1-1.5 hr). **(D2)** hft-ops orchestrator does not auto-thread `--bin-seconds` (operator `extra_args` today) — cross-repo (~1-2 hr). **(D3) #PY-NEW-V3** DirectionalAccuracy/SignalRate computed on trading signals (`engine/vectorized.py:675-679`) → metric-hygiene cycle; NOT a free default-list edit — 4 live consumers (`scripts/param_sweep.py:220` + `reports/summary.py:74,139` + `stats/stats.py:375`). **(D4) #PY-NEW-V4** `allow_short=True` dataclass default (`config.py:495`) → **DO NOT FLIP** — it is the load-bearing put-leg mechanism (a SELL on a flat book is skipped when `allow_short=False` at `vectorized.py:350-353` → the put leg vanishes from every documented R1-R8/E5/E6 result; `ZeroDtePnLTransformer` maps SELL→put via `zero_dte.py:373`); document-only. **(D5) #PY-NEW-N1** `CostConfig.maker_rebate_bps` intentionally-inert (taker-only engine; 0DTE discards `CostConfig`) — optional doc-note, NOT a fix. **(D6)** dead-code: N2 Expectancy default-drift (engine vs `BacktestStats`, "T2.2 OPEN") / N3 orphan `reports/` (~519 LOC, 0 callers) / N5 `ReadabilityHybridStrategy` orchestrator-orphan / N6 `ExitOnReverseStrategy` dead / N7 `_build_holding_policy` vs `create_holding_policy` divergence / N9 prediction-metric family → dead-code-sweep cycle. **Cosmetic** (non-blocking): G1b attaches `enabled=True` zero_dte to the metrics config in-memory (never persisted — the ExperimentRunner registry stores `_serialize_config` not `result.config_dict`). [Intervening cycles not logged here, documented in their commit footers: CI-hygiene `a341c40` 2026-05-29 + validation/V1/V2 `00638ac` 2026-05-30.]
- **G1a readability-ledger annualization symmetry fix** (pending commit, 2026-05-30) — Follow-up completing `a646187`'s G1a leg. A read-only adversarial re-validation of the full module (7 agents; the core money-math + the G1b annualization-only invariant + the signal-consumption contract all re-verified SOUND — ZERO production-corrupting bugs) surfaced ONE in-scope defect: the original G1a persisted `resolved_periods_per_day`/`annualization_factor` into the readability *registry* `config_dict` (nested) but NOT its hft-ops **ledger record** (`run_readability_backtest.py:480-491`), while the regression sister-record carries them top-level (`:653-654`) — so `compare_experiments` cross-run grouping silently mis-handled readability ledger rows, partially defeating G1a's own comparability goal. **Fix**: add the two keys top-level to the readability ledger record (`float()`-cast, reusing the `BacktestConfig` properties per §0 — symmetric with regression) + `TestPy263ReadabilityLedgerAnnualizationPersisted` value-locked test (PROVEN to fail without the fix via stash-revert). Test count 624 → 625 (609 pass + 16 skip); ZERO regressions; zero new lint (6 pre-existing ruff F541 unchanged). Validation also corrected two over-amplified prior-agent claims by ground-truth grep: `data/prices.py` denorm is OFF the production path (run_regression/readability consume the pre-exported `prices.npy`; denorm is used only by the test-only `DataLoader` + 2 fossils; its math at `prices.py:122` is the correct z-score inverse) — NOT "CRITICAL feeds all P&L"; `ConfusionMetrics` is dead code (zero callers) — NOT "HIGH untested". Remaining findings all LOW/non-production/cross-repo, left as-is: D3 DirectionalAccuracy/SignalRate-on-signals fires only in 2 fossil scripts' `metrics=None` path (never a persisted artifact); D1 `backtest_deeplob.py` is the hft-ops `BacktestingStage` default yet argparse-rejects the orchestrator flags (latent §5 fail-late trap; primary fix is cross-repo in hft-ops); Cap-3 sub-$100 silent undersizing + negative-equity distortion are NVDA-unreachable latents; the scripts-vs-`ExperimentRunner` two-run-path divergence is the standing years-horizon design item (mitigated by the §11 banner + tests; candidate for a shared config-builder SSoT). **Full per-finding report + test-coverage map + ranked design recommendations (R1-R7) + next-session options: `VALIDATION_FINDINGS_2026_05_30.md`.**
- **0DTE assembled-P&L golden lock + F3 reach correction** (pending commit, 2026-05-30) — A post-compaction fresh-eye 5-agent re-validation (4 anti-anchored bottom-up from code + 1 top-down auditing the prior conclusions) **re-confirmed the SOUND verdict** (zero production-corrupting bugs; sound producer→consumer boundary; no blocking incomplete work in lob-backtester's own code) and closed the one in-scope "rely-blindly" gap. **The gap**: the headline 0DTE Deep-ITM money-math *assembly* (`zero_dte.py:405-434` — `option_trade_pnls[i]`, the `move_bps` directional sign at `:388`, the BUY→call/SELL→put `is_call` mapping at `:373`, and `option_total_return`/`option_final_equity`) had NO value-lock — only its *components* (`theta_bsm_per_share`, `round_trip_cost_per_contract`) did → a sign-flip or assembly error would silently flip the reported return and pass the full suite. Not a bug (hand-verified correct today) — a regression-guard gap on the most important output. **Closed** by `TestZeroDteAssembledPnlGolden` (3 tests in `tests/test_engine/test_zero_dte.py`): a 2-leg fixture (BUY/call 100→101 + SELL/put 100→99, `events_per_minute=1.0`) value-locks `underlying_moves_bps=[+100,+100]` (directional sign for BOTH call and short/put legs), `is_call=[True,False]`, `spread_costs=[3.0,2.0]` (is_call→half-spread selection), `option_trade_pnls=[43.2766…,44.2766…]` (the full `gross − spread − comm − theta` assembly), and the `option_equity_curve`/`option_final_equity`/`option_total_return` aggregate. Expected values were independently re-derived AND verified bit-exact against the real `transform()` by a pre-impl gate; the lock was PROVEN to bite via a stash-revert mutation test (direction sign-flip → FAIL; dropped-theta → FAIL; restore → byte-identical, pass). Test count 625 → 628 (612 pass + 16 skip); ZERO regressions; zero new lint. The asymmetric fixture prices (exit ≠ entry) are load-bearing — do NOT flatten the round-trips. **Also corrected a stale claim**: the prior `VALIDATION_FINDINGS` F3 cell read "None today (all 9 manifests override `script:`)" — **wrong**. Ground-truth grep of `hft-ops/experiments/` shows **3 enabled full-pipeline manifests do NOT override `script:`** (`nvda_tlob_h10_v1.yaml`, `nvda_tlob_h100_tb_volscaled.yaml`, `nvda_hmhp_tb_volscaled.yaml`; each `stages.backtesting.enabled: true`) → they route to the `schema.py:277` default `backtest_deeplob.py`, which defines none of the orchestrator's `--signals/--name/--max-spread-bps/--output-dir/--manifest` flags → `SystemExit(2)`. Remains **latent** (no e2e backtest-run evidence; the 3 carry the legacy `model_checkpoint`/`data_dir`/`horizon_idx` deeplob interface) and **fail-loud, not corrupting**. **CROSS-REPO action item for the hft-ops-owning session** (out of lob-backtester scope — NO hft-ops file touched this cycle): change the `BacktestingStage.script` default away from `backtest_deeplob.py`, OR make `backtest_deeplob.py` accept/`parse_known_args` the orchestrator flags, OR set `script:` explicitly on those 3 manifests; the lob-backtester-side mitigation is the prior-logged R1/R2 (delete the deeplob fossil). Full detail: `VALIDATION_FINDINGS_2026_05_30.md` §9.
- **Phase 4 B1/B2/B3/B4 engine realism** (`e08fb89`, 2026-06-19) — greenfield realism layer. **B2**: NEW `engine/option_pricing.py` — the first BSM *valuation* code in the repo (`bs_value`/`bs_call`/`bs_put`/`bs_delta`/`bs_gamma`/`bs_vega`; q=0 / no-dividend, put floored at intrinsic; mirrors the `opra-statistical-profiler` `bsm.rs` SSoT). **B3/B4**: `ZeroDteConfig.payoff_model: Literal["linear_delta","bsm"]` + `moneyness` — under `payoff_model="bsm"` the transformer re-prices a real call OR put via `bs_value` (endogenous theta via tau-shrink over the hold; strike `= moneyness * entry_price`; static sigma), lifting the ATM-call-only `linear_delta` default; `prefer_calls=False` is now allowed under `"bsm"` (still raises under `"linear_delta"`). **B1**: `CostConfig.use_realized_spread: bool` — when True the engine swaps the flat `spread_bps` for the per-row realized half-spread from `data.spreads` (`VectorizedEngine._realized_spread_bps`, NaN/missing → flat fallback + WARN). Equity-path P&L is byte-identical when the flags are at defaults.
- **FIND-049 deep-ITM demo** (`6295d82`, 2026-06-19) — `demos/deep_itm_monthly_direction_null_demo.py` (a ~1-month intraday deep-ITM option = the direction null).
- **FINDING-046 maker-rebate honesty + deep-ITM cost caveat** (`f0da215` + `f8df0fc`, 2026-07-05, unpushed at time of writing) — `config.py` `CostConfig.maker_rebate_bps` carries a FINDING-046 warning (a RETAIL operator PAYS the fee and does NOT earn the exchange rebate; the negative `_EXCHANGE_PRESETS` values encode a non-retail-honest assumption — and the field is engine-INERT anyway: `total_bps`/`compute_cost` never read it). `OpraCalibratedCosts.deep_itm()` now documents the $0.005 deep-ITM half-spread → ~1.4 bps breakeven as an OPTIMISTIC, UNVALIDATED assumption (not derived from the fills). `BACKTEST_INDEX.md` frozen at the Phase-1 boundary.

## Architecture

- **Engine**: Per-sample loop in `engine/vectorized.py` (name is historical; actual algorithm is NOT vectorized). `engine/zero_dte.py` layers IBKR-calibrated 0DTE options P&L transformation on top of the equity backtest.
- **Strategy Pattern**: `Strategy` ABC in `strategies/base.py` + 7 concretes (direction, readability, regression, hybrid, holding policies, twap SKIP). `HoldingPolicy` ABC is composable via `CompositePolicy(mode="any"|"all")`.
- **Metrics ABC**: `metrics/base.py::Metric` + 4 groupings across returns/risk/trading/prediction.
- **Contract plane via hft_contracts**: `SignalManifest` canonical home at `hft_contracts.signal_manifest` (Phase 6 6B.5); `CONTENT_HASH_RE` regex imported from same SSoT (REV 2 public rename from `_CONTENT_HASH_RE`, 2026-04-20 — legacy name is a DeprecationWarning shim through 2026-10-31); `ContractError` imported from `hft_contracts.validation` (REV 2 F1 consolidation — was two independent classes); `atomic_write_json` imported from `hft_contracts.atomic_io` (REV 2 public rename from `_atomic_io`); label encoding defers to `hft_contracts.labels.LabelContract` for cross-module agreement.
- **IBKR/OPRA-calibrated 0DTE cost model**: commission validated from real NVDA option fills; ATM half-spreads from the OPRA CMBP-1 profiler; ATM breakevens 4.9 / 3.8 bps for Call / Put. **The Deep-ITM 1.4 bps breakeven is an OPTIMISTIC, UNVALIDATED assumption** — the $0.005 deep-ITM half-spread is not derived from the fills (real deep-ITM spreads are materially wider → treat as ~3–7 bps; see `config.py` `OpraCalibratedCosts.deep_itm()` warning + root FINDING-114 + the `BACKTEST_INDEX.md` freeze). Provenance in `engine/zero_dte.py` docstring + `IBKR-transactions-trades/COST_AUDIT_2026_03.md`.
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
hft-contracts = ">=2.7.0" # SignalManifest/label/canonical_hash/atomic_io SSoT (load-bearing)
numpy = ">=1.24.0"     # Core numerical operations
scipy = ">=1.10"       # scipy.stats.norm (BSM in scripts/run_spread_signal_backtest.py)
matplotlib = ">=3.7.0" # Visualization
pyyaml = ">=6.0"       # Configuration loading
```

---

## 2. Module Architecture

```
lob-backtester/
├── src/lobbacktest/
│   ├── __init__.py          # Public API exports
│   ├── version.py           # Version information
│   ├── types.py             # Core types: Trade, Position, BacktestResult
│   ├── config.py            # Configuration: BacktestConfig, CostConfig, ZeroDteConfig, OpraCalibratedCosts
│   ├── registry.py          # BacktestRegistry — append-only backtest-experiment/result registry (+ BacktestSummary)
│   │
│   ├── data/                # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── loader.py        # DataLoader: Load exported data
│   │   └── prices.py        # PriceExtractor: Denormalize prices
│   │
│   ├── strategies/          # Trading strategy implementations
│   │   ├── __init__.py      # Exports all strategies
│   │   ├── base.py          # Strategy ABC, Signal enum, SignalOutput
│   │   ├── direction.py     # DirectionStrategy, ThresholdStrategy
│   │   ├── readability.py   # ReadabilityStrategy (HMHP agreement + confidence gate)
│   │   ├── regression.py    # RegressionStrategy (magnitude gate for continuous predictions)
│   │   ├── hybrid.py        # ReadabilityHybridStrategy (classification direction + regression magnitude)
│   │   ├── holding.py       # HoldingPolicy, HorizonAlignedPolicy, HoldingState
│   │   └── twap.py          # TWAPStrategy (time-weighted execution)
│   │
│   ├── engine/              # Backtest execution
│   │   ├── __init__.py
│   │   ├── vectorized.py    # VectorizedEngine, Backtester, BacktestData
│   │   ├── zero_dte.py      # ZeroDtePnLTransformer, ZeroDteResult (0DTE options P&L; payoff_model linear_delta|bsm)
│   │   └── option_pricing.py # BSM valuation: bs_value/bs_call/bs_put/bs_delta/bs_gamma/bs_vega (q=0; put floored at intrinsic)
│   │
│   ├── metrics/             # Performance metrics
│   │   ├── __init__.py
│   │   ├── base.py          # Metric ABC, MetricResult
│   │   ├── returns.py       # TotalReturn, AnnualReturn
│   │   ├── risk.py          # SharpeRatio, SortinoRatio, MaxDrawdown, CalmarRatio
│   │   ├── trading.py       # WinRate, ProfitFactor, Expectancy, etc.
│   │   └── prediction.py    # DirectionalAccuracy, SignalRate, UpPrecision, DownPrecision
│   │
│   ├── stats/               # Statistics and aggregation
│   │   ├── __init__.py
│   │   └── stats.py         # BacktestStats fluent API
│   │
│   └── reports/             # Reporting and visualization
│       ├── __init__.py
│       ├── summary.py       # Text reports, comparison tables
│       └── plots.py         # Equity curves, drawdown charts
│
└── scripts/             # Runnable scripts (7 total)
    ├── backtest_deeplob.py                  # DeepLOB backtest runner
    ├── param_sweep.py                       # Parameter sweep (multi-config)
    ├── run_readability_backtest.py          # Readability strategy backtest
    ├── run_regression_backtest.py           # Regression strategy backtest
    ├── run_spread_signal_backtest.py        # Spread-based signal backtest (imports scipy.stats.norm)
    ├── e5_regime_filter_test.py             # E5 regime-filter diagnostic
    └── check_backtest_index_completeness.py # BACKTEST_INDEX completeness checker
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
    exchange: Optional[str] = None    # Exchange name for presets
    maker_rebate_bps: float = 0.0    # ENGINE-INERT (total_bps/compute_cost never read it);
                                     # the negative _EXCHANGE_PRESETS values are NON-retail-honest
                                     # per FINDING-046 (retail PAYS the fee, never earns the rebate)
    taker_fee_bps: float = 0.0
    use_realized_spread: bool = False # B1 (2026-06-19): replace flat spread_bps with the per-row
                                     # realized half-spread from data.spreads (NaN → flat + WARN)

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
    fill_price: Literal["close", "midpoint"] = "close"  # 'midpoint' is DEAD CODE — engine never reads it (emits DeprecationWarning, removal 2026-10-31)
    stop_loss_pct: Optional[float] = None   # DEAD CODE (FIND-058-EXT) — engine never reads it; emits DeprecationWarning
    take_profit_pct: Optional[float] = None # DEAD CODE (FIND-058-EXT) — engine never reads it; emits DeprecationWarning
    trading_days_per_year: float = 252.0
    periods_per_day: Optional[float] = None  # #PY-263: was 1000.0; None → resolved_periods_per_day derives from zero_dte.bin_seconds (23400/bin_seconds) else legacy 1000.0 w/ DeprecationWarning
    min_confidence: Optional[float] = None  # DEPRECATED 2026-10-31 — emits DeprecationWarning; use strategy.min_confidence
    min_agreement: Optional[float] = None   # DEPRECATED 2026-10-31 — emits DeprecationWarning; use strategy.min_agreement
```

### ZeroDteConfig and OpraCalibratedCosts

```python
@dataclass
class OpraCalibratedCosts:
    """IBKR/OPRA-calibrated 0DTE option cost model."""
    atm_call_half_spread: float = 0.015     # OPRA CMBP-1 profiler ($0.030 full -> $0.015 half)
    atm_put_half_spread: float = 0.010      # OPRA CMBP-1 profiler ($0.020 full -> $0.010 half)
    atm_call_premium: float = 1.88          # OPRA median ATM 0DTE call premium
    atm_put_premium: float = 1.31           # OPRA median ATM 0DTE put premium
    commission_per_contract: float = 0.70   # IBKR commission median (316-fill audit)
    implied_vol: float = 0.40
    entry_minutes_before_close: float = 120.0
    # deep_itm(*, implied_vol=0.25, ...) classmethod factory: half_spread=0.005 (OPTIMISTIC,
    # UNVALIDATED assumption — NOT from the fills; real deep-ITM spreads are materially wider).

@dataclass
class ZeroDteConfig:
    """0DTE option P&L transformation configuration."""
    enabled: bool = False
    delta: float = 0.50                     # Option delta (0.50 = ATM)
    opra_costs: OpraCalibratedCosts = field(default_factory=OpraCalibratedCosts)
    max_holding_minutes: float = 60.0
    contracts_per_trade: int = 1
    prefer_calls: bool = True               # False raises under linear_delta; allowed under 'bsm'
    payoff_model: Literal["linear_delta", "bsm"] = "linear_delta"  # B3/B4: 'bsm' = real BSM call/put re-pricing
    moneyness: float = 1.0                   # B3/B4: BSM strike = moneyness * entry_price (1.0 = ATM)
    # events_per_minute / bin_seconds (FIND-NEW-01, mutually exclusive): sampling cadence for
    # the transformer's holding-minutes derivation. target_holding_minutes / entry_window_*_et
    # are DEAD CODE (serialized but never read; emit DeprecationWarning on non-default).
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

> These validators fire, but `stop_loss_pct`, `take_profit_pct`, and `fill_price="midpoint"` are **DEAD CODE** (FIND-058 / FIND-058-EXT): the engine never reads them and each emits a `DeprecationWarning` (scheduled removal 2026-10-31). Use `StopLossTakeProfitPolicy` via the holding-policy path for SL/TP.

### ExperimentRunner YAML Schema (config-driven orchestration)

`ExperimentRunner.from_yaml(path)` (`experiment.py`) is the config-driven entry point: load a YAML → validate the signal directory → run one or more backtests (optionally a sweep) → register each to `BacktestRegistry`. It is **not** the production path — `hft-ops` shells out to the standalone `scripts/` (see §13 + the "run paths" note in `README.md`); `ExperimentRunner` is exercised by this repo's own tests. The blocks below map onto the §5 dataclasses; **per-field defaults + ranges live in those dataclass tables above** — this section documents block/key STRUCTURE + the type ENUMS only.

| Block | Load-bearing keys (consumed by `ExperimentRunner`) | Notes |
|---|---|---|
| `experiment` | `name` | Names the run + registry records (default `unnamed`). `description` is stored for provenance, not consumed. |
| `signals` | `dir` | Path to the trainer signal directory; loaded via `BacktestData.from_signal_dir(validate=True)`. **Required** — absent/empty crashes at load. |
| `backtest` | `initial_capital`, `position_size`, `allow_short`, `exchange`, `trading_days_per_year`, `periods_per_day`, nested `zero_dte` | Costs are derived from `exchange` via `CostConfig.for_exchange` — a `costs:` sub-block is accepted but **not** consumed on this path (only `BacktestConfig.from_dict`/`load_yaml` read it). `max_position`/`fill_price`/`stop_loss_pct`/`take_profit_pct`/`min_confidence`/`min_agreement` are accepted-without-warning but inert here (dead/deprecated per §5). |
| `strategy` | `type` (discriminator) + per-type keys | See enum below. |
| `holding` | `type` + `hold_events` (+ `stop_loss_bps`/`take_profit_bps` for the SL/TP policy) | See enum below. |
| `zero_dte` | `enabled`, `delta`, `contracts_per_trade`, exactly one of `bin_seconds`/`events_per_minute`, nested `opra_costs` | The overlay transform fires only on a **top-level** `zero_dte.enabled: true` (the `_run_single` activation gate reads the top-level block); a sampling cadence (`bin_seconds` **or** `events_per_minute`) is then **required** (fail-loud, FIND-NEW-01). The cost/annualization builder *additionally* accepts the block nested under `backtest:` (top-level **and** nested → `ValueError`, #PY-226), and `opra_costs` fields nested **or** at the `zero_dte` top-level (both → `ValueError`). Only the keys listed are threaded from YAML — `payoff_model`/`moneyness`/`prefer_calls`/`max_holding_minutes` take their `ZeroDteConfig` defaults on this path. |
| `sweep` | `{param_name: [values], …}` | One-parameter-at-a-time (see below). |
| `output` | `dir`, `save_equity_curve` | Registry output dir (default `outputs/backtests`); `save_equity_curve` gates equity-curve persistence. |

**`strategy.type` enum** (dispatch in `_build_strategy`; unknown → `ValueError` fail-loud):
- `regression` → `RegressionStrategy` — keys `min_return_bps`, `max_spread_bps`, `primary_horizon_idx`, `cooldown_events`.
- `readability` → `ReadabilityStrategy` — keys `min_agreement`, `min_confidence`, `max_spread_bps`. (FIND-070: these gate values MUST live under `strategy:`, not `backtest:` — the latter raises a precise migration `ValueError`.)
- `direction` → `DirectionStrategy` — key `shifted`.

**`holding.type` enum** (dispatch in `_build_holding_policy`; unknown → **silently** falls back to `horizon_aligned`, NOT fail-loud — unlike `strategy.type`):
- `horizon_aligned` → `HorizonAlignedPolicy(hold_events=…)`.
- `direction_reversal` → `DirectionReversalPolicy(max_hold_events=…)`.
- `stop_loss_take_profit` → `StopLossTakeProfitPolicy(max_hold_events=…, stop_loss_bps=…, take_profit_bps=…)`.

**`sweep` semantic** — one-parameter-at-a-time, **NOT** a Cartesian grid: each list-valued key is swept independently over its own values (non-list values skipped), holding the other `strategy` params at their base; total runs = the SUM of the list lengths, one `BacktestRegistry` record per `(param, value)`. `ExperimentResult.sweep_parameter` reports only the FIRST key — the design targets a single swept parameter. For a true multi-axis grid, use the hardcoded-grid `scripts/param_sweep.py` or an `hft-ops` sweep manifest (root `CLAUDE.md` §Running an Experiment).

**Diagnostics (FIND-070, hft-rules §8):** unknown keys under `backtest`/`strategy`/`holding` emit a `RuntimeWarning` (typo defence) but construction proceeds. **Worked examples:** `configs/e1_deep_itm.yaml` + `configs/e1_atm_comparison.yaml` are `ExperimentRunner`-shaped (they carry `signals.dir` + a `sweep`); `configs/nvda_readability_first_{xnas,arcx}.yaml` show the nested-block production shape but are **not** `ExperimentRunner`-runnable (they lack `signals.dir`).

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
# events_per_minute is a REQUIRED positional arg (FIND-NEW-01 — no silent default; 1.0 for 60s bins)
transformer = ZeroDtePnLTransformer(
    ZeroDteConfig(
        enabled=True, delta=0.50, contracts_per_trade=1,
        opra_costs=OpraCalibratedCosts(commission_per_contract=0.70, implied_vol=0.40),
    ),
    events_per_minute=1.0,
)
option_result = transformer.transform(backtest_result)
# option_result.option_total_return, option_result.option_win_rate,
# option_result.avg_spread_cost / avg_commission_cost / avg_theta_cost, etc.
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
| `periods_per_day` | `resolved_periods_per_day` (e.g. 390 at 60s bins; legacy fallback 1000) |

---

## 9. Stats API

### BacktestStats (Fluent API)

```python
stats = (
    BacktestStats(result)
        .with_book_size(100_000)
        .compute()
)

print(stats.summary())
stats.plot()
```

> **Note (2026-05-14, FIND-040 closure)**: ``.daily()`` / ``.monthly()`` raise
> ``NotImplementedError`` until ``BacktestResult`` exposes ``timestamps_ns``;
> ``.full()`` is preserved as a no-op self-return for fluent-API symmetry.
> See ``DESIGN_CLUSTER_D1_E_2026_05_14.md`` §4.1 +
> ``VALIDATION_FINDINGS_2026_05_14.md`` FIND-040.

### Methods

| Method | Description |
|--------|-------------|
| `.with_book_size(n)` | Set capital for normalization |
| `.daily()` | **NotImplementedError until `BacktestResult.timestamps_ns` lands** (FIND-040) |
| `.monthly()` | **NotImplementedError until `BacktestResult.timestamps_ns` lands** (FIND-040) |
| `.full()` | No-op self-return; ``.compute()`` returns full-corpus metrics regardless |
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
| `periods_per_day` | None (derived) | #PY-263: None → `resolved_periods_per_day` (23400/bin_seconds; e.g. 390 at 60s); legacy fallback 1000 emits DeprecationWarning |

---

*Last updated: 2026-06-19 (v0.1.0 — Phase 4 B1-B4 engine realism: option_pricing.py BSM valuation + payoff_model bsm + use_realized_spread; deep-ITM cost flagged optimistic/unvalidated 2026-07-05)*

> **Note**: `prices.py` imports feature indices from `hft-contracts`.
> `NormalizationParams.from_json()` validates the `normalization_applied` boundary
> contract. `DataLoader.load()` enforces `validate_export_contract()` at load time,
> requires labels for all days (no silent zero-substitution), and validates
> sequence-label shape alignment via `ContractError`.

