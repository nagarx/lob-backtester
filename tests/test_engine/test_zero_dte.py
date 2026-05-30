"""
Tests for 0DTE ATM Options P&L Transformer.

Tests verify:
- BSM theta formula: theta = S * sigma * N'(0) / (2 * sqrt(T))
- Per-trade P&L model: gross - spread - commission - theta
- Cost breakdown accuracy
- Edge cases: zero holding, short time remaining

Per RULE.md:
- Formula tests: BSM theta verified with hand-calculated values
- Reference: Black & Scholes (1973), Hull (2018) Ch 19
- Source calibration: OPRA CMBP-1 (8 days), IBKR 318 fills
"""

import math

import numpy as np
import pytest

from lobbacktest.engine.zero_dte import (
    EPS,
    NPRIME_ZERO,
    TRADING_MINUTES_PER_YEAR,
    ZeroDteAlternationError,
    ZeroDtePnLTransformer,
    ZeroDteResult,
    theta_bsm_per_share,
)
from lobbacktest.config import BacktestConfig, CostConfig, ZeroDteConfig, OpraCalibratedCosts
from lobbacktest.types import BacktestResult, Trade, TradeSide


class TestThetaBsmFormula:
    """
    Verify BSM theta formula:
        theta_annual = S * sigma * N'(0) / (2 * sqrt(T))
        theta_per_min = theta_annual / (252 * 390)
        theta_cost = theta_per_min * holding_minutes

    N'(0) = 1 / sqrt(2*pi) ≈ 0.3989
    """

    def test_theta_formula_at_14_00(self):
        """
        At 14:00 ET (120 min remaining), S=$180, IV=40%:
            T = 120 / (252 * 390) = 0.001221
            theta_annual = 180 * 0.40 * 0.3989 / (2 * sqrt(0.001221))
            theta_annual = 180 * 0.40 * 0.3989 / (2 * 0.03494)
            theta_annual = 28.72 / 0.06989 = 410.9
            theta_per_min = 410.9 / 98280 = 0.004181
            theta_1min = 0.004181 USD/share

        Per-contract (100 shares) for 1 min: $0.418
        Documented value: $0.42/contract/min (CODEBASE.md, zero_dte.py docstring)
        """
        theta = theta_bsm_per_share(
            underlying_price=180.0,
            implied_vol=0.40,
            minutes_remaining=120.0,
            holding_minutes=1.0,
        )
        # Expected: ~0.00418 USD/share/min
        theta_per_contract = theta * 100
        assert 0.35 < theta_per_contract < 0.50, (
            f"BSM theta at 14:00 should be ~$0.42/contract/min, got ${theta_per_contract:.4f}"
        )

    def test_theta_formula_at_15_30(self):
        """
        At 15:30 ET (30 min remaining), theta should be ~2x higher than 14:00.
        Theta ∝ 1/sqrt(T), so T=30 vs T=120 → sqrt(120/30) = 2x.
        Documented: 0.47 bps/min at 15:30.
        """
        theta_14 = theta_bsm_per_share(180.0, 0.40, 120.0, 1.0)
        theta_15 = theta_bsm_per_share(180.0, 0.40, 30.0, 1.0)
        ratio = theta_15 / theta_14
        assert 1.8 < ratio < 2.2, f"Theta at 15:30 should be ~2x 14:00, got {ratio:.2f}x"

    def test_theta_scales_linearly_with_holding(self):
        """Theta cost should scale linearly with holding duration (for short holds)."""
        theta_1 = theta_bsm_per_share(180.0, 0.40, 120.0, 1.0)
        theta_5 = theta_bsm_per_share(180.0, 0.40, 120.0, 5.0)
        assert abs(theta_5 / theta_1 - 5.0) < 0.01, "Theta should scale linearly with holding"

    def test_theta_scales_with_stock_price(self):
        """Theta ∝ S: doubling stock price doubles theta."""
        theta_100 = theta_bsm_per_share(100.0, 0.40, 120.0, 1.0)
        theta_200 = theta_bsm_per_share(200.0, 0.40, 120.0, 1.0)
        assert abs(theta_200 / theta_100 - 2.0) < 0.01

    def test_theta_scales_with_volatility(self):
        """Theta ∝ sigma: doubling IV doubles theta."""
        theta_20 = theta_bsm_per_share(180.0, 0.20, 120.0, 1.0)
        theta_40 = theta_bsm_per_share(180.0, 0.40, 120.0, 1.0)
        assert abs(theta_40 / theta_20 - 2.0) < 0.01

    def test_theta_zero_with_zero_holding(self):
        """Zero holding time → zero theta cost."""
        theta = theta_bsm_per_share(180.0, 0.40, 120.0, 0.0)
        assert theta == 0.0

    def test_theta_zero_with_no_time_remaining(self):
        """< 1 minute remaining → zero theta (expired)."""
        theta = theta_bsm_per_share(180.0, 0.40, 0.5, 1.0)
        assert theta == 0.0


class TestThetaConstants:
    """Verify the BSM constants used in the module."""

    def test_nprime_zero(self):
        """N'(0) = 1 / sqrt(2*pi) ≈ 0.39894."""
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert abs(NPRIME_ZERO - expected) < 1e-10

    def test_trading_minutes_per_year(self):
        """252 trading days * 390 minutes/day = 98,280."""
        assert TRADING_MINUTES_PER_YEAR == 252.0 * 390.0

    def test_eps_is_small(self):
        """EPS should be a very small positive number for numerical stability."""
        assert EPS > 0
        assert EPS < 1e-6


class TestOpraCalibratedCosts:
    """Verify OPRA-calibrated cost model defaults against empirical data."""

    def test_default_call_spread(self):
        """Call half-spread: $0.015 (OPRA median $0.030 full ÷ 2)."""
        costs = OpraCalibratedCosts()
        assert costs.atm_call_half_spread == 0.015

    def test_default_put_spread(self):
        """Put half-spread: $0.010 (OPRA median $0.020 full ÷ 2)."""
        costs = OpraCalibratedCosts()
        assert costs.atm_put_half_spread == 0.010

    def test_default_call_premium(self):
        """Call premium: $1.88 (OPRA median, validated by IBKR $1.86)."""
        costs = OpraCalibratedCosts()
        assert costs.atm_call_premium == 1.88

    def test_default_commission(self):
        """Commission: $0.70 (IBKR 318-fill median, all-inclusive)."""
        costs = OpraCalibratedCosts()
        assert costs.commission_per_contract == 0.70

    def test_round_trip_cost_call(self):
        """
        Round-trip cost for ATM 0DTE call:
            spread = 2 * 0.015 * 100 = $3.00
            commission = 2 * 0.70 = $1.40
            total = $4.40

        Reference: IBKR_REAL_WORLD_TRADING_REPORT.md §6, §7
        """
        costs = OpraCalibratedCosts()
        rt = costs.round_trip_cost_per_contract(is_call=True)
        expected = 2 * 0.015 * 100 + 2 * 0.70  # 3.00 + 1.40 = 4.40
        assert abs(rt - expected) < 0.01, f"Call RT cost should be ${expected:.2f}, got ${rt:.2f}"

    def test_round_trip_cost_put(self):
        """
        Round-trip cost for ATM 0DTE put:
            spread = 2 * 0.010 * 100 = $2.00
            commission = 2 * 0.70 = $1.40
            total = $3.40
        """
        costs = OpraCalibratedCosts()
        rt = costs.round_trip_cost_per_contract(is_call=False)
        expected = 2 * 0.010 * 100 + 2 * 0.70  # 2.00 + 1.40 = 3.40
        assert abs(rt - expected) < 0.01, f"Put RT cost should be ${expected:.2f}, got ${rt:.2f}"


class TestDeepItmFactoryPy273Py274:
    """#PY-273 + #PY-274 closure (2026-05-16) — Deep ITM IV-skew + CLI pass-through.

    Pre-fix `deep_itm()` was a no-args classmethod that hardcoded:
      * `implied_vol=0.40` (inherited ATM IV; #PY-273 root cause)
      * `entry_minutes_before_close=120.0` (hardcoded; #PY-274 ignored CLI flag)

    Post-fix `deep_itm(*, implied_vol=0.25, entry_minutes_before_close=120.0)`:
      * Default `implied_vol=0.25` reflects OPRA empirical Deep ITM IV-skew
        (~0.20-0.30 vs ATM's 0.40). Closes #PY-273 60-100% theta overestimation
        without strike-K plumbing (which would require Phase Z architectural cycle
        per #PY-271 — Deep ITM N'(d1) ≈ 0 vs ATM N'(0)=0.3989; full d1-correction
        is deferred).
      * Keyword-only parameters allow CLI flag pass-through:
        `OpraCalibratedCosts.deep_itm(implied_vol=args.implied_vol, ...)`
        from `run_regression_backtest.py` + `run_readability_backtest.py`.
        Closes #PY-274.

    Theta is linear in σ per BSM ATM formula at zero_dte.py:77 — IV change from
    0.40 → 0.25 reduces Deep ITM theta to 0.625x of pre-fix value (locked by
    `test_deep_itm_theta_proportional_to_atm_via_iv_ratio` below).
    """

    def test_deep_itm_default_iv_is_025(self):
        """#PY-273 closure: factory default `implied_vol=0.25` (not 0.40)."""
        costs = OpraCalibratedCosts.deep_itm()
        assert costs.implied_vol == 0.25, (
            f"Deep ITM IV should default to 0.25 post-#PY-273 closure (was 0.40 "
            f"pre-fix; ATM IV inheritance); got {costs.implied_vol}"
        )

    def test_deep_itm_iv_kwarg_override(self):
        """#PY-274: kwarg propagates through factory for CLI sensitivity sweeps."""
        costs = OpraCalibratedCosts.deep_itm(implied_vol=0.20)
        assert costs.implied_vol == 0.20

    def test_deep_itm_entry_minutes_kwarg_override(self):
        """#PY-274: entry_minutes_before_close kwarg propagates for CLI override."""
        costs = OpraCalibratedCosts.deep_itm(entry_minutes_before_close=60.0)
        assert costs.entry_minutes_before_close == 60.0

    def test_deep_itm_default_entry_minutes_is_120(self):
        """#PY-274 back-compat: factory default `entry_minutes_before_close=120.0`."""
        costs = OpraCalibratedCosts.deep_itm()
        assert costs.entry_minutes_before_close == 120.0, (
            "Default `entry_minutes_before_close` should remain 120 min "
            "(14:00 ET entry) for back-compat with pre-#PY-274 callsites."
        )

    def test_deep_itm_default_spreads_unchanged(self):
        """#PY-273 back-compat: spreads + premiums + commission unchanged post-fix.

        Only `implied_vol` (and now keyword-only `entry_minutes_before_close`)
        changed. The factory's other constants (tight Deep ITM spreads, $20
        premium, $0.70 commission) are preserved bit-exact.
        """
        costs = OpraCalibratedCosts.deep_itm()
        assert costs.atm_call_half_spread == 0.005
        assert costs.atm_put_half_spread == 0.005
        assert costs.atm_call_premium == 20.0
        assert costs.atm_put_premium == 20.0
        assert costs.commission_per_contract == 0.70

    def test_deep_itm_theta_proportional_to_atm_via_iv_ratio(self):
        """#PY-273: theta(deep_itm IV=0.25) ≈ 0.625 × theta(atm IV=0.40).

        BSM ATM theta is linear in σ per `zero_dte.py:77`:
            theta = -S · σ · N'(0) / (2 · √T)

        Therefore theta_deep_itm / theta_atm = 0.25 / 0.40 = 0.625.

        Locks the empirical 37.5% reduction in reported Deep ITM theta cost
        post-#PY-273 closure (a 37.5% reduction equals the lower-end of the
        backlog's "60-100% overestimation" range when interpreted as a
        fractional bias on the *reported* number).
        """
        deep_itm_iv = OpraCalibratedCosts.deep_itm().implied_vol
        atm_iv = OpraCalibratedCosts().implied_vol  # default 0.40

        # Compute theta for both at identical S, T, hold
        theta_deep_itm = theta_bsm_per_share(
            underlying_price=180.0,
            implied_vol=deep_itm_iv,
            minutes_remaining=120.0,
            holding_minutes=1.0,
        )
        theta_atm = theta_bsm_per_share(
            underlying_price=180.0,
            implied_vol=atm_iv,
            minutes_remaining=120.0,
            holding_minutes=1.0,
        )

        ratio = theta_deep_itm / theta_atm
        expected_ratio = deep_itm_iv / atm_iv  # 0.25 / 0.40 = 0.625

        assert abs(ratio - expected_ratio) < 1e-6, (
            f"Deep ITM theta should be {expected_ratio:.4f}x ATM theta per BSM "
            f"σ-linearity, got ratio={ratio:.4f}"
        )

    def test_deep_itm_iv_invalid_raises_via_post_init(self):
        """#PY-273: invalid `implied_vol` raises via dataclass `__post_init__`.

        Closes silent-failure class: pre-#PY-273 the factory hardcoded 0.40
        making this validation path unreachable. Post-fix kwarg can be
        operator-supplied via CLI — invalid values fail-loud per hft-rules §5.
        """
        with pytest.raises(ValueError, match="implied_vol must be > 0"):
            OpraCalibratedCosts.deep_itm(implied_vol=0.0)
        with pytest.raises(ValueError, match="implied_vol must be > 0"):
            OpraCalibratedCosts.deep_itm(implied_vol=-0.10)


class TestZeroDteConfig:
    """Verify ZeroDte configuration defaults."""

    def test_default_delta(self):
        """ATM options have delta ≈ 0.50."""
        config = ZeroDteConfig()
        assert config.delta == 0.50

    def test_default_contracts(self):
        """Default: 1 contract per trade."""
        config = ZeroDteConfig()
        assert config.contracts_per_trade == 1


class TestAlternationContract:
    """FIND-003 lock tests: ZeroDtePnLTransformer raises ZeroDteAlternationError on contract breach.

    Post-FIND-001 fix, engine emits Trade(FLAT) symmetrically with trade_pnls.append, so the
    precondition `len(trades) == 2 * n_round_trips` is structurally guaranteed in production.
    The precondition raise + per-pair side assert are reserved for regression detection +
    external Trade-stream consumers.

    See DESIGN_CLUSTER_D1_E_2026_05_14.md §3.3 + VALIDATION_FINDINGS_2026_05_14.md FIND-003.
    """

    def _make_result(
        self, trades: list, trade_pnls: np.ndarray, n: int = 15
    ) -> BacktestResult:
        """Build a BacktestResult fixture satisfying FIND-002 invariant but stressing FIND-003."""
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

    def test_alternation_orphan_trade_raises_via_precondition(self):
        """FIND-003 lock: precondition raises when len(trades) != 2 * n_round_trips.

        Fixture: 3 trades (BUY, FLAT, BUY) + 1 trade_pnl.
        FIND-002 invariant: n_closes(FLAT) = 1 == len(trade_pnls)=1 → passes construction.
        FIND-003 precondition: 2 * n_round_trips = 2 != len(trades) = 3 → raises.
        """
        trades = [
            Trade(index=0, side=TradeSide.BUY, price=10.0, size=10, cost=0.1),
            Trade(index=5, side=TradeSide.FLAT, price=10.5, size=10, cost=0.1),
            Trade(index=10, side=TradeSide.BUY, price=10.2, size=10, cost=0.1),  # ORPHAN
        ]
        result = self._make_result(trades, trade_pnls=np.array([4.8]))

        transformer = ZeroDtePnLTransformer(
            config=ZeroDteConfig(), events_per_minute=10.0
        )
        with pytest.raises(ZeroDteAlternationError, match="2 trades per round-trip"):
            transformer.transform(result)

    def test_alternation_per_pair_violation_raises(self):
        """FIND-003 lock: per-pair side assert catches reordered trades.

        Fixture: 4 trades + 2 trade_pnls (FIND-002 + precondition pass) but reordered as
        (BUY, BUY, FLAT, FLAT) instead of (BUY, FLAT, BUY, FLAT).
        """
        trades = [
            Trade(index=0, side=TradeSide.BUY, price=10.0, size=10, cost=0.1),
            Trade(index=1, side=TradeSide.BUY, price=10.1, size=10, cost=0.1),  # WRONG — should be FLAT
            Trade(index=5, side=TradeSide.FLAT, price=10.5, size=10, cost=0.1),
            Trade(index=6, side=TradeSide.FLAT, price=10.6, size=10, cost=0.1),
        ]
        result = self._make_result(trades, trade_pnls=np.array([4.8, 5.0]))

        transformer = ZeroDtePnLTransformer(
            config=ZeroDteConfig(), events_per_minute=10.0
        )
        with pytest.raises(ZeroDteAlternationError, match="alternation violated"):
            transformer.transform(result)


# ---------------------------------------------------------------------------
# FIND-NEW-01 closure (2026-05-16) regression lock
# ---------------------------------------------------------------------------


class TestSamplingCadenceRegression:
    """FIND-NEW-01 closure (2026-05-16) — events_per_minute is REQUIRED.

    Pre-FIND-NEW-01, ``ZeroDtePnLTransformer.__init__`` defaulted to
    ``events_per_minute=10.0`` (calibrated for event-based ~1000 events/day
    corpora). When TB v3p0 (60s time-based, 1.0 events/min) backtests
    inherited this default, ``holding_minutes = events / events_per_minute``
    was silently 10x smaller than reality:

      * 30-event hold reported as 3 min (= 30 / 10) but actual = 30 min (= 30 / 1)
      * BSM theta cost reported as ~$1.27/trade (linear in holding_minutes)
        but actual ≈ $12.64/trade
      * R-17a / R-19 / R-16d / R-16e absolute cost-economics analyses were
        all biased toward over-optimistic per-trade economics
      * Comparative ranking (cross-arm within a sweep) was preserved
        because the bias applied uniformly

    Verified empirically via:
      * ``grep events_per_minute lob-backtester/src/lobbacktest/engine/zero_dte.py``
        → confirms default removed
      * ``lob-backtester/scripts/run_spread_signal_backtest.py:549`` already
        passed ``events_per_minute=1.0`` explicitly — author was aware of
        the bug for ONE script but the fix didn't propagate to the others
      * ``lob-backtester/BACKTEST_INDEX.md`` R-17a/R-19 entries literally
        report "30.0 events | 3.0 min hold" + theta $1.27 → math is
        30/10.0 = 3.0 (buggy) vs 30/1.0 = 30.0 (correct)

    See ``lob-backtester/VALIDATION_FINDINGS_2026_05_14.md`` FIND-NEW-01 +
    monorepo-root ``CLAUDE.md`` 2026-05-16 banner for full closure narrative.
    """

    def _make_result_for_30_event_hold(self) -> BacktestResult:
        """Synthetic 2-trade fixture: entry at t=0, exit at t=30.

        Per BacktestResult invariant ``final_equity == equity_curve[-1]``,
        the fixture keeps the equity flat — the test asserts on
        ``holding_periods_minutes`` and ``theta_costs`` (functions of
        ``index_delta`` and ``events_per_minute``), NOT on equity PnL.
        """
        trades = [
            Trade(index=0, side=TradeSide.BUY, price=180.0, size=1, cost=0.0),
            Trade(index=30, side=TradeSide.FLAT, price=181.0, size=1, cost=0.0),
        ]
        n = 31
        return BacktestResult(
            equity_curve=np.array([100.0] * n),
            returns=np.zeros(n - 1),
            positions=np.zeros(n),
            prices=np.array([180.0] * n),
            predictions=np.zeros(n),
            labels=None,
            trades=trades,
            trade_pnls=np.array([0.0]),
            metrics={},
            config_dict={},
            initial_capital=100.0,
            final_equity=100.0,
            total_trades=2,
            start_index=0,
            end_index=n - 1,
        )

    def test_events_per_minute_required_no_default(self):
        """events_per_minute has no default; calling without it raises TypeError.

        Pre-FIND-NEW-01 silently used ``events_per_minute=10.0``; post-fix
        the absent positional/kwarg raises at construction time per
        hft-rules §5 fail-fast.
        """
        with pytest.raises(TypeError, match="events_per_minute"):
            ZeroDtePnLTransformer(ZeroDteConfig())  # type: ignore[call-arg]

    def test_events_per_minute_zero_raises_with_findnew01_context(self):
        """events_per_minute=0 raises ValueError citing FIND-NEW-01 + actionable
        migration hint per hft-rules §5."""
        with pytest.raises(ValueError, match="FIND-NEW-01"):
            ZeroDtePnLTransformer(ZeroDteConfig(), events_per_minute=0.0)

    def test_events_per_minute_negative_raises(self):
        """Negative events_per_minute raises ValueError (defensive — same
        guard as zero)."""
        with pytest.raises(ValueError, match="events_per_minute must be > 0"):
            ZeroDtePnLTransformer(ZeroDteConfig(), events_per_minute=-1.0)

    def test_30_event_hold_yields_30_min_at_60s_cadence(self):
        """TB v3p0 60s bins → events_per_minute=1.0 → 30-event hold = 30 min.

        Pre-FIND-NEW-01: 30 / 10.0 = 3 min (silent 10x bias). Post-fix:
        30 / 1.0 = 30 min (matches wall-clock).
        """
        result = self._make_result_for_30_event_hold()
        config = ZeroDteConfig(
            enabled=True,
            delta=0.95,
            opra_costs=OpraCalibratedCosts.deep_itm(),
            max_holding_minutes=120.0,  # don't clip — verify true value
        )
        transformer = ZeroDtePnLTransformer(config, events_per_minute=1.0)
        out = transformer.transform(result)
        assert out.holding_periods_minutes[0] == 30.0, (
            f"30-event hold at events_per_minute=1.0 (TB v3p0 60s) should "
            f"yield 30 min, got {out.holding_periods_minutes[0]}. "
            f"Pre-FIND-NEW-01 silent default events_per_minute=10.0 gave "
            f"3 min."
        )

    def test_30_event_hold_yields_3_min_at_legacy_cadence(self):
        """Legacy event-based ~1000/day → events_per_minute=10.0 → 30-event
        hold = 3 min. Verifies the legacy calibration path still works when
        operator supplies events_per_minute=10.0 explicitly."""
        result = self._make_result_for_30_event_hold()
        config = ZeroDteConfig(
            enabled=True,
            delta=0.95,
            opra_costs=OpraCalibratedCosts.deep_itm(),
            max_holding_minutes=120.0,
        )
        transformer = ZeroDtePnLTransformer(config, events_per_minute=10.0)
        out = transformer.transform(result)
        assert out.holding_periods_minutes[0] == 3.0, (
            f"30-event hold at events_per_minute=10.0 (legacy event-based "
            f"~1000/day) should yield 3 min, got {out.holding_periods_minutes[0]}"
        )

    def test_theta_cost_on_tb_v3p0_60s_matches_atm_reference(self):
        """Theta cost on TB v3p0 60s (events_per_minute=1.0) at 30-event hold
        should be ~10x larger than legacy event-based (events_per_minute=10.0).

        Reference: BSM theta is linear in holding_minutes for short holds
        relative to T (see zero_dte.py theta_bsm_per_share). So
        theta(30 min) ≈ 10 × theta(3 min). At 14:00 ET (120 min remaining),
        $180 NVDA, 40% IV, ATM (half_spread=$0.015): theta @ 1 min ≈ $0.42
        per contract → theta @ 30 min ≈ $12.60 (vs theta @ 3 min ≈ $1.26).
        """
        result = self._make_result_for_30_event_hold()
        config = ZeroDteConfig(
            enabled=True,
            delta=0.50,
            opra_costs=OpraCalibratedCosts(
                atm_call_half_spread=0.015,
                atm_put_half_spread=0.010,
                commission_per_contract=0.70,
                implied_vol=0.40,
                entry_minutes_before_close=120.0,
            ),
            max_holding_minutes=120.0,
        )
        # 60s cadence (post-fix)
        theta_60s = ZeroDtePnLTransformer(
            config, events_per_minute=1.0
        ).transform(result).theta_costs[0]
        # Legacy event-based cadence (pre-fix silent default)
        theta_legacy = ZeroDtePnLTransformer(
            config, events_per_minute=10.0
        ).transform(result).theta_costs[0]
        # Post-fix should be ~10x the pre-fix value (linear in holding_minutes)
        ratio = theta_60s / theta_legacy if theta_legacy > 0 else 0
        assert 9.5 < ratio < 10.5, (
            f"theta(60s, 30min hold) / theta(legacy, 3min hold) should be ~10 "
            f"(BSM theta linear in holding_minutes), got {ratio:.2f}"
        )
        # Magnitude check: theta_60s should be in [$8, $16] range per BSM
        # reference at 14:00 ET, $180 NVDA, 40% IV.
        assert 8.0 < theta_60s < 16.0, (
            f"Theta on TB v3p0 60s with 30-event hold (= 30 min wall-clock) "
            f"should be ~$8-16/contract (BSM reference: $0.42/min × 30 ≈ "
            f"$12.60), got ${theta_60s:.2f}. Pre-FIND-NEW-01 reported ~$1.27."
        )

    def test_zero_dte_config_mutex_validation(self):
        """ZeroDteConfig.__post_init__ raises when both events_per_minute and
        bin_seconds are set (mutually exclusive per hft-rules §5)."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            ZeroDteConfig(events_per_minute=1.0, bin_seconds=60.0)

    def test_zero_dte_config_resolved_events_per_minute_from_bin_seconds(self):
        """ZeroDteConfig.resolved_events_per_minute derives correctly from
        bin_seconds (= 60.0 / bin_seconds)."""
        cfg = ZeroDteConfig(bin_seconds=60.0)
        assert cfg.resolved_events_per_minute == 1.0
        cfg = ZeroDteConfig(bin_seconds=5.0)
        assert cfg.resolved_events_per_minute == 12.0

    def test_zero_dte_config_resolved_events_per_minute_explicit_wins(self):
        """When events_per_minute is set explicitly, the property returns it
        directly (bin_seconds derivation not used)."""
        cfg = ZeroDteConfig(events_per_minute=2.5)
        assert cfg.resolved_events_per_minute == 2.5

    def test_zero_dte_config_resolved_events_per_minute_none_when_unset(self):
        """When neither field is set, the property returns None (caller must
        supply at transformer-construction time)."""
        cfg = ZeroDteConfig()
        assert cfg.resolved_events_per_minute is None


# ---------------------------------------------------------------------------
# Assembled 0DTE option-P&L golden lock (2026-05-30 post-compaction re-validation)
# ---------------------------------------------------------------------------


class TestZeroDteAssembledPnlGolden:
    """Value-lock the ASSEMBLED 0DTE option P&L — the headline Deep-ITM money-math.

    Gap closed (surfaced by the 2026-05-30 re-validation, `VALIDATION_FINDINGS_2026_05_30.md`
    §9): prior to this class, the *components* of the 0DTE transform were golden-locked
    (``theta_bsm_per_share`` by ``TestThetaBsmFormula``; ``round_trip_cost_per_contract``
    by ``TestOpraCalibratedCosts``) but the **assembly** at ``zero_dte.py:405-434`` was NOT:
    ``option_trade_pnls[i]``, the directional sign in ``underlying_moves_bps[i]``, the
    ``is_call[i]`` BUY→call / SELL→put mapping, and the ``option_total_return`` /
    ``option_final_equity`` aggregate had zero value assertions. A ``direction`` sign-flip
    (``zero_dte.py:388``), an ``is_call`` ternary inversion (``:373``), a dropped/added cost
    term, or ``gross`` using ``exit_price`` instead of ``entry_price`` (``:405``) would have
    silently flipped the reported return and passed the entire suite.

    Fixture: 2 round-trips, ``prefer_calls=True`` (production default; ``False`` raises per
    ``config.py:337``), ``events_per_minute=1.0`` (TB v3p0 60s → 10-event hold = 10 min):
      * leg 0 — BUY (call), 100.0 → 101.0 (+1% up move → +100 bps profit)
      * leg 1 — SELL (put), 100.0 → 99.0 (-1% down move → +100 bps profit; shorts profit on a drop)

    The asymmetric prices (exit != entry) are LOAD-BEARING: they make a "gross uses
    exit_price instead of entry_price" mutation detectable. DO NOT flatten the round-trips
    to exit == entry (pre-impl gate caveat 2026-05-30).

    All expected values were independently re-derived AND verified bit-exact against the
    real ``transform()`` output by the pre-impl adversarial gate (2026-05-30).
    """

    # BSM theta per contract for the fixture's 10-min hold @ S=$100, IV=0.40,
    # 120 min to close: theta_bsm_per_share(100.0, 0.40, 120.0, 10.0) * 100.
    # Gate-verified golden literal; the theta FORMULA itself is locked separately
    # by TestThetaBsmFormula (this constant locks that transform() plumbs the
    # right args — entry_price, IV, entry-minutes, holding — and scales by x100).
    _EXPECTED_THETA_PER_CONTRACT = 2.3233619070671425

    def _make_two_leg_result(self) -> BacktestResult:
        """BUY/call (100→101) + SELL/put (100→99); asymmetric prices are load-bearing."""
        trades = [
            Trade(index=0, side=TradeSide.BUY, price=100.0, size=1, cost=0.0),
            Trade(index=10, side=TradeSide.FLAT, price=101.0, size=1, cost=0.0),
            Trade(index=20, side=TradeSide.SELL, price=100.0, size=1, cost=0.0),
            Trade(index=30, side=TradeSide.FLAT, price=99.0, size=1, cost=0.0),
        ]
        n = 31
        return BacktestResult(
            equity_curve=np.array([100000.0] * n),
            returns=np.zeros(n - 1),
            positions=np.zeros(n),
            prices=np.array([100.0] * n),
            predictions=np.zeros(n),
            labels=None,
            trades=trades,
            trade_pnls=np.array([0.0, 0.0]),  # values unused by transform; len=2 = n_round_trips
            metrics={},
            config_dict={},
            initial_capital=100000.0,
            final_equity=100000.0,
            total_trades=4,
            start_index=0,
            end_index=n - 1,
        )

    def _make_transformer(self) -> ZeroDtePnLTransformer:
        config = ZeroDteConfig(
            enabled=True,
            delta=0.50,
            contracts_per_trade=1,
            opra_costs=OpraCalibratedCosts(
                atm_call_half_spread=0.015,
                atm_put_half_spread=0.010,
                commission_per_contract=0.70,
                implied_vol=0.40,
                entry_minutes_before_close=120.0,
            ),
            # max_holding_minutes=60 (default, no clip at 10 min);
            # target_holding_minutes=15.0 + entry_window defaults → zero DeprecationWarning.
        )
        return ZeroDtePnLTransformer(config, events_per_minute=1.0)

    def test_move_bps_sign_and_is_call_mapping_value_locked(self):
        """Lock the directional sign (move_bps), the BUY→call / SELL→put mapping
        (is_call), and the is_call→half_spread selection.

        A ``direction`` sign-flip (zero_dte.py:388) flips both move_bps to -100;
        an ``is_call`` ternary inversion (:373) swaps is_call AND the selected
        half-spread (call $0.015 ↔ put $0.010). Both are the load-bearing 0DTE
        semantics with no prior value-lock.
        """
        out = self._make_transformer().transform(self._make_two_leg_result())

        # Leg 0 — BUY → call, +1% move → +100 bps profit; call half_spread 0.015.
        assert bool(out.is_call[0]) is True, "BUY entry must map to a call (prefer_calls=True)"
        assert out.underlying_moves_bps[0] == pytest.approx(100.0), (
            "BUY 100→101 is a +100 bps move; a direction sign-flip would give -100"
        )
        assert out.spread_costs[0] == pytest.approx(3.0), "call spread = 2*0.015*100 = $3.00"

        # Leg 1 — SELL → put, -1% price move → +100 bps profit; put half_spread 0.010.
        assert bool(out.is_call[1]) is False, "SELL entry must map to a put (prefer_calls=True)"
        assert out.underlying_moves_bps[1] == pytest.approx(100.0), (
            "SELL 100→99 profits on the down move: -1*(99-100)/100*1e4 = +100 bps"
        )
        assert out.spread_costs[1] == pytest.approx(2.0), "put spread = 2*0.010*100 = $2.00"

        assert out.is_call.dtype == bool

    def test_assembled_option_trade_pnl_value_locked(self):
        """Lock the per-trade assembly: option_pnl = gross - spread - comm - theta
        (zero_dte.py:405-423).

        gross = delta*(move_bps/1e4)*entry_price*100*contracts
              = 0.50 * 0.01 * 100 * 100 * 1 = $50.00 for both legs (entry_price=100).
        A dropped/added cost term, an assembly sign error, or gross using exit_price
        instead of entry_price would all be caught here.
        """
        out = self._make_transformer().transform(self._make_two_leg_result())

        assert out.commission_costs[0] == pytest.approx(1.40), "comm = 2*0.70*1 = $1.40"
        assert out.commission_costs[1] == pytest.approx(1.40)
        assert out.theta_costs[0] == pytest.approx(self._EXPECTED_THETA_PER_CONTRACT)
        assert out.theta_costs[1] == pytest.approx(self._EXPECTED_THETA_PER_CONTRACT), (
            "both legs share entry_price=100, IV=0.40, 120 min, 10-min hold → identical theta"
        )

        # gross $50.00 both legs; call leg pays $3.00 spread, put leg $2.00.
        expected_pnl_0 = 50.0 - 3.0 - 1.40 - self._EXPECTED_THETA_PER_CONTRACT  # ≈ 43.2766
        expected_pnl_1 = 50.0 - 2.0 - 1.40 - self._EXPECTED_THETA_PER_CONTRACT  # ≈ 44.2766
        assert out.option_trade_pnls[0] == pytest.approx(expected_pnl_0)
        assert out.option_trade_pnls[1] == pytest.approx(expected_pnl_1)

    def test_option_equity_curve_and_total_return_value_locked(self):
        """Lock the aggregate assembly: option_equity_curve = initial + cumsum([0, *pnls]),
        option_final_equity, option_total_return (zero_dte.py:425-434)."""
        out = self._make_transformer().transform(self._make_two_leg_result())

        expected_pnl_0 = 50.0 - 3.0 - 1.40 - self._EXPECTED_THETA_PER_CONTRACT
        expected_pnl_1 = 50.0 - 2.0 - 1.40 - self._EXPECTED_THETA_PER_CONTRACT
        expected_final = 100000.0 + expected_pnl_0 + expected_pnl_1

        assert out.option_equity_curve.shape == (3,), "n_round_trips + 1 = 3"
        assert out.option_equity_curve[0] == pytest.approx(100000.0)
        assert out.option_equity_curve[-1] == pytest.approx(expected_final)
        assert out.option_final_equity == pytest.approx(expected_final)
        assert out.option_total_return == pytest.approx(
            (expected_pnl_0 + expected_pnl_1) / 100000.0
        )
