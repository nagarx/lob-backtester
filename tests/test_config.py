"""
Tests for configuration module.

Tests verify:
- CostConfig validation
- BacktestConfig validation
- Serialization/deserialization
"""

import numpy as np
import pytest
import tempfile
import os

from lobbacktest.config import BacktestConfig, CostConfig


class TestCostConfig:
    """Tests for CostConfig dataclass."""

    def test_default_values(self):
        """Test default cost configuration."""
        config = CostConfig()
        assert config.spread_bps == 1.0
        assert config.slippage_bps == 0.5
        assert config.commission_per_trade == 0.0

    def test_total_bps(self):
        """Test total_bps property."""
        config = CostConfig(spread_bps=2.0, slippage_bps=1.0)
        assert config.total_bps == 3.0

    def test_compute_cost(self):
        """
        Test cost computation.

        Formula: cost = notional * (total_bps / 10000) + commission
        """
        config = CostConfig(
            spread_bps=1.0,  # 0.01%
            slippage_bps=0.5,  # 0.005%
            commission_per_trade=1.0,
        )

        # For $10,000 notional:
        # Variable: 10000 * (1.5 / 10000) = $1.50
        # Fixed: $1.00
        # Total: $2.50
        cost = config.compute_cost(10000)
        assert abs(cost - 2.50) < 0.001

    def test_negative_spread_raises(self):
        """Test that negative spread raises error."""
        with pytest.raises(ValueError, match="spread_bps must be >= 0"):
            CostConfig(spread_bps=-1.0)

    def test_negative_slippage_raises(self):
        """Test that negative slippage raises error."""
        with pytest.raises(ValueError, match="slippage_bps must be >= 0"):
            CostConfig(slippage_bps=-1.0)

    def test_negative_commission_raises(self):
        """Test that negative commission raises error."""
        with pytest.raises(ValueError, match="commission_per_trade must be >= 0"):
            CostConfig(commission_per_trade=-1.0)


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BacktestConfig()
        assert config.initial_capital == 100_000.0
        assert config.position_size == 0.1
        assert config.max_position == 1.0
        assert config.allow_short is True
        assert config.fill_price == "close"

    def test_annualization_factor(self):
        """Test annualization factor computation."""
        config = BacktestConfig(
            trading_days_per_year=252,
            periods_per_day=100,
        )
        expected = np.sqrt(252 * 100)
        assert abs(config.annualization_factor - expected) < 0.001

    def test_validation_initial_capital(self):
        """Test that zero/negative capital raises error."""
        with pytest.raises(ValueError, match="initial_capital must be > 0"):
            BacktestConfig(initial_capital=0)

        with pytest.raises(ValueError, match="initial_capital must be > 0"):
            BacktestConfig(initial_capital=-1000)

    def test_validation_position_size_bounds(self):
        """Test that position_size must be in (0, 1]."""
        with pytest.raises(ValueError, match="position_size must be in"):
            BacktestConfig(position_size=0)

        with pytest.raises(ValueError, match="position_size must be in"):
            BacktestConfig(position_size=1.5)

    def test_validation_max_position_bounds(self):
        """Test that max_position must be in (0, 1]."""
        with pytest.raises(ValueError, match="max_position must be in"):
            BacktestConfig(max_position=0)

        with pytest.raises(ValueError, match="max_position must be in"):
            BacktestConfig(max_position=1.5)

    def test_validation_position_size_exceeds_max(self):
        """Test that position_size cannot exceed max_position."""
        with pytest.raises(ValueError, match="position_size.*cannot exceed.*max_position"):
            BacktestConfig(position_size=0.5, max_position=0.3)

    def test_validation_fill_price(self):
        """Test that fill_price must be valid."""
        # Valid values
        BacktestConfig(fill_price="close")
        BacktestConfig(fill_price="midpoint")

        # Invalid value
        with pytest.raises(ValueError, match="fill_price must be"):
            BacktestConfig(fill_price="invalid")

    def test_validation_stop_loss(self):
        """Test that stop_loss must be positive if set."""
        with pytest.raises(ValueError, match="stop_loss_pct must be > 0"):
            BacktestConfig(stop_loss_pct=0)

        with pytest.raises(ValueError, match="stop_loss_pct must be > 0"):
            BacktestConfig(stop_loss_pct=-0.1)

        # Valid positive value
        config = BacktestConfig(stop_loss_pct=0.02)
        assert config.stop_loss_pct == 0.02

    def test_to_dict(self):
        """Test serialization to dict."""
        config = BacktestConfig(
            initial_capital=50000,
            position_size=0.2,
        )
        d = config.to_dict()

        assert d["initial_capital"] == 50000
        assert d["position_size"] == 0.2
        assert "costs" in d
        assert d["costs"]["spread_bps"] == 1.0

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "initial_capital": 75000,
            "position_size": 0.15,
            "costs": {
                "spread_bps": 2.0,
                "slippage_bps": 1.0,
            },
        }
        config = BacktestConfig.from_dict(d)

        assert config.initial_capital == 75000
        assert config.position_size == 0.15
        assert config.costs.spread_bps == 2.0
        assert config.costs.slippage_bps == 1.0

    def test_round_trip_serialization(self):
        """Test that to_dict -> from_dict preserves values."""
        original = BacktestConfig(
            initial_capital=123456,
            position_size=0.25,
            max_position=0.5,
            costs=CostConfig(spread_bps=3.0, slippage_bps=1.5),
            allow_short=False,
            stop_loss_pct=0.05,
        )

        d = original.to_dict()
        restored = BacktestConfig.from_dict(d)

        assert restored.initial_capital == original.initial_capital
        assert restored.position_size == original.position_size
        assert restored.max_position == original.max_position
        assert restored.costs.spread_bps == original.costs.spread_bps
        assert restored.allow_short == original.allow_short
        assert restored.stop_loss_pct == original.stop_loss_pct

    def test_yaml_save_load(self):
        """Test saving and loading from YAML file."""
        config = BacktestConfig(
            initial_capital=200000,
            position_size=0.3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")

            config.save_yaml(path)
            loaded = BacktestConfig.load_yaml(path)

            assert loaded.initial_capital == 200000
            assert loaded.position_size == 0.3


class TestBacktestConfigHf1ModeAwareIvDefault:
    """HF-1 closure (2026-05-16 LATE; Bundle 1 hygiene post Option B Path B'):
    BacktestConfig.from_dict path mirror of TestHf1ModeAwareIvDefault in
    test_experiment.py. Sister cycle's #PY-273 closed
    OpraCalibratedCosts.deep_itm() factory default (0.40 → 0.25) but
    BacktestConfig.from_dict at config.py:566 STILL inherited 0.40 as
    hard-coded default → silent ~60-100% theta overestimation when YAML
    omitted implied_vol on Deep ITM (delta>=0.90) configurations.

    Mode-discrimination via zero_dte.delta>=0.90 fixes this. Class default
    at config.py:173 remains 0.40 (correct for ATM regime — preserved by
    these tests' ATM-default assertions).

    Covers BacktestConfig.from_dict + BacktestConfig.load_yaml paths
    (production-default for any YAML omitting opra_costs.implied_vol).
    """

    def test_deep_itm_delta_inherits_iv_025_default(self):
        """HF-1: zero_dte.delta=0.95 + omitted implied_vol → 0.25 default
        (NOT class default 0.40). Closes silent Deep ITM theta overestimation
        for YAMLs that specify Deep ITM trading without explicit IV override."""
        d = {
            "initial_capital": 100000,
            "zero_dte": {
                "enabled": True,
                "delta": 0.95,
            },
        }
        config = BacktestConfig.from_dict(d)
        assert config.zero_dte.delta == 0.95
        assert config.zero_dte.opra_costs.implied_vol == 0.25, (
            f"HF-1: delta=0.95 (Deep ITM) with omitted implied_vol should "
            f"inherit 0.25 (factory-aligned with #PY-273); got "
            f"{config.zero_dte.opra_costs.implied_vol}"
        )

    def test_atm_delta_inherits_iv_040_default(self):
        """HF-1: zero_dte.delta=0.50 + omitted implied_vol → 0.40 default
        preserved (ATM regime; matches class default at config.py:173).
        Regression-locks ATM correctness — only Deep ITM regime changes."""
        d = {
            "initial_capital": 100000,
            "zero_dte": {
                "enabled": True,
                "delta": 0.50,
            },
        }
        config = BacktestConfig.from_dict(d)
        assert config.zero_dte.delta == 0.50
        assert config.zero_dte.opra_costs.implied_vol == 0.40

    def test_explicit_implied_vol_overrides_mode_aware_default(self):
        """HF-1: explicit YAML implied_vol wins regardless of delta regime.
        Operator override never silently replaced by mode-aware default."""
        d = {
            "initial_capital": 100000,
            "zero_dte": {
                "enabled": True,
                "delta": 0.95,  # Deep ITM regime
                "opra_costs": {
                    "implied_vol": 0.35,  # explicit override
                },
            },
        }
        config = BacktestConfig.from_dict(d)
        assert config.zero_dte.opra_costs.implied_vol == 0.35, (
            f"HF-1: explicit YAML implied_vol=0.35 must win over mode-aware "
            f"default 0.25 (delta=0.95). Got "
            f"{config.zero_dte.opra_costs.implied_vol}"
        )

    def test_omitted_zero_dte_block_preserves_class_default(self):
        """HF-1 back-compat: YAML with no zero_dte block → default delta=0.50
        (from BacktestConfig.from_dict default) → ATM regime → IV=0.40.
        Locks 'missing-block path defaults to ATM' invariant."""
        d = {"initial_capital": 100000}
        config = BacktestConfig.from_dict(d)
        assert config.zero_dte.delta == 0.50
        assert config.zero_dte.opra_costs.implied_vol == 0.40


class TestBacktestConfigPy263PeriodsPerDayResolution:
    """#PY-263 closure (2026-05-21; Cycle A-rev): mode-aware ``periods_per_day``
    dispatch via ``BacktestConfig.resolved_periods_per_day`` property.

    Background: pre-fix ``periods_per_day: float = 1000.0`` default silently
    inflated Sharpe by ``sqrt(1000/X)`` at sub-daily bins (1.6018x at 60s,
    1.131x at 30s, 0.456x at 5s). Wave 1 Agent B (validation cycle 2026-05-21)
    verified math; Wave 2 Agent G refuted Approach A (default 1000→245)
    because it transplanted the bug to non-60s bins. Approach B (this fix)
    uses Optional[float] = None + ``resolved_periods_per_day`` property +
    mutex against ``zero_dte.bin_seconds`` (mirrors ZeroDteConfig L349-353
    events_per_minute/bin_seconds mutex pattern).

    Locks the invariant: SAME (trading_days_per_year, sampling_cadence) →
    SAME annualization_factor across all consumer paths (engine + 4 scripts +
    4 metric classes).
    """

    def test_default_periods_per_day_is_none(self):
        """#PY-263: field default changed from 1000.0 → None per Optional[float] migration."""
        config = BacktestConfig()
        assert config.periods_per_day is None, (
            "#PY-263 (2026-05-21): periods_per_day default must be None "
            "(was 1000.0). None triggers mode-aware dispatch via "
            "resolved_periods_per_day. Got: "
            f"{config.periods_per_day!r}"
        )

    def test_resolved_falls_back_to_1000_with_deprecation_warning(self):
        """#PY-263: legacy fallback emits DeprecationWarning per hft-rules §8.

        When neither explicit periods_per_day nor zero_dte.bin_seconds set,
        resolved_periods_per_day returns 1000.0 (legacy default) BUT emits
        DeprecationWarning so silent degradation is machine-visible.
        """
        config = BacktestConfig()
        with pytest.warns(DeprecationWarning, match=r"#PY-263"):
            resolved = config.resolved_periods_per_day
        assert resolved == 1000.0, (
            f"#PY-263: legacy fallback should return 1000.0 (preserves "
            f"pre-fix behavior for back-compat). Got: {resolved}"
        )

    def test_resolved_from_bin_seconds_60s(self):
        """#PY-263: bin_seconds=60 → resolved=390.0 (RTH 23400/60).

        Closes the silent inflation at 60s bins: sqrt(1000/390) = 1.6018x.
        After fix, Sharpe at 60s bins is correctly annualized.
        """
        from lobbacktest.config import ZeroDteConfig
        config = BacktestConfig(
            zero_dte=ZeroDteConfig(enabled=True, bin_seconds=60.0)
        )
        assert config.resolved_periods_per_day == 390.0, (
            f"#PY-263: bin_seconds=60 should derive 23400/60 = 390.0. "
            f"Got: {config.resolved_periods_per_day}"
        )

    def test_resolved_from_bin_seconds_5s(self):
        """#PY-263: bin_seconds=5 → resolved=4680.0 (RTH 23400/5).

        Verifies the formula scales correctly across bin sizes (Wave 2G
        flagged Approach A's default 245 as wrong for ~5s/30s bins).
        """
        from lobbacktest.config import ZeroDteConfig
        config = BacktestConfig(
            zero_dte=ZeroDteConfig(enabled=True, bin_seconds=5.0)
        )
        assert config.resolved_periods_per_day == 4680.0

    def test_resolved_from_explicit_override(self):
        """#PY-263: explicit periods_per_day=X wins (legacy override path)."""
        config = BacktestConfig(periods_per_day=245.0)
        assert config.resolved_periods_per_day == 245.0, (
            "#PY-263: explicit operator override must take precedence over "
            "mode-aware derivation."
        )

    def test_mutex_explicit_and_bin_seconds_raises(self):
        """#PY-263 mutex per hft-rules §5 fail-fast (mirrors ZeroDteConfig
        L349-353 events_per_minute/bin_seconds mutex pattern).

        Both ``periods_per_day`` and ``zero_dte.bin_seconds`` specify the
        same physical quantity. Setting both is ambiguous; fail-loud at
        construction.
        """
        from lobbacktest.config import ZeroDteConfig
        with pytest.raises(ValueError, match=r"mutually exclusive.*#PY-263"):
            BacktestConfig(
                periods_per_day=500.0,
                zero_dte=ZeroDteConfig(enabled=True, bin_seconds=60.0),
            )

    def test_annualization_factor_uses_resolved(self):
        """#PY-263: annualization_factor property routes through resolved_periods_per_day."""
        from lobbacktest.config import ZeroDteConfig
        import numpy as np
        config = BacktestConfig(
            zero_dte=ZeroDteConfig(enabled=True, bin_seconds=60.0),
            trading_days_per_year=252.0,
        )
        expected = float(np.sqrt(252.0 * 390.0))
        assert abs(config.annualization_factor - expected) < 1e-9, (
            f"#PY-263: annualization_factor must use resolved_periods_per_day "
            f"(=390 at 60s bins). Expected {expected}, got "
            f"{config.annualization_factor}"
        )

    def test_explicit_negative_periods_per_day_raises(self):
        """Validation: explicit periods_per_day must be > 0 when set."""
        with pytest.raises(ValueError, match=r"periods_per_day must be > 0"):
            BacktestConfig(periods_per_day=-1.0)

    def test_none_does_not_trigger_validation(self):
        """#PY-263: None is valid (triggers mode-aware dispatch); only > 0 enforced when set."""
        # Should not raise — None is the new default
        config = BacktestConfig(periods_per_day=None)
        assert config.periods_per_day is None

    def test_round_trip_to_dict_from_dict_with_none(self):
        """#PY-263: to_dict emits None; from_dict reads None default — round-trip preserves Optional semantics."""
        original = BacktestConfig()
        d = original.to_dict()
        assert d["periods_per_day"] is None
        restored = BacktestConfig.from_dict(d)
        assert restored.periods_per_day is None

    def test_round_trip_to_dict_from_dict_with_explicit(self):
        """#PY-263: explicit periods_per_day survives YAML round-trip."""
        original = BacktestConfig(periods_per_day=390.0)
        d = original.to_dict()
        assert d["periods_per_day"] == 390.0
        restored = BacktestConfig.from_dict(d)
        assert restored.periods_per_day == 390.0
        assert restored.resolved_periods_per_day == 390.0

    def test_legacy_yaml_with_explicit_1000_preserved(self):
        """#PY-263: 2 production YAMLs explicitly set `periods_per_day: 1000`
        (nvda_readability_first_arcx.yaml + nvda_readability_first_xnas.yaml).
        Approach B preserves operator-explicit value (no migration required;
        no DeprecationWarning fires for explicit override).
        """
        import warnings
        config = BacktestConfig.from_dict({"periods_per_day": 1000.0})
        assert config.periods_per_day == 1000.0
        # Explicit override does NOT trigger DeprecationWarning
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning = test failure
            resolved = config.resolved_periods_per_day
        assert resolved == 1000.0


class TestExchangePresetsSingleSource:
    """Phase 6 6A.6 regression guards — `_EXCHANGE_PRESETS` is the SINGLE
    SOURCE of exchange-calibrated cost data. Prior state duplicated the
    dict in a dead `CostConfig.EXCHANGE_PRESETS` class-var AND an inline
    literal inside `for_exchange()` — drift hazard (any preset change
    required updating BOTH places to stay consistent).
    """

    def test_no_dead_class_attribute(self):
        """Dead `EXCHANGE_PRESETS` class-var must not be reintroduced."""
        assert not hasattr(CostConfig, "EXCHANGE_PRESETS"), (
            "CostConfig.EXCHANGE_PRESETS was removed in Phase 6 6A.6 "
            "(duplicated module-level _EXCHANGE_PRESETS). Do not reintroduce."
        )

    def test_for_exchange_reads_module_level_source(self):
        """for_exchange() reads _EXCHANGE_PRESETS (single source). A
        runtime patch to the module-level dict must flow through."""
        import lobbacktest.config as _cfg_mod
        original = _cfg_mod._EXCHANGE_PRESETS["XNAS"].copy()
        try:
            # Patch the source — for_exchange() must pick it up.
            _cfg_mod._EXCHANGE_PRESETS["XNAS"] = {
                "spread_bps": 99.0,
                "slippage_bps": 99.0,
                "taker_fee_bps": 99.0,
                "maker_rebate_bps": 0.0,
            }
            cost = CostConfig.for_exchange("XNAS")
            assert cost.spread_bps == 99.0, (
                "for_exchange() must read module-level _EXCHANGE_PRESETS. "
                "If this test fails, a duplicate preset source has been "
                "reintroduced somewhere."
            )
        finally:
            _cfg_mod._EXCHANGE_PRESETS["XNAS"] = original

    def test_for_exchange_unknown_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown exchange"):
            CostConfig.for_exchange("BATS")

    def test_xnas_vwes_calibration_preserved(self):
        """Baseline regression — 233-day NVDA VWES calibration values
        (mbo-statistical-profiler output) preserved post-refactor."""
        xnas = CostConfig.for_exchange("XNAS")
        assert xnas.spread_bps == 1.0
        assert xnas.slippage_bps == 1.97   # XNAS VWES
        assert xnas.taker_fee_bps == 0.30
        assert xnas.maker_rebate_bps == -0.20

    def test_arcx_vwes_calibration_preserved(self):
        arcx = CostConfig.for_exchange("ARCX")
        assert arcx.spread_bps == 1.0
        assert arcx.slippage_bps == 1.10   # ARCX VWES
        assert arcx.taker_fee_bps == 0.25
        assert arcx.maker_rebate_bps == -0.15

