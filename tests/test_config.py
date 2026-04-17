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

