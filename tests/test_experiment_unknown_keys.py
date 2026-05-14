"""FIND-070 closure (2026-05-14): WARN on unknown YAML keys + fail-loud on
wrong-block placement of readability gate parameters.

Validation cycle (Wave 1+2 of 2026-05-14, 8 cumulative adversarial agents)
surfaced a CRITICAL silent-misconfig in ``ExperimentRunner``:

- Production YAMLs ``configs/nvda_readability_first_xnas.yaml`` and
  ``configs/nvda_readability_first_arcx.yaml`` declared
  ``min_agreement: 1.0`` and ``min_confidence: 0.65`` under the ``backtest:``
  block. ``ExperimentRunner._build_strategy`` (experiment.py:191-195,
  354-355) reads from the ``strategy:`` block ONLY. Result: the values
  evaporated silently and the runner used readability defaults
  ``0.667`` / ``0.65`` (per ``readability.py:54`` P5 FIX 2026-03-17).

- Wave-2 adversarial verification showed those YAMLs are not currently
  runnable via ``ExperimentRunner.from_yaml`` (they lack ``signals.dir``),
  so the bug is LATENT-MISCONFIG-TRAP for future operators copying the
  YAML pattern — NOT historical corruption.

Closure fixes:

1. Module-level frozensets ``_KNOWN_BACKTEST_KEYS`` / ``_KNOWN_HOLDING_KEYS``
   / ``_KNOWN_STRATEGY_KEYS_{REGRESSION,READABILITY,DIRECTION}`` enumerate
   schema fields per block.
2. ``_warn_unknown_yaml_keys`` emits a single consolidated ``RuntimeWarning``
   on any unknown keys (mirrors hft-ops Phase 7.5 R5 idiom at commit
   ``3dd3ccb``).
3. ``_build_strategy`` readability branch fails loud with a ``ValueError``
   embedding a concrete migration hint when the wrong-block placement is
   detected (per hft-rules §5 fail-fast with precise error).

Test coverage:

- ``TestWarnUnknownYAMLKeysHelper`` — helper-function contract.
- ``TestBuildBacktestConfigUnknownKeys`` — backtest-block WARN path.
- ``TestBuildHoldingPolicyUnknownKeys`` — holding-block WARN path.
- ``TestBuildStrategyWrongBlockDetection`` — FIND-070 core; readability
  ValueError + correct-placement path.
- ``TestBuildStrategyUnknownStrategyKeys`` — strategy-block WARN path
  (with per-strategy-type frozensets).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from lobbacktest.experiment import (
    ExperimentRunner,
    _KNOWN_BACKTEST_KEYS,
    _KNOWN_HOLDING_KEYS,
    _KNOWN_STRATEGY_KEYS_READABILITY,
    _KNOWN_STRATEGY_KEYS_REGRESSION,
    _warn_unknown_yaml_keys,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _captured_runner_warns(caught):
    """Filter to ExperimentRunner-emitted warnings only (drop unrelated)."""
    return [w for w in caught if "ExperimentRunner" in str(w.message)]


class _ReadabilityData:
    """Minimal BacktestData-like stub for the readability branch."""

    predictions = np.array([0, 1, 2, 1, 0])
    agreement_ratio = np.array([0.333, 0.667, 1.0, 0.667, 0.333])
    confirmation_score = np.array([0.5, 0.7, 0.9, 0.7, 0.5])
    spreads = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    prices = np.array([100.0, 100.1, 100.2, 100.1, 100.0])
    predicted_returns = None


class _RegressionData:
    """Minimal BacktestData-like stub for the regression branch."""

    predictions = np.array([0, 1, 2])
    predicted_returns = np.array([2.0, -3.0, 5.0])
    spreads = np.array([1.0, 1.0, 1.0])
    prices = np.array([100.0, 100.5, 101.0])
    agreement_ratio = None
    confirmation_score = None


# ---------------------------------------------------------------------------
# Helper-function contract
# ---------------------------------------------------------------------------


class TestWarnUnknownYAMLKeysHelper:
    def test_unknown_key_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_unknown_yaml_keys("backtest", {"foo": 1}, frozenset({"bar"}))
        ours = _captured_runner_warns(caught)
        assert len(ours) == 1
        assert "'foo'" in str(ours[0].message)
        assert "backtest" in str(ours[0].message)
        assert ours[0].category is RuntimeWarning

    def test_known_key_only_no_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_unknown_yaml_keys(
                "backtest", {"bar": 1}, frozenset({"bar"}),
            )
        assert _captured_runner_warns(caught) == []

    def test_empty_raw_no_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_unknown_yaml_keys(
                "backtest", {}, frozenset({"bar"}),
            )
        assert _captured_runner_warns(caught) == []

    def test_multiple_unknown_keys_single_warning(self):
        """3 unknown keys → ONE consolidated warning citing all 3."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_unknown_yaml_keys(
                "strategy",
                {"a": 1, "b": 2, "c": 3, "type": "x"},
                frozenset({"type"}),
            )
        ours = _captured_runner_warns(caught)
        assert len(ours) == 1, "Expected single consolidated WARN"
        msg = str(ours[0].message)
        assert "'a'" in msg and "'b'" in msg and "'c'" in msg


# ---------------------------------------------------------------------------
# _build_backtest_config
# ---------------------------------------------------------------------------


class TestBuildBacktestConfigUnknownKeys:
    def test_unknown_backtest_key_warns(self):
        runner = ExperimentRunner({"backtest": {"unknown_field": 42}})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = runner._build_backtest_config()
        ours = _captured_runner_warns(caught)
        assert len(ours) == 1
        assert "backtest" in str(ours[0].message)
        assert "'unknown_field'" in str(ours[0].message)
        assert cfg is not None  # construction proceeds

    def test_known_backtest_keys_no_warn(self):
        runner = ExperimentRunner({
            "backtest": {
                "initial_capital": 50_000.0,
                "position_size": 0.05,
                "allow_short": True,
                "exchange": "XNAS",
                "trading_days_per_year": 252.0,
                "periods_per_day": 1000.0,
            },
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner._build_backtest_config()
        assert _captured_runner_warns(caught) == []

    def test_min_agreement_under_backtest_no_generic_warn(self):
        """``min_agreement``/``min_confidence`` are listed in
        ``_KNOWN_BACKTEST_KEYS`` (BacktestConfig dataclass schema includes
        them, even though ``_build_backtest_config`` does not consume them).
        The generic WARN must NOT fire — wrong-block detection lives at
        ``_build_strategy`` instead.
        """
        runner = ExperimentRunner({
            "backtest": {"min_agreement": 1.0, "min_confidence": 0.65},
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner._build_backtest_config()
        assert _captured_runner_warns(caught) == []


# ---------------------------------------------------------------------------
# _build_holding_policy
# ---------------------------------------------------------------------------


class TestBuildHoldingPolicyUnknownKeys:
    def test_unknown_holding_key_warns(self):
        runner = ExperimentRunner({"holding": {"unknown_holding_field": 7}})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner._build_holding_policy()
        ours = _captured_runner_warns(caught)
        assert len(ours) == 1
        assert "holding" in str(ours[0].message)
        assert "'unknown_holding_field'" in str(ours[0].message)

    def test_known_holding_keys_no_warn(self):
        runner = ExperimentRunner({
            "holding": {
                "type": "horizon_aligned",
                "hold_events": 20,
                "stop_loss_bps": 10.0,
                "take_profit_bps": 20.0,
            },
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner._build_holding_policy()
        assert _captured_runner_warns(caught) == []


# ---------------------------------------------------------------------------
# _build_strategy — FIND-070 core wrong-block detection
# ---------------------------------------------------------------------------


class TestBuildStrategyWrongBlockDetection:
    def test_min_agreement_under_backtest_raises_FIND070(self):
        """``min_agreement`` under ``backtest:`` but NOT under ``strategy:`` →
        precise ValueError citing FIND-070 + migration hint."""
        runner = ExperimentRunner({
            "backtest": {"min_agreement": 1.0},
            "strategy": {"type": "readability"},
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore unrelated WARNs
            with pytest.raises(ValueError) as exc:
                runner._build_strategy(_ReadabilityData(), "readability", {})
        assert "FIND-070" in str(exc.value)
        assert "min_agreement" in str(exc.value)
        assert "strategy:" in str(exc.value)  # migration hint

    def test_min_confidence_under_backtest_raises_FIND070(self):
        runner = ExperimentRunner({
            "backtest": {"min_confidence": 0.7},
            "strategy": {"type": "readability"},
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as exc:
                runner._build_strategy(_ReadabilityData(), "readability", {})
        assert "FIND-070" in str(exc.value)
        assert "min_confidence" in str(exc.value)

    def test_both_keys_under_backtest_raises_with_both_cited(self):
        runner = ExperimentRunner({
            "backtest": {"min_agreement": 1.0, "min_confidence": 0.7},
            "strategy": {"type": "readability"},
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as exc:
                runner._build_strategy(_ReadabilityData(), "readability", {})
        msg = str(exc.value)
        assert "min_agreement" in msg and "min_confidence" in msg

    def test_correct_block_placement_succeeds(self):
        runner = ExperimentRunner({
            "strategy": {
                "type": "readability",
                "min_agreement": 1.0,
                "min_confidence": 0.65,
            },
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            strat = runner._build_strategy(
                _ReadabilityData(),
                "readability",
                {"min_agreement": 1.0, "min_confidence": 0.65},
            )
        assert strat.config.min_agreement == 1.0
        assert strat.config.min_confidence == 0.65

    def test_strategy_block_present_no_raise_even_with_backtest_min_agreement(
        self,
    ):
        """If ``params`` (strategy block) declares the key, no raise — the
        backtest-block value is interpreted as legacy/redundant and the
        strategy-block value wins (the runner already constructs ``params``
        from ``strategy_config.items()`` at line 193)."""
        runner = ExperimentRunner({
            "backtest": {"min_agreement": 0.5},
            "strategy": {
                "type": "readability",
                "min_agreement": 1.0,
                "min_confidence": 0.65,
            },
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            strat = runner._build_strategy(
                _ReadabilityData(),
                "readability",
                {"min_agreement": 1.0, "min_confidence": 0.65},
            )
        assert strat.config.min_agreement == 1.0

    def test_regression_strategy_does_not_raise_on_backtest_min_agreement(self):
        """Wrong-block detection is readability-specific; regression strategy
        doesn't consume ``min_agreement`` and must not raise FIND-070."""
        runner = ExperimentRunner({
            "backtest": {"min_agreement": 1.0},
            "strategy": {"type": "regression"},
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            strat = runner._build_strategy(_RegressionData(), "regression", {})
        assert strat is not None


# ---------------------------------------------------------------------------
# _build_strategy — strategy-block unknown-key WARN path
# ---------------------------------------------------------------------------


class TestBuildStrategyUnknownStrategyKeys:
    def test_unknown_regression_strategy_key_warns(self):
        runner = ExperimentRunner({
            "strategy": {"type": "regression", "unknown_field": 42},
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner._build_strategy(_RegressionData(), "regression", {})
        ours = _captured_runner_warns(caught)
        strategy_warns = [w for w in ours if "strategy" in str(w.message)]
        assert len(strategy_warns) == 1
        assert "'unknown_field'" in str(strategy_warns[0].message)

    def test_unknown_readability_strategy_key_warns(self):
        runner = ExperimentRunner({
            "strategy": {"type": "readability", "weird_field": 42},
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner._build_strategy(_ReadabilityData(), "readability", {})
        ours = _captured_runner_warns(caught)
        strategy_warns = [w for w in ours if "strategy" in str(w.message)]
        assert len(strategy_warns) == 1
        assert "'weird_field'" in str(strategy_warns[0].message)

    def test_known_readability_keys_no_warn(self):
        runner = ExperimentRunner({
            "strategy": {
                "type": "readability",
                "min_agreement": 1.0,
                "min_confidence": 0.65,
                "max_spread_bps": 1.05,
            },
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner._build_strategy(
                _ReadabilityData(),
                "readability",
                {
                    "min_agreement": 1.0,
                    "min_confidence": 0.65,
                    "max_spread_bps": 1.05,
                },
            )
        ours = _captured_runner_warns(caught)
        strategy_warns = [w for w in ours if "strategy" in str(w.message)]
        assert strategy_warns == []


# ---------------------------------------------------------------------------
# Frozenset-schema sanity (lock contract membership)
# ---------------------------------------------------------------------------


class TestFrozensetSchemaSanity:
    """Lock the frozenset memberships against accidental edits."""

    def test_backtest_frozenset_size_and_min_agreement_membership(self):
        """Per BacktestConfig dataclass at config.py:312-313 the dataclass
        DOES declare these fields. The generic WARN must tolerate them; the
        wrong-block detection lives in ``_build_strategy``.

        Lock the total membership count too — drift detector for
        BACKTEST_INDEX.md citation accuracy (FIND-070 closure mentions
        ``14 keys``).
        """
        assert "min_agreement" in _KNOWN_BACKTEST_KEYS
        assert "min_confidence" in _KNOWN_BACKTEST_KEYS
        assert len(_KNOWN_BACKTEST_KEYS) == 14, (
            f"Expected 14 keys (per BACKTEST_INDEX.md FIND-070 Closure section); "
            f"got {len(_KNOWN_BACKTEST_KEYS)}: {sorted(_KNOWN_BACKTEST_KEYS)!r}"
        )

    def test_readability_frozenset_is_subset_of_legal_keys(self):
        # Exactly the fields ReadabilityConfig (readability.py) declares.
        assert _KNOWN_STRATEGY_KEYS_READABILITY == frozenset({
            "type", "min_agreement", "min_confidence", "max_spread_bps",
        })

    def test_regression_frozenset_is_subset_of_legal_keys(self):
        assert _KNOWN_STRATEGY_KEYS_REGRESSION == frozenset({
            "type",
            "min_return_bps",
            "max_spread_bps",
            "primary_horizon_idx",
            "cooldown_events",
        })

    def test_holding_frozenset(self):
        assert _KNOWN_HOLDING_KEYS == frozenset({
            "type", "hold_events", "stop_loss_bps", "take_profit_bps",
        })


# ---------------------------------------------------------------------------
# Sweep path interaction (MED-1 mid-impl gate finding)
# ---------------------------------------------------------------------------


class TestSweepPathFIND070Interaction:
    """The ``_run_sweep`` path builds ``params = {**base_params, sweep_key:
    value}``. If the operator sweeps over ``min_agreement``, each iteration
    has the key in ``params`` so FIND-070 raise is correctly suppressed.
    """

    def test_sweep_over_min_agreement_does_not_raise_FIND070(self):
        runner = ExperimentRunner({
            # Note: NO backtest.min_agreement set here — sweep populates params.
            "strategy": {"type": "readability"},
        })
        sweep_params = {"min_agreement": 1.0}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            strat = runner._build_strategy(
                _ReadabilityData(),
                "readability",
                sweep_params,
            )
        assert strat.config.min_agreement == 1.0

    def test_sweep_over_min_agreement_with_backtest_block_does_not_raise(self):
        """Sweep populates params with min_agreement → FIND-070 detection
        gate sees the key in params → raise suppressed, even if backtest:
        block also has the legacy field set."""
        runner = ExperimentRunner({
            "backtest": {"min_agreement": 0.5},  # legacy, would-be wrong-block
            "strategy": {"type": "readability"},
        })
        sweep_params = {"min_agreement": 1.0}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            strat = runner._build_strategy(
                _ReadabilityData(),
                "readability",
                sweep_params,
            )
        # Strategy-block value (via sweep params) wins.
        assert strat.config.min_agreement == 1.0


# ---------------------------------------------------------------------------
# BacktestConfig DeprecationWarning (HIGH-2 mid-impl gate finding)
# ---------------------------------------------------------------------------


class TestBacktestConfigDeprecatedFields:
    """FIND-070 HIGH-2: ``BacktestConfig.min_agreement`` /
    ``BacktestConfig.min_confidence`` are declared on the dataclass schema
    for legacy compat but NOT consumed by ``ExperimentRunner``. Emit
    ``DeprecationWarning`` at ``__post_init__`` so operators see a
    machine-visible signal before 2026-10-31 field removal.
    """

    def test_min_agreement_non_default_emits_deprecation_warning(self):
        from lobbacktest.config import BacktestConfig

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            BacktestConfig(min_agreement=1.0)
        deprecations = [
            w for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "min_agreement is DEPRECATED" in str(w.message)
        ]
        assert len(deprecations) == 1
        assert "2026-10-31" in str(deprecations[0].message)
        assert "strategy:" in str(deprecations[0].message)

    def test_min_confidence_non_default_emits_deprecation_warning(self):
        from lobbacktest.config import BacktestConfig

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            BacktestConfig(min_confidence=0.65)
        deprecations = [
            w for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "min_confidence is DEPRECATED" in str(w.message)
        ]
        assert len(deprecations) == 1

    def test_default_no_deprecation_warning(self):
        """Defaults are None; no DeprecationWarning fires unless operator
        explicitly sets the field non-None."""
        from lobbacktest.config import BacktestConfig

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            BacktestConfig()
        deprecations = [
            w for w in caught
            if issubclass(w.category, DeprecationWarning)
            and (
                "min_agreement" in str(w.message)
                or "min_confidence" in str(w.message)
            )
        ]
        assert deprecations == []
