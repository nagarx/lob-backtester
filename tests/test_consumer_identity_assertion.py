"""Consumer-side identity assertion (2026-08-15, operator ruling R3).

WHY THIS EXISTS
    The producer (trainer ``SignalExporter``) writes an 11-field
    ``CompatibilityContract`` into ``signal_metadata.json`` — including
    ``label_strategy_hash``, the identity of the dependent variable the model
    was actually fitted to. The consumer asserted essentially ONE of those 11
    (``primary_horizon_idx``), and only when the strategy config happened to
    set it. Measured 2026-08-15: ``label_strategy_hash`` appeared 0 times in
    ``experiment.py`` and 0 times in ``data/signal_manifest.py`` while being
    present in 9 of the 19 signal manifests on disk. The field was emitted,
    persisted, and never compared.

    This is the programme's recurring shape: an identity is DERIVED and then
    DISCARDED at the boundary. Cf. ``ending_indices`` (computed in
    ``alignment.rs``, dropped before persist → a positional join fabricated
    IC +1.0000 on 162/162 days) and ``day_boundaries`` (computed in
    ``data/loader.py``, never passed to the engine → the no-overnight charter
    went unenforced).

THE DISTINCTION THESE TESTS LOCK
    ABSENCE is grandfathered; DISAGREEMENT is fatal. A producer that never
    declared a field has made no claim to contradict, so asserting against it
    would hard-fail historical exports for a fact nobody ever asserted. A
    field present on BOTH sides and differing is real version skew and must
    stop the run. Both halves are tested here; neither is safe alone.

    Measured motivation for the absence half: ``calibration_method`` is
    ``None`` on 8 of the 9 on-disk manifests that carry a compatibility block
    at all. Grandfathering is the majority case, not a hypothetical.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from hft_contracts.compatibility import CompatibilityContract
from hft_contracts.validation import ContractError

from lobbacktest.engine.vectorized import BacktestData
from lobbacktest.experiment import ExperimentRunner

# A real pre-Phase-II export (no `compatibility` block at all). Used to prove
# the historical-load path still works end to end.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_HISTORICAL_SIGNAL_DIR = (
    _REPO_ROOT
    / "lob-model-trainer/outputs/experiments/nvda_tlob_128feat_regression_h10/signals/test"
)

_KNOWN_HASH = "7299e11a14c8466099329282406a14adf550c0e3da331920b1410d6a35d26692"


def _contract(**overrides: Any) -> CompatibilityContract:
    defaults: Dict[str, Any] = dict(
        contract_version="3.0",
        schema_version="3.0",
        feature_count=98,
        window_size=20,
        feature_layout="default",
        data_source="mbo_lob",
        label_strategy_hash=_KNOWN_HASH,
        calibration_method=None,
        primary_horizon_idx=0,
        horizons=(10, 60, 300),
        normalization_strategy="none",
    )
    defaults.update(overrides)
    return CompatibilityContract(**defaults)


def _write_signal_dir(
    path: Path,
    contract: CompatibilityContract | None,
    n: int = 16,
) -> Path:
    """Minimal regression signal dir. ``contract=None`` → legacy (no block)."""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    np.save(path / "prices.npy", np.abs(rng.randn(n)) + 100.0)
    np.save(path / "predicted_returns.npy", rng.randn(n, 3))
    np.save(path / "regression_labels.npy", rng.randn(n, 3))
    np.save(path / "spreads.npy", np.abs(rng.randn(n)) * 0.5 + 1.0)

    meta: Dict[str, Any] = {
        "signal_type": "regression",
        "model_type": "tlob_regression",
        "split": "test",
        "total_samples": n,
        "exported_at": "2026-08-15T00:00:00+00:00",
        "checkpoint": "/tmp/ckpt.pt",
    }
    if contract is not None:
        meta["compatibility"] = {
            "contract_version": contract.contract_version,
            "schema_version": contract.schema_version,
            "feature_count": contract.feature_count,
            "window_size": contract.window_size,
            "feature_layout": contract.feature_layout,
            "data_source": contract.data_source,
            "label_strategy_hash": contract.label_strategy_hash,
            "calibration_method": contract.calibration_method,
            "primary_horizon_idx": contract.primary_horizon_idx,
            "horizons": list(contract.horizons) if contract.horizons else None,
            "normalization_strategy": contract.normalization_strategy,
        }
        meta["compatibility_fingerprint"] = contract.fingerprint()
    (path / "signal_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path


def _runner(signal_dir: Path, expect: Dict[str, Any] | None = None, **strategy: Any):
    signals: Dict[str, Any] = {"dir": str(signal_dir)}
    if expect is not None:
        signals["expect"] = expect
    return ExperimentRunner(
        {
            "experiment": {"name": "identity_assertion_test"},
            "signals": signals,
            "backtest": {"initial_capital": 100_000, "exchange": "XNAS"},
            "strategy": {"type": "regression", **strategy},
            "holding": {"type": "horizon_aligned", "hold_events": 2},
        }
    )


class TestAgreementPasses:
    """A declaration matching the manifest is a no-op — the run proceeds."""

    def test_matching_label_strategy_hash_passes(self, tmp_path):
        d = _write_signal_dir(tmp_path / "sig", _contract())
        fields = _runner(
            d, expect={"label_strategy_hash": _KNOWN_HASH}
        )._expected_compatibility_fields(
            manifest_compat=json.loads((d / "signal_metadata.json").read_text())["compatibility"]
        )
        assert fields == {"label_strategy_hash": _KNOWN_HASH}
        # And it survives the real validate() path.
        BacktestData.from_signal_dir(str(d), validate=True, expected_fields=fields)

    def test_multi_field_agreement_passes(self, tmp_path):
        d = _write_signal_dir(tmp_path / "sig", _contract())
        expect = {"feature_count": 98, "window_size": 20, "data_source": "mbo_lob"}
        BacktestData.from_signal_dir(str(d), validate=True, expected_fields=expect)


class TestDisagreementFailsLoud:
    """Present on both sides + differing == real version skew. Stop the run."""

    def test_label_strategy_hash_mismatch_raises_naming_both_values(self, tmp_path):
        d = _write_signal_dir(tmp_path / "sig", _contract())
        wrong = "b" * 64
        with pytest.raises(ContractError) as exc:
            BacktestData.from_signal_dir(
                str(d), validate=True, expected_fields={"label_strategy_hash": wrong}
            )
        msg = str(exc.value)
        # The message must name the FIELD and BOTH values — a bare "mismatch"
        # is not actionable when 11 fields could be at fault.
        assert "label_strategy_hash" in msg
        assert _KNOWN_HASH in msg, "manifest value must appear in the error"
        assert wrong in msg, "expected value must appear in the error"

    def test_feature_count_mismatch_raises(self, tmp_path):
        d = _write_signal_dir(tmp_path / "sig", _contract(feature_count=98))
        with pytest.raises(ContractError):
            BacktestData.from_signal_dir(
                str(d), validate=True, expected_fields={"feature_count": 148}
            )


class TestGrandfatheredOnAbsence:
    """Absence is not disagreement. The producer made no claim to contradict."""

    def test_producer_side_absence_is_dropped_with_warning(self, tmp_path):
        # calibration_method is None on 8/9 real on-disk manifests.
        d = _write_signal_dir(tmp_path / "sig", _contract(calibration_method=None))
        compat = json.loads((d / "signal_metadata.json").read_text())["compatibility"]
        runner = _runner(d, expect={"calibration_method": "variance_match", "feature_count": 98})

        with pytest.warns(RuntimeWarning, match="calibration_method"):
            fields = runner._expected_compatibility_fields(manifest_compat=compat)

        # Dropped the unverifiable one, KEPT the verifiable one.
        assert fields == {"feature_count": 98}
        # Proof this is load-bearing: without the drop, upstream hard-errors.
        with pytest.raises(ContractError):
            BacktestData.from_signal_dir(
                str(d), validate=True, expected_fields={"calibration_method": "variance_match"}
            )

    def test_all_fields_grandfathered_yields_none_not_empty_dict(self, tmp_path):
        # If EVERY expectation is grandfathered away, the result must be None,
        # not {} — validate() rejects an empty dict as a caller-side logic
        # error (SB-D). Getting this wrong turns a grandfathered legacy export
        # into a crash.
        d = _write_signal_dir(tmp_path / "sig", _contract(calibration_method=None))
        runner = _runner(d, expect={"calibration_method": "variance_match"})
        with pytest.warns(RuntimeWarning):
            fields = runner._expected_compatibility_fields(
                manifest_compat={"calibration_method": None}
            )
        assert fields is None, "empty dict would raise ValueError inside validate()"

    def test_whole_block_absence_is_passed_through_to_upstream(self, tmp_path):
        """Whole-block absence is NOT filtered here — deliberately.

        Two distinct absence cases, handled in two places:

        * whole block absent (``manifest_compat is None``) — upstream already
          grandfathers this, and emits a strictly better diagnostic
          ("version-skew check SKIPPED"). Filtering here would DESTROY that
          signal: upstream would never learn the consumer had an intent.
          §0 reuse-first — do not re-implement the upstream branch.
        * individual field absent (``None`` inside the block) — upstream
          compares with ``!=`` and would RAISE, so it must be filtered here.

        Locking the pass-through so a future "simplification" cannot quietly
        swallow the operator-visible skip warning.
        """
        d = _write_signal_dir(tmp_path / "sig", None)
        runner = _runner(d, expect={"label_strategy_hash": _KNOWN_HASH})
        fields = runner._expected_compatibility_fields(manifest_compat=None)
        assert fields == {"label_strategy_hash": _KNOWN_HASH}
        # Upstream warns and loads; it does not raise.
        BacktestData.from_signal_dir(str(d), validate=True, expected_fields=fields)

    def test_consumer_side_absence_asserts_nothing(self, tmp_path):
        # Config pins nothing → None, and the run is unaffected.
        d = _write_signal_dir(tmp_path / "sig", _contract())
        compat = json.loads((d / "signal_metadata.json").read_text())["compatibility"]
        assert _runner(d)._expected_compatibility_fields(manifest_compat=compat) is None


class TestConfigBugsFailLoud:
    """A typo must not silently assert nothing (the dead-guard pattern)."""

    def test_unknown_expect_key_raises(self, tmp_path):
        d = _write_signal_dir(tmp_path / "sig", _contract())
        runner = _runner(d, expect={"lable_strategy_hash": _KNOWN_HASH})  # typo
        with pytest.raises(ValueError, match="non-contract field"):
            runner._expected_compatibility_fields()

    def test_contradiction_between_derived_and_declared_raises(self, tmp_path):
        d = _write_signal_dir(tmp_path / "sig", _contract())
        runner = _runner(d, expect={"primary_horizon_idx": 2}, primary_horizon_idx=0)
        with pytest.raises(ValueError, match="contradicts itself"):
            runner._expected_compatibility_fields()

    def test_unknown_signals_block_key_warns(self, tmp_path):
        d = _write_signal_dir(tmp_path / "sig", _contract())
        runner = ExperimentRunner(
            {
                "experiment": {"name": "t"},
                "signals": {"dir": str(d), "expects": {}},  # typo: expects
                # Synthetic fixture, no day structure — take the recorded
                # charter override (see ExperimentRunner._run_single).
                "allow_unenforced_charter": True,
                "backtest": {"initial_capital": 100_000},
                "strategy": {"type": "regression"},
                "holding": {"type": "horizon_aligned", "hold_events": 2},
            }
        )
        with pytest.warns(RuntimeWarning, match="expects"):
            runner.run()


class TestBackwardCompatibility:
    """Historical behaviour is preserved exactly."""

    def test_no_arg_call_shape_still_works(self, tmp_path):
        # The pre-2026-08-15 signature took no arguments; four existing tests
        # in test_phase2_expected_fields_wiring.py still call it that way.
        d = _write_signal_dir(tmp_path / "sig", _contract())
        runner = _runner(d, primary_horizon_idx=1)
        assert runner._expected_compatibility_fields() == {"primary_horizon_idx": 1}

    @pytest.mark.skipif(
        not (_HISTORICAL_SIGNAL_DIR / "signal_metadata.json").exists(),
        reason="real historical export not present on this machine",
    )
    def test_real_historical_signal_dir_on_disk_still_loads(self):
        """A genuine pre-Phase-II export must not be broken by any of this.

        ``nvda_tlob_128feat_regression_h10`` carries no ``compatibility``
        block (verified 2026-08-15: 10 of 19 on-disk manifests do not).
        """
        meta = json.loads((_HISTORICAL_SIGNAL_DIR / "signal_metadata.json").read_text())
        assert meta.get("compatibility") is None, "fixture assumption changed"

        runner = _runner(_HISTORICAL_SIGNAL_DIR, expect={"feature_count": 128})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fields = runner._expected_compatibility_fields(
                manifest_compat=meta.get("compatibility")
            )
        # Whole-block absence passes through so upstream can emit its
        # "version-skew check SKIPPED" diagnostic (see
        # test_whole_block_absence_is_passed_through_to_upstream).
        assert fields == {"feature_count": 128}

        # The decisive assertion: a real pre-Phase-II export still LOADS, with
        # an expectation the manifest cannot possibly answer. It must warn,
        # not raise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = BacktestData.from_signal_dir(
                str(_HISTORICAL_SIGNAL_DIR), validate=True, expected_fields=fields
            )
        assert len(data.prices) > 0
