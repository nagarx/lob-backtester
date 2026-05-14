"""Lesson #14 lock test — ``test_twap.py`` must remain module-level skipped.

The TWAPStrategy was empirically failed at R2 (see ``BACKTEST_INDEX.md``) due
to engine C2 incompatibility. Tests in ``test_twap.py`` are preserved for
future re-enablement but MUST stay skipped via ``pytestmark`` at module scope.

This test enforces the discipline: removing the skip marker without
re-validating C2 compatibility would trip Lesson #14. See
``lob-backtester/VALIDATION_FINDINGS_2026_05_14.md`` Appendix A row #14 +
``DESIGN_CLUSTER_D1_E_2026_05_14.md`` §4.2.
"""

import importlib


class TestTwapSkipDiscipline:
    """Lesson #14 lock: module-level skip marker on ``test_twap.py``."""

    def test_twap_module_has_pytestmark_skip(self):
        """``test_twap.py`` must carry ``pytestmark = pytest.mark.skip(...)`` at module scope."""
        mod = importlib.import_module("tests.test_strategies.test_twap")
        pytestmark = getattr(mod, "pytestmark", None)
        assert pytestmark is not None, (
            "Lesson #14: tests/test_strategies/test_twap.py MUST define "
            "`pytestmark = pytest.mark.skip(...)` at module scope. "
            "Removing this skip without re-validating C2 compatibility would trip "
            "the encoded lesson. See VALIDATION_FINDINGS_2026_05_14.md Appendix A."
        )

        # pytestmark may be a single mark or a list — handle both forms
        marks = pytestmark if isinstance(pytestmark, list) else [pytestmark]
        skip_marks = [m for m in marks if m.name == "skip"]
        assert skip_marks, (
            "Lesson #14: tests/test_strategies/test_twap.py pytestmark must include a "
            "`pytest.mark.skip` mark (got marks=%s)" % [m.name for m in marks]
        )

        reason = skip_marks[0].kwargs.get("reason", "")
        assert "C2" in reason, (
            "Lesson #14: skip reason must reference the C2 engine incompatibility "
            "(got reason=%r)" % reason
        )
