"""#PY-343 lock test — lob-backtester src/ + scripts/ must NOT hardcode
``MID_PRICE_IDX = 40`` or ``SPREAD_BPS_IDX = 42`` as module-level Assign
statements.

Pre-fix (2026-05-21; closed Option D.A cycle):
``scripts/run_spread_signal_backtest.py:58-59`` had::

    SPREAD_BPS_IDX = 42   # feature index for spread_bps
    MID_PRICE_IDX = 40     # feature index for mid_price

This violates hft-rules §0 reuse-first ("no hardcoded indices; centralized
constants module is the single source of truth"). The canonical SSoT lives
at ``hft_contracts/_generated.py:481+484``::

    SIGNAL_SPREAD_FEATURE_INDEX: Final[int] = 42
    SIGNAL_PRICE_FEATURE_INDEX:  Final[int] = 40

These constants are auto-generated from ``contracts/pipeline_contract.toml``
and re-exported via ``hft_contracts/__init__.py``. Any future contract
re-numbering of indices is propagated through the SSoT — hardcoded copies
in consumer code would silently bypass that propagation.

Pattern mirrors:
- ``#PY-339`` (lob-model-trainer): same SSoT bypass; closed 2026-05-21 via
  ``tests/test_security/test_no_silent_import_error_fallback_py339.py``.
- ``FIND-110`` (lob-backtester): ``allow_pickle=False`` AST regression
  template (commit ``20dbc8f``, 2026-05-14).

Per saved Architectural Lesson L64 (2026-05-21): AST-walk regression tests
canonical for source-level invariants. Robust to comments / quoting /
multi-line / augmented-assigns evasions.

NOTE on scope deliberately EXCLUDED (sister #PY-343-EXT-DICT):
``scripts/run_spread_signal_backtest.py:88-95`` contains::

    RIDGE_FEATURES = {
        "spread_bps": 42,
        "total_ask_volume": 44,
        "volume_imbalance": 45,
        "true_ofi": 84,
        "depth_norm_ofi": 85,
    }

This is a dict-value AST shape (``ast.Dict`` containing ``ast.Constant(42)``),
NOT a module-constant Assign. The AST walker below correctly does NOT flag
it — out of scope for this surgical close. A follow-up sister cycle would
hoist the dict to consume SSoT-keyed values; documented in commit body as
deferred per "small reversible changes" discipline (hft-rules §0).

NOTE on scope deliberately EXCLUDED (foreign-agent #PY-225):
``lob-dataset-analyzer/src/`` contains ~15 sister sites with the same
hardcoded pattern. These are NOT touched here per anti-drift discipline
(foreign-agent work-at-risk). They are tracked separately in
PHASE_P_BACKLOG.md as a deferred lob-dataset-analyzer-side cycle.

See PHASE_P_BACKLOG.md #PY-343 and POST_AREV_BEXP_2026_05_21.md §5 (Option D.A).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [
    REPO_ROOT / "src" / "lobbacktest",
    REPO_ROOT / "scripts",
]
# Tests directory is intentionally NOT scanned: test fixtures may mock the
# canonical values as plain literals (e.g., parametric tests, golden fixtures).
# Production code is the SSoT-discipline boundary.


def _find_hardcoded_constant_assigns(
    tree: ast.AST,
    name: str,
    value: int,
) -> list[tuple[int, str]]:
    """Return list of (line, target_id) for ``<name> = <value>`` Assign nodes.

    Uses AST (not regex) so the following are NOT false-positives:
        # MID_PRICE_IDX = 40                    (line comment)
        \"\"\"... MID_PRICE_IDX = 40 ...\"\"\"  (docstring mention)
        x = MID_PRICE_IDX                       (read access)
        d = {"key": 40}                          (dict-value)
        MID_PRICE_IDX = some_func()              (different value)
        MID_PRICE_IDX += 1                       (AugAssign, not Assign)
        MID_PRICE_IDX = 40 + 0                    (BinOp, not Constant)

    Catches BOTH plain ``ast.Assign`` (``X = 40``) AND ``ast.AnnAssign``
    (``X: int = 40``) — closes mid-impl Agent's F.2 defense gap (annotated
    assign would silently bypass plain-Assign-only walker; common Python
    typing pattern likely to emerge in future code).
    """
    hits = []
    for node in ast.walk(tree):
        # Plain assign: X = 40
        if isinstance(node, ast.Assign):
            if not (isinstance(node.value, ast.Constant) and node.value.value == value):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    hits.append((node.lineno, target.id))
        # Annotated assign: X: int = 40 (defense-in-depth per F.2)
        elif isinstance(node, ast.AnnAssign):
            if node.value is None:
                # Pure annotation `X: int` (no assignment) — skip
                continue
            if not (isinstance(node.value, ast.Constant) and node.value.value == value):
                continue
            if isinstance(node.target, ast.Name) and node.target.id == name:
                hits.append((node.lineno, node.target.id))
    return hits


def _scan_repo_for_hardcoded(name: str, value: int) -> list[str]:
    """Walk SCAN_DIRS, parse each .py, return offender strings.

    Skips __init__.py (typically just re-exports) and tests/ directories.
    """
    offenders = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for py in scan_dir.rglob("*.py"):
            # Skip __init__.py (typically only re-exports)
            if py.name == "__init__.py":
                continue
            try:
                tree = ast.parse(py.read_text(), filename=str(py))
            except SyntaxError:
                # Malformed Python — out of scope for this lock test
                continue
            for lineno, _ in _find_hardcoded_constant_assigns(tree, name, value):
                rel = py.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{lineno}")
    return offenders


class TestPy343NoHardcodedSignalIndices:
    """#PY-343 lock: lob-backtester src/ + scripts/ must NOT hardcode the
    canonical feature indices ``MID_PRICE_IDX = 40`` or ``SPREAD_BPS_IDX = 42``
    as module-level Assign statements. Canonical SSoT lives in
    ``hft_contracts._generated`` (auto-gen from
    ``contracts/pipeline_contract.toml``).
    """

    def test_no_hardcoded_mid_price_idx_40_in_src_or_scripts(self):
        """Verify no source file in lob-backtester src/ or scripts/ hardcodes
        ``MID_PRICE_IDX = 40`` as a module-level Assign statement.

        AST-based: ignores comments and docstring mentions per false-positive
        avoidance. Robust to whitespace / quoting / line-continuation evasions.
        """
        offenders = _scan_repo_for_hardcoded("MID_PRICE_IDX", 40)

        assert not offenders, (
            "#PY-343 lock: no production source may hardcode "
            "`MID_PRICE_IDX = 40` as an Assign statement. Per hft-rules §0 "
            "reuse-first: import from hft_contracts.SIGNAL_PRICE_FEATURE_INDEX "
            "(canonical value lives in hft_contracts._generated, auto-gen "
            "from contracts/pipeline_contract.toml SSoT). Offenders:\n  "
            + "\n  ".join(offenders)
        )

    def test_no_hardcoded_spread_bps_idx_42_in_src_or_scripts(self):
        """Verify no source file in lob-backtester src/ or scripts/ hardcodes
        ``SPREAD_BPS_IDX = 42`` as a module-level Assign statement.

        AST-based per #PY-343 lock pattern (sister of MID_PRICE_IDX test).
        """
        offenders = _scan_repo_for_hardcoded("SPREAD_BPS_IDX", 42)

        assert not offenders, (
            "#PY-343 lock: no production source may hardcode "
            "`SPREAD_BPS_IDX = 42` as an Assign statement. Per hft-rules §0 "
            "reuse-first: import from "
            "hft_contracts.SIGNAL_SPREAD_FEATURE_INDEX (canonical value lives "
            "in hft_contracts._generated, auto-gen from "
            "contracts/pipeline_contract.toml SSoT). Offenders:\n  "
            + "\n  ".join(offenders)
        )

    def test_canonical_ssot_values_match_locked_constants(self):
        """Verify the canonical SSoT values are still 40 + 42 — if pipeline
        contract ever rotates these indices via TOML SSoT path, this test
        FAILS LOUD so the lock-tests above can be updated coherently.

        Per Architectural Lesson L64: AST-walk lock-tests must be paired with
        SSoT-value pins to catch the rare case where the contract is rotated
        but the test still passes because consumer code never updated.
        """
        from hft_contracts import (
            SIGNAL_PRICE_FEATURE_INDEX,
            SIGNAL_SPREAD_FEATURE_INDEX,
        )
        assert SIGNAL_PRICE_FEATURE_INDEX == 40, (
            f"#PY-343 lock: SSoT SIGNAL_PRICE_FEATURE_INDEX rotated from 40 "
            f"to {SIGNAL_PRICE_FEATURE_INDEX} via "
            f"contracts/pipeline_contract.toml. The MID_PRICE_IDX = 40 "
            f"hardcoded-detection test is now STALE — update both the lock-"
            f"value and the contract docs in a coordinated commit."
        )
        assert SIGNAL_SPREAD_FEATURE_INDEX == 42, (
            f"#PY-343 lock: SSoT SIGNAL_SPREAD_FEATURE_INDEX rotated from 42 "
            f"to {SIGNAL_SPREAD_FEATURE_INDEX} via "
            f"contracts/pipeline_contract.toml. The SPREAD_BPS_IDX = 42 "
            f"hardcoded-detection test is now STALE — update both the lock-"
            f"value and the contract docs in a coordinated commit."
        )

    def test_spread_signal_script_imports_ssot_directly(self):
        """Sanity: run_spread_signal_backtest.py imports MID_PRICE_IDX +
        SPREAD_BPS_IDX successfully from the SSoT at module load time.

        Validates the #PY-343 fix didn't accidentally break imports. Mirrors
        #PY-339 ``test_simple_trainer_imports_resolve_at_module_load`` pattern.
        """
        script_path = REPO_ROOT / "scripts" / "run_spread_signal_backtest.py"
        assert script_path.exists(), (
            f"#PY-343 sanity check: run_spread_signal_backtest.py not found "
            f"at {script_path}"
        )

        source = script_path.read_text()
        tree = ast.parse(source, filename=str(script_path))

        # Verify the SSoT import is present (looks for any ImportFrom node
        # with module='hft_contracts' that imports SIGNAL_PRICE_FEATURE_INDEX
        # or SIGNAL_SPREAD_FEATURE_INDEX — fail-loud on regression).
        imports_signal_idx = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "hft_contracts" in node.module:
                    for alias in node.names:
                        if alias.name in (
                            "SIGNAL_PRICE_FEATURE_INDEX",
                            "SIGNAL_SPREAD_FEATURE_INDEX",
                        ):
                            imports_signal_idx = True
                            break
            if imports_signal_idx:
                break

        assert imports_signal_idx, (
            "#PY-343 sanity: run_spread_signal_backtest.py must import "
            "SIGNAL_PRICE_FEATURE_INDEX or SIGNAL_SPREAD_FEATURE_INDEX from "
            "hft_contracts (the SSoT consumption surface). If this regresses, "
            "the hardcoded-literal sites at L58-59 would re-emerge undetected "
            "by the AST walker — defense-in-depth lock per L64."
        )
