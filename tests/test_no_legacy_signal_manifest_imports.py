"""T1-2 lock test — lob-backtester src/ + scripts/ must NOT import from
``lobbacktest.data.signal_manifest`` (Phase 6 6B.5 deprecation shim).

Pre-fix (2026-05-22; closed by T1-2 of lob-backtester deep validation cycle):
- ``src/lobbacktest/experiment.py:31`` imported through legacy shim:
    from lobbacktest.data.signal_manifest import SignalManifest
- ``src/lobbacktest/engine/vectorized.py:156`` (lazy import inside
  ``from_signal_dir``) imported through legacy shim with same pattern.

Both sites triggered a ``DeprecationWarning`` via the shim's PEP 562
``__getattr__`` at every import + every test collection. Per saved
feedback memory MANDATORY pre-impl + mid-impl + pre-commit adversarial
gates, this lock test prevents regression to legacy-import-path during
the runway leading up to the shim's 2026-10-31 removal deadline.

Canonical SSoT lives at ``hft_contracts.signal_manifest`` (Phase 6 6B.5
co-move, 2026-04-17). Both ``SignalManifest`` and ``ContractError`` (and
all 11 ``_PUBLIC_NAMES`` constants) resolve via canonical path with
identical API — migration is import-source-only, zero runtime semantic
change.

Pattern mirrors:
- ``#PY-343`` (lob-backtester): hardcoded SIGNAL_*_FEATURE_INDEX SSoT
  bypass (commit ``811f167``, 2026-05-21).
- ``#PY-339`` (lob-model-trainer): silent ImportError SSoT bypass
  (commit ``cb5ac4d``, 2026-05-21).
- ``FIND-110`` (lob-backtester): ``allow_pickle=False`` AST regression
  template (commit ``20dbc8f``, 2026-05-14).

Per saved Architectural Lesson L64 (2026-05-21): AST-walk regression
tests canonical for source-level invariants. Robust to comments / quoting
/ multi-line / augmented-assigns evasions. This is the **4th consumer**
of the AST-walk template per the L66 SSoT-promotion threshold note —
when a 5th consumer emerges, hoist the walker into a shared helper.

NOTE on scope deliberately EXCLUDED:

1. ``src/lobbacktest/data/signal_manifest.py`` — IS the shim itself
   (Phase 6 6B.5 module). Its module docstring contains example import
   strings; those are pedagogical, NOT imports the AST walker would
   flag (ast.ImportFrom vs string in docstring).

2. ``tests/test_signal_manifest.py:15,394,402`` — deliberately exercises
   the shim's ``__getattr__`` deprecation chain (``TestRev2CanonicalPublicName``
   class). The shim's behavior is LOAD-BEARING for the 2026-10-31 removal
   deadline; tests that lock it must keep importing through the shim.

3. ``tests/test_signal_manifest_feature_set_ref.py:16`` — pre-2026-05-22
   test that imports through the shim for convenience (not exercising
   shim behavior). Could be migrated in a separate cycle but out of
   scope for T1-2 surgical close (saved feedback memory L73: "Cycle
   SHRINK > GROW during pre-impl gate").

4. ``tests/`` directory in general — not scanned. Production code is
   the SSoT-discipline boundary; tests may legitimately exercise
   deprecated paths to lock their behavior.

After 2026-10-31 (shim removal date documented at
``src/lobbacktest/data/signal_manifest.py:_REMOVAL_DATE``), the entire
shim file deletes and this lock becomes vacuous (import target won't
resolve). At that point this test can be retired.

See LOB_BACKTESTER_DEEP_VALIDATION_2026_05_22.md §3 T1-2 for cycle context.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [
    REPO_ROOT / "src" / "lobbacktest",
    REPO_ROOT / "scripts",
]
# Tests directory is intentionally NOT scanned: tests/test_signal_manifest.py
# deliberately exercises the shim's __getattr__ deprecation chain (lines 394,
# 402 inside TestRev2CanonicalPublicName). Including tests/ would cause
# false positives.
#
# Production code (src/) + operator-facing scripts (scripts/) form the
# SSoT-discipline boundary.

# The shim module itself is excluded — it IS the shim; its own docstring
# contains example import strings that the AST walker correctly ignores
# (those are inside docstring nodes, not ImportFrom nodes).
SHIM_MODULE_PATH = REPO_ROOT / "src" / "lobbacktest" / "data" / "signal_manifest.py"

LEGACY_SHIM_MODULE = "lobbacktest.data.signal_manifest"


def _find_legacy_shim_imports(tree: ast.AST) -> list[int]:
    """Return list of line numbers for ``from lobbacktest.data.signal_manifest
    import ...`` AST nodes.

    Robust to:
    - Comments containing ``lobbacktest.data.signal_manifest`` (not imports)
    - Docstrings containing the legacy path as example text
    - Multi-line import statements
    - Aliased imports (``from ... import SignalManifest as SM``)
    - All symbol names (``SignalManifest``, ``ContractError``,
      ``CONTENT_HASH_RE``, file-list constants, etc.)
    """
    offending_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == LEGACY_SHIM_MODULE:
                offending_lines.append(node.lineno)
        elif isinstance(node, ast.Import):
            # `import lobbacktest.data.signal_manifest` or
            # `import lobbacktest.data.signal_manifest as X` (rare but possible)
            for alias in node.names:
                if alias.name == LEGACY_SHIM_MODULE:
                    offending_lines.append(node.lineno)
                    break
    return offending_lines


def _scan_repo_for_legacy_shim_imports() -> list[str]:
    """Scan SCAN_DIRS for legacy shim imports.

    Returns list of "<relative_path>:<line>" offender strings.

    Excludes:
    - The shim module itself (src/lobbacktest/data/signal_manifest.py)
    - Files that fail to parse (SyntaxError; out of scope for this lock test)
    """
    offenders: list[str] = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for py in scan_dir.rglob("*.py"):
            # Skip the shim module itself — it documents the legacy path
            # in its docstring (pedagogical), not as an import.
            if py.resolve() == SHIM_MODULE_PATH.resolve():
                continue
            try:
                tree = ast.parse(py.read_text(), filename=str(py))
            except SyntaxError:
                # Malformed Python — out of scope for this lock test
                continue
            for lineno in _find_legacy_shim_imports(tree):
                rel = py.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{lineno}")
    return offenders


class TestNoLegacyShimImports:
    """T1-2 lock: lob-backtester src/ + scripts/ must NOT import from the
    Phase 6 6B.5 deprecation shim ``lobbacktest.data.signal_manifest``.
    Production code consumes the canonical SSoT
    ``hft_contracts.signal_manifest`` directly.
    """

    def test_no_legacy_shim_imports_in_src_or_scripts(self):
        """Verify no production source imports through the Phase 6 6B.5 shim.

        AST-based: walks both ``ast.ImportFrom`` (``from X import Y``) and
        ``ast.Import`` (``import X``) nodes. Robust to comments / docstrings
        / aliasing / multi-line imports.
        """
        offenders = _scan_repo_for_legacy_shim_imports()

        assert not offenders, (
            "T1-2 lock: no production source may import from "
            "`lobbacktest.data.signal_manifest` (Phase 6 6B.5 shim; "
            "removal deadline 2026-10-31). Per hft-rules §0 reuse-first: "
            "migrate to `from hft_contracts.signal_manifest import ...` "
            "(canonical SSoT). Offenders:\n  " + "\n  ".join(offenders)
        )

    def test_canonical_path_resolves(self):
        """Sanity: hft_contracts.signal_manifest exposes the migrated symbols.

        Validates the T1-2 fix didn't accidentally break imports + locks
        the canonical SSoT availability. Mirrors #PY-343
        ``test_spread_signal_script_imports_ssot_directly`` pattern.

        Note: ``ContractError.__module__`` resolves to
        ``hft_contracts.validation`` per REV 2 consolidation (2026-04-20;
        single canonical class re-exported across both modules). Importing
        ``from hft_contracts.signal_manifest import ContractError`` still
        works via re-export.
        """
        from hft_contracts.signal_manifest import (
            CONTENT_HASH_RE,
            ContractError,
            SignalManifest,
        )

        assert SignalManifest.__module__ == "hft_contracts.signal_manifest", (
            f"T1-2 sanity: SignalManifest must resolve to "
            f"hft_contracts.signal_manifest (canonical SSoT). Got: "
            f"{SignalManifest.__module__}"
        )
        # ContractError lives in hft_contracts.validation post-REV-2
        # consolidation but is re-exported via hft_contracts.signal_manifest
        # for back-compat. The import statement above must succeed.
        assert ContractError is not None
        assert CONTENT_HASH_RE is not None

    def test_shim_removal_deadline_documented(self):
        """Verify the shim's ``_REMOVAL_DATE`` constant is still set to
        the canonical 2026-10-31 deadline.

        If the deadline is moved, this test catches that — and the
        documentation reflecting it across CLAUDE.md banners, the shim's
        own docstring, and this test's preamble must be updated coherently.

        Per Architectural Lesson L64: SSoT-value pins prevent silent
        deadline-drift hazards. If the shim is REMOVED entirely (post
        2026-10-31), this test will fail at import-time, which is the
        correct fail-loud signal to retire this lock test.
        """
        from lobbacktest.data import signal_manifest as shim_module

        assert hasattr(shim_module, "_REMOVAL_DATE"), (
            "T1-2 sanity: shim module must declare _REMOVAL_DATE constant. "
            "If the shim is being removed entirely, retire this lock test "
            "instead of letting it silently pass."
        )
        assert shim_module._REMOVAL_DATE == "2026-10-31", (
            f"T1-2 lock: shim _REMOVAL_DATE drifted from canonical "
            f"2026-10-31 to {shim_module._REMOVAL_DATE!r}. Update CLAUDE.md "
            f"banners + shim docstring + this test's preamble in a "
            f"coordinated commit."
        )
