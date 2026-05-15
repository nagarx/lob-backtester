"""Regression tests for FIND-090 sister-site closure in script outputs.

Shipped 2026-05-15 R-19 cycle C5 (post-bundle expansion authorized by
user "5 commits + expand sister sites" Option A):

Sites closed:
- scripts/run_regression_backtest.py:374 (operator summary JSON;
  sort_keys=False per operator-facing rationale)
- scripts/run_regression_backtest.py:460 (hft-ops ledger linkage;
  sort_keys=True per cross-repo SSoT convention)
- scripts/run_readability_backtest.py:349 (hft-ops ledger linkage;
  sort_keys=True per cross-repo SSoT convention)

All 3 sites migrated from raw `with open(path, "w") + json.dump(...)`
to `hft_contracts.atomic_io.atomic_write_json(...)` SSoT. SIGKILL
mid-write hazard CLOSED.

Lock disciplines:
- AST-walk catches `with open(path, "w") as f: json.dump(...)` pattern
  regardless of line-number drift. Future contributor adding a new bare
  pattern in either script fails this test.
- Pattern is conservative: it does NOT flag `open + json.dump` outside
  a `with` statement (rare); flags ARE checked for both `json.dump` and
  `json.dumps` (the dump-to-string variant is similarly unsafe when
  followed by `f.write(...)`).

Closes Adv Y HIGH-ACTIVE finding from pre-impl Wave 2 round (2026-05-15).
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

import pytest


def _walk_non_atomic_json_writes(script_path: Path) -> List[Tuple[int, str]]:
    """AST-walk for `with open(path, "w") as f: json.dump(...)` patterns.

    Returns:
        List of (lineno, snippet) for non-atomic JSON write sites.

    Notes:
        - Matches mode-string "w" or "w+" or "wb" (write modes).
        - Caller decides ship-blocker semantics.
        - False-positive risk: `with open(..., "w")` could write
          non-json content; we filter to json.dump/json.dumps inside
          the with-body to scope correctly.
    """
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    offenders: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            ctx = item.context_expr
            if not isinstance(ctx, ast.Call):
                continue
            # Match `open(...)` direct call (Name(id='open')); skip
            # aliased opens like `io.open(...)` to keep scope tight.
            if not (isinstance(ctx.func, ast.Name) and ctx.func.id == "open"):
                continue
            # Second positional arg should be a string-constant write mode
            if len(ctx.args) < 2 or not isinstance(ctx.args[1], ast.Constant):
                continue
            mode = ctx.args[1].value
            if not isinstance(mode, str) or not mode.startswith("w"):
                continue
            # Look inside with-body for json.dump or json.dumps call
            has_json_write = False
            for body_node in node.body:
                for sub in ast.walk(body_node):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                        if (
                            sub.func.attr in ("dump", "dumps")
                            and isinstance(sub.func.value, ast.Name)
                            and sub.func.value.id == "json"
                        ):
                            has_json_write = True
                            break
                if has_json_write:
                    break
            if has_json_write:
                offenders.append(
                    (node.lineno, f"open + json.dump pattern @ L{node.lineno}")
                )
    return offenders


class TestFind090SisterScriptsAtomicWrites:
    """FIND-090 C5 sister-site closure: 3 script-output JSON writes use SSoT.

    AST-walk regression lock; line-number drift-tolerant.
    """

    @pytest.mark.parametrize(
        "script_relative",
        [
            "scripts/run_regression_backtest.py",
            "scripts/run_readability_backtest.py",
        ],
    )
    def test_script_has_no_bare_json_dump_to_open_w(
        self, script_relative: str
    ) -> None:
        """AST: zero `with open(path, "w"): json.dump(...)` patterns.

        Closes Adv Y HIGH-ACTIVE finding (2026-05-15) where:
        - run_regression_backtest.py:374 (operator summary) was non-atomic
        - run_regression_backtest.py:460 (hft-ops ledger linkage) was non-atomic
        - run_readability_backtest.py:349 (hft-ops ledger linkage) was non-atomic

        All 3 migrated to `atomic_write_json(...)` from
        `hft_contracts.atomic_io`. SIGKILL mid-write hazard closed for
        operator outputs + cross-repo hft-ops ledger linkage.
        """
        repo_root = Path(__file__).parent.parent
        script_path = repo_root / script_relative
        if not script_path.exists():
            pytest.skip(f"script not found at {script_path}")

        offenders = _walk_non_atomic_json_writes(script_path)
        assert offenders == [], (
            f"{script_relative} has non-atomic `open + json.dump` at "
            f"lines {[lineno for lineno, _ in offenders]} — FIND-090 "
            f"sister-site regression. Migrate to "
            f"`atomic_write_json(path, obj, sort_keys=..., indent=2)` "
            f"from `hft_contracts.atomic_io`."
        )

    def test_regression_script_imports_atomic_write_json(self) -> None:
        """`run_regression_backtest.py` imports atomic_write_json + AtomicWriteError.

        Defense-in-depth: if the migration ever regresses to bare
        open+json.dump, this test will FAIL even before the AST-walk
        catches it (the import surface mirrors the SSoT consumption).
        """
        repo_root = Path(__file__).parent.parent
        script_path = repo_root / "scripts" / "run_regression_backtest.py"
        if not script_path.exists():
            pytest.skip(f"script not found at {script_path}")

        src = script_path.read_text(encoding="utf-8")
        assert "atomic_write_json" in src, (
            "run_regression_backtest.py must import atomic_write_json "
            "from hft_contracts.atomic_io (FIND-090 C5 closure)"
        )
        assert "AtomicWriteError" in src, (
            "run_regression_backtest.py must import AtomicWriteError "
            "for narrow except tuple at the hft-ops ledger linkage site"
        )

    def test_readability_script_imports_atomic_write_json(self) -> None:
        """`run_readability_backtest.py` imports atomic_write_json."""
        repo_root = Path(__file__).parent.parent
        script_path = repo_root / "scripts" / "run_readability_backtest.py"
        if not script_path.exists():
            pytest.skip(f"script not found at {script_path}")

        src = script_path.read_text(encoding="utf-8")
        assert "from hft_contracts.atomic_io import atomic_write_json" in src, (
            "run_readability_backtest.py must import atomic_write_json "
            "from hft_contracts.atomic_io (FIND-090 C5 closure). "
            "Note: bare `except Exception` already catches AtomicWriteError "
            "so explicit import is not required for the script's own "
            "exception handling."
        )
