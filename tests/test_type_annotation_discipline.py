"""#PY-228 lock test — no ``Dict[str, any]`` (lowercase ``any``) annotations.

The Python built-in ``any()`` is a callable, NOT a type. Using ``Dict[str, any]``
in a type annotation is silently accepted by ``typing.get_type_hints`` but
produces a meaningless annotation (``Dict[str, builtins.any]``). Static type
checkers (mypy, pyright) flag it; Pydantic v2 rejects it; future migration
work blocks until every occurrence is fixed.

Per hft-rules §0 small-reversible-changes and §11 "docs reflect code exactly",
this lock test scans every ``.py`` file in ``src/``, ``tests/``, and
``scripts/`` via AST walk and asserts ZERO ``Dict[str, any]`` (lowercase)
annotations remain.

**Scope gap (intentional)**: this lock targets uppercase ``typing.Dict``
subscripts. PEP 585 lowercase ``dict[str, any]`` forms are NOT caught — the
lob-backtester codebase convention is uppercase ``Dict`` via ``from typing
import Dict`` (verified at fix time). Cluster F.2 Pydantic migration will
sweep PEP 585 holistically; extending this lock to cover ``dict`` is deferred
until that migration lands.

See ``VALIDATION_FINDINGS_2026_05_14.md`` FIND-067 (sister: dead
``ComparisonConfig`` deletion) and ``PHASE_P_BACKLOG.md`` #PY-228 +
Appendix A lesson #30.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_lowercase_any_in_dict_subscript(tree: ast.AST):
    """Yield (lineno, col_offset) for every ``Dict[str, any]`` pattern.

    Detects the lowercase ``any`` (built-in callable) used in second position
    of a ``Dict[...]`` subscript. Does NOT flag legitimate uses of ``any()``
    as a function call.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        # Subscript value must be `Dict`
        value = node.value
        if not (isinstance(value, ast.Name) and value.id == "Dict"):
            continue
        # Slice must be a tuple-shape `(str, X)`
        slice_node = node.slice
        # Python 3.9+: slice is the inner expression directly; for tuple,
        # it's an ast.Tuple
        if not isinstance(slice_node, ast.Tuple):
            continue
        if len(slice_node.elts) != 2:
            continue
        second = slice_node.elts[1]
        if isinstance(second, ast.Name) and second.id == "any":
            yield (node.lineno, node.col_offset)


class TestPy228TypeAnnotationDisciplineLock:
    """#PY-228 lock: no ``Dict[str, any]`` (lowercase ``any``) annotations."""

    def test_no_lowercase_any_in_dict_annotations(self):
        """Every ``Dict[str, ...]`` annotation must use ``typing.Any`` (uppercase).

        Lowercase ``any`` refers to the Python built-in callable, which is not
        a type. Static analyzers flag it; Pydantic v2 rejects it. This lock
        prevents regression of the FIND-067 + #PY-228 type-hygiene work.
        """
        offenders = []
        for sub in ("src", "tests", "scripts"):
            base = REPO_ROOT / sub
            if not base.exists():
                continue
            for py in base.rglob("*.py"):
                # Skip THIS file (own docstring discusses the pattern).
                if py.name == "test_type_annotation_discipline.py":
                    continue
                try:
                    tree = ast.parse(py.read_text())
                except SyntaxError:
                    # Skip files we can't parse (none expected in production
                    # code; an intentional fixture might fail parse).
                    continue
                for lineno, col in _find_lowercase_any_in_dict_subscript(tree):
                    offenders.append(
                        f"{py.relative_to(REPO_ROOT)}:{lineno}:{col}"
                    )
        assert not offenders, (
            "#PY-228 lock: no `Dict[str, any]` (lowercase `any`) annotations "
            "allowed. Use `typing.Any` (uppercase) instead. Offenders:\n  "
            + "\n  ".join(offenders)
        )
