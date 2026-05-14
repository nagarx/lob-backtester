"""FIND-110 lock test — every ``np.load()`` must pass ``allow_pickle=False``.

Without ``allow_pickle=False``, ``np.load()`` accepts pickled Python objects in
``.npy`` files, opening the door to remote code execution if a ``.npy`` file
arrives from an untrusted source (CVE-class hazard per hft-rules §8). This
test scans every ``np.load(`` callsite in ``src/``, ``tests/``, and
``scripts/`` and asserts the ``allow_pickle=False`` keyword appears within
each call's argument span.

Known limitation: the test scans for the textual pattern ``np.load(``. It does
not catch aliases like ``from numpy import load as _l; _l(...)`` or
fully-qualified ``numpy.load(...)``. The lob-backtester codebase convention is
``import numpy as np`` (verified at fix time via ``grep -rE
"from numpy import .*load|numpy\\.load\\("``, returned zero hits). If a future
contributor introduces an alias, extend the patterns below.

See ``VALIDATION_FINDINGS_2026_05_14.md`` FIND-110 and Appendix A lesson #29.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NP_LOAD_RE = re.compile(r"np\.load\s*\(")
ALLOW_PICKLE_FALSE_RE = re.compile(r"allow_pickle\s*=\s*False")


def _extract_call_span(text: str, open_paren_idx: int) -> str:
    """Return the call's argument span: from ``(`` through the matching ``)``.

    Handles nested parens via depth tracking. Does not attempt to parse string
    literals containing unbalanced parens (negligible for ``np.load`` calls,
    which take a path-like first argument).
    """
    depth = 0
    end = open_paren_idx
    for i in range(open_paren_idx, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    return text[open_paren_idx : end + 1]


class TestFind110AllowPickleFalseLock:
    """FIND-110 lock: every ``np.load()`` must pass ``allow_pickle=False``."""

    def test_every_np_load_passes_allow_pickle_false(self):
        """No ``np.load()`` callsite in src/, tests/, scripts/ may omit ``allow_pickle=False``.

        Removing ``allow_pickle=False`` from any callsite re-opens the
        pickle-RCE vector closed by FIND-110 (commit on 2026-05-14). If this
        test fails, the listed offenders MUST be hardened before merge.
        """
        offenders = []
        for sub in ("src", "tests", "scripts"):
            base = REPO_ROOT / sub
            if not base.exists():
                continue
            for py in base.rglob("*.py"):
                # Skip THIS file (would self-match the docstring + regex literals).
                if py.name == "test_np_load_allow_pickle_false.py":
                    continue
                text = py.read_text()
                for m in NP_LOAD_RE.finditer(text):
                    # m.end() - 1 points at the `(` character.
                    span = _extract_call_span(text, m.end() - 1)
                    if not ALLOW_PICKLE_FALSE_RE.search(span):
                        line = text[: m.start()].count("\n") + 1
                        offenders.append(
                            f"{py.relative_to(REPO_ROOT)}:{line}"
                        )
        assert not offenders, (
            "FIND-110 lock: every np.load() callsite must pass "
            "allow_pickle=False (prevents pickle-RCE on malicious .npy "
            "files; hft-rules §8). Offenders:\n  "
            + "\n  ".join(offenders)
        )
