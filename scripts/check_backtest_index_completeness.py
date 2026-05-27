#!/usr/bin/env python3
# PRODUCTION INFRA — ledger validator (not an experiment)
"""Soft validator for wiki_consultation: discipline in BACKTEST_INDEX.md.

Per Cycle 14 Option δ Phase 2 LITE implementation (#PY-NEW-CONSUMPTION-ENFORCEMENT
HARD-ESCALATION TIER 1 response).

Mirrors lob-model-trainer/scripts/check_experiment_index_completeness.py
(Cycle 11 Option δ Phase 1 ship) with these BACKTEST-specific adaptations:

- Entry boundary regex matches `## Round N` (top-level header) NOT `### R-NN`
- Round number suffix-aware: matches R-16a / R-16c / R-19a / R-19b / R-17a etc.
- Date locate window widened to 800 chars (BACKTEST headers run longer than
  EXPERIMENT_INDEX with strategy / cost-model / signal-source sub-fields)
- Grandfather threshold: CYCLE_14_SHIP_DATE = "2026-05-27" (separate from
  Cycle 11 EXPERIMENT_INDEX threshold; entries dated before 2026-05-27 are
  GRANDFATHERED — covers Rounds 1-7 + all post-FIND/R-XX retrofits)

DESIGN PRINCIPLES (parity with sister validator):
- Exit code 0 (WARN-not-ERROR) by default
- --verbose for per-entry detail
- --json for machine-readable output
- --strict to escalate WARN -> exit 1 (opt-in by operator)

USAGE:
    cd lob-backtester
    python3 scripts/check_backtest_index_completeness.py
    python3 scripts/check_backtest_index_completeness.py --verbose
    python3 scripts/check_backtest_index_completeness.py --json
    python3 scripts/check_backtest_index_completeness.py --strict

DOCUMENTED IN:
    - lob-backtester/CONTRIBUTING.md (field discipline)
    - lob-backtester/BACKTEST_INDEX.md (per-entry template at top)
    - hft-wiki/playbooks/record-backtest-result.md (operator workflow)

PRE-REGISTERED HYPOTHESIS H_Cycle14:
    See hft-wiki/ledgers/phases/PHASE-CYCLE-14-2026-05-27.md for the formal
    falsifiable hypothesis + falsification clause + Cycle 15/16 measurement
    criterion. Cycle 17 = falsification trigger if Cycles 15 + 16 both ship
    with 0 organic citations meeting criteria (a)-(d).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Cycle 14 ship date — entries dated >= this require wiki_consultation
CYCLE_14_SHIP_DATE = "2026-05-27"

# Minimum justification length per cite (SOFT — WARN below; parity with sister)
MIN_JUSTIFICATION_CHARS = 20

# Regex for citable IDs (verbatim parity with sister validator)
CITATION_REGEX = re.compile(r"`(theory|synthesis|FINDING)[-:][a-z0-9_]+(?:-[a-z0-9_]+)*`?", re.IGNORECASE)

# Regex for date in entry headers — BACKTEST uses formats like:
#   "## Round 3: IBKR-Validated + BSM Theta (2026-03-14)"
#   "## Round 17a: ... (PASS, 2026-05-14)"
#   "## Round 19a: ... (2026-05-04 night)"
#   "## Round 20: ... (STRONGEST ..., 2026-05-05 morning)"
# Slightly broader than EXPERIMENT_INDEX (matches REFUTE / PASS / STRONGEST verdict prefixes
# OR a comma-separated verdict before the date).
DATE_REGEX = re.compile(r"\((?:.*?,\s*)?(\d{4}-\d{2}-\d{2})\)")

# Regex for Round-N entry headers (BACKTEST schema).
# Matches:
#   "## Round 1: title"      (numeric)
#   "## Round 16a: title"    (lowercase suffix)
#   "## Round 17a: title"    (alphanumeric suffix)
#   "## Round 19b: title"
# Round numbers can have suffix [a-z]? per actual BACKTEST_INDEX.md usage
# (verified: L1171 R-17a, L1403 R-19a, L1554 R-19b at design time).
ROUND_HEADER_REGEX = re.compile(r"^##\s+Round\s+([0-9]+[a-z]?)[\s:—-]", re.MULTILINE)


@dataclass
class EntryAuditResult:
    entry_id: str
    line_number: int
    date_string: Optional[str] = None
    grandfathered: bool = False
    has_wiki_consultation_block: bool = False
    citations_found: list[str] = field(default_factory=list)
    has_negative_fallback: bool = False
    short_justifications: list[tuple[str, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_entries(text: str) -> list[tuple[str, int, int]]:
    """Returns list of (entry_id, start_offset, end_offset) per ## Round N entry."""
    matches = list(ROUND_HEADER_REGEX.finditer(text))
    entries = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entries.append((m.group(1), start, end))
    return entries


def find_line_number(text: str, offset: int) -> int:
    return text[:offset].count("\n") + 1


def extract_date(entry_text: str) -> Optional[str]:
    """Extract YYYY-MM-DD from entry header + immediate context.

    BACKTEST headers can run longer than EXPERIMENT_INDEX (strategy + cost-model
    + signal-source sub-fields), so search window widened to 800 chars.
    """
    m = DATE_REGEX.search(entry_text[:800])
    return m.group(1) if m else None


def audit_entry(entry_id: str, entry_text: str, line_number: int) -> EntryAuditResult:
    result = EntryAuditResult(entry_id=entry_id, line_number=line_number)

    # Determine date + grandfathered status
    date_str = extract_date(entry_text)
    result.date_string = date_str
    if date_str is None:
        # Unable to date — grandfather by default (safer than false-positive)
        result.grandfathered = True
        result.warnings.append("INFO: could not extract date from entry header; grandfathered by default")
        return result

    if date_str < CYCLE_14_SHIP_DATE:
        result.grandfathered = True
        return result

    # Post-Cycle-14: must have **Wiki consultation** block
    # Accepts inline block ("**Wiki consultation**" followed by content)
    # OR table row ("| **Wiki consultation** | ..." )
    # OR section header ("### Wiki consultation ...")
    has_block = "**Wiki consultation**" in entry_text or "Wiki consultation" in entry_text
    result.has_wiki_consultation_block = has_block
    if not has_block:
        result.warnings.append("WARN: missing **Wiki consultation** block (REQUIRED post-Cycle-14)")
        return result

    # Check for citations OR negative-result fallback
    citations = CITATION_REGEX.findall(entry_text)
    result.citations_found = citations

    # Detect negative fallback (e.g., "None applicable — queried ...")
    has_fallback = bool(re.search(r"None applicable.*queried.*list", entry_text, re.IGNORECASE))
    result.has_negative_fallback = has_fallback

    if not citations and not has_fallback:
        result.warnings.append(
            "WARN: **Wiki consultation** block present but contains no citations AND no 'None applicable' fallback"
        )
        return result

    # Check justification length per cite
    # Heuristic: find each cite + the text following it on the same line, until newline
    cite_lines = re.findall(
        r"`(?:theory|synthesis|FINDING)[-:][a-z0-9_]+(?:-[a-z0-9_]+)*`?\s*[—–-]\s*(.+?)(?:\n|$)",
        entry_text,
        re.IGNORECASE,
    )
    for justification in cite_lines:
        if len(justification.strip()) < MIN_JUSTIFICATION_CHARS:
            result.short_justifications.append((justification.strip()[:80], len(justification.strip())))

    if result.short_justifications:
        result.warnings.append(
            f"WARN: {len(result.short_justifications)} cite(s) have justification < {MIN_JUSTIFICATION_CHARS} chars"
        )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--backtest-index",
        default="BACKTEST_INDEX.md",
        help="Path to BACKTEST_INDEX.md (default: ./BACKTEST_INDEX.md)",
    )
    parser.add_argument("--verbose", action="store_true", help="Per-entry detail")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any WARN found (default: always exit 0)")
    args = parser.parse_args()

    path = Path(args.backtest_index)
    if not path.exists():
        print(f"ERROR: {path} not found. Run from lob-backtester/ root.", file=sys.stderr)
        return 2

    text = path.read_text()
    entries = parse_entries(text)
    results = []

    for entry_id, start, end in entries:
        entry_text = text[start:end]
        line_number = find_line_number(text, start)
        result = audit_entry(entry_id, entry_text, line_number)
        results.append(result)

    # Summary
    total = len(results)
    grandfathered = sum(1 for r in results if r.grandfathered)
    post_c14 = total - grandfathered
    with_warns = sum(1 for r in results if any(w.startswith("WARN") for w in r.warnings))
    total_warns = sum(len([w for w in r.warnings if w.startswith("WARN")]) for r in results)

    if args.json:
        out = {
            "summary": {
                "total_entries": total,
                "grandfathered": grandfathered,
                "post_cycle_14": post_c14,
                "entries_with_warnings": with_warns,
                "total_warnings": total_warns,
            },
            "entries": [
                {
                    "id": r.entry_id,
                    "line": r.line_number,
                    "date": r.date_string,
                    "grandfathered": r.grandfathered,
                    "has_wiki_block": r.has_wiki_consultation_block,
                    "citations": r.citations_found,
                    "has_fallback": r.has_negative_fallback,
                    "short_justifications": [j for j, _ in r.short_justifications],
                    "warnings": r.warnings,
                }
                for r in results
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"=== check_backtest_index_completeness.py — Cycle 14 Option δ Phase 2 LITE validator ===")
        print(f"File: {path}")
        print(f"Cycle 14 ship date (grandfather threshold): {CYCLE_14_SHIP_DATE}")
        print(f"")
        print(f"Total entries: {total}")
        print(f"  Grandfathered (pre-Cycle-14): {grandfathered}")
        print(f"  Post-Cycle-14: {post_c14}")
        print(f"  Entries with warnings: {with_warns}")
        print(f"  Total warnings: {total_warns}")
        print(f"")

        if args.verbose or with_warns > 0:
            print("Per-entry detail (entries with warnings or all if --verbose):")
            for r in results:
                if args.verbose or r.warnings:
                    status = "[grandfathered]" if r.grandfathered else ""
                    block = "block present" if r.has_wiki_consultation_block else "NO BLOCK"
                    cites = f"{len(r.citations_found)} cites"
                    fb = " + fallback" if r.has_negative_fallback else ""
                    print(f"  Round {r.entry_id} (L{r.line_number}, date={r.date_string}) {status}")
                    if not r.grandfathered:
                        print(f"    {block}, {cites}{fb}")
                    for w in r.warnings:
                        print(f"    {w}")

    if args.strict and total_warns > 0:
        print(f"\nSTRICT: {total_warns} warnings; exit 1", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
