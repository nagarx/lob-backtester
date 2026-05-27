# Contributing to lob-backtester

> **Created**: Cycle 14 (2026-05-27) per Option δ Phase 2 LITE implementation. Per `#PY-NEW-CONSUMPTION-ENFORCEMENT` HARD-ESCALATION TIER 1 response (Cycles 12 + 13 both shipped 0% organic R-NN citation across 3 of 4 canonical surfaces).

This document captures contribution discipline specific to `lob-backtester` that is NOT in `CLAUDE.md` (which describes module structure, design patterns, key constraints, and data contract).

## Authoritative References

- **Module structure + design patterns**: `CLAUDE.md` (this repo)
- **Architecture + cost-model deep-dive**: `CODEBASE.md` (this repo)
- **Backtest round reference**: `BACKTEST_INDEX.md` (this repo, Round-N ledger)
- **Validation findings**: `VALIDATION_FINDINGS_2026_05_14.md` (Cluster D.1+E close-out artifact)
- **Audit plan**: `BACKTESTER_AUDIT_PLAN.md` (this repo)
- **Cross-pipeline findings**: `../lob-model-trainer/reports/CONSOLIDATED_FINDINGS_2026_05.md`
- **Pipeline architecture deep-dive**: `../PIPELINE_ARCHITECTURE.md` §12 (~300 LOC backtester section)
- **IBKR cost-model substance anchor**: `../IBKR-transactions-trades/COST_AUDIT_2026_03.md` (316-fill empirical audit; 43 organic BACKTEST_INDEX mentions empirically verified)
- **Theoretical backbone**: `../hft-wiki/research/theory/` (19 entries as of 2026-05-27) + `../hft-wiki/research/synthesis/` (4 entries including `hmhp_cascade_architecture` shipped Cycle 14)
- **Wiki consultation playbook**: `../hft-wiki/playbooks/record-backtest-result.md` (Cycle 14 ship)

## `wiki_consultation:` Discipline (REQUIRED post-Cycle-14)

Every NEW `## Round N` entry in `BACKTEST_INDEX.md` authored after Cycle 14 ship (2026-05-27) MUST include a `**Wiki consultation**` block citing relevant `theory:` / `synthesis:` / `FINDING-` IDs from `hft-wiki`. This is the consumer side of `#PY-NEW-CONSUMPTION-ENFORCEMENT` (TIER 2 HIGH) — Cycle 14 is the HARD-ESCALATION TIER 1 response after Cycles 12 + 13 both shipped < 20% organic per-surface citation rate.

### Field Format

```markdown
**Wiki consultation** (REQUIRED — list theory: / synthesis: / FINDING- IDs reviewed before running):
- `theory:<slug>` — <one-line justification, ≥ 20 chars>
- `synthesis:<slug>` — <one-line justification>
- `FINDING-NNN-<slug>` — <known anti-pattern context>

— OR explicit negative-result fallback:

- **None applicable** — queried `hft-wiki list theory --tag=<X>` returned 0 matches against this backtest's substance scope `<X>`.
```

### Requirements

| Requirement | Hard? | Notes |
|---|---|---|
| **Block PRESENCE** in markdown table OR as dedicated `**Wiki consultation**` / `### Wiki consultation` section | REQUIRED post-Cycle-14 | Validator WARNs if absent (default exit 0) |
| **Block CONTENT** | REQUIRED | Either ≥1 cited ID OR "None applicable" fallback |
| **Justification length** per cite | SOFT ≥ 20 chars | Validator WARNs below threshold |
| **ID resolution** | SOFT (validator opt-in via `--strict`) | Future cycle: run `hft-wiki show <id>` per cite; WARN on resolution failure |
| **Citation completeness** | NOT enforced; operator judgment | Cite IDs that ACTUALLY informed design — not symbolic compliance |
| **Cost-model anchor** | RECOMMENDED | Backtest entries SHOULD cite the IBKR audit anchor (`IBKR-transactions-trades/COST_AUDIT_2026_03.md`) for cost-aware sweep rationale |

### Grandfathering

Pre-Cycle-14 entries (Rounds 1-7 + all post-FIND-070/FIND-090/R-19/R-16a-e/R-17a/R-20 retrofits + Cluster D.1+E close-out entries) are GRANDFATHERED and exempt. The validator skips with INFO note when date extraction confirms pre-2026-05-27 authorship. Retrofit is OPTIONAL (Cycle 15+ batch retrofit deferred per `#PY-NEW-CONSUMPTION-ENFORCEMENT` closure criterion measurement decisiveness — see `../hft-wiki/ledgers/phases/PHASE-CYCLE-14-2026-05-27.md` for the formal H_Cycle14 falsifiable hypothesis + falsification clause).

### Worked Example

See `../hft-wiki/playbooks/record-backtest-result.md` §"Worked Example — Hypothetical R-21 Deep-ITM cost-aware sweep" for a 5-citation worked example using Cycle 14 entries (TLOB + HMHP cascade + Huber loss + FINDING-008 + IBKR 0DTE).

### Running the Soft Validator

```bash
cd lob-backtester
python3 scripts/check_backtest_index_completeness.py            # WARN-not-ERROR (exit 0)
python3 scripts/check_backtest_index_completeness.py --verbose  # per-entry detail
python3 scripts/check_backtest_index_completeness.py --strict   # WARN → exit 1
python3 scripts/check_backtest_index_completeness.py --json     # machine-readable output
```

Run BEFORE every commit that adds a new Round-N entry. Not yet wired to pre-commit hooks (no `.pre-commit-config.yaml` exists in this repo as of Cycle 14; Phase 3 will consider CI integration via escalation path A if H_Cycle14 falsifies — see PHASE-CYCLE-14 ledger).

## Failure Modes

- **Fake-compliance**: filling the block with stale/generic cites that don't justify design. **Cycle-close PR review** is the primary detection; `consumption_ratio.py --strict` measures organic-vs-backfill ratio over time.
- **Cost-model anchor drift**: IBKR audit was superseded by a newer empirical fill set. Always cite the CURRENT audit version. If the audit is migrated as `theory:` or `synthesis:` in a future cycle, switch to that ID.
- **Block schema-vs-impl divergence**: if the validator script's expected format diverges from documented format here, the validator wins (it's the SSoT for what counts as "compliant"). Update this doc to match the validator.
- **Validator unavailable in CI**: validator is opt-in operator-run helper; no CI gate. Operators MUST run manually for now.

## What NOT to Cite

- Cost-model recalibration runs that don't propose new strategy logic (e.g., Round 4 BSM theta correction) — use "None applicable" fallback or omit the Wiki consultation block (validator grandfathers entries that genuinely have no theory anchor).
- Pure infrastructure changes (FIND-NNN closure validation runs, atomic-write SSoT closure follow-ons).
- Hot-path live-incident hotfixes (per hft-rules §13 exception clause).
- Trivial bug fixes (`fix: typo in BACKTEST_INDEX.md row`).

## Cycle 14 Reasoning

This `wiki_consultation:` discipline was extended to BACKTEST surface via Cycle 14 Option δ Phase 2 LITE implementation (#PY-NEW-CONSUMPTION-ENFORCEMENT HARD-ESCALATION TIER 1 response). Phase 1 (Cycle 11, lob-model-trainer EXPERIMENT_INDEX) shipped the template forcing function pattern; Cycles 12 + 13 then shipped 2 substance entries (sample_weights + 5-path Framework) but achieved 0% organic R-NN citation on per-surface basis (3 of 4 canonical surfaces at R = 0.000).

Cycle 14 designs this consumer-side forcing function for BACKTEST_INDEX so that Cycle 15+ R-NN authors can produce ORGANIC citations (not backfilled). If Cycles 15+16 ship 0 organic citations meeting H_Cycle14 criteria (a)-(d), Cycle 17 = FALSIFICATION TRIGGER per closure criterion.

**EXPORT_INDEX excluded this cycle by DESIGN** (not oversight): producer-side data lineage surface, NOT consumer-side R-NN authoring surface. Re-evaluate in Cycle 16+ if the surface evolves into hypothesis-driven authorship.

## Related Documents

- `BACKTEST_INDEX.md` — primary target ledger; per-entry template embedded at top of file
- `../hft-wiki/playbooks/record-backtest-result.md` — operator workflow playbook (Cycle 14 sibling of record-experiment-result.md)
- `../hft-wiki/scripts/consumption_ratio.py` — operator-runnable Goodhart trajectory measurement
- `../hft-wiki/ledgers/phases/PHASE-CYCLE-14-2026-05-27.md` — Cycle 14 phase ledger with H_Cycle14 pre-registered falsifiable hypothesis + falsification clause + Cycle 15/16 measurement criterion
- `../PHASE_P_BACKLOG.md #PY-NEW-CONSUMPTION-ENFORCEMENT` — closure criterion (root-level local-only)
- `../lob-model-trainer/CONTRIBUTING.md` — sibling discipline doc (Cycle 11 Phase 1 EXPERIMENT_INDEX target)
- `../hft-wiki/meta/2026-05-26-option-delta-design.md` — original design doc (Cycle 9 DRAFT + Cycle 10 L106 in-place revision)
