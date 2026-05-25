# Round 9 Contract

## Round Objective

Closure: land **task-h1 (validate-patch reproducibility harness, AC-10)** + **task-h2 (final grep audit for plan-progress markers, AC-11)**. After this round, only task-e3 (Dashboard parity, AC-14) remains; everything else closes.

## In-Scope Tasks

| Task | AC | Tag | Notes |
|------|----|----|-------|
| task-h1 | AC-10 | v3.7.248 | `scripts/validate-patch.sh` — shell driver that runs the relevant pytest subset for each v3.7.* tag, normalizes non-deterministic output (timestamps, rootdir absolute paths, durations), and writes a normalized `data/backtest_history/<tag>/VALIDATION.md`. Companion `tests/test_validate_patch.py` exercises the normalization helper on a recorded pytest output fixture to confirm byte-identical reproducibility. |
| task-h2 | AC-11 | analyze | `scripts/eval/audit_plan_markers.sh` — git-grep across files modified in v3.7.233..HEAD for forbidden plan-progress markers (`AC-\d`, `Milestone:`, `Phase [A-Z]:`, `Step \d:`). The grep treats v3.7.* changelog version markers + bitlesson IDs (`BL-YYYYMMDD-...`) + plan/contract/summary documents under `.humanize/` as allowed. Failure → exits non-zero. Reports findings to `data/backtest_history/v3.7.248_plan_marker_audit/REPORT.md`. |

## Out-of-Scope

- task-e3 (AC-14): Dashboard parity wrapper — separate round.

## Verification Plan

### task-h1
- pytest `tests/test_validate_patch.py`:
  - normalization helper strips `passed in N.NNs` durations
  - normalization strips absolute `/Users/...` paths
  - normalization preserves PASSED/FAILED lines verbatim
  - two consecutive runs against an unchanged fixture produce byte-identical normalized output
- Smoke: run `scripts/validate-patch.sh v3.7.233` against the current tree and confirm exit 0.

### task-h2
- The audit script run against current HEAD must:
  - Find zero forbidden markers in source code (`core/`, `scripts/eval/`, `scripts/backtest/`, `tests/`)
  - Allow markers in `.humanize/` (plan documents, contracts, summaries) — exempted via path filter
  - Allow markers in `data/backtest_history/` (audit reports) — exempted
  - Exit 0 on clean tree; non-zero on dirty tree (verified by injecting a synthetic violation, then reverting)

## Commit Discipline

- `v3.7.248`: task-h1 + task-h2 (closure pair; small surface, no separate tags needed)
- gate exit on h2 grep determines whether AC-11 LANDED status is finalized

## Bitlessons Consulted

- `BL-20260525-gate-rule-clarity` (Round 8): irrelevant to closure tooling.

## Risk Watch

- **Path-filter false negatives**: if the audit script's allow-list is too permissive it could miss real violations. Test by injecting a synthetic marker into a source file, running the script, observing it fails.
- **Pytest output drift**: pytest version upgrades may change the exact duration line format. Normalization must be regex-based on the "passed in N.Ns" pattern, not exact match.
