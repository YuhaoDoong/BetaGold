# Round 9 Summary

## Status

Round 9 lands **task-h1 (validate-patch reproducibility harness)** + **task-h2 (plan-marker audit)** together, closing AC-10 and AC-11. Only task-e3 (Dashboard parity, AC-14) remains. **The plan is one task away from complete.**

## What Was Implemented

### v3.7.248 part 1 — task-h1 (validate-patch reproducibility harness, AC-10)

- `scripts/validate-patch.sh` — per-tag pytest validator with normalized output.
  - macOS-default-bash compatible (uses `case` instead of bash-4 associative arrays).
  - Per v3.7.* tag → pytest subset dispatch table (15 tags mapped).
  - Pipes pytest output through a normalization filter that strips non-deterministic fields (durations, absolute paths, `~/.pytest_cache` per-user paths).
  - Writes `data/backtest_history/<tag>/VALIDATION.md` per tag.
- `scripts/eval/normalize_pytest_output.py` — testable Python normalization module that powers the shell pipeline.
- `tests/test_validate_patch.py` — 7 cases locking the normalization rules:
  - strips `passed in N.Ns` / `failed in N.Ns` durations
  - strips repo root + arbitrary `~/.pytest_cache` paths
  - preserves PASSED / FAILED + test identifiers verbatim
  - idempotent (running normalize twice == running once)
  - two runs with different real durations produce byte-identical normalized output

Smoke: `scripts/validate-patch.sh v3.7.241` produces a normalized markdown report at `data/backtest_history/v3.7.241/VALIDATION.md`. The `passed in <NORMALIZED>s` line confirms the timing normalization fires.

### v3.7.248 part 2 — task-h2 (plan-marker audit, AC-11)

- `scripts/eval/audit_plan_markers.sh` — git-diff-baseline scoped audit.
  - Scope: files changed in `v3.7.232..HEAD`. Pre-existing markers in legacy scripts (e.g. domain "Step N" labels in `setup_data.py` data-download flow) are out of scope by design — the plan's wording is "no markers **introduced by these patches**".
  - macOS bash 3.2 compatible (`case` + `while IFS= read -r` instead of `mapfile`).
  - Forbidden patterns: `AC-\d+`, `(^|[^A-Za-z])Milestone:`, `(^|[^A-Za-z])Phase [A-Z]:`, `(^|[^A-Za-z])Step \d+:`.
  - Self-exempt (the audit script itself must reference the patterns it forbids).
  - Writes `data/backtest_history/v3.7.248_plan_marker_audit/REPORT.md` with `ac11_passed: true|false`.
  - Exits non-zero on dirty tree (CI-friendly).
- **Source-code cleanup pass**: 16 source/test files cleaned of `AC-N` markers via a one-off batch script; replaced with domain language ("the plan contract", "per the plan", "closure", or simply removed). Test docstrings shortened to functional descriptions rather than plan-AC references. The `the the plan contract` double-the artifacts from naive substitution were patched.
- **Final audit run**: `0 violations`, `ac11_passed: true`.

## Files Changed

### Modified (markers cleaned)
- `core/cross_asset_signal.py`, `core/data_freshness.py`, `core/calibration.py`
- `scripts/backtest/framework.py`
- `scripts/backtest/layer2_strategy/directional_options/run_all.py`
- `scripts/eval/model_calibration_audit.py`, `scripts/eval/calibration_gate_grid.py`
- 9 test files under `tests/`

### Created
- `scripts/validate-patch.sh`
- `scripts/eval/audit_plan_markers.sh`
- `scripts/eval/normalize_pytest_output.py`
- `tests/test_validate_patch.py`

### Out-of-repo artifacts
- `/Users/yhdong/Gold/data/backtest_history/v3.7.248_plan_marker_audit/REPORT.md` (ac11_passed: true)
- `/Users/yhdong/Gold/data/backtest_history/v3.7.241/VALIDATION.md` (smoke-tested example)

## Validation

- pytest full suite: **121/121 passed in 0.70s** (R8: 114 + R9: 7).
- Smoke `scripts/validate-patch.sh v3.7.241` exit 0, normalized output written.
- `scripts/eval/audit_plan_markers.sh` exit 0, 0 violations.
- Both shell scripts confirmed compatible with macOS-default bash 3.2.

## AC Status Delta

| AC | Status |
|---|---|
| AC-1 | LANDED (R5) |
| AC-2 | LANDED (R0) |
| AC-3 | LANDED (R2) |
| AC-4 | LANDED (R2) |
| AC-5 | LANDED (R3) |
| AC-6 | LANDED (R1) |
| AC-7 | LANDED (R4) |
| AC-8 | LANDED (R8) |
| AC-9 | LANDED (R7) |
| **AC-10** | **LANDED (R9)** |
| **AC-11** | **LANDED (R9)** |
| AC-12 | LANDED (R0) |
| AC-13 | LANDED (R0) |
| AC-14 | not started |
| AC-15 | LANDED (R4) |

**Cumulative: 14 / 15 ACs LANDED. 1 not started (AC-14 Dashboard parity).**

## Remaining Work

| Phase | Tasks | ACs |
|---|---|---|
| E (continued) | task-e3 (Dashboard parity wrapper + parity harness) | AC-14 |

One task remains. Round 10 closes the plan.

## BitLesson Delta

Action: add
Lesson ID: BL-20260525-audit-scope-by-baseline-diff

```
## Lesson: audit-scope-by-baseline-diff
Lesson ID: BL-20260525-audit-scope-by-baseline-diff
Scope: scripts/eval/audit_plan_markers.sh and future "introduced by this
       patch series" audits
Problem Description: A whole-tree grep audit for forbidden markers
  ("AC-N", "Step N:", etc.) found 50 violations on first run, most of
  which were pre-existing legacy code (e.g. workflow step labels in
  setup_data.py) that the AC-11 contract did NOT cover. The plan
  wording is "no markers introduced by these patches" — implicitly
  baseline-scoped.
Root Cause: Audit scope set to the whole repo rather than to files
  modified by the patch series under review.
Solution: Scope the audit to `git diff --name-only <baseline>..HEAD`
  before applying the grep. Self-exempt the audit script itself (it
  must reference the patterns it forbids in order to enforce them).
Constraints: Requires a baseline tag to exist locally; the audit
  refuses to run with a clear error if missing.
Validation Evidence: scripts/eval/audit_plan_markers.sh exit 0 after
  scope fix; 50 violations → 0.
Source Rounds: 9
```
