# Round 8 Summary

## Status

Round 8 lands **task-g6 (calibration gate analyze, AC-8 closure)** with a real empirical decision: **gate_passed = FALSE** for both GLD and SLV. The scaler infrastructure ships shadow-only, which is exactly what AC-8 + DEC-5 prescribed for this outcome. Only task-e3 (Dashboard parity, AC-14) and closure (h1/h2) remain. **The full plan is not yet complete, but Phase G is fully shipped (the algorithmic pieces + the gate that decides cutover).**

## What Was Implemented

### v3.7.247 — calibration gate decision + gate_report.md

`scripts/eval/calibration_gate_grid.py`:
- 5 trailing windows × 2 assets: 10y / 5y / 3y / 1y / 113d for GLD and SLV.
- For each window: computes raw vs calibrated `coverage_upper / coverage_lower / coverage_both` and a per-window pass/fail under the AC-8 + DEC-5 compound rule.
- **Single coherent gate rule** (replaces an earlier two-criterion design that mis-fired when raw was above target):
  `cal_distance_from_target - raw_distance_from_target < max_degradation`.
  Correctly handles both "raw too low → lift toward target" and "raw too high → shrink toward target", with `max_degradation` as a small tolerance band.
- Three independent gates (upper / lower / both); overall `gate_passed = all three`.
- Writes `gate_report.md` with a machine-parseable `gate_passed: true|false` line + per-asset per-window tables.

### Real gate decision (empirical result)

**Overall: `gate_passed: false`**

GLD breakdown:
| window | n | raw_both | cal_both | distance_delta | pass |
|---|---:|---:|---:|---:|:-:|
| 10y | 2520 | 0.752 | 0.533 | +0.219 | False |
| 5y  | 1260 | 0.751 | 0.544 | +0.207 | False |
| 3y  | 756  | 0.733 | 0.552 | +0.181 | False |
| 1y  | 252  | 0.690 | 0.631 | +0.060 | False |
| 113d | 113 | 0.549 | 0.593 | -0.044 | **True** |

Per-side passes:
- `coverage_upper` gate: **5/5 PASS** for GLD
- `coverage_lower` gate: **5/5 PASS** for GLD
- `coverage_both`  gate: **1/5 PASS** for GLD

SLV mirrors the same pattern (upper 5/5, lower 4/5, both 1/5).

### Why the gate failed (honest diagnosis)

The split-conformal scaler picks a quantile per side that targets the per-side coverage at `target_coverage=0.80`. Each side independently moves closer to 0.80. But the joint event (`actual_upper ≤ cal_upper AND actual_lower ≥ cal_lower`) regresses because tightening both sides simultaneously reduces the probability that BOTH events hold for the same observation. The aggregate (`coverage_both`) is dragged down 15-22 pp on long windows.

The most recent 113d window is the one exception (`distance_delta = -0.044`, pass). That window already had low raw coverage (54.9%), and the calibrated coverage (59.3%) is closer to the 80% target on the joint metric. This is the regime the original v3.7.232 idea-draft framed the calibration concern around — and on that narrow window, the scaler does in fact help.

The implication: shadow-only is correct here. The retrain trigger + per-regime alpha (v3.7.245/246) give the team the data to iterate `target_coverage` or refine the regime classifier in a follow-on v3.8 round, then re-run this gate.

### tests/test_calibration_gate.py — 9 cases

- All-windows-pass → `gate_passed=True`
- Single-window severe degrade in a 4-window set → 3/4 pass → `gate_passed=True` (min_pass=3 default); the test explicitly checks this lenient boundary works.
- Strict `min_pass=4` → 1 degradation tanks the whole gate.
- 2/4 windows pass (true insufficient improvement) → `gate_passed=False`.
- Raw above target (over-coverage) → calibrated moving down toward target counts as "toward".
- Empty / NaN windows skipped without polluting `n_total`.
- Report serialization emits `gate_passed: true|false` machine-parseable line + the "shadow-only / keep build_band on raw" next-action in failure path.

## Files Changed

### Created
- `scripts/eval/calibration_gate_grid.py`
- `tests/test_calibration_gate.py`
- (outside repo) `/Users/yhdong/Gold/data/backtest_history/v3.7.247_calibration_gate/gate_report.md` + `gate_decision.json`

## Validation

- pytest full suite: **114/114 passed in 0.67s** (R7: 105 + R8: 9).
- Real-data smoke produced the gate decision; report includes both the machine-parseable line and a per-window human-readable table.
- The gate's correctness for "raw above target" case explicitly tested (regression for an earlier two-criterion mis-design).

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
| **AC-8** | **LANDED (R8)** — scaler + per-regime + maturity-lag + gate report all in place; gate decision = `false`; cutover correctly stays disabled |
| AC-9 | LANDED (R7) |
| AC-10 | partial — 114 pytest cases |
| AC-11 | partial — no plan markers in v3.7.247 |
| AC-12 | LANDED (R0) |
| AC-13 | LANDED (R0) |
| AC-15 | LANDED (R4) |
| AC-14 | not started |

**Cumulative: 12 / 15 ACs LANDED. 2 partial. 1 not started (AC-14 Dashboard parity).**

## Remaining Work

| Phase | Tasks | ACs |
|---|---|---|
| E (continued) | task-e3 (Dashboard parity) | AC-14 |
| Closure | task-h1 (make validate-patch), task-h2 (final grep audit) | AC-10, AC-11 |

3 tasks remain across 3 ACs.

## BitLesson Delta

Action: add
Lesson ID: BL-20260525-gate-rule-clarity

```
## Lesson: gate-rule-clarity
Lesson ID: BL-20260525-gate-rule-clarity
Scope: scripts/eval/calibration_gate_grid.py, future gate-design tasks
Problem Description: Initial two-criterion gate (toward_target AND not_degraded)
  with separate `raw - cal <= max_degradation` mis-fired when raw was above
  target (over-coverage). Correct shrink of an over-wide band registered as
  "degradation" even though it moved coverage closer to target.
Root Cause: Conflating "coverage went down" with "calibration is worse" when
  the target is between raw and cal. The metric of interest is *distance from
  target*, not absolute coverage value.
Solution: Use a single coherent rule on signed distance:
  `cal_distance_from_target - raw_distance_from_target < max_degradation`.
  Symmetric around target; small overshoots tolerated by max_degradation.
Constraints: Applies anywhere a "gate" or "regression check" compares two
  measurements against a target value (calibration coverage, drift metrics,
  any tracked-toward-target KPI).
Validation Evidence: tests/test_calibration_gate.py::test_gate_handles_raw_above_target
  caught the original mis-design; test now passes with the single-rule.
Source Rounds: 8
```

(Will be appended to `.humanize/bitlesson.md` in the round commit.)
