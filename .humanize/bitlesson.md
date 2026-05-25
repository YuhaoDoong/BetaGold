# BitLesson Knowledge Base

This file is project-specific. Keep entries precise and reusable for future rounds.

## Entry Template (Strict)

Use this exact field order for every entry:

```markdown
## Lesson: <unique-id>
Lesson ID: <BL-YYYYMMDD-short-name>
Scope: <component/subsystem/files>
Problem Description: <specific failure mode with trigger conditions>
Root Cause: <direct technical cause>
Solution: <exact fix that resolved the problem>
Constraints: <limits, assumptions, non-goals>
Validation Evidence: <tests/commands/logs/PR evidence>
Source Rounds: <round numbers where problem appeared and was solved>
```

## Entries

<!-- Add lessons below using the strict template. -->

## Lesson: gate-rule-clarity
Lesson ID: BL-20260525-gate-rule-clarity
Scope: scripts/eval/calibration_gate_grid.py, future gate-design tasks
Problem Description: Initial two-criterion gate (toward_target AND not_degraded) with separate `raw - cal <= max_degradation` mis-fired when raw was above target (over-coverage). Correct shrink of an over-wide band registered as "degradation" even though it moved coverage closer to target.
Root Cause: Conflating "coverage went down" with "calibration is worse" when the target is between raw and cal. The metric of interest is *distance from target*, not absolute coverage value.
Solution: Use a single coherent rule on signed distance — `cal_distance_from_target - raw_distance_from_target < max_degradation`. Symmetric around target; small overshoots tolerated by max_degradation.
Constraints: Applies anywhere a "gate" or "regression check" compares two measurements against a target value (calibration coverage, drift metrics, any tracked-toward-target KPI).
Validation Evidence: tests/test_calibration_gate.py::test_gate_handles_raw_above_target caught the original mis-design; test now passes with the single-rule.
Source Rounds: 8
