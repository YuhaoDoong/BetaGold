# Round 5 Summary

## Status

Round 5 lands **Phase G start — calibration audit** (task-g1 + task-g2), targeting AC-1. The audit ratifies Codex's Round 1 numbers against the original `train_dl_range.build_targets` label definition, and surfaces an **asymmetric per-month drift** signature that constrains how Phase G's scaler must be designed. Phase G g3..g6 (scaler, retrain, per-regime, gate) still queued. **The full plan is not yet complete.**

## What Was Implemented

### v3.7.243 — calibration audit (task-g1, AC-1)

`scripts/eval/model_calibration_audit.py`:
- Reads the OOS parquet's authoritative `actual_upper_pct` / `actual_lower_pct` columns directly. These are 5-day forward max-high / min-low vs t-day close per `src/models/train_dl_range.build_targets:72-77`.
- Hard-fails with `ValueError` if those columns are missing. The error message explicitly cites the build_targets path so future maintainers cannot accidentally substitute single-day overnight returns (the exact bug v3.7.232's draft made).
- Per `(month [× regime])` group emits `n`, `pred_upper_mean`, `pred_lower_mean`, `actual_upper_mean`, `actual_lower_mean`, `width_ratio_upper`, `width_ratio_lower`, `coverage_upper`, `coverage_lower`, `coverage_both`. Plus an `ALL` aggregate row.
- Width ratio uses `mean(pred) / mean(actual)` with tiny-denominator guard → `NaN`, not `Inf`.
- Coverage formulas match `src/models/train_dl_range.eval_range`: `actual_upper ≤ pred_upper` and `actual_lower ≥ pred_lower`.
- CLI: `--asset {GLD, SLV} --start --end [--regime-col regime] [--out-dir ...] [--parquet-path ...]`.
- Output: CSV + markdown report under `data/backtest_history/v3.7.243_calibration_audit/`.

`tests/test_calibration_audit.py` (8 cases):
- analytical synthetic fixture (pred ±4, actual ±2) → exact ratio 2.0, coverage 100%
- per-month grouping with bdate range
- per-regime grouping with 3 regimes
- empty window → empty DataFrame, exit 0
- missing `actual_*_pct` columns → `ValueError` mentioning `src/models/train_dl_range`
- zero-actual denominator → `NaN` width ratio, coverage still well-defined
- coverage formula tied to `eval_range` semantics (4-row hand-crafted breach matrix)
- `REQUIRED_COLUMNS` constant locked

### task-g2 (analyze) — `AUDIT_REPORT.md`

Real audit ran for both assets over `2025-12-01..2026-05-13`.

**GLD aggregate (n=113):**
| metric | value |
|---|---:|
| width_ratio_upper | **1.948×** |
| width_ratio_lower | **1.663×** |
| coverage_upper | 82.30% |
| coverage_lower | 70.80% |
| coverage_both | **54.87%** |

**SLV aggregate (n=82):**
| metric | value |
|---|---:|
| width_ratio_upper | **1.425×** |
| width_ratio_lower | **1.871×** |
| coverage_upper | 67.07% |
| coverage_lower | 84.15% |
| coverage_both | **53.66%** |

**Per-month asymmetric drift** (the key finding for Phase G design):
- GLD 2026-01: `width_ratio_upper = 0.77` → model *under-predicts* the upper bound.
- GLD 2026-03: `width_ratio_upper = 3.53, width_ratio_lower = 0.73` → over-predicts upper AND under-predicts lower simultaneously.
- GLD 2026-05: `width_ratio_lower = 0.85` → under-predicts lower again.
- SLV 2025-12: `width_ratio_upper = 0.92, width_ratio_lower = 4.76` → mirror image, over-predicts lower.

A symmetric "shrink the band" scaler would worsen the under-predicted side in every one of these months. **The Phase G scaler (task-g3) must support per-side, signed adjustments**, which AC-8's interface (`apply_rolling_conformal_scaler(... target_coverage=0.80)`) already permits.

**Correction of draft figures** (recorded in the report):
- draft 5-6× / 87.6% used single-day overnight returns (`H/L vs prior Close`).
- correct figures use 5-day forward labels per parquet.
- ratio gap: draft 5/0.9 ≈ 5.5× vs correct 5/3 ≈ 1.7×.
- coverage gap: draft 87.6% (±5% band trivially contains 1d ±0.9% move) vs correct 54.87% (against true 5d ±3% range).

## Files Changed

### Created
- `scripts/eval/__init__.py` (implicit via mkdir; no content)
- `scripts/eval/model_calibration_audit.py`
- `tests/test_calibration_audit.py`
- (outside repo) `/Users/yhdong/Gold/data/backtest_history/v3.7.243_calibration_audit/AUDIT_REPORT.md` + GLD/SLV CSV+md per-month tables

## Validation

- pytest full suite: **76/76 passed in 0.64s** (R0-R4 totals + R5 8).
- Real-data audit confirms Codex Round 1 baseline (1.948× / 54.87% on GLD).
- Asymmetric drift exposed in 4 of the 6 months → drives Phase G's per-side scaler design.

## AC Status Delta

| AC | Status |
|---|---|
| **AC-1** | **LANDED (R5)** — audit reproducible + label definition locked + real baseline documented |
| AC-2 | LANDED (R0) |
| AC-3 | LANDED (R2) |
| AC-4 | LANDED (R2) |
| AC-5 | LANDED (R3, shadow-only) |
| AC-6 | LANDED (R1) |
| AC-7 | LANDED (R4) |
| AC-10 | partial — 76 pytest cases |
| AC-11 | partial — no plan markers in v3.7.243 |
| AC-12 | LANDED (R0) |
| AC-13 | LANDED (R0) |
| AC-15 | LANDED (R4) |
| AC-8, AC-9, AC-14 | not started |

**Cumulative: 10 / 15 ACs LANDED. 2 partial. 3 not started.**

## Remaining Work

| Phase | Tasks | ACs |
|---|---|---|
| E (continued) | task-e3 | AC-14 (Dashboard parity) |
| G (continued) | task-g3, g4, g5, g6 + task-f4 | AC-8, AC-9 |
| Closure | task-h1, task-h2 | AC-10, AC-11 |

5 of 15 ACs still need work (was 6 entering Round 5). **Plan is not complete.**

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 5 ran clean. The `tabulate` missing-dependency error during the smoke run was resolved in-flight (replaced `to_markdown()` with a hand-rolled markdown table) and is not a recurring class. Still no recurring fixture-arithmetic bug to graduate.
