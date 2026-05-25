# Round 8 Contract

## Round Objective

Land **task-g6 (calibration gate analyze)** + Phase G closure: integrate the scaler into `extend_oos_predictions` shadow-only, expose calibrated columns in OOS parquet, write `gate_report.md` deciding whether the AC-8 compound gate passes. Once `gate_report.md` carries `gate_passed: true`, `build_band()` can read calibrated columns; this round leaves `live_cutover=False` per the shadow-first discipline.

## In-Scope Tasks

| Task | AC | Tag | Notes |
|------|----|----|-------|
| task-g6 | AC-8 closure | v3.7.247 | (a) `scripts/eval/calibration_gate_grid.py` analyze runner. Reads OOS, applies scaler with default params, computes raw vs calibrated coverage_both / coverage_upper / coverage_lower across trailing 10y / 5y / 3y / 1y / 113d windows for GLD and SLV. (b) Writes `data/backtest_history/v3.7.247_calibration_gate/gate_report.md` with the compound-gate decision. (c) Adds an optional CLI flag `--use-calibrated` to the runner so the same script can replay either band set for downstream Layer 1 integration work. (d) Pytest verifying the gate decision logic on a synthetic fixture. |

## Out-of-Scope

- Wiring calibrated columns into `extend_oos_predictions` live OOS extension (separate v3.8 work since the production OOS file shouldn't be mutated until the gate flips).
- Modifying `build_band()` to read calibrated columns under a flag — also v3.8, gated on `gate_passed=true`.
- task-e3 / closure h1/h2.

## Verification Plan

### task-g6
pytest `tests/test_calibration_gate.py`:
- **gate logic — pass**: synthetic raw_coverage = 0.50 vs calibrated_coverage = 0.70 in 4/4 windows → `gate_passed=True`.
- **gate logic — coverage_degradation**: any single window where calibrated coverage drops by > 5pp → `gate_passed=False`.
- **gate logic — insufficient_improvement**: only 2 of 4 windows move toward target → `gate_passed=False`.
- **target-direction**: gate criterion correctly identifies "toward target" regardless of whether raw is above or below target.
- **report serialization**: `gate_report.md` written with `gate_passed: true` (machine-parseable line) + human-readable summary.

Real-data smoke: run against GLD + SLV across the four trailing windows; record actual gate decision.

## Commit Discipline

- `v3.7.247`: code (`scripts/eval/calibration_gate_grid.py`) + tests
- `gate_report.md` archived under `/Users/yhdong/Gold/data/backtest_history/v3.7.247_calibration_gate/` (outside GoldDash repo, so committed separately if at all)

## Bitlessons Consulted

- Empty knowledge base.

## Risk Watch

- **Real-data SLV staleness**: SLV's most recent ~1 month of OOS has NaN `actual_*_pct` (5-day forward not yet materialized). Gate computation must exclude NaN dates rather than treat NaN as a coverage failure.
- **Window-size effects**: 10y window for GLD has > 2000 rows; for SLV it's smaller because the SLV OOS starts later. Document per-asset effective sample size in the report.
- **Gate decision is non-binding now**: this round only writes the gate report. The `build_band()` cutover lives behind a config flag flipped in v3.8 ONLY if the report carries `gate_passed: true`. If today's data already shows `gate_passed: true`, that's evidence the calibration scaler is production-ready; if not, we ship the audit and retrain trigger but defer the band cutover.
