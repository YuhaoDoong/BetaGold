# Round 5 Contract

## Round Objective

Land **Phase G start — calibration audit** (task-g1 + task-g2), targeting AC-1. These two tasks ground the rest of Phase G (g3-g6: scaler, retrain trigger, per-regime alpha, Layer 1 gate report) on **reproducible empirical evidence** with the correct label definition, replacing the draft's incorrect 5-6× / 87.6% figures with the actual 5-day forward H/L semantics per `src/models/train_dl_range.py:build_targets`.

## In-Scope Tasks This Round

| Task | AC | Tag | Notes |
|------|----|----|-------|
| task-g1 | AC-1 | v3.7.243 | New `scripts/eval/model_calibration_audit.py` reads `data/models/dl_range_v2_oos.parquet` (GLD) and `dl_range_slv_oos.parquet` (SLV) using parquet's `actual_upper_pct` / `actual_lower_pct` columns directly (NOT recomputed from OHLC overnight). Per-month × per-regime report covering: `n`, `pred_upper_mean`, `pred_lower_mean`, `actual_upper_mean`, `actual_lower_mean`, `width_ratio_upper`, `width_ratio_lower`, `coverage_both`, `coverage_upper`, `coverage_lower`. CLI: `--asset {GLD,SLV} --start --end [--regime-col regime]`. Output to `data/backtest_history/v3.7.243_calibration_audit/{asset}_{start}_{end}.csv` + per-asset markdown report. |
| task-g2 | AC-1 | analyze | Run the audit against both GLD and SLV over the trailing 113 trading days ending 2026-05-13 (the same window the idea draft asserted 5-6× / 87.6% for). Produce `AUDIT_REPORT.md` with the corrected baseline. Compare to the v3.7.232 draft figures and document the discrepancy: draft used single-day overnight (H/L vs prior Close), parquet uses 5-day forward (max_high_5d / min_low_5d vs t-day Close). |

## Out-of-Scope This Round

- task-g3 (conformal scaler) — depends on this audit's empirical baseline
- task-g4/g5/g6 — depend on g3
- task-e3 / closure tasks

## Verification Plan

### task-g1 (AC-1)
- pytest `tests/test_calibration_audit.py`:
  - **Label definition lock**: a synthetic OOS DataFrame fixture with hand-crafted `actual_upper_pct` and `actual_lower_pct` (5-day forward semantics) produces width_ratio and coverage values that match analytical expectation within 1e-6.
  - **Per-month grouping**: a 60-row fixture spanning 3 months produces a 3-row report.
  - **Per-regime grouping** (when regime column present): a fixture with regime in {Bull, Bear, Sideways} produces ≤ 3 rows per month.
  - **Edge: empty window** → report file with header only; script exits 0 with WARNING.
  - **Edge: missing `actual_*` columns** → script raises explicit `ValueError` (do not silently substitute).
- Smoke run against real GLD parquet over 2025-12-01..2026-05-13 (the draft's window) confirms width_ratio ≈ 1.95 upper / 1.66 lower and coverage ≈ 54.9% (Codex Round 1 numbers).

### task-g2 (analyze)
- Real audit run for GLD (10y, 5y, 1y, 113-day windows) + SLV (5y, 1y, 113-day).
- Report includes:
  - The corrected baseline table per asset/window.
  - Explicit discrepancy note: draft cited 5-6× / 87.6% from single-day overnight returns; parquet's `actual_*_pct` is 5-day forward (verified by reading `train_dl_range.build_targets`).
  - Per-month break-out for the most recent 6 months to surface regime drift.
  - Calibration goal restatement (per AC-8 + DEC-5): coverage repair toward training target, NOT band narrowing.

## Commit Discipline

- `v3.7.243`: task-g1 code + pytest
- Analyze artifacts (task-g2): markdown + CSVs in archive directory (path outside repo, per the GoldDash / Gold split)

## Bitlessons Consulted

- Empty knowledge base; selector returns NONE.
- One recurring pattern across rounds: I have repeatedly used the wrong fixture date or wrong calendar assumption in tests (R1 stale-day-3, R3 tz-aware/naive, R4 stale-vs-open expiry confusion). Three occurrences across four rounds is enough to graduate a BitLesson entry titled "verify test fixture exercises the intended branch" — record this round if it recurs again.

## Round 5 Risk Watch

- **Audit must not silently substitute label definition**: the whole point of this audit is to fix the draft's wrong label. Hard-fail when `actual_upper_pct` / `actual_lower_pct` are missing rather than recomputing from OHLC.
- **Regime column source**: the OOS parquet may or may not carry a regime column. If absent, the audit emits only per-month rows and notes regime is unavailable; per-regime breakdown waits until the regime classifier output is joined externally.
- **GLD vs SLV labels**: `train_dl_range.build_targets` is the GLD definition. SLV trains via `dl_range_slv` — verify the SLV parquet uses the same label semantics; if different, the report must surface this (no silent equivalence assumption).
