# Round 6 Contract

## Round Objective

Land **Phase G core — horizon-aware conformal scaler** (task-g3 + task-f4 scaler portion), targeting AC-8. The scaler is the structural fix the calibration audit (Round 5) prescribed: it must support per-side signed adjustments (not symmetric shrinking) because the Round 5 audit found 4 of 12 audited months had `width_ratio < 1` on at least one side (model under-predicted that side).

## In-Scope Tasks This Round

| Task | AC | Tag | Notes |
|------|----|----|-------|
| task-g3 | AC-8 | v3.7.244 | New `core/calibration.py:apply_rolling_conformal_scaler(dates, pred_upper, pred_lower, actual_upper, actual_lower, horizon=5, window=60, target_coverage=0.80) → (pred_upper_calibrated, pred_lower_calibrated)`. Horizon-aware maturity lag (`label_end_date < calibration_as_of_date`). Per-side split-conformal: `delta_upper = quantile(actual_upper - pred_upper, target_coverage)`, `delta_lower = quantile(pred_lower - actual_lower, target_coverage)`. Calibrated bounds: `pred_upper + delta_upper`, `pred_lower - delta_lower`. Insufficient eligible samples → return raw (no calibration). |
| task-f4 (scaler subset) | AC-8 | bundled with v3.7.244 | `tests/test_calibration_scaler.py` covering maturity-lag invariant, per-side asymmetric repair, target_coverage tracking, insufficient-sample fallback, zero-residual edge. |

## Out-of-Scope This Round

- task-g4 (retrain trigger, AC-9) — depends on this scaler's output residuals
- task-g5 (per-regime alpha) — depends on this scaler
- task-g6 (Layer 1 grid gate, analyze) — runs after g5
- `extend_oos_predictions` integration (writing calibrated columns) — bundled with g6 once gate decision is known
- `build_band()` cutover preflight — also waits for g6
- task-e3 / closure

This round lands the **algorithmic core**. Glue (integration into the live OOS extension job + the `live_cutover` flag plumbing) lands once the Layer 1 grid validates the algorithm works.

## Verification Plan

### task-g3 + task-f4-scaler (AC-8)

pytest `tests/test_calibration_scaler.py`:
- **Maturity-lag invariant**: a synthetic series where `actual_*_pct[s]` for `s = dates[t-1]` would mathematically improve calibration but `label_end_date(s) >= dates[t]` (still maturing); assert scaler does NOT use it. The eligibility predicate is `label_end_date < calibration_as_of_date` with `label_end_date(s) = s + horizon_trading_days`.
- **Per-side asymmetric repair**: fixture where model over-predicts upper (`pred_upper = actual_upper + 2.0`) and under-predicts lower (`pred_lower = actual_lower + 2.0`, i.e., not negative enough); assert `delta_upper < 0` (shrink upper) AND `delta_lower > 0` (widen lower) AND signs of the calibrated columns make sense.
- **target_coverage tracking**: after applying scaler with `target_coverage=0.90`, empirical coverage on the eligibility pool is approximately ≥ 0.90; same with `target_coverage=0.50` yields tighter bands.
- **Insufficient samples**: with `window=60` but only 10 matured residuals, scaler returns raw bounds and sets a `_scaler_meta` field (or similar) noting fallback.
- **Zero-residual edge**: when residuals are all 0 (pred == actual perfectly), `delta == 0` and calibrated == raw.
- **Function is pure**: no I/O, no globals beyond constants; deterministic with the same input.

## Commit Discipline

- `v3.7.244`: task-g3 code + pytest (task-f4 scaler portion)

## Bitlessons Consulted

- Empty knowledge base; selector returns NONE.

## Round 6 Risk Watch

- **Quantile direction**: the split-conformal direction is easy to flip and produce a bound that widens when the model over-predicts (worse coverage). Double-check the test "per-side asymmetric repair" exercises both directions explicitly.
- **Maturity-lag off-by-one**: AC-8 specifies `label_end_date(s) < calibration_as_of_date` (strict less-than). `label_end_date(s) = s + horizon_trading_days` uses business days. Test must construct a series where the strict boundary matters.
- **Insufficient-sample threshold**: I'll set `min_pool = max(20, window // 3)` as a conservative floor. Below that, return raw + warn.
