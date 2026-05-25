# Round 6 Summary

## Status

Round 6 lands **Phase G core — horizon-aware conformal scaler** (task-g3 + task-f4 scaler subset), AC-8 algorithmic core. Glue (integration into `extend_oos_predictions`, `build_band()` cutover, Layer 1 grid gate) waits for Round 7+. **The full plan is not yet complete.**

## What Was Implemented

### v3.7.244 — apply_rolling_conformal_scaler (task-g3, AC-8 core)

`core/calibration.py:apply_rolling_conformal_scaler(dates, pred_upper, pred_lower, actual_upper, actual_lower, horizon=5, window=60, target_coverage=0.80, min_pool=20) → (out_upper, out_lower, meta_df)`:

- **Per-side split-conformal**: chooses `delta_upper = quantile(actual_upper - pred_upper, target_coverage)` from the eligible pool; calibrated upper = `pred_upper + delta_upper`. Mirror logic for lower with `delta_lower = quantile(pred_lower - actual_lower, target_coverage)`; calibrated lower = `pred_lower - delta_lower`. Negative `delta_upper` shrinks the upper bound (model over-predicts); positive `delta_lower` widens the lower bound (model under-predicts). This per-side signed adjustment is required because Round 5's audit found 4 of 12 audited months had `width_ratio < 1` on at least one side — a symmetric "shrink" would worsen those.
- **Maturity-lag (AC-8 strict)**: at calibration time `t`, only residuals from source date `s` where `label_end_date(s) = s + horizon_trading_days < t` enter the pool. Implemented as positional `cutoff_idx = t - horizon - 1`, then `pool = residuals[max(0, cutoff_idx - window + 1) : cutoff_idx + 1]`. NaN residuals (label not yet materialized) dropped from the pool.
- **Insufficient-sample fallback**: pool size < `min_pool` (default 20) → raw bounds returned, `meta_df.fallback_reason = "insufficient_pool"`.
- **`meta_df`** (indexed by dates): `n_eligible`, `delta_upper`, `delta_lower`, `realized_coverage_upper`, `realized_coverage_lower`, `fallback_reason`. Future retrain trigger (task-g4) consumes this directly.
- **Pure function**: no I/O, no globals beyond module constants, deterministic. Verified hermetic via pytest `builtins.open` block.

### tests/test_calibration_scaler.py — 12 cases (task-f4 scaler subset)

- **maturity-lag invariant**: synthetic fixture where actual at `s = t-1/-2/-3` is poisoned to 999.0; scaler must NOT pull these into the pool for `t` (since `label_end_date(s) >= t` for `s > t-6`). delta stays at the clean value -1.0.
- **per-side asymmetric repair**: upper over-predicted → `delta_upper < 0` shrinks; lower under-predicted → `delta_lower > 0` widens. Both directions covered in a single mirror-image fixture matching the GLD 2026-03 audit signature.
- **target_coverage tracking**: on a random fixture, realized coverage on the eligibility pool matches the requested target (0.50 / 0.80 / 0.95) within ±0.10.
- **zero-residual edge**: perfect fit → `delta == 0`, calibrated == raw.
- **input validation**: out-of-range `target_coverage`, `horizon < 1`, `window < min_pool` all raise `ValueError`.
- **purity**: `builtins.open` monkey-patched to fail; scaler still runs end-to-end. Deterministic across repeated calls.

### Real-data smoke (GLD 113-day audit window, target_coverage=0.80)

| Metric | Raw | Calibrated | Δ |
|---|---:|---:|---:|
| coverage_upper | 0.823 | 0.832 | **+0.9 pp** |
| coverage_lower | 0.708 | 0.735 | **+2.7 pp** |
| coverage_both  | 0.549 | 0.593 | **+4.4 pp** |

`delta_upper` mean +0.214 (slight widening), `delta_lower` median -1.668 (lower bound shrinks because Round 5 audit showed the model over-predicted the downside on average over the 60-day historical pool). Movement is in the direction the audit prescribed. The +4.4pp on `coverage_both` is the structural change; whether it survives Layer 1 grid scoring (task-g6) decides the live cutover.

## Files Changed

### Created
- `core/calibration.py` — scaler + `ScalerMeta`
- `tests/test_calibration_scaler.py` — 12 cases

## Validation

- pytest full suite: **88/88 passed in 0.55s** (Round 5 76 + Round 6 12).
- Real-data smoke verified scaler direction matches audit signature.
- Maturity-lag invariant explicitly tested with poisoned future labels — guards against the leak the AC was designed to prevent.

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
| **AC-8** | **partial (R6)** — scaler algorithm + tests landed; cutover flag + grid gate report waits for tasks g4/g5/g6 |
| AC-10 | partial — 88 pytest cases |
| AC-11 | partial — no plan markers in v3.7.244 |
| AC-12 | LANDED (R0) |
| AC-13 | LANDED (R0) |
| AC-15 | LANDED (R4) |
| AC-9, AC-14 | not started |

**Cumulative: 10 / 15 ACs LANDED. 3 partial. 2 not started.**

## Remaining Work

| Phase | Tasks | ACs |
|---|---|---|
| G (continued) | task-g4 (retrain trigger), task-g5 (per-regime alpha), task-g6 (Layer 1 grid + gate_report) + scaler integration into `extend_oos_predictions` and `build_band()` preflight | AC-8 closure + AC-9 |
| E (continued) | task-e3 | AC-14 |
| F-body | task-f4 retrain hysteresis subset | AC-9 |
| Closure | task-h1, task-h2 | AC-10, AC-11 |

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 6 ran clean. Two pytest assertions in the asymmetric-repair tests initially used the wrong sign-direction reasoning when I wrote them, but the failing test output immediately revealed the mistake before commit, so this was caught within the same write-test → run-test cycle. Not a multi-round pattern.
