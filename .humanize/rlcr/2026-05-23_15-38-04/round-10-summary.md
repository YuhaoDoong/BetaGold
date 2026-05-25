# Round 10 Summary (FINAL — Plan 100% LANDED)

## Status

Round 10 lands **task-e3 (Dashboard `run_backtest` deprecation wrapper + parity harness, AC-14)**. **All 15 ACs are now LANDED.** The plan is complete. Codex code-review phase is ready to engage.

## What Was Implemented

### v3.7.249 — Dashboard run_backtest deprecation + parity harness (task-e3, AC-14)

`core/dashboard_parity.py` — new module:
- **`run_unified_backtest(close_d, high_d, low_d, upper_band, lower_band, regime, rv_pctile, asset, gvz_series=None, **kwargs)`**: routes signal generation through `generate_daily_signals` (canonical pipeline) and reuses the StopLoss → BandExit → Pullback → Timeout priority order from legacy `run_backtest`. Returns `{trades, sig_df}`.
- **`_replay_one_pass(...)`**: slim daily replay that takes pre-computed `buy_signal` and `exit_signal` series. Mirrors legacy exit logic but does not duplicate signal generation.
- **`parity_check(legacy_trades, unified_trades, legacy_buy_signal=None, unified_buy_signal=None, drift_attribution_path=None, max_count_drift=1)`**: contract check returning a `ParityVerdict` dataclass with `status ∈ {PASS, PASS_WITH_DRIFT, FAIL_EXIT_EVENT_COUNT}`.
  - `PASS`: identical event counts + identical `buy_signal` columns
  - `PASS_WITH_DRIFT`: event counts within ±1 + signal columns differ on some dates; each drifted date is written to `signal_drift_attribution.csv` with one of two reasons (`canonical_pipeline_added_filter` or `canonical_pipeline_admitted_signal`)
  - `FAIL_EXIT_EVENT_COUNT`: any event type's count differs by > 1 → harness fails
- `EXIT_TYPES = ("StopLoss", "BandExit", "Pullback", "Timeout")` — constant locked by pytest schema-lock case.

`core/signals_v2.run_backtest` — added DeprecationWarning emission at function entry (no body changes). Warning text points users to `core.dashboard_parity.run_unified_backtest`. Removal target: v3.8.

`tests/test_dashboard_parity.py` — 8 cases:
1. Identical inputs → identical event counts → `PASS`
2. `buy_signal` drift (legacy entered day-5, unified entered day-6) with same event types → `PASS_WITH_DRIFT` + attribution CSV written
3. Drift without `drift_attribution_path` → `PASS_WITH_DRIFT` but no file written
4. Extreme drift (2 vs 8 StopLoss events) → `FAIL_EXIT_EVENT_COUNT` with `max_count_drift=6`
5. Exactly ±1 drift → `PASS` (AC contract boundary)
6. `EXIT_TYPES` constant matches expected set
7. Event-count dict has every known type even with empty inputs
8. Calling legacy `run_backtest` on a synthetic fixture emits `DeprecationWarning`

## Files Changed

### Modified
- `core/signals_v2.py` — DeprecationWarning emission at top of `run_backtest`

### Created
- `core/dashboard_parity.py`
- `tests/test_dashboard_parity.py`

## Validation

- pytest full suite: **129/129 passed in 0.66s** (R9: 121 + R10: 8).
- `scripts/eval/audit_plan_markers.sh` final run: **0 violations**, `ac11_passed: true`.
- 17 v3.7.* tags landed locally (v3.7.232 baseline + v3.7.233 … v3.7.249).

## Final AC Status

| AC | Status | Round |
|---|---|---|
| AC-1  | LANDED | R5 |
| AC-2  | LANDED | R0 |
| AC-3  | LANDED | R2 |
| AC-4  | LANDED | R2 |
| AC-5  | LANDED | R3 (shadow-only; live cutover gated) |
| AC-6  | LANDED | R1 |
| AC-7  | LANDED | R4 |
| AC-8  | LANDED | R8 (gate_passed=false; scaler shadow-only as designed) |
| AC-9  | LANDED | R7 |
| AC-10 | LANDED | R9 |
| AC-11 | LANDED | R9 (0 violations on baseline-scoped audit) |
| AC-12 | LANDED | R0 |
| AC-13 | LANDED | R0 |
| AC-14 | **LANDED** | **R10** |
| AC-15 | LANDED | R4 |

**15 / 15 ACs LANDED. Plan 100% complete.**

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 10 ran clean. The parity-harness design intentionally separates legacy/unified-path WIRING (this round) from FLIPPING the default path (deferred to v3.8 once the parity harness has been observed on real traffic) — this is the same shadow-first discipline established for the calibration cutover (gate_report.md) and the cross-asset selector (shadow_logging vs live_cutover). Worth noting that this discipline has emerged as a repeating project pattern; consider promoting "shadow-first, gate, then flip" as a BitLesson in a future round if it crosses a fourth instance.
