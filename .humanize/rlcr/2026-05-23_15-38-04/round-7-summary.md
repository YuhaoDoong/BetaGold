# Round 7 Summary

## Status

Round 7 lands **task-g4 (retrain trigger, AC-9)**, **task-g5 (per-regime alpha, AC-8 closure)** and **task-f4 (calibration test bodies, AC-8/AC-9)** all together. Only `task-g6` (Layer 1 grid analyze → `gate_report.md`), `task-e3` (Dashboard parity), and closure tasks remain. **The full plan is not yet complete.**

## What Was Implemented

### v3.7.245 — calibration-gated retrain trigger (task-g4, AC-9)

`core/calibration.py:evaluate_retrain_trigger(meta_df, actual_widths, today, last_retrain_at=None, window=30, hysteresis_days=5, cooldown_days=7, ratio_threshold_queue=2.5, ratio_threshold_immediate=4.0, zero_width_floor=0.10)`:

- Pure function returning `{outcome, ratio_value, consecutive_days, zero_width_excluded_count, cooldown_until, triggered_at}`.
- **Cooldown** evaluated FIRST: if `today` ≤ `last_retrain_at + cooldown_days` trading days, return `"suppressed_cooldown"` regardless of ratio.
- **Immediate** branch: latest day's ratio > 4.0 → return `"immediate"` (no hysteresis required for catastrophic miscalibration).
- **Queued** branch: ratio > 2.5 for the *last* 5 consecutive scaler rows → return `"queued"`. Hysteresis at the meta-row level; a gap resets the counter.
- **Zero-width guard**: days where `|actual_width| < 0.10%` are excluded from the rolling mean and counted in `zero_width_excluded_count`, preventing degenerate single-day actuals from poisoning the ratio.
- Caller writes the returned dict to `data/models/retrain_queue.jsonl`; this module never touches the filesystem.

### v3.7.246 — per-regime conformal alpha (task-g5, AC-8 closure)

`apply_rolling_conformal_scaler` extended with two optional kwargs:
- `regime: pd.Series` aligned with `dates`. NaN values coalesced to `"UNKNOWN"`.
- `min_regime_pool: int = 20`. When `regime` is supplied, the per-date eligibility pool is further restricted to same-regime past dates; if same-regime matured residual count is below `min_regime_pool`, the date *falls back* to the global pool and `meta_df.fallback_reason = "regime_undersampled[<regime>:<count>]"` records the fallback.

When `regime=None` (the v3.7.244 default), behavior is identical to before — backwards compatible, verified by pytest.

### tests/test_calibration_retrain.py (v3.7.245) — 11 cases

- Hysteresis: single-day breach below immediate threshold → `no_action`; 5 consecutive breaches → `queued`; exactly 4 → still `no_action`.
- Immediate threshold: a single day above 4.0 → `immediate`.
- Cooldown: inside the window → `suppressed_cooldown`; outside → re-enables.
- Zero-width guard: low-width days excluded from the rolling mean; counter reflects them; all-zero-width → `no_action`.
- Pure function (`builtins.open` block) + deterministic.
- Empty meta → `no_action`.

### tests/test_calibration_per_regime.py (v3.7.246) — 6 cases

- `regime=None` produces identical output to a single-regime `regime=['Bull']*n` fixture (backwards compat).
- Balanced 3-regime fixture (Bull residual = -1, Bear = +3, Sideways = 0) with `window=120` produces a Bull-day `delta_upper` near -1 (the Bull-only quantile), NOT the global mixture.
- Under-sampled regime (95 Bull / 5 Bear): the Bear day falls back to global pool; `fallback_reason` captures `"regime_undersampled[Bear:N]"`.
- All-NaN regime coalesces to single `"UNKNOWN"` regime, treated consistently.
- Isolated Bull at index 95 of 100 (after 95 Bear days) triggers fallback and the meta string contains "Bull".
- `MIN_REGIME_POOL_SIZE` constant locked within sensible bounds.

## Files Changed

### Modified
- `core/calibration.py` — `evaluate_retrain_trigger` + retrain constants; `apply_rolling_conformal_scaler` regime parameter + fallback logic

### Created
- `tests/test_calibration_retrain.py` — 11 cases (v3.7.245)
- `tests/test_calibration_per_regime.py` — 6 cases (v3.7.246)

## Validation

- pytest full suite: **105/105 passed in 0.75s** (Round 6: 88 + Round 7: 17).
- Backwards compatibility: all 12 cases in `test_calibration_scaler.py` still pass with the new `regime`/`min_regime_pool` parameters.
- Pure-function discipline maintained for both new entrypoints.

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
| **AC-8** | **algorithmic core complete (R6+R7)** — scaler + per-regime + maturity-lag all in place; cutover preflight + `gate_report.md` from task-g6 remains |
| **AC-9** | **LANDED (R7)** |
| AC-10 | partial — 105 pytest cases |
| AC-11 | partial — no plan markers in v3.7.245/246 |
| AC-12 | LANDED (R0) |
| AC-13 | LANDED (R0) |
| AC-15 | LANDED (R4) |
| AC-14 | not started |

**Cumulative: 11 / 15 ACs LANDED, AC-8 algorithmically complete (gate report pending), 2 partial, 1 not started.**

## Remaining Work

| Phase | Tasks | ACs |
|---|---|---|
| G (analyze) | task-g6 (Layer 1 grid + gate_report.md) + scaler integration into `extend_oos_predictions` + `build_band()` cutover preflight | AC-8 closure |
| E (continued) | task-e3 (Dashboard parity) | AC-14 |
| Closure | task-h1 (make validate-patch), task-h2 (final grep audit) | AC-10, AC-11 |

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 7 ran clean. The retrain trigger's "ratio_value" definition was non-obvious — I initially considered using `pred_width / actual_width` directly but that requires carrying `pred_width` in `meta_df`. Settled on `(|delta_upper| + |delta_lower| + actual_width) / actual_width` which is equivalent in spirit and uses only fields already recorded. This is a design choice worth noting in the module docstring (already done in the function's body comment) but does not rise to a project-wide BitLesson.
