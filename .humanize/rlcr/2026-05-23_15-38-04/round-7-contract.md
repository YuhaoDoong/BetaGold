# Round 7 Contract

## Round Objective

Close out Phase G's algorithmic surface: **task-g4 retrain trigger** (AC-9 full), **task-g5 per-regime alpha** (AC-8 closure modulo grid gate), and the **task-f4 retrain hysteresis test subset**. After Round 7 only `task-g6` (Layer 1 grid analyze → `gate_report.md`), `task-e3` (Dashboard parity), and closure (`task-h1/h2`) remain.

## In-Scope Tasks

| Task | AC | Tag | Notes |
|------|----|----|-------|
| task-g4 | AC-9 | v3.7.245 | `core/calibration.py:evaluate_retrain_trigger(meta_df, actual_widths, today, last_retrain_at=None, hysteresis_days=5, cooldown_days=7, ratio_threshold_queue=2.5, ratio_threshold_immediate=4.0, zero_width_floor=0.10)` returns a structured decision dict. Pure function. Caller writes to `data/models/retrain_queue.jsonl`. |
| task-g5 | AC-8 closure | v3.7.246 | Extend `apply_rolling_conformal_scaler` to accept an optional `regime: pd.Series` parameter. When provided, computes per-regime quantiles using same-regime past residuals only; falls back to global pool when `n_regime < min_pool_per_regime` (default 20). Per-regime fallback events recorded in `meta_df.fallback_reason`. |
| task-f4 (retrain subset) | AC-9 | bundled with v3.7.245 | `tests/test_calibration_retrain.py`: hysteresis (single-day breach NOT triggering), 5-day consecutive triggers queue, immediate-trigger threshold, cooldown blocks re-trigger, zero-width guard excludes degenerate days. |

## Out-of-Scope

- task-g6 (analyze): Layer 1 grid run + `gate_report.md`
- `extend_oos_predictions` integration + `build_band()` cutover preflight
- task-e3 / closure

## Verification Plan

### task-g4
pytest:
- single-day ratio breach → no queue (hysteresis)
- 5 consecutive days above 2.5 → `outcome="queued"`
- single day above 4.0 → `outcome="immediate"`
- retrain inside cooldown window → `outcome="suppressed_cooldown"`
- zero-width days excluded from ratio mean → `zero_width_excluded_count` reflected
- pure function: no I/O, deterministic

### task-g5
pytest:
- regime=None → identical output to v3.7.244 (backwards compatibility)
- 3-regime fixture where one regime has n<20 → that date falls back to global pool, `fallback_reason="regime_undersampled"`
- balanced 3-regime fixture (all n≥20) → per-regime deltas differ from global (when underlying residuals truly differ by regime)
- regime Series with NaN values handled (treat NaN as 'UNKNOWN' regime; respects min_pool too)

## Commit Discipline

- `v3.7.245`: task-g4 (retrain trigger) + retrain hysteresis tests
- `v3.7.246`: task-g5 (per-regime alpha) + per-regime tests

## Bitlessons Consulted

- Empty knowledge base; selector returns NONE.

## Risk Watch

- **Hysteresis edge**: "5 consecutive days" must be defined precisely (e.g., the last 5 entries in `meta_df` all breach 2.5). If there's a gap (NaN day, weekend, missing meta row), policy needs to be explicit. We treat consecutive at the meta_df-row level — gaps reset the counter.
- **Cooldown clock**: 7 trading days vs 7 calendar days. Per AC-9 spec the unit is "trading days". Use `pd.bdate_range` to count.
- **Per-regime fallback rule**: a single date can use mixed pools across calibration history. We compute the per-date eligible pool first, then within that pool partition by regime. If regime-of-current-date has fewer than `min_pool_per_regime` matured residuals in that pool, fall back to the full pool (NOT a different-regime pool).
