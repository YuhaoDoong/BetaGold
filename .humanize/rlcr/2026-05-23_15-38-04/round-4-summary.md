# Round 4 Summary

## Status

Round 4 lands **Phase D — derived-metric correctness** (task-d1 + task-d2), targeting AC-15 and AC-7. Phases E2 (Dashboard, AC-14), F-body (calibration tests), G (calibration scaler + retrain) still queued. **The full plan is not yet complete.**

## What Was Implemented

### v3.7.241 — max_move_{h}d off-by-one repair (task-d1, AC-15)

`scripts/backtest/framework.py:max_move_{h}d` legacy implementation used `series.rolling(h).max().shift(-(h+1))` which, at signal index ``i``, returned `max(series[i+2..i+h+1])` instead of the documented intent `max(series[i+1..i+h])`. The error excluded the entry-day high (`i+1`) and included one extra future day (`i+h+1`).

New helper `forward_window_extreme(series, window_h, anchor_offset=1, op='max'|'min')` uses explicit `lo = i + anchor_offset, hi = lo + window_h` indexing with NaN-safe reducers (`nanmax` / `nanmin`). Insufficient future-window indices produce `NaN`. The Layer 1 `max_up_{h}d` / `max_down_{h}d` / `max_move_{h}d` columns now use the helper.

Pytest `tests/test_max_move_window.py` (7 cases): known synthetic series (increasing 100..129), boundary NaN, NaN-in-window handling, invalid op raise, anchor_offset=0 inclusive-today variant, and a direct comparison `legacy ≡ correct + 1.0` on the strictly-increasing interior — making the off-by-one quantitatively visible.

### v3.7.242 — Layer 2 disposition reporting + per-leg DTE filter (task-d2, AC-7)

Three new helpers in `scripts/backtest/framework.py`:
- `kline_db_max_date()` — cached parquet read of the kline_db's latest available date.
- `leg_max_expiry(legs)` — parses each option-code expiry via `parse_option_code` and returns the latest (calendar-spread-safe).
- `run_layer2_backtest_with_disposition(dates, ohlc, asset, strategy, price_fn, exit_fn, dte_target, hold_buffer_days=5, today=None)` — iterates signals, tracks `{n_signal, n_entered, n_closed, n_open, n_skipped_stale, n_skipped_no_contract}`, applies the per-leg DTE pre-filter (`max(leg.expiry) + hold_buffer > kline_db.max_date` → `skipped_stale`), threads `asset=` into `exit_fn` (so per-asset cfg from v3.7.238 actually applies in research scripts too). Enforces the reconciliation invariant `n_signal == n_closed + n_open + n_skipped_stale + n_skipped_no_contract` by construction.

Both option Layer 2 runners refactored:
- `scripts/backtest/layer2_strategy/directional_options/run_all.py`: `backtest_option` / `grid_bc_pt` / `grid_sp_pt` now return `(rows, disp)`. Output CSV gains `n_signal / n_entered / n_closed / n_open / n_skipped_stale / n_skipped_no_contract` columns. Console prints disposition inline.
- `scripts/backtest/layer2_strategy/vol_options/run_all.py`: `backtest_strategy` returns `(closed_df, disp)`. Same output augmentation.

Futures Layer 2 runner is unchanged because it uses Binance live data, not kline_db; the disposition concepts don't apply identically and the existing closed-only collection is correct there.

Pytest `tests/test_layer2_disposition.py` (10 cases):
- `leg_max_expiry`: single leg, multi-leg same date, calendar-spread max, unparseable, empty.
- Reconciliation invariant for all-closed, all-no-contract, all-open (the original survivorship-bug scenario — without the fix these would silently vanish), all-stale (pre-filter trips on far-future expiry), and a mixed-disposition reconciliation (2 no_contract + 2 stale + 4 entered with alternating closed/open).

## Files Changed

### Modified
- `scripts/backtest/framework.py` — `forward_window_extreme`, `kline_db_max_date`, `leg_max_expiry`, `run_layer2_backtest_with_disposition`; `max_move` rewritten to use forward_window_extreme
- `scripts/backtest/layer2_strategy/directional_options/run_all.py` — uses disposition helper; main() output gains disposition columns
- `scripts/backtest/layer2_strategy/vol_options/run_all.py` — uses disposition helper; same output augmentation

### Created
- `tests/test_max_move_window.py` — 7 cases
- `tests/test_layer2_disposition.py` — 10 cases

## Validation

- pytest full suite: **68/68 passed in 0.34s** (R0 2 + R1 7 + R2 23 + R3 19 + R4 17).
- Legacy off-by-one explicitly demonstrated: on strictly-increasing series, `legacy ≡ correct + 1.0` for every interior index (confirms bug excludes entry-day + adds extra future-day).
- Reconciliation invariant: tested with mixed inputs (no_contract + stale + open + closed), holds in every combination.

## AC Status Delta

| AC | Status |
|----|--------|
| AC-2 | LANDED (R0) |
| AC-3 | LANDED (R2) |
| AC-4 | LANDED (R2) |
| AC-5 | LANDED (R3) — shadow-only; live cutover gated |
| AC-6 | LANDED (R1) |
| **AC-7** | **LANDED (R4)** |
| AC-10 | partial — 68 pytest cases |
| AC-11 | partial — no plan markers in v3.7.241/242 |
| AC-12 | LANDED (R0) |
| AC-13 | LANDED (R0) |
| **AC-15** | **LANDED (R4)** |
| AC-1, AC-8, AC-9, AC-14 | not started |

**Cumulative: 9 / 15 ACs LANDED. 2 partial. 4 not started.**

## Remaining Work

| Phase | Tasks | ACs |
|---|---|---|
| E (continued) | task-e3 | AC-14 (Dashboard parity, heavier) |
| F-body | task-f4 (needs Phase G g3/g4 done) | AC-8, AC-9 |
| G | task-g1..g6 | AC-1, AC-8, AC-9 |
| Closure | task-h1, task-h2 | AC-10, AC-11 |

6 of 15 ACs still need work (was 8 entering Round 4). **Plan is not complete.**

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 4 ran clean apart from one test fixture bug (used a far-future expiry where the stale pre-filter correctly tripped, when I actually wanted to test the open-but-not-stale path). Fixed by swapping to a past expiry. Same class of "verify your test fixture matches the branch you intend to exercise" hygiene that hasn't recurred yet across two rounds — still not promoted to a bitlesson entry.
