# Round 4 Contract

## Round Objective

Land **Phase D — derived-metric correctness** (task-d1 + task-d2), targeting AC-7 and AC-15. Both fix statistical/reporting correctness in the Layer 2 backtest pipeline. These are small, independent, fully-testable patches with no dependencies on Phase E2 / F / G.

## In-Scope Tasks This Round

| Task | AC | Tag | Notes |
|------|----|----|-------|
| task-d1 | AC-15 | v3.7.241 | `scripts/backtest/framework.py:max_move_{h}d`: replace `high.rolling(h).max().shift(-(h+1))` reverse-rolling pattern with explicit `entry_i = i + 1; exit_i = entry_i + h - 1; window = High[entry_i:exit_i+1].max()` indexed helper. Add pytest fixture with deterministic synthetic OHLC to assert window equivalence. |
| task-d2 | AC-7 | v3.7.242 | Layer 2 backtests (`scripts/backtest/layer2_strategy/{futures,directional_options,vol_options}/run_all.py`) emit `n_signal / n_entered / n_closed / n_open / n_skipped_stale / n_skipped_no_contract` columns. Sample-restriction logic uses per-leg DTE (parse expiry from leg code; entry_date + leg_max_dte + hold_buffer ≤ kline_max_date). Reconciliation invariant: `n_closed + n_open + n_skipped_* == n_signal`. |

## Out-of-Scope This Round

- task-e3 (AC-14): Dashboard parity harness
- Phase F-body / G / closure

## Verification Plan

### task-d1 (AC-15)
- pytest `tests/test_max_move_window.py`:
  - **Synthetic OHLC fixture**: deterministic 30-day series where High[i+1..i+5] is known analytically.
  - **Index equivalence**: for h=5, signal index i; helper returns `max(High[i+1..i+5]) / Close[i] - 1`. Asserted against hand-calculated value.
  - **Boundary**: last 5 days produce `NaN` (insufficient future window).
  - **Old-pattern equivalence on stable interior**: reverse-rolling and indexed helper agree where neither hits an edge; this confirms the bug was strictly an edge/off-by-one artifact.

### task-d2 (AC-7)
- Modify each Layer 2 runner to:
  - Track disposition counts during iteration.
  - Parse leg expiry from leg code (`parse_option_code` helper already in `core/strategies/options_exit.py`) to compute per-leg DTE.
  - Filter rule: a signal is "in-sample" only if `max(leg.expiry) + hold_buffer ≤ kline_db.max_date`. Otherwise count as `n_skipped_stale`.
  - `n_skipped_no_contract` accounts for `legs == []` after `price_strategy_at`.
- Per-window output table gains 6 columns; existing P&L columns unchanged.
- pytest `tests/test_layer2_disposition.py`:
  - Synthetic signal list with known mix of (closed, open, stale, no-contract); reconciliation `n_signal == sum(disposition cols)` holds.

## Commit Discipline

- `v3.7.241`: task-d1 (max_move helper + pytest)
- `v3.7.242`: task-d2 (Layer 2 disposition + per-leg DTE + pytest)

## Bitlessons Consulted

- Empty knowledge base; selector returns NONE for both tasks.

## Round 4 Risk Watch

- **task-d1 windowed-array off-by-one**: my own definition could itself have an off-by-one. The pytest hand-calculated assertion is the cross-check.
- **task-d2 leg expiry parsing**: option codes like `US.GLD260515P445000` parse to expiry 2026-05-15. Cross-asset cross-strike SP spreads have multiple legs whose expiries are the same (we don't have calendar spreads here), so `max(leg.expiry)` reduces to a single date; still, write the helper assuming heterogeneous expiries for future-proofing.
- **task-d2 hold_buffer choice**: backtest framework uses `dte_target` (45 for futures, 30 for options). A `hold_buffer = 0` would over-trim; `hold_buffer = 5` adds margin for late-fill exits. Use 5 by default with a comment so future tuners can adjust.
