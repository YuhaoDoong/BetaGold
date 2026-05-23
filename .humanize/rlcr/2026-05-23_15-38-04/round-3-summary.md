# Round 3 Summary

## Status

Round 3 lands **Phase E partial — IV-aware cross-asset selector + shadow log + replay analysis** (task-e1 + task-e2), targeting AC-5. The replay produces strong empirical support for the eventual live cutover. task-e3 (Dashboard `run_backtest` deprecation, AC-14) is deferred to a dedicated round because of the intraday StopLoss/Pullback/ACTIVE parity-harness work involved. **The full plan is not yet complete.**

## What Was Implemented

### v3.7.240 — IV-aware selector + caller-side shadow log + 14-day cutover gate (task-e1, AC-5)

`core/cross_asset_signal.py` adds three new symbols:

1. **`select_gld_sync_strategy(signal_date, gld_signal_row, gvz_value, gvz_asof_date) → {strategy, reason, gvz_status}`**
   - Pure function: no I/O, no globals beyond module constants, no wall-clock queries. Verified hermetic via pytest `monkeypatch` blocking `builtins.open`.
   - Staleness anchored on `signal_date`, NOT `today` — replay semantics match historical signals.
   - Truth table:
     - `gvz_value is None or NaN, or asof None` → `BUY CALL / GVZ_UNAVAILABLE / missing`
     - `signal_date − gvz_asof_date` in trading days > 2 → `BUY CALL / GVZ_UNAVAILABLE / stale`
     - `gld_signal_row.bp_low` missing or NaN → `BUY CALL / GLD_BP_LOW_MISSING / fresh`
     - `bp_low ≤ 0.10 AND gvz_value ≥ 25` → **`SELL PUT / DEEP_BREAK_HIGH_IV / fresh`**
     - otherwise → `BUY CALL / DEFAULT / fresh`
2. **`write_shadow_record(decision, signal_date, slv_tier, inputs, log_path)`** — caller-side JSONL append writer. Records the full input/output snapshot. Separate from the selector to preserve purity.
3. **`live_cutover_allowed(today, log_path, min_days=14) → (allowed, first_record_at, days_accumulated)`** — gate function that reads the shadow log JSONL and enforces the 14-calendar-day accumulation requirement before live flip. Handles tz-aware ↔ tz-naive boundary correctly.

`scripts/build_positions_ledger.py` cross-asset block now invokes the selector for every SLV-S trigger date, pulls `gvz_close` (already in scope at module level), pulls the GLD `bp_low` from `sig_df_history`, and writes a shadow record. The actual entered strategy is gated on `CROSS_LIVE_CUTOVER` (default `False` → still ships fixed `BUY CALL` while shadow log accumulates). When `CROSS_LIVE_CUTOVER=True` is flipped later (after the 14-day gate passes), the selector's recommendation flows into the ledger.

### task-e2 (analyze) — March 2026 5/5 BC cluster replay

The 5 GLD cross-asset BUY CALL entries in March 2026 that produced the historical `sum -334.3% / WR 0/5` loss cluster were replayed through the new selector with `bp_low` from `sig_df_history` and `^GVZ` from yfinance:

| Signal Date | bp_low | GVZ | Selector | Reason | Historical BC pnl |
|---|---:|---:|---|---|---:|
| 2026-03-03 | -0.111 | 38.8 | **SELL PUT** | DEEP_BREAK_HIGH_IV | -79.1% |
| 2026-03-19 | -0.940 | 31.0 | **SELL PUT** | DEEP_BREAK_HIGH_IV | -73.3% |
| 2026-03-20 | -0.633 | 35.2 | **SELL PUT** | DEEP_BREAK_HIGH_IV | -70.5% |
| 2026-03-23 | -0.595 | 43.4 | **SELL PUT** | DEEP_BREAK_HIGH_IV | -44.4% |
| 2026-03-26 | -0.009 | 45.1 | **SELL PUT** | DEEP_BREAK_HIGH_IV | -67.0% |

**5/5 entries** satisfy both selector conditions (deep break + high IV). Every one would have been switched to SELL PUT.

Using **native** GLD SELL PUT entries on adjacent March dates as the counterfactual P&L proxy: `n=5, WR=4/5, sum +66.7%`. Swing vs historical cross-asset BC sum -334.3% is **+401 pp**. Caveats (small sample, strike-selection drift, kline coverage) documented in `data/backtest_history/v3.7.240_cross_asset_replay/REPLAY.md`.

The selector ships shadow-only this round — a probabilistic ≥14-day shadow log accumulation is required before flipping `CROSS_LIVE_CUTOVER=True`.

## Files Changed

### Modified
- `core/cross_asset_signal.py` — `select_gld_sync_strategy`, `write_shadow_record`, `live_cutover_allowed`, IV/bp_low/stale thresholds as module constants
- `scripts/build_positions_ledger.py` — cross-asset block uses selector + shadow writer + gate

### Created
- `tests/test_cross_asset_selector.py` — 19 pytest cases
- `data/backtest_history/v3.7.240_cross_asset_replay/REPLAY.md` — analyze report
- `data/backtest_history/v3.7.240_cross_asset_replay/march_2026_replay.csv` — raw replay data

## Validation

- pytest full suite: **51/51 passed in 0.15s** (R0 2 + R1 7 + R2 23 + R3 19).
- Key purity invariants verified:
  - Selector runs with `builtins.open` monkey-patched to fail → still returns correct decision.
  - Selector with `signal_date=2020-01-15` and same inputs returns identical decision regardless of "today" — staleness uses signal_date anchor, not wall-clock.
- Boundary fixture: `bp_low=0.10 AND gvz=25.0` (exactly at thresholds) → `SELL PUT` (inclusive boundaries).
- March 2026 cluster fixture: `bp_low=0.045, GVZ=27.5` → `SELL PUT / DEEP_BREAK_HIGH_IV` (the regime that produced the historical loss).

## AC Status Delta

| AC | Status |
|----|--------|
| AC-2 | LANDED (R0) |
| AC-3 | LANDED (R2) |
| AC-4 | LANDED (R2) |
| **AC-5** | **LANDED (R3)** — shadow-only; live cutover gated on 14-day accumulation |
| AC-6 | LANDED (R1) |
| AC-10 | partial — 51 pytest cases |
| AC-11 | partial — no plan markers in v3.7.240 |
| AC-12 | LANDED (R0) |
| AC-13 | LANDED (R0) |
| AC-1, AC-7, AC-8, AC-9, AC-14, AC-15 | not started |

**Cumulative: 7 / 15 ACs LANDED. 2 partial. 6 not started.**

## Remaining Work

| Phase | Tasks | ACs |
|---|---|---|
| D | task-d1, task-d2 | AC-7, AC-15 |
| E (continued) | task-e3 | AC-14 (Dashboard parity, heavier) |
| F-body | task-f4 (also needs Phase G g3/g4 done) | AC-8, AC-9 |
| G | task-g1..g6 | AC-1, AC-8, AC-9 |
| Closure | task-h1, task-h2 | AC-10, AC-11 |

8 of 15 ACs still need work (was 9 entering Round 3). **Plan is not complete.**

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: The single test that initially failed (tz-aware `written_at` vs tz-naive `today` in `live_cutover_allowed`) was resolved with a tz-normalize step inside the function rather than coercing callers. This is the same class of "verify arithmetic at the boundary" that Round 1's stale-day-3 failure represented — not yet promoted to a bitlesson because each instance is local-knowledge that pytest catches immediately. If a third tz/calendar arithmetic boundary bug shows up in a future round, it earns an entry titled something like "tz/calendar boundary discipline: always normalize at module boundaries".
