# Round 3 Contract

## Round Objective

Land **Phase E partial — cross-asset IV-aware selector + shadow log + replay analysis** (task-e1 + task-e2), targeting AC-5. Phase E2 task-e3 (Dashboard `run_backtest` deprecation, AC-14) is heavier (intraday StopLoss/Pullback/ACTIVE semantic parity) and is deferred to a dedicated round so its parity-harness work does not contaminate this round's selector landing.

## In-Scope Tasks This Round

| Task | AC | Tag | Notes |
|------|----|----|-------|
| task-e1 | AC-5 | v3.7.240 | `core/cross_asset_signal.py:select_gld_sync_strategy(signal_date, gld_signal_row, gvz_value, gvz_asof_date)` pure function returning `{strategy, reason, gvz_status}`; caller-side `write_shadow_record(...)` invoked from `scripts/build_positions_ledger.py`; `shadow_logging` defaults `True`, `live_cutover` defaults `False`; 14-day shadow accumulation manifest required before flip; pytest for purity + truth table + 14-day gate |
| task-e2 | AC-5 | analyze | Replay all March 2026 SLV-S triggers through the new selector (with `gvz_value` / `gvz_asof_date` from `^GVZ` history); document the would-be P&L delta versus the historical BC outcomes (5/5 loss, sum -334%) into `data/backtest_history/v3.7.240_cross_asset_replay/`; report whether the IV-aware switch would have produced ≥3 SP entries |

## Out-of-Scope This Round

- task-e3 (AC-14) Dashboard `run_backtest` deprecation
- Phase D/F-body/G/closure

## Verification Plan

### task-e1 (coding)
- pytest `tests/test_cross_asset_selector.py`:
  - **purity**: selector reads zero filesystem state; selector is deterministic given the same inputs (run twice, assert equal results). A hermetic test using `monkeypatch` to fake-fail `open()` ensures no I/O sneaks in.
  - **truth table**: GVZ None → BUY CALL with reason=GVZ_UNAVAILABLE; GVZ stale (`signal_date - gvz_asof_date > 2 trading days`) → BUY CALL; deep break + high IV (`bp_low ≤ 0.10 AND gvz ≥ 25`) → SELL PUT with reason=DEEP_BREAK_HIGH_IV; default → BUY CALL.
  - **`signal_date` reference frame**: replaying with `signal_date=2026-03-15` and `gvz_asof_date=2026-03-14` returns FRESH GVZ regardless of wall clock (NOT `today` based).
  - **March 5/5 cluster expectation**: a SP-eligible fixture (bp_low=0.05, GVZ=28) returns SELL PUT; a BC-default fixture (bp_low=0.30, GVZ=18) returns BUY CALL.
- pytest `tests/test_cross_asset_shadow_gate.py`:
  - Shadow log writer appends valid JSON records to a tmp path; running twice with `shadow_logging=True` produces two records.
  - `live_cutover_allowed(shadow_log_path, today)` returns False when manifest has <14 days of records; True when ≥14 days.

### task-e2 (analyze)
- Codex consultation reads `data/positions_ledger.json` for the 5 March BC cross-asset entries (sum -334%), pulls GVZ + `bp_low` for those signal dates, applies the selector logic offline, and writes a report into `data/backtest_history/v3.7.240_cross_asset_replay/REPLAY.md` covering:
  - Per-signal-date: original BC outcome vs would-be SP outcome (based on the selector's verdict)
  - Aggregate: counterfactual sum_pnl% if SP had been used where the selector says SP
  - Caveat: actual SP P&L is path-dependent on kline_db; this analyze step produces estimates not live trades

## Commit Discipline

- `v3.7.240`: task-e1 code + tests
- analyze (task-e2): report file committed alongside or separately based on file size; archive directory under `data/backtest_history/v3.7.240_cross_asset_replay/`

## Bitlessons Consulted

- Empty knowledge base; selector returns NONE for both tasks.
- One Round 0 risk worth tracking again: tag boundary discipline. v3.7.238 had a smudge into v3.7.235 prerequisite; v3.7.239 came out clean; v3.7.240 should also stay surgical (only cross_asset_signal.py + caller + tests, no shared-file overflow).

## Round 3 Risk Watch

- **Selector purity**: tempting to inline GVZ fetch inside the function ("convenience") — that would tank purity and re-introduce wall-clock leak. Selector MUST take pre-resolved values; fetching belongs to the caller.
- **bp_low / GVZ NaN handling**: real ledger rows sometimes have NaN bp_low or NaN GVZ during data outages. The selector must treat NaN GVZ as "stale/missing → BUY CALL" without raising.
- **Shadow log JSONL append concurrency**: ledger daemon and ad-hoc scripts might both write. Use append-only line-buffered writes with `flush=True`; ordering does not matter for accumulation gate.
- **March 5 sample size**: 5 cluster trades is small; the analyze step is illustrative, not statistically significant. Report should call this out explicitly.
