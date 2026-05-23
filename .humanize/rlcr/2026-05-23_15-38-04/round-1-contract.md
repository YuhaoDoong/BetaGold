# Round 1 Contract

## Round Objective

Land **Phase B defensive ingestion guards** (task-b1 + task-b2) targeting AC-6 (Data Freshness Gate). This phase makes kline_db staleness explicit and observable: option entries gated, futures and MTM unaffected, distinct `PENDING_KLINE` vs `NO_CONTRACT` states with deduplication on retry.

## In-Scope Tasks This Round

| Task | AC | Tag | Notes |
|------|----|----|-------|
| task-b1 | AC-6 | v3.7.236 | `pick_liquid_monthly_option(max_fallback_days=7)`; return `None` + `source="PENDING_KLINE"` when exceeded; `price_strategy_at` distinguishes PENDING_KLINE from NO_CONTRACT in return dict |
| task-b2 | AC-6 | v3.7.237 | `core/data_freshness.py` with `kline_db_state(today, max_age_days=3) → FRESH/STALE/FROZEN`; ledger daemon consults gate before option entry block; gating applies only to new option entries; futures + MTM + force_close_at_expiry unaffected; dedup logic so a once-blocked entry resumes at original signal_date (not a fresh row) |

## Out-of-Scope This Round

- Phase C (task-c1/c2): per-asset cfg threading + STRADDLE/SHORT_VOL expiry
- Phase D/E/F-bodies/G/closure

## Verification Plan

### task-b1
- Unit-level: Constructing a `pick_liquid_monthly_option` call with synthetic kline_db whose `max_date < signal_date - 7 trading days` returns `None`; the `price_strategy_at` return dict carries `source="PENDING_KLINE"` (distinct from `"—"` which indicates `NO_CONTRACT`).
- Negative: kline freshness within 7 trading days returns normal `c` dict (no PENDING_KLINE flag).

### task-b2
- Unit-level: `data_freshness.kline_db_state(today=2026-05-23, max_age_days=3)` against the actual db (max=2026-05-06) returns `"FROZEN"` (3+ trading days stale); against today's date returns `"FRESH"`.
- Behavior: `build_positions_ledger.py` invoked with FROZEN state writes no new option entries (verified by row count delta), continues MTM and exit logic, and logs the gate decision.
- Dedup: Two consecutive ledger builds with the same FROZEN signal_date produce only one PENDING_KLINE row (not duplicates).

### pytest regression
- `tests/test_data_freshness.py`: covers (a) FRESH/STALE/FROZEN state mapping, (b) PENDING_KLINE deduplication, (c) futures path exempt from gate.

## Commit Discipline

- `v3.7.236`: task-b1 (pick_liquid_monthly_option + price_strategy_at)
- `v3.7.237`: task-b2 (core/data_freshness.py + ledger_daemon gate + pytest)

Local tags only. Push deferred to user authorization.

## Bitlessons Consulted

- Empty knowledge base; selector returns NONE for both tasks.
- This round may add an entry if a non-obvious resolution arises during integration.
