# Round 1 Summary

## Status

Round 1 lands **Phase B defensive ingestion guards** (task-b1 + task-b2), both targeting AC-6 (Data Freshness Gate). Phases C/D/E/F-body/G/closure still queued. **The full plan is not yet complete.**

## What Was Implemented

### v3.7.236 — kline_db 新鲜度硬闸 (task-b1, AC-6)

- `core/paper_positions.py`: added module-level `KLINE_MAX_FALLBACK_DAYS = 7` constant + `_kline_db_freshness_status(signal_date) → (status, gap_days)` helper returning `FRESH | PENDING_KLINE | NO_DB`.
- `pick_liquid_monthly_option` gains a `max_fallback_days=7` keyword; when the gap to the nearest available kline exceeds the cap it returns `None` (defensive fail).
- `price_strategy_at` consults the freshness status upfront for all option strategies. PENDING_KLINE → `source="PENDING_KLINE (Nd stale)"`, empty legs, early return (lets ledger daemon defer). NO_DB → `source="NO_KLINE_DB"`. FRESH → normal flow.
- Live validation against the real db (`today=2026-05-23`, `db.max=2026-05-06`, gap=17 trading days): status correctly returns `PENDING_KLINE` and `pick_liquid_monthly_option` returns `None`. Boundary tests confirmed: signal_date=5-12 (gap=6) is FRESH, signal_date=5-14 (gap=8) is PENDING_KLINE.
- Tag: `v3.7.236` (local).

### v3.7.237 — 数据新鲜度状态机 + ledger 期权入口闸 (task-b2, AC-6)

- `core/data_freshness.py` (new module):
  - `FreshnessRecord` dataclass with `source / state / max_date / gap_trading_days / as_of` and `to_dict()`.
  - `kline_db_state(today, fresh_max_days=2, frozen_min_days=3, db_path=…)` returns the record. Tiered semantics: FRESH for gap ≤ 2; STALE for gap == 3; FROZEN for gap > 3; MISSING for absent/empty parquet.
  - `gate_new_option_entry()` convenience helper: `allow = state in (FRESH, STALE)`; FROZEN and MISSING block new entries.
  - Trading-day gap computed via `pd.bdate_range` (Mon–Fri); adequate at the 2/3 day decision boundary.
  - Two thresholds intentionally layered with v3.7.236: this module is for **daemon-level observability + dashboard banner**; `paper_positions._kline_db_freshness_status` (binary FRESH/PENDING_KLINE) is for **per-entry pricing fallback**. They may diverge.
- `core/ledger_daemon.py:_rebuild_loop`: each rebuild iteration now logs the current `kline_db` state to `/tmp/ledger_daemon.log` for operational visibility.
- `scripts/build_positions_ledger.py`: the option-entry block and the cross-asset sync block now print `[freshness] skip {asset} {strat} @ {date}: {source}` when the v3.7.236 source flag indicates `PENDING_KLINE` or `NO_KLINE_DB`. Previously this skip was silent.
- Futures path (Binance live data), MTM/exit on existing positions, and `force_close_at_expiry` are exempt from the gate (Codex's AC-6 scope check).
- `tests/test_data_freshness.py`: 7 pytest cases — FRESH window, STALE band (`gap=3` boundary), FROZEN, MISSING, gate-blocks-frozen, gate-allows-fresh, `to_dict()` roundtrip.
- Total pytest: **9/9 PASS** (2 from Round 0 + 7 new).
- Tag: `v3.7.237` (local).

## Files Changed

### Modified
- `core/paper_positions.py` — `KLINE_MAX_FALLBACK_DAYS`, `_kline_db_freshness_status`, `pick_liquid_monthly_option` cap, `price_strategy_at` upfront gate
- `core/ledger_daemon.py` — freshness log line per rebuild
- `scripts/build_positions_ledger.py` — explicit `[freshness] skip` print on option-entry block and cross-asset block

### Created
- `core/data_freshness.py`
- `tests/test_data_freshness.py`

## Validation

- pytest: `conda run -n gold python -m pytest tests/ -v` → **9 passed in 0.11s**.
- Live runtime check against the real db: `_kline_db_freshness_status(today=2026-05-23)` → `("PENDING_KLINE", 17)`; `price_strategy_at(GLD, BUY CALL, today, …)` → `source="PENDING_KLINE (17d stale)"`, legs=`[]`.
- Boundary check: gap=6 FRESH, gap=8 PENDING_KLINE (correctly straddles `max_fallback_days=7`).

## AC Status Delta

| AC | Status |
|----|--------|
| AC-2 | LANDED (Round 0) |
| AC-6 | LANDED (Round 1) |
| AC-10 | partial — pytest harness + 9 cases; per-AC tests still rolling in |
| AC-11 | partial — no plan markers in v3.7.236-237 code |
| AC-12 | LANDED (Round 0) |
| AC-13 | LANDED (Round 0) |
| AC-1, AC-3, AC-4, AC-5, AC-7, AC-8, AC-9, AC-14, AC-15 | not started |

Cumulative: 4 / 15 ACs LANDED (AC-2, AC-6, AC-12, AC-13). 2 partial (AC-10, AC-11). 9 not started.

## Remaining Work

### Phase C — Per-asset cfg + Straddle/ShortVol expiry (next round)
- task-c1 (AC-3): `core/strategy_config.py:get_option_exit_config(asset, strategy)` resolver; thread asset down through `simulate_option_exit` → `simulate_*_position`
- task-c2 (AC-4): `force_close_at_expiry` → STRADDLE (long_vol intrinsic = sum of leg intrinsics) + SHORT_VOL (IC max wing) — deprecate `cfg=None` silent fallback

### Phase D — Derived metrics
- task-d1 (AC-15): `max_move_{h}d` explicit `entry_i/exit_i` window helper
- task-d2 (AC-7): Layer 2 sample disposition reporting + per-leg DTE filter

### Phase E — Cross-asset IV-aware + Dashboard parity
- task-e1 (AC-5): `select_gld_sync_strategy` pure function + caller-side shadow log
- task-e2 (AC-5): analyze — replay March SLV-S triggers
- task-e3 (AC-14): Dashboard `run_backtest` wrapper + parity assertion

### Phase F — Test bodies
- task-f2 (AC-4): 16-scenario expiry intrinsic tests
- task-f3 (AC-3): per-asset cfg test
- task-f4 (AC-8, AC-9): calibration scaler + retrain trigger tests

### Phase G — Model calibration
- task-g1..g6 (AC-1, AC-8, AC-9)

### Closure
- task-h1 (AC-10): `make validate-patch` + normalization
- task-h2 (AC-11): final grep audit

11 of 15 ACs still need work. **Plan is not complete.**

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 1 work landed cleanly on first attempt. One transient test failure (test_stale_window expected gap=3 but actual gap=4 between 5-18 Mon and 5-22 Fri) was a date-arithmetic miscount on my side, fixed by adjusting the test fixture date (5-21 Thu gives gap=3). Not a project-wide pattern worth a bitlesson entry; standard "verify arithmetic for boundary tests" hygiene.
