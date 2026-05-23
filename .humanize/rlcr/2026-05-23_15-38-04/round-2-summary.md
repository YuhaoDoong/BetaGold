# Round 2 Summary

## Status

Round 2 lands **Phase C — per-asset exit configuration + STRADDLE/SHORT_VOL expiry intrinsic** (task-c1 + task-c2), targeting AC-3 and AC-4. Phases D/E/F-body/G/closure still queued. **The full plan is not yet complete.**

## What Was Implemented

### v3.7.238 — per-asset option-exit cfg resolver (task-c1, AC-3)

- `core/strategy_config.py:get_option_exit_config(asset, strategy)` is the new single-point resolver returning the right strategy dataclass (`BCConfig | SPConfig | StraddleConfig | ShortVolConfig`). It applies per-asset overrides from a small module-level registry `_OPTION_EXIT_OVERRIDES`:
  - `(GLD, SELL PUT)`: `profit_target_credit_pct=70.0` (v3.7.184 grid winner)
  - `(SLV, SELL PUT)`: `profit_target_credit_pct=30.0` (v3.7.184 per-asset split)
  - BC / STRADDLE / SHORT_VOL: cross-asset robust, no overrides, dataclass defaults.
  - Field-name validation against the target dataclass prevents silent typos in the registry.
- `core/paper_positions.py:simulate_option_exit` gains an `asset: Optional[str] = None` keyword. When `asset is None`, the function emits `DeprecationWarning` (the legacy silent-default path is now visible). When `asset` is supplied, it imports `get_option_exit_config` once and passes the resolved cfg into each `simulate_*_position`.
- All three production call sites in `scripts/build_positions_ledger.py` now pass `asset=`:
  - Main option entry block (line 282): `asset=asset` (loop variable)
  - Exit-only re-evaluate (line 408): `asset=asset` (row's asset)
  - Cross-asset sync block (line 553): `asset="GLD"` (cross target is always GLD)
- Research scripts (`scripts/options_modern_exit_compare.py`, `scripts/cross_asset_strategy_pnl.py`, `scripts/walk_forward_exits.py`, `scripts/full_history_backtest.py`, `scripts/backtest/layer2_strategy/*/run_all.py`) still call without `asset` and will emit `DeprecationWarning`; per the Round 2 contract, those migrate in subsequent rounds rather than this one.
- pytest: `tests/test_per_asset_cfg.py` — 6 cases: BC no override, SP GLD vs SLV split, Straddle/ShortVol type check, unknown-strategy raise, simulate_option_exit `asset=None` warns, simulate_option_exit `asset=...` does NOT warn. All PASS.
- Tag: `v3.7.238` (local).

### v3.7.239 — STRADDLE/SHORT_VOL expiry-intrinsic force-close + asymmetric IC max_risk (task-c2, AC-4)

- `core/strategies/options_exit.py:force_close_at_expiry` extends `strategy_kind` from {`"long_call"`, `"credit_spread"`, `"long_vol"`} to also support `"iron_condor"`:
  - `"long_vol"` (STRADDLE): same `sum(qty × intrinsic)` math as `long_call`, with zero-entry-value guard so a degenerate fixture does not divide by zero.
  - `"iron_condor"` (SHORT_VOL): debit-to-close = `-sum(qty × intrinsic)`. Internally parses the 4 leg strikes by type+sign, computes `call_wing = abs(long_call_K − short_call_K)` and `put_wing = abs(short_put_K − long_put_K)`, then `max_risk_eff = max(call_wing, put_wing) − entry_value`. This is the AC-4 + DEC-6 asymmetric-aware formula. Any `max_risk` kwarg is ignored in the IC branch (the helper derives it from legs to keep callers from mis-passing the symmetric-wing assumption).
- `core/strategies/straddle.py:simulate_straddle_position` calls `force_close_at_expiry(..., strategy_kind="long_vol")` before the kline lookup, mirroring the BC/SP wiring landed in v3.7.232.
- `core/strategies/short_vol.py:simulate_short_vol_position` calls `force_close_at_expiry(..., strategy_kind="iron_condor")` likewise.
- pytest: `tests/test_expiry_intrinsic.py` — 17 cases covering all four strategies × the four expiry states declared in AC-4:
  - **BC**: today<expiry (None) / today==expiry (None, awaiting-close) / today>expiry kline missing (intrinsic close at $12.29 against the actual 5-15 GLD close $417.29) / unparseable code (None)
  - **SP credit spread**: today<expiry (None) / today==expiry (None) / today>expiry kline missing (full $20 wing loss, pnl -100%) / omitted max_risk (fallback to entry_value as denominator)
  - **STRADDLE long_vol**: today<expiry / today==expiry / today>expiry kline missing (pin near $417 ATM → ~$0.29 intrinsic, pnl -97.1%) / zero entry value safe
  - **SHORT_VOL IC**: today<expiry / today==expiry / symmetric IC profit pin (pnl +42.86%) / **asymmetric IC profit pin** (pnl +17.65% — verifies the wider $10 put wing is used as max_risk denominator, not the smaller $5 call wing) / unparseable codes
- Total project pytest: **32/32 PASS in 0.14s** (R0 2 + R1 7 + R2 6 + R2 17).
- Tag: `v3.7.239` (local).

## Files Changed

### Modified
- `core/strategy_config.py` — `_OPTION_EXIT_OVERRIDES` registry + `get_option_exit_config` resolver
- `core/paper_positions.py` — `simulate_option_exit` `asset` kwarg + DeprecationWarning + cfg resolution + cfg passthrough to all 4 strategy modules
- `scripts/build_positions_ledger.py` — 3 call sites pass `asset=`
- `core/strategies/options_exit.py` — extend `strategy_kind` to `iron_condor` with asymmetric max_risk
- `core/strategies/straddle.py` — call `force_close_at_expiry` before db lookup
- `core/strategies/short_vol.py` — same

### Created
- `tests/test_per_asset_cfg.py` — 6 cases
- `tests/test_expiry_intrinsic.py` — 17 cases

## Validation

- pytest full suite: **32 passed in 0.14s**.
- Key arithmetic verified:
  - Asymmetric IC ($5 call wing / $10 put wing, $1.50 credit): max_risk = $10 − $1.50 = $8.50; profit pin → pnl = $1.50 / $8.50 × 100 = **+17.65%** (NOT the $5-wing-based +42.86%).
  - Symmetric IC ($5/$5 wing): max_risk = $5 − $1.50 = $3.50; same profit pin → pnl = **+42.86%**.
  - STRADDLE pin at K=417 with spot $417.29: cur_value = $0.29 (call intrinsic only); pnl = (0.29/10 − 1) × 100 = **−97.1%**.

## AC Status Delta

| AC | Status |
|----|--------|
| AC-2 | LANDED (Round 0) |
| **AC-3** | **LANDED (Round 2)** |
| **AC-4** | **LANDED (Round 2)** |
| AC-6 | LANDED (Round 1) |
| AC-10 | partial — pytest harness + 32 cases; closure tests rolling in |
| AC-11 | partial — no plan markers in v3.7.236-239 code |
| AC-12 | LANDED (Round 0) |
| AC-13 | LANDED (Round 0) |
| AC-1, AC-5, AC-7, AC-8, AC-9, AC-14, AC-15 | not started |

**Cumulative: 6 / 15 ACs LANDED (AC-2, AC-3, AC-4, AC-6, AC-12, AC-13). 2 partial. 7 not started.**

## Remaining Work

| Phase | Tasks | ACs |
|---|---|---|
| D | task-d1, task-d2 | AC-7, AC-15 |
| E | task-e1, task-e2, task-e3 | AC-5, AC-14 |
| F-body | task-f3 (already partly via test_per_asset_cfg), task-f4 | AC-8, AC-9 |
| G | task-g1..g6 | AC-1, AC-8, AC-9 |
| Closure | task-h1, task-h2 | AC-10, AC-11 |

9 of 15 ACs still need work (was 11 of 15 entering Round 2). **Plan is not complete.**

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 2 landed cleanly on first attempts. The "tag boundary smudge" risk flagged in Round 0's BitLesson Delta did not recur this round — both tags carved a clean diff. If a third clean round runs without recurrence we can consider it project hygiene rather than something needing a BitLesson entry.
