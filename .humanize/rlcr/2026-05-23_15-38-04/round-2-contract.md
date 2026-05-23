# Round 2 Contract

## Round Objective

Land **Phase C — per-asset exit configuration + STRADDLE/SHORT_VOL expiry intrinsic** (task-c1 + task-c2), targeting AC-3 and AC-4. These are tightly coupled: c2's asymmetric-IC intrinsic formula needs the asset string to resolve the right per-asset config in c1, so they ship together.

## In-Scope Tasks This Round

| Task | AC | Tag | Notes |
|------|----|----|-------|
| task-c1 | AC-3 | v3.7.238 | `core/strategy_config.py:get_option_exit_config(asset, strategy)` returning the appropriate cfg dataclass (BCConfig/SPConfig/StraddleConfig/ShortVolConfig); `core/paper_positions.py:simulate_option_exit` accepts `asset` and propagates to all `simulate_*_position`; legacy `cfg=None` path emits `DeprecationWarning` when called without `asset` |
| task-c2 | AC-4 | v3.7.239 | `core/strategies/options_exit.py:force_close_at_expiry` extends `strategy_kind` to support `"long_vol"` (STRADDLE) and `"iron_condor"` (SHORT_VOL with asymmetric `max(call_wing, put_wing) - credit` max_risk); `core/strategies/straddle.py` and `core/strategies/short_vol.py` call `force_close_at_expiry` before db lookup, mirroring the BC/SP wiring from v3.7.232 |

## Out-of-Scope This Round

- Phase D/E/F-bodies/G/closure

## Verification Plan

### task-c1
- Unit-level: `get_option_exit_config("GLD", "SELL PUT")` returns an SPConfig with `profit_target_credit_pct=50` (current default per `core/strategies/sell_put.py`); `get_option_exit_config("SLV", "SELL PUT")` returns a different SPConfig where the per-asset SLV override is applied if present in the registry. Asserted via direct attribute comparison.
- `simulate_sp_position(entry_pricing, signal_date, today_dt, db, asset="GLD")` vs `…, asset="SLV")` returns different `exit_value` for the same legs/spot fixture (only if the registry actually carries different thresholds — Round 2 establishes the resolver; per-asset tuning values themselves remain at whatever current code carries).
- `simulate_option_exit(..., asset=None)` emits a `DeprecationWarning` (caught via `pytest.warns(DeprecationWarning)`).

### task-c2
- Pytest: `tests/test_expiry_intrinsic.py` covering BC/SP/STRADDLE/SHORT_VOL × {today < expiry, today == expiry with close known, today == expiry without close, today > expiry kline missing} = 16 scenarios.
- SHORT_VOL asymmetric-wings fixture: call wing $5 wide, put wing $10 wide, credit $1.50 → `max_risk = max(5, 10) - 1.50 = 8.50`; an at-expiry pin between the short strikes returns the credit (max profit); a pin past the wider put long strike returns full max_risk loss.
- STRADDLE pin at exact $K returns $0 intrinsic (close-to-strike at expiry zeros out both legs).

## Commit Discipline

- `v3.7.238`: task-c1 (resolver + threading + deprecation)
- `v3.7.239`: task-c2 (expiry-intrinsic wiring for STRADDLE/SHORT_VOL)

Local tags only. Tests committed alongside v3.7.239.

## Bitlessons Consulted

- Empty knowledge base; selector returns NONE.
- One round-0/round-1 takeaway worth promoting if it recurs: keep git add scope explicit so per-tag boundaries stay crisp (no bitlesson entry yet — needs a second occurrence to graduate from "hygiene reminder" to "knowledge").

## Round-2 Risk Watch

- **Asset threading completeness** (Codex Round 2 concern): the resolver is useless if any `simulate_*_position` site still falls through to a hard-coded `BCConfig()`/`SPConfig()`. After task-c1, grep all `simulate_bc/sp/straddle/short_vol_position(` and `force_close_at_expiry(` invocations and confirm every site either passes an explicit cfg or resolves via the new registry.
- **`force_close_at_expiry` `strategy_kind` enum** (Codex Round 1+2): the helper currently accepts `"long_call" | "credit_spread"`. Extending to `"long_vol" | "iron_condor"` must include IC max_risk via `max(call_wing, put_wing) - credit_received` per AC-4 and DEC-6; symmetric-wings shortcut is optional, default is asymmetric-aware.
- **STRADDLE close-to-strike edge case**: at expiry both legs' intrinsic = 0; ensure the helper does not divide by zero when computing pnl_pct.
