# Cross-Asset IV-Aware Selector — March 2026 Replay (task-e2, AC-5)

## Question

If the IV-aware `select_gld_sync_strategy` selector landed in v3.7.240 had been live during March 2026, would it have routed the 5 SLV-S cross-asset entries that produced the historical -334.3% loss cluster to SELL PUT instead of the fixed BUY CALL?

## Method

For each of the 5 March 2026 GLD cross-asset entries in `data/positions_ledger.json` (tier=`S-sync`, strategy=`BUY CALL`):

1. Pull `bp_low` for the signal_date from `core.sig_df_history.load_history()` (GLD)
2. Pull GLD `^GVZ` close from yfinance, anchored to the signal_date (NOT wall-clock)
3. Invoke `select_gld_sync_strategy(signal_date, gld_signal_row, gvz_value, gvz_asof_date)`
4. Compare to historical outcome (BC pnl_pct)
5. Use native GLD SP entries in the same month as a proxy for the counterfactual SP P&L path

## Selector Outputs

| Signal Date | bp_low | GVZ | GVZ as-of | Selector → | Reason | Actual BC pnl |
|---|---:|---:|---|---|---|---:|
| 2026-03-03 | -0.111 | 38.8 | 2026-03-03 | **SELL PUT** | DEEP_BREAK_HIGH_IV | -79.1% |
| 2026-03-19 | -0.940 | 31.0 | 2026-03-19 | **SELL PUT** | DEEP_BREAK_HIGH_IV | -73.3% |
| 2026-03-20 | -0.633 | 35.2 | 2026-03-20 | **SELL PUT** | DEEP_BREAK_HIGH_IV | -70.5% |
| 2026-03-23 | -0.595 | 43.4 | 2026-03-23 | **SELL PUT** | DEEP_BREAK_HIGH_IV | -44.4% |
| 2026-03-26 | -0.009 | 45.1 | 2026-03-26 | **SELL PUT** | DEEP_BREAK_HIGH_IV | -67.0% |

**5/5 entries** satisfy both `bp_low ≤ 0.10` AND `GVZ ≥ 25`. Every single one would have been switched to SELL PUT.

Aggregate historical BC outcome: **sum -334.3% (0/5 winners)**.

## Counterfactual SP P&L (proxy)

Native GLD SELL PUT entries in the same month (where GLD's own signal pipeline produced SP, not cross-asset BC):

| Signal Date | SP Strikes | Closed | Outcome | pnl_pct |
|---|---|---|---|---:|
| 2026-03-18 | -P$445/+P$425 | yes | expiry intrinsic (both legs ITM @ expiry $417.29) | -100.0% |
| 2026-03-19 | -P$420/+P$400 | yes | +50% credit | **+36.5%** |
| 2026-03-20 | -P$430/+P$405 | yes | +50% credit | **+32.2%** |
| 2026-03-23 | -P$405/+P$385 | yes | +50% credit | **+62.4%** |
| 2026-03-24 | -P$400/+P$380 | yes | +50% credit | **+35.6%** |

Native March GLD SP aggregate: **n=5, WR=4/5, sum +66.7%**.

## Counterfactual Comparison

| Outcome | Historical BC | Native SP (proxy) | Δ |
|---|---:|---:|---:|
| sum_pnl | **-334.3%** | **+66.7%** | **+401.0 pp** |
| WR | 0/5 | 4/5 | +80 pp |

**The selector would have rerouted all 5 entries to SP. Using same-month native SP as a P&L proxy, the swing is on the order of +400 pp on those 5 trades.**

## Caveats

1. **Small sample**: 5 trades; statistical noise dominates. Treat the +400 pp as directional, not precise.
2. **Strike selection is path-dependent**: cross-asset SP would have used `price_strategy_at("GLD", "SELL PUT", ...)` at the SLV-S trigger date, which may pick strikes slightly different from GLD-native SP that fired on adjacent days. The native-SP proxy is the best available historical match but not an exact counterfactual.
3. **kline_db coverage**: the 3-18 native SP that lost -100% landed because `kline_db` was missing the $445/$425 contracts at expiry and the v3.7.232 expiry-intrinsic fallback computed full $20 spread loss against $417.29 spot. A cross-asset SP placed on 3-3 or 3-26 would likely use different strikes (closer to ATM) where the intrinsic-loss risk is smaller.
4. **Live flip is gated**: v3.7.240 lands the selector as shadow-only (`CROSS_LIVE_CUTOVER=False`). 14 calendar days of shadow log accumulation is required before the live cutover flag can be flipped (per AC-5). This replay is evidence for the eventual flip decision, not a directive to flip immediately.
5. **March's deep break regime is non-stationary**: the 5/5 selector match means March was 100% in the SP regime; in calmer regimes the selector would still route to BUY CALL and the historical +492% BUY CALL sum from v3.7.226 cross-asset analysis remains.

## Recommendation

Promote shadow logging immediately (already done in v3.7.240). After ≥14 calendar days of shadow records, evaluate the live cutover with a fresh replay covering the entire shadow window plus a probabilistic-equivalence check between the shadow log decisions and what `live_cutover=True` would have entered. If the shadow log shows the selector continues to recommend SP in deep-break high-IV conditions (consistent with this March cohort), flip `CROSS_LIVE_CUTOVER=True`.

## Artifacts

- Raw replay CSV: `march_2026_replay.csv` (this directory)
- Code: `core/cross_asset_signal.py:select_gld_sync_strategy` (pure)
- Tests: `tests/test_cross_asset_selector.py` (19 cases, all PASS)
