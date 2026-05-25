# v3.8 Backlog

Deferred items recorded during the v3.7.233..v3.7.250 plan execution and
the Codex code-review of that series. Each item names the v3.8 gate
condition that must hold before it ships.

## Plan-Originated Deferrals

### DEC-7 — out-of-scope optimization items

Confirmed by the user at plan-time (Phase 6 dialogue) and reaffirmed by the
Codex Round 8 audit decision (`gate_passed: false` on calibration cutover).
These are optimization-tier items the original Codex audit flagged but the
plan explicitly excluded.

| Item | Source | Trigger |
|---|---|---|
| OI correction promotion into the main signal chain | Codex audit medium-priority | New cross-asset/OI study round |
| Full risk-controls migration (exposure caps, max_open_per_asset, ≥2 consecutive-loss circuit breakers) | Codex audit medium-priority | After live trading exposes regression patterns |
| yfinance bid/ask mid fallback (instead of last-price) | Codex audit medium-priority | Live trading data-quality review |
| Broader portfolio-level exposure caps | Codex audit medium-priority | Multi-asset roll-out |

## Calibration Cutover (AC-8 follow-on)

The conformal scaler (`core.calibration.apply_rolling_conformal_scaler`)
shipped as **shadow-only** with `gate_passed: false` from the Round 8 gate
report (`data/backtest_history/v3.7.247_calibration_gate/gate_report.md`).
Per-side coverage improves (4-5/5 windows for both GLD and SLV) but joint
`coverage_both` regresses on long windows because shrinking each side
independently reduces the joint event probability.

| Item | Trigger | Acceptance gate |
|---|---|---|
| `build_band()` reads calibrated columns under config flag (review P2#3) | After re-tuning produces gate_passed=true | Config preflight check at app/daemon startup reads `gate_report.md:gate_passed: true` AND `calibration.live_cutover=True`; `build_band()` switches to `*_pct_calibrated` columns; rollback via single flag flip |
| `extend_oos_predictions` writes calibrated columns alongside raw | After cutover decision | Schema migration documented in `data/positions_ledger_meta.json` |
| Re-tune `target_coverage` per side (e.g. 0.90 each so joint ≈ 0.80) | v3.8 round 1 | New gate report carries the chosen target + passes the compound gate |
| Re-tune per-regime classifier integration | After regime ML revisit | Per-regime coverage all-side > 70% per regime |

## Cross-Asset IV-Aware Live Cutover (AC-5 follow-on)

The cross-asset selector (`core.cross_asset_signal.select_gld_sync_strategy`)
shipped as **shadow-only** with shadow log written to
`data/cross_asset_shadow_log.jsonl`. The v3.7.250 preflight check enforces
≥14-day shadow accumulation before allowing `CROSS_LIVE_CUTOVER=true`.

| Item | Trigger | Acceptance gate |
|---|---|---|
| Flip `GOLD_CROSS_LIVE_CUTOVER=true` env var | Shadow log ≥ 14 calendar days AND replay analysis confirms ≥3pp improvement on the cohort | `live_cutover_allowed()` returns True AND replay archive shows positive expectancy |
| Shadow log enrichment — dual-branch P&L estimates per record (review fix P4#2) | Higher fidelity replay needed | Each shadow record carries `branches: {buy_call, sell_put}` estimated 10-day P&L |
| Manifest record (single summary entry, not per-decision) tracking `first_record_at` | After v3.7.250 cutover preflight is in production | Manifest file `data/cross_asset_shadow_manifest.json` written on first record |

## Architectural Improvements (Idea-Draft Alt-2, Alt-4)

These are the Alt-N directions from the gen-idea swarm that the user
chose NOT to take as primary but are clean follow-on consolidations.

| Item | Source | Trigger |
|---|---|---|
| `ExitContext` typed dataclass replacing per-strategy positional cfg | Alt-2 typed-context exploration | After production stabilizes on v3.7.* surface; tightens 43 call sites |
| Look-ahead eradication CI gate | Alt-4 systematic leak detection | Build-time hook on `features_all.parquet` schema diffs |

## Data Freshness Integration Tests (Review P3#2)

| Item | Source | Trigger |
|---|---|---|
| Integration-level test that drives `build_positions_ledger.py` end-to-end across a `PENDING_KLINE` cycle: (1) first run produces pending and clamps waterline; (2) second run with refreshed kline produces the entry without duplicating | Review P3#2 | Production-data-shaped pytest fixture (cached parquet snippets representing kline_db at two timestamps); paired with v3.7.251 dedup invariant tests |
| Tiered FRESH/STALE/FROZEN dashboard UX (vs current binary gate) | DEC-2 lower-bound deferred | When sidebar UX is being touched anyway |

## Code Quality (Codex Review Notes)

| Item | Source | Trigger |
|---|---|---|
| Dashboard parity harness running against real 60-day legacy vs unified replay | Review P3#3 | After v3.7.250 lands; before legacy `run_backtest` removal in v3.8 |
| `core/paper_positions.py:simulate_option_exit` fail-loud on cfg-resolver exception (no silent fallback) | Review P2#1 | Audit pass on production logs for unexpected fallback hits |
| Legacy `core.signals_v2.run_backtest` body removal | Review P3#3 + AC-14 | Parity harness confirms 60-day equivalence on real data |

## Cleanup

| Item | Trigger |
|---|---|
| Delete stale `/Users/yhdong/GoldDash/data/options_history/kline_db/all_klines.parquet` (74 codes, 2026-03-10 stub) | Confirmed unused by production code (grep complete in Round 0) |
| Bash 3.2 compatibility comments in `scripts/validate-patch.sh` + `scripts/eval/audit_plan_markers.sh` removed | When CI Bash version policy formalizes |
