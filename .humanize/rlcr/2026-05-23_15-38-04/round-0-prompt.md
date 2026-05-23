Read and execute below with ultrathink

## Goal Tracker Setup (REQUIRED FIRST STEP)

Before starting implementation, you MUST initialize the Goal Tracker:

1. Read @/Users/yhdong/GoldDash/.humanize/rlcr/2026-05-23_15-38-04/goal-tracker.md
2. If the "Ultimate Goal" section says "[To be extracted...]", extract a clear goal statement from the plan
3. If the "Acceptance Criteria" section says "[To be defined...]", define 3-7 specific, testable criteria
4. Populate the "Active Tasks" table with MAINLINE tasks from the plan, mapping each to an AC and filling Tag/Owner
5. Record any already-known side issues in either "Blocking Side Issues" or "Queued Side Issues"
6. Write the updated goal-tracker.md

## Round Contract Setup (REQUIRED BEFORE CODING)

Before starting implementation, create @/Users/yhdong/GoldDash/.humanize/rlcr/2026-05-23_15-38-04/round-0-contract.md with:

1. **One mainline objective** for this round
2. **Target ACs** (1-2 ACs only)
3. **Blocking side issues in scope** for this round
4. **Queued side issues out of scope** for this round
5. **Round success criteria**

Use this contract to keep the round focused. Do NOT let non-blocking bugs or cleanup work replace the mainline objective.

**IMPORTANT**: The IMMUTABLE SECTION can only be modified in Round 0. After this round, it becomes read-only.

---

## Implementation Plan

For all tasks that need to be completed, please use the Task system (TaskCreate, TaskUpdate, TaskList).

Every task MUST start with exactly one lane tag:
- `[mainline]` for plan-derived work that directly advances the round objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` tasks are the primary success condition for the round
- `[blocking]` tasks may be resolved in the round only if they truly block mainline progress
- `[queued]` tasks must NOT become the round objective and do NOT need to be cleared before moving on
- If a new issue is not blocking the current objective, tag it `[queued]` and keep moving on the mainline

## Task Tag Routing (MUST FOLLOW)

Each task must have one routing tag from the plan: `coding` or `analyze`.

- Tag `coding`: Claude executes the task directly.
- Tag `analyze`: Claude must execute via `/humanize:ask-codex`, then integrate Codex output.
- Keep Goal Tracker "Active Tasks" columns **Tag** and **Owner** aligned with execution (`coding -> claude`, `analyze -> codex`).
- If a task has no explicit tag, default to `coding` (Claude executes directly).

# GLD/SLV Trading System Correctness Floor + DL Range Calibration Repair

## Goal Description

Repair the GLD/SLV trading system's correctness floor and the DL Range predictor's calibration floor through a sequence of small, independently verifiable patches that each (a) target one root cause, (b) ship with a focused validation script or pytest case, (c) can be reverted without cascading state corruption, and (d) integrate cleanly with the existing "改 cfg → 跑 grid → 归档版本" workflow described in `CLAUDE.md`. Concretely, the plan covers exactly the behaviors enumerated in this document's Acceptance Criteria: production regime look-ahead removal, per-asset exit configuration threading, expiry-intrinsic force-close for all four option strategies, data-freshness gating on option entries, cross-asset IV-aware strategy selection with shadow gating, Layer 2 sample-disposition reporting and per-leg DTE filtering, calibration audit using the correct 5-day forward label definition, conformal scaler with horizon-aware maturity lag and shadow-only-then-gated cutover, calibration-gated retrain trigger with hysteresis + cooldown, pytest harness, SP fallback sign correction, `entry_spot` schema migration, and Dashboard `run_backtest` deprecation with intraday parity. The four optimization-tier items the original Codex audit flagged (OI correction promotion into the main chain, full risk controls migration, comprehensive yfinance bid/ask mid fallback, broader exposure caps) are explicitly **out of scope** for this plan and tracked separately. Current calibration baseline (GLD v2 OOS, 113 trading days ending 2026-05-13): predicted band [-4.76%, +5.99%] vs realized [-2.86%, +3.08%] (5-day forward H/L vs t-day close per `src/models/train_dl_range.py:build_targets`), width ratios 1.95× / 1.66×, coverage 54.9% versus 80% training target. **Calibration goal is coverage repair (raise to training target band) and tail-distribution match, not unconditional band narrowing — narrowing without coverage repair would further reduce coverage.** The 3 March GLD BUY CALL loss cluster (5/5, sum -334%, 100% cross-asset SLV-S sync triggered) must be diagnosable to the cross-asset rule + IV regime interaction after the cross-asset selector patches land; calibration is a contributing but secondary factor.

## Acceptance Criteria

- AC-1: **Calibration Audit Is Reproducible and Uses the Correct Label Definition**
  - Positive Tests (expected to PASS):
    - Running `scripts/eval/model_calibration_audit.py --asset GLD --start 2025-12-01 --end 2026-05-13` against `data/models/dl_range_v2_oos.parquet` produces a per-month report whose `actual_upper_pct` and `actual_lower_pct` aggregate to the same values when re-derived from `src.models.train_dl_range.build_targets()` semantics (5-day forward max-high / min-low versus t-day close).
    - The report's documented label definition explicitly cites the 5-day forward window and matches the parquet's `actual_*_pct` columns within a ±0.01 percentage-point tolerance.
    - Running the same audit on `data/models/dl_range_slv_oos.parquet` produces a separate report with the SLV-specific label definition documented.
  - Negative Tests (expected to FAIL):
    - An audit implementation that computes single-day overnight high/low (e.g., `(High / Close.shift(1) - 1) * 100`) and reports those as "realized" fails the label-definition equivalence assertion.
    - An audit that omits coverage and width-ratio columns or reports only an aggregate mean (no per-month / per-regime breakdown) is rejected.

- AC-2: **Production Regime Classifier Has No Forward Lookback**
  - Positive Tests (expected to PASS):
    - Grep across `core/`, `scripts/`, and `app.py` shows every production-path `RegimeClassifier(...)` invocation explicitly passes `min_hold_days=1` or reads it from a configuration that resolves to 1 in production.
    - The default value of `RegimeClassifier.__init__`'s `min_hold_days` parameter is 1.
    - A diff of production sig_df before vs after the change shows the regime label at day t never depends on data after day t (verified by re-classifying truncated history and comparing).
  - Negative Tests (expected to FAIL):
    - A research-only script that legitimately needs `min_hold_days=20` for retrospective regime tagging must be on an explicit allow-list (research path); any path not on the list fails the audit.
    - Removing the explicit `min_hold_days=1` from a production file (e.g., `app.py:112`) is detected by the audit script.

- AC-3: **Exit Simulation Receives Per-Asset Configuration**
  - Positive Tests (expected to PASS):
    - `core/paper_positions.py:simulate_option_exit(entry_pricing, signal_date, today_dt, db, strategy, asset, ...)` accepts `asset` and propagates it to `simulate_bc/sp/straddle/short_vol_position`.
    - For an identical legs/entry pricing fixture, `simulate_sp_position(asset="GLD")` and `simulate_sp_position(asset="SLV")` return different exit thresholds reflecting GLD's `profit_target_credit_pct=70` versus SLV's `30` (or whichever values the chosen registry holds for the current release).
    - The asset-to-exit-config mapping lives in exactly one resolver (e.g., `core/strategy_config.py:get_option_exit_config(asset, strategy)` or an explicit registry); call sites do not duplicate this lookup logic.
  - Negative Tests (expected to FAIL):
    - Calling `simulate_bc_position(...)` without an asset (or with `asset=None`) silently falling back to a global `BCConfig()` default fails: implementation must either raise or emit a deprecation warning and log the fallback into a dedicated stream.
    - Two call sites that construct their own ad-hoc `BCConfig()` rather than going through the resolver fail the registry-discipline audit.

- AC-4: **Expiry-Intrinsic Force-Close Covers All Four Option Strategies**
  - Positive Tests (expected to PASS):
    - Given `today_dt > expiry_dt` parsed from leg codes AND `kline_db` lacking the contract codes, each of `simulate_bc/sp/straddle/short_vol_position` returns `is_closed=True` with `exit_date=expiry_dt`, `exit_reason` containing "expiry intrinsic", and an `exit_value` computed from underlying spot at `expiry_dt` close via the strategy-specific intrinsic formula:
      - long-call: `max(S − K, 0)`
      - credit spread: `short_put_intrinsic − long_put_intrinsic` (debit to close)
      - long-vol / STRADDLE: `long_call_intrinsic + long_put_intrinsic`
      - short-vol IC: short-strikes intrinsic minus long-wing intrinsic, computed leg-by-leg with `qty` signs; `max_risk = max(call_wing_width, put_wing_width) − credit_received` (worst-case wing); when wings are confirmed symmetric by leg-code inspection, the implementation MAY assert symmetry and use either value.
    - For `today_dt == expiry_dt` AND kline missing AND underlying spot close at `expiry_dt` available from the ETF daily CSV (loaded via `core/strategies/options_exit.py:spot_close_on_or_before`): intrinsic close runs and `exit_date` is set to `expiry_dt`.
    - For `today_dt == expiry_dt` AND underlying spot close at `expiry_dt` NOT yet available (intraday, post-market data not yet written): the position remains `is_closed=False` with `pnl_pct` set to most-recent MTM and an explicit `state="AWAITING_EXPIRY_CLOSE"` flag in the return dict.
    - A pytest fixture covers 4 strategies × {today < expiry, today = expiry with close known, today = expiry without close, today > expiry with kline missing} = 16 scenarios and all pass.
  - Negative Tests (expected to FAIL):
    - A STRADDLE position with `today > expiry_dt` and missing kline that returns `is_closed=False` fails.
    - A SHORT_VOL position whose intrinsic calculation uses `min(call_wing_width, put_wing_width)` for asymmetric wings (where the wider side is the true loss) fails the IC max_risk assertion on an asymmetric-wings test fixture.
    - A position closed at `expiry_dt` using a stale underlying spot from earlier than `expiry_dt` (when the actual `expiry_dt` close exists in the ETF CSV) fails the close-source-date assertion.

- AC-5: **Cross-Asset Strategy Selector Is Pure, IV-Aware, and Shadow-Gated**
  - Positive Tests (expected to PASS):
    - `select_gld_sync_strategy(signal_date, gld_signal_row, gvz_value, gvz_asof_date)` is a pure function (no I/O, no globals, deterministic) defined in `core/cross_asset_signal.py` that returns a structured decision dict `{"strategy": str, "reason": str, "gvz_status": str}` with the following truth table evaluated against `signal_date` (NOT against wall-clock `today`):
      - `gvz_value` is `None` OR `(signal_date - gvz_asof_date)` exceeds 2 trading days: return `{"strategy": "BUY CALL", "reason": "GVZ_UNAVAILABLE", "gvz_status": "stale" | "missing"}`.
      - `gld_signal_row.bp_low ≤ 0.10 AND gvz_value ≥ 25`: return `{"strategy": "SELL PUT", "reason": "DEEP_BREAK_HIGH_IV", "gvz_status": "fresh"}`.
      - Otherwise: return `{"strategy": "BUY CALL", "reason": "DEFAULT", "gvz_status": "fresh"}`.
    - The shadow log writer is a separate caller-side function (e.g., `core/cross_asset_signal.py:write_shadow_record(...)` invoked by `scripts/build_positions_ledger.py`); the pure selector returns only the decision and never touches the filesystem.
    - `scripts/build_positions_ledger.py` consumes the selector exclusively for cross-asset entries (no inline `CROSS_STRATEGY` constant in the live path) and is the sole site responsible for invoking the shadow log writer.
    - Two independent flags govern behavior: `shadow_logging` (default `True`, the caller writes both-branch P&L estimates to `data/cross_asset_shadow_log.jsonl` regardless of live behavior) and `live_cutover` (default `False`, controls whether the selector's output flows into the ledger or only into the shadow log).
    - Replaying the March 2026 SLV-S triggers under the selector (with `gvz_value` and `gvz_asof_date` from `^GVZ` history) yields ≥3 SP entries (instead of the historical 5 BC) for the days where GLD GVZ ≥25.
    - Flipping `live_cutover=True` is allowed only after `shadow_logging` has accumulated ≥14 calendar days of dual-branch records, demonstrated by a manifest entry in `data/cross_asset_shadow_log.jsonl` showing `first_record_at` ≥ 14 days prior to the flip request.
  - Negative Tests (expected to FAIL):
    - A selector implementation that accesses `RegimeClassifier()`, opens a file, or imports a module that touches the filesystem fails the purity test (verified by a hermetic pytest with mocked file/os modules).
    - A selector that uses wall-clock `datetime.now()` to determine `gvz_status` (instead of computing from `signal_date - gvz_asof_date`) fails the "decisions are reproducible across replay time" assertion.
    - Flipping `live_cutover=True` without the 14-day shadow accumulation manifest fails the gate.
    - A `gvz_value=None` input that returns `"SELL PUT"` (instead of the `"BUY CALL"` fallback) fails the missing-data-policy test.

- AC-6: **Data Freshness Gate Blocks Stale Option Entries Without Blocking Other Flows**
  - Positive Tests (expected to PASS):
    - When `kline_db max_date < today - 3 trading days`, the ledger daemon (and `build_positions_ledger.py`) writes no new option entries; existing positions continue MTM and exit logic; dashboard sidebar shows a 🟡/🔴 staleness indicator with the gap in trading days.
    - When `kline_db max_date ≥ today - 2 trading days`, new option entries are permitted.
    - Futures entries (driven by Binance live data, not kline_db) and force_close_at_expiry intrinsic closes are unaffected by kline staleness.
    - Each blocked entry is logged with `status="PENDING_KLINE"` (distinct from `"NO_CONTRACT"` for liquidity reasons) and the daemon retries on the next refresh once freshness is restored, with deduplication so the same signal_date does not produce two entries.
  - Negative Tests (expected to FAIL):
    - A blocked option entry that re-enters the ledger as a fresh row when kline catches up (instead of resuming the original signal_date) fails the deduplication assertion.
    - A futures-only run that fails because kline is stale fails the scope assertion.

- AC-7: **Layer 2 Backtest Reports Full Sample Disposition**
  - Positive Tests (expected to PASS):
    - `scripts/backtest/layer2_strategy/*/run_all.py` outputs include columns `n_signal, n_entered, n_closed, n_open, n_skipped_stale, n_skipped_no_contract` per window in addition to the existing P&L metrics.
    - Sample-restriction logic uses per-leg DTE (e.g., `entry_date + leg_max_dte + hold_buffer ≤ kline_max_date`) rather than a single signal-date offset.
    - Re-running the existing trailing 1y/6m/3m windows after the patch produces a report where `n_closed + n_open + n_skipped_* = n_signal`.
  - Negative Tests (expected to FAIL):
    - A report that omits `n_open` or `n_skipped_*` columns fails.
    - A sample filter using `signal_date ≤ kline_max - max_dte - hold_buffer` (the original simplistic estimate) that produces `n_signal ≠ n_closed + n_open + n_skipped_*` fails reconciliation.

- AC-8: **Calibrated Bands Shadow-First, Live Cutover Gated, Horizon-Aware Maturity Lag**
  - Positive Tests (expected to PASS):
    - `core/calibration.py` exposes a pure `apply_rolling_conformal_scaler(dates, pred_upper, pred_lower, actual_upper, actual_lower, horizon=5, window=60, target_coverage=0.80)` that returns `(pred_upper_calibrated, pred_lower_calibrated)` for each `dates[t]` using only residuals whose 5-day forward label window has fully matured strictly before `dates[t]` — that is, the residual at source date `s` is eligible iff its `label_end_date(s) < dates[t]`, where `label_end_date(s) = s + horizon_trading_days`. Implementation enforces this through an explicit `eligible = label_end_dates < calibration_as_of_date` filter on the residual pool; no generic `shift(1)` is used.
    - The calibration objective is coverage repair: the scaler selects scaling factors `(s_upper, s_lower)` that, on the matured residual window, raise empirical coverage toward `target_coverage` while reporting (but not mandating) the resulting width change. The implementation MAY widen bands when coverage is below target (current state), narrow them when coverage exceeds target, or shift them asymmetrically when one tail is the dominant error source.
    - Two flags govern behavior: `calibration.shadow_logging` (default `True`, writes calibrated columns to `*_oos.parquet` for audit) and `calibration.live_cutover` (default `False`, controls whether `build_band()` reads calibrated vs raw columns); `shadow_logging` may be `True` while `live_cutover` is `False`.
    - `extend_oos_predictions` writes both raw and calibrated columns when `shadow_logging=True`.
    - Layer 1 directional grid (`scripts/backtest/layer1_signal/directional/run_all.py`) accepts a `--use-calibrated` flag; running with and without it across trailing 10y/5y/3y/1y windows produces side-by-side scoreB AND coverage comparisons.
    - Cutover gate is met when BOTH: calibrated scoreB ≥ raw scoreB in ≥3 of 4 windows AND no window worse than -10%; AND calibrated coverage moves toward `target_coverage` (closer than raw) in ≥3 of 4 windows.
    - Only after the gate is documented as passing in `data/backtest_history/v3.7.246_calibration/gate_report.md` may `live_cutover=True` be set; the flip is implemented as a config preflight check at app/daemon startup that reads the report's `gate_passed: true` field, not as a runtime import-time assertion inside `core/calibration.py`.
  - Negative Tests (expected to FAIL):
    - A scaler implementation that includes any residual whose `label_end_date >= calibration_as_of_date` (i.e., uses 5-day forward actuals not yet fully realized at calibration time) fails the maturity-lag assertion on a synthetic fixture with a leaked future label.
    - Flipping `live_cutover=True` without a `gate_passed: true` field in `gate_report.md` is rejected by the startup preflight check.
    - A test where the scaler is asked to narrow bands while coverage is below target AND no window improves coverage by more than 0% fails the coverage-repair-direction assertion (band narrowing as a goal in itself is rejected).

- AC-9: **Calibration-Gated Retrain Trigger With Hysteresis And Zero-Width Guard**
  - Positive Tests (expected to PASS):
    - `scripts/extend_oos_and_retune.py` (or equivalent extension point) computes a rolling 30-day band-overshoot ratio = mean(pred_width / actual_width) using only past actuals (same maturity-lag policy as AC-8). When the realized `actual_width` for any day in the window is below a floor of `0.10%` (or zero), that day is excluded from the ratio mean rather than producing a divide-by-zero or extreme outlier; the exclusion count is logged.
    - If the smoothed ratio > 2.5 for ≥5 consecutive trading days, queue a retrain into `data/models/retrain_queue.jsonl`; if > 4.0, queue immediately.
    - After a retrain runs, a 7-trading-day cooldown is enforced (no retrain re-trigger even if ratio remains high).
    - The retrain log captures `triggered_at`, `ratio_value`, `consecutive_days`, `cooldown_until`, `zero_width_excluded_count`, and `outcome` (queued / immediate / suppressed_cooldown).
  - Negative Tests (expected to FAIL):
    - A single-day ratio breach triggering retrain (no 5-day hysteresis) fails.
    - A retrain triggering during cooldown fails.
    - A ratio computation that lets a zero `actual_width` day produce `inf` or absorb into the mean fails the zero-width-guard assertion.

- AC-10: **Test Harness And Per-Patch Validation Are Reproducible**
  - Positive Tests (expected to PASS):
    - `tests/test_expiry_intrinsic.py` and `tests/test_per_asset_cfg.py` exist; `pytest tests/` passes from a clean conda `gold` environment after `pip install -r requirements.txt` (which must add `pytest` if missing).
    - Each tagged patch's diff archive in `data/backtest_history/<tag>/` includes a `VALIDATION.md` describing the grid/replay run, the input data range, and the produced metrics.
    - A `make validate-patch TAG=<tag>` target (or equivalent shell script) reproduces the validation output bit-identically given the same input data.
  - Negative Tests (expected to FAIL):
    - A patch tagged but lacking `VALIDATION.md` fails the audit.
    - Tests requiring network access or live yfinance/Moomoo at test time fail the offline-reproducibility assertion (must use cached fixtures).

- AC-11: **Plan-Specific Workflow Markers Are Not Leaked Into Code**
  - Positive Tests (expected to PASS):
    - Grep across modified source files shows no occurrences of literal "AC-", "Milestone:", "Step N:", "Phase A:" workflow markers introduced by these patches.
    - Code identifiers use domain language (e.g., `select_gld_sync_strategy`, `apply_rolling_conformal_scaler`) rather than plan-progress language.
  - Negative Tests (expected to FAIL):
    - A code comment like `# AC-3: per-asset cfg threading` fails review.
    - A class named `Phase G Conformal Calibrator` fails review.

- AC-12: **SELL PUT Realized P&L Sign Is Correct In Spot Fallback Path**
  - Positive Tests (expected to PASS):
    - `core/paper_positions.py:177` (or whatever line the SP fallback sign assignment lives at post-Phase A) sets `sign = +1` for SELL PUT in the spot-fallback branch, reflecting SP's positive delta.
    - A pytest case constructing an SP entry, advancing spot up by 1%, and computing realized_pnl_pct via the spot-fallback path yields a positive value (winning side) and a negative value when spot moves down by 1%.
    - A grep audit of consumers of `realized_pnl_pct` (in `app.py`, `notifier.py`, `compute_strategy_stats.py`, and any dashboard renderers) confirms no consumer interprets the SP sign in a direction inconsistent with `+1`.
  - Negative Tests (expected to FAIL):
    - A reverted `sign = -1` (the historical bug) produces a negative `realized_pnl_pct` for an SP entry on a bullish spot move and fails the directional test.
    - A consumer that flips the sign for SP (compensating for the old bug) is detected and flagged for review.

- AC-13: **`entry_spot` Schema Migration Preserves Backward Compatibility**
  - Positive Tests (expected to PASS):
    - `core/positions_ledger.py` writes both `entry_spot` (legacy alias, deprecated) and `underlying_entry_price` (new, semantically correct) for one release; the new field is populated from `price_strategy_at(...).get("underlying_entry_price")` (added to `price_strategy_at`'s return dict).
    - A migration note in `data/positions_ledger_meta.json` records `entry_spot_alias_until_version` for downstream consumers to plan their cutover.
    - All current readers of `entry_spot` in the codebase (grep audit) are updated to read `underlying_entry_price` with a fallback to `entry_spot` for old rows.
  - Negative Tests (expected to FAIL):
    - A ledger row written post-patch lacking the new `underlying_entry_price` field fails the schema-version assertion.
    - A reader that uses `entry_spot` without the new-name fallback chain fails the migration audit.

- AC-14: **Dashboard `run_backtest` Deprecation Preserves Intraday Exit Semantics; Signal-Column Differences Are Audited Not Asserted**
  - Positive Tests (expected to PASS):
    - The new path routes signal generation through `core/signals_v2.py:generate_daily_signals` (canonical) and the intraday exit replay through a dedicated module that preserves the legacy `app.py:run_backtest` StopLoss/Pullback/ACTIVE event semantics.
    - A parity assertion harness (pytest fixture under `tests/test_dashboard_parity.py`) runs both paths against ≥ 60 historical trading days and asserts:
      - Intraday exit-event semantics (StopLoss/Pullback/ACTIVE event counts) match within ±1 — this is an exact-parity contract because exit semantics are the part being preserved.
      - Signal columns (`buy_signal / buy_type / signal_tier / exit_signal`) may differ; each difference is explained by a documented row in `data/backtest_history/v3.7.243_dashboard/signal_drift_attribution.csv` identifying the canonical-filter change responsible (e.g., "IV three-tier filter applied", "MA filter applied", "sp_score gated"). If any difference is unexplained, the harness fails.
    - The legacy `run_backtest()` remains callable but emits a `DeprecationWarning` for one minor release; the deprecation is removed in v3.8.
  - Negative Tests (expected to FAIL):
    - A new-path replay that drops the `StopLoss` event tagging or whose intraday-exit event count diverges by more than ±1 fails.
    - A signal-column difference without a matching attribution row in `signal_drift_attribution.csv` fails the audit.
    - Removing the legacy `run_backtest()` before v3.8 fails the deprecation-window policy.

- AC-15: **Layer 1 `max_move_{h}d` Window Has No Off-By-One**
  - Positive Tests (expected to PASS):
    - The Layer 1 `max_move_{h}d` calculation in `scripts/backtest/framework.py` uses an explicit indexed-window helper rather than reverse-rolling-and-shifting; for signal index `i`, the window is `high[entry_i:exit_i+1].max()` where `entry_i = i + 1` and `exit_i = entry_i + h - 1`, computed across the date axis.
    - A pytest fixture with a known synthetic OHLC series confirms `max_move_5d` for index `i` equals `max(High[i+1..i+5])`/`Close[i] - 1`.
  - Negative Tests (expected to FAIL):
    - An implementation using `high.rolling(h).max().shift(-(h+1))` (the original reverse pattern) fails the index-equivalence assertion on the synthetic fixture.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The implementation delivers all 17 tagged patches across Phases A through G, with per-tag validation evidence archived under `data/backtest_history/<tag>/`. Per-asset exit configuration lands as a new explicit registry (`get_option_exit_config(asset, strategy)`) rather than overloading `AssetConfig`. `force_close_at_expiry` extends to STRADDLE and SHORT_VOL with strategy-specific intrinsic formulas. Dashboard's `run_backtest()` is rewritten to route through `generate_daily_signals()` while preserving its intraday StopLoss/Pullback/ACTIVE semantics (deprecation wrapper kept for one minor release). A pytest suite covers all critical paths (BC/SP/STRADDLE/SHORT_VOL exits × kline {missing, expiry-day, post-expiry} × asset {GLD, SLV}, plus calibration scaler look-ahead invariants, plus retrain trigger hysteresis). Calibration is shadow-tested across Layer 1 trailing 10y/5y/3y/1y windows, gate documented in `gate_report.md`, and live-cutover flag flipped if the gate passes. The freshness state machine surfaces per-source FRESH/STALE/FROZEN states in dashboard and ledger metadata. The `entry_spot` rename ships with a schema migration script and one-release backward-compat alias. The 3 March 2026 BC loss cluster is back-tested under the new selector and the would-be P&L improvement is documented.

### Lower Bound (Minimum Acceptable Scope)

The implementation delivers AC-1, AC-2, AC-3, AC-4, AC-5 (selector + shadow log; `live_cutover` may stay `False`), AC-6 (binary kline freshness gate; full FRESH/STALE/FROZEN tiered state machine optional), AC-7 (full per-leg DTE sample restriction + disposition reporting — required, not optional, since AC-7 mandates this), AC-8 (scaler exists with maturity lag, calibrated columns written to parquet in shadow mode; `live_cutover` flag stays `False` if the Layer 1 grid gate fails or coverage gate fails), AC-9 (retrain trigger with hysteresis + cooldown), AC-10 (pytest harness + `tests/test_expiry_intrinsic.py` + `tests/test_per_asset_cfg.py` + `tests/test_calibration.py`), AC-11 (grep audit), AC-12 (SP sign), AC-13 (entry_spot migration), AC-14 (Dashboard parity assertion), and AC-15 (max_move off-by-one). The deferable items at Lower Bound are: full tiered FRESH/STALE/FROZEN dashboard UX (binary gate suffices), Dashboard `run_backtest` legacy removal (deprecation warning suffices for one release), and per-regime conformal alpha sub-routine if any regime has `n < 20` (global scaler fallback covers it).

### Allowed Choices

- Can use: per-tag git tags `v3.7.233`..`v3.7.249` matching the existing v3.7.* cadence OR humanize-internal patch labels — user decides via DEC-3.
- Can use: extension of `core/strategy_config.py:AssetConfig` to include exit-config fields, OR introduction of a new `core/strategy_config.py:get_option_exit_config(asset, strategy)` mapping that returns `BCConfig|SPConfig|StraddleConfig|ShortVolConfig` — user decides via DEC-1.
- Can use: per-asset overrides defined inline in Python, OR in `config/strategy.yaml` if a YAML schema is added later (deferred to v3.8).
- Can use: `pytest` (newly added to `requirements.txt`) as the test harness; cached parquet fixtures in `tests/fixtures/` for offline reproducibility.
- Can use: `rolling 60-day, quantile 0.85` conformal scaler as the default, with parameters exposed for grid search if the Layer 1 gate fails on the default.
- Cannot use: silent fallback to default `BCConfig()`/`SPConfig()` when asset is not supplied (must raise or warn explicitly).
- Cannot use: calibrated bands as `build_band()` default until the Layer 1 grid gate documented in `gate_report.md` passes.
- Cannot use: a single git commit that bundles multiple Phase changes without per-tag boundaries (violates the surgical-patch constraint).
- Cannot use: `--no-verify` or hook bypass on commits introduced by these patches.

## Feasibility Hints and Suggestions

### Conceptual Approach

A reference implementation sequence (one possible path; alternatives allowed per Allowed Choices):

1. **Phase A first** because parameter flips are independent and reversible (regime min_hold_days, SP sign, entry_spot rename), giving the team a confidence-building first landing. The `entry_spot` rename adds an explicit `underlying_entry_price` field populated from `price_strategy_at()` while keeping `entry_spot` as a one-release alias.

2. **Phase B** adds `max_fallback_days` and the freshness gate. The gate consults a small `core/data_freshness.py` module that inspects `_KLINE_DB_PATH` mtime and `max(date)`, returns `FRESH/STALE/FROZEN`. The ledger daemon consults the gate before invoking the option-entry block; futures path bypasses (sources from Binance directly). Blocked entries log `status="PENDING_KLINE"` with the original signal_date so retries do not duplicate.

3. **Phase C** introduces the per-asset exit config resolver. The implementation adds `core/strategy_config.py:get_option_exit_config(asset: str, strategy: str)` returning the appropriate dataclass; `core/paper_positions.py:simulate_option_exit` resolves the config from `asset + strategy` and passes it down. Existing call sites in `scripts/exit_grid_v2.py` and `scripts/options_per_tier_validate.py` are updated; legacy `cfg=None` arguments emit a deprecation warning. Phase C also wires `force_close_at_expiry` into STRADDLE and SHORT_VOL with strategy-specific intrinsic formulas: long-vol/STRADDLE = `sum(qty × intrinsic)` for both legs; SHORT_VOL = IC payoff at expiry with proper wing-width-based max_risk.

4. **Phase D** rewrites `framework.py:120` `max_move` with an explicit indexed window helper and updates Layer 2 sample-restriction to use per-leg DTE.

5. **Phase E** lands the cross-asset selector as a pure decision-returning function and a separate caller-side shadow-log writer. `app.py:run_backtest` deprecation is the heaviest piece; one approach is to add a thin wrapper that calls both the legacy path and the unified `generate_daily_signals` + intraday-exit replay; the parity contract is exact on intraday exit events (StopLoss/Pullback/ACTIVE counts within ±1), and signal-column drift (if any) is recorded in `signal_drift_attribution.csv` with the canonical-filter change that produced it — that audit row is what justifies the drift rather than blocking on it.

6. **Phase F** establishes the pytest harness. Cached fixtures live under `tests/fixtures/` (small parquet/csv snippets). `pytest tests/` runs offline.

7. **Phase G** runs the calibration audit (`scripts/eval/model_calibration_audit.py`), writes calibrated columns to parquet, runs the Layer 1 grid gate; if the gate passes, sets `live_cutover=True`. Retrain trigger appends to `data/models/retrain_queue.jsonl`.

### Relevant References

- `core/paper_positions.py` — exit simulation dispatcher (line ~689), entry pricing fallback (line ~366), SP sign bug (line ~177).
- `core/strategies/buy_call.py`, `sell_put.py`, `straddle.py`, `short_vol.py`, `options_exit.py` — per-strategy simulate_*_position functions and the existing `force_close_at_expiry` helper (already covering BC/SP).
- `core/positions_ledger.py` — ledger writer; `entry_spot` field at line ~87.
- `core/ledger_daemon.py` — 300s daemon loop; integration point for freshness gate.
- `core/signals_v2.py` — canonical signal pipeline (`generate_daily_signals`).
- `core/cross_asset_signal.py` — current cross-asset constants to lift into a pure selector.
- `core/regime.py` — `RegimeClassifier`, default `min_hold_days=20` (line ~24).
- `core/data.py:extend_oos_predictions` (line ~442) — boundary where calibration scaler attaches.
- `core/strategy_config.py` — `AssetConfig` registry; site for new `get_option_exit_config`.
- `scripts/backtest/framework.py` — `max_move_{h}d` off-by-one (line ~120); regime call with `min_hold_days=1` (line ~70).
- `scripts/backtest/layer1_signal/directional/run_all.py`, `layer2_strategy/{futures,directional_options,vol_options}/run_all.py` — backtest harnesses to gate calibration cutover and to add sample-disposition reporting.
- `scripts/build_positions_ledger.py` — historical ledger rebuild; ~12 RegimeClassifier call sites in the production tree (audit required).
- `src/models/train_dl_range.py:build_targets` (line ~72) — authoritative 5-day forward label definition that `actual_upper_pct/actual_lower_pct` columns derive from.
- `data/models/dl_range_v2_oos.parquet`, `dl_range_slv_oos.parquet` — GLD and SLV OOS history for audit + scaler input.
- `data/positions_ledger.json`, `data/positions_ledger_meta.json` — current ledger (28 rows after v3.7.232 expiry closes) and watermark.
- `CLAUDE.md` (project root) — "改 cfg → grid 验证 → archive version" workflow.

## Dependencies and Sequence

### Milestones

1. **Milestone 1 — Correctness Floor (Phases A–C, F partial):**
   - Phase A: Parameter flips and sign corrections (`regime.min_hold_days=1`, SP fallback sign, entry_spot rename with alias). Independent; revertible.
   - Phase B: Defensive ingestion guards (`max_fallback_days`, freshness gate with PENDING_KLINE state). Depends on Phase A only for the regime classifier consistency.
   - Phase C: Per-asset exit configuration registry; `force_close_at_expiry` extension to STRADDLE/SHORT_VOL with strategy-specific intrinsic. Depends on the DEC-1 user decision on registry shape.
   - Phase F partial: pytest harness scaffold + tests for AC-3 / AC-4 / AC-2. Depends on Phases A–C completing because tests exercise their behaviors.

2. **Milestone 2 — Derived-Metric Repair + Cross-Asset IV-Awareness (Phases D, E1):**
   - Phase D: `max_move` off-by-one; Layer 2 sample-disposition reporting and leg-level DTE filter.
   - Phase E1: `select_gld_sync_strategy` pure function + shadow log + ≥14 day shadow period.
   - Depends on Milestone 1's exit infrastructure (E1 cross-asset entries flow through the per-asset cfg resolver landed in Phase C).

3. **Milestone 3 — Calibration Audit + Conformal Scaler Shadow + Retrain Trigger (Phase G):**
   - G1: Calibration audit script (`scripts/eval/model_calibration_audit.py`) reads parquet `actual_*_pct` columns as the authoritative realized labels.
   - G2: Rolling-residual conformal scaler with horizon-aware maturity-lag discipline (`label_end_date < calibration_as_of_date`); calibrated columns appended to parquet under `shadow_logging=True`; `build_band()` reads calibrated columns only when `live_cutover=True`.
   - G3: Calibration-gated retrain trigger with 5-day hysteresis + 7-day cooldown.
   - G4: Per-regime alpha (Bull/Bear/Sideways) with `n≥20` sample minimum; falls back to global scaler when under-sampled.
   - G5: Layer 1 grid gate (scoreB + coverage compound test, see DEC-5) documented in `gate_report.md`; `live_cutover` flip via startup preflight only if gate passes.
   - Depends on Milestone 1's regime correction (Phase A) so per-regime alpha uses leak-free regime labels.

4. **Milestone 4 — Dashboard Parity + Acceptance Test Closure (Phase E2, F closure):**
   - Phase E2: Dashboard `run_backtest` deprecation with intraday semantic parity preserved behind a feature flag.
   - Phase F closure: Full pytest coverage including calibration look-ahead invariants and retrain hysteresis.
   - Depends on Milestone 2 (cross-asset selector available) and Milestone 3 (calibrated columns reachable from dashboard).

### Component Dependencies

- Per-asset exit cfg resolver (Phase C) depends on DEC-1 user decision on registry shape (extend AssetConfig vs new `get_option_exit_config`).
- Freshness gate (Phase B) depends on DEC-2 user decision on scope (kline-stale blocks cross-asset option sync? blocks futures?).
- Tagging style (Phase A onward) depends on DEC-3 (git tags vs humanize labels).
- Dashboard `run_backtest` rewrite (Phase E2) depends on DEC-4 (compat wrapper retention vs one-shot replacement).
- Calibration `live_cutover` flip (G5) depends on Layer 1 grid gate result; cutover is gated, not unconditional.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task-a1 | Flip `core/regime.py` default `min_hold_days` to 1; audit and patch all production call sites to pass `min_hold_days=1` explicitly; produce allow-list for research-only paths using `=20` | AC-2 | coding | — |
| task-a2 | Fix `core/paper_positions.py:177` SP fallback sign from -1 to +1; audit downstream consumers of `realized_pnl_pct` sign | AC-12, AC-11 | coding | — |
| task-a3 | Rename `core/positions_ledger.py:entry_spot` field semantics: add `underlying_entry_price` populated from `price_strategy_at()` result; keep `entry_spot` as one-release alias; write schema migration note into `data/positions_ledger_meta.json` | AC-13, AC-11 | coding | — |
| task-b1 | Add `max_fallback_days=7` parameter to `pick_liquid_monthly_option`; return `None` with `source="PENDING_KLINE"` when exceeded; ensure `price_strategy_at` distinguishes `PENDING_KLINE` from `NO_CONTRACT` | AC-6 | coding | — |
| task-b2 | Add `core/data_freshness.py` with `kline_db_state(today, max_age_days)` returning FRESH/STALE/FROZEN; wire into `core/ledger_daemon.py` so FROZEN blocks new option entries; dashboard sidebar renders state | AC-6 | coding | task-b1 |
| task-c1 | Add `core/strategy_config.py:get_option_exit_config(asset, strategy)` returning BC/SP/Straddle/ShortVol config dataclass (or extend `AssetConfig`, per DEC-1); update `core/paper_positions.py:simulate_option_exit` to accept and propagate `asset`; deprecate `cfg=None` silent path | AC-3 | coding | DEC-1 |
| task-c2 | Extend `force_close_at_expiry` to STRADDLE (strategy_kind="long_vol", sum-of-leg intrinsic) and SHORT_VOL (strategy_kind="iron_condor", IC-specific max_risk) | AC-4 | coding | task-c1 |
| task-d1 | Rewrite `scripts/backtest/framework.py:max_move_{h}d` using explicit `entry_i`/`exit_i` indexing helper; remove reverse-rolling pattern; add pytest with synthetic OHLC fixture | AC-15 | coding | — |
| task-d2 | Update Layer 2 backtest harnesses (`scripts/backtest/layer2_strategy/*/run_all.py`) to emit `n_signal/n_entered/n_closed/n_open/n_skipped_*` columns; sample-restriction uses per-leg DTE | AC-7 | coding | task-d1 |
| task-e1 | Add `core/cross_asset_signal.py:select_gld_sync_strategy(signal_date, gld_signal_row, gvz_value, gvz_asof_date)` pure function returning `{strategy, reason, gvz_status}`; separate caller-side `write_shadow_record(...)` invoked from `scripts/build_positions_ledger.py`; ≥14-day shadow accumulation manifest before live cutover | AC-5 | coding | task-c1 |
| task-e2 | Replay 3 March 2026 SLV-S triggers through the new selector; document the would-be P&L delta vs historical BC outcomes | AC-5 | analyze | task-e1 |
| task-e3 | Dashboard `app.py:run_backtest` deprecation: wrap legacy path; route through `generate_daily_signals()`; assert intraday StopLoss/Pullback parity behind feature flag; emit DeprecationWarning for one release; remove in v3.8 | AC-14 | coding | task-e1 |
| task-f1 | Add `pytest` to `requirements.txt` (if missing); scaffold `tests/` with `conftest.py` and shared fixtures under `tests/fixtures/`; cached parquet snippets for offline runs | AC-10 | coding | — |
| task-f2 | Write `tests/test_expiry_intrinsic.py` covering 4 strategies × {today<exp, today=exp with close known, today=exp without close (AWAITING_EXPIRY_CLOSE), today>exp with kline missing} = 16 scenarios; include one SHORT_VOL asymmetric-wings fixture for the IC max_risk assertion | AC-4 | coding | task-c2, task-f1 |
| task-f3 | Write `tests/test_per_asset_cfg.py` asserting GLD vs SLV exit thresholds differ; assert legacy `cfg=None` path raises/warns | AC-3 | coding | task-c1, task-f1 |
| task-f4 | Write `tests/test_calibration.py` covering scaler look-ahead invariants, retrain trigger hysteresis, per-regime fallback to global scaler | AC-8, AC-9 | coding | task-g3, task-f1 |
| task-g1 | Implement `scripts/eval/model_calibration_audit.py` reading `data/models/dl_range_*_oos.parquet`, using parquet's `actual_upper_pct/actual_lower_pct` columns (5-day forward labels); per-month × per-regime coverage + width-ratio report; archives output | AC-1 | coding | task-a1 |
| task-g2 | Re-run the audit on the corrected 2025-12-01..2026-05-13 GLD window; document the 1.95×/1.66× width ratio + 54.9% coverage (updating the draft's incorrect 5-6× figure) into the audit report and into `Claude-Codex Deliberation` | AC-1 | analyze | task-g1 |
| task-g3 | Implement `core/calibration.py:apply_rolling_conformal_scaler(dates, pred_upper, pred_lower, actual_upper, actual_lower, horizon=5, window=60, target_coverage=0.80)` with horizon-aware maturity-lag discipline (only residuals with `label_end_date < calibration_as_of_date` enter the pool); integrate into `core/data.py:extend_oos_predictions` to append calibrated columns when `shadow_logging=True`; `build_band()` reads via `calibration.live_cutover` flag (default `False`); cutover preflight check at startup reads `gate_report.md`'s `gate_passed: true` field | AC-8 | coding | task-g1 |
| task-g4 | Implement calibration-gated retrain trigger in `scripts/extend_oos_and_retune.py` (or new module) with 5-day hysteresis + 7-day cooldown + `data/models/retrain_queue.jsonl` audit trail | AC-9 | coding | task-g3 |
| task-g5 | Implement per-regime conformal alpha (Bull/Bear/Sideways) with `n≥20` minimum + fallback-to-global; log fallback events into `retrain_queue.jsonl` | AC-8 | coding | task-g3, task-a1 |
| task-g6 | Run Layer 1 directional grid `scripts/backtest/layer1_signal/directional/run_all.py` with `--use-calibrated` flag across 10y/5y/3y/1y windows; document calibrated vs raw scoreB; produce `data/backtest_history/v3.7.246_calibration/gate_report.md` deciding `live_cutover` flag value | AC-8 | analyze | task-g3, task-g5 |
| task-h1 | Set up `make validate-patch TAG=<tag>` (or equivalent shell script) that re-runs each tag's validation script against archived fixtures; normalize timestamp and rng-seed fields in outputs before bit-identical comparison so non-deterministic metadata does not produce false fails | AC-10 | coding | task-f1 |
| task-h2 | Final audit grep for plan-progress markers (`AC-`, `Milestone:`, `Phase A:`, `Step N:`) in all modified source files; fail if any present | AC-11 | analyze | All coding tasks |

## Claude-Codex Deliberation

### Agreements

- The 5 severe + 6 medium Codex audit findings are well-localized and surgical patches with per-tag validation is the right cadence.
- The Phase G calibration concern is real; the OOS predicted band is wider than realized and coverage is below training target, but the magnitude is ~2× and 54.9% coverage (5-day forward labels), not the draft's 5-6× and 87.6% (which conflated 5-day forward labels with single-day overnight returns).
- The 3 March 2026 GLD BUY CALL 5/5 cluster is dominated by the cross-asset rule + IV regime interaction, with calibration as a contributing but secondary factor.
- `force_close_at_expiry` must extend to STRADDLE and SHORT_VOL with strategy-specific intrinsic formulas (not blind credit_spread reuse for IC).
- Conformal recalibration must use horizon-aware maturity-lag past-only residuals (`label_end_date < calibration_as_of_date` for horizon=5 forward labels); Layer 1 grid gate must precede live cutover.
- pytest is missing from `requirements.txt` and the test infrastructure must be scaffolded before per-patch validation can be reproducible.

### Resolved Disagreements

- **Calibration empirical numbers** (Round 1): Claude's draft cited 5-6× width and 87.6% coverage (derived from single-day overnight returns). Codex independently re-derived from parquet's `actual_*_pct` columns (5-day forward labels per `src/models/train_dl_range.py:build_targets`) and got 1.95×/1.66× width and 54.9% coverage. Resolution: Codex is correct; the plan adopts Codex's label definition, AC-1 enforces this label definition, task-g2 records the correction in the audit output. Original draft numbers are marked NON-NORMATIVE in the appendix.
- **Per-asset cfg surface** (Round 1): Claude's draft assumed `get_config(asset)` suffices. Codex flagged that `AssetConfig` holds signal thresholds, not exit thresholds (which live in `BCConfig`/`SPConfig` etc.). Resolution: introduce explicit `get_option_exit_config(asset, strategy)` registry, recorded in DEC-1 for user choice between extending AssetConfig vs new registry.
- **STRADDLE/SHORT_VOL force_close** (Round 1): Claude's draft suggested mirroring BC/SP wiring. Codex flagged IC max_risk needs wing-width logic. Resolution: task-c2 specifies strategy-specific intrinsic formulas (long_vol = sum-of-leg, IC = wing-width-based max_risk).
- **Layer 2 sample restriction** (Round 1): Claude's draft suggested `signal_date ≤ kline_max - max_dte - hold_buffer`. Codex flagged per-leg DTE matters. Resolution: AC-7 + task-d2 specifies per-leg DTE, and Lower Bound was tightened (Round 2) to require it.
- **Dashboard `run_backtest` rewrite scope** (Round 1): Claude's draft suggested thin replay. Codex flagged intraday StopLoss/Pullback/ACTIVE semantics would be lost. Resolution: AC-14 + task-e3 keep the legacy path behind a feature flag and add a parity assertion harness.
- **Freshness gate scope** (Round 1): Codex asked whether futures and cross-asset option sync should also be gated. Resolution: gate scopes to new option entries only (kline_db-dependent); futures (Binance live) and force_close intrinsic are exempt. Cross-asset option sync is treated as an option entry, so it is gated.
- **Calibration scaler temporal discipline** (Round 2): Round 1 said `shift(1)`. Codex correctly pointed out that 5-day forward labels do not mature until t+5; a shift(1) policy still leaks. Resolution: AC-8 + task-g3 specifies horizon-aware maturity lag (`label_end_date < calibration_as_of_date`).
- **Calibration goal — narrowing vs coverage repair** (Round 2): Round 1 implicitly assumed bands should narrow. Codex pointed out current state is low coverage with moderately wide bands; narrowing would worsen coverage. Resolution: AC-8 + Goal Description state coverage repair is the structural objective; narrowing is not a test target. DEC-5 captures the compound-gate-vs-decoupled-metrics decision.
- **SHORT_VOL IC max_risk asymmetric wings** (Round 2): Round 1 used `min(...)` which is wrong on asymmetric wings. Resolution: AC-4 + task-c2 use `max(call_wing_width, put_wing_width) - credit_received` with optional symmetric assertion. DEC-6 captures the assert-symmetric-vs-always-handle-asymmetric decision.
- **Lower Bound vs AC-7 contradiction** (Round 2): Round 1 lower bound allowed AC-7 to ship report-only. Codex flagged this conflicts with AC-7's "must use per-leg DTE" requirement. Resolution: Lower Bound rewritten to require AC-7 fully.
- **Missing ACs for SP sign / entry_spot / dashboard parity / OI / risk / yfinance** (Round 2): Round 1 omitted explicit ACs for several behaviors and Goal claimed broader scope than ACs covered. Resolution: AC-12 (SP sign), AC-13 (entry_spot migration), AC-14 (dashboard parity) added; Goal narrowed to enumerated ACs; OI / risk controls / yfinance bid-ask / exposure caps explicitly listed as out-of-scope (DEC-7).
- **AC-4 expiry-day kline-missing edge case** (Round 2): Round 1 was ambiguous about `today == expiry_dt`. Resolution: AC-4 enumerates spot-close-known vs spot-close-unknown branches, with explicit `state="AWAITING_EXPIRY_CLOSE"` for the latter.
- **AC-5 GVZ missing/stale handling** (Round 2): Round 1 was silent. Resolution: AC-5 explicit truth table — GVZ missing or >2 trading days stale ⇒ return `"BUY CALL"` with `reason="GVZ_UNAVAILABLE"` log entry.
- **AC-5 / AC-8 flag separation** (Round 2): Round 1 conflated `shadow_logging` and `live_cutover`. Resolution: two independent flags in both AC-5 and AC-8; default `shadow_logging=True`, `live_cutover=False`.
- **`core/calibration.py` runtime assertion vs preflight check** (Round 2): Round 1 had the cutover gate as a runtime `assert`. Codex flagged this couples production imports to filesystem state. Resolution: gate is a config preflight at app/daemon startup, not a runtime assertion.
- **`make validate-patch` bit-identical reproducibility** (Round 2): Round 1 was strict on bit-identical. Codex flagged timestamp/seed fields would cause false fails. Resolution: task-h1 normalizes timestamp/rng-seed fields before comparison.
- **AC-5 selector purity vs filesystem writes** (Round 3): Round 2 declared the selector pure but also asked it to write a JSONL log. Codex flagged the contradiction. Resolution: selector returns a structured decision dict; caller (`scripts/build_positions_ledger.py`) owns the shadow log write.
- **AC-5 staleness reference frame** (Round 3): Round 2 used implicit "today" for GVZ staleness, which mixes wall-clock with replay. Resolution: AC-5 takes `gvz_asof_date` and computes staleness relative to `signal_date`.
- **AC-8 temporal boundary phrasing** (Round 3): Round 2 mixed `≤ t − horizon` and `label_end_date < calibration_as_of_date`. Codex flagged potential off-by-one. Resolution: unified to `label_end_date(s) = s + horizon_trading_days` with `label_end_date < calibration_as_of_date` as the sole eligibility predicate; removed every `shift(1)` reference from normative sections.
- **task-f2 scenario count** (Round 3): Round 2 said 12 scenarios; AC-4 said 16. Resolution: task-f2 updated to 16 scenarios with the four expiry-day states enumerated.
- **AC-14 Dashboard signal-column parity** (Round 3): Round 2 required exact match for signal columns, which would block the very fix the deprecation enables. Resolution: AC-14 asserts exact parity for intraday exit semantics (StopLoss/Pullback/ACTIVE within ±1) but treats signal-column drift as auditable via `signal_drift_attribution.csv`.
- **`max_move_{h}d` orphan task** (Round 3): Round 2 had task-d1 without a matching AC. Resolution: AC-15 added explicitly for the off-by-one repair.

### Unresolved (Carry-Over to User Decision)

- DEC-1 through DEC-7 (see below).

### Convergence Status

- Final Status: `converged` — material disagreements resolved across 3 convergence rounds (Phase 5 Round 1 → Round 2 → Round 3); all 7 DEC items resolved during Phase 6 user dialogue. Codex Round 3 review treated all REQUIRED_CHANGES as addressed; user confirmed DEC-1/3/7 with recommended options and confirmed 80% coverage as a trend target (DEC-5 implication). DEC-2/4/6 defaulted to Claude positions which both reviewers regarded as safe.
- Rounds executed: 3 (Phase 5 Round 1 + Round 2 + Round 3).
- Quantitative thresholds confirmed as trend targets (not hard requirements): 80% target_coverage, 3-day kline freshness threshold, 7-day max_fallback_days, 60-day rolling residual window, 2.5/4.0 retrain ratio thresholds, 5-day hysteresis, 7-day cooldown, n≥20 per-regime sample minimum, 14 calendar days shadow accumulation. The intraday exit event ±1 parity in AC-14 is a hard requirement because exit semantics are the part being structurally preserved.

## Pending User Decisions

> All seven DEC items have been resolved during Phase 6 user dialogue. Their resolutions are recorded below for traceability.

- DEC-1: **Per-asset exit configuration surface — extend `AssetConfig` or introduce a new `get_option_exit_config(asset, strategy)` registry?**
  - Claude Position: Introduce a new `get_option_exit_config` resolver in `core/strategy_config.py`. Rationale: `AssetConfig` is semantically about signal thresholds; mixing exit thresholds creates a wider dataclass and conflates two unrelated concerns. A separate resolver keeps responsibilities clean and is the smallest change that satisfies AC-3.
  - Codex Position: Either acceptable provided the resolver lives in exactly one place and `simulate_*_position` does not duplicate lookup logic. Codex notes that extending `AssetConfig` could later collide with the v3.9 declarative policy layer (Alt-5 in the idea draft); a separate resolver is friendlier to that future consolidation.
  - Tradeoff Summary: Extending `AssetConfig` minimizes new files but conflates concerns. Separate resolver adds one function/import surface but cleanly separates signal vs exit configuration.
  - Decision Status: RESOLVED — User chose the new `get_option_exit_config(asset, strategy)` resolver. task-c1 implements it in `core/strategy_config.py`.

- DEC-2: **Freshness gate scope — when `kline_db` is stale, what is allowed to continue?**
  - Claude Position: Block new option entries (including cross-asset sync option entries) when `kline_db max_date < today - 3 trading days`; permit futures entries (Binance live data); permit MTM and `force_close_at_expiry` on existing positions.
  - Codex Position: Same as Claude's, but additionally proposes a finer-grained tiered gate: STALE (2-3 days) emits warning only; FROZEN (>3 days) blocks new entries; this matches Alt-3 in the idea draft.
  - Tradeoff Summary: Binary gate is simpler to reason about; tiered gate matches the freshness state machine alternative but adds operational complexity in the first release.
  - Decision Status: RESOLVED — Default to Claude's binary gate (3 trading days threshold blocks option entries; futures + MTM + expiry-intrinsic unaffected). Tiered FRESH/STALE/FROZEN state machine is upper-bound, may land alongside if low cost; tracked as v3.8 enhancement if not in scope.

- DEC-3: **Patch labeling — real git tags `v3.7.233`..`v3.7.249` or humanize-internal patch labels?**
  - Claude Position: Real git tags continuing the v3.7.* cadence. Rationale: the existing v3.7.* cadence is the project's authoritative changelog and grid-validation archive naming is keyed by tag.
  - Codex Position: Humanize patch labels are equally workable provided the archives under `data/backtest_history/<tag>/` use the same tag string. Real git tags require pushing 17 tags to the remote, which the user may or may not want.
  - Tradeoff Summary: Git tags give external visibility and align with current archive naming; humanize labels are lighter-weight and avoid 17 push operations.
  - Decision Status: RESOLVED — User chose real git tags `v3.7.233`..`v3.7.249`. Note: per `CLAUDE.md` git safety protocol, pushing tags to remote requires explicit per-tag authorization at push time; local tagging is unrestricted.

- DEC-4: **Dashboard `run_backtest` rewrite — keep deprecation wrapper for one release, or one-shot replace?**
  - Claude Position: Keep a one-release deprecation wrapper. The wrapper internally runs both paths and asserts intraday exit semantics (StopLoss/Pullback/ACTIVE event counts within ±1); any signal-column drift is recorded in `signal_drift_attribution.csv` and treated as expected (per AC-14) rather than blocking. Rationale: intraday exit semantics are non-trivial to re-implement and the parity assertion provides a regression net while still permitting the very signal-correctness improvements the unified pipeline brings.
  - Codex Position: Either acceptable, but a one-shot replace risks breaking dashboard pages that consume the legacy output schema; the wrapper-with-parity-assertion is the safer choice.
  - Tradeoff Summary: Wrapper-with-parity is safer; one-shot is cleaner but riskier.
  - Decision Status: RESOLVED — Default to Claude's wrapper-for-one-release approach (both reviewers agreed it is safer). task-e3 implements the wrapper with the AC-14 parity contract.

- DEC-5: **Calibration scaler primary metric — coverage repair, scoreB gate, or both?**
  - Claude Position: Both, expressed as a compound gate: calibrated coverage must move toward `target_coverage` AND calibrated Layer 1 scoreB must be ≥ raw scoreB in ≥3 of 4 windows. Coverage repair is the structural objective; scoreB is the trading-utility safety net.
  - Codex Position: Decouple the two: report coverage calibration and trading scoreB as independent metrics; do not require "calibrated narrower" as a goal. Compound gate is acceptable provided neither metric individually drops materially.
  - Tradeoff Summary: Compound gate is rigorous but may block useful asymmetric calibrations (e.g., widening only the lower band) when scoreB happens to dip marginally. Independent reporting is more flexible but pushes the decision to ad-hoc judgment.
  - Decision Status: RESOLVED — Default to Claude's compound gate (AC-8 already states this). The 80% coverage is explicitly a trend target (per Phase 6 user confirmation), not a hard cutover blocker; Layer 1 scoreB ≥ raw in ≥3 of 4 windows is the binding criterion. Both metrics are reported independently in `gate_report.md` for transparency.

- DEC-6: **SHORT_VOL Iron Condor — assume symmetric wings (assert) or always handle asymmetric?**
  - Claude Position: Always handle asymmetric with `max(call_wing_width, put_wing_width) - credit`; add an optional `assert_symmetric_wings=False` parameter that downstream callers can flip on if the production strategy is known to always trade symmetric ICs.
  - Codex Position: Same default direction (handle asymmetric), but Codex flags that the current production `core/strategies/short_vol.py` may already assume symmetric wings; if so, an explicit symmetric assertion in the legs builder is a defensible alternative.
  - Tradeoff Summary: Asymmetric-aware is safer and matches the wider risk surface; symmetric-assert is simpler but adds an upstream guard at the legs-builder.
  - Decision Status: RESOLVED — Default to Claude's asymmetric-aware implementation with optional symmetric-wings assertion at the legs builder. task-c2 implements `max(call_wing_width, put_wing_width) - credit_received` as the primary path.

- DEC-7: **Out-of-scope items — confirm OI mainline / risk controls migration / yfinance bid-ask fallback / exposure caps stay deferred?**
  - Claude Position: Defer all four to a follow-on plan after this correctness floor lands. Rationale: they are optimization-tier items in the Codex audit, not blockers for the current loss-cluster diagnosis, and bundling them dilutes the surgical-patch cadence.
  - Codex Position: Accepts deferral provided the Goal Description states this explicitly (now done) and the four items are tracked in a follow-on backlog (not silently forgotten).
  - Tradeoff Summary: Deferring keeps this plan focused; bundling expands scope but unifies the surgical-patch effort.
  - Decision Status: RESOLVED — User confirmed deferral. The four items are tracked in `docs/BACKLOG_v3.8.md` (to be created as a follow-on artifact, not part of this plan's tasks).

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone:", "Step N:", "Phase A:" or similar workflow markers. These terms exist in the plan document only.
- Use domain-appropriate identifiers (e.g., `select_gld_sync_strategy`, `apply_rolling_conformal_scaler`, `get_option_exit_config`, `kline_db_state`) rather than plan-progress identifiers.
- Per the project's `CLAUDE.md` rule, no destructive operations (push --force, reset --hard, --no-verify) without explicit user authorization.
- Per `CLAUDE.md` "事实驱动" discipline: every cfg/code change requires a grid replay or pytest evidence before being tagged.

### Workflow Integration

- Each tag's validation evidence is archived under `data/backtest_history/<tag>/VALIDATION.md` plus any supporting CSVs/figures.
- `make validate-patch TAG=<tag>` (or equivalent shell script established by task-h1) reproduces the validation output offline.
- The pytest suite is run via `pytest tests/` from the `gold` conda environment; offline-only fixtures live in `tests/fixtures/`.
- Calibration cutover flip flag lives in `config/strategy.yaml` or `core/strategy_config.py` constant; flipping requires the `gate_report.md` audit to be present and to attest the Layer 1 grid gate passed.

### Risk Mitigations

- Asset-threading audit (per AC-3): after task-c1 lands, a grep audit of `simulate_*_position(` and `force_close_at_expiry(` call sites confirms every site passes asset or constructs an `OptionExitConfig` through the resolver.
- Calibration look-ahead audit (per AC-8): task-f4 includes a pytest case that constructs a synthetic OOS series with a known leak (day-t actuals in day-t residual window) and asserts the scaler rejects it.
- Cross-asset shadow log (per AC-5): the 14-day shadow window plus the explicit `gate_report.md` analog for the selector live-flip gate ensure no behavior change ships without empirical justification.
- Freshness gate fall-open prevention: the FROZEN state is the default when freshness metadata is missing (cold-start safety).
- `entry_spot` migration: one-release alias on the field name preserves backward compatibility with any external readers (notifier, dashboard, analysis scripts) that may still read the old name.



--- Original Design Draft Start ---

> **NON-NORMATIVE / SUPERSEDED**: The draft below was the initial exploration input. Several quantitative claims and minor design details have been REVISED by the Acceptance Criteria above. The normative source of truth for implementation is the structured plan; the draft is retained for traceability only.
>
> Key corrections from the structured plan that supersede this draft:
> - Calibration baseline: draft cites "5-6× over-wide bands, 87.6% in-band coverage" (computed from single-day overnight returns). Correct figures, derived from the parquet's `actual_*_pct` 5-day forward labels per `src/models/train_dl_range.py:build_targets`, are **width ratios 1.95× upper / 1.66× lower, coverage 54.9%** versus 80% training target. AC-1 enforces the correct label definition.
> - Calibration direction: draft assumes bands should be narrowed. Corrected goal per AC-8 is **coverage repair toward `target_coverage`**, which may widen, narrow, or shift bands depending on the residual distribution.
> - Conformal scaler temporal discipline: draft cites "shift(1) residual scaler". AC-8 requires **horizon-aware maturity lag (`label_end_date < calibration_as_of_date`)**, not generic `shift(1)`, because the 5-day forward label only matures at t+5.
> - SHORT_VOL Iron Condor max_risk: draft is silent. AC-4 specifies `max(call_wing_width, put_wing_width) - credit_received` for asymmetric wings; symmetric-wings shortcut is allowed only with an explicit assertion.
> - Layer 2 sample restriction: draft (v3.7.241) cites `signal_date <= kline_max - max_dte - hold_buffer`. AC-7 specifies **per-leg DTE** filtering plus full disposition reporting (`n_signal/n_entered/n_closed/n_open/n_skipped_*`).
> - Calibration column cutover: draft says `build_band()` "switches to read calibrated columns when present". AC-8 specifies two independent flags: `shadow_logging` (default True) and `live_cutover` (default False, requires `gate_report.md` `gate_passed: true`).
> - Dashboard `run_backtest` deprecation: draft (v3.7.243) cites "thin replay". AC-14 specifies parity assertion harness preserving intraday StopLoss/Pullback semantics + one-release DeprecationWarning + removal in v3.8.
> - Goal scope: draft says "5 severe + 6 medium + 4 optimization". Plan Goal narrows to **the ACs enumerated in this document**; the four optimization-tier items are explicitly out of scope (DEC-7).


# Surgical Per-Bug Patch Series For Mainline Hardening

## Original Idea

主题: 量化交易系统（GLD/SLV 期权 + 期货）架构鲁棒性与正确性优化

仓库根: /Users/yhdong/GoldDash (主代码, git repo) + /Users/yhdong/Gold (数据 + 训练, 非 git)

现状摘要:
- v3.7.232 期权交易系统, 主链: 信号生成 (signals_v2) → 策略选择 (BC/SP/STRADDLE/SHORT_VOL/FUTURES) → 仓位 ledger → 实时退出
- 多窗口回测两层: Layer1 信号验证 (10y/5y/3y/1y), Layer2 策略 P&L (1y/6m/3m)
- 刚加 expiry-intrinsic 强平兜底 (force_close_at_expiry), 仅 BC/SP 接入
- kline_db 滞后 17 天 (max=2026-05-06, today=2026-05-23), Moomoo 每日 100 额度
- Cross-asset: SLV-S → GLD BC 固定策略, 未 IV-aware

刚拿到的 Codex 架构审查结论 (5 严重 / 6 中等 / 4 优化):
严重:
1. paper_positions.py:689 simulate_option_exit 没传 asset 给 simulate_sp/bc/straddle/short_vol_position → per-asset cfg 全部失效, GLD/SLV 都用默认值
2. paper_positions.py:366 entry pricing 无 max_fallback_days, 用 stale kline 定价并冻结
3. v3.7.232 expiry 兜底只接 BC/SP, STRADDLE/SHORT_VOL 仍可能永远 OPEN
4. 生产 regime min_hold_days=20 (look-ahead), 回测=1, 不一致
5. Dashboard '真实策略回测' 走旧 run_backtest, 没应用 IV 三档/sp_score/MA filter/tier

中等:
- Cross-asset 应 IV-aware (GLD bp_low<=0.10 AND GVZ>=25 → SP)
- Layer2 回测只统计 closed → 幸存者偏差
- framework.py:120 max_move 反向 rolling 有 off-by-one
- OI 修正只在 Dashboard, 没进主链
- paper_positions.py:177 SP fallback sign 算反 (-1 应为 +1)
- positions_ledger.py:87 entry_spot 实际是期权 daily close 不是 ETF spot

优化:
- ledger daemon 加 kline 新鲜度硬闸
- 风控迁入主链 (exposure gate, max_open, 连亏熔断)
- v3.7.232 加回归测试
- yfinance fallback 用 bid/ask mid 优先

目标: 形成一份完整 idea draft, 系统性消除上述问题, 同时为后续 humanize 工作流 (explore-idea → gen-plan → refine-plan → start-rlcr-loop) 提供 grounded 起点。

约束:
- 一切要数据/事实说话, 不偷工减料, 不用模拟数据
- 数据必须当日更新, 不能用陈旧数据预测
- 优先消除正确性问题, 性能/优化次之
- 改动要可分阶段独立验证, 避免大爆炸式重构

## Primary Direction: Surgical Per-Bug Patch Series

### Rationale

Treat each Codex finding as a standalone, independently verifiable patch (one file, one test) sequenced by severity — maximizes safety, traceability, and rollback granularity at the cost of more PR overhead than architectural rewrites.

### Approach Summary

Implement a modular, version-tagged patch cadence addressing each of the 5 severe + 6 medium + 4 optimization Codex findings as **independent, single-file or tightly-scoped commits** sequenced by severity-then-dependency. Each patch:

1. **Targets ONE root cause** with minimal surface area (typically 1–3 files affected, ≤ 15 LOC change).
2. **Includes a focused validation script** (synthetic dataset or historical ledger replay) before merge.
3. **Receives its own v3.7.* tag** (continuing v3.7.233, v3.7.234, … following the existing cadence visible at commits v3.7.150, v3.7.156, v3.7.184, v3.7.232).
4. **Is independently reversible** via `git revert <tag>` with no cascading state corruption.

**Patch Series Ordering** (dependency-aware, severity-weighted):

- **Phase A — Pure parameter flips & sign corrections (zero structural risk):**
  - **v3.7.233**: `core/regime.py:24` default `min_hold_days=20` → `1`; audit `app.py:112` + `scripts/build_positions_ledger.py:92` to remove explicit `RegimeClassifier()` defaults that re-introduce look-ahead.
  - **v3.7.234**: `core/paper_positions.py:179` SELL PUT spot-fallback sign `-1` → `+1` (SP is bullish; current sign reverses realized_pnl semantics).
  - **v3.7.235**: `core/positions_ledger.py:87` rename `entry_spot` → `entry_option_close`; add separate `underlying_entry_price` populated from `price_strategy_at()` return value.

- **Phase B — Defensive guards on data ingestion (additive, no behavior change when fresh):**
  - **v3.7.236**: `core/paper_positions.py:344-381` `pick_liquid_monthly_option` add `max_fallback_days` kwarg (default 7), if exceeded return `None` + `source="PENDING_KLINE"`; ledger daemon defers entry instead of freezing stale price.
  - **v3.7.237**: ledger daemon (`core/ledger_daemon.py`) `kline_db max_date > 2 trading days` ⇒ MTM-only mode (no new option entries), surface via dashboard sidebar.

- **Phase C — Exit-simulation contract & per-asset cfg threading (touches all strategy modules):**
  - **v3.7.238**: `core/paper_positions.py:630-700` `simulate_option_exit(..., asset=asset)` threads asset down; per-strategy `simulate_*_position(..., asset, cfg=None)` calls `get_config(asset)` when `cfg is None`. Single dispatch site, 4 strategy modules touched (≤ 20 LOC each).
  - **v3.7.239**: `core/strategies/straddle.py:33` + `core/strategies/short_vol.py:73` call `force_close_at_expiry(...)` before db lookup (mirror v3.7.232 wiring already in buy_call/sell_put).

- **Phase D — Logic corrections in derived metrics (validated via small grid replay):**
  - **v3.7.240**: `scripts/backtest/framework.py:120-126` `max_move_{h}d` replace reverse-rolling with explicit `high[entry_i:exit_i+1].max()` helper to fix off-by-one.
  - **v3.7.241**: Layer 2 backtests (`scripts/backtest/layer2_strategy/*/run_all.py`) restrict samples to `signal_date <= kline_max - max_dte - hold_buffer` to eliminate survivorship bias from un-closed positions.

- **Phase E — Cross-asset IV-awareness & Dashboard parity (validated against existing IV-split backtest):**
  - **v3.7.242**: `core/cross_asset_signal.py` `select_gld_sync_strategy(d, gld_sig, gvz)`: if `GLD bp_low ≤ 0.10 AND GVZ ≥ 25` → `SELL PUT`, else `BUY CALL`. Shadow-log first 2 weeks before flipping live.
  - **v3.7.243**: Dashboard `app.py:1345` deprecate `run_backtest()`; route Dashboard's "真实策略回测" through `generate_daily_signals()` + a thin replay using `buy_signal/buy_type/signal_tier` (no parallel filter logic).

- **Phase F — Regression test harness for the entire correctness floor:**
  - **v3.7.244**: Add `tests/test_expiry_intrinsic.py` covering BC/SP/STRADDLE/SHORT_VOL × {kline-missing, today=expiry, today>expiry} = 12 scenarios; add `tests/test_per_asset_cfg.py` asserting GLD vs SLV simulate_sp_position yield different exit thresholds.

- **Phase G — Model calibration floor (DL Range predictor conformal recalibration):**
  - **Empirical trigger**: Audit on `data/models/dl_range_v2_oos.parquet` (max 2026-05-13) over the last 113 trading days reveals predicted range `[-4.76%, +5.99%]` versus realized `[-0.84%, +0.90%]` — **band width is 5-6× too wide**, with 2026-03 the worst month (pred upper `+7.64%` vs actual `+0.59%`, a 13× upper-band overshoot). In-band rate is a misleading 87.6% because the band is so wide it almost always contains the realized move; `bp_low` rarely approaches 0, so signals fire infrequently AND the few that do fire coincide with genuine breakouts (March 5/5 BC losses concentrated here, in addition to the cross-asset rule defect).
  - **v3.7.245**: New script `scripts/eval/model_calibration_audit.py` produces per-month × per-regime coverage table: predicted vs realized {p10, p50, p90} of |return|, residual quantile ratio (predicted_width / realized_width), upper-only and lower-only coverage. Archives output to `data/backtest_history/v3.7.245_calibration/` with same versioning cadence as other archives. Run on first invocation against both `dl_range_v2_oos.parquet` (GLD) and `dl_range_slv_oos.parquet` (SLV).
  - **v3.7.246**: Adaptive conformal correction (`core/calibration.py`): after `extend_oos_predictions` (`core/data.py:442`) writes raw model output, apply a rolling residual scaler `s = quantile(|actual_lower|/|pred_lower|, 0.85, window=60d)` and corresponding upper scaler; output `pred_upper_calibrated = pred_upper * s_upper`, `pred_lower_calibrated = pred_lower * s_lower`. New columns appended to `*_oos.parquet`, `build_band()` in `core/signals.py` switches to read the `_calibrated` columns when present (graceful fallback). Prior art: project already uses 252-day rolling for `rv_pctile` (signals_v2.py) so rolling residual scaling is structurally familiar.
  - **v3.7.247**: Calibration-gated retrain trigger in `scripts/extend_oos_and_retune.py` (or equivalent cron entry): if 30-day rolling band-overshoot ratio > 2.5, queue retrain on next scheduled window; if > 4.0, immediately retrain. Replaces the current implicit mtime-based heuristic. Logs the decision into `data/models/retrain_log.jsonl` for audit.
  - **v3.7.248**: Per-regime conformal alpha — separate residual scalers per regime (Bull / Bear / Sideways) so a bull-market model isn't penalized by bear-market residuals. Reads regime from the same canonical classifier (which after v3.7.233 will be `min_hold_days=1` everywhere). Per-regime samples must reach `n≥20` before that regime's scaler is used; otherwise fall back to global scaler.
  - **v3.7.249**: Add `tests/test_calibration.py` asserting (a) calibrated bands narrower than raw bands on the test fixture; (b) calibrated `bp_low` distribution shifted toward 0 vs raw (more signals near the boundary, by construction of the scaler); (c) retrain trigger fires when synthetic overshoot exceeds threshold.
  - **Validation gate before live adoption**: re-run Layer 1 directional `scripts/backtest/layer1_signal/directional/run_all.py` with calibrated bands across trailing 10y/5y/3y/1y windows; required outcome: scoreB ≥ raw-band scoreB in at least 3 of 4 windows AND no window worse than -10% vs raw. If gate fails, ship the audit (v3.7.245) and retrain trigger (v3.7.247) but keep raw bands in production until the calibrator can be re-tuned.

**Affected components per patch** (each row maps a patch to its diff surface; phases are mutually independent except where noted):

| Tag | Files | LOC est. | Depends on |
|---|---|---:|---|
| v3.7.233 | regime.py, app.py, build_positions_ledger.py | 3 | — |
| v3.7.234 | paper_positions.py | 1 | — |
| v3.7.235 | positions_ledger.py, paper_positions.py | 8 | — |
| v3.7.236 | paper_positions.py | 6 | — |
| v3.7.237 | ledger_daemon.py | 12 | v3.7.236 |
| v3.7.238 | paper_positions.py, strategies/*.py (4 files) | ~25 | — |
| v3.7.239 | strategies/straddle.py, strategies/short_vol.py | 8 | — |
| v3.7.240 | scripts/backtest/framework.py | 10 | — |
| v3.7.241 | scripts/backtest/layer2_strategy/*/run_all.py (3 files) | 9 | v3.7.240 |
| v3.7.242 | cross_asset_signal.py | 18 | — |
| v3.7.243 | app.py, signals_v2.py | ~40 | v3.7.238 |
| v3.7.244 | tests/ (new) | ~150 | All above |
| v3.7.245 | scripts/eval/ (new) | ~120 | — |
| v3.7.246 | core/calibration.py (new), core/data.py, core/signals.py | ~80 | v3.7.245 |
| v3.7.247 | scripts/extend_oos_and_retune.py, data/models/retrain_log.jsonl (new) | ~40 | v3.7.246 |
| v3.7.248 | core/calibration.py | ~25 | v3.7.246, v3.7.233 |
| v3.7.249 | tests/ | ~80 | v3.7.246-248 |

Total expected diff: ~555 LOC + ~230 LOC tests across 17 tags. Every patch must independently pass a focused grid-replay before being tagged (per the CLAUDE.md "改 cfg → 跑 exit_grid_v2.py 验证" rule). Phase G additionally requires the Layer 1 directional grid gate described in v3.7.249 before promoting calibrated bands to production.

### Objective Evidence

- `core/paper_positions.py:689–700` — `simulate_bc/sp/straddle/short_vol_position(entry_pricing, signal_date, today_dt, db)` called WITHOUT asset/cfg parameter; signatures at `core/strategies/buy_call.py:34`, `sell_put.py:42`, `straddle.py:22`, `short_vol.py:24` declare `cfg=None`, and each module's `if cfg is None: cfg = BCConfig()` (or peer) silently substitutes defaults — confirms Codex finding #1.
- `core/paper_positions.py:344–381` `pick_liquid_monthly_option` computes `fallback_offset_days` (added v3.7.200) but applies no guard; no `max_fallback_days` keyword in the function signature — confirms Codex finding #2.
- `core/strategies/options_exit.py:force_close_at_expiry` only called from `core/strategies/buy_call.py:58` and `sell_put.py:73`; `straddle.py:33` and `short_vol.py:73` still go directly to `first_kdb` lookup and silently return `is_closed=False` when db lacks the contract — confirms Codex finding #3.
- `core/regime.py:24` default `min_hold_days=20`; `_apply_min_hold` (lines 118–138) rewrites historical regime values when a new chunk fails to persist 20 days, which is a forward-looking dependency. 11 production call sites use `RegimeClassifier()` with the default; `scripts/backtest/framework.py:70` explicitly overrides to `=1` — confirms Codex finding #4 and the inconsistency.
- `core/paper_positions.py:177` `sign = 1 if strategy in ("BUY CALL", "SPOT") else -1` applies to SELL PUT, which has positive delta (bullish), so the sign is inverted relative to economic reality — confirms Codex finding #5 (sign error).
- `core/positions_ledger.py:87` stores `entry_pricing["daily_close_price"]` (option's own daily close) into a column conventionally named `entry_spot`, which downstream code treats as the underlying ETF spot — confirms Codex middle finding (field-name vs content mismatch).
- Prior art for incremental v3.7.* single-file patches: commits `v3.7.184` (per-asset SP config split, one file + one validation script), `v3.7.156` (KLINE_DB mtime cache fix, one file), `v3.7.150` (CSV corruption fix, one file), `v3.7.232` (expiry intrinsic helper, three files + one helper). This cadence is the dominant pattern; multi-file rollups (`v3.7.200–211`) explicitly group ≤ 12 micro-fixes and remain individually testable.
- Test infrastructure pattern: `scripts/bc_entry_filter_test.py`, `scripts/test_oi_adjust.py`, `scripts/full_history_backtest.py` all follow "load historical CSV → apply patch → compare WR/sum/scoreB before vs after" — reusable shape for per-patch validation. `tests/` directory currently empty, so v3.7.244 introduces the project's first formal pytest suite without disturbing existing scripts.
- `CLAUDE.md:75-78` Workflow section codifies: "改 cfg → 跑 exit_grid_v2.py 验证 → backtest_pipeline.py all 自动归档版本 → kelly_analysis.py 仓位更新 → compute_strategy_stats.py 刷新 → build_positions_ledger.py 重建" — surgical patches integrate cleanly into this validate-then-archive loop.
- **Model calibration evidence** (drives Phase G):
  - `data/models/dl_range_v2_oos.parquet`: predicted-band mean (last 113d) = `[-4.76%, +5.99%]`, realized mean = `[-0.84%, +0.90%]` ⇒ ~5-6× over-wide bands; in-band coverage 87.6% is misleading because the band almost always engulfs reality.
  - Per-month breakdown 2026-03 pred upper `+7.64%` vs actual `+0.59%` (13× overshoot), 2026-04 11×, 2026-05 8× — calibration drift worsens in trending months.
  - 3 月 GLD BUY CALL 5/5 全亏 (sum=-334%) — all 5 entries are `tier=S-sync, source=SLV-S sync → …` (zero GLD-native triggers); cross-asset rule + over-wide band combine to produce this concentrated loss segment.
  - Model retrain mtime (`dl_range_v2_model.pkl` = 2026-05-21) confirms the issue is calibration, not stale weights.
  - Live OOS extension exists (`core/data.py:442 extend_oos_predictions`) so any calibration scaler can be inserted at the same boundary without disturbing training.

### Known Risks

- **Risk: Per-asset cfg threading depth** (v3.7.238). The asset string must reach `force_close_at_expiry`, `simulate_*_position`, and any future helper. Incomplete threading silently re-introduces the original bug. *Mitigation*: add a guard `assert asset in {"GLD","SLV"}` at every dispatch site, run grep audit before tagging.
- **Risk: `max_fallback_days` clamp too aggressive** (v3.7.236). 7 days may reject legitimate entries during prolonged Moomoo outages. *Mitigation*: emit WARNING (not ERROR) on exceed, allow override via `MAX_KLINE_FALLBACK_DAYS` env var.
- **Risk: SP sign flip cascading semantics** (v3.7.234). Downstream code that reads `realized_pnl_pct` sign as a proxy for "winning side" may invert. *Mitigation*: audit `app.py`, `notifier.py`, and `compute_strategy_stats.py` for direct `pnl > 0` comparisons on SP rows.
- **Risk: Regime `min_hold_days=1` causes intraday jitter** (v3.7.233). Daily regime flips might trigger spurious filter changes. *Mitigation*: regime is computed once per EOD anyway; intraday signal page reads the cached daily regime. Add a 2-week shadow log comparing pre/post WR before declaring done.
- **Risk: Cross-asset IV-aware rule degrades sample size** (v3.7.242). `GLD bp_low ≤ 0.10 AND GVZ ≥ 25` is a narrow regime with only ~10–12 closed trades in kline_db. *Mitigation*: BS-proxy backtest on 5y synthetic option chain to extend evidence base; shadow-log live for 2 weeks before flipping.
- **Risk: Dashboard run_backtest deprecation breaks existing user workflows** (v3.7.243). Streamlit pages may rely on its specific output schema. *Mitigation*: keep `run_backtest()` as a thin wrapper around the unified pipeline for one release; add deprecation warning; remove in v3.8.
- **Risk: Validation script load on developer time**. 12 tags × per-patch grid replay = real elapsed time. *Mitigation*: tie tests into `make validate-patch TAG=v3.7.238` so each patch's evidence is reproducible by anyone.
- **Risk: Conformal calibration shrinks bands so aggressively that signal frequency over-corrects** (v3.7.246). Narrower bands ⇒ `bp_low` near 0 more often ⇒ false positives. *Mitigation*: hard gate at the Layer 1 directional grid (3-of-4 windows scoreB ≥ raw) before promoting calibrated bands; v3.7.246 lands the column shadow-first (`build_band` still defaults to raw until cutover flag).
- **Risk: Per-regime conformal alpha (v3.7.248) hits small-sample regimes (e.g., Bear in recent history is short)**. *Mitigation*: enforce `n ≥ 20` per-regime sample minimum; fall back to global scaler otherwise; log the fallback into `retrain_log.jsonl` so under-sampled regimes are visible.
- **Risk: Calibration-gated retrain (v3.7.247) thrashes when overshoot oscillates near threshold**. *Mitigation*: hysteresis — require 5 consecutive days above threshold before triggering, and a 7-day cooldown after retrain.

## Alternative Directions Considered

### Alt-1: Single-Source-of-Truth Signal Pipeline
- Gist: Collapse 34 scattered call sites that currently bypass or re-implement parts of `generate_daily_signals()` into one canonical `SignalPipeline.build_signals(asset, ..., gvz_series, force_apply_iv_filter, include_cross_asset)` factory under `core/signal_pipeline.py`. Three Dashboard modes (`app.py:1286`, `:4733`, `:5301`), the ledger builder, the backtest framework, and 29 analysis scripts all consume this factory, so IV filter, sp_score, MA filter, regime, and tier semantics drift becomes structurally impossible. Includes optional metadata dict (`iv_filter_applied`, `gvz_freshness_days`, `regime_classifier`) so callers can introspect what the pipeline actually applied.
- Objective Evidence:
  - `core/signals_v2.py:87` canonical `generate_daily_signals()` already accepts `asset` and `gvz_series` — the factory needs to wrap, not replace.
  - `app.py:4733` calls `generate_daily_signals(...)` WITHOUT `gvz_series`, silently disabling IV filter (signals_v2.py:298–321) for "回测分析" page.
  - `scripts/build_positions_ledger.py:94` and `scripts/backtest/framework.py:82` already pass both `asset` and `gvz_series` — the correct pattern, but is not centralized.
  - 29 analysis scripts grep-found with inconsistent parameter sets.
- Why not primary: Higher architectural change surface; 34 call sites must migrate, and Dashboard hot-path performance must be re-validated. Surgical (Primary) addresses the same dashboard-drift bug (v3.7.243) with smaller blast radius. This alt is the natural next step once Primary's per-bug correctness floor is in place.

### Alt-2: Typed Exit Context Contract
- Gist: Replace each strategy's positional cfg argument with a single typed `ExitContext` dataclass (asset, signal_date, today_dt, kline_db, cfg, regime, max_stale_days, live_spot/high/low). Construction at the dispatcher (`paper_positions.simulate_option_exit`) requires all fields, making "missing asset" or "missing cfg" structurally impossible. `force_close_at_expiry` is rewritten to consume `ExitContext`, removing its current dependency on parsing the asset out of an option code string.
- Objective Evidence:
  - 5 existing `@dataclass` configs (`AssetConfig`, `BCConfig`, `SPConfig`, `StraddleConfig`, `ShortVolConfig`) plus `PaperPosition` show strong precedent.
  - 43 invocations of `simulate_*_position` across `paper_positions.py`, `scripts/exit_grid_v2.py:240`, `scripts/options_per_tier_validate.py:65`, etc., would all migrate.
  - No mypy/pyright config in the repo currently → contract enforcement is runtime-assertion-based; some leak risk remains until a type checker is wired up.
- Why not primary: The contract refactor delivers the same correctness wins as Primary's v3.7.238, but at a wider migration surface (43 call sites vs 6). Primary's threading approach gets us to correctness faster; the contract becomes an attractive follow-on once correctness is stabilized.

### Alt-3: Data-Freshness State Machine + UX Banner
- Gist: Model each data source (kline_db, ETF OHLC, GVZ, OOS predictions) as an explicit FRESH ≤ 2d / STALE 2-7d / FROZEN > 7d state machine wired into `ledger_daemon` and the Dashboard sidebar. FROZEN ⇒ no new option entries (only MTM via `force_close_at_expiry` fallback); STALE ⇒ degraded confidence in sidebar; FRESH ⇒ normal. Snapshot file augmented with `data_sources` block listing per-source `{state, last_update_date, age_days}` for offline auditability.
- Objective Evidence:
  - `core/paper_positions.py:320-341` `_KLINE_DB_PATH` + `_load_kline_db()` mtime cache (v3.7.156) — precedent for mtime-based freshness detection.
  - `app.py:4670-4681` sidebar already renders ledger mtime with 🟢/🟡/🔴 by `age_min` thresholds — direct UI precedent.
  - `core/training_status.py:47-68` `get_model_age_days()` + `is_stale(max_age_days=7)` — age threshold pattern.
  - `data/positions_ledger_meta.json` already has `evaluated_through` per asset + `last_run_at` — extension point.
  - No `Enum`/`FreshnessState` class exists in code — new abstraction is required.
- Why not primary: Solves Codex bug #2 + the freshness optimizations comprehensively, but reaches beyond strict correctness into UX surface. Primary tackles correctness floor first; v3.7.236+v3.7.237 in Primary deliver the kline freshness hard-gate; this alt elevates that gate to a full lifecycle view as a follow-on.

### Alt-4: Look-Ahead Eradication Audit
- Gist: Beyond the known regime min_hold_days=20 bug and the `framework.py:120` `max_move` off-by-one, sweep every shift/rolling/expanding operation in `core/signals_v2.py`, `core/signals.py`, `core/regime.py`, `scripts/backtest/framework.py`, `/Users/yhdong/Gold/src/features/`, and `core/features_1h.py`. Build a runtime validator that, for every column in `features_all.parquet`, asserts the value at day t depends only on info-set ≤ t (or is explicitly prefixed `fwd_`). Walk-forward train/test boundary already validated; this adds enforcement.
- Objective Evidence:
  - `walk_forward_full.py:1-20` already documents no-look-ahead assumptions in a comment but does not enforce them.
  - `scripts/backtest/framework.py:108-134` contains `next_open = shift(-1)`, `ext_close = shift(-(h+1))`, `rv_fwd = rolling(h).std().shift(-h)` — intentional fwd labels, all correctly prefixed `fwd_` or named `*_fwd_*`.
  - `scripts/backtest/framework.py:120` `high.rolling(h).max().shift(-(h+1))` has the off-by-one Codex flagged.
  - `scripts/setup_data.py:608-609` `high.shift(-1).rolling(5).max().shift(-4)` = net shift(-5), purpose unclear from naming — suspect mislabel.
  - `core/features_1h.py:314-315` `fwd_high/fwd_low` lacks integration audit into `features_all.parquet`.
  - 0 columns in `features_all.parquet` are prefixed `fwd_` — features and labels are correctly separated today, but the enforcement is by convention only.
- Why not primary: Highest statistical-correctness leverage but tooling-heavy (new detector, new test harness, no existing validator to extend). Primary's v3.7.240 fixes the named off-by-one cheaply; this alt converts that one-shot fix into a permanent CI gate, which is best layered on after the correctness floor stabilizes.

### Alt-5: Declarative Risk & Cross-Asset Policy Layer
- Gist: Externalize all gate-keeping (cross-asset rules, IV-regime strategy switches, consecutive-loss circuit breaker, max_open_per_asset, exposure caps, leverage tiers, expiry handling) into a single `risk_policy.yaml` evaluated by a new `core/policy_engine.py`. The PolicyEngine returns structured verdicts `{entry_allowed, strategy_override, leverage, cross_asset_enabled}` per signal, replacing the 14+ inline conditionals scattered across `signals_v2.py`, `cross_asset_signal.py`, `strategy_config.py`, `paper_positions.py`, `regime.py`, `options_exit.py`. ~150 lines of declarative YAML replace ~400 lines of scattered Python branches.
- Objective Evidence:
  - `core/strategy_config.py:34-147` `AssetConfig` dataclass + `ASSET_CONFIGS` dict — per-asset parameterization precedent.
  - `core/strategy_configs.py:51-89` `FuturesConfig` with tier-aware leverage (`tier_s_leverage=10`, `tier_a=10`, `tier_b=5`) — tier-rule precedent.
  - `/Users/yhdong/Gold/config/settings.yaml` (and `config.yaml`) — declarative YAML infrastructure already accepted.
  - `core/signals_v2.py:24,41-96` `CONSECUTIVE_STOP`, `BUY_BP`, inline IV regime checks — concrete inline gate-keepers to externalize.
  - `core/cross_asset_signal.py:29-83` hard-coded `CROSS_ENABLED`, `CROSS_TIERS`, `CROSS_STRATEGY` constants — the canonical example to lift into YAML.
- Why not primary: The largest re-architecture in the set (new engine, new schema, new YAML lifecycle), and the cross-asset IV-aware change (the highest-value single rule) is tackled in Primary's v3.7.242 at ~18 LOC. Treat this alt as the v3.8 milestone after Primary's correctness floor + Alt-2's typed contract are in place — that's when an explicit policy engine has the most reuse.

## Synthesis Notes

The Primary series is intentionally minimal so each fix can be tagged, validated, and reverted independently — but it is not architecturally maximal. With Phase G added, the Primary now spans two complementary floors: (a) **architectural correctness** (v3.7.233-244, Phases A-F) eliminates Codex-flagged seams where per-asset cfg, regime, expiry handling, and Dashboard signal logic silently diverge from the canonical chain; (b) **model calibration** (v3.7.245-249, Phase G) corrects the 5-6× over-wide DL Range predictor bands that combine with the cross-asset defect to concentrate March's 5/5 BC losses, plus installs a retrain trigger so future drift becomes self-healing rather than silent. The two floors are independent — Phase G's calibration scaler reads OOS predictions whose generation Phase F's tests now cover — and can be merged in either order.

If the project later wants to consolidate, the alternatives fold in cleanly: Alt-1 (Signal Pipeline) absorbs v3.7.243 (Dashboard run_backtest deprecation) and rationalizes the 29 analysis scripts; Alt-2 (Typed Exit Context) absorbs v3.7.238 (asset threading) and v3.7.239 (Straddle/ShortVol expiry wiring) into one structural change; Alt-3 (Freshness State Machine) extends v3.7.236/237 (kline freshness guard) from a single-purpose gate into a full lifecycle observable that naturally hosts the Phase G calibration freshness signal; Alt-4 (Look-Ahead Audit) elevates v3.7.240 (max_move off-by-one) and v3.7.233 (regime min_hold_days) into a permanent enforcement layer and pairs naturally with Phase G's per-regime conformal alpha; Alt-5 (Policy Layer) lifts v3.7.242 (cross-asset IV-awareness) into a YAML-driven engine that can host future risk rules without code churn. The recommended trajectory: Primary delivers correctness + calibration now in 17 tags, then v3.8 picks Alt-2 + Alt-4 together (typed context + leak detector form a coherent "structural correctness" layer that the calibrator inherits for free), and v3.9 picks Alt-1 + Alt-5 (one canonical pipeline driven by one declarative policy, with calibration as a first-class policy input).

--- Original Design Draft End ---

---

## BitLesson Selection (REQUIRED FOR EACH TASK)

Before executing each task or sub-task, you MUST:

1. Read @/Users/yhdong/GoldDash/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task to select relevant lesson IDs
3. Follow the selected lesson IDs (or `NONE`) during implementation

Include a `## BitLesson Delta` section in your summary with:
- Action: none|add|update
- Lesson ID(s): NONE or comma-separated IDs
- Notes: what changed and why (required if action is add or update)

Reference: @/Users/yhdong/GoldDash/.humanize/bitlesson.md

---

## Goal Tracker Rules

Throughout your work, you MUST maintain the Goal Tracker:

1. **Before starting a round**: Re-anchor on the original plan and current round contract
2. **Before starting a task**: Mark the relevant mainline task as "in_progress" in Active Tasks
   - Confirm Tag/Owner routing is correct before execution
3. **Active Tasks** are MAINLINE tasks only - side issues do not belong there
4. **Blocking Side Issues** are reserved for issues that truly stop mainline progress
5. **Queued Side Issues** are non-blocking and must not take over the round
6. **After completing a mainline task**: Move it to "Completed and Verified" with evidence (but mark as "pending verification")
7. **If you discover the plan has errors**:
   - Do NOT silently change direction
   - Add entry to "Plan Evolution Log" with justification
   - Explain how the change still serves the Ultimate Goal
8. **If you need to defer a task**:
   - Move it to "Explicitly Deferred" section
   - Provide strong justification
   - Explain impact on Acceptance Criteria
9. **If you discover new issues**:
   - Add to "Blocking Side Issues" only if mainline progress is blocked
   - Otherwise add to "Queued Side Issues" or keep them as `[queued]` tasks/backlog

---

Note: You MUST NOT try to exit `start-rlcr-loop` loop by lying or edit loop state file or try to execute `cancel-rlcr-loop`

After completing the work, please:
0. If you have access to the `code-simplifier` agent, use it to review and optimize the code you just wrote
1. Finalize @/Users/yhdong/GoldDash/.humanize/rlcr/2026-05-23_15-38-04/goal-tracker.md (this is Round 0, so you are initializing it - see "Goal Tracker Setup" above)
2. Write your round contract into @/Users/yhdong/GoldDash/.humanize/rlcr/2026-05-23_15-38-04/round-0-contract.md
3. Commit your changes with a descriptive commit message
4. Write your work summary into @/Users/yhdong/GoldDash/.humanize/rlcr/2026-05-23_15-38-04/round-0-summary.md
