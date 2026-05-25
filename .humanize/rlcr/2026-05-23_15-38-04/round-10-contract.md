# Round 10 Contract (FINAL)

## Round Objective

Close the plan: **task-e3 (Dashboard `run_backtest` deprecation + parity harness, AC-14)**. After this round all 15 ACs are LANDED.

## In-Scope Tasks

| Task | AC | Tag | Notes |
|------|----|----|-------|
| task-e3 | AC-14 | v3.7.249 | (a) `core/dashboard_parity.py` — new module with `run_unified_backtest(...)` and `parity_check(legacy_result, unified_result)`; (b) `app.py:run_backtest` emits a `DeprecationWarning` and continues to function as the legacy spot-level replay (the wrapper does NOT replace legacy behavior this round — it adds the parity check alongside); (c) pytest verifies the parity contract on a synthetic 60-day fixture: intraday exit-event (StopLoss / Pullback / ACTIVE) counts match within ±1, and any signal-column drift is recorded with a documented attribution. |

## Out-of-Scope

- Removing the legacy `run_backtest` body (deferred to v3.8 — the contract specifies "one-release deprecation window").
- Live wire-up of the unified path into the Streamlit page (the unified path becomes the default in v3.8 once the parity harness has been observed on real traffic).

## Verification Plan

### task-e3
pytest `tests/test_dashboard_parity.py`:
- **Identical-pipeline equivalence**: when both legacy and unified paths are passed the SAME synthetic fixture and SAME signal series, intraday exit-event counts match exactly (±0); the parity harness reports `parity=PASS`.
- **Signal-drift attribution**: when the unified path's `buy_signal` column differs from the legacy on a known row (e.g., IV filter active in unified, not in legacy), the difference appears in `signal_drift_attribution.csv` with a documented reason; the parity harness reports `parity=PASS_WITH_DRIFT`.
- **Exit-event count drift > 1 → fail**: synthetic mismatch that produces 5 extra StopLoss events triggers `parity=FAIL_EXIT_EVENT_COUNT`.
- **`run_backtest` DeprecationWarning**: importing `app` and calling the wrapper-augmented `run_backtest` (or however the deprecation lives) emits the warning.

## Commit Discipline

- `v3.7.249`: code + tests + final goal-tracker close-out.

## Bitlessons Consulted

- `BL-20260525-gate-rule-clarity` — applies to the parity check too: "compare against contract intent, not raw numbers".
- `BL-20260525-audit-scope-by-baseline-diff` — n/a here.

## Risk Watch

- **Streamlit import in tests**: `app.py` is a Streamlit app and importing it may trigger Streamlit's initialization. Keep tests focused on the parity *module* (`core/dashboard_parity.py`) and not on importing `app.py` directly. The `DeprecationWarning` verification can be done by reading the source for the warning emission rather than executing the Streamlit page.
- **Legacy `run_backtest` complexity**: I do NOT need to rewrite the legacy body. The parity wrapper compares two side-by-side function outputs; the unified function is a simpler analog that I construct from `generate_daily_signals` + a small intraday-replay helper.
- **`signal_drift_attribution.csv` format**: lock the column schema with the test so future writers don't drift the schema.
