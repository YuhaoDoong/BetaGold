# Goal Tracker

## IMMUTABLE SECTION

### Ultimate Goal
Repair the GLD/SLV trading system's correctness floor and the DL Range predictor's calibration floor through a sequence of small, independently verifiable patches that each (a) target one root cause, (b) ship with a focused validation script or pytest case, (c) can be reverted without cascading state corruption, and (d) integrate cleanly with the existing "改 cfg → 跑 grid → 归档版本" workflow. Calibration goal is coverage repair (current GLD v2 OOS: 54.9% versus 80% training target, 5-day forward labels, width ratios 1.95×/1.66×) NOT unconditional band narrowing. Plan covers exactly the AC enumerated below; OI mainline / risk controls migration / yfinance bid-ask fallback / exposure caps are explicitly out of scope and tracked separately.

### Acceptance Criteria
- AC-1: Calibration audit reproducible and uses correct 5-day forward label definition (per `src/models/train_dl_range.py:build_targets`).
- AC-2: Production `RegimeClassifier` has no forward lookback; every production call site passes `min_hold_days=1`.
- AC-3: Exit simulation receives per-asset configuration via `get_option_exit_config(asset, strategy)` resolver; legacy `cfg=None` silent path is deprecated.
- AC-4: Expiry-intrinsic force-close covers BC/SP/STRADDLE/SHORT_VOL with strategy-specific intrinsic; SHORT_VOL uses `max(call_wing, put_wing) - credit` for asymmetric wings.
- AC-5: Cross-asset strategy selector is a pure function `select_gld_sync_strategy(signal_date, gld_signal_row, gvz_value, gvz_asof_date)` returning `{strategy, reason, gvz_status}`; shadow log is caller-side; `live_cutover` requires ≥14-day shadow accumulation.
- AC-6: Data freshness gate blocks new option entries when `kline_db max_date < today - 3 trading days`; futures + MTM + expiry-intrinsic unaffected; PENDING_KLINE distinct from NO_CONTRACT; dedup by signal_date.
- AC-7: Layer 2 backtest reports `n_signal/n_entered/n_closed/n_open/n_skipped_stale/n_skipped_no_contract`; per-leg DTE filter.
- AC-8: Calibrated bands shadow-first; live cutover requires Layer 1 grid gate pass; conformal scaler uses horizon-aware maturity lag (`label_end_date < calibration_as_of_date`), not generic `shift(1)`.
- AC-9: Calibration-gated retrain trigger with 5-day hysteresis + 7-day cooldown + zero-width guard.
- AC-10: Test harness reproducible via `pytest tests/` from clean `gold` env; per-tag VALIDATION.md archive.
- AC-11: No plan-progress markers ("AC-", "Milestone", "Phase A", "Step N") leak into source files.
- AC-12: SELL PUT realized P&L sign correct in spot fallback path (sign = +1 for SP, reflecting positive delta).
- AC-13: `entry_spot` schema migration: new `underlying_entry_price` field, one-release alias, migration note in `positions_ledger_meta.json`.
- AC-14: Dashboard `run_backtest` deprecation preserves intraday exit semantics (StopLoss/Pullback/ACTIVE within ±1); signal-column drift audited via `signal_drift_attribution.csv`.
- AC-15: Layer 1 `max_move_{h}d` window has no off-by-one (explicit `entry_i`/`exit_i` indexing).

---

## MUTABLE SECTION

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |

#### Active Tasks
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| task-a1 | AC-2 | done (Round 0, pending verification) | coding | claude | v3.7.233: default → 1, 6 production sites explicit, 24 research sites use new default, doc archive |
| task-a2 | AC-12, AC-11 | done (Round 0, pending verification) | coding | claude | v3.7.234: SP sign -1 → +1; STRADDLE/SHORT_VOL sign=0; grep audit clean |
| task-a3 | AC-13, AC-11 | done (Round 0, pending verification) | coding | claude | v3.7.235: underlying_entry_price field added; entry_spot alias; meta migration note |
| task-f1 | AC-10 | done (Round 0, pending verification) | coding | claude | v3.7.236-prep: pytest>=7.0 in requirements.txt; tests/conftest.py + fixtures/; first regime no-leak regression test PASSES |
| task-b1 | AC-6 | done (Round 1, pending verification) | coding | claude | v3.7.236: max_fallback_days=7 + _kline_db_freshness_status + price_strategy_at PENDING_KLINE source |
| task-b2 | AC-6 | done (Round 1, pending verification) | coding | claude | v3.7.237: core/data_freshness.py FRESH/STALE/FROZEN + ledger daemon log + build_positions_ledger [freshness] skip print + 7 pytest cases PASS |
| task-c1 | AC-3 | done (Round 2, pending verification) | coding | claude | v3.7.238: get_option_exit_config resolver + simulate_option_exit asset 穿透 + DeprecationWarning + 6 pytest PASS |
| task-c2 | AC-4 | done (Round 2, pending verification) | coding | claude | v3.7.239: force_close_at_expiry 扩 long_vol/iron_condor + asymmetric IC max_risk + 17 pytest PASS (含 asymmetric wing fixture) |
| task-d1 | AC-15 | done (Round 4, pending verification) | coding | claude | v3.7.241: forward_window_extreme helper + 7 pytest (legacy off-by-one 强对比) |
| task-d2 | AC-7 | done (Round 4, pending verification) | coding | claude | v3.7.242: run_layer2_backtest_with_disposition + per-leg DTE + 2 runners 重构 + 10 pytest (reconciliation invariant + 5 disposition 分支) |
| task-e1 | AC-5 | done (Round 3, pending verification) | coding | claude | v3.7.240: select_gld_sync_strategy pure fn + write_shadow_record + live_cutover_allowed + 19 pytest PASS |
| task-e2 | AC-5 | done (Round 3, pending verification) | analyze | claude | v3.7.240 replay archive: 5/5 March BC entries would switch to SP; native SP same-month +66.7% vs BC -334.3% = +401pp counterfactual |
| task-e3 | AC-14 | pending | coding | claude | Dashboard run_backtest wrapper + parity assertion + DeprecationWarning |
| task-f2 | AC-4 | pending | coding | claude | tests/test_expiry_intrinsic.py — 16 scenarios incl asymmetric IC fixture |
| task-f3 | AC-3 | pending | coding | claude | tests/test_per_asset_cfg.py |
| task-f4 | AC-8, AC-9 | done (Round 7, pending verification) | coding | claude | scaler 12 (R6) + retrain 11 (R7) + per-regime 6 (R7) = 29 calibration tests, 全套 105 PASS |
| task-g1 | AC-1 | done (Round 5, pending verification) | coding | claude | v3.7.243: scripts/eval/model_calibration_audit.py + 8 pytest (label-def lock + zero-actual NaN + coverage formula 锁定 eval_range) |
| task-g2 | AC-1 | done (Round 5, pending verification) | analyze | claude | AUDIT_REPORT.md: GLD width 1.948/1.663 cov 54.87%, SLV 1.425/1.871 cov 53.66%, per-month asymmetric drift signature, draft 5-6× 错算原因详记 |
| task-g3 | AC-8 | done (Round 6, pending verification) | coding | claude | v3.7.244: core/calibration.py:apply_rolling_conformal_scaler + horizon-aware maturity-lag + per-side split-conformal + 12 pytest; smoke on GLD 113d cov_both +4.4pp |
| task-g4 | AC-9 | done (Round 7, pending verification) | coding | claude | v3.7.245: evaluate_retrain_trigger 纯函数 + hysteresis(5)+cooldown(7)+zero_width_floor + 11 pytest |
| task-g5 | AC-8 closure | done (Round 7, pending verification) | coding | claude | v3.7.246: apply_rolling_conformal_scaler 加 regime + min_regime_pool=20 fallback + 6 pytest |
| task-g6 | AC-8 closure | done (Round 8, pending verification) | analyze | claude | v3.7.247: calibration_gate_grid.py 5 windows × 2 assets + 9 pytest. 实测 gate_passed=False for both (joint coverage_both 在长窗回退); 验证 gate 设计目的, scaler ship shadow-only |
| task-h1 | AC-10 | done (Round 9, pending verification) | coding | claude | v3.7.248: scripts/validate-patch.sh + scripts/eval/normalize_pytest_output.py + 7 pytest (byte-identical reproducibility) |
| task-h2 | AC-11 | done (Round 9, pending verification) | analyze | claude | v3.7.248: scripts/eval/audit_plan_markers.sh + REPORT.md ac11_passed=true (clean diff vs v3.7.232, 0 violations) |

### Blocking Side Issues
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|

### Queued Side Issues
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|
| Moomoo daily 100 kline quota requires multi-day backfill to reach near-real-time kline_db | 0 | Not blocking correctness floor; AC-6 freshness gate handles staleness gracefully | When AC-6 freshness banner shows persistent FROZEN state >14 days |
| Stale duplicate `kline_db` parquet in `GoldDash/data/options_history/` (74 codes, max 2026-03-10) | 0 | Not referenced by any production code path (verified by grep); delete deferred to housekeeping | When freshness gate UX is implemented |

### Completed and Verified
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
| OI mainline correction promotion | (out-of-scope) | Plan inception | DEC-7 user-confirmed defer; optimization-tier not blocking correctness | v3.8 follow-on plan |
| Full risk controls migration (exposure caps, max_open_per_asset, ≥2 consecutive-loss circuit) | (out-of-scope) | Plan inception | DEC-7 user-confirmed defer | v3.8 follow-on plan |
| yfinance bid/ask mid fallback | (out-of-scope) | Plan inception | DEC-7 user-confirmed defer | v3.8 follow-on plan |
| Full FRESH/STALE/FROZEN tiered freshness UX (binary gate suffices for Lower Bound) | (AC-6 reduced scope) | Round 0 | Lower Bound allows binary gate; tiered UX is upper bound optional | When sidebar UX is being touched |
