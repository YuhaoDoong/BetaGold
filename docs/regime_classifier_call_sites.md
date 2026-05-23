# RegimeClassifier Call-Site Audit (v3.7.233)

## Default Behavior

`core/regime.py:RegimeClassifier.__init__` default `min_hold_days` was changed from `20` to `1` in v3.7.233.

The 20-day debounce silently rewrote historical regime labels by looking forward (`_apply_min_hold` overwrote `values[i:j] = current` whenever a new regime chunk was shorter than 20 days). That is a structural look-ahead bias when the classifier is consumed by anything that does not have access to future data — i.e., live production signal generation, ledger building, intraday backfill, and continuous runners.

## Production-Critical Call Sites (explicit `min_hold_days=1` added)

These paths feed into live trading decisions or daily ledger writes. They carry an explicit `min_hold_days=1` in addition to the new default, as belt-and-suspenders against future default reverts:

- `app.py:112` — Streamlit dashboard daily signal page
- `scripts/build_positions_ledger.py:92` — daily ledger rebuild (single source of truth)
- `scripts/continuous_runner.py:81` — continuous live signal runner
- `scripts/backfill_intraday_signals.py:85` — intraday signal backfill
- `scripts/build_futures_signals.py:71` — futures live signal builder
- `scripts/cross_asset_sync.py:48` — cross-asset live sync

Backtest framework (`scripts/backtest/framework.py:70`) was already explicit `min_hold_days=1` prior to v3.7.233 and is unchanged.

## Research / Backtest Call Sites (default-default `RegimeClassifier()`)

These scripts use the now-default `min_hold_days=1`. Most are research/grid/validation scripts and the 1-day behavior is correct for them as well (avoiding label leakage in any historical analysis is the prudent default).

If a specific research script genuinely needs the old 20-day forward-looking debounce (e.g., for a published reference number), it must explicitly write `RegimeClassifier(min_hold_days=20)` and document why look-ahead is acceptable in that script's docstring.

Affected files (all default `RegimeClassifier()` after v3.7.233 = 1-day, not 20-day):
- `scripts/futures_iv_filter_test.py`
- `scripts/signal_alpha_test.py`
- `scripts/bc_paired_single_vs_spread.py`
- `scripts/options_per_tier_validate.py`
- `scripts/signal_filter_deep.py`
- `scripts/options_sl_grid_5y.py`
- `scripts/bc_gld_spot_baseline.py`
- `scripts/futures_real_tp_grid.py`
- `scripts/futures_early_tp_grid.py`
- `scripts/walk_forward_validate.py`
- `scripts/options_5y_bs_proxy.py`
- `scripts/backtest_pipeline.py` (two sites)
- `scripts/full_history_backtest.py`
- `scripts/real_options_backtest.py`
- `scripts/exit_grid_v2.py`
- `scripts/tune_thresholds.py`
- `scripts/three_lever_grid.py`
- `scripts/diag_2026q1_bc_signals.py`
- `scripts/multi_window_validate.py`
- `scripts/bc_gld_single_transparent.py`
- `scripts/train_1h_model.py`
- `scripts/bc_single_vs_spread_grid.py`
- `scripts/futures_grid_5y.py`
- `scripts/regime_and_signal_optim.py` (two sites)

## How To Validate The Flip

After v3.7.233 lands, running any production path against historical data should produce a `regime` series whose value at day `t` depends only on observations `≤ t`. The simplest check is:

```python
from core.regime import RegimeClassifier

cfg_full = RegimeClassifier(min_hold_days=1).classify(features).iloc[:, -1]
cfg_truncated = RegimeClassifier(min_hold_days=1).classify(features.iloc[:-30]).iloc[:, -1]
# regime at day t in the full series must equal regime at day t in the truncated series
# (when t ≤ len(features) - 30)
assert cfg_full.loc[: cfg_truncated.index[-1]].equals(cfg_truncated)
```

If the assertion holds, no forward leak exists. Under the old `min_hold_days=20`, this assertion would fail because the full series might rewrite labels at day `t` based on what happened at `t+1..t+19`.
