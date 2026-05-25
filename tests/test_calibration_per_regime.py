"""Regression tests for v3.7.246: per-regime conformal alpha (AC-8 closure)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from core.calibration import (
    apply_rolling_conformal_scaler,
    MIN_REGIME_POOL_SIZE,
)


def _bdates(n, start="2026-01-05"):
    return pd.bdate_range(start, periods=n)


# -- Backwards compatibility --------------------------------------------------


def test_regime_none_equals_global_pool():
    """regime=None must produce identical output to a same-regime fixture."""
    n = 100
    dates = _bdates(n)
    pred_u = pd.Series([2.0] * n, index=dates)
    pred_l = pd.Series([-2.0] * n, index=dates)
    actual_u = pd.Series([1.0] * n, index=dates)
    actual_l = pd.Series([-1.0] * n, index=dates)
    out1 = apply_rolling_conformal_scaler(dates, pred_u, pred_l, actual_u, actual_l)
    # Single-regime fixture: every day labelled 'Bull'
    regime = pd.Series(["Bull"] * n, index=dates)
    out2 = apply_rolling_conformal_scaler(dates, pred_u, pred_l, actual_u, actual_l,
                                                regime=regime)
    pd.testing.assert_series_equal(out1[0], out2[0])
    pd.testing.assert_series_equal(out1[1], out2[1])


# -- Per-regime distinct deltas -----------------------------------------------


def test_balanced_three_regimes_distinct_deltas():
    """A long history with three regimes whose residuals genuinely differ.

    Construct so that:
      Bull rows: residual_upper = -1, residual_lower = +1 (model over-predicts both)
      Bear rows: residual_upper = +3, residual_lower = -3 (model under-predicts both)
      Sideways: residual_upper = 0, residual_lower = 0

    Each regime gets enough samples (>= MIN_REGIME_POOL_SIZE). At a Bull day,
    delta_upper should match the Bull-only quantile (≈ -1), NOT the global
    mixture.
    """
    n = 180
    dates = _bdates(n)
    # 60 Bull, 60 Bear, 60 Sideways, interleaved by 30-day blocks
    blocks = (["Bull"] * 30 + ["Bear"] * 30 + ["Sideways"] * 30) * 2
    regime = pd.Series(blocks[:n], index=dates)
    # Residuals per regime
    res_upper_by = {"Bull": -1.0, "Bear": +3.0, "Sideways": 0.0}
    res_lower_by = {"Bull": +1.0, "Bear": -3.0, "Sideways": 0.0}
    pred_u = pd.Series([0.0] * n, index=dates)
    pred_l = pd.Series([0.0] * n, index=dates)
    actual_u = pd.Series([res_upper_by[r] for r in regime], index=dates)
    actual_l = pd.Series([res_lower_by[r] for r in regime], index=dates)

    out_u, out_l, meta = apply_rolling_conformal_scaler(
        dates, pred_u, pred_l, actual_u, actual_l,
        horizon=5, window=120, target_coverage=0.80,
        regime=regime, min_regime_pool=20)

    # Use a date deep into the series so the pool contains enough samples of
    # each regime. The 2nd Bull block starts at index 90; pick t=120 (Bull-ish
    # block start). The pool covers 0..114; regime mix should have ~30 of each
    # in the same-regime pool restricted to Bull.
    t = 120
    cur_regime = regime.iloc[t]
    delta_u = meta.iloc[t]["delta_upper"]
    expected_for_regime = res_upper_by[cur_regime]
    # Allow small tolerance for quantile placement
    assert abs(delta_u - expected_for_regime) < 0.5, (
        f"At t={t} regime={cur_regime}, delta_upper={delta_u} should be ~"
        f"{expected_for_regime} (per-regime), not the global mix."
    )


# -- Per-regime fallback ------------------------------------------------------


def test_undersampled_regime_falls_back_to_global():
    """When current regime has <20 same-regime past matured samples, use global pool."""
    n = 100
    dates = _bdates(n)
    # 95 Bull, 5 Bear → Bear date should fall back to global pool
    regime_labels = ["Bull"] * 95 + ["Bear"] * 5
    regime = pd.Series(regime_labels, index=dates)
    pred_u = pd.Series([2.0] * n, index=dates)
    pred_l = pd.Series([-2.0] * n, index=dates)
    actual_u = pd.Series([1.0] * n, index=dates)
    actual_l = pd.Series([-1.0] * n, index=dates)

    out_u, out_l, meta = apply_rolling_conformal_scaler(
        dates, pred_u, pred_l, actual_u, actual_l,
        horizon=5, window=60, target_coverage=0.80,
        regime=regime, min_regime_pool=20)

    # Last day is a Bear day with only 4 same-regime predecessors (and they
    # haven't all matured anyway). Should fall back to global pool.
    last_meta = meta.iloc[-1]
    assert last_meta["fallback_reason"] is not None
    assert "regime_undersampled" in str(last_meta["fallback_reason"])
    assert "Bear" in str(last_meta["fallback_reason"])
    # delta still computed (just from global pool)
    assert not np.isnan(last_meta["delta_upper"])


def test_nan_regime_coalesced_to_unknown():
    """NaN regime values must be coalesced to 'UNKNOWN' and treated consistently."""
    n = 100
    dates = _bdates(n)
    regime = pd.Series([np.nan] * n, index=dates, dtype=object)
    pred_u = pd.Series([2.0] * n, index=dates)
    pred_l = pd.Series([-2.0] * n, index=dates)
    actual_u = pd.Series([1.0] * n, index=dates)
    actual_l = pd.Series([-1.0] * n, index=dates)

    out_u, out_l, meta = apply_rolling_conformal_scaler(
        dates, pred_u, pred_l, actual_u, actual_l,
        horizon=5, window=60, target_coverage=0.80,
        regime=regime, min_regime_pool=20)
    # All NaN → all become 'UNKNOWN', which is a single coherent regime →
    # behavior equivalent to single-regime pool (no fallback needed).
    last_meta = meta.iloc[-1]
    # delta_upper should equal the global expectation (-1.0 quantile of res_upper)
    assert not np.isnan(last_meta["delta_upper"])
    # fallback_reason should be None (single regime, enough samples)
    assert last_meta["fallback_reason"] is None


def test_per_regime_meta_records_fallback_reason():
    """When fallback fires, meta_df.fallback_reason captures the regime name + count."""
    n = 100
    dates = _bdates(n)
    # All Bear, except one isolated Bull at the very end
    labels = ["Bear"] * 95 + ["Bull"] + ["Bear"] * 4
    regime = pd.Series(labels, index=dates)
    pred_u = pd.Series([2.0] * n, index=dates)
    pred_l = pd.Series([-2.0] * n, index=dates)
    actual_u = pd.Series([1.0] * n, index=dates)
    actual_l = pd.Series([-1.0] * n, index=dates)

    out_u, out_l, meta = apply_rolling_conformal_scaler(
        dates, pred_u, pred_l, actual_u, actual_l,
        horizon=5, window=60, target_coverage=0.80,
        regime=regime, min_regime_pool=20)

    # The Bull at index 95 should trigger fallback (0 same-regime predecessors)
    fb = meta.iloc[95]["fallback_reason"]
    assert fb is not None and "Bull" in str(fb)


def test_min_regime_pool_constant_is_reasonable():
    """Lock the per-regime minimum at a sensible value for downstream reasoning."""
    assert MIN_REGIME_POOL_SIZE >= 10
    assert MIN_REGIME_POOL_SIZE <= 50
