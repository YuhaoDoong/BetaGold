"""Regression tests for v3.7.244 (AC-8): horizon-aware split-conformal scaler.

Covers:
- Maturity-lag invariant (no use of unmatured forward labels).
- Per-side asymmetric repair (upper shrinks when over-predicted; lower widens
  when under-predicted).
- target_coverage tracking on the eligible pool.
- Insufficient-sample fallback.
- Zero-residual edge.
- Pure function invariants (no I/O, deterministic).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from core.calibration import (
    apply_rolling_conformal_scaler,
    DEFAULT_HORIZON,
    DEFAULT_WINDOW,
    DEFAULT_TARGET_COVERAGE,
    MIN_POOL_SIZE,
)


def _bdates(n, start="2026-01-05"):
    return pd.bdate_range(start, periods=n)


# -- Maturity-lag invariant ----------------------------------------------------


def test_maturity_lag_excludes_unmatured_labels():
    """An actual at source date s is usable only if s + horizon < t.

    Construct a fixture where the actual at s = t - 3 is EXTREMELY misleading
    (would push delta toward an absurd value). If the scaler is correct, that
    actual is NOT in the pool for t (because s + 5 = t + 2 > t).
    """
    n = 80
    dates = _bdates(n)
    # Stable history: pred=+2, actual=+1 → upper residual = -1 (model
    # over-predicts by exactly 1).
    pred_u = pd.Series([2.0] * n, index=dates)
    pred_l = pd.Series([-2.0] * n, index=dates)
    actual_u = pd.Series([1.0] * n, index=dates)
    actual_l = pd.Series([-1.0] * n, index=dates)
    # Poison the next few rows (NOT yet matured by their own horizon at any t
    # within the pool window): set actual_upper at indices t-1, t-2, t-3 to a
    # huge value. These should NOT enter the pool for t.
    poison_t = 60
    for s in (poison_t - 1, poison_t - 2, poison_t - 3):
        actual_u.iat[s] = 999.0  # Would be massively over-covered if leaked

    out_u, out_l, meta = apply_rolling_conformal_scaler(
        dates, pred_u, pred_l, actual_u, actual_l,
        horizon=5, window=DEFAULT_WINDOW, target_coverage=0.80)

    # At t=poison_t, eligibility cutoff is s <= t - 6 = 54. Poisoned indices
    # 57/58/59 must NOT be in the pool.
    delta_u_t = meta.iloc[poison_t]["delta_upper"]
    # If poisoned values had leaked, delta_u would jump toward ~999.
    # With proper maturity gate, delta_u remains close to -1 (the clean residual).
    assert delta_u_t == pytest.approx(-1.0, abs=1e-6), (
        f"Maturity lag failed: delta_upper={delta_u_t}, expected ~-1.0; "
        f"poisoned actuals at t-1/-2/-3 likely leaked."
    )


def test_no_history_returns_raw():
    n = 5  # Far less than horizon + min_pool
    dates = _bdates(n)
    pred_u = pd.Series([2.0] * n, index=dates)
    pred_l = pd.Series([-2.0] * n, index=dates)
    actual_u = pd.Series([1.0] * n, index=dates)
    actual_l = pd.Series([-1.0] * n, index=dates)
    out_u, out_l, meta = apply_rolling_conformal_scaler(
        dates, pred_u, pred_l, actual_u, actual_l, horizon=5,
        window=DEFAULT_WINDOW, target_coverage=0.80)
    # All meta should mark fallback
    fallbacks = meta["fallback_reason"].dropna().unique()
    assert "no_history" in fallbacks or "insufficient_pool" in fallbacks
    # Raw bounds passed through
    np.testing.assert_array_equal(out_u.values, pred_u.values)
    np.testing.assert_array_equal(out_l.values, pred_l.values)


# -- Per-side asymmetric repair ------------------------------------------------


def test_upper_overpredicted_shrinks():
    """pred_upper = actual_upper + 2 (model over-predicts upper).

    delta_upper = quantile(actual - pred, 0.80) should be ≈ -2.0,
    i.e., shrink the upper bound by 2.
    """
    n = 200
    dates = _bdates(n)
    actual_u = pd.Series([1.0] * n, index=dates)
    pred_u = actual_u + 2.0  # always over-predicts upper
    actual_l = pd.Series([-1.0] * n, index=dates)
    pred_l = actual_l.copy()  # exactly on lower (residual = 0)

    out_u, out_l, meta = apply_rolling_conformal_scaler(
        dates, pred_u, pred_l, actual_u, actual_l,
        horizon=5, window=60, target_coverage=0.80)
    # Take a row that has full eligibility
    delta_u = meta.iloc[150]["delta_upper"]
    assert delta_u == pytest.approx(-2.0, abs=1e-6)
    assert out_u.iloc[150] == pytest.approx(pred_u.iloc[150] - 2.0)


def test_lower_underpredicted_widens():
    """pred_lower = actual_lower + 2 (model's lower is ABOVE actual lower,
    i.e., lower bound is not negative enough → under-predicts the downside).

    residual_l = pred_lower - actual_lower = +2.0 always.
    delta_lower = quantile(+2.0, 0.80) = +2.0.
    Calibrated lower = pred_lower - 2.0 → widens downward, what we want.
    """
    n = 200
    dates = _bdates(n)
    actual_u = pd.Series([1.0] * n, index=dates)
    pred_u = actual_u.copy()
    actual_l = pd.Series([-3.0] * n, index=dates)
    pred_l = actual_l + 2.0  # = -1.0; insufficient downside (under-predicts)

    out_u, out_l, meta = apply_rolling_conformal_scaler(
        dates, pred_u, pred_l, actual_u, actual_l,
        horizon=5, window=60, target_coverage=0.80)
    delta_l = meta.iloc[150]["delta_lower"]
    assert delta_l == pytest.approx(2.0, abs=1e-6)
    # Calibrated lower bound widened to match actual
    assert out_l.iloc[150] == pytest.approx(actual_l.iloc[150])


def test_simultaneously_asymmetric_repair():
    """Mirror of the GLD 2026-03 audit signature:
    upper over-predicted (residual = -2), lower also under-predicted (residual = +2)."""
    n = 200
    dates = _bdates(n)
    actual_u = pd.Series([1.0] * n, index=dates)
    pred_u = actual_u + 2.0
    actual_l = pd.Series([-3.0] * n, index=dates)
    pred_l = actual_l + 2.0
    out_u, out_l, meta = apply_rolling_conformal_scaler(
        dates, pred_u, pred_l, actual_u, actual_l,
        horizon=5, window=60, target_coverage=0.80)
    row = meta.iloc[150]
    assert row["delta_upper"] < 0, "upper should shrink"
    assert row["delta_lower"] > 0, "lower should widen"
    assert out_u.iloc[150] < pred_u.iloc[150]
    assert out_l.iloc[150] < pred_l.iloc[150]


# -- target_coverage tracking --------------------------------------------------


def test_target_coverage_tracked_on_pool():
    """Empirical coverage on the residual pool should match the requested target
    (by construction of the quantile)."""
    rng = np.random.default_rng(seed=42)
    n = 200
    dates = _bdates(n)
    # Random residuals
    actual_u = pd.Series(rng.normal(0.5, 1.0, n), index=dates)
    pred_u = pd.Series(rng.normal(2.0, 0.5, n), index=dates)
    actual_l = pd.Series(rng.normal(-0.5, 1.0, n), index=dates)
    pred_l = pd.Series(rng.normal(-2.0, 0.5, n), index=dates)

    for tgt in (0.50, 0.80, 0.95):
        _, _, meta = apply_rolling_conformal_scaler(
            dates, pred_u, pred_l, actual_u, actual_l,
            horizon=5, window=60, target_coverage=tgt)
        row = meta.iloc[150]
        # Realized coverage on the pool is, by quantile construction,
        # approximately the target.
        assert abs(row["realized_coverage_upper"] - tgt) < 0.1
        assert abs(row["realized_coverage_lower"] - tgt) < 0.1


# -- Zero-residual edge -------------------------------------------------------


def test_zero_residual_no_change():
    """When pred == actual perfectly, delta == 0 and calibrated == raw."""
    n = 200
    dates = _bdates(n)
    pred_u = pd.Series([2.0] * n, index=dates)
    pred_l = pd.Series([-2.0] * n, index=dates)
    actual_u = pred_u.copy()
    actual_l = pred_l.copy()
    out_u, out_l, meta = apply_rolling_conformal_scaler(
        dates, pred_u, pred_l, actual_u, actual_l,
        horizon=5, window=60, target_coverage=0.80)
    row = meta.iloc[150]
    assert row["delta_upper"] == pytest.approx(0.0, abs=1e-9)
    assert row["delta_lower"] == pytest.approx(0.0, abs=1e-9)
    assert out_u.iloc[150] == pred_u.iloc[150]
    assert out_l.iloc[150] == pred_l.iloc[150]


# -- Input validation ---------------------------------------------------------


def test_invalid_target_coverage_raises():
    n = 100
    dates = _bdates(n)
    s = pd.Series([1.0] * n, index=dates)
    for bad in (-0.1, 0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="target_coverage must be in"):
            apply_rolling_conformal_scaler(dates, s, -s, s, -s,
                                                 target_coverage=bad)


def test_invalid_horizon_raises():
    n = 100
    dates = _bdates(n)
    s = pd.Series([1.0] * n, index=dates)
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        apply_rolling_conformal_scaler(dates, s, -s, s, -s, horizon=0)


def test_window_below_min_pool_raises():
    n = 100
    dates = _bdates(n)
    s = pd.Series([1.0] * n, index=dates)
    with pytest.raises(ValueError, match="window .* must be >= min_pool"):
        apply_rolling_conformal_scaler(dates, s, -s, s, -s,
                                             window=10, min_pool=20)


# -- Purity -------------------------------------------------------------------


def test_scaler_no_filesystem_io(monkeypatch):
    import builtins
    real_open = builtins.open

    def blocked(*a, **k):
        raise RuntimeError("scaler tried to open a file (purity violation)")

    monkeypatch.setattr(builtins, "open", blocked)
    try:
        n = 100
        dates = _bdates(n)
        s = pd.Series([1.0] * n, index=dates)
        out_u, out_l, meta = apply_rolling_conformal_scaler(
            dates, s + 1, -s - 1, s, -s, target_coverage=0.80)
        assert len(meta) == n
    finally:
        monkeypatch.setattr(builtins, "open", real_open)


def test_scaler_is_deterministic():
    n = 150
    dates = _bdates(n)
    pred_u = pd.Series([2.0] * n, index=dates)
    pred_l = pd.Series([-2.0] * n, index=dates)
    actual_u = pd.Series([1.0] * n, index=dates)
    actual_l = pd.Series([-1.0] * n, index=dates)
    out1 = apply_rolling_conformal_scaler(dates, pred_u, pred_l,
                                                actual_u, actual_l)
    out2 = apply_rolling_conformal_scaler(dates, pred_u, pred_l,
                                                actual_u, actual_l)
    pd.testing.assert_series_equal(out1[0], out2[0])
    pd.testing.assert_series_equal(out1[1], out2[1])
    pd.testing.assert_frame_equal(out1[2], out2[2])
