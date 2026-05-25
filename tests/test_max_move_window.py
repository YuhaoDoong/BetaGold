"""Regression tests for v3.7.241: max_move_{h}d off-by-one fix.

The legacy implementation `series.rolling(h).max().shift(-(h+1))` produced
``max(series[i+2..i+h+1])`` at signal index ``i``. The correct window for the
plan's semantics ("h-day hold starting at next-day open") is
``max(series[i+1..i+h])``. We verify the new helper against hand-calculated
values on a deterministic synthetic series.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from scripts.backtest.framework import forward_window_extreme


@pytest.fixture
def synthetic_high():
    """High prices = i + 100 so windowed max is trivially the last index value."""
    idx = pd.bdate_range("2026-01-02", periods=30)
    return pd.Series([100 + i for i in range(len(idx))], index=idx, dtype=float)


@pytest.fixture
def synthetic_low():
    """Low prices = 200 - i so windowed min is trivially the last index value."""
    idx = pd.bdate_range("2026-01-02", periods=30)
    return pd.Series([200 - i for i in range(len(idx))], index=idx, dtype=float)


def test_forward_max_window_known_values(synthetic_high):
    """For h=5, signal i=0: max(High[1..5]) = max(101..105) = 105."""
    out = forward_window_extreme(synthetic_high, window_h=5, anchor_offset=1, op="max")
    assert out.iloc[0] == 105.0
    assert out.iloc[1] == 106.0   # max(102..106)
    assert out.iloc[10] == 115.0  # max(111..115)


def test_forward_min_window_known_values(synthetic_low):
    """For h=5, signal i=0: min(Low[1..5]) = min(199..195) = 195."""
    out = forward_window_extreme(synthetic_low, window_h=5, anchor_offset=1, op="min")
    assert out.iloc[0] == 195.0
    assert out.iloc[1] == 194.0   # min(198..194)
    assert out.iloc[10] == 185.0  # min(189..185)


def test_insufficient_future_window_returns_nan(synthetic_high):
    """The last h indices cannot form a full forward window → NaN."""
    h = 5
    out = forward_window_extreme(synthetic_high, window_h=h, anchor_offset=1, op="max")
    # Last index where the window fits: i + 1 + h - 1 <= n - 1, so i <= n - h - 1
    n = len(synthetic_high)
    cutoff = n - h - 1
    assert not np.isnan(out.iloc[cutoff])
    assert np.isnan(out.iloc[cutoff + 1])
    assert np.isnan(out.iloc[-1])


def test_legacy_offbyone_vs_corrected(synthetic_high):
    """Demonstrate the magnitude of the legacy off-by-one bug."""
    h = 5
    correct = forward_window_extreme(synthetic_high, window_h=h, anchor_offset=1, op="max")
    legacy = synthetic_high.rolling(h).max().shift(-(h + 1))
    # On the stable interior (away from edges), legacy reads [i+2..i+h+1]
    # whereas correct reads [i+1..i+h]. For a strictly increasing series, legacy
    # is exactly 1 unit higher than correct.
    interior = slice(2, 20)
    diff = (legacy.iloc[interior] - correct.iloc[interior]).dropna()
    assert (diff == 1.0).all(), (
        "Legacy implementation should differ from corrected by exactly +1.0 unit "
        "on a strictly-increasing synthetic series (it includes one extra future "
        "day and excludes the entry-day high)."
    )


def test_op_invalid_raises(synthetic_high):
    with pytest.raises(ValueError, match="op must be 'max' or 'min'"):
        forward_window_extreme(synthetic_high, window_h=5, op="median")


def test_nan_in_window_skipped(synthetic_high):
    """If the future window is entirely NaN the output is NaN; partial NaN uses nanmax."""
    series = synthetic_high.copy()
    series.iloc[5] = np.nan
    out = forward_window_extreme(series, window_h=5, anchor_offset=1, op="max")
    # At i=0, window is series[1..5] = [101,102,103,104,NaN]; nanmax = 104
    assert out.iloc[0] == 104.0


def test_anchor_offset_zero_means_inclusive_today(synthetic_high):
    """anchor_offset=0 makes the window include the signal day itself."""
    out = forward_window_extreme(synthetic_high, window_h=5, anchor_offset=0, op="max")
    # i=0: max(High[0..4]) = max(100..104) = 104
    assert out.iloc[0] == 104.0
