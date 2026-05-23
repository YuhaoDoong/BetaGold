"""Regression test for v3.7.233: RegimeClassifier default has no forward leak.

Invariant: for any prefix of the feature series, the regime label at day t
must equal the regime label computed from the full series at day t, when
``min_hold_days=1`` is used.
"""
from __future__ import annotations
import pandas as pd
import pytest

from core.regime import RegimeClassifier


def test_default_min_hold_days_is_one():
    rc = RegimeClassifier()
    assert rc.min_hold_days == 1, (
        "RegimeClassifier default must be min_hold_days=1 to prevent "
        "forward-looking historical rewrites in production paths."
    )


def test_synthetic_no_leak_invariant():
    """Truncating the feature tail must not change historical regime labels."""
    n = 80
    feat = pd.DataFrame({
        "ret_60d": [0.01 + 0.001 * i for i in range(n)],
        "fed_funds_rate_change_60d": [0.0] * n,
    }, index=pd.bdate_range("2025-01-01", periods=n))

    rc = RegimeClassifier()
    full = rc.classify(feat)["regime"]
    truncated = rc.classify(feat.iloc[:-20])["regime"]

    overlap = full.loc[: truncated.index[-1]]
    assert overlap.equals(truncated), (
        "Regime at day t depends on data after day t. Forward-leak detected."
    )
