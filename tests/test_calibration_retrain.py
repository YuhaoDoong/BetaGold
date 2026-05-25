"""Regression tests for v3.7.245: calibration-gated retrain trigger (AC-9)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from core.calibration import (
    evaluate_retrain_trigger,
    DEFAULT_RETRAIN_RATIO_QUEUE,
    DEFAULT_RETRAIN_RATIO_IMMEDIATE,
    DEFAULT_RETRAIN_HYSTERESIS_DAYS,
    DEFAULT_RETRAIN_COOLDOWN_DAYS,
)


def _build_meta(deltas_upper, deltas_lower, dates):
    """Construct a minimal meta_df with the columns the trigger reads."""
    n = len(dates)
    return pd.DataFrame({
        "n_eligible": [60] * n,
        "delta_upper": deltas_upper,
        "delta_lower": deltas_lower,
        "realized_coverage_upper": [0.80] * n,
        "realized_coverage_lower": [0.80] * n,
        "fallback_reason": [None] * n,
    }, index=dates)


def _build_aw(values, dates):
    return pd.Series(values, index=dates, dtype=float)


# -- Hysteresis (5-day requirement) -------------------------------------------


def test_single_day_breach_does_not_queue():
    """A single day above the queue threshold must NOT trigger; hysteresis
    requires 5 consecutive breaches."""
    dates = pd.bdate_range("2026-04-01", periods=10)
    # actual_width=1, |delta_upper|+|delta_lower|=4 on the last day only
    deltas_u = [0.5] * 9 + [3.0]
    deltas_l = [0.5] * 9 + [3.0]
    aw = [1.0] * 10
    meta = _build_meta(deltas_u, deltas_l, dates)
    res = evaluate_retrain_trigger(meta, _build_aw(aw, dates),
                                          today=dates[-1])
    # The single-day ratio at the end is (3+3 + 1)/1 = 7 > 4.0 → immediate.
    # That's the immediate path firing on a real outlier — test that branch
    # below; here we move the spike one day BACK so the latest is calm.
    deltas_u2 = [0.5] * 8 + [3.0] + [0.5]
    deltas_l2 = [0.5] * 8 + [3.0] + [0.5]
    meta2 = _build_meta(deltas_u2, deltas_l2, dates)
    res2 = evaluate_retrain_trigger(meta2, _build_aw(aw, dates),
                                            today=dates[-1])
    assert res2["outcome"] == "no_action"
    assert res2["consecutive_days"] <= 1


def test_five_consecutive_breaches_queues():
    """5 consecutive days above the queue threshold (2.5) but below immediate
    (4.0) → outcome 'queued'."""
    dates = pd.bdate_range("2026-04-01", periods=20)
    # Ratio = (|du| + |dl| + aw) / aw. Want ratio ~ 3.0 for last 5 days:
    # aw=1, |du|+|dl|=2.0 → ratio=3.0. Earlier days calm (ratio=1.5).
    deltas_u = [0.25] * 15 + [1.0] * 5
    deltas_l = [0.25] * 15 + [1.0] * 5
    aw = [1.0] * 20
    meta = _build_meta(deltas_u, deltas_l, dates)
    res = evaluate_retrain_trigger(meta, _build_aw(aw, dates),
                                          today=dates[-1])
    assert res["outcome"] == "queued"
    assert res["consecutive_days"] >= DEFAULT_RETRAIN_HYSTERESIS_DAYS
    assert res["ratio_value"] > 1.0


def test_four_consecutive_breaches_does_not_queue():
    """Hysteresis requires ≥5; exactly 4 days above must NOT queue."""
    dates = pd.bdate_range("2026-04-01", periods=20)
    deltas_u = [0.25] * 16 + [1.0] * 4
    deltas_l = [0.25] * 16 + [1.0] * 4
    aw = [1.0] * 20
    meta = _build_meta(deltas_u, deltas_l, dates)
    res = evaluate_retrain_trigger(meta, _build_aw(aw, dates),
                                          today=dates[-1])
    assert res["outcome"] == "no_action"
    assert res["consecutive_days"] == 4


# -- Immediate threshold (>4.0 single day) ------------------------------------


def test_single_day_above_immediate_triggers():
    """Latest day's ratio above 4.0 → outcome 'immediate' even if prior days
    were calm (no hysteresis required for the catastrophic branch)."""
    dates = pd.bdate_range("2026-04-01", periods=20)
    # Latest: aw=1, |du|+|dl|=5 → ratio=6.0 > 4.0
    deltas_u = [0.25] * 19 + [2.5]
    deltas_l = [0.25] * 19 + [2.5]
    aw = [1.0] * 20
    meta = _build_meta(deltas_u, deltas_l, dates)
    res = evaluate_retrain_trigger(meta, _build_aw(aw, dates),
                                          today=dates[-1])
    assert res["outcome"] == "immediate"


# -- Cooldown gate ------------------------------------------------------------


def test_cooldown_blocks_retrigger_within_window():
    """Inside the 7-trading-day cooldown after a retrain, the trigger must
    suppress regardless of how bad the ratios get."""
    dates = pd.bdate_range("2026-04-01", periods=20)
    deltas_u = [3.0] * 20  # always breach
    deltas_l = [3.0] * 20
    aw = [1.0] * 20
    meta = _build_meta(deltas_u, deltas_l, dates)
    # last_retrain 2 trading days before today
    last = dates[-3]
    res = evaluate_retrain_trigger(meta, _build_aw(aw, dates),
                                          today=dates[-1], last_retrain_at=last)
    assert res["outcome"] == "suppressed_cooldown"
    assert res["cooldown_until"] is not None


def test_after_cooldown_window_trigger_returns():
    """Beyond 7 trading days post-retrain, the cooldown gate releases."""
    dates = pd.bdate_range("2026-04-01", periods=30)
    deltas_u = [3.0] * 30
    deltas_l = [3.0] * 30
    aw = [1.0] * 30
    meta = _build_meta(deltas_u, deltas_l, dates)
    # Set last_retrain WELL before the cooldown window
    last = dates[0]
    res = evaluate_retrain_trigger(meta, _build_aw(aw, dates),
                                          today=dates[-1], last_retrain_at=last)
    # We're outside cooldown → expect immediate (ratio massively above 4)
    assert res["outcome"] in ("immediate", "queued")


# -- Zero-width guard ---------------------------------------------------------


def test_zero_width_days_excluded_from_ratio():
    """Days with |actual_width| < zero_width_floor are excluded; counter tracks them."""
    dates = pd.bdate_range("2026-04-01", periods=20)
    deltas_u = [1.0] * 20
    deltas_l = [1.0] * 20
    # First 5 days: actual_width = 0.05 (below floor 0.10) → excluded
    aw = [0.05] * 5 + [1.0] * 15
    meta = _build_meta(deltas_u, deltas_l, dates)
    res = evaluate_retrain_trigger(meta, _build_aw(aw, dates),
                                          today=dates[-1])
    assert res["zero_width_excluded_count"] == 5


def test_all_zero_widths_returns_no_action():
    """If every day has near-zero width, ratio cannot be computed → no_action."""
    dates = pd.bdate_range("2026-04-01", periods=10)
    deltas_u = [1.0] * 10
    deltas_l = [1.0] * 10
    aw = [0.0] * 10
    meta = _build_meta(deltas_u, deltas_l, dates)
    res = evaluate_retrain_trigger(meta, _build_aw(aw, dates),
                                          today=dates[-1])
    assert res["outcome"] == "no_action"
    assert res["zero_width_excluded_count"] == 10


# -- Purity & determinism -----------------------------------------------------


def test_trigger_is_pure_no_io(monkeypatch):
    import builtins
    real_open = builtins.open

    def blocked(*a, **k):
        raise RuntimeError("retrain trigger tried to open a file")

    monkeypatch.setattr(builtins, "open", blocked)
    try:
        dates = pd.bdate_range("2026-04-01", periods=10)
        deltas = [0.5] * 10
        meta = _build_meta(deltas, deltas, dates)
        res = evaluate_retrain_trigger(meta, _build_aw([1.0] * 10, dates),
                                              today=dates[-1])
        assert res["outcome"] in {"no_action", "queued", "immediate",
                                    "suppressed_cooldown"}
    finally:
        monkeypatch.setattr(builtins, "open", real_open)


def test_trigger_is_deterministic():
    dates = pd.bdate_range("2026-04-01", periods=20)
    meta = _build_meta([1.0] * 20, [1.0] * 20, dates)
    aw = _build_aw([1.0] * 20, dates)
    r1 = evaluate_retrain_trigger(meta, aw, today=dates[-1])
    r2 = evaluate_retrain_trigger(meta, aw, today=dates[-1])
    assert r1 == r2


# -- Empty meta ---------------------------------------------------------------


def test_empty_meta_no_action():
    dates = pd.DatetimeIndex([])
    meta = _build_meta([], [], dates)
    aw = _build_aw([], dates)
    res = evaluate_retrain_trigger(meta, aw, today=pd.Timestamp("2026-05-25"))
    assert res["outcome"] == "no_action"
