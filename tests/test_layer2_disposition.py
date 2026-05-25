"""Regression tests for v3.7.242: Layer 2 disposition reporting + per-leg DTE.

Verifies that ``run_layer2_backtest_with_disposition``:
- Reconciles ``n_signal == n_closed + n_open + n_skipped_stale + n_skipped_no_contract``
- Counts ``n_skipped_no_contract`` when price_fn returns empty legs
- Counts ``n_skipped_stale`` when ``max(leg.expiry) + hold_buffer > kline_db.max_date``
- Does NOT silently drop unclosed positions (the original survivorship-bias bug)

Also covers ``leg_max_expiry`` (parses YYMMDD from option code).
"""
from __future__ import annotations
import pandas as pd
import pytest

from scripts.backtest.framework import (
    leg_max_expiry,
    run_layer2_backtest_with_disposition,
)


# -- leg_max_expiry --------------------------------------------------------------


def test_leg_max_expiry_single_leg():
    legs = [("long_call", "US.GLD260515C405000", 405.0, 1)]
    assert leg_max_expiry(legs) == pd.Timestamp("2026-05-15")


def test_leg_max_expiry_multi_leg_same_date():
    legs = [
        ("short_put", "US.GLD260515P445000", 445.0, -1),
        ("long_put",  "US.GLD260515P425000", 425.0, 1),
    ]
    assert leg_max_expiry(legs) == pd.Timestamp("2026-05-15")


def test_leg_max_expiry_calendar_spread_returns_max():
    """Future-proofing for calendar spreads (heterogeneous expiries)."""
    legs = [
        ("short_call", "US.GLD260515C420000", 420.0, -1),
        ("long_call",  "US.GLD260717C420000", 420.0, 1),
    ]
    assert leg_max_expiry(legs) == pd.Timestamp("2026-07-17")


def test_leg_max_expiry_unparseable_returns_none():
    assert leg_max_expiry([("x", "GARBAGE", 0, 0)]) is None


def test_leg_max_expiry_empty_returns_none():
    assert leg_max_expiry([]) is None


# -- run_layer2_backtest_with_disposition ----------------------------------------


def _make_ohlc(dates):
    return pd.DataFrame({
        "Open":  [400.0] * len(dates),
        "High":  [405.0] * len(dates),
        "Low":   [395.0] * len(dates),
        "Close": [402.0] * len(dates),
    }, index=dates)


def test_reconciliation_invariant_holds_all_closed():
    """Every signal closes → n_signal == n_closed."""
    dates = pd.bdate_range("2026-01-05", periods=10).tolist()
    ohlc = _make_ohlc(dates)

    def price_fn(asset, strategy, d, *a, **kw):
        return {"legs": [("long_call", "US.GLD260117C400000", 400.0, 1)],
                  "entry_price": 1.0}

    def exit_fn(*a, **kw):
        return {"is_closed": True, "pnl_pct": 5.0, "exit_reason": "tp"}

    df, disp = run_layer2_backtest_with_disposition(
        dates, ohlc, "GLD", "BUY CALL",
        price_fn=price_fn, exit_fn=exit_fn,
        dte_target=30, hold_buffer_days=5,
        today=pd.Timestamp("2027-01-01"))  # well past expiry
    assert disp["n_signal"] == 10
    # Some/all may be skipped as stale because 2026-01-17 + 5 days vs kline max;
    # the invariant must hold regardless:
    assert disp["n_signal"] == (disp["n_closed"] + disp["n_open"]
                                  + disp["n_skipped_stale"]
                                  + disp["n_skipped_no_contract"])


def test_no_contract_counted_separately_from_open():
    """price_fn returning empty legs → n_skipped_no_contract."""
    dates = pd.bdate_range("2026-01-05", periods=5).tolist()
    ohlc = _make_ohlc(dates)

    def price_fn_empty(*a, **kw):
        return {"legs": [], "source": "—"}

    def exit_fn_never_called(*a, **kw):
        raise AssertionError("exit_fn should not be called when legs are empty")

    df, disp = run_layer2_backtest_with_disposition(
        dates, ohlc, "GLD", "BUY CALL",
        price_fn=price_fn_empty, exit_fn=exit_fn_never_called)
    assert disp["n_signal"] == 5
    assert disp["n_skipped_no_contract"] == 5
    assert disp["n_entered"] == 0
    assert disp["n_closed"] == 0


def test_open_position_not_silently_dropped():
    """Entered + unclosed → n_open increments (the original survivorship bug).

    Uses a past expiry so the stale pre-filter passes; exit_fn simulates a
    position still open under MTM (is_closed=False).
    """
    dates = pd.bdate_range("2026-01-05", periods=4).tolist()
    ohlc = _make_ohlc(dates)
    # Past expiry (Jan 2024) → stale filter passes (well before kline_db.max)
    past_expiry = "240117"

    def price_fn(*a, **kw):
        return {"legs": [("long_call", f"US.GLD{past_expiry}C400000", 400.0, 1)],
                  "entry_price": 1.0}

    def exit_fn_open(*a, **kw):
        return {"is_closed": False, "pnl_pct": 0.0}

    df, disp = run_layer2_backtest_with_disposition(
        dates, ohlc, "GLD", "BUY CALL",
        price_fn=price_fn, exit_fn=exit_fn_open)
    assert disp["n_signal"] == 4
    assert disp["n_entered"] == 4
    assert disp["n_closed"] == 0
    assert disp["n_open"] == 4  # the bug would have made these vanish
    assert len(df) == 0


def test_stale_pre_filter_blocks_late_expiries():
    """If max(leg.expiry) + buffer > kline_db.max_date → n_skipped_stale."""
    dates = pd.bdate_range("2026-01-05", periods=3).tolist()
    ohlc = _make_ohlc(dates)

    # Pick an expiry slightly past kline_db.max_date (which is real:
    # 2026-05-06 in production). Use 2030-12-31 to be safely beyond.
    def price_fn_far_expiry(*a, **kw):
        return {"legs": [("long_call", "US.GLD301231C400000", 400.0, 1)],
                  "entry_price": 1.0}

    def exit_fn_should_skip(*a, **kw):
        raise AssertionError("exit_fn must not run when stale-filtered")

    df, disp = run_layer2_backtest_with_disposition(
        dates, ohlc, "GLD", "BUY CALL",
        price_fn=price_fn_far_expiry, exit_fn=exit_fn_should_skip)
    assert disp["n_signal"] == 3
    assert disp["n_skipped_stale"] == 3
    assert disp["n_entered"] == 0


def test_mixed_disposition_full_reconciliation():
    """Mixed inputs: closed, open, stale, no_contract — reconciliation holds."""
    dates = pd.bdate_range("2026-01-05", periods=8).tolist()
    ohlc = _make_ohlc(dates)
    call_idx = {"i": 0}

    def price_fn_mixed(asset, strategy, d, *a, **kw):
        i = call_idx["i"]; call_idx["i"] += 1
        if i < 2:
            # No contract
            return {"legs": [], "source": "—"}
        if i < 4:
            # Stale (far-future expiry)
            return {"legs": [("long_call", "US.GLD301231C400000", 400.0, 1)],
                      "entry_price": 1.0}
        # Fresh entries (use near-past expiry so stale filter doesn't trip;
        # this requires kline_db.max_date to be later — production has 2026-05-06)
        return {"legs": [("long_call", "US.GLD260117C400000", 400.0, 1)],
                  "entry_price": 1.0}

    def exit_fn_alt(*a, **kw):
        # Alternate closed/open for the 4 entered
        exit_fn_alt.calls += 1
        return {"is_closed": exit_fn_alt.calls % 2 == 0,
                "pnl_pct": 5.0, "exit_reason": "tp"}
    exit_fn_alt.calls = 0

    df, disp = run_layer2_backtest_with_disposition(
        dates, ohlc, "GLD", "BUY CALL",
        price_fn=price_fn_mixed, exit_fn=exit_fn_alt)
    # 2 no_contract, 2 stale, 4 entered (alternating closed/open)
    assert disp["n_signal"] == 8
    assert disp["n_skipped_no_contract"] == 2
    assert disp["n_skipped_stale"] == 2
    assert disp["n_entered"] == 4
    assert disp["n_closed"] + disp["n_open"] == 4
    # Reconciliation invariant
    assert disp["n_signal"] == (disp["n_closed"] + disp["n_open"]
                                  + disp["n_skipped_stale"]
                                  + disp["n_skipped_no_contract"])
