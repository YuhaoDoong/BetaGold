"""Regression tests for v3.7.239 (and v3.7.232): expiry intrinsic force-close.

Covers BC / SP / STRADDLE / SHORT_VOL across the four expiry states:
- today < expiry (must NOT force-close; helper returns None)
- today == expiry with spot close available (force-close at intrinsic)
- today == expiry without spot close (helper returns None to defer)
- today > expiry with kline missing (force-close at intrinsic)

Includes one SHORT_VOL asymmetric-wings fixture to verify the plan contract + DEC-6
``max(call_wing, put_wing) - credit`` max_risk formula.
"""
from __future__ import annotations
import pandas as pd
import pytest

from core.strategies.options_exit import force_close_at_expiry


SIGNAL_DATE = pd.Timestamp("2026-04-15")
EXPIRY_STR = "260515"  # YYMMDD encoded in option code, parses to 2026-05-15


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def bc_legs():
    """Long call C$405 (intrinsic at 5/15 close $417.29 ≈ $12.29)."""
    return [("long_call", f"US.GLD{EXPIRY_STR}C405000", 405.0, 1)]


@pytest.fixture
def sp_legs():
    """Bear put credit spread: short P$445 / long P$425. At spot $417.29 both legs ITM."""
    return [
        ("short_put", f"US.GLD{EXPIRY_STR}P445000", 445.0, -1),
        ("long_put",  f"US.GLD{EXPIRY_STR}P425000", 425.0, 1),
    ]


@pytest.fixture
def straddle_legs():
    """ATM long straddle: long C$417 + long P$417 (near the actual close)."""
    return [
        ("long_call", f"US.GLD{EXPIRY_STR}C417000", 417.0, 1),
        ("long_put",  f"US.GLD{EXPIRY_STR}P417000", 417.0, 1),
    ]


@pytest.fixture
def symmetric_ic_legs():
    """Iron Condor with equal $5 wings on both sides."""
    return [
        ("short_call", f"US.GLD{EXPIRY_STR}C425000", 425.0, -1),
        ("long_call",  f"US.GLD{EXPIRY_STR}C430000", 430.0, 1),
        ("short_put",  f"US.GLD{EXPIRY_STR}P410000", 410.0, -1),
        ("long_put",   f"US.GLD{EXPIRY_STR}P405000", 405.0, 1),
    ]


@pytest.fixture
def asymmetric_ic_legs():
    """IC with $5 call wing, $10 put wing — the plan contract + DEC-6 boundary case."""
    return [
        ("short_call", f"US.GLD{EXPIRY_STR}C425000", 425.0, -1),
        ("long_call",  f"US.GLD{EXPIRY_STR}C430000", 430.0, 1),
        ("short_put",  f"US.GLD{EXPIRY_STR}P410000", 410.0, -1),
        ("long_put",   f"US.GLD{EXPIRY_STR}P400000", 400.0, 1),
    ]


# -- BC: 4 expiry states -------------------------------------------------------


def test_bc_today_before_expiry(bc_legs):
    """today < expiry → helper returns None, normal exit loop must run."""
    res = force_close_at_expiry(bc_legs, entry_value=22.0,
                                   today_dt=pd.Timestamp("2026-05-10"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="long_call")
    assert res is None


def test_bc_today_equals_expiry_close_known(bc_legs):
    """today == expiry, ETF spot CSV carries 5-15 close ($417.29) → force-close.

    v3.7.250 (review fix): when the exact expiry-day close is available in
    the ETF daily CSV, settle at intrinsic on the expiry date itself.
    """
    res = force_close_at_expiry(bc_legs, entry_value=22.0,
                                   today_dt=pd.Timestamp("2026-05-15"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="long_call")
    assert res is not None
    assert res.get("is_closed") is True
    assert abs(res["exit_value"] - 12.29) < 0.05  # intrinsic at $417.29 - $405


def test_bc_today_past_expiry_kline_missing(bc_legs):
    """today > expiry, kline_db missing the leg → force-close at intrinsic."""
    res = force_close_at_expiry(bc_legs, entry_value=22.0,
                                   today_dt=pd.Timestamp("2026-05-22"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="long_call")
    assert res is not None
    assert res["is_closed"]
    # Intrinsic at 5-15 close $417.29 for C$405 ≈ $12.29
    assert abs(res["exit_value"] - 12.29) < 0.05
    assert "expiry intrinsic" in res["exit_reason"]


def test_bc_unparseable_code_returns_none():
    """Garbage option code → helper returns None gracefully."""
    res = force_close_at_expiry(
        [("long_call", "NOT_A_VALID_CODE", 100.0, 1)],
        entry_value=1.0, today_dt=pd.Timestamp("2026-06-01"),
        signal_date=SIGNAL_DATE, strategy_kind="long_call")
    assert res is None


# -- SP: 4 expiry states -------------------------------------------------------


def test_sp_today_before_expiry(sp_legs):
    res = force_close_at_expiry(sp_legs, entry_value=7.0,
                                   today_dt=pd.Timestamp("2026-05-10"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="credit_spread",
                                   max_risk=13.0)
    assert res is None


def test_sp_today_equals_expiry(sp_legs):
    """v3.7.250: with exact 5-15 close available, settle on expiry day."""
    res = force_close_at_expiry(sp_legs, entry_value=7.0,
                                   today_dt=pd.Timestamp("2026-05-15"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="credit_spread",
                                   max_risk=13.0)
    assert res is not None and res["is_closed"]
    assert abs(res["exit_value"] - 20.0) < 0.05  # full spread width


def test_sp_today_past_expiry_kline_missing(sp_legs):
    """Both legs ITM ($417.29 < $425 < $445), debit-to-close = $20 (full wing)."""
    res = force_close_at_expiry(sp_legs, entry_value=7.0,
                                   today_dt=pd.Timestamp("2026-05-22"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="credit_spread",
                                   max_risk=13.0)
    assert res is not None and res["is_closed"]
    # short_put intrinsic = 445 - 417.29 = 27.71
    # long_put  intrinsic = 425 - 417.29 = 7.71
    # debit to close = 27.71 - 7.71 = 20.0 (spread width, max loss)
    assert abs(res["exit_value"] - 20.0) < 0.05
    # pnl = (entry - cur)/max_risk * 100 = (7 - 20)/13 * 100 = -100%
    assert abs(res["pnl_pct"] - (-100.0)) < 0.5


def test_sp_omitted_max_risk_uses_entry_value(sp_legs):
    """When caller forgets max_risk, helper falls back to entry_value as denominator."""
    res = force_close_at_expiry(sp_legs, entry_value=7.0,
                                   today_dt=pd.Timestamp("2026-05-22"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="credit_spread")
    assert res is not None and res["is_closed"]


# -- STRADDLE: 4 expiry states -------------------------------------------------


def test_straddle_today_before_expiry(straddle_legs):
    res = force_close_at_expiry(straddle_legs, entry_value=10.0,
                                   today_dt=pd.Timestamp("2026-05-10"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="long_vol")
    assert res is None


def test_straddle_today_equals_expiry(straddle_legs):
    """v3.7.250: exact 5-15 close available → settle on expiry day."""
    res = force_close_at_expiry(straddle_legs, entry_value=10.0,
                                   today_dt=pd.Timestamp("2026-05-15"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="long_vol")
    assert res is not None and res["is_closed"]
    assert abs(res["exit_value"] - 0.29) < 0.05


def test_straddle_today_past_expiry_kline_missing(straddle_legs):
    """At expiry, spot 417.29, K=417 for both legs:
    call_intrinsic = max(417.29 - 417, 0) = 0.29
    put_intrinsic  = max(417 - 417.29, 0) = 0.00
    cur_value      = 0.29 + 0.00 = 0.29 → near-total loss vs entry $10."""
    res = force_close_at_expiry(straddle_legs, entry_value=10.0,
                                   today_dt=pd.Timestamp("2026-05-22"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="long_vol")
    assert res is not None and res["is_closed"]
    assert abs(res["exit_value"] - 0.29) < 0.05
    # pnl_pct = (0.29/10 - 1)*100 ≈ -97.1%
    assert abs(res["pnl_pct"] - (-97.1)) < 0.5


def test_straddle_zero_entry_pnl_safe(straddle_legs):
    """Edge: entry_value ≈ 0 must not divide-by-zero."""
    res = force_close_at_expiry(straddle_legs, entry_value=0.0,
                                   today_dt=pd.Timestamp("2026-05-22"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="long_vol")
    assert res is not None and res["is_closed"]
    assert res["pnl_pct"] == 0.0


# -- SHORT_VOL (Iron Condor): 4 expiry states + asymmetric ---------------------


def test_short_vol_today_before_expiry(symmetric_ic_legs):
    res = force_close_at_expiry(symmetric_ic_legs, entry_value=1.50,
                                   today_dt=pd.Timestamp("2026-05-10"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="iron_condor")
    assert res is None


def test_short_vol_today_equals_expiry(symmetric_ic_legs):
    """v3.7.250: exact 5-15 close available → settle on expiry day."""
    res = force_close_at_expiry(symmetric_ic_legs, entry_value=1.50,
                                   today_dt=pd.Timestamp("2026-05-15"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="iron_condor")
    assert res is not None and res["is_closed"]


def test_short_vol_symmetric_pin_between_shorts(symmetric_ic_legs):
    """Spot $417.29 falls between short_put $410 and short_call $425 → max profit.
    All legs expire worthless, cur_value (debit to close) ≈ 0, P&L = +100% (credit kept)."""
    res = force_close_at_expiry(symmetric_ic_legs, entry_value=1.50,
                                   today_dt=pd.Timestamp("2026-05-22"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="iron_condor")
    assert res is not None and res["is_closed"]
    assert abs(res["exit_value"] - 0.0) < 0.05
    # max_risk = max(5, 5) - 1.50 = 3.50, pnl = (1.50 - 0)/3.50 * 100 ≈ +42.86%
    assert abs(res["pnl_pct"] - 42.86) < 0.5


def test_short_vol_asymmetric_max_risk_uses_wider_wing(asymmetric_ic_legs):
    """the plan contract + DEC-6: with $5 call wing and $10 put wing, max_risk = $10 - $1.50 = $8.50,
    NOT $5 - $1.50 = $3.50. Spot pin in profit zone → P&L positive but smaller %
    relative to the wider max_risk."""
    res = force_close_at_expiry(asymmetric_ic_legs, entry_value=1.50,
                                   today_dt=pd.Timestamp("2026-05-22"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="iron_condor")
    assert res is not None and res["is_closed"]
    # cur_value ≈ 0 (between shorts), max_risk_eff = max(5, 10) - 1.50 = 8.50
    # pnl = (1.50 - 0)/8.50 * 100 ≈ +17.65%, NOT +42.86%
    assert abs(res["pnl_pct"] - 17.65) < 0.5


def test_awaiting_expiry_close_when_etf_csv_missing_date():
    """v3.7.250 (review fix P1#2 + P3#1): when ETF CSV does NOT have the exact
    expiry date close, helper returns ``state='AWAITING_EXPIRY_CLOSE'`` rather
    than mis-marking against a stale earlier-day close."""
    # Construct legs with a far-future expiry that the real ETF CSV cannot
    # possibly carry (year 2099). Today >= expiry triggers force-close path;
    # exact close missing → AWAITING.
    legs = [("long_call", "US.GLD990515C400000", 400.0, 1)]
    res = force_close_at_expiry(
        legs, entry_value=10.0,
        today_dt=pd.Timestamp("2099-06-01"),
        signal_date=pd.Timestamp("2099-04-01"),
        strategy_kind="long_call")
    assert res is not None
    assert res.get("is_closed") is False
    assert res.get("state") == "AWAITING_EXPIRY_CLOSE"


def test_short_vol_codes_unparseable_skip():
    """Bad leg code in IC → helper returns None rather than crashing."""
    legs = [
        ("short_call", "GARBAGE_CODE_1", 425.0, -1),
        ("long_call",  "GARBAGE_CODE_2", 430.0, 1),
        ("short_put",  "GARBAGE_CODE_3", 410.0, -1),
        ("long_put",   "GARBAGE_CODE_4", 405.0, 1),
    ]
    res = force_close_at_expiry(legs, entry_value=1.5,
                                   today_dt=pd.Timestamp("2026-06-01"),
                                   signal_date=SIGNAL_DATE,
                                   strategy_kind="iron_condor")
    assert res is None
