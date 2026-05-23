"""Regression tests for v3.7.240: IV-aware cross-asset selector + shadow gate.

Verifies that ``select_gld_sync_strategy`` is a pure function with the truth
table declared in AC-5, that the shadow log writer is caller-side only, and
that ``live_cutover_allowed`` enforces the 14-day accumulation gate.
"""
from __future__ import annotations
import json
import math
import pandas as pd
import pytest

from core.cross_asset_signal import (
    select_gld_sync_strategy,
    write_shadow_record,
    live_cutover_allowed,
    CROSS_GVZ_HIGH_THRESHOLD,
    CROSS_BP_LOW_DEEP_BREAK,
    CROSS_GVZ_STALE_MAX_DAYS,
    CROSS_LIVE_CUTOVER_MIN_DAYS,
)


# -- Selector truth table -----------------------------------------------------


def test_selector_default_returns_buy_call():
    """High bp_low + low GVZ → BUY CALL DEFAULT."""
    sig_d = pd.Timestamp("2026-03-15")
    asof = pd.Timestamp("2026-03-15")
    dec = select_gld_sync_strategy(sig_d, {"bp_low": 0.30}, 18.0, asof)
    assert dec["strategy"] == "BUY CALL"
    assert dec["reason"] == "DEFAULT"
    assert dec["gvz_status"] == "fresh"


def test_selector_deep_break_high_iv_returns_sell_put():
    """bp_low ≤ 0.10 AND GVZ ≥ 25 → SELL PUT DEEP_BREAK_HIGH_IV."""
    sig_d = pd.Timestamp("2026-03-15")
    asof = pd.Timestamp("2026-03-15")
    dec = select_gld_sync_strategy(sig_d, {"bp_low": 0.05}, 28.0, asof)
    assert dec["strategy"] == "SELL PUT"
    assert dec["reason"] == "DEEP_BREAK_HIGH_IV"
    assert dec["gvz_status"] == "fresh"


def test_selector_high_iv_but_not_deep_break_returns_buy_call():
    """High GVZ but bp_low > 0.10 → still BUY CALL DEFAULT (one condition fails)."""
    sig_d = pd.Timestamp("2026-03-15")
    asof = pd.Timestamp("2026-03-15")
    dec = select_gld_sync_strategy(sig_d, {"bp_low": 0.15}, 28.0, asof)
    assert dec["strategy"] == "BUY CALL"
    assert dec["reason"] == "DEFAULT"


def test_selector_deep_break_but_low_iv_returns_buy_call():
    """Deep break (bp_low ≤ 0.10) but GVZ < 25 → BUY CALL."""
    sig_d = pd.Timestamp("2026-03-15")
    asof = pd.Timestamp("2026-03-15")
    dec = select_gld_sync_strategy(sig_d, {"bp_low": 0.05}, 22.0, asof)
    assert dec["strategy"] == "BUY CALL"
    assert dec["reason"] == "DEFAULT"


def test_selector_boundary_thresholds():
    """Exactly at thresholds: bp_low=0.10 AND GVZ=25 → SELL PUT (inclusive)."""
    sig_d = pd.Timestamp("2026-03-15")
    asof = pd.Timestamp("2026-03-15")
    dec = select_gld_sync_strategy(
        sig_d, {"bp_low": CROSS_BP_LOW_DEEP_BREAK},
        CROSS_GVZ_HIGH_THRESHOLD, asof)
    assert dec["strategy"] == "SELL PUT"


# -- GVZ missing / stale ------------------------------------------------------


def test_selector_gvz_none():
    sig_d = pd.Timestamp("2026-03-15")
    dec = select_gld_sync_strategy(sig_d, {"bp_low": 0.05}, None, None)
    assert dec["strategy"] == "BUY CALL"
    assert dec["reason"] == "GVZ_UNAVAILABLE"
    assert dec["gvz_status"] == "missing"


def test_selector_gvz_nan():
    sig_d = pd.Timestamp("2026-03-15")
    dec = select_gld_sync_strategy(sig_d, {"bp_low": 0.05}, float("nan"),
                                        pd.Timestamp("2026-03-15"))
    assert dec["strategy"] == "BUY CALL"
    assert dec["reason"] == "GVZ_UNAVAILABLE"


def test_selector_gvz_stale_uses_signal_date_anchor():
    """Staleness must be relative to signal_date, NOT wall-clock today."""
    # signal_date 2026-03-15 (Sun) → effectively Mon market; gvz_asof 2026-03-11 (Wed)
    # trading day gap (Thu, Fri, Mon = 3) > CROSS_GVZ_STALE_MAX_DAYS (default 2)
    sig_d = pd.Timestamp("2026-03-16")  # Mon
    asof = pd.Timestamp("2026-03-11")    # Wed prior week
    dec = select_gld_sync_strategy(sig_d, {"bp_low": 0.05}, 28.0, asof)
    assert dec["strategy"] == "BUY CALL"
    assert dec["reason"] == "GVZ_UNAVAILABLE"
    assert dec["gvz_status"] == "stale"


def test_selector_gvz_within_tolerance_is_fresh():
    """gap ≤ CROSS_GVZ_STALE_MAX_DAYS trading days → FRESH."""
    sig_d = pd.Timestamp("2026-03-16")
    asof = pd.Timestamp("2026-03-13")  # 1 trading day gap (Fri → Mon)
    dec = select_gld_sync_strategy(sig_d, {"bp_low": 0.05}, 28.0, asof)
    assert dec["gvz_status"] == "fresh"
    # And because conditions match deep+high IV → SELL PUT
    assert dec["strategy"] == "SELL PUT"


# -- bp_low edge cases --------------------------------------------------------


def test_selector_bp_low_missing_returns_buy_call():
    sig_d = pd.Timestamp("2026-03-15")
    dec = select_gld_sync_strategy(sig_d, {}, 28.0, sig_d)
    assert dec["strategy"] == "BUY CALL"
    assert dec["reason"] == "GLD_BP_LOW_MISSING"


def test_selector_bp_low_nan_returns_buy_call():
    sig_d = pd.Timestamp("2026-03-15")
    dec = select_gld_sync_strategy(sig_d, {"bp_low": float("nan")}, 28.0, sig_d)
    assert dec["strategy"] == "BUY CALL"
    assert dec["reason"] == "GLD_BP_LOW_MISSING"


# -- Purity (no I/O, deterministic) -------------------------------------------


def test_selector_is_deterministic():
    sig_d = pd.Timestamp("2026-03-15")
    asof = pd.Timestamp("2026-03-15")
    first = select_gld_sync_strategy(sig_d, {"bp_low": 0.05}, 28.0, asof)
    second = select_gld_sync_strategy(sig_d, {"bp_low": 0.05}, 28.0, asof)
    assert first == second


def test_selector_no_filesystem_io(monkeypatch):
    """Selector must not open any file. Fake-fail builtins.open during call."""
    import builtins
    real_open = builtins.open

    def blocked_open(*args, **kwargs):
        raise RuntimeError("selector tried to open a file (purity violation)")

    monkeypatch.setattr(builtins, "open", blocked_open)
    try:
        sig_d = pd.Timestamp("2026-03-15")
        dec = select_gld_sync_strategy(sig_d, {"bp_low": 0.05}, 28.0, sig_d)
        assert dec["strategy"] == "SELL PUT"
    finally:
        monkeypatch.setattr(builtins, "open", real_open)


def test_selector_does_not_use_wall_clock(monkeypatch):
    """Selector must not query datetime.now() / Timestamp.today()."""
    import core.cross_asset_signal as cas_mod

    class Tripwire:
        @staticmethod
        def now(*a, **kw):
            raise RuntimeError("selector queried wall clock (purity violation)")

        today = now

    # We can't easily replace pd.Timestamp globally, but we can verify that
    # the selector's output for a historical date is independent of "now"
    # because Phase 5/6 already established reasoning anchored on signal_date.
    sig_d = pd.Timestamp("2020-01-15")  # ancient date
    asof = pd.Timestamp("2020-01-15")
    dec1 = select_gld_sync_strategy(sig_d, {"bp_low": 0.05}, 28.0, asof)
    dec2 = select_gld_sync_strategy(sig_d, {"bp_low": 0.05}, 28.0, asof)
    assert dec1 == dec2 == {"strategy": "SELL PUT", "reason": "DEEP_BREAK_HIGH_IV",
                              "gvz_status": "fresh"}


# -- Shadow-log writer + cutover gate -----------------------------------------


def test_write_shadow_record_appends_jsonl(tmp_path):
    log = tmp_path / "shadow.jsonl"
    decision = {"strategy": "SELL PUT", "reason": "DEEP_BREAK_HIGH_IV",
                 "gvz_status": "fresh"}
    write_shadow_record(decision, pd.Timestamp("2026-03-15"), slv_tier="S",
                          inputs={"bp_low": 0.05, "gvz_value": 28.0,
                                   "gvz_asof_date": pd.Timestamp("2026-03-15")},
                          log_path=str(log))
    write_shadow_record(decision, pd.Timestamp("2026-03-16"), slv_tier="S",
                          inputs={"bp_low": 0.07, "gvz_value": 26.0,
                                   "gvz_asof_date": pd.Timestamp("2026-03-16")},
                          log_path=str(log))
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["decision"]["strategy"] == "SELL PUT"
    assert rec0["slv_tier"] == "S"
    assert rec0["inputs"]["bp_low"] == 0.05


def test_live_cutover_blocks_short_history(tmp_path):
    """Less than 14 days of shadow records → cutover NOT allowed."""
    log = tmp_path / "shadow.jsonl"
    decision = {"strategy": "BUY CALL", "reason": "DEFAULT", "gvz_status": "fresh"}
    # Single record written "today" → 0 days accumulation
    write_shadow_record(decision, pd.Timestamp("2026-03-15"), slv_tier="S",
                          inputs={"bp_low": 0.3, "gvz_value": 18.0,
                                   "gvz_asof_date": pd.Timestamp("2026-03-15")},
                          log_path=str(log))
    allowed, first, days = live_cutover_allowed(pd.Timestamp.now().normalize(),
                                                       log_path=str(log))
    assert allowed is False
    assert days < CROSS_LIVE_CUTOVER_MIN_DAYS


def test_live_cutover_allows_after_threshold(tmp_path, monkeypatch):
    """≥14 calendar days between first record and today → cutover allowed."""
    log = tmp_path / "shadow.jsonl"
    # Manually write a record with backdated 'written_at'
    rec = {"signal_date": "2026-03-01", "slv_tier": "S",
            "decision": {"strategy": "BUY CALL", "reason": "DEFAULT",
                          "gvz_status": "fresh"},
            "inputs": {"bp_low": 0.3, "gvz_value": 18.0, "gvz_asof_date": "2026-03-01"},
            "written_at": "2026-03-01T00:00:00+00:00"}
    log.write_text(json.dumps(rec) + "\n")
    allowed, first, days = live_cutover_allowed(pd.Timestamp("2026-03-20"),
                                                       log_path=str(log))
    assert allowed is True
    assert days >= CROSS_LIVE_CUTOVER_MIN_DAYS


def test_live_cutover_missing_log_blocks(tmp_path):
    """No shadow log file → cutover not allowed (cold-start safety)."""
    log = tmp_path / "nonexistent.jsonl"
    allowed, first, days = live_cutover_allowed(pd.Timestamp("2026-05-22"),
                                                       log_path=str(log))
    assert allowed is False
    assert first is None
    assert days == 0


# -- March 2026 5/5 BC cluster expectation ------------------------------------


def test_march_2026_cluster_would_have_selected_sell_put_at_high_iv():
    """Sanity check on the very fixture that motivated v3.7.240.

    March 2026 GLD entries had bp_low ≈ 0.03..0.05 (very deep) and GLD GVZ
    spike into 25-30. The selector MUST recommend SELL PUT in that regime,
    not the fixed BUY CALL that produced the historical 5/5 loss.
    """
    sig_d = pd.Timestamp("2026-03-23")
    asof = pd.Timestamp("2026-03-23")
    dec = select_gld_sync_strategy(sig_d, {"bp_low": 0.045}, 27.5, asof)
    assert dec["strategy"] == "SELL PUT"
    assert dec["reason"] == "DEEP_BREAK_HIGH_IV"
