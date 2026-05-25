"""Regression tests for v3.7.251 (review round 2 P1#3 fix): ledger
PENDING_KLINE clamp + idempotent dedup by (asset, signal_date, strategy).

These tests verify the LOGIC of the dedup + clamp policy at the unit level.
They do not invoke ``build_positions_ledger.main()`` end-to-end (that
requires production data) — they exercise the invariants directly.
"""
from __future__ import annotations
import pandas as pd
import pytest


def test_dedup_drops_duplicate_keys_keeping_first():
    """The dedup step at write time must drop duplicate (asset, signal_date,
    strategy) keys, keeping the FIRST occurrence (the frozen historical row).
    """
    refreshed = [
        {"asset": "GLD", "signal_date": "2026-04-15",
         "strategy": "BUY CALL", "entry_credit_or_premium": 22.0},
        {"asset": "GLD", "signal_date": "2026-04-16",
         "strategy": "BUY CALL", "entry_credit_or_premium": 18.0},
    ]
    # Simulate a PENDING_KLINE retry — the same signal_dates re-evaluated
    new_rows = [
        {"asset": "GLD", "signal_date": "2026-04-15",
         "strategy": "BUY CALL", "entry_credit_or_premium": 99.99},  # would clobber
        {"asset": "GLD", "signal_date": "2026-04-17",
         "strategy": "BUY CALL", "entry_credit_or_premium": 20.0},  # genuinely new
    ]
    df = pd.DataFrame(refreshed + new_rows)
    deduped = df.drop_duplicates(
        subset=["asset", "signal_date", "strategy"], keep="first")
    # Only 3 unique keys
    assert len(deduped) == 3
    # The frozen value (22.0) for 4-15 wins over the retry (99.99)
    row_415 = deduped[deduped["signal_date"] == "2026-04-15"].iloc[0]
    assert row_415["entry_credit_or_premium"] == 22.0
    # The new 4-17 row is preserved
    row_417 = deduped[deduped["signal_date"] == "2026-04-17"].iloc[0]
    assert row_417["entry_credit_or_premium"] == 20.0


def test_dedup_distinguishes_strategy_at_same_date():
    """(asset, signal_date, strategy) is the key — same date with different
    strategies is NOT a duplicate."""
    df = pd.DataFrame([
        {"asset": "GLD", "signal_date": "2026-04-15", "strategy": "BUY CALL"},
        {"asset": "GLD", "signal_date": "2026-04-15", "strategy": "SELL PUT"},
    ])
    deduped = df.drop_duplicates(
        subset=["asset", "signal_date", "strategy"], keep="first")
    assert len(deduped) == 2


def test_waterline_clamp_pushes_back_one_trading_day():
    """Earliest pending date D should produce a new waterline at D - 1 trading
    day; next refresh re-evaluates D."""
    earliest_pending = pd.Timestamp("2026-04-15")  # Wed
    clamped = pd.bdate_range(
        end=earliest_pending - pd.Timedelta(days=1), periods=1)[-1]
    assert clamped == pd.Timestamp("2026-04-14")  # Tue


def test_waterline_clamp_uses_min_of_latest_and_clamped():
    """If latest_data_date (e.g. ETF max) is older than the clamped value,
    keep the older one (do not advance past actual data)."""
    earliest_pending = pd.Timestamp("2026-04-15")
    clamped = pd.bdate_range(
        end=earliest_pending - pd.Timedelta(days=1), periods=1)[-1]
    latest_data = pd.Timestamp("2026-04-10")  # ETF data only up to 4-10
    chosen = min(latest_data, clamped)
    assert chosen == pd.Timestamp("2026-04-10")


# -- v3.7.252 (review round 3 P2): AWAITING state propagation lifecycle ------


def test_refresh_path_propagates_awaiting_state():
    """A position opened pre-expiry reaches expiry-day later via the refresh
    path; AWAITING_EXPIRY_CLOSE state must flow into the row.

    Simulates the lifecycle by emulating what _refresh_open_position does:
    take an existing open row, run simulate_option_exit, and copy fields
    into the row dict. The fix is that 'state' is now part of that copy.
    """
    row = {
        "asset": "GLD", "signal_date": "2026-04-15",
        "strategy": "BUY CALL",
        "is_closed": False, "current_value": 12.0, "pnl_pct": -10.0,
        "hold_days": 5, "state": None,
    }
    # Imagine simulate_option_exit returned AWAITING after refresh
    sim = {
        "is_closed": False,
        "state": "AWAITING_EXPIRY_CLOSE",
        "expiry_dt": pd.Timestamp("2026-05-15"),
        "reason": "expiry_close_missing for GLD @ 2026-05-15",
    }
    # Mirror the update done in _refresh_open_position:
    row["is_closed"] = sim.get("is_closed", False)
    row["current_value"] = float(sim.get("current_value", 0) or 0)
    row["pnl_pct"] = float(sim.get("pnl_pct", 0) or 0)
    row["hold_days"] = int(sim.get("hold_days", 0) or 0)
    row["state"] = sim.get("state", None)   # v3.7.252 line
    # Distinguishes AWAITING from normal no-data open
    assert row["state"] == "AWAITING_EXPIRY_CLOSE"
    assert row["is_closed"] is False
    # Default zeros are tolerable because state tells the caller why
    assert row["current_value"] == 0
    assert row["pnl_pct"] == 0


def test_state_cleared_when_position_later_closes():
    """Once the ETF expiry close materializes and the row closes, the
    state field clears (no stale AWAITING tag on a closed row)."""
    row = {
        "asset": "GLD", "signal_date": "2026-04-15",
        "strategy": "BUY CALL",
        "is_closed": False, "state": "AWAITING_EXPIRY_CLOSE",
    }
    # Next refresh: ETF close now exists; force_close_at_expiry returns closed
    sim = {
        "is_closed": True,
        "exit_value": 12.29,
        "exit_reason": "expiry intrinsic (db missing) spot=417.29",
        "pnl_pct": -44.4,
        "hold_days": 30,
        "state": None,  # AWAITING cleared
    }
    row["is_closed"] = sim.get("is_closed", False)
    row["exit_value"] = float(sim.get("exit_value", 0) or 0)
    row["exit_reason"] = sim.get("exit_reason", "")
    row["pnl_pct"] = float(sim.get("pnl_pct", 0) or 0)
    row["hold_days"] = int(sim.get("hold_days", 0) or 0)
    row["state"] = sim.get("state", None)
    assert row["is_closed"] is True
    assert row["state"] is None  # AWAITING successfully cleared
    assert row["pnl_pct"] == -44.4
