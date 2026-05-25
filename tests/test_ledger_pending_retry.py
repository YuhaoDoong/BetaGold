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
