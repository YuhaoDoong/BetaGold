"""Data freshness state machine for kline_db (and any future data sources).

v3.7.237: Surfaces explicit FRESH / STALE / FROZEN states so consumers can
gracefully degrade rather than silently use stale data.

State semantics for kline_db (per AC-6):
  FRESH   : max_date is within ``fresh_max_days`` trading days of ``today``
  STALE   : within (fresh_max_days, frozen_min_days] trading days
  FROZEN  : strictly more than ``frozen_min_days`` trading days behind

The ledger daemon gates *new option entries* on FROZEN only; futures (Binance
live), MTM/exit on existing positions, and force_close_at_expiry are exempt
because they do not depend on kline_db pricing.

The per-entry gate in ``core.paper_positions._kline_db_freshness_status`` uses
a different (binary FRESH / PENDING_KLINE) threshold (``KLINE_MAX_FALLBACK_DAYS``)
specifically for individual pricing fallback. This module's tiered states are
for *observability + ledger-daemon-level gating*; the two thresholds may differ.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

KLINE_DB_PATH = "/Users/yhdong/Gold/data/raw/options_history/kline_db/all_klines.parquet"

# Tiered thresholds (in trading days; weekend-naive 'days' on bdate is fine for
# practical use given US options trade Mon-Fri):
DEFAULT_FRESH_MAX_DAYS = 2     # ≤ this gap → FRESH
DEFAULT_FROZEN_MIN_DAYS = 3    # > this gap → FROZEN; in (FRESH, FROZEN] → STALE


@dataclass(frozen=True)
class FreshnessRecord:
    """Structured snapshot of a single data source's freshness state."""

    source: str           # 'kline_db', 'gld_csv', 'gvz', ...
    state: str            # 'FRESH' | 'STALE' | 'FROZEN' | 'MISSING'
    max_date: Optional[pd.Timestamp]
    gap_trading_days: Optional[int]
    as_of: pd.Timestamp   # When this was evaluated

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "state": self.state,
            "max_date": (self.max_date.isoformat() if self.max_date is not None
                          else None),
            "gap_trading_days": self.gap_trading_days,
            "as_of": self.as_of.isoformat(),
        }


def _trading_day_gap(today: pd.Timestamp, max_date: pd.Timestamp) -> int:
    """Approximate trading-day gap using pandas bdate_range (Mon-Fri).

    For freshness gating this is sufficient — exact NYSE-calendar accuracy is
    not necessary at the 2-3 day decision boundary, and we never compute
    cumulative durations from this number.
    """
    if today <= max_date:
        return 0
    return max(0, len(pd.bdate_range(max_date + pd.Timedelta(days=1), today)))


def kline_db_state(today: pd.Timestamp,
                     fresh_max_days: int = DEFAULT_FRESH_MAX_DAYS,
                     frozen_min_days: int = DEFAULT_FROZEN_MIN_DAYS,
                     db_path: str = KLINE_DB_PATH) -> FreshnessRecord:
    """Evaluate kline_db freshness as of ``today`` (normalized).

    Args:
        today: Anchor date (typically pd.Timestamp.today().normalize()).
        fresh_max_days: Inclusive upper bound for FRESH state.
        frozen_min_days: Exclusive lower bound for FROZEN state.
        db_path: Override the parquet path (tests pass a fixture path).

    Returns:
        A ``FreshnessRecord`` describing the current state.
    """
    today = pd.Timestamp(today).normalize()
    p = Path(db_path)
    if not p.exists():
        return FreshnessRecord(
            source="kline_db", state="MISSING", max_date=None,
            gap_trading_days=None, as_of=today)
    try:
        df = pd.read_parquet(db_path, columns=["date"])
    except Exception:
        return FreshnessRecord(
            source="kline_db", state="MISSING", max_date=None,
            gap_trading_days=None, as_of=today)
    if not len(df):
        return FreshnessRecord(
            source="kline_db", state="MISSING", max_date=None,
            gap_trading_days=None, as_of=today)
    max_date = pd.to_datetime(df["date"]).max().normalize()
    gap = _trading_day_gap(today, max_date)
    if gap <= fresh_max_days:
        state = "FRESH"
    elif gap > frozen_min_days:
        state = "FROZEN"
    else:
        state = "STALE"
    return FreshnessRecord(
        source="kline_db", state=state, max_date=max_date,
        gap_trading_days=gap, as_of=today)


def gate_new_option_entry(today: Optional[pd.Timestamp] = None,
                             frozen_min_days: int = DEFAULT_FROZEN_MIN_DAYS
                             ) -> tuple:
    """Convenience gate for ledger daemon and similar callers.

    Returns:
        (allow_new_entry: bool, record: FreshnessRecord)

    Allow when state ∈ {FRESH, STALE}; block when FROZEN or MISSING.
    Futures, MTM, and force_close_at_expiry paths must NOT consult this gate
    — they have their own data sources (Binance, ETF daily) and remain
    operational regardless of kline freshness.
    """
    if today is None:
        today = pd.Timestamp.today().normalize()
    rec = kline_db_state(today, frozen_min_days=frozen_min_days)
    allow = rec.state in ("FRESH", "STALE")
    return allow, rec
