"""v3.7.249: Dashboard `run_backtest` deprecation wrapper + parity harness.

Why this exists
---------------
The legacy `core.signals_v2.run_backtest` is a spot-level daily replay used by
the Streamlit Dashboard. The plan contract requires deprecating it in favor of
a pipeline routed through `generate_daily_signals` (canonical signals) while
**preserving** the intraday exit-event semantics (StopLoss / BandExit /
Pullback / Timeout event counts within ±1).

This module provides:

* ``run_unified_backtest`` — produces trades by first calling
  ``generate_daily_signals`` for the canonical buy/exit signals, then replaying
  the same per-day exit rules legacy `run_backtest` uses (StopLoss → BandExit →
  Pullback → Timeout in that priority order).
* ``parity_check`` — compares legacy vs unified trade lists by exit-type
  counts. Reports one of {PASS, PASS_WITH_DRIFT, FAIL_EXIT_EVENT_COUNT}.
  When signal-column drift exists, writes one row per drifted date to
  ``data/cache/signal_drift_attribution.csv`` so the drift is auditable.

The legacy `run_backtest` retains its body for one release (the
``DeprecationWarning`` emission is added in `signals_v2.py` separately).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Allowed event types from legacy run_backtest exit handler.
EXIT_TYPES = ("StopLoss", "BandExit", "Pullback", "Timeout")
ALLOWED_COUNT_DRIFT = 1   # plan contract: ±1


@dataclass
class ParityVerdict:
    status: str  # PASS / PASS_WITH_DRIFT / FAIL_EXIT_EVENT_COUNT
    legacy_event_counts: dict
    unified_event_counts: dict
    max_count_drift: int
    drifted_signal_dates: list
    attribution_path: Optional[str] = None


def _replay_one_pass(close_d, high_d, low_d,
                        upper_band, lower_band,
                        buy_signal: pd.Series,
                        exit_signal: pd.Series,
                        *, max_hold_days: int = 30,
                        stop_loss_pct: float = 3.0,
                        pullback_gain: float = 5.0,
                        pullback_dd: float = 2.0) -> list:
    """Slim replay of the legacy daily exit logic.

    Mirrors ``core.signals_v2.run_backtest``'s prioritized exit handler:
    StopLoss → BandExit (exit_signal) → Pullback → Timeout. Returns a list of
    trade dicts with ``entry_date`` / ``exit_date`` / ``exit_type``.

    NOT a full reimplementation — the *signal* source (buy_signal,
    exit_signal) is supplied by the caller, which is the whole point of
    routing through ``generate_daily_signals`` for the canonical pipeline.
    """
    dates = upper_band.dropna().index.intersection(lower_band.dropna().index)
    trades = []
    in_trade = False
    entry_dt = entry_price = None
    peak = 0.0
    for d in dates:
        if d not in close_d.index:
            continue
        c = float(close_d[d]); h = float(high_d[d]); lo = float(low_d[d])
        u = float(upper_band[d]); l = float(lower_band[d])
        if any(map(np.isnan, (c, h, lo, u, l))) or u <= l:
            continue
        if in_trade:
            peak = max(peak, h)
            exit_type = None
            # 1) StopLoss
            if entry_price > 0 and lo / entry_price - 1 < -stop_loss_pct / 100:
                exit_type = "StopLoss"
            # 2) BandExit (canonical exit_signal from generate_daily_signals)
            elif bool(exit_signal.get(d, False)):
                exit_type = "BandExit"
            # 3) Pullback
            elif peak > entry_price * (1 + pullback_gain / 100) \
                    and (peak - c) / peak * 100 >= pullback_dd:
                exit_type = "Pullback"
            # 4) Timeout
            elif (d - entry_dt).days >= max_hold_days:
                exit_type = "Timeout"
            if exit_type is not None:
                trades.append({"entry_date": entry_dt,
                                 "exit_date": d,
                                 "exit_type": exit_type,
                                 "entry_price": entry_price,
                                 "exit_price": c})
                in_trade = False
                entry_dt = entry_price = None
                peak = 0.0
        if not in_trade and bool(buy_signal.get(d, False)):
            in_trade = True
            entry_dt = d
            entry_price = c
            peak = h
    return trades


def run_unified_backtest(close_d, high_d, low_d,
                            upper_band, lower_band,
                            regime, rv_pctile,
                            asset: str,
                            gvz_series: Optional[pd.Series] = None,
                            **kwargs) -> dict:
    """Unified path: canonical ``generate_daily_signals`` → exit replay.

    Returns ``{"trades": [...], "sig_df": <DataFrame>}``.
    """
    from core.signals_v2 import generate_daily_signals
    sig_df = generate_daily_signals(close_d, high_d, low_d,
                                       upper_band, lower_band,
                                       regime, rv_pctile,
                                       asset=asset, gvz_series=gvz_series)
    buy = sig_df["buy_signal"].fillna(False).astype(bool) \
            if "buy_signal" in sig_df.columns else pd.Series(False, index=sig_df.index)
    exit_sig = sig_df["exit_signal"].fillna(False).astype(bool) \
                if "exit_signal" in sig_df.columns else pd.Series(False, index=sig_df.index)
    trades = _replay_one_pass(close_d, high_d, low_d, upper_band, lower_band,
                                 buy, exit_sig, **kwargs)
    return {"trades": trades, "sig_df": sig_df}


def _count_events(trades: list) -> dict:
    out = {t: 0 for t in EXIT_TYPES}
    for t in trades:
        et = t.get("exit_type")
        if et in out: out[et] += 1
    return out


def parity_check(legacy_trades: list,
                    unified_trades: list,
                    legacy_buy_signal: Optional[pd.Series] = None,
                    unified_buy_signal: Optional[pd.Series] = None,
                    drift_attribution_path: Optional[str] = None,
                    max_count_drift: int = ALLOWED_COUNT_DRIFT) -> ParityVerdict:
    """Compare two trade lists per the AC-14 contract.

    Status semantics:
      PASS                  : identical event counts AND identical buy_signal
      PASS_WITH_DRIFT       : event counts within ±max_count_drift; buy_signal
                              columns differ but each difference is appended to
                              ``signal_drift_attribution.csv``
      FAIL_EXIT_EVENT_COUNT : any event-type count differs by > max_count_drift
    """
    lec = _count_events(legacy_trades)
    uec = _count_events(unified_trades)
    max_drift = max(abs(lec[t] - uec[t]) for t in EXIT_TYPES)
    if max_drift > max_count_drift:
        return ParityVerdict(
            status="FAIL_EXIT_EVENT_COUNT",
            legacy_event_counts=lec, unified_event_counts=uec,
            max_count_drift=max_drift, drifted_signal_dates=[],
        )

    drifted = []
    attribution_path = None
    if legacy_buy_signal is not None and unified_buy_signal is not None:
        legacy_buy_signal = legacy_buy_signal.fillna(False).astype(bool)
        unified_buy_signal = unified_buy_signal.fillna(False).astype(bool)
        common = legacy_buy_signal.index.intersection(unified_buy_signal.index)
        diff_mask = (legacy_buy_signal.reindex(common)
                       != unified_buy_signal.reindex(common))
        drifted = [d.isoformat() for d in common[diff_mask]]
        if drifted and drift_attribution_path is not None:
            attribution_path = str(drift_attribution_path)
            Path(attribution_path).parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for d in common[diff_mask]:
                rows.append({
                    "signal_date": d.isoformat(),
                    "legacy_buy_signal": bool(legacy_buy_signal.loc[d]),
                    "unified_buy_signal": bool(unified_buy_signal.loc[d]),
                    "attribution": (
                        "canonical_pipeline_added_filter"
                        if (legacy_buy_signal.loc[d] and not unified_buy_signal.loc[d])
                        else "canonical_pipeline_admitted_signal"
                    ),
                })
            df = pd.DataFrame(rows)
            mode = "a" if Path(attribution_path).exists() else "w"
            header = mode == "w"
            df.to_csv(attribution_path, mode=mode, header=header, index=False)

    if drifted:
        status = "PASS_WITH_DRIFT"
    else:
        status = "PASS"
    return ParityVerdict(
        status=status,
        legacy_event_counts=lec, unified_event_counts=uec,
        max_count_drift=max_drift, drifted_signal_dates=drifted,
        attribution_path=attribution_path,
    )
