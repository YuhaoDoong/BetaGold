"""STRADDLE (long ATM call + ATM put) — 做多波动率.

v3.7.96 实证 exit 规则:
  +100% premium (cur >= entry × 2) → TP
  hold >= 14d 强平 (long vol 衰减期)
  expiry → 强平
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class StraddleConfig:
    profit_target_mult: float = 2.0        # +100% (cur >= entry × 2)
    hold_max_days: int = 14
    base_dte: int = 14                      # 短 DTE long vol


def simulate_straddle_position(entry_pricing: dict,
                                  signal_date: pd.Timestamp,
                                  today_dt: pd.Timestamp,
                                  db: pd.DataFrame,
                                  cfg: StraddleConfig = None) -> dict:
    if cfg is None: cfg = StraddleConfig()
    if not entry_pricing.get("legs"):
        return {"is_closed": False, "reason": "no legs"}
    entry_value = entry_pricing["entry_price"]  # call + put combined premium
    if abs(entry_value) < 0.01:
        return {"is_closed": False, "reason": "entry~0"}
    legs = entry_pricing["legs"]
    profit_target = entry_value * cfg.profit_target_mult
    first_kdb = db[db["code"] == legs[0][1]]
    if not len(first_kdb):
        return {"is_closed": False, "reason": "no db data"}
    expiry_dt = pd.Timestamp(first_kdb.iloc[0]["expiry"])
    sig_d = pd.Timestamp(signal_date).normalize()
    days = sorted(set(db[db["code"].isin([l[1] for l in legs])]["date"].unique()))
    days_after = [d for d in days if pd.Timestamp(d) > sig_d]
    hold = 0; leg_prices_at_exit = []; cur_value = entry_value
    for d in days_after:
        d_ts = pd.Timestamp(d)
        if d_ts > today_dt: break
        cur_total = 0.0; ok = True; leg_prices_today = []
        for _lab, _code, _K, _qty in legs:
            r = db[(db["code"] == _code) & (db["date"] == d_ts)]
            if not len(r): ok = False; break
            _p = float(r.iloc[0]["close"])
            leg_prices_today.append((_lab, _p))
            cur_total += _qty * _p
        if not ok: continue
        leg_prices_at_exit = leg_prices_today
        cur_value = cur_total
        hold += 1
        pnl_pct = (cur_value / entry_value - 1) * 100
        # TP
        if cur_value >= profit_target:
            return {"is_closed": True, "exit_date": d_ts, "exit_value": cur_value,
                     "exit_reason": "+100% profit", "pnl_pct": pnl_pct,
                     "hold_days": hold, "leg_prices": leg_prices_at_exit}
        # 14d 定时 (long vol 衰减期)
        if hold >= cfg.hold_max_days:
            return {"is_closed": True, "exit_date": d_ts, "exit_value": cur_value,
                     "exit_reason": f"{cfg.hold_max_days}d 定时", "pnl_pct": pnl_pct,
                     "hold_days": hold, "leg_prices": leg_prices_at_exit}
        # Expiry
        if d_ts >= expiry_dt:
            return {"is_closed": True, "exit_date": d_ts, "exit_value": cur_value,
                     "exit_reason": "expiry", "pnl_pct": pnl_pct,
                     "hold_days": hold, "leg_prices": leg_prices_at_exit}
    if hold > 0:
        pnl_pct = (cur_value / entry_value - 1) * 100
        return {"is_closed": False, "current_value": cur_value, "hold_days": hold,
                 "pnl_pct": pnl_pct, "leg_prices": leg_prices_at_exit}
    return {"is_closed": False, "reason": "no data after entry"}
