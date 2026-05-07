"""BUY CALL long call (单腿 ATM call OR bull call spread).

v3.7.96/120 实证 exit 规则 (旧 simulate_option_exit 已是最优):
  +100% premium → TP
  -50% premium → SL
  expiry → 强平

可选现代化 (core/strategies/options_exit.simulate_bc_exit):
  + DTE-cliff 强平
  + signal-reversal 早平
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class BCConfig:
    """BC 入场 + 退出参数 (v3.7.170 WR-first grid 后).

    90d kline_db grid (n=11 GLD / n=29 SLV):
      GLD 候选 (按 WR):
        1.5x/0.3x/30d wr=55% sum=+43% (WR 高但 SL 太紧丢收益)
        1.5x/0.5x/30d wr=55% sum=+112%
        1.5x/0.7x/30d wr=55% sum=+231% ← 同 WR, sum 最高
      SLV 候选 (按 WR):
        1.5x/0.3x/30d wr=56% sum=+180% ← 最高 WR
        1.5x/0.7x/30d wr=45% sum=+351% (sum 高但 WR 低)
      共识: pt=1.5x (50% premium TP) / DTE=30
      SL: 0.5x (中庸 — 不丢趋势也不过度等待反弹)
    """
    profit_target_mult: float = 1.5      # +50% premium 早平
    stop_loss_mult: float = 0.5          # v3.7.170: 0.7→0.5 (中庸 SL)
                                          # 0.7 太紧噪声打, 0.3 太松等爆零
    base_dte: int = 30                   # 近 DTE theta 优势


def simulate_bc_position(entry_pricing: dict,
                            signal_date: pd.Timestamp,
                            today_dt: pd.Timestamp,
                            db: pd.DataFrame,
                            cfg: BCConfig = None) -> dict:
    """BC long call MTM + 真实 OHLC exit (v3.7.96 实证最优规则).

    entry_pricing: {entry_price, legs=[(label, code, K, qty)], source, ...}
    db: kline_db OHLC (date, code, close, expiry)

    Returns:
        is_closed, exit_date, exit_value, exit_reason, pnl_pct,
        hold_days, leg_prices
    """
    if cfg is None: cfg = BCConfig()
    if not entry_pricing.get("legs"):
        return {"is_closed": False, "reason": "no legs"}
    entry_value = entry_pricing["entry_price"]
    if abs(entry_value) < 0.01:
        return {"is_closed": False, "reason": "entry~0"}
    legs = entry_pricing["legs"]
    profit_target = entry_value * cfg.profit_target_mult
    stop_loss = entry_value * cfg.stop_loss_mult
    # Expiry
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
        # SL
        if cur_value <= stop_loss:
            return {"is_closed": True, "exit_date": d_ts, "exit_value": cur_value,
                     "exit_reason": "-50% stop loss", "pnl_pct": pnl_pct,
                     "hold_days": hold, "leg_prices": leg_prices_at_exit}
        # Expiry
        if d_ts >= expiry_dt:
            return {"is_closed": True, "exit_date": d_ts, "exit_value": cur_value,
                     "exit_reason": "expiry", "pnl_pct": pnl_pct,
                     "hold_days": hold, "leg_prices": leg_prices_at_exit}
    # OPEN
    if hold > 0:
        pnl_pct = (cur_value / entry_value - 1) * 100
        return {"is_closed": False, "current_value": cur_value, "hold_days": hold,
                 "pnl_pct": pnl_pct, "leg_prices": leg_prices_at_exit}
    # v3.7.153: kline_db 滞后, OPEN MTM fallback 用最近可用日 (>= sig_d)
    nearest = db[db["code"].isin([l[1] for l in legs]) &
                  (db["date"] >= sig_d) & (db["date"] <= today_dt)]
    if not nearest.empty:
        latest_date = nearest["date"].max()
        cur_total = 0.0; ok = True; leg_prices_today = []
        for _lab, _code, _K, _qty in legs:
            r = db[(db["code"] == _code) & (db["date"] == latest_date)]
            if not len(r): ok = False; break
            _p = float(r.iloc[0]["close"])
            leg_prices_today.append((_lab, _p))
            cur_total += _qty * _p
        if ok:
            cur_value = cur_total
            pnl_pct = (cur_value / entry_value - 1) * 100
            return {"is_closed": False, "current_value": cur_value,
                     "hold_days": max(0, (today_dt - sig_d).days),
                     "pnl_pct": pnl_pct, "leg_prices": leg_prices_today}
    # 真没数据 — 用 entry leg prices 作 OPEN MTM (无变化)
    ent_leg_prices = entry_pricing.get("leg_prices", [])
    return {"is_closed": False, "current_value": entry_value,
             "hold_days": max(0, (today_dt - sig_d).days),
             "pnl_pct": 0.0, "leg_prices": ent_leg_prices}
