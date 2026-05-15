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
    """BC 入场 + 退出参数.

    v3.7.205: 5y BS proxy + 真实 kline_db (n=83, v3.7.201 信号过滤) grid 实证:
      36 combo (pt 1.2-3.0 × sl 0.3-0.8):
        pt=2.5/sl=0.3 ★ WR=86.7% sum=+11851% mean=+143% scoreB=476 (推荐)
        pt=3.0/sl=0.3   WR=83.1% sum=+13360% (最高 sum 但 WR 略低)
        pt=1.5/sl=0.5 旧 WR=74.7% sum=+4078% scoreB=121 (旧 cfg)
      反直觉发现: SL 越紧 (0.3 = premium 跌 70% 才止损) WR 反而越高
                  因 BC 杠杆 + theta, 早 SL 错过反弹; 等到 expiry 也 max -100%
      旧 v3.7.170 是 90d n=11 小样本 grid, 已被 5y n=83 推翻
    """
    profit_target_mult: float = 2.5      # +150% premium 才平 (BC 长线持仓)
    stop_loss_mult: float = 0.3          # premium 跌 70% 才止损 (容忍深亏等反弹)
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
    # v3.7.199: kline_db 没数据 (cron 滞后或 contract 不在 db) → 拉 yfinance live
    try:
        from core.paper_positions import fetch_live_leg_prices
        live_map = fetch_live_leg_prices(legs)
    except Exception:
        live_map = {}
    if live_map and all(l[1] in live_map for l in legs):
        leg_prices_live = []; cur_total = 0.0
        for _lab, _code, _K, _qty in legs:
            _p = float(live_map[_code])
            leg_prices_live.append((_lab, _p))
            cur_total += _qty * _p
        cur_value = cur_total
        pnl_pct = (cur_value / entry_value - 1) * 100
        return {"is_closed": False, "current_value": cur_value,
                 "hold_days": max(0, (today_dt - sig_d).days),
                 "pnl_pct": pnl_pct, "leg_prices": leg_prices_live,
                 "price_source": "yfinance_live"}
    # 真没数据 — 用 entry leg prices 作 OPEN MTM (无变化)
    ent_leg_prices = entry_pricing.get("leg_prices", [])
    return {"is_closed": False, "current_value": entry_value,
             "hold_days": max(0, (today_dt - sig_d).days),
             "pnl_pct": 0.0, "leg_prices": ent_leg_prices,
             "price_source": "stale_entry"}
