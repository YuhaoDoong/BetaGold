"""SHORT_VOL Iron Condor (-ATM put / +OTM put / -ATM call / +OTM call) — 做空波动率.

当前简化为 SP credit spread (4-leg IC 真实模型待后续).

v3.7.96 实证 exit 规则:
  +50% credit → TP
  -50% credit → SL
  hold >= 30d → 强平
  expiry → 强平
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class ShortVolConfig:
    profit_target_credit_pct: float = 50.0   # +50% credit
    stop_loss_pct: float = 50.0              # -50% credit
    hold_max_days: int = 30
    base_dte: int = 30


def simulate_short_vol_position(entry_pricing: dict,
                                    signal_date: pd.Timestamp,
                                    today_dt: pd.Timestamp,
                                    db: pd.DataFrame,
                                    cfg: ShortVolConfig = None) -> dict:
    if cfg is None: cfg = ShortVolConfig()
    if not entry_pricing.get("legs"):
        return {"is_closed": False, "reason": "no legs"}
    entry_value = entry_pricing["entry_price"]
    if abs(entry_value) < 0.01:
        return {"is_closed": False, "reason": "entry~0"}
    legs = entry_pricing["legs"]
    ent_leg_prices = entry_pricing.get("leg_prices", [])
    # v3.7.141/148: SHORT_VOL 真 Iron Condor 4-leg
    # 修正 v3.7.148 max_risk bug: 之前用 max_spread - total_credit (错!)
    # IC 只能一边 max-lose, max_risk = max(put_width - put_credit, call_width - call_credit)
    # 这才是券商保证金 (Reg-T) 算法
    if len(legs) == 4:
        sp_k = next((l[2] for l in legs if l[0] == "short_put"), 0)
        lp_k = next((l[2] for l in legs if l[0] == "long_put"), 0)
        sc_k = next((l[2] for l in legs if l[0] == "short_call"), 0)
        lc_k = next((l[2] for l in legs if l[0] == "long_call"), 0)
        put_width = abs(sp_k - lp_k) if sp_k and lp_k else 0
        call_width = abs(lc_k - sc_k) if sc_k and lc_k else 0
        # 单边 credit (从入场 leg prices 重算)
        ent_dict = {lab: p for lab, p in ent_leg_prices}
        put_credit = ent_dict.get("short_put", 0) - ent_dict.get("long_put", 0)
        call_credit = ent_dict.get("short_call", 0) - ent_dict.get("long_call", 0)
        max_loss_put = max(0, put_width - put_credit)
        max_loss_call = max(0, call_width - call_credit)
        max_risk = max(0.01, max_loss_put, max_loss_call)
    else:
        # 兼容 2-leg 旧实现 (put credit spread only)
        spread_width = 0.0
        if len(legs) >= 2:
            ks = [l[2] for l in legs if "short" in l[0]]
            kl = [l[2] for l in legs if l[0] == "long_put" or
                   (l[0] == "long_call" and "short" not in l[0])]
            if ks and kl: spread_width = abs(ks[0] - kl[0])
        max_risk = max(0.01, spread_width - entry_value) \
                    if spread_width > 0 else entry_value
    profit_target = entry_value * (1 - cfg.profit_target_credit_pct / 100)
    stop_loss = entry_value + (cfg.stop_loss_pct / 100) * max_risk

    def _pnl(cv): return (entry_value - cv) / max_risk * 100

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
        cur_value = -cur_total
        hold += 1
        pnl_pct = _pnl(cur_value)
        if cur_value <= profit_target:
            return {"is_closed": True, "exit_date": d_ts, "exit_value": cur_value,
                     "exit_reason": "+50% credit", "pnl_pct": pnl_pct,
                     "hold_days": hold, "leg_prices": leg_prices_at_exit}
        if cur_value >= stop_loss:
            return {"is_closed": True, "exit_date": d_ts, "exit_value": cur_value,
                     "exit_reason": "-50% SL", "pnl_pct": pnl_pct,
                     "hold_days": hold, "leg_prices": leg_prices_at_exit}
        if hold >= cfg.hold_max_days:
            return {"is_closed": True, "exit_date": d_ts, "exit_value": cur_value,
                     "exit_reason": f"{cfg.hold_max_days}d 定时", "pnl_pct": pnl_pct,
                     "hold_days": hold, "leg_prices": leg_prices_at_exit}
        if d_ts >= expiry_dt:
            return {"is_closed": True, "exit_date": d_ts, "exit_value": cur_value,
                     "exit_reason": "expiry", "pnl_pct": pnl_pct,
                     "hold_days": hold, "leg_prices": leg_prices_at_exit}
    if hold > 0:
        pnl_pct = _pnl(cur_value)
        return {"is_closed": False, "current_value": cur_value, "hold_days": hold,
                 "pnl_pct": pnl_pct, "leg_prices": leg_prices_at_exit,
                 "max_risk": max_risk}
    return {"is_closed": False, "reason": "no data after entry"}
