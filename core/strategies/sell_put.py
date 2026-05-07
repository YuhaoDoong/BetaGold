"""SELL PUT credit spread (-ATM put / +OTM put).

v3.7.96/120 实证 exit 规则 (用 max_risk = spread_width - credit 当 PnL 分母):
  +50% credit (cur_credit ≤ 0.5 × entry) → TP 早平
  -50% margin (cur_credit ≥ entry + 0.5 × max_risk) → SL
  expiry → 强平

PnL% = (entry_credit - cur_credit) / max_risk × 100
  跟券商 Reg-T margin 一致, BC 用 premium / SP 用 margin 对称.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class SPConfig:
    """SP credit spread 退出参数 (v3.7.169 grid 后).

    90d kline_db grid (n=10 GLD / n=24 SLV):
      GLD 现行 (50/50/45d): wr=60% sum=-47%
      GLD 最优 (50/100/30d): wr=78% sum=+70%
      SLV 现行 (50/50/45d): wr=46% sum=-114%
      SLV 最优 (30/100/30d): wr=92% sum=+709% ← 极大改善
    SL 50%→100% margin: 不主动止损, 让 spread 走到 expiry (premium decay 帮我们).
    profit_target 30 vs 50: SLV 短取 30%, GLD 50% 留空间.
    """
    # v3.7.170 WR-first: SLV grid wr=92% (n=24, 30/100/30d) — 极优.
    # GLD: 50/100/30d wr=78% sum=+70% — 选 50 让 GLD 留更多收益.
    # 折中 pt=50 (两 asset 兼容), SL=100 (= 不主动止损, expiry 收 partial credit)
    profit_target_credit_pct: float = 50.0   # +50% credit 早平 (GLD 优于 30)
    stop_loss_margin_pct: float = 100.0      # = 等价无主动 SL, 让 spread 走完
    base_dte: int = 30


def simulate_sp_position(entry_pricing: dict,
                            signal_date: pd.Timestamp,
                            today_dt: pd.Timestamp,
                            db: pd.DataFrame,
                            cfg: SPConfig = None) -> dict:
    """SP credit spread MTM + 真实 OHLC exit.

    entry_pricing: {entry_price=net credit, legs=[(label, code, K, qty)], ...}
    """
    if cfg is None: cfg = SPConfig()
    if not entry_pricing.get("legs"):
        return {"is_closed": False, "reason": "no legs"}
    entry_value = entry_pricing["entry_price"]  # net credit collected
    if abs(entry_value) < 0.01:
        return {"is_closed": False, "reason": "entry~0"}
    legs = entry_pricing["legs"]
    # max_risk = spread_width - credit (Reg-T margin)
    spread_width = 0.0
    if len(legs) >= 2:
        ks = [l[2] for l in legs if "short" in l[0]]
        kl = [l[2] for l in legs if "put" in l[0] and "short" not in l[0]]
        if ks and kl: spread_width = abs(ks[0] - kl[0])
    max_risk = max(0.01, spread_width - entry_value) \
                if spread_width > 0 else entry_value
    profit_target = entry_value * (1 - cfg.profit_target_credit_pct / 100)
    stop_loss = entry_value + (cfg.stop_loss_margin_pct / 100) * max_risk

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
        cur_value = -cur_total  # net debit when closing credit spread
        hold += 1
        pnl_pct = _pnl(cur_value)
        # TP
        if cur_value <= profit_target:
            return {"is_closed": True, "exit_date": d_ts, "exit_value": cur_value,
                     "exit_reason": "+50% credit", "pnl_pct": pnl_pct,
                     "hold_days": hold, "leg_prices": leg_prices_at_exit,
                     "max_risk": max_risk}
        # SL
        if cur_value >= stop_loss:
            return {"is_closed": True, "exit_date": d_ts, "exit_value": cur_value,
                     "exit_reason": "-50% margin SL", "pnl_pct": pnl_pct,
                     "hold_days": hold, "leg_prices": leg_prices_at_exit,
                     "max_risk": max_risk}
        # Expiry
        if d_ts >= expiry_dt:
            return {"is_closed": True, "exit_date": d_ts, "exit_value": cur_value,
                     "exit_reason": "expiry", "pnl_pct": pnl_pct,
                     "hold_days": hold, "leg_prices": leg_prices_at_exit,
                     "max_risk": max_risk}
    # OPEN
    if hold > 0:
        pnl_pct = _pnl(cur_value)
        return {"is_closed": False, "current_value": cur_value, "hold_days": hold,
                 "pnl_pct": pnl_pct, "leg_prices": leg_prices_at_exit,
                 "max_risk": max_risk}
    # v3.7.153: kline_db 滞后 fallback (跟 short_vol/buy_call 同款)
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
            cur_value = -cur_total
            return {"is_closed": False, "current_value": cur_value,
                     "hold_days": max(0, (today_dt - sig_d).days),
                     "pnl_pct": _pnl(cur_value),
                     "leg_prices": leg_prices_today, "max_risk": max_risk}
    # 真没数据 — 用 entry leg prices (无变化)
    ent_leg_prices = entry_pricing.get("leg_prices", [])
    return {"is_closed": False, "current_value": entry_value,
             "hold_days": max(0, (today_dt - sig_d).days),
             "pnl_pct": 0.0, "leg_prices": ent_leg_prices,
             "max_risk": max_risk}
