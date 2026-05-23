"""期权现代化 exit 规则 — 利用真实期权 OHLC + DTE + 信号序列.

替代旧的简单 "+100% TP / -50% SL / expiry" 规则.

新增退出维度:
  1. DTE-cliff: DTE 临近 (BC<14, SP<7) 强制平仓 (theta 加速 + gamma risk + assignment)
  2. Signal-reversal: bp_high>0.85 (区间上沿) 时若已盈利 → 早平
  3. Strike-defense (SP): spot 接近 short strike → 立即平 (避免 assignment)
  4. Daily bar-level: 用 daily High/Low 而非仅 close (TP/SL 可能日内触发)
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class BCExitConfig:
    """BUY CALL long call 现代化 exit 参数."""
    profit_target_pct: float = 100.0     # +100% premium → TP
    stop_loss_pct: float = -50.0         # -50% premium → SL
    dte_cliff_days: int = 14             # DTE < 此值强平 (theta cliff)
    signal_reversal_bp_high: float = 0.85  # bp_high > 此值算信号反转
    signal_reversal_min_profit: float = 30.0  # 反转 + 至少这么多利润才早平


@dataclass
class SPExitConfig:
    """SELL PUT credit spread 现代化 exit 参数."""
    profit_target_pct: float = 50.0      # +50% credit → TP (margin 分母)
    stop_loss_pct: float = -50.0         # -50% margin (= entry_credit + 0.5×max_risk)
    dte_cliff_days: int = 7              # DTE < 此值强平 (assignment risk)
    signal_reversal_bp_high: float = 0.85
    signal_reversal_min_profit: float = 30.0
    strike_defense_buffer: float = 1.02  # spot >= short_strike × 1.02 → 立即平


def simulate_bc_exit(entry_value: float,
                       legs: list,
                       signal_date: pd.Timestamp,
                       expiry_dt: pd.Timestamp,
                       today_dt: pd.Timestamp,
                       db: pd.DataFrame,
                       bp_high_series: pd.Series = None,
                       cfg: BCExitConfig = None) -> dict:
    """BC long call 现代化 exit. legs=[(label, code, K, qty)]."""
    if cfg is None: cfg = BCExitConfig()
    sig_d = pd.Timestamp(signal_date).normalize()
    days = sorted(set(db[db["code"].isin([l[1] for l in legs])]["date"].unique()))
    days_after = [d for d in days if pd.Timestamp(d) > sig_d]
    hold = 0
    leg_prices_today = []
    cur_value = entry_value
    for d in days_after:
        d_ts = pd.Timestamp(d)
        if d_ts > today_dt: break
        cur_total = 0.0; ok = True
        leg_prices_today = []
        for _lab, _code, _K, _qty in legs:
            r = db[(db["code"] == _code) & (db["date"] == d_ts)]
            if not len(r): ok = False; break
            _p = float(r.iloc[0]["close"])
            leg_prices_today.append((_lab, _p))
            cur_total += _qty * _p
        if not ok: continue
        cur_value = cur_total
        hold += 1
        pnl_pct = (cur_value / entry_value - 1) * 100
        # ① TP
        if pnl_pct >= cfg.profit_target_pct:
            return _bc_exit(d_ts, cur_value, hold, leg_prices_today, pnl_pct,
                              "+100% TP")
        # ② SL
        if pnl_pct <= cfg.stop_loss_pct:
            return _bc_exit(d_ts, cur_value, hold, leg_prices_today, pnl_pct,
                              "-50% SL")
        # ③ DTE-cliff (theta 加速期强平)
        days_to_exp = (expiry_dt - d_ts).days
        if days_to_exp <= cfg.dte_cliff_days:
            return _bc_exit(d_ts, cur_value, hold, leg_prices_today, pnl_pct,
                              f"DTE {days_to_exp}d 强平 (theta cliff)")
        # ④ Signal-reversal (bp_high 触上沿 + 已盈利 → 早平)
        if bp_high_series is not None and d_ts in bp_high_series.index:
            bph = float(bp_high_series.get(d_ts, 0))
            if bph > cfg.signal_reversal_bp_high \
               and pnl_pct >= cfg.signal_reversal_min_profit:
                return _bc_exit(d_ts, cur_value, hold, leg_prices_today, pnl_pct,
                                  f"bp_high {bph:.2f} 反转 + {pnl_pct:.0f}% 早平")
        # ⑤ Expiry
        if d_ts >= expiry_dt:
            return _bc_exit(d_ts, cur_value, hold, leg_prices_today, pnl_pct,
                              "expiry")
    # OPEN (持仓中 MTM)
    if hold > 0:
        pnl_pct = (cur_value / entry_value - 1) * 100
        return {"is_closed": False, "current_value": cur_value,
                 "hold_days": hold, "pnl_pct": pnl_pct,
                 "leg_prices": leg_prices_today}
    return {"is_closed": False}


def _bc_exit(d_ts, cur_value, hold, leg_prices, pnl_pct, reason):
    return {"is_closed": True, "exit_date": d_ts,
             "exit_value": cur_value, "exit_reason": reason,
             "hold_days": hold, "pnl_pct": pnl_pct,
             "leg_prices": leg_prices}


def simulate_sp_exit(entry_value: float,
                       legs: list,
                       signal_date: pd.Timestamp,
                       expiry_dt: pd.Timestamp,
                       today_dt: pd.Timestamp,
                       db: pd.DataFrame,
                       bp_high_series: pd.Series = None,
                       spot_series: pd.Series = None,
                       cfg: SPExitConfig = None) -> dict:
    """SP credit spread 现代化 exit.

    PnL% 用 max_risk (margin) 分母.
    """
    if cfg is None: cfg = SPExitConfig()
    # 算 max_risk
    spread_width = 0.0
    if len(legs) >= 2:
        ks = [l[2] for l in legs if "short" in l[0]]
        kl = [l[2] for l in legs if "put" in l[0] and "short" not in l[0]]
        if ks and kl: spread_width = abs(ks[0] - kl[0])
    max_risk = max(0.01, spread_width - entry_value)
    short_strike = ks[0] if (len(legs) >= 2 and "short" in legs[0][0]) \
                    else (legs[0][2] if legs else 0)

    profit_target = entry_value * (1 - cfg.profit_target_pct / 100)
    stop_loss = entry_value + abs(cfg.stop_loss_pct / 100) * max_risk

    def _pnl(cv): return (entry_value - cv) / max_risk * 100

    sig_d = pd.Timestamp(signal_date).normalize()
    days = sorted(set(db[db["code"].isin([l[1] for l in legs])]["date"].unique()))
    days_after = [d for d in days if pd.Timestamp(d) > sig_d]
    hold = 0
    leg_prices_today = []
    cur_value = entry_value
    for d in days_after:
        d_ts = pd.Timestamp(d)
        if d_ts > today_dt: break
        cur_total = 0.0; ok = True
        leg_prices_today = []
        for _lab, _code, _K, _qty in legs:
            r = db[(db["code"] == _code) & (db["date"] == d_ts)]
            if not len(r): ok = False; break
            _p = float(r.iloc[0]["close"])
            leg_prices_today.append((_lab, _p))
            cur_total += _qty * _p
        if not ok: continue
        cur_value = -cur_total  # credit spread net debit when closing
        hold += 1
        pnl_pct = _pnl(cur_value)
        # ① TP +50%
        if cur_value <= profit_target:
            return _sp_exit(d_ts, cur_value, hold, leg_prices_today, pnl_pct,
                              "+50% credit TP")
        # ② SL -50% margin
        if cur_value >= stop_loss:
            return _sp_exit(d_ts, cur_value, hold, leg_prices_today, pnl_pct,
                              "-50% margin SL")
        # ③ DTE-cliff (assignment risk 强平)
        days_to_exp = (expiry_dt - d_ts).days
        if days_to_exp <= cfg.dte_cliff_days:
            return _sp_exit(d_ts, cur_value, hold, leg_prices_today, pnl_pct,
                              f"DTE {days_to_exp}d 强平 (assignment risk)")
        # ④ Strike-defense (spot 接近 short strike 立即平)
        if short_strike > 0 and spot_series is not None and d_ts in spot_series.index:
            spot = float(spot_series.get(d_ts, 0))
            if spot > 0 and spot >= short_strike * cfg.strike_defense_buffer:
                return _sp_exit(d_ts, cur_value, hold, leg_prices_today, pnl_pct,
                                  f"spot ${spot:.0f} 接近 short ${short_strike:.0f} (strike defense)")
        # ⑤ Signal-reversal
        if bp_high_series is not None and d_ts in bp_high_series.index:
            bph = float(bp_high_series.get(d_ts, 0))
            if bph > cfg.signal_reversal_bp_high \
               and pnl_pct >= cfg.signal_reversal_min_profit:
                return _sp_exit(d_ts, cur_value, hold, leg_prices_today, pnl_pct,
                                  f"bp_high {bph:.2f} 反转 + {pnl_pct:.0f}% 早平")
        # ⑥ Expiry
        if d_ts >= expiry_dt:
            return _sp_exit(d_ts, cur_value, hold, leg_prices_today, pnl_pct,
                              "expiry")
    # OPEN
    if hold > 0:
        pnl_pct = _pnl(cur_value)
        return {"is_closed": False, "current_value": cur_value,
                 "hold_days": hold, "pnl_pct": pnl_pct,
                 "leg_prices": leg_prices_today, "max_risk": max_risk}
    return {"is_closed": False}


def _sp_exit(d_ts, cur_value, hold, leg_prices, pnl_pct, reason):
    return {"is_closed": True, "exit_date": d_ts,
             "exit_value": cur_value, "exit_reason": reason,
             "hold_days": hold, "pnl_pct": pnl_pct,
             "leg_prices": leg_prices}


# ── v3.7.232: expiry-aware 强平 (kline_db 缺到期合约时兜底) ─────────────
import re as _re

_OPT_CODE_RE = _re.compile(r"^US\.([A-Z]+?)(\d{6})([CP])(\d+)$")
_SPOT_CACHE = {}


def parse_option_code(code: str):
    """Parse OpenAPI option code → (asset, expiry_dt, opt_type, strike).

    e.g. ``US.GLD260515P445000`` → ("GLD", 2026-05-15, "P", 445.0).
    """
    m = _OPT_CODE_RE.match(str(code or ""))
    if not m: return None
    asset, yymmdd, typ, k = m.groups()
    try:
        exp = pd.Timestamp(year=2000 + int(yymmdd[:2]),
                            month=int(yymmdd[2:4]),
                            day=int(yymmdd[4:6]))
    except Exception:
        return None
    return asset, exp, typ, int(k) / 1000.0


def _load_spot_daily(asset: str):
    if asset in _SPOT_CACHE: return _SPOT_CACHE[asset]
    try:
        df = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                          index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        _SPOT_CACHE[asset] = df
    except Exception:
        _SPOT_CACHE[asset] = None
    return _SPOT_CACHE[asset]


def spot_close_on_or_before(asset: str, dt: pd.Timestamp):
    df = _load_spot_daily(asset)
    if df is None or df.empty: return None
    sub = df[df.index <= pd.Timestamp(dt).normalize()]
    if sub.empty: return None
    return float(sub.iloc[-1]["Close"])


def force_close_at_expiry(legs, entry_value: float,
                            today_dt, signal_date,
                            strategy_kind: str,
                            max_risk: float = None):
    """If ``today_dt > parsed_expiry`` → close at intrinsic value.

    Args:
        legs: ``[(label, code, K, qty), ...]``.
        strategy_kind:
            * ``"long_call"`` — sum of qty×intrinsic; pnl = (cur/entry - 1)
            * ``"credit_spread"`` — debit to close; pnl = (entry - cur) / max_risk
            * ``"long_vol"`` — STRADDLE, same math as ``long_call``
            * ``"iron_condor"`` — SHORT_VOL 4-leg IC; computes asymmetric
              ``max_risk = max(call_wing_width, put_wing_width) - entry_value``
              from the legs themselves, ignoring any ``max_risk`` kwarg.

    Returns:
        closed-position dict, or ``None`` if not past expiry / unparseable.
    """
    if not legs: return None
    parsed = parse_option_code(legs[0][1])
    if not parsed: return None
    asset, expiry_dt, _, _ = parsed
    today = pd.Timestamp(today_dt).normalize()
    if today <= expiry_dt: return None
    spot = spot_close_on_or_before(asset, expiry_dt)
    if spot is None: return None
    cur_total = 0.0; leg_prices = []
    for lab, code, K, qty in legs:
        p = parse_option_code(code)
        if not p:
            return None
        typ = p[2]
        intrinsic = max(spot - K, 0.0) if typ == "C" else max(K - spot, 0.0)
        leg_prices.append((lab, intrinsic))
        cur_total += qty * intrinsic
    if strategy_kind in ("credit_spread", "iron_condor"):
        cur_value = -cur_total  # debit to close (both branches share this)
        if strategy_kind == "iron_condor":
            # v3.7.239 + DEC-6: asymmetric IC, max loss is on the wider wing.
            # Parse strikes per option type; pair short with long on the same
            # side; use abs wing widths so symmetric IC degenerates correctly.
            call_short_K = call_long_K = put_short_K = put_long_K = None
            for lab, code, K, qty in legs:
                p = parse_option_code(code)
                if not p: continue
                typ = p[2]
                if typ == "C" and qty < 0: call_short_K = K
                elif typ == "C" and qty > 0: call_long_K = K
                elif typ == "P" and qty < 0: put_short_K = K
                elif typ == "P" and qty > 0: put_long_K = K
            call_wing = (abs(call_long_K - call_short_K)
                          if (call_short_K is not None and call_long_K is not None)
                          else 0.0)
            put_wing = (abs(put_short_K - put_long_K)
                         if (put_short_K is not None and put_long_K is not None)
                         else 0.0)
            wider = max(call_wing, put_wing)
            max_risk_eff = max(0.01, wider - float(entry_value))
        else:
            max_risk_eff = max_risk if (max_risk and max_risk > 0) \
                            else max(0.01, entry_value)
        pnl_pct = (entry_value - cur_value) / max_risk_eff * 100
    else:
        cur_value = cur_total
        pnl_pct = ((cur_value / entry_value) - 1) * 100 \
                    if abs(entry_value) > 0.01 else 0.0
    sig_d = pd.Timestamp(signal_date).normalize()
    return {
        "is_closed": True,
        "exit_date": expiry_dt,
        "exit_value": cur_value,
        "exit_reason": f"expiry intrinsic (db missing) spot={spot:.2f}",
        "pnl_pct": pnl_pct,
        "hold_days": max(0, (expiry_dt - sig_d).days),
        "leg_prices": leg_prices,
    }
