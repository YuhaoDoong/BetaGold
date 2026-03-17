"""v2.1 信号系统 — v1.0 Band + 盘中触发 + 1h 确认.

设计:
  - Band 预测: 复用 v1.0 日线模型 (20年数据, 校准可靠)
  - 触发判断: 用 GLD 盘中价格 (含盘前盘后 04:00~19:00 ET)
             + GC=F 补充 19:00~04:00 盲区 (换算为 GLD 等价)
  - 1h 确认 (方向A): RSI/动量/止跌形态 → 提高入场置信度
  - 持仓: 2-5天, 适合期权交易

信号逻辑:
  BUY CALL  = Bull + 盘中价 < bp=0.30 + RV≤85% [+ 1h确认]
  SELL PUT  = Bull + 盘中价 < bp=0.30 + RV>85% [+ 1h确认]
  EXIT      = 盘中价 > bp=0.90 | Regime退出Bull
"""

import numpy as np
import pandas as pd
from datetime import timezone, timedelta

# 可配置时区
DEFAULT_TZ_OFFSET = 8  # SGT/北京 = UTC+8


def compute_1h_confirmation(close_1h, high_1h, low_1h, lookback=6):
    """计算 1h 级别入场确认指标.

    Args:
        close_1h: GLD 或 GC=F 的 1h close
        lookback: 回看K线数 (6根≈半天)

    Returns: DataFrame with confirmation signals.
    """
    c = close_1h

    # 1h RSI (14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    # 短期动量 (最近 lookback 根的方向)
    momentum = c.pct_change(lookback)

    # 止跌信号: 最近 lookback 根中最低点在前半段 (已经反弹)
    rolling_low_pos = low_1h.rolling(lookback).apply(
        lambda x: np.argmin(x) / len(x) if len(x) > 0 else 0.5,
        raw=True)

    result = pd.DataFrame(index=c.index)
    result["rsi_1h"] = rsi
    result["momentum_6h"] = momentum
    result["low_position"] = rolling_low_pos  # 0=低点在最近, 1=低点在最早

    # 综合确认: RSI 超卖 + 低点已过 (低点在前半段)
    result["buy_confirmed"] = (rsi < 35) | (rolling_low_pos < 0.4)
    # 卖出确认: RSI 超买 + 高点已过
    rolling_high_pos = high_1h.rolling(lookback).apply(
        lambda x: np.argmax(x) / len(x) if len(x) > 0 else 0.5,
        raw=True)
    result["exit_confirmed"] = (rsi > 70) | (rolling_high_pos < 0.4)

    return result


def generate_signals_v2(close_daily, high_daily, low_daily,
                        upper_band, lower_band, regime, rv_pctile,
                        close_1h=None, high_1h=None, low_1h=None,
                        use_1h_confirm=False):
    """v2.1 信号生成 — 日线 Band + 盘中 High/Low + 可选 1h 确认.

    Args:
        close/high/low_daily: 日线 OHLC
        upper_band, lower_band: v1.0 Band (日线)
        regime: 日线 Regime Series
        rv_pctile: 日线 RV percentile Series
        close/high/low_1h: GLD 1h K线 (可选, 用于确认)
        use_1h_confirm: 是否使用 1h 确认过滤

    Returns: DataFrame with columns:
        date, close, high, low, bp_close, bp_low, bp_high,
        buy_signal, buy_type, exit_signal, signal_text,
        1h_confirmed (if use_1h_confirm)
    """
    bp_dates = upper_band.dropna().index.intersection(lower_band.dropna().index)
    records = []

    # 1h 确认指标
    confirm_1h = None
    if use_1h_confirm and close_1h is not None:
        confirm_1h = compute_1h_confirmation(close_1h, high_1h, low_1h)

    for d in bp_dates:
        ub = upper_band[d]
        lb = lower_band[d]
        if ub <= lb:
            continue

        c = close_daily.get(d, np.nan)
        h = high_daily.get(d, np.nan)
        lo = low_daily.get(d, np.nan)
        if np.isnan(c):
            continue

        bp_close = (c - lb) / (ub - lb)
        bp_low = (lo - lb) / (ub - lb)
        bp_high = (h - lb) / (ub - lb)
        bp030 = lb + 0.30 * (ub - lb)
        bp090 = lb + 0.90 * (ub - lb)

        is_bull = regime.get(d, "?") == "Bull"
        rv = rv_pctile.get(d, 0.5)

        # 买入: 盘中 low 触及 bp=0.30
        buy_sig = False
        buy_type = None
        if is_bull and bp_low < 0.30:
            buy_sig = True
            buy_type = "BUY CALL" if rv <= 0.85 else "SELL PUT"

        # 退出: 盘中 high 触及 bp=0.90
        exit_sig = bp_high > 0.90

        # Regime 退出
        if d in regime.index:
            loc = regime.index.get_loc(d)
            if loc > 0:
                prev = regime.iloc[loc - 1]
                if prev == "Bull" and regime[d] != "Bull":
                    exit_sig = True

        # 1h 确认
        confirmed = True
        if use_1h_confirm and confirm_1h is not None:
            # 找到当天的 1h 确认信号
            day_mask = confirm_1h.index.normalize() == d
            day_confirm = confirm_1h[day_mask]
            if len(day_confirm) > 0:
                if buy_sig:
                    confirmed = day_confirm["buy_confirmed"].any()
                elif exit_sig:
                    confirmed = day_confirm["exit_confirmed"].any()

        rec = {
            "date": d,
            "close": c, "high": h, "low": lo,
            "upper": ub, "lower": lb,
            "bp_close": bp_close, "bp_low": bp_low, "bp_high": bp_high,
            "bp030_price": bp030, "bp090_price": bp090,
            "buy_signal": buy_sig, "buy_type": buy_type,
            "exit_signal": exit_sig,
            "regime": regime.get(d, "?"), "rv_pctile": rv,
        }
        if use_1h_confirm:
            rec["confirmed_1h"] = confirmed

        # 综合信号文本
        parts = []
        if buy_sig:
            parts.append(buy_type)
            if use_1h_confirm and not confirmed:
                parts.append("(未确认)")
        if exit_sig:
            parts.append("EXIT")
            if use_1h_confirm and not confirmed and not buy_sig:
                parts.append("(未确认)")
        rec["signal_text"] = " + ".join(parts) if parts else ""

        records.append(rec)

    return pd.DataFrame(records).set_index("date")


def format_time_for_tz(dt, tz_offset=DEFAULT_TZ_OFFSET):
    """将 ET 时间转为指定时区显示."""
    if dt.tzinfo is None:
        # 假设输入是 ET (UTC-5 或 UTC-4)
        et_offset = -5  # 简化, 实际需处理夏令时
        utc = dt - timedelta(hours=et_offset)
        local = utc + timedelta(hours=tz_offset)
        return local
    return dt
