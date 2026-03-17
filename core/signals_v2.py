"""v2.2 信号系统 — v1.0 Band + 1h 精确入场 + 1h 止盈.

设计:
  Band 预测: 复用 v1.0 日线模型 (20年数据, 校准可靠)
  入场: 盘中 1h 价格触及 bp=0.30 → 等待止跌确认 → 入场
  止盈: 盘中 1h 跟踪最高价 → 回撤超阈值 → 止盈
  退出: bp>0.90 (盘中) | Pullback (1h) | Timeout | Regime退出

关键改进 (vs v2.1):
  - 入场不再"一触即发", 而是等 1h 确认止跌 (RSI反弹/K线反转)
  - 止盈/止损基于 1h 盘中最高价回撤, 不等收盘
  - 所有信号都基于 1h 时间精度
"""

import numpy as np
import pandas as pd
from datetime import timedelta

DEFAULT_TZ_OFFSET = 8  # SGT


# ── 1h 技术指标 ──

def _rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=3).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=3).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _is_reversal_bar(close, low, lookback=3):
    """止跌反转: 当前 close > 前 lookback 根的最低 close, 且 low 是近期最低."""
    prev_min_close = close.rolling(lookback, min_periods=1).min().shift(1)
    is_low_recent = low <= low.rolling(lookback * 2, min_periods=1).min().shift(1) * 1.005
    rebounded = close > prev_min_close
    return is_low_recent & rebounded


# ── 日线信号 (基础层, 同 v2.1) ──

def generate_daily_signals(close_d, high_d, low_d,
                           upper_band, lower_band,
                           regime, rv_pctile):
    """日线级别信号: Band + H/L 触发 (无 1h 确认).

    返回每天的 Band 参数 + 买入/退出触发状态.
    """
    bp_dates = upper_band.dropna().index.intersection(lower_band.dropna().index)
    records = []

    for d in bp_dates:
        ub, lb = upper_band[d], lower_band[d]
        if ub <= lb:
            continue
        c = close_d.get(d, np.nan)
        h = high_d.get(d, np.nan)
        lo = low_d.get(d, np.nan)
        if np.isnan(c):
            continue

        bp_close = (c - lb) / (ub - lb)
        bp_low = (lo - lb) / (ub - lb)
        bp_high = (h - lb) / (ub - lb)
        bp030 = lb + 0.30 * (ub - lb)
        bp090 = lb + 0.90 * (ub - lb)

        is_bull = regime.get(d, "?") == "Bull"
        rv = rv_pctile.get(d, 0.5)

        buy_sig = is_bull and bp_low < 0.30
        buy_type = None
        if buy_sig:
            buy_type = "BUY CALL" if rv <= 0.85 else "SELL PUT"

        exit_sig = bp_high > 0.90
        # Regime 退出
        if d in regime.index:
            loc = regime.index.get_loc(d)
            if loc > 0 and regime.iloc[loc-1] == "Bull" and regime[d] != "Bull":
                exit_sig = True

        records.append({
            "date": d, "close": c, "high": h, "low": lo,
            "upper": ub, "lower": lb,
            "bp_close": bp_close, "bp_low": bp_low, "bp_high": bp_high,
            "bp030_price": bp030, "bp090_price": bp090,
            "buy_signal": buy_sig, "buy_type": buy_type,
            "exit_signal": exit_sig,
            "regime": regime.get(d, "?"), "rv_pctile": rv,
        })

    df = pd.DataFrame(records).set_index("date")

    # 信号文本
    parts_list = []
    for _, r in df.iterrows():
        parts = []
        if r["buy_signal"]:
            parts.append(r["buy_type"])
        if r["exit_signal"]:
            parts.append("EXIT")
        parts_list.append(" + ".join(parts))
    df["signal_text"] = parts_list

    return df


# ── 1h 精确入场 ──

def compute_1h_entry_signals(gld_1h, daily_signals,
                             rsi_threshold=35, lookback=3):
    """在日线买入信号日, 用 1h 数据找最佳入场时刻.

    逻辑: 日线 low 触及 bp=0.30 的日子里, 找 1h 级别止跌确认:
      - 1h close < bp030 (价格在买入区内)
      - 1h RSI < rsi_threshold (超卖)
      - 止跌反转 K线 (low 是近期最低, 但 close 回升)

    Returns: DataFrame, index=1h datetime, 含 entry_signal, entry_price, entry_type
    """
    if gld_1h is None:
        return pd.DataFrame()

    close_1h = gld_1h["Close"]
    low_1h = gld_1h["Low"]
    rsi_1h = _rsi(close_1h)
    reversal = _is_reversal_bar(close_1h, low_1h, lookback)

    buy_days = daily_signals[daily_signals["buy_signal"]].index

    entries = []
    for day in buy_days:
        bp030 = daily_signals.loc[day, "bp030_price"]
        buy_type = daily_signals.loc[day, "buy_type"]

        # 当天 1h K线
        day_mask = close_1h.index.normalize() == day
        day_bars = close_1h[day_mask]

        for dt in day_bars.index:
            c = close_1h.get(dt, np.nan)
            r = rsi_1h.get(dt, 50)
            rev = reversal.get(dt, False)

            if np.isnan(c):
                continue

            in_buy_zone = c < bp030
            rsi_oversold = r < rsi_threshold
            confirmed = in_buy_zone and (rsi_oversold or rev)

            entries.append({
                "datetime": dt, "date": day,
                "price": c, "bp030": bp030,
                "rsi": r, "reversal": rev,
                "in_zone": in_buy_zone,
                "confirmed": confirmed,
                "type": buy_type,
            })

    return pd.DataFrame(entries).set_index("datetime") if entries else pd.DataFrame()


# ── 1h 止盈/止损 ──

def compute_1h_exit_signals(gld_1h, daily_signals,
                            pullback_gain=2.0, pullback_dd=1.5,
                            trailing_dd=1.5):
    """基于 1h 数据的精确退出信号.

    三种退出:
      1. BandExit: 1h close > bp090 价位
      2. Pullback: 从持仓期间 1h 最高价回撤超阈值
      3. TrailingStop: 从近期 1h 峰值回撤超阈值 (不需要持仓状态)

    Returns: DataFrame, index=1h datetime
    """
    if gld_1h is None:
        return pd.DataFrame()

    close_1h = gld_1h["Close"]
    high_1h = gld_1h["High"]

    exit_days = daily_signals[daily_signals["exit_signal"]].index

    exits = []

    # BandExit: 1h 级别触及 bp090
    for day in exit_days:
        bp090 = daily_signals.loc[day, "bp090_price"]
        day_mask = high_1h.index.normalize() == day
        day_highs = high_1h[day_mask]

        for dt in day_highs.index:
            h = high_1h.get(dt, 0)
            if h > bp090:
                exits.append({
                    "datetime": dt, "date": day,
                    "price": close_1h.get(dt, h),
                    "trigger_price": bp090,
                    "type": "BandExit",
                    "detail": f"High ${h:.1f} > bp090 ${bp090:.1f}",
                })
                break  # 一天只记第一次触发

    # TrailingStop: 滚动峰值回撤 (覆盖所有日子, 不只是 exit_days)
    # 用 5天 (约 85根 1h) 滚动窗口
    rolling_peak = high_1h.rolling(85, min_periods=1).max()
    drawdown_from_peak = (rolling_peak - close_1h) / rolling_peak * 100
    peak_gain_pct = (rolling_peak / close_1h.shift(85).fillna(close_1h.iloc[0]) - 1) * 100

    # 触发: 从峰值涨了 >pullback_gain% 后, 回撤 >pullback_dd%
    pullback_trigger = (peak_gain_pct > pullback_gain) & (drawdown_from_peak > pullback_dd)

    # 只在有信号价值的日子输出
    all_dates = daily_signals.index
    for day in all_dates:
        day_mask = pullback_trigger.index.normalize() == day
        day_pb = pullback_trigger[day_mask]
        if day_pb.any():
            first_trigger = day_pb[day_pb].index[0]
            if first_trigger not in [e["datetime"] for e in exits]:
                exits.append({
                    "datetime": first_trigger, "date": day,
                    "price": close_1h.get(first_trigger, 0),
                    "trigger_price": rolling_peak.get(first_trigger, 0),
                    "type": "Pullback",
                    "detail": f"Peak ${rolling_peak.get(first_trigger,0):.1f}"
                              f" dd={drawdown_from_peak.get(first_trigger,0):.1f}%",
                })

    return pd.DataFrame(exits).set_index("datetime") if exits else pd.DataFrame()


# ── 综合信号 (日线 + 1h) ──

def generate_signals_v22(close_d, high_d, low_d,
                         upper_band, lower_band,
                         regime, rv_pctile,
                         gld_1h=None):
    """v2.2 综合信号: 日线 Band + 1h 入场确认 + 1h 止盈.

    Returns:
        daily_signals: DataFrame (日线级别, 含基础信号)
        entry_1h: DataFrame (1h 入场时刻, 含确认状态)
        exit_1h: DataFrame (1h 退出时刻, 含退出类型)
    """
    daily = generate_daily_signals(
        close_d, high_d, low_d, upper_band, lower_band,
        regime, rv_pctile)

    entry_1h = compute_1h_entry_signals(gld_1h, daily) \
        if gld_1h is not None else pd.DataFrame()
    exit_1h = compute_1h_exit_signals(gld_1h, daily) \
        if gld_1h is not None else pd.DataFrame()

    return daily, entry_1h, exit_1h
