"""v2.2 信号系统 — v1.0 Band + 盘中H/L入场 + 多尺度止盈.

架构:
  Band: v1.0 日线模型 (20年训练, 校准可靠)
  入场: 日线 Low 触及 bp < BUY_BP 即入场
  止盈: 在可配置的时间尺度上检测退出 (默认 12h)
    优先级: BandExit (bp>EXIT_BP) > Pullback (峰值回撤) > MACD弱化 > Timeout
  持仓: 2-5天

注意:
  - 信号基于金价, 不限定交易品种 (期权/期货/现货均可)
  - 期权策略是独立的推荐层, 不影响核心信号
  - 时间不限定美盘, GLD 含盘前盘后, GC=F 覆盖全球时段

参数 (全部可配置):
  EXIT_TIMEFRAME = "12h"   # 止盈检测尺度 (1h/2h/4h/8h/12h)
  PULLBACK_GAIN  = 2.0     # Pullback 触发: 持仓期涨幅>N%
  PULLBACK_DD    = 1.5     # Pullback 触发: 从峰值回撤>N%
  MACD_MIN_GAIN  = 1.0     # MACD弱化止盈: 至少盈利>N%
  MAX_HOLD_DAYS  = 10      # 最大持仓天数
  BUY_BP         = 0.30    # 买入阈值 (Band Position)
  EXIT_BP        = 0.90    # 退出阈值
"""

import numpy as np
import pandas as pd
from datetime import timedelta

# ── 可配置参数 ──
EXIT_TIMEFRAME = "12h"
PULLBACK_GAIN = 2.0
PULLBACK_DD = 1.5
MACD_MIN_GAIN = 1.0
MAX_HOLD_DAYS = 10
BUY_BP = 0.30
EXIT_BP = 0.90
STOP_LOSS_PCT = 3.0       # 单笔止损: 入场后跌超N%
CONSECUTIVE_STOP = 2      # 连续止损N笔后暂停买入信号
DEFAULT_TZ_OFFSET = 8     # SGT


def _macd_hist(c, fast=12, slow=26, sig=9):
    ef = c.ewm(span=fast, min_periods=1).mean()
    es = c.ewm(span=slow, min_periods=1).mean()
    ml = ef - es
    return ml - ml.ewm(span=sig, min_periods=1).mean()


def resample_1h(gld_1h, timeframe):
    """将 1h 数据 resample 到指定时间尺度."""
    if timeframe == "1h":
        return gld_1h
    return gld_1h.resample(timeframe).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()


def generate_daily_signals(close_d, high_d, low_d,
                           upper_band, lower_band,
                           regime, rv_pctile,
                           buy_bp=BUY_BP, exit_bp=EXIT_BP):
    """日线级别信号: v1.0 Band + H/L 触发.

    每天输出: Band 参数 + 买入/退出触发状态 + 阈值价位.
    """
    bp_dates = upper_band.dropna().index.intersection(
        lower_band.dropna().index)
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
        bp030 = lb + buy_bp * (ub - lb)
        bp090 = lb + exit_bp * (ub - lb)

        is_bull = regime.get(d, "?") == "Bull"
        rv = rv_pctile.get(d, 0.5)

        buy_sig = is_bull and bp_low < buy_bp
        buy_type = None
        if buy_sig:
            buy_type = "BUY CALL" if rv <= 0.85 else "SELL PUT"

        exit_sig = bp_high > exit_bp
        # Regime 退出
        if d in regime.index:
            loc = regime.index.get_loc(d)
            if loc > 0 and regime.iloc[loc - 1] == "Bull" \
                    and regime[d] != "Bull":
                exit_sig = True

        parts = []
        if buy_sig:
            parts.append(buy_type)
        if exit_sig:
            parts.append("EXIT")

        records.append({
            "date": d, "close": c, "high": h, "low": lo,
            "upper": ub, "lower": lb,
            "bp_close": bp_close, "bp_low": bp_low, "bp_high": bp_high,
            "bp030_price": bp030, "bp090_price": bp090,
            "buy_signal": buy_sig, "buy_type": buy_type,
            "exit_signal": exit_sig,
            "regime": regime.get(d, "?"), "rv_pctile": rv,
            "signal_text": " + ".join(parts),
        })

    return pd.DataFrame(records).set_index("date")


def run_backtest(close_d, high_d, low_d,
                 upper_band, lower_band,
                 regime, rv_pctile,
                 gld_1h=None,
                 exit_timeframe=EXIT_TIMEFRAME,
                 pullback_gain=PULLBACK_GAIN,
                 pullback_dd=PULLBACK_DD,
                 macd_min_gain=MACD_MIN_GAIN,
                 max_hold_days=MAX_HOLD_DAYS,
                 buy_bp=BUY_BP, exit_bp=EXIT_BP,
                 start_date=None):
    """v2.2 回测: 日线入场 + 可配置时间尺度止盈.

    Returns: list of trade dicts
    """
    bp_dates = upper_band.dropna().index.intersection(
        lower_band.dropna().index)
    if start_date:
        bp_dates = bp_dates[bp_dates >= start_date]

    # 准备止盈用的 K 线数据
    tf_df = None
    tf_macd = None
    if gld_1h is not None:
        tf_df = resample_1h(gld_1h, exit_timeframe)
        tf_macd = _macd_hist(tf_df["Close"])

    trades = []
    in_trade = False
    entry_dt = entry_price = entry_type = None
    peak = 0
    consecutive_stops = 0  # 连续止损计数

    for d in bp_dates:
        u, l = upper_band.get(d, np.nan), lower_band.get(d, np.nan)
        if np.isnan(u) or np.isnan(l) or u <= l:
            continue
        c, h, lo = close_d[d], high_d[d], low_d[d]
        bp_lo = (lo - l) / (u - l)
        bp_hi = (h - l) / (u - l)
        bp030 = l + buy_bp * (u - l)
        bp090 = l + exit_bp * (u - l)
        is_bull = regime.get(d, "?") == "Bull"
        rv = rv_pctile.get(d, 0.5)

        # ── 退出 ──
        exit_type = exit_price = None

        if in_trade and tf_df is not None:
            day_bars = tf_df[tf_df.index.normalize() == d]
            for dt_bar in day_bars.index:
                h_bar = tf_df["High"].get(dt_bar, 0)
                c_bar = tf_df["Close"].get(dt_bar, 0)
                peak = max(peak, h_bar)

                # 0) StopLoss: 从入场价直接跌超阈值
                if entry_price > 0:
                    loss = (c_bar / entry_price - 1) * 100
                    if loss < -STOP_LOSS_PCT:
                        exit_type, exit_price = "StopLoss", c_bar
                        break

                # 1) BandExit
                if h_bar > bp090:
                    exit_type, exit_price = "BandExit", bp090
                    break

                # 2) Pullback
                if entry_price > 0:
                    gain = (peak / entry_price - 1) * 100
                    dd = (peak - c_bar) / peak * 100
                    if gain > pullback_gain and dd >= pullback_dd:
                        exit_type, exit_price = "Pullback", c_bar
                        break

                # 3) MACD 弱化
                m = tf_macd.get(dt_bar, 0)
                mp = tf_macd.shift(1).get(dt_bar, 0)
                if mp > 0 and m < 0 and entry_price > 0:
                    gain = (c_bar / entry_price - 1) * 100
                    if gain > macd_min_gain:
                        exit_type, exit_price = "MACD", c_bar
                        break

        elif in_trade:
            # 无 1h 数据: 用日线 H/L
            peak = max(peak, h)

            # StopLoss (日线)
            if entry_price > 0:
                loss = (lo / entry_price - 1) * 100
                if loss < -STOP_LOSS_PCT:
                    exit_type, exit_price = "StopLoss", lo

            if not exit_type and bp_hi > exit_bp:
                exit_type, exit_price = "BandExit", bp090
            elif not exit_type and entry_price > 0:
                gain = (peak / entry_price - 1) * 100
                dd = (peak - c) / peak * 100
                if gain > pullback_gain and dd >= pullback_dd:
                    exit_type, exit_price = "Pullback", c

        # Timeout
        if in_trade and not exit_type:
            if (d - entry_dt).days >= max_hold_days:
                exit_type, exit_price = "Timeout", c

        if in_trade and exit_type:
            gain = (exit_price / entry_price - 1) * 100
            trades.append({
                "entry_date": entry_dt, "exit_date": d,
                "entry_price": entry_price, "exit_price": exit_price,
                "type": entry_type, "exit_type": exit_type,
                "gain": gain,
                "hold_days": (d - entry_dt).days,
                "peak": peak,
            })
            in_trade = False

            # 连续止损计数
            if exit_type == "StopLoss":
                consecutive_stops += 1
            else:
                consecutive_stops = 0

        # ── 入场 ──
        # 连续止损后暂停买入
        if consecutive_stops >= CONSECUTIVE_STOP:
            # 等下一个 BandExit 或 bp > 0.50 才恢复
            if bp_hi > exit_bp or (c - l) / (u - l) > 0.50:
                consecutive_stops = 0  # 恢复
            else:
                continue  # 暂停

        if not in_trade and is_bull and bp_lo < buy_bp:
            entry_type = "BUY CALL" if rv <= 0.85 else "SELL PUT"
            entry_price = min(bp030, lo)
            entry_dt = d
            peak = h
            in_trade = True

    return trades
