"""1h 信号系统 — 基于 1h Band Position 的盘中信号.

与 v1.0 信号的区别:
  - v1.0: 收盘价计算 bp → 收盘后才知道信号 → 次日执行
  - v2.0: 每根 1h K线实时计算 bp → 盘中触发 → 即时执行

信号逻辑:
  BUY CALL  = Bull(日线) + bp_1h < 0.30 + RV≤85%
  SELL PUT  = Bull(日线) + bp_1h < 0.30 + RV>85%
  EXIT      = bp_1h > 0.90 | Regime退出Bull

Band 构建 (1h 版):
  upper = close_1h(t-1) * (1 + pred_upper%(t-1) / 100)
  lower = avg(close_1h(t-k) * (1 + pred_lower%(t-k) / 100)), k=1,2,3
"""

import numpy as np
import pandas as pd


def build_band_1h(pred_df, close_1h, upper_lags=(1,), lower_lags=(1, 2, 3),
                  horizon="7h"):
    """构建 1h Hybrid Band.

    Args:
        pred_df: DataFrame with pred_{horizon}_upper_pct / lower_pct columns
        close_1h: Series, 1h close prices
        horizon: "7h" or "35h"

    Returns: (upper_band, lower_band, bp) — all pd.Series
    """
    close = close_1h.reindex(pred_df.index)
    u_col = f"pred_{horizon}_upper_pct"
    l_col = f"pred_{horizon}_lower_pct"

    uppers = []
    for lag in upper_lags:
        cl = close.shift(lag)
        pu = pred_df[u_col].shift(lag)
        uppers.append(cl * (1 + pu / 100))

    lowers = []
    for lag in lower_lags:
        cl = close.shift(lag)
        pl = pred_df[l_col].shift(lag)
        lowers.append(cl * (1 + pl / 100))

    upper_band = pd.concat(uppers, axis=1).mean(axis=1)
    lower_band = pd.concat(lowers, axis=1).mean(axis=1)
    bp = (close - lower_band) / (upper_band - lower_band)
    return upper_band, lower_band, bp


def generate_signals_1h(bp_1h, regime_daily, rv_pctile_daily,
                        buy_threshold=0.30, exit_threshold=0.90,
                        rv_high_pctile=0.85):
    """生成 1h 盘中信号.

    Args:
        bp_1h: Series, 1h band position
        regime_daily: Series (index=date, values=Bull/Bear/Mixed)
        rv_pctile_daily: Series (index=date, values=0~1)

    Returns: (buy_call, sell_put, exit_sig) — bool Series, index=1h datetime
    """
    # 对齐日线 Regime/RV 到 1h index (每根K线继承当日值)
    dates_1h = bp_1h.index.normalize()
    is_bull = regime_daily.reindex(dates_1h).values == "Bull"
    is_bull = pd.Series(is_bull, index=bp_1h.index)

    rv_p = rv_pctile_daily.reindex(dates_1h).values
    rv_p = pd.Series(rv_p, index=bp_1h.index).fillna(0.5)
    rv_high = rv_p > rv_high_pctile

    buy_zone = is_bull & (bp_1h < buy_threshold)
    buy_call = buy_zone & (~rv_high)
    sell_put = buy_zone & rv_high

    # Exit: bp > threshold or regime exits bull
    bull_prev = is_bull.shift(1).fillna(False)
    bull_exit = bull_prev & (~is_bull)
    exit_sig = (bp_1h > exit_threshold) | bull_exit

    return buy_call, sell_put, exit_sig


def backtest_1h(close_1h, high_1h, bp_1h,
                buy_call, sell_put, exit_sig,
                max_hold_hours=70, pullback_gain=2.0, pullback_dd=1.5):
    """1h 信号回测.

    Args:
        max_hold_hours: 最长持仓 (70h ≈ 10 交易日)

    Returns: list of trade dicts
    """
    trades = []
    in_trade = False
    entry_dt = entry_price = sig_type = None
    peak = hold = 0

    for dt in bp_1h.index:
        c = close_1h.get(dt, np.nan)
        h = high_1h.get(dt, np.nan)
        if np.isnan(c):
            continue

        if not in_trade:
            if buy_call.get(dt, False):
                in_trade = True
                entry_dt, entry_price, sig_type = dt, c, "BUY CALL"
                peak, hold = c, 0
            elif sell_put.get(dt, False):
                in_trade = True
                entry_dt, entry_price, sig_type = dt, c, "SELL PUT"
                peak, hold = c, 0
        else:
            hold += 1
            peak = max(peak, h if not np.isnan(h) else c)

            should_exit = False
            exit_type = "Timeout"

            if exit_sig.get(dt, False):
                should_exit, exit_type = True, "BandExit"
            else:
                ppct = (peak / entry_price - 1) * 100
                dd = (peak - c) / peak * 100
                if ppct > pullback_gain and dd >= pullback_dd:
                    should_exit, exit_type = True, "Pullback"

            if hold >= max_hold_hours:
                should_exit = True

            if should_exit:
                g = (c / entry_price - 1) * 100
                trades.append({
                    "entry_date": entry_dt, "exit_date": dt,
                    "sig_type": sig_type, "exit_type": exit_type,
                    "entry_price": entry_price, "exit_price": c,
                    "gain": g, "hold_hours": hold,
                })
                in_trade = False

    return trades
