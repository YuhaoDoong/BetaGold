"""信号生成模块 — Band 构建 + V2 交易信号.

Hybrid Band:
  上界 = Daily(lag1)  — 灵敏, 及时捕捉顶部
  下界 = LagAvg(lag1,2,3) — 平滑, 过滤买入噪声

信号:
  Buy Call  = Bull + bp<0.30 + RV<=85%
  Sell Put  = Bull + bp<0.30 + RV>85%
  Exit      = bp>0.90 | Regime退出Bull
"""

import numpy as np
import pandas as pd


def build_band(range_df, gld_close,
               upper_lags=(1,), lower_lags=(1, 2, 3)):
    """构建 Hybrid Band, 返回 (upper_band, lower_band, bp)."""
    close = gld_close.reindex(range_df.index)

    uppers = []
    for lag in upper_lags:
        cl = close.shift(lag)
        pu = range_df["pred_upper_pct"].shift(lag)
        uppers.append(cl * (1 + pu / 100))

    lowers = []
    for lag in lower_lags:
        cl = close.shift(lag)
        pl = range_df["pred_lower_pct"].shift(lag)
        lowers.append(cl * (1 + pl / 100))

    upper_band = pd.concat(uppers, axis=1).mean(axis=1)
    lower_band = pd.concat(lowers, axis=1).mean(axis=1)
    bp = (close - lower_band) / (upper_band - lower_band)
    return upper_band, lower_band, bp


def compute_rv_pctile(rv, window=252):
    """RV 滚动百分位排名."""
    return rv.rolling(window, min_periods=60).rank(pct=True)


def generate_signals(bp_s, rv_p, is_bull):
    """
    V2 信号生成 (水平触发 + 期权类型区分).

    返回 (buy_call, sell_put, exit_sig) — 三个 bool Series.
    """
    rv_high = rv_p > 0.85

    buy_zone = is_bull & (bp_s < 0.30)
    buy_call = buy_zone & (~rv_high)
    sell_put = buy_zone & rv_high

    bull_exit = is_bull.shift(1).fillna(False) & (~is_bull)
    exit_sig = (bp_s > 0.90) | bull_exit

    return buy_call, sell_put, exit_sig
