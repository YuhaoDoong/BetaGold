"""真实期权数据建模 — 桥接 RV-based 估算与 Moomoo/yfinance 实际报价.

桥接两个世界:
  RV-based 简化模型 (现有)         真实期权报价 (本模块)
  ──────────────────────         ─────────────────────
  cost = RV × √(T/252)           cost = 实际 Long Call + Long Put
  win = max_move > cost          win = max_move > 真实 BE
  假设 IV ≈ RV                    实测 IV (反推自 premium)

提供功能:
  1. 反推 IV (从期权 premium 反推)
  2. ATM Straddle 真实 cost 估算 (用 IV 不是 RV)
  3. Breakeven 计算
  4. 历史窗口突破概率 (用真实 BE 替换 RV 估算)
"""
from typing import Optional
from datetime import date, datetime, timedelta
import math
import pandas as pd
import numpy as np


def implied_vol_from_straddle(call_price: float, put_price: float,
                                spot: float, dte: int) -> float:
    """从 ATM Long Straddle premium 反推隐含波动率 (IV).

    简化公式: P ≈ 0.8 × σ × S × √(T/365)
    σ ≈ P / (0.8 × S × √(T/365))

    对 ATM 期权 ±5% 内精度足够; 远 OTM 需用 Black-Scholes 数值反推.
    """
    if dte <= 0 or spot <= 0:
        return float("nan")
    total = call_price + put_price
    T = dte / 365.0
    return total / (0.8 * spot * math.sqrt(T))


def straddle_breakeven(call_price: float, put_price: float,
                         strike: float) -> tuple:
    """Long Straddle 盈亏平衡点.

    Returns: (be_low, be_high)
    """
    total = call_price + put_price
    return (strike - total, strike + total)


def required_move_pct(call_price: float, put_price: float,
                        spot: float) -> float:
    """达到盈亏平衡所需价格移动百分比."""
    return (call_price + put_price) / spot * 100


def atm_straddle_cost_iv(spot: float, iv: float, dte: int) -> float:
    """根据 IV 估算 ATM Long Straddle premium (反向公式).

    用于检验: 给定 IV, 应该付多少 premium?
    """
    T = dte / 365.0
    return 0.8 * iv * spot * math.sqrt(T)


def historical_breakout_prob(close: pd.Series, high: pd.Series,
                                low: pd.Series, threshold_pct: float,
                                dte: int, lookback_days: Optional[int] = None
                                ) -> dict:
    """历史窗口期内价格突破阈值的频率.

    Args:
        close/high/low: 价格序列
        threshold_pct: 阈值百分比 (e.g. 8.22)
        dte: 持仓天数 (e.g. 16)
        lookback_days: 仅用最近 N 天数据 (None = 全样本)

    Returns: {n_windows, n_breaks, prob, mean_max_move}
    """
    if lookback_days is not None:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        close = close[close.index >= cutoff]
        high = high[high.index >= cutoff]
        low = low[low.index >= cutoff]

    n_breaks = 0
    n_windows = 0
    max_moves = []
    for i in range(len(close) - dte):
        c0 = close.iloc[i]
        if c0 <= 0:
            continue
        win_high = high.iloc[i+1:i+dte+1].max()
        win_low = low.iloc[i+1:i+dte+1].min()
        move_up = (win_high / c0 - 1) * 100
        move_down = (1 - win_low / c0) * 100
        max_move = max(move_up, move_down)
        max_moves.append(max_move)
        if max_move > threshold_pct:
            n_breaks += 1
        n_windows += 1

    if n_windows == 0:
        return {"n_windows": 0, "n_breaks": 0, "prob": float("nan"),
                "mean_max_move": float("nan")}
    return {
        "n_windows": n_windows,
        "n_breaks": n_breaks,
        "prob": n_breaks / n_windows,
        "mean_max_move": float(np.mean(max_moves)),
        "median_max_move": float(np.median(max_moves)),
    }


def compare_real_vs_model(real_call: float, real_put: float,
                            spot: float, strike: float, dte: int,
                            rv: float, close: pd.Series,
                            high: pd.Series, low: pd.Series) -> dict:
    """对比真实期权 vs RV-based 模型, 返回详细诊断.

    用于实盘验证: 用户提供真实 entry premium, 系统自动算偏差和期望胜率.
    """
    real_total = real_call + real_put
    real_pct = real_total / spot * 100
    iv_implied = implied_vol_from_straddle(real_call, real_put, spot, dte)
    rv_decimal = rv / 100
    iv_rv_ratio = iv_implied / rv_decimal if rv_decimal > 0 else float("nan")

    rv_estimate = atm_straddle_cost_iv(spot, rv_decimal, dte)
    iv_estimate = atm_straddle_cost_iv(spot, iv_implied, dte)

    be_low, be_high = straddle_breakeven(real_call, real_put, strike)
    move_needed = required_move_pct(real_call, real_put, spot)

    # 历史突破频率
    prob_all = historical_breakout_prob(close, high, low, move_needed, dte)
    prob_5y = historical_breakout_prob(close, high, low, move_needed, dte,
                                          lookback_days=5*365)
    prob_1y = historical_breakout_prob(close, high, low, move_needed, dte,
                                          lookback_days=365)

    return {
        "real_premium": real_total,
        "real_premium_pct": real_pct,
        "iv_implied": iv_implied * 100,  # 百分比
        "rv_input": rv,
        "iv_rv_ratio": iv_rv_ratio,
        "event_premium_pct": (iv_rv_ratio - 1) * 100,
        "rv_estimate_dollar": rv_estimate,
        "rv_estimate_pct": rv_estimate / spot * 100,
        "model_underestimate_pct": (real_total / rv_estimate - 1) * 100
            if rv_estimate > 0 else float("nan"),
        "be_low": be_low,
        "be_high": be_high,
        "move_needed_pct": move_needed,
        "breakout_prob_all": prob_all["prob"],
        "breakout_prob_5y": prob_5y["prob"],
        "breakout_prob_1y": prob_1y["prob"],
        "n_windows_5y": prob_5y["n_windows"],
    }


# 示例用法 (用户实仓):
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/yhdong/GoldDash")
    slv = pd.read_csv("/Users/yhdong/Gold/data/raw/market/slv.csv",
                       index_col=0, parse_dates=True)
    ret = slv["Close"].pct_change()
    rv = (ret.rolling(10).std() * (252 ** 0.5) * 100).dropna().iloc[-1]

    result = compare_real_vs_model(
        real_call=2.58, real_put=2.74,
        spot=64.70, strike=65.0, dte=16,
        rv=rv,
        close=slv["Close"], high=slv["High"], low=slv["Low"],
    )
    print("=== 实仓 vs 模型对比 ===")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
