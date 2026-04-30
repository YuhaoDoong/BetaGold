"""历史期权 K线 + 信号回测校准 (用真实期权价格).

核心: 任何今天仍活跃的期权, 通过 yfinance OCC 代码可拉 6+ 个月历史日 K 线.
这意味着对任何近 6 月的信号, 我们都可以用真实期权价格算"模拟仓"P&L.

OCC 期权代码格式:
  <UNDERLYING><YYMMDD>C<STRIKE×1000(8位)>  例: SLV260515C00065000
  <UNDERLYING><YYMMDD>P<STRIKE×1000(8位)>      SLV260515P00065000

用例:
  3月18 GLD STRADDLE 信号 → 选 5/15 GLD 期权 (60 DTE) → 拉历史 K 线 → 算 P&L
  4月17 SLV STRADDLE 信号 → 选 5/15 SLV 期权 (28 DTE) → 同上

工作流:
  1. occ_symbol(underlying, expiry, strike, type) — 构造 OCC 代码
  2. fetch_option_history(symbol, period) — yfinance 拉历史
  3. compute_real_straddle_pnl(call_hist, put_hist, entry_d, exit_d) — 算 P&L
  4. real_pnl_for_signal(...) — 一站式
"""
from typing import Optional, Dict, List, Tuple
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np


def occ_symbol(underlying: str, expiry: str, strike: float,
                option_type: str = "C") -> str:
    """构造 OCC 期权 ticker.

    Args:
        underlying: 'SLV', 'GLD'
        expiry: 'YYYY-MM-DD'
        strike: 65.0 → 00065000
        option_type: 'C' or 'P'

    Returns: 'SLV260515C00065000'
    """
    exp_d = pd.Timestamp(expiry)
    yymmdd = exp_d.strftime("%y%m%d")
    strike_i = int(round(strike * 1000))
    strike_str = f"{strike_i:08d}"
    return f"{underlying.upper()}{yymmdd}{option_type.upper()}{strike_str}"


def fetch_option_history(symbol: str, period: str = "6mo"
                            ) -> Optional[pd.DataFrame]:
    """拉单期权历史日 K 线 (yfinance)."""
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        df = t.history(period=period)
        if df is None or len(df) == 0:
            return None
        # 去 timezone, normalize 日期
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print(f"[fetch_option_history] {symbol}: {e}")
        return None


def compute_real_straddle_pnl(call_hist: pd.DataFrame,
                                 put_hist: pd.DataFrame,
                                 entry_date: str,
                                 hold_days: int = 5,
                                 entry_price_mode: str = "open") -> Optional[Dict]:
    """从 call/put 真实历史 K 线算 Long Straddle P&L.

    Args:
        call_hist, put_hist: fetch_option_history 输出
        entry_date: 'YYYY-MM-DD' 信号日
        hold_days: 持仓天数
        entry_price_mode: 'open' (开盘买入, 实际入场价) / 'close' (收盘价)
    """
    entry_d = pd.Timestamp(entry_date).normalize()
    exit_d_target = entry_d + pd.Timedelta(days=hold_days)

    if entry_d not in call_hist.index:
        avail = call_hist.index[call_hist.index >= entry_d]
        if len(avail) == 0:
            return None
        entry_d = avail[0]
    if entry_d not in put_hist.index:
        avail = put_hist.index[put_hist.index >= entry_d]
        if len(avail) == 0:
            return None
        entry_d = max(entry_d, avail[0])

    col = "Open" if entry_price_mode == "open" else "Close"
    entry_call = float(call_hist.loc[entry_d, col])
    entry_put = float(put_hist.loc[entry_d, col])
    entry_total = entry_call + entry_put

    # 持仓期窗口
    win_call = call_hist[(call_hist.index >= entry_d)
                          & (call_hist.index <= exit_d_target)]
    win_put = put_hist[(put_hist.index >= entry_d)
                         & (put_hist.index <= exit_d_target)]
    common = win_call.index.intersection(win_put.index)
    if len(common) == 0:
        return None

    # 跟踪每天 Straddle 价值
    daily_straddle = (win_call.loc[common, "Close"]
                       + win_put.loc[common, "Close"])
    daily_straddle_high = (win_call.loc[common, "High"]
                            + win_put.loc[common, "High"])

    # 平仓: 持仓期最后一天收盘
    exit_actual = common[-1]
    exit_call = float(call_hist.loc[exit_actual, "Close"])
    exit_put = float(put_hist.loc[exit_actual, "Close"])
    exit_total = exit_call + exit_put

    # 最大可能价值 (理论早平最优)
    max_close = float(daily_straddle.max())
    max_high = float(daily_straddle_high.max())

    pnl_close = exit_total - entry_total
    pnl_close_pct = pnl_close / entry_total * 100 if entry_total > 0 else 0
    max_pnl_close = max_close - entry_total
    max_pnl_close_pct = max_pnl_close / entry_total * 100 \
        if entry_total > 0 else 0
    max_pnl_high = max_high - entry_total
    max_pnl_high_pct = max_pnl_high / entry_total * 100 \
        if entry_total > 0 else 0

    return {
        "entry_date": entry_d.strftime("%Y-%m-%d"),
        "exit_date": exit_actual.strftime("%Y-%m-%d"),
        "actual_hold_days": (exit_actual - entry_d).days,
        "entry_call": entry_call, "entry_put": entry_put,
        "entry_total": entry_total,
        "exit_call": exit_call, "exit_put": exit_put,
        "exit_total": exit_total,
        # 持有到末平仓 P&L
        "pnl_close": pnl_close, "pnl_close_pct": pnl_close_pct,
        # 持仓期最大价值 (上帝视角早平)
        "max_straddle_close": max_close,
        "max_pnl_close": max_pnl_close,
        "max_pnl_close_pct": max_pnl_close_pct,
        # 含日内冲高
        "max_straddle_high": max_high,
        "max_pnl_high": max_pnl_high,
        "max_pnl_high_pct": max_pnl_high_pct,
        # 每日轨迹
        "daily_close": daily_straddle.to_dict(),
    }


def find_nearest_atm_strike(underlying: str, signal_date: str,
                              expiry: str, strike_step: float = 0.5) -> float:
    """估算信号日 ATM strike (用 yfinance underlying 历史)."""
    try:
        import yfinance as yf
        d = pd.Timestamp(signal_date)
        df = yf.Ticker(underlying).history(
            start=d.strftime("%Y-%m-%d"),
            end=(d + pd.Timedelta(days=2)).strftime("%Y-%m-%d"))
        spot = float(df["Open"].iloc[0])
        # 取最近 0.5 整数倍
        return round(spot / strike_step) * strike_step
    except Exception as e:
        print(f"[find_nearest_atm_strike] {e}")
        return None


def real_pnl_for_signal(underlying: str, signal_date: str,
                          expiry: str = None,
                          strike: float = None,
                          hold_days: int = 5,
                          target_dte: int = 30) -> Optional[Dict]:
    """全流程: 信号日 → 找 ATM 期权 → 拉历史 → 算 P&L.

    Args:
        underlying: 'SLV', 'GLD'
        signal_date: '2026-04-17'
        expiry: 指定到期日; None 则按 target_dte 自动选下一月度第三周五
        strike: 指定 strike; None 则自动 ATM
        hold_days: 模拟持仓天数
        target_dte: 自动选 expiry 时的目标 DTE

    Returns: {真实 P&L + 入场参数} 或错误
    """
    sig_d = pd.Timestamp(signal_date)

    if expiry is None:
        # 自动选: 信号日 +target_dte 附近的月度第三周五
        target = sig_d + pd.Timedelta(days=target_dte)
        first = target.replace(day=1)
        while first.weekday() != 4:
            first += pd.Timedelta(days=1)
        expiry = (first + pd.Timedelta(weeks=2)).strftime("%Y-%m-%d")

    if strike is None:
        strike = find_nearest_atm_strike(underlying, signal_date, expiry)
        if strike is None:
            return {"error": "找不到 ATM strike"}

    call_sym = occ_symbol(underlying, expiry, strike, "C")
    put_sym = occ_symbol(underlying, expiry, strike, "P")
    print(f"信号日 {signal_date}: ATM ${strike}, expiry {expiry}")
    print(f"  Call: {call_sym}")
    print(f"  Put:  {put_sym}")

    call_hist = fetch_option_history(call_sym, period="2y")
    put_hist = fetch_option_history(put_sym, period="2y")
    if call_hist is None or put_hist is None:
        return {"error": "无法拉历史 K 线",
                "call_sym": call_sym, "put_sym": put_sym}

    result = compute_real_straddle_pnl(
        call_hist, put_hist, signal_date, hold_days, "open")
    if result is None:
        return {"error": "P&L 计算失败",
                "call_sym": call_sym, "put_sym": put_sym}

    result.update({
        "underlying": underlying, "strike": strike, "expiry": expiry,
        "call_sym": call_sym, "put_sym": put_sym,
    })
    return result


if __name__ == "__main__":
    # 测试: 用户实仓 SLV 4/29 65 strike 5/15 (1d 持仓看)
    print("=== 用户 SLV 4/29 STRADDLE 校准 ===")
    res = real_pnl_for_signal(
        underlying="SLV", signal_date="2026-04-29",
        expiry="2026-05-15", strike=65.0, hold_days=1,
    )
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"  {k}: ...")
        else:
            print(f"  {k}: {v}")
