"""系统化真实期权回测 — 对所有历史信号用真实 yfinance 期权 K 线算 P&L.

核心思路:
  1. 不带任何过滤, 列出所有信号触发候选
  2. 对每个信号, 选 30-90d 后的月度第三周五到期
  3. 用 yfinance OCC 代码拉真实历史 K 线 (近 6 月)
  4. 对 4 种策略并行计算真实 P&L:
     - Long Call (BUY CALL 信号 → 期权)
     - Short Put (SELL PUT 信号 → 收 premium)
     - Long Straddle (STRADDLE 信号)
     - Iron Condor (SHORT_VOL 信号, 短 1.6σ + 长 3σ)
  5. 保存到 data/real_options_backtest/<asset>_<type>.csv
  6. 聚合统计: 不同 RV %tile bucket / 事件邻近 / regime 下的真实 P&L

用法:
    python scripts/real_options_backtest.py --asset GLD --hold 5
    python scripts/real_options_backtest.py --asset SLV --types STRADDLE,BUY_CALL
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import date, datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from core.data import load_features, load_config
from core.signals import compute_rv_pctile, build_band
from core.signals_v2 import generate_daily_signals
from core.events import (detect_straddle_signal, detect_short_vol_signal,
                          days_to_next_event)
from core.regime import RegimeClassifier
from core.options_history import (occ_symbol, fetch_option_history,
                                     compute_real_straddle_pnl)


def nearest_monthly_third_friday(d: pd.Timestamp,
                                    target_dte: int = 45) -> pd.Timestamp:
    """找信号日 +target_dte 附近的月度第三周五."""
    target = d + pd.Timedelta(days=target_dte)
    first = target.replace(day=1)
    while first.weekday() != 4:
        first += pd.Timedelta(days=1)
    return first + pd.Timedelta(weeks=2)


def smart_pick_expiry(signal_date) -> pd.Timestamp:
    """智能选 expiry — 老信号用 LEAPS (270115 上市最早), 新信号用月度.

    yfinance 期权数据可达性:
      270115 LEAPS: 上市约 2024-10 (18 个月历史)
      260918 (中长): 2025-08 (8 月历史)
      月度近月: 通常上市 6-12 个月前
    """
    sig_d = pd.Timestamp(signal_date)
    age = (pd.Timestamp.now() - sig_d).days
    if age < 90:
        # 新信号: 月度 ~45 DTE
        return nearest_monthly_third_friday(sig_d, 45)
    if age < 180:
        # 中老: 260918 (2026 三季度月度) 或 270115
        return pd.Timestamp("2026-09-18")
    # 老信号 (>6 月前): LEAPS 必选
    return pd.Timestamp("2027-01-15")


_LEAPS_STRIKE_CACHE: dict = {}


def leaps_strike(asset: str, spot: float,
                   expiry_str: str | None = None) -> float:
    """LEAPS strike — yfinance 探测 ATM 可用 strike (v3.7.41 strike 修正).

    旧版用硬编码 candidates 列表, GLD 间隔 $20+, 误差大.
    新版按真实 LEAPS strike 间距 (GLD $5, SLV $1) 由 ATM 向外探测,
    取首个有 yfinance 历史数据的 strike — 真正最接近 ATM.

    传入 expiry_str 才能探测 (否则 fallback 到旧 candidates).
    """
    if expiry_str is None:
        # fallback: 老接口, 用粗糙 candidates
        if asset == "GLD":
            cand = [380, 400, 420, 440, 460, 480, 500, 520, 540]
        else:
            cand = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
        return min(cand, key=lambda x: abs(x - spot))

    cache_key = (asset, round(spot, 1), expiry_str)
    if cache_key in _LEAPS_STRIKE_CACHE:
        return _LEAPS_STRIKE_CACHE[cache_key]

    if asset == "GLD":
        base = round(spot / 5) * 5
        offsets = [0, 5, -5, 10, -10, 15, -15, 20, -20, 25, -25]
    else:  # SLV $1 间隔
        base = round(spot)
        offsets = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]

    import yfinance as yf
    for off in offsets:
        k = base + off
        if k <= 0:
            continue
        sym = occ_symbol(asset, expiry_str, k, "C")
        try:
            df = yf.Ticker(sym).history(period="1mo")
            if df is not None and len(df) > 5:
                _LEAPS_STRIKE_CACHE[cache_key] = float(k)
                return float(k)
        except Exception:
            pass
    _LEAPS_STRIKE_CACHE[cache_key] = float(base)
    return float(base)


def round_strike(spot: float, step: float = None) -> float:
    """ATM strike — 自动按价位选 step.

    GLD ~$400+: $1 间隔
    SLV ~$60: $0.5 间隔
    """
    if step is None:
        step = 1.0 if spot > 100 else 0.5
    return round(spot / step) * step


def find_valid_strike(asset, spot, expiry_str, period="2y"):
    """尝试 ATM ± 多个 strike, 找第一个能拉到数据的 (优先 ATM)."""
    if asset == "GLD":
        # GLD 大多 $1 间隔, 但远月可能 $5
        base = round(spot)
        step_seq = [(0, 1), (1, 1), (-1, 1), (2, 1), (-2, 1),
                     (5, 5), (-5, 5)]
    else:
        # SLV $0.5 间隔
        base = round(spot * 2) / 2
        step_seq = [(0, 0.5), (0.5, 0.5), (-0.5, 0.5),
                     (1, 1), (-1, 1)]
    for offset, _ in step_seq:
        s = base + offset
        sym = occ_symbol(asset, expiry_str, s, "C")
        try:
            import yfinance as yf
            df = yf.Ticker(sym).history(period="1mo")
            if df is not None and len(df) > 5:
                return s
        except Exception:
            pass
    return base


def real_long_call_pnl(call_hist, entry_d, hold_days=5,
                          stop_loss_pct=-50.0, take_profit_pct=None):
    """单腿 Long Call P&L. v3.7.42 加止损 (默认 -50%)."""
    entry_d = pd.Timestamp(entry_d).normalize()
    exit_target = entry_d + pd.Timedelta(days=hold_days)
    if entry_d not in call_hist.index:
        avail = call_hist.index[call_hist.index >= entry_d]
        if len(avail) == 0:
            return None
        entry_d = avail[0]
    entry_p = float(call_hist.loc[entry_d, "Open"])
    if entry_p <= 0:
        return None
    win = call_hist[(call_hist.index >= entry_d) & (call_hist.index <= exit_target)]
    if len(win) < 2:
        return None

    # 逐日检查止损/止盈 (用 Low 触发以模拟盘中)
    exit_d = win.index[-1]; exit_p = float(win.iloc[-1]["Close"]); stopped = False
    for d, row in win.iterrows():
        if d == entry_d:
            continue
        low_pct = (float(row["Low"]) / entry_p - 1) * 100
        high_pct = (float(row["High"]) / entry_p - 1) * 100
        if low_pct <= stop_loss_pct:
            exit_d = d; exit_p = entry_p * (1 + stop_loss_pct / 100); stopped = True
            break
        if take_profit_pct is not None and high_pct >= take_profit_pct:
            exit_d = d; exit_p = entry_p * (1 + take_profit_pct / 100); stopped = True
            break

    max_close = float(win["Close"].max())
    max_high = float(win["High"].max())
    return {
        "entry_date": entry_d.strftime("%Y-%m-%d"),
        "exit_date": exit_d.strftime("%Y-%m-%d"),
        "entry": entry_p, "exit_close": exit_p,
        "stopped": stopped,
        "pnl_close": exit_p - entry_p,
        "pnl_close_pct": (exit_p / entry_p - 1) * 100,
        "max_close": max_close,
        "max_pnl_close_pct": (max_close / entry_p - 1) * 100,
        "max_high": max_high,
    }


def real_short_put_pnl(put_hist, entry_d, hold_days=5,
                          stop_loss_pct=-25.0):
    """单腿 Short Put P&L. v3.7.42 加止损 (premium 翻倍 ≈ -100% pnl_pct, 但实盘 -25% 即应平).

    SELL PUT P&L_pct = (entry-exit)/entry. 即 exit ≥ 1.25 × entry 时, pnl_pct = -25%, 止损.
    """
    entry_d = pd.Timestamp(entry_d).normalize()
    exit_target = entry_d + pd.Timedelta(days=hold_days)
    if entry_d not in put_hist.index:
        avail = put_hist.index[put_hist.index >= entry_d]
        if len(avail) == 0:
            return None
        entry_d = avail[0]
    entry_p = float(put_hist.loc[entry_d, "Open"])
    if entry_p <= 0:
        return None
    win = put_hist[(put_hist.index >= entry_d) & (put_hist.index <= exit_target)]
    if len(win) < 2:
        return None

    # 止损阈值: premium 涨过 (1 - stop_loss_pct/100) 倍 → 止损平仓
    stop_mult = 1 - stop_loss_pct / 100  # -25% 止损 → 1.25
    exit_d = win.index[-1]; exit_p = float(win.iloc[-1]["Close"]); stopped = False
    for d, row in win.iterrows():
        if d == entry_d:
            continue
        if float(row["High"]) >= entry_p * stop_mult:
            exit_d = d; exit_p = entry_p * stop_mult; stopped = True
            break

    return {
        "entry_date": entry_d.strftime("%Y-%m-%d"),
        "exit_date": exit_d.strftime("%Y-%m-%d"),
        "entry_premium": entry_p,
        "exit_premium": exit_p,
        "stopped": stopped,
        "pnl_close": entry_p - exit_p,
        "pnl_close_pct": (entry_p - exit_p) / entry_p * 100,
    }


def real_iron_condor_full(asset, signal_date, spot, expiry, hold_days=5,
                            short_sigma=1.6, wing_sigma=3.0, rv_for_strike=20):
    """构建 4 腿 Iron Condor 并计算真实 P&L.

    Args:
        asset: 'GLD' / 'SLV'
        signal_date: 'YYYY-MM-DD'
        spot: 信号日 underlying 价
        expiry: 'YYYY-MM-DD'
        hold_days: 持仓天数
        short_sigma/wing_sigma: 1.6σ short, 3σ long wing
        rv_for_strike: RV (%) 估算 strike 距离 (sigma_pct = RV × √(DTE/365))

    Returns: P&L dict 或 None
    """
    sig_d = pd.Timestamp(signal_date)
    exp_d = pd.Timestamp(expiry)
    dte = (exp_d - sig_d).days
    sigma_pct = rv_for_strike / 100 * (dte / 365) ** 0.5
    short_offset = sigma_pct * short_sigma * spot
    wing_offset = sigma_pct * wing_sigma * spot

    # 4 strike
    if asset == "GLD":
        strike_step = 5  # GLD LEAPS 通常 $5 间隔
    else:
        strike_step = 5
    short_call_k = round((spot + short_offset) / strike_step) * strike_step
    long_call_k = round((spot + wing_offset) / strike_step) * strike_step
    short_put_k = round((spot - short_offset) / strike_step) * strike_step
    long_put_k = round((spot - wing_offset) / strike_step) * strike_step

    # 拉 4 个 series
    syms = {
        "sc": occ_symbol(asset, expiry, short_call_k, "C"),
        "lc": occ_symbol(asset, expiry, long_call_k, "C"),
        "sp": occ_symbol(asset, expiry, short_put_k, "P"),
        "lp": occ_symbol(asset, expiry, long_put_k, "P"),
    }
    hists = {}
    for k, sym in syms.items():
        h = fetch_option_history(sym, period="2y")
        if h is None or len(h) < 5:
            return None
        hists[k] = h

    entry_d_ts = pd.Timestamp(signal_date).normalize()
    exit_target = entry_d_ts + pd.Timedelta(days=hold_days)
    common = hists["sc"].index
    for k in ["lc", "sp", "lp"]:
        common = common & hists[k].index
    avail = common[(common >= entry_d_ts) & (common <= exit_target)]
    if len(avail) < 2:
        return None
    e_d = avail[0]

    # 入场 (Open): 卖 sc + sp, 买 lc + lp
    sc_e = hists["sc"].loc[e_d, "Open"]
    sp_e = hists["sp"].loc[e_d, "Open"]
    lc_e = hists["lc"].loc[e_d, "Open"]
    lp_e = hists["lp"].loc[e_d, "Open"]
    credit = (sc_e + sp_e) - (lc_e + lp_e)

    # v3.7.42: IC 止损 — net debit 涨过 2× credit 即平
    # IC 平仓成本 = sc_close + sp_close - lc_close - lp_close
    # 若 平仓成本 > 2 × credit, pnl = credit - 2c = -credit, 即 -100% of credit
    stop_debit = 2 * abs(credit) if credit > 0 else float("inf")
    stopped = False
    x_d = avail[-1]
    debit = (hists["sc"].loc[x_d,"Close"] + hists["sp"].loc[x_d,"Close"]
              - hists["lc"].loc[x_d,"Close"] - hists["lp"].loc[x_d,"Close"])
    for d in avail[1:]:
        d_debit = (hists["sc"].loc[d,"Close"] + hists["sp"].loc[d,"Close"]
                    - hists["lc"].loc[d,"Close"] - hists["lp"].loc[d,"Close"])
        if d_debit >= stop_debit:
            x_d = d; debit = d_debit; stopped = True
            break

    pnl = credit - debit

    return {
        "entry_date": e_d.strftime("%Y-%m-%d"),
        "exit_date": x_d.strftime("%Y-%m-%d"),
        "ic_short_call_k": short_call_k, "ic_long_call_k": long_call_k,
        "ic_short_put_k": short_put_k, "ic_long_put_k": long_put_k,
        "ic_credit": credit, "ic_debit_at_exit": debit,
        "ic_stopped": stopped,
        "ic_pnl": pnl,
        "ic_pnl_pct_of_credit": pnl / credit * 100 if abs(credit) > 0.01 else 0,
        "ic_max_loss": (wing_offset - short_offset) - credit,
    }


def real_iron_condor_pnl(short_call, long_call, short_put, long_put,
                            entry_d, hold_days=5):
    """4 腿 Iron Condor P&L.

    入场: 卖 short call/put, 买 long call/put 翼
    净 credit = (sc + sp) - (lc + lp)
    平仓: 反向操作
    """
    entry_d = pd.Timestamp(entry_d).normalize()
    exit_target = entry_d + pd.Timedelta(days=hold_days)
    avail_dates = (short_call.index & long_call.index
                    & short_put.index & long_put.index)
    avail_dates = avail_dates[avail_dates >= entry_d]
    if len(avail_dates) < 2:
        return None
    entry_d = avail_dates[0]
    exit_d = min(avail_dates[-1], exit_target)
    avail_in_win = avail_dates[(avail_dates >= entry_d) & (avail_dates <= exit_d)]
    if len(avail_in_win) < 2:
        return None
    exit_d = avail_in_win[-1]

    # 入场: open
    sc_e = float(short_call.loc[entry_d, "Open"])
    lc_e = float(long_call.loc[entry_d, "Open"])
    sp_e = float(short_put.loc[entry_d, "Open"])
    lp_e = float(long_put.loc[entry_d, "Open"])
    net_credit = (sc_e + sp_e) - (lc_e + lp_e)
    # 平仓: close
    sc_x = float(short_call.loc[exit_d, "Close"])
    lc_x = float(long_call.loc[exit_d, "Close"])
    sp_x = float(short_put.loc[exit_d, "Close"])
    lp_x = float(long_put.loc[exit_d, "Close"])
    net_close = (sc_x + sp_x) - (lc_x + lp_x)
    pnl = net_credit - net_close  # 短 vol: credit 缩小赚
    return {
        "entry_date": entry_d.strftime("%Y-%m-%d"),
        "exit_date": exit_d.strftime("%Y-%m-%d"),
        "entry_credit": net_credit,
        "exit_value": net_close,
        "pnl": pnl,
        "pnl_pct_of_credit": pnl / net_credit * 100
            if abs(net_credit) > 0.01 else 0,
    }


_EXPIRY_LIST_CACHE: dict = {}


def get_active_expiries(asset: str) -> list:
    """获取 yfinance 当前 SLV/GLD 全部活跃 expiry 列表 (含 LEAPS).

    所有 expiry > 今天, 一次性拉到内存, 缓存重用.
    """
    if asset in _EXPIRY_LIST_CACHE:
        return _EXPIRY_LIST_CACHE[asset]
    try:
        import yfinance as yf
        exps = yf.Ticker(asset).options
        out = sorted(pd.Timestamp(e) for e in exps)
    except Exception:
        out = [pd.Timestamp("2026-09-18"), pd.Timestamp("2027-01-15")]  # fallback
    _EXPIRY_LIST_CACHE[asset] = out
    return out


def _is_monthly_third_friday(d: pd.Timestamp) -> bool:
    """是否月度第三周五 (标准月度期权 expiry, 上市最早, 历史最深)."""
    if d.weekday() != 4:  # not Friday
        return False
    return 15 <= d.day <= 21


def pick_best_expiry(asset: str, sig_d: pd.Timestamp,
                       target_dte: int = 45,
                       hold_days: int = 5) -> list:
    """对信号日返回 expiry 候选列表 (v3.7.43).

    每个候选 expiry 满足:
      - expiry > sig_d + hold_days
      - expiry > 今天 (现在 yf 仍能拉到数据)

    优先级:
      1. 月度第三周五 (上市早, 历史深) — 按 DTE 适配排
      2. weekly 仅作为补充 (上市晚, 历史可能不覆盖信号日)
    返回 top ~10.
    """
    today = pd.Timestamp.now().normalize()
    all_exp = get_active_expiries(asset)
    valid = [
        e for e in all_exp
        if e > today and (e - sig_d).days >= hold_days + 1
    ]
    if not valid:
        return []
    monthly = [e for e in valid if _is_monthly_third_friday(e)]
    weekly = [e for e in valid if not _is_monthly_third_friday(e)]
    # 按 DTE 适配 (越接近 target 越优), 月度优先
    monthly.sort(key=lambda e: abs((e - sig_d).days - target_dte))
    weekly.sort(key=lambda e: abs((e - sig_d).days - target_dte))
    out = monthly[:8] + weekly[:3]
    # 去重保序
    seen = set(); dedup = []
    for e in out:
        if e not in seen:
            seen.add(e); dedup.append(e)
    return dedup


def backtest_signal(asset, signal_date, signal_type, spot,
                      hold_days=5, target_dte=45, rv=None):
    """对单个信号跑真实期权 P&L (4 类策略并行).

    v3.7.43: 选 expiry 改为遍历 yfinance 当前所有活跃 expiry,
            按 |DTE_at_signal - target_dte| 排序, 取首个有完整数据的.
    """
    sig_d = pd.Timestamp(signal_date)
    age_days = (pd.Timestamp.now() - sig_d).days

    expiry_candidates = pick_best_expiry(asset, sig_d, target_dte, hold_days)
    if not expiry_candidates:
        return {
            "signal_date": signal_date, "signal_type": signal_type,
            "spot": spot, "error": "无可用 expiry (信号距今 > 最远 LEAPS 上市期)",
        }

    call_hist = None; put_hist = None
    expiry = None; strike = None
    sig_norm = sig_d.normalize()

    sig_window_end = sig_norm + pd.Timedelta(days=hold_days + 2)
    for exp in expiry_candidates:
        exp_str = exp.strftime("%Y-%m-%d")
        dte_at_sig = (exp - sig_d).days
        # 月度 (DTE ≤ 90) 用 find_valid_strike, 远月 (DTE > 90) 用 leaps_strike
        if dte_at_sig <= 90:
            k = find_valid_strike(asset, spot, exp_str)
        else:
            k = leaps_strike(asset, spot, exp_str)
        c_sym = occ_symbol(asset, exp_str, k, "C")
        p_sym = occ_symbol(asset, exp_str, k, "P")
        c_hist = fetch_option_history(c_sym, period="2y")
        p_hist = fetch_option_history(p_sym, period="2y")
        if c_hist is None or p_hist is None or len(c_hist) < 5 or len(p_hist) < 5:
            continue
        common = c_hist.index.intersection(p_hist.index)
        # v3.7.43: 必须在 [信号日, 信号日+持仓+2d] 窗口内有 ≥2 个数据点
        in_window = common[(common >= sig_norm) & (common <= sig_window_end)]
        if len(in_window) >= 2:
            call_hist = c_hist; put_hist = p_hist
            expiry = exp; strike = k
            break

    expiry_str = expiry.strftime("%Y-%m-%d") if expiry is not None else None
    actual_dte = (expiry - sig_d).days if expiry is not None else None

    result = {
        "signal_date": signal_date,
        "signal_type": signal_type,
        "spot": spot,
        "strike_atm": strike,
        "expiry": expiry_str,
        "actual_dte": actual_dte,
        "rv": rv,
    }

    if call_hist is None or put_hist is None:
        result["error"] = "无历史 K 线 (所有候选 expiry 在信号日均无数据)"
        return result

    # Long Call (BUY CALL)
    lc = real_long_call_pnl(call_hist, signal_date, hold_days)
    if lc:
        result["long_call_entry"] = lc["entry"]
        result["long_call_pnl_close"] = lc["pnl_close"]
        result["long_call_pnl_pct"] = lc["pnl_close_pct"]
        result["long_call_max_pnl_pct"] = lc["max_pnl_close_pct"]

    # Short Put (SELL PUT — 卖 ATM Put)
    sp = real_short_put_pnl(put_hist, signal_date, hold_days)
    if sp:
        result["short_put_entry_premium"] = sp["entry_premium"]
        result["short_put_pnl_close"] = sp["pnl_close"]
        result["short_put_pnl_pct"] = sp["pnl_close_pct"]

    # Long Straddle
    st = compute_real_straddle_pnl(call_hist, put_hist, signal_date, hold_days)
    if st:
        result["straddle_entry"] = st["entry_total"]
        result["straddle_pnl_close"] = st["pnl_close"]
        result["straddle_pnl_pct"] = st["pnl_close_pct"]
        result["straddle_max_pnl_pct"] = st["max_pnl_close_pct"]

    # Iron Condor (4 腿真实 P&L) — 仅在 SHORT_VOL 信号触发时跑 (节约 yfinance 调用)
    if signal_type == "SHORT_VOL" and rv is not None:
        try:
            ic = real_iron_condor_full(
                asset, signal_date, spot, expiry_str, hold_days,
                rv_for_strike=rv,
            )
            if ic:
                result.update(ic)
        except Exception as e:
            result["ic_error"] = str(e)

    return result


def run_full_backtest(asset, hold_days=5, target_dte=45,
                        rv_pctile_min=0.0, rv_pctile_max=1.01,
                        date_min=None, date_max=None):
    """对历史所有候选信号跑真实期权 P&L."""
    cfg = load_config()
    fname = "gld.csv" if asset == "GLD" else "slv.csv"
    df = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{fname}",
                      index_col=0, parse_dates=True)
    features = load_features(cfg)
    common = features.index.intersection(df.index)
    features = features.loc[common]
    close = df["Close"][common]
    high = df["High"][common]
    low = df["Low"][common]
    if asset == "SLV":
        ret = close.pct_change()
        rv_10d = ret.rolling(10).std() * (252 ** 0.5) * 100
    else:
        rv_10d = features["rv_10d"]
    rv_pct = compute_rv_pctile(rv_10d)
    oos_path = f"/Users/yhdong/Gold/data/models/dl_range_{asset.lower()}_oos.parquet"
    if not os.path.exists(oos_path):
        oos_path = "/Users/yhdong/Gold/data/models/dl_range_v2_oos.parquet"
    oos = pd.read_parquet(oos_path)
    upper, lower, _ = build_band(oos, close)
    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(features[feat_cols])["regime"]

    # 信号: 不带任何过滤
    sig_df = generate_daily_signals(close, high, low, upper, lower,
                                       regime, rv_pct, rv_filter=False)
    straddle_df = detect_straddle_signal(rv_10d, close.index, rv_pctile=rv_pct,
                                              close=close, high=high, low=low,
                                              asset=asset)
    short_vol_df = detect_short_vol_signal(rv_10d, rv_pct, close.index,
                                              regime=regime,
                                              daily_range=(high-low)/close*100,
                                              close=close, high=high, low=low,
                                              asset=asset)

    # 限定范围 (LEAPS 拉到 ~18mo, 月度 ~6mo)
    if date_min is None:
        date_min = pd.Timestamp.now() - pd.Timedelta(days=540)  # 18 月
    if date_max is None:
        date_max = pd.Timestamp.now() - pd.Timedelta(days=hold_days+5)
    if isinstance(date_min, str):
        date_min = pd.Timestamp(date_min)
    if isinstance(date_max, str):
        date_max = pd.Timestamp(date_max)

    candidates = []
    for d in sig_df.index:
        if d < date_min or d > date_max:
            continue
        rvp = rv_pct.get(d, 0.5)
        if rvp < rv_pctile_min or rvp > rv_pctile_max:
            continue
        rv_v = rv_10d.get(d, 20)
        spot = close[d]
        reg = regime.get(d, "?")
        d_fomc, _, _ = days_to_next_event(d, "FOMC")
        d_nfp, _, _ = days_to_next_event(d, "NFP")

        # v3.7.48: 把 straddle_score 也存进去, 供后续阈值扫描
        st_score = (float(straddle_df.loc[d, "straddle_score"])
                      if d in straddle_df.index
                      and "straddle_score" in straddle_df.columns
                      else 0)
        sv_score = (float(short_vol_df.loc[d, "short_vol_score"])
                      if d in short_vol_df.index
                      and "short_vol_score" in short_vol_df.columns
                      else 0)
        meta = {
            "signal_date": d.strftime("%Y-%m-%d"),
            "spot": spot,
            "rv": rv_v,
            "rv_pctile": rvp,
            "regime": reg,
            "days_to_fomc": d_fomc,
            "days_to_nfp": d_nfp,
            "straddle_score": st_score,
            "short_vol_score": sv_score,
        }
        # 哪些信号触发?
        types = []
        if sig_df.loc[d, "buy_signal"]:
            types.append(sig_df.loc[d, "buy_type"])
        if sig_df.loc[d, "exit_signal"]:
            types.append("EXIT")
        if d in straddle_df.index and straddle_df.loc[d, "straddle_signal"]:
            types.append("STRADDLE")
        if d in short_vol_df.index and short_vol_df.loc[d, "short_vol_signal"]:
            types.append("SHORT_VOL")
        if not types:
            continue
        candidates.append((d, types, meta))

    print(f"[{asset}] 候选信号: {len(candidates)} 天")
    print(f"  日期范围: {date_min.date()} → {date_max.date()}")

    # 对每个候选跑真实期权 P&L
    results = []
    for i, (d, types, meta) in enumerate(candidates):
        if i % 5 == 0:
            print(f"  进度 {i+1}/{len(candidates)}: {d.date()}...")
        for t in types:
            r = backtest_signal(asset, meta["signal_date"], t,
                                  meta["spot"], hold_days, target_dte,
                                  rv=meta["rv"])
            r.update({k: v for k, v in meta.items()
                       if k not in ("signal_date", "spot", "rv")})
            results.append(r)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="GLD", choices=["GLD", "SLV"])
    parser.add_argument("--hold", type=int, default=5)
    parser.add_argument("--dte", type=int, default=45,
                         help="target DTE for option expiry (~30-90)")
    parser.add_argument("--date-min", default=None,
                         help="YYYY-MM-DD, 默认 today-180d")
    parser.add_argument("--date-max", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or str(
        Path(__file__).parent.parent.parent / "Gold" / "data"
        / "real_options_backtest")
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== 真实期权回测 {args.asset} ===")
    df = run_full_backtest(
        args.asset, args.hold, args.dte,
        date_min=args.date_min, date_max=args.date_max,
    )

    fpath = os.path.join(out_dir,
                            f"{args.asset}_real_pnl_hold{args.hold}d.csv")
    df.to_csv(fpath, index=False)
    print(f"\n保存 → {fpath}, 共 {len(df)} 条记录")

    # v3.7.46: 自动归档历史快照 (每次跑后保留时间戳版本)
    history_dir = os.path.join(out_dir, "history")
    os.makedirs(history_dir, exist_ok=True)
    today = date.today().isoformat()
    archive = os.path.join(history_dir,
                              f"{args.asset}_hold{args.hold}d_{today}.csv")
    df.to_csv(archive, index=False)
    print(f"快照 → {archive}\n")

    # 聚合: 各信号类型胜率/平均 P&L
    print(f"\n=== 聚合统计 ({args.asset}) ===")
    for sig_type in ["BUY CALL", "SELL PUT", "STRADDLE", "SHORT_VOL", "EXIT"]:
        sub = df[df["signal_type"] == sig_type]
        if len(sub) == 0:
            continue

        # Long Call (BUY CALL 信号)
        if sig_type == "BUY CALL" and "long_call_pnl_pct" in sub.columns:
            valid = sub[sub["long_call_pnl_pct"].notna()]
            if len(valid) > 0:
                wr = (valid["long_call_pnl_pct"] > 0).mean()
                print(f"  {sig_type} → Long Call: {len(valid)} 笔, "
                      f"胜 {wr:.0%}, "
                      f"avg {valid['long_call_pnl_pct'].mean():+.1f}%, "
                      f"max_avg {valid['long_call_max_pnl_pct'].mean():+.1f}%")

        if sig_type == "SELL PUT" and "short_put_pnl_pct" in sub.columns:
            valid = sub[sub["short_put_pnl_pct"].notna()]
            if len(valid) > 0:
                wr = (valid["short_put_pnl_pct"] > 0).mean()
                print(f"  {sig_type} → Short Put: {len(valid)} 笔, "
                      f"胜 {wr:.0%}, "
                      f"avg {valid['short_put_pnl_pct'].mean():+.1f}%")

        if sig_type == "STRADDLE" and "straddle_pnl_pct" in sub.columns:
            valid = sub[sub["straddle_pnl_pct"].notna()]
            if len(valid) > 0:
                wr = (valid["straddle_pnl_pct"] > 0).mean()
                print(f"  {sig_type} → Long Straddle: {len(valid)} 笔, "
                      f"胜 {wr:.0%}, "
                      f"avg {valid['straddle_pnl_pct'].mean():+.1f}%, "
                      f"max_avg {valid['straddle_max_pnl_pct'].mean():+.1f}%")

    return df


if __name__ == "__main__":
    main()
