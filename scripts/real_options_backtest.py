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


def real_long_call_pnl(call_hist, entry_d, hold_days=5):
    """单腿 Long Call P&L (BUY CALL 信号用)."""
    entry_d = pd.Timestamp(entry_d).normalize()
    exit_target = entry_d + pd.Timedelta(days=hold_days)
    if entry_d not in call_hist.index:
        avail = call_hist.index[call_hist.index >= entry_d]
        if len(avail) == 0:
            return None
        entry_d = avail[0]
    entry_p = float(call_hist.loc[entry_d, "Open"])
    win = call_hist[(call_hist.index >= entry_d) & (call_hist.index <= exit_target)]
    if len(win) < 2:
        return None
    exit_d = win.index[-1]
    exit_p = float(win.iloc[-1]["Close"])
    max_close = float(win["Close"].max())
    max_high = float(win["High"].max())
    return {
        "entry_date": entry_d.strftime("%Y-%m-%d"),
        "exit_date": exit_d.strftime("%Y-%m-%d"),
        "entry": entry_p, "exit_close": exit_p,
        "pnl_close": exit_p - entry_p,
        "pnl_close_pct": (exit_p / entry_p - 1) * 100 if entry_p > 0 else 0,
        "max_close": max_close,
        "max_pnl_close_pct": (max_close / entry_p - 1) * 100 if entry_p > 0 else 0,
        "max_high": max_high,
    }


def real_short_put_pnl(put_hist, entry_d, hold_days=5):
    """单腿 Short Put P&L (SELL PUT 信号: 卖 ATM Put 收 premium).

    P&L = entry premium - exit premium (期权下跌赚)
    """
    entry_d = pd.Timestamp(entry_d).normalize()
    exit_target = entry_d + pd.Timedelta(days=hold_days)
    if entry_d not in put_hist.index:
        avail = put_hist.index[put_hist.index >= entry_d]
        if len(avail) == 0:
            return None
        entry_d = avail[0]
    entry_p = float(put_hist.loc[entry_d, "Open"])
    win = put_hist[(put_hist.index >= entry_d) & (put_hist.index <= exit_target)]
    if len(win) < 2:
        return None
    exit_d = win.index[-1]
    exit_p = float(win.iloc[-1]["Close"])
    return {
        "entry_date": entry_d.strftime("%Y-%m-%d"),
        "exit_date": exit_d.strftime("%Y-%m-%d"),
        "entry_premium": entry_p,
        "exit_premium": exit_p,
        "pnl_close": entry_p - exit_p,  # 卖期权: 收 - 付
        "pnl_close_pct": (entry_p - exit_p) / entry_p * 100 if entry_p > 0 else 0,
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
    x_d = avail[-1]

    # 入场 (Open): 卖 sc + sp, 买 lc + lp
    sc_e = hists["sc"].loc[e_d, "Open"]
    sp_e = hists["sp"].loc[e_d, "Open"]
    lc_e = hists["lc"].loc[e_d, "Open"]
    lp_e = hists["lp"].loc[e_d, "Open"]
    credit = (sc_e + sp_e) - (lc_e + lp_e)

    # 平仓 (Close): 反向操作
    sc_x = hists["sc"].loc[x_d, "Close"]
    sp_x = hists["sp"].loc[x_d, "Close"]
    lc_x = hists["lc"].loc[x_d, "Close"]
    lp_x = hists["lp"].loc[x_d, "Close"]
    debit = (sc_x + sp_x) - (lc_x + lp_x)
    pnl = credit - debit  # short vol: credit 缩小赚

    return {
        "entry_date": e_d.strftime("%Y-%m-%d"),
        "exit_date": x_d.strftime("%Y-%m-%d"),
        "ic_short_call_k": short_call_k, "ic_long_call_k": long_call_k,
        "ic_short_put_k": short_put_k, "ic_long_put_k": long_put_k,
        "ic_credit": credit, "ic_debit_at_exit": debit,
        "ic_pnl": pnl,
        "ic_pnl_pct_of_credit": pnl / credit * 100 if abs(credit) > 0.01 else 0,
        "ic_max_loss": (wing_offset - short_offset) - credit,  # 最大亏 (理论)
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


def backtest_signal(asset, signal_date, signal_type, spot,
                      hold_days=5, target_dte=45, rv=None):
    """对单个信号跑真实期权 P&L (4 类策略并行)."""
    sig_d = pd.Timestamp(signal_date)
    age_days = (pd.Timestamp.now() - sig_d).days
    # v3.7.39: 智能选 expiry — 老信号用 LEAPS, 新信号用月度
    # v3.7.41: LEAPS strike 由 yfinance 探测 ATM 真实可用 strike (修正错配)
    if age_days >= 90:
        expiry = smart_pick_expiry(sig_d)
        expiry_str = expiry.strftime("%Y-%m-%d")
        strike = leaps_strike(asset, spot, expiry_str)
    else:
        expiry = nearest_monthly_third_friday(sig_d, target_dte)
        expiry_str = expiry.strftime("%Y-%m-%d")
        strike = find_valid_strike(asset, spot, expiry_str)
    actual_dte = (expiry - sig_d).days

    result = {
        "signal_date": signal_date,
        "signal_type": signal_type,
        "spot": spot,
        "strike_atm": strike,
        "expiry": expiry_str,
        "actual_dte": actual_dte,
        "rv": rv,
    }

    # 拉两腿历史 (Call + Put ATM)
    call_sym = occ_symbol(asset, expiry_str, strike, "C")
    put_sym = occ_symbol(asset, expiry_str, strike, "P")
    call_hist = fetch_option_history(call_sym, period="2y")
    put_hist = fetch_option_history(put_sym, period="2y")

    if call_hist is None or put_hist is None:
        result["error"] = "无历史 K 线 (期权可能未出, 或 yf 限制)"
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
    straddle_df = detect_straddle_signal(rv_10d, close.index, rv_pctile=None)
    short_vol_df = detect_short_vol_signal(rv_10d, rv_pct, close.index,
                                              regime=regime,
                                              daily_range=(high-low)/close*100)

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

        meta = {
            "signal_date": d.strftime("%Y-%m-%d"),
            "spot": spot,
            "rv": rv_v,
            "rv_pctile": rvp,
            "regime": reg,
            "days_to_fomc": d_fomc,
            "days_to_nfp": d_nfp,
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
    print(f"\n保存 → {fpath}, 共 {len(df)} 条记录\n")

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
