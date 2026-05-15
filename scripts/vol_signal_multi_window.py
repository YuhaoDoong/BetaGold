"""v3.7.211 波动率信号 (STRADDLE + SHORT_VOL) 多窗口 grid

STRADDLE 测:
  - straddle_priority_score (5/6/7/8)
  - straddle_rv_pctile_max (0.3/0.5/0.7/1.0)
  - hold_days (10/14/21/30)

SHORT_VOL 测:
  - 是否解除 disabled
  - 当前 cfg 在近 1y / 3y 表现 (跟 v3.7.177 disable 时对比)

数据源:
  ≤ 1y kline_db (真实期权)
  > 1y BS proxy
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf
import numpy as np

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier
from core.events import detect_straddle_signal, detect_short_vol_signal
from core.paper_positions import (_load_kline_db, pick_liquid_monthly_option,
                                      interpolate_option_intraday, bs_price)
from core.strategies.straddle import simulate_straddle_position, StraddleConfig
from core.strategies.short_vol import simulate_short_vol_position, ShortVolConfig


def build_inputs():
    cfg = load_config()
    oos = load_oos_predictions(cfg)
    feat = pd.read_parquet("/Users/yhdong/Gold/data/processed/features_all.parquet")
    ohlc = pd.read_csv("/Users/yhdong/Gold/data/raw/market/gld.csv",
                          index_col=0, parse_dates=True)
    common = ohlc.index.intersection(feat.index).intersection(oos.index)
    close = ohlc.loc[common,"Close"]; high = ohlc.loc[common,"High"]; low = ohlc.loc[common,"Low"]
    rv_p = compute_rv_pctile(feat.loc[common,"rv_10d"])
    gvz = yf.Ticker("^GVZ").history(period="5y")
    gvz.index = pd.to_datetime(gvz.index).tz_localize(None).normalize()
    return cfg, close, high, low, rv_p, feat.loc[common,"rv_10d"], gvz["Close"], ohlc


def sim_straddle_bs(sig_d, entry_spot, ohlc, gvz, hold_max=21):
    """BS 长 straddle proxy: ATM call + ATM put DTE=14."""
    DTE = 14
    K = round(entry_spot)
    iv0 = float(gvz.get(sig_d, 0.18)) / 100 if sig_d in gvz.index else 0.18
    iv0 = max(0.10, min(0.50, iv0))
    T0 = DTE / 365.0
    c0 = bs_price(entry_spot, K, T0, 0.04, iv0, "C")
    p0 = bs_price(entry_spot, K, T0, 0.04, iv0, "P")
    entry = c0 + p0
    if entry <= 0.01: return None
    tp = entry * 2.0  # +100% TP
    today = ohlc.index.max()
    days_after = ohlc.index[ohlc.index > sig_d]
    hold = 0; cur = entry
    for d in days_after:
        if d > today: break
        hold += 1
        if hold > hold_max: break
        spot_d = float(ohlc.loc[d, "Close"])
        T_d = (DTE - hold) / 365.0
        if T_d <= 0:
            cur = max(spot_d - K, 0) + max(K - spot_d, 0)
            pnl = (cur / entry - 1) * 100
            return {"pnl_pct": pnl, "hold": hold, "reason": "expiry"}
        iv_d = float(gvz.get(d, iv0*100)) / 100 if d in gvz.index else iv0
        iv_d = max(0.10, min(0.50, iv_d))
        c = bs_price(spot_d, K, T_d, 0.04, iv_d, "C")
        p = bs_price(spot_d, K, T_d, 0.04, iv_d, "P")
        cur = c + p
        pnl = (cur / entry - 1) * 100
        if cur >= tp:
            return {"pnl_pct": pnl, "hold": hold, "reason": "+100% TP"}
    pnl = (cur / entry - 1) * 100
    return {"pnl_pct": pnl, "hold": hold, "reason": f"{hold_max}d 定时"}


def sim_short_vol_bs(sig_d, entry_spot, ohlc, gvz, hold_max=30):
    """BS Iron Condor: -ATM±3% / +ATM±7% DTE=30."""
    DTE = 30
    sp_k = round(entry_spot * 0.97); lp_k = round(entry_spot * 0.93)
    sc_k = round(entry_spot * 1.03); lc_k = round(entry_spot * 1.07)
    iv0 = float(gvz.get(sig_d, 0.18)) / 100 if sig_d in gvz.index else 0.18
    iv0 = max(0.10, min(0.50, iv0))
    T0 = DTE / 365.0
    sp0 = bs_price(entry_spot, sp_k, T0, 0.04, iv0, "P")
    lp0 = bs_price(entry_spot, lp_k, T0, 0.04, iv0, "P")
    sc0 = bs_price(entry_spot, sc_k, T0, 0.04, iv0, "C")
    lc0 = bs_price(entry_spot, lc_k, T0, 0.04, iv0, "C")
    credit = (sp0 - lp0) + (sc0 - lc0)
    width = max(sp_k - lp_k, lc_k - sc_k)
    max_risk = width - credit
    if credit <= 0.01 or max_risk <= 0: return None
    tp_credit = credit * 0.5  # 收 50% credit 平
    sl_credit = credit + 0.5 * max_risk
    today = ohlc.index.max()
    days_after = ohlc.index[ohlc.index > sig_d]
    hold = 0; cur = credit
    for d in days_after:
        if d > today: break
        hold += 1
        if hold > hold_max: break
        spot_d = float(ohlc.loc[d, "Close"])
        T_d = (DTE - hold) / 365.0
        if T_d <= 0:
            # expiry intrinsic
            sp_e = max(sp_k - spot_d, 0); lp_e = max(lp_k - spot_d, 0)
            sc_e = max(spot_d - sc_k, 0); lc_e = max(spot_d - lc_k, 0)
            cur = (sp_e - lp_e) + (sc_e - lc_e)
            pnl = (credit - cur) / max_risk * 100
            return {"pnl_pct": pnl, "hold": hold, "reason": "expiry"}
        iv_d = float(gvz.get(d, iv0*100)) / 100 if d in gvz.index else iv0
        iv_d = max(0.10, min(0.50, iv_d))
        sp = bs_price(spot_d, sp_k, T_d, 0.04, iv_d, "P")
        lp = bs_price(spot_d, lp_k, T_d, 0.04, iv_d, "P")
        sc = bs_price(spot_d, sc_k, T_d, 0.04, iv_d, "C")
        lc = bs_price(spot_d, lc_k, T_d, 0.04, iv_d, "C")
        cur = (sp - lp) + (sc - lc)
        pnl = (credit - cur) / max_risk * 100
        if cur <= tp_credit:
            return {"pnl_pct": pnl, "hold": hold, "reason": "+50% TP"}
        if cur >= sl_credit:
            return {"pnl_pct": pnl, "hold": hold, "reason": "-50% SL"}
    pnl = (credit - cur) / max_risk * 100
    return {"pnl_pct": pnl, "hold": hold, "reason": f"{hold_max}d"}


def main():
    cfg, close, high, low, rv_p, rv_abs, gvz, ohlc = build_inputs()
    today = ohlc.index.max()

    # ── STRADDLE detect (per window 不变 cfg, 只看 score/RV/event filter) ──
    strad = detect_straddle_signal(rv_abs, close.index, rv_pctile=rv_p,
                                          close=close, high=high, low=low,
                                          asset="GLD")
    sv = detect_short_vol_signal(rv_abs, rv_p, close.index, regime=None,
                                       close=close, high=high, low=low,
                                       asset="GLD")
    print(f"STRADDLE 历史触发: {strad['straddle_signal'].sum()} 笔")
    print(f"SHORT_VOL 历史触发: {sv['short_vol_signal'].sum()} 笔\n")

    # ── 1. STRADDLE 多窗口 (用现 detect filter + 多 hold) ──
    print("=" * 100)
    print("1. STRADDLE × 多窗口 × hold grid (现 detect filter: score>=6, rv_pct<0.5, event<=5d)")
    print("=" * 100)
    strad_idx = strad[strad['straddle_signal']].index
    windows = [('5y',5*365),('3y',3*365),('1y',365),('6m',180),('3m',90)]
    print(f'{"hold":>5}  ' + '  '.join(f'{w:>22}' for w,_ in windows))
    for hold in [10, 14, 21, 30, 45]:
        line = f'{hold:>3}d  '
        for w_name, w_days in windows:
            sub_idx = [d for d in strad_idx if d >= today - pd.Timedelta(days=w_days)]
            if not sub_idx: line += f'  {"n=0":>22}'; continue
            rows = []
            for sig_d in sub_idx:
                if sig_d not in ohlc.index: continue
                entry_spot = float(ohlc.loc[sig_d, "Open"])
                r = sim_straddle_bs(sig_d, entry_spot, ohlc, gvz, hold_max=hold)
                if r: rows.append(r["pnl_pct"])
            if not rows: line += f'  {"n=0":>22}'; continue
            p = pd.Series(rows)
            line += f'  n={len(p):2d} WR={(p>0).mean()*100:.0f}% sum={p.sum():+.0f}%  '[:24].rjust(22)
        print(line)

    # ── 2. SHORT_VOL 多窗口 (现 disabled, 看真实数据值不值得解除) ──
    print("\n" + "=" * 100)
    print("2. SHORT_VOL × 多窗口 (BS IC -ATM±3% / +±7%, DTE=30, TP 50% / SL 50%)")
    print("=" * 100)
    sv_idx = sv[sv['short_vol_signal']].index
    print(f"SHORT_VOL signal n total: {len(sv_idx)}")
    print(f'{"hold":>5}  ' + '  '.join(f'{w:>22}' for w,_ in windows))
    for hold in [21, 30, 45]:
        line = f'{hold:>3}d  '
        for w_name, w_days in windows:
            sub_idx = [d for d in sv_idx if d >= today - pd.Timedelta(days=w_days)]
            if not sub_idx: line += f'  {"n=0":>22}'; continue
            rows = []
            for sig_d in sub_idx:
                if sig_d not in ohlc.index: continue
                entry_spot = float(ohlc.loc[sig_d, "Open"])
                r = sim_short_vol_bs(sig_d, entry_spot, ohlc, gvz, hold_max=hold)
                if r: rows.append(r["pnl_pct"])
            if not rows: line += f'  {"n=0":>22}'; continue
            p = pd.Series(rows)
            line += f'  n={len(p):2d} WR={(p>0).mean()*100:.0f}% sum={p.sum():+.0f}%  '[:24].rjust(22)
        print(line)

    # ── 3. STRADDLE score threshold grid (内置 detect 调) ──
    print("\n" + "=" * 100)
    print("3. STRADDLE priority_score 阈值 × 多窗口 (hold=21 固定)")
    print("=" * 100)
    print(f'{"score":>5}  ' + '  '.join(f'{w:>22}' for w,_ in windows))
    for score_thr in [4, 5, 6, 7, 8]:
        line = f'>={score_thr}     '
        for w_name, w_days in windows:
            sub_strad = strad[(strad['straddle_signal']) &
                                  (strad.get('straddle_score', strad.get('score',0)) >= score_thr)] \
                              if 'straddle_score' in strad.columns or 'score' in strad.columns \
                              else strad[strad['straddle_signal']]
            # 没 score 列就只用 signal
            sub_idx = [d for d in sub_strad.index if d >= today - pd.Timedelta(days=w_days)]
            if not sub_idx: line += f'  {"n=0":>22}'; continue
            rows = []
            for sig_d in sub_idx:
                if sig_d not in ohlc.index: continue
                entry_spot = float(ohlc.loc[sig_d, "Open"])
                r = sim_straddle_bs(sig_d, entry_spot, ohlc, gvz, hold_max=21)
                if r: rows.append(r["pnl_pct"])
            if not rows: line += f'  {"n=0":>22}'; continue
            p = pd.Series(rows)
            line += f'  n={len(p):2d} WR={(p>0).mean()*100:.0f}% sum={p.sum():+.0f}%  '[:24].rjust(22)
        print(line)


if __name__ == "__main__":
    main()
