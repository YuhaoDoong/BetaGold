"""v3.7.200 双任务 grid

Q1: regime classifier 优化 — 让 2026Q1 暴跌段被识别为 Mix/Bear
   - lever: bull_threshold / bear_threshold / smooth_window / price_momentum 权重
   - 评判: 2026Q1 期间各 regime 占比 + 是否过早把 2024 牛市判 Bear

Q2: 方向性信号 filter combo 优化
   - 在现有 IV + ma_trend 基础上, 加上 ret_20d (跌幅过滤) + bp_low + regime 联合过滤
   - 3y 窗口, 5d/10d/20d hold
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf, copy
import numpy as np

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier
import core.regime as rmod
import core.strategy_config as sc


def build_inputs():
    cfg = load_config()
    oos = load_oos_predictions(cfg)
    feat = pd.read_parquet("/Users/yhdong/Gold/data/processed/features_all.parquet")
    ohlc = pd.read_csv("/Users/yhdong/Gold/data/raw/market/gld.csv",
                          index_col=0, parse_dates=True)
    common = ohlc.index.intersection(feat.index).intersection(oos.index)
    close = ohlc.loc[common,"Close"]; high = ohlc.loc[common,"High"]; low = ohlc.loc[common,"Low"]
    upper, lower, _ = build_band(oos.loc[common], close)
    rv_p = compute_rv_pctile(feat.loc[common,"rv_10d"])
    feat_cols = [c for c in feat.columns if not c.startswith("fwd_")]
    gvz = yf.Ticker("^GVZ").history(period="5y")
    gvz.index = pd.to_datetime(gvz.index).tz_localize(None).normalize()
    return ohlc, feat, feat_cols, close, high, low, upper, lower, rv_p, gvz["Close"]


# ===== Q1: REGIME GRID =====

def regime_summary(scores: pd.DataFrame, label: str) -> dict:
    """评估 regime 分布 + 2026Q1 暴跌段是否被识别."""
    q1 = scores[(scores.index >= "2026-01-30") & (scores.index <= "2026-04-30")]
    q1_dist = q1["regime"].value_counts().to_dict()
    n_q1 = len(q1)
    pct_bull_q1 = q1_dist.get("Bull", 0) / n_q1 * 100 if n_q1 else 0
    pct_mix_q1 = q1_dist.get("Mixed", 0) / n_q1 * 100 if n_q1 else 0
    pct_bear_q1 = q1_dist.get("Bear", 0) / n_q1 * 100 if n_q1 else 0

    # 2024 牛市段 (2024-03 → 2025-02) 应该是 Bull
    bull24 = scores[(scores.index >= "2024-03-01") & (scores.index <= "2025-02-28")]
    pct_bull_24 = (bull24["regime"] == "Bull").mean() * 100
    return {
        "config": label,
        "Q1暴跌_Bull%": round(pct_bull_q1, 1),
        "Q1暴跌_Mix%": round(pct_mix_q1, 1),
        "Q1暴跌_Bear%": round(pct_bear_q1, 1),
        "2024牛_Bull%": round(pct_bull_24, 1),
    }


def regime_grid(feat, feat_cols):
    print("=" * 80)
    print("Q1: REGIME 分类 grid (2026/1/30-4/30 暴跌段)")
    print("=" * 80)
    rows = []
    # baseline
    rows.append(regime_summary(
        RegimeClassifier().classify(feat[feat_cols]),
        "BASELINE (bull=+0.2 bear=-0.2 sw=60)"))
    # 单独压 bull_threshold
    for bt in [0.3, 0.4, 0.5]:
        rows.append(regime_summary(
            RegimeClassifier(bull_threshold=bt).classify(feat[feat_cols]),
            f"bull_thr={bt}"))
    # 单独压 smooth_window
    for sw in [10, 20, 30]:
        rows.append(regime_summary(
            RegimeClassifier(smooth_window=sw).classify(feat[feat_cols]),
            f"sw={sw}"))
    # 组合
    for bt, sw in [(0.4, 20), (0.5, 30), (0.4, 30), (0.5, 20)]:
        rows.append(regime_summary(
            RegimeClassifier(bull_threshold=bt, smooth_window=sw).classify(feat[feat_cols]),
            f"bull_thr={bt} sw={sw}"))
    # 同时 bear_threshold 上调
    for bt, br, sw in [(0.4, -0.05, 20), (0.4, -0.1, 30), (0.3, -0.05, 20)]:
        rows.append(regime_summary(
            RegimeClassifier(bull_threshold=bt, bear_threshold=br,
                                  smooth_window=sw).classify(feat[feat_cols]),
            f"bull={bt} bear={br} sw={sw}"))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ===== Q2: 信号 filter combo =====

def evaluate_filter(df_signals: pd.DataFrame, mask: pd.Series, ohlc) -> dict:
    """对 mask=True 的信号子集, 算 5d/10d hold P&L."""
    sub = df_signals[mask]
    out = {"n": len(sub)}
    if not len(sub): return out
    for h in (5, 10, 20):
        s = sub[f"r{h}d"].dropna()
        if len(s):
            out[f"WR{h}d"] = round((s > 0).mean() * 100, 1)
            out[f"sum{h}d"] = round(s.sum(), 1)
            out[f"mean{h}d"] = round(s.mean(), 2)
    return out


def signal_filter_grid(ohlc, feat, feat_cols, close, high, low, upper, lower,
                         rv_p, gvz):
    print("\n" + "=" * 80)
    print("Q2: 方向性信号 filter combo (3y, GLD only)")
    print("=" * 80)

    # 用当前 regime cfg 跑信号 (含 IV filter, ma_trend=0.975)
    regime = RegimeClassifier().classify(feat.loc[close.index, feat_cols])["regime"]
    sig = generate_daily_signals(close, high, low, upper, lower, regime, rv_p,
                                       asset="GLD", gvz_series=gvz)
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=3*365)
    buy = sig[sig["buy_signal"] & (sig.index >= cutoff)].copy()

    # 加 ret_20d 列 (近 20 日累计涨跌)
    ret20 = close.pct_change(20)
    buy["ret_20d"] = ret20.reindex(buy.index)
    # 加 GVZ
    buy["GVZ"] = gvz.reindex(buy.index)
    # 加 forward returns
    for h in (5, 10, 20):
        for sig_d, _ in buy.iterrows():
            if sig_d not in ohlc.index: continue
            idx = ohlc.index.get_loc(sig_d)
            entry = float(ohlc.iloc[idx]["Open"])
            if idx + h < len(ohlc):
                exit_p = float(ohlc.iloc[idx+h]["Close"])
                buy.loc[sig_d, f"r{h}d"] = (exit_p / entry - 1) * 100
            else:
                buy.loc[sig_d, f"r{h}d"] = np.nan

    # baseline
    print(f"\nBASELINE (现 prod cfg, 3y BUY 信号): {evaluate_filter(buy, pd.Series(True, buy.index), ohlc)}")

    # 单因子 + 组合 filter
    filters = {
        "bp_low<=0.1 (极深破)": buy["bp_low"] <= 0.10,
        "bp_low<=0.15":         buy["bp_low"] <= 0.15,
        "bp_low<=0.20":         buy["bp_low"] <= 0.20,
        "ret_20d > -3%":        buy["ret_20d"] > -0.03,
        "ret_20d > -5%":        buy["ret_20d"] > -0.05,
        "ret_20d > 0 (上行段)":  buy["ret_20d"] > 0,
        "ma_trend >= 0.99":     buy["ma_trend"] >= 0.99,
        "ma_trend < 1.0 (pullback)": buy["ma_trend"] < 1.0,
        "rv_pct < 0.5":         buy["rv_pctile"] < 0.50,
        "rv_pct < 0.75":        buy["rv_pctile"] < 0.75,
        "GVZ < 25":             buy["GVZ"] < 25,
        # 组合
        "bp_low≤0.15 AND ret_20d>-3%": (buy["bp_low"]<=0.15) & (buy["ret_20d"]>-0.03),
        "bp_low≤0.20 AND ret_20d>-3%": (buy["bp_low"]<=0.20) & (buy["ret_20d"]>-0.03),
        "bp_low≤0.15 AND rv<0.5":      (buy["bp_low"]<=0.15) & (buy["rv_pctile"]<0.50),
        "bp_low≤0.15 AND ret_20d>0":   (buy["bp_low"]<=0.15) & (buy["ret_20d"]>0),
        "ret_20d>-3% AND rv<0.5":      (buy["ret_20d"]>-0.03) & (buy["rv_pctile"]<0.50),
        "ret_20d>0 AND ma_trend<1.0":  (buy["ret_20d"]>0) & (buy["ma_trend"]<1.0),
    }
    rows = []
    for name, mask in filters.items():
        r = evaluate_filter(buy, mask, ohlc)
        r["filter"] = name
        rows.append(r)
    df = pd.DataFrame(rows)
    cols = ["filter","n","WR5d","sum5d","mean5d","WR10d","sum10d","WR20d","sum20d"]
    print(df[cols].to_string(index=False))

    # 看 2026Q1 这 20 笔过滤情况
    print(f"\n2026Q1 (n=20) 在各 filter 下保留情况:")
    q1mask = (buy.index >= "2026-01-01") & (buy.index <= "2026-03-31")
    for name, mask in filters.items():
        q1_kept = (mask & q1mask).sum()
        kept_pnl = buy.loc[mask & q1mask, "r5d"].sum() if q1_kept else 0
        print(f"  {name:42}: Q1 保留 {q1_kept:2d}/20  Q1 sum {kept_pnl:+.1f}%")


def main():
    inputs = build_inputs()
    ohlc, feat, feat_cols, close, high, low, upper, lower, rv_p, gvz = inputs
    regime_grid(feat, feat_cols)
    signal_filter_grid(ohlc, feat, feat_cols, close, high, low, upper, lower, rv_p, gvz)


if __name__ == "__main__":
    main()
