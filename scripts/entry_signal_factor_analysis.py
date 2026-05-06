"""方向性入场信号优化 — 哪些因子能区分 BC 赢 vs BC 输?

用 backtest CSV 的 BC 行 (50/47 笔), 看每个特征在 win (pnl>0) vs lose 子集的分布差异.
分化越大的因子越值得加入入场打分.

候选因子:
  A. 区间: bp_low / bp_close / bp_high
  B. 波动率: rv_pctile / gvz_iv_pct / iv_rv_gap_pct
  C. 技术: macd_hist / rsi_14 / stoch_k
  D. Regime: Bull/Mixed/Bear
  E. 派生: rv_pctile vs gvz (已隐含 in IV-RV gap)
  F. ATR / volume (need to compute)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np

CSV = Path("/Users/yhdong/Gold/data/backtest_history")


def add_volume_atr(df, asset):
    """给 BC 信号加 volume_ratio + atr 因子."""
    ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                        index_col=0, parse_dates=True)
    # MA20 volume
    ohlc["vol_ma20"] = ohlc["Volume"].rolling(20).mean()
    ohlc["vol_ratio"] = ohlc["Volume"] / ohlc["vol_ma20"]
    # ATR 14 / ATR 50
    pc = ohlc["Close"].shift(1)
    tr = pd.concat([(ohlc["High"]-ohlc["Low"]).abs(),
                      (ohlc["High"]-pc).abs(), (ohlc["Low"]-pc).abs()],
                     axis=1).max(axis=1)
    ohlc["atr14"] = tr.rolling(14).mean()
    ohlc["atr50"] = tr.rolling(50).mean()
    ohlc["atr_ratio"] = ohlc["atr14"] / ohlc["atr50"]
    # MA20 vs MA50
    ohlc["ma20"] = ohlc["Close"].rolling(20).mean()
    ohlc["ma50"] = ohlc["Close"].rolling(50).mean()
    ohlc["close_vs_ma20"] = ohlc["Close"] / ohlc["ma20"]
    ohlc["ma_trend"] = ohlc["ma20"] / ohlc["ma50"]

    enrich = []
    for d in df["signal_date"]:
        d = pd.Timestamp(d).normalize()
        if d in ohlc.index:
            row = ohlc.loc[d]
            enrich.append({
                "vol_ratio": row.get("vol_ratio", np.nan),
                "atr_ratio": row.get("atr_ratio", np.nan),
                "close_vs_ma20": row.get("close_vs_ma20", np.nan),
                "ma_trend": row.get("ma_trend", np.nan),
            })
        else:
            enrich.append({"vol_ratio": np.nan, "atr_ratio": np.nan,
                           "close_vs_ma20": np.nan, "ma_trend": np.nan})
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(enrich)], axis=1)


def factor_separation(bc, factor, bins, label_fn=None):
    """看 factor 各 bin 内 BC 的 wr 和 mean."""
    print(f"\n  {factor}")
    print(f"    {'区间':<25}{'n':>4}{'wr':>8}{'mean PnL':>11}{'separation':>13}")
    for lo, hi in bins:
        sub = bc[(bc[factor] >= lo) & (bc[factor] < hi)]
        if not len(sub): continue
        wr = (sub["pnl_pct"] > 0).mean() * 100
        m = sub["pnl_pct"].mean()
        sep = wr - 50  # 离 baseline 50% 距离
        label = label_fn(lo, hi) if label_fn else f"[{lo:>+6.2f}, {hi:>+6.2f})"
        print(f"    {label:<25}{len(sub):>4}{wr:>7.1f}%{m:>+10.2f}%{sep:>+12.1f}pp")


def winner_loser_diff(bc):
    """统计每个因子 winner mean vs loser mean."""
    win = bc[bc["pnl_pct"] > 0]
    lose = bc[bc["pnl_pct"] <= 0]
    print(f"\n  Winner ({len(win)}) vs Loser ({len(lose)}) 因子 mean 差异:")
    print(f"    {'factor':<20}{'winner mean':>14}{'loser mean':>13}"
          f"{'diff':>10}{'std normed':>12}")
    factors = ["bp_low", "bp_close", "bp_high", "rv_pctile", "gvz_iv_pct",
                 "iv_rv_gap_pct", "macd_hist", "rsi_14", "stoch_k",
                 "vol_ratio", "atr_ratio", "close_vs_ma20", "ma_trend"]
    rows = []
    for f in factors:
        if f not in bc.columns: continue
        wm = win[f].mean()
        lm = lose[f].mean()
        diff = wm - lm
        std = bc[f].std()
        normed = diff / std if std > 0 else 0
        rows.append((f, wm, lm, diff, normed))
    rows.sort(key=lambda r: abs(r[4]), reverse=True)
    for f, wm, lm, diff, normed in rows:
        marker = "★" if abs(normed) > 0.3 else ""
        print(f"    {f:<20}{wm:>+13.2f}{lm:>+12.2f}{diff:>+9.2f}"
              f"{normed:>+11.2f} {marker}")


for asset in ["GLD", "SLV"]:
    print(f"\n{'='*70}\n{asset} — BC 入场信号因子分析 (winner vs loser)\n{'='*70}")
    df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                      parse_dates=["signal_date"])
    bc = df[df["strategy"]=="BUY CALL"].copy()
    bc = add_volume_atr(bc, asset)
    print(f"BC 总数: {len(bc)}, 整体 wr: {(bc['pnl_pct']>0).mean()*100:.1f}%, "
          f"mean PnL: {bc['pnl_pct'].mean():+.2f}%")

    # 1) Winner vs Loser 因子差异 (找方向感)
    winner_loser_diff(bc)

    # 2) 单因子区间 wr (找入场 cutoff)
    print(f"\n  单因子区间 wr (找最强分化点):")
    factor_separation(bc, "bp_low",
                       [(-1, 0.05), (0.05, 0.15), (0.15, 0.30)])
    factor_separation(bc, "bp_close",
                       [(-1, 0.30), (0.30, 0.50), (0.50, 1.0), (1.0, 2.0)])
    factor_separation(bc, "rsi_14",
                       [(0, 30), (30, 50), (50, 70), (70, 100)])
    factor_separation(bc, "macd_hist",
                       [(-100, -1), (-1, 0), (0, 1), (1, 100)])
    factor_separation(bc, "stoch_k",
                       [(0, 30), (30, 50), (50, 70), (70, 100)])
    factor_separation(bc, "iv_rv_gap_pct",
                       [(-100, -3), (-3, 0), (0, 3), (3, 100)])
    factor_separation(bc, "rv_pctile",
                       [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)])
    factor_separation(bc, "vol_ratio",
                       [(0, 0.8), (0.8, 1.2), (1.2, 1.6), (1.6, 10)])
    factor_separation(bc, "close_vs_ma20",
                       [(0, 0.97), (0.97, 1.0), (1.0, 1.03), (1.03, 2.0)])
    factor_separation(bc, "ma_trend",
                       [(0, 0.99), (0.99, 1.01), (1.01, 1.05), (1.05, 2)])
    factor_separation(bc, "atr_ratio",
                       [(0, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 10)])
