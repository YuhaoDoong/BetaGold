"""测试 BC 入场组合过滤 — 找出最佳过滤组合.

核心发现 (entry_signal_factor_analysis):
  ★ ma_trend < 0.99 时 BC wr ~0-12% (下行趋势绝对不要做 BC)
  ★ atr_ratio < 0.8 时 BC wr 14-52% (波动收缩 BC 无动力)
  ★ close_vs_ma20 < 0.97 时 BC wr 20-37% (真破位)
  ★ stoch_k 30-50 + SLV: wr 0% (中段反弹力枯竭)
  ★ vol_ratio > 1.6: wr 33% (恐慌放量反指标, GLD)

验证: 应用这些过滤后 BC 净 wr 提升多少, 留多少笔.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np

CSV = Path("/Users/yhdong/Gold/data/backtest_history")


def add_features(df, asset):
    ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                        index_col=0, parse_dates=True)
    pc = ohlc["Close"].shift(1)
    tr = pd.concat([(ohlc["High"]-ohlc["Low"]).abs(),
                      (ohlc["High"]-pc).abs(), (ohlc["Low"]-pc).abs()],
                     axis=1).max(axis=1)
    ohlc["atr_ratio"] = tr.rolling(14).mean() / tr.rolling(50).mean()
    ohlc["vol_ratio"] = ohlc["Volume"] / ohlc["Volume"].rolling(20).mean()
    ohlc["ma20"] = ohlc["Close"].rolling(20).mean()
    ohlc["ma50"] = ohlc["Close"].rolling(50).mean()
    ohlc["close_vs_ma20"] = ohlc["Close"] / ohlc["ma20"]
    ohlc["ma_trend"] = ohlc["ma20"] / ohlc["ma50"]
    rows = []
    for d in df["signal_date"]:
        d = pd.Timestamp(d).normalize()
        if d in ohlc.index:
            r = ohlc.loc[d]
            rows.append({"atr_ratio": r["atr_ratio"], "vol_ratio": r["vol_ratio"],
                          "close_vs_ma20": r["close_vs_ma20"],
                          "ma_trend": r["ma_trend"]})
        else:
            rows.append({k: np.nan for k in ["atr_ratio","vol_ratio",
                                                  "close_vs_ma20","ma_trend"]})
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def test_filter(bc, name, mask):
    """单一过滤条件下 BC 子集 wr."""
    sub = bc[mask]
    n_skip = (~mask).sum()
    if not len(sub):
        print(f"  {name:<35} 全过滤掉 ({n_skip} skip)")
        return
    wr = (sub["pnl_pct"] > 0).mean() * 100
    m = sub["pnl_pct"].mean()
    s = sub["pnl_pct"].sum()
    print(f"  {name:<35} keep={len(sub):>3} skip={n_skip:>3}  "
          f"wr={wr:>5.1f}% mean={m:>+6.2f}% sum={s:>+8.1f}%")


for asset in ["GLD", "SLV"]:
    print(f"\n{'='*70}\n{asset} BC 入场过滤组合测试\n{'='*70}")
    df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                      parse_dates=["signal_date"])
    bc = df[df["strategy"]=="BUY CALL"].copy()
    bc = add_features(bc, asset)
    bc = bc.dropna(subset=["atr_ratio","ma_trend","close_vs_ma20"])

    n0 = len(bc)
    wr0 = (bc["pnl_pct"]>0).mean()*100
    m0 = bc["pnl_pct"].mean()
    s0 = bc["pnl_pct"].sum()
    print(f"全集 baseline:  n={n0} wr={wr0:.1f}% mean={m0:+.2f}% sum={s0:+.1f}%")

    print("\n【单一过滤】 (bc 入场需满足):")
    test_filter(bc, "ma_trend >= 0.99",  bc["ma_trend"] >= 0.99)
    test_filter(bc, "atr_ratio >= 0.8",   bc["atr_ratio"] >= 0.8)
    test_filter(bc, "close_vs_ma20 >= 0.97", bc["close_vs_ma20"] >= 0.97)
    test_filter(bc, "vol_ratio < 1.6",    bc["vol_ratio"] < 1.6)
    test_filter(bc, "iv_rv_gap_pct < 5",  bc["iv_rv_gap_pct"] < 5)
    test_filter(bc, "rsi_14 >= 30",       bc["rsi_14"] >= 30)

    print("\n【双因子组合】:")
    test_filter(bc, "ma_trend>=0.99 + atr>=0.8",
                  (bc["ma_trend"]>=0.99) & (bc["atr_ratio"]>=0.8))
    test_filter(bc, "ma_trend>=0.99 + close>=0.97",
                  (bc["ma_trend"]>=0.99) & (bc["close_vs_ma20"]>=0.97))
    test_filter(bc, "atr>=0.8 + close>=0.97",
                  (bc["atr_ratio"]>=0.8) & (bc["close_vs_ma20"]>=0.97))

    print("\n【三因子组合 (推荐)】:")
    f3 = (bc["ma_trend"]>=0.99) & (bc["atr_ratio"]>=0.8) & (bc["close_vs_ma20"]>=0.97)
    test_filter(bc, "ma_trend + atr + close",  f3)
    f3b = (bc["ma_trend"]>=0.99) & (bc["atr_ratio"]>=0.8) & (bc["vol_ratio"]<1.6)
    test_filter(bc, "ma_trend + atr + vol<1.6",  f3b)

    print("\n【四因子组合 (强过滤)】:")
    f4 = (bc["ma_trend"]>=0.99) & (bc["atr_ratio"]>=0.8) \
         & (bc["close_vs_ma20"]>=0.97) & (bc["vol_ratio"]<1.6)
    test_filter(bc, "ma + atr + close + vol",  f4)
