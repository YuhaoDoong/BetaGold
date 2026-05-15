"""v3.7.200 GLD BC 信号 纯现货基线 — 看信号本身 WR

不挑期权, 不跑 TP/SL, 单纯:
  sig_d 触发 BUY → 当日 open 买 GLD spot, 5/10/20 日后 close 卖
  统计 WR / mean return / std

回答的问题:
  - 信号本身是否赚钱 (信号 alpha)
  - BC 期权 29% WR 是不是因为 (a) 信号烂 还是 (b) 期权包装烂
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier


def collect_signals(asset="GLD"):
    cfg = load_config()
    oos = load_oos_predictions(cfg)
    feat = pd.read_parquet("/Users/yhdong/Gold/data/processed/features_all.parquet")
    ohlc = pd.read_csv("/Users/yhdong/Gold/data/raw/market/gld.csv",
                          index_col=0, parse_dates=True)
    common = ohlc.index.intersection(feat.index).intersection(oos.index)
    close = ohlc.loc[common, "Close"]; high = ohlc.loc[common, "High"]
    low = ohlc.loc[common, "Low"]
    upper, lower, _ = build_band(oos.loc[common], close)
    feat_cols = [c for c in feat.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(feat.loc[common, feat_cols])["regime"]
    rv_p = compute_rv_pctile(feat.loc[common, "rv_10d"])
    sig = generate_daily_signals(close, high, low, upper, lower, regime, rv_p, asset=asset)
    bc = sig[sig["buy_type"] == "BUY CALL"].copy()
    bc = bc.join(ohlc[["Open", "Close"]], how="inner")
    return bc, ohlc


def main():
    bc, ohlc = collect_signals("GLD")
    print(f"GLD BUY CALL 信号 (全历史): {len(bc)} 笔")
    print(f"  range: {bc.index.min().date()} → {bc.index.max().date()}\n")

    # 多 horizon 5/10/20 日
    rows = []
    for sig_d, _ in bc.iterrows():
        entry = float(ohlc.loc[sig_d, "Open"])  # 信号日 open 买
        d_idx = ohlc.index.get_loc(sig_d)
        rec = {"sig_d": sig_d.date(), "entry": entry}
        for h in [5, 10, 20]:
            exit_idx = d_idx + h
            if exit_idx >= len(ohlc):
                rec[f"h{h}_pnl"] = None
                continue
            exit_p = float(ohlc.iloc[exit_idx]["Close"])
            rec[f"h{h}_pnl"] = (exit_p / entry - 1) * 100
        rows.append(rec)

    df = pd.DataFrame(rows)
    print(f"全历史 GLD BC 信号现货持仓回测:")
    print("="*60)
    for h in [5, 10, 20]:
        col = f"h{h}_pnl"
        sub = df[col].dropna()
        n = len(sub)
        if n == 0: continue
        wr = (sub > 0).mean() * 100
        print(f"  hold {h:2d}d: n={n:3d}, WR={wr:.1f}%, "
              f"mean={sub.mean():+.2f}%, median={sub.median():+.2f}%, "
              f"sum={sub.sum():+.1f}%, std={sub.std():.2f}%, "
              f"max={sub.max():+.1f}%, min={sub.min():+.1f}%")

    # 按年看
    df["sig_d_ts"] = pd.to_datetime(df["sig_d"])
    df["year"] = df["sig_d_ts"].dt.year
    print(f"\n按年看 5 日持仓:")
    print("="*60)
    for year, sub in df.groupby("year"):
        s = sub["h5_pnl"].dropna()
        if not len(s): continue
        wr = (s > 0).mean() * 100
        print(f"  {year}: n={len(s):3d}, WR={wr:.1f}%, mean={s.mean():+.2f}%, "
              f"sum={s.sum():+.1f}%")

    # 按季度看近 1y (跟 BC option 同窗口)
    df_recent = df[df["sig_d_ts"] >= pd.Timestamp("2025-05-01")]
    df_recent["quarter"] = df_recent["sig_d_ts"].dt.to_period("Q")
    print(f"\n近 1y 按季度看 5 日持仓 (跟 BC 期权回测同窗口):")
    print("="*60)
    for q, sub in df_recent.groupby("quarter"):
        s = sub["h5_pnl"].dropna()
        if not len(s): continue
        wr = (s > 0).mean() * 100
        print(f"  {q}: n={len(s):3d}, WR={wr:.1f}%, mean={s.mean():+.2f}%, "
              f"sum={s.sum():+.1f}%, max={s.max():+.1f}%, min={s.min():+.1f}%")

    out = "/Users/yhdong/Gold/data/backtest_history/bc_gld_spot_baseline.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
