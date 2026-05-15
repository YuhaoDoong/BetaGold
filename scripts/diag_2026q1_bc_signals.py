"""v3.7.200 诊断 2026Q1 13 笔逆势 BC 信号根因.

列每笔:
  signal day OHLC, 5日后 close (实际结果)
  regime classifier 输出 (bull/bear/range)
  bp_low / bp_high
  rv_pctile, GVZ (IV)
  spot vs 20MA / 50MA / 200MA (趋势位置)
  v3.7.117 IV 三阶过滤是否生效

目标: 找出共同特征 → 提炼"应该拒绝"的条件
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import yfinance as yf

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier


def main():
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

    # GVZ (IV) for IV 三阶
    gvz = yf.Ticker("^GVZ").history(period="5y")
    gvz.index = pd.to_datetime(gvz.index).tz_localize(None).normalize()
    gvz_close = gvz["Close"]

    sig = generate_daily_signals(close, high, low, upper, lower, regime, rv_p,
                                       asset="GLD", gvz_series=gvz_close)
    # 2026Q1 BC 信号
    bc = sig[(sig["buy_type"] == "BUY CALL")
              & (sig.index >= "2026-01-01") & (sig.index <= "2026-03-31")].copy()
    print(f"2026Q1 GLD BC 信号: {len(bc)} 笔\n")
    print(f"sig_df columns: {sig.columns.tolist()}\n")

    # 计算趋势指标
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    # bp_low: position relative to lower band (depth of break)

    rows = []
    for sig_d in bc.index:
        eO = float(ohlc.loc[sig_d, "Open"])
        eC = float(ohlc.loc[sig_d, "Close"])
        # 5 日后
        d_idx = ohlc.index.get_loc(sig_d)
        f5 = float(ohlc.iloc[d_idx + 5]["Close"]) if d_idx + 5 < len(ohlc) else None
        f10 = float(ohlc.iloc[d_idx + 10]["Close"]) if d_idx + 10 < len(ohlc) else None
        pnl5 = (f5 / eO - 1) * 100 if f5 else None

        u = float(upper.loc[sig_d]) if sig_d in upper.index else None
        l = float(lower.loc[sig_d]) if sig_d in lower.index else None
        bp_low = ((u - eO) / (u - l)) if (u and l and u > l) else None  # 1=底, 0=顶
        rg = regime.loc[sig_d] if sig_d in regime.index else "?"
        rvp = float(rv_p.loc[sig_d]) if sig_d in rv_p.index else None
        gvz_v = float(gvz_close.loc[sig_d]) if sig_d in gvz_close.index else None
        gvz_pct = float((gvz_close.loc[:sig_d].iloc[-252:].rank(pct=True)
                          .iloc[-1])) if sig_d in gvz_close.index else None

        m20 = float(ma20.loc[sig_d]) if sig_d in ma20.index else None
        m50 = float(ma50.loc[sig_d]) if sig_d in ma50.index else None
        m200 = float(ma200.loc[sig_d]) if sig_d in ma200.index else None

        rows.append({
            "sig_d": sig_d.date(),
            "spot": round(eO, 1),
            "5d_pnl": f"{pnl5:+.1f}%" if pnl5 is not None else "?",
            "10d_pnl": f"{((f10/eO-1)*100):+.1f}%" if f10 else "?",
            "regime": str(rg)[:6],
            "bp_low": round(bp_low, 3) if bp_low else None,
            "rv_pct": round(rvp, 2) if rvp else None,
            "GVZ": round(gvz_v, 1) if gvz_v else None,
            "GVZ_pct(1y)": round(gvz_pct, 2) if gvz_pct else None,
            "vs_20MA": f"{(eO/m20-1)*100:+.1f}%" if m20 else "?",
            "vs_50MA": f"{(eO/m50-1)*100:+.1f}%" if m50 else "?",
            "vs_200MA": f"{(eO/m200-1)*100:+.1f}%" if m200 else "?",
            "buy_signal_raw": bool(bc.loc[sig_d, "buy_signal"]),
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(df.to_string(index=False))

    # 共性提炼
    print("\n" + "=" * 60)
    print("共性提炼")
    print("=" * 60)
    win = df[df["5d_pnl"].str.startswith("+")]
    lose = df[df["5d_pnl"].str.startswith("-")]
    print(f"\nwin (n={len(win)}):")
    if len(win):
        print(f"  vs_20MA 中位: {pd.Series([float(s.rstrip('%')) for s in win['vs_20MA']]).median():+.2f}%")
        print(f"  bp_low 中位: {win['bp_low'].median():.3f}")
        print(f"  regime: {win['regime'].value_counts().to_dict()}")
    print(f"\nlose (n={len(lose)}):")
    if len(lose):
        print(f"  vs_20MA 中位: {pd.Series([float(s.rstrip('%')) for s in lose['vs_20MA']]).median():+.2f}%")
        print(f"  bp_low 中位: {lose['bp_low'].median():.3f}")
        print(f"  regime: {lose['regime'].value_counts().to_dict()}")

    out = "/Users/yhdong/Gold/data/backtest_history/diag_2026q1_bc.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
