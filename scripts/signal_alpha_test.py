"""v3.7.200 方向性信号 alpha 测试 (不管期权包装)

测的是 generate_daily_signals 的 buy_signal=True 在现货上的真实表现.
3 年 (2023-05-15 → 今), GLD only.

多 horizon: 1d / 5d / 10d / 20d 持仓 close-to-close
按以下维度切片:
  - 整体: 总 WR / sum / mean / max_loss
  - bp_low 桶 (深破/浅破/中位)
  - regime (Bull/Bear/Mixed)
  - GVZ 桶 (低<22 / 中22-28 / 高>=28)
  - rv_pctile 桶
  - ma_trend (above/below 0.99)
  - 按季度
  - 按是否被 IV 过滤拦下 (iv_filter_reason 非空)

输出: per-signal CSV + 多维 summary 表
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


WINDOW_DAYS = 3 * 365  # 近 3 年


def build_sig_with_meta():
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
    regime = RegimeClassifier().classify(feat.loc[common, feat_cols])["regime"]
    gvz = yf.Ticker("^GVZ").history(period="5y")
    gvz.index = pd.to_datetime(gvz.index).tz_localize(None).normalize()
    gvz_s = gvz["Close"]

    # WITHOUT IV filter — 原始信号集
    sig_raw = generate_daily_signals(close, high, low, upper, lower, regime,
                                              rv_p, asset="GLD", gvz_series=None)
    # WITH IV filter (production)
    sig_prod = generate_daily_signals(close, high, low, upper, lower, regime,
                                               rv_p, asset="GLD", gvz_series=gvz_s)
    return ohlc, sig_raw, sig_prod, gvz_s


def main():
    ohlc, sig_raw, sig_prod, gvz_s = build_sig_with_meta()
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=WINDOW_DAYS)
    print(f"窗口: {cutoff.date()} → {ohlc.index.max().date()} (近 {WINDOW_DAYS} 天)")

    # 收集 BUY 信号 (不区分 BC/SP/STRADDLE — 看方向性 alpha)
    buy_raw = sig_raw[sig_raw["buy_signal"] & (sig_raw.index >= cutoff)].copy()
    print(f"原始 BUY 信号 (无 IV 过滤): {len(buy_raw)} 笔")

    # 算 forward returns
    rows = []
    for sig_d, r in buy_raw.iterrows():
        if sig_d not in ohlc.index: continue
        idx = ohlc.index.get_loc(sig_d)
        entry = float(ohlc.iloc[idx]["Open"])  # 信号日 open 买入
        rec = {
            "sig_d": sig_d.date(),
            "entry": round(entry, 2),
            "bp_low": round(float(r.get("bp_low", 0)), 3),
            "regime": str(r.get("regime","-"))[:6],
            "rv_pct": round(float(r.get("rv_pctile", 0)), 2),
            "ma_trend": round(float(r.get("ma_trend", 1.0)), 3),
            "iv_reason_prod": (sig_prod.loc[sig_d, "iv_filter_reason"]
                                 if sig_d in sig_prod.index else ""),
            "blocked_in_prod": (not bool(sig_prod.loc[sig_d, "buy_signal"]))
                                  if sig_d in sig_prod.index else False,
        }
        # GVZ
        if sig_d in gvz_s.index:
            rec["GVZ"] = round(float(gvz_s.loc[sig_d]), 1)
        # forward returns
        for h in [1, 5, 10, 20]:
            if idx + h < len(ohlc):
                exit_p = float(ohlc.iloc[idx + h]["Close"])
                rec[f"r{h}d"] = round((exit_p / entry - 1) * 100, 2)
            else:
                rec[f"r{h}d"] = None
        rows.append(rec)

    df = pd.DataFrame(rows)
    out = "/Users/yhdong/Gold/data/backtest_history/signal_alpha_3y.csv"
    df.to_csv(out, index=False)
    print(f"per-signal CSV: {out}\n")

    # ── 整体 summary (多 horizon) ──
    print("=" * 80)
    print(f"整体 (raw signals, no IV filter, n={len(df)}):")
    print("=" * 80)
    for h in [1, 5, 10, 20]:
        c = f"r{h}d"
        s = df[c].dropna()
        if not len(s): continue
        wr = (s > 0).mean() * 100
        print(f"  {h:2d}d hold: n={len(s):3d}  WR={wr:5.1f}%  "
              f"mean={s.mean():+6.2f}%  sum={s.sum():+7.1f}%  "
              f"max={s.max():+6.1f}%  min={s.min():+6.1f}%")

    # ── 按 IV 过滤前/后对比 (5d hold) ──
    print(f"\n按 IV 过滤切分 (5d hold):")
    print("-" * 60)
    blocked = df[df["blocked_in_prod"] == True]
    kept = df[df["blocked_in_prod"] == False]
    for label, sub in [("IV blocked", blocked), ("IV kept (prod 实际开仓的)", kept)]:
        s = sub["r5d"].dropna()
        if not len(s): print(f"  {label}: n=0"); continue
        wr = (s>0).mean()*100
        print(f"  {label}: n={len(s):3d}  WR={wr:5.1f}%  mean={s.mean():+6.2f}%  sum={s.sum():+7.1f}%")

    # ── bp_low 桶 (5d hold) ──
    print(f"\n按 bp_low 桶 (5d hold) — bp_low 越低=越深破下沿:")
    print("-" * 60)
    for lo, hi, lab in [(0, 0.1, "极深破 [0,0.1]"),
                            (0.1, 0.2, "深破 [0.1,0.2]"),
                            (0.2, 0.3, "中破 [0.2,0.3]"),
                            (0.3, 1.0, "浅破 [0.3,1.0]")]:
        sub = df[(df["bp_low"] >= lo) & (df["bp_low"] < hi)]
        s = sub["r5d"].dropna()
        if not len(s): print(f"  {lab:18}: n=0"); continue
        wr = (s>0).mean()*100
        print(f"  {lab:18}: n={len(s):3d}  WR={wr:5.1f}%  mean={s.mean():+6.2f}%  sum={s.sum():+7.1f}%  max_loss={s.min():+.1f}%")

    # ── regime 桶 ──
    print(f"\n按 regime (5d hold):")
    print("-" * 60)
    for rg, sub in df.groupby("regime"):
        s = sub["r5d"].dropna()
        if not len(s): continue
        wr = (s>0).mean()*100
        print(f"  {rg:6}: n={len(s):3d}  WR={wr:5.1f}%  mean={s.mean():+6.2f}%  sum={s.sum():+7.1f}%")

    # ── GVZ 桶 ──
    print(f"\n按 GVZ 桶 (5d hold):")
    print("-" * 60)
    if "GVZ" in df.columns:
        for lo, hi, lab in [(0, 22, "低 IV [<22]"),
                                (22, 25, "中低 [22,25]"),
                                (25, 28, "中高 [25,28]"),
                                (28, 200, "高 IV [>=28]")]:
            sub = df[(df["GVZ"] >= lo) & (df["GVZ"] < hi)]
            s = sub["r5d"].dropna()
            if not len(s): print(f"  {lab:14}: n=0"); continue
            wr = (s>0).mean()*100
            print(f"  {lab:14}: n={len(s):3d}  WR={wr:5.1f}%  mean={s.mean():+6.2f}%  sum={s.sum():+7.1f}%")

    # ── rv_pct 桶 ──
    print(f"\n按 rv_pctile 桶 (5d hold):")
    print("-" * 60)
    for lo, hi, lab in [(0, 0.25, "低 RV [<0.25]"),
                            (0.25, 0.5, "[0.25,0.5]"),
                            (0.5, 0.75, "[0.5,0.75]"),
                            (0.75, 1.01, "高 RV [>=0.75]")]:
        sub = df[(df["rv_pct"] >= lo) & (df["rv_pct"] < hi)]
        s = sub["r5d"].dropna()
        if not len(s): print(f"  {lab:14}: n=0"); continue
        wr = (s>0).mean()*100
        print(f"  {lab:14}: n={len(s):3d}  WR={wr:5.1f}%  mean={s.mean():+6.2f}%  sum={s.sum():+7.1f}%")

    # ── ma_trend 桶 ──
    print(f"\n按 ma_trend (5d hold) — ma_trend < 1 = 短均线在长均线下:")
    print("-" * 60)
    for lo, hi, lab in [(0, 0.97, "[<0.97]"),
                            (0.97, 0.99, "[0.97,0.99]"),
                            (0.99, 1.0, "[0.99,1.00]"),
                            (1.0, 1.1, "[>=1.00]")]:
        sub = df[(df["ma_trend"] >= lo) & (df["ma_trend"] < hi)]
        s = sub["r5d"].dropna()
        if not len(s): print(f"  {lab:14}: n=0"); continue
        wr = (s>0).mean()*100
        print(f"  {lab:14}: n={len(s):3d}  WR={wr:5.1f}%  mean={s.mean():+6.2f}%  sum={s.sum():+7.1f}%")

    # ── 按季度 ──
    print(f"\n按季度 (5d hold):")
    print("-" * 60)
    df["sig_d_ts"] = pd.to_datetime(df["sig_d"])
    df["q"] = df["sig_d_ts"].dt.to_period("Q")
    for q, sub in df.groupby("q"):
        s = sub["r5d"].dropna()
        if not len(s): continue
        wr = (s>0).mean()*100
        print(f"  {q}: n={len(s):3d}  WR={wr:5.1f}%  mean={s.mean():+6.2f}%  sum={s.sum():+7.1f}%")


if __name__ == "__main__":
    main()
