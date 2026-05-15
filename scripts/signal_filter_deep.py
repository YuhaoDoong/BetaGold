"""v3.7.200 信号 filter 深度 grid

目标: 砍少, 但 WR/sum 显著升, Q1 过滤 100%.

3y GLD BUY 信号 (143 笔 baseline), 测:
  - 单因子分位 (rv_pct 各 cutoff, ret_20d 各 cutoff, bp_low 各 cutoff)
  - 二因子组合
  - 三因子组合
  - 看每个 filter 的: n, WR(5/10/20d), sum, mean, max_loss, Q1 命中率, 按季度分布

判定标准:
  - 主指标: score_B = WR² × log(1+n) × mean (CLAUDE.md 决策规则)
  - 副指标: Q1 拦截率, 季度分布是否均衡
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf, numpy as np

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier


WINDOW_YEARS = 3


def build():
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

    sig = generate_daily_signals(close, high, low, upper, lower, regime, rv_p,
                                       asset="GLD", gvz_series=gvz["Close"])
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=WINDOW_YEARS*365)
    buy = sig[sig["buy_signal"] & (sig.index >= cutoff)].copy()
    ret20 = close.pct_change(20)
    buy["ret_20d"] = ret20.reindex(buy.index) * 100
    buy["GVZ"] = gvz["Close"].reindex(buy.index)
    # forward returns
    for h in (5, 10, 20):
        for sig_d in buy.index:
            if sig_d not in ohlc.index: continue
            idx = ohlc.index.get_loc(sig_d)
            entry = float(ohlc.iloc[idx]["Open"])
            if idx + h < len(ohlc):
                exit_p = float(ohlc.iloc[idx+h]["Close"])
                buy.loc[sig_d, f"r{h}d"] = (exit_p / entry - 1) * 100
            else:
                buy.loc[sig_d, f"r{h}d"] = np.nan
    return buy, cutoff


def score_filter(buy, mask, label):
    sub = buy[mask]
    n = len(sub)
    if n == 0: return {"filter": label, "n": 0}
    s5 = sub["r5d"].dropna(); s10 = sub["r10d"].dropna(); s20 = sub["r20d"].dropna()
    wr5 = (s5 > 0).mean() * 100 if len(s5) else None
    wr10 = (s10 > 0).mean() * 100 if len(s10) else None
    wr20 = (s20 > 0).mean() * 100 if len(s20) else None
    # score_B = WR² × log(1+n) × mean  (CLAUDE.md)
    score_B5 = (wr5/100)**2 * math.log(1+n) * s5.mean() if len(s5) and wr5 else 0
    # Q1
    q1 = mask & ((buy.index >= "2026-01-01") & (buy.index <= "2026-03-31"))
    q1_kept = q1.sum()
    # 5/12-14
    may = mask & buy.index.isin([pd.Timestamp(d) for d in
                                       ("2026-05-12","2026-05-13","2026-05-14")])
    may_kept = may.sum()
    return {
        "filter": label, "n": n,
        "WR5d": round(wr5, 1) if wr5 else None,
        "mean5d": round(s5.mean(), 2) if len(s5) else None,
        "sum5d": round(s5.sum(), 1) if len(s5) else None,
        "max_loss5d": round(s5.min(), 1) if len(s5) else None,
        "WR10d": round(wr10, 1) if wr10 else None,
        "sum10d": round(s10.sum(), 1) if len(s10) else None,
        "WR20d": round(wr20, 1) if wr20 else None,
        "sum20d": round(s20.sum(), 1) if len(s20) else None,
        "scoreB5d": round(score_B5, 2),
        "Q1_kept": int(q1_kept),
        "may_kept": int(may_kept),
    }


def main():
    buy, cutoff = build()
    print(f"窗口: {cutoff.date()} → 今")
    print(f"BASELINE n={len(buy)}\n")

    rows = [score_filter(buy, pd.Series(True, buy.index), "BASELINE (现 prod cfg)")]

    # 单因子精细
    for c in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        rows.append(score_filter(buy, buy["rv_pctile"] < c, f"rv<{c}"))
    print("\n--- rv_pctile 单因子 (越严信号越少, WR 越高) ---")
    for r in rows[1:8]:
        print(f"  {r['filter']:12} n={r['n']:3d} WR5d={r['WR5d']}% sum={r['sum5d']}%  "
              f"WR20d={r['WR20d']}% sum20={r['sum20d']}%  Q1={r['Q1_kept']}/20  scoreB={r['scoreB5d']}")

    rows2 = []
    for c in [-5, -3, -1, 0, +1, +3]:
        rows2.append(score_filter(buy, buy["ret_20d"] > c, f"ret_20d>{c}%"))
    print("\n--- ret_20d 单因子 ---")
    for r in rows2:
        print(f"  {r['filter']:12} n={r['n']:3d} WR5d={r['WR5d']}% sum={r['sum5d']}%  "
              f"WR20d={r['WR20d']}% sum20={r['sum20d']}%  Q1={r['Q1_kept']}/20  scoreB={r['scoreB5d']}")
    rows += rows2

    rows3 = []
    for c in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        rows3.append(score_filter(buy, buy["bp_low"] <= c, f"bp_low<={c}"))
    print("\n--- bp_low 单因子 ---")
    for r in rows3:
        print(f"  {r['filter']:14} n={r['n']:3d} WR5d={r['WR5d']}% sum={r['sum5d']}%  "
              f"WR20d={r['WR20d']}% sum20={r['sum20d']}%  Q1={r['Q1_kept']}/20  scoreB={r['scoreB5d']}")
    rows += rows3

    # 双因子组合 (关注高 score_B 的)
    print("\n--- 双因子组合 ---")
    combos2 = [
        ("rv<0.75 AND ret>-3", (buy["rv_pctile"]<0.75) & (buy["ret_20d"]>-3)),
        ("rv<0.70 AND ret>-3", (buy["rv_pctile"]<0.70) & (buy["ret_20d"]>-3)),
        ("rv<0.65 AND ret>-3", (buy["rv_pctile"]<0.65) & (buy["ret_20d"]>-3)),
        ("rv<0.60 AND ret>-3", (buy["rv_pctile"]<0.60) & (buy["ret_20d"]>-3)),
        ("rv<0.75 AND ret>-1", (buy["rv_pctile"]<0.75) & (buy["ret_20d"]>-1)),
        ("rv<0.70 AND ret>-1", (buy["rv_pctile"]<0.70) & (buy["ret_20d"]>-1)),
        ("rv<0.65 AND ret>0",  (buy["rv_pctile"]<0.65) & (buy["ret_20d"]>0)),
        ("rv<0.75 AND bp_low<=0.20", (buy["rv_pctile"]<0.75) & (buy["bp_low"]<=0.20)),
        ("rv<0.70 AND bp_low<=0.20", (buy["rv_pctile"]<0.70) & (buy["bp_low"]<=0.20)),
        ("ret>-3 AND bp_low<=0.20", (buy["ret_20d"]>-3) & (buy["bp_low"]<=0.20)),
        ("ret>0 AND bp_low<=0.20", (buy["ret_20d"]>0) & (buy["bp_low"]<=0.20)),
    ]
    for lab, m in combos2:
        r = score_filter(buy, m, lab)
        print(f"  {lab:30} n={r['n']:3d} WR5d={r['WR5d']}% sum={r['sum5d']}%  "
              f"WR20d={r['WR20d']}% sum20={r['sum20d']}%  Q1={r['Q1_kept']}/20 may={r['may_kept']}/3  scoreB={r['scoreB5d']}")
        rows.append(r)

    # 三因子组合
    print("\n--- 三因子组合 ---")
    combos3 = [
        ("rv<0.75 AND ret>-3 AND bp<=0.20", (buy["rv_pctile"]<0.75)&(buy["ret_20d"]>-3)&(buy["bp_low"]<=0.20)),
        ("rv<0.70 AND ret>-3 AND bp<=0.20", (buy["rv_pctile"]<0.70)&(buy["ret_20d"]>-3)&(buy["bp_low"]<=0.20)),
        ("rv<0.75 AND ret>-1 AND bp<=0.20", (buy["rv_pctile"]<0.75)&(buy["ret_20d"]>-1)&(buy["bp_low"]<=0.20)),
        ("rv<0.75 AND ret>0 AND bp<=0.20",  (buy["rv_pctile"]<0.75)&(buy["ret_20d"]>0)&(buy["bp_low"]<=0.20)),
    ]
    for lab, m in combos3:
        r = score_filter(buy, m, lab)
        print(f"  {lab:38} n={r['n']:3d} WR5d={r['WR5d']}% sum={r['sum5d']}%  "
              f"WR20d={r['WR20d']}% sum20={r['sum20d']}%  Q1={r['Q1_kept']}/20 may={r['may_kept']}/3  scoreB={r['scoreB5d']}")
        rows.append(r)

    # 按季度分布 — 看 top 候选 filter 是否信号均衡
    print("\n--- top 候选 filter 季度分布 ---")
    top_filters = [
        ("BASELINE", pd.Series(True, buy.index)),
        ("rv<0.75 AND ret>-3", (buy["rv_pctile"]<0.75) & (buy["ret_20d"]>-3)),
        ("rv<0.65 AND ret>-3", (buy["rv_pctile"]<0.65) & (buy["ret_20d"]>-3)),
        ("rv<0.5", buy["rv_pctile"]<0.5),
        ("rv<0.75 AND ret>0", (buy["rv_pctile"]<0.75) & (buy["ret_20d"]>0)),
    ]
    buy["q"] = pd.to_datetime(buy.index).to_period("Q")
    for lab, m in top_filters:
        sub = buy[m]
        if not len(sub): continue
        qd = sub.groupby("q").size().to_dict()
        kept_total = sum(qd.values())
        qs = ", ".join(f"{q}:{n}" for q,n in sorted(qd.items()))
        print(f"  {lab:28} 总 {kept_total:3d} 笔: {qs}")

    out = "/Users/yhdong/Gold/data/backtest_history/signal_filter_deep.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
