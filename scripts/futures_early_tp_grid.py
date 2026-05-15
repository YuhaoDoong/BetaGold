"""v3.7.208 期货 early_tp_lock 早平阈值 grid

之前 5y grid 显示 tp_margin/sl_margin 是死参数 (early_tp_lock 主导退出).
本脚本测 early_tp_locks tuple 的实际 grid.

变种 (3 elements tuple):
  现 cfg: (3, 5), (7, 3), (12, 1)
  激进早平: (2, 3), (5, 2), (10, 0.5)
  中度激进: (3, 4), (7, 2.5), (12, 1)
  保守锁利: (5, 7), (10, 5), (15, 3)
  极保守:   (8, 8), (15, 5), (25, 2)
  关闭早平: ()
"""
from __future__ import annotations
import sys, copy, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier
from core.strategies.futures_long import FuturesConfig, simulate_long_position


def build_buys():
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
    return sig[sig["buy_signal"]].copy()


def run(buy, gc, cfg, today):
    rows = []
    for sig_d in buy.index:
        gc_after = gc.loc[gc.index >= sig_d]
        if not len(gc_after): continue
        es = float(gc_after.iloc[0]["Open"])
        tier = buy.loc[sig_d, "signal_tier"] or "B"
        res = simulate_long_position(entry_d=gc_after.index[0], entry_spot=es,
                                            ohlc=gc_after, today=today, cfg=cfg,
                                            live_spot=float(gc.iloc[-1]["Close"]),
                                            signal_tier=tier)
        if not res.get("closed"): continue
        rows.append({"sig_d": sig_d.date(), "tier": tier,
                       "ret": max(-100, min(500, res.get("net_pnl_pct", 0))),
                       "hold": res.get("hold_days", 0),
                       "reason": res.get("reason", ""),
                       "is_liq": res.get("is_liquidation", False)})
    return rows


def summarize(rows, label):
    if not rows: return f"{label}: n=0"
    p = pd.Series([r["ret"] for r in rows])
    h = pd.Series([r["hold"] for r in rows])
    sc = (p.gt(0).mean())**2 * math.log(1+len(p)) * p.mean()
    n_liq = sum(1 for r in rows if r.get("is_liq"))
    return ("%s: n=%3d WR=%5.1f%% sum=%+7.1f%% mean=%+6.2f%% "
            "hold_avg=%4.1fd max_loss=%+6.1f%% liq=%d sB=%.1f"
            % (label, len(p), (p>0).mean()*100, p.sum(), p.mean(),
               h.mean(), p.min(), n_liq, sc))


def main():
    buy = build_buys()
    gc = yf.Ticker("GC=F").history(period="6y", auto_adjust=True)
    gc.index = pd.to_datetime(gc.index).tz_localize(None).normalize()
    gc = gc[["Open","High","Low","Close"]]
    today = gc.index.max()
    buy = buy[buy.index >= today - pd.Timedelta(days=5*365)]
    print(f"5y BUY 信号: {len(buy)} 笔, tier {buy['signal_tier'].value_counts().to_dict()}\n")

    base_cfg = dict(leverage=5, tier_s_leverage=10, tier_a_leverage=10,
                       tier_b_leverage=5, hold_max_days=30,
                       funding_rate_8h=-0.00002, tp_margin_pct=200,
                       sl_margin_pct=100)

    variants = [
        ("当前 (3,5)(7,3)(12,1)", ((3,5.0),(7,3.0),(12,1.0))),
        ("激进 (2,3)(5,2)(10,0.5)", ((2,3.0),(5,2.0),(10,0.5))),
        ("中激进 (3,4)(7,2.5)(12,1)", ((3,4.0),(7,2.5),(12,1.0))),
        ("中保守 (5,5)(10,3)(15,1)", ((5,5.0),(10,3.0),(15,1.0))),
        ("保守 (5,7)(10,5)(15,3)", ((5,7.0),(10,5.0),(15,3.0))),
        ("极保守 (8,8)(15,5)(25,2)", ((8,8.0),(15,5.0),(25,2.0))),
        ("无早平", ()),
    ]
    print("=" * 100)
    rows_all = []
    for lab, et in variants:
        c = FuturesConfig(**base_cfg, early_tp_locks=et)
        rows = run(buy, gc, c, today)
        rows_all.append((lab, rows))
        print(summarize(rows, lab.ljust(28)))

    # 按 tier 看 top 2-3 候选
    print("\n--- top 3 候选 per-tier 拆 ---")
    # 按 scoreB 排序
    sorted_v = sorted(rows_all, key=lambda x: -sum(r["ret"] for r in x[1])/max(len(x[1]),1) * (sum(1 for r in x[1] if r["ret"]>0)/max(len(x[1]),1))**2 if x[1] else 0)
    for lab, rows in sorted_v[:3]:
        print(f"\n{lab}:")
        for t in ['S','A','B']:
            sub = [r for r in rows if r['tier']==t]
            print("  " + summarize(sub, f"tier {t}"))


if __name__ == "__main__":
    main()
