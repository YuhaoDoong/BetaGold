"""v3.7.205 期权止损 5y grid (BS proxy + real kline_db)

Grid:
  BC: TP (1.2, 1.5, 1.8, 2.0, 2.5) × SL (0.3, 0.4, 0.5, 0.6, 0.7)
  SP: TP (30, 50, 70) × SL (50, 75, 100, 150)

每 combo 拆 5y / per-tier (S/A/B 互斥 + nested).
找 score_B (WR² × log(1+n) × mean) 最高的 combo.
"""
from __future__ import annotations
import sys, math, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf
import numpy as np

# import 5y BS proxy 函数
from scripts.options_5y_bs_proxy import run_unified


def main():
    # ── inline 重做 build ──
    from core.data import load_oos_predictions, load_config
    from core.signals_v2 import generate_daily_signals
    from core.signals import build_band, compute_rv_pctile
    from core.regime import RegimeClassifier
    from core.paper_positions import _load_kline_db
    from core.strategies.buy_call import BCConfig
    from core.strategies.sell_put import SPConfig

    cfg = load_config()
    oos = load_oos_predictions(cfg)
    feat = pd.read_parquet("/Users/yhdong/Gold/data/processed/features_all.parquet")
    ohlc = pd.read_csv("/Users/yhdong/Gold/data/raw/market/gld.csv",
                          index_col=0, parse_dates=True)
    common = ohlc.index.intersection(feat.index).intersection(oos.index)
    close = ohlc.loc[common,"Close"]; high=ohlc.loc[common,"High"]; low=ohlc.loc[common,"Low"]
    upper, lower, _ = build_band(oos.loc[common], close)
    rv_p = compute_rv_pctile(feat.loc[common,"rv_10d"])
    feat_cols = [c for c in feat.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(feat.loc[common, feat_cols])["regime"]
    gvz_raw = yf.Ticker("^GVZ").history(period="5y")
    gvz_raw.index = pd.to_datetime(gvz_raw.index).tz_localize(None).normalize()
    gvz_s = gvz_raw["Close"]
    sig = generate_daily_signals(close, high, low, upper, lower, regime, rv_p,
                                       asset="GLD", gvz_series=gvz_s)
    buy = sig[sig["buy_signal"]].copy().join(
        ohlc[["Open","High","Low","Close"]], how="inner")
    today = ohlc.index.max()
    cutoff = today - pd.Timedelta(days=5*365)
    buy = buy[buy.index >= cutoff]
    db = _load_kline_db()
    db_min = db["date"].min() if db is not None else today

    print(f"5y BUY 信号: {len(buy)} 笔, tier {buy['signal_tier'].value_counts().to_dict()}")

    def score_combo(rows, n_signals):
        if not rows: return 0
        p = pd.Series([r["pnl_pct"] for r in rows])
        wr = (p>0).mean()
        return (wr**2) * math.log(1+len(p)) * p.mean() if len(p) else 0

    # ── BC grid ──
    print("\n" + "=" * 90)
    print("BC TP/SL grid (5y) — score_B 排序 top 15")
    print("=" * 90)
    bc_rows = []
    for pt in [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
        for sl in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            c = BCConfig(profit_target_mult=pt, stop_loss_mult=sl, base_dte=30)
            r = run_unified(buy, ohlc, db, db_min, today, gvz_s,
                              strategy="BC", bc_cfg=c)
            if not r: continue
            p = pd.Series([x["pnl_pct"] for x in r])
            n_w = (p>0).sum(); n_l = (p<=0).sum()
            sb = score_combo(r, len(buy))
            # nested S/A
            s_only = [x for x in r if x['tier']=='S']
            a_inc = [x for x in r if x['tier'] in ('S','A')]
            s_p = pd.Series([x['pnl_pct'] for x in s_only]) if s_only else pd.Series([],dtype=float)
            a_p = pd.Series([x['pnl_pct'] for x in a_inc]) if a_inc else pd.Series([],dtype=float)
            bc_rows.append({
                "pt": pt, "sl": sl, "n": len(p),
                "WR": round(p.gt(0).mean()*100, 1),
                "sum": round(p.sum(), 0),
                "mean": round(p.mean(), 2),
                "max_loss": round(p.min(), 1),
                "scoreB": round(sb, 2),
                "S_WR": round(s_p.gt(0).mean()*100, 1) if len(s_p) else None,
                "S_mean": round(s_p.mean(), 1) if len(s_p) else None,
                "A_WR": round(a_p.gt(0).mean()*100, 1) if len(a_p) else None,
                "A_mean": round(a_p.mean(), 1) if len(a_p) else None,
            })
    bc_df = pd.DataFrame(bc_rows).sort_values("scoreB", ascending=False)
    print(bc_df.head(15).to_string(index=False))
    print(f"\n  当前 cfg (pt=1.5/sl=0.5):")
    print(bc_df[(bc_df['pt']==1.5)&(bc_df['sl']==0.5)].to_string(index=False))

    # ── SP grid ──
    print("\n" + "=" * 90)
    print("SP TP/SL grid (5y) — score_B 排序 top 15")
    print("=" * 90)
    sp_rows = []
    for pt in [20, 30, 40, 50, 60, 70]:
        for sl in [50, 75, 100, 150, 200]:
            c = SPConfig(profit_target_credit_pct=pt,
                                stop_loss_margin_pct=sl, base_dte=30)
            r = run_unified(buy, ohlc, db, db_min, today, gvz_s,
                              strategy="SP", sp_cfg=c)
            if not r: continue
            p = pd.Series([x["pnl_pct"] for x in r])
            sb = score_combo(r, len(buy))
            s_only = [x for x in r if x['tier']=='S']
            a_inc = [x for x in r if x['tier'] in ('S','A')]
            s_p = pd.Series([x['pnl_pct'] for x in s_only]) if s_only else pd.Series([],dtype=float)
            a_p = pd.Series([x['pnl_pct'] for x in a_inc]) if a_inc else pd.Series([],dtype=float)
            sp_rows.append({
                "pt": pt, "sl": sl, "n": len(p),
                "WR": round(p.gt(0).mean()*100, 1),
                "sum": round(p.sum(), 0),
                "mean": round(p.mean(), 2),
                "max_loss": round(p.min(), 1),
                "scoreB": round(sb, 2),
                "S_WR": round(s_p.gt(0).mean()*100, 1) if len(s_p) else None,
                "S_max_loss": round(s_p.min(), 1) if len(s_p) else None,
                "A_WR": round(a_p.gt(0).mean()*100, 1) if len(a_p) else None,
                "A_max_loss": round(a_p.min(), 1) if len(a_p) else None,
            })
    sp_df = pd.DataFrame(sp_rows).sort_values("scoreB", ascending=False)
    print(sp_df.head(15).to_string(index=False))
    print(f"\n  当前 cfg (pt=50/sl=100):")
    print(sp_df[(sp_df['pt']==50)&(sp_df['sl']==100)].to_string(index=False))

    bc_df.to_csv("/Users/yhdong/Gold/data/backtest_history/bc_sl_grid_5y.csv", index=False)
    sp_df.to_csv("/Users/yhdong/Gold/data/backtest_history/sp_sl_grid_5y.csv", index=False)


if __name__ == "__main__":
    main()
