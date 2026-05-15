"""v3.7.203 期货 5y grid (GC=F yfinance 替代 Binance 5个月)

GLD BUY 信号 (v3.7.201 + tier) × GC=F daily 持仓 模拟 FUTURES_LONG.

Grid:
  leverage: [3, 5, 10, 20]
  tp_margin: [100, 150, 200, 300, 500]  (%)
  sl_margin: [50, 75, 100, 150]  (%)
  hold_max: [10, 15, 20, 30]

输出 per-combo:
  n, WR, sum, mean, max_loss, 爆仓数, 早平命中
  分 tier (S/A/B) 看

按 scoreB = WR² × log(1+n) × mean 排序找最优.
"""
from __future__ import annotations
import sys, math, copy, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf
import numpy as np

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier
from core.strategies.futures_long import FuturesConfig, simulate_long_position


LOOKBACK_YEARS = 5


def build_signals():
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
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=LOOKBACK_YEARS*365)
    buy = sig[sig["buy_signal"] & (sig.index >= cutoff)].copy()
    return buy


def load_gc_data():
    """yfinance GC=F daily 5y."""
    gc = yf.Ticker("GC=F").history(period=f"{LOOKBACK_YEARS+1}y", auto_adjust=True)
    gc.index = pd.to_datetime(gc.index).tz_localize(None).normalize()
    return gc[["Open","High","Low","Close"]]


def run_combo(buy: pd.DataFrame, gc: pd.DataFrame,
                lev: int, tp_m: float, sl_m: float, hold_max: int):
    """Run one cfg combo, return summary."""
    cfg = FuturesConfig(leverage=lev, tp_margin_pct=tp_m, sl_margin_pct=sl_m,
                              hold_max_days=hold_max)
    results = []
    today = gc.index.max()
    for sig_d in buy.index:
        # 找 entry 当日 GC=F price (or 最近一日)
        gc_match = gc.loc[gc.index <= sig_d]
        if not len(gc_match): continue
        # 取 sig_d 之后的 OHLC (entry 当日 + 之后)
        gc_after = gc.loc[gc.index >= sig_d]
        if not len(gc_after): continue
        entry_spot = float(gc_after.iloc[0]["Open"])
        if entry_spot <= 0: continue
        res = simulate_long_position(
            entry_d=gc_after.index[0],
            entry_spot=entry_spot,
            ohlc=gc_after,
            today=today,
            cfg=cfg,
            live_spot=float(gc.iloc[-1]["Close"]))
        if not res.get("closed"): continue
        results.append({
            "sig_d": sig_d.date(),
            "tier": buy.loc[sig_d, "signal_tier"],
            "ret_lev_pct": max(-100, min(500, res.get("ret_levered_pct", 0))),
            "reason": res.get("reason", ""),
            "hold": res.get("hold_days", 0),
            "is_liq": res.get("is_liquidation", False),
        })
    return results


def summarize(results, label):
    if not results: return {"label": label, "n": 0}
    pnls = pd.Series([r["ret_lev_pct"] for r in results])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() else float("inf")
    wr = (pnls > 0).mean() * 100
    scoreB = (wr/100)**2 * math.log(1+len(pnls)) * pnls.mean()
    n_liq = sum(1 for r in results if r.get("is_liq"))
    return {
        "label": label, "n": len(pnls),
        "WR%": round(wr, 1),
        "sum%": round(pnls.sum(), 0),
        "mean%": round(pnls.mean(), 1),
        "max_loss%": round(pnls.min(), 1),
        "max_win%": round(pnls.max(), 1),
        "n_liq": n_liq,
        "PF": round(pf, 2),
        "scoreB": round(scoreB, 2),
    }


def main():
    print(f"Loading signals & GC=F (5y)...")
    buy = build_signals()
    gc = load_gc_data()
    print(f"GLD BUY 信号 5y: {len(buy)} 笔 (tier 分布: {buy['signal_tier'].value_counts().to_dict()})")
    print(f"GC=F: {gc.index.min().date()} → {gc.index.max().date()} ({len(gc)} bars)\n")

    # 1. 当前 baseline: lev=5, tp=200, sl=100, hold=15
    print("=" * 80)
    print("Step 1: 当前 cfg baseline (lev=5, tp=200%, sl=100%, hold=15)")
    print("=" * 80)
    base = run_combo(buy, gc, 5, 200, 100, 15)
    print(summarize(base, "BASELINE 全信号"))
    for t in ["S","A","B"]:
        sub = [r for r in base if r["tier"] == t]
        print(summarize(sub, f"BASELINE tier={t}"))

    # 2. leverage grid
    print("\n" + "=" * 80)
    print("Step 2: leverage grid (tp=200% sl=100% hold=15 固定)")
    print("=" * 80)
    rows = []
    for lev in [3, 5, 10, 15, 20]:
        r = summarize(run_combo(buy, gc, lev, 200, 100, 15), f"lev={lev}")
        print(f"  lev={lev:2d}: n={r['n']:3d} WR={r['WR%']}% sum={r['sum%']}% mean={r['mean%']}% "
              f"max_loss={r['max_loss%']}% liq={r['n_liq']}/{r['n']} PF={r['PF']} scoreB={r['scoreB']}")
        rows.append(r)

    # 3. TP grid (lev=5 固定)
    print("\n" + "=" * 80)
    print("Step 3: TP grid (lev=5 sl=100% hold=15 固定)")
    print("=" * 80)
    for tp in [50, 100, 150, 200, 300, 500, 1000]:
        r = summarize(run_combo(buy, gc, 5, tp, 100, 15), f"tp={tp}")
        print(f"  tp={tp:4d}%: n={r['n']:3d} WR={r['WR%']}% sum={r['sum%']}% mean={r['mean%']}% "
              f"max_loss={r['max_loss%']}% scoreB={r['scoreB']}")
        rows.append(r)

    # 4. SL grid (lev=5 固定)
    print("\n" + "=" * 80)
    print("Step 4: SL grid (lev=5 tp=200% hold=15 固定)")
    print("=" * 80)
    for sl in [30, 50, 75, 100, 150, 200, 300]:
        r = summarize(run_combo(buy, gc, 5, 200, sl, 15), f"sl={sl}")
        print(f"  sl={sl:4d}%: n={r['n']:3d} WR={r['WR%']}% sum={r['sum%']}% mean={r['mean%']}% "
              f"max_loss={r['max_loss%']}% scoreB={r['scoreB']}")
        rows.append(r)

    # 5. hold_max grid
    print("\n" + "=" * 80)
    print("Step 5: hold_max grid (lev=5 tp=200% sl=100% 固定)")
    print("=" * 80)
    for hm in [5, 10, 15, 20, 30, 45]:
        r = summarize(run_combo(buy, gc, 5, 200, 100, hm), f"hold={hm}")
        print(f"  hold={hm:2d}d: n={r['n']:3d} WR={r['WR%']}% sum={r['sum%']}% mean={r['mean%']}% "
              f"max_loss={r['max_loss%']}% scoreB={r['scoreB']}")
        rows.append(r)

    # 6. 全网格 (top 候选)
    print("\n" + "=" * 80)
    print("Step 6: 全网格 (4 × 4 × 3 × 3 = 144 combos, 找 top scoreB)")
    print("=" * 80)
    all_rows = []
    for lev, tp, sl, hm in itertools.product([3, 5, 10], [100, 150, 200, 300],
                                                       [50, 100, 150], [10, 15, 20, 30]):
        r = summarize(run_combo(buy, gc, lev, tp, sl, hm),
                        f"lev={lev} tp={tp} sl={sl} hm={hm}")
        all_rows.append(r)
    df = pd.DataFrame(all_rows)
    df = df.sort_values("scoreB", ascending=False).head(20)
    print(df[["label","n","WR%","sum%","mean%","max_loss%","n_liq","PF","scoreB"]].to_string(index=False))

    # 7. 按 tier 看 top combo 表现
    print("\n" + "=" * 80)
    print("Step 7: top 3 combo 按 tier 拆分")
    print("=" * 80)
    top3 = df.head(3)["label"].tolist()
    for lab in top3:
        parts = dict(p.split("=") for p in lab.split(" "))
        lev, tp, sl, hm = int(parts["lev"]), float(parts["tp"]), float(parts["sl"]), int(parts["hm"])
        res = run_combo(buy, gc, lev, tp, sl, hm)
        print(f"\n  Combo: {lab}")
        for t in ["S","A","B"]:
            sub = [r for r in res if r["tier"] == t]
            s = summarize(sub, f"tier {t}")
            print(f"    tier {t}: n={s.get('n',0)} WR={s.get('WR%','?')}% sum={s.get('sum%','?')}% "
                  f"mean={s.get('mean%','?')}% scoreB={s.get('scoreB','?')}")

    out = "/Users/yhdong/Gold/data/backtest_history/futures_grid_5y.csv"
    pd.DataFrame(rows + all_rows).to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
