"""v3.7.168 退出参数 grid v2 — bug 修复后重新验证最优解.

跑改动后的模块化 sim:
  期货: simulate_long_position (futures_long) — per-asset lev (GLD 20× / SLV 10×)
  SHORT_VOL: simulate_short_vol_position (short_vol) — 修后的 max_risk
  SP/BC: 跳过 (paired_grid_multi.py 已专门处理)

数据窗口: 近 1 年 (Binance kline + ETF csv + kline_db)
输出: 每 grid 组合 cum_pnl/wr/avg/sharpe + 标记现行参数.

用法:
  python scripts/exit_grid_v2.py [futures|shortvol|all]
"""
from __future__ import annotations
import sys, os, argparse
from pathlib import Path
from itertools import product
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from core.signals_v2 import generate_daily_signals
from core.events import detect_short_vol_signal
from core.regime import RegimeClassifier
from core.binance_futures import (fetch_perp_klines, fetch_perp_price_at_date,
                                       ASSET_SYMBOL)
from core.strategies.futures_long import (simulate_long_position, FuturesConfig)
from core.strategies.short_vol import (simulate_short_vol_position, ShortVolConfig)
from core.paper_positions import price_strategy_at, _load_kline_db


CSV_DIR = Path("/Users/yhdong/Gold/data/backtest_history")


# ───────────────────────── 期货 GRID ─────────────────────────
def load_futures_signals(asset: str, days_back: int):
    """从 backtest CSV 读已生成信号 (避免重跑信号生成)."""
    asset_lc = asset.lower()
    ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset_lc}.csv",
                        index_col=0, parse_dates=True)
    today = pd.Timestamp.now().normalize()
    cutoff = today - pd.Timedelta(days=days_back)
    # 从最新 backtest CSV 读
    csvs = sorted(CSV_DIR.glob(f"backtest_{asset_lc}_*.csv"))
    if not csvs:
        return [], ohlc
    df = pd.read_csv(csvs[-1], parse_dates=["signal_date"])
    fut = df[(df["strategy"] == "FUTURES_LONG") &
              (df["signal_date"] >= cutoff)]
    sigs = sorted(fut["signal_date"].drop_duplicates().to_list())
    return sigs, ohlc


def fetch_binance_window(symbol: str, days_back: int) -> pd.DataFrame:
    """拉一年 Binance daily kline → OHLC DataFrame."""
    today_ms = int(pd.Timestamp.now().normalize().timestamp() * 1000)
    start_ms = int((pd.Timestamp.now().normalize() -
                    pd.Timedelta(days=days_back + 30)).timestamp() * 1000)
    klines = []
    cur = start_ms
    while cur < today_ms:
        batch = fetch_perp_klines(symbol, cur, today_ms, "1d")
        if not batch: break
        klines.extend(batch)
        last_ts = int(batch[-1][0])
        if last_ts == cur or len(batch) < 1000: break
        cur = last_ts + 86400000
    rows = [{"Date": pd.Timestamp(int(k[0]), unit="ms").normalize(),
             "Open": float(k[1]), "High": float(k[2]),
             "Low": float(k[3]), "Close": float(k[4])} for k in klines]
    df = pd.DataFrame(rows).drop_duplicates("Date").set_index("Date").sort_index()
    return df


def grid_futures(asset: str, days_back: int = 365):
    sym = ASSET_SYMBOL[asset]
    print(f"\n{'='*78}")
    print(f"【期货 grid: {asset} ({sym})】 近 {days_back} 天")
    print(f"{'='*78}")
    sigs, _ = load_futures_signals(asset, days_back)
    if not sigs:
        print(f"  {asset} 无信号"); return
    df_perp = fetch_binance_window(sym, days_back)
    if df_perp.empty:
        print(f"  Binance 数据空"); return
    today = pd.Timestamp.now().normalize()

    # Grid: leverage × TP_margin × SL_margin × hold_max
    levs = [10, 15, 20] if asset == "GLD" else [5, 10, 15]
    tps = [100, 150, 200, 250]   # margin %
    sls = [30, 50, 75, 100]      # margin %
    holds = [10, 15, 20]

    results = []
    for lev, tp, sl, hold_max in product(levs, tps, sls, holds):
        cfg = FuturesConfig(leverage=lev, tp_margin_pct=tp,
                              sl_margin_pct=sl, hold_max_days=hold_max)
        pnls = []
        for d in sigs:
            d = pd.Timestamp(d).normalize()
            if d not in df_perp.index: continue
            entry = float(df_perp.loc[d, "Open"])
            if entry <= 0: continue
            res = simulate_long_position(d, entry, df_perp, today, cfg)
            if res.get("closed"):
                p = max(-100.0, float(res.get("ret_levered_pct", 0) or 0))
                pnls.append(p)
        if not pnls: continue
        s = pd.Series(pnls)
        results.append({
            "lev": lev, "tp": tp, "sl": sl, "hold": hold_max,
            "n": len(s), "wr": (s > 0).mean() * 100,
            "avg": s.mean(), "sum": s.sum(), "med": s.median(),
            "sharpe": s.mean() / s.std() if s.std() > 0 else 0,
        })

    rep = pd.DataFrame(results).sort_values("sum", ascending=False)
    cur_lev = 20 if asset == "GLD" else 10
    cur = rep[(rep.lev == cur_lev) & (rep.tp == 200) & (rep.sl == 50) & (rep.hold == 15)]
    print(f"\n现行参数: lev={cur_lev}× TP=200% SL=50% hold=15d")
    if len(cur):
        c = cur.iloc[0]
        print(f"  → n={int(c['n'])} wr={c['wr']:.1f}% avg={c['avg']:+.2f}% "
              f"sum={c['sum']:+.0f}% sharpe={c['sharpe']:+.3f}")
    print(f"\nTop 8 by cum sum:")
    print(f"  {'lev':>4}{'tp':>5}{'sl':>4}{'hold':>5}{'n':>4}"
          f"{'wr':>7}{'avg':>9}{'sum':>9}{'sharpe':>8}")
    for _, r in rep.head(8).iterrows():
        mark = "  ←现行" if (r["lev"] == cur_lev and r["tp"] == 200
                              and r["sl"] == 50 and r["hold"] == 15) else ""
        print(f"  {int(r['lev']):>3}×{int(r['tp']):>4}%{int(r['sl']):>3}%{int(r['hold']):>4}d"
              f"{int(r['n']):>4}{r['wr']:>6.1f}%{r['avg']:>+8.2f}%{r['sum']:>+8.0f}%"
              f"{r['sharpe']:>+7.3f}{mark}")
    return rep


# ───────────────────────── SHORT_VOL GRID ─────────────────────────
def grid_short_vol(asset: str, days_back: int = 365):
    print(f"\n{'='*78}")
    print(f"【SHORT_VOL grid: {asset}】 修后 max_risk + 近 {days_back} 天")
    print(f"{'='*78}")
    asset_lc = asset.lower()
    feat = pd.read_parquet(
        f"/Users/yhdong/Gold/data/processed/features_{'all' if asset == 'GLD' else 'slv'}.parquet")
    ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset_lc}.csv",
                        index_col=0, parse_dates=True)
    common = ohlc.index.intersection(feat.index)
    today = pd.Timestamp.now().normalize()
    cutoff = today - pd.Timedelta(days=days_back)
    common = common[(common >= cutoff) & (common <= today)]
    # 从 backtest CSV 读 SHORT_VOL 信号
    csvs = sorted(CSV_DIR.glob(f"backtest_{asset_lc}_*.csv"))
    if not csvs:
        print(f"  无 backtest CSV"); return
    df = pd.read_csv(csvs[-1], parse_dates=["signal_date"])
    sv_df = df[(df["strategy"] == "SHORT_VOL") &
                 (df["signal_date"] >= cutoff)]
    sigs = sorted(sv_df["signal_date"].drop_duplicates().to_list())
    if not sigs:
        print(f"  {asset} 无 SHORT_VOL 信号"); return
    print(f"  {asset} 信号数: {len(sigs)}")
    db = _load_kline_db()
    if db is not None and asset and "asset" in db.columns:
        db = db[db["asset"] == asset]
    if db is None or db.empty:
        print(f"  kline_db 空"); return

    # Grid: TP_credit × SL × hold_max × DTE
    tps = [30, 50, 70]
    sls = [30, 50, 70, 100]
    holds = [14, 21, 30, 45]
    dtes = [14, 30, 45]

    results = []
    for tp, sl, hold_max, dte in product(tps, sls, holds, dtes):
        cfg = ShortVolConfig(profit_target_credit_pct=tp,
                              stop_loss_pct=sl,
                              hold_max_days=hold_max,
                              base_dte=dte)
        pnls = []
        for d in sigs:
            d = pd.Timestamp(d).normalize()
            if d not in ohlc.index: continue
            eO = float(ohlc.loc[d, "Open"])
            eC = float(ohlc.loc[d, "Close"])
            eH = float(ohlc.loc[d, "High"])
            eL = float(ohlc.loc[d, "Low"])
            ent = price_strategy_at(asset, "SHORT_VOL", d,
                                       d + pd.Timedelta(hours=9, minutes=30),
                                       eO, eO, eC, eH, eL, dte_target=dte)
            if not ent.get("legs"): continue
            res = simulate_short_vol_position(ent, d, today, db, cfg=cfg)
            if res.get("is_closed"):
                p = max(-100.0, min(100.0, float(res.get("pnl_pct", 0) or 0)))
                pnls.append(p)
        if not pnls: continue
        s = pd.Series(pnls)
        results.append({
            "tp": tp, "sl": sl, "hold": hold_max, "dte": dte,
            "n": len(s), "wr": (s > 0).mean() * 100,
            "avg": s.mean(), "sum": s.sum(),
            "sharpe": s.mean() / s.std() if s.std() > 0 else 0,
        })

    if not results:
        print("  无可平仓数据"); return
    rep = pd.DataFrame(results).sort_values("sum", ascending=False)
    cur = rep[(rep.tp == 50) & (rep.sl == 50) & (rep.hold == 30) & (rep.dte == 30)]
    print(f"\n现行: TP=+50% credit SL=-50% hold=30d DTE=30")
    if len(cur):
        c = cur.iloc[0]
        print(f"  → n={int(c['n'])} wr={c['wr']:.1f}% avg={c['avg']:+.2f}% "
              f"sum={c['sum']:+.0f}% sharpe={c['sharpe']:+.3f}")
    print(f"\nTop 8 by cum sum:")
    print(f"  {'tp':>4}{'sl':>5}{'hold':>5}{'dte':>5}{'n':>4}"
          f"{'wr':>7}{'avg':>9}{'sum':>9}{'sharpe':>8}")
    for _, r in rep.head(8).iterrows():
        mark = "  ←现行" if (r["tp"] == 50 and r["sl"] == 50
                              and r["hold"] == 30 and r["dte"] == 30) else ""
        print(f"  {int(r['tp']):>3}%{int(r['sl']):>4}%{int(r['hold']):>4}d{int(r['dte']):>4}d"
              f"{int(r['n']):>4}{r['wr']:>6.1f}%{r['avg']:>+8.2f}%{r['sum']:>+8.0f}%"
              f"{r['sharpe']:>+7.3f}{mark}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all",
                      choices=["futures", "shortvol", "all"])
    ap.add_argument("--days", type=int, default=365)
    args = ap.parse_args()

    if args.which in ("futures", "all"):
        grid_futures("GLD", args.days)
        grid_futures("SLV", args.days)
    if args.which in ("shortvol", "all"):
        grid_short_vol("GLD", args.days)
        grid_short_vol("SLV", args.days)
