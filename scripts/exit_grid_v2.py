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
from core.strategies.buy_call import (simulate_bc_position, BCConfig)
from core.strategies.sell_put import (simulate_sp_position, SPConfig)
from core.paper_positions import price_strategy_at, _load_kline_db
from core.events import detect_short_vol_signal as _det_sv
from core.signals import compute_rv_pctile as _rv_pct


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


def grid_futures(asset: str, days_back: int = 365, source: str = "auto"):
    """source: 'binance' / 'comex' / 'auto' (auto: ≤90d binance, >90d comex)."""
    sym = ASSET_SYMBOL[asset]
    print(f"\n{'='*78}")
    print(f"【期货 grid: {asset} ({sym})】 近 {days_back} 天 source={source}")
    print(f"{'='*78}")
    sigs, _ = load_futures_signals(asset, days_back)
    if not sigs:
        print(f"  {asset} 无信号"); return
    today = pd.Timestamp.now().normalize()
    # Source selection
    use_comex = (source == "comex" or (source == "auto" and days_back > 150))
    if use_comex:
        import yfinance as yf
        comex_sym = "GC=F" if asset == "GLD" else "SI=F"
        df_perp = yf.Ticker(comex_sym).history(period=f"{max(days_back, 365)+30}d")
        df_perp.index = pd.to_datetime(df_perp.index).tz_localize(None).normalize()
        df_perp = df_perp[["Open", "High", "Low", "Close"]].dropna()
        print(f"  using COMEX {comex_sym}: {len(df_perp)} bars "
              f"({df_perp.index.min().date()}..{df_perp.index.max().date()})")
    else:
        df_perp = fetch_binance_window(sym, days_back)
        print(f"  using Binance {sym}: {len(df_perp)} bars "
              f"({df_perp.index.min().date()}..{df_perp.index.max().date()})"
              if not df_perp.empty else "  Binance empty")
    if df_perp.empty:
        print(f"  数据空"); return

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
        wr_v = (s > 0).mean()
        wins = s[s > 0]; losses = s[s <= 0]
        avg_w = wins.mean() if len(wins) else 0.0
        avg_l = losses.mean() if len(losses) else 0.0
        pf = wins.sum() / abs(losses.sum()) if losses.sum() < 0 else float("inf")
        kelly_f = max(0.0, wr_v - (1 - wr_v) * abs(avg_l) / avg_w) if avg_w > 0 else 0.0
        results.append({
            "lev": lev, "tp": tp, "sl": sl, "hold": hold_max,
            "n": len(s), "wr": wr_v * 100,
            "avg": s.mean(), "sum": s.sum(),
            "sharpe": s.mean() / s.std() if s.std() > 0 else 0,
            # v3.7.170 高杠杆评分 (用户建议)
            "scoreA": wr_v * len(s) * s.mean(),                  # WR × n × avg
            "scoreB": (wr_v ** 2) * np.log(1 + len(s)) * s.mean(),  # WR² × log(n) × avg
            "scoreC": kelly_f * np.sqrt(len(s)) * s.mean(),     # Kelly 加权
            "pf": min(pf, 99.99),
        })

    df_all = pd.DataFrame(results)
    rep = df_all.sort_values("scoreB", ascending=False)
    cur_lev = 20 if asset == "GLD" else 10
    cur = df_all[(df_all.lev == cur_lev) & (df_all.tp == 200) &
                  (df_all.sl == 50) & (df_all.hold == 15)]
    print(f"\n现行参数: lev={cur_lev}× TP=200% SL=50% hold=15d")
    if len(cur):
        c = cur.iloc[0]
        print(f"  → n={int(c['n'])} wr={c['wr']:.1f}% avg={c['avg']:+.2f}% "
              f"sum={c['sum']:+.0f}% sharpe={c['sharpe']:+.3f} "
              f"scoreB={c['scoreB']:+.2f}")
    print(f"\nTop 8 by scoreB = WR² × log(1+n) × avg (高杠杆评分):")
    print(f"  {'lev':>4}{'tp':>5}{'sl':>4}{'hold':>5}{'n':>4}"
          f"{'wr':>7}{'avg':>9}{'sum':>9}{'sharpe':>8}{'scoreB':>9}")
    for _, r in rep.head(8).iterrows():
        mark = "  ←现行" if (r["lev"] == cur_lev and r["tp"] == 200
                              and r["sl"] == 50 and r["hold"] == 15) else ""
        print(f"  {int(r['lev']):>3}×{int(r['tp']):>4}%{int(r['sl']):>3}%{int(r['hold']):>4}d"
              f"{int(r['n']):>4}{r['wr']:>6.1f}%{r['avg']:>+8.2f}%{r['sum']:>+8.0f}%"
              f"{r['sharpe']:>+7.3f}{r['scoreB']:>+8.1f}{mark}")
    print(f"\n参考 — Top by 单一指标:")
    rep_wr = df_all.sort_values("wr", ascending=False).head(2)
    rep_pf = df_all.sort_values("pf", ascending=False).head(2)
    rep_sum = df_all.sort_values("sum", ascending=False).head(2)
    print(f"  by WR:")
    for _, r in rep_wr.iterrows():
        print(f"    {int(r['lev']):>3}× TP{int(r['tp'])}/SL{int(r['sl'])}/h{int(r['hold'])}d "
              f"wr={r['wr']:.1f}% sum={r['sum']:+.0f}% scoreB={r['scoreB']:+.1f}")
    print(f"  by sum:")
    for _, r in rep_sum.iterrows():
        print(f"    {int(r['lev']):>3}× TP{int(r['tp'])}/SL{int(r['sl'])}/h{int(r['hold'])}d "
              f"wr={r['wr']:.1f}% sum={r['sum']:+.0f}% scoreB={r['scoreB']:+.1f}")
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
    rep = pd.DataFrame(results).sort_values("wr", ascending=False)
    cur = rep[(rep.tp == 50) & (rep.sl == 50) & (rep.hold == 30) & (rep.dte == 30)]
    print(f"\n现行: TP=+50% credit SL=-50% hold=30d DTE=30")
    if len(cur):
        c = cur.iloc[0]
        print(f"  → n={int(c['n'])} wr={c['wr']:.1f}% avg={c['avg']:+.2f}% "
              f"sum={c['sum']:+.0f}% sharpe={c['sharpe']:+.3f}")
    print(f"\nTop 8 by WIN RATE (n≥{min(20, max(5, len(rep)//5))} 过滤后):")
    print(f"  {'tp':>4}{'sl':>5}{'hold':>5}{'dte':>5}{'n':>4}"
          f"{'wr':>7}{'avg':>9}{'sum':>9}{'sharpe':>8}")
    for _, r in rep.head(8).iterrows():
        mark = "  ←现行" if (r["tp"] == 50 and r["sl"] == 50
                              and r["hold"] == 30 and r["dte"] == 30) else ""
        print(f"  {int(r['tp']):>3}%{int(r['sl']):>4}%{int(r['hold']):>4}d{int(r['dte']):>4}d"
              f"{int(r['n']):>4}{r['wr']:>6.1f}%{r['avg']:>+8.2f}%{r['sum']:>+8.0f}%"
              f"{r['sharpe']:>+7.3f}{mark}")
    return rep


# ───────────────────────── BC / SP GRID ─────────────────────────
def _load_signals_from_csv(asset: str, strategy: str, days_back: int):
    asset_lc = asset.lower()
    today = pd.Timestamp.now().normalize()
    cutoff = today - pd.Timedelta(days=days_back)
    csvs = sorted(CSV_DIR.glob(f"backtest_{asset_lc}_*.csv"))
    if not csvs: return []
    df = pd.read_csv(csvs[-1], parse_dates=["signal_date"])
    sub = df[(df["strategy"] == strategy) & (df["signal_date"] >= cutoff)]
    return sorted(sub["signal_date"].drop_duplicates().to_list())


def grid_bc(asset: str, days_back: int = 365):
    print(f"\n{'='*78}")
    print(f"【BUY CALL grid: {asset}】 近 {days_back} 天")
    print(f"{'='*78}")
    sigs = _load_signals_from_csv(asset, "BUY CALL", days_back)
    if not sigs: print("  无信号"); return
    asset_lc = asset.lower()
    ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset_lc}.csv",
                        index_col=0, parse_dates=True)
    today = pd.Timestamp.now().normalize()
    db = _load_kline_db()
    if db is None or db.empty:
        print("  kline_db 空"); return
    if "asset" in db.columns: db = db[db["asset"] == asset]
    print(f"  信号数: {len(sigs)} | kline_db rows: {len(db)}")

    # Grid: profit_target_mult × stop_loss_mult × DTE
    pts = [1.5, 2.0, 2.5, 3.0]    # +50%/+100%/+150%/+200% premium
    sls = [0.3, 0.5, 0.7, 1.0]    # -70%/-50%/-30%/0% premium
    dtes = [30, 45, 60]

    results = []
    for pt, sl, dte in product(pts, sls, dtes):
        cfg = BCConfig(profit_target_mult=pt, stop_loss_mult=sl, base_dte=dte)
        pnls = []
        for d in sigs:
            d = pd.Timestamp(d).normalize()
            if d not in ohlc.index: continue
            eO = float(ohlc.loc[d, "Open"])
            eC = float(ohlc.loc[d, "Close"])
            eH = float(ohlc.loc[d, "High"])
            eL = float(ohlc.loc[d, "Low"])
            ent = price_strategy_at(asset, "BUY CALL", d,
                                       d + pd.Timedelta(hours=9, minutes=30),
                                       eO, eO, eC, eH, eL, dte_target=dte)
            if not ent.get("legs"): continue
            res = simulate_bc_position(ent, d, today, db, cfg=cfg)
            if res.get("is_closed"):
                p = max(-100.0, min(500.0, float(res.get("pnl_pct", 0) or 0)))
                pnls.append(p)
        if not pnls: continue
        s = pd.Series(pnls)
        results.append({
            "pt": pt, "sl": sl, "dte": dte, "n": len(s),
            "wr": (s > 0).mean() * 100, "avg": s.mean(), "sum": s.sum(),
            "sharpe": s.mean() / s.std() if s.std() > 0 else 0,
        })
    if not results: print("  no results"); return
    rep = pd.DataFrame(results).sort_values("wr", ascending=False)
    cur = rep[(rep.pt == 2.0) & (rep.sl == 0.5) & (rep.dte == 45)]
    print(f"\n现行: pt=+100% sl=-50% DTE=45")
    if len(cur):
        c = cur.iloc[0]
        print(f"  → n={int(c['n'])} wr={c['wr']:.1f}% avg={c['avg']:+.2f}% "
              f"sum={c['sum']:+.0f}% sharpe={c['sharpe']:+.3f}")
    print(f"\nTop 8:")
    print(f"  {'pt':>5}{'sl':>5}{'dte':>5}{'n':>4}{'wr':>7}{'avg':>9}{'sum':>9}{'sharpe':>8}")
    for _, r in rep.head(8).iterrows():
        mark = "  ←现行" if (r["pt"] == 2.0 and r["sl"] == 0.5 and r["dte"] == 45) else ""
        print(f"  {r['pt']:>4.1f}x{r['sl']:>4.1f}x{int(r['dte']):>4}d"
              f"{int(r['n']):>4}{r['wr']:>6.1f}%{r['avg']:>+8.2f}%"
              f"{r['sum']:>+8.0f}%{r['sharpe']:>+7.3f}{mark}")


def grid_sp(asset: str, days_back: int = 365):
    print(f"\n{'='*78}")
    print(f"【SELL PUT grid: {asset}】 近 {days_back} 天")
    print(f"{'='*78}")
    sigs = _load_signals_from_csv(asset, "SELL PUT", days_back)
    if not sigs: print("  无信号"); return
    asset_lc = asset.lower()
    ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset_lc}.csv",
                        index_col=0, parse_dates=True)
    today = pd.Timestamp.now().normalize()
    db = _load_kline_db()
    if db is None or db.empty: print("  kline_db 空"); return
    if "asset" in db.columns: db = db[db["asset"] == asset]
    print(f"  信号数: {len(sigs)} | kline_db rows: {len(db)}")

    pts = [30, 50, 70]            # profit_target_credit_pct
    sls = [30, 50, 70, 100]       # stop_loss_margin_pct
    dtes = [30, 45, 60]

    results = []
    for pt, sl, dte in product(pts, sls, dtes):
        cfg = SPConfig(profit_target_credit_pct=pt,
                         stop_loss_margin_pct=sl, base_dte=dte)
        pnls = []
        for d in sigs:
            d = pd.Timestamp(d).normalize()
            if d not in ohlc.index: continue
            eO = float(ohlc.loc[d, "Open"]); eC = float(ohlc.loc[d, "Close"])
            eH = float(ohlc.loc[d, "High"]); eL = float(ohlc.loc[d, "Low"])
            ent = price_strategy_at(asset, "SELL PUT", d,
                                       d + pd.Timedelta(hours=9, minutes=30),
                                       eO, eO, eC, eH, eL, dte_target=dte)
            if not ent.get("legs"): continue
            res = simulate_sp_position(ent, d, today, db, cfg=cfg)
            if res.get("is_closed"):
                p = max(-100.0, min(150.0, float(res.get("pnl_pct", 0) or 0)))
                pnls.append(p)
        if not pnls: continue
        s = pd.Series(pnls)
        results.append({
            "pt": pt, "sl": sl, "dte": dte, "n": len(s),
            "wr": (s > 0).mean() * 100, "avg": s.mean(), "sum": s.sum(),
            "sharpe": s.mean() / s.std() if s.std() > 0 else 0,
        })
    if not results: print("  no results"); return
    rep = pd.DataFrame(results).sort_values("wr", ascending=False)
    cur = rep[(rep.pt == 50) & (rep.sl == 50) & (rep.dte == 45)]
    print(f"\n现行: pt=+50% credit sl=-50% margin DTE=45")
    if len(cur):
        c = cur.iloc[0]
        print(f"  → n={int(c['n'])} wr={c['wr']:.1f}% avg={c['avg']:+.2f}% "
              f"sum={c['sum']:+.0f}% sharpe={c['sharpe']:+.3f}")
    print(f"\nTop 8:")
    print(f"  {'pt':>4}{'sl':>5}{'dte':>5}{'n':>4}{'wr':>7}{'avg':>9}{'sum':>9}{'sharpe':>8}")
    for _, r in rep.head(8).iterrows():
        mark = "  ←现行" if (r["pt"] == 50 and r["sl"] == 50 and r["dte"] == 45) else ""
        print(f"  {int(r['pt']):>3}%{int(r['sl']):>4}%{int(r['dte']):>4}d"
              f"{int(r['n']):>4}{r['wr']:>6.1f}%{r['avg']:>+8.2f}%"
              f"{r['sum']:>+8.0f}%{r['sharpe']:>+7.3f}{mark}")


# ───────────────────────── SHORT_VOL via DETECTION ─────────────────────────
def grid_sv_via_detect(asset: str, days_back: int = 60):
    """SHORT_VOL grid — 直接 detect 信号 (不依赖 backtest CSV)."""
    print(f"\n{'='*78}")
    print(f"【SHORT_VOL grid (detect): {asset}】 近 {days_back} 天")
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
    feat_cols = [c for c in feat.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(feat[feat_cols])["regime"]
    rv_col = "rv_10d" if "rv_10d" in feat.columns else "rv_5d"
    rv_pct = _rv_pct(feat[rv_col])
    sv = _det_sv(feat[rv_col], rv_pct, common, regime=regime)
    sigs = [d for d in common if d in sv.index and bool(sv.loc[d, "short_vol_signal"])]
    if not sigs: print(f"  {asset} 无 SHORT_VOL 信号"); return
    print(f"  {asset} detect 信号数: {len(sigs)}")
    db = _load_kline_db()
    if db is None or db.empty: print("  kline_db 空"); return
    if "asset" in db.columns: db = db[db["asset"] == asset]

    tps = [30, 50, 70]
    sls = [30, 50, 70, 100]
    holds = [14, 21, 30]
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
            eO = float(ohlc.loc[d, "Open"]); eC = float(ohlc.loc[d, "Close"])
            eH = float(ohlc.loc[d, "High"]); eL = float(ohlc.loc[d, "Low"])
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
            "tp": tp, "sl": sl, "hold": hold_max, "dte": dte, "n": len(s),
            "wr": (s > 0).mean() * 100, "avg": s.mean(), "sum": s.sum(),
            "sharpe": s.mean() / s.std() if s.std() > 0 else 0,
        })
    if not results: print("  无平仓样本"); return
    rep = pd.DataFrame(results).sort_values("wr", ascending=False)
    cur = rep[(rep.tp == 50) & (rep.sl == 50) & (rep.hold == 30) & (rep.dte == 30)]
    print(f"\n现行: TP=+50% SL=-50% hold=30d DTE=30")
    if len(cur):
        c = cur.iloc[0]
        print(f"  → n={int(c['n'])} wr={c['wr']:.1f}% avg={c['avg']:+.2f}% "
              f"sum={c['sum']:+.0f}% sharpe={c['sharpe']:+.3f}")
    print(f"\nTop 8:")
    print(f"  {'tp':>4}{'sl':>5}{'hold':>5}{'dte':>5}{'n':>4}{'wr':>7}{'avg':>9}{'sum':>9}{'sharpe':>8}")
    for _, r in rep.head(8).iterrows():
        mark = "  ←现行" if (r["tp"] == 50 and r["sl"] == 50
                              and r["hold"] == 30 and r["dte"] == 30) else ""
        print(f"  {int(r['tp']):>3}%{int(r['sl']):>4}%{int(r['hold']):>4}d{int(r['dte']):>4}d"
              f"{int(r['n']):>4}{r['wr']:>6.1f}%{r['avg']:>+8.2f}%"
              f"{r['sum']:>+8.0f}%{r['sharpe']:>+7.3f}{mark}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all",
                      choices=["futures", "shortvol", "bc", "sp", "options", "all"])
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--source", default="auto",
                      choices=["auto", "binance", "comex"],
                      help="期货价源: binance (5m perp), comex (年级 GC/SI=F)")
    args = ap.parse_args()

    if args.which in ("futures", "all"):
        grid_futures("GLD", args.days, source=args.source)
        grid_futures("SLV", args.days, source=args.source)
    if args.which in ("bc", "options", "all"):
        grid_bc("GLD", args.days)
        grid_bc("SLV", args.days)
    if args.which in ("sp", "options", "all"):
        grid_sp("GLD", args.days)
        grid_sp("SLV", args.days)
    if args.which in ("shortvol", "options", "all"):
        grid_sv_via_detect("GLD", min(args.days, 60))
        grid_sv_via_detect("SLV", min(args.days, 60))
