"""v3.7.225: SLV-S → GLD 同步, 三种策略的 BACKTEST P&L 对比.

Task 2 历史样本用 spot 回报, 但实际交易要看策略层 P&L:
  - BC 期权: theta 风险, 杠杆大 (option)
  - SELL PUT 期权: theta 友好, 但极端下跌全亏
  - FUTURES_LONG: 5x lev, 无 theta, 直接跟 spot

跑 historical SLV-S 日期 × GLD 三策略 → 看真实回测胜率 / mean / max_loss.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf, numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier
from core.paper_positions import price_strategy_at, simulate_option_exit
from core.strategy_configs import get_futures_config
from core.strategies.futures_long import simulate_long_position
from core.binance_futures import (fetch_perp_price_at_date, fetch_perp_klines,
                                      ASSET_SYMBOL)


def find_slv_s_dates():
    """跑 SLV signals 找所有 tier=S 日期."""
    cfg = load_config()
    oos = pd.read_parquet(Path(cfg["data_root"]) / "models/dl_range_slv_oos.parquet")
    feat = pd.read_parquet("/Users/yhdong/Gold/data/processed/features_slv.parquet")
    ohlc = pd.read_csv("/Users/yhdong/Gold/data/raw/market/slv.csv",
                          index_col=0, parse_dates=True)
    common = ohlc.index.intersection(feat.index).intersection(oos.index)
    close = ohlc.loc[common,"Close"]; high = ohlc.loc[common,"High"]; low = ohlc.loc[common,"Low"]
    upper, lower, _ = build_band(oos.loc[common], close)
    rv_p = compute_rv_pctile(feat.loc[common,"rv_10d"])
    feat_cols = [c for c in feat.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier(min_hold_days=1).classify(
        feat.loc[common, feat_cols])["regime"]
    gvz = yf.Ticker("^GVZ").history(period="10y")
    gvz.index = pd.to_datetime(gvz.index).tz_localize(None).normalize()
    sig = generate_daily_signals(close, high, low, upper, lower, regime, rv_p,
                                       asset="SLV", gvz_series=gvz["Close"])
    s_dates = sig.index[(sig["buy_signal"]) & (sig["signal_tier"] == "S")]
    return list(s_dates)


def backtest_gld_at(dates, strategy):
    """对给定日期列表跑 GLD 策略入场+模拟, 返回 P&L list."""
    g_ohlc = pd.read_csv("/Users/yhdong/Gold/data/raw/market/gld.csv",
                            index_col=0, parse_dates=True)
    today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    results = []
    for d in dates:
        if d not in g_ohlc.index: continue
        eO = float(g_ohlc.loc[d,"Open"]); eC = float(g_ohlc.loc[d,"Close"])
        eH = float(g_ohlc.loc[d,"High"]); eL = float(g_ohlc.loc[d,"Low"])
        if strategy == "FUTURES_LONG":
            sym = ASSET_SYMBOL["GLD"]
            try:
                bin_entry = fetch_perp_price_at_date(sym, d)
                if not bin_entry: continue
                entry_perp = bin_entry["open"]
                start_ms = int((d + pd.Timedelta(days=1)).timestamp() * 1000)
                end_ms = int(today.timestamp() * 1000)
                kl = fetch_perp_klines(sym, start_ms, end_ms, "1d")
                recs = [{"Date": pd.Timestamp(k[0], unit="ms").normalize(),
                            "Open": float(k[1]), "High": float(k[2]),
                            "Low": float(k[3]), "Close": float(k[4])} for k in kl]
                if not recs: continue
                df_perp = pd.DataFrame(recs).set_index("Date")
                if d not in df_perp.index:
                    df_perp.loc[d] = {"Open": entry_perp, "High": bin_entry["high"],
                                          "Low": bin_entry["low"],
                                          "Close": bin_entry["close"]}
                    df_perp = df_perp.sort_index()
                cfg = get_futures_config("GLD")
                sim = simulate_long_position(
                    entry_d=d, entry_spot=entry_perp,
                    ohlc=df_perp, today=today, cfg=cfg,
                    live_spot=df_perp.iloc[-1]["Close"], signal_tier="S")
                pnl = max(-100.0, float(sim.get("ret_levered_pct", 0) or 0))
                results.append({"date": d, "strategy": strategy,
                                  "closed": sim.get("closed", False),
                                  "pnl_pct": pnl,
                                  "hold_days": sim.get("hold_days", 0),
                                  "reason": sim.get("reason", "")})
            except Exception as e:
                continue
        else:
            ent = price_strategy_at("GLD", strategy, d,
                                          d + pd.Timedelta(hours=9, minutes=30),
                                          eO, eO, eC, eH, eL,
                                          dte_target=30)
            if not ent.get("legs"): continue
            sim = simulate_option_exit(ent, d, strategy, today,
                                              live_spot=eC, live_high=eH, live_low=eL)
            results.append({"date": d, "strategy": strategy,
                              "closed": sim.get("is_closed", False),
                              "pnl_pct": float(sim.get("pnl_pct", 0) or 0),
                              "hold_days": int(sim.get("hold_days", 0) or 0),
                              "reason": sim.get("exit_reason", "open")})
    return results


def summary(results, label):
    closed = [r for r in results if r["closed"]]
    open_ = [r for r in results if not r["closed"]]
    print(f"\n{label}:  total={len(results)}, closed={len(closed)}, open={len(open_)}")
    if closed:
        pnls = [r["pnl_pct"] for r in closed]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        print(f"  closed: WR={wr:.1f}%, mean={np.mean(pnls):+.1f}%, "
              f"sum={sum(pnls):+.1f}%, max_loss={min(pnls):+.1f}%, "
              f"max_gain={max(pnls):+.1f}%")
        for r in closed:
            print(f"    {r['date'].date()} pnl={r['pnl_pct']:+.1f}% "
                  f"hold={r['hold_days']}d reason={r['reason']}")
    if open_:
        pnls = [r["pnl_pct"] for r in open_]
        print(f"  open  : mean={np.mean(pnls):+.1f}%, max_loss={min(pnls):+.1f}%, "
              f"max_gain={max(pnls):+.1f}%")
        for r in open_:
            print(f"    {r['date'].date()} pnl={r['pnl_pct']:+.1f}% open hold={r['hold_days']}d")


def main():
    print("找 SLV-S 信号日期 ...")
    s_dates = find_slv_s_dates()
    print(f"SLV tier=S 共 {len(s_dates)} 笔: {[d.date() for d in s_dates]}")

    for strat in ["BUY CALL", "SELL PUT", "FUTURES_LONG"]:
        print(f"\n{'='*100}\n{strat}\n{'='*100}")
        res = backtest_gld_at(s_dates, strat)
        summary(res, f"GLD {strat} (SLV-S sync 入场)")


if __name__ == "__main__":
    main()
