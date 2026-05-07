"""一键生成 positions ledger — single source of truth.

每个 (signal_date, asset, strategy) 一行, 含完整 entry pricing snapshot.
Dashboard 读这个文件就行, 不重算.

输出:
  /Users/yhdong/Gold/data/positions_ledger.parquet
  /Users/yhdong/Gold/data/positions_ledger.json (人类可读)

用法:
  python scripts/build_positions_ledger.py [--days 90]
"""
import sys, os, json, argparse
from pathlib import Path
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf

# 强制重载 paper_positions cache
import core.paper_positions as _pp
_pp._KLINE_DB_CACHE = None
_pp._KLINE_DB_MTIME = None

from core.signals_v2 import generate_daily_signals
from core.events import detect_short_vol_signal, detect_straddle_signal
from core.regime import RegimeClassifier
from core.paper_positions import (price_strategy_at, simulate_option_exit)


LEDGER_PARQUET = "/Users/yhdong/Gold/data/positions_ledger.parquet"
LEDGER_JSON = "/Users/yhdong/Gold/data/positions_ledger.json"


def build_for_asset(asset: str, days_back: int, today_dt: pd.Timestamp,
                       gvz: pd.Series) -> list:
    """对 asset 跑 last N days 信号 + 定价, 生成 ledger rows."""
    asset_lc = asset.lower()
    ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset_lc}.csv",
                        index_col=0, parse_dates=True)
    feat_path = ("/Users/yhdong/Gold/data/processed/features_all.parquet"
                  if asset == "GLD" else
                  "/Users/yhdong/Gold/data/processed/features_slv.parquet")
    features = pd.read_parquet(feat_path)
    common = ohlc.index.intersection(features.index)
    close_d = ohlc.loc[common, "Close"]
    high_d = ohlc.loc[common, "High"]
    low_d = ohlc.loc[common, "Low"]
    sma = close_d.rolling(20).mean()
    std = close_d.rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std

    rv = features.loc[common, "rv_10d"]
    rv_pctile = rv.rank(pct=True)
    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(features.loc[common, feat_cols])["regime"]

    sig_df = generate_daily_signals(close_d, high_d, low_d, upper, lower,
                                       regime, rv_pctile, asset=asset, gvz_series=gvz)

    window_start = today_dt - timedelta(days=days_back)
    u_dates = sig_df.index[sig_df.index >= window_start]

    strad_df = detect_straddle_signal(rv, u_dates, rv_pctile=rv_pctile,
                                          close=close_d, high=high_d, low=low_d,
                                          asset=asset)
    sv_df = detect_short_vol_signal(rv, rv_pctile, u_dates, regime=regime,
                                          close=close_d, high=high_d, low=low_d,
                                          asset=asset)

    rows = []
    for _du, _ru in sig_df.loc[u_dates].iterrows():
        is_strad = (_du in strad_df.index
                     and bool(strad_df.loc[_du, "straddle_signal"]))
        is_sv = (_du in sv_df.index
                  and bool(sv_df.loc[_du, "short_vol_signal"]))
        strats = []
        if is_strad: strats.append("STRADDLE")
        if is_sv: strats.append("SHORT_VOL")
        if _ru.get("buy_signal", False):
            bt = _ru.get("buy_type") or ""
            if bt: strats.append(bt)
            strats.append("FUTURES_LONG")
        if not strats: continue

        if _du not in ohlc.index: continue
        eO = float(ohlc.loc[_du, "Open"])
        eC = float(ohlc.loc[_du, "Close"])
        eH = float(ohlc.loc[_du, "High"])
        eL = float(ohlc.loc[_du, "Low"])

        for strat in strats:
            ent = price_strategy_at(asset, strat, _du,
                                       _du + pd.Timedelta(hours=9, minutes=30),
                                       eO, eO, eC, eH, eL,
                                       dte_target=(14 if strat == "STRADDLE" else 30))
            legs = ent.get("legs", [])
            if not legs and strat != "FUTURES_LONG":
                continue  # kline_db 没数据
            sim = simulate_option_exit(ent, _du, strat, today_dt,
                                            live_spot=eC, live_high=eH, live_low=eL)
            is_closed = sim.get("is_closed", False)
            row = {
                "asset": asset,
                "signal_date": _du.isoformat(),
                "strategy": strat,
                "entry_etf": eO,
                "entry_open": eO,
                "entry_close": eC,
                "source": ent.get("source", "—"),
                "entry_credit_or_premium": float(ent.get("entry_price", 0) or 0),
                "legs": [list(l) for l in legs],
                "entry_leg_prices": [list(p) for p in ent.get("leg_prices", [])],
                "exit_legs": [list(p) for p in sim.get("leg_prices", [])],
                "is_closed": is_closed,
                "exit_date": (sim.get("exit_date").isoformat()
                                if isinstance(sim.get("exit_date"), pd.Timestamp)
                                else None),
                "exit_value": float(sim.get("exit_value", 0) or 0),
                "exit_reason": sim.get("exit_reason", ""),
                "current_value": float(sim.get("current_value", 0) or 0),
                "pnl_pct": float(sim.get("pnl_pct", 0) or 0),
                "hold_days": int(sim.get("hold_days", 0) or 0),
            }
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    args = ap.parse_args()

    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    today_dt = pd.Timestamp(today_et)
    print(f"[ledger] today_et = {today_et}, days_back = {args.days}")

    gvz = yf.Ticker("^GVZ").history(period="5y")
    gvz.index = pd.to_datetime(gvz.index).tz_localize(None).normalize()
    gvz_close = gvz["Close"]

    all_rows = []
    for asset in ["GLD", "SLV"]:
        rows = build_for_asset(asset, args.days, today_dt, gvz_close)
        print(f"[ledger] {asset}: {len(rows)} positions")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("[ledger] 无信号, 不写入"); return

    os.makedirs(os.path.dirname(LEDGER_JSON), exist_ok=True)
    # 写 JSON (主格式 — 人类可读, dashboard 也读这个)
    df_sorted = df.sort_values(["asset", "signal_date", "strategy"])
    with open(LEDGER_JSON, "w") as f:
        json.dump(df_sorted.to_dict(orient="records"), f,
                    indent=2, default=str, ensure_ascii=False)

    print(f"\n[ledger] saved {len(df)} rows:")
    print(f"  parquet: {LEDGER_PARQUET}")
    print(f"  json:    {LEDGER_JSON}")
    print(f"\n按 asset / strategy 汇总:")
    print(df.groupby(["asset", "strategy"]).size().to_string())
    print(f"\nSLV recent (last 5d):")
    slv = df[df["asset"] == "SLV"].sort_values("signal_date", ascending=False).head(15)
    print(slv[["signal_date", "strategy", "source", "is_closed",
                "pnl_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
