"""v3.7.200 paired: 单腿 BC vs +5% 价差 BC (同 sig_date 才纳入)

避免之前 n=40 vs n=19 的样本选择偏差.

只保留 sig_date 同时能开 single AND spread 的笔, 一一比较.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd

from core.data import load_features, load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier
from core.paper_positions import (_load_kline_db, pick_liquid_monthly_option,
                                      interpolate_option_intraday)
from core.strategies.buy_call import simulate_bc_position, BCConfig


def make_bc_entry(asset, sig_d, spot_at_trigger, eO, eC, eH, eL,
                    spread_pct=0.0, dte_target=30):
    lc = pick_liquid_monthly_option(asset, sig_d, spot_at_trigger, "C",
                                       dte_target=dte_target, min_dte=14)
    if not lc: return None
    lc_intra = interpolate_option_intraday(lc, eO, eC, spot_at_trigger, eH, eL)
    if spread_pct <= 0:
        return {"legs": [("long_call", lc["code"], lc["strike"], 1)],
                  "entry_price": lc_intra,
                  "leg_prices": [("long_call", lc_intra)],
                  "source": f"C${lc['strike']:.0f}",
                  "_long_strike": lc["strike"], "_long_close": lc["close"]}
    short_target = round(lc["strike"] * (1 + spread_pct))
    sc = pick_liquid_monthly_option(asset, sig_d, short_target, "C",
                                        dte_target=dte_target, min_dte=14)
    if not sc or sc["strike"] <= lc["strike"]: return None
    sc_intra = interpolate_option_intraday(sc, eO, eC, spot_at_trigger, eH, eL)
    debit = lc_intra - sc_intra
    if debit <= 0.01: return None
    return {"legs": [("long_call", lc["code"], lc["strike"], 1),
                       ("short_call", sc["code"], sc["strike"], -1)],
              "entry_price": debit,
              "leg_prices": [("long_call", lc_intra), ("short_call", sc_intra)],
              "source": f"+C${lc['strike']:.0f}/-C${sc['strike']:.0f}",
              "_long_strike": lc["strike"], "_short_strike": sc["strike"]}


def collect_bc(asset, lookback):
    cfg = load_config()
    if asset == "GLD":
        oos = load_oos_predictions(cfg)
        feat = pd.read_parquet("/Users/yhdong/Gold/data/processed/features_all.parquet")
    else:
        oos = pd.read_parquet("/Users/yhdong/Gold/data/models/dl_range_slv_oos.parquet")
        feat = pd.read_parquet("/Users/yhdong/Gold/data/processed/features_slv.parquet")
    ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                          index_col=0, parse_dates=True)
    common = ohlc.index.intersection(feat.index).intersection(oos.index)
    close = ohlc.loc[common, "Close"]; high = ohlc.loc[common, "High"]
    low = ohlc.loc[common, "Low"]
    upper, lower, _ = build_band(oos.loc[common], close)
    feat_cols = [c for c in feat.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(feat.loc[common, feat_cols])["regime"]
    rv_p = compute_rv_pctile(feat.loc[common, "rv_10d"])
    sig = generate_daily_signals(close, high, low, upper, lower, regime, rv_p, asset=asset)
    bc = sig[sig["buy_type"] == "BUY CALL"].copy()
    cutoff = ohlc.index.max() - pd.Timedelta(days=lookback)
    bc = bc[bc.index >= cutoff].join(ohlc[["Open","High","Low","Close"]], how="inner")
    return bc


def main():
    db = _load_kline_db()
    today = db["date"].max()
    print(f"kline_db: {db['date'].min().date()} → {today.date()}\n")
    cfg = BCConfig()
    out_rows = []
    for asset in ["GLD", "SLV"]:
        for lookback in [365]:
            bc = collect_bc(asset, lookback)
            print(f"=== {asset} {lookback}d ({len(bc)} signals) ===")
            paired = []
            for sig_d, row in bc.iterrows():
                eO,eC,eH,eL = float(row["Open"]),float(row["Close"]),float(row["High"]),float(row["Low"])
                a = make_bc_entry(asset, sig_d, eO, eO, eC, eH, eL, spread_pct=0.0, dte_target=cfg.base_dte)
                b = make_bc_entry(asset, sig_d, eO, eO, eC, eH, eL, spread_pct=0.05, dte_target=cfg.base_dte)
                if not (a and b): continue
                ra = simulate_bc_position(a, sig_d, today, db, cfg)
                rb = simulate_bc_position(b, sig_d, today, db, cfg)
                if not (ra.get("is_closed") and rb.get("is_closed")): continue
                paired.append({
                    "sig_d": sig_d.date(),
                    "a_long_K": a["_long_strike"],
                    "b_short_K": b.get("_short_strike"),
                    "a_pnl": max(-100, min(500, float(ra.get("pnl_pct", 0)))),
                    "b_pnl": max(-100, min(500, float(rb.get("pnl_pct", 0)))),
                    "a_exit": ra.get("exit_reason"),
                    "b_exit": rb.get("exit_reason"),
                    "a_hold": ra.get("hold_days"),
                    "b_hold": rb.get("hold_days"),
                })
            if not paired: print("  no paired"); continue
            pdf = pd.DataFrame(paired)
            print(pdf.to_string(index=False))
            n = len(pdf)
            a_wr = (pdf["a_pnl"]>0).mean()*100
            b_wr = (pdf["b_pnl"]>0).mean()*100
            both_win = ((pdf["a_pnl"]>0)&(pdf["b_pnl"]>0)).sum()
            both_lose = ((pdf["a_pnl"]<=0)&(pdf["b_pnl"]<=0)).sum()
            a_win_b_lose = ((pdf["a_pnl"]>0)&(pdf["b_pnl"]<=0)).sum()
            b_win_a_lose = ((pdf["a_pnl"]<=0)&(pdf["b_pnl"]>0)).sum()
            print(f"\n{asset} paired n={n}:")
            print(f"  single WR={a_wr:.1f}% sum={pdf['a_pnl'].sum():+.0f}% mean={pdf['a_pnl'].mean():+.1f}%")
            print(f"  +5% sp WR={b_wr:.1f}% sum={pdf['b_pnl'].sum():+.0f}% mean={pdf['b_pnl'].mean():+.1f}%")
            print(f"  both win: {both_win}/{n}, both lose: {both_lose}/{n}, "
                    f"single win/spread lose: {a_win_b_lose}, spread win/single lose: {b_win_a_lose}")
            out_rows.append({"asset": asset, "lookback": lookback, "n_paired": n,
                              "single_WR": a_wr, "single_sum": pdf["a_pnl"].sum(),
                              "spread_WR": b_wr, "spread_sum": pdf["b_pnl"].sum(),
                              "both_win": both_win, "both_lose": both_lose,
                              "sig_only_single_win": a_win_b_lose,
                              "sig_only_spread_win": b_win_a_lose})
            print()
    Path("/Users/yhdong/Gold/data/backtest_history/bc_paired_grid.csv").write_text(
        pd.DataFrame(out_rows).to_csv(index=False))


if __name__ == "__main__":
    main()
