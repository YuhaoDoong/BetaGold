"""v3.7.200 GLD 单腿 BUY CALL 透明回测

只 GLD, 只单腿, 用生产同款方法学:
  signals  : generate_daily_signals (signals_v2, IV 三阶过滤)
  entry    : pick_liquid_monthly_option(dte=30, min_dte=14)
              + interpolate_option_intraday(eO, eO, eC, eH, eL)
  exit     : simulate_bc_position(BCConfig: TP 1.5x, SL 0.5x)
  sample   : is_closed=True only

输出:
  per-trade CSV  → bc_gld_per_trade.csv  (每笔细节)
  console table  → 直接给用户看
  summary        → WR / sum / mean / max_loss / max_win / PF
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier
from core.paper_positions import (_load_kline_db, pick_liquid_monthly_option,
                                      interpolate_option_intraday)
from core.strategies.buy_call import simulate_bc_position, BCConfig


ASSET = "GLD"
OUT_CSV = "/Users/yhdong/Gold/data/backtest_history/bc_gld_per_trade.csv"


def collect_signals() -> pd.DataFrame:
    cfg = load_config()
    oos = load_oos_predictions(cfg)
    feat = pd.read_parquet("/Users/yhdong/Gold/data/processed/features_all.parquet")
    ohlc = pd.read_csv("/Users/yhdong/Gold/data/raw/market/gld.csv",
                          index_col=0, parse_dates=True)
    common = ohlc.index.intersection(feat.index).intersection(oos.index)
    close = ohlc.loc[common, "Close"]; high = ohlc.loc[common, "High"]
    low = ohlc.loc[common, "Low"]
    upper, lower, _ = build_band(oos.loc[common], close)
    feat_cols = [c for c in feat.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(feat.loc[common, feat_cols])["regime"]
    rv_p = compute_rv_pctile(feat.loc[common, "rv_10d"])
    sig = generate_daily_signals(close, high, low, upper, lower, regime, rv_p, asset=ASSET)
    bc = sig[sig["buy_type"] == "BUY CALL"].copy()
    bc = bc.join(ohlc[["Open", "High", "Low", "Close"]], how="inner")
    return bc


def main():
    db = _load_kline_db()
    if db is None: return print("kline_db 缺失")
    today = db["date"].max()
    kdb_min = db["date"].min()
    print(f"kline_db: {kdb_min.date()} → {today.date()}")
    print(f"BC config: TP=1.5x premium (+50%), SL=0.5x (-50%), DTE=30, min_dte=14")
    cfg = BCConfig()

    bc = collect_signals()
    bc = bc[(bc.index >= kdb_min) & (bc.index <= today)]
    print(f"\nGLD BUY CALL 信号 (kline_db 时间窗内): {len(bc)} 笔\n")

    rows = []
    skipped = {"no_entry": 0, "open": 0}
    for sig_d, r in bc.iterrows():
        eO, eC = float(r["Open"]), float(r["Close"])
        eH, eL = float(r["High"]), float(r["Low"])
        lc = pick_liquid_monthly_option(ASSET, sig_d, eO, "C",
                                              dte_target=cfg.base_dte, min_dte=14)
        if not lc:
            skipped["no_entry"] += 1
            rows.append({"sig_d": sig_d.date(), "status": "no_entry",
                          "spot_open": round(eO, 2)})
            continue
        entry = interpolate_option_intraday(lc, eO, eC, eO, eH, eL)
        ent = {"legs": [("long_call", lc["code"], lc["strike"], 1)],
                "entry_price": entry,
                "leg_prices": [("long_call", entry)]}
        res = simulate_bc_position(ent, sig_d, today, db, cfg)
        if not res.get("is_closed"):
            skipped["open"] += 1
            rows.append({"sig_d": sig_d.date(), "status": "OPEN",
                          "spot_open": round(eO, 2),
                          "strike": lc["strike"], "expiry": lc["expiry"],
                          "entry_price": round(entry, 2),
                          "cur_value": round(float(res.get("current_value", entry)), 2),
                          "hold_days": int(res.get("hold_days", 0)),
                          "unrealized_pnl_pct": round(float(res.get("pnl_pct", 0)), 1)})
            continue
        pnl = max(-100, min(500, float(res.get("pnl_pct", 0))))
        rows.append({
            "sig_d": sig_d.date(), "status": "CLOSED",
            "spot_open": round(eO, 2),
            "strike": lc["strike"], "expiry": lc["expiry"],
            "entry_price": round(entry, 2),
            "exit_date": (res.get("exit_date").date()
                            if hasattr(res.get("exit_date"), "date") else None),
            "exit_value": round(float(res.get("exit_value", 0)), 2),
            "exit_reason": res.get("exit_reason", ""),
            "hold_days": int(res.get("hold_days", 0)),
            "pnl_pct": round(pnl, 1),
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nskipped: no_entry={skipped['no_entry']}, OPEN={skipped['open']}")

    closed = df[df["status"] == "CLOSED"].copy()
    n = len(closed)
    if n:
        pnls = closed["pnl_pct"].astype(float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        pf = (wins.sum() / abs(losses.sum())) if losses.sum() else float("inf")
        print(f"\n{'=' * 60}")
        print(f"SUMMARY (CLOSED only, n={n})")
        print(f"{'=' * 60}")
        print(f"  WR        : {len(wins) / n * 100:.1f}% ({len(wins)} wins / {len(losses)} losses)")
        print(f"  sum%      : {pnls.sum():+.1f}%")
        print(f"  mean%     : {pnls.mean():+.2f}%")
        print(f"  median%   : {pnls.median():+.2f}%")
        print(f"  max_win%  : {pnls.max():+.1f}%")
        print(f"  max_loss% : {pnls.min():+.1f}%")
        print(f"  PF        : {pf:.2f}")
        print(f"\n  退出原因分布:")
        print(closed.groupby("exit_reason").agg(
            n=("pnl_pct", "size"),
            mean_pnl=("pnl_pct", "mean"),
            sum_pnl=("pnl_pct", "sum"),
        ).to_string())

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nper-trade saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
