"""v3.7.200 grid: 单腿 BC vs Bull Call Spread (apples-to-apples)

方法学跟 backtest_pipeline 一致:
  - 入场: price_strategy_at → pick_liquid_monthly_option + interpolate_option_intraday
  - 退出: simulate_bc_position (TP=1.5x, SL=0.5x, expiry)
  - 只统计 is_closed=True 的笔 (剔除 OPEN MTM)

三档:
  A_single: 当前默认 (单腿 ATM long call)
  B_+5%:    bull call spread (long ATM / short ATM×1.05)
  C_+10%:   bull call spread (long ATM / short ATM×1.10)

数据源: kline_db (真实期权 OHLC, 1y).
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


def make_bc_entry(asset: str, sig_d: pd.Timestamp,
                    spot_at_trigger: float, eO: float, eC: float,
                    eH: float, eL: float,
                    spread_pct: float = 0.0,
                    dte_target: int = 30) -> dict | None:
    """构造 BC entry. spread_pct=0 → 单腿; >0 → bull call spread."""
    lc = pick_liquid_monthly_option(asset, sig_d, spot_at_trigger, "C",
                                       dte_target=dte_target, min_dte=14)
    if not lc: return None
    lc_intra = interpolate_option_intraday(lc, eO, eC, spot_at_trigger, eH, eL)
    if spread_pct <= 0:
        return {
            "legs": [("long_call", lc["code"], lc["strike"], 1)],
            "entry_price": lc_intra,
            "leg_prices": [("long_call", lc_intra)],
            "source": f"C${lc['strike']:.0f}",
        }
    short_target = round(lc["strike"] * (1 + spread_pct))
    sc = pick_liquid_monthly_option(asset, sig_d, short_target, "C",
                                        dte_target=dte_target, min_dte=14)
    if not sc or sc["strike"] <= lc["strike"]:
        return None
    sc_intra = interpolate_option_intraday(sc, eO, eC, spot_at_trigger, eH, eL)
    debit = lc_intra - sc_intra
    if debit <= 0.01: return None
    return {
        "legs": [("long_call", lc["code"], lc["strike"], 1),
                  ("short_call", sc["code"], sc["strike"], -1)],
        "entry_price": debit,
        "leg_prices": [("long_call", lc_intra), ("short_call", sc_intra)],
        "source": f"+C${lc['strike']:.0f}/-C${sc['strike']:.0f}",
    }


def collect_bc_signals(asset: str, lookback_days: int) -> pd.DataFrame:
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
    cutoff = ohlc.index.max() - pd.Timedelta(days=lookback_days)
    bc = bc[bc.index >= cutoff]
    bc = bc.join(ohlc[["Open", "High", "Low", "Close"]], how="inner")
    return bc


def summarize(results: list[dict], label: str) -> dict:
    pnls = [r["pnl_pct"] for r in results]
    if not pnls: return {"variant": label, "n": 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else float("inf")
    return {
        "variant": label, "n": len(results),
        "WR%": round(len(wins) / len(pnls) * 100, 1),
        "sum%": round(sum(pnls), 1),
        "mean%": round(sum(pnls) / len(pnls), 2),
        "max_win%": round(max(pnls), 1),
        "max_loss%": round(min(pnls), 1),
        "PF": round(pf, 2),
    }


def main():
    db = _load_kline_db()
    if db is None: return print("kline_db 不存在")
    today = db["date"].max()
    print(f"kline_db: {db['date'].min().date()} → {today.date()}")
    cfg = BCConfig()
    rows = []
    for asset in ["GLD", "SLV"]:
        for lookback in [90, 365]:
            bc = collect_bc_signals(asset, lookback)
            if not len(bc): continue
            results = {"A_single": [], "B_+5%": [], "C_+10%": []}
            variants_map = {"A_single": 0.0, "B_+5%": 0.05, "C_+10%": 0.10}
            for sig_d, row in bc.iterrows():
                eO = float(row["Open"]); eC = float(row["Close"])
                eH = float(row["High"]); eL = float(row["Low"])
                for vk, sp in variants_map.items():
                    ent = make_bc_entry(asset, sig_d, eO, eO, eC, eH, eL,
                                              spread_pct=sp, dte_target=cfg.base_dte)
                    if not ent: continue
                    res = simulate_bc_position(ent, sig_d, today, db, cfg)
                    # apples-to-apples with archive: 只算 is_closed=True
                    if not res.get("is_closed"): continue
                    pnl = max(-100.0, min(500.0, float(res.get("pnl_pct", 0))))
                    results[vk].append({"sig_d": sig_d, "pnl_pct": pnl,
                                          "exit_reason": res.get("exit_reason","")})
            for vk in variants_map:
                s = summarize(results[vk], f"{asset}_{vk}_{lookback}d")
                rows.append(s); print(f"  {s}")
            print()
    df = pd.DataFrame(rows)
    print("=" * 80)
    print(df.to_string(index=False))
    out = Path("/Users/yhdong/Gold/data/backtest_history/bc_single_vs_spread_grid.csv")
    df.to_csv(out, index=False); print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
