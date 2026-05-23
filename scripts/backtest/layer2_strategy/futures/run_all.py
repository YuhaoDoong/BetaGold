"""v3.7.229 Layer 2 期货模块: trailing 1y/6m/3m 多窗.

注: 期货数据是 21y, 1y/6m/3m 是 Layer 2 标准窗口 (最近期).
但期货也跑 5y/10y 看长期对比 (短窗信号样本小, 长窗看 robustness).
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from scripts.backtest.framework import build_raw_universe, trailing_slice
from core.strategy_configs import get_futures_config
from core.strategies.futures_long import simulate_long_position


FUT_CSV = {
    "GLD": "/Users/yhdong/Gold/data/raw/market/gold_futures.csv",
    "SLV": "/Users/yhdong/Gold/data/raw/market/silver_futures.csv",
}

# 期货按 1y/6m/3m 短窗 + 5y/3y 长窗对照 (含最新数据)
WINDOWS = [
    ("5y",  5 * 365),
    ("3y",  3 * 365),
    ("1y",      365),
    ("6m",      180),
    ("3m",       90),
]


def load_futures(asset):
    df = pd.read_csv(FUT_CSV[asset], index_col=0, parse_dates=True)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    return df[["Open", "High", "Low", "Close"]].sort_index()


def backtest_futures(dates, fut_ohlc, cfg):
    today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    rows = []
    for d in dates:
        if d not in fut_ohlc.index: continue
        entry = float(fut_ohlc.loc[d, "Open"])
        sub = fut_ohlc[fut_ohlc.index >= d]
        if not len(sub): continue
        sim = simulate_long_position(
            entry_d=d, entry_spot=entry,
            ohlc=sub, today=today, cfg=cfg,
            live_spot=sub.iloc[-1]["Close"], signal_tier="S")
        rows.append({
            "date": d,
            "closed": bool(sim.get("closed", False)),
            "pnl_pct": max(-100.0, float(sim.get("ret_levered_pct", 0) or 0)),
        })
    return pd.DataFrame(rows)


def score(s):
    if not len(s): return {"n": 0, "scoreB": 0}
    n = len(s); wr = (s > 0).mean()
    return {"n": n, "WR": round(wr*100, 1), "mean": round(s.mean(), 2),
              "sum": round(s.sum(), 1), "max_loss": round(s.min(), 1),
              "blowup_rate": round((s <= -99).mean() * 100, 1),
              "scoreB": round((wr**2) * math.log(1+n) * s.mean(), 2)}


def get_dates(raw, tier):
    bs = raw["buy_signal"].fillna(False).astype(bool)
    st = raw["signal_tier"].fillna("")
    if tier == "ALL": return raw.index[bs].tolist()
    if tier == "S+A": return raw.index[bs & st.isin(["S","A"])].tolist()
    return raw.index[bs & (st == tier)].tolist()


def grid_lev(asset, dates, fut_ohlc):
    """对一组日期 grid leverage."""
    cfg = get_futures_config(asset)
    rows = []
    for lev in [3, 5, 8, 10, 15, 20]:
        old = cfg.leverage
        cfg.leverage = lev
        cfg.tier_s_leverage = lev; cfg.tier_a_leverage = lev; cfg.tier_b_leverage = lev
        try:
            df = backtest_futures(dates, fut_ohlc, cfg)
            closed = df[df["closed"]]
            s = score(closed["pnl_pct"]) if len(closed) else {"n": 0, "scoreB": -1e9}
            s["lev"] = lev
            s["n_total"] = len(df)
            s["n_closed"] = len(closed)
            rows.append(s)
        finally:
            cfg.leverage = old
    return pd.DataFrame(rows)


def main():
    out_dir = Path("/Users/yhdong/Gold/data/backtest_history/v3.7.229_layer2_futures")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for asset in ["GLD", "SLV"]:
        print(f"\n{'='*100}\n资产: {asset} (期货 trailing 5y/3y/1y/6m/3m)\n{'='*100}")
        raw, _ = build_raw_universe(asset)
        fut_ohlc = load_futures(asset)

        for tier in ["S", "A", "S+A", "B", "ALL"]:
            print(f"\n--- tier={tier} ---")
            for win_label, win_days in WINDOWS:
                sub_raw = trailing_slice(raw, win_days)
                dates = get_dates(sub_raw, tier)
                dates = [d for d in dates if d in fut_ohlc.index]
                if len(dates) < 3:
                    continue
                grid = grid_lev(asset, dates, fut_ohlc)
                valid = grid[grid["n"] >= 3]
                if not len(valid): continue
                best = valid.loc[valid["scoreB"].idxmax()]
                print(f"  {win_label} ({len(dates)} signals): "
                      f"best lev={int(best['lev'])} WR={best['WR']}% "
                      f"mean={best['mean']}% sum={best['sum']}% "
                      f"blowup={best['blowup_rate']}% scoreB={best['scoreB']}")
                all_rows.append({"asset": asset, "tier": tier, "window": win_label,
                                  "n_signals": len(dates),
                                  "best_lev": int(best["lev"]),
                                  "n_closed": int(best["n"]),
                                  "WR": best["WR"], "mean": best["mean"],
                                  "sum": best["sum"], "max_loss": best["max_loss"],
                                  "blowup_rate": best["blowup_rate"],
                                  "scoreB": best["scoreB"]})

    df = pd.DataFrame(all_rows)
    print(f"\n\n=== 期货 Layer 2 多窗总览 ===")
    print(df.to_string(index=False))
    df.to_csv(out_dir / "futures_trailing_windows.csv", index=False)

    # 跨窗一致性: 各 (asset, tier) best_lev 一致度
    print(f"\n=== 跨窗 best_lev 一致性 ===")
    consist = df.groupby(["asset", "tier"])["best_lev"].agg(
        lambda x: list(x.values))
    print(consist.to_string())


if __name__ == "__main__":
    main()
