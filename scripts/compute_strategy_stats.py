"""v3.7.181 策略实战统计 → strategy_stats.json (Dashboard 读).

每个 (asset, strategy) 计算近 1y real:
  - n (closed 笔数)
  - WR (胜率)
  - avg pnl%
  - Half Kelly (推荐仓位 % 资金)
  - Quarter Kelly (保守仓位)

输出: /Users/yhdong/Gold/data/strategy_stats.json
Dashboard 加载后, 每个信号推荐里展示"近1y WR + 推荐仓位".
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np


PIPE_S2 = Path("/Users/yhdong/Gold/data/backtest_pipeline/stage2_simulated")
HIST_CSV = Path("/Users/yhdong/Gold/data/backtest_history")
OUT = Path("/Users/yhdong/Gold/data/strategy_stats.json")


def kelly(p: float, W_dec: float, L_dec: float) -> float:
    if W_dec <= 0 or L_dec <= 0: return 0.0
    return max(0.0, p / L_dec - (1 - p) / W_dec)


def stats_from_pnls(pnls: pd.Series) -> dict:
    pnls = pd.Series(pnls).dropna()
    n = len(pnls)
    if n < 3: return {"n": n, "wr_pct": None, "avg_pct": None,
                      "half_kelly_pct": None, "qtr_kelly_pct": None,
                      "EV_pct": None}
    wins = pnls[pnls > 0]; losses = pnls[pnls <= 0]
    p = len(wins) / n
    W_dec = (wins.mean() / 100) if len(wins) else 0
    L_dec = (abs(losses.mean()) / 100) if len(losses) else 0.01
    f = kelly(p, W_dec, L_dec)
    half_k = min(25.0, f * 50)        # 单策略 cap 25%
    qtr_k = min(15.0, f * 25)         # 保守 cap 15%
    return {
        "n": n,
        "wr_pct": round(p * 100, 1),
        "avg_pct": round(pnls.mean(), 2),
        "EV_pct": round(p * (W_dec * 100) - (1 - p) * (L_dec * 100), 2),
        "half_kelly_pct": round(half_k, 1),
        "qtr_kelly_pct": round(qtr_k, 1),
    }


def load_real_pnls(asset: str, strat: str) -> pd.Series:
    """近 1y real (kline_db EOD options 或 Binance perp)."""
    src_pat = "binance" if strat == "FUTURES_LONG" else "klinedb"
    files = list(PIPE_S2.glob(f"pnl_{asset.lower()}_{strat}_real_{src_pat}_*.parquet"))
    if not files: return pd.Series(dtype=float)
    df = pd.concat([pd.read_parquet(f) for f in files])
    return df["pnl_pct"].dropna()


def load_5y_pnls(asset: str, strat: str) -> pd.Series:
    """10y full history (含 sim, 仅作 fallback / 参考)."""
    files = sorted(HIST_CSV.glob(f"backtest_{asset.lower()}_*.csv"))
    if not files: return pd.Series(dtype=float)
    df = pd.read_csv(files[-1])
    return df[df["strategy"] == strat]["pnl_pct"].dropna()


def main():
    out = {"_meta": {
        "generated_at": pd.Timestamp.now().isoformat(),
        "data_window_real": "1y kline_db + Binance perp",
        "data_window_5y": "10y backtest_history (含 LEAPS BS proxy)",
        "kelly_caps": {"half_kelly": 25.0, "qtr_kelly": 15.0},
    }}
    for asset in ["GLD", "SLV"]:
        out[asset] = {}
        for strat in ["BUY CALL", "SELL PUT", "STRADDLE", "FUTURES_LONG"]:
            r1y = stats_from_pnls(load_real_pnls(asset, strat))
            r5y = stats_from_pnls(load_5y_pnls(asset, strat))
            out[asset][strat] = {
                "real_1y": r1y,
                "all_history": r5y,
                # 推荐仓位 — 优先 1y 真实, 不足 fallback 5y
                "recommended_half_kelly": (r1y.get("half_kelly_pct")
                                              if r1y.get("n", 0) >= 5
                                              else r5y.get("half_kelly_pct")),
                "recommended_qtr_kelly": (r1y.get("qtr_kelly_pct")
                                              if r1y.get("n", 0) >= 5
                                              else r5y.get("qtr_kelly_pct")),
            }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"[stats] saved → {OUT}")
    # 打印
    for asset in ["GLD", "SLV"]:
        print(f"\n{asset}:")
        print(f"  {'strategy':<14}{'1y_n':>6}{'1y_wr':>8}{'1y_avg':>9}"
              f"{'rec ½K':>9}{'rec ¼K':>9}")
        for strat in ["BUY CALL", "SELL PUT", "STRADDLE", "FUTURES_LONG"]:
            d = out[asset][strat]
            r = d["real_1y"]
            wr = f"{r['wr_pct']}%" if r['wr_pct'] is not None else "—"
            avg = f"{r['avg_pct']:+}%" if r['avg_pct'] is not None else "—"
            hk = f"{d['recommended_half_kelly']}%" if d['recommended_half_kelly'] is not None else "—"
            qk = f"{d['recommended_qtr_kelly']}%" if d['recommended_qtr_kelly'] is not None else "—"
            print(f"  {strat:<14}{r['n']:>6}{wr:>8}{avg:>9}{hk:>9}{qk:>9}")


if __name__ == "__main__":
    main()
