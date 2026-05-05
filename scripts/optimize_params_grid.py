"""基于全历史回测 CSV 找最优参数 (Step 3 of 用户 plan).

输入: data/backtest_history/backtest_<asset>_<date>.csv (来自 full_history_backtest.py)

变量网格:
  RV%tile 切点 (BUY CALL ↔ SELL PUT)
  STRADDLE 最低 RV%tile threshold (or score)
  策略组合: 哪些策略保留
  Regime filter (Bull/Bear/Mixed)

目标函数: 综合分 = 累计 P&L × √n × win_rate

输出: 最优参数对应表 + per-asset 推荐
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/Users/yhdong/GoldDash")
import pandas as pd
import numpy as np
from pathlib import Path

CSV_DIR = Path("/Users/yhdong/Gold/data/backtest_history")


def score(sub: pd.DataFrame) -> float:
    """综合分: 累计 × √n × win_rate, 鼓励大 n + 高 win + 高累计."""
    if len(sub) == 0: return -1e9
    wr = (sub["pnl_pct"] > 0).mean()
    cum = sub["pnl_pct"].sum()
    return cum * np.sqrt(len(sub)) * wr / 100


def grid_for_asset(asset: str, today_str: str):
    csv = CSV_DIR / f"backtest_{asset.lower()}_{today_str}.csv"
    if not csv.exists():
        print(f"找不到 {csv}"); return None
    df = pd.read_csv(csv, parse_dates=["signal_date", "exit_date"])
    print(f"\n=== {asset} ({len(df)} 笔) ===")

    # 1. 各策略全期表现
    print("\n【1. 全期 per-strategy】")
    print(f'{"strategy":<14} {"n":>5} {"win%":>7} {"avg":>8} {"cum":>10} {"score":>10}')
    for strat, sub in df.groupby("strategy"):
        wr = (sub["pnl_pct"] > 0).mean() * 100
        avg = sub["pnl_pct"].mean()
        cum = sub["pnl_pct"].sum()
        sc = score(sub)
        print(f'  {strat:<12} {len(sub):>5} {wr:>6.1f}% {avg:>+7.2f}% {cum:>+9.0f}% {sc:>9.1f}')

    # 2. RV %tile 切点 grid (BUY CALL vs SELL PUT)
    print("\n【2. RV%tile 切点 grid (低于切点 = BUY CALL, 高于切点 = SELL PUT)】")
    print(f'{"切点":>7} {"BC n":>5} {"BC win":>7} {"BC cum":>8} {"SP n":>5} {"SP win":>7} {"SP cum":>9}')
    for cut in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8]:
        bc = df[(df["strategy"] == "BUY CALL") & (df["rv_pctile"] < cut)]
        sp = df[(df["strategy"] == "SELL PUT") & (df["rv_pctile"] >= cut)]
        bc_wr = (bc["pnl_pct"] > 0).mean() * 100 if len(bc) else 0
        sp_wr = (sp["pnl_pct"] > 0).mean() * 100 if len(sp) else 0
        bc_cum = bc["pnl_pct"].sum()
        sp_cum = sp["pnl_pct"].sum()
        total = bc_cum + sp_cum
        print(f'  {cut:>5.2f}  {len(bc):>5} {bc_wr:>6.1f}% {bc_cum:>+7.0f}%  {len(sp):>5} {sp_wr:>6.1f}% {sp_cum:>+8.0f}% combined={total:+.0f}%')

    # 3. STRADDLE 全期最优
    strad = df[df["strategy"] == "STRADDLE"]
    if len(strad):
        wr = (strad["pnl_pct"] > 0).mean() * 100
        cum = strad["pnl_pct"].sum()
        avg = strad["pnl_pct"].mean()
        print(f"\n【3. STRADDLE 全期 (event-mode)】 n={len(strad)} wr={wr:.1f}% avg={avg:+.2f}% cum={cum:+.0f}%")
        # 按 regime
        print(f"  按 regime:")
        for reg, ssub in strad.groupby("regime"):
            wr_r = (ssub["pnl_pct"] > 0).mean() * 100
            cum_r = ssub["pnl_pct"].sum()
            print(f"    {reg}: n={len(ssub)} wr={wr_r:.1f}% cum={cum_r:+.0f}%")

    # 4. 策略组合 (推荐 portfolio)
    print("\n【4. 推荐 portfolio】")
    portfolios = {
        "STRADDLE only": df[df["strategy"] == "STRADDLE"],
        "STRADDLE + BUY CALL (低 RV)": pd.concat([
            df[df["strategy"] == "STRADDLE"],
            df[(df["strategy"] == "BUY CALL") & (df["rv_pctile"] < 0.45)]
        ]),
        "STRADDLE + FUTURES": pd.concat([
            df[df["strategy"] == "STRADDLE"],
            df[df["strategy"] == "FUTURES_LONG"]
        ]),
        "all positive (排除 SELL PUT/SHORT_VOL)": df[
            df["strategy"].isin(["STRADDLE", "BUY CALL", "FUTURES_LONG"])],
        "current setup (5 全开)": df,
    }
    for name, sub in portfolios.items():
        if len(sub) == 0: continue
        wr = (sub["pnl_pct"] > 0).mean() * 100
        cum = sub["pnl_pct"].sum()
        avg = sub["pnl_pct"].mean()
        sc = score(sub)
        print(f'  {name:<40} n={len(sub):<5} wr={wr:>5.1f}% cum={cum:>+7.0f}% score={sc:>7.0f}')

    # 5. 时段稳定性 (是否近期不同)
    print("\n【5. 时段稳定性 (最近 vs 历史)】")
    for stage, sub in df.groupby("stage"):
        if len(sub) == 0: continue
        wr = (sub["pnl_pct"] > 0).mean() * 100
        cum = sub["pnl_pct"].sum()
        print(f'  {stage:<22} n={len(sub):>5} wr={wr:>5.1f}% cum={cum:>+7.0f}%')


if __name__ == "__main__":
    today_str = pd.Timestamp.now().strftime("%Y%m%d")
    for asset in ["GLD", "SLV"]:
        grid_for_asset(asset, today_str)
