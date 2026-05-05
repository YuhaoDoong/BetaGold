"""BUY CALL vs SELL PUT 切点 grid search.

输入: backtest_<asset>_<date>.csv (含 BC + SP 双模拟 per signal)
变量: RV %tile 切点 (低于切点 用 BC, 高于 用 SP)
其他维度: regime / IV (GVZ 历史) / score / ATR 收缩

输出: 最优切点 per asset + 综合 portfolio (BC+SP 二选 + STRADDLE 始终 + FUTURES)
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/Users/yhdong/GoldDash")
import pandas as pd
import numpy as np
from pathlib import Path

CSV_DIR = Path("/Users/yhdong/Gold/data/backtest_history")


def grid_for_asset(asset: str, today_str: str):
    csv = CSV_DIR / f"backtest_{asset.lower()}_{today_str}.csv"
    if not csv.exists():
        print(f"找不到 {csv}"); return
    df = pd.read_csv(csv, parse_dates=["signal_date", "exit_date"])
    print(f"\n{'='*70}\n{asset} BC vs SP 切点 grid ({len(df)} 总记录)\n{'='*70}")

    # 拆 BC / SP
    bc = df[df["strategy"] == "BUY CALL"].copy()
    sp = df[df["strategy"] == "SELL PUT"].copy()
    if not len(bc) or not len(sp):
        print("BC 或 SP 数据缺失"); return
    print(f"\nBC 总: n={len(bc)} wr={(bc['pnl_pct']>0).mean()*100:.1f}% "
          f"cum={bc['pnl_pct'].sum():+.0f}%")
    print(f"SP 总: n={len(sp)} wr={(sp['pnl_pct']>0).mean()*100:.1f}% "
          f"cum={sp['pnl_pct'].sum():+.0f}%")

    # 配对 (同 signal_date 的 BC + SP 应同源)
    bc_idx = bc.set_index("signal_date")
    sp_idx = sp.set_index("signal_date")
    common = bc_idx.index.intersection(sp_idx.index)
    print(f"\n配对信号: {len(common)} 天 (BC+SP 同期)")

    # === RV%tile 切点 grid ===
    print("\n【RV%tile 切点】(低于切点 用 BC, 高于切点 用 SP)")
    print(f'{"切点":>6} {"BC n":>6} {"BC win":>8} {"BC cum":>10} '
          f'{"SP n":>6} {"SP win":>8} {"SP cum":>10} {"合计":>10}')
    best = (-1e9, None)
    for cut in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        bc_use = bc[bc["rv_pctile"] < cut]
        sp_use = sp[sp["rv_pctile"] >= cut]
        if not len(bc_use) or not len(sp_use): continue
        bc_wr = (bc_use["pnl_pct"] > 0).mean() * 100
        sp_wr = (sp_use["pnl_pct"] > 0).mean() * 100
        bc_cum = bc_use["pnl_pct"].sum()
        sp_cum = sp_use["pnl_pct"].sum()
        total = bc_cum + sp_cum
        print(f'  {cut:>4.2f}  {len(bc_use):>6} {bc_wr:>7.1f}% {bc_cum:>+9.0f}%  '
              f'{len(sp_use):>6} {sp_wr:>7.1f}% {sp_cum:>+9.0f}%  {total:>+9.0f}%')
        if total > best[0]:
            best = (total, cut)
    print(f"\n  ★ 最优切点: {best[1]:.2f} (合计 {best[0]:+.0f}%)")

    # === Regime 分段 ===
    print("\n【Regime × Strategy】(每 regime 推荐用哪个)")
    for reg in ["Bull", "Mixed", "Bear"]:
        bc_r = bc[bc["regime"] == reg]
        sp_r = sp[sp["regime"] == reg]
        if not len(bc_r) and not len(sp_r): continue
        bc_wr = (bc_r["pnl_pct"] > 0).mean() * 100 if len(bc_r) else 0
        sp_wr = (sp_r["pnl_pct"] > 0).mean() * 100 if len(sp_r) else 0
        bc_cum = bc_r["pnl_pct"].sum() if len(bc_r) else 0
        sp_cum = sp_r["pnl_pct"].sum() if len(sp_r) else 0
        winner = "BC" if bc_cum > sp_cum else "SP"
        print(f'  {reg:<6} BC: n={len(bc_r):>4} wr={bc_wr:>5.1f}% cum={bc_cum:>+7.0f}% | '
              f'SP: n={len(sp_r):>4} wr={sp_wr:>5.1f}% cum={sp_cum:>+7.0f}%  ★ {winner}')

    # === Stage 分段 ===
    print("\n【Stage × Strategy】(stage2 LEAPS 最可信)")
    for stage in df["stage"].unique():
        bc_s = bc[bc["stage"] == stage]
        sp_s = sp[sp["stage"] == stage]
        bc_wr = (bc_s["pnl_pct"] > 0).mean() * 100 if len(bc_s) else 0
        sp_wr = (sp_s["pnl_pct"] > 0).mean() * 100 if len(sp_s) else 0
        bc_cum = bc_s["pnl_pct"].sum() if len(bc_s) else 0
        sp_cum = sp_s["pnl_pct"].sum() if len(sp_s) else 0
        print(f'  {stage:<22} BC: n={len(bc_s):>4} wr={bc_wr:>5.1f}% cum={bc_cum:>+7.0f}% | '
              f'SP: n={len(sp_s):>4} wr={sp_wr:>5.1f}% cum={sp_cum:>+7.0f}%')

    # === 推荐 portfolio ===
    print("\n【推荐 portfolio】")
    portfolios = {
        "全 BC (无 SP 切换)": bc,
        "全 SP (无 BC)": sp,
        f"切点 {best[1]:.2f} BC<RV<SP": pd.concat([
            bc[bc["rv_pctile"] < best[1]], sp[sp["rv_pctile"] >= best[1]]]),
        "STRADDLE only": df[df["strategy"] == "STRADDLE"],
        "BC + STRADDLE + FUTURES": df[df["strategy"].isin(
            ["BUY CALL", "STRADDLE", "FUTURES_LONG"])],
        f"切点{best[1]:.2f} + STRADDLE + FUTURES": pd.concat([
            bc[bc["rv_pctile"] < best[1]],
            sp[sp["rv_pctile"] >= best[1]],
            df[df["strategy"] == "STRADDLE"],
            df[df["strategy"] == "FUTURES_LONG"]]),
    }
    print(f'{"portfolio":<45} {"n":>6} {"win":>7} {"cum":>10}')
    for name, sub in portfolios.items():
        if not len(sub): continue
        wr = (sub["pnl_pct"] > 0).mean() * 100
        cum = sub["pnl_pct"].sum()
        print(f'  {name:<43} {len(sub):>6} {wr:>6.1f}% {cum:>+9.0f}%')


if __name__ == "__main__":
    today_str = pd.Timestamp.now().strftime("%Y%m%d")
    for asset in ["GLD", "SLV"]:
        grid_for_asset(asset, today_str)
