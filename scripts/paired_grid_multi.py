"""多维 paired BC vs SP grid: 找哪些条件下显著选 BC 还是 SP.

维度:
  1. regime (Bull/Mixed/Bear)
  2. RV%tile (4 区间)
  3. GVZ IV (4 区间)
  4. IV-RV gap (5 区间)
  5. bp_low 深破 (3 区间)
  6. 信号前 5d spot 趋势 (前期涨/跌)
  7. 信号前 10d spot 趋势
  8. 信号日 spot 距 entry 偏离 ((spot - close) / close)

输出: 各维度下 paired 比较 (SP-BC pnl 差均值) + 推荐切换条件.
"""
import pandas as pd
import numpy as np
from pathlib import Path

CSV = Path("/Users/yhdong/Gold/data/backtest_history")


def add_pre_trend(df, asset_csv):
    """加 pre_5d_ret / pre_10d_ret 列 (信号前 5/10 天 spot return)."""
    asset_csv = asset_csv.copy()
    df = df.copy()
    pre_5d = []; pre_10d = []
    for d in df["signal_date"]:
        prior = asset_csv.loc[asset_csv.index < d]
        if len(prior) >= 10:
            spot = prior["Close"].iloc[-1]
            spot_5 = prior["Close"].iloc[-5]
            spot_10 = prior["Close"].iloc[-10]
            pre_5d.append((spot / spot_5 - 1) * 100)
            pre_10d.append((spot / spot_10 - 1) * 100)
        else:
            pre_5d.append(np.nan); pre_10d.append(np.nan)
    df["pre_5d_ret"] = pre_5d
    df["pre_10d_ret"] = pre_10d
    return df


def paired_grid(paired, dim_col, bins, label_fn=None):
    """对单一维度做 paired grid."""
    print(f"\n  {'区间':<25}{'n':>4}{'SP-BC mean':>12}{'BC wr':>8}{'SP wr':>8}{'优胜':>8}")
    for lo, hi in bins:
        sub = paired[(paired[dim_col] >= lo) & (paired[dim_col] < hi)]
        if not len(sub): continue
        diff = (sub["sp_pnl"] - sub["bc_pnl"]).mean()
        bc_wr = (sub["bc_pnl"] > 0).mean() * 100
        sp_wr = (sub["sp_pnl"] > 0).mean() * 100
        winner = "SP" if diff > 0 else "BC"
        label = label_fn(lo, hi) if label_fn else f"[{lo:>5.2f}, {hi:>5.2f})"
        print(f"  {label:<25}{len(sub):>4}{diff:>+11.1f}%{bc_wr:>7.0f}%{sp_wr:>7.0f}%   {winner:>4}")


for asset in ["GLD", "SLV"]:
    print(f"\n{'='*70}\n{asset} 多维 Paired BC vs SP\n{'='*70}")
    csv = CSV / f"backtest_{asset.lower()}_20260505.csv"
    df = pd.read_csv(csv, parse_dates=["signal_date","exit_date"])
    df = df[df["stage"].isin(["stage2_main_3m", "stage2_leaps_aux"])]
    bc = df[df["strategy"]=="BUY CALL"][["signal_date","pnl_pct","rv_pctile","gvz_iv_pct",
                                            "iv_rv_gap_pct","bp_low","regime"]].set_index("signal_date")
    sp = df[df["strategy"]=="SELL PUT"][["signal_date","pnl_pct"]].set_index("signal_date")
    bc.columns = ["bc_pnl","rv","gvz","gap","bp_low","regime"]
    sp.columns = ["sp_pnl"]
    paired = bc.join(sp, how="inner").reset_index()
    print(f"配对信号: {len(paired)}")

    # 加前期趋势
    asset_csv = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                              index_col=0, parse_dates=True)
    paired = add_pre_trend(paired, asset_csv)

    print("\n【1. RV%tile】")
    paired_grid(paired, "rv", [(0,0.25),(0.25,0.5),(0.5,0.75),(0.75,1.01)])

    print("\n【2. GVZ IV】")
    paired_grid(paired, "gvz", [(0,18),(18,22),(22,28),(28,100)])

    print("\n【3. IV-RV gap】")
    paired_grid(paired, "gap", [(-100,-3),(-3,0),(0,3),(3,8),(8,100)])

    print("\n【4. bp_low (深破)】")
    paired_grid(paired, "bp_low", [(-1,0.05),(0.05,0.15),(0.15,0.30),(0.30,2.0)])

    print("\n【5. 前 5d 趋势】(>0=涨势, <0=跌势)")
    paired_grid(paired, "pre_5d_ret", [(-100,-2),(-2,-0.5),(-0.5,0.5),(0.5,2),(2,100)])

    print("\n【6. 前 10d 趋势】")
    paired_grid(paired, "pre_10d_ret", [(-100,-3),(-3,-1),(-1,1),(1,3),(3,100)])

    print("\n【7. Regime】")
    for reg in ["Bull","Mixed","Bear"]:
        sub = paired[paired["regime"]==reg]
        if not len(sub): continue
        diff = (sub["sp_pnl"]-sub["bc_pnl"]).mean()
        bc_wr = (sub["bc_pnl"]>0).mean()*100
        sp_wr = (sub["sp_pnl"]>0).mean()*100
        winner = "SP" if diff>0 else "BC"
        print(f"  {reg:<10}n={len(sub):>3} SP-BC {diff:>+7.1f}%  BC wr {bc_wr:.0f}%  SP wr {sp_wr:.0f}%  ★ {winner}")
