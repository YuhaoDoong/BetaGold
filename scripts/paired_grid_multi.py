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
    cols = ["signal_date","pnl_pct","rv_pctile","gvz_iv_pct","iv_rv_gap_pct",
            "bp_low","bp_close","bp_high","regime","macd_hist","rsi_14","stoch_k"]
    cols = [c for c in cols if c in df.columns]
    bc = df[df["strategy"]=="BUY CALL"][cols].set_index("signal_date")
    sp = df[df["strategy"]=="SELL PUT"][["signal_date","pnl_pct"]].set_index("signal_date")
    rename_map = {"pnl_pct":"bc_pnl","rv_pctile":"rv","gvz_iv_pct":"gvz",
                   "iv_rv_gap_pct":"gap"}
    bc = bc.rename(columns=rename_map)
    sp.columns = ["sp_pnl"]
    paired = bc.join(sp, how="inner").reset_index()
    print(f"配对信号: {len(paired)}")

    print("\n【1. RV%tile】")
    paired_grid(paired, "rv", [(0,0.25),(0.25,0.5),(0.5,0.75),(0.75,1.01)])

    print("\n【2. GVZ IV】")
    paired_grid(paired, "gvz", [(0,18),(18,22),(22,28),(28,100)])

    print("\n【3. IV-RV gap (>0 = IV 高于 RV)】")
    paired_grid(paired, "gap", [(-100,-3),(-3,0),(0,3),(3,8),(8,100)])

    print("\n【4. bp_low (深破程度)】")
    paired_grid(paired, "bp_low", [(-1,0.05),(0.05,0.15),(0.15,0.30),(0.30,2.0)])

    print("\n【5. bp_close (close 在 band 位置 — 区间预测)】")
    paired_grid(paired, "bp_close", [(-1,0.10),(0.10,0.30),(0.30,0.50),(0.50,2.0)])

    if "macd_hist" in paired.columns:
        print("\n【6. MACD hist (>0=多头动能, <0=空头动能)】")
        # 5 个等量分位
        q = paired["macd_hist"].quantile([0,0.2,0.4,0.6,0.8,1.0]).values
        bins = [(q[i], q[i+1]) for i in range(5)]
        paired_grid(paired, "macd_hist", bins,
                     label_fn=lambda lo,hi:f"MACD [{lo:>+5.2f}, {hi:>+5.2f})")

    if "rsi_14" in paired.columns:
        print("\n【7. RSI 14 (>70 超买, <30 超卖)】")
        paired_grid(paired, "rsi_14", [(0,30),(30,45),(45,55),(55,70),(70,100)])

    if "stoch_k" in paired.columns:
        print("\n【8. Stoch %K】")
        paired_grid(paired, "stoch_k", [(0,20),(20,40),(40,60),(60,80),(80,100)])

    print("\n【9. Regime】")
    for reg in ["Bull","Mixed","Bear"]:
        sub = paired[paired["regime"]==reg]
        if not len(sub): continue
        diff = (sub["sp_pnl"]-sub["bc_pnl"]).mean()
        bc_wr = (sub["bc_pnl"]>0).mean()*100
        sp_wr = (sub["sp_pnl"]>0).mean()*100
        winner = "SP" if diff>0 else "BC"
        print(f"  {reg:<10}n={len(sub):>3} SP-BC {diff:>+7.1f}%  BC wr {bc_wr:.0f}%  SP wr {sp_wr:.0f}%  ★ {winner}")
