"""sp_score 系统 paired 验证 — 用现有回测 CSV 模拟"如果按 sp_score 选 BC vs SP, 对比单切 RV"的实际效果.

读 backtest_{gld,slv}_20260505.csv (已含 macd_hist/rsi_14/stoch_k/bp_close/bp_low/iv_rv_gap_pct/gvz_iv_pct).
对每个 signal_date, 计算 sp_score, 看推荐 (BC/SP) 与实际配对中赢家是否一致.

对比:
  A. 单切 RV 决策 (现状): rv_pctile 与 cfg.rv_filter_low 比, 决定 BC 或 SP
  B. sp_score 决策 (新): score >= threshold → SP, 否则 BC
  C. 完美决策: 已知谁赢, 直接选赢家 (上限基线)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, "/Users/yhdong/GoldDash")
from core.strategy_config import get_config

CSV = Path("/Users/yhdong/Gold/data/backtest_history")


def compute_score(row, cfg):
    score = 0.0
    if row["iv_rv_gap_pct"] > 0:           score += cfg.sp_score_w_iv_rv_gap
    if row["bp_low"] < 0.05:               score += cfg.sp_score_w_bp_low_deep
    if row["bp_close"] < 0.30:             score += cfg.sp_score_w_bp_close_low
    if row["gvz_iv_pct"] >= 28:            score += cfg.sp_score_w_gvz_high
    if row["rsi_14"] < 30:                 score += cfg.sp_score_w_rsi_oversold
    if row["stoch_k"] < 40:                score += cfg.sp_score_w_stoch_low
    if row["macd_hist"] < -0.5:            score += cfg.sp_score_w_macd_bear
    return score


for asset in ["GLD", "SLV"]:
    cfg = get_config(asset)
    print(f"\n{'='*70}\n{asset} sp_score 验证 (threshold={cfg.sp_score_threshold})\n{'='*70}")
    csv = CSV / f"backtest_{asset.lower()}_20260506.csv"
    df = pd.read_csv(csv, parse_dates=["signal_date","exit_date"])
    df = df[df["stage"].isin(["stage2_main_3m", "stage2_leaps_aux"])]
    bc = df[df["strategy"]=="BUY CALL"].set_index("signal_date")
    sp = df[df["strategy"]=="SELL PUT"][["signal_date","pnl_pct"]].set_index("signal_date")
    sp.columns = ["sp_pnl"]
    paired = bc.join(sp, how="inner").reset_index()
    print(f"配对信号: {len(paired)}")
    if not len(paired): continue

    # 计算 score
    paired["score"] = paired.apply(lambda r: compute_score(r, cfg), axis=1)
    paired["score_pick"] = np.where(paired["score"] >= cfg.sp_score_threshold,
                                       "SP", "BC")
    paired["rv_pick"] = np.where(paired["rv_pctile"] < cfg.rv_filter_low,
                                    "BC", "SP")
    paired["winner"] = np.where(paired["sp_pnl"] > paired["pnl_pct"], "SP", "BC")
    paired["score_chosen_pnl"] = np.where(paired["score_pick"]=="SP",
                                              paired["sp_pnl"], paired["pnl_pct"])
    paired["rv_chosen_pnl"] = np.where(paired["rv_pick"]=="SP",
                                          paired["sp_pnl"], paired["pnl_pct"])
    paired["best_pnl"] = paired[["pnl_pct","sp_pnl"]].max(axis=1)

    # 准确率: pick == winner
    score_acc = (paired["score_pick"] == paired["winner"]).mean() * 100
    rv_acc = (paired["rv_pick"] == paired["winner"]).mean() * 100
    print(f"\n准确率 (pick == 实际赢家):")
    print(f"  RV 单切:      {rv_acc:.1f}%")
    print(f"  sp_score:     {score_acc:.1f}%")

    print(f"\n胜率 (chosen_pnl > 0):")
    print(f"  RV 单切:      {(paired['rv_chosen_pnl']>0).mean()*100:.1f}%")
    print(f"  sp_score:     {(paired['score_chosen_pnl']>0).mean()*100:.1f}%")
    print(f"  完美 (上限):  {(paired['best_pnl']>0).mean()*100:.1f}%")

    print(f"\n累计 PnL %:")
    print(f"  RV 单切:      {paired['rv_chosen_pnl'].sum():+.1f}%  mean {paired['rv_chosen_pnl'].mean():+.2f}%")
    print(f"  sp_score:     {paired['score_chosen_pnl'].sum():+.1f}%  mean {paired['score_chosen_pnl'].mean():+.2f}%")
    print(f"  完美 (上限):  {paired['best_pnl'].sum():+.1f}%  mean {paired['best_pnl'].mean():+.2f}%")

    # score 阈值灵敏度扫描
    print(f"\n阈值灵敏度 (找最优):")
    print(f"  {'thr':>5}{'n_SP':>6}{'wr':>7}{'sum':>9}{'mean':>9}{'acc':>7}")
    for thr in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        pick = np.where(paired["score"] >= thr, "SP", "BC")
        chosen = np.where(pick=="SP", paired["sp_pnl"], paired["pnl_pct"])
        n_sp = (pick=="SP").sum()
        wr = (chosen>0).mean()*100
        s = chosen.sum()
        m = chosen.mean()
        acc = (pick == paired["winner"]).mean()*100
        print(f"  {thr:>5.1f}{n_sp:>6}{wr:>6.1f}%{s:>+8.1f}%{m:>+8.2f}%{acc:>6.1f}%")
