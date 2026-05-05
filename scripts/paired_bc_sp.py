"""Paired BC vs SP 比较 — 同信号日, 哪个赢 (不是跨时段平均).

用户诉求: SP 横盘+上涨赢, BC 仅上涨赢 → SP wr 应高.
方法: 同 signal_date 对 BC pnl vs SP pnl 一对一比较.

输出:
- BC > SP 笔数 (BC 赢 SP)
- SP > BC 笔数 (SP 赢 BC)
- BC > 0 wr (BC 单独胜率)
- SP > 0 wr (SP 单独胜率)
- BC + SP 都 win 笔数
"""
import pandas as pd
from pathlib import Path
CSV = Path("/Users/yhdong/Gold/data/backtest_history")

for asset in ["GLD", "SLV"]:
    df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260505.csv",
                       parse_dates=["signal_date","exit_date"])
    df = df[df["stage"] == "stage2_options_real"]
    bc = df[df["strategy"] == "BUY CALL"][["signal_date", "pnl_pct", "rv_pctile",
                                              "gvz_iv_pct"]].set_index("signal_date")
    sp = df[df["strategy"] == "SELL PUT"][["signal_date", "pnl_pct"]].set_index("signal_date")
    bc.columns = ["bc_pnl", "rv", "gvz"]
    sp.columns = ["sp_pnl"]
    paired = bc.join(sp, how="inner").reset_index()
    n = len(paired)
    if not n:
        print(f"\n{asset}: 无 BC+SP 配对"); continue

    # 统计
    bc_only_win = (paired["bc_pnl"] > 0).sum()
    sp_only_win = (paired["sp_pnl"] > 0).sum()
    bc_beats_sp = (paired["bc_pnl"] > paired["sp_pnl"]).sum()
    sp_beats_bc = (paired["sp_pnl"] > paired["bc_pnl"]).sum()
    both_win = ((paired["bc_pnl"] > 0) & (paired["sp_pnl"] > 0)).sum()
    both_lose = ((paired["bc_pnl"] < 0) & (paired["sp_pnl"] < 0)).sum()
    bc_win_sp_lose = ((paired["bc_pnl"] > 0) & (paired["sp_pnl"] < 0)).sum()
    sp_win_bc_lose = ((paired["sp_pnl"] > 0) & (paired["bc_pnl"] < 0)).sum()

    print(f"\n{'='*60}")
    print(f"{asset} BC vs SP Paired ({n} 配对信号)")
    print(f"{'='*60}")
    print(f"\n个体胜率 (单策略 P&L > 0):")
    print(f"  BC win:    {bc_only_win}/{n} ({bc_only_win/n*100:.1f}%)")
    print(f"  SP win:    {sp_only_win}/{n} ({sp_only_win/n*100:.1f}%)")

    print(f"\nPaired 比较:")
    print(f"  BC > SP (这信号下 BC 单笔更赚): {bc_beats_sp}/{n} ({bc_beats_sp/n*100:.1f}%)")
    print(f"  SP > BC (这信号下 SP 单笔更赚): {sp_beats_bc}/{n} ({sp_beats_bc/n*100:.1f}%)")

    print(f"\n四象限:")
    print(f"  BC win + SP win  (都赢): {both_win}/{n} ({both_win/n*100:.1f}%)")
    print(f"  BC win + SP lose:        {bc_win_sp_lose}/{n} ({bc_win_sp_lose/n*100:.1f}%)")
    print(f"  SP win + BC lose:        {sp_win_bc_lose}/{n} ({sp_win_bc_lose/n*100:.1f}%)")
    print(f"  BC lose + SP lose (都输): {both_lose}/{n} ({both_lose/n*100:.1f}%)")

    print(f"\n累计 P&L:")
    print(f"  BC sum: {paired['bc_pnl'].sum():+.0f}%, mean: {paired['bc_pnl'].mean():+.2f}%")
    print(f"  SP sum: {paired['sp_pnl'].sum():+.0f}%, mean: {paired['sp_pnl'].mean():+.2f}%")

    # 用户理论检验: SP wr > BC wr (因横盘+涨都赢)
    print(f"\n用户理论 (SP wr 应高于 BC):")
    if sp_only_win > bc_only_win:
        print(f"  ✓ 验证: SP {sp_only_win/n*100:.1f}% > BC {bc_only_win/n*100:.1f}%")
    else:
        print(f"  ✗ 反: SP {sp_only_win/n*100:.1f}% ≤ BC {bc_only_win/n*100:.1f}%")
