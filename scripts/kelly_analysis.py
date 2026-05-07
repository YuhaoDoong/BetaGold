"""Kelly criterion 资金管理分析 — 每策略最优 sizing fraction.

Kelly 公式 (binary outcome):
  f* = (p × W - q × L) / (W × L)
  其中:
    p = 胜率
    q = 1-p (败率)
    W = avg_win_pct (赢面平均盈利)
    L = avg_loss_pct (亏面平均损失, 取正值)

Full Kelly 太激进 (drawdown 50%+ 心理崩盘), 实务用:
  Half Kelly (× 0.5): 经典平衡
  Quarter Kelly (× 0.25): 极保守 (推荐起步)

每策略独立 Kelly:
  - BC long call: 高 wr 高 avg, Kelly 较小 (单笔风险 -50% 已大)
  - SP credit spread: 高 wr 低 avg, Kelly 中 (max_risk = -50% margin)
  - STRADDLE: 中 wr 高 avg, Kelly 略大
  - 期货 20×: wr 75% / +12% avg (lev), Kelly 计算 lev_pnl
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np

CSV = Path("/Users/yhdong/Gold/data/backtest_history")


def kelly_fraction(p: float, W: float, L: float) -> float:
    """Kelly fraction. p: win prob, W: avg win %, L: avg loss % (正值)."""
    if W <= 0 or L <= 0: return 0
    q = 1 - p
    f = (p * W - q * L) / (W * L)
    return max(0, f)  # 期望负时 = 不下注


def analyze_strategy(df, asset, strat):
    """单策略 Kelly + 推荐 sizing."""
    if "FUTURES" in strat:
        # 期货用 lev_pnl (20× ROI on margin)
        # 但 backtest CSV 没存 levered_pnl_pct 列, 用 spot × 20 估
        sub = df[df["strategy"] == strat].copy()
        if "levered_pnl_pct" in sub.columns and sub["levered_pnl_pct"].notna().any():
            pnls = sub["levered_pnl_pct"].dropna()
        else:
            pnls = sub["pnl_pct"].dropna() * 20
    else:
        sub = df[df["strategy"] == strat].copy()
        pnls = sub["pnl_pct"].dropna()
    if not len(pnls):
        return None
    wins = pnls[pnls > 0]; losses = pnls[pnls <= 0]
    p = len(wins) / len(pnls) if len(pnls) else 0
    W = wins.mean() if len(wins) else 0
    L = abs(losses.mean()) if len(losses) else 1.0  # 避免 div 0
    f_full = kelly_fraction(p, W, L)
    f_half = f_full / 2
    f_qtr = f_full / 4
    expected = p * W - (1 - p) * L
    return {
        "asset": asset, "strategy": strat,
        "n": len(pnls), "wr": p * 100,
        "avg_win": W, "avg_loss": -L,  # 负值显示
        "expected_per_trade": expected,
        "kelly_full": f_full * 100,
        "kelly_half": f_half * 100,
        "kelly_quarter": f_qtr * 100,
    }


print("="*90)
print("Kelly Criterion 资金管理分析 (5y 全历史)")
print("="*90)
print(f"\n{'asset':<5}{'strategy':<14}{'n':>4}{'wr':>7}{'W':>10}{'L':>9}"
      f"{'EV/笔':>9}{'Kelly':>9}{'½K (rec)':>10}{'¼K (safe)':>11}")
print("-" * 90)

results = []
for asset in ["GLD", "SLV"]:
    df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                      parse_dates=["signal_date"])
    for strat in ["BUY CALL", "SELL PUT", "STRADDLE", "FUTURES_LONG"]:
        r = analyze_strategy(df, asset, strat)
        if r is None: continue
        results.append(r)
        # 标 Kelly 异常值
        marker = ""
        if r["kelly_full"] > 100: marker = " ⚠️ >100% (杠杆)"
        elif r["kelly_full"] < 5: marker = " ⚠️ 边际"
        print(f"{r['asset']:<5}{r['strategy']:<14}{r['n']:>4}"
              f"{r['wr']:>6.1f}%{r['avg_win']:>+9.2f}%{r['avg_loss']:>+8.2f}%"
              f"{r['expected_per_trade']:>+8.2f}%{r['kelly_full']:>7.1f}%"
              f"{r['kelly_half']:>9.1f}%{r['kelly_quarter']:>10.1f}%{marker}")

print()
print("="*90)
print("推荐资金分配 (按¼Kelly per asset, 同时持仓上限不超过 100% capital)")
print("="*90)
print()
print("理论 Kelly 含义:")
print("  - Kelly fraction: 每笔投入资金占 capital 的最优比例")
print("  - >100% Kelly = 借款交易 (实务不可行, 杠杆策略另算)")
print("  - <5% Kelly = 单笔 EV 微弱, 不值得交易")
print("  - 推荐: 期权用 ¼ Kelly, 期货用 ⅛ Kelly (杠杆放大波动)")

# 组合 Kelly: 同时持有多策略时的总仓位上限
print("\n组合策略同时持仓时的资金分配建议:")
print(f"  {'资产':<5}{'BC':>10}{'SP':>10}{'STRADDLE':>11}{'FUTURES':>11}{'合计':>10}")
for asset in ["GLD", "SLV"]:
    asset_results = [r for r in results if r["asset"] == asset]
    sub_pct = {r["strategy"]: min(25, r["kelly_quarter"]) for r in asset_results}  # 单策略上限 25%
    total = sum(sub_pct.values())
    print(f"  {asset:<5}"
          f"{sub_pct.get('BUY CALL', 0):>9.1f}%"
          f"{sub_pct.get('SELL PUT', 0):>9.1f}%"
          f"{sub_pct.get('STRADDLE', 0):>10.1f}%"
          f"{sub_pct.get('FUTURES_LONG', 0):>10.1f}%"
          f"{total:>9.1f}%")

# 按 sp_score 分档 Kelly (用 BC 子集)
print("\n按 sp_score 分档 Kelly (GLD/SLV BC 信号):")
print(f"  {'asset':<5}{'score 范围':<14}{'n':>4}{'wr':>7}{'avg':>9}"
      f"{'EV':>9}{'Kelly':>9}{'¼K':>9}")
from core.strategy_config import get_config
for asset in ["GLD", "SLV"]:
    cfg = get_config(asset)
    df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                      parse_dates=["signal_date"])
    bc = df[df["strategy"] == "BUY CALL"].copy()

    def score(r):
        s = 0.0
        s += cfg.sp_score_w_iv_rv_gap   * (r.get("iv_rv_gap_pct", 0) > 0)
        s += cfg.sp_score_w_bp_low_deep * (r.get("bp_low", 1) < 0.05)
        s += cfg.sp_score_w_bp_close_low * (r.get("bp_close", 1) < 0.30)
        s += cfg.sp_score_w_gvz_high    * (r.get("gvz_iv_pct", 0) >= 28)
        s += cfg.sp_score_w_rsi_oversold * (r.get("rsi_14", 50) < 30)
        s += cfg.sp_score_w_stoch_low   * (r.get("stoch_k", 50) < 40)
        s += cfg.sp_score_w_macd_bear   * (r.get("macd_hist", 0) < -0.5)
        return s
    bc["score"] = bc.apply(score, axis=1)
    for lo, hi, label in [(0, 1, "<1 (顶质)"), (1, 2, "1-2 (优)"),
                              (2, 3, "2-3 (中)"), (3, 100, "≥3 (转 SP)")]:
        sub = bc[(bc["score"] >= lo) & (bc["score"] < hi)]
        if not len(sub): continue
        wins = sub[sub["pnl_pct"] > 0]["pnl_pct"]
        losses = sub[sub["pnl_pct"] <= 0]["pnl_pct"]
        p = len(wins) / len(sub)
        W = wins.mean() if len(wins) else 0
        L = abs(losses.mean()) if len(losses) else 1
        f = kelly_fraction(p, W, L)
        ev = p * W - (1-p) * L
        print(f"  {asset:<5}{label:<14}{len(sub):>4}"
              f"{p*100:>6.1f}%{(W if not np.isnan(W) else 0):>+8.2f}%"
              f"{ev:>+8.2f}%{f*100:>7.1f}%{f*25:>7.1f}%")
