"""Kelly Criterion 资金管理 v2 (v3.7.180 — 修 decimal 公式 bug + 用最新 backtest).

Kelly 公式 (binary outcome, 收益率为单位):
  f* = p/L_dec - q/W_dec    (推导后简化形式)
  ≡ (p × W_dec - q × L_dec) / (W_dec × L_dec)
  其中:
    p = 胜率
    q = 1-p
    W_dec = avg_win 的 DECIMAL 形式 (例: +50% → 0.50)
    L_dec = abs(avg_loss) 的 DECIMAL (例: -30% → 0.30)

⚠️ 旧脚本 bug: 直接传 % 数 (如 50, 30) 导致结果偏小 100×

实务用 Half / Quarter Kelly 抑制波动:
  Full Kelly:   理论最优 (但单笔 drawdown 50%+, 心理崩盘)
  Half Kelly:   收益约 75% Full, drawdown 减半
  Quarter Kelly: 极保守 (推荐起步 / 杠杆策略)

数据源: data/backtest_pipeline/stage2_simulated/ (近 1y real options + Binance)
        + data/backtest_history/backtest_*_20260507.csv (10y, 含 sim 期权)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np


PIPE_S2 = Path("/Users/yhdong/Gold/data/backtest_pipeline/stage2_simulated")
HIST_CSV = Path("/Users/yhdong/Gold/data/backtest_history")


def kelly_fraction_decimal(p: float, W_dec: float, L_dec: float) -> float:
    """Kelly fraction. p 胜率, W_dec/L_dec 用 DECIMAL (0.50 即 +50%)."""
    if W_dec <= 0 or L_dec <= 0: return 0.0
    q = 1 - p
    f = p / L_dec - q / W_dec
    return max(0.0, f)


def kelly_from_pnls(pnls: pd.Series) -> dict:
    """从 pnl_pct 系列直接算 Kelly + EV. pnls 是 % 数 (如 +50.0 表示 +50%)."""
    if len(pnls) < 3:
        return {"n": len(pnls), "wr": 0, "W_dec": 0, "L_dec": 0,
                "EV_pct": 0, "kelly_full_pct": 0,
                "kelly_half_pct": 0, "kelly_qtr_pct": 0}
    wins = pnls[pnls > 0]; losses = pnls[pnls <= 0]
    p = len(wins) / len(pnls)
    W_pct = wins.mean() if len(wins) else 0
    L_pct = abs(losses.mean()) if len(losses) else 1.0
    W_dec = W_pct / 100
    L_dec = L_pct / 100
    f = kelly_fraction_decimal(p, W_dec, L_dec)
    EV_pct = p * W_pct - (1 - p) * L_pct
    return {
        "n": len(pnls), "wr": p * 100,
        "W_pct": W_pct, "L_pct": -L_pct,  # 负值显示亏损
        "EV_pct": EV_pct,
        "kelly_full_pct": min(100.0, f * 100),  # cap 100% 显示用
        "kelly_half_pct": min(100.0, f * 50),
        "kelly_qtr_pct": min(100.0, f * 25),
        "kelly_raw": f,  # 不 cap
    }


def load_pnls_real(asset: str, strategy: str) -> pd.Series:
    """从 backtest_pipeline stage2 拿真实数据 PnL (近 1y kline_db / Binance)."""
    fname = f"pnl_{asset.lower()}_{strategy}_real_*"
    fname += "klinedb_" if strategy != "FUTURES_LONG" else "binance_"
    files = list(PIPE_S2.glob(fname + "*.parquet"))
    if not files: return pd.Series(dtype=float)
    df = pd.concat([pd.read_parquet(f) for f in files])
    return df["pnl_pct"].dropna()


def load_pnls_5y(asset: str, strategy: str) -> pd.Series:
    """从 backtest_history CSV 拿 10y 全历史 (含 sim 期权)."""
    files = sorted(HIST_CSV.glob(f"backtest_{asset.lower()}_*.csv"))
    if not files: return pd.Series(dtype=float)
    df = pd.read_csv(files[-1], parse_dates=["signal_date"])
    sub = df[df["strategy"] == strategy]
    return sub["pnl_pct"].dropna()


def report():
    print("=" * 92)
    print("Kelly Criterion 资金管理 (v2 修正版)")
    print("=" * 92)
    print(f"\n格式说明: W=avg_win(+%) L=avg_loss(-%) EV=单笔期望(%) "
          f"K=Full Kelly(%) ½K=Half ¼K=Quarter\n")

    # ── 真实数据 (近 1y) ──
    print("──" * 46)
    print("【近 1 年 真实数据】 期权 = kline_db EOD, 期货 = Binance perp")
    print("──" * 46)
    print(f"{'asset':<5}{'strategy':<14}{'n':>4}{'wr':>7}{'W':>9}{'L':>9}"
          f"{'EV':>9}{'Full K':>9}{'½K':>8}{'¼K':>8}")
    for asset in ["GLD", "SLV"]:
        for strat in ["BUY CALL", "SELL PUT", "STRADDLE", "FUTURES_LONG"]:
            pnls = load_pnls_real(asset, strat)
            if not len(pnls): continue
            r = kelly_from_pnls(pnls)
            note = ""
            if r["kelly_raw"] > 1.0: note = " ⚠ >100% (用杠杆放大)"
            elif r["kelly_raw"] < 0.05: note = " ⚠ 边际"
            print(f"{asset:<5}{strat:<14}{r['n']:>4}{r['wr']:>6.1f}%"
                  f"{r['W_pct']:>+8.1f}%{r['L_pct']:>+8.1f}%"
                  f"{r['EV_pct']:>+8.1f}%{r['kelly_full_pct']:>7.1f}%"
                  f"{r['kelly_half_pct']:>7.1f}%{r['kelly_qtr_pct']:>7.1f}%{note}")

    # ── 10y 全历史 (含 sim) ──
    print()
    print("──" * 46)
    print("【10y 全历史回测】 期权含 LEAPS BS proxy / 期货含 GC=F COMEX")
    print("──" * 46)
    print(f"{'asset':<5}{'strategy':<14}{'n':>4}{'wr':>7}{'W':>9}{'L':>9}"
          f"{'EV':>9}{'Full K':>9}{'½K':>8}{'¼K':>8}")
    rows_5y = {}
    for asset in ["GLD", "SLV"]:
        for strat in ["BUY CALL", "SELL PUT", "STRADDLE", "FUTURES_LONG"]:
            pnls = load_pnls_5y(asset, strat)
            if not len(pnls): continue
            r = kelly_from_pnls(pnls)
            rows_5y[(asset, strat)] = r
            note = ""
            if r["kelly_raw"] > 1.0: note = " ⚠ >100%"
            elif r["kelly_raw"] < 0.05: note = " ⚠ 边际"
            print(f"{asset:<5}{strat:<14}{r['n']:>4}{r['wr']:>6.1f}%"
                  f"{r['W_pct']:>+8.1f}%{r['L_pct']:>+8.1f}%"
                  f"{r['EV_pct']:>+8.1f}%{r['kelly_full_pct']:>7.1f}%"
                  f"{r['kelly_half_pct']:>7.1f}%{r['kelly_qtr_pct']:>7.1f}%{note}")

    # ── 推荐组合分配 (基于 10y, ½K) ──
    print()
    print("=" * 92)
    print("【推荐组合分配】 — 多策略并行, ½ Kelly 各 cap 25%/笔, 总曝露 ≤ 60%")
    print("=" * 92)
    print()
    print("分配公式: per_strategy_alloc = min(½Kelly, 25%) [单策略上限]")
    print("         total_alloc = sum 各策略, 若 > 60% 按比例缩放\n")
    print(f"{'asset':<5}{'BC':>9}{'SP':>9}{'STRADDLE':>11}{'FUT':>9}{'合计':>9}{'缩放':>9}")
    portfolio = {}
    for asset in ["GLD", "SLV"]:
        allocs = {}
        for strat in ["BUY CALL", "SELL PUT", "STRADDLE", "FUTURES_LONG"]:
            r = rows_5y.get((asset, strat))
            if r is None:
                allocs[strat] = 0.0
            else:
                allocs[strat] = min(25.0, r["kelly_half_pct"])
        total = sum(allocs.values())
        scale = min(1.0, 60.0 / total) if total > 60 else 1.0
        portfolio[asset] = {k: v * scale for k, v in allocs.items()}
        print(f"{asset:<5}"
              f"{portfolio[asset]['BUY CALL']:>8.1f}%"
              f"{portfolio[asset]['SELL PUT']:>8.1f}%"
              f"{portfolio[asset]['STRADDLE']:>10.1f}%"
              f"{portfolio[asset]['FUTURES_LONG']:>8.1f}%"
              f"{sum(portfolio[asset].values()):>8.1f}%"
              f"{scale*100:>8.0f}%")

    # ── 风险说明 ──
    print()
    print("=" * 92)
    print("【实务建议】")
    print("=" * 92)
    print("""
1. ½ Kelly = 收益约 75% Full, 但 drawdown 仅 ½ → 心理可承受 ★ 推荐
2. ¼ Kelly = 极保守 (适合杠杆策略 / 高波动期 / 资金 < $50k)
3. Full Kelly 数学最优但 max DD 通常 30-50% — 实务难持续

期货 lev 已含 leverage, Kelly fraction = capital % (非合约 %)
  例: GLD ½K=10% × $100k 资金 → $10k 保证金 × lev 5× = $50k notional

仓位上限规则:
  - 单策略上限 25% (避免 single-strat blowup)
  - 总曝露 ≤ 60% (留 40% 现金缓冲)
  - 当前 SP -100% case (3-17 GLD): 即使发生, ½K 5% × -100% = -5% 总账户

回测 vs 实盘风险:
  - 5y 含 sim, 真实期权窗口仅 1y → 用 ½K 估的实盘 ¼K 更稳
  - 极端波动期 (如 2026-03 暴跌) 实测 1y 数据已含
""")


if __name__ == "__main__":
    report()
