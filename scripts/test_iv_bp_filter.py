"""测试 GVZ IV 区间下, bp_low 深破阈值过滤效果.

核心规律 (v3.7.116):
  IV < 22:  无需过滤, BC 100% wr
  IV 22-28: 深破不改善, 需二次确认 (RSI+MACD+Stoch align)
  IV > 28:  bp_low ≤ 0.10 + 切 SP (BC 全错向, SP 50%+ wr 保本)

数据源: scripts/full_history_backtest.py 输出 CSV (含 bp_low 列).
"""
import pandas as pd
from pathlib import Path
CSV = Path("/Users/yhdong/Gold/data/backtest_history")

for asset in ["GLD", "SLV"]:
    df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260505.csv",
                       parse_dates=["signal_date","exit_date"])
    df = df[df["stage"]=="stage2_kline_real"]
    bc = df[df["strategy"]=="BUY CALL"]
    sp = df[df["strategy"]=="SELL PUT"]
    print(f"\n=== {asset} (真实期权 stage2) ===")
    for iv_lo, iv_hi, label in [(0, 22, "IV<22 (低 vol)"),
                                  (22, 28, "IV 22-28 (中)"),
                                  (28, 100, "IV>28 (高 vol)")]:
        bc_iv = bc[(bc["gvz_iv_pct"] >= iv_lo) & (bc["gvz_iv_pct"] < iv_hi)]
        sp_iv = sp[(sp["gvz_iv_pct"] >= iv_lo) & (sp["gvz_iv_pct"] < iv_hi)]
        print(f"\n  【{label}】(BC {len(bc_iv)} / SP {len(sp_iv)})")
        print(f"  {'bp_low':<14} {'BC n':>5} {'BC wr':>7} {'BC cum':>9} "
              f"{'SP n':>5} {'SP wr':>7} {'SP cum':>9}")
        for thresh in [1.0, 0.30, 0.20, 0.10, 0.05, 0.0]:
            bc_use = bc_iv[bc_iv["bp_low"] <= thresh]
            sp_use = sp_iv[sp_iv["bp_low"] <= thresh]
            if not len(bc_use) and not len(sp_use): continue
            bc_wr = (bc_use["pnl_pct"]>0).mean()*100 if len(bc_use) else 0
            sp_wr = (sp_use["pnl_pct"]>0).mean()*100 if len(sp_use) else 0
            print(f"  bp<={thresh:>5.2f}      {len(bc_use):>5} {bc_wr:>6.1f}% "
                  f"{bc_use['pnl_pct'].sum():>+8.0f}% "
                  f"{len(sp_use):>5} {sp_wr:>6.1f}% "
                  f"{sp_use['pnl_pct'].sum():>+8.0f}%")
