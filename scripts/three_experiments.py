"""三实验:
(A) deep_zone (bp_low<0.10) vs 普通 (0.10-0.30) 信号质量分层
(B) 期货止损 -2% vs -3% wr 对比
(C) sp_score 高 → BC skip (镜像 sp_score 到 BC 入场)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
from core.strategy_config import get_config

CSV = Path("/Users/yhdong/Gold/data/backtest_history")


def exp_A_deep_zone():
    print("="*70)
    print("(A) deep_zone (bp_low<0.10) vs 普通 (0.10-0.30) 信号 5d/10d 反弹质量")
    print("="*70)
    for asset in ["GLD", "SLV"]:
        ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                            index_col=0, parse_dates=True)
        df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                          parse_dates=["signal_date"])
        sigs = df[df["strategy"]=="BUY CALL"][["signal_date","bp_low"]].drop_duplicates("signal_date")
        rows = []
        for _, r in sigs.iterrows():
            d = r["signal_date"].normalize()
            if d not in ohlc.index: continue
            entry = float(ohlc.loc[d, "Close"])
            for h in [5, 10, 20]:
                future = ohlc[ohlc.index > d].head(h)
                if len(future) < min(h, 2): continue
                rows.append({
                    "bp_low": r["bp_low"],
                    "h": h,
                    "max_up": (future["High"].max()/entry-1)*100,
                    "close": (future["Close"].iloc[-1]/entry-1)*100,
                })
        rep = pd.DataFrame(rows)
        if not len(rep): continue
        rep["zone"] = np.where(rep["bp_low"]<0.10, "deep<0.10",
                                  np.where(rep["bp_low"]<0.20, "mid 0.10-0.20",
                                              "shallow 0.20-0.30"))
        print(f"\n{asset}")
        print(f"  {'zone':<20}{'h':>4}{'n':>5}{'wr_close>0':>12}"
              f"{'wr_max_up>2%':>14}{'avg_close':>11}")
        for zone in ["deep<0.10", "mid 0.10-0.20", "shallow 0.20-0.30"]:
            for h in [5, 10, 20]:
                sub = rep[(rep["zone"]==zone) & (rep["h"]==h)]
                if not len(sub): continue
                wr = (sub["close"]>0).mean()*100
                wr2 = (sub["max_up"]>2).mean()*100
                ac = sub["close"].mean()
                print(f"  {zone:<20}{h:>4}{len(sub):>5}{wr:>11.1f}%"
                      f"{wr2:>13.1f}%{ac:>+10.2f}%")


def exp_B_futures_stoploss():
    print("\n" + "="*70)
    print("(B) 期货 -2% vs -3% 止损 wr 对比 (5d 持仓, +3% 止盈不变)")
    print("="*70)
    for asset in ["GLD", "SLV"]:
        ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                            index_col=0, parse_dates=True)
        df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                          parse_dates=["signal_date"])
        sigs = df[df["strategy"]=="FUTURES_LONG"][["signal_date"]].drop_duplicates()
        for sl_pct in [2.0, 3.0, 4.0]:
            n=0; wins=0; pnl_sum=0; stops=0; tps=0; expires=0
            for _, r in sigs.iterrows():
                d = r["signal_date"].normalize()
                if d not in ohlc.index: continue
                entry = float(ohlc.loc[d, "Close"])
                future = ohlc[ohlc.index > d].head(5)
                if not len(future): continue
                # 模拟 (按时间序模拟止盈/止损/到期)
                exited = False
                for _, b in future.iterrows():
                    rL = (b["Low"]/entry - 1)*100
                    rH = (b["High"]/entry - 1)*100
                    if rL <= -sl_pct:
                        pnl = -sl_pct; stops+=1; exited=True; break
                    if rH >= 3.0:
                        pnl = 3.0; tps+=1; exited=True; break
                if not exited:
                    pnl = (future["Close"].iloc[-1]/entry-1)*100
                    expires+=1
                # 20× leverage
                lev_pnl = pnl * 20
                # 但用户报告 wr 60% 是裸 spot wr (pnl>0); 仍比同基准
                pnl_sum += pnl
                if pnl > 0: wins += 1
                n += 1
            wr = wins/n*100 if n else 0
            print(f"  {asset} stoploss={sl_pct}%  n={n} wr={wr:.1f}% "
                  f"avg={pnl_sum/n:+.2f}% (止损{stops} 止盈{tps} 到期{expires})")


def exp_C_sp_score_to_bc_skip():
    """高 sp_score → 该信号 SP 优于 BC, 应 skip BC. 看 BC 在不同 score 下的胜率."""
    print("\n" + "="*70)
    print("(C) sp_score 镜像到 BC: 高 score 时跳过 BC, 是否提升 BC 净 wr?")
    print("="*70)
    for asset in ["GLD", "SLV"]:
        cfg = get_config(asset)
        df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                          parse_dates=["signal_date"])
        bc = df[df["strategy"]=="BUY CALL"].copy()

        def score(r):
            s = 0.0
            if r.get("iv_rv_gap_pct", 0) > 0:    s += cfg.sp_score_w_iv_rv_gap
            if r.get("bp_low", 1) < 0.05:        s += cfg.sp_score_w_bp_low_deep
            if r.get("bp_close", 1) < 0.30:      s += cfg.sp_score_w_bp_close_low
            if r.get("gvz_iv_pct", 0) >= 28:     s += cfg.sp_score_w_gvz_high
            if r.get("rsi_14", 50) < 30:         s += cfg.sp_score_w_rsi_oversold
            if r.get("stoch_k", 50) < 40:        s += cfg.sp_score_w_stoch_low
            if r.get("macd_hist", 0) < -0.5:     s += cfg.sp_score_w_macd_bear
            return s

        bc["score"] = bc.apply(score, axis=1)
        print(f"\n{asset} (BC 全集 n={len(bc)} wr={(bc['pnl_pct']>0).mean()*100:.1f}% "
              f"avg={bc['pnl_pct'].mean():+.2f}%)")
        print(f"  {'score 阈值 (高于此 → SP 不走 BC)':<32}{'保留 BC n':>10}"
              f"{'BC wr':>9}{'BC avg':>10}")
        for thr in [10, 5, 4, 3.5, 3, 2.5, 2, 1.5, 1]:
            keep = bc[bc["score"] < thr]
            if not len(keep): continue
            wr = (keep["pnl_pct"]>0).mean()*100
            av = keep["pnl_pct"].mean()
            print(f"  score<{thr} (skip 高分){' '*15}{len(keep):>10}"
                  f"{wr:>8.1f}%{av:>+9.2f}%")


if __name__ == "__main__":
    exp_A_deep_zone()
    exp_B_futures_stoploss()
    exp_C_sp_score_to_bc_skip()
