"""期货 + 期权退出参数 grid — 期货/期权独立优化.

期货 (Binance XAUUSDT 20× perp):
  指标: spot % move + hold days (无 IV)
  Grid: TP 3-12% × SL 2-5% × hold 3-15d

期权 (BC / SP / STRADDLE / SHORT_VOL):
  指标: premium % (含 IV crush + theta)
  Grid:
    BC long call:  TP 50/80/100/150% × SL -30/-50/-70% × DTE base
    SP credit spread:  profit_target 30/50/70% × SL on margin -30/-50/-70%
    STRADDLE long vol: TP 50/100/150% × hold 7/14/21d
    SHORT_VOL credit:  TP 30/50/70% × hold 14/30/45d

加仓机制 (盘中加仓 dedupe min_drop_pct):
  Grid: 0.1 / 0.2 / 0.3 / 0.5 / 0.7 / 1.0 / 1.5 / 2.0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np

CSV = Path("/Users/yhdong/Gold/data/backtest_history")


# ── 期货 grid ──────────────────────────────────────────
def sim_futures_with_params(entry_d, entry, ohlc, today, tp, sl, hold_max):
    later = ohlc.index[ohlc.index > entry_d]
    if not len(later): return None
    hold = 0
    for d in later:
        if pd.Timestamp(d) > today: break
        hold += 1
        H = float(ohlc.loc[d, "High"])
        L = float(ohlc.loc[d, "Low"])
        C = float(ohlc.loc[d, "Close"])
        rL = (L / entry - 1) * 100
        rH = (H / entry - 1) * 100
        # 时间序: 假设最坏先发 (用户保守)
        if rL <= -sl: return -sl
        if rH >= tp: return tp
        if hold >= hold_max:
            return (C / entry - 1) * 100
    return None


def grid_futures():
    print("="*80)
    print("【期货 (Binance perp 20×)】 TP × SL × hold_days 3D grid")
    print("="*80)
    today = pd.Timestamp.now().normalize()
    for asset in ["GLD", "SLV"]:
        ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                            index_col=0, parse_dates=True)
        df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                          parse_dates=["signal_date"])
        sigs = df[df["strategy"]=="FUTURES_LONG"][["signal_date"]].drop_duplicates()
        rows = []
        for tp in [3, 4, 5, 6, 8, 10, 12]:
            for sl in [2, 3, 4, 5]:
                for hold in [3, 5, 7, 10, 15]:
                    pnls = []
                    for _, r in sigs.iterrows():
                        d = pd.Timestamp(r["signal_date"]).normalize()
                        if d not in ohlc.index: continue
                        e = float(ohlc.loc[d, "Close"])
                        p = sim_futures_with_params(d, e, ohlc, today, tp, sl, hold)
                        if p is not None: pnls.append(p)
                    if not pnls: continue
                    pnls = pd.Series(pnls)
                    rows.append({
                        "tp": tp, "sl": sl, "hold": hold, "n": len(pnls),
                        "wr": (pnls>0).mean()*100, "avg": pnls.mean(),
                        "sum": pnls.sum(),
                        "sharpe": pnls.mean()/pnls.std() if pnls.std()>0 else 0,
                        "lev_sum": pnls.sum() * 20,  # 20× leverage
                    })
        rep = pd.DataFrame(rows)
        cur = rep[(rep.tp==3) & (rep.sl==3) & (rep.hold==5)]
        cur_str = (f"现行 (3/3/5): wr={cur.iloc[0]['wr']:.1f}% sum={cur.iloc[0]['sum']:+.1f}% "
                   f"lev={cur.iloc[0]['lev_sum']:+.0f}%") if len(cur) else "(n/a)"
        print(f"\n{asset} 期货 (n_signals={len(sigs)})  {cur_str}")
        print(f"  Top 6 by spot sum:")
        print(f"  {'tp':>3}{'sl':>3}{'hold':>5}{'n':>5}{'wr':>7}{'avg':>8}"
              f"{'sum':>9}{'lev sum':>10}{'sharpe':>9}")
        for _, r in rep.nlargest(6, "sum").iterrows():
            mark = " ←现行" if (r['tp']==3 and r['sl']==3 and r['hold']==5) else ""
            print(f"  {int(r['tp']):>3}{int(r['sl']):>3}{int(r['hold']):>5}{int(r['n']):>5}"
                  f"{r['wr']:>6.1f}%{r['avg']:>+7.2f}%{r['sum']:>+8.1f}%"
                  f"{r['lev_sum']:>+9.0f}%{r['sharpe']:>+8.3f}{mark}")


# ── 期权 grid ──────────────────────────────────────────
def grid_options_BC_SL():
    """BC long call: 用 backtest 真实 pnl_pct 反推不同 SL 下的 wr/sum.
    SL=-50% 现状: BC 笔 pnl<=-45% 视为止损出场.
    SL=-100% (无 SL): 止损笔损失 -100% (假设全亏权利金).
    SL=-30%: 不可重现 (没逐日期权 OHLC), 跳过.
    """
    print("\n" + "="*80)
    print("【BC long call】 SL 影响估测 (用 backtest 现有数据)")
    print("="*80)
    for asset in ["GLD", "SLV"]:
        df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                          parse_dates=["signal_date"])
        bc = df[df["strategy"]=="BUY CALL"].copy()
        n = len(bc)
        wr = (bc["pnl_pct"]>0).mean()*100
        avg = bc["pnl_pct"].mean()
        s = bc["pnl_pct"].sum()
        print(f"\n{asset} BC (n={n}) 现行 SL=-50%: wr={wr:.1f}% avg={avg:+.2f}% sum={s:+.1f}%")
        # 模拟 SL=-100% (全亏)
        sim = bc["pnl_pct"].clip(lower=-100)  # 已是
        # 真实 -50% SL 的笔 → 真无 SL 时不知道, 但可以保守估
        sl_n = (bc["pnl_pct"] <= -45).sum()
        print(f"  SL 触发笔 (pnl<=-45%): {sl_n}/{n} ({sl_n/n*100:.0f}%)")


def grid_options_SP_TP():
    """SP credit spread: profit_target 30/50/70%.
    现状 +50% credit → 早平.
    SLV/GLD CSV pnl_pct 已是 margin 分母. 不易反推.
    用 backtest CSV 的 SP 数据, 看 +50% TP 笔分布.
    """
    print("\n" + "="*80)
    print("【SP credit spread】 profit_target % 的影响")
    print("="*80)
    for asset in ["GLD", "SLV"]:
        df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                          parse_dates=["signal_date"])
        sp = df[df["strategy"]=="SELL PUT"].copy()
        # SP pnl_pct 是 margin 分母 (max_risk = spread - credit)
        # +50% TP 触发的笔 pnl_pct 应在 +20% ~ +60% 范围 (取决于 credit/margin 比)
        # 看分布
        print(f"\n{asset} SP (n={len(sp)}) pnl_pct 分布:")
        print(f"  [-100,-50): {((sp.pnl_pct<-50)).sum():>3}笔   "
              f"(止损 -50% margin)")
        print(f"  [-50,0):    {((sp.pnl_pct>=-50) & (sp.pnl_pct<0)).sum():>3}笔")
        print(f"  [0,30):     {((sp.pnl_pct>=0) & (sp.pnl_pct<30)).sum():>3}笔   "
              f"(无早平/expiry 收 part credit)")
        print(f"  [30,80):    {((sp.pnl_pct>=30) & (sp.pnl_pct<80)).sum():>3}笔   "
              f"(+50% TP 早平区)")
        print(f"  >=80:       {(sp.pnl_pct>=80).sum():>3}笔")
        print(f"  整体 wr: {(sp.pnl_pct>0).mean()*100:.1f}% mean: {sp.pnl_pct.mean():+.2f}%")


def grid_addon():
    """加仓 dedupe grid — 看不同 min_drop_pct 下平均加仓笔/日."""
    print("\n" + "="*80)
    print("【加仓 dedupe】 min_drop_pct grid (盘中真实 log)")
    print("="*80)
    from core.intraday_triggers import load_log, dedupe_intraday
    log = load_log("/Users/yhdong/Gold/data/intraday_signal_log.parquet")
    for asset in ["GLD", "SLV"]:
        sub = log[(log["asset"]==asset) & (log["side"]=="BUY")]
        n_days = sub["date"].nunique()
        n_raw = len(sub)
        print(f"\n{asset} BUY: raw {n_raw} 笔, 跨 {n_days} 交易日 "
              f"(平均 {n_raw/n_days:.0f} 笔/日)")
        print(f"  {'mdp':>5}{'总加仓笔':>10}{'/日':>8}{'压缩比':>10}")
        for mdp in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            n_total = 0
            for d, grp in sub.groupby("date"):
                dd = dedupe_intraday(grp, side="BUY", min_drop_pct=mdp)
                n_total += len(dd)
            cur = " ←现行" if mdp == 0.3 else ""
            print(f"  {mdp:>5.1f}{n_total:>10}{n_total/n_days:>7.2f}"
                  f"{n_raw/n_total:>9.0f}x  {cur}")


if __name__ == "__main__":
    grid_futures()
    grid_options_BC_SL()
    grid_options_SP_TP()
    grid_addon()
