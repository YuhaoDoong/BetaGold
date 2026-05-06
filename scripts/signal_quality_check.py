"""信号质量分析 — 回答两个用户问题:

(1) buy_signal (bp_low<0.30) 本身的方向准确度如何? 是否方向性信号本身假?
    对每个 buy_signal=True 信号日 d, 看后续 H 天内 max_up / max_down,
    判断"反弹胜率"作为信号质量上限.

(2) DTE paired 比较 (同信号日同时模拟近月 DTE45 vs LEAPS DTE365) 的真实差异
    用 simulate_option_stage23 在最近 90 天 kline_db 覆盖期内,
    强制两种 DTE 各模拟一次, paired 比较.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

CSV = Path("/Users/yhdong/Gold/data/backtest_history")


def part1_signal_directional_quality():
    """读 backtest CSV, 抽 buy_signal 信号日 + 用 raw price 算 forward return.

    问题: bp_low<0.30 触发后, 1d/3d/5d/10d 涨/跌幅分布?
    """
    print("="*70)
    print("【问题1】 方向性信号本身的质量 (bp_low<0.30 触发后实际走势)")
    print("="*70)
    for asset in ["GLD", "SLV"]:
        # 读 OHLC 算 forward 收益
        ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                            index_col=0, parse_dates=True)
        # 读最新 backtest CSV (含信号日) — 用 BUY CALL 行去重, signal_date 唯一
        df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                          parse_dates=["signal_date"])
        sigs = df[df["strategy"]=="BUY CALL"][["signal_date","bp_low","bp_close",
                                                 "rv_pctile","gvz_iv_pct"]].drop_duplicates("signal_date")
        print(f"\n{asset} 信号日: {len(sigs)}")
        rows = []
        for _, r in sigs.iterrows():
            d = r["signal_date"].normalize()
            if d not in ohlc.index: continue
            entry = float(ohlc.loc[d, "Close"])
            for h in [1, 3, 5, 10, 20]:
                future = ohlc[(ohlc.index > d) &
                                (ohlc.index <= d + pd.Timedelta(days=h*1.6))]
                if len(future) < min(h, 2): continue
                future = future.head(h)
                max_up = (future["High"].max() / entry - 1) * 100
                max_dn = (future["Low"].min() / entry - 1) * 100
                close_h = (future["Close"].iloc[-1] / entry - 1) * 100
                rows.append({"d": d, "h": h, "max_up": max_up,
                              "max_dn": max_dn, "close": close_h,
                              "bp_low": r["bp_low"]})
        rep = pd.DataFrame(rows)
        if not len(rep):
            print(f"  {asset}: 无数据"); continue
        print(f"  {'h':>4}{'n':>5}{'wr_close>0':>12}{'avg_max_up':>12}{'avg_max_dn':>12}"
              f"{'avg_close':>11}{'wr_max_up>2%':>14}")
        for h in [1, 3, 5, 10, 20]:
            sub = rep[rep["h"]==h]
            if not len(sub): continue
            wr = (sub["close"]>0).mean()*100
            au = sub["max_up"].mean()
            ad = sub["max_dn"].mean()
            ac = sub["close"].mean()
            wr_2pct = (sub["max_up"]>2.0).mean()*100
            print(f"  {h:>4}{len(sub):>5}{wr:>11.1f}%{au:>+11.2f}%{ad:>+11.2f}%"
                  f"{ac:>+10.2f}%{wr_2pct:>13.1f}%")


def part2_paired_dte():
    """同信号日, 强制 DTE=45 vs DTE=365, paired 比较.
    用 simulate_option_stage23 在最近 90 天 kline_db 覆盖范围内.
    """
    print("\n" + "="*70)
    print("【问题2】 paired DTE 比较 (同信号日 DTE45 vs DTE365)")
    print("="*70)
    try:
        from scripts.full_history_backtest import simulate_option_stage23
    except ImportError as e:
        print(f"  导入失败: {e}"); return

    today = datetime.now(ZoneInfo("America/New_York")).date()
    today = pd.Timestamp(today)
    KLINE_START = pd.Timestamp("2025-04-29")  # kline_db 覆盖起点

    for asset in ["GLD", "SLV"]:
        df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                          parse_dates=["signal_date"])
        # 抽近 90 天信号 (kline_db 必有数据), 仅 BUY CALL 信号日去重
        cutoff = today - pd.Timedelta(days=400)  # 扩到 ~400d, kline_db ~1年
        sigs = df[(df["strategy"]=="BUY CALL") &
                    (df["signal_date"] >= cutoff) &
                    (df["signal_date"] >= KLINE_START)
                  ][["signal_date","entry_spot_etf"]].drop_duplicates("signal_date")
        print(f"\n{asset} 候选信号 ({cutoff.date()} ~): {len(sigs)}")
        if not len(sigs): continue

        rows = []
        ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                            index_col=0, parse_dates=True)
        # 假 daily DataFrame for simulate (要 Open/High/Low/Close)
        daily = ohlc[["Open","High","Low","Close"]].copy()

        for _, r in sigs.iterrows():
            d = r["signal_date"]
            spot = r["entry_spot_etf"]
            for label, dte in [("近月45", 45), ("LEAPS365", 365)]:
                try:
                    res = simulate_option_stage23(d, spot, "BUY CALL", asset,
                                                       daily, today, force_dte=dte)
                    if "pnl_pct" in res:
                        rows.append({"d": d, "label": label,
                                       "pnl": res["pnl_pct"],
                                       "closed": res.get("closed", False),
                                       "is_mtm": res.get("is_mtm", False),
                                       "reason": res.get("reason", "")})
                except TypeError:
                    # 不支持 force_dte → 跳过 (走默认智能 DTE)
                    pass
                except Exception:
                    pass

        if not len(rows):
            print(f"  无可比较结果 (simulate_option_stage23 不支持 force_dte; "
                   f"需要修改函数加参数)"); continue
        rep = pd.DataFrame(rows)
        # paired
        m45 = rep[rep["label"]=="近月45"].set_index("d")["pnl"]
        m365 = rep[rep["label"]=="LEAPS365"].set_index("d")["pnl"]
        paired = pd.concat([m45.rename("near"), m365.rename("leaps")],
                              axis=1).dropna()
        print(f"  {asset} paired (同信号日两种 DTE): n={len(paired)}")
        print(f"    近月 45 DTE:  wr {(paired['near']>0).mean()*100:.1f}% "
              f"avg {paired['near'].mean():+.2f}% 累 {paired['near'].sum():+.1f}%")
        print(f"    LEAPS 365:    wr {(paired['leaps']>0).mean()*100:.1f}% "
              f"avg {paired['leaps'].mean():+.2f}% 累 {paired['leaps'].sum():+.1f}%")
        print(f"    LEAPS - 近月 mean diff: "
              f"{(paired['leaps']-paired['near']).mean():+.2f}%")
        print(f"    LEAPS 比近月赢 (paired): "
              f"{((paired['leaps']>paired['near'])).sum()}/{len(paired)} "
              f"({((paired['leaps']>paired['near'])).mean()*100:.1f}%)")


if __name__ == "__main__":
    part1_signal_directional_quality()
    part2_paired_dte()
