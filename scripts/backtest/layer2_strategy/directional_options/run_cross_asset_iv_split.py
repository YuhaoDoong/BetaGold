"""v3.7.231: Cross-asset SLV-S → GLD BC vs SP, 按 GLD 当日 GVZ 分桶.

假设: SLV-S 历史多数日子 GLD 不在高 IV regime → BC 占优 (long gamma 翻倍).
      但 GLD GVZ ≥25 高 IV 日, SP 收 premium 可能 > BC long gamma.

测试方法:
  1. SLV {S/A/S+A} 信号日列表
  2. 按 GLD 当日 GVZ 分桶: high (>=25), mid (22-25), low (<22)
  3. 每桶分别跑 GLD BC + GLD SP backtest
  4. 看 WR / mean / sum / scoreB 差异
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from scripts.backtest.framework import build_raw_universe
from core.paper_positions import price_strategy_at, simulate_option_exit


def get_slv_signal_dates(raw_slv, tier):
    bs = raw_slv["buy_signal"].fillna(False).astype(bool)
    st = raw_slv["signal_tier"].fillna("")
    if tier == "S+A":
        return raw_slv.index[bs & st.isin(["S","A"])].tolist()
    return raw_slv.index[bs & (st == tier)].tolist()


def backtest_option(dates, ohlc, asset, strategy):
    today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    rows = []
    for d in dates:
        if d not in ohlc.index: continue
        eO = float(ohlc.loc[d,"Open"]); eC = float(ohlc.loc[d,"Close"])
        eH = float(ohlc.loc[d,"High"]); eL = float(ohlc.loc[d,"Low"])
        ent = price_strategy_at(asset, strategy, d,
                                      d + pd.Timedelta(hours=9, minutes=30),
                                      eO, eO, eC, eH, eL, dte_target=30)
        if not ent.get("legs"): continue
        sim = simulate_option_exit(ent, d, strategy, today,
                                          live_spot=eC, live_high=eH, live_low=eL)
        rows.append({"date": d, "closed": bool(sim.get("is_closed", False)),
                      "pnl": float(sim.get("pnl_pct", 0) or 0),
                      "hold": int(sim.get("hold_days", 0) or 0)})
    return pd.DataFrame(rows)


def score(pnls):
    if not len(pnls): return {"n": 0, "scoreB": 0}
    s = pnls if isinstance(pnls, pd.Series) else pd.Series(pnls)
    n = len(s); wr = (s > 0).mean()
    return {"n": n, "WR": round(wr*100, 1), "mean": round(s.mean(), 2),
              "sum": round(s.sum(), 1), "max_loss": round(s.min(), 1),
              "max_gain": round(s.max(), 1),
              "scoreB": round((wr**2) * math.log(1+n) * s.mean(), 2)}


def main():
    print("=" * 100)
    print("Cross-asset SLV → GLD: BC vs SP 按 GLD GVZ 分桶")
    print("=" * 100)

    raw_slv, _ = build_raw_universe("SLV")
    raw_gld, ohlc_gld = build_raw_universe("GLD")

    # GLD GVZ 序列
    gld_gvz = raw_gld["gvz"]

    out_rows = []

    for tier in ["S", "A", "S+A"]:
        slv_dates = get_slv_signal_dates(raw_slv, tier)
        # 限定到 kline_db 范围 + GLD 数据存在
        slv_dates = [d for d in slv_dates if d >= pd.Timestamp("2025-04-29")
                      and d in ohlc_gld.index]
        if not slv_dates:
            print(f"\nSLV-{tier}: 无 in-window 日期, 跳过"); continue

        # 每个日期取 GLD 当日 GVZ
        date_gvz = pd.Series({d: gld_gvz.get(d, None) for d in slv_dates}).dropna()
        print(f"\n{'─'*100}\nSLV-{tier}: n={len(date_gvz)} 笔历史信号 (kline_db 窗内)")
        print(f"GVZ 分布: min={date_gvz.min():.1f}, max={date_gvz.max():.1f}, "
              f"mean={date_gvz.mean():.1f}, median={date_gvz.median():.1f}")
        print(f"  GVZ>=25 (高 IV): {(date_gvz>=25).sum()} 笔")
        print(f"  GVZ 22-25 (中):  {((date_gvz>=22)&(date_gvz<25)).sum()} 笔")
        print(f"  GVZ <22 (低):    {(date_gvz<22).sum()} 笔")

        # 分桶 backtest
        for bucket_label, mask in [
            ("ALL (全部)",       date_gvz.notna()),
            ("HIGH IV (≥25)",    date_gvz >= 25),
            ("MID IV (22-25)",   (date_gvz >= 22) & (date_gvz < 25)),
            ("LOW IV (<22)",     date_gvz < 22),
        ]:
            bucket_dates = date_gvz[mask].index.tolist()
            if len(bucket_dates) < 2:
                continue
            print(f"\n  [{bucket_label}] n_signal={len(bucket_dates)}:")
            for strategy in ["BUY CALL", "SELL PUT"]:
                df = backtest_option(bucket_dates, ohlc_gld, "GLD", strategy)
                closed = df[df["closed"]]
                s = score(closed["pnl"]) if len(closed) else {"n": 0}
                print(f"    GLD {strategy:9s}: closed={s.get('n')}/{len(df)}  "
                      f"WR={s.get('WR')}%  mean={s.get('mean')}%  "
                      f"sum={s.get('sum')}  max_loss={s.get('max_loss')}  "
                      f"scoreB={s.get('scoreB')}")
                out_rows.append({"slv_tier": tier, "bucket": bucket_label,
                                  "strategy": strategy,
                                  "n_signal": len(bucket_dates),
                                  **{k: s.get(k) for k in ["n","WR","mean","sum","max_loss","max_gain","scoreB"]}})

    df_all = pd.DataFrame(out_rows)
    print(f"\n\n=== 总览 ===")
    print(df_all.to_string(index=False))
    out = "/Users/yhdong/Gold/data/backtest_history/v3.7.231_cross_iv_split/cross_asset_bc_vs_sp_by_iv.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
