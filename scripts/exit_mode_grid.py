"""期权 + 期货 SL 模式 grid: 跟随金价 (spot SL) vs 策略盈亏 (premium/margin SL)
找哪种 SL 模式累计 PnL 最优.

期权 (BC long call / SP credit spread):
  spot SL: 当 spot 跌破 X%, 强平 (跟金价直接挂钩)
  premium/margin SL: 当 option 价格 / margin 亏 Y%, 强平 (跟实际亏损挂钩)
  二者不等价 — theta/IV 让 premium 独立于 spot 变化

期货 (linear PnL): spot SL X% = margin SL (X×lev)% 数学等价, 不需 grid

策略:
  对每个 BC/SP 信号, 重模拟若干 (mode, threshold) 组合
  统计 wr/avg/sum, 找最优.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from core.paper_positions import _load_kline_db, price_strategy_at

CSV = Path("/Users/yhdong/Gold/data/backtest_history")
TODAY = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())


def sim_with_sl_mode(entry_pricing, sig_d, today_dt, db,
                       sl_mode: str, sl_threshold: float,
                       tp_mode: str = "premium", tp_threshold: float = 100.0,
                       spot_series: pd.Series = None,
                       strategy: str = "BUY CALL"):
    """模拟 BC/SP 退出, SL 模式可选 spot 或 premium/margin.

    sl_mode: "premium" (BC) / "margin" (SP) / "spot"
    sl_threshold: 该 mode 下的阈值 (e.g. 50 = 50%)
    """
    legs = entry_pricing["legs"]
    entry_value = entry_pricing["entry_price"]
    if abs(entry_value) < 0.01: return None
    is_credit = "SELL PUT" in strategy
    # max_risk for credit
    spread_width = 0.0
    if is_credit and len(legs) >= 2:
        ks = [l[2] for l in legs if "short" in l[0]]
        kl = [l[2] for l in legs if "put" in l[0] and "short" not in l[0]]
        if ks and kl: spread_width = abs(ks[0] - kl[0])
    max_risk = max(0.01, spread_width - entry_value) if (is_credit and spread_width > 0) else entry_value

    first_kdb = db[db["code"] == legs[0][1]]
    if not len(first_kdb): return None
    expiry_dt = pd.Timestamp(first_kdb.iloc[0]["expiry"])

    days = sorted(set(db[db["code"].isin([l[1] for l in legs])]["date"].unique()))
    days_after = [d for d in days if pd.Timestamp(d) > sig_d]
    entry_spot = float(spot_series.get(sig_d, 0)) if spot_series is not None else 0
    hold = 0
    for d in days_after:
        d_ts = pd.Timestamp(d)
        if d_ts > today_dt: break
        cur_total = 0.0; ok = True
        for _lab, _code, _K, _qty in legs:
            r = db[(db["code"] == _code) & (db["date"] == d_ts)]
            if not len(r): ok = False; break
            cur_total += _qty * float(r.iloc[0]["close"])
        if not ok: continue
        cur_value = -cur_total if is_credit else cur_total
        hold += 1

        # PnL
        if is_credit:
            pnl = (entry_value - cur_value) / max_risk * 100
        else:
            pnl = (cur_value / entry_value - 1) * 100

        # 退出检查
        # TP (统一 premium/margin 角度, 简化用 +50%/+100%)
        if not is_credit and pnl >= tp_threshold:
            return {"pnl": pnl, "reason": f"+{tp_threshold:.0f}% TP", "hold": hold}
        if is_credit and pnl >= 50:
            return {"pnl": pnl, "reason": "+50% credit TP", "hold": hold}

        # SL — 视模式
        if sl_mode == "spot":
            # 检查 spot 是否跌 sl_threshold%
            cur_spot = float(spot_series.get(d_ts, entry_spot))
            if entry_spot > 0:
                spot_change = (cur_spot / entry_spot - 1) * 100
                if spot_change <= -sl_threshold:
                    return {"pnl": pnl, "reason": f"spot -{sl_threshold}% SL",
                             "hold": hold}
        else:
            # premium / margin SL
            if is_credit:
                if pnl <= -sl_threshold:
                    return {"pnl": pnl, "reason": f"-{sl_threshold}% margin SL",
                             "hold": hold}
            else:
                if pnl <= -sl_threshold:
                    return {"pnl": pnl, "reason": f"-{sl_threshold}% premium SL",
                             "hold": hold}
        # Expiry
        if d_ts >= expiry_dt:
            return {"pnl": pnl, "reason": "expiry", "hold": hold}
    return None  # 未触发


def grid(strategy: str):
    print(f"\n{'='*78}")
    print(f"{strategy} SL 模式 grid (spot vs premium/margin)")
    print(f"{'='*78}")
    db = _load_kline_db()
    if db is None:
        print("kline_db 未加载"); return

    for asset in ["GLD", "SLV"]:
        ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                            index_col=0, parse_dates=True)
        spot_series = ohlc["Close"]
        df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                          parse_dates=["signal_date"])
        sigs = df[df["strategy"] == strategy][["signal_date","entry_spot_etf"]].drop_duplicates("signal_date")
        if not len(sigs): continue
        print(f"\n{asset} 信号: {len(sigs)}")

        # 跑各模式
        results = []
        # 1) Premium/Margin SL 各阈值
        sl_unit = "margin" if "SELL" in strategy else "premium"
        for sl_thr in [30, 50, 70, 100]:
            pnls = []
            for _, r in sigs.iterrows():
                d = pd.Timestamp(r["signal_date"]).normalize()
                spot = r["entry_spot_etf"]
                try:
                    ent = price_strategy_at(asset, strategy, d,
                                                 d + pd.Timedelta(hours=9, minutes=30),
                                                 spot, spot, spot, spot, spot,
                                                 dte_target=45)
                except Exception: continue
                if not ent.get("legs"): continue
                if abs(ent.get("entry_price", 0)) < 0.01: continue
                res = sim_with_sl_mode(ent, d, TODAY, db,
                                            sl_mode=sl_unit, sl_threshold=sl_thr,
                                            spot_series=spot_series, strategy=strategy)
                if res: pnls.append(res["pnl"])
            if not pnls: continue
            pnls = pd.Series(pnls)
            results.append({"mode": sl_unit, "thr": sl_thr, "n": len(pnls),
                             "wr": (pnls>0).mean()*100, "avg": pnls.mean(),
                             "sum": pnls.sum()})

        # 2) Spot SL 各阈值
        for sl_thr in [2, 3, 5, 7, 10]:
            pnls = []
            for _, r in sigs.iterrows():
                d = pd.Timestamp(r["signal_date"]).normalize()
                spot = r["entry_spot_etf"]
                try:
                    ent = price_strategy_at(asset, strategy, d,
                                                 d + pd.Timedelta(hours=9, minutes=30),
                                                 spot, spot, spot, spot, spot,
                                                 dte_target=45)
                except Exception: continue
                if not ent.get("legs"): continue
                if abs(ent.get("entry_price", 0)) < 0.01: continue
                res = sim_with_sl_mode(ent, d, TODAY, db,
                                            sl_mode="spot", sl_threshold=sl_thr,
                                            spot_series=spot_series, strategy=strategy)
                if res: pnls.append(res["pnl"])
            if not pnls: continue
            pnls = pd.Series(pnls)
            results.append({"mode": "spot", "thr": sl_thr, "n": len(pnls),
                             "wr": (pnls>0).mean()*100, "avg": pnls.mean(),
                             "sum": pnls.sum()})

        # 输出排序
        print(f"  {'mode':<10}{'thr':>5}{'n':>5}{'wr':>8}{'avg':>10}{'sum':>11}")
        for r in sorted(results, key=lambda x: -x["sum"]):
            mark = ""
            if r["mode"] == sl_unit and r["thr"] == 50:
                mark = " ←现行"
            print(f"  {r['mode']:<10}{int(r['thr']):>4}%"
                  f"{int(r['n']):>5}{r['wr']:>7.1f}%"
                  f"{r['avg']:>+9.2f}%{r['sum']:>+10.1f}%{mark}")


grid("BUY CALL")
grid("SELL PUT")
