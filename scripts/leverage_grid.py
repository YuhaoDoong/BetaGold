"""杠杆 grid — 不同 leverage 下最优 (TP, SL, hold) + 爆仓敏感性.

Binance XAUUSDT 上限 20×, 但 dYdX/Bybit 等可至 50/100×.
高杠杆爆仓距离极小 (lev=100 → 爆仓 0.5%), 必须 SL 提前.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
from core.strategies.futures_long import (
    FuturesConfig, simulate_long_position,
    liquidation_distance_pct, effective_sl_pct,
)

CSV = Path("/Users/yhdong/Gold/data/backtest_history")


def grid_per_leverage():
    """每个 leverage 跑 (TP, SL, hold) grid 找最优."""
    today = pd.Timestamp.now().normalize()
    results = []
    for asset in ["GLD", "SLV"]:
        ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                            index_col=0, parse_dates=True)
        df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                          parse_dates=["signal_date"])
        sigs = df[df["strategy"]=="FUTURES_LONG"][["signal_date"]].drop_duplicates()
        print(f"\n{'='*78}")
        print(f"{asset} 杠杆 grid (信号 {len(sigs)} 笔)")
        print(f"{'='*78}")
        for lev in [5, 10, 20, 50, 100]:
            liq_pct = liquidation_distance_pct(FuturesConfig(leverage=lev))
            print(f"\nLeverage {lev}× (爆仓距 {liq_pct:.1f}%)")
            print(f"  {'TP':>4}{'SL':>5}{'hold':>5}{'effSL':>7}{'n':>5}"
                   f"{'wr':>7}{'avg_lev':>10}{'sum_lev':>11}{'liq%':>6}")
            for tp in [3, 5, 8, 12]:
                for sl in [1, 2, 3, 5, 8]:
                    for hold in [5, 10, 15]:
                        cfg = FuturesConfig(leverage=lev, tp_pct=tp, sl_pct=sl,
                                                hold_max_days=hold)
                        eff_sl = effective_sl_pct(cfg)
                        if eff_sl < 0.4: continue  # 小于 0.4% SL 没意义
                        pnls = []; liqs = 0
                        for _, r in sigs.iterrows():
                            d = pd.Timestamp(r["signal_date"]).normalize()
                            if d not in ohlc.index: continue
                            e = float(ohlc.loc[d, "Close"])
                            res = simulate_long_position(d, e, ohlc, today, cfg)
                            if res.get("closed"):
                                pnls.append(res["ret_levered_pct"])
                                if res.get("is_liquidation"): liqs += 1
                        if not pnls: continue
                        pnls = pd.Series(pnls)
                        results.append({
                            "asset": asset, "lev": lev, "tp": tp, "sl": sl,
                            "hold": hold, "eff_sl": eff_sl,
                            "n": len(pnls), "wr": (pnls>0).mean()*100,
                            "avg_lev": pnls.mean(), "sum_lev": pnls.sum(),
                            "n_liq": liqs,
                        })
            # 该 lev 内 top 3
            sub = [r for r in results if r["asset"]==asset and r["lev"]==lev]
            sub.sort(key=lambda r: r["sum_lev"], reverse=True)
            for r in sub[:3]:
                print(f"  {int(r['tp']):>4}{int(r['sl']):>5}{int(r['hold']):>5}"
                      f"{r['eff_sl']:>6.1f}%{int(r['n']):>5}"
                      f"{r['wr']:>6.1f}%{r['avg_lev']:>+9.2f}%"
                      f"{r['sum_lev']:>+10.0f}%{int(r['n_liq']):>5}")

    # 跨杠杆最佳 per 资产
    print("\n\n" + "="*78)
    print("跨杠杆最优 (按累计 lev_sum 排序)")
    print("="*78)
    for asset in ["GLD", "SLV"]:
        sub = [r for r in results if r["asset"]==asset]
        sub.sort(key=lambda r: r["sum_lev"], reverse=True)
        print(f"\n{asset} Top 5:")
        print(f"  {'lev':>4}{'TP':>4}{'SL':>4}{'hold':>5}{'wr':>7}"
               f"{'avg_lev':>10}{'sum_lev':>11}{'n_liq':>6}")
        for r in sub[:5]:
            print(f"  {int(r['lev']):>3}×{int(r['tp']):>4}{int(r['sl']):>4}"
                  f"{int(r['hold']):>5}"
                  f"{r['wr']:>6.1f}%{r['avg_lev']:>+9.2f}%"
                  f"{r['sum_lev']:>+10.0f}%{int(r['n_liq']):>6}")


if __name__ == "__main__":
    grid_per_leverage()
