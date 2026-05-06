"""比较旧 (simulate_option_exit) vs 新 (options_exit.simulate_bc/sp_exit) BC/SP 退出规则.

旧规则: +100%/+50% TP, -50% SL, expiry
新规则: + DTE-cliff 强平 (BC<14, SP<7) + signal-reversal + strike-defense
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from core.paper_positions import (
    simulate_option_exit, price_strategy_at, _load_kline_db
)
from core.strategies.options_exit import (
    simulate_bc_exit, simulate_sp_exit, BCExitConfig, SPExitConfig
)

CSV = Path("/Users/yhdong/Gold/data/backtest_history")
TODAY = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())


def compare(asset):
    print(f"\n{'='*70}")
    print(f"{asset} BC/SP — 旧 vs 新 exit 规则对比")
    print(f"{'='*70}")
    df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                      parse_dates=["signal_date"])
    df = df[df["stage"].isin(["stage2_main_3m", "stage2_leaps_aux"])]
    sigs = df[df["strategy"]=="BUY CALL"][["signal_date","entry_spot_etf",
                                              "bp_high","bp_close","bp_low"]].drop_duplicates("signal_date")

    # 加载 daily bp_high 序列 (从 backtest CSV 提取 + 全 close 数据)
    ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                        index_col=0, parse_dates=True)
    spot_series = ohlc["Close"]
    # bp_high 用 backtest CSV 内 signal_date 的 bp_high 即可 (生产里有完整 sig_df.bp_high)
    bp_high_series = sigs.set_index("signal_date")["bp_high"]

    db = _load_kline_db()
    if db is None:
        print(f"  kline_db 未加载, 跳过"); return

    for strat in ["BUY CALL", "SELL PUT"]:
        print(f"\n--- {strat} ---")
        old_pnls = []; new_pnls = []
        for _, r in sigs.iterrows():
            d = pd.Timestamp(r["signal_date"]).normalize()
            spot = r["entry_spot_etf"]
            # entry pricing
            try:
                ent = price_strategy_at(asset, strat, d,
                                            d + pd.Timedelta(hours=9, minutes=30),
                                            spot, spot, spot, spot, spot,
                                            dte_target=45)
            except Exception:
                continue
            if not ent.get("legs"): continue
            if abs(ent.get("entry_price", 0)) < 0.01: continue
            ev = ent["entry_price"]
            legs = ent["legs"]
            # 取 expiry
            first_kdb = db[db["code"] == legs[0][1]]
            if not len(first_kdb): continue
            expiry_dt = pd.Timestamp(first_kdb.iloc[0]["expiry"])
            # 旧
            try:
                old_res = simulate_option_exit(ent, d, strat, TODAY)
                if old_res.get("is_closed") and "pnl_pct" in old_res:
                    old_pnls.append(old_res["pnl_pct"])
            except Exception: pass
            # 新
            try:
                if strat == "BUY CALL":
                    new_res = simulate_bc_exit(ev, legs, d, expiry_dt, TODAY, db,
                                                  bp_high_series=bp_high_series)
                else:
                    # SP 调参: 关 strike_defense (buffer→1.50 等同关闭),
                    #         DTE-cliff 7→14 (更晚强平)
                    sp_cfg = SPExitConfig(
                        dte_cliff_days=14,
                        strike_defense_buffer=1.50,  # 实质禁用
                    )
                    new_res = simulate_sp_exit(ev, legs, d, expiry_dt, TODAY, db,
                                                  bp_high_series=bp_high_series,
                                                  spot_series=spot_series,
                                                  cfg=sp_cfg)
                if new_res.get("is_closed") and "pnl_pct" in new_res:
                    new_pnls.append(new_res["pnl_pct"])
            except Exception as e:
                pass
        if not old_pnls or not new_pnls: continue
        op = pd.Series(old_pnls); np_ = pd.Series(new_pnls)
        print(f"  旧规则 (n={len(op)}): wr={(op>0).mean()*100:5.1f}% "
              f"avg={op.mean():+7.2f}% sum={op.sum():+8.1f}%")
        print(f"  新规则 (n={len(np_)}): wr={(np_>0).mean()*100:5.1f}% "
              f"avg={np_.mean():+7.2f}% sum={np_.sum():+8.1f}%")
        print(f"  Δ wr: {(np_>0).mean()*100 - (op>0).mean()*100:+.1f}pp,  "
              f"Δ sum: {np_.sum() - op.sum():+.1f}%")


for asset in ["GLD", "SLV"]:
    compare(asset)
