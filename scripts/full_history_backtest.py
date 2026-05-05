"""全历史信号回测 (无过滤, 5 策略并行, 3 时间段路由数据源).

时间段路由:
  阶段 1 (>1 年前):     日线简易 spot 视角 P&L (close-to-close)
  阶段 2 (近 1 年):     LEAPS 期权 (yfinance daily) + Binance 期货 daily
  阶段 3 (近 60 天):    kline_db 真实 EOD 期权 OHLC + Binance daily

5 策略并行:
  FUTURES_LONG: Binance XAUUSDT 20× perp, 5d/+3%/-2%
  BUY CALL:    ATM call (单腿), 45 DTE, +100%/-50%/expiry
  SELL PUT:    credit spread (-ATM/+-5%), 45 DTE, +50%/-50%/expiry
  STRADDLE:    long ATM call+put, 14 DTE, +100%/14d/expiry
  SHORT_VOL:   credit spread (近似 IC), 30 DTE, +50%/-50%/30d/expiry

输出: CSV per-strategy + 汇总报告 + 时段分类统计.
"""
from __future__ import annotations
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/yhdong/GoldDash")
sys.path.insert(0, "/Users/yhdong/Gold")

from core.data import load_features, load_config, load_oos_predictions
from core.signals import build_band, compute_rv_pctile
from core.signals_v2 import generate_daily_signals
from core.regime import RegimeClassifier
from core.events import detect_straddle_signal, detect_short_vol_signal
from core.paper_positions import (price_strategy_at, simulate_option_exit,
                                     pick_liquid_monthly_option,
                                     interpolate_option_intraday,
                                     _load_kline_db, bs_price)
from core.binance_futures import compute_liquidation_price, estimate_futures_pnl


def stage_for(date: pd.Timestamp, today: pd.Timestamp) -> str:
    """v3.7.112 重新划分:
    stage1 (>2y): 简易日线 spot 方向胜率 (只 FUTURES + 方向性 BC/SP win-only)
    stage2 (2y~90d): LEAPS 期权估价 (yfinance LEAPS daily 或 BS + GVZ 历史)
    stage3 (<90d):  kline_db EOD 真实期权 OHLC + 真退出规则
    """
    days = (today - date).days
    if days <= 90: return "stage3_kline_db"
    if days <= 730: return "stage2_leaps"
    return "stage1_spot_direction"


def simulate_futures_history(entry_d: pd.Timestamp, entry_spot_etf: float,
                                ratio: float, asset_csv: pd.DataFrame,
                                today: pd.Timestamp) -> dict:
    """期货多头模拟 (用 GC=F equivalent + Binance 模型规格)."""
    entry_gc = entry_spot_etf * ratio
    later = asset_csv.index[asset_csv.index > entry_d]
    if not len(later): return {"closed": False, "reason": "no later data"}
    hold = 0
    for d in later:
        if pd.Timestamp(d) > today: break
        hold += 1
        cur_etf = float(asset_csv.loc[d, "Close"])
        cur_gc = cur_etf * ratio
        ret_pct = (cur_gc / entry_gc - 1) * 100
        if ret_pct >= 3.0:
            return {"closed": True, "exit_date": pd.Timestamp(d),
                     "entry_gc": entry_gc, "exit_gc": cur_gc,
                     "ret_spot_pct": ret_pct,
                     "ret_levered_pct": ret_pct * 20,  # 20× leverage on margin
                     "reason": f"+{ret_pct:.2f}% TP",
                     "hold_days": hold,
                     "liq_price": compute_liquidation_price(entry_gc, 20)}
        if ret_pct <= -2.0:
            return {"closed": True, "exit_date": pd.Timestamp(d),
                     "entry_gc": entry_gc, "exit_gc": cur_gc,
                     "ret_spot_pct": ret_pct,
                     "ret_levered_pct": ret_pct * 20,
                     "reason": f"{ret_pct:.2f}% SL",
                     "hold_days": hold,
                     "liq_price": compute_liquidation_price(entry_gc, 20)}
        if hold >= 5:
            return {"closed": True, "exit_date": pd.Timestamp(d),
                     "entry_gc": entry_gc, "exit_gc": cur_gc,
                     "ret_spot_pct": ret_pct,
                     "ret_levered_pct": ret_pct * 20,
                     "reason": "5d 时间出场", "hold_days": hold,
                     "liq_price": compute_liquidation_price(entry_gc, 20)}
    return {"closed": False, "reason": "持仓中", "hold_days": hold}


def simulate_stage1_spot_direction(entry_d: pd.Timestamp, entry_spot: float,
                                       strategy: str, asset_csv: pd.DataFrame,
                                       today: pd.Timestamp) -> dict:
    """阶段 1 (>2 年): 简易 spot 方向胜率, NOT 期权 pnl.

    用户原话: "stage 1 一年以上的不是简易日线价格信号就可以了么? 哪来的 sell put?"
    实施: 只跑 FUTURES_LONG (spot move%) 和 方向性 BC/SP win-only (信号方向是否对).
          STRADDLE/SHORT_VOL 跳过 (没真实期权数据, 无意义模拟).

    持有 5 trading days, 看 spot close-to-close move.
    """
    s = strategy.upper()
    if s in ("STRADDLE", "SHORT_VOL"):
        return {"closed": False, "reason": "stage1 跳过 vol 策略"}
    later = asset_csv.index[asset_csv.index > entry_d]
    if len(later) < 5: return {"closed": False}
    exit_d = later[4]
    if pd.Timestamp(exit_d) > today: return {"closed": False}
    exit_spot = float(asset_csv.loc[exit_d, "Close"])
    move = (exit_spot / entry_spot - 1) * 100
    if s == "FUTURES_LONG" or s == "BUY CALL":
        # 上涨赢 / 下跌输. 期货按真实 move (杠杆另算), BC 信号方向胜率
        pnl = move
    elif s == "SELL PUT":
        # 横盘+上涨赢, 大跌输. spot 视角: 反 move 直觉
        pnl = -move  # spot 跌 = SELL PUT 输, spot 涨 = SELL PUT 赢 (信号方向)
    else:
        pnl = move
    return {"closed": True, "exit_date": pd.Timestamp(exit_d),
             "entry_spot": entry_spot, "exit_spot": exit_spot,
             "ret_spot_pct": move, "pnl_pct": pnl,
             "reason": "5d spot direction", "hold_days": 5}


def simulate_stage2_leaps(entry_d: pd.Timestamp, entry_spot: float,
                              strategy: str, asset_key: str,
                              asset_csv: pd.DataFrame,
                              today: pd.Timestamp,
                              gvz_df: pd.DataFrame = None) -> dict:
    """阶段 2 (2y~90d): LEAPS 期权估价 (BS + GVZ 历史 IV, yfinance LEAPS 历史 fallback).

    LEAPS DTE 远 (180-730d), theta 影响小, vega 主导.
    """
    s = strategy.upper()
    if s in ("FUTURES_LONG",):
        # 期货跟 stage1 同 spot move
        return simulate_stage1_spot_direction(entry_d, entry_spot, s, asset_csv, today)
    later = asset_csv.index[asset_csv.index > entry_d]
    if len(later) < 30:
        return {"closed": False, "reason": "stage2 数据不足"}
    # LEAPS hold 30d
    exit_d = later[29]
    if pd.Timestamp(exit_d) > today: return {"closed": False}
    exit_spot = float(asset_csv.loc[exit_d, "Close"])
    move_pct = (exit_spot / entry_spot - 1) * 100
    # GVZ 入场日 IV (估)
    iv_entry = 0.18
    if gvz_df is not None and entry_d in gvz_df.index:
        iv_entry = float(gvz_df.loc[entry_d, "Close"]) / 100
    iv_entry = max(0.10, min(0.40, iv_entry))
    # LEAPS DTE 365d, ATM
    T_e = 365 / 365.0
    T_x = (365 - 30) / 365.0
    K = round(entry_spot)
    if s == "BUY CALL":
        ec = bs_price(entry_spot, K, T_e, 0.04, iv_entry, "C")
        cc = bs_price(exit_spot, K, T_x, 0.04, iv_entry, "C")
        pnl = (cc / ec - 1) * 100 if ec > 0 else 0
    elif s == "SELL PUT":
        # LEAPS put credit spread, -ATM / +-5%
        K2 = round(entry_spot * 0.95)
        ep_s = bs_price(entry_spot, K, T_e, 0.04, iv_entry, "P")
        ep_l = bs_price(entry_spot, K2, T_e, 0.04, iv_entry, "P")
        ec_s = bs_price(exit_spot, K, T_x, 0.04, iv_entry, "P")
        ec_l = bs_price(exit_spot, K2, T_x, 0.04, iv_entry, "P")
        ent_credit = ep_s - ep_l
        cur_credit = ec_s - ec_l
        pnl = ((ent_credit - cur_credit) / ent_credit * 100) if ent_credit > 0 else 0
    elif s == "STRADDLE":
        ec = bs_price(entry_spot, K, T_e, 0.04, iv_entry, "C")
        ep = bs_price(entry_spot, K, T_e, 0.04, iv_entry, "P")
        cc = bs_price(exit_spot, K, T_x, 0.04, iv_entry, "C")
        cp = bs_price(exit_spot, K, T_x, 0.04, iv_entry, "P")
        ent_total = ec + ep
        cur_total = cc + cp
        pnl = (cur_total / ent_total - 1) * 100 if ent_total > 0 else 0
    elif s == "SHORT_VOL":
        # 反 STRADDLE
        ec = bs_price(entry_spot, K, T_e, 0.04, iv_entry, "C")
        ep = bs_price(entry_spot, K, T_e, 0.04, iv_entry, "P")
        cc = bs_price(exit_spot, K, T_x, 0.04, iv_entry, "C")
        cp = bs_price(exit_spot, K, T_x, 0.04, iv_entry, "P")
        pnl = ((ec + ep) / (cc + cp) - 1) * 100 if (cc + cp) > 0 else 0
    else:
        pnl = move_pct
    return {"closed": True, "exit_date": pd.Timestamp(exit_d),
             "entry_spot": entry_spot, "exit_spot": exit_spot,
             "ret_spot_pct": move_pct, "pnl_pct": pnl,
             "reason": f"LEAPS 30d (IV {iv_entry:.2f})", "hold_days": 30}


def simulate_option_stage23(entry_d: pd.Timestamp, entry_spot: float,
                                strategy: str, asset_key: str,
                                asset_csv: pd.DataFrame,
                                today: pd.Timestamp) -> dict:
    """阶段 2/3: 用 kline_db 真实期权 OHLC + 插值 + 真实退出规则."""
    if entry_d not in asset_csv.index: return {"closed": False}
    eO = float(asset_csv.loc[entry_d, "Open"])
    eC = float(asset_csv.loc[entry_d, "Close"])
    eH = float(asset_csv.loc[entry_d, "High"])
    eL = float(asset_csv.loc[entry_d, "Low"])
    ent = price_strategy_at(asset_key, strategy, entry_d,
                              entry_d + pd.Timedelta(hours=9, minutes=30),
                              entry_spot, eO, eC, eH, eL,
                              dte_target=(14 if strategy == "STRADDLE" else
                                           30 if strategy == "SHORT_VOL" else 45))
    if not ent.get("legs"):
        return {"closed": False, "reason": "kline_db 无期权"}
    if abs(ent.get("entry_price", 0)) < 0.01:
        return {"closed": False, "reason": "entry_value~0"}  # 防除零
    sim = simulate_option_exit(ent, entry_d, strategy, today)
    return {**sim, "entry_value": ent["entry_price"], "source": ent["source"]}


def main():
    cfg = load_config()
    features = load_features(cfg)
    today = pd.Timestamp.now().normalize()
    out_dir = Path("/Users/yhdong/Gold/data/backtest_history")
    out_dir.mkdir(exist_ok=True)
    # GVZ 黄金 VIX 历史 (用于 stage2 IV 估)
    try:
        import yfinance as yf
        gvz_df = yf.Ticker("^GVZ").history(period="5y")
        gvz_df.index = pd.to_datetime(gvz_df.index).tz_localize(None).normalize()
        print(f"GVZ 历史: {len(gvz_df)} 行 (用作 stage2 IV 估)")
    except Exception as e:
        gvz_df = None
        print(f"GVZ 拉失败: {e}")
    print(f"=== 全历史回测启动 (today={today.date()}) ===\n")

    for asset_key, csv_name in [("GLD", "gld.csv"), ("SLV", "slv.csv")]:
        print(f"\n=== {asset_key} ===")
        daily = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{csv_name}",
                            index_col=0, parse_dates=True)
        common = features.index.intersection(daily.index)
        close_d = daily["Close"][common]
        high_d = daily["High"][common]
        low_d = daily["Low"][common]
        oos = load_oos_predictions(cfg)
        upper, lower, _ = build_band(oos, close_d)
        rv_pct = compute_rv_pctile(features.loc[common, "rv_10d"])
        feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
        regime = RegimeClassifier().classify(features[feat_cols])["regime"]
        sig_df = generate_daily_signals(close_d, high_d, low_d, upper, lower,
                                           regime, rv_pct, asset=asset_key)
        # 检测 vol 信号
        rv_s = features.loc[close_d.index, "rv_10d"]
        strad_df = detect_straddle_signal(rv_s, sig_df.index, rv_pctile=rv_pct,
                                              asset=asset_key)
        sv_df = detect_short_vol_signal(rv_s, rv_pct, sig_df.index, regime=regime)

        # 收集所有信号 (无过滤)
        all_records = []
        # 假设 ratio = 10.97 (GLD) / 1.10 (SLV) — 历史 stable
        ratio = 10.97 if asset_key == "GLD" else 1.10
        n_total = 0; n_processed = 0
        for d, row in sig_df.iterrows():
            strats = []
            if d in strad_df.index and bool(strad_df.loc[d, "straddle_signal"]):
                strats.append("STRADDLE")
            if d in sv_df.index and bool(sv_df.loc[d, "short_vol_signal"]):
                strats.append("SHORT_VOL")
            # v3.7.113: buy_signal=True 时同时测 BUY CALL + SELL PUT 两次
            #   后续根据 RV/IV 优化决定哪个最优
            #   FUTURES_LONG 也并行
            if row.get("buy_signal", False):
                strats.extend(["BUY CALL", "SELL PUT", "FUTURES_LONG"])
            # v3.7.113: SHORT_VOL 屏蔽 (用户暂时关 — IC 4-leg 待真实模型)
            if "SHORT_VOL" in strats:
                strats.remove("SHORT_VOL")
            if not strats: continue
            entry_spot = float(close_d.get(d, 0))
            if entry_spot <= 0: continue
            stage = stage_for(d, today)
            for strat in strats:
                n_total += 1
                if "FUTURES" in strat:
                    res = simulate_futures_history(d, entry_spot, ratio, daily, today)
                elif stage == "stage1_spot_direction":
                    res = simulate_stage1_spot_direction(d, entry_spot, strat, daily, today)
                elif stage == "stage2_leaps":
                    res = simulate_stage2_leaps(d, entry_spot, strat, asset_key, daily, today, gvz_df)
                else:  # stage3_kline_db
                    res = simulate_option_stage23(d, entry_spot, strat, asset_key, daily, today)
                # v3.7.114: stage3 OPEN 但 mark-to-market 也算入 grid (近 90d 多数还在持仓)
                if not res.get("closed"):
                    if stage == "stage3_kline_db" and "pnl_pct" in res:
                        res["closed"] = True
                        res["reason"] = "MTM (持仓中)"
                    else:
                        continue
                # v3.7.114: 加 raw RV (10d annualized %) + GVZ IV 列, 给 grid IV-RV gap
                _raw_rv = float(rv_s.get(d, 0)) if d in rv_s.index else 0
                _gvz_iv = (float(gvz_df.loc[d, "Close"]) if (gvz_df is not None
                            and d in gvz_df.index) else 0)
                rec = {
                    "asset": asset_key,
                    "signal_date": d,
                    "strategy": strat,
                    "stage": stage,
                    "regime": regime.get(d, "?"),
                    "rv_pctile": float(rv_pct.get(d, 0)),
                    "rv_10d_pct": _raw_rv,        # raw RV % annualized
                    "gvz_iv_pct": _gvz_iv,         # GVZ IV %
                    "iv_rv_gap_pct": _gvz_iv - _raw_rv,  # IV-RV (>0 = SP 优)
                    "entry_spot_etf": entry_spot,
                    "entry_spot_gc": entry_spot * ratio,
                    "exit_date": res.get("exit_date"),
                    "hold_days": res.get("hold_days", 0),
                    "exit_reason": res.get("reason") or res.get("exit_reason", ""),
                    "pnl_pct": res.get("pnl_pct", res.get("ret_spot_pct", 0)),
                }
                if "FUTURES" in strat:
                    rec["levered_pnl_pct"] = res.get("ret_levered_pct", 0)
                    rec["liq_price"] = res.get("liq_price", 0)
                all_records.append(rec)
                n_processed += 1
                if n_total % 200 == 0:
                    print(f"  [{n_total}] processed {n_processed} closed records...")

        df_out = pd.DataFrame(all_records)
        out_file = out_dir / f"backtest_{asset_key.lower()}_{today.strftime('%Y%m%d')}.csv"
        df_out.to_csv(out_file, index=False)
        print(f"\n{asset_key} 全历史回测完成: {len(df_out)} 笔已平 / {n_total} 笔信号")
        print(f"保存: {out_file}")

        # 汇总
        if len(df_out):
            print(f"\n按策略汇总:")
            for strat, sub in df_out.groupby("strategy"):
                wins = (sub["pnl_pct"] > 0).sum()
                wr = wins / len(sub) * 100
                avg = sub["pnl_pct"].mean()
                std = sub["pnl_pct"].std()
                cum = sub["pnl_pct"].sum()
                print(f"  {strat:<12} n={len(sub):<4} wr={wr:>5.1f}% "
                      f"avg={avg:>+6.2f}% std={std:>5.2f}% cum={cum:>+8.1f}%")
            print(f"\n按时段汇总:")
            for stage, sub in df_out.groupby("stage"):
                wins = (sub["pnl_pct"] > 0).sum()
                wr = wins / len(sub) * 100
                avg = sub["pnl_pct"].mean()
                print(f"  {stage:<22} n={len(sub):<5} wr={wr:>5.1f}% avg={avg:>+5.2f}%")


if __name__ == "__main__":
    main()
