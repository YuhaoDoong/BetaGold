"""期权盈亏计算 — 用 EOD 快照的实际价格.

两套盈亏:
  1. 金价收益: (exit_price / entry_price - 1) × 100
  2. 期权收益: 用 EOD 快照中 ATM 期权的实际 mid 价格计算

有快照日 → 用真实价格; 无快照 → 返回 None.
"""

import pandas as pd
import numpy as np


def find_atm_option(eod_df, spot, option_type, dte_range=(14, 45)):
    """从 EOD 快照中找 ATM 附近最优合约.

    Returns: dict with strike, mid, delta, iv, dte, oi, or None
    """
    if eod_df is None or len(eod_df) == 0:
        return None

    mask = (
        (eod_df["option_type"] == option_type) &
        (eod_df["option_strike_price"].between(spot - 10, spot + 10)) &
        (eod_df["dte"].between(dte_range[0], dte_range[1])) &
        (eod_df["bid_price"] > 0)
    )
    if "option_open_interest" in eod_df.columns:
        mask = mask & (eod_df["option_open_interest"] >= 50)

    opts = eod_df[mask]
    if len(opts) == 0:
        # 放宽范围
        mask2 = (
            (eod_df["option_type"] == option_type) &
            (eod_df["option_strike_price"].between(spot - 20, spot + 20)) &
            (eod_df["dte"].between(5, 60)) &
            (eod_df["bid_price"] > 0)
        )
        opts = eod_df[mask2]
        if len(opts) == 0:
            return None

    oi_col = "option_open_interest" if "option_open_interest" in opts.columns else None
    best = opts.nlargest(1, oi_col).iloc[0] if oi_col else opts.iloc[0]
    mid = (best["bid_price"] + best["ask_price"]) / 2

    return {
        "strike": best["option_strike_price"],
        "mid": mid,
        "delta": best.get("option_delta", 0),
        "iv": best.get("option_implied_volatility", 0),
        "dte": best.get("dte", 0),
        "oi": best.get("option_open_interest", 0),
        "code": best.get("code", ""),
    }


def compute_options_pnl(trades, snapshots, close):
    """为每笔交易计算期权盈亏.

    Args:
        trades: list of trade dicts (from run_backtest)
        snapshots: {pd.Timestamp: eod_df} 所有快照
        close: GLD close Series

    Returns: trades with added fields:
        opt_entry_price, opt_exit_price, opt_pnl_pct, opt_source
    """
    snap_dates = sorted(snapshots.keys())

    def _nearest_snap(d, max_days=3):
        """找距 d 最近的快照 (前后3天内)."""
        best = None
        best_diff = 999
        for sd in snap_dates:
            diff = abs((sd - d).days)
            if diff <= max_days and diff < best_diff:
                best = sd
                best_diff = diff
        return best

    results = []
    for t in trades:
        t = dict(t)  # copy

        entry_d = t["entry_date"]
        exit_d = t["exit_date"]
        entry_spot = close.get(entry_d, t.get("entry_price", 0))

        # 入场快照
        entry_snap_d = _nearest_snap(entry_d)
        exit_snap_d = _nearest_snap(exit_d)

        if entry_snap_d is not None:
            entry_eod = snapshots[entry_snap_d]
            # 根据信号类型选择期权
            sig_type = t.get("type", "BUY CALL")
            if "CALL" in sig_type:
                opt = find_atm_option(entry_eod, entry_spot, "CALL")
            elif "PUT" in sig_type:
                opt = find_atm_option(entry_eod, entry_spot, "PUT")
            else:
                opt = find_atm_option(entry_eod, entry_spot, "CALL")

            if opt:
                t["opt_entry_price"] = opt["mid"]
                t["opt_entry_strike"] = opt["strike"]
                t["opt_entry_delta"] = opt["delta"]
                t["opt_entry_iv"] = opt["iv"]

                # 估算退出时的期权价格 (简化: delta × 金价变动 + theta 衰减)
                gold_move = t.get("exit_price", entry_spot) - entry_spot
                days_held = t.get("hold_days", 1)
                theta = opt.get("theta", -opt["mid"] / max(opt["dte"], 1))
                # theta 不在快照里, 估算: daily decay ≈ mid / dte
                if "option_theta" in entry_eod.columns:
                    theta = entry_eod.loc[
                        entry_eod["option_open_interest"] == opt["oi"],
                        "option_theta"
                    ]
                    theta = theta.iloc[0] if len(theta) > 0 else -opt["mid"] / max(opt["dte"], 1)

                opt_exit = opt["mid"] + abs(opt["delta"]) * gold_move + theta * days_held
                opt_exit = max(opt_exit, 0)  # 不低于0

                # 如果有退出快照, 用实际价格
                if exit_snap_d is not None and exit_snap_d != entry_snap_d:
                    exit_eod = snapshots[exit_snap_d]
                    exit_opt = find_atm_option(exit_eod,
                        close.get(exit_d, entry_spot),
                        "CALL" if "CALL" in sig_type else "PUT")
                    if exit_opt and exit_opt["strike"] == opt["strike"]:
                        opt_exit = exit_opt["mid"]
                        t["opt_source"] = "实际"
                    else:
                        t["opt_source"] = "估算(δ)"
                else:
                    t["opt_source"] = "估算(δ)"

                t["opt_exit_price"] = opt_exit
                t["opt_pnl_pct"] = (opt_exit / opt["mid"] - 1) * 100
            else:
                t["opt_entry_price"] = None
                t["opt_pnl_pct"] = None
                t["opt_source"] = None
        else:
            t["opt_entry_price"] = None
            t["opt_pnl_pct"] = None
            t["opt_source"] = None

        results.append(t)

    return results


def compute_straddle_pnl(straddle_trades, snapshots, close):
    """为 Straddle 交易计算期权实际成本.

    Args:
        straddle_trades: from backtest_straddle()

    Returns: trades with opt_cost, opt_pnl_pct
    """
    snap_dates = sorted(snapshots.keys())

    def _nearest(d, max_days=3):
        best = None
        for sd in snap_dates:
            if abs((sd - d).days) <= max_days:
                best = sd
        return best

    results = []
    for t in straddle_trades:
        t = dict(t)
        entry_d = t["entry_date"]
        spot = t["entry_price"]

        snap_d = _nearest(entry_d)
        if snap_d is not None:
            eod = snapshots[snap_d]
            call = find_atm_option(eod, spot, "CALL")
            put = find_atm_option(eod, spot, "PUT")

            if call and put:
                real_cost = call["mid"] + put["mid"]
                real_cost_pct = real_cost / spot * 100
                real_pnl = t["max_move"] - real_cost_pct
                t["opt_cost"] = real_cost
                t["opt_cost_pct"] = real_cost_pct
                t["opt_pnl_pct"] = real_pnl
                t["opt_source"] = "EOD"
            else:
                t["opt_cost"] = None
                t["opt_pnl_pct"] = None
                t["opt_source"] = None
        else:
            t["opt_cost"] = None
            t["opt_pnl_pct"] = None
            t["opt_source"] = None

        results.append(t)

    return results
