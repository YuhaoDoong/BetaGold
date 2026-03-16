"""期权策略推荐模块.

BUY CALL → 单腿 Call + Bull Call Spread, 含推荐 + 理由
SELL PUT → Bull Put Spread (卖出看跌价差)
所有 ROI 计算均含5日 theta 衰减
"""

import numpy as np
import pandas as pd


def find_options(eod_df, option_type, strike_range, dte_range):
    """从 EOD 快照中筛选期权合约."""
    if eod_df is None or len(eod_df) == 0:
        return pd.DataFrame()
    mask = (
        (eod_df["option_type"] == option_type) &
        (eod_df["option_strike_price"] >= strike_range[0]) &
        (eod_df["option_strike_price"] <= strike_range[1]) &
        (eod_df["dte"] >= dte_range[0]) &
        (eod_df["dte"] <= dte_range[1]) &
        (eod_df["bid_price"] > 0) &
        (eod_df["option_open_interest"] >= 50)
    )
    df = eod_df[mask].copy()
    df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    return df.sort_values(["dte", "option_strike_price"])


def _find_spread_leg(eod_df, anchor_row, option_type, strike_offset):
    """找到同到期日、指定 strike 偏移的合约 (价差另一腿)."""
    target_strike = anchor_row["option_strike_price"] + strike_offset

    mask = (
        (eod_df["option_type"] == option_type) &
        (eod_df["bid_price"] > 0)
    )
    if "strike_time" in eod_df.columns:
        exp_str = str(anchor_row.get("strike_time", ""))[:10]
        if exp_str:
            mask = mask & (
                eod_df["strike_time"].astype(str).str[:10] == exp_str)
    else:
        mask = mask & (eod_df["dte"] == anchor_row["dte"])

    candidates = eod_df[mask].copy()
    if len(candidates) == 0:
        return None

    candidates["_dist"] = abs(
        candidates["option_strike_price"] - target_strike)
    candidates = candidates[candidates["_dist"] <= 2]
    if len(candidates) == 0:
        return None

    best = candidates.nsmallest(1, "_dist").iloc[0].copy()
    best["mid"] = (best["bid_price"] + best["ask_price"]) / 2
    return best


# ══════════════════════════════════════════════════════════
# 对外接口
# ══════════════════════════════════════════════════════════

def get_strategy_table(signal_type, gld_price, exit_price, eod_df):
    """获取期权策略推荐.

    exit_price: 平仓触发价位 (bp=0.90)

    Returns dict:
        BUY_CALL → {"single_leg": [...], "spread": [...], "rec": "..."}
        SELL_PUT → {"spread": [...], "rec": None}
    """
    if eod_df is None or signal_type not in ("BUY_CALL", "SELL_PUT"):
        return {"spread": [], "rec": None}

    price_move = exit_price - gld_price
    hold_days = 5

    if signal_type == "BUY_CALL":
        single_leg = _build_call_table(
            eod_df, gld_price, price_move, hold_days)
        spread = _build_call_spread_table(
            eod_df, gld_price, price_move, hold_days)
        rec = _build_call_recommendation(
            eod_df, gld_price, price_move, hold_days, exit_price)
        return {"single_leg": single_leg, "spread": spread, "rec": rec}
    else:
        spread = _build_put_spread_table(
            eod_df, gld_price, price_move, hold_days)
        return {"spread": spread, "rec": None}


# ══════════════════════════════════════════════════════════
# BUY CALL — 单腿 Call
# ══════════════════════════════════════════════════════════

def _build_call_table(eod_df, gld_price, price_move, hold_days):
    configs = [
        ("A. 稳健 (ITM)",
         (gld_price - 20, gld_price - 5), (25, 45),
         "Delta≈0.70, theta较低"),
        ("B. 中性 (ATM)",
         (gld_price - 5, gld_price + 5), (17, 35),
         "Delta≈0.50, 平衡"),
        ("C. 激进 (OTM)",
         (gld_price + 5, gld_price + 20), (14, 28),
         "Delta≈0.30, 高杠杆"),
    ]

    rows = []
    for label, strike_range, dte_range, desc in configs:
        opts = find_options(eod_df, "CALL", strike_range, dte_range)
        if len(opts) == 0:
            rows.append({
                "策略": label, "合约": "—", "成本": "—",
                "Δ": "—", "Θ/日": "—",
                "5日ROI": "—", "OI": "—", "说明": desc,
            })
            continue

        best = opts.nlargest(1, "option_open_interest").iloc[0]
        exp = pd.Timestamp(best["strike_time"]).strftime("%m/%d")
        strike = best["option_strike_price"]
        mid = best["mid"]
        delta = abs(best.get("option_delta", 0))
        theta = best.get("option_theta", 0)
        oi = best["option_open_interest"]

        pnl = delta * price_move + theta * hold_days
        roi = pnl / mid * 100 if mid > 0 else 0

        rows.append({
            "策略": label,
            "合约": f"GLD {exp} ${strike:.0f}C",
            "成本": f"${mid:.2f}",
            "Δ": f"{delta:.2f}",
            "Θ/日": f"${theta:.2f}",
            "5日ROI": f"{roi:+.0f}%",
            "OI": f"{oi:,.0f}",
            "说明": desc,
        })

    return rows


# ══════════════════════════════════════════════════════════
# BUY CALL — Bull Call Spread (牛市看涨价差)
# ══════════════════════════════════════════════════════════

def _build_call_spread_table(eod_df, gld_price, price_move, hold_days):
    """买近ATM + 卖OTM, 三档宽度."""
    widths = [
        ("A. 窄幅 ($10)", 10, "theta对冲高, 盈利上限$10"),
        ("B. 中幅 ($15)", 15, "平衡theta与收益"),
        ("C. 宽幅 ($20)", 20, "收益空间大, 成本更低"),
    ]

    rows = []
    for label, width, desc in widths:
        buy_opts = find_options(
            eod_df, "CALL",
            (gld_price - 5, gld_price + 5), (17, 45))
        if len(buy_opts) == 0:
            rows.append(_empty_call_spread_row(label, desc))
            continue

        buy = buy_opts.nlargest(1, "option_open_interest").iloc[0]
        sell = _find_spread_leg(eod_df, buy, "CALL", +width)
        if sell is None:
            rows.append(_empty_call_spread_row(label, desc))
            continue

        buy_strike = buy["option_strike_price"]
        sell_strike = sell["option_strike_price"]
        actual_width = sell_strike - buy_strike
        if actual_width <= 0:
            rows.append(_empty_call_spread_row(label, desc))
            continue

        buy_mid = buy["mid"]
        sell_mid = sell["mid"]
        net_debit = buy_mid - sell_mid
        if net_debit <= 0:
            rows.append(_empty_call_spread_row(label, desc))
            continue

        max_profit = actual_width - net_debit

        # Position Greeks: long buy leg + short sell leg
        buy_delta = abs(buy.get("option_delta", 0))
        sell_delta = abs(sell.get("option_delta", 0))
        buy_theta = buy.get("option_theta", 0)   # negative
        sell_theta = sell.get("option_theta", 0)  # negative, less so

        pos_delta = buy_delta - sell_delta         # positive (bullish)
        pos_theta = buy_theta - sell_theta         # less negative

        pnl = pos_delta * price_move + pos_theta * hold_days
        pnl = min(pnl, max_profit)  # cap at max profit
        roi = pnl / net_debit * 100 if net_debit > 0 else 0

        exp = pd.Timestamp(buy["strike_time"]).strftime("%m/%d")
        contract = f"GLD {exp} ${buy_strike:.0f}/${sell_strike:.0f}C"

        rows.append({
            "策略": label,
            "价差合约": contract,
            "净成本": f"${net_debit:.2f}",
            "最大盈利": f"${max_profit:.2f}",
            "Pos Δ": f"{pos_delta:.2f}",
            "Pos Θ/日": f"${pos_theta:.2f}",
            "5日ROI": f"{roi:+.0f}%",
            "说明": desc,
        })

    return rows


def _empty_call_spread_row(label, desc):
    return {
        "策略": label, "价差合约": "—", "净成本": "—",
        "最大盈利": "—", "Pos Δ": "—", "Pos Θ/日": "—",
        "5日ROI": "—", "说明": desc,
    }


# ══════════════════════════════════════════════════════════
# BUY CALL — 推荐逻辑
# ══════════════════════════════════════════════════════════

def _build_call_recommendation(eod_df, gld_price, price_move,
                                hold_days, exit_price):
    """基于 IV / theta 比较, 给出单腿 vs 价差推荐."""
    atm_opts = find_options(
        eod_df, "CALL", (gld_price - 3, gld_price + 3), (17, 45))
    if len(atm_opts) == 0:
        return None

    atm = atm_opts.nlargest(1, "option_open_interest").iloc[0]
    iv = atm.get("iv_decimal", 0) * 100
    theta = atm.get("option_theta", 0)
    mid = atm["mid"]
    theta_pct = abs(theta) / mid * 100 if mid > 0 else 0
    theta_5d = abs(theta) * hold_days

    # 参考 $15 价差的 theta 对冲效果
    sell_leg = _find_spread_leg(eod_df, atm, "CALL", 15)
    if sell_leg is not None:
        net_theta = theta - sell_leg.get("option_theta", 0)
        theta_saved_pct = (1 - abs(net_theta) / max(abs(theta), 0.01)) * 100
        net_theta_5d = abs(net_theta) * hold_days
    else:
        theta_saved_pct = 0
        net_theta_5d = theta_5d

    if iv >= 28:
        title = "推荐: Bull Call Spread (牛市看涨价差)"
        reason = (
            f"IV={iv:.0f}%偏高, 单腿theta ${abs(theta):.2f}/日 "
            f"({theta_pct:.1f}%/日), 5日损耗${theta_5d:.1f}。"
            f"价差对冲{theta_saved_pct:.0f}%的theta "
            f"(净损耗${net_theta_5d:.1f}/5日)。"
            f"退出价${exit_price:.0f}高于价差卖腿, "
            f"单腿在极端上涨时收益更高, 但theta代价大"
        )
    elif iv <= 20:
        title = "推荐: Long Call (单腿)"
        reason = (
            f"IV={iv:.0f}%较低, theta仅${abs(theta):.2f}/日 "
            f"({theta_pct:.1f}%/日), 5日损失${theta_5d:.1f}可接受。"
            f"单腿保留涨至退出价${exit_price:.0f}的完整收益空间, "
            f"操作简单"
        )
    else:
        title = "单腿 / 价差均可"
        reason = (
            f"IV={iv:.0f}%, theta ${abs(theta):.2f}/日 "
            f"({theta_pct:.1f}%/日)。"
            f"看好涨至${exit_price:.0f}选单腿 (无上限); "
            f"重视theta控制选价差 (对冲{theta_saved_pct:.0f}%)"
        )

    return f"**{title}**\n\n{reason}"


# ══════════════════════════════════════════════════════════
# SELL PUT → Bull Put Spread (卖出看跌价差)
# ══════════════════════════════════════════════════════════

def _build_put_spread_table(eod_df, gld_price, price_move, hold_days):
    spread_width = 5

    configs = [
        ("A. 稳健 (远OTM价差)",
         (gld_price * 0.88, gld_price * 0.93), (25, 45),
         "|Δ|≈0.10, 极低行权概率"),
        ("B. 中性 (OTM价差)",
         (gld_price * 0.93, gld_price * 0.97), (17, 35),
         "|Δ|≈0.25, 适中权利金"),
        ("C. 激进 (近ATM价差)",
         (gld_price * 0.97, gld_price + 2), (14, 28),
         "|Δ|≈0.40, 高权利金"),
    ]

    rows = []
    for label, strike_range, dte_range, desc in configs:
        sell_opts = find_options(eod_df, "PUT", strike_range, dte_range)
        if len(sell_opts) == 0:
            rows.append(_empty_put_spread_row(label, desc))
            continue

        sell = sell_opts.nlargest(1, "option_open_interest").iloc[0]
        buy = _find_spread_leg(eod_df, sell, "PUT", -spread_width)
        if buy is None:
            rows.append(_empty_put_spread_row(label, desc))
            continue

        sell_strike = sell["option_strike_price"]
        buy_strike = buy["option_strike_price"]
        actual_width = sell_strike - buy_strike
        sell_mid = sell["mid"]
        buy_mid = buy["mid"]
        net_credit = sell_mid - buy_mid

        if net_credit <= 0 or actual_width <= 0:
            rows.append(_empty_put_spread_row(label, desc))
            continue

        max_loss = actual_width - net_credit

        sell_delta = sell.get("option_delta", 0)
        buy_delta = buy.get("option_delta", 0)
        sell_theta = sell.get("option_theta", 0)
        buy_theta = buy.get("option_theta", 0)

        pos_delta = -sell_delta + buy_delta
        pos_theta = -sell_theta + buy_theta

        pnl = pos_delta * price_move + pos_theta * hold_days
        roi = pnl / max_loss * 100 if max_loss > 0 else 0

        exp = pd.Timestamp(sell["strike_time"]).strftime("%m/%d")
        contract = f"GLD {exp} ${sell_strike:.0f}/${buy_strike:.0f}P"

        rows.append({
            "策略": label,
            "价差合约": contract,
            "净权利金": f"${net_credit:.2f}",
            "最大亏损": f"${max_loss:.2f}",
            "Pos Δ": f"{pos_delta:.3f}",
            "Pos Θ/日": f"${pos_theta:.3f}",
            "5日ROI": f"{roi:+.0f}%",
            "说明": desc,
        })

    return rows


def _empty_put_spread_row(label, desc):
    return {
        "策略": label, "价差合约": "—", "净权利金": "—",
        "最大亏损": "—", "Pos Δ": "—", "Pos Θ/日": "—",
        "5日ROI": "—", "说明": desc,
    }
