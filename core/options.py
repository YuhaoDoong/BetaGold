"""期权策略推荐模块 v2.

改进:
  - 优先用 Moomoo 实时报价 (盘中), 无法获取时降级到 EOD 快照
  - 每个策略同时展示最大盈利和最大亏损
  - 推荐止损点 (基于 GLD 价格回撤)
  - ROI 计算含5日 theta 衰减
"""

import numpy as np
import pandas as pd


def _get_live_snapshot(spot, expiry_start=None, expiry_end=None):
    """尝试从 Moomoo 获取实时期权报价."""
    try:
        from core.data import fetch_live_options
        snap = fetch_live_options(spot, expiry_start, expiry_end)
        if snap is not None and len(snap) > 0:
            snap["mid"] = (snap["bid_price"] + snap["ask_price"]) / 2
            # Moomoo 原生列已有 option_type, option_strike_price, strike_time 等
            # 只需补 dte (从 option_expiry_date_distance 或 strike_time 计算)
            if "option_expiry_date_distance" in snap.columns:
                snap["dte"] = snap["option_expiry_date_distance"]
            elif "strike_time" in snap.columns:
                snap["dte"] = (pd.to_datetime(snap["strike_time"].astype(str).str[:10])
                               - pd.Timestamp.now().normalize()).dt.days
            if "iv_decimal" not in snap.columns and "option_implied_volatility" in snap.columns:
                snap["iv_decimal"] = snap["option_implied_volatility"] / 100
            return snap, "LIVE"
    except Exception:
        pass
    return None, None


def find_options(eod_df, option_type, strike_range, dte_range):
    """从快照中筛选期权合约."""
    if eod_df is None or len(eod_df) == 0:
        return pd.DataFrame()
    mask = (
        (eod_df["option_type"] == option_type) &
        (eod_df["option_strike_price"] >= strike_range[0]) &
        (eod_df["option_strike_price"] <= strike_range[1]) &
        (eod_df["dte"] >= dte_range[0]) &
        (eod_df["dte"] <= dte_range[1]) &
        (eod_df["bid_price"] > 0)
    )
    # OI 过滤 (live 数据可能 OI 字段不同)
    if "option_open_interest" in eod_df.columns:
        mask = mask & (eod_df["option_open_interest"] >= 50)
    df = eod_df[mask].copy()
    if "mid" not in df.columns:
        df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    return df.sort_values(["dte", "option_strike_price"])


def _find_spread_leg(eod_df, anchor_row, option_type, strike_offset):
    """找到同到期日、指定 strike 偏移的合约."""
    target_strike = anchor_row["option_strike_price"] + strike_offset
    mask = (eod_df["option_type"] == option_type) & (eod_df["bid_price"] > 0)

    if "strike_time" in eod_df.columns:
        exp_str = str(anchor_row.get("strike_time", ""))[:10]
        if exp_str:
            mask = mask & (eod_df["strike_time"].astype(str).str[:10] == exp_str)
    else:
        mask = mask & (eod_df["dte"] == anchor_row["dte"])

    candidates = eod_df[mask].copy()
    if len(candidates) == 0:
        return None

    candidates["_dist"] = abs(candidates["option_strike_price"] - target_strike)
    candidates = candidates[candidates["_dist"] <= 3]
    if len(candidates) == 0:
        return None

    best = candidates.nsmallest(1, "_dist").iloc[0].copy()
    if "mid" not in best.index:
        best["mid"] = (best["bid_price"] + best["ask_price"]) / 2
    return best


SINGLE_STOP_PCT = 50  # 单腿止损: 权利金亏损 N%


# ══════════════════════════════════════════════════════════
# 对外接口
# ══════════════════════════════════════════════════════════

def get_strategy_table(signal_type, gld_price, exit_price, eod_df,
                       use_live=True):
    """获取期权策略推荐.

    优先用 Moomoo 实时报价, 降级到 EOD 快照.
    """
    if signal_type not in ("BUY_CALL", "SELL_PUT"):
        return {"spread": [], "rec": None, "source": "none"}

    # 尝试实时数据
    data_df = eod_df
    source = "EOD"
    if use_live:
        live, src = _get_live_snapshot(gld_price)
        if live is not None and len(live) > 0:
            data_df = live
            source = "LIVE"

    if data_df is None:
        return {"spread": [], "rec": None, "source": source}

    price_move = exit_price - gld_price
    hold_days = 5

    if signal_type == "BUY_CALL":
        single_leg = _build_call_table(data_df, gld_price, price_move, hold_days)
        spread = _build_call_spread_table(data_df, gld_price, price_move, hold_days)
        rec = _build_call_recommendation(
            data_df, gld_price, price_move, hold_days, exit_price)
        return {"single_leg": single_leg, "spread": spread,
                "rec": rec, "source": source}
    else:
        spread = _build_put_spread_table(data_df, gld_price, price_move, hold_days)
        return {"spread": spread, "rec": None, "source": source}


# ══════════════════════════════════════════════════════════
# BUY CALL — 单腿
# ══════════════════════════════════════════════════════════

def _build_call_table(eod_df, gld_price, price_move, hold_days):
    configs = [
        ("A. 稳健 (ITM)", (gld_price - 20, gld_price - 5), (25, 45),
         "Delta≈0.70, theta较低"),
        ("B. 中性 (ATM)", (gld_price - 5, gld_price + 5), (17, 35),
         "Delta≈0.50, 平衡"),
        ("C. 激进 (OTM)", (gld_price + 5, gld_price + 20), (14, 28),
         "Delta≈0.30, 高杠杆"),
    ]

    rows = []
    for label, strike_range, dte_range, desc in configs:
        opts = find_options(eod_df, "CALL", strike_range, dte_range)
        if len(opts) == 0:
            rows.append({"策略": label, "合约": "—", "成本": "—",
                         "Δ": "—", "Θ/日": "—", "5日盈利": "—",
                         "最大亏损": "—", "止损": "—"})
            continue

        oi_col = "option_open_interest" if "option_open_interest" in opts.columns else None
        best = opts.nlargest(1, oi_col).iloc[0] if oi_col else opts.iloc[0]
        exp = pd.Timestamp(best["strike_time"]).strftime("%m/%d")
        strike = best["option_strike_price"]
        mid = best["mid"]
        delta = abs(best.get("option_delta", 0))
        theta = best.get("option_theta", 0)

        pnl = delta * price_move + theta * hold_days
        roi = pnl / mid * 100 if mid > 0 else 0
        max_loss = mid
        stop_price = mid * (1 - SINGLE_STOP_PCT / 100)

        rows.append({
            "策略": label,
            "合约": f"GLD {exp} ${strike:.0f}C",
            "成本": f"${mid:.2f}",
            "Δ": f"{delta:.2f}",
            "Θ/日": f"${theta:.2f}",
            "5日盈利": f"${pnl:.2f} ({roi:+.0f}%)",
            "最大亏损": f"-${max_loss:.2f}",
            "止损": f"期权跌至${stop_price:.2f} (-{SINGLE_STOP_PCT}%)",
        })
    return rows


# ══════════════════════════════════════════════════════════
# BUY CALL — Bull Call Spread
# ══════════════════════════════════════════════════════════

def _build_call_spread_table(eod_df, gld_price, price_move, hold_days):
    widths = [
        ("A. 窄幅 ($10)", 10, "theta对冲高, 盈利上限$10"),
        ("B. 中幅 ($15)", 15, "平衡theta与收益"),
        ("C. 宽幅 ($20)", 20, "收益空间大, 成本更低"),
    ]

    rows = []
    for label, width, desc in widths:
        buy_opts = find_options(
            eod_df, "CALL", (gld_price - 5, gld_price + 5), (17, 45))
        if len(buy_opts) == 0:
            rows.append(_empty_spread_row(label, desc))
            continue

        oi_col = "option_open_interest" if "option_open_interest" in buy_opts.columns else None
        buy = buy_opts.nlargest(1, oi_col).iloc[0] if oi_col else buy_opts.iloc[0]
        sell = _find_spread_leg(eod_df, buy, "CALL", +width)
        if sell is None:
            rows.append(_empty_spread_row(label, desc))
            continue

        buy_strike = buy["option_strike_price"]
        sell_strike = sell["option_strike_price"]
        actual_width = sell_strike - buy_strike
        if actual_width <= 0:
            rows.append(_empty_spread_row(label, desc))
            continue

        net_debit = buy["mid"] - sell["mid"]
        if net_debit <= 0:
            rows.append(_empty_spread_row(label, desc))
            continue

        max_profit = actual_width - net_debit
        max_loss = net_debit  # 价差最大亏损 = 净成本

        buy_delta = abs(buy.get("option_delta", 0))
        sell_delta = abs(sell.get("option_delta", 0))
        pos_delta = buy_delta - sell_delta
        pos_theta = buy.get("option_theta", 0) - sell.get("option_theta", 0)

        pnl = min(pos_delta * price_move + pos_theta * hold_days, max_profit)
        roi = pnl / net_debit * 100 if net_debit > 0 else 0


        exp = pd.Timestamp(buy["strike_time"]).strftime("%m/%d")

        rows.append({
            "策略": label,
            "价差合约": f"GLD {exp} ${buy_strike:.0f}/${sell_strike:.0f}C",
            "净成本": f"${net_debit:.2f}",
            "5日盈利": f"${pnl:.2f} ({roi:+.0f}%)",
            "最大盈利": f"+${max_profit:.2f} (+{max_profit/net_debit*100:.0f}%)",
            "最大亏损": f"-${net_debit:.2f} (已锁定)",
        })
    return rows


def _empty_spread_row(label, desc):
    return {"策略": label, "价差合约": "—", "净成本": "—",
            "5日盈利": "—", "最大盈利": "—", "最大亏损": "—"}


# ══════════════════════════════════════════════════════════
# BUY CALL — 推荐逻辑
# ══════════════════════════════════════════════════════════

def _build_call_recommendation(eod_df, gld_price, price_move,
                                hold_days, exit_price):
    atm_opts = find_options(
        eod_df, "CALL", (gld_price - 3, gld_price + 3), (17, 45))
    if len(atm_opts) == 0:
        return None

    oi_col = "option_open_interest" if "option_open_interest" in atm_opts.columns else None
    atm = atm_opts.nlargest(1, oi_col).iloc[0] if oi_col else atm_opts.iloc[0]
    iv = atm.get("iv_decimal", atm.get("option_implied_volatility", 0) / 100
                  if atm.get("option_implied_volatility", 0) > 1 else 0) * 100
    theta = atm.get("option_theta", 0)
    mid = atm["mid"]
    theta_pct = abs(theta) / mid * 100 if mid > 0 else 0
    theta_5d = abs(theta) * hold_days


    sell_leg = _find_spread_leg(eod_df, atm, "CALL", 15)
    if sell_leg is not None:
        net_theta = theta - sell_leg.get("option_theta", 0)
        theta_saved_pct = (1 - abs(net_theta) / max(abs(theta), 0.01)) * 100
        net_debit = mid - sell_leg["mid"]
        max_loss_spread = net_debit
    else:
        theta_saved_pct = 0
        max_loss_spread = mid

    stop_premium = mid * (1 - SINGLE_STOP_PCT / 100)

    if iv >= 28:
        title = "推荐: Bull Call Spread"
        reason = (
            f"IV={iv:.0f}%偏高, theta ${abs(theta):.2f}/日, 5日损耗${theta_5d:.1f}。"
            f"价差对冲{theta_saved_pct:.0f}%theta。"
            f"\n单腿止损: 期权跌至${stop_premium:.2f} (-{SINGLE_STOP_PCT}%权利金)"
            f"\n价差: 最大亏损已锁定 (-${max_loss_spread:.2f}), 无需额外止损")
    elif iv <= 20:
        title = "推荐: Long Call (单腿)"
        reason = (
            f"IV={iv:.0f}%较低, theta仅${abs(theta):.2f}/日, 5日损失${theta_5d:.1f}可接受。"
            f"\n止损: 期权跌至${stop_premium:.2f} (-{SINGLE_STOP_PCT}%权利金)")
    else:
        title = "单腿 / 价差均可"
        reason = (
            f"IV={iv:.0f}%, theta ${abs(theta):.2f}/日。"
            f"\n单腿止损: 期权跌至${stop_premium:.2f} (-{SINGLE_STOP_PCT}%)"
            f"\n价差: 最大亏损已锁定 (-${max_loss_spread:.2f})")

    return f"**{title}**\n\n{reason}"


# ══════════════════════════════════════════════════════════
# SELL PUT → Bull Put Spread
# ══════════════════════════════════════════════════════════

def _build_put_spread_table(eod_df, gld_price, price_move, hold_days):
    spread_width = 5
    configs = [
        ("A. 稳健 (远OTM)", (gld_price * 0.88, gld_price * 0.93), (25, 45),
         "|Δ|≈0.10"),
        ("B. 中性 (OTM)", (gld_price * 0.93, gld_price * 0.97), (17, 35),
         "|Δ|≈0.25"),
        ("C. 激进 (近ATM)", (gld_price * 0.97, gld_price + 2), (14, 28),
         "|Δ|≈0.40"),
    ]

    rows = []
    for label, strike_range, dte_range, desc in configs:
        sell_opts = find_options(eod_df, "PUT", strike_range, dte_range)
        if len(sell_opts) == 0:
            rows.append(_empty_put_row(label, desc))
            continue

        oi_col = "option_open_interest" if "option_open_interest" in sell_opts.columns else None
        sell = sell_opts.nlargest(1, oi_col).iloc[0] if oi_col else sell_opts.iloc[0]
        buy = _find_spread_leg(eod_df, sell, "PUT", -spread_width)
        if buy is None:
            rows.append(_empty_put_row(label, desc))
            continue

        sell_strike = sell["option_strike_price"]
        buy_strike = buy["option_strike_price"]
        actual_width = sell_strike - buy_strike
        net_credit = sell["mid"] - buy["mid"]

        if net_credit <= 0 or actual_width <= 0:
            rows.append(_empty_put_row(label, desc))
            continue

        max_loss = actual_width - net_credit
        max_profit = net_credit


        sell_delta = sell.get("option_delta", 0)
        buy_delta = buy.get("option_delta", 0)
        pos_delta = -sell_delta + buy_delta
        pos_theta = -sell.get("option_theta", 0) + buy.get("option_theta", 0)

        pnl = pos_delta * price_move + pos_theta * hold_days
        roi_on_risk = pnl / max_loss * 100 if max_loss > 0 else 0

        exp = pd.Timestamp(sell["strike_time"]).strftime("%m/%d")

        rows.append({
            "策略": label,
            "价差合约": f"GLD {exp} ${sell_strike:.0f}/${buy_strike:.0f}P",
            "净权利金": f"+${net_credit:.2f}",
            "最大盈利": f"+${max_profit:.2f}",
            "最大亏损": f"-${max_loss:.2f} (已锁定)",
            "5日ROI": f"{roi_on_risk:+.0f}%",
        })
    return rows


def _empty_put_row(label, desc):
    return {"策略": label, "价差合约": "—", "净权利金": "—",
            "最大盈利": "—", "最大亏损": "—", "5日ROI": "—"}
